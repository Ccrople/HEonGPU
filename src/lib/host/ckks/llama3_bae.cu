// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_bae.cuh>

#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            struct BaeBlockRange
            {
                explicit BaeBlockRange(const char* name)
                {
                    nvtxRangePushA(name);
                }
                ~BaeBlockRange() { nvtxRangePop(); }
                BaeBlockRange(const BaeBlockRange&) = delete;
                BaeBlockRange& operator=(const BaeBlockRange&) = delete;
            };
        } // namespace

        Llama3BaeOperator::Llama3BaeOperator(HEContext<Scheme::CKKS> context,
                                             HEEncoder<Scheme::CKKS>& encoder,
                                             int tokens, double scale)
            : context_(context), encoder_(encoder),
              arith_(context, encoder, scale),
              // cols = N is k = 1: ModDecomp and ModPack are both the
              // identity, so the product never looks inside a ciphertext and
              // therefore never commits to a reading of one. That is the whole
              // premise of this module and it is set here, once.
              pcmm_(context, static_cast<int>(context->get_poly_modulus_degree())),
              n_(static_cast<int>(context->get_poly_modulus_degree())),
              slot_count_(encoder.slot_count()), tokens_(tokens),
              default_scale_(scale)
        {
            if (tokens <= 0 || tokens > slot_count_)
            {
                throw std::invalid_argument(
                    "an activation carries between one and slot_count tokens: "
                    "a longer sequence is more ciphertexts per channel, not a "
                    "wider one");
            }
            // Both GEMMs then run on the stored limbs and the NTT round trip
            // disappears. Legal only at k == 1, which the operator itself
            // rechecks.
            pcmm_.set_transform_free(true);
        }

        void Llama3BaeOperator::require_uniform(const BaeActivation& x,
                                                const char* name) const
        {
            if (x.empty())
            {
                throw std::invalid_argument(std::string(name) +
                                            " received an empty activation");
            }
            if (static_cast<int>(x.column.size()) != x.channels)
            {
                throw std::invalid_argument(
                    std::string(name) +
                    ": one ciphertext per channel is the encoding, so the "
                    "column count and the channel count are the same number");
            }
            for (std::size_t i = 1; i < x.column.size(); ++i)
            {
                // The product adds these together weighted by U, so a mismatch
                // is a silently weighted sum rather than an error.
                if (x.column[i].depth() != x.column[0].depth() ||
                    x.column[i].scale() != x.column[0].scale())
                {
                    throw std::invalid_argument(
                        std::string(name) +
                        " needs every channel at one level and one scale");
                }
            }
        }

        // -------------------------------------------------------------------
        // Staging
        // -------------------------------------------------------------------

        BaeActivation
        Llama3BaeOperator::encrypt(const std::vector<double>& x, int tokens,
                                   int channels,
                                   HEEncryptor<Scheme::CKKS>& encryptor)
        {
            if (tokens != tokens_)
            {
                throw std::invalid_argument(
                    "this operator was built for a different token count");
            }
            if (x.size() != static_cast<std::size_t>(tokens) *
                                static_cast<std::size_t>(channels))
            {
                throw std::invalid_argument(
                    "the host matrix must be tokens x channels, row major");
            }

            BaeActivation out;
            out.tokens = tokens;
            out.channels = channels;
            out.column.reserve(static_cast<std::size_t>(channels));
            for (int c = 0; c < channels; ++c)
            {
                std::vector<double> slots(
                    static_cast<std::size_t>(slot_count_), 0.0);
                for (int t = 0; t < tokens; ++t)
                {
                    slots[static_cast<std::size_t>(t)] =
                        x[static_cast<std::size_t>(t) * channels + c];
                }
                Plaintext<Scheme::CKKS> plain(context_);
                encoder_.encode(plain, slots, default_scale_);
                Ciphertext<Scheme::CKKS> ct(context_);
                encryptor.encrypt(ct, plain);
                out.column.push_back(std::move(ct));
            }
            return out;
        }

        std::vector<double>
        Llama3BaeOperator::decrypt(BaeActivation& x,
                                   HEDecryptor<Scheme::CKKS>& decryptor)
        {
            std::vector<double> out(static_cast<std::size_t>(x.tokens) *
                                        static_cast<std::size_t>(x.channels),
                                    0.0);
            for (int c = 0; c < x.channels; ++c)
            {
                Plaintext<Scheme::CKKS> plain(context_);
                decryptor.decrypt(plain, x.column[static_cast<std::size_t>(c)]);
                std::vector<double> slots;
                encoder_.decode(slots, plain);
                for (int t = 0; t < x.tokens; ++t)
                {
                    out[static_cast<std::size_t>(t) * x.channels + c] =
                        slots[static_cast<std::size_t>(t)];
                }
            }
            return out;
        }

        // -------------------------------------------------------------------
        // Projection
        // -------------------------------------------------------------------

        BaeActivation
        Llama3BaeOperator::project(BaeActivation& x,
                                   const std::vector<double>& weight,
                                   int in_channels, int out_channels,
                                   const char* name)
        {
            require_uniform(x, name);
            if (x.channels != in_channels)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " weight expects a different number of input channels than "
                    "the activation carries");
            }

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "bae.project.%s",
                          name);
            BaeBlockRange _r(range_name);

            std::vector<Ciphertext<Scheme::CKKS>*> in;
            in.reserve(x.column.size());
            for (auto& c : x.column)
            {
                in.push_back(&c);
            }

            BaeActivation out;
            out.tokens = x.tokens;
            out.channels = out_channels;
            // No Galois key, no relin key, no boot key. The signature is the
            // claim: nothing is passed because nothing is needed.
            pcmm_.project(out.column, in, weight, in_channels, out_channels,
                          arith_);
            return out;
        }

        void Llama3BaeOperator::fold_channel_gain(
            std::vector<double>& weight, const std::vector<double>& gain,
            int in_channels, int out_channels)
        {
            if (gain.empty())
            {
                return;
            }
            if (static_cast<int>(gain.size()) != in_channels)
            {
                throw std::invalid_argument(
                    "the gain has one entry per input channel, which is one "
                    "entry per ROW of the weight");
            }
            if (weight.size() != static_cast<std::size_t>(in_channels) *
                                     static_cast<std::size_t>(out_channels))
            {
                throw std::invalid_argument(
                    "the weight must be in_channels x out_channels, row major");
            }
            for (int c = 0; c < in_channels; ++c)
            {
                const double g = gain[static_cast<std::size_t>(c)];
                for (int o = 0; o < out_channels; ++o)
                {
                    weight[static_cast<std::size_t>(c) * out_channels + o] *= g;
                }
            }
        }

        // -------------------------------------------------------------------
        // RMSNorm
        // -------------------------------------------------------------------

        BaeActivation
        Llama3BaeOperator::rms_norm(BaeActivation& x,
                                    const std::vector<double>& gain,
                                    const BaeRMSNormConfig& config,
                                    Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "rms_norm");
            if (!gain.empty() &&
                static_cast<int>(gain.size()) != x.channels)
            {
                throw std::invalid_argument(
                    "RMSNorm takes one learned scale per channel");
            }
            if (!(config.sum_hi > config.sum_lo) || !(config.sum_lo > 0.0))
            {
                throw std::invalid_argument(
                    "RMSNorm needs a positive range for the summed square");
            }
            if (config.output_scale != 1.0 && config.newton_iterations > 0)
            {
                throw std::invalid_argument(
                    "RMSNorm's output_scale rides on the fitted 1/sqrt, and a "
                    "Newton step refines toward 1/sqrt itself: the two cannot "
                    "both be had");
            }

            BaeBlockRange _r("bae.rms_norm");

            const int channels = x.channels;
            const double channels_d = static_cast<double>(channels);
            const bool fold_mean =
                config.fold_mean_into_fit && config.newton_iterations <= 0;

            const double lo = config.sum_lo / channels_d + config.eps;
            const double hi = config.sum_hi / channels_d + config.eps;

            // THE POINT OF THE WHOLE MODULE, IN ONE LOOP.
            //
            // The channel axis is the ciphertext axis, so summing the squares
            // over the channels is summing ciphertexts. There is no reduction
            // INSIDE a ciphertext to perform, so there is no rotation, no
            // mask, no Galois key and no level beyond the squaring. The rect
            // path puts N/(2d) channels on the fast slot axis and pays a
            // masked blocked reduction of span N/(2d) for exactly this line.
            Ciphertext<Scheme::CKKS> total;
            {
                BaeBlockRange _rr("bae.rms_norm.sum_of_squares");
                total = x.column[0];
                arith_.square(total, relin_key);
                for (int c = 1; c < channels; ++c)
                {
                    Ciphertext<Scheme::CKKS> term =
                        x.column[static_cast<std::size_t>(c)];
                    arith_.square(term, relin_key);
                    arith_.add_inplace(total, term);
                }

                if (!fold_mean)
                {
                    arith_.multiply_constant(total, 1.0 / channels_d);
                    arith_.add_constant(total, config.eps);
                }
            }

            Ciphertext<Scheme::CKKS> scale_factor;
            {
                BaeBlockRange _rr("bae.rms_norm.inverse_sqrt");
                const double out_gain = config.output_scale;
                if (fold_mean)
                {
                    // The same value of the same ciphertext, fitted over the
                    // summed square instead of over the mean: one level
                    // cheaper, and the level is the whole reason.
                    const double eps = config.eps;
                    scale_factor = arith_.evaluate_function(
                        total,
                        [channels_d, eps, out_gain](double s) {
                            return out_gain / std::sqrt(s / channels_d + eps);
                        },
                        config.sum_lo, config.sum_hi, config.degree, relin_key,
                        false);
                }
                else if (out_gain != 1.0)
                {
                    scale_factor = arith_.evaluate_function(
                        total,
                        [out_gain](double v)
                        { return out_gain / std::sqrt(v); },
                        lo, hi, config.degree, relin_key, false);
                }
                else
                {
                    scale_factor = arith_.inverse_sqrt(
                        total, lo, hi, config.degree,
                        config.newton_iterations, relin_key);
                }
            }

            BaeActivation out;
            out.tokens = x.tokens;
            out.channels = channels;
            out.column.reserve(static_cast<std::size_t>(channels));
            {
                BaeBlockRange _rr("bae.rms_norm.rescale_channels");
                for (int c = 0; c < channels; ++c)
                {
                    Ciphertext<Scheme::CKKS> normalised =
                        arith_.multiply_and_rescale(
                            x.column[static_cast<std::size_t>(c)],
                            scale_factor, relin_key);

                    // A channel is a whole ciphertext, so its learned scale is
                    // a CONSTANT rather than a slot vector. It still costs a
                    // level applied here; fold_channel_gain() puts the same
                    // numbers on the next weight for nothing, and the block
                    // driver does that instead of calling this.
                    if (!gain.empty())
                    {
                        arith_.multiply_constant(
                            normalised, gain[static_cast<std::size_t>(c)]);
                    }
                    out.column.push_back(std::move(normalised));
                }
            }
            return out;
        }

        // -------------------------------------------------------------------
        // Feed forward
        // -------------------------------------------------------------------

        BaeActivation Llama3BaeOperator::feed_forward(
            BaeActivation& x, const BaeFeedForwardWeights& weights,
            const BaeFeedForwardConfig& config,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "feed_forward");
            if (config.in_channels <= 0 || config.hidden <= 0)
            {
                throw std::invalid_argument(
                    "the feed-forward shape needs both widths");
            }
            if (x.channels != config.in_channels)
            {
                throw std::invalid_argument(
                    "the feed-forward input width does not match the "
                    "activation");
            }

            BaeBlockRange _r("bae.feed_forward");

            BaeActivation gate = project(x, weights.gate, config.in_channels,
                                         config.hidden, "ffn.gate");
            BaeActivation up = project(x, weights.up, config.in_channels,
                                       config.hidden, "ffn.up");

            BaeActivation hidden;
            hidden.tokens = x.tokens;
            hidden.channels = config.hidden;
            hidden.column.reserve(static_cast<std::size_t>(config.hidden));
            {
                BaeBlockRange _rr("bae.feed_forward.swiglu");
                for (int c = 0; c < config.hidden; ++c)
                {
                    Ciphertext<Scheme::CKKS> activated = arith_.silu(
                        gate.column[static_cast<std::size_t>(c)],
                        config.silu_bound, config.silu_degree, relin_key,
                        /*pre_scaled=*/false, config.silu_gain);
                    hidden.column.push_back(arith_.multiply_and_rescale(
                        activated, up.column[static_cast<std::size_t>(c)],
                        relin_key));
                }
            }

            if (weights.down.empty())
            {
                return hidden;
            }
            return project(hidden, weights.down, config.hidden,
                           config.in_channels, "ffn.down");
        }

        BaeActivation
        Llama3BaeOperator::residual_add(BaeActivation& x,
                                        BaeActivation& sublayer)
        {
            if (x.channels != sublayer.channels ||
                x.tokens != sublayer.tokens)
            {
                throw std::invalid_argument(
                    "a residual adds an activation to one of its own shape");
            }
            BaeBlockRange _r("bae.residual");
            BaeActivation out;
            out.tokens = x.tokens;
            out.channels = x.channels;
            out.column = arith_.residual_add(x.column, sublayer.column);
            return out;
        }

        // -------------------------------------------------------------------
        // The feed-forward half of a block
        // -------------------------------------------------------------------

        BaeActivation Llama3BaeOperator::feed_forward_block(
            BaeActivation& x, const BaeFeedForwardBlockWeights& weights,
            const BaeRMSNormConfig& norm_config,
            const BaeFeedForwardConfig& ffn_config,
            Relinkey<Scheme::CKKS>& relin_key,
            std::vector<std::pair<std::string, int>>* depth_trace)
        {
            require_uniform(x, "feed_forward_block");
            BaeBlockRange _r("bae.feed_forward_block");

            const auto note = [&](const char* name, const BaeActivation& a)
            {
                if (depth_trace != nullptr)
                {
                    depth_trace->emplace_back(name, a.column.front().depth());
                }
            };
            note("block.entry", x);

            // The learned gain is applied to the WEIGHTS, not to the stream:
            // exact, and it costs nothing. So the norm is called with no gain
            // and the folding happens below. See fold_channel_gain.
            const std::vector<double> no_gain;
            BaeActivation normalised =
                rms_norm(x, no_gain, norm_config, relin_key);
            note("block.after_feed_forward_norm", normalised);

            BaeFeedForwardWeights w = weights.feed_forward;
            fold_channel_gain(w.gate, weights.norm_gain,
                              ffn_config.in_channels, ffn_config.hidden);
            fold_channel_gain(w.up, weights.norm_gain, ffn_config.in_channels,
                              ffn_config.hidden);

            BaeActivation sublayer =
                feed_forward(normalised, w, ffn_config, relin_key);
            note("block.after_feed_forward", sublayer);

            BaeActivation out = residual_add(x, sublayer);
            note("block.out", out);
            return out;
        }

        // -------------------------------------------------------------------
        // Cost model
        // -------------------------------------------------------------------

        Llama3BaeOperator::Counts
        Llama3BaeOperator::feed_forward_block_counts(int d_model,
                                                     int hidden) const
        {
            Counts c;

            // A projection is d1 * d2 * (N + cols) MACs per limb, and at k = 1
            // cols is N, so both GEMMs are the same width and the total is
            // 2 * d1 * d2 * N.
            const double n = static_cast<double>(n_);
            const auto gemm = [n](double d1, double d2)
            { return 2.0 * d1 * d2 * n; };
            c.gemm_macs = gemm(hidden, d_model)      // gate
                          + gemm(hidden, d_model)    // up
                          + gemm(d_model, hidden);   // down

            // Ciphertext-ciphertext products this module ISSUES ITSELF, one
            // relinearisation each: the norm's squarings and the product that
            // applies its fit, and the SwiGLU's gate product. The
            // Paterson-Stockmeyer interior of the two fits is Llama3Operator's
            // and is deliberately not counted here, because it is the same
            // circuit on either path and counting it would blur the one
            // difference this table exists to show.
            c.relinearisations = 2LL * d_model + static_cast<long long>(hidden);

            // The three numbers the module is for. They are zero by
            // construction and not by tuning: no entry point above takes a
            // Galois key, so there is nothing to set them from.
            c.rotations = 0;
            c.galois_keys = 0;
            c.crossings = 0;
            return c;
        }

    } // namespace llama
} // namespace heongpu
