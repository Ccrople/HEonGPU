// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_batch16.cuh>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            struct Range
            {
                explicit Range(const char* name) { nvtxRangePushA(name); }
                Range(const Range&) = delete;
                Range& operator=(const Range&) = delete;
                ~Range() { nvtxRangePop(); }
            };

            bool is_power_of_two(int v)
            {
                return v > 0 && (v & (v - 1)) == 0;
            }
        } // namespace

        // -------------------------------------------------------------------
        // Construction and validation
        // -------------------------------------------------------------------

        Llama3Batch16Operator::Llama3Batch16Operator(
            Llama3BatchOperator& batch, const Batch16Shape& shape)
            : batch_(batch), shape_(shape),
              slot_count_(batch.layout().N / 2)
        {
            const BatchMatrixLayout& l = batch_.layout();

            // Every one of these is silent if it is not checked here. A shape
            // that does not fit the ring does not fail: it produces an
            // activation whose columns mean something other than what the
            // caller thinks, and the first sign of it is a wrong answer at the
            // end of a block.
            if (shape_.instances != l.batch)
            {
                throw std::invalid_argument(
                    "A batch-16 run carries exactly k/2 = " +
                    std::to_string(l.batch) +
                    " independent inputs at this ring, not " +
                    std::to_string(shape_.instances) +
                    "; the instance count is the ring's, not the caller's");
            }
            if (shape_.d_model <= 0 || shape_.hidden <= 0)
            {
                throw std::invalid_argument(
                    "A batch-16 shape needs a positive d_model and hidden");
            }
            if (shape_.heads < 1 || shape_.head_dim < 1)
            {
                throw std::invalid_argument(
                    "A batch-16 shape needs at least one head of at least one "
                    "channel");
            }
            if (shape_.head_dim % l.d != 0)
            {
                throw std::invalid_argument(
                    "head_dim must be a whole number of layout.d channel "
                    "blocks, because Algorithm 4's operands are square at d");
            }
            if (shape_.kv_heads < 0 ||
                (shape_.kv_heads > 0 && shape_.heads % shape_.kv_heads != 0))
            {
                throw std::invalid_argument(
                    "Grouped-query attention needs kv_heads to divide heads");
            }
            if (shape_.q_channels() > shape_.d_model &&
                shape_.q_channels() % l.d != 0)
            {
                throw std::invalid_argument(
                    "The query width must be a whole number of channel blocks");
            }
            if (!is_power_of_two(l.batch) || !is_power_of_two(l.d))
            {
                throw std::invalid_argument(
                    "The ring layout must be a power of two in both d and the "
                    "batch");
            }
            if (l.batch * l.d != slot_count_)
            {
                // The published invariant is a bijection onto every slot, and
                // that is what makes masking unnecessary everywhere below. If
                // it ever stopped holding, nothing here would notice.
                throw std::invalid_argument(
                    "The slot map b + (k/2)*u must cover every slot exactly "
                    "once: k/2 * d has to be N/2");
            }
        }

        int Llama3Batch16Operator::slot_of(int instance, int token) const
        {
            if (instance < 0 || instance >= instances() || token < 0 ||
                token >= tokens())
            {
                throw std::invalid_argument(
                    "slot_of is out of range for this ring");
            }
            return instance + instances() * token;
        }

        void Llama3Batch16Operator::validate(const BatchActivation& x,
                                             int channels,
                                             const char* name) const
        {
            if (x.rows != tokens())
            {
                throw std::invalid_argument(
                    std::string(name) + " takes an activation of d = " +
                    std::to_string(tokens()) + " rows, not " +
                    std::to_string(x.rows));
            }
            if (x.columns() != channels)
            {
                throw std::invalid_argument(
                    std::string(name) + " takes " + std::to_string(channels) +
                    " channels, not " + std::to_string(x.columns()));
            }
            for (int j = 1; j < x.columns(); ++j)
            {
                // The columns are added together inside the reduction, and
                // CKKS addition is only meaningful between equal scales, so a
                // drift here is a silently wrong sum rather than an error from
                // the library.
                if (x.column[j].depth() != x.column[0].depth() ||
                    x.column[j].scale() != x.column[0].scale())
                {
                    throw std::invalid_argument(
                        std::string(name) +
                        " needs every channel at one level and one scale");
                }
            }
        }

        // -------------------------------------------------------------------
        // The reshape: index arithmetic, and nothing else
        // -------------------------------------------------------------------

        Llama3Batch16Operator::ChannelRange
        Llama3Batch16Operator::query_head(int h) const
        {
            if (h < 0 || h >= shape_.heads)
            {
                throw std::invalid_argument("Query head is out of range");
            }
            return ChannelRange{h * shape_.head_dim, shape_.head_dim};
        }

        Llama3Batch16Operator::ChannelRange
        Llama3Batch16Operator::key_value_head(int h) const
        {
            const int kv =
                shape_.kv_heads > 0 ? shape_.kv_heads : shape_.heads;
            if (h < 0 || h >= kv)
            {
                throw std::invalid_argument("Key/value head is out of range");
            }
            return ChannelRange{h * shape_.head_dim, shape_.head_dim};
        }

        Llama3Batch16Operator::ChannelRange
        Llama3Batch16Operator::key_value_head_for_query(int h) const
        {
            if (h < 0 || h >= shape_.heads)
            {
                throw std::invalid_argument("Query head is out of range");
            }
            // The whole of grouped-query attention on this encoding. Several
            // query heads are handed the SAME run of ciphertexts; nothing is
            // copied and nothing is widened, so the key and value projections
            // stay narrow by the full heads/kv_heads factor.
            return key_value_head(h / shape_.group());
        }

        int Llama3Batch16Operator::channel_of(int head, int lane) const
        {
            if (lane < 0 || lane >= shape_.head_dim)
            {
                throw std::invalid_argument("Head lane is out of range");
            }
            return query_head(head).base + lane;
        }

        Llama3Batch16Operator::HeadLane
        Llama3Batch16Operator::head_of(int channel) const
        {
            if (channel < 0 || channel >= shape_.q_channels())
            {
                throw std::invalid_argument("Channel is out of range");
            }
            HeadLane out;
            out.head = channel / shape_.head_dim;
            out.lane = channel % shape_.head_dim;
            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>*>
        Llama3Batch16Operator::columns(BatchActivation& x, ChannelRange range)
        {
            if (range.base < 0 || range.count < 0 ||
                range.end() > x.columns())
            {
                throw std::invalid_argument(
                    "Channel range does not lie inside the activation");
            }
            std::vector<Ciphertext<Scheme::CKKS>*> out;
            out.reserve(static_cast<std::size_t>(range.count));
            for (int j = range.base; j < range.end(); ++j)
            {
                out.push_back(&x.column[static_cast<std::size_t>(j)]);
            }
            return out;
        }

        // -------------------------------------------------------------------
        // RMSNorm
        // -------------------------------------------------------------------

        double Llama3Batch16Operator::sum_pre_scale_factor(double sum_lo,
                                                           double sum_hi)
        {
            if (!(sum_hi > sum_lo) || !(sum_lo > 0.0))
            {
                throw std::invalid_argument(
                    "The summed square needs a positive range");
            }
            // The fit maps its argument with domain_scale = 2/(hi - lo), and
            // the sum is QUADRATIC in the input, so the factor that has to
            // ride on the upstream weight is the square root of it.
            return std::sqrt(2.0 / (sum_hi - sum_lo));
        }

        void Llama3Batch16Operator::fold_gain(std::vector<double>& weight,
                                              const std::vector<double>& gain,
                                              int in_channels,
                                              int out_channels)
        {
            if (in_channels <= 0 || out_channels <= 0)
            {
                throw std::invalid_argument(
                    "Folding a gain needs a positive weight shape");
            }
            if (weight.size() != static_cast<std::size_t>(in_channels) *
                                     static_cast<std::size_t>(out_channels))
            {
                throw std::invalid_argument(
                    "The weight must be row major in_channels x out_channels");
            }
            if (gain.size() != static_cast<std::size_t>(in_channels))
            {
                throw std::invalid_argument(
                    "A learned RMSNorm gain has one entry per INPUT channel of "
                    "the projection that reads the normalised stream");
            }
            for (int i = 0; i < in_channels; ++i)
            {
                const double g = gain[static_cast<std::size_t>(i)];
                double* row = weight.data() +
                              static_cast<std::size_t>(i) * out_channels;
                for (int o = 0; o < out_channels; ++o)
                {
                    row[o] *= g;
                }
            }
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Batch16Operator::pre_scaled_norm(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const RMSNormConfig& config, Relinkey<Scheme::CKKS>& relin_key)
        {
            // The same circuit Llama3Operator::rms_norm runs, written out so
            // that the fit can be told its argument is ALREADY mapped. That
            // one flag is the whole difference and it is worth a level: the
            // slot core reaches it only through fold_affine_into_mask, which
            // is gated on a blocked reduction this encoding does not have,
            // because a channel here is a whole ciphertext and there is no
            // mask between the reduction and the fit to carry the map.
            //
            // What the caller must have done, and what sum_pre_scale_factor()
            // returns: multiplied the plaintext weight that produced @p slots
            // by c = sqrt(domain_scale(sum_lo, sum_hi)), and set output_scale
            // to 1/c. The sum of squares is QUADRATIC in the input, so c*c is
            // the domain scale exactly; the 1/c takes the pre-scaling back out
            // of the numerator and rides in the fit's coefficients, where a
            // host-side number costs nothing.
            const int channels = static_cast<int>(slots.size());

            Range _r("b16.rms_norm.pre_scaled");

            Ciphertext<Scheme::CKKS> total = slots.front();
            {
                Range _r_sum("b16.rms_norm.sum_of_squares");
                arith().square(total, relin_key);
                for (int j = 1; j < channels; ++j)
                {
                    Ciphertext<Scheme::CKKS> term =
                        slots[static_cast<std::size_t>(j)];
                    arith().square(term, relin_key);
                    // Every term left the same sequence of operations behind
                    // it, so they meet at one level and one scale. validate()
                    // is what makes that true of the inputs.
                    arith().add_inplace(total, term);
                }
            }

            // Named apart from the pre-scale factor c above, which is a
            // different number entirely and lives on the host.
            const double channel_count = static_cast<double>(channels);
            const double eps = config.eps;
            const double gain = config.output_scale;

            Ciphertext<Scheme::CKKS> factor;
            {
                Range _r_fit("b16.rms_norm.inverse_sqrt");
                factor = arith().evaluate_function(
                    total,
                    [channel_count, eps, gain](double s)
                    { return gain / std::sqrt(s / channel_count + eps); },
                    config.sum_lo, config.sum_hi, config.degree, relin_key,
                    /*pre_scaled=*/true);
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(slots.size());
            {
                Range _r_apply("b16.rms_norm.rescale_channels");
                for (int j = 0; j < channels; ++j)
                {
                    out.push_back(arith().multiply_and_rescale(
                        slots[static_cast<std::size_t>(j)], factor,
                        relin_key));
                }
            }
            return out;
        }

        BatchActivation
        Llama3Batch16Operator::rms_norm(BatchActivation& x,
                                        const std::vector<double>& gain,
                                        const RMSNormConfig& config,
                                        Galoiskey<Scheme::CKKS>& galois_key,
                                        Relinkey<Scheme::CKKS>& relin_key)
        {
            const int channels = x.columns();
            validate(x, channels, "rms_norm");
            if (!gain.empty() &&
                static_cast<int>(gain.size()) != channels)
            {
                throw std::invalid_argument(
                    "RMSNorm takes one learned scale per channel, and a "
                    "channel is a whole ciphertext here -- or none at all, "
                    "which is the fast path: fold_gain() puts it on the "
                    "projection that reads this stream, for free");
            }
            if (config.sum_pre_scaled && config.newton_iterations > 0)
            {
                throw std::invalid_argument(
                    "A Newton step refines against the unmapped summed square, "
                    "and a pre-scaled sum arrives mapped");
            }
            if (config.sum_pre_scaled && !config.fold_mean_into_fit)
            {
                throw std::invalid_argument(
                    "A pre-scaled sum has the fit's domain map already on it, "
                    "and forming the mean would multiply that by 1/channels: "
                    "the fit has to carry the division instead");
            }
            if (!(config.sum_hi > config.sum_lo) || !(config.sum_lo > 0.0))
            {
                throw std::invalid_argument(
                    "RMSNorm needs a positive calibrated range for the summed "
                    "square");
            }

            Range _r("b16.rms_norm");
            note_depth("rms_norm.in", x);

            std::vector<Ciphertext<Scheme::CKKS>> slots;
            {
                Range _r_in("b16.rms_norm.to_slots");
                slots = batch_.to_slots(x, galois_key);
            }

            Llama3Operator::RMSNormConfig slot_config;
            // THE LINE THIS MODULE EXISTS FOR. A channel is a whole
            // ciphertext, so the channel axis is not inside a ciphertext at
            // all: count = 1 makes Llama3Operator's reduction loop
            // (`for (t = 1; t < count; t <<= 1)`) not execute, and the sum
            // over channels is the slot-wise addition of the squares that
            // precedes it. No rotation, no mask, no Galois key, no level.
            slot_config.stride = arith().slot_count();
            slot_config.count = 1;
            slot_config.blocked_span = 0;
            slot_config.channels = channels;
            slot_config.token_blocks = 1;
            slot_config.eps = config.eps;
            slot_config.sum_lo = config.sum_lo;
            slot_config.sum_hi = config.sum_hi;
            slot_config.degree = config.degree;
            slot_config.newton_iterations = config.newton_iterations;
            slot_config.fold_mean_into_fit = config.fold_mean_into_fit;
            // There is no mask on this path to carry the fit's domain map, so
            // the slot core's fold_affine_into_mask is inapplicable and is
            // left alone. The equivalent saving here rides on the upstream
            // plaintext instead; see RMSNormConfig::sum_pre_scaled.
            slot_config.fold_affine_into_mask = false;
            slot_config.output_scale = config.output_scale;
            slot_config.refresh_sum = false;

            std::vector<Ciphertext<Scheme::CKKS>> normalised;
            if (config.sum_pre_scaled)
            {
                normalised = pre_scaled_norm(slots, config, relin_key);
            }
            else
            {
                // The proven path. Llama3Operator::rms_norm is what the
                // existing tests assert against, and nothing above changes it
                // -- only the config it is handed.
                std::vector<Plaintext<Scheme::CKKS>> no_weights;
                normalised = arith().rms_norm(slots, no_weights, slot_config,
                                              galois_key, relin_key);
            }
            slots.clear();
            note_depth("rms_norm.normalised", normalised);

            if (!gain.empty())
            {
                // The slow path, kept so that the fold can be checked against
                // it rather than believed.
                //
                // multiply_constant encodes at the prime the following rescale
                // removes and at the depth the ciphertext has NOW, which is
                // also why it is used here rather than the encoded-plaintext
                // route the other paths take: they build the gain plaintext at
                // the depth of the INPUT and consume it against the output
                // fifteen levels lower, so multiply_plaintext re-drops a fresh
                // copy per channel and the trailing rescale divides by a
                // different prime than the plaintext was encoded at. The value
                // survives -- the tracked scale is exact -- but the activation
                // stops sitting at default_scale, and on this path that is
                // 4096 avoidable re-drops per norm.
                Range _r_g("b16.rms_norm.gain");
                for (int j = 0; j < channels; ++j)
                {
                    arith().multiply_constant(
                        normalised[static_cast<std::size_t>(j)],
                        gain[static_cast<std::size_t>(j)]);
                }
            }

            Range _r_out("b16.rms_norm.from_slots");
            BatchActivation out =
                batch_.from_slots(normalised, x.rows, galois_key);
            note_depth("rms_norm.out", out);
            return out;
        }

        // -------------------------------------------------------------------
        // SwiGLU
        // -------------------------------------------------------------------

        std::vector<double>
        Llama3Batch16Operator::gate_slice(const std::vector<double>& weight,
                                          int in_channels, int hidden,
                                          int base, int count,
                                          double scale) const
        {
            // The gate and up weights are in_channels x hidden, so a chunk of
            // the hidden axis is a set of COLUMNS and has to be gathered; the
            // down weight is hidden x in_channels, where the same chunk is a
            // contiguous span of rows and no gather is needed.
            std::vector<double> out(static_cast<std::size_t>(in_channels) *
                                    static_cast<std::size_t>(count));
            for (int i = 0; i < in_channels; ++i)
            {
                const std::size_t src =
                    static_cast<std::size_t>(i) * hidden + base;
                const std::size_t dst =
                    static_cast<std::size_t>(i) * count;
                for (int j = 0; j < count; ++j)
                {
                    out[dst + j] = weight[src + j] * scale;
                }
            }
            return out;
        }

        BatchActivation Llama3Batch16Operator::feed_forward(
            BatchActivation& x, const FeedForwardWeights& weights,
            const FeedForwardConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            const int in_channels = shape_.d_model;
            const int hidden = shape_.hidden;
            validate(x, in_channels, "feed_forward");

            const std::size_t want = static_cast<std::size_t>(in_channels) *
                                     static_cast<std::size_t>(hidden);
            if (weights.gate.size() != want || weights.up.size() != want ||
                weights.down.size() != want)
            {
                throw std::invalid_argument(
                    "The SwiGLU weights must be d_model by hidden, and down "
                    "its transpose shape");
            }
            if (!(config.silu_bound > 0.0))
            {
                throw std::invalid_argument("The SiLU bound must be positive");
            }

            Range _r("b16.feed_forward");
            note_depth("feed_forward.in", x);

            int block =
                config.hidden_block > 0 ? config.hidden_block : hidden;
            block = std::min(block, hidden);

            // The domain map of the SiLU fit, carried on the gate weight
            // instead of on a homomorphic product. The gate projection feeds
            // the SiLU and nothing else, so the factor needs no undoing.
            const double gate_scale =
                config.fold_silu_domain_into_gate ? 1.0 / config.silu_bound
                                                  : 1.0;

            BatchActivation out;
            out.rows = x.rows;

            for (int base = 0; base < hidden; base += block)
            {
                const int cols = std::min(block, hidden - base);

                std::vector<double> gate_w = gate_slice(
                    weights.gate, in_channels, hidden, base, cols, gate_scale);
                std::vector<double> up_w = gate_slice(
                    weights.up, in_channels, hidden, base, cols, 1.0);

                BatchActivation gate = batch_.project(
                    x, gate_w, in_channels, cols, "b16.ffn.gate");
                BatchActivation up =
                    batch_.project(x, up_w, in_channels, cols, "b16.ffn.up");

                // Both branches cross, because the gate product is a Hadamard
                // product and a matrix encryption has none: multiplying two
                // columns convolves them. This is the one place this encoding
                // pays more than the rectangular one, and it is still the
                // cheaper side of the trade.
                std::vector<Ciphertext<Scheme::CKKS>> gate_slots;
                std::vector<Ciphertext<Scheme::CKKS>> up_slots;
                {
                    Range _r_in("b16.ffn.to_slots");
                    gate_slots = batch_.to_slots(gate, galois_key);
                    up_slots = batch_.to_slots(up, galois_key);
                }
                gate.column.clear();
                up.column.clear();
                note_depth("feed_forward.activation", gate_slots);

                std::vector<Ciphertext<Scheme::CKKS>> hidden_slots;
                hidden_slots.reserve(gate_slots.size());
                {
                    Range _r_silu("b16.ffn.silu");
                    for (std::size_t j = 0; j < gate_slots.size(); ++j)
                    {
                        Ciphertext<Scheme::CKKS> activated = arith().silu(
                            gate_slots[j], config.silu_bound,
                            config.silu_degree, relin_key,
                            config.fold_silu_domain_into_gate);
                        hidden_slots.push_back(arith().multiply_and_rescale(
                            activated, up_slots[j], relin_key));
                    }
                }
                gate_slots.clear();
                up_slots.clear();
                note_depth("feed_forward.hidden", hidden_slots);

                BatchActivation h =
                    batch_.from_slots(hidden_slots, x.rows, galois_key);
                hidden_slots.clear();

                const std::vector<double> down_w(
                    weights.down.begin() +
                        static_cast<std::size_t>(base) * in_channels,
                    weights.down.begin() +
                        static_cast<std::size_t>(base + cols) * in_channels);
                BatchActivation part = batch_.project(
                    h, down_w, cols, in_channels, "b16.ffn.down");

                if (out.column.empty())
                {
                    out.column = std::move(part.column);
                }
                else
                {
                    // Every chunk's partial product leaves the same sequence
                    // of operations behind it, so they meet at one level and
                    // one scale and the sum is a plain addition.
                    Range _r_acc("b16.ffn.accumulate");
                    for (std::size_t j = 0; j < out.column.size(); ++j)
                    {
                        arith().add_inplace(out.column[j], part.column[j]);
                    }
                }
            }

            note_depth("feed_forward.out", out);
            return out;
        }

        // -------------------------------------------------------------------
        // The bridge
        // -------------------------------------------------------------------

        void Llama3Batch16Operator::use_fast_bridge(int baby_steps,
                                                    std::size_t cache_sets)
        {
            batch_.set_bridge_baby_steps(baby_steps);
            batch_.set_hoisted_crossings(true);
            batch_.set_bridge_plain_capacity(cache_sets);
        }

        long long Llama3Batch16Operator::bridged_columns_per_block() const
        {
            // Two norms, each down and back over the residual width; and the
            // SwiGLU's gate, up and hidden over the hidden width. The
            // attention sublayer's own two crossings belong to another module
            // and are deliberately not counted here.
            return 2LL * 2LL * shape_.d_model + 3LL * shape_.hidden;
        }

        void Llama3Batch16Operator::note_depth(
            const char* name,
            const std::vector<Ciphertext<Scheme::CKKS>>& ct) const
        {
            if (depth_trace && !ct.empty())
            {
                depth_trace(name, ct.front().depth());
            }
        }

        void Llama3Batch16Operator::note_depth(const char* name,
                                               const BatchActivation& x) const
        {
            if (depth_trace && !x.column.empty())
            {
                depth_trace(name, x.column.front().depth());
            }
        }

    } // namespace llama
} // namespace heongpu
