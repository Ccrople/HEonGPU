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

        void Llama3Batch16Operator::check_norm_config(
            const RMSNormConfig& config) const
        {
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
        }

        void Llama3Batch16Operator::fill_slot_config(
            Llama3Operator::RMSNormConfig& slot_config,
            const RMSNormConfig& config, int channels)
        {
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
            check_norm_config(config);

            Range _r("b16.rms_norm");
            note_depth("rms_norm.in", x);

            std::vector<Ciphertext<Scheme::CKKS>> slots;
            {
                Range _r_in("b16.rms_norm.to_slots");
                slots = batch_.to_slots(x, galois_key);
            }

            Llama3Operator::RMSNormConfig slot_config;
            fill_slot_config(slot_config, config, channels);

            std::vector<Ciphertext<Scheme::CKKS>> normalised =
                norm_core(slots, gain, config, slot_config, galois_key,
                          relin_key);
            // Released before the return crossing, which
            // Llama3BatchOperator::rms_norm does not do: 4096 ciphertexts and
            // 17.5 GiB at the 8B shape, and the difference between its frame
            // peaking at 87.5 GiB and at 70.
            slots.clear();

            Range _r_out("b16.rms_norm.from_slots");
            BatchActivation out =
                batch_.from_slots(normalised, x.rows, galois_key);
            note_depth("rms_norm.out", out);
            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Batch16Operator::rms_norm_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const std::vector<double>& gain, const RMSNormConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (slots.empty())
            {
                throw std::invalid_argument("RMSNorm needs a channel");
            }
            const int channels = static_cast<int>(slots.size());
            if (!gain.empty() && static_cast<int>(gain.size()) != channels)
            {
                throw std::invalid_argument(
                    "RMSNorm takes one learned scale per channel");
            }
            check_norm_config(config);
            for (std::size_t j = 1; j < slots.size(); ++j)
            {
                if (slots[j].depth() != slots[0].depth() ||
                    slots[j].scale() != slots[0].scale())
                {
                    throw std::invalid_argument(
                        "rms_norm_slots needs every channel at one level and "
                        "one scale: the squares are added together");
                }
            }

            Range _r("b16.rms_norm_slots");
            note_depth("rms_norm_slots.in", slots);

            Llama3Operator::RMSNormConfig slot_config;
            fill_slot_config(slot_config, config, channels);
            std::vector<Ciphertext<Scheme::CKKS>> out = norm_core(
                slots, gain, config, slot_config, galois_key, relin_key);
            note_depth("rms_norm_slots.out", out);
            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Batch16Operator::norm_core(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const std::vector<double>& gain, const RMSNormConfig& config,
            const Llama3Operator::RMSNormConfig& slot_config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            const int channels = static_cast<int>(slots.size());

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

            return normalised;
        }

        // -------------------------------------------------------------------
        // Rotary position embedding
        // -------------------------------------------------------------------

        void Llama3Batch16Operator::rope_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const RopeConfig& config)
        {
            // One implementation, on the operator attention() can reach. This
            // is the shape-aware front door: head_dim comes from the model
            // shape rather than from the caller, so a channel count that is
            // not a whole number of heads is refused with the shape named.
            batch_.rope_slots(slots, shape_.head_dim, config.theta,
                              config.position_offset);
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
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int in_channels = shape_.d_model;
            const int hidden = shape_.hidden;
            validate(x, in_channels, "feed_forward");
            if (config.refresh_hidden && boot_key == nullptr)
            {
                // Falling back to no refresh would change the level schedule
                // the caller sized its chain for, silently, and the failure
                // would surface as a throw several layers away.
                throw std::invalid_argument(
                    "refresh_hidden needs the boot Galois key");
            }

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

            if (config.slot_resident)
            {
                // Cross ONCE around the whole sublayer instead of three times
                // over the hidden width. 2 * d_model = 8,192 columns instead
                // of 3 * hidden = 43,008 at the 8B shape, and one level fewer
                // because two crossings replace three. Legal because a
                // batch-shared real weight is the constant polynomial of R_k,
                // so Algorithm 1 is a scalar multiply-accumulate across
                // ciphertexts and does not read the encoding.
                std::vector<Ciphertext<Scheme::CKKS>> slots;
                {
                    Range _r_in("b16.ffn.to_slots");
                    slots = batch_.to_slots(x, galois_key);
                }
                std::vector<Ciphertext<Scheme::CKKS>> hidden_out =
                    feed_forward_slots(slots, weights, config, relin_key,
                                       boot_key);
                slots.clear();

                Range _r_out("b16.ffn.from_slots");
                BatchActivation out =
                    batch_.from_slots(hidden_out, x.rows, galois_key);
                note_depth("feed_forward.out", out);
                return out;
            }

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

                if (config.refresh_hidden)
                {
                    // Taken here rather than at either end of the sublayer:
                    // this is the point at which the SwiGLU's stretch actually
                    // runs out, and it is already in slot form, so the refresh
                    // costs no crossing of its own.
                    bootstrap(hidden_slots, "feed_forward.hidden", *boot_key,
                              relin_key);
                    note_depth("feed_forward.hidden_refreshed", hidden_slots);
                }

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

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Batch16Operator::feed_forward_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const FeedForwardWeights& weights,
            const FeedForwardConfig& config,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int in_channels = shape_.d_model;
            const int hidden = shape_.hidden;
            if (static_cast<int>(slots.size()) != in_channels)
            {
                throw std::invalid_argument(
                    "feed_forward_slots takes d_model slot-form ciphertexts");
            }
            if (config.refresh_hidden && boot_key == nullptr)
            {
                throw std::invalid_argument(
                    "refresh_hidden needs the boot Galois key");
            }
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

            Range _r("b16.feed_forward_slots");
            note_depth("feed_forward_slots.in", slots);

            int block =
                config.hidden_block > 0 ? config.hidden_block : hidden;
            block = std::min(block, hidden);

            const double gate_scale =
                config.fold_silu_domain_into_gate ? 1.0 / config.silu_bound
                                                  : 1.0;

            // project() takes a BatchActivation, so the slot-form ciphertexts
            // are lent to one and taken back. Nothing is copied and nothing is
            // reinterpreted: the projection is a scalar multiply-accumulate
            // across whole ciphertexts, and `rows` is metadata pcmm does not
            // read -- it strides by layout_.d regardless.
            BatchActivation lent;
            lent.rows = tokens();
            lent.column = std::move(slots);

            std::vector<Ciphertext<Scheme::CKKS>> out;

            for (int base = 0; base < hidden; base += block)
            {
                const int cols = std::min(block, hidden - base);

                std::vector<double> gate_w = gate_slice(
                    weights.gate, in_channels, hidden, base, cols, gate_scale);
                std::vector<double> up_w = gate_slice(
                    weights.up, in_channels, hidden, base, cols, 1.0);

                BatchActivation gate = batch_.project(
                    lent, gate_w, in_channels, cols, "b16.ffn_slots.gate");
                BatchActivation up = batch_.project(
                    lent, up_w, in_channels, cols, "b16.ffn_slots.up");

                // No crossing. The SiLU and the gate product are slot-wise and
                // the operands are already in slot form.
                std::vector<Ciphertext<Scheme::CKKS>> hidden_slots;
                hidden_slots.reserve(gate.column.size());
                {
                    Range _r_silu("b16.ffn_slots.silu");
                    for (std::size_t j = 0; j < gate.column.size(); ++j)
                    {
                        Ciphertext<Scheme::CKKS> activated = arith().silu(
                            gate.column[j], config.silu_bound,
                            config.silu_degree, relin_key,
                            config.fold_silu_domain_into_gate);
                        hidden_slots.push_back(arith().multiply_and_rescale(
                            activated, up.column[j], relin_key));
                    }
                }
                gate.column.clear();
                up.column.clear();
                note_depth("feed_forward_slots.hidden", hidden_slots);

                if (config.refresh_hidden)
                {
                    bootstrap(hidden_slots, "feed_forward_slots.hidden",
                              *boot_key, relin_key);
                    note_depth("feed_forward_slots.hidden_refreshed",
                               hidden_slots);
                }

                BatchActivation h;
                h.rows = tokens();
                h.column = std::move(hidden_slots);

                const std::vector<double> down_w(
                    weights.down.begin() +
                        static_cast<std::size_t>(base) * in_channels,
                    weights.down.begin() +
                        static_cast<std::size_t>(base + cols) * in_channels);
                BatchActivation part = batch_.project(
                    h, down_w, cols, in_channels, "b16.ffn_slots.down");

                if (out.empty())
                {
                    out = std::move(part.column);
                }
                else
                {
                    Range _r_acc("b16.ffn_slots.accumulate");
                    for (std::size_t j = 0; j < out.size(); ++j)
                    {
                        arith().add_inplace(out[j], part.column[j]);
                    }
                }
            }

            // Give the caller's ciphertexts back.
            slots = std::move(lent.column);

            note_depth("feed_forward_slots.out", out);
            return out;
        }

        // -------------------------------------------------------------------
        // The refresh
        // -------------------------------------------------------------------

        int Llama3Batch16Operator::Batch16RefreshConfig::count() const
        {
            if (!enabled)
            {
                return 0;
            }
            return static_cast<int>(entry) +
                   static_cast<int>(after_attention_norm) +
                   static_cast<int>(after_attention) + static_cast<int>(mid) +
                   static_cast<int>(after_feed_forward_norm) +
                   static_cast<int>(feed_forward_hidden);
        }

        bool Llama3Batch16Operator::Batch16RefreshConfig::at(int position) const
        {
            switch (position)
            {
                case seam_entry:
                    return entry;
                case seam_after_attention_norm:
                    return after_attention_norm;
                case seam_after_attention:
                    return after_attention;
                case seam_mid:
                    return mid;
                case seam_after_feed_forward_norm:
                    return after_feed_forward_norm;
                case seam_feed_forward_hidden:
                    return feed_forward_hidden;
                default:
                    throw std::out_of_range(
                        "There is no such seam position on this block");
            }
        }

        void Llama3Batch16Operator::Batch16RefreshConfig::set(int position,
                                                              bool take)
        {
            switch (position)
            {
                case seam_entry:
                    entry = take;
                    break;
                case seam_after_attention_norm:
                    after_attention_norm = take;
                    break;
                case seam_after_attention:
                    after_attention = take;
                    break;
                case seam_mid:
                    mid = take;
                    break;
                case seam_after_feed_forward_norm:
                    after_feed_forward_norm = take;
                    break;
                case seam_feed_forward_hidden:
                    feed_forward_hidden = take;
                    break;
                default:
                    throw std::out_of_range(
                        "There is no such seam position on this block");
            }
        }

        const char* Llama3Batch16Operator::seam_name(int position)
        {
            switch (position)
            {
                case seam_entry:
                    return "block.entry";
                case seam_after_attention_norm:
                    return "block.after_attention_norm";
                case seam_after_attention:
                    return "block.after_attention";
                case seam_mid:
                    return "block.mid";
                case seam_after_feed_forward_norm:
                    return "block.after_feed_forward_norm";
                case seam_feed_forward_hidden:
                    return "feed_forward.hidden";
                default:
                    throw std::out_of_range(
                        "There is no such seam position on this block");
            }
        }

        Llama3Batch16Operator::RefreshPlan
        Llama3Batch16Operator::plan_refresh(const RefreshPlanInput& input)
        {
            if (static_cast<int>(input.stretch.size()) != seam_count)
            {
                throw std::invalid_argument(
                    "A refresh plan needs one stretch per seam position");
            }
            if (!input.width.empty() &&
                static_cast<int>(input.width.size()) != seam_count)
            {
                throw std::invalid_argument(
                    "The widths are either absent or one per seam position");
            }
            if (input.chain_limbs < 0 || input.refresh_levels < 0 ||
                input.entry_limbs < 0)
            {
                throw std::invalid_argument(
                    "A chain, a refresh and an entry are non-negative");
            }
            for (int s : input.stretch)
            {
                if (s < 0)
                {
                    throw std::invalid_argument(
                        "A stretch spends a non-negative number of levels");
                }
            }
            for (long long w : input.width)
            {
                if (w < 0)
                {
                    throw std::invalid_argument(
                        "A seam refreshes a non-negative number of "
                        "ciphertexts");
                }
            }

            const int declared_entry =
                input.entry_limbs > 0 ? input.entry_limbs : input.chain_limbs;
            // Limbs a refresh hands back. The run after it may therefore spend
            // one fewer than that, because the last prime is not spendable.
            const int after_refresh = input.chain_limbs - input.refresh_levels;

            RefreshPlan best;
            best.feasible = false;

            for (int mask = 0; mask < (1 << seam_count); ++mask)
            {
                // A plan is a partition of the six stretches into runs: the
                // PREFIX run, spent out of whatever the block was handed, and
                // one run after each refresh, spent out of what a refresh
                // hands back. Writing it that way rather than as a walk is
                // what makes the steady-state fixed point readable.
                int prefix = 0;
                int i = 0;
                for (; i < seam_count && !((mask >> i) & 1); ++i)
                {
                    prefix += input.stretch[i];
                }

                bool ok = true;
                int worst = prefix;
                int last_run = -1; // -1 while no refresh has been seen
                while (i < seam_count && ok)
                {
                    // A refresh must hand back something to spend, or the run
                    // after it cannot move at all.
                    if (after_refresh < 2)
                    {
                        ok = false;
                        break;
                    }
                    int run = input.stretch[i];
                    ++i;
                    for (; i < seam_count && !((mask >> i) & 1); ++i)
                    {
                        run += input.stretch[i];
                    }
                    if (run > after_refresh - 1)
                    {
                        ok = false;
                        break;
                    }
                    worst = std::max(worst, run);
                    last_run = run;
                }
                if (!ok)
                {
                    continue;
                }

                int entry_limbs = declared_entry;
                int exit_limbs = 0;
                if (last_run < 0)
                {
                    // No refresh anywhere: everything comes out of the entry.
                    exit_limbs = entry_limbs - prefix;
                }
                else
                {
                    exit_limbs = after_refresh - last_run;
                }

                if (input.steady_state)
                {
                    if (last_run < 0)
                    {
                        // A block that never refreshes strictly loses limbs
                        // unless it spends nothing, so it has no fixed point
                        // above zero.
                        if (prefix != 0)
                        {
                            continue;
                        }
                    }
                    // The block may be handed at most what it hands on, and it
                    // must be handed enough for the prefix run. When the entry
                    // seam is taken the prefix is empty and this is free,
                    // which is exactly what that seam buys.
                    entry_limbs = exit_limbs;
                }

                if (prefix > entry_limbs - 1)
                {
                    continue;
                }

                RefreshPlan candidate;
                candidate.feasible = true;
                candidate.worst_run = worst;
                candidate.exit_limbs = exit_limbs;
                candidate.config = Batch16RefreshConfig();
                candidate.config.enabled = mask != 0;
                for (int j = 0; j < seam_count; ++j)
                {
                    const bool take = ((mask >> j) & 1) != 0;
                    candidate.config.set(j, take);
                    if (take)
                    {
                        ++candidate.refreshes;
                        candidate.bootstraps +=
                            input.width.empty() ? 1 : input.width[j];
                    }
                }

                // Bootstraps first, seams second. The two disagree exactly
                // when one wide seam could be traded for two narrow ones,
                // which at the 8B shape is the trade worth making: the hidden
                // seam is 14,336 columns against 4,096 for every other.
                if (!best.feasible || candidate.bootstraps < best.bootstraps ||
                    (candidate.bootstraps == best.bootstraps &&
                     candidate.refreshes < best.refreshes))
                {
                    best = candidate;
                }
            }

            return best;
        }

        int Llama3Batch16Operator::refresh_levels(
            const BootstrappingConfig& config)
        {
            // The library's own accounting, not a measurement of it:
            // regular_bootstrapping runs CoeffToSlot in CtoS_piece linear
            // maps, EvalMod's sine at taylor_number, SlotToCoeff in
            // StoC_piece, and spends eight more on the modular reduction's
            // scaffolding. 25 at the default (3, 3, 11).
            return config.CtoS_piece_ + config.taylor_number_ +
                   config.StoC_piece_ + 8;
        }

        int Llama3Batch16Operator::chain_limbs_for(
            int worst_stretch, const BootstrappingConfig& config)
        {
            if (worst_stretch < 0)
            {
                throw std::invalid_argument(
                    "A stretch spends a non-negative number of levels");
            }
            // The refresh's own slice, the levels the stretch spends, and the
            // one prime the next bootstrap is handed. Longer than this is not
            // safer: every key switch to the next seam carries the unspent
            // limbs and then the seam throws them away.
            return refresh_levels(config) + worst_stretch + 1;
        }

        Ciphertext<Scheme::CKKS> Llama3Batch16Operator::bootstrap(
            Ciphertext<Scheme::CKKS>& ct, Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            // Encoding-blind on purpose. The procedure puts the plaintext
            // polynomial back the way it found it, so a matrix encryption is
            // refreshed where it stands and no crossing is needed in front of
            // it -- which is what makes batch 16's two-prime island legal.
            return arith().bootstrap(ct, boot_key, relin_key);
        }

        void Llama3Batch16Operator::bootstrap(
            std::vector<Ciphertext<Scheme::CKKS>>& ct, const char* name,
            Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (ct.empty())
            {
                return;
            }

            Range _r(name);
            // One at a time. A bootstrap fills the slots on its own, so there
            // is nothing for two of them to share, and holding a second
            // ciphertext at the raised modulus is exactly what a card running
            // this width does not have.
            for (auto& c : ct)
            {
                c = bootstrap(c, boot_key, relin_key);
            }

            apply_level_budget(ct, name);
        }

        void Llama3Batch16Operator::bootstrap(
            BatchActivation& x, const char* name,
            Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            bootstrap(x.column, name, boot_key, relin_key);
        }

        void Llama3Batch16Operator::apply_level_budget(
            std::vector<Ciphertext<Scheme::CKKS>>& ct, const char* seam_name)
        {
            if (!level_budget || ct.empty())
            {
                return;
            }
            const int keep = level_budget(seam_name);
            const int total = batch_.chain_limbs();
            if (keep <= 0 || total <= 0 || keep >= total)
            {
                return;
            }
            const int target = total - keep;
            for (auto& c : ct)
            {
                if (c.depth() < target)
                {
                    arith().drop_to_depth(c, target);
                }
            }
        }

        std::vector<int> Llama3Batch16Operator::rotation_indices() const
        {
            return batch_.rotation_indices();
        }

        std::vector<int> Llama3Batch16Operator::boot_rotation_indices() const
        {
            std::vector<int> all = batch_.rotation_indices();
            const std::vector<int> boot = batch_.arith().
                bootstrapping_key_indexs();
            all.insert(all.end(), boot.begin(), boot.end());
            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
            return all;
        }

        void Llama3Batch16Operator::seam(const char* name, BatchActivation& x,
                                         bool take,
                                         Galoiskey<Scheme::CKKS>* boot_key,
                                         Relinkey<Scheme::CKKS>& relin_key)
        {
            note_depth(name, x);
            if (!take)
            {
                return;
            }
            if (boot_key == nullptr)
            {
                throw std::invalid_argument(
                    std::string("The refresh at seam '") + name +
                    "' needs the boot Galois key");
            }
            bootstrap(x, name, *boot_key, relin_key);
            note_depth(name, x);
        }

        void Llama3Batch16Operator::seam(const char* name, Batch16Sequence& x,
                                         bool take,
                                         Galoiskey<Scheme::CKKS>* boot_key,
                                         Relinkey<Scheme::CKKS>& relin_key)
        {
            if (x.empty())
            {
                return;
            }
            note_depth(name, x.block.front());
            if (!take)
            {
                return;
            }
            if (boot_key == nullptr)
            {
                throw std::invalid_argument(
                    std::string("The refresh at seam '") + name +
                    "' needs the boot Galois key");
            }
            // Token blocks are independent here: a refresh is per ciphertext
            // and a block is a set of them, so a sequence costs blocks times
            // one block's refreshes and nothing else changes.
            for (auto& b : x.block)
            {
                bootstrap(b, name, *boot_key, relin_key);
            }
            note_depth(name, x.block.front());
        }

        void Llama3Batch16Operator::check_refresh(
            const TransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>* boot_key) const
        {
            if (!config.refresh.enabled)
            {
                return;
            }
            if (boot_key == nullptr && config.refresh.count() > 0)
            {
                // Checked before any work rather than at the first seam, so a
                // caller who forgot the key does not discover it half a block
                // in with the levels already spent.
                throw std::invalid_argument(
                    "A block with refresh.enabled needs the boot Galois key, "
                    "built from boot_rotation_indices()");
            }
        }

        // -------------------------------------------------------------------
        // The whole block
        // -------------------------------------------------------------------

        BatchActivation Llama3Batch16Operator::transformer_block(
            BatchActivation& x, const TransformerBlockWeights& weights,
            const TransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            validate(x, shape_.d_model, "transformer_block");
            check_refresh(config, boot_key);

            Range _r("b16.transformer_block");

            BatchActivation stream;
            stream.rows = x.rows;
            stream.column = x.column;

            seam("block.entry", stream,
                 config.refresh.enabled && config.refresh.entry, boot_key,
                 relin_key);

            {
                Range _r_half("b16.block.attention_half");
                Llama3BatchOperator::BatchAttentionWeights aw =
                    weights.attention;
                std::vector<double> gain = weights.attention_norm;
                if (config.fold_norm_scale && !gain.empty())
                {
                    // Q, K and V all read the normalised stream, so all three
                    // carry the gain. Doing it here rather than in the caller
                    // keeps the two halves of the identity together: the gain
                    // that leaves the norm is the gain that enters the weight.
                    fold_gain(aw.query, gain, shape_.d_model,
                              config.attention.q_channels);
                    fold_gain(aw.key, gain, shape_.d_model,
                              config.attention.kv_channels);
                    fold_gain(aw.value, gain, shape_.d_model,
                              config.attention.kv_channels);
                    gain.clear();
                }

                BatchActivation normed = rms_norm(
                    stream, gain, config.attention_norm, galois_key, relin_key);
                seam("block.after_attention_norm", normed,
                     config.refresh.enabled && config.refresh.after_attention_norm,
                     boot_key, relin_key);

                BatchActivation sub =
                    batch_.attention(normed, aw, config.attention, galois_key,
                                     relin_key, boot_key);
                normed.column.clear();
                seam("block.after_attention", sub,
                     config.refresh.enabled && config.refresh.after_attention,
                     boot_key, relin_key);

                // The two operands have been through completely different
                // circuits, so neither the level nor the scale lines up; this
                // is the one level a pre-norm residual pays.
                stream.column =
                    arith().residual_add(stream.column, sub.column);
                seam("block.mid", stream,
                     config.refresh.enabled && config.refresh.mid, boot_key,
                     relin_key);
            }

            {
                Range _r_half("b16.block.feed_forward_half");
                FeedForwardWeights fw = weights.feed_forward;
                FeedForwardConfig fc = config.feed_forward;
                fc.refresh_hidden = config.refresh.enabled &&
                                    config.refresh.feed_forward_hidden;
                std::vector<double> gain = weights.feed_forward_norm;
                if (config.fold_norm_scale && !gain.empty())
                {
                    // Only the gate and the up projection read the norm; the
                    // down projection reads the hidden and must NOT carry it.
                    fold_gain(fw.gate, gain, shape_.d_model, shape_.hidden);
                    fold_gain(fw.up, gain, shape_.d_model, shape_.hidden);
                    gain.clear();
                }

                BatchActivation normed =
                    rms_norm(stream, gain, config.feed_forward_norm,
                             galois_key, relin_key);
                seam("block.after_feed_forward_norm", normed,
                     config.refresh.enabled &&
                         config.refresh.after_feed_forward_norm,
                     boot_key, relin_key);

                BatchActivation sub =
                    feed_forward(normed, fw, fc, galois_key, relin_key,
                                 boot_key);
                normed.column.clear();
                stream.column =
                    arith().residual_add(stream.column, sub.column);
                note_depth("block.out", stream);
            }

            return stream;
        }

        // -------------------------------------------------------------------
        // A sequence longer than d tokens
        // -------------------------------------------------------------------

        void Llama3Batch16Operator::validate(const Batch16Sequence& x,
                                             int channels,
                                             const char* name) const
        {
            if (x.empty())
            {
                throw std::invalid_argument(
                    std::string(name) +
                    " takes at least one token block; an empty sequence is a "
                    "caller bug, not an identity");
            }
            for (int t = 0; t < x.blocks(); ++t)
            {
                validate(x.block[static_cast<std::size_t>(t)], channels, name);
                // Attention forms ONE denominator across every block, so a
                // level or scale drift between blocks is a silently wrong
                // SoftMax rather than an error from the library. This is the
                // only place that sees it before the sum happens.
                if (x.block[static_cast<std::size_t>(t)]
                        .column.front()
                        .depth() != x.block.front().column.front().depth() ||
                    x.block[static_cast<std::size_t>(t)]
                            .column.front()
                            .scale() != x.block.front().column.front().scale())
                {
                    throw std::invalid_argument(
                        std::string(name) +
                        " needs every token block at one level and one scale");
                }
            }
        }

        Llama3Batch16Operator::Batch16Sequence
        Llama3Batch16Operator::transformer_block(
            Batch16Sequence& x, const TransformerBlockWeights& weights,
            const TransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            validate(x, shape_.d_model, "transformer_block");
            check_refresh(config, boot_key);

            Range _r("b16.transformer_block_sequence");

            const int blocks = x.blocks();

            Batch16Sequence stream;
            stream.block.reserve(static_cast<std::size_t>(blocks));
            for (int t = 0; t < blocks; ++t)
            {
                BatchActivation copy;
                copy.rows = x.block[static_cast<std::size_t>(t)].rows;
                copy.column = x.block[static_cast<std::size_t>(t)].column;
                stream.block.push_back(std::move(copy));
            }

            seam("block.entry", stream,
                 config.refresh.enabled && config.refresh.entry, boot_key,
                 relin_key);

            {
                Range _r_half("b16.block_seq.attention_half");
                Llama3BatchOperator::BatchAttentionWeights aw =
                    weights.attention;
                std::vector<double> gain = weights.attention_norm;
                if (config.fold_norm_scale && !gain.empty())
                {
                    fold_gain(aw.query, gain, shape_.d_model,
                              config.attention.q_channels);
                    fold_gain(aw.key, gain, shape_.d_model,
                              config.attention.kv_channels);
                    fold_gain(aw.value, gain, shape_.d_model,
                              config.attention.kv_channels);
                    gain.clear();
                }

                // RMSNorm reduces over CHANNELS, so it never looks along the
                // token axis and a sequence is a plain loop over blocks. The
                // whole of token blocking's cost is in attention, and this is
                // the half of the block where that shows.
                Batch16Sequence normed;
                normed.block.reserve(static_cast<std::size_t>(blocks));
                for (int t = 0; t < blocks; ++t)
                {
                    normed.block.push_back(rms_norm(
                        stream.block[static_cast<std::size_t>(t)], gain,
                        config.attention_norm, galois_key, relin_key));
                }
                seam("block.after_attention_norm", normed,
                     config.refresh.enabled &&
                         config.refresh.after_attention_norm,
                     boot_key, relin_key);

                std::vector<BatchActivation> sub = batch_.attention_sequence(
                    normed.block, aw, config.attention, galois_key, relin_key,
                    boot_key);
                normed.block.clear();

                Batch16Sequence out_seq;
                out_seq.block = std::move(sub);
                seam("block.after_attention", out_seq,
                     config.refresh.enabled && config.refresh.after_attention,
                     boot_key, relin_key);

                for (int t = 0; t < blocks; ++t)
                {
                    stream.block[static_cast<std::size_t>(t)].column =
                        arith().residual_add(
                            stream.block[static_cast<std::size_t>(t)].column,
                            out_seq.block[static_cast<std::size_t>(t)].column);
                }
                seam("block.mid", stream,
                     config.refresh.enabled && config.refresh.mid, boot_key,
                     relin_key);
            }

            {
                Range _r_half("b16.block_seq.feed_forward_half");
                FeedForwardWeights fw = weights.feed_forward;
                FeedForwardConfig fc = config.feed_forward;
                fc.refresh_hidden = config.refresh.enabled &&
                                    config.refresh.feed_forward_hidden;
                std::vector<double> gain = weights.feed_forward_norm;
                if (config.fold_norm_scale && !gain.empty())
                {
                    fold_gain(fw.gate, gain, shape_.d_model, shape_.hidden);
                    fold_gain(fw.up, gain, shape_.d_model, shape_.hidden);
                    gain.clear();
                }

                Batch16Sequence normed;
                normed.block.reserve(static_cast<std::size_t>(blocks));
                for (int t = 0; t < blocks; ++t)
                {
                    normed.block.push_back(rms_norm(
                        stream.block[static_cast<std::size_t>(t)], gain,
                        config.feed_forward_norm, galois_key, relin_key));
                }
                seam("block.after_feed_forward_norm", normed,
                     config.refresh.enabled &&
                         config.refresh.after_feed_forward_norm,
                     boot_key, relin_key);

                for (int t = 0; t < blocks; ++t)
                {
                    // SwiGLU is slot-wise and blind to every index, so a token
                    // block is an independent call. Released as it goes rather
                    // than held: the whole sequence's hidden at once is
                    // blocks * 4 * hidden ciphertexts and no card has that.
                    BatchActivation sub = feed_forward(
                        normed.block[static_cast<std::size_t>(t)], fw, fc,
                        galois_key, relin_key, boot_key);
                    normed.block[static_cast<std::size_t>(t)].column.clear();
                    stream.block[static_cast<std::size_t>(t)].column =
                        arith().residual_add(
                            stream.block[static_cast<std::size_t>(t)].column,
                            sub.column);
                }
                note_depth("block.out", stream.block.front());
            }

            return stream;
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
