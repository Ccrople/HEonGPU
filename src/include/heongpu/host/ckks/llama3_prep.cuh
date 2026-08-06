// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The client-side model preparation of Sylph Section 3.1, for the rectangular
// Llama-3 path: what the model owner does to REAL weights, on the host, before
// anything is encrypted.
//
// WHY THIS EXISTS
// ---------------
// Everything the rect block runs was tuned on random matrices, where nothing
// is an outlier and every range is whatever the harness declares. Real Llama-3
// activations are not like that: the residual stream carries a few coordinates
// hundreds of times larger than the rest, and those coordinates are exactly
// what a CKKS refresh cannot carry -- EvalMod is fitted for values inside
// [-1, 1] and hands back noise, silently, outside it. Section 3.1 answers with
// three moves, all of them free at runtime, all of them implemented here:
//
//   1. SINK PREPENDING (3.1.1). A short public prefix -- BOS and a delimiter
//      run -- is prepended to the window, giving the attention mass a place to
//      park and keeping any single position from owning a SoftMax row. The
//      prefix is baked into the input the fetch script wrote; this module just
//      measures what it is worth.
//
//   2. ORTHOGONAL ROTATIONS (3.1.1). A randomised Hadamard R1 on the residual
//      stream spreads a dimension-wise outlier over all 4096 coordinates,
//      dividing its peak by up to sqrt(4096); a second rotation R2 does the
//      same per head between V and W_o. Both are fused into the weights --
//      R1 into the embedding/input and every projection touching the stream,
//      R2 into v_proj and o_proj -- so the circuit never sees them. RMSNorm
//      commutes with an orthogonal map of its argument (the sum of squares is
//      what it reads, and that is invariant), SiLU and the SoftMax act on
//      spaces R1 does not touch, so the fusion is EXACT, not approximate,
//      and the tests pin it at double precision.
//
//   3. THE 1/B PRE-SCALING (3.1.3). Every refreshed seam gets its values into
//      [-1, 1] by a constant that rides on something already being paid for:
//      the stream on the input and the two projections that write the stream
//      (B_s), the normalised stream on the RMSNorm fit's own coefficients
//      (B_n, undone by the projections that read it), V on v_proj (B_v,
//      undone by o_proj), the SwiGLU hidden on up_proj (B_h, undone by
//      down_proj). Zero additional operations and zero levels; the cost is
//      log2(B) bits of the refresh's precision, which is the paper's stated
//      trade. eps is scaled by 1/B_s^2 so the norm is the SAME function of
//      the SAME stream, not a nearby one.
//
// The bounds B are MEASUREMENTS, not assumptions: prepare_block runs the exact
// circuit in double precision on the real input, reads every seam's peak, and
// rounds up to a power of two with margin. The same run supplies every
// calibrated range the fits need -- the RMSNorm sum, the score interval, the
// SoftMax denominator -- which is Section 4.3 done with real numbers.

#ifndef HEONGPU_CKKS_LLAMA3_PREP_H
#define HEONGPU_CKKS_LLAMA3_PREP_H

#include <heongpu/host/ckks/llama3_rect.cuh>

#include <cstdint>
#include <string>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief One real Llama-3 block as fetch_llama3_weights.py wrote it.
         *
         * Weights are row-major in_channels x out_channels -- the operator's
         * own storage, transposed from the checkpoint by the fetch script.
         * The two inputs are the real residual stream reaching this layer,
         * tokens x channels, computed by the true model prefix (RoPE and
         * all): one behind the sink prefix, one without it.
         */
        struct RealBlockBundle
        {
            int layer = 0;
            int tokens = 0;
            int channels = 0;
            int kv_channels = 0;
            int head_dim = 0;
            int hidden = 0;
            int sink_tokens = 0;

            std::vector<double> query, key, value, output;
            std::vector<double> gate, up, down;
            std::vector<double> attention_norm, feed_forward_norm;
            std::vector<double> input, input_nosink;
        };

        /** @brief Read a bundle directory; throws on any missing piece. */
        RealBlockBundle load_real_block(const std::string& dir);

        /** @brief Shape of a block, as the host reference needs it. */
        struct BlockShape
        {
            int tokens = 0;
            int channels = 0;
            int kv_channels = 0;
            int head_dim = 0;
            int hidden = 0;

            int heads() const { return channels / head_dim; }
            int kv_heads() const { return kv_channels / head_dim; }
        };

        /**
         * @brief What one exact host run of the profiled circuit measured.
         *
         * Everything is in the units of the weights it was run with: run it
         * on folded weights and these are the circuit's own operating ranges,
         * ready to be handed to the configs.
         */
        struct BlockCalibration
        {
            /// Peak |entry| of the residual stream at the three points a
            /// refresh sees it: entry, between the halves, and on the way out
            /// (the next block's entry).
            double stream_entry_abs = 0.0;
            double stream_mid_abs = 0.0;
            double stream_out_abs = 0.0;
            /// Summed-square range per norm: [0] attention, [1] feed-forward.
            double norm_sum_lo[2] = {0.0, 0.0};
            double norm_sum_hi[2] = {0.0, 0.0};
            /// Peak |entry| of each normalised stream.
            double normed_abs[2] = {0.0, 0.0};
            /// Causal-visible score range, head scale applied.
            double score_lo = 0.0;
            double score_hi = 0.0;
            /// Range over rows and heads of the round-zero SoftMax
            /// denominator sum_j exp(2 (s - score_hi) / 2^k) under the
            /// calibrated shift, indexed by k - 1: the denominator depends
            /// on how many normalise-and-square rounds the circuit runs, and
            /// the calibrated score range is what decides that, so all four
            /// are measured in one pass and the profile reads the one its k
            /// picked.
            double softmax_sum_lo[4] = {0.0, 0.0, 0.0, 0.0};
            double softmax_sum_hi[4] = {0.0, 0.0, 0.0, 0.0};
            /// Worst sum of squared probabilities over rows and heads: how
            /// concentrated the sharpest row is. SoftmaxConfig::concentration
            /// wants it as a multiple of the uniform 1/d, so multiply by the
            /// axis length before handing it over. Read only when the score
            /// range forces more than one normalise-and-square round.
            double prob_sq_hi = 0.0;
            /// Peak |entry| of V, of the SiLU argument, of the SwiGLU hidden.
            double value_abs = 0.0;
            double silu_in_abs = 0.0;
            double hidden_abs = 0.0;
        };

        /**
         * @brief The exact circuit, in double precision on the host.
         *
         * This is the PROFILED block, not the true model: no RoPE, grouped
         * queries read repeated kv heads, the SoftMax is causal and exact.
         * Handed the same weights the operator is handed, it computes what
         * the FHE block ideally computes, which makes it both the calibrator
         * and the yardstick the decrypted result is judged against.
         *
         * @param attn_norm_gain,ffn_norm_gain the RMSNorm output_scale each
         *        norm runs at (the 1/B_n of the fold, or 1).
         */
        std::vector<double> reference_block(
            const Llama3RectOperator::RectTransformerBlockWeights& weights,
            const std::vector<double>& input, const BlockShape& shape,
            double eps, double attn_norm_gain, double ffn_norm_gain,
            BlockCalibration* calibration = nullptr);

        /** @brief In-place orthonormal Walsh-Hadamard transform; n a power
         *         of two. */
        void fwht_normalised(double* v, std::size_t n);

        /**
         * @brief Apply or undo the stream rotation prepare_block fused.
         *
         * The client-side half of Section 3.1.1: the same seed regenerates
         * the same signs, forward is what prepare_block did to the input,
         * inverse is what a reader does to a block output. Rows are tokens,
         * and the rotation acts on each row's @p channels.
         */
        void rotate_stream(std::vector<double>& x, int rows, int channels,
                           std::uint64_t seed, bool inverse);

        /** @brief Switches of the preparation. */
        struct SylphPrepConfig
        {
            /// Section 3.1.1's rotations, fused into the weights.
            bool rotate = true;
            /// Section 3.1.3's 1/B folds, fused into the weights and the
            /// norm gains.
            bool prescale = true;
            /// Headroom multiplied onto every measured peak before the bound
            /// is rounded up to a power of two.
            double margin = 1.25;
            double eps = 1e-5;
            /// Seed of the Hadamard sign vectors. Client-side and public:
            /// the rotation hides nothing, it only spreads.
            std::uint64_t seed = 20260806u;
        };

        /** @brief The bounds the folds chose; all one when prescale is off. */
        struct SylphScales
        {
            double stream = 1.0;      ///< B_s
            double normed_attn = 1.0; ///< B_n of the attention norm
            double normed_ffn = 1.0;  ///< B_n of the feed-forward norm
            double value = 1.0;       ///< B_v
            double hidden = 1.0;      ///< B_h
        };

        /**
         * @brief A real block, prepared: rotated, folded, calibrated.
         *
         * @c weights carries every fusion, with the norm gain vectors left
         * EMPTY -- the learned gains are folded into the projections here, so
         * the block must run with fold_norm_scale off. @c input is rotated
         * and scaled, ready to encrypt as it stands. The stream the block
         * hands back is in the same convention, so the host reads it by
         * multiplying B_s back and undoing the rotation -- or hands it
         * straight to the next block, which is the point of a convention.
         */
        struct PreparedBlock
        {
            Llama3RectOperator::RectTransformerBlockWeights weights;
            std::vector<double> input;
            std::vector<double> input_nosink;
            BlockShape shape;
            SylphScales scales;
            /// config.eps / B_s^2: the same norm on the scaled stream.
            double eps = 0.0;
            /// The circuit's operating ranges: measured on the folded
            /// weights and the sink input, so they feed the configs as they
            /// stand.
            BlockCalibration calibration;
            /// The same measurement before any fold, sink and no-sink: what
            /// the ranges WOULD be, which is the mitigation table.
            BlockCalibration raw;
            BlockCalibration raw_nosink;
            /// reference_block on the folded weights and @c input: what the
            /// decrypted block output should be, in circuit units.
            std::vector<double> expected;
        };

        /**
         * @brief Fold Section 3.1 into one real block.
         *
         * Rotations first, then one exact reference run to measure the
         * bounds, then the folds, then a second run for the calibrated
         * ranges and the expected output. Pure host arithmetic; nothing here
         * touches a ciphertext.
         */
        PreparedBlock prepare_block(const RealBlockBundle& bundle,
                                    const SylphPrepConfig& config);

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_PREP_H
