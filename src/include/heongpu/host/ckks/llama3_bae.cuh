// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// A LLAMA-3 BLOCK ON THE BAE PRODUCT, WITH NO ENCODING CROSSING AT ALL
// ====================================================================
//
// LLAMA3_8B_LAYER_FLOW.md §25.5 left one thing open: HEBaePcmmOperator is a
// drop-in for a projection, but Llama3RectOperator still drives Algorithm 5,
// "because the two do not agree on the activation layout, so switching the
// BLOCK over means moving every crossing, not changing a call".
//
// This module is that step, and the answer turned out to be better than
// moving them.
//
// THE OBSERVATION
// ---------------
// At k = 1 the Bae product is
//
//     out_i = sum_j U[i][j] * ct_j,
//
// a scalar multiply-accumulate over WHOLE ciphertexts. ModDecomp and ModPack
// are the identity, nothing inside a ciphertext is touched, and no coefficient
// ever meets another coefficient. A linear combination of whole ciphertexts
// commutes with every linear map -- with the NTT, and therefore with the
// canonical embedding as well. So the k = 1 product does not care what a
// ciphertext ENCODES.
//
// That is a stronger statement than "the projection chains with itself", which
// is all §25.3 measured. It says the projection can be taken in SLOT form.
//
// WHAT FOLLOWS FROM IT
// --------------------
// Hold the stream as ONE CHANNEL PER CIPHERTEXT with the tokens in the slots.
// Then:
//
//   projection    the Bae product, in place, in slot form. One level, no
//                 rotation, no Galois key, no relinearisation, and -- with
//                 set_transform_free -- not even an NTT.
//   RMSNorm       the channel axis IS the ciphertext axis, so the sum of
//                 squares is an ADDITION ACROSS CIPHERTEXTS. No rotation, no
//                 mask, no level for the reduction. The learned gain is a
//                 per-channel CONSTANT and folds into the next projection's
//                 weight on the host, for nothing.
//   SwiGLU        elementwise, so it is elementwise here too.
//   residual      an addition.
//
// None of those needs a coefficient reading, so NONE OF THEM NEEDS A CROSSING.
// The eight crossings §25.3 inventoried for the rect path do not move to a
// different seam; outside attention they cease to exist, and with them every
// Galois key the block owns. What remains is one relinearisation key.
//
// This is the same structural fact the batch-16 path measured for Kang's
// Algorithm 1 ("a channel is a whole ciphertext, so RMSNorm's channel
// reduction is a slot-wise ADDITION"), reached without Algorithm 1's
// constraint that the batch size choose the ring.
//
// WHAT IT COSTS, STATED BEFORE THE API RATHER THAN AFTER
// ------------------------------------------------------
// Two things, and neither is hidden anywhere below.
//
// 1. ARITHMETIC. k = 1 means d3 = N, so the b-part GEMM is as wide as the
//    a-part and the product is 2*d1*d2*N modular MACs per limb. Slot form
//    carries N/2 real tokens in those N coefficients, so it is 4*d1*d2 MACs
//    per token per limb -- twice the coefficient-domain figure of §25.2, and
//    that factor of two is the price of the encoding, paid knowingly.
//
// 2. ATTENTION IS NOT HERE, and it is not an oversight. In this packing a
//    score S[t][t'] = sum_c Q[c][t] K[c][t'] needs one ciphertext-ciphertext
//    product per (channel, token offset) pair, i.e. head_dim * tokens per
//    head, and the score matrix itself is quadratic in the token count -- the
//    very axis this encoding spends the ring on. Bae's paper says outright
//    that it does not do ciphertext-ciphertext multiplication. So attention
//    needs a partner primitive (Kang's Algorithm 4) and a crossing to reach
//    it, and that crossing is the whole remaining integration cost. See
//    §25.10. What this module delivers END TO END is the FEED-FORWARD half of
//    a block: RMSNorm, three projections, the SiLU, the gate product and the
//    residual -- three of a block's seven projections. rms_norm() and
//    project() are not specific to that half and serve the attention
//    sublayer's norm and its four projections equally; what is missing there
//    is only Q K^T and P V.

#ifndef HEONGPU_CKKS_LLAMA3_BAE_H
#define HEONGPU_CKKS_LLAMA3_BAE_H

#include <heongpu/host/ckks/baepcmm.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/decryptor.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/encryptor.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/llama3.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/plaintext.cuh>

#include <string>
#include <utility>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief An activation with one channel per ciphertext.
         *
         * @c column[c] carries channel @c c, and its slot @c t carries token
         * @c t. That is the whole encoding: there is no block index, no
         * interleave and no padding rule, because the two axes of the
         * activation are the two axes the ciphertext already has.
         *
         * The ciphertexts are ordinary slot-encoded CKKS ciphertexts. Nothing
         * in this module reads them as coefficients, which is exactly why
         * nothing in this module crosses.
         */
        struct BaeActivation
        {
            std::vector<Ciphertext<Scheme::CKKS>> column;

            /// Tokens carried in the slots. At most slot_count().
            int tokens = 0;
            /// Channels, one per ciphertext. Always equals column.size().
            int channels = 0;

            bool empty() const { return column.empty(); }
        };

        /**
         * @brief The Llama-3 block on the Bae product at k = 1.
         *
         * Composed on Llama3Operator for the slot-form polynomial primitives
         * (1/sqrt, SiLU, the Chebyshev evaluator) rather than duplicating
         * them, and on HEBaePcmmOperator for the product itself.
         */
        class Llama3BaeOperator
        {
          public:
            /**
             * @param tokens Tokens an activation carries. Must be positive and
             *               at most the slot count; the rest of the slots are
             *               zero and cost what zeros cost.
             * @param scale  Default scaling factor, shared with the slot-form
             *               operator.
             */
            Llama3BaeOperator(HEContext<Scheme::CKKS> context,
                              HEEncoder<Scheme::CKKS>& encoder, int tokens,
                              double scale);

            int tokens() const noexcept { return tokens_; }
            int slot_count() const noexcept { return slot_count_; }
            int ring_degree() const noexcept { return n_; }
            double default_scale() const noexcept { return default_scale_; }

            Llama3Operator& arith() noexcept { return arith_; }
            HEBaePcmmOperator<Scheme::CKKS>& product() noexcept
            {
                return pcmm_;
            }

            /**
             * @brief Galois indices this module needs: none, ever.
             *
             * Returned as an empty vector rather than simply documented,
             * because a caller assembling a key union should be able to ask
             * this module the same question it asks Llama3RectOperator and get
             * an answer it can concatenate. Attention, when it exists, will
             * not be able to say this.
             */
            static std::vector<int> rotation_indices() { return {}; }

            // ---------------------------------------------------------------
            // Staging
            // ---------------------------------------------------------------

            /**
             * @brief Encrypt a tokens x channels row-major host matrix.
             *
             * The transpose of what the encoding stores, which is the form a
             * caller has: a row is a token.
             */
            BaeActivation encrypt(const std::vector<double>& x, int tokens,
                                  int channels,
                                  HEEncryptor<Scheme::CKKS>& encryptor);

            /** @brief Inverse of encrypt, tokens x channels row major. */
            std::vector<double>
            decrypt(BaeActivation& x, HEDecryptor<Scheme::CKKS>& decryptor);

            // ---------------------------------------------------------------
            // Projection
            // ---------------------------------------------------------------

            /**
             * @brief Y = X W, on the Bae product.
             *
             * @param weight Row major @c in_channels x @c out_channels, the
             *               transpose of the mathematical weight -- the same
             *               form Llama3RectOperator::project takes, so the two
             *               are interchangeable at the call site.
             *
             * Spends exactly one level and touches no key of any kind. The
             * result is a fresh activation; @p x is left alone.
             */
            BaeActivation project(BaeActivation& x,
                                  const std::vector<double>& weight,
                                  int in_channels, int out_channels,
                                  const char* name);

            /**
             * @brief Fold a per-channel gain into the weight that follows it.
             *
             * RMSNorm's learned scale multiplies channel c of its output, and
             * the next thing a normalised stream meets is always a projection,
             * whose weight row c multiplies the same channel. So
             * W'[c][o] = gain[c] * W[c][o] applies the gain EXACTLY and for
             * nothing, where applying it to the ciphertext costs a level.
             *
             * This is free here and is not free on the rect path, where a
             * channel is (ciphertext, fast slot index) and the gain is a slot
             * vector rather than a constant.
             *
             * @param weight In place, row major in_channels x out_channels.
             */
            static void fold_channel_gain(std::vector<double>& weight,
                                          const std::vector<double>& gain,
                                          int in_channels, int out_channels);

            // ---------------------------------------------------------------
            // RMSNorm
            // ---------------------------------------------------------------

            struct BaeRMSNormConfig
            {
                double eps = 1e-5;
                /// Range of the SUMMED square over all channels.
                double sum_lo = 0.0;
                double sum_hi = 0.0;
                int degree = 31;
                int newton_iterations = 2;
                /// Fit 1/sqrt over the summed square rather than over the
                /// mean, which saves the level that forming the mean costs.
                /// Ignored when @c newton_iterations is above zero, since a
                /// Newton step refines against the unmapped argument.
                bool fold_mean_into_fit = true;
                /// Multiplied into the fitted 1/sqrt, so the norm hands back
                /// output_scale * x / rms(x) for the price of x / rms(x).
                /// Needs @c newton_iterations = 0 when it is not one.
                double output_scale = 1.0;
            };

            /**
             * @brief RMSNorm over the channel axis.
             *
             * The channel axis is the ciphertext axis, so the sum of squares
             * is @c channels squarings and @c channels-1 additions and there
             * is NO reduction inside a ciphertext: no rotation, no mask, no
             * Galois key and no level beyond the squaring itself. That is the
             * whole difference from the rect path, whose channel axis runs
             * partly along the fast slot axis and pays a masked blocked
             * reduction for it.
             *
             * @param gain One learned scale per channel, or empty. Applying it
             *             here costs one level; fold_channel_gain() applies
             *             the same numbers to the following weight for free,
             *             and is what the block driver does.
             */
            BaeActivation rms_norm(BaeActivation& x,
                                   const std::vector<double>& gain,
                                   const BaeRMSNormConfig& config,
                                   Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Feed forward
            // ---------------------------------------------------------------

            struct BaeFeedForwardWeights
            {
                /// Row major in_channels x hidden.
                std::vector<double> gate;
                /// Row major in_channels x hidden.
                std::vector<double> up;
                /// Row major hidden x in_channels. Empty skips W_down.
                std::vector<double> down;
            };

            struct BaeFeedForwardConfig
            {
                int in_channels = 0;
                int hidden = 0;
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;
                /// Multiplies the fitted SiLU, so the gate leg returns
                /// gain * SiLU(x) for the same levels. Free; it is a change of
                /// host-side Chebyshev coefficients.
                double silu_gain = 1.0;
            };

            /**
             * @brief W_down ( SiLU(W_gate x) * W_up x ).
             *
             * Every leg is elementwise or a projection, so the whole sublayer
             * runs without a rotation. The gate product is the only
             * ciphertext-ciphertext multiplication in it, one per hidden
             * channel.
             */
            BaeActivation feed_forward(BaeActivation& x,
                                       const BaeFeedForwardWeights& weights,
                                       const BaeFeedForwardConfig& config,
                                       Relinkey<Scheme::CKKS>& relin_key);

            /** @brief x + sublayer, levels and scales aligned. */
            BaeActivation residual_add(BaeActivation& x,
                                       BaeActivation& sublayer);

            // ---------------------------------------------------------------
            // The feed-forward half of a block
            // ---------------------------------------------------------------

            struct BaeFeedForwardBlockWeights
            {
                /// One learned scale per channel, or empty.
                std::vector<double> norm_gain;
                BaeFeedForwardWeights feed_forward;
            };

            /**
             * @brief RMSNorm, SwiGLU, residual -- the second half of a block.
             *
             * The norm's learned gain is folded into the gate and up weights
             * rather than multiplied into the stream, which is exact and costs
             * nothing; see fold_channel_gain.
             *
             * @param depth_trace Optional. Receives (name, depth) at each seam,
             *                    in execution order, so a caller can read the
             *                    level ledger off a run instead of arguing it.
             */
            BaeActivation feed_forward_block(
                BaeActivation& x, const BaeFeedForwardBlockWeights& weights,
                const BaeRMSNormConfig& norm_config,
                const BaeFeedForwardConfig& ffn_config,
                Relinkey<Scheme::CKKS>& relin_key,
                std::vector<std::pair<std::string, int>>* depth_trace =
                    nullptr);

            // ---------------------------------------------------------------
            // Cost model
            // ---------------------------------------------------------------

            /**
             * @brief What one feed-forward half costs, exactly, by count.
             *
             * Arithmetic over the shape, not a measurement. Every field is a
             * count of operations the code above actually issues, so it can be
             * put beside §25.2's Algorithm 5 figures without a benchmark in
             * between.
             */
            struct Counts
            {
                /// Ciphertext-ciphertext products, each a relinearisation.
                long long relinearisations = 0;
                /// Galois rotations. Zero, and that is the result.
                long long rotations = 0;
                /// Distinct Galois keys the circuit needs. Also zero.
                long long galois_keys = 0;
                /// Encoding crossings. Also zero.
                long long crossings = 0;
                /// Modular multiply-accumulates per RNS limb, both GEMMs.
                double gemm_macs = 0.0;
            };

            /**
             * @param d_model Residual width.
             * @param hidden  Feed-forward width.
             *
             * Levels are deliberately NOT modelled here. They depend on the
             * fit degrees and on which folds are on, and the module already
             * reports them exactly: pass a depth trace to
             * feed_forward_block() and read them off a run.
             */
            Counts feed_forward_block_counts(int d_model, int hidden) const;

          private:
            void require_uniform(const BaeActivation& x,
                                 const char* name) const;

            HEContext<Scheme::CKKS> context_;
            HEEncoder<Scheme::CKKS>& encoder_;
            Llama3Operator arith_;
            HEBaePcmmOperator<Scheme::CKKS> pcmm_;

            int n_;
            int slot_count_;
            int tokens_;
            double default_scale_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_BAE_H
