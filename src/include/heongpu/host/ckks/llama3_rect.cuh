// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The Llama-3 primitives with EVERY plaintext-ciphertext product on Kang's
// Algorithm 5 (rectangular batch PCMM) and every ciphertext-ciphertext product
// on Kang's Algorithm 4 (batch CCMM), from Cheon, Kang and Lee, "Fast Batch
// Matrix Multiplication in Ciphertexts".
//
// WHY THIS FILE EXISTS
// --------------------
// llama3_batch.cuh runs the model on Algorithm 1, where the k/2 packed matrices
// are k/2 INDEPENDENT INPUTS. That is pure throughput and it is the right answer
// when there are k/2 users. It is the wrong answer for one: a single input
// leaves k/2 - 1 slots empty, and the activation is still one ciphertext per
// CHANNEL, which is the encoding that ran out of ciphertext memory before it ran
// out of anything else.
//
// Algorithm 5 spends the batch axis on the input instead. A d x (N/2) matrix is
// k/2 blocks of d x d, and Theorem 3 sums the block products, so ONE input
// occupies the whole ring: N/2 channels in d ciphertexts rather than N/2 of
// them, a factor of k/2 fewer.
//
// THE THREE ENCODINGS, AND WHY THERE ARE THREE
// --------------------------------------------
// This module moves between three representations of the same d x C activation.
// They are not relabellings of each other and each product below states which
// one it consumes.
//
//   RECT     d ciphertexts per group of N/2 channels. Column ciphertext j holds
//   (Y axis) X[token i][channel t*d + j] at coefficient i + d*t, so the channel
//            BLOCK index t rides the Y = X^d axis. This is what Algorithm 5
//            hands back, and feeding it straight back in is what makes a chain
//            of projections free: the constant coefficient of an R_k product is
//            already the contraction over t once the plaintext carries its own
//            blocks reversed and negated (Y^k = -1). The stream lives here.
//
//   BATCH    the matrix encryption of Definition 2: the block index rides the
//   (batch)  BATCH axis, entry (i,j) evaluating to X[i][b*d + j] at zeta^{5^b}.
//            This is what Algorithm 4 consumes, and it is what makes the batch
//            CCMM the right tool for attention: with head_dim = d a block IS a
//            head, so one Algorithm 4 call produces k/2 heads' score matrices
//            at once.
//
//   SLOT     ordinary CKKS slots, slot b + (k/2)*u of column j holding
//            X[token u][channel b*d + j]. Every non-linearity -- RMSNorm, the
//            SoftMax, SiLU -- is slot-wise polynomial evaluation and needs this.
//
// THE MISSING STAGE, WHICH IS THE ENGINEERING CONTENT OF THIS FILE
// ---------------------------------------------------------------
// RECT and BATCH differ by a transform, and it is easy to convince oneself it
// is free. It is not. Evaluating a rect column at the slot points gives
//
//     slot_{b,u} = sum_i psi^{5^{b + u k/2} i} * ( sum_t X[i][t d + j] zeta^{5^b t} ),
//
// so the rect encoding carries, at batch point b, the DISCRETE FOURIER
// TRANSFORM of the blocks rather than block b. Undoing it is a (k/2)-point
// transform along the batch axis. That axis is the FAST slot axis, span k/2, so
// the map is the blocked linear map of k - 1 diagonals: k - 1 rotations, k - 1
// plaintext products and one level.
//
// It does NOT commute with the row Vandermonde of the slot bridge, because that
// Vandermonde's nodes psi^{5^{b + u k/2}} depend on b as well as on u. So the
// two cannot be fused, and RECT -> BATCH is three stages and three levels:
// the row bridge down to slots, the block transform, the row bridge back up.
// A single BSGS linear map over all N/2 diagonals would do it in one level and
// about 2 sqrt(N/2) rotations, at the cost of N/2 plaintext encodings per
// ciphertext; that trade has not been measured and is not taken here.
//
// This is the transform profile_llama3_nobatch.cpp records as "not written
// yet", and without it Algorithm 5 can only do projections.
//
// WHAT IT COSTS, STATED UP FRONT
// ------------------------------
// Algorithm 5's summation is a CMT read at k = 2, whose automorphisms run over
// the WHOLE rotation group: N/2 - 1 Galois keys, a count that depends on the
// ring degree ALONE -- not on d, not on the width of the model. HEonGPU's
// minimum ring is 4096, so the floor is 2047 keys, and at the ~60-limb chain a
// whole transformer block wants that is far past one card. Nothing in this file
// can move that; N is the only lever, and it is capped below. Read the shape
// checks accordingly: this module is correct at any width, and it FITS only at
// short chains.
//
// ORIENTATION AND SHAPE CONSTRAINTS
// ---------------------------------
// Kang's products put the encrypted operand on the LEFT, while a Llama
// projection is W(plaintext) X(encrypted). As on the Algorithm 1 path the
// activation is therefore held transposed, X[token][channel], and the weights
// are stored transposed on the host where a transpose is free.
//
//   rows      d, the tokens of one block. Fixed by the ring: d is the rank of
//             R_N over R_k.
//   channels  free, in groups of N/2. A width that is not a whole number of
//             groups is padded, and the padding is paid for.
//   head_dim  EXACTLY d. Algorithm 4 contracts over the column index and leaves
//             the batch index alone, so a head has to be one channel block for
//             its contraction to be the one the batch CCMM performs. This is
//             the single real shape constraint the encoding imposes.

#ifndef HEONGPU_CKKS_LLAMA3_RECT_H
#define HEONGPU_CKKS_LLAMA3_RECT_H

#include <heongpu/host/ckks/batchmatrix.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/decryptor.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/encryptor.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/llama3.cuh>
#include <heongpu/host/ckks/llama3_batch.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/plaintext.cuh>

#include <complex>
#include <cstdint>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief An activation in the rectangular encoding of Algorithm 5.
         *
         * @c column holds @c groups * d ciphertexts, group major: group g owns
         * @c column[g*d .. g*d + d), and carries channels
         * [g * (N/2), (g+1) * (N/2)). Within a group, column ciphertext j holds
         * X[token i][channel t*d + j] at coefficient i + d*t, so the block index
         * t rides the Y axis and the whole group is d ciphertexts however wide
         * the ring is.
         *
         * @c channels may be less than @c groups * (N/2); the remainder is
         * zero padding, and it is padding that gets multiplied like anything
         * else, so it costs what it costs.
         */
        struct RectActivation
        {
            std::vector<Ciphertext<Scheme::CKKS>> column;

            /// Rows of the encrypted matrix: layout.d, the tokens of one block.
            int rows = 0;
            /// Groups of N/2 channels the width is cut into.
            int groups = 0;
            /// Channels actually carried, at most groups * (N/2).
            int channels = 0;

            bool empty() const { return column.empty(); }
        };

        /**
         * @brief The Llama-3 linear algebra on Kang's Algorithms 4 and 5.
         *
         * Every projection is Algorithm 5 and every encrypted product is
         * Algorithm 4. The non-linear layers are not reimplemented: they stay on
         * Llama3Operator in slot form, behind the bridge.
         *
         * Composed on Llama3BatchOperator rather than duplicating it. That class
         * already owns the row Vandermonde bridge, the batch matrix operator and
         * the slot-form arithmetic operator, and building second copies of all
         * three would double the twiddle caches for nothing.
         */
        class Llama3RectOperator
        {
          public:
            /**
             * @param layout Subring layout. @c layout.d is the token block size
             *               and fixes both the channel block size and the number
             *               of blocks per group, k/2.
             * @param scale  Default scaling factor, shared with the slot-form
             *               operator.
             */
            Llama3RectOperator(HEContext<Scheme::CKKS> context,
                               HEEncoder<Scheme::CKKS>& encoder,
                               const BatchMatrixLayout& layout, double scale);

            const BatchMatrixLayout& layout() const noexcept { return layout_; }

            /** @brief Tokens one activation carries, layout.d. */
            int rows() const noexcept { return layout_.d; }

            /** @brief Channels one group carries, N/2. */
            int channels_per_group() const noexcept { return layout_.N / 2; }

            /** @brief Channel blocks in one group, k/2. */
            int blocks_per_group() const noexcept { return layout_.batch; }

            /** @brief Groups a width of @p channels needs. */
            int groups_for(int channels) const;

            double default_scale() const noexcept { return default_scale_; }

            // ---------------------------------------------------------------
            // Rotation keys
            // ---------------------------------------------------------------

            /**
             * @brief Rotations the block transform needs a Galois key for.
             *
             * The transform is blocked at span k/2 with the block index on the
             * FAST slot axis, so its diagonals are the offsets
             * -(k/2 - 1) .. k/2 - 1, taken modulo the slot count.
             */
            std::vector<int> block_rotation_indices() const;

            /**
             * @brief Every rotation index this operator can ask for.
             *
             * The union of Algorithm 5's own indices, the row bridge's, the
             * block transform's and the blocked reduction RMSNorm runs. In
             * practice Algorithm 5's set already covers it: the CMT at k = 2
             * runs the automorphisms over the whole rotation group, so this is
             * N/2 - 1 indices whatever else is added to it. It is still built
             * honestly, so the list means something if the rectangular product
             * is ever taken out.
             */
            std::vector<int> rotation_indices() const;

            // ---------------------------------------------------------------
            // Host-side staging, for tests and for the client side
            // ---------------------------------------------------------------

            /**
             * @brief Encode and encrypt a d x @p channels real matrix.
             *
             * @param x Row major, @c rows() * @p channels entries.
             */
            RectActivation encrypt(const std::vector<double>& x, int channels,
                                   HEEncryptor<Scheme::CKKS>& encryptor,
                                   double scale);

            /** @brief Inverse of encrypt, for checking a result. */
            std::vector<double> decrypt(RectActivation& in,
                                        HEDecryptor<Scheme::CKKS>& decryptor,
                                        double scale);

            // ---------------------------------------------------------------
            // Moving between the three encodings
            // ---------------------------------------------------------------

            /**
             * @brief The block transform, in place, on slot-form ciphertexts.
             *
             * @param inverse false applies F, the map that takes block values to
             *                their transform along the batch axis; true applies
             *                F^-1, which is what turns a rect group into
             *                separated channels. F[b][t] = zeta^{5^b t} is a
             *                Vandermonde in k/2 distinct 2k-th roots of unity,
             *                so it is invertible and well conditioned.
             *
             * Costs k - 1 rotations, k - 1 plaintext products and one level per
             * ciphertext. The diagonals are shared by every ciphertext in the
             * call and encoded once, which the row bridge does not do.
             */
            void block_map(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                           bool inverse, const char* name,
                           Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Rectangular form to slot form.
             *
             * Slot b + (k/2)*u of output ciphertext g*d + j holds
             * X[token u][channel g*(N/2) + b*d + j]: the token index on the slow
             * axis and the channel block on the fast one. Two levels, being the
             * row bridge and then the block transform.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            to_slots(RectActivation& in, Galoiskey<Scheme::CKKS>& galois_key);

            /** @brief Slot form back to rectangular form. Two levels. */
            RectActivation from_slots(std::vector<Ciphertext<Scheme::CKKS>>& in,
                                      int channels,
                                      Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief One group of a rectangular activation as a matrix
             *        encryption, ready for Algorithm 4.
             *
             * Batch slot b of the result is channel block b of the group, so
             * with head_dim = d it is head b. Three levels: down to slots, the
             * block transform, and back up.
             */
            BatchActivation to_batch(RectActivation& in, int group,
                                     Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Matrix encryptions back to one rectangular activation.
             *
             * The inverse of to_batch over every group at once, so the groups
             * have to arrive in channel order.
             */
            RectActivation from_batch(std::vector<BatchActivation>& groups,
                                      int channels,
                                      Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // The products
            // ---------------------------------------------------------------

            /**
             * @brief Y = X W, Kang's Algorithm 5.
             *
             * One call per (input group, output group) pair, and the partial
             * products of one output group are summed. Every term leaves the
             * same sequence of operations behind it, so they meet at one level
             * and one scale and the sum is a plain addition: widening a model
             * buys products and no depth at all.
             *
             * @param weight Row major @p in_channels x @p out_channels, the
             *               transpose of the mathematical weight. Held on the
             *               host as one copy.
             *
             * The uploaded plaintext is d * (N/2) * k coefficients per limb --
             * N^2/2 of them, regardless of how much of the weight is actually
             * non-zero. That is Algorithm 5's plaintext and not an inefficiency
             * here, but it does mean a narrow projection costs a wide one.
             */
            RectActivation project(RectActivation& x,
                                   const std::vector<double>& weight,
                                   int in_channels, int out_channels,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief A B with both operands encrypted, Kang's Algorithm 4.
             *
             * Both operands are matrix encryptions at exactly d columns, which
             * is what Algorithm 4's three internal CMTs assume, and the batch
             * axis carries the heads.
             */
            BatchActivation matmul(BatchActivation& a, BatchActivation& b,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key,
                                   Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // RMSNorm
            // ---------------------------------------------------------------

            /** @brief Shape and approximation settings for rms_norm. */
            struct RectRMSNormConfig
            {
                double eps = 1e-5;
                double sum_lo = 0.0; ///< Range of the summed square.
                double sum_hi = 0.0;
                int degree = 31;
                int newton_iterations = 2;
            };

            /**
             * @brief RMSNorm over the channel axis.
             *
             * The channel axis runs partly across ciphertexts and partly along
             * the fast slot axis, so the mean is a free slot-wise addition of
             * the groups' columns followed by ONE blocked reduction of span
             * k/2. That reduction costs the level a mask costs; the Algorithm 1
             * path pays nothing here because there a channel is a whole
             * ciphertext, and this is the price of putting k/2 channels in one.
             *
             * @param weight One learned scale per channel, or empty to leave the
             *               scaling out.
             */
            RectActivation rms_norm(RectActivation& x,
                                    const std::vector<double>& weight,
                                    const RectRMSNormConfig& config,
                                    Galoiskey<Scheme::CKKS>& galois_key,
                                    Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Attention
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one attention sublayer. */
            struct RectAttentionWeights
            {
                /// Row major in_channels x out_channels, the transpose of the
                /// mathematical weight, as project() wants it.
                std::vector<double> query;
                std::vector<double> key;
                std::vector<double> value;
                std::vector<double> output; ///< Empty skips W_o.
            };

            /** @brief Shape and approximation settings for attention. */
            struct RectAttentionConfig
            {
                int in_channels = 0;
                /// heads * d. head_dim is not a free parameter here: a head has
                /// to be exactly one channel block for Algorithm 4's
                /// contraction to be the head's own.
                int heads = 1;
                /// Distinct key and value heads, for grouped-query attention;
                /// 0 means one apiece. Must divide @c heads.
                ///
                /// GQA is expanded on the HOST: the key and value weights are
                /// widened to @c heads heads by repeating each kv head's
                /// columns. Homomorphically that is the only way to put a kv
                /// head in every batch slot that reads it, since the batch slots
                /// of a matrix encryption never talk to each other. It costs
                /// projection work in the ratio heads / kv_heads and buys back
                /// exactly nothing, which is worth knowing before choosing d.
                int kv_heads = 0;
                bool causal = true;
                /// 1/sqrt(d) if left at 0. Folded into the query weight on the
                /// host, where a scaling is free.
                double head_scale = 0.0;
                /// Subtracted from the scores so they land in [-bound, 0].
                /// Calibrated, as in the paper, not computed homomorphically.
                double score_shift = 0.0;
                /// bound, iterations and the degrees are the caller's; stride
                /// and count are fixed by this encoding and overwritten, because
                /// the key axis is entirely across ciphertexts.
                Llama3Operator::SoftmaxConfig softmax;
            };

            /**
             * @brief One attention sublayer.
             *
             * Projections through Algorithm 5, scores and the value product
             * through Algorithm 4, and the SoftMax in slot form behind the
             * bridge. A group carries k/2 heads and one Algorithm 4 call does
             * all of them, which is the batch CCMM used for what it is for.
             */
            RectActivation attention(RectActivation& x,
                                     const RectAttentionWeights& weights,
                                     const RectAttentionConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // SwiGLU
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one SwiGLU sublayer. */
            struct RectFeedForwardWeights
            {
                /// Row major in_channels x hidden_channels; down is the
                /// transpose shape.
                std::vector<double> gate;
                std::vector<double> up;
                std::vector<double> down;
            };

            /** @brief Shape and approximation settings for feed_forward. */
            struct RectFeedForwardConfig
            {
                int in_channels = 0;
                int hidden_channels = 0;
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;     ///< Section 3.1.3.
                /// Hidden GROUPS held at once; 0 takes the whole width.
                ///
                /// The Hadamard product needs both branches in both encodings at
                /// once, and the down projection sums over the hidden axis, so a
                /// chunk can be projected down and accumulated the moment it is
                /// formed and then released. Same products, same scales, only
                /// the association of a homomorphic sum changes.
                int hidden_block_groups = 0;
            };

            /** @brief W_down (SiLU(W_gate x) * W_up x). */
            RectActivation feed_forward(RectActivation& x,
                                        const RectFeedForwardWeights& weights,
                                        const RectFeedForwardConfig& config,
                                        Galoiskey<Scheme::CKKS>& galois_key,
                                        Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // The whole block
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one transformer block. */
            struct RectTransformerBlockWeights
            {
                std::vector<double> attention_norm;
                std::vector<double> feed_forward_norm;
                RectAttentionWeights attention;
                RectFeedForwardWeights feed_forward;
            };

            /** @brief Shape and approximation settings for one block. */
            struct RectTransformerBlockConfig
            {
                RectRMSNormConfig attention_norm;
                RectAttentionConfig attention;
                RectRMSNormConfig feed_forward_norm;
                RectFeedForwardConfig feed_forward;
            };

            /**
             * @brief One pre-norm transformer block.
             *
             * Norm, attention, residual; then norm, SwiGLU, residual. The
             * residual needs no encoding move: it reconciles level and scale
             * with a mod drop and a multiplication by a constant, and a constant
             * is the constant POLYNOMIAL, which scales every coefficient of a
             * rectangular encoding exactly as it scales every slot of a slot
             * encoding. The stream therefore stays rectangular from one end of
             * the block to the other and the only crossings are the ones the
             * non-linearities force.
             *
             * There is no refresh here. Whether CKKS bootstrapping carries a
             * rectangular encoding unchanged is untested -- it is the identity
             * on the plaintext polynomial, so it should, but "should" is not
             * "does" -- and a chain long enough to run a whole block on this
             * path does not fit alongside N/2 - 1 rotation keys anyway.
             */
            RectActivation
            transformer_block(RectActivation& x,
                              const RectTransformerBlockWeights& weights,
                              const RectTransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Relinkey<Scheme::CKKS>& relin_key);

          private:
            /// F and its inverse, as blocked slot diagonals: entry
            /// [eps + k/2 - 1] multiplies the input rotated by eps.
            void build_block_tables();

            /// The whole weight of one projection, arranged as Algorithm 5's
            /// plaintext wants it: block row t at Y^{k-t} with a minus sign.
            std::vector<int64_t>
            rectangular_weight(const std::vector<double>& weight,
                               int in_channels, int out_channels, int in_group,
                               int out_group, double scale) const;

            /// One group's columns borrowed as a BatchActivation, so a bridge
            /// call never deep copies a ciphertext just to name its operands.
            /// The columns are moved out of @p x and must be moved back.
            BatchActivation borrow_group(RectActivation& x, int group);
            void return_group(RectActivation& x, int group,
                              BatchActivation& borrowed);

            /// Refuse columns that have drifted apart in level or scale.
            void require_uniform(const RectActivation& in,
                                 const char* name) const;

            /// The prime the next rescale of @p ct will divide by.
            double rescale_prime(const Ciphertext<Scheme::CKKS>& ct) const;

            /// Encode @p values at @p scale, dropped onto @p depth.
            Plaintext<Scheme::CKKS> encode(const std::vector<Complex64>& values,
                                           double scale, int depth);

            HEContext<Scheme::CKKS> context_;
            HEEncoder<Scheme::CKKS> encoder_;
            /// Owns the row bridge, the batch matrix operator and the slot-form
            /// arithmetic. Everything Algorithm 4 needs is already in it.
            Llama3BatchOperator batch_;
            BatchMatrixLayout layout_;
            std::vector<Modulus64> primes_;
            int slot_count_;
            double default_scale_;

            /// [2][k - 1] slot vectors: [0] is F, [1] is F^-1, and the second
            /// index is the diagonal offset shifted by k/2 - 1.
            std::vector<std::vector<Complex64>> forward_block_;
            std::vector<std::vector<Complex64>> inverse_block_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_RECT_H
