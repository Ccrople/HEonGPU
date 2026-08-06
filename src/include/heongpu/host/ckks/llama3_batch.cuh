// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The Llama-3 primitives carried over to the batch matrix multiplication of
// Cheon, Kang and Lee, "Fast Batch Matrix Multiplication in Ciphertexts".
//
// WHY THIS FILE EXISTS
// --------------------
// llama3.cuh holds the model on the Section 4.2 BSGS product of Park et al.,
// where every projection is a plaintext-ciphertext product costing
// giant + baby - 2 rotations, and every rotation is a key switch. The profile
// of a forward pass says key switching is the pass, and the rotation keys are
// what puts the published packing out of reach of a single card.
//
// Kang's Algorithm 1 does the same projection with NO rotations and NO key
// switching at all: it is two matrix products over R_{q,k}, one per ciphertext
// component. That is the whole reason for this file.
//
// WHAT IT COSTS TO GET THERE
// --------------------------
// The two algorithms do not share an encoding, and this is not a detail that
// can be papered over.
//
//   slot form    one ciphertext holds a whole d x d matrix, entry (r, c) of
//                matrix m at slot (r * d + c) * batch + m. This is what every
//                non-linear layer needs, because SiLU, the SoftMax exponential
//                and the Newton steps are all slot-wise polynomial evaluation,
//                and it is what bootstrapping consumes and returns.
//
//   matrix       a d x C matrix is C ciphertexts, one per column, in the
//   encryption   COEFFICIENT domain: column j is
//                m_j = sum_i M[i][j] X^i with M[i][j] in R_k = Z[Y]/(Y^k + 1),
//                Y = X^d, so entry (i, j)'s t-th subring coefficient sits at
//                coefficient i + d * t of ciphertext j. This is what Kang's
//                algorithms consume and return.
//
// Neither is a reinterpretation of the other. Evaluating a column at the slot
// points gives
//
//     slot_s(m_j) = sum_{i<d} psi^{i * 5^s} * Mb[s mod (k/2)][i][j],        (*)
//
// with psi a primitive 2N-th root: every slot is a Vandermonde combination of
// the d rows, so recovering a row is a linear transform and not a relabelling.
// That transform is the bridge, and it is the price of the swap.
//
// WHY THE BRIDGE IS AFFORDABLE ANYWAY
// -----------------------------------
// The two forms carry different amounts of work per ciphertext, and comparing
// them per ciphertext is the mistake that makes the bridge look ruinous. A
// matrix encryption batches k/2 = N/(2d) independent instances; the slot form
// batches N/(2d^2), a factor d fewer. Counted per instance, for one
// C_in -> C_out projection:
//
//     slot path    4 * C_in * C_out * sqrt(d) / N key switches
//     Kang         0
//     one bridge   2 * d^2 * C / N rotations
//
// At the shape this branch targets -- d = 64, N = 8192, C = 4096 -- that is
// 65,536 key switches per projection against 4,096 per bridge. A transformer
// block runs seven projections and needs about eight bridges, so the swap is
// worth roughly fourteen times fewer key switches even before the rotation
// keys PCMM no longer needs at all are counted.
//
// ORIENTATION
// -----------
// Kang's PCMM computes M(encrypted) * U(plaintext): the encrypted operand is
// on the LEFT. A Llama projection is W(plaintext) * X(encrypted), the other
// way round. Rather than transpose homomorphically -- which would hand back
// the key switches the swap just saved -- the activation is held as
// X[token][channel] and the weights are stored transposed on the host, where a
// transpose is free. So:
//
//     rows    d, the tokens of one block. Fixed by the ring, since d is the
//             rank of R_N over R_k.
//     cols    channels. NOT constrained by the ring, which is the other half
//             of the win: the channel axis no longer has to be cut into
//             d-sized blocks, it is simply the number of ciphertexts.
//     batch   k/2 independent instances, carried for free.
//
// The one thing that does stay blocked is the plaintext weight. A C_in x C_out
// weight uploads as C_in * C_out * k coefficients per limb, so a full
// 4096 x 4096 projection in a single call would be tens of gigabytes of
// twiddled plaintext. project() therefore cuts the output axis into
// column_block-sized pieces and streams them, which changes no arithmetic.

#ifndef HEONGPU_CKKS_LLAMA3_BATCH_H
#define HEONGPU_CKKS_LLAMA3_BATCH_H

#include <heongpu/host/ckks/batchmatrix.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/decryptor.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/encryptor.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/llama3.cuh>
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
         * @brief An activation in matrix-encryption form.
         *
         * @c column[j] encrypts column j of a @c rows x @c column.size() matrix
         * over R_k, in the coefficient domain of Definition 2. The rows are
         * tokens and the columns are channels, so widening a model appends
         * ciphertexts and never touches the ring.
         *
         * Every column is at one level and one scale; the operations here
         * refuse a set that has drifted, because they add columns together and
         * CKKS addition is only meaningful between equal scales.
         */
        struct BatchActivation
        {
            std::vector<Ciphertext<Scheme::CKKS>> column;

            /// Rows of the encrypted matrix: layout.d, and therefore the token
            /// count of one block.
            int rows = 0;

            int columns() const { return static_cast<int>(column.size()); }
            bool empty() const { return column.empty(); }
        };

        /**
         * @brief The Llama-3 linear algebra on Kang's Algorithms 1 and 4.
         *
         * Holds the batch matrix operator and the arithmetic operator the
         * relinearisation inside CCMM needs. The non-linear layers are not
         * reimplemented here: they stay on Llama3Operator in slot form, and
         * to_slots/from_slots move between the two.
         */
        class Llama3BatchOperator
        {
          public:
            /**
             * @param layout Subring layout. @c layout.d is the token block
             *               size and fixes the batch at k/2 = N/(2d).
             * @param scale  Default scaling factor, shared with the slot-form
             *               operator so a bridged ciphertext needs no rescale
             *               purely to change representation.
             */
            Llama3BatchOperator(HEContext<Scheme::CKKS> context,
                                HEEncoder<Scheme::CKKS>& encoder,
                                const BatchMatrixLayout& layout, double scale);

            const BatchMatrixLayout& layout() const noexcept
            {
                return layout_;
            }

            /** @brief Independent instances one activation carries, k/2. */
            int batch() const noexcept { return layout_.batch; }

            /** @brief Default scaling factor. */
            double default_scale() const noexcept { return default_scale_; }

            // ---------------------------------------------------------------
            // Host-side staging, for tests and for the client side
            // ---------------------------------------------------------------

            /**
             * @brief Encode and encrypt a batch of d x cols real matrices.
             *
             * @param batch Exactly k/2 matrices, each rows * cols row-major.
             */
            BatchActivation
            encrypt(const std::vector<std::vector<double>>& batch, int rows,
                    int cols, HEEncryptor<Scheme::CKKS>& encryptor,
                    double scale);

            /** @brief Inverse of encrypt, for checking a result. */
            std::vector<std::vector<double>>
            decrypt(BatchActivation& in, HEDecryptor<Scheme::CKKS>& decryptor,
                    double scale);

            // ---------------------------------------------------------------
            // The bridge
            // ---------------------------------------------------------------

            /**
             * @brief Rotations the bridge needs a Galois key for.
             *
             * Both directions mix only slots that share a batch index, and
             * those sit d apart in steps of k/2, so the shifts are the d - 1
             * multiples of k/2 and their negatives. This is a much shorter
             * list than the slot path's projections ask for, and it does not
             * grow with the width of the model.
             */
            std::vector<int> bridge_rotation_indices() const;

            /**
             * @brief Matrix-encryption form to slot form.
             *
             * Inverts the Vandermonde of (*) within each batch index. Slot
             * b + (k/2) * u of the result holds entry (u, j) of matrix b, so
             * one column ciphertext becomes one slot ciphertext holding that
             * channel across every token and every instance -- which is
             * exactly the layout the strided reductions of RMSNorm and the
             * SoftMax want, with the token index on the slow axis.
             *
             * Costs d - 1 rotations, d plaintext multiplications and one
             * level, per column.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            to_slots(BatchActivation& in,
                     Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Slot form back to matrix-encryption form.
             *
             * The forward Vandermonde, and the exact inverse of to_slots up to
             * the level each spends.
             */
            BatchActivation
            from_slots(std::vector<Ciphertext<Scheme::CKKS>>& in, int rows,
                       Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // The products
            // ---------------------------------------------------------------

            /**
             * @brief Y = X * W, Kang's Algorithm 1. No rotations, no key
             *        switching.
             *
             * @param weight Row-major @c in_channels x @c out_channels, the
             *               transpose of the mathematical weight, shared by
             *               the batch. Held on the host as one copy.
             * @param column_block Output channels per upload; 0 takes a
             *               default that keeps the twiddled plaintext to a
             *               sane size. Changes cost, never the result.
             */
            BatchActivation project(BatchActivation& x,
                                    const std::vector<double>& weight,
                                    int in_channels, int out_channels,
                                    const char* name, int column_block = 0);

            /**
             * @brief A * B with both operands encrypted, Kang's Algorithm 4.
             *
             * Used for the attention scores and the value product, the two
             * places a Llama block multiplies two things it computed itself.
             * Costs three CMTs, four matrix products over R_{q,k} and one
             * relinearisation per output column.
             *
             * @param galois_key Must carry
             *                   get_batch_cmt_rotation_indices(layout()).
             */
            BatchActivation matmul(BatchActivation& a, BatchActivation& b,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key,
                                   Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief Transpose every matrix of a matrix encryption, Kang's
             *        Algorithm 3 (CMT).
             *
             * Exactly @c layout.d columns, because a transpose is only defined
             * on the square block the ring fixes. Attention needs it once, for
             * K, since matmul forms A B and the scores are Q K^T.
             */
            BatchActivation transpose(BatchActivation& in, const char* name,
                                      Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Rotations the CMT inside transpose and matmul needs.
             *
             * This is get_batch_cmt_rotation_indices(layout()), and it is the
             * same set as bridge_rotation_indices(): the CMT applies the
             * automorphisms X -> X^(2kt+1), whose subgroup is generated by
             * 5^(k/2), so both lists are the multiples of k/2. The bridge
             * therefore asks for no key material CCMM did not already need.
             */
            std::vector<int> product_rotation_indices() const;

            /** @brief Every rotation index this operator can ask for. */
            std::vector<int> rotation_indices() const;

            /**
             * @brief The batch matrix operator behind this one.
             *
             * Exposed so a caller running a different algorithm of the same
             * paper -- Algorithm 5, say -- can reuse this operator's twiddle
             * caches and its bridge rather than construct a second copy of
             * everything. It is the same object the products below use.
             */
            HEBatchMatrixOperator<Scheme::CKKS>& matrix() noexcept
            {
                return matrix_;
            }

            /** @brief The slot-form operator the non-linearities run on. */
            Llama3Operator& arith() noexcept { return arith_; }

            // ---------------------------------------------------------------
            // Attention
            // ---------------------------------------------------------------
            //
            // The score matrix is where this encoding pays a second time, and
            // the reason is worth stating.
            //
            // Q is d tokens by head_dim channels and so is K, so S = Q K^T is
            // d by d with the QUERY on the rows and the KEY on the columns. A
            // matrix encryption puts a column in its own ciphertext, so the
            // SoftMax axis -- the key -- runs across ciphertexts. Its reduction
            // is then a slot-wise addition of d ciphertexts: no rotation, no
            // mask, no level. The slot path forms K^T Q instead of Q^T K
            // precisely to buy the cheaper of its two reductions; here the
            // cheaper one is free.
            //
            // Widening a head costs nothing in depth either. A head spanning
            // several channel blocks has S = sum_t Q[t] K[t]^T, and every term
            // is one CCMM at the same level and scale, so the sum is free.
            // Both operands of every product are square at d, which is what
            // Algorithm 4 requires, and that holds for any head_dim that is a
            // multiple of d.

            /** @brief Plaintext weights of one attention sublayer. */
            struct BatchAttentionWeights
            {
                /// Row-major in_channels x out_channels, the transpose of the
                /// mathematical weight, as project() wants it.
                std::vector<double> query;
                std::vector<double> key;
                std::vector<double> value;
                std::vector<double> output; ///< Empty skips W_o.
            };

            /** @brief Shape and approximation settings for attention. */
            struct BatchAttentionConfig
            {
                int in_channels = 0;
                /// heads * head_dim. Must be a multiple of layout.d.
                int q_channels = 0;
                /// kv_heads * head_dim, and the same head_dim as the query.
                int kv_channels = 0;
                int heads = 1;
                /// Distinct key and value heads, for grouped-query attention;
                /// 0 means one apiece. Must divide @c heads.
                int kv_heads = 0;
                bool causal = true;
                /// 1/sqrt(head_dim) if left at 0. Folded into the query weight
                /// on the host, where a scaling is free, rather than paid for
                /// as a homomorphic level.
                double head_scale = 0.0;
                /// Subtracted from the scores so they land in [-bound, 0].
                /// Calibrated, as in the paper, not computed homomorphically.
                double score_shift = 0.0;
                /// bound, iterations and the degrees are the caller's. stride
                /// and count are fixed by this encoding and overwritten: the
                /// key axis is entirely across ciphertexts, so count is one.
                Llama3Operator::SoftmaxConfig softmax;
            };

            /**
             * @brief The causal mask of score column @p key, in slot form.
             *
             * Column @p key is visible to query u only when key <= u, and the
             * surviving weight is sqrt(d / (u + 1)): constant along the key
             * axis for a given query, so the SoftMax rounds cancel it exactly,
             * while leaving the sum of squares in the range a full row would
             * produce. Without it the first reciprocal would have to be fitted
             * over a range that widens with every position masked off.
             */
            std::vector<double> causal_column_mask(int key) const;

            /**
             * @brief One attention sublayer on the batch path.
             *
             * Projections through Algorithm 1, scores and the value product
             * through Algorithm 4, and the SoftMax in slot form behind a pair
             * of bridges -- which is the only place this sublayer leaves the
             * matrix encoding at all.
             */
            BatchActivation attention(BatchActivation& x,
                                      const BatchAttentionWeights& weights,
                                      const BatchAttentionConfig& config,
                                      Galoiskey<Scheme::CKKS>& galois_key,
                                      Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // RMSNorm and SwiGLU
            // ---------------------------------------------------------------
            //
            // Both are slot-wise, so both live behind the bridge, and both
            // reduce over the channel axis -- which a matrix encryption puts
            // across ciphertexts, exactly as it does the SoftMax's key axis.
            // The reduction is a slot-wise addition and costs no rotation.
            //
            // SwiGLU is the one place this encoding is at a disadvantage, and
            // it is worth naming: SiLU(W_g x) * (W_u x) is a HADAMARD product,
            // and a matrix encryption has no such operation -- multiplying two
            // columns convolves them. So both branches cross to slot form and
            // the result crosses back, three bridges over the hidden width
            // rather than one. It is still the cheaper side of the trade by an
            // order of magnitude, because the three projections it replaces
            // were the widest key switching in the block.

            /** @brief Shape and approximation settings for rms_norm. */
            struct BatchRMSNormConfig
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
             * @param weight One learned scale per channel, or empty to leave
             *               the scaling out. A channel is a whole ciphertext
             *               here, so its weight is a constant and not a slot
             *               vector.
             */
            BatchActivation rms_norm(BatchActivation& x,
                                     const std::vector<double>& weight,
                                     const BatchRMSNormConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Plaintext weights of one SwiGLU sublayer. */
            struct BatchFeedForwardWeights
            {
                /// Row-major in_channels x hidden_channels, as project() wants
                /// them; down is hidden_channels x in_channels.
                std::vector<double> gate;
                std::vector<double> up;
                std::vector<double> down;
            };

            /** @brief Shape and approximation settings for feed_forward. */
            struct BatchFeedForwardConfig
            {
                int in_channels = 0;
                int hidden_channels = 0;
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;     ///< Section 3.1.3.
                /// Hidden channels held at once; 0 takes the whole width.
                ///
                /// This is the memory lever on the sublayer and, at Llama-3
                /// widths, the difference between fitting on one card and not.
                /// A matrix encryption spends one ciphertext per channel, and
                /// the Hadamard product needs the gate and the up projection
                /// in BOTH forms at once, so holding the whole hidden width
                /// costs 4 * hidden_channels ciphertexts -- 57,344 of them at
                /// the published 14336, which is hundreds of gigabytes at any
                /// useful chain length.
                ///
                /// The down projection sums over the hidden axis, and a sum
                /// splits over disjoint ranges of its index, so a chunk can be
                /// projected down and accumulated the moment it is formed and
                /// then released. The arithmetic is unchanged -- same products,
                /// same scales, same levels, only the order of a homomorphic
                /// sum -- and the peak falls to 4 * hidden_block.
                int hidden_block = 0;
            };

            /** @brief The SwiGLU sublayer, W_down (SiLU(W_gate x) * W_up x). */
            BatchActivation
            feed_forward(BatchActivation& x,
                         const BatchFeedForwardWeights& weights,
                         const BatchFeedForwardConfig& config,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Plaintext weights of one whole transformer block. */
            struct BatchTransformerBlockWeights
            {
                /// One learned scale per channel, or empty to leave the
                /// scaling out.
                std::vector<double> attention_norm;
                std::vector<double> feed_forward_norm;
                BatchAttentionWeights attention;
                BatchFeedForwardWeights feed_forward;
            };

            /** @brief Shape and approximation settings for a whole block. */
            struct BatchTransformerBlockConfig
            {
                BatchRMSNormConfig attention_norm;
                BatchAttentionConfig attention;
                BatchRMSNormConfig feed_forward_norm;
                BatchFeedForwardConfig feed_forward;
            };

            /**
             * @brief One pre-norm transformer block on the batch path.
             *
             * Norm, attention, residual; then norm, SwiGLU, residual. The
             * residual addition needs no bridge: it reconciles level and scale
             * with a mod drop and a multiplication by a constant, and a
             * constant is the constant POLYNOMIAL, which scales every
             * coefficient of a matrix encryption exactly as it scales every
             * slot of a slot encoding. The stream therefore stays in matrix
             * form from one end of the block to the other, and the only
             * crossings are the ones the non-linearities force.
             *
             * There is no refresh here yet, and that is the open question on
             * this path rather than an omission. CKKS bootstrapping is the
             * identity on the plaintext polynomial, so it should carry a
             * matrix encryption unchanged, but "should" is not "does" and
             * nothing here has measured it. A caller wanting a stack must
             * bootstrap between blocks itself, and check what comes back.
             */
            BatchActivation
            transformer_block(BatchActivation& x,
                              const BatchTransformerBlockWeights& weights,
                              const BatchTransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Relinkey<Scheme::CKKS>& relin_key);

          private:
            /// The d x d Vandermonde of (*) at batch index b, and its inverse.
            /// Built once per operator: it depends only on the ring.
            void build_bridge_tables();

            /// One direction of the bridge; @p inverse picks V^-1 over V.
            std::vector<Ciphertext<Scheme::CKKS>>
            bridge(std::vector<Ciphertext<Scheme::CKKS>>& in, bool inverse,
                   const char* name, Galoiskey<Scheme::CKKS>& galois_key);

            /// Refuse columns that have drifted apart in level or scale.
            void require_uniform(const BatchActivation& in,
                                 const char* name) const;

            /// Algorithm 4 on borrowed columns, so a block product never has
            /// to copy a ciphertext just to name its operands.
            std::vector<Ciphertext<Scheme::CKKS>>
            product(const std::vector<Ciphertext<Scheme::CKKS>*>& a,
                    const std::vector<Ciphertext<Scheme::CKKS>*>& b,
                    const char* name, Galoiskey<Scheme::CKKS>& galois_key,
                    Relinkey<Scheme::CKKS>& relin_key);

            /// The prime the next rescale of @p ct will divide by.
            double rescale_prime(const Ciphertext<Scheme::CKKS>& ct) const;

            /// Encode @p values at @p scale, dropped onto @p depth. The
            /// encoder only ever encodes at the top of the chain, so a
            /// plaintext meeting a ciphertext further down has to come to it.
            Plaintext<Scheme::CKKS> encode(const std::vector<Complex64>& values,
                                           double scale, int depth);

            HEContext<Scheme::CKKS> context_;
            HEEncoder<Scheme::CKKS> encoder_;
            /// The slot-form operator, not merely an arithmetic one: the
            /// SoftMax behind the bridge is Llama3Operator's, so the two paths
            /// share one implementation of every approximation rather than
            /// growing a second.
            Llama3Operator arith_;
            HEBatchMatrixOperator<Scheme::CKKS> matrix_;
            BatchMatrixEncoder batch_encoder_;
            BatchMatrixLayout layout_;
            /// Cached: the context hands out its modulus chain by value.
            std::vector<Modulus64> primes_;
            int slot_count_;
            double default_scale_;

            /// The bridge transform as d slot vectors, one per shift: entry
            /// delta multiplies the input rotated by delta * (k/2). The
            /// Vandermonde depends on the batch index, so a diagonal is a full
            /// slot vector rather than a constant, and it is complex, because
            /// the nodes psi^{i 5^s} are.
            std::vector<std::vector<Complex64>> forward_diagonal_;
            std::vector<std::vector<Complex64>> inverse_diagonal_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_BATCH_H
