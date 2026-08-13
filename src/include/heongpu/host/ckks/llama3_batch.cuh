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
#include <map>
#include <tuple>
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

            /**
             * @brief Q primes in the chain this operator was built on.
             *
             * The denominator of every level statement on this path: a
             * ciphertext at depth t has chain_limbs() - t active primes, and a
             * level budget is quoted in those rather than in depths, because a
             * depth is only meaningful against a chain.
             */
            int chain_limbs() const
            {
                return context_->get_ciphertext_modulus_count();
            }

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

            /** @brief Everything between the two Algorithm-4 products. */
            struct BatchSoftmaxSeamConfig
            {
                bool causal = true;

                /// Subtracted from the scores so they land in [-bound, 0],
                /// in RAW score units. If the scores arrive already carrying
                /// the exponential's domain map, this is scaled to match
                /// here -- the caller never has to remember to do it.
                double score_shift = 0.0;
                /// The same shift per QUERY row: d entries. A causal row of
                /// length u + 1 has a different range from a full one, so one
                /// number for the whole block is the loosest calibration
                /// available and this is the sharp one. Overrides
                /// @c score_shift.
                std::vector<double> score_shift_rows;
                /// The same shift per (input, query): slot_count entries, in
                /// the layout's own index b + (k/2) * u. Overrides both of
                /// the above.
                ///
                /// This exists only because the batch axis carries INDEPENDENT
                /// INPUTS here. On a path that spends the batch axis on heads
                /// or on channel blocks, a per-input calibration is not a slot
                /// vector and cannot be expressed at all.
                std::vector<double> score_shift_slots;

                /// The scores already carry the exponential's domain map,
                /// 2/bound, folded into whatever plaintext formed them.
                ///
                /// This is an ASSERTION ABOUT THE INPUT, not a request: set it
                /// and the seam skips the affine multiply that would otherwise
                /// cost a level, so setting it when the scores do NOT carry
                /// the map is silently wrong rather than an error. attention()
                /// owns the query weight and sets this itself; a caller
                /// driving the seam directly owns the fold.
                /// @see exp_domain_scale.
                bool scores_carry_exp_domain = false;

                /// Carry the reciprocal's domain map on the causal mask.
                /// Costs nothing -- the mask is a plaintext product that is
                /// paid for either way -- and saves one level per round. Needs
                /// a mask, so it needs @c causal, and it is incompatible with
                /// a Newton step, which wants its argument unmapped; with
                /// @c softmax.inverse_newton above zero it is a silent no-op,
                /// and softmax_seam_levels() reports the level it did not
                /// save.
                bool fold_affine_into_mask = true;

                /// Refresh the denominator before fitting its reciprocal --
                /// the paper's narrow auxiliary track.
                ///
                /// This is the largest single lever on the seam and it is
                /// almost free here: the denominator is ONE ciphertext however
                /// many parts the key axis is cut into, so the whole fit moves
                /// off the wide track for one bootstrap per round, while the
                /// d parts pay only the square and the normalising product.
                /// Needs the boot key at the call and no Newton step.
                bool refresh_denominator = false;

                /// Hoist the crossings' rotation trains. Bit-identical; what
                /// falls is decompositions (n1 + n2 - 2 -> n2 per column) and
                /// launches.
                bool hoisted_crossings = true;

                /// Encode each distinct causal mask once instead of once per
                /// head. @see causal_column_mask.
                bool cache_masks = true;
            };

            /**
             * @brief The domain map the exponential wants on its argument.
             *
             * Multiply it into whatever plaintext forms the scores -- the
             * query weight, where a scaling is free -- and set
             * @c scores_carry_exp_domain. That is one level of the deepest
             * stretch in the block, for nothing.
             */
            static double exp_domain_scale(double bound)
            {
                return Llama3Operator::domain_scale(-bound, 0.0);
            }

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
                /// Rotary position embedding on Q and K.
                ///
                /// OFF by default, and the default is not a preference: this
                /// path had no RoPE AT ALL until it was added, so every
                /// measurement taken before it was taken without positional
                /// information. Leaving it off keeps those reproducible;
                /// turning it on is what makes the sublayer Llama-3's.
                ///
                /// It costs THREE levels here and only one of them is the
                /// rotation. The angle varies with the TOKEN, which is the
                /// slow slot axis, so the map is an ordinary slot plaintext
                /// product -- but Q and K live in the coefficient encoding
                /// between the projection and the score product, so reaching
                /// slot form and returning costs a crossing each way. Under a
                /// slot-resident stream (LLAMA3_8B_LAYER_FLOW.md 22.6) Q and
                /// K are already in slot form when they are formed and RoPE
                /// costs its one level and nothing else.
                bool rope = false;
                /// Llama-3's base is 500000; Llama-2's is 10000.
                double rope_theta = 500000.0;
                /// Absolute position of token 0, for a windowed run.
                int rope_position_offset = 0;

                Llama3Operator::SoftmaxConfig softmax;
                /// Everything between the two Algorithm-4 products.
                /// @see BatchSoftmaxSeamConfig, softmax_seam.
                BatchSoftmaxSeamConfig seam;
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
             *
             * Note what this does NOT depend on: the head. The triangle is a
             * property of the query and key indices, and the batch axis here
             * carries independent INPUTS, all of which see the same causality.
             * So the d masks of a sublayer serve every one of its heads, which
             * is what makes them worth encoding once. @see
             * Llama3Operator::set_mask_plain_capacity.
             */
            std::vector<double> causal_column_mask(int key) const;

            /**
             * @brief The causal mask of one key column of one BLOCK PAIR.
             *
             * The generalisation of the above to a sequence longer than d
             * tokens, and the only piece of token blocking that is not
             * bookkeeping. A sequence of T tokens is T/d activations of d rows
             * each; query block @p query_block attends key blocks 0 ..
             * query_block, and the SoftMax reduces over all of them at once,
             * so the seam is handed (query_block + 1) * d parts and needs one
             * mask apiece.
             *
             * Three cases, and only the third has a triangle in it:
             *
             *   key_block >  query_block   never fed to the seam at all --
             *                              those keys are in the future and
             *                              the product is not computed
             *   key_block <  query_block   every query sees every key: the
             *                              mask is the row WEIGHT alone, with
             *                              no zeros
             *   key_block == query_block   the diagonal block, where the
             *                              triangle lives: query u keeps keys
             *                              0..u
             *
             * The weight is the same idea as the single-block one and reduces
             * to it exactly at query_block = 0. Query u of block p sees
             * p*d + u + 1 keys out of the (p+1)*d the seam reduces over, so
             * the weight is sqrt((p+1)*d / (p*d + u + 1)) -- constant along
             * the key axis, therefore cancelled exactly by the SoftMax rounds,
             * and chosen so the sum of squares lands where a full row of
             * (p+1)*d keys would leave it. Getting this wrong is not an error:
             * it is a reciprocal fitted over the wrong range, which is the
             * defect that cost this project a day on the other path.
             *
             * @param query_block Block of the query, in [0, T/d).
             * @param key_block   Block of the key, in [0, query_block].
             * @param key         Key inside its block, in [0, d).
             */
            std::vector<double> causal_column_mask(int query_block,
                                                   int key_block,
                                                   int key) const;

            /**
             * @brief The cache id of that mask, so the plaintext cache can
             *        name it.
             *
             * A query block uses exactly d + 1 distinct masks -- d triangles
             * on its diagonal block and one full-visibility mask for every
             * block below it -- and every head of that block reuses the same
             * d + 1. So a cache capacity of d + 1 gives a 100% hit rate across
             * heads, which is where the reuse actually is; capacity for a
             * whole sequence would be T/d times that and buys nothing, since a
             * query block is visited once.
             */
            int causal_mask_id(int query_block, int key_block, int key) const;

            /**
             * @brief Rotary position embedding, in place on slot-form Q or K.
             *
             * Cheap on this encoding for two reasons, both of them the same
             * fact: a channel is a whole ciphertext. The head-dim pairing
             * c <-> c + head_dim/2 is therefore a pairing of whole
             * CIPHERTEXTS -- no homomorphic rotation, no Galois key -- and the
             * angle (u + offset) * theta^(-2c/head_dim) depends on the TOKEN,
             * which is the slow slot axis, so one plaintext per lane pair
             * serves every instance and every head.
             *
             * Four plaintext products, two additions, ONE level. The minus
             * sign rides on the plaintext: negating homomorphically is a
             * plaintext product and a rescale, and the two halves of a pair
             * would then sit at different depths and could not be added.
             *
             * @param slots     heads * head_dim ciphertexts in the slot
             *                  reading, at one level and one scale.
             * @param head_dim  Channels in one head; must be even and divide
             *                  the column count.
             */
            void rope_slots(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                            int head_dim, double theta, int position_offset);

            // ---------------------------------------------------------------
            // The SoftMax seam, continued
            // ---------------------------------------------------------------
            //
            // The slot vector is N/2 = d * (k/2) wide and the score block of
            // one head fills it exactly: (k/2) independent inputs on the fast
            // axis, d queries on the slow one. There is no room left for the
            // key axis, so the key axis is the CIPHERTEXT axis -- not by
            // preference but by arithmetic, and the layout is therefore
            // determined rather than chosen.
            //
            // That forced layout is also the cheap one, which is the happy
            // part. The reduction the SoftMax needs runs along the key, so it
            // is a slot-wise addition of the d parts: `count = 1`, the loop in
            // sum_strided is dead code, and the denominator costs ZERO
            // rotations, zero levels and zero Galois indices. The alternative
            // -- key inside the ciphertext, query across them -- is reachable
            // (it is the transpose of the score block, and Algorithm 4 will
            // hand it over if you ask for K Q^T instead of Q K^T) but it costs
            // iterations * log2(d) rotations per part, de-amortises the
            // reciprocal d-fold, and needs a CMT to undo P^T before the value
            // product. It buys nothing back: the reduction it moves was
            // already free.
            //
            // Both seams are conversion-free, which is the other half of the
            // layout question. Algorithm 4 hands the scores over column-wise
            // with the KEY on the columns, which is the axis this reduces; and
            // from_slots hands P back column-wise, which is what Algorithm 4
            // wants as the left operand of P V, with V still carrying the key
            // on its rows where the projection left it. The only conversion is
            // the row bridge, one level each way, and it is unavoidable: a
            // ring product of two columns convolves over the row index, so no
            // entrywise polynomial -- no exponential, no square, no reciprocal
            // -- can be evaluated in matrix form at all.
            //
            // So the levers here are not the layout. They are the levels the
            // seam spends and the encodes it repeats, and both are below.

            /** @brief Where the SoftMax reduces, and what that costs. */
            struct SoftmaxLayout
            {
                int parts = 0;  ///< ciphertexts the key axis is cut into
                int count = 0;  ///< key positions inside ONE ciphertext
                int stride = 0;
                bool strided = true;
                /// Rotations the denominator costs. Zero, and that is the
                /// whole point of the layout.
                int reduction_rotations = 0;
                /// Rotations the two row bridges cost, both directions.
                int crossing_rotations = 0;
                /// Galois indices the seam needs beyond the ones Algorithm 4
                /// already forces. Zero.
                int new_galois_indices = 0;
            };

            /**
             * @brief The layout the seam runs in, for a d x d score block.
             *
             * Reports rather than decides: the values are read off the ring,
             * so a test can pin the invariant instead of restating it.
             */
            SoftmaxLayout softmax_layout() const;

            /**
             * @brief Levels the seam spends, crossings included.
             *
             * The closed form of the ledger, so a caller can size its chain
             * without running the circuit and a test can assert the ledger it
             * claims. Counts to_slots, the SoftMax and from_slots; not the two
             * Algorithm-4 products either side.
             */
            static int
            softmax_seam_levels(const Llama3Operator::SoftmaxConfig& softmax,
                                const BatchSoftmaxSeamConfig& seam);

            /**
             * @brief The SoftMax seam: scores in matrix form, P in matrix
             *        form.
             *
             * bridge to slots, shift, SoftMax, bridge back. @p scores must be
             * exactly layout.d columns -- the square block Algorithm 4 hands
             * back -- with the key on the columns and the query on the rows.
             * What comes back is the same shape, ready to be the LEFT operand
             * of the value product with no transpose.
             *
             * @param boot_key Needed only for @c refresh_denominator, which
             *                 throws without it rather than quietly putting
             *                 the fit's levels back on the wide track and
             *                 changing the schedule the caller sized its chain
             *                 for.
             */
            BatchActivation
            softmax_seam(BatchActivation& scores,
                         const Llama3Operator::SoftmaxConfig& softmax,
                         const BatchSoftmaxSeamConfig& seam,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key,
                         Galoiskey<Scheme::CKKS>* boot_key = nullptr);

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
                                      Relinkey<Scheme::CKKS>& relin_key,
                                      Galoiskey<Scheme::CKKS>* boot_key =
                                          nullptr);

            // ---------------------------------------------------------------
            // Token blocking: a sequence longer than d tokens
            // ---------------------------------------------------------------
            //
            // d is the sequence length on this path -- permanently, and by
            // arithmetic rather than by choice. A BatchActivation has exactly
            // layout.d rows, three places enforce it, and at batch 16 the ring
            // pins d = 128. Llama-3-8B's context is 8192, so everything above
            // ran at 1/64 of the model and there was NO code path to the rest
            // of it. These entry points are that path.
            //
            // WHAT IT COSTS AND WHAT IT DOES NOT
            //
            // The expensive-looking part is free and the free-looking part is
            // the expensive one, so it is worth being precise.
            //
            //   - The SoftMax costs NO new rotations. The key axis is already
            //     the ciphertext axis (d * (k/2) == N/2 leaves it nowhere else
            //     to be), so lengthening the key axis just means handing the
            //     reduction more parts. `count` stays 1, `sum_strided`'s loop
            //     stays dead, and the denominator over 8192 keys costs exactly
            //     what the denominator over 128 keys cost: zero. This is the
            //     one place this encoding is strictly better than every other
            //     one in the repo.
            //   - The non-linear layers cost nothing new either. RMSNorm and
            //     SwiGLU are token-wise, so a sequence is a loop over blocks
            //     with no interaction between them.
            //   - The products are the m^2, and there is no way around it:
            //     query block p needs a score product against every key block
            //     q <= p and a value product against each as well. A sequence
            //     of B blocks costs B(B+1)/2 of each per head instead of B --
            //     at B = 64 that is 2080 against 64, a 32.5x factor on the
            //     Algorithm-4 half. Causality is what halves it; a
            //     bidirectional model would pay B^2.
            //   - The memory is the other m: query block p holds (p+1)*d slot
            //     ciphertexts in the wide track at once, and the SoftMax
            //     squares them. At B = 64 the last query block alone is 8192
            //     ciphertexts. attention_sequence_peak_columns() reports it,
            //     because a caller that discovers this by OOM has learnt
            //     nothing.
            //
            // WHAT IS DELIBERATELY NOT HERE: a KV cache. Every key and value
            // block is recomputed from the sequence on every call, because
            // there is no incremental decode on this path -- the batch axis
            // carries sixteen independent PROMPTS, not sixteen positions of
            // one, so there is no autoregressive step to cache for.

            /**
             * @brief The SoftMax seam over a query block's whole visible past.
             *
             * @param scores    One square score block per key block, in key
             *                  order 0 .. query_block. Every one is d columns
             *                  at one level and one scale.
             * @param query_block Which query block these scores belong to.
             *                  Fixes the mask triangle and the row weights.
             *
             * @return One probability block per key block, in the same order,
             *         each ready to be the LEFT operand of its own value
             *         product with no transpose.
             *
             * The reduction runs across every part of every block at once --
             * that is what makes it the SoftMax of the whole row rather than
             * of one block -- so this cannot be decomposed into per-block
             * seams and the whole visible past is resident here.
             */
            std::vector<BatchActivation> softmax_seam_blocked(
                std::vector<BatchActivation>& scores, int query_block,
                const Llama3Operator::SoftmaxConfig& softmax,
                const BatchSoftmaxSeamConfig& seam,
                Galoiskey<Scheme::CKKS>& galois_key,
                Relinkey<Scheme::CKKS>& relin_key,
                Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            /**
             * @brief One attention sublayer over a sequence of token blocks.
             *
             * @param x One activation per token block, each d rows by
             *          config.in_channels columns, block t holding tokens
             *          [t*d, (t+1)*d) of every one of the k/2 instances. All
             *          blocks at one level and one scale.
             *
             * @return One output activation per token block, same shape.
             *
             * Identical arithmetic to attention() when there is one block --
             * asserted by test, not by inspection -- and the causal
             * generalisation of it when there are more. RoPE positions
             * continue across the blocks: block t starts at
             * t*d + config.rope_position_offset, so the sequence is one
             * sequence and not B independent ones.
             */
            std::vector<BatchActivation>
            attention_sequence(std::vector<BatchActivation>& x,
                               const BatchAttentionWeights& weights,
                               const BatchAttentionConfig& config,
                               Galoiskey<Scheme::CKKS>& galois_key,
                               Relinkey<Scheme::CKKS>& relin_key,
                               Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            /** @brief What a B-block causal sublayer costs, in products. */
            struct SequenceProductCount
            {
                /// Algorithm-4 score products, B(B+1)/2 per head per block
                /// pair.
                long long score = 0;
                /// Algorithm-4 value products, the same count.
                long long value = 0;
                /// Bridged columns, both directions, over the whole sublayer.
                long long bridged_columns = 0;
                /// Slot-form ciphertexts resident at the deepest query block.
                long long peak_slot_columns = 0;
            };

            /**
             * @brief That count, from the shape alone.
             *
             * Reports rather than decides, so a test can pin the m^2 and a
             * caller can size a card before it runs anything.
             */
            SequenceProductCount
            sequence_product_count(int blocks,
                                   const BatchAttentionConfig& config) const;

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

            /// The d causal masks as slot vectors, built on first use. They
            /// depend on d and k alone -- not on the head, not on the level,
            /// not on the data -- so a sublayer that rebuilt them per head was
            /// writing d * slot_count doubles H times over. The whole table is
            /// 2 MiB at the Llama-3 shape.
            const std::vector<std::vector<double>>& causal_masks();
            std::vector<std::vector<double>> causal_mask_cache_;

            /// The d + 1 masks of query block @p query_block: d triangles for
            /// its own diagonal, then one full-visibility mask at index d for
            /// every key block below it. Keyed by query block because the row
            /// weight sqrt((p+1)*d / (p*d + u + 1)) depends on it; the single
            /// block case is query_block = 0 and is served by the table above,
            /// so an existing caller shares the plaintexts it already had.
            const std::vector<std::vector<double>>&
            causal_masks_blocked(int query_block);
            std::map<int, std::vector<std::vector<double>>>
                causal_block_mask_cache_;

            /// One direction of the bridge; @p inverse picks V^-1 over V.
            std::vector<Ciphertext<Scheme::CKKS>>
            bridge(std::vector<Ciphertext<Scheme::CKKS>>& in, bool inverse,
                   const char* name, Galoiskey<Scheme::CKKS>& galois_key);


            /// Refuse columns that have drifted apart in level or scale.
            void require_uniform(const BatchActivation& in,
                                 const char* name) const;

            /// Algorithm 4 on borrowed columns, so a block product never has
            /// to copy a ciphertext just to name its operands.
            ///
            /// @param b_form Whether @p b still owes Algorithm 4's step-1
            ///               transpose. Handing over an operand that is
            ///               already row-wise computes A * B^T for the B that
            ///               operand encrypts column-wise -- which is what
            ///               the score product wants, and is how it stops
            ///               transposing K twice for nothing.
            std::vector<Ciphertext<Scheme::CKKS>>
            product(const std::vector<Ciphertext<Scheme::CKKS>*>& a,
                    const std::vector<Ciphertext<Scheme::CKKS>*>& b,
                    const char* name, Galoiskey<Scheme::CKKS>& galois_key,
                    Relinkey<Scheme::CKKS>& relin_key,
                    HEBatchMatrixOperator<Scheme::CKKS>::RightOperandForm
                        b_form = HEBatchMatrixOperator<
                            Scheme::CKKS>::RightOperandForm::column_wise);

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

            /// The same diagonals ENCODED, keyed by what an encoding depends
            /// on: direction, the level it was dropped to, and the prime the
            /// rescale after it will divide by.
            ///
            /// A bridge call converts every column of an activation, and
            /// require_uniform has already established that they share a level
            /// and a scale -- so all d columns want the same d plaintexts and
            /// the uncached code encoded them d times over. That is d^2 full
            /// encodes per crossing, each one an NTT over the whole live chain,
            /// and at d = 64 it measured 20.1 s of a 26.6 s QK stage: more than
            /// the CMT, Algorithm 4 and the SoftMax put together. The arithmetic
            /// is unchanged; only the number of times it is performed is.
            ///
            /// Bounded, because a plaintext is Q_size * N * 8 bytes per limb
            /// and d of them is not small. The key that matters is the one the
            /// current call uses, so a handful of entries holds every level a
            /// block actually visits.
            /// One cached set: the encoded plaintexts, and -- once the
            /// hoisted path has touched the entry -- the same set packed
            /// contiguously at its level's limb count, which is the layout
            /// the fused multiply-accumulate launch walks.
            struct EncodedDiagonalSet
            {
                std::vector<Plaintext<Scheme::CKKS>> plains;
                DeviceVector<Data64> packed;
            };
            std::map<std::tuple<bool, int, uint64_t>, EncodedDiagonalSet>
                bridge_plain_;

            /// Encoded diagonal sets kept at once, over all levels and both
            /// directions. Eviction is whole-set and the set is rebuilt on the
            /// next call, so this trades memory for encodes and never for
            /// correctness.
            std::size_t bridge_plain_capacity_ = 4;

            /// @see set_bridge_plain_limb_limit. 0 means no limit.
            int bridge_plain_limb_limit_ = 0;

            /// Live limbs of the set behind a cache key, which is what its
            /// memory is proportional to: the key carries the depth and the
            /// context carries the chain length.
            int set_limbs(
                const std::tuple<bool, int, uint64_t>& key) const
            {
                return context_->get_ciphertext_modulus_count() -
                       std::get<1>(key);
            }

            /// Drop the WIDEST resident set -- the one whose memory is
            /// largest. The map is ordered by (direction, depth), so its
            /// first element is an arbitrary victim rather than a cheap one,
            /// and evicting arbitrarily is how a 2-limb set that seven calls
            /// want gets displaced by a 7-limb set that four do.
            void evict_widest()
            {
                if (bridge_plain_.empty())
                {
                    return;
                }
                auto victim = bridge_plain_.begin();
                int widest = set_limbs(victim->first);
                for (auto it = std::next(victim); it != bridge_plain_.end();
                     ++it)
                {
                    const int limbs = set_limbs(it->first);
                    if (limbs > widest)
                    {
                        widest = limbs;
                        victim = it;
                    }
                }
                bridge_plain_.erase(victim);
            }

            /// Baby steps for the bridge's BSGS split; 0 takes sqrt(d). One
            /// forces the whole split onto the giant side, which is the
            /// d - 1 rotations the bridge used to take and is here so the two
            /// can be measured against each other rather than argued about.
            /// Must divide d. Changes cost, never the result.
            int bridge_baby_steps_ = 0;

            /// @see set_hoisted_crossings.
            bool hoisted_crossings_ = false;

          public:
            /// The bridge's diagonal tables, read-only. The rectangular
            /// operator composes them with its block transform to build the
            /// one-level fused crossings, and building them twice from the ring
            /// would only be a second chance to disagree about a convention.
            /// Entry delta multiplies the input rotated by delta * (k/2).
            const std::vector<std::vector<Complex64>>&
            bridge_forward_diagonals() const
            {
                return forward_diagonal_;
            }
            const std::vector<std::vector<Complex64>>&
            bridge_inverse_diagonals() const
            {
                return inverse_diagonal_;
            }

            /// Baby steps the bridge splits its d diagonals into, n1, with
            /// n2 = d / n1 giant steps; the crossing costs n1 + n2 - 2 key
            /// switches per column instead of d - 1.
            int baby_steps() const;

            /// @see bridge_baby_steps_. Set before a crossing; a change
            /// invalidates the encoded diagonals, which are rebuilt on demand.
            void set_bridge_baby_steps(int n1)
            {
                if (n1 < 0 || (n1 > 0 && layout_.d % n1 != 0))
                {
                    throw std::invalid_argument(
                        "The baby step count must divide d, because the two "
                        "index sets have to cover the d diagonals exactly");
                }
                bridge_baby_steps_ = n1;
                bridge_plain_.clear();
            }

            /// Encoded diagonal sets kept at once. @see bridge_plain_. A set
            /// is d plaintexts of N words per live limb, so at a big ring it
            /// is the largest single allocation the bridge makes: 4.3 GiB at
            /// (N = 65536, d = 1024, one limb), linear in the level. Holding
            /// both directions resident is what exhausts a device pool
            /// there; one costs a re-encode per direction change and never
            /// changes a result.
            void set_bridge_plain_capacity(std::size_t sets)
            {
                if (sets == 0)
                {
                    throw std::invalid_argument(
                        "The bridge needs room for at least one diagonal set");
                }
                bridge_plain_capacity_ = sets;
                while (bridge_plain_.size() > bridge_plain_capacity_)
                {
                    evict_widest();
                }
            }

            /**
             * @brief Refuse to CACHE a diagonal set wider than @p limbs.
             *
             * A set costs d plaintexts of N words per live limb, so its size
             * is linear in the level while its VALUE is how often that
             * (direction, level) pair comes back. Those two are unrelated,
             * and on a two-ring block they point opposite ways: the
             * to_slots set sits at 2 limbs and seven calls want it, while a
             * from_slots set at 7 limbs is three and a half times the memory
             * for four calls. A capacity counted in SETS lets the second
             * evict the first, which is the worst of both.
             *
             * With a limit set, a wider set is still built and used -- it is
             * simply not kept, so the narrow set it would have displaced
             * survives. 0, the default, means no limit.
             */
            void set_bridge_plain_limb_limit(int limbs)
            {
                bridge_plain_limb_limit_ = limbs;
                if (limbs <= 0)
                {
                    return;
                }
                for (auto it = bridge_plain_.begin();
                     it != bridge_plain_.end();)
                {
                    it = (set_limbs(it->first) > limbs)
                             ? bridge_plain_.erase(it)
                             : std::next(it);
                }
            }
            int bridge_plain_limb_limit() const
            {
                return bridge_plain_limb_limit_;
            }

            /**
             * @brief Hoist the bridge's rotation trains.
             *
             * The n1 baby shifts of a bridge call all read ONE source, and a
             * key switch spends most of its time decomposing its input,
             * which does not depend on the shift. With this on, the babies
             * share one decomposition (hoisted_rotation_train) and each
             * giant group's n1 plaintext products and additions collapse
             * into one fused launch (hoisted_bsgs_group_sum). The modular
             * arithmetic is unchanged to the bit; what falls is the work and
             * the launch count. OFF by default so existing measurements stay
             * reproducible.
             */
            void set_hoisted_crossings(bool on) { hoisted_crossings_ = on; }
            bool hoisted_crossings() const { return hoisted_crossings_; }
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_BATCH_H
