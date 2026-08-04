// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The homomorphic primitives a Llama-3 layer is built from, following Park,
// Park, Park, Ahn, Cheon, Hanrot, Kim, Park and Stehle, "Scaling up FHE-based
// Privacy-Preserving ML: Higher Throughput, Longer Inputs for LLama-3-8B".
//
// Every operator here works on slot-encoded CKKS ciphertexts and is meant to
// be composed later into the attention and feed-forward layers; nothing in
// this file assumes a particular layer, only a particular slot layout.
//
// SLOT LAYOUT
// -----------
// Two layouts appear, and each reduction primitive states which one it wants.
//
//   strided  slot(j, i) = j * stride + i, with stride * count == slot_count.
//            The reduced axis j is the SLOW axis and the independent instances
//            i are the FAST axis. A slot rotation by stride * t then acts as a
//            cyclic shift of j alone, because the wraparound of the whole slot
//            vector is exactly the wraparound of j. Reductions along it are
//            therefore exact, need log2(count) rotations and consume no level.
//            This is the layout of Section 3.2, where the token index varies
//            fastest, so a sum over channels is strided.
//
//   blocked  slot(b, j) = b * span + j, the reduced axis j contiguous. A slot
//            rotation mixes neighbouring blocks, so a reduction here costs an
//            extra masking multiplication, hence one level. This is the layout
//            a SoftMax over the token axis lands in.
//
// ROTATION KEYS
// -------------
// Every routine that rotates publishes the exact shifts it needs, and a caller
// must build the Galois key from those lists. This is not an optimisation.
// Galoiskey(context, shift_vec) stores exactly the shifts it is handed, with no
// power-of-two fallback set, and leaves max_shift_ and max_log_slot_
// uninitialised; they are only assigned by the Galoiskey(context, max_shift)
// constructor. A rotation by an unlisted shift therefore does not report a
// missing key, it reaches rotation_index_generator with uninitialised bounds.
// Take the union of the index lists and generate one key from that.
//
// SCALE DISCIPLINE
// ----------------
// A ciphertext is held at the scale it happens to carry; no routine assumes
// that scale equals the operator's default. Constants are always encoded
// relative to the operand's own scale_ field, so the small drift that CKKS
// rescaling introduces is tracked rather than fought.

#ifndef HEONGPU_CKKS_LLAMA3_H
#define HEONGPU_CKKS_LLAMA3_H

#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/plaintext.cuh>

#include <functional>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief Chebyshev coefficients of @p f on [a, b].
         *
         * Returns c_0 .. c_degree with f(x) ~ sum_k c_k T_k(t) and
         * t = (2x - a - b) / (b - a), the convention
         * HEOperator::evaluate_poly consumes: c_0 is already halved, so the
         * series is summed with no further scaling.
         */
        std::vector<double>
        chebyshev_coefficients(const std::function<double(double)>& f, double a,
                               double b, int degree);

        /** @brief Evaluate the output of chebyshev_coefficients on the host. */
        double chebyshev_evaluate(const std::vector<double>& coeffs, double a,
                                  double b, double x);

        /**
         * @brief Layout of a square matrix batch inside one ciphertext.
         *
         * Matrix @c m entry (r, c) occupies slot ((r * d + c) * batch) + m.
         * Putting the matrix index on the fast axis makes a row rotation an
         * ordinary slot rotation by d * batch whose wraparound is exactly the
         * wraparound of the row index, so the JKLS row rotations of Section 4.2
         * stay exact with several matrices per ciphertext.
         */
        struct MatrixLayout
        {
            int d = 0;     ///< Rows and columns of every matrix.
            int batch = 0; ///< Matrices packed together.
            int slots = 0; ///< d * d * batch; must equal the slot count.

            MatrixLayout() = default;

            /** @throws std::invalid_argument if d is not a power of two. */
            MatrixLayout(int d, int batch);
        };

        /// The permutations of Equation (4), on a d x d host matrix in
        /// row-major order. Indices are taken modulo d throughout.
        std::vector<double> permute_sigma(const std::vector<double>& a, int d);
        std::vector<double> permute_tau(const std::vector<double>& a, int d);
        std::vector<double> rotate_rows_host(const std::vector<double>& a,
                                             int d, int k);
        std::vector<double> rotate_cols_host(const std::vector<double>& a,
                                             int d, int k);

        /** @brief Reference d x d transpose, for tests and for masks. */
        std::vector<double> transpose_host(const std::vector<double>& a, int d);

        /** @brief Reference d x d product, for tests and for building masks. */
        std::vector<double> matmul_host(const std::vector<double>& a,
                                        const std::vector<double>& b, int d);

        /**
         * @brief Homomorphic building blocks of a Llama-3 layer.
         *
         * Derives from HEArithmeticOperator so that the library's own
         * Paterson-Stockmeyer Chebyshev evaluator and its scale targeting are
         * reused rather than reimplemented.
         */
        class Llama3Operator : public HEArithmeticOperator<Scheme::CKKS>
        {
          public:
            /**
             * @param scale Default scaling factor. Also seeds scale_boot_,
             *              which evaluate_poly consults when deciding whether
             *              an intermediate needs rescaling.
             */
            Llama3Operator(HEContext<Scheme::CKKS> context,
                           HEEncoder<Scheme::CKKS>& encoder, double scale);

            /** @brief Slots available to every routine here. */
            int slot_count() const noexcept { return slot_count_; }

            /** @brief Default scaling factor. */
            double default_scale() const noexcept { return default_scale_; }

            // ---------------------------------------------------------------
            // Level and scale plumbing
            // ---------------------------------------------------------------

            /** @brief Drop the shallower of two ciphertexts onto the deeper. */
            void align_levels(Ciphertext<Scheme::CKKS>& a,
                              Ciphertext<Scheme::CKKS>& b);

            /** @brief Drop @p ct to @p depth; no-op if it is already there. */
            void drop_to_depth(Ciphertext<Scheme::CKKS>& ct, int depth);

            /**
             * @brief Multiply by a real constant, preserving the scale.
             *
             * Encodes the constant at the prime the following rescale divides
             * by, so the result comes back at exactly the input scale one
             * level down.
             */
            void multiply_constant(Ciphertext<Scheme::CKKS>& ct, double c);

            /**
             * @brief Multiply by a slot vector, preserving the scale.
             *
             * @param values Exactly slot_count() entries.
             */
            void multiply_vector(Ciphertext<Scheme::CKKS>& ct,
                                 const std::vector<double>& values);

            /** @brief Add a real constant. Consumes no level. */
            void add_constant(Ciphertext<Scheme::CKKS>& ct, double c);

            /**
             * @brief Multiply by a plaintext held at any level.
             *
             * multiply_plain insists the two operands sit at the same level,
             * while HEEncoder only ever encodes at the top of the chain, so a
             * plaintext prepared once and reused down a circuit has to be
             * dropped first. The drop is taken on a copy, leaving the caller's
             * plaintext reusable at its original level.
             *
             * Unlike multiply_constant and multiply_vector, this leaves the
             * rescale pending, because the caller chose the plaintext's scale
             * and may want to fold another product in first. Until it is
             * rescaled the result cannot be multiplied, rotated or dropped.
             */
            void multiply_plaintext(Ciphertext<Scheme::CKKS>& ct,
                                    Plaintext<Scheme::CKKS>& plain);

            /**
             * @brief Bring @p ct to exactly @p target, costing one level.
             *
             * Rescaling divides by a prime that only approximates the nominal
             * scale, so two ciphertexts that have been through different
             * numbers of products no longer agree on scale even when they
             * agree on level. Addition needs them to agree, so a residual
             * connection has to pay this level.
             */
            void match_scale(Ciphertext<Scheme::CKKS>& ct, double target);

            /**
             * @brief @p x + @p sublayer, reconciling level and scale first.
             *
             * The two operands of a residual connection have been through
             * completely different circuits, so neither the level nor the
             * scale lines up. Both are brought onto the deeper of the two and
             * onto one scale; that costs the single level match_scale needs.
             */
            Ciphertext<Scheme::CKKS>
            residual_add(Ciphertext<Scheme::CKKS>& x,
                         Ciphertext<Scheme::CKKS>& sublayer);

            /** @brief Square, relinearise and rescale. */
            void square(Ciphertext<Scheme::CKKS>& ct,
                        Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Multiply, relinearise and rescale, levels aligned. */
            Ciphertext<Scheme::CKKS>
            multiply_and_rescale(Ciphertext<Scheme::CKKS>& a,
                                 Ciphertext<Scheme::CKKS>& b,
                                 Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Slot reductions
            // ---------------------------------------------------------------

            /** @brief Rotations sum_strided needs a Galois key for. */
            static std::vector<int> strided_rotation_indices(int stride,
                                                             int count);

            /** @brief Rotations sum_blocked needs a Galois key for. */
            static std::vector<int> blocked_rotation_indices(int span);

            /**
             * @brief Sum the strided axis, replicating the total across it.
             *
             * Requires the strided layout: stride * count must equal the slot
             * count. Costs log2(count) rotations and no level.
             */
            void sum_strided(Ciphertext<Scheme::CKKS>& ct, int stride,
                             int count, Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Sum each aligned block of @p span slots, replicating the
             *        total across the block.
             *
             * The sliding-window sum a plain rotate-and-add produces is only
             * correct at multiples of @p span, so this masks those positions
             * and fans them back out. Costs 2 log2(span) rotations and one
             * level for the mask.
             */
            void sum_blocked(Ciphertext<Scheme::CKKS>& ct, int span,
                             Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // Polynomial primitives
            // ---------------------------------------------------------------

            /**
             * @brief Evaluate a Chebyshev series on [a, b].
             *
             * Maps the input into [-1, 1] first, which costs one level, then
             * hands the series to the library's BSGS evaluator.
             */
            Ciphertext<Scheme::CKKS>
            evaluate_chebyshev(Ciphertext<Scheme::CKKS>& ct,
                               const std::vector<double>& coeffs, double a,
                               double b, Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Chebyshev approximation of an arbitrary function. */
            Ciphertext<Scheme::CKKS>
            evaluate_function(Ciphertext<Scheme::CKKS>& ct,
                              const std::function<double(double)>& f, double a,
                              double b, int degree,
                              Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief 1 / sqrt(x) on [lo, hi].
             *
             * A Chebyshev approximation seeds Newton's y <- y (3 - x y^2) / 2,
             * which is division-free and quadratically convergent, so a low
             * seed degree plus a couple of steps beats a high-degree fit.
             * The halving is folded into a single scaling of x, making each
             * step cost three levels rather than four.
             *
             * @param newton_iterations Refinement steps; 0 returns the seed.
             */
            Ciphertext<Scheme::CKKS>
            inverse_sqrt(Ciphertext<Scheme::CKKS>& ct, double lo, double hi,
                         int degree, int newton_iterations,
                         Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief 1 / x on [lo, hi], seeded then refined by Newton's
             *        y <- y (2 - x y). Two levels per step.
             */
            Ciphertext<Scheme::CKKS>
            inverse(Ciphertext<Scheme::CKKS>& ct, double lo, double hi,
                    int degree, int newton_iterations,
                    Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief SiLU, x * sigmoid(x), on [-bound, bound].
             *
             * Section 3.1.3 reports degree 31 after calibration brings the
             * input range down; Table 2 puts that range at about 10.8.
             */
            Ciphertext<Scheme::CKKS> silu(Ciphertext<Scheme::CKKS>& ct,
                                          double bound, int degree,
                                          Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief exp(x / 2^k) for x in [-bound, 0], the y^(0) of the
             *        SoftMax of Section 2.3.
             *
             * The division by 2^k is folded into the approximated function, so
             * the scaling step of the algorithm is free.
             */
            Ciphertext<Scheme::CKKS>
            exp_scaled_negative(Ciphertext<Scheme::CKKS>& ct, double bound,
                                int squarings, int degree,
                                Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Llama-3 layers
            // ---------------------------------------------------------------

            /** @brief Shape and approximation settings for rms_norm. */
            struct RMSNormConfig
            {
                int stride = 0;  ///< Strided layout: instances per ciphertext.
                int count = 0;   ///< Channels held in one ciphertext.
                /// Total channels summed over. Must lie in
                /// (count * (inputs - 1), count * inputs]: every input
                /// contributes count channels and only the last may be padded.
                int channels = 0;
                double eps = 1e-5;
                double sum_lo = 0.0; ///< Range of the summed square.
                double sum_hi = 0.0;
                int degree = 31;
                int newton_iterations = 2;
            };

            /**
             * @brief RMSNorm: x_i * w_i / sqrt(mean_i(x_i^2) + eps).
             *
             * The channels of one token block may be split over several
             * ciphertexts; the squares are accumulated across @p in before the
             * strided reduction, so the mean is over all @c channels. The
             * division by the channel count is folded into the approximated
             * function and costs nothing.
             *
             * @param in      Channel-split ciphertexts, all at one level.
             * @param weights One plaintext per input, or empty to skip the
             *                learned scaling.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            rms_norm(std::vector<Ciphertext<Scheme::CKKS>>& in,
                     std::vector<Plaintext<Scheme::CKKS>>& weights,
                     const RMSNormConfig& config,
                     Galoiskey<Scheme::CKKS>& galois_key,
                     Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Shape and approximation settings for softmax. */
            struct SoftmaxConfig
            {
                bool strided = true; ///< Layout of the reduced axis.
                int stride = 0;      ///< Strided layout only.
                int count = 0;       ///< SoftMax length d, either layout.
                double bound = 0.0;  ///< Inputs lie in [-bound, 0].
                int iterations = 2;  ///< Section 4.3 fixes this at two.
                int exp_degree = 31;
                int inverse_degree = 15;
                int inverse_newton = 2;
            };

            /**
             * @brief SoftMax over @c count coordinates, Cho et al.
             *
             * Evaluates exp(x / 2^k), then k rounds of normalise-and-square.
             * Because (y_i / ||y||_2)^2 is y_i^2 / sum_j y_j^2, each round is
             * one squaring, one free reduction and one multiplication by the
             * reciprocal of the sum, and the reciprocal is what is
             * approximated rather than the reciprocal square root.
             *
             * Every round leaves sum_i y_i = 1 exactly, so after k rounds the
             * result is exp(x_i) / sum_j exp(x_j).
             *
             * Inputs must already be translated into [-bound, 0]; the paper
             * takes that translation from calibration rather than from a
             * homomorphic maximum.
             */
            Ciphertext<Scheme::CKKS>
            softmax(Ciphertext<Scheme::CKKS>& ct, const SoftmaxConfig& config,
                    Galoiskey<Scheme::CKKS>& galois_key,
                    Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief SoftMax restricted to the positions @p mask keeps.
             *
             * The mask is applied to the exponentials, where a zero removes a
             * coordinate from both the numerator and the sum, which is what
             * attention wants of a causal mask. It costs the one level a slot
             * multiplication costs.
             *
             * A mask entry may be any positive weight rather than one, and
             * this is what makes a variable-length mask usable: the rounds
             * normalise, so a factor constant along the reduced axis cancels
             * exactly, and choosing it as sqrt(count / kept) leaves the sum of
             * squares in the same range a full row would produce. Without that
             * the first round would have to approximate a reciprocal over a
             * range that widens with every position masked off.
             *
             * A masked coordinate is dropped, not excused: the exponential is
             * evaluated before the mask, so its input still has to lie in
             * [-bound, 0]. A Chebyshev fit diverges quickly outside its
             * interval, and a large enough garbage value would survive being
             * multiplied by the encoding of zero.
             *
             * @param mask Exactly slot_count() entries, or empty for none.
             */
            Ciphertext<Scheme::CKKS>
            softmax(Ciphertext<Scheme::CKKS>& ct, const SoftmaxConfig& config,
                    const std::vector<double>& mask,
                    Galoiskey<Scheme::CKKS>& galois_key,
                    Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief RoPE as one plaintext-ciphertext product pair.
             *
             * out = x * cos + swap(x) * sin, where swap exchanges the two
             * halves of each head's channel pair and is a single slot rotation
             * by @p swap_shift. Costs one level and one rotation; the angles
             * live entirely in the two plaintexts.
             */
            Ciphertext<Scheme::CKKS> rope(Ciphertext<Scheme::CKKS>& ct,
                                          Plaintext<Scheme::CKKS>& cos_plain,
                                          Plaintext<Scheme::CKKS>& sin_plain,
                                          int swap_shift,
                                          Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // PCMM for PC-attention, Section 4.2
            // ---------------------------------------------------------------

            /**
             * @brief The rearranged plaintext pt_{A,i,j,l} of Equation (5).
             *
             * pt_{A,i,j,l} = rot_R^{-l(i+jb)-jb} . rot_C^{i+jb} (tau^l . sigma(A)).
             * The paper stores only tau^l . sigma(A) and derives each of these
             * at run time, which is what keeps the weight footprint at one
             * copy; this reproduces that derivation, taking the stored matrix
             * as input.
             */
            static std::vector<double>
            pcmm_plaintext_block(const std::vector<double>& tau_sigma_a, int d,
                                 int i, int j, int b, int ell);

            /** @brief Rotations pcmm needs a Galois key for. */
            static std::vector<int>
            pcmm_rotation_indices(const MatrixLayout& layout, int giant,
                                  int baby);

            /**
             * @brief Depth-one PCMM of Section 4.2, Equation (5).
             *
             * Consumes tau^{l+1}(B) as the ciphertext and produces tau^l(A B),
             * so the layout conversion the JKLS algorithm would need is
             * absorbed into the operation itself. One multiplicative level and
             * giant + baby - 2 rotations.
             *
             * @param a_stored tau^l . sigma(A), the single stored copy of the
             *                 plaintext weight matrix, d * d entries.
             * @param giant,baby BSGS factors with giant * baby == d.
             * @param ell      The exponent l of Equation (5).
             */
            Ciphertext<Scheme::CKKS>
            pcmm(Ciphertext<Scheme::CKKS>& ct,
                 const std::vector<double>& a_stored,
                 const MatrixLayout& layout, int giant, int baby, int ell,
                 Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // Linear maps on a packed matrix, and the encrypted product
            // ---------------------------------------------------------------

            /** @brief Rotations tau needs a Galois key for. */
            static std::vector<int>
            tau_rotation_indices(const MatrixLayout& layout);

            /** @brief Rotations transpose needs a Galois key for. */
            static std::vector<int>
            transpose_rotation_indices(const MatrixLayout& layout);

            /** @brief Rotations ccmm needs a Galois key for. */
            static std::vector<int>
            ccmm_rotation_indices(const MatrixLayout& layout);

            /**
             * @brief The permutation tau of Equation (4), homomorphically.
             *
             * tau(X)[i][j] = X[(i + j) mod d][j]. A row rotation is an
             * ordinary slot rotation in this layout, so this is d - 1
             * rotations and the one level the column masks cost.
             *
             * Equation (5) consumes tau^{l+1} of its operand, so an isolated
             * pcmm has to be handed tau of its input. A chain of them does not
             * pay this per product: it feeds tau^L in once and peels one power
             * off at each step, which is the whole point of the tau tower.
             */
            Ciphertext<Scheme::CKKS> tau(Ciphertext<Scheme::CKKS>& ct,
                                         const MatrixLayout& layout,
                                         Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Matrix transpose, homomorphically.
             *
             * Entry (i, j) moves by (j - i)(d - 1) positions within its
             * matrix, so the map has 2d - 1 diagonals: 2d - 2 rotations and
             * one masking level.
             */
            Ciphertext<Scheme::CKKS>
            transpose(Ciphertext<Scheme::CKKS>& ct, const MatrixLayout& layout,
                      Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief @p scale * A * B with both operands encrypted.
             *
             * Section 4.2 covers the plaintext-weight product only, and
             * attention needs Q^T K and V P^T, where both operands came out of
             * a previous ciphertext. This is the Jiang-Kim-Lauter-Song
             * identity A B = sum_k rot_C^k(sigma(A)) * rot_R^k(tau(B)) carried
             * over to the batched layout.
             *
             * Depth is two rather than the usual three because sigma and the
             * column rotation are taken together: rot_C^k(sigma(A)) reads
             * A[i][(i + j + k) mod d], which is still one diagonal per shift of
             * A itself, so the 2d - 2 rotations of A are shared by every k and
             * only the masks differ. That also makes @p scale free, since it
             * multiplies masks that are being encoded anyway.
             *
             * Costs 4d - 4 rotations, d(2d - 1) + d plaintext products and d
             * encrypted products. The encrypted products are accumulated
             * before relinearising, so there is one key switch rather than d.
             */
            Ciphertext<Scheme::CKKS>
            ccmm(Ciphertext<Scheme::CKKS>& a, Ciphertext<Scheme::CKKS>& b,
                 const MatrixLayout& layout, double scale,
                 Galoiskey<Scheme::CKKS>& galois_key,
                 Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Sublayers
            // ---------------------------------------------------------------
            //
            // Both sublayers hold activations TRANSPOSED, X[channel][token],
            // packed by MatrixLayout with the batch axis carrying independent
            // instances such as heads. Two things fall out of that and neither
            // is a coincidence:
            //
            //   - a projection is W X, so Equation (5) applies with the weight
            //     on the left, exactly where it needs the plaintext to be;
            //   - a sum over channels is a sum over the slow axis, which is
            //     the exact, level-free strided reduction of Section 3.2.
            //
            // The weights are per block: one ciphertext holds d channels of d
            // tokens, so a model whose width exceeds d needs the caller to sum
            // the products of the channel blocks, the way rms_norm already
            // accumulates over its inputs. That orchestration is Table 4 and is
            // not attempted here.

            /** @brief Plaintext weights of one attention sublayer. */
            struct AttentionWeights
            {
                /// d x d, or d x d per batch entry. Empty skips the
                /// projection and takes the input as already projected.
                std::vector<double> query;
                std::vector<double> key;
                std::vector<double> value;
                std::vector<double> output; ///< Empty skips W_o.
            };

            /** @brief Shape and approximation settings for attention. */
            struct AttentionConfig
            {
                MatrixLayout layout;
                int giant = 0; ///< BSGS split of the projections; 0 balances.
                int baby = 0;
                bool causal = true;   ///< Mask keys ahead of the query.
                bool rope = false;    ///< Apply RoPE to Q and K.
                double head_scale = 0.0;  ///< 1/sqrt(head_dim); 0 uses 1/sqrt(d).
                /// Subtracted from the scores so they land in [-bound, 0].
                /// Calibrated, as in the paper, not computed homomorphically.
                double score_shift = 0.0;
                /// bound, iterations and the degrees are the caller's; the
                /// layout fixes strided, stride and count and they are
                /// overwritten.
                SoftmaxConfig softmax;
            };

            /** @brief The mask a causal attention wants, ready for softmax. */
            static std::vector<double> causal_mask(const MatrixLayout& layout);

            /** @brief Rotations attention needs a Galois key for. */
            static std::vector<int>
            attention_rotation_indices(const AttentionConfig& config);

            /** @brief The RoPE half-swap shift implied by a layout. */
            static int rope_swap_shift(const MatrixLayout& layout);

            /**
             * @brief One attention sublayer over a transposed activation
             *        block.
             *
             * Q, K and V are projected out of @p x, optionally rotated by
             * RoPE, and the scores are formed as K^T Q rather than Q^T K.
             * That is deliberate: it puts the key position on the slow axis,
             * which is the axis the free strided reduction sums over, so the
             * SoftMax denominator costs no level and no mask. The result of
             * the SoftMax is then already P^T, which is the operand V needs.
             *
             * @param rope Empty, or exactly {cos, sin} as rope() wants them.
             */
            Ciphertext<Scheme::CKKS>
            attention(Ciphertext<Scheme::CKKS>& x,
                      const AttentionWeights& weights,
                      std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
                      const AttentionConfig& config,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Plaintext weights of one SwiGLU sublayer. */
            struct FeedForwardWeights
            {
                std::vector<double> gate;
                std::vector<double> up;
                std::vector<double> down;
            };

            /** @brief Shape and approximation settings for feed_forward. */
            struct FeedForwardConfig
            {
                MatrixLayout layout;
                int giant = 0; ///< BSGS split; 0 balances.
                int baby = 0;
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;     ///< Section 3.1.3.
            };

            /** @brief Rotations feed_forward needs a Galois key for. */
            static std::vector<int>
            feed_forward_rotation_indices(const FeedForwardConfig& config);

            /**
             * @brief The SwiGLU sublayer, W_down (SiLU(W_gate x) * W_up x).
             *
             * The gate and the up projection share one tau of the input, so
             * the sublayer pays for that map once rather than twice.
             */
            Ciphertext<Scheme::CKKS>
            feed_forward(Ciphertext<Scheme::CKKS>& x,
                         const FeedForwardWeights& weights,
                         const FeedForwardConfig& config,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key);

          private:
            /// The prime the next rescale of @p ct will divide by.
            double rescale_prime(const Ciphertext<Scheme::CKKS>& ct) const;

            /**
             * @brief a += b, refusing operands that disagree on scale.
             *
             * CKKS addition is only meaningful between equal scales, and the
             * library checks the level but not the scale, so a mismatch here
             * is a silently wrong sum. Every addition in this file goes
             * through this.
             */
            void add_same_scale(Ciphertext<Scheme::CKKS>& a,
                                Ciphertext<Scheme::CKKS>& b,
                                const char* context);

            /// Encode @p values at @p scale, dropped onto @p depth.
            Plaintext<Scheme::CKKS> encode(const std::vector<double>& values,
                                           double scale, int depth);

            /// Multiply @p ct by @p values and fold the product into @p acc,
            /// which is left with its rescale still pending.
            void accumulate_masked(Ciphertext<Scheme::CKKS>& acc,
                                   bool& started,
                                   Ciphertext<Scheme::CKKS>& ct,
                                   const std::vector<double>& values,
                                   double plain_scale, const char* context);

            /// Balanced BSGS factors of @p d, or the caller's if given.
            static void bsgs_split(int d, int& giant, int& baby);

            /// sigma applied to every d x d block of a stored weight.
            static std::vector<double> sigma_blocks(const std::vector<double>& w,
                                                    const MatrixLayout& layout,
                                                    const char* name);

            /// One W x projection: pcmm at l = 0 on an operand already tau'd.
            Ciphertext<Scheme::CKKS>
            project(Ciphertext<Scheme::CKKS>& tau_x,
                    const std::vector<double>& weight,
                    const MatrixLayout& layout, int giant, int baby,
                    const char* name, Galoiskey<Scheme::CKKS>& galois_key);

            HEEncoder<Scheme::CKKS> encoder_;
            /// Cached: the context hands out its modulus chain by value.
            std::vector<Modulus64> primes_;
            int slot_count_;
            double default_scale_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_H
