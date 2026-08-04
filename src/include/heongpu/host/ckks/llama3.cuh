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

        /**
         * @brief A weight matrix cut into the d x d blocks a layout can hold.
         *
         * Block (i, j) carries input channel block j into output channel block
         * i, so a C_out by C_in weight is out_blocks by in_blocks of them, each
         * either d * d entries shared by the batch, d * d per batch entry, or
         * empty. An empty block is a zero block and is skipped rather than
         * encoded, which is what lets a block-diagonal weight -- a per-head
         * projection, say -- cost only its diagonal.
         */
        struct BlockMatrix
        {
            int out_blocks = 0;
            int in_blocks = 0;
            /// Row-major, out_blocks * in_blocks entries.
            std::vector<std::vector<double>> blocks;

            BlockMatrix() = default;

            /** @brief An all-empty, and therefore all-zero, block matrix. */
            BlockMatrix(int out_blocks, int in_blocks);

            /// One block wide and one block tall: the unblocked weight. An
            /// empty vector stays empty, which is how the sublayers spell
            /// "there is no such weight".
            BlockMatrix(std::vector<double> single);

            std::vector<double>& at(int i, int j);
            const std::vector<double>& at(int i, int j) const;

            bool empty() const { return blocks.empty(); }
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

            /** @brief The same, block by block, over a blocked activation. */
            std::vector<Ciphertext<Scheme::CKKS>>
            residual_add(std::vector<Ciphertext<Scheme::CKKS>>& x,
                         std::vector<Ciphertext<Scheme::CKKS>>& sublayer);

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
                /// Token blocks the sequence is cut into. The inputs are then
                /// channel-major, block (j, s) at j * token_blocks + s, and
                /// each token block is normalised over its own channels.
                int token_blocks = 1;
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
             * A sequence longer than d is several token blocks, and they are
             * independent: each is normalised over its own channels only. The
             * inputs are then channel-major and the weights follow them, since
             * a learned scale belongs to a channel and not to a token.
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
                /// Reduced coordinates held in ONE ciphertext, either layout.
                /// When the axis is split over several, the SoftMax length is
                /// this times their number.
                int count = 0;
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
             * @brief One SoftMax whose reduced axis spans several ciphertexts.
             *
             * A sequence longer than d puts the keys of one query in more than
             * one ciphertext, and the denominator has to cover all of them.
             * The parts share a slot layout and differ only in which stretch
             * of the axis they hold, so the sum of their reductions is the
             * reduction of their sum: the squares are added slot-wise first
             * and reduced once, and the whole SoftMax pays for one reduction
             * and one reciprocal however many ciphertexts the axis is cut
             * into. Only the squaring and the final product are per part.
             *
             * Depth is therefore exactly that of the single-ciphertext form.
             * Splitting the axis costs products, not levels.
             *
             * The SoftMax length is @c config.count times the number of parts,
             * and it is that total the mask weights and the fitted ranges are
             * taken against.
             *
             * @param masks Empty for none, or one per part; an individual mask
             *              may be empty to leave its part unmasked. A part
             *              that is masked off entirely should be left out
             *              rather than passed as zeros, since a dropped
             *              coordinate still has to have been in [-bound, 0]
             *              before the exponential.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            softmax(std::vector<Ciphertext<Scheme::CKKS>>& parts,
                    const SoftmaxConfig& config,
                    const std::vector<std::vector<double>>& masks,
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
            // One ciphertext holds d channels of d tokens, so a model wider
            // than d is carried by several of them, block j holding channels
            // [j d, (j + 1) d). Every routine below therefore comes in two
            // forms: one ciphertext, which is the width-d case, and a vector of
            // them, which is the general one. The single-ciphertext form is the
            // vector form at one block and is implemented as such.
            //
            // What makes the general form affordable is that a projection is
            // then the block product Y[i] = sum_j W(i, j) X[j], and every term
            // of that sum leaves Equation (5) at the same level and the same
            // scale, so the sum itself is free. Widening a model buys products,
            // not depth.
            //
            // The sequence is cut the same way. A prompt longer than d is
            // several TOKEN blocks, and an activation is then a grid: channel
            // block j of token block s at index j * token_blocks + s, which
            // makes the width-only case token_blocks = 1 and leaves it exactly
            // as it was. Every configuration below carries that count.
            //
            // Token blocking is not symmetric with channel blocking, because
            // the two axes meet differently:
            //
            //   - a weight touches channels only, so a projection is the same
            //     block product run once per token block and the two blockings
            //     simply multiply;
            //   - attention is the axis talking to itself. Scores become a
            //     grid S(u, s) over key block u and query block s, the SoftMax
            //     denominator of a query has to sum over every u, and the
            //     value product sums over u again. Causality then deletes the
            //     half with u > s outright, which is why those blocks are
            //     never formed rather than formed and masked.
            //
            // Neither costs depth. The score blocks of one query share a level
            // and a scale, the SoftMax over them reduces once, and the value
            // products land together, so a long sequence buys products and
            // ciphertexts. That is the whole of Table 4's blocking; what is
            // still absent is its multi-ring, multi-encoding half.

            /**
             * @brief Y[i] = sum_j W(i, j) X[j], over channel blocks.
             *
             * The inputs are already tau'd, because a projection consumes
             * tau of its operand and the sublayers take that map once and
             * share it across the weights that read the same activation.
             *
             * A weight reads channels and says nothing about tokens, so a
             * blocked sequence runs the same block product once per token
             * block against the same plaintexts.
             *
             * @param tau_x One tau'd block per input channel block and token
             *              block, at index j * token_blocks + s, all at one
             *              level and one scale.
             * @param giant,baby BSGS factors with giant * baby == d, or both
             *                   zero to take the balanced split.
             * @param token_blocks Token blocks the sequence is cut into.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            project_blocks(std::vector<Ciphertext<Scheme::CKKS>>& tau_x,
                           const BlockMatrix& weight,
                           const MatrixLayout& layout, int giant, int baby,
                           const char* name,
                           Galoiskey<Scheme::CKKS>& galois_key,
                           int token_blocks = 1);

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
                /// Heads sharing the query channel blocks between them; each
                /// takes a contiguous run of query_blocks / heads of them, so
                /// head_dim is that run times d.
                int heads = 1;
                /// Distinct key and value heads, for grouped-query attention;
                /// 0 means one apiece, which is multi-head attention. Must
                /// divide @c heads, and heads / kv_heads consecutive query
                /// heads then read the same key and value blocks. The key and
                /// value weights produce kv_heads runs rather than heads, so
                /// they are narrower than the query weight by that factor,
                /// which is the point of the arrangement.
                int kv_heads = 0;
                /// Token blocks the sequence is cut into, so the sequence
                /// length is token_blocks * d. The activations are then
                /// channel-major, block (j, s) at j * token_blocks + s.
                int token_blocks = 1;
                bool causal = true;   ///< Mask keys ahead of the query.
                bool rope = false;    ///< Apply RoPE to Q and K.
                /// 1/sqrt(head_dim); 0 derives it from the blocks per head.
                double head_scale = 0.0;
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

            /**
             * @brief The causal mask of one score block of a cut sequence.
             *
             * Block (@p key_block, @p query_block) of the score grid, keys on
             * the slow axis. Three cases and only the third has any shape to
             * it: a key block entirely behind the query block is kept whole, a
             * key block entirely ahead of it is dropped whole and comes back
             * as an empty vector the caller should skip, and the diagonal
             * block is the triangle.
             *
             * The surviving weight is sqrt(visible / kept), where kept is the
             * global count of keys the query admits and visible is the size of
             * the key blocks that were formed, (query_block + 1) * d. Neither
             * is per block, and both matter: the weight cancels only because
             * it is constant along the key axis, and that axis now runs across
             * the blocks, while the SoftMax fits its first reciprocal over the
             * coordinates it is actually handed, which is the blocks a causal
             * query block is given rather than the whole sequence.
             */
            static std::vector<double>
            causal_block_mask(const MatrixLayout& layout, int query_block,
                              int key_block, int token_blocks);

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

            /** @brief Blocked weights of one attention sublayer. */
            struct BlockAttentionWeights
            {
                /// The key and the value must agree on out_blocks. The query
                /// agrees with them too under multi-head attention, and is
                /// heads / kv_heads times wider under grouped-query attention.
                BlockMatrix query;
                BlockMatrix key;
                BlockMatrix value;
                BlockMatrix output; ///< Empty skips W_o.
            };

            /**
             * @brief Attention over a model wider or longer than one block.
             *
             * A head owns a contiguous run of the query blocks, and its scores
             * are the sum of that run's K[j]^T Q[j]. Every term lands at one
             * level and one scale, so a head spanning several blocks costs
             * products rather than depth, and the SoftMax that follows is one
             * per head however wide the head is. Under grouped-query attention
             * several query heads read one run of key and value blocks; they
             * still get their own scores and their own SoftMax, because it is
             * the query that differs.
             *
             * A sequence cut into token blocks makes the scores a grid. Query
             * block s takes one SoftMax per head over the key blocks it is
             * allowed to see, which under a causal mask is u <= s; the blocks
             * ahead are never formed. The value product then sums over the
             * same u. Both sums are free, so the sequence length costs
             * ciphertexts and products and no depth at all.
             *
             * @param rope_plain Empty, or 2 * (blocks per head) plaintexts per
             *                   token block, token-block-major: the pair for
             *                   block t of a head in token block s sits at
             *                   2 * (s * per_head + t). RoPE pairs channel c
             *                   with c + head_dim / 2: within one block when a
             *                   head is one block, and between blocks
             *                   otherwise, where it needs no rotation at all.
             *                   The angles depend on the channel inside the
             *                   head and on the absolute token position, so
             *                   every head reads the same plaintexts but every
             *                   token block needs its own.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            attention(std::vector<Ciphertext<Scheme::CKKS>>& x,
                      const BlockAttentionWeights& weights,
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
                /// Token blocks the sequence is cut into. Every token is its
                /// own SwiGLU, so this only says how the inputs are indexed.
                int token_blocks = 1;
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

            /** @brief Blocked weights of one SwiGLU sublayer. */
            struct BlockFeedForwardWeights
            {
                /// The gate and the up projection must agree on out_blocks,
                /// which is the hidden width; down carries that back.
                BlockMatrix gate;
                BlockMatrix up;
                BlockMatrix down;
            };

            /**
             * @brief SwiGLU over a model wider than one channel block.
             *
             * The hidden width is free to differ from the input width, which
             * is the point: Llama-3 widens by a factor of three and a half
             * before contracting, and here that is simply a gate and an up
             * weight with more output blocks than input ones.
             *
             * A cut sequence adds nothing but indices. SwiGLU is a map on one
             * token's channels, so the token blocks never meet.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            feed_forward(std::vector<Ciphertext<Scheme::CKKS>>& x,
                         const BlockFeedForwardWeights& weights,
                         const FeedForwardConfig& config,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // Bootstrapping, and the whole block it makes possible
            // ---------------------------------------------------------------

            /**
             * @brief Refresh the modulus chain of @p x.
             *
             * A bootstrap is not a longer chain. It spends a fixed slice of
             * the chain it is given on itself, and hands back the rest: with
             * the regular procedure and a BootstrappingConfig of (CtoS, StoC,
             * taylor) that slice is CtoS + taylor + StoC + 8 levels, so what a
             * caller gets is the chain less that and not the chain.
             *
             * @p x is dropped to the bottom of the chain first, which is where
             * the procedure insists on being handed it, and is left there.
             * Dropping is free and anything still unspent was about to be
             * discarded anyway, so a caller with levels in hand simply loses
             * them; that is why the bootstrap belongs at the point in a
             * circuit where the chain has actually run out.
             *
             * This is by a wide margin the least accurate operation in the
             * module. Everything else lands on the CKKS noise floor near 1e-8,
             * while the sine standing in for the modular reduction holds
             * around 1e-3 relative, so a circuit that bootstraps carries that
             * error to its end.
             *
             * generate_bootstrapping_params must have run first, and @p
             * boot_key must hold exactly the indices bootstrapping_key_indexs
             * advertises. That is a different list from the one the sublayers
             * ask for, so it is a second key and not a wider one.
             */
            Ciphertext<Scheme::CKKS>
            bootstrap(Ciphertext<Scheme::CKKS>& x,
                      Galoiskey<Scheme::CKKS>& boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /** @brief The same, block by block, over a blocked activation. */
            std::vector<Ciphertext<Scheme::CKKS>>
            bootstrap(std::vector<Ciphertext<Scheme::CKKS>>& x,
                      Galoiskey<Scheme::CKKS>& boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Plaintext weights of one whole transformer block. */
            struct TransformerBlockWeights
            {
                /// Learned scale of each norm, one plaintext per block of the
                /// activation, or empty to leave the scaling out.
                std::vector<Plaintext<Scheme::CKKS>> attention_norm;
                std::vector<Plaintext<Scheme::CKKS>> feed_forward_norm;
                BlockAttentionWeights attention;
                BlockFeedForwardWeights feed_forward;
                /// RoPE plaintexts, as the attention sublayer wants them.
                std::vector<Plaintext<Scheme::CKKS>> rope;
            };

            /** @brief Shape and approximation settings for a whole block. */
            struct TransformerBlockConfig
            {
                RMSNormConfig attention_norm;
                AttentionConfig attention;
                RMSNormConfig feed_forward_norm;
                FeedForwardConfig feed_forward;
                /// Refresh the residual stream between the two sublayers.
                ///
                /// One refresh is enough and this is where it goes. Attention
                /// under its pre-norm is the deeper half by a long way, so the
                /// stream arrives here with the chain spent, and the SwiGLU
                /// half then fits in what a bootstrap gives back. False runs
                /// the whole block on one chain, which wants about fifty
                /// levels.
                bool bootstrap = true;
            };

            /** @brief Rotations a whole block needs a Galois key for. */
            static std::vector<int> transformer_block_rotation_indices(
                const TransformerBlockConfig& config);

            /**
             * @brief One pre-norm transformer block, the unit Llama-3 repeats.
             *
             * Norm, attention, residual; then norm, SwiGLU, residual. The
             * pre-norm arrangement is what makes the residual cheap: the
             * sublayer reads a normalised copy while the stream itself is
             * carried around untouched, so the addition that closes each half
             * pays only the level match_scale needs.
             *
             * The bootstrap sits between the two halves rather than at the top
             * or the bottom of the block. That is not an arbitrary choice: a
             * refresh is worth the most where the chain is emptiest, and the
             * attention half spends roughly three levels for every one the
             * SwiGLU half does.
             *
             * @param galois_key The sublayers' rotations,
             *                   transformer_block_rotation_indices.
             * @param boot_key   Bootstrapping's own rotations, which are a
             *                   different list; unused when the config turns
             *                   the refresh off.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            transformer_block(std::vector<Ciphertext<Scheme::CKKS>>& x,
                              TransformerBlockWeights& weights,
                              const TransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Galoiskey<Scheme::CKKS>& boot_key,
                              Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Settings of a run of blocks, one entry per block. */
            struct TransformerStackConfig
            {
                /// The blocks, in the order they run.
                ///
                /// Every block of Llama-3 has the same shape, so these agree
                /// on strides, degrees and head counts. They do not agree on
                /// the intervals the approximations are fitted over: each
                /// block meets its own distribution of activations, and a
                /// Chebyshev fit is worth nothing outside the interval it was
                /// fitted on. That is why a stack is a config per block and
                /// not one config and a count.
                std::vector<TransformerBlockConfig> blocks;

                /// Refresh the stream at the seam between one block and the
                /// next.
                ///
                /// A block already refreshes once, between its halves, and
                /// hands back a stream with only what the SwiGLU half did not
                /// spend. That is nowhere near what the next block's attention
                /// half wants, so a stack costs two bootstraps per block and
                /// not one.
                bool bootstrap_between_blocks = true;
            };

            /** @brief Rotations a whole stack needs a Galois key for. */
            static std::vector<int> transformer_stack_rotation_indices(
                const TransformerStackConfig& config);

            /**
             * @brief A run of transformer blocks: the body of the model.
             *
             * The chain a stack needs is not set by the block but by the
             * deepest *half* of a block, and that is a different number. A
             * block starts on a full chain and only has to fit its second half
             * into what the refresh in its middle hands back. Every block
             * after the first starts on a refreshed chain instead, so its
             * attention half has to fit there too — and the attention half is
             * the deeper of the two by a long way. A chain that runs one block
             * will therefore not run two.
             *
             * The stream is left as the last block produced it, unrefreshed,
             * so a caller has the SwiGLU half's remainder to spend on whatever
             * reads the stack.
             *
             * @param galois_key The blocks' rotations,
             *                   transformer_stack_rotation_indices.
             * @param boot_key   Bootstrapping's own rotations, which are a
             *                   different list.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            transformer_stack(std::vector<Ciphertext<Scheme::CKKS>>& x,
                              std::vector<TransformerBlockWeights>& weights,
                              const TransformerStackConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Galoiskey<Scheme::CKKS>& boot_key,
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

            /// Refuse blocks that have drifted apart, since the sums below add
            /// them and CKKS addition needs one level and one scale. An empty
            /// vector is refused too: there is no such model.
            void require_uniform(const std::vector<Ciphertext<Scheme::CKKS>>& in,
                                 const char* name) const;

            /// tau of every channel block, which the projections then share.
            std::vector<Ciphertext<Scheme::CKKS>>
            tau_blocks(std::vector<Ciphertext<Scheme::CKKS>>& in,
                       const MatrixLayout& layout,
                       Galoiskey<Scheme::CKKS>& galois_key);

            /// RoPE across a head's channel blocks, in place. @p in is
            /// channel-major over @p token_blocks, as everything here is.
            void rope_blocks(std::vector<Ciphertext<Scheme::CKKS>>& in,
                             int per_head, int token_blocks,
                             std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
                             const MatrixLayout& layout,
                             Galoiskey<Scheme::CKKS>& galois_key);

            /// The scores of one head against one query token block, summed
            /// over the head's channel blocks: sum_t K[t][u]^T Q[t][s].
            Ciphertext<Scheme::CKKS>
            head_scores(std::vector<Ciphertext<Scheme::CKKS>>& key_t,
                        std::vector<Ciphertext<Scheme::CKKS>>& query,
                        int first_key, int first_query, int per_head,
                        int token_blocks, int key_block, int query_block,
                        const MatrixLayout& layout, double scale,
                        Galoiskey<Scheme::CKKS>& galois_key,
                        Relinkey<Scheme::CKKS>& relin_key);

            HEEncoder<Scheme::CKKS> encoder_;
            /// Cached: the context hands out its modulus chain by value.
            std::vector<Modulus64> primes_;
            int slot_count_;
            double default_scale_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_H
