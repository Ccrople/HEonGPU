// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Batch matrix encoding, batch matrix encryption and the batch matrix
// multiplication primitives PCMM, CMT and CCMM.
//
// Implements Definition 1 (batch matrix encoding), Definition 2 (batch matrix
// encryption), Algorithm 1 (batch CPMM), Algorithm 2 (TWEAK), Algorithm 3
// (batch CMT), Algorithm 4 (batch CCMM) and Algorithm 5 (rectangular CPMM) of
// Cheon, Kang and Lee, "Fast Batch Matrix Multiplication in Ciphertexts".
//
// R_N = Z[X]/(X^N + 1) is viewed as a rank-d module over the subring
// R_k = Z[Y]/(Y^k + 1) with Y = X^d and N = d * k. A matrix over R_k carries
// k/2 complex matrices at once, and one matrix multiplication over R_k performs
// all k/2 complex products simultaneously.

#ifndef HEONGPU_CKKS_BATCHMATRIX_H
#define HEONGPU_CKKS_BATCHMATRIX_H

#include <heongpu/util/schemes.h>
#include <heongpu/kernel/defines.h>
#include <heongpu/util/devicevector.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/operator.cuh>

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace heongpu
{
    /**
     * @brief Shape of a batch matrix operation.
     *
     * The row count @c d is the rank of R_N over R_k and therefore fixes both
     * the subring degree @c k = N/d and the batch size @c k/2. The column count
     * is not constrained by the subring and is carried separately: it is simply
     * the number of RLWE ciphertexts holding the matrix.
     */
    struct BatchMatrixLayout
    {
        int N = 0;     ///< Ring degree of the ambient ring R_N.
        int d = 0;     ///< Matrix rows; rank of R_N as an R_k-module.
        int k = 0;     ///< Subring degree, N/d.
        int batch = 0; ///< Number of complex matrices packed together, k/2.

        BatchMatrixLayout() = default;

        /**
         * @throws std::invalid_argument if d does not divide N, if either is not
         *         a power of two, or if the resulting subring degree is below 2.
         */
        BatchMatrixLayout(int N, int d);
    };

    /**
     * @brief Batch matrix encoding of Definition 1, evaluated on the host.
     *
     * Applies the length-k inverse DFT entrywise across a batch of complex
     * matrices, yielding a single matrix whose entries are elements of R_k.
     * Because the forward map m -> (m(zeta^{5^j}))_j is a ring homomorphism from
     * R_k onto C^{k/2}, multiplication in R_k realises the entire batch of
     * complex matrix products at once.
     *
     * This is a self-contained encoding: it is deliberately not HEEncoder's slot
     * encoding, and no conversion between the two is required.
     */
    class BatchMatrixEncoder
    {
      public:
        /** @param k Subring degree; must be a power of two of at least 2. */
        explicit BatchMatrixEncoder(int k);

        /** @brief Number of complex matrices carried per encoded matrix, k/2. */
        inline int slots() const noexcept { return k_ / 2; }

        /** @brief Subring degree k. */
        inline int degree() const noexcept { return k_; }

        /**
         * @brief Encode a batch of complex matrices into one matrix over R_k.
         *
         * @param batch k/2 matrices, each @p rows x @p cols in row-major order.
         * @param rows  Row count of every matrix in the batch.
         * @param cols  Column count of every matrix in the batch.
         * @param scale CKKS scaling factor applied before rounding.
         * @param out   Receives (rows * cols) * k coefficients; entry (i,j)
         *              occupies the k-element run at (i * cols + j) * k.
         */
        void encode(
            const std::vector<std::vector<std::complex<double>>>& batch,
            int rows, int cols, double scale, std::vector<int64_t>& out) const;

        /** @brief Inverse of encode; recovers the k/2 complex matrices. */
        void decode(const std::vector<int64_t>& in, int rows, int cols,
                    double scale,
                    std::vector<std::vector<std::complex<double>>>& batch)
            const;

      private:
        int k_;
        /// pow_[j * k_ + t] = zeta^{5^j * t} with zeta = exp(i * pi / k_).
        std::vector<std::complex<double>> pow_;
    };

    /**
     * @brief Build the R_N coefficient vectors of a matrix encryption's columns.
     *
     * Given the encoded matrix produced by BatchMatrixEncoder::encode, returns
     * the @c cols coefficient vectors m_j = sum_{i<d} M[i][j] * X^i of
     * Definition 2. Callers hand these to an RLWE encryption routine.
     *
     * @param coeffs Encoded matrix, (rows * cols) * k coefficients.
     * @param layout Subring layout; @c layout.d must equal @p rows.
     * @param out    Receives @p cols vectors of length @c layout.N.
     */
    void build_matrix_encryption_coefficients(
        const std::vector<int64_t>& coeffs, const BatchMatrixLayout& layout,
        int rows, int cols, std::vector<std::vector<int64_t>>& out);

    /**
     * @brief Inverse of build_matrix_encryption_coefficients.
     *
     * Recovers the encoded matrix from the coefficient vectors of the columns,
     * so a decrypted matrix encryption can be handed back to
     * BatchMatrixEncoder::decode.
     */
    void split_matrix_encryption_coefficients(
        const std::vector<std::vector<int64_t>>& in,
        const BatchMatrixLayout& layout, int rows, int cols,
        std::vector<int64_t>& coeffs);

    /**
     * @brief Rotation indices whose keys batch CMT, and hence batch CCMM,
     *        require.
     *
     * CMT applies the automorphisms X -> X^(2kt+1) for t in [d]. That subgroup
     * of Z*_2N is generated by 5^(k/2), so every element is an ordinary CKKS
     * slot rotation; this returns the corresponding rotation indices, ready to
     * pass to HEKeyGenerator::generate_galois_key.
     */
    std::vector<int>
    get_batch_cmt_rotation_indices(const BatchMatrixLayout& layout);

    /**
     * @brief Rotation indices the rectangular algorithms require.
     *
     * Algorithm 5 runs two CMT stages: one at the layout (N, N/2), which is
     * what turns the constant terms of Theorem 3 into the leading d
     * ciphertexts, and one at @p layout, which returns the result to the
     * caller's column-wise encryption. The key set is the union.
     *
     * The first stage is the expensive half and it is expensive by
     * construction: at k = 2 the automorphisms X -> X^(2kt+1) run over the
     * whole rotation group, so it asks for N/2 - 1 keys. That count depends on
     * the ring degree ALONE -- not on d, not on the model width -- so N is the
     * only lever on the key budget of this path.
     */
    std::vector<int>
    get_rectangular_rotation_indices(const BatchMatrixLayout& layout);

    /**
     * @brief Per-limb twiddle tables for the subring transform.
     *
     * Depends only on the subring degree and the RNS primes in play, so one set
     * is built per (layout, level) and reused across operations.
     */
    struct BatchSubringTables
    {
        int k = 0;
        int d = 0;
        int num_limbs = 0;
        DeviceVector<Data64> psi;       ///< [limbs][k] forward twiddles.
        DeviceVector<Data64> psi_inv;   ///< [limbs][k] inverse twiddles.
        DeviceVector<Data64> kinv;      ///< [limbs] k^-1 mod p.
        DeviceVector<Data64> wfwd;      ///< [limbs][d] omega^m.
        DeviceVector<Data64> winv;      ///< [limbs][d] omega^-m.
        DeviceVector<Data64> psi_fwd_n; ///< [limbs][k][d] psi_N^(+h0(s)*i).
        DeviceVector<Data64> psi_inv_n; ///< [limbs][k][d] psi_N^(-h0(s)*i)/d.
        DeviceVector<Data64> psi_n;     ///< [limbs] primitive 2N-th root.
        /// [limbs][2N] every power of psi_N. The monomial twiddle is a lookup
        /// rather than a per-element modular exponentiation, which otherwise
        /// dominates the whole CMT.
        DeviceVector<Data64> psi_n_pow;
        DeviceVector<Data64> dinv;      ///< [limbs] d^-1 mod p.
        DeviceVector<Modulus64> modulus;
    };

    /**
     * @brief Batch matrix operations: PCMM, CMT and CCMM.
     *
     * Constructed from a generated CKKS context. Twiddle tables are built lazily
     * per level and cached for the lifetime of the operator.
     */
    template <> class HEBatchMatrixOperator<Scheme::CKKS>
    {
      public:
        HEBatchMatrixOperator(HEContext<Scheme::CKKS>& context,
                              const BatchMatrixLayout& layout);

        /// Out of line: half_ points at this same type, so the deleter needs
        /// the class complete and an implicit destructor would not have it.
        ~HEBatchMatrixOperator();

        /** @brief Subring layout this operator was built for. */
        const BatchMatrixLayout& layout() const noexcept { return layout_; }

        /**
         * @brief Upload an encoded plaintext matrix and transform it into the
         *        R_k NTT domain, ready to be a PCMM right operand.
         *
         * @param coeffs Output of BatchMatrixEncoder::encode for a rows x cols
         *               matrix.
         * @param scale  Scaling factor the coefficients were encoded with.
         */
        void encode_plaintext_matrix(const std::vector<int64_t>& coeffs,
                                     int rows, int cols, int depth,
                                     double scale);

        /**
         * @brief Upload a plaintext matrix that is the SAME real matrix for
         *        every one of the k/2 batched instances.
         *
         * This is the only kind of plaintext a model weight ever is. The batch
         * axis carries k/2 independent INPUTS through one model, so a
         * projection's weight does not vary along it -- and Definition 1 puts
         * the batch index in the EVALUATION domain of R_k, so an entry that is
         * constant across the batch inverts to the CONSTANT polynomial.
         * BatchMatrixEncoder::encode returns exactly that: v at Y^0 and zero at
         * every other power, because the constant polynomial is the unique
         * preimage of a constant vector under a bijective transform.
         *
         * Taking the general route to discover that is expensive. encode() runs
         * an O(k^2) transform per entry to produce a delta;
         * encode_plaintext_matrix() then stores and transforms k coefficients
         * per entry where one carries the information; and pcmm() moves four
         * [limb][row][col][k] tensors through the subring domain in order to
         * multiply by it. Here the host writes one rounded integer per entry,
         * the device lifts it into the RNS base, and nothing is transformed at
         * all.
         *
         * A subsequent pcmm() picks this form up automatically. Given the same
         * integers the two routes agree BIT FOR BIT -- the cancellation in
         * pcmm() is exact in Z_p, not approximate -- so the only question is
         * whether the two ENCODERS produce the same integers, and there this
         * one is the more accurate of the two rather than the equal of it.
         * BatchMatrixEncoder::encode sums k/2 double products per coefficient
         * and leaves a relative residue of ~1.8e-16 at the powers that should
         * be zero; llround kills it while scale * |w| < 2^51.5 and does not
         * above, which a 60-bit prime scale reaches. This route rounds the
         * value itself and is exact at every scale. **So do not write an
         * equality regression against the general encoder** -- it holds only
         * below that threshold, and where it fails this side is right.
         *
         * Real and batch-invariant are enforced by the signature rather than
         * by a check: the parameter is one real matrix, so there is no way to
         * express a per-instance or complex weight and reach this path by
         * accident. A caller that grows one must go back to
         * encode_plaintext_matrix, which still handles the general case.
         *
         * @param weight Row-major @p rows x @p cols real matrix, shared by the
         *               whole batch.
         * @param scale  Scaling factor; entry (i,j) is stored as
         *               llround(weight[i][j] * scale).
         */
        void encode_shared_plaintext_matrix(const std::vector<double>& weight,
                                            int rows, int cols, int depth,
                                            double scale);

        /**
         * @brief Whether the uploaded plaintext is in the shared form.
         *
         * True after encode_shared_plaintext_matrix, false after
         * encode_plaintext_matrix. Exposed so a test can pin that the two
         * routes agree rather than assume it.
         */
        bool plaintext_is_shared() const noexcept { return plain_shared_; }

        /**
         * @brief Batch PCMM, Algorithm 1.
         *
         * Treats @p in as the columns of a matrix encryption (Definition 2) and
         * right-multiplies by the matrix most recently uploaded through
         * encode_plaintext_matrix. The whole operation is two matrix products
         * over R_{q,k}, one per ciphertext component, with no rotations and no
         * key switching.
         *
         * @param out     Receives cols ciphertexts.
         * @param in      rows ciphertexts forming the matrix encryption.
         * @param rescale Perform step 2 of Algorithm 1. Pass false to inspect
         *                the unscaled product.
         */
        void pcmm(std::vector<Ciphertext<Scheme::CKKS>>& out,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                  bool rescale = true);

        /**
         * @brief Where the block index of a rectangular operand is carried.
         *
         * A rectangular d x (N/2) matrix is k/2 blocks of d x d, and the block
         * index has to live somewhere inside the R_k entry. There are two
         * consistent places and Algorithm 5 does not treat them alike.
         */
        enum class BlockAxis
        {
            /// Block l sits in batch slot l, i.e. in the EVALUATION domain of
            /// R_k: entry (i,j) evaluates to M_l[i][j] at zeta^{5^l}. This is
            /// the encoding of Definition 1 and the one Algorithm 5 is stated
            /// for. Theorem 3's constant term is then (2/k) times the sum over
            /// blocks, so the product is scaled by k/2 to recover the sum.
            slot,
            /// Block t sits in the Y^t COEFFICIENT of entry (i,j). This is the
            /// form Algorithm 5 hands back, and feeding it straight back in is
            /// what makes a chain of projections possible: the constant
            /// coefficient of an R_k product is already the contraction over
            /// t, provided the plaintext carries its own blocks reversed and
            /// negated (Y^k = -1). No scaling, and no conversion between
            /// calls -- the whole cost is a host-side arrangement of the
            /// plaintext, which is free.
            coefficient
        };

        /**
         * @brief Rectangular batch PCMM, Algorithm 5.
         *
         * Multiplies a d x (N/2) matrix encryption by the (N/2) x (N/2)
         * plaintext matrix most recently uploaded through
         * encode_plaintext_matrix as a d x (N/2) batch matrix. Both operands
         * are partitioned into d x d blocks; the block products are one batch
         * PCMM, and the k/2 partial products are then summed by extracting
         * constant terms (Theorem 3).
         *
         * The summation is where the CMT enters and where the cost of this
         * algorithm lives: Algorithm 1 needs no key switching at all, while
         * this needs N/2 of them per call, one per intermediate column. What
         * it buys is that the contraction runs over the batch axis, so a
         * SINGLE input occupies the whole ring instead of k/2 independent
         * ones, and an activation is d ciphertexts rather than N/2.
         *
         * @param out        Receives layout.d ciphertexts, the same shape and
         *                   encoding as @p in when @p axis is
         *                   BlockAxis::coefficient.
         * @param in         Exactly layout.d ciphertexts.
         * @param galois_key Must carry get_rectangular_rotation_indices().
         * @param axis       Where the block index of @p in is carried; see
         *                   BlockAxis. The OUTPUT is always
         *                   BlockAxis::coefficient.
         * @param rescale    Mark the result for rescaling, as Algorithm 1's
         *                   step 2 does.
         */
        void rectangular_pcmm(std::vector<Ciphertext<Scheme::CKKS>>& out,
                              const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              HEArithmeticOperator<Scheme::CKKS>& ops,
                              BlockAxis axis = BlockAxis::slot,
                              bool rescale = true);

        /**
         * @brief Transform the given ciphertexts into the R_k NTT domain and
         *        straight back, in place.
         *
         * The subring transform is a pure change of basis, so this is the
         * identity on any well-formed ciphertext and must recover the input
         * bit-exactly. Exposed as the diagnostic for the transform pair, which
         * is otherwise only reachable from inside pcmm.
         */
        void subring_round_trip(
            const std::vector<Ciphertext<Scheme::CKKS>*>& ct);

        /**
         * @brief Stage a raw R_N coefficient vector into a plaintext.
         *
         * A matrix encryption (Definition 2) encrypts m_j = sum_i M[i][j] X^i,
         * which is a coefficient-domain object and not the image of any slot
         * encoding, so it cannot be produced by HEEncoder. This takes the
         * centered integer coefficients directly, reduces them into the active
         * RNS base and transforms them into the NTT domain, leaving a plaintext
         * that HEEncryptor accepts unchanged.
         *
         * @param coeffs Exactly N centered coefficients.
         */
        void load_coefficients(Plaintext<Scheme::CKKS>& plain,
                               const std::vector<int64_t>& coeffs,
                               double scale);

        /**
         * @brief Inverse of load_coefficients: recover centered coefficients
         *        from a plaintext.
         *
         * Works at any level: the active limbs are recombined by Garner's
         * algorithm and the result is centered about the product of the active
         * primes. That product is not bounded by any builtin integer type, so
         * the reconstruction runs in a big integer even though the answer is
         * returned as int64.
         *
         * @throws std::runtime_error if a centered coefficient does not fit an
         *         int64, which means the plaintext has outgrown the range this
         *         can report rather than that any limb count is unsupported.
         */
        void extract_coefficients(std::vector<int64_t>& coeffs,
                                  Plaintext<Scheme::CKKS>& plain);

        /**
         * @brief Diagnostic: forward then inverse length-N NTT on a raw
         *        coefficient vector, with no Plaintext involved.
         *
         * Isolates the transform pair from plaintext storage handling. Returns
         * the input if the context's NTT tables and configuration are used
         * correctly.
         */
        std::vector<int64_t>
        ntt_intt_probe(const std::vector<int64_t>& coeffs);

        /**
         * @brief Multiply a ciphertext by the monomial X^power, in place.
         *
         * Exact: no key switching and no transform round trip. The building
         * block of the Algorithm 2 TWEAK and of steps 1 and 5 of Algorithm 3.
         * @param power Exponent, reduced modulo 2N; negative values are folded.
         */
        void mult_monomial(Ciphertext<Scheme::CKKS>& ct, int power);

        /**
         * @brief The TWEAK of Algorithm 2.
         *
         * Replaces d ciphertexts encrypting {m_j} with ciphertexts encrypting
         * m'_i = sum_j X^(2kij*sgn) m_j, by a Cooley-Tukey recursion over
         * monomial multiplications and additions. Involves no key switching, so
         * it is exact.
         *
         * @param ct  Exactly layout.d ciphertexts, transformed in place.
         * @param sgn +1 for the forward pass, -1 for the inverse pass. Running
         *            both in sequence scales by d.
         */
        void tweak(std::vector<Ciphertext<Scheme::CKKS>>& ct, int sgn);

        /**
         * @brief Batch ciphertext matrix transpose (CMT), Algorithm 3.
         *
         * Takes the d column ciphertexts of a matrix encryption of MatEcd({M_l})
         * and replaces them, in place, with a matrix encryption of
         * MatEcd({M_l^T}). Equivalently it converts between column-wise and
         * row-wise matrix encryption.
         *
         * @param galois_key Must carry the indices from
         *                   get_batch_cmt_rotation_indices(layout).
         */
        void cmt(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                 Galoiskey<Scheme::CKKS>& galois_key,
                 HEArithmeticOperator<Scheme::CKKS>& ops);

        /**
         * @brief Which orientation @p b is already in when it reaches ccmm.
         *
         * Algorithm 4's step 1 needs the right operand ROW-wise and gets there
         * with a CMT. A caller that already holds the row-wise form -- or that
         * holds a column-wise encryption of the matrix it wants transposed --
         * has that CMT for free, and paying for it again is paying for the
         * identity.
         */
        enum class RightOperandForm
        {
            /// @p b encrypts B column-wise, as Definition 2 and every other
            /// entry point here produce it. ccmm transposes it itself.
            column_wise,
            /// @p b IS the row-wise encryption step 1 would have produced, so
            /// step 1 is skipped. Because a row-wise encryption of B is the
            /// same data as a column-wise encryption of B^T, this is how a
            /// caller asks for A * B^T while handing over a column-wise B:
            /// pass B here and the result is A * B^T.
            ///
            /// The saving is not only the d - 1 rotations of one CMT. It also
            /// drops the deep copy of d ciphertexts step 1 makes, and it
            /// LOWERS the error: the CMT's key-switching noise is multiplied
            /// by the left operand in the GEMM and is the dominant term in the
            /// product's error, as the reference test records.
            row_wise
        };

        /**
         * @brief Batch CCMM, Algorithm 4.
         *
         * Multiplies two matrix encryptions. Writing A = a_0 + a_1 * s and
         * B for the matrix the second operand encrypts, the product is
         * A * B = a_0 * B + s * (a_1 * B); each half is obtained by treating
         * one component of the left operand as a plaintext matrix, so the
         * whole operation is four matrix products over R_{q,k} followed by a
         * single relinearisation per output column.
         *
         * Step 1 turns @p b into a row-wise matrix encryption with a CMT; the
         * transpose that implies is then absorbed into the GEMM strides rather
         * than paid for as a data movement. @p b_form skips step 1 for a
         * caller that already holds that form.
         *
         * The other two CMTs, the ones inside the fold, are NOT removable and
         * it is worth recording why, because the reordering that would remove
         * one of them is arithmetically inviting and silently wrong. They
         * transpose each half BEFORE the combine multiplies the second half by
         * the secret. Under R_N = R_k^d that multiplication is Toep(s) acting
         * on the LEFT, and the transpose the CMT performs is R_k-linear but
         * not R_N-linear: T(Toep(s) * M) = M^T * Toep(s)^T, not
         * Toep(s) * M^T. So combining the four GEMM outputs into one
         * degree-two ciphertext and transposing once at the end computes a
         * different matrix, and it agrees with this one exactly when the left
         * operand's c_1 is zero -- which is to say, on trivial encryptions,
         * which is exactly what a test that cannot see the bug is built from.
         *
         * @param out        Receives d ciphertexts, the column-wise matrix
         *                   encryption of the product.
         * @param a,b        Exactly layout.d ciphertexts each, at a common
         *                   level.
         * @param galois_key Must carry the indices from
         *                   get_batch_cmt_rotation_indices(layout); the
         *                   internal CMTs need them.
         * @param relin_key  Relinearisation key for the degree-two part.
         * @param rescale    Mark the result for rescaling. Pass false to
         *                   inspect the product at the combined scale.
         * @param b_form     Whether @p b still needs step 1's transpose.
         */
        void ccmm(std::vector<Ciphertext<Scheme::CKKS>>& out,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& a,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& b,
                  Galoiskey<Scheme::CKKS>& galois_key,
                  Relinkey<Scheme::CKKS>& relin_key,
                  HEArithmeticOperator<Scheme::CKKS>& ops,
                  bool rescale = true,
                  RightOperandForm b_form = RightOperandForm::column_wise);

      private:
        const BatchSubringTables& tables_for(int depth);

        /**
         * @brief The CMT's per-index rotation and reordering schedule.
         *
         * @c rot_index[t] is the slot-rotation index of the automorphism
         * X -> X^(2kt+1); @c source[t] is the t* the permutation reads from.
         */
        struct CmtSchedule
        {
            std::vector<int> rot_index;
            std::vector<int> source;
        };

        /**
         * @brief The schedule, built on first use and cached.
         *
         * It is a function of @c n_, @c layout_.d and @c layout_.k, which are
         * fixed at construction and never reassigned, so there is no cache key
         * -- unlike tables_for, which genuinely varies with the level. The
         * cache must stay per-operator: half_operator() is a distinct object
         * at a distinct layout and needs its own.
         *
         * Recomputing it per call cost an n_/2-entry std::map walk of the
         * whole rotation group to serve d lookups, on the host, with the
         * device idle.
         */
        const CmtSchedule& schedule();

        /// Algorithm 1 against a plaintext uploaded by
        /// encode_shared_plaintext_matrix. Validation lives in pcmm, which
        /// dispatches here once the operands have been checked.
        void pcmm_shared(std::vector<Ciphertext<Scheme::CKKS>>& out,
                         const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                         bool rescale);

        /// The operator at the layout (N, N/2), built on first use. Algorithm
        /// 5's summation is a CMT read at k = 2, and cmt()/tweak()/tables_for()
        /// are all bound to layout_, so the second reading needs its own
        /// operator rather than an extra argument threaded through all three.
        HEBatchMatrixOperator<Scheme::CKKS>& half_operator();

        /// Multiply every ciphertext by a small non-negative integer, exactly.
        void mult_int_scalar_batch(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                                   uint64_t value);

        /**
         * @brief Multiply each of @p ct by its own monomial, in one launch.
         *
         * @param powers One exponent per ciphertext; zero entries are skipped
         *               on the device, so callers need not filter them out.
         */
        void mult_monomial_batch(const std::vector<Data64*>& ct,
                                 const std::vector<int>& powers, int depth);

        /**
         * @brief Metadata clone of @p src with uninitialised device memory.
         *
         * A copy-construct would deep-copy coefficients that the caller is
         * about to overwrite in full.
         */
        Ciphertext<Scheme::CKKS> allocate_like(const Ciphertext<Scheme::CKKS>&
                                                   src,
                                               size_t elems) const;

        /// [limb][j] -> start of limb @p l inside one component of ciphertext
        /// @p j, the addressing every batch matrix kernel expects.
        std::vector<Data64*>
        component_pointers(const std::vector<Data64*>& base, int count,
                           bool second_component, int num_limbs) const;

        HEContext<Scheme::CKKS> context_;
        BatchMatrixLayout layout_;
        int n_;
        int q_size_;

        std::map<int, BatchSubringTables> table_cache_;
        std::unique_ptr<HEBatchMatrixOperator<Scheme::CKKS>> half_;

        /// @see schedule(). Empty until first use.
        CmtSchedule schedule_;

        // Most recently uploaded plaintext matrix.
        DeviceVector<Data64> plain_;
        int plain_rows_ = 0;
        int plain_cols_ = 0;
        int plain_depth_ = -1;
        double plain_scale_ = 0.0;
        /// Set by encode_shared_plaintext_matrix, cleared by
        /// encode_plaintext_matrix. When set, plain_ holds
        /// [limb][row][col] scalars rather than [limb][row][col][k] subring
        /// elements, and pcmm takes the scalar path.
        bool plain_shared_ = false;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_BATCHMATRIX_H
