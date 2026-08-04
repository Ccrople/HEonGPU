// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Batch matrix encoding, batch matrix encryption and the batch matrix
// multiplication primitives PCMM, CMT and CCMM.
//
// Implements Definition 1 (batch matrix encoding), Definition 2 (batch matrix
// encryption), Algorithm 1 (batch CPMM), Algorithm 2 (TWEAK), Algorithm 3
// (batch CMT) and Algorithm 4 (batch CCMM) of Cheon, Kang and Lee, "Fast Batch
// Matrix Multiplication in Ciphertexts".
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
         * than paid for as a data movement.
         *
         * @param out        Receives d ciphertexts, the column-wise matrix
         *                   encryption of the product.
         * @param a,b        Exactly layout.d ciphertexts each, at a common
         *                   level.
         * @param galois_key Must carry the indices from
         *                   get_batch_cmt_rotation_indices(layout); the three
         *                   internal CMTs need them.
         * @param relin_key  Relinearisation key for the degree-two part.
         * @param rescale    Mark the result for rescaling. Pass false to
         *                   inspect the product at the combined scale.
         */
        void ccmm(std::vector<Ciphertext<Scheme::CKKS>>& out,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& a,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& b,
                  Galoiskey<Scheme::CKKS>& galois_key,
                  Relinkey<Scheme::CKKS>& relin_key,
                  HEArithmeticOperator<Scheme::CKKS>& ops,
                  bool rescale = true);

      private:
        const BatchSubringTables& tables_for(int depth);

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

        // Most recently uploaded plaintext matrix.
        DeviceVector<Data64> plain_;
        int plain_rows_ = 0;
        int plain_cols_ = 0;
        int plain_depth_ = -1;
        double plain_scale_ = 0.0;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_BATCHMATRIX_H
