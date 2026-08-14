// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// THE BAE PLAINTEXT-CIPHERTEXT MATRIX PRODUCT
// ===========================================
//
// Bae, Cheon, Hanrot, Park and Stehle, "Plaintext-Ciphertext Matrix
// Multiplication and FHE Bootstrapping: Fast and Fused", CRYPTO 2024
// (eprint 2024/1284). Sections 4.1 and 4.2; Algorithms 1 and 2.
//
// This is NOT another packing. It is a different thing entirely from the
// Kang primitives in batchmatrix.cuh, and the difference is worth stating
// before any API, because every intuition carried over from Algorithm 5 is
// wrong here.
//
// THE IDENTITY
// ------------
// Lemma 3 of the paper: for u, s, v, w in R_{q,N},
//
//     u*s + v = w   <=>   Vec(u) * Toep(s) + Vec(v) = Vec(w),
//
// where Vec(.) is the raw coefficient row vector and Toep(s) is the negacyclic
// Toeplitz matrix whose i-th row is Vec(X^i * s). So RLWE decryption of a
// STACK of ciphertexts is literally a matrix identity over Z_q:
//
//     A * Toep(sk) + B = M,
//
// with A and B the raw coefficient arrays of the a-parts and b-parts, one row
// per ciphertext, and M the message matrix. Left-multiplying by a plaintext
// matrix U preserves the form:
//
//     (U*A) * Toep(sk) + (U*B) = (U*M).
//
// So (U*A, U*B) is already a valid ciphertext stack for U*M. The product is
// two ordinary dense GEMMs over Z_q on the ciphertext limbs.
//
// WHAT THAT BUYS, AND IT IS THE WHOLE POINT
// -----------------------------------------
//   rotations           0
//   automorphisms       0
//   Galois keys         0
//   relinearisations    0
//   monomial products   0
//   Hadamard products   0
//   levels              1, and the prime it spends is the plaintext scale,
//                       which may be a SMALL prime (the paper uses 2^18
//                       against a 2^58 chain) because it only has to carry
//                       the precision of U.
//
// Compare Algorithm 5 in batchmatrix.cuh, whose own doc comment reads "this
// needs N/2 of them per call, one per intermediate column". The Bae product
// replaces N/2 key switches with a GEMM.
//
// THE COST, AND IT IS WHY THE SHAPE MATTERS MORE THAN ANYTHING ELSE
// -----------------------------------------------------------------
// Write the product as U (d1 x d2) times M (d2 x d3), M encrypted. Then:
//
//     b-part GEMM   d1 x d2 x d3
//     a-part GEMM   d1 x d2 x N
//
// and THE a-PART GEMM DOES NOT SHRINK WITH d3. Its width is the ring degree,
// always. So with k = N / d3,
//
//     total work  =  d1 * d2 * (N + d3)  =  d1 * d2 * d3 * (k + 1),
//     work per column  =  d1 * d2 * (k + 1).
//
// k = 1 is therefore not a special case, it is THE case. At k = 1 the two
// GEMMs are the same size, ModDecomp and ModPack are both the identity, and
// the algorithm is exactly Liu-Zhang: two GEMMs and nothing else at all. At
// k = 32 the same product costs 16.5x more per column and additionally owes
// d1 key switches for the ModPack.
//
// d3 IS THE COLUMN COUNT OF THE ENCRYPTED MATRIX, and in a transformer that
// is the TOKEN axis (times the batch, if there is one), because U is the
// weight and it must act on the channel axis, which is therefore the row
// axis. The paper's constraint d3 >= sqrt(N) and the divisibility d3 | N are
// both checked in the constructor.
//
// So: this operator wants a WIDE activation. A batch widens it; a longer
// sequence widens it identically and for free. That is the opposite of the
// Kang path, where the token count is pinned to d by the ring and a batch is
// the only way to fill the ring.
//
// WHAT IS AND IS NOT IMPLEMENTED HERE
// -----------------------------------
//   k == 1  complete. RLWE in, RLWE out, no keys of any kind, one level.
//           This is the configuration the algorithm is good in and the one
//           the tests cover end to end against a decryption.
//   k >  1  the PRODUCT is implemented and tested (ModDecomp, both GEMMs,
//           the re-assembly); the result is returned as a rank-k MLWE stack.
//           ModPack back to degree-N RLWE is NOT implemented: it needs k
//           switching keys carrying the sub-secrets s_j (the X^j components
//           of sk), which is a key ceremony this library has no entry point
//           for. Its cost, if built, is k key switches per output ciphertext,
//           i.e. d1 in total. See pcmm_mlwe().
//
// ORIENTATION
// -----------
// A Llama projection is Y = X * W with X (tokens x in_channels). Bae computes
// U * M, so M = X^T (in_channels x tokens) and U = W^T (out_channels x
// in_channels). Both are the transposes the host already stores, so neither
// transpose costs anything. Right-multiplication is NOT available: Toep(sk)
// sits on the right of the identity and does not commute past a right factor.

#ifndef HEONGPU_CKKS_BAEPCMM_H
#define HEONGPU_CKKS_BAEPCMM_H

#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/secretkey.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/keygenerator.cuh>

#include <cstdint>
#include <memory>
#include <vector>

namespace heongpu
{
    /**
     * @brief Shape of a Bae matrix encryption.
     *
     * @c cols is d3, the column count of the ENCRYPTED matrix, and the only
     * shape parameter the encoding has. @c k = N / cols is the MLWE rank the
     * decomposition needs; @c rows_per_ciphertext is the same number, because
     * one degree-N ciphertext carries exactly k rows.
     */
    struct BaeLayout
    {
        int N = 0;
        int cols = 0;
        int k = 0;

        BaeLayout() = default;
        BaeLayout(int N_, int cols_);

        int rows_per_ciphertext() const noexcept { return k; }
    };

    /**
     * @brief A rank-k MLWE ciphertext stack, one MLWE ciphertext per row.
     *
     * The output of the product when k > 1. Row r is
     * ((a[r][j])_{j<k}, b[r]) over R_{q,cols}, satisfying
     * b[r] = -sum_j a[r][j] * s_j + m_r, where s_j is the X^j component of
     * the degree-N secret. Coefficient domain throughout: an MLWE component
     * has no NTT table in this library and does not need one, since nothing
     * multiplies it here.
     */
    struct BaeMlweStack
    {
        /// [row][component][limb][cols]
        std::vector<Data64> a;
        /// [row][limb][cols]
        std::vector<Data64> b;

        int rows = 0;
        int cols = 0;
        int k = 0;
        int limbs = 0;
        int depth = 0;
        double scale = 0.0;

        bool empty() const { return rows == 0; }
    };

    template <Scheme S> class HEBaePcmmOperator;

    template <> class HEBaePcmmOperator<Scheme::CKKS>
    {
      public:
        /**
         * @param context Generated CKKS context.
         * @param cols    d3, the column count of the encrypted matrix. Must
         *                divide N and satisfy cols * cols >= N, which is the
         *                paper's d >= sqrt(N).
         */
        HEBaePcmmOperator(HEContext<Scheme::CKKS>& context, int cols);

        const BaeLayout& layout() const noexcept { return layout_; }
        int cols() const noexcept { return layout_.cols; }
        int k() const noexcept { return layout_.k; }
        int ring_degree() const noexcept { return n_; }

        /**
         * @brief Map a rows x cols real matrix onto ring coefficients.
         *
         * The decimation of App. A, Eq. 14: row r = k*i + u of M lands in
         * ciphertext i, and its column t sits at coefficient t*k + u. So one
         * ciphertext carries k rows INTERLEAVED, not concatenated -- a
         * contiguous split would decrypt to a different matrix and symmetric
         * random test data cannot tell the two apart, which is why the tests
         * assert the interleave explicitly.
         *
         * @param matrix Row-major rows x cols.
         * @param scale  Scaling factor applied before rounding.
         * @return       rows/k vectors of N centred coefficients.
         */
        std::vector<std::vector<int64_t>>
        encode_matrix(const std::vector<double>& matrix, int rows,
                      double scale) const;

        /** @brief Inverse of encode_matrix. */
        std::vector<double>
        decode_matrix(const std::vector<std::vector<int64_t>>& coeffs,
                      int rows, double scale) const;

        /**
         * @brief Stage N centred coefficients into a plaintext.
         *
         * A Bae matrix encryption is a coefficient-domain object and is not
         * the image of any slot encoding, so HEEncoder cannot produce it.
         */
        void load_coefficients(Plaintext<Scheme::CKKS>& plain,
                               const std::vector<int64_t>& coeffs,
                               double scale) const;

        /**
         * @brief Recover centred coefficients from a plaintext, any level.
         */
        std::vector<int64_t>
        extract_coefficients(Plaintext<Scheme::CKKS>& plain) const;

        /**
         * @brief Upload the plaintext matrix U and lift it into the RNS base.
         *
         * U is never encoded as a polynomial, never transformed and never
         * encrypted -- it is an integer matrix multiplied straight into the
         * ciphertext limbs. That is the single largest structural difference
         * from every diagonal-method PCMM, and it is why this operator needs
         * no plaintext cache, no diagonal tables and no Galois indices.
         *
         * @param weight Row-major d1 x d2 real matrix.
         * @param depth  Level of the ciphertexts it will meet.
         * @param scale  Delta_1. Entry (i,j) is stored as
         *               llround(weight[i][j] * scale). The paper takes this
         *               MUCH smaller than the ciphertext scale (2^18 against
         *               2^58) because it carries only U's own precision; a
         *               caller that wants the product to land back on the
         *               input's scale passes the prime the following rescale
         *               will divide by, exactly as the Kang path does.
         */
        void upload_plaintext(const std::vector<double>& weight, int d1,
                              int d2, int depth, double scale);

        int plain_rows() const noexcept { return d1_; }
        int plain_cols() const noexcept { return d2_; }

        /**
         * @brief The product, k == 1 only: RLWE in, RLWE out.
         *
         * @param out     Receives d1 ciphertexts.
         * @param in      d2 ciphertexts, one per row of M, coefficient
         *                encoded, all at one level and one scale.
         * @param rescale Mark the result for rescaling, as Algorithm 1 step 8
         *                and Algorithm 2 step 12 do. The product sits at
         *                in_scale * plain_scale until it is spent.
         *
         * Throws if k != 1. There is no fallback: at k > 1 an output row is
         * an MLWE ciphertext and no degree-N ciphertext exists to return
         * until ModPack has run.
         */
        void pcmm(std::vector<Ciphertext<Scheme::CKKS>>& out,
                  const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                  bool rescale = true);

        /**
         * @brief The product for any k, returning the rank-k MLWE stack.
         *
         * Steps 1-4 of Algorithm 2: ModDecomp (free), the two GEMMs, and the
         * re-assembly. Step 5, ModPack, is the caller's problem and is not
         * implemented -- see the header note. The returned stack is what a
         * ModPack would consume.
         *
         * At k == 1 this returns the same numbers pcmm() does, one row per
         * ciphertext with a single a-component, and is used by the tests to
         * pin the two paths against each other.
         */
        BaeMlweStack
        pcmm_mlwe(const std::vector<Ciphertext<Scheme::CKKS>*>& in);


        /**
         * @brief Generate the k switching keys ModPack needs.
         *
         * ModPack (Algorithm 2 step 5) has to turn k rank-k MLWE rows back
         * into one degree-N RLWE ciphertext. After the product the k output
         * a-vectors no longer share the "shifted decimation of one alpha"
         * structure ModDecomp gave the input, so they cannot simply be
         * re-interleaved; each component polynomial has to be MULTIPLIED by
         * the sub-secret it pairs with, and that is a key switch.
         *
         * The sub-secret s_j is the X^j component of sk, embedded back into
         * R_N as s_j(X^k) -- coefficients at multiples of k, exactly the
         * embedded-secret shape HERingSwitchOperator already uses. It is
         * ternary because sk is, so it is a legal Secretkey.
         *
         * THIS IS A KEY-CEREMONY CHANGE AND IT IS WHY THE CALLER MUST SUPPLY
         * sk's COEFFICIENTS. Secretkey stores NTT-domain RNS residues and
         * cannot be read back, so there is no way to derive s_j from an
         * already-generated key. Build the master secret with
         * Secretkey(coefficients, context) and pass the same vector here.
         *
         * @param keygen          Key generator of this context.
         * @param sk              The master secret, the key switches' target.
         * @param sk_coefficients sk's N ternary coefficients.
         */
        void generate_modpack_keys(HEKeyGenerator<Scheme::CKKS>& keygen,
                                   Secretkey<Scheme::CKKS>& sk,
                                   const std::vector<int>& sk_coefficients);

        bool modpack_keys_generated() const noexcept
        {
            return static_cast<int>(modpack_keys_.size()) == layout_.k;
        }

        /**
         * @brief The complete Algorithm 2 for any k: RLWE in, RLWE out.
         *
         * ModDecomp, both GEMMs, the re-assembly, the rescale mark, and
         * ModPack. At k == 1 ModPack is the identity and no switching key is
         * touched, so this is pcmm() and needs no key generation; above that
         * it costs k key switches per output ciphertext, d1 in total, and
         * generate_modpack_keys must have run.
         *
         * @param out Receives d1 / k ciphertexts -- one per group of k output
         *            rows, in the same decimated encoding as the input.
         */
        void pcmm_packed(std::vector<Ciphertext<Scheme::CKKS>>& out,
                         const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                         HEArithmeticOperator<Scheme::CKKS>& ops,
                         bool rescale = true);

        /**
         * @brief Cost model, in modular multiply-accumulates per RNS limb.
         *
         * Exposed because the shape decision this operator forces is made by
         * the caller and should be made on arithmetic rather than on a
         * benchmark of one shape. Returns {a_part, b_part}.
         */
        static std::pair<uint64_t, uint64_t> gemm_macs(int d1, int d2, int d3,
                                                       int N);

      private:
        void run_gemms(const std::vector<Ciphertext<Scheme::CKKS>*>& in,
                       DeviceVector<Data64>& a_out, DeviceVector<Data64>& b_out,
                       int& limbs, int& depth, double& scale);

        /// Held by value: HEContext is a shared_ptr handle, so this is a
        /// reference count and not a copy of the context.
        HEContext<Scheme::CKKS> context_;
        BaeLayout layout_;
        int n_;
        int n_power_;
        int q_size_;

        /// U lifted into the RNS base, [limb][d1][d2]. Rebuilt per upload.
        DeviceVector<Data64> plain_;
        int d1_ = 0;
        int d2_ = 0;
        int plain_depth_ = -1;
        double plain_scale_ = 0.0;

        /// One switching key per sub-secret s_j, empty until
        /// generate_modpack_keys runs. Never needed at k == 1.
        std::vector<std::unique_ptr<Switchkey<Scheme::CKKS>>> modpack_keys_;
    };

} // namespace heongpu

#endif // HEONGPU_CKKS_BAEPCMM_H
