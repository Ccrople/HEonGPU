// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// THE BAE PRODUCT ON A LOW RING — SYLPH'S CONFIGURATION
// =====================================================
//
// Sylph (arXiv 2601.18511v2), Table IV, row 1:
//
//     Operator | Method | Ring Degree | Encoding | Packing Layout
//     PCMM     |  [27]  |     256     |  Coeff   |   Row-split
//
// against 65536 for the slot-encoded non-linear layers in the same table. The
// PCMM is not run on the pipeline's ring at all. It is run 256 times smaller,
// and it is ring switching that carries the ciphertext there:
//
//     "To connect different ring degrees, we utilize the ring-switching
//      technique [41] mostly with coefficient encoding"
//
// This file is that configuration, built out of the two pieces the branch
// already has — HERingSwitchOperator and HEBaePcmmOperator — plus the one
// thing that makes them compose, which is that their layouts already agree.
//
// WHY IT IS THE ONLY LEVER, AND IT IS A ONE-LINE ARGUMENT
// -------------------------------------------------------
// baepcmm.cuh derives the cost of U (d1 x d2) times encrypted M (d2 x d3):
//
//     b-part GEMM   d1 * d2 * d3
//     a-part GEMM   d1 * d2 * N        <- the ring degree, always
//
// Say it once more with the right name on N. The a-part is the ciphertexts'
// a-vectors, and an a-vector is exactly as long as the secret it pairs with.
// So N there is **the lattice dimension**, and the whole cost model is
//
//     MACs per (in-channel, out-channel, token) per RNS limb  =  (D + d3) / T
//
// with D the lattice dimension, d3 the encoded column count and T the tokens
// actually carried. **D and T are the only two levers there are.** Nothing
// about the packing, the encoding, the MLWE rank or the ModPack schedule
// appears in that expression, and none of them can.
//
// T is the lever Sylph's other half pulls ("longer inputs", thousands of
// tokens). At B = 1 with a 128-token prompt T is 128 and cannot be argued
// with. D is what is left, and D is a *security* parameter — which is why
// this file has a parameter section and not just a benchmark.
//
// MODDECOMP DOES NOT MOVE D. RING SWITCHING DOES. THAT IS THE WHOLE POINT
// -----------------------------------------------------------------------
// There are two ways to see a degree-N ciphertext as degree-n objects, and
// this branch now has both. They are not variants of one thing:
//
//   ModDecomp (Bae App. A; pcmm_mlwe/pcmm_packed here)
//       Free. No key, no noise, no level. Splits the coefficients by residue
//       class into a rank-k MLWE ciphertext over R_n, k = N/n.
//       **The rank is kept, so k*n = N and the lattice dimension is still N.**
//       The a-part is k component polynomials of n coefficients each — still
//       N coefficients — so the a-part GEMM is still d1*d2*N. It buys
//       ciphertext count, not arithmetic.
//
//   Ring switching (Sylph §3.3; HERingSwitchOperator here)
//       One key switch per ciphertext, zero levels. Key-switches to the
//       embedded secret s'(X^k) FIRST, and only then splits — so each piece
//       is a rank-1 RLWE ciphertext under a genuine degree-n secret.
//       **The lattice dimension really is n now.** The a-part GEMM becomes
//       d1*d2*n, a factor of k.
//
// One key switch per ciphertext is the entire price of a k-fold cut in the
// dominant GEMM. It also *removes* key switches on the other side: the
// big-ring path owes ModPack d1 of them per projection, the low-ring path
// owes (d1 + d2)/k for the descent and the ascent together.
//
// PARAMETERS AND SECURITY, WHICH ARE THE SAME QUESTION HERE
// ---------------------------------------------------------
// Because the dimension really drops, n is not a free parameter. It is bounded
// below by the modulus the product has to run under, and the library owns that
// bound — heongpu_128bit_std_parms in secstdparams.h, checked on log(P*Q):
//
//     N = 1024   27 bits      N = 8192    218 bits
//     N = 2048   54 bits      N = 16384   438 bits
//     N = 4096   109 bits     N = 65536  1761 bits
//
// A Bae product needs no key switch of its own (k = 1 needs no ModPack, and
// nothing else here rotates), so the small context needs Q and the one special
// prime the library insists on, and Q needs two primes: one to carry the value
// and one for the product's own rescale to spend. That is the floor, and
// benchmark/profile_bae_ring_security.cpp asks the library itself which
// (N, chain) pairs clear it rather than re-deriving the table here.
//
// THREE FLOORS, MEASURED, AND THEY DO NOT AGREE
// ----------------------------------------------
//   1. defines.h: MIN_POLY_DEGREE = 4096. **Sylph's 256 is not a parameter
//      choice in this library, it is not representable** — every context below
//      4096 dies with "Poly modulus degree is not supported", at every
//      security level. This floor fires before security is consulted at all.
//   2. sec128 with uniform primes: 4096 accepts two 30-bit Q primes and a
//      30-bit special prime (log PQ 90 <= 109), i.e. one usable level. So it
//      clears the security table, but only at 30-bit primes — which is also
//      the library's MIN_USER_DEFINED_MOD_BIT_COUNT, so there is nothing below.
//   3. THE ONE THAT BINDS: the shared-prefix rule means the small ring
//      inherits the pipeline's own primes, and the logN 16 chain starts at
//      q0 = 41 (§17.2). With |P| >= q0 that is 41 + 33 + 41 = 115 > 109, so
//      **4096 holds one prefix prime and therefore zero usable levels; the
//      floor for the chain as it stands is 8192**, which holds five.
//
// Ring 4096 opens at exactly q0 = 38 and not at 41 — measured, not derived —
// and HEonGPU's CKKS bootstrap is built around q0 / scale ~ 2^10, so at a
// 33-bit scale it wants q0 >= 43. **The PCMM's floor ring is set by the
// bootstrap's bottom prime**, three subsystems away.
//
// **Sylph's 256 clears none of the three, and the paper does not claim it
// does.** The arXiv HTML contains no security analysis, no bit-security claim
// and no lattice-estimator citation to weigh against that. Two readings are
// possible and they have different costs:
//
//   * the degree-256 objects keep the module rank (MLWE, dimension preserved).
//     Secure, but then it is ModDecomp, and by the paragraph above the a-part
//     GEMM never shrank — so the 256 in Table IV would be buying packing, not
//     arithmetic.
//   * they are rank-1 RLWE at degree 256, which is what "ring-switching" plus
//     "we utilize [41]" reads as, and what makes the arithmetic claim work.
//     Then the dimension is 256 and it is not a 128-bit parameter set.
//
// This file implements the second — the one with the arithmetic in it — and
// stops the ring where the library stops it. The win survives the correction:
// 65536 -> 8192 is k = 8, and measured on an A6000 at width 512 it is
// **4.84x on a block's seven projections** (620 ms each at the big ring
// against 32 + 7*120 + 26 ms through the descent), at parameters that generate
// at sec_level_type::sec128.
//
// THE LAYOUTS ALREADY AGREE, WHICH IS WHY THIS FILE IS SHORT
// -----------------------------------------------------------
// Sylph's "Row-split" is BaeLayout with cols = n:
//
//     row r = k*i + u of M lands in ciphertext i, column t at coefficient t*k+u
//
// and HERingSwitchOperator's DOWN split is
//
//     m_j[i] = m[i*k + j].
//
// Substituting, piece u of big ciphertext i is the polynomial whose t-th
// coefficient is row (k*i + u), column t — **exactly one matrix row, whole**.
// So the row-split encoder at the big ring and the ring switch's own splitting
// are the same index map, and the descent needs no glue at all: k rows in, k
// rank-1 small ciphertexts out, one per row. The ascent is the mirror.
//
// One consequence worth stating because it is where the time goes: a chain of
// projections **stays down**. Q, K, V, O and the three FFN matrices are all
// row -> row, so a block descends once and ascends once, not once per product.

#ifndef HEONGPU_CKKS_BAE_LOWRING_H
#define HEONGPU_CKKS_BAE_LOWRING_H

#include <heongpu/host/ckks/baepcmm.cuh>
#include <heongpu/host/ckks/ringswitch.cuh>

#include <memory>
#include <vector>

namespace heongpu
{
    /**
     * @brief The Bae plaintext-ciphertext product, taken on a smaller ring.
     *
     * Owns two HEBaePcmmOperators because the two rings play different parts:
     *
     *   encoder()  BaeLayout(N_big, cols = N_small), k = N_big/N_small. Used
     *              ONLY for its encode/decode index map — the row-split layout
     *              a big-ring ciphertext must be in for the descent to hand
     *              back whole rows. No product is ever taken with it.
     *   product()  BaeLayout(N_small, cols = N_small), k = 1. The product
     *              itself, transform-free, no ModDecomp, no ModPack, no key of
     *              any kind at the small ring.
     */
    class HEBaeLowRingPcmm
    {
      public:
        /**
         * @param big   The pipeline's context.
         * @param small The product's context. Its Q chain must be a
         *              value-identical prefix of the big one — the same
         *              requirement HERingSwitchOperator already checks, and
         *              @p rs having been constructed is the proof of it.
         * @param rs    Ring switch bound to exactly these two contexts, with
         *              its keys already generated.
         *
         * Rejects N_small * N_small < N_big: that is Bae's own d >= sqrt(N)
         * floor applied to the big-ring row-split layout, and it is the reason
         * 256 is the smallest ring a 65536-ring pipeline can split into
         * this way — 256 is exactly sqrt(65536), so Sylph's Table IV entry
         * sits on the constraint rather than near it.
         */
        HEBaeLowRingPcmm(HEContext<Scheme::CKKS>& big,
                         HEContext<Scheme::CKKS>& small,
                         HERingSwitchOperator<Scheme::CKKS>& rs);

        int k() const noexcept { return k_; }
        int big_ring() const noexcept { return n_big_; }
        int small_ring() const noexcept { return n_small_; }

        HEBaePcmmOperator<Scheme::CKKS>& encoder() noexcept { return *encoder_; }
        HEBaePcmmOperator<Scheme::CKKS>& product() noexcept { return *product_; }

        /**
         * @brief Big-ring row-split ciphertexts -> one small ciphertext per row.
         *
         * @param rows_big d2/k ciphertexts in the row-split layout, NTT domain,
         *                 no pending rescale or relinearisation, at a level
         *                 inside the shared prefix.
         * @return d2 small-ring ciphertexts, row r at index r.
         *
         * One key switch per INPUT ciphertext, i.e. d2/k for the whole
         * activation, and zero levels.
         */
        std::vector<Ciphertext<Scheme::CKKS>>
        descend(std::vector<Ciphertext<Scheme::CKKS>>& rows_big,
                HEArithmeticOperator<Scheme::CKKS>& big_ops);

        /**
         * @brief One small ciphertext per row -> big-ring row-split.
         *
         * The inverse of descend(); rows must be a multiple of k and all
         * inputs must share a level and a scale. One key switch per OUTPUT
         * ciphertext, d1/k in total, and zero levels.
         */
        std::vector<Ciphertext<Scheme::CKKS>>
        ascend(std::vector<Ciphertext<Scheme::CKKS>>& rows_small,
               HEArithmeticOperator<Scheme::CKKS>& big_ops);

        /**
         * @brief A projection, entirely at the small ring. One level.
         *
         * Takes the weight in the same transposed row-major form
         * HEBaePcmmOperator::project and Llama3RectOperator::project take
         * (@p in_channels x @p out_channels), and spends the rescale, so the
         * output lands on the input's own scale one level down.
         *
         * @param out Receives out_channels small ciphertexts, one per row.
         *
         * Chains: out is in exactly the form in is, so a second projection may
         * be taken on it without ascending.
         */
        void project(std::vector<Ciphertext<Scheme::CKKS>>& out,
                     std::vector<Ciphertext<Scheme::CKKS>>& in,
                     const std::vector<double>& weight, int in_channels,
                     int out_channels,
                     HEArithmeticOperator<Scheme::CKKS>& small_ops);

        /**
         * @brief Modular multiply-accumulates and key switches for ONE
         *        projection, per RNS limb. Counted, not measured.
         *
         * @c per_token divides by the useful work — d1 * d2 * tokens — so it
         * is directly the "MACs per (in-channel, out-channel, token) per limb"
         * of §25.9's comparison table, and is the number to quote.
         */
        struct Cost
        {
            long long a_macs = 0;
            long long b_macs = 0;
            long long key_switches = 0;
            int dimension = 0;
            double per_token = 0.0;

            long long macs() const noexcept { return a_macs + b_macs; }
        };

        /// Today's path: ModDecomp + both GEMMs + ModPack, all at the big ring.
        static Cost big_ring_cost(int d1, int d2, int tokens, int n_big,
                                  int cols);

        /// Sylph's: descend, k == 1 product at the small ring, ascend.
        static Cost low_ring_cost(int d1, int d2, int tokens, int n_big,
                                  int n_small);

      private:
        HEContext<Scheme::CKKS> big_;
        HEContext<Scheme::CKKS> small_;
        HERingSwitchOperator<Scheme::CKKS>& rs_;

        int n_big_;
        int n_small_;
        int k_;

        std::unique_ptr<HEBaePcmmOperator<Scheme::CKKS>> encoder_;
        std::unique_ptr<HEBaePcmmOperator<Scheme::CKKS>> product_;
    };

} // namespace heongpu

#endif // HEONGPU_CKKS_BAE_LOWRING_H
