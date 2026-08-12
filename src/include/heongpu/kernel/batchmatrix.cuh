// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Device kernels for the batch matrix primitives. All buffers use the layout
// [limb][row][col][k] with the length-k transform index fastest.

#ifndef HEONGPU_KERNEL_BATCHMATRIX_H
#define HEONGPU_KERNEL_BATCHMATRIX_H

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

namespace heongpu
{
    /**
     * @brief Lift centred int64 coefficients into the RNS base, one limb per
     * grid row.
     *
     * out[limb][i] = coeffs[i] mod modulus[limb], with a negative coefficient
     * mapped to p - (-v mod p) and -0 mapped to 0.
     *
     * The plaintext matrix is one array of coefficients repeated across every
     * limb, so building it limb-by-limb on the host meant writing num_limbs
     * copies of it through host memory and uploading all of them: 960 MiB per
     * projection at the 8B shape, against 64 MiB for the coefficients alone.
     *
     * One thread per coefficient, walking every limb, so the coefficient is
     * read once and the launch is one-dimensional.
     *
     * @param out      [num_limbs][per_limb] destination.
     * @param coeffs   [per_limb] centred coefficients, shared by every limb.
     *                 May alias out's first limb, which is how the caller
     *                 expands in place; each thread loads its element before
     *                 the loop stores, and owns index i in every limb.
     */
    __global__ void bm_crt_expand_kernel(Data64* out, const int64_t* coeffs,
                                         const Modulus64* modulus,
                                         size_t per_limb, int num_limbs);

    /**
     * @brief Forward negacyclic NTT of length k, one transform per block.
     *
     * Output lands in bit-reversed order, matching the ordering that
     * bm_ntt_to_subring_kernel produces for the ciphertext operand, so the two
     * GEMM operands agree. bm_intt_k_kernel undoes it.
     */
    __global__ void bm_ntt_k_kernel(Data64* data, const Data64* psi,
                                    const Modulus64* modulus, int k,
                                    int transforms_per_limb);

    /** @brief Inverse of bm_ntt_k_kernel; restores natural coefficient order. */
    __global__ void bm_intt_k_kernel(Data64* data, const Data64* psi_inv,
                                     const Data64* kinv,
                                     const Modulus64* modulus, int k,
                                     int transforms_per_limb);

    /**
     * @brief Length-N NTT domain -> R_k NTT domain, in one pass.
     *
     * Writing j = j_hi*d + j_lo, the length-N transform satisfies
     *     out[j] = m(psi^(2*brv_N(j)+1)),  brv_N(j) = brv_k(j_hi) + k*brv_d(j_lo)
     * so each j_hi is one residue class h0 = 2*brv_k(j_hi)+1 and, within it,
     *     out[j] = sum_i [m_i(zeta^h0) * psi^(h0*i)] * omega^(t*i)
     * with t = brv_d(j_lo). That is a plain cyclic d-point DFT. Inverting just
     * this stage lands directly in the R_k NTT domain: the INTT_k that a full
     * INTT_N would perform is exactly cancelled by the NTT_k that would follow.
     *
     * The residue class occupies d contiguous indices, so loads coalesce, and
     * feeding natural j_lo order into a Cooley-Tukey butterfly consumes the
     * bit-reversal for free.
     */
    __global__ void bm_ntt_to_subring_kernel(Data64* dst,
                                             const Data64* const* src,
                                             const Data64* winv,
                                             const Data64* psi_inv_n,
                                             const Modulus64* modulus, int d,
                                             int cols, int k);

    /** @brief Inverse of bm_ntt_to_subring_kernel. */
    __global__ void bm_subring_to_ntt_kernel(Data64* const* dst,
                                             const Data64* src,
                                             const Data64* wfwd,
                                             const Data64* psi_fwd_n,
                                             const Modulus64* modulus, int d,
                                             int cols, int k);

    /**
     * @brief Modular GEMM over R_{q,k}, pointwise in the length-k NTT index.
     *
     * C[i][j][s] = sum_t A[i][t][s] * B[t][j][s] (mod p), for every limb.
     * @param b_row_stride,b_col_stride Strides used to index B, so the right
     *        operand can be consumed transposed without moving data.
     * @param s_blocks Blocks covering the k coefficients. grid.x must be
     *        s_blocks * d * cols: the output entry and the coefficient share
     *        grid.x, because grid.y would cap d * cols at 65535.
     */
    __global__ void bm_gemm_kernel(Data64* C, const Data64* A, const Data64* B,
                                   const Modulus64* modulus, int d, int inner,
                                   int cols, int k, int b_row_stride,
                                   int b_col_stride, int s_blocks);

    /**
     * @brief Multiply ciphertexts by monomials X^power_j, in place.
     *
     * Step 1 and step 5 of Algorithm 3. In the length-N NTT domain the
     * coefficient at index j is the evaluation at psi^(2*brv_N(j)+1), so
     * multiplying the polynomial by X^power is a pointwise multiply by
     * psi^(power*(2*brv(j)+1)). That keeps the operation exact and avoids any
     * transform round trip.
     *
     * The twiddle comes from @p psi_pow rather than from a per-thread modular
     * exponentiation. Since 2N is a power of two the exponent reduces with a
     * mask, so each element costs one table load and one modular multiply.
     *
     * One launch covers every ciphertext of a step: grid.z selects the
     * ciphertext and both components share the twiddle, so it is loaded once
     * per element rather than once per component.
     *
     * @param data      [count] pointers to ciphertext memory,
     *                  laid out [component][limb][coefficient].
     * @param powers    [count] exponents, already reduced into [0, 2N).
     *                  A zero entry leaves that ciphertext untouched.
     * @param psi_pow   [limbs][2N] powers of the primitive 2N-th root.
     */
    __global__ void bm_mult_monomial_batch_kernel(Data64* const* data,
                                                  const int* powers,
                                                  const Data64* psi_pow,
                                                  const Modulus64* modulus,
                                                  int n_power, int num_limbs);

    /**
     * @brief One TWEAK stage: (e, o) <- (e + X^power * o, e - X^power * o).
     *
     * The monomial twiddle and the butterfly of Algorithm 2 fused into a
     * single pass, for every pair of a stage at once. Fusing matters because
     * both halves are elementwise over the same ciphertexts: done separately
     * the odd operand is read and written twice.
     *
     * @param even,odd [pairs] pointers to the two ciphertexts of each pair.
     * @param powers   [pairs] twiddle exponents in [0, 2N); zero skips the
     *                 multiply. Uniform across a block, so it does not diverge.
     */
    __global__ void bm_tweak_stage_kernel(Data64* const* even,
                                          Data64* const* odd,
                                          const int* powers,
                                          const Data64* psi_pow,
                                          const Modulus64* modulus,
                                          int n_power, int num_limbs);

    /** @brief Scale ciphertexts by a per-limb constant, in place. */
    __global__ void bm_mult_scalar_batch_kernel(Data64* const* data,
                                                const Data64* scalar,
                                                const Modulus64* modulus,
                                                int n_power, int num_limbs);

} // namespace heongpu
#endif // HEONGPU_KERNEL_BATCHMATRIX_H
