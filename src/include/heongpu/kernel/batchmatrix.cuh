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
     * @brief Multiply a ciphertext by the monomial X^power, in place.
     *
     * Step 1 and step 5 of Algorithm 3, and the odd-branch twiddle of the
     * Algorithm 2 butterfly. In the length-N NTT domain the coefficient at
     * index j is the evaluation at psi^(2*brv_N(j)+1), so multiplying the
     * polynomial by X^power is a pointwise multiply by psi^(power*(2*brv(j)+1)).
     * That keeps the operation exact and avoids any transform round trip.
     *
     * @param data      Ciphertext memory, [component][limb][coefficient].
     * @param psi_n     [limbs] primitive 2N-th root per limb.
     * @param power     Exponent, taken modulo 2N.
     */
    __global__ void bm_mult_monomial_kernel(Data64* data, const Data64* psi_n,
                                            const Modulus64* modulus,
                                            int n_power, int power,
                                            int num_limbs);

    /**
     * @brief In-place butterfly: (e, o) <- (e + o, e - o) over every limb.
     *
     * The combine step of the Algorithm 2 TWEAK recursion. Operating directly
     * on both ciphertext components at once avoids the copy-then-accumulate
     * that a three-operand add would cost.
     */
    __global__ void bm_butterfly_kernel(Data64* e, Data64* o,
                                        const Modulus64* modulus, int n,
                                        int num_limbs);

    /** @brief Scale a ciphertext by a per-limb constant, in place. */
    __global__ void bm_mult_scalar_kernel(Data64* data, const Data64* scalar,
                                          const Modulus64* modulus, int n,
                                          int num_limbs);

} // namespace heongpu
#endif // HEONGPU_KERNEL_BATCHMATRIX_H
