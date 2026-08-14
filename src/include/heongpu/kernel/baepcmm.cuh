// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Device kernels for the Bae et al. (CRYPTO 2024, eprint 2024/1284)
// plaintext-ciphertext matrix product.
//
// Every buffer here is a DENSE ROW-MAJOR MATRIX PER RNS LIMB, laid out
// [limb][row][col]. That is deliberately not the [limb][row][col][k] layout
// the Kang batch primitives use: there is no subring transform index in this
// algorithm, because there is no subring transform. The whole product is a
// pair of ordinary dense GEMMs over Z_q on the raw coefficient arrays of the
// ciphertexts.

#ifndef HEONGPU_KERNEL_BAEPCMM_H
#define HEONGPU_KERNEL_BAEPCMM_H

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

namespace heongpu
{
    /**
     * @brief Gather the b-parts of the input ciphertexts into the matrix B.
     *
     * Lemma 4's B: row r of the d2 x cols matrix holds the coefficients of
     * b_r, the b-part of the MLWE ciphertext carrying row r of M.
     *
     * ModDecomp is a pure decimation (App. A, Eq. 14): the extractor
     * e_j(x) = sum_t x_{t*k + j} Y^t collects the coefficients whose index is
     * congruent to j modulo k. So for the global row r = k*i + u,
     *
     *     B[r][t] = beta_i[t*k + u],
     *
     * where beta_i is the coefficient-domain b-part of input ciphertext i.
     * No arithmetic, no keys, no NTT -- this is the whole of ModDecomp on the
     * b side.
     *
     * @param out     [limb][d2][cols] destination.
     * @param in      Per-ciphertext coefficient-domain pointers; in[i] points
     *                at the a-part, so the b-part is one component further on.
     * @param n       Ring degree N.
     * @param cols    d3, the column count; k = n / cols.
     * @param limbs   Active RNS limbs.
     */
    __global__ void bae_gather_b_kernel(Data64* out,
                                        const Data64* const* __restrict__ in,
                                        int n, int cols, int limbs);

    /**
     * @brief Gather the a-parts of the input ciphertexts into the matrix A.
     *
     * Lemma 4's A: row r of the d2 x (k*cols) = d2 x N matrix is the
     * concatenation of the k components of the MLWE a-vector for row r.
     *
     * Writing r = k*i + u, App. A gives that a-vector as the signed cyclic
     * shift by u of the k extracted polynomials,
     *
     *     a_r = ( e_u, e_{u-1}, ..., e_0, Y*e_{k-1}, ..., Y*e_{u+1} ),
     *
     * all taken of alpha_i. The shift is what makes < a_r, sk > reproduce the
     * X^u component of alpha_i * sk: for j <= u the pairing wants e_{u-j}, and
     * for j > u it wants e_{u+k-j} carried across the X^k = Y boundary. The
     * Y factor is a negacyclic shift of the coefficient index, so position 0
     * picks up the MINUS SIGN of Y^cols = -1. Getting that sign wrong is the
     * one silent failure mode in this kernel: the product still type-checks
     * and still decrypts, to the wrong matrix.
     *
     *     j <= u:  A[r][j*cols + t] =  alpha_i[t*k + (u - j)]
     *     j >  u:  A[r][j*cols + 0] = -alpha_i[(cols-1)*k + (u + k - j)]
     *              A[r][j*cols + t] =  alpha_i[(t-1)*k + (u + k - j)],  t >= 1
     *
     * At k = 1 this degenerates to A[r][t] = alpha_r[t], the identity, which
     * is exactly Liu-Zhang / the d = N case of the paper.
     *
     * @param out     [limb][d2][n] destination.
     * @param in      Per-ciphertext coefficient-domain a-part pointers.
     * @param n       Ring degree N.
     * @param cols    d3, the column count; k = n / cols.
     * @param limbs   Active RNS limbs.
     * @param modulus Per-limb modulus, needed for the negation at t = 0.
     */
    __global__ void bae_gather_a_kernel(Data64* out,
                                        const Data64* const* __restrict__ in,
                                        int n, int cols, int limbs,
                                        const Modulus64* modulus);

    /**
     * @brief C = U * X mod p, per RNS limb, tiled.
     *
     * U is the plaintext matrix, already lifted into the RNS base and shared
     * by every limb only in the sense that it was expanded once -- it is
     * passed as [limb][d1][d2] like everything else, because a centred
     * negative entry reduces differently in each limb.
     *
     * X is [limb][d2][width] and C is [limb][d1][width]. The same kernel
     * serves the a-part (width = N) and the b-part (width = cols); they
     * differ only in the width, which is the entire content of Section 4.2's
     * remark that the a-part GEMM is k times the b-part one.
     *
     * Accumulation is reduced every step rather than at the end: a 64-bit
     * accumulator overflows after two products at a 60-bit prime, so there is
     * no deferred-reduction variant of this loop to be had.
     */
    __global__ void bae_gemm_kernel(Data64* C, const Data64* __restrict__ U,
                                    const Data64* __restrict__ X,
                                    const Modulus64* modulus, int d1, int d2,
                                    int width);

    /**
     * @brief Scatter the product's b-part rows back into MLWE b-parts.
     *
     * The exact inverse of bae_gather_b_kernel's index map, writing into
     * per-row degree-cols buffers rather than back into degree-N ciphertexts:
     * after the product the d1 output rows are d1 independent MLWE
     * ciphertexts and there is no degree-N object to write until ModPack has
     * run.
     *
     * @param out   [row][limb][cols] destination, row-major over rows.
     * @param B     [limb][d1][cols] product.
     */
    __global__ void bae_scatter_b_kernel(Data64* out,
                                         const Data64* __restrict__ B, int d1,
                                         int cols, int limbs);

    /**
     * @brief Scatter the product's a-part rows into MLWE a-vectors.
     *
     * Row r of A' is already the concatenation of the k components of the
     * output MLWE a-vector, so this is a straight restride from
     * [limb][d1][k*cols] to [row][component][limb][cols]. No sign, no shift:
     * the shift of bae_gather_a_kernel belongs to the INPUT's decomposition
     * and must not be undone here. Undoing it is the second silent failure
     * mode, and it is silent for the same reason as the first.
     */
    __global__ void bae_scatter_a_kernel(Data64* out,
                                         const Data64* __restrict__ A, int d1,
                                         int cols, int k, int limbs);

    /**
     * @brief The k = 1 fast path: write A' and B' straight back as degree-N
     *        ciphertexts.
     *
     * When cols = N there is no decomposition and no packing, so an output
     * row IS an ordinary RLWE ciphertext and the whole algorithm is two
     * GEMMs. This kernel exists so that path never allocates the MLWE
     * staging buffers at all.
     *
     * @param outs  Per-ciphertext destinations, a-part first.
     */
    __global__ void bae_emit_rlwe_kernel(Data64* const* __restrict__ outs,
                                         const Data64* __restrict__ A,
                                         const Data64* __restrict__ B, int d1,
                                         int n, int limbs);


    /**
     * @brief ModPack step 1: interleave k MLWE rows back up to degree N.
     *
     * Algorithm 2 step 5. Output ciphertext i owns MLWE rows k*i .. k*i+k-1,
     * and the degree-N polynomials it needs are
     *
     *     Atilde_j[t*k + u] = A'[k*i + u][j*cols + t]      (one per component j)
     *     Btilde  [t*k + u] = B'[k*i + u][t]
     *
     * which is the exact inverse of the decimation ModDecomp performed --
     * this is an interleave, and the SHIFT that bae_gather_a_kernel applied
     * to the input is deliberately NOT undone, because it belonged to the
     * input's own decomposition.
     *
     * What this kernel cannot do is finish the job: the k a-vectors of the
     * output rows no longer share the "shifted decimation of one alpha"
     * structure, so Atilde_j is not an a-part. It is a ring element that must
     * be multiplied by the sub-secret s_j, and that is a key switch.
     *
     * @param a_out [j][limb][n] the k interleaved component polynomials.
     * @param b_out [limb][n]    the interleaved b polynomial.
     * @param A     [limb][d1][n] product a-part.
     * @param B     [limb][d1][cols] product b-part.
     * @param row0  k*i, the first MLWE row of this output ciphertext.
     */
    __global__ void bae_modpack_assemble_kernel(
        Data64* __restrict__ a_out, Data64* __restrict__ b_out,
        const Data64* __restrict__ A, const Data64* __restrict__ B, int d1,
        int cols, int k, int limbs, int row0);

} // namespace heongpu

#endif // HEONGPU_KERNEL_BAEPCMM_H
