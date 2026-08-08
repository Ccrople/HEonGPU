// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Device kernels for CKKS ring switching (Sylph §3.3).
//
// Every kernel here is a pure data movement in the COEFFICIENT (non-NTT)
// domain. The ring decomposition R_N = ⊕_j X^j · R_{N'}[X^k] is exact on
// coefficients:
//
//     a(X) = Σ_j a_j(X^k) · X^j   ⟺   a_j[i] = a[i·k + j],
//
// so switching a ciphertext between the rings needs no twiddle factors, no
// modular arithmetic and no rounding — only the key switch that brackets it
// touches the algebra, and that key switch is the existing switchkey pipeline.
//
// The staging buffer used between the split/interleave and the (single,
// batched) small-ring NTT is laid out [j][component][limb][coefficient] with
// the coefficient fastest, so that
//   * each small ciphertext j occupies one contiguous [c][limb][i] slice, and
//   * the flat polynomial order (j·2 + c)·limbs + limb has the limb index
//     varying fastest, which is exactly the (poly % mod_count) convention the
//     gpuntt batch launchers use to pick a modulus per polynomial.

#ifndef HEONGPU_KERNEL_RINGSWITCH_H
#define HEONGPU_KERNEL_RINGSWITCH_H

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

namespace heongpu
{
    /**
     * @brief Coefficient split, big ring -> k-slice staging buffer.
     *
     * Input is one big-ring ciphertext in the coefficient domain, laid out
     * [c][limb][i_big] with component stride limbs·n_big (the CURRENT level's
     * stride, not the top-of-chain one). Output is the staging buffer
     * described above: staged[j][c][limb][i_small] = big[c][limb][i_small·k+j].
     *
     * Grid (n_big >> 8, limbs, 2), 256 threads. Reads are coalesced; the
     * scattered writes touch k distinct slices and are the cheap side of a
     * transform whose cost is dominated by the surrounding key switch.
     */
    __global__ void ringswitch_split_kernel(const Data64* __restrict__ big,
                                            Data64* __restrict__ staged,
                                            int n_power_big, int k_power,
                                            int limbs);

    /**
     * @brief Coefficient interleave, k-slice staging buffer -> big ring.
     *
     * Exact inverse of ringswitch_split_kernel:
     * big[c][limb][i_small·k+j] = staged[j][c][limb][i_small].
     * Same grid; writes are coalesced, reads scattered.
     */
    __global__ void
    ringswitch_interleave_kernel(const Data64* __restrict__ staged,
                                 Data64* __restrict__ big, int n_power_big,
                                 int k_power, int limbs);

    /**
     * @brief Copy each staging slice into its own ciphertext buffer.
     *
     * outs is a device array of k pointers, outs[j] the data pointer of small
     * ciphertext j (layout [c][limb][i_small], component stride limbs·n_small).
     * Grid (n_small >> 8, limbs, 2·k) with blockIdx.z = j·2 + c.
     */
    __global__ void
    ringswitch_distribute_kernel(const Data64* __restrict__ staged,
                                 Data64* const* __restrict__ outs,
                                 int n_power_small, int limbs);

    /**
     * @brief Inverse of ringswitch_distribute_kernel: gather k ciphertext
     * buffers into the contiguous staging layout for one batched INTT.
     */
    __global__ void
    ringswitch_gather_kernel(const Data64* const* __restrict__ ins,
                             Data64* __restrict__ staged, int n_power_small,
                             int limbs);

} // namespace heongpu

#endif // HEONGPU_KERNEL_RINGSWITCH_H
