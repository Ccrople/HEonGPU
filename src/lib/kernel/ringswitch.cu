// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/kernel/ringswitch.cuh>

namespace heongpu
{
    __global__ void ringswitch_split_kernel(const Data64* __restrict__ big,
                                            Data64* __restrict__ staged,
                                            int n_power_big, int k_power,
                                            int limbs)
    {
        const int i_big = (blockIdx.x << 8) + threadIdx.x;
        const int limb = blockIdx.y;
        const int c = blockIdx.z;

        const int k_mask = (1 << k_power) - 1;
        const int j = i_big & k_mask;
        const int i_small = i_big >> k_power;
        const int n_small = 1 << (n_power_big - k_power);

        const size_t src = (static_cast<size_t>(c) * limbs + limb)
                               * (static_cast<size_t>(1) << n_power_big) +
                           i_big;
        const size_t dst =
            (static_cast<size_t>((j << 1) + c) * limbs + limb) * n_small +
            i_small;

        staged[dst] = big[src];
    }

    __global__ void
    ringswitch_interleave_kernel(const Data64* __restrict__ staged,
                                 Data64* __restrict__ big, int n_power_big,
                                 int k_power, int limbs)
    {
        const int i_big = (blockIdx.x << 8) + threadIdx.x;
        const int limb = blockIdx.y;
        const int c = blockIdx.z;

        const int k_mask = (1 << k_power) - 1;
        const int j = i_big & k_mask;
        const int i_small = i_big >> k_power;
        const int n_small = 1 << (n_power_big - k_power);

        const size_t dst = (static_cast<size_t>(c) * limbs + limb)
                               * (static_cast<size_t>(1) << n_power_big) +
                           i_big;
        const size_t src =
            (static_cast<size_t>((j << 1) + c) * limbs + limb) * n_small +
            i_small;

        big[dst] = staged[src];
    }

    __global__ void
    ringswitch_distribute_kernel(const Data64* __restrict__ staged,
                                 Data64* const* __restrict__ outs,
                                 int n_power_small, int limbs)
    {
        const int i = (blockIdx.x << 8) + threadIdx.x;
        const int limb = blockIdx.y;
        const int j = blockIdx.z >> 1;
        const int c = blockIdx.z & 1;

        const size_t n_small = static_cast<size_t>(1) << n_power_small;
        const size_t slice =
            (static_cast<size_t>((j << 1) + c) * limbs + limb) * n_small + i;
        const size_t inner =
            (static_cast<size_t>(c) * limbs + limb) * n_small + i;

        outs[j][inner] = staged[slice];
    }

    __global__ void
    ringswitch_gather_kernel(const Data64* const* __restrict__ ins,
                             Data64* __restrict__ staged, int n_power_small,
                             int limbs)
    {
        const int i = (blockIdx.x << 8) + threadIdx.x;
        const int limb = blockIdx.y;
        const int j = blockIdx.z >> 1;
        const int c = blockIdx.z & 1;

        const size_t n_small = static_cast<size_t>(1) << n_power_small;
        const size_t slice =
            (static_cast<size_t>((j << 1) + c) * limbs + limb) * n_small + i;
        const size_t inner =
            (static_cast<size_t>(c) * limbs + limb) * n_small + i;

        staged[slice] = ins[j][inner];
    }

} // namespace heongpu
