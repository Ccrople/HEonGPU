// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/kernel/baepcmm.cuh>

namespace heongpu
{
    __global__ void bae_gather_b_kernel(Data64* out,
                                        const Data64* const* __restrict__ in,
                                        int n, int cols, int limbs)
    {
        const int t = (blockIdx.x << 8) + threadIdx.x;
        if (t >= cols)
            return;
        const int r = blockIdx.y;
        const int limb = blockIdx.z;

        const int k = n / cols;
        const int i = r / k;
        const int u = r - i * k;

        // COMPONENT ORDER. sk_multiplication_ckks computes
        // plaintext = ct_0 + ct_1 * sk (src/lib/kernel/decryption.cu:359-364),
        // so component 0 is the b-part and component 1 is the a-part. That is
        // the reverse of the paper's (a, b) naming, and it is invisible at
        // k = 1 -- U hits both components alike, so a consistent swap cancels
        // -- but it is fatal in ModPack, which multiplies exactly one of them
        // by the secret.
        const Data64* beta = in[i] + static_cast<size_t>(limb) * n;

        out[(static_cast<size_t>(limb) * gridDim.y + r) * cols + t] =
            beta[static_cast<size_t>(t) * k + u];
    }

    __global__ void bae_gather_a_kernel(Data64* out,
                                        const Data64* const* __restrict__ in,
                                        int n, int cols, int limbs,
                                        const Modulus64* modulus)
    {
        const int idx = (blockIdx.x << 8) + threadIdx.x;
        if (idx >= n)
            return;
        const int r = blockIdx.y;
        const int limb = blockIdx.z;

        const int k = n / cols;
        const int i = r / k;
        const int u = r - i * k;

        const int j = idx / cols; // which MLWE a-component
        const int t = idx - j * cols; // coefficient inside it

        // Component 1 is the a-part; see the note in bae_gather_b_kernel.
        const Data64* alpha = in[i] + static_cast<size_t>(limbs) * n +
                              static_cast<size_t>(limb) * n;

        Data64 v;
        if (j <= u)
        {
            v = alpha[static_cast<size_t>(t) * k + (u - j)];
        }
        else
        {
            const int phase = u + k - j;
            if (t == 0)
            {
                const Data64 w =
                    alpha[static_cast<size_t>(cols - 1) * k + phase];
                // Y^cols = -1, so the coefficient carried across the wrap
                // changes sign. A zero must stay zero rather than become p.
                v = (w == 0) ? 0 : (modulus[limb].value - w);
            }
            else
            {
                v = alpha[static_cast<size_t>(t - 1) * k + phase];
            }
        }

        out[(static_cast<size_t>(limb) * gridDim.y + r) * n + idx] = v;
    }

#define BAE_TILE 16

    __global__ void bae_gemm_kernel(Data64* C, const Data64* __restrict__ U,
                                    const Data64* __restrict__ X,
                                    const Modulus64* modulus, int d1, int d2,
                                    int width)
    {
        __shared__ Data64 us[BAE_TILE][BAE_TILE];
        __shared__ Data64 xs[BAE_TILE][BAE_TILE];

        const int limb = blockIdx.z;
        const Modulus64 p = modulus[limb];

        const int row = blockIdx.y * BAE_TILE + threadIdx.y;
        const int col = blockIdx.x * BAE_TILE + threadIdx.x;

        const Data64* Ul = U + static_cast<size_t>(limb) * d1 * d2;
        const Data64* Xl = X + static_cast<size_t>(limb) * d2 * width;

        Data64 acc = 0;
        for (int t0 = 0; t0 < d2; t0 += BAE_TILE)
        {
            const int ut = t0 + threadIdx.x;
            us[threadIdx.y][threadIdx.x] =
                (row < d1 && ut < d2)
                    ? Ul[static_cast<size_t>(row) * d2 + ut]
                    : 0;

            const int xt = t0 + threadIdx.y;
            xs[threadIdx.y][threadIdx.x] =
                (xt < d2 && col < width)
                    ? Xl[static_cast<size_t>(xt) * width + col]
                    : 0;

            __syncthreads();

#pragma unroll
            for (int t = 0; t < BAE_TILE; ++t)
            {
                // Reduce every step: two 60-bit products overflow a 64-bit
                // accumulator, so there is no deferred reduction available.
                acc += OPERATOR_GPU_64::mult(us[threadIdx.y][t],
                                             xs[t][threadIdx.x], p);
                acc = (acc >= p.value) ? acc - p.value : acc;
            }

            __syncthreads();
        }

        if (row < d1 && col < width)
        {
            C[(static_cast<size_t>(limb) * d1 + row) * width + col] = acc;
        }
    }

#undef BAE_TILE

    __global__ void bae_scatter_b_kernel(Data64* out,
                                         const Data64* __restrict__ B, int d1,
                                         int cols, int limbs)
    {
        const int t = (blockIdx.x << 8) + threadIdx.x;
        if (t >= cols)
            return;
        const int r = blockIdx.y;
        const int limb = blockIdx.z;

        out[(static_cast<size_t>(r) * limbs + limb) * cols + t] =
            B[(static_cast<size_t>(limb) * d1 + r) * cols + t];
    }

    __global__ void bae_scatter_a_kernel(Data64* out,
                                         const Data64* __restrict__ A, int d1,
                                         int cols, int k, int limbs)
    {
        const int t = (blockIdx.x << 8) + threadIdx.x;
        if (t >= cols)
            return;
        const int rj = blockIdx.y; // r * k + j
        const int limb = blockIdx.z;

        const int r = rj / k;
        const int j = rj - r * k;
        const int n = cols * k;

        out[((static_cast<size_t>(r) * k + j) * limbs + limb) * cols + t] =
            A[(static_cast<size_t>(limb) * d1 + r) * n + j * cols + t];
    }

    __global__ void bae_emit_rlwe_kernel(Data64* const* __restrict__ outs,
                                         const Data64* __restrict__ A,
                                         const Data64* __restrict__ B, int d1,
                                         int n, int limbs)
    {
        const int t = (blockIdx.x << 8) + threadIdx.x;
        if (t >= n)
            return;
        const int r = blockIdx.y;
        const int limb = blockIdx.z;

        // b-part into component 0, a-part into component 1.
        Data64* dst = outs[r] + static_cast<size_t>(limb) * n;
        dst[t] = B[(static_cast<size_t>(limb) * d1 + r) * n + t];
        dst[static_cast<size_t>(limbs) * n + t] =
            A[(static_cast<size_t>(limb) * d1 + r) * n + t];
    }


    __global__ void bae_modpack_assemble_kernel(
        Data64* __restrict__ a_out, Data64* __restrict__ b_out,
        const Data64* __restrict__ A, const Data64* __restrict__ B, int d1,
        int cols, int k, int limbs, int row0)
    {
        const int idx = (blockIdx.x << 8) + threadIdx.x;
        const int n = cols * k;
        if (idx >= n)
            return;
        const int limb = blockIdx.y;
        const int j = blockIdx.z; // component, or k for the b polynomial

        const int u = idx % k; // which MLWE row inside the group
        const int t = idx / k; // which column of that row
        const int row = row0 + u;

        if (j == k)
        {
            b_out[static_cast<size_t>(limb) * n + idx] =
                B[(static_cast<size_t>(limb) * d1 + row) * cols + t];
        }
        else
        {
            a_out[(static_cast<size_t>(j) * limbs + limb) * n + idx] =
                A[(static_cast<size_t>(limb) * d1 + row) * n + j * cols + t];
        }
    }

} // namespace heongpu
