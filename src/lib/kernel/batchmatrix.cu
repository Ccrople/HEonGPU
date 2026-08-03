// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/kernel/batchmatrix.cuh>

namespace heongpu
{
    __global__ void bm_ntt_k_kernel(Data64* data, const Data64* psi,
                                    const Modulus64* modulus, int k,
                                    int transforms_per_limb)
    {
        extern __shared__ Data64 sh[];

        const int limb = blockIdx.y;
        const Modulus64 p = modulus[limb];

        Data64* g =
            data + (static_cast<size_t>(limb) * transforms_per_limb +
                    blockIdx.x) *
                       k;
        const Data64* tw = psi + static_cast<size_t>(limb) * k;

        for (int i = threadIdx.x; i < k; i += blockDim.x)
            sh[i] = g[i];
        __syncthreads();

        int t = k;
        for (int m = 1; m < k; m <<= 1)
        {
            t >>= 1;
            for (int b = threadIdx.x; b < (k >> 1); b += blockDim.x)
            {
                const int i = b / t;
                const int j = 2 * i * t + (b % t);
                const Data64 U = sh[j];
                const Data64 V =
                    OPERATOR_GPU_64::mult(sh[j + t], tw[m + i], p);
                Data64 x = U + V;
                x = (x >= p.value) ? x - p.value : x;
                Data64 y = U + p.value - V;
                y = (y >= p.value) ? y - p.value : y;
                sh[j] = x;
                sh[j + t] = y;
            }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < k; i += blockDim.x)
            g[i] = sh[i];
    }

    __global__ void bm_intt_k_kernel(Data64* data, const Data64* psi_inv,
                                     const Data64* kinv,
                                     const Modulus64* modulus, int k,
                                     int transforms_per_limb)
    {
        extern __shared__ Data64 sh[];

        const int limb = blockIdx.y;
        const Modulus64 p = modulus[limb];

        Data64* g =
            data + (static_cast<size_t>(limb) * transforms_per_limb +
                    blockIdx.x) *
                       k;
        const Data64* tw = psi_inv + static_cast<size_t>(limb) * k;

        for (int i = threadIdx.x; i < k; i += blockDim.x)
            sh[i] = g[i];
        __syncthreads();

        int t = 1;
        for (int m = k; m > 1; m >>= 1)
        {
            const int h = m >> 1;
            for (int b = threadIdx.x; b < (k >> 1); b += blockDim.x)
            {
                const int i = b / t;
                const int j = 2 * i * t + (b % t);
                const Data64 U = sh[j];
                const Data64 V = sh[j + t];
                Data64 x = U + V;
                x = (x >= p.value) ? x - p.value : x;
                Data64 y = U + p.value - V;
                y = (y >= p.value) ? y - p.value : y;
                sh[j] = x;
                sh[j + t] = OPERATOR_GPU_64::mult(y, tw[h + i], p);
            }
            __syncthreads();
            t <<= 1;
        }

        const Data64 ki = kinv[limb];
        for (int i = threadIdx.x; i < k; i += blockDim.x)
            g[i] = OPERATOR_GPU_64::mult(sh[i], ki, p);
    }

    __global__ void bm_ntt_to_subring_kernel(Data64* dst,
                                             const Data64* const* src,
                                             const Data64* winv,
                                             const Data64* psi_inv_n,
                                             const Modulus64* modulus, int d,
                                             int cols, int k)
    {
        extern __shared__ Data64 sh[];

        const int s = blockIdx.x;
        const int col = blockIdx.y;
        const int limb = blockIdx.z;
        const Modulus64 p = modulus[limb];

        const Data64* in = src[static_cast<size_t>(limb) * cols + col] +
                           static_cast<size_t>(s) * d;
        const Data64* wt = winv + static_cast<size_t>(limb) * d;
        const Data64* pt =
            psi_inv_n + (static_cast<size_t>(limb) * k + s) * d;

        for (int u = threadIdx.x; u < d; u += blockDim.x)
            sh[u] = in[u];
        __syncthreads();

        // Cooley-Tukey: bit-reversed input, natural output.
        for (int len = 2; len <= d; len <<= 1)
        {
            const int half = len >> 1;
            const int step = d / len;
            for (int b = threadIdx.x; b < (d >> 1); b += blockDim.x)
            {
                const int blk = b / half;
                const int j = b % half;
                const int base = blk * len;
                const Data64 u = sh[base + j];
                const Data64 v = OPERATOR_GPU_64::mult(sh[base + j + half],
                                                       wt[step * j], p);
                Data64 x = u + v;
                x = (x >= p.value) ? x - p.value : x;
                Data64 y = u + p.value - v;
                y = (y >= p.value) ? y - p.value : y;
                sh[base + j] = x;
                sh[base + j + half] = y;
            }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < d; i += blockDim.x)
            dst[((static_cast<size_t>(limb) * d + i) * cols + col) * k + s] =
                OPERATOR_GPU_64::mult(sh[i], pt[i], p);
    }

    __global__ void bm_subring_to_ntt_kernel(Data64* const* dst,
                                             const Data64* src,
                                             const Data64* wfwd,
                                             const Data64* psi_fwd_n,
                                             const Modulus64* modulus, int d,
                                             int cols, int k)
    {
        extern __shared__ Data64 sh[];

        const int s = blockIdx.x;
        const int col = blockIdx.y;
        const int limb = blockIdx.z;
        const Modulus64 p = modulus[limb];

        Data64* out = dst[static_cast<size_t>(limb) * cols + col] +
                      static_cast<size_t>(s) * d;
        const Data64* wt = wfwd + static_cast<size_t>(limb) * d;
        const Data64* pt =
            psi_fwd_n + (static_cast<size_t>(limb) * k + s) * d;

        for (int i = threadIdx.x; i < d; i += blockDim.x)
            sh[i] = OPERATOR_GPU_64::mult(
                src[((static_cast<size_t>(limb) * d + i) * cols + col) * k + s],
                pt[i], p);
        __syncthreads();

        // Gentleman-Sande: natural input, bit-reversed output.
        for (int len = d; len >= 2; len >>= 1)
        {
            const int half = len >> 1;
            const int step = d / len;
            for (int b = threadIdx.x; b < (d >> 1); b += blockDim.x)
            {
                const int blk = b / half;
                const int j = b % half;
                const int base = blk * len;
                const Data64 u = sh[base + j];
                const Data64 v = sh[base + j + half];
                Data64 x = u + v;
                x = (x >= p.value) ? x - p.value : x;
                Data64 y = u + p.value - v;
                y = (y >= p.value) ? y - p.value : y;
                sh[base + j] = x;
                sh[base + j + half] =
                    OPERATOR_GPU_64::mult(y, wt[step * j], p);
            }
            __syncthreads();
        }

        for (int u = threadIdx.x; u < d; u += blockDim.x)
            out[u] = sh[u];
    }

    namespace
    {
        __device__ inline uint32_t bm_bitrev(uint32_t v, int bits)
        {
            uint32_t r = 0;
            for (int i = 0; i < bits; ++i)
                r |= ((v >> i) & 1u) << (bits - 1 - i);
            return r;
        }

        __device__ inline Data64 bm_powmod(Data64 b, Data64 e, Data64 m)
        {
            Data64 r = 1;
            b %= m;
            while (e)
            {
                if (e & 1)
                    r = static_cast<__uint128_t>(r) * b % m;
                b = static_cast<__uint128_t>(b) * b % m;
                e >>= 1;
            }
            return r;
        }
    } // namespace

    __global__ void bm_mult_monomial_kernel(Data64* data, const Data64* psi_n,
                                            const Modulus64* modulus,
                                            int n_power, int power,
                                            int num_limbs)
    {
        const int n = 1 << n_power;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;
        const int limb = blockIdx.y;
        const int comp = blockIdx.z;
        const Modulus64 p = modulus[limb];

        const Data64 two_n = 2ull * static_cast<Data64>(n);
        const Data64 h = 2ull * bm_bitrev(static_cast<uint32_t>(idx), n_power) + 1ull;
        const Data64 e = (static_cast<Data64>(power) % two_n) * h % two_n;
        const Data64 w = bm_powmod(psi_n[limb], e, p.value);

        const size_t off =
            (static_cast<size_t>(comp) * num_limbs + limb) * n + idx;
        data[off] = OPERATOR_GPU_64::mult(data[off], w, p);
    }

    __global__ void bm_butterfly_kernel(Data64* e, Data64* o,
                                        const Modulus64* modulus, int n,
                                        int num_limbs)
    {
        const size_t total = static_cast<size_t>(2) * num_limbs * n;
        const size_t idx =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;
        const int limb = static_cast<int>((idx / n) % num_limbs);
        const Data64 p = modulus[limb].value;

        const Data64 x = e[idx];
        const Data64 y = o[idx];
        Data64 s = x + y;
        s = (s >= p) ? s - p : s;
        Data64 t = x + p - y;
        t = (t >= p) ? t - p : t;
        e[idx] = s;
        o[idx] = t;
    }

    __global__ void bm_mult_scalar_kernel(Data64* data, const Data64* scalar,
                                          const Modulus64* modulus, int n,
                                          int num_limbs)
    {
        const size_t total = static_cast<size_t>(2) * num_limbs * n;
        const size_t idx =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;
        const int limb = static_cast<int>((idx / n) % num_limbs);
        data[idx] =
            OPERATOR_GPU_64::mult(data[idx], scalar[limb], modulus[limb]);
    }

    __global__ void bm_gemm_kernel(Data64* C, const Data64* A, const Data64* B,
                                   const Modulus64* modulus, int d, int inner,
                                   int cols, int k, int b_row_stride,
                                   int b_col_stride, int s_blocks)
    {
        // The output entry and the coefficient share grid.x. Carrying the
        // entry in grid.y instead would cap d * cols at 65535, which a 256x256
        // matrix reaches exactly; grid.x is bounded by 2^31 - 1.
        const int sb = blockIdx.x % s_blocks;
        const int ij = blockIdx.x / s_blocks;
        const int s = sb * blockDim.x + threadIdx.x;
        if (s >= k)
            return;

        const int i = ij / cols;
        const int j = ij % cols;
        const int limb = blockIdx.z;
        const Modulus64 p = modulus[limb];

        const Data64* a =
            A + ((static_cast<size_t>(limb) * d + i) * inner) * k + s;
        const Data64* b =
            B + static_cast<size_t>(limb) * inner * cols * k +
            static_cast<size_t>(j) * b_col_stride * k + s;

        Data64 acc = 0;
        for (int t = 0; t < inner; ++t)
        {
            const Data64 av = a[static_cast<size_t>(t) * k];
            const Data64 bv =
                b[static_cast<size_t>(t) * b_row_stride * k];
            acc += OPERATOR_GPU_64::mult(av, bv, p);
            acc = (acc >= p.value) ? acc - p.value : acc;
        }

        C[((static_cast<size_t>(limb) * d + i) * cols + j) * k + s] = acc;
    }

} // namespace heongpu
