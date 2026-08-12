// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/kernel/batchmatrix.cuh>

namespace heongpu
{
    __global__ void bm_crt_expand_kernel(Data64* out, const int64_t* coeffs,
                                         const Modulus64* modulus,
                                         size_t per_limb, int num_limbs)
    {
        const size_t i =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= per_limb)
            return;

        // One load of the coefficient serves every limb: the array is shared
        // across the RNS base, which is exactly what made the host expansion
        // write num_limbs copies of it. Walking the limbs inside the thread
        // rather than across a grid dimension keeps that single load, and
        // keeps this loop on the path every shape takes -- spreading limbs
        // over blockIdx.y would leave it dead below 33 limbs, which is every
        // shape the tests cover and none of the ones the Llama chain runs.
        //
        // coeffs is allowed to alias the first limb of out, which is how the
        // caller avoids a second buffer. That is safe in exactly one way and
        // it is load-before-store: this thread reads its own element once,
        // here, before the loop writes anything, and no other thread touches
        // index i in any limb. Neither pointer is __restrict__, and int64_t
        // and Data64 are the corresponding signed/unsigned types, so the
        // compiler has to keep this load ahead of the stores.
        const int64_t v = coeffs[i];
        for (int limb = 0; limb < num_limbs; ++limb)
        {
            const Data64 p = modulus[limb].value;
            Data64 r;
            if (v >= 0)
            {
                r = static_cast<Data64>(v) % p;
            }
            else
            {
                const Data64 m = static_cast<Data64>(-v) % p;
                r = (m == 0) ? 0 : p - m;
            }
            out[static_cast<size_t>(limb) * per_limb + i] = r;
        }
    }

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
        /**
         * @brief psi_N^(power * (2*brv(idx)+1)) for the length-N NTT domain.
         *
         * 2N is a power of two, so reducing the exponent is a mask. The
         * product needs 64 bits: both factors reach 2N = 2^17 here.
         */
        __device__ inline Data64 bm_monomial_twiddle(const Data64* psi_pow,
                                                     int limb, int n_power,
                                                     int power, int idx)
        {
            const unsigned mask = (1u << (n_power + 1)) - 1u;
            const unsigned h =
                2u * (__brev(static_cast<unsigned>(idx)) >> (32 - n_power)) +
                1u;
            const unsigned e = static_cast<unsigned>(
                (static_cast<unsigned long long>(static_cast<unsigned>(power)) *
                 h) &
                mask);
            return psi_pow[(static_cast<size_t>(limb) << (n_power + 1)) + e];
        }
    } // namespace

    __global__ void bm_mult_monomial_batch_kernel(Data64* const* data,
                                                  const int* powers,
                                                  const Data64* psi_pow,
                                                  const Modulus64* modulus,
                                                  int n_power, int num_limbs)
    {
        const int n = 1 << n_power;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const int power = powers[blockIdx.z];
        if (power == 0)
            return;

        const int limb = blockIdx.y;
        const Modulus64 p = modulus[limb];
        const Data64 w =
            bm_monomial_twiddle(psi_pow, limb, n_power, power, idx);

        Data64* g = data[blockIdx.z] + static_cast<size_t>(limb) * n + idx;
        const size_t comp = static_cast<size_t>(num_limbs) * n;
        g[0] = OPERATOR_GPU_64::mult(g[0], w, p);
        g[comp] = OPERATOR_GPU_64::mult(g[comp], w, p);
    }

    __global__ void bm_tweak_stage_kernel(Data64* const* even,
                                          Data64* const* odd,
                                          const int* powers,
                                          const Data64* psi_pow,
                                          const Modulus64* modulus,
                                          int n_power, int num_limbs)
    {
        const int n = 1 << n_power;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const int limb = blockIdx.y;
        const Modulus64 mp = modulus[limb];
        const Data64 p = mp.value;

        // Uniform across the block: the pair is carried by grid.z.
        const int power = powers[blockIdx.z];
        const Data64 w =
            (power == 0)
                ? 0
                : bm_monomial_twiddle(psi_pow, limb, n_power, power, idx);

        const size_t off = static_cast<size_t>(limb) * n + idx;
        const size_t comp = static_cast<size_t>(num_limbs) * n;
        Data64* ep = even[blockIdx.z] + off;
        Data64* op = odd[blockIdx.z] + off;

#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            const size_t o = static_cast<size_t>(c) * comp;
            const Data64 u = ep[o];
            const Data64 v =
                (power == 0) ? op[o] : OPERATOR_GPU_64::mult(op[o], w, mp);
            Data64 s = u + v;
            s = (s >= p) ? s - p : s;
            Data64 t = u + p - v;
            t = (t >= p) ? t - p : t;
            ep[o] = s;
            op[o] = t;
        }
    }

    __global__ void bm_mult_scalar_batch_kernel(Data64* const* data,
                                                const Data64* scalar,
                                                const Modulus64* modulus,
                                                int n_power, int num_limbs)
    {
        const int n = 1 << n_power;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const int limb = blockIdx.y;
        const Modulus64 p = modulus[limb];
        const Data64 s = scalar[limb];

        Data64* g = data[blockIdx.z] + static_cast<size_t>(limb) * n + idx;
        const size_t comp = static_cast<size_t>(num_limbs) * n;
        g[0] = OPERATOR_GPU_64::mult(g[0], s, p);
        g[comp] = OPERATOR_GPU_64::mult(g[comp], s, p);
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
