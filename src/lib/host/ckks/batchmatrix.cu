// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/batchmatrix.cuh>
#include <heongpu/kernel/batchmatrix.cuh>
#include <heongpu/kernel/addition.cuh>
#include <heongpu/host/ckks/plaintext.cuh>
#include <heongpu/util/util.cuh>

#include <gpuntt/ntt_merge/ntt.cuh>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace heongpu
{
    namespace
    {
        /// Scoped NVTX range, so a capture can attribute GPU time to the step
        /// of Algorithm 3/4 that issued it. Every step boundary below already
        /// ends in cudaDeviceSynchronize, so these host-side ranges bracket the
        /// device work tightly enough to bucket kernels by timestamp.
        struct BmRange
        {
            explicit BmRange(const char* name) { nvtxRangePushA(name); }
            ~BmRange() { nvtxRangePop(); }
            BmRange(const BmRange&) = delete;
            BmRange& operator=(const BmRange&) = delete;
        };

        inline bool is_power_of_two(int v) noexcept
        {
            return v > 0 && (v & (v - 1)) == 0;
        }

        inline int log2i(int v) noexcept
        {
            int r = 0;
            while ((1 << r) < v)
                ++r;
            return r;
        }

        inline uint32_t bitrev(uint32_t v, int bits) noexcept
        {
            uint32_t r = 0;
            for (int i = 0; i < bits; ++i)
                r |= ((v >> i) & 1u) << (bits - 1 - i);
            return r;
        }

        Data64 powmod(Data64 b, Data64 e, Data64 m)
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

        /// p is prime here, so Fermat gives the inverse.
        inline Data64 invmod(Data64 a, Data64 p) { return powmod(a, p - 2, p); }

        /// Inverse modulo a non-prime modulus, via the extended Euclidean
        /// algorithm; 2N is not prime so Fermat does not apply here.
        uint64_t invmod_generic(uint64_t a, uint64_t m)
        {
            int64_t t = 0, newt = 1;
            int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a);
            while (newr != 0)
            {
                const int64_t q = r / newr;
                int64_t tmp = t - q * newt;
                t = newt;
                newt = tmp;
                tmp = r - q * newr;
                r = newr;
                newr = tmp;
            }
            if (r > 1)
                throw std::runtime_error("element is not invertible mod 2N");
            if (t < 0)
                t += static_cast<int64_t>(m);
            return static_cast<uint64_t>(t);
        }

        inline Data64 centered_to_modular(int64_t v, Data64 p)
        {
            if (v >= 0)
                return static_cast<Data64>(v) % p;
            const Data64 m = static_cast<Data64>(-v) % p;
            return m == 0 ? 0 : p - m;
        }

        inline Data64 mulmod(Data64 a, Data64 b, Data64 p)
        {
            return static_cast<Data64>(static_cast<__uint128_t>(a) * b % p);
        }

        // Minimal little-endian fixed-width unsigned integer, carrying just the
        // four operations CRT reconstruction needs. A product of RNS primes
        // outgrows __int128 as soon as there are three limbs, so the mixed
        // radix accumulation cannot be done in any builtin type.
        using BigUInt = std::vector<Data64>;

        /// acc += v * w
        void big_add_scaled(BigUInt& acc, const BigUInt& v, Data64 w)
        {
            __uint128_t carry = 0;
            for (size_t i = 0; i < acc.size(); ++i)
            {
                const __uint128_t cur =
                    static_cast<__uint128_t>(v[i]) * w + acc[i] + carry;
                acc[i] = static_cast<Data64>(cur);
                carry = cur >> 64;
            }
        }

        /// v *= w
        void big_mul_word(BigUInt& v, Data64 w)
        {
            __uint128_t carry = 0;
            for (size_t i = 0; i < v.size(); ++i)
            {
                const __uint128_t cur =
                    static_cast<__uint128_t>(v[i]) * w + carry;
                v[i] = static_cast<Data64>(cur);
                carry = cur >> 64;
            }
        }

        /// a -= b, wrapping; a negative result is left in two's complement.
        void big_sub(BigUInt& a, const BigUInt& b)
        {
            __uint128_t borrow = 0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                const __uint128_t cur =
                    static_cast<__uint128_t>(a[i]) - b[i] - borrow;
                a[i] = static_cast<Data64>(cur);
                borrow = (cur >> 64) ? 1 : 0;
            }
        }

        int big_cmp(const BigUInt& a, const BigUInt& b)
        {
            for (size_t i = a.size(); i-- > 0;)
                if (a[i] != b[i])
                    return a[i] < b[i] ? -1 : 1;
            return 0;
        }

        void big_shr1(BigUInt& v)
        {
            for (size_t i = 0; i + 1 < v.size(); ++i)
                v[i] = (v[i] >> 1) | (v[i + 1] << 63);
            v.back() >>= 1;
        }
    } // namespace

    // -----------------------------------------------------------------------
    // Layout
    // -----------------------------------------------------------------------

    BatchMatrixLayout::BatchMatrixLayout(int N_, int d_) : N(N_), d(d_)
    {
        if (!is_power_of_two(N))
            throw std::invalid_argument(
                "ring degree must be a positive power of two");
        if (!is_power_of_two(d) || d > N)
            throw std::invalid_argument("matrix row count must be a power of "
                                        "two dividing the ring degree");
        k = N / d;
        if (k < 2)
            throw std::invalid_argument("subring degree must be at least 2; "
                                        "reduce the matrix row count");
        batch = k / 2;
    }

    // -----------------------------------------------------------------------
    // Batch matrix encoding (Definition 1)
    // -----------------------------------------------------------------------

    BatchMatrixEncoder::BatchMatrixEncoder(int k) : k_(k)
    {
        if (k < 2 || !is_power_of_two(k))
            throw std::invalid_argument(
                "subring degree must be a power of two of at least 2");

        pow_.resize(static_cast<size_t>(k_ / 2) * k_);
        const double pi = std::acos(-1.0);
        const uint64_t mod = 2ull * static_cast<uint64_t>(k_);
        uint64_t g = 1;
        for (int j = 0; j < k_ / 2; ++j)
        {
            for (int t = 0; t < k_; ++t)
            {
                const uint64_t e = (g * static_cast<uint64_t>(t)) % mod;
                pow_[static_cast<size_t>(j) * k_ + t] =
                    std::polar(1.0, pi * static_cast<double>(e) / k_);
            }
            g = (g * 5) % mod;
        }
    }

    void BatchMatrixEncoder::encode(
        const std::vector<std::vector<std::complex<double>>>& batch, int rows,
        int cols, double scale, std::vector<int64_t>& out) const
    {
        const int nslots = k_ / 2;
        const size_t entries = static_cast<size_t>(rows) * cols;
        if (static_cast<int>(batch.size()) != nslots)
            throw std::invalid_argument("batch must hold exactly k/2 matrices");
        for (const auto& m : batch)
            if (m.size() != entries)
                throw std::invalid_argument(
                    "every matrix in the batch must have rows*cols entries");

        out.assign(entries * k_, 0);
        const double norm = 2.0 / k_;

        for (size_t e = 0; e < entries; ++e)
        {
            int64_t* dst = out.data() + e * k_;
            // Entries that are zero across the whole batch encode to zero;
            // skipping them keeps sparse operands cheap, since this transform
            // is O(k^2) per entry.
            bool nonzero = false;
            for (int j = 0; j < nslots && !nonzero; ++j)
                nonzero = batch[j][e] != std::complex<double>(0.0, 0.0);
            if (!nonzero)
                continue;

            for (int t = 0; t < k_; ++t)
            {
                double acc = 0.0;
                for (int j = 0; j < nslots; ++j)
                {
                    const std::complex<double>& w =
                        pow_[static_cast<size_t>(j) * k_ + t];
                    const std::complex<double>& z = batch[j][e];
                    acc += z.real() * w.real() + z.imag() * w.imag();
                }
                dst[t] = std::llround(scale * norm * acc);
            }
        }
    }

    void BatchMatrixEncoder::decode(
        const std::vector<int64_t>& in, int rows, int cols, double scale,
        std::vector<std::vector<std::complex<double>>>& batch) const
    {
        const int nslots = k_ / 2;
        const size_t entries = static_cast<size_t>(rows) * cols;
        if (in.size() != entries * static_cast<size_t>(k_))
            throw std::invalid_argument("encoded matrix has the wrong length");

        batch.assign(nslots, std::vector<std::complex<double>>(
                                 entries, std::complex<double>(0.0, 0.0)));
        for (size_t e = 0; e < entries; ++e)
        {
            const int64_t* src = in.data() + e * k_;
            for (int j = 0; j < nslots; ++j)
            {
                std::complex<double> acc(0.0, 0.0);
                for (int t = 0; t < k_; ++t)
                    acc += static_cast<double>(src[t]) *
                           pow_[static_cast<size_t>(j) * k_ + t];
                batch[j][e] = acc / scale;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Matrix encryption layout (Definition 2)
    // -----------------------------------------------------------------------

    void build_matrix_encryption_coefficients(
        const std::vector<int64_t>& coeffs, const BatchMatrixLayout& layout,
        int rows, int cols, std::vector<std::vector<int64_t>>& out)
    {
        if (rows != layout.d)
            throw std::invalid_argument("matrix encryption requires the row "
                                        "count to equal the module rank d");
        if (coeffs.size() !=
            static_cast<size_t>(rows) * cols * layout.k)
            throw std::invalid_argument("encoded matrix has the wrong length");

        out.assign(cols, std::vector<int64_t>(layout.N, 0));
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
            {
                const int64_t* e =
                    coeffs.data() +
                    (static_cast<size_t>(i) * cols + j) * layout.k;
                for (int t = 0; t < layout.k; ++t)
                    out[j][i + static_cast<size_t>(layout.d) * t] = e[t];
            }
    }

    void split_matrix_encryption_coefficients(
        const std::vector<std::vector<int64_t>>& in,
        const BatchMatrixLayout& layout, int rows, int cols,
        std::vector<int64_t>& coeffs)
    {
        if (rows != layout.d)
            throw std::invalid_argument("matrix encryption requires the row "
                                        "count to equal the module rank d");
        if (static_cast<int>(in.size()) != cols)
            throw std::invalid_argument(
                "expected one coefficient vector per column");

        coeffs.assign(static_cast<size_t>(rows) * cols * layout.k, 0);
        for (int j = 0; j < cols; ++j)
        {
            if (in[j].size() != static_cast<size_t>(layout.N))
                throw std::invalid_argument(
                    "coefficient vectors must have length N");
            for (int i = 0; i < rows; ++i)
            {
                int64_t* e = coeffs.data() +
                             (static_cast<size_t>(i) * cols + j) * layout.k;
                for (int t = 0; t < layout.k; ++t)
                    e[t] = in[j][i + static_cast<size_t>(layout.d) * t];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rotation keys required by CMT (Algorithm 3)
    // -----------------------------------------------------------------------

    std::vector<int>
    get_batch_cmt_rotation_indices(const BatchMatrixLayout& layout)
    {
        const int N = layout.N;
        const int k = layout.k;
        const int d = layout.d;
        const uint64_t mod = 2ull * static_cast<uint64_t>(N);
        const int half = N / 2;

        // Rotation index r corresponds to the Galois element 5^r mod 2N.
        std::map<uint64_t, int> galois_to_index;
        uint64_t g = 1;
        for (int r = 0; r < half; ++r)
        {
            galois_to_index.emplace(g, r);
            g = (g * 5) % mod;
        }

        std::vector<int> res;
        for (int t = 0; t < d; ++t)
        {
            const uint64_t h = (2ull * k * t + 1) % mod;
            auto it = galois_to_index.find(h);
            if (it == galois_to_index.end())
                throw std::runtime_error("automorphism X -> X^(2kt+1) is not a "
                                         "slot rotation for this layout");
            if (it->second != 0)
                res.push_back(it->second);
        }
        return res;
    }

    std::vector<int>
    get_rectangular_rotation_indices(const BatchMatrixLayout& layout)
    {
        std::vector<int> res = get_batch_cmt_rotation_indices(layout);
        const BatchMatrixLayout half(layout.N, layout.N / 2);
        const std::vector<int> other = get_batch_cmt_rotation_indices(half);
        res.insert(res.end(), other.begin(), other.end());
        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());
        return res;
    }

    // -----------------------------------------------------------------------
    // Batch matrix operator
    // -----------------------------------------------------------------------

    HEBatchMatrixOperator<Scheme::CKKS>::~HEBatchMatrixOperator() = default;

    HEBatchMatrixOperator<Scheme::CKKS>::HEBatchMatrixOperator(
        HEContext<Scheme::CKKS>& context, const BatchMatrixLayout& layout)
        : context_(context), layout_(layout)
    {
        n_ = context_->get_poly_modulus_degree();
        q_size_ = context_->get_ciphertext_modulus_count();
        if (n_ != layout_.N)
            throw std::invalid_argument(
                "layout ring degree does not match the context");
    }

    const BatchSubringTables&
    HEBatchMatrixOperator<Scheme::CKKS>::tables_for(int depth)
    {
        auto it = table_cache_.find(depth);
        if (it != table_cache_.end())
            return it->second;

        const int num_limbs = q_size_ - depth;
        if (num_limbs <= 0)
            throw std::invalid_argument("ciphertext has no remaining limbs");

        const int k = layout_.k;
        const int d = layout_.d;
        const int logk = log2i(k);

        // The active RNS base is the first num_limbs primes of Q.
        std::vector<Modulus64> all = context_->get_key_modulus();
        std::vector<Modulus64> primes(all.begin(), all.begin() + num_limbs);

        // psi_N per limb, exactly the root the context's own NTT uses.
        std::vector<Data64> psi_n =
            generate_primitive_root_of_unity(n_, primes);

        std::vector<Data64> psi(static_cast<size_t>(num_limbs) * k);
        std::vector<Data64> psi_i(static_cast<size_t>(num_limbs) * k);
        std::vector<Data64> kinv(num_limbs);
        std::vector<Data64> wf(static_cast<size_t>(num_limbs) * d);
        std::vector<Data64> wi(static_cast<size_t>(num_limbs) * d);
        std::vector<Data64> pf(static_cast<size_t>(num_limbs) * k * d);
        std::vector<Data64> pin(static_cast<size_t>(num_limbs) * k * d);

        for (int l = 0; l < num_limbs; ++l)
        {
            const Data64 p = primes[l].value;

            // The subring root must be the one the length-N transform induces:
            // zeta = psi_N^d. An unrelated 2k-th root would be a valid
            // transform on its own but would not line up with the residue
            // classes of the length-N NTT.
            const Data64 root = powmod(psi_n[l], static_cast<Data64>(d), p);
            if (powmod(root, k, p) != p - 1)
                throw std::runtime_error(
                    "psi_N^d is not a primitive 2k-th root of unity");
            const Data64 iroot = invmod(root, p);

            for (int i = 0; i < k; ++i)
            {
                const uint32_t e = bitrev(static_cast<uint32_t>(i), logk);
                psi[static_cast<size_t>(l) * k + i] = powmod(root, e, p);
                psi_i[static_cast<size_t>(l) * k + i] = powmod(iroot, e, p);
            }
            kinv[l] = invmod(static_cast<Data64>(k) % p, p);

            // omega = psi_N^(2k), a primitive d-th root.
            const Data64 omega = powmod(psi_n[l], 2ull * k, p);
            const Data64 iomega = invmod(omega, p);
            Data64 accf = 1, acci = 1;
            for (int m = 0; m < d; ++m)
            {
                wf[static_cast<size_t>(l) * d + m] = accf;
                wi[static_cast<size_t>(l) * d + m] = acci;
                accf = static_cast<Data64>(
                    (static_cast<__uint128_t>(accf) * omega) % p);
                acci = static_cast<Data64>(
                    (static_cast<__uint128_t>(acci) * iomega) % p);
            }

            // psi_N^(+-h0(s)*i); geometric in i, so one powmod per slot. The
            // inverse table folds in the 1/d of the inverse DFT.
            const Data64 dinv = invmod(static_cast<Data64>(d) % p, p);
            for (int s = 0; s < k; ++s)
            {
                const Data64 h0 =
                    2ull * bitrev(static_cast<uint32_t>(s), logk) + 1ull;
                const Data64 bf = powmod(psi_n[l], h0, p);
                const Data64 bi = invmod(bf, p);
                Data64 cf = 1, ci = dinv;
                for (int i = 0; i < d; ++i)
                {
                    const size_t off =
                        (static_cast<size_t>(l) * k + s) * d + i;
                    pf[off] = cf;
                    pin[off] = ci;
                    cf = static_cast<Data64>(
                        (static_cast<__uint128_t>(cf) * bf) % p);
                    ci = static_cast<Data64>(
                        (static_cast<__uint128_t>(ci) * bi) % p);
                }
            }
        }

        BatchSubringTables t;
        t.k = k;
        t.d = d;
        t.num_limbs = num_limbs;
        t.psi = DeviceVector<Data64>(psi);
        t.psi_inv = DeviceVector<Data64>(psi_i);
        t.kinv = DeviceVector<Data64>(kinv);
        t.wfwd = DeviceVector<Data64>(wf);
        t.winv = DeviceVector<Data64>(wi);
        t.psi_fwd_n = DeviceVector<Data64>(pf);
        t.psi_inv_n = DeviceVector<Data64>(pin);
        t.psi_n = DeviceVector<Data64>(psi_n);
        {
            // Every power of psi_N, so the monomial multiply reduces to one
            // table load. Built by repeated multiplication: 2N muls per limb
            // once per level, against one modular exponentiation per element
            // per call if the kernel derived it itself.
            const size_t two_n = 2ull * static_cast<size_t>(n_);
            std::vector<Data64> ppow(static_cast<size_t>(num_limbs) * two_n);
            for (int l = 0; l < num_limbs; ++l)
            {
                const Data64 p = primes[l].value;
                Data64 acc = 1;
                Data64* row = ppow.data() + static_cast<size_t>(l) * two_n;
                for (size_t e = 0; e < two_n; ++e)
                {
                    row[e] = acc;
                    acc = static_cast<Data64>(
                        (static_cast<__uint128_t>(acc) * psi_n[l]) % p);
                }
            }
            t.psi_n_pow = DeviceVector<Data64>(ppow);
        }
        {
            std::vector<Data64> dv(num_limbs);
            for (int l = 0; l < num_limbs; ++l)
                dv[l] = invmod(static_cast<Data64>(d) % primes[l].value,
                               primes[l].value);
            t.dinv = DeviceVector<Data64>(dv);
        }
        t.modulus = DeviceVector<Modulus64>(primes);

        return table_cache_.emplace(depth, std::move(t)).first->second;
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::encode_plaintext_matrix(
        const std::vector<int64_t>& coeffs, int rows, int cols, int depth,
        double scale)
    {
        const int k = layout_.k;
        const size_t per_limb = static_cast<size_t>(rows) * cols * k;
        if (coeffs.size() != per_limb)
            throw std::invalid_argument(
                "encoded plaintext matrix has the wrong length");

        const BatchSubringTables& t = tables_for(depth);
        const int num_limbs = t.num_limbs;

        std::vector<Modulus64> all = context_->get_key_modulus();
        std::vector<Data64> host(static_cast<size_t>(num_limbs) * per_limb);
        for (int l = 0; l < num_limbs; ++l)
        {
            const Data64 p = all[l].value;
            for (size_t i = 0; i < per_limb; ++i)
                host[static_cast<size_t>(l) * per_limb + i] =
                    centered_to_modular(coeffs[i], p);
        }

        plain_ = DeviceVector<Data64>(host);

        const int threads = (k < 256) ? k : 256;
        const size_t shared = static_cast<size_t>(k) * sizeof(Data64);
        // The whole length-k transform lives in shared memory, so a large
        // subring outgrows the 48 KB a block gets by default. Anything above
        // that has to be opted into explicitly, once per kernel.
        if (shared > 48 * 1024)
            HEONGPU_CUDA_CHECK(cudaFuncSetAttribute(
                bm_ntt_k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared)));
        const dim3 grid(static_cast<unsigned>(per_limb / k),
                        static_cast<unsigned>(num_limbs));
        bm_ntt_k_kernel<<<grid, threads, shared>>>(
            plain_.data(), t.psi.data(), t.modulus.data(), k,
            static_cast<int>(per_limb / k));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plain_rows_ = rows;
        plain_cols_ = cols;
        plain_depth_ = depth;
        plain_scale_ = scale;
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::load_coefficients(
        Plaintext<Scheme::CKKS>& plain, const std::vector<int64_t>& coeffs,
        double scale)
    {
        if (static_cast<int>(coeffs.size()) != n_)
            throw std::invalid_argument(
                "coefficient vector must have length N");

        const int depth = plain.depth_;
        const int num_limbs = q_size_ - depth;
        std::vector<Modulus64> all = context_->get_key_modulus();

        std::vector<Data64> host(static_cast<size_t>(num_limbs) * n_);
        for (int l = 0; l < num_limbs; ++l)
        {
            const Data64 p = all[l].value;
            for (int i = 0; i < n_; ++i)
                host[static_cast<size_t>(l) * n_ + i] =
                    centered_to_modular(coeffs[i], p);
        }

        DeviceVector<Data64> mem(host);

        // Plain assignment rather than designated initializers: nvcc has been
        // observed to silently drop initializers in this struct, which leaves
        // mod_inverse/stream holding garbage.
        gpuntt::ntt_rns_configuration<Data64> cfg{};
        cfg.n_power = context_->n_power;
        cfg.ntt_type = gpuntt::FORWARD;
        cfg.ntt_layout = gpuntt::PerPolynomial;
        cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
        cfg.zero_padding = false;
        cfg.mod_inverse = nullptr;
        cfg.stream = cudaStreamDefault;

        gpuntt::GPU_NTT_Inplace(mem.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg, num_limbs,
                                num_limbs);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // memory_set resets the plaintext's descriptive fields, so the domain
        // flag and scale must be stamped after it, not before.
        plain.memory_set(std::move(mem));
        plain.scale_ = scale;
        plain.in_ntt_domain_ = true;
        plain.depth_ = depth;
        // memory_set does not update the descriptor, and HEEncryptor checks
        // size() against n * Q_size. At Q_size == 1 the default happens to
        // satisfy that, so this only bites once there is more than one limb.
        plain.plain_size_ = static_cast<int>(host.size());
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::mult_monomial_batch(
        const std::vector<Data64*>& ct, const std::vector<int>& powers,
        int depth)
    {
        if (ct.empty())
            return;
        if (ct.size() != powers.size())
            throw std::invalid_argument(
                "mult_monomial_batch needs one power per ciphertext");

        const BatchSubringTables& t = tables_for(depth);
        DeviceVector<Data64*> dp(ct);
        DeviceVector<int> dpow(powers);

        const int threads = 256;
        const dim3 grid(static_cast<unsigned>((n_ + threads - 1) / threads),
                        static_cast<unsigned>(t.num_limbs),
                        static_cast<unsigned>(ct.size()));
        bm_mult_monomial_batch_kernel<<<grid, threads>>>(
            dp.data(), dpow.data(), t.psi_n_pow.data(), t.modulus.data(),
            context_->n_power, t.num_limbs);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        // The pointer and power buffers are freed when this returns, so the
        // launch has to have consumed them first.
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::mult_monomial(
        Ciphertext<Scheme::CKKS>& ct, int power)
    {
        const int two_n = 2 * n_;
        int e = power % two_n;
        if (e < 0)
            e += two_n;
        if (e == 0)
            return;

        mult_monomial_batch({ct.data()}, {e}, ct.depth_);
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::tweak(
        std::vector<Ciphertext<Scheme::CKKS>>& ct, int sgn)
    {
        const int d = layout_.d;
        if (static_cast<int>(ct.size()) != d)
            throw std::invalid_argument("tweak expects exactly d ciphertexts");

        // Iterative Cooley-Tukey rather than the recursion of Algorithm 2.
        // Same butterflies in the same order, but a whole stage is one launch:
        // the recursion issued one monomial multiply and one butterfly per
        // pair, 129 + 192 launches at d = 64, each far too small to fill the
        // device.
        //
        // Decimation in time, so the working order is bit-reversed: logical
        // position i is held by ct[brv(i)] throughout, and only the final
        // reorder touches the caller's vector.
        const int logd = log2i(d);
        std::vector<int> slot(d);
        for (int i = 0; i < d; ++i)
            slot[i] = static_cast<int>(bitrev(static_cast<uint32_t>(i), logd));

        const BatchSubringTables& t = tables_for(ct[0].depth_);
        const long long two_n = 2ll * n_;
        const int threads = 256;

        for (int len = 2; len <= d; len <<= 1)
        {
            const int half = len >> 1;
            // The subproblem of size len sits at the recursion level whose
            // modulus parameter is k * (d / len).
            const long long k_level =
                static_cast<long long>(layout_.k) * (d / len);

            std::vector<Data64*> even, odd;
            std::vector<int> powers;
            even.reserve(d / 2);
            odd.reserve(d / 2);
            powers.reserve(d / 2);
            for (int start = 0; start < d; start += len)
                for (int j = 0; j < half; ++j)
                {
                    even.push_back(ct[slot[start + j]].data());
                    odd.push_back(ct[slot[start + j + half]].data());
                    const long long e = 2ll * k_level * j * sgn;
                    powers.push_back(
                        static_cast<int>(((e % two_n) + two_n) % two_n));
                }

            DeviceVector<Data64*> de(even), dod(odd);
            DeviceVector<int> dpow(powers);
            const dim3 grid(static_cast<unsigned>((n_ + threads - 1) / threads),
                            static_cast<unsigned>(t.num_limbs),
                            static_cast<unsigned>(even.size()));
            bm_tweak_stage_kernel<<<grid, threads>>>(
                de.data(), dod.data(), dpow.data(), t.psi_n_pow.data(),
                t.modulus.data(), context_->n_power, t.num_limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Logical result i lives in ct[brv(i)]. Bit reversal is an involution,
        // so this is a permutation and every element is moved exactly once --
        // pointer moves, not the d device-to-device copies this used to cost.
        std::vector<Ciphertext<Scheme::CKKS>> result;
        result.reserve(d);
        for (int i = 0; i < d; ++i)
            result.push_back(std::move(ct[slot[i]]));
        ct = std::move(result);
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::cmt(
        std::vector<Ciphertext<Scheme::CKKS>>& ct,
        Galoiskey<Scheme::CKKS>& galois_key,
        HEArithmeticOperator<Scheme::CKKS>& ops)
    {
        const int d = layout_.d;
        const int k = layout_.k;
        if (static_cast<int>(ct.size()) != d)
            throw std::invalid_argument("cmt expects exactly d ciphertexts");

        BmRange _r_cmt("CMT");

        // Step 1: ct_i <- X^i * ct_i. One launch: index 0 carries power 0 and
        // is skipped on the device.
        {
            BmRange _r("CMT.step1_monomial");
            std::vector<Data64*> base(d);
            std::vector<int> powers(d);
            for (int i = 0; i < d; ++i)
            {
                base[i] = ct[i].data();
                powers[i] = i;
            }
            mult_monomial_batch(base, powers, ct[0].depth_);
        }

        // Step 2
        {
            BmRange _r("CMT.step2_tweak_fwd");
            tweak(ct, +1);
        }

        // Step 3: scale by d^-1. The map t -> t* is a bijection, so scaling
        // every ciphertext once is equivalent to the per-t scaling written in
        // the algorithm.
        {
            BmRange _r("CMT.step3_scale");
            const BatchSubringTables& t = tables_for(ct[0].depth_);
            std::vector<Data64*> base(d);
            for (int i = 0; i < d; ++i)
                base[i] = ct[i].data();
            DeviceVector<Data64*> dp(base);

            const int threads = 256;
            const dim3 grid(static_cast<unsigned>((n_ + threads - 1) / threads),
                            static_cast<unsigned>(t.num_limbs),
                            static_cast<unsigned>(d));
            bm_mult_scalar_batch_kernel<<<grid, threads>>>(
                dp.data(), t.dinv.data(), t.modulus.data(), context_->n_power,
                t.num_limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        }

        const uint64_t mod = 2ull * static_cast<uint64_t>(n_);
        std::map<uint64_t, int> galois_to_index;
        {
            uint64_t g = 1;
            for (int r = 0; r < n_ / 2; ++r)
            {
                galois_to_index.emplace(g, r);
                g = (g * 5) % mod;
            }
        }

        // t -> t* is a bijection, so the permutation is a pure reordering and
        // each automorphism then runs in place.
        std::vector<int> rot_index(d, 0);
        std::vector<int> source(d, 0);
        std::vector<bool> taken(d, false);
        for (int t = 0; t < d; ++t)
        {
            const uint64_t h = (2ull * k * t + 1) % mod;
            const uint64_t hinv = invmod_generic(h, mod);
            const int tstar = static_cast<int>((hinv - 1) / (2ull * k));
            if (tstar < 0 || tstar >= d)
                throw std::runtime_error(
                    "inverse Galois element fell outside the CMT index range");
            auto it = galois_to_index.find(h);
            if (it == galois_to_index.end())
                throw std::runtime_error("automorphism X -> X^(2kt+1) is not a "
                                         "slot rotation for this layout");
            rot_index[t] = it->second;
            // Reordering by moving is only sound because t -> t* is injective;
            // a repeat would leave a moved-from ciphertext behind rather than
            // fail, so check it rather than trust it.
            if (taken[tstar])
                throw std::runtime_error(
                    "t -> t* is not injective; the CMT reordering would drop a "
                    "ciphertext");
            taken[tstar] = true;
            source[t] = tstar;
        }

        {
            std::vector<Ciphertext<Scheme::CKKS>> permuted;
            permuted.reserve(d);
            for (int t = 0; t < d; ++t)
                permuted.push_back(std::move(ct[source[t]]));
            ct = std::move(permuted);
        }

        {
            BmRange _r("CMT.automorphisms");
            for (int t = 0; t < d; ++t)
                if (rot_index[t] != 0)
                    ops.rotate_rows_inplace(ct[t], galois_key, rot_index[t]);
        }

        // Step 4
        {
            BmRange _r("CMT.step4_tweak_inv");
            tweak(ct, -1);
        }

        // Step 5: ct'_i <- X^-i * ct'_i
        {
            BmRange _r("CMT.step5_monomial");
            std::vector<Data64*> base(d);
            std::vector<int> powers(d);
            for (int i = 0; i < d; ++i)
            {
                base[i] = ct[i].data();
                powers[i] = (2 * n_ - i) % (2 * n_);
            }
            mult_monomial_batch(base, powers, ct[0].depth_);
        }
    }

    std::vector<int64_t> HEBatchMatrixOperator<Scheme::CKKS>::ntt_intt_probe(
        const std::vector<int64_t>& coeffs)
    {
        const Data64 p = context_->get_key_modulus()[0].value;

        std::vector<Data64> host(n_);
        for (int i = 0; i < n_; ++i)
            host[i] = centered_to_modular(coeffs[i], p);

        DeviceVector<Data64> mem(host);

        gpuntt::ntt_rns_configuration<Data64> fwd{};
        fwd.n_power = context_->n_power;
        fwd.ntt_type = gpuntt::FORWARD;
        fwd.ntt_layout = gpuntt::PerPolynomial;
        fwd.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
        fwd.zero_padding = false;
        fwd.mod_inverse = nullptr;
        fwd.stream = cudaStreamDefault;
        gpuntt::GPU_NTT_Inplace(mem.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(), fwd, 1, 1);

        gpuntt::ntt_rns_configuration<Data64> inv{};
        inv.n_power = context_->n_power;
        inv.ntt_type = gpuntt::INVERSE;
        inv.ntt_layout = gpuntt::PerPolynomial;
        inv.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
        inv.zero_padding = false;
        inv.mod_inverse = context_->n_inverse_->data();
        inv.stream = cudaStreamDefault;
        // The inverse transform is a distinct entry point; GPU_NTT_Inplace
        // does not perform it even with ntt_type = INVERSE.
        gpuntt::GPU_INTT_Inplace(mem.data(), context_->intt_table_->data(),
                                 context_->modulus_->data(), inv, 1, 1);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        std::vector<Data64> back(n_);
        cudaMemcpy(back.data(), mem.data(), n_ * sizeof(Data64),
                   cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        const Data64 half = p >> 1;
        std::vector<int64_t> out(n_);
        for (int i = 0; i < n_; ++i)
            out[i] = (back[i] > half) ? -static_cast<int64_t>(p - back[i])
                                      : static_cast<int64_t>(back[i]);
        return out;
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::extract_coefficients(
        std::vector<int64_t>& coeffs, Plaintext<Scheme::CKKS>& plain)
    {
        const int depth = plain.depth_;
        const int num_limbs = q_size_ - depth;
        if (num_limbs < 1)
            throw std::invalid_argument(
                "extract_coefficients needs at least one active limb");

        // The plaintext may be resident on the host, in which case data()
        // returns a host pointer; cudaMemcpyDefault resolves either case
        // through unified addressing.
        if (!plain.is_on_device())
            plain.store_in_device();

        const size_t elems = static_cast<size_t>(num_limbs) * n_;
        DeviceVector<Data64> mem(elems);
        cudaMemcpy(mem.data(), plain.data(), elems * sizeof(Data64),
                   cudaMemcpyDefault);

        if (plain.in_ntt_domain_)
        {
            gpuntt::ntt_rns_configuration<Data64> cfg{};
            cfg.n_power = context_->n_power;
            cfg.ntt_type = gpuntt::INVERSE;
            cfg.ntt_layout = gpuntt::PerPolynomial;
            cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            cfg.zero_padding = false;
            cfg.mod_inverse = context_->n_inverse_->data();
            cfg.stream = cudaStreamDefault;
            gpuntt::GPU_INTT_Inplace(mem.data(), context_->intt_table_->data(),
                                     context_->modulus_->data(), cfg,
                                     num_limbs, num_limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        std::vector<Data64> host(elems);
        cudaMemcpy(host.data(), mem.data(), elems * sizeof(Data64),
                   cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<Modulus64> all = context_->get_key_modulus();
        coeffs.resize(n_);

        if (num_limbs == 1)
        {
            const Data64 p = all[0].value;
            const Data64 half = p >> 1;
            for (int i = 0; i < n_; ++i)
                coeffs[i] = (host[i] > half)
                                ? -static_cast<int64_t>(p - host[i])
                                : static_cast<int64_t>(host[i]);
            return;
        }

        // Garner reconstruction. Writing P_l for prod_{j<l} p_j, the value is
        //   x = sum_l t_l * P_l  with  t_l = (r_l - (x mod p_l)) * P_l^-1,
        // the mixed radix digits being computed one limb at a time. Only the
        // digits are modular; the sum itself is exact and reaches the full
        // width of prod p_l, so it is accumulated in a big integer. One spare
        // word above the limb count absorbs every carry.
        std::vector<Data64> p(num_limbs);
        for (int l = 0; l < num_limbs; ++l)
            p[l] = all[l].value;

        std::vector<Data64> pinv(num_limbs, 0);
        for (int l = 1; l < num_limbs; ++l)
        {
            Data64 acc = p[0] % p[l];
            for (int j = 1; j < l; ++j)
                acc = mulmod(acc, p[j] % p[l], p[l]);
            pinv[l] = invmod(acc, p[l]);
        }

        const size_t words = static_cast<size_t>(num_limbs) + 1;
        BigUInt modulus(words, 0);
        modulus[0] = 1;
        for (int l = 0; l < num_limbs; ++l)
            big_mul_word(modulus, p[l]);
        BigUInt half = modulus;
        big_shr1(half);

        std::vector<Data64> digit(num_limbs);
        BigUInt x(words, 0), radix(words, 0);

        for (int i = 0; i < n_; ++i)
        {
            digit[0] = host[i] % p[0];
            for (int l = 1; l < num_limbs; ++l)
            {
                Data64 partial = digit[0] % p[l];
                Data64 weight = 1;
                for (int j = 1; j < l; ++j)
                {
                    weight = mulmod(weight, p[j - 1] % p[l], p[l]);
                    partial = (partial + mulmod(digit[j], weight, p[l])) % p[l];
                }
                const Data64 r =
                    host[static_cast<size_t>(l) * n_ + i] % p[l];
                const Data64 diff =
                    (r >= partial) ? (r - partial) : (r + p[l] - partial);
                digit[l] = mulmod(diff, pinv[l], p[l]);
            }

            std::fill(x.begin(), x.end(), 0);
            std::fill(radix.begin(), radix.end(), 0);
            radix[0] = 1;
            for (int l = 0; l < num_limbs; ++l)
            {
                big_add_scaled(x, radix, digit[l]);
                big_mul_word(radix, p[l]);
            }

            const bool negative = big_cmp(x, half) > 0;
            if (negative)
                big_sub(x, modulus);

            // The centered value is only meaningful to a caller if it fits an
            // int64; past that the plaintext has outgrown what this can report
            // and narrowing would silently wrap. In two's complement that means
            // every word above the first must be a pure sign extension.
            const Data64 fill = negative ? ~Data64{0} : Data64{0};
            for (size_t w = 1; w < words; ++w)
                if (x[w] != fill)
                    throw std::runtime_error(
                        "extracted coefficient does not fit in int64; the "
                        "plaintext exceeds the representable range");
            if (((x[0] >> 63) != 0) != negative)
                throw std::runtime_error(
                    "extracted coefficient does not fit in int64; the "
                    "plaintext exceeds the representable range");

            coeffs[i] = static_cast<int64_t>(x[0]);
        }
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::subring_round_trip(
        const std::vector<Ciphertext<Scheme::CKKS>*>& ct)
    {
        if (ct.empty())
            throw std::invalid_argument("no ciphertexts");

        const int d = layout_.d;
        const int k = layout_.k;
        const int cols = static_cast<int>(ct.size());

        const int depth = ct[0]->depth_;
        const BatchSubringTables& t = tables_for(depth);
        const int num_limbs = t.num_limbs;
        const size_t limb_stride = static_cast<size_t>(n_);
        const size_t comp_stride = static_cast<size_t>(num_limbs) * n_;

        std::vector<Data64*> base(cols);
        for (int j = 0; j < cols; ++j)
            base[j] = ct[j]->data();

        auto make_ptrs = [&](bool second_component)
        {
            std::vector<Data64*> h(static_cast<size_t>(num_limbs) * cols);
            for (int l = 0; l < num_limbs; ++l)
                for (int j = 0; j < cols; ++j)
                    h[static_cast<size_t>(l) * cols + j] =
                        base[j] + (second_component ? comp_stride : 0) +
                        static_cast<size_t>(l) * limb_stride;
            return h;
        };

        const size_t elems = static_cast<size_t>(num_limbs) * d * cols * k;
        DeviceVector<Data64> tmp(elems);
        const int dthreads = (d < 256) ? d : 256;
        const dim3 grid(static_cast<unsigned>(k),
                        static_cast<unsigned>(cols),
                        static_cast<unsigned>(num_limbs));

        for (int comp = 0; comp < 2; ++comp)
        {
            DeviceVector<Data64*> ptrs(make_ptrs(comp == 1));
            bm_ntt_to_subring_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                tmp.data(), ptrs.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, cols, k);
            bm_subring_to_ntt_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                ptrs.data(), tmp.data(), t.wfwd.data(), t.psi_fwd_n.data(),
                t.modulus.data(), d, cols, k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::pcmm(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& in, bool rescale)
    {
        if (plain_depth_ < 0)
            throw std::logic_error(
                "no plaintext matrix uploaded; call encode_plaintext_matrix");
        if (in.empty())
            throw std::invalid_argument("no input ciphertexts");
        if (static_cast<int>(in.size()) != plain_rows_)
            throw std::invalid_argument("number of input ciphertexts must "
                                        "equal the plaintext row count");

        const int d = layout_.d;
        const int k = layout_.k;
        const int inner = plain_rows_;
        const int cols_out = plain_cols_;

        const int depth = in[0]->depth_;
        if (depth != plain_depth_)
            throw std::invalid_argument(
                "plaintext matrix level does not match the ciphertexts");
        for (const auto* c : in)
            if (c->depth_ != depth)
                throw std::invalid_argument(
                    "all input ciphertexts must share a level");

        const BatchSubringTables& t = tables_for(depth);
        const int num_limbs = t.num_limbs;
        const size_t limb_stride = static_cast<size_t>(n_);
        const size_t comp_stride = static_cast<size_t>(num_limbs) * n_;

        out.clear();
        out.reserve(cols_out);
        for (int j = 0; j < cols_out; ++j)
            out.emplace_back(*in[0]); // inherits level, scale and shape

        // Pointer tables: [limb][col] -> start of that limb inside the
        // component, so the kernels can address each ciphertext independently.
        auto make_ptrs = [&](const std::vector<Data64*>& base, int count,
                             bool second_component)
        {
            std::vector<Data64*> h(static_cast<size_t>(num_limbs) * count);
            for (int l = 0; l < num_limbs; ++l)
                for (int j = 0; j < count; ++j)
                    h[static_cast<size_t>(l) * count + j] =
                        base[j] + (second_component ? comp_stride : 0) +
                        static_cast<size_t>(l) * limb_stride;
            return h;
        };

        std::vector<Data64*> in_base(inner), out_base(cols_out);
        for (int j = 0; j < inner; ++j)
            in_base[j] = in[j]->data();
        for (int j = 0; j < cols_out; ++j)
            out_base[j] = out[j].data();

        const size_t in_elems =
            static_cast<size_t>(num_limbs) * d * inner * k;
        const size_t out_elems =
            static_cast<size_t>(num_limbs) * d * cols_out * k;

        DeviceVector<Data64> A0(in_elems), A1(in_elems);
        DeviceVector<Data64> C0(out_elems), C1(out_elems);

        DeviceVector<Data64*> src0(make_ptrs(in_base, inner, false));
        DeviceVector<Data64*> src1(make_ptrs(in_base, inner, true));
        DeviceVector<Data64*> dst0(make_ptrs(out_base, cols_out, false));
        DeviceVector<Data64*> dst1(make_ptrs(out_base, cols_out, true));

        const int dthreads = (d < 256) ? d : 256;

        // Length-N NTT domain -> R_k NTT domain, both components.
        {
            const dim3 grid(static_cast<unsigned>(k),
                            static_cast<unsigned>(inner),
                            static_cast<unsigned>(num_limbs));
            bm_ntt_to_subring_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                A0.data(), src0.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, inner, k);
            bm_ntt_to_subring_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                A1.data(), src1.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, inner, k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // Two matrix products over R_{q,k}, one per ciphertext component.
        {
            const int threads = (k < 256) ? k : 256;
            const int s_blocks = (k + threads - 1) / threads;
            const dim3 grid(static_cast<unsigned>(s_blocks * d * cols_out), 1u,
                            static_cast<unsigned>(num_limbs));
            bm_gemm_kernel<<<grid, threads>>>(C0.data(), A0.data(),
                                              plain_.data(), t.modulus.data(),
                                              d, inner, cols_out, k, cols_out,
                                              1, s_blocks);
            bm_gemm_kernel<<<grid, threads>>>(C1.data(), A1.data(),
                                              plain_.data(), t.modulus.data(),
                                              d, inner, cols_out, k, cols_out,
                                              1, s_blocks);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        {
            const dim3 grid(static_cast<unsigned>(k),
                            static_cast<unsigned>(cols_out),
                            static_cast<unsigned>(num_limbs));
            bm_subring_to_ntt_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                dst0.data(), C0.data(), t.wfwd.data(), t.psi_fwd_n.data(),
                t.modulus.data(), d, cols_out, k);
            bm_subring_to_ntt_kernel<<<grid, dthreads, d * sizeof(Data64)>>>(
                dst1.data(), C1.data(), t.wfwd.data(), t.psi_fwd_n.data(),
                t.modulus.data(), d, cols_out, k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2 of Algorithm 1: the product carries the plaintext scaling
        // factor. Mark it so the caller's rescale removes exactly that.
        for (int j = 0; j < cols_out; ++j)
        {
            out[j].scale_ = in[0]->scale_ * plain_scale_;
            out[j].rescale_required_ = rescale;
        }
    }

    // -----------------------------------------------------------------------
    // Rectangular matrix multiplication (Algorithm 5)
    // -----------------------------------------------------------------------

    HEBatchMatrixOperator<Scheme::CKKS>&
    HEBatchMatrixOperator<Scheme::CKKS>::half_operator()
    {
        if (!half_)
        {
            half_.reset(new HEBatchMatrixOperator<Scheme::CKKS>(
                context_, BatchMatrixLayout(n_, n_ / 2)));
        }
        return *half_;
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::mult_int_scalar_batch(
        std::vector<Ciphertext<Scheme::CKKS>>& ct, uint64_t value)
    {
        if (ct.empty() || value == 1)
            return;

        const BatchSubringTables& t = tables_for(ct[0].depth_);

        // A plain integer is the constant polynomial, so this is exact in the
        // NTT domain and costs neither a level nor a rescale.
        const std::vector<Modulus64> all = context_->get_key_modulus();
        std::vector<Data64> scalar(t.num_limbs);
        for (int l = 0; l < t.num_limbs; ++l)
            scalar[l] = value % all[l].value;
        DeviceVector<Data64> dscalar(scalar);

        std::vector<Data64*> base(ct.size());
        for (size_t j = 0; j < ct.size(); ++j)
            base[j] = ct[j].data();
        DeviceVector<Data64*> dp(base);

        const int threads = 256;
        const dim3 grid(static_cast<unsigned>((n_ + threads - 1) / threads),
                        static_cast<unsigned>(t.num_limbs),
                        static_cast<unsigned>(ct.size()));
        bm_mult_scalar_batch_kernel<<<grid, threads>>>(
            dp.data(), dscalar.data(), t.modulus.data(), context_->n_power,
            t.num_limbs);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::rectangular_pcmm(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& in,
        Galoiskey<Scheme::CKKS>& galois_key,
        HEArithmeticOperator<Scheme::CKKS>& ops, BlockAxis axis, bool rescale)
    {
        const int d = layout_.d;
        const int k = layout_.k;
        const int half = n_ / 2;

        if (static_cast<int>(in.size()) != d)
            throw std::invalid_argument(
                "rectangular_pcmm expects exactly d ciphertexts");
        if (plain_rows_ != d || plain_cols_ != half)
            throw std::invalid_argument(
                "rectangular_pcmm expects a d x (N/2) plaintext matrix");

        BmRange _r_rect("RectangularPCMM");

        // Step 1: the k/2 block products, in one batch PCMM.
        //
        // The rescale is deliberately NOT marked here even when the caller
        // asked for one. rotate_rows refuses a ciphertext carrying an unspent
        // rescale, and the summation below is nothing but rotations, so
        // marking it now would make step 3 throw. The mark goes on at the end
        // instead, which changes no arithmetic: key switching is scale
        // agnostic, and the product sits at in_scale * plain_scale either way.
        {
            BmRange _r("RectangularPCMM.blocks");
            pcmm(out, in, /*rescale=*/false);
        }

        // Lemma 1: the constant term of an R_k element is (2/k) times the sum
        // over its k/2 evaluation points. When the block index rides on the
        // batch axis the sum over blocks is what step 3 has to produce, so the
        // product is scaled to carry the sum rather than the mean. When it
        // rides on the Y axis the constant term is ALREADY the contraction
        // over blocks and scaling it would be wrong, not merely wasteful.
        if (axis == BlockAxis::slot)
        {
            BmRange _r("RectangularPCMM.lemma1_scale");
            mult_int_scalar_batch(out, static_cast<uint64_t>(k / 2));
        }

        // Steps 2 and 3, the summation with encoding conversion of Theorem 3.
        //
        // The constant terms sit at coefficients 0..d-1 of each of the N/2
        // intermediate columns, and they have to end up at coefficient
        // i + d*t of column j, where the intermediate column was t*d + j.
        // Reading the same N coefficients as a batch matrix at k = 2 makes
        // that gather a transpose: a CMT at (N, N/2) moves the constant terms
        // into the LEADING d ciphertexts, the rest are dropped outright, and a
        // CMT at the caller's own layout restores the column-wise encryption.
        {
            BmRange _r("RectangularPCMM.summation");
            half_operator().cmt(out, galois_key, ops);

            // pop_back rather than resize: Ciphertext has no default
            // constructor, so resize would not compile against the shrink.
            while (static_cast<int>(out.size()) > d)
                out.pop_back();

            cmt(out, galois_key, ops);
        }

        // Step 2 of Algorithm 1, deferred to here: the result carries the
        // plaintext scaling factor and the caller's rescale removes exactly
        // that.
        for (auto& c : out)
            c.rescale_required_ = rescale;
    }

    std::vector<Data64*>
    HEBatchMatrixOperator<Scheme::CKKS>::component_pointers(
        const std::vector<Data64*>& base, int count, bool second_component,
        int num_limbs) const
    {
        const size_t limb_stride = static_cast<size_t>(n_);
        const size_t comp_stride = static_cast<size_t>(num_limbs) * n_;

        std::vector<Data64*> h(static_cast<size_t>(num_limbs) * count);
        for (int l = 0; l < num_limbs; ++l)
            for (int j = 0; j < count; ++j)
                h[static_cast<size_t>(l) * count + j] =
                    base[j] + (second_component ? comp_stride : 0) +
                    static_cast<size_t>(l) * limb_stride;
        return h;
    }

    Ciphertext<Scheme::CKKS> HEBatchMatrixOperator<Scheme::CKKS>::allocate_like(
        const Ciphertext<Scheme::CKKS>& src, size_t elems) const
    {
        Ciphertext<Scheme::CKKS> c;
        c.scheme_ = src.scheme_;
        c.ring_size_ = src.ring_size_;
        c.coeff_modulus_count_ = src.coeff_modulus_count_;
        c.cipher_size_ = src.cipher_size_;
        c.depth_ = src.depth_;
        c.in_ntt_domain_ = src.in_ntt_domain_;
        c.scale_ = src.scale_;
        c.encoding_ = src.encoding_;
        c.rescale_required_ = src.rescale_required_;
        c.relinearization_required_ = src.relinearization_required_;
        c.ciphertext_generated_ = true;
        c.storage_type_ = storage_type::DEVICE;
        c.memory_set(DeviceVector<Data64>(elems));
        return c;
    }

    void HEBatchMatrixOperator<Scheme::CKKS>::ccmm(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& a,
        const std::vector<Ciphertext<Scheme::CKKS>*>& b,
        Galoiskey<Scheme::CKKS>& galois_key, Relinkey<Scheme::CKKS>& relin_key,
        HEArithmeticOperator<Scheme::CKKS>& ops, bool rescale)
    {
        const int d = layout_.d;
        const int k = layout_.k;
        if (static_cast<int>(a.size()) != d || static_cast<int>(b.size()) != d)
            throw std::invalid_argument(
                "ccmm expects exactly d ciphertexts per operand");

        const int depth = a[0]->depth_;
        for (const auto* c : a)
            if (c->depth_ != depth)
                throw std::invalid_argument(
                    "all left operand ciphertexts must share a level");
        for (const auto* c : b)
            if (c->depth_ != depth)
                throw std::invalid_argument(
                    "both operands must sit at the same level");

        const BatchSubringTables& t = tables_for(depth);
        const int num_limbs = t.num_limbs;
        const size_t comp_stride = static_cast<size_t>(num_limbs) * n_;
        const size_t elems = static_cast<size_t>(num_limbs) * d * d * k;

        // Step 1: the right operand becomes a row-wise matrix encryption. The
        // transpose that implies is applied by the GEMM strides below rather
        // than by moving data.
        BmRange _r_ccmm("CCMM");

        std::vector<Ciphertext<Scheme::CKKS>> bcmt;
        {
            BmRange _r("CCMM.step1_cmt_right");
            bcmt.reserve(d);
            for (int j = 0; j < d; ++j)
                bcmt.push_back(*b[j]);
            cmt(bcmt, galois_key, ops);
        }

        std::vector<Data64*> a_base(d), b_base(d);
        for (int j = 0; j < d; ++j)
        {
            a_base[j] = a[j]->data();
            b_base[j] = bcmt[j].data();
        }

        DeviceVector<Data64> A0(elems), A1(elems), B0(elems), B1(elems);
        const int dthreads = (d < 256) ? d : 256;
        {
            BmRange _r("CCMM.step2_ntt_to_subring");
            DeviceVector<Data64*> pa0(
                component_pointers(a_base, d, false, num_limbs));
            DeviceVector<Data64*> pa1(
                component_pointers(a_base, d, true, num_limbs));
            DeviceVector<Data64*> pb0(
                component_pointers(b_base, d, false, num_limbs));
            DeviceVector<Data64*> pb1(
                component_pointers(b_base, d, true, num_limbs));

            const dim3 grid(static_cast<unsigned>(k),
                            static_cast<unsigned>(d),
                            static_cast<unsigned>(num_limbs));
            const size_t shared = d * sizeof(Data64);
            bm_ntt_to_subring_kernel<<<grid, dthreads, shared>>>(
                A0.data(), pa0.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, d, k);
            bm_ntt_to_subring_kernel<<<grid, dthreads, shared>>>(
                A1.data(), pa1.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, d, k);
            bm_ntt_to_subring_kernel<<<grid, dthreads, shared>>>(
                B0.data(), pb0.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, d, k);
            bm_ntt_to_subring_kernel<<<grid, dthreads, shared>>>(
                B1.data(), pb1.data(), t.winv.data(), t.psi_inv_n.data(),
                t.modulus.data(), d, d, k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step 2, already transposed for steps 3 and 4.
        //
        // Step 2 proper is (C00,C01,C10,C11) = (A0,A0,A1,A1) * (B0,B1,B0,B1)^T,
        // which leaves each half row-wise: C_x0 + C_x1 * Toep(sk)^T is the half
        // product. Steps 3 and 4 then want (C_xy)^T, because transposing that
        // identity gives C_x0^T + Toep(sk) * C_x1^T, the column-wise form a CMT
        // consumes. Since (X * Y^T)^T = Y * X^T, swapping the two operands
        // produces the transpose directly: the left operand becomes the one
        // read with swapped strides, and nothing has to move afterwards. One
        // stride trick therefore covers both transposes Algorithm 4 asks for --
        // the (.)^T on the right operand of step 2, and the one on the whole
        // product in steps 3 and 4.
        DeviceVector<Data64> C00(elems), C01(elems), C10(elems), C11(elems);
        {
            BmRange _r("CCMM.step2_gemm");
            const int threads = (k < 256) ? k : 256;
            const int s_blocks = (k + threads - 1) / threads;
            const dim3 grid(static_cast<unsigned>(s_blocks * d * d), 1u,
                            static_cast<unsigned>(num_limbs));
            bm_gemm_kernel<<<grid, threads>>>(C00.data(), B0.data(), A0.data(),
                                              t.modulus.data(), d, d, d, k, 1,
                                              d, s_blocks);
            bm_gemm_kernel<<<grid, threads>>>(C01.data(), B1.data(), A0.data(),
                                              t.modulus.data(), d, d, d, k, 1,
                                              d, s_blocks);
            bm_gemm_kernel<<<grid, threads>>>(C10.data(), B0.data(), A1.data(),
                                              t.modulus.data(), d, d, d, k, 1,
                                              d, s_blocks);
            bm_gemm_kernel<<<grid, threads>>>(C11.data(), B1.data(), A1.data(),
                                              t.modulus.data(), d, d, d, k, 1,
                                              d, s_blocks);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Steps 3 and 4: fold each transposed half back into ciphertexts and
        // convert it from a row-wise to a column-wise matrix encryption.
        //
        // The transpose has to happen on the raw tensors above and not on the
        // ciphertexts a CMT returns. It permutes coefficients between
        // ciphertexts, which is exact relabelling on these tensors but invalid
        // on a real ciphertext, where c1 * s mixes coefficients across the very
        // runs being moved, so P(c0 + c1 * s) != P(c0) + P(c1) * s. The two
        // orders agree only when c1 = 0, which is why a test built entirely
        // from trivial encryptions cannot tell them apart.
        auto fold = [&](DeviceVector<Data64>& c0src, DeviceVector<Data64>& c1src,
                        std::vector<Ciphertext<Scheme::CKKS>>& dst)
        {
            BmRange _r_fold("CCMM.step34_fold");
            {
            BmRange _r("CCMM.step34_subring_to_ntt");
            dst.clear();
            dst.reserve(d);
            // Inherits level, scale and shape, but not coefficients: the two
            // subring_to_ntt launches below overwrite every element, so
            // copy-constructing from a[0] would move 2*comp_stride words per
            // column for nothing.
            for (int j = 0; j < d; ++j)
                dst.push_back(allocate_like(*a[0], 2 * comp_stride));

            std::vector<Data64*> base(d);
            for (int j = 0; j < d; ++j)
                base[j] = dst[j].data();
            DeviceVector<Data64*> p0(
                component_pointers(base, d, false, num_limbs));
            DeviceVector<Data64*> p1(
                component_pointers(base, d, true, num_limbs));

            const dim3 grid(static_cast<unsigned>(k),
                            static_cast<unsigned>(d),
                            static_cast<unsigned>(num_limbs));
            const size_t shared = d * sizeof(Data64);
            bm_subring_to_ntt_kernel<<<grid, dthreads, shared>>>(
                p0.data(), c0src.data(), t.wfwd.data(), t.psi_fwd_n.data(),
                t.modulus.data(), d, d, k);
            bm_subring_to_ntt_kernel<<<grid, dthreads, shared>>>(
                p1.data(), c1src.data(), t.wfwd.data(), t.psi_fwd_n.data(),
                t.modulus.data(), d, d, k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
            }

            cmt(dst, galois_key, ops);
        };

        std::vector<Ciphertext<Scheme::CKKS>> D01, D23;
        fold(C00, C01, D01);
        fold(C10, C11, D23);

        BmRange _r_comb("CCMM.step56_combine_relin");

        // Steps 5 and 6: the product is now
        //   D0 + Toep(sk) * (D1 + D2) + Toep(sk^2) * D3,
        // the degree-two ciphertext (D0, D1 + D2, D3). One relinearisation per
        // column absorbs the sk^2 term and returns it to two components.
        out.clear();
        out.reserve(d);
        for (int j = 0; j < d; ++j)
        {
            // All three components are written below, so the ciphertext is
            // allocated rather than copy-constructed from a[0].
            Ciphertext<Scheme::CKKS> c = allocate_like(*a[0], 3 * comp_stride);
            Data64* mem = c.data();

            cudaMemcpyAsync(mem, D01[j].data(), comp_stride * sizeof(Data64),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(mem + 2 * comp_stride,
                            D23[j].data() + comp_stride,
                            comp_stride * sizeof(Data64),
                            cudaMemcpyDeviceToDevice);
            addition<<<dim3(static_cast<unsigned>(n_ >> 8),
                            static_cast<unsigned>(num_limbs), 1u),
                       256>>>(D01[j].data() + comp_stride, D23[j].data(),
                              mem + comp_stride, context_->modulus_->data(),
                              context_->n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

            c.cipher_size_ = 3;
            c.rescale_required_ = false;
            c.relinearization_required_ = true;

            ops.relinearize_inplace(c, relin_key);

            c.cipher_size_ = 2;
            out.push_back(std::move(c));
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        // Step 7: the product carries both scaling factors; mark it so the
        // caller's rescale removes exactly one of them.
        for (int j = 0; j < d; ++j)
        {
            out[j].scale_ = a[0]->scale_ * b[0]->scale_;
            out[j].rescale_required_ = rescale;
        }
    }

} // namespace heongpu
