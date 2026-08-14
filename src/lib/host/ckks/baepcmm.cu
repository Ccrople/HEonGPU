// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/baepcmm.cuh>
#include <heongpu/kernel/baepcmm.cuh>
#include <heongpu/util/util.cuh>

#include <gpuntt/ntt_merge/ntt.cuh>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace heongpu
{
    namespace
    {
        struct BaeRange
        {
            explicit BaeRange(const char* name) { nvtxRangePushA(name); }
            ~BaeRange() { nvtxRangePop(); }
            BaeRange(const BaeRange&) = delete;
            BaeRange& operator=(const BaeRange&) = delete;
        };

        inline bool is_power_of_two(int v) noexcept
        {
            return v > 0 && (v & (v - 1)) == 0;
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

        Data64 powmod(Data64 b, Data64 e, Data64 m)
        {
            Data64 r = 1;
            b %= m;
            while (e)
            {
                if (e & 1)
                    r = mulmod(r, b, m);
                b = mulmod(b, b, m);
                e >>= 1;
            }
            return r;
        }

        inline Data64 invmod(Data64 a, Data64 p) { return powmod(a, p - 2, p); }

        // Same minimal big integer the Kang path uses for CRT reconstruction;
        // a product of three or more RNS primes outgrows __int128, so the
        // mixed radix accumulation cannot run in a builtin type.
        using BigUInt = std::vector<Data64>;

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

        gpuntt::ntt_rns_configuration<Data64> forward_cfg(int n_power)
        {
            // Plain assignment rather than designated initializers: nvcc has
            // been observed to drop initializers after a skipped member in
            // this struct, leaving mod_inverse and stream holding garbage.
            gpuntt::ntt_rns_configuration<Data64> cfg{};
            cfg.n_power = n_power;
            cfg.ntt_type = gpuntt::FORWARD;
            cfg.ntt_layout = gpuntt::PerPolynomial;
            cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            cfg.zero_padding = false;
            cfg.mod_inverse = nullptr;
            cfg.stream = cudaStreamDefault;
            return cfg;
        }

        gpuntt::ntt_rns_configuration<Data64> inverse_cfg(int n_power,
                                                          Data64* n_inverse)
        {
            gpuntt::ntt_rns_configuration<Data64> cfg{};
            cfg.n_power = n_power;
            cfg.ntt_type = gpuntt::INVERSE;
            cfg.ntt_layout = gpuntt::PerPolynomial;
            cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            cfg.zero_padding = false;
            cfg.mod_inverse = n_inverse;
            cfg.stream = cudaStreamDefault;
            return cfg;
        }
    } // namespace

    // -----------------------------------------------------------------------
    // Layout
    // -----------------------------------------------------------------------

    BaeLayout::BaeLayout(int N_, int cols_) : N(N_), cols(cols_)
    {
        if (!is_power_of_two(N))
            throw std::invalid_argument(
                "ring degree must be a positive power of two");
        if (!is_power_of_two(cols) || cols > N)
            throw std::invalid_argument(
                "column count must be a power of two dividing the ring "
                "degree");
        // Table 1's caption and the text of Sections 4.2 and 4.3: "which
        // requires that d >= N^{1/2}". Below that the input is fewer than one
        // ciphertext and the algorithm does not apply.
        if (static_cast<long long>(cols) * cols < static_cast<long long>(N))
            throw std::invalid_argument(
                "Bae PCMM requires cols >= sqrt(N); below that one ciphertext "
                "does not hold a whole strip of rows");
        k = N / cols;
    }

    // -----------------------------------------------------------------------
    // Operator
    // -----------------------------------------------------------------------

    HEBaePcmmOperator<Scheme::CKKS>::HEBaePcmmOperator(
        HEContext<Scheme::CKKS>& context, int cols)
        : context_(context)
    {
        n_ = context_->get_poly_modulus_degree();
        n_power_ = context_->n_power;
        q_size_ = context_->get_ciphertext_modulus_count();
        layout_ = BaeLayout(n_, cols);
    }

    std::pair<uint64_t, uint64_t>
    HEBaePcmmOperator<Scheme::CKKS>::gemm_macs(int d1, int d2, int d3, int N)
    {
        const uint64_t base =
            static_cast<uint64_t>(d1) * static_cast<uint64_t>(d2);
        return {base * static_cast<uint64_t>(N),
                base * static_cast<uint64_t>(d3)};
    }

    std::vector<std::vector<int64_t>>
    HEBaePcmmOperator<Scheme::CKKS>::encode_matrix(
        const std::vector<double>& matrix, int rows, double scale) const
    {
        const int cols = layout_.cols;
        const int k = layout_.k;
        if (rows % k != 0)
            throw std::invalid_argument(
                "row count must be a multiple of k = N / cols; one ciphertext "
                "carries exactly k rows");
        if (matrix.size() !=
            static_cast<size_t>(rows) * static_cast<size_t>(cols))
            throw std::invalid_argument("matrix must be rows x cols, row "
                                        "major");

        const int count = rows / k;
        std::vector<std::vector<int64_t>> out(
            count, std::vector<int64_t>(n_, 0));

        for (int r = 0; r < rows; ++r)
        {
            const int i = r / k;
            const int u = r - i * k;
            for (int t = 0; t < cols; ++t)
            {
                const double v =
                    matrix[static_cast<size_t>(r) * cols + t] * scale;
                out[i][static_cast<size_t>(t) * k + u] = std::llround(v);
            }
        }
        return out;
    }

    std::vector<double> HEBaePcmmOperator<Scheme::CKKS>::decode_matrix(
        const std::vector<std::vector<int64_t>>& coeffs, int rows,
        double scale) const
    {
        const int cols = layout_.cols;
        const int k = layout_.k;
        if (rows % k != 0)
            throw std::invalid_argument("row count must be a multiple of k");
        if (static_cast<int>(coeffs.size()) != rows / k)
            throw std::invalid_argument(
                "expected rows / k coefficient vectors");

        std::vector<double> out(static_cast<size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r)
        {
            const int i = r / k;
            const int u = r - i * k;
            for (int t = 0; t < cols; ++t)
            {
                out[static_cast<size_t>(r) * cols + t] =
                    static_cast<double>(
                        coeffs[i][static_cast<size_t>(t) * k + u]) /
                    scale;
            }
        }
        return out;
    }

    void HEBaePcmmOperator<Scheme::CKKS>::load_coefficients(
        Plaintext<Scheme::CKKS>& plain, const std::vector<int64_t>& coeffs,
        double scale) const
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
        gpuntt::GPU_NTT_Inplace(mem.data(), context_->ntt_table_->data(),
                                context_->modulus_->data(),
                                forward_cfg(n_power_), num_limbs, num_limbs);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // memory_set resets the descriptive fields, so everything else is
        // stamped after it rather than before.
        plain.memory_set(std::move(mem));
        plain.scale_ = scale;
        plain.in_ntt_domain_ = true;
        plain.depth_ = depth;
        plain.plain_size_ = static_cast<int>(host.size());
    }

    std::vector<int64_t> HEBaePcmmOperator<Scheme::CKKS>::extract_coefficients(
        Plaintext<Scheme::CKKS>& plain) const
    {
        const int depth = plain.depth_;
        const int num_limbs = q_size_ - depth;
        if (num_limbs < 1)
            throw std::invalid_argument(
                "extract_coefficients needs at least one active limb");

        if (!plain.is_on_device())
            plain.store_in_device();

        const size_t elems = static_cast<size_t>(num_limbs) * n_;
        DeviceVector<Data64> mem(elems);
        cudaMemcpy(mem.data(), plain.data(), elems * sizeof(Data64),
                   cudaMemcpyDefault);

        if (plain.in_ntt_domain_)
        {
            gpuntt::GPU_INTT_Inplace(
                mem.data(), context_->intt_table_->data(),
                context_->modulus_->data(),
                inverse_cfg(n_power_, context_->n_inverse_->data()), num_limbs,
                num_limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        std::vector<Data64> host(elems);
        cudaMemcpy(host.data(), mem.data(), elems * sizeof(Data64),
                   cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<Modulus64> all = context_->get_key_modulus();
        std::vector<int64_t> coeffs(n_);

        if (num_limbs == 1)
        {
            const Data64 p = all[0].value;
            const Data64 half = p >> 1;
            for (int i = 0; i < n_; ++i)
                coeffs[i] = (host[i] > half)
                                ? -static_cast<int64_t>(p - host[i])
                                : static_cast<int64_t>(host[i]);
            return coeffs;
        }

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
                const Data64 r = host[static_cast<size_t>(l) * n_ + i] % p[l];
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
        return coeffs;
    }

    void HEBaePcmmOperator<Scheme::CKKS>::upload_plaintext(
        const std::vector<double>& weight, int d1, int d2, int depth,
        double scale)
    {
        if (d1 < 1 || d2 < 1)
            throw std::invalid_argument("plaintext matrix must be non-empty");
        if (weight.size() !=
            static_cast<size_t>(d1) * static_cast<size_t>(d2))
            throw std::invalid_argument(
                "plaintext matrix must be d1 x d2, row major");

        const int num_limbs = q_size_ - depth;
        if (num_limbs < 1)
            throw std::invalid_argument("no active limbs at that depth");

        BaeRange _r("BaePCMM.upload_plaintext");

        std::vector<Modulus64> all = context_->get_key_modulus();
        const size_t entries = static_cast<size_t>(d1) * d2;

        // U is rounded ONCE and then reduced per limb. Rounding per limb
        // would be the same numbers here, but keeping one integer matrix is
        // what makes the "U is never encoded" claim in the header literally
        // true: there is no polynomial anywhere in this function.
        std::vector<int64_t> centred(entries);
        for (size_t i = 0; i < entries; ++i)
            centred[i] = std::llround(weight[i] * scale);

        std::vector<Data64> host(static_cast<size_t>(num_limbs) * entries);
        for (int l = 0; l < num_limbs; ++l)
        {
            const Data64 p = all[l].value;
            for (size_t i = 0; i < entries; ++i)
                host[static_cast<size_t>(l) * entries + i] =
                    centered_to_modular(centred[i], p);
        }

        plain_ = DeviceVector<Data64>(host);
        d1_ = d1;
        d2_ = d2;
        plain_depth_ = depth;
        plain_scale_ = scale;
    }

    void HEBaePcmmOperator<Scheme::CKKS>::run_gemms(
        const std::vector<Ciphertext<Scheme::CKKS>*>& in,
        DeviceVector<Data64>& a_out, DeviceVector<Data64>& b_out, int& limbs,
        int& depth, double& scale)
    {
        if (plain_depth_ < 0)
            throw std::logic_error(
                "no plaintext matrix uploaded; call upload_plaintext");
        if (in.empty())
            throw std::invalid_argument("no input ciphertexts");

        const int cols = layout_.cols;
        const int k = layout_.k;
        const int rows_in = static_cast<int>(in.size()) * k;
        if (rows_in != d2_)
            throw std::invalid_argument(
                "input carries a different number of matrix rows than the "
                "plaintext's column count: expected d2 / k ciphertexts");

        depth = in[0]->depth_;
        scale = in[0]->scale_;
        if (depth != plain_depth_)
            throw std::invalid_argument(
                "plaintext matrix level does not match the ciphertexts");
        for (const auto* c : in)
        {
            if (c->depth_ != depth)
                throw std::invalid_argument(
                    "all input ciphertexts must share a level");
            if (c->scale_ != scale)
                throw std::invalid_argument(
                    "all input ciphertexts must share a scale");
            if (c->rescale_required_ || c->relinearization_required_)
                throw std::invalid_argument(
                    "spend any pending rescale or relinearisation before the "
                    "Bae product: it reads raw coefficients and a pending "
                    "flag means the buffer is not what its scale says");
            if (transform_free_ && c->in_ntt_domain_ != in[0]->in_ntt_domain_)
                throw std::invalid_argument(
                    "the transform-free path combines the limbs as they are "
                    "stored, so every input must be in the same domain: "
                    "mixing an NTT-domain ciphertext with a coefficient-domain "
                    "one sums two different polynomials and reports no error");
            // A size-3 ciphertext has a second a-part and Lemma 3 does not
            // cover it -- but cipher_size_ CANNOT be used to detect one.
            // relinearize_inplace clears relinearization_required_ and never
            // writes cipher_size_ back to 2, so every ciphertext that has been
            // through a product reports three for the rest of its life. The
            // library treats the flag as the authority and derives the size
            // from it (operator.cuh), and the flag is checked above, so there
            // is nothing left to check here. Reading the field instead
            // rejected every activation that had met a multiplication, which
            // is every activation a real block hands to a projection.
        }

        limbs = q_size_ - depth;
        const int count = static_cast<int>(in.size());

        BaeRange _r("BaePCMM.product");

        // ---- Step 1: to the coefficient domain, then ModDecomp ------------
        // ModDecomp itself is free; what is not free is that the ciphertexts
        // arrive in the NTT domain and the identity of Lemma 3 is a statement
        // about COEFFICIENTS. This INTT is the whole of the "pre-processing"
        // the paper charges the algorithm for.
        const size_t ct_words = static_cast<size_t>(2) * limbs * n_;
        DeviceVector<Data64> coeff(ct_words * count);
        std::vector<Data64*> ptrs(count);
        for (int i = 0; i < count; ++i)
        {
            Data64* dst = coeff.data() + static_cast<size_t>(i) * ct_words;
            cudaMemcpyAsync(dst, in[i]->data(), ct_words * sizeof(Data64),
                            cudaMemcpyDeviceToDevice, cudaStreamDefault);
            ptrs[i] = dst;
        }
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (!transform_free_)
        {
            BaeRange _rr("BaePCMM.intt");
            gpuntt::GPU_INTT_Inplace(
                coeff.data(), context_->intt_table_->data(),
                context_->modulus_->data(),
                inverse_cfg(n_power_, context_->n_inverse_->data()),
                2 * limbs * count, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        DeviceVector<Data64*> dptrs(ptrs);

        // ---- Step 2: assemble A (d2 x N) and B (d2 x cols) ----------------
        DeviceVector<Data64> A(static_cast<size_t>(limbs) * d2_ * n_);
        DeviceVector<Data64> B(static_cast<size_t>(limbs) * d2_ * cols);
        {
            BaeRange _rr("BaePCMM.moddecomp");
            bae_gather_a_kernel<<<dim3((n_ + 255) / 256, d2_, limbs), 256>>>(
                A.data(), dptrs.data(), n_, cols, limbs,
                context_->modulus_->data());
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            bae_gather_b_kernel<<<dim3((cols + 255) / 256, d2_, limbs), 256>>>(
                B.data(), dptrs.data(), n_, cols, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // ---- Step 3: the two GEMMs ---------------------------------------
        a_out = DeviceVector<Data64>(static_cast<size_t>(limbs) * d1_ * n_);
        b_out = DeviceVector<Data64>(static_cast<size_t>(limbs) * d1_ * cols);

        const int tile = 16;
        {
            BaeRange _rr("BaePCMM.gemm_a");
            bae_gemm_kernel<<<dim3((n_ + tile - 1) / tile,
                                   (d1_ + tile - 1) / tile, limbs),
                              dim3(tile, tile)>>>(
                a_out.data(), plain_.data(), A.data(),
                context_->modulus_->data(), d1_, d2_, n_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        {
            BaeRange _rr("BaePCMM.gemm_b");
            bae_gemm_kernel<<<dim3((cols + tile - 1) / tile,
                                   (d1_ + tile - 1) / tile, limbs),
                              dim3(tile, tile)>>>(
                b_out.data(), plain_.data(), B.data(),
                context_->modulus_->data(), d1_, d2_, cols);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void HEBaePcmmOperator<Scheme::CKKS>::pcmm(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& in, bool rescale)
    {
        if (layout_.k != 1)
            throw std::invalid_argument(
                "pcmm() returns RLWE ciphertexts and is therefore k == 1 "
                "only: at k > 1 an output row is a rank-k MLWE ciphertext and "
                "there is no degree-N ciphertext to return until ModPack has "
                "run. Use pcmm_mlwe(), or widen the encrypted matrix to "
                "cols == N");

        DeviceVector<Data64> A, B;
        int limbs = 0, depth = 0;
        double scale = 0.0;
        run_gemms(in, A, B, limbs, depth, scale);

        out.clear();
        out.reserve(d1_);
        std::vector<Data64*> ptrs(d1_);
        for (int r = 0; r < d1_; ++r)
        {
            out.emplace_back(*in[0]); // inherits level, scale and shape
            ptrs[r] = out.back().data();
        }
        DeviceVector<Data64*> dptrs(ptrs);

        {
            BaeRange _rr("BaePCMM.emit");
            bae_emit_rlwe_kernel<<<dim3((n_ + 255) / 256, d1_, limbs), 256>>>(
                dptrs.data(), A.data(), B.data(), d1_, n_, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // Back to the NTT domain: everything downstream of this operator in
        // the library assumes it, and the identity is finished with. The
        // transform-free path never left it, so it has nothing to undo.
        if (!transform_free_)
        {
            BaeRange _rr("BaePCMM.ntt");
            for (int r = 0; r < d1_; ++r)
            {
                gpuntt::GPU_NTT_Inplace(
                    out[r].data(), context_->ntt_table_->data(),
                    context_->modulus_->data(), forward_cfg(n_power_),
                    2 * limbs, limbs);
            }
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        // Algorithm 1 step 8 / Algorithm 2 step 12: the result carries the
        // plaintext scaling factor and the caller's rescale removes exactly
        // that. Only MARKED here, as the Kang path also does, so that a
        // caller can inspect the unscaled product.
        const bool ntt_domain =
            transform_free_ ? in[0]->in_ntt_domain_ : true;
        for (auto& c : out)
        {
            c.scale_ = scale * plain_scale_;
            c.rescale_required_ = rescale;
            c.in_ntt_domain_ = ntt_domain;
            // Stamped, not inherited. The emit kernel wrote exactly two
            // components, so the result IS size two whatever the input's
            // stale field said; copying that field forward would propagate
            // the library's lie about relinearised ciphertexts.
            c.cipher_size_ = 2;
            c.relinearization_required_ = false;
        }
    }

    void HEBaePcmmOperator<Scheme::CKKS>::set_transform_free(bool on)
    {
        if (on && layout_.k != 1)
            throw std::invalid_argument(
                "the transform-free path is k == 1 only: above that the "
                "product mixes decimation phases inside a ciphertext, which is "
                "a statement about COEFFICIENTS, and running it on NTT-domain "
                "limbs computes a different matrix without saying so");
        transform_free_ = on;
    }

    BaeMlweStack HEBaePcmmOperator<Scheme::CKKS>::pcmm_mlwe(
        const std::vector<Ciphertext<Scheme::CKKS>*>& in)
    {
        if (transform_free_)
            throw std::invalid_argument(
                "an MLWE stack is a COEFFICIENT-domain object by definition, "
                "and the transform-free path never leaves the NTT domain: it "
                "would hand back evaluation-point values labelled as "
                "coefficients. Turn it off for this entry point");

        DeviceVector<Data64> A, B;
        int limbs = 0, depth = 0;
        double scale = 0.0;
        run_gemms(in, A, B, limbs, depth, scale);

        const int cols = layout_.cols;
        const int k = layout_.k;

        DeviceVector<Data64> a_dev(static_cast<size_t>(d1_) * k * limbs * cols);
        DeviceVector<Data64> b_dev(static_cast<size_t>(d1_) * limbs * cols);

        {
            BaeRange _rr("BaePCMM.scatter");
            bae_scatter_a_kernel<<<dim3((cols + 255) / 256, d1_ * k, limbs),
                                   256>>>(a_dev.data(), A.data(), d1_, cols, k,
                                          limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            bae_scatter_b_kernel<<<dim3((cols + 255) / 256, d1_, limbs), 256>>>(
                b_dev.data(), B.data(), d1_, cols, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

        BaeMlweStack stack;
        stack.rows = d1_;
        stack.cols = cols;
        stack.k = k;
        stack.limbs = limbs;
        stack.depth = depth;
        stack.scale = scale * plain_scale_;
        stack.a.resize(a_dev.size());
        stack.b.resize(b_dev.size());
        cudaMemcpy(stack.a.data(), a_dev.data(),
                   a_dev.size() * sizeof(Data64), cudaMemcpyDeviceToHost);
        cudaMemcpy(stack.b.data(), b_dev.data(),
                   b_dev.size() * sizeof(Data64), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
        return stack;
    }


    void HEBaePcmmOperator<Scheme::CKKS>::generate_modpack_keys(
        HEKeyGenerator<Scheme::CKKS>& keygen, Secretkey<Scheme::CKKS>& sk,
        const std::vector<int>& sk_coefficients)
    {
        const int k = layout_.k;
        const int cols = layout_.cols;

        if (static_cast<int>(sk_coefficients.size()) != n_)
            throw std::invalid_argument(
                "the secret coefficient vector must have length N");
        for (int v : sk_coefficients)
            if (v < -1 || v > 1)
                throw std::invalid_argument("secret coefficients must be "
                                            "ternary");

        modpack_keys_.clear();
        if (k == 1)
        {
            // ModPack is the identity here and no key is needed. Silently
            // generating one anyway would make modpack_keys_generated() lie
            // about what the k = 1 path depends on.
            return;
        }

        modpack_keys_.reserve(k);
        for (int j = 0; j < k; ++j)
        {
            // s_j is the X^j component of sk, embedded back into R_N as
            // s_j(X^k): coefficient t of s_j sits at X^{t*k}. Both the
            // extraction and the embedding are the decimation of App. A, and
            // the composition is just "keep the coefficients congruent to j
            // mod k, and slide them onto multiples of k".
            std::vector<int> embedded(n_, 0);
            for (int t = 0; t < cols; ++t)
                embedded[static_cast<size_t>(t) * k] =
                    sk_coefficients[static_cast<size_t>(t) * k + j];

            Secretkey<Scheme::CKKS> sub(embedded, context_);
            auto swk = std::make_unique<Switchkey<Scheme::CKKS>>(context_);
            // (swk, new_sk, old_sk): transports FROM s_j TO sk, which is the
            // direction that turns a ciphertext (Atilde_j, 0) under s_j into
            // an encryption of Atilde_j * s_j under sk. Backwards produces
            // noise, not an error.
            keygen.generate_switch_key(*swk, sk, sub);
            modpack_keys_.push_back(std::move(swk));
        }
    }

    void HEBaePcmmOperator<Scheme::CKKS>::pcmm_packed(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& in,
        HEArithmeticOperator<Scheme::CKKS>& ops, bool rescale)
    {
        const int k = layout_.k;
        const int cols = layout_.cols;

        if (k == 1)
        {
            pcmm(out, in, rescale);
            return;
        }
        if (!modpack_keys_generated())
            throw std::logic_error(
                "ModPack needs the k sub-secret switching keys; call "
                "generate_modpack_keys first");

        DeviceVector<Data64> A, B;
        int limbs = 0, depth = 0;
        double scale = 0.0;
        run_gemms(in, A, B, limbs, depth, scale);

        if (d1_ % k != 0)
            throw std::invalid_argument(
                "the output row count must be a multiple of k, because one "
                "packed ciphertext carries exactly k rows");
        const int groups = d1_ / k;

        BaeRange _r("BaePCMM.modpack");

        out.clear();
        out.reserve(groups);

        const size_t poly = static_cast<size_t>(limbs) * n_;
        DeviceVector<Data64> a_stage(static_cast<size_t>(k) * poly);
        DeviceVector<Data64> b_stage(poly);

        for (int g = 0; g < groups; ++g)
        {
            // Interleave this group's k MLWE rows back up to degree N.
            bae_modpack_assemble_kernel<<<dim3((n_ + 255) / 256, limbs, k + 1),
                                          256>>>(
                a_stage.data(), b_stage.data(), A.data(), B.data(), d1_, cols,
                k, limbs, g * k);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Everything the key switch touches has to be in the NTT domain.
            gpuntt::GPU_NTT_Inplace(a_stage.data(), context_->ntt_table_->data(),
                                    context_->modulus_->data(),
                                    forward_cfg(n_power_), k * limbs, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            gpuntt::GPU_NTT_Inplace(b_stage.data(), context_->ntt_table_->data(),
                                    context_->modulus_->data(),
                                    forward_cfg(n_power_), limbs, limbs);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // mu = Btilde + sum_j Atilde_j * s_j. Each term is one key switch
            // of (Atilde_j, 0) against the key carrying s_j: a key switch
            // preserves a*old + b, so it hands back (a', b') with
            // a'*sk + b' = Atilde_j * s_j exactly.
            Ciphertext<Scheme::CKKS> acc(*in[0]);
            bool have_acc = false;

            for (int j = 0; j < k; ++j)
            {
                Ciphertext<Scheme::CKKS> term(*in[0]);
                cudaMemsetAsync(term.data(), 0, 2 * poly * sizeof(Data64),
                                cudaStreamDefault);
                // Atilde_j is the part that gets multiplied by the secret,
                // so it goes in component 1; component 0 stays zero.
                cudaMemcpyAsync(term.data() + poly,
                                a_stage.data() + static_cast<size_t>(j) * poly,
                                poly * sizeof(Data64), cudaMemcpyDeviceToDevice,
                                cudaStreamDefault);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                term.rescale_required_ = false;
                term.relinearization_required_ = false;
                term.in_ntt_domain_ = true;

                Ciphertext<Scheme::CKKS> switched(*in[0]);
                ops.keyswitch(term, switched, *modpack_keys_[j]);

                if (!have_acc)
                {
                    acc = std::move(switched);
                    have_acc = true;
                }
                else
                {
                    ops.add(acc, switched, acc);
                }
            }

            // ... and the b polynomial, which rides for free in the b-part.
            Ciphertext<Scheme::CKKS> btilde(*in[0]);
            cudaMemsetAsync(btilde.data(), 0, 2 * poly * sizeof(Data64),
                            cudaStreamDefault);
            // Btilde is added straight in, so it is component 0.
            cudaMemcpyAsync(btilde.data(), b_stage.data(),
                            poly * sizeof(Data64), cudaMemcpyDeviceToDevice,
                            cudaStreamDefault);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            btilde.rescale_required_ = false;
            btilde.relinearization_required_ = false;
            btilde.in_ntt_domain_ = true;

            ops.add(acc, btilde, acc);

            acc.scale_ = scale * plain_scale_;
            acc.rescale_required_ = rescale;
            acc.in_ntt_domain_ = true;
            // As in pcmm(): stamped rather than inherited, because
            // cipher_size_ is stale on anything that has met a product.
            acc.cipher_size_ = 2;
            acc.relinearization_required_ = false;
            out.push_back(std::move(acc));
        }
        HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    }


    double HEBaePcmmOperator<Scheme::CKKS>::rescale_prime(int depth) const
    {
        const int active = q_size_ - depth;
        if (active < 1)
            throw std::invalid_argument("no active limbs at that depth");
        return static_cast<double>(
            context_->get_key_modulus()[active - 1].value);
    }

    void HEBaePcmmOperator<Scheme::CKKS>::project(
        std::vector<Ciphertext<Scheme::CKKS>>& out,
        const std::vector<Ciphertext<Scheme::CKKS>*>& x,
        const std::vector<double>& weight, int in_channels, int out_channels,
        HEArithmeticOperator<Scheme::CKKS>& ops)
    {
        const int k = layout_.k;
        if (static_cast<int>(x.size()) * k != in_channels)
            throw std::invalid_argument(
                "the activation must carry in_channels / k ciphertexts");
        if (weight.size() != static_cast<size_t>(in_channels) *
                                 static_cast<size_t>(out_channels))
            throw std::invalid_argument(
                "the weight must be in_channels by out_channels, row major, "
                "which is the transpose of the mathematical weight");
        if (out_channels % k != 0)
            throw std::invalid_argument(
                "out_channels must be a multiple of k");

        BaeRange _r("BaePCMM.project");

        // U = weight^T, out_channels x in_channels. A transpose on the host
        // is free, and it is the only preparation this plaintext ever needs:
        // no diagonals, no encoding, no NTT, no cache.
        std::vector<double> U(static_cast<size_t>(out_channels) * in_channels);
        for (int i = 0; i < in_channels; ++i)
            for (int j = 0; j < out_channels; ++j)
                U[static_cast<size_t>(j) * in_channels + i] =
                    weight[static_cast<size_t>(i) * out_channels + j];

        const int depth = x.front()->depth_;
        upload_plaintext(U, out_channels, in_channels, depth,
                         rescale_prime(depth));

        pcmm_packed(out, x, ops, /*rescale=*/true);
        for (auto& c : out)
            ops.rescale_inplace(c);
    }

} // namespace heongpu
