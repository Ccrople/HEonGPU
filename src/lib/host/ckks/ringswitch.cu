// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/ringswitch.cuh>
#include <heongpu/kernel/ringswitch.cuh>
#include <heongpu/util/util.cuh>

#include <gpuntt/ntt_merge/ntt.cuh>

#include <random>
#include <stdexcept>

namespace heongpu
{
    namespace
    {
        /// Forward-NTT configuration for a given ring. Plain assignment
        /// rather than designated initializers: nvcc has been observed to
        /// silently drop initializers in this struct, leaving
        /// mod_inverse/stream holding garbage.
        gpuntt::ntt_rns_configuration<Data64> forward_cfg(int n_power,
                                                          cudaStream_t stream)
        {
            gpuntt::ntt_rns_configuration<Data64> cfg{};
            cfg.n_power = n_power;
            cfg.ntt_type = gpuntt::FORWARD;
            cfg.ntt_layout = gpuntt::PerPolynomial;
            cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            cfg.zero_padding = false;
            cfg.mod_inverse = nullptr;
            cfg.stream = stream;
            return cfg;
        }

        gpuntt::ntt_rns_configuration<Data64>
        inverse_cfg(int n_power, Data64* n_inverse, cudaStream_t stream)
        {
            gpuntt::ntt_rns_configuration<Data64> cfg{};
            cfg.n_power = n_power;
            cfg.ntt_type = gpuntt::INVERSE;
            cfg.ntt_layout = gpuntt::PerPolynomial;
            cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            cfg.zero_padding = false;
            cfg.mod_inverse = n_inverse;
            cfg.stream = stream;
            return cfg;
        }
    } // namespace

    HERingSwitchOperator<Scheme::CKKS>::HERingSwitchOperator(
        HEContext<Scheme::CKKS>& big_context,
        HEContext<Scheme::CKKS>& small_context)
        : big_(big_context), small_(small_context)
    {
        if (!big_ || !big_->context_generated_)
            throw std::invalid_argument("big HEContext is not generated");
        if (!small_ || !small_->context_generated_)
            throw std::invalid_argument("small HEContext is not generated");
        if (big_->scheme_ != scheme_type::ckks ||
            small_->scheme_ != scheme_type::ckks)
            throw std::invalid_argument("ring switching is CKKS only");

        n_big_ = big_->n;
        n_small_ = small_->n;
        n_power_big_ = big_->n_power;
        n_power_small_ = small_->n_power;

        if (n_big_ <= n_small_ || (n_big_ % n_small_) != 0)
            throw std::invalid_argument(
                "big ring degree must be a proper multiple of the small one");

        k_ = n_big_ / n_small_;
        k_power_ = n_power_big_ - n_power_small_;

        big_q_size_ = big_->Q_size;
        small_q_size_ = small_->Q_size;

        if (small_q_size_ > big_q_size_)
            throw std::invalid_argument(
                "small context carries more Q primes than the big one");

        // The switch is modulus-preserving: the small chain must be a
        // value-identical bottom prefix of the big chain. Rescale drops the
        // LAST active prime, so the reachable big-ring levels are exactly the
        // prefixes — sharing anything but the prefix would correspond to no
        // level at all.
        for (int i = 0; i < small_q_size_; i++)
        {
            if (big_->prime_vector_[i].value != small_->prime_vector_[i].value)
                throw std::invalid_argument(
                    "small context Q chain is not a prefix of the big chain; "
                    "build it with set_coeff_modulus_values on the big "
                    "chain's leading values");
        }
    }

    HERingSwitchOperator<Scheme::CKKS>::SecretPair
    HERingSwitchOperator<Scheme::CKKS>::make_secret_pair(
        int hamming_weight, unsigned long long seed) const
    {
        if (hamming_weight <= 0 || hamming_weight > n_small_)
            throw std::invalid_argument(
                "hamming weight must be in (0, n_small]");

        std::mt19937_64 rng(seed);
        std::vector<int> positions(n_small_);
        for (int i = 0; i < n_small_; i++)
            positions[i] = i;
        // Partial Fisher–Yates: the first hamming_weight entries end up a
        // uniform sample without replacement.
        for (int i = 0; i < hamming_weight; i++)
        {
            std::uniform_int_distribution<int> pick(i, n_small_ - 1);
            std::swap(positions[i], positions[pick(rng)]);
        }

        std::vector<int> coefficients(n_small_, 0);
        std::uniform_int_distribution<int> sign(0, 1);
        for (int i = 0; i < hamming_weight; i++)
            coefficients[positions[i]] = sign(rng) ? 1 : -1;

        return make_secret_pair(coefficients);
    }

    HERingSwitchOperator<Scheme::CKKS>::SecretPair
    HERingSwitchOperator<Scheme::CKKS>::make_secret_pair(
        const std::vector<int>& coefficients) const
    {
        if (static_cast<int>(coefficients.size()) != n_small_)
            throw std::invalid_argument(
                "secret coefficient vector must have length n_small");
        for (int v : coefficients)
            if (v < -1 || v > 1)
                throw std::invalid_argument(
                    "secret coefficients must be ternary");

        // s'(X^k): coefficient j of the small secret sits at X^{j·k}.
        std::vector<int> embedded(n_big_, 0);
        for (int j = 0; j < n_small_; j++)
            embedded[static_cast<size_t>(j) * k_] = coefficients[j];

        Secretkey<Scheme::CKKS> sk_small(coefficients, small_);
        Secretkey<Scheme::CKKS> sk_embedded(embedded, big_);

        SecretPair pair{coefficients, std::move(sk_small),
                        std::move(sk_embedded)};
        return pair;
    }

    void HERingSwitchOperator<Scheme::CKKS>::generate_keys(
        HEKeyGenerator<Scheme::CKKS>& keygen, Secretkey<Scheme::CKKS>& sk_big,
        Secretkey<Scheme::CKKS>& embedded)
    {
        // Fresh Switchkey objects: generate_switch_key refuses to overwrite a
        // generated key, so regeneration means new objects.
        swk_down_ = std::make_unique<Switchkey<Scheme::CKKS>>(big_);
        swk_up_ = std::make_unique<Switchkey<Scheme::CKKS>>(big_);

        // Argument order is (swk, new_sk, old_sk): the key transports
        // ciphertexts FROM old TO new. Getting this backwards produces noise,
        // not an error.
        keygen.generate_switch_key(*swk_down_, embedded, sk_big);
        keygen.generate_switch_key(*swk_up_, sk_big, embedded);
    }

    Ciphertext<Scheme::CKKS>
    HERingSwitchOperator<Scheme::CKKS>::metadata_at_level(
        const HEContext<Scheme::CKKS>& ctx, int limbs,
        const Ciphertext<Scheme::CKKS>& like) const
    {
        Ciphertext<Scheme::CKKS> c;
        c.scheme_ = like.scheme_;
        c.ring_size_ = ctx->n;
        c.coeff_modulus_count_ = ctx->Q_size;
        c.cipher_size_ = 2;
        c.depth_ = ctx->Q_size - limbs;
        c.in_ntt_domain_ = true;
        c.scale_ = like.scale_;
        c.encoding_ = like.encoding_;
        c.rescale_required_ = false;
        c.relinearization_required_ = false;
        c.ciphertext_generated_ = true;
        c.storage_type_ = storage_type::DEVICE;
        return c;
    }

    Ciphertext<Scheme::CKKS>
    HERingSwitchOperator<Scheme::CKKS>::allocate_at_level(
        const HEContext<Scheme::CKKS>& ctx, int limbs,
        const Ciphertext<Scheme::CKKS>& like, cudaStream_t stream) const
    {
        Ciphertext<Scheme::CKKS> c = metadata_at_level(ctx, limbs, like);
        c.memory_set(DeviceVector<Data64>(
            static_cast<size_t>(2) * limbs * ctx->n, stream));
        return c;
    }

    void HERingSwitchOperator<Scheme::CKKS>::validate_big_input(
        const Ciphertext<Scheme::CKKS>& input, int active_primes) const
    {
        if (active_primes < 1 || active_primes > small_q_size_)
            throw std::invalid_argument(
                "active prime count must lie within the shared chain prefix");
        if (input.ring_size_ != n_big_ ||
            input.coeff_modulus_count_ != big_q_size_)
            throw std::invalid_argument(
                "input is not a big-context ciphertext");
        if (!input.in_ntt_domain_)
            throw std::invalid_argument("input must be in NTT domain");
        if (input.rescale_required_ || input.relinearization_required_)
            throw std::invalid_argument(
                "spend the pending rescale/relinearisation before switching");
        if (input.storage_type_ != storage_type::DEVICE)
            throw std::invalid_argument("input must be device-resident");
        if (input.device_locations_.size() <
            static_cast<size_t>(2) * active_primes * n_big_)
            throw std::invalid_argument("invalid ciphertext size");
    }

    std::vector<Ciphertext<Scheme::CKKS>>
    HERingSwitchOperator<Scheme::CKKS>::switch_down(
        Ciphertext<Scheme::CKKS>& input,
        HEArithmeticOperator<Scheme::CKKS>& big_operators,
        const ExecutionOptions& options)
    {
        if (!keys_generated())
            throw std::logic_error("ring switch keys are not generated");

        const int l = big_q_size_ - input.depth_;
        // Validate BEFORE the key switch: keyswitch itself never checks the
        // NTT-domain flag and would happily key-switch a coefficient-domain
        // input into garbage first.
        validate_big_input(input, l);

        // Step 1: sk_big -> s'(X^k), the only step that touches the algebra.
        // The output ciphertext carries metadata only — the switchkey
        // pipeline installs its own right-sized buffer.
        Ciphertext<Scheme::CKKS> under_embedded =
            metadata_at_level(big_, l, input);
        big_operators.keyswitch(input, under_embedded, *swk_down_, options);

        return split_embedded(under_embedded, options);
    }

    std::vector<Ciphertext<Scheme::CKKS>>
    HERingSwitchOperator<Scheme::CKKS>::split_embedded(
        const Ciphertext<Scheme::CKKS>& input, const ExecutionOptions& options)
    {
        const int l = big_q_size_ - input.depth_;
        validate_big_input(input, l);

        const cudaStream_t stream = options.stream_;
        const size_t total = static_cast<size_t>(2) * l * n_big_;

        // Working copy in the coefficient domain; the caller's buffer is
        // never mutated.
        DeviceVector<Data64> coeff(total, stream);
        cudaMemcpyAsync(coeff.data(), input.device_locations_.data(),
                        total * sizeof(Data64), cudaMemcpyDeviceToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_INTT_Inplace(coeff.data(), big_->intt_table_->data(),
                                 big_->modulus_->data(),
                                 inverse_cfg(n_power_big_,
                                             big_->n_inverse_->data(), stream),
                                 2 * l, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // Stride-k split into the [j][c][limb][i] staging layout.
        DeviceVector<Data64> staged(total, stream);
        ringswitch_split_kernel<<<dim3((n_big_ >> 8), l, 2), 256, 0, stream>>>(
            coeff.data(), staged.data(), n_power_big_, k_power_, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // One batched forward NTT over all k·2·l small polynomials. The
        // staging poly order has the limb fastest, matching gpuntt's
        // (poly % mod_count) modulus selection.
        gpuntt::GPU_NTT_Inplace(staged.data(), small_->ntt_table_->data(),
                                small_->modulus_->data(),
                                forward_cfg(n_power_small_, stream), 2 * l * k_,
                                l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        std::vector<Ciphertext<Scheme::CKKS>> outs;
        outs.reserve(k_);
        std::vector<Data64*> ptrs(k_);
        for (int j = 0; j < k_; j++)
        {
            outs.push_back(allocate_at_level(small_, l, input, stream));
            ptrs[j] = outs[static_cast<size_t>(j)].data();
        }

        DeviceVector<Data64*> dptrs(ptrs, stream);
        ringswitch_distribute_kernel<<<dim3((n_small_ >> 8), l, 2 * k_), 256,
                                       0, stream>>>(
            staged.data(), dptrs.data(), n_power_small_, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // The staging buffers and the pointer array die at return; the work
        // on this stream has to have consumed them first.
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return outs;
    }

    Ciphertext<Scheme::CKKS> HERingSwitchOperator<Scheme::CKKS>::compose_up(
        std::vector<Ciphertext<Scheme::CKKS>>& inputs,
        HEArithmeticOperator<Scheme::CKKS>& big_operators,
        const ExecutionOptions& options)
    {
        if (!keys_generated())
            throw std::logic_error("ring switch keys are not generated");

        Ciphertext<Scheme::CKKS> under_embedded =
            interleave_embedded(inputs, options);

        Ciphertext<Scheme::CKKS> output = metadata_at_level(
            big_, big_q_size_ - under_embedded.depth_, under_embedded);
        big_operators.keyswitch(under_embedded, output, *swk_up_, options);
        // switch_down returns with its stream drained (its temporaries force
        // it); returning compose_up the same way keeps the pair symmetric and
        // keeps wall-clock timings of either direction honest.
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
        return output;
    }

    Ciphertext<Scheme::CKKS>
    HERingSwitchOperator<Scheme::CKKS>::interleave_embedded(
        const std::vector<Ciphertext<Scheme::CKKS>>& inputs,
        const ExecutionOptions& options)
    {
        if (static_cast<int>(inputs.size()) != k_)
            throw std::invalid_argument(
                "compose expects exactly k small-ring ciphertexts");

        const int depth = inputs[0].depth_;
        const double scale = inputs[0].scale_;
        for (const auto& ct : inputs)
        {
            if (ct.ring_size_ != n_small_ ||
                ct.coeff_modulus_count_ != small_q_size_)
                throw std::invalid_argument(
                    "compose input is not a small-context ciphertext");
            if (ct.depth_ != depth)
                throw std::invalid_argument(
                    "all inputs must sit at one level");
            if (ct.scale_ != scale)
                throw std::invalid_argument("all inputs must share one scale");
            if (!ct.in_ntt_domain_)
                throw std::invalid_argument("inputs must be in NTT domain");
            if (ct.rescale_required_ || ct.relinearization_required_)
                throw std::invalid_argument(
                    "spend pending rescale/relinearisation before composing");
            if (ct.storage_type_ != storage_type::DEVICE)
                throw std::invalid_argument("inputs must be device-resident");
        }

        const int l = small_q_size_ - depth;
        if (l < 1)
            throw std::invalid_argument("inputs are below the shared chain");

        const cudaStream_t stream = options.stream_;
        const size_t total = static_cast<size_t>(2) * l * n_big_;

        std::vector<const Data64*> ptrs(k_);
        for (int j = 0; j < k_; j++)
        {
            if (inputs[static_cast<size_t>(j)].device_locations_.size() <
                static_cast<size_t>(2) * l * n_small_)
                throw std::invalid_argument("invalid ciphertext size");
            ptrs[j] = inputs[static_cast<size_t>(j)].device_locations_.data();
        }

        // Gather the k ciphertexts into the contiguous staging layout, then
        // take the whole batch out of the small NTT domain in one launch.
        DeviceVector<Data64> staged(total, stream);
        DeviceVector<const Data64*> dptrs(ptrs, stream);
        ringswitch_gather_kernel<<<dim3((n_small_ >> 8), l, 2 * k_), 256, 0,
                                   stream>>>(dptrs.data(), staged.data(),
                                             n_power_small_, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_INTT_Inplace(
            staged.data(), small_->intt_table_->data(),
            small_->modulus_->data(),
            inverse_cfg(n_power_small_, small_->n_inverse_->data(), stream),
            2 * l * k_, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        Ciphertext<Scheme::CKKS> composed =
            allocate_at_level(big_, l, inputs[0], stream);

        ringswitch_interleave_kernel<<<dim3((n_big_ >> 8), l, 2), 256, 0,
                                       stream>>>(
            staged.data(), composed.data(), n_power_big_, k_power_, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(composed.data(), big_->ntt_table_->data(),
                                big_->modulus_->data(),
                                forward_cfg(n_power_big_, stream), 2 * l, l);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return composed;
    }

} // namespace heongpu
