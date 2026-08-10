// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/encryptor.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace heongpu
{
    namespace
    {
        /// Debug switch shared with the staged mod-down rewrite: when
        /// HEONGPU_MODDOWN_CHECK is set, the staged encryption mod-down also
        /// runs the legacy kernel and compares the outputs word for word.
        bool enc_moddown_check_enabled()
        {
            static const bool enabled = [] {
                const char* env = std::getenv("HEONGPU_MODDOWN_CHECK");
                return (env != nullptr) && (std::atoi(env) != 0);
            }();
            return enabled;
        }
    } // namespace
    __host__
    HEEncryptor<Scheme::CKKS>::HEEncryptor(HEContext<Scheme::CKKS> context,
                                           Publickey<Scheme::CKKS>& public_key)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        context_ = std::move(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        if (public_key.storage_type_ == storage_type::DEVICE)
        {
            public_key_ = public_key.device_locations_;
        }
        else
        {
            public_key.store_in_device();
            public_key_ = public_key.device_locations_;
        }
    }

    __host__ void HEEncryptor<Scheme::CKKS>::encrypt_ckks(
        Ciphertext<Scheme::CKKS>& ciphertext,
        Plaintext<Scheme::CKKS>& plaintext, const cudaStream_t stream)
    {
        const auto* ctx = context_.get();
        const int n = ctx->n;
        const int n_power = ctx->n_power;
        const int Q_prime_size = ctx->Q_prime_size;
        const int Q_size = ctx->Q_size;

        DeviceVector<Data64> output_memory((2 * n * Q_size), stream);

        DeviceVector<Data64> gpu_space(5 * Q_prime_size * n, stream);
        Data64* u_poly = gpu_space.data();
        Data64* error_poly = u_poly + (Q_prime_size * n);
        Data64* pk_u_poly = error_poly + (2 * Q_prime_size * n);

        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                u_poly, ctx->modulus_->data(), n_power, Q_prime_size, 1,
                stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, ctx->modulus_->data(), n_power,
                Q_prime_size, 2, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(u_poly, ctx->ntt_table_->data(),
                                ctx->modulus_->data(), cfg_ntt, Q_prime_size,
                                Q_prime_size);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size, 2), 256, 0, stream>>>(
            public_key_.data(), u_poly, pk_u_poly, ctx->modulus_->data(),
            n_power, Q_prime_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = ctx->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(pk_u_poly, ctx->intt_table_->data(),
                                 ctx->modulus_->data(), cfg_intt,
                                 2 * Q_prime_size, Q_prime_size);

        // Staged mod-down: chain over pk+e once, then the per-limb tail.
        DeviceVector<Data64> moddown_stage(
            static_cast<size_t>(2) * ctx->P_size * n, stream);
        enc_div_lastq_ckks_p_chain(
            pk_u_poly, error_poly, moddown_stage.data(), ctx->modulus_->data(),
            ctx->half_p_->data(), ctx->half_mod_->data(),
            ctx->last_q_modinv_->data(), n, n_power, Q_prime_size, Q_size,
            ctx->P_size, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        enc_div_lastq_ckks_stage_two_kernel<<<dim3((n >> 8), Q_size, 2), 256, 0,
                                              stream>>>(
            pk_u_poly, error_poly, moddown_stage.data(), output_memory.data(),
            ctx->modulus_->data(), ctx->half_mod_->data(),
            ctx->last_q_modinv_->data(), n_power, Q_prime_size, Q_size,
            ctx->P_size);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (enc_moddown_check_enabled())
        {
            DeviceVector<Data64> legacy(static_cast<size_t>(2) * Q_size * n,
                                        stream);
            enc_div_lastq_ckks_kernel<<<dim3((n >> 8), Q_size, 2), 256, 0,
                                        stream>>>(
                pk_u_poly, error_poly, legacy.data(), ctx->modulus_->data(),
                ctx->half_p_->data(), ctx->half_mod_->data(),
                ctx->last_q_modinv_->data(), n_power, Q_prime_size, Q_size,
                ctx->P_size);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            const size_t count = static_cast<size_t>(2) * Q_size * n;
            std::vector<Data64> host_legacy(count);
            std::vector<Data64> host_staged(count);
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                host_legacy.data(), legacy.data(), count * sizeof(Data64),
                cudaMemcpyDeviceToHost, stream));
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                host_staged.data(), output_memory.data(),
                count * sizeof(Data64), cudaMemcpyDeviceToHost, stream));
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < count; i++)
            {
                if (host_legacy[i] != host_staged[i])
                {
                    std::fprintf(stderr,
                                 "HEONGPU_MODDOWN_CHECK: enc_div_lastq_ckks "
                                 "mismatch at word %zu: legacy %llu staged "
                                 "%llu\n",
                                 i,
                                 static_cast<unsigned long long>(
                                     host_legacy[i]),
                                 static_cast<unsigned long long>(
                                     host_staged[i]));
                    throw std::runtime_error(
                        "staged encryption mod-down diverged from the legacy "
                        "kernel");
                }
            }
        }

        gpuntt::GPU_NTT_Inplace(output_memory.data(), ctx->ntt_table_->data(),
                                ctx->modulus_->data(), cfg_ntt, 2 * Q_size,
                                Q_size);

        cipher_message_add_kernel<<<dim3((n >> 8), Q_size, 1), 256, 0,
                                    stream>>>(output_memory.data(),
                                              plaintext.data(),
                                              ctx->modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ciphertext.memory_set(std::move(output_memory));
    }

} // namespace heongpu
