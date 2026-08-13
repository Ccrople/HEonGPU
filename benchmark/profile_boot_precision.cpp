// Copyright 2026 FIDESlib/PCMM measurement work.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Bootstrapping precision sweep driver.
//
// Measures decrypted precision immediately after ONE bootstrapping call (no
// Meta-BTS style iteration) for the three real/complex-domain CKKS
// bootstrapping models in HEonGPU, as a function of the first-modulus size q0
// and the scaling-prime size p (working scale = 2^p, every data prime p bits).
//
// Usage:
//   boot_precision_sweep regular <logN> <q0> <p> [taylor=11] [ctos=3] [stoc=3]
//   boot_precision_sweep slim    <logN> <q0> <p> [taylor=7]  [ctos=3] [stoc=3]
//   boot_precision_sweep v2      <logN> <q0> <p> [sine=60] [cts=56] [stc=p-1]
//                                [nbase=9] [sinedeg=30] [dangle=3] [K=16]
//
// Messages are uniform random in [-1, 1) (complex for regular/v2, real for
// slim), the standard range precision is quoted against. One greppable
// "RESULT" line goes to stdout; any exception is caught and reported in that
// line so a sweep script keeps going.
//
// The v2 model additionally reads the parameters that are NOT positional
// arguments from the environment, so a sweep can move them without a rebuild.
// These are the linear-transform and key-switching knobs — the 56% of a
// bootstrap that the sine-degree arguments above do not reach:
//
//   HEONGPU_BP_CTS_PIECE  (default 4)   CoeffToSlot DFT radix pieces
//   HEONGPU_BP_STC_PIECE  (default 3)   SlotToCoeff DFT radix pieces
//   HEONGPU_BP_CTS_RATIO  (default 2)   BSGS giant/baby ratio, CoeffToSlot
//   HEONGPU_BP_STC_RATIO  (default 2)   BSGS giant/baby ratio, SlotToCoeff
//   HEONGPU_BP_SPECIALS   (default 5)   number of 61-bit special (P) primes
//   HEONGPU_BP_HAMMING    (default 192) dense secret hamming weight
//   HEONGPU_BP_SPARSE_HW  (default 32)  sparse boot secret hamming weight
//   HEONGPU_BP_REPS       (default 1)   timed bootstraps; when reps > 1 the
//                                       first is a discarded warm-up
//
// Each CtoS/StoC piece costs exactly one prime, so the chain follows the
// setting: L = 1 + nbase + stc_piece + n_sine + cts_piece. Both the reported
// L and the generated modulus chain move with it.

#include <heongpu/heongpu.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct Args
{
    std::string model;
    int logN;
    int q0;
    int p;
    int a5, a6, a7, a8, a9, a10, a11; // model-specific extras, -1 = default
};

int parse_int(const char* s)
{
    return std::stoi(std::string(s));
}

// Environment knobs for the v2 model. Kept out of the positional argument
// list so existing sweep scripts keep working unchanged.
int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return fallback;
    }
    try
    {
        return std::stoi(std::string(v));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

double env_double(const char* name, double fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return fallback;
    }
    try
    {
        return std::stod(std::string(v));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

void result_line(const Args& a, const std::string& extras,
                 const std::string& status)
{
    std::cout << "RESULT model=" << a.model << " logN=" << a.logN
              << " q0=" << a.q0 << " p=" << a.p << " " << extras
              << " status=" << status << std::endl;
}

std::vector<Complex64> random_complex(int slots, bool real_only)
{
    std::mt19937_64 rng(20260806ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex64> v;
    v.reserve(slots);
    for (int i = 0; i < slots; i++)
    {
        double re = dist(rng);
        double im = real_only ? 0.0 : dist(rng);
        v.push_back(Complex64(re, im));
    }
    return v;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "usage: boot_precision_sweep "
                     "<regular|slim|v2> <logN> <q0> <p> [extras]"
                  << std::endl;
        return 2;
    }

    Args a;
    a.model = argv[1];
    a.logN = parse_int(argv[2]);
    a.q0 = parse_int(argv[3]);
    a.p = parse_int(argv[4]);
    a.a5 = (argc > 5) ? parse_int(argv[5]) : -1;
    a.a6 = (argc > 6) ? parse_int(argv[6]) : -1;
    a.a7 = (argc > 7) ? parse_int(argv[7]) : -1;
    a.a8 = (argc > 8) ? parse_int(argv[8]) : -1;
    a.a9 = (argc > 9) ? parse_int(argv[9]) : -1;
    a.a10 = (argc > 10) ? parse_int(argv[10]) : -1;
    a.a11 = (argc > 11) ? parse_int(argv[11]) : -1;

    const size_t N = size_t(1) << a.logN;
    const int slots = int(N / 2);
    const double scale = std::pow(2.0, a.p);

    std::string extras = "";

    try
    {
        if (a.model == "regular" || a.model == "slim")
        {
            const bool regular = (a.model == "regular");
            const int taylor = (a.a5 > 0) ? a.a5 : (regular ? 11 : 7);
            const int ctos = (a.a6 > 0) ? a.a6 : 3;
            const int stoc = (a.a7 > 0) ? a.a7 : 3;
            // Levels the bootstrap itself consumes are ctos + taylor + 8
            // (regular) plus the StoC pieces; leave one working level after.
            // HEONGPU_BP_EXTRA_L lengthens the chain WITHOUT touching the
            // algorithm, so v1 can be priced at the same number of returned
            // levels as v2 rather than only at its own minimal chain.
            const int extra_L = std::max(0, env_int("HEONGPU_BP_EXTRA_L", 0));
            const int reps = std::max(1, env_int("HEONGPU_BP_REPS", 1));
            const int L = ctos + stoc + taylor + 8 + 2 + extra_L;

            {
                std::ostringstream os;
                os << "taylor=" << taylor << " ctos=" << ctos
                   << " stoc=" << stoc << " L=" << L << " extra_L=" << extra_L
                   << " reps=" << reps;
                extras = os.str();
            }

            heongpu::HEContext<heongpu::Scheme::CKKS> context =
                heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                    heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(N);
            std::vector<int> q_bits(L, a.p);
            q_bits[0] = a.q0;
            context->set_coeff_modulus_bit_sizes(q_bits, {60, 60, 60});
            context->generate();

            heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
            heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, 16);
            keygen.generate_secret_key(secret_key);
            heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
            keygen.generate_public_key(public_key, secret_key);
            heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
            keygen.generate_relin_key(relin_key, secret_key);

            heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
            heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                                  public_key);
            heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                                  secret_key);
            heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(
                context, encoder);

            std::vector<Complex64> message = random_complex(slots, !regular);
            std::vector<double> message_real(slots);
            for (int i = 0; i < slots; i++)
            {
                message_real[i] = message[i].real();
            }

            heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
            if (regular)
            {
                encoder.encode(P1, message, scale);
            }
            else
            {
                encoder.encode(P1, message_real, scale);
            }
            heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
            encryptor.encrypt(C1, P1);

            heongpu::BootstrappingConfig boot_config(ctos, stoc, taylor, true);
            operators.generate_bootstrapping_params(
                scale, boot_config,
                regular ? heongpu::arithmetic_bootstrapping_type::
                              REGULAR_BOOTSTRAPPING
                        : heongpu::arithmetic_bootstrapping_type::
                              SLIM_BOOTSTRAPPING);

            std::vector<int> key_index = operators.bootstrapping_key_indexs();
            std::cout << "galois keys: " << key_index.size() << std::endl;
            heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context,
                                                                 key_index);
            keygen.generate_galois_key(galois_key, secret_key);

            const int drops = regular ? (L - 1) : (L - 1 - stoc);
            for (int i = 0; i < drops; i++)
            {
                operators.mod_drop_inplace(C1);
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot(context);
            std::vector<float> timings;
            timings.reserve(reps);
            for (int r = 0; r < reps; r++)
            {
                heongpu::Ciphertext<heongpu::Scheme::CKKS> C_in = C1;
                cudaEventRecord(start);
                cipher_boot =
                    regular ? operators.regular_bootstrapping(C_in, galois_key,
                                                              relin_key)
                            : operators.slim_bootstrapping(C_in, galois_key,
                                                           relin_key);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float rep_ms = 0;
                cudaEventElapsedTime(&rep_ms, start, stop);
                timings.push_back(rep_ms);
            }
            const int first_kept = (reps > 1) ? 1 : 0;
            const float ms =
                std::accumulate(timings.begin() + first_kept, timings.end(),
                                0.0f) /
                float(timings.size() - first_kept);

            heongpu::Plaintext<heongpu::Scheme::CKKS> P_res(context);
            decryptor.decrypt(P_res, cipher_boot);
            std::vector<Complex64> decrypted;
            encoder.decode(decrypted, P_res);

            heongpu::PrecisionStats prec =
                regular ? heongpu::get_precision_stats(message, decrypted)
                        : heongpu::get_precision_stats(message_real,
                                                       decrypted);
            std::cout << prec.to_string() << std::endl;

            std::ostringstream os;
            os << extras << " depth_after=" << cipher_boot.depth()
               << " boot_ms=" << ms
               << " mean_prec_real=" << prec.mean_precision.real
               << " mean_prec_l2=" << prec.mean_precision.l2
               << " worst_prec_l2=" << prec.min_precision.l2
               << " max_err_l2=" << prec.max_delta.l2;
            extras = os.str();
            result_line(a, extras, "OK");
        }
        else if (a.model == "v2")
        {
            const int sine_bits = (a.a5 > 0) ? a.a5 : 60;
            const int cts_bits = (a.a6 > 0) ? a.a6 : 56;
            const int stc_bits = (a.a7 > 0) ? a.a7 : (a.p - 1);
            const int nbase = (a.a8 > 0) ? a.a8 : 9;
            const int sinedeg = (a.a9 > 0) ? a.a9 : 30;
            const int dangle = (a.a10 >= 0) ? a.a10 : 3;
            const int K = (a.a11 > 0) ? a.a11 : 16;

            // Chebyshev depth for the cosine poly plus the double-angle
            // squarings decides how many sine-stage primes the chain needs
            // (deg 30, da 3 gives the shipped 8).
            const int sine_depth =
                int(std::ceil(std::log2(double(sinedeg + 1))));
            const int n_sine = sine_depth + dangle;

            // Linear-transform and key-switch knobs come from the
            // environment; each DFT piece costs exactly one prime, so the
            // chain layout below is derived from them rather than fixed.
            const int cts_piece = env_int("HEONGPU_BP_CTS_PIECE", 4);
            const int stc_piece = env_int("HEONGPU_BP_STC_PIECE", 3);
            const double cts_ratio = env_double("HEONGPU_BP_CTS_RATIO", 2.0);
            const double stc_ratio = env_double("HEONGPU_BP_STC_RATIO", 2.0);
            const int specials = env_int("HEONGPU_BP_SPECIALS", 5);
            const int hamming = env_int("HEONGPU_BP_HAMMING", 192);
            const int sparse_hw = env_int("HEONGPU_BP_SPARSE_HW", 32);
            const int reps = std::max(1, env_int("HEONGPU_BP_REPS", 1));

            if (cts_piece < 2 || cts_piece > 5 || stc_piece < 2 ||
                stc_piece > 5)
            {
                throw std::out_of_range(
                    "HEONGPU_BP_CTS_PIECE / HEONGPU_BP_STC_PIECE must be in "
                    "[2, 5]");
            }
            if (specials < 1)
            {
                throw std::out_of_range("HEONGPU_BP_SPECIALS must be >= 1");
            }

            const int stc_start = nbase + stc_piece;
            const int em_start = stc_start + n_sine;
            const int cts_start = em_start + cts_piece;
            const int L = 1 + nbase + stc_piece + n_sine + cts_piece;

            {
                std::ostringstream os;
                os << "sine=" << sine_bits << " cts=" << cts_bits
                   << " stc=" << stc_bits << " nbase=" << nbase
                   << " sinedeg=" << sinedeg << " dangle=" << dangle
                   << " K=" << K << " L=" << L
                   << " cts_piece=" << cts_piece
                   << " stc_piece=" << stc_piece << " cts_ratio=" << cts_ratio
                   << " stc_ratio=" << stc_ratio << " specials=" << specials
                   << " hw=" << hamming << " sparse_hw=" << sparse_hw
                   << " reps=" << reps;
                extras = os.str();
            }

            heongpu::HEContext<heongpu::Scheme::CKKS> context =
                heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                    heongpu::sec_level_type::none);
            context->set_poly_modulus_degree(N);

            std::vector<int> q_bits;
            q_bits.push_back(a.q0);
            for (int i = 0; i < nbase; i++)
                q_bits.push_back(a.p);
            for (int i = 0; i < stc_piece; i++)
                q_bits.push_back(stc_bits);
            for (int i = 0; i < n_sine; i++)
                q_bits.push_back(sine_bits);
            for (int i = 0; i < cts_piece; i++)
                q_bits.push_back(cts_bits);

            // The shipped v2 example uses 61-bit special primes; the bit-size
            // API caps user primes at 60, so build the values directly.
            std::vector<Modulus64> q_mods =
                heongpu::generate_primes(N, q_bits);
            std::vector<Data64> q_vals;
            for (auto& m : q_mods)
                q_vals.push_back(m.value);
            std::vector<Data64> p_vals = heongpu::generate_proper_primes(
                Data64(2) * Data64(N), 61, specials);
            context->set_coeff_modulus_values(q_vals, p_vals);
            context->generate();

            heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
            heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context,
                                                                 hamming);
            keygen.generate_secret_key_v2(secret_key);
            heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
            keygen.generate_public_key(public_key, secret_key);
            heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
            keygen.generate_relin_key(relin_key, secret_key);

            heongpu::Secretkey<heongpu::Scheme::CKKS> sparse_key(context,
                                                                 sparse_hw);
            keygen.generate_secret_key_v2(sparse_key);
            heongpu::Switchkey<heongpu::Scheme::CKKS> swk_dense_to_sparse(
                context);
            keygen.generate_switch_key(swk_dense_to_sparse, sparse_key,
                                       secret_key);
            heongpu::Switchkey<heongpu::Scheme::CKKS> swk_sparse_to_dense(
                context);
            keygen.generate_switch_key(swk_sparse_to_dense, secret_key,
                                       sparse_key);

            heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
            heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context,
                                                                  public_key);
            heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context,
                                                                  secret_key);
            heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(
                context, encoder);

            std::vector<Complex64> message = random_complex(slots, false);
            heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
            encoder.encode(P1, message, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
            encryptor.encrypt(C1, P1);

            heongpu::EvalModConfig eval_mod_config(0, em_start, 256.0, K,
                                                   sinedeg, dangle, 0, 0.0);
            heongpu::BootstrappingConfigV2 boot_config(
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::SLOTS_TO_COEFFS, stc_start,
                    stc_ratio, stc_piece),
                eval_mod_config,
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::COEFFS_TO_SLOTS, cts_start,
                    cts_ratio, cts_piece));
            operators.generate_bootstrapping_params_v2(scale, boot_config);

            std::vector<int> key_index = operators.bootstrapping_key_indexs();
            std::cout << "galois keys: " << key_index.size() << std::endl;
            heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context,
                                                                 key_index);
            keygen.generate_galois_key(galois_key, secret_key);

            for (int i = 0; i < L - 1; i++)
            {
                operators.mod_drop_inplace(C1);
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // With reps > 1 the first bootstrap is a warm-up and is dropped:
            // it pays the pool's first-touch and the lazy device-side key
            // material, which is not what a steady-state block sees.
            heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot(context);
            std::vector<float> timings;
            timings.reserve(reps);
            for (int r = 0; r < reps; r++)
            {
                heongpu::Ciphertext<heongpu::Scheme::CKKS> C_in = C1;
                cudaEventRecord(start);
                cipher_boot = operators.regular_bootstrapping_v2(
                    C_in, galois_key, relin_key, &swk_dense_to_sparse,
                    &swk_sparse_to_dense);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float rep_ms = 0;
                cudaEventElapsedTime(&rep_ms, start, stop);
                timings.push_back(rep_ms);
            }

            const int first_kept = (reps > 1) ? 1 : 0;
            const float ms =
                std::accumulate(timings.begin() + first_kept, timings.end(),
                                0.0f) /
                float(timings.size() - first_kept);
            const float ms_min =
                *std::min_element(timings.begin() + first_kept, timings.end());

            heongpu::Plaintext<heongpu::Scheme::CKKS> P_res(context);
            decryptor.decrypt(P_res, cipher_boot);
            std::vector<Complex64> decrypted;
            encoder.decode(decrypted, P_res);

            heongpu::PrecisionStats prec =
                heongpu::get_precision_stats(message, decrypted);
            std::cout << prec.to_string() << std::endl;

            std::ostringstream os;
            os << extras << " depth_after=" << cipher_boot.depth()
               << " boot_ms=" << ms << " boot_ms_min=" << ms_min
               << " galois_keys=" << key_index.size()
               << " mean_prec_real=" << prec.mean_precision.real
               << " mean_prec_l2=" << prec.mean_precision.l2
               << " worst_prec_l2=" << prec.min_precision.l2
               << " max_err_l2=" << prec.max_delta.l2;
            extras = os.str();
            result_line(a, extras, "OK");
        }
        else
        {
            std::cerr << "unknown model: " << a.model << std::endl;
            return 2;
        }
    }
    catch (const std::exception& e)
    {
        std::ostringstream os;
        os << extras << " msg=\"" << e.what() << "\"";
        result_line(a, os.str(), "EXCEPTION");
    }
    catch (...)
    {
        result_line(a, extras + " msg=\"unknown\"", "EXCEPTION");
    }

    return 0;
}
