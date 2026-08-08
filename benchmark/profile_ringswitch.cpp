// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cost and precision of CKKS ring switching at the two-ring Llama-3 shape:
// logN 16 (where the nonlinears and the level budget live) down to logN 12
// (where Algorithm 5's N/2 − 1 Galois keys fit), and back.
//
// What it measures, per direction:
//   - the full switch (one key switch + the coefficient move), and
//   - the keyswitch-free half alone (split/interleave + the two NTT hops),
// so the key switch's share — expected to dominate — is read off directly.
// A round trip's worst absolute coefficient error closes the loop: the
// decomposition itself is exact, so everything observed is the two key
// switches.
//
// Environment knobs (defaults in parentheses):
//   HEONGPU_RS_LOGN_BIG   (16)   big ring degree, log2
//   HEONGPU_RS_LOGN_SMALL (12)   small ring degree, log2
//   HEONGPU_RS_LIMBS      (14)   big Q chain length: 60-bit q0 + 50-bit rest
//   HEONGPU_RS_SHARED     (=LIMBS) how many bottom primes the small ring gets
//   HEONGPU_RS_ITERS      (10)   timed iterations per direction
//   HEONGPU_RS_HW         (64)   Hamming weight of the small-ring secret

#include <heongpu/heongpu.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    int env_int(const char* name, int fallback)
    {
        const char* v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    }

    double ms_since(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    }

    struct Stats
    {
        double mean = 0.0;
        double best = 1e300;
        int n = 0;
        void add(double ms)
        {
            mean += ms;
            best = std::min(best, ms);
            n++;
        }
        double avg() const { return n ? mean / n : 0.0; }
    };
} // namespace

int main()
{
    const int logn_big = env_int("HEONGPU_RS_LOGN_BIG", 16);
    const int logn_small = env_int("HEONGPU_RS_LOGN_SMALL", 12);
    const int limbs = env_int("HEONGPU_RS_LIMBS", 14);
    const int shared = env_int("HEONGPU_RS_SHARED", limbs);
    const int iters = env_int("HEONGPU_RS_ITERS", 10);
    const int hw = env_int("HEONGPU_RS_HW", 64);

    const int n_big = 1 << logn_big;
    const int n_small = 1 << logn_small;
    const int k = n_big / n_small;

    std::cout << "ring switch profile: logN " << logn_big << " <-> "
              << logn_small << " (k = " << k << "), big chain " << limbs
              << " limbs, shared prefix " << shared << ", iters " << iters
              << std::endl;

    // Big context: 60-bit q0 + 50-bit body, two 61-bit specials (method II —
    // a single special prime has no key-switch noise budget at logN 16).
    std::vector<int> q_bits{60};
    for (int i = 1; i < limbs; i++)
        q_bits.push_back(50);

    auto big = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    big->set_poly_modulus_degree(static_cast<size_t>(n_big));
    big->set_coeff_modulus_bit_sizes(q_bits, {61, 61});
    big->generate();

    const auto primes = big->get_key_modulus();
    std::vector<Data64> q_vals, p_vals;
    for (int i = 0; i < shared; i++)
        q_vals.push_back(primes[i].value);
    p_vals.push_back(primes[limbs].value);

    auto small = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    small->set_poly_modulus_degree(static_cast<size_t>(n_small));
    small->set_coeff_modulus_values(q_vals, p_vals);
    small->generate();

    heongpu::HEKeyGenerator<S> keygen(big);
    heongpu::Secretkey<S> secret(big);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(big);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(big, pub);
    heongpu::HEDecryptor<S> decryptor(big, secret);
    heongpu::HEEncoder<S> encoder(big);
    heongpu::HEArithmeticOperator<S> ops(big, encoder);

    heongpu::HERingSwitchOperator<S> rs(big, small);
    auto pair = rs.make_secret_pair(hw, 0xC0FFEEULL);

    auto t0 = std::chrono::steady_clock::now();
    rs.generate_keys(keygen, secret, pair.embedded);
    std::cout << "switch key generation: " << ms_since(t0) << " ms (both "
              << "directions)" << std::endl;

    heongpu::HEDecryptor<S> small_decryptor(small, pair.small);
    heongpu::HEEncoder<S> small_encoder(small);

    const double scale = std::pow(2.0, 40);
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> message(n_big);
    for (auto& v : message)
        v = dist(rng);

    heongpu::Plaintext<S> plain(big);
    encoder.encode(plain, message, scale, heongpu::ExecutionOptions(),
                   heongpu::encoding::COEFFICIENT);
    heongpu::Ciphertext<S> ct(big);
    encryptor.encrypt(ct, plain);

    // Warm-up, and the working set the timed loops reuse.
    std::vector<heongpu::Ciphertext<S>> parts = rs.switch_down(ct, ops);
    heongpu::Ciphertext<S> back = rs.compose_up(parts, ops);

    Stats down_full, down_raw, up_full, up_raw;
    for (int it = 0; it < iters; it++)
    {
        auto t = std::chrono::steady_clock::now();
        std::vector<heongpu::Ciphertext<S>> p1 = rs.switch_down(ct, ops);
        down_full.add(ms_since(t));

        // The keyswitch-free half, on the identical data path. The output is
        // algebraically meaningless (the input is under sk_big, not the
        // embedded key) but the cost is exactly the split's.
        t = std::chrono::steady_clock::now();
        std::vector<heongpu::Ciphertext<S>> p2 = rs.split_embedded(ct);
        down_raw.add(ms_since(t));

        t = std::chrono::steady_clock::now();
        heongpu::Ciphertext<S> c1 = rs.compose_up(p1, ops);
        up_full.add(ms_since(t));

        t = std::chrono::steady_clock::now();
        heongpu::Ciphertext<S> c2 = rs.interleave_embedded(p1);
        up_raw.add(ms_since(t));
    }

    std::cout << "switch_down  : avg " << down_full.avg() << " ms, best "
              << down_full.best << " ms" << std::endl;
    std::cout << "  split half : avg " << down_raw.avg() << " ms  (keyswitch "
              << down_full.avg() - down_raw.avg() << " ms)" << std::endl;
    std::cout << "compose_up   : avg " << up_full.avg() << " ms, best "
              << up_full.best << " ms" << std::endl;
    std::cout << "  weave half : avg " << up_raw.avg() << " ms  (keyswitch "
              << up_full.avg() - up_raw.avg() << " ms)" << std::endl;

    // Precision. One slice at the small ring pins the split itself; the round
    // trip pins the pair of key switches.
    {
        heongpu::Plaintext<S> sp(small);
        small_decryptor.decrypt(sp, parts[k / 2]);
        std::vector<double> slice;
        small_encoder.decode(slice, sp);
        double worst = 0.0;
        for (int i = 0; i < n_small; i++)
            worst = std::max(
                worst, std::fabs(slice[i] -
                                 message[static_cast<size_t>(i) * k + k / 2]));
        std::cout << "slice " << (k / 2) << " worst abs error at small ring: "
                  << worst << std::endl;
    }
    {
        heongpu::Plaintext<S> bp(big);
        decryptor.decrypt(bp, back);
        std::vector<double> got;
        encoder.decode(got, bp);
        double worst = 0.0;
        for (int i = 0; i < n_big; i++)
            worst = std::max(worst, std::fabs(got[i] - message[i]));
        std::cout << "round trip worst abs error: " << worst << std::endl;
    }

    std::cout << "done" << std::endl;
    return 0;
}
