// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// THE LOW-RING BAE PRODUCT, ON THE CLOCK.
// =======================================
//
// bae_lowring.cuh derives that the a-part GEMM's width is the lattice
// dimension, so ring switching down by k cuts the product's arithmetic by
// (k+1)/2 and its key switches by k. This target measures whether that
// survives contact with the kernels, because two things in the derivation are
// not free and neither is in the MAC count:
//
//   * the descent is a real key switch at the BIG ring, and a key switch at
//     logN 14 is not cheap;
//   * the small-ring GEMMs are k times narrower, and narrow GPU work is
//     launch-bound — the effect §15.2 recorded for the crossings and §25.2 for
//     the k = 32 product.
//
// So the honest unit is not one projection. A block descends ONCE, takes
// seven projections, and ascends ONCE, because every projection is row -> row.
// Both are reported: the single product, which flatters the big ring, and the
// seven-projection block, which is what a layer actually pays.
//
// Both paths are checked against a host reference before either is timed. A
// benchmark of a wrong answer is worth nothing and this file has two very
// different code paths in it.
//
// The small ring cannot be Sylph's 256: defines.h sets MIN_POLY_DEGREE = 4096
// and the library refuses anything below it outright ("Poly modulus degree is
// not supported"), at every security level. 4096 is the floor here by
// construction, before security is consulted at all.
//
//   HEONGPU_BLR_LOGN_BIG    pipeline ring exponent        14
//   HEONGPU_BLR_LOGN_SMALL  product ring exponent         12
//   HEONGPU_BLR_WIDTH       in and out channels          512
//   HEONGPU_BLR_REPS        timed repetitions              3

#include <heongpu/heongpu.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;

namespace
{
    using Clock = std::chrono::steady_clock;

    double ms_since(Clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0)
            .count();
    }

    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }

    std::vector<double> random_matrix(int rows, int cols, unsigned seed,
                                      double lo, double hi)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        std::vector<double> m(static_cast<size_t>(rows) * cols);
        for (auto& v : m)
            v = dist(rng);
        return m;
    }

    std::vector<double> transpose(const std::vector<double>& w, int rows,
                                  int cols)
    {
        std::vector<double> t(static_cast<size_t>(cols) * rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                t[static_cast<size_t>(j) * rows + i] =
                    w[static_cast<size_t>(i) * cols + j];
        return t;
    }

    std::vector<double> host_product(const std::vector<double>& U,
                                     const std::vector<double>& M, int d1,
                                     int d2, int d3)
    {
        std::vector<double> out(static_cast<size_t>(d1) * d3, 0.0);
        for (int i = 0; i < d1; ++i)
            for (int t = 0; t < d2; ++t)
            {
                const double u = U[static_cast<size_t>(i) * d2 + t];
                for (int j = 0; j < d3; ++j)
                    out[static_cast<size_t>(i) * d3 + j] +=
                        u * M[static_cast<size_t>(t) * d3 + j];
            }
        return out;
    }

    double worst_error(const std::vector<double>& a,
                       const std::vector<double>& b)
    {
        double worst = 0.0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
            worst = std::max(worst, std::abs(a[i] - b[i]));
        return worst;
    }
} // namespace

int main()
{
    const int logn_big = EnvInt("HEONGPU_BLR_LOGN_BIG", 14);
    const int logn_small = EnvInt("HEONGPU_BLR_LOGN_SMALL", 12);
    const int width = EnvInt("HEONGPU_BLR_WIDTH", 512);
    const int reps = EnvInt("HEONGPU_BLR_REPS", 3);

    const int n_big = 1 << logn_big;
    const int n_small = 1 << logn_small;
    const int k = n_big / n_small;

    std::cout << "[blr] pipeline ring " << n_big << ", product ring " << n_small
              << ", k = " << k << ", width " << width << std::endl;

    if (width % k != 0)
    {
        std::cout << "[blr] width must be a multiple of k; nothing to do"
                  << std::endl;
        return 0;
    }

    // sec_level_type::none here deliberately: this target measures ARITHMETIC.
    // Whether the pair is legal is profile_bae_ring_security's question and it
    // is answered there, against the library, rather than assumed here.
    const std::vector<int> q_bits = {60, 50, 50, 50};
    heongpu::HEContext<S> big =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    big->set_poly_modulus_degree(static_cast<size_t>(n_big));
    big->set_coeff_modulus_bit_sizes(q_bits, {60, 60});
    big->generate();

    const auto primes = big->get_key_modulus();
    std::vector<Data64> q_vals, p_vals;
    for (size_t i = 0; i < q_bits.size(); ++i)
        q_vals.push_back(primes[i].value);
    p_vals.push_back(primes[q_bits.size()].value);

    heongpu::HEContext<S> small =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    small->set_poly_modulus_degree(static_cast<size_t>(n_small));
    small->set_coeff_modulus_values(q_vals, p_vals);
    small->generate();

    heongpu::HEKeyGenerator<S> keygen(big);
    std::vector<int> sk_coefficients(n_big, 0);
    {
        std::mt19937 rng(20260814u);
        std::uniform_int_distribution<int> pick(0, 2);
        for (int i = 0; i < n_big; ++i)
            sk_coefficients[i] = pick(rng) - 1;
    }
    heongpu::Secretkey<S> secret(sk_coefficients, big);
    heongpu::Publickey<S> pub(big);
    keygen.generate_public_key(pub, secret);

    heongpu::HEEncryptor<S> encryptor(big, pub);
    heongpu::HEDecryptor<S> decryptor(big, secret);
    heongpu::HEEncoder<S> encoder(big);
    heongpu::HEArithmeticOperator<S> ops(big, encoder);

    heongpu::HERingSwitchOperator<S> rs(big, small);
    auto pair = rs.make_secret_pair(64, 0xBAE10002ULL);
    rs.generate_keys(keygen, secret, pair.embedded);

    heongpu::HEEncoder<S> small_encoder(small);
    heongpu::HEArithmeticOperator<S> small_ops(small, small_encoder);

    heongpu::HEBaeLowRingPcmm low(big, small, rs);

    const double scale = std::pow(2.0, 40);
    const int tokens = n_small;

    const std::vector<double> M = random_matrix(width, tokens, 7u, -1.0, 1.0);
    const std::vector<double> weight =
        random_matrix(width, width, 8u, -0.25, 0.25);
    const std::vector<double> want =
        host_product(transpose(weight, width, width), M, width, width, tokens);

    auto encrypt_rows = [&]()
    {
        auto coeffs = low.encoder().encode_matrix(M, width, scale);
        std::vector<heongpu::Ciphertext<S>> out;
        out.reserve(coeffs.size());
        for (auto& c : coeffs)
        {
            heongpu::Plaintext<S> p(big);
            low.encoder().load_coefficients(p, c, scale);
            heongpu::Ciphertext<S> ct(big);
            encryptor.encrypt(ct, p);
            out.push_back(std::move(ct));
        }
        return out;
    };

    auto decrypt_rows = [&](std::vector<heongpu::Ciphertext<S>>& ct, int rows)
    {
        std::vector<std::vector<int64_t>> coeffs;
        coeffs.reserve(ct.size());
        for (auto& c : ct)
        {
            heongpu::Plaintext<S> p(big);
            decryptor.decrypt(p, c);
            coeffs.push_back(low.encoder().extract_coefficients(p));
        }
        return low.encoder().decode_matrix(coeffs, rows, ct.front().scale());
    };

    // ---------------------------------------------------------------------
    // Correctness first, both paths, before anything is timed.
    // ---------------------------------------------------------------------
    std::cout << "[blr] generating ModPack keys for the big-ring path (k = "
              << k << ")..." << std::endl;
    low.encoder().generate_modpack_keys(keygen, secret, sk_coefficients);

    {
        auto in = encrypt_rows();
        std::vector<heongpu::Ciphertext<S>*> ptrs;
        for (auto& c : in)
            ptrs.push_back(&c);
        std::vector<heongpu::Ciphertext<S>> out;
        low.encoder().project(out, ptrs, weight, width, width, ops);
        std::cout << "[blr] big-ring path worst error  "
                  << worst_error(decrypt_rows(out, width), want) << std::endl;
    }
    {
        auto in = encrypt_rows();
        auto down = low.descend(in, ops);
        std::vector<heongpu::Ciphertext<S>> out;
        low.project(out, down, weight, width, width, small_ops);
        auto up = low.ascend(out, ops);
        std::cout << "[blr] low-ring path worst error  "
                  << worst_error(decrypt_rows(up, width), want) << std::endl;
    }

    // ---------------------------------------------------------------------
    // Timing.
    // ---------------------------------------------------------------------
    auto time_big_projection = [&]()
    {
        auto in = encrypt_rows();
        std::vector<heongpu::Ciphertext<S>*> ptrs;
        for (auto& c : in)
            ptrs.push_back(&c);
        std::vector<heongpu::Ciphertext<S>> out;
        low.encoder().project(out, ptrs, weight, width, width, ops);
        cudaDeviceSynchronize();

        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            std::vector<heongpu::Ciphertext<S>> o;
            low.encoder().project(o, ptrs, weight, width, width, ops);
        }
        cudaDeviceSynchronize();
        return ms_since(t0) / reps;
    };

    auto in_big = encrypt_rows();

    auto time_descend = [&]()
    {
        auto warm = encrypt_rows();
        (void) low.descend(warm, ops);
        cudaDeviceSynchronize();

        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            auto fresh = in_big;
            (void) low.descend(fresh, ops);
        }
        cudaDeviceSynchronize();
        return ms_since(t0) / reps;
    };

    auto down_ref = low.descend(in_big, ops);

    auto time_small_projection = [&]()
    {
        std::vector<heongpu::Ciphertext<S>> out;
        low.project(out, down_ref, weight, width, width, small_ops);
        cudaDeviceSynchronize();

        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
        {
            std::vector<heongpu::Ciphertext<S>> o;
            low.project(o, down_ref, weight, width, width, small_ops);
        }
        cudaDeviceSynchronize();
        return ms_since(t0) / reps;
    };

    std::vector<heongpu::Ciphertext<S>> product_ref;
    low.project(product_ref, down_ref, weight, width, width, small_ops);

    auto time_ascend = [&]()
    {
        (void) low.ascend(product_ref, ops);
        cudaDeviceSynchronize();

        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r)
            (void) low.ascend(product_ref, ops);
        cudaDeviceSynchronize();
        return ms_since(t0) / reps;
    };

    const double t_big = time_big_projection();
    const double t_down = time_descend();
    const double t_small = time_small_projection();
    const double t_up = time_ascend();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[blr] "
                 "---------------------------------------------------------"
              << std::endl;
    std::cout << "[blr] big-ring projection (ModDecomp + GEMMs + ModPack)  "
              << std::setw(10) << t_big << " ms" << std::endl;
    std::cout << "[blr] descend  (" << (width / k) << " key switches at "
              << n_big << ")                    " << std::setw(10) << t_down
              << " ms" << std::endl;
    std::cout << "[blr] low-ring projection (two GEMMs at " << n_small
              << ")            " << std::setw(10) << t_small << " ms"
              << std::endl;
    std::cout << "[blr] ascend   (" << (width / k) << " key switches at "
              << n_big << ")                    " << std::setw(10) << t_up
              << " ms" << std::endl;
    std::cout << "[blr] "
                 "---------------------------------------------------------"
              << std::endl;

    for (int projections : {1, 7})
    {
        const double big_total = projections * t_big;
        const double low_total = t_down + projections * t_small + t_up;
        std::cout << "[blr] " << projections
                  << " projection(s): big ring " << std::setw(10) << big_total
                  << " ms   low ring " << std::setw(10) << low_total
                  << " ms   " << std::setprecision(2)
                  << (big_total / low_total) << "x" << std::setprecision(3)
                  << std::endl;
    }

    const auto counted_big =
        heongpu::HEBaeLowRingPcmm::big_ring_cost(width, width, tokens, n_big,
                                                 n_small);
    const auto counted_low =
        heongpu::HEBaeLowRingPcmm::low_ring_cost(width, width, tokens, n_big,
                                                 n_small);
    std::cout << "[blr] counted MAC ratio for one projection: "
              << std::setprecision(2)
              << (counted_big.per_token / counted_low.per_token)
              << "x  (measured above includes the key switches the count "
                 "does not price)"
              << std::endl;

    return 0;
}
