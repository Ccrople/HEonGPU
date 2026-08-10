// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// WHERE SHOULD EACH KANG PRODUCT RUN IN A TWO-RING LLAMA-3 BLOCK?
// ===============================================================
//
// LLAMA3_8B_LAYER_FLOW.md §6 proposes a two-ring flow (Sylph's placement):
// ring-switch DOWN before the matrix stage, run batch CCMM (Algorithm 4) AND
// rectangular PCMM (Algorithm 5) at the low ring, compose back up. §7bis
// measured that only Algorithm 5 needs the low ring (its N/2 - 1 Galois keys),
// while CCMM is ~2x cheaper per MAC at a larger ring and needs only ~d keys.
// This target settles the placement with the ring switch that now exists
// (HERingSwitchOperator), at the real 8B attention shape, by timing every leg
// of both variants:
//
//   Variant A (Sylph):  down -> CCMM(low) -> Alg5(low)  -> up
//   Variant B (split):  CCMM(high) -> down -> Alg5(low) -> up
//   Corner  (no island): CCMM(high) -> Alg5(high, N/2-1 keys at the HIGH ring)
//
// The ring pair defaults to 13 <-> 12, not Sylph's 16 <-> 12, for measured
// reasons recorded in the flow doc:
//   - attention() requires heads*d % (N/2) == 0; 32 heads of 128 = 4096
//     channels, so N <= 8192. At logN 16 the 8B attention cannot run at all.
//   - CCMM per MAC is flat above logN 13 (0.97 at 13, 0.84 at 14-16 vs 1.64
//     at 12), so a bigger high ring buys nothing further for the products.
//   - The CKKS bootstrap loses its precision above logN 14 (v2: 20.5 bits at
//     14, 17.4 at 15, 16.75 at 16), so the non-linear side wants logN <= 14.
//   - At 128-bit, N = 8192 is the SMALLEST ring holding one bootstrap-
//     compatible Kang level (2^50 under q0 2^60, log PQ 170 <= 218); N = 4096
//     holds nothing at any level (cap 109). So 2^13 is also the only
//     deployable LOW ring; 2^12 remains the profiling floor (MIN_POLY_DEGREE).
//
// Data enters the island through the descent contract: a big-ring ciphertext
// is the stride-k interleave of k small-ring column polynomials (the layout
// LLAMA3_8B_LAYER_FLOW.md §8 assigns the big ring on the matrix path, pinned
// by CkksRingSwitch.Alg5OnSwitchedDataMatchesReference). This target builds
// contract data by encrypting native small-ring operands and COMPOSE-ing them
// up, which is also exactly the up-crossing being timed; a full block
// integration produces the same layout by retargeting the high-ring bridge
// diagonals, which costs no extra levels but is not built yet.
//
//   HEONGPU_TR_LOGN_HIGH   high ring degree, log2               13
//   HEONGPU_TR_LOGN_LOW    low ring degree, log2                12
//   HEONGPU_TR_D           head dim / block size                128
//   HEONGPU_TR_HEADS       attention heads (CCMM blocks)        32
//   HEONGPU_TR_LIMBS       big Q chain length (60 + 50*rest)    3
//   HEONGPU_TR_SHARED      bottom primes the low ring receives  = LIMBS
//   HEONGPU_TR_REPS        timed repetitions per leg            3
//   HEONGPU_TR_CHECK       decrypt and compare vs host          1
//   HEONGPU_TR_ALG5_HIGH   also key + run Alg 5 at the high
//                          ring (the no-island corner; needs
//                          ~N_high/2 keys of GPU memory)        1
//   HEONGPU_TR_MODEL       model width for the Alg 5 leg
//                          (in = out channels)                  4096

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;
using Clock = std::chrono::steady_clock;

namespace
{
    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }

    std::vector<double> RandomMatrix(std::size_t count, double amp,
                                     std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-amp, amp);
        std::vector<double> out(count);
        for (double& v : out)
            v = dist(rng);
        return out;
    }

    template <typename F> double BestMs(F&& f, int reps)
    {
        double best = 1e30;
        for (int r = 0; r < reps; ++r)
        {
            cudaDeviceSynchronize();
            const auto t0 = Clock::now();
            f();
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    }

    double FreeGiB()
    {
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
    }

    struct Ledger
    {
        double down_ms = -1.0;    // full operand-set descent
        double up_ms = -1.0;      // full output-set composition
        double ccmm_low_ms = -1.0;
        double ccmm_high_ms = -1.0;
        double alg5_low_ms = -1.0;
        double alg5_high_ms = -1.0;
        double rs_keygen_ms = -1.0;
        double low_keys_gib = 0.0;
        double high_rect_keys_gib = 0.0;
    };
} // namespace

int main()
{
    const int logn_hi = EnvInt("HEONGPU_TR_LOGN_HIGH", 13);
    const int logn_lo = EnvInt("HEONGPU_TR_LOGN_LOW", 12);
    const int d = EnvInt("HEONGPU_TR_D", 128);
    const int heads = EnvInt("HEONGPU_TR_HEADS", 32);
    const int limbs = EnvInt("HEONGPU_TR_LIMBS", 3);
    const int shared = EnvInt("HEONGPU_TR_SHARED", limbs);
    const int reps = EnvInt("HEONGPU_TR_REPS", 3);
    const bool check = EnvInt("HEONGPU_TR_CHECK", 1) != 0;
    const bool alg5_high = EnvInt("HEONGPU_TR_ALG5_HIGH", 1) != 0;
    const int model = EnvInt("HEONGPU_TR_MODEL", 4096);

    const int n_hi = 1 << logn_hi;
    const int n_lo = 1 << logn_lo;
    const int k = n_hi / n_lo;
    const int half_hi = n_hi / 2;
    const int half_lo = n_lo / 2;
    const int blocks_hi = (n_hi / d) / 2; // heads one high call covers
    const int blocks_lo = (n_lo / d) / 2;

    std::cout << "[tworing] high logN " << logn_hi << ", low logN " << logn_lo
              << ", k = " << k << ", d = " << d << ", heads = " << heads
              << ", limbs = " << limbs << " (shared " << shared << ")"
              << std::endl;

    if (heads * d % half_hi != 0 || heads * d % half_lo != 0)
    {
        std::cout << "[tworing] heads*d = " << heads * d
                  << " is not a whole number of channel groups at both rings; "
                     "pick shapes the attention path accepts"
                  << std::endl;
        return 1;
    }
    if (heads % blocks_hi != 0 || heads % blocks_lo != 0)
    {
        std::cout << "[tworing] heads must be a multiple of one call's block "
                     "count at both rings ("
                  << blocks_hi << " high, " << blocks_lo << " low)"
                  << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // Contexts: the RingFixture contract. Big chain 60 + 50*(limbs-1) under
    // {60,60} specials; the low ring receives the bottom `shared` primes
    // verbatim plus the big context's first special prime.
    // ------------------------------------------------------------------
    const double scale = std::pow(2.0, 50);

    heongpu::HEContext<S> big =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    std::vector<int> q_bits(static_cast<std::size_t>(limbs), 50);
    q_bits[0] = 60;
    big->set_poly_modulus_degree(static_cast<std::size_t>(n_hi));
    big->set_coeff_modulus_bit_sizes(q_bits, {60, 60});
    big->generate();

    heongpu::HEContext<S> small =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    {
        const auto primes = big->get_key_modulus();
        std::vector<Data64> q_vals, p_vals;
        for (int i = 0; i < shared; i++)
            q_vals.push_back(primes[i].value);
        p_vals.push_back(primes[limbs].value);
        small->set_poly_modulus_degree(static_cast<std::size_t>(n_lo));
        small->set_coeff_modulus_values(q_vals, p_vals);
        small->generate();
    }

    heongpu::HEKeyGenerator<S> keygen(big);
    heongpu::Secretkey<S> secret(big, 16);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(big);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(big, pub);
    heongpu::HEDecryptor<S> decryptor(big, secret);
    heongpu::HEEncoder<S> encoder(big);
    heongpu::HEArithmeticOperator<S> ops(big, encoder);

    heongpu::HERingSwitchOperator<S> rs(big, small);
    auto pair = std::make_unique<heongpu::HERingSwitchOperator<S>::SecretPair>(
        rs.make_secret_pair(64, 0xC0FFEEULL));

    Ledger ledger;
    ledger.rs_keygen_ms = BestMs(
        [&]() { rs.generate_keys(keygen, secret, pair->embedded); }, 1);
    std::cout << "[tworing] ring-switch key generation: " << std::fixed
              << std::setprecision(2) << ledger.rs_keygen_ms << " ms"
              << std::endl;

    heongpu::HEKeyGenerator<S> small_keygen(small);
    heongpu::Publickey<S> small_pub(small);
    small_keygen.generate_public_key(small_pub, pair->small);
    heongpu::HEEncryptor<S> small_encryptor(small, small_pub);
    heongpu::HEDecryptor<S> small_decryptor(small, pair->small);
    heongpu::HEEncoder<S> small_encoder(small);

    heongpu::BatchMatrixLayout layout_hi(n_hi, d);
    heongpu::BatchMatrixLayout layout_lo(n_lo, d);
    heongpu::llama::Llama3BatchOperator batch_hi(big, encoder, layout_hi,
                                                 scale);
    heongpu::llama::Llama3RectOperator rect_hi(big, encoder, layout_hi, scale);
    heongpu::llama::Llama3BatchOperator batch_lo(small, small_encoder,
                                                 layout_lo, scale);
    heongpu::llama::Llama3RectOperator rect_lo(small, small_encoder, layout_lo,
                                               scale);

    // Keys. The low ring carries Algorithm 5's full rotation group (the very
    // count that forces the island) plus a relin key for its CCMM; the high
    // ring carries only the d-1 CMT indices and a relin key -- CCMM's whole
    // requirement -- unless the no-island corner asks for the full high set.
    const double before_low_keys = FreeGiB();
    std::vector<int> rect_shifts_lo = rect_lo.rotation_indices();
    heongpu::Galoiskey<S> galois_lo(small, rect_shifts_lo);
    small_keygen.generate_galois_key(galois_lo, pair->small);
    heongpu::Relinkey<S> relin_lo(small);
    small_keygen.generate_relin_key(relin_lo, pair->small);
    cudaDeviceSynchronize();
    ledger.low_keys_gib = before_low_keys - FreeGiB();
    // The island context has ONE special prime (method I), so per key:
    // 16 * L * (L+P) * N bytes. Printed alongside the device delta because
    // the RMM pool can mask the allocation.
    const double low_key_bytes =
        16.0 * shared * (shared + 1) * n_lo;
    std::cout << "[tworing] low-ring keys: "
              << rect_shifts_lo.size() << " Galois indices + relin, "
              << std::setprecision(2) << ledger.low_keys_gib
              << " GiB device delta, "
              << (low_key_bytes * (rect_shifts_lo.size() + 1) /
                  (1024.0 * 1024.0 * 1024.0))
              << " GiB by formula (method I)" << std::endl;

    std::vector<int> ccmm_shifts_hi = batch_hi.product_rotation_indices();
    heongpu::Galoiskey<S> ccmm_galois_hi(big, ccmm_shifts_hi);
    keygen.generate_galois_key(ccmm_galois_hi, secret);
    heongpu::Relinkey<S> relin_hi(big);
    keygen.generate_relin_key(relin_hi, secret);
    std::cout << "[tworing] high-ring CCMM keys: " << ccmm_shifts_hi.size()
              << " Galois indices + relin (all Algorithm 4 ever needs)"
              << std::endl;

    std::mt19937_64 rng(20260810u);
    const double amp = 1.0 / std::sqrt(static_cast<double>(d));

    // ------------------------------------------------------------------
    // LEG 1 -- the 32-head CCMM (the P*V / QK^T-shaped product) per venue.
    // Identical logical work: one 32-block call at the high ring, or two
    // 16-block calls at the low ring on operands that ARRIVED by descent.
    // ------------------------------------------------------------------
    const int groups_lo = heads / blocks_lo; // low calls for all heads
    const int groups_hi = heads / blocks_hi; // high calls for all heads

    std::vector<std::vector<double>> pa(static_cast<std::size_t>(heads)),
        pb(static_cast<std::size_t>(heads));
    for (int h = 0; h < heads; ++h)
    {
        pa[h] = RandomMatrix(static_cast<std::size_t>(d) * d, amp, rng);
        pb[h] = RandomMatrix(static_cast<std::size_t>(d) * d, amp, rng);
    }

    // High-ring native operands.
    std::vector<heongpu::llama::BatchActivation> A_hi, B_hi;
    for (int g = 0; g < groups_hi; ++g)
    {
        std::vector<std::vector<double>> sa(pa.begin() + g * blocks_hi,
                                            pa.begin() + (g + 1) * blocks_hi);
        std::vector<std::vector<double>> sb(pb.begin() + g * blocks_hi,
                                            pb.begin() + (g + 1) * blocks_hi);
        A_hi.push_back(batch_hi.encrypt(sa, d, d, encryptor, scale));
        B_hi.push_back(batch_hi.encrypt(sb, d, d, encryptor, scale));
    }

    std::vector<std::unique_ptr<heongpu::llama::BatchActivation>> C_hi(
        static_cast<std::size_t>(groups_hi));
    ledger.ccmm_high_ms = BestMs(
        [&]()
        {
            for (int g = 0; g < groups_hi; ++g)
                C_hi[g] = std::make_unique<heongpu::llama::BatchActivation>(
                    batch_hi.matmul(A_hi[g], B_hi[g], "tworing.ccmm_hi",
                                    ccmm_galois_hi, relin_hi));
        },
        reps);

    // Low-ring operands, entered through the crossing: encrypt native small
    // columns, compose pairs UP (timed as the up-crossing on this operand
    // set), then switch DOWN (timed) and run the product on what descended.
    std::vector<heongpu::llama::BatchActivation> A_lo, B_lo;
    for (int g = 0; g < groups_lo; ++g)
    {
        std::vector<std::vector<double>> sa(pa.begin() + g * blocks_lo,
                                            pa.begin() + (g + 1) * blocks_lo);
        std::vector<std::vector<double>> sb(pb.begin() + g * blocks_lo,
                                            pb.begin() + (g + 1) * blocks_lo);
        A_lo.push_back(batch_lo.encrypt(sa, d, d, small_encryptor, scale));
        B_lo.push_back(batch_lo.encrypt(sb, d, d, small_encryptor, scale));
    }

    // Round-trip every operand column through compose_up / switch_down so the
    // timed low-ring product runs on descended data, as it would in the flow.
    auto cross_columns =
        [&](std::vector<heongpu::llama::BatchActivation>& acts,
            double& up_ms_total, double& down_ms_total)
    {
        std::vector<heongpu::Ciphertext<S>*> cols;
        for (auto& act : acts)
            for (auto& c : act.column)
                cols.push_back(&c);
        const int big_count = static_cast<int>(cols.size()) / k;

        std::vector<heongpu::Ciphertext<S>> big_cts;
        big_cts.reserve(static_cast<std::size_t>(big_count));
        const auto t_up0 = Clock::now();
        for (int b = 0; b < big_count; ++b)
        {
            std::vector<heongpu::Ciphertext<S>> group;
            group.reserve(static_cast<std::size_t>(k));
            for (int j = 0; j < k; ++j)
                group.push_back(*cols[static_cast<std::size_t>(b) * k + j]);
            big_cts.push_back(rs.compose_up(group, ops));
        }
        cudaDeviceSynchronize();
        const auto t_up1 = Clock::now();
        up_ms_total +=
            std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

        const auto t_dn0 = Clock::now();
        std::size_t idx = 0;
        for (auto& c : big_cts)
        {
            std::vector<heongpu::Ciphertext<S>> parts = rs.switch_down(c, ops);
            for (auto& part : parts)
                *cols[idx++] = std::move(part);
        }
        cudaDeviceSynchronize();
        const auto t_dn1 = Clock::now();
        down_ms_total +=
            std::chrono::duration<double, std::milli>(t_dn1 - t_dn0).count();
    };

    double ccmm_up_ms = 0.0, ccmm_down_ms = 0.0;
    cross_columns(A_lo, ccmm_up_ms, ccmm_down_ms);
    cross_columns(B_lo, ccmm_up_ms, ccmm_down_ms);
    std::cout << "[tworing] ccmm operand crossing: up " << std::setprecision(1)
              << ccmm_up_ms << " ms, down " << ccmm_down_ms << " ms for "
              << 2 * groups_lo * d << " small columns" << std::endl;

    std::vector<std::unique_ptr<heongpu::llama::BatchActivation>> C_lo(
        static_cast<std::size_t>(groups_lo));
    ledger.ccmm_low_ms = BestMs(
        [&]()
        {
            for (int g = 0; g < groups_lo; ++g)
                C_lo[g] = std::make_unique<heongpu::llama::BatchActivation>(
                    batch_lo.matmul(A_lo[g], B_lo[g], "tworing.ccmm_lo",
                                    galois_lo, relin_lo));
        },
        reps);

    if (check)
    {
        double err_hi = 0.0, err_lo = 0.0;
        for (int g = 0; g < groups_hi; ++g)
        {
            const auto got =
                batch_hi.decrypt(*C_hi[g], decryptor,
                                 C_hi[g]->column.front().scale());
            for (int t = 0; t < blocks_hi; ++t)
            {
                const int h = g * blocks_hi + t;
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j)
                    {
                        double want = 0.0;
                        for (int m = 0; m < d; ++m)
                            want += pa[h][static_cast<std::size_t>(i) * d + m] *
                                    pb[h][static_cast<std::size_t>(m) * d + j];
                        err_hi = std::max(
                            err_hi,
                            std::abs(got[t][static_cast<std::size_t>(i) * d +
                                            j] -
                                     want));
                    }
            }
        }
        for (int g = 0; g < groups_lo; ++g)
        {
            const auto got =
                batch_lo.decrypt(*C_lo[g], small_decryptor,
                                 C_lo[g]->column.front().scale());
            for (int t = 0; t < blocks_lo; ++t)
            {
                const int h = g * blocks_lo + t;
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j)
                    {
                        double want = 0.0;
                        for (int m = 0; m < d; ++m)
                            want += pa[h][static_cast<std::size_t>(i) * d + m] *
                                    pb[h][static_cast<std::size_t>(m) * d + j];
                        err_lo = std::max(
                            err_lo,
                            std::abs(got[t][static_cast<std::size_t>(i) * d +
                                            j] -
                                     want));
                    }
            }
        }
        std::cout << "[tworing] ccmm max abs error: high " << std::scientific
                  << std::setprecision(3) << err_hi
                  << ", low (on descended data) " << err_lo
                  << std::defaultfloat << std::endl;
    }

    // ------------------------------------------------------------------
    // LEG 2 -- the W_o-shaped Algorithm 5 (model x model projection).
    // Low ring: operands descend, project runs under the 2047-key group,
    // outputs compose back up. High ring (optional corner): the same
    // projection under the N_high/2-1 key group, no crossing at all.
    // ------------------------------------------------------------------
    const std::vector<double> xw =
        RandomMatrix(static_cast<std::size_t>(d) * model, 0.5, rng);
    const std::vector<double> wo =
        RandomMatrix(static_cast<std::size_t>(model) * model, amp, rng);

    heongpu::llama::RectActivation X_lo =
        rect_lo.encrypt(xw, model, small_encryptor, scale);
    double alg5_up_ms = 0.0, alg5_down_ms = 0.0;
    {
        // Same crossing helper, but on the rect operand set.
        std::vector<heongpu::Ciphertext<S>*> cols;
        for (auto& c : X_lo.column)
            cols.push_back(&c);
        const int big_count = static_cast<int>(cols.size()) / k;
        std::vector<heongpu::Ciphertext<S>> big_cts;
        big_cts.reserve(static_cast<std::size_t>(big_count));
        const auto t_up0 = Clock::now();
        for (int b = 0; b < big_count; ++b)
        {
            std::vector<heongpu::Ciphertext<S>> group;
            for (int j = 0; j < k; ++j)
                group.push_back(*cols[static_cast<std::size_t>(b) * k + j]);
            big_cts.push_back(rs.compose_up(group, ops));
        }
        cudaDeviceSynchronize();
        const auto t_up1 = Clock::now();
        alg5_up_ms =
            std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

        const auto t_dn0 = Clock::now();
        std::size_t idx = 0;
        for (auto& c : big_cts)
        {
            std::vector<heongpu::Ciphertext<S>> parts = rs.switch_down(c, ops);
            for (auto& part : parts)
                *cols[idx++] = std::move(part);
        }
        cudaDeviceSynchronize();
        const auto t_dn1 = Clock::now();
        alg5_down_ms =
            std::chrono::duration<double, std::milli>(t_dn1 - t_dn0).count();
    }
    ledger.down_ms = alg5_down_ms;
    ledger.up_ms = alg5_up_ms;
    std::cout << "[tworing] Alg-5 operand crossing: up " << std::fixed
              << std::setprecision(1) << alg5_up_ms << " ms, down "
              << alg5_down_ms << " ms for " << X_lo.column.size()
              << " small columns" << std::endl;

    std::unique_ptr<heongpu::llama::RectActivation> Y_lo;
    ledger.alg5_low_ms = BestMs(
        [&]()
        {
            Y_lo = std::make_unique<heongpu::llama::RectActivation>(
                rect_lo.project(X_lo, wo, model, model, "tworing.alg5_lo",
                                galois_lo));
        },
        reps);

    // Compose the projection's outputs back up -- the island's exit.
    double alg5_out_up_ms = 0.0;
    {
        const int big_count = static_cast<int>(Y_lo->column.size()) / k;
        const auto t0 = Clock::now();
        for (int b = 0; b < big_count; ++b)
        {
            std::vector<heongpu::Ciphertext<S>> group;
            for (int j = 0; j < k; ++j)
                group.push_back(
                    Y_lo->column[static_cast<std::size_t>(b) * k + j]);
            heongpu::Ciphertext<S> composed = rs.compose_up(group, ops);
            (void) composed;
        }
        cudaDeviceSynchronize();
        const auto t1 = Clock::now();
        alg5_out_up_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "[tworing] Alg-5 output composition: " << alg5_out_up_ms
              << " ms for " << Y_lo->column.size() << " small columns"
              << std::endl;

    if (check)
    {
        const std::vector<double> got =
            rect_lo.decrypt(*Y_lo, small_decryptor,
                            Y_lo->column.front().scale());
        double err = 0.0;
        for (int i = 0; i < d; ++i)
            for (int o = 0; o < model; ++o)
            {
                double want = 0.0;
                for (int c = 0; c < model; ++c)
                    want += xw[static_cast<std::size_t>(i) * model + c] *
                            wo[static_cast<std::size_t>(c) * model + o];
                err = std::max(
                    err, std::abs(got[static_cast<std::size_t>(i) * model + o] -
                                  want));
            }
        std::cout << "[tworing] Alg 5 (low, descended operands) max abs error "
                  << std::scientific << std::setprecision(3) << err
                  << std::defaultfloat << std::endl;
    }

    if (alg5_high)
    {
        const double before = FreeGiB();
        std::vector<int> rect_shifts_hi = rect_hi.rotation_indices();
        heongpu::Galoiskey<S> galois_rect_hi(big, rect_shifts_hi);
        keygen.generate_galois_key(galois_rect_hi, secret);
        cudaDeviceSynchronize();
        ledger.high_rect_keys_gib = before - FreeGiB();
        // Big context: two special primes, method II, ceil(L/2) digits.
        const double hi_key_bytes =
            16.0 * std::ceil(limbs / 2.0) * (limbs + 2) * n_hi;
        std::cout << "[tworing] HIGH-ring Alg-5 keys: "
                  << rect_shifts_hi.size() << " Galois indices, "
                  << std::fixed << std::setprecision(2)
                  << ledger.high_rect_keys_gib << " GiB device delta, "
                  << (hi_key_bytes * (rect_shifts_hi.size() + 1) /
                      (1024.0 * 1024.0 * 1024.0))
                  << " GiB by formula (method II)" << std::endl;

        heongpu::llama::RectActivation X_hi =
            rect_hi.encrypt(xw, model, encryptor, scale);
        std::unique_ptr<heongpu::llama::RectActivation> Y_hi;
        ledger.alg5_high_ms = BestMs(
            [&]()
            {
                Y_hi = std::make_unique<heongpu::llama::RectActivation>(
                    rect_hi.project(X_hi, wo, model, model, "tworing.alg5_hi",
                                    galois_rect_hi));
            },
            reps);
        if (check)
        {
            const std::vector<double> got = rect_hi.decrypt(
                *Y_hi, decryptor, Y_hi->column.front().scale());
            double err = 0.0;
            for (int i = 0; i < d; ++i)
                for (int o = 0; o < model; ++o)
                {
                    double want = 0.0;
                    for (int c = 0; c < model; ++c)
                        want += xw[static_cast<std::size_t>(i) * model + c] *
                                wo[static_cast<std::size_t>(c) * model + o];
                    err = std::max(
                        err,
                        std::abs(got[static_cast<std::size_t>(i) * model + o] -
                                 want));
                }
            std::cout << "[tworing] Alg 5 (high, no island) max abs error "
                      << std::scientific << std::setprecision(3) << err
                      << std::defaultfloat << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // The verdict table. Crossings are charged to the variant that needs
    // them: A descends the CCMM operands too, B only the Alg-5 operands.
    // ------------------------------------------------------------------
    const double a_total = ccmm_down_ms + ledger.ccmm_low_ms +
                           ledger.alg5_low_ms + alg5_out_up_ms;
    const double b_total = ledger.ccmm_high_ms + ledger.down_ms +
                           ledger.alg5_low_ms + alg5_out_up_ms;

    std::cout << std::endl << std::fixed << std::setprecision(1);
    std::cout << "[tworing] ---- legs (ms) ------------------------------"
              << std::endl;
    std::cout << "[tworing] ccmm  " << heads << " heads   high "
              << std::setw(9) << ledger.ccmm_high_ms << "   low "
              << std::setw(9) << ledger.ccmm_low_ms << std::endl;
    std::cout << "[tworing] alg5  " << model << "->" << model << "  high ";
    if (ledger.alg5_high_ms >= 0.0)
        std::cout << std::setw(9) << ledger.alg5_high_ms;
    else
        std::cout << "       --";
    std::cout << "   low " << std::setw(9) << ledger.alg5_low_ms << std::endl;
    std::cout << "[tworing] crossings: ccmm-operand down " << ccmm_down_ms
              << ", alg5-operand down " << ledger.down_ms
              << ", output up " << alg5_out_up_ms << std::endl;
    std::cout << "[tworing] ---- variants (ms) --------------------------"
              << std::endl;
    std::cout << "[tworing] A  down, ccmm low, alg5 low, up : "
              << std::setw(9) << a_total << std::endl;
    std::cout << "[tworing] B  ccmm high, down, alg5 low, up: "
              << std::setw(9) << b_total << std::endl;
    if (ledger.alg5_high_ms >= 0.0)
        std::cout << "[tworing] C  no island, everything high  : "
                  << std::setw(9)
                  << (ledger.ccmm_high_ms + ledger.alg5_high_ms)
                  << "   (but its keys do not scale: see below)" << std::endl;

    // Key-memory feasibility at the REAL chains, from the allocator's own
    // formula (evaluationkey.cu): method II bytes/key =
    // 16 * ceil(L/P) * (L+P) * N; the Alg-5 set is N/2-1 keys + 1.
    auto keyset_gib = [](double n, double l, double p)
    {
        const double digits = std::ceil(l / p);
        const double per = 16.0 * digits * (l + p) * n;
        return per * (n / 2.0) / (1024.0 * 1024.0 * 1024.0);
    };
    std::cout << std::endl;
    std::cout << "[tworing] ---- Alg-5 key set at real chains (computed) --"
              << std::endl;
    std::cout << "[tworing] island, this run's chain      : "
              << ledger.low_keys_gib << " GiB (measured)" << std::endl;
    std::cout << "[tworing] logN 13 on the 40-limb stream : "
              << keyset_gib(8192, 40, 40) << " GiB  <- why the island exists"
              << std::endl;
    std::cout << "[tworing] logN 16, 5-limb island        : "
              << keyset_gib(65536, 5, 5) << " GiB  <- why 2^16 is out"
              << std::endl;
    std::cout << "[tworing] (plus: heads*d % (N/2) != 0 at logN 16, and the "
                 "bootstrap loses 4+ bits above logN 14)"
              << std::endl;

    return 0;
}
