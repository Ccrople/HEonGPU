// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// THE ACCEPTED TWO-RING STRUCTURE, PRICED AT ITS REAL ENVELOPE
// ============================================================
//
// Decision (2026-08-10): the deployable Llama-3 flow keeps the non-linear
// track and the bootstrap at logN 16 -- the only ring whose 128-bit cap
// (1761 bits) holds the v2-class chain -- and runs BOTH Kang products in a
// logN 13 island bracketed by ring switches (Sylph's shape, one ring higher
// than Sylph's island). This target prices every leg of one Stage-3 pass at
// that envelope, with the high ring at the requested decomposition:
//
//   high ring   logN 16, v2 chain {41, 33 x nbase, 32 x 3, 60 x n_sine,
//               56 x 4}, |P| 61-bit specials chosen for dnum = 4
//   island      logN 13, Q = the big chain's bottom {41, 33, 33}, P = the
//               big ring's first special -- the natural shared prefix
//
// Legs measured, each at its natural level (products at the chain bottom,
// SoftMax from the refresh ceiling, bootstrap from l = 1):
//   1. v2 bootstrap at 16 per ciphertext, with precision and depth_after
//   2. SoftMax at 16 (8B degrees: exp 15, 1/x 63, one round), levels live
//   3. crossings 16<->13 at island depth, per ciphertext
//   4. island products: CCMM all 32 heads in one call, Alg 5 model->model
//   5. the row bridge at 16 (the SlotToSinC cost class), 128 columns
//
// The three phases run in scopes so each phase's keys free before the next;
// the printed per-phase key sizes sum to the DEPLOYMENT footprint, which
// must coexist on one card in a real block and is reported as such.
//
// Security envelope note printed at startup: at nbase = 16 the dnum = 4
// decomposition (8 x 61-bit specials) puts log PQ ~ 96 bits OVER the sec128
// cap; the 128-bit-legal floor at this chain is |P| = 6 (dnum 6). The run
// itself uses sec_level_type::none like every profile in this tree.
//
//   HEONGPU_T16_NBASE      usable levels the refresh returns      16
//   HEONGPU_T16_SPECIALS   61-bit special primes (8 -> dnum 4)     8
//   HEONGPU_T16_D          head dim                               128
//   HEONGPU_T16_HEADS      attention heads                        32
//   HEONGPU_T16_MODEL      Alg-5 width (in = out)                4096
//   HEONGPU_T16_LOGN_ISLAND island ring, log2                     13
//   HEONGPU_T16_REPS       timed repetitions per leg               2
//   HEONGPU_T16_CHECK      decrypt and compare vs host             1
//   HEONGPU_T16_BOOT / _SOFTMAX / _ISLAND / _BRIDGE   leg switches, all 1

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
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

    std::vector<double> RandomMatrix(std::size_t count, double amp,
                                     std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-amp, amp);
        std::vector<double> out(count);
        for (double& v : out)
            v = dist(rng);
        return out;
    }
} // namespace

int main()
{
    const int nbase = EnvInt("HEONGPU_T16_NBASE", 16);
    const int specials = EnvInt("HEONGPU_T16_SPECIALS", 8);
    const int d = EnvInt("HEONGPU_T16_D", 128);
    const int heads = EnvInt("HEONGPU_T16_HEADS", 32);
    const int model = EnvInt("HEONGPU_T16_MODEL", 4096);
    const int logn_island = EnvInt("HEONGPU_T16_LOGN_ISLAND", 13);
    const int reps = EnvInt("HEONGPU_T16_REPS", 2);
    const bool check = EnvInt("HEONGPU_T16_CHECK", 1) != 0;
    const bool leg_boot = EnvInt("HEONGPU_T16_BOOT", 1) != 0;
    const bool leg_softmax = EnvInt("HEONGPU_T16_SOFTMAX", 1) != 0;
    const bool leg_island = EnvInt("HEONGPU_T16_ISLAND", 1) != 0;
    const bool leg_bridge = EnvInt("HEONGPU_T16_BRIDGE", 1) != 0;

    const int logn_hi = 16;
    const std::size_t n_hi = std::size_t(1) << logn_hi;
    const int n_is = 1 << logn_island;

    // The v2 chain, exactly as profile_boot_precision.cpp builds it at the
    // shipped sine/CtS stage shape (sine deg 30, double-angle 3).
    const int q0 = 41, p = 33, stc_bits = 32, sine_bits = 60, cts_bits = 56;
    const int sinedeg = 30, dangle = 3, K = 16;
    const int n_sine = int(std::ceil(std::log2(double(sinedeg + 1)))) + dangle;
    const int stc_start = nbase + 3;
    const int em_start = stc_start + n_sine;
    const int cts_start = em_start + 4;
    const int L = 1 + nbase + 3 + n_sine + 4;
    const double scale = std::pow(2.0, p);

    std::vector<int> q_bits;
    q_bits.push_back(q0);
    for (int i = 0; i < nbase; i++)
        q_bits.push_back(p);
    for (int i = 0; i < 3; i++)
        q_bits.push_back(stc_bits);
    for (int i = 0; i < n_sine; i++)
        q_bits.push_back(sine_bits);
    for (int i = 0; i < 4; i++)
        q_bits.push_back(cts_bits);

    int log_q = 0;
    for (int b : q_bits)
        log_q += b;
    const int log_pq = log_q + 61 * specials;
    const int dnum = (L + specials - 1) / specials;

    std::cout << "[t16] high logN 16: v2 chain L = " << L << " (nbase "
              << nbase << ", sine " << n_sine << "), log Q = " << log_q
              << ", |P| = " << specials << " x 61 -> dnum = " << dnum
              << ", log PQ = " << log_pq << std::endl;
    std::cout << "[t16] sec128 cap at N = 65536 is 1761: this decomposition "
              << (log_pq <= 1761 ? "FITS" : "is OVER by")
              << (log_pq <= 1761 ? "" : (" " + std::to_string(log_pq - 1761) +
                                         " bits (legal floor is |P| = 6)"))
              << std::endl;

    heongpu::HEContext<S> big =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    big->set_poly_modulus_degree(n_hi);
    std::vector<Modulus64> q_mods = heongpu::generate_primes(n_hi, q_bits);
    std::vector<Data64> q_vals;
    for (auto& m : q_mods)
        q_vals.push_back(m.value);
    std::vector<Data64> p_vals = heongpu::generate_proper_primes(
        Data64(2) * Data64(n_hi), 61, specials);
    big->set_coeff_modulus_values(q_vals, p_vals);
    big->generate();

    heongpu::HEKeyGenerator<S> keygen(big);
    heongpu::Secretkey<S> secret(big, 192);
    keygen.generate_secret_key_v2(secret);
    heongpu::Publickey<S> pub(big);
    keygen.generate_public_key(pub, secret);
    heongpu::Relinkey<S> relin(big);
    keygen.generate_relin_key(relin, secret);
    heongpu::HEEncoder<S> encoder(big);
    heongpu::HEEncryptor<S> encryptor(big, pub);
    heongpu::HEDecryptor<S> decryptor(big, secret);
    heongpu::llama::Llama3Operator arith(big, encoder, scale);

    const double key_bytes_hi = 16.0 * dnum * (L + specials) * double(n_hi);
    std::cout << "[t16] one high-ring switching key at dnum " << dnum << ": "
              << std::fixed << std::setprecision(1)
              << key_bytes_hi / (1024.0 * 1024.0) << " MiB" << std::endl;

    std::mt19937_64 rng(20260810u);
    double boot_keys_gib = 0.0, bridge_keys_gib = 0.0, island_keys_gib = 0.0;
    double boot_ms = -1.0, softmax_ms = -1.0, bridge_ms = -1.0;
    double down_ms = -1.0, up_ms = -1.0, ccmm_ms = -1.0, alg5_ms = -1.0;
    int softmax_levels = 0;

    // ==================================================================
    // PHASE 1 -- the v2 bootstrap and the SoftMax stretch at logN 16.
    // ==================================================================
    if (leg_boot || leg_softmax)
    {
        try
        {
        heongpu::Secretkey<S> sparse(big, 32);
        keygen.generate_secret_key_v2(sparse);
        heongpu::Switchkey<S> swk_d2s(big);
        keygen.generate_switch_key(swk_d2s, sparse, secret);
        heongpu::Switchkey<S> swk_s2d(big);
        keygen.generate_switch_key(swk_s2d, secret, sparse);

        heongpu::EvalModConfig eval_mod_config(0, em_start, 256.0, K, sinedeg,
                                               dangle, 0, 0.0);
        heongpu::BootstrappingConfigV2 boot_config(
            heongpu::EncodingMatrixConfig(
                heongpu::LinearTransformType::SLOTS_TO_COEFFS, stc_start),
            eval_mod_config,
            heongpu::EncodingMatrixConfig(
                heongpu::LinearTransformType::COEFFS_TO_SLOTS, cts_start));
        arith.generate_bootstrapping_params_v2(scale, boot_config);

        std::vector<int> key_index = arith.bootstrapping_key_indexs();
        boot_keys_gib = key_bytes_hi * (key_index.size() + 1) /
                        (1024.0 * 1024.0 * 1024.0);
        std::cout << "[t16] v2 boot Galois keys: " << key_index.size()
                  << " indices, " << std::setprecision(2) << boot_keys_gib
                  << " GiB by formula" << std::endl;
        heongpu::Galoiskey<S> boot_galois(big, key_index);
        keygen.generate_galois_key(boot_galois, secret);

        const int slots = int(n_hi) / 2;
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        if (leg_boot)
        {
            std::vector<Complex64> msg(slots);
            for (auto& v : msg)
                v = Complex64(dist(rng), dist(rng));
            heongpu::Plaintext<S> P1(big);
            encoder.encode(P1, msg, scale);
            heongpu::Ciphertext<S> C1(big);
            encryptor.encrypt(C1, P1);
            for (int i = 0; i < L - 1; i++)
                arith.mod_drop_inplace(C1);

            std::unique_ptr<heongpu::Ciphertext<S>> out;
            boot_ms = BestMs(
                [&]()
                {
                    out = std::make_unique<heongpu::Ciphertext<S>>(
                        arith.regular_bootstrapping_v2(C1, boot_galois, relin,
                                                       &swk_d2s, &swk_s2d));
                },
                reps);

            heongpu::Plaintext<S> Pr(big);
            decryptor.decrypt(Pr, *out);
            std::vector<Complex64> got;
            encoder.decode(got, Pr);
            heongpu::PrecisionStats prec =
                heongpu::get_precision_stats(msg, got);
            std::cout << "[t16] v2 bootstrap: " << std::setprecision(1)
                      << boot_ms << " ms per ct, depth_after = "
                      << out->depth() << " (window = 1+" << nbase << "), "
                      << "mean prec " << std::setprecision(2)
                      << prec.mean_precision.real << " bits" << std::endl;
        }

        if (leg_softmax)
        {
            try
            {
                // The 8B block's recalibrated fits: exp degree 15 (one
                // round), reciprocal degree 63, no Newton. The block reduces
                // across PARTS (the key axis is the ciphertext axis, zero
                // rotations), so the cost-true unit is the vector overload;
                // two parts keep the reduction shape and halve to per-part.
                heongpu::llama::Llama3Operator::SoftmaxConfig cfg;
                cfg.strided = true;
                cfg.stride = slots;
                cfg.count = 1;
                cfg.bound = 21.0;
                cfg.iterations = 1;
                cfg.exp_degree = 15;
                cfg.inverse_degree = 63;
                cfg.inverse_newton = 0;

                const int nparts = 2;
                std::uniform_real_distribution<double> neg(-cfg.bound, 0.0);
                std::vector<heongpu::Ciphertext<S>> parts;
                for (int j = 0; j < nparts; ++j)
                {
                    std::vector<double> msg(slots);
                    for (auto& v : msg)
                        v = neg(rng);
                    heongpu::Plaintext<S> P1(big);
                    encoder.encode(P1, msg, scale);
                    heongpu::Ciphertext<S> C1(big);
                    encryptor.encrypt(C1, P1);
                    while (L - C1.depth() > 1 + nbase)
                        arith.mod_drop_inplace(C1);
                    parts.push_back(std::move(C1));
                }
                const int depth_in = parts.front().depth();
                const std::vector<std::vector<double>> no_masks;

                std::vector<heongpu::Ciphertext<S>> sm;
                softmax_ms = BestMs(
                    [&]()
                    {
                        std::vector<heongpu::Ciphertext<S>> work = parts;
                        sm = arith.softmax(work, cfg, no_masks, boot_galois,
                                           relin);
                    },
                    reps);
                softmax_ms /= nparts;
                softmax_levels = sm.front().depth() - depth_in;
                std::cout << "[t16] softmax at 16 (exp 15, 1/x 63, "
                          << nparts << "-part, unmasked): "
                          << std::setprecision(1) << softmax_ms
                          << " ms per ct, levels " << softmax_levels
                          << " (from l = " << (L - depth_in) << ")"
                          << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cout << "[t16] softmax leg failed: " << e.what()
                          << std::endl;
            }
        }
        }
        catch (const std::exception& e)
        {
            std::cout << "[t16] phase 1 failed: " << e.what() << std::endl;
        }
    }

    // ==================================================================
    // PHASE 2 -- the logN 13 island: crossings and both Kang products on
    // the shared-prefix chain {41, 33, 33}.
    // ==================================================================
    if (leg_island)
    {
        try
        {
        const int shared = 3;
        heongpu::HEContext<S> small =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        {
            const auto primes = big->get_key_modulus();
            std::vector<Data64> qv, pv;
            for (int i = 0; i < shared; i++)
                qv.push_back(primes[i].value);
            pv.push_back(primes[L].value);
            small->set_poly_modulus_degree(std::size_t(n_is));
            small->set_coeff_modulus_values(qv, pv);
            small->generate();
        }

        heongpu::HERingSwitchOperator<S> rs(big, small);
        auto pair =
            std::make_unique<heongpu::HERingSwitchOperator<S>::SecretPair>(
                rs.make_secret_pair(64, 0xC0FFEEULL));
        rs.generate_keys(keygen, secret, pair->embedded);

        heongpu::HEKeyGenerator<S> small_keygen(small);
        heongpu::Publickey<S> small_pub(small);
        small_keygen.generate_public_key(small_pub, pair->small);
        heongpu::HEEncryptor<S> small_encryptor(small, small_pub);
        heongpu::HEDecryptor<S> small_decryptor(small, pair->small);
        heongpu::HEEncoder<S> small_encoder(small);
        heongpu::HEArithmeticOperator<S> ops_big(big, encoder);

        heongpu::BatchMatrixLayout layout(n_is, d);
        heongpu::llama::Llama3BatchOperator batch_is(small, small_encoder,
                                                     layout, scale);
        heongpu::llama::Llama3RectOperator rect_is(small, small_encoder,
                                                   layout, scale);

        std::vector<int> shifts = rect_is.rotation_indices();
        const double key_bytes_is = 16.0 * shared * (shared + 1) * n_is;
        island_keys_gib =
            key_bytes_is * (shifts.size() + 1) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[t16] island keys: " << shifts.size()
                  << " Galois indices + relin, " << std::setprecision(2)
                  << island_keys_gib << " GiB by formula (method I)"
                  << std::endl;
        heongpu::Galoiskey<S> galois_is(small, shifts);
        small_keygen.generate_galois_key(galois_is, pair->small);
        heongpu::Relinkey<S> relin_is(small);
        small_keygen.generate_relin_key(relin_is, pair->small);

        const int blocks = layout.batch;
        if (heads != blocks)
            std::cout << "[t16] note: " << heads << " heads vs batch "
                      << blocks << " at the island ring" << std::endl;

        const double amp = 1.0 / std::sqrt(double(d));

        // Crossings, on the CCMM operand set (heads x d x d, both operands).
        std::vector<std::vector<double>> pa(heads), pb(heads);
        for (int h = 0; h < heads; ++h)
        {
            pa[h] = RandomMatrix(std::size_t(d) * d, amp, rng);
            pb[h] = RandomMatrix(std::size_t(d) * d, amp, rng);
        }
        heongpu::llama::BatchActivation A =
            batch_is.encrypt(pa, d, d, small_encryptor, scale);
        heongpu::llama::BatchActivation B =
            batch_is.encrypt(pb, d, d, small_encryptor, scale);

        const int k = int(n_hi) / n_is;
        auto cross = [&](std::vector<heongpu::Ciphertext<S>>& cols,
                         double& up_total, double& dn_total)
        {
            const int big_count = int(cols.size()) / k;
            std::vector<heongpu::Ciphertext<S>> big_cts;
            big_cts.reserve(big_count);
            const auto t0 = Clock::now();
            for (int b = 0; b < big_count; ++b)
            {
                std::vector<heongpu::Ciphertext<S>> group(
                    cols.begin() + std::size_t(b) * k,
                    cols.begin() + std::size_t(b + 1) * k);
                big_cts.push_back(rs.compose_up(group, ops_big));
            }
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();
            up_total +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            const auto t2 = Clock::now();
            std::size_t idx = 0;
            for (auto& c : big_cts)
            {
                std::vector<heongpu::Ciphertext<S>> parts =
                    rs.switch_down(c, ops_big);
                for (auto& part : parts)
                    cols[idx++] = std::move(part);
            }
            cudaDeviceSynchronize();
            const auto t3 = Clock::now();
            dn_total +=
                std::chrono::duration<double, std::milli>(t3 - t2).count();
        };

        up_ms = 0.0;
        down_ms = 0.0;
        cross(A.column, up_ms, down_ms);
        cross(B.column, up_ms, down_ms);
        std::cout << "[t16] crossings 16<->" << logn_island << " (k = " << k
                  << "): up " << std::setprecision(2) << up_ms << " ms, down "
                  << down_ms << " ms for " << 2 * d << " island columns ("
                  << (down_ms / (2.0 * d / k)) << " ms per big ct)"
                  << std::endl;

        std::unique_ptr<heongpu::llama::BatchActivation> C;
        ccmm_ms = BestMs(
            [&]()
            {
                C = std::make_unique<heongpu::llama::BatchActivation>(
                    batch_is.matmul(A, B, "t16.ccmm", galois_is, relin_is));
            },
            reps);
        if (check)
        {
            const auto got = batch_is.decrypt(*C, small_decryptor,
                                              C->column.front().scale());
            double err = 0.0;
            for (int t = 0; t < blocks && t < heads; ++t)
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j)
                    {
                        double want = 0.0;
                        for (int m = 0; m < d; ++m)
                            want += pa[t][std::size_t(i) * d + m] *
                                    pb[t][std::size_t(m) * d + j];
                        err = std::max(err,
                                       std::abs(got[t][std::size_t(i) * d +
                                                       j] -
                                                want));
                    }
            std::cout << "[t16] ccmm max abs error " << std::scientific
                      << std::setprecision(3) << err << std::defaultfloat
                      << std::endl;
        }

        const std::vector<double> xw =
            RandomMatrix(std::size_t(d) * model, 0.5, rng);
        const std::vector<double> wo =
            RandomMatrix(std::size_t(model) * model, amp, rng);
        heongpu::llama::RectActivation X =
            rect_is.encrypt(xw, model, small_encryptor, scale);
        std::unique_ptr<heongpu::llama::RectActivation> Y;
        alg5_ms = BestMs(
            [&]()
            {
                Y = std::make_unique<heongpu::llama::RectActivation>(
                    rect_is.project(X, wo, model, model, "t16.alg5",
                                    galois_is));
            },
            reps);
        if (check)
        {
            const std::vector<double> got = rect_is.decrypt(
                *Y, small_decryptor, Y->column.front().scale());
            double err = 0.0;
            for (int i = 0; i < d; ++i)
                for (int o = 0; o < model; ++o)
                {
                    double want = 0.0;
                    for (int c2 = 0; c2 < model; ++c2)
                        want += xw[std::size_t(i) * model + c2] *
                                wo[std::size_t(c2) * model + o];
                    err = std::max(
                        err,
                        std::abs(got[std::size_t(i) * model + o] - want));
                }
            std::cout << "[t16] alg5 max abs error " << std::scientific
                      << std::setprecision(3) << err << std::defaultfloat
                      << std::endl;
        }
        std::cout << "[t16] island products: ccmm (" << heads << " heads, "
                  << (heads + blocks - 1) / blocks << " call) "
                  << std::fixed << std::setprecision(1) << ccmm_ms
                  << " ms, alg5 " << model << "->" << model << " " << alg5_ms
                  << " ms" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "[t16] island leg failed: " << e.what() << std::endl;
        }
    }

    // ==================================================================
    // PHASE 3 -- the RETARGETED crossing at logN 16 at dnum 4: the row
    // bridge at the WIDE layout (N_H, d*k), which test_ckks_tworing_bridge
    // proves is the SlotToSinC map onto the descent contract. Keys are the
    // BSGS subset -- n1 + n2 - 2 shifts, not d*k - 1.
    // ==================================================================
    double bridge_up_ms = -1.0;
    if (leg_bridge)
    {
        try
        {
            const int k = int(n_hi) / n_is;
            const int d_wide = d * k;
            heongpu::BatchMatrixLayout layout_wide(int(n_hi), d_wide);
            heongpu::llama::Llama3BatchOperator wide(big, encoder,
                                                     layout_wide, scale);
            int n1 = 1;
            while (n1 * n1 * 2 <= d_wide)
                n1 <<= 1;
            wide.set_bridge_baby_steps(n1);
            const int n2 = d_wide / n1;
            const int step = layout_wide.k / 2;
            std::vector<int> shifts;
            for (int j = 1; j < n1; ++j)
                shifts.push_back(j * step);
            for (int i = 1; i < n2; ++i)
                shifts.push_back(i * n1 * step);
            bridge_keys_gib = key_bytes_hi * (shifts.size() + 1) /
                              (1024.0 * 1024.0 * 1024.0);
            std::cout << "[t16] wide crossing keys (BSGS subset): "
                      << shifts.size() << " of " << d_wide - 1
                      << " indices, " << std::setprecision(2)
                      << bridge_keys_gib << " GiB by formula at dnum "
                      << dnum << std::endl;
            heongpu::Galoiskey<S> bridge_galois(big, shifts);
            keygen.generate_galois_key(bridge_galois, secret);

            // The stage operand at the 8B shape is d/k big ciphertexts (128
            // island columns interleaved 8 apiece). Values are irrelevant to
            // the cost; the layout law is pinned by the test.
            const int nct = d / k;
            heongpu::llama::BatchActivation Bh;
            Bh.rows = d_wide;
            std::uniform_real_distribution<double> dist(-0.5, 0.5);
            for (int c = 0; c < nct; ++c)
            {
                std::vector<double> msg(int(n_hi) / 2);
                for (auto& v : msg)
                    v = dist(rng);
                heongpu::Plaintext<S> P1(big);
                encoder.encode(P1, msg, scale);
                heongpu::Ciphertext<S> C1(big);
                encryptor.encrypt(C1, P1);
                while (L - C1.depth() > 4)
                    arith.mod_drop_inplace(C1);
                Bh.column.push_back(std::move(C1));
            }

            std::vector<heongpu::Ciphertext<S>> slots_out;
            bridge_ms = BestMs(
                [&]()
                {
                    heongpu::llama::BatchActivation work = Bh;
                    slots_out = wide.to_slots(work, bridge_galois);
                },
                reps);
            heongpu::llama::BatchActivation back;
            bridge_up_ms = BestMs(
                [&]()
                {
                    std::vector<heongpu::Ciphertext<S>> work = slots_out;
                    back = wide.from_slots(work, d_wide, bridge_galois);
                },
                reps);
            std::cout << "[t16] wide crossing at 16 (l = 4, " << nct
                      << " big cts): to_slots " << std::fixed
                      << std::setprecision(1) << bridge_ms << " ms, "
                      << "from_slots " << bridge_up_ms << " ms ("
                      << (bridge_ms + bridge_up_ms) / nct
                      << " ms per big ct both ways)" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "[t16] bridge leg failed: " << e.what() << std::endl;
        }
    }

    // ==================================================================
    // The ledger.
    // ==================================================================
    std::cout << std::endl << std::fixed << std::setprecision(1);
    std::cout << "[t16] ---- Stage-3 ledger at the accepted structure ----"
              << std::endl;
    std::cout << "[t16] softmax(16) per ct        : " << softmax_ms << " ms"
              << std::endl;
    const double wide_cts = double(d) * n_is / double(n_hi);
    std::cout << "[t16] wide crossing per big ct  : "
              << (bridge_ms >= 0 && bridge_up_ms >= 0
                      ? (bridge_ms + bridge_up_ms) / wide_cts
                      : -1.0)
              << " ms (to_slots + from_slots)" << std::endl;
    std::cout << "[t16] crossing down per big ct  : "
              << (down_ms >= 0 ? down_ms / (2.0 * d * n_is / double(n_hi))
                               : -1.0)
              << " ms" << std::endl;
    std::cout << "[t16] ccmm all heads (island)   : " << ccmm_ms << " ms"
              << std::endl;
    std::cout << "[t16] alg5 model->model (island): " << alg5_ms << " ms"
              << std::endl;
    std::cout << "[t16] v2 bootstrap(16) per ct   : " << boot_ms << " ms"
              << std::endl;
    std::cout << "[t16] key footprints (GiB): boot " << std::setprecision(2)
              << boot_keys_gib << ", bridge " << bridge_keys_gib
              << ", island " << island_keys_gib << ", DEPLOYMENT SUM "
              << (boot_keys_gib + bridge_keys_gib + island_keys_gib)
              << std::endl;
    return 0;
}
