// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// ONE TRANSFORMER BLOCK ON THE TWO-RING STRUCTURE
// ===============================================
//
// The accepted envelope (LLAMA3_8B_LAYER_FLOW.md §13): activations live as
// island-ring RECT/BATCH ciphertexts and every product -- Algorithm 5
// projections, the CCMM pair, the crossings between the island encodings --
// runs at the island (logN 13, the natural shared prefix of the big chain).
// Every non-linearity and every refresh runs at the big ring (logN 16, v2
// chain): k = N_H/N_s consecutive island SinC columns compose_up into ONE
// big ciphertext whose coefficient layout is the wide matrix encryption at
// block size d*k (proven, test_ckks_tworing_bridge.cpp), the v2 bootstrap
// refreshes it 8-16x packed, the WIDE row bridge (BatchMatrixLayout(N_H,
// d*k), BSGS-subset keys) reads it into slots, the non-linear runs there,
// and the mirror path descends again.
//
// The wide slot law that everything below indexes by:
//
//   slot b + step_H*(j' + k*u) of big ct g  =  M_b[u][g*k + j']
//
// with step_H = k_H/2 = (N_H/(d*k))/2, b the matrix-block (= head or
// channel-block) index, u the token, and g*k + j' the island column. For a
// RECT activation (channel c = b*d + col, col = g*k + j') this means
// token t of channel c sits at slot b + step_H*(j' + k*t) of slot-ct g:
// one aligned run of step_H*k slots holds ALL channels of one token, and
// the in-ct key/column axis rides at stride step_H.
//
// Stages (HEONGPU_TB_STAGE):
//   softmax   unit-validate the wide SoftMax config + causal masks against
//             a host reference on DIRECTLY encoded wide slot vectors -- no
//             island, no bootstrap, isolates the index law.
//   rmsnorm   same trick for the wide RMSNorm blocked reduction + gains.
//   attention full two-ring attention sublayer with v2 refreshes.
//   ffn       full two-ring feed-forward sublayer with v2 refreshes.
//   block     the whole block: norms, attention, residuals, ffn.
//
// Correctness first: random weights, host reference computed in-driver,
// fit ranges taken from the plaintext data. The mini pair (13<->12) runs
// every stage in minutes; the real pair (16<->13) is the measurement.
//
//   HEONGPU_TB_LOGN_HI      big ring log2                        16
//   HEONGPU_TB_LOGN_ISLAND  island ring log2                     13
//   HEONGPU_TB_D            tokens = head dim                    128
//   HEONGPU_TB_SHARED       island Q limbs (shared prefix)       4
//   HEONGPU_TB_HEADS        attention heads (0 -> island batch)  0
//   HEONGPU_TB_KV_HEADS     KV heads for GQA (0 -> heads/4)      0
//   HEONGPU_TB_MODEL        channels (0 -> island N/2)           0
//   HEONGPU_TB_HIDDEN       FFN hidden (0 -> 2*model)            0
//   HEONGPU_TB_NBASE        v2 usable window                     16
//   HEONGPU_TB_SPECIALS     61-bit specials at the big ring      8
//   HEONGPU_TB_STAGE        see above                            block
//   HEONGPU_TB_CHECK        decrypt and compare                  1
//   HEONGPU_TB_EXP_DEGREE / _INV_DEGREE / _INV_NEWTON / _SILU_DEGREE /
//   HEONGPU_TB_NORM_DEGREE / _NORM_NEWTON     fit degrees, 8B defaults

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;
using Clock = std::chrono::steady_clock;
using Ct = heongpu::Ciphertext<S>;

namespace
{
    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }

    std::string EnvStr(const char* name, const char* fallback)
    {
        const char* raw = std::getenv(name);
        return (raw == nullptr || *raw == '\0') ? fallback : raw;
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

    // Wall-clock ledger: every seam charges its cost to a named leg.
    struct Ledger
    {
        std::map<std::string, double> ms;
        std::map<std::string, int> calls;

        template <typename F> void charge(const std::string& leg, F&& f)
        {
            cudaDeviceSynchronize();
            const auto t0 = Clock::now();
            f();
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();
            ms[leg] +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            calls[leg] += 1;
        }

        void print(const char* tag) const
        {
            double total = 0.0;
            for (const auto& kv : ms)
                total += kv.second;
            std::cout << "[tb] ---- " << tag << " ledger ----" << std::endl;
            for (const auto& kv : ms)
                std::cout << "[tb]   " << std::left << std::setw(24)
                          << kv.first << std::right << std::fixed
                          << std::setprecision(1) << std::setw(10) << kv.second
                          << " ms  x" << calls.at(kv.first) << std::endl;
            std::cout << "[tb]   " << std::left << std::setw(24) << "TOTAL"
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(10) << total << " ms" << std::endl;
        }
    };

    // ------------------------------------------------------------------
    // Host references
    // ------------------------------------------------------------------

    // x: d x C row-major (token-major). w: C x O column-consumed as
    // out[t][o] = sum_c x[t][c] * w[c][o].
    std::vector<double> HostMatmul(const std::vector<double>& x, int d, int C,
                                   const std::vector<double>& w, int O)
    {
        std::vector<double> out(std::size_t(d) * O, 0.0);
        for (int t = 0; t < d; ++t)
            for (int c = 0; c < C; ++c)
            {
                const double xv = x[std::size_t(t) * C + c];
                if (xv == 0.0)
                    continue;
                const double* wr = &w[std::size_t(c) * O];
                double* orow = &out[std::size_t(t) * O];
                for (int o = 0; o < O; ++o)
                    orow[o] += xv * wr[o];
            }
        return out;
    }

    std::vector<double> HostRmsNorm(const std::vector<double>& x, int d, int C,
                                    const std::vector<double>& gain,
                                    double eps)
    {
        std::vector<double> out(x.size());
        for (int t = 0; t < d; ++t)
        {
            double s = 0.0;
            for (int c = 0; c < C; ++c)
            {
                const double v = x[std::size_t(t) * C + c];
                s += v * v;
            }
            const double inv = 1.0 / std::sqrt(s / C + eps);
            for (int c = 0; c < C; ++c)
                out[std::size_t(t) * C + c] =
                    x[std::size_t(t) * C + c] * inv * gain[c];
        }
        return out;
    }

    // Masked softmax over the key axis: exponentials are multiplied by the
    // 0/1 mask BEFORE the denominator, which is how the circuit's masks
    // ride on the exp output.
    std::vector<double> HostMaskedSoftmax(const std::vector<double>& scores,
                                          int rows, int keys, bool causal)
    {
        std::vector<double> out(scores.size());
        for (int u = 0; u < rows; ++u)
        {
            double denom = 0.0;
            for (int j = 0; j < keys; ++j)
            {
                const bool live = !causal || j <= u;
                const double e =
                    live ? std::exp(scores[std::size_t(u) * keys + j]) : 0.0;
                out[std::size_t(u) * keys + j] = e;
                denom += e;
            }
            for (int j = 0; j < keys; ++j)
                out[std::size_t(u) * keys + j] /= denom;
        }
        return out;
    }

    double HostSilu(double x) { return x / (1.0 + std::exp(-x)); }

    double MaxAbsDiff(const std::vector<double>& a,
                      const std::vector<double>& b)
    {
        double m = 0.0;
        for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
            m = std::max(m, std::abs(a[i] - b[i]));
        return m;
    }
} // namespace

// ----------------------------------------------------------------------
// The two-ring harness: both contexts, all keys, and the four seams.
// ----------------------------------------------------------------------
struct TwoRing
{
    // Shapes.
    int logn_hi, logn_is, d, k, d_wide, step_h, shared;
    std::size_t n_hi;
    int n_is;
    double scale;
    int L_hi; // big chain length
    bool boot_enabled;

    // Big ring.
    heongpu::HEContext<S> big{nullptr};
    std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
    std::unique_ptr<heongpu::Secretkey<S>> secret;
    std::unique_ptr<heongpu::Publickey<S>> pub;
    std::unique_ptr<heongpu::Relinkey<S>> relin_hi;
    std::unique_ptr<heongpu::HEEncoder<S>> encoder_hi;
    std::unique_ptr<heongpu::HEEncryptor<S>> encryptor_hi;
    std::unique_ptr<heongpu::HEDecryptor<S>> decryptor_hi;
    std::unique_ptr<heongpu::llama::Llama3Operator> arith_hi;
    std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops_big;
    std::unique_ptr<heongpu::llama::Llama3BatchOperator> wide;
    std::unique_ptr<heongpu::llama::Llama3RectOperator> wide_rect;
    std::unique_ptr<heongpu::Galoiskey<S>> galois_hi; // bridge subset + extras
    std::unique_ptr<heongpu::Galoiskey<S>> boot_galois;
    std::unique_ptr<heongpu::Secretkey<S>> sparse;
    std::unique_ptr<heongpu::Switchkey<S>> swk_d2s, swk_s2d;

    // Island ring.
    heongpu::HEContext<S> island{nullptr};
    std::unique_ptr<heongpu::HERingSwitchOperator<S>> rs;
    std::unique_ptr<heongpu::HERingSwitchOperator<S>::SecretPair> pair;
    std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen_is;
    std::unique_ptr<heongpu::Publickey<S>> pub_is;
    std::unique_ptr<heongpu::HEEncoder<S>> encoder_is;
    std::unique_ptr<heongpu::HEEncryptor<S>> encryptor_is;
    std::unique_ptr<heongpu::HEDecryptor<S>> decryptor_is;
    std::unique_ptr<heongpu::llama::Llama3RectOperator> rect_is;
    std::unique_ptr<heongpu::llama::Llama3BatchOperator> batch_is;
    std::unique_ptr<heongpu::Galoiskey<S>> galois_is;
    std::unique_ptr<heongpu::Relinkey<S>> relin_is;

    Ledger* ledger = nullptr;

    // The v2 chain of profile_boot_precision / profile_tworing_stage16.
    void build_big(int nbase, int specials, bool with_boot)
    {
        boot_enabled = with_boot;
        const int q0 = 41, p = 33, stc_bits = 32, sine_bits = 60,
                  cts_bits = 56;
        const int sinedeg = 30, dangle = 3, K = 16;
        const int n_sine =
            int(std::ceil(std::log2(double(sinedeg + 1)))) + dangle;
        const int stc_start = nbase + 3;
        const int em_start = stc_start + n_sine;
        const int cts_start = em_start + 4;
        L_hi = 1 + nbase + 3 + n_sine + 4;
        scale = std::pow(2.0, p);

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

        big = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        big->set_poly_modulus_degree(n_hi);
        std::vector<Modulus64> q_mods = heongpu::generate_primes(n_hi, q_bits);
        std::vector<Data64> q_vals;
        for (auto& m : q_mods)
            q_vals.push_back(m.value);
        std::vector<Data64> p_vals = heongpu::generate_proper_primes(
            Data64(2) * Data64(n_hi), 61, specials);
        big->set_coeff_modulus_values(q_vals, p_vals);
        big->generate();

        keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(big);
        secret = std::make_unique<heongpu::Secretkey<S>>(big, 192);
        keygen->generate_secret_key_v2(*secret);
        pub = std::make_unique<heongpu::Publickey<S>>(big);
        keygen->generate_public_key(*pub, *secret);
        relin_hi = std::make_unique<heongpu::Relinkey<S>>(big);
        keygen->generate_relin_key(*relin_hi, *secret);
        encoder_hi = std::make_unique<heongpu::HEEncoder<S>>(big);
        encryptor_hi = std::make_unique<heongpu::HEEncryptor<S>>(big, *pub);
        decryptor_hi = std::make_unique<heongpu::HEDecryptor<S>>(big, *secret);
        arith_hi = std::make_unique<heongpu::llama::Llama3Operator>(
            big, *encoder_hi, scale);
        ops_big = std::make_unique<heongpu::HEArithmeticOperator<S>>(
            big, *encoder_hi);

        if (with_boot)
        {
            sparse = std::make_unique<heongpu::Secretkey<S>>(big, 32);
            keygen->generate_secret_key_v2(*sparse);
            swk_d2s = std::make_unique<heongpu::Switchkey<S>>(big);
            keygen->generate_switch_key(*swk_d2s, *sparse, *secret);
            swk_s2d = std::make_unique<heongpu::Switchkey<S>>(big);
            keygen->generate_switch_key(*swk_s2d, *secret, *sparse);

            heongpu::EvalModConfig eval_mod_config(0, em_start, 256.0, K,
                                                   sinedeg, dangle, 0, 0.0);
            heongpu::BootstrappingConfigV2 boot_config(
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::SLOTS_TO_COEFFS, stc_start),
                eval_mod_config,
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::COEFFS_TO_SLOTS, cts_start));
            arith_hi->generate_bootstrapping_params_v2(scale, boot_config);
            std::vector<int> key_index = arith_hi->bootstrapping_key_indexs();
            boot_galois =
                std::make_unique<heongpu::Galoiskey<S>>(big, key_index);
            keygen->generate_galois_key(*boot_galois, *secret);
        }
    }

    // The wide bridge + the big-ring rotation set the non-linears need:
    // BSGS subset {j*step_h} u {i*n1*step_h}, the strided key-axis comb
    // (+-step_h*2^i up to k), and the blocked-reduction span for RMSNorm
    // (+-2^i up to step_h*k).
    void build_wide()
    {
        heongpu::BatchMatrixLayout layout_wide(int(n_hi), d_wide);
        step_h = layout_wide.k / 2;
        wide = std::make_unique<heongpu::llama::Llama3BatchOperator>(
            big, *encoder_hi, layout_wide, scale);
        // The wide-layout rect operator supplies block_map: a RECT stream's
        // coefficients are raw (not interpolated), so its crossing is the
        // row bridge PLUS the block transform, exactly as on the island.
        wide_rect = std::make_unique<heongpu::llama::Llama3RectOperator>(
            big, *encoder_hi, layout_wide, scale);
        int n1 = 1;
        while (n1 * n1 * 2 <= d_wide)
            n1 <<= 1;
        wide->set_bridge_baby_steps(n1);
        const int n2 = d_wide / n1;
        const int slots = int(n_hi) / 2;

        std::vector<int> shifts;
        for (int j = 1; j < n1; ++j)
            shifts.push_back(j * step_h);
        for (int i = 1; i < n2; ++i)
            shifts.push_back(i * n1 * step_h);
        // Key-axis comb for the SoftMax denominator: rotate by step_h*2^i
        // forward for the sliding sum, backward to fan the total out.
        for (int s = step_h; s < step_h * k; s <<= 1)
        {
            shifts.push_back(s);
            shifts.push_back(slots - s);
        }
        // Blocked reduction (sum_blocked) for RMSNorm at span step_h*k.
        for (int s = 1; s < step_h * k; s <<= 1)
        {
            shifts.push_back(s);
            shifts.push_back(slots - s);
        }
        // The wide block map walks the whole batch axis: +-1..+-(step_h-1).
        for (int s = 1; s < step_h; ++s)
        {
            shifts.push_back(s);
            shifts.push_back(slots - s);
        }
        std::sort(shifts.begin(), shifts.end());
        shifts.erase(std::unique(shifts.begin(), shifts.end()), shifts.end());
        shifts.erase(std::remove(shifts.begin(), shifts.end(), 0),
                     shifts.end());

        galois_hi = std::make_unique<heongpu::Galoiskey<S>>(big, shifts);
        keygen->generate_galois_key(*galois_hi, *secret);
        std::cout << "[tb] big-ring rotation set: " << shifts.size()
                  << " indices (bridge n1 = " << n1 << ")" << std::endl;
    }

    // RECT-stream crossings at the big ring: bridge + block map, 2 levels
    // per direction. BATCH (matrix-encryption) data uses wide_to_slots /
    // wide_from_slots alone.
    std::vector<Ct> rect_cross_up(std::vector<Ct>& big_cts)
    {
        std::vector<Ct> slots = wide_to_slots(big_cts);
        ledger->charge("wide.block_map", [&]() {
            wide_rect->block_map(slots, true, "wide.block_inverse",
                                 *galois_hi);
        });
        return slots;
    }

    std::vector<Ct> rect_cross_down(std::vector<Ct>& slot_cts)
    {
        ledger->charge("wide.block_map", [&]() {
            wide_rect->block_map(slot_cts, false, "wide.block_forward",
                                 *galois_hi);
        });
        return wide_from_slots(slot_cts);
    }

    void build_island()
    {
        island = heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        {
            const auto primes = big->get_key_modulus();
            std::vector<Data64> qv, pv;
            for (int i = 0; i < shared; i++)
                qv.push_back(primes[i].value);
            pv.push_back(primes[L_hi].value);
            island->set_poly_modulus_degree(std::size_t(n_is));
            island->set_coeff_modulus_values(qv, pv);
            island->generate();
        }
        rs = std::make_unique<heongpu::HERingSwitchOperator<S>>(big, island);
        pair =
            std::make_unique<heongpu::HERingSwitchOperator<S>::SecretPair>(
                rs->make_secret_pair(64, 0xC0FFEEULL));
        rs->generate_keys(*keygen, *secret, pair->embedded);

        keygen_is = std::make_unique<heongpu::HEKeyGenerator<S>>(island);
        pub_is = std::make_unique<heongpu::Publickey<S>>(island);
        keygen_is->generate_public_key(*pub_is, pair->small);
        encoder_is = std::make_unique<heongpu::HEEncoder<S>>(island);
        encryptor_is =
            std::make_unique<heongpu::HEEncryptor<S>>(island, *pub_is);
        decryptor_is =
            std::make_unique<heongpu::HEDecryptor<S>>(island, pair->small);

        heongpu::BatchMatrixLayout layout(n_is, d);
        rect_is = std::make_unique<heongpu::llama::Llama3RectOperator>(
            island, *encoder_is, layout, scale);
        rect_is->set_fused_crossings(true);
        batch_is = std::make_unique<heongpu::llama::Llama3BatchOperator>(
            island, *encoder_is, layout, scale);

        std::vector<int> shifts = rect_is->rotation_indices();
        galois_is = std::make_unique<heongpu::Galoiskey<S>>(island, shifts);
        keygen_is->generate_galois_key(*galois_is, pair->small);
        relin_is = std::make_unique<heongpu::Relinkey<S>>(island);
        keygen_is->generate_relin_key(*relin_is, pair->small);
        std::cout << "[tb] island keys: " << shifts.size()
                  << " Galois indices + relin (method I, " << shared
                  << " limbs)" << std::endl;
    }

    // ---- Seam 1: island columns -> big SinC ciphertexts. --------------
    std::vector<Ct> ascend(std::vector<Ct>& cols)
    {
        std::vector<Ct> big_cts;
        ledger->charge("cross.up", [&]() {
            const int big_count = int(cols.size()) / k;
            big_cts.reserve(big_count);
            for (int b = 0; b < big_count; ++b)
            {
                std::vector<Ct> group(cols.begin() + std::size_t(b) * k,
                                      cols.begin() + std::size_t(b + 1) * k);
                big_cts.push_back(rs->compose_up(group, *ops_big));
            }
        });
        return big_cts;
    }

    // ---- Seam 2: big SinC ciphertexts -> island columns. --------------
    std::vector<Ct> descend(std::vector<Ct>& big_cts, int target_l)
    {
        std::vector<Ct> cols;
        ledger->charge("cross.down", [&]() {
            for (auto& c : big_cts)
            {
                while (L_hi - c.depth() > target_l)
                    arith_hi->mod_drop_inplace(c);
                std::vector<Ct> parts = rs->switch_down(c, *ops_big);
                for (auto& part : parts)
                    cols.push_back(std::move(part));
            }
        });
        return cols;
    }

    // ---- Seam 3: refresh big SinC ciphertexts (drop to l=1, v2 boot). --
    void refresh(std::vector<Ct>& big_cts, const char* name)
    {
        if (!boot_enabled)
            return;
        ledger->charge(std::string("boot.") + name, [&]() {
            for (auto& c : big_cts)
            {
                while (L_hi - c.depth() > 1)
                    arith_hi->mod_drop_inplace(c);
                c = arith_hi->regular_bootstrapping_v2(
                    c, *boot_galois, *relin_hi, swk_d2s.get(), swk_s2d.get());
            }
        });
    }

    // ---- Seam 4: the wide row bridge. ----------------------------------
    std::vector<Ct> wide_to_slots(std::vector<Ct>& big_cts)
    {
        std::vector<Ct> out;
        ledger->charge("wide.to_slots", [&]() {
            heongpu::llama::BatchActivation act;
            act.rows = d_wide;
            act.column = std::move(big_cts);
            out = wide->to_slots(act, *galois_hi);
        });
        return out;
    }

    std::vector<Ct> wide_from_slots(std::vector<Ct>& slot_cts)
    {
        std::vector<Ct> out;
        ledger->charge("wide.from_slots", [&]() {
            heongpu::llama::BatchActivation back =
                wide->from_slots(slot_cts, d_wide, *galois_hi);
            out = std::move(back.column);
        });
        return out;
    }

    // The wide slot index of (block b, token u, in-group column j').
    inline std::size_t wide_slot(int b, int u, int jp) const
    {
        return std::size_t(b) + std::size_t(step_h) * (jp + std::size_t(k) * u);
    }
};

// ----------------------------------------------------------------------
// Wide-layout constructions: the plaintext vectors the non-linears need.
// ----------------------------------------------------------------------

// Causal mask for slot-ct g: key = g*k + j' visible to query u iff
// key <= u, carrying the row-count equalizer sqrt(d/(u+1)) the island
// mask carries (llama3_batch.cu causal_column_mask): constant along the
// key axis, so it cancels in the normalisation while keeping every row's
// denominator at full-row scale. Identical for every block b.
static std::vector<double> WideCausalMask(const TwoRing& tr, int g, int d)
{
    std::vector<double> m(tr.n_hi / 2, 0.0);
    for (int u = 0; u < d; ++u)
    {
        const double w = std::sqrt(double(d) / double(u + 1));
        for (int jp = 0; jp < tr.k; ++jp)
        {
            const int key = g * tr.k + jp;
            if (key <= u)
                for (int b = 0; b < tr.step_h; ++b)
                    m[tr.wide_slot(b, u, jp)] = w;
        }
    }
    return m;
}

// The learned gain vector for slot-ct g of a RECT activation: channel
// c = b*d + (g*k + j') at every token u.
static std::vector<double> WideGainVector(const TwoRing& tr, int g, int d,
                                          const std::vector<double>& gain)
{
    std::vector<double> w(tr.n_hi / 2, 0.0);
    for (int b = 0; b < tr.step_h; ++b)
        for (int jp = 0; jp < tr.k; ++jp)
        {
            const double gv = gain[std::size_t(b) * d + g * tr.k + jp];
            for (int u = 0; u < d; ++u)
                w[tr.wide_slot(b, u, jp)] = gv;
        }
    return w;
}

// Read a decrypted wide slot vector back into matrix entries
// M_b[u][g*k + j'] for checking.
static double WideSlotError(const TwoRing& tr,
                            const std::vector<double>& slots, int g, int d,
                            int blocks,
                            const std::function<double(int, int, int)>& want)
{
    double err = 0.0;
    for (int b = 0; b < blocks; ++b)
        for (int u = 0; u < d; ++u)
            for (int jp = 0; jp < tr.k; ++jp)
                err = std::max(err, std::abs(slots[tr.wide_slot(b, u, jp)] -
                                             want(b, u, g * tr.k + jp)));
    return err;
}

int main()
{
    const int logn_hi = EnvInt("HEONGPU_TB_LOGN_HI", 16);
    const int logn_is = EnvInt("HEONGPU_TB_LOGN_ISLAND", 13);
    const int d = EnvInt("HEONGPU_TB_D", 128);
    // shared = 4: every boot input must first be brought to the EXACT
    // nominal scale (the v2 boot silently corrupts drifted-scale inputs,
    // measured: 2^33*1.00036 -> garbage, 2^33 exact -> clean), and the
    // match_scale that does it costs the level a 3-limb island cannot
    // spare in the attention tail.
    const int shared = EnvInt("HEONGPU_TB_SHARED", 4);
    const int nbase = EnvInt("HEONGPU_TB_NBASE", 16);
    const int specials = EnvInt("HEONGPU_TB_SPECIALS", 8);
    const bool check = EnvInt("HEONGPU_TB_CHECK", 1) != 0;
    const bool dbg = EnvInt("HEONGPU_TB_DEBUG", 0) != 0;
    const std::string stage = EnvStr("HEONGPU_TB_STAGE", "block");

    TwoRing tr;
    tr.logn_hi = logn_hi;
    tr.logn_is = logn_is;
    tr.d = d;
    tr.shared = shared;
    tr.n_hi = std::size_t(1) << logn_hi;
    tr.n_is = 1 << logn_is;
    tr.k = int(tr.n_hi) / tr.n_is;
    tr.d_wide = d * tr.k;
    tr.step_h = (int(tr.n_hi) / tr.d_wide) / 2;

    const int step_is = (tr.n_is / d) / 2;
    const int heads = EnvInt("HEONGPU_TB_HEADS", step_is);
    const int kv_heads = EnvInt("HEONGPU_TB_KV_HEADS", std::max(1, heads / 4));
    const int half_is = tr.n_is / 2;
    const int model = EnvInt("HEONGPU_TB_MODEL", half_is);
    const int hidden = EnvInt("HEONGPU_TB_HIDDEN", 2 * model);

    const bool unit_stage = (stage == "softmax" || stage == "rmsnorm");
    const bool with_boot = !unit_stage;

    std::cout << "[tb] two-ring block: high 2^" << logn_hi << " / island 2^"
              << logn_is << " (k = " << tr.k << "), d = " << d
              << ", d_wide = " << tr.d_wide << ", step_h = " << tr.step_h
              << ", model = " << model << ", heads = " << heads << " (kv "
              << kv_heads << "), hidden = " << hidden << ", shared = "
              << shared << ", stage = " << stage << std::endl;

    Ledger ledger;
    tr.ledger = &ledger;

    tr.build_big(nbase, specials, with_boot);
    tr.build_wide();
    if (!unit_stage)
        tr.build_island();

    std::mt19937_64 rng(20260810u);
    const int slots_hi = int(tr.n_hi) / 2;

    // ------------------------------------------------------------------
    // SoftMax at the wide layout, built from the public primitives.
    //
    // Llama3Operator::softmax cannot reduce the wide key axis: its strided
    // reduction demands stride * count == slot_count, and here the key
    // axis is a MIDDLE axis -- j' in [0,k) at stride step_h under the
    // token axis. So this is the same exp -> mask -> square -> denominator
    // -> 1/x -> product pipeline, with the in-ciphertext half of the
    // denominator done as a comb: log2(k) sliding rotations, one mask at
    // j' = 0 (which carries the reciprocal fit's domain scale, the same
    // fold the library rides on its own mask), log2(k) fan-out rotations.
    // With the exp domain pre-folded into the query weight the whole thing
    // is 14 levels, the accepted schedule's stretch.
    // ------------------------------------------------------------------
    // Decrypt-at-seam debug: max |value| of a big-ring ciphertext.
    auto dbg_slot = [&](Ct& c, const char* label) {
        if (!dbg)
            return;
        heongpu::Plaintext<S> Pr(tr.big);
        tr.decryptor_hi->decrypt(Pr, c);
        std::vector<double> got;
        tr.encoder_hi->decode(got, Pr);
        double m = 0.0;
        for (double v : got)
            m = std::max(m, std::abs(v));
        std::cout << "[tb.dbg] " << label << ": max|v| = " << std::scientific
                  << std::setprecision(3) << m << std::defaultfloat
                  << " (l = " << tr.L_hi - c.depth() << ")" << std::endl;
    };

    // Decrypt every slot ct and compare against a host law.
    auto dbg_slots_vs = [&](std::vector<Ct>& cts, const char* label,
                            const std::function<double(int, int, int)>& want) {
        if (!dbg)
            return;
        double e = 0.0;
        for (int g = 0; g < int(cts.size()); ++g)
        {
            heongpu::Plaintext<S> Pr(tr.big);
            tr.decryptor_hi->decrypt(Pr, cts[g]);
            std::vector<double> got;
            tr.encoder_hi->decode(got, Pr);
            for (int b = 0; b < tr.step_h; ++b)
                for (int u = 0; u < d; ++u)
                    for (int jp = 0; jp < tr.k; ++jp)
                        e = std::max(e,
                                     std::abs(got[tr.wide_slot(b, u, jp)] -
                                              want(b, u, g * tr.k + jp)));
        }
        std::cout << "[tb.dbg] " << label << " vs host: max err "
                  << std::scientific << std::setprecision(3) << e
                  << std::defaultfloat << std::endl;
    };

    auto wide_softmax = [&](std::vector<Ct>& slot_cts, double bound,
                            int exp_degree, int inv_degree, double sum_lo,
                            double sum_hi,
                            const std::vector<std::vector<double>>& masks,
                            bool pre_scaled) {
        const int log2k = int(std::round(std::log2(double(tr.k))));
        const double dom0 = heongpu::llama::Llama3Operator::domain_scale(
            sum_lo, sum_hi);

        std::vector<Ct> y;
        y.reserve(slot_cts.size());
        for (std::size_t g = 0; g < slot_cts.size(); ++g)
        {
            y.push_back(tr.arith_hi->exp_scaled_negative(
                slot_cts[g], bound, 1, exp_degree, *tr.relin_hi, pre_scaled));
            if (!masks.empty() && !masks[g].empty())
                tr.arith_hi->multiply_vector(y.back(), masks[g]);
        }
        dbg_slot(y[0], "softmax.exp_masked[0]");

        // Denominator: squares summed across parts, then the j' comb.
        std::vector<Ct> ysq;
        ysq.reserve(y.size());
        for (auto& part : y)
        {
            Ct sq = part;
            tr.arith_hi->square(sq, *tr.relin_hi);
            ysq.push_back(std::move(sq));
        }
        Ct total = ysq[0];
        for (std::size_t g = 1; g < ysq.size(); ++g)
            tr.ops_big->add_inplace(total, ysq[g]);
        for (int i = 0; i < log2k; ++i)
        {
            Ct tmp = total;
            tr.ops_big->rotate_rows_inplace(tmp, *tr.galois_hi,
                                            tr.step_h << i);
            tr.ops_big->add_inplace(total, tmp);
        }
        // Correct totals live at j' = 0 only; mask them (carrying the
        // reciprocal's domain scale) and fan back out.
        {
            std::vector<double> j0(slots_hi, 0.0);
            for (int b = 0; b < tr.step_h; ++b)
                for (int u = 0; u < tr.d; ++u)
                    j0[tr.wide_slot(b, u, 0)] = dom0;
            tr.arith_hi->multiply_vector(total, j0);
        }
        for (int i = 0; i < log2k; ++i)
        {
            Ct tmp = total;
            tr.ops_big->rotate_rows_inplace(tmp, *tr.galois_hi,
                                            slots_hi - (tr.step_h << i));
            tr.ops_big->add_inplace(total, tmp);
        }

        // The domain scale rides the DENOMINATOR's comb mask only (unlike
        // the library path, where it rides the numerator via the exp mask
        // and needs cancelling): pre_scaled consumes it in the map, so the
        // fit is exactly 1/x -- gain stays 1.
        dbg_slot(total, "softmax.denominator(mapped)");
        Ct recip = tr.arith_hi->inverse(total, sum_lo, sum_hi, inv_degree, 0,
                                        *tr.relin_hi, /*pre_scaled=*/true,
                                        /*gain=*/1.0);
        dbg_slot(recip, "softmax.reciprocal");
        for (std::size_t g = 0; g < ysq.size(); ++g)
            slot_cts[g] = tr.arith_hi->multiply_and_rescale(ysq[g], recip,
                                                            *tr.relin_hi);
        dbg_slot(slot_cts[0], "softmax.out[0]");
    };

    // ==================================================================
    // Unit stage: SoftMax at the wide layout, direct-encoded.
    // ==================================================================
    if (stage == "softmax")
    {
        // Scores in a CALIBRATED-like range: the unit tests the layout law,
        // not the reciprocal's reach over adversarial ranges, so keep the
        // per-row denominators inside what a deg-15 1/x fit covers.
        const int blocks = tr.step_h; // heads carried per slot-ct
        const int nct = d / tr.k;
        std::uniform_real_distribution<double> dist(-2.5, 0.0);
        // want[b][u][key]
        std::vector<std::vector<double>> scores(
            blocks, std::vector<double>(std::size_t(d) * d));
        for (auto& m : scores)
            for (auto& v : m)
                v = dist(rng);

        std::vector<Ct> slot_cts;
        for (int g = 0; g < nct; ++g)
        {
            std::vector<double> msg(slots_hi, 0.0);
            for (int b = 0; b < blocks; ++b)
                for (int u = 0; u < d; ++u)
                    for (int jp = 0; jp < tr.k; ++jp)
                        msg[tr.wide_slot(b, u, jp)] =
                            scores[b][std::size_t(u) * d + g * tr.k + jp];
            heongpu::Plaintext<S> P1(tr.big);
            tr.encoder_hi->encode(P1, msg, tr.scale);
            Ct C1(tr.big);
            tr.encryptor_hi->encrypt(C1, P1);
            // Run in the 33-bit working window, as the real flow does after
            // a refresh -- the chain's top primes are 56/60-bit and are not
            // where the non-linears belong.
            while (tr.L_hi - C1.depth() > 17)
                tr.arith_hi->mod_drop_inplace(C1);
            slot_cts.push_back(std::move(C1));
        }

        const bool causal = EnvInt("HEONGPU_TB_CAUSAL", 1) != 0;
        std::vector<std::vector<double>> masks;
        if (causal)
            for (int g = 0; g < nct; ++g)
                masks.push_back(WideCausalMask(tr, g, d));

        // The denominator range from the plaintext (calibration stand-in).
        // With the causal row weight the circuit's denominator is
        // (d/(u+1)) * sum_{j<=u} exp(s).
        double u_lo = 1e30, u_hi = 0.0;
        for (int b = 0; b < blocks; ++b)
            for (int u = 0; u < d; ++u)
            {
                double denom = 0.0;
                for (int j = 0; j < d; ++j)
                    if (!causal || j <= u)
                        denom += std::exp(scores[b][std::size_t(u) * d + j]);
                if (causal)
                    denom *= double(d) / double(u + 1);
                u_lo = std::min(u_lo, denom);
                u_hi = std::max(u_hi, denom);
            }

        ledger.charge("softmax", [&]() {
            wide_softmax(slot_cts, 3.0, EnvInt("HEONGPU_TB_EXP_DEGREE", 15),
                         EnvInt("HEONGPU_TB_INV_DEGREE", 15), 0.8 * u_lo,
                         1.25 * u_hi, masks, /*pre_scaled=*/false);
        });

        if (check)
        {
            double err = 0.0;
            for (int b = 0; b < blocks; ++b)
            {
                const auto want =
                    HostMaskedSoftmax(scores[b], d, d, causal);
                for (int g = 0; g < nct; ++g)
                {
                    heongpu::Plaintext<S> Pr(tr.big);
                    tr.decryptor_hi->decrypt(Pr, slot_cts[g]);
                    std::vector<double> got;
                    tr.encoder_hi->decode(got, Pr);
                    for (int u = 0; u < d; ++u)
                        for (int jp = 0; jp < tr.k; ++jp)
                            err = std::max(
                                err,
                                std::abs(got[tr.wide_slot(b, u, jp)] -
                                         want[std::size_t(u) * d + g * tr.k +
                                              jp]));
                }
            }
            std::cout << "[tb] wide softmax max abs error " << std::scientific
                      << std::setprecision(3) << err << std::defaultfloat
                      << std::endl;
        }
        ledger.print("softmax unit");
        return 0;
    }

    // ==================================================================
    // Unit stage: RMSNorm at the wide layout, direct-encoded.
    // ==================================================================
    if (stage == "rmsnorm")
    {
        const int nct = d / tr.k;              // slot-cts per activation
        const int chan_per_ct = tr.step_h * tr.k; // channels per token per ct
        const int channels = chan_per_ct * nct;
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> x(std::size_t(d) * channels);
        for (auto& v : x)
            v = dist(rng);
        std::vector<double> gain(channels);
        for (auto& v : gain)
            v = 0.5 + 0.5 * (dist(rng) + 1.0) * 0.5;

        // Channel c of slot-ct g: c = b*d? No -- at the unit stage there is
        // no island grouping; use the same law as the full path: channel
        // c = b*d + (g*k + j') with d columns split k apiece over g.
        // channels == step_h * d must hold for that reading.
        if (channels != tr.step_h * d)
        {
            std::cout << "[tb] rmsnorm unit wants step_h*d channels, got "
                      << channels << std::endl;
            return 2;
        }

        std::vector<Ct> slot_cts;
        for (int g = 0; g < nct; ++g)
        {
            std::vector<double> msg(slots_hi, 0.0);
            for (int b = 0; b < tr.step_h; ++b)
                for (int u = 0; u < d; ++u)
                    for (int jp = 0; jp < tr.k; ++jp)
                        msg[tr.wide_slot(b, u, jp)] =
                            x[std::size_t(u) * channels + b * d + g * tr.k +
                              jp];
            heongpu::Plaintext<S> P1(tr.big);
            tr.encoder_hi->encode(P1, msg, tr.scale);
            Ct C1(tr.big);
            tr.encryptor_hi->encrypt(C1, P1);
            while (tr.L_hi - C1.depth() > 17)
                tr.arith_hi->mod_drop_inplace(C1);
            slot_cts.push_back(std::move(C1));
        }

        // Fit range from the data.
        double s_lo = 1e30, s_hi = 0.0;
        for (int u = 0; u < d; ++u)
        {
            double s = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                const double v = x[std::size_t(u) * channels + c];
                s += v * v;
            }
            s_lo = std::min(s_lo, s);
            s_hi = std::max(s_hi, s);
        }

        heongpu::llama::Llama3Operator::RMSNormConfig cfg;
        cfg.stride = slots_hi;
        cfg.count = chan_per_ct;
        cfg.channels = channels;
        cfg.token_blocks = 1;
        cfg.eps = 1e-5;
        cfg.sum_lo = 0.8 * s_lo;
        cfg.sum_hi = 1.2 * s_hi;
        cfg.degree = EnvInt("HEONGPU_TB_NORM_DEGREE", 15);
        cfg.newton_iterations = EnvInt("HEONGPU_TB_NORM_NEWTON", 0);
        cfg.blocked_span = chan_per_ct;
        // The proven single-ring path: mask carries the domain map and the
        // mean division, the fit carries the rest; no Newton step.
        cfg.fold_mean_into_fit = true;
        cfg.fold_affine_into_mask = true;

        std::vector<heongpu::Plaintext<S>> weights;
        for (int g = 0; g < nct; ++g)
        {
            const std::vector<double> w = WideGainVector(tr, g, d, gain);
            heongpu::Plaintext<S> P1(tr.big);
            tr.encoder_hi->encode(P1, w, tr.scale);
            weights.push_back(std::move(P1));
        }

        std::vector<Ct> out;
        ledger.charge("rmsnorm", [&]() {
            out = tr.arith_hi->rms_norm(slot_cts, weights, cfg, *tr.galois_hi,
                                        *tr.relin_hi);
        });

        if (check)
        {
            const auto want = HostRmsNorm(x, d, channels, gain, cfg.eps);
            double err = 0.0;
            for (int g = 0; g < nct; ++g)
            {
                heongpu::Plaintext<S> Pr(tr.big);
                tr.decryptor_hi->decrypt(Pr, out[g]);
                std::vector<double> got;
                tr.encoder_hi->decode(got, Pr);
                err = std::max(
                    err, WideSlotError(tr, got, g, d, tr.step_h,
                                       [&](int b, int u, int col) {
                                           return want[std::size_t(u) *
                                                           channels +
                                                       b * d + col];
                                       }));
            }
            std::cout << "[tb] wide rmsnorm max abs error " << std::scientific
                      << std::setprecision(3) << err << std::defaultfloat
                      << std::endl;
        }
        ledger.print("rmsnorm unit");
        return 0;
    }

    // ==================================================================
    // Full two-ring stages. Everything below needs the island and the v2
    // bootstrap: island level windows are at most shared-1 products deep,
    // and every non-linear visit to the big ring starts with a refresh.
    // ==================================================================

    using RectActivation = heongpu::llama::RectActivation;
    using BatchActivation = heongpu::llama::BatchActivation;

    if (model != tr.step_h * d)
    {
        // step_h*d == N_is/2 always; the driver keeps one rect group.
        std::cout << "[tb] this driver runs one rect group: model must be "
                  << tr.step_h * d << std::endl;
        return 2;
    }

    const double amp_w = 1.0 / std::sqrt(double(model));
    std::uniform_real_distribution<double> xdist(-0.5, 0.5);
    std::vector<double> x_plain(std::size_t(d) * model);
    for (auto& v : x_plain)
        v = xdist(rng);

    // Host-side attention intermediates, filled by the conditioning block
    // further down; declared here so the seam lambdas can capture them.
    std::vector<double> p_host;     // masked softmax result, per head
    std::vector<double> scores_dbg; // gain-scaled scores, pre-shift

    // Every wide bridge runs on the CHEAP side: up-crossings at l = 2
    // (pre-boot; a slot-form boot is clean at exact scale), down-crossings
    // standardized at l = 5 -- so the bridge's plaintext-diagonal cache
    // holds exactly two small sets (l=2 up, l=5 down) instead of the
    // ~8.6 GiB a single high-level set costs. Only the 63-diagonal block
    // map runs high.
    const int FROM_L = 5;
    auto drop_big_to = [&](std::vector<Ct>& cts, int l) {
        for (auto& c : cts)
            while (tr.L_hi - c.depth() > l)
                tr.arith_hi->mod_drop_inplace(c);
    };
    auto from_slots_low = [&](std::vector<Ct>& slots) {
        drop_big_to(slots, FROM_L);
        return tr.wide_from_slots(slots);
    };

    // ---- RMSNorm at the big ring, on an island RECT activation. --------
    // One ascend + boot serves TWO consumers: the norm path (block map ->
    // rms_norm -> block map -> bridge down) and the residual "skip" copy
    // (the same refreshed slots bridged straight back down, untouched).
    const int nct_wide = d / tr.k;
    const double eps_norm = 1e-5;
    auto wide_rms_norm = [&](RectActivation& x, const std::vector<double>& g,
                             double s_lo, double s_hi, RectActivation& normed,
                             RectActivation& skip) {
        std::vector<Ct> big = tr.ascend(x.column);
        drop_big_to(big, 2);
        std::vector<Ct> slots = tr.wide_to_slots(big); // l = 1
        tr.refresh(slots, "norm");                     // boot -> 17

        // The skip copy: the raw bridged slots go straight back down --
        // from_slots(to_slots(x)) is the identity, so this IS the
        // refreshed stream.
        {
            std::vector<Ct> skip_slots = slots;
            std::vector<Ct> skip_big = from_slots_low(skip_slots);
            std::vector<Ct> cols = tr.descend(skip_big, shared);
            skip.column = std::move(cols);
            skip.rows = d;
            skip.groups = 1;
            skip.channels = model;
        }

        // The block map completes the RECT crossing at the top.
        ledger.charge("wide.block_map", [&]() {
            tr.wide_rect->block_map(slots, true, "wide.block_inverse",
                                    *tr.galois_hi);
        });

        heongpu::llama::Llama3Operator::RMSNormConfig cfg;
        cfg.stride = slots_hi;
        cfg.count = tr.step_h * tr.k;
        cfg.channels = model;
        cfg.token_blocks = 1;
        cfg.eps = eps_norm;
        cfg.sum_lo = s_lo;
        cfg.sum_hi = s_hi;
        cfg.degree = EnvInt("HEONGPU_TB_NORM_DEGREE", 15);
        cfg.newton_iterations = EnvInt("HEONGPU_TB_NORM_NEWTON", 0);
        cfg.blocked_span = tr.step_h * tr.k;
        cfg.fold_mean_into_fit = true;
        cfg.fold_affine_into_mask = true;

        std::vector<heongpu::Plaintext<S>> weights;
        for (int gi = 0; gi < nct_wide; ++gi)
        {
            const std::vector<double> w = WideGainVector(tr, gi, d, g);
            heongpu::Plaintext<S> P1(tr.big);
            tr.encoder_hi->encode(P1, w, tr.scale);
            weights.push_back(std::move(P1));
        }
        ledger.charge("rmsnorm", [&]() {
            slots = tr.arith_hi->rms_norm(slots, weights, cfg, *tr.galois_hi,
                                          *tr.relin_hi);
        });
        // The norm's multiplies drifted the scale; everything the normed
        // stream feeds (projections -> to_batch -> the next boots)
        // preserves scale, so normalize once here.
        ledger.charge("scale_norm", [&]() {
            for (auto& c : slots)
                tr.arith_hi->match_scale(c, tr.scale);
        });
        ledger.charge("wide.block_map", [&]() {
            tr.wide_rect->block_map(slots, false, "wide.block_forward",
                                    *tr.galois_hi);
        });
        std::vector<Ct> big_out = from_slots_low(slots);
        std::vector<Ct> cols = tr.descend(big_out, shared);
        normed.column = std::move(cols);
        normed.rows = d;
        normed.groups = 1;
        normed.channels = model;
    };

    // ---- The pure refresh crossing for island data: up, boot, down. -----
    auto island_refresh = [&](std::vector<Ct>& cols, const char* name) {
        std::vector<Ct> big = tr.ascend(cols);
        tr.refresh(big, name);
        cols = tr.descend(big, shared);
    };

    // ---- Attention sublayer, two-ring. ---------------------------------
    // x arrives as island RECT at l = shared. Weights are model x model
    // row-major (in x out), K/V already GQA-expanded.
    // The verified window structure at shared = 3 (2 products per descent):
    //   I1: project q/k/v (1) + to_batch (1)           -> BATCH @ 1
    //   ^   pure crossing refresh for q, k (V idles refreshed too)
    //   I2: K^T transpose (0) + QK matmul (1)          -> scores @ 2
    //   B1: ascend, wide to_slots CHEAP at l=2 (BATCH: no block map),
    //       boot from l=1 in slot form -> softmax entry 17, softmax,
    //       from_slots, drop, descend                  -> P BATCH @ 3
    //   I3: PV matmul (1) + from_batch fused (1)       -> attn RECT @ 1
    //   ^   pure crossing refresh
    //   I4: W_o project (1)                            -> out @ 2
    auto run_attention = [&](RectActivation& x, const std::vector<double>& wq,
                             const std::vector<double>& wk,
                             const std::vector<double>& wv,
                             const std::vector<double>& wo, double shift,
                             double bound, double sum_lo, double sum_hi,
                             bool causal) -> RectActivation {
        RectActivation q, kk, v;
        ledger.charge("island.project_qkv", [&]() {
            q = tr.rect_is->project(x, wq, model, model, "attn.q",
                                    *tr.galois_is);
            kk = tr.rect_is->project(x, wk, model, model, "attn.k",
                                     *tr.galois_is);
            v = tr.rect_is->project(x, wv, model, model, "attn.v",
                                    *tr.galois_is);
        });
        BatchActivation qb, kb, vb, kt, scores;
        ledger.charge("island.to_batch", [&]() {
            qb = tr.rect_is->to_batch(q, 0, *tr.galois_is);
            kb = tr.rect_is->to_batch(kk, 0, *tr.galois_is);
            vb = tr.rect_is->to_batch(v, 0, *tr.galois_is);
        });
        island_refresh(qb.column, "qkv");
        island_refresh(kb.column, "qkv");
        island_refresh(vb.column, "qkv"); // idles at shared until PV
        ledger.charge("island.qk", [&]() {
            kt = tr.batch_is->transpose(kb, "attn.kt", *tr.galois_is);
            scores = tr.batch_is->matmul(qb, kt, "attn.qk", *tr.galois_is,
                                         *tr.relin_is);
        });
        const double a_dbg =
            heongpu::llama::Llama3Operator::domain_scale(-bound, 0.0);
        if (dbg)
        {
            const auto got = tr.batch_is->decrypt(
                scores, *tr.decryptor_is, scores.column.front().scale());
            double e = 0.0;
            for (int b = 0; b < heads && b < int(got.size()); ++b)
                for (int u = 0; u < d; ++u)
                    for (int j = 0; j < d; ++j)
                        e = std::max(
                            e, std::abs(got[b][std::size_t(u) * d + j] -
                                        a_dbg * scores_dbg[(std::size_t(b) *
                                                                d +
                                                            u) *
                                                               d +
                                                           j]));
            std::cout << "[tb.dbg] island scores vs host: max err "
                      << std::scientific << std::setprecision(3) << e
                      << std::defaultfloat << std::endl;
        }

        // Scores up: normalize the QK product's drifted scale FIRST (the
        // v2 boot silently corrupts inputs whose tracked scale deviates
        // from the nominal 2^33 -- measured on this exact path: drifted
        // 2^33*1.00036 boots to garbage, exact boots to 5e-7), bridge on
        // the cheap side at l = 2, then boot the exact-scale slot form.
        // Softmax enters at the full window, E = 17.
        std::vector<Ct> sbig = tr.ascend(scores.column);
        ledger.charge("scale_norm", [&]() {
            for (auto& c : sbig)
                tr.arith_hi->match_scale(c, tr.scale);
        });
        std::vector<Ct> sslots = tr.wide_to_slots(sbig);
        tr.refresh(sslots, "post_qk");
        dbg_slots_vs(sslots, "scores.slots(post-boot)",
                     [&](int b, int u, int key) {
                         return a_dbg *
                                scores_dbg[(std::size_t(b) * d + u) * d + key];
                     });
        ledger.charge("softmax", [&]() {
            // The exp fit's domain scale is already folded into the query
            // weight; the score shift rides the same map as a free constant.
            // pre_scaled applies the SCALE only -- the fit adds its own
            // domain shift internally, so it must not be added here.
            const double a =
                heongpu::llama::Llama3Operator::domain_scale(-bound, 0.0);
            for (auto& c : sslots)
                tr.arith_hi->add_constant(c, a * shift);

            std::vector<std::vector<double>> masks;
            if (causal)
                for (int gi = 0; gi < nct_wide; ++gi)
                    masks.push_back(WideCausalMask(tr, gi, d));
            wide_softmax(sslots, bound, EnvInt("HEONGPU_TB_EXP_DEGREE", 15),
                         EnvInt("HEONGPU_TB_INV_DEGREE", 15), sum_lo, sum_hi,
                         masks, /*pre_scaled=*/true);
        });
        dbg_slots_vs(sslots, "softmax.P",
                     [&](int b, int u, int key) {
                         return p_host[(std::size_t(b) * d + u) * d + key];
                     });
        std::vector<Ct> pbig = from_slots_low(sslots);
        // The PV window needs P at l = shared with 2 products ahead. If a
        // deeper-than-budgeted softmax fit ate the slack, refresh P here
        // (the D4 fallback); with the S = 12 budget this never fires.
        if (!pbig.empty() && tr.L_hi - pbig.front().depth() < shared)
            tr.refresh(pbig, "post_softmax");
        std::vector<Ct> pcols = tr.descend(pbig, shared);
        BatchActivation P, V2;
        P.rows = d;
        P.column = std::move(pcols);
        V2.rows = d;
        V2.column = std::move(vb.column);

        BatchActivation outb;
        RectActivation out;
        ledger.charge("island.pv", [&]() {
            int depth = 0;
            for (auto& c : P.column)
                depth = std::max(depth, c.depth());
            for (auto& c : V2.column)
                depth = std::max(depth, c.depth());
            for (auto& c : P.column)
                tr.rect_is->arith().drop_to_depth(c, depth);
            for (auto& c : V2.column)
                tr.rect_is->arith().drop_to_depth(c, depth);
            outb = tr.batch_is->matmul(P, V2, "attn.pv", *tr.galois_is,
                                       *tr.relin_is);
        });
        ledger.charge("island.out", [&]() {
            std::vector<BatchActivation> groups;
            groups.push_back(std::move(outb));
            out = tr.rect_is->from_batch(groups, model, *tr.galois_is);
        });
        // The PV product drifted the scale; the tail refresh must see the
        // exact nominal or the boot corrupts it.
        ledger.charge("scale_norm", [&]() {
            for (auto& c : out.column)
                tr.rect_is->arith().match_scale(c, tr.scale);
        });
        island_refresh(out.column, "attn_tail");
        ledger.charge("island.out", [&]() {
            out = tr.rect_is->project(out, wo, model, model, "attn.o",
                                      *tr.galois_is);
        });
        return out;
    };

    // ---- Feed-forward sublayer, two-ring. -------------------------------
    // Gate and up project at the island, both ride a refresh to the big
    // ring, SiLU and the gate product run in wide slots, and the product
    // descends for the down projection.
    // FFN chunk walk: gate/up project on the island (1 level each from the
    // normed entry), refresh both branches at the big ring, cross with the
    // block map (RECT data), SiLU pre-scaled (its 1/B rides the circuit's
    // gate weights) and the up branch 1/B_up-scaled for the boot's message
    // range (B_up rides the down weights), product, cross back, descend,
    // down-project. Chunks share an identical history, so they accumulate
    // with a level-free add.
    auto run_ffn = [&](RectActivation& x, const std::vector<double>& wg,
                       const std::vector<double>& wu,
                       const std::vector<double>& wdn, int hidden_eff,
                       double silu_bound, double up_bound) -> RectActivation {
        RectActivation acc;
        const int chunks = hidden_eff / model;
        for (int ch = 0; ch < chunks; ++ch)
        {
            std::vector<double> wg_c(std::size_t(model) * model);
            std::vector<double> wu_c(std::size_t(model) * model);
            for (int i = 0; i < model; ++i)
                for (int o = 0; o < model; ++o)
                {
                    wg_c[std::size_t(i) * model + o] =
                        wg[std::size_t(i) * hidden_eff + ch * model + o] /
                        silu_bound;
                    wu_c[std::size_t(i) * model + o] =
                        wu[std::size_t(i) * hidden_eff + ch * model + o] /
                        up_bound;
                }
            RectActivation gate, up;
            ledger.charge("island.project_ffn", [&]() {
                gate = tr.rect_is->project(x, wg_c, model, model, "ffn.gate",
                                           *tr.galois_is);
                up = tr.rect_is->project(x, wu_c, model, model, "ffn.up",
                                         *tr.galois_is);
            });
            std::vector<Ct> gbig = tr.ascend(gate.column);
            std::vector<Ct> ubig = tr.ascend(up.column);
            drop_big_to(gbig, 2);
            drop_big_to(ubig, 2);
            std::vector<Ct> gslots = tr.wide_to_slots(gbig);
            std::vector<Ct> uslots = tr.wide_to_slots(ubig);
            tr.refresh(gslots, "ffn_gate");
            tr.refresh(uslots, "ffn_up");
            ledger.charge("wide.block_map", [&]() {
                tr.wide_rect->block_map(gslots, true, "wide.block_inverse",
                                        *tr.galois_hi);
                tr.wide_rect->block_map(uslots, true, "wide.block_inverse",
                                        *tr.galois_hi);
            });
            ledger.charge("swiglu", [&]() {
                const int silu_degree = EnvInt("HEONGPU_TB_SILU_DEGREE", 31);
                for (std::size_t j = 0; j < gslots.size(); ++j)
                {
                    // Gate slots arrive pre-mapped onto [-1, 1] by the
                    // weight fold; pre_scaled skips the domain multiply.
                    Ct act = tr.arith_hi->silu(gslots[j], silu_bound,
                                               silu_degree, *tr.relin_hi,
                                               /*pre_scaled=*/true);
                    tr.arith_hi->drop_to_depth(uslots[j], act.depth());
                    gslots[j] = tr.arith_hi->multiply_and_rescale(
                        act, uslots[j], *tr.relin_hi);
                }
            });
            ledger.charge("wide.block_map", [&]() {
                tr.wide_rect->block_map(gslots, false, "wide.block_forward",
                                        *tr.galois_hi);
            });
            std::vector<Ct> hbig = from_slots_low(gslots);
            std::vector<Ct> hcols = tr.descend(hbig, shared);
            RectActivation h;
            h.column = std::move(hcols);
            h.rows = d;
            h.groups = 1;
            h.channels = model;

            std::vector<double> wd_c(std::size_t(model) * model);
            for (int i = 0; i < model; ++i)
                for (int o = 0; o < model; ++o)
                    wd_c[std::size_t(i) * model + o] =
                        wdn[std::size_t(ch * model + i) * model + o] *
                        up_bound;
            RectActivation part;
            ledger.charge("island.project_down", [&]() {
                part = tr.rect_is->project(h, wd_c, model, model, "ffn.down",
                                           *tr.galois_is);
            });
            if (ch == 0)
                acc = std::move(part);
            else
                ledger.charge("island.accumulate", [&]() {
                    for (std::size_t j = 0; j < acc.column.size(); ++j)
                        tr.rect_is->arith().add_inplace(acc.column[j],
                                                        part.column[j]);
                });
        }
        return acc;
    };

    // ---- Weights, host reference, and the stage dispatch. ---------------
    std::vector<double> wq = RandomMatrix(std::size_t(model) * model, amp_w,
                                          rng);
    std::vector<double> wk(std::size_t(model) * model);
    std::vector<double> wv(std::size_t(model) * model);
    {
        // GQA: kv_heads distinct head blocks, replicated across the query
        // heads that share them. A head block is d output columns.
        const int group = heads / kv_heads;
        std::vector<double> kv_k = RandomMatrix(
            std::size_t(model) * kv_heads * d, amp_w, rng);
        std::vector<double> kv_v = RandomMatrix(
            std::size_t(model) * kv_heads * d, amp_w, rng);
        for (int i = 0; i < model; ++i)
            for (int h = 0; h < heads; ++h)
            {
                const int src = h / group;
                for (int j = 0; j < d; ++j)
                {
                    wk[std::size_t(i) * model + h * d + j] =
                        kv_k[std::size_t(i) * kv_heads * d + src * d + j];
                    wv[std::size_t(i) * model + h * d + j] =
                        kv_v[std::size_t(i) * kv_heads * d + src * d + j];
                }
            }
    }
    std::vector<double> wo = RandomMatrix(std::size_t(model) * model, amp_w,
                                          rng);

    // Gains and the attention input are host-known up front, so the score
    // conditioning below fits the input attention actually sees (the block
    // stage feeds the NORMED stream into the projections).
    const bool causal = EnvInt("HEONGPU_TB_CAUSAL", 1) != 0;
    std::vector<double> gain1(model), gain2(model);
    {
        std::uniform_real_distribution<double> gd(0.5, 1.0);
        for (auto& v : gain1)
            v = gd(rng);
        for (auto& v : gain2)
            v = gd(rng);
    }
    const std::vector<double> attn_in =
        (stage == "block") ? HostRmsNorm(x_plain, d, model, gain1, eps_norm)
                           : x_plain;

    // Score conditioning: scale wq so shifted scores land in a friendly
    // exp domain, computed from the plaintext (calibration stands in).
    double score_shift = 0.0, score_bound = 8.0;
    double den_lo = 1.0, den_hi = double(d);
    std::vector<double> q_host, k_host, v_host;
    {
        q_host = HostMatmul(attn_in, d, model, wq, model);
        k_host = HostMatmul(attn_in, d, model, wk, model);
        v_host = HostMatmul(attn_in, d, model, wv, model);
        double s_lo = 1e30, s_hi = -1e30;
        std::vector<double> scores_all(std::size_t(heads) * d * d);
        for (int h = 0; h < heads; ++h)
            for (int u = 0; u < d; ++u)
                for (int j = 0; j < d; ++j)
                {
                    double s = 0.0;
                    for (int m = 0; m < d; ++m)
                        s += q_host[std::size_t(u) * model + h * d + m] *
                             k_host[std::size_t(j) * model + h * d + m];
                    scores_all[(std::size_t(h) * d + u) * d + j] = s;
                    s_lo = std::min(s_lo, s);
                    s_hi = std::max(s_hi, s);
                }
        // Rescale wq (and the host q/scores) so the score span is ~3.
        const double target_span = 3.0;
        const double gain = target_span / std::max(1e-9, s_hi - s_lo);
        for (auto& vw : wq)
            vw *= gain;
        for (auto& vq : q_host)
            vq *= gain;
        for (auto& s : scores_all)
            s *= gain;
        s_lo *= gain;
        s_hi *= gain;
        scores_dbg = scores_all;
        score_shift = -s_hi;
        // The shifted scores span [-(s_hi-s_lo), 0]; the exp fit interval
        // is widened so the pre-scaled circuit values 2x/bound stay inside
        // the v2 boot's |m| <= 1 message contract (the deg-15 exp fit
        // tolerates the wider interval; the real block fits [-20.9, 0]).
        score_bound = (s_hi - s_lo) * 2.5;

        // Host softmax on shifted scores; record denominator range.
        p_host.assign(scores_all.size(), 0.0);
        den_lo = 1e30;
        den_hi = 0.0;
        for (int h = 0; h < heads; ++h)
        {
            for (int u = 0; u < d; ++u)
            {
                double denom = 0.0;
                for (int j = 0; j < d; ++j)
                {
                    const bool live = !causal || j <= u;
                    const double e =
                        live ? std::exp(scores_all[(std::size_t(h) * d + u) *
                                                       d +
                                                   j] +
                                        score_shift)
                             : 0.0;
                    p_host[(std::size_t(h) * d + u) * d + j] = e;
                    denom += e;
                }
                // The circuit's denominator carries the causal mask's
                // row-count equalizer squared: (d/(u+1)) * sum exp.
                const double den_circuit =
                    causal ? denom * double(d) / double(u + 1) : denom;
                den_lo = std::min(den_lo, den_circuit);
                den_hi = std::max(den_hi, den_circuit);
                for (int j = 0; j < d; ++j)
                    p_host[(std::size_t(h) * d + u) * d + j] /= denom;
            }
        }
        den_lo *= 0.8;
        den_hi *= 1.25;

        // Fold the exp fit's domain scale into the query weight -- the
        // free level the schedule counts on. Host math above used the
        // unfolded scores; only the circuit weight carries the map.
        const double a = heongpu::llama::Llama3Operator::domain_scale(
            -score_bound, 0.0);
        for (auto& vw : wq)
            vw *= a;
    }

    // The attention sublayer's host output: ctx = P.V per head, then Wo.
    std::vector<double> attn_sub_host;
    {
        std::vector<double> ctx(std::size_t(d) * model, 0.0);
        for (int h = 0; h < heads; ++h)
            for (int u = 0; u < d; ++u)
                for (int j = 0; j < d; ++j)
                {
                    const double p = p_host.empty()
                                         ? 0.0
                                         : p_host[(std::size_t(h) * d + u) *
                                                      d +
                                                  j];
                    if (p == 0.0)
                        continue;
                    for (int m = 0; m < d; ++m)
                        ctx[std::size_t(u) * model + h * d + m] +=
                            p * v_host[std::size_t(j) * model + h * d + m];
                }
        attn_sub_host = HostMatmul(ctx, d, model, wo, model);
    }

    if (stage == "attention" || stage == "block")
    {
        RectActivation x0 = tr.rect_is->encrypt(x_plain, model,
                                                *tr.encryptor_is, tr.scale);

        RectActivation stream = std::move(x0);
        RectActivation attn_out;
        if (stage == "block")
        {
            // norm1 -> attention -> residual against the refreshed skip.
            double s_lo = 1e30, s_hi = 0.0;
            for (int u = 0; u < d; ++u)
            {
                double s = 0.0;
                for (int c = 0; c < model; ++c)
                {
                    const double v = x_plain[std::size_t(u) * model + c];
                    s += v * v;
                }
                s_lo = std::min(s_lo, s);
                s_hi = std::max(s_hi, s);
            }
            RectActivation normed, skip;
            wide_rms_norm(stream, gain1, 0.8 * s_lo, 1.2 * s_hi, normed,
                          skip);

            attn_out = run_attention(normed, wq, wk, wv, wo, score_shift,
                                     score_bound, den_lo, den_hi, causal);
            ledger.charge("residual", [&]() {
                stream.column = tr.rect_is->arith().residual_add(
                    skip.column, attn_out.column);
            });
        }
        else
        {
            attn_out = run_attention(stream, wq, wk, wv, wo, score_shift,
                                     score_bound, den_lo, den_hi, causal);
            stream = std::move(attn_out);
        }

        if (stage == "attention" && check)
        {
            const auto got = tr.rect_is->decrypt(
                stream, *tr.decryptor_is, stream.column.front().scale());
            std::cout << "[tb] attention max abs error " << std::scientific
                      << std::setprecision(3)
                      << MaxAbsDiff(got, attn_sub_host) << std::defaultfloat
                      << std::endl;
        }

        if (stage == "attention")
        {
            ledger.print("attention");
            return 0;
        }

        // ---- block: norm2 -> ffn -> residual. ---------------------------
        // Round the hidden width UP to whole island groups; the pad columns
        // are zero-free random weights here, and the host reference uses
        // the same effective width, so correctness is unaffected while the
        // cost is the honest padded-chunk cost.
        const int hidden_eff = ((hidden + model - 1) / model) * model;
        if (hidden_eff != hidden)
            std::cout << "[tb] hidden " << hidden << " padded to "
                      << hidden_eff << " (" << hidden_eff / model
                      << " island chunks)" << std::endl;
        std::vector<double> wg = RandomMatrix(
            std::size_t(model) * hidden_eff, amp_w, rng);
        std::vector<double> wu = RandomMatrix(
            std::size_t(model) * hidden_eff, amp_w, rng);
        std::vector<double> wdn = RandomMatrix(
            std::size_t(hidden_eff) * model,
            1.0 / std::sqrt(double(hidden_eff)), rng);

        // Host stream after the attention residual: p_host and v_host were
        // computed on attn_in, which IS the normed stream here.
        std::vector<double> stream_host(x_plain);
        for (std::size_t i = 0; i < stream_host.size(); ++i)
            stream_host[i] += attn_sub_host[i];

        // norm2 fit range from the host stream.
        double s2_lo = 1e30, s2_hi = 0.0;
        for (int u = 0; u < d; ++u)
        {
            double s = 0.0;
            for (int c = 0; c < model; ++c)
            {
                const double v = stream_host[std::size_t(u) * model + c];
                s += v * v;
            }
            s2_lo = std::min(s2_lo, s);
            s2_hi = std::max(s2_hi, s);
        }
        RectActivation normed2, skip2;
        wide_rms_norm(stream, gain2, 0.8 * s2_lo, 1.2 * s2_hi, normed2,
                      skip2);

        const auto normed2_host =
            HostRmsNorm(stream_host, d, model, gain2, eps_norm);
        double g_amp = 0.0, u_amp = 0.0;
        {
            const auto gh = HostMatmul(normed2_host, d, model, wg, hidden_eff);
            for (const double v : gh)
                g_amp = std::max(g_amp, std::abs(v));
            const auto uh = HostMatmul(normed2_host, d, model, wu, hidden_eff);
            for (const double v : uh)
                u_amp = std::max(u_amp, std::abs(v));
        }
        const double silu_bound = g_amp * 1.25;
        const double up_bound = u_amp * 1.25;

        RectActivation ffn_out = run_ffn(normed2, wg, wu, wdn, hidden_eff,
                                         silu_bound, up_bound);
        ledger.charge("residual", [&]() {
            stream.column = tr.rect_is->arith().residual_add(skip2.column,
                                                             ffn_out.column);
        });

        if (check)
        {
            const auto gh = HostMatmul(normed2_host, d, model, wg, hidden_eff);
            const auto uh = HostMatmul(normed2_host, d, model, wu, hidden_eff);
            std::vector<double> hh(gh.size());
            for (std::size_t i = 0; i < gh.size(); ++i)
                hh[i] = HostSilu(gh[i]) * uh[i];
            const auto dh = HostMatmul(hh, d, hidden_eff, wdn, model);
            std::vector<double> want(stream_host);
            for (std::size_t i = 0; i < want.size(); ++i)
                want[i] += dh[i];
            const auto got = tr.rect_is->decrypt(
                stream, *tr.decryptor_is, stream.column.front().scale());
            const double err = MaxAbsDiff(got, want);
            double sig = 0.0;
            for (const double v : want)
                sig = std::max(sig, std::abs(v));
            std::cout << "[tb] block max abs error " << std::scientific
                      << std::setprecision(3) << err << std::defaultfloat
                      << " (signal " << std::setprecision(3) << sig
                      << ", " << std::log2(sig / std::max(err, 1e-300))
                      << " bits)" << std::endl;
        }
        ledger.print("block");
        std::cout << "[tb] final island depth: "
                  << stream.column.front().depth() << std::endl;
        return 0;
    }

    if (stage == "ffn")
    {
        RectActivation x0 = tr.rect_is->encrypt(x_plain, model,
                                                *tr.encryptor_is, tr.scale);
        const int hidden_eff = ((hidden + model - 1) / model) * model;
        std::vector<double> wg = RandomMatrix(
            std::size_t(model) * hidden_eff, amp_w, rng);
        std::vector<double> wu = RandomMatrix(
            std::size_t(model) * hidden_eff, amp_w, rng);
        std::vector<double> wdn = RandomMatrix(
            std::size_t(hidden_eff) * model,
            1.0 / std::sqrt(double(hidden_eff)), rng);
        double g_amp = 0.0, u_amp = 0.0;
        const auto gh = HostMatmul(x_plain, d, model, wg, hidden_eff);
        for (const double v : gh)
            g_amp = std::max(g_amp, std::abs(v));
        const auto uh_amp = HostMatmul(x_plain, d, model, wu, hidden_eff);
        for (const double v : uh_amp)
            u_amp = std::max(u_amp, std::abs(v));
        const double silu_bound = g_amp * 1.25;
        const double up_bound = u_amp * 1.25;
        RectActivation out = run_ffn(x0, wg, wu, wdn, hidden_eff, silu_bound,
                                     up_bound);
        if (check)
        {
            const auto uh = HostMatmul(x_plain, d, model, wu, hidden_eff);
            std::vector<double> hh(gh.size());
            for (std::size_t i = 0; i < gh.size(); ++i)
                hh[i] = HostSilu(gh[i]) * uh[i];
            const auto want = HostMatmul(hh, d, hidden_eff, wdn, model);
            const auto got = tr.rect_is->decrypt(out, *tr.decryptor_is,
                                                 out.column.front().scale());
            std::cout << "[tb] ffn max abs error " << std::scientific
                      << std::setprecision(3) << MaxAbsDiff(got, want)
                      << std::defaultfloat << std::endl;
        }
        ledger.print("ffn");
        return 0;
    }

    std::cout << "[tb] unknown stage '" << stage << "'" << std::endl;
    return 2;
}
