// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// CLOSING THE ISLAND: DOES SlotToCoeff UNDO THE BIT REVERSAL BY ITSELF?
// =====================================================================
//
// profile_boot_to_slots established two things by measurement. The rect ->
// slot crossing IS the bootstrap's CoeffToSlot, so a refresh can hand back the
// slot reading directly and save six levels; and the order it hands back is
// BIT REVERSED, which is what a decimation-in-time factorisation leaves
// behind. The second finding is what stopped the first from being usable: the
// module's way back to rect form, from_slots(), reads the natural order.
//
// The claim under test here is that the bit reversal never has to be undone at
// all, because the way back is not from_slots() but the bootstrap's OWN fourth
// stage:
//
//   CLAIM  solo_slot_to_coeff is the same Vandermonde that CoeffToSlot ran,
//          run backwards. Whatever relabelling the forward map applied, this
//          one inverts -- by construction, not by agreement about twiddle
//          conventions. So
//
//              refresh_to_slots  ->  (slot work)  ->  slots_to_rect
//
//          is the identity on a rect activation, and the reversal is invisible
//          to anything in between that respects the index map.
//
// If the claim holds the level ledger of a refresh-plus-norm seam goes from
//
//     bootstrap 27 + to_slots 2 + ... + from_slots 2      =  31
//   to
//     refresh_to_slots 21 + ... + slots_to_rect StoC      =  21 + StoC_piece
//
// and four homomorphic linear maps out of six disappear.
//
// Three measurements, in increasing order of how much they can go wrong:
//
//   1. THE ROUND TRIP. Encrypt, refresh into slots, come straight back, and
//      decrypt as a rect activation. Compared against a plain bootstrap of the
//      same data, which is the accuracy floor -- neither can beat it, and the
//      island must not be worse.
//
//   2. THE INDEX MAP. island_slot() claims the island puts block b of token u
//      at revbits(b) + (k/2)*revbits(u), the same two fields to_slots uses
//      with each one reversed. Checked directly against the decoded slots, for
//      both readings, so a wrong field assignment cannot hide.
//
//   3. A WHOLE RMSNorm, both ways, against the plaintext answer. This is the
//      one that exercises the part of the argument that is not about the two
//      transforms: that a reduction over a WHOLE field -- sum_blocked of span
//      k/2 -- is untouched by an order change inside that field, because it
//      sums slot positions and not slot contents.
//
//   HEONGPU_ISLAND_LOGN        ring degree                        12
//   HEONGPU_ISLAND_D           block size                         64
//   HEONGPU_ISLAND_LIMBS       chain length                       36
//   HEONGPU_ISLAND_PRIME_BITS  scale prime size                   50
//   HEONGPU_ISLAND_CTOS        CoeffToSlot pieces                  3
//   HEONGPU_ISLAND_STOC        SlotToCoeff pieces                  3
//   HEONGPU_ISLAND_TAYLOR      EvalMod Taylor degree              11
//   HEONGPU_ISLAND_DEGREE      1/sqrt fit degree                  15
//   HEONGPU_ISLAND_ALIGN       levels dropped before the StoC      1
//   HEONGPU_ISLAND_SWEEP       try the other alignments too        0
//   HEONGPU_ISLAND_TRIP        run measurements 1 and 2            1
//   HEONGPU_ISLAND_NORM        run measurement 3                   1

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;
using Rect = heongpu::llama::Llama3RectOperator;
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

    struct Fit
    {
        double gain = 0.0;
        double residual = 0.0;
        double rel = 0.0;
    };

    // Least squares through the origin, then the worst absolute residual. One
    // scalar is the only freedom allowed, so a missing normalisation shows up
    // as a gain away from 1 while a permutation error survives as residual.
    Fit FitTo(const std::vector<double>& measured,
              const std::vector<double>& predicted)
    {
        Fit f;
        double num = 0.0, den = 0.0;
        const std::size_t count = std::min(measured.size(), predicted.size());
        for (std::size_t i = 0; i < count; ++i)
        {
            num += measured[i] * predicted[i];
            den += predicted[i] * predicted[i];
        }
        f.gain = (den > 0.0) ? num / den : 0.0;
        double worst = 0.0, span = 0.0;
        for (std::size_t i = 0; i < count; ++i)
        {
            worst =
                std::max(worst, std::abs(measured[i] - f.gain * predicted[i]));
            span = std::max(span, std::abs(f.gain * predicted[i]));
        }
        f.residual = worst;
        f.rel = (span > 0.0) ? worst / span : 0.0;
        return f;
    }

    // What a ciphertext actually holds, without asking it to be anything. A
    // slot decode has no range check, so it answers "noise or signal" even
    // when extract_coefficients refuses the same plaintext -- which is the one
    // distinction worth having when a transform comes back wrong.
    void Probe(const char* label, heongpu::Ciphertext<S>& ct,
               heongpu::HEDecryptor<S>& decryptor,
               heongpu::HEEncoder<S>& encoder,
               heongpu::HEContext<S>& context)
    {
        heongpu::Plaintext<S> p(context);
        decryptor.decrypt(p, ct);
        std::vector<double> v;
        encoder.decode(v, p);
        double worst = 0.0;
        for (double e : v)
            worst = std::max(worst, std::abs(e));
        std::cout << "[island]   probe " << std::left << std::setw(24) << label
                  << std::right << " depth " << std::setw(3) << ct.depth()
                  << "  scale 2^" << std::fixed << std::setprecision(2)
                  << std::log2(ct.scale()) << "  max|slot| " << std::scientific
                  << std::setprecision(3) << worst << std::endl;
    }

    void Report(const char* label, const Fit& f)
    {
        std::cout << "[island] " << std::left << std::setw(38) << label
                  << std::right << " gain " << std::fixed
                  << std::setprecision(6) << std::setw(10) << f.gain
                  << "   worst " << std::scientific << std::setprecision(2)
                  << f.residual << "   rel " << f.rel << std::endl;
    }
} // namespace

int main()
{
    const int log_n = EnvInt("HEONGPU_ISLAND_LOGN", 12);
    const int d = EnvInt("HEONGPU_ISLAND_D", 64);
    const int limbs = EnvInt("HEONGPU_ISLAND_LIMBS", 36);
    const int prime_bits = EnvInt("HEONGPU_ISLAND_PRIME_BITS", 50);
    const int ctos = EnvInt("HEONGPU_ISLAND_CTOS", 3);
    const int stoc = EnvInt("HEONGPU_ISLAND_STOC", 3);
    const int taylor = EnvInt("HEONGPU_ISLAND_TAYLOR", 11);
    const int degree = EnvInt("HEONGPU_ISLAND_DEGREE", 15);
    const int align_default = EnvInt("HEONGPU_ISLAND_ALIGN", 1);
    const bool run_trip = EnvInt("HEONGPU_ISLAND_TRIP", 1) != 0;
    const bool run_norm = EnvInt("HEONGPU_ISLAND_NORM", 1) != 0;
    const bool run_sweep = EnvInt("HEONGPU_ISLAND_SWEEP", 0) != 0;

    const int n = 1 << log_n;
    const int k = n / d;
    const int step = k / 2;
    const int half = n / 2;
    const double scale = std::pow(2.0, prime_bits);

    std::cout << "[island] ring   : logN " << log_n << " (N = " << n
              << "), d = " << d << ", blocks k/2 = " << step
              << ", channels N/2 = " << half << std::endl;
    std::cout << "[island] chain  : " << limbs << " limbs (60 + " << (limbs - 1)
              << " x " << prime_bits << "), scale 2^" << prime_bits
              << std::endl;
    std::cout << "[island] boot   : CtoS " << ctos << ", StoC " << stoc
              << ", taylor " << taylor << std::endl;

    std::vector<int> q_bits(limbs, prime_bits);
    q_bits.front() = 60;
    std::vector<int> p_bits(limbs, 60);

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(static_cast<std::size_t>(n));
    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context, 16);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder(context);

    heongpu::BatchMatrixLayout layout(n, d);
    Rect op(context, encoder, layout, scale);

    heongpu::BootstrappingConfig boot_config(ctos, stoc, taylor, true);
    op.arith().generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

    std::set<int> shift_set;
    for (int r : op.rotation_indices())
        shift_set.insert(r);
    for (int r : op.arith().bootstrapping_key_indexs())
        shift_set.insert(r);
    std::vector<int> shifts(shift_set.begin(), shift_set.end());
    std::cout << "[island] keys   : " << shifts.size()
              << " rotation indices (crossing + bootstrap union)" << std::endl;

    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);

    std::mt19937_64 rng(20260808u);
    std::uniform_real_distribution<double> dist(-0.4, 0.4);
    std::vector<double> values(static_cast<std::size_t>(d) * half);
    for (double& v : values)
        v = dist(rng);

    std::cout << "[island] ------------------------------------------------"
              << std::endl;

    // -------------------------------------------------------------------
    // 1. The round trip.
    // -------------------------------------------------------------------
    if (run_trip)
    {
        heongpu::llama::RectActivation xb =
            op.encrypt(values, half, encryptor, scale);
        cudaDeviceSynchronize();
        const auto t0 = Clock::now();
        std::vector<heongpu::Ciphertext<S>> island =
            op.refresh_to_slots(xb, galois, relin);
        cudaDeviceSynchronize();
        const auto t1 = Clock::now();
        const int slot_depth = island.front().depth();
        heongpu::llama::RectActivation back =
            op.slots_to_rect(island, half, galois, align_default);
        cudaDeviceSynchronize();
        const auto t2 = Clock::now();

        Probe("island slots", island.front(), decryptor, encoder, context);
        Probe("after slots_to_rect", back.column.front(), decryptor, encoder,
              context);

        std::vector<double> got_b;
        try
        {
            got_b = op.decrypt(back, decryptor, scale);
        }
        catch (const std::exception& e)
        {
            std::cout << "[island]   slots_to_rect did not decode as a rect "
                         "column: "
                      << e.what() << std::endl;
            got_b.assign(values.size(), 0.0);
        }

        // THE ALIGNMENT SWEEP. The encoded StoC diagonals live at one level
        // and multiply_matrix slices them by the ciphertext's own limb count,
        // with no check: the wrong alignment is noise and not an error. Which
        // one is right depends on how the bootstrapping parameters were laid
        // out, so it is measured here rather than reasoned about.
        for (int drop = 0; run_sweep && drop <= 2; ++drop)
        {
            if (drop == align_default)
                continue;
            heongpu::llama::RectActivation xs =
                op.encrypt(values, half, encryptor, scale);
            std::vector<heongpu::Ciphertext<S>> sl =
                op.refresh_to_slots(xs, galois, relin);
            heongpu::llama::RectActivation bk =
                op.slots_to_rect(sl, half, galois, drop);
            const std::string tag = "align_drop " + std::to_string(drop);
            Probe(tag.c_str(), bk.column.front(), decryptor, encoder, context);
            try
            {
                const std::vector<double> g =
                    op.decrypt(bk, decryptor, scale);
                Report((tag + " vs input").c_str(), FitTo(g, values));
            }
            catch (const std::exception& e)
            {
                std::cout << "[island]   " << tag << " did not decode: "
                          << e.what() << std::endl;
            }
        }

        // The floor: a plain bootstrap of the same data, no crossing at all.
        // The island cannot be more accurate than this and must not be less.
        heongpu::llama::RectActivation xa =
            op.encrypt(values, half, encryptor, scale);
        cudaDeviceSynchronize();
        const auto t3 = Clock::now();
        op.bootstrap(xa, "floor", galois, relin);
        cudaDeviceSynchronize();
        const auto t4 = Clock::now();
        Probe("bootstrap alone", xa.column.front(), decryptor, encoder,
              context);
        const std::vector<double> got_a = op.decrypt(xa, decryptor, scale);

        const double ms_in =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms_out =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double ms_boot =
            std::chrono::duration<double, std::milli>(t4 - t3).count();

        std::cout << "[island] 1. ROUND TRIP, " << d << " columns"
                  << std::endl;
        std::cout << "[island]   depth  after refresh_to_slots " << slot_depth
                  << ", after slots_to_rect "
                  << back.column.front().depth() << "   -> the closing map "
                  << "costs " << (back.column.front().depth() - slot_depth)
                  << std::endl;
        std::cout << "[island]   depth  a plain bootstrap leaves "
                  << xa.column.front().depth() << ", and the crossing that "
                  << "would follow costs 2 more" << std::endl;
        std::cout << "[island]   time   refresh_to_slots " << std::fixed
                  << std::setprecision(1) << ms_in << " ms, slots_to_rect "
                  << ms_out << " ms, total " << (ms_in + ms_out)
                  << " ms; bootstrap alone " << ms_boot << " ms" << std::endl;

        Report("round trip vs input", FitTo(got_b, values));
        Report("bootstrap alone vs input (floor)", FitTo(got_a, values));

    }

    // -------------------------------------------------------------------
    // 2. The index map, both readings, checked against the decoded slots.
    // -------------------------------------------------------------------
    if (run_trip)
    {
        heongpu::llama::RectActivation xi =
            op.encrypt(values, half, encryptor, scale);
        std::vector<heongpu::Ciphertext<S>> island =
            op.refresh_to_slots(xi, galois, relin);

        heongpu::llama::RectActivation xn =
            op.encrypt(values, half, encryptor, scale);
        std::vector<heongpu::Ciphertext<S>> natural = op.to_slots(xn, galois);

        std::cout << "[island] 2. INDEX MAP" << std::endl;

        for (int j = 0; j < std::min(2, d); ++j)
        {
            // What island_slot() says should be where, for column j.
            std::vector<double> want_island(static_cast<std::size_t>(half), 0.0);
            std::vector<double> want_natural(static_cast<std::size_t>(half),
                                             0.0);
            for (int b = 0; b < step; ++b)
            {
                for (int u = 0; u < d; ++u)
                {
                    const double v =
                        values[static_cast<std::size_t>(u) * half + b * d + j];
                    want_island[static_cast<std::size_t>(
                        op.island_slot(b, u, true))] = v;
                    want_natural[static_cast<std::size_t>(
                        op.island_slot(b, u, false))] = v;
                }
            }

            heongpu::Plaintext<S> pi(context);
            decryptor.decrypt(pi, island[static_cast<std::size_t>(j)]);
            std::vector<double> got_i;
            encoder.decode(got_i, pi);
            got_i.resize(static_cast<std::size_t>(half));

            heongpu::Plaintext<S> pn(context);
            decryptor.decrypt(pn, natural[static_cast<std::size_t>(j)]);
            std::vector<double> got_n;
            encoder.decode(got_n, pn);
            got_n.resize(static_cast<std::size_t>(half));

            const std::string a =
                "col " + std::to_string(j) + " island, island_slot";
            const std::string b =
                "col " + std::to_string(j) + " island, natural_slot";
            const std::string c =
                "col " + std::to_string(j) + " to_slots, natural_slot";
            Report(a.c_str(), FitTo(got_i, want_island));
            Report(b.c_str(), FitTo(got_i, want_natural));
            Report(c.c_str(), FitTo(got_n, want_natural));
        }
    }

    if (!run_norm)
        return 0;

    // -------------------------------------------------------------------
    // 3. A whole RMSNorm, both ways, against the plaintext answer.
    // -------------------------------------------------------------------
    {
        const int channels = half;

        std::vector<double> weight(static_cast<std::size_t>(channels));
        for (int c = 0; c < channels; ++c)
            weight[static_cast<std::size_t>(c)] =
                0.8 + 0.4 * static_cast<double>(c % 7) / 6.0;

        std::vector<double> reference(values.size());
        double sum_lo = 1e300, sum_hi = 0.0;
        for (int i = 0; i < d; ++i)
        {
            double s = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                const double v =
                    values[static_cast<std::size_t>(i) * channels + c];
                s += v * v;
            }
            sum_lo = std::min(sum_lo, s);
            sum_hi = std::max(sum_hi, s);
            const double inv = 1.0 / std::sqrt(s / channels + 1e-5);
            for (int c = 0; c < channels; ++c)
            {
                reference[static_cast<std::size_t>(i) * channels + c] =
                    values[static_cast<std::size_t>(i) * channels + c] * inv *
                    weight[static_cast<std::size_t>(c)];
            }
        }
        // The fit needs an interval, not the exact range: widen it so the
        // measurement is of the two paths and not of a fit sitting on its
        // own endpoint.
        const double lo = sum_lo * 0.9;
        const double hi = sum_hi * 1.1;
        std::cout << "[island] 3. RMSNorm over " << channels
                  << " channels, summed square in [" << std::fixed
                  << std::setprecision(2) << sum_lo << ", " << sum_hi
                  << "], fitted on [" << lo << ", " << hi << "]" << std::endl;

        Rect::RectRMSNormConfig cfg;
        cfg.eps = 1e-5;
        cfg.sum_lo = lo;
        cfg.sum_hi = hi;
        cfg.degree = degree;
        cfg.newton_iterations = 0;
        cfg.fold_mean_into_fit = true;
        cfg.fold_affine_into_mask = true;

        // Path B first, because it is the one that has room. Path A can run
        // out of chain at a limb count where B does not -- that IS the six
        // levels of headroom, and it should be reported as a result rather
        // than taking the other measurement down with it.
        {
            heongpu::llama::RectActivation xb =
                op.encrypt(values, half, encryptor, scale);
            Rect::RectRMSNormConfig b = cfg;
            b.fused_refresh = true;
            cudaDeviceSynchronize();
            const auto t0 = Clock::now();
            heongpu::llama::RectActivation yb =
                op.rms_norm(xb, weight, b, galois, relin, &galois);
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();
            const std::vector<double> got = op.decrypt(yb, decryptor, scale);
            std::cout << "[island]   path B  depth out "
                      << yb.column.front().depth() << ", "
                      << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double, std::milli>(t1 - t0)
                             .count()
                      << " ms" << std::endl;
            Report("path B (refresh_to_slots, norm, back)",
                   FitTo(got, reference));
        }

        // Path A: refresh, then cross, then norm, then cross back. It starts
        // the norm six levels deeper, so at a chain this short it may simply
        // not fit -- which is the finding, not an accident, and is caught and
        // reported rather than allowed to abort the run.
        try
        {
            heongpu::llama::RectActivation xa =
                op.encrypt(values, half, encryptor, scale);
            Rect::RectRMSNormConfig a = cfg;
            a.fused_refresh = false;
            cudaDeviceSynchronize();
            const auto t0 = Clock::now();
            op.bootstrap(xa, "path_a.refresh", galois, relin);
            heongpu::llama::RectActivation ya =
                op.rms_norm(xa, weight, a, galois, relin, &galois);
            cudaDeviceSynchronize();
            const auto t1 = Clock::now();
            const std::vector<double> got = op.decrypt(ya, decryptor, scale);
            std::cout << "[island]   path A  depth out "
                      << ya.column.front().depth() << ", "
                      << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double, std::milli>(t1 - t0)
                             .count()
                      << " ms" << std::endl;
            Report("path A (bootstrap, cross, norm, cross)",
                   FitTo(got, reference));
        }
        catch (const std::exception& e)
        {
            std::cout << "[island]   path A did not fit in " << limbs
                      << " limbs: " << e.what() << std::endl;
        }

        // What the island had to build to close at the level the norm left
        // it at. One entry means one level was reached, which is the case a
        // real circuit is in; several would mean the diagonals are being
        // re-encoded and the memory is worth knowing about.
        std::cout << "[island]   transform contexts built at levels:";
        for (int l : op.arith().slot_transform_levels())
            std::cout << " " << l;
        std::cout << std::endl;
    }

    std::cout << "[island] ------------------------------------------------"
              << std::endl;
    std::cout << "[island] a rel near 1e-3 is a bootstrap that worked; a rel "
                 "near 1 is the wrong reading."
              << std::endl;
    return 0;
}
