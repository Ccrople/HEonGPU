// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// IS THE RECT -> SLOT CROSSING REALLY THE BOOTSTRAP'S CoeffToSlot?
// ================================================================
//
// The claim under test, which was an argument on paper and is a measurement
// here:
//
//   A rect column's plaintext polynomial is supported on R_N coefficients
//   0..N/2-1, with coefficient i + d*t holding token i of block t. The module's
//   crossing to slot form is a d-point row bridge along the token axis followed
//   by a (k/2)-point block_map along the block axis. That composition is a
//   factorisation of the N/2-point DFT -- which is the transform a bootstrap
//   already evaluates, as CoeffToSlot, on the way in.
//
// If the claim holds, then bootstrap() followed by to_slots() evaluates that
// same DFT three times: forwards inside the bootstrap, backwards inside the
// bootstrap as SlotToCoeff, and forwards again as the crossing. Two of the
// three are avoidable, and coeff_to_slot_bootstrapping avoids them by simply
// not running stage 4.
//
// This target does not argue about twiddle conventions, bit reversal or
// conjugate-symmetry factors. It states a prediction in terms the decoder can
// check and then checks it:
//
//   PREDICTION  slot c of bootstrap_to_slots(column j) decodes to
//               X[i][t*d + j], where i = c mod d and t = c div d.
//
// A permutation error, a missing 1/N, a bit-reversed axis or a swapped
// conjugate half would each break that prediction in a different and visible
// way, so the fit statistics below distinguish them:
//
//   * the best-fit scalar between prediction and measurement, so a constant
//     normalisation shows up as a factor rather than as failure;
//   * the residual after that scalar, against the ~1e-3 a bootstrap costs;
//   * the same residual under the identity relabelling and under
//     slot_reading_permutation(), so which of the two orders is which is
//     answered rather than assumed.
//
// Path A -- bootstrap() then to_slots() -- is measured alongside, in levels and
// in milliseconds, because the point of the exercise is the difference.
//
//   HEONGPU_B2S_LOGN         ring degree                        12
//   HEONGPU_B2S_D            block size                         64
//   HEONGPU_B2S_LIMBS        chain length                       36
//   HEONGPU_B2S_PRIME_BITS   scale prime size                   50
//   HEONGPU_B2S_CTOS         CoeffToSlot pieces                  3
//   HEONGPU_B2S_STOC         SlotToCoeff pieces                  3
//   HEONGPU_B2S_TAYLOR       EvalMod Taylor degree              11
//   HEONGPU_B2S_COLUMNS      how many columns to check           2

#include <heongpu/heongpu.cuh>

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

using S = heongpu::Scheme::CKKS;
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
        double gain = 0.0;     // least-squares a in measured ~ a * predicted
        double residual = 0.0; // max |measured - a * predicted|
        double rel = 0.0;      // residual / max |predicted * a|
    };

    // Least squares through the origin, then the worst absolute residual. A
    // single scalar is the only freedom allowed: anything else the transform
    // might have done to the data -- a permutation, a per-slot twiddle, a
    // conjugate fold -- survives as residual and is meant to.
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
            worst = std::max(worst,
                             std::abs(measured[i] - f.gain * predicted[i]));
            span = std::max(span, std::abs(f.gain * predicted[i]));
        }
        f.residual = worst;
        f.rel = (span > 0.0) ? worst / span : 0.0;
        return f;
    }

    void Report(const char* label, const Fit& f)
    {
        std::cout << "[b2s]   " << std::left << std::setw(34) << label
                  << std::right << " gain " << std::fixed
                  << std::setprecision(6) << std::setw(10) << f.gain
                  << "   worst " << std::scientific << std::setprecision(2)
                  << f.residual << "   rel " << f.rel << std::endl;
    }
} // namespace

int main()
{
    const int log_n = EnvInt("HEONGPU_B2S_LOGN", 12);
    const int d = EnvInt("HEONGPU_B2S_D", 64);
    const int limbs = EnvInt("HEONGPU_B2S_LIMBS", 36);
    const int prime_bits = EnvInt("HEONGPU_B2S_PRIME_BITS", 50);
    const int ctos = EnvInt("HEONGPU_B2S_CTOS", 3);
    const int stoc = EnvInt("HEONGPU_B2S_STOC", 3);
    const int taylor = EnvInt("HEONGPU_B2S_TAYLOR", 11);
    const int columns = EnvInt("HEONGPU_B2S_COLUMNS", 2);

    const int n = 1 << log_n;
    const int k = n / d;
    const int blocks = k / 2;
    const int half = n / 2;
    const double scale = std::pow(2.0, prime_bits);

    std::cout << "[b2s] ring   : logN " << log_n << " (N = " << n
              << "), d = " << d << ", blocks k/2 = " << blocks
              << ", channels N/2 = " << half << std::endl;
    std::cout << "[b2s] chain  : " << limbs << " limbs (60 + " << (limbs - 1)
              << " x " << prime_bits << "), scale 2^" << prime_bits
              << std::endl;
    std::cout << "[b2s] boot   : CtoS " << ctos << ", StoC " << stoc
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
    const std::vector<int> shifts(shift_set.begin(), shift_set.end());
    std::cout << "[b2s] keys   : " << shifts.size()
              << " rotation indices (crossing + bootstrap union)" << std::endl;

    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);

    // A smooth, well-separated pattern. Smooth because the bootstrap's message
    // range is what it is; well separated because a residual has to be
    // attributable to a wrong slot and not to two neighbours happening to
    // agree.
    std::mt19937_64 rng(20260807u);
    std::uniform_real_distribution<double> dist(-0.4, 0.4);
    std::vector<double> values(static_cast<std::size_t>(d) * half);
    for (double& v : values)
        v = dist(rng);

    heongpu::llama::RectActivation x0 =
        op.encrypt(values, half, encryptor, scale);
    const int start_depth = x0.column.front().depth();
    std::cout << "[b2s] input  : depth " << start_depth << ", "
              << (limbs - start_depth) << " limbs live" << std::endl;
    std::cout << "[b2s] -----------------------------------------------------"
              << std::endl;

    const std::vector<int> perm = op.slot_reading_permutation();

    const int checked = std::min(columns, d);
    for (int j = 0; j < checked; ++j)
    {
        // PREDICTION for column j: coefficient c = i + d*t carries X[i][t*d+j],
        // and CoeffToSlot reads coefficient c into slot c.
        std::vector<double> predicted(static_cast<std::size_t>(half));
        for (int t = 0; t < blocks; ++t)
        {
            for (int i = 0; i < d; ++i)
            {
                predicted[static_cast<std::size_t>(i) + d * t] =
                    values[static_cast<std::size_t>(i) * half + t * d + j];
            }
        }

        // -------------------------------------------------------------------
        // Path B, the new one: ModRaise -> CoeffToSlot -> EvalMod, stop.
        // -------------------------------------------------------------------
        heongpu::llama::RectActivation xb =
            op.encrypt(values, half, encryptor, scale);
        cudaDeviceSynchronize();
        const auto tb0 = Clock::now();
        heongpu::Ciphertext<S> slots_b =
            op.bootstrap_to_slots(xb.column[j], galois, relin);
        cudaDeviceSynchronize();
        const auto tb1 = Clock::now();
        const double ms_b =
            std::chrono::duration<double, std::milli>(tb1 - tb0).count();

        heongpu::Plaintext<S> pb(context);
        decryptor.decrypt(pb, slots_b);
        std::vector<double> got_b;
        encoder.decode(got_b, pb);
        got_b.resize(static_cast<std::size_t>(half));

        // -------------------------------------------------------------------
        // Path A, today's: a full bootstrap, then the crossing.
        // -------------------------------------------------------------------
        heongpu::llama::RectActivation xa =
            op.encrypt(values, half, encryptor, scale);
        cudaDeviceSynchronize();
        const auto ta0 = Clock::now();
        for (auto& c : xa.column)
            c = op.bootstrap(c, galois, relin);
        std::vector<heongpu::Ciphertext<S>> slots_a = op.to_slots(xa, galois);
        cudaDeviceSynchronize();
        const auto ta1 = Clock::now();
        const double ms_a =
            std::chrono::duration<double, std::milli>(ta1 - ta0).count();

        heongpu::Plaintext<S> pa(context);
        decryptor.decrypt(pa, slots_a.front());
        std::vector<double> got_a;
        encoder.decode(got_a, pa);
        got_a.resize(static_cast<std::size_t>(half));

        std::cout << "[b2s] column " << j << std::endl;
        std::cout << "[b2s]   depth  path A (bootstrap + to_slots) "
                  << slots_a.front().depth() << ", path B (bootstrap_to_slots) "
                  << slots_b.depth() << "   -> B keeps "
                  << (slots_a.front().depth() - slots_b.depth())
                  << " more levels" << std::endl;
        std::cout << "[b2s]   time   path A " << std::fixed
                  << std::setprecision(1) << ms_a << " ms for all " << d
                  << " columns + crossing, path B " << ms_b
                  << " ms for one column" << std::endl;

        // The prediction, straight through.
        Report("B vs prediction, identity", FitTo(got_b, predicted));

        // The prediction under the stride transpose, which is what would hold
        // if the two axes were the other way round.
        std::vector<double> permuted(static_cast<std::size_t>(half), 0.0);
        for (int c = 0; c < half; ++c)
            permuted[static_cast<std::size_t>(perm[c])] = predicted[c];
        Report("B vs prediction, stride perm", FitTo(got_b, permuted));

        // And what path A produced, for the relabelling between the two.
        Report("A vs prediction, identity", FitTo(got_a, predicted));
        Report("A vs prediction, stride perm", FitTo(got_a, permuted));

        std::vector<double> b_perm(static_cast<std::size_t>(half), 0.0);
        for (int c = 0; c < half; ++c)
            b_perm[static_cast<std::size_t>(perm[c])] = got_b[c];
        Report("A vs B, stride perm", FitTo(got_a, b_perm));
        Report("A vs B, identity", FitTo(got_a, got_b));
    }

    std::cout << "[b2s] -----------------------------------------------------"
              << std::endl;
    std::cout << "[b2s] a rel near 1e-3 is a bootstrap that worked; a rel near "
                 "1 is the wrong reading."
              << std::endl;
    return 0;
}
