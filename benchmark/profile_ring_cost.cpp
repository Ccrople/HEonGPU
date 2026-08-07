// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DOES RING SWITCHING ACTUALLY PAY FOR THE TWO KANG PRODUCTS?
// ===========================================================
//
// Sylph runs its batch CCMM at N = 4096 and its PCMM lower still, so the
// received wisdom is that a matrix product wants the smallest ring it can get.
// That is an argument about total cost, and total cost is the product of two
// things that move in opposite directions when N falls: a call gets cheaper,
// and you need more calls. This target measures both and divides.
//
// The unit is one scalar multiply-accumulate of the logical matrix product, so
// the number printed is directly comparable across rings. A call at layout
// (N, d) contracts over k/2 = N/2d blocks of d x d, which is N*d^2/2 MACs
// either way, for both algorithms. Fewer slots per ciphertext at a small N
// therefore shows up as fewer MACs per call and the division handles it.
//
// WHAT THE CODE SAYS BEFORE ANY MEASUREMENT
// -----------------------------------------
// The two algorithms have completely different key-switch counts, and that is
// the whole story:
//
//   ccmm             three CMTs at layout (N, d), d automorphisms each, plus
//                    d relinearisations.                 ~4d key switches.
//   rectangular_pcmm one CMT at layout (N, N/2) -- k = 2, so its automorphisms
//                    run over the whole rotation group -- plus one at (N, d).
//                                                        ~N/2 key switches.
//
// A key switch on a ring-N ciphertext costs O(N log N) per limb pair. So per
// MAC:
//
//   ccmm             4d * N log N / (N d^2 / 2)  =  8 log N / d
//   rect pcmm        (N/2) * N log N / (N d^2/2) =  N log N / d^2
//
// The CCMM's per-MAC cost depends on the ring only through log N: halving the
// ring degree four times over should buy about 16/12 = 1.33x, and no more.
// The rectangular PCMM's grows LINEARLY in N, so the same four halvings should
// buy about 16 * 1.33 = 21x. If that holds, "ring switching helps matrix
// multiplication" is false as stated -- it helps one of these two a great deal
// and the other barely at all, and a design that pays for a ring switch to get
// the CCMM down has bought almost nothing.
//
// The limb count is the other axis and it is not a fixed multiplier either.
// With one special prime the decomposition has dnum = Q_size groups, so a key
// switch costs O(Q_size * (Q_size + 1)) and the chain length enters
// QUADRATICALLY. A low-ring island that also runs at three limbs instead of
// twenty is where the real factor would come from -- but that is an argument
// for scheduling the products at the bottom of the chain, which costs nothing
// and needs no ring switch at all. The sweep separates the two by measuring a
// matched-limb row and a realistic-limb row.
//
// WHAT CANNOT BE MEASURED HERE, AND WHY THAT IS ITSELF THE ANSWER
// ---------------------------------------------------------------
// rectangular_pcmm needs N/2 - 1 Galois keys. At logN 16 and five limbs that
// is 32767 * 5 * 65536 * 2 * 8 bytes = 172 GB of key material, so the high-ring
// end of the rect sweep does not exist on one card at any speed. The sweep
// stops where the keys stop fitting and says so.
//
//   HEONGPU_RING_LOGN     comma-separated ring degrees      12,13,14
//   HEONGPU_RING_LIMBS    comma-separated chain lengths     4
//   HEONGPU_RING_D        block size, the head dimension    128
//   HEONGPU_RING_REPS     timed repetitions per point       3
//   HEONGPU_RING_KEY_CAP  skip rect pcmm above this many
//                         Galois keys                       20000
//   HEONGPU_RING_CCMM_ONLY  1 to skip rect pcmm entirely    0

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
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

    std::vector<int> EnvList(const char* name, const std::vector<int>& fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        std::vector<int> out;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            if (!item.empty())
                out.push_back(std::atoi(item.c_str()));
        }
        return out.empty() ? fallback : out;
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

    struct Point
    {
        int log_n = 0;
        int limbs = 0;
        double ccmm_ms = 0.0; // one call
        double rect_ms = 0.0; // one call, negative if not measured
        // The two algorithms do DIFFERENT amounts of work per call and the
        // difference is a factor of k/2, so one shared figure would be wrong
        // for one of them. ccmm multiplies two d x d matrix encryptions, k/2
        // of them batched: (k/2) * d^3 = N*d^2/2. rectangular_pcmm multiplies
        // a d x (N/2) encryption by an (N/2) x (N/2) plaintext, which is
        // (k/2)^2 blocks of d x d: (k/2)^2 * d^3 = N^2*d/4.
        double ccmm_macs = 0.0;
        double rect_macs = 0.0;
        int galois_keys = 0;
    };

    // Wall time of one call, best of reps. Best rather than mean: every one of
    // these is a long sequence of synchronous launches on an otherwise idle
    // card, so the spread is interference and the minimum is the signal.
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
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            best = std::min(best, ms);
        }
        return best;
    }
} // namespace

int main()
{
    const std::vector<int> log_ns = EnvList("HEONGPU_RING_LOGN", {12, 13, 14});
    const std::vector<int> limb_list = EnvList("HEONGPU_RING_LIMBS", {4});
    const int d = EnvInt("HEONGPU_RING_D", 128);
    const int reps = EnvInt("HEONGPU_RING_REPS", 3);
    const int key_cap = EnvInt("HEONGPU_RING_KEY_CAP", 20000);
    const bool ccmm_only = EnvInt("HEONGPU_RING_CCMM_ONLY", 0) != 0;

    std::cout << "[ring] d = " << d << ", reps = " << reps
              << ", rect pcmm skipped above " << key_cap << " Galois keys"
              << std::endl;
    std::cout << "[ring] work per call: ccmm (k/2)*d^3 = N*d^2/2, rect pcmm "
                 "(k/2)^2*d^3 = N^2*d/4"
              << std::endl;

    std::vector<Point> points;

    for (int limbs : limb_list)
    {
        for (int log_n : log_ns)
        {
            const int n = 1 << log_n;
            if (d >= n / 2)
            {
                std::cout << "[ring] logN " << log_n << ": d = " << d
                          << " leaves no block axis, skipped" << std::endl;
                continue;
            }

            Point p;
            p.log_n = log_n;
            p.limbs = limbs;
            p.ccmm_macs = 0.5 * static_cast<double>(n) * d * d;
            p.rect_macs =
                0.25 * static_cast<double>(n) * static_cast<double>(n) * d;

            const int prime_bits = 40;
            const double scale = std::pow(2.0, prime_bits);
            std::vector<int> q_bits(limbs, prime_bits);
            // One special prime, which is HEonGPU's KEYSWITCHING_METHOD_I. It
            // has to be at least as large as the biggest single Q prime,
            // because coefficient_validator groups Q into chunks of P_size and
            // demands each chunk fit inside P.
            std::vector<int> p_bits(1, prime_bits + 5);

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
            heongpu::HEEncoder<S> encoder(context);

            heongpu::BatchMatrixLayout layout(n, d);
            heongpu::llama::Llama3BatchOperator batch(context, encoder, layout,
                                                      scale);
            heongpu::llama::Llama3RectOperator rect(context, encoder, layout,
                                                    scale);

            const int half = n / 2;
            const int blocks = layout.batch;

            // The CCMM needs only the CMT indices at (N, d) -- d of them. The
            // rect PCMM needs the whole rotation group. Keeping the two key
            // sets apart is what lets the CCMM sweep reach rings the rect one
            // cannot.
            std::vector<int> ccmm_shifts = batch.product_rotation_indices();
            heongpu::Galoiskey<S> ccmm_galois(context, ccmm_shifts);
            keygen.generate_galois_key(ccmm_galois, secret);
            heongpu::Relinkey<S> relin(context);
            keygen.generate_relin_key(relin, secret);

            std::mt19937_64 rng(20260807u + log_n);
            const double amp = 1.0 / std::sqrt(static_cast<double>(d));

            // ---------------------------------------------------------------
            // Algorithm 4, batch CCMM
            // ---------------------------------------------------------------
            {
                std::vector<std::vector<double>> ba(blocks), bb(blocks);
                for (int t = 0; t < blocks; ++t)
                {
                    ba[t] = RandomMatrix(static_cast<std::size_t>(d) * d, amp,
                                         rng);
                    bb[t] = RandomMatrix(static_cast<std::size_t>(d) * d, amp,
                                         rng);
                }
                heongpu::llama::BatchActivation A =
                    batch.encrypt(ba, d, d, encryptor, scale);
                heongpu::llama::BatchActivation B =
                    batch.encrypt(bb, d, d, encryptor, scale);

                p.ccmm_ms = BestMs(
                    [&]()
                    {
                        heongpu::llama::BatchActivation C = batch.matmul(
                            A, B, "ring.ccmm", ccmm_galois, relin);
                    },
                    reps);
            }

            // ---------------------------------------------------------------
            // Algorithm 5, rectangular PCMM
            // ---------------------------------------------------------------
            const int rect_keys =
                static_cast<int>(rect.rotation_indices().size());
            p.galois_keys = rect_keys;

            if (ccmm_only || rect_keys > key_cap)
            {
                p.rect_ms = -1.0;
            }
            else
            {
                std::vector<int> rect_shifts = rect.rotation_indices();
                heongpu::Galoiskey<S> rect_galois(context, rect_shifts);
                keygen.generate_galois_key(rect_galois, secret);

                const std::vector<double> x = RandomMatrix(
                    static_cast<std::size_t>(d) * half, 0.5, rng);
                const std::vector<double> w = RandomMatrix(
                    static_cast<std::size_t>(half) * half, amp, rng);
                heongpu::llama::RectActivation X =
                    rect.encrypt(x, half, encryptor, scale);

                p.rect_ms = BestMs(
                    [&]()
                    {
                        heongpu::llama::RectActivation Y = rect.project(
                            X, w, half, half, "ring.rect", rect_galois);
                    },
                    reps);
            }

            points.push_back(p);

            std::cout << "[ring] logN " << std::setw(2) << log_n << "  limbs "
                      << std::setw(2) << limbs << "  ccmm " << std::fixed
                      << std::setprecision(2) << std::setw(9) << p.ccmm_ms
                      << " ms  rect ";
            if (p.rect_ms < 0.0)
                std::cout << "     --    (" << rect_keys << " keys)";
            else
                std::cout << std::setw(9) << p.rect_ms << " ms";
            std::cout << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // The verdict: cost per unit of logical matrix work, normalised to the
    // smallest ring measured, per limb count.
    // -----------------------------------------------------------------------
    std::cout << std::endl;
    std::cout
        << "[ring] cost per giga-MAC of the logical product (lower is better)"
        << std::endl;
    std::cout << "[ring] "
                 "logN limbs |    ccmm ns/MAC  rel |    rect ns/MAC  rel"
              << std::endl;

    for (int limbs : limb_list)
    {
        double ccmm_base = 0.0;
        double rect_base = 0.0;
        for (const Point& p : points)
        {
            if (p.limbs != limbs)
                continue;
            if (ccmm_base == 0.0)
                ccmm_base = p.ccmm_ms * 1e6 / p.ccmm_macs;
            if (rect_base == 0.0 && p.rect_ms > 0.0)
                rect_base = p.rect_ms * 1e6 / p.rect_macs;
        }

        for (const Point& p : points)
        {
            if (p.limbs != limbs)
                continue;
            const double c = p.ccmm_ms * 1e6 / p.ccmm_macs;
            std::cout << "[ring]   " << std::setw(2) << p.log_n << "   "
                      << std::setw(2) << p.limbs << "  | " << std::fixed
                      << std::setprecision(4) << std::setw(14) << c << " "
                      << std::setprecision(2) << std::setw(5)
                      << (ccmm_base > 0.0 ? c / ccmm_base : 0.0) << "x | ";
            if (p.rect_ms > 0.0)
            {
                const double r = p.rect_ms * 1e6 / p.rect_macs;
                std::cout << std::setprecision(4) << std::setw(14) << r << " "
                          << std::setprecision(2) << std::setw(5)
                          << (rect_base > 0.0 ? r / rect_base : 0.0) << "x";
            }
            else
            {
                std::cout << "            --       ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "[ring] read the rel columns as the price of NOT ring "
                 "switching down to the first row."
              << std::endl;
    return 0;
}
