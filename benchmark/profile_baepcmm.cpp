// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Prices the Bae plaintext-ciphertext product (eprint 2024/1284) against
// Kang's Algorithm 5 (rectangular_pcmm), at the shapes a Llama-3-8B block
// actually asks for.
//
// TWO KINDS OF NUMBER COME OUT OF THIS, AND THEY ARE NOT EQUALLY TRUSTWORTHY.
//
// The COUNTS -- key switches, rotations, Galois keys, levels, modular
// multiply-accumulates -- are exact, derived from the algorithms and the
// shapes, and independent of the machine. They are the answer to "which
// algorithm should this path use".
//
// The TIMES are wall clock on whatever card this runs on. Sicily's three
// A6000s are shared with other users, and a run taken while another job is
// resident measures the other job as much as this one. The header of every
// timing table prints the GPU's utilisation and memory at launch so a reader
// can throw the table away when it deserves it.
//
// Usage:  llama3_baepcmm_profile [--logn 12] [--limbs 13] [--reps 3]
//                                [--shape all|q|ffn|sweep] [--skip-alg5]

#include <heongpu/heongpu.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    using Clock = std::chrono::steady_clock;

    double ms_since(Clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0)
            .count();
    }

    struct Shape
    {
        const char* name;
        int d1; // out_channels: rows of U
        int d2; // in_channels:  cols of U, rows of the encrypted matrix
    };

    // One Llama-3-8B block's seven projections. d3 (the column count of the
    // encrypted matrix) is the token axis and is a separate parameter,
    // because it is exactly the thing this benchmark is about.
    const std::vector<Shape>& block_shapes()
    {
        static const std::vector<Shape> s = {
            {"attn.q", 4096, 4096},   {"attn.k", 1024, 4096},
            {"attn.v", 1024, 4096},   {"attn.o", 4096, 4096},
            {"ffn.gate", 14336, 4096}, {"ffn.up", 14336, 4096},
            {"ffn.down", 4096, 14336},
        };
        return s;
    }

    void print_gpu_state(const char* when)
    {
        std::printf("# GPU at %s:\n", when);
        std::fflush(stdout);
        if (std::system("nvidia-smi --query-gpu=index,utilization.gpu,"
                        "memory.used,memory.total --format=csv,noheader,"
                        "nounits | sed 's/^/#   /'") != 0)
        {
            std::printf("#   (nvidia-smi unavailable)\n");
        }
    }

    // ---------------------------------------------------------------------
    // The count model. Exact; no hardware in it.
    // ---------------------------------------------------------------------

    struct Counts
    {
        double key_switches = 0;
        double rotations = 0;
        double galois_keys = 0;
        double levels = 0;
        double macs = 0; // modular multiply-accumulates, per RNS limb
    };

    /// Bae, Algorithm 2 (or Algorithm 1 at k = 1), for U (d1 x d2) times an
    /// encrypted d2 x d3 matrix at ring degree N.
    Counts bae_counts(int d1, int d2, int d3, int N)
    {
        const int k = N / d3;
        Counts c;
        // Sections 4.1-4.2: the product itself has none of these.
        c.rotations = 0;
        c.galois_keys = 0;
        c.levels = 1;
        // Two GEMMs: a-part is d1 x d2 x N whatever d3 is, b-part d1 x d2 x d3.
        c.macs = static_cast<double>(d1) * d2 * (static_cast<double>(N) + d3);
        // ModDecomp is free. ModPack costs k key switches per output
        // ciphertext, and there are d1*d3/N = d1/k of them, so d1 in total --
        // and exactly zero when k = 1, where ModPack is the identity.
        c.key_switches = (k == 1) ? 0.0 : static_cast<double>(d1);
        return c;
    }

    /// Kang, Algorithm 5, as Llama3RectOperator::project drives it: the
    /// activation is d = layout.d rows by N/2 channels per group, so a
    /// projection is (d2 / (N/2)) x (d1 / (N/2)) calls, each of which is one
    /// Algorithm-1 product plus a CMT at the half layout (N/2 key switches)
    /// and a CMT at the caller's layout (d - 1 rotations).
    Counts alg5_counts(int d1, int d2, int d3, int N, int d)
    {
        const int half = N / 2;
        const int gin = (d2 + half - 1) / half;
        const int gout = (d1 + half - 1) / half;
        const double calls = static_cast<double>(gin) * gout;

        Counts c;
        // The half-layout CMT is N/2 rotations, the layout CMT d - 1.
        c.rotations = calls * (static_cast<double>(half) + (d - 1));
        c.key_switches = c.rotations; // one key switch per rotation
        // N/2 - 1 distinct Galois indices for the half-layout CMT.
        c.galois_keys = static_cast<double>(half - 1);
        c.levels = 1;
        // The Algorithm-1 core underneath. bm_gemm_kernel walks
        //   C[i][j][s] = sum_t A[i][t][s] * P[t][j][s]
        // with i < d entries, j < cols = N/2, t < inner = d and s < k
        // (src/lib/kernel/batchmatrix.cu:360-396), so one call is
        // d * d * (N/2) * k MACs per ciphertext component and there are two
        // components. Dropping the `inner` factor here understates Algorithm
        // 5 by 128x at these shapes and inverts the comparison, so it is
        // spelled out rather than folded.
        c.macs = calls * static_cast<double>(d) * d * half *
                 static_cast<double>(N / d) * 2.0;
        // Algorithm 5 covers only d3 = d tokens per call; scale to d3.
        const double token_calls =
            static_cast<double>(d3) / static_cast<double>(d);
        c.rotations *= token_calls;
        c.key_switches *= token_calls;
        c.macs *= token_calls;
        return c;
    }

    void print_count_table(int N, int d, int d3)
    {
        std::printf("\n");
        std::printf("=== COUNTS per Llama-3-8B block, %d tokens, N = %d, "
                    "Kang d = %d ===\n",
                    d3, N, d);
        std::printf("# Exact. No hardware in these numbers.\n");
        std::printf("%-10s %14s %14s %14s %14s\n", "proj", "bae.ks",
                    "alg5.ks", "bae.macs", "alg5.macs");

        Counts bae_tot, a5_tot;
        for (const auto& s : block_shapes())
        {
            const Counts b = bae_counts(s.d1, s.d2, d3, N);
            const Counts a = alg5_counts(s.d1, s.d2, d3, N, d);
            std::printf("%-10s %14.0f %14.0f %14.3g %14.3g\n", s.name,
                        b.key_switches, a.key_switches, b.macs, a.macs);
            bae_tot.key_switches += b.key_switches;
            bae_tot.macs += b.macs;
            a5_tot.key_switches += a.key_switches;
            a5_tot.macs += a.macs;
            bae_tot.rotations += b.rotations;
            a5_tot.rotations += a.rotations;
        }
        std::printf("%-10s %14.0f %14.0f %14.3g %14.3g\n", "TOTAL",
                    bae_tot.key_switches, a5_tot.key_switches, bae_tot.macs,
                    a5_tot.macs);
        std::printf("# bae rotations %.0f, alg5 rotations %.0f\n",
                    bae_tot.rotations, a5_tot.rotations);
        std::printf("# bae Galois keys 0, alg5 Galois keys %d (N/2 - 1)\n",
                    N / 2 - 1);
        std::printf("# key switches: Bae / Alg5 = %.4f\n",
                    a5_tot.key_switches > 0
                        ? bae_tot.key_switches / a5_tot.key_switches
                        : 0.0);
    }

    void print_width_sweep(int N)
    {
        std::printf("\n=== The shape argument: cost per TOKEN against the "
                    "column count ===\n");
        std::printf("# d3 is the token axis (times the batch, if any). The "
                    "a-part GEMM is\n# d1*d2*N whatever d3 is, so per token "
                    "the product costs d1*d2*(k+1).\n");
        std::printf("%8s %6s %16s %16s %10s\n", "d3", "k", "macs/blk/limb",
                    "macs/token/limb", "vs k=1");

        double base = 0.0;
        for (int d3 = 64; d3 <= N; d3 *= 2)
        {
            const int k = N / d3;
            double macs = 0;
            for (const auto& s : block_shapes())
                macs += bae_counts(s.d1, s.d2, d3, N).macs;
            const double per_token = macs / d3;
            if (d3 == N)
                base = per_token;
            std::printf("%8d %6d %16.4g %16.4g", d3, k, macs, per_token);
            if (base > 0.0)
                std::printf(" %9.2fx", per_token / base);
            std::printf("\n");
        }
        std::printf("# (the last column is filled once d3 = N is reached; "
                    "read it bottom-up)\n");
    }

    // ---------------------------------------------------------------------
    // The measured half
    // ---------------------------------------------------------------------

    struct Fixture
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;
        int n = 0;
        int q_size = 0;

        Fixture(int logn, int limbs)
            : context(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            n = 1 << logn;
            std::vector<int> q_bits;
            q_bits.push_back(60);
            for (int i = 1; i < limbs; ++i)
                q_bits.push_back(40);
            q_size = limbs;

            context->set_poly_modulus_degree(static_cast<size_t>(n));
            context->set_coeff_modulus_bit_sizes(q_bits, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(context,
                                                                     *encoder);
        }
    };

    /// One timed Bae product at (d1, d2, d3). Returns milliseconds, or a
    /// negative number if the shape did not fit.
    double time_bae(Fixture& f, int d1, int d2, int d3, int reps,
                    std::string& note)
    {
        try
        {
            heongpu::HEBaePcmmOperator<S> bae(f.context, d3);
            const int k = bae.k();
            if (d2 % k != 0)
            {
                note = "d2 not a multiple of k";
                return -1.0;
            }

            std::mt19937 rng(7);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            std::vector<double> M(static_cast<size_t>(d2) * d3);
            for (auto& v : M)
                v = dist(rng);
            std::vector<double> U(static_cast<size_t>(d1) * d2);
            for (auto& v : U)
                v = dist(rng);

            auto coeffs = bae.encode_matrix(M, d2, std::pow(2.0, 40));
            std::vector<heongpu::Ciphertext<S>> ct;
            ct.reserve(coeffs.size());
            for (auto& c : coeffs)
            {
                heongpu::Plaintext<S> p(f.context);
                bae.load_coefficients(p, c, std::pow(2.0, 40));
                heongpu::Ciphertext<S> e(f.context);
                f.encryptor->encrypt(e, p);
                ct.push_back(std::move(e));
            }
            std::vector<heongpu::Ciphertext<S>*> in;
            for (auto& c : ct)
                in.push_back(&c);

            const double plain_scale = static_cast<double>(
                f.context->get_key_modulus()[f.q_size - 1].value);
            bae.upload_plaintext(U, d1, d2, 0, plain_scale);

            // Warm up, then time.
            if (k == 1)
            {
                std::vector<heongpu::Ciphertext<S>> out;
                bae.pcmm(out, in, true);
            }
            else
            {
                (void) bae.pcmm_mlwe(in);
            }
            cudaDeviceSynchronize();

            const auto t0 = Clock::now();
            for (int r = 0; r < reps; ++r)
            {
                if (k == 1)
                {
                    std::vector<heongpu::Ciphertext<S>> out;
                    bae.pcmm(out, in, true);
                }
                else
                {
                    (void) bae.pcmm_mlwe(in);
                }
            }
            cudaDeviceSynchronize();
            return ms_since(t0) / reps;
        }
        catch (const std::exception& e)
        {
            note = e.what();
            return -1.0;
        }
    }
} // namespace

int main(int argc, char** argv)
{
    int logn = 12;
    int limbs = 13;
    int reps = 3;
    int device = 0;
    std::string shape = "all";

    for (int i = 1; i < argc; ++i)
    {
        auto next = [&](int& dst)
        { if (i + 1 < argc) dst = std::atoi(argv[++i]); };
        if (!std::strcmp(argv[i], "--logn"))
            next(logn);
        else if (!std::strcmp(argv[i], "--limbs"))
            next(limbs);
        else if (!std::strcmp(argv[i], "--reps"))
            next(reps);
        else if (!std::strcmp(argv[i], "--device"))
            next(device);
        else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc)
            shape = argv[++i];
    }

    cudaSetDevice(device);

    const int N = 1 << logn;
    std::printf("# Bae PCMM profile: N = %d, limbs = %d, reps = %d\n", N,
                limbs, reps);
    print_gpu_state("launch");

    // The counts first: they are the part worth trusting.
    print_count_table(N, /*Kang d=*/128, /*d3 tokens=*/128);
    print_count_table(N, 128, N);
    print_width_sweep(N);

    if (shape == "counts")
        return 0;

    std::printf("\n=== MEASURED, and read the GPU state above before you "
                "quote any of it ===\n");
    std::printf("# A contended card measures the other job as much as this "
                "one.\n");

    Fixture f(logn, limbs);
    print_gpu_state("after context generate");

    std::printf("%-14s %6s %6s %6s %6s %12s %12s\n", "case", "d1", "d2", "d3",
                "k", "ms", "note");

    struct Case
    {
        const char* name;
        int d1, d2, d3;
    };
    std::vector<Case> cases;
    // Small, so that they fit alongside another user's job. The point of the
    // measured half is the SHAPE TREND, not an absolute block time.
    cases.push_back({"wide.k1", 64, 64, N});
    cases.push_back({"wide.k1.big", 128, 128, N});
    cases.push_back({"narrow.k32", 64, 64, N / 32});
    cases.push_back({"narrow.k16", 64, 64, N / 16});
    cases.push_back({"narrow.k8", 64, 64, N / 8});
    cases.push_back({"narrow.k4", 64, 64, N / 4});
    cases.push_back({"narrow.k2", 64, 64, N / 2});

    for (const auto& c : cases)
    {
        std::string note;
        const double ms = time_bae(f, c.d1, c.d2, c.d3, reps, note);
        const int k = (c.d3 > 0) ? N / c.d3 : 0;
        if (ms < 0)
            std::printf("%-14s %6d %6d %6d %6d %12s %12s\n", c.name, c.d1,
                        c.d2, c.d3, k, "-", note.c_str());
        else
            std::printf("%-14s %6d %6d %6d %6d %12.3f %12s\n", c.name, c.d1,
                        c.d2, c.d3, k, ms, "");
        std::fflush(stdout);
    }

    print_gpu_state("exit");
    return 0;
}
