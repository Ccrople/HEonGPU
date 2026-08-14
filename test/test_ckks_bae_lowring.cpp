// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU validation of the Bae product taken on a low ring — Sylph's Table IV
// configuration (PCMM at ring degree 256, coefficient encoding, row-split
// packing), built from HERingSwitchOperator + HEBaePcmmOperator.
//
// The claim under test is narrow and it is an arithmetic one:
//
//     the a-part GEMM's width is the LATTICE DIMENSION, so ModDecomp — which
//     keeps the rank — cannot shrink it, and ring switching — which does not —
//     shrinks it by exactly k.
//
// Everything else follows from two index maps agreeing, and that is what
// TheDescentHandsBackOneMatrixRowPerCiphertext pins directly: the row-split
// encoder puts row k*i+u of M at coefficients (t*k + u) of big ciphertext i,
// the ring switch's DOWN split reads coefficient stride k, and the two compose
// to "piece u of ciphertext i is row k*i+u, whole". If that were off by so
// much as a phase, every later test would fail as noise rather than as drift.
//
// The comparison test is the point of the file: the SAME matrix, the SAME
// weight, taken once through today's big-ring pcmm_packed (ModDecomp, both
// GEMMs, ModPack) and once through the descent. Both are decrypted and both
// are checked against the host, so "the low ring computes the same product"
// is asserted rather than argued.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    struct LowRingFixture
    {
        heongpu::HEContext<S> big;
        heongpu::HEContext<S> small;

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;

        std::unique_ptr<heongpu::HERingSwitchOperator<S>> rs;
        std::unique_ptr<heongpu::HERingSwitchOperator<S>::SecretPair> pair;

        std::unique_ptr<heongpu::HEDecryptor<S>> small_decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> small_encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> small_ops;

        std::unique_ptr<heongpu::HEBaeLowRingPcmm> low;

        int n_big = 0;
        int n_small = 0;
        int k = 0;
        std::vector<int> sk_coefficients;

        LowRingFixture(int logn_big, int logn_small, std::vector<int> q_bits,
                       int shared_primes)
            : big(heongpu::GenHEContext<S>(heongpu::sec_level_type::none)),
              small(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            n_big = 1 << logn_big;
            n_small = 1 << logn_small;
            k = n_big / n_small;

            big->set_poly_modulus_degree(static_cast<size_t>(n_big));
            big->set_coeff_modulus_bit_sizes(q_bits, {60, 60});
            big->generate();

            const auto primes = big->get_key_modulus();
            const int big_q = static_cast<int>(q_bits.size());

            std::vector<Data64> q_vals, p_vals;
            for (int i = 0; i < shared_primes; i++)
                q_vals.push_back(primes[i].value);
            p_vals.push_back(primes[big_q].value);

            small->set_poly_modulus_degree(static_cast<size_t>(n_small));
            small->set_coeff_modulus_values(q_vals, p_vals);
            small->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(big);
            // Explicit coefficients only so that the COMPARISON test can build
            // ModPack keys for the big-ring path. The low-ring path never asks
            // for them; that is one of the things being demonstrated.
            {
                std::mt19937 rng(20260814u);
                std::uniform_int_distribution<int> pick(0, 2);
                sk_coefficients.assign(n_big, 0);
                for (int i = 0; i < n_big; ++i)
                    sk_coefficients[i] = pick(rng) - 1;
            }
            secret = std::make_unique<heongpu::Secretkey<S>>(sk_coefficients,
                                                             big);
            pub = std::make_unique<heongpu::Publickey<S>>(big);
            keygen->generate_public_key(*pub, *secret);

            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(big, *pub);
            decryptor = std::make_unique<heongpu::HEDecryptor<S>>(big, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(big);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(big,
                                                                     *encoder);

            rs = std::make_unique<heongpu::HERingSwitchOperator<S>>(big, small);
            pair = std::make_unique<
                heongpu::HERingSwitchOperator<S>::SecretPair>(
                rs->make_secret_pair(64, 0xBAE10001ULL));
            rs->generate_keys(*keygen, *secret, pair->embedded);

            small_decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(small, pair->small);
            small_encoder = std::make_unique<heongpu::HEEncoder<S>>(small);
            small_ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                small, *small_encoder);

            low = std::make_unique<heongpu::HEBaeLowRingPcmm>(big, small, *rs);
        }

        /// Encrypt a rows x n_small matrix in the big ring's row-split layout.
        std::vector<heongpu::Ciphertext<S>>
        encrypt_row_split(const std::vector<double>& m, int rows, double scale)
        {
            auto coeffs = low->encoder().encode_matrix(m, rows, scale);
            std::vector<heongpu::Ciphertext<S>> out;
            out.reserve(coeffs.size());
            for (auto& c : coeffs)
            {
                heongpu::Plaintext<S> p(big);
                low->encoder().load_coefficients(p, c, scale);
                heongpu::Ciphertext<S> ct(big);
                encryptor->encrypt(ct, p);
                out.push_back(std::move(ct));
            }
            return out;
        }

        std::vector<double>
        decrypt_row_split(std::vector<heongpu::Ciphertext<S>>& ct, int rows,
                          double scale)
        {
            std::vector<std::vector<int64_t>> coeffs;
            coeffs.reserve(ct.size());
            for (auto& c : ct)
            {
                heongpu::Plaintext<S> p(big);
                decryptor->decrypt(p, c);
                coeffs.push_back(low->encoder().extract_coefficients(p));
            }
            return low->encoder().decode_matrix(coeffs, rows, scale);
        }

        /// One small-ring ciphertext, read as the n_small coefficients it is.
        ///
        /// Deliberately NOT through HEEncoder::decode: a Bae matrix
        /// encryption is staged with load_coefficients, which does not set the
        /// plaintext's encoding tag, so the encoder's dispatch would be a
        /// coin toss between the DFT and the coefficient reading. The Bae
        /// operator's own extractor is the reading the encoding was written
        /// by.
        std::vector<double> decrypt_small_row(heongpu::Ciphertext<S>& c)
        {
            heongpu::Plaintext<S> p(small);
            small_decryptor->decrypt(p, c);
            const std::vector<int64_t> raw =
                low->product().extract_coefficients(p);
            std::vector<double> m(raw.size());
            for (size_t i = 0; i < raw.size(); ++i)
                m[i] = static_cast<double>(raw[i]) / c.scale();
            return m;
        }
    };

    std::vector<double> random_matrix(int rows, int cols, unsigned seed,
                                      double lo = -1.0, double hi = 1.0)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        std::vector<double> m(static_cast<size_t>(rows) * cols);
        for (auto& v : m)
            v = dist(rng);
        return m;
    }

    /// Row-major host reference for U (d1 x d2) times M (d2 x d3).
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

    /// The weight as a projection stores it: in_channels x out_channels, which
    /// is the transpose of the U the identity multiplies by.
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

    double max_abs_error(const std::vector<double>& a,
                         const std::vector<double>& b)
    {
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            worst = std::max(worst, std::abs(a[i] - b[i]));
        return worst;
    }

    // logN 14 over logN 12, k = 4.
    //
    // The small ring is 4096 and not Sylph's 256 because **the library will
    // not build a context below 4096**: defines.h sets MIN_POLY_DEGREE = 4096,
    // so ring 256 is not a parameter choice here, it is not representable.
    // That floor and the sec128 table's floor turn out to be the same number —
    // see §26.4 — so nothing is lost by it, but the reason to stop at 4096 is
    // this one and it fires first.
    constexpr int kLogBig = 14;
    constexpr int kLogSmall = 12;
} // namespace

// The load-bearing index claim of the whole file: the row-split encoder and
// the ring switch's stride-k split are the SAME map, so a descent hands back
// whole matrix rows and needs no glue. Asserted against a matrix whose entries
// carry their own row and column, so a transposed, rotated or phase-shifted
// answer cannot pass by symmetry.
TEST(CkksBaeLowRing, TheDescentHandsBackOneMatrixRowPerCiphertext)
{
    LowRingFixture f(kLogBig, kLogSmall, {60, 50, 50, 50}, 4);
    const double scale = std::pow(2.0, 40);

    const int rows = 2 * f.k;
    std::vector<double> M(static_cast<size_t>(rows) * f.n_small);
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < f.n_small; ++t)
                // Both terms are well above the 1e-4 tolerance, so a row shift
                // AND a column shift are each detectable; a matrix whose
                // column signal is below tolerance would let a phase error
                // through.
                M[static_cast<size_t>(r) * f.n_small + t] =
                    0.1 * static_cast<double>(r + 1) +
                    0.0002 * static_cast<double>(t);

    std::vector<heongpu::Ciphertext<S>> big_ct =
        f.encrypt_row_split(M, rows, scale);
    ASSERT_EQ(static_cast<int>(big_ct.size()), rows / f.k);

    std::vector<heongpu::Ciphertext<S>> small_ct = f.low->descend(big_ct, *f.ops);
    ASSERT_EQ(static_cast<int>(small_ct.size()), rows);

    for (int r = 0; r < rows; ++r)
    {
        std::vector<double> got = f.decrypt_small_row(small_ct[r]);
        ASSERT_GE(static_cast<int>(got.size()), f.n_small);
        double worst = 0.0;
        for (int t = 0; t < f.n_small; ++t)
            worst = std::max(
                worst, std::abs(got[t] -
                                M[static_cast<size_t>(r) * f.n_small + t]));
        EXPECT_LT(worst, 1e-4) << "row " << r;
    }
}

// End to end, and the one that would catch a broken composition: descend,
// project at the small ring, ascend, decrypt at the big ring. One level spent,
// and no key of any kind beyond the two ring-switch keys — in particular no
// ModPack key, which the big-ring path at this same k would require.
TEST(CkksBaeLowRing, AProjectionAtTheSmallRingMatchesTheHost)
{
    LowRingFixture f(kLogBig, kLogSmall, {60, 50, 50, 50}, 4);
    const double scale = std::pow(2.0, 40);

    const int in_channels = 2 * f.k;
    const int out_channels = f.k;
    const int tokens = f.n_small;

    const std::vector<double> M = random_matrix(in_channels, tokens, 11u);
    const std::vector<double> weight =
        random_matrix(in_channels, out_channels, 12u, -0.5, 0.5);

    std::vector<heongpu::Ciphertext<S>> big_ct =
        f.encrypt_row_split(M, in_channels, scale);
    std::vector<heongpu::Ciphertext<S>> down = f.low->descend(big_ct, *f.ops);

    std::vector<heongpu::Ciphertext<S>> product;
    f.low->project(product, down, weight, in_channels, out_channels,
                   *f.small_ops);
    ASSERT_EQ(static_cast<int>(product.size()), out_channels);

    // The product is k == 1 at the small ring, so it is ModDecomp-free and
    // ModPack-free and no switching key was ever generated for it.
    EXPECT_FALSE(f.low->product().modpack_keys_generated());

    std::vector<heongpu::Ciphertext<S>> up = f.low->ascend(product, *f.ops);
    ASSERT_EQ(static_cast<int>(up.size()), out_channels / f.k);

    const std::vector<double> got =
        f.decrypt_row_split(up, out_channels, up.front().scale());
    const std::vector<double> want =
        host_product(transpose(weight, in_channels, out_channels), M,
                     out_channels, in_channels, tokens);

    EXPECT_LT(max_abs_error(got, want), 1e-3);
}

// A block's seven projections are all row -> row, so the descent is paid once
// per block and not once per product. Two products back to back, no ascent
// between them, checked against the composed host reference.
TEST(CkksBaeLowRing, ProjectionsChainWithoutReturningToTheBigRing)
{
    LowRingFixture f(kLogBig, kLogSmall, {60, 50, 50, 50}, 4);
    const double scale = std::pow(2.0, 40);

    const int d0 = 2 * f.k;
    const int d1 = 2 * f.k;
    const int d2 = f.k;
    const int tokens = f.n_small;

    const std::vector<double> M = random_matrix(d0, tokens, 21u);
    const std::vector<double> w1 = random_matrix(d0, d1, 22u, -0.5, 0.5);
    const std::vector<double> w2 = random_matrix(d1, d2, 23u, -0.5, 0.5);

    std::vector<heongpu::Ciphertext<S>> big_ct =
        f.encrypt_row_split(M, d0, scale);
    std::vector<heongpu::Ciphertext<S>> stream = f.low->descend(big_ct, *f.ops);

    std::vector<heongpu::Ciphertext<S>> mid;
    f.low->project(mid, stream, w1, d0, d1, *f.small_ops);

    std::vector<heongpu::Ciphertext<S>> out;
    f.low->project(out, mid, w2, d1, d2, *f.small_ops);

    std::vector<heongpu::Ciphertext<S>> up = f.low->ascend(out, *f.ops);
    const std::vector<double> got =
        f.decrypt_row_split(up, d2, up.front().scale());

    const std::vector<double> first =
        host_product(transpose(w1, d0, d1), M, d1, d0, tokens);
    const std::vector<double> want =
        host_product(transpose(w2, d1, d2), first, d2, d1, tokens);

    EXPECT_LT(max_abs_error(got, want), 1e-3);
}

// THE COMPARISON. Same M, same weight, same level, two products:
//
//   big ring   ModDecomp + a-part GEMM of width 4096 + b-part of width 256 +
//              ModPack (16 key switches per output ciphertext), dimension 4096
//   low ring   one key switch down per input ciphertext + two GEMMs of width
//              256 + one key switch up per output ciphertext, dimension 256
//
// Both must land on the host reference, and on each other. That is what makes
// the cost table a comparison of two ways to compute one thing rather than of
// two different things.
TEST(CkksBaeLowRing, TheLowRingComputesTheSameProductAsTheBigRingPath)
{
    LowRingFixture f(kLogBig, kLogSmall, {60, 50, 50, 50}, 4);
    const double scale = std::pow(2.0, 40);

    const int in_channels = 2 * f.k;
    const int out_channels = f.k;
    const int tokens = f.n_small;

    const std::vector<double> M = random_matrix(in_channels, tokens, 31u);
    const std::vector<double> weight =
        random_matrix(in_channels, out_channels, 32u, -0.5, 0.5);
    const std::vector<double> want =
        host_product(transpose(weight, in_channels, out_channels), M,
                     out_channels, in_channels, tokens);

    // ---- today's path: everything at the big ring --------------------------
    f.low->encoder().generate_modpack_keys(*f.keygen, *f.secret,
                                           f.sk_coefficients);
    ASSERT_TRUE(f.low->encoder().modpack_keys_generated());

    std::vector<heongpu::Ciphertext<S>> big_in =
        f.encrypt_row_split(M, in_channels, scale);
    std::vector<heongpu::Ciphertext<S>*> big_ptrs;
    for (auto& c : big_in)
        big_ptrs.push_back(&c);

    std::vector<heongpu::Ciphertext<S>> big_out;
    f.low->encoder().project(big_out, big_ptrs, weight, in_channels,
                             out_channels, *f.ops);
    const std::vector<double> from_big =
        f.decrypt_row_split(big_out, out_channels, big_out.front().scale());

    // ---- Sylph's: down, product at 256, up ---------------------------------
    std::vector<heongpu::Ciphertext<S>> big_in2 =
        f.encrypt_row_split(M, in_channels, scale);
    std::vector<heongpu::Ciphertext<S>> down = f.low->descend(big_in2, *f.ops);
    std::vector<heongpu::Ciphertext<S>> low_out;
    f.low->project(low_out, down, weight, in_channels, out_channels,
                   *f.small_ops);
    std::vector<heongpu::Ciphertext<S>> up = f.low->ascend(low_out, *f.ops);
    const std::vector<double> from_low =
        f.decrypt_row_split(up, out_channels, up.front().scale());

    EXPECT_LT(max_abs_error(from_big, want), 1e-3) << "big-ring path";
    EXPECT_LT(max_abs_error(from_low, want), 1e-3) << "low-ring path";
    // Against each other, which is tighter than either against the host: the
    // two paths differ only in noise, not in what they compute.
    EXPECT_LT(max_abs_error(from_big, from_low), 2e-3);

    // Both spent exactly one level, so the comparison is at equal depth.
    EXPECT_EQ(big_out.front().depth(), up.front().depth());
}

// The arithmetic, as counts rather than as prose. These are the numbers §26 of
// LLAMA3_8B_LAYER_FLOW.md quotes, at the real 8B width and the real 128-token
// prompt, so a change that breaks the model breaks a test.
TEST(CkksBaeLowRing, TheDimensionIsTheOnlyLeverAndItMovesBothCosts)
{
    const int d = 4096;      // Llama-3-8B d_model, in and out
    const int tokens = 128;  // B = 1, the prompt this path actually runs
    const int n_big = 65536; // logN 16, where the level budget lives

    // Sylph's own ring, which is also exactly sqrt(65536).
    const auto sylph =
        heongpu::HEBaeLowRingPcmm::low_ring_cost(d, d, tokens, n_big, 256);
    const auto today =
        heongpu::HEBaeLowRingPcmm::big_ring_cost(d, d, tokens, n_big, 256);

    // (N + cols) / tokens against 2 * n_small / tokens.
    EXPECT_NEAR(today.per_token, (65536.0 + 256.0) / 128.0, 1e-9);
    EXPECT_NEAR(sylph.per_token, (256.0 + 256.0) / 128.0, 1e-9);
    EXPECT_NEAR(today.per_token / sylph.per_token, 128.5, 1e-6);

    // ModPack owes one key switch per output ROW; the descent and ascent owe
    // one per big-ring CIPHERTEXT, and a ciphertext is k rows.
    EXPECT_EQ(today.key_switches, d);
    EXPECT_EQ(sylph.key_switches, (d + d) / (n_big / 256));

    // And the dimension is the whole difference, stated as such.
    EXPECT_EQ(today.dimension, n_big);
    EXPECT_EQ(sylph.dimension, 256);

    // 4096 is the library's hard MIN_POLY_DEGREE and clears sec128 at 30-bit
    // primes: one usable level, 8x.
    const auto floor4096 =
        heongpu::HEBaeLowRingPcmm::low_ring_cost(d, d, tokens, n_big, 4096);
    EXPECT_EQ(floor4096.dimension, 4096);
    EXPECT_NEAR(floor4096.per_token, (4096.0 + 4096.0) / 128.0, 1e-9);
    EXPECT_NEAR(today.per_token / floor4096.per_token, 65792.0 / 8192.0, 1e-6);
    EXPECT_EQ(floor4096.key_switches, (d + d) / (n_big / 4096));

    // But the binding floor is 8192, because the shared-prefix rule hands the
    // small ring the pipeline's own q0 = 41 and 41 + 33 + 41 > 109. This is
    // the row that is deployable, and it is 4x rather than 8x or 128x.
    const auto deployable =
        heongpu::HEBaeLowRingPcmm::low_ring_cost(d, d, tokens, n_big, 8192);
    EXPECT_EQ(deployable.dimension, 8192);
    EXPECT_NEAR(deployable.per_token, (8192.0 + 8192.0) / 128.0, 1e-9);
    EXPECT_NEAR(today.per_token / deployable.per_token, 65792.0 / 16384.0,
                1e-6);
    EXPECT_EQ(deployable.key_switches, (d + d) / (n_big / 8192));
}

// ModDecomp keeps the rank and therefore cannot move the a-part. Asserted so
// that the header's central claim is a test and not a comment: at the same
// column count, the big-ring cost does not depend on how the rows are split.
TEST(CkksBaeLowRing, ModDecompDoesNotShrinkTheAPart)
{
    const int d = 4096;
    const int tokens = 128;
    const int n_big = 65536;

    const auto k256 =
        heongpu::HEBaeLowRingPcmm::big_ring_cost(d, d, tokens, n_big, 256);
    const auto k1 =
        heongpu::HEBaeLowRingPcmm::big_ring_cost(d, d, tokens, n_big, 65536);

    EXPECT_EQ(k256.a_macs, k1.a_macs);
    EXPECT_EQ(k256.dimension, k1.dimension);
}

TEST(CkksBaeLowRing, ValidationRejectsMisuse)
{
    LowRingFixture f(kLogBig, kLogSmall, {60, 50, 50, 50}, 4);
    const double scale = std::pow(2.0, 40);

    std::vector<heongpu::Ciphertext<S>> empty;
    EXPECT_THROW(f.low->descend(empty, *f.ops), std::invalid_argument);
    EXPECT_THROW(f.low->ascend(empty, *f.ops), std::invalid_argument);

    const std::vector<double> M = random_matrix(f.k, f.n_small, 41u);
    std::vector<heongpu::Ciphertext<S>> big_ct =
        f.encrypt_row_split(M, f.k, scale);
    std::vector<heongpu::Ciphertext<S>> down = f.low->descend(big_ct, *f.ops);

    // A partial big-ring ciphertext does not exist: the row count going up
    // must be a multiple of k.
    std::vector<heongpu::Ciphertext<S>> partial;
    for (int i = 0; i < f.k - 1; ++i)
        partial.push_back(down[static_cast<size_t>(i)]);
    EXPECT_THROW(f.low->ascend(partial, *f.ops), std::invalid_argument);

    // One ciphertext per row, not per k rows: handing the low-ring projection
    // the big-ring count is the mistake the layouts make easy.
    std::vector<heongpu::Ciphertext<S>> out;
    EXPECT_THROW(f.low->project(out, down, random_matrix(1, f.k, 42u), 1, f.k,
                                *f.small_ops),
                 std::invalid_argument);

    // The cost model is not a free function of anything: the small ring must
    // divide the big one and the tokens must fit in it.
    EXPECT_THROW(
        heongpu::HEBaeLowRingPcmm::low_ring_cost(16, 16, 512, 65536, 256),
        std::invalid_argument);
    EXPECT_THROW(
        heongpu::HEBaeLowRingPcmm::low_ring_cost(16, 16, 128, 65536, 384),
        std::invalid_argument);
}
