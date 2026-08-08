// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU validation of CKKS ring switching (Sylph §3.3, LLAMA3_8B_LAYER_FLOW.md
// §10 item 5).
//
// The decomposition under test is exact: R_N = ⊕_j X^j · R_{N'}[X^k], so a
// big-ring ciphertext under the embedded secret s'(X^k) splits into k
// small-ring ciphertexts under s' with NO new noise — the only noise in a
// full switch is the ordinary key switch that brackets it. That gives the
// tests sharp expectations:
//
//   - DOWN followed by small-ring decryption must reproduce the big message
//     on coefficient strides: m_j[i] = m[i·k + j].
//   - UP is the inverse interleave.
//   - DOWN then UP is the identity up to two key switches.
//   - A rescale taken INSIDE the small-ring island divides by the very prime
//     the big ring would have divided by (the chains are value-identical), so
//     scales and levels re-align on composition with no correction — the
//     contract the two-ring Llama-3 flow stands on.
//
// The fixture builds the big context first and hands its leading Q primes to
// the small context verbatim through set_coeff_modulus_values; the small
// context could not generate those primes itself, because bit-size prime
// generation steps by 2N and is therefore ring-degree dependent. The big
// context carries TWO special primes deliberately: a single special prime is
// method I, whose noise at large N has no budget at all.

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

    struct RingFixture
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

        std::unique_ptr<heongpu::HEKeyGenerator<S>> small_keygen;
        std::unique_ptr<heongpu::Publickey<S>> small_pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> small_encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> small_decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> small_encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> small_ops;

        int n_big = 0;
        int n_small = 0;
        int k = 0;

        /// @param q_bits        Bit sizes of the big Q chain, bottom first.
        /// @param shared_primes How many of those primes the small context
        ///                      receives (its whole Q chain).
        RingFixture(int logn_big, int logn_small, std::vector<int> q_bits,
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
            // The big context's first special prime works as the small
            // context's special prime too: distinct from every shared Q prime
            // and ≡ 1 mod 2N_big, hence NTT-friendly at the small ring.
            p_vals.push_back(primes[big_q].value);

            small->set_poly_modulus_degree(static_cast<size_t>(n_small));
            small->set_coeff_modulus_values(q_vals, p_vals);
            small->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(big);
            secret = std::make_unique<heongpu::Secretkey<S>>(big);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(big);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(big, *pub);
            decryptor = std::make_unique<heongpu::HEDecryptor<S>>(big, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(big);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(big,
                                                                     *encoder);

            rs = std::make_unique<heongpu::HERingSwitchOperator<S>>(big,
                                                                    small);
            pair = std::make_unique<heongpu::HERingSwitchOperator<S>::
                                        SecretPair>(
                rs->make_secret_pair(64, 0xC0FFEEULL));
            rs->generate_keys(*keygen, *secret, pair->embedded);

            small_keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(small);
            small_pub = std::make_unique<heongpu::Publickey<S>>(small);
            small_keygen->generate_public_key(*small_pub, pair->small);
            small_encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(small, *small_pub);
            small_decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(small, pair->small);
            small_encoder = std::make_unique<heongpu::HEEncoder<S>>(small);
            small_ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                small, *small_encoder);
        }

        std::vector<double> random_message(int length, unsigned seed) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<double> m(length);
            for (auto& v : m)
                v = dist(rng);
            return m;
        }

        heongpu::Ciphertext<S> encrypt_big_coeff(const std::vector<double>& m,
                                                 double scale)
        {
            heongpu::Plaintext<S> p(big);
            encoder->encode(p, m, scale, heongpu::ExecutionOptions(),
                            heongpu::encoding::COEFFICIENT);
            heongpu::Ciphertext<S> c(big);
            encryptor->encrypt(c, p);
            return c;
        }

        std::vector<double> decrypt_big_coeff(heongpu::Ciphertext<S>& c)
        {
            heongpu::Plaintext<S> p(big);
            decryptor->decrypt(p, c);
            std::vector<double> m;
            encoder->decode(m, p);
            return m;
        }

        heongpu::Ciphertext<S> encrypt_small_coeff(const std::vector<double>& m,
                                                   double scale)
        {
            heongpu::Plaintext<S> p(small);
            small_encoder->encode(p, m, scale, heongpu::ExecutionOptions(),
                                  heongpu::encoding::COEFFICIENT);
            heongpu::Ciphertext<S> c(small);
            small_encryptor->encrypt(c, p);
            return c;
        }

        std::vector<double> decrypt_small_coeff(heongpu::Ciphertext<S>& c)
        {
            heongpu::Plaintext<S> p(small);
            small_decryptor->decrypt(p, c);
            std::vector<double> m;
            small_encoder->decode(m, p);
            return m;
        }
    };

    double max_abs_diff(const std::vector<double>& a,
                        const std::vector<double>& b, int count)
    {
        double worst = 0.0;
        for (int i = 0; i < count; i++)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        return worst;
    }
} // namespace

// DOWN alone: each of the k small ciphertexts must decrypt, under the SMALL
// secret in the SMALL context, to the stride-k slice of the big message. This
// is the sharp end of the test file: it exercises the embedded-secret key
// switch, the cross-context coefficient split, and the shared prime chain all
// at once, and any index or table mix-up lands here as noise, not as a small
// error.
TEST(CkksRingSwitch, SwitchDownDecryptsToStrides)
{
    RingFixture f(13, 12, {60, 50, 50}, 3);
    const double scale = std::pow(2.0, 40);

    const std::vector<double> m = f.random_message(f.n_big, 1);
    heongpu::Ciphertext<S> ct = f.encrypt_big_coeff(m, scale);

    std::vector<heongpu::Ciphertext<S>> parts =
        f.rs->switch_down(ct, *f.ops);
    ASSERT_EQ(static_cast<int>(parts.size()), f.k);

    for (int j = 0; j < f.k; j++)
    {
        std::vector<double> got = f.decrypt_small_coeff(parts[j]);
        double worst = 0.0;
        for (int i = 0; i < f.n_small; i++)
            worst = std::max(worst,
                             std::fabs(got[i] - m[static_cast<size_t>(i) *
                                                      f.k +
                                                  j]));
        EXPECT_LT(worst, 1e-4) << "slice " << j;
    }
}

// UP alone: k independently encrypted small messages compose into one big
// ciphertext under sk_big whose coefficients interleave them.
TEST(CkksRingSwitch, ComposeUpInterleaves)
{
    RingFixture f(13, 12, {60, 50, 50}, 3);
    const double scale = std::pow(2.0, 40);

    std::vector<std::vector<double>> m(f.k);
    std::vector<heongpu::Ciphertext<S>> parts;
    for (int j = 0; j < f.k; j++)
    {
        m[j] = f.random_message(f.n_small, 100 + j);
        parts.push_back(f.encrypt_small_coeff(m[j], scale));
    }

    heongpu::Ciphertext<S> composed = f.rs->compose_up(parts, *f.ops);
    std::vector<double> got = f.decrypt_big_coeff(composed);

    double worst = 0.0;
    for (int i = 0; i < f.n_small; i++)
        for (int j = 0; j < f.k; j++)
            worst = std::max(
                worst, std::fabs(got[static_cast<size_t>(i) * f.k + j] -
                                 m[j][i]));
    EXPECT_LT(worst, 1e-4);
}

// DOWN then UP: identity up to the two key switches.
TEST(CkksRingSwitch, RoundTripIsIdentity)
{
    RingFixture f(13, 12, {60, 50, 50}, 3);
    const double scale = std::pow(2.0, 40);

    const std::vector<double> m = f.random_message(f.n_big, 2);
    heongpu::Ciphertext<S> ct = f.encrypt_big_coeff(m, scale);

    std::vector<heongpu::Ciphertext<S>> parts =
        f.rs->switch_down(ct, *f.ops);
    heongpu::Ciphertext<S> back = f.rs->compose_up(parts, *f.ops);

    std::vector<double> got = f.decrypt_big_coeff(back);
    EXPECT_LT(max_abs_diff(got, m, f.n_big), 1e-4);
}

// The two-ring island contract: switch down at the top, do real small-ring
// arithmetic INCLUDING a rescale, compose up one level lower. The rescale
// inside the island divides by the same prime value the big chain holds at
// that position, so the composed ciphertext's level and scale line up with
// what a big-ring rescale would have produced — no correction anywhere.
TEST(CkksRingSwitch, SmallRingIslandOpTracksLevelsAndScale)
{
    RingFixture f(13, 12, {60, 50, 50}, 3);
    const double scale = std::pow(2.0, 40);
    const double gain = 0.75;

    const std::vector<double> m = f.random_message(f.n_big, 3);
    heongpu::Ciphertext<S> ct = f.encrypt_big_coeff(m, scale);

    std::vector<heongpu::Ciphertext<S>> parts =
        f.rs->switch_down(ct, *f.ops);

    for (auto& part : parts)
    {
        f.small_ops->multiply_plain_inplace(part, gain, scale);
        f.small_ops->rescale_inplace(part);
    }

    heongpu::Ciphertext<S> composed = f.rs->compose_up(parts, *f.ops);
    EXPECT_EQ(composed.depth(), 1);

    std::vector<double> got = f.decrypt_big_coeff(composed);
    double worst = 0.0;
    for (int i = 0; i < f.n_big; i++)
        worst = std::max(worst, std::fabs(got[i] - gain * m[i]));
    EXPECT_LT(worst, 1e-3);
}

// Switching below the top of the chain: consume a big-ring level first, then
// switch at l = 2 active primes. The small ciphertexts land at depth 1 of
// their own chain and come back cleanly.
TEST(CkksRingSwitch, LeveledSwitchAfterBigRescale)
{
    RingFixture f(13, 12, {60, 50, 50}, 3);
    const double scale = std::pow(2.0, 40);
    const double gain = 0.5;

    const std::vector<double> m = f.random_message(f.n_big, 4);
    heongpu::Ciphertext<S> ct = f.encrypt_big_coeff(m, scale);

    f.ops->multiply_plain_inplace(ct, gain, scale);
    f.ops->rescale_inplace(ct);

    std::vector<heongpu::Ciphertext<S>> parts =
        f.rs->switch_down(ct, *f.ops);
    ASSERT_EQ(parts[0].depth(), 1);

    heongpu::Ciphertext<S> back = f.rs->compose_up(parts, *f.ops);
    EXPECT_EQ(back.depth(), 1);

    std::vector<double> got = f.decrypt_big_coeff(back);
    double worst = 0.0;
    for (int i = 0; i < f.n_big; i++)
        worst = std::max(worst, std::fabs(got[i] - gain * m[i]));
    EXPECT_LT(worst, 1e-3);
}

// The Llama-3 shape: logN 16 ↔ logN 12, k = 16 — one big ciphertext descends
// into sixteen small ones and returns. The chain is kept short to keep the
// context affordable inside the test timeout; the shape, not the depth, is
// what this case pins.
TEST(CkksRingSwitch, K16RoundTripAtLlamaShape)
{
    RingFixture f(16, 12, {60, 50}, 2);
    ASSERT_EQ(f.k, 16);
    const double scale = std::pow(2.0, 40);

    const std::vector<double> m = f.random_message(f.n_big, 5);
    heongpu::Ciphertext<S> ct = f.encrypt_big_coeff(m, scale);

    std::vector<heongpu::Ciphertext<S>> parts =
        f.rs->switch_down(ct, *f.ops);
    ASSERT_EQ(static_cast<int>(parts.size()), 16);

    // Spot-check one slice at the small ring before returning.
    std::vector<double> slice = f.decrypt_small_coeff(parts[7]);
    double worst_slice = 0.0;
    for (int i = 0; i < f.n_small; i++)
        worst_slice = std::max(
            worst_slice,
            std::fabs(slice[i] - m[static_cast<size_t>(i) * 16 + 7]));
    EXPECT_LT(worst_slice, 1e-4);

    heongpu::Ciphertext<S> back = f.rs->compose_up(parts, *f.ops);
    std::vector<double> got = f.decrypt_big_coeff(back);
    EXPECT_LT(max_abs_diff(got, m, f.n_big), 1e-4);
}

// ---------------------------------------------------------------------------
// The point of the whole subsystem: a Kang rectangular PCMM (Algorithm 5)
// running at the SMALL ring on data that arrived from the BIG ring by ring
// switch. Matrix columns are interleaved into big-ring ciphertexts host-side,
// encrypted at logN 13, switched down to logN 12 — where the N/2 − 1 Galois
// keys Algorithm 5 needs actually fit — multiplied, and verified against the
// CPU product. This is LLAMA3_8B_LAYER_FLOW.md §7bis's open question answered
// operationally: the descent preserves a matrix encryption.
// ---------------------------------------------------------------------------
TEST(CkksRingSwitch, Alg5OnSwitchedDataMatchesReference)
{
    using cd = std::complex<double>;

    RingFixture f(13, 12, {50, 40, 40}, 3);
    ASSERT_EQ(f.k, 2);

    const int d = 128;
    heongpu::BatchMatrixLayout layout(f.n_small, d);
    const int k_small = layout.k;      // 32
    const int blocks = layout.batch;   // 16
    const int half = f.n_small / 2;    // 2048

    // Small-ring operator stack for Algorithm 5: the batch operator, its
    // rotation group (N/2 − 1 keys — the count that forced the small ring),
    // and coefficient extraction for the reference comparison.
    heongpu::HEBatchMatrixOperator<S> small_bm(f.small, layout);
    std::vector<int> rot = heongpu::get_rectangular_rotation_indices(layout);
    heongpu::Galoiskey<S> small_galois(f.small, rot);
    f.small_keygen->generate_galois_key(small_galois, f.pair->small);

    // Big-ring coefficient staging (raw int64 load, no re-scaling).
    heongpu::HEBatchMatrixOperator<S> big_bm(
        f.big, heongpu::BatchMatrixLayout(f.n_big, d));

    const double scale_m = std::pow(2.0, 25);
    const double scale_u = std::pow(2.0, 25);

    std::mt19937_64 rng(20260808u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::vector<cd>> M(
        blocks, std::vector<cd>(static_cast<size_t>(d) * d));
    std::vector<std::vector<cd>> U(
        blocks, std::vector<cd>(static_cast<size_t>(d) * half, cd(0.0, 0.0)));
    for (int l = 0; l < blocks; ++l)
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
            {
                M[l][static_cast<size_t>(i) * d + j] = cd(dist(rng), 0.0);
                U[l][static_cast<size_t>(i) * half + j] = cd(dist(rng), 0.0);
            }

    // Reference: W_0 = sum_l M_l * U_l[:, 0:d].
    std::vector<double> W0(static_cast<size_t>(d) * d, 0.0);
    for (int l = 0; l < blocks; ++l)
        for (int i = 0; i < d; ++i)
            for (int t = 0; t < d; ++t)
            {
                const double a = M[l][static_cast<size_t>(i) * d + t].real();
                for (int j = 0; j < d; ++j)
                    W0[static_cast<size_t>(i) * d + j] +=
                        a * U[l][static_cast<size_t>(t) * half + j].real();
            }

    heongpu::BatchMatrixEncoder bm(k_small);
    std::vector<int64_t> m_coeffs, u_coeffs;
    bm.encode(M, d, d, scale_m, m_coeffs);
    bm.encode(U, d, half, scale_u, u_coeffs);

    std::vector<std::vector<int64_t>> columns;
    heongpu::build_matrix_encryption_coefficients(m_coeffs, layout, d, d,
                                                  columns);

    // Interleave column pairs into big-ring polynomials — the layout the big
    // ring would naturally hold them in — and encrypt at logN 13.
    std::vector<heongpu::Ciphertext<S>> big_cts;
    big_cts.reserve(d / f.k);
    for (int b = 0; b < d / f.k; ++b)
    {
        std::vector<int64_t> interleaved(static_cast<size_t>(f.n_big), 0);
        for (int i = 0; i < f.n_small; ++i)
            for (int j = 0; j < f.k; ++j)
                interleaved[static_cast<size_t>(i) * f.k + j] =
                    columns[static_cast<size_t>(b) * f.k + j][i];

        heongpu::Plaintext<S> p(f.big);
        big_bm.load_coefficients(p, interleaved, scale_m);
        heongpu::Ciphertext<S> c(f.big);
        f.encryptor->encrypt(c, p);
        big_cts.push_back(std::move(c));
    }

    // DOWN: every big ciphertext descends into its k column ciphertexts.
    std::vector<heongpu::Ciphertext<S>> cols;
    cols.reserve(d);
    for (auto& c : big_cts)
    {
        std::vector<heongpu::Ciphertext<S>> parts =
            f.rs->switch_down(c, *f.ops);
        for (auto& part : parts)
            cols.push_back(std::move(part));
    }
    ASSERT_EQ(static_cast<int>(cols.size()), d);

    // Algorithm 5 at the small ring.
    small_bm.encode_plaintext_matrix(u_coeffs, d, half, 0, scale_u);
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : cols)
        in.push_back(&c);
    std::vector<heongpu::Ciphertext<S>> out;
    small_bm.rectangular_pcmm(
        out, in, small_galois, *f.small_ops,
        heongpu::HEBatchMatrixOperator<S>::BlockAxis::slot, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(d));

    const double combined = scale_m * scale_u;
    double worst = 0.0;
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> p(f.small);
        f.small_decryptor->decrypt(p, out[j]);
        std::vector<int64_t> got;
        small_bm.extract_coefficients(got, p);
        for (int i = 0; i < d; ++i)
        {
            const double v = static_cast<double>(got[i]) / combined;
            const double want = W0[static_cast<size_t>(i) * d + j];
            worst = std::max(worst, std::abs(v - want));
            ASSERT_NEAR(v, want, 1e-2 * std::max(1.0, std::abs(want)))
                << "W0 at (" << i << "," << j << ")";
        }
    }
    std::cout << "Alg 5 on ring-switched data, worst error: " << worst
              << std::endl;
}
