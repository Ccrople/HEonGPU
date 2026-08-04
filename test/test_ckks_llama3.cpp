// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Per-function checks of the Llama-3 homomorphic primitives.
//
// Each operator is measured against the plaintext function it approximates,
// one operator per test, so a regression names the primitive that broke rather
// than the layer that contained it. The host-only tests come first: they pin
// the permutation algebra of Section 4.2 and the Chebyshev fits before any GPU
// time is spent on them.
//
// TOLERANCES
// ----------
// Every bound here is set from the error actually measured, with a couple of
// orders of headroom, not from what would merely look acceptable. Almost all of
// these primitives land on the CKKS noise floor around 1e-8, so a bound of 1e-6
// still passes comfortably while catching a real loss of precision; a bound of
// 1e-3 would have let a five hundred fold regression through. The exceptions
// are the Chebyshev approximations themselves, where the fit error dominates
// and the bound is set from the degree instead.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;

    /// Largest absolute difference over the compared prefix.
    double max_error(const std::vector<double>& got,
                     const std::vector<double>& want)
    {
        double worst = 0.0;
        const std::size_t n = std::min(got.size(), want.size());
        for (std::size_t i = 0; i < n; i++)
        {
            worst = std::max(worst, std::abs(got[i] - want[i]));
        }
        return worst;
    }

    /// Print the measured error alongside the bound it is checked against, so
    /// a primitive that is still passing but has lost a digit is visible.
    double reported(const char* label, const std::vector<double>& got,
                    const std::vector<double>& want)
    {
        const double worst = max_error(got, want);
        std::cout << "[ MEASURED ] " << label << " max error " << worst
                  << std::endl;
        return worst;
    }

    std::vector<double> apply_tau(const std::vector<double>& a, int d,
                                  int times)
    {
        std::vector<double> out = a;
        for (int t = 0; t < times; t++)
        {
            out = llama::permute_tau(out, d);
        }
        return out;
    }

    std::vector<double> random_matrix(int d, std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> m(static_cast<std::size_t>(d) * d);
        for (auto& v : m)
        {
            v = dist(rng);
        }
        return m;
    }

    std::vector<std::vector<double>> random_blocks(
        const llama::MatrixLayout& layout, double amplitude,
        std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-amplitude, amplitude);
        std::vector<std::vector<double>> blocks(
            layout.batch,
            std::vector<double>(static_cast<std::size_t>(layout.d) * layout.d));
        for (auto& block : blocks)
        {
            for (double& v : block)
            {
                v = dist(rng);
            }
        }
        return blocks;
    }

    /// Matrix m entry e goes to slot e * batch + m, the MatrixLayout packing.
    std::vector<double> pack_blocks(
        const std::vector<std::vector<double>>& blocks,
        const llama::MatrixLayout& layout)
    {
        const int entries = layout.d * layout.d;
        std::vector<double> out(static_cast<std::size_t>(layout.slots), 0.0);
        for (int m = 0; m < layout.batch; m++)
        {
            for (int e = 0; e < entries; e++)
            {
                out[static_cast<std::size_t>(e) * layout.batch + m] =
                    blocks[m][e];
            }
        }
        return out;
    }

    std::vector<std::vector<double>> unpack_blocks(
        const std::vector<double>& slots, const llama::MatrixLayout& layout)
    {
        const int entries = layout.d * layout.d;
        std::vector<std::vector<double>> blocks(
            layout.batch, std::vector<double>(entries));
        for (int m = 0; m < layout.batch; m++)
        {
            for (int e = 0; e < entries; e++)
            {
                blocks[m][e] =
                    slots[static_cast<std::size_t>(e) * layout.batch + m];
            }
        }
        return blocks;
    }

    double silu_host(double x) { return x / (1.0 + std::exp(-x)); }

    // -----------------------------------------------------------------------
    // Host-only checks
    // -----------------------------------------------------------------------

    TEST(CKKS_Llama3, ChebyshevFitsTheNonLinearities)
    {
        struct Case
        {
            const char* name;
            std::function<double(double)> f;
            double a;
            double b;
            int degree;
            double tolerance;
        };

        const std::vector<Case> cases = {
            {"silu", [](double x) { return x / (1.0 + std::exp(-x)); }, -11.0,
             11.0, 31, 1e-3},
            {"inverse sqrt", [](double x) { return 1.0 / std::sqrt(x); }, 0.5,
             2.0, 15, 1e-6},
            {"inverse", [](double x) { return 1.0 / x; }, 2.0, 48.0, 15, 5e-3},
            {"exp scaled", [](double x) { return std::exp(x / 2.0); }, -2.0,
             0.0, 15, 1e-9},
        };

        for (const Case& c : cases)
        {
            const std::vector<double> coeffs =
                llama::chebyshev_coefficients(c.f, c.a, c.b, c.degree);
            ASSERT_EQ(static_cast<int>(coeffs.size()), c.degree + 1) << c.name;

            double worst = 0.0;
            for (int i = 0; i <= 400; i++)
            {
                const double x = c.a + (c.b - c.a) * i / 400.0;
                worst = std::max(
                    worst, std::abs(llama::chebyshev_evaluate(coeffs, c.a, c.b,
                                                              x) -
                                    c.f(x)));
            }
            EXPECT_LT(worst, c.tolerance)
                << c.name << " fit is worse than expected: " << worst;
        }
    }

    /// The Jiang-Kim-Lauter-Song identity ccmm is built on.
    ///
    /// Section 4.2 only covers a plaintext weight, and attention needs two
    /// encrypted operands, so this is the algebra that fills the gap. Checking
    /// it here costs nothing and a failure names the identity rather than the
    /// homomorphic machinery around it.
    TEST(CKKS_Llama3, EncryptedProductIdentityReproducesTheProduct)
    {
        std::mt19937_64 rng(20260805);
        const int d = 8;
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> b = random_matrix(d, rng);

        const std::vector<double> sigma_a = llama::permute_sigma(a, d);
        const std::vector<double> tau_b = llama::permute_tau(b, d);

        std::vector<double> acc(static_cast<std::size_t>(d) * d, 0.0);
        for (int k = 0; k < d; k++)
        {
            const std::vector<double> left =
                llama::rotate_cols_host(sigma_a, d, k);
            const std::vector<double> right =
                llama::rotate_rows_host(tau_b, d, k);
            for (std::size_t e = 0; e < acc.size(); e++)
            {
                acc[e] += left[e] * right[e];
            }
        }

        EXPECT_LT(max_error(acc, llama::matmul_host(a, b, d)), 1e-12);
    }

    /// The decomposition that keeps ccmm at depth two.
    ///
    /// rot_C^k(sigma(A)) is usually built by applying sigma and then rotating
    /// the columns, two levels. It does not have to be: every entry of it is an
    /// entry of A reached by a flat shift that depends only on the diagonal the
    /// entry lies on, so one rotation of A serves a whole diagonal and all d
    /// values of k share the same 2d - 1 rotations. Only the masks differ, and
    /// masking is the one level. This pins that claim.
    TEST(CKKS_Llama3, ColumnRotatedSigmaIsOneShiftOfAPerDiagonal)
    {
        std::mt19937_64 rng(20260806);
        const int d = 8;
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> sigma_a = llama::permute_sigma(a, d);

        for (int k = 0; k < d; k++)
        {
            std::vector<double> got(static_cast<std::size_t>(d) * d, 0.0);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    const int shift = ((i + j + k) % d) - j;
                    // A flat shift by `shift`, which is what a slot rotation
                    // gives, and it never leaves row i.
                    got[i * d + j] = a[i * d + j + shift];
                }
            }

            EXPECT_EQ(got, llama::rotate_cols_host(sigma_a, d, k))
                << "k = " << k;
        }
    }

    /// The causal mask keeps the past and leaves the first round's range put.
    ///
    /// A plain zero-one mask would not: the sum of squares the first
    /// reciprocal has to cover would shrink with the number of keys a query
    /// can see, so the fit would have to span the whole sequence length. The
    /// weight sqrt(d / kept) makes the sum of squared weights exactly d for
    /// every query, which is what an unmasked row would give, and the rounds
    /// normalise so the weight itself cancels.
    TEST(CKKS_Llama3, CausalMaskHidesTheFutureAtAConstantWeight)
    {
        const int d = 8;
        const llama::MatrixLayout layout(d, 4);
        const std::vector<double> mask =
            llama::Llama3Operator::causal_mask(layout);
        ASSERT_EQ(static_cast<int>(mask.size()), layout.slots);

        for (int query = 0; query < d; query++)
        {
            for (int m = 0; m < layout.batch; m++)
            {
                double squares = 0.0;
                for (int key = 0; key < d; key++)
                {
                    const double w =
                        mask[(key * d + query) * layout.batch + m];
                    if (key > query)
                    {
                        EXPECT_EQ(w, 0.0) << "key " << key << " of query "
                                          << query;
                    }
                    else
                    {
                        EXPECT_GT(w, 0.0);
                    }
                    squares += w * w;
                }
                EXPECT_NEAR(squares, static_cast<double>(d), 1e-12)
                    << "query " << query;
            }
        }
    }

    /// Theorem 2 and Equation (5), evaluated entirely on the host.
    ///
    /// If the permutation algebra is wrong the homomorphic PCMM cannot be
    /// right either, and finding that out here costs nothing.
    TEST(CKKS_Llama3, EquationFiveReproducesTauOfTheProduct)
    {
        std::mt19937_64 rng(20260804);
        const int d = 16;

        for (int ell = 0; ell <= 2; ell++)
        {
            for (int baby : {2, 4, 8})
            {
                const int giant = d / baby;

                const std::vector<double> a = random_matrix(d, rng);
                const std::vector<double> b = random_matrix(d, rng);

                const std::vector<double> stored =
                    apply_tau(llama::permute_sigma(a, d), d, ell);
                const std::vector<double> operand = apply_tau(b, d, ell + 1);

                std::vector<double> acc(static_cast<std::size_t>(d) * d, 0.0);
                for (int j = 0; j < giant; j++)
                {
                    std::vector<double> inner(
                        static_cast<std::size_t>(d) * d, 0.0);
                    for (int i = 0; i < baby; i++)
                    {
                        const std::vector<double> block =
                            heongpu::llama::Llama3Operator::
                                pcmm_plaintext_block(stored, d, i, j, baby,
                                                     ell);
                        const std::vector<double> rotated =
                            llama::rotate_rows_host(operand, d, i);
                        for (std::size_t e = 0; e < inner.size(); e++)
                        {
                            inner[e] += block[e] * rotated[e];
                        }
                    }

                    const std::vector<double> shifted =
                        llama::rotate_rows_host(inner, d, j * baby);
                    for (std::size_t e = 0; e < acc.size(); e++)
                    {
                        acc[e] += shifted[e];
                    }
                }

                const std::vector<double> want =
                    apply_tau(llama::matmul_host(a, b, d), d, ell);
                EXPECT_LT(max_error(acc, want), 1e-10)
                    << "ell=" << ell << " baby=" << baby;
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU fixture
    // -----------------------------------------------------------------------

    /// One context deep enough for every primitive below.
    ///
    /// The chain is long because the tests exercise the primitives back to
    /// back without bootstrapping between them; Sylph itself would bootstrap
    /// and run each of these at a far shorter chain.
    class Llama3Env : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 8192;
        static constexpr int kLimbs = 24;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            // Two special primes so key switching runs method II; a single one
            // leaves no noise budget worth speaking of.
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();

            std::set<int> indices;
            for (int r : llama::Llama3Operator::strided_rotation_indices(
                     slots / 32, 32))
            {
                indices.insert(r);
            }
            for (int r : llama::Llama3Operator::strided_rotation_indices(
                     slots / 64, 64))
            {
                indices.insert(r);
            }
            for (int span : {16, 32})
            {
                for (int r :
                     llama::Llama3Operator::blocked_rotation_indices(span))
                {
                    indices.insert(r);
                }
            }
            const llama::MatrixLayout layout(16, slots / 256);
            // Every BSGS split of d = 16 the tests exercise.
            for (int baby : {2, 4, 8})
            {
                for (int r : llama::Llama3Operator::pcmm_rotation_indices(
                         layout, 16 / baby, baby))
                {
                    indices.insert(r);
                }
            }
            indices.insert(kRopeSwap);

            std::vector<int> shifts(indices.begin(), indices.end());
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        static constexpr int kRopeSwap = 64;

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& values)
        {
            heongpu::Plaintext<S> plain(context);
            encoder->encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor->encrypt(cipher, plain);
            return cipher;
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& cipher)
        {
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, cipher);
            std::vector<double> values;
            encoder->decode(values, plain);
            return values;
        }

        /// A Galois key holding exactly @p shifts and nothing else.
        ///
        /// Galoiskey(context, shift_vec) stores only what it is handed, with
        /// no power-of-two fallback, and leaves the bounds the fallback path
        /// reads uninitialised. Building a key from exactly what a primitive
        /// advertises is therefore the only way to find out whether the
        /// advertised list is complete; the fixture's union key would hide a
        /// missing index until it became undefined behaviour in a layer.
        std::unique_ptr<heongpu::Galoiskey<S>> narrow_key(std::vector<int>
                                                              shifts)
        {
            auto key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*key, *secret);
            return key;
        }

        std::vector<double> uniform(double lo, double hi, uint64_t seed)
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(lo, hi);
            std::vector<double> v(slots);
            for (double& x : v)
            {
                x = dist(rng);
            }
            return v;
        }
    };

    // -----------------------------------------------------------------------
    // Slot reductions
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, SumStridedReducesTheSlowAxis)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-1.0, 1.0, 11);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->sum_strided(cipher, stride, count, *galois);
        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                total += values[j * stride + i];
            }
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = total;
            }
        }

        EXPECT_LT(max_error(got, want), 1e-6);
    }

    TEST_F(Llama3Env, SumBlockedReducesContiguousBlocks)
    {
        const int span = 16;

        const std::vector<double> values = uniform(-1.0, 1.0, 12);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->sum_blocked(cipher, span, *galois);
        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int base = 0; base < slots; base += span)
        {
            double total = 0.0;
            for (int j = 0; j < span; j++)
            {
                total += values[base + j];
            }
            for (int j = 0; j < span; j++)
            {
                want[base + j] = total;
            }
        }

        EXPECT_LT(max_error(got, want), 1e-5);
    }

    /// Plaintext operands below the top of the chain.
    ///
    /// HEEncoder always encodes at the top level and multiply_plain insists
    /// the two operands agree, so every plaintext step is a no-op away from
    /// throwing once it runs anywhere but on a fresh ciphertext. Testing the
    /// primitives only at depth zero hides that completely.
    TEST_F(Llama3Env, PlaintextStepsWorkBelowTheTopLevel)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-1.0, 1.0, 13);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        ops->square(cipher, *relin);
        ASSERT_EQ(cipher.depth(), 1);
        ops->sum_strided(cipher, stride, count, *galois);
        ops->multiply_constant(cipher, 0.25);
        ASSERT_EQ(cipher.depth(), 2);

        std::vector<double> mask(slots, 2.0);
        ops->multiply_vector(cipher, mask);
        ASSERT_EQ(cipher.depth(), 3);

        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = total * 0.25 * 2.0;
            }
        }

        EXPECT_LT(max_error(got, want), 1e-5);
    }

    // -----------------------------------------------------------------------
    // Polynomial primitives
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, InverseSqrtMatchesTheFunction)
    {
        const std::vector<double> values = uniform(0.5, 2.0, 21);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->inverse_sqrt(cipher, 0.5, 2.0, 15, 2, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = 1.0 / std::sqrt(values[i]);
        }

        EXPECT_LT(reported("inverse_sqrt", got, want), 1e-6);
    }

    TEST_F(Llama3Env, InverseMatchesTheFunction)
    {
        const std::vector<double> values = uniform(2.0, 48.0, 22);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->inverse(cipher, 2.0, 48.0, 15, 2, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = 1.0 / values[i];
        }

        EXPECT_LT(reported("inverse", got, want), 1e-6);
    }

    TEST_F(Llama3Env, SiluMatchesTheActivation)
    {
        // Table 2 puts the calibrated SiLU input inside about 10.8, and
        // Section 3.1.3 reports degree 31 for that range.
        const double bound = 11.0;
        const std::vector<double> values = uniform(-bound, bound, 23);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result = ops->silu(cipher, bound, 31, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] / (1.0 + std::exp(-values[i]));
        }

        EXPECT_LT(reported("silu", got, want), 1e-3);
    }

    // -----------------------------------------------------------------------
    // Layers
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, RopeAppliesTheRotation)
    {
        const int half = kRopeSwap;
        const std::vector<double> values = uniform(-1.0, 1.0, 31);

        // A head of 2 * half channels: the first half pairs with the second,
        // and the swap is a single slot rotation.
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int i = 0; i < slots; i++)
        {
            const double theta = 0.01 * ((i / (2 * half)) + 1) *
                                 ((i % half) + 1);
            const bool lower = (i % (2 * half)) < half;
            cos_values[i] = std::cos(theta);
            sin_values[i] = lower ? -std::sin(theta) : std::sin(theta);
        }

        heongpu::Ciphertext<S> cipher = encrypt(values);
        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, cos_values, scale);
        encoder->encode(sin_plain, sin_values, scale);

        heongpu::Ciphertext<S> result =
            ops->rope(cipher, cos_plain, sin_plain, half, *galois);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] * cos_values[i] +
                      values[(i + half) % slots] * sin_values[i];
        }

        EXPECT_LT(max_error(got, want), 1e-6);
    }

    TEST_F(Llama3Env, RMSNormMatchesThePlaintextLayer)
    {
        const int count = 64;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count;
        config.eps = 1e-5;
        config.sum_lo = 32.0;
        config.sum_hi = 128.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(41);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        std::vector<double> weight(slots);
        for (int i = 0; i < slots; i++)
        {
            weight[i] = 0.5 + 0.5 * ((i % 7) / 7.0);
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values)};
        heongpu::Plaintext<S> weight_plain(context);
        encoder->encode(weight_plain, weight, scale);
        std::vector<heongpu::Plaintext<S>> weights{weight_plain};

        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, weights, config, *galois, *relin);
        ASSERT_EQ(out.size(), 1u);
        const std::vector<double> got = decrypt(out[0]);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            const double factor =
                1.0 / std::sqrt(total / count + config.eps);
            for (int j = 0; j < count; j++)
            {
                const int p = j * stride + i;
                want[p] = values[p] * factor * weight[p];
            }
        }

        EXPECT_LT(reported("rms_norm", got, want), 1e-6);
    }

    TEST_F(Llama3Env, SoftmaxNormalisesEachInstance)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.count = 32;
        config.stride = slots / config.count;
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        const std::vector<double> values = uniform(-config.bound, 0.0, 51);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[j * config.stride + i]);
            }
            for (int j = 0; j < config.count; j++)
            {
                const int p = j * config.stride + i;
                want[p] = std::exp(values[p]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=1 strided", got, want), 1e-6);

        // Whatever the approximation error, every instance must still sum to
        // one: that is what the normalise-and-square round enforces.
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += got[j * config.stride + i];
            }
            EXPECT_NEAR(total, 1.0, 1e-3) << "instance " << i;
        }
    }

    TEST_F(Llama3Env, PcmmProducesTauOfTheProduct)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        ASSERT_EQ(layout.slots, slots);

        std::mt19937_64 rng(61);
        const int batch = layout.batch;

        for (int ell = 0; ell <= 1; ell++)
        {
            std::vector<std::vector<double>> a(batch);
            std::vector<std::vector<double>> b(batch);
            std::vector<double> stored;
            std::vector<double> operand_slots(slots, 0.0);
            std::vector<double> want(slots, 0.0);

            for (int m = 0; m < batch; m++)
            {
                a[m] = random_matrix(d, rng);
                b[m] = random_matrix(d, rng);

                const std::vector<double> stored_m =
                    apply_tau(llama::permute_sigma(a[m], d), d, ell);
                stored.insert(stored.end(), stored_m.begin(), stored_m.end());

                const std::vector<double> operand =
                    apply_tau(b[m], d, ell + 1);
                const std::vector<double> expected =
                    apply_tau(llama::matmul_host(a[m], b[m], d), d, ell);

                for (int e = 0; e < d * d; e++)
                {
                    operand_slots[e * batch + m] = operand[e];
                    want[e * batch + m] = expected[e];
                }
            }

            heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
            heongpu::Ciphertext<S> result =
                ops->pcmm(cipher, stored, layout, 4, 4, ell, *galois);
            const std::vector<double> got = decrypt(result);

            EXPECT_LT(reported("pcmm", got, want), 1e-6) << "ell=" << ell;
        }
    }

    // -----------------------------------------------------------------------
    // Branches the first round of tests never reached
    // -----------------------------------------------------------------------

    /// A polynomial result must be usable, not merely correct.
    ///
    /// evaluate_poly rescales its result only when the scale has grown past
    /// half the target, so it can hand back a ciphertext with a rescale still
    /// pending. multiply, rotate and mod_drop all refuse such a ciphertext, so
    /// a primitive that returns one is a delayed exception rather than a
    /// value. Squaring and rotating the output is the cheapest way to say so.
    TEST_F(Llama3Env, PolynomialResultsAreReadyForFurtherWork)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-11.0, 11.0, 71);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> activated = ops->silu(cipher, 11.0, 15, *relin);
        ASSERT_NO_THROW(ops->sum_strided(activated, stride, count, *galois));
        ASSERT_NO_THROW(ops->square(activated, *relin));

        heongpu::Ciphertext<S> wide = encrypt(uniform(2.0, 48.0, 72));
        heongpu::Ciphertext<S> inverted =
            ops->inverse(wide, 2.0, 48.0, 15, 1, *relin);
        ASSERT_NO_THROW(ops->multiply_constant(inverted, 2.0));

        // newton_iterations = 0 returns the bare Chebyshev seed, the one path
        // that leaves evaluate_poly's output completely untouched.
        heongpu::Ciphertext<S> narrow = encrypt(uniform(0.5, 2.0, 73));
        heongpu::Ciphertext<S> root =
            ops->inverse_sqrt(narrow, 0.5, 2.0, 15, 0, *relin);
        ASSERT_NO_THROW(ops->square(root, *relin));
    }

    /// The channel split, which is the whole reason rms_norm takes a vector.
    TEST_F(Llama3Env, RMSNormSplitsChannelsOverSeveralCiphertexts)
    {
        const int count = 32;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = 2 * count; // two ciphertexts of count channels
        config.eps = 1e-5;
        config.sum_lo = 20.0;
        config.sum_hi = 150.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(81);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<std::vector<double>> values(2,
                                                std::vector<double>(slots));
        for (auto& v : values)
        {
            for (double& x : v)
            {
                x = dist(rng);
            }
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values[0]),
                                               encrypt(values[1])};
        std::vector<heongpu::Plaintext<S>> no_weights;

        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, no_weights, config, *galois, *relin);
        ASSERT_EQ(out.size(), 2u);

        for (int part = 0; part < 2; part++)
        {
            const std::vector<double> got = decrypt(out[part]);
            std::vector<double> want(slots);
            for (int i = 0; i < stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < count; j++)
                {
                    for (int p = 0; p < 2; p++)
                    {
                        const double v = values[p][j * stride + i];
                        total += v * v;
                    }
                }
                const double factor =
                    1.0 / std::sqrt(total / config.channels + config.eps);
                for (int j = 0; j < count; j++)
                {
                    const int p = j * stride + i;
                    want[p] = values[part][p] * factor;
                }
            }
            EXPECT_LT(max_error(got, want), 1e-6) << "part " << part;
        }
    }

    /// The blocked layout, which is what a SoftMax over the token axis needs
    /// when the reduced axis is contiguous rather than strided.
    TEST_F(Llama3Env, SoftmaxNormalisesContiguousBlocks)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = false;
        config.count = 32;
        config.stride = 0; // unused in the blocked layout
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        const std::vector<double> values = uniform(-config.bound, 0.0, 91);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int base = 0; base < slots; base += config.count)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[base + j]);
            }
            for (int j = 0; j < config.count; j++)
            {
                want[base + j] = std::exp(values[base + j]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=1 blocked", got, want), 1e-6);
    }

    /// One weight matrix shared by every matrix in the batch, the form a
    /// projection actually takes: one plaintext W against many token blocks.
    TEST_F(Llama3Env, PcmmSharesOneBlockAcrossTheBatch)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;

        std::mt19937_64 rng(101);
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<double> operand_slots(slots, 0.0);
        std::vector<double> want(slots, 0.0);
        for (int m = 0; m < batch; m++)
        {
            const std::vector<double> b = random_matrix(d, rng);
            const std::vector<double> operand = llama::permute_tau(b, d);
            const std::vector<double> expected = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                operand_slots[e * batch + m] = operand[e];
                want[e * batch + m] = expected[e];
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
        heongpu::Ciphertext<S> result =
            ops->pcmm(cipher, stored, layout, 4, 4, 0, *galois);
        EXPECT_LT(max_error(decrypt(result), want), 1e-6);
    }

    /// Lopsided BSGS splits. The rotation sets differ from the square split,
    /// so a key or an index that is only right when giant == baby shows here.
    TEST_F(Llama3Env, PcmmHandlesLopsidedBsgsSplits)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;

        std::mt19937_64 rng(111);
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<double> operand_slots(slots, 0.0);
        std::vector<double> want(slots, 0.0);
        for (int m = 0; m < batch; m++)
        {
            const std::vector<double> b = random_matrix(d, rng);
            const std::vector<double> operand = llama::permute_tau(b, d);
            const std::vector<double> expected = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                operand_slots[e * batch + m] = operand[e];
                want[e * batch + m] = expected[e];
            }
        }

        for (int baby : {2, 8})
        {
            heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
            heongpu::Ciphertext<S> result =
                ops->pcmm(cipher, stored, layout, d / baby, baby, 0, *galois);
            EXPECT_LT(max_error(decrypt(result), want), 1e-6)
                << "baby=" << baby;
        }
    }

    /// A weight plaintext is prepared once and used at every level it meets.
    /// multiply_plaintext promises to take the drop on a copy; if it dropped
    /// the caller's plaintext instead, the second use would be at the wrong
    /// level and would either throw or decode as noise.
    TEST_F(Llama3Env, PlaintextOperandSurvivesReuseAtSeveralLevels)
    {
        const std::vector<double> values = uniform(-1.0, 1.0, 121);
        std::vector<double> weight(slots);
        for (int i = 0; i < slots; i++)
        {
            weight[i] = 0.25 + 0.5 * ((i % 5) / 5.0);
        }

        heongpu::Plaintext<S> weight_plain(context);
        encoder->encode(weight_plain, weight, scale);

        // Deep first, so a mutated plaintext would break the shallow use.
        heongpu::Ciphertext<S> deep = encrypt(values);
        ops->square(deep, *relin);
        ops->multiply_constant(deep, 0.5);
        ASSERT_EQ(deep.depth(), 2);
        ops->multiply_plaintext(deep, weight_plain);
        ops->rescale_inplace(deep);

        heongpu::Ciphertext<S> shallow = encrypt(values);
        ASSERT_EQ(shallow.depth(), 0);
        ops->multiply_plaintext(shallow, weight_plain);
        ops->rescale_inplace(shallow);

        std::vector<double> want_deep(slots);
        std::vector<double> want_shallow(slots);
        for (int i = 0; i < slots; i++)
        {
            want_deep[i] = values[i] * values[i] * 0.5 * weight[i];
            want_shallow[i] = values[i] * weight[i];
        }

        EXPECT_LT(max_error(decrypt(deep), want_deep), 1e-6);
        EXPECT_LT(max_error(decrypt(shallow), want_shallow), 1e-6);
    }

    // -----------------------------------------------------------------------
    // The advertised rotation indices are the whole contract
    // -----------------------------------------------------------------------

    /// Each reduction, given a key holding only what it asked for.
    TEST_F(Llama3Env, ReductionsAskForEveryRotationTheyUse)
    {
        for (int count : {2, 8, 32, 128})
        {
            SCOPED_TRACE("strided count=" + std::to_string(count));
            const int stride = slots / count;
            auto key = narrow_key(
                llama::Llama3Operator::strided_rotation_indices(stride, count));

            const std::vector<double> values = uniform(-1.0, 1.0, 200 + count);
            heongpu::Ciphertext<S> cipher = encrypt(values);
            ASSERT_NO_THROW(ops->sum_strided(cipher, stride, count, *key));

            std::vector<double> want(slots);
            for (int i = 0; i < stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < count; j++)
                {
                    total += values[j * stride + i];
                }
                for (int j = 0; j < count; j++)
                {
                    want[j * stride + i] = total;
                }
            }
            EXPECT_LT(max_error(decrypt(cipher), want), 1e-5);
        }

        for (int span : {2, 8, 64})
        {
            SCOPED_TRACE("blocked span=" + std::to_string(span));
            auto key = narrow_key(
                llama::Llama3Operator::blocked_rotation_indices(span));

            const std::vector<double> values = uniform(-1.0, 1.0, 300 + span);
            heongpu::Ciphertext<S> cipher = encrypt(values);
            ASSERT_NO_THROW(ops->sum_blocked(cipher, span, *key));

            std::vector<double> want(slots);
            for (int base = 0; base < slots; base += span)
            {
                double total = 0.0;
                for (int j = 0; j < span; j++)
                {
                    total += values[base + j];
                }
                for (int j = 0; j < span; j++)
                {
                    want[base + j] = total;
                }
            }
            EXPECT_LT(max_error(decrypt(cipher), want), 1e-5);
        }
    }

    /// PCMM over every square matrix that fills these slots, both BSGS
    /// orientations and both tau exponents, each against a key built from
    /// pcmm_rotation_indices alone. d = 64 leaves batch = 1, the degenerate
    /// packing where the batch axis disappears.
    TEST_F(Llama3Env, PcmmCoversEveryShapeWithTheKeysItAsksFor)
    {
        std::mt19937_64 rng(401);
        constexpr int kElls = 2;

        for (int d : {8, 16, 32, 64})
        {
            const llama::MatrixLayout layout(d, slots / (d * d));
            ASSERT_EQ(layout.slots, slots);
            const int batch = layout.batch;
            const std::size_t entries = static_cast<std::size_t>(d) * d;

            // One operand set per tau exponent, built once: the rotation keys
            // do not depend on ell, so keeping the splits outside keeps the
            // key generations down to one per split.
            std::vector<std::vector<double>> stored(kElls);
            std::vector<std::vector<double>> operand_slots(
                kElls, std::vector<double>(slots, 0.0));
            std::vector<std::vector<double>> want(
                kElls, std::vector<double>(slots, 0.0));

            for (int m = 0; m < batch; m++)
            {
                const std::vector<double> a = random_matrix(d, rng);
                const std::vector<double> b = random_matrix(d, rng);
                const std::vector<double> product = llama::matmul_host(a, b, d);
                const std::vector<double> sigma_a = llama::permute_sigma(a, d);

                for (int ell = 0; ell < kElls; ell++)
                {
                    const std::vector<double> stored_m =
                        apply_tau(sigma_a, d, ell);
                    stored[ell].insert(stored[ell].end(), stored_m.begin(),
                                       stored_m.end());

                    const std::vector<double> operand =
                        apply_tau(b, d, ell + 1);
                    const std::vector<double> expected =
                        apply_tau(product, d, ell);
                    for (std::size_t e = 0; e < entries; e++)
                    {
                        operand_slots[ell][e * batch + m] = operand[e];
                        want[ell][e * batch + m] = expected[e];
                    }
                }
            }

            for (int baby = 2; baby <= d / 2; baby <<= 1)
            {
                const int giant = d / baby;
                auto key =
                    narrow_key(llama::Llama3Operator::pcmm_rotation_indices(
                        layout, giant, baby));

                for (int ell = 0; ell < kElls; ell++)
                {
                    SCOPED_TRACE("d=" + std::to_string(d) + " ell=" +
                                 std::to_string(ell) + " giant=" +
                                 std::to_string(giant) + " baby=" +
                                 std::to_string(baby));

                    heongpu::Ciphertext<S> cipher =
                        encrypt(operand_slots[ell]);
                    heongpu::Ciphertext<S> result(context);
                    ASSERT_NO_THROW(result = ops->pcmm(cipher, stored[ell],
                                                       layout, giant, baby,
                                                       ell, *key));
                    // The product sums d terms, so the error grows with d.
                    EXPECT_LT(max_error(decrypt(result), want[ell]), 1e-6 * d);
                }
            }
        }
    }

    /// RoPE below the top of the chain, on a key holding only its own shift.
    TEST_F(Llama3Env, RopeWorksBelowTheTopOfTheChain)
    {
        const int half = kRopeSwap;
        auto key = narrow_key({half});

        const std::vector<double> values = uniform(-1.0, 1.0, 501);
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int i = 0; i < slots; i++)
        {
            const double theta = 0.01 * ((i / (2 * half)) + 1) * ((i % half) + 1);
            cos_values[i] = std::cos(theta);
            sin_values[i] = ((i % (2 * half)) < half) ? -std::sin(theta)
                                                      : std::sin(theta);
        }

        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, cos_values, scale);
        encoder->encode(sin_plain, sin_values, scale);

        // Square first, so the plaintexts arrive three levels above the
        // ciphertext and must be dropped to meet it.
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->square(cipher, *relin);
        ops->multiply_constant(cipher, 0.5);
        ASSERT_EQ(cipher.depth(), 2);

        heongpu::Ciphertext<S> result =
            ops->rope(cipher, cos_plain, sin_plain, half, *key);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            const double x = 0.5 * values[i] * values[i];
            const int j = (i + half) % slots;
            want[i] = x * cos_values[i] +
                      0.5 * values[j] * values[j] * sin_values[i];
        }
        EXPECT_LT(reported("rope at depth 2", decrypt(result), want), 1e-6);
    }

    /// A channel count that does not fill the last ciphertext.
    ///
    /// The guard admits this on purpose: padding contributes zero squares, so
    /// the mean is still over the real channels only.
    TEST_F(Llama3Env, RMSNormAcceptsAPaddedLastInput)
    {
        const int count = 32;
        const int stride = slots / count;
        const int real = 20; // channels carried by the second ciphertext

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count + real;
        config.eps = 1e-5;
        config.sum_lo = 20.0;
        config.sum_hi = 110.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(601);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<std::vector<double>> values(2,
                                                std::vector<double>(slots, 0.0));
        for (double& x : values[0])
        {
            x = dist(rng);
        }
        for (int j = 0; j < real; j++)
        {
            for (int i = 0; i < stride; i++)
            {
                values[1][j * stride + i] = dist(rng);
            }
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values[0]),
                                               encrypt(values[1])};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, no_weights, config, *galois, *relin);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                for (int p = 0; p < 2; p++)
                {
                    const double v = values[p][j * stride + i];
                    total += v * v;
                }
            }
            const double factor =
                1.0 / std::sqrt(total / config.channels + config.eps);
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = values[0][j * stride + i] * factor;
            }
        }
        EXPECT_LT(reported("rms_norm padded", decrypt(out[0]), want), 1e-6);
    }

    /// The mistakes that would otherwise pass silently.
    ///
    /// Each of these produces a plausible-looking ciphertext rather than an
    /// error: a wrong channel count rescales every output by a constant, a
    /// mismatched RoPE pair weights the two terms differently, and too short a
    /// chain reaches a kernel launch with a nonpositive extent.
    TEST_F(Llama3Env, WrongConfigurationIsRefusedRatherThanApproximated)
    {
        const int count = 32;
        const int stride = slots / count;
        const std::vector<double> values = uniform(-1.0, 1.0, 141);

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.eps = 1e-5;
        config.sum_lo = 8.0;
        config.sum_hi = 72.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::vector<heongpu::Ciphertext<S>> two{encrypt(values),
                                                encrypt(values)};
        std::vector<heongpu::Plaintext<S>> no_weights;

        // Forgetting that a second ciphertext doubles the channel count.
        config.channels = count;
        EXPECT_THROW(ops->rms_norm(two, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        // Counting channels that no input actually carries.
        config.channels = 3 * count;
        EXPECT_THROW(ops->rms_norm(two, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        config.channels = 2 * count;
        EXPECT_NO_THROW(ops->rms_norm(two, no_weights, config, *galois,
                                      *relin));

        // Inputs that have drifted apart in level.
        std::vector<heongpu::Ciphertext<S>> uneven{encrypt(values),
                                                   encrypt(values)};
        ops->multiply_constant(uneven[1], 1.0);
        EXPECT_THROW(ops->rms_norm(uneven, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        // A RoPE pair encoded at two different scales.
        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, std::vector<double>(slots, 1.0), scale);
        encoder->encode(sin_plain, std::vector<double>(slots, 1.0), scale / 2);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        EXPECT_THROW(ops->rope(cipher, cos_plain, sin_plain, kRopeSwap,
                               *galois),
                     std::invalid_argument);

        // A polynomial that does not fit in what is left of the chain.
        heongpu::Ciphertext<S> shallow = encrypt(values);
        ops->drop_to_depth(shallow, kLimbs - 3);
        EXPECT_THROW(ops->silu(shallow, 11.0, 31, *relin),
                     std::invalid_argument);
    }

    /// Primitives back to back on one ciphertext.
    ///
    /// Each test above starts from a fresh encryption, which is exactly the
    /// state in which the plaintext-level bug was invisible. A layer never
    /// does that, so this runs a normalisation, a projection and an activation
    /// in sequence and checks the result against the same sequence on doubles.
    TEST_F(Llama3Env, PrimitivesComposeIntoOneChain)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;
        const int count = 32;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count;
        config.eps = 1e-5;
        config.sum_lo = 8.0;
        config.sum_hi = 72.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(131);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values)};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normed =
            ops->rms_norm(in, no_weights, config, *galois, *relin);

        // The projection consumes tau(X), so the normalised block is read in
        // that layout; the chain is what matters here, not the packing.
        heongpu::Ciphertext<S> projected =
            ops->pcmm(normed[0], stored, layout, 4, 4, 0, *galois);
        // Each product is a sum of d terms of size about one, so the range is
        // several times wider than the normalised input it came from.
        heongpu::Ciphertext<S> activated =
            ops->silu(projected, 14.0, 31, *relin);

        // The same sequence on doubles.
        std::vector<double> normalised(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            const double factor =
                1.0 / std::sqrt(total / config.channels + config.eps);
            for (int j = 0; j < count; j++)
            {
                normalised[j * stride + i] = values[j * stride + i] * factor;
            }
        }

        std::vector<double> want(slots);
        for (int m = 0; m < batch; m++)
        {
            std::vector<double> block(static_cast<std::size_t>(d) * d);
            for (int e = 0; e < d * d; e++)
            {
                block[e] = normalised[e * batch + m];
            }
            // pcmm was handed tau(B) and returns A B, so undo tau to recover B.
            std::vector<double> b(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    b[((i + j) % d) * d + j] = block[i * d + j];
                }
            }
            const std::vector<double> product = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                const double x = product[e];
                want[e * batch + m] = x / (1.0 + std::exp(-x));
            }
        }

        EXPECT_LT(reported("rms_norm -> pcmm -> silu", decrypt(activated), want),
                  1e-2);
    }

    // -----------------------------------------------------------------------
    // The SoftMax the paper actually specifies
    // -----------------------------------------------------------------------

    /// Section 4.3 fixes two normalise-and-square rounds, which does not fit
    /// the 24-limb chain the other tests share, so this one gets its own.
    ///
    /// Two rounds matter beyond costing more: only the second round runs on an
    /// input that already sums to one, which is the regime the round bounds
    /// switch to, and the reciprocal is approximated over a much wider range
    /// there than in the first round.
    class Llama3DeepEnv : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 8192;
        static constexpr int kLimbs = 32;
        static constexpr int kCount = 8;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();

            std::vector<int> shifts =
                llama::Llama3Operator::strided_rotation_indices(slots / kCount,
                                                                kCount);
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }
    };

    TEST_F(Llama3DeepEnv, SoftmaxRunsTheTwoRoundsOfSectionFourPointThree)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.count = kCount;
        config.stride = slots / kCount;
        config.bound = 2.0;
        config.iterations = 2;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        std::mt19937_64 rng(701);
        std::uniform_real_distribution<double> dist(-config.bound, 0.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        heongpu::Plaintext<S> plain(context);
        encoder->encode(plain, values, scale);
        heongpu::Ciphertext<S> cipher(context);
        encryptor->encrypt(cipher, plain);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);

        heongpu::Plaintext<S> out_plain(context);
        decryptor->decrypt(out_plain, result);
        std::vector<double> got;
        encoder->decode(got, out_plain);

        std::vector<double> want(slots);
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[j * config.stride + i]);
            }
            for (int j = 0; j < config.count; j++)
            {
                const int p = j * config.stride + i;
                want[p] = std::exp(values[p]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=2", got, want), 1e-6);

        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += got[j * config.stride + i];
            }
            EXPECT_NEAR(total, 1.0, 1e-3) << "instance " << i;
        }
    }

    // -----------------------------------------------------------------------
    // The sublayers
    // -----------------------------------------------------------------------

    /// Attention and the feed-forward network, on a chain long enough to hold
    /// one of them.
    ///
    /// Activations are held TRANSPOSED, X[channel][token]. That is what makes
    /// a projection W X, which is where Equation (5) wants the plaintext, and
    /// it puts the channel axis on the slow axis, where a reduction is exact
    /// and free. The matrices are small on purpose: these check that the
    /// sublayers compute what a transformer sublayer computes, not that a
    /// 4096-wide one fits, which it does not without bootstrapping.
    class Llama3LayerEnv : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 4096;
        // Attention under a pre-norm and a residual is the deepest thing here,
        // at about thirty levels.
        static constexpr int kLimbs = 38;
        static constexpr int kD = 8;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;
        llama::MatrixLayout layout;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();
            layout = llama::MatrixLayout(kD, slots / (kD * kD));

            // Exactly what attention advertises, and nothing more. The
            // feed-forward list and the strided reduction a pre-norm needs are
            // both subsets of it, so this key verifies all three: a shift left
            // out of any of those lists is undefined behaviour here, not a
            // missing-key error.
            llama::Llama3Operator::AttentionConfig index_config;
            index_config.layout = layout;
            index_config.rope = true;
            std::vector<int> shifts =
                llama::Llama3Operator::attention_rotation_indices(index_config);

            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& values)
        {
            heongpu::Plaintext<S> plain(context);
            encoder->encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor->encrypt(cipher, plain);
            return cipher;
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& cipher)
        {
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, cipher);
            std::vector<double> values;
            encoder->decode(values, plain);
            return values;
        }

        std::unique_ptr<heongpu::Galoiskey<S>> narrow_key(std::vector<int>
                                                              shifts)
        {
            auto key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*key, *secret);
            return key;
        }

        /// RMSNorm over the channel axis of the packed layout, which is the
        /// slow axis and therefore the free one.
        llama::Llama3Operator::RMSNormConfig
        norm_config(const std::vector<double>& x) const
        {
            llama::Llama3Operator::RMSNormConfig config;
            config.stride = layout.d * layout.batch;
            config.count = layout.d;
            config.channels = layout.d;
            config.eps = 1e-5;
            config.degree = 31;
            config.newton_iterations = 0;

            double lowest = std::numeric_limits<double>::max();
            double highest = 0.0;
            for (int i = 0; i < config.stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < config.count; j++)
                {
                    const double v = x[j * config.stride + i];
                    total += v * v;
                }
                lowest = std::min(lowest, total);
                highest = std::max(highest, total);
            }
            config.sum_lo = 0.8 * lowest;
            config.sum_hi = 1.2 * highest;
            return config;
        }

        std::vector<double> rms_norm_host(
            const std::vector<double>& x,
            const llama::Llama3Operator::RMSNormConfig& config) const
        {
            std::vector<double> out(x.size());
            for (int i = 0; i < config.stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < config.count; j++)
                {
                    const double v = x[j * config.stride + i];
                    total += v * v;
                }
                const double factor =
                    1.0 / std::sqrt(total / config.channels + config.eps);
                for (int j = 0; j < config.count; j++)
                {
                    out[j * config.stride + i] =
                        x[j * config.stride + i] * factor;
                }
            }
            return out;
        }
    };

    TEST_F(Llama3LayerEnv, TauPermutesEveryBlock)
    {
        auto key = narrow_key(
            llama::Llama3Operator::tau_rotation_indices(layout));

        std::mt19937_64 rng(810);
        const std::vector<std::vector<double>> blocks =
            random_blocks(layout, 1.0, rng);
        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(blocks, layout));

        heongpu::Ciphertext<S> result = ops->tau(cipher, layout, *key);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::permute_tau(blocks[m], layout.d);
        }
        EXPECT_LT(reported("tau", decrypt(result), pack_blocks(want, layout)),
                  1e-6);
    }

    TEST_F(Llama3LayerEnv, TransposeExchangesRowsAndColumns)
    {
        auto key = narrow_key(
            llama::Llama3Operator::transpose_rotation_indices(layout));

        std::mt19937_64 rng(811);
        const std::vector<std::vector<double>> blocks =
            random_blocks(layout, 1.0, rng);
        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(blocks, layout));

        heongpu::Ciphertext<S> result = ops->transpose(cipher, layout, *key);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::transpose_host(blocks[m], layout.d);
        }
        EXPECT_LT(
            reported("transpose", decrypt(result), pack_blocks(want, layout)),
            1e-6);
    }

    /// The product Section 4.2 does not cover, with both operands encrypted.
    TEST_F(Llama3LayerEnv, CcmmMultipliesTwoEncryptedBlocks)
    {
        auto key = narrow_key(
            llama::Llama3Operator::ccmm_rotation_indices(layout));

        std::mt19937_64 rng(812);
        const std::vector<std::vector<double>> a = random_blocks(layout, 1.0,
                                                                 rng);
        const std::vector<std::vector<double>> b = random_blocks(layout, 1.0,
                                                                 rng);

        heongpu::Ciphertext<S> ca = encrypt(pack_blocks(a, layout));
        heongpu::Ciphertext<S> cb = encrypt(pack_blocks(b, layout));

        // A scaling that rides on masks the algorithm encodes anyway, so it
        // has to come out exactly as if it had been applied afterwards.
        const double factor = 0.25;
        heongpu::Ciphertext<S> result =
            ops->ccmm(ca, cb, layout, factor, *key, *relin);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::matmul_host(a[m], b[m], layout.d);
            for (double& v : want[m])
            {
                v *= factor;
            }
        }
        EXPECT_LT(reported("ccmm", decrypt(result), pack_blocks(want, layout)),
                  1e-6);
    }

    /// A residual connection joins two ciphertexts that share neither level
    /// nor scale, and CKKS addition needs both.
    TEST_F(Llama3LayerEnv, ResidualAddReconcilesLevelAndScale)
    {
        std::mt19937_64 rng(813);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> values(slots);
        for (double& v : values)
        {
            v = dist(rng);
        }

        heongpu::Ciphertext<S> skip = encrypt(values);
        heongpu::Ciphertext<S> sublayer = encrypt(values);
        ops->square(sublayer, *relin);

        ASSERT_NE(skip.depth(), sublayer.depth());
        ASSERT_NE(skip.scale(), sublayer.scale());

        heongpu::Ciphertext<S> out = ops->residual_add(skip, sublayer);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] + values[i] * values[i];
        }
        EXPECT_LT(reported("residual", decrypt(out), want), 1e-6);
    }

    TEST_F(Llama3LayerEnv, SoftmaxDropsTheKeysACausalMaskHides)
    {
        const int d = layout.d;
        auto key = narrow_key(llama::Llama3Operator::strided_rotation_indices(
            d * layout.batch, d));

        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.stride = d * layout.batch;
        config.count = d;
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        std::mt19937_64 rng(814);
        std::uniform_real_distribution<double> dist(-config.bound, 0.0);
        std::vector<std::vector<double>> scores(
            layout.batch, std::vector<double>(static_cast<std::size_t>(d) * d));
        for (auto& block : scores)
        {
            for (double& v : block)
            {
                v = dist(rng);
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(scores, layout));
        const std::vector<double> mask =
            llama::Llama3Operator::causal_mask(layout);
        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, mask, *key, *relin);

        // Row index is the key and column index the query, so a query sums
        // over the keys at or before it.
        std::vector<std::vector<double>> want(
            layout.batch,
            std::vector<double>(static_cast<std::size_t>(d) * d, 0.0));
        for (int m = 0; m < layout.batch; m++)
        {
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int k = 0; k <= query; k++)
                {
                    total += std::exp(scores[m][k * d + query]);
                }
                for (int k = 0; k <= query; k++)
                {
                    want[m][k * d + query] =
                        std::exp(scores[m][k * d + query]) / total;
                }
            }
        }

        EXPECT_LT(reported("causal softmax", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-6);
    }

    TEST_F(Llama3LayerEnv, FeedForwardMatchesSwiGLU)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        auto key = narrow_key(
            llama::Llama3Operator::feed_forward_rotation_indices(config));

        std::mt19937_64 rng(815);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> gate =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> up =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> down =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::FeedForwardWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.gate.insert(weights.gate.end(), gate[m].begin(),
                                gate[m].end());
            weights.up.insert(weights.up.end(), up[m].begin(), up[m].end());
            weights.down.insert(weights.down.end(), down[m].begin(),
                                down[m].end());
        }

        std::vector<std::vector<double>> want(layout.batch);
        double widest = 0.0;
        for (int m = 0; m < layout.batch; m++)
        {
            const std::vector<double> g =
                llama::matmul_host(gate[m], x[m], d);
            const std::vector<double> u = llama::matmul_host(up[m], x[m], d);
            std::vector<double> hidden(g.size());
            for (std::size_t e = 0; e < g.size(); e++)
            {
                widest = std::max(widest, std::abs(g[e]));
                hidden[e] = silu_host(g[e]) * u[e];
            }
            want[m] = llama::matmul_host(down[m], hidden, d);
        }
        // The activation is a Chebyshev fit on a fixed interval, so a gate
        // outside it would be extrapolation rather than approximation.
        ASSERT_LT(widest, config.silu_bound);

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(x, layout));
        heongpu::Ciphertext<S> result =
            ops->feed_forward(cipher, weights, config, *key, *relin);

        // The only bound here that is not on the noise floor: the degree 31
        // SiLU fit dominates and its error is then summed over d terms by the
        // down projection.
        EXPECT_LT(reported("feed_forward", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-5);
    }

    /// The whole attention sublayer: projections, RoPE, scores, a causal
    /// SoftMax, the value product and the output projection.
    TEST_F(Llama3LayerEnv, AttentionMatchesTheReference)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(816);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> wq =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wk =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wv =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wo =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::AttentionWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.query.insert(weights.query.end(), wq[m].begin(),
                                 wq[m].end());
            weights.key.insert(weights.key.end(), wk[m].begin(), wk[m].end());
            weights.value.insert(weights.value.end(), wv[m].begin(),
                                 wv[m].end());
            weights.output.insert(weights.output.end(), wo[m].begin(),
                                  wo[m].end());
        }

        // RoPE, in slot space, exactly as the layer applies it: one rotation
        // by half the channel axis, with the sign of the sine term carried by
        // the plaintext.
        const int swap = llama::Llama3Operator::rope_swap_shift(layout);
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int r = 0; r < d; r++)
        {
            for (int c = 0; c < d; c++)
            {
                const double theta =
                    0.05 * (c + 1) * std::pow(2.0, -(r % (d / 2)));
                for (int m = 0; m < layout.batch; m++)
                {
                    const int p = (r * d + c) * layout.batch + m;
                    cos_values[p] = std::cos(theta);
                    sin_values[p] =
                        (r < d / 2) ? -std::sin(theta) : std::sin(theta);
                }
            }
        }
        std::vector<heongpu::Plaintext<S>> rope_plain;
        rope_plain.emplace_back(context);
        rope_plain.emplace_back(context);
        encoder->encode(rope_plain[0], cos_values, scale);
        encoder->encode(rope_plain[1], sin_values, scale);

        // The reference, one block at a time.
        std::vector<std::vector<double>> q(layout.batch);
        std::vector<std::vector<double>> k(layout.batch);
        std::vector<std::vector<double>> v(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            q[m] = llama::matmul_host(wq[m], x[m], d);
            k[m] = llama::matmul_host(wk[m], x[m], d);
            v[m] = llama::matmul_host(wv[m], x[m], d);
        }

        std::vector<double> q_slots = pack_blocks(q, layout);
        std::vector<double> k_slots = pack_blocks(k, layout);
        std::vector<double> q_roped(slots);
        std::vector<double> k_roped(slots);
        for (int p = 0; p < slots; p++)
        {
            const int shifted = (p + swap) % slots;
            q_roped[p] =
                q_slots[p] * cos_values[p] + q_slots[shifted] * sin_values[p];
            k_roped[p] =
                k_slots[p] * cos_values[p] + k_slots[shifted] * sin_values[p];
        }
        q = unpack_blocks(q_roped, layout);
        k = unpack_blocks(k_roped, layout);

        // Raw scores, then the calibration the paper takes offline: a scaling
        // that puts the spread at the SoftMax bound and a shift that puts the
        // top of the range at zero.
        std::vector<std::vector<double>> raw(layout.batch);
        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (int m = 0; m < layout.batch; m++)
        {
            raw[m] = llama::matmul_host(
                llama::transpose_host(k[m], d), q[m], d);
            for (double s : raw[m])
            {
                lowest = std::min(lowest, s);
                highest = std::max(highest, s);
            }
        }

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.causal = true;
        config.rope = true;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            std::vector<double> probabilities(
                static_cast<std::size_t>(d) * d, 0.0);
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int key = 0; key <= query; key++)
                {
                    total += std::exp(raw[m][key * d + query] *
                                          config.head_scale -
                                      config.score_shift);
                }
                for (int key = 0; key <= query; key++)
                {
                    probabilities[key * d + query] =
                        std::exp(raw[m][key * d + query] * config.head_scale -
                                 config.score_shift) /
                        total;
                }
            }
            const std::vector<double> out =
                llama::matmul_host(v[m], probabilities, d);
            want[m] = llama::matmul_host(wo[m], out, d);
        }

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(x, layout));
        heongpu::Ciphertext<S> result = ops->attention(
            cipher, weights, rope_plain, config, *galois, *relin);

        // Thirty levels deep and still on the noise floor: nothing in the
        // attention path is approximated over a wide interval, so the bound
        // belongs with the exact primitives rather than with SiLU.
        EXPECT_LT(reported("attention", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-6);
    }

    /// A pre-norm feed-forward block: RMSNorm, the sublayer, the residual.
    TEST_F(Llama3LayerEnv, FeedForwardRunsUnderPreNormAndResidual)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        std::mt19937_64 rng(817);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> gate =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> up =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> down =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::FeedForwardWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.gate.insert(weights.gate.end(), gate[m].begin(),
                                gate[m].end());
            weights.up.insert(weights.up.end(), up[m].begin(), up[m].end());
            weights.down.insert(weights.down.end(), down[m].begin(),
                                down[m].end());
        }

        const std::vector<double> x_slots = pack_blocks(x, layout);
        const llama::Llama3Operator::RMSNormConfig norm = norm_config(x_slots);
        const std::vector<std::vector<double>> normed =
            unpack_blocks(rms_norm_host(x_slots, norm), layout);

        std::vector<std::vector<double>> want(layout.batch);
        double widest = 0.0;
        for (int m = 0; m < layout.batch; m++)
        {
            const std::vector<double> g =
                llama::matmul_host(gate[m], normed[m], d);
            const std::vector<double> u =
                llama::matmul_host(up[m], normed[m], d);
            std::vector<double> hidden(g.size());
            for (std::size_t e = 0; e < g.size(); e++)
            {
                widest = std::max(widest, std::abs(g[e]));
                hidden[e] = silu_host(g[e]) * u[e];
            }
            const std::vector<double> out =
                llama::matmul_host(down[m], hidden, d);
            want[m].resize(out.size());
            for (std::size_t e = 0; e < out.size(); e++)
            {
                want[m][e] = x[m][e] + out[e];
            }
        }
        ASSERT_LT(widest, config.silu_bound);

        heongpu::Ciphertext<S> cipher = encrypt(x_slots);
        std::vector<heongpu::Ciphertext<S>> in{cipher};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normalised =
            ops->rms_norm(in, no_weights, norm, *galois, *relin);

        heongpu::Ciphertext<S> sublayer = ops->feed_forward(
            normalised[0], weights, config, *galois, *relin);
        heongpu::Ciphertext<S> result = ops->residual_add(cipher, sublayer);

        EXPECT_LT(reported("pre-norm ffn block", decrypt(result),
                           pack_blocks(want, layout)),
                  2e-4);
    }

    /// A pre-norm attention block, which is the deepest circuit here.
    ///
    /// No output projection and one normalise-and-square round: the point is
    /// that the sublayer survives arriving below the top of the chain and that
    /// the residual closes over it, and Sylph would bootstrap in the middle of
    /// this rather than stretch the chain.
    TEST_F(Llama3LayerEnv, AttentionRunsUnderPreNormAndResidual)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(818);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> wq =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wk =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wv =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::AttentionWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.query.insert(weights.query.end(), wq[m].begin(),
                                 wq[m].end());
            weights.key.insert(weights.key.end(), wk[m].begin(), wk[m].end());
            weights.value.insert(weights.value.end(), wv[m].begin(),
                                 wv[m].end());
        }

        const std::vector<double> x_slots = pack_blocks(x, layout);
        const llama::Llama3Operator::RMSNormConfig norm = norm_config(x_slots);
        const std::vector<std::vector<double>> normed =
            unpack_blocks(rms_norm_host(x_slots, norm), layout);

        std::vector<std::vector<double>> q(layout.batch);
        std::vector<std::vector<double>> k(layout.batch);
        std::vector<std::vector<double>> v(layout.batch);
        std::vector<std::vector<double>> raw(layout.batch);
        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (int m = 0; m < layout.batch; m++)
        {
            q[m] = llama::matmul_host(wq[m], normed[m], d);
            k[m] = llama::matmul_host(wk[m], normed[m], d);
            v[m] = llama::matmul_host(wv[m], normed[m], d);
            raw[m] = llama::matmul_host(llama::transpose_host(k[m], d), q[m],
                                        d);
            for (double s : raw[m])
            {
                lowest = std::min(lowest, s);
                highest = std::max(highest, s);
            }
        }

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.causal = true;
        config.rope = false;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 1;

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            std::vector<double> probabilities(
                static_cast<std::size_t>(d) * d, 0.0);
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int key = 0; key <= query; key++)
                {
                    total += std::exp(raw[m][key * d + query] *
                                          config.head_scale -
                                      config.score_shift);
                }
                for (int key = 0; key <= query; key++)
                {
                    probabilities[key * d + query] =
                        std::exp(raw[m][key * d + query] * config.head_scale -
                                 config.score_shift) /
                        total;
                }
            }
            const std::vector<double> out =
                llama::matmul_host(v[m], probabilities, d);
            want[m].resize(out.size());
            for (std::size_t e = 0; e < out.size(); e++)
            {
                want[m][e] = x[m][e] + out[e];
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(x_slots);
        std::vector<heongpu::Ciphertext<S>> in{cipher};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normalised =
            ops->rms_norm(in, no_weights, norm, *galois, *relin);

        std::vector<heongpu::Plaintext<S>> no_rope;
        heongpu::Ciphertext<S> sublayer = ops->attention(
            normalised[0], weights, no_rope, config, *galois, *relin);
        heongpu::Ciphertext<S> result = ops->residual_add(cipher, sublayer);

        EXPECT_LT(reported("pre-norm attention block", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-4);
    }

} // namespace
