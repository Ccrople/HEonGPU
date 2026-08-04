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

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

        EXPECT_LT(max_error(got, want), 1e-5);
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

        EXPECT_LT(max_error(got, want), 1e-4);
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

        EXPECT_LT(max_error(got, want), 5e-3);
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

        EXPECT_LT(max_error(got, want), 1e-4);
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

        EXPECT_LT(max_error(got, want), 5e-3);
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

        EXPECT_LT(max_error(got, want), 5e-3);

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

            EXPECT_LT(max_error(got, want), 1e-4) << "ell=" << ell;
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
            EXPECT_LT(max_error(got, want), 5e-3) << "part " << part;
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

        EXPECT_LT(max_error(got, want), 5e-3);
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
        EXPECT_LT(max_error(decrypt(result), want), 1e-4);
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
            EXPECT_LT(max_error(decrypt(result), want), 1e-4)
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

        EXPECT_LT(max_error(decrypt(deep), want_deep), 1e-4);
        EXPECT_LT(max_error(decrypt(shallow), want_shallow), 1e-4);
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

        EXPECT_LT(max_error(decrypt(activated), want), 2e-2);
    }

} // namespace
