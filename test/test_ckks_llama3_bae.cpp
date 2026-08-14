// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU validation of a Llama-3 block driven by the Bae product (CRYPTO 2024,
// eprint 2024/1284) with one channel per ciphertext.
//
// WHAT THIS SUITE IS FOR
// ----------------------
// The module's whole claim is that at k = 1 the product is a linear
// combination of WHOLE ciphertexts and therefore does not care what a
// ciphertext encodes -- so the stream can stay in slot form, the non-linear
// layers need no crossing, and the block needs no Galois key.
//
// THE FIXTURE BELOW HOLDS NO GALOIS KEY. That is not an economy; it is the
// assertion. Every test here runs against a key set of exactly one
// relinearisation key, so a rotation appearing anywhere in this path would not
// produce a worse answer, it would fail to run at all.
//
// Everything is decrypted against a host reference computed from the same
// inputs. The tolerances are RELATIVE to the reference's own magnitude,
// because a Chebyshev fit's error grows with the interval it is fitted over
// and an absolute bound would mean different things at different widths.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Bae = heongpu::llama::Llama3BaeOperator;

    struct Fixture
    {
        static constexpr int degree = 4096;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        double scale = std::pow(2.0, 40);

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;
        std::unique_ptr<Bae> op;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        // Deliberately absent: heongpu::Galoiskey<S>. See the file header.

        int tokens;

        Fixture(int limbs, int special, int tokens_)
            : tokens(tokens_)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limbs - 1, 40);
            std::vector<int> logp(special, 60);

            context->set_poly_modulus_degree(static_cast<size_t>(degree));
            context->set_coeff_modulus_bit_sizes(logq, logp);
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(context,
                                                                    *encoder);
            op = std::make_unique<Bae>(context, *encoder, tokens, scale);

            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }
    };

    std::vector<double> random_matrix(int rows, int cols, uint64_t seed,
                                      double amplitude = 1.0)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-amplitude, amplitude);
        std::vector<double> out(static_cast<size_t>(rows) * cols);
        for (double& v : out)
            v = dist(rng);
        return out;
    }

    /// rows x inner times inner x cols, row major throughout.
    std::vector<double> host_product(const std::vector<double>& a,
                                     const std::vector<double>& b, int rows,
                                     int inner, int cols)
    {
        std::vector<double> out(static_cast<size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int t = 0; t < inner; ++t)
            {
                const double v = a[static_cast<size_t>(r) * inner + t];
                for (int c = 0; c < cols; ++c)
                    out[static_cast<size_t>(r) * cols + c] +=
                        v * b[static_cast<size_t>(t) * cols + c];
            }
        return out;
    }

    double worst_diff(const std::vector<double>& a,
                      const std::vector<double>& b)
    {
        double worst = 0.0;
        const size_t n = std::min(a.size(), b.size());
        for (size_t e = 0; e < n; ++e)
            worst = std::max(worst, std::abs(a[e] - b[e]));
        return worst;
    }

    double peak(const std::vector<double>& a)
    {
        double m = 0.0;
        for (double v : a)
            m = std::max(m, std::abs(v));
        return m;
    }

    /// Relative to the reference's own peak, so the verdict survives a change
    /// of width. An absolute bound does not.
    double relative_error(const std::vector<double>& got,
                          const std::vector<double>& want)
    {
        const double p = peak(want);
        return p > 0.0 ? worst_diff(got, want) / p : worst_diff(got, want);
    }

    /// The summed square per token, and the interval it actually occupies.
    /// Calibrating the fit from the data rather than from a worst case is
    /// what makes a low degree enough; see the RMSNorm note in §25.9.
    std::pair<double, double> summed_square_range(const std::vector<double>& x,
                                                  int tokens, int channels)
    {
        double lo = std::numeric_limits<double>::max();
        double hi = 0.0;
        for (int t = 0; t < tokens; ++t)
        {
            double s = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                const double v = x[static_cast<size_t>(t) * channels + c];
                s += v * v;
            }
            lo = std::min(lo, s);
            hi = std::max(hi, s);
        }
        // A margin, because the fit must cover the interval and not merely
        // touch it, and because eps sits inside the fitted function.
        return {std::max(lo * 0.5, 1e-6), hi * 1.5};
    }

    std::vector<double> host_rms_norm(const std::vector<double>& x, int tokens,
                                      int channels,
                                      const std::vector<double>& gain,
                                      double eps)
    {
        std::vector<double> out(x.size(), 0.0);
        for (int t = 0; t < tokens; ++t)
        {
            double s = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                const double v = x[static_cast<size_t>(t) * channels + c];
                s += v * v;
            }
            const double inv =
                1.0 / std::sqrt(s / static_cast<double>(channels) + eps);
            for (int c = 0; c < channels; ++c)
            {
                const double g =
                    gain.empty() ? 1.0 : gain[static_cast<size_t>(c)];
                out[static_cast<size_t>(t) * channels + c] =
                    x[static_cast<size_t>(t) * channels + c] * inv * g;
            }
        }
        return out;
    }

    double host_silu(double v) { return v / (1.0 + std::exp(-v)); }
} // namespace

// ---------------------------------------------------------------------------
// The encoding
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_EncryptDecryptRoundTrip)
{
    Fixture fx(3, 2, 32);
    const int channels = 6;
    const auto x = random_matrix(fx.tokens, channels, 11);

    auto ct = fx.op->encrypt(x, fx.tokens, channels, *fx.encryptor);
    EXPECT_EQ(static_cast<int>(ct.column.size()), channels);
    EXPECT_EQ(ct.channels, channels);
    EXPECT_EQ(ct.tokens, fx.tokens);

    const auto got = fx.op->decrypt(ct, *fx.decryptor);
    EXPECT_LT(relative_error(got, x), 1e-6);
}

TEST(HEonGPU, CKKS_Llama3Bae_MoreTokensThanSlotsIsRejected)
{
    // The token axis is the slot axis, so a longer sequence is more
    // ciphertexts per channel and not a wider one. Saying so at construction
    // is better than silently truncating.
    Fixture fx(3, 2, 32);
    EXPECT_THROW(Bae(fx.context, *fx.encoder, fx.encoder->slot_count() + 1,
                     fx.scale),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The projection, taken in slot form
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_AProjectionIsTakenInSlotFormAndMatchesTheHost)
{
    // THE CENTRAL TEST OF THE MODULE.
    //
    // The activation is produced by HEEncoder::encode -- an ordinary
    // SLOT-encoded CKKS ciphertext -- and the result is read back by
    // HEEncoder::decode. A coefficient-encoded ciphertext does not decode
    // through that path, so the fact that this passes IS the statement that
    // the k = 1 Bae product commutes with the canonical embedding.
    //
    // HEBaePcmmOperator was written entirely in the coefficient domain and
    // knows nothing about any of this.
    Fixture fx(4, 2, 64);
    const int in_channels = 6, out_channels = 5;

    const auto x = random_matrix(fx.tokens, in_channels, 21);
    const auto w = random_matrix(in_channels, out_channels, 22);

    auto ct = fx.op->encrypt(x, fx.tokens, in_channels, *fx.encryptor);
    auto out = fx.op->project(ct, w, in_channels, out_channels, "test");

    const auto got = fx.op->decrypt(out, *fx.decryptor);
    const auto want =
        host_product(x, w, fx.tokens, in_channels, out_channels);
    EXPECT_LT(relative_error(got, want), 1e-5);
}

TEST(HEonGPU, CKKS_Llama3Bae_AProjectionSpendsExactlyOneLevel)
{
    Fixture fx(4, 2, 32);
    const int in_channels = 4, out_channels = 4;
    const auto x = random_matrix(fx.tokens, in_channels, 31);
    const auto w = random_matrix(in_channels, out_channels, 32);

    auto ct = fx.op->encrypt(x, fx.tokens, in_channels, *fx.encryptor);
    const int before = ct.column.front().depth();
    auto out = fx.op->project(ct, w, in_channels, out_channels, "test");
    for (const auto& c : out.column)
        EXPECT_EQ(c.depth(), before + 1);
}

TEST(HEonGPU, CKKS_Llama3Bae_TwoProjectionsChainWithNoConversion)
{
    Fixture fx(5, 3, 32);
    const int a = 4, b = 6, c = 3;
    const auto x = random_matrix(fx.tokens, a, 41);
    const auto w1 = random_matrix(a, b, 42);
    const auto w2 = random_matrix(b, c, 43);

    auto ct = fx.op->encrypt(x, fx.tokens, a, *fx.encryptor);
    auto mid = fx.op->project(ct, w1, a, b, "first");
    auto out = fx.op->project(mid, w2, b, c, "second");

    const auto got = fx.op->decrypt(out, *fx.decryptor);
    const auto want = host_product(
        host_product(x, w1, fx.tokens, a, b), w2, fx.tokens, b, c);
    EXPECT_LT(relative_error(got, want), 1e-5);
    EXPECT_EQ(out.column.front().depth(), 2);
}

// ---------------------------------------------------------------------------
// The transform-free path
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_TheTransformFreePathAgreesWithTheTransformingOne)
{
    // Skipping the INTT/NTT round trip is only correct because the k = 1
    // product is a linear combination and the NTT is linear. If that were
    // wrong the two would disagree grossly, not marginally, so a tight bound
    // is the right assertion here.
    Fixture fx(4, 2, 64);
    const int d1 = 5, d2 = 6;

    const auto x = random_matrix(fx.tokens, d2, 51);
    const auto w = random_matrix(d2, d1, 52);

    // ONE encryption, copied. Two calls to encrypt() would draw two different
    // encryption noises and the two paths could then only be compared to the
    // noise floor -- which is what an earlier version of this test did, and
    // it reported 3.4e-09 where the truth is zero.
    auto ct = fx.op->encrypt(x, fx.tokens, d2, *fx.encryptor);
    auto plain_ct = ct;
    auto free_ct = ct;

    // U as the operator wants it: out_channels x in_channels.
    std::vector<double> U(static_cast<size_t>(d1) * d2);
    for (int i = 0; i < d2; ++i)
        for (int j = 0; j < d1; ++j)
            U[static_cast<size_t>(j) * d2 + i] =
                w[static_cast<size_t>(i) * d1 + j];

    const auto run = [&](heongpu::llama::BaeActivation& act,
                         bool transform_free)
    {
        heongpu::HEBaePcmmOperator<S> bae(fx.context, Fixture::degree);
        bae.set_transform_free(transform_free);
        std::vector<heongpu::Ciphertext<S>*> in;
        for (auto& c : act.column)
            in.push_back(&c);
        bae.upload_plaintext(U, d1, d2, 0, bae.rescale_prime(0));
        std::vector<heongpu::Ciphertext<S>> out;
        bae.pcmm(out, in, /*rescale=*/true);
        for (auto& c : out)
            fx.ops->rescale_inplace(c);

        heongpu::llama::BaeActivation act_out;
        act_out.tokens = act.tokens;
        act_out.channels = d1;
        act_out.column = std::move(out);
        return fx.op->decrypt(act_out, *fx.decryptor);
    };

    const auto staged = run(plain_ct, false);
    const auto direct = run(free_ct, true);
    const auto want = host_product(x, w, fx.tokens, d2, d1);

    EXPECT_LT(relative_error(staged, want), 1e-5);
    EXPECT_LT(relative_error(direct, want), 1e-5);
    // INTT then NTT is the identity in RNS, exactly, and everything after it
    // is integer arithmetic on identical inputs. So the two paths do not
    // merely agree to the noise floor: they agree BIT FOR BIT, and asserting
    // anything weaker would let a real divergence hide under the noise.
    EXPECT_EQ(worst_diff(direct, staged), 0.0);
}

TEST(HEonGPU, CKKS_Llama3Bae_TheTransformFreePathIsRejectedAboveKOne)
{
    // Above k = 1 the product mixes decimation phases inside a ciphertext,
    // which is a statement about coefficients. Running it on NTT-domain limbs
    // would compute a different matrix and say nothing, which is exactly the
    // failure mode §25.8 already cost this module once.
    Fixture fx(3, 2, 32);
    heongpu::HEBaePcmmOperator<S> bae(fx.context, Fixture::degree / 4);
    EXPECT_EQ(bae.k(), 4);
    EXPECT_THROW(bae.set_transform_free(true), std::invalid_argument);
    EXPECT_FALSE(bae.transform_free());
}

// ---------------------------------------------------------------------------
// RMSNorm, whose channel reduction is an addition
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_RmsNormMatchesHostReferenceWithNoRotation)
{
    // The fixture owns no Galois key. A rotation anywhere in this reduction
    // would not degrade the answer, it would fail to run -- which is the
    // difference between "a channel is a whole ciphertext" and every other
    // encoding on this branch.
    Fixture fx(14, 7, 64);
    const int channels = 8;
    const auto x = random_matrix(fx.tokens, channels, 61);

    Bae::BaeRMSNormConfig config;
    config.eps = 1e-5;
    const auto range = summed_square_range(x, fx.tokens, channels);
    config.sum_lo = range.first;
    config.sum_hi = range.second;
    config.degree = 31;
    config.newton_iterations = 0;
    config.fold_mean_into_fit = true;

    auto ct = fx.op->encrypt(x, fx.tokens, channels, *fx.encryptor);
    const std::vector<double> no_gain;
    auto out = fx.op->rms_norm(ct, no_gain, config, *fx.relin);

    const auto got = fx.op->decrypt(out, *fx.decryptor);
    const auto want =
        host_rms_norm(x, fx.tokens, channels, no_gain, config.eps);
    EXPECT_LT(relative_error(got, want), 5e-3);
}

TEST(HEonGPU, CKKS_Llama3Bae_FoldingTheGainIntoTheNextWeightIsExact)
{
    // The learned scale is a per-channel CONSTANT here, so it can move onto
    // the next projection's weight instead of costing a level. The two routes
    // must agree to the noise floor, not merely approximately -- they are the
    // same numbers multiplied in a different order.
    Fixture fx(16, 8, 64);
    const int channels = 6, out_channels = 4;
    const auto x = random_matrix(fx.tokens, channels, 71);
    const auto gain = random_matrix(1, channels, 72, 0.5);
    const auto w = random_matrix(channels, out_channels, 73);

    Bae::BaeRMSNormConfig config;
    const auto range = summed_square_range(x, fx.tokens, channels);
    config.sum_lo = range.first;
    config.sum_hi = range.second;
    config.degree = 31;
    config.newton_iterations = 0;

    // Route A: apply the gain to the stream, then project.
    auto ct_a = fx.op->encrypt(x, fx.tokens, channels, *fx.encryptor);
    auto norm_a = fx.op->rms_norm(ct_a, gain, config, *fx.relin);
    auto out_a =
        fx.op->project(norm_a, w, channels, out_channels, "after_gain");

    // Route B: fold the gain into the weight, and never touch the stream.
    auto ct_b = fx.op->encrypt(x, fx.tokens, channels, *fx.encryptor);
    const std::vector<double> no_gain;
    auto norm_b = fx.op->rms_norm(ct_b, no_gain, config, *fx.relin);
    std::vector<double> folded = w;
    Bae::fold_channel_gain(folded, gain, channels, out_channels);
    auto out_b =
        fx.op->project(norm_b, folded, channels, out_channels, "folded");

    const auto a = fx.op->decrypt(out_a, *fx.decryptor);
    const auto b = fx.op->decrypt(out_b, *fx.decryptor);
    EXPECT_LT(relative_error(b, a), 1e-4);

    // And the fold is a level cheaper, which is the reason for it.
    EXPECT_LT(out_b.column.front().depth(), out_a.column.front().depth());
}

// ---------------------------------------------------------------------------
// The feed-forward sublayer
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_FeedForwardMatchesHostReference)
{
    Fixture fx(14, 7, 64);
    const int in_channels = 6, hidden = 8;

    const auto x = random_matrix(fx.tokens, in_channels, 81, 0.5);
    Bae::BaeFeedForwardWeights weights;
    weights.gate = random_matrix(in_channels, hidden, 82, 0.5);
    weights.up = random_matrix(in_channels, hidden, 83, 0.5);
    weights.down = random_matrix(hidden, in_channels, 84, 0.5);

    const auto gate =
        host_product(x, weights.gate, fx.tokens, in_channels, hidden);
    const auto up =
        host_product(x, weights.up, fx.tokens, in_channels, hidden);
    std::vector<double> h(gate.size());
    for (size_t i = 0; i < gate.size(); ++i)
        h[i] = host_silu(gate[i]) * up[i];
    const auto want =
        host_product(h, weights.down, fx.tokens, hidden, in_channels);

    Bae::BaeFeedForwardConfig config;
    config.in_channels = in_channels;
    config.hidden = hidden;
    // Calibrated from the data, as §25.9 argues it must be: the fit's error
    // grows with the interval, and a worst-case bound here is a wrong answer
    // dressed as a conservative one.
    config.silu_bound = peak(gate) * 1.25;
    config.silu_degree = 31;

    auto ct = fx.op->encrypt(x, fx.tokens, in_channels, *fx.encryptor);
    auto out = fx.op->feed_forward(ct, weights, config, *fx.relin);

    const auto got = fx.op->decrypt(out, *fx.decryptor);
    EXPECT_LT(relative_error(got, want), 5e-3);
}

TEST(HEonGPU, CKKS_Llama3Bae_TheFeedForwardHalfOfABlockRunsWithNoGaloisKey)
{
    // The whole point, end to end: RMSNorm, three projections, a SiLU, a gate
    // product and a residual, with a key set of ONE relinearisation key.
    Fixture fx(22, 11, 64);
    const int d_model = 6, hidden = 8;

    const auto x = random_matrix(fx.tokens, d_model, 91, 0.5);

    Bae::BaeFeedForwardBlockWeights weights;
    weights.norm_gain = random_matrix(1, d_model, 92, 0.5);
    weights.feed_forward.gate = random_matrix(d_model, hidden, 93, 0.5);
    weights.feed_forward.up = random_matrix(d_model, hidden, 94, 0.5);
    weights.feed_forward.down = random_matrix(hidden, d_model, 95, 0.5);

    Bae::BaeRMSNormConfig norm_config;
    const auto range = summed_square_range(x, fx.tokens, d_model);
    norm_config.sum_lo = range.first;
    norm_config.sum_hi = range.second;
    norm_config.degree = 31;
    norm_config.newton_iterations = 0;

    // Host reference, stage by stage from the same inputs.
    const auto normalised = host_rms_norm(x, fx.tokens, d_model,
                                          weights.norm_gain, norm_config.eps);
    const auto gate = host_product(normalised, weights.feed_forward.gate,
                                   fx.tokens, d_model, hidden);
    const auto up = host_product(normalised, weights.feed_forward.up,
                                 fx.tokens, d_model, hidden);
    std::vector<double> h(gate.size());
    for (size_t i = 0; i < gate.size(); ++i)
        h[i] = host_silu(gate[i]) * up[i];
    const auto down = host_product(h, weights.feed_forward.down, fx.tokens,
                                   hidden, d_model);
    std::vector<double> want(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        want[i] = x[i] + down[i];

    Bae::BaeFeedForwardConfig ffn_config;
    ffn_config.in_channels = d_model;
    ffn_config.hidden = hidden;
    ffn_config.silu_bound = peak(gate) * 1.25;
    ffn_config.silu_degree = 31;

    auto ct = fx.op->encrypt(x, fx.tokens, d_model, *fx.encryptor);
    std::vector<std::pair<std::string, int>> trace;
    auto out = fx.op->feed_forward_block(ct, weights, norm_config, ffn_config,
                                         *fx.relin, &trace);

    const auto got = fx.op->decrypt(out, *fx.decryptor);
    EXPECT_LT(relative_error(got, want), 1e-2);

    // The level ledger, read off the run rather than argued.
    ASSERT_EQ(trace.size(), 4u);
    EXPECT_EQ(trace[0].first, "block.entry");
    EXPECT_EQ(trace[0].second, 0);
    EXPECT_EQ(trace[3].first, "block.out");
    EXPECT_GT(trace[3].second, trace[0].second);
    for (size_t i = 1; i < trace.size(); ++i)
        EXPECT_GE(trace[i].second, trace[i - 1].second);

    std::printf("[bae] feed-forward half level ledger:\n");
    for (const auto& e : trace)
        std::printf("[bae]   %-34s depth %d\n", e.first.c_str(), e.second);
}

// ---------------------------------------------------------------------------
// The counts
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Bae_TheCostModelReportsNoRotationAndNoCrossing)
{
    Fixture fx(3, 2, 32);
    const auto c = fx.op->feed_forward_block_counts(4096, 14336);

    EXPECT_EQ(c.rotations, 0);
    EXPECT_EQ(c.galois_keys, 0);
    EXPECT_EQ(c.crossings, 0);
    EXPECT_TRUE(Bae::rotation_indices().empty());

    // 2 * d1 * d2 * N per projection, three projections.
    const double n = static_cast<double>(Fixture::degree);
    const double want = 2.0 * 14336.0 * 4096.0 * n * 2.0 +
                        2.0 * 4096.0 * 14336.0 * n;
    EXPECT_NEAR(c.gemm_macs, want, want * 1e-12);
    EXPECT_EQ(c.relinearisations, 2LL * 4096 + 14336);
}

TEST(HEonGPU, CKKS_Llama3Bae_MixedLevelsAreRejected)
{
    // The product weights and sums these together, so a mismatch is a
    // silently weighted answer rather than an error from the library.
    Fixture fx(5, 3, 32);
    const int channels = 4;
    const auto x = random_matrix(fx.tokens, channels, 101);
    auto ct = fx.op->encrypt(x, fx.tokens, channels, *fx.encryptor);
    fx.ops->mod_drop_inplace(ct.column[1]);

    const auto w = random_matrix(channels, channels, 102);
    EXPECT_THROW(fx.op->project(ct, w, channels, channels, "mixed"),
                 std::invalid_argument);
}
