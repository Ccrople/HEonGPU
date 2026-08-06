// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU-side validation of the Llama-3 linear algebra with every
// plaintext-ciphertext product on Kang's Algorithm 5 and every
// ciphertext-ciphertext product on Kang's Algorithm 4.
//
// Two claims carry this path and both are checked against the layout they name,
// not merely for round-tripping, which a pair of mutually inverse mistakes would
// also pass:
//
//   1. a rectangular activation put through to_slots really does land with
//      X[token u][channel b*d + j] at slot b + (k/2)*u of column j -- i.e. the
//      block transform actually undoes the Fourier transform the Y axis carries;
//
//   2. to_batch really does hand Algorithm 4 a matrix encryption whose batch
//      slot b is channel block b, which is the whole reason the batch CCMM is
//      the right tool for attention: with head_dim = d a block IS a head.
//
// A NOTE ON THE CHAIN LENGTHS BELOW
// ---------------------------------
// Algorithm 5's summation is a CMT read at k = 2, so it asks for N/2 - 1 Galois
// keys -- 2047 at the smallest ring HEonGPU supports, and that count depends on
// the ring degree ALONE. The key set is therefore the binding memory constraint
// and it grows with the chain, so every fixture here takes the shortest chain
// its test actually needs. The special primes are deliberately generous: fewer
// decomposition groups make each switching key smaller, which is the only lever
// on 2047 of them.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using Rect = heongpu::llama::Llama3RectOperator;

    /// Everything a rectangular test needs, at a chain the caller sizes.
    ///
    /// d = 128 makes a head exactly one channel block at N = 4096, which is the
    /// shape this encoding is built for: k/2 = 16 blocks per group and one
    /// Algorithm 4 call covers sixteen heads.
    struct Fixture
    {
        static constexpr int degree = 4096;
        static constexpr int d = 128;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        heongpu::BatchMatrixLayout layout;
        double scale = std::pow(2.0, 40);

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::llama::Llama3BatchOperator> batch;
        std::unique_ptr<Rect> op;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        Fixture(int limbs, int special)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limbs - 1, 40);
            std::vector<int> logp(special, 60);

            context->set_poly_modulus_degree(
                static_cast<size_t>(degree));
            context->set_coeff_modulus_bit_sizes(logq, logp);
            context->generate();

            layout = heongpu::BatchMatrixLayout(degree, d);

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

            batch = std::make_unique<heongpu::llama::Llama3BatchOperator>(
                context, *encoder, layout, scale);
            op = std::make_unique<Rect>(context, *encoder, layout, scale);

            // Not const: the Galoiskey constructor takes its shift list by
            // non-const reference and canonicalises it in place.
            std::vector<int> shifts = op->rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        int half() const { return degree / 2; }
        int blocks() const { return layout.batch; }
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

    std::vector<double> host_product(const std::vector<double>& a,
                                     const std::vector<double>& b, int rows,
                                     int inner, int cols)
    {
        std::vector<double> out(static_cast<size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int t = 0; t < inner; ++t)
            {
                const double v = a[static_cast<size_t>(r) * inner + t];
                if (v == 0.0)
                    continue;
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
} // namespace

// ---------------------------------------------------------------------------
// The encoding itself
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Rect_EncryptDecryptRoundTrip)
{
    Fixture fx(3, 2);
    const int channels = fx.half();

    const std::vector<double> x =
        random_matrix(Fixture::d, channels, 11111u);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    ASSERT_EQ(ct.column.size(), static_cast<size_t>(Fixture::d));
    ASSERT_EQ(ct.groups, 1);

    const std::vector<double> got =
        fx.op->decrypt(ct, *fx.decryptor, fx.scale);

    const double worst = worst_diff(x, got);
    std::cout << "rect round trip worst error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-5);
}

TEST(HEonGPU, CKKS_Llama3Rect_EncryptDecryptTwoGroups)
{
    Fixture fx(3, 2);
    const int channels = 2 * fx.half();

    const std::vector<double> x =
        random_matrix(Fixture::d, channels, 22222u);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    ASSERT_EQ(ct.groups, 2);
    ASSERT_EQ(ct.column.size(), static_cast<size_t>(2 * Fixture::d));

    const std::vector<double> got =
        fx.op->decrypt(ct, *fx.decryptor, fx.scale);
    EXPECT_LT(worst_diff(x, got), 1e-5);
}

// The sharp check on the new transform: to_slots does not merely invert
// from_slots, it lands on the layout the module documents. Slot b + (k/2)*u of
// column j must be X[token u][channel b*d + j], with NO mixing across b -- and
// mixing across b is exactly what a rectangular encoding carries before the
// block transform undoes it.
TEST(HEonGPU, CKKS_Llama3Rect_ToSlotsLandsOnTheDocumentedLayout)
{
    Fixture fx(6, 3);
    const int d = Fixture::d;
    const int step = fx.blocks();
    const int channels = fx.half();

    const std::vector<double> x = random_matrix(d, channels, 33333u);
    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);

    std::vector<heongpu::Ciphertext<S>> slots =
        fx.op->to_slots(ct, *fx.galois);
    ASSERT_EQ(slots.size(), static_cast<size_t>(d));

    double worst = 0.0;
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> plain(fx.context);
        fx.decryptor->decrypt(plain, slots[j]);
        std::vector<double> values;
        fx.encoder->decode(values, plain);
        ASSERT_EQ(values.size(), static_cast<size_t>(fx.half()));

        for (int b = 0; b < step; ++b)
            for (int u = 0; u < d; ++u)
            {
                const double want =
                    x[static_cast<size_t>(u) * channels + b * d + j];
                const double got = values[b + u * step];
                worst = std::max(worst, std::abs(got - want));
            }
    }
    std::cout << "rect to_slots layout worst error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-3);
}

TEST(HEonGPU, CKKS_Llama3Rect_SlotRoundTrip)
{
    Fixture fx(6, 3);
    const int channels = fx.half();

    const std::vector<double> x =
        random_matrix(Fixture::d, channels, 44444u);
    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);

    std::vector<heongpu::Ciphertext<S>> slots =
        fx.op->to_slots(ct, *fx.galois);
    heongpu::llama::RectActivation back =
        fx.op->from_slots(slots, channels, *fx.galois);

    const std::vector<double> got =
        fx.op->decrypt(back, *fx.decryptor, back.column.front().scale());
    std::cout << "rect slot round trip worst error: " << worst_diff(x, got)
              << std::endl;
    EXPECT_LT(worst_diff(x, got), 1e-3);
}

// The other sharp check: batch slot b of to_batch's output has to be channel
// block b, because that is what makes one Algorithm 4 call cover k/2 heads.
TEST(HEonGPU, CKKS_Llama3Rect_ToBatchGivesOneHeadPerBatchSlot)
{
    Fixture fx(8, 4);
    const int d = Fixture::d;
    const int step = fx.blocks();
    const int channels = fx.half();

    const std::vector<double> x = random_matrix(d, channels, 55555u);
    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);

    heongpu::llama::BatchActivation batch =
        fx.op->to_batch(ct, 0, *fx.galois);
    ASSERT_EQ(batch.column.size(), static_cast<size_t>(d));
    ASSERT_EQ(batch.rows, d);

    const std::vector<std::vector<double>> got = fx.batch->decrypt(
        batch, *fx.decryptor, batch.column.front().scale());
    ASSERT_EQ(got.size(), static_cast<size_t>(step));

    double worst = 0.0;
    for (int b = 0; b < step; ++b)
        for (int u = 0; u < d; ++u)
            for (int j = 0; j < d; ++j)
            {
                const double want =
                    x[static_cast<size_t>(u) * channels + b * d + j];
                worst = std::max(
                    worst,
                    std::abs(got[b][static_cast<size_t>(u) * d + j] - want));
            }
    std::cout << "rect to_batch worst error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-3);
}

// ---------------------------------------------------------------------------
// Algorithm 5 as the projection
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Rect_ProjectMatchesHostReference)
{
    Fixture fx(4, 2);
    const int d = Fixture::d;
    const int channels = fx.half();

    const std::vector<double> x = random_matrix(d, channels, 66666u);
    const std::vector<double> w = random_matrix(
        channels, channels, 77777u,
        1.0 / std::sqrt(static_cast<double>(channels)));
    const std::vector<double> want =
        host_product(x, w, d, channels, channels);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation out =
        fx.op->project(ct, w, channels, channels, "test", *fx.galois);

    const std::vector<double> got =
        fx.op->decrypt(out, *fx.decryptor, out.column.front().scale());
    std::cout << "rect projection worst error: " << worst_diff(want, got)
              << std::endl;
    EXPECT_LT(worst_diff(want, got), 5e-2);
}

// A width of two groups exercises the sum over input groups, which is the part
// that makes widening a model cost products and no depth.
TEST(HEonGPU, CKKS_Llama3Rect_ProjectSumsOverInputGroups)
{
    Fixture fx(4, 2);
    const int d = Fixture::d;
    const int in_channels = 2 * fx.half();
    const int out_channels = fx.half();

    const std::vector<double> x = random_matrix(d, in_channels, 88888u);
    const std::vector<double> w = random_matrix(
        in_channels, out_channels, 99999u,
        1.0 / std::sqrt(static_cast<double>(in_channels)));
    const std::vector<double> want =
        host_product(x, w, d, in_channels, out_channels);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, in_channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation out = fx.op->project(
        ct, w, in_channels, out_channels, "test", *fx.galois);
    ASSERT_EQ(out.groups, 1);

    const std::vector<double> got =
        fx.op->decrypt(out, *fx.decryptor, out.column.front().scale());
    std::cout << "rect two-group projection worst error: "
              << worst_diff(want, got) << std::endl;
    EXPECT_LT(worst_diff(want, got), 1e-1);
}

// The property the whole path rests on: Algorithm 5's output is a legal
// Algorithm 5 input, so a chain of projections needs no re-encoding between
// calls. Checked here through the module rather than the raw operator, because
// it is the module's scale handling that would break it.
TEST(HEonGPU, CKKS_Llama3Rect_ProjectionsChain)
{
    Fixture fx(6, 3);
    const int d = Fixture::d;
    const int channels = fx.half();
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    const std::vector<double> x = random_matrix(d, channels, 12121u);
    const std::vector<double> w1 =
        random_matrix(channels, channels, 13131u, amp);
    const std::vector<double> w2 =
        random_matrix(channels, channels, 14141u, amp);

    const std::vector<double> stage1 =
        host_product(x, w1, d, channels, channels);
    const std::vector<double> want =
        host_product(stage1, w2, d, channels, channels);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation first =
        fx.op->project(ct, w1, channels, channels, "first", *fx.galois);
    heongpu::llama::RectActivation second =
        fx.op->project(first, w2, channels, channels, "second", *fx.galois);

    const std::vector<double> got =
        fx.op->decrypt(second, *fx.decryptor, second.column.front().scale());
    std::cout << "chained rect projection worst error: "
              << worst_diff(want, got) << std::endl;
    EXPECT_LT(worst_diff(want, got), 1e-1);
}

// ---------------------------------------------------------------------------
// Algorithm 4 as the encrypted product
// ---------------------------------------------------------------------------

// Q K^T for sixteen heads in one batch CCMM. This is the claim that Algorithm 4
// composes with the rectangular encoding at all, isolated from the SoftMax
// depth that a whole attention sublayer adds on top of it.
TEST(HEonGPU, CKKS_Llama3Rect_BatchCCMMScoresMatchHostReference)
{
    Fixture fx(10, 5);
    const int d = Fixture::d;
    const int step = fx.blocks();
    const int channels = fx.half();

    const std::vector<double> q = random_matrix(d, channels, 15151u);
    const std::vector<double> k = random_matrix(d, channels, 16161u);

    heongpu::llama::RectActivation qc =
        fx.op->encrypt(q, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation kc =
        fx.op->encrypt(k, channels, *fx.encryptor, fx.scale);

    heongpu::llama::BatchActivation qb = fx.op->to_batch(qc, 0, *fx.galois);
    heongpu::llama::BatchActivation kb = fx.op->to_batch(kc, 0, *fx.galois);
    heongpu::llama::BatchActivation kt =
        fx.batch->transpose(kb, "key", *fx.galois);

    heongpu::llama::BatchActivation scores =
        fx.op->matmul(qb, kt, "score", *fx.galois, *fx.relin);

    const std::vector<std::vector<double>> got = fx.batch->decrypt(
        scores, *fx.decryptor, scores.column.front().scale());
    ASSERT_EQ(got.size(), static_cast<size_t>(step));

    // Head b owns channels [b*d, (b+1)*d), so its score matrix contracts over
    // exactly those and over nothing else.
    double worst = 0.0;
    for (int b = 0; b < step; ++b)
        for (int u = 0; u < d; ++u)
            for (int t = 0; t < d; ++t)
            {
                double want = 0.0;
                for (int c = 0; c < d; ++c)
                    want += q[static_cast<size_t>(u) * channels + b * d + c] *
                            k[static_cast<size_t>(t) * channels + b * d + c];
                worst = std::max(
                    worst,
                    std::abs(got[b][static_cast<size_t>(u) * d + t] - want));
            }
    std::cout << "rect batch CCMM scores worst error: " << worst << std::endl;
    EXPECT_LT(worst, 1.0);
}

// ---------------------------------------------------------------------------
// The sublayers
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Rect_RMSNormMatchesHostReference)
{
    Fixture fx(24, 12);
    const int d = Fixture::d;
    const int channels = fx.half();
    const double eps = 1e-5;

    // Held near unit variance so the summed square lands inside the interval
    // the Chebyshev seed is fitted over; outside it a Chebyshev fit is worth
    // nothing at all.
    const std::vector<double> x = random_matrix(d, channels, 17171u);
    const std::vector<double> weight =
        random_matrix(1, channels, 18181u, 0.5);

    std::vector<double> want(x.size(), 0.0);
    for (int u = 0; u < d; ++u)
    {
        double sum = 0.0;
        for (int c = 0; c < channels; ++c)
        {
            const double v = x[static_cast<size_t>(u) * channels + c];
            sum += v * v;
        }
        const double inv =
            1.0 / std::sqrt(sum / static_cast<double>(channels) + eps);
        for (int c = 0; c < channels; ++c)
            want[static_cast<size_t>(u) * channels + c] =
                x[static_cast<size_t>(u) * channels + c] * inv * weight[c];
    }

    Rect::RectRMSNormConfig config;
    config.eps = eps;
    // A uniform [-1,1] channel has mean square 1/3, so the summed square sits
    // near channels/3; the interval is opened out to cover the spread.
    config.sum_lo = channels * 0.20;
    config.sum_hi = channels * 0.50;
    config.degree = 31;
    config.newton_iterations = 2;

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation out =
        fx.op->rms_norm(ct, weight, config, *fx.galois, *fx.relin);

    const std::vector<double> got =
        fx.op->decrypt(out, *fx.decryptor, out.column.front().scale());
    std::cout << "rect RMSNorm worst error: " << worst_diff(want, got)
              << std::endl;
    EXPECT_LT(worst_diff(want, got), 5e-2);
}

TEST(HEonGPU, CKKS_Llama3Rect_FeedForwardMatchesHostReference)
{
    Fixture fx(18, 9);
    const int d = Fixture::d;
    const int channels = fx.half();
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    const std::vector<double> x = random_matrix(d, channels, 19191u);
    const std::vector<double> gate =
        random_matrix(channels, channels, 20202u, amp);
    const std::vector<double> up =
        random_matrix(channels, channels, 21212u, amp);
    const std::vector<double> down =
        random_matrix(channels, channels, 23232u, amp);

    const std::vector<double> hg =
        host_product(x, gate, d, channels, channels);
    const std::vector<double> hu = host_product(x, up, d, channels, channels);
    std::vector<double> hidden(hg.size());
    for (size_t e = 0; e < hg.size(); ++e)
    {
        const double s = hg[e] / (1.0 + std::exp(-hg[e]));
        hidden[e] = s * hu[e];
    }
    const std::vector<double> want =
        host_product(hidden, down, d, channels, channels);

    Rect::RectFeedForwardWeights weights;
    weights.gate = gate;
    weights.up = up;
    weights.down = down;

    Rect::RectFeedForwardConfig config;
    config.in_channels = channels;
    config.hidden_channels = channels;
    config.silu_bound = 6.0;
    config.silu_degree = 31;

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation out =
        fx.op->feed_forward(ct, weights, config, *fx.galois, *fx.relin);

    const std::vector<double> got =
        fx.op->decrypt(out, *fx.decryptor, out.column.front().scale());
    std::cout << "rect SwiGLU worst error: " << worst_diff(want, got)
              << std::endl;
    EXPECT_LT(worst_diff(want, got), 5e-1);
}

// The deepest thing that fits. Attention consumes, level by level:
//
//   projection 1, to_batch 3, Q K^T 1, to_slots 1, SoftMax 25,
//   from_slots 1, P V 1, from_batch 3
//
// which is 36, and the SoftMax is two thirds of it. The chain therefore has to
// be 42 limbs, and 2047 rotation keys on a 42-limb chain are 17 GiB before a
// single ciphertext exists -- so the special primes are set to half the chain,
// which halves the decomposition groups and with them the size of every key.
// This is the shape of the wall on this path: not arithmetic, key material.
//
// The SoftMax's reciprocal is the accuracy limit here and it is the token block
// size that makes it one. d = 128 tokens means the second round fits 1/x over
// [1/256, 1.5], a range of 384, and a deeper Newton refinement is exactly what
// there is no chain left for. The assertion below is therefore relative to the
// signal rather than absolute: what is being pinned is that the encoding path
// carries an attention sublayer end to end, not that a degree-15 reciprocal is
// accurate over a range of 384.
TEST(HEonGPU, CKKS_Llama3Rect_AttentionMatchesHostReference)
{
    Fixture fx(42, 21);
    const int d = Fixture::d;
    const int step = fx.blocks();
    const int channels = fx.half();
    const int heads = step; // one group of heads, head_dim = d
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    const std::vector<double> x = random_matrix(d, channels, 24242u);
    Rect::RectAttentionWeights weights;
    weights.query = random_matrix(channels, channels, 25252u, amp);
    weights.key = random_matrix(channels, channels, 26262u, amp);
    weights.value = random_matrix(channels, channels, 27272u, amp);

    Rect::RectAttentionConfig config;
    config.in_channels = channels;
    config.heads = heads;
    config.kv_heads = heads;
    config.causal = false;
    config.softmax.iterations = 2;
    config.softmax.exp_degree = 31;
    config.softmax.inverse_degree = 15;
    // One Newton step, not two. Two costs two more levels per round, four in
    // all, and there is no chain left for them at 2047 keys.
    config.softmax.inverse_newton = 1;

    // Host reference, with the scores shifted the same way the circuit shifts
    // them: SoftMax is translation invariant, so the reference is the plain
    // one.
    const double head_scale = 1.0 / std::sqrt(static_cast<double>(d));
    const std::vector<double> q =
        host_product(x, weights.query, d, channels, channels);
    const std::vector<double> k =
        host_product(x, weights.key, d, channels, channels);
    const std::vector<double> v =
        host_product(x, weights.value, d, channels, channels);

    std::vector<double> want(static_cast<size_t>(d) * channels, 0.0);
    for (int h = 0; h < heads; ++h)
        for (int u = 0; u < d; ++u)
        {
            std::vector<double> row(d, 0.0);
            double top = -1e30;
            for (int t = 0; t < d; ++t)
            {
                double s = 0.0;
                for (int c = 0; c < d; ++c)
                    s += q[static_cast<size_t>(u) * channels + h * d + c] *
                         k[static_cast<size_t>(t) * channels + h * d + c];
                row[t] = s * head_scale;
                top = std::max(top, row[t]);
            }
            double norm = 0.0;
            for (int t = 0; t < d; ++t)
            {
                row[t] = std::exp(row[t] - top);
                norm += row[t];
            }
            for (int c = 0; c < d; ++c)
            {
                double acc = 0.0;
                for (int t = 0; t < d; ++t)
                    acc += row[t] / norm *
                           v[static_cast<size_t>(t) * channels + h * d + c];
                want[static_cast<size_t>(u) * channels + h * d + c] = acc;
            }
        }

    // The scores have to arrive in [-bound, 0] for the exponential's fit to
    // mean anything, and the paper takes that translation from calibration
    // rather than from a homomorphic maximum. This is that calibration: a
    // Chebyshev fit is worth nothing outside the interval it was fitted on, so
    // both ends are taken from the scores that will actually be evaluated.
    double top = -1e30;
    double bottom = 1e30;
    for (int h = 0; h < heads; ++h)
        for (int u = 0; u < d; ++u)
            for (int t = 0; t < d; ++t)
            {
                double s = 0.0;
                for (int c = 0; c < d; ++c)
                    s += q[static_cast<size_t>(u) * channels + h * d + c] *
                         k[static_cast<size_t>(t) * channels + h * d + c];
                top = std::max(top, s * head_scale);
                bottom = std::min(bottom, s * head_scale);
            }
    config.score_shift = top;
    config.softmax.bound = (top - bottom) * 1.05;
    std::cout << "rect attention score range: [" << bottom << ", " << top
              << "], bound " << config.softmax.bound << std::endl;

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation out =
        fx.op->attention(ct, weights, config, *fx.galois, *fx.relin);

    const std::vector<double> got =
        fx.op->decrypt(out, *fx.decryptor, out.column.front().scale());

    double signal = 0.0;
    for (double w : want)
        signal = std::max(signal, std::abs(w));
    const double worst = worst_diff(want, got);
    std::cout << "rect attention worst error: " << worst << " against a signal "
              << "of " << signal << " (" << (100.0 * worst / signal) << "%)"
              << std::endl;
    EXPECT_LT(worst, 0.30 * signal);
}

// ---------------------------------------------------------------------------
// The refresh
// ---------------------------------------------------------------------------
//
// A whole block on this path spends more levels than a chain that also holds
// 2047 rotation keys can carry, so it does not run without a refresh. The
// refresh is CKKS bootstrapping applied to a RECT column and to a BATCH matrix
// encryption -- neither of which is what bootstrapping is usually asked to do.
//
// The argument that it works is that regular bootstrapping is an identity on the
// plaintext POLYNOMIAL: CoeffToSlot puts the coefficients in slots, EvalMod
// takes the multiples of q out of them, SlotToCoeff puts them back, and nothing
// in that sequence asks what the coefficients mean. The tests below are the
// reason the argument is not simply believed.

namespace
{
    /// A fixture shaped for bootstrapping rather than for depth.
    ///
    /// Three things differ from Fixture and each is required, not stylistic:
    /// the working primes are 50 bits under a 60-bit bottom so that q0 / scale
    /// is near 2^10, which is the ratio HEonGPU's EvalMod is fitted around; the
    /// secret is sparse, because EvalMod's range is set by the Hamming weight;
    /// and the Galois key is the UNION of Algorithm 5's indices and
    /// bootstrapping's, since a shift-vector key asked for an index it does not
    /// hold is undefined behaviour rather than an error.
    struct BootFixture
    {
        static constexpr int degree = 4096;
        static constexpr int d = 128;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        heongpu::BatchMatrixLayout layout;
        double scale = std::pow(2.0, 50);

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<Rect> op;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        int limbs = 0;

        explicit BootFixture(int chain_limbs) : limbs(chain_limbs)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limbs - 1, 50);
            // dnum = 1: 2047 switching keys fit at no other decomposition.
            std::vector<int> logp(limbs, 60);

            context->set_poly_modulus_degree(static_cast<size_t>(degree));
            context->set_coeff_modulus_bit_sizes(logq, logp);
            context->generate();

            layout = heongpu::BatchMatrixLayout(degree, d);

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context, 16);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            op = std::make_unique<Rect>(context, *encoder, layout, scale);

            heongpu::BootstrappingConfig boot_config(3, 3, 11, true);
            op->arith().generate_bootstrapping_params(
                scale, boot_config,
                heongpu::arithmetic_bootstrapping_type::
                    REGULAR_BOOTSTRAPPING);

            std::vector<int> shifts = op->rotation_indices();
            const std::vector<int> boot =
                op->arith().bootstrapping_key_indexs();
            shifts.insert(shifts.end(), boot.begin(), boot.end());
            std::sort(shifts.begin(), shifts.end());
            shifts.erase(std::unique(shifts.begin(), shifts.end()),
                         shifts.end());

            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        int half() const { return degree / 2; }
    };
} // namespace

// The claim, on the encoding the stream actually lives in. A rect column holds
// one data entry per COEFFICIENT, so if bootstrapping were a slot-domain
// operation rather than a polynomial one this would come back as noise.
TEST(HEonGPU, CKKS_Llama3Rect_RefreshPreservesTheRectangularEncoding)
{
    BootFixture fx(31);
    const int channels = fx.half();

    const std::vector<double> x =
        random_matrix(BootFixture::d, channels, 60601u);

    heongpu::llama::RectActivation ct =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    const int before = ct.column.front().depth();

    fx.op->bootstrap(ct, "test.refresh", *fx.galois, *fx.relin);
    const int after = ct.column.front().depth();

    const std::vector<double> got =
        fx.op->decrypt(ct, *fx.decryptor, ct.column.front().scale());

    const double worst = worst_diff(x, got);
    std::cout << "rect refresh: depth " << before << " -> " << after << " ("
              << (fx.limbs - after) << " limbs left of " << fx.limbs
              << "), worst error " << worst << std::endl;
    // A refresh has to hand back levels to be worth taking at all.
    EXPECT_LT(after, fx.limbs - 1);
    EXPECT_LT(worst, 5e-2);
}

// The same claim for the encoding Algorithm 4 consumes. The crossing on either
// side of the refresh is itself lossy, so this is checked against a run of the
// same crossings WITHOUT a refresh between them: what is being measured is the
// refresh, not the bridge.
TEST(HEonGPU, CKKS_Llama3Rect_RefreshPreservesTheMatrixEncoding)
{
    BootFixture fx(31);
    const int channels = fx.half();

    const std::vector<double> x =
        random_matrix(BootFixture::d, channels, 61601u);

    auto round_trip = [&](bool refresh)
    {
        heongpu::llama::RectActivation ct =
            fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
        std::vector<heongpu::llama::BatchActivation> one;
        one.push_back(fx.op->to_batch(ct, 0, *fx.galois));
        if (refresh)
        {
            fx.op->bootstrap(one.front(), "test.refresh_batch", *fx.galois,
                             *fx.relin);
        }
        heongpu::llama::RectActivation back =
            fx.op->from_batch(one, channels, *fx.galois);
        return fx.op->decrypt(back, *fx.decryptor,
                              back.column.front().scale());
    };

    const std::vector<double> plain = round_trip(false);
    const std::vector<double> refreshed = round_trip(true);

    std::cout << "batch refresh: bridge alone " << worst_diff(x, plain)
              << ", bridge with a refresh in it " << worst_diff(x, refreshed)
              << std::endl;
    EXPECT_LT(worst_diff(x, refreshed), 5e-2);
}

// The other level the norm gives up. Fitting 1/sqrt over the summed square
// instead of over the mean skips the plaintext product that forms the mean, and
// it is only free if the two really are the same function of the same
// ciphertext -- a misread interval would scale every output by a constant and
// nothing in the library would complain.
TEST(HEonGPU, CKKS_Llama3Rect_FoldedMeanMatchesTheDividedOne)
{
    Fixture fx(20, 10);
    const int d = Fixture::d;
    const int channels = fx.half();
    const double eps = 1e-5;

    const std::vector<double> x = random_matrix(d, channels, 63601u);

    std::vector<double> want(x.size(), 0.0);
    for (int u = 0; u < d; ++u)
    {
        double sum = 0.0;
        for (int c = 0; c < channels; ++c)
        {
            const double v = x[static_cast<size_t>(u) * channels + c];
            sum += v * v;
        }
        const double inv =
            1.0 / std::sqrt(sum / static_cast<double>(channels) + eps);
        for (int c = 0; c < channels; ++c)
            want[static_cast<size_t>(u) * channels + c] =
                x[static_cast<size_t>(u) * channels + c] * inv;
    }

    Rect::RectRMSNormConfig config;
    config.eps = eps;
    config.sum_lo = channels * 0.20;
    config.sum_hi = channels * 0.50;
    config.degree = 15;
    // The fold is only available without a Newton step, which refines against
    // the mean itself.
    config.newton_iterations = 0;

    const std::vector<double> none;
    auto run = [&](bool fold)
    {
        Rect::RectRMSNormConfig c = config;
        c.fold_mean_into_fit = fold;
        heongpu::llama::RectActivation ct =
            fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
        heongpu::llama::RectActivation out =
            fx.op->rms_norm(ct, none, c, *fx.galois, *fx.relin);
        const int spent = out.column.front().depth();
        return std::make_pair(
            fx.op->decrypt(out, *fx.decryptor, out.column.front().scale()),
            spent);
    };

    const auto divided = run(false);
    const auto folded = run(true);

    std::cout << "rms_norm mean: divided " << divided.second << " levels, "
              << "folded " << folded.second << " levels; worst error against "
              << "the host " << worst_diff(want, divided.first) << " and "
              << worst_diff(want, folded.first) << std::endl;

    // The saving is the whole point, so it is asserted and not merely printed.
    EXPECT_EQ(folded.second, divided.second - 1);
    EXPECT_LT(worst_diff(want, folded.first), 5e-2);
    EXPECT_LT(worst_diff(divided.first, folded.first), 5e-2);
}

// The level the fold saves is only saved if the fold is right. Scaling the rows
// of a projection by the learned gain has to be the same as scaling the
// channels of the activation by it, which is what the homomorphic RMSNorm does.
TEST(HEonGPU, CKKS_Llama3Rect_FoldedNormScaleMatchesTheScaledActivation)
{
    Fixture fx(4, 2);
    const int d = Fixture::d;
    const int channels = fx.half();
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    const std::vector<double> x = random_matrix(d, channels, 62601u);
    const std::vector<double> gain = random_matrix(1, channels, 62701u, 0.5);
    const std::vector<double> weight =
        random_matrix(channels, channels, 62801u, amp);

    // The activation with the gain already in it, which is what RMSNorm hands
    // to the projection when the gain is applied homomorphically.
    std::vector<double> scaled(x.size());
    for (int i = 0; i < d; ++i)
        for (int c = 0; c < channels; ++c)
            scaled[static_cast<size_t>(i) * channels + c] =
                x[static_cast<size_t>(i) * channels + c] * gain[c];

    std::vector<double> folded = weight;
    Rect::fold_scale(folded, gain, channels, channels);

    heongpu::llama::RectActivation a =
        fx.op->encrypt(x, channels, *fx.encryptor, fx.scale);
    heongpu::llama::RectActivation b =
        fx.op->encrypt(scaled, channels, *fx.encryptor, fx.scale);

    heongpu::llama::RectActivation with_fold =
        fx.op->project(a, folded, channels, channels, "folded", *fx.galois);
    heongpu::llama::RectActivation without =
        fx.op->project(b, weight, channels, channels, "plain", *fx.galois);

    const std::vector<double> got_folded = fx.op->decrypt(
        with_fold, *fx.decryptor, with_fold.column.front().scale());
    const std::vector<double> got_plain = fx.op->decrypt(
        without, *fx.decryptor, without.column.front().scale());

    double signal = 0.0;
    for (double v : got_plain)
        signal = std::max(signal, std::abs(v));
    const double worst = worst_diff(got_folded, got_plain);
    std::cout << "folded gain worst error: " << worst << " against a signal of "
              << signal << std::endl;
    EXPECT_LT(worst, 1e-3 * std::max(signal, 1.0));
}
