// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU-side validation of the Llama-3 linear algebra on Kang's batch matrix
// multiplication.
//
// The bridge is the only genuinely new piece of arithmetic on this path, and
// everything else rests on it, so it is checked in both directions and against
// the layout it claims to produce -- not merely for round-tripping, which a
// pair of mutually inverse mistakes would also pass.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    /// A context deep enough for a bridge, a projection, a CCMM and the
    /// rescales they spend.
    ///
    /// The scale matches the rescale primes, which is not cosmetic here: a
    /// projection is scale-preserving whatever the primes are, because the
    /// weight is encoded at the prime the following rescale divides by, but a
    /// CCMM multiplies two ciphertext scales and hands back scale^2 / prime.
    /// Only scale == prime keeps a chain of them stationary.
    struct Fixture
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 8;
        static constexpr int limbs = 6;

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
        std::unique_ptr<heongpu::llama::Llama3BatchOperator> op;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        explicit Fixture(int limb_count = limbs)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limb_count - 1, 40);
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            layout = heongpu::BatchMatrixLayout(static_cast<int>(degree), d);

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
            op = std::make_unique<heongpu::llama::Llama3BatchOperator>(
                context, *encoder, layout, scale);

            // Not const: the Galoiskey constructor takes its shift list by
            // non-const reference.
            std::vector<int> shifts = op->rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        /// k/2 random d x cols real matrices, small enough that a product of
        /// them stays well inside the prime.
        std::vector<std::vector<double>> random_batch(int rows, int cols,
                                                      uint64_t seed) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<std::vector<double>> out(
                layout.batch,
                std::vector<double>(static_cast<size_t>(rows) * cols));
            for (auto& m : out)
                for (auto& v : m)
                    v = dist(rng);
            return out;
        }
    };

    double max_abs_diff(const std::vector<std::vector<double>>& a,
                        const std::vector<std::vector<double>>& b)
    {
        double worst = 0.0;
        for (size_t s = 0; s < a.size(); ++s)
            for (size_t e = 0; e < a[s].size(); ++e)
                worst = std::max(worst, std::abs(a[s][e] - b[s][e]));
        return worst;
    }

    std::vector<double> host_product(const std::vector<double>& a,
                                     const std::vector<double>& b, int rows,
                                     int inner, int cols)
    {
        std::vector<double> out(static_cast<size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
            {
                double acc = 0.0;
                for (int t = 0; t < inner; ++t)
                    acc += a[static_cast<size_t>(r) * inner + t] *
                           b[static_cast<size_t>(t) * cols + c];
                out[static_cast<size_t>(r) * cols + c] = acc;
            }
        return out;
    }

    std::vector<double> host_transpose(const std::vector<double>& a, int rows,
                                       int cols)
    {
        std::vector<double> out(static_cast<size_t>(rows) * cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out[static_cast<size_t>(c) * rows + r] =
                    a[static_cast<size_t>(r) * cols + c];
        return out;
    }

    std::vector<double> random_weight(int rows, int cols, uint64_t seed,
                                      double amplitude = 1.0)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-amplitude, amplitude);
        std::vector<double> w(static_cast<size_t>(rows) * cols);
        for (auto& v : w)
            v = dist(rng);
        return w;
    }
} // namespace

// The encrypt/decrypt staging is the reference the rest is measured against,
// so it is checked on its own first: a fault here would look like a bridge
// fault everywhere else.
TEST(HEonGPU, CKKS_Llama3Batch_ActivationRoundTrip)
{
    Fixture f;
    const int cols = 5;
    const auto want = f.random_batch(Fixture::d, cols, 4242u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    ASSERT_EQ(ct.columns(), cols);
    ASSERT_EQ(ct.rows, Fixture::d);

    const auto got = f.op->decrypt(ct, *f.decryptor, f.scale);
    EXPECT_LT(max_abs_diff(want, got), 1e-6);
}

// The bridge, both ways. Round-tripping alone would pass a pair of mutually
// inverse mistakes, so the intermediate slot layout is checked too: slot
// b + (k/2) * u must hold entry (u, j) of matrix b, which is the property the
// strided reductions downstream depend on.
TEST(HEonGPU, CKKS_Llama3Batch_BridgeProducesTheClaimedSlotLayout)
{
    Fixture f;
    const int cols = 3;
    const auto want = f.random_batch(Fixture::d, cols, 909u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    ASSERT_EQ(slots.size(), static_cast<size_t>(cols));

    const int step = f.layout.k / 2;
    for (int j = 0; j < cols; ++j)
    {
        heongpu::Plaintext<S> plain(f.context);
        f.decryptor->decrypt(plain, slots[j]);
        std::vector<double> message;
        f.encoder->decode(message, plain);

        for (int b = 0; b < f.layout.batch; ++b)
            for (int u = 0; u < Fixture::d; ++u)
            {
                const double expect =
                    want[b][static_cast<size_t>(u) * cols + j];
                EXPECT_NEAR(message[b + u * step], expect, 1e-4)
                    << "column " << j << " batch " << b << " row " << u;
            }
    }
}

TEST(HEonGPU, CKKS_Llama3Batch_BridgeIsInvertible)
{
    Fixture f;
    const int cols = 4;
    const auto want = f.random_batch(Fixture::d, cols, 31337u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    auto back = f.op->from_slots(slots, Fixture::d, *f.galois);

    ASSERT_EQ(back.columns(), cols);
    const auto got = f.op->decrypt(back, *f.decryptor, f.scale);
    EXPECT_LT(max_abs_diff(want, got), 1e-3);
}

// The bridge costs no key material. Its shifts are the multiples of k/2, and
// so are the CMT's, because the automorphisms X -> X^(2kt+1) that Algorithm 3
// applies form the subgroup generated by 5^(k/2). Asserted rather than assumed:
// rotation_indices() takes the union either way, so a divergence here would
// cost only key size, and would go unnoticed.
TEST(HEonGPU, CKKS_Llama3Batch_BridgeReusesTheProductRotationKeys)
{
    Fixture f;

    std::vector<int> bridge = f.op->bridge_rotation_indices();
    std::vector<int> cmt = f.op->product_rotation_indices();
    std::sort(bridge.begin(), bridge.end());
    std::sort(cmt.begin(), cmt.end());

    EXPECT_EQ(bridge, cmt);
    EXPECT_EQ(f.op->rotation_indices().size(), bridge.size());
    EXPECT_EQ(bridge.size(), static_cast<size_t>(Fixture::d - 1));
}

// Kang's Algorithm 1 standing in for a projection, against a host product.
// The weight is handed over transposed, because the encrypted operand is on
// the left and a Llama projection is W X.
TEST(HEonGPU, CKKS_Llama3Batch_ProjectMatchesHostProduct)
{
    Fixture f;
    const int in_channels = Fixture::d;
    const int out_channels = 4;

    const auto x = f.random_batch(Fixture::d, in_channels, 5150u);
    const auto weight = random_weight(in_channels, out_channels, 2718u);

    auto ct = f.op->encrypt(x, Fixture::d, in_channels, *f.encryptor, f.scale);
    auto out = f.op->project(ct, weight, in_channels, out_channels, "test");
    ASSERT_EQ(out.columns(), out_channels);

    const auto got = f.op->decrypt(out, *f.decryptor, f.scale);

    std::vector<std::vector<double>> want(f.layout.batch);
    for (int b = 0; b < f.layout.batch; ++b)
        want[b] = host_product(x[b], weight, Fixture::d, in_channels,
                               out_channels);

    EXPECT_LT(max_abs_diff(want, got), 1e-3);
}

// Kang's Algorithm 3. The CMT is exercised inside every CCMM already, but only
// where a mistake would be hidden by the two that follow it; attention calls it
// on its own, for K, so it is checked on its own.
TEST(HEonGPU, CKKS_Llama3Batch_TransposeMatchesHostTranspose)
{
    Fixture f;
    const auto x = f.random_batch(Fixture::d, Fixture::d, 8675309u);

    auto ct =
        f.op->encrypt(x, Fixture::d, Fixture::d, *f.encryptor, f.scale);
    auto out = f.op->transpose(ct, "test", *f.galois);
    ASSERT_EQ(out.columns(), Fixture::d);

    const auto got = f.op->decrypt(out, *f.decryptor, f.scale);

    std::vector<std::vector<double>> want(f.layout.batch);
    for (int b = 0; b < f.layout.batch; ++b)
        want[b] = host_transpose(x[b], Fixture::d, Fixture::d);

    EXPECT_LT(max_abs_diff(want, got), 1e-3);
}

// Kang's Algorithm 4, against a host product. Both operands are real
// encryptions carrying a c1, so all four cross products and the
// relinearisation are exercised; a test built from trivial encryptions would
// not tell a transposed GEMM apart from a correct one.
//
// The result is decrypted at the scale it reports rather than at the nominal
// one. Algorithm 4 hands back scale_a * scale_b and the rescale removes one
// prime, and the prime only approximates the scale, so insisting on the
// nominal value here would be checking the fixture and not the algorithm.
TEST(HEonGPU, CKKS_Llama3Batch_MatmulMatchesHostProduct)
{
    Fixture f;
    const auto a = f.random_batch(Fixture::d, Fixture::d, 11u);
    const auto b = f.random_batch(Fixture::d, Fixture::d, 22u);

    auto ca = f.op->encrypt(a, Fixture::d, Fixture::d, *f.encryptor, f.scale);
    auto cb = f.op->encrypt(b, Fixture::d, Fixture::d, *f.encryptor, f.scale);

    auto out = f.op->matmul(ca, cb, "test", *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), Fixture::d);

    const double got_scale = out.column.front().scale();
    const auto got = f.op->decrypt(out, *f.decryptor, got_scale);

    std::vector<std::vector<double>> want(f.layout.batch);
    for (int s = 0; s < f.layout.batch; ++s)
        want[s] = host_product(a[s], b[s], Fixture::d, Fixture::d, Fixture::d);

    const double worst = max_abs_diff(want, got);
    std::cout << "batch CCMM worst absolute error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-2);
}

// Q K^T through Algorithm 4, with the transpose taken separately. This is the
// exact shape attention forms its scores in, and it is checked before the
// SoftMax is put on top of it so that a failure in the deep test below can be
// attributed.
TEST(HEonGPU, CKKS_Llama3Batch_ScoresMatchHostQKT)
{
    Fixture f;
    const auto q = f.random_batch(Fixture::d, Fixture::d, 33u);
    const auto k = f.random_batch(Fixture::d, Fixture::d, 44u);

    auto cq = f.op->encrypt(q, Fixture::d, Fixture::d, *f.encryptor, f.scale);
    auto ck = f.op->encrypt(k, Fixture::d, Fixture::d, *f.encryptor, f.scale);

    auto ckt = f.op->transpose(ck, "key", *f.galois);
    auto out = f.op->matmul(cq, ckt, "scores", *f.galois, *f.relin);

    const auto got = f.op->decrypt(out, *f.decryptor,
                                   out.column.front().scale());

    std::vector<std::vector<double>> want(f.layout.batch);
    for (int s = 0; s < f.layout.batch; ++s)
        want[s] = host_product(
            q[s], host_transpose(k[s], Fixture::d, Fixture::d), Fixture::d,
            Fixture::d, Fixture::d);

    EXPECT_LT(max_abs_diff(want, got), 1e-2);
}

// The whole attention sublayer: three projections through Algorithm 1, the
// scores and the value product through Algorithm 4, and a causal SoftMax in
// slot form behind the bridge.
//
// The reference is the true SoftMax, not the iteration the operator runs, so
// the tolerance here is that of the approximation and not of the encoding.
TEST(HEonGPU, CKKS_Llama3Batch_AttentionMatchesHostAttention)
{
    // The deepest circuit on this path, and it is the SoftMax that makes it
    // so: the linear algebra either side of it costs five levels between them.
    Fixture f(34);
    const int d = Fixture::d;
    const int channels = d;

    namespace llama = heongpu::llama;

    const auto x = f.random_batch(d, channels, 606u);
    // Small weights keep the raw scores in a range a degree-15 Chebyshev fit
    // can carry, which is what calibration does for the real model.
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));
    const auto wq = random_weight(channels, channels, 71u, amp);
    const auto wk = random_weight(channels, channels, 72u, amp);
    const auto wv = random_weight(channels, channels, 73u, amp);

    // The host reference, and the calibration taken off it, exactly as the
    // paper takes its intervals off a calibration run rather than computing
    // them homomorphically.
    std::vector<std::vector<double>> raw(f.layout.batch);
    std::vector<std::vector<double>> value(f.layout.batch);
    double highest = -1e300;
    double lowest = 1e300;
    for (int s = 0; s < f.layout.batch; ++s)
    {
        const auto q = host_product(x[s], wq, d, channels, channels);
        const auto k = host_product(x[s], wk, d, channels, channels);
        value[s] = host_product(x[s], wv, d, channels, channels);
        raw[s] = host_product(q, host_transpose(k, d, channels), d, channels,
                              d);
        for (int u = 0; u < d; ++u)
            for (int j = 0; j <= u; ++j)
            {
                const double v = raw[s][static_cast<size_t>(u) * d + j];
                highest = std::max(highest, v);
                lowest = std::min(lowest, v);
            }
    }

    llama::Llama3BatchOperator::BatchAttentionConfig config;
    config.in_channels = channels;
    config.q_channels = channels;
    config.kv_channels = channels;
    config.heads = 1;
    config.causal = true;
    config.head_scale = 2.0 / (highest - lowest);
    config.score_shift = highest * config.head_scale;
    config.softmax.bound = 2.0;
    config.softmax.iterations = 2;
    config.softmax.exp_degree = 15;
    config.softmax.inverse_degree = 15;
    config.softmax.inverse_newton = 2;

    // The true causal SoftMax of the scores the circuit is actually handed.
    // The shift cancels, but the scaling does not, so the reference has to be
    // taken on the scaled scores and not on the raw ones.
    std::vector<std::vector<double>> want(f.layout.batch);
    for (int s = 0; s < f.layout.batch; ++s)
    {
        std::vector<double> p(static_cast<size_t>(d) * d, 0.0);
        for (int u = 0; u < d; ++u)
        {
            double total = 0.0;
            for (int j = 0; j <= u; ++j)
            {
                const double e = std::exp(
                    raw[s][static_cast<size_t>(u) * d + j] * config.head_scale -
                    config.score_shift);
                p[static_cast<size_t>(u) * d + j] = e;
                total += e;
            }
            for (int j = 0; j <= u; ++j)
                p[static_cast<size_t>(u) * d + j] /= total;
        }
        want[s] = host_product(p, value[s], d, d, channels);
    }

    llama::Llama3BatchOperator::BatchAttentionWeights weights;
    weights.query = wq;
    weights.key = wk;
    weights.value = wv;

    auto ct = f.op->encrypt(x, d, channels, *f.encryptor, f.scale);
    auto out = f.op->attention(ct, weights, config, *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), channels);

    const auto got =
        f.op->decrypt(out, *f.decryptor, out.column.front().scale());

    const double worst = max_abs_diff(want, got);
    std::cout << "batch attention worst absolute error: " << worst
              << std::endl;
    EXPECT_LT(worst, 5e-2);
}

// RMSNorm behind the bridge. The channel axis runs across ciphertexts here, so
// the mean of the squares is a slot-wise addition and the reduction that costs
// the slot path log2(channels) rotations costs this path nothing; what is
// being checked is that the free reduction is also the right one.
TEST(HEonGPU, CKKS_Llama3Batch_RMSNormMatchesHostNorm)
{
    Fixture f(20);
    const int d = Fixture::d;
    const int channels = 6;
    const double eps = 1e-5;

    const auto x = f.random_batch(d, channels, 4711u);
    const auto weight = random_weight(1, channels, 4712u, 1.5);

    double sum_lo = 1e300;
    double sum_hi = 0.0;
    std::vector<std::vector<double>> want(f.layout.batch);
    for (int s = 0; s < f.layout.batch; ++s)
    {
        want[s].resize(static_cast<size_t>(d) * channels);
        for (int u = 0; u < d; ++u)
        {
            double total = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                const double v = x[s][static_cast<size_t>(u) * channels + c];
                total += v * v;
            }
            sum_lo = std::min(sum_lo, total);
            sum_hi = std::max(sum_hi, total);
            const double inv =
                1.0 / std::sqrt(total / static_cast<double>(channels) + eps);
            for (int c = 0; c < channels; ++c)
                want[s][static_cast<size_t>(u) * channels + c] =
                    x[s][static_cast<size_t>(u) * channels + c] * weight[c] *
                    inv;
        }
    }

    heongpu::llama::Llama3BatchOperator::BatchRMSNormConfig config;
    config.eps = eps;
    // Calibrated, with the margin a real run would take off a calibration set
    // rather than off the batch it is about to normalise.
    config.sum_lo = sum_lo * 0.8;
    config.sum_hi = sum_hi * 1.2;
    config.degree = 31;
    config.newton_iterations = 2;

    auto ct = f.op->encrypt(x, d, channels, *f.encryptor, f.scale);
    auto out = f.op->rms_norm(ct, weight, config, *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), channels);

    const auto got =
        f.op->decrypt(out, *f.decryptor, out.column.front().scale());

    const double worst = max_abs_diff(want, got);
    std::cout << "batch RMSNorm worst absolute error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-2);
}

// The SwiGLU sublayer. This is the one place the matrix encoding is at a
// disadvantage -- SiLU(W_g x) * (W_u x) is a Hadamard product, which a matrix
// encryption does not have -- so both branches cross to slot form and the
// result crosses back. Three bridges over the hidden width, against the three
// widest projections in the block, which is the trade this checks is sound.
TEST(HEonGPU, CKKS_Llama3Batch_FeedForwardMatchesHostSwiGLU)
{
    Fixture f(20);
    const int d = Fixture::d;
    const int channels = d;
    const int hidden = 12;

    const auto x = f.random_batch(d, channels, 1234u);
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));
    const auto wg = random_weight(channels, hidden, 91u, amp);
    const auto wu = random_weight(channels, hidden, 92u, amp);
    const auto wd = random_weight(hidden, channels, 93u, amp);

    auto silu = [](double z) { return z / (1.0 + std::exp(-z)); };

    double gate_bound = 0.0;
    std::vector<std::vector<double>> want(f.layout.batch);
    for (int s = 0; s < f.layout.batch; ++s)
    {
        const auto g = host_product(x[s], wg, d, channels, hidden);
        const auto u = host_product(x[s], wu, d, channels, hidden);
        std::vector<double> h(g.size());
        for (size_t e = 0; e < g.size(); ++e)
        {
            gate_bound = std::max(gate_bound, std::abs(g[e]));
            h[e] = silu(g[e]) * u[e];
        }
        want[s] = host_product(h, wd, d, hidden, channels);
    }

    heongpu::llama::Llama3BatchOperator::BatchFeedForwardConfig config;
    config.in_channels = channels;
    config.hidden_channels = hidden;
    // Calibrated, as Table 2 is, with a margin: a Chebyshev fit is worth
    // nothing outside the interval it was fitted on.
    config.silu_bound = gate_bound * 1.25;
    config.silu_degree = 31;

    heongpu::llama::Llama3BatchOperator::BatchFeedForwardWeights weights;
    weights.gate = wg;
    weights.up = wu;
    weights.down = wd;

    auto ct = f.op->encrypt(x, d, channels, *f.encryptor, f.scale);
    auto out = f.op->feed_forward(ct, weights, config, *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), channels);

    const auto got =
        f.op->decrypt(out, *f.decryptor, out.column.front().scale());

    const double worst = max_abs_diff(want, got);
    std::cout << "batch SwiGLU worst absolute error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-2);
}
