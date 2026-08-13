// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The batch-16 non-linear layers: RMSNorm, the reshape, and SwiGLU.
//
// Two things are being asserted here and they are not the same thing.
//
// The first is ARITHMETIC: that the layers compute what a plaintext Llama-3
// block computes. That is the ordinary test and it is the easy half.
//
// The second is SEPARATION, and it is the half that a batch-16 run can get
// silently wrong. Sixteen independent inputs share every ciphertext in this
// encoding, so an operation that mixes two of them does not fail -- it
// returns a plausible number that is a function of somebody else's prompt.
// InstancesAreIndependent perturbs ONE instance and requires every other one
// to come back bit-for-bit unchanged, which no round-trip or reference test
// would catch.
//
// The layout invariant the other three sessions build on is asserted here from
// this side as well, so that a change to the bridge breaks a test in the module
// that depends on it rather than only in the module that owns it.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;

    /// The production ring for a batch-16 run: batch = k/2 = 16 forces k = 32,
    /// and d = N/32 = 128 is both the token count and Llama-3-8B's head dim.
    /// The model widths are cut down; the RING is the real one, which is the
    /// half that decides whether the layout claims hold.
    struct Fixture
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 128;
        static constexpr int instances = 16;
        static constexpr int limbs = 14;

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
        std::unique_ptr<llama::Llama3BatchOperator> op;
        std::unique_ptr<llama::Llama3Batch16Operator> nl;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        explicit Fixture(const llama::Batch16Shape& shape,
                         int limb_count = limbs)
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
            op = std::make_unique<llama::Llama3BatchOperator>(
                context, *encoder, layout, scale);
            nl = std::make_unique<llama::Llama3Batch16Operator>(*op, shape);

            std::vector<int> shifts = op->rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        /// The 16 independent inputs, each d tokens by cols channels.
        std::vector<std::vector<double>> random_batch(int cols, uint64_t seed,
                                                      double amp = 1.0) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-amp, amp);
            std::vector<std::vector<double>> out(
                instances,
                std::vector<double>(static_cast<size_t>(d) * cols));
            for (auto& m : out)
                for (auto& v : m)
                    v = dist(rng);
            return out;
        }

        std::vector<double> decode(const heongpu::Ciphertext<S>& ct) const
        {
            heongpu::Ciphertext<S> copy = ct;
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, copy);
            std::vector<double> out;
            encoder->decode(out, plain);
            return out;
        }
    };

    /// The default 8B-shaped model, narrowed so a test runs in seconds. The
    /// head geometry is Llama-3-8B's own ratio: heads / kv_heads = 4.
    llama::Batch16Shape SmallShape()
    {
        llama::Batch16Shape shape;
        shape.instances = Fixture::instances;
        shape.d_model = 8;
        shape.hidden = 16;
        shape.heads = 4;
        shape.kv_heads = 1;
        shape.head_dim = Fixture::d; // must be a multiple of layout.d
        return shape;
    }

    double max_abs_diff(const std::vector<std::vector<double>>& a,
                        const std::vector<std::vector<double>>& b)
    {
        double worst = 0.0;
        for (size_t s = 0; s < a.size(); ++s)
            for (size_t e = 0; e < a[s].size(); ++e)
                worst = std::max(worst, std::abs(a[s][e] - b[s][e]));
        return worst;
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

    /// y[u][j] = gain[j] * x[u][j] / sqrt(mean_j x[u][j]^2 + eps), per
    /// instance and per token. The reduction is over the channel axis and
    /// NOTHING else -- that is what the separation tests below check.
    std::vector<std::vector<double>>
    rms_norm_host(const std::vector<std::vector<double>>& x, int rows,
                  int cols, const std::vector<double>& gain, double eps)
    {
        std::vector<std::vector<double>> out(x.size());
        for (size_t b = 0; b < x.size(); ++b)
        {
            out[b].assign(x[b].size(), 0.0);
            for (int u = 0; u < rows; ++u)
            {
                double acc = 0.0;
                for (int j = 0; j < cols; ++j)
                {
                    const double v = x[b][static_cast<size_t>(u) * cols + j];
                    acc += v * v;
                }
                const double r = std::sqrt(acc / cols + eps);
                for (int j = 0; j < cols; ++j)
                {
                    const size_t at = static_cast<size_t>(u) * cols + j;
                    const double g = gain.empty() ? 1.0 : gain[j];
                    out[b][at] = g * x[b][at] / r;
                }
            }
        }
        return out;
    }

    /// The interval the summed square actually visits, widened a little. A
    /// range is a measurement of the data, not a property of the algorithm --
    /// fitting over a worst-case bound instead is what made the SoftMax
    /// reciprocal wrong by 98.6% on the rectangular path.
    void bracket_sum(const std::vector<std::vector<double>>& x, int rows,
                     int cols, double& lo, double& hi)
    {
        lo = 1e300;
        hi = 0.0;
        for (const auto& m : x)
            for (int u = 0; u < rows; ++u)
            {
                double acc = 0.0;
                for (int j = 0; j < cols; ++j)
                {
                    const double v = m[static_cast<size_t>(u) * cols + j];
                    acc += v * v;
                }
                lo = std::min(lo, acc);
                hi = std::max(hi, acc);
            }
        lo *= 0.8;
        hi *= 1.2;
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

    double silu_host(double x) { return x / (1.0 + std::exp(-x)); }
} // namespace

// ----------------------------------------------------------------------
// The shape contract
// ----------------------------------------------------------------------

// Every shape error this encoding can make is silent: the activation still
// has columns, they just mean something other than what the caller thinks.
TEST(HEonGPU, CKKS_Llama3Batch16_ShapeIsCheckedAgainstTheRing)
{
    Fixture f(SmallShape(), 4);

    llama::Batch16Shape bad_instances = SmallShape();
    bad_instances.instances = 8; // the ring says 16
    EXPECT_THROW((llama::Llama3Batch16Operator(*f.op, bad_instances)),
                 std::invalid_argument);

    llama::Batch16Shape bad_head = SmallShape();
    bad_head.head_dim = Fixture::d / 2; // not a whole number of blocks
    EXPECT_THROW((llama::Llama3Batch16Operator(*f.op, bad_head)),
                 std::invalid_argument);

    llama::Batch16Shape bad_gqa = SmallShape();
    bad_gqa.heads = 4;
    bad_gqa.kv_heads = 3; // does not divide
    EXPECT_THROW((llama::Llama3Batch16Operator(*f.op, bad_gqa)),
                 std::invalid_argument);

    // The ring really does carry sixteen inputs and 128 tokens.
    EXPECT_EQ(f.nl->instances(), 16);
    EXPECT_EQ(f.nl->tokens(), 128);
    EXPECT_EQ(f.nl->slot_count(), 2048);
    EXPECT_EQ(f.nl->instances() * f.nl->tokens(), f.nl->slot_count());
}

// The invariant the SoftMax, the PCMM and the CCMM sessions all build on,
// asserted from this side too: slot b + (k/2)*u of the ciphertext for column j
// holds entry (u, j) of instance b.
TEST(HEonGPU, CKKS_Llama3Batch16_SlotMapIsThePublishedInvariant)
{
    Fixture f(SmallShape(), 4);
    const int cols = 3;
    const auto want = f.random_batch(cols, 4242u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    ASSERT_EQ(slots.size(), static_cast<size_t>(cols));

    for (int j = 0; j < cols; ++j)
    {
        const std::vector<double> message = f.decode(slots[j]);
        for (int b = 0; b < Fixture::instances; ++b)
            for (int u = 0; u < Fixture::d; ++u)
            {
                const double expect =
                    want[b][static_cast<size_t>(u) * cols + j];
                EXPECT_NEAR(message[f.nl->slot_of(b, u)], expect, 1e-4)
                    << "column " << j << " instance " << b << " token " << u;
            }
    }
}

// ----------------------------------------------------------------------
// The reshape
// ----------------------------------------------------------------------

// (token, d_model) <-> (token, head, head_dim) is not an operation here. A
// channel is a whole ciphertext, so a head is a run of ciphertext handles and
// the reshape is the arithmetic on their indices. This asserts that the
// borrowed view ALIASES the activation rather than copying it, which is the
// difference between free and d ciphertext deep copies per head.
TEST(HEonGPU, CKKS_Llama3Batch16_ReshapeIsIndexArithmeticAndAliasesTheInput)
{
    llama::Batch16Shape shape = SmallShape();
    shape.heads = 4;
    shape.kv_heads = 2;
    shape.head_dim = Fixture::d;
    shape.d_model = shape.q_channels();
    // No homomorphic arithmetic happens here, so the chain is as short as a
    // context can be: this test is about index arithmetic and aliasing.
    Fixture f(shape, 3);

    for (int h = 0; h < shape.heads; ++h)
    {
        const auto r = f.nl->query_head(h);
        EXPECT_EQ(r.base, h * shape.head_dim);
        EXPECT_EQ(r.count, shape.head_dim);
        for (int lane = 0; lane < shape.head_dim; ++lane)
        {
            EXPECT_EQ(f.nl->channel_of(h, lane), r.base + lane);
            const auto hl = f.nl->head_of(r.base + lane);
            EXPECT_EQ(hl.head, h);
            EXPECT_EQ(hl.lane, lane);
        }
    }

    // Grouped-query attention is index REUSE. Four query heads over two kv
    // heads means heads 0,1 read kv head 0 and heads 2,3 read kv head 1 --
    // and the key and value projections stay narrow by exactly that factor,
    // which is the saving the rectangular path has to give up.
    EXPECT_EQ(shape.group(), 2);
    EXPECT_EQ(f.nl->key_value_head_for_query(0).base,
              f.nl->key_value_head(0).base);
    EXPECT_EQ(f.nl->key_value_head_for_query(1).base,
              f.nl->key_value_head(0).base);
    EXPECT_EQ(f.nl->key_value_head_for_query(2).base,
              f.nl->key_value_head(1).base);
    EXPECT_EQ(f.nl->key_value_head_for_query(3).base,
              f.nl->key_value_head(1).base);
    EXPECT_EQ(shape.kv_channels(),
              shape.q_channels() / shape.group());

    const auto data = f.random_batch(shape.d_model, 77u);
    auto ct =
        f.op->encrypt(data, Fixture::d, shape.d_model, *f.encryptor, f.scale);

    for (int h = 0; h < shape.heads; ++h)
    {
        auto view = llama::Llama3Batch16Operator::columns(
            ct, f.nl->query_head(h));
        ASSERT_EQ(view.size(), static_cast<size_t>(shape.head_dim));
        for (int lane = 0; lane < shape.head_dim; ++lane)
        {
            // Aliasing, not copying: the view has to be the SAME object.
            EXPECT_EQ(view[static_cast<size_t>(lane)],
                      &ct.column[static_cast<size_t>(
                          f.nl->channel_of(h, lane))]);
        }
    }
}

// ----------------------------------------------------------------------
// RMSNorm
// ----------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Batch16_RMSNormMatchesThePlaintextLayer)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape);
    const int cols = shape.d_model;

    const auto x = f.random_batch(cols, 5150u);
    auto ct = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);

    llama::Llama3Batch16Operator::RMSNormConfig config;
    bracket_sum(x, Fixture::d, cols, config.sum_lo, config.sum_hi);
    config.degree = 15;
    config.newton_iterations = 0;
    config.fold_mean_into_fit = true;

    const std::vector<double> no_gain;
    auto out = f.nl->rms_norm(ct, no_gain, config, *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), cols);

    const auto got = f.op->decrypt(out, *f.decryptor, f.scale);
    const auto want =
        rms_norm_host(x, Fixture::d, cols, no_gain, config.eps);
    const double err15 = max_abs_diff(want, got);

    // 5e-3, and the number is worth explaining rather than tuning. What is
    // being measured here is the CHEBYSHEV FIT, not the encoding: 1/sqrt has
    // a branch point at 0, so its interpolation error is set by how close the
    // fitted interval comes to it, and this fixture is deliberately the worst
    // case for that. Eight channels of uniform noise put the summed square in
    // a range spanning about 14x, which at degree 15 is worth ~3e-3.
    //
    // The real shape is the opposite: at d_model = 4096 the summed square
    // concentrates as 1/sqrt(4096), the range collapses towards 1.1x, and the
    // same degree is many orders better. So this bound is pessimistic ON
    // PURPOSE and must not be read as the layer's accuracy.
    EXPECT_LT(err15, 5e-3);

    // And the proof that it IS the fit and not the layer: the same circuit at
    // degree 31 over the same data, which changes nothing except the series.
    auto config31 = config;
    config31.degree = 31;
    auto ct31 = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);
    auto out31 = f.nl->rms_norm(ct31, no_gain, config31, *f.galois, *f.relin);
    const double err31 =
        max_abs_diff(want, f.op->decrypt(out31, *f.decryptor, f.scale));

    EXPECT_LT(err31, err15 / 4.0)
        << "raising only the fit degree should collapse the error; if it does "
           "not, the error is coming from the encoding and not the series "
           "(deg 15: " << err15 << ", deg 31: " << err31 << ")";
}

// THE batch-16 test. Sixteen prompts share every ciphertext, so an operation
// that reduces across the instance axis instead of the channel axis returns a
// plausible number that depends on somebody else's data. Perturbing exactly
// one instance and requiring the other fifteen to be unchanged is the only
// thing that catches it; a reference test would not, because the reference
// would be wrong in the same way only if the host code had the same bug.
TEST(HEonGPU, CKKS_Llama3Batch16_RMSNormKeepsInstancesIndependent)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape);
    const int cols = shape.d_model;
    constexpr int poisoned = 3;

    auto x = f.random_batch(cols, 909u);
    auto y = x;
    // Instance 3 gets completely different data of the SAME amplitude. Same
    // amplitude matters: a perturbation that moved the summed square out of
    // the fitted range would move every instance through the fit's error and
    // hide exactly the mixing this test is looking for.
    {
        const auto fresh = f.random_batch(cols, 5555u);
        y[poisoned] = fresh[poisoned];
    }

    llama::Llama3Batch16Operator::RMSNormConfig config;
    double lo_x, hi_x, lo_y, hi_y;
    bracket_sum(x, Fixture::d, cols, lo_x, hi_x);
    bracket_sum(y, Fixture::d, cols, lo_y, hi_y);
    config.sum_lo = std::min(lo_x, lo_y);
    config.sum_hi = std::max(hi_x, hi_y);
    config.degree = 15;
    config.newton_iterations = 0;

    const std::vector<double> no_gain;

    auto ct_x = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);
    auto out_x = f.nl->rms_norm(ct_x, no_gain, config, *f.galois, *f.relin);
    const auto got_x = f.op->decrypt(out_x, *f.decryptor, f.scale);

    auto ct_y = f.op->encrypt(y, Fixture::d, cols, *f.encryptor, f.scale);
    auto out_y = f.nl->rms_norm(ct_y, no_gain, config, *f.galois, *f.relin);
    const auto got_y = f.op->decrypt(out_y, *f.decryptor, f.scale);

    for (int b = 0; b < Fixture::instances; ++b)
    {
        double worst = 0.0;
        for (size_t e = 0; e < got_x[b].size(); ++e)
            worst = std::max(worst, std::abs(got_x[b][e] - got_y[b][e]));

        if (b == poisoned)
        {
            // The test needs power as well as a null: a layer that ignored
            // its input entirely would pass every "unchanged" check below.
            EXPECT_GT(worst, 1e-2)
                << "instance " << b
                << " did not move even though its own data changed";
        }
        else
        {
            // Not exact zero: the two runs are separate encryptions, so the
            // floor is CKKS noise, not bit equality. Mixing would show up as
            // an O(1) move, so this sits three orders above the noise and
            // three below the signal.
            EXPECT_LT(worst, 1e-4)
                << "instance " << b
                << " moved when only instance " << poisoned
                << " changed: the reduction is crossing the batch axis";
        }
    }
}

// The gain is a constant per channel, so it folds into the projection that
// reads the normalised stream -- exactly, on the host, for free. This asserts
// the identity W^T diag(g) y = (diag(g) W)^T y end to end, which is the whole
// argument for the level it saves.
TEST(HEonGPU, CKKS_Llama3Batch16_GainFoldEqualsTheHomomorphicGain)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape);
    const int cols = shape.d_model;
    const int out_channels = 4;

    const auto x = f.random_batch(cols, 2718u);
    const auto gain = random_weight(1, cols, 6006u, 0.9);
    const auto weight = random_weight(cols, out_channels, 1234u, 0.5);

    llama::Llama3Batch16Operator::RMSNormConfig config;
    bracket_sum(x, Fixture::d, cols, config.sum_lo, config.sum_hi);
    config.degree = 15;
    config.newton_iterations = 0;

    // (a) gain applied homomorphically, then the plain weight.
    auto ct_a = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);
    auto norm_a = f.nl->rms_norm(ct_a, gain, config, *f.galois, *f.relin);
    const int depth_a = norm_a.column.front().depth();
    auto proj_a =
        f.op->project(norm_a, weight, cols, out_channels, "fold.a");
    const auto got_a = f.op->decrypt(proj_a, *f.decryptor, f.scale);

    // (b) no homomorphic gain, folded weight.
    std::vector<double> folded = weight;
    llama::Llama3Batch16Operator::fold_gain(folded, gain, cols,
                                            out_channels);
    const std::vector<double> no_gain;
    auto ct_b = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);
    auto norm_b = f.nl->rms_norm(ct_b, no_gain, config, *f.galois, *f.relin);
    const int depth_b = norm_b.column.front().depth();
    auto proj_b =
        f.op->project(norm_b, folded, cols, out_channels, "fold.b");
    const auto got_b = f.op->decrypt(proj_b, *f.decryptor, f.scale);

    EXPECT_LT(max_abs_diff(got_a, got_b), 1e-3);

    // And it is a level, which is the point. depth counts levels SPENT.
    EXPECT_EQ(depth_a, depth_b + 1)
        << "folding the gain should save exactly one level";
}

// The domain map of the 1/sqrt fit is the one level this encoding cannot fold
// away by itself: the reduction is a slot-wise addition, so unlike the
// rectangular path there is no mask between it and the fit to carry the map.
// A caller who owns the weight upstream can carry it there instead -- the sum
// is quadratic in the input, so the factor is the square root and the
// numerator's copy of it comes back out in the fit's coefficients.
TEST(HEonGPU, CKKS_Llama3Batch16_PreScaledSumSavesALevelAndAgrees)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape);
    const int cols = shape.d_model;

    const auto x = f.random_batch(cols, 31337u);

    llama::Llama3Batch16Operator::RMSNormConfig plain_config;
    bracket_sum(x, Fixture::d, cols, plain_config.sum_lo, plain_config.sum_hi);
    plain_config.degree = 15;
    plain_config.newton_iterations = 0;
    plain_config.fold_mean_into_fit = true;

    const std::vector<double> no_gain;
    auto ct_a = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);
    auto out_a = f.nl->rms_norm(ct_a, no_gain, plain_config, *f.galois,
                                *f.relin);
    const int depth_a = out_a.column.front().depth();
    const auto got_a = f.op->decrypt(out_a, *f.decryptor, f.scale);

    // The caller's half: scale the data by c, and hand the fit 1/c back.
    const double c = llama::Llama3Batch16Operator::sum_pre_scale_factor(
        plain_config.sum_lo, plain_config.sum_hi);
    auto scaled = x;
    for (auto& m : scaled)
        for (auto& v : m)
            v *= c;

    llama::Llama3Batch16Operator::RMSNormConfig fast_config = plain_config;
    fast_config.sum_pre_scaled = true;
    fast_config.output_scale = 1.0 / c;

    auto ct_b =
        f.op->encrypt(scaled, Fixture::d, cols, *f.encryptor, f.scale);
    auto out_b =
        f.nl->rms_norm(ct_b, no_gain, fast_config, *f.galois, *f.relin);
    const int depth_b = out_b.column.front().depth();
    const auto got_b = f.op->decrypt(out_b, *f.decryptor, f.scale);

    EXPECT_LT(max_abs_diff(got_a, got_b), 2e-3);
    EXPECT_EQ(depth_b, depth_a - 1)
        << "a pre-scaled sum should skip the fit's own affine multiply";
}

TEST(HEonGPU, CKKS_Llama3Batch16_RMSNormRejectsAWrongGainLength)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape);
    const int cols = shape.d_model;
    const auto x = f.random_batch(cols, 5u);
    auto ct = f.op->encrypt(x, Fixture::d, cols, *f.encryptor, f.scale);

    llama::Llama3Batch16Operator::RMSNormConfig config;
    bracket_sum(x, Fixture::d, cols, config.sum_lo, config.sum_hi);

    const std::vector<double> wrong(static_cast<size_t>(cols) + 1, 1.0);
    EXPECT_THROW(f.nl->rms_norm(ct, wrong, config, *f.galois, *f.relin),
                 std::invalid_argument);

    // A range that was never calibrated is the failure mode Section 4.3 is
    // about, so it is refused rather than fitted over.
    llama::Llama3Batch16Operator::RMSNormConfig uncalibrated;
    const std::vector<double> none;
    EXPECT_THROW(f.nl->rms_norm(ct, none, uncalibrated, *f.galois, *f.relin),
                 std::invalid_argument);
}

// ----------------------------------------------------------------------
// Rotary position embedding
// ----------------------------------------------------------------------

// RoPE is absent from the Algorithm-1 path entirely, so this is the first
// assertion of it on this encoding. It is cheap here for the two reasons
// everything else is: the lane pairing c <-> c + head_dim/2 is a pairing of
// whole ciphertexts (no rotation, no Galois key), and the angle depends on
// the TOKEN, which is the slow slot axis -- so one plaintext per lane pair
// serves every instance and every head.
TEST(HEonGPU, CKKS_Llama3Batch16_RopeMatchesThePlaintextRotation)
{
    llama::Batch16Shape shape = SmallShape();
    shape.heads = 1;
    shape.kv_heads = 1;
    shape.head_dim = Fixture::d;
    shape.d_model = shape.q_channels();
    Fixture f(shape, 6);

    const int channels = shape.d_model;
    const int half = shape.head_dim / 2;
    const auto x = f.random_batch(channels, 271828u, 0.5);

    llama::Llama3Batch16Operator::RopeConfig rope;
    rope.theta = 500000.0;
    rope.position_offset = 0;

    auto ct = f.op->encrypt(x, Fixture::d, channels, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    const int before = slots.front().depth();
    f.nl->rope_slots(slots, rope);
    const int after = slots.front().depth();

    EXPECT_EQ(after, before + 1) << "RoPE should cost exactly one level";

    double worst = 0.0;
    for (int c = 0; c < half; ++c)
    {
        const double omega = std::pow(
            rope.theta, -2.0 * static_cast<double>(c) /
                            static_cast<double>(shape.head_dim));
        const auto got0 = f.decode(slots[static_cast<size_t>(c)]);
        const auto got1 = f.decode(slots[static_cast<size_t>(c + half)]);
        for (int u = 0; u < Fixture::d; ++u)
        {
            const double angle = static_cast<double>(u) * omega;
            const double cs = std::cos(angle);
            const double sn = std::sin(angle);
            for (int b = 0; b < Fixture::instances; ++b)
            {
                const size_t at0 =
                    static_cast<size_t>(u) * channels + c;
                const size_t at1 = at0 + static_cast<size_t>(half);
                const double x0 = x[b][at0];
                const double x1 = x[b][at1];
                const int s = f.nl->slot_of(b, u);
                worst = std::max(worst,
                                 std::abs(got0[s] - (x0 * cs - x1 * sn)));
                worst = std::max(worst,
                                 std::abs(got1[s] - (x0 * sn + x1 * cs)));
            }
        }
    }
    EXPECT_LT(worst, 1e-3);
}

// ----------------------------------------------------------------------
// SwiGLU
// ----------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3Batch16_FeedForwardMatchesThePlaintextLayer)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape);

    const auto x = f.random_batch(shape.d_model, 8675309u, 0.5);
    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 11u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 22u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 33u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig config;
    config.silu_bound = 4.0;
    config.silu_degree = 15;

    auto ct =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out = f.nl->feed_forward(ct, w, config, *f.galois, *f.relin);
    ASSERT_EQ(out.columns(), shape.d_model);
    const auto got = f.op->decrypt(out, *f.decryptor, f.scale);

    std::vector<std::vector<double>> want(x.size());
    for (size_t b = 0; b < x.size(); ++b)
    {
        const auto g =
            host_product(x[b], w.gate, Fixture::d, shape.d_model,
                         shape.hidden);
        const auto u = host_product(x[b], w.up, Fixture::d, shape.d_model,
                                    shape.hidden);
        std::vector<double> hidden(g.size());
        for (size_t e = 0; e < g.size(); ++e)
            hidden[e] = silu_host(g[e]) * u[e];
        want[b] = host_product(hidden, w.down, Fixture::d, shape.hidden,
                               shape.d_model);
    }

    // The SiLU is a degree-15 Chebyshev fit, so the tolerance is the fit's
    // and not the encoding's.
    EXPECT_LT(max_abs_diff(want, got), 5e-2);
}

// The SiLU's domain map, 1/bound, rides on the gate weight instead of on a
// homomorphic product. Same answer, one level less -- and the gate projection
// feeds the SiLU and nothing else, so the factor needs no undoing.
TEST(HEonGPU, CKKS_Llama3Batch16_SiluDomainFoldEqualsTheUnfoldedFit)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape);

    const auto x = f.random_batch(shape.d_model, 4711u, 0.5);
    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 101u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 202u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 303u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig folded;
    folded.silu_bound = 4.0;
    folded.silu_degree = 15;
    folded.fold_silu_domain_into_gate = true;

    llama::Llama3Batch16Operator::FeedForwardConfig unfolded = folded;
    unfolded.fold_silu_domain_into_gate = false;

    auto ct_a =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_a = f.nl->feed_forward(ct_a, w, folded, *f.galois, *f.relin);
    const int depth_a = out_a.column.front().depth();
    const auto got_a = f.op->decrypt(out_a, *f.decryptor, f.scale);

    auto ct_b =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_b = f.nl->feed_forward(ct_b, w, unfolded, *f.galois, *f.relin);
    const int depth_b = out_b.column.front().depth();
    const auto got_b = f.op->decrypt(out_b, *f.decryptor, f.scale);

    EXPECT_LT(max_abs_diff(got_a, got_b), 1e-2);
    EXPECT_EQ(depth_a, depth_b - 1)
        << "folding the SiLU domain map should save exactly one level";
}

// The hidden axis is streamed because the whole width will not fit, and the
// association of a homomorphic sum is the only thing that changes. If it
// changed anything else the memory lever would not be usable.
TEST(HEonGPU, CKKS_Llama3Batch16_HiddenStreamingChangesNothing)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape);

    const auto x = f.random_batch(shape.d_model, 9001u, 0.5);
    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 1u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 2u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 3u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig whole;
    whole.silu_bound = 4.0;
    whole.silu_degree = 15;

    llama::Llama3Batch16Operator::FeedForwardConfig streamed = whole;
    streamed.hidden_block = 2;

    auto ct_a =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_a = f.nl->feed_forward(ct_a, w, whole, *f.galois, *f.relin);
    const auto got_a = f.op->decrypt(out_a, *f.decryptor, f.scale);

    auto ct_b =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_b = f.nl->feed_forward(ct_b, w, streamed, *f.galois, *f.relin);
    const auto got_b = f.op->decrypt(out_b, *f.decryptor, f.scale);

    // Two separate encryptions, so the floor is CKKS noise. What is being
    // asserted is that the ASSOCIATION of the down projection's sum is all
    // that changed -- a different chunking that changed the arithmetic would
    // show up far above this.
    EXPECT_LT(max_abs_diff(got_a, got_b), 1e-4);
    EXPECT_EQ(out_a.column.front().depth(), out_b.column.front().depth());
}

// The SwiGLU is slot-wise from end to end, so it cannot mix instances even in
// principle -- but "cannot in principle" is what the reshape said too, and the
// projections it calls are not this module's code.
TEST(HEonGPU, CKKS_Llama3Batch16_FeedForwardKeepsInstancesIndependent)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape);
    constexpr int poisoned = 11;

    auto x = f.random_batch(shape.d_model, 606u, 0.5);
    auto y = x;
    for (auto& v : y[poisoned])
        v = -v;

    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 7u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 8u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 9u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig config;
    config.silu_bound = 4.0;
    config.silu_degree = 15;

    auto ct_x =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_x = f.nl->feed_forward(ct_x, w, config, *f.galois, *f.relin);
    const auto got_x = f.op->decrypt(out_x, *f.decryptor, f.scale);

    auto ct_y =
        f.op->encrypt(y, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_y = f.nl->feed_forward(ct_y, w, config, *f.galois, *f.relin);
    const auto got_y = f.op->decrypt(out_y, *f.decryptor, f.scale);

    double moved = 0.0;
    for (int b = 0; b < Fixture::instances; ++b)
    {
        double worst = 0.0;
        for (size_t e = 0; e < got_x[b].size(); ++e)
            worst = std::max(worst, std::abs(got_x[b][e] - got_y[b][e]));
        if (b == poisoned)
        {
            moved = worst;
            continue;
        }
        // Two separate encryptions, so the floor is CKKS noise rather than
        // bit equality; mixing would be an O(1) move.
        EXPECT_LT(worst, 1e-4)
            << "instance " << b << " moved when only instance " << poisoned
            << " changed";
    }
    EXPECT_GT(moved, 1e-2)
        << "instance " << poisoned << " did not move even though its own "
                                     "data changed";
}

// ----------------------------------------------------------------------
// The one experiment that decides how much layout conversion a block needs
// ----------------------------------------------------------------------

// Bridging is 98.3% of every Galois rotation in a block, and the SwiGLU's
// three crossings are 64% of the bridging. All three exist for one reason:
// project() is assumed to need matrix form. It may not.
//
// Algorithm 1 computes X.W with the weight encoded through BatchMatrixEncoder.
// The weight handed to project() is one real matrix SHARED by all sixteen
// instances, so its R_k image is the CONSTANT polynomial -- and multiplying by
// a constant polynomial of R_k is scalar multiplication. The projection
// therefore degenerates to out_c = sum_j W[j][c] * ct_j, a scalar
// multiply-accumulate ACROSS ciphertexts, which cannot care what a ciphertext
// encodes. The bridge is linear and acts WITHIN a column. Two such maps
// commute.
//
// If that holds, the SwiGLU never has to leave slot form, and neither does
// RMSNorm: the only thing on this path that genuinely needs the coefficient
// encoding is Algorithm 4, because a ciphertext-ciphertext matrix product is
// what the R_k structure is FOR. This test is the whole question, and it is
// cheap.
TEST(HEonGPU, CKKS_Llama3Batch16_ProjectionCommutesWithTheBridge)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    Fixture f(shape, 8);
    const int in_channels = 4;
    const int out_channels = 3;

    const auto x = f.random_batch(in_channels, 1357u, 0.5);
    const auto w = random_weight(in_channels, out_channels, 2468u, 0.5);

    auto ct =
        f.op->encrypt(x, Fixture::d, in_channels, *f.encryptor, f.scale);

    // (a) the way the code does it today: project in MATRIX form, then bridge.
    auto proj =
        f.op->project(ct, w, in_channels, out_channels, "commute.matrix");
    auto a_slots = f.op->to_slots(proj, *f.galois);
    ASSERT_EQ(a_slots.size(), static_cast<size_t>(out_channels));

    // (b) bridge FIRST, then hand project() slot-form ciphertexts.
    auto ct2 =
        f.op->encrypt(x, Fixture::d, in_channels, *f.encryptor, f.scale);
    llama::BatchActivation slot_act;
    slot_act.rows = Fixture::d;
    slot_act.column = f.op->to_slots(ct2, *f.galois);
    auto b = f.op->project(slot_act, w, in_channels, out_channels,
                           "commute.slot");
    ASSERT_EQ(b.columns(), out_channels);

    // Both must equal the host product read through the slot map -- otherwise
    // they could agree by being wrong in the same way.
    double worst_ab = 0.0;
    double worst_host = 0.0;
    for (int c = 0; c < out_channels; ++c)
    {
        const auto ga = f.decode(a_slots[static_cast<size_t>(c)]);
        const auto gb = f.decode(b.column[static_cast<size_t>(c)]);
        for (int inst = 0; inst < Fixture::instances; ++inst)
        {
            const auto want = host_product(x[inst], w, Fixture::d,
                                           in_channels, out_channels);
            for (int u = 0; u < Fixture::d; ++u)
            {
                const int s = f.nl->slot_of(inst, u);
                const double ref =
                    want[static_cast<size_t>(u) * out_channels + c];
                worst_ab = std::max(worst_ab, std::abs(ga[s] - gb[s]));
                worst_host = std::max(worst_host, std::abs(gb[s] - ref));
            }
        }
    }

    EXPECT_LT(worst_host, 1e-3)
        << "projecting slot-form ciphertexts does not compute the product";
    EXPECT_LT(worst_ab, 1e-3)
        << "project() does not commute with the bridge; the SwiGLU's three "
           "crossings and the RMSNorm's return crossing are all load bearing "
           "after all";
}

// The commutation lets the SwiGLU cross twice around the whole sublayer
// instead of three times over the hidden width -- 8,192 columns instead of
// 43,008 at the 8B shape. Same answer, and the SAME DEPTH: three crossing
// CALLS are only two crossing LEVELS, because the gate and the up projection
// cross concurrently. The saving is bridged columns, not levels, and the
// equality below is what pins that down.
TEST(HEonGPU, CKKS_Llama3Batch16_SlotResidentFeedForwardAgreesAtEqualDepth)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape);

    const auto x = f.random_batch(shape.d_model, 24680u, 0.5);
    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 41u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 42u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 43u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig crossing;
    crossing.silu_bound = 4.0;
    crossing.silu_degree = 15;

    llama::Llama3Batch16Operator::FeedForwardConfig resident = crossing;
    resident.slot_resident = true;

    auto ct_a =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_a = f.nl->feed_forward(ct_a, w, crossing, *f.galois, *f.relin);
    const int depth_a = out_a.column.front().depth();
    const auto got_a = f.op->decrypt(out_a, *f.decryptor, f.scale);

    auto ct_b =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto out_b = f.nl->feed_forward(ct_b, w, resident, *f.galois, *f.relin);
    const int depth_b = out_b.column.front().depth();
    const auto got_b = f.op->decrypt(out_b, *f.decryptor, f.scale);

    EXPECT_LT(max_abs_diff(got_a, got_b), 1e-2);
    EXPECT_EQ(depth_b, depth_a)
        << "moving the crossings should change what is bridged, not how deep "
           "the sublayer is: the gate and the up projection already crossed "
           "at the same depth";
}

// The end state the layout question is actually asking about: a stream held
// in slot form across the block. Both non-linear sublayers then cost NO
// crossing, no rotation and no Galois key at all -- Algorithm 1 needs none by
// construction and everything else here is slot-wise.
TEST(HEonGPU, CKKS_Llama3Batch16_SlotResidentSublayersNeedNoCrossing)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    // Two whole sublayers back to back: the norm is 9 levels with its
    // crossings and the SwiGLU 9, so the matrix-resident arm needs 18 and the
    // default 14-limb fixture runs out inside the SiLU.
    Fixture f(shape, 22);

    const auto x = f.random_batch(shape.d_model, 13579u, 0.5);
    llama::Llama3Batch16Operator::FeedForwardWeights w;
    w.gate = random_weight(shape.d_model, shape.hidden, 51u, 0.5);
    w.up = random_weight(shape.d_model, shape.hidden, 52u, 0.5);
    w.down = random_weight(shape.hidden, shape.d_model, 53u, 0.5);

    llama::Llama3Batch16Operator::FeedForwardConfig config;
    config.silu_bound = 4.0;
    config.silu_degree = 15;

    llama::Llama3Batch16Operator::RMSNormConfig norm;
    bracket_sum(x, Fixture::d, shape.d_model, norm.sum_lo, norm.sum_hi);
    norm.degree = 15;
    norm.newton_iterations = 0;
    const std::vector<double> no_gain;

    // (a) matrix-resident: norm crosses twice, SwiGLU crosses twice.
    auto ct_a =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto norm_a = f.nl->rms_norm(ct_a, no_gain, norm, *f.galois, *f.relin);
    llama::Llama3Batch16Operator::FeedForwardConfig resident = config;
    resident.slot_resident = true;
    auto out_a = f.nl->feed_forward(norm_a, w, resident, *f.galois, *f.relin);
    const auto got_a = f.op->decrypt(out_a, *f.decryptor, f.scale);

    // (b) slot-resident: ONE crossing in, one out, for both sublayers
    // together.
    auto ct_b =
        f.op->encrypt(x, Fixture::d, shape.d_model, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct_b, *f.galois);
    auto normed = f.nl->rms_norm_slots(slots, no_gain, norm, *f.galois,
                                       *f.relin);
    auto hidden =
        f.nl->feed_forward_slots(normed, w, config, *f.relin);
    auto out_b = f.op->from_slots(hidden, Fixture::d, *f.galois);
    const auto got_b = f.op->decrypt(out_b, *f.decryptor, f.scale);

    EXPECT_LT(max_abs_diff(got_a, got_b), 1e-2);

    // THIS is where the levels are. The norm's return crossing and the
    // SwiGLU's entry crossing are adjacent and cancel outright when the
    // stream simply stays in slot form, so the pair is two levels shallower
    // -- and every one of the non-attention bridged columns is gone with
    // them.
    EXPECT_EQ(out_b.column.front().depth(),
              out_a.column.front().depth() - 2)
        << "a slot-resident norm/SwiGLU pair should drop exactly the two "
           "crossings that meet between them";
}

// ----------------------------------------------------------------------
// The cost accounting
// ----------------------------------------------------------------------

// The bridge is the only thing this module spends, so the count of what it
// bridges is the cost model, and it belongs in a test rather than in a
// comment that can drift.
TEST(HEonGPU, CKKS_Llama3Batch16_BridgeAccountingIsTheCostModel)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4096;
    shape.hidden = 14336;
    shape.heads = 32;
    shape.kv_heads = 8;
    shape.head_dim = Fixture::d;
    Fixture f(shape, 3); // accounting only; nothing is encrypted

    // Two norms down and back over the residual width, plus the SwiGLU's
    // gate, up and hidden over the hidden width. The attention sublayer's own
    // crossings belong to another module and are deliberately not counted.
    EXPECT_EQ(f.nl->bridged_columns_per_block(),
              2LL * 2LL * 4096LL + 3LL * 14336LL);

    // Grouped-query attention narrows the key and value projections by the
    // full factor and nothing widens them back.
    EXPECT_EQ(shape.q_channels(), 4096);
    EXPECT_EQ(shape.kv_channels(), 1024);
    EXPECT_EQ(shape.group(), 4);
}
