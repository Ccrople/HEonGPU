// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The two things a whole Llama-3-8B block at batch 16 was still missing: a
// REFRESH, and a sequence longer than d = 128 tokens.
//
// Both were recorded as open in Doing.md's own gap table -- "bootstrapping:
// still absent, and unreachable: no boot key is ever passed" and "sequences
// longer than d = 128 tokens: still no code path; there is no token blocking
// here" -- and both are load-bearing for the phrase "8B block":
//
//   - at batch 16 the ring is N = 4096, the 128-bit cap is 109 bits of log QP,
//     and 41 + 33 + 33 = 107 buys exactly ONE usable level. A block spends
//     around sixty. An unrefreshed batch-16 block is not slow, it is
//     impossible.
//   - d = 128 tokens is 1/64 of Llama-3-8B's 8192-token context, and d is
//     pinned by the same arithmetic that pins the batch.
//
// WHAT THESE TESTS ARE FOR, which is not the same as what they compute.
//
// Token blocking has three failure modes and only one of them is an error.
//
//   1. Getting the ARITHMETIC wrong -- caught by a host reference.
//   2. Getting CAUSALITY wrong. A mask that leaks one future key returns a
//      perfectly plausible number; no round trip and no reference-free check
//      sees it. CausalityHoldsAcrossTokenBlocks perturbs a LATER token and
//      requires every earlier token's output to be unchanged, which is the
//      only assertion that fails when a triangle is off by one block.
//   3. Getting the GENERALISATION wrong. The blocked path must agree with the
//      single-block path where they overlap, or every measurement taken before
//      this existed is describing a different circuit. Two tests pin that at
//      one block, one on the seam and one on the whole sublayer.
//
// The refresh has a failure mode of its own: a bootstrap that quietly does
// not happen. Every seam that can refresh throws without the boot key rather
// than falling through, and that is asserted -- because a silent fallback
// changes the level schedule the caller sized its chain for and surfaces as an
// unrelated throw several layers later.
//
// CHAIN LENGTHS ARE COMPUTED, NOT GUESSED. Every fixture below is sized from
// Llama3BatchOperator::softmax_seam_levels plus the ledger of the layers
// around it, because the seam is 23 levels at the production fit degrees and a
// chain that is one short fails inside a Chebyshev recursion with an error
// that names neither the seam nor the test.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;
    using Batch = llama::Llama3BatchOperator;

    /// The production batch-16 ring: batch = k/2 = 16 forces k = 32, and
    /// d = N/32 = 128 is at once the token block, the head dim, and the ring's
    /// only legal choice. The model widths are cut down; the RING is real,
    /// which is the half the layout and causality claims depend on.
    struct Fixture
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 128;
        static constexpr int instances = 16;

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
        std::unique_ptr<Batch> op;
        std::unique_ptr<llama::Llama3Batch16Operator> nl;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        explicit Fixture(const llama::Batch16Shape& shape, int limb_count)
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
            op = std::make_unique<Batch>(context, *encoder, layout, scale);
            nl = std::make_unique<llama::Llama3Batch16Operator>(*op, shape);

            std::vector<int> shifts = op->rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        std::vector<std::vector<double>> random_batch(int rows, int cols,
                                                      uint64_t seed,
                                                      double amp = 1.0) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-amp, amp);
            std::vector<std::vector<double>> out(
                instances,
                std::vector<double>(static_cast<size_t>(rows) * cols));
            for (auto& m : out)
                for (auto& v : m)
                    v = dist(rng);
            return out;
        }
    };

    /// A model narrow enough to run in seconds. head_dim is layout.d, which is
    /// not a preference: Algorithm 4's operands are square at d.
    llama::Batch16Shape SmallShape()
    {
        llama::Batch16Shape shape;
        shape.instances = Fixture::instances;
        shape.d_model = 8;
        shape.hidden = 16;
        shape.heads = 1;
        shape.kv_heads = 1;
        shape.head_dim = Fixture::d;
        return shape;
    }

    /// The seam settings the SoftMax session settled on, and the ONLY ones the
    /// chain arithmetic below is valid for. Degree 63 on the reciprocal is not
    /// caution: a bare degree-15 fit with no Newton step is 33% wrong over the
    /// range a causal row actually produces.
    void production_softmax(llama::Llama3Operator::SoftmaxConfig& softmax,
                            Batch::BatchSoftmaxSeamConfig& seam, double bound)
    {
        softmax.bound = bound;
        softmax.iterations = 2;
        softmax.exp_degree = 15;
        softmax.inverse_degree = 63;
        softmax.inverse_newton = 0;

        seam.causal = true;
        seam.scores_carry_exp_domain = true;
        seam.fold_affine_into_mask = true;
        seam.hoisted_crossings = true;
        seam.cache_masks = true;
    }

    /// The cheap settings, for tests that COMPARE two paths rather than
    /// measure one. Halves the seam, which halves the chain, which is the
    /// difference between a test that runs in a minute and one that does not.
    void cheap_softmax(llama::Llama3Operator::SoftmaxConfig& softmax,
                       Batch::BatchSoftmaxSeamConfig& seam, double bound)
    {
        softmax.bound = bound;
        softmax.iterations = 1;
        softmax.exp_degree = 15;
        softmax.inverse_degree = 15;
        softmax.inverse_newton = 0;

        seam.causal = true;
        seam.scores_carry_exp_domain = true;
        seam.fold_affine_into_mask = true;
        seam.hoisted_crossings = true;
        seam.cache_masks = true;
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

    double worst_abs(const std::vector<std::vector<double>>& a,
                     const std::vector<std::vector<double>>& b)
    {
        double worst = 0.0;
        for (size_t s = 0; s < a.size(); ++s)
            for (size_t e = 0; e < a[s].size(); ++e)
                worst = std::max(worst, std::abs(a[s][e] - b[s][e]));
        // std::max against a NaN keeps the running maximum, so a NaN would
        // sail through as a small error rather than a large one.
        if (std::isnan(worst))
            return std::numeric_limits<double>::infinity();
        return worst;
    }

    /// RMSNorm on the host: x_j / sqrt(mean_j(x_j^2) + eps), with the channel
    /// axis running fastest, which is this encoding's ciphertext axis.
    std::vector<std::vector<double>>
    rms_norm_host(const std::vector<std::vector<double>>& x, int rows,
                  int cols, double eps)
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
                    out[b][at] = x[b][at] / r;
                }
            }
        }
        return out;
    }

    /// The interval the summed square actually visits, widened a little. A
    /// range is a measurement of the data, not a property of the algorithm,
    /// and fitting 1/sqrt over a worst-case bound instead is what made the
    /// SoftMax reciprocal wrong by 98.6% on the rectangular path.
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

    /// Mean magnitude, to turn an absolute error into a relative one. An
    /// absolute error means nothing across shapes: a wider model gives a wider
    /// range and the same circuit looks worse.
    double mean_abs(const std::vector<std::vector<double>>& a)
    {
        double acc = 0.0;
        size_t n = 0;
        for (const auto& m : a)
            for (double v : m)
            {
                acc += std::abs(v);
                n++;
            }
        return n == 0 ? 0.0 : acc / static_cast<double>(n);
    }
} // namespace

// ======================================================================
// 1. The blocked causal mask -- host arithmetic, no GPU work
// ======================================================================

// The generalisation has to REDUCE to what it generalises, exactly, or every
// measurement the single-block path produced is describing a different
// circuit. Checked entry by entry rather than by a tolerance: both sides are
// host arithmetic and there is no reason for them to differ in the last bit.
TEST(HEonGPU, CKKS_Llama3Batch16Block_BlockedMaskReducesToTheSingleBlockOne)
{
    Fixture f(SmallShape(), 3);

    // EVERY key, not a sample. This is the only thing standing between the
    // blocked mask and Llama3RectOperator::attention, which calls
    // batch_.causal_column_mask(j) for j in [0, d) (llama3_rect.cu:2485) --
    // so if the delegation drifted by one ulp on one key, the rectangular
    // path's SoftMax would move and nothing else in this suite would notice.
    for (int key = 0; key < Fixture::d; ++key)
    {
        const std::vector<double> single = f.op->causal_column_mask(key);
        const std::vector<double> blocked = f.op->causal_column_mask(0, 0, key);
        ASSERT_EQ(single.size(), blocked.size()) << "key " << key;
        for (size_t s = 0; s < single.size(); ++s)
            ASSERT_DOUBLE_EQ(single[s], blocked[s])
                << "key " << key << " slot " << s;
    }

    // At query block zero the id IS the key index, so the encoded-plaintext
    // cache does not see two names for one mask when a caller mixes the
    // single-block and the blocked entry points.
    for (int key : {0, 5, Fixture::d - 1})
        EXPECT_EQ(f.op->causal_mask_id(0, 0, key), key);
}

// The pattern and the weights, both, at a query block that is not the first.
// The pattern is what causality IS; the weight is what keeps the reciprocal's
// range where it was calibrated, and getting that wrong is silent.
TEST(HEonGPU, CKKS_Llama3Batch16Block_BlockedMaskPatternAndWeights)
{
    Fixture f(SmallShape(), 3);
    const int d = Fixture::d;
    const int step = Fixture::instances;
    const int p = 2; // query block 2: at most 3 * d = 384 keys visible

    auto weight_of = [&](int u)
    {
        return std::sqrt(static_cast<double>(p + 1) * d /
                         (static_cast<double>(p) * d + u + 1));
    };

    // On the diagonal block: query u keeps keys 0..u of it.
    for (int key : {0, 1, 64, d - 1})
    {
        const std::vector<double> m = f.op->causal_column_mask(p, p, key);
        for (int u = 0; u < d; ++u)
            for (int b = 0; b < step; ++b)
            {
                const double got = m[static_cast<size_t>(b + u * step)];
                if (u < key)
                    ASSERT_DOUBLE_EQ(got, 0.0)
                        << "a query must not see a key above it: u " << u
                        << " key " << key;
                else
                    ASSERT_NEAR(got, weight_of(u), 1e-12)
                        << "u " << u << " key " << key;
            }
    }

    // Below the diagonal: every query sees every key, so there is no zero at
    // all -- and the mask does not depend on WHICH block below it is.
    const std::vector<double> below = f.op->causal_column_mask(p, 0, 5);
    const std::vector<double> below2 = f.op->causal_column_mask(p, 1, 99);
    ASSERT_EQ(below.size(), below2.size());
    for (size_t s = 0; s < below.size(); ++s)
    {
        ASSERT_DOUBLE_EQ(below[s], below2[s]) << "slot " << s;
        ASSERT_GT(below[s], 0.0) << "slot " << s;
    }
    for (int u = 0; u < d; ++u)
        ASSERT_NEAR(below[static_cast<size_t>(u * step)], weight_of(u), 1e-12);

    // The weight really is the one that leaves the sum of squares where a full
    // row would leave it: visible * w^2 == full.
    for (int u : {0, 1, 50, d - 1})
    {
        const double visible = static_cast<double>(p) * d + u + 1;
        const double w = weight_of(u);
        EXPECT_NEAR(visible * w * w, static_cast<double>(p + 1) * d, 1e-9);
    }

    // A future key block is refused rather than clamped: an all-zero mask
    // would hide a wrong schedule instead of reporting it.
    EXPECT_THROW(f.op->causal_column_mask(1, 2, 0), std::invalid_argument);
    EXPECT_THROW(f.op->causal_column_mask(0, 0, d), std::invalid_argument);
    EXPECT_THROW(f.op->causal_mask_id(1, 2, 0), std::invalid_argument);
}

// A query block uses exactly d + 1 distinct masks whatever its index, and
// every key block below it shares one id. That is what makes a plaintext
// cache of d + 1 enough for a whole sublayer's heads.
TEST(HEonGPU, CKKS_Llama3Batch16Block_MaskIdsAreDPlusOnePerQueryBlock)
{
    Fixture f(SmallShape(), 3);
    const int d = Fixture::d;

    for (int p : {0, 1, 5})
    {
        std::set<int> ids;
        for (int q = 0; q <= p; ++q)
            for (int j = 0; j < d; ++j)
                ids.insert(f.op->causal_mask_id(p, q, j));
        const size_t want =
            p == 0 ? static_cast<size_t>(d) : static_cast<size_t>(d + 1);
        EXPECT_EQ(ids.size(), want) << "query block " << p;
    }

    // Ids of different query blocks never collide, because their weights
    // differ -- a collision would serve one block another block's calibration.
    std::set<int> all;
    for (int p = 0; p < 4; ++p)
        for (int q = 0; q <= p; ++q)
            for (int j = 0; j < d; ++j)
                all.insert(f.op->causal_mask_id(p, q, j));
    EXPECT_EQ(all.size(), static_cast<size_t>(d + (d + 1) * 3));
}

// ======================================================================
// 2. The cost model of the m^2 schedule
// ======================================================================

TEST(HEonGPU, CKKS_Llama3Batch16Block_SequenceProductCountIsTheTriangle)
{
    Fixture f(SmallShape(), 3);

    Batch::BatchAttentionConfig cfg;
    cfg.in_channels = 8;
    cfg.heads = 4;
    cfg.kv_heads = 1;
    cfg.q_channels = cfg.heads * Fixture::d;
    cfg.kv_channels = cfg.kv_heads * Fixture::d;

    for (int blocks : {1, 2, 8, 64})
    {
        const auto c = f.op->sequence_product_count(blocks, cfg);
        const long long pairs =
            static_cast<long long>(blocks) * (blocks + 1) / 2;
        EXPECT_EQ(c.score, pairs * cfg.heads) << "blocks " << blocks;
        EXPECT_EQ(c.value, c.score);
        EXPECT_EQ(c.bridged_columns, 2LL * pairs * cfg.heads * Fixture::d);
        EXPECT_EQ(c.peak_slot_columns,
                  static_cast<long long>(blocks) * Fixture::d);
    }

    // The number the 8B shape actually implies, so the m^2 is on the record
    // rather than in a comment: 64 token blocks is 8192 tokens, and causality
    // is what keeps it at 2080 block pairs rather than 4096.
    const auto full = f.op->sequence_product_count(64, cfg);
    EXPECT_EQ(full.score, 2080LL * 4);
    EXPECT_EQ(full.peak_slot_columns, 8192LL);
    std::cout << "64 token blocks (8192 tokens), 4 heads: " << full.score
              << " score products, " << full.bridged_columns
              << " bridged columns, peak " << full.peak_slot_columns
              << " slot ciphertexts" << std::endl;

    EXPECT_THROW(f.op->sequence_product_count(0, cfg), std::invalid_argument);
}

// ======================================================================
// 3. The blocked SoftMax seam, against a host reference
// ======================================================================

// The decisive arithmetic test of token blocking, and the reason the seam is a
// public entry point rather than an implementation detail: it is checked
// against an exact host SoftMax over the whole 2d-long visible row, so a
// denominator that summed only its own block, or a mask off by one block,
// fails HERE rather than at the end of a sublayer where nothing would name it.
TEST(HEonGPU,
     CKKS_Llama3Batch16Block_BlockedSeamMatchesAHostSoftmaxOverTwoBlocks)
{
    llama::Llama3Operator::SoftmaxConfig softmax;
    Batch::BatchSoftmaxSeamConfig seam;
    const double bound = 4.0;
    const double shift = 0.0;
    production_softmax(softmax, seam, bound);

    const int d = Fixture::d;
    const int inst = Fixture::instances;

    // Scores already in [-bound, 0], drawn rather than formed by a product so
    // that this test isolates the seam from everything around it.
    std::mt19937_64 rng(90210u);
    std::uniform_real_distribution<double> dist(-bound, 0.0);
    std::vector<std::vector<std::vector<double>>> raw(2);
    for (int q = 0; q < 2; ++q)
    {
        raw[q].assign(inst,
                      std::vector<double>(static_cast<size_t>(d) * d, 0.0));
        for (auto& m : raw[q])
            for (auto& v : m)
                v = dist(rng);
    }

    // Section 4.3: a sharp range, measured from the data. Query block 1
    // reduces over 2d = 256 keys, and the mask weight is part of the
    // denominator because the mask multiplies the exponentials.
    {
        double lo = 1e300;
        double hi = 0.0;
        const double half = std::pow(2.0, softmax.iterations);
        for (int b = 0; b < inst; ++b)
            for (int u = 0; u < d; ++u)
            {
                double acc = 0.0;
                for (int q = 0; q <= 1; ++q)
                    for (int j = 0; j < d; ++j)
                    {
                        if (q == 1 && j > u)
                            continue;
                        const double s =
                            raw[q][b][static_cast<size_t>(u) * d + j];
                        const double w = std::sqrt(
                            2.0 * d / (static_cast<double>(d) + u + 1.0));
                        const double e = w * std::exp(s / half);
                        acc += e * e;
                    }
                lo = std::min(lo, acc);
                hi = std::max(hi, acc);
            }
        softmax.sum_lo = lo * 0.8;
        softmax.sum_hi = hi * 1.25;
    }

    // Section 4.3 again, and for the same reason -- this one is what the
    // blocked seam actually needs. `concentration` bounds the post-round sum
    // of squares as a multiple of its uniform value 1/D, and zero keeps the
    // WORST case, D. At D = 2d = 256 that asks a reciprocal to cover 768:1,
    // which no affordable degree fits and which this data does not remotely
    // visit. After round zero the coordinates are exp(x/2)/S over the visible
    // keys -- the mask weight is constant along the key axis and cancels
    // exactly, which is the property the weight was chosen for -- so the
    // bound is computable here.
    {
        const double D = 2.0 * d;
        double top = 0.0;
        for (int b = 0; b < inst; ++b)
            for (int u = 0; u < d; ++u)
            {
                double total = 0.0;
                double squares = 0.0;
                for (int q = 0; q <= 1; ++q)
                    for (int j = 0; j < d; ++j)
                    {
                        if (q == 1 && j > u)
                            continue;
                        total += std::exp(
                            raw[q][b][static_cast<size_t>(u) * d + j] / 2.0);
                    }
                for (int q = 0; q <= 1; ++q)
                    for (int j = 0; j < d; ++j)
                    {
                        if (q == 1 && j > u)
                            continue;
                        const double p =
                            std::exp(raw[q][b][static_cast<size_t>(u) * d + j] /
                                     2.0) /
                            total;
                        squares += p * p;
                    }
                top = std::max(top, D * squares);
            }
        softmax.concentration = top * 1.25;
        std::cout << "calibrated concentration " << softmax.concentration
                  << " against a worst case of " << D << std::endl;
    }
    seam.score_shift = shift;

    // 23 levels at these degrees, plus the three the encryption and the two
    // crossings want on top.
    const int levels = Batch::softmax_seam_levels(softmax, seam);
    std::cout << "blocked seam ledger: " << levels << " levels" << std::endl;
    Fixture f(SmallShape(), levels + 3);

    // scores_carry_exp_domain is an ASSERTION about the input, so the test
    // has to make it true: the domain map rides on whatever formed the score,
    // which here is the test itself.
    const double map = Batch::exp_domain_scale(bound);
    std::vector<llama::BatchActivation> scores;
    for (int q = 0; q < 2; ++q)
    {
        std::vector<std::vector<double>> scaled = raw[q];
        for (auto& m : scaled)
            for (auto& v : m)
                v *= map;
        scores.push_back(f.op->encrypt(scaled, d, d, *f.encryptor, f.scale));
    }

    std::vector<llama::BatchActivation> got = f.op->softmax_seam_blocked(
        scores, 1, softmax, seam, *f.galois, *f.relin);
    ASSERT_EQ(got.size(), 2u);

    // The reference: an exact causal SoftMax over the 256-long row.
    std::vector<std::vector<std::vector<double>>> want(2);
    for (int q = 0; q < 2; ++q)
        want[q].assign(inst,
                       std::vector<double>(static_cast<size_t>(d) * d, 0.0));
    for (int b = 0; b < inst; ++b)
        for (int u = 0; u < d; ++u)
        {
            double total = 0.0;
            for (int q = 0; q <= 1; ++q)
                for (int j = 0; j < d; ++j)
                {
                    if (q == 1 && j > u)
                        continue;
                    total +=
                        std::exp(raw[q][b][static_cast<size_t>(u) * d + j]);
                }
            for (int q = 0; q <= 1; ++q)
                for (int j = 0; j < d; ++j)
                {
                    if (q == 1 && j > u)
                        continue;
                    want[q][b][static_cast<size_t>(u) * d + j] =
                        std::exp(raw[q][b][static_cast<size_t>(u) * d + j]) /
                        total;
                }
        }

    double worst = 0.0;
    double worst_masked = 0.0;
    double peak = 0.0;
    for (int q = 0; q < 2; ++q)
    {
        std::vector<std::vector<double>> out =
            f.op->decrypt(got[q], *f.decryptor, f.scale);
        for (int b = 0; b < inst; ++b)
            for (int u = 0; u < d; ++u)
                for (int j = 0; j < d; ++j)
                {
                    const size_t at = static_cast<size_t>(u) * d + j;
                    const double delta = std::abs(out[b][at] - want[q][b][at]);
                    peak = std::max(peak, want[q][b][at]);
                    if (q == 1 && j > u)
                        worst_masked = std::max(worst_masked, delta);
                    else
                        worst = std::max(worst, delta);
                }
    }
    std::cout << "blocked seam over 2 blocks: worst " << worst << " ("
              << 100.0 * worst / peak << "% of the peak probability " << peak
              << "), worst inside the mask " << worst_masked << std::endl;

    ASSERT_FALSE(std::isnan(worst));
    // Judged on the RATIO, which is this project's own rule: an absolute error
    // means nothing across shapes, and a 256-long row's probabilities are half
    // the size of a 128-long row's before anything goes wrong. The SoftMax and
    // reciprocal fits carry a few per cent here; what this bound is really
    // guarding is the DENOMINATOR, and a seam that summed only its own block
    // would be 50-100% out rather than a few per cent.
    EXPECT_LT(worst, 0.05 * peak);
    // Causality from the arithmetic side, and this one IS sharp: a masked-off
    // key is really zero, not merely small.
    EXPECT_LT(worst_masked, 5e-5);

    // A schedule that hands over the wrong number of blocks is refused, not
    // silently run over a truncated row.
    std::vector<llama::BatchActivation> one;
    one.push_back(f.op->encrypt(raw[0], d, d, *f.encryptor, f.scale));
    EXPECT_THROW(
        f.op->softmax_seam_blocked(one, 1, softmax, seam, *f.galois, *f.relin),
        std::invalid_argument);
}

// ======================================================================
// 4. The blocked sublayer
// ======================================================================

namespace
{
    /// Everything the sublayer tests configure identically, so that a
    /// difference between two of them is the thing under test and not a
    /// difference in setup.
    struct SublayerSetup
    {
        Batch::BatchAttentionConfig cfg;
        Batch::BatchAttentionWeights w;
        int limbs = 0;
    };

    SublayerSetup make_sublayer(const llama::Batch16Shape& shape,
                                uint64_t seed, bool cheap)
    {
        SublayerSetup s;
        s.cfg.in_channels = shape.d_model;
        s.cfg.heads = shape.heads;
        s.cfg.kv_heads = shape.kv_heads;
        s.cfg.q_channels = shape.q_channels();
        s.cfg.kv_channels = shape.kv_channels();
        s.cfg.causal = true;
        if (cheap)
            cheap_softmax(s.cfg.softmax, s.cfg.seam, 4.0);
        else
            production_softmax(s.cfg.softmax, s.cfg.seam, 4.0);
        // The scores are a product of two projections of a bounded input, so
        // they straddle zero; the shift is what puts them inside [-bound, 0],
        // which is where the exponential is FITTED. Leaving them straddling
        // zero extrapolates a degree-15 Chebyshev past its domain, and that
        // does not fail, it just stops meaning anything.
        s.cfg.score_shift = 1.0;
        s.cfg.seam.score_shift = 1.0;
        // Round zero's denominator over a 2d-long row, bracketed loosely.
        // These tests compare two code paths against each other or perturb one
        // input, so the fit's own error cancels or is irrelevant; the
        // calibration that matters is asserted in the seam test above, against
        // a host reference.
        s.cfg.softmax.sum_lo = 5.0;
        s.cfg.softmax.sum_hi = 6.0e2;

        s.w.query = random_weight(shape.d_model, s.cfg.q_channels, seed, 0.05);
        s.w.key =
            random_weight(shape.d_model, s.cfg.kv_channels, seed + 1, 0.05);
        s.w.value =
            random_weight(shape.d_model, s.cfg.kv_channels, seed + 2, 0.5);
        s.w.output =
            random_weight(s.cfg.q_channels, shape.d_model, seed + 3, 0.05);

        // Q/K/V projection 1, the score product 1, the seam, the value
        // product 1, the output projection 1 -- and one prime left over.
        s.limbs = Batch::softmax_seam_levels(s.cfg.softmax, s.cfg.seam) + 6;
        return s;
    }
} // namespace

// At one token block the blocked schedule IS the old one, so it has to give
// the old answer at the old depth. This is what stops token blocking from
// silently redefining every measurement taken before it existed.
TEST(HEonGPU,
     CKKS_Llama3Batch16Block_AttentionSequenceAtOneBlockMatchesAttention)
{
    llama::Batch16Shape shape = SmallShape();
    shape.heads = 2;
    shape.kv_heads = 1;

    SublayerSetup s = make_sublayer(shape, 11u, /*cheap=*/true);
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 777u, 0.5);

    llama::BatchActivation a =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    llama::BatchActivation single =
        f.op->attention(a, s.w, s.cfg, *f.galois, *f.relin);
    const auto single_out = f.op->decrypt(single, *f.decryptor, f.scale);

    std::vector<llama::BatchActivation> seq;
    seq.push_back(f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));
    std::vector<llama::BatchActivation> blocked =
        f.op->attention_sequence(seq, s.w, s.cfg, *f.galois, *f.relin);
    ASSERT_EQ(blocked.size(), 1u);
    const auto blocked_out = f.op->decrypt(blocked[0], *f.decryptor, f.scale);

    EXPECT_EQ(blocked[0].column.front().depth(),
              single.column.front().depth())
        << "the blocked schedule must not spend a different number of levels";

    const double worst = worst_abs(single_out, blocked_out);
    const double magnitude = mean_abs(single_out);
    std::cout << "one block, blocked vs single: worst " << worst
              << " on a mean magnitude of " << magnitude << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    // Not merely close: the two run the same masks, the same plaintexts and
    // the same products, so what is left is the two encryptions' own noise.
    EXPECT_LT(worst, 1e-5 + 1e-3 * magnitude);
}

namespace
{
    /// Run a two-block sequence and hand back both blocks in the clear.
    std::vector<std::vector<std::vector<double>>>
    run_sequence(Fixture& f, const SublayerSetup& s,
                 const std::vector<std::vector<double>>& b0,
                 const std::vector<std::vector<double>>& b1)
    {
        const int d = Fixture::d;
        std::vector<llama::BatchActivation> seq;
        seq.push_back(
            f.op->encrypt(b0, d, s.cfg.in_channels, *f.encryptor, f.scale));
        seq.push_back(
            f.op->encrypt(b1, d, s.cfg.in_channels, *f.encryptor, f.scale));
        std::vector<llama::BatchActivation> out =
            f.op->attention_sequence(seq, s.w, s.cfg, *f.galois, *f.relin);
        std::vector<std::vector<std::vector<double>>> plain;
        for (auto& b : out)
            plain.push_back(f.op->decrypt(b, *f.decryptor, f.scale));
        return plain;
    }
} // namespace

// The assertion no reference test and no round trip can make. Perturb a token
// in the LAST block and require every token of the FIRST to come back
// unchanged: a causal mask that leaks one future key, or a schedule that
// computes the q > p products, still returns plausible numbers and fails only
// here.
TEST(HEonGPU, CKKS_Llama3Batch16Block_CausalityHoldsAcrossTokenBlocks)
{
    llama::Batch16Shape shape = SmallShape();
    SublayerSetup s = make_sublayer(shape, 21u, /*cheap=*/true);
    s.w.output.clear(); // W_o mixes channels and is not what is under test
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x0 = f.random_batch(d, shape.d_model, 555u, 0.5);
    const auto x1 = f.random_batch(d, shape.d_model, 556u, 0.5);

    const auto before = run_sequence(f, s, x0, x1);

    // One token of the SECOND block, in one instance and one channel.
    auto x1_moved = x1;
    x1_moved[3][static_cast<size_t>(9) * shape.d_model + 2] += 0.9;
    const auto after = run_sequence(f, s, x0, x1_moved);

    ASSERT_EQ(before.size(), 2u);
    ASSERT_EQ(after.size(), 2u);

    const double block0_move = worst_abs(before[0], after[0]);
    const double block1_move = worst_abs(before[1], after[1]);
    std::cout << "perturbing a token of block 1 moved block 0 by "
              << block0_move << " and block 1 by " << block1_move << std::endl;

    ASSERT_FALSE(std::isnan(block0_move));
    ASSERT_FALSE(std::isnan(block1_move));
    // The past cannot see the future. What remains in block 0 is CKKS noise
    // from re-encrypting, which is why this is a bound and not an equality.
    EXPECT_LT(block0_move, 1e-4);
    // And the perturbation really did reach the circuit, so the bound above is
    // not passing because nothing happened.
    EXPECT_GT(block1_move, 1e-3);
}

// The other direction, and the one a per-block SoftMax would get wrong: a
// token of the FIRST block must move the SECOND, because every query of block
// 1 attends every key of block 0.
TEST(HEonGPU, CKKS_Llama3Batch16Block_TheFutureDoesSeeThePast)
{
    llama::Batch16Shape shape = SmallShape();
    SublayerSetup s = make_sublayer(shape, 31u, /*cheap=*/true);
    s.w.output.clear();
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x0 = f.random_batch(d, shape.d_model, 606u, 0.5);
    const auto x1 = f.random_batch(d, shape.d_model, 707u, 0.5);

    const auto before = run_sequence(f, s, x0, x1);
    auto x0_moved = x0;
    x0_moved[3][static_cast<size_t>(9) * shape.d_model + 2] += 0.9;
    const auto after = run_sequence(f, s, x0_moved, x1);

    const double block1_move = worst_abs(before[1], after[1]);
    std::cout << "perturbing a token of block 0 moved block 1 by "
              << block1_move << std::endl;
    ASSERT_FALSE(std::isnan(block1_move));
    EXPECT_GT(block1_move, 1e-3)
        << "if this fails the second block is not attending the first at all, "
           "and the sequence is two independent 128-token runs";
}

// Sixteen independent prompts share every ciphertext, so an operation that
// mixes two of them returns a plausible number that is a function of somebody
// else's input. Token blocking gives that a second axis to go wrong on.
TEST(HEonGPU, CKKS_Llama3Batch16Block_InstancesStaySeparateAcrossBlocks)
{
    llama::Batch16Shape shape = SmallShape();
    SublayerSetup s = make_sublayer(shape, 41u, /*cheap=*/true);
    s.w.output.clear();
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x0 = f.random_batch(d, shape.d_model, 808u, 0.5);
    const auto x1 = f.random_batch(d, shape.d_model, 909u, 0.5);

    const auto before = run_sequence(f, s, x0, x1);

    // Instance 5 only, in both token blocks.
    auto x0m = x0;
    auto x1m = x1;
    for (auto& v : x0m[5])
        v += 0.4;
    for (auto& v : x1m[5])
        v += 0.4;
    const auto after = run_sequence(f, s, x0m, x1m);

    double others = 0.0;
    double moved = 0.0;
    for (int t = 0; t < 2; ++t)
        for (int b = 0; b < Fixture::instances; ++b)
        {
            double worst = 0.0;
            for (size_t e = 0; e < before[t][b].size(); ++e)
                worst =
                    std::max(worst, std::abs(before[t][b][e] - after[t][b][e]));
            if (b == 5)
                moved = std::max(moved, worst);
            else
                others = std::max(others, worst);
        }
    std::cout << "instance 5 moved by " << moved
              << ", every other instance by at most " << others << std::endl;
    ASSERT_FALSE(std::isnan(others));
    EXPECT_LT(others, 1e-4);
    EXPECT_GT(moved, 1e-3);
}

// A bidirectional sequence is refused rather than run causally. Returning a
// causal answer to a non-causal request is a different model that still
// produces numbers.
TEST(HEonGPU, CKKS_Llama3Batch16Block_NonCausalBlockingIsRefused)
{
    llama::Batch16Shape shape = SmallShape();
    SublayerSetup s = make_sublayer(shape, 71u, /*cheap=*/true);
    s.cfg.causal = false;
    Fixture f(shape, 6);

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 3579u, 0.5);
    std::vector<llama::BatchActivation> seq;
    seq.push_back(
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));
    seq.push_back(
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));

    EXPECT_THROW(f.op->attention_sequence(seq, s.w, s.cfg, *f.galois, *f.relin),
                 std::invalid_argument);

    // The guard is on the SCHEDULE, not on the flag: one block has no block
    // pair for the missing q > p products to be needed for, so a bidirectional
    // single block is not refused here. (It cannot be run on this fixture's
    // three-level chain either, which is why only the guard is asserted.)
    std::vector<llama::BatchActivation> one;
    one.push_back(f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));
    try
    {
        f.op->attention_sequence(one, s.w, s.cfg, *f.galois, *f.relin);
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(std::string(e.what()).find("bidirectional"),
                  std::string::npos)
            << "one block must not be refused by the causality guard: "
            << e.what();
    }
}

// ======================================================================
// 5. The refresh: arithmetic that needs no GPU
// ======================================================================

TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshLevelArithmetic)
{
    // The library's own accounting: CtoS + taylor + StoC + 8.
    EXPECT_EQ(llama::Llama3Batch16Operator::refresh_levels(
                  heongpu::BootstrappingConfig(3, 3, 11)),
              25);
    EXPECT_EQ(llama::Llama3Batch16Operator::refresh_levels(
                  heongpu::BootstrappingConfig(4, 4, 6)),
              22);

    // A chain is the refresh, the stretch, and the one prime the next
    // bootstrap is handed. Nothing about the model shape enters -- the stretch
    // is a measurement, which is what depth_trace is for.
    const heongpu::BootstrappingConfig cfg(3, 3, 11);
    EXPECT_EQ(llama::Llama3Batch16Operator::chain_limbs_for(0, cfg), 26);
    EXPECT_EQ(llama::Llama3Batch16Operator::chain_limbs_for(8, cfg), 34);
    EXPECT_EQ(llama::Llama3Batch16Operator::chain_limbs_for(14, cfg), 40);
    EXPECT_THROW(llama::Llama3Batch16Operator::chain_limbs_for(-1, cfg),
                 std::invalid_argument);
}

TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshConfigCountsItsSeams)
{
    llama::Llama3Batch16Operator::Batch16RefreshConfig cfg;
    // Off by default, so every measurement taken before this existed
    // reproduces unchanged.
    EXPECT_FALSE(cfg.enabled);
    EXPECT_EQ(cfg.count(), 0);

    cfg.enabled = true;
    EXPECT_EQ(cfg.count(), 5); // entry off, the other five on
    cfg.entry = true;
    EXPECT_EQ(cfg.count(), 6);
    cfg.mid = false;
    cfg.feed_forward_hidden = false;
    EXPECT_EQ(cfg.count(), 4);
}

// ======================================================================
// 6. The refresh, on the GPU
// ======================================================================

namespace
{
    /// A fixture on a bootstrappable chain. q0 = 60 over 50-bit primes puts
    /// q0/scale at 2^10, which is the ratio the CKKS bootstrap here is built
    /// around and returns noise -- silently -- at any other. The chain is far
    /// outside any security level, deliberately: what is measured here is
    /// whether the ENCODING survives a refresh, and the encoding does not know
    /// what N's cap is.
    struct BootFixture
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 128;
        static constexpr int instances = 16;

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
        std::unique_ptr<Batch> op;
        std::unique_ptr<llama::Llama3Batch16Operator> nl;
        std::unique_ptr<heongpu::Galoiskey<S>> boot_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        BootFixture(const llama::Batch16Shape& shape, int limb_count)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limb_count - 1, 50);
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60, 60});
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
            op = std::make_unique<Batch>(context, *encoder, layout, scale);
            nl = std::make_unique<llama::Llama3Batch16Operator>(*op, shape);

            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);

            // The bootstrapping context is per HEArithmeticOperator instance,
            // so it has to be generated on the operator this module actually
            // calls -- not on a second one the test happens to own.
            op->arith().generate_bootstrapping_params(
                scale, heongpu::BootstrappingConfig(3, 3, 11),
                heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

            // ONE key, from the union. A shift-vector Galois key asked for an
            // index it does not hold is undefined behaviour rather than an
            // error, so two keys is the shape of a silent wrong answer.
            std::vector<int> shifts = nl->boot_rotation_indices();
            boot_key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*boot_key, *secret);
        }

        std::vector<std::vector<double>> random_batch(int rows, int cols,
                                                      uint64_t seed,
                                                      double amp = 1.0) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-amp, amp);
            std::vector<std::vector<double>> out(
                instances,
                std::vector<double>(static_cast<size_t>(rows) * cols));
            for (auto& m : out)
                for (auto& v : m)
                    v = dist(rng);
            return out;
        }
    };
} // namespace

// The union really is a union, sorted and deduplicated, and it really does
// contain both halves. Cheap to get wrong, and impossible to notice.
TEST(HEonGPU, CKKS_Llama3Batch16Block_BootRotationIndicesAreTheUnion)
{
    BootFixture f(SmallShape(), 30);

    const std::vector<int> mine = f.nl->rotation_indices();
    const std::vector<int> boot = f.op->arith().bootstrapping_key_indexs();
    const std::vector<int> all = f.nl->boot_rotation_indices();

    EXPECT_TRUE(std::is_sorted(all.begin(), all.end()));
    EXPECT_EQ(std::adjacent_find(all.begin(), all.end()), all.end())
        << "the union must be deduplicated";

    for (int i : mine)
        EXPECT_NE(std::find(all.begin(), all.end(), i), all.end())
            << "the union dropped a module index: " << i;
    for (int i : boot)
        EXPECT_NE(std::find(all.begin(), all.end(), i), all.end())
            << "the union dropped a bootstrapping index: " << i;
    EXPECT_GE(all.size(), std::max(mine.size(), boot.size()));
    std::cout << "module indices " << mine.size() << ", bootstrapping "
              << boot.size() << ", union " << all.size() << std::endl;
}

// The claim the whole batch-16 parameter set rests on, asserted through THIS
// module's own entry point and on a BatchActivation rather than on a bare
// ciphertext: a refresh carries the Kang matrix encryption, so the island
// needs no bridge in front of it and can exit at one limb.
TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshCarriesTheMatrixEncoding)
{
    BootFixture f(SmallShape(), 30);
    const int d = BootFixture::d;
    const int cols = 4;

    const auto want = f.random_batch(d, cols, 31337u, 1.0);
    llama::BatchActivation x =
        f.op->encrypt(want, d, cols, *f.encryptor, f.scale);

    // A bootstrap is handed a ciphertext with one prime left; that is the
    // point in a circuit where the chain has actually run out.
    const int chain = f.op->chain_limbs();
    for (auto& c : x.column)
        for (int i = 0; i < chain - 1; ++i)
            f.op->arith().mod_drop_inplace(c);
    ASSERT_EQ(x.column.front().depth(), chain - 1);

    f.nl->bootstrap(x, "test.refresh", *f.boot_key, *f.relin);

    EXPECT_LT(x.column.front().depth(), chain - 1)
        << "a refresh has to return levels";
    std::cout << "depth after refresh: " << x.column.front().depth() << " of "
              << chain << std::endl;

    const auto got = f.op->decrypt(x, *f.decryptor, f.scale);
    const double worst = worst_abs(want, got);
    std::cout << "matrix encryption through a refresh, worst error: " << worst
              << " (" << -std::log2(worst) << " bits)" << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    // The bootstrap is by a wide margin the least accurate thing in the
    // library: the sine standing in for the modular reduction holds around
    // 1e-3 while everything else lands near the CKKS noise floor.
    EXPECT_LT(worst, 1e-2);

    // And every column came back at ONE level and ONE scale, which is what
    // every reduction downstream silently assumes.
    for (const auto& c : x.column)
    {
        EXPECT_EQ(c.depth(), x.column.front().depth());
        EXPECT_DOUBLE_EQ(c.scale(), x.column.front().scale());
    }
}

// The level budget is not decoration: METHOD_II sizes its digit count and its
// RNS width off the current depth, so an unspent limb is carried by every key
// switch until the next refresh and then discarded.
TEST(HEonGPU, CKKS_Llama3Batch16Block_LevelBudgetIsHonoured)
{
    // The chain has to be long enough that the refresh hands back MORE than
    // the budget, or the budget is not exercised at all: a bootstrap returns
    // chain - refresh_levels() limbs, which at 30 is five and at 40 is fifteen.
    const int chain_wanted = 40;
    const int returned =
        chain_wanted - llama::Llama3Batch16Operator::refresh_levels(
                           heongpu::BootstrappingConfig(3, 3, 11));
    const int keep = 6;
    ASSERT_GT(returned, keep)
        << "the budget would be a no-op and the test would assert nothing";

    BootFixture f(SmallShape(), chain_wanted);
    const int d = BootFixture::d;
    const int cols = 2;

    const auto want = f.random_batch(d, cols, 24680u, 0.5);
    llama::BatchActivation x =
        f.op->encrypt(want, d, cols, *f.encryptor, f.scale);

    const int chain = f.op->chain_limbs();
    ASSERT_EQ(chain, chain_wanted);
    for (auto& c : x.column)
        for (int i = 0; i < chain - 1; ++i)
            f.op->arith().mod_drop_inplace(c);

    std::string saw;
    f.nl->level_budget = [&](const char* name)
    {
        saw = name;
        return keep;
    };
    f.nl->bootstrap(x, "test.budget", *f.boot_key, *f.relin);
    f.nl->level_budget = nullptr;

    EXPECT_EQ(saw, "test.budget") << "the hook is called with the seam's name";
    EXPECT_EQ(chain - x.column.front().depth(), keep)
        << "the budget is in LIMBS remaining, not in depth";
    for (const auto& c : x.column)
        EXPECT_EQ(c.depth(), x.column.front().depth());

    // The values survived the drop: a mod drop is free, and what it discards
    // was about to be thrown away.
    const auto got = f.op->decrypt(x, *f.decryptor, f.scale);
    EXPECT_LT(worst_abs(want, got), 1e-2);
}

// A refresh that quietly does not happen is the failure mode worth guarding,
// because it changes the level schedule the caller sized its chain for and
// then surfaces as an unrelated throw several layers away.
TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshWithoutTheKeyThrows)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape, 6);
    const int d = Fixture::d;

    llama::Llama3Batch16Operator::TransformerBlockWeights w;
    w.attention.query = random_weight(shape.d_model, d, 1u, 0.05);
    w.attention.key = random_weight(shape.d_model, d, 2u, 0.05);
    w.attention.value = random_weight(shape.d_model, d, 3u, 0.05);
    w.feed_forward.gate = random_weight(shape.d_model, shape.hidden, 4u, 0.05);
    w.feed_forward.up = random_weight(shape.d_model, shape.hidden, 5u, 0.05);
    w.feed_forward.down = random_weight(shape.hidden, shape.d_model, 6u, 0.05);

    llama::Llama3Batch16Operator::TransformerBlockConfig cfg;
    cfg.attention.in_channels = shape.d_model;
    cfg.attention.q_channels = d;
    cfg.attention.kv_channels = d;
    cfg.attention.softmax.bound = 4.0;
    cfg.refresh.enabled = true;

    const auto x = f.random_batch(d, shape.d_model, 1u, 0.5);
    llama::BatchActivation a =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);

    // Checked before any work rather than at the first seam, so the levels are
    // not already spent by the time it is reported.
    EXPECT_THROW(
        f.nl->transformer_block(a, w, cfg, *f.galois, *f.relin, nullptr),
        std::invalid_argument);

    llama::Llama3Batch16Operator::Batch16Sequence seq;
    seq.block.push_back(
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));
    EXPECT_THROW(
        f.nl->transformer_block(seq, w, cfg, *f.galois, *f.relin, nullptr),
        std::invalid_argument);

    // And the SwiGLU's own internal seam, which cannot be driven from outside
    // the sublayer at all.
    llama::Llama3Batch16Operator::FeedForwardConfig ff;
    ff.refresh_hidden = true;
    llama::BatchActivation b =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    EXPECT_THROW(
        f.nl->feed_forward(b, w.feed_forward, ff, *f.galois, *f.relin, nullptr),
        std::invalid_argument);
}

// ======================================================================
// 7. The sequence block driver
// ======================================================================

TEST(HEonGPU, CKKS_Llama3Batch16Block_SequenceValidationCatchesDrift)
{
    llama::Batch16Shape shape = SmallShape();
    Fixture f(shape, 6);
    const int d = Fixture::d;
    const int cols = shape.d_model;
    const auto x = f.random_batch(d, cols, 1234u, 0.5);

    llama::Llama3Batch16Operator::Batch16Sequence empty;
    EXPECT_THROW(f.nl->validate(empty, cols, "test"), std::invalid_argument);

    llama::Llama3Batch16Operator::Batch16Sequence ok;
    ok.block.push_back(f.op->encrypt(x, d, cols, *f.encryptor, f.scale));
    ok.block.push_back(f.op->encrypt(x, d, cols, *f.encryptor, f.scale));
    EXPECT_NO_THROW(f.nl->validate(ok, cols, "test"));
    EXPECT_EQ(ok.blocks(), 2);
    EXPECT_EQ(ok.columns(), cols);
    EXPECT_EQ(f.nl->sequence_tokens(ok.blocks()), 2 * d);
    EXPECT_THROW(f.nl->validate(ok, cols + 1, "test"), std::invalid_argument);

    // One block dropped a level. Attention forms ONE denominator across every
    // block, so this is a silently wrong SoftMax and not an error from the
    // library -- this check is the only thing between the two.
    llama::Llama3Batch16Operator::Batch16Sequence drifted;
    drifted.block.push_back(f.op->encrypt(x, d, cols, *f.encryptor, f.scale));
    drifted.block.push_back(f.op->encrypt(x, d, cols, *f.encryptor, f.scale));
    for (auto& c : drifted.block[1].column)
        f.op->arith().mod_drop_inplace(c);
    EXPECT_THROW(f.nl->validate(drifted, cols, "test"), std::invalid_argument);
}

namespace
{
    /// The whole-block configuration both driver tests use, at the cheap fit
    /// degrees -- these tests compare two code paths against each other, so
    /// the fits' own error cancels and only a difference in SCHEDULE shows.
    struct BlockSetup
    {
        llama::Llama3Batch16Operator::TransformerBlockWeights w;
        llama::Llama3Batch16Operator::TransformerBlockConfig cfg;
        int limbs = 0;
    };

    BlockSetup make_block(const llama::Batch16Shape& shape, uint64_t seed)
    {
        BlockSetup s;
        s.w.attention.query =
            random_weight(shape.d_model, shape.q_channels(), seed, 0.05);
        s.w.attention.key =
            random_weight(shape.d_model, shape.kv_channels(), seed + 1, 0.05);
        s.w.attention.value =
            random_weight(shape.d_model, shape.kv_channels(), seed + 2, 0.4);
        s.w.attention.output =
            random_weight(shape.q_channels(), shape.d_model, seed + 3, 0.05);
        s.w.feed_forward.gate =
            random_weight(shape.d_model, shape.hidden, seed + 4, 0.05);
        s.w.feed_forward.up =
            random_weight(shape.d_model, shape.hidden, seed + 5, 0.05);
        s.w.feed_forward.down =
            random_weight(shape.hidden, shape.d_model, seed + 6, 0.05);

        s.cfg.attention.in_channels = shape.d_model;
        s.cfg.attention.q_channels = shape.q_channels();
        s.cfg.attention.kv_channels = shape.kv_channels();
        s.cfg.attention.heads = shape.heads;
        s.cfg.attention.kv_heads = shape.kv_heads;
        s.cfg.attention.causal = true;
        cheap_softmax(s.cfg.attention.softmax, s.cfg.attention.seam, 4.0);
        s.cfg.attention.softmax.sum_lo = 1e-2;
        s.cfg.attention.softmax.sum_hi = 6.0e2;

        s.cfg.attention_norm.degree = 15;
        s.cfg.attention_norm.newton_iterations = 0;
        s.cfg.attention_norm.sum_lo = 1e-3;
        s.cfg.attention_norm.sum_hi = 3.0e1;
        s.cfg.feed_forward_norm = s.cfg.attention_norm;

        s.cfg.feed_forward.silu_bound = 4.0;
        s.cfg.feed_forward.silu_degree = 15;

        // norm 9 (7 + two crossings), attention 5 + seam, residual 1, norm 9,
        // SwiGLU 9, residual 1 -- and one prime left over at the end.
        s.limbs = 9 + 5 +
                  Batch::softmax_seam_levels(s.cfg.attention.softmax,
                                             s.cfg.attention.seam) +
                  1 + 9 + 9 + 1 + 2;
        return s;
    }
} // namespace

// The sequence driver at one block has to BE the single-block driver. If this
// drifts, the two entry points are two models.
// ======================================================================
// 8. The narrow auxiliary track, which fill_slot_config had nailed shut
// ======================================================================

// RMSNormConfig::refresh_sum was hard-coded false at the boundary between this
// module and the slot core, so the slot core's own implementation was
// unreachable from here. This is the encoding where it is cheapest: the
// channel axis IS the ciphertext index, so the reduction lands in exactly ONE
// ciphertext however wide the model is, and refreshing it is a single
// bootstrap that moves the entire 1/sqrt fit off the d_model-wide residual
// track.
//
// The saving is CONDITIONAL and the condition is worth stating, because it is
// the opposite of the intuition. A bootstrap always returns its ciphertext to
// depth refresh_levels, so refreshing the sum at depth D buys D - refresh_levels
// levels: it is a GAIN on a stream that is already deep and a LOSS on a fresh
// one. A norm at the top of a chain should leave this off. This test therefore
// runs the norm where a block actually runs it -- deep -- and asserts both
// directions, so neither is mistaken for the other.
TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshSumMovesTheFitOffTheWideTrack)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 8;
    shape.hidden = 16;

    const heongpu::BootstrappingConfig boot(3, 3, 11);
    const int refresh = llama::Llama3Batch16Operator::refresh_levels(boot);
    const int limbs = 40;
    BootFixture f(shape, limbs);

    const int d = BootFixture::d;
    const auto x = f.random_batch(d, shape.d_model, 606u, 0.5);
    const std::vector<double> no_gain;

    llama::Llama3Batch16Operator::RMSNormConfig plain_cfg;
    plain_cfg.degree = 15;
    plain_cfg.newton_iterations = 0;
    // Calibrated, not bounded. A fit over a range the data does not visit is
    // wrong by more than any level saving is worth, and it would be charged
    // to the auxiliary track here because that is what this test varies.
    bracket_sum(x, d, shape.d_model, plain_cfg.sum_lo, plain_cfg.sum_hi);

    llama::Llama3Batch16Operator::RMSNormConfig aux_cfg = plain_cfg;
    aux_cfg.refresh_sum = true;

    const auto want = rms_norm_host(x, d, shape.d_model, plain_cfg.eps);

    // Where a block's second norm actually sits: deep enough that the fit's
    // levels are the scarce thing. Five above the refresh floor, so the
    // bootstrap has something to give back.
    const int deep = refresh + 5;

    auto run = [&](const llama::Llama3Batch16Operator::RMSNormConfig& cfg,
                   uint64_t seed, bool with_key, int start_depth)
    {
        llama::BatchActivation a =
            f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
        for (auto& c : a.column)
        {
            f.op->arith().drop_to_depth(c, start_depth);
        }
        (void) seed;
        return f.nl->rms_norm(a, no_gain, cfg, *f.boot_key, *f.relin,
                              with_key ? f.boot_key.get() : nullptr);
    };

    llama::BatchActivation wide = run(plain_cfg, 1u, false, deep);
    const int wide_depth = wide.column[0].depth();
    const auto wide_out = f.op->decrypt(wide, *f.decryptor, f.scale);

    llama::BatchActivation aux = run(aux_cfg, 2u, true, deep);
    const int aux_depth = aux.column[0].depth();
    const auto aux_out = f.op->decrypt(aux, *f.decryptor, f.scale);

    std::cout << "rms_norm from depth " << deep << " of " << limbs
              << ": wide track ends at " << wide_depth
              << ", auxiliary track at " << aux_depth << " -- "
              << wide_depth - aux_depth << " levels back" << std::endl;

    // The saving, which is the whole reason to pay a bootstrap.
    EXPECT_LT(aux_depth, wide_depth)
        << "refresh_sum did not move the fit off the wide track";

    // Both judged against the HOST, not against each other, so that the
    // trade is visible instead of one circuit being scored on the other's
    // noise.
    //
    // MEASURED, and it refutes the reason one might reach for this: the
    // auxiliary track is NOT more accurate. 6.4e-04 for the wide track
    // against 9.1e-03 for the auxiliary one at this shape -- about 14x, near
    // enough a decimal digit, and it is the bootstrap's own precision that
    // pays for it. The fit does not lose accuracy by running on the last limb
    // of the chain; a v1 refresh loses more than that by running at all.
    //
    // So refresh_sum is a LEVELS-for-PRECISION trade and nothing else. Worth
    // it where levels are the binding constraint, which on this path is the
    // whole point -- see the security section -- but it must not be sold as
    // free, and it is why this stays a flag with a default of false.
    const double magnitude = mean_abs(want);
    const double wide_err = worst_abs(want, wide_out);
    const double aux_err = worst_abs(want, aux_out);
    std::cout << "against the host: wide track " << wide_err
              << ", auxiliary track " << aux_err << " on a mean magnitude of "
              << magnitude << " (auxiliary costs " << aux_err / wide_err
              << "x)" << std::endl;
    ASSERT_FALSE(std::isnan(aux_err));

    // The contract: still accurate enough to be useful.
    EXPECT_LT(aux_err, 1e-4 + 3e-2 * magnitude);
    // And a bound on the trade, so that a real regression -- a wrong level, a
    // dropped rescale -- still trips even though the measured cost is a digit.
    EXPECT_LT(aux_err, 50.0 * wide_err + 1e-4)
        << "the auxiliary track costs about 14x here; far more than that is a "
           "defect rather than bootstrap precision";

    // And the other direction, which is why this is a flag and not a default:
    // on a FRESH stream the refresh costs levels instead of returning them,
    // because a bootstrap lands at refresh_levels however shallow its input.
    llama::BatchActivation fresh_wide = run(plain_cfg, 3u, false, 0);
    llama::BatchActivation fresh_aux = run(aux_cfg, 4u, true, 0);
    std::cout << "from a FRESH stream: wide " << fresh_wide.column[0].depth()
              << ", auxiliary " << fresh_aux.column[0].depth() << std::endl;
    EXPECT_GT(fresh_aux.column[0].depth(), fresh_wide.column[0].depth())
        << "if a refresh were free on a fresh stream it should be the default";
}

// The two ways to get refresh_sum silently wrong, both refused.
TEST(HEonGPU, CKKS_Llama3Batch16Block_RefreshSumRefusesItsBadCombinations)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    Fixture f(shape, 12);

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 707u, 0.5);
    const std::vector<double> no_gain;

    llama::Llama3Batch16Operator::RMSNormConfig cfg;
    cfg.degree = 15;
    cfg.newton_iterations = 0;
    cfg.sum_lo = 1e-3;
    cfg.sum_hi = 6.0e1;
    cfg.refresh_sum = true;

    // No boot key: the slot core throws rather than quietly not refreshing.
    llama::BatchActivation a =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    EXPECT_THROW(f.nl->rms_norm(a, no_gain, cfg, *f.galois, *f.relin),
                 std::invalid_argument);

    // sum_pre_scaled runs its own circuit and never reaches the slot core, so
    // refresh_sum there would be accepted and do nothing. Refused up front.
    llama::Llama3Batch16Operator::RMSNormConfig both = cfg;
    both.sum_pre_scaled = true;
    llama::BatchActivation b =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    EXPECT_THROW(f.nl->rms_norm(b, no_gain, both, *f.galois, *f.relin,
                                f.galois.get()),
                 std::invalid_argument);

    // A Newton step refines against the unmapped argument.
    llama::Llama3Batch16Operator::RMSNormConfig newton = cfg;
    newton.newton_iterations = 1;
    llama::BatchActivation c =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    EXPECT_THROW(f.nl->rms_norm(c, no_gain, newton, *f.galois, *f.relin,
                                f.galois.get()),
                 std::invalid_argument);
}

// ======================================================================
// 9. The hoisting lever, which this path had no way to pull
// ======================================================================

// set_hoisted_crossings shares one key-switch decomposition across the 15
// baby shifts instead of paying 15, so a bridged column costs 8 decompositions
// instead of 22. Llama3RectOperator has exposed it all along; here it was
// reachable only inside use_fast_bridge(), which has no callers, so every
// batch-16 measurement to date ran with the norm and SwiGLU bridges unhoisted
// while the SoftMax seam hoisted itself through its own config.
//
// The lever is only worth having if it changes nothing, and "nothing" has to
// be asserted rather than assumed: hoisting reassociates a sum of products, so
// the two paths differ in the last bits and must not differ in more than that.
// A whole block is the right unit here because the seam restores the flag
// afterwards, so a narrower test would measure the flag the seam sets rather
// than the one the caller does.
TEST(HEonGPU, CKKS_Llama3Batch16Block_HoistingTheBridgeChangesNoAnswer)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    BlockSetup s = make_block(shape, 8191u);
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 31337u, 0.5);

    // Two special primes, so the context is METHOD_II and there IS a
    // decomposition to share. Under METHOD_I hoisting is a no-op and this
    // test would pass by testing nothing, which is worth ruling out.
    const int specials = f.context->get_key_modulus_count() -
                         f.context->get_ciphertext_modulus_count();
    ASSERT_GT(specials, 1) << "hoisting needs KEYSWITCHING_METHOD_II";

    ASSERT_FALSE(f.nl->hoisted_crossings())
        << "the operator default is off, and the whole point of this test is "
           "that it did not have to be";

    llama::BatchActivation a0 =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    llama::BatchActivation plain =
        f.nl->transformer_block(a0, s.w, s.cfg, *f.galois, *f.relin);
    const auto plain_out = f.op->decrypt(plain, *f.decryptor, f.scale);

    f.nl->set_hoisted_crossings(true);
    ASSERT_TRUE(f.nl->hoisted_crossings());
    // The forwarding must reach the operator that owns the bridge, not stop
    // at this one.
    ASSERT_TRUE(f.op->hoisted_crossings());

    llama::BatchActivation a1 =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    llama::BatchActivation hoisted =
        f.nl->transformer_block(a1, s.w, s.cfg, *f.galois, *f.relin);
    const auto hoisted_out = f.op->decrypt(hoisted, *f.decryptor, f.scale);

    const double worst = worst_abs(plain_out, hoisted_out);
    const double magnitude = mean_abs(plain_out);
    std::cout << "hoisted vs unhoisted, whole block: worst " << worst
              << " on a mean magnitude of " << magnitude << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    // Reassociation only. Judged on the ratio, because an absolute bound says
    // nothing across shapes.
    EXPECT_LT(worst, 1e-6 + 1e-4 * magnitude);

    // And it goes back off, so a caller can A/B without rebuilding anything.
    f.nl->set_hoisted_crossings(false);
    EXPECT_FALSE(f.nl->hoisted_crossings());
    EXPECT_FALSE(f.op->hoisted_crossings());

    // use_fast_bridge reaches the same flag, which is what made the lever
    // look present when it was not separately reachable.
    f.nl->use_fast_bridge();
    EXPECT_TRUE(f.nl->hoisted_crossings());
}

TEST(HEonGPU, CKKS_Llama3Batch16Block_SequenceBlockAtOneBlockMatchesTheBlock)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    BlockSetup s = make_block(shape, 51u);
    std::cout << "whole-block chain: " << s.limbs << " limbs" << std::endl;
    Fixture f(shape, s.limbs);

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 2468u, 0.5);

    std::map<std::string, int> single_depths;
    f.nl->depth_trace = [&](const char* name, int depth)
    { single_depths[name] = depth; };

    llama::BatchActivation a =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    llama::BatchActivation single =
        f.nl->transformer_block(a, s.w, s.cfg, *f.galois, *f.relin);
    const auto single_out = f.op->decrypt(single, *f.decryptor, f.scale);

    std::map<std::string, int> seq_depths;
    f.nl->depth_trace = [&](const char* name, int depth)
    { seq_depths[name] = depth; };

    llama::Llama3Batch16Operator::Batch16Sequence seq;
    seq.block.push_back(
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale));
    llama::Llama3Batch16Operator::Batch16Sequence blocked =
        f.nl->transformer_block(seq, s.w, s.cfg, *f.galois, *f.relin);
    f.nl->depth_trace = nullptr;

    ASSERT_EQ(blocked.blocks(), 1);
    const auto blocked_out =
        f.op->decrypt(blocked.block[0], *f.decryptor, f.scale);

    // The same seams at the same depths. The two drivers name their seams
    // identically on purpose, so a schedule that has drifted shows up here
    // rather than as a number that is merely a bit different.
    EXPECT_EQ(single_depths, seq_depths);
    for (const auto& entry : single_depths)
        std::cout << "  seam " << entry.first << " at depth " << entry.second
                  << std::endl;

    const double worst = worst_abs(single_out, blocked_out);
    const double magnitude = mean_abs(single_out);
    std::cout << "one block, sequence driver vs block driver: worst " << worst
              << " on a mean magnitude of " << magnitude << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    EXPECT_LT(worst, 1e-5 + 1e-3 * magnitude);
}

// A whole block over TWO token blocks with a refresh at one seam: everything
// the two gaps added, running together, at the real batch-16 ring.
TEST(HEonGPU, CKKS_Llama3Batch16Block_TwoTokenBlocksWithARefreshRun)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    BlockSetup s = make_block(shape, 61u);

    // ONE seam, and the one in the middle: this test is about the seams
    // composing, not about the schedule being optimal. All six over two token
    // blocks would be twelve times d_model bootstraps for no extra
    // information.
    s.cfg.refresh.enabled = true;
    s.cfg.refresh.entry = false;
    s.cfg.refresh.after_attention_norm = false;
    s.cfg.refresh.after_attention = false;
    s.cfg.refresh.mid = true;
    s.cfg.refresh.after_feed_forward_norm = false;
    s.cfg.refresh.feed_forward_hidden = false;
    ASSERT_EQ(s.cfg.refresh.count(), 1);

    // The chain has to carry the DEEPEST STRETCH plus the refresh, not the
    // whole block: entry to the mid seam, then the mid seam to the end. The
    // attention half is the deeper of the two.
    const int attention_half =
        9 + 5 +
        Batch::softmax_seam_levels(s.cfg.attention.softmax,
                                   s.cfg.attention.seam) +
        1;
    const int limbs = llama::Llama3Batch16Operator::chain_limbs_for(
        attention_half, heongpu::BootstrappingConfig(3, 3, 11));
    std::cout << "refreshed block: attention half " << attention_half
              << " levels, chain " << limbs << " limbs" << std::endl;
    BootFixture f(shape, limbs);

    const int d = BootFixture::d;
    const auto x0 = f.random_batch(d, shape.d_model, 1111u, 0.5);
    const auto x1 = f.random_batch(d, shape.d_model, 2222u, 0.5);

    std::map<std::string, int> depths;
    f.nl->depth_trace = [&](const char* name, int depth)
    { depths[name] = depth; };

    llama::Llama3Batch16Operator::Batch16Sequence seq;
    seq.block.push_back(
        f.op->encrypt(x0, d, shape.d_model, *f.encryptor, f.scale));
    seq.block.push_back(
        f.op->encrypt(x1, d, shape.d_model, *f.encryptor, f.scale));

    // ONE key for both roles, which is the whole point of the union: the
    // sublayers' shifts and the bootstrap's are in the same Galoiskey, so
    // there is no second key for a caller to hand over by mistake.
    llama::Llama3Batch16Operator::Batch16Sequence out;
    ASSERT_NO_THROW({
        out = f.nl->transformer_block(seq, s.w, s.cfg, *f.boot_key, *f.relin,
                                      f.boot_key.get());
    });
    f.nl->depth_trace = nullptr;

    ASSERT_EQ(out.blocks(), 2);
    ASSERT_NE(depths.find("block.mid"), depths.end());
    ASSERT_NE(depths.find("block.out"), depths.end());

    std::cout << "two token blocks, one refresh -- block.mid left at depth "
              << depths["block.mid"] << ", block.out at " << depths["block.out"]
              << " of " << f.op->chain_limbs() << std::endl;
    // The refresh returned levels: the stream is shallower leaving the mid
    // seam than the attention half alone would have left it.
    EXPECT_LT(depths["block.mid"], attention_half + 5)
        << "the mid seam does not look like it refreshed anything";

    // Finite, bounded and per-instance separate. A precise reference is not
    // the point -- the SoftMax and SiLU fits carry their own error and the
    // earlier tests pin those; what is under test here is that the refresh and
    // the token blocking compose without destroying the stream.
    for (int t = 0; t < 2; ++t)
    {
        const auto got = f.op->decrypt(out.block[t], *f.decryptor, f.scale);
        ASSERT_EQ(got.size(), static_cast<size_t>(BootFixture::instances));
        for (const auto& m : got)
            for (double v : m)
            {
                ASSERT_FALSE(std::isnan(v)) << "token block " << t;
                ASSERT_LT(std::abs(v), 1e3) << "token block " << t;
            }
    }
}

// ======================================================================
// 8. Which seams are load-bearing
// ======================================================================

// The refresh schedule was shipped with six named seams and no evidence about
// WHICH of them a batch-16 block actually needs. This measures it, and the
// method is the only honest one available: run the block ONCE with the refresh
// off, record the depth at every seam point in execution order, and read the
// stretches off. That is valid because a stage's level spend is a property of
// its operations and not of the depth it starts at -- which is exactly why a
// bootstrap can be moved without changing anything else.
//
// Then the search is host arithmetic over the 32 subsets of the five optional
// seams, and it costs nothing. What it reports is the chain each schedule
// implies, which is the number a caller has to pick before it can run at all.
TEST(HEonGPU, CKKS_Llama3Batch16Block_WhichRefreshSeamsAreLoadBearing)
{
    llama::Batch16Shape shape = SmallShape();
    shape.d_model = 4;
    shape.hidden = 8;
    BlockSetup s = make_block(shape, 71u);
    Fixture f(shape, s.limbs);

    // Every seam point of the block, in the order the circuit reaches them.
    // `feed_forward.hidden` is the SwiGLU's internal seam, which cannot be
    // driven from outside the sublayer and is therefore a flag rather than a
    // list entry -- but it splits a stretch exactly as the others do.
    const std::vector<std::string> points{
        "block.entry",          "block.after_attention_norm",
        "block.after_attention", "block.mid",
        "block.after_feed_forward_norm", "feed_forward.hidden",
        "block.out"};

    std::vector<std::pair<std::string, int>> trace;
    f.nl->depth_trace = [&](const char* name, int depth)
    { trace.emplace_back(name, depth); };

    const int d = Fixture::d;
    const auto x = f.random_batch(d, shape.d_model, 13579u, 0.5);
    llama::BatchActivation a =
        f.op->encrypt(x, d, shape.d_model, *f.encryptor, f.scale);
    llama::BatchActivation out =
        f.nl->transformer_block(a, s.w, s.cfg, *f.galois, *f.relin);
    f.nl->depth_trace = nullptr;
    ASSERT_FALSE(out.column.empty());

    // First occurrence of each point, in trace order.
    std::vector<int> depth(points.size(), -1);
    for (const auto& entry : trace)
        for (size_t i = 0; i < points.size(); ++i)
            if (depth[i] < 0 && entry.first == points[i])
                depth[i] = entry.second;
    for (size_t i = 0; i < points.size(); ++i)
        ASSERT_GE(depth[i], 0) << "the block never reached seam " << points[i];
    for (size_t i = 1; i < points.size(); ++i)
        ASSERT_GT(depth[i], depth[i - 1])
            << "seam " << points[i] << " must come after " << points[i - 1];

    std::cout << "seam depths:";
    for (size_t i = 0; i < points.size(); ++i)
        std::cout << "  " << points[i] << "=" << depth[i];
    std::cout << std::endl;

    const heongpu::BootstrappingConfig boot(3, 3, 11);
    // The five optional seams are the interior points; entry and out are the
    // block's boundaries and are not a choice.
    const int optional = static_cast<int>(points.size()) - 2;

    auto worst_stretch = [&](unsigned mask)
    {
        int worst = 0;
        int last = depth.front();
        for (int i = 1; i <= optional; ++i)
            if (mask & (1u << (i - 1)))
            {
                worst = std::max(worst, depth[static_cast<size_t>(i)] - last);
                last = depth[static_cast<size_t>(i)];
            }
        return std::max(worst, depth.back() - last);
    };

    int best_chain = 1 << 30;
    for (unsigned mask = 0; mask < (1u << optional); ++mask)
        best_chain = std::min(
            best_chain,
            llama::Llama3Batch16Operator::chain_limbs_for(worst_stretch(mask),
                                                          boot));

    // The cheapest schedule that reaches that chain: fewest refreshes, because
    // every one of them is a bootstrap per column and the chain is the same.
    unsigned cheapest = (1u << optional) - 1;
    int cheapest_count = optional;
    for (unsigned mask = 0; mask < (1u << optional); ++mask)
    {
        const int chain =
            llama::Llama3Batch16Operator::chain_limbs_for(worst_stretch(mask),
                                                          boot);
        int count = 0;
        for (int i = 0; i < optional; ++i)
            count += (mask >> i) & 1;
        if (chain == best_chain && count < cheapest_count)
        {
            cheapest_count = count;
            cheapest = mask;
        }
    }

    const int all_on = (1u << optional) - 1;
    std::cout << "all " << optional << " seams: chain "
              << llama::Llama3Batch16Operator::chain_limbs_for(
                     worst_stretch(static_cast<unsigned>(all_on)), boot)
              << " limbs, worst stretch "
              << worst_stretch(static_cast<unsigned>(all_on)) << std::endl;
    std::cout << "minimum chain " << best_chain << " limbs, reached with "
              << cheapest_count << " seam(s):";
    for (int i = 0; i < optional; ++i)
        if (cheapest & (1u << i))
            std::cout << " " << points[static_cast<size_t>(i + 1)];
    std::cout << std::endl;

    // Turning every seam on cannot beat the best schedule -- a refresh never
    // lengthens a stretch -- so all-on is one of the minimisers.
    EXPECT_EQ(llama::Llama3Batch16Operator::chain_limbs_for(
                  worst_stretch(static_cast<unsigned>(all_on)), boot),
              best_chain);

    // THE FINDING, and it is what makes this test worth its runtime: some of
    // the six are redundant. The attention sublayer is by far the deepest
    // stretch on this path, so seams that split a stretch already shorter than
    // it buy nothing and cost a bootstrap per column of the stream.
    EXPECT_LT(cheapest_count, optional)
        << "if this ever fails, every named seam has become load-bearing and "
           "the schedule is no longer over-provisioned";

    // And the binding stretch really is the attention sublayer's, which is
    // where any further saving has to come from.
    const int attention_stretch = depth[2] - depth[1];
    std::cout << "the binding stretch is the attention sublayer at "
              << attention_stretch << " levels" << std::endl;
    EXPECT_EQ(best_chain, llama::Llama3Batch16Operator::chain_limbs_for(
                              attention_stretch, boot))
        << "the chain is set by the attention sublayer, so a shorter chain "
           "needs a seam INSIDE it -- the SoftMax's refresh_denominator is "
           "the one that exists";
}
