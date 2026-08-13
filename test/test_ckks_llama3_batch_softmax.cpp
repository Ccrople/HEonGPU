// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The SoftMax seam of the 16-batched-input path, at the shape it is meant for.
//
// WHY THIS FILE IS AT d = 128 AND NOT AT THE d = 8 THE OTHER SUITE USES
// --------------------------------------------------------------------
// "Sixteen batched inputs" is not a runtime choice; it is the ring. The batch
// is k/2 = N/(2d), so sixteen inputs at N = 4096 pins d = 128 -- which is also
// Llama-3-8B's head_dim, and also the token block. The existing batch suite
// runs d = 8, i.e. k = 512 and 256 inputs: the opposite corner of the design
// space, where the BSGS split is 4 rather than 16 and a crossing does 64
// plaintext products rather than 16,384. Nothing in the tree exercised the
// corner the branch is named after, so these tests do.
//
// WHAT THEY PIN
// -------------
// 1. That the layout is FORCED. d * (k/2) == N/2 exactly, so the slot vector
//    is full of (input, query) and the key axis has nowhere to go but the
//    ciphertext axis -- which is the zero-rotation one.
// 2. That the seam is numerically the SoftMax, at that shape.
// 3. That the level ledger softmax_seam_levels() advertises is the ledger the
//    circuit actually spends. A closed form nobody checks is a comment.
// 4. That the two folds and the auxiliary track cost levels and not accuracy.
// 5. That the mask cache and the hoisted crossings change the work and not the
//    answer.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;

    using Batch = llama::Llama3BatchOperator;
    using SeamConfig = Batch::BatchSoftmaxSeamConfig;
    using SoftmaxConfig = llama::Llama3Operator::SoftmaxConfig;

    /// N = 4096, d = 128 => k = 32, batch = k/2 = 16 independent inputs.
    /// That is the whole point of the branch, so it is a static assertion of
    /// the fixture rather than a parameter.
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
        std::unique_ptr<Batch> op;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;

        explicit Fixture(int limb_count)
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limb_count - 1, 40);
            context->set_poly_modulus_degree(
                static_cast<size_t>(degree));
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
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
            op = std::make_unique<Batch>(context, *encoder, layout, scale);

            // Not const: the Galoiskey constructor takes its shift list by
            // non-const reference.
            std::vector<int> shifts = op->rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        int batch() const { return layout.batch; }
    };

    /// One d x d score block per input, drawn on [lo, hi]. The SoftMax wants
    /// its argument in [-bound, 0] AFTER the shift, so the caller picks the
    /// interval and the shift together, exactly as calibration does.
    std::vector<std::vector<double>> random_scores(int batch, int d, double lo,
                                                   double hi, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        std::vector<std::vector<double>> out(
            batch, std::vector<double>(static_cast<size_t>(d) * d));
        for (auto& block : out)
        {
            for (auto& v : block)
            {
                v = dist(rng);
            }
        }
        return out;
    }

    /// The true causal SoftMax of the scores, row by row. This is the answer
    /// the circuit is approximating, not the iteration it runs.
    std::vector<std::vector<double>>
    host_causal_softmax(const std::vector<std::vector<double>>& scores, int d,
                        double shift)
    {
        std::vector<std::vector<double>> out(scores.size());
        for (size_t b = 0; b < scores.size(); ++b)
        {
            out[b].assign(static_cast<size_t>(d) * d, 0.0);
            for (int u = 0; u < d; ++u)
            {
                double total = 0.0;
                for (int j = 0; j <= u; ++j)
                {
                    const double e = std::exp(
                        scores[b][static_cast<size_t>(u) * d + j] - shift);
                    out[b][static_cast<size_t>(u) * d + j] = e;
                    total += e;
                }
                for (int j = 0; j <= u; ++j)
                {
                    out[b][static_cast<size_t>(u) * d + j] /= total;
                }
            }
        }
        return out;
    }

    double max_abs_diff(const std::vector<std::vector<double>>& a,
                        const std::vector<std::vector<double>>& b)
    {
        double worst = 0.0;
        for (size_t s = 0; s < a.size(); ++s)
        {
            for (size_t e = 0; e < a[s].size(); ++e)
            {
                worst = std::max(worst, std::abs(a[s][e] - b[s][e]));
            }
        }
        return worst;
    }

    /// The calibrated range of the round-zero denominator, taken off the host
    /// scores the way Section 4.3 takes it off a calibration run. Without this
    /// the reciprocal is fitted over the worst case the paper is explicitly
    /// telling us not to fit, and no degree a circuit can afford will carry
    /// it.
    void calibrate(const std::vector<std::vector<double>>& scores, int d,
                   double shift, int iterations, SoftmaxConfig& config)
    {
        const double divisor = std::pow(2.0, iterations);
        double lo = 1e300;
        double hi = 0.0;
        // Round zero's denominator, and how concentrated a row is after it.
        double sharpest = 0.0;
        for (const auto& block : scores)
        {
            for (int u = 0; u < d; ++u)
            {
                // The mask weight the seam applies rides on every kept
                // coordinate, and the denominator is the sum of SQUARES.
                const double w =
                    static_cast<double>(d) / static_cast<double>(u + 1);
                double total = 0.0;
                double squares = 0.0;
                for (int j = 0; j <= u; ++j)
                {
                    const double e = std::exp(
                        (block[static_cast<size_t>(u) * d + j] - shift) /
                        divisor);
                    total += w * e * e;
                    squares += e * e;
                }
                lo = std::min(lo, total);
                hi = std::max(hi, total);

                // A round leaves y_j = e_j^2 / sum e^2, which sums to one, so
                // every round after the first sees a denominator in [1/d, 1]
                // and what calibration supplies is how far up that the
                // sharpest row actually reaches. The mask weight cancels
                // here, which is exactly what it is for.
                double after = 0.0;
                for (int j = 0; j <= u; ++j)
                {
                    const double e = std::exp(
                        (block[static_cast<size_t>(u) * d + j] - shift) /
                        divisor);
                    const double y = e * e / squares;
                    after += y * y;
                }
                sharpest = std::max(sharpest, after);
            }
        }
        // A little room either side, so the fit is not evaluated at its own
        // endpoints.
        config.sum_lo = lo * 0.95;
        config.sum_hi = hi * 1.05;
        // As a multiple of the uniform value 1/d. Without this the later
        // rounds are fitted over [0.5/d, 1.5], a range of 3d, and a degree-15
        // reciprocal with no Newton step cannot carry it -- which is the
        // whole of Section 4.3's argument, and the reason the folded
        // configuration is safe at all.
        config.concentration =
            std::min(static_cast<double>(d),
                     sharpest * static_cast<double>(d) * 1.05);
    }

    /// Levels a degree-D Chebyshev evaluation spends, matching
    /// softmax_seam_levels' own recursion.
    int fit_levels(int degree)
    {
        int levels = 0;
        for (int reach = 1; reach < degree; reach <<= 1)
        {
            levels++;
        }
        return levels;
    }

    /// The seam's production settings: both folds on, no Newton step, ranges
    /// calibrated, and a degree-63 reciprocal. This is the configuration the
    /// branch exists to make available.
    ///
    /// The degree is not a free parameter and it is the thing this file
    /// measured rather than assumed. Taking fold_affine_into_mask forces
    /// inverse_newton = 0, and a causal triangle that starts at u = 0 has a
    /// row attending to ONE key, whose sum of squares after a round is
    /// exactly 1 -- so `concentration` is d whatever the calibration, the
    /// later rounds are fitted over [0.5/d, 1.5], and that range is 384:1 at
    /// d = 128. Degree 15 over that is 33% wrong, which is what the first run
    /// of this suite measured. Degree 63 is what the rect path uses in
    /// production and it is what this needs, for the same reason.
    void production(SoftmaxConfig& softmax, SeamConfig& seam, double bound,
                    double shift)
    {
        softmax.bound = bound;
        softmax.iterations = 2;
        softmax.exp_degree = 15;
        softmax.inverse_degree = 63;
        softmax.inverse_newton = 0;

        seam.causal = true;
        seam.score_shift = shift;
        seam.scores_carry_exp_domain = true;
        seam.fold_affine_into_mask = true;
        seam.hoisted_crossings = true;
        seam.cache_masks = true;
    }

    /// Encrypt the scores the seam is to be handed. When the seam is told the
    /// scores carry the exponential's domain map, they have to actually carry
    /// it -- here on the host, where attention() would carry it on the query
    /// weight.
    llama::BatchActivation
    encrypt_scores(Fixture& f, const std::vector<std::vector<double>>& scores,
                   const SeamConfig& seam, double bound)
    {
        std::vector<std::vector<double>> staged = scores;
        if (seam.scores_carry_exp_domain)
        {
            const double map = Batch::exp_domain_scale(bound);
            for (auto& block : staged)
            {
                for (auto& v : block)
                {
                    v *= map;
                }
            }
        }
        return f.op->encrypt(staged, Fixture::d, Fixture::d, *f.encryptor,
                             f.scale);
    }
} // namespace

// ---------------------------------------------------------------------------
// 1. The layout is forced, and the forced one is free
// ---------------------------------------------------------------------------

// The load-bearing claim of the whole seam. It is arithmetic, not a tuning
// choice: the slot vector is exactly d * (k/2) wide and is entirely spent on
// (input, query), so the key axis cannot be a slot axis. Being the ciphertext
// axis is then what makes the reduction free.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_LayoutIsForcedAndTheReductionIsFree)
{
    Fixture f(6);

    ASSERT_EQ(f.layout.k, 32);
    ASSERT_EQ(f.batch(), 16) << "sixteen batched inputs is the branch";
    ASSERT_EQ(f.layout.d, 128) << "and it pins d = N/32, which is head_dim";

    const int slots = f.encoder->slot_count();
    // The reason there is no room: every slot already carries an (input,
    // query) pair.
    EXPECT_EQ(f.layout.d * f.batch(), slots);

    const auto shape = f.op->softmax_layout();
    EXPECT_EQ(shape.parts, Fixture::d);
    EXPECT_EQ(shape.count, 1);
    EXPECT_EQ(shape.stride, slots);
    EXPECT_TRUE(shape.strided);

    // The whole point.
    EXPECT_EQ(shape.reduction_rotations, 0);
    EXPECT_EQ(shape.new_galois_indices, 0);

    // And the crossings, which are what the seam does cost: two bridges of d
    // columns at n1 + n2 - 2 rotations each.
    const int n1 = f.op->baby_steps();
    ASSERT_GT(n1, 0);
    EXPECT_EQ(shape.crossing_rotations,
              2 * Fixture::d * (n1 + Fixture::d / n1 - 2));

    // Nothing the seam asks for is a key Algorithm 4 did not already force.
    std::vector<int> bridge = f.op->bridge_rotation_indices();
    std::vector<int> product = f.op->product_rotation_indices();
    std::sort(bridge.begin(), bridge.end());
    std::sort(product.begin(), product.end());
    EXPECT_EQ(bridge, product);
}

// ---------------------------------------------------------------------------
// 2. It is the SoftMax
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3BatchSoftmax_SeamMatchesHostCausalSoftmax)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 20260813u);
    const auto want = host_causal_softmax(scores, Fixture::d, shift);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, shift);
    calibrate(scores, Fixture::d, shift, softmax.iterations, softmax);

    Fixture f(Batch::softmax_seam_levels(softmax, seam) + 3);
    auto ct = encrypt_scores(f, scores, seam, bound);

    auto p = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    ASSERT_EQ(p.columns(), Fixture::d);

    const auto got =
        f.op->decrypt(p, *f.decryptor, p.column.front().scale());
    const double worst = max_abs_diff(want, got);
    std::cout << "batch16 softmax seam worst absolute error: " << worst
              << std::endl;
    // The tolerance of the approximation, not of the encoding: the reference
    // is the true SoftMax and the circuit runs two normalise-and-square
    // rounds behind a degree-15 exponential.
    EXPECT_LT(worst, 5e-2);
}

// ---------------------------------------------------------------------------
// 3. The advertised ledger is the ledger it spends
// ---------------------------------------------------------------------------

// softmax_seam_levels() is what a caller sizes its modulus chain against, and
// a closed form that nobody measures is a comment. This runs the seam and
// counts.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_LevelLedgerIsWhatItAdvertises)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 4242u);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, shift);
    calibrate(scores, Fixture::d, shift, softmax.iterations, softmax);

    const int predicted = Batch::softmax_seam_levels(softmax, seam);
    Fixture f(predicted + 3);
    auto ct = encrypt_scores(f, scores, seam, bound);

    const int before = ct.column.front().depth();
    auto p = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    const int after = p.column.front().depth();

    std::cout << "batch16 softmax seam levels: predicted " << predicted
              << ", spent " << (after - before) << std::endl;
    EXPECT_EQ(after - before, predicted);
}

// The folds are the free half of the branch's answer: the exponential's domain
// map rides on whatever plaintext formed the scores, the reciprocal's rides on
// the causal mask, and neither is a homomorphic operation at all. What they
// buy is levels off the deepest stretch in the block.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_FoldsCostLevelsAndNotAccuracy)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 90210u);
    const auto want = host_causal_softmax(scores, Fixture::d, shift);

    SoftmaxConfig folded;
    SeamConfig folded_seam;
    production(folded, folded_seam, bound, shift);
    calibrate(scores, Fixture::d, shift, folded.iterations, folded);

    // The same circuit with both folds off: the affine maps become homomorphic
    // multiplications and each costs a level.
    SoftmaxConfig plain = folded;
    SeamConfig plain_seam = folded_seam;
    plain_seam.scores_carry_exp_domain = false;
    plain_seam.fold_affine_into_mask = false;

    const int folded_levels = Batch::softmax_seam_levels(folded, folded_seam);
    const int plain_levels = Batch::softmax_seam_levels(plain, plain_seam);

    std::cout << "batch16 softmax seam levels: folded " << folded_levels
              << ", unfolded " << plain_levels << std::endl;
    // One for the exponential's map, one per round for the reciprocal's.
    EXPECT_EQ(plain_levels - folded_levels, 1 + folded.iterations);

    // And both are the SoftMax, so the levels really were free.
    Fixture f(plain_levels + 3);
    {
        auto ct = encrypt_scores(f, scores, folded_seam, bound);
        auto p =
            f.op->softmax_seam(ct, folded, folded_seam, *f.galois, *f.relin);
        const auto got =
            f.op->decrypt(p, *f.decryptor, p.column.front().scale());
        const double worst = max_abs_diff(want, got);
        std::cout << "  folded worst absolute error: " << worst << std::endl;
        EXPECT_LT(worst, 5e-2);
    }
    {
        auto ct = encrypt_scores(f, scores, plain_seam, bound);
        auto p = f.op->softmax_seam(ct, plain, plain_seam, *f.galois, *f.relin);
        const auto got =
            f.op->decrypt(p, *f.decryptor, p.column.front().scale());
        const double worst = max_abs_diff(want, got);
        std::cout << "  unfolded worst absolute error: " << worst << std::endl;
        EXPECT_LT(worst, 5e-2);
    }
}

// The other eight levels, and they are NOT free -- so they are measured rather
// than asserted. fold_affine_into_mask is silently ANDed with
// inverse_newton <= 0, so taking the fold forces the Newton refinement off and
// the reciprocal becomes the bare degree-15 fit. That is safe only if the fit
// is taken over a MEASURED range instead of the worst case, which is the whole
// of Section 4.3's argument. This runs both and compares them to the truth.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_CalibrationIsWhatPaysForTheNewtonSteps)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 8675309u);
    const auto want = host_causal_softmax(scores, Fixture::d, shift);

    // The refined form: a cheap degree-15 fit rescued by two Newton steps,
    // and therefore no fold.
    SoftmaxConfig refined;
    SeamConfig refined_seam;
    production(refined, refined_seam, bound, shift);
    refined.inverse_degree = 15;
    refined.inverse_newton = 2;
    refined_seam.fold_affine_into_mask = false;
    calibrate(scores, Fixture::d, shift, refined.iterations, refined);

    // The folded form: no Newton step, a wider fit instead, same calibrated
    // range.
    SoftmaxConfig folded;
    SeamConfig folded_seam;
    production(folded, folded_seam, bound, shift);
    calibrate(scores, Fixture::d, shift, folded.iterations, folded);

    const int refined_levels =
        Batch::softmax_seam_levels(refined, refined_seam);
    const int folded_levels = Batch::softmax_seam_levels(folded, folded_seam);
    std::cout << "batch16 softmax seam levels: refined " << refined_levels
              << ", folded+calibrated " << folded_levels << std::endl;
    // Per round the refined form pays the 1/x affine map and two levels per
    // Newton step; the folded form pays a wider fit instead. Both configs
    // carry the exponential's map on the input, so that level does not enter.
    const int expected =
        refined.iterations * (1 + 2 * refined.inverse_newton) -
        refined.iterations *
            (fit_levels(folded.inverse_degree) -
             fit_levels(refined.inverse_degree));
    EXPECT_EQ(refined_levels - folded_levels, expected);

    Fixture f(refined_levels + 3);

    auto a = encrypt_scores(f, scores, refined_seam, bound);
    auto p_refined =
        f.op->softmax_seam(a, refined, refined_seam, *f.galois, *f.relin);
    const auto got_refined = f.op->decrypt(p_refined, *f.decryptor,
                                           p_refined.column.front().scale());
    const double worst_refined = max_abs_diff(want, got_refined);

    auto b = encrypt_scores(f, scores, folded_seam, bound);
    auto p_folded =
        f.op->softmax_seam(b, folded, folded_seam, *f.galois, *f.relin);
    const auto got_folded = f.op->decrypt(p_folded, *f.decryptor,
                                          p_folded.column.front().scale());
    const double worst_folded = max_abs_diff(want, got_folded);

    std::cout << "  refined worst absolute error:          " << worst_refined
              << std::endl;
    std::cout << "  folded + calibrated worst abs error:   " << worst_folded
              << std::endl;

    // Both are the SoftMax. The claim being pinned is not that the cheaper one
    // is better, but that calibration buys back what the Newton steps were
    // paying for -- so the levels really are recoverable and not merely
    // deleted.
    EXPECT_LT(worst_refined, 5e-2);
    EXPECT_LT(worst_folded, 5e-2);
}

// ---------------------------------------------------------------------------
// 4. The work changes and the answer does not
// ---------------------------------------------------------------------------

// The masks depend on the query and key indices alone, never on the head, so a
// sublayer running H heads encodes the same d vectors H times over. The cache
// is keyed by the caller, so what has to be checked is that the key is right:
// the same seam run twice must hit d times and return the same numbers.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_MaskCacheChangesEncodesNotAnswers)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 5150u);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, shift);
    calibrate(scores, Fixture::d, shift, softmax.iterations, softmax);

    Fixture f(Batch::softmax_seam_levels(softmax, seam) + 3);

    // ONE encryption, three seams. Encrypting per run would compare two
    // different samples of the encryption noise and measure nothing: the
    // first cut of this test did exactly that and reported a 4e-7 "difference"
    // that was the noise and not the cache.
    auto ct = encrypt_scores(f, scores, seam, bound);

    // Uncached, so that the cached runs have something to be compared against.
    SeamConfig uncached = seam;
    uncached.cache_masks = false;
    auto reference =
        f.op->softmax_seam(ct, softmax, uncached, *f.galois, *f.relin);
    const auto want =
        f.op->decrypt(reference, *f.decryptor,
                      reference.column.front().scale());
    EXPECT_EQ(f.op->arith().mask_plain_hits(), 0u)
        << "cache_masks = false must not consult the cache";

    // First cached run: d distinct masks, so d misses and no hits.
    auto once = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    const auto after_one =
        f.op->decrypt(once, *f.decryptor, once.column.front().scale());
    EXPECT_EQ(f.op->arith().mask_plain_hits(), 0u);

    // Second cached run at the same depth -- which is what the next head is.
    // Every one of the d encodes is now served.
    auto twice = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    const auto after_two =
        f.op->decrypt(twice, *f.decryptor, twice.column.front().scale());
    EXPECT_EQ(f.op->arith().mask_plain_hits(),
              static_cast<size_t>(Fixture::d))
        << "a second head must encode no mask at all";

    // A cache that changed an answer would be a cache with the wrong key.
    EXPECT_EQ(max_abs_diff(want, after_one), 0.0);
    EXPECT_EQ(max_abs_diff(want, after_two), 0.0);
}

// Hoisting shares one decomposition across a crossing's baby shifts. The
// modular arithmetic is identically ordered, so it is the same circuit at a
// fraction of the work -- and "the same" here means to the bit, not to a
// tolerance.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_HoistedCrossingsAreBitIdentical)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 31337u);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, shift);
    calibrate(scores, Fixture::d, shift, softmax.iterations, softmax);

    Fixture f(Batch::softmax_seam_levels(softmax, seam) + 3);

    // One encryption, two seams: "bit-identical" is a claim about the
    // arithmetic, and re-encrypting would compare two samples of the
    // encryption noise instead.
    auto ct = encrypt_scores(f, scores, seam, bound);

    SeamConfig plain = seam;
    plain.hoisted_crossings = false;
    auto p_plain = f.op->softmax_seam(ct, softmax, plain, *f.galois, *f.relin);
    const auto want =
        f.op->decrypt(p_plain, *f.decryptor, p_plain.column.front().scale());

    auto p_hoisted = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    const auto got = f.op->decrypt(p_hoisted, *f.decryptor,
                                   p_hoisted.column.front().scale());

    EXPECT_EQ(max_abs_diff(want, got), 0.0);

    // And the operator's own setting survives the call, because the seam is
    // not the only thing that crosses.
    const bool before = f.op->hoisted_crossings();
    f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    EXPECT_EQ(f.op->hoisted_crossings(), before);
}

// ---------------------------------------------------------------------------
// 5. The shift is calibration, and the seam owns its scaling
// ---------------------------------------------------------------------------

// A causal row of length u + 1 has a different score range from a full one, so
// one number for the whole block is the loosest calibration available. The
// per-row form is the sharp one, and it is the same slot vector and the same
// zero levels. What is checked here is that the seam scales it to match the
// folded domain map by itself -- the trap the rect path leaves to its caller.
TEST(HEonGPU, CKKS_Llama3BatchSoftmax_PerRowShiftAgreesWithTheScalarOne)
{
    const double bound = 4.0;
    const double shift = 2.0;
    const auto scores =
        random_scores(16, Fixture::d, shift - bound, shift, 606u);
    const auto want = host_causal_softmax(scores, Fixture::d, shift);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, shift);
    calibrate(scores, Fixture::d, shift, softmax.iterations, softmax);

    // The same shift, said per row. The answer must not move, and it must not
    // move whether or not the exponential's map is folded -- which is the
    // scaling the seam does on the caller's behalf.
    SeamConfig rows = seam;
    rows.score_shift = 0.0;
    rows.score_shift_rows.assign(Fixture::d, shift);

    Fixture f(Batch::softmax_seam_levels(softmax, seam) + 3);

    // Both seams read the same ciphertext: the two shifts are the same
    // numbers said two ways, so what is being pinned is the seam's scaling of
    // them and not the encryption.
    auto ct = encrypt_scores(f, scores, seam, bound);

    auto scalar = f.op->softmax_seam(ct, softmax, seam, *f.galois, *f.relin);
    const auto from_scalar =
        f.op->decrypt(scalar, *f.decryptor, scalar.column.front().scale());

    auto per_row = f.op->softmax_seam(ct, softmax, rows, *f.galois, *f.relin);
    const auto from_rows =
        f.op->decrypt(per_row, *f.decryptor, per_row.column.front().scale());

    EXPECT_LT(max_abs_diff(from_scalar, from_rows), 1e-6);
    EXPECT_LT(max_abs_diff(want, from_rows), 5e-2);
}

// ---------------------------------------------------------------------------
// 6. Contracts that must fail loudly
// ---------------------------------------------------------------------------

TEST(HEonGPU, CKKS_Llama3BatchSoftmax_SeamRefusesWhatItCannotDo)
{
    Fixture f(8);
    const double bound = 4.0;
    const auto scores = random_scores(16, Fixture::d, -bound, 0.0, 77u);

    SoftmaxConfig softmax;
    SeamConfig seam;
    production(softmax, seam, bound, 0.0);
    softmax.sum_lo = 1.0;
    softmax.sum_hi = 2.0;

    auto ct = encrypt_scores(f, scores, seam, bound);

    // The auxiliary track needs the boot key. Falling back silently would put
    // the fit's levels back on the wide track and change the schedule the
    // caller sized its chain for.
    SeamConfig aux = seam;
    aux.refresh_denominator = true;
    EXPECT_THROW(f.op->softmax_seam(ct, softmax, aux, *f.galois, *f.relin),
                 std::invalid_argument);

    // A score block is square at d, because that is what Algorithm 4 hands
    // back and what the value product will take.
    llama::BatchActivation narrow;
    narrow.rows = Fixture::d;
    narrow.column.push_back(ct.column.front());
    EXPECT_THROW(f.op->softmax_seam(narrow, softmax, seam, *f.galois, *f.relin),
                 std::invalid_argument);

    // A per-row shift is one entry per query.
    SeamConfig wrong_rows = seam;
    wrong_rows.score_shift_rows.assign(Fixture::d + 1, 0.0);
    EXPECT_THROW(
        f.op->softmax_seam(ct, softmax, wrong_rows, *f.galois, *f.relin),
        std::invalid_argument);

    // A named mask list names every part or none of them: a short list would
    // cache under the wrong name, which is the one failure mode of a
    // caller-keyed cache and is not detectable downstream.
    std::vector<heongpu::Ciphertext<S>> parts;
    parts.push_back(ct.column.front());
    parts.push_back(ct.column.back());
    std::vector<std::vector<double>> masks{
        f.op->causal_column_mask(0), f.op->causal_column_mask(1)};
    std::vector<int> short_ids{0};
    SoftmaxConfig direct = softmax;
    direct.strided = true;
    direct.stride = f.encoder->slot_count();
    direct.count = 1;
    EXPECT_THROW(f.op->arith().softmax(parts, direct, masks, short_ids,
                                       *f.galois, *f.relin, nullptr),
                 std::invalid_argument);
}
