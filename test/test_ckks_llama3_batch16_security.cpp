// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Does the batch-16 Llama-3 block satisfy the security cap?
//
// It does not, and this file is the proof rather than the assertion. Every
// number below is either taken from the library's own predicates or checked
// against them, and the central test is AgreesWithTheContextOnEveryPlan: it
// puts fourteen parameter sets through BOTH this module and a real
// HEContext, and requires the two to agree in both directions. A module that
// merely restated the rules could drift from them silently; one that is
// differentially tested against the thing it describes cannot.
//
// None of this needs a GPU. set_poly_modulus_degree and
// set_coeff_modulus_bit_sizes are host-side, and the security check, the pair
// rule and the prime-range check all run inside them -- before generate()
// touches a device. So the one question a card cannot be spared for is the one
// question that can be answered without one, which is why these tests exist
// separately from the block tests they are about.
//
// WHAT IS ASSERTED, IN ORDER
// --------------------------
//   1. The module and the context agree, on admissible and inadmissible sets
//      alike, and for the right reason each time.
//   2. LLAMA3_8B_LAYER_FLOW.md section 24.5's island -- Q = {41, 33},
//      P = {33}, "107 <= 109, legal by two bits" -- DOES NOT BUILD. It is
//      rejected by coefficient_validator before the security check is
//      reached, because one special prime must cover every individual Q
//      prime and 41 > 33. The write-up checked the cap and not the pair
//      rule.
//   3. What N = 4096 does admit, exactly: two Q primes, one special, nothing
//      above 36 bits. Three Q primes are impossible at ANY prime sizes, and
//      that is asserted by exhaustive search rather than by argument.
//   4. The ladder: the batch axis is the ring axis, so it is also the
//      security axis, and the smallest batch that carries the block's own
//      chain is 256.
//   5. The block's real numbers, taken from Llama3Batch16Operator itself:
//      refresh_levels() = 25, chain_limbs_for(35) = 61, and 61 against an
//      admissible 2 is a factor of thirty.
//   6. The two-ring pair that does close, and the constraint that decides its
//      prime size -- which is not the island's own cap but the shared prefix
//      the ring switch requires.
//   7. That the fixtures the batch-16 tests and the block profiler actually
//      use are 19.3x and 24.0x over, computed from their literal arguments.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;
    using heongpu::sec_level_type;

    constexpr int kHeadDim = 128; // Llama-3-8B, and Algorithm 4 pins it.

    /// What the LIBRARY does with a parameter set, and why.
    enum class Build
    {
        Ok,
        PairRule, ///< coefficient_validator rejected it.
        Cap, ///< the lattice-estimator cap rejected it.
        PrimeSize, ///< a bit size is outside [30, 60].
        Ring, ///< the ring degree is not supported.
    };

    /// Ask the context itself. Deliberately stops short of generate(), which
    /// is the only part that needs a device -- every rule under test runs in
    /// the two setters.
    Build library_verdict(const llama::ModulusPlan& plan, sec_level_type level)
    {
        heongpu::HEContext<S> context = heongpu::GenHEContext<S>(level);
        try
        {
            context->set_poly_modulus_degree(static_cast<size_t>(plan.n));
        }
        catch (const std::logic_error&)
        {
            return Build::Ring;
        }

        try
        {
            context->set_coeff_modulus_bit_sizes(plan.log_q, plan.log_p);
        }
        catch (const std::logic_error& e)
        {
            // Two different rules throw logic_error, and telling them apart
            // matters: the pair rule is the one section 24 missed.
            const std::string what(e.what());
            if (what.find("P should be bigger") != std::string::npos)
            {
                return Build::PairRule;
            }
            return Build::PrimeSize;
        }
        catch (const std::runtime_error&)
        {
            return Build::Cap;
        }
        return Build::Ok;
    }

    llama::ModulusPlan plan_of(int n, std::vector<int> q, std::vector<int> p)
    {
        llama::ModulusPlan plan;
        plan.n = n;
        plan.log_q = std::move(q);
        plan.log_p = std::move(p);
        return plan;
    }

    /// The block test's own fixture chain, rebuilt from its literal
    /// arguments (test_ckks_llama3_batch16_block.cpp:95-98).
    llama::ModulusPlan block_test_fixture(int limbs)
    {
        std::vector<int> q{60};
        q.insert(q.end(), limbs - 1, 40);
        return plan_of(4096, std::move(q), {60, 60});
    }
} // namespace

// =====================================================================
// 1. The module and the library agree
// =====================================================================

TEST(Batch16Security, AgreesWithTheContextOnEveryPlan)
{
    struct Case
    {
        const char* name;
        llama::ModulusPlan plan;
        sec_level_type level;
    };

    const std::vector<Case> cases{
        // Admissible, and only just: the largest uniform set N = 4096 has.
        {"island max uniform", plan_of(4096, {36, 36}, {36}),
         sec_level_type::sec128},
        // One bit more on q0 and the PAIR rule takes it, not the cap -- which
        // is the trap this whole file exists for.
        {"island, q0 one bit over P", plan_of(4096, {37, 36}, {36}),
         sec_level_type::sec128},
        // Widen the special to match and the cap takes it instead.
        {"island, both widened", plan_of(4096, {37, 36}, {37}),
         sec_level_type::sec128},
        // Section 24.5's island. Rejected by the PAIR rule, not the cap.
        {"flow doc island", plan_of(4096, {41, 33}, {33}),
         sec_level_type::sec128},
        // The same set with P raised to clear the pair rule: now the cap
        // rejects it, which is the trade the write-up did not price.
        {"flow doc island, P raised", plan_of(4096, {41, 33}, {41}),
         sec_level_type::sec128},
        // Three Q primes at the smallest legal prime: still over.
        {"three at the floor", plan_of(4096, {30, 30, 30}, {30}),
         sec_level_type::sec128},
        // Method II at N = 4096 with two Q primes: over before it starts.
        {"method II island", plan_of(4096, {30, 30}, {30, 30}),
         sec_level_type::sec128},
        // A prime below the library's floor.
        {"below the prime floor", plan_of(4096, {29, 29}, {29}),
         sec_level_type::sec128},
        // A ring the library does not support at all.
        {"ring too small", plan_of(2048, {30}, {30}), sec_level_type::sec128},
        // The block test's own 49-limb fixture, at the level it runs at...
        {"block fixture, none", block_test_fixture(49), sec_level_type::none},
        // ... and the same set with the check on.
        {"block fixture, sec128", block_test_fixture(49),
         sec_level_type::sec128},
        // The big ring of the two-ring pair.
        {"big ring", llama::make_plan(65536, 43, 36, 33, 8, 40),
         sec_level_type::sec128},
        // A logN 16 chain long enough for the block, single ring.
        {"logN 16, 61 limbs", llama::make_plan(65536, 61, 33, 33, 1, 33),
         sec_level_type::sec128},
        // logN 15 cannot hold a refresh plus work.
        {"logN 15, 27 limbs", llama::make_plan(32768, 27, 33, 33, 1, 33),
         sec_level_type::sec128},
        // 192-bit, to prove the level is a parameter and not a constant.
        {"island at 192-bit", plan_of(4096, {36, 36}, {36}),
         sec_level_type::sec192},
    };

    for (const Case& c : cases)
    {
        const llama::ParameterVerdict v = llama::inspect(c.plan, c.level);
        const Build actual = library_verdict(c.plan, c.level);

        EXPECT_EQ(v.admissible(), actual == Build::Ok)
            << c.name << ": the module says "
            << (v.admissible() ? "admissible" : "not")
            << " and the context says " << static_cast<int>(actual) << "\n"
            << v.explain();

        // Agreeing on the verdict is not enough -- agreeing on the REASON is
        // what stops a right answer for a wrong reason, which is exactly the
        // failure section 24 made.
        if (actual == Build::PairRule)
        {
            EXPECT_FALSE(v.passes_pair_rule) << c.name;
        }
        if (actual == Build::Cap)
        {
            EXPECT_FALSE(v.within_cap) << c.name;
            EXPECT_TRUE(v.passes_pair_rule)
                << c.name
                << ": the context only reaches the cap check after the pair "
                   "rule passes, so the module must agree it passed";
        }
        if (actual == Build::PrimeSize)
        {
            EXPECT_FALSE(v.primes_in_range) << c.name;
        }
        if (actual == Build::Ring)
        {
            EXPECT_FALSE(v.ring_supported) << c.name;
        }
    }
}

TEST(Batch16Security, NoneIsNoCapRatherThanAWeakOne)
{
    const llama::ModulusPlan huge = block_test_fixture(49);
    const llama::ParameterVerdict none =
        llama::inspect(huge, sec_level_type::none);

    EXPECT_TRUE(none.admissible());
    // A verdict at `none` must not be quotable as a ratio: there is no cap to
    // be a fraction of, and reporting 0 is what keeps a cost model from being
    // read as a parameter set.
    EXPECT_EQ(none.cap, 0);
    EXPECT_DOUBLE_EQ(none.cap_ratio(), 0.0);
    EXPECT_EQ(library_verdict(huge, sec_level_type::none), Build::Ok);
}

// =====================================================================
// 2. The flow document's island does not build
// =====================================================================

TEST(Batch16Security, TheDocumentedIslandIsRejectedByThePairRuleNotTheCap)
{
    const llama::ModulusPlan island = plan_of(4096, {41, 33}, {33});

    // The write-up's arithmetic is right as far as it goes.
    EXPECT_EQ(island.log_qp(), 107);
    EXPECT_LE(island.log_qp(), llama::security_cap(4096, sec_level_type::sec128));

    // And it does not build, because the cap is the second rule, not the
    // first. With one special prime the pair rule degenerates to
    // "every q_i <= P", and 41 > 33.
    EXPECT_FALSE(heongpu::coefficient_validator(island.log_q, island.log_p));
    EXPECT_EQ(library_verdict(island, sec_level_type::sec128), Build::PairRule);

    // It does not build even with the check off, which is the part that makes
    // this a fact about the library rather than about security.
    EXPECT_EQ(library_verdict(island, sec_level_type::none), Build::PairRule);

    const llama::ParameterVerdict v =
        llama::inspect(island, sec_level_type::sec128);
    EXPECT_FALSE(v.admissible());
    EXPECT_FALSE(v.passes_pair_rule);
    EXPECT_TRUE(v.within_cap);
}

TEST(Batch16Security, RaisingPToClearThePairRuleThenBreaksTheCap)
{
    // The obvious repair costs exactly the eight bits it was inside by.
    const llama::ModulusPlan repaired = plan_of(4096, {41, 33}, {41});
    EXPECT_TRUE(
        heongpu::coefficient_validator(repaired.log_q, repaired.log_p));
    EXPECT_EQ(repaired.log_qp(), 115);
    EXPECT_GT(repaired.log_qp(),
              llama::security_cap(4096, sec_level_type::sec128));
    EXPECT_EQ(library_verdict(repaired, sec_level_type::sec128), Build::Cap);
}

TEST(Batch16Security, TheLargestIslandThatDoesBuildIsThirtySixBitPrimes)
{
    // Uniform primes are optimal here: the pair rule forces P >= max q_i, so
    // widening any one prime widens P too and costs twice.
    const llama::ModulusPlan best = plan_of(4096, {36, 36}, {36});
    EXPECT_EQ(best.log_qp(), 108);
    EXPECT_EQ(library_verdict(best, sec_level_type::sec128), Build::Ok);
    EXPECT_EQ(best.usable_levels(), 1);
    EXPECT_EQ(best.method(), heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    EXPECT_EQ(library_verdict(plan_of(4096, {37, 36}, {37}),
                              sec_level_type::sec128),
              Build::Cap);
    EXPECT_EQ(library_verdict(plan_of(4096, {37, 37}, {37}),
                              sec_level_type::sec128),
              Build::Cap);
}

TEST(Batch16Security, ThreeQPrimesAreImpossibleAtFourZeroNineSixAtAnySizes)
{
    // Exhaustive, not argued. Every assignment of three Q primes and any
    // number of specials within the library's own bit range.
    int admissible = 0;
    for (int q0 = MIN_USER_DEFINED_MOD_BIT_COUNT;
         q0 <= MAX_USER_DEFINED_MOD_BIT_COUNT; ++q0)
    {
        for (int q1 = MIN_USER_DEFINED_MOD_BIT_COUNT;
             q1 <= MAX_USER_DEFINED_MOD_BIT_COUNT; ++q1)
        {
            for (int q2 = MIN_USER_DEFINED_MOD_BIT_COUNT;
                 q2 <= MAX_USER_DEFINED_MOD_BIT_COUNT; ++q2)
            {
                for (int specials = 1; specials <= 3; ++specials)
                {
                    for (int pb = MIN_USER_DEFINED_MOD_BIT_COUNT;
                         pb <= MAX_USER_DEFINED_MOD_BIT_COUNT; ++pb)
                    {
                        llama::ModulusPlan plan =
                            plan_of(4096, {q0, q1, q2},
                                    std::vector<int>(specials, pb));
                        if (llama::inspect(plan, sec_level_type::sec128)
                                .admissible())
                        {
                            ++admissible;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(admissible, 0)
        << "three Q primes fit at N = 4096 after all -- the island analysis "
           "needs redoing";

    // And two do, so the search is not vacuously empty.
    EXPECT_TRUE(llama::inspect(plan_of(4096, {36, 36}, {36}),
                               sec_level_type::sec128)
                    .admissible());
}

TEST(Batch16Security, MethodTwoIsUnreachableAtFourZeroNineSixWithAnyLevel)
{
    // Two specials cost at least 60 bits, and two Q primes another 60, which
    // is 120 against a cap of 109 before anything is chosen. So the island is
    // method I necessarily -- and that is why hoisted rotation trains, which
    // need method II, are unavailable there.
    for (int specials = 2; specials <= 4; ++specials)
    {
        for (int limbs = 2; limbs <= 4; ++limbs)
        {
            const llama::ModulusPlan plan =
                llama::make_plan(4096, limbs, MIN_USER_DEFINED_MOD_BIT_COUNT,
                                 MIN_USER_DEFINED_MOD_BIT_COUNT, specials,
                                 MIN_USER_DEFINED_MOD_BIT_COUNT);
            EXPECT_FALSE(
                llama::inspect(plan, sec_level_type::sec128).admissible())
                << "specials " << specials << ", limbs " << limbs;
        }
    }
}

// =====================================================================
// 3. The ladder: the batch axis is the security axis
// =====================================================================

TEST(Batch16Security, TheBatchSizeChoosesTheRing)
{
    // N = 2 * batch * head_dim, from batch = k/2 and d = N/k = head_dim.
    EXPECT_EQ(llama::ring_for_batch(16, kHeadDim), 4096);
    EXPECT_EQ(llama::ring_for_batch(32, kHeadDim), 8192);
    EXPECT_EQ(llama::ring_for_batch(64, kHeadDim), 16384);
    EXPECT_EQ(llama::ring_for_batch(128, kHeadDim), 32768);
    EXPECT_EQ(llama::ring_for_batch(256, kHeadDim), 65536);

    for (int batch : {16, 32, 64, 128, 256})
    {
        EXPECT_EQ(llama::batch_for_ring(llama::ring_for_batch(batch, kHeadDim),
                                        kHeadDim),
                  batch);
    }

    // Batch 8 would want N = 2048, which the library refuses outright, and
    // batch 512 would want 131072, likewise. The batch axis is a five-valued
    // enum and this is why.
    EXPECT_FALSE(llama::inspect(llama::make_plan(llama::ring_for_batch(8,
                                                                      kHeadDim),
                                                 1, 30, 30, 1, 30),
                                sec_level_type::sec128)
                     .ring_supported);
    EXPECT_FALSE(llama::inspect(llama::make_plan(llama::ring_for_batch(512,
                                                                       kHeadDim),
                                                 1, 30, 30, 1, 30),
                                sec_level_type::sec128)
                     .ring_supported);
}

TEST(Batch16Security, TheAdmissibleChainLadderAtThirtyThreeBitPrimes)
{
    // One special of the same size, which the pair rule makes the cheapest
    // arrangement. The closed form is floor(cap/w) - specials; the module
    // searches rather than solves, so this checks the search against it.
    struct Row
    {
        int batch;
        int n;
        int cap;
        int limbs;
    };
    const std::vector<Row> ladder{
        {16, 4096, 109, 2},   {32, 8192, 218, 5},
        {64, 16384, 438, 12}, {128, 32768, 881, 25},
        {256, 65536, 1761, 52},
    };

    for (const Row& r : ladder)
    {
        EXPECT_EQ(llama::ring_for_batch(r.batch, kHeadDim), r.n)
            << "batch " << r.batch;
        EXPECT_EQ(llama::security_cap(r.n, sec_level_type::sec128), r.cap)
            << "batch " << r.batch;

        const int limbs =
            llama::max_limbs(r.n, 33, 33, 1, 33, sec_level_type::sec128);
        EXPECT_EQ(limbs, r.limbs) << "batch " << r.batch;
        EXPECT_EQ(limbs, r.cap / 33 - 1) << "batch " << r.batch;

        // The boundary is real in both directions.
        EXPECT_TRUE(llama::inspect(llama::make_plan(r.n, limbs, 33, 33, 1, 33),
                                   sec_level_type::sec128)
                        .admissible());
        EXPECT_FALSE(
            llama::inspect(llama::make_plan(r.n, limbs + 1, 33, 33, 1, 33),
                           sec_level_type::sec128)
                .admissible());
    }
}

TEST(Batch16Security, ARefreshDoesNotFitBelowLogNSixteen)
{
    // A regular bootstrap at the library's default configuration spends 25
    // levels on itself, so a ring that refreshes at all needs 27 limbs -- 25
    // spent, one handed in, one handed back with something to spend.
    const heongpu::BootstrappingConfig boot;
    const int refresh =
        llama::Llama3Batch16Operator::refresh_levels(boot);
    ASSERT_EQ(refresh, 25);
    const int needed = llama::Llama3Batch16Operator::chain_limbs_for(1, boot);
    ASSERT_EQ(needed, 27);

    EXPECT_LT(llama::max_limbs(4096, 33, 33, 1, 33, sec_level_type::sec128),
              needed);
    EXPECT_LT(llama::max_limbs(8192, 33, 33, 1, 33, sec_level_type::sec128),
              needed);
    EXPECT_LT(llama::max_limbs(16384, 33, 33, 1, 33, sec_level_type::sec128),
              needed);
    EXPECT_LT(llama::max_limbs(32768, 33, 33, 1, 33, sec_level_type::sec128),
              needed);
    EXPECT_GE(llama::max_limbs(65536, 33, 33, 1, 33, sec_level_type::sec128),
              needed);

    // So the smallest batch in this family that can refresh at all is 256 --
    // and every smaller batch must therefore put its refresh at another ring.
    EXPECT_EQ(llama::smallest_batch_for(needed, kHeadDim, 33, 33, 1, 33,
                                        sec_level_type::sec128),
              256);
}

// =====================================================================
// 4. The block's own numbers
// =====================================================================

TEST(Batch16Security, TheBlocksChainIsThirtyTimesWhatBatchSixteenAdmits)
{
    const heongpu::BootstrappingConfig boot;

    // The two stretches the block has been measured at. 17 is the cheap fit
    // degrees the block tests use; 35 is the attention sublayer at the
    // headers' own defaults.
    const int cheap = llama::Llama3Batch16Operator::chain_limbs_for(17, boot);
    const int production =
        llama::Llama3Batch16Operator::chain_limbs_for(35, boot);
    ASSERT_EQ(cheap, 43);
    ASSERT_EQ(production, 61);

    const llama::BlockSecurity report =
        llama::audit_block(16, kHeadDim, production, 25, 33, 33, 1, 33,
                           sec_level_type::sec128);

    EXPECT_EQ(report.n, 4096);
    EXPECT_EQ(report.cap, 109);
    EXPECT_EQ(report.admissible_limbs, 2);
    EXPECT_FALSE(report.single_ring);
    EXPECT_FALSE(report.ring_can_refresh);
    EXPECT_EQ(report.refresh_limbs, 27);
    EXPECT_NEAR(report.over_by(), 30.5, 0.05);

    // And raising the batch does NOT rescue the production block: no ring in
    // this library holds a 61-limb chain at 128-bit. See
    // NoRingHoldsTheProductionChainAtAnyPrimeSize.
    EXPECT_EQ(report.smallest_batch, 0);

    // The cheap-fit block is the one that a bigger batch does rescue, and the
    // difference between the two is entirely the fit degrees -- which is what
    // makes shortening the chain a SECURITY lever and not only a speed one.
    const llama::BlockSecurity cheap_report =
        llama::audit_block(16, kHeadDim, cheap, 25, 33, 33, 1, 33,
                           sec_level_type::sec128);
    EXPECT_FALSE(cheap_report.single_ring);
    EXPECT_EQ(cheap_report.smallest_batch, 256);
}

TEST(Batch16Security, NoRingHoldsTheProductionChainAtAnyPrimeSize)
{
    // The strongest form of the result, and it does not depend on a choice of
    // prime size: the cheapest possible chain of C Q primes plus one special
    // costs (C + 1) * 30 bits, because 30 is the library's own floor. At
    // C = 61 that is 1860 against the largest cap the library has, 1761.
    const heongpu::BootstrappingConfig boot;
    const int production =
        llama::Llama3Batch16Operator::chain_limbs_for(35, boot);
    ASSERT_EQ(production, 61);

    const int biggest_cap =
        llama::security_cap(65536, sec_level_type::sec128);
    ASSERT_EQ(biggest_cap, 1761);
    EXPECT_GT((production + 1) * MIN_USER_DEFINED_MOD_BIT_COUNT, biggest_cap);

    // Which the ladder confirms directly, at both the widest sensible prime
    // and the library's floor.
    for (int w : {33, MIN_USER_DEFINED_MOD_BIT_COUNT})
    {
        EXPECT_EQ(llama::smallest_batch_for(production, kHeadDim, w, w, 1, w,
                                            sec_level_type::sec128),
                  0)
            << "prime bits " << w;
    }

    // The longest chain 128-bit security admits ANYWHERE in this library is
    // 57 Q primes, at logN 16 and the 30-bit floor. Every schedule this
    // project runs has to fit inside that number or use two rings.
    EXPECT_EQ(llama::max_limbs(65536, MIN_USER_DEFINED_MOD_BIT_COUNT,
                               MIN_USER_DEFINED_MOD_BIT_COUNT, 1,
                               MIN_USER_DEFINED_MOD_BIT_COUNT,
                               sec_level_type::sec128),
              57);

    // The cheap-fit chain is inside it; the production one is not.
    EXPECT_LE(llama::Llama3Batch16Operator::chain_limbs_for(17, boot), 57);
    EXPECT_GT(production, 57);
}

TEST(Batch16Security, TheFixturesTheTestsActuallyBuildAreTwentyTimesOver)
{
    // Not a hypothetical set -- these are rebuilt from the literal arguments
    // in the block test and the block profiler, so that a change to either
    // shows up here.
    struct Row
    {
        const char* name;
        llama::ModulusPlan plan;
        int log_qp;
        double ratio;
    };

    const std::vector<Row> rows{
        {"block test, 3 limbs", block_test_fixture(3), 260, 2.385},
        {"block test, 19 limbs", block_test_fixture(19), 900, 8.257},
        {"block test, 49 limbs", block_test_fixture(49), 2100, 19.266},
        // The whole-block profiler: 62 limbs, N fixed at 4096.
        {"block profiler, 62 limbs", block_test_fixture(62), 2620, 24.037},
    };

    for (const Row& r : rows)
    {
        const llama::ParameterVerdict v =
            llama::inspect(r.plan, sec_level_type::sec128);
        EXPECT_EQ(v.log_qp, r.log_qp) << r.name;
        EXPECT_EQ(v.cap, 109) << r.name;
        EXPECT_NEAR(v.cap_ratio(), r.ratio, 0.01) << r.name;
        EXPECT_FALSE(v.admissible()) << r.name;
        EXPECT_EQ(library_verdict(r.plan, sec_level_type::sec128), Build::Cap)
            << r.name;
    }

    // The two special primes alone are already over the cap, so no batch-16
    // context in this tree can be rescued by shortening Q.
    EXPECT_GT(60 + 60, llama::security_cap(4096, sec_level_type::sec128));
}

// =====================================================================
// 5. The two-ring pair that does close
// =====================================================================

TEST(Batch16Security, ATwoRingPairThatBothHalvesClear)
{
    // The island carries exactly one Algorithm 4 product, which is what one
    // usable level buys; everything else, the refresh included, sits at the
    // big ring. The prime size is 36 because the ISLAND's cap sets it, and
    // the shared prefix then imposes that size on the big ring's q0 too.
    const llama::TwoRingPlan pair = llama::make_two_ring_plan(
        /*island_n=*/4096, /*big_n=*/65536,
        /*island_limbs=*/2, /*big_limbs=*/43,
        /*q0_bits=*/36, /*island_prime_bits=*/36, /*big_prime_bits=*/33,
        /*island_specials=*/1, /*big_specials=*/8,
        /*island_special_bits=*/36, /*big_special_bits=*/40);

    EXPECT_TRUE(pair.shares_prefix());
    EXPECT_EQ(pair.ratio(), 16);

    const llama::ParameterVerdict island =
        llama::inspect(pair.island, sec_level_type::sec128);
    const llama::ParameterVerdict big =
        llama::inspect(pair.big, sec_level_type::sec128);

    EXPECT_TRUE(island.admissible()) << island.explain();
    EXPECT_TRUE(big.admissible()) << big.explain();
    EXPECT_EQ(island.log_qp, 108);
    EXPECT_EQ(big.log_qp, 1745);
    EXPECT_LE(big.log_qp, 1761);
    EXPECT_EQ(pair.island.usable_levels(), 1);

    // Both halves build, which is the claim -- composite security is the MIN
    // of the two rings and the secrets are tied, so there is no averaging.
    EXPECT_EQ(library_verdict(pair.island, sec_level_type::sec128), Build::Ok);
    EXPECT_EQ(library_verdict(pair.big, sec_level_type::sec128), Build::Ok);

    // And the big ring is long enough for the block's own chain.
    const heongpu::BootstrappingConfig boot;
    EXPECT_GE(pair.big.limbs(),
              llama::Llama3Batch16Operator::chain_limbs_for(17, boot));
}

TEST(Batch16Security, TheIslandPrimeSizeReachesIntoTheBigRingsBootstrap)
{
    // The ring switch requires the island's Q chain to be a value prefix of
    // the big one, so the island's 36-bit ceiling is also the big ring's q0.
    // That is the cost nobody had priced: q0 is what a bootstrap's scale
    // ratio is measured against, so the island's cap constrains the
    // refresh's precision at a ring sixteen times bigger.
    const llama::TwoRingPlan pair = llama::make_two_ring_plan(
        4096, 65536, 2, 43, 36, 36, 33, 1, 8, 36, 40);
    ASSERT_TRUE(pair.shares_prefix());
    EXPECT_EQ(pair.big.log_q.front(), 36);
    EXPECT_EQ(pair.island.log_q.front(), 36);

    // A big ring built with the 41-bit q0 that bootstrap v2 wants does NOT
    // share a prefix with any admissible island -- 41 is above the island's
    // 36-bit ceiling, so the two cannot both be had. That is the conflict,
    // and it is a parameter conflict rather than a coding one.
    llama::TwoRingPlan v2_wanted = pair;
    v2_wanted.big.log_q.front() = 41;
    EXPECT_FALSE(v2_wanted.shares_prefix());

    llama::ModulusPlan island_at_41 = pair.island;
    island_at_41.log_q.front() = 41;
    EXPECT_FALSE(
        llama::inspect(island_at_41, sec_level_type::sec128).admissible());
}

TEST(Batch16Security, APrefixThatIsNotAPrefixIsRejected)
{
    llama::TwoRingPlan pair =
        llama::make_two_ring_plan(4096, 65536, 2, 43, 36, 36, 33, 1, 8, 36, 40);
    ASSERT_TRUE(pair.shares_prefix());

    pair.island.log_q[1] = 35;
    EXPECT_FALSE(pair.shares_prefix());

    // An island longer than the big chain is not a prefix either, and the
    // builder refuses to make one.
    EXPECT_THROW(
        llama::make_two_ring_plan(4096, 65536, 44, 43, 36, 36, 33, 1, 8),
        std::invalid_argument);
}

// =====================================================================
// 6. The module's own contracts
// =====================================================================

TEST(Batch16Security, MaxLimbsRefusesToAnswerWithoutACap)
{
    // sec_level_type::none is not a weak cap; there is no longest chain to
    // report, and returning a number would be a lie a caller could quote.
    EXPECT_THROW(llama::max_limbs(4096, 33, 33, 1, 33, sec_level_type::none),
                 std::invalid_argument);
}

TEST(Batch16Security, MakePlanDefaultsTheSpecialToWhatThePairRuleNeeds)
{
    // Passing 0 asks for the smallest special that can clear the pair rule,
    // which is max(q0, prime): a group of `specials` Q primes then sums to
    // exactly the total P.
    const llama::ModulusPlan plan = llama::make_plan(65536, 10, 41, 33, 4);
    EXPECT_EQ(plan.log_p.size(), 4u);
    for (int p : plan.log_p)
    {
        EXPECT_EQ(p, 41);
    }
    EXPECT_TRUE(heongpu::coefficient_validator(plan.log_q, plan.log_p));

    EXPECT_THROW(llama::make_plan(4096, 0, 36, 36, 1), std::invalid_argument);
    EXPECT_THROW(llama::make_plan(4096, 2, 36, 36, 0), std::invalid_argument);
}

TEST(Batch16Security, TheCapsMatchTheLibraryTableAtEveryLegalRing)
{
    for (int n : {4096, 8192, 16384, 32768, 65536})
    {
        EXPECT_EQ(llama::security_cap(n, sec_level_type::sec128),
                  heongpu::heongpu_128bit_std_parms(static_cast<size_t>(n)));
        EXPECT_EQ(llama::security_cap(n, sec_level_type::sec192),
                  heongpu::heongpu_192bit_std_parms(static_cast<size_t>(n)));
        EXPECT_EQ(llama::security_cap(n, sec_level_type::sec256),
                  heongpu::heongpu_256bit_std_parms(static_cast<size_t>(n)));
        EXPECT_EQ(llama::security_cap(n, sec_level_type::none), 0);
    }
}

// =====================================================================
// 7. The chain is the security parameter, so shortening it is a security
//    lever -- and the refresh schedule is what sets it
// =====================================================================

namespace
{
    using Batch16 = llama::Llama3Batch16Operator;

    /// The measured level map of a batch-16 block at the block tests' cheap
    /// fit degrees (LLAMA3_8B_LAYER_FLOW.md section 25.4): RMSNorm 9,
    /// attention 17, residual 1, RMSNorm 9, SwiGLU + residual 10.
    ///
    /// The last figure is a MEASURED total and the split between the two
    /// SwiGLU positions is not measured, so it is a parameter here rather
    /// than a constant, and the tests that depend on the split say so.
    Batch16::RefreshPlanInput measured_map(int hidden_split)
    {
        Batch16::RefreshPlanInput in;
        in.stretch = {9, 17, 1, 9, 10 - hidden_split, hidden_split};
        // Five residual-width seams and one hidden-width one, at the real 8B
        // widths. Counting seams instead would understate the sixth by 3.5x.
        in.width = {4096, 4096, 4096, 4096, 4096, 14336};
        in.chain_limbs = 43;
        in.refresh_levels = 25;
        return in;
    }

    long long cost_of(const Batch16::Batch16RefreshConfig& config,
                      const std::vector<long long>& width)
    {
        long long total = 0;
        for (int i = 0; i < Batch16::seam_count; ++i)
        {
            if (config.at(i))
            {
                total += width[i];
            }
        }
        return config.enabled ? total : 0;
    }
} // namespace

TEST(Batch16Refresh, TheDefaultScheduleIsFeasibleAndAlmostTwiceTooExpensive)
{
    Batch16::RefreshPlanInput in = measured_map(5);
    in.steady_state = true;

    const Batch16::RefreshPlan plan = Batch16::plan_refresh(in);
    ASSERT_TRUE(plan.feasible);

    // The cheapest schedule that a REPEATING block can run: the entry seam
    // plus the two around the attention sublayer plus the feed-forward norm.
    EXPECT_EQ(plan.refreshes, 4);
    EXPECT_TRUE(plan.config.enabled);
    EXPECT_TRUE(plan.config.entry);
    EXPECT_TRUE(plan.config.after_attention_norm);
    EXPECT_TRUE(plan.config.after_attention);
    EXPECT_FALSE(plan.config.mid);
    EXPECT_TRUE(plan.config.after_feed_forward_norm);
    EXPECT_FALSE(plan.config.feed_forward_hidden);
    EXPECT_EQ(plan.bootstraps, 4LL * 4096);

    // Against the header defaults, which are five seams including the widest.
    Batch16::Batch16RefreshConfig defaults;
    defaults.enabled = true;
    const long long default_cost = cost_of(defaults, in.width);
    EXPECT_EQ(default_cost, 4LL * 4096 + 14336);
    EXPECT_LT(plan.bootstraps, default_cost);
    EXPECT_NEAR(1.0 - static_cast<double>(plan.bootstraps) /
                          static_cast<double>(default_cost),
                0.467, 0.005);

    // The binding stretch is the attention sublayer, and no seam splits it,
    // so the chain the plan needs is the chain the block already asked for.
    EXPECT_EQ(plan.worst_run, 17);
    const heongpu::BootstrappingConfig boot;
    EXPECT_EQ(Batch16::chain_limbs_for(plan.worst_run, boot), in.chain_limbs);
}

TEST(Batch16Refresh, AFirstBlockAndARepeatingBlockWantDifferentSchedules)
{
    // This is the distinction that makes a measured schedule dangerous to
    // reuse. A block handed a fresh 43-limb chain has 42 levels before its
    // first refresh; a block in the middle of a stack has whatever the
    // previous one left, which here is seven.
    Batch16::RefreshPlanInput fresh = measured_map(5);
    fresh.steady_state = false;

    const Batch16::RefreshPlan first = Batch16::plan_refresh(fresh);
    ASSERT_TRUE(first.feasible);
    EXPECT_LT(first.bootstraps, 4LL * 4096);
    EXPECT_FALSE(first.config.entry);

    // And that schedule does NOT survive being repeated: the limbs it hands
    // on are fewer than the limbs its own prefix run needs.
    Batch16::RefreshPlanInput again = measured_map(5);
    again.steady_state = false;
    again.entry_limbs = first.exit_limbs;
    const Batch16::RefreshPlan second = Batch16::plan_refresh(again);
    ASSERT_TRUE(second.feasible);
    EXPECT_NE(second.config.entry, first.config.entry)
        << "a block handed " << first.exit_limbs
        << " limbs must refresh on the way in, and one handed a fresh chain "
           "need not -- if these agree the entry budget is being ignored";

    // The steady-state plan is a fixed point by construction, which is the
    // property the fresh-entry plan lacks.
    Batch16::RefreshPlanInput steady = measured_map(5);
    steady.steady_state = true;
    const Batch16::RefreshPlan repeating = Batch16::plan_refresh(steady);
    ASSERT_TRUE(repeating.feasible);
    EXPECT_TRUE(repeating.config.entry);

    Batch16::RefreshPlanInput check = measured_map(5);
    check.steady_state = false;
    check.entry_limbs = repeating.exit_limbs;
    const Batch16::RefreshPlan rerun = Batch16::plan_refresh(check);
    ASSERT_TRUE(rerun.feasible);
    EXPECT_GE(rerun.exit_limbs, repeating.exit_limbs);
}

TEST(Batch16Refresh, ThePlanIsNeverWorseThanTheDefaultAtAnySwiGLUSplit)
{
    // The split of the measured 10-level SwiGLU stretch between the two
    // positions is not measured, so the conclusion must not depend on it.
    Batch16::Batch16RefreshConfig defaults;
    defaults.enabled = true;

    for (int split = 0; split <= 10; ++split)
    {
        Batch16::RefreshPlanInput in = measured_map(split);
        in.steady_state = true;
        const Batch16::RefreshPlan plan = Batch16::plan_refresh(in);

        // The search ranges over every subset, the default among them, so a
        // feasible default implies a feasible plan and the plan's cost is a
        // lower bound on the default's.
        ASSERT_TRUE(plan.feasible) << "split " << split;
        EXPECT_LE(plan.bootstraps, cost_of(defaults, in.width))
            << "split " << split;
        EXPECT_EQ(plan.worst_run, 17) << "split " << split;
    }
}

TEST(Batch16Refresh, AChainTooShortIsReportedRatherThanThrownLater)
{
    // The attention sublayer is 17 levels and no seam splits it, so a chain
    // that cannot hold 17 after a refresh has no feasible schedule at all --
    // and saying so is the point. Without this a caller discovers it as a
    // level underflow inside the SoftMax.
    Batch16::RefreshPlanInput in = measured_map(5);
    in.chain_limbs = 40; // 40 - 25 = 15 limbs back, 14 spendable, 17 needed
    in.steady_state = true;
    const Batch16::RefreshPlan plan = Batch16::plan_refresh(in);
    EXPECT_FALSE(plan.feasible);

    // One limb more than the stretch needs, and it becomes feasible.
    in.chain_limbs = Batch16::chain_limbs_for(17, heongpu::BootstrappingConfig());
    EXPECT_EQ(in.chain_limbs, 43);
    EXPECT_TRUE(Batch16::plan_refresh(in).feasible);
}

TEST(Batch16Refresh, TheWidthsAreWhatDecideTheAnswer)
{
    // Counting seams and counting bootstraps do not agree here, and the
    // disagreement is the whole reason the widths are an input: a schedule
    // with fewer seams can cost more ciphertexts when one of them is the
    // 14,336-column hidden seam.
    Batch16::RefreshPlanInput weighted = measured_map(5);
    weighted.steady_state = true;
    const Batch16::RefreshPlan by_cost = Batch16::plan_refresh(weighted);
    ASSERT_TRUE(by_cost.feasible);

    Batch16::RefreshPlanInput unweighted = weighted;
    unweighted.width.clear();
    const Batch16::RefreshPlan by_seams = Batch16::plan_refresh(unweighted);
    ASSERT_TRUE(by_seams.feasible);

    // Whatever each picks, the weighted plan must not be beaten on
    // ciphertexts by the seam-counting one.
    EXPECT_LE(cost_of(by_cost.config, weighted.width),
              cost_of(by_seams.config, weighted.width));
    EXPECT_LE(by_seams.refreshes, by_cost.refreshes);
}

TEST(Batch16Refresh, TheSeamIndexAndTheSeamNamesAgree)
{
    // plan_refresh writes through set(); transformer_block reads the named
    // fields. If those two ever disagree the plan is applied to the wrong
    // seams and nothing complains, so the correspondence is asserted.
    for (int i = 0; i < Batch16::seam_count; ++i)
    {
        Batch16::Batch16RefreshConfig config;
        config.enabled = true;
        for (int j = 0; j < Batch16::seam_count; ++j)
        {
            config.set(j, false);
        }
        config.set(i, true);
        EXPECT_EQ(config.count(), 1) << i;
        for (int j = 0; j < Batch16::seam_count; ++j)
        {
            EXPECT_EQ(config.at(j), i == j) << i << " vs " << j;
        }
        EXPECT_NE(Batch16::seam_name(i), nullptr);
    }

    EXPECT_EQ(std::string(Batch16::seam_name(Batch16::seam_entry)),
              "block.entry");
    EXPECT_EQ(std::string(Batch16::seam_name(Batch16::seam_after_attention)),
              "block.after_attention");
    EXPECT_EQ(
        std::string(Batch16::seam_name(Batch16::seam_feed_forward_hidden)),
        "feed_forward.hidden");

    EXPECT_THROW(Batch16::seam_name(Batch16::seam_count), std::out_of_range);
    Batch16::Batch16RefreshConfig config;
    EXPECT_THROW(config.at(-1), std::out_of_range);
    EXPECT_THROW(config.set(Batch16::seam_count, true), std::out_of_range);
}

TEST(Batch16Refresh, ThePlannerRejectsMalformedInput)
{
    Batch16::RefreshPlanInput in = measured_map(5);
    in.stretch.pop_back();
    EXPECT_THROW(Batch16::plan_refresh(in), std::invalid_argument);

    in = measured_map(5);
    in.width.pop_back();
    EXPECT_THROW(Batch16::plan_refresh(in), std::invalid_argument);

    in = measured_map(5);
    in.stretch[2] = -1;
    EXPECT_THROW(Batch16::plan_refresh(in), std::invalid_argument);

    in = measured_map(5);
    in.chain_limbs = -1;
    EXPECT_THROW(Batch16::plan_refresh(in), std::invalid_argument);
}

TEST(Batch16Security, TheSparseSecretCaveatIsCarried)
{
    // A cleared modulus budget is half a 128-bit claim. This exists so that a
    // report cannot quietly omit the other half.
    const std::string caveat = llama::sparse_secret_caveat();
    EXPECT_NE(caveat.find("ternary"), std::string::npos);
    EXPECT_NE(caveat.find("not"), std::string::npos);
}
