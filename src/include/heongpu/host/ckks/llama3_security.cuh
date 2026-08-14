// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Is a Llama-3 parameter set ADMISSIBLE? -- the library's own rules, in one
// place, so that a schedule can be priced against them without building a
// context and without a GPU.
//
// WHY THIS FILE EXISTS
// --------------------
// Every measurement any Llama-3 module here has produced was taken at
// sec_level_type::none. That is the right default for a cost model and the
// wrong one for a claim, and the two have been mixed up more than once in this
// tree. The check that decides the matter lives at context.cu:113 and :223 and
// runs only when the caller asks for a security level, i.e. never, in any test
// on any of these paths.
//
// So the question "does the batch-16 block satisfy the security" has never
// been asked of the library. This module asks it, in a form a host-side test
// can assert, and it asks it with the library's OWN predicates rather than a
// paraphrase of them -- inspect() calls coefficient_validator() and
// heongpu_128bit_std_parms() directly, so this module cannot drift from the
// context it is describing.
//
// THREE RULES, NOT ONE
// --------------------
// A parameter set is admissible only if it clears all three. Section 24 of
// LLAMA3_8B_LAYER_FLOW.md checked the second and missed the first, which is
// why its headline island -- Q = {41, 33}, P = {33}, "107 <= 109, legal by two
// bits" -- does not build:
//
//   1. THE PAIR RULE, coefficient_validator (util.cu:11). With P_size special
//      primes, every consecutive group of P_size Q primes must sum to no more
//      than the TOTAL P bit count. At P_size = 1 that degenerates to
//      "every individual q_i <= P", so a 41-bit q0 over a 33-bit special is
//      rejected outright -- before security is even considered. Raising P to
//      41 then costs 8 bits of the very budget the set was two bits inside.
//   2. THE SECURITY CAP, heongpu_128bit_std_parms (secstdparams.h:25).
//      sum(log Q) + sum(log P) <= cap(N). 109 at N = 4096.
//   3. THE PRIME RANGE, generate_primes (util.cu:253). Every bit size must lie
//      in [MIN_USER_DEFINED_MOD_BIT_COUNT, MAX_USER_DEFINED_MOD_BIT_COUNT] =
//      [30, 60], and N itself in [MIN_POLY_DEGREE, MAX_POLY_DEGREE].
//
// Rules 1 and 2 pull against each other, and that is the whole content of the
// small-ring result. Rule 1 forces P to be at least as big as the widest Q
// prime it covers; rule 2 counts P against the same budget as Q. At P_size = 1
// and uniform w-bit primes the two together give
//
//     (limbs + 1) * w <= cap(N),      w >= 30,
//
// so N = 4096 admits at most floor(109/30) - 1 = 2 Q primes, and only at
// w <= 36 -- ONE usable multiplicative level, at a 36-bit scale ceiling. Three
// Q primes would need 4 * 30 = 120 > 109 and there is no arrangement of prime
// sizes that avoids it. Method II is worse, not better: two specials put
// q0 + q1 <= P0 + P1 with everything at least 30, i.e. 120 bits minimum.
//
// WHAT THAT MEANS FOR THE BATCH AXIS
// ----------------------------------
// Kang's batch encoding pins d = head_dim and batch = k/2 with N = d*k, so
//
//     N = 2 * batch * head_dim
//
// and choosing the batch size chooses the ring, hence the cap. The batch axis
// is the security parameter (section 24.1). ring_for_batch and batch_for_ring
// are that identity; smallest_batch_for inverts the ladder -- given a chain
// length, which is the smallest batch whose ring will carry it.
//
// The two numbers that matter for a whole block:
//
//   * A regular bootstrap spends CtoS + taylor + StoC + 8 = 25 levels at the
//     default configuration, so ANY ring that refreshes needs >= 26 limbs.
//     At 33-bit primes that first happens at N = 65536, i.e. batch 256.
//   * The measured batch-16 block wants a 43-limb chain at cheap fits and 53
//     at production fits (section 25). Only N = 65536 admits either.
//
// So a single-ring batch-16 block has no 128-bit parameter set, by a factor of
// about twenty, and the gap is not a tuning matter. What closes it is the
// two-ring structure of section 24.5: Algorithm 4 at the island, everything
// else -- including the refresh -- at a ring big enough to hold a chain.
// make_two_ring_plan builds such a pair and enforces the constraint the ring
// switch actually imposes, which is that the island's Q chain be a value
// prefix of the big one (ringswitch.cuh:83). That prefix is why the island's
// prime size is not a free choice: it sets the big ring's q0 too, and q0 is
// what the bootstrap's scale ratio is measured against.
//
// WHAT THIS MODULE DOES NOT CLAIM
// -------------------------------
// Clearing all three rules clears the MODULUS budget of a 128-bit set. It does
// not clear the SECRET distribution: heongpu_128bit_std_parms is tabulated for
// a uniform ternary secret, and every driver in this tree that bootstraps uses
// a sparse one -- Secretkey(context, hamming_weight). A verdict from here is
// therefore necessary and not sufficient, and sparse_secret_caveat() exists so
// that a report cannot quietly omit it.

#ifndef HEONGPU_CKKS_LLAMA3_SECURITY_H
#define HEONGPU_CKKS_LLAMA3_SECURITY_H

#include <heongpu/util/schemes.h>
#include <heongpu/util/secstdparams.h>
#include <heongpu/util/util.cuh>

#include <string>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief A CKKS parameter set in exactly the form the context takes.
         *
         * Bit sizes, not values: the security cap, the pair rule and the prime
         * range are all decided on bit sizes, and generate_primes turns sizes
         * into values afterwards. So a plan can be judged with no GPU, no
         * context and no prime search.
         */
        struct ModulusPlan
        {
            /// Ring degree. Must be a power of two in [4096, 65536].
            int n = 0;
            /// Q prime bit sizes, bottom first: log_q[0] is q0, the prime a
            /// fully consumed ciphertext is left with.
            std::vector<int> log_q;
            /// Special (P) prime bit sizes. Never empty in a usable plan --
            /// the context rejects an empty P outright.
            std::vector<int> log_p;

            /// sum(log Q) + sum(log P), the quantity the cap bounds.
            int log_qp() const noexcept;
            /// Q primes, i.e. limbs at the top of the chain.
            int limbs() const noexcept { return static_cast<int>(log_q.size()); }
            /// Multiplicative levels a fresh ciphertext can spend. One fewer
            /// than the limbs, because the last prime is not spendable.
            int usable_levels() const noexcept
            {
                return limbs() > 0 ? limbs() - 1 : 0;
            }
            /// What the context will select from this P count, by the same
            /// rule it uses (context.cu:65): one special is method I.
            keyswitching_type method() const noexcept;

            bool empty() const noexcept { return log_q.empty(); }
        };

        /**
         * @brief Why a plan is, or is not, admissible.
         *
         * Every flag is one of the library's own predicates, evaluated by
         * calling it rather than by restating it.
         */
        struct ParameterVerdict
        {
            int n = 0;
            int log_qp = 0;
            /// The cap for @c n at the requested level; 0 when the level is
            /// none, and also 0 when @c n is not in the table.
            int cap = 0;
            int limbs = 0;

            /// n is a power of two in [MIN_POLY_DEGREE, MAX_POLY_DEGREE].
            bool ring_supported = false;
            /// Every Q and P bit size is in [30, 60], and P is non-empty.
            bool primes_in_range = false;
            /// coefficient_validator, called.
            bool passes_pair_rule = false;
            /// log_qp <= cap, or the level is none.
            bool within_cap = false;

            bool admissible() const noexcept
            {
                return ring_supported && primes_in_range && passes_pair_rule &&
                       within_cap;
            }

            /// How far over the cap the set is, as a multiple. 0 when there is
            /// no cap to be over.
            double cap_ratio() const noexcept;

            /// One line per failed rule, in the order the context checks them.
            /// Empty when the plan is admissible.
            std::string explain() const;
        };

        /**
         * @brief The cap the library would apply at this ring and level.
         *
         * @return 0 for sec_level_type::none (no cap), and 0 for a ring the
         *         table does not cover -- which is itself a failure, and is
         *         reported through ParameterVerdict::ring_supported rather
         *         than here.
         */
        int security_cap(int n, sec_level_type level);

        /**
         * @brief Judge a plan against all three of the library's rules.
         *
         * Does not build a context and does not allocate on the device, so it
         * is usable in a test that cannot get a GPU. The tests assert that its
         * verdict and the context's behaviour agree, in both directions.
         */
        ParameterVerdict inspect(const ModulusPlan& plan, sec_level_type level);

        /**
         * @brief Q = {q0_bits, prime_bits x (limbs-1)}, P = specials x
         *        special_bits.
         *
         * @param special_bits 0 asks for the smallest uniform special that can
         *        clear the pair rule, which is max(q0_bits, prime_bits): a
         *        group of @p specials Q primes then sums to at most
         *        specials * special_bits, exactly the total P. Passing a value
         *        explicitly is how a caller prices a P that is deliberately
         *        larger (method-II noise) or deliberately smaller (and finds
         *        out that it does not build).
         *
         * @throws std::invalid_argument on a non-positive limb or special
         *         count. Bit sizes are NOT validated here -- an out-of-range
         *         size is a verdict, not an exception, because the point is to
         *         be able to ask about one.
         */
        ModulusPlan make_plan(int n, int limbs, int q0_bits, int prime_bits,
                              int specials, int special_bits = 0);

        /**
         * @brief The longest chain make_plan can build admissibly at @p n.
         *
         * Searched, not solved, and searched by calling inspect(), so it
         * agrees with the library by construction rather than by algebra. The
         * closed form for the uniform case is floor(cap/w) - specials, and the
         * tests check the search against it.
         *
         * @return 0 when no chain of any length is admissible.
         */
        int max_limbs(int n, int q0_bits, int prime_bits, int specials,
                      int special_bits, sec_level_type level);

        // -------------------------------------------------------------------
        // The batch axis IS the ring axis
        // -------------------------------------------------------------------

        /**
         * @brief N = 2 * batch * head_dim.
         *
         * Kang's encoding sets batch = k/2 and d = N/k, and Algorithm 4 needs
         * d to divide head_dim, so at the Llama-3 head dim of 128 the batch
         * size and the ring are the same knob. This is the identity section
         * 24.1 is built on, in code, so nothing has to rederive it.
         *
         * @throws std::invalid_argument on a non-positive argument.
         */
        int ring_for_batch(int batch, int head_dim);

        /** @brief The inverse, batch = N / (2 * head_dim). */
        int batch_for_ring(int n, int head_dim);

        /**
         * @brief The smallest batch in this family whose ring admits a chain
         *        of @p limbs.
         *
         * Walks the five legal rings upward. This is the ladder that answers
         * "what would it take to run this schedule securely" without changing
         * the schedule.
         *
         * @return 0 when no ring up to MAX_POLY_DEGREE admits it.
         */
        int smallest_batch_for(int limbs, int head_dim, int q0_bits,
                               int prime_bits, int specials, int special_bits,
                               sec_level_type level);

        // -------------------------------------------------------------------
        // Two rings
        // -------------------------------------------------------------------

        /**
         * @brief An island and a big ring, as the ring switch requires them.
         *
         * Composite security is the MIN of the two rings and the secrets are
         * tied by s_small(X) = s_big(X^k), so BOTH halves must be admissible
         * on their own -- there is no averaging.
         */
        struct TwoRingPlan
        {
            ModulusPlan island;
            ModulusPlan big;

            /**
             * @brief Does the island's Q chain prefix the big one's?
             *
             * HERingSwitchOperator requires the small context's Q values to be
             * a value-identical prefix of the big context's
             * (ringswitch.cuh:83). Bit sizes agreeing is necessary, not
             * sufficient -- the values must be taken verbatim through
             * set_coeff_modulus_values -- but a bit-size mismatch rules the
             * pair out on inspection, which is what this catches.
             */
            bool shares_prefix() const;

            /// N_big / N_small, or 0 if that is not a positive integer.
            int ratio() const;
        };

        /**
         * @brief Build an island/big pair sharing a prefix by construction.
         *
         * The big chain is literally the island's Q vector followed by
         * @p big_limbs - @p island_limbs further primes, so shares_prefix() is
         * true by construction rather than by coincidence. The two prime sizes
         * are separate arguments because they have to be: the island's cap is
         * the tightest in the system and pushes its primes down, while the big
         * ring wants them as wide as its own cap allows.
         *
         * The consequence worth naming is that the ISLAND's prime size sets
         * the BIG ring's q0, and q0 is what a bootstrap's scale ratio is
         * measured against -- so the island's cap reaches all the way into the
         * precision of a refresh taken at a ring sixteen times bigger. That is
         * the real price of the two-ring structure, and it is not a cost any
         * level ledger shows.
         */
        TwoRingPlan make_two_ring_plan(int island_n, int big_n,
                                       int island_limbs, int big_limbs,
                                       int q0_bits, int island_prime_bits,
                                       int big_prime_bits, int island_specials,
                                       int big_specials,
                                       int island_special_bits = 0,
                                       int big_special_bits = 0);

        // -------------------------------------------------------------------
        // The block-shaped question
        // -------------------------------------------------------------------

        /**
         * @brief What a schedule of a given depth costs in security terms.
         *
         * Deliberately takes the chain length rather than measuring it: the
         * depth of a block is Llama3Batch16Operator::chain_limbs_for's
         * business and depends on the fit degrees, and this module should not
         * acquire an opinion about either.
         */
        struct BlockSecurity
        {
            /// The chain the schedule asked for.
            int chain_limbs = 0;
            int batch = 0;
            int head_dim = 0;
            int n = 0;
            int cap = 0;
            /// The longest chain this ring admits at the given prime sizes.
            int admissible_limbs = 0;
            /// chain_limbs <= admissible_limbs.
            bool single_ring = false;
            /// The smallest batch in the family that would carry it, 0 if
            /// none.
            int smallest_batch = 0;
            /// Levels a refresh spends on itself, hence limbs+1 a ring must
            /// hold before any work at all.
            int refresh_limbs = 0;
            /// Can THIS ring hold a refresh, never mind the work?
            bool ring_can_refresh = false;

            /// Multiple by which the requested chain overruns what is
            /// admissible. 0 when it does not overrun.
            double over_by() const noexcept;
        };

        BlockSecurity audit_block(int batch, int head_dim, int chain_limbs,
                                  int refresh_levels, int q0_bits,
                                  int prime_bits, int specials,
                                  int special_bits, sec_level_type level);

        /**
         * @brief The sentence a report must not omit.
         *
         * heongpu_128bit_std_parms is tabulated for a UNIFORM TERNARY secret.
         * Every bootstrapping driver in this tree uses a sparse one, so a
         * cleared modulus budget is half of a 128-bit claim and this is the
         * other half, unresolved.
         */
        const char* sparse_secret_caveat();

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_SECURITY_H
