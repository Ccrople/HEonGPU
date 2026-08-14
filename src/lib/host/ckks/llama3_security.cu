// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_security.cuh>
#include <heongpu/kernel/defines.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            bool power_of_two(int v)
            {
                return v > 0 && (v & (v - 1)) == 0;
            }

            bool sizes_in_range(const std::vector<int>& bits)
            {
                for (int b : bits)
                {
                    if (b < MIN_USER_DEFINED_MOD_BIT_COUNT ||
                        b > MAX_USER_DEFINED_MOD_BIT_COUNT)
                    {
                        return false;
                    }
                }
                return true;
            }

            /// The five rings this family can use, smallest first. Not a
            /// preference order -- it is every power of two the library
            /// admits, and the batch encoding turns each into one batch size.
            const std::vector<int>& legal_rings()
            {
                static const std::vector<int> rings{4096, 8192, 16384, 32768,
                                                    65536};
                return rings;
            }
        } // namespace

        int ModulusPlan::log_qp() const noexcept
        {
            int total = 0;
            for (int b : log_q)
            {
                total += b;
            }
            for (int b : log_p)
            {
                total += b;
            }
            return total;
        }

        keyswitching_type ModulusPlan::method() const noexcept
        {
            if (log_p.empty())
            {
                return keyswitching_type::NONE;
            }
            return log_p.size() == 1 ? keyswitching_type::KEYSWITCHING_METHOD_I
                                     : keyswitching_type::KEYSWITCHING_METHOD_II;
        }

        double ParameterVerdict::cap_ratio() const noexcept
        {
            if (cap <= 0)
            {
                return 0.0;
            }
            return static_cast<double>(log_qp) / static_cast<double>(cap);
        }

        std::string ParameterVerdict::explain() const
        {
            std::ostringstream out;
            if (!ring_supported)
            {
                out << "ring " << n
                    << " is not a power of two in [" << MIN_POLY_DEGREE << ", "
                    << MAX_POLY_DEGREE << "]\n";
            }
            if (!primes_in_range)
            {
                out << "a prime bit size is outside ["
                    << MIN_USER_DEFINED_MOD_BIT_COUNT << ", "
                    << MAX_USER_DEFINED_MOD_BIT_COUNT
                    << "], or the special primes are empty\n";
            }
            if (!passes_pair_rule)
            {
                out << "the pair rule rejects it: with this many special "
                       "primes, a consecutive group of Q primes outweighs the "
                       "whole of P (coefficient_validator)\n";
            }
            if (!within_cap)
            {
                out << "log QP = " << log_qp << " against a cap of " << cap
                    << " at ring " << n << " -- " << cap_ratio() << "x over\n";
            }
            return out.str();
        }

        int security_cap(int n, sec_level_type level)
        {
            const size_t degree = static_cast<size_t>(n);
            switch (level)
            {
                case sec_level_type::none:
                    return 0;
                case sec_level_type::sec128:
                    return heongpu_128bit_std_parms(degree);
                case sec_level_type::sec192:
                    return heongpu_192bit_std_parms(degree);
                case sec_level_type::sec256:
                    return heongpu_256bit_std_parms(degree);
                default:
                    throw std::invalid_argument("Invalid security level");
            }
        }

        ParameterVerdict inspect(const ModulusPlan& plan, sec_level_type level)
        {
            ParameterVerdict v;
            v.n = plan.n;
            v.log_qp = plan.log_qp();
            v.limbs = plan.limbs();
            v.cap = security_cap(plan.n, level);

            v.ring_supported = power_of_two(plan.n) &&
                               plan.n >= MIN_POLY_DEGREE &&
                               plan.n <= MAX_POLY_DEGREE;

            v.primes_in_range = !plan.log_q.empty() && !plan.log_p.empty() &&
                                sizes_in_range(plan.log_q) &&
                                sizes_in_range(plan.log_p);

            // The library's own predicate, called rather than restated. It
            // divides by P_size, so an empty P is guarded here instead of
            // there.
            v.passes_pair_rule =
                !plan.log_p.empty() && !plan.log_q.empty() &&
                coefficient_validator(plan.log_q, plan.log_p);

            // sec_level_type::none is not a weaker cap, it is no cap: the
            // context skips the comparison entirely. Reporting "within" is
            // therefore accurate and cap_ratio() reports 0, which is what
            // stops a none-level verdict from being quoted as a ratio.
            v.within_cap =
                (level == sec_level_type::none) || (v.log_qp <= v.cap);

            return v;
        }

        ModulusPlan make_plan(int n, int limbs, int q0_bits, int prime_bits,
                              int specials, int special_bits)
        {
            if (limbs <= 0)
            {
                throw std::invalid_argument(
                    "A chain has at least one Q prime");
            }
            if (specials <= 0)
            {
                throw std::invalid_argument(
                    "The context rejects an empty special prime set");
            }
            if (special_bits == 0)
            {
                // The smallest uniform special that can clear the pair rule:
                // a group of `specials` Q primes then sums to at most
                // specials * special_bits, which is exactly the total P.
                special_bits = std::max(q0_bits, prime_bits);
            }

            ModulusPlan plan;
            plan.n = n;
            plan.log_q.push_back(q0_bits);
            plan.log_q.insert(plan.log_q.end(), limbs - 1, prime_bits);
            plan.log_p.assign(specials, special_bits);
            return plan;
        }

        int max_limbs(int n, int q0_bits, int prime_bits, int specials,
                      int special_bits, sec_level_type level)
        {
            if (level == sec_level_type::none)
            {
                throw std::invalid_argument(
                    "sec_level_type::none imposes no cap, so there is no "
                    "longest admissible chain to report");
            }

            const int cap = security_cap(n, level);
            if (cap <= 0)
            {
                return 0;
            }
            // No chain can be longer than the cap allows at the smallest legal
            // prime, so the search is bounded by the parameters themselves
            // rather than by a constant.
            const int bound = cap / MIN_USER_DEFINED_MOD_BIT_COUNT + 1;

            int best = 0;
            for (int limbs = 1; limbs <= bound; ++limbs)
            {
                const ModulusPlan plan =
                    make_plan(n, limbs, q0_bits, prime_bits, specials,
                              special_bits);
                if (inspect(plan, level).admissible())
                {
                    best = limbs;
                }
                else if (best > 0)
                {
                    // Admissibility is monotone in the limb count at fixed
                    // prime sizes -- each extra limb only adds to log QP and
                    // adds a group no heavier than the first -- so the first
                    // failure after a success is the end of the run.
                    break;
                }
            }
            return best;
        }

        int ring_for_batch(int batch, int head_dim)
        {
            if (batch <= 0 || head_dim <= 0)
            {
                throw std::invalid_argument(
                    "The batch size and the head dimension are positive");
            }
            return 2 * batch * head_dim;
        }

        int batch_for_ring(int n, int head_dim)
        {
            if (n <= 0 || head_dim <= 0)
            {
                throw std::invalid_argument(
                    "The ring degree and the head dimension are positive");
            }
            const int denom = 2 * head_dim;
            if (n % denom != 0)
            {
                throw std::invalid_argument(
                    "This ring carries no whole number of batch instances at "
                    "this head dimension");
            }
            return n / denom;
        }

        int smallest_batch_for(int limbs, int head_dim, int q0_bits,
                               int prime_bits, int specials, int special_bits,
                               sec_level_type level)
        {
            if (limbs <= 0)
            {
                throw std::invalid_argument(
                    "A chain has at least one Q prime");
            }
            for (int n : legal_rings())
            {
                if (n % (2 * head_dim) != 0)
                {
                    continue;
                }
                if (max_limbs(n, q0_bits, prime_bits, specials, special_bits,
                              level) >= limbs)
                {
                    return batch_for_ring(n, head_dim);
                }
            }
            return 0;
        }

        bool TwoRingPlan::shares_prefix() const
        {
            if (island.log_q.empty() || big.log_q.empty())
            {
                return false;
            }
            if (island.log_q.size() > big.log_q.size())
            {
                return false;
            }
            return std::equal(island.log_q.begin(), island.log_q.end(),
                              big.log_q.begin());
        }

        int TwoRingPlan::ratio() const
        {
            if (island.n <= 0 || big.n <= 0 || big.n % island.n != 0)
            {
                return 0;
            }
            return big.n / island.n;
        }

        TwoRingPlan make_two_ring_plan(int island_n, int big_n,
                                       int island_limbs, int big_limbs,
                                       int q0_bits, int island_prime_bits,
                                       int big_prime_bits, int island_specials,
                                       int big_specials,
                                       int island_special_bits,
                                       int big_special_bits)
        {
            if (island_limbs > big_limbs)
            {
                throw std::invalid_argument(
                    "The island's chain is a prefix of the big one, so it "
                    "cannot be longer");
            }
            TwoRingPlan pair;
            pair.island =
                make_plan(island_n, island_limbs, q0_bits, island_prime_bits,
                          island_specials, island_special_bits);

            // Built by extension rather than rebuilt, so that the prefix the
            // ring switch demands cannot drift apart from the island it is
            // supposed to be a prefix of.
            pair.big.n = big_n;
            pair.big.log_q = pair.island.log_q;
            pair.big.log_q.insert(pair.big.log_q.end(),
                                  big_limbs - island_limbs, big_prime_bits);
            const int big_p = big_special_bits == 0
                                  ? std::max(q0_bits, std::max(island_prime_bits,
                                                               big_prime_bits))
                                  : big_special_bits;
            if (big_specials <= 0)
            {
                throw std::invalid_argument(
                    "The context rejects an empty special prime set");
            }
            pair.big.log_p.assign(big_specials, big_p);
            return pair;
        }

        double BlockSecurity::over_by() const noexcept
        {
            if (admissible_limbs <= 0)
            {
                return 0.0;
            }
            if (chain_limbs <= admissible_limbs)
            {
                return 0.0;
            }
            return static_cast<double>(chain_limbs) /
                   static_cast<double>(admissible_limbs);
        }

        BlockSecurity audit_block(int batch, int head_dim, int chain_limbs,
                                  int refresh_levels, int q0_bits,
                                  int prime_bits, int specials,
                                  int special_bits, sec_level_type level)
        {
            if (chain_limbs <= 0)
            {
                throw std::invalid_argument(
                    "A schedule needs at least one limb");
            }
            if (refresh_levels < 0)
            {
                throw std::invalid_argument(
                    "A refresh spends a non-negative number of levels");
            }

            BlockSecurity report;
            report.chain_limbs = chain_limbs;
            report.batch = batch;
            report.head_dim = head_dim;
            report.n = ring_for_batch(batch, head_dim);
            report.cap = security_cap(report.n, level);
            report.admissible_limbs = max_limbs(report.n, q0_bits, prime_bits,
                                                specials, special_bits, level);
            report.single_ring =
                report.admissible_limbs >= report.chain_limbs;
            report.smallest_batch =
                smallest_batch_for(chain_limbs, head_dim, q0_bits, prime_bits,
                                   specials, special_bits, level);

            // A refresh must be handed a ciphertext with a prime left and must
            // return one with a level to spend, so the shortest chain that can
            // carry one at all is its own cost plus two.
            report.refresh_limbs = refresh_levels + 2;
            report.ring_can_refresh =
                report.admissible_limbs >= report.refresh_limbs;

            return report;
        }

        const char* sparse_secret_caveat()
        {
            return "The cap cleared here is the MODULUS budget only. "
                   "heongpu_128bit_std_parms is tabulated for a uniform "
                   "ternary secret; the CKKS secret key defaults to hamming "
                   "weight n/2 and the bootstrapping drivers use sparser keys "
                   "still, neither of which that table models. A cleared "
                   "verdict is necessary for a 128-bit claim and not "
                   "sufficient for one.";
        }

    } // namespace llama
} // namespace heongpu
