// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// HOW SMALL A RING MAY THE BAE PRODUCT RUN ON? ASK THE LIBRARY.
// ==============================================================
//
// Sylph's Table IV runs the PCMM at ring degree 256 against 65536 for the rest
// of the pipeline, and reaches it by ring switching. Ring switching — unlike
// ModDecomp — moves the lattice dimension, because it key-switches to the
// embedded secret first and the pieces are then rank-1 under a genuine
// degree-n secret. So "how small a ring" is a security question with a
// performance answer attached, and this target refuses to answer it from a
// paper or from an arithmetic re-derivation of a table.
//
// It builds real contexts at sec_level_type::sec128 and reports which ones
// generate(). Everything downstream — the usable level count, the MAC cost,
// the verdict — is computed from what the library accepted.
//
// WHAT COUNTS AS USABLE
// ---------------------
// A Bae product at k = 1 needs no key switch of its own: no ModPack, no
// rotation, no relinearisation. So the small context needs
//
//   * the one special prime the library requires to exist, and
//   * at least TWO Q primes — one to carry the value and one for the
//     product's own rescale to spend.
//
// Two Q primes is one usable level, which is exactly what one projection
// costs. A ring that cannot hold two Q primes at 128-bit cannot host the
// product at all, whatever its arithmetic would have been.
//
// THE SECOND CONSTRAINT, WHICH IS NOT A SECURITY ONE
// --------------------------------------------------
// HERingSwitchOperator requires the small chain to be a value-identical
// PREFIX of the big chain, so the small ring must hold the big pipeline's
// bottom primes at the big pipeline's own bit sizes. A row that only clears
// the cap by shrinking the primes has changed the precision at the boundary,
// which is a correctness question and not a tuning one, so the sweep reports
// the largest prime size each ring accepts rather than only a yes/no.
//
//   HEONGPU_BRS_LOGN    comma-separated ring exponents   8,9,10,11,12,13,14,16
//   HEONGPU_BRS_MAXQ    highest Q prime count to try     8

#include <heongpu/heongpu.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;

namespace
{
    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }

    std::vector<int> EnvList(const char* name, const std::vector<int>& fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        std::vector<int> out;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty())
                out.push_back(std::atoi(item.c_str()));
        return out.empty() ? fallback : out;
    }

    /// Accepted by the library, at 128-bit, for real. Nothing here is derived.
    bool Accepted(int n, const std::vector<int>& q_bits,
                  const std::vector<int>& p_bits, std::string& why)
    {
        try
        {
            heongpu::HEContext<S> context =
                heongpu::GenHEContext<S>(heongpu::sec_level_type::sec128);
            context->set_poly_modulus_degree(static_cast<std::size_t>(n));
            context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
            context->generate();
            why.clear();
            return true;
        }
        catch (const std::exception& e)
        {
            why = e.what();
            return false;
        }
        catch (...)
        {
            why = "unknown failure";
            return false;
        }
    }

    /// Largest Q prime count this ring accepts at this prime size, with one
    /// special prime of the same size (the coefficient validator requires
    /// log P >= the largest Q prime when |P| = 1).
    int MaxQPrimes(int n, int bits, int cap, std::string& why)
    {
        int best = 0;
        for (int count = 1; count <= cap; ++count)
        {
            std::vector<int> q_bits(static_cast<size_t>(count), bits);
            std::vector<int> p_bits(1, bits);
            std::string reason;
            if (!Accepted(n, q_bits, p_bits, reason))
            {
                if (best == 0)
                    why = reason;
                break;
            }
            best = count;
            why.clear();
        }
        return best;
    }
} // namespace

int main()
{
    const std::vector<int> log_ns =
        EnvList("HEONGPU_BRS_LOGN", {8, 9, 10, 11, 12, 13, 14, 16});
    const int cap = EnvInt("HEONGPU_BRS_MAXQ", 8);
    const std::vector<int> prime_bits = {60, 50, 40, 30, 25, 20};

    std::cout << "[brs] every cell below is accept/reject by the library at "
                 "sec_level_type::sec128"
              << std::endl;
    std::cout << "[brs] cell = largest Q prime count accepted, with one "
                 "special prime of the same size"
              << std::endl;
    std::cout << "[brs] a Bae product needs >= 2 Q primes: one to carry the "
                 "value, one for its rescale"
              << std::endl;
    std::cout << "[brs] "
                 "---------------------------------------------------------"
              << std::endl;

    std::cout << "[brs]   ring |";
    for (int b : prime_bits)
        std::cout << std::setw(6) << ("2^" + std::to_string(b));
    std::cout << "  | verdict" << std::endl;

    // The smallest ring that hosts the product, per the library. Used by the
    // cost section below so the two halves cannot drift apart.
    int smallest_hosting_ring = 0;

    for (int log_n : log_ns)
    {
        const int n = 1 << log_n;
        std::cout << "[brs] " << std::setw(6) << n << " |";

        int best_bits = 0;
        int best_primes = 0;
        std::string first_reason;
        for (int b : prime_bits)
        {
            std::string why;
            const int primes = MaxQPrimes(n, b, cap, why);
            if (primes == 0 && first_reason.empty())
                first_reason = why;
            std::cout << std::setw(6) << primes;
            if (primes >= 2 && best_bits == 0)
            {
                best_bits = b;
                best_primes = primes;
            }
        }

        if (best_bits > 0)
        {
            if (smallest_hosting_ring == 0)
                smallest_hosting_ring = n;
            std::cout << "  | hosts the product: " << (best_primes - 1)
                      << " usable level(s) at 2^" << best_bits
                      << ", log PQ <= " << (best_primes + 1) * best_bits;
        }
        else
        {
            std::cout << "  | CANNOT host the product";
            if (!first_reason.empty())
                std::cout << " (" << first_reason << ")";
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // The question that actually decides it: not "some chain", but THE chain.
    //
    // HERingSwitchOperator requires the small Q to be a value-identical bottom
    // PREFIX of the big one, so the small ring inherits the pipeline's own
    // primes and cannot choose its own. §17.2 lays the logN 16 chain out as
    // q0 = 41 then 33-bit primes with P = 61, and the prefix therefore starts
    // at q0 = 41 whether the product wants it there or not. The special prime
    // must cover the largest Q chunk, which is that same q0, so |P| = 1 costs
    // 41 bits before a single level is bought.
    // -----------------------------------------------------------------------
    const int q0_bits = EnvInt("HEONGPU_BRS_Q0", 41);
    const int p_bits = EnvInt("HEONGPU_BRS_P", 33);

    std::cout << "[brs] "
                 "---------------------------------------------------------"
              << std::endl;
    std::cout << "[brs] the pipeline's own prefix: q0 = 2^" << q0_bits
              << ", then 2^" << p_bits << ", special prime 2^" << q0_bits
              << std::endl;

    int prefix_hosting_ring = 0;
    for (int log_n : log_ns)
    {
        const int n = 1 << log_n;
        int best = 0;
        std::string why;
        for (int l = 1; l <= cap; ++l)
        {
            std::vector<int> q(static_cast<size_t>(l), p_bits);
            q.front() = q0_bits;
            std::string reason;
            if (!Accepted(n, q, {q0_bits}, reason))
            {
                if (best == 0)
                    why = reason;
                break;
            }
            best = l;
        }
        std::cout << "[brs] " << std::setw(6) << n << " | " << best
                  << " prefix prime(s) => " << (best >= 2 ? best - 1 : 0)
                  << " usable level(s)";
        if (best >= 2)
        {
            if (prefix_hosting_ring == 0)
                prefix_hosting_ring = n;
            std::cout << "  <== hosts the product";
        }
        else if (!why.empty())
        {
            std::cout << "  (" << why << ")";
        }
        std::cout << std::endl;
    }
    std::cout << "[brs] smallest ring the PIPELINE'S OWN CHAIN can descend to: "
              << prefix_hosting_ring << std::endl;

    if (smallest_hosting_ring == 0)
    {
        std::cout << "[brs] no ring in the sweep hosts the product; widen "
                     "HEONGPU_BRS_LOGN"
                  << std::endl;
        return 0;
    }

    // -----------------------------------------------------------------------
    // What that costs, at the real 8B width and the real B = 1 prompt.
    // -----------------------------------------------------------------------
    const int d = 4096;      // Llama-3-8B d_model
    const int tokens = 128;  // B = 1
    const int n_big = 65536; // logN 16, the pipeline ring

    std::cout << "[brs] "
                 "---------------------------------------------------------"
              << std::endl;
    std::cout << "[brs] one projection, d1 = d2 = " << d << ", tokens = "
              << tokens << ", pipeline ring " << n_big << std::endl;
    std::cout << "[brs] MACs are per RNS limb; per-token is MACs / (d1*d2*"
                 "tokens), i.e. §25.9's unit"
              << std::endl;
    std::cout << "[brs]      path |  dim  | per-token |   a-part   |"
                 "   b-part   | key switches"
              << std::endl;

    const auto today =
        heongpu::HEBaeLowRingPcmm::big_ring_cost(d, d, tokens, n_big, 256);
    std::cout << "[brs]  big ring | " << std::setw(5) << today.dimension
              << " | " << std::setw(9) << std::fixed << std::setprecision(2)
              << today.per_token << " | " << std::setw(10) << today.a_macs
              << " | " << std::setw(10) << today.b_macs << " | "
              << today.key_switches << "  (ModPack)" << std::endl;

    for (int log_n = 8; log_n <= 16; ++log_n)
    {
        const int n = 1 << log_n;
        if (n > n_big || tokens > n)
            continue;
        const auto c =
            heongpu::HEBaeLowRingPcmm::low_ring_cost(d, d, tokens, n_big, n);
        const bool secure = n >= smallest_hosting_ring;
        std::cout << "[brs]   ring " << std::setw(5) << n << " | "
                  << std::setw(5) << c.dimension << " | " << std::setw(9)
                  << std::fixed << std::setprecision(2) << c.per_token << " | "
                  << std::setw(10) << c.a_macs << " | " << std::setw(10)
                  << c.b_macs << " | " << c.key_switches << "  (down+up)"
                  << (secure ? "" : "   <-- REJECTED at sec128") << std::endl;
    }

    const auto floor_cost = heongpu::HEBaeLowRingPcmm::low_ring_cost(
        d, d, tokens, n_big, smallest_hosting_ring);
    std::cout << "[brs] "
                 "---------------------------------------------------------"
              << std::endl;
    std::cout << "[brs] smallest ring the library accepts: "
              << smallest_hosting_ring << std::endl;
    std::cout << "[brs] speedup over the big-ring path at that ring: "
              << std::fixed << std::setprecision(2)
              << (today.per_token / floor_cost.per_token) << "x arithmetic, "
              << (static_cast<double>(today.key_switches) /
                  static_cast<double>(floor_cost.key_switches))
              << "x key switches" << std::endl;

    return 0;
}
