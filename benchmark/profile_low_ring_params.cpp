// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// WHAT DOES A LOW RING ACTUALLY HOLD, ACCORDING TO THE LIBRARY?
// =============================================================
//
// The Sylph-style design puts the batch CCMM on a small ring and everything
// else on a large one. Whether that is possible at all is a parameter question
// before it is a performance question, and it is decided by two things the
// library owns rather than by anything in a paper:
//
//   coefficient_validator (util.cu)  chunks Q into groups of |P| and requires
//                                    every chunk to be <= sum log P. With one
//                                    special prime that means log P >= the
//                                    LARGEST Q prime, so P is not a free
//                                    parameter -- it tracks the scale.
//
//   heongpu_128bit_std_parms         log(P*Q) <= 109 at N = 4096
//   (secstdparams.h)                            <= 218 at N = 8192
//                                               <= 1761 at N = 65536
//                                    and the bound is checked on the PRODUCT,
//                                    so the special primes come out of the
//                                    same budget as the chain.
//
// This target asks the library directly. It builds real contexts at
// sec_level_type::sec128 and reports which survive generate(), so the answer
// is the implementation's and not an arithmetic re-derivation of it.
//
// THE THIRD CONSTRAINT, WHICH IS THE ONE THAT USUALLY BITES
// ---------------------------------------------------------
// A low-ring island is only useful if what leaves it can be bootstrapped, and
// HEonGPU's CKKS bootstrap is built around q0 / scale ~ 2^10 -- outside that
// it returns noise rather than an error. So the row that matters is not "how
// many levels fit" but "how many levels fit with a bottom prime ten bits above
// the scale". Every candidate here is reported with that ratio so a row that
// technically validates but cannot feed a bootstrap is visible as such.
//
// WHAT ONE LEVEL HAS TO COVER
// ---------------------------
// Kang's Algorithm 4 is one multiplicative level and Algorithm 3 (the CMT) is
// zero, so a QK island needs exactly one level plus whatever the bridge in and
// out costs -- and on the fused path the bridge is the bootstrap's own
// CoeffToSlot, which is spent on the HIGH ring. So one usable level on the low
// ring is the real requirement.
//
//   HEONGPU_LRP_LOGN   comma-separated ring exponents   12,13
//   HEONGPU_LRP_MAXLV  highest level count to try       6

#include <heongpu/heongpu.hpp>

#include <algorithm>
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
        {
            if (!item.empty())
                out.push_back(std::atoi(item.c_str()));
        }
        return out.empty() ? fallback : out;
    }

    // Accepted by the library, at 128-bit, for real. Nothing here is derived.
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

    int Sum(const std::vector<int>& v)
    {
        int s = 0;
        for (int x : v)
            s += x;
        return s;
    }
} // namespace

int main()
{
    const std::vector<int> log_ns = EnvList("HEONGPU_LRP_LOGN", {12, 13});
    const int max_levels = EnvInt("HEONGPU_LRP_MAXLV", 6);

    std::cout << "[lrp] every row below is accept/reject by the library at "
                 "sec_level_type::sec128"
              << std::endl;

    for (int log_n : log_ns)
    {
        const int n = 1 << log_n;
        std::cout << "[lrp] "
                     "================================================"
                  << std::endl;
        std::cout << "[lrp] N = " << n << " (logN " << log_n << ")"
                  << std::endl;

        // ---------------------------------------------------------------
        // 1. The largest scale that holds a given number of levels, with
        //    one special prime and a bottom prime equal to the scale.
        // ---------------------------------------------------------------
        std::cout << "[lrp] -- plain chain, q0 = scale, |P| = 1 --"
                  << std::endl;
        for (int levels = 1; levels <= max_levels; ++levels)
        {
            int best = 0;
            for (int bits = 60; bits >= 20; --bits)
            {
                std::vector<int> q_bits(levels + 1, bits);
                std::vector<int> p_bits(1, bits);
                std::string why;
                if (Accepted(n, q_bits, p_bits, why))
                {
                    best = bits;
                    break;
                }
            }
            if (best == 0)
            {
                std::cout << "[lrp]   levels " << std::setw(2) << levels
                          << ": nothing down to 20 bits" << std::endl;
                continue;
            }
            std::vector<int> q_bits(levels + 1, best);
            std::cout << "[lrp]   levels " << std::setw(2) << levels
                      << ": scale 2^" << std::setw(2) << best << ", log PQ "
                      << std::setw(4) << (Sum(q_bits) + best) << std::endl;
        }

        // ---------------------------------------------------------------
        // 2. The row that can actually feed a bootstrap: q0 ten bits above
        //    the scale, which is the ratio HEonGPU's CKKS bootstrap is
        //    built around.
        // ---------------------------------------------------------------
        std::cout << "[lrp] -- bootstrap-compatible chain, q0 = scale + 10, "
                     "|P| = 1 --"
                  << std::endl;
        for (int levels = 1; levels <= max_levels; ++levels)
        {
            int best = 0;
            for (int bits = 55; bits >= 15; --bits)
            {
                std::vector<int> q_bits(levels + 1, bits);
                q_bits.front() = bits + 10;
                // P must cover the largest chunk, which is now q0.
                std::vector<int> p_bits(1, bits + 10);
                std::string why;
                if (Accepted(n, q_bits, p_bits, why))
                {
                    best = bits;
                    break;
                }
            }
            if (best == 0)
            {
                std::cout << "[lrp]   levels " << std::setw(2) << levels
                          << ": nothing down to 15 bits" << std::endl;
                continue;
            }
            const int total = (best + 10) + best * levels + (best + 10);
            std::cout << "[lrp]   levels " << std::setw(2) << levels
                      << ": scale 2^" << std::setw(2) << best << ", q0 2^"
                      << (best + 10) << ", log PQ " << std::setw(4) << total
                      << std::endl;
        }

        // ---------------------------------------------------------------
        // 3. What the rest of the pipeline runs at, for comparison. If the
        //    low ring cannot hold this scale the island has to change
        //    precision at the boundary, which is a correctness question
        //    and not a tuning one.
        // ---------------------------------------------------------------
        {
            std::vector<int> q_bits = {60, 50, 50};
            std::vector<int> p_bits = {60};
            std::string why;
            const bool ok = Accepted(n, q_bits, p_bits, why);
            std::cout << "[lrp] -- the pipeline's own precision (60, 50, 50 | "
                         "60): "
                      << (ok ? "ACCEPTED" : "REJECTED") << " --" << std::endl;
            if (!ok)
                std::cout << "[lrp]   " << why << std::endl;
        }
    }

    std::cout << "[lrp] "
                 "================================================"
              << std::endl;
    return 0;
}
