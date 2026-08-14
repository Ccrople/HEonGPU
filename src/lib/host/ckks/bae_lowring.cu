// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/bae_lowring.cuh>

#include <stdexcept>
#include <utility>

namespace heongpu
{
    namespace
    {
        bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }
    } // namespace

    HEBaeLowRingPcmm::HEBaeLowRingPcmm(HEContext<Scheme::CKKS>& big,
                                       HEContext<Scheme::CKKS>& small,
                                       HERingSwitchOperator<Scheme::CKKS>& rs)
        : big_(big), small_(small), rs_(rs)
    {
        // Both contexts being generated, and the small Q chain being a
        // value-identical prefix of the big one, are HERingSwitchOperator's
        // constructor checks. @p rs existing is therefore the proof of them,
        // and re-checking here would need a friendship this class has no
        // reason to hold.
        if (!big_ || !small_)
            throw std::invalid_argument("both contexts must be non-null");
        if (!rs.keys_generated())
            throw std::invalid_argument(
                "the ring switch must have its keys generated: the descent is "
                "a key switch and there is no keyless fallback for it");

        n_big_ = static_cast<int>(big_->get_poly_modulus_degree());
        n_small_ = static_cast<int>(small_->get_poly_modulus_degree());

        if (!is_power_of_two(n_big_) || !is_power_of_two(n_small_) ||
            n_small_ > n_big_)
            throw std::invalid_argument(
                "the small ring must be a power of two dividing the big one");

        k_ = n_big_ / n_small_;
        if (k_ != rs.k())
            throw std::invalid_argument(
                "the ring switch is bound to a different pair of contexts than "
                "the ones handed here");

        // Bae's d >= sqrt(N), applied to the BIG ring's row-split layout. The
        // constructor of HEBaePcmmOperator would raise this itself; raising it
        // here says which of the two rings is at fault.
        if (static_cast<long long>(n_small_) * n_small_ <
            static_cast<long long>(n_big_))
            throw std::invalid_argument(
                "the small ring is below sqrt(big ring), so one big-ring "
                "ciphertext does not hold a whole strip of rows in the "
                "row-split layout and the descent has nothing well formed to "
                "split");

        encoder_ =
            std::make_unique<HEBaePcmmOperator<Scheme::CKKS>>(big, n_small_);
        product_ =
            std::make_unique<HEBaePcmmOperator<Scheme::CKKS>>(small, n_small_);

        // k == 1 at the small ring, so the product is a linear combination of
        // whole ciphertexts and commutes with the NTT (§25.9). The descent
        // hands back NTT-domain ciphertexts and the ascent wants them back the
        // same way, so the round trip run_gemms would otherwise perform is
        // pure loss on both ends.
        product_->set_transform_free(true);
    }

    std::vector<Ciphertext<Scheme::CKKS>>
    HEBaeLowRingPcmm::descend(std::vector<Ciphertext<Scheme::CKKS>>& rows_big,
                              HEArithmeticOperator<Scheme::CKKS>& big_ops)
    {
        if (rows_big.empty())
            throw std::invalid_argument("no ciphertexts to descend");

        std::vector<Ciphertext<Scheme::CKKS>> out;
        out.reserve(rows_big.size() * static_cast<size_t>(k_));

        for (auto& ct : rows_big)
        {
            std::vector<Ciphertext<Scheme::CKKS>> pieces =
                rs_.switch_down(ct, big_ops);
            // Piece u of big ciphertext i is row k*i + u, whole. Appending in
            // (i, u) order therefore produces the rows in row order, which is
            // the order project() indexes the weight in.
            for (auto& piece : pieces)
                out.push_back(std::move(piece));
        }
        return out;
    }

    std::vector<Ciphertext<Scheme::CKKS>>
    HEBaeLowRingPcmm::ascend(std::vector<Ciphertext<Scheme::CKKS>>& rows_small,
                             HEArithmeticOperator<Scheme::CKKS>& big_ops)
    {
        const int rows = static_cast<int>(rows_small.size());
        if (rows == 0)
            throw std::invalid_argument("no ciphertexts to ascend");
        if (rows % k_ != 0)
            throw std::invalid_argument(
                "the row count must be a multiple of k: a big-ring ciphertext "
                "in the row-split layout carries exactly k rows and there is "
                "no partial one");

        std::vector<Ciphertext<Scheme::CKKS>> out;
        out.reserve(static_cast<size_t>(rows / k_));

        for (int g = 0; g < rows / k_; ++g)
        {
            std::vector<Ciphertext<Scheme::CKKS>> group;
            group.reserve(static_cast<size_t>(k_));
            for (int u = 0; u < k_; ++u)
                group.push_back(rows_small[static_cast<size_t>(g) * k_ + u]);
            out.push_back(rs_.compose_up(group, big_ops));
        }
        return out;
    }

    void HEBaeLowRingPcmm::project(std::vector<Ciphertext<Scheme::CKKS>>& out,
                                   std::vector<Ciphertext<Scheme::CKKS>>& in,
                                   const std::vector<double>& weight,
                                   int in_channels, int out_channels,
                                   HEArithmeticOperator<Scheme::CKKS>& small_ops)
    {
        if (static_cast<int>(in.size()) != in_channels)
            throw std::invalid_argument(
                "the low-ring activation is one ciphertext per row, so it must "
                "carry in_channels of them");

        std::vector<Ciphertext<Scheme::CKKS>*> ptrs;
        ptrs.reserve(in.size());
        for (auto& ct : in)
            ptrs.push_back(&ct);

        // k == 1 here, so this is ModDecomp-free and ModPack-free: the whole
        // product is out_i = sum_j U[i][j] * ct_j on the small ring's limbs.
        product_->project(out, ptrs, weight, in_channels, out_channels,
                          small_ops);
    }

    HEBaeLowRingPcmm::Cost HEBaeLowRingPcmm::big_ring_cost(int d1, int d2,
                                                           int tokens,
                                                           int n_big, int cols)
    {
        if (d1 < 1 || d2 < 1 || tokens < 1 || cols < 1 || n_big < 1)
            throw std::invalid_argument("cost model needs positive shapes");
        if (n_big % cols != 0)
            throw std::invalid_argument("cols must divide the ring degree");
        if (tokens > cols)
            throw std::invalid_argument(
                "the token count cannot exceed the encoded column count");

        const long long units = static_cast<long long>(d1) * d2;

        Cost c;
        // The a-part is the ciphertexts' a-vectors and an a-vector is as long
        // as the secret. ModDecomp splits it into k components of cols each
        // and keeps the rank, so the total width is the ring degree either
        // way, and this line is the whole reason the low ring exists.
        c.a_macs = units * n_big;
        c.b_macs = units * cols;
        // ModPack: k key switches per output ciphertext, d1/k output
        // ciphertexts.
        c.key_switches = d1;
        c.dimension = n_big;
        c.per_token = static_cast<double>(c.macs()) /
                      static_cast<double>(units * tokens);
        return c;
    }

    HEBaeLowRingPcmm::Cost HEBaeLowRingPcmm::low_ring_cost(int d1, int d2,
                                                           int tokens,
                                                           int n_big,
                                                           int n_small)
    {
        if (d1 < 1 || d2 < 1 || tokens < 1 || n_small < 1 || n_big < 1)
            throw std::invalid_argument("cost model needs positive shapes");
        if (n_big % n_small != 0)
            throw std::invalid_argument(
                "the small ring must divide the big one");
        if (tokens > n_small)
            throw std::invalid_argument(
                "the token count cannot exceed the small ring degree");

        const int k = n_big / n_small;
        const long long units = static_cast<long long>(d1) * d2;

        Cost c;
        // Rank 1 at degree n_small: the a-vector is n_small long because the
        // SECRET is, which is the same sentence as "the dimension dropped".
        c.a_macs = units * n_small;
        c.b_macs = units * n_small;
        // One per input ciphertext down, one per output ciphertext up, and
        // both counts are already divided by k because a big-ring ciphertext
        // carries k rows.
        c.key_switches = (static_cast<long long>(d2) + d1) / k;
        c.dimension = n_small;
        c.per_token = static_cast<double>(c.macs()) /
                      static_cast<double>(units * tokens);
        return c;
    }

} // namespace heongpu
