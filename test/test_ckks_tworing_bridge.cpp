// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// THE BRIDGE RETARGET IS A LAYOUT CHOICE, NOT A NEW TRANSFORM
// ===========================================================
//
// The two-ring flow needs the big ring to hold matrix data in the descent
// contract: big coefficient i*k + j' = coefficient i of island SinC column
// (g*k + j'). This file proves the algebraic identity that makes that layout
// reachable with EXISTING code:
//
//     compose_up of k island SinC columns
//         = sum_j' X^j' * col_{gk+j'}(X^k)
//         = sum_m X^m E'_m(X^{d*k}),   m = j' + k*u,
//
// i.e. the interleaved big ciphertext IS the SinC coefficient layout of the
// big ring at block size d_H = d*k over the SAME subring R_{k_s}, because
// k_s = N_s/d = N_H/(d*k). So the ROW BRIDGE at BatchMatrixLayout(N_H, d*k)
// converts it to and from slot form -- the retargeted SlotToSinC of
// LLAMA3_8B_LAYER_FLOW.md §6 is Llama3BatchOperator::to_slots/from_slots at
// a wider layout, with slot law
//
//     slot b + (k_H/2) * (j' + k*u)  of big ct g  =  M_b[u][g*k + j'].
//
// The tests pin, at a fast shape (13<->12, k=2, d_H=256) and at the real
// Llama shape (16<->13, k=8, d_H=1024):
//   1. to_slots at the wide layout reads the interleaved composition by
//      exactly that slot law;
//   2. from_slots then switch_down hands back the original island columns;
//   3. only the BSGS SUBSET of bridge keys is generated -- n1+n2-2 shifts,
//      not d_H-1 -- which is what makes the wide layout's key set affordable
//      (62 keys instead of 1023 at the Llama shape).
//
// The wide constructor inverts step matrices of d_H x d_H on the host, so
// the Llama-shape test spends minutes in setup; run the binary directly
// rather than under ctest's 30 s timeout.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using cd = std::complex<double>;

    void run_case(int logn_big, int logn_small, int d)
    {
        const int n_big = 1 << logn_big;
        const int n_small = 1 << logn_small;
        const int k = n_big / n_small;
        const int d_wide = d * k;

        // Contexts on the ring-switch shared-prefix contract.
        heongpu::HEContext<S> big =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        big->set_poly_modulus_degree(static_cast<size_t>(n_big));
        big->set_coeff_modulus_bit_sizes({60, 50, 50}, {60, 60});
        big->generate();

        heongpu::HEContext<S> small =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        {
            const auto primes = big->get_key_modulus();
            std::vector<Data64> q_vals, p_vals;
            for (int i = 0; i < 3; i++)
                q_vals.push_back(primes[i].value);
            p_vals.push_back(primes[3].value);
            small->set_poly_modulus_degree(static_cast<size_t>(n_small));
            small->set_coeff_modulus_values(q_vals, p_vals);
            small->generate();
        }

        heongpu::HEKeyGenerator<S> keygen(big);
        heongpu::Secretkey<S> secret(big, 16);
        keygen.generate_secret_key(secret);
        heongpu::Publickey<S> pub(big);
        keygen.generate_public_key(pub, secret);
        heongpu::HEEncryptor<S> encryptor(big, pub);
        heongpu::HEDecryptor<S> decryptor(big, secret);
        heongpu::HEEncoder<S> encoder(big);
        heongpu::HEArithmeticOperator<S> ops(big, encoder);

        heongpu::HERingSwitchOperator<S> rs(big, small);
        auto pair =
            std::make_unique<heongpu::HERingSwitchOperator<S>::SecretPair>(
                rs.make_secret_pair(64, 0xC0FFEEULL));
        rs.generate_keys(keygen, secret, pair->embedded);

        heongpu::HEKeyGenerator<S> small_keygen(small);
        heongpu::Publickey<S> small_pub(small);
        small_keygen.generate_public_key(small_pub, pair->small);
        heongpu::HEEncryptor<S> small_encryptor(small, small_pub);
        heongpu::HEDecryptor<S> small_decryptor(small, pair->small);
        heongpu::HEEncoder<S> small_encoder(small);

        // Island layout and staging operator (load/extract only).
        heongpu::BatchMatrixLayout layout_is(n_small, d);
        const int k_s = layout_is.k;
        const int blocks = layout_is.batch;
        heongpu::HEBatchMatrixOperator<S> small_bm(small, layout_is);

        // The wide operator at the big ring: same subring by construction.
        heongpu::BatchMatrixLayout layout_wide(n_big, d_wide);
        ASSERT_EQ(layout_wide.k, k_s)
            << "d_H = d*k must reproduce the island subring";
        heongpu::llama::Llama3BatchOperator wide(big, encoder, layout_wide,
                                                 std::pow(2.0, 40));

        // BSGS subset key: n1 + n2 - 2 shifts, NOT the d_H - 1 the full
        // bridge set would want. This is the affordability claim under test.
        int n1 = 1;
        while (n1 * n1 * 2 <= d_wide)
            n1 <<= 1;
        wide.set_bridge_baby_steps(n1);
        const int n2 = d_wide / n1;
        const int step = layout_wide.k / 2;
        std::vector<int> shifts;
        for (int j = 1; j < n1; ++j)
            shifts.push_back(j * step);
        for (int i = 1; i < n2; ++i)
            shifts.push_back(i * n1 * step);
        heongpu::Galoiskey<S> bridge_galois(big, shifts);
        keygen.generate_galois_key(bridge_galois, secret);

        // Data: one island matrix encryption of `blocks` d x d matrices.
        const double scale = std::pow(2.0, 40);
        std::mt19937_64 rng(20260810u + static_cast<unsigned>(logn_big));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::vector<cd>> M(
            blocks, std::vector<cd>(static_cast<size_t>(d) * d));
        for (int b = 0; b < blocks; ++b)
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    M[b][static_cast<size_t>(i) * d + j] =
                        cd(dist(rng), 0.0);

        heongpu::BatchMatrixEncoder bm(k_s);
        std::vector<int64_t> coeffs;
        bm.encode(M, d, d, scale, coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout_is, d, d,
                                                      columns);
        ASSERT_EQ(static_cast<int>(columns.size()), d);

        // Encrypt the island columns and compose k-groups up: the descent
        // contract's big-ring layout, produced by the crossing itself.
        std::vector<heongpu::Ciphertext<S>> island_cols;
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> p(small);
            small_bm.load_coefficients(p, columns[j], scale);
            heongpu::Ciphertext<S> c(small);
            small_encryptor.encrypt(c, p);
            island_cols.push_back(std::move(c));
        }

        std::vector<heongpu::Ciphertext<S>> big_cts;
        for (int g = 0; g < d / k; ++g)
        {
            std::vector<heongpu::Ciphertext<S>> group(
                island_cols.begin() + static_cast<size_t>(g) * k,
                island_cols.begin() + static_cast<size_t>(g + 1) * k);
            big_cts.push_back(rs.compose_up(group, ops));
        }

        // ---- Claim 1: to_slots at the wide layout reads the interleave. --
        heongpu::llama::BatchActivation act;
        act.rows = d_wide;
        act.column = std::move(big_cts);
        std::vector<heongpu::Ciphertext<S>> slot_cts =
            wide.to_slots(act, bridge_galois);
        ASSERT_EQ(static_cast<int>(slot_cts.size()), d / k);

        double worst_slot = 0.0;
        for (int g = 0; g < d / k; ++g)
        {
            heongpu::Plaintext<S> p(big);
            decryptor.decrypt(p, slot_cts[g]);
            std::vector<Complex64> got;
            encoder.decode(got, p);
            for (int b = 0; b < blocks; ++b)
                for (int u = 0; u < d; ++u)
                    for (int jp = 0; jp < k; ++jp)
                    {
                        const int m = jp + k * u;
                        const size_t slot =
                            static_cast<size_t>(b) +
                            static_cast<size_t>(step) * m;
                        const double want =
                            M[b][static_cast<size_t>(u) * d + (g * k + jp)]
                                .real();
                        const double err =
                            std::abs(got[slot].real() - want);
                        worst_slot = std::max(worst_slot, err);
                        ASSERT_NEAR(got[slot].real(), want, 1e-2)
                            << "slot law at (g,b,u,j') = (" << g << "," << b
                            << "," << u << "," << jp << ")";
                    }
        }
        std::cout << "wide to_slots slot-law worst error: " << worst_slot
                  << std::endl;

        // ---- Claim 2: from_slots then descend returns the columns. -------
        heongpu::llama::BatchActivation back =
            wide.from_slots(slot_cts, d_wide, bridge_galois);

        double worst_coeff = 0.0;
        for (int g = 0; g < d / k; ++g)
        {
            std::vector<heongpu::Ciphertext<S>> parts =
                rs.switch_down(back.column[g], ops);
            ASSERT_EQ(static_cast<int>(parts.size()), k);
            for (int jp = 0; jp < k; ++jp)
            {
                heongpu::Plaintext<S> p(small);
                small_decryptor.decrypt(p, parts[jp]);
                std::vector<int64_t> got;
                small_bm.extract_coefficients(got, p);
                const double got_scale = parts[jp].scale();
                const std::vector<int64_t>& want = columns[g * k + jp];
                for (int i = 0; i < n_small; ++i)
                {
                    const double v =
                        static_cast<double>(got[i]) / got_scale;
                    const double w =
                        static_cast<double>(want[i]) / scale;
                    const double err = std::abs(v - w);
                    worst_coeff = std::max(worst_coeff, err);
                    ASSERT_NEAR(v, w, 1e-2 * std::max(1.0, std::abs(w)))
                        << "column " << g * k + jp << " coeff " << i;
                }
            }
        }
        std::cout << "descended-column worst error: " << worst_coeff
                  << std::endl;
    }
} // namespace

// Fast shape: island logN 12 (d = 128, R_32) inside big logN 13, k = 2,
// wide layout (8192, 256). Constructor inverts 16 matrices of 256 x 256.
TEST(CkksTworingBridge, InterleaveIsWideSinC_Small)
{
    run_case(13, 12, 128);
}

// The Llama shape: island logN 13 (d = 128, R_64) inside big logN 16, k = 8,
// wide layout (65536, 1024). Constructor inverts 32 matrices of 1024 x 1024
// on the host -- minutes of setup, which is a one-time cost per process and
// a known follow-up (the Vandermonde has an analytic inverse).
TEST(CkksTworingBridge, InterleaveIsWideSinC_Llama)
{
    run_case(16, 13, 128);
}
