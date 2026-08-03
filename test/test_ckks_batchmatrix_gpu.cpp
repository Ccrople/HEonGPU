// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU-side validation of the batch matrix primitives.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using cd = std::complex<double>;

    /// Reference complex matrix product, row-major.
    std::vector<cd> matmul(const std::vector<cd>& a, const std::vector<cd>& b,
                           int n, int m, int p)
    {
        std::vector<cd> c(static_cast<size_t>(n) * p, cd(0.0, 0.0));
        for (int i = 0; i < n; ++i)
            for (int t = 0; t < m; ++t)
            {
                const cd av = a[static_cast<size_t>(i) * m + t];
                for (int j = 0; j < p; ++j)
                    c[static_cast<size_t>(i) * p + j] +=
                        av * b[static_cast<size_t>(t) * p + j];
            }
        return c;
    }

    struct Fixture
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        int n = 0;
        double scale = 0.0;

        explicit Fixture(size_t degree)
            : context(
                  heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes({50, 40, 40}, {50});
            context->generate();
            n = static_cast<int>(degree);
            scale = std::pow(2.0, 40);

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);

            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
        }

        heongpu::Ciphertext<S> encrypt(double v)
        {
            std::vector<double> msg(static_cast<size_t>(n) / 2, v);
            heongpu::Plaintext<S> p(context);
            encoder->encode(p, msg, scale);
            heongpu::Ciphertext<S> c(context);
            encryptor->encrypt(c, p);
            return c;
        }
    };
} // namespace

// Isolates the raw coefficient path (load_coefficients -> encrypt -> decrypt ->
// extract_coefficients) from the subring transform. If this fails, the bug is
// in the plaintext staging rather than in any batch matrix kernel.
TEST(HEonGPU, CKKS_BatchMatrix_RawCoefficientRoundTrip)
{
    const size_t degree = 4096;
    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), 8);
    heongpu::HEBatchMatrixOperator<S> op(context, layout);

    std::vector<int64_t> coeffs(degree);
    std::mt19937_64 rng(31337u);
    std::uniform_int_distribution<int64_t> dist(-100000, 100000);
    for (auto& c : coeffs)
        c = dist(rng);

    heongpu::Plaintext<S> p(context);
    op.load_coefficients(p, coeffs, std::pow(2.0, 30));

    // (a0) transform pair alone, no Plaintext involved at all.
    {
        const std::vector<int64_t> probe = op.ntt_intt_probe(coeffs);
        ASSERT_EQ(probe.size(), coeffs.size());
        int64_t w = 0;
        size_t first_bad = probe.size();
        for (size_t i = 0; i < coeffs.size(); ++i)
        {
            const int64_t diff = std::llabs(probe[i] - coeffs[i]);
            if (diff != 0 && first_bad == probe.size())
                first_bad = i;
            w = std::max<int64_t>(w, diff);
        }
        std::cout << "NTT/INTT probe worst error: " << w << std::endl;
        if (w != 0)
        {
            std::cout << "  first mismatch at index " << first_bad << ": got "
                      << probe[first_bad] << " expected " << coeffs[first_bad]
                      << std::endl;
            std::cout << "  in[0..3]  = " << coeffs[0] << " " << coeffs[1]
                      << " " << coeffs[2] << " " << coeffs[3] << std::endl;
            std::cout << "  out[0..3] = " << probe[0] << " " << probe[1] << " "
                      << probe[2] << " " << probe[3] << std::endl;
        }
        EXPECT_EQ(w, 0) << "the length-N NTT/INTT pair is not the identity";
    }

    // (a) staging alone: NTT then INTT, no encryption in the loop.
    {
        heongpu::Plaintext<S> probe(context);
        op.load_coefficients(probe, coeffs, std::pow(2.0, 30));
        std::vector<int64_t> staged;
        op.extract_coefficients(staged, probe);
        ASSERT_EQ(staged.size(), coeffs.size());
        int64_t w = 0;
        for (size_t i = 0; i < coeffs.size(); ++i)
            w = std::max<int64_t>(w, std::llabs(staged[i] - coeffs[i]));
        std::cout << "staging-only worst error: " << w << std::endl;
        EXPECT_EQ(w, 0) << "load_coefficients/extract_coefficients are not "
                           "inverse; the bug is in plaintext staging";
    }

    heongpu::Ciphertext<S> c(context);
    encryptor.encrypt(c, p);

    heongpu::Plaintext<S> back(context);
    decryptor.decrypt(back, c);
    std::vector<int64_t> got;
    op.extract_coefficients(got, back);

    ASSERT_EQ(got.size(), coeffs.size());
    int64_t worst = 0;
    for (size_t i = 0; i < coeffs.size(); ++i)
        worst = std::max<int64_t>(worst, std::llabs(got[i] - coeffs[i]));
    std::cout << "raw coefficient worst error: " << worst << std::endl;
    // Encryption noise is tiny compared to these magnitudes.
    EXPECT_LT(worst, 1000);
}

// Multiplying by X^power must permute coefficients negacyclically:
// (X^power * m)[i+power] = m[i], with a sign flip on wraparound. This is the
// exactness the TWEAK recursion depends on, so it is checked coefficient-wise
// rather than through decoded values.
TEST(HEonGPU, CKKS_BatchMatrix_MonomialMultiplyShiftsCoefficients)
{
    const size_t degree = 4096;
    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), 8);
    heongpu::HEBatchMatrixOperator<S> op(context, layout);

    const int n = static_cast<int>(degree);
    std::vector<int64_t> coeffs(n, 0);
    std::mt19937_64 rng(90210u);
    std::uniform_int_distribution<int64_t> dist(-50000, 50000);
    for (auto& c : coeffs)
        c = dist(rng);

    for (int power : {1, 7, 64, n - 1, n, n + 5})
    {
        heongpu::Plaintext<S> p(context);
        op.load_coefficients(p, coeffs, std::pow(2.0, 30));
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);

        op.mult_monomial(c, power);

        heongpu::Plaintext<S> back(context);
        decryptor.decrypt(back, c);
        std::vector<int64_t> got;
        op.extract_coefficients(got, back);

        // Reference negacyclic shift.
        std::vector<int64_t> ref(n, 0);
        for (int i = 0; i < n; ++i)
        {
            int dst = i + power;
            int64_t v = coeffs[i];
            while (dst >= n)
            {
                dst -= n;
                v = -v;
            }
            ref[dst] = v;
        }

        int64_t worst = 0;
        for (int i = 0; i < n; ++i)
            worst = std::max<int64_t>(worst, std::llabs(got[i] - ref[i]));
        std::cout << "monomial X^" << power << " worst error: " << worst
                  << std::endl;
        EXPECT_LT(worst, 1000) << "power=" << power;
    }
}

// TWEAK (Algorithm 2) is a size-d transform built from monomial multiplies and
// additions, so running the forward pass and then the inverse pass must return
// exactly d times the input. No key switching is involved, so the only error
// admitted is the original encryption noise scaled by d.
TEST(HEonGPU, CKKS_BatchMatrix_TweakForwardInverseScalesByD)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    heongpu::HEBatchMatrixOperator<S> op(context, layout);

    const int n = static_cast<int>(degree);
    std::mt19937_64 rng(5150u);
    std::uniform_int_distribution<int64_t> dist(-20000, 20000);

    std::vector<std::vector<int64_t>> original(d,
                                               std::vector<int64_t>(n, 0));
    std::vector<heongpu::Ciphertext<S>> cts;
    cts.reserve(d);
    for (int i = 0; i < d; ++i)
    {
        for (auto& c : original[i])
            c = dist(rng);
        heongpu::Plaintext<S> p(context);
        op.load_coefficients(p, original[i], std::pow(2.0, 30));
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);
        cts.push_back(std::move(c));
    }

    op.tweak(cts, +1);
    op.tweak(cts, -1);
    ASSERT_EQ(cts.size(), static_cast<size_t>(d));

    int64_t worst = 0;
    for (int i = 0; i < d; ++i)
    {
        heongpu::Plaintext<S> back(context);
        decryptor.decrypt(back, cts[i]);
        std::vector<int64_t> got;
        op.extract_coefficients(got, back);
        ASSERT_EQ(got.size(), static_cast<size_t>(n));
        for (int j = 0; j < n; ++j)
            worst = std::max<int64_t>(
                worst, std::llabs(got[j] - static_cast<int64_t>(d) *
                                                original[i][j]));
    }
    std::cout << "TWEAK round-trip worst error: " << worst << std::endl;
    EXPECT_LT(worst, 100000) << "forward+inverse TWEAK did not scale by d";
}

// CMT (Algorithm 3) must turn a matrix encryption of {M_l} into a matrix
// encryption of {M_l^T}. Unlike TWEAK this key-switches, so it admits real
// noise and is compared on decoded values rather than coefficients.
TEST(HEonGPU, CKKS_BatchMatrix_CMTTransposesBatch)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    // A single active limb keeps extract_coefficients' centered lift exact.
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder_slots(context);
    heongpu::HEArithmeticOperator<S> ops(context, encoder_slots);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
    heongpu::Galoiskey<S> galois_key(context, rot);
    keygen.generate_galois_key(galois_key, secret);

    heongpu::HEBatchMatrixOperator<S> op(context, layout);
    heongpu::BatchMatrixEncoder bm(layout.k);

    const int nslots = bm.slots();
    // CMT key-switches, so it admits real noise; this scale keeps that noise
    // small relative to the values while staying inside the 50-bit prime.
    const double scale = std::pow(2.0, 30);
    std::vector<std::vector<cd>> M(
        nslots, std::vector<cd>(static_cast<size_t>(d) * d));
    std::mt19937_64 rng(24680u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& m : M)
        for (auto& z : m)
            z = cd(dist(rng), dist(rng));

    std::vector<int64_t> coeffs;
    bm.encode(M, d, d, scale, coeffs);
    std::vector<std::vector<int64_t>> columns;
    heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, d,
                                                  columns);

    std::vector<heongpu::Ciphertext<S>> cts;
    cts.reserve(d);
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> p(context);
        op.load_coefficients(p, columns[j], scale);
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);
        cts.push_back(std::move(c));
    }

    op.cmt(cts, galois_key, ops);
    ASSERT_EQ(cts.size(), static_cast<size_t>(d));

    std::vector<std::vector<int64_t>> out_columns(d);
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> p(context);
        decryptor.decrypt(p, cts[j]);
        op.extract_coefficients(out_columns[j], p);
    }

    std::vector<int64_t> got_coeffs;
    heongpu::split_matrix_encryption_coefficients(out_columns, layout, d, d,
                                                  got_coeffs);
    std::vector<std::vector<cd>> got;
    bm.decode(got_coeffs, d, d, scale, got);

    double worst = 0.0;
    for (int s = 0; s < nslots; ++s)
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
            {
                const cd expected = M[s][static_cast<size_t>(j) * d + i];
                const cd actual = got[s][static_cast<size_t>(i) * d + j];
                worst = std::max(worst, std::abs(actual - expected));
            }
    std::cout << "CMT transpose worst error: " << worst << std::endl;
    EXPECT_LT(worst, 0.05);
}

// Everything above runs at a single active limb so the centered lift is a
// plain subtraction. This repeats PCMM and CMT at two limbs, where the
// coefficients must be rebuilt by CRT, to confirm the kernels are limb-count
// agnostic rather than accidentally correct at numLimbs == 1.
TEST(HEonGPU, CKKS_BatchMatrix_MultiLimbPCMMAndCMT)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({50, 40}, {50});
    context->generate();
    ASSERT_EQ(context->get_ciphertext_modulus_count(), 2);

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder_slots(context);
    heongpu::HEArithmeticOperator<S> ops(context, encoder_slots);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    heongpu::HEBatchMatrixOperator<S> op(context, layout);
    heongpu::BatchMatrixEncoder bm(layout.k);
    const int nslots = bm.slots();

    // (1) raw coefficient round trip across two limbs.
    {
        std::vector<int64_t> coeffs(degree);
        std::mt19937_64 rng(1234u);
        std::uniform_int_distribution<int64_t> dist(-100000, 100000);
        for (auto& c : coeffs)
            c = dist(rng);

        heongpu::Plaintext<S> p(context);
        op.load_coefficients(p, coeffs, std::pow(2.0, 30));
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);
        heongpu::Plaintext<S> back(context);
        decryptor.decrypt(back, c);
        std::vector<int64_t> got;
        op.extract_coefficients(got, back);

        int64_t worst = 0;
        for (size_t i = 0; i < coeffs.size(); ++i)
            worst = std::max<int64_t>(worst, std::llabs(got[i] - coeffs[i]));
        std::cout << "multi-limb raw round trip worst error: " << worst
                  << std::endl;
        EXPECT_LT(worst, 1000);
    }

    // (2) PCMM at two limbs.
    {
        const int cols_out = 3;
        const double sm = std::pow(2.0, 20), su = std::pow(2.0, 20);
        std::vector<std::vector<cd>> M(
            nslots, std::vector<cd>(static_cast<size_t>(d) * d));
        std::vector<std::vector<cd>> U(
            nslots, std::vector<cd>(static_cast<size_t>(d) * cols_out));
        std::mt19937_64 rng(777u);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& m : M)
            for (auto& z : m)
                z = cd(dist(rng), dist(rng));
        for (auto& u : U)
            for (auto& z : u)
                z = cd(dist(rng), dist(rng));

        std::vector<int64_t> mc, uc;
        bm.encode(M, d, d, sm, mc);
        bm.encode(U, d, cols_out, su, uc);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(mc, layout, d, d,
                                                      columns);

        std::vector<heongpu::Ciphertext<S>> cts;
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> p(context);
            op.load_coefficients(p, columns[j], sm);
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, p);
            cts.push_back(std::move(c));
        }
        op.encode_plaintext_matrix(uc, d, cols_out, 0, su);

        std::vector<heongpu::Ciphertext<S>*> in;
        for (auto& c : cts)
            in.push_back(&c);
        std::vector<heongpu::Ciphertext<S>> out;
        op.pcmm(out, in, /*rescale=*/false);

        std::vector<std::vector<int64_t>> oc(cols_out);
        for (int j = 0; j < cols_out; ++j)
        {
            heongpu::Plaintext<S> p(context);
            decryptor.decrypt(p, out[j]);
            op.extract_coefficients(oc[j], p);
        }
        std::vector<int64_t> gc;
        heongpu::split_matrix_encryption_coefficients(oc, layout, d, cols_out,
                                                      gc);
        std::vector<std::vector<cd>> got;
        bm.decode(gc, d, cols_out, sm * su, got);

        double worst = 0.0;
        for (int s = 0; s < nslots; ++s)
        {
            const std::vector<cd> ref = matmul(M[s], U[s], d, d, cols_out);
            for (size_t e = 0; e < ref.size(); ++e)
                worst = std::max(worst, std::abs(got[s][e] - ref[e]));
        }
        std::cout << "multi-limb PCMM worst error: " << worst << std::endl;
        EXPECT_LT(worst, 0.05);
    }

    // (3) CMT at two limbs.
    {
        std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
        heongpu::Galoiskey<S> gk(context, rot);
        keygen.generate_galois_key(gk, secret);

        const double scale = std::pow(2.0, 30);
        std::vector<std::vector<cd>> M(
            nslots, std::vector<cd>(static_cast<size_t>(d) * d));
        std::mt19937_64 rng(31415u);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& m : M)
            for (auto& z : m)
                z = cd(dist(rng), dist(rng));

        std::vector<int64_t> coeffs;
        bm.encode(M, d, d, scale, coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, d,
                                                      columns);
        std::vector<heongpu::Ciphertext<S>> cts;
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> p(context);
            op.load_coefficients(p, columns[j], scale);
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, p);
            cts.push_back(std::move(c));
        }

        op.cmt(cts, gk, ops);

        std::vector<std::vector<int64_t>> oc(d);
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> p(context);
            decryptor.decrypt(p, cts[j]);
            op.extract_coefficients(oc[j], p);
        }
        std::vector<int64_t> gc;
        heongpu::split_matrix_encryption_coefficients(oc, layout, d, d, gc);
        std::vector<std::vector<cd>> got;
        bm.decode(gc, d, d, scale, got);

        double worst = 0.0;
        for (int s = 0; s < nslots; ++s)
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    worst = std::max(
                        worst, std::abs(got[s][static_cast<size_t>(i) * d + j] -
                                        M[s][static_cast<size_t>(j) * d + i]));
        std::cout << "multi-limb CMT worst error: " << worst << std::endl;
        EXPECT_LT(worst, 0.05);
    }
}

// End-to-end batch PCMM (Algorithm 1): encrypt a matrix encryption of a batch
// of complex matrices, multiply by a plaintext matrix, decrypt and compare
// against the reference complex products. A single-prime modulus keeps the
// centered lift exact, and rescale is skipped so the product stays at the
// combined scale.
TEST(HEonGPU, CKKS_BatchMatrix_PCMMMatchesReference)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    heongpu::HEBatchMatrixOperator<S> op(context, layout);
    heongpu::BatchMatrixEncoder encoder(layout.k);

    const int inner = d;      // M is d x d
    const int cols_out = 3;   // U is d x 3
    // The product lands at scale_m * scale_u and accumulates k * inner terms,
    // so these are chosen to keep the result well inside the 50-bit prime
    // while leaving enough precision for a tight comparison.
    const double scale_m = std::pow(2.0, 20);
    const double scale_u = std::pow(2.0, 20);

    // Small magnitudes: the product accumulates d terms at scale_m * scale_u
    // and must stay well inside the 50-bit prime.
    const int nslots = encoder.slots();
    std::vector<std::vector<cd>> M(
        nslots, std::vector<cd>(static_cast<size_t>(d) * inner));
    std::vector<std::vector<cd>> U(
        nslots, std::vector<cd>(static_cast<size_t>(inner) * cols_out));
    std::mt19937_64 rng(20260803u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& m : M)
        for (auto& z : m)
            z = cd(dist(rng), dist(rng));
    for (auto& u : U)
        for (auto& z : u)
            z = cd(dist(rng), dist(rng));

    // Encrypt the matrix encryption of M.
    std::vector<int64_t> m_coeffs;
    encoder.encode(M, d, inner, scale_m, m_coeffs);
    std::vector<std::vector<int64_t>> columns;
    heongpu::build_matrix_encryption_coefficients(m_coeffs, layout, d, inner,
                                                  columns);

    std::vector<heongpu::Ciphertext<S>> cts;
    cts.reserve(inner);
    for (int j = 0; j < inner; ++j)
    {
        heongpu::Plaintext<S> p(context);
        op.load_coefficients(p, columns[j], scale_m);
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);
        cts.push_back(std::move(c));
    }

    // Upload the plaintext matrix U and run PCMM.
    std::vector<int64_t> u_coeffs;
    encoder.encode(U, inner, cols_out, scale_u, u_coeffs);
    op.encode_plaintext_matrix(u_coeffs, inner, cols_out, 0, scale_u);

    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : cts)
        in.push_back(&c);
    std::vector<heongpu::Ciphertext<S>> out;
    op.pcmm(out, in, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(cols_out));

    // Decrypt back into a matrix encryption and decode.
    std::vector<std::vector<int64_t>> out_columns(cols_out);
    for (int j = 0; j < cols_out; ++j)
    {
        heongpu::Plaintext<S> p(context);
        decryptor.decrypt(p, out[j]);
        op.extract_coefficients(out_columns[j], p);
    }

    std::vector<int64_t> got_coeffs;
    heongpu::split_matrix_encryption_coefficients(out_columns, layout, d,
                                                  cols_out, got_coeffs);
    std::vector<std::vector<cd>> got;
    encoder.decode(got_coeffs, d, cols_out, scale_m * scale_u, got);

    // Reference: the batch of complex matrix products.
    ASSERT_EQ(got.size(), static_cast<size_t>(nslots));
    double worst = 0.0;
    for (int s = 0; s < nslots; ++s)
    {
        const std::vector<cd> ref = matmul(M[s], U[s], d, inner, cols_out);
        for (size_t e = 0; e < ref.size(); ++e)
        {
            worst = std::max(worst, std::abs(got[s][e] - ref[s == 0 ? e : e]));
            ASSERT_NEAR(got[s][e].real(), ref[e].real(), 1e-2)
                << "slot " << s << " entry " << e;
            ASSERT_NEAR(got[s][e].imag(), ref[e].imag(), 1e-2)
                << "slot " << s << " entry " << e;
        }
    }
    std::cout << "PCMM worst absolute error: " << worst << std::endl;
}

// The subring transform is a change of basis, so transforming into the R_k NTT
// domain and back must be the identity, bit for bit. This isolates the two
// transform kernels, which carry all the twiddle-factor risk, from the GEMM.
TEST(HEonGPU, CKKS_BatchMatrix_SubringTransformIsExactRoundTrip)
{
    const size_t degree = 4096;
    Fixture fx(degree);

    for (int d : {2, 4, 8, 16})
    {
        heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
        heongpu::HEBatchMatrixOperator<S> op(fx.context, layout);

        std::vector<heongpu::Ciphertext<S>> cts;
        cts.reserve(d);
        for (int i = 0; i < d; ++i)
            cts.push_back(fx.encrypt(1.0 + 0.25 * i));

        std::vector<std::vector<Data64>> before(d);
        for (int i = 0; i < d; ++i)
            cts[i].get_data(before[i]);

        std::vector<heongpu::Ciphertext<S>*> ptrs;
        for (auto& c : cts)
            ptrs.push_back(&c);

        op.subring_round_trip(ptrs);

        for (int i = 0; i < d; ++i)
        {
            std::vector<Data64> after;
            cts[i].get_data(after);
            ASSERT_EQ(after.size(), before[i].size()) << "d=" << d;

            size_t mismatches = 0;
            for (size_t j = 0; j < after.size(); ++j)
                mismatches += (after[j] != before[i][j]);
            EXPECT_EQ(mismatches, 0u)
                << "d=" << d << " ciphertext " << i << ": "
                << mismatches << " of " << after.size()
                << " coefficients changed across the subring round trip";
        }
    }
}

namespace
{
    /// acc += lhs * rhs in Z_p[Y]/(Y^k + 1), the ring one matrix entry lives in.
    void negacyclic_mul_acc(std::vector<uint64_t>& acc,
                            const std::vector<uint64_t>& lhs,
                            const std::vector<uint64_t>& rhs, uint64_t p)
    {
        const int k = static_cast<int>(acc.size());
        for (int i = 0; i < k; ++i)
        {
            if (lhs[i] == 0)
                continue;
            for (int j = 0; j < k; ++j)
            {
                const uint64_t prod = static_cast<uint64_t>(
                    (static_cast<__uint128_t>(lhs[i]) * rhs[j]) % p);
                const int t = i + j;
                if (t < k)
                    acc[t] = (acc[t] + prod) % p;
                else
                    acc[t - k] = (acc[t - k] + p - prod) % p;
            }
        }
    }

    /// Centered coefficients back to residues mod p.
    std::vector<uint64_t> to_residues(const std::vector<int64_t>& v, uint64_t p)
    {
        std::vector<uint64_t> r(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            r[i] = (v[i] < 0)
                       ? static_cast<uint64_t>(v[i] + static_cast<int64_t>(p))
                       : static_cast<uint64_t>(v[i]);
        return r;
    }
} // namespace

// Batch CCMM (Algorithm 4) checked at the ring level.
//
// Both operands are given a zero c1 component. That makes every key switch in
// the pipeline exact -- key switching a zero polynomial yields exactly zero, as
// does relinearising (0, 0) -- so the three CMTs, the TWEAKs, the automorphisms,
// the four GEMMs, the transposes and the recombination can be checked for exact
// equality rather than against a noise tolerance. With c1 = 0 the matrix the
// ciphertexts carry is simply the c0 matrix, so the expected result is the plain
// product of the two c0 matrices over R_{q,k}.
TEST(HEonGPU, CKKS_BatchMatrix_CCMMIsExactWithZeroC1)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    // A single active limb keeps extract_coefficients' centered lift exact.
    context->set_coeff_modulus_bit_sizes({50}, {50});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder_slots(context);
    heongpu::HEArithmeticOperator<S> ops(context, encoder_slots);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    const int k = layout.k;

    // CMT needs the automorphisms X -> X^(2kt+1); CCMM additionally needs the
    // relinearisation key.
    std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
    heongpu::Galoiskey<S> galois_key(context, rot);
    keygen.generate_galois_key(galois_key, secret);
    heongpu::Relinkey<S> relin_key(context);
    keygen.generate_relin_key(relin_key, secret);

    heongpu::HEBatchMatrixOperator<S> op(context, layout);

    const uint64_t p = context->get_key_modulus()[0].value;
    // A component spans every active limb, not just the first; hardcoding one
    // limb here silently truncates the trivial encryptions built below.
    const size_t comp =
        static_cast<size_t>(context->get_ciphertext_modulus_count()) * degree;

    std::mt19937_64 rng(90210u);
    std::uniform_int_distribution<int64_t> dist(-1000, 1000);

    // Builds one operand and reads back its c0 residues. The snapshot has to
    // happen after c1 is zeroed: encryption puts pk0 * u + m + e into c0, so it
    // is only once c1 is cleared that decryption returns c0 itself.
    auto make_operand = [&](std::vector<heongpu::Ciphertext<S>>& cts,
                            std::vector<std::vector<uint64_t>>& c0)
    {
        cts.clear();
        c0.assign(d, std::vector<uint64_t>());
        for (int j = 0; j < d; ++j)
        {
            std::vector<int64_t> coeffs(degree);
            for (auto& v : coeffs)
                v = dist(rng);

            heongpu::Plaintext<S> pt(context);
            op.load_coefficients(pt, coeffs, std::pow(2.0, 30));
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, pt);
            c.store_in_device();
            ASSERT_EQ(cudaMemset(c.data() + comp, 0, comp * sizeof(Data64)),
                      cudaSuccess);
            cts.push_back(std::move(c));

            heongpu::Plaintext<S> back(context);
            decryptor.decrypt(back, cts[j]);
            std::vector<int64_t> got;
            op.extract_coefficients(got, back);
            c0[j] = to_residues(got, p);
        }
    };

    std::vector<heongpu::Ciphertext<S>> ca, cb;
    std::vector<std::vector<uint64_t>> a0, b0;
    make_operand(ca, a0);
    make_operand(cb, b0);

    std::vector<heongpu::Ciphertext<S>*> pa, pb;
    for (auto& c : ca)
        pa.push_back(&c);
    for (auto& c : cb)
        pb.push_back(&c);

    std::vector<heongpu::Ciphertext<S>> out;
    op.ccmm(out, pa, pb, galois_key, relin_key, ops, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(d));

    // c1 must have stayed exactly zero. Checking it separately keeps a key
    // switch that is not exact on zero from being mistaken for a CCMM bug, and
    // it is what makes the decryption below equal to c0. The NTT of zero is
    // zero, so this reads the raw data with no transform.
    for (int j = 0; j < d; ++j)
    {
        std::vector<Data64> raw;
        out[j].get_data(raw);
        ASSERT_GE(raw.size(), 2 * comp);
        size_t nonzero = 0;
        for (size_t i = comp; i < 2 * comp; ++i)
            nonzero += (raw[i] != 0);
        ASSERT_EQ(nonzero, 0u) << "ciphertext " << j << ": c1 picked up "
                               << nonzero << " nonzero coefficients, so some "
                                             "key switch was not exact on zero";
    }

    std::vector<std::vector<uint64_t>> got(d);
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> back(context);
        decryptor.decrypt(back, out[j]);
        std::vector<int64_t> v;
        op.extract_coefficients(v, back);
        got[j] = to_residues(v, p);
    }

    // The naive reference costs d * k^2 per entry, so check a spread of entries
    // rather than all d^2 of them.
    const std::vector<std::pair<int, int>> checks = {
        {0, 0}, {0, 1}, {1, 0}, {3, 5}, {4, 4}, {7, 7}, {7, 0}, {0, 7}};

    for (auto [i, j] : checks)
    {
        std::vector<uint64_t> acc(k, 0), lhs(k), rhs(k);
        for (int t = 0; t < d; ++t)
        {
            for (int s = 0; s < k; ++s)
            {
                lhs[s] = a0[t][i + static_cast<size_t>(d) * s];
                rhs[s] = b0[j][t + static_cast<size_t>(d) * s];
            }
            negacyclic_mul_acc(acc, lhs, rhs, p);
        }
        for (int s = 0; s < k; ++s)
            ASSERT_EQ(got[j][i + static_cast<size_t>(d) * s], acc[s])
                << "row " << i << " col " << j << " coefficient " << s;
    }
}

// End-to-end batch CCMM: encrypt matrix encryptions of two batches of complex
// matrices, multiply them, decrypt and compare against the reference complex
// products. Unlike the exact test above, both operands carry a real c1, so this
// is what exercises all four cross products and the relinearisation.
TEST(HEonGPU, CKKS_BatchMatrix_CCMMMatchesReference)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({60, 50}, {60});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder_slots(context);
    heongpu::HEArithmeticOperator<S> ops(context, encoder_slots);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
    heongpu::Galoiskey<S> galois_key(context, rot);
    keygen.generate_galois_key(galois_key, secret);
    heongpu::Relinkey<S> relin_key(context);
    keygen.generate_relin_key(relin_key, secret);

    heongpu::HEBatchMatrixOperator<S> op(context, layout);
    heongpu::BatchMatrixEncoder encoder(layout.k);

    // The product lands at scale_a * scale_b and accumulates k * d terms, so
    // these keep it well inside the 50-bit prime while leaving enough precision
    // for a tight comparison.
    // The two scales are deliberately asymmetric. Step 1 CMTs the right
    // operand, and that key switching noise ends up multiplied by the left
    // operand in the GEMM, so after decoding by scale_a * scale_b the surviving
    // error is e_cmt * sqrt(d*k) / scale_b -- it depends on scale_b alone.
    // scale_b therefore takes as many bits as the product can spare, and
    // scale_a only enough to quantise the left operand.
    const double scale_a = std::pow(2.0, 25);
    const double scale_b = std::pow(2.0, 35);

    const int nslots = encoder.slots();
    std::vector<std::vector<cd>> M(
        nslots, std::vector<cd>(static_cast<size_t>(d) * d));
    std::vector<std::vector<cd>> U(
        nslots, std::vector<cd>(static_cast<size_t>(d) * d));
    std::mt19937_64 rng(13579u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& m : M)
        for (auto& z : m)
            z = cd(dist(rng), dist(rng));
    for (auto& u : U)
        for (auto& z : u)
            z = cd(dist(rng), dist(rng));

    auto encrypt_matrix = [&](const std::vector<std::vector<cd>>& mat,
                              double scale,
                              std::vector<heongpu::Ciphertext<S>>& cts)
    {
        std::vector<int64_t> coeffs;
        encoder.encode(mat, d, d, scale, coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, d,
                                                      columns);
        cts.clear();
        cts.reserve(d);
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> pt(context);
            op.load_coefficients(pt, columns[j], scale);
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, pt);
            cts.push_back(std::move(c));
        }
    };

    std::vector<heongpu::Ciphertext<S>> ca, cb;
    encrypt_matrix(M, scale_a, ca);
    encrypt_matrix(U, scale_b, cb);

    std::vector<heongpu::Ciphertext<S>*> pa, pb;
    for (auto& c : ca)
        pa.push_back(&c);
    for (auto& c : cb)
        pb.push_back(&c);

    std::vector<heongpu::Ciphertext<S>> out;
    op.ccmm(out, pa, pb, galois_key, relin_key, ops, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(d));

    std::vector<std::vector<int64_t>> out_columns(d);
    for (int j = 0; j < d; ++j)
    {
        heongpu::Plaintext<S> pt(context);
        decryptor.decrypt(pt, out[j]);
        op.extract_coefficients(out_columns[j], pt);
    }

    std::vector<int64_t> got_coeffs;
    heongpu::split_matrix_encryption_coefficients(out_columns, layout, d, d,
                                                  got_coeffs);
    std::vector<std::vector<cd>> got;
    encoder.decode(got_coeffs, d, d, scale_a * scale_b, got);

    ASSERT_EQ(got.size(), static_cast<size_t>(nslots));
    double worst = 0.0;
    for (int s = 0; s < nslots; ++s)
    {
        const std::vector<cd> ref = matmul(M[s], U[s], d, d, d);
        for (size_t e = 0; e < ref.size(); ++e)
        {
            worst = std::max(worst, std::abs(got[s][e] - ref[e]));
            ASSERT_NEAR(got[s][e].real(), ref[e].real(), 1e-2)
                << "slot " << s << " entry " << e;
            ASSERT_NEAR(got[s][e].imag(), ref[e].imag(), 1e-2)
                << "slot " << s << " entry " << e;
        }
    }
    std::cout << "CCMM worst absolute error: " << worst << std::endl;
}

// Diagnostic: which of the four cross products is at fault.
//
// Giving an operand a zero c1 (a trivial encryption of the same message)
// switches off the two GEMMs that consume it, so each configuration isolates a
// different part of the pipeline:
//   a trivial, b trivial -> C00 only
//   a trivial, b real    -> C00 + C01, no relinearisation contribution
//   a real,    b trivial -> C00 + C10, no relinearisation contribution
//   a real,    b real    -> all four, plus the relinearisation
// The message is the same in every case, so all four must land on the same
// reference product.
TEST(HEonGPU, CKKS_BatchMatrix_CCMMComponentIsolation)
{
    const size_t degree = 4096;
    const int d = 8;

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes({60, 50}, {60});
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder_slots(context);
    heongpu::HEArithmeticOperator<S> ops(context, encoder_slots);

    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
    std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
    heongpu::Galoiskey<S> galois_key(context, rot);
    keygen.generate_galois_key(galois_key, secret);
    heongpu::Relinkey<S> relin_key(context);
    keygen.generate_relin_key(relin_key, secret);

    heongpu::HEBatchMatrixOperator<S> op(context, layout);
    heongpu::BatchMatrixEncoder encoder(layout.k);

    // A component spans every active limb, not just the first; hardcoding one
    // limb here silently truncates the trivial encryptions built below.
    const size_t comp =
        static_cast<size_t>(context->get_ciphertext_modulus_count()) * degree;
    // The two scales are deliberately asymmetric. Step 1 CMTs the right
    // operand, and that key switching noise ends up multiplied by the left
    // operand in the GEMM, so after decoding by scale_a * scale_b the surviving
    // error is e_cmt * sqrt(d*k) / scale_b -- it depends on scale_b alone.
    // scale_b therefore takes as many bits as the product can spare, and
    // scale_a only enough to quantise the left operand.
    const double scale_a = std::pow(2.0, 25);
    const double scale_b = std::pow(2.0, 35);
    const int nslots = encoder.slots();

    std::vector<std::vector<cd>> M(
        nslots, std::vector<cd>(static_cast<size_t>(d) * d));
    std::vector<std::vector<cd>> U(
        nslots, std::vector<cd>(static_cast<size_t>(d) * d));
    std::mt19937_64 rng(13579u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& m : M)
        for (auto& z : m)
            z = cd(dist(rng), dist(rng));
    for (auto& u : U)
        for (auto& z : u)
            z = cd(dist(rng), dist(rng));

    auto make_operand = [&](const std::vector<std::vector<cd>>& mat,
                            double scale, bool trivial,
                            std::vector<heongpu::Ciphertext<S>>& cts)
    {
        std::vector<int64_t> coeffs;
        encoder.encode(mat, d, d, scale, coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, d,
                                                      columns);
        cts.clear();
        cts.reserve(d);
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> pt(context);
            op.load_coefficients(pt, columns[j], scale);
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, pt);
            if (trivial)
            {
                // c0 = m, c1 = 0 encrypts m with no noise and no secret.
                c.store_in_device();
                pt.store_in_device();
                cudaMemcpy(c.data(), pt.data(), comp * sizeof(Data64),
                           cudaMemcpyDeviceToDevice);
                cudaMemset(c.data() + comp, 0, comp * sizeof(Data64));
            }
            cts.push_back(std::move(c));
        }
    };

    auto run = [&](bool a_trivial, bool b_trivial)
    {
        std::vector<heongpu::Ciphertext<S>> ca, cb;
        make_operand(M, scale_a, a_trivial, ca);
        make_operand(U, scale_b, b_trivial, cb);

        std::vector<heongpu::Ciphertext<S>*> pa, pb;
        for (auto& c : ca)
            pa.push_back(&c);
        for (auto& c : cb)
            pb.push_back(&c);

        std::vector<heongpu::Ciphertext<S>> out;
        op.ccmm(out, pa, pb, galois_key, relin_key, ops, /*rescale=*/false);

        std::vector<std::vector<int64_t>> out_columns(d);
        for (int j = 0; j < d; ++j)
        {
            heongpu::Plaintext<S> pt(context);
            decryptor.decrypt(pt, out[j]);
            op.extract_coefficients(out_columns[j], pt);
        }
        std::vector<int64_t> got_coeffs;
        heongpu::split_matrix_encryption_coefficients(out_columns, layout, d, d,
                                                      got_coeffs);
        std::vector<std::vector<cd>> got;
        encoder.decode(got_coeffs, d, d, scale_a * scale_b, got);

        double worst = 0.0;
        for (int s = 0; s < nslots; ++s)
        {
            const std::vector<cd> ref = matmul(M[s], U[s], d, d, d);
            for (size_t e = 0; e < ref.size(); ++e)
                worst = std::max(worst, std::abs(got[s][e] - ref[e]));
        }
        return worst;
    };

    const std::pair<const char*, std::pair<bool, bool>> cases[] = {
        {"a=trivial b=trivial", {true, true}},
        {"a=trivial b=real   ", {true, false}},
        {"a=real    b=trivial", {false, true}},
        {"a=real    b=real   ", {false, false}}};

    for (const auto& c : cases)
    {
        const double worst = run(c.second.first, c.second.second);
        std::cout << "[isolation] " << c.first << " : " << worst << std::endl;
        EXPECT_LT(worst, 0.01) << c.first;
    }
}
