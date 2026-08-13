// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The shared-plaintext form of Algorithm 1: the batch PCMM against a weight
// that is the same real matrix for every batched instance.
//
// This is the only kind of plaintext a model weight ever is, because the batch
// axis carries k/2 independent INPUTS through one model. Definition 1 puts the
// batch index in the evaluation domain of R_k, so such a weight encodes to the
// CONSTANT polynomial -- and the whole subring apparatus on the plaintext side
// then has nothing to do.
//
// The shape used throughout is the 16-batch one: N = 4096 with d = 128 gives
// k = 32 and batch = k/2 = 16.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using cd = std::complex<double>;

    /// Reference real-by-real matrix product against complex batched data.
    std::vector<cd> matmul(const std::vector<cd>& a,
                           const std::vector<double>& b, int n, int m, int p)
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

    /// N = 4096, d = 128 => k = 32, batch = 16. The activation is d tokens by
    /// @c in_channels channels; the weight is in_channels by out_channels.
    struct Shape
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 128;
        static constexpr int in_channels = 8;
        // Deliberately not a multiple of the widest column tile, so the
        // kernel's short-tile tail runs in every test rather than in none.
        static constexpr int out_channels = 6;
    };

    struct Fixture
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder_slots;
        heongpu::BatchMatrixLayout layout;
        std::unique_ptr<heongpu::HEBatchMatrixOperator<S>> op;
        std::unique_ptr<heongpu::BatchMatrixEncoder> encoder;

        Fixture()
            : context(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            context->set_poly_modulus_degree(Shape::degree);
            context->set_coeff_modulus_bit_sizes({60, 50}, {60});
            context->generate();

            keygen.reset(new heongpu::HEKeyGenerator<S>(context));
            secret.reset(new heongpu::Secretkey<S>(context));
            keygen->generate_secret_key(*secret);
            pub.reset(new heongpu::Publickey<S>(context));
            keygen->generate_public_key(*pub, *secret);
            encryptor.reset(new heongpu::HEEncryptor<S>(context, *pub));
            decryptor.reset(new heongpu::HEDecryptor<S>(context, *secret));
            encoder_slots.reset(new heongpu::HEEncoder<S>(context));

            layout = heongpu::BatchMatrixLayout(
                static_cast<int>(Shape::degree), Shape::d);
            op.reset(new heongpu::HEBatchMatrixOperator<S>(context, layout));
            encoder.reset(new heongpu::BatchMatrixEncoder(layout.k));
        }

        /// Encrypt a batch of d x cols complex matrices as a matrix
        /// encryption, one ciphertext per column.
        void encrypt_matrix(const std::vector<std::vector<cd>>& mat, int cols,
                            double scale,
                            std::vector<heongpu::Ciphertext<S>>& cts)
        {
            std::vector<int64_t> coeffs;
            encoder->encode(mat, Shape::d, cols, scale, coeffs);
            std::vector<std::vector<int64_t>> columns;
            heongpu::build_matrix_encryption_coefficients(
                coeffs, layout, Shape::d, cols, columns);
            cts.clear();
            cts.reserve(cols);
            for (int j = 0; j < cols; ++j)
            {
                heongpu::Plaintext<S> pt(context);
                op->load_coefficients(pt, columns[j], scale);
                heongpu::Ciphertext<S> c(context);
                encryptor->encrypt(c, pt);
                cts.push_back(std::move(c));
            }
        }

        /// The batched instances a matrix encryption decrypts to.
        std::vector<std::vector<cd>>
        decrypt_matrix(std::vector<heongpu::Ciphertext<S>>& cts, int cols,
                       double scale)
        {
            std::vector<std::vector<int64_t>> out_columns(cols);
            for (int j = 0; j < cols; ++j)
            {
                heongpu::Plaintext<S> pt(context);
                decryptor->decrypt(pt, cts[j]);
                op->extract_coefficients(out_columns[j], pt);
            }
            std::vector<int64_t> coeffs;
            heongpu::split_matrix_encryption_coefficients(
                out_columns, layout, Shape::d, cols, coeffs);
            std::vector<std::vector<cd>> got;
            encoder->decode(coeffs, Shape::d, cols, scale, got);
            return got;
        }
    };

    /// Every device word of a ciphertext, both components, every live limb.
    std::vector<Data64> download(const heongpu::HEContext<S>& context,
                                 heongpu::Ciphertext<S>& ct)
    {
        const int limbs =
            context->get_ciphertext_modulus_count() - ct.depth();
        const size_t words =
            2u * static_cast<size_t>(limbs) * Shape::degree;
        std::vector<Data64> host(words);
        EXPECT_EQ(cudaMemcpy(host.data(), ct.data(), words * sizeof(Data64),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return host;
    }

    std::vector<double> random_weight(int rows, int cols, uint64_t seed)
    {
        std::vector<double> w(static_cast<size_t>(rows) * cols);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& v : w)
            v = dist(rng);
        return w;
    }

    std::vector<std::vector<cd>> random_batch(int nslots, int rows, int cols,
                                              uint64_t seed)
    {
        std::vector<std::vector<cd>> m(
            nslots, std::vector<cd>(static_cast<size_t>(rows) * cols));
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& mat : m)
            for (auto& z : mat)
                z = cd(dist(rng), dist(rng));
        return m;
    }
} // namespace

// The premise the whole fast path rests on, checked on the host and
// independently of any GPU: a real matrix that does not vary across the batch
// encodes to the CONSTANT polynomial, so only the Y^0 coefficient survives.
//
// If this ever fails, the fast path is wrong and not merely slow.
TEST(HEonGPU, CKKS_BatchMatrix_SharedEncodingIsConstantPolynomial)
{
    for (const int k : {8, 16, 32, 64})
    {
        heongpu::BatchMatrixEncoder encoder(k);
        const int nslots = encoder.slots();
        const int rows = 3;
        const int cols = 5;
        const double scale = std::pow(2.0, 30);

        const std::vector<double> w = random_weight(rows, cols, 24680u + k);
        std::vector<std::vector<cd>> batch(
            nslots, std::vector<cd>(static_cast<size_t>(rows) * cols));
        for (int s = 0; s < nslots; ++s)
            for (size_t e = 0; e < w.size(); ++e)
                batch[s][e] = cd(w[e], 0.0);

        std::vector<int64_t> coeffs;
        encoder.encode(batch, rows, cols, scale, coeffs);
        ASSERT_EQ(coeffs.size(), w.size() * static_cast<size_t>(k));

        for (size_t e = 0; e < w.size(); ++e)
        {
            const int64_t* run = coeffs.data() + e * k;
            EXPECT_EQ(run[0], std::llround(scale * w[e]))
                << "k " << k << " entry " << e;
            for (int t = 1; t < k; ++t)
                EXPECT_EQ(run[t], 0)
                    << "k " << k << " entry " << e << " power " << t;
        }
    }
}

// Where the two encoders agree on the integers, the two PCMM routes agree on
// every word. That is the claim that makes the fast path safe: the subring
// transform is Z_p-linear and the plaintext is a scalar, so
// T^-1(sum_t T(a_t) * v_t) = sum_t v_t * a_t EXACTLY -- not to within noise, so
// a tolerance-based comparison would pass even if the reasoning were wrong.
//
// The agreement of the ENCODERS is a separate and weaker matter, and this test
// asserts its precondition rather than assuming it: at this scale the general
// encoder's floating-point residue must round away to exact zeros. See
// SharedEncodingIsExactWhereTheGeneralOneDrifts for the regime where it does
// not, and where these two deliberately differ.
TEST(HEonGPU, CKKS_BatchMatrix_SharedPcmmAgreesWithGeneralWordForWord)
{
    Fixture f;
    const int nslots = f.encoder->slots();
    ASSERT_EQ(nslots, 16) << "the batch-16 shape is the point of this file";

    // The product lands at scale_in * scale_w and a coefficient of a matrix
    // encryption is bounded by sqrt(2) * scale, so the output reaches
    // inner * sqrt(2) * scale_in * scale_w ~ 2^59.5 here. extract_coefficients
    // reports a centered coefficient as an int64 and throws above 2^63, which
    // 2^30 apiece would reach. 2^28 also keeps the general encoder's rounding
    // residue below the 2^51.5 where it stops vanishing, which the
    // word-for-word comparison needs.
    const double scale_in = std::pow(2.0, 28);
    const double scale_w = std::pow(2.0, 28);

    const std::vector<std::vector<cd>> M =
        random_batch(nslots, Shape::d, Shape::in_channels, 97531u);
    std::vector<heongpu::Ciphertext<S>> ct;
    f.encrypt_matrix(M, Shape::in_channels, scale_in, ct);

    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);

    const std::vector<double> w =
        random_weight(Shape::in_channels, Shape::out_channels, 1357u);
    const int depth = ct.front().depth();

    // The general route: a full length-k encode, then the subring transforms.
    std::vector<std::vector<cd>> weight_batch(
        nslots, std::vector<cd>(w.size()));
    for (int s = 0; s < nslots; ++s)
        for (size_t e = 0; e < w.size(); ++e)
            weight_batch[s][e] = cd(w[e], 0.0);
    std::vector<int64_t> coeffs;
    f.encoder->encode(weight_batch, Shape::in_channels, Shape::out_channels,
                      scale_w, coeffs);

    // The precondition. 2^30 * 1 * 1.8e-16 is far below a half, so every power
    // above Y^0 must have rounded to exactly zero and the two encoders must
    // have produced the same integers. If this ever fires, the word-for-word
    // comparison below is testing the encoders and not the cancellation, and
    // the scale here is what needs lowering.
    for (size_t e = 0; e < w.size(); ++e)
    {
        const int64_t* run = coeffs.data() + e * f.layout.k;
        ASSERT_EQ(run[0], std::llround(scale_w * w[e])) << "entry " << e;
        for (int t = 1; t < f.layout.k; ++t)
            ASSERT_EQ(run[t], 0) << "entry " << e << " power " << t;
    }

    f.op->encode_plaintext_matrix(coeffs, Shape::in_channels,
                                  Shape::out_channels, depth, scale_w);
    ASSERT_FALSE(f.op->plaintext_is_shared());

    std::vector<heongpu::Ciphertext<S>> general;
    f.op->pcmm(general, in, /*rescale=*/false);
    ASSERT_EQ(general.size(), static_cast<size_t>(Shape::out_channels));

    // The shared route: one rounded integer per entry, no transform at all.
    f.op->encode_shared_plaintext_matrix(w, Shape::in_channels,
                                         Shape::out_channels, depth, scale_w);
    ASSERT_TRUE(f.op->plaintext_is_shared());

    std::vector<heongpu::Ciphertext<S>> shared;
    f.op->pcmm(shared, in, /*rescale=*/false);
    ASSERT_EQ(shared.size(), static_cast<size_t>(Shape::out_channels));

    for (int j = 0; j < Shape::out_channels; ++j)
    {
        EXPECT_EQ(general[j].depth(), shared[j].depth()) << "column " << j;
        EXPECT_DOUBLE_EQ(general[j].scale(), shared[j].scale())
            << "column " << j;

        const std::vector<Data64> a = download(f.context, general[j]);
        const std::vector<Data64> b = download(f.context, shared[j]);
        ASSERT_EQ(a.size(), b.size()) << "column " << j;
        for (size_t i = 0; i < a.size(); ++i)
        {
            ASSERT_EQ(a[i], b[i])
                << "column " << j << " word " << i
                << ": the two PCMM routes disagree on identical integers, so "
                   "the subring cancellation is not exact";
        }
    }
}

// The regime the corrections record warns about, pinned as a property rather
// than left as a caveat in a comment.
//
// project() encodes its weight at rescale_prime(x), a PRIME of 40 to 60 bits --
// which is exactly where the general encoder's ~1.8e-16 relative residue stops
// rounding to zero. At 2^60 it survives as hundreds of integer units at powers
// of Y that carry no information at all, and those are real plaintext error
// riding on the weight. The shared route rounds the value itself, so it is
// exact at every scale.
//
// So the two are NOT interchangeable here, and the shared one is the correct
// one. An equality assertion at this scale would fail, and it would be the
// general encoder that was wrong.
TEST(HEonGPU, CKKS_BatchMatrix_SharedEncodingIsExactWhereTheGeneralOneDrifts)
{
    const int k = 32;
    heongpu::BatchMatrixEncoder encoder(k);
    const int nslots = encoder.slots();
    const int rows = 4;
    const int cols = 4;

    // A 60-bit scale, the top of the prime range project() can meet.
    const double scale = std::pow(2.0, 60);

    const std::vector<double> w = random_weight(rows, cols, 31337u);
    std::vector<std::vector<cd>> batch(
        nslots, std::vector<cd>(static_cast<size_t>(rows) * cols));
    for (int s = 0; s < nslots; ++s)
        for (size_t e = 0; e < w.size(); ++e)
            batch[s][e] = cd(w[e], 0.0);

    std::vector<int64_t> coeffs;
    encoder.encode(batch, rows, cols, scale, coeffs);

    int64_t worst_residue = 0;
    for (size_t e = 0; e < w.size(); ++e)
    {
        const int64_t* run = coeffs.data() + e * k;
        for (int t = 1; t < k; ++t)
            worst_residue =
                std::max<int64_t>(worst_residue, std::llabs(run[t]));
    }

    // The point of the test: at this scale the general encoder does NOT return
    // a clean delta. If this ever becomes zero the encoder has been made exact
    // and the caveat in encode_shared_plaintext_matrix can be dropped.
    EXPECT_GT(worst_residue, 0)
        << "the general encoder no longer drifts at 2^60; re-read the "
           "accuracy note on encode_shared_plaintext_matrix";
    std::cout << "general encoder worst spurious coefficient at 2^60: "
              << worst_residue << std::endl;
}

// End to end at the 16-batch shape: sixteen independent inputs, one shared
// weight, checked against a host reference per instance.
TEST(HEonGPU, CKKS_BatchMatrix_SharedPcmmMatchesReference)
{
    Fixture f;
    const int nslots = f.encoder->slots();
    ASSERT_EQ(nslots, 16);

    // The product lands at scale_in * scale_w and a coefficient of a matrix
    // encryption is bounded by sqrt(2) * scale, so the output reaches
    // inner * sqrt(2) * scale_in * scale_w ~ 2^59.5 here. extract_coefficients
    // reports a centered coefficient as an int64 and throws above 2^63, which
    // 2^30 apiece would reach. 2^28 also keeps the general encoder's rounding
    // residue below the 2^51.5 where it stops vanishing, which the
    // word-for-word comparison needs.
    const double scale_in = std::pow(2.0, 28);
    const double scale_w = std::pow(2.0, 28);

    const std::vector<std::vector<cd>> M =
        random_batch(nslots, Shape::d, Shape::in_channels, 24680u);
    std::vector<heongpu::Ciphertext<S>> ct;
    f.encrypt_matrix(M, Shape::in_channels, scale_in, ct);

    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);

    const std::vector<double> w =
        random_weight(Shape::in_channels, Shape::out_channels, 8642u);
    f.op->encode_shared_plaintext_matrix(w, Shape::in_channels,
                                         Shape::out_channels,
                                         ct.front().depth(), scale_w);

    std::vector<heongpu::Ciphertext<S>> out;
    f.op->pcmm(out, in, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(Shape::out_channels));

    const std::vector<std::vector<cd>> got =
        f.decrypt_matrix(out, Shape::out_channels, scale_in * scale_w);
    ASSERT_EQ(got.size(), static_cast<size_t>(nslots));

    double worst = 0.0;
    for (int s = 0; s < nslots; ++s)
    {
        // Every instance carries its OWN input through the SAME weight, which
        // is exactly what the batch axis is for on this path.
        const std::vector<cd> ref = matmul(M[s], w, Shape::d,
                                           Shape::in_channels,
                                           Shape::out_channels);
        ASSERT_EQ(got[s].size(), ref.size());
        for (size_t e = 0; e < ref.size(); ++e)
        {
            worst = std::max(worst, std::abs(got[s][e] - ref[e]));
            ASSERT_NEAR(got[s][e].real(), ref[e].real(), 1e-3)
                << "instance " << s << " entry " << e;
            ASSERT_NEAR(got[s][e].imag(), ref[e].imag(), 1e-3)
                << "instance " << s << " entry " << e;
        }
    }
    std::cout << "shared PCMM worst absolute error: " << worst << std::endl;
}

// The two forms are exclusive, and uploading one must clear the other. Getting
// this wrong would read a [limb][row][col] scalar buffer with
// [limb][row][col][k] strides, which reads off the end rather than failing.
TEST(HEonGPU, CKKS_BatchMatrix_SharedPlaintextFormIsExclusive)
{
    Fixture f;
    const int nslots = f.encoder->slots();
    const double scale = std::pow(2.0, 30);

    const std::vector<double> w =
        random_weight(Shape::in_channels, Shape::out_channels, 5150u);
    f.op->encode_shared_plaintext_matrix(w, Shape::in_channels,
                                         Shape::out_channels, 0, scale);
    EXPECT_TRUE(f.op->plaintext_is_shared());

    std::vector<std::vector<cd>> batch(nslots, std::vector<cd>(w.size()));
    for (int s = 0; s < nslots; ++s)
        for (size_t e = 0; e < w.size(); ++e)
            batch[s][e] = cd(w[e], 0.0);
    std::vector<int64_t> coeffs;
    f.encoder->encode(batch, Shape::in_channels, Shape::out_channels, scale,
                      coeffs);
    f.op->encode_plaintext_matrix(coeffs, Shape::in_channels,
                                  Shape::out_channels, 0, scale);
    EXPECT_FALSE(f.op->plaintext_is_shared());

    f.op->encode_shared_plaintext_matrix(w, Shape::in_channels,
                                         Shape::out_channels, 0, scale);
    EXPECT_TRUE(f.op->plaintext_is_shared());
}

TEST(HEonGPU, CKKS_BatchMatrix_SharedPlaintextRejectsWrongShape)
{
    Fixture f;
    const std::vector<double> w =
        random_weight(Shape::in_channels, Shape::out_channels, 11u);
    EXPECT_THROW(f.op->encode_shared_plaintext_matrix(
                     w, Shape::in_channels, Shape::out_channels + 1, 0,
                     std::pow(2.0, 30)),
                 std::invalid_argument);
}
