// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU-side validation of the Llama-3 linear algebra on Kang's batch matrix
// multiplication.
//
// The bridge is the only genuinely new piece of arithmetic on this path, and
// everything else rests on it, so it is checked in both directions and against
// the layout it claims to produce -- not merely for round-tripping, which a
// pair of mutually inverse mistakes would also pass.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    /// A context small enough to be quick and deep enough for a bridge, a
    /// projection and the rescales they spend.
    struct Fixture
    {
        static constexpr size_t degree = 4096;
        static constexpr int d = 8;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        heongpu::BatchMatrixLayout layout;
        double scale = std::pow(2.0, 30);

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::llama::Llama3BatchOperator> op;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;

        Fixture()
        {
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes({50, 50, 50}, {50});
            context->generate();

            layout = heongpu::BatchMatrixLayout(static_cast<int>(degree), d);

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
            op = std::make_unique<heongpu::llama::Llama3BatchOperator>(
                context, *encoder, layout, scale);

            // Not const: the Galoiskey constructor takes its shift list by
            // non-const reference.
            std::vector<int> shifts = op->bridge_rotation_indices();
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
        }

        /// k/2 random d x cols real matrices, small enough that a product of
        /// them stays well inside the prime.
        std::vector<std::vector<double>> random_batch(int rows, int cols,
                                                      uint64_t seed) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<std::vector<double>> out(
                layout.batch,
                std::vector<double>(static_cast<size_t>(rows) * cols));
            for (auto& m : out)
                for (auto& v : m)
                    v = dist(rng);
            return out;
        }
    };

    double max_abs_diff(const std::vector<std::vector<double>>& a,
                        const std::vector<std::vector<double>>& b)
    {
        double worst = 0.0;
        for (size_t s = 0; s < a.size(); ++s)
            for (size_t e = 0; e < a[s].size(); ++e)
                worst = std::max(worst, std::abs(a[s][e] - b[s][e]));
        return worst;
    }
} // namespace

// The encrypt/decrypt staging is the reference the rest is measured against,
// so it is checked on its own first: a fault here would look like a bridge
// fault everywhere else.
TEST(HEonGPU, CKKS_Llama3Batch_ActivationRoundTrip)
{
    Fixture f;
    const int cols = 5;
    const auto want = f.random_batch(Fixture::d, cols, 4242u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    ASSERT_EQ(ct.columns(), cols);
    ASSERT_EQ(ct.rows, Fixture::d);

    const auto got = f.op->decrypt(ct, *f.decryptor, f.scale);
    EXPECT_LT(max_abs_diff(want, got), 1e-6);
}

// The bridge, both ways. Round-tripping alone would pass a pair of mutually
// inverse mistakes, so the intermediate slot layout is checked too: slot
// b + (k/2) * u must hold entry (u, j) of matrix b, which is the property the
// strided reductions downstream depend on.
TEST(HEonGPU, CKKS_Llama3Batch_BridgeProducesTheClaimedSlotLayout)
{
    Fixture f;
    const int cols = 3;
    const auto want = f.random_batch(Fixture::d, cols, 909u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    ASSERT_EQ(slots.size(), static_cast<size_t>(cols));

    const int step = f.layout.k / 2;
    for (int j = 0; j < cols; ++j)
    {
        heongpu::Plaintext<S> plain(f.context);
        f.decryptor->decrypt(plain, slots[j]);
        std::vector<double> message;
        f.encoder->decode(message, plain);

        for (int b = 0; b < f.layout.batch; ++b)
            for (int u = 0; u < Fixture::d; ++u)
            {
                const double expect =
                    want[b][static_cast<size_t>(u) * cols + j];
                EXPECT_NEAR(message[b + u * step], expect, 1e-4)
                    << "column " << j << " batch " << b << " row " << u;
            }
    }
}

TEST(HEonGPU, CKKS_Llama3Batch_BridgeIsInvertible)
{
    Fixture f;
    const int cols = 4;
    const auto want = f.random_batch(Fixture::d, cols, 31337u);

    auto ct = f.op->encrypt(want, Fixture::d, cols, *f.encryptor, f.scale);
    auto slots = f.op->to_slots(ct, *f.galois);
    auto back = f.op->from_slots(slots, Fixture::d, *f.galois);

    ASSERT_EQ(back.columns(), cols);
    const auto got = f.op->decrypt(back, *f.decryptor, f.scale);
    EXPECT_LT(max_abs_diff(want, got), 1e-3);
}

// Kang's Algorithm 1 standing in for a projection, against a host product.
// The weight is handed over transposed, because the encrypted operand is on
// the left and a Llama projection is W X.
TEST(HEonGPU, CKKS_Llama3Batch_ProjectMatchesHostProduct)
{
    Fixture f;
    const int in_channels = Fixture::d;
    const int out_channels = 4;

    const auto x = f.random_batch(Fixture::d, in_channels, 5150u);

    std::mt19937_64 rng(2718u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> weight(static_cast<size_t>(in_channels) *
                               out_channels);
    for (auto& v : weight)
        v = dist(rng);

    auto ct = f.op->encrypt(x, Fixture::d, in_channels, *f.encryptor, f.scale);
    auto out = f.op->project(ct, weight, in_channels, out_channels, "test");
    ASSERT_EQ(out.columns(), out_channels);

    const auto got = f.op->decrypt(out, *f.decryptor, f.scale);

    std::vector<std::vector<double>> want(
        f.layout.batch,
        std::vector<double>(static_cast<size_t>(Fixture::d) * out_channels,
                            0.0));
    for (int b = 0; b < f.layout.batch; ++b)
        for (int r = 0; r < Fixture::d; ++r)
            for (int c = 0; c < out_channels; ++c)
            {
                double acc = 0.0;
                for (int t = 0; t < in_channels; ++t)
                    acc += x[b][static_cast<size_t>(r) * in_channels + t] *
                           weight[static_cast<size_t>(t) * out_channels + c];
                want[b][static_cast<size_t>(r) * out_channels + c] = acc;
            }

    EXPECT_LT(max_abs_diff(want, got), 1e-3);
}
