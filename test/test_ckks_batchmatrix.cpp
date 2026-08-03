// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <complex>
#include <random>
#include <vector>

namespace
{
    using cd = std::complex<double>;

    std::vector<std::vector<cd>> random_batch(int nslots, int rows, int cols,
                                              unsigned seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-4.0, 4.0);
        std::vector<std::vector<cd>> batch(
            nslots, std::vector<cd>(static_cast<size_t>(rows) * cols));
        for (auto& m : batch)
            for (auto& z : m)
                z = cd(dist(rng), dist(rng));
        return batch;
    }

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

    /// Negacyclic product in R_k = Z[Y]/(Y^k + 1).
    std::vector<double> negacyclic_mul(const std::vector<double>& a,
                                       const std::vector<double>& b, int k)
    {
        std::vector<double> c(k, 0.0);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
            {
                const int s = i + j;
                if (s < k)
                    c[s] += a[i] * b[j];
                else
                    c[s - k] -= a[i] * b[j];
            }
        return c;
    }
} // namespace

TEST(HEonGPU, CKKS_BatchMatrix_Layout)
{
    heongpu::BatchMatrixLayout layout(8192, 16);
    EXPECT_EQ(layout.N, 8192);
    EXPECT_EQ(layout.d, 16);
    EXPECT_EQ(layout.k, 512);
    EXPECT_EQ(layout.batch, 256);

    // d must divide N, be a power of two, and leave k >= 2.
    EXPECT_THROW(heongpu::BatchMatrixLayout(8192, 3), std::invalid_argument);
    EXPECT_THROW(heongpu::BatchMatrixLayout(8192, 8192), std::invalid_argument);
    EXPECT_THROW(heongpu::BatchMatrixLayout(1000, 4), std::invalid_argument);
}

// Definition 1: encode then decode must recover the batch.
TEST(HEonGPU, CKKS_BatchMatrix_EncodeDecodeRoundTrip)
{
    const int k = 32;
    const int rows = 4, cols = 3;
    const double scale = std::pow(2.0, 40);

    heongpu::BatchMatrixEncoder encoder(k);
    ASSERT_EQ(encoder.slots(), k / 2);

    const auto batch = random_batch(encoder.slots(), rows, cols, 12345u);

    std::vector<int64_t> coeffs;
    encoder.encode(batch, rows, cols, scale, coeffs);
    ASSERT_EQ(coeffs.size(),
              static_cast<size_t>(rows) * cols * k);

    std::vector<std::vector<cd>> back;
    encoder.decode(coeffs, rows, cols, scale, back);
    ASSERT_EQ(back.size(), batch.size());

    for (size_t j = 0; j < batch.size(); ++j)
        for (size_t e = 0; e < batch[j].size(); ++e)
        {
            EXPECT_NEAR(back[j][e].real(), batch[j][e].real(), 1e-6);
            EXPECT_NEAR(back[j][e].imag(), batch[j][e].imag(), 1e-6);
        }
}

// The encoding is a ring homomorphism: multiplying two encoded entries in R_k
// must equal the entrywise complex product of the batches. This is the property
// the whole batch matrix construction rests on.
TEST(HEonGPU, CKKS_BatchMatrix_EncodingIsMultiplicative)
{
    const int k = 16;
    const double scale = std::pow(2.0, 25);
    heongpu::BatchMatrixEncoder encoder(k);
    const int nslots = encoder.slots();

    const auto a = random_batch(nslots, 1, 1, 777u);
    const auto b = random_batch(nslots, 1, 1, 999u);

    std::vector<int64_t> ea, eb;
    encoder.encode(a, 1, 1, scale, ea);
    encoder.encode(b, 1, 1, scale, eb);

    std::vector<double> fa(k), fb(k);
    for (int t = 0; t < k; ++t)
    {
        fa[t] = static_cast<double>(ea[t]);
        fb[t] = static_cast<double>(eb[t]);
    }
    const std::vector<double> prod = negacyclic_mul(fa, fb, k);

    // The product carries scale^2, so decode against that.
    std::vector<int64_t> pi(k);
    for (int t = 0; t < k; ++t)
        pi[t] = static_cast<int64_t>(std::llround(prod[t]));

    std::vector<std::vector<cd>> got;
    encoder.decode(pi, 1, 1, scale * scale, got);

    for (int j = 0; j < nslots; ++j)
    {
        const cd expected = a[j][0] * b[j][0];
        EXPECT_NEAR(got[j][0].real(), expected.real(), 1e-3)
            << "slot " << j;
        EXPECT_NEAR(got[j][0].imag(), expected.imag(), 1e-3)
            << "slot " << j;
    }
}

// Definition 2: the column coefficient vectors must round-trip, and must place
// entry (i,j) at coefficient i + d*t as the definition requires.
TEST(HEonGPU, CKKS_BatchMatrix_MatrixEncryptionCoefficientRoundTrip)
{
    const int N = 1024, d = 8;
    heongpu::BatchMatrixLayout layout(N, d);
    const int cols = 5;
    const double scale = std::pow(2.0, 30);

    heongpu::BatchMatrixEncoder encoder(layout.k);
    const auto batch = random_batch(encoder.slots(), d, cols, 4242u);

    std::vector<int64_t> coeffs;
    encoder.encode(batch, d, cols, scale, coeffs);

    std::vector<std::vector<int64_t>> columns;
    heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, cols,
                                                  columns);
    ASSERT_EQ(columns.size(), static_cast<size_t>(cols));
    for (const auto& c : columns)
        ASSERT_EQ(c.size(), static_cast<size_t>(N));

    // Spot-check the placement rule m_j[i + d*t] = M[i][j][t].
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < d; ++i)
            for (int t = 0; t < layout.k; ++t)
                ASSERT_EQ(
                    columns[j][i + static_cast<size_t>(d) * t],
                    coeffs[(static_cast<size_t>(i) * cols + j) * layout.k + t])
                    << "i=" << i << " j=" << j << " t=" << t;

    std::vector<int64_t> back;
    heongpu::split_matrix_encryption_coefficients(columns, layout, d, cols,
                                                  back);
    EXPECT_EQ(back, coeffs);
}

// Every automorphism X -> X^(2kt+1) used by CMT must be an ordinary CKKS slot
// rotation, otherwise Algorithm 3 cannot run on standard Galois keys.
TEST(HEonGPU, CKKS_BatchMatrix_CMTRotationIndicesAreSlotRotations)
{
    for (int d : {2, 4, 8, 16})
    {
        const int N = 1024;
        heongpu::BatchMatrixLayout layout(N, d);
        const std::vector<int> idx =
            heongpu::get_batch_cmt_rotation_indices(layout);

        // One index per t in [d], minus the identity at t = 0.
        EXPECT_EQ(idx.size(), static_cast<size_t>(d - 1)) << "d=" << d;

        // Each returned index r must satisfy 5^r == 2kt+1 (mod 2N) for some t.
        const uint64_t mod = 2ull * N;
        for (int r : idx)
        {
            uint64_t g = 1;
            for (int s = 0; s < r; ++s)
                g = (g * 5) % mod;
            EXPECT_EQ((g - 1) % (2ull * layout.k), 0u)
                << "d=" << d << " r=" << r;
        }
    }
}
