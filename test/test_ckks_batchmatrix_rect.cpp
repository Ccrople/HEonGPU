// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU validation of Algorithm 5, the rectangular batch PCMM.
//
// Algorithm 5 exists because Algorithm 1 cannot contract over the batch axis.
// A batch matrix encryption carries k/2 independent d x d matrices and
// Algorithm 1 multiplies each of them by its own plaintext block, in parallel
// and without ever mixing them. That is exactly what is wanted when the k/2
// slots hold k/2 independent INPUTS. It is exactly what is not wanted when a
// single input is too wide for one d x d block and the k/2 slots have to hold
// the BLOCKS of that one input: the block products then have to be summed, and
// no amount of Algorithm 1 will sum them.
//
// Theorem 3 sums them. The sum over the k/2 evaluation points of an R_k element
// is k/2 times its constant term, so the summation is a constant-term
// extraction, and the CMT is what makes that extraction a data movement rather
// than an arithmetic one: read at k = 2 the constant terms are a transpose away
// from being the leading d ciphertexts, and the rest are then simply dropped.
//
// Two conventions are tested, and the difference between them is the whole
// reason a chain of projections is affordable. See BlockAxis.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using cd = std::complex<double>;

    /// Everything the rectangular tests need from a context, at one shape.
    ///
    /// The ring is the smallest HEonGPU supports, 4096, and that is not a
    /// convenience: Algorithm 5's summation asks for the whole rotation group,
    /// N/2 - 1 Galois keys, and that count depends on N ALONE -- not on d, not
    /// on the width of whatever is being multiplied. So the ring degree is the
    /// only thing standing between a test and a key set that does not fit on
    /// the card. What makes 2047 keys affordable here is the chain: at three
    /// limbs and one special prime a key is 768 KiB, so the set is 1.5 GiB.
    /// The same 2047 keys on the 70-limb chain a transformer block needs would
    /// be 96 GiB.
    struct RectFixture
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::HEBatchMatrixOperator<S>> op;
        heongpu::BatchMatrixLayout layout;
        int n = 0;

        RectFixture(int degree, int d)
            : context(heongpu::GenHEContext<S>(heongpu::sec_level_type::none)),
              layout(degree, d), n(degree)
        {
            context->set_poly_modulus_degree(static_cast<size_t>(degree));
            context->set_coeff_modulus_bit_sizes({50, 40, 40}, {50});
            context->generate();

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
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(context,
                                                                    *encoder);

            // Not const: Galoiskey takes its index list by non-const
            // reference, which it uses to canonicalise the list in place.
            std::vector<int> rot =
                heongpu::get_rectangular_rotation_indices(layout);
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, rot);
            keygen->generate_galois_key(*galois, *secret);

            op = std::make_unique<heongpu::HEBatchMatrixOperator<S>>(context,
                                                                    layout);
        }

        /// Encrypt d column ciphertexts from raw R_k coefficients.
        std::vector<heongpu::Ciphertext<S>>
        encrypt_columns(const std::vector<int64_t>& coeffs, int rows, int cols,
                        double scale)
        {
            std::vector<std::vector<int64_t>> columns;
            heongpu::build_matrix_encryption_coefficients(coeffs, layout, rows,
                                                          cols, columns);
            std::vector<heongpu::Ciphertext<S>> cts;
            cts.reserve(cols);
            for (int j = 0; j < cols; ++j)
            {
                heongpu::Plaintext<S> p(context);
                op->load_coefficients(p, columns[j], scale);
                heongpu::Ciphertext<S> c(context);
                encryptor->encrypt(c, p);
                cts.push_back(std::move(c));
            }
            return cts;
        }

        /// Decrypt d columns back to raw R_k coefficients, column major:
        /// out[j][i + d*t].
        std::vector<std::vector<int64_t>>
        decrypt_columns(std::vector<heongpu::Ciphertext<S>>& ct)
        {
            std::vector<std::vector<int64_t>> res(ct.size());
            for (size_t j = 0; j < ct.size(); ++j)
            {
                heongpu::Plaintext<S> p(context);
                decryptor->decrypt(p, ct[j]);
                op->extract_coefficients(res[j], p);
            }
            return res;
        }
    };

    /// Round a real onto the integer grid of @p scale.
    inline int64_t q(double v, double scale)
    {
        return static_cast<int64_t>(std::llround(v * scale));
    }
} // namespace

// ---------------------------------------------------------------------------
// Convention A: the paper's own statement of Algorithm 5.
// ---------------------------------------------------------------------------
//
// M is d x (N/2), held as k/2 blocks of d x d with block l in BATCH SLOT l.
// U is (N/2) x (N/2), held as the batch plaintext matrix whose slot l is the
// block row l. The product W = M U is d x (N/2).
//
// U is made nonzero only in its leading d columns, so W has exactly one
// nonzero block and every later block must come out zero -- which is a much
// sharper check than a single dense comparison, because the summation is the
// only thing that could put mass in the later blocks.
TEST(HEonGPU, CKKS_BatchMatrix_RectangularPCMMMatchesReference)
{
    const int degree = 4096;
    const int d = 128;
    RectFixture fx(degree, d);

    const int k = fx.layout.k;
    const int blocks = fx.layout.batch; // k/2
    const int half = degree / 2;
    ASSERT_EQ(half, blocks * d);

    heongpu::BatchMatrixEncoder bm(k);
    ASSERT_EQ(bm.slots(), blocks);

    const double scale_m = std::pow(2.0, 25);
    const double scale_u = std::pow(2.0, 25);

    std::mt19937_64 rng(20260805u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::vector<cd>> M(
        blocks, std::vector<cd>(static_cast<size_t>(d) * d));
    std::vector<std::vector<cd>> U(
        blocks, std::vector<cd>(static_cast<size_t>(d) * half, cd(0.0, 0.0)));
    for (int l = 0; l < blocks; ++l)
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
            {
                M[l][static_cast<size_t>(i) * d + j] = cd(dist(rng), 0.0);
                U[l][static_cast<size_t>(i) * half + j] = cd(dist(rng), 0.0);
            }

    // W_0 = sum_l M_l * U_l[:, 0:d], the whole sum over the batch axis.
    std::vector<double> W0(static_cast<size_t>(d) * d, 0.0);
    for (int l = 0; l < blocks; ++l)
        for (int i = 0; i < d; ++i)
            for (int t = 0; t < d; ++t)
            {
                const double a = M[l][static_cast<size_t>(i) * d + t].real();
                for (int j = 0; j < d; ++j)
                    W0[static_cast<size_t>(i) * d + j] +=
                        a * U[l][static_cast<size_t>(t) * half + j].real();
            }

    std::vector<int64_t> m_coeffs, u_coeffs;
    bm.encode(M, d, d, scale_m, m_coeffs);
    bm.encode(U, d, half, scale_u, u_coeffs);

    std::vector<heongpu::Ciphertext<S>> cts =
        fx.encrypt_columns(m_coeffs, d, d, scale_m);
    fx.op->encode_plaintext_matrix(u_coeffs, d, half, 0, scale_u);

    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : cts)
        in.push_back(&c);

    std::vector<heongpu::Ciphertext<S>> out;
    fx.op->rectangular_pcmm(
        out, in, *fx.galois, *fx.ops,
        heongpu::HEBatchMatrixOperator<S>::BlockAxis::slot, /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(d));

    const std::vector<std::vector<int64_t>> got = fx.decrypt_columns(out);
    const double combined = scale_m * scale_u;

    double worst_signal = 0.0, worst_null = 0.0;
    for (int j = 0; j < d; ++j)
    {
        for (int i = 0; i < d; ++i)
        {
            const double v = static_cast<double>(got[j][i]) / combined;
            const double want = W0[static_cast<size_t>(i) * d + j];
            worst_signal = std::max(worst_signal, std::abs(v - want));
            ASSERT_NEAR(v, want, 1e-2 * std::max(1.0, std::abs(want)))
                << "W0 at (" << i << "," << j << ")";
        }
        // Every later block must vanish: U has no mass outside its leading d
        // columns, so anything here is the summation leaking across blocks.
        for (int t = 1; t < blocks; ++t)
            for (int i = 0; i < d; ++i)
            {
                const double v =
                    static_cast<double>(got[j][i + static_cast<size_t>(d) * t]) /
                    combined;
                worst_null = std::max(worst_null, std::abs(v));
                ASSERT_NEAR(v, 0.0, 1e-2)
                    << "block " << t << " row " << i << " column " << j;
            }
    }
    std::cout << "rectangular PCMM (slot axis) worst error: " << worst_signal
              << ", worst residue outside block 0: " << worst_null << std::endl;
}

// ---------------------------------------------------------------------------
// Convention B: the same algorithm, chainable.
// ---------------------------------------------------------------------------
//
// Algorithm 5 hands back its blocks on the Y axis, not on the batch axis, so
// its own output is not a legal input to itself under convention A. Feeding it
// back anyway is legal -- and free -- provided the plaintext is arranged to
// match, because the constant coefficient of an R_k product is
//
//     (a b)_0 = a_0 b_0 - sum_{t>0} a_t b_{k-t},
//
// which is already a contraction over t once b carries its own blocks reversed
// and negated. So: encrypted blocks at Y^t, plaintext block t at Y^{k-t} with a
// minus sign, and no scaling by k/2 because the constant term IS the sum rather
// than k/2 times the mean of it.
//
// This test pins the single-call case; the next pins that the output really is
// a legal input.
namespace
{
    /// Encrypted operand: X is d(tokens) x half(channels), channel
    /// c = t*d + j lands at coefficient i + d*t of column ciphertext j.
    std::vector<int64_t> coefficient_axis_operand(const std::vector<double>& X,
                                                  int d, int half, int k,
                                                  double scale)
    {
        const int blocks = half / d;
        std::vector<int64_t> coeffs(static_cast<size_t>(d) * d * k, 0);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
            {
                int64_t* e =
                    coeffs.data() + (static_cast<size_t>(i) * d + j) * k;
                for (int t = 0; t < blocks; ++t)
                    e[t] = q(X[static_cast<size_t>(i) * half + t * d + j],
                             scale);
            }
        return coeffs;
    }

    /// Plaintext operand: W is half x half, and the block row t of W has to sit
    /// at Y^{k-t} with a minus sign so that the constant coefficient of the
    /// product contracts over t.
    std::vector<int64_t> coefficient_axis_weight(const std::vector<double>& W,
                                                 int d, int half, int k,
                                                 double scale)
    {
        const int blocks = half / d;
        std::vector<int64_t> coeffs(static_cast<size_t>(d) * half * k, 0);
        for (int j = 0; j < d; ++j)
            for (int c = 0; c < half; ++c)
            {
                int64_t* e =
                    coeffs.data() + (static_cast<size_t>(j) * half + c) * k;
                e[0] = q(W[static_cast<size_t>(j) * half + c], scale);
                for (int t = 1; t < blocks; ++t)
                    e[k - t] = -q(
                        W[static_cast<size_t>(t * d + j) * half + c], scale);
            }
        return coeffs;
    }

    std::vector<double> host_project(const std::vector<double>& X,
                                     const std::vector<double>& W, int rows,
                                     int inner, int cols)
    {
        std::vector<double> Y(static_cast<size_t>(rows) * cols, 0.0);
        for (int i = 0; i < rows; ++i)
            for (int c = 0; c < inner; ++c)
            {
                const double a = X[static_cast<size_t>(i) * inner + c];
                if (a == 0.0)
                    continue;
                for (int j = 0; j < cols; ++j)
                    Y[static_cast<size_t>(i) * cols + j] +=
                        a * W[static_cast<size_t>(c) * cols + j];
            }
        return Y;
    }

    /// Read a decrypted convention-B result back as a d x half matrix.
    std::vector<double>
    read_coefficient_axis(const std::vector<std::vector<int64_t>>& got, int d,
                          int half, double scale)
    {
        std::vector<double> Y(static_cast<size_t>(d) * half, 0.0);
        const int blocks = half / d;
        for (int j = 0; j < d; ++j)
            for (int t = 0; t < blocks; ++t)
                for (int i = 0; i < d; ++i)
                    Y[static_cast<size_t>(i) * half + t * d + j] =
                        static_cast<double>(
                            got[j][i + static_cast<size_t>(d) * t]) /
                        scale;
        return Y;
    }
} // namespace

TEST(HEonGPU, CKKS_BatchMatrix_RectangularPCMMOnCoefficientAxis)
{
    const int degree = 4096;
    const int d = 128;
    RectFixture fx(degree, d);

    const int k = fx.layout.k;
    const int half = degree / 2;

    const double scale_x = std::pow(2.0, 25);
    const double scale_w = std::pow(2.0, 25);

    std::mt19937_64 rng(13579u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> X(static_cast<size_t>(d) * half);
    std::vector<double> W(static_cast<size_t>(half) * half);
    for (auto& v : X)
        v = dist(rng);
    for (auto& v : W)
        v = dist(rng) / std::sqrt(static_cast<double>(half));

    const std::vector<double> Y = host_project(X, W, d, half, half);

    const std::vector<int64_t> x_coeffs =
        coefficient_axis_operand(X, d, half, k, scale_x);
    const std::vector<int64_t> w_coeffs =
        coefficient_axis_weight(W, d, half, k, scale_w);

    std::vector<heongpu::Ciphertext<S>> cts =
        fx.encrypt_columns(x_coeffs, d, d, scale_x);
    fx.op->encode_plaintext_matrix(w_coeffs, d, half, 0, scale_w);

    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : cts)
        in.push_back(&c);

    std::vector<heongpu::Ciphertext<S>> out;
    fx.op->rectangular_pcmm(
        out, in, *fx.galois, *fx.ops,
        heongpu::HEBatchMatrixOperator<S>::BlockAxis::coefficient,
        /*rescale=*/false);
    ASSERT_EQ(out.size(), static_cast<size_t>(d));

    const std::vector<double> got = read_coefficient_axis(
        fx.decrypt_columns(out), d, half, scale_x * scale_w);

    double worst = 0.0;
    for (size_t e = 0; e < Y.size(); ++e)
        worst = std::max(worst, std::abs(got[e] - Y[e]));
    std::cout << "rectangular PCMM (coefficient axis) worst error: " << worst
              << std::endl;
    EXPECT_LT(worst, 5e-2);
}

// The property the whole no-batch model rests on: Algorithm 5's output is a
// legal Algorithm 5 input, so a chain of projections needs no re-encoding
// between calls. If this fails, every projection after the first would have to
// pay a homomorphic transform to put its blocks back on the batch axis.
TEST(HEonGPU, CKKS_BatchMatrix_RectangularPCMMChainsWithoutReencoding)
{
    const int degree = 4096;
    const int d = 128;
    RectFixture fx(degree, d);

    const int k = fx.layout.k;
    const int half = degree / 2;

    const double scale_x = std::pow(2.0, 25);
    const double scale_w = std::pow(2.0, 25);

    std::mt19937_64 rng(24680u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> X(static_cast<size_t>(d) * half);
    std::vector<double> W1(static_cast<size_t>(half) * half);
    std::vector<double> W2(static_cast<size_t>(half) * half);
    for (auto& v : X)
        v = dist(rng);
    const double norm = std::sqrt(static_cast<double>(half));
    for (auto& v : W1)
        v = dist(rng) / norm;
    for (auto& v : W2)
        v = dist(rng) / norm;

    const std::vector<double> Y1 = host_project(X, W1, d, half, half);
    const std::vector<double> Y2 = host_project(Y1, W2, d, half, half);

    std::vector<heongpu::Ciphertext<S>> cts = fx.encrypt_columns(
        coefficient_axis_operand(X, d, half, k, scale_x), d, d, scale_x);

    using Axis = heongpu::HEBatchMatrixOperator<S>::BlockAxis;

    // First projection. rescale = true, and it is spent immediately, exactly
    // as the model path does: left unspent the coefficients outgrow int64 long
    // before the second call.
    std::vector<heongpu::Ciphertext<S>> stage1;
    {
        fx.op->encode_plaintext_matrix(
            coefficient_axis_weight(W1, d, half, k, scale_w), d, half, 0,
            scale_w);
        std::vector<heongpu::Ciphertext<S>*> in;
        for (auto& c : cts)
            in.push_back(&c);
        fx.op->rectangular_pcmm(stage1, in, *fx.galois, *fx.ops,
                                Axis::coefficient, /*rescale=*/true);
        for (auto& c : stage1)
            fx.ops->rescale_inplace(c);
    }

    // The rescale divided by the second prime, so the stream is back at
    // scale_x and the second weight is encoded at the next prime down.
    const double stage1_scale = stage1.front().scale();
    const double scale_w2 = std::pow(2.0, 40);

    std::vector<heongpu::Ciphertext<S>> stage2;
    {
        fx.op->encode_plaintext_matrix(
            coefficient_axis_weight(W2, d, half, k, scale_w2), d, half,
            stage1.front().depth(), scale_w2);
        std::vector<heongpu::Ciphertext<S>*> in;
        for (auto& c : stage1)
            in.push_back(&c);
        fx.op->rectangular_pcmm(stage2, in, *fx.galois, *fx.ops,
                                Axis::coefficient, /*rescale=*/false);
    }

    const std::vector<double> got = read_coefficient_axis(
        fx.decrypt_columns(stage2), d, half, stage1_scale * scale_w2);

    double worst = 0.0;
    for (size_t e = 0; e < Y2.size(); ++e)
        worst = std::max(worst, std::abs(got[e] - Y2[e]));
    std::cout << "chained rectangular PCMM worst error: " << worst
              << std::endl;
    EXPECT_LT(worst, 5e-2);
}
