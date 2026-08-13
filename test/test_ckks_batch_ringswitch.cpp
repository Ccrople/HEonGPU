// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Ring switching under Kang's BATCH matrix encoding (Definition 2).
//
// The batch-16 Llama-3 path pins its ring: batch = k/2 = 16 forces k = 32, and
// d = head_dim = 128 then forces N = d*k = 4096. At N = 4096 the 128-bit cap
// is heongpu_128bit_std_parms(4096) = 109 bits of log QP, so the path can hold
// two Q primes and one special and nothing more. Every level beyond that has
// to live at a bigger ring, which makes ring switching the only way this
// encoding is ever run at a real security level.
//
// What this file establishes is the one fact the whole two-ring batch flow
// stands on, and it is NOT the fact the slot-form flow stands on:
//
//   THE DESCENT THEOREM. A matrix encryption at layout (N, d, k) descends
//   under switch_down(m) to m matrix encryptions at layout (N/m, d/m, k) —
//   the SAME k, hence the SAME batch size — and small ciphertext r holds
//   exactly the rows i = r (mod m) of the big matrix, at small row i / m.
//
// The proof is two lines of index algebra and it is worth writing down,
// because it is what makes the crossing free rather than a transform:
// Definition 2 puts entry (i, j) of subring coordinate t at coefficient
// i + d*t of column ciphertext j, and switch_down sends big coefficient p to
// position p / m of small ciphertext p mod m. With m | d, write i = m*i' + r;
// then p = i + d*t = m*(i' + (d/m)*t) + r, so r = p mod m = i mod m and
// p / m = i' + (d/m)*t — which is Definition 2 again at (N/m, d/m, k). The
// subring index t is untouched, so the batch axis crosses the ring intact.
//
// The contrast that matters: a SLOT-form ciphertext does not descend this way
// at all. Big slot s of compose_up's output is sum_j zeta^{j*5^s} * (small
// slot s of ciphertext j) — an m-point twiddled mixture, not a partition. So
// the crossing has to be taken in matrix form, and that is a placement
// constraint on the flow rather than a cost.
//
// Everything here runs at the batch encoding's own scales and shapes, and the
// products are the real ones (pcmm = Algorithm 1, ccmm = Algorithm 4), not
// stand-ins.

#include <heongpu/heongpu.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    using cd = std::complex<double>;

    // A big and a small context sharing an RNS prefix, plus the batch matrix
    // machinery at BOTH rings. The two layouts differ only in (N, d): k is the
    // ring switch's invariant, so one BatchMatrixEncoder serves both.
    struct BatchRingFixture
    {
        heongpu::HEContext<S> big;
        heongpu::HEContext<S> small;

        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;

        std::unique_ptr<heongpu::HERingSwitchOperator<S>> rs;
        std::unique_ptr<heongpu::HERingSwitchOperator<S>::SecretPair> pair;

        std::unique_ptr<heongpu::HEKeyGenerator<S>> small_keygen;
        std::unique_ptr<heongpu::Publickey<S>> small_pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> small_encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> small_decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> small_encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> small_ops;

        heongpu::BatchMatrixLayout layout_big;
        heongpu::BatchMatrixLayout layout_small;
        std::unique_ptr<heongpu::HEBatchMatrixOperator<S>> op_big;
        std::unique_ptr<heongpu::HEBatchMatrixOperator<S>> op_small;
        std::unique_ptr<heongpu::BatchMatrixEncoder> bm;

        int n_big = 0;
        int n_small = 0;
        int m = 0; ///< N_big / N_small, the ring switch's split factor.
        int k = 0; ///< Subring degree, identical at both rings.
        int d_big = 0;
        int d_small = 0;

        /// @param logn_big  Big ring degree, log2.
        /// @param logn_small Small ring degree, log2.
        /// @param k_subring Subring degree; fixes batch = k/2 at BOTH rings.
        /// @param q_bits    Bit sizes of the big Q chain, bottom first.
        /// @param shared    How many of those the small context receives.
        BatchRingFixture(int logn_big, int logn_small, int k_subring,
                         std::vector<int> q_bits, int shared)
            : big(heongpu::GenHEContext<S>(heongpu::sec_level_type::none)),
              small(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            n_big = 1 << logn_big;
            n_small = 1 << logn_small;
            m = n_big / n_small;
            k = k_subring;
            d_big = n_big / k;
            d_small = n_small / k;

            big->set_poly_modulus_degree(static_cast<size_t>(n_big));
            big->set_coeff_modulus_bit_sizes(q_bits, {60, 60});
            big->generate();

            const auto primes = big->get_key_modulus();
            const int big_q = static_cast<int>(q_bits.size());

            std::vector<Data64> q_vals, p_vals;
            for (int i = 0; i < shared; i++)
                q_vals.push_back(primes[i].value);
            p_vals.push_back(primes[big_q].value);

            small->set_poly_modulus_degree(static_cast<size_t>(n_small));
            small->set_coeff_modulus_values(q_vals, p_vals);
            small->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(big);
            secret = std::make_unique<heongpu::Secretkey<S>>(big);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(big);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(big, *pub);
            decryptor = std::make_unique<heongpu::HEDecryptor<S>>(big, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(big);
            ops =
                std::make_unique<heongpu::HEArithmeticOperator<S>>(big,
                                                                   *encoder);

            rs = std::make_unique<heongpu::HERingSwitchOperator<S>>(big, small);
            pair = std::make_unique<
                heongpu::HERingSwitchOperator<S>::SecretPair>(
                rs->make_secret_pair(64, 0xB16C0DEULL));
            rs->generate_keys(*keygen, *secret, pair->embedded);

            small_keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(small);
            small_pub = std::make_unique<heongpu::Publickey<S>>(small);
            small_keygen->generate_public_key(*small_pub, pair->small);
            small_encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(small, *small_pub);
            small_decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(small, pair->small);
            small_encoder = std::make_unique<heongpu::HEEncoder<S>>(small);
            small_ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                small, *small_encoder);

            layout_big = heongpu::BatchMatrixLayout(n_big, d_big);
            layout_small = heongpu::BatchMatrixLayout(n_small, d_small);
            op_big = std::make_unique<heongpu::HEBatchMatrixOperator<S>>(
                big, layout_big);
            op_small = std::make_unique<heongpu::HEBatchMatrixOperator<S>>(
                small, layout_small);
            bm = std::make_unique<heongpu::BatchMatrixEncoder>(k);
        }

        int slots() const { return k / 2; }

        std::vector<std::vector<cd>> random_batch(int rows, int cols,
                                                  unsigned seed) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<std::vector<cd>> batch(
                slots(), std::vector<cd>(static_cast<size_t>(rows) * cols));
            for (auto& mat : batch)
                for (auto& z : mat)
                    z = cd(dist(rng), dist(rng));
            return batch;
        }

        /// Encrypt a rows x cols batch matrix as a big-ring matrix encryption.
        std::vector<heongpu::Ciphertext<S>>
        encrypt_big(const std::vector<std::vector<cd>>& mat, int rows,
                    int cols, double scale)
        {
            std::vector<int64_t> coeffs;
            bm->encode(mat, rows, cols, scale, coeffs);
            std::vector<std::vector<int64_t>> columns;
            heongpu::build_matrix_encryption_coefficients(coeffs, layout_big,
                                                          rows, cols, columns);
            std::vector<heongpu::Ciphertext<S>> cts;
            cts.reserve(static_cast<size_t>(cols));
            for (int j = 0; j < cols; j++)
            {
                heongpu::Plaintext<S> p(big);
                op_big->load_coefficients(p, columns[j], scale);
                heongpu::Ciphertext<S> c(big);
                encryptor->encrypt(c, p);
                cts.push_back(std::move(c));
            }
            return cts;
        }

        /// Encrypt a rows x cols batch matrix directly at the small ring.
        std::vector<heongpu::Ciphertext<S>>
        encrypt_small(const std::vector<std::vector<cd>>& mat, int rows,
                      int cols, double scale)
        {
            std::vector<int64_t> coeffs;
            bm->encode(mat, rows, cols, scale, coeffs);
            std::vector<std::vector<int64_t>> columns;
            heongpu::build_matrix_encryption_coefficients(
                coeffs, layout_small, rows, cols, columns);
            std::vector<heongpu::Ciphertext<S>> cts;
            cts.reserve(static_cast<size_t>(cols));
            for (int j = 0; j < cols; j++)
            {
                heongpu::Plaintext<S> p(small);
                op_small->load_coefficients(p, columns[j], scale);
                heongpu::Ciphertext<S> c(small);
                small_encryptor->encrypt(c, p);
                cts.push_back(std::move(c));
            }
            return cts;
        }

        std::vector<std::vector<cd>>
        decrypt_big(std::vector<heongpu::Ciphertext<S>>& cts, int rows,
                    int cols, double scale)
        {
            std::vector<std::vector<int64_t>> out(cols);
            for (int j = 0; j < cols; j++)
            {
                heongpu::Plaintext<S> p(big);
                decryptor->decrypt(p, cts[j]);
                op_big->extract_coefficients(out[j], p);
            }
            std::vector<int64_t> coeffs;
            heongpu::split_matrix_encryption_coefficients(out, layout_big, rows,
                                                          cols, coeffs);
            std::vector<std::vector<cd>> batch;
            bm->decode(coeffs, rows, cols, scale, batch);
            return batch;
        }

        std::vector<std::vector<cd>>
        decrypt_small(std::vector<heongpu::Ciphertext<S>>& cts, int rows,
                      int cols, double scale)
        {
            std::vector<std::vector<int64_t>> out(cols);
            for (int j = 0; j < cols; j++)
            {
                heongpu::Plaintext<S> p(small);
                small_decryptor->decrypt(p, cts[j]);
                op_small->extract_coefficients(out[j], p);
            }
            std::vector<int64_t> coeffs;
            heongpu::split_matrix_encryption_coefficients(out, layout_small,
                                                          rows, cols, coeffs);
            std::vector<std::vector<cd>> batch;
            bm->decode(coeffs, rows, cols, scale, batch);
            return batch;
        }
    };

    // Descend every column of a big-ring matrix encryption. Returns
    // [row class r][column j], which is the small-ring matrix encryption of
    // rows i = r (mod m).
    std::vector<std::vector<heongpu::Ciphertext<S>>>
    descend_columns(BatchRingFixture& f,
                    std::vector<heongpu::Ciphertext<S>>& big_columns)
    {
        const int cols = static_cast<int>(big_columns.size());
        std::vector<std::vector<heongpu::Ciphertext<S>>> parts(f.m);
        for (int r = 0; r < f.m; r++)
            parts[r].reserve(static_cast<size_t>(cols));
        for (int j = 0; j < cols; j++)
        {
            std::vector<heongpu::Ciphertext<S>> split =
                f.rs->switch_down(big_columns[j], *f.ops);
            for (int r = 0; r < f.m; r++)
                parts[r].push_back(std::move(split[r]));
        }
        return parts;
    }

    // The mirror: interleave m small-ring matrix encryptions back into one.
    std::vector<heongpu::Ciphertext<S>>
    ascend_columns(BatchRingFixture& f,
                   std::vector<std::vector<heongpu::Ciphertext<S>>>& parts)
    {
        const int cols = static_cast<int>(parts[0].size());
        std::vector<heongpu::Ciphertext<S>> out;
        out.reserve(static_cast<size_t>(cols));
        for (int j = 0; j < cols; j++)
        {
            std::vector<heongpu::Ciphertext<S>> column;
            column.reserve(static_cast<size_t>(f.m));
            for (int r = 0; r < f.m; r++)
                column.push_back(parts[r][j]);
            out.push_back(f.rs->compose_up(column, *f.ops));
        }
        return out;
    }

    double worst_error(const std::vector<std::vector<cd>>& got,
                       const std::vector<std::vector<cd>>& want)
    {
        double worst = 0.0;
        for (size_t s = 0; s < want.size(); s++)
        {
            for (size_t i = 0; i < want[s].size(); i++)
            {
                const double e = std::abs(got[s][i] - want[s][i]);
                // std::max keeps the running maximum against a NaN, so a
                // NaN-filled result would report zero error. Test explicitly.
                if (std::isnan(e))
                    return std::numeric_limits<double>::quiet_NaN();
                worst = std::max(worst, e);
            }
        }
        return worst;
    }

    // Host reference: rows x inner times inner x cols, per batch instance.
    std::vector<std::vector<cd>>
    host_matmul(const std::vector<std::vector<cd>>& a,
                const std::vector<std::vector<cd>>& b, int rows, int inner,
                int cols)
    {
        std::vector<std::vector<cd>> out(
            a.size(), std::vector<cd>(static_cast<size_t>(rows) * cols, cd()));
        for (size_t s = 0; s < a.size(); s++)
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    cd acc(0.0, 0.0);
                    for (int t = 0; t < inner; t++)
                        acc += a[s][static_cast<size_t>(i) * inner + t] *
                               b[s][static_cast<size_t>(t) * cols + j];
                    out[s][static_cast<size_t>(i) * cols + j] = acc;
                }
        return out;
    }

} // namespace

// -----------------------------------------------------------------------
// 1. The descent theorem, at the shape the batch-16 path actually pins.
// -----------------------------------------------------------------------

// k = 32 is batch 16, and d_small = 128 is head_dim: this is the one legal
// batch-16 island. The big ring here is logN 13 rather than the eventual 16,
// because what is being asserted is an index identity, and the index identity
// does not care how many times m is applied. Test 3 runs m = 4 for that.
//
// Asserted against the ENCODER, not against a second descent: a shared
// convention error between two ring-switch paths would cancel, and a row
// permutation that happened to be an involution would survive a round trip.
TEST(HEonGPU, CKKS_BatchRingSwitch_MatrixEncodingDescendsByRowDecimation)
{
    BatchRingFixture f(13, 12, 32, {50, 40, 40}, 3);
    ASSERT_EQ(f.m, 2);
    ASSERT_EQ(f.d_big, 256);
    ASSERT_EQ(f.d_small, 128);
    ASSERT_EQ(f.layout_big.batch, f.layout_small.batch);
    ASSERT_EQ(f.layout_big.batch, 16);

    const int cols = 3;
    const double scale = std::pow(2.0, 30);
    const std::vector<std::vector<cd>> M =
        f.random_batch(f.d_big, cols, 1234u);

    std::vector<heongpu::Ciphertext<S>> columns =
        f.encrypt_big(M, f.d_big, cols, scale);
    std::vector<std::vector<heongpu::Ciphertext<S>>> parts =
        descend_columns(f, columns);
    ASSERT_EQ(static_cast<int>(parts.size()), f.m);

    double worst = 0.0;
    for (int r = 0; r < f.m; r++)
    {
        std::vector<std::vector<cd>> got =
            f.decrypt_small(parts[r], f.d_small, cols, scale);
        for (int b = 0; b < f.slots(); b++)
            for (int i = 0; i < f.d_small; i++)
                for (int j = 0; j < cols; j++)
                {
                    const cd want =
                        M[b][static_cast<size_t>(i * f.m + r) * cols + j];
                    const double e = std::abs(
                        got[b][static_cast<size_t>(i) * cols + j] - want);
                    ASSERT_FALSE(std::isnan(e));
                    worst = std::max(worst, e);
                }
    }
    std::cout << "batch descent worst error: " << worst << std::endl;
    EXPECT_LT(worst, 1e-3);
}

// The decimation is a genuine interleave and not, say, a contiguous halving
// that the symmetric random data above could not tell apart. Row i of the big
// matrix is given the value i, so a contiguous split would put 0..127 in part
// 0 and this fails loudly.
TEST(HEonGPU, CKKS_BatchRingSwitch_DescentIsInterleavedNotContiguous)
{
    BatchRingFixture f(13, 12, 32, {50, 40, 40}, 3);

    const int cols = 2;
    const double scale = std::pow(2.0, 30);
    std::vector<std::vector<cd>> M(
        f.slots(), std::vector<cd>(static_cast<size_t>(f.d_big) * cols));
    for (int b = 0; b < f.slots(); b++)
        for (int i = 0; i < f.d_big; i++)
            for (int j = 0; j < cols; j++)
                M[b][static_cast<size_t>(i) * cols + j] =
                    cd(static_cast<double>(i) / f.d_big, 0.0);

    std::vector<heongpu::Ciphertext<S>> columns =
        f.encrypt_big(M, f.d_big, cols, scale);
    std::vector<std::vector<heongpu::Ciphertext<S>>> parts =
        descend_columns(f, columns);

    for (int r = 0; r < f.m; r++)
    {
        std::vector<std::vector<cd>> got =
            f.decrypt_small(parts[r], f.d_small, cols, scale);
        // Part r carries rows r, r+m, r+2m, ... so its first row is r/d_big
        // and its rows step by m/d_big. A contiguous split would start part 1
        // at 0.5 and step by 1/d_big.
        EXPECT_NEAR(got[0][0].real(), static_cast<double>(r) / f.d_big, 1e-3);
        EXPECT_NEAR(got[0][static_cast<size_t>(1) * cols].real(),
                    static_cast<double>(r + f.m) / f.d_big, 1e-3);
    }
}

// -----------------------------------------------------------------------
// 2. m > 2: the theorem is not an accident of halving.
// -----------------------------------------------------------------------

TEST(HEonGPU, CKKS_BatchRingSwitch_DescentHoldsAtSplitFactorFour)
{
    BatchRingFixture f(14, 12, 64, {50, 40, 40}, 3);
    ASSERT_EQ(f.m, 4);
    ASSERT_EQ(f.d_big, 256);
    ASSERT_EQ(f.d_small, 64);
    ASSERT_EQ(f.layout_small.batch, 32);

    const int cols = 3;
    const double scale = std::pow(2.0, 30);
    const std::vector<std::vector<cd>> M =
        f.random_batch(f.d_big, cols, 4321u);

    std::vector<heongpu::Ciphertext<S>> columns =
        f.encrypt_big(M, f.d_big, cols, scale);
    std::vector<std::vector<heongpu::Ciphertext<S>>> parts =
        descend_columns(f, columns);

    double worst = 0.0;
    for (int r = 0; r < f.m; r++)
    {
        std::vector<std::vector<cd>> got =
            f.decrypt_small(parts[r], f.d_small, cols, scale);
        for (int b = 0; b < f.slots(); b++)
            for (int i = 0; i < f.d_small; i++)
                for (int j = 0; j < cols; j++)
                    worst = std::max(
                        worst,
                        std::abs(
                            got[b][static_cast<size_t>(i) * cols + j] -
                            M[b][static_cast<size_t>(i * f.m + r) * cols + j]));
    }
    std::cout << "batch descent worst error at m = 4: " << worst << std::endl;
    EXPECT_LT(worst, 1e-3);
}

// -----------------------------------------------------------------------
// 3. Algorithm 1 commutes with the crossing.
// -----------------------------------------------------------------------

// The batch PCMM contracts over CHANNELS, which are the ciphertext axis, while
// the ring switch decimates ROWS, which are the token axis. They act on
// disjoint axes, so a projection may be placed at either ring — and that is
// the placement freedom the two-ring flow needs, because it means only
// Algorithm 4 is pinned to the small ring.
//
// Checked both ways round rather than only against a host reference: the point
// is commutation, and a host reference alone would not distinguish "both are
// right" from "both are wrong in the same way". The host reference is here too.
TEST(HEonGPU, CKKS_BatchRingSwitch_SharedPCMMCommutesWithTheCrossing)
{
    BatchRingFixture f(13, 12, 64, {50, 40, 40}, 3);
    ASSERT_EQ(f.d_big, 128);
    ASSERT_EQ(f.d_small, 64);

    const int in_cols = 8;
    const int out_cols = 5;
    const double scale = std::pow(2.0, 40);
    const std::vector<std::vector<cd>> X =
        f.random_batch(f.d_big, in_cols, 777u);

    std::mt19937_64 rng(999u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> W(static_cast<size_t>(in_cols) * out_cols);
    for (auto& w : W)
        w = dist(rng);

    // Route A: project at the BIG ring, then descend.
    std::vector<heongpu::Ciphertext<S>> big_in =
        f.encrypt_big(X, f.d_big, in_cols, scale);
    f.op_big->encode_shared_plaintext_matrix(W, in_cols, out_cols,
                                             big_in[0].depth(), scale);
    std::vector<heongpu::Ciphertext<S>> big_out;
    {
        std::vector<heongpu::Ciphertext<S>*> ptr;
        for (auto& c : big_in)
            ptr.push_back(&c);
        f.op_big->pcmm(big_out, ptr);
    }
    for (auto& c : big_out)
        f.ops->rescale_inplace(c);
    const double out_scale = big_out[0].scale();
    std::vector<std::vector<heongpu::Ciphertext<S>>> route_a =
        descend_columns(f, big_out);

    // Route B: descend first, then project at the SMALL ring.
    std::vector<heongpu::Ciphertext<S>> big_in_b =
        f.encrypt_big(X, f.d_big, in_cols, scale);
    std::vector<std::vector<heongpu::Ciphertext<S>>> small_in =
        descend_columns(f, big_in_b);
    std::vector<std::vector<heongpu::Ciphertext<S>>> route_b(f.m);
    f.op_small->encode_shared_plaintext_matrix(W, in_cols, out_cols,
                                               small_in[0][0].depth(), scale);
    for (int r = 0; r < f.m; r++)
    {
        std::vector<heongpu::Ciphertext<S>*> ptr;
        for (auto& c : small_in[r])
            ptr.push_back(&c);
        f.op_small->pcmm(route_b[r], ptr);
        for (auto& c : route_b[r])
            f.small_ops->rescale_inplace(c);
    }

    // Host reference on the interleaved rows.
    std::vector<std::vector<cd>> Wbatch(
        f.slots(), std::vector<cd>(static_cast<size_t>(in_cols) * out_cols));
    for (auto& mat : Wbatch)
        for (size_t i = 0; i < W.size(); i++)
            mat[i] = cd(W[i], 0.0);
    const std::vector<std::vector<cd>> want =
        host_matmul(X, Wbatch, f.d_big, in_cols, out_cols);

    double worst_ab = 0.0, worst_host = 0.0;
    for (int r = 0; r < f.m; r++)
    {
        std::vector<std::vector<cd>> a =
            f.decrypt_small(route_a[r], f.d_small, out_cols, out_scale);
        std::vector<std::vector<cd>> b = f.decrypt_small(
            route_b[r], f.d_small, out_cols, route_b[r][0].scale());
        worst_ab = std::max(worst_ab, worst_error(a, b));
        for (int s = 0; s < f.slots(); s++)
            for (int i = 0; i < f.d_small; i++)
                for (int j = 0; j < out_cols; j++)
                    worst_host = std::max(
                        worst_host,
                        std::abs(b[s][static_cast<size_t>(i) * out_cols + j] -
                                 want[s][static_cast<size_t>(i * f.m + r) *
                                             out_cols +
                                         j]));
    }
    std::cout << "PCMM route A vs route B: " << worst_ab
              << ", small-ring PCMM vs host: " << worst_host << std::endl;
    ASSERT_FALSE(std::isnan(worst_ab));
    EXPECT_LT(worst_ab, 1e-2);
    EXPECT_LT(worst_host, 1e-2);
}

// -----------------------------------------------------------------------
// 4. Algorithm 4 at the island, composed back up.
// -----------------------------------------------------------------------

// This is the flow the two-ring batch design exists for: a ciphertext-
// ciphertext product that CANNOT be run at the big ring, because Algorithm 4's
// operands are square at d and d is head_dim.
//
// The right operand is given period m down its rows, so every row class
// descends to the SAME small matrix and the interleave of the m island
// products is exactly the big-ring product A*B. Without that the composed
// result is still well defined but has no closed form to check against, and a
// test that can only compare against itself is not a test.
TEST(HEonGPU, CKKS_BatchRingSwitch_CCMMAtTheIslandComposesBackToTheBigProduct)
{
    BatchRingFixture f(13, 12, 64, {50, 40, 40}, 3);
    const int d = f.d_small; // 64: Algorithm 4's operands are square at d.
    ASSERT_EQ(f.d_big, 2 * d);

    std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(
        f.layout_small);
    heongpu::Galoiskey<S> galois_key(f.small, rot);
    f.small_keygen->generate_galois_key(galois_key, f.pair->small);
    heongpu::Relinkey<S> relin_key(f.small);
    f.small_keygen->generate_relin_key(relin_key, f.pair->small);

    // ccmm key-switches, so the operand scales must leave room for its noise.
    const double scale_a = std::pow(2.0, 35);
    const double scale_b = std::pow(2.0, 25);

    const std::vector<std::vector<cd>> A = f.random_batch(f.d_big, d, 246u);
    // B with period m in its rows: B[i] depends only on i / m.
    const std::vector<std::vector<cd>> Bbase = f.random_batch(d, d, 135u);
    std::vector<std::vector<cd>> B(
        f.slots(), std::vector<cd>(static_cast<size_t>(f.d_big) * d));
    for (int s = 0; s < f.slots(); s++)
        for (int i = 0; i < f.d_big; i++)
            for (int j = 0; j < d; j++)
                B[s][static_cast<size_t>(i) * d + j] =
                    Bbase[s][static_cast<size_t>(i / f.m) * d + j];

    std::vector<heongpu::Ciphertext<S>> a_cols =
        f.encrypt_big(A, f.d_big, d, scale_a);
    std::vector<heongpu::Ciphertext<S>> b_cols =
        f.encrypt_big(B, f.d_big, d, scale_b);

    std::vector<std::vector<heongpu::Ciphertext<S>>> a_parts =
        descend_columns(f, a_cols);
    std::vector<std::vector<heongpu::Ciphertext<S>>> b_parts =
        descend_columns(f, b_cols);

    std::vector<std::vector<heongpu::Ciphertext<S>>> s_parts(f.m);
    for (int r = 0; r < f.m; r++)
    {
        std::vector<heongpu::Ciphertext<S>*> pa, pb;
        for (auto& c : a_parts[r])
            pa.push_back(&c);
        for (auto& c : b_parts[r])
            pb.push_back(&c);
        f.op_small->ccmm(s_parts[r], pa, pb, galois_key, relin_key,
                         *f.small_ops);
        // ccmm MARKS a rescale; compose_up refuses an unsettled ciphertext,
        // and rightly so.
        for (auto& c : s_parts[r])
            f.small_ops->rescale_inplace(c);
    }

    std::vector<heongpu::Ciphertext<S>> composed =
        ascend_columns(f, s_parts);
    ASSERT_EQ(static_cast<int>(composed.size()), d);

    const double out_scale = composed[0].scale();
    std::vector<std::vector<cd>> got =
        f.decrypt_big(composed, f.d_big, d, out_scale);
    const std::vector<std::vector<cd>> want =
        host_matmul(A, Bbase, f.d_big, d, d);

    const double worst = worst_error(got, want);
    std::cout << "island CCMM composed to the big ring, worst error: " << worst
              << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    EXPECT_LT(worst, 0.05);
}

// -----------------------------------------------------------------------
// 5. The level ledger across the crossing.
// -----------------------------------------------------------------------

// The two-ring flow's whole cost question is how many levels an island visit
// can afford, so the ledger has to be asserted rather than assumed: a crossing
// must be free, and a rescale taken inside the island must be the same level
// the big ring would have spent.
TEST(HEonGPU, CKKS_BatchRingSwitch_CrossingIsLevelFreeAndTheIslandSpendsShared)
{
    BatchRingFixture f(13, 12, 32, {50, 40, 40}, 3);

    const int cols = 2;
    const double scale = std::pow(2.0, 30);
    const std::vector<std::vector<cd>> M = f.random_batch(f.d_big, cols, 55u);
    std::vector<heongpu::Ciphertext<S>> columns =
        f.encrypt_big(M, f.d_big, cols, scale);
    ASSERT_EQ(columns[0].depth(), 0);

    std::vector<std::vector<heongpu::Ciphertext<S>>> parts =
        descend_columns(f, columns);
    for (int r = 0; r < f.m; r++)
        EXPECT_EQ(parts[r][0].depth(), 0) << "the crossing must cost no level";

    // One island level, spent by a plaintext multiply and rescale.
    const double gain = 0.5;
    for (int r = 0; r < f.m; r++)
        for (auto& c : parts[r])
        {
            f.small_ops->multiply_plain_inplace(c, gain, scale);
            f.small_ops->rescale_inplace(c);
        }
    EXPECT_EQ(parts[0][0].depth(), 1);

    std::vector<heongpu::Ciphertext<S>> back = ascend_columns(f, parts);
    EXPECT_EQ(back[0].depth(), 1)
        << "an island rescale must land the big ring one level down";

    // And the composed value is the island's own arithmetic, not just a
    // level bookkeeping match.
    std::vector<std::vector<cd>> got =
        f.decrypt_big(back, f.d_big, cols, back[0].scale());
    double worst = 0.0;
    for (int b = 0; b < f.slots(); b++)
        for (size_t i = 0; i < M[b].size(); i++)
            worst = std::max(worst, std::abs(got[b][i] - gain * M[b][i]));
    std::cout << "island multiply composed up, worst error: " << worst
              << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    EXPECT_LT(worst, 1e-3);

    // The shared prefix is the island's whole level budget: a big ciphertext
    // is only allowed down while it still has an active prime to descend with.
    // (See the bootstrap test at the end of this file for why one level either
    // way decides whether batch 16 has a legal parameter set at all.)
    heongpu::Ciphertext<S> deeper = back[0];
    f.ops->multiply_plain_inplace(deeper, gain, back[0].scale());
    f.ops->rescale_inplace(deeper);
    EXPECT_EQ(deeper.depth(), 2);
    EXPECT_NO_THROW({
        std::vector<heongpu::Ciphertext<S>> ok =
            f.rs->switch_down(deeper, *f.ops);
        (void) ok;
    }) << "depth 2 of a 3-prime shared chain is still one active prime";
}

// -----------------------------------------------------------------------
// 6. Does a bootstrap carry a matrix encryption?
// -----------------------------------------------------------------------

// This is the question the whole batch-16 parameter set turns on, and it is
// not about ring switching at all -- it is about what has to happen at the top
// of the ascent.
//
// The island exits Algorithm 4 in MATRIX form. If the refresh has to be
// preceded by a bridge back to slots, that bridge is a level, and the island
// chain needs three Q primes plus a special = 140 bits against a cap of 109 at
// N = 4096: batch 16 has no legal 128-bit parameter set. If instead the
// refresh takes the matrix encryption as it stands, two Q primes and a special
// (41 + 33 + 33 = 107) close it with two bits to spare.
//
// Structurally it ought to work: regular_bootstrapping is ModRaise ->
// CoeffToSlot -> EvalMod -> SlotToCoeff, whose net effect on the PLAINTEXT
// POLYNOMIAL is the identity with the modulus restored, and it reads no
// encoding tag. What is not obvious is EvalMod's precision, because its bound
// is on the plaintext COEFFICIENTS and the batch encoding's coefficients are
// an inverse length-k DFT of the batch values rather than the values.
//
// Ring degree 4096 with k = 32 is the real batch-16 island shape; the chain
// here is the library's own bootstrapping demo chain, which is far outside any
// security level. That is deliberate -- what is being measured is whether the
// encoding survives, and the encoding does not know what N's cap is.
TEST(HEonGPU, CKKS_BatchRingSwitch_RegularBootstrapCarriesAMatrixEncryption)
{
    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    const size_t degree = 4096;
    context->set_poly_modulus_degree(degree);
    context->set_coeff_modulus_bit_sizes(
        {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
         50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
        {60, 60, 60});
    context->generate();

    // q0 / scale = 2^10 is the ratio the EvalMod fit is built around; any
    // other ratio returns noise without saying so.
    const double scale = std::pow(2.0, 50);

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context, 16);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::Relinkey<S> relin_key(context);
    keygen.generate_relin_key(relin_key, secret);

    heongpu::HEEncoder<S> encoder(context);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEArithmeticOperator<S> ops(context, encoder);

    heongpu::BootstrappingConfig boot_config(3, 3, 11, true);
    ops.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
    std::vector<int> key_index = ops.bootstrapping_key_indexs();
    heongpu::Galoiskey<S> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret);

    // The batch-16 island shape exactly: d = 128 = head_dim, k = 32,
    // batch = 16.
    heongpu::BatchMatrixLayout layout(static_cast<int>(degree), 128);
    ASSERT_EQ(layout.k, 32);
    ASSERT_EQ(layout.batch, 16);
    heongpu::HEBatchMatrixOperator<S> bop(context, layout);
    heongpu::BatchMatrixEncoder bm(layout.k);

    const int rows = layout.d;
    const int cols = 2;
    std::mt19937_64 rng(31337u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<std::vector<cd>> M(
        bm.slots(), std::vector<cd>(static_cast<size_t>(rows) * cols));
    for (auto& mat : M)
        for (auto& z : mat)
            z = cd(dist(rng), dist(rng));

    std::vector<int64_t> coeffs;
    bm.encode(M, rows, cols, scale, coeffs);
    std::vector<std::vector<int64_t>> columns;
    heongpu::build_matrix_encryption_coefficients(coeffs, layout, rows, cols,
                                                  columns);

    std::vector<heongpu::Ciphertext<S>> cts;
    for (int j = 0; j < cols; j++)
    {
        heongpu::Plaintext<S> p(context);
        bop.load_coefficients(p, columns[j], scale);
        heongpu::Ciphertext<S> c(context);
        encryptor.encrypt(c, p);
        cts.push_back(std::move(c));
    }

    // ModRaise starts from one prime, so the input has to be at the bottom.
    const int chain = context->get_ciphertext_modulus_count();
    for (auto& c : cts)
        for (int i = 0; i < chain - 1; i++)
            ops.mod_drop_inplace(c);
    ASSERT_EQ(cts[0].depth(), chain - 1);

    std::vector<heongpu::Ciphertext<S>> refreshed;
    for (auto& c : cts)
        refreshed.push_back(ops.regular_bootstrapping(c, galois_key,
                                                      relin_key));
    std::cout << "depth after bootstrap: " << refreshed[0].depth() << " of "
              << chain << ", scale 2^"
              << std::log2(refreshed[0].scale()) << std::endl;
    EXPECT_LT(refreshed[0].depth(), chain - 1)
        << "a refresh has to return levels";

    std::vector<std::vector<int64_t>> out(cols);
    for (int j = 0; j < cols; j++)
    {
        heongpu::Plaintext<S> p(context);
        decryptor.decrypt(p, refreshed[j]);
        bop.extract_coefficients(out[j], p);
    }
    std::vector<int64_t> got_coeffs;
    heongpu::split_matrix_encryption_coefficients(out, layout, rows, cols,
                                                  got_coeffs);
    std::vector<std::vector<cd>> got;
    bm.decode(got_coeffs, rows, cols, refreshed[0].scale(), got);

    const double worst = worst_error(got, M);
    std::cout << "bootstrapped batch matrix worst error: " << worst
              << "  (" << (worst > 0.0 ? -std::log2(worst) : 99.0)
              << " bits)" << std::endl;
    ASSERT_FALSE(std::isnan(worst));
    EXPECT_LT(worst, 0.05);
}
