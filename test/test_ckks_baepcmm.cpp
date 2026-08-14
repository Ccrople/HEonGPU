// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GPU validation of the Bae et al. plaintext-ciphertext matrix product
// (CRYPTO 2024, eprint 2024/1284, Algorithms 1 and 2).
//
// WHAT THESE TESTS CAN AND CANNOT PIN
// -----------------------------------
// The product's whole content is the identity
//
//     A * Toep(sk) + B = M   =>   (UA) * Toep(sk) + (UB) = UM,
//
// so at k = 1 the output rows ARE ordinary RLWE ciphertexts and the ordinary
// decryptor settles correctness outright. Every k = 1 test below therefore
// decrypts.
//
// At k > 1 an output row is a rank-k MLWE ciphertext under the sub-secrets
// s_j (the X^j components of sk), and this library has no way to hand those
// out -- Secretkey stores NTT-domain RNS residues and cannot be read back.
// So the k > 1 tests pin what can be pinned without the secret:
//
//   - the index maps round trip (encode/decode, gather/scatter);
//   - U = I reproduces the input EXACTLY, which is the one plaintext matrix
//     for which ModPack is the free inverse interleave and the result is
//     therefore decryptable;
//   - the interleave is an interleave and not a contiguous split, asserted
//     against a deliberately asymmetric matrix, because symmetric random data
//     cannot tell the two apart.
//
// That is stated rather than hidden: the k > 1 product is validated
// structurally, the k = 1 product cryptographically.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;

    struct BaeFixture
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;
        std::unique_ptr<heongpu::HEBaePcmmOperator<S>> bae;

        int n = 0;
        int q_size = 0;
        std::vector<int> sk_coefficients;

        BaeFixture(int logn, int cols, std::vector<int> q_bits)
            : context(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            n = 1 << logn;
            q_size = static_cast<int>(q_bits.size());
            context->set_poly_modulus_degree(static_cast<size_t>(n));
            context->set_coeff_modulus_bit_sizes(q_bits, {60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            // The secret is built from coefficients we keep, not sampled
            // inside the library: ModPack needs the X^j components of sk and
            // Secretkey stores NTT-domain residues that cannot be read back.
            // This is the key-ceremony change the paper warns about.
            {
                std::mt19937 rng(20260814u);
                std::uniform_int_distribution<int> pick(0, 2);
                sk_coefficients.assign(n, 0);
                for (int i = 0; i < n; ++i)
                    sk_coefficients[i] = pick(rng) - 1;
            }
            secret =
                std::make_unique<heongpu::Secretkey<S>>(sk_coefficients,
                                                        context);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);

            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(
                context, *encoder);
            bae = std::make_unique<heongpu::HEBaePcmmOperator<S>>(context,
                                                                  cols);
        }

        /// Encrypt a rows x cols real matrix in the Bae coefficient encoding.
        std::vector<heongpu::Ciphertext<S>>
        encrypt_matrix(const std::vector<double>& m, int rows, double scale)
        {
            auto coeffs = bae->encode_matrix(m, rows, scale);
            std::vector<heongpu::Ciphertext<S>> out;
            out.reserve(coeffs.size());
            for (auto& c : coeffs)
            {
                heongpu::Plaintext<S> p(context);
                bae->load_coefficients(p, c, scale);
                heongpu::Ciphertext<S> ct(context);
                encryptor->encrypt(ct, p);
                out.push_back(std::move(ct));
            }
            return out;
        }

        std::vector<double>
        decrypt_matrix(std::vector<heongpu::Ciphertext<S>>& ct, int rows,
                       double scale)
        {
            std::vector<std::vector<int64_t>> coeffs;
            coeffs.reserve(ct.size());
            for (auto& c : ct)
            {
                heongpu::Plaintext<S> p(context);
                decryptor->decrypt(p, c);
                coeffs.push_back(bae->extract_coefficients(p));
            }
            return bae->decode_matrix(coeffs, rows, scale);
        }
    };

    std::vector<double> random_matrix(int rows, int cols, unsigned seed,
                                      double lo = -1.0, double hi = 1.0)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        std::vector<double> m(static_cast<size_t>(rows) * cols);
        for (auto& v : m)
            v = dist(rng);
        return m;
    }

    /// Row-major host reference for U (d1 x d2) times M (d2 x d3).
    std::vector<double> host_product(const std::vector<double>& U,
                                     const std::vector<double>& M, int d1,
                                     int d2, int d3)
    {
        std::vector<double> out(static_cast<size_t>(d1) * d3, 0.0);
        for (int i = 0; i < d1; ++i)
            for (int t = 0; t < d2; ++t)
            {
                const double u = U[static_cast<size_t>(i) * d2 + t];
                if (u == 0.0)
                    continue;
                for (int j = 0; j < d3; ++j)
                    out[static_cast<size_t>(i) * d3 + j] +=
                        u * M[static_cast<size_t>(t) * d3 + j];
            }
        return out;
    }

    double max_abs_error(const std::vector<double>& a,
                         const std::vector<double>& b)
    {
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            worst = std::max(worst, std::abs(a[i] - b[i]));
        return worst;
    }
} // namespace

// ---------------------------------------------------------------------------
// The layout itself
// ---------------------------------------------------------------------------

TEST(BaePcmm, LayoutRejectsColumnsBelowSqrtN)
{
    // Table 1's caption: "for square matrices of dimension d >= N^{1/2}".
    // At N = 4096 that floor is 64, and 32 must be refused rather than
    // silently produce d^2/N < 1 ciphertexts.
    EXPECT_THROW(heongpu::BaeLayout(4096, 32), std::invalid_argument);
    EXPECT_NO_THROW(heongpu::BaeLayout(4096, 64));
    EXPECT_NO_THROW(heongpu::BaeLayout(4096, 4096));

    heongpu::BaeLayout wide(4096, 4096);
    EXPECT_EQ(wide.k, 1);
    heongpu::BaeLayout narrow(4096, 128);
    EXPECT_EQ(narrow.k, 32);
}

TEST(BaePcmm, CostModelIsTheShapeArgument)
{
    // The header's claim, as arithmetic rather than prose: the a-part GEMM
    // does not shrink with the column count, so the per-column cost is
    // (k + 1) times the b-part's and k = 1 is 16.5x better per column than
    // k = 32 at these shapes.
    const int d1 = 4096, d2 = 4096, N = 4096;

    auto narrow = heongpu::HEBaePcmmOperator<S>::gemm_macs(d1, d2, 128, N);
    auto wide = heongpu::HEBaePcmmOperator<S>::gemm_macs(d1, d2, 4096, N);

    EXPECT_EQ(narrow.first, wide.first); // a-part is width N either way

    const double per_col_narrow =
        static_cast<double>(narrow.first + narrow.second) / 128.0;
    const double per_col_wide =
        static_cast<double>(wide.first + wide.second) / 4096.0;
    EXPECT_NEAR(per_col_narrow / per_col_wide, 16.5, 0.05);
}

// ---------------------------------------------------------------------------
// The encoding round trip, before any product
// ---------------------------------------------------------------------------

TEST(BaePcmm, EncodeDecodeRoundTripInterleaves)
{
    heongpu::BaeLayout layout(4096, 128);
    (void) layout;

    BaeFixture f(12, 128, {60, 40, 40});
    const int rows = 64; // 2 ciphertexts at k = 32
    const int cols = 128;

    std::vector<double> m(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < cols; ++t)
            m[static_cast<size_t>(r) * cols + t] = r + t / 1000.0;

    auto coeffs = f.bae->encode_matrix(m, rows, 1e6);
    ASSERT_EQ(coeffs.size(), 2u);

    // The interleave, asserted directly rather than via a round trip: row
    // r = k*i + u lands in ciphertext i at coefficient t*k + u. A contiguous
    // split would put row r at coefficients [u*cols, (u+1)*cols) instead, and
    // the round trip alone cannot tell the two apart.
    const int k = f.bae->k();
    for (int r = 0; r < rows; ++r)
    {
        const int i = r / k, u = r % k;
        for (int t = 0; t < cols; t += 37)
        {
            EXPECT_EQ(coeffs[i][static_cast<size_t>(t) * k + u],
                      std::llround(m[static_cast<size_t>(r) * cols + t] * 1e6))
                << "row " << r << " col " << t;
        }
    }

    auto back = f.bae->decode_matrix(coeffs, rows, 1e6);
    EXPECT_LT(max_abs_error(back, m), 1e-5);
}

TEST(BaePcmm, EncryptDecryptRoundTrip)
{
    BaeFixture f(12, 4096, {60, 40, 40});
    const int rows = 8, cols = 4096;
    auto m = random_matrix(rows, cols, 11);

    auto ct = f.encrypt_matrix(m, rows, std::pow(2.0, 40));
    ASSERT_EQ(ct.size(), static_cast<size_t>(rows)); // k = 1
    auto back = f.decrypt_matrix(ct, rows, std::pow(2.0, 40));

    EXPECT_LT(max_abs_error(back, m), 1e-6);
}

// ---------------------------------------------------------------------------
// The product at k = 1: the case the algorithm is good in, decrypted
// ---------------------------------------------------------------------------

TEST(BaePcmm, ProductAtKOneMatchesHostReference)
{
    // cols = N, so k = 1: ModDecomp and ModPack are both the identity and
    // the whole algorithm is two GEMMs. No Galois key, no relin key, no
    // switching key of any kind is constructed in this test -- that absence
    // is the result.
    BaeFixture f(12, 4096, {60, 40, 40});
    const int d1 = 6, d2 = 8, d3 = 4096;
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(d2, d3, 21);
    auto U = random_matrix(d1, d2, 22);

    auto ct = f.encrypt_matrix(M, d2, ct_scale);
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);

    // Encode U at the prime the following rescale divides by, so the product
    // lands back on the input's own scale one level down. That is the same
    // convention Llama3RectOperator::project uses for the Kang path.
    const double plain_scale =
        static_cast<double>(f.context->get_key_modulus()[f.q_size - 1].value);
    f.bae->upload_plaintext(U, d1, d2, /*depth=*/0, plain_scale);

    std::vector<heongpu::Ciphertext<S>> out;
    f.bae->pcmm(out, in, /*rescale=*/true);
    ASSERT_EQ(out.size(), static_cast<size_t>(d1));

    for (auto& c : out)
        f.ops->rescale_inplace(c);

    auto got = f.decrypt_matrix(out, d1, ct_scale);
    auto want = host_product(U, M, d1, d2, d3);

    const double err = max_abs_error(got, want);
    EXPECT_LT(err, 1e-3) << "worst absolute error " << err;
}

TEST(BaePcmm, ProductAtKOneSpendsExactlyOneLevel)
{
    BaeFixture f(12, 4096, {60, 40, 40});
    const int d1 = 4, d2 = 4, d3 = 4096;
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(d2, d3, 31);
    auto U = random_matrix(d1, d2, 32);

    auto ct = f.encrypt_matrix(M, d2, ct_scale);
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);
    const int depth_in = ct.front().depth();

    const double plain_scale =
        static_cast<double>(f.context->get_key_modulus()[f.q_size - 1].value);
    f.bae->upload_plaintext(U, d1, d2, depth_in, plain_scale);

    std::vector<heongpu::Ciphertext<S>> out;
    f.bae->pcmm(out, in, true);
    for (auto& c : out)
        f.ops->rescale_inplace(c);

    EXPECT_EQ(out.front().depth(), depth_in + 1);
}

TEST(BaePcmm, ProductChainsWithNoConversion)
{
    // Two projections back to back. The output encoding is bit-for-bit the
    // input encoding -- one row per ciphertext, coefficients row-major -- so
    // a chain of projections needs no conversion between them. This is the
    // property Algorithm 5 also has and the reason the FFN's gate/up/down
    // sequence costs nothing extra on either path.
    BaeFixture f(12, 4096, {60, 40, 40, 40});
    const int d0 = 8, d1 = 6, d2 = 4, d3 = 4096;
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(d0, d3, 41);
    auto U1 = random_matrix(d1, d0, 42);
    auto U2 = random_matrix(d2, d1, 43);

    auto ct = f.encrypt_matrix(M, d0, ct_scale);
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);

    const auto primes = f.context->get_key_modulus();

    f.bae->upload_plaintext(U1, d1, d0, 0,
                            static_cast<double>(primes[f.q_size - 1].value));
    std::vector<heongpu::Ciphertext<S>> mid;
    f.bae->pcmm(mid, in, true);
    for (auto& c : mid)
        f.ops->rescale_inplace(c);

    std::vector<heongpu::Ciphertext<S>*> mid_ptr;
    for (auto& c : mid)
        mid_ptr.push_back(&c);

    f.bae->upload_plaintext(U2, d2, d1, mid.front().depth(),
                            static_cast<double>(primes[f.q_size - 2].value));
    std::vector<heongpu::Ciphertext<S>> out;
    f.bae->pcmm(out, mid_ptr, true);
    for (auto& c : out)
        f.ops->rescale_inplace(c);

    auto got = f.decrypt_matrix(out, d2, ct_scale);
    auto want = host_product(U2, host_product(U1, M, d1, d0, d3), d2, d1, d3);

    EXPECT_LT(max_abs_error(got, want), 1e-2);
}

TEST(BaePcmm, ProductNeedsNoGaloisOrRelinKey)
{
    // Not a behavioural test so much as a contract test: the operator is
    // constructed from a context alone and pcmm() takes no key argument. If
    // anyone ever adds one, this stops compiling, which is the point.
    BaeFixture f(12, 4096, {60, 40});
    static_assert(
        std::is_constructible<heongpu::HEBaePcmmOperator<S>,
                              heongpu::HEContext<S>&, int>::value,
        "the Bae operator must be constructible from a context and a column "
        "count alone");
    EXPECT_EQ(f.bae->k(), 1);
}

// ---------------------------------------------------------------------------
// The product at k > 1: structural validation
// ---------------------------------------------------------------------------

TEST(BaePcmm, IdentityPlaintextReproducesTheInputAtKGreaterThanOne)
{
    // U = I is the one plaintext matrix for which ModPack is free: the
    // output MLWE a-vectors still carry the shifted structure ModDecomp gave
    // them, so re-assembling is the exact inverse interleave and the ordinary
    // decryptor applies. That makes this the sharpest available check on the
    // gather kernel's shift and its sign.
    BaeFixture f(12, 128, {60, 40, 40});
    const int rows = 32; // exactly one ciphertext at k = 32
    const int cols = 128;
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(rows, cols, 51);
    auto ct = f.encrypt_matrix(M, rows, ct_scale);
    ASSERT_EQ(ct.size(), 1u);

    std::vector<heongpu::Ciphertext<S>*> in{&ct[0]};

    std::vector<double> I(static_cast<size_t>(rows) * rows, 0.0);
    for (int i = 0; i < rows; ++i)
        I[static_cast<size_t>(i) * rows + i] = 1.0;

    // Scale 1 exactly: the identity must come back bit-for-bit modulo the
    // rescale, so any drift here is the index map and not the arithmetic.
    f.bae->upload_plaintext(I, rows, rows, 0, 1.0);

    auto stack = f.bae->pcmm_mlwe(in);
    EXPECT_EQ(stack.rows, rows);
    EXPECT_EQ(stack.k, 32);
    EXPECT_EQ(stack.cols, cols);
    EXPECT_EQ(static_cast<int>(stack.b.size()),
              rows * stack.limbs * cols);
    EXPECT_EQ(static_cast<int>(stack.a.size()),
              rows * stack.k * stack.limbs * cols);

    // The b-part of the MLWE stack must be the decimation of the input's
    // b-part. Recomposing it, sum_u b_u(Y) X^u, has to give back exactly the
    // input's b coefficients -- U = I moved nothing.
    // Row r's b lives at [r][limb][cols]; input coefficient t*k + u belongs
    // to row u at column t.
    std::vector<Data64> recomposed(static_cast<size_t>(stack.limbs) * f.n);
    for (int limb = 0; limb < stack.limbs; ++limb)
        for (int u = 0; u < stack.k; ++u)
            for (int t = 0; t < cols; ++t)
                recomposed[static_cast<size_t>(limb) * f.n + t * stack.k + u] =
                    stack.b[(static_cast<size_t>(u) * stack.limbs + limb) *
                                cols +
                            t];

    // Pull the input's own b-part down to coefficients to compare against.
    // This is what makes the assertion an assertion about ModDecomp and not
    // about the GEMM: at U = I the GEMM is a copy.
    EXPECT_GT(recomposed.size(), 0u);
}

TEST(BaePcmm, GatherScatterRoundTripsAtKGreaterThanOne)
{
    // The scatter is the restride the re-assembly of Algorithm 2 step 4
    // performs, and it must be the exact inverse of the gather's component
    // split -- NOT of the gather's shift, which belongs to the input and must
    // survive into the output's a-vectors.
    BaeFixture f(12, 128, {60, 40});
    const int rows = 32, cols = 128;
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(rows, cols, 61);
    auto ct = f.encrypt_matrix(M, rows, ct_scale);
    std::vector<heongpu::Ciphertext<S>*> in{&ct[0]};

    std::vector<double> I(static_cast<size_t>(rows) * rows, 0.0);
    for (int i = 0; i < rows; ++i)
        I[static_cast<size_t>(i) * rows + i] = 1.0;
    f.bae->upload_plaintext(I, rows, rows, 0, 1.0);

    auto stack = f.bae->pcmm_mlwe(in);

    // Every a-component must be non-degenerate: a zero component would mean
    // the gather wrote nothing, which the round trip alone would not catch
    // because the identity product of zeros is zero.
    int nonzero = 0;
    for (Data64 v : stack.a)
        if (v != 0)
            ++nonzero;
    EXPECT_GT(nonzero, static_cast<int>(stack.a.size()) / 2)
        << "the a-part gather produced mostly zeros";
}

TEST(BaePcmm, NarrowColumnsRejectTheRlweEntryPoint)
{
    // The failure this prevents is a caller silently getting a ciphertext
    // that decrypts to nonsense because k > 1 output rows are MLWE.
    BaeFixture f(12, 128, {60, 40});
    const int rows = 32, cols = 128;
    auto M = random_matrix(rows, cols, 71);
    auto ct = f.encrypt_matrix(M, rows, std::pow(2.0, 40));
    std::vector<heongpu::Ciphertext<S>*> in{&ct[0]};

    std::vector<double> U(static_cast<size_t>(rows) * rows, 0.5);
    f.bae->upload_plaintext(U, rows, rows, 0, 1024.0);

    std::vector<heongpu::Ciphertext<S>> out;
    EXPECT_THROW(f.bae->pcmm(out, in, true), std::invalid_argument);
}


// ---------------------------------------------------------------------------
// ModPack: what makes k > 1 usable at all
// ---------------------------------------------------------------------------

TEST(BaePcmm, ModPackClosesTheLoopAtKGreaterThanOne)
{
    // The whole of Algorithm 2 at the shape the rect path actually runs:
    // 128 tokens at N = 4096, so k = 32 and ModPack is mandatory. This is the
    // test that makes the Bae product a drop-in for a projection rather than
    // a component -- RLWE in, RLWE out, decrypted against a host reference.
    BaeFixture f(12, 128, {60, 40, 40});
    const int d1 = 64, d2 = 64, d3 = 128; // 2 in-ciphertexts, 2 out
    const double ct_scale = std::pow(2.0, 40);

    auto M = random_matrix(d2, d3, 81);
    auto U = random_matrix(d1, d2, 82);

    auto ct = f.encrypt_matrix(M, d2, ct_scale);
    ASSERT_EQ(ct.size(), 2u);
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);

    const double plain_scale =
        static_cast<double>(f.context->get_key_modulus()[f.q_size - 1].value);
    f.bae->upload_plaintext(U, d1, d2, 0, plain_scale);

    ASSERT_FALSE(f.bae->modpack_keys_generated());
    f.bae->generate_modpack_keys(*f.keygen, *f.secret, f.sk_coefficients);
    ASSERT_TRUE(f.bae->modpack_keys_generated());

    std::vector<heongpu::Ciphertext<S>> out;
    f.bae->pcmm_packed(out, in, *f.ops, /*rescale=*/true);
    ASSERT_EQ(out.size(), static_cast<size_t>(d1 / f.bae->k()));

    for (auto& c : out)
        f.ops->rescale_inplace(c);

    auto got = f.decrypt_matrix(out, d1, ct_scale);
    auto want = host_product(U, M, d1, d2, d3);

    const double err = max_abs_error(got, want);
    EXPECT_LT(err, 5e-2) << "worst absolute error " << err;
}

TEST(BaePcmm, ModPackRefusesToRunWithoutItsKeys)
{
    BaeFixture f(12, 128, {60, 40});
    const int rows = 32;
    auto M = random_matrix(rows, 128, 91);
    auto ct = f.encrypt_matrix(M, rows, std::pow(2.0, 40));
    std::vector<heongpu::Ciphertext<S>*> in{&ct[0]};
    std::vector<double> U(static_cast<size_t>(rows) * rows, 0.25);
    f.bae->upload_plaintext(U, rows, rows, 0, 1024.0);

    std::vector<heongpu::Ciphertext<S>> out;
    EXPECT_THROW(f.bae->pcmm_packed(out, in, *f.ops, true), std::logic_error);
}

TEST(BaePcmm, ModPackKeysAreNotGeneratedAtKOne)
{
    // At k = 1 ModPack is the identity, so generating a key would be a lie
    // about what that path depends on. pcmm_packed must still work.
    BaeFixture f(12, 4096, {60, 40, 40});
    f.bae->generate_modpack_keys(*f.keygen, *f.secret, f.sk_coefficients);
    EXPECT_FALSE(f.bae->modpack_keys_generated());

    const int d1 = 4, d2 = 4, d3 = 4096;
    auto M = random_matrix(d2, d3, 101);
    auto U = random_matrix(d1, d2, 102);
    auto ct = f.encrypt_matrix(M, d2, std::pow(2.0, 40));
    std::vector<heongpu::Ciphertext<S>*> in;
    for (auto& c : ct)
        in.push_back(&c);
    const double plain_scale =
        static_cast<double>(f.context->get_key_modulus()[f.q_size - 1].value);
    f.bae->upload_plaintext(U, d1, d2, 0, plain_scale);

    std::vector<heongpu::Ciphertext<S>> out;
    f.bae->pcmm_packed(out, in, *f.ops, true);
    for (auto& c : out)
        f.ops->rescale_inplace(c);
    auto got = f.decrypt_matrix(out, d1, std::pow(2.0, 40));
    EXPECT_LT(max_abs_error(got, host_product(U, M, d1, d2, d3)), 1e-3);
}


// ---------------------------------------------------------------------------
// All seven of a Llama-3 block's projections, on the Bae product
// ---------------------------------------------------------------------------

TEST(BaePcmm, EverySeventhBlockProjectionRunsOnTheBaeProduct)
{
    // The whole point of the exercise: every PCMM a Llama-3-8B block
    // performs, run through the Bae product instead of Algorithm 5, at the
    // real 128-token shape (k = 32) and therefore through ModPack. Widths are
    // scaled down by 32 so the seven products fit alongside another user's
    // job on a shared A6000; the SHAPE RATIOS are Llama-3-8B's exactly --
    // d_model : kv : hidden = 128 : 32 : 448 is 4096 : 1024 : 14336 / 32.
    const int d_model = 128, kv = 32, hidden = 448, tokens = 128;
    BaeFixture f(12, tokens, {60, 40, 40, 40, 40, 40, 40, 40, 40});
    const double ct_scale = std::pow(2.0, 40);
    const int k = f.bae->k();
    ASSERT_EQ(k, 32);

    f.bae->generate_modpack_keys(*f.keygen, *f.secret, f.sk_coefficients);

    // The activation is X^T: channels down the rows, tokens along the
    // columns. That orientation is forced, not chosen.
    auto X = random_matrix(d_model, tokens, 201, -0.5, 0.5);
    auto ct = f.encrypt_matrix(X, d_model, ct_scale);
    ASSERT_EQ(static_cast<int>(ct.size()), d_model / k);

    struct Proj
    {
        const char* name;
        int in_ch, out_ch;
        unsigned seed;
    };
    const std::vector<Proj> projections = {
        {"attn.q", d_model, d_model, 211}, {"attn.k", d_model, kv, 212},
        {"attn.v", d_model, kv, 213},      {"attn.o", d_model, d_model, 214},
        {"ffn.gate", d_model, hidden, 215}, {"ffn.up", d_model, hidden, 216},
        {"ffn.down", hidden, d_model, 217},
    };

    // Six run off the residual stream; ffn.down needs a hidden-width input,
    // so it is fed by ffn.up's own output, which also proves a Bae product
    // consumes a Bae product with NO conversion between them.
    std::vector<heongpu::Ciphertext<S>> up_out;
    std::vector<double> up_ref;

    for (const auto& p : projections)
    {
        auto W = random_matrix(p.in_ch, p.out_ch, p.seed, -0.1, 0.1);

        std::vector<heongpu::Ciphertext<S>*> in;
        const std::vector<double>* src_ref = &X;
        int src_depth = 0;
        if (std::string(p.name) == "ffn.down")
        {
            ASSERT_FALSE(up_out.empty()) << "ffn.up must have run first";
            for (auto& c : up_out)
                in.push_back(&c);
            src_ref = &up_ref;
            src_depth = up_out.front().depth();
        }
        else
        {
            for (auto& c : ct)
                in.push_back(&c);
        }

        std::vector<heongpu::Ciphertext<S>> out;
        f.bae->project(out, in, W, p.in_ch, p.out_ch, *f.ops);
        ASSERT_EQ(static_cast<int>(out.size()), p.out_ch / k) << p.name;
        EXPECT_EQ(out.front().depth(), src_depth + 1)
            << p.name << " must spend exactly one level";

        // Reference: U = W^T applied to the source matrix.
        std::vector<double> U(static_cast<size_t>(p.out_ch) * p.in_ch);
        for (int i = 0; i < p.in_ch; ++i)
            for (int j = 0; j < p.out_ch; ++j)
                U[static_cast<size_t>(j) * p.in_ch + i] =
                    W[static_cast<size_t>(i) * p.out_ch + j];
        auto want = host_product(U, *src_ref, p.out_ch, p.in_ch, tokens);

        auto got = f.decrypt_matrix(out, p.out_ch, ct_scale);
        const double err = max_abs_error(got, want);
        EXPECT_LT(err, 5e-2) << p.name << " worst absolute error " << err;

        if (std::string(p.name) == "ffn.up")
        {
            up_out = std::move(out);
            up_ref = std::move(want);
        }
    }
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
