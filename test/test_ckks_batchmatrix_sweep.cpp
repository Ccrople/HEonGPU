// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Parameter sweep for the batch matrix primitives.
//
// The focused tests in test_ckks_batchmatrix_gpu.cpp all sit at one point of
// the parameter space -- N = 4096, d = 8, one or two active limbs -- which is
// enough to say the algorithms are right but not enough to say the kernels are
// shape agnostic. This walks the parameters that actually change which code
// runs: the module rank d, which sets the TWEAK recursion depth, the subring
// transform length, every GEMM shape and the number of automorphisms CMT needs;
// the ring degree N; the number of active limbs; the output column count; and
// the level the operands sit at.
//
// Every configuration is run inside a try/catch and recorded, so one bad shape
// reports itself instead of taking the rest of the sweep down with it.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
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

    std::vector<std::vector<cd>> random_batch(int slots, int rows, int cols,
                                              std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::vector<cd>> m(
            slots, std::vector<cd>(static_cast<size_t>(rows) * cols));
        for (auto& mat : m)
            for (auto& z : mat)
                z = cd(dist(rng), dist(rng));
        return m;
    }

    /// One CKKS context and its keys, shared by every d tested against it.
    ///
    /// The Galois key carries the union of the CMT rotation indices over all
    /// the d values in the sweep, so the expensive key generation happens once
    /// per context rather than once per shape.
    struct Env
    {
        heongpu::HEContext<S> context;
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> slots;
        std::unique_ptr<heongpu::HEArithmeticOperator<S>> ops;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        int n = 0;

        Env(size_t degree, const std::vector<int>& logq,
            const std::vector<int>& logp, const std::vector<int>& ds)
            : context(heongpu::GenHEContext<S>(heongpu::sec_level_type::none))
        {
            context->set_poly_modulus_degree(degree);
            context->set_coeff_modulus_bit_sizes(logq, logp);
            context->generate();
            n = static_cast<int>(degree);

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            slots = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<heongpu::HEArithmeticOperator<S>>(context,
                                                                    *slots);

            if (ds.empty())
                return;

            std::set<int> indices;
            for (int d : ds)
            {
                heongpu::BatchMatrixLayout layout(n, d);
                for (int r : heongpu::get_batch_cmt_rotation_indices(layout))
                    indices.insert(r);
            }
            std::vector<int> rot(indices.begin(), indices.end());
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, rot);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }
    };

    /// Encrypt the matrix encryption (Definition 2) of a batch of matrices.
    void encrypt_matrix(Env& env, heongpu::HEBatchMatrixOperator<S>& op,
                        const heongpu::BatchMatrixEncoder& enc,
                        const heongpu::BatchMatrixLayout& layout,
                        const std::vector<std::vector<cd>>& mat, int rows,
                        int cols, double scale, int depth,
                        std::vector<heongpu::Ciphertext<S>>& out)
    {
        std::vector<int64_t> coeffs;
        enc.encode(mat, rows, cols, scale, coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout, rows,
                                                      cols, columns);
        out.clear();
        out.reserve(cols);
        for (int j = 0; j < cols; ++j)
        {
            heongpu::Plaintext<S> pt(env.context);
            op.load_coefficients(pt, columns[j], scale);
            heongpu::Ciphertext<S> c(env.context);
            env.encryptor->encrypt(c, pt);
            for (int t = 0; t < depth; ++t)
                env.ops->mod_drop_inplace(c);
            out.push_back(std::move(c));
        }
    }

    /// Inverse of encrypt_matrix: decrypt and decode a matrix encryption.
    std::vector<std::vector<cd>>
    decrypt_matrix(Env& env, heongpu::HEBatchMatrixOperator<S>& op,
                   const heongpu::BatchMatrixEncoder& enc,
                   const heongpu::BatchMatrixLayout& layout,
                   std::vector<heongpu::Ciphertext<S>>& cts, int rows,
                   int cols, double scale)
    {
        std::vector<std::vector<int64_t>> out_columns(cols);
        for (int j = 0; j < cols; ++j)
        {
            heongpu::Plaintext<S> pt(env.context);
            env.decryptor->decrypt(pt, cts[j]);
            op.extract_coefficients(out_columns[j], pt);
        }
        std::vector<int64_t> coeffs;
        heongpu::split_matrix_encryption_coefficients(out_columns, layout,
                                                      rows, cols, coeffs);
        std::vector<std::vector<cd>> got;
        enc.decode(coeffs, rows, cols, scale, got);
        return got;
    }

    double worst_error(const std::vector<std::vector<cd>>& got,
                       const std::vector<std::vector<cd>>& ref)
    {
        double worst = 0.0;
        for (size_t s = 0; s < ref.size(); ++s)
            for (size_t e = 0; e < ref[s].size(); ++e)
                worst = std::max(worst, std::abs(got[s][e] - ref[s][e]));
        return worst;
    }

    /// One row of the printed summary.
    struct Result
    {
        std::string config;
        bool ok = false;
        double error = 0.0;
        std::string failure;
    };

    class Sweep
    {
      public:
        void record(const std::string& config, double error, double tolerance)
        {
            Result r;
            r.config = config;
            r.error = error;
            r.ok = error < tolerance;
            if (!r.ok)
            {
                std::ostringstream os;
                os << "error " << error << " exceeds tolerance " << tolerance;
                r.failure = os.str();
            }
            rows_.push_back(std::move(r));
        }

        void record_failure(const std::string& config, const std::string& why)
        {
            Result r;
            r.config = config;
            r.ok = false;
            r.error = std::numeric_limits<double>::quiet_NaN();
            r.failure = why;
            rows_.push_back(std::move(r));
        }

        /// Run one configuration, turning any throw into a recorded failure.
        template <typename F>
        void run(const std::string& config, double tolerance, F&& body)
        {
            try
            {
                record(config, body(), tolerance);
            }
            catch (const std::exception& e)
            {
                record_failure(config, std::string("threw: ") + e.what());
            }
        }

        void report(const char* title) const
        {
            std::cout << "\n=== " << title << " ===" << std::endl;
            size_t failed = 0;
            for (const auto& r : rows_)
            {
                std::cout << (r.ok ? "  ok   " : "  FAIL ") << std::left
                          << std::setw(46) << r.config << std::right;
                if (r.ok)
                    std::cout << " err=" << std::scientific
                              << std::setprecision(3) << r.error;
                else
                    std::cout << " " << r.failure;
                std::cout << std::defaultfloat << std::endl;
                failed += !r.ok;
            }
            std::cout << "  " << (rows_.size() - failed) << "/" << rows_.size()
                      << " configurations passed" << std::endl;
        }

        void expect_all_passed() const
        {
            for (const auto& r : rows_)
                EXPECT_TRUE(r.ok) << r.config << ": " << r.failure;
        }

      private:
        std::vector<Result> rows_;
    };

    std::string label(size_t degree, int d, int limbs, const char* extra = "")
    {
        std::ostringstream os;
        os << "N=" << degree << " d=" << d << " k=" << (degree / d)
           << " limbs=" << limbs << extra;
        return os.str();
    }
} // namespace

// PCMM (Algorithm 1) across ring degrees, module ranks, limb counts and output
// column counts. PCMM needs no evaluation keys, so this is the cheapest place
// to push d hard -- d is what sets the subring transform length and the GEMM
// shapes.
TEST(HEonGPU, CKKS_BatchMatrix_SweepPCMM)
{
    Sweep sweep;

    struct Case
    {
        size_t degree;
        std::vector<int> logq;
        std::vector<int> logp;
        std::vector<int> ds;
    };
    // The product lands at scale_m * scale_u and accumulates d * k = N terms,
    // so 2^20 each keeps it inside even the single 50-bit prime.
    //
    // Two of these deliberately sit on a launch boundary. d = 256 with
    // cols = 256 puts 65536 entries in the GEMM grid, one past what a grid
    // dimension other than x can address; N = 16384 with d = 2 gives k = 8192,
    // whose length-k transform wants 64 KB of shared memory against a 48 KB
    // default. Both are shapes a caller could reasonably ask for.
    const std::vector<Case> cases = {
        {4096, {50}, {50}, {2, 4, 8, 16, 32, 64, 128}},
        {4096, {50, 40}, {50}, {2, 8, 64}},
        {8192, {50, 40, 40}, {50}, {2, 4, 8, 16, 32, 64, 128, 256}},
        {16384, {50, 40}, {50}, {2, 8, 64, 256}},
    };

    const double scale_m = std::pow(2.0, 20);
    const double scale_u = std::pow(2.0, 20);

    for (const auto& c : cases)
    {
        Env env(c.degree, c.logq, c.logp, {});
        const int limbs = static_cast<int>(c.logq.size());
        // The noise is the encryption noise carried through an N-term sum, so
        // the tolerance tracks sqrt(N) rather than d.
        const double tol = 0.05 * std::sqrt(c.degree / 4096.0);

        for (int d : c.ds)
        {
            for (int cols : {1, 3, d})
            {
                std::ostringstream os;
                os << "cols=" << cols;
                const std::string name =
                    label(c.degree, d, limbs, (" " + os.str()).c_str());

                sweep.run(name, tol, [&]() {
                    heongpu::BatchMatrixLayout layout(
                        static_cast<int>(c.degree), d);
                    heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
                    heongpu::BatchMatrixEncoder enc(layout.k);
                    const int nslots = enc.slots();

                    std::mt19937_64 rng(0xB47C4u + d * 131u + cols);
                    auto M = random_batch(nslots, d, d, rng);
                    auto U = random_batch(nslots, d, cols, rng);

                    std::vector<heongpu::Ciphertext<S>> cts;
                    encrypt_matrix(env, op, enc, layout, M, d, d, scale_m, 0,
                                   cts);

                    std::vector<int64_t> u_coeffs;
                    enc.encode(U, d, cols, scale_u, u_coeffs);
                    op.encode_plaintext_matrix(u_coeffs, d, cols, 0, scale_u);

                    std::vector<heongpu::Ciphertext<S>*> in;
                    for (auto& ct : cts)
                        in.push_back(&ct);
                    std::vector<heongpu::Ciphertext<S>> out;
                    op.pcmm(out, in, /*rescale=*/false);
                    if (static_cast<int>(out.size()) != cols)
                        throw std::runtime_error("wrong output column count");

                    auto got = decrypt_matrix(env, op, enc, layout, out, d,
                                              cols, scale_m * scale_u);

                    std::vector<std::vector<cd>> ref(nslots);
                    for (int s = 0; s < nslots; ++s)
                        ref[s] = matmul(M[s], U[s], d, d, cols);
                    return worst_error(got, ref);
                });
            }
        }
    }

    sweep.report("PCMM sweep");
    sweep.expect_all_passed();
}

// CMT (Algorithm 3) across the same axes. Each d needs its own set of d
// automorphisms, so this is the sweep that says the rotation index derivation
// is right for every module rank rather than just for d = 8.
TEST(HEonGPU, CKKS_BatchMatrix_SweepCMT)
{
    Sweep sweep;

    struct Case
    {
        size_t degree;
        std::vector<int> logq;
        std::vector<int> logp;
        std::vector<int> ds;
    };
    const std::vector<Case> cases = {
        {4096, {50}, {50}, {2, 4, 8, 16, 32, 64}},
        {4096, {50, 40}, {50}, {2, 8, 64}},
        {8192, {50, 40, 40}, {50}, {2, 8, 32, 128}},
        {16384, {50, 40}, {50}, {8, 64}},
    };

    // CMT key-switches, so it admits real noise; 2^30 keeps that small
    // relative to entries of size one while staying inside the prime.
    const double scale = std::pow(2.0, 30);

    for (const auto& c : cases)
    {
        Env env(c.degree, c.logq, c.logp, c.ds);
        const int limbs = static_cast<int>(c.logq.size());
        const double tol = 0.05 * std::sqrt(c.degree / 4096.0);

        for (int d : c.ds)
        {
            sweep.run(label(c.degree, d, limbs), tol, [&]() {
                heongpu::BatchMatrixLayout layout(static_cast<int>(c.degree),
                                                  d);
                heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
                heongpu::BatchMatrixEncoder enc(layout.k);
                const int nslots = enc.slots();

                std::mt19937_64 rng(0x24680u + d * 7919u);
                auto M = random_batch(nslots, d, d, rng);

                std::vector<heongpu::Ciphertext<S>> cts;
                encrypt_matrix(env, op, enc, layout, M, d, d, scale, 0, cts);

                op.cmt(cts, *env.galois, *env.ops);

                auto got =
                    decrypt_matrix(env, op, enc, layout, cts, d, d, scale);

                std::vector<std::vector<cd>> ref(nslots);
                for (int s = 0; s < nslots; ++s)
                {
                    ref[s].resize(static_cast<size_t>(d) * d);
                    for (int i = 0; i < d; ++i)
                        for (int j = 0; j < d; ++j)
                            ref[s][static_cast<size_t>(i) * d + j] =
                                M[s][static_cast<size_t>(j) * d + i];
                }
                return worst_error(got, ref);
            });
        }
    }

    sweep.report("CMT sweep");
    sweep.expect_all_passed();
}

// CCMM (Algorithm 4), end to end with both operands carrying a real c1, across
// ring degrees, module ranks and limb counts.
//
// Both scales need headroom independently: the step 1 CMT's key switching noise
// lands in the right operand and is then multiplied by the left, so the decoded
// error is e_a * sqrt(N) / scale_a + e_cmt * sqrt(N) / scale_b. One 50-bit
// prime cannot supply both, which is why every case here starts from a 60-bit
// prime and at least two limbs.
TEST(HEonGPU, CKKS_BatchMatrix_SweepCCMM)
{
    Sweep sweep;

    struct Case
    {
        size_t degree;
        std::vector<int> logq;
        std::vector<int> logp;
        std::vector<int> ds;
    };
    const std::vector<Case> cases = {
        {4096, {60, 50}, {60}, {2, 4, 8, 16, 32, 64}},
        {4096, {60, 50, 40}, {60}, {4, 8, 32}},
        {8192, {60, 50}, {60}, {2, 8, 32, 128}},
        {16384, {60, 50, 40}, {60}, {8, 64}},
    };

    const double scale_a = std::pow(2.0, 25);
    const double scale_b = std::pow(2.0, 35);

    for (const auto& c : cases)
    {
        Env env(c.degree, c.logq, c.logp, c.ds);
        const int limbs = static_cast<int>(c.logq.size());
        const double tol = 0.02 * std::sqrt(c.degree / 4096.0);

        for (int d : c.ds)
        {
            sweep.run(label(c.degree, d, limbs), tol, [&]() {
                heongpu::BatchMatrixLayout layout(static_cast<int>(c.degree),
                                                  d);
                heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
                heongpu::BatchMatrixEncoder enc(layout.k);
                const int nslots = enc.slots();

                std::mt19937_64 rng(0x13579u + d * 104729u);
                auto M = random_batch(nslots, d, d, rng);
                auto U = random_batch(nslots, d, d, rng);

                std::vector<heongpu::Ciphertext<S>> ca, cb;
                encrypt_matrix(env, op, enc, layout, M, d, d, scale_a, 0, ca);
                encrypt_matrix(env, op, enc, layout, U, d, d, scale_b, 0, cb);

                std::vector<heongpu::Ciphertext<S>*> pa, pb;
                for (auto& ct : ca)
                    pa.push_back(&ct);
                for (auto& ct : cb)
                    pb.push_back(&ct);

                std::vector<heongpu::Ciphertext<S>> out;
                op.ccmm(out, pa, pb, *env.galois, *env.relin, *env.ops,
                        /*rescale=*/false);

                auto got = decrypt_matrix(env, op, enc, layout, out, d, d,
                                          scale_a * scale_b);

                std::vector<std::vector<cd>> ref(nslots);
                for (int s = 0; s < nslots; ++s)
                    ref[s] = matmul(M[s], U[s], d, d, d);
                return worst_error(got, ref);
            });
        }
    }

    sweep.report("CCMM sweep");
    sweep.expect_all_passed();
}

// The three primitives at a level below the top of the modulus chain.
//
// Everything else in the sweep runs on freshly encrypted ciphertexts, where
// depth is zero and the active limbs are the whole chain. Dropping a limb first
// changes the prime set the twiddle tables are built from and the limb count
// every kernel strides by, and it is the state operands are actually in part
// way through a circuit.
TEST(HEonGPU, CKKS_BatchMatrix_SweepLevels)
{
    Sweep sweep;

    const size_t degree = 8192;
    const std::vector<int> ds = {4, 16};
    Env env(degree, {60, 50, 40, 40}, {60}, ds);

    const double scale_m = std::pow(2.0, 20);
    const double scale_u = std::pow(2.0, 20);
    const double scale_cmt = std::pow(2.0, 30);
    const double scale_a = std::pow(2.0, 25);
    const double scale_b = std::pow(2.0, 35);
    const double tol = 0.05 * std::sqrt(degree / 4096.0);

    for (int depth : {1, 2})
    {
        for (int d : ds)
        {
            const int limbs = 4 - depth;
            std::ostringstream os;
            os << " depth=" << depth;
            const std::string suffix = os.str();

            heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
            heongpu::BatchMatrixEncoder enc(layout.k);
            const int nslots = enc.slots();

            sweep.run(label(degree, d, limbs, (suffix + " PCMM").c_str()), tol,
                      [&]() {
                          heongpu::HEBatchMatrixOperator<S> op(env.context,
                                                               layout);
                          std::mt19937_64 rng(0xFEEDu + d * 31u + depth);
                          auto M = random_batch(nslots, d, d, rng);
                          auto U = random_batch(nslots, d, 3, rng);

                          std::vector<heongpu::Ciphertext<S>> cts;
                          encrypt_matrix(env, op, enc, layout, M, d, d,
                                         scale_m, depth, cts);

                          std::vector<int64_t> u_coeffs;
                          enc.encode(U, d, 3, scale_u, u_coeffs);
                          op.encode_plaintext_matrix(u_coeffs, d, 3, depth,
                                                     scale_u);

                          std::vector<heongpu::Ciphertext<S>*> in;
                          for (auto& ct : cts)
                              in.push_back(&ct);
                          std::vector<heongpu::Ciphertext<S>> out;
                          op.pcmm(out, in, /*rescale=*/false);

                          auto got = decrypt_matrix(env, op, enc, layout, out,
                                                    d, 3, scale_m * scale_u);
                          std::vector<std::vector<cd>> ref(nslots);
                          for (int s = 0; s < nslots; ++s)
                              ref[s] = matmul(M[s], U[s], d, d, 3);
                          return worst_error(got, ref);
                      });

            sweep.run(label(degree, d, limbs, (suffix + " CMT").c_str()), tol,
                      [&]() {
                          heongpu::HEBatchMatrixOperator<S> op(env.context,
                                                               layout);
                          std::mt19937_64 rng(0xC0FFEEu + d * 17u + depth);
                          auto M = random_batch(nslots, d, d, rng);

                          std::vector<heongpu::Ciphertext<S>> cts;
                          encrypt_matrix(env, op, enc, layout, M, d, d,
                                         scale_cmt, depth, cts);
                          op.cmt(cts, *env.galois, *env.ops);

                          auto got = decrypt_matrix(env, op, enc, layout, cts,
                                                    d, d, scale_cmt);
                          std::vector<std::vector<cd>> ref(nslots);
                          for (int s = 0; s < nslots; ++s)
                          {
                              ref[s].resize(static_cast<size_t>(d) * d);
                              for (int i = 0; i < d; ++i)
                                  for (int j = 0; j < d; ++j)
                                      ref[s][static_cast<size_t>(i) * d + j] =
                                          M[s][static_cast<size_t>(j) * d + i];
                          }
                          return worst_error(got, ref);
                      });

            sweep.run(label(degree, d, limbs, (suffix + " CCMM").c_str()), tol,
                      [&]() {
                          heongpu::HEBatchMatrixOperator<S> op(env.context,
                                                               layout);
                          std::mt19937_64 rng(0xBEEFu + d * 13u + depth);
                          auto M = random_batch(nslots, d, d, rng);
                          auto U = random_batch(nslots, d, d, rng);

                          std::vector<heongpu::Ciphertext<S>> ca, cb;
                          encrypt_matrix(env, op, enc, layout, M, d, d,
                                         scale_a, depth, ca);
                          encrypt_matrix(env, op, enc, layout, U, d, d,
                                         scale_b, depth, cb);

                          std::vector<heongpu::Ciphertext<S>*> pa, pb;
                          for (auto& ct : ca)
                              pa.push_back(&ct);
                          for (auto& ct : cb)
                              pb.push_back(&ct);

                          std::vector<heongpu::Ciphertext<S>> out;
                          op.ccmm(out, pa, pb, *env.galois, *env.relin,
                                  *env.ops, /*rescale=*/false);

                          auto got =
                              decrypt_matrix(env, op, enc, layout, out, d, d,
                                             scale_a * scale_b);
                          std::vector<std::vector<cd>> ref(nslots);
                          for (int s = 0; s < nslots; ++s)
                              ref[s] = matmul(M[s], U[s], d, d, d);
                          return worst_error(got, ref);
                      });
        }
    }

    sweep.report("level sweep");
    sweep.expect_all_passed();
}

// ---------------------------------------------------------------------------
// Llama-3-8B tile shapes
// ---------------------------------------------------------------------------
//
// The sweeps above stop at N = 16384, d = 256 and four limbs, which is enough
// to say the kernels are shape agnostic in the small. A transformer layer asks
// for considerably more, and at the ring degree bootstrapping needs. Per
// decoder layer, Llama-3-8B multiplies by:
//
//     Q, O projections    4096 x 4096
//     K, V projections    4096 x 1024   (grouped query attention, 8 KV heads)
//     gate, up            4096 x 14336
//     down                14336 x 4096
//     per-head attention  head dimension 128
//
// The batch scheme multiplies d x d blocks over R_k with d dividing N, so a
// layer is tiled with a block size d and a block count. The block sizes a
// Llama-3 tiling would pick are 128 (the head dimension) through 1024; the
// block counts are 1024/d, 4096/d and 14336/d, and those are what the plaintext
// column axis carries. 14336 = 2^11 * 7 is not a power of two, so it is never a
// valid d -- it only ever appears as a column count.
//
// Ring degree is 65536 throughout: the same N the bootstrapping parameters use,
// so one context can carry both a matrix multiplication and a bootstrap.

namespace
{
    /// Decrypt only the named columns of a matrix encryption.
    ///
    /// extract_coefficients reconstructs each coefficient with a Garner CRT on
    /// the host, so decrypting all d columns of a d = 1024 encryption at
    /// N = 65536 means 67 million big-integer reconstructions. The unnamed
    /// columns are filled with zeros so the layout split still sees a full
    /// matrix; only entries in the named columns are ever compared.
    std::vector<std::vector<cd>>
    decrypt_columns(Env& env, heongpu::HEBatchMatrixOperator<S>& op,
                    const heongpu::BatchMatrixEncoder& enc,
                    const heongpu::BatchMatrixLayout& layout,
                    std::vector<heongpu::Ciphertext<S>>& cts, int rows,
                    int cols, double scale, const std::vector<int>& which)
    {
        std::vector<std::vector<int64_t>> out_columns(cols);
        for (int j : which)
        {
            heongpu::Plaintext<S> pt(env.context);
            env.decryptor->decrypt(pt, cts[j]);
            op.extract_coefficients(out_columns[j], pt);
        }
        for (int j = 0; j < cols; ++j)
            if (out_columns[j].empty())
                out_columns[j].assign(layout.N, 0);

        std::vector<int64_t> coeffs;
        heongpu::split_matrix_encryption_coefficients(out_columns, layout,
                                                      rows, cols, coeffs);
        std::vector<std::vector<cd>> got;
        enc.decode(coeffs, rows, cols, scale, got);
        return got;
    }

    /// Worst error over the named columns only.
    double column_error(const std::vector<std::vector<cd>>& got,
                        const std::vector<std::vector<cd>>& ref, int rows,
                        int cols, const std::vector<int>& which)
    {
        double worst = 0.0;
        for (size_t s = 0; s < ref.size(); ++s)
            for (int i = 0; i < rows; ++i)
                for (int j : which)
                {
                    const size_t e = static_cast<size_t>(i) * cols + j;
                    worst = std::max(worst, std::abs(got[s][e] - ref[s][e]));
                }
        return worst;
    }

    /// A spread of column indices: both edges, the middle and one off-centre.
    std::vector<int> probe_columns(int cols)
    {
        std::set<int> s;
        s.insert(0);
        s.insert(cols - 1);
        s.insert(cols / 2);
        s.insert(cols / 3);
        return std::vector<int>(s.begin(), s.end());
    }

    /// Reference product restricted to the named output columns.
    std::vector<std::vector<cd>>
    matmul_columns(const std::vector<std::vector<cd>>& a,
                   const std::vector<std::vector<cd>>& b, int n, int m, int p,
                   const std::vector<int>& which)
    {
        std::vector<std::vector<cd>> c(
            a.size(),
            std::vector<cd>(static_cast<size_t>(n) * p, cd(0.0, 0.0)));
        for (size_t s = 0; s < a.size(); ++s)
            for (int i = 0; i < n; ++i)
                for (int j : which)
                {
                    cd acc(0.0, 0.0);
                    for (int t = 0; t < m; ++t)
                        acc += a[s][static_cast<size_t>(i) * m + t] *
                               b[s][static_cast<size_t>(t) * p + j];
                    c[s][static_cast<size_t>(i) * p + j] = acc;
                }
        return c;
    }
} // namespace

// PCMM at every Llama-3 block size and block count. PCMM needs no evaluation
// keys, so this is the axis that can be pushed to d = 1024 without paying for
// d Galois keys first.
TEST(HEonGPU, CKKS_BatchMatrix_Llama3PCMM)
{
    Sweep sweep;

    struct Case
    {
        size_t degree;
        std::vector<int> logq;
        std::vector<int> logp;
        int d;
        std::vector<int> cols; // block counts 1024/d, 4096/d, 14336/d
    };
    const std::vector<Case> cases = {
        {65536, {50, 40}, {50}, 128, {1, 8, 32, 112}},
        {65536, {50, 40}, {50}, 256, {1, 4, 16, 56}},
        {65536, {50, 40}, {50}, 512, {1, 2, 8, 28}},
        {65536, {50, 40}, {50}, 1024, {1, 4, 14}},
        // The limb axis, at the two block sizes a real pipeline uses most.
        {65536, {50, 40, 40, 40}, {50}, 128, {32}},
        {65536, {50, 40, 40, 40}, {50}, 256, {16}},
    };

    const double scale_m = std::pow(2.0, 20);
    const double scale_u = std::pow(2.0, 20);

    for (const auto& c : cases)
    {
        Env env(c.degree, c.logq, c.logp, {});
        const int limbs = static_cast<int>(c.logq.size());
        const double tol = 0.05 * std::sqrt(c.degree / 4096.0);

        for (int cols : c.cols)
        {
            std::ostringstream os;
            os << " cols=" << cols;
            const std::string name =
                label(c.degree, c.d, limbs, os.str().c_str());

            sweep.run(name, tol, [&]() {
                heongpu::BatchMatrixLayout layout(static_cast<int>(c.degree),
                                                  c.d);
                heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
                heongpu::BatchMatrixEncoder enc(layout.k);
                const int nslots = enc.slots();

                std::mt19937_64 rng(0x11A3Au + c.d * 131u + cols);
                auto M = random_batch(nslots, c.d, c.d, rng);
                auto U = random_batch(nslots, c.d, cols, rng);

                std::vector<heongpu::Ciphertext<S>> cts;
                encrypt_matrix(env, op, enc, layout, M, c.d, c.d, scale_m, 0,
                               cts);

                std::vector<int64_t> u_coeffs;
                enc.encode(U, c.d, cols, scale_u, u_coeffs);
                op.encode_plaintext_matrix(u_coeffs, c.d, cols, 0, scale_u);

                std::vector<heongpu::Ciphertext<S>*> in;
                for (auto& ct : cts)
                    in.push_back(&ct);
                std::vector<heongpu::Ciphertext<S>> out;
                op.pcmm(out, in, /*rescale=*/false);
                if (static_cast<int>(out.size()) != cols)
                    throw std::runtime_error("wrong output column count");

                auto got = decrypt_matrix(env, op, enc, layout, out, c.d, cols,
                                          scale_m * scale_u);

                std::vector<std::vector<cd>> ref(nslots);
                for (int s = 0; s < nslots; ++s)
                    ref[s] = matmul(M[s], U[s], c.d, c.d, cols);
                return worst_error(got, ref);
            });
        }
    }

    sweep.report("Llama-3 PCMM");
    sweep.expect_all_passed();
}

// CMT at the Llama-3 block sizes. Each d needs its own d Galois keys, so every
// block size gets its own context rather than one context carrying the union.
//
// Two chains, because CMT key-switches and the key-switching noise, not the
// shape, is what binds at this ring degree:
//
//   {50,40} over one 50-bit P is what the small-N sweeps use. A single P prime
//   selects KEYSWITCHING_METHOD_I, so the whole 90-bit Q is switched against a
//   50-bit P and the noise carries a factor of Q/P. That is survivable at
//   N = 16384 and is not at N = 65536, where the noise also grows with
//   sqrt(N).
//
//   {60,50,50} over two 60-bit P primes selects METHOD_II: two digits of at
//   most 110 bits each against a 120-bit P, so the ratio is below one.
//
// Running both says whether a failure is the block size or the budget.
TEST(HEonGPU, CKKS_BatchMatrix_Llama3CMT)
{
    // Only the method II chain is a requirement. Method I is kept as a
    // reference row -- it is expected to fail here, and it is what says a
    // failure is the budget rather than the block size -- so it is reported
    // but not asserted on.
    Sweep sweep, reference;

    const size_t degree = 65536;
    const double tol = 0.05 * std::sqrt(degree / 4096.0);

    struct Chain
    {
        const char* name;
        std::vector<int> logq;
        std::vector<int> logp;
        double scale_exp;
        bool required;
    };
    const std::vector<Chain> chains = {
        {" ks=I", {50, 40}, {50}, 30.0, false},
        {" ks=II", {60, 50, 50}, {60, 60}, 40.0, true},
    };

    for (const auto& chain : chains)
    for (int d : {128, 256, 512, 1024})
    {
        Env env(degree, chain.logq, chain.logp, {d});
        const std::vector<int> which = probe_columns(d);
        const double scale = std::pow(2.0, chain.scale_exp);

        (chain.required ? sweep : reference)
            .run(label(degree, d, static_cast<int>(chain.logq.size()),
                        chain.name),
                  tol, [&]() {
            heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
            heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
            heongpu::BatchMatrixEncoder enc(layout.k);
            const int nslots = enc.slots();

            std::mt19937_64 rng(0x11A3Bu + d * 7919u);
            auto M = random_batch(nslots, d, d, rng);

            std::vector<heongpu::Ciphertext<S>> cts;
            encrypt_matrix(env, op, enc, layout, M, d, d, scale, 0, cts);

            op.cmt(cts, *env.galois, *env.ops);

            auto got =
                decrypt_columns(env, op, enc, layout, cts, d, d, scale, which);

            std::vector<std::vector<cd>> ref(nslots);
            for (int s = 0; s < nslots; ++s)
            {
                ref[s].assign(static_cast<size_t>(d) * d, cd(0.0, 0.0));
                for (int i = 0; i < d; ++i)
                    for (int j : which)
                        ref[s][static_cast<size_t>(i) * d + j] =
                            M[s][static_cast<size_t>(j) * d + i];
            }
            return column_error(got, ref, d, d, which);
        });
    }

    reference.report("Llama-3 CMT, method I reference (expected to fail)");
    sweep.report("Llama-3 CMT");
    sweep.expect_all_passed();
}

// CCMM at the Llama-3 block sizes. This is the memory-hungry one: it holds
// eight limbs x d x N tensors at once, 8.6 GB at d = 1024 with two limbs, on
// top of 2d ciphertexts and d Galois keys.
//
// Same two chains as the CMT check, for the same reason: CCMM's step 1 is a
// CMT, so its key-switching noise lands in the right operand and is then
// multiplied by the left.
//
// The product scale is also capped from above here, and not by noise.
// extract_coefficients returns int64, and with rescale = false the product
// coefficient is a sum of d*k = N terms at scale_a * scale_b; at 2^25 * 2^35
// that overflows int64 by d = 1024. 2^25 * 2^28 leaves seven bits of headroom.
TEST(HEonGPU, CKKS_BatchMatrix_Llama3CCMM)
{
    // As in the CMT check: method I is a reference row, not a requirement.
    Sweep sweep, reference;

    const size_t degree = 65536;
    const double tol = 0.02 * std::sqrt(degree / 4096.0);

    struct Chain
    {
        const char* name;
        std::vector<int> logq;
        std::vector<int> logp;
        double a_exp;
        double b_exp;
        bool required;
    };
    const std::vector<Chain> chains = {
        {" ks=I", {60, 50}, {60}, 25.0, 28.0, false},
        {" ks=II", {60, 50, 50}, {60, 60}, 25.0, 28.0, true},
    };

    for (const auto& chain : chains)
    for (int d : {128, 256, 512, 1024})
    {
        Env env(degree, chain.logq, chain.logp, {d});
        const std::vector<int> which = probe_columns(d);
        const double scale_a = std::pow(2.0, chain.a_exp);
        const double scale_b = std::pow(2.0, chain.b_exp);

        (chain.required ? sweep : reference)
            .run(label(degree, d, static_cast<int>(chain.logq.size()),
                        chain.name),
                  tol, [&]() {
            heongpu::BatchMatrixLayout layout(static_cast<int>(degree), d);
            heongpu::HEBatchMatrixOperator<S> op(env.context, layout);
            heongpu::BatchMatrixEncoder enc(layout.k);
            const int nslots = enc.slots();

            std::mt19937_64 rng(0x11A3Cu + d * 104729u);
            auto M = random_batch(nslots, d, d, rng);
            auto U = random_batch(nslots, d, d, rng);

            std::vector<heongpu::Ciphertext<S>> ca, cb;
            encrypt_matrix(env, op, enc, layout, M, d, d, scale_a, 0, ca);
            encrypt_matrix(env, op, enc, layout, U, d, d, scale_b, 0, cb);

            std::vector<heongpu::Ciphertext<S>*> pa, pb;
            for (auto& ct : ca)
                pa.push_back(&ct);
            for (auto& ct : cb)
                pb.push_back(&ct);

            std::vector<heongpu::Ciphertext<S>> out;
            op.ccmm(out, pa, pb, *env.galois, *env.relin, *env.ops,
                    /*rescale=*/false);

            auto got = decrypt_columns(env, op, enc, layout, out, d, d,
                                       scale_a * scale_b, which);
            auto ref = matmul_columns(M, U, d, d, d, which);
            return column_error(got, ref, d, d, which);
        });
    }

    reference.report("Llama-3 CCMM, method I reference (expected to fail)");
    sweep.report("Llama-3 CCMM");
    sweep.expect_all_passed();
}
