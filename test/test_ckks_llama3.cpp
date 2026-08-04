// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Per-function checks of the Llama-3 homomorphic primitives.
//
// Each operator is measured against the plaintext function it approximates,
// one operator per test, so a regression names the primitive that broke rather
// than the layer that contained it. The host-only tests come first: they pin
// the permutation algebra of Section 4.2 and the Chebyshev fits before any GPU
// time is spent on them.
//
// TOLERANCES
// ----------
// Every bound here is set from the error actually measured, with a couple of
// orders of headroom, not from what would merely look acceptable. Almost all of
// these primitives land on the CKKS noise floor around 1e-8, so a bound of 1e-6
// still passes comfortably while catching a real loss of precision; a bound of
// 1e-3 would have let a five hundred fold regression through. The exceptions
// are the Chebyshev approximations themselves, where the fit error dominates
// and the bound is set from the degree instead.
//
// Those two regimes behave differently from run to run, and how much headroom a
// bound needs follows from which one it is in. The plaintext operands are drawn
// from fixed seeds, so a fit-dominated bound sees the same error every run: the
// spread over six consecutive runs of the whole suite was under a part in ten
// thousand, and a factor of two of headroom is plenty there. A noise-dominated
// bound is a maximum over slots of the encryption noise, which is redrawn every
// run; the same six runs moved those by up to a factor of four, so anything
// resting on the noise floor is given at least an order and a half. Where the
// two are mixed the noisy one governs.
//
// Every precision check therefore goes through reported(), which prints the
// measured error next to the bound it is checked against. A primitive that
// still passes but has lost a digit is then visible in the log rather than
// waiting to be found by the first test whose bound is tight enough to catch
// it. The exceptions are the two rotation-index sweeps, which are contract
// tests and not precision tests: what they assert is that the advertised key
// list suffices, and they cover enough shapes that reporting every one of them
// would bury the rest of the log.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;

    /// Largest absolute difference over the compared prefix.
    double max_error(const std::vector<double>& got,
                     const std::vector<double>& want)
    {
        double worst = 0.0;
        const std::size_t n = std::min(got.size(), want.size());
        for (std::size_t i = 0; i < n; i++)
        {
            worst = std::max(worst, std::abs(got[i] - want[i]));
        }
        return worst;
    }

    /// Print the measured error alongside the bound it is checked against, so
    /// a primitive that is still passing but has lost a digit is visible.
    double reported(const char* label, const std::vector<double>& got,
                    const std::vector<double>& want)
    {
        const double worst = max_error(got, want);
        std::cout << "[ MEASURED ] " << label << " max error " << worst
                  << std::endl;
        return worst;
    }

    std::vector<double> apply_tau(const std::vector<double>& a, int d,
                                  int times)
    {
        std::vector<double> out = a;
        for (int t = 0; t < times; t++)
        {
            out = llama::permute_tau(out, d);
        }
        return out;
    }

    std::vector<double> random_matrix(int d, std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> m(static_cast<std::size_t>(d) * d);
        for (auto& v : m)
        {
            v = dist(rng);
        }
        return m;
    }

    std::vector<std::vector<double>> random_blocks(
        const llama::MatrixLayout& layout, double amplitude,
        std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> dist(-amplitude, amplitude);
        std::vector<std::vector<double>> blocks(
            layout.batch,
            std::vector<double>(static_cast<std::size_t>(layout.d) * layout.d));
        for (auto& block : blocks)
        {
            for (double& v : block)
            {
                v = dist(rng);
            }
        }
        return blocks;
    }

    /// Matrix m entry e goes to slot e * batch + m, the MatrixLayout packing.
    std::vector<double> pack_blocks(
        const std::vector<std::vector<double>>& blocks,
        const llama::MatrixLayout& layout)
    {
        const int entries = layout.d * layout.d;
        std::vector<double> out(static_cast<std::size_t>(layout.slots), 0.0);
        for (int m = 0; m < layout.batch; m++)
        {
            for (int e = 0; e < entries; e++)
            {
                out[static_cast<std::size_t>(e) * layout.batch + m] =
                    blocks[m][e];
            }
        }
        return out;
    }

    std::vector<std::vector<double>> unpack_blocks(
        const std::vector<double>& slots, const llama::MatrixLayout& layout)
    {
        const int entries = layout.d * layout.d;
        std::vector<std::vector<double>> blocks(
            layout.batch, std::vector<double>(entries));
        for (int m = 0; m < layout.batch; m++)
        {
            for (int e = 0; e < entries; e++)
            {
                blocks[m][e] =
                    slots[static_cast<std::size_t>(e) * layout.batch + m];
            }
        }
        return blocks;
    }

    double silu_host(double x) { return x / (1.0 + std::exp(-x)); }

    // -----------------------------------------------------------------------
    // Channel blocks, for the models wider than one ciphertext
    // -----------------------------------------------------------------------

    /// One channel block on the host: a d x d matrix per batch entry, which is
    /// exactly what random_blocks and pack_blocks already speak.
    using ChannelBlock = std::vector<std::vector<double>>;

    ChannelBlock zero_block(const llama::MatrixLayout& layout)
    {
        return ChannelBlock(
            layout.batch,
            std::vector<double>(static_cast<std::size_t>(layout.d) * layout.d,
                                0.0));
    }

    void add_into(ChannelBlock& acc, const ChannelBlock& term)
    {
        for (std::size_t m = 0; m < acc.size(); m++)
        {
            for (std::size_t e = 0; e < acc[m].size(); e++)
            {
                acc[m][e] += term[m][e];
            }
        }
    }

    ChannelBlock matmul_block(const ChannelBlock& a, const ChannelBlock& b,
                              int d)
    {
        ChannelBlock out(a.size());
        for (std::size_t m = 0; m < a.size(); m++)
        {
            out[m] = llama::matmul_host(a[m], b[m], d);
        }
        return out;
    }

    /// RMSNorm over the strided axis, which is the channel axis of the packed
    /// layout. Shared rather than a fixture member: two fixtures need it and
    /// it reads nothing but its arguments.
    std::vector<double> rms_norm_host(
        const std::vector<double>& x,
        const llama::Llama3Operator::RMSNormConfig& config)
    {
        std::vector<double> out(x.size());
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                const double v = x[j * config.stride + i];
                total += v * v;
            }
            const double factor =
                1.0 / std::sqrt(total / config.channels + config.eps);
            for (int j = 0; j < config.count; j++)
            {
                out[j * config.stride + i] = x[j * config.stride + i] * factor;
            }
        }
        return out;
    }

    /// Bracket the summed squares an RMSNorm will meet, which is the interval
    /// its 1/sqrt has to be fitted over.
    void square_sum_range(const std::vector<double>& x,
                          llama::Llama3Operator::RMSNormConfig& config)
    {
        double lowest = std::numeric_limits<double>::max();
        double highest = 0.0;
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                const double v = x[j * config.stride + i];
                total += v * v;
            }
            lowest = std::min(lowest, total);
            highest = std::max(highest, total);
        }
        config.sum_lo = 0.8 * lowest;
        config.sum_hi = 1.2 * highest;
    }

    /// The one-per-batch-entry form a BlockMatrix block takes.
    std::vector<double> flatten_block(const ChannelBlock& block)
    {
        std::vector<double> out;
        for (const auto& matrix : block)
        {
            out.insert(out.end(), matrix.begin(), matrix.end());
        }
        return out;
    }

    /// want[i] = sum_j w[i][j] x[j], the block product the layers perform.
    std::vector<ChannelBlock> block_project_host(
        const std::vector<std::vector<ChannelBlock>>& w,
        const std::vector<ChannelBlock>& x, const llama::MatrixLayout& layout)
    {
        std::vector<ChannelBlock> want(w.size());
        for (std::size_t i = 0; i < w.size(); i++)
        {
            want[i] = zero_block(layout);
            for (std::size_t j = 0; j < x.size(); j++)
            {
                add_into(want[i], matmul_block(w[i][j], x[j], layout.d));
            }
        }
        return want;
    }

    /// The causal SoftMax of a score block, keys on the slow axis.
    ChannelBlock causal_softmax_host(const ChannelBlock& scores, double scale,
                                     double shift, int d)
    {
        ChannelBlock out(scores.size());
        for (std::size_t m = 0; m < scores.size(); m++)
        {
            out[m].assign(static_cast<std::size_t>(d) * d, 0.0);
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int key = 0; key <= query; key++)
                {
                    total +=
                        std::exp(scores[m][key * d + query] * scale - shift);
                }
                for (int key = 0; key <= query; key++)
                {
                    out[m][key * d + query] =
                        std::exp(scores[m][key * d + query] * scale - shift) /
                        total;
                }
            }
        }
        return out;
    }

    /// The causal SoftMax of one query token block whose keys are cut into
    /// blocks: @p scores holds key block 0 up to @p query_block, and every
    /// query normalises over every key it admits, wherever that key sits.
    ///
    /// The mask weights are absent here on purpose. They are constant along
    /// the key axis and the normalisation cancels them, so a reference that
    /// reproduced them would only be testing that they cancel.
    std::vector<ChannelBlock> causal_block_softmax_host(
        const std::vector<ChannelBlock>& scores, double scale, double shift,
        int d, int query_block)
    {
        const std::size_t blocks = scores.size();
        const std::size_t batch = scores.front().size();
        std::vector<ChannelBlock> out(
            blocks,
            ChannelBlock(batch, std::vector<double>(
                                    static_cast<std::size_t>(d) * d, 0.0)));

        for (std::size_t m = 0; m < batch; m++)
        {
            for (int query = 0; query < d; query++)
            {
                const int reach = query_block * d + query;
                double total = 0.0;
                for (std::size_t u = 0; u < blocks; u++)
                {
                    for (int key = 0; key < d; key++)
                    {
                        if (static_cast<int>(u) * d + key > reach)
                        {
                            continue;
                        }
                        total += std::exp(
                            scores[u][m][key * d + query] * scale - shift);
                    }
                }
                for (std::size_t u = 0; u < blocks; u++)
                {
                    for (int key = 0; key < d; key++)
                    {
                        if (static_cast<int>(u) * d + key > reach)
                        {
                            continue;
                        }
                        out[u][m][key * d + query] =
                            std::exp(scores[u][m][key * d + query] * scale -
                                     shift) /
                            total;
                    }
                }
            }
        }
        return out;
    }

    /// The widest scaled score, for calibrating a SoftMax into [-bound, 0].
    void score_range(const std::vector<ChannelBlock>& blocks, double& lowest,
                     double& highest)
    {
        lowest = std::numeric_limits<double>::max();
        highest = std::numeric_limits<double>::lowest();
        for (const auto& block : blocks)
        {
            for (const auto& entry : block)
            {
                for (double v : entry)
                {
                    lowest = std::min(lowest, v);
                    highest = std::max(highest, v);
                }
            }
        }
    }

    /// K^T Q with the keys landing on the slow axis, as attention forms them.
    ChannelBlock scores_host(const ChannelBlock& key, const ChannelBlock& query,
                             int d)
    {
        ChannelBlock out(key.size());
        for (std::size_t m = 0; m < key.size(); m++)
        {
            out[m] = llama::matmul_host(llama::transpose_host(key[m], d),
                                        query[m], d);
        }
        return out;
    }

    /// The plaintext weights of one block, in the host's block form.
    struct BlockWeightsHost
    {
        ChannelBlock query;
        ChannelBlock key;
        ChannelBlock value;
        ChannelBlock gate;
        ChannelBlock up;
        ChannelBlock down;
    };

    /// What one host block leaves behind: its output, the residual stream
    /// halfway through it, and the widest gate the SiLU fit has to cover.
    struct BlockHostRun
    {
        ChannelBlock out;
        /// After the attention half's residual, which is what the refresh in
        /// the middle of the block is handed.
        ChannelBlock stream;
        double widest_gate = 0.0;
    };

    /// One whole transformer block on the host, and the calibration that goes
    /// with it.
    ///
    /// The parts of @p config a caller cannot know in advance are filled in
    /// here, because they are properties of the activations and not of the
    /// architecture: the interval each norm's reciprocal square root is fitted
    /// over, and the scale and shift that carry the scores into the SoftMax's
    /// fitted interval. In a stack no two blocks see the same activations, so
    /// no two blocks get the same numbers. The caller sets the shapes and the
    /// degrees; this sets the ranges.
    ///
    /// The second norm takes its shape from the first, which is what a
    /// pre-norm block does, and then gets its own interval.
    ///
    /// RoPE is not applied. Every caller here runs without it, and a
    /// reference that pretended otherwise would be untested code.
    BlockHostRun transformer_block_host(
        const ChannelBlock& x, const BlockWeightsHost& w,
        llama::Llama3Operator::TransformerBlockConfig& config,
        const llama::MatrixLayout& layout)
    {
        const int d = layout.d;

        const std::vector<double> x_slots = pack_blocks(x, layout);
        square_sum_range(x_slots, config.attention_norm);
        const ChannelBlock normed =
            unpack_blocks(rms_norm_host(x_slots, config.attention_norm),
                          layout);

        const ChannelBlock q = matmul_block(w.query, normed, d);
        const ChannelBlock k = matmul_block(w.key, normed, d);
        const ChannelBlock v = matmul_block(w.value, normed, d);
        const ChannelBlock raw = scores_host(k, q, d);

        double lowest = 0.0;
        double highest = 0.0;
        score_range({raw}, lowest, highest);
        config.attention.head_scale = 2.0 / (highest - lowest);
        config.attention.score_shift = highest * config.attention.head_scale;

        const ChannelBlock probabilities =
            causal_softmax_host(raw, config.attention.head_scale,
                                config.attention.score_shift, d);

        // The residual stream after the attention half. This is what the
        // refresh in the middle of the block is handed, and what the second
        // norm reduces over.
        ChannelBlock stream = matmul_block(v, probabilities, d);
        add_into(stream, x);
        const std::vector<double> stream_slots = pack_blocks(stream, layout);

        config.feed_forward_norm = config.attention_norm;
        square_sum_range(stream_slots, config.feed_forward_norm);
        const ChannelBlock stream_normed = unpack_blocks(
            rms_norm_host(stream_slots, config.feed_forward_norm), layout);

        BlockHostRun run;
        run.stream = stream;
        run.out.resize(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            const std::vector<double> g =
                llama::matmul_host(w.gate[m], stream_normed[m], d);
            const std::vector<double> u =
                llama::matmul_host(w.up[m], stream_normed[m], d);
            std::vector<double> hidden(g.size());
            for (std::size_t e = 0; e < g.size(); e++)
            {
                run.widest_gate = std::max(run.widest_gate, std::abs(g[e]));
                hidden[e] = silu_host(g[e]) * u[e];
            }
            const std::vector<double> out =
                llama::matmul_host(w.down[m], hidden, d);
            run.out[m].resize(out.size());
            for (std::size_t e = 0; e < out.size(); e++)
            {
                run.out[m][e] = stream[m][e] + out[e];
            }
        }
        return run;
    }

    // -----------------------------------------------------------------------
    // Host-only checks
    // -----------------------------------------------------------------------

    TEST(CKKS_Llama3, ChebyshevFitsTheNonLinearities)
    {
        struct Case
        {
            const char* name;
            std::function<double(double)> f;
            double a;
            double b;
            int degree;
            double tolerance;
        };

        const std::vector<Case> cases = {
            {"silu", [](double x) { return x / (1.0 + std::exp(-x)); }, -11.0,
             11.0, 31, 1e-3},
            {"inverse sqrt", [](double x) { return 1.0 / std::sqrt(x); }, 0.5,
             2.0, 15, 1e-6},
            {"inverse", [](double x) { return 1.0 / x; }, 2.0, 48.0, 15, 5e-3},
            {"exp scaled", [](double x) { return std::exp(x / 2.0); }, -2.0,
             0.0, 15, 1e-9},
        };

        for (const Case& c : cases)
        {
            const std::vector<double> coeffs =
                llama::chebyshev_coefficients(c.f, c.a, c.b, c.degree);
            ASSERT_EQ(static_cast<int>(coeffs.size()), c.degree + 1) << c.name;

            double worst = 0.0;
            for (int i = 0; i <= 400; i++)
            {
                const double x = c.a + (c.b - c.a) * i / 400.0;
                worst = std::max(
                    worst, std::abs(llama::chebyshev_evaluate(coeffs, c.a, c.b,
                                                              x) -
                                    c.f(x)));
            }
            EXPECT_LT(worst, c.tolerance)
                << c.name << " fit is worse than expected: " << worst;
        }
    }

    /// The Jiang-Kim-Lauter-Song identity ccmm is built on.
    ///
    /// Section 4.2 only covers a plaintext weight, and attention needs two
    /// encrypted operands, so this is the algebra that fills the gap. Checking
    /// it here costs nothing and a failure names the identity rather than the
    /// homomorphic machinery around it.
    TEST(CKKS_Llama3, EncryptedProductIdentityReproducesTheProduct)
    {
        std::mt19937_64 rng(20260805);
        const int d = 8;
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> b = random_matrix(d, rng);

        const std::vector<double> sigma_a = llama::permute_sigma(a, d);
        const std::vector<double> tau_b = llama::permute_tau(b, d);

        std::vector<double> acc(static_cast<std::size_t>(d) * d, 0.0);
        for (int k = 0; k < d; k++)
        {
            const std::vector<double> left =
                llama::rotate_cols_host(sigma_a, d, k);
            const std::vector<double> right =
                llama::rotate_rows_host(tau_b, d, k);
            for (std::size_t e = 0; e < acc.size(); e++)
            {
                acc[e] += left[e] * right[e];
            }
        }

        EXPECT_LT(max_error(acc, llama::matmul_host(a, b, d)), 1e-12);
    }

    /// The decomposition that keeps ccmm at depth two.
    ///
    /// rot_C^k(sigma(A)) is usually built by applying sigma and then rotating
    /// the columns, two levels. It does not have to be: every entry of it is an
    /// entry of A reached by a flat shift that depends only on the diagonal the
    /// entry lies on, so one rotation of A serves a whole diagonal and all d
    /// values of k share the same 2d - 1 rotations. Only the masks differ, and
    /// masking is the one level. This pins that claim.
    TEST(CKKS_Llama3, ColumnRotatedSigmaIsOneShiftOfAPerDiagonal)
    {
        std::mt19937_64 rng(20260806);
        const int d = 8;
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> sigma_a = llama::permute_sigma(a, d);

        for (int k = 0; k < d; k++)
        {
            std::vector<double> got(static_cast<std::size_t>(d) * d, 0.0);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    const int shift = ((i + j + k) % d) - j;
                    // A flat shift by `shift`, which is what a slot rotation
                    // gives, and it never leaves row i.
                    got[i * d + j] = a[i * d + j + shift];
                }
            }

            EXPECT_EQ(got, llama::rotate_cols_host(sigma_a, d, k))
                << "k = " << k;
        }
    }

    /// The causal mask keeps the past and leaves the first round's range put.
    ///
    /// A plain zero-one mask would not: the sum of squares the first
    /// reciprocal has to cover would shrink with the number of keys a query
    /// can see, so the fit would have to span the whole sequence length. The
    /// weight sqrt(d / kept) makes the sum of squared weights exactly d for
    /// every query, which is what an unmasked row would give, and the rounds
    /// normalise so the weight itself cancels.
    TEST(CKKS_Llama3, CausalMaskHidesTheFutureAtAConstantWeight)
    {
        const int d = 8;
        const llama::MatrixLayout layout(d, 4);
        const std::vector<double> mask =
            llama::Llama3Operator::causal_mask(layout);
        ASSERT_EQ(static_cast<int>(mask.size()), layout.slots);

        for (int query = 0; query < d; query++)
        {
            for (int m = 0; m < layout.batch; m++)
            {
                double squares = 0.0;
                for (int key = 0; key < d; key++)
                {
                    const double w =
                        mask[(key * d + query) * layout.batch + m];
                    if (key > query)
                    {
                        EXPECT_EQ(w, 0.0) << "key " << key << " of query "
                                          << query;
                    }
                    else
                    {
                        EXPECT_GT(w, 0.0);
                    }
                    squares += w * w;
                }
                EXPECT_NEAR(squares, static_cast<double>(d), 1e-12)
                    << "query " << query;
            }
        }
    }

    /// Theorem 2 and Equation (5), evaluated entirely on the host.
    ///
    /// If the permutation algebra is wrong the homomorphic PCMM cannot be
    /// right either, and finding that out here costs nothing.
    TEST(CKKS_Llama3, EquationFiveReproducesTauOfTheProduct)
    {
        std::mt19937_64 rng(20260804);
        const int d = 16;

        for (int ell = 0; ell <= 2; ell++)
        {
            for (int baby : {2, 4, 8})
            {
                const int giant = d / baby;

                const std::vector<double> a = random_matrix(d, rng);
                const std::vector<double> b = random_matrix(d, rng);

                const std::vector<double> stored =
                    apply_tau(llama::permute_sigma(a, d), d, ell);
                const std::vector<double> operand = apply_tau(b, d, ell + 1);

                std::vector<double> acc(static_cast<std::size_t>(d) * d, 0.0);
                for (int j = 0; j < giant; j++)
                {
                    std::vector<double> inner(
                        static_cast<std::size_t>(d) * d, 0.0);
                    for (int i = 0; i < baby; i++)
                    {
                        const std::vector<double> block =
                            heongpu::llama::Llama3Operator::
                                pcmm_plaintext_block(stored, d, i, j, baby,
                                                     ell);
                        const std::vector<double> rotated =
                            llama::rotate_rows_host(operand, d, i);
                        for (std::size_t e = 0; e < inner.size(); e++)
                        {
                            inner[e] += block[e] * rotated[e];
                        }
                    }

                    const std::vector<double> shifted =
                        llama::rotate_rows_host(inner, d, j * baby);
                    for (std::size_t e = 0; e < acc.size(); e++)
                    {
                        acc[e] += shifted[e];
                    }
                }

                const std::vector<double> want =
                    apply_tau(llama::matmul_host(a, b, d), d, ell);
                EXPECT_LT(max_error(acc, want), 1e-10)
                    << "ell=" << ell << " baby=" << baby;
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU fixture
    // -----------------------------------------------------------------------

    /// One context deep enough for every primitive below.
    ///
    /// The chain is long because the tests exercise the primitives back to
    /// back without bootstrapping between them; Sylph itself would bootstrap
    /// and run each of these at a far shorter chain.
    class Llama3Env : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 8192;
        static constexpr int kLimbs = 24;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            // Two special primes so key switching runs method II; a single one
            // leaves no noise budget worth speaking of.
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();

            std::set<int> indices;
            for (int r : llama::Llama3Operator::strided_rotation_indices(
                     slots / 32, 32))
            {
                indices.insert(r);
            }
            for (int r : llama::Llama3Operator::strided_rotation_indices(
                     slots / 64, 64))
            {
                indices.insert(r);
            }
            for (int span : {16, 32})
            {
                for (int r :
                     llama::Llama3Operator::blocked_rotation_indices(span))
                {
                    indices.insert(r);
                }
            }
            const llama::MatrixLayout layout(16, slots / 256);
            // Every BSGS split of d = 16 the tests exercise.
            for (int baby : {2, 4, 8})
            {
                for (int r : llama::Llama3Operator::pcmm_rotation_indices(
                         layout, 16 / baby, baby))
                {
                    indices.insert(r);
                }
            }
            indices.insert(kRopeSwap);

            std::vector<int> shifts(indices.begin(), indices.end());
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        static constexpr int kRopeSwap = 64;

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& values)
        {
            heongpu::Plaintext<S> plain(context);
            encoder->encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor->encrypt(cipher, plain);
            return cipher;
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& cipher)
        {
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, cipher);
            std::vector<double> values;
            encoder->decode(values, plain);
            return values;
        }

        /// A Galois key holding exactly @p shifts and nothing else.
        ///
        /// Galoiskey(context, shift_vec) stores only what it is handed, with
        /// no power-of-two fallback, and leaves the bounds the fallback path
        /// reads uninitialised. Building a key from exactly what a primitive
        /// advertises is therefore the only way to find out whether the
        /// advertised list is complete; the fixture's union key would hide a
        /// missing index until it became undefined behaviour in a layer.
        std::unique_ptr<heongpu::Galoiskey<S>> narrow_key(std::vector<int>
                                                              shifts)
        {
            auto key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*key, *secret);
            return key;
        }

        std::vector<double> uniform(double lo, double hi, uint64_t seed)
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(lo, hi);
            std::vector<double> v(slots);
            for (double& x : v)
            {
                x = dist(rng);
            }
            return v;
        }
    };

    // -----------------------------------------------------------------------
    // Slot reductions
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, SumStridedReducesTheSlowAxis)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-1.0, 1.0, 11);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->sum_strided(cipher, stride, count, *galois);
        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                total += values[j * stride + i];
            }
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = total;
            }
        }

        EXPECT_LT(reported("sum_strided", got, want), 1e-6);
    }

    TEST_F(Llama3Env, SumBlockedReducesContiguousBlocks)
    {
        const int span = 16;

        const std::vector<double> values = uniform(-1.0, 1.0, 12);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->sum_blocked(cipher, span, *galois);
        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int base = 0; base < slots; base += span)
        {
            double total = 0.0;
            for (int j = 0; j < span; j++)
            {
                total += values[base + j];
            }
            for (int j = 0; j < span; j++)
            {
                want[base + j] = total;
            }
        }

        EXPECT_LT(reported("sum_blocked", got, want), 1e-5);
    }

    /// Plaintext operands below the top of the chain.
    ///
    /// HEEncoder always encodes at the top level and multiply_plain insists
    /// the two operands agree, so every plaintext step is a no-op away from
    /// throwing once it runs anywhere but on a fresh ciphertext. Testing the
    /// primitives only at depth zero hides that completely.
    TEST_F(Llama3Env, PlaintextStepsWorkBelowTheTopLevel)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-1.0, 1.0, 13);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        ops->square(cipher, *relin);
        ASSERT_EQ(cipher.depth(), 1);
        ops->sum_strided(cipher, stride, count, *galois);
        ops->multiply_constant(cipher, 0.25);
        ASSERT_EQ(cipher.depth(), 2);

        std::vector<double> mask(slots, 2.0);
        ops->multiply_vector(cipher, mask);
        ASSERT_EQ(cipher.depth(), 3);

        const std::vector<double> got = decrypt(cipher);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = total * 0.25 * 2.0;
            }
        }

        EXPECT_LT(reported("plaintext below the top", got, want), 1e-5);
    }

    // -----------------------------------------------------------------------
    // Polynomial primitives
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, InverseSqrtMatchesTheFunction)
    {
        const std::vector<double> values = uniform(0.5, 2.0, 21);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->inverse_sqrt(cipher, 0.5, 2.0, 15, 2, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = 1.0 / std::sqrt(values[i]);
        }

        EXPECT_LT(reported("inverse_sqrt", got, want), 1e-6);
    }

    TEST_F(Llama3Env, InverseMatchesTheFunction)
    {
        const std::vector<double> values = uniform(2.0, 48.0, 22);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->inverse(cipher, 2.0, 48.0, 15, 2, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = 1.0 / values[i];
        }

        EXPECT_LT(reported("inverse", got, want), 1e-6);
    }

    TEST_F(Llama3Env, SiluMatchesTheActivation)
    {
        // Table 2 puts the calibrated SiLU input inside about 10.8, and
        // Section 3.1.3 reports degree 31 for that range.
        const double bound = 11.0;
        const std::vector<double> values = uniform(-bound, bound, 23);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result = ops->silu(cipher, bound, 31, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] / (1.0 + std::exp(-values[i]));
        }

        EXPECT_LT(reported("silu", got, want), 1e-3);
    }

    // -----------------------------------------------------------------------
    // Layers
    // -----------------------------------------------------------------------

    TEST_F(Llama3Env, RopeAppliesTheRotation)
    {
        const int half = kRopeSwap;
        const std::vector<double> values = uniform(-1.0, 1.0, 31);

        // A head of 2 * half channels: the first half pairs with the second,
        // and the swap is a single slot rotation.
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int i = 0; i < slots; i++)
        {
            const double theta = 0.01 * ((i / (2 * half)) + 1) *
                                 ((i % half) + 1);
            const bool lower = (i % (2 * half)) < half;
            cos_values[i] = std::cos(theta);
            sin_values[i] = lower ? -std::sin(theta) : std::sin(theta);
        }

        heongpu::Ciphertext<S> cipher = encrypt(values);
        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, cos_values, scale);
        encoder->encode(sin_plain, sin_values, scale);

        heongpu::Ciphertext<S> result =
            ops->rope(cipher, cos_plain, sin_plain, half, *galois);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] * cos_values[i] +
                      values[(i + half) % slots] * sin_values[i];
        }

        EXPECT_LT(reported("rope", got, want), 1e-6);
    }

    TEST_F(Llama3Env, RMSNormMatchesThePlaintextLayer)
    {
        const int count = 64;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count;
        config.eps = 1e-5;
        config.sum_lo = 32.0;
        config.sum_hi = 128.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(41);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        std::vector<double> weight(slots);
        for (int i = 0; i < slots; i++)
        {
            weight[i] = 0.5 + 0.5 * ((i % 7) / 7.0);
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values)};
        heongpu::Plaintext<S> weight_plain(context);
        encoder->encode(weight_plain, weight, scale);
        std::vector<heongpu::Plaintext<S>> weights{weight_plain};

        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, weights, config, *galois, *relin);
        ASSERT_EQ(out.size(), 1u);
        const std::vector<double> got = decrypt(out[0]);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            const double factor =
                1.0 / std::sqrt(total / count + config.eps);
            for (int j = 0; j < count; j++)
            {
                const int p = j * stride + i;
                want[p] = values[p] * factor * weight[p];
            }
        }

        EXPECT_LT(reported("rms_norm", got, want), 1e-6);
    }

    TEST_F(Llama3Env, SoftmaxNormalisesEachInstance)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.count = 32;
        config.stride = slots / config.count;
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        const std::vector<double> values = uniform(-config.bound, 0.0, 51);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[j * config.stride + i]);
            }
            for (int j = 0; j < config.count; j++)
            {
                const int p = j * config.stride + i;
                want[p] = std::exp(values[p]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=1 strided", got, want), 1e-6);

        // Whatever the approximation error, every instance must still sum to
        // one: that is what the normalise-and-square round enforces.
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += got[j * config.stride + i];
            }
            EXPECT_NEAR(total, 1.0, 1e-3) << "instance " << i;
        }
    }

    TEST_F(Llama3Env, PcmmProducesTauOfTheProduct)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        ASSERT_EQ(layout.slots, slots);

        std::mt19937_64 rng(61);
        const int batch = layout.batch;

        for (int ell = 0; ell <= 1; ell++)
        {
            std::vector<std::vector<double>> a(batch);
            std::vector<std::vector<double>> b(batch);
            std::vector<double> stored;
            std::vector<double> operand_slots(slots, 0.0);
            std::vector<double> want(slots, 0.0);

            for (int m = 0; m < batch; m++)
            {
                a[m] = random_matrix(d, rng);
                b[m] = random_matrix(d, rng);

                const std::vector<double> stored_m =
                    apply_tau(llama::permute_sigma(a[m], d), d, ell);
                stored.insert(stored.end(), stored_m.begin(), stored_m.end());

                const std::vector<double> operand =
                    apply_tau(b[m], d, ell + 1);
                const std::vector<double> expected =
                    apply_tau(llama::matmul_host(a[m], b[m], d), d, ell);

                for (int e = 0; e < d * d; e++)
                {
                    operand_slots[e * batch + m] = operand[e];
                    want[e * batch + m] = expected[e];
                }
            }

            heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
            heongpu::Ciphertext<S> result =
                ops->pcmm(cipher, stored, layout, 4, 4, ell, *galois);
            const std::vector<double> got = decrypt(result);

            EXPECT_LT(reported("pcmm", got, want), 1e-6) << "ell=" << ell;
        }
    }

    // -----------------------------------------------------------------------
    // Branches the first round of tests never reached
    // -----------------------------------------------------------------------

    /// A polynomial result must be usable, not merely correct.
    ///
    /// evaluate_poly rescales its result only when the scale has grown past
    /// half the target, so it can hand back a ciphertext with a rescale still
    /// pending. multiply, rotate and mod_drop all refuse such a ciphertext, so
    /// a primitive that returns one is a delayed exception rather than a
    /// value. Squaring and rotating the output is the cheapest way to say so.
    TEST_F(Llama3Env, PolynomialResultsAreReadyForFurtherWork)
    {
        const int count = 32;
        const int stride = slots / count;

        const std::vector<double> values = uniform(-11.0, 11.0, 71);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> activated = ops->silu(cipher, 11.0, 15, *relin);
        ASSERT_NO_THROW(ops->sum_strided(activated, stride, count, *galois));
        ASSERT_NO_THROW(ops->square(activated, *relin));

        heongpu::Ciphertext<S> wide = encrypt(uniform(2.0, 48.0, 72));
        heongpu::Ciphertext<S> inverted =
            ops->inverse(wide, 2.0, 48.0, 15, 1, *relin);
        ASSERT_NO_THROW(ops->multiply_constant(inverted, 2.0));

        // newton_iterations = 0 returns the bare Chebyshev seed, the one path
        // that leaves evaluate_poly's output completely untouched.
        heongpu::Ciphertext<S> narrow = encrypt(uniform(0.5, 2.0, 73));
        heongpu::Ciphertext<S> root =
            ops->inverse_sqrt(narrow, 0.5, 2.0, 15, 0, *relin);
        ASSERT_NO_THROW(ops->square(root, *relin));
    }

    /// The channel split, which is the whole reason rms_norm takes a vector.
    TEST_F(Llama3Env, RMSNormSplitsChannelsOverSeveralCiphertexts)
    {
        const int count = 32;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = 2 * count; // two ciphertexts of count channels
        config.eps = 1e-5;
        config.sum_lo = 20.0;
        config.sum_hi = 150.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(81);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<std::vector<double>> values(2,
                                                std::vector<double>(slots));
        for (auto& v : values)
        {
            for (double& x : v)
            {
                x = dist(rng);
            }
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values[0]),
                                               encrypt(values[1])};
        std::vector<heongpu::Plaintext<S>> no_weights;

        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, no_weights, config, *galois, *relin);
        ASSERT_EQ(out.size(), 2u);

        for (int part = 0; part < 2; part++)
        {
            const std::vector<double> got = decrypt(out[part]);
            std::vector<double> want(slots);
            for (int i = 0; i < stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < count; j++)
                {
                    for (int p = 0; p < 2; p++)
                    {
                        const double v = values[p][j * stride + i];
                        total += v * v;
                    }
                }
                const double factor =
                    1.0 / std::sqrt(total / config.channels + config.eps);
                for (int j = 0; j < count; j++)
                {
                    const int p = j * stride + i;
                    want[p] = values[part][p] * factor;
                }
            }
            const std::string label =
                "rms_norm split[" + std::to_string(part) + "]";
            EXPECT_LT(reported(label.c_str(), got, want), 1e-6);
        }
    }

    /// The blocked layout, which is what a SoftMax over the token axis needs
    /// when the reduced axis is contiguous rather than strided.
    TEST_F(Llama3Env, SoftmaxNormalisesContiguousBlocks)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = false;
        config.count = 32;
        config.stride = 0; // unused in the blocked layout
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        const std::vector<double> values = uniform(-config.bound, 0.0, 91);
        heongpu::Ciphertext<S> cipher = encrypt(values);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);
        const std::vector<double> got = decrypt(result);

        std::vector<double> want(slots);
        for (int base = 0; base < slots; base += config.count)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[base + j]);
            }
            for (int j = 0; j < config.count; j++)
            {
                want[base + j] = std::exp(values[base + j]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=1 blocked", got, want), 1e-6);
    }

    /// One weight matrix shared by every matrix in the batch, the form a
    /// projection actually takes: one plaintext W against many token blocks.
    TEST_F(Llama3Env, PcmmSharesOneBlockAcrossTheBatch)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;

        std::mt19937_64 rng(101);
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<double> operand_slots(slots, 0.0);
        std::vector<double> want(slots, 0.0);
        for (int m = 0; m < batch; m++)
        {
            const std::vector<double> b = random_matrix(d, rng);
            const std::vector<double> operand = llama::permute_tau(b, d);
            const std::vector<double> expected = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                operand_slots[e * batch + m] = operand[e];
                want[e * batch + m] = expected[e];
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
        heongpu::Ciphertext<S> result =
            ops->pcmm(cipher, stored, layout, 4, 4, 0, *galois);
        EXPECT_LT(reported("pcmm one block", decrypt(result), want), 1e-6);
    }

    /// Lopsided BSGS splits. The rotation sets differ from the square split,
    /// so a key or an index that is only right when giant == baby shows here.
    TEST_F(Llama3Env, PcmmHandlesLopsidedBsgsSplits)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;

        std::mt19937_64 rng(111);
        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<double> operand_slots(slots, 0.0);
        std::vector<double> want(slots, 0.0);
        for (int m = 0; m < batch; m++)
        {
            const std::vector<double> b = random_matrix(d, rng);
            const std::vector<double> operand = llama::permute_tau(b, d);
            const std::vector<double> expected = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                operand_slots[e * batch + m] = operand[e];
                want[e * batch + m] = expected[e];
            }
        }

        for (int baby : {2, 8})
        {
            heongpu::Ciphertext<S> cipher = encrypt(operand_slots);
            heongpu::Ciphertext<S> result =
                ops->pcmm(cipher, stored, layout, d / baby, baby, 0, *galois);
            const std::string label =
                "pcmm baby=" + std::to_string(baby);
            EXPECT_LT(reported(label.c_str(), decrypt(result), want), 1e-6);
        }
    }

    /// A weight plaintext is prepared once and used at every level it meets.
    /// multiply_plaintext promises to take the drop on a copy; if it dropped
    /// the caller's plaintext instead, the second use would be at the wrong
    /// level and would either throw or decode as noise.
    TEST_F(Llama3Env, PlaintextOperandSurvivesReuseAtSeveralLevels)
    {
        const std::vector<double> values = uniform(-1.0, 1.0, 121);
        std::vector<double> weight(slots);
        for (int i = 0; i < slots; i++)
        {
            weight[i] = 0.25 + 0.5 * ((i % 5) / 5.0);
        }

        heongpu::Plaintext<S> weight_plain(context);
        encoder->encode(weight_plain, weight, scale);

        // Deep first, so a mutated plaintext would break the shallow use.
        heongpu::Ciphertext<S> deep = encrypt(values);
        ops->square(deep, *relin);
        ops->multiply_constant(deep, 0.5);
        ASSERT_EQ(deep.depth(), 2);
        ops->multiply_plaintext(deep, weight_plain);
        ops->rescale_inplace(deep);

        heongpu::Ciphertext<S> shallow = encrypt(values);
        ASSERT_EQ(shallow.depth(), 0);
        ops->multiply_plaintext(shallow, weight_plain);
        ops->rescale_inplace(shallow);

        std::vector<double> want_deep(slots);
        std::vector<double> want_shallow(slots);
        for (int i = 0; i < slots; i++)
        {
            want_deep[i] = values[i] * values[i] * 0.5 * weight[i];
            want_shallow[i] = values[i] * weight[i];
        }

        EXPECT_LT(reported("plaintext reuse deep", decrypt(deep), want_deep),
                  1e-6);
        EXPECT_LT(
            reported("plaintext reuse shallow", decrypt(shallow), want_shallow),
            1e-6);
    }

    // -----------------------------------------------------------------------
    // The advertised rotation indices are the whole contract
    // -----------------------------------------------------------------------

    /// Each reduction, given a key holding only what it asked for.
    TEST_F(Llama3Env, ReductionsAskForEveryRotationTheyUse)
    {
        for (int count : {2, 8, 32, 128})
        {
            SCOPED_TRACE("strided count=" + std::to_string(count));
            const int stride = slots / count;
            auto key = narrow_key(
                llama::Llama3Operator::strided_rotation_indices(stride, count));

            const std::vector<double> values = uniform(-1.0, 1.0, 200 + count);
            heongpu::Ciphertext<S> cipher = encrypt(values);
            ASSERT_NO_THROW(ops->sum_strided(cipher, stride, count, *key));

            std::vector<double> want(slots);
            for (int i = 0; i < stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < count; j++)
                {
                    total += values[j * stride + i];
                }
                for (int j = 0; j < count; j++)
                {
                    want[j * stride + i] = total;
                }
            }
            EXPECT_LT(max_error(decrypt(cipher), want), 1e-5);
        }

        for (int span : {2, 8, 64})
        {
            SCOPED_TRACE("blocked span=" + std::to_string(span));
            auto key = narrow_key(
                llama::Llama3Operator::blocked_rotation_indices(span));

            const std::vector<double> values = uniform(-1.0, 1.0, 300 + span);
            heongpu::Ciphertext<S> cipher = encrypt(values);
            ASSERT_NO_THROW(ops->sum_blocked(cipher, span, *key));

            std::vector<double> want(slots);
            for (int base = 0; base < slots; base += span)
            {
                double total = 0.0;
                for (int j = 0; j < span; j++)
                {
                    total += values[base + j];
                }
                for (int j = 0; j < span; j++)
                {
                    want[base + j] = total;
                }
            }
            EXPECT_LT(max_error(decrypt(cipher), want), 1e-5);
        }
    }

    /// PCMM over every square matrix that fills these slots, both BSGS
    /// orientations and both tau exponents, each against a key built from
    /// pcmm_rotation_indices alone. d = 64 leaves batch = 1, the degenerate
    /// packing where the batch axis disappears.
    TEST_F(Llama3Env, PcmmCoversEveryShapeWithTheKeysItAsksFor)
    {
        std::mt19937_64 rng(401);
        constexpr int kElls = 2;

        for (int d : {8, 16, 32, 64})
        {
            const llama::MatrixLayout layout(d, slots / (d * d));
            ASSERT_EQ(layout.slots, slots);
            const int batch = layout.batch;
            const std::size_t entries = static_cast<std::size_t>(d) * d;

            // One operand set per tau exponent, built once: the rotation keys
            // do not depend on ell, so keeping the splits outside keeps the
            // key generations down to one per split.
            std::vector<std::vector<double>> stored(kElls);
            std::vector<std::vector<double>> operand_slots(
                kElls, std::vector<double>(slots, 0.0));
            std::vector<std::vector<double>> want(
                kElls, std::vector<double>(slots, 0.0));

            for (int m = 0; m < batch; m++)
            {
                const std::vector<double> a = random_matrix(d, rng);
                const std::vector<double> b = random_matrix(d, rng);
                const std::vector<double> product = llama::matmul_host(a, b, d);
                const std::vector<double> sigma_a = llama::permute_sigma(a, d);

                for (int ell = 0; ell < kElls; ell++)
                {
                    const std::vector<double> stored_m =
                        apply_tau(sigma_a, d, ell);
                    stored[ell].insert(stored[ell].end(), stored_m.begin(),
                                       stored_m.end());

                    const std::vector<double> operand =
                        apply_tau(b, d, ell + 1);
                    const std::vector<double> expected =
                        apply_tau(product, d, ell);
                    for (std::size_t e = 0; e < entries; e++)
                    {
                        operand_slots[ell][e * batch + m] = operand[e];
                        want[ell][e * batch + m] = expected[e];
                    }
                }
            }

            for (int baby = 2; baby <= d / 2; baby <<= 1)
            {
                const int giant = d / baby;
                auto key =
                    narrow_key(llama::Llama3Operator::pcmm_rotation_indices(
                        layout, giant, baby));

                for (int ell = 0; ell < kElls; ell++)
                {
                    SCOPED_TRACE("d=" + std::to_string(d) + " ell=" +
                                 std::to_string(ell) + " giant=" +
                                 std::to_string(giant) + " baby=" +
                                 std::to_string(baby));

                    heongpu::Ciphertext<S> cipher =
                        encrypt(operand_slots[ell]);
                    heongpu::Ciphertext<S> result(context);
                    ASSERT_NO_THROW(result = ops->pcmm(cipher, stored[ell],
                                                       layout, giant, baby,
                                                       ell, *key));
                    // The product sums d terms, so the error grows with d.
                    EXPECT_LT(max_error(decrypt(result), want[ell]), 1e-6 * d);
                }
            }
        }
    }

    /// RoPE below the top of the chain, on a key holding only its own shift.
    TEST_F(Llama3Env, RopeWorksBelowTheTopOfTheChain)
    {
        const int half = kRopeSwap;
        auto key = narrow_key({half});

        const std::vector<double> values = uniform(-1.0, 1.0, 501);
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int i = 0; i < slots; i++)
        {
            const double theta = 0.01 * ((i / (2 * half)) + 1) * ((i % half) + 1);
            cos_values[i] = std::cos(theta);
            sin_values[i] = ((i % (2 * half)) < half) ? -std::sin(theta)
                                                      : std::sin(theta);
        }

        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, cos_values, scale);
        encoder->encode(sin_plain, sin_values, scale);

        // Square first, so the plaintexts arrive three levels above the
        // ciphertext and must be dropped to meet it.
        heongpu::Ciphertext<S> cipher = encrypt(values);
        ops->square(cipher, *relin);
        ops->multiply_constant(cipher, 0.5);
        ASSERT_EQ(cipher.depth(), 2);

        heongpu::Ciphertext<S> result =
            ops->rope(cipher, cos_plain, sin_plain, half, *key);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            const double x = 0.5 * values[i] * values[i];
            const int j = (i + half) % slots;
            want[i] = x * cos_values[i] +
                      0.5 * values[j] * values[j] * sin_values[i];
        }
        EXPECT_LT(reported("rope at depth 2", decrypt(result), want), 1e-6);
    }

    /// A channel count that does not fill the last ciphertext.
    ///
    /// The guard admits this on purpose: padding contributes zero squares, so
    /// the mean is still over the real channels only.
    TEST_F(Llama3Env, RMSNormAcceptsAPaddedLastInput)
    {
        const int count = 32;
        const int stride = slots / count;
        const int real = 20; // channels carried by the second ciphertext

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count + real;
        config.eps = 1e-5;
        config.sum_lo = 20.0;
        config.sum_hi = 110.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(601);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<std::vector<double>> values(2,
                                                std::vector<double>(slots, 0.0));
        for (double& x : values[0])
        {
            x = dist(rng);
        }
        for (int j = 0; j < real; j++)
        {
            for (int i = 0; i < stride; i++)
            {
                values[1][j * stride + i] = dist(rng);
            }
        }

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values[0]),
                                               encrypt(values[1])};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> out =
            ops->rms_norm(in, no_weights, config, *galois, *relin);

        std::vector<double> want(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                for (int p = 0; p < 2; p++)
                {
                    const double v = values[p][j * stride + i];
                    total += v * v;
                }
            }
            const double factor =
                1.0 / std::sqrt(total / config.channels + config.eps);
            for (int j = 0; j < count; j++)
            {
                want[j * stride + i] = values[0][j * stride + i] * factor;
            }
        }
        EXPECT_LT(reported("rms_norm padded", decrypt(out[0]), want), 1e-6);
    }

    /// The mistakes that would otherwise pass silently.
    ///
    /// Each of these produces a plausible-looking ciphertext rather than an
    /// error: a wrong channel count rescales every output by a constant, a
    /// mismatched RoPE pair weights the two terms differently, and too short a
    /// chain reaches a kernel launch with a nonpositive extent.
    TEST_F(Llama3Env, WrongConfigurationIsRefusedRatherThanApproximated)
    {
        const int count = 32;
        const int stride = slots / count;
        const std::vector<double> values = uniform(-1.0, 1.0, 141);

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.eps = 1e-5;
        config.sum_lo = 8.0;
        config.sum_hi = 72.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::vector<heongpu::Ciphertext<S>> two{encrypt(values),
                                                encrypt(values)};
        std::vector<heongpu::Plaintext<S>> no_weights;

        // Forgetting that a second ciphertext doubles the channel count.
        config.channels = count;
        EXPECT_THROW(ops->rms_norm(two, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        // Counting channels that no input actually carries.
        config.channels = 3 * count;
        EXPECT_THROW(ops->rms_norm(two, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        config.channels = 2 * count;
        EXPECT_NO_THROW(ops->rms_norm(two, no_weights, config, *galois,
                                      *relin));

        // Inputs that have drifted apart in level.
        std::vector<heongpu::Ciphertext<S>> uneven{encrypt(values),
                                                   encrypt(values)};
        ops->multiply_constant(uneven[1], 1.0);
        EXPECT_THROW(ops->rms_norm(uneven, no_weights, config, *galois, *relin),
                     std::invalid_argument);

        // A RoPE pair encoded at two different scales.
        heongpu::Plaintext<S> cos_plain(context);
        heongpu::Plaintext<S> sin_plain(context);
        encoder->encode(cos_plain, std::vector<double>(slots, 1.0), scale);
        encoder->encode(sin_plain, std::vector<double>(slots, 1.0), scale / 2);
        heongpu::Ciphertext<S> cipher = encrypt(values);
        EXPECT_THROW(ops->rope(cipher, cos_plain, sin_plain, kRopeSwap,
                               *galois),
                     std::invalid_argument);

        // A polynomial that does not fit in what is left of the chain.
        heongpu::Ciphertext<S> shallow = encrypt(values);
        ops->drop_to_depth(shallow, kLimbs - 3);
        EXPECT_THROW(ops->silu(shallow, 11.0, 31, *relin),
                     std::invalid_argument);
    }

    /// Primitives back to back on one ciphertext.
    ///
    /// Each test above starts from a fresh encryption, which is exactly the
    /// state in which the plaintext-level bug was invisible. A layer never
    /// does that, so this runs a normalisation, a projection and an activation
    /// in sequence and checks the result against the same sequence on doubles.
    TEST_F(Llama3Env, PrimitivesComposeIntoOneChain)
    {
        const int d = 16;
        const llama::MatrixLayout layout(d, slots / (d * d));
        const int batch = layout.batch;
        const int count = 32;
        const int stride = slots / count;

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = stride;
        config.count = count;
        config.channels = count;
        config.eps = 1e-5;
        config.sum_lo = 8.0;
        config.sum_hi = 72.0;
        config.degree = 15;
        config.newton_iterations = 1;

        std::mt19937_64 rng(131);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        const std::vector<double> a = random_matrix(d, rng);
        const std::vector<double> stored = llama::permute_sigma(a, d);

        std::vector<heongpu::Ciphertext<S>> in{encrypt(values)};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normed =
            ops->rms_norm(in, no_weights, config, *galois, *relin);

        // The projection consumes tau(X), so the normalised block is read in
        // that layout; the chain is what matters here, not the packing.
        heongpu::Ciphertext<S> projected =
            ops->pcmm(normed[0], stored, layout, 4, 4, 0, *galois);
        // Each product is a sum of d terms of size about one, so the range is
        // several times wider than the normalised input it came from.
        heongpu::Ciphertext<S> activated =
            ops->silu(projected, 14.0, 31, *relin);

        // The same sequence on doubles.
        std::vector<double> normalised(slots);
        for (int i = 0; i < stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < count; j++)
            {
                const double v = values[j * stride + i];
                total += v * v;
            }
            const double factor =
                1.0 / std::sqrt(total / config.channels + config.eps);
            for (int j = 0; j < count; j++)
            {
                normalised[j * stride + i] = values[j * stride + i] * factor;
            }
        }

        std::vector<double> want(slots);
        for (int m = 0; m < batch; m++)
        {
            std::vector<double> block(static_cast<std::size_t>(d) * d);
            for (int e = 0; e < d * d; e++)
            {
                block[e] = normalised[e * batch + m];
            }
            // pcmm was handed tau(B) and returns A B, so undo tau to recover B.
            std::vector<double> b(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    b[((i + j) % d) * d + j] = block[i * d + j];
                }
            }
            const std::vector<double> product = llama::matmul_host(a, b, d);
            for (int e = 0; e < d * d; e++)
            {
                const double x = product[e];
                want[e * batch + m] = x / (1.0 + std::exp(-x));
            }
        }

        EXPECT_LT(reported("rms_norm -> pcmm -> silu", decrypt(activated), want),
                  1e-2);
    }

    // -----------------------------------------------------------------------
    // The SoftMax the paper actually specifies
    // -----------------------------------------------------------------------

    /// Section 4.3 fixes two normalise-and-square rounds, which does not fit
    /// the 24-limb chain the other tests share, so this one gets its own.
    ///
    /// Two rounds matter beyond costing more: only the second round runs on an
    /// input that already sums to one, which is the regime the round bounds
    /// switch to, and the reciprocal is approximated over a much wider range
    /// there than in the first round.
    class Llama3DeepEnv : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 8192;
        static constexpr int kLimbs = 32;
        static constexpr int kCount = 8;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();

            std::vector<int> shifts =
                llama::Llama3Operator::strided_rotation_indices(slots / kCount,
                                                                kCount);
            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }
    };

    TEST_F(Llama3DeepEnv, SoftmaxRunsTheTwoRoundsOfSectionFourPointThree)
    {
        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.count = kCount;
        config.stride = slots / kCount;
        config.bound = 2.0;
        config.iterations = 2;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        std::mt19937_64 rng(701);
        std::uniform_real_distribution<double> dist(-config.bound, 0.0);
        std::vector<double> values(slots);
        for (double& x : values)
        {
            x = dist(rng);
        }

        heongpu::Plaintext<S> plain(context);
        encoder->encode(plain, values, scale);
        heongpu::Ciphertext<S> cipher(context);
        encryptor->encrypt(cipher, plain);

        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, *galois, *relin);

        heongpu::Plaintext<S> out_plain(context);
        decryptor->decrypt(out_plain, result);
        std::vector<double> got;
        encoder->decode(got, out_plain);

        std::vector<double> want(slots);
        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += std::exp(values[j * config.stride + i]);
            }
            for (int j = 0; j < config.count; j++)
            {
                const int p = j * config.stride + i;
                want[p] = std::exp(values[p]) / total;
            }
        }

        EXPECT_LT(reported("softmax k=2", got, want), 1e-6);

        for (int i = 0; i < config.stride; i++)
        {
            double total = 0.0;
            for (int j = 0; j < config.count; j++)
            {
                total += got[j * config.stride + i];
            }
            EXPECT_NEAR(total, 1.0, 1e-3) << "instance " << i;
        }
    }

    // -----------------------------------------------------------------------
    // The sublayers
    // -----------------------------------------------------------------------

    /// Attention and the feed-forward network, on a chain long enough to hold
    /// one of them.
    ///
    /// Activations are held TRANSPOSED, X[channel][token]. That is what makes
    /// a projection W X, which is where Equation (5) wants the plaintext, and
    /// it puts the channel axis on the slow axis, where a reduction is exact
    /// and free. The matrices are small on purpose: these check that the
    /// sublayers compute what a transformer sublayer computes, not that a
    /// 4096-wide one fits, which it does not without bootstrapping.
    class Llama3LayerEnv : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 4096;
        // Attention under a pre-norm and a residual is the deepest thing here,
        // at about thirty levels.
        static constexpr int kLimbs = 38;
        static constexpr int kD = 8;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, 40);
        int slots = 0;
        llama::MatrixLayout layout;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), kLimbs - 1, 40);
            context->set_poly_modulus_degree(kDegree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();
            layout = llama::MatrixLayout(kD, slots / (kD * kD));

            // Exactly what attention advertises, and nothing more. The
            // feed-forward list and the strided reduction a pre-norm needs are
            // both subsets of it, so this key verifies all three: a shift left
            // out of any of those lists is undefined behaviour here, not a
            // missing-key error.
            llama::Llama3Operator::AttentionConfig index_config;
            index_config.layout = layout;
            index_config.rope = true;
            std::vector<int> shifts =
                llama::Llama3Operator::attention_rotation_indices(index_config);

            galois = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*galois, *secret);
            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);
        }

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& values)
        {
            heongpu::Plaintext<S> plain(context);
            encoder->encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor->encrypt(cipher, plain);
            return cipher;
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& cipher)
        {
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, cipher);
            std::vector<double> values;
            encoder->decode(values, plain);
            return values;
        }

        std::unique_ptr<heongpu::Galoiskey<S>> narrow_key(std::vector<int>
                                                              shifts)
        {
            auto key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*key, *secret);
            return key;
        }

        /// RMSNorm over the channel axis of the packed layout, which is the
        /// slow axis and therefore the free one.
        ///
        /// @p newton_iterations has no default on purpose. Dropping the Newton
        /// step caps the normalisation at the bare Chebyshev fit, around 1e-5,
        /// which is invisible inside a block test whose bound is looser than
        /// that and fatal in a standalone one whose bound is not. The pre-norm
        /// blocks pass 0 because they cannot spare the level; a caller with
        /// levels to spend must say 1 rather than inherit the compromise.
        llama::Llama3Operator::RMSNormConfig
        norm_config(const std::vector<double>& x,
                    int newton_iterations) const
        {
            llama::Llama3Operator::RMSNormConfig config;
            config.stride = layout.d * layout.batch;
            config.count = layout.d;
            config.channels = layout.d;
            config.eps = 1e-5;
            config.degree = 31;
            config.newton_iterations = newton_iterations;
            square_sum_range(x, config);
            return config;
        }

    };

    TEST_F(Llama3LayerEnv, TauPermutesEveryBlock)
    {
        auto key = narrow_key(
            llama::Llama3Operator::tau_rotation_indices(layout));

        std::mt19937_64 rng(810);
        const std::vector<std::vector<double>> blocks =
            random_blocks(layout, 1.0, rng);
        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(blocks, layout));

        heongpu::Ciphertext<S> result = ops->tau(cipher, layout, *key);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::permute_tau(blocks[m], layout.d);
        }
        EXPECT_LT(reported("tau", decrypt(result), pack_blocks(want, layout)),
                  1e-6);
    }

    TEST_F(Llama3LayerEnv, TransposeExchangesRowsAndColumns)
    {
        auto key = narrow_key(
            llama::Llama3Operator::transpose_rotation_indices(layout));

        std::mt19937_64 rng(811);
        const std::vector<std::vector<double>> blocks =
            random_blocks(layout, 1.0, rng);
        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(blocks, layout));

        heongpu::Ciphertext<S> result = ops->transpose(cipher, layout, *key);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::transpose_host(blocks[m], layout.d);
        }
        EXPECT_LT(
            reported("transpose", decrypt(result), pack_blocks(want, layout)),
            1e-6);
    }

    /// The product Section 4.2 does not cover, with both operands encrypted.
    TEST_F(Llama3LayerEnv, CcmmMultipliesTwoEncryptedBlocks)
    {
        auto key = narrow_key(
            llama::Llama3Operator::ccmm_rotation_indices(layout));

        std::mt19937_64 rng(812);
        const std::vector<std::vector<double>> a = random_blocks(layout, 1.0,
                                                                 rng);
        const std::vector<std::vector<double>> b = random_blocks(layout, 1.0,
                                                                 rng);

        heongpu::Ciphertext<S> ca = encrypt(pack_blocks(a, layout));
        heongpu::Ciphertext<S> cb = encrypt(pack_blocks(b, layout));

        // A scaling that rides on masks the algorithm encodes anyway, so it
        // has to come out exactly as if it had been applied afterwards.
        const double factor = 0.25;
        heongpu::Ciphertext<S> result =
            ops->ccmm(ca, cb, layout, factor, *key, *relin);

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            want[m] = llama::matmul_host(a[m], b[m], layout.d);
            for (double& v : want[m])
            {
                v *= factor;
            }
        }
        EXPECT_LT(reported("ccmm", decrypt(result), pack_blocks(want, layout)),
                  1e-6);
    }

    /// A residual connection joins two ciphertexts that share neither level
    /// nor scale, and CKKS addition needs both.
    TEST_F(Llama3LayerEnv, ResidualAddReconcilesLevelAndScale)
    {
        std::mt19937_64 rng(813);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> values(slots);
        for (double& v : values)
        {
            v = dist(rng);
        }

        heongpu::Ciphertext<S> skip = encrypt(values);
        heongpu::Ciphertext<S> sublayer = encrypt(values);
        ops->square(sublayer, *relin);

        ASSERT_NE(skip.depth(), sublayer.depth());
        ASSERT_NE(skip.scale(), sublayer.scale());

        heongpu::Ciphertext<S> out = ops->residual_add(skip, sublayer);

        std::vector<double> want(slots);
        for (int i = 0; i < slots; i++)
        {
            want[i] = values[i] + values[i] * values[i];
        }
        EXPECT_LT(reported("residual", decrypt(out), want), 1e-6);
    }

    TEST_F(Llama3LayerEnv, SoftmaxDropsTheKeysACausalMaskHides)
    {
        const int d = layout.d;
        auto key = narrow_key(llama::Llama3Operator::strided_rotation_indices(
            d * layout.batch, d));

        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.stride = d * layout.batch;
        config.count = d;
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        std::mt19937_64 rng(814);
        std::uniform_real_distribution<double> dist(-config.bound, 0.0);
        std::vector<std::vector<double>> scores(
            layout.batch, std::vector<double>(static_cast<std::size_t>(d) * d));
        for (auto& block : scores)
        {
            for (double& v : block)
            {
                v = dist(rng);
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(scores, layout));
        const std::vector<double> mask =
            llama::Llama3Operator::causal_mask(layout);
        heongpu::Ciphertext<S> result =
            ops->softmax(cipher, config, mask, *key, *relin);

        // Row index is the key and column index the query, so a query sums
        // over the keys at or before it.
        std::vector<std::vector<double>> want(
            layout.batch,
            std::vector<double>(static_cast<std::size_t>(d) * d, 0.0));
        for (int m = 0; m < layout.batch; m++)
        {
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int k = 0; k <= query; k++)
                {
                    total += std::exp(scores[m][k * d + query]);
                }
                for (int k = 0; k <= query; k++)
                {
                    want[m][k * d + query] =
                        std::exp(scores[m][k * d + query]) / total;
                }
            }
        }

        EXPECT_LT(reported("causal softmax", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-6);
    }

    TEST_F(Llama3LayerEnv, FeedForwardMatchesSwiGLU)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        auto key = narrow_key(
            llama::Llama3Operator::feed_forward_rotation_indices(config));

        std::mt19937_64 rng(815);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> gate =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> up =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> down =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::FeedForwardWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.gate.insert(weights.gate.end(), gate[m].begin(),
                                gate[m].end());
            weights.up.insert(weights.up.end(), up[m].begin(), up[m].end());
            weights.down.insert(weights.down.end(), down[m].begin(),
                                down[m].end());
        }

        std::vector<std::vector<double>> want(layout.batch);
        double widest = 0.0;
        for (int m = 0; m < layout.batch; m++)
        {
            const std::vector<double> g =
                llama::matmul_host(gate[m], x[m], d);
            const std::vector<double> u = llama::matmul_host(up[m], x[m], d);
            std::vector<double> hidden(g.size());
            for (std::size_t e = 0; e < g.size(); e++)
            {
                widest = std::max(widest, std::abs(g[e]));
                hidden[e] = silu_host(g[e]) * u[e];
            }
            want[m] = llama::matmul_host(down[m], hidden, d);
        }
        // The activation is a Chebyshev fit on a fixed interval, so a gate
        // outside it would be extrapolation rather than approximation.
        ASSERT_LT(widest, config.silu_bound);

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(x, layout));
        heongpu::Ciphertext<S> result =
            ops->feed_forward(cipher, weights, config, *key, *relin);

        // The only bound here that is not on the noise floor: the degree 31
        // SiLU fit dominates and its error is then summed over d terms by the
        // down projection.
        EXPECT_LT(reported("feed_forward", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-5);
    }

    /// The whole attention sublayer: projections, RoPE, scores, a causal
    /// SoftMax, the value product and the output projection.
    TEST_F(Llama3LayerEnv, AttentionMatchesTheReference)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(816);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> wq =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wk =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wv =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wo =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::AttentionWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.query.insert(weights.query.end(), wq[m].begin(),
                                 wq[m].end());
            weights.key.insert(weights.key.end(), wk[m].begin(), wk[m].end());
            weights.value.insert(weights.value.end(), wv[m].begin(),
                                 wv[m].end());
            weights.output.insert(weights.output.end(), wo[m].begin(),
                                  wo[m].end());
        }

        // RoPE, in slot space, exactly as the layer applies it: one rotation
        // by half the channel axis, with the sign of the sine term carried by
        // the plaintext.
        const int swap = llama::Llama3Operator::rope_swap_shift(layout);
        std::vector<double> cos_values(slots);
        std::vector<double> sin_values(slots);
        for (int r = 0; r < d; r++)
        {
            for (int c = 0; c < d; c++)
            {
                const double theta =
                    0.05 * (c + 1) * std::pow(2.0, -(r % (d / 2)));
                for (int m = 0; m < layout.batch; m++)
                {
                    const int p = (r * d + c) * layout.batch + m;
                    cos_values[p] = std::cos(theta);
                    sin_values[p] =
                        (r < d / 2) ? -std::sin(theta) : std::sin(theta);
                }
            }
        }
        std::vector<heongpu::Plaintext<S>> rope_plain;
        rope_plain.emplace_back(context);
        rope_plain.emplace_back(context);
        encoder->encode(rope_plain[0], cos_values, scale);
        encoder->encode(rope_plain[1], sin_values, scale);

        // The reference, one block at a time.
        std::vector<std::vector<double>> q(layout.batch);
        std::vector<std::vector<double>> k(layout.batch);
        std::vector<std::vector<double>> v(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            q[m] = llama::matmul_host(wq[m], x[m], d);
            k[m] = llama::matmul_host(wk[m], x[m], d);
            v[m] = llama::matmul_host(wv[m], x[m], d);
        }

        std::vector<double> q_slots = pack_blocks(q, layout);
        std::vector<double> k_slots = pack_blocks(k, layout);
        std::vector<double> q_roped(slots);
        std::vector<double> k_roped(slots);
        for (int p = 0; p < slots; p++)
        {
            const int shifted = (p + swap) % slots;
            q_roped[p] =
                q_slots[p] * cos_values[p] + q_slots[shifted] * sin_values[p];
            k_roped[p] =
                k_slots[p] * cos_values[p] + k_slots[shifted] * sin_values[p];
        }
        q = unpack_blocks(q_roped, layout);
        k = unpack_blocks(k_roped, layout);

        // Raw scores, then the calibration the paper takes offline: a scaling
        // that puts the spread at the SoftMax bound and a shift that puts the
        // top of the range at zero.
        std::vector<std::vector<double>> raw(layout.batch);
        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (int m = 0; m < layout.batch; m++)
        {
            raw[m] = llama::matmul_host(
                llama::transpose_host(k[m], d), q[m], d);
            for (double s : raw[m])
            {
                lowest = std::min(lowest, s);
                highest = std::max(highest, s);
            }
        }

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.causal = true;
        config.rope = true;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            std::vector<double> probabilities(
                static_cast<std::size_t>(d) * d, 0.0);
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int key = 0; key <= query; key++)
                {
                    total += std::exp(raw[m][key * d + query] *
                                          config.head_scale -
                                      config.score_shift);
                }
                for (int key = 0; key <= query; key++)
                {
                    probabilities[key * d + query] =
                        std::exp(raw[m][key * d + query] * config.head_scale -
                                 config.score_shift) /
                        total;
                }
            }
            const std::vector<double> out =
                llama::matmul_host(v[m], probabilities, d);
            want[m] = llama::matmul_host(wo[m], out, d);
        }

        heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(x, layout));
        heongpu::Ciphertext<S> result = ops->attention(
            cipher, weights, rope_plain, config, *galois, *relin);

        // Thirty levels deep and still on the noise floor: nothing in the
        // attention path is approximated over a wide interval, so the bound
        // belongs with the exact primitives rather than with SiLU.
        EXPECT_LT(reported("attention", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-6);
    }

    /// A pre-norm feed-forward block: RMSNorm, the sublayer, the residual.
    TEST_F(Llama3LayerEnv, FeedForwardRunsUnderPreNormAndResidual)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        std::mt19937_64 rng(817);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> gate =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> up =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> down =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::FeedForwardWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.gate.insert(weights.gate.end(), gate[m].begin(),
                                gate[m].end());
            weights.up.insert(weights.up.end(), up[m].begin(), up[m].end());
            weights.down.insert(weights.down.end(), down[m].begin(),
                                down[m].end());
        }

        const std::vector<double> x_slots = pack_blocks(x, layout);
        // 0 Newton steps: the sublayer that follows needs the level more
        // than this norm needs the last digit.
        const llama::Llama3Operator::RMSNormConfig norm =
            norm_config(x_slots, 0);
        const std::vector<std::vector<double>> normed =
            unpack_blocks(rms_norm_host(x_slots, norm), layout);

        std::vector<std::vector<double>> want(layout.batch);
        double widest = 0.0;
        for (int m = 0; m < layout.batch; m++)
        {
            const std::vector<double> g =
                llama::matmul_host(gate[m], normed[m], d);
            const std::vector<double> u =
                llama::matmul_host(up[m], normed[m], d);
            std::vector<double> hidden(g.size());
            for (std::size_t e = 0; e < g.size(); e++)
            {
                widest = std::max(widest, std::abs(g[e]));
                hidden[e] = silu_host(g[e]) * u[e];
            }
            const std::vector<double> out =
                llama::matmul_host(down[m], hidden, d);
            want[m].resize(out.size());
            for (std::size_t e = 0; e < out.size(); e++)
            {
                want[m][e] = x[m][e] + out[e];
            }
        }
        ASSERT_LT(widest, config.silu_bound);

        heongpu::Ciphertext<S> cipher = encrypt(x_slots);
        std::vector<heongpu::Ciphertext<S>> in{cipher};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normalised =
            ops->rms_norm(in, no_weights, norm, *galois, *relin);

        heongpu::Ciphertext<S> sublayer = ops->feed_forward(
            normalised[0], weights, config, *galois, *relin);
        heongpu::Ciphertext<S> result = ops->residual_add(cipher, sublayer);

        EXPECT_LT(reported("pre-norm ffn block", decrypt(result),
                           pack_blocks(want, layout)),
                  2e-4);
    }

    /// A pre-norm attention block, which is the deepest circuit here.
    ///
    /// No output projection and one normalise-and-square round: the point is
    /// that the sublayer survives arriving below the top of the chain and that
    /// the residual closes over it, and Sylph would bootstrap in the middle of
    /// this rather than stretch the chain.
    TEST_F(Llama3LayerEnv, AttentionRunsUnderPreNormAndResidual)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(818);
        const std::vector<std::vector<double>> x =
            random_blocks(layout, 1.0, rng);
        const std::vector<std::vector<double>> wq =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wk =
            random_blocks(layout, weight_scale, rng);
        const std::vector<std::vector<double>> wv =
            random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::AttentionWeights weights;
        for (int m = 0; m < layout.batch; m++)
        {
            weights.query.insert(weights.query.end(), wq[m].begin(),
                                 wq[m].end());
            weights.key.insert(weights.key.end(), wk[m].begin(), wk[m].end());
            weights.value.insert(weights.value.end(), wv[m].begin(),
                                 wv[m].end());
        }

        const std::vector<double> x_slots = pack_blocks(x, layout);
        // 0 Newton steps: the sublayer that follows needs the level more
        // than this norm needs the last digit.
        const llama::Llama3Operator::RMSNormConfig norm =
            norm_config(x_slots, 0);
        const std::vector<std::vector<double>> normed =
            unpack_blocks(rms_norm_host(x_slots, norm), layout);

        std::vector<std::vector<double>> q(layout.batch);
        std::vector<std::vector<double>> k(layout.batch);
        std::vector<std::vector<double>> v(layout.batch);
        std::vector<std::vector<double>> raw(layout.batch);
        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (int m = 0; m < layout.batch; m++)
        {
            q[m] = llama::matmul_host(wq[m], normed[m], d);
            k[m] = llama::matmul_host(wk[m], normed[m], d);
            v[m] = llama::matmul_host(wv[m], normed[m], d);
            raw[m] = llama::matmul_host(llama::transpose_host(k[m], d), q[m],
                                        d);
            for (double s : raw[m])
            {
                lowest = std::min(lowest, s);
                highest = std::max(highest, s);
            }
        }

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.causal = true;
        config.rope = false;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 1;

        std::vector<std::vector<double>> want(layout.batch);
        for (int m = 0; m < layout.batch; m++)
        {
            std::vector<double> probabilities(
                static_cast<std::size_t>(d) * d, 0.0);
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int key = 0; key <= query; key++)
                {
                    total += std::exp(raw[m][key * d + query] *
                                          config.head_scale -
                                      config.score_shift);
                }
                for (int key = 0; key <= query; key++)
                {
                    probabilities[key * d + query] =
                        std::exp(raw[m][key * d + query] * config.head_scale -
                                 config.score_shift) /
                        total;
                }
            }
            const std::vector<double> out =
                llama::matmul_host(v[m], probabilities, d);
            want[m].resize(out.size());
            for (std::size_t e = 0; e < out.size(); e++)
            {
                want[m][e] = x[m][e] + out[e];
            }
        }

        heongpu::Ciphertext<S> cipher = encrypt(x_slots);
        std::vector<heongpu::Ciphertext<S>> in{cipher};
        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> normalised =
            ops->rms_norm(in, no_weights, norm, *galois, *relin);

        std::vector<heongpu::Plaintext<S>> no_rope;
        heongpu::Ciphertext<S> sublayer = ops->attention(
            normalised[0], weights, no_rope, config, *galois, *relin);
        heongpu::Ciphertext<S> result = ops->residual_add(cipher, sublayer);

        EXPECT_LT(reported("pre-norm attention block", decrypt(result),
                           pack_blocks(want, layout)),
                  1e-4);
    }

    /// A projection of a model wider than one ciphertext. The sum over the
    /// input channel blocks is the whole of what makes width possible, and it
    /// has to be exact: every term leaves Equation (5) at one level and one
    /// scale, so nothing here may cost depth.
    TEST_F(Llama3LayerEnv, ProjectBlocksSumsTheChannelBlocks)
    {
        const int in_blocks = 3;
        const int out_blocks = 2;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(
                                              layout.d * in_blocks));

        llama::Llama3Operator::FeedForwardConfig index_config;
        index_config.layout = layout;
        auto key = narrow_key(
            llama::Llama3Operator::feed_forward_rotation_indices(index_config));

        std::mt19937_64 rng(901);
        std::vector<ChannelBlock> x;
        for (int j = 0; j < in_blocks; j++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        llama::BlockMatrix weight(out_blocks, in_blocks);
        std::vector<std::vector<ChannelBlock>> w(out_blocks);
        for (int i = 0; i < out_blocks; i++)
        {
            w[i].resize(in_blocks);
            for (int j = 0; j < in_blocks; j++)
            {
                // One block is left out entirely. A zero block is skipped
                // rather than encoded, so the answer has to account for a
                // weight that was never handed over.
                if (i == 1 && j == 0)
                {
                    w[i][j] = zero_block(layout);
                    continue;
                }
                w[i][j] = random_blocks(layout, weight_scale, rng);
                weight.at(i, j) = flatten_block(w[i][j]);
            }
        }

        const std::vector<ChannelBlock> want =
            block_project_host(w, x, layout);

        std::vector<heongpu::Ciphertext<S>> tau_x;
        for (int j = 0; j < in_blocks; j++)
        {
            heongpu::Ciphertext<S> cipher = encrypt(pack_blocks(x[j], layout));
            tau_x.push_back(ops->tau(cipher, layout, *key));
        }

        std::vector<heongpu::Ciphertext<S>> got = ops->project_blocks(
            tau_x, weight, layout, 0, 0, "block projection", *key);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(out_blocks));

        for (int i = 0; i < out_blocks; i++)
        {
            const std::string label =
                "project_blocks[" + std::to_string(i) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[i]),
                               pack_blocks(want[i], layout)),
                      1e-6);
        }
    }

    /// Attention with a head two channel blocks wide: the projections sum over
    /// the input blocks, the scores sum over the head's blocks, and RoPE pairs
    /// a channel with one in the other block, which needs no rotation at all.
    TEST_F(Llama3LayerEnv, AttentionSpansSeveralChannelBlocks)
    {
        const int d = layout.d;
        const int blocks = 2;
        const int entries = d * d;
        const int half = blocks / 2;
        const double weight_scale =
            1.0 / std::sqrt(static_cast<double>(d * blocks));

        std::mt19937_64 rng(902);
        std::vector<ChannelBlock> x;
        for (int j = 0; j < blocks; j++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        llama::Llama3Operator::BlockAttentionWeights weights;
        weights.query = llama::BlockMatrix(blocks, blocks);
        weights.key = llama::BlockMatrix(blocks, blocks);
        weights.value = llama::BlockMatrix(blocks, blocks);
        weights.output = llama::BlockMatrix(blocks, blocks);

        std::vector<std::vector<ChannelBlock>> wq(blocks), wk(blocks),
            wv(blocks), wo(blocks);
        for (int i = 0; i < blocks; i++)
        {
            wq[i].resize(blocks);
            wk[i].resize(blocks);
            wv[i].resize(blocks);
            wo[i].resize(blocks);
            for (int j = 0; j < blocks; j++)
            {
                wq[i][j] = random_blocks(layout, weight_scale, rng);
                wk[i][j] = random_blocks(layout, weight_scale, rng);
                wv[i][j] = random_blocks(layout, weight_scale, rng);
                wo[i][j] = random_blocks(layout, weight_scale, rng);
                weights.query.at(i, j) = flatten_block(wq[i][j]);
                weights.key.at(i, j) = flatten_block(wk[i][j]);
                weights.value.at(i, j) = flatten_block(wv[i][j]);
                weights.output.at(i, j) = flatten_block(wo[i][j]);
            }
        }

        // RoPE pairs channel c with c + head_dim / 2. The head is two blocks
        // wide, so that partner is in the other block: block 0 carries the
        // first half of the head and takes the negative sine, block 1 the
        // second half and the positive one. The angles depend on the channel
        // inside the head and on the token, so both heads would read these.
        std::vector<std::vector<double>> cos_values(blocks,
                                                    std::vector<double>(entries));
        std::vector<std::vector<double>> sin_values(blocks,
                                                    std::vector<double>(entries));
        for (int t = 0; t < blocks; t++)
        {
            for (int r = 0; r < d; r++)
            {
                for (int j = 0; j < d; j++)
                {
                    const double theta =
                        0.05 * (j + 1) * std::pow(2.0, -r);
                    cos_values[t][r * d + j] = std::cos(theta);
                    sin_values[t][r * d + j] =
                        (t < half) ? -std::sin(theta) : std::sin(theta);
                }
            }
        }

        std::vector<std::vector<double>> rope_slots;
        for (int t = 0; t < blocks; t++)
        {
            std::vector<double> cos_slots(slots);
            std::vector<double> sin_slots(slots);
            for (int e = 0; e < entries; e++)
            {
                for (int m = 0; m < layout.batch; m++)
                {
                    cos_slots[e * layout.batch + m] = cos_values[t][e];
                    sin_slots[e * layout.batch + m] = sin_values[t][e];
                }
            }
            rope_slots.push_back(cos_slots);
            rope_slots.push_back(sin_slots);
        }

        std::vector<heongpu::Plaintext<S>> rope_plain;
        for (std::size_t i = 0; i < rope_slots.size(); i++)
        {
            rope_plain.emplace_back(context);
        }
        for (std::size_t i = 0; i < rope_slots.size(); i++)
        {
            encoder->encode(rope_plain[i], rope_slots[i], scale);
        }

        // The reference, block by block.
        const std::vector<ChannelBlock> q = block_project_host(wq, x, layout);
        const std::vector<ChannelBlock> k = block_project_host(wk, x, layout);
        const std::vector<ChannelBlock> v = block_project_host(wv, x, layout);

        std::vector<ChannelBlock> q_roped(blocks), k_roped(blocks);
        for (int t = 0; t < blocks; t++)
        {
            const int partner = (t < half) ? t + half : t - half;
            q_roped[t] = zero_block(layout);
            k_roped[t] = zero_block(layout);
            for (int m = 0; m < layout.batch; m++)
            {
                for (int e = 0; e < entries; e++)
                {
                    q_roped[t][m][e] = q[t][m][e] * cos_values[t][e] +
                                       q[partner][m][e] * sin_values[t][e];
                    k_roped[t][m][e] = k[t][m][e] * cos_values[t][e] +
                                       k[partner][m][e] * sin_values[t][e];
                }
            }
        }

        // A head spanning several blocks sums their scores, which is exactly
        // the channel sum K^T Q performs when K and Q are stacked.
        ChannelBlock raw = zero_block(layout);
        for (int t = 0; t < blocks; t++)
        {
            ChannelBlock term(layout.batch);
            for (int m = 0; m < layout.batch; m++)
            {
                term[m] = llama::matmul_host(
                    llama::transpose_host(k_roped[t][m], d), q_roped[t][m], d);
            }
            add_into(raw, term);
        }

        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (const auto& block : raw)
        {
            for (double s : block)
            {
                lowest = std::min(lowest, s);
                highest = std::max(highest, s);
            }
        }

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.heads = 1;
        config.causal = true;
        config.rope = true;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        const ChannelBlock probabilities = causal_softmax_host(
            raw, config.head_scale, config.score_shift, d);

        std::vector<ChannelBlock> weighted(blocks);
        for (int t = 0; t < blocks; t++)
        {
            weighted[t] = matmul_block(v[t], probabilities, d);
        }
        const std::vector<ChannelBlock> want =
            block_project_host(wo, weighted, layout);

        std::vector<heongpu::Ciphertext<S>> in;
        for (int j = 0; j < blocks; j++)
        {
            in.push_back(encrypt(pack_blocks(x[j], layout)));
        }

        std::vector<heongpu::Ciphertext<S>> got = ops->attention(
            in, weights, rope_plain, config, *galois, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(blocks));

        for (int t = 0; t < blocks; t++)
        {
            const std::string label =
                "wide attention[" + std::to_string(t) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[t]),
                               pack_blocks(want[t], layout)),
                      1e-6);
        }
    }

    /// Two heads, one channel block each. Nothing may cross between them: the
    /// blocks carry different data, so a head reading the wrong one is a large
    /// error rather than a subtle one.
    TEST_F(Llama3LayerEnv, AttentionKeepsItsHeadsApart)
    {
        const int d = layout.d;
        const int heads = 2;

        std::mt19937_64 rng(903);
        std::vector<ChannelBlock> x;
        for (int h = 0; h < heads; h++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        std::vector<ChannelBlock> raw(heads);
        double lowest = std::numeric_limits<double>::max();
        double highest = std::numeric_limits<double>::lowest();
        for (int h = 0; h < heads; h++)
        {
            raw[h].resize(layout.batch);
            for (int m = 0; m < layout.batch; m++)
            {
                raw[h][m] = llama::matmul_host(
                    llama::transpose_host(x[h][m], d), x[h][m], d);
                for (double s : raw[h][m])
                {
                    lowest = std::min(lowest, s);
                    highest = std::max(highest, s);
                }
            }
        }

        // No projections at all, so the input blocks are the query, the key
        // and the value, and head h sees only block h.
        llama::Llama3Operator::BlockAttentionWeights weights;

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.heads = heads;
        config.causal = true;
        config.rope = false;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        std::vector<ChannelBlock> want(heads);
        for (int h = 0; h < heads; h++)
        {
            const ChannelBlock probabilities = causal_softmax_host(
                raw[h], config.head_scale, config.score_shift, d);
            want[h] = matmul_block(x[h], probabilities, d);
        }

        std::vector<heongpu::Ciphertext<S>> in;
        for (int h = 0; h < heads; h++)
        {
            in.push_back(encrypt(pack_blocks(x[h], layout)));
        }

        std::vector<heongpu::Plaintext<S>> no_rope;
        std::vector<heongpu::Ciphertext<S>> got =
            ops->attention(in, weights, no_rope, config, *galois, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(heads));

        for (int h = 0; h < heads; h++)
        {
            const std::string label = "head " + std::to_string(h);
            EXPECT_LT(reported(label.c_str(), decrypt(got[h]),
                               pack_blocks(want[h], layout)),
                      1e-6);
        }
    }

    /// SwiGLU widening past the input width and contracting back, which is
    /// what Llama-3's feed-forward does by a factor of three and a half.
    TEST_F(Llama3LayerEnv, FeedForwardWidensAndContractsChannelBlocks)
    {
        const int d = layout.d;
        const int in_blocks = 2;
        const int hidden_blocks = 3;
        const double weight_scale =
            1.0 / std::sqrt(static_cast<double>(d * in_blocks));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        auto key = narrow_key(
            llama::Llama3Operator::feed_forward_rotation_indices(config));

        std::mt19937_64 rng(904);
        std::vector<ChannelBlock> x;
        for (int j = 0; j < in_blocks; j++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        llama::Llama3Operator::BlockFeedForwardWeights weights;
        weights.gate = llama::BlockMatrix(hidden_blocks, in_blocks);
        weights.up = llama::BlockMatrix(hidden_blocks, in_blocks);
        weights.down = llama::BlockMatrix(in_blocks, hidden_blocks);

        std::vector<std::vector<ChannelBlock>> wg(hidden_blocks),
            wu(hidden_blocks), wd(in_blocks);
        for (int i = 0; i < hidden_blocks; i++)
        {
            wg[i].resize(in_blocks);
            wu[i].resize(in_blocks);
            for (int j = 0; j < in_blocks; j++)
            {
                wg[i][j] = random_blocks(layout, weight_scale, rng);
                wu[i][j] = random_blocks(layout, weight_scale, rng);
                weights.gate.at(i, j) = flatten_block(wg[i][j]);
                weights.up.at(i, j) = flatten_block(wu[i][j]);
            }
        }
        const double down_scale =
            1.0 / std::sqrt(static_cast<double>(d * hidden_blocks));
        for (int k = 0; k < in_blocks; k++)
        {
            wd[k].resize(hidden_blocks);
            for (int i = 0; i < hidden_blocks; i++)
            {
                wd[k][i] = random_blocks(layout, down_scale, rng);
                weights.down.at(k, i) = flatten_block(wd[k][i]);
            }
        }

        const std::vector<ChannelBlock> gate =
            block_project_host(wg, x, layout);
        const std::vector<ChannelBlock> up = block_project_host(wu, x, layout);

        std::vector<ChannelBlock> hidden(hidden_blocks);
        double widest = 0.0;
        for (int i = 0; i < hidden_blocks; i++)
        {
            hidden[i] = zero_block(layout);
            for (int m = 0; m < layout.batch; m++)
            {
                for (std::size_t e = 0; e < gate[i][m].size(); e++)
                {
                    widest = std::max(widest, std::abs(gate[i][m][e]));
                    hidden[i][m][e] =
                        silu_host(gate[i][m][e]) * up[i][m][e];
                }
            }
        }
        // The activation is a Chebyshev fit on a fixed interval, so a gate
        // outside it would be extrapolation rather than approximation.
        ASSERT_LT(widest, config.silu_bound);

        const std::vector<ChannelBlock> want =
            block_project_host(wd, hidden, layout);

        std::vector<heongpu::Ciphertext<S>> in;
        for (int j = 0; j < in_blocks; j++)
        {
            in.push_back(encrypt(pack_blocks(x[j], layout)));
        }

        std::vector<heongpu::Ciphertext<S>> got =
            ops->feed_forward(in, weights, config, *key, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(in_blocks));

        for (int k = 0; k < in_blocks; k++)
        {
            const std::string label = "wide feed_forward[" +
                                      std::to_string(k) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[k]),
                               pack_blocks(want[k], layout)),
                      1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // Sequences longer than one token block
    // -----------------------------------------------------------------------

    /// The mask is what tells a cut sequence which score blocks exist, so its
    /// three cases are worth pinning without a GPU in the way.
    TEST(CKKS_Llama3, CausalMaskCutsTheSequenceIntoBlocks)
    {
        const int d = 4;
        const int batch = 2;
        const int token_blocks = 3;
        const llama::MatrixLayout layout(d, batch);

        // A key block ahead of the query block survives nowhere, and says so
        // by coming back empty rather than as a slot vector of zeros.
        EXPECT_TRUE(
            llama::Llama3Operator::causal_block_mask(layout, 0, 1, token_blocks)
                .empty());
        EXPECT_TRUE(
            llama::Llama3Operator::causal_block_mask(layout, 1, 2, token_blocks)
                .empty());

        // One token block of sequence is the mask attention already had.
        EXPECT_EQ(llama::Llama3Operator::causal_block_mask(layout, 0, 0, 1),
                  llama::Llama3Operator::causal_mask(layout));

        for (int query_block = 0; query_block < token_blocks; query_block++)
        {
            const double visible = static_cast<double>(d) * (query_block + 1);

            for (int key_block = 0; key_block <= query_block; key_block++)
            {
                const std::vector<double> mask =
                    llama::Llama3Operator::causal_block_mask(
                        layout, query_block, key_block, token_blocks);
                ASSERT_EQ(mask.size(), static_cast<std::size_t>(layout.slots));

                for (int query = 0; query < d; query++)
                {
                    const int kept = query_block * d + query + 1;
                    const double weight =
                        std::sqrt(visible / static_cast<double>(kept));

                    for (int key = 0; key < d; key++)
                    {
                        // Behind the diagonal every key is already in the
                        // past, so only the diagonal block is a triangle.
                        const bool admitted =
                            key_block * d + key <= query_block * d + query;
                        for (int m = 0; m < batch; m++)
                        {
                            EXPECT_DOUBLE_EQ(
                                mask[(key * d + query) * batch + m],
                                admitted ? weight : 0.0);
                        }
                    }
                }
            }
        }
    }

    /// The denominator of a query has to cover keys held in another
    /// ciphertext, which is what makes a sequence longer than d possible.
    TEST_F(Llama3LayerEnv, SoftmaxSumsAcrossCiphertexts)
    {
        const int d = layout.d;
        const int parts = 2;
        auto key = narrow_key(llama::Llama3Operator::strided_rotation_indices(
            d * layout.batch, d));

        llama::Llama3Operator::SoftmaxConfig config;
        config.strided = true;
        config.stride = d * layout.batch;
        config.count = d;
        config.bound = 2.0;
        config.iterations = 1;
        config.exp_degree = 15;
        config.inverse_degree = 15;
        config.inverse_newton = 2;

        std::mt19937_64 rng(905);
        std::uniform_real_distribution<double> dist(-config.bound, 0.0);
        std::vector<ChannelBlock> scores(parts);
        std::vector<heongpu::Ciphertext<S>> in;
        for (int p = 0; p < parts; p++)
        {
            scores[p] = zero_block(layout);
            for (auto& block : scores[p])
            {
                for (double& v : block)
                {
                    v = dist(rng);
                }
            }
            in.push_back(encrypt(pack_blocks(scores[p], layout)));
        }

        std::vector<std::vector<double>> no_masks;
        std::vector<heongpu::Ciphertext<S>> got =
            ops->softmax(in, config, no_masks, *key, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(parts));

        // Every part holds a stretch of one axis of length count, so the
        // SoftMax is over parts * count coordinates.
        std::vector<ChannelBlock> want(parts, zero_block(layout));
        for (int m = 0; m < layout.batch; m++)
        {
            for (int query = 0; query < d; query++)
            {
                double total = 0.0;
                for (int p = 0; p < parts; p++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        total += std::exp(scores[p][m][k * d + query]);
                    }
                }
                for (int p = 0; p < parts; p++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        want[p][m][k * d + query] =
                            std::exp(scores[p][m][k * d + query]) / total;
                    }
                }
            }
        }

        for (int p = 0; p < parts; p++)
        {
            const std::string label =
                "split softmax[" + std::to_string(p) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[p]),
                               pack_blocks(want[p], layout)),
                      1e-6);
        }
    }

    /// A prompt longer than d, which makes the scores a grid: two query
    /// blocks, and the second attends back into the first.
    TEST_F(Llama3LayerEnv, AttentionSpansSeveralTokenBlocks)
    {
        const int d = layout.d;
        const int token_blocks = 2;
        const int entries = d * d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(906);
        std::vector<ChannelBlock> x;
        for (int s = 0; s < token_blocks; s++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        const ChannelBlock wq = random_blocks(layout, weight_scale, rng);
        const ChannelBlock wk = random_blocks(layout, weight_scale, rng);
        const ChannelBlock wv = random_blocks(layout, weight_scale, rng);
        const ChannelBlock wo = random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::BlockAttentionWeights weights;
        weights.query = llama::BlockMatrix(flatten_block(wq));
        weights.key = llama::BlockMatrix(flatten_block(wk));
        weights.value = llama::BlockMatrix(flatten_block(wv));
        weights.output = llama::BlockMatrix(flatten_block(wo));

        // The head is one block wide, so RoPE is the half-swap inside it. The
        // angle carries the ABSOLUTE token position, so the second token block
        // gets its own pair of plaintexts and they are what pins the indexing.
        std::vector<std::vector<double>> cos_values(
            token_blocks, std::vector<double>(entries));
        std::vector<std::vector<double>> sin_values(
            token_blocks, std::vector<double>(entries));
        for (int s = 0; s < token_blocks; s++)
        {
            for (int r = 0; r < d; r++)
            {
                for (int t = 0; t < d; t++)
                {
                    const double position = s * d + t;
                    const double theta =
                        0.05 * position * std::pow(2.0, -(r % (d / 2)));
                    cos_values[s][r * d + t] = std::cos(theta);
                    sin_values[s][r * d + t] =
                        (r < d / 2) ? -std::sin(theta) : std::sin(theta);
                }
            }
        }

        std::vector<std::vector<double>> rope_slots;
        for (int s = 0; s < token_blocks; s++)
        {
            std::vector<double> cos_slots(slots);
            std::vector<double> sin_slots(slots);
            for (int e = 0; e < entries; e++)
            {
                for (int m = 0; m < layout.batch; m++)
                {
                    cos_slots[e * layout.batch + m] = cos_values[s][e];
                    sin_slots[e * layout.batch + m] = sin_values[s][e];
                }
            }
            rope_slots.push_back(cos_slots);
            rope_slots.push_back(sin_slots);
        }

        std::vector<heongpu::Plaintext<S>> rope_plain;
        for (std::size_t i = 0; i < rope_slots.size(); i++)
        {
            rope_plain.emplace_back(context);
        }
        for (std::size_t i = 0; i < rope_slots.size(); i++)
        {
            encoder->encode(rope_plain[i], rope_slots[i], scale);
        }

        // The reference, one token block at a time.
        std::vector<ChannelBlock> q(token_blocks), k(token_blocks),
            v(token_blocks);
        for (int s = 0; s < token_blocks; s++)
        {
            q[s] = matmul_block(wq, x[s], d);
            k[s] = matmul_block(wk, x[s], d);
            v[s] = matmul_block(wv, x[s], d);
        }

        std::vector<ChannelBlock> q_roped(token_blocks), k_roped(token_blocks);
        for (int s = 0; s < token_blocks; s++)
        {
            q_roped[s] = zero_block(layout);
            k_roped[s] = zero_block(layout);
            for (int m = 0; m < layout.batch; m++)
            {
                for (int r = 0; r < d; r++)
                {
                    // The swap is a row rotation by half the matrix, so the
                    // partner of channel r is channel (r + d / 2) mod d.
                    const int partner = (r + d / 2) % d;
                    for (int t = 0; t < d; t++)
                    {
                        const int at = r * d + t;
                        const int from = partner * d + t;
                        q_roped[s][m][at] = q[s][m][at] * cos_values[s][at] +
                                            q[s][m][from] * sin_values[s][at];
                        k_roped[s][m][at] = k[s][m][at] * cos_values[s][at] +
                                            k[s][m][from] * sin_values[s][at];
                    }
                }
            }
        }

        // The score grid, upper half only: key block u is formed against query
        // block s exactly when u <= s.
        std::vector<std::vector<ChannelBlock>> raw(token_blocks);
        std::vector<ChannelBlock> formed;
        for (int s = 0; s < token_blocks; s++)
        {
            for (int u = 0; u <= s; u++)
            {
                raw[s].push_back(scores_host(k_roped[u], q_roped[s], d));
                formed.push_back(raw[s].back());
            }
        }

        double lowest = 0.0;
        double highest = 0.0;
        score_range(formed, lowest, highest);

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.heads = 1;
        config.token_blocks = token_blocks;
        config.causal = true;
        config.rope = true;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        std::vector<ChannelBlock> want(token_blocks);
        for (int s = 0; s < token_blocks; s++)
        {
            const std::vector<ChannelBlock> probabilities =
                causal_block_softmax_host(raw[s], config.head_scale,
                                          config.score_shift, d, s);

            // The value product sums over the key blocks the same way.
            ChannelBlock weighted = zero_block(layout);
            for (int u = 0; u <= s; u++)
            {
                add_into(weighted, matmul_block(v[u], probabilities[u], d));
            }
            want[s] = matmul_block(wo, weighted, d);
        }

        std::vector<heongpu::Ciphertext<S>> in;
        for (int s = 0; s < token_blocks; s++)
        {
            in.push_back(encrypt(pack_blocks(x[s], layout)));
        }

        std::vector<heongpu::Ciphertext<S>> got =
            ops->attention(in, weights, rope_plain, config, *galois, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(token_blocks));

        for (int s = 0; s < token_blocks; s++)
        {
            const std::string label =
                "long attention[" + std::to_string(s) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[s]),
                               pack_blocks(want[s], layout)),
                      1e-6);
        }
    }

    /// Grouped-query attention: two query heads reading one key and value
    /// head, which is the shape Llama-3-8B actually has.
    TEST_F(Llama3LayerEnv, AttentionSharesKeyValueHeads)
    {
        const int d = layout.d;
        const int in_blocks = 2;
        const int heads = 2;
        const double weight_scale =
            1.0 / std::sqrt(static_cast<double>(d * in_blocks));

        std::mt19937_64 rng(907);
        std::vector<ChannelBlock> x;
        for (int j = 0; j < in_blocks; j++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        // The query weight is as wide as the model; the key and value weights
        // are half of it, which is the whole saving grouped-query buys.
        llama::Llama3Operator::BlockAttentionWeights weights;
        weights.query = llama::BlockMatrix(heads, in_blocks);
        weights.key = llama::BlockMatrix(1, in_blocks);
        weights.value = llama::BlockMatrix(1, in_blocks);
        weights.output = llama::BlockMatrix(in_blocks, heads);

        std::vector<std::vector<ChannelBlock>> wq(heads), wk(1), wv(1),
            wo(in_blocks);
        wk[0].resize(in_blocks);
        wv[0].resize(in_blocks);
        for (int j = 0; j < in_blocks; j++)
        {
            wk[0][j] = random_blocks(layout, weight_scale, rng);
            wv[0][j] = random_blocks(layout, weight_scale, rng);
            weights.key.at(0, j) = flatten_block(wk[0][j]);
            weights.value.at(0, j) = flatten_block(wv[0][j]);
        }
        for (int h = 0; h < heads; h++)
        {
            wq[h].resize(in_blocks);
            for (int j = 0; j < in_blocks; j++)
            {
                wq[h][j] = random_blocks(layout, weight_scale, rng);
                weights.query.at(h, j) = flatten_block(wq[h][j]);
            }
        }
        for (int i = 0; i < in_blocks; i++)
        {
            wo[i].resize(heads);
            for (int h = 0; h < heads; h++)
            {
                wo[i][h] = random_blocks(layout, weight_scale, rng);
                weights.output.at(i, h) = flatten_block(wo[i][h]);
            }
        }

        const std::vector<ChannelBlock> q = block_project_host(wq, x, layout);
        const std::vector<ChannelBlock> k = block_project_host(wk, x, layout);
        const std::vector<ChannelBlock> v = block_project_host(wv, x, layout);

        // Both heads read k[0] and v[0]; only the query differs, which is
        // exactly what makes the two outputs differ.
        std::vector<ChannelBlock> raw(heads);
        for (int h = 0; h < heads; h++)
        {
            raw[h] = scores_host(k[0], q[h], d);
        }

        double lowest = 0.0;
        double highest = 0.0;
        score_range(raw, lowest, highest);

        llama::Llama3Operator::AttentionConfig config;
        config.layout = layout;
        config.heads = heads;
        config.kv_heads = 1;
        config.causal = true;
        config.rope = false;
        config.head_scale = 2.0 / (highest - lowest);
        config.score_shift = highest * config.head_scale;
        config.softmax.bound = 2.0;
        config.softmax.iterations = 1;
        config.softmax.exp_degree = 15;
        config.softmax.inverse_degree = 15;
        config.softmax.inverse_newton = 2;

        std::vector<ChannelBlock> weighted(heads);
        for (int h = 0; h < heads; h++)
        {
            const ChannelBlock probabilities = causal_softmax_host(
                raw[h], config.head_scale, config.score_shift, d);
            weighted[h] = matmul_block(v[0], probabilities, d);
        }
        const std::vector<ChannelBlock> want =
            block_project_host(wo, weighted, layout);

        std::vector<heongpu::Ciphertext<S>> in;
        for (int j = 0; j < in_blocks; j++)
        {
            in.push_back(encrypt(pack_blocks(x[j], layout)));
        }

        std::vector<heongpu::Plaintext<S>> no_rope;
        std::vector<heongpu::Ciphertext<S>> got =
            ops->attention(in, weights, no_rope, config, *galois, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(in_blocks));

        for (int i = 0; i < in_blocks; i++)
        {
            const std::string label =
                "grouped attention[" + std::to_string(i) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[i]),
                               pack_blocks(want[i], layout)),
                      1e-6);
        }
    }

    /// SwiGLU never mixes tokens, so a cut sequence is only an indexing
    /// question -- which is worth a test precisely because it is silent when
    /// it is wrong.
    TEST_F(Llama3LayerEnv, FeedForwardRunsOverTokenBlocks)
    {
        const int d = layout.d;
        const int token_blocks = 2;
        const int hidden_blocks = 2;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        llama::Llama3Operator::FeedForwardConfig config;
        config.layout = layout;
        config.token_blocks = token_blocks;
        config.silu_bound = 4.0;
        config.silu_degree = 31;

        auto key = narrow_key(
            llama::Llama3Operator::feed_forward_rotation_indices(config));

        std::mt19937_64 rng(908);
        std::vector<ChannelBlock> x;
        for (int s = 0; s < token_blocks; s++)
        {
            x.push_back(random_blocks(layout, 1.0, rng));
        }

        llama::Llama3Operator::BlockFeedForwardWeights weights;
        weights.gate = llama::BlockMatrix(hidden_blocks, 1);
        weights.up = llama::BlockMatrix(hidden_blocks, 1);
        weights.down = llama::BlockMatrix(1, hidden_blocks);

        std::vector<ChannelBlock> wg(hidden_blocks), wu(hidden_blocks),
            wd(hidden_blocks);
        const double down_scale =
            1.0 / std::sqrt(static_cast<double>(d * hidden_blocks));
        for (int i = 0; i < hidden_blocks; i++)
        {
            wg[i] = random_blocks(layout, weight_scale, rng);
            wu[i] = random_blocks(layout, weight_scale, rng);
            wd[i] = random_blocks(layout, down_scale, rng);
            weights.gate.at(i, 0) = flatten_block(wg[i]);
            weights.up.at(i, 0) = flatten_block(wu[i]);
            weights.down.at(0, i) = flatten_block(wd[i]);
        }

        std::vector<ChannelBlock> want(token_blocks);
        double widest = 0.0;
        for (int s = 0; s < token_blocks; s++)
        {
            want[s] = zero_block(layout);
            for (int i = 0; i < hidden_blocks; i++)
            {
                const ChannelBlock gate = matmul_block(wg[i], x[s], d);
                const ChannelBlock up = matmul_block(wu[i], x[s], d);

                ChannelBlock hidden = zero_block(layout);
                for (int m = 0; m < layout.batch; m++)
                {
                    for (std::size_t e = 0; e < gate[m].size(); e++)
                    {
                        widest = std::max(widest, std::abs(gate[m][e]));
                        hidden[m][e] = silu_host(gate[m][e]) * up[m][e];
                    }
                }
                add_into(want[s], matmul_block(wd[i], hidden, d));
            }
        }
        ASSERT_LT(widest, config.silu_bound);

        std::vector<heongpu::Ciphertext<S>> in;
        for (int s = 0; s < token_blocks; s++)
        {
            in.push_back(encrypt(pack_blocks(x[s], layout)));
        }

        std::vector<heongpu::Ciphertext<S>> got =
            ops->feed_forward(in, weights, config, *key, *relin);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(token_blocks));

        for (int s = 0; s < token_blocks; s++)
        {
            const std::string label =
                "long feed_forward[" + std::to_string(s) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[s]),
                               pack_blocks(want[s], layout)),
                      1e-5);
        }
    }

    /// Each token block is normalised over its own channels, so a wrong walk
    /// of the inputs would average a token against a different one.
    TEST_F(Llama3LayerEnv, RmsNormKeepsTheTokenBlocksApart)
    {
        const int d = layout.d;
        const int token_blocks = 2;
        const int channel_blocks = 2;

        auto key = narrow_key(llama::Llama3Operator::strided_rotation_indices(
            d * layout.batch, d));

        std::mt19937_64 rng(909);
        std::vector<ChannelBlock> x;
        for (int j = 0; j < channel_blocks; j++)
        {
            for (int s = 0; s < token_blocks; s++)
            {
                // Every block is a fresh draw, so the four summed squares are
                // all different and normalising against the wrong partner is a
                // whole-number error rather than a subtle one.
                x.push_back(random_blocks(layout, 1.0, rng));
            }
        }

        llama::Llama3Operator::RMSNormConfig config;
        config.stride = d * layout.batch;
        config.count = d;
        config.channels = d * channel_blocks;
        config.token_blocks = token_blocks;
        config.eps = 1e-5;
        config.degree = 31;
        // Nothing follows this call, so unlike the pre-norm blocks it can
        // afford the Newton step. Without it the Chebyshev fit alone holds
        // 1e-5 and the fit error, not the indexing, would be what is measured.
        config.newton_iterations = 1;

        std::vector<std::vector<double>> totals(
            token_blocks, std::vector<double>(config.stride, 0.0));
        double lowest = std::numeric_limits<double>::max();
        double highest = 0.0;
        for (int s = 0; s < token_blocks; s++)
        {
            for (int i = 0; i < config.stride; i++)
            {
                double total = 0.0;
                for (int j = 0; j < channel_blocks; j++)
                {
                    const std::vector<double> packed =
                        pack_blocks(x[j * token_blocks + s], layout);
                    for (int c = 0; c < config.count; c++)
                    {
                        const double value = packed[c * config.stride + i];
                        total += value * value;
                    }
                }
                totals[s][i] = total;
                lowest = std::min(lowest, total);
                highest = std::max(highest, total);
            }
        }
        config.sum_lo = 0.8 * lowest;
        config.sum_hi = 1.2 * highest;

        std::vector<std::vector<double>> want(x.size());
        for (int s = 0; s < token_blocks; s++)
        {
            for (int j = 0; j < channel_blocks; j++)
            {
                const std::size_t at = j * token_blocks + s;
                const std::vector<double> packed = pack_blocks(x[at], layout);
                want[at].assign(packed.size(), 0.0);
                for (int i = 0; i < config.stride; i++)
                {
                    const double factor =
                        1.0 / std::sqrt(totals[s][i] / config.channels +
                                        config.eps);
                    for (int c = 0; c < config.count; c++)
                    {
                        want[at][c * config.stride + i] =
                            packed[c * config.stride + i] * factor;
                    }
                }
            }
        }

        std::vector<heongpu::Ciphertext<S>> in;
        for (const auto& block : x)
        {
            in.push_back(encrypt(pack_blocks(block, layout)));
        }

        std::vector<heongpu::Plaintext<S>> no_weights;
        std::vector<heongpu::Ciphertext<S>> got =
            ops->rms_norm(in, no_weights, config, *key, *relin);
        ASSERT_EQ(got.size(), x.size());

        for (std::size_t at = 0; at < x.size(); at++)
        {
            const std::string label =
                "rms_norm block[" + std::to_string(at) + "]";
            EXPECT_LT(reported(label.c_str(), decrypt(got[at]), want[at]),
                      1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Bootstrapping
    // -----------------------------------------------------------------------

    /// A context that can bootstrap, which is a different regime and not
    /// merely a longer chain.
    ///
    /// Bootstrapping spends a fixed slice of the modulus chain on itself: the
    /// transform that moves the coefficients into the slots, the sine that
    /// stands in for the modular reduction, and the transform back. What a
    /// caller gets is the chain minus that slice, so the chain has to be long
    /// enough to pay for the slice and for the work afterwards. It also wants
    /// a sparse secret, which is what keeps the sine argument small enough to
    /// approximate, and that is one reason nothing here claims a security
    /// level.
    class Llama3BootstrapEnv : public ::testing::Test
    {
      protected:
        static constexpr int kDegree = 4096;
        static constexpr int kD = 8;

        /// What each half of a block spends, as measured by
        /// EachHalfOfABlockCostsWhatAChainMustCover at the settings both it
        /// and the whole-block tests use. Fifty three together, which is why
        /// a block does not fit on any chain here without a refresh.
        static constexpr int kAttentionHalf = 32;
        static constexpr int kSwiGLUHalf = 21;

        /// The chain, which a derived fixture raises.
        ///
        /// Forty eight is what one block wants: a bootstrap keeps twenty five
        /// of them and the twenty three left over cover the SwiGLU half's
        /// twenty one. They do not cover the attention half's thirty two,
        /// which is why a stack of blocks needs Llama3StackEnv's longer chain
        /// and not this one. BootstrappingBuysBackTheChain measures the
        /// twenty three.
        virtual int limbs() const { return 48; }
        /// The shipped bootstrapping example's configuration.
        static constexpr int kCtoSPiece = 3;
        static constexpr int kStoCPiece = 3;
        static constexpr int kTaylor = 11;
        static constexpr int kHammingWeight = 16;
        /// Fifty rather than the forty the layer fixtures use, and not a free
        /// choice: the sine that stands in for the modular reduction is set up
        /// around the ratio of the bottom prime to the scale, and at 2^60 over
        /// 2^40 it comes back as noise. Sixty over fifty is the ratio the
        /// procedure is built for.
        static constexpr int kPrimeBits = 50;
        static constexpr int kLogScale = 50;

        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Publickey<S>> pub;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::Galoiskey<S>> boot_galois;
        std::unique_ptr<heongpu::Relinkey<S>> relin;
        std::unique_ptr<llama::Llama3Operator> ops;

        double scale = std::pow(2.0, kLogScale);
        int slots = 0;
        llama::MatrixLayout layout;

        void SetUp() override
        {
            std::vector<int> logq{60};
            logq.insert(logq.end(), limbs() - 1, kPrimeBits);
            context->set_poly_modulus_degree(kDegree);
            context->set_coeff_modulus_bit_sizes(logq, {60, 60, 60});
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret =
                std::make_unique<heongpu::Secretkey<S>>(context,
                                                        kHammingWeight);
            keygen->generate_secret_key(*secret);
            pub = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*pub, *secret);
            encryptor = std::make_unique<heongpu::HEEncryptor<S>>(context, *pub);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            ops = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                          scale);
            slots = encoder->slot_count();
            layout = llama::MatrixLayout(kD, slots / (kD * kD));

            relin = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin, *secret);

            // The bootstrapping scale is the working scale on purpose.
            // generate_bootstrapping_params writes it into scale_boot_, and
            // that is the same field evaluate_poly consults when deciding
            // whether an intermediate has grown enough to need rescaling, so
            // a different one would quietly change how every Llama-3
            // polynomial rescales.
            heongpu::BootstrappingConfig boot_config(kCtoSPiece, kStoCPiece,
                                                     kTaylor, true);
            ops->generate_bootstrapping_params(
                scale, boot_config,
                heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

            std::vector<int> boot_shifts = ops->bootstrapping_key_indexs();
            boot_galois =
                std::make_unique<heongpu::Galoiskey<S>>(context, boot_shifts);
            keygen->generate_galois_key(*boot_galois, *secret);
        }

        heongpu::Ciphertext<S> encrypt(const std::vector<double>& values)
        {
            heongpu::Plaintext<S> plain(context);
            encoder->encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor->encrypt(cipher, plain);
            return cipher;
        }

        std::vector<double> decrypt(heongpu::Ciphertext<S>& cipher)
        {
            heongpu::Plaintext<S> plain(context);
            decryptor->decrypt(plain, cipher);
            std::vector<double> values;
            encoder->decode(values, plain);
            return values;
        }

        std::unique_ptr<heongpu::Galoiskey<S>> narrow_key(std::vector<int>
                                                              shifts)
        {
            auto key = std::make_unique<heongpu::Galoiskey<S>>(context, shifts);
            keygen->generate_galois_key(*key, *secret);
            return key;
        }

        /// Moduli still available to spend.
        int levels_left(heongpu::Ciphertext<S>& cipher) const
        {
            return limbs() - cipher.depth();
        }

        /// What a bootstrap hands back on this chain.
        int levels_after_a_refresh() const
        {
            return limbs() - (kCtoSPiece + kTaylor + kStoCPiece + 8);
        }
    };

    /// What one bootstrap costs and what it returns, on the packed layout
    /// every sublayer holds an activation in.
    ///
    /// The block structure is the thing to watch: bootstrapping is a
    /// slot-wise operation and knows nothing about the batch packing, so if
    /// the layout survived by accident rather than by construction it would
    /// show up here as blocks coming back interleaved.
    TEST_F(Llama3BootstrapEnv, BootstrappingBuysBackTheChain)
    {
        std::mt19937_64 rng(1001);
        const std::vector<std::vector<double>> blocks =
            random_blocks(layout, 0.5, rng);
        const std::vector<double> packed = pack_blocks(blocks, layout);

        heongpu::Ciphertext<S> cipher = encrypt(packed);
        ops->drop_to_depth(cipher, limbs() - 1);
        ASSERT_EQ(levels_left(cipher), 1);

        heongpu::Ciphertext<S> refreshed =
            ops->regular_bootstrapping(cipher, *boot_galois, *relin);

        const int usable = levels_left(refreshed);
        std::cout << "[ MEASURED ] bootstrap returns " << usable << " of "
                  << limbs() << " levels" << std::endl;

        // The slice a bootstrap keeps is CtoS + taylor + StoC + 8, and that
        // is the arithmetic the whole-block test's budget rests on, so it is
        // pinned here rather than left as a comment.
        EXPECT_EQ(usable, levels_after_a_refresh());

        // Bootstrapping is the least accurate thing in this file by a wide
        // margin: the sine standing in for the modular reduction is a
        // Taylor-and-double-angle approximation rather than a fit over the
        // data, so the bound here is set from what it delivers and not from
        // the noise floor everything else lands on.
        EXPECT_LT(reported("bootstrap", decrypt(refreshed), packed), 5e-3);
    }

    /// One whole transformer block: norm, attention, residual, refresh, norm,
    /// SwiGLU, residual.
    ///
    /// This is the first circuit here that does not fit on a chain, which is
    /// why there was no whole-block test before. The attention half spends
    /// about thirty levels and the SwiGLU half about twenty, and no chain
    /// these tests can afford holds fifty. The refresh in the middle is what
    /// turns one demand for fifty into two for thirty, and putting it between
    /// the halves rather than anywhere else is the whole of the arrangement:
    /// it lands where the chain is emptiest.
    ///
    /// The reference treats the refresh as the identity it is meant to be, so
    /// what the bound measures is how far from the identity it actually is.
    TEST_F(Llama3BootstrapEnv, TransformerBlockRunsAcrossARefresh)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(1002);
        const ChannelBlock x = random_blocks(layout, 1.0, rng);
        BlockWeightsHost host_weights;
        host_weights.query = random_blocks(layout, weight_scale, rng);
        host_weights.key = random_blocks(layout, weight_scale, rng);
        host_weights.value = random_blocks(layout, weight_scale, rng);
        host_weights.gate = random_blocks(layout, weight_scale, rng);
        host_weights.up = random_blocks(layout, weight_scale, rng);
        host_weights.down = random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::TransformerBlockConfig config;
        config.bootstrap = true;

        // The pre-attention norm, over the channels of the one block.
        config.attention_norm.stride = d * layout.batch;
        config.attention_norm.count = d;
        config.attention_norm.channels = d;
        config.attention_norm.eps = 1e-5;
        config.attention_norm.degree = 31;
        // Both norms skip the Newton step. A sublayer follows each of them and
        // wants the level more than the norm wants its last digit, and against
        // a bootstrap's own error the difference is invisible anyway.
        config.attention_norm.newton_iterations = 0;

        config.attention.layout = layout;
        config.attention.causal = true;
        config.attention.rope = false;
        config.attention.softmax.bound = 2.0;
        config.attention.softmax.iterations = 1;
        config.attention.softmax.exp_degree = 15;
        config.attention.softmax.inverse_degree = 15;
        config.attention.softmax.inverse_newton = 1;

        config.feed_forward.layout = layout;
        config.feed_forward.silu_bound = 4.0;
        config.feed_forward.silu_degree = 31;

        // Fills in the two norms' intervals and the score scale and shift,
        // which are the parts of the config that come from the data.
        const BlockHostRun want =
            transformer_block_host(x, host_weights, config, layout);
        ASSERT_LT(want.widest_gate, config.feed_forward.silu_bound);

        llama::Llama3Operator::TransformerBlockWeights weights;
        weights.attention.query =
            llama::BlockMatrix(flatten_block(host_weights.query));
        weights.attention.key =
            llama::BlockMatrix(flatten_block(host_weights.key));
        weights.attention.value =
            llama::BlockMatrix(flatten_block(host_weights.value));
        weights.feed_forward.gate =
            llama::BlockMatrix(flatten_block(host_weights.gate));
        weights.feed_forward.up =
            llama::BlockMatrix(flatten_block(host_weights.up));
        weights.feed_forward.down =
            llama::BlockMatrix(flatten_block(host_weights.down));

        // Exactly the list the block advertises, and nothing more. The
        // bootstrapping indices are a separate key on purpose.
        auto key = narrow_key(
            llama::Llama3Operator::transformer_block_rotation_indices(config));

        const std::vector<double> x_slots = pack_blocks(x, layout);
        std::vector<heongpu::Ciphertext<S>> in{encrypt(x_slots)};
        std::vector<heongpu::Ciphertext<S>> got = ops->transformer_block(
            in, weights, config, *key, *boot_galois, *relin);
        ASSERT_EQ(got.size(), std::size_t{1});

        // Two orders looser than the deepest single-chain circuit here, and
        // essentially all of it is the refresh: this lands within a few parts
        // in ten of what the bootstrap alone delivers, so the SwiGLU half that
        // follows it adds almost nothing to the error it inherits.
        EXPECT_LT(reported("transformer block", decrypt(got[0]),
                           pack_blocks(want.out, layout)),
                  5e-3);

        // And the refresh is not decorative. The same block, the same chain,
        // nothing else changed: fifty one levels of work do not fit in
        // forty eight, and it runs out part way through the SwiGLU half.
        llama::Llama3Operator::TransformerBlockConfig without = config;
        without.bootstrap = false;
        std::vector<heongpu::Ciphertext<S>> again{encrypt(x_slots)};
        EXPECT_ANY_THROW(ops->transformer_block(again, weights, without, *key,
                                                *boot_galois, *relin));
    }

    /// What each half of a block costs, which is the number that decides how
    /// long a chain has to be.
    ///
    /// Neither half is bootstrapped here, so both are cheap to measure: each
    /// one starts on a fresh chain and the depth it ends at is the depth it
    /// spent. Only the depths are checked. The second half is run on the
    /// stream the host says it would receive, so its config is calibrated for
    /// what it is handed, but its output is left to the whole-block test.
    ///
    /// The two inequalities below are the whole of the level argument. A
    /// refresh covers the SwiGLU half, so a block that starts on a full chain
    /// runs. It does not cover the attention half, so a block that starts on
    /// a refreshed one does not, and every block of a stack but the first
    /// starts on a refreshed one.
    TEST_F(Llama3BootstrapEnv, EachHalfOfABlockCostsWhatAChainMustCover)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));

        std::mt19937_64 rng(1004);
        const ChannelBlock x = random_blocks(layout, 1.0, rng);
        BlockWeightsHost host_weights;
        host_weights.query = random_blocks(layout, weight_scale, rng);
        host_weights.key = random_blocks(layout, weight_scale, rng);
        host_weights.value = random_blocks(layout, weight_scale, rng);
        host_weights.gate = random_blocks(layout, weight_scale, rng);
        host_weights.up = random_blocks(layout, weight_scale, rng);
        host_weights.down = random_blocks(layout, weight_scale, rng);

        llama::Llama3Operator::TransformerBlockConfig config;
        config.attention_norm.stride = d * layout.batch;
        config.attention_norm.count = d;
        config.attention_norm.channels = d;
        config.attention_norm.eps = 1e-5;
        config.attention_norm.degree = 31;
        config.attention_norm.newton_iterations = 0;
        config.attention.layout = layout;
        config.attention.causal = true;
        config.attention.rope = false;
        config.attention.softmax.bound = 2.0;
        config.attention.softmax.iterations = 1;
        config.attention.softmax.exp_degree = 15;
        config.attention.softmax.inverse_degree = 15;
        config.attention.softmax.inverse_newton = 1;
        config.feed_forward.layout = layout;
        config.feed_forward.silu_bound = 4.0;
        config.feed_forward.silu_degree = 31;

        const BlockHostRun host =
            transformer_block_host(x, host_weights, config, layout);
        ASSERT_LT(host.widest_gate, config.feed_forward.silu_bound);

        llama::Llama3Operator::TransformerBlockWeights weights;
        weights.attention.query =
            llama::BlockMatrix(flatten_block(host_weights.query));
        weights.attention.key =
            llama::BlockMatrix(flatten_block(host_weights.key));
        weights.attention.value =
            llama::BlockMatrix(flatten_block(host_weights.value));
        weights.feed_forward.gate =
            llama::BlockMatrix(flatten_block(host_weights.gate));
        weights.feed_forward.up =
            llama::BlockMatrix(flatten_block(host_weights.up));
        weights.feed_forward.down =
            llama::BlockMatrix(flatten_block(host_weights.down));

        auto key = narrow_key(
            llama::Llama3Operator::transformer_block_rotation_indices(config));

        std::vector<heongpu::Ciphertext<S>> first{
            encrypt(pack_blocks(x, layout))};
        std::vector<heongpu::Ciphertext<S>> normed =
            ops->rms_norm(first, weights.attention_norm, config.attention_norm,
                          *key, *relin);
        std::vector<heongpu::Ciphertext<S>> sublayer =
            ops->attention(normed, weights.attention, weights.rope,
                           config.attention, *key, *relin);
        std::vector<heongpu::Ciphertext<S>> stream =
            ops->residual_add(first, sublayer);
        const int attention_half = stream[0].depth();

        std::vector<heongpu::Ciphertext<S>> second{
            encrypt(pack_blocks(host.stream, layout))};
        std::vector<heongpu::Ciphertext<S>> renormed =
            ops->rms_norm(second, weights.feed_forward_norm,
                          config.feed_forward_norm, *key, *relin);
        std::vector<heongpu::Ciphertext<S>> swiglu =
            ops->feed_forward(renormed, weights.feed_forward,
                              config.feed_forward, *key, *relin);
        std::vector<heongpu::Ciphertext<S>> out =
            ops->residual_add(second, swiglu);
        const int swiglu_half = out[0].depth();

        std::cout << "[ MEASURED ] the attention half costs " << attention_half
                  << " levels and the SwiGLU half " << swiglu_half
                  << ", against " << levels_after_a_refresh()
                  << " a refresh hands back on a chain of " << limbs()
                  << std::endl;

        // Why one block runs on this chain, and why it needs the refresh.
        EXPECT_LE(swiglu_half, levels_after_a_refresh());
        EXPECT_GT(attention_half + swiglu_half, limbs());

        // And why a stack does not run on it. Every block after the first
        // begins where a refresh left off.
        EXPECT_GT(attention_half, levels_after_a_refresh());

        // The two numbers themselves, because the stack fixture's chain is
        // chosen from them and its leftover levels are checked against them.
        // A change in the sublayers moves these, and this is where it should
        // be noticed.
        EXPECT_EQ(attention_half, kAttentionHalf);
        EXPECT_EQ(swiglu_half, kSwiGLUHalf);
    }

    /// The same environment on the chain a run of blocks needs.
    ///
    /// A block gets away with forty eight because it starts on a full chain,
    /// so only its second half has to fit into what a refresh hands back.
    /// Every block after the first starts on a refreshed chain instead, so the
    /// deeper half has to fit there too. Sixty leaves thirty five after a
    /// refresh, against the attention half's thirty two: three in hand, and
    /// fifty six would have been the smallest chain that worked at all.
    ///
    /// Nothing else changes: the same sparse secret, the same fifty bit primes
    /// over a sixty bit bottom, the same bootstrapping configuration.
    class Llama3StackEnv : public Llama3BootstrapEnv
    {
      protected:
        int limbs() const override { return 60; }
    };

    /// Transformer blocks run one after another, which is the model's body.
    ///
    /// A stack is not a longer block. Each block refreshes once in its own
    /// middle, and the stack refreshes again at every seam, so three blocks
    /// cost five bootstraps and Llama-3's thirty two would cost sixty three.
    /// The seam is not optional: a block hands back a stream with only what
    /// its SwiGLU half did not spend, and the next block's attention half
    /// wants a chain as long as the one the first block started on.
    ///
    /// The stack is run at every prefix length and not only at the full one.
    /// What matters about a stack of thirty two is not the error at three but
    /// how the error grows per block, and a single total cannot separate a
    /// block that adds error from a block that multiplies it. Three points
    /// can, which is why there are three blocks here and not two.
    ///
    /// It adds. Averaged over six runs the three depths came out at 5.5e-4,
    /// 9.7e-4 and 1.3e-3, so each block contributed about 3.5e-4 of its own
    /// and inherited the rest unamplified. That is roughly what one bootstrap
    /// delivers, and it is the answer one would want: the pre-norm at the top
    /// of every block rescales the stream, so an error that arrives at a block
    /// leaves it the same size instead of being multiplied by whatever the
    /// block does. A thirty two block model would land near a percent at these
    /// settings, which is a number one can plan around; a compounding error
    /// would not have been.
    TEST_F(Llama3StackEnv, BlocksStackAcrossTheirSeams)
    {
        const int d = layout.d;
        const double weight_scale = 1.0 / std::sqrt(static_cast<double>(d));
        constexpr int kBlocks = 3;

        std::mt19937_64 rng(1005);
        const ChannelBlock x = random_blocks(layout, 1.0, rng);

        std::vector<BlockWeightsHost> host_weights(kBlocks);
        for (BlockWeightsHost& w : host_weights)
        {
            w.query = random_blocks(layout, weight_scale, rng);
            w.key = random_blocks(layout, weight_scale, rng);
            w.value = random_blocks(layout, weight_scale, rng);
            w.gate = random_blocks(layout, weight_scale, rng);
            w.up = random_blocks(layout, weight_scale, rng);
            w.down = random_blocks(layout, weight_scale, rng);
        }

        llama::Llama3Operator::TransformerStackConfig config;
        config.bootstrap_between_blocks = true;
        config.blocks.resize(kBlocks);

        // Each block is calibrated on the stream the block before it produced,
        // which is what a config per block is for: block two meets a wider
        // stream than block one, and a fit made for block one would be
        // extrapolating by the time it arrived.
        std::vector<ChannelBlock> want(kBlocks);
        ChannelBlock stream = x;
        for (int l = 0; l < kBlocks; l++)
        {
            llama::Llama3Operator::TransformerBlockConfig& block =
                config.blocks[l];
            block.bootstrap = true;
            block.attention_norm.stride = d * layout.batch;
            block.attention_norm.count = d;
            block.attention_norm.channels = d;
            block.attention_norm.eps = 1e-5;
            block.attention_norm.degree = 31;
            block.attention_norm.newton_iterations = 0;

            block.attention.layout = layout;
            block.attention.causal = true;
            block.attention.rope = false;
            block.attention.softmax.bound = 2.0;
            block.attention.softmax.iterations = 1;
            block.attention.softmax.exp_degree = 15;
            block.attention.softmax.inverse_degree = 15;
            block.attention.softmax.inverse_newton = 1;

            block.feed_forward.layout = layout;
            block.feed_forward.silu_degree = 31;

            const BlockHostRun run =
                transformer_block_host(stream, host_weights[l], block, layout);

            // The SiLU fit covers what this block's gate actually reaches,
            // with a fifth in hand. One fixed bound for every block would
            // extrapolate in the later ones or waste accuracy in the earlier.
            block.feed_forward.silu_bound = 1.2 * run.widest_gate;
            want[l] = run.out;
            stream = run.out;
        }

        std::vector<llama::Llama3Operator::TransformerBlockWeights> weights(
            kBlocks);
        for (int l = 0; l < kBlocks; l++)
        {
            weights[l].attention.query =
                llama::BlockMatrix(flatten_block(host_weights[l].query));
            weights[l].attention.key =
                llama::BlockMatrix(flatten_block(host_weights[l].key));
            weights[l].attention.value =
                llama::BlockMatrix(flatten_block(host_weights[l].value));
            weights[l].feed_forward.gate =
                llama::BlockMatrix(flatten_block(host_weights[l].gate));
            weights[l].feed_forward.up =
                llama::BlockMatrix(flatten_block(host_weights[l].up));
            weights[l].feed_forward.down =
                llama::BlockMatrix(flatten_block(host_weights[l].down));
        }

        // The blocks agree on shape, so this is one block's list; asking the
        // stack for it rather than assuming that is the point.
        auto key = narrow_key(
            llama::Llama3Operator::transformer_stack_rotation_indices(config));

        const std::vector<double> x_slots = pack_blocks(x, layout);

        std::vector<double> measured(kBlocks, 0.0);
        int leftover = 0;
        for (int n = 1; n <= kBlocks; n++)
        {
            llama::Llama3Operator::TransformerStackConfig prefix = config;
            prefix.blocks.resize(n);
            std::vector<llama::Llama3Operator::TransformerBlockWeights> few(
                weights.begin(), weights.begin() + n);

            std::vector<heongpu::Ciphertext<S>> in{encrypt(x_slots)};
            std::vector<heongpu::Ciphertext<S>> got = ops->transformer_stack(
                in, few, prefix, *key, *boot_galois, *relin);
            ASSERT_EQ(got.size(), std::size_t{1});

            const std::string label = "stack of " + std::to_string(n);
            measured[n - 1] = reported(label.c_str(), decrypt(got[0]),
                                       pack_blocks(want[n - 1], layout));

            // One block's allowance per block, which is the bootstrap's own
            // bound: the measurement is that a block *adds* error rather than
            // multiplying what it inherits, and this is the shape of bound
            // that says so. Over six runs the worst at one, two and three
            // blocks was 6.6e-4, 1.2e-3 and 2.0e-3, so the headroom here is a
            // steady eight or so at every depth rather than one that erodes.
            // Three blocks is not yet far enough for a linear allowance and a
            // geometric one to part company; what rules out compounding is
            // the series printed below, where the ratio falls off as the
            // increment stays put.
            EXPECT_LT(measured[n - 1], n * 5e-3);
            leftover = levels_left(got[0]);
        }

        for (int n = 1; n < kBlocks; n++)
        {
            std::cout << "[ MEASURED ] block " << (n + 1) << " multiplied the "
                      << "error by " << (measured[n] / measured[n - 1])
                      << " and added " << (measured[n] - measured[n - 1])
                      << std::endl;
        }
        std::cout << "[ MEASURED ] a stack of " << kBlocks << " leaves "
                  << leftover << " of " << limbs() << " levels" << std::endl;

        // The stream comes back as the last block left it, unrefreshed, with
        // what that block's SwiGLU half did not spend. However many blocks ran
        // is irrelevant: every one of them started from a refresh.
        EXPECT_EQ(leftover, levels_after_a_refresh() - kSwiGLUHalf);

        // A config per block is a requirement rather than a convenience, so a
        // list that disagrees with the weights is refused instead of being
        // stretched to fit.
        llama::Llama3Operator::TransformerStackConfig one = config;
        one.blocks.resize(1);
        std::vector<heongpu::Ciphertext<S>> in_bad{encrypt(x_slots)};
        EXPECT_THROW(ops->transformer_stack(in_bad, weights, one, *key,
                                            *boot_galois, *relin),
                     std::invalid_argument);
    }

} // namespace
