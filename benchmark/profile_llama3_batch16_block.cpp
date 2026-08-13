// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// ONE TRANSFORMER BLOCK ON SIXTEEN BATCHED INPUTS
// ===============================================
//
// The integration the four batch-16 sessions did not have. Each of them
// validated its own seam against a host reference; nothing ran the seams
// against EACH OTHER, and a dataflow fault lives exactly there -- in a level
// that does not line up, a scale that drifted, an orientation that two
// functions disagree about. Every unit test in the suite would pass with such
// a fault in place.
//
// So this walks one whole block and, at every seam, compares the decrypted
// homomorphic value against a host reference computed from the SAME input to
// that stage. Per-stage error isolates WHERE a fault is; the end-to-end error
// says whether the block as a whole computes a transformer block.
//
//   norm1 -> Q,K,V -> scores -> SoftMax seam -> P V -> W_o -> residual
//         -> norm2 -> SwiGLU -> residual
//
// PARTIAL BY DESIGN. The real 8B width does not fit on a 48 GiB A6000 -- the
// first RMSNorm alone holds four cohorts of d_model ciphertexts -- so the
// shape is a parameter and the driver reports the peak it reached. Run
// HEONGPU_B16_STAGE=norm|attention|ffn|block; each stage is independently
// checked and the wider ones can simply be skipped on a card that cannot hold
// them.
//
//   HEONGPU_B16_MODEL      residual width (channels)              128
//   HEONGPU_B16_HIDDEN     SwiGLU hidden width                    256
//   HEONGPU_B16_HEADS      query heads                              2
//   HEONGPU_B16_KV_HEADS   key/value heads (GQA)                    1
//   HEONGPU_B16_LIMBS      chain length                            62
//   HEONGPU_B16_HIDDEN_BLOCK  hidden channels held at once         64
//   HEONGPU_B16_STAGE      norm | attention | ffn | block        block
//   HEONGPU_B16_SLOT_RESIDENT  keep the stream in slot form         0
//   HEONGPU_B16_ROPE       apply rotary embedding to Q and K        0
//
// d = 128 and batch = k/2 = 16 are NOT parameters: batch 16 with head_dim 128
// pins (N = 4096, d = 128, k = 32) uniquely. See LLAMA3_8B_LAYER_FLOW.md 22.1.

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;
    using Clock = std::chrono::steady_clock;

    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }
    std::string EnvStr(const char* name, const char* fallback)
    {
        const char* raw = std::getenv(name);
        return (raw == nullptr || *raw == '\0') ? fallback : raw;
    }

    // ------------------------------------------------------------------
    // Host reference. Row-major [token][channel], one per batch instance.
    // ------------------------------------------------------------------
    using Mat = std::vector<double>;
    using Batch = std::vector<Mat>;

    Mat matmul(const Mat& a, const Mat& w, int rows, int inner, int cols)
    {
        Mat out(static_cast<size_t>(rows) * cols, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int t = 0; t < inner; ++t)
            {
                const double av = a[static_cast<size_t>(r) * inner + t];
                if (av == 0.0)
                    continue;
                for (int c = 0; c < cols; ++c)
                    out[static_cast<size_t>(r) * cols + c] +=
                        av * w[static_cast<size_t>(t) * cols + c];
            }
        return out;
    }

    Mat rms_norm_host(const Mat& x, int rows, int cols,
                      const std::vector<double>& gain, double eps)
    {
        Mat out(x.size(), 0.0);
        for (int u = 0; u < rows; ++u)
        {
            double acc = 0.0;
            for (int j = 0; j < cols; ++j)
            {
                const double v = x[static_cast<size_t>(u) * cols + j];
                acc += v * v;
            }
            const double r = std::sqrt(acc / cols + eps);
            for (int j = 0; j < cols; ++j)
            {
                const size_t at = static_cast<size_t>(u) * cols + j;
                out[at] = (gain.empty() ? 1.0 : gain[j]) * x[at] / r;
            }
        }
        return out;
    }

    double silu(double v) { return v / (1.0 + std::exp(-v)); }

    Mat ffn_host(const Mat& x, const Mat& wg, const Mat& wu, const Mat& wd,
                 int rows, int in_c, int hid)
    {
        const Mat g = matmul(x, wg, rows, in_c, hid);
        const Mat u = matmul(x, wu, rows, in_c, hid);
        Mat h(g.size());
        for (size_t e = 0; e < g.size(); ++e)
            h[e] = silu(g[e]) * u[e];
        return matmul(h, wd, rows, hid, in_c);
    }

    struct AttnWeights
    {
        Mat q, k, v, o;
    };

    // Causal single-head-group attention, matching the implementation's
    // conventions: 1/sqrt(head_dim) on the query, the causal set j <= u, and
    // a true SoftMax -- the mask's row-count equaliser and the score shift
    // both cancel in the normalisation, which is what they are for.
    // The same rotation rope_slots applies: lane c pairs with c + hd/2 and
    // the angle depends on the token.
    void rope_host(Mat& m, int rows, int cols, int hd, double theta,
                   int offset)
    {
        const int half = hd / 2;
        for (int h = 0; h * hd < cols; ++h)
            for (int c = 0; c < half; ++c)
            {
                const double omega = std::pow(
                    theta, -2.0 * static_cast<double>(c) /
                               static_cast<double>(hd));
                for (int u = 0; u < rows; ++u)
                {
                    const double a =
                        static_cast<double>(u + offset) * omega;
                    const size_t i0 =
                        static_cast<size_t>(u) * cols + h * hd + c;
                    const size_t i1 = i0 + static_cast<size_t>(half);
                    const double x0 = m[i0], x1 = m[i1];
                    m[i0] = x0 * std::cos(a) - x1 * std::sin(a);
                    m[i1] = x0 * std::sin(a) + x1 * std::cos(a);
                }
            }
    }

    Mat attention_host(const Mat& x, const AttnWeights& w, int rows, int in_c,
                       int q_c, int kv_c, int heads, int kv_heads, int hd,
                       bool rope = false, double theta = 500000.0,
                       int offset = 0)
    {
        Mat q = matmul(x, w.q, rows, in_c, q_c);
        Mat k = matmul(x, w.k, rows, in_c, kv_c);
        const Mat v = matmul(x, w.v, rows, in_c, kv_c);
        if (rope)
        {
            rope_host(q, rows, q_c, hd, theta, offset);
            rope_host(k, rows, kv_c, hd, theta, offset);
        }
        const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
        const int group = heads / kv_heads;

        Mat ctx(static_cast<size_t>(rows) * q_c, 0.0);
        for (int h = 0; h < heads; ++h)
        {
            const int qb = h * hd;
            const int kb = (h / group) * hd;
            for (int u = 0; u < rows; ++u)
            {
                std::vector<double> s(static_cast<size_t>(u) + 1, 0.0);
                double top = -1e300;
                for (int j = 0; j <= u; ++j)
                {
                    double acc = 0.0;
                    for (int c = 0; c < hd; ++c)
                        acc += q[static_cast<size_t>(u) * q_c + qb + c] *
                               k[static_cast<size_t>(j) * kv_c + kb + c];
                    s[static_cast<size_t>(j)] = acc * scale;
                    top = std::max(top, s[static_cast<size_t>(j)]);
                }
                double denom = 0.0;
                for (int j = 0; j <= u; ++j)
                {
                    s[static_cast<size_t>(j)] =
                        std::exp(s[static_cast<size_t>(j)] - top);
                    denom += s[static_cast<size_t>(j)];
                }
                for (int j = 0; j <= u; ++j)
                {
                    const double p = s[static_cast<size_t>(j)] / denom;
                    for (int c = 0; c < hd; ++c)
                        ctx[static_cast<size_t>(u) * q_c + qb + c] +=
                            p * v[static_cast<size_t>(j) * kv_c + kb + c];
                }
            }
        }
        return w.o.empty() ? ctx : matmul(ctx, w.o, rows, q_c, in_c);
    }

    Batch random_batch(int instances, int rows, int cols, uint64_t seed,
                       double amp)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-amp, amp);
        Batch out(instances,
                  Mat(static_cast<size_t>(rows) * cols, 0.0));
        for (auto& m : out)
            for (auto& e : m)
                e = dist(rng);
        return out;
    }

    Mat random_weight(int rows, int cols, uint64_t seed, double amp)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-amp, amp);
        Mat w(static_cast<size_t>(rows) * cols);
        for (auto& e : w)
            e = dist(rng);
        return w;
    }

    double worst_error(const Batch& want, const Batch& got)
    {
        double worst = 0.0;
        for (size_t b = 0; b < want.size(); ++b)
            for (size_t e = 0; e < want[b].size(); ++e)
                worst = std::max(worst, std::abs(want[b][e] - got[b][e]));
        return worst;
    }

    struct Stage
    {
        std::string name;
        double ms = 0.0;
        int depth_in = 0;
        int depth_out = 0;
        double error = 0.0;
        int columns = 0;
    };
    std::vector<Stage> ledger;

    void note(const std::string& name, double ms, int din, int dout,
              double err, int cols)
    {
        ledger.push_back(Stage{name, ms, din, dout, err, cols});
    }

    double gib(std::size_t bytes)
    {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    }
} // namespace

int main()
{
    // ---- shape -------------------------------------------------------
    const int d = 128;   // tokens per block; fixed by the ring at batch 16
    const int hd = 128;  // head_dim; must be a multiple of d
    const size_t N = 4096;

    const int model = EnvInt("HEONGPU_B16_MODEL", 128);
    const int hidden = EnvInt("HEONGPU_B16_HIDDEN", 256);
    const int heads = EnvInt("HEONGPU_B16_HEADS", 2);
    const int kv_heads = EnvInt("HEONGPU_B16_KV_HEADS", 1);
    const int limbs = EnvInt("HEONGPU_B16_LIMBS", 62);
    const int hidden_block = EnvInt("HEONGPU_B16_HIDDEN_BLOCK", 64);
    const std::string stage = EnvStr("HEONGPU_B16_STAGE", "block");
    const bool slot_resident = EnvInt("HEONGPU_B16_SLOT_RESIDENT", 0) != 0;
    const bool use_rope = EnvInt("HEONGPU_B16_ROPE", 0) != 0;

    const int q_channels = heads * hd;
    const int kv_channels = kv_heads * hd;

    std::cout << "[b16] one block on 16 batched inputs\n"
              << "[b16] N " << N << ", d " << d << ", k 32, batch 16\n"
              << "[b16] model " << model << ", hidden " << hidden << ", heads "
              << heads << " over " << kv_heads << " kv, head_dim " << hd
              << "\n[b16] limbs " << limbs << ", stage " << stage
              << ", slot_resident " << slot_resident << ", rope " << use_rope
              << std::endl;

    // ---- context and keys --------------------------------------------
    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    std::vector<int> logq{60};
    logq.insert(logq.end(), limbs - 1, 40);
    context->set_poly_modulus_degree(N);
    context->set_coeff_modulus_bit_sizes(logq, {60, 60});
    context->generate();

    const double scale = std::pow(2.0, 40);
    heongpu::BatchMatrixLayout layout(static_cast<int>(N), d);
    if (layout.batch != 16)
    {
        std::cerr << "[b16] FATAL: this ring carries " << layout.batch
                  << " instances, not 16" << std::endl;
        return 1;
    }
    const int instances = layout.batch;

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder(context);

    llama::Llama3BatchOperator op(context, encoder, layout, scale);
    llama::Batch16Shape shape;
    shape.instances = instances;
    shape.d_model = model;
    shape.hidden = hidden;
    shape.heads = heads;
    shape.kv_heads = kv_heads;
    shape.head_dim = hd;
    llama::Llama3Batch16Operator nl(op, shape);

    std::vector<int> shifts = op.rotation_indices();
    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);
    std::cout << "[b16] keys: " << shifts.size() << " Galois indices + relin"
              << std::endl;

    // ---- data and weights --------------------------------------------
    Batch x = random_batch(instances, d, model, 20260813u, 0.5);
    const std::vector<double> gain1 = random_weight(1, model, 11u, 0.4);
    const std::vector<double> gain2 = random_weight(1, model, 12u, 0.4);
    for (auto& g : const_cast<std::vector<double>&>(gain1))
        g += 1.0;
    for (auto& g : const_cast<std::vector<double>&>(gain2))
        g += 1.0;

    AttnWeights aw;
    aw.q = random_weight(model, q_channels, 21u, 0.3);
    aw.k = random_weight(model, kv_channels, 22u, 0.3);
    aw.v = random_weight(model, kv_channels, 23u, 0.3);
    aw.o = random_weight(q_channels, model, 24u, 0.3);
    const Mat wg = random_weight(model, hidden, 31u, 0.3);
    const Mat wu = random_weight(model, hidden, 32u, 0.3);
    const Mat wd = random_weight(hidden, model, 33u, 0.3);

    // ---- calibration, taken off the host exactly as Section 4.3 does --
    auto bracket_sum = [&](const Batch& data, int cols, double& lo,
                           double& hi)
    {
        lo = 1e300;
        hi = 0.0;
        for (const auto& m : data)
            for (int u = 0; u < d; ++u)
            {
                double acc = 0.0;
                for (int j = 0; j < cols; ++j)
                {
                    const double v = m[static_cast<size_t>(u) * cols + j];
                    acc += v * v;
                }
                lo = std::min(lo, acc);
                hi = std::max(hi, acc);
            }
        lo *= 0.8;
        hi *= 1.2;
    };

    llama::Llama3Batch16Operator::RMSNormConfig norm_cfg;
    bracket_sum(x, model, norm_cfg.sum_lo, norm_cfg.sum_hi);
    norm_cfg.degree = 15;
    norm_cfg.newton_iterations = 0;
    norm_cfg.fold_mean_into_fit = true;

    // ---- encrypt ------------------------------------------------------
    llama::BatchActivation stream =
        op.encrypt(x, d, model, encryptor, scale);
    Batch ref = x;

    auto decrypt = [&](llama::BatchActivation& a) {
        return op.decrypt(a, decryptor, scale);
    };
    auto tick = [] { return Clock::now(); };
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const bool do_norm = (stage == "norm" || stage == "block");
    const bool do_attn = (stage == "attention" || stage == "block");
    const bool do_ffn = (stage == "ffn" || stage == "block");

    // ================= norm 1 =========================================
    llama::BatchActivation normed;
    Batch ref_normed;
    if (do_norm || do_attn)
    {
        // The gain is folded into the weights that read the normalised
        // stream, which is exact and free; the host reference therefore
        // applies it too, on the same side.
        auto t0 = tick();
        const int din = stream.column.front().depth();
        const std::vector<double> no_gain;
        normed = nl.rms_norm(stream, no_gain, norm_cfg, galois, relin);
        const int dout = normed.column.front().depth();
        auto t1 = tick();

        ref_normed.resize(instances);
        for (int b = 0; b < instances; ++b)
            ref_normed[b] =
                rms_norm_host(ref[b], d, model, {}, norm_cfg.eps);
        note("norm1", ms(t0, t1), din, dout,
             worst_error(ref_normed, decrypt(normed)), model);
    }

    // ================= attention ======================================
    if (do_attn)
    {
        // The learned gain rides on the projections that read the norm.
        AttnWeights folded = aw;
        llama::Llama3Batch16Operator::fold_gain(folded.q, gain1, model,
                                                q_channels);
        llama::Llama3Batch16Operator::fold_gain(folded.k, gain1, model,
                                                kv_channels);
        llama::Llama3Batch16Operator::fold_gain(folded.v, gain1, model,
                                                kv_channels);

        llama::Llama3BatchOperator::BatchAttentionWeights bw;
        bw.query = folded.q;
        bw.key = folded.k;
        bw.value = folded.v;
        bw.output = aw.o;

        llama::Llama3BatchOperator::BatchAttentionConfig cfg;
        cfg.in_channels = model;
        cfg.q_channels = q_channels;
        cfg.kv_channels = kv_channels;
        cfg.heads = heads;
        cfg.kv_heads = kv_heads;
        cfg.causal = true;
        cfg.rope = use_rope;
        cfg.rope_theta = 500000.0;
        cfg.rope_position_offset = 0;

        // The score range, off the host, the way calibration supplies it.
        AttnWeights gw = aw;
        llama::Llama3Batch16Operator::fold_gain(gw.q, gain1, model,
                                                q_channels);
        llama::Llama3Batch16Operator::fold_gain(gw.k, gain1, model,
                                                kv_channels);
        double smax = -1e300, smin = 1e300;
        const double hscale = 1.0 / std::sqrt(static_cast<double>(hd));
        for (int b = 0; b < instances; ++b)
        {
            Mat q = matmul(ref_normed[b], gw.q, d, model, q_channels);
            Mat k = matmul(ref_normed[b], gw.k, d, model, kv_channels);
            if (use_rope)
            {
                rope_host(q, d, q_channels, hd, 500000.0, 0);
                rope_host(k, d, kv_channels, hd, 500000.0, 0);
            }
            const int group = heads / kv_heads;
            for (int h = 0; h < heads; ++h)
                for (int u = 0; u < d; ++u)
                    for (int j = 0; j <= u; ++j)
                    {
                        double acc = 0.0;
                        for (int c = 0; c < hd; ++c)
                            acc += q[static_cast<size_t>(u) * q_channels +
                                     h * hd + c] *
                                   k[static_cast<size_t>(j) * kv_channels +
                                     (h / group) * hd + c];
                        acc *= hscale;
                        smax = std::max(smax, acc);
                        smin = std::min(smin, acc);
                    }
        }
        const double bound = (smax - smin) * 1.05 + 1e-6;
        cfg.score_shift = smax;
        cfg.softmax.bound = bound;
        cfg.softmax.iterations = 2;
        cfg.softmax.exp_degree = 15;
        cfg.softmax.inverse_degree = 63;
        cfg.softmax.inverse_newton = 0;
        cfg.seam.causal = true;
        cfg.seam.scores_carry_exp_domain = true;
        cfg.seam.fold_affine_into_mask = true;
        cfg.seam.hoisted_crossings = true;
        cfg.seam.cache_masks = true;

        // Round-zero denominator range and concentration, calibrated.
        {
            const double divisor = std::pow(2.0, cfg.softmax.iterations);
            double lo = 1e300, hi = 0.0, sharpest = 0.0;
            const int group = heads / kv_heads;
            for (int b = 0; b < instances; ++b)
            {
                Mat q = matmul(ref_normed[b], gw.q, d, model, q_channels);
                Mat k = matmul(ref_normed[b], gw.k, d, model, kv_channels);
                if (use_rope)
                {
                    rope_host(q, d, q_channels, hd, 500000.0, 0);
                    rope_host(k, d, kv_channels, hd, 500000.0, 0);
                }
                for (int h = 0; h < heads; ++h)
                    for (int u = 0; u < d; ++u)
                    {
                        const double w = static_cast<double>(d) /
                                         static_cast<double>(u + 1);
                        double total = 0.0, squares = 0.0;
                        std::vector<double> e(static_cast<size_t>(u) + 1);
                        for (int j = 0; j <= u; ++j)
                        {
                            double acc = 0.0;
                            for (int c = 0; c < hd; ++c)
                                acc += q[static_cast<size_t>(u) * q_channels +
                                         h * hd + c] *
                                       k[static_cast<size_t>(j) * kv_channels +
                                         (h / group) * hd + c];
                            acc = acc * hscale - cfg.score_shift;
                            e[static_cast<size_t>(j)] =
                                std::exp(acc / divisor);
                            const double sq = e[static_cast<size_t>(j)] *
                                              e[static_cast<size_t>(j)];
                            total += w * sq;
                            squares += sq;
                        }
                        lo = std::min(lo, total);
                        hi = std::max(hi, total);
                        double after = 0.0;
                        for (int j = 0; j <= u; ++j)
                        {
                            const double y = e[static_cast<size_t>(j)] *
                                             e[static_cast<size_t>(j)] /
                                             squares;
                            after += y * y;
                        }
                        sharpest = std::max(sharpest, after);
                    }
            }
            cfg.softmax.sum_lo = lo * 0.95;
            cfg.softmax.sum_hi = hi * 1.05;
            cfg.softmax.concentration =
                std::min(static_cast<double>(d),
                         sharpest * static_cast<double>(d) * 1.05);
        }

        auto t0 = tick();
        const int din = normed.column.front().depth();
        llama::BatchActivation sub =
            op.attention(normed, bw, cfg, galois, relin);
        const int dout = sub.column.front().depth();
        auto t1 = tick();

        Batch ref_sub(instances);
        {
            AttnWeights hw = aw;
            llama::Llama3Batch16Operator::fold_gain(hw.q, gain1, model,
                                                    q_channels);
            llama::Llama3Batch16Operator::fold_gain(hw.k, gain1, model,
                                                    kv_channels);
            llama::Llama3Batch16Operator::fold_gain(hw.v, gain1, model,
                                                    kv_channels);
            for (int b = 0; b < instances; ++b)
                ref_sub[b] =
                    attention_host(ref_normed[b], hw, d, model, q_channels,
                                   kv_channels, heads, kv_heads, hd, use_rope,
                                   500000.0, 0);
        }
        note("attention", ms(t0, t1), din, dout,
             worst_error(ref_sub, decrypt(sub)), q_channels);

        auto t2 = tick();
        stream.column = op.arith().residual_add(stream.column, sub.column);
        auto t3 = tick();
        for (int b = 0; b < instances; ++b)
            for (size_t e = 0; e < ref[b].size(); ++e)
                ref[b][e] += ref_sub[b][e];
        note("residual1", ms(t2, t3), dout, stream.column.front().depth(),
             worst_error(ref, decrypt(stream)), model);
    }

    // ================= norm 2 + SwiGLU ================================
    if (do_ffn)
    {
        llama::Llama3Batch16Operator::RMSNormConfig cfg2 = norm_cfg;
        bracket_sum(ref, model, cfg2.sum_lo, cfg2.sum_hi);

        auto t0 = tick();
        const int din = stream.column.front().depth();
        const std::vector<double> no_gain;
        llama::BatchActivation n2 =
            nl.rms_norm(stream, no_gain, cfg2, galois, relin);
        const int dout = n2.column.front().depth();
        auto t1 = tick();

        Batch ref_n2(instances);
        for (int b = 0; b < instances; ++b)
            ref_n2[b] = rms_norm_host(ref[b], d, model, {}, cfg2.eps);
        note("norm2", ms(t0, t1), din, dout,
             worst_error(ref_n2, decrypt(n2)), model);

        // The gain rides on the gate and up weights.
        llama::Llama3Batch16Operator::FeedForwardWeights fw;
        fw.gate = wg;
        fw.up = wu;
        fw.down = wd;
        llama::Llama3Batch16Operator::fold_gain(fw.gate, gain2, model, hidden);
        llama::Llama3Batch16Operator::fold_gain(fw.up, gain2, model, hidden);

        // The SiLU's range is a measurement of the data, not a constant.
        double gmax = 0.0;
        for (int b = 0; b < instances; ++b)
        {
            const Mat g = matmul(ref_n2[b], fw.gate, d, model, hidden);
            for (double v : g)
                gmax = std::max(gmax, std::abs(v));
        }

        llama::Llama3Batch16Operator::FeedForwardConfig fcfg;
        fcfg.silu_bound = gmax * 1.15 + 1e-6;
        fcfg.silu_degree = 15;
        fcfg.fold_silu_domain_into_gate = true;
        fcfg.hidden_block = hidden_block;
        fcfg.slot_resident = slot_resident;

        auto t2 = tick();
        llama::BatchActivation sub =
            nl.feed_forward(n2, fw, fcfg, galois, relin);
        const int fout = sub.column.front().depth();
        auto t3 = tick();

        Batch ref_sub(instances);
        for (int b = 0; b < instances; ++b)
            ref_sub[b] =
                ffn_host(ref_n2[b], fw.gate, fw.up, fw.down, d, model, hidden);
        note("swiglu", ms(t2, t3), dout, fout,
             worst_error(ref_sub, decrypt(sub)), hidden);

        auto t4 = tick();
        stream.column = op.arith().residual_add(stream.column, sub.column);
        auto t5 = tick();
        for (int b = 0; b < instances; ++b)
            for (size_t e = 0; e < ref[b].size(); ++e)
                ref[b][e] += ref_sub[b][e];
        note("residual2", ms(t4, t5), fout, stream.column.front().depth(),
             worst_error(ref, decrypt(stream)), model);
    }

    // ---- report -------------------------------------------------------
    std::cout << "\n[b16] " << std::left << std::setw(18) << "stage"
              << std::right << std::setw(11) << "ms" << std::setw(8) << "cols"
              << std::setw(8) << "lvl in" << std::setw(9) << "lvl out"
              << std::setw(14) << "max abs err" << std::endl;
    double total = 0.0;
    for (const Stage& s : ledger)
    {
        total += s.ms;
        std::cout << "[b16] " << std::left << std::setw(18) << s.name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(11) << s.ms << std::setw(8) << s.columns
                  << std::setw(8) << s.depth_in << std::setw(9) << s.depth_out
                  << std::scientific << std::setprecision(3) << std::setw(14)
                  << s.error << std::endl;
    }
    std::cout << "[b16] " << std::left << std::setw(18) << "TOTAL"
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(11) << total << std::endl;

    const int spent = ledger.empty() ? 0 : ledger.back().depth_out;
    std::cout << "[b16] levels spent " << spent << " of " << limbs - 1
              << " available" << std::endl;

    // This is the RMM POOL RESERVATION plus the context, not the working
    // set: the pool takes a fixed fraction of free VRAM at context generate
    // and holds it whatever the shape is. Reading it as demand is the mistake
    // the project has already made once -- a 128-channel norm does not use
    // 43 GiB. What it IS good for is the ceiling: if a shape runs at all, its
    // working set fitted inside this, and if it OOMs at this number the cause
    // is fragmentation rather than size.
    std::size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cout << "[b16] device " << std::fixed << std::setprecision(2)
              << gib(total_b - free_b) << " / " << gib(total_b)
              << " GiB reserved (RMM pool + context, NOT the working set)"
              << std::endl;

    double worst = 0.0;
    for (const Stage& s : ledger)
        worst = std::max(worst, s.error);
    const bool ok = worst < 5e-2 && spent < limbs - 1;
    std::cout << "[b16] " << (ok ? "DATAFLOW OK" : "DATAFLOW SUSPECT")
              << " (worst stage error " << std::scientific << worst << ")"
              << std::endl;
    return ok ? 0 : 1;
}
