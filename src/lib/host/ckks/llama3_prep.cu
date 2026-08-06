// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_prep.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

            std::vector<double> read_f32(const std::string& path,
                                         std::size_t count)
            {
                std::ifstream in(path, std::ios::binary);
                if (!in)
                {
                    throw std::runtime_error("Cannot open " + path);
                }
                std::vector<float> raw(count);
                in.read(reinterpret_cast<char*>(raw.data()),
                        static_cast<std::streamsize>(count * sizeof(float)));
                if (static_cast<std::size_t>(in.gcount()) !=
                    count * sizeof(float))
                {
                    throw std::runtime_error(path + " is shorter than the "
                                                    "shape in meta.txt says");
                }
                char extra;
                if (in.read(&extra, 1))
                {
                    throw std::runtime_error(path + " is longer than the "
                                                    "shape in meta.txt says");
                }
                return std::vector<double>(raw.begin(), raw.end());
            }

            /// c[m x n] += / = a[m x k] b[k x n], all row major.
            void gemm(const double* a, const double* b, double* c, int m,
                      int k, int n)
            {
                for (std::size_t i = 0; i < static_cast<std::size_t>(m) * n;
                     i++)
                {
                    c[i] = 0.0;
                }
#pragma omp parallel for schedule(static)
                for (int i = 0; i < m; i++)
                {
                    const double* ai = a + static_cast<std::size_t>(i) * k;
                    double* ci = c + static_cast<std::size_t>(i) * n;
                    for (int t = 0; t < k; t++)
                    {
                        const double av = ai[t];
                        if (av == 0.0)
                        {
                            continue;
                        }
                        const double* bt =
                            b + static_cast<std::size_t>(t) * n;
                        for (int j = 0; j < n; j++)
                        {
                            ci[j] += av * bt[j];
                        }
                    }
                }
            }

            double absmax(const std::vector<double>& v)
            {
                double worst = 0.0;
                for (double x : v)
                {
                    worst = std::max(worst, std::abs(x));
                }
                return worst;
            }

            /// The bound the fold uses: the measured peak with margin,
            /// rounded up to a power of two so log2(B) -- the precision the
            /// scaling spends -- is a whole number of bits. Powers below one
            /// are allowed: a seam whose values are small is scaled UP, which
            /// spends nothing and wastes less of the refresh's range.
            double pow2_bound(double peak, double margin)
            {
                if (!(peak > 0.0))
                {
                    return 1.0;
                }
                return std::pow(2.0, std::ceil(std::log2(peak * margin)));
            }

            /// Scale row i of a row-major in x out weight by s[i].
            void scale_rows(std::vector<double>& w, const std::vector<double>& s,
                            int in, int out)
            {
                for (int i = 0; i < in; i++)
                {
                    const double f = s[i];
                    double* row = w.data() + static_cast<std::size_t>(i) * out;
                    for (int j = 0; j < out; j++)
                    {
                        row[j] *= f;
                    }
                }
            }

            void scale_all(std::vector<double>& w, double f)
            {
                for (double& x : w)
                {
                    x *= f;
                }
            }

            /// Right-multiply every aligned block of @p span columns of every
            /// row by the randomised Hadamard H diag(sign) / sqrt(span):
            /// transform the block, then the sign lands on the TRANSFORMED
            /// index. With span == cols this is the whole row.
            void hadamard_cols(std::vector<double>& w, int rows, int cols,
                               int span, const std::vector<double>& sign)
            {
                for (int i = 0; i < rows; i++)
                {
                    double* row = w.data() + static_cast<std::size_t>(i) * cols;
                    for (int base = 0; base < cols; base += span)
                    {
                        fwht_normalised(row + base, span);
                        for (int j = 0; j < span; j++)
                        {
                            row[base + j] *= sign[j];
                        }
                    }
                }
            }

            /// Left-multiply every aligned block of @p span rows by the
            /// TRANSPOSE of the same map: for a symmetric H that is the same
            /// transform along the row index, sign on the transformed index.
            void hadamard_rows(std::vector<double>& w, int rows, int cols,
                               int span, const std::vector<double>& sign)
            {
                const double inv = 1.0 / std::sqrt(static_cast<double>(span));
                for (int base = 0; base < rows; base += span)
                {
                    for (int len = 1; len < span; len <<= 1)
                    {
                        for (int i = 0; i < span; i += len << 1)
                        {
                            for (int j = i; j < i + len; j++)
                            {
                                double* p =
                                    w.data() +
                                    static_cast<std::size_t>(base + j) * cols;
                                double* q =
                                    w.data() +
                                    static_cast<std::size_t>(base + j + len) *
                                        cols;
                                for (int c = 0; c < cols; c++)
                                {
                                    const double u = p[c];
                                    const double v = q[c];
                                    p[c] = u + v;
                                    q[c] = u - v;
                                }
                            }
                        }
                    }
                    for (int j = 0; j < span; j++)
                    {
                        const double f = sign[j] * inv;
                        double* row =
                            w.data() +
                            static_cast<std::size_t>(base + j) * cols;
                        for (int c = 0; c < cols; c++)
                        {
                            row[c] *= f;
                        }
                    }
                }
            }

            std::vector<double> random_signs(int n, std::mt19937_64& rng)
            {
                std::vector<double> s(n);
                for (double& x : s)
                {
                    x = (rng() & 1) ? 1.0 : -1.0;
                }
                return s;
            }
        } // namespace

        void fwht_normalised(double* v, std::size_t n)
        {
            for (std::size_t len = 1; len < n; len <<= 1)
            {
                for (std::size_t i = 0; i < n; i += len << 1)
                {
                    for (std::size_t j = i; j < i + len; j++)
                    {
                        const double a = v[j];
                        const double b = v[j + len];
                        v[j] = a + b;
                        v[j + len] = a - b;
                    }
                }
            }
            const double inv = 1.0 / std::sqrt(static_cast<double>(n));
            for (std::size_t j = 0; j < n; j++)
            {
                v[j] *= inv;
            }
        }

        void rotate_stream(std::vector<double>& x, int rows, int channels,
                           std::uint64_t seed, bool inverse)
        {
            // The stream signs are the FIRST draws of the seeded generator,
            // exactly as prepare_block takes them; the head signs follow and
            // are not needed here.
            std::mt19937_64 rng(seed);
            const std::vector<double> sign = random_signs(channels, rng);
            if (!inverse)
            {
                hadamard_cols(x, rows, channels, channels, sign);
                return;
            }
            // Q = H diag(s) / sqrt(n) is undone by diag(s) H / sqrt(n): the
            // sign first, then the involution.
            for (int i = 0; i < rows; i++)
            {
                double* row = x.data() + static_cast<std::size_t>(i) * channels;
                for (int c = 0; c < channels; c++)
                {
                    row[c] *= sign[c];
                }
                fwht_normalised(row, channels);
            }
        }

        RealBlockBundle load_real_block(const std::string& dir)
        {
            RealBlockBundle b;
            {
                std::ifstream meta(dir + "/meta.txt");
                if (!meta)
                {
                    throw std::runtime_error("Cannot open " + dir +
                                             "/meta.txt");
                }
                std::string line;
                while (std::getline(meta, line))
                {
                    std::istringstream in(line);
                    std::string key;
                    in >> key;
                    if (key == "layer")
                        in >> b.layer;
                    else if (key == "tokens")
                        in >> b.tokens;
                    else if (key == "channels")
                        in >> b.channels;
                    else if (key == "kv_channels")
                        in >> b.kv_channels;
                    else if (key == "head_dim")
                        in >> b.head_dim;
                    else if (key == "hidden")
                        in >> b.hidden;
                    else if (key == "sink_tokens")
                        in >> b.sink_tokens;
                }
            }
            if (b.tokens <= 0 || b.channels <= 0 || b.kv_channels <= 0 ||
                b.head_dim <= 0 || b.hidden <= 0)
            {
                throw std::runtime_error(dir + "/meta.txt is missing a shape");
            }

            const std::size_t c = b.channels;
            const std::size_t kv = b.kv_channels;
            const std::size_t h = b.hidden;
            const std::size_t t = b.tokens;
            b.query = read_f32(dir + "/wq.f32", c * c);
            b.key = read_f32(dir + "/wk.f32", c * kv);
            b.value = read_f32(dir + "/wv.f32", c * kv);
            b.output = read_f32(dir + "/wo.f32", c * c);
            b.gate = read_f32(dir + "/wgate.f32", c * h);
            b.up = read_f32(dir + "/wup.f32", c * h);
            b.down = read_f32(dir + "/wdown.f32", h * c);
            b.attention_norm = read_f32(dir + "/attn_norm.f32", c);
            b.feed_forward_norm = read_f32(dir + "/ffn_norm.f32", c);
            b.input = read_f32(dir + "/input.f32", t * c);
            b.input_nosink = read_f32(dir + "/input_nosink.f32", t * c);
            return b;
        }

        std::vector<double> reference_block(
            const Llama3RectOperator::RectTransformerBlockWeights& weights,
            const std::vector<double>& input, const BlockShape& shape,
            double eps, double attn_norm_gain, double ffn_norm_gain,
            BlockCalibration* calibration)
        {
            const int t = shape.tokens;
            const int c = shape.channels;
            const int kv = shape.kv_channels;
            const int hd = shape.head_dim;
            const int hidden = shape.hidden;
            const int heads = shape.heads();
            const int kv_heads = shape.kv_heads();
            const int group = heads / kv_heads;
            const double head_scale =
                1.0 / std::sqrt(static_cast<double>(hd));

            BlockCalibration cal;
            cal.score_lo = std::numeric_limits<double>::infinity();
            cal.score_hi = -std::numeric_limits<double>::infinity();

            std::vector<double> x = input;
            cal.stream_entry_abs = absmax(x);

            // The norm, exactly as the circuit's fit approximates it: the
            // gain rides on the fitted function and eps is the config's.
            auto norm = [&](const std::vector<double>& in, int which,
                            double gain)
            {
                std::vector<double> out(in.size());
                for (int i = 0; i < t; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < c; j++)
                    {
                        const double v =
                            in[static_cast<std::size_t>(i) * c + j];
                        sum += v * v;
                    }
                    cal.norm_sum_lo[which] =
                        (i == 0) ? sum : std::min(cal.norm_sum_lo[which], sum);
                    cal.norm_sum_hi[which] =
                        std::max(cal.norm_sum_hi[which], sum);
                    const double f =
                        gain / std::sqrt(sum / static_cast<double>(c) + eps);
                    for (int j = 0; j < c; j++)
                    {
                        out[static_cast<std::size_t>(i) * c + j] =
                            in[static_cast<std::size_t>(i) * c + j] * f;
                    }
                }
                cal.normed_abs[which] = absmax(out);
                return out;
            };

            // ---------------- attention half ----------------
            {
                std::vector<double> normed = norm(x, 0, attn_norm_gain);

                std::vector<double> q(static_cast<std::size_t>(t) * c);
                std::vector<double> k(static_cast<std::size_t>(t) * kv);
                std::vector<double> v(static_cast<std::size_t>(t) * kv);
                gemm(normed.data(), weights.attention.query.data(), q.data(),
                     t, c, c);
                gemm(normed.data(), weights.attention.key.data(), k.data(), t,
                     c, kv);
                gemm(normed.data(), weights.attention.value.data(), v.data(),
                     t, c, kv);
                for (double& e : q)
                {
                    e *= head_scale;
                }
                cal.value_abs = absmax(v);

                // Scores, exact SoftMax, the value product -- per head, with
                // each group of queries reading its shared kv head, which is
                // the same expansion the operator performs on the weights.
                std::vector<double> attn(static_cast<std::size_t>(t) * c,
                                         0.0);
                std::vector<double> scores(static_cast<std::size_t>(t) * t);
                cal.row_shift.assign(static_cast<std::size_t>(t) * heads,
                                     0.0);
                for (int kk = 0; kk < 4; kk++)
                {
                    cal.softmax_row_sum_lo[kk] =
                        std::numeric_limits<double>::infinity();
                    cal.softmax_row_sum_hi[kk] = 0.0;
                }
                for (int head = 0; head < heads; head++)
                {
                    const int kv_head = head / group;
                    for (int i = 0; i < t; i++)
                    {
                        for (int j = 0; j <= i; j++)
                        {
                            double s = 0.0;
                            for (int e = 0; e < hd; e++)
                            {
                                s += q[static_cast<std::size_t>(i) * c +
                                       head * hd + e] *
                                     k[static_cast<std::size_t>(j) * kv +
                                       kv_head * hd + e];
                            }
                            scores[static_cast<std::size_t>(i) * t + j] = s;
                            cal.score_lo = std::min(cal.score_lo, s);
                            cal.score_hi = std::max(cal.score_hi, s);
                        }
                    }
                    for (int i = 0; i < t; i++)
                    {
                        double top = -std::numeric_limits<double>::infinity();
                        double bottom =
                            std::numeric_limits<double>::infinity();
                        for (int j = 0; j <= i; j++)
                        {
                            const double s =
                                scores[static_cast<std::size_t>(i) * t + j];
                            top = std::max(top, s);
                            bottom = std::min(bottom, s);
                        }
                        cal.row_shift[static_cast<std::size_t>(i) * heads +
                                      head] = top;
                        cal.score_row_span =
                            std::max(cal.score_row_span, top - bottom);
                        for (int kk = 0; kk < 4; kk++)
                        {
                            double denom_k = 0.0;
                            for (int j = 0; j <= i; j++)
                            {
                                denom_k += std::exp(
                                    2.0 *
                                    (scores[static_cast<std::size_t>(i) * t +
                                            j] -
                                     top) /
                                    std::pow(2.0, kk + 1));
                            }
                            cal.softmax_row_sum_lo[kk] = std::min(
                                cal.softmax_row_sum_lo[kk], denom_k);
                            cal.softmax_row_sum_hi[kk] = std::max(
                                cal.softmax_row_sum_hi[kk], denom_k);
                        }
                        double denom = 0.0;
                        for (int j = 0; j <= i; j++)
                        {
                            denom += std::exp(
                                scores[static_cast<std::size_t>(i) * t + j] -
                                top);
                        }
                        double prob_sq = 0.0;
                        for (int j = 0; j <= i; j++)
                        {
                            const double p =
                                std::exp(
                                    scores[static_cast<std::size_t>(i) * t +
                                           j] -
                                    top) /
                                denom;
                            prob_sq += p * p;
                        }
                        cal.prob_sq_hi = std::max(cal.prob_sq_hi, prob_sq);
                        for (int e = 0; e < hd; e++)
                        {
                            double acc = 0.0;
                            for (int j = 0; j <= i; j++)
                            {
                                const double p =
                                    std::exp(scores[static_cast<std::size_t>(
                                                        i) *
                                                        t +
                                                    j] -
                                             top) /
                                    denom;
                                acc += p *
                                       v[static_cast<std::size_t>(j) * kv +
                                         kv_head * hd + e];
                            }
                            attn[static_cast<std::size_t>(i) * c + head * hd +
                                 e] = acc;
                        }
                    }
                }

                // The SoftMax denominator the CIRCUIT will meet: under one
                // global calibrated shift, not the per-row exact one above,
                // and once per squaring count, since the round-zero
                // denominator is sum exp(2 (s - shift) / 2^kk) and kk is
                // chosen from the score range this same pass measured.
                for (int kk = 0; kk < 4; kk++)
                {
                    cal.softmax_sum_lo[kk] =
                        std::numeric_limits<double>::infinity();
                    cal.softmax_sum_hi[kk] = 0.0;
                }
                for (int head = 0; head < heads; head++)
                {
                    for (int i = 0; i < t; i++)
                    {
                        double denom[4] = {0.0, 0.0, 0.0, 0.0};
                        for (int j = 0; j <= i; j++)
                        {
                            double s = 0.0;
                            for (int e = 0; e < hd; e++)
                            {
                                s += q[static_cast<std::size_t>(i) * c +
                                       head * hd + e] *
                                     k[static_cast<std::size_t>(j) * kv +
                                       (head / group) * hd + e];
                            }
                            for (int kk = 0; kk < 4; kk++)
                            {
                                denom[kk] += std::exp(
                                    2.0 * (s - cal.score_hi) /
                                    std::pow(2.0, kk + 1));
                            }
                        }
                        for (int kk = 0; kk < 4; kk++)
                        {
                            cal.softmax_sum_lo[kk] =
                                std::min(cal.softmax_sum_lo[kk], denom[kk]);
                            cal.softmax_sum_hi[kk] =
                                std::max(cal.softmax_sum_hi[kk], denom[kk]);
                        }
                    }
                }

                std::vector<double> projected(
                    static_cast<std::size_t>(t) * c);
                gemm(attn.data(), weights.attention.output.data(),
                     projected.data(), t, c, c);
                for (std::size_t i = 0; i < x.size(); i++)
                {
                    x[i] += projected[i];
                }
                cal.stream_mid_abs = absmax(x);
            }

            // ---------------- feed-forward half ----------------
            {
                std::vector<double> normed = norm(x, 1, ffn_norm_gain);

                std::vector<double> gate(static_cast<std::size_t>(t) *
                                         hidden);
                std::vector<double> up(gate.size());
                gemm(normed.data(), weights.feed_forward.gate.data(),
                     gate.data(), t, c, hidden);
                gemm(normed.data(), weights.feed_forward.up.data(), up.data(),
                     t, c, hidden);
                cal.silu_in_abs = absmax(gate);

                for (std::size_t i = 0; i < gate.size(); i++)
                {
                    const double g = gate[i];
                    gate[i] = g / (1.0 + std::exp(-g)) * up[i];
                }
                cal.hidden_abs = absmax(gate);

                std::vector<double> down(static_cast<std::size_t>(t) * c);
                gemm(gate.data(), weights.feed_forward.down.data(),
                     down.data(), t, hidden, c);
                for (std::size_t i = 0; i < x.size(); i++)
                {
                    x[i] += down[i];
                }
                cal.stream_out_abs = absmax(x);
            }

            if (calibration != nullptr)
            {
                *calibration = cal;
            }
            return x;
        }

        PreparedBlock prepare_block(const RealBlockBundle& bundle,
                                    const SylphPrepConfig& config)
        {
            const int c = bundle.channels;
            const int kv = bundle.kv_channels;
            const int hd = bundle.head_dim;
            const int hidden = bundle.hidden;
            if (!is_pow2(c) || !is_pow2(hd))
            {
                throw std::invalid_argument(
                    "The rotations are randomised Hadamards, so the stream "
                    "width and the head dim must be powers of two");
            }
            if (kv % hd != 0 || c % hd != 0)
            {
                throw std::invalid_argument(
                    "head_dim must divide both projection widths");
            }

            PreparedBlock out;
            out.shape.tokens = bundle.tokens;
            out.shape.channels = c;
            out.shape.kv_channels = kv;
            out.shape.head_dim = hd;
            out.shape.hidden = hidden;

            auto& w = out.weights;
            w.attention.query = bundle.query;
            w.attention.key = bundle.key;
            w.attention.value = bundle.value;
            w.attention.output = bundle.output;
            w.feed_forward.gate = bundle.gate;
            w.feed_forward.up = bundle.up;
            w.feed_forward.down = bundle.down;
            // The learned gains are folded HERE, before the rotation, because
            // diag(gamma) acts on the unrotated stream and does not commute
            // with R1. The norm vectors are left empty and the block must run
            // with fold_norm_scale off.
            scale_rows(w.attention.query, bundle.attention_norm, c, c);
            scale_rows(w.attention.key, bundle.attention_norm, c, kv);
            scale_rows(w.attention.value, bundle.attention_norm, c, kv);
            scale_rows(w.feed_forward.gate, bundle.feed_forward_norm, c,
                       hidden);
            scale_rows(w.feed_forward.up, bundle.feed_forward_norm, c, hidden);

            out.input = bundle.input;
            out.input_nosink = bundle.input_nosink;

            if (config.rotate)
            {
                std::mt19937_64 rng(config.seed);
                const std::vector<double> stream_sign = random_signs(c, rng);
                const std::vector<double> head_sign = random_signs(hd, rng);

                // R1: the stream reads rotated, so everything READING it is
                // left-multiplied by Q^T on the in axis, everything WRITING
                // it is right-multiplied by Q on the out axis, and the input
                // itself is rotated row by row.
                hadamard_rows(w.attention.query, c, c, c, stream_sign);
                hadamard_rows(w.attention.key, c, kv, c, stream_sign);
                hadamard_rows(w.attention.value, c, kv, c, stream_sign);
                hadamard_rows(w.feed_forward.gate, c, hidden, c, stream_sign);
                hadamard_rows(w.feed_forward.up, c, hidden, c, stream_sign);
                hadamard_cols(w.attention.output, c, c, c, stream_sign);
                hadamard_cols(w.feed_forward.down, hidden, c, c, stream_sign);
                hadamard_cols(out.input, bundle.tokens, c, c, stream_sign);
                hadamard_cols(out.input_nosink, bundle.tokens, c, c,
                              stream_sign);

                // R2: V's head blocks are rotated on the way out and o_proj
                // undoes it on the way in. One sign vector serves every
                // head, so the host-side GQA expansion -- which copies whole
                // head blocks -- commutes with it.
                hadamard_cols(w.attention.value, c, kv, hd, head_sign);
                hadamard_rows(w.attention.output, c, c, hd, head_sign);
            }

            out.eps = config.eps;
            std::vector<double> raw_out = reference_block(
                w, out.input, out.shape, config.eps, 1.0, 1.0, &out.raw);
            reference_block(w, out.input_nosink, out.shape, config.eps, 1.0,
                            1.0, &out.raw_nosink);

            if (!config.prescale)
            {
                out.calibration = out.raw;
                out.expected = std::move(raw_out);
                return out;
            }

            const double stream_peak =
                std::max({out.raw.stream_entry_abs, out.raw.stream_mid_abs,
                          out.raw.stream_out_abs});
            out.scales.stream = pow2_bound(stream_peak, config.margin);
            out.scales.normed_attn =
                pow2_bound(out.raw.normed_abs[0], config.margin);
            out.scales.normed_ffn =
                pow2_bound(out.raw.normed_abs[1], config.margin);
            out.scales.value = pow2_bound(out.raw.value_abs, config.margin);
            out.scales.hidden = pow2_bound(out.raw.hidden_abs, config.margin);

            const double bs = out.scales.stream;
            const double bn1 = out.scales.normed_attn;
            const double bn2 = out.scales.normed_ffn;
            const double bv = out.scales.value;
            const double bh = out.scales.hidden;

            // The folds. Every one is a constant on a host array something
            // already multiplies, so the circuit gains no operation and
            // loses no level; see the header for who undoes whom.
            scale_all(out.input, 1.0 / bs);
            scale_all(out.input_nosink, 1.0 / bs);
            scale_all(w.attention.query, bn1);
            scale_all(w.attention.key, bn1);
            scale_all(w.attention.value, bn1 / bv);
            scale_all(w.attention.output, bv / bs);
            scale_all(w.feed_forward.gate, bn2);
            scale_all(w.feed_forward.up, bn2 / bh);
            scale_all(w.feed_forward.down, bh / bs);
            // The same norm of the same stream, seen through the scaling:
            // mean((x/B)^2) + eps/B^2 is (mean(x^2) + eps) / B^2, so B_s
            // cancels and the norm of the scaled stream IS the true norm.
            // The gain 1/B_n on the fit then puts the seam where the fold
            // wants it, and the projections reading it carry B_n back.
            out.eps = config.eps / (bs * bs);

            out.expected = reference_block(
                w, out.input, out.shape, out.eps, 1.0 / bn1, 1.0 / bn2,
                &out.calibration);
            return out;
        }

    } // namespace llama
} // namespace heongpu
