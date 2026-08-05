// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_batch.cuh>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            /// Scoped NVTX range, matching the taxonomy llama3.cu publishes so
            /// a capture can put the two matrix paths side by side.
            struct Range
            {
                explicit Range(const char* name) { nvtxRangePushA(name); }
                Range(const Range&) = delete;
                Range& operator=(const Range&) = delete;
                ~Range() { nvtxRangePop(); }
            };

            /// The same, for a step inside a helper several callers share.
            struct SuffixRange
            {
                SuffixRange(const char* prefix, const char* suffix)
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
                    nvtxRangePushA(name);
                }
                SuffixRange(const SuffixRange&) = delete;
                SuffixRange& operator=(const SuffixRange&) = delete;
                ~SuffixRange() { nvtxRangePop(); }
            };

            using cd = std::complex<double>;

            /// Gauss-Jordan with partial pivoting on a dense complex matrix.
            ///
            /// The matrix being inverted is a Vandermonde whose nodes are the
            /// d-th roots of unity times a common phase, so it is a scaled DFT
            /// and about as well conditioned as a dense inverse ever is. That
            /// is why a general elimination is honest here rather than
            /// reckless; the closed form would be faster and no more accurate,
            /// and this runs once per operator.
            std::vector<cd> invert(std::vector<cd> a, int n)
            {
                std::vector<cd> inv(static_cast<size_t>(n) * n, cd(0.0, 0.0));
                for (int i = 0; i < n; ++i)
                    inv[static_cast<size_t>(i) * n + i] = cd(1.0, 0.0);

                for (int col = 0; col < n; ++col)
                {
                    int pivot = col;
                    double best = std::abs(a[static_cast<size_t>(col) * n + col]);
                    for (int r = col + 1; r < n; ++r)
                    {
                        const double m =
                            std::abs(a[static_cast<size_t>(r) * n + col]);
                        if (m > best)
                        {
                            best = m;
                            pivot = r;
                        }
                    }
                    if (best < 1e-12)
                    {
                        throw std::runtime_error(
                            "the bridge Vandermonde is singular, which means "
                            "the layout's automorphisms are not d distinct "
                            "slot rotations");
                    }
                    if (pivot != col)
                    {
                        for (int c = 0; c < n; ++c)
                        {
                            std::swap(a[static_cast<size_t>(col) * n + c],
                                      a[static_cast<size_t>(pivot) * n + c]);
                            std::swap(inv[static_cast<size_t>(col) * n + c],
                                      inv[static_cast<size_t>(pivot) * n + c]);
                        }
                    }

                    const cd d = a[static_cast<size_t>(col) * n + col];
                    for (int c = 0; c < n; ++c)
                    {
                        a[static_cast<size_t>(col) * n + c] /= d;
                        inv[static_cast<size_t>(col) * n + c] /= d;
                    }

                    for (int r = 0; r < n; ++r)
                    {
                        if (r == col)
                            continue;
                        const cd f = a[static_cast<size_t>(r) * n + col];
                        if (f == cd(0.0, 0.0))
                            continue;
                        for (int c = 0; c < n; ++c)
                        {
                            a[static_cast<size_t>(r) * n + c] -=
                                f * a[static_cast<size_t>(col) * n + c];
                            inv[static_cast<size_t>(r) * n + c] -=
                                f * inv[static_cast<size_t>(col) * n + c];
                        }
                    }
                }
                return inv;
            }
        } // namespace

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        Llama3BatchOperator::Llama3BatchOperator(
            HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder,
            const BatchMatrixLayout& layout, double scale)
            : context_(context), encoder_(encoder), arith_(context, encoder),
              matrix_(context, layout), batch_encoder_(layout.k),
              layout_(layout), primes_(context->get_key_modulus()),
              slot_count_(encoder.slot_count()), default_scale_(scale)
        {
            if (layout_.N != slot_count_ * 2)
            {
                throw std::invalid_argument(
                    "The batch matrix layout must be built for this context's "
                    "ring: layout.N has to be twice the slot count");
            }
            build_bridge_tables();
        }

        // -------------------------------------------------------------------
        // The bridge
        // -------------------------------------------------------------------

        std::vector<int> Llama3BatchOperator::bridge_rotation_indices() const
        {
            // A shift by delta * (k/2) leaves the batch index alone -- it adds
            // a multiple of k/2 and N/2 is a multiple of k/2 -- and cycles the
            // row index u modulo d, because N/2 = d * (k/2). So one rotation
            // serves every batch index at once, and the whole bridge needs
            // only these d - 1 shifts however wide the model is.
            const int step = layout_.k / 2;
            std::vector<int> indices;
            indices.reserve(layout_.d - 1);
            for (int delta = 1; delta < layout_.d; ++delta)
            {
                indices.push_back(delta * step);
            }
            return indices;
        }

        void Llama3BatchOperator::build_bridge_tables()
        {
            const int d = layout_.d;
            const int k = layout_.k;
            const int N = layout_.N;
            const int half = N / 2;
            const int step = k / 2;
            const double pi = std::acos(-1.0);
            const uint64_t mod = 2ull * static_cast<uint64_t>(N);

            // exponent[s] = 5^s mod 2N, the Galois element rotation index s
            // names. Slot s of a plaintext p is p(psi^{5^s}), which is the
            // convention get_batch_cmt_rotation_indices already relies on.
            std::vector<uint64_t> exponent(half);
            {
                uint64_t g = 1;
                for (int s = 0; s < half; ++s)
                {
                    exponent[s] = g;
                    g = (g * 5ull) % mod;
                }
            }

            auto psi_pow = [&](uint64_t e) {
                return std::polar(1.0, pi * static_cast<double>(e % mod) /
                                           static_cast<double>(N));
            };

            forward_diagonal_.assign(
                d, std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));
            inverse_diagonal_.assign(
                d, std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));

            std::vector<cd> V(static_cast<size_t>(d) * d);

            for (int b = 0; b < step; ++b)
            {
                // V[u][i] = psi^{i * 5^{b + u * k/2}}: the weight slot
                // b + u*(k/2) puts on matrix row i, straight off (*).
                for (int u = 0; u < d; ++u)
                {
                    const uint64_t e = exponent[b + u * step];
                    for (int i = 0; i < d; ++i)
                    {
                        V[static_cast<size_t>(u) * d + i] =
                            psi_pow(e * static_cast<uint64_t>(i));
                    }
                }
                const std::vector<cd> Vinv = invert(V, d);

                for (int u = 0; u < d; ++u)
                {
                    const int slot = b + u * step;

                    // from_slots: source[s] = sum_i V[u][i] x[i], and the
                    // input rotated by delta * step brings x[(u + delta) % d]
                    // into slot s.
                    for (int delta = 0; delta < d; ++delta)
                    {
                        const int i = (u + delta) % d;
                        const cd& z = V[static_cast<size_t>(u) * d + i];
                        forward_diagonal_[delta][slot] =
                            Complex64(z.real(), z.imag());
                    }

                    // to_slots: x[u] = sum_{u2} Vinv[u][u2] y[u2], and the
                    // input rotated by delta * step brings y[(u + delta) % d]
                    // into slot s.
                    for (int delta = 0; delta < d; ++delta)
                    {
                        const int u2 = (u + delta) % d;
                        const cd& z = Vinv[static_cast<size_t>(u) * d + u2];
                        inverse_diagonal_[delta][slot] =
                            Complex64(z.real(), z.imag());
                    }
                }
            }
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3BatchOperator::bridge(std::vector<Ciphertext<Scheme::CKKS>>& in,
                                    bool inverse, const char* name,
                                    Galoiskey<Scheme::CKKS>& galois_key)
        {
            const int d = layout_.d;
            const int step = layout_.k / 2;
            const std::vector<std::vector<Complex64>>& diagonal =
                inverse ? inverse_diagonal_ : forward_diagonal_;

            Range _r_bridge(name);

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.size());

            for (auto& source : in)
            {
                // The d shifted copies are shared by every diagonal of this
                // column, so they are taken once. This is the only key
                // switching the bridge does, and it does not grow with the
                // width of the model.
                std::vector<Ciphertext<Scheme::CKKS>> rotated;
                rotated.reserve(d);
                {
                    SuffixRange _r(name, "rotations");
                    for (int delta = 0; delta < d; ++delta)
                    {
                        if (delta == 0)
                        {
                            rotated.push_back(source);
                        }
                        else
                        {
                            Ciphertext<Scheme::CKKS> shifted(context_);
                            arith_.rotate_rows(source, shifted, galois_key,
                                               delta * step);
                            rotated.push_back(std::move(shifted));
                        }
                    }
                }

                const double plain_scale = rescale_prime(source);

                Ciphertext<Scheme::CKKS> acc(context_);
                bool started = false;
                {
                    SuffixRange _r(name, "diagonals");
                    for (int delta = 0; delta < d; ++delta)
                    {
                        Plaintext<Scheme::CKKS> plain =
                            encode(diagonal[delta], plain_scale,
                                   source.depth());

                        Ciphertext<Scheme::CKKS> term(context_);
                        arith_.multiply_plain(rotated[delta], plain, term);

                        if (!started)
                        {
                            acc = std::move(term);
                            started = true;
                        }
                        else
                        {
                            arith_.add_inplace(acc, term);
                        }
                    }
                    arith_.rescale_inplace(acc);
                }

                out.push_back(std::move(acc));
            }

            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3BatchOperator::to_slots(BatchActivation& in,
                                      Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_uniform(in, "to_slots");
            return bridge(in.column, /*inverse=*/true, "bridge.to_slots",
                          galois_key);
        }

        BatchActivation
        Llama3BatchOperator::from_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& in, int rows,
            Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (rows != layout_.d)
            {
                throw std::invalid_argument(
                    "A matrix encryption has exactly layout.d rows, because d "
                    "is the rank of the module and not a free parameter");
            }
            BatchActivation out;
            out.rows = rows;
            out.column =
                bridge(in, /*inverse=*/false, "bridge.from_slots", galois_key);
            return out;
        }

        // -------------------------------------------------------------------
        // The products
        // -------------------------------------------------------------------

        BatchActivation
        Llama3BatchOperator::project(BatchActivation& x,
                                     const std::vector<double>& weight,
                                     int in_channels, int out_channels,
                                     const char* name, int column_block)
        {
            require_uniform(x, name);
            if (x.columns() != in_channels)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " weight expects a different number of input channels than "
                    "the activation carries");
            }
            if (weight.size() != static_cast<size_t>(in_channels) *
                                     static_cast<size_t>(out_channels))
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " weight must be in_channels by out_channels, row major, "
                    "which is the transpose of the mathematical weight");
            }

            // The twiddled plaintext matrix is in_channels * cols * k
            // coefficients per limb, so the output axis is streamed rather
            // than uploaded whole. This changes cost and not arithmetic.
            if (column_block <= 0)
            {
                column_block = std::max(1, layout_.d);
            }
            column_block = std::min(column_block, out_channels);

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "project.%s", name);
            Range _r_project(range_name);

            std::vector<Ciphertext<Scheme::CKKS>*> in;
            in.reserve(x.column.size());
            for (auto& c : x.column)
            {
                in.push_back(&c);
            }

            const int nslots = batch_encoder_.slots();
            // The weight is encoded at the prime the rescale that follows will
            // divide by, so the product comes back at exactly the activation's
            // own scale one level down. Encoding it at the nominal scale
            // instead leaves the result at x_scale * plain_scale / prime, which
            // is not a scale anything downstream expects and reads as garbage
            // rather than as a mismatch.
            const double plain_scale = rescale_prime(x.column.front());

            BatchActivation out;
            out.rows = x.rows;
            out.column.reserve(out_channels);

            for (int base = 0; base < out_channels; base += column_block)
            {
                const int cols = std::min(column_block, out_channels - base);

                // The weight is real and shared by the batch, so every one of
                // the k/2 matrices handed to the encoder is the same slice.
                std::vector<std::vector<std::complex<double>>> batch(
                    nslots, std::vector<std::complex<double>>(
                                static_cast<size_t>(in_channels) * cols));
                for (int i = 0; i < in_channels; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        const double v =
                            weight[static_cast<size_t>(i) * out_channels +
                                   base + j];
                        batch[0][static_cast<size_t>(i) * cols + j] =
                            std::complex<double>(v, 0.0);
                    }
                }
                for (int s = 1; s < nslots; ++s)
                {
                    batch[s] = batch[0];
                }

                std::vector<int64_t> coeffs;
                {
                    SuffixRange _r(range_name, "weight_encode");
                    batch_encoder_.encode(batch, in_channels, cols,
                                          plain_scale, coeffs);
                    matrix_.encode_plaintext_matrix(coeffs, in_channels, cols,
                                                    x.column.front().depth(),
                                                    plain_scale);
                }

                // Algorithm 1 itself: two matrix products over R_{q,k}, one
                // per ciphertext component. No rotations, no key switching,
                // and therefore no rotation keys at all.
                std::vector<Ciphertext<Scheme::CKKS>> piece;
                {
                    SuffixRange _r(range_name, "pcmm");
                    matrix_.pcmm(piece, in, /*rescale=*/true);
                }

                // Algorithm 1 marks the rescale rather than performing it: it
                // leaves the result at x_scale * plain_scale and sets
                // rescale_required_. Left unspent that is not merely an
                // untidy scale, it is a plaintext whose coefficients have
                // outgrown int64, so the caller pays it here and the column
                // comes back at the activation's own scale one level down.
                for (auto& c : piece)
                {
                    arith_.rescale_inplace(c);
                    out.column.push_back(std::move(c));
                }
            }

            return out;
        }

        BatchActivation
        Llama3BatchOperator::matmul(BatchActivation& a, BatchActivation& b,
                                    const char* name,
                                    Galoiskey<Scheme::CKKS>& galois_key,
                                    Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(a, name);
            require_uniform(b, name);
            if (a.columns() != layout_.d || b.columns() != layout_.d)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " product needs both operands square at layout.d columns, "
                    "which is what Algorithm 4's three internal CMTs assume");
            }

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "matmul.%s", name);
            Range _r(range_name);

            std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
            lhs.reserve(a.column.size());
            rhs.reserve(b.column.size());
            for (auto& c : a.column)
            {
                lhs.push_back(&c);
            }
            for (auto& c : b.column)
            {
                rhs.push_back(&c);
            }

            BatchActivation out;
            out.rows = a.rows;
            matrix_.ccmm(out.column, lhs, rhs, galois_key, relin_key, arith_,
                         /*rescale=*/true);
            return out;
        }

        // -------------------------------------------------------------------
        // Host-side staging
        // -------------------------------------------------------------------

        BatchActivation
        Llama3BatchOperator::encrypt(
            const std::vector<std::vector<double>>& batch, int rows, int cols,
            HEEncryptor<Scheme::CKKS>& encryptor, double scale)
        {
            const int nslots = batch_encoder_.slots();
            if (static_cast<int>(batch.size()) != nslots)
            {
                throw std::invalid_argument(
                    "A matrix encryption carries exactly k/2 matrices");
            }

            std::vector<std::vector<std::complex<double>>> complex_batch(
                nslots);
            for (int s = 0; s < nslots; ++s)
            {
                if (batch[s].size() !=
                    static_cast<size_t>(rows) * static_cast<size_t>(cols))
                {
                    throw std::invalid_argument(
                        "every matrix in the batch must have rows * cols "
                        "entries");
                }
                complex_batch[s].resize(batch[s].size());
                for (size_t e = 0; e < batch[s].size(); ++e)
                {
                    complex_batch[s][e] =
                        std::complex<double>(batch[s][e], 0.0);
                }
            }

            std::vector<int64_t> coeffs;
            batch_encoder_.encode(complex_batch, rows, cols, scale, coeffs);

            std::vector<std::vector<int64_t>> columns;
            build_matrix_encryption_coefficients(coeffs, layout_, rows, cols,
                                                 columns);

            BatchActivation out;
            out.rows = rows;
            out.column.reserve(cols);
            for (int j = 0; j < cols; ++j)
            {
                Plaintext<Scheme::CKKS> plain(context_);
                matrix_.load_coefficients(plain, columns[j], scale);
                Ciphertext<Scheme::CKKS> c(context_);
                encryptor.encrypt(c, plain);
                out.column.push_back(std::move(c));
            }
            return out;
        }

        std::vector<std::vector<double>>
        Llama3BatchOperator::decrypt(BatchActivation& in,
                                     HEDecryptor<Scheme::CKKS>& decryptor,
                                     double scale)
        {
            const int cols = in.columns();
            std::vector<std::vector<int64_t>> columns(cols);
            for (int j = 0; j < cols; ++j)
            {
                Plaintext<Scheme::CKKS> plain(context_);
                decryptor.decrypt(plain, in.column[j]);
                matrix_.extract_coefficients(columns[j], plain);
            }

            std::vector<int64_t> coeffs;
            split_matrix_encryption_coefficients(columns, layout_, in.rows,
                                                 cols, coeffs);

            std::vector<std::vector<std::complex<double>>> complex_batch;
            batch_encoder_.decode(coeffs, in.rows, cols, scale, complex_batch);

            std::vector<std::vector<double>> out(complex_batch.size());
            for (size_t s = 0; s < complex_batch.size(); ++s)
            {
                out[s].resize(complex_batch[s].size());
                for (size_t e = 0; e < complex_batch[s].size(); ++e)
                {
                    out[s][e] = complex_batch[s][e].real();
                }
            }
            return out;
        }

        double Llama3BatchOperator::rescale_prime(
            const Ciphertext<Scheme::CKKS>& ct) const
        {
            const int level = ct.level();
            if (level < 0 || level >= static_cast<int>(primes_.size()))
            {
                throw std::invalid_argument(
                    "Ciphertext has no level left to rescale");
            }
            return static_cast<double>(primes_[level].value);
        }

        Plaintext<Scheme::CKKS>
        Llama3BatchOperator::encode(const std::vector<Complex64>& values,
                                    double scale, int depth)
        {
            Plaintext<Scheme::CKKS> plain(context_);
            encoder_.encode(plain, values, scale);
            for (int i = 0; i < depth; i++)
            {
                arith_.mod_drop_inplace(plain);
            }
            return plain;
        }

        void Llama3BatchOperator::require_uniform(const BatchActivation& in,
                                                  const char* name) const
        {
            if (in.empty())
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " operand has no columns, so there is no such activation");
            }
            const int depth = in.column.front().depth();
            const double scale = in.column.front().scale();
            for (const auto& c : in.column)
            {
                if (c.depth() != depth || c.scale() != scale)
                {
                    throw std::invalid_argument(
                        std::string("The columns of the ") + name +
                        " operand have drifted apart in level or scale, and "
                        "the products below add them");
                }
            }
        }

    } // namespace llama
} // namespace heongpu
