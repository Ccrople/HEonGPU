// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3.cuh>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace heongpu
{
    namespace llama
    {
        namespace
        {
            /// Scoped NVTX range, so a capture can attribute time to the part
            /// of the model that issued it.
            ///
            /// The names are dotted and the ranges nest, so an nvtxppsum rolls
            /// a model up by structure: "block" contains "attention" contains
            /// "attention.scores" contains "ccmm". A parent's total therefore
            /// includes its children's, which is what makes a fraction of the
            /// forward pass readable straight off the summary.
            struct Range
            {
                explicit Range(const char* name) { nvtxRangePushA(name); }
                Range(const Range&) = delete;
                Range& operator=(const Range&) = delete;
                ~Range() { nvtxRangePop(); }
            };

            /// The same, for a range whose name carries an index.
            ///
            /// Blocks are numbered because the question a stack raises is
            /// whether they cost the same, and one name for all of them cannot
            /// answer it.
            struct IndexedRange
            {
                IndexedRange(const char* format, int index)
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), format, index);
                    nvtxRangePushA(name);
                }
                IndexedRange(const IndexedRange&) = delete;
                IndexedRange& operator=(const IndexedRange&) = delete;
                ~IndexedRange() { nvtxRangePop(); }
            };

            /// The same, for a step inside a helper that several primitives
            /// share.
            ///
            /// The summary groups by name rather than by position in the tree,
            /// so a helper marked with one fixed name merges every caller into
            /// a single row and says nothing about which of them paid. The
            /// caller's own name is therefore the prefix.
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

            bool is_power_of_two(int v)
            {
                return v > 0 && (v & (v - 1)) == 0;
            }

            int wrap(int k, int d)
            {
                int r = k % d;
                return r < 0 ? r + d : r;
            }

            void require_layout(const MatrixLayout& layout)
            {
                if (!is_power_of_two(layout.d) || layout.batch <= 0 ||
                    layout.slots != layout.d * layout.d * layout.batch)
                {
                    throw std::invalid_argument(
                        "The matrix layout must be a default-constructed "
                        "MatrixLayout(d, batch), not one assembled by hand");
                }
            }
        } // namespace

        // -------------------------------------------------------------------
        // Host helpers
        // -------------------------------------------------------------------

        std::vector<double>
        chebyshev_coefficients(const std::function<double(double)>& f, double a,
                               double b, int degree)
        {
            if (degree < 0)
            {
                throw std::invalid_argument("Degree must not be negative");
            }
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }

            const int n = degree + 1;
            std::vector<double> values(n);
            for (int j = 0; j < n; j++)
            {
                const double t = std::cos(M_PI * (j + 0.5) / n);
                values[j] = f(0.5 * (b - a) * t + 0.5 * (a + b));
            }

            std::vector<double> coeffs(n, 0.0);
            for (int k = 0; k < n; k++)
            {
                double acc = 0.0;
                for (int j = 0; j < n; j++)
                {
                    acc += values[j] * std::cos(M_PI * k * (j + 0.5) / n);
                }
                // The k = 0 term is halved so the series needs no correction
                // when it is summed straight through.
                coeffs[k] = (k == 0 ? 1.0 : 2.0) * acc / n;
            }

            return coeffs;
        }

        double chebyshev_evaluate(const std::vector<double>& coeffs, double a,
                                  double b, double x)
        {
            const double t = (2.0 * x - a - b) / (b - a);
            double t0 = 1.0;
            double t1 = t;
            double acc = coeffs.empty() ? 0.0 : coeffs[0];
            for (std::size_t k = 1; k < coeffs.size(); k++)
            {
                acc += coeffs[k] * t1;
                const double t2 = 2.0 * t * t1 - t0;
                t0 = t1;
                t1 = t2;
            }
            return acc;
        }

        int chebyshev_levels(int degree)
        {
            if (degree < 1)
            {
                return 0;
            }
            int levels = 0;
            int reach = 1;
            while (reach <= degree)
            {
                reach *= 2;
                levels++;
            }
            return levels;
        }

        int chebyshev_round_degree(int degree)
        {
            if (degree < 1)
            {
                return degree < 0 ? 0 : degree;
            }
            int rounded = 1;
            while (rounded < degree)
            {
                rounded = 2 * rounded + 1;
            }
            return rounded;
        }

        double chebyshev_max_error(const std::function<double(double)>& f,
                                   double a, double b, int degree, bool relative,
                                   int samples)
        {
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }
            if (samples < 2)
            {
                throw std::invalid_argument("A fit needs at least two samples");
            }

            const std::vector<double> coeffs =
                chebyshev_coefficients(f, a, b, degree);

            // Sampled uniformly rather than at the nodes: the nodes are where
            // the interpolant is exact, so measuring there would report zero
            // for any degree at all.
            double worst = 0.0;
            for (int i = 0; i < samples; i++)
            {
                const double x =
                    a + (b - a) * static_cast<double>(i) / (samples - 1);
                const double want = f(x);
                double err = std::abs(chebyshev_evaluate(coeffs, a, b, x) - want);
                if (relative)
                {
                    // A function with a zero inside its interval has no
                    // relative error to speak of there; the caller asking for
                    // one on such a function is the bug, not this guard.
                    const double magnitude = std::abs(want);
                    if (magnitude <= 0.0)
                    {
                        continue;
                    }
                    err /= magnitude;
                }
                worst = std::max(worst, err);
            }
            return worst;
        }

        int chebyshev_degree_for(const std::function<double(double)>& f, double a,
                                 double b, double target, bool relative, int cap)
        {
            if (!(target > 0.0))
            {
                throw std::invalid_argument("Target precision must be positive");
            }

            // Only the Paterson-Stockmeyer shapes are tried, because the ones
            // between two of them cost the same levels as the larger and
            // approximate no better.
            for (int degree = 1; degree <= cap; degree = 2 * degree + 1)
            {
                if (chebyshev_max_error(f, a, b, degree, relative) <= target)
                {
                    return degree;
                }
            }
            return 0;
        }

        MatrixLayout::MatrixLayout(int d_, int batch_) : d(d_), batch(batch_)
        {
            if (!is_power_of_two(d) || batch <= 0)
            {
                throw std::invalid_argument(
                    "MatrixLayout needs a power-of-two d and a positive batch");
            }
            slots = d * d * batch;
        }

        BlockMatrix::BlockMatrix(int out_blocks_, int in_blocks_)
            : out_blocks(out_blocks_), in_blocks(in_blocks_)
        {
            if (out_blocks <= 0 || in_blocks <= 0)
            {
                throw std::invalid_argument(
                    "A block matrix needs a positive number of blocks in each "
                    "direction");
            }
            blocks.resize(static_cast<std::size_t>(out_blocks) * in_blocks);
        }

        BlockMatrix::BlockMatrix(std::vector<double> single)
        {
            if (single.empty())
            {
                return;
            }
            out_blocks = 1;
            in_blocks = 1;
            blocks.push_back(std::move(single));
        }

        std::vector<double>& BlockMatrix::at(int i, int j)
        {
            return const_cast<std::vector<double>&>(
                const_cast<const BlockMatrix*>(this)->at(i, j));
        }

        const std::vector<double>& BlockMatrix::at(int i, int j) const
        {
            if (i < 0 || i >= out_blocks || j < 0 || j >= in_blocks)
            {
                throw std::out_of_range(
                    "Block index outside the block matrix");
            }
            return blocks[static_cast<std::size_t>(i) * in_blocks + j];
        }

        std::vector<double> permute_sigma(const std::vector<double>& a, int d)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    out[i * d + j] = a[i * d + wrap(i + j, d)];
                }
            }
            return out;
        }

        std::vector<double> permute_tau(const std::vector<double>& a, int d)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    out[i * d + j] = a[wrap(i + j, d) * d + j];
                }
            }
            return out;
        }

        std::vector<double> rotate_rows_host(const std::vector<double>& a,
                                             int d, int k)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    out[i * d + j] = a[wrap(i + k, d) * d + j];
                }
            }
            return out;
        }

        std::vector<double> rotate_cols_host(const std::vector<double>& a,
                                             int d, int k)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    out[i * d + j] = a[i * d + wrap(j + k, d)];
                }
            }
            return out;
        }

        std::vector<double> transpose_host(const std::vector<double>& a, int d)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    out[i * d + j] = a[j * d + i];
                }
            }
            return out;
        }

        std::vector<double> matmul_host(const std::vector<double>& a,
                                        const std::vector<double>& b, int d)
        {
            std::vector<double> out(static_cast<std::size_t>(d) * d, 0.0);
            for (int i = 0; i < d; i++)
            {
                for (int k = 0; k < d; k++)
                {
                    const double aik = a[i * d + k];
                    for (int j = 0; j < d; j++)
                    {
                        out[i * d + j] += aik * b[k * d + j];
                    }
                }
            }
            return out;
        }

        // -------------------------------------------------------------------
        // Llama3Operator
        // -------------------------------------------------------------------

        Llama3Operator::Llama3Operator(HEContext<Scheme::CKKS> context,
                                       HEEncoder<Scheme::CKKS>& encoder,
                                       double scale)
            : HEArithmeticOperator<Scheme::CKKS>(context, encoder),
              encoder_(encoder), primes_(context->get_key_modulus()),
              slot_count_(encoder.slot_count()), default_scale_(scale)
        {
            if (!(scale > 0.0))
            {
                throw std::invalid_argument("Scale must be positive");
            }

            // evaluate_poly consults scale_boot_ when deciding whether an
            // intermediate has grown enough to need rescaling, and outside
            // bootstrapping nothing sets it. Seeding it with the working scale
            // gives that test the meaning it has inside bootstrapping.
            //
            // generate_bootstrapping_params writes the same field, so a caller
            // that bootstraps at a scale other than the one it works at would
            // quietly change how every polynomial in this file rescales. Pass
            // it this same scale.
            scale_boot_ = scale;
        }

        double
        Llama3Operator::rescale_prime(const Ciphertext<Scheme::CKKS>& ct) const
        {
            const int level = ct.level();
            if (level < 0 || level >= static_cast<int>(primes_.size()))
            {
                throw std::invalid_argument(
                    "Ciphertext has no level left to rescale");
            }
            return static_cast<double>(primes_[level].value);
        }

        void Llama3Operator::add_same_scale(Ciphertext<Scheme::CKKS>& a,
                                            Ciphertext<Scheme::CKKS>& b,
                                            const char* context)
        {
            // Rescaling divides by a prime that is near but not equal to the
            // nominal scale, so tracked scales drift; what must not happen is
            // two operands drifting apart. A relative comparison catches that
            // without tripping on the drift itself.
            const double lhs = a.scale();
            const double rhs = b.scale();
            const double spread = std::abs(lhs - rhs);
            if (spread > 1e-9 * std::max(std::abs(lhs), std::abs(rhs)))
            {
                throw std::logic_error(
                    std::string("Refusing to add ciphertexts at different "
                                "scales in ") +
                    context + ": " + std::to_string(lhs) + " against " +
                    std::to_string(rhs));
            }
            add_inplace(a, b);
        }

        Plaintext<Scheme::CKKS>
        Llama3Operator::encode(const std::vector<double>& values, double scale,
                               int depth)
        {
            // HEEncoder always encodes at the top of the chain, but
            // multiply_plain checks the two operands agree on level, so the
            // plaintext has to come down to meet the ciphertext.
            Plaintext<Scheme::CKKS> plain(context_);
            encoder_.encode(plain, values, scale);
            for (int i = 0; i < depth; i++)
            {
                mod_drop_inplace(plain);
            }
            return plain;
        }

        void Llama3Operator::multiply_plaintext(Ciphertext<Scheme::CKKS>& ct,
                                                Plaintext<Scheme::CKKS>& plain)
        {
            if (plain.depth() > ct.depth())
            {
                throw std::invalid_argument(
                    "Plaintext is already below the ciphertext's level");
            }

            if (plain.depth() == ct.depth())
            {
                multiply_plain_inplace(ct, plain);
                return;
            }

            // Drop a copy so the caller keeps a plaintext it can reuse for the
            // next token or the next layer.
            Plaintext<Scheme::CKKS> dropped(context_);
            mod_drop(plain, dropped);
            for (int i = plain.depth() + 1; i < ct.depth(); i++)
            {
                mod_drop_inplace(dropped);
            }
            multiply_plain_inplace(ct, dropped);
        }

        void Llama3Operator::drop_to_depth(Ciphertext<Scheme::CKKS>& ct,
                                           int depth)
        {
            if (ct.depth() > depth)
            {
                throw std::invalid_argument(
                    "Ciphertext is already deeper than the requested depth");
            }
            while (ct.depth() < depth)
            {
                mod_drop_inplace(ct);
            }
        }

        void Llama3Operator::align_levels(Ciphertext<Scheme::CKKS>& a,
                                          Ciphertext<Scheme::CKKS>& b)
        {
            const int depth = std::max(a.depth(), b.depth());
            drop_to_depth(a, depth);
            drop_to_depth(b, depth);
        }

        void Llama3Operator::multiply_constant(Ciphertext<Scheme::CKKS>& ct,
                                               double c)
        {
            // Encoding the constant at exactly the prime the rescale removes
            // brings the result back to the input scale rather than to some
            // multiple of it.
            std::vector<double> values(slot_count_, c);
            Plaintext<Scheme::CKKS> plain =
                encode(values, rescale_prime(ct), ct.depth());
            multiply_plain_inplace(ct, plain);
            rescale_inplace(ct);
        }

        void Llama3Operator::multiply_vector(Ciphertext<Scheme::CKKS>& ct,
                                             const std::vector<double>& values)
        {
            if (static_cast<int>(values.size()) != slot_count_)
            {
                throw std::invalid_argument(
                    "Slot vector must hold exactly slot_count() entries");
            }
            Plaintext<Scheme::CKKS> plain =
                encode(values, rescale_prime(ct), ct.depth());
            multiply_plain_inplace(ct, plain);
            rescale_inplace(ct);
        }

        void Llama3Operator::add_constant(Ciphertext<Scheme::CKKS>& ct,
                                          double c)
        {
            // Scales the constant by the ciphertext's own scale_, so this
            // stays correct however far the scale has drifted.
            add_plain_v2(ct, Complex64(c, 0.0), ct);
        }

        void Llama3Operator::add_vector(
            std::vector<Ciphertext<Scheme::CKKS>>& cts,
            const std::vector<double>& values)
        {
            if (cts.empty())
            {
                return;
            }
            if (static_cast<int>(values.size()) != slot_count_)
            {
                throw std::invalid_argument(
                    "Slot vector must hold exactly slot_count() entries");
            }
            for (const auto& ct : cts)
            {
                // An addition with a mismatched scale is silently a weighted
                // sum, so the batch must agree before one plaintext serves
                // it all.
                if (ct.scale() != cts.front().scale() ||
                    ct.depth() != cts.front().depth())
                {
                    throw std::invalid_argument(
                        "add_vector needs every ciphertext at one level and "
                        "one scale");
                }
            }
            // Encoded at the ciphertexts' own scale, not the rescale prime:
            // an addition consumes no level and must not change the scale.
            Plaintext<Scheme::CKKS> plain =
                encode(values, cts.front().scale(), cts.front().depth());
            for (auto& ct : cts)
            {
                add_plain_inplace(ct, plain);
            }
        }

        void Llama3Operator::square(Ciphertext<Scheme::CKKS>& ct,
                                    Relinkey<Scheme::CKKS>& relin_key)
        {
            Ciphertext<Scheme::CKKS> copy = ct;
            Ciphertext<Scheme::CKKS> out(context_);
            multiply(ct, copy, out);
            relinearize_inplace(out, relin_key);
            rescale_inplace(out);
            ct = out;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::multiply_and_rescale(Ciphertext<Scheme::CKKS>& a,
                                             Ciphertext<Scheme::CKKS>& b,
                                             Relinkey<Scheme::CKKS>& relin_key)
        {
            Ciphertext<Scheme::CKKS> lhs = a;
            Ciphertext<Scheme::CKKS> rhs = b;
            align_levels(lhs, rhs);

            Ciphertext<Scheme::CKKS> out(context_);
            multiply(lhs, rhs, out);
            relinearize_inplace(out, relin_key);
            rescale_inplace(out);
            return out;
        }

        void Llama3Operator::match_scale(Ciphertext<Scheme::CKKS>& ct,
                                         double target)
        {
            if (!(target > 0.0))
            {
                throw std::invalid_argument("Target scale must be positive");
            }

            // A product by a plaintext at scale s takes the ciphertext to
            // ct.scale() * s, and the rescale then divides by the prime, so
            // encoding the constant one at target * prime / ct.scale() lands
            // exactly on the target.
            const double prime = rescale_prime(ct);
            const std::vector<double> ones(slot_count_, 1.0);
            Plaintext<Scheme::CKKS> plain =
                encode(ones, target * prime / ct.scale(), ct.depth());
            multiply_plain_inplace(ct, plain);
            rescale_inplace(ct);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::residual_add(Ciphertext<Scheme::CKKS>& x,
                                     Ciphertext<Scheme::CKKS>& sublayer)
        {
            Ciphertext<Scheme::CKKS> skip = x;
            Ciphertext<Scheme::CKKS> out = sublayer;
            align_levels(skip, out);

            // match_scale spends a level, so the sublayer output follows it
            // down; the drop leaves its scale alone, which is the scale both
            // are now on.
            match_scale(skip, out.scale());
            drop_to_depth(out, skip.depth());

            add_same_scale(out, skip, "residual");
            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::residual_add(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            std::vector<Ciphertext<Scheme::CKKS>>& sublayer)
        {
            if (x.size() != sublayer.size())
            {
                throw std::invalid_argument(
                    "A residual needs the sublayer to come back on as many "
                    "blocks as went into it");
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(x.size());
            for (std::size_t i = 0; i < x.size(); i++)
            {
                out.push_back(residual_add(x[i], sublayer[i]));
            }
            return out;
        }

        void Llama3Operator::accumulate_masked(
            Ciphertext<Scheme::CKKS>& acc, bool& started,
            Ciphertext<Scheme::CKKS>& ct, const std::vector<double>& values,
            double plain_scale, const char* context)
        {
            // Deriving and encoding the mask is host work in a device
            // pipeline, and every diagonal of every call pays it, so it is
            // separated from the ciphertext product it feeds. This is the same
            // cost pcmm.weight_encode exposes, in the primitives that mask
            // rather than the one that multiplies by a weight.
            Plaintext<Scheme::CKKS> plain = [&]
            {
                SuffixRange _r(context, "mask_encode");
                return encode(values, plain_scale, ct.depth());
            }();

            SuffixRange _r(context, "mask_multiply_add");
            Ciphertext<Scheme::CKKS> term(context_);
            multiply_plain(ct, plain, term);

            if (!started)
            {
                acc = term;
                started = true;
            }
            else
            {
                add_same_scale(acc, term, context);
            }
        }

        // -------------------------------------------------------------------
        // Slot reductions
        // -------------------------------------------------------------------

        std::vector<int> Llama3Operator::strided_rotation_indices(int stride,
                                                                  int count)
        {
            if (stride <= 0 || !is_power_of_two(count))
            {
                throw std::invalid_argument(
                    "A strided reduction needs a positive stride and a "
                    "power-of-two count");
            }

            std::vector<int> indices;
            for (int t = 1; t < count; t <<= 1)
            {
                indices.push_back(stride * t);
            }
            return indices;
        }

        std::vector<int> Llama3Operator::blocked_rotation_indices(int span)
        {
            if (!is_power_of_two(span))
            {
                throw std::invalid_argument("Block span must be a power of "
                                            "two");
            }

            std::vector<int> indices;
            for (int s = 1; s < span; s <<= 1)
            {
                indices.push_back(s);  // window sum
                indices.push_back(-s); // fan the masked totals back out
            }
            return indices;
        }

        void Llama3Operator::sum_strided(Ciphertext<Scheme::CKKS>& ct,
                                         int stride, int count,
                                         Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (stride * count != slot_count_)
            {
                throw std::invalid_argument(
                    "A strided reduction needs stride * count to fill the slot "
                    "vector, otherwise the rotation does not wrap onto the "
                    "reduced axis");
            }
            if (!is_power_of_two(count))
            {
                throw std::invalid_argument("Reduced count must be a power of "
                                            "two");
            }

            Range _r("sum_strided");

            for (int t = 1; t < count; t <<= 1)
            {
                Ciphertext<Scheme::CKKS> shifted(context_);
                rotate_rows(ct, shifted, galois_key, stride * t);
                add_same_scale(ct, shifted, "sum_strided");
            }
        }

        void Llama3Operator::sum_blocked(Ciphertext<Scheme::CKKS>& ct, int span,
                                         Galoiskey<Scheme::CKKS>& galois_key,
                                         double mask_scale)
        {
            if (!is_power_of_two(span) || span > slot_count_ ||
                slot_count_ % span != 0)
            {
                throw std::invalid_argument(
                    "Block span must be a power of two dividing the slot "
                    "count");
            }
            if (span == 1)
            {
                // There is no mask here to carry a folded constant on, so a
                // caller that asked for one would silently not get it.
                if (mask_scale != 1.0)
                {
                    throw std::invalid_argument(
                        "A span of one does not mask, so there is nothing for "
                        "a folded constant to ride on");
                }
                return;
            }

            Range _r_sum("sum_blocked");

            // Rotate-and-add gives slot p the window sum over p .. p+span-1,
            // which is the block total only where p is a multiple of span.
            {
                Range _r("sum_blocked.window");
                for (int s = 1; s < span; s <<= 1)
                {
                    Ciphertext<Scheme::CKKS> shifted(context_);
                    rotate_rows(ct, shifted, galois_key, s);
                    add_same_scale(ct, shifted, "sum_blocked window");
                }
            }

            // Keep those positions and discard the rest. The kept entries
            // carry mask_scale rather than one, which costs nothing: this
            // product is being paid for either way.
            {
                Range _r("sum_blocked.mask");
                std::vector<double> mask(slot_count_, 0.0);
                for (int p = 0; p < slot_count_; p += span)
                {
                    mask[p] = mask_scale;
                }
                multiply_vector(ct, mask);
            }

            // Fan each surviving total across its own block. Shifting right by
            // less than span cannot cross into the next block, because every
            // other slot of the block is zero.
            {
                Range _r("sum_blocked.fan_out");
                for (int s = 1; s < span; s <<= 1)
                {
                    Ciphertext<Scheme::CKKS> shifted(context_);
                    rotate_rows(ct, shifted, galois_key, -s);
                    add_same_scale(ct, shifted, "sum_blocked fan-out");
                }
            }
        }

        // -------------------------------------------------------------------
        // Polynomial primitives
        // -------------------------------------------------------------------

        double Llama3Operator::domain_scale(double a, double b)
        {
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }
            return 2.0 / (b - a);
        }

        double Llama3Operator::domain_shift(double a, double b)
        {
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }
            return -(a + b) / (b - a);
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::evaluate_chebyshev(
            Ciphertext<Scheme::CKKS>& ct, const std::vector<double>& coeffs,
            double a, double b, Relinkey<Scheme::CKKS>& relin_key,
            bool pre_scaled)
        {
            if (coeffs.empty())
            {
                throw std::invalid_argument("Chebyshev series is empty");
            }
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }

            Range _r_cheb("chebyshev");

            Ciphertext<Scheme::CKKS> t = ct;
            if (a != -1.0 || b != 1.0)
            {
                // T_k is defined on [-1, 1], so the argument is mapped there
                // first. The multiplier is the whole cost of that map; the
                // shift is an addition and free. When the caller has already
                // carried the multiplier on a plaintext product of its own,
                // only the free half is left and the fit costs nothing to
                // place on its interval.
                Range _r("chebyshev.affine_map");
                if (!pre_scaled)
                {
                    multiply_constant(t, domain_scale(a, b));
                }
                add_constant(t, domain_shift(a, b));
            }

            // evaluate_poly starts the recursion at level - ceil(log2(degree))
            // + 1 and uses that as a kernel grid dimension, so too short a
            // chain launches a kernel with a nonpositive extent instead of
            // reporting anything. Say what actually went wrong.
            const int degree = static_cast<int>(coeffs.size()) - 1;
            const int needed =
                degree < 1
                    ? 0
                    : static_cast<int>(std::ceil(std::log2(degree))) - 1;
            if (t.level() < needed)
            {
                throw std::invalid_argument(
                    "Not enough levels left for a degree " +
                    std::to_string(degree) + " Chebyshev evaluation: it needs " +
                    std::to_string(needed) + " beyond the affine map and the "
                    "ciphertext has " + std::to_string(t.level()));
            }

            std::vector<Complex64> complex_coeffs(coeffs.size());
            for (std::size_t i = 0; i < coeffs.size(); i++)
            {
                complex_coeffs[i] = Complex64(coeffs[i], 0.0);
            }

            Polynomial poly(static_cast<int>(coeffs.size()) - 1, complex_coeffs,
                            true, PolyType::CHEBYSHEV, a, b);

            // The Paterson-Stockmeyer evaluation itself: the baby-step powers
            // of T and the giant-step recursion over them, all inside
            // evaluate_poly, so this is where a fit's levels are actually paid.
            Range _r("chebyshev.evaluate");
            return evaluate_poly(t, t.scale(), poly, relin_key,
                                 ExecutionOptions());
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::evaluate_function(Ciphertext<Scheme::CKKS>& ct,
                                          const std::function<double(double)>& f,
                                          double a, double b, int degree,
                                          Relinkey<Scheme::CKKS>& relin_key,
                                          bool pre_scaled)
        {
            return evaluate_chebyshev(ct, chebyshev_coefficients(f, a, b, degree),
                                      a, b, relin_key, pre_scaled);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::inverse_sqrt(Ciphertext<Scheme::CKKS>& ct, double lo,
                                     double hi, int degree,
                                     int newton_iterations,
                                     Relinkey<Scheme::CKKS>& relin_key)
        {
            if (!(lo > 0.0))
            {
                throw std::invalid_argument(
                    "1/sqrt(x) needs a strictly positive lower bound");
            }

            Range _r_isqrt("inverse_sqrt");

            Ciphertext<Scheme::CKKS> y = evaluate_function(
                ct, [](double x) { return 1.0 / std::sqrt(x); }, lo, hi, degree,
                relin_key);

            if (newton_iterations <= 0)
            {
                return y;
            }

            Range _r_newton("inverse_sqrt.newton");

            // y <- y (3 - x y^2) / 2. Halving x once up front turns the three
            // halvings the loop would otherwise need into none, so a step
            // costs three levels: y^2, x/2 times it, and the final product.
            Ciphertext<Scheme::CKKS> half_x = ct;
            multiply_constant(half_x, 0.5);

            for (int iteration = 0; iteration < newton_iterations; iteration++)
            {
                Ciphertext<Scheme::CKKS> y_squared = y;
                square(y_squared, relin_key);

                Ciphertext<Scheme::CKKS> scaled = half_x;
                Ciphertext<Scheme::CKKS> correction =
                    multiply_and_rescale(scaled, y_squared, relin_key);

                negate_inplace(correction);
                add_constant(correction, 1.5);

                y = multiply_and_rescale(y, correction, relin_key);
            }

            return y;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::inverse(Ciphertext<Scheme::CKKS>& ct, double lo,
                                double hi, int degree, int newton_iterations,
                                Relinkey<Scheme::CKKS>& relin_key,
                                bool pre_scaled, double gain)
        {
            if (!(lo > 0.0))
            {
                throw std::invalid_argument(
                    "1/x needs a strictly positive lower bound");
            }
            if (newton_iterations > 0 && (pre_scaled || gain != 1.0))
            {
                // The step below multiplies by ct itself and converges on
                // 1/x, so it wants the argument unscaled and the answer
                // unweighted. Either fold or refine, not both.
                throw std::invalid_argument(
                    "A Newton step needs the argument unscaled and the fit "
                    "unweighted, so it cannot be had with a folded domain map "
                    "or a gain");
            }

            Range _r_inv("inverse");

            Ciphertext<Scheme::CKKS> y = evaluate_function(
                ct, [gain](double x) { return gain / x; }, lo, hi, degree,
                relin_key, pre_scaled);

            // y <- y (2 - x y), two levels a step.
            Range _r_newton("inverse.newton");
            for (int iteration = 0; iteration < newton_iterations; iteration++)
            {
                Ciphertext<Scheme::CKKS> x = ct;
                Ciphertext<Scheme::CKKS> correction =
                    multiply_and_rescale(x, y, relin_key);

                negate_inplace(correction);
                add_constant(correction, 2.0);

                y = multiply_and_rescale(y, correction, relin_key);
            }

            return y;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::silu(Ciphertext<Scheme::CKKS>& ct, double bound,
                             int degree, Relinkey<Scheme::CKKS>& relin_key,
                             bool pre_scaled, double gain)
        {
            if (gain == 0.0)
            {
                throw std::invalid_argument(
                    "SiLU gain must not be zero: nothing downstream can "
                    "divide it back out");
            }

            Range _r("silu");
            return evaluate_function(
                ct,
                [gain](double x) { return gain * x / (1.0 + std::exp(-x)); },
                -bound, bound, degree, relin_key, pre_scaled);
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::exp_scaled_negative(
            Ciphertext<Scheme::CKKS>& ct, double bound, int squarings,
            int degree, Relinkey<Scheme::CKKS>& relin_key, bool pre_scaled)
        {
            if (!(bound > 0.0))
            {
                throw std::invalid_argument("Bound must be positive");
            }
            if (squarings < 0)
            {
                throw std::invalid_argument(
                    "Squaring count must not be negative");
            }

            Range _r("exp_scaled");

            // Approximating exp(x / 2^k) rather than exp(x) folds the scaling
            // step of the SoftMax algorithm into the fit, so it is free.
            const double divisor = std::pow(2.0, squarings);
            return evaluate_function(
                ct, [divisor](double x) { return std::exp(x / divisor); },
                -bound, 0.0, degree, relin_key, pre_scaled);
        }

        // -------------------------------------------------------------------
        // Llama-3 layers
        // -------------------------------------------------------------------

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::rms_norm(
            std::vector<Ciphertext<Scheme::CKKS>>& in,
            std::vector<Plaintext<Scheme::CKKS>>& weights,
            const RMSNormConfig& config, Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            if (in.empty())
            {
                throw std::invalid_argument("RMSNorm needs at least one input");
            }
            const bool aux_refresh = config.refresh_sum;
            if (aux_refresh && boot_key == nullptr)
            {
                throw std::invalid_argument(
                    "Refreshing the RMSNorm sum needs the boot Galois key");
            }
            if (aux_refresh && config.newton_iterations > 0)
            {
                throw std::invalid_argument(
                    "A Newton step refines against the unmapped argument, "
                    "and the refreshed sum arrives mapped: raise the fit "
                    "degree instead, which the auxiliary track makes free");
            }
            if (!weights.empty() && weights.size() != in.size())
            {
                throw std::invalid_argument(
                    "RMSNorm needs one weight plaintext per input, or none");
            }
            const int token_blocks =
                config.token_blocks > 0 ? config.token_blocks : 1;
            if (static_cast<int>(in.size()) % token_blocks != 0)
            {
                throw std::invalid_argument(
                    "RMSNorm needs the same channel blocks in every token "
                    "block");
            }
            const int channel_blocks =
                static_cast<int>(in.size()) / token_blocks;

            // The reduction covers count channels in each of the channel
            // blocks, and only the last one may be partly padding, so the true
            // channel count is pinned between those two bounds. Getting this
            // wrong scales every output by a constant and nothing else
            // complains.
            const int reduced = config.count * channel_blocks;
            if (config.blocked_span > 0)
            {
                // A blocked reduction spreads the padding across every
                // ciphertext rather than confining it to the last one, so the
                // tight window below does not apply. A padded coordinate is
                // zero and contributes nothing to the sum of squares, so all
                // that is needed is that the mean divides by no more channels
                // than the reduction actually covers.
                if (config.channels <= 0 || config.channels > reduced)
                {
                    throw std::invalid_argument(
                        "RMSNorm's channel count must be positive and at most "
                        "the number of channels the blocked reduction covers");
                }
            }
            else if (config.channels <= reduced - config.count ||
                     config.channels > reduced)
            {
                throw std::invalid_argument(
                    "RMSNorm's channel count must lie in (count * (channel "
                    "blocks - 1), count * channel blocks]: the reduction "
                    "covers count channels per block and only the last block "
                    "may be padded");
            }

            for (std::size_t i = 1; i < in.size(); i++)
            {
                // The squares are added together, so a mismatch here is a
                // silently wrong sum rather than an error from the library.
                if (in[i].depth() != in[0].depth() ||
                    in[i].scale() != in[0].scale())
                {
                    throw std::invalid_argument(
                        "RMSNorm needs every input at one level and one scale");
                }
            }
            if (!(config.sum_hi > config.sum_lo) || !(config.sum_lo > 0.0))
            {
                throw std::invalid_argument(
                    "RMSNorm needs a positive range for the summed square");
            }
            if (config.output_scale != 1.0 && config.newton_iterations > 0)
            {
                throw std::invalid_argument(
                    "RMSNorm's output_scale rides on the fitted 1/sqrt, and a "
                    "Newton step refines toward 1/sqrt itself: the two cannot "
                    "both be had");
            }

            const double lo =
                config.sum_lo / static_cast<double>(config.channels) +
                config.eps;
            const double hi =
                config.sum_hi / static_cast<double>(config.channels) +
                config.eps;

            // Forming the mean is a plaintext product and a rescale, and the
            // fit does not need it formed: it can be taken over the summed
            // square instead. A Newton step does need it, so the two cannot
            // both be had.
            const bool fold_mean =
                config.fold_mean_into_fit && config.newton_iterations <= 0;

            // The same argument one step further along. The blocked reduction
            // already spends a level on a mask, and the fit that follows
            // spends another placing its argument on [-1, 1]; the first can
            // carry the second. The strided reduction has no mask, so there
            // is nothing to fold into and nothing is claimed.
            const bool fold_affine = config.fold_affine_into_mask &&
                                     config.blocked_span > 0 &&
                                     config.newton_iterations <= 0;
            const double fit_lo = fold_mean ? config.sum_lo : lo;
            const double fit_hi = fold_mean ? config.sum_hi : hi;

            // When the mean is NOT folded into the fit, the division by the
            // channel count sits between the mask and the fit, so the mask
            // carries it too and what is left of forming the mean is the
            // addition of eps -- scaled to match, and free either way.
            const double mean_divisor =
                fold_mean ? 1.0 : 1.0 / static_cast<double>(config.channels);
            const double mask_scale =
                fold_affine
                    ? Llama3Operator::domain_scale(fit_lo, fit_hi) * mean_divisor
                    : 1.0;

            std::vector<Ciphertext<Scheme::CKKS>> out(
                in.size(), Ciphertext<Scheme::CKKS>(context_));

            Range _r_norm("rms_norm");

            // Each token block normalises over its own channels, so the two
            // never mix and the sequence length only says how to walk the
            // inputs.
            for (int s = 0; s < token_blocks; s++)
            {
                // The channels of a token may be split over several
                // ciphertexts, so the squares are accumulated before the
                // reduction and the mean covers every channel.
                Ciphertext<Scheme::CKKS> total = in[s];
                {
                    Range _r("rms_norm.sum_of_squares");
                    square(total, relin_key);
                    for (int j = 1; j < channel_blocks; j++)
                    {
                        Ciphertext<Scheme::CKKS> term =
                            in[j * token_blocks + s];
                        square(term, relin_key);
                        add_same_scale(total, term, "rms_norm channel sum");
                    }

                    // The channels of one ciphertext are either the slow slot
                    // axis, where the reduction is free, or the fast one, where
                    // it costs a mask. Which of the two is a property of the
                    // encoding and not of RMSNorm.
                    if (config.blocked_span > 0)
                    {
                        sum_blocked(total, config.blocked_span, galois_key,
                                    mask_scale);
                    }
                    else
                    {
                        sum_strided(total, config.stride, config.count,
                                    galois_key);
                    }

                    // mean + eps. The addition is free and the division is
                    // not, so it is skipped entirely when the fit below can
                    // carry it -- or when the mask above already has.
                    if (!fold_mean)
                    {
                        if (!fold_affine)
                        {
                            multiply_constant(total, mean_divisor);
                            add_constant(total, config.eps);
                        }
                        else
                        {
                            // total already holds domain_scale * s / channels,
                            // so eps joins it scaled the same way and the sum
                            // is domain_scale * (mean + eps).
                            add_constant(total,
                                         Llama3Operator::domain_scale(lo, hi) *
                                             config.eps);
                        }
                    }
                }

                Ciphertext<Scheme::CKKS> scale_factor;
                {
                    Range _r("rms_norm.inverse_sqrt");
                    if (aux_refresh)
                    {
                        // The same narrow-track refresh the SoftMax
                        // denominator takes: the sum is one ciphertext per
                        // token block however many channel blocks feed it, so
                        // it is refreshed alone and the fit's levels are paid
                        // above the wide track. Completing the fit's domain
                        // map is the whole preparation, and it is what puts
                        // the values in the [-1, 1] a refresh assumes.
                        const double gain = config.output_scale;
                        if (!fold_affine)
                        {
                            multiply_constant(total,
                                              Llama3Operator::domain_scale(
                                                  fit_lo, fit_hi));
                        }
                        add_constant(total, Llama3Operator::domain_shift(
                                                fit_lo, fit_hi));
                        total = bootstrap(total, *boot_key, relin_key);

                        std::function<double(double)> f;
                        if (fold_mean)
                        {
                            const double channels =
                                static_cast<double>(config.channels);
                            const double eps = config.eps;
                            f = [channels, eps, gain](double s) {
                                return gain / std::sqrt(s / channels + eps);
                            };
                        }
                        else
                        {
                            f = [gain](double x)
                            { return gain / std::sqrt(x); };
                        }
                        scale_factor = evaluate_chebyshev(
                            total,
                            chebyshev_coefficients(f, fit_lo, fit_hi,
                                                   config.degree),
                            -1.0, 1.0, relin_key);
                    }
                    else if (fold_mean)
                    {
                        // The same value of the same ciphertext, fitted over
                        // the summed square instead of over the mean. One
                        // level cheaper, and the level is the whole reason.
                        const double channels =
                            static_cast<double>(config.channels);
                        const double eps = config.eps;
                        const double gain = config.output_scale;
                        scale_factor = evaluate_function(
                            total,
                            [channels, eps, gain](double s) {
                                return gain / std::sqrt(s / channels + eps);
                            },
                            config.sum_lo, config.sum_hi, config.degree,
                            relin_key, fold_affine);
                    }
                    else if (fold_affine)
                    {
                        // No Newton step is possible here -- fold_affine says
                        // so -- and without one inverse_sqrt is the fit alone.
                        const double gain = config.output_scale;
                        scale_factor = evaluate_function(
                            total,
                            [gain](double x) { return gain / std::sqrt(x); },
                            lo, hi, config.degree, relin_key, true);
                    }
                    else if (config.output_scale != 1.0)
                    {
                        // The guard above pinned newton_iterations to 0, so
                        // inverse_sqrt would be the bare fit anyway; this is
                        // the same fit with the gain in its coefficients.
                        const double gain = config.output_scale;
                        scale_factor = evaluate_function(
                            total,
                            [gain](double x) { return gain / std::sqrt(x); },
                            lo, hi, config.degree, relin_key, false);
                    }
                    else
                    {
                        scale_factor =
                            inverse_sqrt(total, lo, hi, config.degree,
                                         config.newton_iterations, relin_key);
                    }
                }

                Range _r("rms_norm.rescale_channels");
                for (int j = 0; j < channel_blocks; j++)
                {
                    const std::size_t at = static_cast<std::size_t>(j) *
                                               token_blocks + s;
                    Ciphertext<Scheme::CKKS> normalised = multiply_and_rescale(
                        in[at], scale_factor, relin_key);

                    if (!weights.empty())
                    {
                        multiply_plaintext(normalised, weights[at]);
                        rescale_inplace(normalised);
                    }

                    out[at] = normalised;
                }
            }

            return out;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::softmax(Ciphertext<Scheme::CKKS>& ct,
                                const SoftmaxConfig& config,
                                Galoiskey<Scheme::CKKS>& galois_key,
                                Relinkey<Scheme::CKKS>& relin_key)
        {
            const std::vector<double> none;
            return softmax(ct, config, none, galois_key, relin_key);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::softmax(Ciphertext<Scheme::CKKS>& ct,
                                const SoftmaxConfig& config,
                                const std::vector<double>& mask,
                                Galoiskey<Scheme::CKKS>& galois_key,
                                Relinkey<Scheme::CKKS>& relin_key)
        {
            std::vector<Ciphertext<Scheme::CKKS>> parts{ct};
            std::vector<std::vector<double>> masks;
            if (!mask.empty())
            {
                masks.push_back(mask);
            }
            return softmax(parts, config, masks, galois_key, relin_key)
                .front();
        }

        void Llama3Operator::set_mask_plain_capacity(std::size_t entries)
        {
            mask_plain_capacity_ = entries;
            while (mask_plain_.size() > mask_plain_capacity_)
            {
                mask_plain_.erase(mask_plain_.begin());
            }
        }

        void Llama3Operator::multiply_mask(Ciphertext<Scheme::CKKS>& ct,
                                           const std::vector<double>& values,
                                           double weight, int mask_id)
        {
            if (static_cast<int>(values.size()) != slot_count_)
            {
                throw std::invalid_argument(
                    "A SoftMax mask must hold exactly slot_count() entries");
            }
            const double plain_scale = rescale_prime(ct);

            // The weighted mask, built only when it is not already encoded.
            auto weighted = [&]
            {
                if (weight == 1.0)
                {
                    return values;
                }
                std::vector<double> scaled = values;
                for (auto& entry : scaled)
                {
                    entry *= weight;
                }
                return scaled;
            };

            if (mask_id < 0 || mask_plain_capacity_ == 0)
            {
                Plaintext<Scheme::CKKS> plain =
                    encode(weighted(), plain_scale, ct.depth());
                multiply_plain_inplace(ct, plain);
                rescale_inplace(ct);
                return;
            }

            // Everything an encoding depends on, and nothing else.
            // rescale_prime returns a modulus, so its double is an exact
            // integer and safe to key on; the weight is keyed by its exact
            // bits, so a differently calibrated fold cannot collide with this
            // entry.
            uint64_t weight_bits = 0;
            std::memcpy(&weight_bits, &weight, sizeof(weight_bits));
            const auto key = std::make_tuple(mask_id, weight_bits, ct.depth(),
                                             static_cast<uint64_t>(plain_scale));
            auto found = mask_plain_.find(key);
            if (found == mask_plain_.end())
            {
                while (mask_plain_.size() >= mask_plain_capacity_)
                {
                    // Whole-entry eviction. The next call that wants this
                    // (mask, weight, level) pays the encode again and gets the
                    // same plaintext, so the only thing at stake is time.
                    mask_plain_.erase(mask_plain_.begin());
                }
                found = mask_plain_
                            .emplace(key, encode(weighted(), plain_scale,
                                                 ct.depth()))
                            .first;
            }
            else
            {
                mask_plain_hits_++;
            }

            multiply_plain_inplace(ct, found->second);
            rescale_inplace(ct);
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::softmax(
            std::vector<Ciphertext<Scheme::CKKS>>& parts,
            const SoftmaxConfig& config,
            const std::vector<std::vector<double>>& masks,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            const std::vector<int> unnamed;
            return softmax(parts, config, masks, unnamed, galois_key, relin_key,
                           boot_key);
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::softmax(
            std::vector<Ciphertext<Scheme::CKKS>>& parts,
            const SoftmaxConfig& config,
            const std::vector<std::vector<double>>& masks,
            const std::vector<int>& mask_ids,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            if (!mask_ids.empty() && mask_ids.size() != parts.size())
            {
                // Silently ignoring a short list would cache under the wrong
                // name, which is the one failure mode of a caller-keyed cache
                // and is not detectable downstream.
                throw std::invalid_argument(
                    "A named SoftMax mask list needs one id per part, or none "
                    "at all");
            }
            if (parts.empty())
            {
                throw std::invalid_argument(
                    "SoftMax needs at least one ciphertext to reduce over");
            }
            const bool aux_refresh = config.refresh_denominator;
            if (aux_refresh && boot_key == nullptr)
            {
                // Falling back silently would put the fit's levels back on
                // the wide track and change the schedule the caller sized
                // the chain for.
                throw std::invalid_argument(
                    "Refreshing the SoftMax denominator needs the boot "
                    "Galois key");
            }
            if (aux_refresh && config.inverse_newton > 0)
            {
                throw std::invalid_argument(
                    "A Newton step refines against the unmapped argument, "
                    "and the refreshed denominator arrives mapped: raise the "
                    "fit degree instead, which the auxiliary track makes "
                    "free");
            }
            if (!masks.empty() && masks.size() != parts.size())
            {
                throw std::invalid_argument(
                    "A masked SoftMax needs one mask per part, or none at all");
            }
            for (const auto& mask : masks)
            {
                if (!mask.empty() &&
                    static_cast<int>(mask.size()) != slot_count_)
                {
                    throw std::invalid_argument(
                        "A SoftMax mask must hold exactly slot_count() "
                        "entries");
                }
            }
            // The parts are added together below, so a mismatch here would be
            // a silently wrong denominator rather than an error.
            require_uniform(parts, "SoftMax input");
            if (config.count < 1 || !is_power_of_two(config.count))
            {
                throw std::invalid_argument(
                    "The SoftMax length held inside one ciphertext must be a "
                    "power of two");
            }
            // One is allowed, and it is not a degenerate case: a matrix
            // encryption puts each coordinate of the reduced axis in its own
            // ciphertext, so the whole axis runs across the parts and there is
            // nothing left to reduce inside one. The reduction below is then
            // the slot-wise sum alone, which costs no rotation at all.
            if (static_cast<double>(config.count) * parts.size() <= 1.0)
            {
                throw std::invalid_argument(
                    "SoftMax length must be above one, counting every part");
            }
            if (!(config.bound > 0.0))
            {
                throw std::invalid_argument(
                    "SoftMax needs the input range [-bound, 0]");
            }
            if (config.iterations < 1)
            {
                throw std::invalid_argument(
                    "SoftMax needs at least one normalise-and-square round");
            }
            if (!config.strided && slot_count_ % config.count != 0)
            {
                throw std::invalid_argument(
                    "A blocked SoftMax needs its length to divide the slot "
                    "count, so that every block is whole");
            }

            // The reduced axis runs across the parts, so the SoftMax length is
            // the whole of it and that is what the fitted ranges below and the
            // caller's mask weights are taken against.
            const double d =
                static_cast<double>(config.count) * parts.size();

            // Section 4.3. Round zero's denominator is a sum of exponentials
            // and its range is a measurement; every round after it sees
            // coordinates that already sum to one, so its denominator lies in
            // [1/d, 1] and what calibration supplies is how far up that the
            // sharpest row actually reaches.
            auto round_range = [&](int round, double& lo, double& hi)
            {
                if (round == 0)
                {
                    if (config.sum_hi > config.sum_lo && config.sum_lo > 0.0)
                    {
                        lo = config.sum_lo;
                        hi = config.sum_hi;
                        return;
                    }
                    // Nothing measured: every coordinate might be at either
                    // end at once. A reciprocal over this is the thing the
                    // paper is telling us not to fit.
                    const double smallest = std::exp(
                        -2.0 * config.bound / std::pow(2.0, config.iterations));
                    lo = d * smallest * 0.5;
                    hi = d * 1.5;
                    return;
                }
                const double top =
                    config.concentration > 0.0
                        ? std::min(config.concentration, d)
                        : d;
                lo = 0.5 / d;
                hi = 1.5 * top / d;
            };

            // The domain map of each round's fit, carried upstream. Round
            // zero's rides on the mask; a later round's rides on the fit of
            // the round before, where it is a change of coefficients and free.
            // Both are constant along the reduced axis, which is exactly the
            // condition under which the normalising rounds cancel a factor.
            const bool have_masks = [&] {
                if (masks.size() != parts.size())
                {
                    return false;
                }
                for (const auto& mask : masks)
                {
                    if (mask.empty())
                    {
                        return false;
                    }
                }
                return true;
            }();
            const bool fold_affine = config.fold_affine_into_mask &&
                                     config.inverse_newton <= 0;
            if (fold_affine && !have_masks)
            {
                throw std::invalid_argument(
                    "A folded domain map rides on the mask of every part, so "
                    "every part needs one");
            }

            std::vector<double> domain(config.iterations, 1.0);
            if (fold_affine)
            {
                for (int round = 0; round < config.iterations; round++)
                {
                    double lo = 0.0;
                    double hi = 0.0;
                    round_range(round, lo, hi);
                    domain[round] = Llama3Operator::domain_scale(lo, hi);
                }
            }

            std::vector<Ciphertext<Scheme::CKKS>> y;
            y.reserve(parts.size());
            {
                Range _r("softmax.exp");
                for (std::size_t p = 0; p < parts.size(); p++)
                {
                    y.push_back(exp_scaled_negative(
                        parts[p], config.bound, config.iterations,
                        config.exp_degree, relin_key, config.pre_scaled_input));

                    // Masking the exponentials rather than the scores removes a
                    // coordinate from the numerator and from the sum at once.
                    // The rounds below normalise, so a mask weight that is
                    // constant along the reduced axis cancels and only the
                    // pattern of zeros survives.
                    //
                    // Round zero's denominator is the sum of the SQUARES of
                    // these, so the map it wants rides on the mask as its own
                    // square root.
                    if (!masks.empty() && !masks[p].empty())
                    {
                        // Round zero's denominator is the sum of the SQUARES
                        // of these, so the map it wants rides on the mask as
                        // its own square root. The weight goes into the cache
                        // key rather than into the values, so the same mask at
                        // two calibrations is two entries and never one.
                        const double weight =
                            fold_affine ? std::sqrt(domain[0]) : 1.0;
                        const int id = mask_ids.empty() ? -1 : mask_ids[p];
                        multiply_mask(y.back(), masks[p], weight, id);
                    }
                }
            }
            if (debug_trace)
            {
                debug_trace("softmax.exp_masked", y);
            }

            for (int round = 0; round < config.iterations; round++)
            {
                Range _r_round("softmax.normalise_round");
                // (y_i / ||y||_2)^2 is y_i^2 / sum_j y_j^2, so one squaring
                // serves both the numerator and the sum.
                std::vector<Ciphertext<Scheme::CKKS>> y_squared;
                y_squared.reserve(y.size());
                for (auto& part : y)
                {
                    Ciphertext<Scheme::CKKS> squared = part;
                    square(squared, relin_key);
                    y_squared.push_back(squared);
                }

                // The parts hold different stretches of one axis in the same
                // slots, and the reduction is linear, so summing them slot-wise
                // and reducing once gives the same total as reducing each. One
                // reduction and one reciprocal serve the whole axis however
                // many ciphertexts it was cut into.
                Ciphertext<Scheme::CKKS> total = y_squared[0];
                for (std::size_t p = 1; p < y_squared.size(); p++)
                {
                    add_same_scale(total, y_squared[p],
                                   "SoftMax denominator");
                }

                if (config.strided)
                {
                    sum_strided(total, config.stride, config.count, galois_key);
                }
                else
                {
                    sum_blocked(total, config.count, galois_key);
                }
                if (debug_trace)
                {
                    std::vector<Ciphertext<Scheme::CKKS>> one{total};
                    debug_trace("softmax.denominator", one);
                }

                // Before the first round every coordinate sits in
                // [exp(-bound / 2^k), 1], afterwards the coordinates sum to one
                // so the sum of squares is confined to [1/d, 1].
                double lo;
                double hi;
                round_range(round, lo, hi);

                // What the fit hands back carries the factor the NEXT round's
                // argument wants, and the factor this round's argument already
                // arrived with is divided out. On the last round there is no
                // next round and the answer must be the SoftMax itself, so the
                // gain is exactly what undoes the fold.
                double gain = 1.0;
                if (fold_affine)
                {
                    const bool last = round + 1 == config.iterations;
                    gain = (last ? 1.0 : std::sqrt(domain[round + 1])) /
                           domain[round];
                }

                Ciphertext<Scheme::CKKS> reciprocal;
                {
                    Range _r("softmax.inverse");
                    if (aux_refresh)
                    {
                        // The paper's auxiliary track, Figure 2's orange
                        // triangle: the denominator is one ciphertext however
                        // wide the axis is, so it alone is refreshed and the
                        // fit is paid above the wide track. A refresh assumes
                        // values in [-1, 1], which is exactly where the fit's
                        // domain map puts them, so completing that map is the
                        // whole preparation -- the multiplier is already
                        // riding on the mask or is applied here on the narrow
                        // side, and the shift is an addition either way.
                        if (!fold_affine)
                        {
                            multiply_constant(
                                total, Llama3Operator::domain_scale(lo, hi));
                        }
                        add_constant(total,
                                     Llama3Operator::domain_shift(lo, hi));
                        total = bootstrap(total, *boot_key, relin_key);

                        // The argument arrives mapped, so the series is
                        // evaluated on [-1, 1] directly, with the
                        // coefficients of the fit on [lo, hi] -- the same
                        // series, met one step later. The gain keeps its job:
                        // it divides out the factor the numerator still
                        // carries and hands the next round the factor its
                        // own fit wants.
                        reciprocal = evaluate_chebyshev(
                            total,
                            chebyshev_coefficients(
                                [gain](double x) { return gain / x; }, lo,
                                hi, config.inverse_degree),
                            -1.0, 1.0, relin_key);
                    }
                    else
                    {
                        reciprocal =
                            inverse(total, lo, hi, config.inverse_degree,
                                    config.inverse_newton, relin_key,
                                    fold_affine, gain);
                    }
                }
                if (debug_trace)
                {
                    std::vector<Ciphertext<Scheme::CKKS>> one{reciprocal};
                    debug_trace("softmax.reciprocal", one);
                }

                for (std::size_t p = 0; p < y.size(); p++)
                {
                    y[p] = multiply_and_rescale(y_squared[p], reciprocal,
                                                relin_key);
                }
            }

            return y;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::rope(Ciphertext<Scheme::CKKS>& ct,
                             Plaintext<Scheme::CKKS>& cos_plain,
                             Plaintext<Scheme::CKKS>& sin_plain, int swap_shift,
                             Galoiskey<Scheme::CKKS>& galois_key)
        {
            // The two products are added, so unequal plaintext scales weight
            // the cosine and sine terms differently and the rotation quietly
            // comes out wrong.
            if (cos_plain.scale() != sin_plain.scale())
            {
                throw std::invalid_argument(
                    "RoPE needs the cosine and sine plaintexts at one scale");
            }

            Range _r_rope("rope");

            // The rotation that brings channel c + head_dim/2 alongside c, so
            // the pair the rotation acts on sits in one ciphertext. A head one
            // block wide pays this; a wider one chooses two blocks instead and
            // never gets here.
            Ciphertext<Scheme::CKKS> swapped(context_);
            {
                Range _r("rope.swap_rotation");
                rotate_rows(ct, swapped, galois_key, swap_shift);
            }

            // x cos(theta) + swap(x) sin(theta), both plaintext products, so
            // the whole rotation costs one level and no key switch beyond the
            // swap above.
            Range _r("rope.plain_multiply");
            Ciphertext<Scheme::CKKS> direct = ct;
            multiply_plaintext(direct, cos_plain);
            rescale_inplace(direct);

            multiply_plaintext(swapped, sin_plain);
            rescale_inplace(swapped);

            add_same_scale(direct, swapped, "rope");
            return direct;
        }

        // -------------------------------------------------------------------
        // PCMM for PC-attention, Section 4.2
        // -------------------------------------------------------------------

        std::vector<double>
        Llama3Operator::pcmm_plaintext_block(
            const std::vector<double>& tau_sigma_a, int d, int i, int j, int b,
            int ell)
        {
            if (static_cast<int>(tau_sigma_a.size()) != d * d)
            {
                throw std::invalid_argument(
                    "The stored plaintext matrix must be d by d");
            }

            const int k = i + j * b;
            std::vector<double> block = rotate_cols_host(tau_sigma_a, d, k);
            return rotate_rows_host(block, d, -(ell * k) - j * b);
        }

        std::vector<int>
        Llama3Operator::pcmm_rotation_indices(const MatrixLayout& layout,
                                              int giant, int baby)
        {
            if (giant < 1 || baby < 1 || giant * baby != layout.d)
            {
                throw std::invalid_argument(
                    "BSGS factors must be positive and multiply to the matrix "
                    "dimension");
            }

            const int row = layout.d * layout.batch;
            std::vector<int> indices;
            for (int i = 1; i < baby; i++)
            {
                indices.push_back(i * row);
            }
            for (int j = 1; j < giant; j++)
            {
                indices.push_back(j * baby * row);
            }
            return indices;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::pcmm(Ciphertext<Scheme::CKKS>& ct,
                             const std::vector<double>& a_stored,
                             const MatrixLayout& layout, int giant, int baby,
                             int ell, Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }
            if (giant < 1 || baby < 1 || giant * baby != layout.d)
            {
                throw std::invalid_argument(
                    "BSGS factors must be positive and multiply to the matrix "
                    "dimension");
            }
            if (ell < 0)
            {
                throw std::invalid_argument(
                    "The tau exponent of Equation (5) must not be negative");
            }

            const int d = layout.d;
            const int batch = layout.batch;
            const std::size_t entries = static_cast<std::size_t>(d) * d;

            const bool shared = a_stored.size() == entries;
            if (!shared &&
                a_stored.size() != entries * static_cast<std::size_t>(batch))
            {
                throw std::invalid_argument(
                    "The stored matrix must hold one d by d block, shared by "
                    "the batch, or one per batch entry");
            }

            Range _r_pcmm("pcmm");

            // Baby step: the row rotations of the ciphertext are shared by
            // every giant step, so they are taken once.
            const int row = d * batch;
            std::vector<Ciphertext<Scheme::CKKS>> rotated;
            rotated.reserve(baby);
            {
                Range _r("pcmm.baby_rotations");
                for (int i = 0; i < baby; i++)
                {
                    if (i == 0)
                    {
                        rotated.push_back(ct);
                    }
                    else
                    {
                        Ciphertext<Scheme::CKKS> shifted(context_);
                        rotate_rows(ct, shifted, galois_key, i * row);
                        rotated.push_back(shifted);
                    }
                }
            }

            const double plain_scale = rescale_prime(ct);
            Ciphertext<Scheme::CKKS> result(context_);
            bool started = false;

            for (int j = 0; j < giant; j++)
            {
                Ciphertext<Scheme::CKKS> inner(context_);
                bool inner_started = false;

                for (int i = 0; i < baby; i++)
                {
                    // The rearranged plaintext is derived here rather than
                    // stored, which is what holds the weight footprint at a
                    // single copy of the matrix.
                    Plaintext<Scheme::CKKS> plain(context_);
                    {
                        // Deriving and encoding the weight is host work in the
                        // middle of a device pipeline, and every call repeats
                        // it, so it gets its own range rather than being folded
                        // into the multiply it feeds.
                        Range _r("pcmm.weight_encode");
                        std::vector<double> slots(slot_count_, 0.0);
                        for (int m = 0; m < batch; m++)
                        {
                            const double* source =
                                a_stored.data() +
                                (shared ? 0
                                        : static_cast<std::size_t>(m) * entries);
                            std::vector<double> matrix(source,
                                                       source + entries);
                            const std::vector<double> block =
                                pcmm_plaintext_block(matrix, d, i, j, baby, ell);

                            for (std::size_t e = 0; e < entries; e++)
                            {
                                slots[e * batch + m] = block[e];
                            }
                        }
                        plain = encode(slots, plain_scale, ct.depth());
                    }

                    Range _r("pcmm.multiply_accumulate");
                    Ciphertext<Scheme::CKKS> term(context_);
                    multiply_plain(rotated[i], plain, term);

                    if (!inner_started)
                    {
                        inner = term;
                        inner_started = true;
                    }
                    else
                    {
                        add_same_scale(inner, term, "pcmm baby step");
                    }
                }

                // Rescaling before the giant rotation is not a concession: the
                // rotation refuses a ciphertext with a pending rescale, and
                // this is the single level the whole operation spends.
                rescale_inplace(inner);

                if (j > 0)
                {
                    Range _r("pcmm.giant_rotation");
                    Ciphertext<Scheme::CKKS> shifted(context_);
                    rotate_rows(inner, shifted, galois_key, j * baby * row);
                    inner = shifted;
                }

                if (!started)
                {
                    result = inner;
                    started = true;
                }
                else
                {
                    add_same_scale(result, inner, "pcmm giant step");
                }
            }

            return result;
        }

        // -------------------------------------------------------------------
        // Linear maps on a packed matrix, and the encrypted product
        // -------------------------------------------------------------------

        std::vector<int>
        Llama3Operator::tau_rotation_indices(const MatrixLayout& layout)
        {
            require_layout(layout);

            const int row = layout.d * layout.batch;
            std::vector<int> indices;
            for (int u = 1; u < layout.d; u++)
            {
                indices.push_back(u * row);
            }
            return indices;
        }

        std::vector<int>
        Llama3Operator::transpose_rotation_indices(const MatrixLayout& layout)
        {
            require_layout(layout);

            // Entry (i, j) sits (j - i)(d - 1) positions from entry (j, i).
            const int step = (layout.d - 1) * layout.batch;
            std::vector<int> indices;
            for (int s = 1; s < layout.d; s++)
            {
                indices.push_back(s * step);
                indices.push_back(-s * step);
            }
            return indices;
        }

        std::vector<int>
        Llama3Operator::ccmm_rotation_indices(const MatrixLayout& layout)
        {
            require_layout(layout);

            const int row = layout.d * layout.batch;
            std::vector<int> indices;
            // The diagonals of the left operand, shared by every k.
            for (int s = 1; s < layout.d; s++)
            {
                indices.push_back(s * layout.batch);
                indices.push_back(-s * layout.batch);
            }
            // tau of the right operand and the row rotations that follow it
            // ask for the same shifts.
            for (int u = 1; u < layout.d; u++)
            {
                indices.push_back(u * row);
            }
            return indices;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::tau(Ciphertext<Scheme::CKKS>& ct,
                            const MatrixLayout& layout,
                            Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }

            const int d = layout.d;
            const int batch = layout.batch;
            const int row = d * batch;
            const double plain_scale = rescale_prime(ct);

            Range _r_tau("tau");

            Ciphertext<Scheme::CKKS> acc(context_);
            bool started = false;
            for (int u = 0; u < d; u++)
            {
                // tau(X)[i][j] = X[(i + j) mod d][j], so column j takes its
                // entries from the row rotation by j and from no other.
                std::vector<double> mask(slot_count_, 0.0);
                for (int r = 0; r < d; r++)
                {
                    for (int m = 0; m < batch; m++)
                    {
                        mask[(r * d + u) * batch + m] = 1.0;
                    }
                }

                Ciphertext<Scheme::CKKS> source(context_);
                if (u == 0)
                {
                    source = ct;
                }
                else
                {
                    // The automorphism, and the only key switch tau performs:
                    // one per row rotation, d - 1 of them per call.
                    Range _r("tau.automorphism");
                    rotate_rows(ct, source, galois_key, u * row);
                }

                accumulate_masked(acc, started, source, mask, plain_scale,
                                  "tau");
            }

            rescale_inplace(acc);
            return acc;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::transpose(Ciphertext<Scheme::CKKS>& ct,
                                  const MatrixLayout& layout,
                                  Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }

            const int d = layout.d;
            const int batch = layout.batch;
            const int step = (d - 1) * batch;
            const double plain_scale = rescale_prime(ct);

            Range _r_transpose("transpose");

            Ciphertext<Scheme::CKKS> acc(context_);
            bool started = false;
            for (int s = -(d - 1); s <= d - 1; s++)
            {
                // The diagonal j - i = s, which the rotation by s(d - 1)
                // brings into place and the mask is what keeps.
                std::vector<double> mask;
                for (int i = 0; i < d; i++)
                {
                    const int j = i + s;
                    if (j < 0 || j >= d)
                    {
                        continue;
                    }
                    if (mask.empty())
                    {
                        mask.assign(slot_count_, 0.0);
                    }
                    for (int m = 0; m < batch; m++)
                    {
                        mask[(i * d + j) * batch + m] = 1.0;
                    }
                }
                if (mask.empty())
                {
                    continue;
                }

                Ciphertext<Scheme::CKKS> source(context_);
                if (s == 0)
                {
                    source = ct;
                }
                else
                {
                    // One key switch per diagonal, 2(d - 1) per call, which is
                    // why a transpose costs about twice a tau.
                    Range _r("transpose.automorphism");
                    rotate_rows(ct, source, galois_key, s * step);
                }

                accumulate_masked(acc, started, source, mask, plain_scale,
                                  "transpose");
            }

            rescale_inplace(acc);
            return acc;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::ccmm(Ciphertext<Scheme::CKKS>& a,
                             Ciphertext<Scheme::CKKS>& b,
                             const MatrixLayout& layout, double scale,
                             Galoiskey<Scheme::CKKS>& galois_key,
                             Relinkey<Scheme::CKKS>& relin_key)
        {
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }

            const int d = layout.d;
            const int batch = layout.batch;
            const int row = d * batch;

            Range _r_ccmm("ccmm");

            Ciphertext<Scheme::CKKS> lhs = a;
            Ciphertext<Scheme::CKKS> rhs = b;
            align_levels(lhs, rhs);

            // Right operand: tau costs one level, and the d - 1 row rotations
            // that follow it cost none.
            std::vector<Ciphertext<Scheme::CKKS>> right;
            right.reserve(d);
            {
                Range _r("ccmm.right_operand");
                Ciphertext<Scheme::CKKS> tb = tau(rhs, layout, galois_key);
                for (int k = 0; k < d; k++)
                {
                    if (k == 0)
                    {
                        right.push_back(tb);
                    }
                    else
                    {
                        Ciphertext<Scheme::CKKS> shifted(context_);
                        rotate_rows(tb, shifted, galois_key, k * row);
                        right.push_back(shifted);
                    }
                }
            }

            // Left operand: rot_C^k(sigma(A))[i][j] is A[i][(i + j + k) mod d],
            // which is still a single diagonal of A per shift, so all d of
            // them are built from one set of 2d - 2 rotations and differ only
            // in their masks. That is what keeps the whole product at depth
            // two instead of the three sigma and the column rotation would
            // cost separately.
            std::vector<Ciphertext<Scheme::CKKS>> diagonal;
            diagonal.reserve(2 * d - 1);
            {
                Range _r("ccmm.left_diagonals");
                for (int s = -(d - 1); s <= d - 1; s++)
                {
                    if (s == 0)
                    {
                        diagonal.push_back(lhs);
                    }
                    else
                    {
                        Ciphertext<Scheme::CKKS> shifted(context_);
                        rotate_rows(lhs, shifted, galois_key, s * batch);
                        diagonal.push_back(shifted);
                    }
                }
            }

            const double plain_scale = rescale_prime(lhs);
            Ciphertext<Scheme::CKKS> product(context_);
            bool product_started = false;

            for (int k = 0; k < d; k++)
            {
                std::vector<std::vector<double>> masks(2 * d - 1);
                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        const int s = wrap(i + j + k, d) - j;
                        std::vector<double>& mask = masks[s + d - 1];
                        if (mask.empty())
                        {
                            mask.assign(slot_count_, 0.0);
                        }
                        // The requested scaling rides on masks that have to be
                        // encoded anyway, so it is free.
                        for (int m = 0; m < batch; m++)
                        {
                            mask[(i * d + j) * batch + m] = scale;
                        }
                    }
                }

                Ciphertext<Scheme::CKKS> left(context_);
                {
                    Range _r("ccmm.left_masking");
                    bool left_started = false;
                    for (int t = 0; t < 2 * d - 1; t++)
                    {
                        if (masks[t].empty())
                        {
                            continue;
                        }
                        accumulate_masked(left, left_started, diagonal[t],
                                          masks[t], plain_scale,
                                          "ccmm.left");
                    }
                    rescale_inplace(left);
                }

                Range _r("ccmm.multiply");
                Ciphertext<Scheme::CKKS> term(context_);
                multiply(left, right[k], term);

                // The d terms are summed before relinearising, so the whole
                // product pays for one key switch rather than d.
                if (!product_started)
                {
                    product = term;
                    product_started = true;
                }
                else
                {
                    add_same_scale(product, term, "ccmm product");
                }
            }

            Range _r("ccmm.relinearize");
            relinearize_inplace(product, relin_key);
            rescale_inplace(product);
            return product;
        }

        // -------------------------------------------------------------------
        // Sublayers
        // -------------------------------------------------------------------

        void Llama3Operator::bsgs_split(int d, int& giant, int& baby)
        {
            if (giant > 0 && baby > 0)
            {
                if (giant * baby != d)
                {
                    throw std::invalid_argument(
                        "The BSGS factors given must multiply to the matrix "
                        "dimension");
                }
                return;
            }

            int b = 1;
            while (b * b < d)
            {
                b <<= 1;
            }
            if (b * b > d)
            {
                b >>= 1;
            }
            baby = b;
            giant = d / b;
        }

        std::vector<double>
        Llama3Operator::sigma_blocks(const std::vector<double>& w,
                                     const MatrixLayout& layout,
                                     const char* name)
        {
            const std::size_t entries =
                static_cast<std::size_t>(layout.d) * layout.d;

            if (w.size() == entries)
            {
                return permute_sigma(w, layout.d);
            }

            if (w.size() == entries * static_cast<std::size_t>(layout.batch))
            {
                std::vector<double> out;
                out.reserve(w.size());
                for (int m = 0; m < layout.batch; m++)
                {
                    const std::ptrdiff_t base =
                        static_cast<std::ptrdiff_t>(m) *
                        static_cast<std::ptrdiff_t>(entries);
                    const std::vector<double> block(
                        w.begin() + base,
                        w.begin() + base +
                            static_cast<std::ptrdiff_t>(entries));
                    const std::vector<double> permuted =
                        permute_sigma(block, layout.d);
                    out.insert(out.end(), permuted.begin(), permuted.end());
                }
                return out;
            }

            throw std::invalid_argument(
                std::string("The ") + name +
                " weight must hold one d by d block, shared by the batch, or "
                "one per batch entry");
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::project(Ciphertext<Scheme::CKKS>& tau_x,
                                const std::vector<double>& weight,
                                const MatrixLayout& layout, int giant, int baby,
                                const char* name,
                                Galoiskey<Scheme::CKKS>& galois_key)
        {
            // Equation (5) at l = 0 turns tau of the operand into the plain
            // product, so the stored weight is just sigma of it.
            const std::vector<double> stored =
                sigma_blocks(weight, layout, name);
            return pcmm(tau_x, stored, layout, giant, baby, 0, galois_key);
        }

        void Llama3Operator::require_uniform(
            const std::vector<Ciphertext<Scheme::CKKS>>& in,
            const char* name) const
        {
            if (in.empty())
            {
                throw std::invalid_argument(std::string("The ") + name +
                                            " needs at least one block");
            }

            const int depth = in.front().depth();
            const double scale = in.front().scale();
            for (std::size_t j = 1; j < in.size(); j++)
            {
                if (in[j].depth() != depth)
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " blocks sit at different levels, so the sums below "
                        "them cannot be taken");
                }
                // The same relative test add_same_scale uses: rescaling drifts
                // scales, and what must not happen is two blocks drifting
                // apart before they are added together.
                if (std::abs(in[j].scale() - scale) >
                    1e-9 * std::max(std::abs(scale), std::abs(in[j].scale())))
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " blocks sit at different scales, so the sums below "
                        "them would be silently wrong");
                }
            }
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Operator::tau_blocks(std::vector<Ciphertext<Scheme::CKKS>>& in,
                                   const MatrixLayout& layout,
                                   Galoiskey<Scheme::CKKS>& galois_key)
        {
            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.size());
            for (auto& block : in)
            {
                out.push_back(tau(block, layout, galois_key));
            }
            return out;
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::project_blocks(
            std::vector<Ciphertext<Scheme::CKKS>>& tau_x,
            const BlockMatrix& weight, const MatrixLayout& layout, int giant,
            int baby, const char* name, Galoiskey<Scheme::CKKS>& galois_key,
            int token_blocks)
        {
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }
            require_uniform(tau_x, name);
            bsgs_split(layout.d, giant, baby);

            if (weight.empty())
            {
                throw std::invalid_argument(std::string("The ") + name +
                                            " weight has no blocks");
            }
            if (token_blocks < 1)
            {
                throw std::invalid_argument(
                    "A sequence is at least one token block long");
            }
            if (weight.in_blocks * token_blocks !=
                static_cast<int>(tau_x.size()))
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " weight expects a different number of input channel "
                    "blocks than it was given");
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(static_cast<std::size_t>(weight.out_blocks) *
                        token_blocks);

            // Named after the weight, because what a capture wants to know
            // about a block product is which of them it was. The prefix keeps
            // it distinct from the sublayer range of the same name: "head" is
            // the norm and the projection together, "project.head" is the
            // projection alone.
            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "project.%s", name);
            Range _r_project(range_name);

            for (int i = 0; i < weight.out_blocks; i++)
            {
                // A weight reads channels, so the same plaintexts serve every
                // token block and the two blockings simply multiply.
                for (int s = 0; s < token_blocks; s++)
                {
                    Ciphertext<Scheme::CKKS> row(context_);
                    bool started = false;

                    for (int j = 0; j < weight.in_blocks; j++)
                    {
                        const std::vector<double>& block = weight.at(i, j);
                        if (block.empty())
                        {
                            continue; // A zero block is skipped, not encoded.
                        }

                        Ciphertext<Scheme::CKKS> term =
                            project(tau_x[j * token_blocks + s], block, layout,
                                    giant, baby, name, galois_key);

                        if (!started)
                        {
                            row = term;
                            started = true;
                        }
                        else
                        {
                            // Every term left Equation (5) one level below a
                            // common input and at the input's own scale, so the
                            // sum over the input blocks is exact and free.
                            add_same_scale(row, term, name);
                        }
                    }

                    if (!started)
                    {
                        throw std::invalid_argument(
                            std::string("Row ") + std::to_string(i) +
                            " of the " + name +
                            " weight is entirely empty, so that output block "
                            "would be a zero this cannot produce");
                    }
                    out.push_back(row);
                }
            }

            return out;
        }

        void Llama3Operator::rope_blocks(
            std::vector<Ciphertext<Scheme::CKKS>>& in, int per_head,
            int token_blocks, std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
            const MatrixLayout& layout, Galoiskey<Scheme::CKKS>& galois_key)
        {
            // The angle depends on the absolute position of the token, so a
            // cut sequence needs its own pair per token block; the channel
            // part of the angle is shared, which is why one head's plaintexts
            // serve them all.
            if (static_cast<int>(rope_plain.size()) !=
                2 * per_head * token_blocks)
            {
                throw std::invalid_argument(
                    "RoPE needs a cosine and a sine plaintext for each channel "
                    "block of a head, in each token block");
            }

            const int heads =
                static_cast<int>(in.size()) / (per_head * token_blocks);

            if (per_head == 1)
            {
                // The head is one block wide, so pairing channel c with
                // c + d / 2 stays inside it and is the row rotation rope()
                // takes.
                const int swap = rope_swap_shift(layout);
                for (int h = 0; h < heads; h++)
                {
                    for (int s = 0; s < token_blocks; s++)
                    {
                        Ciphertext<Scheme::CKKS>& block =
                            in[h * token_blocks + s];
                        block = rope(block, rope_plain[2 * s],
                                     rope_plain[2 * s + 1], swap, galois_key);
                    }
                }
                return;
            }

            if (per_head % 2 != 0)
            {
                throw std::invalid_argument(
                    "RoPE splits a head in half, so a head spanning several "
                    "channel blocks must span an even number of them");
            }

            // Wider than one block, the partner of channel c lands in another
            // block entirely, so the rotation disappears: the pairing is
            // already there in the choice of which two blocks to combine. The
            // partner is in the same token block, since RoPE mixes channels
            // and never tokens.
            const int half = per_head / 2;
            std::vector<Ciphertext<Scheme::CKKS>> out(in.size(),
                                                      Ciphertext<Scheme::CKKS>(
                                                          context_));

            for (int h = 0; h < heads; h++)
            {
                for (int t = 0; t < per_head; t++)
                {
                    const int partner_t = (t < half) ? t + half : t - half;

                    for (int s = 0; s < token_blocks; s++)
                    {
                        const int index =
                            (h * per_head + t) * token_blocks + s;
                        const int partner =
                            (h * per_head + partner_t) * token_blocks + s;
                        const int cos_at = 2 * (s * per_head + t);

                        if (rope_plain[cos_at].scale() !=
                            rope_plain[cos_at + 1].scale())
                        {
                            throw std::invalid_argument(
                                "RoPE needs the cosine and sine plaintexts at "
                                "one scale");
                        }

                        Ciphertext<Scheme::CKKS> direct = in[index];
                        multiply_plaintext(direct, rope_plain[cos_at]);
                        rescale_inplace(direct);

                        Ciphertext<Scheme::CKKS> crossed = in[partner];
                        multiply_plaintext(crossed, rope_plain[cos_at + 1]);
                        rescale_inplace(crossed);

                        add_same_scale(direct, crossed, "rope across blocks");
                        out[index] = direct;
                    }
                }
            }

            in = out;
        }

        int Llama3Operator::rope_swap_shift(const MatrixLayout& layout)
        {
            require_layout(layout);
            if (layout.d < 2)
            {
                throw std::invalid_argument(
                    "RoPE needs at least two channels to pair up");
            }
            // Channels are the slow axis, so pairing channel r with
            // r + d / 2 is one row rotation by half the matrix.
            return (layout.d / 2) * layout.d * layout.batch;
        }

        std::vector<double>
        Llama3Operator::causal_mask(const MatrixLayout& layout)
        {
            return causal_block_mask(layout, 0, 0, 1);
        }

        std::vector<double>
        Llama3Operator::causal_block_mask(const MatrixLayout& layout,
                                          int query_block, int key_block,
                                          int token_blocks)
        {
            require_layout(layout);
            if (token_blocks < 1)
            {
                throw std::invalid_argument(
                    "A sequence is at least one token block long");
            }
            if (query_block < 0 || query_block >= token_blocks ||
                key_block < 0 || key_block >= token_blocks)
            {
                throw std::invalid_argument(
                    "The score block asked for is outside the sequence");
            }

            const int d = layout.d;
            const int batch = layout.batch;

            if (key_block > query_block)
            {
                // Every key in the block is ahead of every query, so nothing
                // survives. An empty mask says so, and the caller skips the
                // block rather than forming it and multiplying it away.
                return std::vector<double>();
            }

            std::vector<double> mask(static_cast<std::size_t>(layout.slots),
                                     0.0);
            // The key blocks a causal query block is handed at all: the ones
            // ahead of it are not formed, so they are not what the SoftMax is
            // normalising against either.
            const double visible = static_cast<double>(d) * (query_block + 1);

            for (int query = 0; query < d; query++)
            {
                // A query attends to itself and everything before it, so the
                // number of surviving keys grows down the sequence. Weighting
                // by sqrt(visible / kept) leaves the sum of squares in the
                // range a full set of blocks would give, which is the range
                // the first round's reciprocal is fitted over; the rounds
                // normalise, so a weight constant along the key axis cancels
                // and the result is unchanged. The kept count is global rather
                // than per block, because the key axis runs across the blocks
                // and a weight varying along it would not cancel.
                const int kept = query_block * d + query + 1;
                const double weight =
                    std::sqrt(visible / static_cast<double>(kept));

                // On the diagonal the block is the triangle; behind it every
                // key is already in the past and the block is kept whole.
                const int last =
                    (key_block < query_block) ? d - 1 : query;
                for (int key = 0; key <= last; key++)
                {
                    for (int m = 0; m < batch; m++)
                    {
                        mask[(key * d + query) * batch + m] = weight;
                    }
                }
            }

            return mask;
        }

        std::vector<int> Llama3Operator::attention_rotation_indices(
            const AttentionConfig& config)
        {
            // Neither blocking adds a shift. Blocks are combined by adding and
            // by multiplying, never by moving a slot from one to another, so
            // the list depends on the layout and not on how many blocks a
            // model or a prompt is cut into.
            const MatrixLayout& layout = config.layout;
            require_layout(layout);

            int giant = config.giant;
            int baby = config.baby;
            bsgs_split(layout.d, giant, baby);

            std::set<int> indices;
            for (int r : tau_rotation_indices(layout))
            {
                indices.insert(r);
            }
            for (int r : pcmm_rotation_indices(layout, giant, baby))
            {
                indices.insert(r);
            }
            for (int r : transpose_rotation_indices(layout))
            {
                indices.insert(r);
            }
            for (int r : ccmm_rotation_indices(layout))
            {
                indices.insert(r);
            }
            for (int r : strided_rotation_indices(layout.d * layout.batch,
                                                  layout.d))
            {
                indices.insert(r);
            }
            if (config.rope)
            {
                indices.insert(rope_swap_shift(layout));
            }

            return std::vector<int>(indices.begin(), indices.end());
        }

        std::vector<int> Llama3Operator::feed_forward_rotation_indices(
            const FeedForwardConfig& config)
        {
            const MatrixLayout& layout = config.layout;
            require_layout(layout);

            int giant = config.giant;
            int baby = config.baby;
            bsgs_split(layout.d, giant, baby);

            std::set<int> indices;
            for (int r : tau_rotation_indices(layout))
            {
                indices.insert(r);
            }
            for (int r : pcmm_rotation_indices(layout, giant, baby))
            {
                indices.insert(r);
            }
            return std::vector<int>(indices.begin(), indices.end());
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::attention(
            Ciphertext<Scheme::CKKS>& x, const AttentionWeights& weights,
            std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
            const AttentionConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (config.heads > 1)
            {
                throw std::invalid_argument(
                    "One channel block holds one head; a multi-head layer has "
                    "to be handed its blocks as a vector");
            }
            if (config.token_blocks > 1)
            {
                throw std::invalid_argument(
                    "One channel block holds d tokens; a longer sequence has "
                    "to be handed its blocks as a vector");
            }

            BlockAttentionWeights blocked;
            blocked.query = BlockMatrix(weights.query);
            blocked.key = BlockMatrix(weights.key);
            blocked.value = BlockMatrix(weights.value);
            blocked.output = BlockMatrix(weights.output);

            AttentionConfig one = config;
            one.heads = 1;
            one.kv_heads = 1;
            one.token_blocks = 1;

            std::vector<Ciphertext<Scheme::CKKS>> in{x};
            std::vector<Ciphertext<Scheme::CKKS>> out =
                attention(in, blocked, rope_plain, one, galois_key, relin_key);
            return out.front();
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::attention(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            const BlockAttentionWeights& weights,
            std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
            const AttentionConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            const MatrixLayout& layout = config.layout;
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }
            require_uniform(x, "attention input");

            const int d = layout.d;
            int giant = config.giant;
            int baby = config.baby;
            bsgs_split(d, giant, baby);

            const int token_blocks =
                config.token_blocks > 0 ? config.token_blocks : 1;
            if (static_cast<int>(x.size()) % token_blocks != 0)
            {
                throw std::invalid_argument(
                    "The attention input must hold the same channel blocks in "
                    "every token block");
            }

            const bool any_projection = !weights.query.empty() ||
                                        !weights.key.empty() ||
                                        !weights.value.empty();
            const bool all_projections = !weights.query.empty() &&
                                         !weights.key.empty() &&
                                         !weights.value.empty();
            if (any_projection && !all_projections)
            {
                throw std::invalid_argument(
                    "Attention needs all three of the query, key and value "
                    "weights, or none of them");
            }

            std::vector<Ciphertext<Scheme::CKKS>> query;
            std::vector<Ciphertext<Scheme::CKKS>> key;
            std::vector<Ciphertext<Scheme::CKKS>> value;

            Range _r_attention("attention");

            if (all_projections)
            {
                if (weights.key.out_blocks != weights.value.out_blocks)
                {
                    throw std::invalid_argument(
                        "The key and value weights must produce the same "
                        "number of channel blocks, since one head reads a run "
                        "of both");
                }

                Range _r("attention.qkv_projection");
                // One tau of each input block serves all three projections,
                // which is the only reason an isolated pcmm's tau is
                // affordable here.
                std::vector<Ciphertext<Scheme::CKKS>> tau_x;
                {
                    Range _rt("attention.qkv_projection.tau");
                    tau_x = tau_blocks(x, layout, galois_key);
                }
                query = project_blocks(tau_x, weights.query, layout, giant,
                                       baby, "query", galois_key,
                                       token_blocks);
                key = project_blocks(tau_x, weights.key, layout, giant, baby,
                                     "key", galois_key, token_blocks);
                value = project_blocks(tau_x, weights.value, layout, giant,
                                       baby, "value", galois_key,
                                       token_blocks);
            }
            else
            {
                query = x;
                key = x;
                value = x;
            }

            const int q_blocks =
                static_cast<int>(query.size()) / token_blocks;
            const int kv_blocks = static_cast<int>(key.size()) / token_blocks;
            const int heads = config.heads > 0 ? config.heads : 1;
            const int kv_heads = config.kv_heads > 0 ? config.kv_heads : heads;

            if (heads % kv_heads != 0)
            {
                throw std::invalid_argument(
                    "Grouped-query attention shares one key head between a "
                    "whole number of query heads, so kv_heads must divide "
                    "heads");
            }
            if (q_blocks % heads != 0 || kv_blocks % kv_heads != 0)
            {
                throw std::invalid_argument(
                    "The heads must share the query, key and value channel "
                    "blocks out evenly between them");
            }
            const int per_head = q_blocks / heads;
            if (kv_blocks / kv_heads != per_head)
            {
                throw std::invalid_argument(
                    "A key head must be exactly as wide as a query head, "
                    "since the scores are their product; with kv_heads left "
                    "at the default that means the two weights agree on "
                    "out_blocks");
            }
            // Query heads per key head: consecutive heads share a run.
            const int group = heads / kv_heads;

            if (config.rope)
            {
                Range _r("attention.rope");
                rope_blocks(query, per_head, token_blocks, rope_plain, layout,
                            galois_key);
                rope_blocks(key, per_head, token_blocks, rope_plain, layout,
                            galois_key);
            }

            // The scores are formed as K^T Q rather than Q^T K. That puts the
            // key position on the slow axis, so the SoftMax denominator is the
            // exact strided reduction and costs no level and no mask, and what
            // comes out is already P^T, which is the operand the value product
            // wants.
            //
            // Every key block is read by every query block it precedes, so the
            // transposes are taken once here rather than inside the grid.
            std::vector<Ciphertext<Scheme::CKKS>> key_t;
            key_t.reserve(key.size());
            {
                Range _r("attention.key_transpose");
                for (auto& block : key)
                {
                    key_t.push_back(transpose(block, layout, galois_key));
                }
            }

            SoftmaxConfig softmax_config = config.softmax;
            softmax_config.strided = true;
            softmax_config.stride = d * layout.batch;
            softmax_config.count = d;

            // head_dim is the channel blocks a head owns, times d.
            const double head =
                config.head_scale > 0.0
                    ? config.head_scale
                    : 1.0 / std::sqrt(static_cast<double>(per_head) * d);

            std::vector<Ciphertext<Scheme::CKKS>> out(
                static_cast<std::size_t>(q_blocks) * token_blocks,
                Ciphertext<Scheme::CKKS>(context_));
            std::vector<bool> started(out.size(), false);

            for (int h = 0; h < heads; h++)
            {
                // Under grouped-query attention the key and value run is
                // shared; the query is not, so the scores and the SoftMax are
                // still this head's own.
                const int kv_first = (h / group) * per_head;
                const int q_first = h * per_head;

                for (int s = 0; s < token_blocks; s++)
                {
                    // A causal query block sees itself and what came before,
                    // so the blocks ahead of it are never formed: masking them
                    // afterwards would pay for the products and then throw
                    // them away.
                    const int last_key =
                        config.causal ? s : token_blocks - 1;

                    std::vector<Ciphertext<Scheme::CKKS>> parts;
                    std::vector<std::vector<double>> masks;
                    parts.reserve(last_key + 1);

                    for (int u = 0; u <= last_key; u++)
                    {
                        // A head wider than one block sums its blocks' scores.
                        // Each term is a ccmm at the same level and scale, so
                        // the sum over the head is free.
                        Range _rs("attention.scores");
                        Ciphertext<Scheme::CKKS> scores = head_scores(
                            key_t, query, kv_first, q_first, per_head,
                            token_blocks, u, s, layout, head, galois_key,
                            relin_key);

                        if (config.score_shift != 0.0)
                        {
                            // Translating the scores into [-bound, 0] is
                            // calibration in the paper, not a homomorphic
                            // maximum, so it is a constant.
                            add_constant(scores, -config.score_shift);
                        }

                        parts.push_back(scores);
                        if (config.causal)
                        {
                            masks.push_back(causal_block_mask(
                                layout, s, u, token_blocks));
                        }
                    }

                    // One SoftMax however many key blocks the query block
                    // sees: the denominator is summed across them before the
                    // single reduction.
                    std::vector<Ciphertext<Scheme::CKKS>> probabilities;
                    {
                        Range _r("attention.softmax");
                        probabilities = softmax(parts, softmax_config, masks,
                                                galois_key, relin_key);
                    }

                    Range _rv("attention.value_product");
                    for (int t = 0; t < per_head; t++)
                    {
                        const std::size_t at =
                            static_cast<std::size_t>(q_first + t) *
                                token_blocks + s;

                        for (int u = 0; u <= last_key; u++)
                        {
                            const int from =
                                (kv_first + t) * token_blocks + u;
                            Ciphertext<Scheme::CKKS> term = ccmm(
                                value[from], probabilities[u], layout, 1.0,
                                galois_key, relin_key);

                            if (!started[at])
                            {
                                out[at] = term;
                                started[at] = true;
                            }
                            else
                            {
                                // The probability blocks left one SoftMax
                                // together and the value blocks share a level,
                                // so summing over the key blocks is free.
                                add_same_scale(out[at], term,
                                               "attention value product");
                            }
                        }
                    }
                }
            }

            if (!weights.output.empty())
            {
                Range _r("attention.output_projection");
                std::vector<Ciphertext<Scheme::CKKS>> tau_out;
                {
                    Range _rt("attention.output_projection.tau");
                    tau_out = tau_blocks(out, layout, galois_key);
                }
                out = project_blocks(tau_out, weights.output, layout, giant,
                                     baby, "output", galois_key, token_blocks);
            }

            return out;
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::head_scores(
            std::vector<Ciphertext<Scheme::CKKS>>& key_t,
            std::vector<Ciphertext<Scheme::CKKS>>& query, int first_key,
            int first_query, int per_head, int token_blocks, int key_block,
            int query_block, const MatrixLayout& layout, double scale,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            Ciphertext<Scheme::CKKS> scores(context_);
            bool started = false;

            for (int t = 0; t < per_head; t++)
            {
                const int k = (first_key + t) * token_blocks + key_block;
                const int q = (first_query + t) * token_blocks + query_block;

                Ciphertext<Scheme::CKKS> term = ccmm(
                    key_t[k], query[q], layout, scale, galois_key, relin_key);

                if (!started)
                {
                    scores = term;
                    started = true;
                }
                else
                {
                    add_same_scale(scores, term, "attention scores");
                }
            }

            return scores;
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::feed_forward(
            Ciphertext<Scheme::CKKS>& x, const FeedForwardWeights& weights,
            const FeedForwardConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (config.token_blocks > 1)
            {
                throw std::invalid_argument(
                    "One channel block holds d tokens; a longer sequence has "
                    "to be handed its blocks as a vector");
            }

            BlockFeedForwardWeights blocked;
            blocked.gate = BlockMatrix(weights.gate);
            blocked.up = BlockMatrix(weights.up);
            blocked.down = BlockMatrix(weights.down);

            std::vector<Ciphertext<Scheme::CKKS>> in{x};
            std::vector<Ciphertext<Scheme::CKKS>> out =
                feed_forward(in, blocked, config, galois_key, relin_key);
            return out.front();
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::feed_forward(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            const BlockFeedForwardWeights& weights,
            const FeedForwardConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            const MatrixLayout& layout = config.layout;
            require_layout(layout);
            if (layout.slots != slot_count_)
            {
                throw std::invalid_argument(
                    "The matrix layout must fill the slot vector exactly");
            }
            if (weights.gate.empty() || weights.up.empty() ||
                weights.down.empty())
            {
                throw std::invalid_argument(
                    "SwiGLU needs the gate, up and down weights");
            }
            if (weights.gate.out_blocks != weights.up.out_blocks)
            {
                throw std::invalid_argument(
                    "The gate and the up projection must agree on the hidden "
                    "width, since the two are multiplied together");
            }
            require_uniform(x, "feed-forward input");

            int giant = config.giant;
            int baby = config.baby;
            bsgs_split(layout.d, giant, baby);

            const int token_blocks =
                config.token_blocks > 0 ? config.token_blocks : 1;

            // The gate and the up projection read the same input, so the tau
            // Equation (5) consumes is taken once for both.
            Range _r_ffn("feed_forward");

            std::vector<Ciphertext<Scheme::CKKS>> gate;
            std::vector<Ciphertext<Scheme::CKKS>> up;
            {
                Range _r("feed_forward.gate_up_projection");
                std::vector<Ciphertext<Scheme::CKKS>> tau_x;
                {
                    Range _rt("feed_forward.gate_up_projection.tau");
                    tau_x = tau_blocks(x, layout, galois_key);
                }
                gate = project_blocks(tau_x, weights.gate, layout, giant, baby,
                                      "gate", galois_key, token_blocks);
                up = project_blocks(tau_x, weights.up, layout, giant, baby,
                                    "up", galois_key, token_blocks);
            }

            // SwiGLU is a map on one token's channels, so the token blocks
            // never meet and this loop covers both blockings at once.
            std::vector<Ciphertext<Scheme::CKKS>> hidden;
            hidden.reserve(gate.size());
            for (std::size_t i = 0; i < gate.size(); i++)
            {
                Ciphertext<Scheme::CKKS> activated;
                {
                    Range _r("feed_forward.silu");
                    activated = silu(gate[i], config.silu_bound,
                                     config.silu_degree, relin_key);
                }
                Range _r("feed_forward.gate_times_up");
                hidden.push_back(
                    multiply_and_rescale(activated, up[i], relin_key));
            }

            Range _r("feed_forward.down_projection");
            std::vector<Ciphertext<Scheme::CKKS>> tau_hidden;
            {
                Range _rt("feed_forward.down_projection.tau");
                tau_hidden = tau_blocks(hidden, layout, galois_key);
            }
            return project_blocks(tau_hidden, weights.down, layout, giant, baby,
                                  "down", galois_key, token_blocks);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::bootstrap(Ciphertext<Scheme::CKKS>& x,
                                  Galoiskey<Scheme::CKKS>& boot_key,
                                  Relinkey<Scheme::CKKS>& relin_key)
        {
            Range _r("bootstrap");
            // The procedure raises the modulus of a ciphertext that has one
            // prime left, so anything still unspent has to go first. It is
            // free to drop and it was about to be discarded regardless.
            drop_to_depth(x, context_->get_ciphertext_modulus_count() - 1);
            return regular_bootstrapping(x, boot_key, relin_key);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::bootstrap_to_slots(Ciphertext<Scheme::CKKS>& x,
                                           Galoiskey<Scheme::CKKS>& boot_key,
                                           Relinkey<Scheme::CKKS>& relin_key)
        {
            Range _r("bootstrap_to_slots");
            // Same reason as bootstrap(): ModRaise starts from one prime, and
            // anything unspent was about to be discarded anyway.
            drop_to_depth(x, context_->get_ciphertext_modulus_count() - 1);
            return coeff_to_slot_bootstrapping(x, boot_key, relin_key);
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::slots_to_coeff(Ciphertext<Scheme::CKKS>& x,
                                       Galoiskey<Scheme::CKKS>& boot_key,
                                       int align_drop)
        {
            Range _r("slots_to_coeff");
            if (align_drop < 0)
            {
                throw std::invalid_argument(
                    "A level alignment drops levels or leaves them alone; it "
                    "cannot invent them");
            }

            // No drop_to_depth here, and the asymmetry with bootstrap_to_slots
            // is deliberate: that one has to start from the bottom of the
            // chain because ModRaise does, this one is a linear map and spends
            // levels like any other.
            //
            // The alignment drop is a different matter and is not optional.
            // The encoded StoC diagonals live at ONE level -- see
            // generate_encoding_transform_context, which encodes them at
            // StoC_start_level + 1 -- and multiply_matrix slices that buffer
            // by the ciphertext's own live limb count. Hand it a ciphertext
            // one level too shallow and it reads the diagonals misaligned and
            // returns noise, without complaining: there is no level check on
            // this path. The +1 is there because the PAIR form of the
            // transform spends a level before its matrix multiply, on the i
            // that folds the imaginary half in, and drops the real half to
            // match. The solo form has no imaginary half to fold, so nothing
            // has spent that level for it and it has to be spent here.
            Ciphertext<Scheme::CKKS> aligned = x;
            for (int i = 0; i < align_drop; i++)
            {
                mod_drop_inplace(aligned);
            }
            return solo_slot_to_coeff(aligned, boot_key);
        }

        std::vector<int> Llama3Operator::slot_transform_levels() const
        {
            std::vector<int> out;
            out.reserve(slot_transforms_.size());
            for (const auto& entry : slot_transforms_)
            {
                out.push_back(entry.first);
            }
            return out;
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::slots_to_coeff_at_level(Ciphertext<Scheme::CKKS>& x,
                                                Galoiskey<Scheme::CKKS>& boot_key,
                                                int pieces)
        {
            // No encoding check here: Ciphertext's encoding tag is private to
            // the operator hierarchy and friendship does not inherit. It costs
            // nothing to leave out, because slot_to_coeff below makes exactly
            // that check and throws with exactly that message.
            Range _r("slots_to_coeff_at_level");

            const int level = x.depth();
            const int stages = (pieces > 0) ? pieces : StoC_piece_;

            auto found = slot_transforms_.find(level);
            if (found == slot_transforms_.end())
            {
                SuffixRange _rb("slots_to_coeff_at_level", "build");
                // CtoS_start_level is irrelevant here -- nothing calls the
                // forward half of this context -- but it still has to be a
                // legal level, so it goes at the top of the chain where it
                // always is.
                CKKSEncodingTransformConfig config(stages, stages,
                                                   /*CtoS_start_level=*/0,
                                                   /*StoC_start_level=*/level,
                                                   less_key_mode_);
                // Constructed in place and filled where it lives: the context
                // owns device memory, and there is no reason to ask whether it
                // survives a move.
                found = slot_transforms_
                            .emplace(std::piecewise_construct,
                                     std::forward_as_tuple(level),
                                     std::forward_as_tuple())
                            .first;
                generate_encoding_transform_context(found->second, scale_boot_,
                                                    config);
            }

            // The pair form, with an encryption of zero for the half a rect
            // column does not have. Two things come with it that the solo form
            // does not offer: it CHECKS the level against the one its
            // diagonals were built for, where the solo form silently returns
            // noise; and it performs the alignment drop itself, which is the
            // drop the other overload has to be told about. The zero costs a
            // plaintext product and a rescale on a ciphertext that is
            // identically zero, against a three-stage homomorphic DFT.
            Ciphertext<Scheme::CKKS> zero(context_);
            sub(x, x, zero);
            Ciphertext<Scheme::CKKS> real = x;
            return slot_to_coeff(real, zero, boot_key, found->second);
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3Operator::bootstrap(std::vector<Ciphertext<Scheme::CKKS>>& x,
                                  Galoiskey<Scheme::CKKS>& boot_key,
                                  Relinkey<Scheme::CKKS>& relin_key)
        {
            // Blocks are refreshed one at a time and not together. A bootstrap
            // is slot-wise and every block fills the slots, so there is nothing
            // to share between them.
            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(x.size());
            for (Ciphertext<Scheme::CKKS>& block : x)
            {
                out.push_back(bootstrap(block, boot_key, relin_key));
            }
            return out;
        }

        std::vector<int> Llama3Operator::transformer_block_rotation_indices(
            const TransformerBlockConfig& config)
        {
            std::set<int> indices;
            for (int r : attention_rotation_indices(config.attention))
            {
                indices.insert(r);
            }
            for (int r : feed_forward_rotation_indices(config.feed_forward))
            {
                indices.insert(r);
            }
            // The two norms reduce over the channel axis, which the sublayers
            // need no shift of their own for when a head is one block wide.
            for (const RMSNormConfig* norm :
                 {&config.attention_norm, &config.feed_forward_norm})
            {
                for (int r :
                     strided_rotation_indices(norm->stride, norm->count))
                {
                    indices.insert(r);
                }
            }
            // Bootstrapping's rotations are deliberately absent. They are a
            // different list, generated against a different key, and folding
            // them in here would hide that.
            return std::vector<int>(indices.begin(), indices.end());
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::transformer_block(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            TransformerBlockWeights& weights,
            const TransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (x.empty())
            {
                throw std::invalid_argument("A transformer block needs an "
                                            "input");
            }

            std::vector<Ciphertext<Scheme::CKKS>> normed;
            {
                Range _r("block.attention_norm");
                normed = rms_norm(x, weights.attention_norm,
                                  config.attention_norm, galois_key, relin_key);
            }
            std::vector<Ciphertext<Scheme::CKKS>> sublayer =
                attention(normed, weights.attention, weights.rope,
                          config.attention, galois_key, relin_key);
            std::vector<Ciphertext<Scheme::CKKS>> stream;
            {
                Range _r("block.residual");
                stream = residual_add(x, sublayer);
            }

            if (config.bootstrap)
            {
                Range _r("block.bootstrap_mid");
                stream = bootstrap(stream, boot_key, relin_key);
            }

            {
                Range _r("block.feed_forward_norm");
                normed = rms_norm(stream, weights.feed_forward_norm,
                                  config.feed_forward_norm, galois_key,
                                  relin_key);
            }
            sublayer = feed_forward(normed, weights.feed_forward,
                                    config.feed_forward, galois_key, relin_key);
            Range _r("block.residual");
            return residual_add(stream, sublayer);
        }

        std::vector<int> Llama3Operator::transformer_stack_rotation_indices(
            const TransformerStackConfig& config)
        {
            // In practice this is one block's list, because the blocks of a
            // stack agree on shape and only their fitted intervals differ.
            // Taking the union makes that a property of the configs handed in
            // rather than an assumption about them.
            std::set<int> indices;
            for (const TransformerBlockConfig& block : config.blocks)
            {
                for (int r : transformer_block_rotation_indices(block))
                {
                    indices.insert(r);
                }
            }
            return std::vector<int>(indices.begin(), indices.end());
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::transformer_stack(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            std::vector<TransformerBlockWeights>& weights,
            const TransformerStackConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Galoiskey<Scheme::CKKS>& boot_key, Relinkey<Scheme::CKKS>& relin_key)
        {
            if (weights.empty())
            {
                throw std::invalid_argument("A transformer stack needs at "
                                            "least one block");
            }
            if (weights.size() != config.blocks.size())
            {
                throw std::invalid_argument("A transformer stack needs one "
                                            "config per block");
            }

            std::vector<Ciphertext<Scheme::CKKS>> stream;
            {
                IndexedRange _r("block.%d", 0);
                stream = transformer_block(x, weights[0], config.blocks[0],
                                           galois_key, boot_key, relin_key);
            }

            for (std::size_t block = 1; block < weights.size(); block++)
            {
                IndexedRange _r("block.%d", static_cast<int>(block));
                // The seam is the second refresh a block costs. The stream
                // arrives here with what the SwiGLU half left, which is a
                // handful of levels, and the attention half about to read it
                // wants thirty. This refresh is the same operation the block
                // performs in its own middle; a stack simply needs one at
                // every seam as well.
                if (config.bootstrap_between_blocks)
                {
                    Range _rb("block.bootstrap_seam");
                    stream = bootstrap(stream, boot_key, relin_key);
                }
                stream =
                    transformer_block(stream, weights[block],
                                      config.blocks[block], galois_key,
                                      boot_key, relin_key);
            }
            return stream;
        }

        std::vector<int>
        Llama3Operator::model_rotation_indices(const ModelConfig& config)
        {
            require_layout(config.layout);

            int giant = config.giant;
            int baby = config.baby;
            bsgs_split(config.layout.d, giant, baby);

            std::set<int> indices;
            for (int r : transformer_stack_rotation_indices(config.stack))
            {
                indices.insert(r);
            }
            // The two ends of the model are projections, so they want tau and
            // the BSGS shifts of Equation (5) and nothing else. Both lists are
            // already inside the body's, which is why a model needs no wider a
            // key than its blocks do; asking for them here says so rather than
            // relying on it.
            for (int r : tau_rotation_indices(config.layout))
            {
                indices.insert(r);
            }
            for (int r : pcmm_rotation_indices(config.layout, giant, baby))
            {
                indices.insert(r);
            }
            for (int r : strided_rotation_indices(config.final_norm.stride,
                                                  config.final_norm.count))
            {
                indices.insert(r);
            }
            return std::vector<int>(indices.begin(), indices.end());
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::embed(
            std::vector<Ciphertext<Scheme::CKKS>>& one_hot,
            const BlockMatrix& table, const ModelConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key)
        {
            Range _r("embedding");
            // Nothing distinguishes a lookup from a projection: the one-hot
            // columns are an activation whose channel axis is the vocabulary,
            // and the table is the weight that reads it.
            std::vector<Ciphertext<Scheme::CKKS>> tau_one_hot =
                tau_blocks(one_hot, config.layout, galois_key);
            return project_blocks(tau_one_hot, table, config.layout,
                                  config.giant, config.baby, "embedding",
                                  galois_key, config.token_blocks);
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::output_head(
            std::vector<Ciphertext<Scheme::CKKS>>& x,
            std::vector<Plaintext<Scheme::CKKS>>& norm_weights,
            const BlockMatrix& head, const ModelConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            std::vector<Ciphertext<Scheme::CKKS>> normed;
            {
                Range _r("final_norm");
                normed = rms_norm(x, norm_weights, config.final_norm,
                                  galois_key, relin_key);
            }

            if (head.empty())
            {
                // Not a degenerate model: these are the hidden states, and a
                // caller who means to run the head themselves wants exactly
                // this and would have to pay a level to undo a head here.
                return normed;
            }

            Range _r("head");
            std::vector<Ciphertext<Scheme::CKKS>> tau_normed =
                tau_blocks(normed, config.layout, galois_key);
            return project_blocks(tau_normed, head, config.layout, config.giant,
                                  config.baby, "head", galois_key,
                                  config.token_blocks);
        }

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::forward(
            std::vector<Ciphertext<Scheme::CKKS>>& in, ModelWeights& weights,
            const ModelConfig& config, Galoiskey<Scheme::CKKS>& galois_key,
            Galoiskey<Scheme::CKKS>& boot_key, Relinkey<Scheme::CKKS>& relin_key)
        {
            Range _r("forward");

            std::vector<Ciphertext<Scheme::CKKS>> embedded;
            if (!weights.embedding.empty())
            {
                embedded = embed(in, weights.embedding, config, galois_key);
            }
            // With no table the stack runs on the caller's ciphertexts rather
            // than on a copy, because a copy is a device allocation per block.
            // They come back spent, as they do from transformer_stack itself:
            // a residual drops its operands onto a common level in place.
            std::vector<Ciphertext<Scheme::CKKS>>& entry =
                weights.embedding.empty() ? in : embedded;

            std::vector<Ciphertext<Scheme::CKKS>> stream =
                transformer_stack(entry, weights.blocks, config.stack,
                                  galois_key, boot_key, relin_key);

            if (config.bootstrap_before_head)
            {
                Range _rb("bootstrap_before_head");
                stream = bootstrap(stream, boot_key, relin_key);
            }

            return output_head(stream, weights.final_norm, weights.head, config,
                               galois_key, relin_key);
        }

    } // namespace llama
} // namespace heongpu
