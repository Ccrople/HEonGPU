// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
            // intermediate has grown enough to need rescaling. Nothing here
            // bootstraps, so seed it with the working scale to give that test
            // the meaning it has inside bootstrapping.
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

        void Llama3Operator::accumulate_masked(
            Ciphertext<Scheme::CKKS>& acc, bool& started,
            Ciphertext<Scheme::CKKS>& ct, const std::vector<double>& values,
            double plain_scale, const char* context)
        {
            Plaintext<Scheme::CKKS> plain =
                encode(values, plain_scale, ct.depth());
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

            for (int t = 1; t < count; t <<= 1)
            {
                Ciphertext<Scheme::CKKS> shifted(context_);
                rotate_rows(ct, shifted, galois_key, stride * t);
                add_same_scale(ct, shifted, "sum_strided");
            }
        }

        void Llama3Operator::sum_blocked(Ciphertext<Scheme::CKKS>& ct, int span,
                                         Galoiskey<Scheme::CKKS>& galois_key)
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
                return;
            }

            // Rotate-and-add gives slot p the window sum over p .. p+span-1,
            // which is the block total only where p is a multiple of span.
            for (int s = 1; s < span; s <<= 1)
            {
                Ciphertext<Scheme::CKKS> shifted(context_);
                rotate_rows(ct, shifted, galois_key, s);
                add_same_scale(ct, shifted, "sum_blocked window");
            }

            // Keep those positions and discard the rest.
            std::vector<double> mask(slot_count_, 0.0);
            for (int p = 0; p < slot_count_; p += span)
            {
                mask[p] = 1.0;
            }
            multiply_vector(ct, mask);

            // Fan each surviving total across its own block. Shifting right by
            // less than span cannot cross into the next block, because every
            // other slot of the block is zero.
            for (int s = 1; s < span; s <<= 1)
            {
                Ciphertext<Scheme::CKKS> shifted(context_);
                rotate_rows(ct, shifted, galois_key, -s);
                add_same_scale(ct, shifted, "sum_blocked fan-out");
            }
        }

        // -------------------------------------------------------------------
        // Polynomial primitives
        // -------------------------------------------------------------------

        Ciphertext<Scheme::CKKS> Llama3Operator::evaluate_chebyshev(
            Ciphertext<Scheme::CKKS>& ct, const std::vector<double>& coeffs,
            double a, double b, Relinkey<Scheme::CKKS>& relin_key)
        {
            if (coeffs.empty())
            {
                throw std::invalid_argument("Chebyshev series is empty");
            }
            if (!(b > a))
            {
                throw std::invalid_argument("Interval must satisfy a < b");
            }

            Ciphertext<Scheme::CKKS> t = ct;
            if (a != -1.0 || b != 1.0)
            {
                // T_k is defined on [-1, 1], so the argument is mapped there
                // first. This is the one level the affine map costs.
                multiply_constant(t, 2.0 / (b - a));
                add_constant(t, -(a + b) / (b - a));
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

            return evaluate_poly(t, t.scale(), poly, relin_key,
                                 ExecutionOptions());
        }

        Ciphertext<Scheme::CKKS>
        Llama3Operator::evaluate_function(Ciphertext<Scheme::CKKS>& ct,
                                          const std::function<double(double)>& f,
                                          double a, double b, int degree,
                                          Relinkey<Scheme::CKKS>& relin_key)
        {
            return evaluate_chebyshev(ct, chebyshev_coefficients(f, a, b, degree),
                                      a, b, relin_key);
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

            Ciphertext<Scheme::CKKS> y = evaluate_function(
                ct, [](double x) { return 1.0 / std::sqrt(x); }, lo, hi, degree,
                relin_key);

            if (newton_iterations <= 0)
            {
                return y;
            }

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
                                Relinkey<Scheme::CKKS>& relin_key)
        {
            if (!(lo > 0.0))
            {
                throw std::invalid_argument(
                    "1/x needs a strictly positive lower bound");
            }

            Ciphertext<Scheme::CKKS> y = evaluate_function(
                ct, [](double x) { return 1.0 / x; }, lo, hi, degree,
                relin_key);

            // y <- y (2 - x y), two levels a step.
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
                             int degree, Relinkey<Scheme::CKKS>& relin_key)
        {
            return evaluate_function(
                ct, [](double x) { return x / (1.0 + std::exp(-x)); }, -bound,
                bound, degree, relin_key);
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::exp_scaled_negative(
            Ciphertext<Scheme::CKKS>& ct, double bound, int squarings,
            int degree, Relinkey<Scheme::CKKS>& relin_key)
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

            // Approximating exp(x / 2^k) rather than exp(x) folds the scaling
            // step of the SoftMax algorithm into the fit, so it is free.
            const double divisor = std::pow(2.0, squarings);
            return evaluate_function(
                ct, [divisor](double x) { return std::exp(x / divisor); },
                -bound, 0.0, degree, relin_key);
        }

        // -------------------------------------------------------------------
        // Llama-3 layers
        // -------------------------------------------------------------------

        std::vector<Ciphertext<Scheme::CKKS>> Llama3Operator::rms_norm(
            std::vector<Ciphertext<Scheme::CKKS>>& in,
            std::vector<Plaintext<Scheme::CKKS>>& weights,
            const RMSNormConfig& config, Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (in.empty())
            {
                throw std::invalid_argument("RMSNorm needs at least one input");
            }
            if (!weights.empty() && weights.size() != in.size())
            {
                throw std::invalid_argument(
                    "RMSNorm needs one weight plaintext per input, or none");
            }
            // The reduction covers count channels in each of the inputs, and
            // only the last one may be partly padding, so the true channel
            // count is pinned between those two bounds. Getting this wrong
            // scales every output by a constant and nothing else complains.
            const int reduced =
                config.count * static_cast<int>(in.size());
            if (config.channels <= reduced - config.count ||
                config.channels > reduced)
            {
                throw std::invalid_argument(
                    "RMSNorm's channel count must lie in (count * (inputs - "
                    "1), count * inputs]: the reduction covers count channels "
                    "per input and only the last input may be padded");
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

            // The channels of a token may be split over several ciphertexts,
            // so the squares are accumulated before the reduction and the mean
            // covers every channel.
            Ciphertext<Scheme::CKKS> total = in[0];
            square(total, relin_key);
            for (std::size_t i = 1; i < in.size(); i++)
            {
                Ciphertext<Scheme::CKKS> term = in[i];
                square(term, relin_key);
                add_same_scale(total, term, "rms_norm channel sum");
            }

            sum_strided(total, config.stride, config.count, galois_key);

            // mean + eps. Both are one cheap step on an already reduced value.
            multiply_constant(total, 1.0 / static_cast<double>(config.channels));
            add_constant(total, config.eps);

            const double lo =
                config.sum_lo / static_cast<double>(config.channels) +
                config.eps;
            const double hi =
                config.sum_hi / static_cast<double>(config.channels) +
                config.eps;

            Ciphertext<Scheme::CKKS> scale_factor =
                inverse_sqrt(total, lo, hi, config.degree,
                             config.newton_iterations, relin_key);

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.size());
            for (std::size_t i = 0; i < in.size(); i++)
            {
                Ciphertext<Scheme::CKKS> normalised =
                    multiply_and_rescale(in[i], scale_factor, relin_key);

                if (!weights.empty())
                {
                    multiply_plaintext(normalised, weights[i]);
                    rescale_inplace(normalised);
                }

                out.push_back(normalised);
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
            if (!mask.empty() &&
                static_cast<int>(mask.size()) != slot_count_)
            {
                throw std::invalid_argument(
                    "A SoftMax mask must hold exactly slot_count() entries");
            }
            if (config.count <= 1 || !is_power_of_two(config.count))
            {
                throw std::invalid_argument(
                    "SoftMax length must be a power of two above one");
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

            const double d = static_cast<double>(config.count);

            Ciphertext<Scheme::CKKS> y = exp_scaled_negative(
                ct, config.bound, config.iterations, config.exp_degree,
                relin_key);

            // Masking the exponentials rather than the scores removes a
            // coordinate from the numerator and from the sum at once. The
            // rounds below normalise, so a mask weight that is constant along
            // the reduced axis cancels and only the pattern of zeros survives.
            if (!mask.empty())
            {
                multiply_vector(y, mask);
            }

            for (int round = 0; round < config.iterations; round++)
            {
                // (y_i / ||y||_2)^2 is y_i^2 / sum_j y_j^2, so one squaring
                // serves both the numerator and the sum.
                Ciphertext<Scheme::CKKS> y_squared = y;
                square(y_squared, relin_key);

                Ciphertext<Scheme::CKKS> total = y_squared;
                if (config.strided)
                {
                    sum_strided(total, config.stride, config.count, galois_key);
                }
                else
                {
                    sum_blocked(total, config.count, galois_key);
                }

                // Before the first round every coordinate sits in
                // [exp(-bound / 2^k), 1], afterwards the coordinates sum to one
                // so the sum of squares is confined to [1/d, 1].
                double lo;
                double hi;
                if (round == 0)
                {
                    const double smallest = std::exp(
                        -2.0 * config.bound / std::pow(2.0, config.iterations));
                    lo = d * smallest * 0.5;
                    hi = d * 1.5;
                }
                else
                {
                    lo = 0.5 / d;
                    hi = 1.5;
                }

                Ciphertext<Scheme::CKKS> reciprocal =
                    inverse(total, lo, hi, config.inverse_degree,
                            config.inverse_newton, relin_key);

                y = multiply_and_rescale(y_squared, reciprocal, relin_key);
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

            Ciphertext<Scheme::CKKS> swapped(context_);
            rotate_rows(ct, swapped, galois_key, swap_shift);

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

            // Baby step: the row rotations of the ciphertext are shared by
            // every giant step, so they are taken once.
            const int row = d * batch;
            std::vector<Ciphertext<Scheme::CKKS>> rotated;
            rotated.reserve(baby);
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
                    std::vector<double> slots(slot_count_, 0.0);
                    for (int m = 0; m < batch; m++)
                    {
                        const double* source =
                            a_stored.data() +
                            (shared ? 0 : static_cast<std::size_t>(m) * entries);
                        std::vector<double> matrix(source, source + entries);
                        const std::vector<double> block = pcmm_plaintext_block(
                            matrix, d, i, j, baby, ell);

                        for (std::size_t e = 0; e < entries; e++)
                        {
                            slots[e * batch + m] = block[e];
                        }
                    }

                    Plaintext<Scheme::CKKS> plain =
                        encode(slots, plain_scale, ct.depth());
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

            Ciphertext<Scheme::CKKS> lhs = a;
            Ciphertext<Scheme::CKKS> rhs = b;
            align_levels(lhs, rhs);

            // Right operand: tau costs one level, and the d - 1 row rotations
            // that follow it cost none.
            Ciphertext<Scheme::CKKS> tb = tau(rhs, layout, galois_key);
            std::vector<Ciphertext<Scheme::CKKS>> right;
            right.reserve(d);
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

            // Left operand: rot_C^k(sigma(A))[i][j] is A[i][(i + j + k) mod d],
            // which is still a single diagonal of A per shift, so all d of
            // them are built from one set of 2d - 2 rotations and differ only
            // in their masks. That is what keeps the whole product at depth
            // two instead of the three sigma and the column rotation would
            // cost separately.
            std::vector<Ciphertext<Scheme::CKKS>> diagonal;
            diagonal.reserve(2 * d - 1);
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
                bool left_started = false;
                for (int t = 0; t < 2 * d - 1; t++)
                {
                    if (masks[t].empty())
                    {
                        continue;
                    }
                    accumulate_masked(left, left_started, diagonal[t], masks[t],
                                      plain_scale, "ccmm left operand");
                }
                rescale_inplace(left);

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
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " needs at least one channel block");
            }

            const int depth = in.front().depth();
            const double scale = in.front().scale();
            for (std::size_t j = 1; j < in.size(); j++)
            {
                if (in[j].depth() != depth)
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " channel blocks sit at different levels, so the block "
                        "sums below them cannot be taken");
                }
                // The same relative test add_same_scale uses: rescaling drifts
                // scales, and what must not happen is two blocks drifting
                // apart before they are added together.
                if (std::abs(in[j].scale() - scale) >
                    1e-9 * std::max(std::abs(scale), std::abs(in[j].scale())))
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " channel blocks sit at different scales, so the block "
                        "sums below them would be silently wrong");
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
            int baby, const char* name, Galoiskey<Scheme::CKKS>& galois_key)
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
            if (weight.in_blocks != static_cast<int>(tau_x.size()))
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " weight expects a different number of input channel "
                    "blocks than it was given");
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(weight.out_blocks);

            for (int i = 0; i < weight.out_blocks; i++)
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

                    Ciphertext<Scheme::CKKS> term = project(
                        tau_x[j], block, layout, giant, baby, name, galois_key);

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
                        std::string("Row ") + std::to_string(i) + " of the " +
                        name +
                        " weight is entirely empty, so that output block would "
                        "be a zero this cannot produce");
                }
                out.push_back(row);
            }

            return out;
        }

        void Llama3Operator::rope_blocks(
            std::vector<Ciphertext<Scheme::CKKS>>& in, int per_head,
            std::vector<Plaintext<Scheme::CKKS>>& rope_plain,
            const MatrixLayout& layout, Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (static_cast<int>(rope_plain.size()) != 2 * per_head)
            {
                throw std::invalid_argument(
                    "RoPE needs a cosine and a sine plaintext for each channel "
                    "block of a head");
            }

            if (per_head == 1)
            {
                // The head is one block wide, so pairing channel c with
                // c + d / 2 stays inside it and is the row rotation rope()
                // takes.
                const int swap = rope_swap_shift(layout);
                for (auto& block : in)
                {
                    block = rope(block, rope_plain[0], rope_plain[1], swap,
                                 galois_key);
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
            // already there in the choice of which two blocks to combine.
            const int half = per_head / 2;
            const int heads = static_cast<int>(in.size()) / per_head;
            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.size());

            for (int h = 0; h < heads; h++)
            {
                for (int t = 0; t < per_head; t++)
                {
                    const int index = h * per_head + t;
                    const int partner =
                        h * per_head + (t < half ? t + half : t - half);

                    if (rope_plain[2 * t].scale() !=
                        rope_plain[2 * t + 1].scale())
                    {
                        throw std::invalid_argument(
                            "RoPE needs the cosine and sine plaintexts at one "
                            "scale");
                    }

                    Ciphertext<Scheme::CKKS> direct = in[index];
                    multiply_plaintext(direct, rope_plain[2 * t]);
                    rescale_inplace(direct);

                    Ciphertext<Scheme::CKKS> crossed = in[partner];
                    multiply_plaintext(crossed, rope_plain[2 * t + 1]);
                    rescale_inplace(crossed);

                    add_same_scale(direct, crossed, "rope across blocks");
                    out.push_back(direct);
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
            require_layout(layout);

            const int d = layout.d;
            const int batch = layout.batch;
            std::vector<double> mask(static_cast<std::size_t>(layout.slots),
                                     0.0);

            for (int query = 0; query < d; query++)
            {
                // A query attends to itself and everything before it, so the
                // number of surviving keys grows down the sequence. Weighting
                // by sqrt(d / kept) leaves the sum of squares in the range a
                // full row would give, which is the range the first round's
                // reciprocal is fitted over; the rounds normalise, so a weight
                // constant along the key axis cancels and the result is
                // unchanged.
                const double weight =
                    std::sqrt(static_cast<double>(d) /
                              static_cast<double>(query + 1));
                for (int key = 0; key <= query; key++)
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

            BlockAttentionWeights blocked;
            blocked.query = BlockMatrix(weights.query);
            blocked.key = BlockMatrix(weights.key);
            blocked.value = BlockMatrix(weights.value);
            blocked.output = BlockMatrix(weights.output);

            AttentionConfig one = config;
            one.heads = 1;

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

            if (all_projections)
            {
                if (weights.query.out_blocks != weights.key.out_blocks ||
                    weights.query.out_blocks != weights.value.out_blocks)
                {
                    throw std::invalid_argument(
                        "The query, key and value weights must produce the "
                        "same number of channel blocks, since that is what the "
                        "heads are shared out between");
                }

                // One tau of each input block serves all three projections,
                // which is the only reason an isolated pcmm's tau is
                // affordable here.
                std::vector<Ciphertext<Scheme::CKKS>> tau_x =
                    tau_blocks(x, layout, galois_key);
                query = project_blocks(tau_x, weights.query, layout, giant,
                                       baby, "query", galois_key);
                key = project_blocks(tau_x, weights.key, layout, giant, baby,
                                     "key", galois_key);
                value = project_blocks(tau_x, weights.value, layout, giant,
                                       baby, "value", galois_key);
            }
            else
            {
                query = x;
                key = x;
                value = x;
            }

            const int qkv = static_cast<int>(query.size());
            const int heads = config.heads > 0 ? config.heads : 1;
            if (qkv % heads != 0)
            {
                throw std::invalid_argument(
                    "The heads must share the query, key and value channel "
                    "blocks out evenly between them");
            }
            const int per_head = qkv / heads;

            if (config.rope)
            {
                rope_blocks(query, per_head, rope_plain, layout, galois_key);
                rope_blocks(key, per_head, rope_plain, layout, galois_key);
            }

            SoftmaxConfig softmax_config = config.softmax;
            softmax_config.strided = true;
            softmax_config.stride = d * layout.batch;
            softmax_config.count = d;

            const std::vector<double> mask =
                config.causal ? causal_mask(layout) : std::vector<double>();

            // head_dim is the channel blocks a head owns, times d.
            const double head =
                config.head_scale > 0.0
                    ? config.head_scale
                    : 1.0 / std::sqrt(static_cast<double>(per_head) * d);

            std::vector<Ciphertext<Scheme::CKKS>> probabilities;
            probabilities.reserve(heads);

            for (int h = 0; h < heads; h++)
            {
                // The scores are formed as K^T Q rather than Q^T K. That puts
                // the key position on the slow axis, so the SoftMax
                // denominator is the exact strided reduction and costs no
                // level and no mask, and what comes out is already P^T, which
                // is the operand the value product wants.
                //
                // A head wider than one block sums its blocks' scores. Each
                // term is a ccmm at the same level and scale, so the sum over
                // the head is free and the SoftMax below runs once.
                Ciphertext<Scheme::CKKS> scores(context_);
                bool started = false;

                for (int t = 0; t < per_head; t++)
                {
                    const int index = h * per_head + t;
                    Ciphertext<Scheme::CKKS> key_t =
                        transpose(key[index], layout, galois_key);
                    Ciphertext<Scheme::CKKS> term = ccmm(
                        key_t, query[index], layout, head, galois_key,
                        relin_key);

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

                if (config.score_shift != 0.0)
                {
                    // Translating the scores into [-bound, 0] is calibration
                    // in the paper, not a homomorphic maximum, so it is a
                    // constant.
                    add_constant(scores, -config.score_shift);
                }

                probabilities.push_back(softmax(scores, softmax_config, mask,
                                                galois_key, relin_key));
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(qkv);
            for (int h = 0; h < heads; h++)
            {
                for (int t = 0; t < per_head; t++)
                {
                    const int index = h * per_head + t;
                    out.push_back(ccmm(value[index], probabilities[h], layout,
                                       1.0, galois_key, relin_key));
                }
            }

            if (!weights.output.empty())
            {
                std::vector<Ciphertext<Scheme::CKKS>> tau_out =
                    tau_blocks(out, layout, galois_key);
                out = project_blocks(tau_out, weights.output, layout, giant,
                                     baby, "output", galois_key);
            }

            return out;
        }

        Ciphertext<Scheme::CKKS> Llama3Operator::feed_forward(
            Ciphertext<Scheme::CKKS>& x, const FeedForwardWeights& weights,
            const FeedForwardConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
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

            // The gate and the up projection read the same input, so the tau
            // Equation (5) consumes is taken once for both.
            std::vector<Ciphertext<Scheme::CKKS>> tau_x =
                tau_blocks(x, layout, galois_key);
            std::vector<Ciphertext<Scheme::CKKS>> gate = project_blocks(
                tau_x, weights.gate, layout, giant, baby, "gate", galois_key);
            std::vector<Ciphertext<Scheme::CKKS>> up = project_blocks(
                tau_x, weights.up, layout, giant, baby, "up", galois_key);

            std::vector<Ciphertext<Scheme::CKKS>> hidden;
            hidden.reserve(gate.size());
            for (std::size_t i = 0; i < gate.size(); i++)
            {
                Ciphertext<Scheme::CKKS> activated =
                    silu(gate[i], config.silu_bound, config.silu_degree,
                         relin_key);
                hidden.push_back(
                    multiply_and_rescale(activated, up[i], relin_key));
            }

            std::vector<Ciphertext<Scheme::CKKS>> tau_hidden =
                tau_blocks(hidden, layout, galois_key);
            return project_blocks(tau_hidden, weights.down, layout, giant, baby,
                                  "down", galois_key);
        }

    } // namespace llama
} // namespace heongpu
