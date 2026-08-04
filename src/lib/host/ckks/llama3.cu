// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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
              encoder_(encoder), slot_count_(encoder.slot_count()),
              default_scale_(scale)
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
            if (level < 0 || level >= static_cast<int>(
                                          context_->prime_vector_.size()))
            {
                throw std::invalid_argument(
                    "Ciphertext has no level left to rescale");
            }
            return static_cast<double>(context_->prime_vector_[level].value);
        }

        Plaintext<Scheme::CKKS>
        Llama3Operator::encode(const std::vector<double>& values, double scale)
        {
            Plaintext<Scheme::CKKS> plain(context_);
            encoder_.encode(plain, values, scale);
            return plain;
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
            Plaintext<Scheme::CKKS> plain = encode(values, rescale_prime(ct));
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
            Plaintext<Scheme::CKKS> plain = encode(values, rescale_prime(ct));
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
                add_inplace(ct, shifted);
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
                add_inplace(ct, shifted);
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
                add_inplace(ct, shifted);
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
            if (config.channels <= 0)
            {
                throw std::invalid_argument("RMSNorm needs a channel count");
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
                add_inplace(total, term);
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
                    multiply_plain_inplace(normalised, weights[i]);
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

            const double d = static_cast<double>(config.count);

            Ciphertext<Scheme::CKKS> y = exp_scaled_negative(
                ct, config.bound, config.iterations, config.exp_degree,
                relin_key);

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
            Ciphertext<Scheme::CKKS> swapped(context_);
            rotate_rows(ct, swapped, galois_key, swap_shift);

            Ciphertext<Scheme::CKKS> direct(context_);
            multiply_plain(ct, cos_plain, direct);
            rescale_inplace(direct);

            multiply_plain_inplace(swapped, sin_plain);
            rescale_inplace(swapped);

            add_inplace(direct, swapped);
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
            if (giant * baby != layout.d)
            {
                throw std::invalid_argument(
                    "BSGS factors must multiply to the matrix dimension");
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
            if (giant * baby != layout.d)
            {
                throw std::invalid_argument(
                    "BSGS factors must multiply to the matrix dimension");
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

                    Plaintext<Scheme::CKKS> plain = encode(slots, plain_scale);
                    Ciphertext<Scheme::CKKS> term(context_);
                    multiply_plain(rotated[i], plain, term);

                    if (!inner_started)
                    {
                        inner = term;
                        inner_started = true;
                    }
                    else
                    {
                        add_inplace(inner, term);
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
                    add_inplace(result, inner);
                }
            }

            return result;
        }

    } // namespace llama
} // namespace heongpu
