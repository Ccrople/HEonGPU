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
            : context_(context), encoder_(encoder),
              arith_(context, encoder, scale), matrix_(context, layout),
              batch_encoder_(layout.k),
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

            // Baby-step / giant-step over the d diagonals. Writing
            // delta = i*n1 + j splits the one sum into n1 shifts of the SOURCE
            // and n2 shifts of an accumulator, so the key switches fall from
            // d - 1 to n1 + n2 - 2 -- 63 to 14 at d = 64, 127 to 22 at d = 128.
            // The plaintext count is unchanged at d, which is the point: a
            // plaintext multiply is not what this costs.
            //
            // Both index sets are multiples of step and smaller than d*step, so
            // they are a SUBSET of the shifts bridge_rotation_indices() already
            // asks for. No new Galois key, at any n1.
            const int n1 = baby_steps();
            const int n2 = d / n1;

            Range _r_bridge(name);

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.size());

            for (auto& source : in)
            {
                const double plain_scale = rescale_prime(source);

                // The d diagonals depend on the direction, the level and the
                // prime -- never on the data -- so every column of this call
                // wants the same d plaintexts. Encoding them once instead of
                // once per column is where this crossing's cost lived; see the
                // note on bridge_plain_. rescale_prime returns a modulus, so
                // its double is an exact integer and safe to key on.
                const auto plain_key =
                    std::make_tuple(inverse, source.depth(),
                                    static_cast<uint64_t>(plain_scale));
                // A set too wide to be worth its memory is built, used and
                // dropped rather than cached, so it cannot displace a narrow
                // set that more calls want. @see set_bridge_plain_limb_limit.
                const int this_set_limbs = set_limbs(plain_key);
                const bool cacheable =
                    bridge_plain_limb_limit_ <= 0 ||
                    this_set_limbs <= bridge_plain_limb_limit_;

                auto encoded = bridge_plain_.find(plain_key);
                EncodedDiagonalSet scratch;
                if (encoded == bridge_plain_.end())
                {
                    SuffixRange _r(name, "encode");
                    while (cacheable &&
                           bridge_plain_.size() >= bridge_plain_capacity_)
                    {
                        // Whole-set eviction, widest first. The next call that
                        // wants this level pays the encode again and gets the
                        // same plaintexts, so the only thing at stake is time.
                        evict_widest();
                    }
                    // Stored in BSGS order and PRE-ROTATED: entry i*n1 + j is
                    // diagonal[i*n1 + j] shifted back by the giant step that
                    // will be applied to the accumulator, which is what makes
                    // the two shifts compose to the one the diagonal wanted.
                    // A shift of a known vector is a relabelling on the host.
                    std::vector<Plaintext<Scheme::CKKS>> plains;
                    plains.reserve(d);
                    std::vector<Complex64> shifted(
                        static_cast<std::size_t>(slot_count_));
                    for (int i = 0; i < n2; ++i)
                    {
                        const int giant = i * n1 * step;
                        for (int j = 0; j < n1; ++j)
                        {
                            const std::vector<Complex64>& src =
                                diagonal[static_cast<std::size_t>(i) * n1 + j];
                            for (int p = 0; p < slot_count_; ++p)
                            {
                                // rot(v, -giant)[p] = v[p - giant], the inverse
                                // of the accumulator shift below.
                                const int q =
                                    ((p - giant) % slot_count_ + slot_count_) %
                                    slot_count_;
                                shifted[static_cast<std::size_t>(p)] =
                                    src[static_cast<std::size_t>(q)];
                            }
                            plains.push_back(
                                encode(shifted, plain_scale, source.depth()));
                        }
                    }
                    EncodedDiagonalSet set;
                    set.plains = std::move(plains);
                    if (cacheable)
                    {
                        encoded =
                            bridge_plain_.emplace(plain_key, std::move(set))
                                .first;
                    }
                    else
                    {
                        scratch = std::move(set);
                    }
                }

                // Either the cached entry or this call's throwaway. The rest
                // of the loop does not care which.
                EncodedDiagonalSet& live = (encoded != bridge_plain_.end())
                                               ? encoded->second
                                               : scratch;

                if (hoisted_crossings_)
                {
                    // One decomposition serves every baby shift, and each
                    // giant group is one fused multiply-accumulate launch.
                    // The modular arithmetic is exact and identically
                    // ordered, so this is the loop below to the bit, at a
                    // fraction of the work and the launches.
                    EncodedDiagonalSet& set = live;
                    if (set.packed.size() == 0)
                    {
                        set.packed = arith_.pack_bsgs_plaintexts(
                            set.plains, source.depth());
                    }

                    DeviceVector<Data64> babies;
                    {
                        SuffixRange _r(name, "rotations");
                        std::vector<int> shifts(
                            static_cast<std::size_t>(n1));
                        for (int j = 0; j < n1; ++j)
                        {
                            shifts[static_cast<std::size_t>(j)] = j * step;
                        }
                        babies = arith_.hoisted_rotation_train(
                            source, shifts, n1, galois_key);
                    }

                    Ciphertext<Scheme::CKKS> total(context_);
                    bool total_started = false;
                    {
                        SuffixRange _r(name, "diagonals");
                        for (int i = 0; i < n2; ++i)
                        {
                            Ciphertext<Scheme::CKKS> acc =
                                arith_.hoisted_bsgs_group_sum(
                                    babies, set.packed, i * n1, n1, source,
                                    plain_scale);

                            // The rescale before the giant shift, exactly as
                            // below: it commutes with the rotation and takes
                            // the rotation one limb cheaper.
                            arith_.rescale_inplace(acc);

                            if (i != 0)
                            {
                                Ciphertext<Scheme::CKKS> moved(context_);
                                arith_.rotate_rows(acc, moved, galois_key,
                                                   i * n1 * step);
                                acc = std::move(moved);
                            }

                            if (!total_started)
                            {
                                total = std::move(acc);
                                total_started = true;
                            }
                            else
                            {
                                arith_.add_inplace(total, acc);
                            }
                        }
                    }
                    out.push_back(std::move(total));
                    continue;
                }

                // The n1 baby shifts are shared by every giant step of this
                // column, so they are taken once.
                std::vector<Ciphertext<Scheme::CKKS>> baby;
                baby.reserve(n1);
                {
                    SuffixRange _r(name, "rotations");
                    for (int j = 0; j < n1; ++j)
                    {
                        if (j == 0)
                        {
                            baby.push_back(source);
                        }
                        else
                        {
                            Ciphertext<Scheme::CKKS> shifted(context_);
                            arith_.rotate_rows(source, shifted, galois_key,
                                               j * step);
                            baby.push_back(std::move(shifted));
                        }
                    }
                }

                Ciphertext<Scheme::CKKS> total(context_);
                bool total_started = false;
                {
                    SuffixRange _r(name, "diagonals");
                    for (int i = 0; i < n2; ++i)
                    {
                        Ciphertext<Scheme::CKKS> acc(context_);
                        bool started = false;
                        for (int j = 0; j < n1; ++j)
                        {
                            Plaintext<Scheme::CKKS>& plain =
                                live.plains[static_cast<std::size_t>(i) * n1 +
                                            j];

                            Ciphertext<Scheme::CKKS> term(context_);
                            arith_.multiply_plain(baby[j], plain, term);

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

                        // The giant shift is a rotation and rotate_rows refuses
                        // a ciphertext that still owes a rescale, so the rescale
                        // moves inside the loop. It commutes with the rotation
                        // and every group is at the same level, so this is n2
                        // rescales instead of one and not a change of result.
                        arith_.rescale_inplace(acc);

                        if (i != 0)
                        {
                            Ciphertext<Scheme::CKKS> moved(context_);
                            arith_.rotate_rows(acc, moved, galois_key,
                                               i * n1 * step);
                            acc = std::move(moved);
                        }

                        if (!total_started)
                        {
                            total = std::move(acc);
                            total_started = true;
                        }
                        else
                        {
                            arith_.add_inplace(total, acc);
                        }
                    }
                }

                out.push_back(std::move(total));
            }

            return out;
        }

        int Llama3BatchOperator::baby_steps() const
        {
            if (bridge_baby_steps_ > 0)
            {
                return bridge_baby_steps_;
            }
            // d is a power of two, so the balanced split is too. sqrt is the
            // minimum of n1 + d/n1, and rounding down keeps n1 <= n2 which
            // puts the cheaper half on the shifts of the accumulator.
            int n1 = 1;
            while (n1 * n1 * 2 <= layout_.d)
            {
                n1 <<= 1;
            }
            return n1;
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

        std::vector<Ciphertext<Scheme::CKKS>> Llama3BatchOperator::product(
            const std::vector<Ciphertext<Scheme::CKKS>*>& a,
            const std::vector<Ciphertext<Scheme::CKKS>*>& b, const char* name,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (static_cast<int>(a.size()) != layout_.d ||
                static_cast<int>(b.size()) != layout_.d)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " product needs both operands square at layout.d columns, "
                    "which is what Algorithm 4's three internal CMTs assume");
            }

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "matmul.%s", name);
            Range _r(range_name);

            std::vector<Ciphertext<Scheme::CKKS>> out;
            matrix_.ccmm(out, a, b, galois_key, relin_key, arith_,
                         /*rescale=*/true);

            // Algorithm 4, like Algorithm 1, only MARKS the rescale: it leaves
            // the product at scale_a * scale_b and sets rescale_required_.
            // Spending it here is not tidiness. Unspent, the plaintext's
            // centered coefficients have outgrown int64 and the very next
            // decryption throws, while every downstream operation sees a scale
            // no caller expects.
            for (auto& c : out)
            {
                arith_.rescale_inplace(c);
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
            out.column = product(lhs, rhs, name, galois_key, relin_key);
            return out;
        }

        BatchActivation
        Llama3BatchOperator::transpose(BatchActivation& in, const char* name,
                                       Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_uniform(in, name);
            if (in.columns() != layout_.d || in.rows != layout_.d)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " transpose is only defined on the square block the ring "
                    "fixes, which is layout.d by layout.d");
            }

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "transpose.%s", name);
            Range _r(range_name);

            BatchActivation out;
            out.rows = in.rows;
            out.column = in.column;
            matrix_.cmt(out.column, galois_key, arith_);
            return out;
        }

        std::vector<int> Llama3BatchOperator::product_rotation_indices() const
        {
            return get_batch_cmt_rotation_indices(layout_);
        }

        std::vector<int> Llama3BatchOperator::rotation_indices() const
        {
            std::vector<int> all = bridge_rotation_indices();
            const std::vector<int> cmt = product_rotation_indices();
            all.insert(all.end(), cmt.begin(), cmt.end());
            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
            return all;
        }

        // -------------------------------------------------------------------
        // Attention
        // -------------------------------------------------------------------

        std::vector<double>
        Llama3BatchOperator::causal_column_mask(int key) const
        {
            const int d = layout_.d;
            const int step = layout_.k / 2;

            std::vector<double> mask(slot_count_, 0.0);
            for (int u = key; u < d; ++u)
            {
                // Query u admits keys 0..u, so it keeps u + 1 of the d
                // coordinates. Weighting by sqrt(d / (u + 1)) is constant
                // along the key axis and therefore cancels in the SoftMax
                // rounds, but it leaves the sum of squares where a full row
                // would have left it.
                const double w = std::sqrt(static_cast<double>(d) /
                                           static_cast<double>(u + 1));
                for (int b = 0; b < step; ++b)
                {
                    mask[b + u * step] = w;
                }
            }
            return mask;
        }

        const std::vector<std::vector<double>>&
        Llama3BatchOperator::causal_masks()
        {
            if (causal_mask_cache_.empty())
            {
                causal_mask_cache_.reserve(layout_.d);
                for (int j = 0; j < layout_.d; ++j)
                {
                    causal_mask_cache_.push_back(causal_column_mask(j));
                }
            }
            return causal_mask_cache_;
        }

        // -------------------------------------------------------------------
        // The SoftMax seam
        // -------------------------------------------------------------------

        Llama3BatchOperator::SoftmaxLayout
        Llama3BatchOperator::softmax_layout() const
        {
            const int d = layout_.d;
            const int n1 = baby_steps();

            SoftmaxLayout out;
            // The key axis is the CIPHERTEXT axis, so the whole reduced axis
            // runs across the parts and there is nothing left to reduce inside
            // one. That is not a choice: d * (k/2) == N/2 exactly, so the slot
            // vector is already full of (input, query) and the key has
            // nowhere else to go.
            out.parts = d;
            out.count = 1;
            out.stride = slot_count_;
            out.strided = true;
            out.reduction_rotations = 0;
            // to_slots and from_slots, d columns each, n1 - 1 baby shifts and
            // n2 - 1 giant ones per column.
            out.crossing_rotations = 2 * d * (n1 + d / n1 - 2);
            // Every shift the bridge takes is a multiple of k/2 below
            // d * (k/2), which is exactly the set the CMT inside Algorithm 4
            // already forces.
            out.new_galois_indices = 0;
            return out;
        }

        int Llama3BatchOperator::softmax_seam_levels(
            const Llama3Operator::SoftmaxConfig& softmax,
            const BatchSoftmaxSeamConfig& seam)
        {
            // What a degree-D Chebyshev evaluation costs, matching
            // evaluate_poly's own recursion: ceil(log2 D).
            auto fit_levels = [](int degree)
            {
                int levels = 0;
                for (int reach = 1; reach < degree; reach <<= 1)
                {
                    levels++;
                }
                return levels;
            };

            const int mask = seam.causal ? 1 : 0;
            // The fold rides on the mask, so it needs one; and a Newton step
            // wants its argument unmapped, so it cannot have both. That is the
            // same condition Llama3Operator::softmax applies, restated here so
            // the ledger reports what will actually happen rather than what
            // was asked for.
            const bool fold = seam.fold_affine_into_mask && seam.causal &&
                              softmax.inverse_newton <= 0;
            const int exp_affine = seam.scores_carry_exp_domain ? 0 : 1;

            int soft = exp_affine + fit_levels(softmax.exp_degree) + mask;
            if (seam.refresh_denominator)
            {
                // The reciprocal is fitted above the wide track, on a
                // refreshed denominator, so a round costs the parts the square
                // and the normalising product and nothing else.
                soft += 2 * softmax.iterations;
            }
            else
            {
                soft += softmax.iterations *
                        (1                                     // square
                         + (fold ? 0 : 1)                      // 1/x affine
                         + fit_levels(softmax.inverse_degree)  // the fit
                         + 2 * softmax.inverse_newton          // Newton steps
                         + 1);                                 // normalise
            }
            // to_slots and from_slots, one level each and one is the floor:
            // a d-point linear map cannot cost less.
            return 1 + soft + 1;
        }

        BatchActivation Llama3BatchOperator::softmax_seam(
            BatchActivation& scores,
            const Llama3Operator::SoftmaxConfig& softmax,
            const BatchSoftmaxSeamConfig& seam,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key, Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int d = layout_.d;
            const int step = layout_.k / 2;

            if (scores.columns() != d)
            {
                throw std::invalid_argument(
                    "The SoftMax seam takes the square score block Algorithm 4 "
                    "hands back: exactly layout.d columns, the key on the "
                    "columns and the query on the rows");
            }
            if (scores.rows != d)
            {
                throw std::invalid_argument(
                    "A score block has layout.d rows, one per query");
            }
            // The parts are added together to form the denominator, so a
            // mismatch here would be a silently wrong sum rather than an
            // error.
            require_uniform(scores, "softmax_seam");
            if (!(softmax.bound > 0.0))
            {
                throw std::invalid_argument(
                    "The SoftMax seam needs the input range [-bound, 0]");
            }
            if (!seam.score_shift_rows.empty() &&
                static_cast<int>(seam.score_shift_rows.size()) != d)
            {
                throw std::invalid_argument(
                    "score_shift_rows holds one shift per query row, so it "
                    "must have layout.d entries");
            }
            if (!seam.score_shift_slots.empty() &&
                static_cast<int>(seam.score_shift_slots.size()) != slot_count_)
            {
                throw std::invalid_argument(
                    "score_shift_slots is a slot vector and must hold exactly "
                    "slot_count() entries");
            }
            if (seam.refresh_denominator && boot_key == nullptr)
            {
                // Falling back silently would put the fit's levels back on the
                // wide track and change the schedule the caller sized its
                // chain for.
                throw std::invalid_argument(
                    "Refreshing the SoftMax denominator needs the boot Galois "
                    "key");
            }

            Range _r_seam("softmax_seam");

            // Hoisting is a property of the operator, and the seam is not the
            // only thing that crosses; restore it so a caller's setting
            // survives.
            struct HoistGuard
            {
                Llama3BatchOperator* op;
                bool previous;
                ~HoistGuard() { op->set_hoisted_crossings(previous); }
            } guard{this, hoisted_crossings_};
            set_hoisted_crossings(seam.hoisted_crossings);

            // The d masks serve every head, so one entry per key position is
            // the whole working set. Only ever raised: a caller that wants a
            // deeper cache keeps it.
            if (seam.causal && seam.cache_masks &&
                arith_.mask_plain_capacity() < static_cast<std::size_t>(d))
            {
                arith_.set_mask_plain_capacity(static_cast<std::size_t>(d));
            }

            std::vector<Ciphertext<Scheme::CKKS>> slots;
            {
                Range _r("softmax_seam.to_slots");
                slots = to_slots(scores, galois_key);
            }

            // The shift is subtracted from the scores, so it is scaled with
            // them. It is a constant either way, so this is free either way --
            // and doing it here rather than at the call site is what lets the
            // caller quote a shift in raw score units whichever fold is on.
            const double exp_domain =
                seam.scores_carry_exp_domain ? exp_domain_scale(softmax.bound)
                                             : 1.0;
            if (!seam.score_shift_slots.empty())
            {
                std::vector<double> flat = seam.score_shift_slots;
                for (auto& entry : flat)
                {
                    entry *= -exp_domain;
                }
                arith_.add_vector(slots, flat);
            }
            else if (!seam.score_shift_rows.empty())
            {
                // Slot b + step*u holds query u of input b, so a per-query
                // shift is one slot vector, constant along the batch axis.
                std::vector<double> flat(slot_count_, 0.0);
                for (int u = 0; u < d; ++u)
                {
                    const double value = -seam.score_shift_rows[u] * exp_domain;
                    for (int b = 0; b < step; ++b)
                    {
                        flat[b + u * step] = value;
                    }
                }
                arith_.add_vector(slots, flat);
            }
            else if (seam.score_shift != 0.0)
            {
                for (auto& column : slots)
                {
                    arith_.add_constant(column, -seam.score_shift * exp_domain);
                }
            }

            Llama3Operator::SoftmaxConfig config = softmax;
            // The key axis is entirely across ciphertexts: one coordinate per
            // part, so nothing is reduced inside a ciphertext and the
            // denominator costs a slot-wise addition and no rotation. This is
            // forced by d * (k/2) == N/2, not chosen.
            config.strided = true;
            config.stride = slot_count_;
            config.count = 1;
            config.pre_scaled_input = seam.scores_carry_exp_domain;
            // Round zero's domain map rides on the causal mask, so it can only
            // be folded when there is one.
            config.fold_affine_into_mask =
                seam.fold_affine_into_mask && seam.causal;
            config.refresh_denominator = seam.refresh_denominator;

            // By reference: the table is d slot vectors and copying it per
            // head is the host-side half of the same waste the plaintext
            // cache cures on the device side.
            static const std::vector<std::vector<double>> no_masks;
            const std::vector<std::vector<double>>& masks =
                seam.causal ? causal_masks() : no_masks;
            std::vector<int> mask_ids;
            if (seam.causal)
            {
                mask_ids.reserve(d);
                for (int j = 0; j < d; ++j)
                {
                    mask_ids.push_back(seam.cache_masks ? j : -1);
                }
            }

            std::vector<Ciphertext<Scheme::CKKS>> probabilities;
            {
                Range _r("softmax_seam.softmax");
                probabilities =
                    arith_.softmax(slots, config, masks, mask_ids, galois_key,
                                   relin_key, boot_key);
            }
            slots.clear();

            Range _r("softmax_seam.from_slots");
            return from_slots(probabilities, d, galois_key);
        }

        BatchActivation Llama3BatchOperator::attention(
            BatchActivation& x, const BatchAttentionWeights& weights,
            const BatchAttentionConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key, Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int d = layout_.d;
            const int heads = config.heads;
            const int kv_heads =
                config.kv_heads > 0 ? config.kv_heads : config.heads;

            if (heads < 1 || kv_heads < 1 || heads % kv_heads != 0)
            {
                throw std::invalid_argument(
                    "Grouped-query attention needs kv_heads to divide heads");
            }
            if (config.q_channels % (heads * d) != 0 ||
                config.kv_channels % (kv_heads * d) != 0)
            {
                throw std::invalid_argument(
                    "Every head must own a whole number of layout.d channel "
                    "blocks, because Algorithm 4's operands are square at d");
            }
            const int per_head = config.q_channels / (heads * d);
            if (config.kv_channels / (kv_heads * d) != per_head)
            {
                throw std::invalid_argument(
                    "The key and value heads must be as wide as the query "
                    "heads; only their number may differ");
            }

            Range _r_attention("attention");

            // 1/sqrt(head_dim) rides on the query weight. A scaling is free on
            // the host and one homomorphic level anywhere else, and the score
            // shift that follows is an addition in slot form, which is free
            // too -- so the whole of the score calibration costs nothing.
            const double head_scale =
                config.head_scale != 0.0
                    ? config.head_scale
                    : 1.0 / std::sqrt(static_cast<double>(per_head * d));

            // And so does the domain map of the exponential, for exactly the
            // same reason. The SoftMax fits exp over [-bound, 0] and has to
            // carry its argument onto [-1, 1] first, which is a plaintext
            // product and a level -- unless the scores arrive already carrying
            // it, and every score is a linear function of this weight. So the
            // seam's assertion is made true here, where it is free.
            const double exp_domain =
                config.seam.scores_carry_exp_domain
                    ? exp_domain_scale(config.softmax.bound)
                    : 1.0;

            std::vector<double> query_weight = weights.query;
            for (auto& w : query_weight)
            {
                w *= head_scale * exp_domain;
            }

            BatchActivation q = project(x, query_weight, config.in_channels,
                                        config.q_channels, "attention.q");
            BatchActivation k = project(x, weights.key, config.in_channels,
                                        config.kv_channels, "attention.k");
            BatchActivation v = project(x, weights.value, config.in_channels,
                                        config.kv_channels, "attention.v");

            // K^T, one CMT per channel block. This is the only transpose the
            // sublayer pays for: the value product P V already reads V with
            // the key on its rows, which is where the projection left it.
            std::vector<Ciphertext<Scheme::CKKS>> key_t;
            key_t.reserve(k.column.size());
            {
                Range _r("attention.transpose_key");
                for (int base = 0; base < k.columns(); base += d)
                {
                    std::vector<Ciphertext<Scheme::CKKS>> block(
                        k.column.begin() + base, k.column.begin() + base + d);
                    matrix_.cmt(block, galois_key, arith_);
                    for (auto& c : block)
                    {
                        key_t.push_back(std::move(c));
                    }
                }
            }

            const int group = heads / kv_heads;

            // The seam's settings, with the sublayer's shape filled in. A
            // configuration that names no shift on the seam keeps the
            // sublayer-level one, so an existing caller still means what it
            // meant. Built once: the seam is identical for every head, which
            // is exactly why its masks are worth encoding once.
            BatchSoftmaxSeamConfig seam = config.seam;
            seam.causal = config.causal;
            if (seam.score_shift == 0.0 && seam.score_shift_rows.empty() &&
                seam.score_shift_slots.empty())
            {
                seam.score_shift = config.score_shift;
            }

            BatchActivation out;
            out.rows = x.rows;
            out.column.reserve(static_cast<size_t>(config.q_channels));

            for (int h = 0; h < heads; ++h)
            {
                const int q_base = h * per_head * d;
                const int kv_base = (h / group) * per_head * d;

                // S = sum_t Q[t] K[t]^T. Every term is one CCMM at the same
                // level and the same scale, so a head wider than one block
                // costs products and no depth at all.
                std::vector<Ciphertext<Scheme::CKKS>> scores;
                {
                    Range _r("attention.scores");
                    for (int t = 0; t < per_head; ++t)
                    {
                        std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
                        for (int j = 0; j < d; ++j)
                        {
                            lhs.push_back(&q.column[q_base + t * d + j]);
                            rhs.push_back(&key_t[kv_base + t * d + j]);
                        }
                        std::vector<Ciphertext<Scheme::CKKS>> term = product(
                            lhs, rhs, "attention.score", galois_key, relin_key);
                        if (scores.empty())
                        {
                            scores = std::move(term);
                        }
                        else
                        {
                            for (int j = 0; j < d; ++j)
                            {
                                arith_.add_inplace(scores[j], term[j]);
                            }
                        }
                    }
                }

                // The SoftMax is slot-wise, so this is where the sublayer
                // leaves the matrix encoding -- and the only place it does.
                // Everything between the two Algorithm-4 products is the seam,
                // and it owns the layout decision along with the two folds and
                // the auxiliary track. @see softmax_seam.
                BatchActivation score_matrix;
                score_matrix.rows = d;
                score_matrix.column = std::move(scores);

                BatchActivation p =
                    softmax_seam(score_matrix, config.softmax, seam, galois_key,
                                 relin_key, boot_key);

                // The value blocks are still where the projection left them,
                // several levels above P, so they come down to meet it. The
                // drop is free and what it discards was unreachable anyway.
                {
                    Range _r("attention.value_product");
                    const int depth = p.column.front().depth();
                    for (int t = 0; t < per_head; ++t)
                    {
                        std::vector<Ciphertext<Scheme::CKKS>> value(
                            v.column.begin() + kv_base + t * d,
                            v.column.begin() + kv_base + (t + 1) * d);
                        std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
                        for (int j = 0; j < d; ++j)
                        {
                            arith_.drop_to_depth(value[j], depth);
                            lhs.push_back(&p.column[j]);
                            rhs.push_back(&value[j]);
                        }
                        std::vector<Ciphertext<Scheme::CKKS>> head_out =
                            product(lhs, rhs, "attention.value", galois_key,
                                    relin_key);
                        for (auto& c : head_out)
                        {
                            out.column.push_back(std::move(c));
                        }
                    }
                }
            }

            if (weights.output.empty())
            {
                return out;
            }
            return project(out, weights.output, config.q_channels,
                           config.in_channels, "attention.o");
        }

        // -------------------------------------------------------------------
        // RMSNorm and SwiGLU
        // -------------------------------------------------------------------

        BatchActivation Llama3BatchOperator::rms_norm(
            BatchActivation& x, const std::vector<double>& weight,
            const BatchRMSNormConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "rms_norm");
            const int channels = x.columns();
            if (!weight.empty() &&
                static_cast<int>(weight.size()) != channels)
            {
                throw std::invalid_argument(
                    "RMSNorm takes one learned scale per channel, and a "
                    "channel is a whole ciphertext here");
            }

            Range _r("rms_norm");

            std::vector<Ciphertext<Scheme::CKKS>> slots =
                to_slots(x, galois_key);

            // A channel is a whole ciphertext, so the learned scale is a
            // constant across every slot rather than a vector over them.
            std::vector<Plaintext<Scheme::CKKS>> weights;
            if (!weight.empty())
            {
                weights.reserve(channels);
                const double plain_scale = rescale_prime(slots.front());
                for (int j = 0; j < channels; ++j)
                {
                    std::vector<double> flat(slot_count_, weight[j]);
                    Plaintext<Scheme::CKKS> plain(context_);
                    encoder_.encode(plain, flat, plain_scale);
                    for (int i = 0; i < slots.front().depth(); ++i)
                    {
                        arith_.mod_drop_inplace(plain);
                    }
                    weights.push_back(std::move(plain));
                }
            }

            Llama3Operator::RMSNormConfig slot_config;
            // The channel axis runs entirely across ciphertexts, so nothing is
            // reduced inside one and the mean costs a slot-wise addition.
            slot_config.stride = slot_count_;
            slot_config.count = 1;
            slot_config.channels = channels;
            slot_config.token_blocks = 1;
            slot_config.eps = config.eps;
            slot_config.sum_lo = config.sum_lo;
            slot_config.sum_hi = config.sum_hi;
            slot_config.degree = config.degree;
            slot_config.newton_iterations = config.newton_iterations;

            std::vector<Ciphertext<Scheme::CKKS>> normalised = arith_.rms_norm(
                slots, weights, slot_config, galois_key, relin_key);

            return from_slots(normalised, x.rows, galois_key);
        }

        BatchActivation Llama3BatchOperator::feed_forward(
            BatchActivation& x, const BatchFeedForwardWeights& weights,
            const BatchFeedForwardConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "feed_forward");

            Range _r("feed_forward");

            const int in_channels = config.in_channels;
            const int hidden_channels = config.hidden_channels;
            if (weights.gate.size() != static_cast<size_t>(in_channels) *
                                           static_cast<size_t>(hidden_channels)
                || weights.up.size() != weights.gate.size()
                || weights.down.size() != weights.gate.size())
            {
                throw std::invalid_argument(
                    "The SwiGLU weights must be in_channels by "
                    "hidden_channels, and down its transpose shape");
            }

            // The hidden axis is streamed rather than held whole. See the
            // config for why: the Hadamard product needs both branches in both
            // encodings at once, so the whole width would be four ciphertexts
            // per hidden channel resident simultaneously.
            int block = config.hidden_block > 0 ? config.hidden_block
                                                : hidden_channels;
            block = std::min(block, hidden_channels);

            BatchActivation out;
            out.rows = x.rows;

            for (int base = 0; base < hidden_channels; base += block)
            {
                const int cols = std::min(block, hidden_channels - base);

                // The gate and up weights are in_channels x hidden_channels,
                // so a chunk of the hidden axis is a set of columns and has to
                // be gathered. The down weight is hidden_channels x
                // in_channels, so the same chunk is a contiguous span of rows.
                std::vector<double> gate_w(static_cast<size_t>(in_channels) *
                                           cols);
                std::vector<double> up_w(gate_w.size());
                for (int i = 0; i < in_channels; ++i)
                {
                    const size_t src =
                        static_cast<size_t>(i) * hidden_channels + base;
                    const size_t dst = static_cast<size_t>(i) * cols;
                    for (int j = 0; j < cols; ++j)
                    {
                        gate_w[dst + j] = weights.gate[src + j];
                        up_w[dst + j] = weights.up[src + j];
                    }
                }

                BatchActivation gate =
                    project(x, gate_w, in_channels, cols, "ffn.gate");
                BatchActivation up =
                    project(x, up_w, in_channels, cols, "ffn.up");

                // The gate meets the up projection in a Hadamard product,
                // which the matrix encoding does not have: multiplying two
                // columns convolves their coefficients. Both branches
                // therefore cross to slot form, where a product is slot-wise,
                // and the result crosses back for the down projection.
                std::vector<Ciphertext<Scheme::CKKS>> gate_slots =
                    to_slots(gate, galois_key);
                std::vector<Ciphertext<Scheme::CKKS>> up_slots =
                    to_slots(up, galois_key);
                gate.column.clear();
                up.column.clear();

                std::vector<Ciphertext<Scheme::CKKS>> hidden;
                hidden.reserve(gate_slots.size());
                {
                    Range _r_silu("ffn.silu");
                    for (size_t j = 0; j < gate_slots.size(); ++j)
                    {
                        Ciphertext<Scheme::CKKS> activated =
                            arith_.silu(gate_slots[j], config.silu_bound,
                                        config.silu_degree, relin_key);
                        hidden.push_back(arith_.multiply_and_rescale(
                            activated, up_slots[j], relin_key));
                    }
                }
                gate_slots.clear();
                up_slots.clear();

                BatchActivation h = from_slots(hidden, x.rows, galois_key);
                hidden.clear();

                const std::vector<double> down_w(
                    weights.down.begin() +
                        static_cast<size_t>(base) * in_channels,
                    weights.down.begin() +
                        static_cast<size_t>(base + cols) * in_channels);
                BatchActivation part =
                    project(h, down_w, cols, in_channels, "ffn.down");

                if (out.column.empty())
                {
                    out.column = std::move(part.column);
                }
                else
                {
                    // Every chunk's partial product leaves the same sequence
                    // of operations behind it, so they meet at one level and
                    // one scale and the sum is a plain addition.
                    Range _r_acc("ffn.accumulate");
                    for (size_t j = 0; j < out.column.size(); ++j)
                    {
                        arith_.add_inplace(out.column[j], part.column[j]);
                    }
                }
            }

            return out;
        }

        BatchActivation Llama3BatchOperator::transformer_block(
            BatchActivation& x, const BatchTransformerBlockWeights& weights,
            const BatchTransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "transformer_block");

            Range _r("transformer_block");

            BatchActivation stream;
            stream.rows = x.rows;
            stream.column = x.column;

            {
                Range _r_half("transformer_block.attention_half");
                BatchActivation normed =
                    rms_norm(stream, weights.attention_norm,
                             config.attention_norm, galois_key, relin_key);
                BatchActivation sublayer =
                    attention(normed, weights.attention, config.attention,
                              galois_key, relin_key);
                // The two operands have been through completely different
                // circuits, so neither the level nor the scale lines up; this
                // is the one level a pre-norm residual pays.
                stream.column = arith_.residual_add(stream.column,
                                                    sublayer.column);
            }

            {
                Range _r_half("transformer_block.feed_forward_half");
                BatchActivation normed =
                    rms_norm(stream, weights.feed_forward_norm,
                             config.feed_forward_norm, galois_key, relin_key);
                BatchActivation sublayer =
                    feed_forward(normed, weights.feed_forward,
                                 config.feed_forward, galois_key, relin_key);
                stream.column = arith_.residual_add(stream.column,
                                                    sublayer.column);
            }

            return stream;
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
