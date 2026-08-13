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

                // The weight is real and shared by the batch -- the k/2
                // instances are k/2 independent inputs through ONE model -- so
                // its R_k entries are CONSTANTS, and Definition 1 encodes a
                // constant as the constant polynomial. The general encoder
                // spends an O(k^2) transform per entry to discover that, and
                // then k - 1 zeros per entry ride through the upload and four
                // subring transform passes to multiply by nothing.
                // encode_shared_plaintext_matrix writes the one number that
                // carries the information. At the 8B projection shape that is a
                // 32-fold plaintext upload and gigabytes of subring temporaries
                // that stop happening -- and it is also the more accurate of
                // the two, because plain_scale here is a PRIME (40 to 60 bits),
                // which is exactly where the general encoder's rounding residue
                // stops vanishing.
                std::vector<double> slice(static_cast<size_t>(in_channels) *
                                          cols);
                for (int i = 0; i < in_channels; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        slice[static_cast<size_t>(i) * cols + j] =
                            weight[static_cast<size_t>(i) * out_channels +
                                   base + j];
                    }
                }

                {
                    SuffixRange _r(range_name, "weight_encode");
                    matrix_.encode_shared_plaintext_matrix(
                        slice, in_channels, cols, x.column.front().depth(),
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
            Relinkey<Scheme::CKKS>& relin_key,
            HEBatchMatrixOperator<Scheme::CKKS>::RightOperandForm b_form)
        {
            if (static_cast<int>(a.size()) != layout_.d ||
                static_cast<int>(b.size()) != layout_.d)
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " product needs both operands square at layout.d columns, "
                    "which is what Algorithm 4's internal CMTs assume");
            }

            // ccmm checks the count and the level and then writes
            // scale_a * scale_b unconditionally. A scale mismatch therefore
            // passes every guard and comes back as a silently wrong product,
            // so it is caught here: matmul() screens its operands through
            // require_uniform, but attention() calls this directly on
            // borrowed columns and nothing else looks.
            const double scale_a = a[0]->scale();
            const double scale_b = b[0]->scale();
            for (const auto* c : a)
            {
                if (c->scale() != scale_a)
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " product's left operand has drifted in scale across "
                        "its columns");
                }
            }
            for (const auto* c : b)
            {
                if (c->scale() != scale_b)
                {
                    throw std::invalid_argument(
                        std::string("The ") + name +
                        " product's right operand has drifted in scale across "
                        "its columns");
                }
            }

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "matmul.%s", name);
            Range _r(range_name);

            std::vector<Ciphertext<Scheme::CKKS>> out;
            matrix_.ccmm(out, a, b, galois_key, relin_key, arith_,
                         /*rescale=*/true, b_form);

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
            // The single-block case IS the blocked case at query block zero,
            // weights and all, so it is written once. A second formula here
            // would be a second thing to keep in step, and a mask that has
            // drifted is a reciprocal fitted over the wrong range -- silent.
            return causal_column_mask(0, 0, key);
        }

        std::vector<double>
        Llama3BatchOperator::causal_column_mask(int query_block, int key_block,
                                                int key) const
        {
            const int d = layout_.d;
            const int step = layout_.k / 2;

            if (query_block < 0 || key_block < 0 || key < 0 || key >= d)
            {
                throw std::invalid_argument(
                    "A causal mask needs non-negative block indices and a key "
                    "inside [0, layout.d)");
            }
            if (key_block > query_block)
            {
                // Not a clamp. A future key block is never fed to the seam at
                // all, so asking for its mask means the schedule is wrong, and
                // returning an all-zero mask would hide that.
                throw std::invalid_argument(
                    "A causal query block never sees a key block above it, so "
                    "that mask is not something the schedule can want");
            }

            // The seam reduces over every key block 0..query_block at once, so
            // a full row for this query block is (query_block + 1) * d keys.
            const double full =
                static_cast<double>(query_block + 1) * static_cast<double>(d);

            std::vector<double> mask(slot_count_, 0.0);
            // Below the diagonal every query sees every key; on the diagonal,
            // query u sees keys 0..u of this block -- and every key of the
            // blocks below, which is where the weight's numerator comes from.
            const int first = key_block == query_block ? key : 0;
            for (int u = first; u < d; ++u)
            {
                // Query u of block p admits p*d + u + 1 keys. Weighting by
                // sqrt(full / visible) is constant along the key axis and
                // therefore cancels exactly in the SoftMax rounds, but it
                // leaves the sum of squares where a full row would have left
                // it -- which is the range the reciprocal is fitted over.
                const double visible =
                    static_cast<double>(query_block) * static_cast<double>(d) +
                    static_cast<double>(u) + 1.0;
                const double w = std::sqrt(full / visible);
                for (int b = 0; b < step; ++b)
                {
                    mask[b + u * step] = w;
                }
            }
            return mask;
        }

        int Llama3BatchOperator::causal_mask_id(int query_block, int key_block,
                                                int key) const
        {
            if (query_block < 0 || key_block < 0 || key < 0 ||
                key >= layout_.d || key_block > query_block)
            {
                throw std::invalid_argument(
                    "causal_mask_id takes the same indices causal_column_mask "
                    "does");
            }
            // d + 1 ids per query block: the d triangles of its own diagonal,
            // then ONE shared id for every block below it, since they all use
            // the same full-visibility mask. At query block zero this is
            // exactly the key index, so the single-block path keeps the ids --
            // and therefore the encoded plaintexts -- it already had.
            return query_block * (layout_.d + 1) +
                   (key_block == query_block ? key : layout_.d);
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

        const std::vector<std::vector<double>>&
        Llama3BatchOperator::causal_masks_blocked(int query_block)
        {
            if (query_block < 0)
            {
                throw std::invalid_argument(
                    "A query block index is non-negative");
            }
            auto found = causal_block_mask_cache_.find(query_block);
            if (found != causal_block_mask_cache_.end())
            {
                return found->second;
            }

            std::vector<std::vector<double>> table;
            table.reserve(static_cast<std::size_t>(layout_.d + 1));
            if (query_block == 0)
            {
                // Copy the single-block table rather than recompute it: the
                // two agree by construction (causal_column_mask(key) IS
                // causal_column_mask(0, 0, key)) and this keeps that fact
                // load-bearing instead of decorative.
                table = causal_masks();
                // Index d is the full-visibility mask. At query block zero no
                // key block lies below the diagonal so it is never read; it is
                // present so the table has one shape at every query block.
                table.push_back(std::vector<double>(
                    static_cast<std::size_t>(slot_count_), 0.0));
            }
            else
            {
                for (int j = 0; j < layout_.d; ++j)
                {
                    table.push_back(
                        causal_column_mask(query_block, query_block, j));
                }
                // Any key block strictly below the diagonal: all d keys
                // visible to every query, so the mask is the row weight with
                // no zeros in it. It is the same for every such block.
                table.push_back(
                    causal_column_mask(query_block, query_block - 1, 0));
            }
            return causal_block_mask_cache_
                .emplace(query_block, std::move(table))
                .first->second;
        }

        void Llama3BatchOperator::rope_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& slots, int head_dim,
            double theta, int position_offset)
        {
            const int channels = static_cast<int>(slots.size());
            if (channels == 0 || channels % head_dim != 0)
            {
                throw std::invalid_argument(
                    "RoPE takes a whole number of heads: the channel count "
                    "must be a multiple of head_dim");
            }
            if (head_dim % 2 != 0)
            {
                throw std::invalid_argument(
                    "RoPE pairs lane c with lane c + head_dim/2, so head_dim "
                    "must be even");
            }
            if (!(theta > 0.0))
            {
                throw std::invalid_argument("The RoPE base must be positive");
            }
            for (std::size_t j = 1; j < slots.size(); ++j)
            {
                // The two halves of a pair are added together after their
                // plaintext products, so a level or scale drift is a silently
                // wrong rotation rather than an error.
                if (slots[j].depth() != slots[0].depth() ||
                    slots[j].scale() != slots[0].scale())
                {
                    throw std::invalid_argument(
                        "rope_slots needs every channel at one level and one "
                        "scale");
                }
            }

            Range _r("rope_slots");

            const int heads = channels / head_dim;
            const int half = head_dim / 2;

            // One cosine and one sine vector per LANE PAIR, shared by every
            // head and every instance: the angle depends on the token and the
            // lane, and the token is the slow slot axis. At head_dim = 128
            // that is 64 pairs, not 4096 channels.
            for (int c = 0; c < half; ++c)
            {
                const double omega = std::pow(
                    theta,
                    -2.0 * static_cast<double>(c) /
                        static_cast<double>(head_dim));

                std::vector<double> cos_v(
                    static_cast<std::size_t>(slot_count_), 0.0);
                std::vector<double> sin_v(cos_v.size(), 0.0);
                // The minus sign of the rotation goes on the PLAINTEXT. A
                // homomorphic negation would be multiply_constant, which is a
                // plaintext product and a rescale -- a second level, and then
                // the two halves of the pair would no longer be at the same
                // depth to be added at all.
                std::vector<double> neg_sin_v(cos_v.size(), 0.0);
                for (int u = 0; u < layout_.d; ++u)
                {
                    const double angle =
                        (static_cast<double>(u + position_offset)) *
                        omega;
                    const double cs = std::cos(angle);
                    const double sn = std::sin(angle);
                    for (int b = 0; b < layout_.batch; ++b)
                    {
                        const std::size_t s =
                            static_cast<std::size_t>((b + layout_.batch * u));
                        cos_v[s] = cs;
                        sin_v[s] = sn;
                        neg_sin_v[s] = -sn;
                    }
                }

                for (int h = 0; h < heads; ++h)
                {
                    const std::size_t j0 =
                        static_cast<std::size_t>(h * head_dim + c);
                    const std::size_t j1 = j0 + static_cast<std::size_t>(half);

                    // out0 = x0 cos - x1 sin,  out1 = x0 sin + x1 cos.
                    // Four plaintext products, two additions, ONE level: each
                    // multiply_vector encodes at the prime its own rescale
                    // removes, so all four terms come back at one scale and
                    // one depth and the additions are exact.
                    Ciphertext<Scheme::CKKS> x0_cos = slots[j0];
                    Ciphertext<Scheme::CKKS> x0_sin = slots[j0];
                    Ciphertext<Scheme::CKKS> x1_cos = slots[j1];
                    Ciphertext<Scheme::CKKS> x1_neg_sin = slots[j1];

                    arith_.multiply_vector(x0_cos, cos_v);
                    arith_.multiply_vector(x0_sin, sin_v);
                    arith_.multiply_vector(x1_cos, cos_v);
                    arith_.multiply_vector(x1_neg_sin, neg_sin_v);

                    arith_.add_inplace(x0_cos, x1_neg_sin);
                    arith_.add_inplace(x0_sin, x1_cos);

                    slots[j0] = std::move(x0_cos);
                    slots[j1] = std::move(x0_sin);
                }
            }
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

            // K is NOT transposed here, and that is the point.
            //
            // The scores are S = Q K^T. This used to build K^T with a CMT per
            // channel block and hand it to Algorithm 4, whose step 1 opens by
            // transposing its right operand -- so K was transposed twice and
            // the pair composed to the identity. A CMT is a genuine matrix
            // transpose and therefore an involution on the encoding, so the
            // two cancel exactly.
            //
            // What the algorithm actually needs from step 1 is the ROW-wise
            // encryption of its right operand, and the row-wise encryption of
            // K^T is the column-wise encryption of K -- which is what the
            // projection already handed back. So the projection's own output
            // is passed straight in, marked row_wise, and both CMTs are gone:
            // d - 1 rotations per score product plus one whole standalone CMT
            // per kv channel block, a deep copy of d ciphertexts per product,
            // and the entire K^T array. It is also STRICTLY more accurate --
            // the step-1 CMT's key-switching noise is multiplied by the left
            // operand in the GEMM and is the dominant term in the product's
            // error.
            const int group = heads / kv_heads;

            // Rotary embedding, on Q and K and not on V.
            //
            // Three levels, and only one of them is the rotation: the angle
            // varies with the token, which is the slow SLOT axis, and Q and K
            // are in the coefficient encoding here, so reaching slot form and
            // returning costs a crossing each way. Both operands take exactly
            // the same treatment, which is what keeps them at one level for
            // the score product.
            //
            // Under a slot-resident stream Q and K are already in slot form
            // when they are formed and this is one level and no crossing at
            // all; see LLAMA3_8B_LAYER_FLOW.md 22.6.
            if (config.rope)
            {
                Range _r_rope("attention.rope");
                const int head_dim = per_head * d;
                std::vector<Ciphertext<Scheme::CKKS>> qs =
                    to_slots(q, galois_key);
                rope_slots(qs, head_dim, config.rope_theta,
                           config.rope_position_offset);
                q = from_slots(qs, x.rows, galois_key);
                qs.clear();

                std::vector<Ciphertext<Scheme::CKKS>> ks =
                    to_slots(k, galois_key);
                rope_slots(ks, head_dim, config.rope_theta,
                           config.rope_position_offset);
                k = from_slots(ks, x.rows, galois_key);
            }

            // V does still owe the step-1 transpose: P V wants V row-wise and
            // the projection leaves it column-wise, and no reordering of the
            // product can conjure the transpose the way it cancels for K.
            // But under grouped-query attention `group` consecutive heads
            // share one V block, and Algorithm 4 transposed it once per head.
            // It is transposed here instead -- once per distinct block, in
            // place, at the level P will meet it at -- and every head in the
            // group then reads the same row-wise copy.
            std::vector<bool> value_is_row_wise(
                static_cast<size_t>(std::max(v.columns(), 0)), false);

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
                            rhs.push_back(&k.column[kv_base + t * d + j]);
                        }
                        // row_wise: K's column-wise encryption IS the row-wise
                        // encryption of K^T, so this computes Q K^T with no
                        // transpose at all.
                        std::vector<Ciphertext<Scheme::CKKS>> term = product(
                            lhs, rhs, "attention.score", galois_key, relin_key,
                            HEBatchMatrixOperator<
                                Scheme::CKKS>::RightOperandForm::row_wise);
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
                        const int base = kv_base + t * d;

                        // The value block is used in place rather than sliced
                        // out: ccmm reads its right operand and never writes
                        // it, so the copy this used to make was `group` deep
                        // copies of d ciphertexts for nothing. The mod drop is
                        // idempotent, so heads after the first in a group fall
                        // straight through it.
                        for (int j = 0; j < d; ++j)
                        {
                            arith_.drop_to_depth(v.column[base + j], depth);
                        }

                        // First head of the group pays the transpose; the rest
                        // inherit it. Dropping before transposing rather than
                        // after is deliberate -- the CMT is d - 1 rotations
                        // and a rotation costs what the live limb count says.
                        if (!value_is_row_wise[static_cast<size_t>(base)])
                        {
                            Range _r_t("attention.transpose_value");
                            std::vector<Ciphertext<Scheme::CKKS>> block;
                            block.reserve(static_cast<size_t>(d));
                            for (int j = 0; j < d; ++j)
                            {
                                block.push_back(std::move(v.column[base + j]));
                            }
                            matrix_.cmt(block, galois_key, arith_);
                            for (int j = 0; j < d; ++j)
                            {
                                v.column[base + j] = std::move(block[j]);
                            }
                            value_is_row_wise[static_cast<size_t>(base)] = true;
                        }

                        std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
                        for (int j = 0; j < d; ++j)
                        {
                            lhs.push_back(&p.column[j]);
                            rhs.push_back(&v.column[base + j]);
                        }
                        std::vector<Ciphertext<Scheme::CKKS>> head_out =
                            product(lhs, rhs, "attention.value", galois_key,
                                    relin_key,
                                    HEBatchMatrixOperator<Scheme::CKKS>::
                                        RightOperandForm::row_wise);
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
        // Token blocking
        // -------------------------------------------------------------------

        std::vector<BatchActivation> Llama3BatchOperator::softmax_seam_blocked(
            std::vector<BatchActivation>& scores, int query_block,
            const Llama3Operator::SoftmaxConfig& softmax,
            const BatchSoftmaxSeamConfig& seam,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key, Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int d = layout_.d;
            const int step = layout_.k / 2;
            const int blocks = static_cast<int>(scores.size());

            if (query_block < 0)
            {
                throw std::invalid_argument(
                    "A query block index is non-negative");
            }
            if (blocks != query_block + 1)
            {
                // Exactly the visible past, no more and no less. Fewer would
                // be a SoftMax over a truncated row -- a different model, and
                // one that still returns numbers.
                throw std::invalid_argument(
                    "A causal query block sees exactly query_block + 1 key "
                    "blocks, so that is how many score blocks the seam takes");
            }
            for (int q = 0; q < blocks; ++q)
            {
                if (scores[q].columns() != d || scores[q].rows != d)
                {
                    throw std::invalid_argument(
                        "Every score block is the square block Algorithm 4 "
                        "hands back: layout.d rows and layout.d columns");
                }
                require_uniform(scores[q], "softmax_seam_blocked");
                // The parts of EVERY block are added together to form one
                // denominator, so a drift across blocks is as wrong as a drift
                // within one, and only this check sees it.
                if (scores[q].column.front().depth() !=
                        scores[0].column.front().depth() ||
                    scores[q].column.front().scale() !=
                        scores[0].column.front().scale())
                {
                    throw std::invalid_argument(
                        "Every score block of one query block must be at one "
                        "level and one scale: they share a denominator");
                }
            }
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
                throw std::invalid_argument(
                    "Refreshing the SoftMax denominator needs the boot Galois "
                    "key");
            }

            Range _r_seam("softmax_seam_blocked");

            struct HoistGuard
            {
                Llama3BatchOperator* op;
                bool previous;
                ~HoistGuard() { op->set_hoisted_crossings(previous); }
            } guard{this, hoisted_crossings_};
            set_hoisted_crossings(seam.hoisted_crossings);

            // A query block uses d + 1 distinct masks and every head of that
            // block reuses the same d + 1, so this is the capacity that gives
            // a 100% hit rate where the reuse actually is. Capacity for a
            // whole sequence would be blocks times this and buy nothing: a
            // query block is visited once.
            if (seam.causal && seam.cache_masks &&
                arith_.mask_plain_capacity() <
                    static_cast<std::size_t>(d + 1))
            {
                arith_.set_mask_plain_capacity(
                    static_cast<std::size_t>(d + 1));
            }

            // Cross every score block and concatenate. The key axis is the
            // ciphertext axis, so "concatenate" is literally that -- the
            // reduction below does not care which block a part came from, and
            // that is the whole reason token blocking is cheap here.
            std::vector<Ciphertext<Scheme::CKKS>> slots;
            slots.reserve(static_cast<std::size_t>(blocks) *
                          static_cast<std::size_t>(d));
            {
                Range _r("softmax_seam_blocked.to_slots");
                for (int q = 0; q < blocks; ++q)
                {
                    std::vector<Ciphertext<Scheme::CKKS>> part =
                        to_slots(scores[q], galois_key);
                    for (auto& c : part)
                    {
                        slots.push_back(std::move(c));
                    }
                }
            }

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
            // Unchanged from the single-block seam, and that is the point: a
            // longer key axis is more PARTS, not a different layout. count
            // stays one, sum_strided's loop stays dead, and the denominator
            // over blocks*d keys costs exactly the zero rotations it cost over
            // d of them.
            config.strided = true;
            config.stride = slot_count_;
            config.count = 1;
            config.pre_scaled_input = seam.scores_carry_exp_domain;
            config.fold_affine_into_mask =
                seam.fold_affine_into_mask && seam.causal;
            config.refresh_denominator = seam.refresh_denominator;

            std::vector<std::vector<double>> masks;
            std::vector<int> mask_ids;
            if (seam.causal)
            {
                // COST, stated rather than discovered: this materialises
                // blocks*d slot vectors on the HOST, blocks*d*slot_count*8
                // bytes -- 134 MB at 64 blocks and the 8B ring. It is built
                // once per query block and shared by every head. The device
                // side, blocks*d resident ciphertexts, is the binding one.
                const std::vector<std::vector<double>>& table =
                    causal_masks_blocked(query_block);
                masks.reserve(static_cast<std::size_t>(blocks) *
                              static_cast<std::size_t>(d));
                mask_ids.reserve(masks.capacity());
                for (int q = 0; q < blocks; ++q)
                {
                    const bool diagonal = q == query_block;
                    for (int j = 0; j < d; ++j)
                    {
                        masks.push_back(
                            table[static_cast<std::size_t>(diagonal ? j : d)]);
                        mask_ids.push_back(
                            seam.cache_masks ? causal_mask_id(query_block, q, j)
                                             : -1);
                    }
                }
            }

            std::vector<Ciphertext<Scheme::CKKS>> probabilities;
            {
                Range _r("softmax_seam_blocked.softmax");
                probabilities =
                    arith_.softmax(slots, config, masks, mask_ids, galois_key,
                                   relin_key, boot_key);
            }
            slots.clear();
            masks.clear();
            masks.shrink_to_fit();

            Range _r("softmax_seam_blocked.from_slots");
            std::vector<BatchActivation> out;
            out.reserve(static_cast<std::size_t>(blocks));
            for (int q = 0; q < blocks; ++q)
            {
                std::vector<Ciphertext<Scheme::CKKS>> part;
                part.reserve(static_cast<std::size_t>(d));
                for (int j = 0; j < d; ++j)
                {
                    part.push_back(std::move(
                        probabilities[static_cast<std::size_t>(q * d + j)]));
                }
                out.push_back(from_slots(part, d, galois_key));
            }
            return out;
        }

        Llama3BatchOperator::SequenceProductCount
        Llama3BatchOperator::sequence_product_count(
            int blocks, const BatchAttentionConfig& config) const
        {
            if (blocks < 1)
            {
                throw std::invalid_argument(
                    "A sequence has at least one token block");
            }
            const int d = layout_.d;
            const int heads = config.heads < 1 ? 1 : config.heads;
            const int kv_heads =
                config.kv_heads > 0 ? config.kv_heads : heads;
            const int per_head =
                heads * d > 0 ? config.q_channels / (heads * d) : 0;

            SequenceProductCount out;
            // Causality is what makes this triangular. A bidirectional model
            // would pay blocks^2 here.
            const long long pairs =
                static_cast<long long>(blocks) * (blocks + 1) / 2;
            out.score = pairs * heads * per_head;
            out.value = out.score;
            // Every score block crosses down and every probability block
            // crosses back, d columns apiece.
            out.bridged_columns = 2LL * pairs * heads * d;
            if (config.rope)
            {
                // Q and K, down and back, once per token block.
                out.bridged_columns +=
                    2LL * blocks *
                    (static_cast<long long>(config.q_channels) +
                     static_cast<long long>(config.kv_channels));
            }
            (void) kv_heads;
            // The deepest query block holds its whole visible past in slot
            // form at once, and the SoftMax squares it.
            out.peak_slot_columns = static_cast<long long>(blocks) * d;
            return out;
        }

        std::vector<BatchActivation> Llama3BatchOperator::attention_sequence(
            std::vector<BatchActivation>& x,
            const BatchAttentionWeights& weights,
            const BatchAttentionConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key, Galoiskey<Scheme::CKKS>* boot_key)
        {
            const int d = layout_.d;
            const int heads = config.heads;
            const int kv_heads =
                config.kv_heads > 0 ? config.kv_heads : config.heads;
            const int blocks = static_cast<int>(x.size());

            if (blocks < 1)
            {
                throw std::invalid_argument(
                    "A sequence has at least one token block");
            }
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
            if (!config.causal && blocks > 1)
            {
                // Non-causal blocking would need the key blocks ABOVE the
                // diagonal too, which is a different schedule and a different
                // mask table. Refusing is the honest answer; silently running
                // the causal schedule would return a plausible wrong model.
                throw std::invalid_argument(
                    "Token blocking here is causal: a bidirectional sequence "
                    "needs the full blocks^2 schedule, which this does not "
                    "implement");
            }
            for (int t = 0; t < blocks; ++t)
            {
                if (x[t].rows != d)
                {
                    throw std::invalid_argument(
                        "Every token block has exactly layout.d rows");
                }
                if (x[t].columns() != config.in_channels)
                {
                    throw std::invalid_argument(
                        "Every token block carries config.in_channels "
                        "columns");
                }
                require_uniform(x[t], "attention_sequence");
                if (x[t].column.front().depth() !=
                        x[0].column.front().depth() ||
                    x[t].column.front().scale() != x[0].column.front().scale())
                {
                    throw std::invalid_argument(
                        "Every token block of a sequence must enter at one "
                        "level and one scale");
                }
            }

            Range _r_attention("attention_sequence");

            const double head_scale =
                config.head_scale != 0.0
                    ? config.head_scale
                    : 1.0 / std::sqrt(static_cast<double>(per_head * d));
            const double exp_domain =
                config.seam.scores_carry_exp_domain
                    ? exp_domain_scale(config.softmax.bound)
                    : 1.0;

            std::vector<double> query_weight = weights.query;
            for (auto& w : query_weight)
            {
                w *= head_scale * exp_domain;
            }

            const int group = heads / kv_heads;

            // K and V for the WHOLE sequence have to be resident: query block
            // p reads every one of them at or below itself, and there is no
            // KV cache on this path to hold them anywhere else. Q is formed
            // per query block instead, because Q_p is read by query block p
            // and by nothing else -- that is a third of the projection working
            // set for free.
            std::vector<BatchActivation> k(static_cast<std::size_t>(blocks));
            std::vector<BatchActivation> v(static_cast<std::size_t>(blocks));
            {
                Range _r_kv("attention_sequence.kv");
                for (int t = 0; t < blocks; ++t)
                {
                    k[t] = project(x[t], weights.key, config.in_channels,
                                   config.kv_channels, "attention_seq.k");
                    v[t] = project(x[t], weights.value, config.in_channels,
                                   config.kv_channels, "attention_seq.v");
                    if (config.rope)
                    {
                        // The sequence is ONE sequence: block t starts at
                        // t*d. Restarting the angle per block would give
                        // blocks independent 128-token sequences that happen
                        // to attend to each other.
                        Range _r_rope("attention_seq.rope_k");
                        std::vector<Ciphertext<Scheme::CKKS>> ks =
                            to_slots(k[t], galois_key);
                        rope_slots(ks, per_head * d, config.rope_theta,
                                   config.rope_position_offset + t * d);
                        k[t] = from_slots(ks, d, galois_key);
                    }
                }
            }

            // Whether each (token block, kv channel block) has already paid
            // Algorithm 4's step-1 transpose. V is transposed once per
            // distinct block and every head of every query block above it then
            // reads the same row-wise copy -- so the m^2 schedule does NOT
            // multiply the transposes.
            std::vector<std::vector<bool>> value_is_row_wise(
                static_cast<std::size_t>(blocks),
                std::vector<bool>(
                    static_cast<std::size_t>(std::max(config.kv_channels, 0)),
                    false));

            BatchSoftmaxSeamConfig seam = config.seam;
            seam.causal = config.causal;
            if (seam.score_shift == 0.0 && seam.score_shift_rows.empty() &&
                seam.score_shift_slots.empty())
            {
                seam.score_shift = config.score_shift;
            }

            std::vector<BatchActivation> out;
            out.reserve(static_cast<std::size_t>(blocks));

            for (int p = 0; p < blocks; ++p)
            {
                Range _r_block("attention_sequence.query_block");

                BatchActivation q =
                    project(x[p], query_weight, config.in_channels,
                            config.q_channels, "attention_seq.q");
                if (config.rope)
                {
                    Range _r_rope("attention_seq.rope_q");
                    std::vector<Ciphertext<Scheme::CKKS>> qs =
                        to_slots(q, galois_key);
                    rope_slots(qs, per_head * d, config.rope_theta,
                               config.rope_position_offset + p * d);
                    q = from_slots(qs, d, galois_key);
                }

                BatchActivation block_out;
                block_out.rows = d;
                block_out.column.reserve(
                    static_cast<std::size_t>(config.q_channels));

                for (int h = 0; h < heads; ++h)
                {
                    const int q_base = h * per_head * d;
                    const int kv_base = (h / group) * per_head * d;

                    // One square score block per visible key block.
                    std::vector<BatchActivation> scores;
                    scores.reserve(static_cast<std::size_t>(p + 1));
                    {
                        Range _r("attention_seq.scores");
                        for (int qb = 0; qb <= p; ++qb)
                        {
                            std::vector<Ciphertext<Scheme::CKKS>> block;
                            for (int t = 0; t < per_head; ++t)
                            {
                                std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
                                for (int j = 0; j < d; ++j)
                                {
                                    lhs.push_back(
                                        &q.column[q_base + t * d + j]);
                                    rhs.push_back(
                                        &k[qb].column[kv_base + t * d + j]);
                                }
                                // K's column-wise encryption IS the row-wise
                                // encryption of K^T, so this is Q K_qb^T with
                                // no transpose at all -- exactly as in the
                                // single-block sublayer.
                                std::vector<Ciphertext<Scheme::CKKS>> term =
                                    product(lhs, rhs, "attention_seq.score",
                                            galois_key, relin_key,
                                            HEBatchMatrixOperator<Scheme::CKKS>::
                                                RightOperandForm::row_wise);
                                if (block.empty())
                                {
                                    block = std::move(term);
                                }
                                else
                                {
                                    for (int j = 0; j < d; ++j)
                                    {
                                        arith_.add_inplace(block[j], term[j]);
                                    }
                                }
                            }
                            BatchActivation s;
                            s.rows = d;
                            s.column = std::move(block);
                            scores.push_back(std::move(s));
                        }
                    }

                    // ONE SoftMax over the whole visible row, not p + 1 of
                    // them. The denominator sums across every part of every
                    // block, which is what makes this the SoftMax of a
                    // (p+1)*d-long row rather than of p + 1 short ones.
                    std::vector<BatchActivation> probability =
                        softmax_seam_blocked(scores, p, config.softmax, seam,
                                             galois_key, relin_key, boot_key);
                    scores.clear();

                    {
                        Range _r("attention_seq.value_product");
                        const int depth =
                            probability.front().column.front().depth();
                        for (int t = 0; t < per_head; ++t)
                        {
                            std::vector<Ciphertext<Scheme::CKKS>> acc;
                            for (int qb = 0; qb <= p; ++qb)
                            {
                                const int base = kv_base + t * d;

                                for (int j = 0; j < d; ++j)
                                {
                                    arith_.drop_to_depth(
                                        v[qb].column[base + j], depth);
                                }
                                if (!value_is_row_wise[static_cast<std::size_t>(
                                        qb)][static_cast<std::size_t>(base)])
                                {
                                    Range _r_t("attention_seq.transpose_value");
                                    std::vector<Ciphertext<Scheme::CKKS>> vb;
                                    vb.reserve(static_cast<std::size_t>(d));
                                    for (int j = 0; j < d; ++j)
                                    {
                                        vb.push_back(
                                            std::move(v[qb].column[base + j]));
                                    }
                                    matrix_.cmt(vb, galois_key, arith_);
                                    for (int j = 0; j < d; ++j)
                                    {
                                        v[qb].column[base + j] =
                                            std::move(vb[j]);
                                    }
                                    value_is_row_wise[static_cast<std::size_t>(
                                        qb)][static_cast<std::size_t>(base)] =
                                        true;
                                }

                                std::vector<Ciphertext<Scheme::CKKS>*> lhs, rhs;
                                for (int j = 0; j < d; ++j)
                                {
                                    lhs.push_back(&probability[qb].column[j]);
                                    rhs.push_back(&v[qb].column[base + j]);
                                }
                                std::vector<Ciphertext<Scheme::CKKS>> term =
                                    product(lhs, rhs, "attention_seq.value",
                                            galois_key, relin_key,
                                            HEBatchMatrixOperator<Scheme::CKKS>::
                                                RightOperandForm::row_wise);
                                if (acc.empty())
                                {
                                    acc = std::move(term);
                                }
                                else
                                {
                                    // Attention over a blocked key axis is a
                                    // sum over the blocks, and every term is
                                    // at one level and one scale, so the
                                    // accumulation costs no depth.
                                    for (int j = 0; j < d; ++j)
                                    {
                                        arith_.add_inplace(acc[j], term[j]);
                                    }
                                }
                            }
                            for (auto& c : acc)
                            {
                                block_out.column.push_back(std::move(c));
                            }
                        }
                    }
                }

                if (weights.output.empty())
                {
                    out.push_back(std::move(block_out));
                }
                else
                {
                    out.push_back(project(block_out, weights.output,
                                          config.q_channels,
                                          config.in_channels,
                                          "attention_seq.o"));
                }
            }

            return out;
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
