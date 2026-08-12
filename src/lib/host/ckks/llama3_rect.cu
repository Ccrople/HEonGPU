// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/host/ckks/llama3_rect.cuh>

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
            /// Scoped NVTX range, matching the taxonomy llama3.cu and
            /// llama3_batch.cu publish so a capture can put the three matrix
            /// paths side by side.
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
                    char name[80];
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
            /// The matrix inverted here is F[b][t] = zeta^{5^b t}, a Vandermonde
            /// in k/2 distinct 2k-th roots of unity, so it is a scaled DFT and
            /// about as well conditioned as a dense inverse ever is. This runs
            /// once per operator.
            std::vector<cd> invert(std::vector<cd> a, int n)
            {
                std::vector<cd> inv(static_cast<size_t>(n) * n, cd(0.0, 0.0));
                for (int i = 0; i < n; ++i)
                {
                    inv[static_cast<size_t>(i) * n + i] = cd(1.0, 0.0);
                }

                for (int col = 0; col < n; ++col)
                {
                    int pivot = col;
                    double best =
                        std::abs(a[static_cast<size_t>(col) * n + col]);
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
                            "the block transform is singular, which means the "
                            "layout's k/2 evaluation points are not distinct");
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

                    const cd piv = a[static_cast<size_t>(col) * n + col];
                    for (int c = 0; c < n; ++c)
                    {
                        a[static_cast<size_t>(col) * n + c] /= piv;
                        inv[static_cast<size_t>(col) * n + c] /= piv;
                    }

                    for (int r = 0; r < n; ++r)
                    {
                        if (r == col)
                        {
                            continue;
                        }
                        const cd f = a[static_cast<size_t>(r) * n + col];
                        if (f == cd(0.0, 0.0))
                        {
                            continue;
                        }
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

            inline int64_t quantise(double v, double scale)
            {
                return static_cast<int64_t>(std::llround(v * scale));
            }
        } // namespace

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        Llama3RectOperator::Llama3RectOperator(
            HEContext<Scheme::CKKS> context, HEEncoder<Scheme::CKKS>& encoder,
            const BatchMatrixLayout& layout, double scale)
            : context_(context), encoder_(encoder),
              batch_(context, encoder, layout, scale), layout_(layout),
              primes_(context->get_key_modulus()),
              slot_count_(encoder.slot_count()), default_scale_(scale)
        {
            if (layout_.N != slot_count_ * 2)
            {
                throw std::invalid_argument(
                    "The batch matrix layout must be built for this context's "
                    "ring: layout.N has to be twice the slot count");
            }
            build_block_tables();
        }

        int Llama3RectOperator::groups_for(int channels) const
        {
            if (channels <= 0)
            {
                throw std::invalid_argument(
                    "An activation with no channels is not an activation");
            }
            const int half = channels_per_group();
            return (channels + half - 1) / half;
        }

        // -------------------------------------------------------------------
        // The block transform
        // -------------------------------------------------------------------

        void Llama3RectOperator::build_block_tables()
        {
            const int d = layout_.d;
            const int N = layout_.N;
            const int step = layout_.batch; // k/2
            const double pi = std::acos(-1.0);
            const uint64_t mod = 2ull * static_cast<uint64_t>(N);

            forward_block_.clear();
            inverse_block_.clear();
            if (step == 1)
            {
                // One block per group: the transform is the identity on a
                // single value and there is nothing to build.
                return;
            }

            auto psi_pow = [&](uint64_t e)
            {
                return std::polar(1.0, pi * static_cast<double>(e % mod) /
                                           static_cast<double>(N));
            };

            // exponent[b] = 5^b mod 2N, the Galois element rotation index b
            // names. Slot s evaluates at psi^{5^s}, and 5 has order k/2 modulo
            // 2k, so the SUBRING point psi^{5^s d} depends only on s mod (k/2)
            // -- which is exactly why the block index rides the fast slot axis.
            std::vector<uint64_t> exponent(step);
            {
                uint64_t g = 1;
                for (int b = 0; b < step; ++b)
                {
                    exponent[b] = g;
                    g = (g * 5ull) % mod;
                }
            }

            // F[b][t] = zeta^{5^b t} with zeta = psi^d, the primitive 2k-th
            // root the subring transform induces.
            std::vector<cd> F(static_cast<size_t>(step) * step);
            for (int b = 0; b < step; ++b)
            {
                for (int t = 0; t < step; ++t)
                {
                    const uint64_t e = (exponent[b] * static_cast<uint64_t>(d) *
                                        static_cast<uint64_t>(t)) %
                                       mod;
                    F[static_cast<size_t>(b) * step + t] = psi_pow(e);
                }
            }
            const std::vector<cd> Finv = invert(F, step);

            // Blocked diagonals. The block index is the FAST slot axis with
            // span k/2, so a rotation by eps takes b to b + eps and spills into
            // the neighbouring token when it leaves the block; the diagonal is
            // zero exactly where it spills, which is what confines the map to
            // one token.
            const int count = 2 * step - 1;
            forward_block_.assign(
                count, std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));
            inverse_block_.assign(
                count, std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));

            for (int e = 0; e < count; ++e)
            {
                const int eps = e - (step - 1);
                for (int b = 0; b < step; ++b)
                {
                    const int bp = b + eps;
                    if (bp < 0 || bp >= step)
                    {
                        continue;
                    }
                    const cd& f = F[static_cast<size_t>(b) * step + bp];
                    const cd& g = Finv[static_cast<size_t>(b) * step + bp];
                    for (int u = 0; u < d; ++u)
                    {
                        const int slot = b + u * step;
                        forward_block_[e][slot] = Complex64(f.real(), f.imag());
                        inverse_block_[e][slot] = Complex64(g.real(), g.imag());
                    }
                }
            }
        }

        std::vector<int> Llama3RectOperator::block_rotation_indices() const
        {
            const int step = layout_.batch;
            std::vector<int> indices;
            if (step == 1)
            {
                return indices;
            }
            indices.reserve(2 * step - 2);
            for (int eps = -(step - 1); eps <= step - 1; ++eps)
            {
                if (eps == 0)
                {
                    continue;
                }
                indices.push_back(eps > 0 ? eps : slot_count_ + eps);
            }
            return indices;
        }

        std::vector<int> Llama3RectOperator::rotation_indices() const
        {
            std::vector<int> all = get_rectangular_rotation_indices(layout_);

            const std::vector<int> bridge = batch_.rotation_indices();
            all.insert(all.end(), bridge.begin(), bridge.end());

            const std::vector<int> block = block_rotation_indices();
            all.insert(all.end(), block.begin(), block.end());

            // RMSNorm reduces the channels held inside one ciphertext, and they
            // are the fast axis here, so it is the masked reduction rather than
            // the free one.
            const std::vector<int> reduce =
                Llama3Operator::blocked_rotation_indices(layout_.batch);
            for (int r : reduce)
            {
                all.push_back(r >= 0 ? r : slot_count_ + r);
            }

            // The fused crossings' BSGS split: baby shifts 1 .. n1 - 1 and
            // giant shifts i * n1. A subset of Algorithm 5's full group above,
            // listed so the union stays honest if the rectangular product is
            // ever taken out.
            {
                const int n1 = fused_baby_steps();
                for (int j = 1; j < n1; ++j)
                {
                    all.push_back(j);
                }
                for (int i = 1; i < slot_count_ / n1; ++i)
                {
                    all.push_back(i * n1);
                }
            }

            std::sort(all.begin(), all.end());
            all.erase(std::unique(all.begin(), all.end()), all.end());
            return all;
        }

        void Llama3RectOperator::block_map(
            std::vector<Ciphertext<Scheme::CKKS>>& ct, bool inverse,
            const char* name, Galoiskey<Scheme::CKKS>& galois_key)
        {
            const int step = layout_.batch;
            if (ct.empty() || step == 1)
            {
                return;
            }

            const std::vector<std::vector<Complex64>>& diagonal =
                inverse ? inverse_block_ : forward_block_;

            Range _r_block(name);

            const int depth = ct.front().depth();
            const double plain_scale = rescale_prime(ct.front());
            for (const auto& c : ct)
            {
                if (c.depth() != depth)
                {
                    throw std::invalid_argument(
                        "The block transform adds its diagonals together, so "
                        "every ciphertext in one call has to share a level");
                }
            }

            // Every ciphertext in the call meets the same diagonals at the same
            // level, so they are encoded once rather than once per column.
            //
            // That used to be the difference between this map and the row
            // bridge, and it no longer is: the row bridge caches its encodings
            // too, and encoding is 0.02% of a real 8B block. What separates
            // them now is the OPPOSITE of what this comment used to claim.
            // The loop below is 2*(k/2) - 1 = 31 diagonals walked one rotation
            // at a time, and the row bridge's d = 128 are walked baby-step /
            // giant-step. So the smaller map costs MORE: measured over one 8B
            // block, this one is 16.5% of GPU time against the row bridge's
            // 15.4%, 142080 rotations against 137984. BSGS applies here too --
            // it would take 30 rotations per ciphertext to about 10 -- but the
            // giant shifts run past the +-(k/2 - 1) window this map's Galois
            // indices cover, so unlike the row bridge it is not free of new
            // keys. That is the trade, and it has not been made yet.
            std::vector<Plaintext<Scheme::CKKS>> plain;
            std::vector<int> shift;
            {
                SuffixRange _r(name, "encode_diagonals");
                plain.reserve(diagonal.size());
                shift.reserve(diagonal.size());
                for (size_t e = 0; e < diagonal.size(); ++e)
                {
                    const int eps = static_cast<int>(e) - (step - 1);
                    plain.push_back(encode(diagonal[e], plain_scale, depth));
                    shift.push_back(eps);
                }
            }

            if (hoisted_crossings_)
            {
                // Every one of the 2*(k/2) - 1 shifts reads ONE source and
                // there are no giant steps, so the whole walk above is a
                // single hoisted rotation train and ONE fused
                // multiply-accumulate: the map that measured dearer than the
                // row bridge (see the note above) becomes two launches per
                // ciphertext plus its rescale, bit-identically.
                DeviceVector<Data64> packed =
                    batch_.arith().pack_bsgs_plaintexts(plain, depth);
                std::vector<int> shifts;
                shifts.reserve(shift.size());
                for (const int eps : shift)
                {
                    shifts.push_back(eps >= 0 ? eps : slot_count_ + eps);
                }
                const int count = static_cast<int>(plain.size());

                for (auto& source : ct)
                {
                    SuffixRange _r(name, "diagonals");
                    DeviceVector<Data64> babies =
                        batch_.arith().hoisted_rotation_train(
                            source, shifts, count, galois_key);
                    Ciphertext<Scheme::CKKS> acc =
                        batch_.arith().hoisted_bsgs_group_sum(
                            babies, packed, 0, count, source, plain_scale);
                    batch_.arith().rescale_inplace(acc);
                    source = std::move(acc);
                }
                return;
            }

            for (auto& source : ct)
            {
                Ciphertext<Scheme::CKKS> acc(context_);
                bool started = false;

                SuffixRange _r(name, "diagonals");
                for (size_t e = 0; e < plain.size(); ++e)
                {
                    const int eps = shift[e];

                    Ciphertext<Scheme::CKKS> rotated(context_);
                    if (eps == 0)
                    {
                        rotated = source;
                    }
                    else
                    {
                        batch_.arith().rotate_rows(
                            source, rotated, galois_key,
                            eps > 0 ? eps : slot_count_ + eps);
                    }

                    Ciphertext<Scheme::CKKS> term(context_);
                    batch_.arith().multiply_plain(rotated, plain[e], term);

                    if (!started)
                    {
                        acc = std::move(term);
                        started = true;
                    }
                    else
                    {
                        batch_.arith().add_inplace(acc, term);
                    }
                }
                batch_.arith().rescale_inplace(acc);
                source = std::move(acc);
            }
        }

        // -------------------------------------------------------------------
        // Moving between the three encodings
        // -------------------------------------------------------------------

        BatchActivation Llama3RectOperator::borrow_group(RectActivation& x,
                                                         int group)
        {
            const int d = layout_.d;
            BatchActivation borrowed;
            borrowed.rows = d;
            borrowed.column.reserve(d);
            for (int j = 0; j < d; ++j)
            {
                borrowed.column.push_back(
                    std::move(x.column[static_cast<size_t>(group) * d + j]));
            }
            return borrowed;
        }

        void Llama3RectOperator::return_group(RectActivation& x, int group,
                                              BatchActivation& borrowed)
        {
            const int d = layout_.d;
            for (int j = 0; j < d; ++j)
            {
                x.column[static_cast<size_t>(group) * d + j] =
                    std::move(borrowed.column[j]);
            }
            borrowed.column.clear();
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3RectOperator::to_slots(RectActivation& in,
                                     Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_uniform(in, "to_slots");

            Range _r("bridge.rect_to_slots");

            if (fused_crossings_)
            {
                // Both stages in one BSGS pass: one level instead of two. The
                // map does not depend on the group, so the columns go through
                // flat and come out in the order the staged loop would emit.
                std::vector<Ciphertext<Scheme::CKKS>> fused = in.column;
                fused_apply(fused, FusedMap::ToSlots, "bridge.fused_to_slots",
                            galois_key);
                return fused;
            }

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(in.column.size());
            for (int g = 0; g < in.groups; ++g)
            {
                BatchActivation borrowed = borrow_group(in, g);
                std::vector<Ciphertext<Scheme::CKKS>> slots =
                    batch_.to_slots(borrowed, galois_key);
                return_group(in, g, borrowed);

                block_map(slots, /*inverse=*/true, "bridge.block_inverse",
                          galois_key);
                for (auto& c : slots)
                {
                    out.push_back(std::move(c));
                }
            }
            return out;
        }

        RectActivation
        Llama3RectOperator::from_slots(std::vector<Ciphertext<Scheme::CKKS>>& in,
                                       int channels,
                                       Galoiskey<Scheme::CKKS>& galois_key)
        {
            const int d = layout_.d;
            if (in.empty() || in.size() % static_cast<size_t>(d) != 0)
            {
                throw std::invalid_argument(
                    "A rectangular group is exactly layout.d slot ciphertexts, "
                    "because d is the rank of the module and not a free "
                    "parameter");
            }
            const int groups = static_cast<int>(in.size()) / d;
            if (groups != groups_for(channels))
            {
                throw std::invalid_argument(
                    "The slot ciphertexts do not cover the channel count they "
                    "are said to carry");
            }

            Range _r("bridge.rect_from_slots");

            if (fused_crossings_)
            {
                RectActivation fused;
                fused.rows = d;
                fused.groups = groups;
                fused.channels = channels;
                fused.column = std::move(in);
                fused_apply(fused.column, FusedMap::FromSlots,
                            "bridge.fused_from_slots", galois_key);
                return fused;
            }

            RectActivation out;
            out.rows = d;
            out.groups = groups;
            out.channels = channels;
            out.column.reserve(in.size());

            for (int g = 0; g < groups; ++g)
            {
                std::vector<Ciphertext<Scheme::CKKS>> slots;
                slots.reserve(d);
                for (int j = 0; j < d; ++j)
                {
                    slots.push_back(
                        std::move(in[static_cast<size_t>(g) * d + j]));
                }

                block_map(slots, /*inverse=*/false, "bridge.block_forward",
                          galois_key);
                BatchActivation group = batch_.from_slots(slots, d, galois_key);
                for (auto& c : group.column)
                {
                    out.column.push_back(std::move(c));
                }
            }
            return out;
        }

        BatchActivation
        Llama3RectOperator::to_batch(RectActivation& in, int group,
                                     Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (group < 0 || group >= in.groups)
            {
                throw std::invalid_argument(
                    "No such channel group in this activation");
            }

            Range _r("bridge.to_batch");

            if (fused_crossings_)
            {
                // The three stages composed on the host: one level instead of
                // three. The input group is copied, exactly as the staged path
                // leaves the activation intact.
                const int d = layout_.d;
                BatchActivation fused;
                fused.rows = d;
                fused.column.assign(
                    in.column.begin() + static_cast<size_t>(group) * d,
                    in.column.begin() + static_cast<size_t>(group + 1) * d);
                fused_apply(fused.column, FusedMap::ToBatch,
                            "bridge.fused_to_batch", galois_key);
                return fused;
            }

            BatchActivation borrowed = borrow_group(in, group);
            std::vector<Ciphertext<Scheme::CKKS>> slots =
                batch_.to_slots(borrowed, galois_key);
            return_group(in, group, borrowed);

            block_map(slots, /*inverse=*/true, "bridge.block_inverse",
                      galois_key);
            return batch_.from_slots(slots, layout_.d, galois_key);
        }

        void Llama3RectOperator::rope_slots(
            std::vector<Ciphertext<Scheme::CKKS>>& slots,
            const RectAttentionConfig& config)
        {
            const int d = layout_.d;
            const int step = layout_.batch;
            const int pairs = d / 2;

            if (static_cast<int>(slots.size()) != d)
            {
                throw std::invalid_argument(
                    "RoPE reads one group at a time: exactly d slot "
                    "ciphertexts, because in this encoding the head-dim index "
                    "IS the ciphertext index");
            }
            if (pairs < 1 || d % 2 != 0)
            {
                throw std::invalid_argument(
                    "RoPE pairs channel c with c + head_dim / 2, so the head "
                    "dimension has to be even");
            }

            Range _r_rope("rope");

            const int depth = slots.front().depth();
            const double plain_scale = rescale_prime(slots.front());
            for (const auto& c : slots)
            {
                if (c.depth() != depth)
                {
                    throw std::invalid_argument(
                        "RoPE adds two of the group's ciphertexts together, so "
                        "they have to share a level");
                }
            }

            // The tables. cos and sin are indexed by the head-dim PAIR
            // c = j mod (d/2) and by the token, and the token is the SLOW slot
            // axis of this reading, so one vector per pair serves every block,
            // every head and every channel group. d/2 pairs, not d, because
            // both halves of a pair turn by the same angle.
            if (static_cast<int>(rope_cos_.size()) != pairs ||
                rope_theta_built_ != config.rope_theta ||
                rope_offset_built_ != config.rope_position_offset)
            {
                rope_cos_.assign(
                    pairs,
                    std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));
                rope_sin_.assign(
                    pairs,
                    std::vector<Complex64>(slot_count_, Complex64(0.0, 0.0)));
                for (int c = 0; c < pairs; ++c)
                {
                    // theta^(-2c/d), the inverse frequency of the pair.
                    const double omega =
                        std::pow(config.rope_theta,
                                 -2.0 * static_cast<double>(c) /
                                     static_cast<double>(d));
                    for (int u = 0; u < d; ++u)
                    {
                        const double angle =
                            static_cast<double>(u +
                                                config.rope_position_offset) *
                            omega;
                        const Complex64 cs(std::cos(angle), 0.0);
                        const Complex64 sn(std::sin(angle), 0.0);
                        for (int b = 0; b < step; ++b)
                        {
                            const int slot = b + step * u;
                            rope_cos_[c][slot] = cs;
                            rope_sin_[c][slot] = sn;
                        }
                    }
                }
                rope_theta_built_ = config.rope_theta;
                rope_offset_built_ = config.rope_position_offset;
            }

            // Every ciphertext of the group meets the same d/2 pairs at the
            // same level, so the plaintexts are encoded once per call rather
            // than once per column -- the block transform's own discipline.
            std::vector<Plaintext<Scheme::CKKS>> cos_plain, sin_plain;
            {
                SuffixRange _r("rope", "encode");
                cos_plain.reserve(pairs);
                sin_plain.reserve(pairs);
                for (int c = 0; c < pairs; ++c)
                {
                    cos_plain.push_back(
                        encode(rope_cos_[c], plain_scale, depth));
                    sin_plain.push_back(
                        encode(rope_sin_[c], plain_scale, depth));
                }
            }

            // HuggingFace's rotate_half convention, which is the order the
            // checkpoint's channels are written in: the low half turns into
            // x1 cos - x2 sin and the high half into x2 cos + x1 sin. The
            // partner is a whole ciphertext, so this is two plaintext products
            // and one addition -- no rotation, no key switch, no key.
            //
            // The result is built beside the input rather than in place: every
            // pair reads both of its ciphertexts, and writing the low half
            // first would hand the high half its own already-rotated partner.
            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(slots.size());
            {
                SuffixRange _r("rope", "pairs");
                for (int j = 0; j < d; ++j)
                {
                    const int c = j % pairs;
                    const int partner = (j < pairs) ? j + pairs : j - pairs;

                    Ciphertext<Scheme::CKKS> direct(context_);
                    batch_.arith().multiply_plain(slots[j], cos_plain[c],
                                                  direct);
                    Ciphertext<Scheme::CKKS> cross(context_);
                    batch_.arith().multiply_plain(slots[partner], sin_plain[c],
                                                  cross);
                    if (j < pairs)
                    {
                        batch_.arith().sub_inplace(direct, cross);
                    }
                    else
                    {
                        batch_.arith().add_inplace(direct, cross);
                    }
                    batch_.arith().rescale_inplace(direct);
                    out.push_back(std::move(direct));
                }
            }
            slots = std::move(out);
        }

        BatchActivation
        Llama3RectOperator::to_batch_roped(RectActivation& in, int group,
                                           Galoiskey<Scheme::CKKS>& galois_key,
                                           const RectAttentionConfig& config)
        {
            if (!config.rope)
            {
                return to_batch(in, group, galois_key);
            }
            if (group < 0 || group >= in.groups)
            {
                throw std::invalid_argument(
                    "No such channel group in this activation");
            }

            Range _r("bridge.to_batch_roped");

            // The staged crossing, and only the staged one: the fused map
            // composes all three stages into a single diagonal set, which
            // leaves no slot midpoint to insert at. The caller is told so
            // rather than silently given a crossing without RoPE in it.
            BatchActivation borrowed = borrow_group(in, group);
            std::vector<Ciphertext<Scheme::CKKS>> slots =
                batch_.to_slots(borrowed, galois_key);
            return_group(in, group, borrowed);

            block_map(slots, /*inverse=*/true, "bridge.block_inverse",
                      galois_key);
            rope_slots(slots, config);
            return batch_.from_slots(slots, layout_.d, galois_key);
        }

        RectActivation
        Llama3RectOperator::from_batch(std::vector<BatchActivation>& groups,
                                       int channels,
                                       Galoiskey<Scheme::CKKS>& galois_key)
        {
            const int d = layout_.d;
            if (static_cast<int>(groups.size()) != groups_for(channels))
            {
                throw std::invalid_argument(
                    "The matrix encryptions do not cover the channel count they "
                    "are said to carry");
            }

            Range _r("bridge.from_batch");

            RectActivation out;
            out.rows = d;
            out.groups = static_cast<int>(groups.size());
            out.channels = channels;
            out.column.reserve(groups.size() * static_cast<size_t>(d));

            for (auto& group : groups)
            {
                if (group.columns() != d || group.rows != d)
                {
                    throw std::invalid_argument(
                        "A rectangular group comes from a matrix encryption "
                        "square at layout.d");
                }
                if (fused_crossings_)
                {
                    std::vector<Ciphertext<Scheme::CKKS>> fused =
                        group.column;
                    fused_apply(fused, FusedMap::FromBatch,
                                "bridge.fused_from_batch", galois_key);
                    for (auto& c : fused)
                    {
                        out.column.push_back(std::move(c));
                    }
                    continue;
                }
                std::vector<Ciphertext<Scheme::CKKS>> slots =
                    batch_.to_slots(group, galois_key);
                block_map(slots, /*inverse=*/false, "bridge.block_forward",
                          galois_key);
                BatchActivation rect = batch_.from_slots(slots, d, galois_key);
                for (auto& c : rect.column)
                {
                    out.column.push_back(std::move(c));
                }
            }
            return out;
        }

        // -------------------------------------------------------------------
        // The fused one-level crossings
        // -------------------------------------------------------------------

        namespace
        {
            /// One stage of a crossing: (shift, diagonal) pairs over the slot
            /// ring, exactly as the staged code applies them -- out[s] =
            /// sum diag[s] * in[(s + shift) mod n].
            using stage_t =
                std::vector<std::pair<int, const std::vector<Complex64>*>>;

            /// A dense map under composition: entry [shift] is the diagonal,
            /// empty meaning structurally zero.
            using dense_t = std::vector<std::vector<cd>>;

            dense_t densify(const stage_t& stage, int n)
            {
                dense_t out(static_cast<size_t>(n));
                for (const auto& entry : stage)
                {
                    std::vector<cd>& dst = out[static_cast<size_t>(entry.first)];
                    if (dst.empty())
                    {
                        dst.assign(static_cast<size_t>(n), cd(0.0, 0.0));
                    }
                    const std::vector<Complex64>& src = *entry.second;
                    for (int s = 0; s < n; ++s)
                    {
                        dst[static_cast<size_t>(s)] +=
                            cd(src[static_cast<size_t>(s)].real(),
                               src[static_cast<size_t>(s)].imag());
                    }
                }
                return out;
            }

            /// The map that applies @p first and then @p second.
            ///
            /// If first is sum_a A_a[s] in[s + sh_a] and second is
            /// sum_b B_b[s] mid[s + sh_b], the composition's diagonal at
            /// sh_a + sh_b picks up B_b[s] * A_a[s + sh_b]: the second map's
            /// shift walks over the first map's diagonal before the two shifts
            /// add. The double loop is |second stages| * n * n complex
            /// multiply-adds -- a few seconds of host arithmetic at N = 4096,
            /// paid once per operator per direction.
            dense_t compose(const dense_t& first, const stage_t& second, int n)
            {
                dense_t out(static_cast<size_t>(n));
                for (const auto& entry : second)
                {
                    const int sb = entry.first;
                    const std::vector<Complex64>& bv = *entry.second;
                    for (int sa = 0; sa < n; ++sa)
                    {
                        const std::vector<cd>& av =
                            first[static_cast<size_t>(sa)];
                        if (av.empty())
                        {
                            continue;
                        }
                        std::vector<cd>& dst =
                            out[static_cast<size_t>((sa + sb) % n)];
                        if (dst.empty())
                        {
                            dst.assign(static_cast<size_t>(n), cd(0.0, 0.0));
                        }
                        // Split at the wrap so the inner loop carries no
                        // modulus.
                        const int head = n - sb;
                        for (int s = 0; s < head; ++s)
                        {
                            dst[static_cast<size_t>(s)] +=
                                cd(bv[static_cast<size_t>(s)].real(),
                                   bv[static_cast<size_t>(s)].imag()) *
                                av[static_cast<size_t>(s + sb)];
                        }
                        for (int s = head; s < n; ++s)
                        {
                            dst[static_cast<size_t>(s)] +=
                                cd(bv[static_cast<size_t>(s)].real(),
                                   bv[static_cast<size_t>(s)].imag()) *
                                av[static_cast<size_t>(s + sb - n)];
                        }
                    }
                }
                return out;
            }
        } // namespace

        void Llama3RectOperator::set_fused_baby_steps(int n1)
        {
            if (n1 < 0 || (n1 > 0 && slot_count_ % n1 != 0))
            {
                throw std::invalid_argument(
                    "The fused baby step count must divide N/2, because the "
                    "two index sets have to cover the N/2 diagonals exactly");
            }
            fused_baby_steps_ = n1;
            fused_plain_.clear();
        }

        int Llama3RectOperator::fused_baby_steps() const
        {
            if (fused_baby_steps_ > 0)
            {
                return fused_baby_steps_;
            }
            // The largest power of two with n1 <= sqrt(N/2), so n2 >= n1 and
            // the bigger half of the rotations lands on the accumulator, one
            // limb below the source.
            int n1 = 1;
            while ((n1 * 2) * (n1 * 2) <= slot_count_)
            {
                n1 <<= 1;
            }
            return n1;
        }

        const std::vector<std::vector<Complex64>>&
        Llama3RectOperator::fused_table(FusedMap map)
        {
            std::vector<std::vector<Complex64>>& slot =
                fused_diagonal_[static_cast<size_t>(map)];
            if (!slot.empty())
            {
                return slot;
            }

            const int n = slot_count_;
            const int step = layout_.batch;

            const auto bridge_stage = [&](bool inverse) {
                const std::vector<std::vector<Complex64>>& diag =
                    inverse ? batch_.bridge_inverse_diagonals()
                            : batch_.bridge_forward_diagonals();
                stage_t st;
                st.reserve(diag.size());
                for (size_t delta = 0; delta < diag.size(); ++delta)
                {
                    st.emplace_back(
                        static_cast<int>(delta) * step % n, &diag[delta]);
                }
                return st;
            };
            const auto block_stage = [&](bool inverse) {
                const std::vector<std::vector<Complex64>>& diag =
                    inverse ? inverse_block_ : forward_block_;
                stage_t st;
                st.reserve(diag.size());
                for (size_t e = 0; e < diag.size(); ++e)
                {
                    const int eps = static_cast<int>(e) - (step - 1);
                    st.emplace_back(((eps % n) + n) % n, &diag[e]);
                }
                // Empty at step == 1, where the block transform is the
                // identity and composing it would be a no-op anyway.
                return st;
            };

            // The stages of each crossing in APPLICATION order, verbatim from
            // the staged entry points below -- the fused map must be their
            // product and nothing else.
            std::vector<stage_t> stages;
            switch (map)
            {
                case FusedMap::ToSlots:
                    stages = {bridge_stage(true), block_stage(true)};
                    break;
                case FusedMap::FromSlots:
                    stages = {block_stage(false), bridge_stage(false)};
                    break;
                case FusedMap::ToBatch:
                    stages = {bridge_stage(true), block_stage(true),
                              bridge_stage(false)};
                    break;
                case FusedMap::FromBatch:
                    stages = {bridge_stage(true), block_stage(false),
                              bridge_stage(false)};
                    break;
            }
            stages.erase(std::remove_if(stages.begin(), stages.end(),
                                        [](const stage_t& s)
                                        { return s.empty(); }),
                         stages.end());
            if (stages.empty())
            {
                throw std::runtime_error(
                    "A fused crossing with no stages has nothing to apply");
            }

            dense_t acc = densify(stages.front(), n);
            for (size_t i = 1; i < stages.size(); ++i)
            {
                acc = compose(acc, stages[i], n);
            }

            // Keep every diagonal that is not structural zero. The threshold
            // is relative to the largest entry, and the gap between a real
            // diagonal and cancellation noise is many orders of magnitude, so
            // its exact value does not matter.
            double largest = 0.0;
            for (const std::vector<cd>& v : acc)
            {
                for (const cd& z : v)
                {
                    largest = std::max(largest, std::abs(z));
                }
            }
            const double cutoff = largest * 1e-12;

            slot.assign(static_cast<size_t>(n), {});
            int kept = 0;
            for (int shift = 0; shift < n; ++shift)
            {
                const std::vector<cd>& v = acc[static_cast<size_t>(shift)];
                if (v.empty())
                {
                    continue;
                }
                double peak = 0.0;
                for (const cd& z : v)
                {
                    peak = std::max(peak, std::abs(z));
                }
                if (peak <= cutoff)
                {
                    continue;
                }
                std::vector<Complex64>& dst =
                    slot[static_cast<size_t>(shift)];
                dst.resize(static_cast<size_t>(n));
                for (int s = 0; s < n; ++s)
                {
                    dst[static_cast<size_t>(s)] =
                        Complex64(v[static_cast<size_t>(s)].real(),
                                  v[static_cast<size_t>(s)].imag());
                }
                ++kept;
            }
            if (kept == 0)
            {
                throw std::runtime_error(
                    "The fused crossing composed to zero, which means a stage "
                    "convention above is wrong");
            }
            return slot;
        }

        void Llama3RectOperator::fused_apply(
            std::vector<Ciphertext<Scheme::CKKS>>& ct, FusedMap map,
            const char* name, Galoiskey<Scheme::CKKS>& galois_key)
        {
            if (ct.empty())
            {
                return;
            }

            const int n = slot_count_;
            const std::vector<std::vector<Complex64>>& table =
                fused_table(map);
            const int n1 = fused_baby_steps();
            const int n2 = n / n1;

            const int depth = ct.front().depth();
            const double scale = ct.front().scale();
            for (const auto& c : ct)
            {
                if (c.depth() != depth || c.scale() != scale)
                {
                    throw std::invalid_argument(
                        "The fused crossing adds its diagonals together, so "
                        "every ciphertext in one call has to share a level "
                        "and a scale");
                }
            }

            const double plain_scale = rescale_prime(ct.front());

            // The encoded set, cached exactly the way the bridge caches its
            // own: keyed by what an encoding depends on, evicted whole.
            // Entries are stored in BSGS order for the PRESENT diagonals only,
            // pre-rotated by their giant step; presence is read off the table,
            // so the cursor below walks the two in lockstep.
            const auto plain_key = std::make_tuple(
                static_cast<int>(map), depth,
                static_cast<uint64_t>(plain_scale));
            auto encoded = fused_plain_.find(plain_key);
            if (encoded == fused_plain_.end())
            {
                SuffixRange _r(name, "encode");
                if (fused_plain_.size() >= fused_plain_capacity_)
                {
                    fused_plain_.erase(fused_plain_.begin());
                }
                std::vector<Plaintext<Scheme::CKKS>> plains;
                plains.reserve(static_cast<size_t>(n));
                std::vector<Complex64> shifted(static_cast<size_t>(n));
                const std::vector<Complex64> zero(static_cast<size_t>(n),
                                                  Complex64(0.0, 0.0));
                for (int i = 0; i < n2; ++i)
                {
                    const int giant = i * n1;
                    for (int j = 0; j < n1; ++j)
                    {
                        const std::vector<Complex64>& src =
                            table[static_cast<size_t>(giant + j)];
                        if (src.empty())
                        {
                            // Stored DENSE, so slot i*n1 + j is diagonal
                            // giant + j by arithmetic alone -- which is what
                            // lets the hoisted path hand a whole group to
                            // one fused launch. A structural zero is an
                            // encoded zero; the unhoisted loop still skips
                            // it off the table.
                            plains.push_back(
                                encode(zero, plain_scale, depth));
                            continue;
                        }
                        for (int p = 0; p < n; ++p)
                        {
                            // rot(v, -giant)[p] = v[p - giant], the inverse of
                            // the accumulator shift below.
                            const int q = ((p - giant) % n + n) % n;
                            shifted[static_cast<size_t>(p)] =
                                src[static_cast<size_t>(q)];
                        }
                        plains.push_back(
                            encode(shifted, plain_scale, depth));
                    }
                }
                EncodedDiagonalSet set;
                set.plains = std::move(plains);
                encoded =
                    fused_plain_.emplace(plain_key, std::move(set)).first;
            }

            // Which giant groups multiply anything at all, read off the
            // table: the dense set encodes zeros into the structural gaps,
            // so presence is no longer readable from the plaintexts.
            std::vector<char> group_live(static_cast<size_t>(n2), 0);
            for (int i = 0; i < n2; ++i)
            {
                for (int j = 0; j < n1; ++j)
                {
                    if (!table[static_cast<size_t>(i * n1 + j)].empty())
                    {
                        group_live[static_cast<size_t>(i)] = 1;
                        break;
                    }
                }
            }

            for (auto& source : ct)
            {
                if (hoisted_crossings_)
                {
                    // One decomposition for the whole baby train, one fused
                    // multiply-accumulate per giant group; identical modular
                    // arithmetic, identically ordered, so identical bits.
                    EncodedDiagonalSet& set = encoded->second;
                    if (set.packed.size() == 0)
                    {
                        set.packed = batch_.arith().pack_bsgs_plaintexts(
                            set.plains, depth);
                    }

                    DeviceVector<Data64> babies;
                    {
                        SuffixRange _r(name, "rotations");
                        std::vector<int> shifts(static_cast<size_t>(n1));
                        for (int j = 0; j < n1; ++j)
                        {
                            shifts[static_cast<size_t>(j)] = j;
                        }
                        babies = batch_.arith().hoisted_rotation_train(
                            source, shifts, n1, galois_key);
                    }

                    Ciphertext<Scheme::CKKS> total(context_);
                    bool total_started = false;
                    {
                        SuffixRange _r(name, "diagonals");
                        for (int i = 0; i < n2; ++i)
                        {
                            if (!group_live[static_cast<size_t>(i)])
                            {
                                continue;
                            }
                            Ciphertext<Scheme::CKKS> acc =
                                batch_.arith().hoisted_bsgs_group_sum(
                                    babies, set.packed, i * n1, n1, source,
                                    plain_scale);
                            batch_.arith().rescale_inplace(acc);
                            if (i != 0)
                            {
                                Ciphertext<Scheme::CKKS> moved(context_);
                                batch_.arith().rotate_rows(
                                    acc, moved, galois_key, i * n1);
                                acc = std::move(moved);
                            }
                            if (!total_started)
                            {
                                total = std::move(acc);
                                total_started = true;
                            }
                            else
                            {
                                batch_.arith().add_inplace(total, acc);
                            }
                        }
                    }
                    if (!total_started)
                    {
                        throw std::runtime_error(
                            "The fused crossing multiplied no diagonal at "
                            "all, which the table build above should have "
                            "refused");
                    }
                    source = std::move(total);
                    continue;
                }

                // The n1 baby shifts are shared by every giant step of this
                // column, so they are taken once.
                std::vector<Ciphertext<Scheme::CKKS>> baby;
                baby.reserve(static_cast<size_t>(n1));
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
                            batch_.arith().rotate_rows(source, shifted,
                                                       galois_key, j);
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
                            if (table[static_cast<size_t>(i * n1 + j)].empty())
                            {
                                continue;
                            }
                            Plaintext<Scheme::CKKS>& plain =
                                encoded->second.plains[static_cast<size_t>(
                                    i * n1 + j)];

                            Ciphertext<Scheme::CKKS> term(context_);
                            batch_.arith().multiply_plain(baby[j], plain,
                                                          term);

                            if (!started)
                            {
                                acc = std::move(term);
                                started = true;
                            }
                            else
                            {
                                batch_.arith().add_inplace(acc, term);
                            }
                        }
                        if (!started)
                        {
                            continue;
                        }

                        // rotate_rows refuses a ciphertext that still owes a
                        // rescale, so the rescale moves inside the loop, as in
                        // the bridge: it commutes with the rotation and every
                        // group is at the same level.
                        batch_.arith().rescale_inplace(acc);

                        if (i != 0)
                        {
                            Ciphertext<Scheme::CKKS> moved(context_);
                            batch_.arith().rotate_rows(acc, moved, galois_key,
                                                       i * n1);
                            acc = std::move(moved);
                        }

                        if (!total_started)
                        {
                            total = std::move(acc);
                            total_started = true;
                        }
                        else
                        {
                            batch_.arith().add_inplace(total, acc);
                        }
                    }
                }
                if (!total_started)
                {
                    throw std::runtime_error(
                        "The fused crossing multiplied no diagonal at all, "
                        "which the table build above should have refused");
                }
                source = std::move(total);
            }
        }

        // -------------------------------------------------------------------
        // The refresh
        // -------------------------------------------------------------------

        int Llama3RectOperator::RectRefreshConfig::count() const
        {
            return static_cast<int>(entry) +
                   static_cast<int>(after_attention_norm) +
                   static_cast<int>(post_qk) + static_cast<int>(post_softmax) +
                   static_cast<int>(mid) +
                   static_cast<int>(after_feed_forward_norm) +
                   static_cast<int>(feed_forward_hidden);
        }

        Ciphertext<Scheme::CKKS>
        Llama3RectOperator::bootstrap(Ciphertext<Scheme::CKKS>& ct,
                                      Galoiskey<Scheme::CKKS>& boot_key,
                                      Relinkey<Scheme::CKKS>& relin_key)
        {
            // Nothing here reads the encoding, and that is the point: the
            // procedure is an identity on the plaintext polynomial, so the
            // rectangular and the matrix encodings are refreshed where they
            // stand rather than crossed into slot form and back.
            return batch_.arith().bootstrap(ct, boot_key, relin_key);
        }

        Ciphertext<Scheme::CKKS> Llama3RectOperator::bootstrap_to_slots(
            Ciphertext<Scheme::CKKS>& ct, Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            // The one place this module does read the encoding. bootstrap()
            // above is encoding-blind because it puts the polynomial back the
            // way it found it; this one stops at the slot reading, which is
            // only the crossing the caller wanted because a rect column is
            // supported on coefficients 0..N/2-1.
            return batch_.arith().bootstrap_to_slots(ct, boot_key, relin_key);
        }

        std::vector<int> Llama3RectOperator::slot_reading_permutation() const
        {
            // MEASURED, not derived. profile_boot_to_slots matches every one
            // of the N/2 values by magnitude and reports which slot it landed
            // in; at logN 12, d 64 the answer was bit reversal on 2037 of 2048
            // coefficients, the other 11 being pairs of draws closer together
            // than the bootstrap's own 7e-6 error rather than exceptions.
            //
            // Bit reversal is what a decimation-in-time factorisation leaves
            // behind, and CoeffToSlot's is one. It is NOT what the crossing
            // leaves behind: to_slots() returns the natural order under the
            // d x (k/2) stride transpose, which the same profile confirms to
            // 1.8e-5. So the two readings of one rect column differ by bit
            // reversal composed with that transpose, and this returns the
            // first half of it -- where bootstrap_to_slots puts coefficient c,
            // which is where the rect encoding put token c mod d of block
            // c div d.
            const int half = layout_.N / 2;
            int bits = 0;
            while ((1 << bits) < half)
            {
                bits++;
            }

            std::vector<int> p(static_cast<size_t>(half));
            for (int c = 0; c < half; ++c)
            {
                unsigned r = 0;
                for (int b = 0; b < bits; ++b)
                {
                    r |= ((static_cast<unsigned>(c) >> b) & 1u)
                         << (bits - 1 - b);
                }
                p[static_cast<size_t>(c)] = static_cast<int>(r);
            }
            return p;
        }

        int Llama3RectOperator::island_slot(int block, int token,
                                            bool island) const
        {
            const int step = layout_.batch;
            const int d = layout_.d;
            if (block < 0 || block >= step || token < 0 || token >= d)
            {
                throw std::invalid_argument(
                    "A slot of this encoding is a block of a token, and both "
                    "have to be in range");
            }
            if (!island)
            {
                // to_slots' reading: the stride transpose, block on the fast
                // axis.
                return block + step * token;
            }

            // The island's: the same two fields, each bit reversed. Reversing
            // the WHOLE index of coefficient token + d*block gives exactly
            // this, which is why the fields do not move.
            auto reverse = [](int v, int width)
            {
                int bits = 0;
                while ((1 << bits) < width)
                {
                    bits++;
                }
                unsigned r = 0;
                for (int b = 0; b < bits; ++b)
                {
                    r |= ((static_cast<unsigned>(v) >> b) & 1u)
                         << (bits - 1 - b);
                }
                return static_cast<int>(r);
            };
            return reverse(block, step) + step * reverse(token, d);
        }

        std::vector<Ciphertext<Scheme::CKKS>>
        Llama3RectOperator::refresh_to_slots(RectActivation& x,
                                             Galoiskey<Scheme::CKKS>& boot_key,
                                             Relinkey<Scheme::CKKS>& relin_key)
        {
            require_uniform(x, "refresh_to_slots");

            Range _r("bridge.refresh_to_slots");

            std::vector<Ciphertext<Scheme::CKKS>> out;
            out.reserve(x.column.size());
            for (auto& column : x.column)
            {
                out.push_back(bootstrap_to_slots(column, boot_key, relin_key));
            }
            return out;
        }

        RectActivation Llama3RectOperator::slots_to_rect_at_level(
            std::vector<Ciphertext<Scheme::CKKS>>& in, int channels,
            Galoiskey<Scheme::CKKS>& boot_key, int pieces)
        {
            RectActivation out = empty_rect_for(in, channels);

            Range _r("bridge.slots_to_rect");
            for (auto& c : in)
            {
                out.column.push_back(
                    batch_.arith().slots_to_coeff_at_level(c, boot_key,
                                                           pieces));
            }
            return out;
        }

        RectActivation Llama3RectOperator::empty_rect_for(
            const std::vector<Ciphertext<Scheme::CKKS>>& in, int channels) const
        {
            const int d = layout_.d;
            if (in.empty() || in.size() % static_cast<size_t>(d) != 0)
            {
                throw std::invalid_argument(
                    "A rectangular group is exactly layout.d slot ciphertexts, "
                    "because d is the rank of the module and not a free "
                    "parameter");
            }
            const int groups = static_cast<int>(in.size()) / d;
            if (groups != groups_for(channels))
            {
                throw std::invalid_argument(
                    "The slot ciphertexts do not cover the channel count they "
                    "are said to carry");
            }

            RectActivation out;
            out.rows = d;
            out.groups = groups;
            out.channels = channels;
            out.column.reserve(in.size());
            return out;
        }

        RectActivation Llama3RectOperator::slots_to_rect(
            std::vector<Ciphertext<Scheme::CKKS>>& in, int channels,
            Galoiskey<Scheme::CKKS>& boot_key, int align_drop)
        {
            RectActivation out = empty_rect_for(in, channels);

            Range _r("bridge.slots_to_rect");

            // Column by column and nothing else. from_slots has to gather a
            // whole group because block_map mixes the d ciphertexts of it;
            // this way back does not mix them, because the block axis was
            // never separated out -- CoeffToSlot took the whole N/2-point
            // transform in one piece and SlotToCoeff gives it back the same
            // way.
            for (auto& c : in)
            {
                out.column.push_back(
                    batch_.arith().slots_to_coeff(c, boot_key, align_drop));
            }
            return out;
        }

        void Llama3RectOperator::bootstrap(
            std::vector<Ciphertext<Scheme::CKKS>>& ct, const char* name,
            Galoiskey<Scheme::CKKS>& boot_key,
            Relinkey<Scheme::CKKS>& relin_key)
        {
            if (ct.empty())
            {
                return;
            }

            Range _r(name);
            // One at a time. A bootstrap fills the slots on its own, so there
            // is nothing for two of them to share, and holding a second full
            // ciphertext at the raised modulus is exactly what a card this
            // full does not have.
            for (auto& c : ct)
            {
                c = bootstrap(c, boot_key, relin_key);
            }
        }

        void Llama3RectOperator::bootstrap(RectActivation& x, const char* name,
                                           Galoiskey<Scheme::CKKS>& boot_key,
                                           Relinkey<Scheme::CKKS>& relin_key)
        {
            bootstrap(x.column, name, boot_key, relin_key);
        }

        void Llama3RectOperator::bootstrap(BatchActivation& x, const char* name,
                                           Galoiskey<Scheme::CKKS>& boot_key,
                                           Relinkey<Scheme::CKKS>& relin_key)
        {
            bootstrap(x.column, name, boot_key, relin_key);
        }

        void Llama3RectOperator::note_depth(
            const char* name,
            const std::vector<Ciphertext<Scheme::CKKS>>& ct) const
        {
            if (depth_trace && !ct.empty())
            {
                depth_trace(name, ct.front().depth());
            }
            if (ct_trace && !ct.empty())
            {
                ct_trace(name, ct);
            }
        }

        void Llama3RectOperator::fold_scale(std::vector<double>& weight,
                                            const std::vector<double>& scale,
                                            int in_channels, int out_channels)
        {
            if (scale.empty())
            {
                return;
            }
            if (static_cast<int>(scale.size()) != in_channels ||
                weight.size() != static_cast<size_t>(in_channels) *
                                     static_cast<size_t>(out_channels))
            {
                throw std::invalid_argument(
                    "A folded scale is one entry per input channel of a row "
                    "major in_channels x out_channels weight");
            }

            for (int i = 0; i < in_channels; ++i)
            {
                const double g = scale[i];
                double* row =
                    weight.data() + static_cast<size_t>(i) * out_channels;
                for (int j = 0; j < out_channels; ++j)
                {
                    row[j] *= g;
                }
            }
        }

        // -------------------------------------------------------------------
        // The products
        // -------------------------------------------------------------------

        std::vector<int64_t> Llama3RectOperator::rectangular_weight(
            const std::vector<double>& weight, int in_channels,
            int out_channels, int in_group, int out_group, double scale) const
        {
            const int d = layout_.d;
            const int k = layout_.k;
            const int step = layout_.batch;
            const int half = layout_.N / 2;

            // Algorithm 5's plaintext is always d x (N/2) whatever the weight
            // actually is, so a narrow projection uploads as much as a wide
            // one. Block row t sits at Y^{k-t} with a minus sign, which is what
            // makes the constant coefficient of the R_k product the contraction
            // over t: (ab)_0 = a_0 b_0 - sum_{t>0} a_t b_{k-t}, and Y^k = -1.
            std::vector<int64_t> coeffs(
                static_cast<size_t>(d) * half * k, 0);

            for (int j = 0; j < d; ++j)
            {
                for (int c = 0; c < half; ++c)
                {
                    const int out_c = out_group * half + c;
                    if (out_c >= out_channels)
                    {
                        continue;
                    }
                    int64_t* e =
                        coeffs.data() + (static_cast<size_t>(j) * half + c) * k;
                    for (int t = 0; t < step; ++t)
                    {
                        const int in_c = in_group * half + t * d + j;
                        if (in_c >= in_channels)
                        {
                            continue;
                        }
                        const int64_t v = quantise(
                            weight[static_cast<size_t>(in_c) * out_channels +
                                   out_c],
                            scale);
                        if (t == 0)
                        {
                            e[0] = v;
                        }
                        else
                        {
                            e[k - t] = -v;
                        }
                    }
                }
            }
            return coeffs;
        }

        RectActivation
        Llama3RectOperator::project(RectActivation& x,
                                    const std::vector<double>& weight,
                                    int in_channels, int out_channels,
                                    const char* name,
                                    Galoiskey<Scheme::CKKS>& galois_key)
        {
            require_uniform(x, name);
            if (x.channels != in_channels)
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

            const int d = layout_.d;
            const int half = layout_.N / 2;
            const int gin = x.groups;
            const int gout = groups_for(out_channels);

            char range_name[64];
            std::snprintf(range_name, sizeof(range_name), "project.%s", name);
            Range _r_project(range_name);

            // The weight is encoded at the prime the rescale that follows will
            // divide by, so the product comes back at exactly the activation's
            // own scale one level down. Encoding it at the nominal scale
            // instead leaves the result at x_scale * plain_scale / prime, which
            // is not a scale anything downstream expects and reads as garbage
            // rather than as a mismatch.
            const double plain_scale = rescale_prime(x.column.front());
            const int depth = x.column.front().depth();

            RectActivation out;
            out.rows = d;
            out.groups = gout;
            out.channels = out_channels;
            out.column.reserve(static_cast<size_t>(gout) * d);

            for (int go = 0; go < gout; ++go)
            {
                std::vector<Ciphertext<Scheme::CKKS>> acc;

                for (int gi = 0; gi < gin; ++gi)
                {
                    {
                        SuffixRange _r(range_name, "weight_encode");
                        const std::vector<int64_t> w = rectangular_weight(
                            weight, in_channels, out_channels, gi, go,
                            plain_scale);
                        batch_.matrix().encode_plaintext_matrix(
                            w, d, half, depth, plain_scale);
                    }

                    std::vector<Ciphertext<Scheme::CKKS>*> in;
                    in.reserve(d);
                    for (int j = 0; j < d; ++j)
                    {
                        in.push_back(
                            &x.column[static_cast<size_t>(gi) * d + j]);
                    }

                    std::vector<Ciphertext<Scheme::CKKS>> piece;
                    {
                        SuffixRange _r(range_name, "rectangular_pcmm");
                        batch_.matrix().rectangular_pcmm(
                            piece, in, galois_key, batch_.arith(),
                            HEBatchMatrixOperator<Scheme::CKKS>::BlockAxis::
                                coefficient,
                            /*rescale=*/true);
                    }

                    // Algorithm 5, like Algorithm 1, only MARKS the rescale.
                    // Left unspent that is not merely an untidy scale: the
                    // plaintext's centered coefficients have outgrown int64 and
                    // the next call cannot read them back.
                    for (auto& c : piece)
                    {
                        batch_.arith().rescale_inplace(c);
                    }

                    if (acc.empty())
                    {
                        acc = std::move(piece);
                    }
                    else
                    {
                        // Every input group's partial product leaves the same
                        // sequence of operations behind it, so they meet at one
                        // level and one scale and the sum is a plain addition.
                        SuffixRange _r(range_name, "accumulate");
                        for (int j = 0; j < d; ++j)
                        {
                            batch_.arith().add_inplace(acc[j], piece[j]);
                        }
                    }
                }

                for (auto& c : acc)
                {
                    out.column.push_back(std::move(c));
                }
            }

            return out;
        }

        BatchActivation
        Llama3RectOperator::matmul(BatchActivation& a, BatchActivation& b,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key,
                                   Relinkey<Scheme::CKKS>& relin_key)
        {
            return batch_.matmul(a, b, name, galois_key, relin_key);
        }

        // -------------------------------------------------------------------
        // RMSNorm
        // -------------------------------------------------------------------

        RectActivation
        Llama3RectOperator::rms_norm(RectActivation& x,
                                     const std::vector<double>& weight,
                                     const RectRMSNormConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key,
                                     Galoiskey<Scheme::CKKS>* boot_key)
        {
            require_uniform(x, "rms_norm");
            const int channels = x.channels;
            if (!weight.empty() &&
                static_cast<int>(weight.size()) != channels)
            {
                throw std::invalid_argument(
                    "RMSNorm takes one learned scale per channel");
            }

            Range _r("rms_norm");

            const int d = layout_.d;
            const int step = layout_.batch;
            const int half = layout_.N / 2;
            const int groups = x.groups;

            // The crossings are named for their caller as well as for
            // themselves. bridge.* aggregates every crossing in the circuit,
            // which is the number worth knowing about this path; these say
            // which sublayer paid it.
            const bool island = config.fused_refresh;
            if (island && boot_key == nullptr)
            {
                throw std::invalid_argument(
                    "The fused refresh IS the crossing, so it needs the boot "
                    "key at the call and not merely a Galois key wide enough "
                    "for the bridge");
            }

            std::vector<Ciphertext<Scheme::CKKS>> slots;
            if (island)
            {
                Range _r_in("rms_norm.refresh_to_slots");
                slots = refresh_to_slots(x, *boot_key, relin_key);
            }
            else
            {
                Range _r_in("rms_norm.to_slots");
                slots = to_slots(x, galois_key);
            }

            // A channel is (ciphertext, fast slot index), so the learned scale
            // is a slot vector over the block axis and constant along the token
            // axis -- not a constant, as it is when a channel is a whole
            // ciphertext.
            std::vector<Plaintext<Scheme::CKKS>> weights;
            if (!weight.empty())
            {
                SuffixRange _r_w("rms_norm", "weight_encode");
                const double plain_scale = rescale_prime(slots.front());
                const int depth = slots.front().depth();
                weights.reserve(slots.size());
                for (int g = 0; g < groups; ++g)
                {
                    for (int j = 0; j < d; ++j)
                    {
                        std::vector<double> flat(slot_count_, 0.0);
                        for (int b = 0; b < step; ++b)
                        {
                            const int c = g * half + b * d + j;
                            const double w =
                                (c < channels) ? weight[c] : 0.0;
                            for (int u = 0; u < d; ++u)
                            {
                                // The one line the island changes. The two
                                // readings put the same two fields on the same
                                // two axes and differ only in the order inside
                                // each, so a learned scale that varies along
                                // the block axis has to be written at the
                                // reversed position -- and that is the whole
                                // of the relabelling.
                                flat[island_slot(b, u, island)] = w;
                            }
                        }
                        Plaintext<Scheme::CKKS> plain(context_);
                        encoder_.encode(plain, flat, plain_scale);
                        for (int i = 0; i < depth; ++i)
                        {
                            batch_.arith().mod_drop_inplace(plain);
                        }
                        weights.push_back(std::move(plain));
                    }
                }
            }

            Llama3Operator::RMSNormConfig slot_config;
            // The channel axis runs partly across ciphertexts, where the sum is
            // free, and partly along the fast slot axis, where it is the one
            // masked reduction this encoding pays for.
            slot_config.stride = slot_count_;
            slot_config.count = step;
            slot_config.blocked_span = step;
            slot_config.channels = channels;
            slot_config.token_blocks = 1;
            slot_config.eps = config.eps;
            slot_config.sum_lo = config.sum_lo;
            slot_config.sum_hi = config.sum_hi;
            slot_config.degree = config.degree;
            slot_config.newton_iterations = config.newton_iterations;
            slot_config.fold_mean_into_fit = config.fold_mean_into_fit;
            slot_config.fold_affine_into_mask = config.fold_affine_into_mask;
            slot_config.output_scale = config.output_scale;
            slot_config.refresh_sum = config.refresh_sum;

            std::vector<Ciphertext<Scheme::CKKS>> normalised =
                batch_.arith().rms_norm(slots, weights, slot_config, galois_key,
                                        relin_key, boot_key);

            if (island)
            {
                // Never from_slots() here: that crossing reads the natural
                // order, and this one is reversed. slots_to_rect is the same
                // Vandermonde the refresh ran forwards, run backwards, so the
                // reversal cancels rather than having to be undone.
                //
                // The at_level form and not the locked one, because the norm
                // has just spent levels and the locked form's diagonals sit
                // where the refresh left off.
                Range _r_out("rms_norm.slots_to_rect");
                return slots_to_rect_at_level(normalised, channels, *boot_key);
            }

            Range _r_out("rms_norm.from_slots");
            return from_slots(normalised, channels, galois_key);
        }

        // -------------------------------------------------------------------
        // Attention
        // -------------------------------------------------------------------

        RectActivation
        Llama3RectOperator::attention(RectActivation& x,
                                      const RectAttentionWeights& weights,
                                      const RectAttentionConfig& config,
                                      Galoiskey<Scheme::CKKS>& galois_key,
                                      Relinkey<Scheme::CKKS>& relin_key,
                                      Galoiskey<Scheme::CKKS>* boot_key,
                                      const RectRefreshConfig* refresh)
        {
            const bool refresh_qk =
                boot_key != nullptr && refresh != nullptr && refresh->post_qk;
            const bool refresh_pv = boot_key != nullptr && refresh != nullptr &&
                                    refresh->post_softmax;

            const int d = layout_.d;
            const int half = layout_.N / 2;
            const int heads = config.heads;
            const int kv_heads =
                config.kv_heads > 0 ? config.kv_heads : config.heads;

            if (heads < 1 || kv_heads < 1 || heads % kv_heads != 0)
            {
                throw std::invalid_argument(
                    "Grouped-query attention needs kv_heads to divide heads");
            }

            const int q_channels = heads * d;
            const int kv_channels = kv_heads * d;
            if (q_channels % half != 0)
            {
                throw std::invalid_argument(
                    "heads * d must be a whole number of N/2 channel groups: a "
                    "group is what one Algorithm 4 call covers, and a partly "
                    "filled one would carry heads that do not exist");
            }
            if (weights.query.size() != static_cast<size_t>(config.in_channels) *
                                            static_cast<size_t>(q_channels) ||
                weights.key.size() != static_cast<size_t>(config.in_channels) *
                                          static_cast<size_t>(kv_channels) ||
                weights.value.size() != weights.key.size())
            {
                throw std::invalid_argument(
                    "The attention weights must be in_channels by heads * d, "
                    "with the key and value narrower by heads / kv_heads");
            }

            Range _r_attention("attention");

            // 1/sqrt(head_dim) rides on the query weight, and head_dim is d
            // here. A scaling is free on the host and one homomorphic level
            // anywhere else.
            const double head_scale =
                config.head_scale != 0.0
                    ? config.head_scale
                    : 1.0 / std::sqrt(static_cast<double>(d));

            // And so does the domain map of the exponential, for exactly the
            // same reason. The SoftMax fits exp over [-bound, 0] and has to
            // carry its argument onto [-1, 1] first, which is a plaintext
            // product and a level -- unless the scores arrive already carrying
            // it, and every score is a linear function of this weight.
            const bool fold_exp_domain =
                config.fold_softmax_domain_into_query &&
                config.softmax.bound > 0.0;
            const double exp_domain =
                fold_exp_domain
                    ? Llama3Operator::domain_scale(-config.softmax.bound, 0.0)
                    : 1.0;

            std::vector<double> query_weight = weights.query;
            for (auto& w : query_weight)
            {
                w *= head_scale * exp_domain;
            }

            // Grouped-query attention, expanded on the host. A kv head has to
            // appear in every batch slot that reads it, and the batch slots of
            // a matrix encryption never talk to each other, so there is nowhere
            // else to do this. It costs projection work in the ratio
            // heads / kv_heads.
            const int group_size = heads / kv_heads;
            auto expand = [&](const std::vector<double>& src)
            {
                if (group_size == 1)
                {
                    return src;
                }
                std::vector<double> dst(
                    static_cast<size_t>(config.in_channels) * q_channels);
                for (int i = 0; i < config.in_channels; ++i)
                {
                    for (int h = 0; h < heads; ++h)
                    {
                        const int src_head = h / group_size;
                        for (int c = 0; c < d; ++c)
                        {
                            dst[static_cast<size_t>(i) * q_channels + h * d +
                                c] =
                                src[static_cast<size_t>(i) * kv_channels +
                                    src_head * d + c];
                        }
                    }
                }
                return dst;
            };

            RectActivation q =
                project(x, query_weight, config.in_channels, q_channels,
                        "attention.q", galois_key);
            RectActivation k =
                project(x, expand(weights.key), config.in_channels, q_channels,
                        "attention.k", galois_key);
            RectActivation v =
                project(x, expand(weights.value), config.in_channels,
                        q_channels, "attention.v", galois_key);
            note_depth("attention.projected", q.column);

            const int q_groups = q.groups;
            std::vector<BatchActivation> out_groups;
            out_groups.reserve(q_groups);

            for (int g = 0; g < q_groups; ++g)
            {
                // Batch slot b of a group is channel block b, and with
                // head_dim = d that is head g * (k/2) + b. So one Algorithm 4
                // call below covers k/2 heads.
                BatchActivation qb, kb, vb;
                {
                    Range _r("attention.to_batch");
                    // RoPE rides on the crossing Q and K were going to pay
                    // for anyway; V is not rotated and takes the plain one,
                    // and the level it keeps over them is dropped where it
                    // meets P below.
                    qb = to_batch_roped(q, g, galois_key, config);
                    kb = to_batch_roped(k, g, galois_key, config);
                    vb = to_batch(v, g, galois_key);
                }

                // K^T, one CMT. This is the only transpose the sublayer pays
                // for: the value product P V already reads V with the key on
                // its rows, which is where the projection left it.
                BatchActivation kt =
                    batch_.transpose(kb, "attention.key", galois_key);
                kb.column.clear();

                BatchActivation scores;
                {
                    Range _r("attention.scores");
                    scores = matmul(qb, kt, "attention.score", galois_key,
                                    relin_key);
                }
                qb.column.clear();
                kt.column.clear();

                // The SoftMax is slot-wise, so this is where the sublayer
                // leaves the matrix encoding -- and the only place it does.
                std::vector<Ciphertext<Scheme::CKKS>> slots;
                {
                    Range _r("attention.to_slots");
                    slots = batch_.to_slots(scores, galois_key);
                }
                scores.column.clear();
                note_depth("attention.scored", slots);

                // The post-QK refresh. The SoftMax is the deepest stretch of
                // the sublayer and there is nothing left for it by here, so
                // this seam is the one the schedule is built around.
                if (refresh_qk)
                {
                    bootstrap(slots, "attention.refresh_post_qk", *boot_key,
                              relin_key);
                    note_depth("attention.refresh_post_qk", slots);
                }

                // The shift is scaled with the scores it is subtracted from.
                // It is a constant either way, so this is free either way.
                if (!config.score_shift_rows.empty())
                {
                    // Slot b + step*u of every part holds query u of head
                    // g*step + b, so the per-(row, head) shift is one slot
                    // vector per group, encoded once and added to every
                    // part. Constant along the key axis, it cancels in the
                    // normalisation and only the fitted ranges feel it.
                    if (static_cast<int>(config.score_shift_rows.size()) !=
                        d * config.heads)
                    {
                        throw std::invalid_argument(
                            "score_shift_rows must be d x heads");
                    }
                    std::vector<double> flat(slot_count_, 0.0);
                    const int step = layout_.batch;
                    for (int b = 0; b < step; ++b)
                    {
                        for (int u = 0; u < d; ++u)
                        {
                            flat[b + u * step] =
                                -config.score_shift_rows
                                     [static_cast<std::size_t>(u) *
                                          config.heads +
                                      g * step + b] *
                                exp_domain;
                        }
                    }
                    batch_.arith().add_vector(slots, flat);
                }
                else if (config.score_shift != 0.0)
                {
                    for (auto& c : slots)
                    {
                        batch_.arith().add_constant(
                            c, -config.score_shift * exp_domain);
                    }
                }
                note_depth("attention.shifted", slots);

                Llama3Operator::SoftmaxConfig softmax = config.softmax;
                // The key axis is entirely across ciphertexts: one coordinate
                // per part, so nothing is reduced inside a ciphertext and the
                // denominator costs a slot-wise addition and no rotation.
                softmax.strided = true;
                softmax.stride = slot_count_;
                softmax.count = 1;
                softmax.pre_scaled_input = fold_exp_domain;
                // Round zero's domain map rides on the causal mask, so it can
                // only be folded when there is one.
                softmax.fold_affine_into_mask =
                    config.fold_softmax_affine_into_mask && config.causal;

                std::vector<std::vector<double>> masks;
                if (config.causal)
                {
                    masks.reserve(d);
                    for (int j = 0; j < d; ++j)
                    {
                        masks.push_back(batch_.causal_column_mask(j));
                    }
                }

                std::vector<Ciphertext<Scheme::CKKS>> p_slots;
                {
                    Range _r("attention.softmax");
                    p_slots = batch_.arith().softmax(
                        slots, softmax, masks, galois_key, relin_key,
                        boot_key);
                }
                slots.clear();
                note_depth("attention.softmaxed", p_slots);

                // The post-SoftMax refresh, taken on P and on V together. They
                // meet in the value product below, and the only way down
                // between two levels is a drop: refreshing P alone would put it
                // ABOVE V, which is a direction no ciphertext travels.
                if (refresh_pv)
                {
                    bootstrap(p_slots, "attention.refresh_post_softmax",
                              *boot_key, relin_key);
                    bootstrap(vb, "attention.refresh_value", *boot_key,
                              relin_key);
                    note_depth("attention.refresh_post_softmax", p_slots);
                }

                BatchActivation p;
                {
                    Range _r("attention.from_slots");
                    p = batch_.from_slots(p_slots, d, galois_key);
                }
                p_slots.clear();

                // V is still where the projection left it, several levels
                // above P, so it comes down to meet it. The drop is free and
                // what it discards was unreachable anyway.
                {
                    Range _r("attention.value_product");
                    const int depth = p.column.front().depth();
                    for (auto& c : vb.column)
                    {
                        batch_.arith().drop_to_depth(c, depth);
                    }
                    out_groups.push_back(
                        matmul(p, vb, "attention.value", galois_key,
                               relin_key));
                }
            }

            q.column.clear();
            k.column.clear();
            v.column.clear();

            RectActivation out;
            {
                Range _r("attention.from_batch");
                out = from_batch(out_groups, q_channels, galois_key);
            }
            out_groups.clear();

            if (weights.output.empty())
            {
                note_depth("attention.out", out.column);
                return out;
            }
            RectActivation projected =
                project(out, weights.output, q_channels, config.in_channels,
                        "attention.o", galois_key);
            note_depth("attention.out", projected.column);
            return projected;
        }

        // -------------------------------------------------------------------
        // SwiGLU
        // -------------------------------------------------------------------

        RectActivation
        Llama3RectOperator::feed_forward(RectActivation& x,
                                         const RectFeedForwardWeights& weights,
                                         const RectFeedForwardConfig& config,
                                         Galoiskey<Scheme::CKKS>& galois_key,
                                         Relinkey<Scheme::CKKS>& relin_key,
                                         Galoiskey<Scheme::CKKS>* boot_key,
                                         const RectRefreshConfig* refresh)
        {
            require_uniform(x, "feed_forward");

            const bool refresh_seam = boot_key != nullptr &&
                                      refresh != nullptr &&
                                      refresh->feed_forward_hidden;
            // The same seam, one level earlier: on the activation instead of
            // on the hidden. It is worth a level because the stretch that
            // sets the chain ends at the seam, and the gate product is the
            // last thing before it.
            const bool refresh_act = refresh_seam && config.refresh_activation;
            const bool refresh_hidden = refresh_seam && !refresh_act;

            // A bootstrap wants its slots on [-1, 1] and the fit returns the
            // activation on [-B, B], so the fit is given a gain of 1/B and
            // the up weight carries B back. Both ends are host-side numbers,
            // so the scaling Section 3.1.3 asks for costs nothing here.
            double activation_scale = 1.0;
            if (refresh_act)
            {
                activation_scale = config.activation_bound > 0.0
                                       ? config.activation_bound
                                       : config.silu_bound;
                if (!(activation_scale > 0.0))
                {
                    throw std::invalid_argument(
                        "Refreshing the activation needs a positive bound to "
                        "scale it onto, and silu_bound is not one");
                }
            }

            Range _r("feed_forward");

            const int in_channels = config.in_channels;
            const int hidden_channels = config.hidden_channels;
            const int half = layout_.N / 2;

            if (weights.gate.size() !=
                    static_cast<size_t>(in_channels) *
                        static_cast<size_t>(hidden_channels) ||
                weights.up.size() != weights.gate.size() ||
                weights.down.size() != weights.gate.size())
            {
                throw std::invalid_argument(
                    "The SwiGLU weights must be in_channels by "
                    "hidden_channels, and down its transpose shape");
            }
            if (hidden_channels % half != 0)
            {
                throw std::invalid_argument(
                    "The hidden width must be a whole number of N/2 channel "
                    "groups, since the chunks are projected down and summed");
            }

            const int hidden_groups = hidden_channels / half;
            int block = config.hidden_block_groups > 0
                            ? config.hidden_block_groups
                            : hidden_groups;
            block = std::min(block, hidden_groups);

            RectActivation out;
            out.rows = layout_.d;
            out.groups = groups_for(in_channels);
            out.channels = in_channels;

            for (int base = 0; base < hidden_channels; base += block * half)
            {
                const int cols =
                    std::min(block * half, hidden_channels - base);

                // The gate and up weights are in_channels x hidden_channels, so
                // a chunk of the hidden axis is a set of columns and has to be
                // gathered. The down weight is hidden_channels x in_channels, so
                // the same chunk is a contiguous span of rows.
                // The SiLU's domain map rides on the gate weight when asked
                // to: the gate feeds the fit and nothing else, so the factor
                // never has to come back out, and a host scaling is free
                // where the map is otherwise a plaintext product and a level.
                const bool fold_silu = config.fold_silu_domain_into_gate &&
                                       config.silu_bound > 0.0;
                const double silu_domain =
                    fold_silu ? Llama3Operator::domain_scale(
                                    -config.silu_bound, config.silu_bound)
                              : 1.0;

                std::vector<double> gate_w(
                    static_cast<size_t>(in_channels) * cols);
                std::vector<double> up_w(gate_w.size());
                for (int i = 0; i < in_channels; ++i)
                {
                    const size_t src =
                        static_cast<size_t>(i) * hidden_channels + base;
                    const size_t dst = static_cast<size_t>(i) * cols;
                    for (int j = 0; j < cols; ++j)
                    {
                        gate_w[dst + j] = weights.gate[src + j] * silu_domain;
                        // The up weight undoes the activation's refresh
                        // scaling on the very product the activation was
                        // heading for, so the scaling is free at both ends.
                        up_w[dst + j] =
                            weights.up[src + j] * activation_scale;
                    }
                }

                RectActivation gate = project(x, gate_w, in_channels, cols,
                                              "ffn.gate", galois_key);
                RectActivation up =
                    project(x, up_w, in_channels, cols, "ffn.up", galois_key);

                // The gate meets the up projection in a Hadamard product, which
                // this encoding does not have: multiplying two columns
                // convolves their coefficients. Both branches therefore cross
                // to slot form, where a product is slot-wise, and the result
                // crosses back for the down projection.
                std::vector<Ciphertext<Scheme::CKKS>> gate_slots, up_slots;
                {
                    Range _r("ffn.to_slots");
                    gate_slots = to_slots(gate, galois_key);
                    up_slots = to_slots(up, galois_key);
                }
                gate.column.clear();
                up.column.clear();

                // The fit and the gate product are separated so the seam can
                // sit between them. With activation_scale at 1 the gain is 1
                // and this is the fused loop it replaces, ciphertext for
                // ciphertext.
                std::vector<Ciphertext<Scheme::CKKS>> activated;
                activated.reserve(gate_slots.size());
                {
                    Range _r_silu("ffn.silu");
                    for (size_t j = 0; j < gate_slots.size(); ++j)
                    {
                        activated.push_back(batch_.arith().silu(
                            gate_slots[j], config.silu_bound,
                            config.silu_degree, relin_key, fold_silu,
                            1.0 / activation_scale));
                    }
                }
                gate_slots.clear();
                note_depth("ffn.activation", activated);

                // The seam, taken before the product: the stretch from the
                // norm's refresh ends here rather than one level further
                // down, which is the level this buys.
                if (refresh_act)
                {
                    bootstrap(activated, "ffn.refresh_activation", *boot_key,
                              relin_key);
                    note_depth("ffn.refresh_activation", activated);
                }

                std::vector<Ciphertext<Scheme::CKKS>> hidden;
                hidden.reserve(activated.size());
                {
                    Range _r_product("ffn.gate_product");
                    for (size_t j = 0; j < activated.size(); ++j)
                    {
                        hidden.push_back(batch_.arith().multiply_and_rescale(
                            activated[j], up_slots[j], relin_key));
                    }
                }
                activated.clear();
                up_slots.clear();
                note_depth("ffn.activated", hidden);

                // The SwiGLU half fits between two refreshes without this one,
                // so it is off by default; it is here because the SiLU degree
                // is the knob most likely to be turned back up, and this is the
                // seam that pays for it when it is.
                if (refresh_hidden)
                {
                    bootstrap(hidden, "ffn.refresh_hidden", *boot_key,
                              relin_key);
                    note_depth("ffn.refresh_hidden", hidden);
                }

                RectActivation h;
                {
                    Range _r("ffn.from_slots");
                    h = from_slots(hidden, cols, galois_key);
                }
                hidden.clear();

                const std::vector<double> down_w(
                    weights.down.begin() +
                        static_cast<size_t>(base) * in_channels,
                    weights.down.begin() +
                        static_cast<size_t>(base + cols) * in_channels);
                RectActivation part = project(h, down_w, cols, in_channels,
                                              "ffn.down", galois_key);

                if (out.column.empty())
                {
                    out.column = std::move(part.column);
                }
                else
                {
                    // Every chunk's partial product leaves the same sequence of
                    // operations behind it, so they meet at one level and one
                    // scale and the sum is a plain addition.
                    Range _r_acc("ffn.accumulate");
                    for (size_t j = 0; j < out.column.size(); ++j)
                    {
                        batch_.arith().add_inplace(out.column[j],
                                                   part.column[j]);
                    }
                }
            }

            return out;
        }

        // -------------------------------------------------------------------
        // The whole block
        // -------------------------------------------------------------------

        RectActivation Llama3RectOperator::transformer_block(
            RectActivation& x, const RectTransformerBlockWeights& weights,
            const RectTransformerBlockConfig& config,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            Galoiskey<Scheme::CKKS>* boot_key)
        {
            require_uniform(x, "transformer_block");

            Range _r("transformer_block");

            const RectRefreshConfig& refresh = config.refresh;
            const RectRefreshConfig* plan =
                boot_key != nullptr ? &refresh : nullptr;

            RectActivation stream;
            stream.rows = x.rows;
            stream.groups = x.groups;
            stream.channels = x.channels;
            stream.column = x.column;

            // The learned gains, folded into the projections that read them
            // rather than applied homomorphically. Row i of a projection is
            // the input channel the gain belongs to, so this is exact, and it
            // is the one level a pre-norm block gives up for nothing.
            // Only the projections that read the NORMALISED stream take the
            // gain. W_o reads the attention output and W_down the hidden, so
            // neither of them does.
            RectAttentionWeights folded_attention;
            RectFeedForwardWeights folded_feed_forward;
            const RectAttentionWeights* attention_weights = &weights.attention;
            const RectFeedForwardWeights* feed_forward_weights =
                &weights.feed_forward;
            std::vector<double> attention_gain = weights.attention_norm;
            std::vector<double> feed_forward_gain = weights.feed_forward_norm;
            if (config.fold_norm_scale)
            {
                const int in = config.attention.in_channels;
                const int kv = (config.attention.kv_heads > 0
                                    ? config.attention.kv_heads
                                    : config.attention.heads) *
                               layout_.d;
                folded_attention = weights.attention;
                folded_feed_forward = weights.feed_forward;
                fold_scale(folded_attention.query, attention_gain, in,
                           config.attention.heads * layout_.d);
                fold_scale(folded_attention.key, attention_gain, in, kv);
                fold_scale(folded_attention.value, attention_gain, in, kv);
                fold_scale(folded_feed_forward.gate, feed_forward_gain,
                           config.feed_forward.in_channels,
                           config.feed_forward.hidden_channels);
                fold_scale(folded_feed_forward.up, feed_forward_gain,
                           config.feed_forward.in_channels,
                           config.feed_forward.hidden_channels);
                attention_weights = &folded_attention;
                feed_forward_weights = &folded_feed_forward;
                attention_gain.clear();
                feed_forward_gain.clear();
            }

            note_depth("block.entry", stream.column);
            if (boot_key != nullptr && refresh.entry)
            {
                bootstrap(stream, "block.refresh_entry", *boot_key, relin_key);
                note_depth("block.refresh_entry", stream.column);
            }

            {
                Range _r_half("transformer_block.attention_half");
                RectActivation normed =
                    rms_norm(stream, attention_gain, config.attention_norm,
                             galois_key, relin_key, boot_key);
                note_depth("block.attention_norm", normed.column);

                // RMSNorm is the deepest single stretch on this path and it
                // ends with nothing to spend, so the projections behind it
                // need the stream back at the top of the chain.
                if (boot_key != nullptr && refresh.after_attention_norm)
                {
                    bootstrap(normed, "block.refresh_attention_norm", *boot_key,
                              relin_key);
                    note_depth("block.refresh_attention_norm", normed.column);
                }

                RectActivation sublayer =
                    attention(normed, *attention_weights, config.attention,
                              galois_key, relin_key, boot_key, plan);
                normed.column.clear();
                // The two operands have been through completely different
                // circuits, so neither the level nor the scale lines up; this
                // is the one level a pre-norm residual pays.
                stream.column = batch_.arith().residual_add(stream.column,
                                                            sublayer.column);
                note_depth("block.attention_residual", stream.column);
            }

            if (boot_key != nullptr && refresh.mid)
            {
                bootstrap(stream, "block.refresh_mid", *boot_key, relin_key);
                note_depth("block.refresh_mid", stream.column);
            }

            {
                Range _r_half("transformer_block.feed_forward_half");
                RectActivation normed =
                    rms_norm(stream, feed_forward_gain,
                             config.feed_forward_norm, galois_key, relin_key,
                             boot_key);
                note_depth("block.feed_forward_norm", normed.column);

                if (boot_key != nullptr && refresh.after_feed_forward_norm)
                {
                    bootstrap(normed, "block.refresh_feed_forward_norm",
                              *boot_key, relin_key);
                    note_depth("block.refresh_feed_forward_norm",
                               normed.column);
                }

                RectActivation sublayer = feed_forward(
                    normed, *feed_forward_weights, config.feed_forward,
                    galois_key, relin_key, boot_key, plan);
                normed.column.clear();
                stream.column = batch_.arith().residual_add(stream.column,
                                                            sublayer.column);
                note_depth("block.feed_forward_residual", stream.column);
            }

            return stream;
        }

        // -------------------------------------------------------------------
        // Host-side staging
        // -------------------------------------------------------------------

        RectActivation
        Llama3RectOperator::encrypt(const std::vector<double>& x, int channels,
                                    HEEncryptor<Scheme::CKKS>& encryptor,
                                    double scale)
        {
            const int d = layout_.d;
            const int k = layout_.k;
            const int step = layout_.batch;
            const int half = layout_.N / 2;

            if (x.size() != static_cast<size_t>(d) *
                                static_cast<size_t>(channels))
            {
                throw std::invalid_argument(
                    "A rectangular activation is layout.d rows by channels "
                    "columns, row major");
            }

            const int groups = groups_for(channels);

            RectActivation out;
            out.rows = d;
            out.groups = groups;
            out.channels = channels;
            out.column.reserve(static_cast<size_t>(groups) * d);

            for (int g = 0; g < groups; ++g)
            {
                std::vector<int64_t> coeffs(
                    static_cast<size_t>(d) * d * k, 0);
                for (int i = 0; i < d; ++i)
                {
                    for (int j = 0; j < d; ++j)
                    {
                        int64_t* e =
                            coeffs.data() +
                            (static_cast<size_t>(i) * d + j) * k;
                        for (int t = 0; t < step; ++t)
                        {
                            const int c = g * half + t * d + j;
                            if (c >= channels)
                            {
                                continue;
                            }
                            e[t] = quantise(
                                x[static_cast<size_t>(i) * channels + c],
                                scale);
                        }
                    }
                }

                std::vector<std::vector<int64_t>> columns;
                build_matrix_encryption_coefficients(coeffs, layout_, d, d,
                                                     columns);
                for (int j = 0; j < d; ++j)
                {
                    Plaintext<Scheme::CKKS> plain(context_);
                    batch_.matrix().load_coefficients(plain, columns[j], scale);
                    Ciphertext<Scheme::CKKS> c(context_);
                    encryptor.encrypt(c, plain);
                    out.column.push_back(std::move(c));
                }
            }
            return out;
        }

        std::vector<double>
        Llama3RectOperator::decrypt(RectActivation& in,
                                    HEDecryptor<Scheme::CKKS>& decryptor,
                                    double scale)
        {
            const int d = layout_.d;
            const int step = layout_.batch;
            const int half = layout_.N / 2;
            const int channels = in.channels;

            std::vector<double> out(
                static_cast<size_t>(d) * channels, 0.0);

            for (int g = 0; g < in.groups; ++g)
            {
                for (int j = 0; j < d; ++j)
                {
                    Plaintext<Scheme::CKKS> plain(context_);
                    decryptor.decrypt(
                        plain, in.column[static_cast<size_t>(g) * d + j]);
                    std::vector<int64_t> coeffs;
                    batch_.matrix().extract_coefficients(coeffs, plain);

                    for (int t = 0; t < step; ++t)
                    {
                        const int c = g * half + t * d + j;
                        if (c >= channels)
                        {
                            continue;
                        }
                        for (int i = 0; i < d; ++i)
                        {
                            out[static_cast<size_t>(i) * channels + c] =
                                static_cast<double>(
                                    coeffs[static_cast<size_t>(i) + d * t]) /
                                scale;
                        }
                    }
                }
            }
            return out;
        }

        // -------------------------------------------------------------------
        // Small helpers
        // -------------------------------------------------------------------

        double Llama3RectOperator::rescale_prime(
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
        Llama3RectOperator::encode(const std::vector<Complex64>& values,
                                   double scale, int depth)
        {
            Plaintext<Scheme::CKKS> plain(context_);
            encoder_.encode(plain, values, scale);
            for (int i = 0; i < depth; i++)
            {
                batch_.arith().mod_drop_inplace(plain);
            }
            return plain;
        }

        void Llama3RectOperator::require_uniform(const RectActivation& in,
                                                 const char* name) const
        {
            if (in.empty())
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " operand has no columns, so there is no such activation");
            }
            if (in.column.size() !=
                static_cast<size_t>(in.groups) *
                    static_cast<size_t>(layout_.d))
            {
                throw std::invalid_argument(
                    std::string("The ") + name +
                    " operand does not hold layout.d ciphertexts per group");
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
