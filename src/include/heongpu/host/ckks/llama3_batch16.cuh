// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The Llama-3 NON-LINEAR layers for a BATCH OF SIXTEEN INDEPENDENT INPUTS on
// Kang's Algorithm 1 encoding -- RMSNorm, the reshape, and SwiGLU. Not the
// SoftMax, and not the products: those are other modules and other sessions.
//
// WHY THIS FILE EXISTS
// --------------------
// llama3_rect.cuh spends the batch axis of the matrix encryption on ONE input,
// because Algorithm 5 contracts over it. That is the right answer for a single
// user and it is why that path exists. It is the wrong answer for sixteen: the
// axis is already carrying something, so a second input has nowhere to go and
// costs a second copy of everything.
//
// llama3_batch.cuh spends the same axis the other way -- k/2 packed matrices
// are k/2 INDEPENDENT INPUTS -- and its own header says so: "That is pure
// throughput and it is the right answer when there are k/2 users." Sixteen
// users is exactly that case, and it pins the ring:
//
//     batch = k/2 = 16   =>   k = 32   =>   d = N/32
//     head_dim must be a multiple of d, and Llama-3-8B's head dim is 128
//     d = 128 tokens per block  =>  N = 4096
//
// So this module is the non-linear half of a batch-16 run at N = 4096, d = 128,
// k = 32, and the 2048 slots of one ciphertext are exactly 16 instances x 128
// tokens.
//
// THE LAYOUT, AND WHY THE NON-LINEAR LAYERS WANT NO OTHER ONE
// -----------------------------------------------------------
// The invariant this module is built on, published by the sessions that own
// the products and confirmed from build_bridge_tables rather than from a
// comment (llama3_batch.cu:199-239, and :216 which literally writes
// `const int slot = b + u * step` with `step = k/2`):
//
//     slot b + (k/2)*u of the slot-form ciphertext for column j
//         holds entry (u, j) of batch instance b
//
//     b  batch instance, in [0, k/2), on the FAST slot axis
//     u  token, in [0, d), on the slow axis at stride k/2
//     j  channel -- THE CIPHERTEXT INDEX, not a slot index
//
// That map is a bijection onto all N/2 slots: b + u*(k/2) for b < k/2, u < d
// covers [0, d*(k/2)) = [0, N/2) exactly once. There is no dead slot, so
// nothing here has to mask one off.
//
// The consequence is the whole of the layout answer, and it is short: BECAUSE A
// CHANNEL IS A WHOLE CIPHERTEXT, EVERY NON-LINEAR LAYER BELOW IS SLOT-WISE AND
// SPENDS NO ROTATION AT ALL.
//
//   - RMSNorm reduces over the channel axis, and the channel axis is the
//     ciphertext index, so the reduction is a slot-wise ADDITION of d_model
//     ciphertexts. No sum_blocked, no mask, no fan-out, no Galois key. On the
//     Algorithm-5 path the same reduction is a blocked span of k/2 costing
//     2*log2(k/2) key switches and the level a mask costs; here it costs
//     nothing. Llama3Operator::rms_norm reaches that behaviour at count = 1,
//     where its own reduction loop does not execute.
//   - The learned gain is one number per channel and a channel is one
//     ciphertext, so the gain is a CONSTANT, not a slot vector -- and a
//     constant folds into the projection that reads the normalised stream,
//     exactly, on the host, for free. See fold_gain.
//   - The reshape (token, d_model) <-> (token, head, head_dim) is host-side
//     index arithmetic on ciphertext handles and costs NOTHING homomorphically.
//     Grouped-query attention is index REUSE, not weight expansion.
//   - SwiGLU is slot-wise per ciphertext and never looks at an index.
//
// So the layout the non-linear parts have to use is the layout the products
// already agreed on, and the seam question the user asked answers itself: the
// non-linear layers consume exactly what the previous phase produces and
// produce exactly what the next phase consumes, in the same encoding, with no
// layout conversion of their own.
//
// WHAT THEY DO COST, STATED UP FRONT
// ----------------------------------
// One thing, and naming it is the point of this comment. A Hadamard product
// and a polynomial fit are slot-wise, and the matrix encryption is a
// COEFFICIENT encoding -- multiplying two columns convolves them. So every
// non-linearity crosses the row bridge to slot form and back, and THE BRIDGE IS
// THE ENTIRE COST OF THIS MODULE.
//
// Per column, at N = 4096 and d = 128 (llama3_batch.cu:244 with baby_steps()
// at :504): 22 key switches, d = 128 plaintext products, 8 rescales, ONE
// level. Not d - 1 = 127 -- the doc comment on to_slots still says that and it
// is stale by 5.8x, because bridge() calls baby_steps() unconditionally and
// baby_steps() returns the balanced BSGS split whenever bridge_baby_steps_ is
// 0, which is the default. BSGS IS ALREADY ON. Turning it off takes an
// explicit set_bridge_baby_steps(1).
//
// This module bridges 2 * 2 * d_model + 3 * hidden = 59,392 columns per block
// at the 8B shape; the attention seam adds 8,192 more, which belongs to
// another module. Bridging is 98.3% of every Galois rotation in a block, and
// the SwiGLU's three legs are 64% of the bridging -- so a lever that does not
// touch 3 * hidden_channels is working on a third of the problem.
//
// That cost is not a layout mistake to be designed away; it is what a
// non-linearity costs in a coefficient encoding, and it is why this path is
// still the cheap one -- the three projections it replaces were the widest key
// switching in the block. What is available is to make the bridge cheaper
// rather than rarer:
//
//   set_hoisted_crossings   OFF by default and the one real lever left. The 15
//                           baby shifts share one key-switch decomposition
//                           instead of paying 15, so decompositions per column
//                           go 22 -> 8 and the multiply-accumulate launches go
//                           255 -> 8. Needs KEYSWITCHING_METHOD_II; method I
//                           rebuilds the decomposition inside the shift loop
//                           and gains nothing.
//   set_bridge_baby_steps   already taken by the default. n1 = 16, n2 = 8 at
//                           d = 128 -- note the header there claims n1 <= n2,
//                           which is false for every d that is an odd power of
//                           two. The count n1 + n2 - 2 is the same either way.
//   set_bridge_plain_capacity  one block visits 8 distinct (direction, depth,
//                           prime) sets and never has more than 2 hot, so the
//                           default 4 does not thrash.
//   set_bridge_plain_limb_limit  DO NOT USE ON THIS PATH. A rejected set is
//                           re-encoded inside the per-COLUMN loop, so a 4096
//                           column bridge does 524,288 encodes instead of 128.
//                           It was written for the two-ring driver, where the
//                           bridges are narrow.
//
// use_fast_bridge() turns on the one that is off and sizes the cache; it does
// not touch the limb limit. This module never enables anything behind a
// caller's back, so an unchanged measurement stays unchanged.
//
// MEMORY, which is the other reason this module exists. Llama3BatchOperator::
// rms_norm holds the slot-form copy alive across the return crossing, so at the
// 8B batch-16 shape its frame peaks at 20,480 resident ciphertexts -- 87.5 GiB
// at the profile's 70 limbs, over an 80 GiB A100 before any key material. This
// module releases the slot copy the moment the slot core returns, which is
// 4096 ciphertexts and 17.5 GiB, exactly as feed_forward already does for its
// own branches.
//
// WHERE THE LEVELS WENT
// ---------------------
// Llama3BatchOperator::rms_norm and ::feed_forward are correct and are the
// reference this module is checked against. They also leave levels on the
// floor, because the config knobs the Algorithm-5 path grew were never wired
// through to this one. Every saving below is EXACT -- a host-side rescaling of
// a plaintext, not an approximation traded for depth:
//
//   newton_iterations 2 -> 0        -6 levels   a step is y^2, x/2 * it, and
//                                               the product: three levels each,
//                                               and the fit alone already meets
//                                               12 bits over a calibrated range
//   fold_mean_into_fit              -1 level    1/sqrt(s/C + eps) over the summed
//                                               square is the same value from the
//                                               same ciphertext as 1/sqrt(m + eps)
//                                               over the mean, and skips the
//                                               plaintext product that forms m
//   gain folded into the weight     -1 level    W^T diag(g) y = (diag(g) W)^T y,
//                     per norm      and d_model exactly -- and it also removes one
//                                   plaintext  full-slot plaintext encode per
//                                   encodes    channel, 4096 of them per norm
//   fold_silu_domain_into_gate      -1 level    the fit's argument reaches the SiLU
//                                               through a projection whose weight is
//                                               a host array, so 1/bound rides there
//                                               instead of on a homomorphic product
//
// Eight levels a block, for four host-side multiplications. The defaults here
// take all four; set them false to reproduce Llama3BatchOperator's schedule
// exactly.
//
// WHAT THIS MODULE DELIBERATELY DOES NOT DO
// -----------------------------------------
// It does not touch Llama3BatchOperator. Three other sessions own functions in
// llama3_batch.cu -- the Algorithm-1 projection, the Algorithm-4 products and
// the attention sublayer, and the SoftMax seam -- so this is a separate class
// composed on theirs by reference, exactly as Llama3RectOperator composes on
// Llama3BatchOperator. Nothing here changes an existing measurement.
//
// It does not implement the SoftMax, the projections, the CCMM, the attention
// sublayer, or a whole transformer block: a block driver composes rms_norm,
// somebody else's attention(), rms_norm again and feed_forward, with the
// residual adds Llama3Operator::residual_add already provides.

#ifndef HEONGPU_CKKS_LLAMA3_BATCH16_H
#define HEONGPU_CKKS_LLAMA3_BATCH16_H

#include <heongpu/host/ckks/llama3.cuh>
#include <heongpu/host/ckks/llama3_batch.cuh>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief The shape of one batch-16 Llama-3 run.
         *
         * Everything here is a MODEL dimension. The ring dimensions -- N, d, k
         * and the instance count k/2 -- come from the operator's
         * BatchMatrixLayout and are checked against these, not restated.
         */
        struct Batch16Shape
        {
            /// Independent inputs carried at once. Must equal layout.batch,
            /// which is k/2; 16 at N = 4096 with d = 128.
            int instances = 16;
            /// Residual stream width. 4096 for Llama-3-8B.
            int d_model = 0;
            /// SwiGLU hidden width. 14336 for Llama-3-8B.
            int hidden = 0;
            /// Query heads. 32 for Llama-3-8B.
            int heads = 0;
            /// Distinct key/value heads for grouped-query attention; 0 means
            /// one apiece. Must divide @c heads. 8 for Llama-3-8B.
            int kv_heads = 0;
            /// Channels in one head. Must be a positive multiple of layout.d,
            /// because Algorithm 4's operands are square at d. 128 for
            /// Llama-3-8B, which is exactly d at N = 4096.
            int head_dim = 0;

            /// Channels the query projection produces, heads * head_dim.
            int q_channels() const { return heads * head_dim; }
            /// Channels each of the key and value projections produces.
            /// Narrower than the query by heads/kv_heads under GQA, and that
            /// narrowing is the whole saving -- it is NOT undone by widening
            /// the weights, see key_value_head_for_query().
            int kv_channels() const
            {
                return (kv_heads > 0 ? kv_heads : heads) * head_dim;
            }
            /// Query heads sharing one key/value head.
            int group() const
            {
                return kv_heads > 0 ? heads / kv_heads : 1;
            }
        };

        /**
         * @brief The non-linear layers of a batch-16 Llama-3 block.
         *
         * Composed on Llama3BatchOperator by reference: that class already owns
         * the row bridge, the batch matrix operator, the slot-form arithmetic
         * operator and every twiddle cache, and a second copy of all four would
         * cost memory for nothing. It also belongs to other sessions, and this
         * one edits none of it.
         */
        class Llama3Batch16Operator
        {
          public:
            /**
             * @param batch The operator carrying the ring, the bridge and the
             *              slot arithmetic. Must outlive this object.
             * @param shape The model dimensions. Validated against the ring at
             *              construction, because every shape error this
             *              encoding can make is silent otherwise.
             *
             * @throws std::invalid_argument if the shape does not fit the ring:
             *         instances != k/2, head_dim not a positive multiple of d,
             *         kv_heads not dividing heads, or a non-positive width.
             */
            Llama3Batch16Operator(Llama3BatchOperator& batch,
                                  const Batch16Shape& shape);

            const Batch16Shape& shape() const noexcept { return shape_; }
            const BatchMatrixLayout& layout() const noexcept
            {
                return batch_.layout();
            }
            Llama3BatchOperator& batch() noexcept { return batch_; }
            Llama3Operator& arith() noexcept { return batch_.arith(); }

            /** @brief Independent inputs carried at once, k/2. */
            int instances() const noexcept { return batch_.batch(); }
            /** @brief Tokens one activation carries, d -- also its rows. */
            int tokens() const noexcept { return batch_.layout().d; }
            /** @brief Slots in one ciphertext, N/2 = instances * tokens. */
            int slot_count() const noexcept { return slot_count_; }

            // ---------------------------------------------------------------
            // The layout, as index arithmetic
            // ---------------------------------------------------------------

            /**
             * @brief The slot holding (instance, token) of any slot-form
             *        ciphertext.
             *
             * This is the published invariant, in one place, so that nothing
             * downstream has to rederive `b + (k/2)*u` and get it wrong. Every
             * plaintext vector this module builds goes through here.
             */
            int slot_of(int instance, int token) const;

            /** @brief A contiguous run of channels, i.e. of ciphertexts. */
            struct ChannelRange
            {
                int base = 0;
                int count = 0;

                int end() const { return base + count; }
                bool empty() const { return count <= 0; }
            };

            // ---------------------------------------------------------------
            // The reshape
            // ---------------------------------------------------------------
            //
            // In a plaintext transformer this is where (token, d_model) becomes
            // (token, head, head_dim) and back. Here it is not an operation at
            // all: a channel is a ciphertext, so head h, head-dim lane c is
            // ciphertext h*head_dim + c, and "reshaping" is choosing which
            // handles to pass. These entry points exist so that the choice is
            // made once, named, and asserted by a test -- not so that anything
            // is computed.
            //
            // Two things this reshape is NOT, and both would be silent:
            //
            //   - it is not the K transpose. That is Algorithm 3's CMT, d - 1
            //     key switches per channel block and zero levels, and it
            //     belongs to the session that owns the products.
            //   - it is not a TOKEN reshape, and there is no such thing on
            //     this path. A BatchActivation always has exactly d rows;
            //     llama3_batch.* has no token_blocks concept and attention()
            //     takes no token-block argument, so a sequence longer than
            //     d = 128 tokens has no code path here at all. The slot path
            //     in llama3.cu does have that machinery, so the gap is an
            //     omission rather than a naming difference. Nothing below
            //     pretends otherwise.

            /** @brief Channels of query head @p h. */
            ChannelRange query_head(int h) const;

            /** @brief Channels of key/value head @p h. */
            ChannelRange key_value_head(int h) const;

            /**
             * @brief The key/value head that query head @p h reads.
             *
             * Grouped-query attention costs NOTHING on this encoding, and this
             * function is why. A kv head is a run of ciphertexts, and several
             * query heads can be handed the same run -- so GQA is index reuse.
             * The Algorithm-5 path has to widen the key and value weights on
             * the host to heads/kv_heads times their size, because there a kv
             * head has to be materialised into every batch slot that reads it;
             * here nothing is materialised and the key and value projections
             * stay narrow by the full factor of 4 at Llama-3-8B's 32 over 8.
             */
            ChannelRange key_value_head_for_query(int h) const;

            /** @brief The channel holding lane @p lane of head @p head. */
            int channel_of(int head, int lane) const;

            /** @brief The head and lane a channel belongs to. */
            struct HeadLane
            {
                int head = 0;
                int lane = 0;
            };
            HeadLane head_of(int channel) const;

            /**
             * @brief Borrow one head's columns as pointers, without copying.
             *
             * A BatchActivation owns its ciphertexts, so slicing one by value
             * would deep copy d ciphertexts to name them. The products take
             * their operands by pointer for exactly this reason.
             */
            static std::vector<Ciphertext<Scheme::CKKS>*>
            columns(BatchActivation& x, ChannelRange range);

            /**
             * @brief Check an activation is the shape this module expects.
             *
             * @param channels Columns it must carry.
             *
             * @throws std::invalid_argument naming @p name if the row count is
             *         not d, the column count is not @p channels, or the
             *         columns have drifted apart in level or scale -- the last
             *         of which is silent everywhere else and turns into a
             *         wrong sum inside a reduction.
             */
            void validate(const BatchActivation& x, int channels,
                          const char* name) const;

            // ---------------------------------------------------------------
            // RMSNorm
            // ---------------------------------------------------------------

            /** @brief Shape and approximation settings for rms_norm. */
            struct RMSNormConfig
            {
                double eps = 1e-5;
                /// Range of the SUMMED square, over which the 1/sqrt is
                /// fitted. Calibration data, as in Section 4.3: a range is a
                /// measurement of the model and not a property of the
                /// algorithm, and fitting over a worst-case bound instead is
                /// what made the SoftMax reciprocal wrong by 98.6% on the
                /// other path.
                double sum_lo = 0.0;
                double sum_hi = 0.0;
                /// Degree of the 1/sqrt fit. 15 meets 12 bits over a
                /// calibrated range in four levels; the 31 the other paths
                /// default to buys precision nothing downstream can read.
                int degree = 15;
                /// Newton refinement steps. ZERO here, against 2 on
                /// Llama3BatchOperator: a step is three levels -- y^2, x/2
                /// times it, and the final product -- and the fit alone
                /// already meets the requirement. Raise it only to reproduce
                /// the old schedule.
                int newton_iterations = 0;
                /// Fit 1/sqrt over the summed square rather than over the
                /// mean, which saves the plaintext product and rescale that
                /// forming the mean costs. Ignored when @c newton_iterations
                /// is above zero, because a Newton step refines against the
                /// mean itself.
                bool fold_mean_into_fit = true;
                /// Multiplied into the fitted 1/sqrt: the norm hands back
                /// output_scale * x / rms(x) for no extra level, because a
                /// Chebyshev coefficient is a host-side number. Needs
                /// @c newton_iterations = 0 when it is not one.
                double output_scale = 1.0;
                /// The summed square arrives already multiplied by the fit's
                /// domain scale, so the fit skips the plaintext product that
                /// maps it onto [-1, 1] -- one level.
                ///
                /// This is only reachable when the caller controls what
                /// produced the stream, since the factor has to ride on a
                /// plaintext upstream and the sum of squares is quadratic in
                /// it: scaling the input by c scales the sum by c*c, so the
                /// caller wants c = sqrt(domain_scale(sum_lo, sum_hi)) and
                /// must set @c output_scale to 1/c to take it back out.
                /// OFF by default because a caller who folds one of the two
                /// and not the other gets a silently wrong answer, not an
                /// error. sum_pre_scale_factor() computes c.
                ///
                /// Needs @c fold_mean_into_fit and @c newton_iterations = 0:
                /// forming the mean would multiply the pre-scaling by 1/C and
                /// a Newton step refines against the unmapped sum. This path
                /// runs the norm's circuit explicitly rather than through
                /// Llama3Operator::rms_norm, which reaches the pre-scaled fit
                /// only through a mask this encoding has no reason to build --
                /// the reduction here is a slot-wise addition, so there is no
                /// plaintext product between it and the fit for the domain
                /// map to ride on. That is the one level this encoding cannot
                /// fold away by itself, and this is how a caller who owns the
                /// upstream weight folds it away anyway.
                bool sum_pre_scaled = false;
                /// Refresh the summed square before fitting 1/sqrt over it --
                /// the narrow auxiliary track, and on THIS encoding it is a
                /// better trade than anywhere else in the project.
                ///
                /// The channel axis here is the ciphertext index, so the
                /// reduction is a slot-wise addition of `channels`
                /// ciphertexts into exactly ONE. So the refresh is a single
                /// bootstrap, whatever the model's width, and it moves the
                /// whole fit -- the domain map, the ceil(log2(degree+1))
                /// levels of Chebyshev, and the rescale -- off the
                /// `d_model`-wide residual track and onto that one
                /// ciphertext. What the wide track keeps is the final
                /// product.
                ///
                /// OFF by default, and not only for reproducibility: it is a
                /// LEVELS-for-PRECISION trade, measured. At a 40-limb chain
                /// with the norm entering at depth 30 it hands back **six
                /// levels** and costs **14-32x in worst-slot error, run to
                /// run** -- 6.4e-04 for the wide track against 9.1e-03,
                /// 1.5e-02 and 2.0e-02 on three runs, against a host
                /// reference. That is roughly a decimal digit, it is the v1
                /// bootstrap's own precision rather than the fit's, and it is
                /// VARIABLE because a bootstrap is randomised. Take it where
                /// levels bind and not otherwise.
                ///
                /// The saving is also CONDITIONAL, in the direction opposite
                /// to the intuition. A bootstrap returns its ciphertext to
                /// @c refresh_levels whatever depth it went in at, so
                /// refreshing at depth D buys `D - refresh_levels`: a gain on
                /// a deep stream and a LOSS on a fresh one. Measured from a
                /// fresh stream at the same shape: depth 9 without, 31 with.
                ///
                /// Needs @c newton_iterations = 0 (a Newton step refines
                /// against the unmapped argument, and a refreshed sum arrives
                /// mapped) and is refused together with @c sum_pre_scaled,
                /// which runs its own circuit and would ignore this silently.
                bool refresh_sum = false;
            };

            /**
             * @brief The (c, output_scale) pair @c sum_pre_scaled needs.
             *
             * @return c, the factor to multiply into the plaintext weight that
             *         produces this norm's input. Set
             *         @c RMSNormConfig::output_scale to 1/c.
             */
            static double sum_pre_scale_factor(double sum_lo, double sum_hi);

            /**
             * @brief RMSNorm over the channel axis.
             *
             * The channel axis IS the ciphertext axis, so the mean is a
             * slot-wise addition of the squares and costs no rotation and no
             * level -- the one place this encoding is strictly better than the
             * rectangular one, which pays a masked blocked reduction for the
             * same sum.
             *
             * @param gain One learned scale per channel, or EMPTY. Empty is
             *             the fast path and the intended one: fold_gain() puts
             *             the scale into the projection weight that reads the
             *             normalised stream, exactly and on the host, saving a
             *             level and one full-slot plaintext encode per channel.
             *             Passing it here applies it homomorphically instead
             *             and costs both.
             *
             * Levels, with the defaults above and an empty @p gain:
             * square 1 + domain map 1 + fit ceil(log2(degree+1)) + product 1.
             * At degree 15 that is 7; Llama3BatchOperator's defaults spend 15
             * for the same answer.
             */
            /// @param boot_key Needed only by RMSNormConfig::refresh_sum, and
            ///        required when it is set -- checked before any work.
            BatchActivation rms_norm(BatchActivation& x,
                                     const std::vector<double>& gain,
                                     const RMSNormConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key,
                                     Galoiskey<Scheme::CKKS>* boot_key
                                     = nullptr);

            /**
             * @brief The same norm on a stream that is ALREADY in slot form.
             *
             * No crossing either way, and therefore no rotation and no Galois
             * key used at all -- the reduction is a slot-wise addition and
             * every other step is slot-wise too. @p galois_key is taken only
             * because the slot core's signature demands one; at count = 1 its
             * reduction loop does not execute and the key is never touched.
             *
             * This is the entry point a caller wants when the stream is held
             * in slot form across the block. Whether that is legal at all
             * turns on whether Algorithm 1's projection commutes with the
             * bridge -- see FeedForwardConfig::slot_resident.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            rms_norm_slots(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                           const std::vector<double>& gain,
                           const RMSNormConfig& config,
                           Galoiskey<Scheme::CKKS>& galois_key,
                           Relinkey<Scheme::CKKS>& relin_key,
                           Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            /**
             * @brief Fold a per-channel scale into a projection weight, on the
             *        host.
             *
             * A pre-norm block feeds its norm output to nothing but a
             * projection, and the learned scale is diagonal, so
             * W^T diag(g) y = (diag(g) W)^T y EXACTLY. On the host that is a
             * row scaling of a plaintext and free; homomorphically it is a
             * plaintext product and a rescale, once per channel per norm.
             *
             * @param weight Row major @p in_channels x @p out_channels, the
             *               transpose of the mathematical weight, as project()
             *               takes it. Row i is scaled by @p gain[i], which is
             *               the channel that row reads.
             *
             * The only cost is that the folded weight is a second copy of the
             * weight, which is host memory and not device memory.
             */
            static void fold_gain(std::vector<double>& weight,
                                  const std::vector<double>& gain,
                                  int in_channels, int out_channels);

            // ---------------------------------------------------------------
            // Rotary position embedding
            // ---------------------------------------------------------------
            //
            // RoPE is absent from the Algorithm-1 path entirely -- not
            // configured off, ABSENT: llama3_batch.cu never mentions it and
            // BatchAttentionConfig has no field for it. It is a non-linearity
            // of the position rather than of the value, and it belongs to
            // whoever owns the layer, so it is provided here as a free
            // function on slot-form ciphertexts and called by nobody in this
            // module.
            //
            // It is CHEAP on this encoding, and for the same two reasons
            // everything else here is:
            //
            //   - the head-dim pairing c <-> c + head_dim/2 is a pairing of
            //     whole CIPHERTEXTS, because a channel is a ciphertext. No
            //     homomorphic rotation, no Galois key.
            //   - the angle (u + offset) * theta^(-2c/head_dim) depends on the
            //     TOKEN, which is the slow slot axis at stride k/2 -- so it is
            //     an ordinary slot plaintext, and the SAME one serves every
            //     instance, every head and every ciphertext of a lane pair.
            //
            // Two plaintext products and one addition per output ciphertext,
            // one level. The rectangular path pays the same level but has to
            // insert it at a crossing's slot midpoint to get there; here the
            // stream is already in slot form when Q and K are formed.

            /** @brief Rotary embedding settings. */
            struct RopeConfig
            {
                /// Llama-3's base is 500000; Llama-2's is 10000.
                double theta = 500000.0;
                /// Absolute position of token 0, for a windowed run.
                int position_offset = 0;
            };

            /**
             * @brief RoPE in place on slot-form Q or K.
             *
             * @param slots `heads * head_dim` ciphertexts for Q, or
             *              `kv_heads * head_dim` for K, in the slot reading,
             *              at one level and one scale. Channel
             *              `h*head_dim + c` is head h, head-dim lane c.
             *
             * One level, four plaintext products and two additions per lane
             * pair. Applies to Q and K only -- V is not rotated.
             *
             * KNOWN COST, stated rather than hidden. The three slot vectors a
             * lane needs are built once per lane but applied through
             * multiply_vector, which ENCODES on every call -- so the same
             * plaintext is re-encoded once per head. At head_dim = 128 and 32
             * heads that is 4 * 64 * 32 = 8,192 encodes for Q where 192 would
             * do, a 42x redundancy, and it is the same defect the SiLU's
             * domain map has when its fold is off. The fix is to hold an
             * HEEncoder and encode each lane's three vectors once; it needs a
             * constructor change, so it is named here rather than done
             * quietly. The ARITHMETIC is unaffected.
             */
            void rope_slots(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                            const RopeConfig& config);

            // ---------------------------------------------------------------
            // SwiGLU
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one SwiGLU sublayer. */
            struct FeedForwardWeights
            {
                /// Row major in_channels x hidden_channels, as project() takes
                /// them; down is hidden_channels x in_channels.
                std::vector<double> gate;
                std::vector<double> up;
                std::vector<double> down;
            };

            /** @brief Shape and approximation settings for feed_forward. */
            struct FeedForwardConfig
            {
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;     ///< Section 3.1.3.
                /// Carry the domain map of the SiLU fit, 1/silu_bound, on the
                /// gate weight. The gate projection feeds the SiLU and nothing
                /// else, so the factor needs no undoing, and a host scaling is
                /// free where the map is otherwise the one plaintext product
                /// the fit pays before its series. One level.
                bool fold_silu_domain_into_gate = true;
                /// Hidden channels held at once; 0 takes the whole width.
                ///
                /// This is the memory lever and, at Llama-3 widths, the
                /// difference between fitting on one card and not. A matrix
                /// encryption spends one ciphertext per channel and the
                /// Hadamard product needs the gate and the up projection in
                /// BOTH forms at once, so the whole hidden width is
                /// 4 * hidden ciphertexts -- 57,344 at 14336. The down
                /// projection sums over the hidden axis and a sum splits over
                /// disjoint ranges of its index, so a chunk is projected down
                /// and accumulated the moment it is formed and then released:
                /// same products, same scales, same levels, only the order of
                /// a homomorphic sum, and the peak falls to 4 * hidden_block.
                int hidden_block = 0;
                /// Cross ONCE around the whole sublayer instead of three
                /// times over the hidden width.
                ///
                /// The three crossings exist because project() is assumed to
                /// need the coefficient encoding. It does not. Algorithm 1
                /// encodes its weight through BatchMatrixEncoder, and the
                /// weight is ONE REAL MATRIX SHARED BY ALL SIXTEEN INSTANCES,
                /// so its R_k image is the CONSTANT polynomial -- and
                /// multiplying by a constant of R_k is scalar multiplication.
                /// The projection is therefore
                /// out_c = sum_j W[j][c] * ct_j, a scalar multiply-accumulate
                /// ACROSS ciphertexts, which cannot care what a ciphertext
                /// encodes; and the bridge is linear WITHIN a column. Two such
                /// maps commute.
                ///
                /// So the SiLU and the gate product can meet the projections
                /// in slot form, and the sublayer crosses 2 * in_channels
                /// columns instead of 3 * hidden_channels: 8,192 instead of
                /// 43,008 at the 8B shape, and 946,176 key switches down to
                /// 180,224.
                ///
                /// IT DOES NOT SAVE A LEVEL, and the first version of this
                /// comment said it did. Three crossing CALLS are only two
                /// crossing LEVELS, because the gate and the up projection
                /// cross concurrently at the same depth -- so both
                /// arrangements spend exactly two, and the win is entirely in
                /// bridged columns. Measured: both land at depth 9.
                ///
                /// The commutation is asserted by
                /// CKKS_Llama3Batch16_ProjectionCommutesWithTheBridge, which
                /// passes. This is OFF by default anyway, so no existing
                /// measurement moves until a caller opts in. A caller holding
                /// the stream in slot form across the whole block should use
                /// feed_forward_slots() instead and pay no crossing here at
                /// all -- THAT is where the levels are, two of them per
                /// norm/SwiGLU pair.
                bool slot_resident = false;
                /// Refresh the hidden, in SLOT form, after the SiLU and the
                /// gate product and before the down projection.
                ///
                /// This seam cannot be driven from outside the sublayer, and
                /// that is why it is a flag here rather than one more entry in
                /// Batch16RefreshConfig's list of block seams: the SwiGLU half
                /// is the deepest stretch in the block at the paper's degree-31
                /// SiLU, and the point where it runs out is INSIDE this
                /// function. Refreshing at the sublayer's ends instead would
                /// leave the same stretch unsplit.
                ///
                /// It refreshes the WIDE track -- one bootstrap per hidden
                /// column of the chunk, so hidden_block is what bounds its
                /// cost -- and it is taken in slot form, which is where the
                /// hidden already is. Needs the boot key at the call.
                bool refresh_hidden = false;
            };

            /**
             * @brief W_down (SiLU(W_gate x) * W_up x).
             *
             * Three bridges over the hidden width -- the gate and the up
             * projection down to slots and the hidden back up -- because a
             * Hadamard product is what a coefficient encoding does not have.
             * Nothing about the batch instance enters: the SiLU and the gate
             * product are slot-wise and blind to every index, so sixteen
             * inputs cost exactly what one input costs per ciphertext, and the
             * ciphertext count does not depend on the batch at all.
             *
             * Levels: projection 1 + bridge 1 + SiLU ceil(log2(degree+1)) +
             * gate product 1 + bridge 1 + projection 1, with the SiLU's own
             * domain map folded away by default.
             */
            BatchActivation feed_forward(BatchActivation& x,
                                         const FeedForwardWeights& weights,
                                         const FeedForwardConfig& config,
                                         Galoiskey<Scheme::CKKS>& galois_key,
                                         Relinkey<Scheme::CKKS>& relin_key,
                                         Galoiskey<Scheme::CKKS>* boot_key =
                                             nullptr);

            /**
             * @brief The same SwiGLU on a stream that is already in slot form.
             *
             * NO CROSSING, NO ROTATION, AND NO GALOIS KEY. Algorithm 1 needs
             * none by construction -- it is two matrix products over R_{q,k}
             * and that is the whole reason the batch path exists -- and every
             * other step here is slot-wise. The only key switches the sublayer
             * performs are the relinearisations inside the SiLU's series and
             * the gate product.
             *
             * Legal for the same reason FeedForwardConfig::slot_resident is:
             * a batch-shared real weight is the constant polynomial of R_k, so
             * the projection is a scalar multiply-accumulate across
             * ciphertexts and does not read the encoding.
             *
             * @param slots in_channels ciphertexts in the slot reading, at one
             *              level and one scale.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            feed_forward_slots(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                               const FeedForwardWeights& weights,
                               const FeedForwardConfig& config,
                               Relinkey<Scheme::CKKS>& relin_key,
                               Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            // ---------------------------------------------------------------
            // The refresh
            // ---------------------------------------------------------------
            //
            // Until now this path had NO bootstrapping at all -- not
            // configured off, ABSENT: llama3_batch.cu and llama3_batch16.cu
            // between them did not contain the word, no entry point took a
            // boot key, and nothing could have supplied one. That is a
            // different kind of gap from a missing optimisation, because at
            // batch 16 the chain cannot carry a block:
            //
            //     batch = k/2 = 16  =>  k = 32, d = 128, N = 4096
            //     the 128-bit cap at N = 4096 is 109 bits of log QP
            //     41 + 33 + 33 = 107  =>  two Q primes and a special
            //                          =>  ONE usable level
            //
            // and one block spends around sixty. A batch-16 block without a
            // refresh is not slow, it is impossible. (At the library's demo
            // chains, where every measurement on this path was actually taken,
            // it merely runs ~23x outside 128-bit; LLAMA3_8B_LAYER_FLOW.md 24.)
            //
            // WHAT MAKES IT LEGAL, AND IT IS MEASURED RATHER THAN ARGUED.
            // Regular bootstrapping is ModRaise -> CoeffToSlot -> EvalMod ->
            // SlotToCoeff, whose net effect on the plaintext POLYNOMIAL is the
            // identity with the modulus restored, and it reads no encoding
            // tag. So it refreshes a Kang matrix encryption WHERE IT STANDS,
            // with no crossing in front of it. That was genuinely open,
            // because EvalMod's bound is on the plaintext COEFFICIENTS while
            // this encoding puts an inverse length-k DFT there --
            // test_ckks_batch_ringswitch.cpp's
            // RegularBootstrapCarriesAMatrixEncryption answers it at the real
            // island shape: 3.23e-05, 14.9 bits.
            //
            // It is also what decides the parameter set. A refresh that wanted
            // slot form would need a bridge in front of it, that bridge is a
            // level, the island would need three Q primes plus a special = 140
            // bits against a cap of 109, and batch 16 would have no legal
            // parameter set at all. It has one, by two bits.
            //
            // WHAT IT COSTS. A regular bootstrap spends CtoS + taylor + StoC +
            // 8 levels of whatever chain it is handed, so a chain of L limbs
            // hands back L - (that + 1) usable levels, and a schedule whose
            // worst stretch is S wants exactly L = S + that + 1. Longer is not
            // safer: it is slower, at dnum = 1 in proportion to L^2, for
            // levels the circuit discards at the next seam. refresh_levels()
            // and chain_limbs_for() are that arithmetic, so a caller can size
            // a chain without running a circuit.

            /**
             * @brief Refresh one ciphertext, whatever encoding it carries.
             *
             * Nothing here reads @c encoding_, and that is the point: it is
             * correct on a matrix encryption, on a slot ciphertext and on
             * anything else the polynomial happens to mean.
             *
             * @c generate_bootstrapping_params must have run on arith()
             * first -- on THIS operator's arithmetic half, since the
             * bootstrapping context is per HEArithmeticOperator instance and
             * not per HEContext -- and @p boot_key must hold
             * boot_rotation_indices(). A shift-vector Galois key asked for an
             * index it does not hold is undefined behaviour rather than an
             * error, which is why that union exists as a function.
             */
            Ciphertext<Scheme::CKKS>
            bootstrap(Ciphertext<Scheme::CKKS>& ct,
                      Galoiskey<Scheme::CKKS>& boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Refresh every ciphertext of a column set, in place. */
            void bootstrap(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                           const char* name,
                           Galoiskey<Scheme::CKKS>& boot_key,
                           Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Refresh a matrix encryption, in place. */
            void bootstrap(BatchActivation& x, const char* name,
                           Galoiskey<Scheme::CKKS>& boot_key,
                           Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief Which seams of a block take a refresh.
             *
             * Close to the seam set the rectangular path uses, and
             * deliberately so: the circuit is the same shape, and a schedule
             * comparable across the two paths is worth more than one tuned to
             * this one. Switching a seam off is what shows it was needed --
             * the stretch behind it runs out and the operation that wanted the
             * level throws, rather than returning noise.
             */
            struct Batch16RefreshConfig
            {
                /// Refresh at all. OFF by default, so every measurement taken
                /// before this existed -- which is all of them -- reproduces
                /// unchanged.
                bool enabled = false;
                /// The residual stream on the way in. False for the first
                /// block of a stack, whose input is already fresh; true is
                /// what joins one block to the next.
                bool entry = false;
                /// The normalised stream, before the projections read it.
                bool after_attention_norm = true;
                /// The attention sublayer's output, before the residual add.
                /// The sublayer's own internal refresh -- the SoftMax's narrow
                /// auxiliary track -- is
                /// BatchSoftmaxSeamConfig::refresh_denominator and is not
                /// duplicated here.
                bool after_attention = true;
                /// The residual stream between the two halves.
                bool mid = true;
                /// The normalised stream, before the gate and up projections.
                bool after_feed_forward_norm = true;
                /// The SwiGLU hidden, after the SiLU and the gate product.
                /// The SwiGLU half is the deepest stretch in the block at the
                /// paper's degree-31 SiLU, so it is this seam or a degree
                /// nobody fits a SiLU at.
                bool feed_forward_hidden = true;

                /// Refreshes one block takes with these flags.
                int count() const;

                /// The flag at seam position @p position, in the order of
                /// @c seam_position. Indexing rather than naming is what lets
                /// plan_refresh search the subsets without restating six
                /// field names, and what keeps the two in step if a seventh
                /// seam is ever added.
                bool at(int position) const;
                void set(int position, bool take);
            };

            /**
             * @brief The six positions a block can refresh at, in order.
             *
             * @c seam_feed_forward_hidden sits INSIDE the stretch that follows
             * @c seam_after_feed_forward_norm rather than after it, which is
             * why it is last in this list rather than fifth: the order here is
             * the order the stream reaches them, and that order is what
             * plan_refresh walks.
             */
            enum seam_position
            {
                seam_entry = 0,
                seam_after_attention_norm,
                seam_after_attention,
                seam_mid,
                seam_after_feed_forward_norm,
                seam_feed_forward_hidden,
                seam_count
            };

            /** @brief The trace name of a seam position. */
            static const char* seam_name(int position);

            /**
             * @brief What a refresh schedule has to fit inside.
             *
             * The stretches are a MEASUREMENT -- @c depth_trace produces them
             * and the fit degrees move them -- so they are an input here
             * rather than a table. That is the point: the subset that is
             * minimal at one set of degrees is not minimal at another, and
             * hard-coding an answer measured once is how a default becomes
             * wrong silently.
             */
            struct RefreshPlanInput
            {
                /// Levels spent in the stretch FOLLOWING each seam position,
                /// in seam_position order. Must have seam_count entries.
                std::vector<int> stretch;
                /// Ciphertexts refreshed at each position -- the residual
                /// width at five of them and the SwiGLU hidden width at
                /// seam_feed_forward_hidden. Empty counts seams instead of
                /// bootstraps, which understates the hidden seam by 3.5x at
                /// the 8B shape and is why the widths are here at all.
                std::vector<long long> width;
                /// The chain the schedule runs on, in limbs.
                int chain_limbs = 0;
                /// Levels a refresh spends on itself; refresh_levels().
                int refresh_levels = 0;
                /// Limbs the stream arrives with. 0 means chain_limbs, i.e.
                /// the first block of a stack, whose input is already fresh.
                /// Ignored when @c steady_state is set.
                int entry_limbs = 0;
                /// Plan a REPEATING block rather than the first one.
                ///
                /// This is the distinction that decides the answer, and it is
                /// easy to miss: a block handed a fresh chain has the whole of
                /// it to spend before its first refresh, while a block in the
                /// middle of a stack is handed whatever the previous block
                /// left. So the cheapest schedule for block 0 is not a legal
                /// schedule for block 1, and a plan measured on one block and
                /// applied to a stack fails at the second one -- with a
                /// level underflow deep inside a sublayer rather than at a
                /// seam.
                ///
                /// Under this flag the plan must have a FIXED POINT: the limbs
                /// it hands on must be at least the limbs it needs on the way
                /// in. That is what makes @c seam_entry earn its place.
                bool steady_state = false;
            };

            /** @brief The cheapest schedule that fits, and what it costs. */
            struct RefreshPlan
            {
                Batch16RefreshConfig config;
                /// False when NO subset fits -- which means the chain is too
                /// short for this schedule and no arrangement of seams will
                /// save it. Lengthening the chain or shortening a stretch are
                /// then the only moves, and a caller that ignores this flag
                /// discovers it by throwing at a seam instead.
                bool feasible = false;
                /// Seams taken.
                int refreshes = 0;
                /// Ciphertexts bootstrapped -- the weighted cost that was
                /// actually minimised.
                long long bootstraps = 0;
                /// The longest run of stretches between two refreshes under
                /// this plan; what chain_limbs_for() should be given.
                int worst_run = 0;
                /// Limbs the block hands on. Under @c steady_state this is
                /// also the number it may be handed, which is the fixed point.
                int exit_limbs = 0;
            };

            /**
             * @brief The cheapest set of seams that keeps every stretch inside
             *        the chain.
             *
             * Exhaustive over the 64 subsets, because six positions is small
             * enough that exact is cheaper than clever, and minimised on
             * BOOTSTRAPS rather than on seams: five of the positions are
             * d_model ciphertexts each and the sixth is @c hidden, so at the
             * 8B shape counting seams gets the answer wrong by 3.5x on the one
             * that matters most.
             *
             * This is host arithmetic over a measurement -- no GPU, no
             * context, no key -- which is what makes it usable BEFORE a run
             * rather than after one.
             *
             * @throws std::invalid_argument if @c stretch is not seam_count
             *         long, if @c width is neither empty nor seam_count long,
             *         if a stretch or width is negative, or if the chain or
             *         refresh cost is negative.
             */
            static RefreshPlan plan_refresh(const RefreshPlanInput& input);

            /**
             * @brief Levels one regular bootstrap spends on itself.
             *
             * CtoS + taylor + StoC + 8, read off the configuration rather than
             * hard coded, so a caller who changes the configuration gets an
             * answer instead of a stale constant. 25 at the default (3, 3, 11).
             */
            static int refresh_levels(const BootstrappingConfig& config);

            /**
             * @brief The chain a schedule wants, in limbs.
             *
             * @param worst_stretch The most levels spent between two
             *                      consecutive refreshes. @c depth_trace is
             *                      how that is measured; it does not follow
             *                      from the shape.
             *
             * One more than the stretch plus the refresh, because a bootstrap
             * is handed a ciphertext with one prime left.
             */
            static int chain_limbs_for(int worst_stretch,
                                       const BootstrappingConfig& config);

            /**
             * @brief Every rotation index THIS module's layers can ask for.
             *
             * The bridge's and the products', which are the same set -- both
             * are the multiples of k/2 -- and nothing else: the reductions
             * here are slot-wise additions and RoPE pairs whole ciphertexts,
             * so no layer in this module asks for an index of its own.
             */
            std::vector<int> rotation_indices() const;

            /**
             * @brief The union of that with the bootstrapping key indices.
             *
             * BUILD THE KEY FROM THIS, not from either half. A Galois key in
             * shift-vector form asked for an index it does not hold is
             * undefined behaviour and not an error, so a caller who generates
             * two keys and hands over the wrong one gets a wrong answer
             * silently. Sorted and deduplicated.
             *
             * @c generate_bootstrapping_params must have run on arith()
             * before this is called: the bootstrapping index list does not
             * exist until it has.
             */
            std::vector<int> boot_rotation_indices() const;

            /**
             * @brief Limbs to keep after the refresh at each named seam.
             *
             * A bootstrap hands back a fixed depth wherever it is taken, and a
             * stretch almost never wants all of it -- and the difference is
             * not free to hold. METHOD_II reads its digit count from
             * @c d_leveled[depth] and its RNS width from @c Q_prime_size -
             * depth, so an unspent limb is carried by every key switch until
             * the next refresh and then discarded. On this path that is
             * hundreds of thousands of key switches per block.
             *
             * Called with the seam's name once its refresh is done. Return the
             * limbs the stretch behind it needs -- one MORE than it spends,
             * because the next bootstrap wants a ciphertext with one prime
             * left. Return <= 0, or leave this empty, to keep whatever the
             * bootstrap handed back.
             *
             * Too small does not corrupt anything silently: the stretch runs
             * out and the operation that wanted the level throws.
             */
            std::function<int(const char* seam)> level_budget;

            // ---------------------------------------------------------------
            // The whole block
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one batch-16 transformer block. */
            struct TransformerBlockWeights
            {
                /// One learned scale per channel, or empty. With
                /// @c TransformerBlockConfig::fold_norm_scale these never
                /// reach the ciphertext at all -- they are multiplied into
                /// the projections that read the normalised stream, on the
                /// host, which is exact and saves a level per norm.
                std::vector<double> attention_norm;
                std::vector<double> feed_forward_norm;
                Llama3BatchOperator::BatchAttentionWeights attention;
                FeedForwardWeights feed_forward;
            };

            /** @brief Shape and approximation settings for one block. */
            struct TransformerBlockConfig
            {
                RMSNormConfig attention_norm;
                Llama3BatchOperator::BatchAttentionConfig attention;
                RMSNormConfig feed_forward_norm;
                FeedForwardConfig feed_forward;
                /// Fold the learned gains into the projections rather than
                /// applying them homomorphically. Exact, host-side, and worth
                /// a level and d_model plaintext encodes per norm.
                bool fold_norm_scale = true;
                /// Where the block refreshes. Disabled by default; a block
                /// with @c refresh.enabled needs a boot key at the call and
                /// generate_bootstrapping_params to have run on arith().
                Batch16RefreshConfig refresh;
            };

            /**
             * @brief One pre-norm transformer block on sixteen inputs.
             *
             * Norm, attention, residual; then norm, SwiGLU, residual. The
             * residual needs no crossing: it reconciles level and scale with a
             * mod drop and a multiplication by a CONSTANT, and a constant is
             * the constant polynomial, which scales every coefficient of a
             * matrix encryption exactly as it scales every slot of a slot
             * encoding. The stream is therefore in the same encoding at both
             * ends of the block and the only crossings are the ones the
             * non-linearities force.
             *
             * This differs from Llama3BatchOperator::transformer_block in what
             * it spends rather than in what it computes: the norms are this
             * module's, so the block takes the four exact level savings the
             * batch path never wired through, and releases the slot copy the
             * other one holds across its return crossing.
             *
             * The attention half is NOT reimplemented -- it is
             * Llama3BatchOperator::attention, whose seam belongs to another
             * module.
             */
            BatchActivation
            transformer_block(BatchActivation& x,
                              const TransformerBlockWeights& weights,
                              const TransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Relinkey<Scheme::CKKS>& relin_key,
                              Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            // ---------------------------------------------------------------
            // A sequence longer than d tokens
            // ---------------------------------------------------------------
            //
            // d IS the sequence length on this path, and at batch 16 the ring
            // pins it to 128. Llama-3-8B's context is 8192, so every
            // measurement this path has produced covered 1/64 of the model,
            // and there was no code path to the rest of it.
            // Llama3BatchOperator::attention_sequence is the m^2 attention
            // schedule; this is the block driver over it.
            //
            // The non-linear half needs nothing new, and it is worth saying
            // why rather than merely doing it: RMSNorm reduces over CHANNELS
            // and SwiGLU is slot-wise, so neither looks along the token axis
            // at all. A sequence is a loop over token blocks for both, with no
            // interaction between blocks and no new key material. Attention is
            // the only layer in a transformer block that couples tokens, which
            // is exactly why it is the only one that pays here.

            /** @brief A sequence as token blocks of d rows each. */
            struct Batch16Sequence
            {
                /// Block t holds tokens [t*d, (t+1)*d) of every one of the
                /// k/2 instances. All blocks at one level and one scale.
                std::vector<BatchActivation> block;

                int blocks() const { return static_cast<int>(block.size()); }
                bool empty() const { return block.empty(); }
                /// Channels each block carries.
                int columns() const
                {
                    return block.empty() ? 0 : block.front().columns();
                }
            };

            /**
             * @brief Check a sequence is the shape this module expects.
             *
             * @throws std::invalid_argument if it is empty, if a block is not
             *         d rows by @p channels columns, or if the blocks have
             *         drifted apart in level or scale -- the last of which is
             *         silent everywhere else and turns into a wrong SoftMax
             *         denominator inside attention.
             */
            void validate(const Batch16Sequence& x, int channels,
                          const char* name) const;

            /** @brief Tokens a sequence of @p blocks token blocks carries. */
            int sequence_tokens(int blocks) const { return blocks * tokens(); }

            /**
             * @brief One pre-norm transformer block over a whole sequence.
             *
             * The same circuit as the single-block driver with attention
             * replaced by its causal blocked form, and every other layer run
             * per token block. At one block it is arithmetically identical to
             * transformer_block(BatchActivation&, ...) -- asserted by test
             * rather than by inspection.
             *
             * @param boot_key Required when @c config.refresh.enabled;
             *                 otherwise unused and may be null. Must hold
             *                 boot_rotation_indices().
             */
            Batch16Sequence
            transformer_block(Batch16Sequence& x,
                              const TransformerBlockWeights& weights,
                              const TransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Relinkey<Scheme::CKKS>& relin_key,
                              Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            // ---------------------------------------------------------------
            // The bridge, which is the only thing here that costs anything
            // ---------------------------------------------------------------

            /**
             * @brief Turn on the bridge levers that are not already on.
             *
             * In practice that is ONE lever: hoisted rotation trains. BSGS is
             * already the default (see the note at the top of this file), and
             * passing @p baby_steps = 0 restates it rather than changing it.
             * The cache capacity is set for completeness; the default 4 is
             * already right for one block.
             *
             * Nothing here changes an answer beyond floating-point
             * reassociation, and this module never calls it by itself -- a
             * measurement should say whether it was called.
             *
             * @param baby_steps 0 keeps the balanced split, which is n1 = 16
             *                   at d = 128. 1 turns BSGS OFF and costs 127 key
             *                   switches a column instead of 22.
             * @param cache_sets Encoded diagonal sets kept at once. Each is
             *                   d plaintexts at the FULL chain length --
             *                   mod_drop on a plaintext moves its depth
             *                   without reallocating -- so the resident cost
             *                   does not fall as the stream descends.
             */
            void use_fast_bridge(int baby_steps = 0,
                                 std::size_t cache_sets = 4);

            /**
             * @brief Hoist the bridge's rotation trains, and nothing else.
             *
             * @c use_fast_bridge takes three levers at once, two of which are
             * already at their right values, so a caller who wants only the
             * one that is off has to accept a cache resize and a restatement
             * of the BSGS split as well. Llama3RectOperator has had the
             * single-lever form all along (llama3_rect.cuh:923) and this path
             * did not, which is a large part of why every batch-16
             * measurement to date ran UNHOISTED: the seam turns hoisting on
             * for itself through BatchSoftmaxSeamConfig and back off after,
             * so the attention crossings were hoisted and the norm and SwiGLU
             * bridges -- the other 88% of the block's bridged columns -- were
             * not.
             *
             * Needs KEYSWITCHING_METHOD_II. Under method I the decomposition
             * is rebuilt inside the shift loop and there is nothing to share,
             * so this is a no-op there rather than a gain -- which matters
             * here, because the only 128-bit-admissible island parameter set
             * at N = 4096 has one special prime and is therefore method I.
             * Hoisting is a BIG-ring lever on this path, necessarily.
             *
             * Changes no answer beyond floating-point reassociation.
             */
            void set_hoisted_crossings(bool on)
            {
                batch_.set_hoisted_crossings(on);
            }
            bool hoisted_crossings() const
            {
                return batch_.hoisted_crossings();
            }

            /**
             * @brief Columns this module bridges for one whole block, at the
             *        configured shape.
             *
             * The number the cost of this module is, and the number a
             * measurement should be divided by before it is compared with
             * anything: 2 * d_model for the two norms and 3 * hidden for the
             * SwiGLU. It does NOT include the attention sublayer's own two
             * crossings, which belong to another module.
             */
            long long bridged_columns_per_block() const;

            /**
             * @brief Report the depth at every seam.
             *
             * Set to a sink to record the level schedule; the default is empty
             * and costs a null check. It is the only place the schedule is
             * visible -- a profiler has the times and not the levels.
             */
            std::function<void(const char* name, int depth)> depth_trace;

          private:
            void note_depth(const char* name,
                            const std::vector<Ciphertext<Scheme::CKKS>>& ct)
                const;
            void note_depth(const char* name, const BatchActivation& x) const;

            /// Drop to the limbs @c level_budget asks for at this seam. A
            /// no-op when the hook is empty, which is the default.
            void apply_level_budget(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                                    const char* seam);

            /// One named seam of the block schedule: trace the depth, and
            /// refresh if @p take says so. Every seam in both drivers goes
            /// through here, so a seam cannot be traced without being
            /// refreshable or refreshed without being traced.
            void seam(const char* name, BatchActivation& x, bool take,
                      Galoiskey<Scheme::CKKS>* boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /// The same over a whole sequence: one refresh per token block.
            void seam(const char* name, Batch16Sequence& x, bool take,
                      Galoiskey<Scheme::CKKS>* boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /// Everything the two block drivers share: the gain folds and the
            /// per-block layer calls, with the seams named identically so the
            /// single-block and sequence schedules cannot drift apart.
            void check_refresh(const TransformerBlockConfig& config,
                               Galoiskey<Scheme::CKKS>* boot_key) const;

            /// Shape-independent config validation, shared by the two norm
            /// entry points so a slot-form caller cannot skip it.
            void check_norm_config(const RMSNormConfig& config) const;

            /// Translate this module's config into the slot core's, including
            /// the count = 1 that makes the channel reduction free.
            void fill_slot_config(Llama3Operator::RMSNormConfig& slot_config,
                                  const RMSNormConfig& config, int channels);

            /// Everything between the two crossings: the fit, and the gain if
            /// it was not folded. Shared by rms_norm and rms_norm_slots so the
            /// two cannot drift.
            std::vector<Ciphertext<Scheme::CKKS>>
            norm_core(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                      const std::vector<double>& gain,
                      const RMSNormConfig& config,
                      const Llama3Operator::RMSNormConfig& slot_config,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      Relinkey<Scheme::CKKS>& relin_key,
                      Galoiskey<Scheme::CKKS>* boot_key);

            /// The norm's slot core written out, so that the 1/sqrt fit can be
            /// told its argument arrives already mapped. @see
            /// RMSNormConfig::sum_pre_scaled. Used only on that path; the
            /// default one delegates to Llama3Operator::rms_norm, which is
            /// what the existing tests assert against.
            std::vector<Ciphertext<Scheme::CKKS>>
            pre_scaled_norm(std::vector<Ciphertext<Scheme::CKKS>>& slots,
                            const RMSNormConfig& config,
                            Relinkey<Scheme::CKKS>& relin_key);

            /// The slice of a row-major in_channels x hidden weight covering
            /// hidden columns [base, base + count), and of the transposed
            /// down weight covering the same hidden rows.
            std::vector<double> gate_slice(const std::vector<double>& weight,
                                           int in_channels, int hidden,
                                           int base, int count,
                                           double scale) const;

            Llama3BatchOperator& batch_;
            Batch16Shape shape_;
            int slot_count_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_BATCH16_H
