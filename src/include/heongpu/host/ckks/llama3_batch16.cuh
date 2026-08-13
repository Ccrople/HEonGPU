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
            BatchActivation rms_norm(BatchActivation& x,
                                     const std::vector<double>& gain,
                                     const RMSNormConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key);

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
                           Relinkey<Scheme::CKKS>& relin_key);

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
                                         Relinkey<Scheme::CKKS>& relin_key);

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
                               Relinkey<Scheme::CKKS>& relin_key);

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
                      Relinkey<Scheme::CKKS>& relin_key);

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
