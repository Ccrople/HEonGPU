// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// The Llama-3 primitives with EVERY plaintext-ciphertext product on Kang's
// Algorithm 5 (rectangular batch PCMM) and every ciphertext-ciphertext product
// on Kang's Algorithm 4 (batch CCMM), from Cheon, Kang and Lee, "Fast Batch
// Matrix Multiplication in Ciphertexts".
//
// WHY THIS FILE EXISTS
// --------------------
// llama3_batch.cuh runs the model on Algorithm 1, where the k/2 packed matrices
// are k/2 INDEPENDENT INPUTS. That is pure throughput and it is the right answer
// when there are k/2 users. It is the wrong answer for one: a single input
// leaves k/2 - 1 slots empty, and the activation is still one ciphertext per
// CHANNEL, which is the encoding that ran out of ciphertext memory before it ran
// out of anything else.
//
// Algorithm 5 spends the batch axis on the input instead. A d x (N/2) matrix is
// k/2 blocks of d x d, and Theorem 3 sums the block products, so ONE input
// occupies the whole ring: N/2 channels in d ciphertexts rather than N/2 of
// them, a factor of k/2 fewer.
//
// THE THREE ENCODINGS, AND WHY THERE ARE THREE
// --------------------------------------------
// This module moves between three representations of the same d x C activation.
// They are not relabellings of each other and each product below states which
// one it consumes.
//
//   RECT     d ciphertexts per group of N/2 channels. Column ciphertext j holds
//   (Y axis) X[token i][channel t*d + j] at coefficient i + d*t, so the channel
//            BLOCK index t rides the Y = X^d axis. This is what Algorithm 5
//            hands back, and feeding it straight back in is what makes a chain
//            of projections free: the constant coefficient of an R_k product is
//            already the contraction over t once the plaintext carries its own
//            blocks reversed and negated (Y^k = -1). The stream lives here.
//
//   BATCH    the matrix encryption of Definition 2: the block index rides the
//   (batch)  BATCH axis, entry (i,j) evaluating to X[i][b*d + j] at zeta^{5^b}.
//            This is what Algorithm 4 consumes, and it is what makes the batch
//            CCMM the right tool for attention: with head_dim = d a block IS a
//            head, so one Algorithm 4 call produces k/2 heads' score matrices
//            at once.
//
//   SLOT     ordinary CKKS slots, slot b + (k/2)*u of column j holding
//            X[token u][channel b*d + j]. Every non-linearity -- RMSNorm, the
//            SoftMax, SiLU -- is slot-wise polynomial evaluation and needs this.
//
// THE MISSING STAGE, WHICH IS THE ENGINEERING CONTENT OF THIS FILE
// ---------------------------------------------------------------
// RECT and BATCH differ by a transform, and it is easy to convince oneself it
// is free. It is not. Evaluating a rect column at the slot points gives
//
//     slot_{b,u} = sum_i psi^{5^{b + u k/2} i} * ( sum_t X[i][t d + j] zeta^{5^b t} ),
//
// so the rect encoding carries, at batch point b, the DISCRETE FOURIER
// TRANSFORM of the blocks rather than block b. Undoing it is a (k/2)-point
// transform along the batch axis. That axis is the FAST slot axis, span k/2, so
// the map is the blocked linear map of k - 1 diagonals: k - 1 rotations, k - 1
// plaintext products and one level.
//
// It does NOT commute with the row Vandermonde of the slot bridge, because that
// Vandermonde's nodes psi^{5^{b + u k/2}} depend on b as well as on u. So the
// two cannot be fused, and RECT -> BATCH is three stages and three levels:
// the row bridge down to slots, the block transform, the row bridge back up.
// A single BSGS linear map over all N/2 diagonals would do it in one level and
// about 2 sqrt(N/2) rotations, at the cost of N/2 plaintext encodings per
// ciphertext; that trade has not been measured and is not taken here.
//
// This is the transform profile_llama3_nobatch.cpp records as "not written
// yet", and without it Algorithm 5 can only do projections.
//
// WHAT IT COSTS, STATED UP FRONT
// ------------------------------
// Algorithm 5's summation is a CMT read at k = 2, whose automorphisms run over
// the WHOLE rotation group: N/2 - 1 Galois keys, a count that depends on the
// ring degree ALONE -- not on d, not on the width of the model. HEonGPU's
// minimum ring is 4096, so the floor is 2047 keys, and at the ~60-limb chain a
// whole transformer block wants that is far past one card. Nothing in this file
// can move that; N is the only lever, and it is capped below. Read the shape
// checks accordingly: this module is correct at any width, and it FITS only at
// short chains.
//
// THE REFRESH, AND WHY IT NEEDS NO CROSSING
// -----------------------------------------
// A whole block on this path spends far more levels than a chain that also has
// to hold N/2 - 1 rotation keys can carry, so it does not run without a refresh
// in the middle of it. The refresh is CKKS bootstrapping, and the useful fact is
// that HEonGPU's REGULAR bootstrapping is an identity on the plaintext
// POLYNOMIAL and not on any particular reading of it:
//
//     m(x) --ModRaise--> m(x) + q I(x) --CtoS--> Encode(m + qI)
//          --EvalMod--> Encode(m) --StoC--> m(x)
//
// CoeffToSlot puts the COEFFICIENTS in slots, EvalMod removes the multiples of
// q from them, and SlotToCoeff puts them back. Nothing in that sequence asks
// what the coefficients mean. So it refreshes a RECT column and a BATCH matrix
// encryption exactly as it refreshes an ordinary slot ciphertext, and a refresh
// costs no encoding crossing at all -- which matters here, because a crossing is
// 86% of an unrefreshed block. This is asserted by test, not by argument:
// test_ckks_llama3_rect.cpp bootstraps in all three encodings and checks the
// matrix that comes back.
//
// The coefficient bound is the one condition, and the rect encoding meets it
// more easily than the slot encoding does: a rect coefficient IS one data entry
// times the scale, where a slot coefficient is a transform of N/2 of them.
// Measured on an A6000 at N = 4096, 31 limbs: a rect column comes back with a
// worst absolute error of 1.2e-05 and a matrix encryption 5.2e-05, against the
// 1e-3 a single crossing costs. The refresh is the accurate part of this
// circuit.
//
// The key set is free as well. Bootstrapping asks for 24 rotation indices at
// this ring, and Algorithm 5's CMT already holds every one of the 2047 there
// are, so the union is 2048 -- one more key than the path needed anyway.
//
// WHERE THE REFRESHES GO
// ----------------------
// Six seams per block, chosen so that no stretch between two of them spends more
// levels than one bootstrap hands back. RectRefreshConfig names them and any of
// them can be switched off; set depth_trace to record what each stretch costs,
// which is the number that decides whether a chain is long enough and which
// appears in no timing report.
//
// The chain follows from the schedule and not the other way round. A regular
// bootstrap at these parameters spends 25 levels whatever the chain is, so a
// chain of L limbs hands back L - 26 usable levels, and a schedule whose worst
// stretch is S wants exactly L = S + 26. Longer is not safer: it is slower, in
// proportion to L^2 at dnum = 1, for levels the circuit throws away at the next
// refresh.
//
// Measured, with the defaults this file carries -- RMSNorm at degree 15 with no
// Newton step, the SoftMax at exp degree 15 and reciprocal degree 7, the SiLU at
// the paper's degree 31:
//
//   RMSNorm .......................................... 12
//   Q, K, V projections ............................... 1
//   to_batch, the K transpose, Q K^T, to_slots ........ 5
//   SoftMax .......................................... 12
//   from_slots, P V, from_batch, W_o, the residual ..... 7
//   RMSNorm again .................................... 12
//   gate and up, to_slots, SiLU, the gate product ..... 10
//   from_slots, W_down, the residual ................... 4
//
// Two things in that table are worth reading twice. Algorithm 4's product is ONE
// level and the CMT transpose is FREE, so the levels go where the time goes: the
// encoding crossings. And nothing here is 25, which is why the refresh is worth
// what it costs -- 6 seams of it a block, at 12 levels apiece.
//
// WHERE THE NON-LINEAR LEVELS WENT
// --------------------------------
// That table was measured before Section 3.1.3 was taken seriously, and it paid
// for two things it did not need and one it did.
//
// A DEGREE IS NOT A FREE PARAMETER. It is what the operating range and the
// required precision leave, and 12 bits is the requirement -- Section 3.1.2,
// where that is what matches the perplexity of plaintext FP16. The RMSNorm fit
// spans a factor of 2.5 and reaches 3.2e-06 at degree 7; at 15 it reached
// 1.5e-11, which is nine orders of magnitude nothing downstream can read, for a
// level. chebyshev_degree_for() settles this by measurement rather than by
// argument, and the profile target now prints what every fit in the block
// achieves over its own configured range.
//
// The reciprocal went the OTHER way, and this is the one that mattered. Fitted
// over the worst case -- every score at the bottom of [-bound, 0] or every one
// at the top, a factor of exp(8) -- degree 7 is wrong by 98.6%, and reaching 12
// bits there takes degree 255 and eight levels. Section 4.3 does not raise the
// degree, it narrows the interval: "rather than relying on worst-case bounds as
// [25], we use the distributional data computed during calibration to obtain
// sharp estimates". Over a calibrated range degree 15 meets 12 bits in four
// levels. SoftmaxConfig::sum_lo/sum_hi are that estimate, and they are inputs
// because a range is a measurement of the model and not a property of the
// algorithm.
//
// THE DOMAIN MAP IS FREE IF SOMETHING ELSE IS ALREADY PAYING. A fit on [a, b]
// carries its argument onto [-1, 1] first; the shift is an addition and free,
// the multiplier is a plaintext product and a level. Every one of these fits
// has a plaintext product just upstream of it that can carry the multiplier
// instead -- the mask of the blocked reduction for RMSNorm, the causal mask for
// the reciprocal, and for the exponential the query weight itself, where
// 1/sqrt(head_dim) already rides. This is the fusion discipline of Section 3.2
// applied to the non-linear layers rather than to the format conversions.
//
// Measured again at the same shape, the same six seams, the same 38 limbs:
//
//   RMSNorm ................. 12 -> 10   degree 7, domain map on the mask
//   Q, K, V projections ...... 1 ->  1
//   to_batch, K^T, Q K^T ..... 5 ->  5
//   SoftMax ................. 12 -> 11   exp map on W_q, 1/x map on the mask,
//                                        reciprocal 7 -> 15 and now correct
//   P V, W_o, the residual ... 7 ->  7
//   RMSNorm again ........... 12 -> 10
//   SwiGLU .................. 10 -> 10   already at Table 2's bound and degree
//   W_down, the residual ..... 4 ->  4
//
//   worst stretch 12 -> 11, at the SoftMax rather than the RMSNorm
//
// The chain is sized on the worst stretch, so 11 wants 37 limbs and not 38.
//
// AT THIS SHAPE IT IS NOT FASTER. Measured on one A6000 at d = 64, hidden 4096,
// 8 kv heads -- one shape, three circuits:
//
//   before, 38 limbs ... 563.3 s   the old schedule, reciprocal wrong by 98.6%
//   after,  38 limbs ... 576.8 s   +2.4%, correctness at the old chain
//   after,  37 limbs ... 562.2 s   -0.2%, correctness at the chain it now needs
//
// So at half the width the level savings buy back exactly what correctness
// costs and no more. The reciprocal's degree had to go UP to be an
// approximation at all, and that eats the two folds and the RMSNorm's degree
// drop between them.
//
// DO NOT GENERALISE THAT TABLE TO THE REAL MODEL. The same three circuits at
// Llama-3 8B's own width give -10.9%; the block below has the numbers. A
// half-width block is ten Algorithm 5 calls, so the key switching that a limb
// actually shrinks is a small share of it, and the row above is measuring
// mostly the things a limb does not touch.
//
// The SoftMax is now the worst stretch and the RMSNorm is not, which says where
// to look next: 11 is the exponential's four levels, the reciprocal's four, the
// squaring, the mask and the product. None of those is slack. Note also that 37
// leaves the SoftMax exactly ONE limb, where 38 left it two, so the shorter
// chain spends the margin. At half the width that is a real choice, since 37
// buys nothing there; at the real width it is not, since it buys 11.7%.
//
// THE SWIGLU'S LEVEL IS IN THE SEAM, NOT IN THE FIT
// -------------------------------------------------
// Once the narrow-track refresh takes the SoftMax down to 7, the SwiGLU is the
// worst stretch at 9, and it decomposes as projection 1 + crossing 2 + SiLU 5 +
// gate product 1. Three of those four are load-bearing:
//
//   - the crossing is 2 because it is two transforms, the row bridge and the
//     block map, and the block map is a subring DFT rather than a permutation.
//     That matters: a permutation would COMMUTE with a slot-wise SiLU and with
//     a Hadamard product, so the one in to_slots and the one in from_slots
//     would cancel and the SwiGLU would need neither. A DFT does not commute,
//     and the fit genuinely needs its argument in the value domain.
//
//   - the SiLU is 5 because 5 is what 12 bits over Table 2's range costs, and
//     no rearrangement of the polynomial beats it. SiLU(x) = x/2 + h(x^2) with
//     h even, which halves the degree -- 31 in x becomes 15 in x^2 at exactly
//     the same error -- and then the squaring spends exactly the level the
//     halving saved. Measured on the host, that tie holds at EVERY bound: a
//     degree-31 direct fit and a degree-15 fit in x^2 both reach B = 14.1 in 5
//     levels, both reach 6.1 in 4. The only lever on the fit is the range, and
//     the range is a measurement of the model: 10.8, which wants 31.
//
//   - the projection is one plaintext product.
//
// The gate product is the one that need not be where it is. It does not have to
// sit ABOVE the seam. Refreshing the ACTIVATION instead of the hidden is the
// same single bootstrap one level earlier, and it takes the stretch to 8 --
// which is where the second-worst stretch, the attention's 7, starts to matter.
// RectFeedForwardConfig::refresh_activation is that move. It is legal because
// the fit can be given a gain of 1/B, landing the activation on the [-1, 1] a
// bootstrap assumes (Section 3.1.3), with the up weight carrying B back on the
// product the activation was heading for anyway -- two host-side numbers, so
// the scaling that makes the refresh sound costs no level of its own.
//
// Measured at d = 64, d_model 2048, hidden 2048, N = 4096, seven seams, with
// the narrow-track refreshes and the folds on -- one A6000, three circuits:
//
//   seam on the hidden,     35 limbs ... 305.9 s   stretch 9
//   seam on the activation, 35 limbs ... 301.1 s   stretch 8, chain now overlong
//   seam on the activation, 34 limbs ... 261.9 s   stretch 8, -14.4%
//
// The middle row is the move on its own and it is already slightly ahead: the
// product moves ABOVE the seam so from_slots and the down projection run at
// four fewer limbs, which buys more than the product loses by running at seven
// instead of one. The bottom row is the limb, and it is worth far more than the
// L^2 of key switching suggests, for the reason the chain note above gives --
// a refresh hands back CHAIN LESS 25, so a limb off 35 is a limb off a working
// window of 10.
//
// Both FFN stretches are 8 now (fit-side 8, tail-side 8, balanced), and the
// attention's 7 is next. Going below 8 needs the SiLU in four levels, which
// needs B <= 6.1, which the model does not offer at 10.8 -- or a second wide
// bootstrap on the up branch, which is not obviously worth one.
//
// THE REAL SHAPE, MEASURED
// ------------------------
// That table is one block at d = 64 and half the width. Llama-3 8B's own
// numbers need no rounding to fit here, and the reason is the constraint below:
// head_dim must be exactly d, and 8B's head dim IS 128. So d = 128, 32 heads x
// 128 = 4096 = two channel groups at N = 4096, and 14336 = seven. Grouped-query
// attention at 8 kv heads is the host expansion RectAttentionConfig describes.
//
//   d_model 4096, hidden 14336, 32 heads of 128 over 8 kv heads, 128 tokens
//   37 limbs, 2048 Galois keys at 9.25 GiB, one A6000, 42.9 GiB of 47.5 used
//   worst stretch 11, at the SoftMax -- the same schedule as at half the width,
//   stretch for stretch, with attention.softmaxed landing on its last limb
//   14 refreshes: the attention seams once per head group (32 heads is two
//   Algorithm 4 calls of k/2 = 16), the SwiGLU once per hidden group
//   2964.6 s for one block, against 562.2 s at half the width
//
// The first of those is the useful one: a level schedule is a property of the
// CIRCUIT and not of the model, so a chain sized on a shape that fits in an
// afternoon holds at the real one. The cost is not -- 5.3x for a model 2x as
// wide, because a projection is one Algorithm 5 call per (input group, output
// group) pair and that count went from 10 to 54.
//
// AND HERE THE LIMB IS WORTH SOMETHING. The same three circuits at this width:
//
//   before, 38 limbs ... 3326.8 s
//   after,  38 limbs ... 3358.4 s   +0.9%, the circuit alone -- a wash
//   after,  37 limbs ... 2964.6 s   -10.9% on the baseline, -11.7% on the row up
//
// The circuit change costs about nothing at either width. The whole win is the
// limb, and its size is not the 2.6% that 38 -> 37 looks like, because a
// refresh hands back CHAIN LESS 25: the window the block actually works in went
// 13 -> 12 limbs, which is 7.7%. Pure O(L) on that window would give -7.7% and
// pure O(L^2) -14.8%; the measured -11.7% sits between, which is what linear
// work plus an O(L^2) mod-down should give. The same window shrinks identically
// at half the width and buys 2.5% there, because there is a tenth as much key
// switching for it to shrink.
//
// So read the level schedule off the small shape, and never the clock.
//
// ORIENTATION AND SHAPE CONSTRAINTS
// ---------------------------------
// Kang's products put the encrypted operand on the LEFT, while a Llama
// projection is W(plaintext) X(encrypted). As on the Algorithm 1 path the
// activation is therefore held transposed, X[token][channel], and the weights
// are stored transposed on the host where a transpose is free.
//
//   rows      d, the tokens of one block. Fixed by the ring: d is the rank of
//             R_N over R_k.
//   channels  free, in groups of N/2. A width that is not a whole number of
//             groups is padded, and the padding is paid for.
//   head_dim  EXACTLY d. Algorithm 4 contracts over the column index and leaves
//             the batch index alone, so a head has to be one channel block for
//             its contraction to be the one the batch CCMM performs. This is
//             the single real shape constraint the encoding imposes.

#ifndef HEONGPU_CKKS_LLAMA3_RECT_H
#define HEONGPU_CKKS_LLAMA3_RECT_H

#include <heongpu/host/ckks/batchmatrix.cuh>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/host/ckks/ciphertext.cuh>
#include <heongpu/host/ckks/decryptor.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/encryptor.cuh>
#include <heongpu/host/ckks/evaluationkey.cuh>
#include <heongpu/host/ckks/llama3.cuh>
#include <heongpu/host/ckks/llama3_batch.cuh>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/plaintext.cuh>

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

namespace heongpu
{
    namespace llama
    {
        /**
         * @brief An activation in the rectangular encoding of Algorithm 5.
         *
         * @c column holds @c groups * d ciphertexts, group major: group g owns
         * @c column[g*d .. g*d + d), and carries channels
         * [g * (N/2), (g+1) * (N/2)). Within a group, column ciphertext j holds
         * X[token i][channel t*d + j] at coefficient i + d*t, so the block index
         * t rides the Y axis and the whole group is d ciphertexts however wide
         * the ring is.
         *
         * @c channels may be less than @c groups * (N/2); the remainder is
         * zero padding, and it is padding that gets multiplied like anything
         * else, so it costs what it costs.
         */
        struct RectActivation
        {
            std::vector<Ciphertext<Scheme::CKKS>> column;

            /// Rows of the encrypted matrix: layout.d, the tokens of one block.
            int rows = 0;
            /// Groups of N/2 channels the width is cut into.
            int groups = 0;
            /// Channels actually carried, at most groups * (N/2).
            int channels = 0;

            bool empty() const { return column.empty(); }
        };

        /**
         * @brief The Llama-3 linear algebra on Kang's Algorithms 4 and 5.
         *
         * Every projection is Algorithm 5 and every encrypted product is
         * Algorithm 4. The non-linear layers are not reimplemented: they stay on
         * Llama3Operator in slot form, behind the bridge.
         *
         * Composed on Llama3BatchOperator rather than duplicating it. That class
         * already owns the row Vandermonde bridge, the batch matrix operator and
         * the slot-form arithmetic operator, and building second copies of all
         * three would double the twiddle caches for nothing.
         */
        class Llama3RectOperator
        {
          public:
            /**
             * @param layout Subring layout. @c layout.d is the token block size
             *               and fixes both the channel block size and the number
             *               of blocks per group, k/2.
             * @param scale  Default scaling factor, shared with the slot-form
             *               operator.
             */
            Llama3RectOperator(HEContext<Scheme::CKKS> context,
                               HEEncoder<Scheme::CKKS>& encoder,
                               const BatchMatrixLayout& layout, double scale);

            const BatchMatrixLayout& layout() const noexcept { return layout_; }

            /** @brief Tokens one activation carries, layout.d. */
            int rows() const noexcept { return layout_.d; }

            /** @brief Channels one group carries, N/2. */
            int channels_per_group() const noexcept { return layout_.N / 2; }

            /** @brief Channel blocks in one group, k/2. */
            int blocks_per_group() const noexcept { return layout_.batch; }

            /** @brief Groups a width of @p channels needs. */
            int groups_for(int channels) const;

            double default_scale() const noexcept { return default_scale_; }

            /**
             * @brief The slot-form operator this one is built on.
             *
             * Exposed for one reason: the bootstrapping context is per
             * HEArithmeticOperator instance, so generate_bootstrapping_params
             * has to be called on THIS operator's arithmetic half and not on a
             * second one the caller happens to own. Everything else it offers
             * is reachable through the sublayers above.
             */
            Llama3Operator& arith() noexcept { return batch_.arith(); }

            // ---------------------------------------------------------------
            // The refresh
            // ---------------------------------------------------------------

            /**
             * @brief Refresh one ciphertext, whatever encoding it carries.
             *
             * Regular bootstrapping is an identity on the plaintext polynomial,
             * so this is correct on a RECT column, on a BATCH matrix
             * encryption and on an ordinary slot ciphertext alike, and it needs
             * no crossing to reach a form that can be refreshed. See the note
             * at the top of this file.
             *
             * generate_bootstrapping_params must have run on arith() first,
             * and @p boot_key must hold the union of this operator's
             * rotation_indices() and arith().bootstrapping_key_indexs(). A
             * shift-vector Galois key asked for an index it does not hold is
             * undefined behaviour rather than an error, so build the union.
             *
             * Whatever levels were left are dropped: the procedure starts from
             * a ciphertext with one prime remaining.
             */
            Ciphertext<Scheme::CKKS>
            bootstrap(Ciphertext<Scheme::CKKS>& ct,
                      Galoiskey<Scheme::CKKS>& boot_key,
                      Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief Refresh a RECT column and hand back its SLOT reading.
             *
             * The crossing this saves is not merely fused into the bootstrap;
             * it IS the bootstrap's second stage. A rect column's data occupies
             * R_N coefficients 0..N/2-1 and nothing above -- see
             * build_matrix_encryption_coefficients, which writes block t at
             * i + d*t for t < k/2 only -- and the composition this module
             * otherwise performs to reach slot form, the d-point row bridge
             * along the token axis then the k/2-point block_map along the block
             * axis, is a factorisation of the N/2-point DFT. That is the
             * transform CoeffToSlot evaluates.
             *
             * So bootstrap() followed by to_slots() runs the same DFT three
             * times: once as CoeffToSlot, once backwards as SlotToCoeff, once
             * forwards again as the crossing. This entry point runs it once. It
             * saves the caller the two crossing levels and the bootstrap's
             * StoC_piece, and it halves EvalMod, because the upper coefficient
             * half that solo_coeff_to_slot discards was zero to begin with.
             *
             * The slot ORDER is the one CoeffToSlot produces, which is not the
             * one to_slots() produces; slot_reading_permutation() gives the
             * relabelling between them. Same key and parameter requirements as
             * bootstrap().
             */
            Ciphertext<Scheme::CKKS>
            bootstrap_to_slots(Ciphertext<Scheme::CKKS>& ct,
                               Galoiskey<Scheme::CKKS>& boot_key,
                               Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief Where bootstrap_to_slots leaves entry i + d*t of a column.
             *
             * Returns p of length N/2 with p[c] = the slot that coefficient c
             * lands in. The rect encoding walks the token axis fastest, and
             * CoeffToSlot reads the coefficient index straight through, so the
             * two orders differ by the d x (k/2) stride transpose this returns.
             */
            std::vector<int> slot_reading_permutation() const;

            /** @brief Refresh every ciphertext, in place. */
            void bootstrap(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                           const char* name,
                           Galoiskey<Scheme::CKKS>& boot_key,
                           Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Refresh a rectangular activation, in place. */
            void bootstrap(RectActivation& x, const char* name,
                           Galoiskey<Scheme::CKKS>& boot_key,
                           Relinkey<Scheme::CKKS>& relin_key);

            /** @brief Refresh a matrix encryption, in place. */
            void bootstrap(BatchActivation& x, const char* name,
                           Galoiskey<Scheme::CKKS>& boot_key,
                           Relinkey<Scheme::CKKS>& relin_key);

            /**
             * @brief Which seams of a block take a refresh.
             *
             * A stretch between two refreshes has to fit in what one
             * bootstrap hands back, and these are the points at which the
             * budget runs out on this path. Switching one off is what shows
             * that it was needed.
             */
            struct RectRefreshConfig
            {
                /// The residual stream on the way in. False for the first
                /// block of a stack, whose input is already fresh; true is
                /// what joins one block to the next.
                bool entry = false;
                /// The normalised stream, before the projections that read it.
                /// RMSNorm is the deepest single stretch on this path and it
                /// leaves nothing for the Q, K and V projections behind it.
                bool after_attention_norm = true;
                /// The scores in slot form, before the SoftMax. This is the
                /// image's post-QK refresh, and it is the one the SoftMax's
                /// depth forces.
                bool post_qk = true;
                /// The SoftMax output, together with V. The two meet in the
                /// value product, so refreshing one without the other only
                /// moves the problem: V would then sit above P with no way
                /// down to it.
                bool post_softmax = true;
                /// The residual stream between the two halves.
                bool mid = true;
                /// The normalised stream, before the gate and up projections.
                bool after_feed_forward_norm = true;
                /// The SwiGLU hidden, after the SiLU and the gate product --
                /// or, with RectFeedForwardConfig::refresh_activation, on the
                /// activation just before that product.
                ///
                /// The SwiGLU half is 14 levels with the SiLU at the paper's
                /// degree 31, so it is this seam or a degree nobody fits a
                /// SiLU at. Splitting here rather than before the SiLU keeps
                /// the two stretches at 10 and 4 instead of 3 and 11; taking
                /// it one level earlier still, on the activation, makes them
                /// 8 and 8 for the same single bootstrap.
                bool feed_forward_hidden = true;

                /// Refreshes one block takes with these flags.
                int count() const;
            };

            /**
             * @brief Report the depth at every seam of the next block.
             *
             * Set to a sink to record the level schedule; the default is empty
             * and costs nothing. It is called with the seam's name and the
             * stream's depth at that point, which is the only place the
             * schedule is visible -- an Nsight report has the times and not
             * the levels.
             */
            std::function<void(const char* name, int depth)> depth_trace;

            /**
             * @brief Report every seam's raw ciphertext, for debugging.
             *
             * Called alongside @c depth_trace with the SAME vector, empty by
             * default and costing nothing then. A caller that decrypts here
             * gets a generic (slot-decoded) magnitude at every named point,
             * whatever encoding the ciphertext actually carries -- accurate
             * enough to see WHERE a value leaves its expected range, not
             * accurate enough to read the value itself at a rect or matrix
             * seam. Nothing in this class reads it.
             */
            std::function<void(const char* name,
                               const std::vector<Ciphertext<Scheme::CKKS>>&
                                   ct)>
                ct_trace;

            // ---------------------------------------------------------------
            // Rotation keys
            // ---------------------------------------------------------------

            /**
             * @brief Rotations the block transform needs a Galois key for.
             *
             * The transform is blocked at span k/2 with the block index on the
             * FAST slot axis, so its diagonals are the offsets
             * -(k/2 - 1) .. k/2 - 1, taken modulo the slot count.
             */
            std::vector<int> block_rotation_indices() const;

            /**
             * @brief Every rotation index this operator can ask for.
             *
             * The union of Algorithm 5's own indices, the row bridge's, the
             * block transform's and the blocked reduction RMSNorm runs. In
             * practice Algorithm 5's set already covers it: the CMT at k = 2
             * runs the automorphisms over the whole rotation group, so this is
             * N/2 - 1 indices whatever else is added to it. It is still built
             * honestly, so the list means something if the rectangular product
             * is ever taken out.
             */
            std::vector<int> rotation_indices() const;

            // ---------------------------------------------------------------
            // Host-side staging, for tests and for the client side
            // ---------------------------------------------------------------

            /**
             * @brief Encode and encrypt a d x @p channels real matrix.
             *
             * @param x Row major, @c rows() * @p channels entries.
             */
            RectActivation encrypt(const std::vector<double>& x, int channels,
                                   HEEncryptor<Scheme::CKKS>& encryptor,
                                   double scale);

            /** @brief Inverse of encrypt, for checking a result. */
            std::vector<double> decrypt(RectActivation& in,
                                        HEDecryptor<Scheme::CKKS>& decryptor,
                                        double scale);

            // ---------------------------------------------------------------
            // Moving between the three encodings
            // ---------------------------------------------------------------

            /**
             * @brief The block transform, in place, on slot-form ciphertexts.
             *
             * @param inverse false applies F, the map that takes block values to
             *                their transform along the batch axis; true applies
             *                F^-1, which is what turns a rect group into
             *                separated channels. F[b][t] = zeta^{5^b t} is a
             *                Vandermonde in k/2 distinct 2k-th roots of unity,
             *                so it is invertible and well conditioned.
             *
             * Costs k - 1 rotations, k - 1 plaintext products and one level per
             * ciphertext. The diagonals are shared by every ciphertext in the
             * call and encoded once, which the row bridge does not do.
             */
            void block_map(std::vector<Ciphertext<Scheme::CKKS>>& ct,
                           bool inverse, const char* name,
                           Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Rectangular form to slot form.
             *
             * Slot b + (k/2)*u of output ciphertext g*d + j holds
             * X[token u][channel g*(N/2) + b*d + j]: the token index on the slow
             * axis and the channel block on the fast one. Two levels, being the
             * row bridge and then the block transform.
             */
            std::vector<Ciphertext<Scheme::CKKS>>
            to_slots(RectActivation& in, Galoiskey<Scheme::CKKS>& galois_key);

            /** @brief Slot form back to rectangular form. Two levels. */
            RectActivation from_slots(std::vector<Ciphertext<Scheme::CKKS>>& in,
                                      int channels,
                                      Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief One group of a rectangular activation as a matrix
             *        encryption, ready for Algorithm 4.
             *
             * Batch slot b of the result is channel block b of the group, so
             * with head_dim = d it is head b. Three levels: down to slots, the
             * block transform, and back up.
             */
            BatchActivation to_batch(RectActivation& in, int group,
                                     Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief Matrix encryptions back to one rectangular activation.
             *
             * The inverse of to_batch over every group at once, so the groups
             * have to arrive in channel order.
             */
            RectActivation from_batch(std::vector<BatchActivation>& groups,
                                      int channels,
                                      Galoiskey<Scheme::CKKS>& galois_key);

            // ---------------------------------------------------------------
            // The products
            // ---------------------------------------------------------------

            /**
             * @brief Y = X W, Kang's Algorithm 5.
             *
             * One call per (input group, output group) pair, and the partial
             * products of one output group are summed. Every term leaves the
             * same sequence of operations behind it, so they meet at one level
             * and one scale and the sum is a plain addition: widening a model
             * buys products and no depth at all.
             *
             * @param weight Row major @p in_channels x @p out_channels, the
             *               transpose of the mathematical weight. Held on the
             *               host as one copy.
             *
             * The uploaded plaintext is d * (N/2) * k coefficients per limb --
             * N^2/2 of them, regardless of how much of the weight is actually
             * non-zero. That is Algorithm 5's plaintext and not an inefficiency
             * here, but it does mean a narrow projection costs a wide one.
             */
            RectActivation project(RectActivation& x,
                                   const std::vector<double>& weight,
                                   int in_channels, int out_channels,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key);

            /**
             * @brief A B with both operands encrypted, Kang's Algorithm 4.
             *
             * Both operands are matrix encryptions at exactly d columns, which
             * is what Algorithm 4's three internal CMTs assume, and the batch
             * axis carries the heads.
             */
            BatchActivation matmul(BatchActivation& a, BatchActivation& b,
                                   const char* name,
                                   Galoiskey<Scheme::CKKS>& galois_key,
                                   Relinkey<Scheme::CKKS>& relin_key);

            // ---------------------------------------------------------------
            // RMSNorm
            // ---------------------------------------------------------------

            /** @brief Shape and approximation settings for rms_norm. */
            struct RectRMSNormConfig
            {
                double eps = 1e-5;
                double sum_lo = 0.0; ///< Range of the summed square.
                double sum_hi = 0.0;
                int degree = 31;
                int newton_iterations = 2;
                /// Fit 1/sqrt over the summed square rather than over the
                /// mean, which saves the level that forming the mean costs.
                /// On by default here and off on the slot path: this is the
                /// encoding under level pressure. Ignored when
                /// @c newton_iterations is above zero.
                bool fold_mean_into_fit = true;
                /// Carry the domain map of the fit on the mask of the blocked
                /// reduction. On here for the same reason, and available here
                /// for the same reason: this encoding is the one that pays for
                /// a masked reduction, so it is the one with a plaintext
                /// product to fold into.
                bool fold_affine_into_mask = true;
                /// Multiplied into the fitted 1/sqrt: the norm hands back
                /// output_scale * x / rms(x) at no extra cost. This is where
                /// the 1/B of the bootstrapping bound rides when the seam
                /// after this norm is refreshed; the projections that read the
                /// normalised stream carry B back in, on the host. Needs
                /// @c newton_iterations = 0 when not one.
                double output_scale = 1.0;
                /// Refresh the summed square before its 1/sqrt fit -- the
                /// narrow-track bootstrap of the paper's Figure 2, which
                /// moves the fit's levels above the wide track. Needs the
                /// boot key at the call and no Newton step.
                bool refresh_sum = false;
            };

            /**
             * @brief RMSNorm over the channel axis.
             *
             * The channel axis runs partly across ciphertexts and partly along
             * the fast slot axis, so the mean is a free slot-wise addition of
             * the groups' columns followed by ONE blocked reduction of span
             * k/2. That reduction costs the level a mask costs; the Algorithm 1
             * path pays nothing here because there a channel is a whole
             * ciphertext, and this is the price of putting k/2 channels in one.
             *
             * @param weight One learned scale per channel, or empty to leave the
             *               scaling out.
             */
            RectActivation rms_norm(RectActivation& x,
                                    const std::vector<double>& weight,
                                    const RectRMSNormConfig& config,
                                    Galoiskey<Scheme::CKKS>& galois_key,
                                    Relinkey<Scheme::CKKS>& relin_key,
                                    Galoiskey<Scheme::CKKS>* boot_key = nullptr);

            // ---------------------------------------------------------------
            // Attention
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one attention sublayer. */
            struct RectAttentionWeights
            {
                /// Row major in_channels x out_channels, the transpose of the
                /// mathematical weight, as project() wants it.
                std::vector<double> query;
                std::vector<double> key;
                std::vector<double> value;
                std::vector<double> output; ///< Empty skips W_o.
            };

            /** @brief Shape and approximation settings for attention. */
            struct RectAttentionConfig
            {
                int in_channels = 0;
                /// heads * d. head_dim is not a free parameter here: a head has
                /// to be exactly one channel block for Algorithm 4's
                /// contraction to be the head's own.
                int heads = 1;
                /// Distinct key and value heads, for grouped-query attention;
                /// 0 means one apiece. Must divide @c heads.
                ///
                /// GQA is expanded on the HOST: the key and value weights are
                /// widened to @c heads heads by repeating each kv head's
                /// columns. Homomorphically that is the only way to put a kv
                /// head in every batch slot that reads it, since the batch slots
                /// of a matrix encryption never talk to each other. It costs
                /// projection work in the ratio heads / kv_heads and buys back
                /// exactly nothing, which is worth knowing before choosing d.
                int kv_heads = 0;
                bool causal = true;
                /// 1/sqrt(d) if left at 0. Folded into the query weight on the
                /// host, where a scaling is free.
                double head_scale = 0.0;
                /// Subtracted from the scores so they land in [-bound, 0].
                /// Calibrated, as in the paper, not computed homomorphically.
                double score_shift = 0.0;
                /// Per-(query row, head) shift, row major d x heads, in true
                /// score units; empty keeps the scalar above. Wins over it.
                ///
                /// One global shift leaves the SoftMax denominator spanning
                /// whatever the sharpest and flattest rows disagree by --
                /// measured 256x on real weights, which no reciprocal at any
                /// sane degree covers. Shifting each row-head's maximum to
                /// zero instead FLOORS the denominator at one, because the
                /// coordinate achieving the maximum contributes exactly
                /// exp(0). A shift constant along the key axis cancels in
                /// the normalisation, so the output is unchanged; the range
                /// the reciprocal must cover is not. It is calibration data
                /// like every other range here, and it costs an encoded
                /// ADDITION: no level, no rescale, no key.
                std::vector<double> score_shift_rows;
                /// Carry the domain map of the exponential, 2/bound, on the
                /// query weight -- where 1/sqrt(head_dim) already rides and
                /// where a scaling costs nothing. The shift above is scaled to
                /// match, and the SoftMax is told its input arrives mapped.
                ///
                /// Worth one level of the deepest stretch on this path, and it
                /// is available only because the scores are formed from a
                /// plaintext weight this operator owns.
                bool fold_softmax_domain_into_query = true;
                /// Carry the domain map of the reciprocal on the causal mask,
                /// the SoftMax's own half of the same trick. Needs a mask, so
                /// it needs @c causal.
                bool fold_softmax_affine_into_mask = true;
                /// bound, iterations and the degrees are the caller's; stride
                /// and count are fixed by this encoding and overwritten, because
                /// the key axis is entirely across ciphertexts.
                Llama3Operator::SoftmaxConfig softmax;
            };

            /**
             * @brief One attention sublayer.
             *
             * Projections through Algorithm 5, scores and the value product
             * through Algorithm 4, and the SoftMax in slot form behind the
             * bridge. A group carries k/2 heads and one Algorithm 4 call does
             * all of them, which is the batch CCMM used for what it is for.
             *
             * @param boot_key Non-null with @p refresh to take the post-QK and
             *                 post-SoftMax refreshes; null runs the sublayer in
             *                 one stretch, which is the deepest thing on this
             *                 path and the reason the refresh exists.
             */
            RectActivation attention(RectActivation& x,
                                     const RectAttentionWeights& weights,
                                     const RectAttentionConfig& config,
                                     Galoiskey<Scheme::CKKS>& galois_key,
                                     Relinkey<Scheme::CKKS>& relin_key,
                                     Galoiskey<Scheme::CKKS>* boot_key = nullptr,
                                     const RectRefreshConfig* refresh = nullptr);

            // ---------------------------------------------------------------
            // SwiGLU
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one SwiGLU sublayer. */
            struct RectFeedForwardWeights
            {
                /// Row major in_channels x hidden_channels; down is the
                /// transpose shape.
                std::vector<double> gate;
                std::vector<double> up;
                std::vector<double> down;
            };

            /** @brief Shape and approximation settings for feed_forward. */
            struct RectFeedForwardConfig
            {
                int in_channels = 0;
                int hidden_channels = 0;
                double silu_bound = 10.8; ///< Table 2 after calibration.
                int silu_degree = 31;     ///< Section 3.1.3.
                /// Carry the domain map of the SiLU fit, 1/silu_bound, on the
                /// gate weight -- the SwiGLU half's own copy of the query-
                /// weight fold. The gate projection feeds the SiLU and
                /// nothing else, so the factor needs no undoing, and a host
                /// scaling is free where the map is otherwise the one
                /// plaintext product the fit pays before its series.
                bool fold_silu_domain_into_gate = false;
                /// Take the feed-forward seam on the ACTIVATION, before the
                /// gate product, rather than on the hidden after it.
                ///
                /// It is the same seam and the same one bootstrap -- only
                /// earlier by one level -- and that level is the whole point:
                /// the stretch that sets the chain runs from the norm's
                /// refresh to whatever the seam lands on, so moving the seam
                /// up the gate product takes it from 9 levels to 8. Nothing
                /// downstream notices, because the product it used to precede
                /// still happens, just below the seam instead of above it.
                ///
                /// The activation leaves the fit on [-B, B] and a bootstrap
                /// wants [-1, 1] (Section 3.1.3), so the SiLU is fitted with
                /// a gain of 1/activation_scale() and the up weight carries
                /// the factor back. Both are host-side numbers: the scaling
                /// that makes the refresh legal costs no level at all.
                bool refresh_activation = false;
                /// The bound the activation is scaled onto for that refresh;
                /// 0 takes silu_bound, which is the range the fit already
                /// covers and so an upper bound on what it can return.
                /// Setting the measured activation peak instead recovers the
                /// log2 of the ratio in bootstrap precision.
                double activation_bound = 0.0;
                /// Hidden GROUPS held at once; 0 takes the whole width.
                ///
                /// The Hadamard product needs both branches in both encodings at
                /// once, and the down projection sums over the hidden axis, so a
                /// chunk can be projected down and accumulated the moment it is
                /// formed and then released. Same products, same scales, only
                /// the association of a homomorphic sum changes.
                int hidden_block_groups = 0;
            };

            /** @brief W_down (SiLU(W_gate x) * W_up x). */
            RectActivation
            feed_forward(RectActivation& x,
                         const RectFeedForwardWeights& weights,
                         const RectFeedForwardConfig& config,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key,
                         Galoiskey<Scheme::CKKS>* boot_key = nullptr,
                         const RectRefreshConfig* refresh = nullptr);

            // ---------------------------------------------------------------
            // The whole block
            // ---------------------------------------------------------------

            /** @brief Plaintext weights of one transformer block. */
            struct RectTransformerBlockWeights
            {
                std::vector<double> attention_norm;
                std::vector<double> feed_forward_norm;
                RectAttentionWeights attention;
                RectFeedForwardWeights feed_forward;
            };

            /** @brief Shape and approximation settings for one block. */
            struct RectTransformerBlockConfig
            {
                RectRMSNormConfig attention_norm;
                RectAttentionConfig attention;
                RectRMSNormConfig feed_forward_norm;
                RectFeedForwardConfig feed_forward;
                RectRefreshConfig refresh;

                /// Multiply the learned RMSNorm scale into the rows of the
                /// projections that read the normalised stream, instead of
                /// applying it homomorphically.
                ///
                /// A pre-norm block feeds its norm output to nothing but a
                /// projection, and the scale is diagonal, so
                /// W^T diag(g) y = (diag(g) W)^T y exactly. On the host that
                /// is a scaling of a plaintext and free; homomorphically it is
                /// a plaintext product and a rescale, once per norm. Two levels
                /// a block for a host multiply, and the only cost is that the
                /// folded weights are a second copy.
                bool fold_norm_scale = true;
            };

            /**
             * @brief Fold a per-row scale into a projection weight, on the
             *        host.
             *
             * @param weight Row major @p in_channels x @p out_channels, as
             *               project() takes it. Row i is scaled by
             *               @p scale[i], which is the channel the learned
             *               RMSNorm gain belongs to.
             */
            static void fold_scale(std::vector<double>& weight,
                                   const std::vector<double>& scale,
                                   int in_channels, int out_channels);

            /**
             * @brief One pre-norm transformer block.
             *
             * Norm, attention, residual; then norm, SwiGLU, residual. The
             * residual needs no encoding move: it reconciles level and scale
             * with a mod drop and a multiplication by a constant, and a constant
             * is the constant POLYNOMIAL, which scales every coefficient of a
             * rectangular encoding exactly as it scales every slot of a slot
             * encoding. The stream therefore stays rectangular from one end of
             * the block to the other and the only crossings are the ones the
             * non-linearities force.
             *
             * The refreshes are the seams named in config.refresh, and they are
             * why the block runs at all: unrefreshed it spends more levels than
             * a chain that also has to hold N/2 - 1 rotation keys can carry.
             * Passing a null @p boot_key runs the block in one stretch, which
             * is the measurement the refreshed one is compared against.
             */
            RectActivation
            transformer_block(RectActivation& x,
                              const RectTransformerBlockWeights& weights,
                              const RectTransformerBlockConfig& config,
                              Galoiskey<Scheme::CKKS>& galois_key,
                              Relinkey<Scheme::CKKS>& relin_key,
                              Galoiskey<Scheme::CKKS>* boot_key = nullptr);

          private:
            /// F and its inverse, as blocked slot diagonals: entry
            /// [eps + k/2 - 1] multiplies the input rotated by eps.
            void build_block_tables();

            /// The whole weight of one projection, arranged as Algorithm 5's
            /// plaintext wants it: block row t at Y^{k-t} with a minus sign.
            std::vector<int64_t>
            rectangular_weight(const std::vector<double>& weight,
                               int in_channels, int out_channels, int in_group,
                               int out_group, double scale) const;

            /// One group's columns borrowed as a BatchActivation, so a bridge
            /// call never deep copies a ciphertext just to name its operands.
            /// The columns are moved out of @p x and must be moved back.
            BatchActivation borrow_group(RectActivation& x, int group);
            void return_group(RectActivation& x, int group,
                              BatchActivation& borrowed);

            /// Refuse columns that have drifted apart in level or scale.
            void require_uniform(const RectActivation& in,
                                 const char* name) const;

            /// Hand the seam's name and the stream's depth to depth_trace, if
            /// one is set. Empty by default, so an unmeasured run pays a null
            /// check per seam and nothing else.
            void note_depth(const char* name,
                            const std::vector<Ciphertext<Scheme::CKKS>>& ct)
                const;

            /// The prime the next rescale of @p ct will divide by.
            double rescale_prime(const Ciphertext<Scheme::CKKS>& ct) const;

            /// Encode @p values at @p scale, dropped onto @p depth.
            Plaintext<Scheme::CKKS> encode(const std::vector<Complex64>& values,
                                           double scale, int depth);

            HEContext<Scheme::CKKS> context_;
            HEEncoder<Scheme::CKKS> encoder_;
            /// Owns the row bridge, the batch matrix operator and the slot-form
            /// arithmetic. Everything Algorithm 4 needs is already in it.
            Llama3BatchOperator batch_;
            BatchMatrixLayout layout_;
            std::vector<Modulus64> primes_;
            int slot_count_;
            double default_scale_;

            /// [2][k - 1] slot vectors: [0] is F, [1] is F^-1, and the second
            /// index is the diagonal offset shifted by k/2 - 1.
            std::vector<std::vector<Complex64>> forward_block_;
            std::vector<std::vector<Complex64>> inverse_block_;
        };

    } // namespace llama
} // namespace heongpu

#endif // HEONGPU_CKKS_LLAMA3_RECT_H
