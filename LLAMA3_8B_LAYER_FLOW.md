# One Llama-3-8B layer on Kang batch CCMM + Kang rectangular PCMM

Validation of a proposed ring-switching execution flow against this
implementation. No Bae PCMM, no MaMBo. Every level in this document is a
reading off `benchmark/profile_llama3_levels.cpp`, not an estimate.

Measured 2026-08-07, Sicily GPU 2 (GPU 0 and GPU 1 were both at ~100%),
logN = 12, d = 128, 20 limbs, scale 2^40, degrees norm 7 / exp 15 / 1-over-x
15 / SiLU 31. Repeating at d = 64 reproduces every row: **a level is a
property of the circuit, not of the shape.**

---

## 1. The measured level ledger

```
convert  RECT -> SLOT   (to_slots)          spends  2    row bridge + block map
convert  SLOT -> RECT   (from_slots)        spends  2    block map + row bridge
convert  RECT -> BATCH  (to_batch)          spends  3    bridge + block map + bridge
convert  BATCH -> RECT  (from_batch)        spends  3    bridge + block map + bridge
convert  BATCH -> SLOT  (row bridge)        spends  1    d rotations, d plaintext products
convert  SLOT -> BATCH  (row bridge)        spends  1    d rotations, d plaintext products
convert  block_map (subring DFT)            spends  1    k-1 rotations, NOT a permutation

product  Algorithm 5   rect PCMM            spends  1    RECT in, RECT out
product  Algorithm 5   again, no crossing   spends  1    chains free
product  Algorithm 3   CMT transpose        spends  0    monomials + tweak + automorphisms
product  Algorithm 4   batch CCMM           spends  1    BATCH in, BATCH out

nonlin   RMSNorm  sublayer, crossings in    spends 10
nonlin   RMSNorm  slot core alone           spends  6    square 1 + mask 1 + fit 3 + product 1
nonlin   SoftMax  slot form, 1 round        spends 11    exp 4 + mask 1 + square 1 + 1/x 4 + product 1
nonlin   SiLU     domain map folded         spends  5    the fit alone
nonlin   SiLU     domain map paid           spends  6    the fit plus its own affine map
nonlin   SwiGLU   gate product              spends  1
nonlin   RoPE     as implemented            spends  1    swap rotation + cos/sin plaintext products
nonlin   residual add                       spends  1
```

### The symbolic depths

| symbol | value | decomposition |
|---|---|---|
| `L_rms1` | **6** | square 1 + blocked-mask 1 + `1/sqrt` fit `ceil(log2(deg+1))` = 3 + rescale product 1 |
| `L_rms2` | **6** | identical implementation |
| `L_rope` | **1** | one rotation + cos/sin plaintext products; the K transpose adds **0** |
| `L_soft` | **11** | exp fit 4 + causal mask 1 + square 1 + reciprocal fit 4 + normalise product 1 |
| `L_swiglu` | **6** | SiLU fit 5 + gate product 1 |

The two RMSNorms are the same code with different calibrated ranges, so they
share one variable: **`L_rms = L_rms1 = L_rms2 = 6`**.

Three of these are degree-dependent and the degree is not free — it is what the
calibrated range and the 12-bit target of Sylph §3.1.2 leave:
`L_rms = 3 + 3`, `L_soft = 2·ceil(log2(deg+1)) + 3`, `L_swiglu = ceil(log2(deg+1)) + 1`.

### The auxiliary-track variants, which are NOT constants

With Sylph Figure 2's narrow track on (`refresh_sum`, `refresh_denominator`) the
fit moves off the wide track and the wide-track cost drops to `L_rms = 2`,
`L_soft = 7`. **Do not put these in a table as constants.** The refreshed
auxiliary ciphertext comes back at a *fixed* depth and the closing product takes
the deeper of the two operands, so what the trick saves depends on where the
wide track was standing when the norm began. `profile_llama3_levels` measures it
at two entry depths for exactly this reason. The numbers above are the honest
schedule-independent ones and are what this document uses.

---

## 2. Encoding dictionary

The proposal's notation, mapped onto what exists here. **Encoding conversion and
packing-layout conversion are not the same operation and this table keeps them
apart.**

| proposal | this code | where the data lives | where the block index rides |
|---|---|---|---|
| `S16` / `S*` Slot | `SLOT` | CKKS slots | fast slot axis, span k/2 |
| `H*` SinC / Kang batch-matrix | `BATCH`, = `BlockAxis::slot` | R_N coefficients (token axis) | R_k **evaluation** axis (batch slots), block b at `zeta^{5^b}` |
| `C*` coefficient | `RECT`, = `BlockAxis::coefficient` | R_N coefficients (token axis) | `Y = X^d` **coefficient** axis |

Both `H` and `C` are coefficient encodings in the token index. They differ only
in where the *block* index sits, and Kang's Algorithm 5 does not treat the two
alike. That distinction is the whole of the conversion cost in this design.

**Verified property of a RECT column, and the key to §5:** `encrypt` writes
`out[j][i + d*t] = e[t]` for `t < k/2` only
(`batchmatrix.cu:288` `build_matrix_encryption_coefficients`), so a rect
column's data occupies **exactly R_N coefficients 0 … N/2−1 and the upper half
is identically zero.**

---

## 3. What the proposal assumes that does not exist here

| assumed | status | evidence |
|---|---|---|
| SinC encoding as a distinct object | **exists**, as `BATCH` / `BlockAxis::slot` | `batchmatrix.cuh:250` |
| rect PCMM: SinC in → coefficient out | **exists** | `rectangular_pcmm(..., BlockAxis::slot)`; "The OUTPUT is always `BlockAxis::coefficient`" |
| rect PCMM depth 1 | **verified, = 1** | measured |
| batch CCMM depth 1 | **verified, = 1** | measured |
| ring switching logN 16 → 12 | **does not exist** | no `Tr`, no ring-switch key, no embedding anywhere in `src/` |
| ring-up / composition 12 → 16 | **does not exist** | same |
| any logN = 16 context on this path | **does not exist** | the rect path is logN = 12 throughout |
| conversion fused into a bootstrap | **does not exist** | `Llama3RectOperator::bootstrap` is a bare call to stock `regular_bootstrapping` |
| a hook to compose a linear map into CtoS/StoC | **does not exist** | `Vandermonde` (`operator.cuh:1793`) is a fixed FFT factorisation with fixed diagonal index tables |

So the flow as written is **not runnable today**. §7 gives the minimum
modifications. Everything below validates the arithmetic on the assumption that
those are built.

---

## 4. The Kang product contracts, verified

**Algorithm 5, rectangular PCMM.** Input `d` ciphertexts; the block axis is a
parameter; **output is always `BlockAxis::coefficient`.** So the proposal's
chain `SinC input → rectangular PCMM → coefficient output` is exactly right and
exactly what `BlockAxis::slot` implements. One level, measured.

This implementation additionally runs **coefficient in → coefficient out**, which
Kang's statement does not require and which is what makes a chain of projections
free (measured: a second Algorithm 5 on the first one's output spends 1 level and
needs no conversion). `Llama3RectOperator::project` hardcodes
`BlockAxis::coefficient`; the `slot` variant the proposal wants is reachable at
the `HEBatchMatrixOperator` layer and just needs exposing. Note Theorem 3's
constant term carries a factor `k/2` in the `slot` variant — a host-side scalar,
free.

**Algorithm 4, batch CCMM.** BATCH in, BATCH out, one level, measured. Its three
internal CMTs are free.

**Algorithm 3, CMT.** **Zero levels**, measured. Monomial multiplications, a
Cooley–Tukey tweak, an exact modular scalar and automorphisms — nothing rescales.
The K transpose in Stage 2 is free.

### Stage 3's no-conversion property: **VERIFIED**

Algorithm 4 returns a column-wise matrix encryption at `d` columns with the
`k/2` blocks on the batch axis. Algorithm 5 with `axis = BlockAxis::slot`
consumes exactly that object. **They are the same encoding; no conversion sits
between the two matrix multiplications.**

At the real dimensions this survives: ScoreV output is `d` rows × `k/2` heads ×
`d` head-dim = a `d × (N/2)` rectangular matrix per group, which is Algorithm 5's
input shape. At logN = 12, H = 4096 is exactly 2 groups, so the O projection is
2 input groups × 2 output groups = 4 Algorithm 5 calls, each fed directly by a
CCMM output. ✔

---

## 5. Fusion analysis

### The rule, derived from the bootstrap's own structure

`regular_bootstrapping` (`operator.cu:7041`) is

```
ModRaise  ->  CoeffToSlot  ->  EvalMod  ->  SlotToCoeff
```

A linear map can be absorbed at **exactly two places**, and both are ends:

* **(a)** immediately *before* CoeffToSlot — fold `M^-1` into the CtoS diagonals;
* **(b)** immediately *after* SlotToCoeff — fold `M` into the StoC diagonals.

Nothing may sit between the conversion and the bootstrap. This is precisely the
proposal's own warning, and it is structural rather than a limitation of this
codebase.

Two further conditions, from `Vandermonde`:

* absorbing `M` is **free** only if `M` is compatible with the radix structure of
  the existing factorisation (a stride permutation, or a stage of the same DFT);
* a *general dense* `M` needs its own factorisation stage, which costs the same
  one level it would have cost standalone. **Fusion of a dense map saves nothing.**

### The structural result: RECT ↔ SLOT *is* CoeffToSlot

`to_slots(RECT)` = row bridge (a `d`-point Vandermonde along the token axis)
then `block_map` (a `k/2`-point subring DFT along the block axis). `d · k/2 = N/2`.
**Those two stages composed are the N/2-point DFT — the same transform
CoeffToSlot computes.** And §2 established that a rect column's data is exactly
R_N coefficients `0 … N/2−1`, which is exactly what CtoS's `result0` carries
(`result0 = Re(V^-1 · c)` is the lower coefficient half; `result1`, the upper
half, is **identically zero** for a rect column).

Therefore: **a bootstrap of a RECT column that stops after EvalMod and omits
SlotToCoeff returns the SLOT reading of that column.** The 2-level crossing is
not merely fusible — it is already inside the bootstrap. And omitting StoC hands
back `StoC_piece = 3` *more* levels, and halves EvalMod (one live ciphertext
instead of two).

#### Now measured, and the fix-up is **not** the stride permutation

`HEArithmeticOperator::coeff_to_slot_bootstrapping` implements exactly the above
— ModRaise, `solo_coeff_to_slot`, EvalMod, no stage 4 — and
`benchmark/profile_boot_to_slots.cpp` runs it against `bootstrap()` + `to_slots()`
at logN 12, d 64, 36 limbs. Both claims about *value* hold:

| | measured |
|---|---|
| levels, `bootstrap()` then `to_slots()` | depth **27** |
| levels, `bootstrap_to_slots()` | depth **21** |
| **levels returned** | **6** (StoC_piece 3 + its rescale, + the 2-level crossing) |
| time, per column | **221 ms** vs 526 ms |
| every value present, worst match | **7.4e-6** — the data is all there |

**The index order is bit reversal, not the stride permutation.** Matching all
2048 values by magnitude: `identity 64, stride 64, bit reversal 2037, 5-power 2`
(the 11 misses are pairs of random draws closer together than the bootstrap's own
7e-6 error — a birthday effect at 2048 samples, not exceptions). `perm[2^j]`
came back `1024, 512, 256, …, 1`. That is what a decimation-in-time
factorisation leaves behind, and CoeffToSlot's is one. The reference path
confirms the other half: `to_slots()` matches the natural order under the stride
transpose to **1.8e-5**.

So the two readings differ by **bit reversal ∘ stride transpose**, and the
earlier guess ("a `d × (k/2)` stride permutation, which a DFT factorisation
absorbs") was half right — the stride part is the crossing's, the reversal is the
bootstrap's, and only the first was accounted for.

#### The reversal never has to be undone — measured

The earlier conclusion here was that the 6 levels were real but not spendable,
and that the fix was to fold the reversal into `Vandermonde`'s CtoS diagonals.
**That was the wrong fix.** The reversal does not need removing, because the way
back out of the island is not `from_slots` but the bootstrap's own fourth stage:

```
refresh_to_slots   =  ModRaise -> CoeffToSlot -> EvalMod          (stops)
   ... slot work, on the reversed index ...
slots_to_rect      =  SlotToCoeff                                 (the inverse)
```

`SlotToCoeff` is the *same* Vandermonde run backwards, so whatever relabelling
the forward map applied it inverts by construction — no agreement about twiddle
conventions, no bit-reversal bookkeeping, nothing to name. `benchmark/profile_slot_island.cpp`
measures it at logN 12, d 64, 36 limbs:

| | measured |
|---|---|
| round trip vs input | gain **1.000002**, worst **8.78e-6**, rel 2.19e-5 |
| a plain `bootstrap()` vs input (the floor) | gain 1.000002, worst **8.76e-6**, rel 2.19e-5 |
| depth, island: refresh → close | 0 → **21** → **25** |
| depth, today: bootstrap → `to_slots` → `from_slots` | 0 → 25 → 27 → **29** |
| time, 64 columns | refresh 14121 ms + close 3668 ms = **17789 ms**; `bootstrap()` alone 22017 ms |

**The round trip is the identity, to the bootstrap's own error and no worse.**
Four levels come back on the full crossing, and — the number that matters when
the island is deep — the slot work starts at depth 21 instead of 27, so it has
**six more levels of headroom**.

Two things had to be found by measurement, and both would have been silent bugs:

* **The alignment drop is exactly one level.** The pair form of `slot_to_coeff`
  spends a level folding in the imaginary half and drops the real half to match,
  and the diagonals are encoded for the level *after* that drop
  (`generate_encoding_transform_context` encodes at `StoC_start_level + 1`). The
  solo form has no imaginary half, so nothing spends that level for it. Swept:
  drop 0 → `max|slot| 1.9e168`, drop 2 → `1.5e138`, drop 1 → correct. The library
  raises **no error** on the wrong one; `multiply_matrix` just slices the
  diagonal buffer at the wrong stride.
* **The diagonals are level-locked**, which makes the fast path correct only for
  an island in which nothing was spent — i.e. useless. `slots_to_coeff_at_level`
  builds a second set at the caller's level, cached, at the same `StoC_piece` and
  `less_key_mode` so `key_indexs_` is unchanged and **the existing boot key still
  covers it**.

**What the reversal does and does not cost.** Worked out on the index and then
confirmed: the rect encoding puts token `i` of block `t` at coefficient `i + d*t`
— token in the low `log2(d)` bits. `to_slots` transposes to slot `t + (k/2)*i`.
Reversing the whole index reverses the bit string, which swaps the two fields
*and* reverses each, giving slot `revbits(t) + (k/2)*revbits(i)`. **The field
assignment is therefore the same as `to_slots`'; only the order within each field
differs.** So:

* a reduction over one **whole** field is unchanged — `sum_blocked` of span `k/2`
  sums an aligned run of slot *positions*, and a block's positions are the same
  set however the data inside them is ordered. RMSNorm's one reduction is exactly
  that and needed **no reformulation**;
* a plaintext constant that varies along a field is written at the reversed
  index — host-side relabelling, free, and it is the single line `island_slot()`;
* a rotation by **less** than a whole field — a token shift, the SoftMax's row
  alignment — is *not* preserved, because rotation is additive on the index and
  bit reversal is not. Those stay on `to_slots`.

Measured: `island_slot(b, u, island=true)` matches the decoded island to
**7.2e-6**, the natural reading to rel **25** (garbage), and `to_slots` matches
the natural reading to **7.3e-13**.

#### A whole RMSNorm through the island, against the plaintext answer

`RectRMSNormConfig::fused_refresh` puts the norm on the island: `refresh_to_slots`
in place of `bootstrap()` + `to_slots()`, the same reduction and the same fit
untouched, weights placed through `island_slot()`, and `slots_to_rect_at_level`
in place of `from_slots()`. 2048 channels, degree-15 fit, 36 limbs:

| path | result |
|---|---|
| **B, the island** | depth out **33**, 15710 ms, **gain 1.000000, rel 1.99e-5** vs the plaintext norm |
| **A, today's** | **does not fit in 36 limbs** |

Path A needs `25 (bootstrap) + 2 (to_slots) + 8 (norm) + 2 (from_slots) = 37`.
It runs out of chain mid-transform, and HEonGPU reports that as a CUDA launch
failure from the NTT rather than as a level error — worth knowing, because it
does not look like what it is. Path B needs `21 + 8 + 4 = 33`.

So on this shape the six levels of headroom are not an optimisation, they are
**the difference between the sublayer fitting and not fitting**. And the
reduction needed no reformulation, which is the part of the index argument above
that was load-bearing.

### The five boundaries

Every boundary in the proposal has the shape

```
nonlinear (Slot)  ->  [Slot->SinC]  ->  Kang product  ->  Bootstrap
```

so the conversion is separated from the bootstrap **by the product**.

| # | boundary | fused? | why |
|---|---|---|---|
| 1 | RMSNorm → S2H → rect PCMM → BTS | **NO** | conversion before the product, bootstrap after it |
| 2 | RoPE/transpose → S2H → batch CCMM → BTS | **NO** | same |
| 3 | Softmax → S2H → batch CCMM → rect PCMM → BTS | **CONDITIONAL** | see below |
| 4 | RMSNorm → S2H → rect PCMM → BTS | **NO** | same as 1 |
| 5 | SwiGLU → S2H → rect PCMM → BTS | **NO** | same as 1 |

Per-boundary answers to the specific questions:

* **Is SlotToSinC standalone or fused?** Standalone at 1, 2, 4, 5. The needed map
  is `SLOT → BATCH`, the row bridge alone — **measured at 1 level**, matching the
  proposal's expectation. (It is *not* the 2-level `to_slots`/`from_slots` pair
  the current rect path uses, because Algorithm 5 accepts the batch axis directly.)
* **If fused, which bootstrap?** Only at boundary 3, and only into the refresh
  that already sits between the SoftMax and the value product. In the implemented
  schedule that refresh exists (`RectRefreshConfig::post_softmax`, taken on P and
  V together because they meet in the product) and the crossing follows it
  immediately. That is case (b): the conversion is adjacent to a bootstrap, so
  `C_s2h = 0` there.
* **Is the conversion before or after the nonlinear?** After, at all five.
* **Does the operator need SinC before the bootstrap occurs?** Yes, at all five.
  That is exactly why 1, 2, 4 and 5 cannot fuse.
* **Is the operator being inserted into the bootstrap, or only the conversion?**
  Only the conversion, at boundary 3. Inserting the *product* into the bootstrap
  is the Bae/MaMBo construction and is excluded.
* **Supported by Kang, or a new fused algorithm?** **A new fused algorithm.**
  Kang's Algorithms 3, 4 and 5 say nothing about bootstrapping. Everything in §5
  is a modification of HEonGPU's `Vandermonde`/`coeff_to_slot`/`slot_to_coeff`.

### Where fusion *is* free, and it is the other side

The conversion the proposal charges nothing for — `coefficient output →
Bootstrap → fresh Slot` — is case (b) and it is **genuinely free**, and more than
free: it is StoC omitted, worth `+3` levels. That is the one real win and the
proposal already assumes it, correctly. The conversion it hopes to fuse
(`Slot → SinC`, before the product) is the one that cannot be.

---

## 6. The flow, validated

### Level budget, exactly as proposed

`C_s2h = 1` (measured, standalone, the row bridge) at boundaries 1, 2, 4, 5;
`C_s2h = 0` at boundary 3 if the post-SoftMax refresh carries it.

| stage | input | nonlinear | `C_s2h` | RS | CCMM | PCMM | **start level** | pre-BTS | BTS out |
|---|---|---|---|---|---|---|---|---|---|
| 1 QKV | `S16` | `L_rms` = 6 | 1 | 0 | – | 1 | **8** | `C12(0)` | `S16(L_top)` |
| 2 QK^T | `S16` | `L_rope` = 1 (+ CMT 0) | 1 | 0 | 1 | – | **3** | `H12(0)` | `S16(L_top)` |
| 3 PV + O | `S16` | `L_soft` = 11 | 0 fused / 1 not | 0 | 1 | 1 | **13** fused / **14** not | `C12(0)` | `S16(L_top)` |
| 4 gate/up | `S16` | `L_rms` = 6 | 1 | 0 | – | 1 | **8** | `C12(0)` | `S16(L_top)` |
| 5 down | `S16` | `L_swiglu` = 6 | 1 | 0 | – | 1 | **8** | `C12(0)` | `S16(L_top)` |

**The extra level is not hidden: Stage 3 needs 14, not 13, unless the
post-SoftMax refresh absorbs the crossing.** Stages 1, 2, 4 and 5 each pay one
standalone level for `Slot → SinC` and there is no construction that removes it.

### Complete state transition, one layer

```
                                                     logN  enc    layout                       lvl
  in                                                  16   Slot   token x channel, T=128        14
  |
  +-- residual stream held ------------------------------------------------------------------> 14
  |
  [STAGE 1]  RMSNorm  (L_rms = 6)                     16   Slot                                  8 -> 2
             SlotToSinC  row bridge  (C_s2h = 1)      16   SinC   block on batch axis            2 -> 1
             RingSwitch 16 -> 12     (0)              12   SinC                                  1
             Alg 5  QKV, 12 calls    (1)              12   Coeff  block on Y axis                1 -> 0
             ring-up 12 -> 16, BTS, StoC omitted      16   Slot                                 -> 14
  [STAGE 2]  RoPE on Q,K  (L_rope = 1)                16   Slot                                  3 -> 2
             SlotToSinC              (1)              16   SinC                                  2 -> 1
             RingSwitch              (0)              12   SinC                                  1
             Alg 3  CMT on K         (0)              12   SinC                                  1
             Alg 4  Q K^T            (1)              12   SinC                                  1 -> 0
             ring-up, BTS, StoC omitted               16   Slot                                 -> 14
  [STAGE 3]  SoftMax  (L_soft = 11)                   16   Slot                                 14 -> 3
             SlotToSinC   (1, or 0 fused)             16   SinC                                  3 -> 2
             RingSwitch              (0)              12   SinC                                  2
             Alg 4  P V              (1)              12   SinC                                  2 -> 1
             Alg 5  W_o, 4 calls     (1)              12   Coeff                                 1 -> 0
             ring-up, BTS, StoC omitted               16   Slot                                 -> 14
             residual add            (1)              16   Slot                                 -> 13
  [STAGE 4]  RMSNorm  (L_rms = 6)                     16   Slot                                 13 -> 7
             SlotToSinC              (1)              16   SinC                                  7 -> 6
             RingSwitch              (0)              12   SinC                                  6
             Alg 5  gate+up, 28 calls (1)             12   Coeff                                 6 -> 5
             ring-up, BTS, StoC omitted               16   Slot                                 -> 14
  [STAGE 5]  SwiGLU  (L_swiglu = 6)                   16   Slot                                 14 -> 8
             SlotToSinC              (1)              16   SinC                                  8 -> 7
             RingSwitch              (0)              12   SinC                                  7
             Alg 5  W_down, 14 calls (1)              12   Coeff                                 7 -> 6
             ring-up, BTS, StoC omitted               16   Slot                                 -> 14
             residual add            (1)              16   Slot                                 -> 13
  out                                                 16   Slot                                 13
```

Compressed:

```
S16(14) -> S16(2) -> H16(1) -> H12(1) -> C12(0) -> BTS -> S16(14)      [stage 1]
S16(14) -> S16(2) -> H16(1) -> H12(1) -> H12(0) -> BTS -> S16(14)      [stage 2]
S16(14) -> S16(3) -> H16(2) -> H12(2) -> H12(1) -> C12(0) -> BTS -> S16(14)   [stage 3]
S16(13) -> S16(7) -> H16(6) -> H12(6) -> C12(5) -> BTS -> S16(14)      [stage 4]
S16(14) -> S16(8) -> H16(7) -> H12(7) -> C12(6) -> BTS -> S16(14)      [stage 5]
```

Stages 4 and 5 leave levels on the floor because Stage 3 sets `L_top`. Slack, not
a bug: `L_top` is one number for the whole layer.

### Tiling for the real shapes, logN = 12

`d = head_dim = 128` is forced — Algorithm 4 contracts over the column index and
leaves the batch index alone, so a head must be exactly one channel block. Then
`k = N/d = 32`, `k/2 = 16` blocks per group, `N/2 = 2048` channels per group,
`d = 128` ciphertexts per group, `T = 128 = d` rows. Every 8B width divides
exactly: **no padding at all.**

| projection | shape | in groups | out groups | Alg 5 calls | ciphertexts out |
|---|---|---|---|---|---|
| QKV | 128×4096 · 4096×6144 | 2 | 3 (12288/2048 = 6 with host GQA expansion) | 6 (**12** as implemented) | 384 (768) |
| O | 128×4096 · 4096×4096 | 2 | 2 | 4 | 256 |
| gate/up | 128×4096 · 4096×28672 | 2 | 14 | 28 | 1792 |
| down | 128×14336 · 14336×4096 | 7 | 2 | 14 | 256 |

**58 Algorithm 5 calls per layer** as implemented (52 without the GQA
expansion). Grouped-query attention is expanded on the host because the batch
slots of a matrix encryption never talk to each other; it costs projection work
in the ratio `heads/kv_heads` and buys nothing.

Why the products must be at the small ring: Algorithm 5's CMT runs the
automorphisms over the whole rotation group, so it needs **N/2 − 1 Galois keys —
a count that depends on the ring alone.** 2047 at logN = 12; **32767 at
logN = 16**, which is tens of GB of keys and does not fit any card. That, and
nothing else, is why this design ring-switches down.

Ciphertexts per bootstrap point, and the composition ratio `N16/N12 = 16`:

| stage | ct at logN 12 | ct at logN 16 after composition |
|---|---|---|
| 1 | 768 | 48 |
| 2 | 256 | 16 |
| 3 | 256 | 16 |
| 4 | 1792 | 112 |
| 5 | 256 | 16 |
| **total** | 3328 | **208** |

---

## 7. The low-ring modulus: is 41 + 33 + 33 valid?

**1. What role does the 41-bit prime play?** It is `q0`, the **bottom** prime —
the base modulus a bootstrap's EvalMod is parameterised around. Not a top
modulus, not the scale. `(q0, p) = (41, 33)` is the measured minimum for
**`regular_bootstrapping_v2`** at logN ≤ 14, where `q0` must be *exactly* `p + 8`
(message ratio 256): `q0 = p+7` returns garbage silently, and `q0 > p+8` buys
nothing.

**2. Do two 33-bit primes give exactly the levels needed?** **Yes, exactly.**
`Q = q0 + 2` working primes → 2 rescales. Stage 3 needs 2 (CCMM 1 + PCMM 1);
Stages 1, 2, 4, 5 need 1. So the set is sized on the deepest low-ring stage and
one prime more than the other four need.

**3. Scale compatibility.** A rescale divides by the prime, so the working scale
must be ≈ 2^33 at every one of the three boundaries:
* *SlotToSinC*: the row bridge encodes its diagonals at the prime the following
  rescale divides by (`rescale_prime`), so the scale is preserved. ✔
* *Batch CCMM*: leaves `scale_a · scale_b` and marks the rescale, which
  `Llama3BatchOperator::product` spends immediately. ✔
* *Rect PCMM*: identical discipline, and it is load-bearing — left unspent, the
  plaintext's centred coefficients outgrow `int64` and the next call cannot read
  them back. ✔

*Precision*, though, is tight rather than comfortable. Δ = 2^33 against a d = 128
contraction (7 bits) plus weight quantisation leaves roughly 8 bits of margin
over the 12-bit target of §3.1.2. It works; it is a floor, not headroom. Separate
issue: with `Q = 107` bits the product of active primes exceeds `int64`, so
`extract_coefficients` (Garner into `int64`) overflows — a **read-back** limit,
not a compute limit, but it breaks the decrypt path the tests use.

**4. Modulus alignment across the switch.** A ring switch is
modulus-preserving, so the logN = 16 chain must **end** with exactly these three
primes. Since the ciphertext reaches the switch at `C12(0)` — one prime left —
that prime *is* the bootstrap's `q0`. So `q0 = 41` at logN = 16 too, which forces
**v2**, and v2's precision at logN = 16 is a measured **16.75 bits** (it is a
cliff: 23.97 / 23.40 / 20.49 / 17.40 / 16.75 at logN 12…16). That clears Sylph's
12-bit requirement and misses 20 bits. `regular` at logN = 16 peaks at
`(q0, p) = (60, 56)` for 20.12 bits — but 60 + 56 + 56 is a different low ring.
**This is the one real internal conflict in the proposal and it must be decided,
not averaged.**

**5. Key switching and ring-switching keys.** Algorithm 5's CMT needs 2047 Galois
keys at logN = 12. Ring switching needs its own switch key in each direction and
the two rings' secrets must be related (`s_12(X) = s_16(X^16)` for the embedding;
a trace for the descent). **Neither key nor either direction exists.**

**6. Special P — read from the validator, not assumed.** `P` is whatever the
caller passes; `set_coeff_modulus_bit_sizes` sets
`KEYSWITCHING_METHOD_I` when `|P| == 1` and II otherwise, and
`coefficient_validator` (`util.cu:11`) is the only structural constraint: it cuts
`Q` into consecutive chunks of exactly `|P|` primes and demands **each chunk's
bit-sum ≤ Σ log P**. So `dnum = ceil(|Q| / |P|)`, and the minimum legal `P` is
the largest chunk.

For `Q = {41, 33, 33}` that gives three options, and only one of them is the
`log PQ ≈ 214` this document previously quoted:

| `\|P\|` | dnum | chunks | min log P | log PQ |
|---|---|---|---|---|
| 1 | 3 | {41} {33} {33} | 41 | **148** |
| 2 | 2 | {41,33} {33} | 74 | 181 |
| 3 | 1 | {41,33,33} | 107 | 214 |

**214 was a `dnum = 1` figure, and `dnum = 1` was a key-budget choice, not a
requirement of the primes.** The honest minimum is **148**.

**7. 128-bit security at logN = 12 — still NO, by 39 bits, not by 105.**
`context.cu:113` checks `Σ log Q + Σ log P` against `heongpu_128bit_std_parms(N)`,
which is **109** at N = 4096 (`secstdparams.h:29`). The best case above is 148.
No choice of `dnum` rescues it.

What N = 4096 *can* hold, uniform primes of `b` bits, `L` of them, `|P| = 1`
(which is optimal — a larger `|P|` forces a larger `P`), so `(L+1)·b ≤ 109`:

| levels | primes | scale | log PQ | verdict |
|---|---|---|---|---|
| **1** | 2 × 36, P = 36 | 2^36 | 108 | ✔ comfortable |
| **2** | 3 × 27, P = 27 | 2^27 | 108 | ✔ tight |
| 3 | 4 × 21, P = 21 | 2^21 | 105 | ✔ useless |

Kang's own S12 sits in the first row (`log Q = 36 + 28`, `log QP = 104`) and their
first three-level parameter set, S13b, is at **N = 8192** (`log QP = 160`). They
move rings exactly where this table says they must.

**So the real constraint is not "41+33+33 is insecure" but "N = 4096 holds ONE
comfortable multiplicative level".** That reframes the design question, and §7bis
answers it with measurement.

The 27-bit two-level row is not obviously dead but it is **not verified**: Δ = 2^27
against a d = 128 contraction (7 bits of growth per product, twice) leaves roughly
13 bits over the 12-bit target of §3.1.2 with nothing for input quantisation. No
measurement of product precision at 27 bits exists in this tree.

Separately, and applying to the **whole existing pipeline** and not only to this
proposal: every profile here builds its context with `sec_level_type::none` and a
**sparse secret of Hamming weight 16**. The 109 in that table is for a *ternary
uniform* secret; a weight-16 secret at N = 4096 is weaker, not stronger, against
hybrid dual attacks. These are honest **cost models**, not deployable parameter
sets, and nothing in this document changes that.

**8. Are the total Q and P valid at N = 2^12?** `Q = 107 ≤ 109` on its own, but
the check is on `PQ` and the smallest legal `P` is 41. Not valid.

## 7bis. Does ring switching actually make the products cheaper? **No.**

The premise behind the whole low-ring island is that a matrix product wants the
smallest ring it can get. `benchmark/profile_ring_cost.cpp` tests it by dividing
a call's wall time by the matrix work that call performs, which is the only way
to compare rings — a smaller ring makes each call cheaper *and* makes each call
do proportionally less. The two algorithms do different amounts of work per call
and the difference is a factor of `k/2`, so they are normalised separately:

* `ccmm` multiplies two `d × d` encryptions, `k/2` batched: `(k/2)·d³ = N·d²/2`.
* `rectangular_pcmm` multiplies a `d × (N/2)` encryption by an `(N/2) × (N/2)`
  plaintext: `(k/2)²·d³ = N²·d/4`.

A6000, GPU 2, `d = 128`, 4 limbs, one 45-bit special prime, best of 3:

| logN | ccmm ns/MAC | rel | rect pcmm ns/MAC | rel | Galois keys for Alg 5 |
|---|---|---|---|---|---|
| 12 | 1.642 | 1.00× | 1.821 | 1.00× | 2047 |
| 13 | 0.973 | **0.59×** | 1.616 | **0.89×** | 4095 |
| 14 | 0.841 | **0.51×** | *OOM* | — | 8191 |
| 15 | 0.813 | 0.50× | — | — | 16383 |
| 16 | 0.837 | 0.51× | — | — | 32767 |

**Both products are cheaper per unit of matrix work at a LARGER ring.** The CCMM
is 2× cheaper at logN ≥ 14 than at logN 12 and then flat; the rect PCMM is 1.12×
cheaper at logN 13 than at logN 12. Ring switching *down* does not reduce the
cost of either — it increases it.

That is not a contradiction of the theory, it is the theory. A key switch costs
`O(N log N)` per limb pair, and the key-switch **count** per call is `~4d` for
`ccmm` and `~N/2` for `rectangular_pcmm`; divide either by that call's MACs and
the ring cancels, leaving only `log N` and device occupancy. Occupancy is what
the table is actually showing: at logN 12 a CCMM issues 512 key switches on
4096-coefficient ciphertexts and is launch-bound, and the small ring loses.

### So why a low ring at all? Feasibility, not speed

`rectangular_pcmm` needs `N/2 − 1` Galois keys — a count that depends on the ring
**alone**. The sweep did not stop at logN 14 by choice: it died with
`rmm::out_of_memory` trying to grow past a 40.8 GiB pool for 8191 keys at four
limbs. At logN 16 the same set is 32767 keys and around 172 GB. **Algorithm 5
cannot be keyed at a high ring at any speed**, and that — not throughput — is the
entire argument for the low ring.

`ccmm` has no such problem: its three CMTs are all at layout `(N, d)` and ask for
`d` keys, independent of `N`.

### Which changes the better option

The proposal ring-switches **Stage 3 wholesale** — CCMM then rect PCMM, two
levels, both low. The measurement says to split them:

* **Batch CCMM stays high.** It has no key-count constraint and is 2× cheaper per
  MAC there. Ring-switching it down pays a conversion to make it slower.
* **Only Algorithm 5 goes low**, because only Algorithm 5 has to.

And Algorithm 5 is **one level**. So the low-ring island needs exactly one
multiplicative level — which is precisely the row §7's table says N = 4096 *can*
carry at 128-bit security, at a 2^36 scale, and it is the regime Kang's own S12
parameter set sits in. **The two-level requirement that made 41 + 33 + 33
impossible at logN 12 was an artefact of pairing the two products in the same
island, and it dissolves when they are separated.**

Two things this does not settle, and they are the cost of the split:
1. §4's no-conversion property between Algorithm 4 and Algorithm 5 holds *at one
   ring*. A ring switch between them re-introduces a boundary, and whether the
   descent preserves a matrix encryption is unverified — ring switching still
   does not exist in either direction.
2. `Llama3RectOperator::project` hardcodes `BlockAxis::coefficient`, so the
   CCMM → PCMM hand-off is not reachable from the wrapper today regardless.

### Closest valid alternative

**Move the low ring to logN = 13.** `log PQ ≤ 218` at N = 8192:
`Q = 41 + 33 + 33 = 107`, `P = 60` ⇒ `log PQ = 167`. ✔ Secure, same three primes,
same two levels, same v2 bootstrap regime.

What it costs and what it buys:

| | logN 12 | logN 13 |
|---|---|---|
| Galois keys (N/2 − 1) | 2047 | 4095 |
| key bytes at Q = 3, dnum = 1 | 384 KB → **786 MB total** | 786 KB → **3.1 GB total** |
| `k/2` blocks per group | 16 | 32 |
| channels per group | 2048 | 4096 |
| groups for H = 4096 / I = 14336 | 2 / 7 | 1 / 4 |
| **Algorithm 5 calls per layer** | **58** | **16** |
| padding waste on I | 0% | 12.5% |
| 128-bit secure | **no** | **yes** |

The short chain is the whole point: the current path carries 2048 keys at
**9.25 GiB** because its chain is 36 limbs. At three limbs the same key set is
under a gigabyte, and logN = 13 costs 3.1 GB while cutting the Algorithm 5 call
count by 3.6×. **logN = 13 is strictly better than logN = 12 here on every axis
except padding.**

---

## 8. Bootstrap requirements, per bootstrap

All five are identical in form; only the ciphertext count differs (§6).

| property | value |
|---|---|
| input logN | **16** — after composition; the ring-up is **required**, it is not optional |
| input encoding | `C16` coefficient (`RECT`), or `H16` SinC out of Stage 2 |
| input level | exactly **1 prime remaining**. `regular_bootstrapping` throws `"Ciphertexts leveled should be at max!"` otherwise |
| input modulus | `q0` alone = the 41-bit prime |
| ring-up needed first | **yes**: 16 logN-12 ciphertexts compose into one logN-16 ciphertext |
| C2S/S2C executed | ModRaise → **full CoeffToSlot** → EvalMod → **SlotToCoeff OMITTED** |
| output encoding | **Slot** — because StoC is omitted, so the ciphertext still holds the input's coefficients in slots |
| output packing layout | token on the slow slot axis, channel block on the fast one, after the `d × (k/2)` stride permutation |
| restored level | `L − (CtoS_piece + taylor + StoC_piece + 8)`; with StoC omitted, `L − (3 + 11 + 0 + 8) = L − 22`, i.e. **3 levels more than a full bootstrap** |
| fused conversion | **yes** — the coefficient→slot crossing, worth 2 levels, at no cost. This is the only fusion in the design |

Stage 2's bootstrap takes `H12(0)` (SinC) rather than `C12(0)`. The block index
is on the batch axis there, so its post-EvalMod slot reading differs from the
rect one by the block DFT — the `block_map`, 1 level, which is **not** a
permutation and so is **not** absorbable. Either pay it, or convert
`H12(0) → C12(0)` before the switch. There is no level to pay it with at depth 0,
so **Stage 2 must hand the CCMM output to Algorithm 5's coefficient axis, or
budget one more level.**

---

## 9. Answers

**1. Bootstraps per layer:** **5** — one closing each stage. No sixth is needed
for the residual stream: it is held at `L_top` across the attention half and
`residual_add` (1 level, measured) leaves 13, which clears Stage 4's 8.
In ciphertexts: **3328 at logN 12, 208 after composition to logN 16.**

**2. Maximum level required at logN = 16:** **14** (13 with the boundary-3
fusion), set by Stage 3 = `L_soft + C_s2h + 2`. With the auxiliary track under
the SoftMax it would be `7 + 1 + 2 = 10`, but see §1 on why that is not a
constant.

**3. Levels required at logN = 12:** **2**, set by Stage 3 (CCMM 1 + PCMM 1).
The other four stages need 1. Hence three ciphertext primes.

**4. Does 41/33/33 work?** The *level* count: yes, exactly. The *bootstrap*
role: yes, and only under `regular_bootstrapping_v2`, whose `q0 = p + 8` wall the
pair sits on precisely. The *security*: **no at logN = 12** — `log PQ ≈ 214`
against a bound of 109. The *precision*: 16.75 bits at logN = 16, which meets 12
and misses 20. **Closest valid alternative: the same three primes at logN = 13**,
which is secure, cheaper in Algorithm 5 calls, and costs 3.1 GB of keys.

**5. Where fusion is possible:**
* every `coefficient → Slot` conversion *after* a bootstrap — free, and worth
  `+3` levels because StoC is omitted rather than composed (5 of these per layer);
* boundary 3's `Slot → SinC`, into the post-SoftMax refresh that the schedule
  already places there.

**6. Where fusion is impossible:** boundaries 1, 2, 4 and 5. The conversion is
before the product and the bootstrap is after it, and only the Bae/MaMBo
construction closes that gap. Also impossible: absorbing the `block_map` into
any bootstrap — a subring DFT is not a permutation and does not commute with the
slot-wise work around it.

**7. Where an extra level must be allocated:** one at each of stages 1, 2, 4, 5
for the standalone `Slot → SinC` row bridge; one at stage 3 unless the
post-SoftMax refresh carries it; one at Stage 2's exit if the CCMM output is not
handed to Algorithm 5's coefficient axis. **Total unavoidable: 4 per layer.**

**8. Is the full flow implementable with Kang batch CCMM + Kang rectangular PCMM
and no Bae PCMM?** **The matrix algebra: yes, completely.** Both products are
verified at depth 1, the CMT is free, the two products' encodings are directly
compatible, projections chain with no conversion, and the real 8B widths tile
with zero padding at logN = 12. Nothing in the layer needs a Bae-style PCMM.

**The surrounding machinery: no, not today.** Three things must be built, and
they are all outside Kang:

1. **Ring switching, both directions**, with its keys and a compatible secret
   pair. This is the largest piece and it does not exist in any form.
2. **A bootstrap that omits SlotToCoeff** and applies the `d × (k/2)` stride
   permutation, so the C→S crossing is fused. Buildable from the existing
   separately-callable `mod_raise` / `coeff_to_slot` / `exp_scaled` pieces plus an
   index-table change in `Vandermonde`.
3. **Exposing `BlockAxis::slot` on `Llama3RectOperator::project`**, so
   `Slot → SinC` is the 1-level row bridge rather than the 2-level
   `to_slots`/`from_slots` pair. This is a signature change and a constant.

Item 3 is an afternoon. Item 2 is a week. Item 1 is the project.

---

## 10. Minimum modifications, in dependency order

| # | change | size | unblocks | status |
|---|---|---|---|---|
| 1 | expose `BlockAxis` on `Llama3RectOperator::project` | trivial | `C_s2h = 1` instead of 2 | open |
| 2 | move the low ring to **logN = 13**, or split Stage 3 and stay at 12 | parameter | 128-bit security | open, see §7bis |
| 3 | decide the bootstrap model: **v2 at (41,33) for 12-bit**, or **regular at (60,56) for 20-bit** with a different low ring | decision | resolves the §7.4 conflict | open |
| 4 | bootstrap without SlotToCoeff, and a way back that consumes its order | new algorithm | 4 levels on the crossing, 6 of headroom | **done, measured** |
| 5 | ring switch down (`Tr`) and up (embed + compose), with keys | new subsystem | the whole flow | open |

Item 4 landed as `coeff_to_slot_bootstrapping` / `refresh_to_slots` on the way in
and `slots_to_coeff_at_level` / `slots_to_rect_at_level` on the way out, wired
into `RectRMSNormConfig::fused_refresh` and measured by
`benchmark/profile_slot_island.cpp`; see §5. Its shape changed from the plan: the
"stride permutation" it was going to need does not exist, because the return
transform consumes the forward transform's own order.

Until 5 exists, this flow cannot run. Everything above it is worth doing anyway:
1 and 2 are improvements to the path that runs today, and 4 already is one.
