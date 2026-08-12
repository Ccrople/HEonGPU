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
| conversion fused into a bootstrap, RECT | **exists, measured** | `bootstrap_to_slots` / `slots_to_rect_at_level`; §5 |
| conversion fused into a bootstrap, BATCH | **impossible without the hook below** | measured: the truncated bootstrap returns *coefficients*, and a Kang matrix entry is an R_k **slot**, not a coefficient; §11 |
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

---

## 11. The QK → SoftMax seam, traced and measured

Target: `benchmark/profile_attention_layout.cpp`. Parameters: A6000, logN 12,
d = 64, 36 limbs (60 + 35 × 50), scale 2^50, CtoS 3 / StoC 3 / taylor 11,
78 rotation indices. Everything below is a reading of the decryption, not of a
function name.

### 11.1 The layout, boundary by boundary

Shape at logN 12, d = 64: **k/2 = 32 heads of 64 × 64 in one Algorithm 4 call**,
64 ciphertexts, 2048 useful values each. At logN 13, d = 128 the same call
carries **32 heads of 128 × 128** — the real 8B attention shape exactly, in one
call, 128 ciphertexts of 4096 values.

| boundary | ring | encoding | cts | useful/ct | depth | permutation |
|---|---|---|---|---|---|---|
| RMSNorm out | logN 12 | RECT | `heads·d / (N/2)` groups × d | N/2 | ⊥ | — |
| Q, K, V (Alg 5) | logN 12 | RECT | same | N/2 | +1 | — |
| `to_batch` | logN 12 | BATCH | d per group | (k/2)·d | +3 | — |
| K^T (Alg 3, CMT) | logN 12 | BATCH | d | (k/2)·d | +0 | — |
| S = Q K^T (Alg 4) | logN 12 | BATCH | d | (k/2)·d | +1 | — |
| `to_slots` (bridge) | logN 12 | SLOT | d | (k/2)·d | +1 | **identity, 2048/2048** |
| post-QK bootstrap | logN 12 | SLOT | d | (k/2)·d | → 25 | none (full BTS is an identity) |
| SoftMax | logN 12 | SLOT | d | (k/2)·d | +11 | — |

Secret key never changes: there is one context and one ring on this whole path.

**The score index, measured.** Slot `b + (k/2)·u` of part `j` holds
`S[head b][query u][key j]`, exactly as `to_slots` documents and to
**worst 1.5e-12** against the host `Q K^T`. The map is **the identity** —
2048/2048 — and it is **independent of the column**, which is the load-bearing
part.

### 11.2 The SoftMax invariant: satisfied, and for free

The key axis is **the ciphertext axis**, not a slot axis. Both attention paths
set `strided = true, stride = slot_count, count = 1`, so the in-ciphertext
reduction is a no-op and the denominator is a slot-wise sum of the `d` parts.
Measured consequences:

```
rows 2048, distinct reduction slots 2048,
one slot per row YES, no two rows share a group YES
reduction group of (h, r) = { part j : j in [0, d) } at one fixed slot
```

So the invariant *all `d` values `S[h][r][:]` lie in one reduction group* holds,
each group is one slot index across `d` ciphertexts, the groups are disjoint,
and the reduction costs **zero rotations** — cheaper than the `sum_blocked<128>`
tree the proposal assumes, which would cost 7 rotations, a mask and 7 more.

A whole causal SoftMax through it, against the host answer: **gain 0.999468,
worst 4.54e-4**, depth 31 out.

**Any slot permutation applied to every part alike is therefore harmless here**,
because nothing rotates. Only the causal mask and the row shift are indexed by
slot, and both are host-side plaintext vectors.

### 11.3 The fused crossing does **not** cross this encoding

`bootstrap_to_slots` — §5's island, which crosses a RECT column at 8.8e-6 —
returns garbage on a BATCH column: **1978 of 2048 slots do not decode to
anything that was put in**, the reading is **not** column independent, and it
matches identity, full bit reversal and per-field reversal **0/2048** apiece.

The diagnosis is exact, not plausible. Make the batch axis constant and the
transform collapses:

```
batch axis made constant: live slots 64 of 2048 (expect 64)
the d values at bit-reversed indices   gain 1.000001   worst 3.68e-06
```

So `bootstrap_to_slots` returns **the plaintext polynomial's coefficients,
bit reversed** — which is what it does for RECT too. The difference is what a
coefficient *means*:

* **RECT** puts value `X[token i][block t]` at coefficient `i + d·t`. A
  coefficient is a value, so CoeffToSlot is the crossing.
* **BATCH** (Kang, Definition 2) holds a matrix over `R_k` whose entries are
  `R_k` **slots**. The `k`-point transform along the batch axis sits between the
  coefficients and the values, so CoeffToSlot lands one transform short.

The island is therefore exactly the **k = 2** case — Algorithm 5's regime, where
the batch axis is trivial. Algorithm 4 with k/2 = 32 heads is precisely where it
fails. Closing that gap needs the batch transform composed into the CtoS
diagonals, which is the same missing `Vandermonde` hook §3 already lists.

**Consequence: the post-QK refresh must stay a full bootstrap, and the bridge
must stay a separate map.** The order in the code — bridge, then bootstrap — is
the right one.

### 11.4 Where the seam's time actually goes, and a 4.4× on it

Measured. Left: logN 12, d = 64, 32 heads of 64 × 64. Right: **logN 13,
d = 128, 32 heads of 128 × 128 — the real 8B attention shape, one call.**

| stage | 12/64 before | 12/64 after | 13/128 before | 13/128 after | share after |
|---|---:|---:|---:|---:|---:|
| K transpose (Alg 3, CMT) | 354.9 | 354.9 | 1301.3 | 1301.3 | 1.1% |
| **S = Q K^T (Alg 4, CCMM)** | 1332.4 | 1332.4 | 5142.9 | 5142.9 | **4.3%** |
| `to_slots` bridge | 19920.4 | **4517.3** | 149769.5 | **26805.2** | 22.5% |
| post-QK bootstrap | 22176.0 | 22176.0 | 85639.0 | 85639.0 | **72.0%** |
| **stage total** | **43731.2** | **28380.7** | **241852.7** | **118888.3** | |
| (SoftMax, after) | 4923.7 | 4923.7 | 17947.1 | 17947.1 | |

All times ms. The bridge is **4.41×** at d = 64 and **5.59×** at d = 128 —
against the 63/14 = 4.5 and 127/22 = 5.8 the key-switch count predicts. The
whole stage is **1.54×** and **2.03×**.

At the real shape the invariant is unchanged and the arithmetic is exact:
scores worst **4.00e-12**, rows 4096, distinct reduction slots 4096, one slot
per row YES, no two rows share a group YES, SoftMax rel **5.98e-3**.

Two changes, both in `Llama3BatchOperator::bridge`:

1. **Baby-step / giant-step over the d diagonals.** Writing `delta = i·n1 + j`
   splits one sum into `n1` shifts of the source and `n2 = d/n1` shifts of an
   accumulator: **d − 1 → n1 + n2 − 2 key switches per column**, 63 → 14 at
   d = 64 and 127 → 22 at d = 128. Both index sets are multiples of `k/2`
   smaller than `d·(k/2)`, so they are a **subset of the shifts the bridge
   already holds keys for — no new Galois key at any n1**. The diagonals are
   pre-rotated on the host, which is a relabelling. `rotate_rows` refuses a
   ciphertext that still owes a rescale, so the rescale moves inside the giant
   loop: `n2` rescales instead of one, and they commute with the rotation.
   Verified equivalent in the same run — n1 = 1 gives worst 3.5e-12 against the
   host scores and n1 = 8 gives 1.5e-12.
2. **The encoded diagonals cached** by (direction, level, prime). A crossing
   converts every column at one level, so all `d` columns wanted the same `d`
   plaintexts and the old code encoded them `d` times. **Worth 0.4%** — the
   hypothesis that encoding dominated was wrong, and the rotations are the cost.
   Kept because it is correct and free, and reported as measured rather than
   as the win it was expected to be.

**Ring switching targets the 4.7% row.** The bootstrap is 78%.

### 11.5 Parameter feasibility, from the library not from arithmetic

`benchmark/profile_low_ring_params.cpp` builds real contexts at
`sec_level_type::sec128` and reports what survives `generate()`.

| N | plain chain, best scale per level | with `q0 = scale + 10` (bootstrap-compatible) |
|---|---|---|
| **4096** | 1 level @ 2^36 (log PQ 108); **2+ levels: nothing ≥ 2^20** | **nothing, at any level, down to 2^15** |
| **8192** | 1 @ 2^60, 2 @ 2^54, 3 @ 2^43, 4 @ 2^36, 5 @ 2^31 | 1 @ **2^50, q0 2^60** (log PQ 170); 2 @ 2^49; 3 @ 2^39; 4 @ 2^33 |

The pipeline's own precision `(60, 50, 50 | 60)` is **rejected at both** — it is
two levels, and two levels at 2^50 need 220 > 218.

So:

* **`N_L = 2^12` is not implementable.** It holds the one Kang level at 2^36,
  but it cannot produce a ciphertext HEonGPU's bootstrap will accept at all,
  and the bootstrap is the whole point of composing back up. This is a
  feasibility result, not a tuning one.
* **`N_L = 2^13` is.** One Kang level at **exactly the 2^50 the rest of the
  pipeline runs at**, with `q0/scale = 2^10` — the ratio the CKKS bootstrap is
  built around — and 48 bits of budget spare.

This agrees with §7bis from the other direction: the CCMM is measured at
**1.64 ns/MAC at logN 12 and 0.97 at logN 13**, so logN 13 is ~1.7× *faster*
per MAC as well. And it agrees with the code's own constraint — `attention()`
requires `heads·d ≡ 0 (mod N/2)`, which at 32 heads of d = 128 means
**N ≤ 8192**, with 8192 the unique ring that fits all 32 heads in exactly one
Algorithm 4 call and wastes none of the batch axis.

Every argument points at **2^13**, and none at **2^12**.

### 11.6 Variant table

At the real 8B attention shape — 32 heads, T = 128, d_h = 128, one Algorithm 4
call, 128 ciphertexts of 4096 values:

| variant | low ring | QK CCMM | RingSwitch | Compose | BTS + conversion | SoftMax prep | total | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **current, before** | none (logN 13 throughout) | 5143 ms | — | — | 149770 + 85639 ms | 17947 ms | **259.8 s** | ran |
| **current, after** | none | 5143 ms | — | — | **26805** + 85639 ms | 17947 ms | **136.8 s** | ran |
| Sylph-12 | 2^12 | — | — | — | — | — | — | **not implementable**: no ring switch, no compose, **and no bootstrap-compatible chain exists at 128-bit** |
| Sylph-13 | 2^13 | — | — | — | — | — | — | **not implementable today**: parameters are fine, the two operations do not exist |

Peak GPU memory is not the binding constraint at this shape: the Galois key is
**138 rotation indices** (bridge ∪ CMT ∪ bootstrap) — the bridge and the CMT ask
for the same `d − 1` multiples of `k/2`, and BSGS adds none — which at 36 limbs
and N = 8192 is about **0.65 GiB**. Contrast Algorithm 5's `N/2 − 1`.

Sylph-12 and Sylph-13 cannot be benchmarked end to end because
`ring switch` and `compose` do not exist in this library in either direction
(§3), and neither is a small change: compose is cheap arithmetic —
`A(X) = Σ_j X^j a_j(X^16)` is coefficient interleaving, no level, no key switch
— but it lands under the **embedded key `s̃_L(X) = s_L(X^16)`**, and reaching the
bootstrap key needs a switching key between two *different ring degrees*, which
`HEKeyGenerator` cannot produce: every key it makes lives in one context, and a
context owns one `n`.

## 12. One whole block at the real 8B shape, on real weights, by kernel

Target `benchmark/profile_llama3_rect_boot.cpp`, stage `block`, with
`HEONGPU_BOOT_WEIGHTS` pointing at the directory `fetch_llama3_weights.py`
wrote: **Meta-Llama-3-8B layer 2**, its true residual stream, d_model 4096,
hidden 14336, 32 heads of 128 over 8 KV heads, 128 tokens. One A6000.

### 12.1 What it costs

| | |
|---|---|
| block compute | **2,425,507 ms = 2425.5 s** |
| the same block under nsys | 2,444,603 ms — **0.8%** overhead, so the capture is the run |
| GPU kernel time | 2304.8 s of 2444.6 s = **94.3% busy** — this is *not* launch-bound |
| kernel launches | 11,503,929 |
| refreshes | 14 seams, 2432 ciphertext-bootstraps at 493.5 ms each |
| output | max abs 1.058e-2 against a stream of max 0.6367 → **5.91 bits**, private rows |
| peak memory | 42.9 GiB of 47.5, Galois key 2048 indices at **10.0 GiB** |

Faster than the 2964.6 s this target's header used to document, and **at a
longer chain** — 40x80 against 37x74 of key-switch work per switch, ~1.17x
more. The BSGS row bridge of section 11 is what paid for that.

### 12.2 The chain the schedule now needs: 39, not 37

The header said 37 limbs and a worst stretch of 11. Both are stale, and the
reason is a *correctness* fix, not a regression: the SoftMax denominator used
to be calibrated without `causal_column_mask`'s own weight. Correcting it
widens the range to `[0.7597, 105.7]`, and a Chebyshev `1/x` over a 139x
spread wants **degree 63 — 6 levels, where degree 15 was 4**.

```
attention.refresh_post_qk   depth 25   limbs left 15   REFRESH
attention.shifted           depth 25   spent  0
attention.softmaxed         depth 38   spent 13        <- the worst stretch
```

exp deg 15 (4) + causal mask (1) + square (1) + reciprocal deg 63 (6) +
normalise (1) = **13**. A refresh hands back `chain - 25`, so the chain wants
`13 + 26 = 39`. At 37 the block dies inside the SoftMax with gpuntt's
`invalid configuration argument` — which is what running out of chain looks
like in HEonGPU, not a level error. Measured at 40: worst stretch 13, block
leaves at depth 29 with 11 limbs unspent.

### 12.3 Which kernel: the mod-down, and it is not close

| kernel | ms | % | calls | role |
|---|---:|---:|---:|---|
| `divide_round_lastq_permute_ckks_kernel` | 1,574,750 | **68.32** | 566,030 | mod-down + Galois permute — the tail of every **rotation** |
| `divide_round_lastq_extended_leveled_kernel` | 375,735 | **16.30** | 94,902 | mod-down — the tail of every **relinearization** |
| `base_conversion_DtoQtilde_relin_leveled_kernel` | 231,529 | **10.04** | 660,932 | **BConv up**, the decomposition base extension |
| all 5 NTT kernels together | 63,816 | **2.77** | ~6.2 M | forward + inverse NTT |
| `bm_gemm_kernel` | 21,045 | 0.91 | 132 | **the actual modular GEMM** |
| `keyswitch_multiply_accumulate_leveled_method_II_kernel` | 9,496 | 0.41 | 660,932 | the key inner product |

**Key switching is 95.1% of GPU time** by kernel name, ~97.8% counting its
NTTs. Inside it: **mod-down 84.6%, BConv 10.0%, NTT 2.8%, key product 0.4%.**

So the answer to "is it the NTT for key switching, or the BConv" is **neither**.
`|P| = |Q| = 40` gives dnum = 1, and the mod-down loops over all 40 special
primes for each of the L output limbs — **O(L x |P|) = O(L^2) per rotation**.
This target's own header predicted that ("dnum = 1 ... makes the mod-down
O(L^2)"); the measurement is that the prediction is 84.6% of the block.
dnum = 1 is a *key-budget* choice — Algorithm 5 needs N/2 - 1 = 2047 Galois
keys and nothing else fits on one card — so this is a memory decision being
paid for in time.

Note the method: `keyswitching_type_` is METHOD_I only when `|P| == 1`, so 40
special primes selects **METHOD_II**.

### 12.4 Which layout conversion: there are two, and the smaller one costs more

Every kernel charged to the innermost NVTX range open at its launch. nsys
2022.4.2 has no `nvtx_gpu_proj_sum`, so the interval join is done by hand
(`nvtx_kernel_join.py`, a stack sweep over range opens/closes and launches in
timestamp order).

| group | % of GPU time |
|---|---:|
| **bootstrapping** — CtoS 22.66, EvalMod 18.89, StoC 10.46 | **52.0** |
| **layout conversion** | **31.9** |
| **CMT automorphisms + tweaks** | **14.0** |
| the matrix products themselves | **1.0** |
| the non-linear fits | **~1.0** |

The 31.9% is two different objects, and naming them separately is the point:

| conversion | % | rotations | structure |
|---|---:|---:|---|
| `bridge.block_inverse` + `bridge.block_forward` | **16.5** | 142,080 | the **block-axis map**: `2(k/2) - 1 = 31` diagonals at unit shift, 30 rotations per ciphertext, **not BSGS'd** |
| `bridge.to_slots` + `bridge.from_slots` | **15.4** | 137,984 | the **row bridge**, BATCH <-> SLOT: `d = 128` diagonals, BSGS'd to `n1 + n2 - 2 = 22` per column |

**The 31-diagonal map costs more than the 128-diagonal one**, for exactly the
reason section 11 fixed on the other side: one has baby-step/giant-step and one
does not. It is the obvious next target — BSGS would take 30 rotations per
ciphertext to about 10 — but unlike the row bridge it is **not free of new
keys**: the giant shifts run past the `+-(k/2 - 1)` window this map's Galois
indices cover.

And `to_batch` / `from_batch` are not single conversions. Each is
**row bridge -> block map -> row bridge**: three levels, three key-switch
sweeps.

Plaintext encoding, which the `block_map` comment used to call "the single
largest kernel in every profile of that path", is now **0.02%**.

### 12.5 Parameters and layout, per function

Context is `sec_level_type::none` — a profiling context, not a secure one:
N = 4096, log Q = 60 + 39x50 = 2010, log P = 40x60 = 2400, **log PQ = 4410**
against the 128-bit table's 109 at this ring. Scale 2^50 under a 2^60 bottom
(the ~2^10 ratio the bootstrap needs), sparse secret h = 16, CtoS 3 / StoC 3 /
taylor 11, 2047 Algorithm-5 indices + 24 bootstrap indices.

A RECT group is a 128x128 matrix over `R_32` held as 128 ciphertexts:
ciphertext `j`, block `t`, row `i` is channel `g*2048 + t*128 + j` of token `i`.
d_model 4096 is 2 groups, so the residual stream is **256 ciphertexts**.

| function | encoding in -> out | cts | levels | dominant cost |
|---|---|---:|---:|---|
| `rms_norm` | RECT -> SLOT -> RECT | 256 | 11 | bridge + deg-15 `1/sqrt(x)` |
| `project.*` (Algorithm 5) | RECT -> RECT | 128/grp | 1 | `CMT.automorphisms`; the GEMM is 6% of it |
| `attention.to_batch` | RECT -> BATCH | 128 | 3 | row bridge x2 + block map |
| `CMT` (K transpose) | BATCH -> BATCH | 128 | 0 | automorphisms |
| `attention.scores` (Algorithm 4) | BATCH x BATCH -> BATCH | 128 | 1 | |
| `attention.to_slots` | BATCH -> SLOT | 128 | 1 | row bridge |
| `softmax` | SLOT -> SLOT | 128 | **13** | see 12.2 |
| `attention.from_slots` | SLOT -> BATCH | 128 | 1 | |
| `attention.value_product` (Algorithm 4) | BATCH x BATCH -> BATCH | 128 | 1 | |
| `attention.from_batch` + `W_o` | BATCH -> RECT | 128 | 4 | row bridge x2 + block map + PCMM |
| `ffn.to_slots` | RECT -> SLOT | 128 | 2 | row bridge + block map |
| `ffn.silu` | SLOT -> SLOT | 128 | 4 | deg-15 fit |
| `ffn.gate_product` | SLOT | 128 | 1 | |
| `ffn.from_slots` | SLOT -> RECT | 128 | 2 | |

Halves by wall time: attention 1001.1 s (41.0%), feed-forward 1315.5 s (53.9%),
`bootstrap` 1200.1 s (49.1%) spread across both.

### 12.6 What this says to do next

1. **The mod-down is the block.** 84.6% of GPU time in one O(L^2) kernel that is
   quadratic only because dnum = 1, and dnum = 1 only because Algorithm 5 wants
   2047 Galois keys. The lever is the key count, not the kernel.
2. **BSGS the block-axis map** — 16.5%, and the only conversion left walking its
   diagonals one at a time. Costs new Galois indices, which is why it is a
   trade and not a free win.
3. **The arithmetic is 1%.** Every remaining optimisation on this path is an
   optimisation of key switching, refreshing, or layout — not of the products
   the layer exists to compute.

## 13. The two-ring structure, priced and unblocked (2026-08-10)

Every §10 item below item 3 is now closed on branch `HEonGPU_LLama3_8B_tworing`.

**Placement is settled by measurement** (`tworing_attention_profile`,
`tworing_stage16_profile`, `test_ckks_tworing_bridge.cpp`):

- At a 2^13 high ring (profiling regime), §7bis's split wins: CCMM high, only
  Algorithm 5 low (measured A 3206 ms vs B 3147 ms vs no-island 2799 ms at
  the 8B attention shape).
- At the 2^16 high ring the security envelope forces (§7.4's conflict is
  resolved by force: only the v2-class chain fits 1761 bits, so this is a
  12-bit-class system), the placement INVERTS to Sylph's shape — but with
  the island at 2^13, not 2^12: one CCMM call covers all 32 heads at 2^13
  (61 ms vs 321 ms at 2^16, where 32 heads fill 32 of 256 batch slots), and
  2^12 admits no 128-bit chain at all.

**§8's crossing exists and is a layout choice, not new code.** compose_up of
k island SinC columns IS the big ring's SinC coefficient layout at block
size d*k over the same subring (k_s = N_s/d = N_H/(d k)), so SlotToSinC at
the big ring is `Llama3BatchOperator::to_slots/from_slots` at
`BatchMatrixLayout(N_H, d*k)`. Slot law: slot b + (k_H/2)(j' + k u) of big
ct g = M_b[u][g k + j']. Proven on GPU at 13<->12 (errors 2.9e-9/4.8e-10)
and at the real 16<->13 shape (1.2e-8/1.8e-9), full path slots -> SinC ->
switch_down -> island columns, with only the BSGS SUBSET of bridge keys
(62 shifts, not 1023 — 9.8 GiB at dnum 4 instead of 160).

**The Stage-3 ledger at the accepted envelope** (2^16, v2 chain, nbase 16,
dnum 4 per decision; island 2^13 on the natural prefix {41,33,33}; A6000):

| leg | measured |
|---|---|
| v2 bootstrap (16), 48 keys 7.7 GiB       | 356.8 ms/ct, 16.76 bits, depth_after 15 |
| SoftMax (16), exp 15 / 1/x 63, unmasked  | 58.1 ms/ct, 14 levels (13 with the folds) |
| wide crossing (16), l = 4                | 134.3 ms/big ct both ways (77 down + 57 up) |
| ring switch 16<->13, k = 8               | 0.63 ms/big ct |
| CCMM, 32 heads, one island call          | 57-61 ms (err 5e-8) |
| Alg 5, 4096 -> 4096, island              | 2.8-3.0 s (err 6e-9) |

Assembled 8B block projection: ~150-200 s against §12's measured 2425.5 s —
bootstraps 1200 s -> ~74 s (16x packing), conversions ~735 s -> tens of
seconds, Alg 5 -> 16 island calls. The dnum = 1 mod-down disease (§12.3's
84.6%) is gone by construction: the high ring holds ~50 boot + 62 bridge
indices, nothing forces dnum = 1.

**Open after this:** (1) dnum 4 is 96-190 bits over sec128 at the needed
nbase; the legal floor is dnum 6-7, affordable ONLY with the subset keys.
(2) The wide constructor inverts `step` d_H x d_H matrices on the host
(197 s at (65536,1024)); the Vandermonde inverse should be analytic.
(3) v2 at 2^16 is 16.75 bits mean / ~11-12 worst-slot — the 12-bit target
holds at the mean only. (4) `Llama3RectOperator::transformer_block` still
runs single-ring; wiring it onto these proven pieces is the remaining
engineering. (5) Generating the 62-key wide set exhausts a cold 40.9 GiB
RMM pool (keygen transients); a short-chain context for standalone key
generation, or a warmed pool, works around it.

### 13.1 The dnum decision record (folded from the untracked Doing.md, 2026-08-10)

The key-switch memory investigation that fixed dnum = 4 for the single-ring
block, in brief: Galois keys are always allocated at the top-of-chain width
(16*ceil(|Q|/|P|)*(|Q|+|P|)*N per element, no leveling); the 42.9 GiB
"peak" in §12.1 is the RMM pool's 0.9-of-free reservation, not a working
set (live use ~13-15 GiB), so key memory was never the binding constraint
it appeared to be. The dnum sweep model puts the knee at **dnum = 4
(|P| = 10)**: mod-down 84.6% -> ~5%, ~5.4x on the block, 25 GiB of keys,
and |P| <= 15 also clears two fixed-size local-array overruns in
switchkey.cu (`last_ct[15]`, `partial[20]`) that |P| = 40 and |P| = 20
silently exceed. The staged mod-down kernel merged from
`HEonGPU_LLama3_8B_moddown` attacks the same P^2 term at dnum = 1 with no
key cost — the two are substitutes; measure them together before stacking.
Still open from that investigation: a rotate-after-mod-drop test (the
leveled key-switch path has no coverage), pointing ReportMemory at the
pool's live counter, and level-truncated Galois keys (~30% off any dnum
choice).

## 14. The same block on an A100, re-profiled (2026-08-12)

Everything in §12 was measured on one A6000 at dnum = 1, before the staged
mod-down and before the plaintext expansion moved to the device. Both landed
on the union branch and **neither had ever been measured on hardware**. This
section is that measurement, on the vessl A100-SXM4-80GB at the same real
shape: Meta-Llama-3-8B layer 2, its true residual stream, d_model 4096, hidden
14336, 32 heads of 128 over 8 KV heads, 128 tokens, 40 limbs, `sec_level::none`.

Target and command are unchanged — `benchmark/profile_llama3_rect_boot.cpp`,
stage `block`, `HEONGPU_BOOT_WEIGHTS` pointing at the fetch script's directory.

### 14.1 What one block costs now

| | §12, A6000, dnum 1 | this, A100, dnum 4 |
|---|---:|---:|
| block compute | 2,425,507 ms | **149,341 ms** |
| GPU kernel time | 2304.8 s (94.3% busy) | 132.0 s (**88.4% busy**) |
| kernel launches | 11,503,929 | 12,728,402 |
| ciphertext-bootstraps | 2432 at 493.5 ms | 2432 at **23.8 ms** |
| worst stretch / refreshes | 13 at `attention.softmaxed` / 14 | unchanged: 13 / 14 |
| output | 1.058e-2, **5.91 bits** | 1.080e-2, **5.88 bits** |
| Galois key | 2048 indices, 10.0 GiB | 2048 indices, **25 GiB** |

**16.2x end to end**, and the accuracy is the same circuit's, unchanged. Three
separate things paid for it and they are separable: the staged mod-down
(§13.1's 5.0x), dnum 1 -> 4 (1.7x more), and the A100 over the A6000 (2.06x on
the same code and the same dnum, 307.2 -> 149.3 s).

**nsys is no longer free.** The same run under `nsys profile --trace=cuda,nvtx`
takes 173,550 ms — **16.2% overhead**, against §12's 0.8%. That is what a block
becoming launch-bound does to a per-launch profiler, and every absolute number
in §14.3 and §14.4 carries it. The shares do not.

### 14.2 The host is no longer the bottleneck, and the fix is measured

The bottleneck session's trace put `project.*.weight_encode` at **138.7 s of a
352.5 s block (39.3%)**: `encode_plaintext_matrix` built the whole
`num_limbs x per_limb` RNS-expanded plaintext on the host and uploaded it
pageable, 51.9 GiB per block. `2885349` / `cff71c5` moved that expansion into
`bm_crt_expand_kernel` and uploaded only the int64 coefficients, and the commit
message says compile-verified only.

Measured now, on the same 58 projections:

| | before | after |
|---|---:|---:|
| `project.*.weight_encode` | 138.7 s | **3.6 s** |
| share of the block | 39.3% | **2.1%** |
| GPU busy | 45% | **88.4%** |

The device expansion is also checked, not assumed: `HEONGPU_BM_ENCODE_CHECK=1`
compares every expansion against the host reference word for word, and the
batch-matrix, batch-matrix-GPU and rect test binaries are all green under it.

### 14.3 Which kernel: the NTT, and the mod-down is no longer first

12,728,402 launches, 132.05 s of kernel time, grouped by family.

| family | s | % | launches |
|---|---:|---:|---:|
| **NTT** (5 `gpuntt` kernels) | 49.70 | **37.6** | 6,144,294 |
| **mod-down** (staged chain + stage two + rescale) | 32.14 | **24.3** | 1,763,584 |
| **base conversion** (D->Q~ partial + gather) | 18.97 | **14.4** | 1,282,952 |
| **`bm_gemm_kernel`** — the actual modular GEMM | 10.04 | **7.6** | 132 |
| key-switch inner product (method II) | 5.33 | 4.0 | 660,932 |
| plaintext products | 5.02 | 3.8 | 1,170,360 |
| batch-matrix tweaks / subring transforms | 4.76 | 3.6 | 3,054 |
| elementwise (add, move, drop) | 4.33 | 3.3 | 1,412,924 |
| everything else | 1.77 | 1.3 | 290,170 |

**Key switching is 80.4% of GPU time** (NTT + mod-down + BConv + inner product)
across **660,932 key switches**, at 160 us each. §12.3's headline — "mod-down
84.6%, and it is not close" — is gone: the staged kernels plus dnum 4 took the
mod-down from 84.6% to 24.3%, and what surfaced underneath is the NTT.

The single largest kernel is now
`divide_round_lastq_p_chain_leveled_kernel<16>` at **11.0%** (14.54 s,
660,932 launches, 22.0 us each, standard deviation 838 ns). It is not
arithmetic-bound: the whole call moves 1.3 MB. It is **grid-bound** — one
thread per (coefficient, component) is 8192 threads at N = 4096, i.e. 32
blocks, so it occupies 32 of the A100's 108 SMs no matter what. §14.6 fixes
that.

`bm_gemm_kernel` is worth naming for the opposite reason: 132 launches of
76 ms, and at 3.7 M blocks reading ~120 GB it is running at roughly the card's
memory bandwidth. **The matrix multiply the layer exists to compute is 7.6% of
GPU time and is already at the hardware limit.**

### 14.4 Which function, and which layout conversion

`nsys stats --report nvtx_pushpop_sum`, so these are wall times and they nest.
Percentages are of the 172.4 s profiled block.

| activity | s | % of block |
|---|---:|---:|
| **layout conversion, all of it** | **67.6** | **39.2** |
| &nbsp;&nbsp;row bridge (`bridge.to_slots` + `bridge.from_slots`) | 44.2 | 25.6 |
| &nbsp;&nbsp;block map (`bridge.block_inverse` + `bridge.block_forward`) | 23.3 | 13.5 |
| **bootstrapping** (2432 ciphertexts) | **58.1** | **33.7** |
| &nbsp;&nbsp;`boot.eval_mod` / `boot.coeff_to_slot` / `boot.slot_to_coeff` | 25.2 / 21.1 / 11.6 | |
| **Algorithm 5, `RectangularPCMM`** (58 calls) | **34.3** | **19.9** |
| &nbsp;&nbsp;`.summation` — Theorem 3's two CMTs | 22.5 | 13.1 |
| &nbsp;&nbsp;`.blocks` — the batch PCMM itself | 11.7 | 6.8 |
| non-linear fits (`chebyshev`, `silu`, `softmax`) | ~5.7 | ~3.3 |
| plaintext weight encoding | 3.6 | 2.1 |

By half: feed-forward 97.0 s (56.3%), attention 69.3 s (40.2%).

Named crossings, which is the form §12.4 asked for: `bridge.rect_to_slots`
27.8 s over 16 calls, `bridge.rect_from_slots` 15.8 s over 9,
`bridge.to_batch` 15.7 s over 6, `bridge.from_batch` 4.0 s over 1, plus the
standalone `attention.to_slots` / `.from_slots` at 2.1 s each.

**§12.4's finding has inverted.** There it was block map 16.5% against row
bridge 15.4%, and the block map cost more despite having a quarter of the
diagonals. Here the row bridge is 25.6% and the block map 13.5% — but *per
ciphertext* they are still nearly equal (`bridge.to_slots.diagonals` 5.24 ms
against `bridge.block_inverse.diagonals` 4.94 ms), which is exactly §12.4's
point restated: 30 unhoisted rotations cost what 22 baby-step/giant-step ones
do. The row bridge is bigger now only because the block runs 6272
column-crossings of it against 4736 of the block map.

### 14.4bis Which kernel inside which function — the join

`nsys stats --report nvtx_kern_sum`, as a share of each range's OWN GPU time.
This is the cut neither §14.3 nor §14.4 shows, and it settles what the two
conversions differ by.

| function | GPU s | NTT | ModDown | BConv | KeyProd | other |
|---|---:|---:|---:|---:|---:|---|
| `bootstrap` (2432 ct) | 53.0 | **44%** | 27% | 18% | 5% | |
| `boot.eval_mod` | 22.7 | 45% | 27% | 16% | | 5% |
| `boot.coeff_to_slot` | 19.8 | 42% | 25% | 22% | 6% | |
| `boot.slot_to_coeff` | 10.4 | 44% | 32% | 15% | 4% | |
| `bridge.block_inverse.diagonals` | 12.7 | **43%** | 30% | 17% | 5% | |
| `bridge.block_forward.diagonals` | 7.1 | **44%** | 30% | 17% | 5% | |
| `CMT.automorphisms` | 15.9 | **46%** | 32% | 17% | 5% | |
| `RectangularPCMM.summation` | 19.2 | 38% | 26% | 14% | | BatchMatrix 18% |
| `bridge.to_slots.diagonals` | 7.3 | 29% | 18% | | | **PlainProduct 24%** |
| `bridge.from_slots.diagonals` | 5.3 | 30% | 18% | | | **PlainProduct 24%** |
| `RectangularPCMM.blocks` | 10.8 | | | | | **GEMM 93%** |
| `chebyshev.evaluate` | 2.1 | 39% | 20% | 8% | | 28% |

**The block map's signature is indistinguishable from the bootstrap's** —
43/30/17/5 against 44/27/18/5 — so it is pure key switching and nothing else.
The row bridge's is visibly different (NTT 29%, plaintext products 24%), and
that difference IS baby-step/giant-step: the row bridge has already traded its
rotations for plaintext multiply-accumulates and the block map has not. §12.6
item 2 asked whether BSGS on the block map was worth new Galois indices; this
is the measurement that says the map is 100% rotation cost and nothing else,
so the answer is about the rotation count alone.

**`RectangularPCMM.blocks` at 93% GEMM is the only range in the block where
the matrix arithmetic dominates its own time** — and it is 10.8 s of 132.0.
`CMT.automorphisms` at 15.9 s costs more than the GEMM it exists to prepare.

And the operation, from the mod-down stage-two call counts, which split
exactly one per key switch:

| operation | calls | share of the block's key switches |
|---|---:|---:|
| **rotation** (Galois) | 566,030 | **85.6%** |
| relinearization | 94,902 | 14.4% |
| total | 660,932 | 160 us each |

### 14.5 The dnum answer

Sweep at the real shape, all seven decompositions, `|P|` chosen so
`dnum = ceil(40 / |P|)`. Key memory is `2 * dnum * (|Q| + |P|) * N * 8` per
Galois index times 2048 indices, which is why this sweep needs an 80 GiB card:
the A6000 could not hold dnum 5 and above at all.

| dnum | `\|P\|` | block | vs dnum 4 | Galois keys | bits |
|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 380,132 ms | 2.55x slower | 10 GiB | 5.94 |
| 2 | 20 | 201,186 ms | 1.35x slower | 15 GiB | 5.76 |
| 3 | 14 | 167,219 ms | 1.12x slower | 20.25 GiB | 5.86 |
| 4 | 10 | 149,341 ms | — | 25 GiB | 5.88 |
| **5** | **8** | **141,358 ms** | **1.056x** | **30 GiB** | **5.88** |
| 6 | 7 | 139,031 ms | 1.074x | 35.25 GiB | 5.84 |
| 8 | 5 | 137,624 ms | 1.085x | 45 GiB | 5.88 |

**dnum = 4 is not the answer any more; dnum = 5 is the knee.** 4 -> 5 buys 5.3%
for 5 GiB. 5 -> 8 buys 2.7% more for fifteen. Below 4 the curve is steep and
above 6 it is flat, so the whole decision is the 4-6 band and the memory-cheap
end of it wins.

**The accuracy is flat across the entire sweep** — 5.76 to 5.94 bits, and the
spread is run-to-run noise, not a trend. dnum costs memory and buys time; it
does not touch precision. That confirms the moddown session's finding at the
real shape and settles it.

Two things worth keeping:

* **dnum 1 is now 380 s, not 2425 s.** Same decomposition, same shape, same
  circuit as §12 — the staged mod-down, the device-side expansion and the A100
  are 6.4x on their own, before any dnum is chosen.
* **This sweep is not runnable on an A6000.** Key memory is
  `2 * dnum * (\|Q\| + \|P\|) * N * 8` per index times 2048 indices, so dnum 5
  is 30 GiB and dnum 8 is 45 — past the 40.88 GiB pool ceiling that killed
  seven runs there. dnum 8's keygen cleared on this card without a single pool
  knob, which also retires the keygen-transient trap at this shape.

### 14.6 What was done about it

Three levers were tried against the profile above. Two are wins, one is a
win only in a range this path does not run in, and saying which is which is
the point of measuring them separately.

| configuration | block | vs baseline |
|---|---:|---:|
| dnum 4, hoisting off — where §14.1 starts | 149,341 ms | — |
| dnum 5, hoisting off | 141,358 ms | 1.056x |
| **dnum 5, hoisting on** | **122,303 ms** | **1.221x** |

**Hoisting is 14.1% and it was off by default.** `set_hoisted_crossings`
shares ONE key-switch decomposition — the INTT, the base extension and the
forward NTTs of every digit — across a whole rotation train, and every
crossing on this path is a train: the block map walks `2(k/2) - 1 = 31`
diagonals off one source, the row bridge's baby steps likewise. Measured
back to back at dnum 4: **150,131 ms off, 128,930 ms on**, and the same
again at dnum 5. When the crosstime session measured it the mod-down was
84.6% of the block and it was worth 3%; §14.3 puts the NTT at 37.6% and the
base conversion at 14.4%, which is precisely what a shared ModUp removes.
It is bit-identical (`test_ckks_llama3_rect.cpp` checks the hoisted crossing
against the staged one and the gap is 0), so this is a default worth
changing.

**RoPE is free.** The rect path had none — `llama3_prep.cuh` said so, and
the host reference matched it, so the profiled circuit was the real model in
every respect but this one. It costs one level and NO rotation, because a
RECT group holds channel `g*(N/2) + b*d + j` in ciphertext `j`, so with
`head_dim = d` the head is the block index and the head-dim index is the
CIPHERTEXT index: RoPE's `c <-> c + d/2` pairing is a host-side pairing of
whole ciphertexts. The angle needs the token in SLOTS, and the sublayer
already crosses there on its way to the batch encoding, so `to_batch_roped`
inserts it at that midpoint. Measured at the real shape: **145,564 ms with
it against 146,353 ms without, 5.92 bits against 5.94** — inside run-to-run
noise, and the worst stretch stays 13 at `attention.softmaxed`, so the extra
level lands in slack the schedule already had.

**The warp mod-down chain wins only at a long chain.** §14.3's largest
kernel is grid-bound — one thread per (coefficient, component) is 8192
threads, i.e. 32 blocks on a 108-SM card — so the chain was rewritten one
WARP per coefficient, lane `j` owning residue `j`, `__shfl_sync` carrying the
one shared scalar, `O(P^2)` per thread becoming `O(P)` per lane. It is
bit-exact (`HEONGPU_MODDOWN_CHECK=1` passes on rotation II, relinearization
and multiplication). It is also **not** a win where this path runs:

| `\|P\|` | dnum | array form | warp form | |
|---:|---:|---:|---:|---|
| 20 | 2 | 201,186 ms | **186,385 ms** | 7.4% faster |
| 10 | 4 | 149,341 ms | 152,315 ms | 2.0% slower |
| 8 | 5 | 141,539 ms | 146,353 ms | 3.4% slower |
| 5 | 8 | 137,624 ms | 142,613 ms | 3.6% slower |

The gain scales with `P` and the cost is fixed at `32 - P` idle lanes plus a
`__shfl_sync` dependency per step, so the crossover is real and lies between
10 and 20. The kernel is therefore gated at `P_size >= 16`, which keeps every
decomposition the Llama-3 path runs at on the array form and gives dnum 2 —
the memory-cheap option at 15 GiB of keys — 7.4% back. The grid argument was
right about the geometry and wrong about the consequence, and only the
measurement separates those.

### 14.7 The two-ring 16<->13 memory wall is gone, and the blocker moved

§13.5 and the gpu-watch analysis recorded seven consecutive deaths of the
real-shape two-ring legs, every one at exactly `cap x free-at-start` during
island keygen, on an A6000 with 43.03 GiB free. Re-run here with no pool
knob, no shortened chain and no key-residency trick:

| leg | result |
|---|---|
| `norm`, 16<->13 | **runs**, 8340.7 ms, error 7.15e-3 |
| `attention`, 16<->13, **full 4095-index Alg-5 island set** | **runs**, 69.2 s, `TWORING_RC=0` |

Free memory held flat at 35.26 GiB straight through island keygen — the
exact point that killed every A6000 attempt. **80 GiB is simply enough**, and
the keygen-transient pool trap does not exist at this shape on this card.

Two honest negatives come with it:

* **The 16<->13 attention answer is wrong**: `max abs error 4.444e+01`
  against a signal of order one. The stage is green at 13<->12 per §13, so
  the blocker has moved from VRAM to correctness, and the open item §13
  already names — the seam applying the exp domain map as a ciphertext
  multiply, spending a level the real path does not — is now the thing in
  the way.
* **It is not faster here.** 69.2 s for the attention sublayer against
  ~60 s for the single-ring attention half, because `island.to_batch` alone
  is 28.7 s (41%) and `island.out` 15.8 s (23%). The predicted win came from
  16x-packed bootstraps, and the bootstraps are indeed only 19.3 s of it —
  but the island's own crossings eat the difference.

The `norm` leg is the encouraging half: 8.34 s against 24.63 s recorded on
the A6000, same shape, 2.95x.

### 14.8 What this says to do next

1. **Turn hoisting on.** 14.1%, bit-identical, already implemented, already
   tested, and off by default for no reason that survives this profile.
2. **Run at dnum 5, not 4.** 5.3% for 5 GiB, and nothing above 6 is worth
   its memory.
3. **The block map is still the one conversion walking its diagonals one
   rotation at a time**, and it is 13.5% of the block and 29.1% of the
   two-ring norm leg. Hoisting covers it now; BSGS on top would need new
   Galois indices, which is the trade §12.6 named and it is still unmade.
4. **The remaining 80.4% is key switching, and the lever left is
   concurrency, not arithmetic.** The CMT's `d` automorphisms are `d`
   INDEPENDENT rotations of `d` different ciphertexts — a stream or a batched
   key-switch away from filling a card that a single 4096-coefficient key
   switch cannot. Every kernel in §14.3 averages 3-22 us; the machine is
   waiting, not working.
5. **The two-ring path needs a correctness pass at 16<->13**, not a memory
   one.


---

## 15. The level budget, and what it says about security (2026-08-12)

§14 measured the block by kernel, by function and by their join. This section
measures it by **modulus level**, which is the axis none of those show — and
which turns out to be the axis the security question is asked on.

Same machine, same shape, same weights as §14: vessl A100-SXM4-80GB, real
Meta-Llama-3-8B layer 2, logN 12, 40 limbs, dnum 5, hoisting on.

### 15.1 A refresh hands back 15 limbs; one stretch of seven wants them

The schedule refreshes to a fixed depth wherever it refreshes, because that is
what a bootstrap does. What the stretch behind the seam actually spends is not
fixed, and the printed depth ledger has always said so:

| stretch | spends | handed back | needs |
|---|---:|---:|---:|
| entry -> RMSNorm | 11 | **40** | 12 |
| post-norm -> QK^T | 6 | 15 | 7 |
| post-QK -> SoftMax | **13** | 15 | 14 |
| post-SoftMax -> PV + O + residual | 7 | 15 | 8 |
| mid -> FFN norm | 11 | 15 | 12 |
| post-FFN-norm -> SwiGLU | 9 | 15 | 10 |
| post-hidden -> down + residual | 4 | 15 | **5** |

"Needs" is one MORE than the stretch spends, because a bootstrap reads a
ciphertext with one prime left.

Those unspent limbs are not free to hold. `apply_galois_ckks_method_II` reads
its digit count from `d_leveled[depth]` and its RNS width from
`Q_prime_size - depth`, so a limb the stretch will never reach is carried by
every key switch until the next seam and then discarded — across **660,932
key switches** in a block.

`level_budget` (a42df7c) is the fix, and it is a hook rather than a schedule:
the operator asks, at each seam, what the stretch behind it needs and drops to
that. Empty by default, so nothing changes for an existing caller. The
benchmark derives all seven stretches from the same configuration the circuit
runs on, which makes the ledger the check on itself: **every stretch must land
on exactly one limb left**, and it does.

### 15.2 Measured

| run | crossings | budget | block | vs base | bits |
|---|---|---|---:|---:|---:|
| base | staged 2/2/3/3 | 15 flat | 123,266 ms | — | 5.93 |
| **budgeted** | staged 2/2/3/3 | 12/7/14/8/12/10/5 | **107,455 ms** | **-12.8%** | 5.92 |
| budgeted + fused | fused 1/1/1/1 | 10/5/14/6/10/9/4 | 142,573 ms | +15.7% | 5.96 |

**-12.8%, and it is free.** 5.92 against 5.93 bits is run-to-run noise:
discarding a limb the stretch cannot reach leaves the scale untouched, so
there is nothing for it to cost.

**It is also sublinear, and the reason matters.** The digit arithmetic says
the feed-forward tail alone (15 -> 5 limbs) should be ~3.5x cheaper; the block
moved 12.8%. §14.4bis says why: inside a crossing,
`divide_round_lastq_p_chain_leveled_kernel<16>` runs at 22,998 ns with a
**477 ns standard deviation** — flat at every level, because 8192 threads is
32 blocks on a 108-SM card. At N = 4096 a large part of the cost is
launch-bound and cannot be addressed by removing arithmetic. **At logN 16,
where each kernel does 16x the work, the same change should convert far
closer to the arithmetic** — which is an argument for the two-ring split on
speed grounds, independent of the security one below.

### 15.3 The one-level crossings lose, now measured where it counts

`575b76e` composes each crossing's stage matrices on the host into one dense
N/2-diagonal map: `to_batch`/`from_batch` 3 -> 1, rect `to_slots`/`from_slots`
2 -> 1, no new Galois keys. It has been off by default since the crossing-time
session found it a wall-time regression at d = 64 (staged+hoist 54.8 s
against fused+hoist 61.1 s, +11.5%).

That verdict was measured **without** a level budget, which was the obvious
objection to it: a shorter stretch buys nothing when the seam hands back 15
limbs regardless. The third row above is that objection tested at the real
8B shape with the budget in place. The level accounting is exactly right —
every stretch still lands on one limb, two levels cheaper each, deepest point
in the block 14 limbs instead of 15 — and fused is **still 32.7% slower than
staged**. The dense map's 2048 plaintext products per crossing cost more than
the two levels save, and at d = 128 the penalty is worse than the d = 64
measurement predicted.

**Run staged. The fused crossings are correct, are the level floor, and are
the wrong choice at this shape** — for a second, independent reason now.

### 15.4 Neither lever shortens the chain, and that is the security problem

The budget changes what is carried WITHIN a stretch. It does not change the
worst stretch, and the worst stretch is what sets the chain: 25 (bootstrap) +
13 (SoftMax) = 38, run at 40. The fused crossings shorten five of the seven
stretches and leave the SoftMax at 13, so they do not shorten it either.

That matters because **the chain length is the security parameter.**
`context.cu:113` checks `sum(log Q) + sum(log P)` against
`heongpu_128bit_std_parms(N)`. For the configuration every number in §14 and
§15 was measured on:

* Q = 60 + 39 x 50 = **2010 bits**, P = 8 x 60 = **480 bits**, log QP = **2490**
* `heongpu_128bit_std_parms(4096)` = **109**

**Over the 128-bit budget by 22.8x**, on top of a sparse h = 16 secret that
the ternary-uniform table does not cover. Every profile in this document is a
cost model, not a parameter set, and §7 said so before any of them were run.

The caps the design has to fit inside:

| ring | 128-bit cap on log QP | usable levels |
|---|---:|---|
| logN 12 | 109 | 1 comfortable, 2 tight (§7) |
| logN 13 | 218 | 4-6, by prime size |
| logN 16 | 1761 | 38 at 33-bit primes |

**So the island is logN 13, not 12.** logN 12 holds Stage 3's two low-ring
levels only at 27-bit primes, which §7 flags as precision-unverified against
a d = 128 contraction. logN 13 has real headroom.

A two-ring set that closes at 128 bits, ternary:

* **High, logN 16** — Q = 41 + 38 x 33 = 1295, P = 8 x 33 = 264,
  log QP = **1559 <= 1761** ✔
* **Island, logN 13** — Q = 41 + 4 x 33 = 173, P = 33,
  log QP = **206 <= 218** ✔

Note what forces the prime size: 50-bit primes do not fit at logN 16 either
(60 + 38 x 50 = 1960 > 1761 before P is counted). The chain has to come down
to ~33-bit primes, which forces bootstrap **v2** (`q0 = p + 8 = 41`) and its
measured 16.75 bits at logN 16 — clearing Sylph's 12-bit target and missing
20, exactly the conflict §7.4 named.

### 15.5 What this says to do next

1. **The SoftMax's 13 levels are the blocking item for both remaining goals.**
   They set the chain, the chain sets log QP, and log QP is what puts the
   configuration 22.8x outside 128-bit security. The reciprocal is **6 of the
   13**, at degree 63, purely because its calibrated range spans 139x
   (`[0.7597, 105.7]`). Narrowing that range is worth more than any kernel in
   §14.3.
2. **Move the QKV seam behind the projections.** The refresh currently sits
   between the norm and the projections, so an Algorithm-5 PCMM worth 8.5 s
   runs at the top of the next budget. Folding it into the norm's stretch
   (11 + 1 = 12, budget 13) puts it at 2 limbs instead of 7 — a better
   schedule than 15.1 implements, and not yet built.
3. **Re-test the fused crossings only after (1).** They are the tool for
   turning a shorter stretch into a shorter CHAIN, and they are worth their
   wall-time penalty only when the chain actually moves.
4. §14.8 items 1-5 are unchanged.
