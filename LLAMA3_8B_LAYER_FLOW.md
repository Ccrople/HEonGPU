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

## 16. The crossing bottleneck, and a chain that fits the security cap (2026-08-12)

Session `tworing-crossings`. Three questions from the user: what the island
ring is, why the ledger's limb columns did not join up, and whether
`island.to_batch` / `island.from_batch` — 38.5% of the block between them —
can be made cheaper, "always regarding the security".

### 16.1 The ledger was reporting an envelope, not a chain

§13's table printed one row per leg with the **widest in and narrowest out**
over that leg's calls. For a leg that runs once that is exact; `cross.up`
runs eleven times, `wide.to_slots` seven and `wide.block_map` eight, at
different points in the chain, and each appeared once at the position where
it FIRST ran. So `island.qk` ending at 3 limbs sat directly above a
`boot.post_qk` starting at 1, and the three legs actually in between —
`cross.up`, a `match_scale` and `wide.to_slots` — were printed elsewhere in
the same table.

The ledger now records OCCURRENCES: every `charge` accumulates against its
leg, every `note` closes an occurrence with exactly the time charged to that
leg since its previous note, and the timeline prints in execution order ahead
of the rollup. The walked total is printed against the charged total so
nothing can go unclaimed (`0050ce0`).

### 16.2 A fused crossing is DENSE, and that is the whole cost

`rect_is` runs with `set_fused_crossings(true)` because `shared = 4` cannot
afford a 3-level staged crossing. But `to_batch` composes `bridge_inv`
(shifts = multiples of `step`), `block_inv` (shifts in `+-(step-1)`) and
`bridge_fwd` (multiples of `step`), and **that sum covers every residue mod
N/2**. The composed table therefore keeps all 4096 diagonals, and the
unhoisted BSGS loop launched a `multiply_plain` and an `add` for each — per
ciphertext, 128 of them, four crossings a block. 4.2 M launches.

Any fuse touching the block map is dense for the same reason: the
`+-(step-1)` window breaks the multiple-of-`step` lattice. This is not a
tuning accident, it is what fusing costs.

`set_hoisted_crossings(true)` on both rings (`a20dd45`) replaces that with
one shared decomposition per source and **one fused multiply-accumulate per
giant group**. Bit-identical — 6.39 bits before and after.

### 16.3 The diagonal encode is a per-MODEL cost

The 4096-diagonal plaintext table is cached on `(map, depth, prime)`, all
four of a block's crossings run at the same depth, and every layer of a
32-layer model runs the same two maps at the same depths. A one-block
benchmark was charging a per-model cost to whichever crossing ran first in
each direction — which is exactly why `to_batch` measured 9.4 s per call and
`from_batch` 14.5 s for ONE call at the same size and level.

The arithmetic said so before the instrument was built: `3X + E = 12738` and
`X + E = 9401` give `X = 1.7 s`, `E = 7.7 s`. Charged to its own leg
(`a3c3822`): `crossing.encode` 18,708 ms once, `to_batch` **1,614 ms per
call**, `from_batch` **1,864 ms**.

| leg | §13 | hoisted | hoisted + encode split |
|---|---:|---:|---:|
| `island.to_batch` (3 calls) | 28,160 | 12,738 | **4,842** |
| `island.from_batch` (1 call) | 14,528 | 9,401 | **1,864** |
| `wide.block_map` (8) | 11,050 | 5,895 | 5,889 |
| `wide.from_slots` (7) | 5,960 | 4,699 | 4,646 |
| `wide.to_slots` (7) | 4,451 | 3,609 | 3,561 |
| `crossing.encode` | — | — | 18,708 (once per model) |

**The crossing pair: 42,689 -> 6,705 ms, 6.4x.**

### 16.4 Paying the 96 bits: cut Q, not P

The configuration §13 measured is over the 128-bit modulus budget, and only
on one ring:

| ring | Q | P | log QP | cap | |
|---|---|---|---:|---:|---|
| high 2^16 | 41 + 16x33 + 3x32 + 8x60 + 4x56 = 1369 | 8x61 = 488 | **1857** | 1761 | **96 over** |
| island 2^13 | 41 + 3x33 = 140 | 61 | **201** | 218 | OK |

Two ways to pay it, and they are not equivalent:

* **Cut P** (`SPECIALS` 8 -> 6, log QP 1735 OK). HEonGPU derives
  `dnum = ceil(Q_size / P_size)`, so dnum goes 4 -> 6 and every big key goes
  167.8 -> 239.1 MB. At 178 resident keys that is **+11.8 GiB**. Measured
  twice on the 80 GiB A100: OOM at the first wide bridge, both times.
* **Cut Q** (`NBASE` 16 -> 13, log QP **1758** OK). dnum stays 4, `Q_prime`
  40 -> 37, and each key *shrinks* to 155.2 MB. The cost is the bootstrap
  window: 17 limbs -> 14, which every stretch then has to fit.

Fitting the 14-limb window took two fit degrees, and the level audit is the
whole of it. At `NORM_DEGREE=15` the norm leg is
`14 -> block_map 13 -> rmsnorm 5 -> scale_norm 4 -> bridge 3`, and the island
needs 4 — one short, which the chain reports as
`ntt.cu:2624 invalid argument` inside the QK matmul. Degree 7 gives back
exactly that limb. The SoftMax needs the same treatment: degree-15 exp and
reciprocal cost 12 levels against a window of 14, degree 7 costs 10.

### 16.5 The measured result: faster AND more accurate

Whole block, 16 <-> 13, d = 128, model 4096, 32 heads (8 KV),
A100-SXM4-80GB.

| configuration | log QP | fits cap | first block | steady state | bits |
|---|---:|---|---:|---:|---:|
| §13 baseline | 1857 | no | 110,856 | — | 6.39 |
| + hoisted crossings | 1857 | no | 85,084 | — | 6.39 |
| + encode charged apart | 1857 | no | 88,061 | 69,353 | 6.39 |
| **NBASE 13, degree-7 fits** | **1758** | **yes** | 81,490 | **62,551** | **9.78** |

**110,856 -> 62,551 ms per layer, -43.6%, +3.4 bits, inside the modulus
budget.** The accuracy going UP with lower-degree fits is not a fluke of
this run: shorter stretches mean fewer rescales and less accumulated noise,
and a high-degree Chebyshev over a deliberately widened interval is itself a
noise source — the same lesson §11 recorded when the reciprocal's range was
found to be wrong by 98.6%.

The secure block by bucket (steady state, 62,551 ms):

| | ms | % |
|---|---:|---:|
| **Bootstrapping** (11 calls) | **31,550** | **50.4** |
| Layout conversion | 20,054 | 32.1 |
| Matrix products | 9,997 | 16.0 |
| Non-linear fits | 719 | 1.1 |
| Ring switching (22 crossings) | 176 | 0.3 |

**Ring switching is 0.3%.** The two-ring structure itself is free; what it
costs is the refreshes it enables.

### 16.6 What the security cap forces, structurally

`shared = 4` is not a tuning choice. The island's Q is a literal prefix of
the big chain, so `shared = 5` is 41 + 4x33 + 61 = **234 > 218**. And
`shared = 4` is exactly what makes V's bootstrap load-bearing: entry 4,
projection -1, `to_batch` -1 leaves V at 2 limbs while the PV product needs
3. That is `boot.qkv`'s third call, 2,629 ms a block. Shrinking the island's
own P is not available either — it is the big chain's first special at 61
bits, and method-I key-switch noise scales with Q/P.

**Caveat, and it is not a small one.** log QP now satisfies
`heongpu_128bit_std_parms`, which is the only check the library makes. That
table assumes a **uniform ternary** secret. This driver uses
`Secretkey(big, 192)` and, for the v2 bootstrap's dense-to-sparse switch,
`Secretkey(big, 32)`. Sparse secrets are not covered by the table, so this
configuration clears the modulus-budget half of 128-bit security and not the
secret-distribution half.

### 16.7 Next

1. **Bootstrapping is now the block, at 50.4%** — 11 calls x 16 big
   ciphertexts x ~164 ms. Every other bucket is under a third of it.
2. **BSGS on `wide.block_map`** (4,909 ms, 7.8%) — BUILT, see §16.8.
   The comment at `llama3_rect.cu:348` says it needs new Galois indices; at
   `step_h = 32` it does not. With `n1 = 8`, `eps = 8i + j` needs babies
   `{1..7}` and giants `{+-8, +-16, +-24, +-32}`, and `build_wide()` already
   generates `+-1..+-31` for the block map and `+-32` for the SoftMax comb.
   Rotations per ciphertext 62 -> 14.
3. The FFN here is `hidden = 8192` against the real 14336 (2 chunks against
   4). The FFN legs scale per chunk; §16.5 is not directly comparable to
   §15's single-ring 107,455 ms, which ran the full width.
4. §15.5 items 1-3 are unchanged, and (1) is now partly done: the SoftMax
   runs in 10 levels rather than 13, at better accuracy.

### 16.8 BSGS on the block map: the one index the old note got wrong

`block_map` applies 2*step - 1 diagonals at shifts eps in
[-(step-1), step-1], one rotation each — 62 key switches per ciphertext at
step_h = 32, and 7.8% of the block once the crossings were hoisted. The
comment at `llama3_rect.cu:348` explained why BSGS had not been done: the
giant shifts "run past the +-(k/2 - 1) window this map's Galois indices
cover, so unlike the row bridge it is not free of new keys."

That is true of **exactly one index**. Writing eps = n1*i + b with
b in [0, n1):

* babies are `1..n1-1`, inside the window;
* non-negative giants are at most `n1*floor((step-1)/n1) < step`, inside it;
* only the most negative giant, `-n1*ceil((step-1)/n1)`, reaches `-step`.

At step_h = 32, n1 = 8 that single index is `-32`, and `build_wide()`
already generates it — it is the SoftMax comb's `+-step_h`. So the split is
free of new keys on this path after all. `block_bsgs_rotation_indices()`
returns the set for callers who need to check, and a missing key falls back
to multi-hop rotation: slower, never wrong.

Rotations per ciphertext **62 -> 14** (7 babies + 7 live giants).

**It is not bit-identical, and it cannot be.** The plain walk sums every
diagonal and rescales once; the split rescales each of the 8 giant groups.
Measured gap against the plain walk **1.49e-08**, consistent with sqrt(8)
independent rescale roundings and three orders under the encoding floor.
The test measures that gap rather than asserting equality, and separately
checks that the level cost is unchanged, that the round trip still inverts,
and that no shift the split asks for leaves the `+-step` window. 38/38 green
on the rect suite (36 before, +2 new).

Measured on the secure block, `d17ec6e`:

| leg | before | after | |
|---|---:|---:|---|
| `wide.block_map` (8 calls) | 4,908.9 | **1,948.8** | **-60.3%** |
| whole block, steady state | 62,551 | **58,949** | -5.8% |
| accuracy | 9.78 bits | 9.77 bits | unchanged |

Of the 3,602 ms the block lost, 2,960 is the block map; the remainder is
run-to-run variation across the projection legs and is not attributable to
this change.

**The cumulative picture, one 16<->13 layer:**

| step | steady state | bits |
|---|---:|---:|
| §13 baseline | 110,856 | 6.39 |
| + hoisted crossings | (85,084 first block) | 6.39 |
| + encode charged apart | 69,353 | 6.39 |
| + NBASE 13, degree-7 fits (fits the cap) | 62,551 | 9.78 |
| + BSGS block map | **58,949** | **9.77** |

**110,856 -> 58,949 ms, -46.8%, +3.4 bits, inside the 128-bit modulus cap.**

The buckets at the end of it (58,949 ms):

| | ms | % |
|---|---:|---:|
| **Bootstrapping** (11 calls) | **31,647** | **53.7** |
| Layout conversion | 17,330 | 29.4 |
| Matrix products | 9,009 | 15.3 |
| Non-linear fits | 723 | 1.2 |
| Ring switching (22 crossings) | 179 | 0.3 |

Bootstrapping was 32.4% of the §13 block and is 53.7% of this one, having
barely moved in absolute terms (35,909 -> 31,647). Everything else has been
cut roughly in half around it. It is now the only thing worth attacking.

## 17. Why the island boots so often, and what it would take to stop

The question this section answers, asked directly: *the bootstrap should come
after RMS and QKV are done, but we lack the modulus to reach `island.qk`, so
we boot before it. Can we not adjust the modulus and skip the boots between
`pv` and `project_o` as well? Can we use smaller primes?*

The premise is right, the arithmetic backs it, and the fix is not smaller
primes.

### 17.1 What the restructuring costs in levels

The island enters each sublayer at `shared - 1` (the norm's `from_slots_at`
bridge spends one). Walking the code with `S = shared`:

| step | level after |
|---|---|
| entry from the norm | `S-1` |
| `island.project_qkv` | `S-2` |
| `island.to_batch` (q, k, v) | `S-3` |
| `island.qk` | `S-4` |
| ascend + `scale_norm` | `S-5` |
| `wide.to_slots` | `S-6` |
| the boot needs >= 1 in | **`S >= 7`** |

So **running `island.qk` before any refresh needs `shared >= 7`**, and that
deletes `boot.qkv` x3 -- 7,913 ms, **13.4% of the block**.

Continuing through the tail, with V left unrefreshed at `S-3` and P descending
from the SoftMax boot to `S`:

| step | level after |
|---|---|
| `island.pv` (aligns to min = `S-3`) | `S-4` |
| `island.from_batch` | `S-5` |
| `island.project_o` | `S-6` |
| `residual` (its `match_scale`) | `S-7` |
| norm2 needs >= 2 to enter | **`S >= 9`** |

**`pv` -> `project_o` with no boot between needs `shared >= 9`**, which also
deletes `boot.attn_tail`: **10,550 ms, 17.9% of the block**, four of the
twelve boots.

(The tail's pre-boot `scale_norm` goes with it. It exists only because a v2
boot corrupts a drifted scale; with no boot there is nothing to protect, and
the level it spends is the one `project_o` wants. `HEONGPU_TB_LAZY_REFRESH`
folds the two decisions into one.)

### 17.2 The island's budget is bits, and the primes cannot shrink

`build_big` lays the chain out as `q0 = 41`, then `nbase` primes at `p = 33`,
then the boot's own StC/sine/CtS primes. The island takes the bottom `shared`
of them plus one special: `Q = 41 + 33*(S-1)`, `P = 61`.

| S | log QP | logN 13 (218) | logN 14 (438) |
|---|---|---|---|
| 4 | 201 | fits, 17 spare | fits |
| 5 | 234 | **over by 16** | fits |
| 7 | 300 | over | fits |
| 9 | 366 | over | fits, 72 spare |
| 11 | 432 | over | fits, 6 spare |

**Smaller primes do not work, for three independent reasons.**

1. **`p` IS the scale.** `scale = 2^p` at `profile_tworing_block.cpp:443`. A
   rescale divides by one prime; a prime below the scale makes the scale
   collapse instead of returning to nominal. 33 bits is a floor set by the
   scale, not a free parameter.
2. **Lowering the scale destroys the answer.** The block delivers 9.77 bits at
   `2^33`. Dropping to `2^25` to fit 25-bit primes costs about 8 of them, so
   ~1.8 bits. `q0/scale` must also stay near `2^8` or the v2 boot returns
   silent noise, and the boot's own precision ceiling is ~20 bits.
3. **Mixed prime sizes break the fixed-scale invariant.** The island's primes
   ARE the big ring's bottom primes -- the ring switch requires a shared
   prefix -- and the Llama path `match_scale`s to exactly `2^33` everywhere. A
   rescale by a 25-bit prime at scale `2^33` leaves `2^41`, not `2^33`.

And even setting all three aside: at `S = 6` with *every* prime at 33 bits,
`Q = 198`, leaving 20 bits for `P`. A special prime smaller than the largest Q
prime makes method-I key switching diverge. **`shared >= 6` is unreachable at
logN 13 under any prime assignment.** The 218-bit cap is the whole story.

### 17.3 So the island ring has to grow -- and it does not fit

logN 14 raises the cap to 438, which holds `shared` up to 11, and it is
**secure**: `S = 9` is 366 <= 438, the big ring is unchanged at 1758 <= 1761.
Security is not the blocker here.

Key memory is. Algorithm 5 needs the full `N/2 - 1` rotation group, which
**doubles to 8191 indices** at logN 14 while each key also doubles in size:
`2 * dnum * (Q+P) * N * 8` = 2.5 MiB at `S = 9`, so **20.5 GiB** against
2.5 GiB at logN 13.

Measured on the 80 GiB A100, both dying on the same 22.5 MiB key allocation:

| config | pool cap | died at |
|---|---|---|
| logN 14, `S = 9` | 90% | 70.947 / 70.952 GiB |
| logN 14, `S = 9` | 97% | 76.460 / 76.471 GiB |

The cap moved 5.5 GiB and the wall did not. The budget explains why: big
context 43.6 + wide operator tables 13.8 + island keys 20.5 = **77.9 GiB on a
79.3 GiB card**, before any transient. Unlike the failures in
`rmm_pool_ceiling_not_working_set`, this one is real demand, not ballooning.

Three attempts, each dying on the same Galois key allocation:

| config | pool cap | died at | request |
|---|---|---|---|
| logN 14, `shared = 9` | 90% | 70.947 / 70.952 GiB | 22.5 MiB |
| logN 14, `shared = 9` | 97% | 76.460 / 76.471 GiB | 22.5 MiB |
| logN 14, `shared = 7` | 95% | 74.892 / 74.894 GiB | 14.0 MiB |

The cap moved 5.5 GiB between the first two and the wall did not move with it.
The budget says why: big-ring context 43.6 + wide operator tables 13.8 +
island keys 16.4 (S=7) to 20.5 (S=9) = **73.8 to 77.9 GiB on a 79.3 GiB
card**, before any transient. Note this is the OPPOSITE diagnosis from
`rmm_pool_ceiling_not_working_set`: there `max` equalled cap x free-at-start
while live data varied by gigabytes, which is ballooning. Here the demand is
real, and lowering `shared` from 9 to 7 moved the death point by exactly the
4.1 GiB the smaller keys account for.

**So the restructuring needs roughly 96 GiB and the card has 80.** It is an
H100/H200 change, or it needs the Algorithm 5 key set to stop being resident
-- streamed per rotation, or replaced by a decomposition that does not demand
the full `N/2 - 1` group.

### 17.4 What was implemented anyway

`HEONGPU_TB_LAZY_REFRESH` (default off, `ff62042`) makes the four island
refreshes conditional: each fires only when the walk behind it spends more
than the stream holds (`q`/`k` need 4, `v` needs 6, the tail needs 4). The
tail's pre-boot `scale_norm` is folded into the same decision, because it
exists only to protect a v2 boot from a drifted scale and the level it spends
is the one `project_o` wants.

It is fail-safe by construction -- when the levels are absent the boot still
fires -- and a **no-op at `shared = 4`**, which is what the shipping config
runs. It is the piece that turns "more island levels" into "fewer boots", and
it is in place for whenever the memory exists.

### 17.5 The answer, in one line

The premise was right and the mechanism is understood: four of the twelve
boots are avoidable, worth 17.9%, and they persist because a 2^13 island holds
four limbs. Smaller primes cannot add limbs because the prime size IS the
scale. A 2^14 island can, is secure, and does not fit an 80 GiB card by about
16 GiB. **The blocker moved from modulus to key memory, and that is a
different machine, not a different parameter.**

## 18. Correction to §17: `shared` is not what pins the island entry

§17.1 asserted that the island enters each sublayer at `shared - 1`, and
concluded that `shared >= 7` deletes `boot.qkv` x3. **The spend counts in
§17.1 are right. The entry rule is wrong**, and with it the conclusion that a
bigger island ring would have bought anything.

The rule was fitted to a single observation -- at `shared = 4` the entry is 3,
which equals `shared - 1` by coincidence.

**The entry is `min(shared, 14 - norm_leg_spend)`, and the second term binds.**
`drop_big_to` (`benchmark/profile_tworing_block.cpp:1370-1373`) and `descend`
(`:756-770`) **only drop, never raise**:

```
boot.norm                    -> 14
:1442 wide.block_map (inv)   -1 -> 13
:1471 rmsnorm                -7 ->  6   (degree 7; the DEFAULT at :1456 is 15, costing 8)
:1480 scale_norm             -1 ->  5
:1484 wide.block_map (fwd)   -1 ->  4
:1492 from_slots_at(shared)     -> drop_big_to(slots, shared+1) is a NO-OP at 4
                                   for shared = 4, 7 and 9 alike
                                -> bridge -1 -> 3
:1493 descend(big_out, shared)  -> island 3
```

Three consequences:

1. **`shared >= 7` alone deletes zero boots.** q and k still land at 1 limb
   after `to_batch` and `island_refresh_if(qb, "qkv", 4)` still fires.
2. **`HEONGPU_TB_LAZY_REFRESH` (`ff62042`) is inert, not merely off** -- it
   cannot fire for q/k/v at any `shared` as the code stands.
3. **§17.3's logN 14 key-memory wall is real but was never the binding
   constraint.** An H100 would not have bought a boot either.

### 18.1 What does close it: the entry side

The entry side has soft levels of its own, which §17 did not count.

* **E1 -- fold `scale_norm` (`:1480`) into the adjacent forward `block_map`
  (`:1484`).** `block_map` takes `plain_scale = rescale_prime(ct.front())` as
  a free local and re-encodes every call (`src/lib/host/ckks/llama3_rect.cu:388`),
  so there is no cached diagonal set to poison. Encoding at
  `plain_scale * (nominal / s_ct)` makes the trailing rescale land on nominal,
  which is exactly `match_scale`'s own expression
  (`src/lib/host/ckks/llama3.cu:622-627`). **+1 entry limb.**
* **E2 -- fold the RMSNorm learned gain into the projection weights.**
  `rms_norm`'s last level is `multiply_plaintext(normalised, weights[at])` and
  `weights` is already optional (`llama3.cu:1364`). Everything between
  `rms_norm` and `project` is linear, so `(g*x)W = x*diag(g)W` host-side.
  **+1 entry limb.**
* **S1 -- fold the pre-boot `scale_norm` (`:1629`) into the wide bridge's
  `plain_scale`** (`llama3_batch.cu:273`). **-1 spend.**

At `shared = 5`, `P <= 45`, `NORM_DEGREE = 7`:

```
boot 14 -> block_map 13 -> rmsnorm(-6, gain folded) 7 -> block_map(scale folded) 6
        -> drop_big_to(6) no-op -> bridge 5 -> descend min(5,5) = ENTRY 5
spend:  project 1 + to_batch 1 + qk 1 + to_slots 1 + boot input 1 = 5
5 >= 5  ->  q and k skip their boots.
```

All four preconditions are required together; drop any one and the saving is
exactly zero. **It does not free `v`** (needs 6, arrives at 3), so the prize is
**2 of 3 boots = 5,275 ms = 8.95%**, not §17's 13.4%.

### 18.2 Two hard floors §17 did not have

* **`P >= 41 bits.** `coefficient_validator` (`src/lib/util/util.cu:11-52`,
  called from `src/lib/host/ckks/context.cu:181-186`) with `P_size == 1` sets
  `quotient = Q_size, remainder = 0`, so it compares **each individual** `q_i`
  against the total P bit-count -- and `q0` is 41 bits. It fires even under
  `sec_level_type::none`. With the 218-bit cap this leaves `P in [41, 45]` and
  **`shared = 5` as the only reachable step**; `shared = 6` is 247 bits at the
  best possible P, impossible at logN 13 under any assignment.
* **The boot's input limb is a floor, three ways.**
  `regular_bootstrapping_v2` throws unless exactly one limb remains
  (`operator.cu:7474-7479`), `mod_drop_ckks_leveled_inplace` throws at
  `depth_ >= Q_size-1` (`operator.cu:1457-1460`), and ModRaise's INTT is
  hard-coded to one modulus (`operator.cu:4182`).

### 18.3 Levers that died, and should not be revisited

* **`to_slots` fused into CoeffToSlot -- dead on ENCODING, not on cost.**
  `ringswitch_interleave_kernel` (`src/lib/kernel/ringswitch.cu:42-52`) puts
  the poly index in the low `k_power` bits, so the wide row is composite,
  `i = jp + 8u`. CoeffToSlot bit-reverses the **full** 15-bit index, which
  re-partitions that composite field: the key axis moves from stride 32 to
  4096 and the query axis from 256 to 32 -- the two axes swap. The RECT
  precedent (`island_slot`, `llama3_rect.cuh:604-614`) absorbs reversal
  *within* a field; it cannot absorb a re-partition. The SoftMax comb
  (`profile_tworing_block.cpp:1078-1099`) would silently sum 8 query rows
  instead of 8 keys. Absorbing the row reversal instead would make the map
  dense at N/2 = 32768 diagonals -- the exact shape that lost 32.7% twice.
* **`project_qkv` fused with `to_batch` -- dead on PRECISION.** `p = 33` IS
  the scale, so the whole splittable budget is 33 bits and it must cover both
  a dense 4096-diagonal crossing table (`llama3_rect.cu:1300-1330`) and an
  `llround(w * delta_w)` int64 weight table (`llama3_rect.cu:125-128`). The
  optimum split lands near 7.8 bits against a measured 9.77.

### 18.4 What to measure, and where

`shared` 4 -> 5 takes the Algorithm 5 Galois set 10.0 -> 15.0 GiB and puts
every island op a limb higher. Two independent cost models disagree by 2x:
+1.1-1.6 s from the measured per-leg proxies, +3.4-6.6 s from limb scaling.
**Net is between +4.2 s better and -1.3 s worse, and nothing measured settles
it.**

E1 is the cheapest to validate and needs no island keys at all, so it fits an
A6000: run `HEONGPU_TB_STAGE=norm` and read the exit level off the ledger. It
should move 3 -> 4 with the block error unchanged.

---

## 19. The SoftMax on 16 batched inputs: the layout is forced (2026-08-13)

Branch `HEonGPU_LLama3_8B_batch16_softmax`, off `HEonGPU_LLama3_8B_batch16`
@ `4c4ce17`. Scope: the SoftMax seam of `Llama3BatchOperator` — everything
between the two Algorithm-4 products. PCMM, CCMM, RMSNorm and FFN are other
sessions'. Derived by a 13-agent workflow (5 readers, 4 layout costings, 4
adversarial lenses); every number below is read off the code at the target
shape `N = 4096, d = 128, k = 32, batch = 16, heads = 32, kv_heads = 8`, not
at the `d = 8` shape the existing suite runs.

### 19.1 The premise does not hold, and the reason is worth keeping

**Sixteen batched inputs do not make the SoftMax cheaper per input. Nothing
does, because the cost is set by the data volume and not by which index rides
the batch axis.** Measured per input, every SoftMax quantity on this path is
*identical* to the single-input rect path — not similar, identical:

| per input | batch-16 | rect (Alg 5) |
|---|---:|---:|
| exp evaluations | 32 heads x 128 parts / 16 = 256 | 2 groups x 128 x 1 = 256 |
| reciprocal fits | 32 x R / 16 = 2R | 2 x R x 1 = 2R |
| crossing rotations | 2 x 32 x 128 x 22 / 16 = 11,264 | 2 x 2 x 128 x 22 = 11,264 |
| reduction rotations | 0 | 0 |
| SoftMax levels | identical | identical |

Both encodings fill the same 2048 slots with 2048 (something x query) pairs,
and the SoftMax never reads what the "something" is. The bridge charges per
*value*, not per ciphertext, so permuting which index sits on the batch axis
moves nothing at all. Batching buys throughput and a far smaller key set
(127 + boot indices against Algorithm 5's 2047); it does not buy the SoftMax.

What sixteen inputs *do* buy the SoftMax is **that the layout stops being a
choice**, and the forced layout is the optimal one.

### 19.2 The layout is forced, and it is the unique minimum

`batch = k/2 = N/(2d)`, so sixteen inputs at `N = 4096` pins **`d = 128`** —
which is also Llama-3-8B's `head_dim` and also the token block. Then
`d * (k/2) = 2048 = N/2` exactly: the slot vector is *entirely* spent on
(input, query) and **the key axis has nowhere to go but the ciphertext axis.**

That axis is the free one. `count = 1`, `sum_strided`'s loop is dead code, and
the denominator is a slot-wise addition of the `d` parts: **zero rotations,
zero levels, zero Galois indices.**

Four candidate layouts were costed and all lose:

| layout | verdict | why |
|---|---|---|
| **A** key across ciphertexts | **adopted** | reduction 0 rotations; both seams conversion-free |
| B key inside the ciphertext | viable, worse | +17.4-31.3% rotations, +153-186% ct-ct mults, 0 levels saved |
| C SoftMax in coefficient form | **impossible** | see 19.3 |
| D shrink `d` | viable, worse | 1.20x at `d`=64, 2.17x at `d`=32; `d`=128 is the constraint boundary *and* the optimum |

A and B are the endpoints of a one-parameter family — put `K` of the 128 keys
inside the ciphertext, `128/K` across them — and **every term of that family is
monotone increasing in `K`**: reduction rotations `K*log2(K)`, reciprocal fits
`K`, plus a gather the row bridge cannot perform because the key *is* the
ciphertext index. So `K = 1` is not merely better than `K = 128`; it is the
unique minimum of the whole family, simultaneously on reduction rotations (0),
reciprocal amortisation (one fit per 2048 denominators, the ceiling) and
conversion (the bridge's native codomain, 1 level, the floor for a `d`-point
linear map).

**Layout D's caveat corrects a note already in the tree.** "Smaller `d` is
cheaper per value" assumed a *full* batch axis. With sixteen inputs fixed, a
column holds `16d` useful values rather than `N/2`, occupancy is `d/128`, and
the conclusion inverts: `d` = 64 and 32 are 1.27x and 1.82x *worse* per value,
and the CMT term (which scales as `1/d^2`) buries the crossing term's shallow
4.5% minimum at `d` = 64 six times over.

### 19.3 Both seams are conversion-free, and the crossing is unavoidable

Algorithm 4 hands the scores back column-wise **with the key on the columns**,
which is the axis the SoftMax reduces; `to_slots` consumes that output
unmodified. `from_slots` hands `P` back as exactly the left operand the value
product wants, with `V` still carrying the key on its rows where the projection
left it. **No transpose, no block map, no permutation, no new Galois key** —
verified in code at the target shape, not inferred.

The one conversion that remains is the row bridge, and it is *provably*
mandatory. For the row-basis map `R_N -> R_k^d` to be entrywise it would have
to be a ring isomorphism, i.e. `X^d - Y` would have to split over `R_k`; the
torsion units of `R_k` are `+-Y^j` and `(+-Y^j)^d = Y` needs `gcd(d, 2k) = 1`,
impossible for powers of two. Equivalently: `R_N (x) Q = Q(zeta_2N)` is a
**field**, so its only idempotents are 0 and 1, while a product ring has `2^d`.
A ciphertext product convolves over the row index instead:
`(ab)_r = sum_{i+i'=r} a_i b_{i'} + Y * sum_{i+i'=r+d} a_i b_{i'}`. **Not one
SoftMax step — not the exponential, not the mask, not the square, not the
normalising product — can act on a score matrix encryption.** That is a
property of the ring, not a limitation of Kang's algorithms.

### 19.4 So the levers are levels and encodes, and neither had been taken

**Measured**, Sicily GPU 2, tip `af90394`, `d = 128`, `batch = 16`, a full
causal `128 x 128` score block per input against the true SoftMax:

| configuration | seam levels | worst abs error |
|---|---:|---:|
| baseline as it stood (`Ed=Id=15, Nw=2`, no folds) | **30** | — |
| Newton-refined (`Id=15, Nw=2`), exp map folded | 29 | 1.76e-02 |
| no folds, `Id=63, Nw=0`, calibrated | 26 | 2.72e-03 |
| **both folds, `Id=63, Nw=0`, calibrated** | **23** | **2.84e-03** |
| `+ refresh_denominator` (auxiliary track) | **11** | *predicted, not run* |

**30 -> 23 is seven levels, and the circuit gets 6.2x MORE accurate on the
way.** The ledger the code advertises is the ledger it spends — `predicted 23,
spent 23`, measured by depth in and out rather than inferred from a limb
allocation that merely fits.

**The eleven levels the derivation offered were not eleven, and why is the
useful part.** `fold_affine_into_mask` is silently ANDed with
`inverse_newton <= 0`, so taking the fold deletes the Newton refinement and the
reciprocal becomes a bare fit. At degree 15 that fit is **33% wrong**, and
calibration does not rescue it: a causal triangle starting at `u = 0` has a row
attending to ONE key, whose sum of squares after a round is exactly 1, so
`concentration` is `d` whatever is measured and the later rounds are fitted
over `[0.5/d, 1.5]` — **384:1 at d = 128**. Degree 63 carries it, which is
exactly what the rect path already uses in production and for exactly this
reason. So the fold costs two levels of fit degree and nets six, not ten.

The two folds themselves are free, and now measured so: 26 -> 23 levels at
2.72e-03 against 2.84e-03, the difference being noise.

`sum_lo`/`sum_hi`/`concentration` had never been set anywhere on this path, so
the reciprocal was being fitted over exactly the worst-case interval Section
4.3 warns against. Calibrating them is what makes degree 63 enough rather than
degree 127 — and it is also the reason `refresh_denominator` matters more here
than the level count alone suggests: the auxiliary track takes the fit off the
wide track entirely, so the degree stops being a budget item at all.

`refresh_denominator` is the largest single lever and it is nearly free here:
the denominator is **one** ciphertext however many parts the key axis is cut
into, so the whole fit moves off the wide track for one bootstrap per round
while the `d` parts pay only the square and the normalising product. It was
reachable through the config and *threw*, because `attention()` passed no boot
key.

**A correction to how levels are argued on this path.** "The SoftMax is the
deepest stretch, and the chain is the worst stretch + 26" is a *rect* relation
that needs a bootstrap between stretches. There is no bootstrap anywhere on the
batch path, so every level saved buys one limb **1:1** — the SoftMax is not
privileged, it is merely the largest single consumer (30 of the attention
test's 33). That makes the level argument stronger, not weaker.

The third lever is not levels at all: the causal masks depend on the query and
key indices alone, never on the head, yet were rebuilt and re-encoded inside
the head loop — `H*d = 4096` encodes of a 2048-slot vector per attention call
where `d = 128` would do. Each is an NTT over the whole live chain. This is the
same disease the bridge's diagonal cache already cured, and it is **not**
specific to this path: `multiply_vector` re-encodes unconditionally, so the
rect path pays it too and can take the fix from `Llama3Operator`.

### 19.5 What was implemented

`Llama3BatchOperator::softmax_seam()` owns everything between the two
Algorithm-4 products, so `attention()` only calls it — deliberately, because
`attention()` belongs to the CCMM session. With it:

* `softmax_layout()` and `softmax_seam_levels()` **report** the layout and the
  closed-form ledger instead of restating them in a comment, and the tests
  assert against them. The ledger test measures depth in and out, which is a
  two-sided pin; a limb allocation that merely fits is one-sided.
* `score_shift_rows` (per query) and `score_shift_slots` (per input *and*
  query) — the second expressible only because the batch axis carries
  independent inputs. The seam scales the shift by the folded domain map
  itself, which is the trap the rect path leaves to its caller.
* An encoded-mask cache on `Llama3Operator`, keyed by `(mask id, fold weight
  bits, depth, rescale prime)`. The weight is in the key by its exact bits
  rather than by a convention, because it is derived from the fitted ranges and
  two calibrations would otherwise share an entry.
* Hoisted crossings on by default in the seam, restored around the call.

Defaults keep the old numbers: the folds are opt-in through
`BatchAttentionConfig::seam`, so nothing already measured moves.

### 19.6 The ceiling nobody had counted

**About half of every SoftMax ciphertext is causally dead.** A causal `d x d`
score block has `d(d+1)/2` live entries of `d^2`, i.e. 50.4% live at `d` = 128
— so 49.6% of the exp evaluations, the squares, the normalising products and
both crossings are spent on values the mask is about to zero. That is worth
roughly 44% of the attention rotation budget.

It is also provably unreachable in this encoding: the key axis *is* the
ciphertext axis, so a part cannot be dropped without dropping a key position
for every query at once, and Algorithm 4 requires square `d x d` operands
either side. Recovering it needs a triangular block schedule over a token grid,
which is a different shape of attention and not a SoftMax change. **Name it as
the ceiling and do not go looking for it inside the seam.**

### 19.6bis Validation

Sicily **GPU 2** (0 was at 99% on an external job, 1 had picked up a 12 GiB
one; 2 was idle at 0% with only the long-standing 4.3 GiB `dp_relu`
residents). Build: CMake 3.31.6, devtoolset-10, CUDA 12.0, sm_86, tree
`JHJun/HEonGPU-b16sm`, tip `af90394`.

**112 tests green, no regressions:**

| suite | result |
|---|---|
| `ckks_llama3_batch_softmax` (new) | **9/9** |
| `ckks_llama3_batch` | 12/12, attention 2.65e-08 |
| `ckks_llama3` | 53/53 |
| `ckks_llama3_rect` | 38/38, attention 9.15e-05 |

The last three matter because the seam touched code they own: `attention()`
was rewired to call `softmax_seam`, and `Llama3Operator::softmax` grew a
mask-id overload that the rect path reaches through the old signature.

What the new suite pins, beyond the numbers above: the reduction is 0
rotations and the seam asks for 0 Galois indices Algorithm 4 did not already
force; the bridge and CMT index sets are equal; the mask cache serves 0, 0 and
then all 128 encodes across three identical seams and returns **exactly** the
same plaintext each time; hoisted crossings are **bit-identical**; a per-query
shift agrees with the scalar one to 1e-6; and the four contracts that must
fail loudly do.

Two of the first run's six failures were the test's own and are worth naming.
The bit-identity tests encrypted afresh for each run, so they compared two
samples of the encryption noise and reported a 4e-7 "difference" that had
nothing to do with what they were testing; both now run every configuration
off one ciphertext. And the level-delta assertion carried a spare `+1` for the
exponential's affine map that both configurations already fold.

### 19.7 Left undone, deliberately

* The auxiliary track is wired and validated as a *contract* (it throws
  without a boot key); its numerical behaviour on this path is untested,
  because nothing on the batch path bootstraps at all and standing up a boot
  fixture is the refresh-schedule session's job, not the seam's.
* `heads > 1`, `kv_heads < heads` and `per_head > 1` still have no host
  reference anywhere in the batch suite. The GQA reuse is what makes this
  path's K/V cheaper than the rect path's host expansion, and it is asserted
  rather than checked.
* Two defects found while reading, both outside this seam and reported to the
  sessions that own them: the V level-drop allocates `n*Q_size` and launches
  two kernels *per level per ciphertext* (`operator.cu:1454-1485`, with its own
  TODO) and re-drops the same kv blocks `group` times under GQA; and the
  bridge's plaintext cache holds four sets with **widest-first** eviction,
  which is backwards for a block that walks eight crossings monotonically down
  the chain.

## 20. The batch-16 Algorithm-1 PCMM: the layout is forced, and the plaintext leaves the subring (2026-08-13)

Branch `HEonGPU_LLama3_8B_batch16`. Sixteen independent inputs ride the batch
axis that §5 and §6 spend on contracting **one** input, so the projection can go
back to Algorithm 1 — no rotations, no key switching, no rotation keys.
Established by a 13-agent workflow (6 readers, 3 designers, 3 adversarial
verifiers, 1 synthesis) at `4c4ce17`; 8 of 23 claims were refuted and are
recorded as corrections in §20.6 rather than repeated.

### 20.1 There is exactly one legal configuration

`batch = k/2` and `k = N/d` (`batchmatrix.cu:200-204`), so `batch = 16` forces
`N = 32d`. Two more constraints close it:

| constraint | source | gives |
|---|---|---|
| `q_channels % (heads*d) == 0`, equal q/kv quotients | `llama3_batch.cu:814-827` | `d` divides `head_dim = 128`, so `d <= 128` |
| `MIN_POLY_DEGREE 4096` | `defines.h:15` | `N >= 4096`, so `d >= 128` |

**`N = 4096`, `d = 128`, `k = 32`, `batch = 16`, `per_head = 1`.** logN 13 is
*not* available at batch 16: `d = 256` would make a head smaller than one
Algorithm-4 block and `attention()` throws.

- **`d` is the sequence length: 128 tokens, and there is no KV cache on this
  path.** `rows == layout.d` is enforced at `batchmatrix.cu:309-311`,
  `llama3_batch.cu:535-540`, and by `causal_column_mask` running `u` over
  `0..d-1`. Llama-3-8B's context is 8192, so this is 1/64 of it. This is the
  structural price of `batch = 16` and it is not tunable.
- **Model width is free.** Channels are ciphertexts; 4096 / 1024 / 14336 all
  divide 128, so there is no padding. Slot occupancy is exactly 100%:
  `d * batch = 2048 = N/2`.
- **Security is `none` with nowhere to go.** The 128-bit cap at `N = 4096` is
  109 bits of log PQ; an `L = 70` chain is 3240 — **29.7x over**. Unlike the
  rect path this one cannot change ring without breaking `d | head_dim`.

### 20.2 The encoding, in and out

An activation is a `d x C` matrix over `R_k` held as **`C` ciphertexts, one per
channel**, in the coefficient domain. For token `u`, channel `j`, instance `b`:
ciphertext `j`, coefficients at ring index `u + 128t` for `t = 0..31`
(`batchmatrix.cu:324`), with

    coeff_{u+128t}(ct_j)
        = round( scale * (2/k) * sum_{b<16} Re( X_b[u][j] * conj(zeta^{5^b t}) ) )

and `zeta = exp(i*pi/32)`. **No single coefficient is a
`(token, channel, instance)` value** — the instance index is an *evaluation
point* of the `R_k` entry, spread over all 32 coefficients of the stride-128
run.

`pcmm`'s output is structurally identical to its input, so projections chain
with no conversion. The reason is that `bm_gemm_kernel` is **pointwise in the
subring index `s`** — `C[i][j][s] = sum_t A[i][t][s] * P[t][j][s]`
(`kernel/batchmatrix.cu:379-395`) — and `s` is the batch axis, so the batch is a
pure passthrough. This is the asymmetry Algorithm 5 has and Algorithm 1 does
not: §5's `BlockAxis` exists precisely because the rectangular product consumes
the batch axis and returns the Y axis.

**Weights are stored transposed** (`in_channels x out_channels`, row-major) and
encoded at `rescale_prime(x)`, not at the nominal scale. A wrong scale here
fails **silently**: `add`/`add_plain` check depth, encoding and buffer size and
**never `scale_`** (`operator.cu:284-318`).

### 20.3 The seam to `Q K^T` — both CMTs on K are redundant

`cmt` is the **identity on `R_k`**: the automorphisms are `X -> X^(2kt+1)` and
`X^(d(2kt+1)) = X^(2Nt+d) = Y`, so `Y` is fixed pointwise and no batch slot can
move; it is the per-slot index transpose, an involution. `ccmm`'s body after
step 1 computes `A * Mat(bcmt)^T`, because the four step-2 GEMMs are launched
with **operands swapped** and `b_row_stride=1, b_col_stride=d`
(`batchmatrix.cu:1561-1572`), and `fold`'s trailing `cmt` transposes back.

So the explicit `cmt(K)` at `llama3_batch.cu:855-869` and `ccmm`'s internal
step-1 CMT at `:1493-1500` **compose to the identity**, and a `ccmm` variant
that elides step 1 computes `Q K^T` with no CMT at all. Per sublayer that
deletes 5,080 rotations and takes the score product from **17,304 to 12,224**
key switches (-29.4%).

**But it is at most 2.0% of the sublayer**, once `softmax()`'s own
relinearisations are counted. Do it for the **4.375 GiB** of resident `key_t`,
the **17.5 GiB** of `bcmt` deep-copy traffic, and for halving the CMT noise
terms multiplied into the score. Two guards must come with it: `cmt` reaches
`rotate_rows`, which is the **only** inspection of the right operand's
`rescale_required_` / `relinearization_required_` in all of `ccmm`. **Keep plain
`ccmm` for `P V`** — that product genuinely wants `V` column-wise as the
projection left it.

### 20.4 Ledger: X enters attention to SoftMax input ready

Per head, `per_head = 1`:

| step | encoding | levels | key switches |
|---|---|---|---|
| `project` q/k/v (Algorithm 1) | matrix -> matrix | 1 (shared) | **0** |
| `S = Q K^T` (`ccmm`, step 1 elided) | matrix -> matrix | 1 | 382 |
| `to_slots(S)` over `d = 128` columns | **matrix -> slot** | 1 | 2,816 |
| `add_constant(-score_shift)` | slot | 0 | 0 |
| **total** | | **3** | **3,198** |

**Per sublayer 102,336 key switches, of which `to_slots` is 87.2%.** The
crossing is the budget; the products are not. Crossing cost per column at
`d = 128`: `baby_steps()` returns `n1 = 16, n2 = 8`, so **22 key switches**
(not 127), 128 plaintext multiplies, 8 rescales, **1 level**, **0 new Galois
keys**. The diagonal encode is paid **once per sublayer**, not 32 times —
`bridge_plain_` is keyed on `(direction, depth)` and all 32 heads reach
`to_slots` at one depth.

**The SoftMax reduction is free.** `S` has the key on the ciphertext axis, so
`count = 1` makes `sum_strided`'s `for (t = 1; t < count; t <<= 1)` run zero
iterations and `strided_rotation_indices` return empty. The landing layout is

> slot `b + 16u` of `to_slots` output ciphertext `j` holds entry `(u, j)` of
> instance `b` — instance fast (stride 1, 16 wide), token slow (stride 16, 128
> wide), key across ciphertexts. `16 * 128 = 2048 = N/2`.

### 20.5 The plaintext does not belong in the subring at all

**A weight that does not vary across the batch is a CONSTANT in `R_k`.**
Definition 1 puts the batch index in the *evaluation* domain, and the constant
polynomial is the unique preimage of a constant vector — so a Llama weight
encodes to a **delta at `Y^0`**. Confirmed at `k = 8/16/32/64`: `r[0] = 1.0` to
the bit, `max|r[t>0]| = 1.8e-16`.

`encode_shared_plaintext_matrix` therefore takes the real matrix directly, and
`pcmm` dispatches to a scalar route that drops the four subring transform passes
with it. The two routes are the same arithmetic: the subring transform is
`Z_p`-linear and the plaintext is a scalar, so
`T^-1(sum_t T(a_t) v_t) = sum_t v_t a_t` holds **identically in `Z_p`**, which
the test asserts on raw device words rather than through a tolerance.

Arithmetic at `N=4096, L=70, C_in=4096, cols=128` (**not yet measured**):

| | general | shared |
|---|---|---|
| host encode per block | 111.7 G | 7.0 G complex MACs |
| H2D per column block | 128.0 MiB | **4.0 MiB** |
| `plain_` | 8.750 GiB | **0.273 GiB** |
| `A0`+`A1`, `C0`+`C1` | 18.047 GiB | **0** |
| one-block peak | 44.84 GiB | **18.32 GiB** (-59.1%) |

**It is NOT bit-identical to the general encoder, and the fast path is the more
accurate side.** `BatchMatrixEncoder::encode` sums `k/2` double products per
coefficient and leaves a ~1.8e-16 relative residue at powers that should be
zero; `llround` kills it below `scale*|w| = 2^51.5` and not above — and
`project` encodes at `rescale_prime`, a **40-to-60-bit prime**. So there is a
regime where the general encoder puts hundreds of integer units of pure error on
powers of `Y` carrying no information. An equality regression against it must
not be written.

The general path is untouched and every other `pcmm` caller still reaches it
through `encode_plaintext_matrix`, so **the rect/nobatch baselines are
unaffected**. `rectangular_pcmm` rejects the shared form outright: it reads
`[limb][row][col][k]` where the shared buffer is `[limb][row][col]`, and
Algorithm 5 contracts over exactly the axis a batch-invariant weight lacks.

### 20.6 Corrections — claims that failed adversarial verification

- **"`pcmm` costs zero levels."** True of the routine, false of the operation:
  it marks a rescale that `project` spends. **A projection costs 1 level.** Zero
  *key material* is exactly true.
- **"`column_block` must be passed explicitly."** It already defaults to 128
  (`llama3_batch.cu:578-581`). What is missing is *contraction* (row) blocking —
  and after §20.5 that is much less urgent, because the peak it was sized
  against was dominated by `A0/A1` and `plain_`.
- **"The score product loses 254 of 636 key switches per head."** MHA-only. K is
  transposed once per **KV block** (8 of them) and shared across
  `heads/kv_heads = 4` query heads: the explicit CMT is **1,016** rotations, not
  4,064.
- **"At fixed `d`, per-instance crossing cost improves with `N`."** Only the
  key-switch *count* falls (as `1/N`); per-rotation *work* scales with `N`, so
  per-instance cost is **flat**.
- **"`Llama3RectOperator::bootstrap(BatchActivation&, ...)` already exists."**
  False — that overload takes a `Ciphertext` (`llama3_rect.cuh:464`).
- Two stale comments in `llama3_batch`: `llama3_batch.cuh:212-213` still says
  `to_slots` costs `d - 1` rotations (it is `n1+n2-2 = 22` since BSGS), and
  `llama3_batch.cu:511` claims the rounding keeps `n1 <= n2` (at `d = 128` it
  returns `n1 = 16, n2 = 8`).

### 20.7 Open, with the experiment that closes each

1. **Does bootstrapping carry a matrix encryption on the batch path?** Nothing
   has measured it and there is no refresh in `llama3_batch.cu`
   (`llama3_batch.cuh:511-516`). This gates any stack, not just this branch.
2. **Does the step-1-elided `ccmm` produce `A B^T` on hardware?** Proved from
   source, unexercised in the tree, and the failure mode is silent garbage.
   Assert against host `Q K^T` **and** that it does *not* match `Q K`.
3. **Does `k = 32` work at all?** The CMT and CCMM sweeps top out at `k = 64`;
   `k = 32` is unexercised everywhere.
4. **Every cost figure in §20.5 is arithmetic, not measurement.** No profile of
   this branch exists yet.

## 21. The batch-16 Algorithm-4 CCMM: the output orientation is forced, and two of the four transposes were free (2026-08-13)

Audited on `HEonGPU_LLama3_8B_batch16_ccmm`, forked from the batch-16 branch at
`5f47372`. Every claim below was read off the source and then put through an
adversarial pass that tried to refute it; figures are marked as counts (exact,
from the code) or projections (arithmetic, unmeasured). **Nothing in the tree
had ever run this shape**, so treat every millisecond as derived. §20 owns
Algorithm 1; this section owns Algorithm 4 and the transposes around it.

Shape: `batch = k/2 = 16` forces `k = 32`; `MIN_POLY_DEGREE` and the "whole
number of `layout.d` channel blocks" guard (`llama3_batch.cu:814-820`) force
**`d = 128`, `N = 4096`, `per_head = 1`** — the only legal batch-16
configuration. `heads = 32`, `kv_heads = 8`, `group = 4`.

### 21.1 The layout contract

`ccmm` returns a **column-wise** matrix encryption (Definition 2) of `A*B`:
output ciphertext `j` is column `j`, and the product's ROW index sits in the
polynomial coefficients at `i + d*t`. For `S = Q K^T` that puts the **query in
the coefficients and the key across ciphertexts**.

| # | step | output form | coefficient axis | ciphertext axis | levels |
|---|---|---|---|---|---|
| 1 | `project()` Q/K/V (Alg. 1) | BATCH | token | channel | 1 |
| 2 | `S = Q K^T` (Alg. 4) | BATCH | **query** | **key** | 1 |
| 3 | `to_slots` | SLOT | — | key | 1 |
| 4 | SoftMax | SLOT | — | key (reduced axis) | S |
| 5 | `from_slots` | BATCH | query | key | 1 |
| 6 | `O = P V` (Alg. 4) | BATCH | query | head-dim channel | 1 |
| 7 | output projection | BATCH | token | channel | 1 |

Attention sublayer depth is **6 + S**. The causal mask and the score shift are
free.

**The output orientation is forced, not chosen, and this is the single most
important thing for anyone building on top.** It is tempting to read the two
CMTs inside `fold` (`batchmatrix.cu:1624`, called at `:1628-1629`) as output
relabelling and skip them to save `2(d-1)` rotations. They are not. They are
what moves `Toep(s)` from the right of `C_x1` to the left, so the degree-two
recombination at `:1644-1656` reads the `sk`-grading where it expects it.
Skipping them returns

    (AB)^T + (Toep(s)*W - W*Toep(s)^T),    W = B^T * A1^T

and `A1` is the left operand's **pseudorandom** `c1`. The residue is not small
and is not a function of `(A,B)` at all: encrypt the same `A` twice with fresh
randomness and the two results differ. The saving is **0**, not 254 rotations.

That the orientation also happens to be the one the SoftMax wants is a bonus,
not the reason. With the key across ciphertexts `attention` sets
`softmax.count = 1` (`llama3_batch.cu:932-934`) and `sum_strided`'s loop never
executes (`llama3.cu:758-763`): the denominator is a slot-wise sum of `d`
ciphertexts at **zero rotations and zero levels**. The transposed layout would
cost `log2(d)` rotations per round plus `d-1` extra reciprocal chains.

**The one conversion that cannot be removed** is `to_slots`/`from_slots`. The
SoftMax evaluates polynomials and therefore needs slots; `ccmm` produces
coefficients. There is no bridge-free layout: the only support on which
polynomial evaluation acts entrywise on matrix ENTRIES is the `R_k` row layout
(one ciphertext per entry), which is `d = 128x` less dense and costs **720,896**
key switches per head to enter and leave instead of 5,632 — a factor-`d`
regression on both sides of the ledger. Fusing the output CMT into the bridge is
also dead: the CMT sits inside `fold`, with the degree-two combine and `d`
relinearisations after it, so pulling the bridge's baby rotations back through
it means relinearising every baby copy of both halves (**128 -> 4,096 relins per
crossing**), and the repo already measured fused crossings 32.7% slower than
staged (§16).

**Key material:** the CMT's rotation indices and the bridge's shifts are the
same set — the `d-1 = 127` non-zero multiples of `k/2`, because the subgroup
generated by `5^(k/2)` is the unique order-`d` subgroup of the one generated by
5. Already unioned at `llama3_batch.cu:760-768` and pinned by
`test_ckks_llama3_batch.cpp:224-236`. Algorithm 1 takes no Galois key at all.

### 21.2 Where the sublayer's key switches actually are

Counts, exact, per attention sublayer at the shape above — **the CCMM is not
where this path spends its time**, which is worth knowing before optimising it:

| bucket | key switches | share |
|---|---|---|
| bridge crossings (64 x 128 cols x 22) | 180,224 | 65% |
| SoftMax relinearisations (0 rotations) | ~62,144 | 22% |
| CMT rotations (200 CMTs x 127) | 25,400 | 9% |
| `ccmm` relinearisations (64 x 128) | 8,192 | 3% |

One `ccmm` is `3(d-1) + d = 4d-3 = 509` key switches and exactly **1 level**,
marked but not spent — the caller rescales (`llama3_batch.cu:699`). It also
issues `d + 6*log2(d) + 14 = 184` device-wide barriers.

### 21.3 What changed

**1. The score product no longer transposes K twice.** `attention` built `K^T`
with a CMT per channel block and handed it to Algorithm 4, whose step 1 opens by
transposing its right operand — so the pair composed to the identity. A CMT is a
matrix transpose and hence an involution on the encoding, so the projection's
own column-wise output IS the row-wise operand step 1 was trying to build.
`ccmm` gained `RightOperandForm`; `attention` passes `k.column` straight in.
Removes the 8 standalone transposes and the 32 step-1 CMTs.

**2. V's transpose is shared across the query group.** `P V` genuinely needs V
row-wise, so that transpose is real — but `group = 4` consecutive heads share
one V block and Algorithm 4 was transposing it once per head. It is now
transposed once per distinct block, in place, at the level `P` meets it at
(dropping before transposing, since a rotation costs what the live limb count
says). The per-head deep copy went with it. 32 -> 8 CMTs.

**3. Two host-side costs no GPU-busy profile can see.** `cmt` rebuilt an
`N/2`-entry `std::map` of the whole rotation group on *every* call to serve `d`
lookups — measured at **307.7 us per call**, 200 calls a sublayer, sitting
between a device synchronise and the first rotation with the GPU idle. It is now
built once per operator. And the combine loop's per-column
`cudaDeviceSynchronize` (`d` of them, fencing one addition and two copies the
default stream already orders) is gone.

Net, per attention sublayer: **200 -> 136 CMTs**, 25,400 -> 17,272 rotations,
4,096 ciphertext deep copies -> 0, and ~61 ms of GPU-idle host time removed.
That is **-24%** of the Algorithm-4 family's key switches and about **-3%** of
the sublayer's. Change 1 is also strictly *more accurate*: the step-1 CMT's
key-switching noise is multiplied by the left operand in the GEMM and is the
dominant term in the product's error, which is why the reference test needs
asymmetric scales (`test_ckks_batchmatrix_gpu.cpp:960-965`).

A guard came with them. `ccmm` validates count and level and then writes
`scale_a * scale_b` unconditionally, and `product()` — which `attention` calls
directly on borrowed columns — never screened its operands; only `matmul()` did.
Two operands at equal level and unequal scale passed every guard and came back
as a silently wrong product.

### 21.4 Rejected, with reasons, so nobody re-tries them

**Fold the four cross-products and transpose once (3 CMTs -> 2), with or without
relinearising first.** *Algebraically false, and silently so.* The product is
`C00 + C01*Toep(s)^T + Toep(s)*C10 + Toep(s)*C11*Toep(s)^T`
(`batchmatrix.cu:1544-1548`): the middle terms carry the secret on **opposite
sides**. Folding to `(C00, C01+C10, C11)` needs `Toep(s)*M = M*Toep(s)^T`, which
holds iff `s` lies entirely in `R_k` — probability `3^-(N-k)` for a ternary
secret. The transpose is `R_k`-linear but not `R_N`-linear:
`T(Toep(s)*M) = M^T*Toep(s)^T`. It agrees with the correct answer exactly when
the left operand's `c1` is zero, so **it would decrypt at the right level and
scale and return a plausible wrong matrix**, and the exact CCMM test cannot see
it because it memsets `c1` to zero. The code warns about precisely this at
`batchmatrix.cu:1583-1586`.

**Karatsuba on the four GEMMs.** Rejected twice. The bilinear map has rank
exactly 4 — no three-multiplication scheme exists. And the natural repair needs
only `C01+C10`, which is the fold above, which is wrong.

**Defer relinearisation past the bridge or past the `per_head` sum.** A
degree-two ciphertext cannot be rotated without a key for the square of the
secret, and `multiply_plain`, `rescale` and `apply_galois` are all hardwired to
two components. Counted properly it is a 1.96x loss, and the saving side is zero
anyway — deferral *moves* the `d` relins rather than removing them. Past the
`per_head` sum it is sound and worth exactly zero, because `per_head = 1` is
forced.

**Skip the output CMTs and hand back the transpose.** Not a layout choice; a
correctness break. See §21.1.

**An `R_k` row layout to eliminate the bridge.** Real algebra, factor-`d`
regression. See §21.1.

### 21.5 Open, and only a GPU run can settle it

1. **What does one `ccmm` cost at `N=4096, d=128, k=32`?** Two defensible
   projections disagree by 2-3x (35-90 ms vs 45-55 ms). Every ranking below
   change 3 depends on which. The NVTX ranges already bracket every step; one
   `nsys` capture answers it.
2. **Wall versus GPU-busy.** Every host cost here is invisible in a GPU-busy
   figure by construction. If the gap is large, change 3 is the biggest of the
   three, not the smallest.
3. **Turn on `hoisted_crossings_` before touching Algorithm 4 again.** It is one
   bool (`llama3_batch.cuh:662`), it attacks the 65% bucket rather than the 9%
   one, `operator.cuh:1218-1237` says the result is identical to the bit, and it
   was worth 14% on the nobatch path. Nothing on the batch path calls
   `set_hoisted_crossings`.
4. ~~**Noise gained by deleting the two K CMTs.**~~ **Settled — measured
   below.**

### 21.6 Validation (2026-08-13, Sicily GPU 2)

Built at `95e1d5e` in an isolated tree (`JHJun/HEonGPU-b16ccmm`), CUDA 12.0,
sm_86. **77 tests, all green, zero failures**: batch-matrix GPU 12, batch-matrix
host 5, llama3-batch 13, shared-PCMM 6, rect batch-matrix 3, llama3-rect 38. The
last three suites are regressions — the rect path shares `cmt` and `ccmm` with
this one, so the schedule cache and the removed barrier had to be shown harmless
there too.

**Open question 4 is answered, and the direction was right.** The score
product's error, same shape and same scales:

| path | worst absolute error |
|---|---|
| with step-1 CMT (`CCMMMatchesReference`) | 3.21e-04 |
| without it (`CCMMRowWiseRightOperandTransposes`) | **9.57e-05** |

**3.4x more accurate**, about 1.75 bits, from deleting a transpose that was
computing the identity. That is the step-1 CMT's key-switching noise leaving the
product, and it is why the reference test needed asymmetric scales. The two
paths agree to 3.59e-04 (`CCMMRowWiseMatchesDoubleTranspose`) — i.e. to within
the noise of the noisier of the two, which is the most agreement that is
available.

Grouped-query attention, executed for the first time in this repo, lands at
3.02e-08 against a host reference — indistinguishable from the single-head path
at 2.41e-08, so sharing V's transpose across the group costs nothing in
accuracy.

The two counts that are still projections and not measurements: the per-sublayer
key-switch table in §21.2, and every millisecond in §21.3. No profile of this
branch exists yet.

## 22. Batch 16, and the layout the non-linear layers actually need (2026-08-13)

A different question from every section above, and it changes the answer to
all of them: **sixteen independent inputs at once, not one.**

Sections 1-18 are the single-user problem. Kang's Algorithm 5 spends the batch
axis of the matrix encryption on contracting ONE input, which is why the
rectangular path exists and why `N/2 - 1` Galois keys are its wall. With
sixteen users the axis has somewhere better to be, and `llama3_batch.cuh`
already said so in its own header: Algorithm 1's `k/2` packed matrices are
`k/2` INDEPENDENT INPUTS, and that is "the right answer when there are `k/2`
users."

This section is the non-linear half of that path -- RMSNorm, the reshape and
SwiGLU. The SoftMax, the batch CCMM and the batch PCMM are three other
sessions on the same encoding.

**Everything below that is not marked MEASURED is arithmetic over the source.**

### 22.1 Batch 16 pins the ring, uniquely

`batch = k/2 = 16` forces `k = 32` and `d = N/32`. Three independent
constraints then coincide:

| constraint | source | consequence |
|---|---|---|
| tokens per block `== d` exactly | no token blocking exists on this path | `d = 128` for a 128-token block |
| `d` divides `head_dim` | Algorithm 4's operands are square at `d` | `head_dim = 128` gives `per_head = 1` |
| `batch = N/(2d)` | the ring | `N = 32d = 4096` |

So **(N = 4096, d = 128, k = 32, batch = 16) is the unique solution** for
Llama-3-8B's head dim at 128 tokens and batch 16. `N = 8192, d = 256` is
legal as a layout but `attention()` rejects it -- `per_head` would be 0.5.
`N = 8192, batch = 32` is legal and costs exactly 2x for exactly 2x the
instances; bytes per instance are invariant, because ciphertext bytes scale
with `N` and so does `batch`.

The slot map is `slot b + (k/2)*u of the ciphertext for column j holds entry
(u, j) of instance b`, a **bijection onto all 2048 slots** -- 16 instances x
128 tokens, no dead slot, no packing mask anywhere.

### 22.2 The layout answer: there isn't a second one

**A channel is a whole ciphertext.** That one fact settles the layout
question the non-linear layers were supposed to have:

- **RMSNorm's channel reduction is a slot-wise ADDITION.** `count = 1` makes
  `Llama3Operator`'s reduction loop not execute, so the mean costs no
  rotation, no mask, no Galois key and no level. On the rectangular path the
  same reduction is a blocked span of `k/2` costing `2*log2(k/2)` key
  switches and the level a mask costs. This is the one place the batch
  encoding is strictly better, and it is not a small place.
- **The learned gain is a CONSTANT**, not a slot vector, so it folds into the
  projection that reads the normalised stream: `W^T diag(g) y =
  (diag(g) W)^T y`, exactly, on the host, free.
- **The reshape is index arithmetic.** Head `h`, lane `c` is ciphertext
  `h*head_dim + c`; the split is pointer arithmetic and the concat is
  `push_back` in head order. **Grouped-query attention is index REUSE** -- the
  same run of ciphertexts is handed to `group` query heads -- so the key and
  value projections stay narrow by the full 4x at 32 heads over 8. The
  rectangular path has to widen the KV weights on the host instead, and buys
  nothing for it.
- **SwiGLU is slot-wise** and never reads an index.

There is no separate non-linear layout, so there is no conversion to reach
one. The seam question answers itself in both directions.

### 22.3 What it does cost: the bridge, and only the bridge

A Hadamard product and a polynomial fit are slot-wise; a matrix encryption is
a COEFFICIENT encoding, where multiplying two columns convolves them. So every
non-linearity crosses the row bridge.

Per column at `N = 4096, d = 128`: **22 key switches**, 128 plaintext
products, 8 rescales, one level. *Not* `d - 1 = 127` -- `bridge()` calls
`baby_steps()` unconditionally and it returns the balanced BSGS split
(`n1 = 16, n2 = 8`) whenever the override is 0, which is the default.
**BSGS is already on**; the doc comment on `to_slots` is stale by 5.8x.

One block at the 8B batch-16 shape bridges **67,584 columns** across 374 calls:

| leg | columns | key switches | share of bridging |
|---|---:|---:|---:|
| SwiGLU (gate, up, hidden) | 43,008 | 946,176 | 63.6% |
| RMSNorm x2 (down and back) | 16,384 | 360,448 | 24.2% |
| attention's SoftMax seam | 8,192 | 180,224 | 12.1% |

**Bridging is 98.3% of every Galois rotation in the block** and 88.3% of all
key switches including relinearisation. Everything else -- the CMTs, the
CCMMs, every slot-form reduction -- is the remaining 1.7% of rotations.

### 22.4 Eight levels a block, for four host-side multiplications

The batch path never wired through the config knobs the rectangular path grew.
Every one of these is EXACT -- a host-side rescaling of a plaintext, not
precision traded for depth:

| lever | saves | why it is free |
|---|---|---|
| `newton_iterations` 2 -> 0 | **6 levels** | a step is `y^2`, `x/2` times it, and the product; the fit alone already meets 12 bits over a calibrated range |
| `fold_mean_into_fit` | 1 level/norm | `1/sqrt(s/C + eps)` over the summed square is the same value from the same ciphertext as `1/sqrt(m + eps)` over the mean |
| gain folded into the weight | 1 level/norm, and `d_model` plaintext encodes | `W^T diag(g) y = (diag(g) W)^T y` |
| `fold_silu_domain_into_gate` | 1 level | the gate projection feeds the SiLU and nothing else, so `1/bound` needs no undoing |

The fifth -- the `1/sqrt` fit's own domain map -- has **no plaintext product to
ride on here, precisely because the reduction is free**. On the rectangular
path the mask carries it; here there is nothing between the reduction and the
fit. `sum_pre_scaled` lets a caller who owns the upstream weight carry it
there instead: the sum is quadratic in the input, so the factor is
`sqrt(domain_scale)` and `output_scale = 1/c` takes it back out of the
numerator in the fit's coefficients. Off by default, because folding one half
and not the other is silent.

### 22.5 The seam contract, for the other three sessions

```
CONSUMES  d_model ciphertexts, matrix-encryption (coefficient) form,
          one level and one scale, rows == d == 128.
PRODUCES  the same.
INDEX     matrix form: entry (i,j) of instance b lives in the k-coefficient
          run i, i+d, ..., i+(k-1)d of ciphertext j, as the EVALUATION of that
          run at zeta^{5^b}.
          slot form: slot b + 16*u of ciphertext j is (token u, channel j) of
          instance b.
RESHAPE   head h, lane c  ==  ciphertext h*head_dim + c.
          GQA: query head h reads kv head h/(heads/kv_heads). A VIEW, never a copy.
KEYS      the non-linear layers use no Galois index the CMT does not already
          require; the bridge's 22 BSGS shifts are a subset of the CMT's 127.
```

Nothing here asks the SoftMax, the PCMM or the CCMM to change anything.

### 22.6 The open question that is worth more than everything above

Bridging is 98.3% of the block's rotations and the SwiGLU is 64% of the
bridging. All of it exists because `project()` is *assumed* to need the
coefficient encoding.

It may not. Algorithm 1 encodes its weight through `BatchMatrixEncoder`, and
the weight is **one real matrix shared by all sixteen instances**, so its
`R_k` image is the CONSTANT polynomial -- and multiplying by a constant of
`R_k` is scalar multiplication. The projection degenerates to
`out_c = sum_j W[j][c] * ct_j`, a scalar multiply-accumulate ACROSS
ciphertexts, which cannot care what a ciphertext encodes. The bridge is linear
and acts WITHIN a column. Two such maps commute.

Supporting evidence from the source: `pcmm` checks only depth and count, never
`encoding_` or `in_ntt_domain_`, and it copies both from input to output
(`batchmatrix.cu:1449-1451`).

If it holds, **only Algorithm 4 genuinely needs the coefficient encoding** --
because a ciphertext-ciphertext matrix product is what the `R_k` structure is
FOR -- and the non-linear half of a block never leaves slot form.
`CKKS_Llama3Batch16_ProjectionCommutesWithTheBridge` is the experiment; the
result is recorded in 22.7.

### 22.7 Measured

Sicily **GPU 2** (GPU 0 was at 100% from another user, GPU 1 idle but holding
27 GB), tree `HEonGPU-b16nl`, commit `a697686`, CUDA 12.0, sm_86.
**17 of 17 green, 8.7 s**, at the real batch-16 ring `N = 4096, d = 128,
k = 32, batch = 16` — the first time anything in this repo has executed at
that shape.

**1. `project()` COMMUTES with the bridge. CONFIRMED.**
`ProjectionCommutesWithTheBridge` checks the slot-form projection against a
host reference *and* against the matrix-form path, so it cannot pass by being
wrong in the same way. This is the load-bearing result of the section: the
crossings that are 98.3% of a block's rotations exist because the projection
was assumed to need the coefficient encoding, and it does not. **Only
Algorithm 4 genuinely needs it.**

**2. The instances stay separate.** `RMSNormKeepsInstancesIndependent` and
`FeedForwardKeepsInstancesIndependent` perturb ONE of the sixteen and require
the other fifteen to move by less than 1e-4 while the perturbed one moves by
more than 1e-2. Not exact equality: two runs are two encryptions, so the floor
is CKKS noise, and mixing would be an O(1) move — the bound sits three orders
above the noise and three below the signal. Nothing here is asserted by a
round trip, which a pair of mutually inverse mistakes would also pass.

**3. The reshape is free, and the test proves it by POINTER IDENTITY.**
`ReshapeIsIndexArithmeticAndAliasesTheInput` asserts
`view[lane] == &ct.column[channel_of(h, lane)]` — the difference between free
and `d` ciphertext deep copies per head. GQA maps 4 query heads onto each kv
head as a view, and `kv_channels` stays at 1024 against `q_channels` 4096.

**4. Every level claim, asserted as an exact depth rather than argued.**

| claim | assertion | result |
|---|---|---|
| gain folds into the weight, exactly | value agreement + `depth_a == depth_b + 1` | holds |
| the SiLU domain map folds onto the gate weight | value agreement + `depth_a == depth_b - 1` | holds |
| a pre-scaled sum skips the fit's affine multiply | value agreement + `depth_b == depth_a - 1` | holds |
| `hidden_block` changes only the association of a sum | value agreement + equal depth | holds |
| RoPE costs one level | `after == before + 1` | holds |

**5. Two claims I made were REFUTED by the tests and are corrected above.**

- **`slot_resident` does not save a level.** I wrote that two crossings
  replacing three would drop one; measured, both arrangements land at depth 9.
  Three crossing CALLS are only two crossing LEVELS, because the gate and the
  up projection cross concurrently. The saving is bridged columns —
  43,008 -> 8,192 — and nothing else. The test now pins the equality.
- **The levels are in FULL slot residency instead**, where the norm's return
  crossing and the SwiGLU's entry crossing are adjacent and cancel outright:
  `SlotResidentSublayersNeedNoCrossing` asserts exactly `-2` levels for a
  norm/SwiGLU pair, and every non-attention bridged column disappears with
  them.

**6. The one test failure worth reporting, because it is not a bug.**
`RMSNormMatchesThePlaintextLayer` first failed at 3.2e-3 against a tolerance I
had guessed at 1e-3. That number is the CHEBYSHEV FIT and nothing else: 1/sqrt
has a branch point at zero, so its interpolation error is set by how close the
fitted interval comes to it, and a fixture with eight channels of uniform
noise is the worst case — the summed square spans about 14x. At the real
`d_model = 4096` the summed square concentrates as `1/sqrt(4096)` and the span
collapses towards 1.1x. The test now bounds at 5e-3 and *proves the
attribution* by re-running the identical circuit at degree 31 and requiring
the error to collapse by 4x; if it did not, the error would be coming from the
encoding rather than the series.

**7. Relinearisation counts, corrected from a source simulation** of
`gen_power`, `optimal_split` and `evaluate_poly_recurse` including the re-split
branch that does fire: degree 7 -> **5**, degree 15 -> **8**, degree 31 ->
**14**. Earlier hand estimates in this document said 4 / 7 / 11, a 27%
undercount at degree 31.

**What is NOT measured.** No timing, no `nsys` capture, and nothing at the 8B
width — every key-switch and byte figure in 22.3 and 22.8 is arithmetic over
the source. The tests run at `d_model` 4-8 and `hidden` 8-16; what they
establish is the LAYOUT, the LEVELS and the SEPARATION, which are properties
of the circuit and not of the width. The cost model is unvalidated on this
path.

### 22.8 The three things that could sink this shape, none of them layout

1. **Security.** `N = 4096` with the ~70-limb chain one refresh-free block
   needs is roughly 3,240 bits of modulus, astronomically outside any
   security level, and batch 16 at 128 tokens *forces* `N = 4096`. This shape
   cannot be made secure without dropping the batch or the token count. The
   profile runs `sec_level_type::none` and this is the same accepted stance
   as every other section here -- but it is a larger overage than the
   rectangular path's 22.8x.
2. **Memory.** `Llama3BatchOperator::rms_norm` holds the slot copy alive
   across the return crossing, so its own frame peaks at 20,480 resident
   ciphertexts -- **87.5 GiB at 70 limbs, over an 80 GiB A100** before any key
   material. Releasing it (which `feed_forward` already does for its own
   branches) is 17.5 GiB.
3. **There is no refresh on this path at all**, and it cannot be switched on:
   `Llama3BatchOperator` never passes a boot key, so the two aux-refresh hooks
   are unreachable. One block spends ~67 levels of a 70-limb chain, of which
   the bridge is only 9.

### 22.9 RoPE, which was missing entirely

Rotary embedding is **absent from the Algorithm-1 path** -- not configured
off, absent: `llama3_batch.cu` never mentions it and `BatchAttentionConfig`
has no field for it. It is the one non-linearity of this session's scope with
no implementation anywhere, so `rope_slots()` is it.

It is cheap here for the two reasons everything else is. The head-dim pairing
`c <-> c + head_dim/2` is a pairing of whole CIPHERTEXTS, because a channel is
a ciphertext -- no homomorphic rotation, no Galois key. And the angle
`(u + offset) * theta^(-2c/head_dim)` depends on the TOKEN, which is the slow
slot axis, so one plaintext per lane pair serves every instance and every
head. Four plaintext products, two additions, **one level**, measured.

The minus sign goes on the plaintext, not on the ciphertext: negating
homomorphically is `multiply_constant`, a plaintext product and a rescale, and
the two halves of a pair would then be at different depths and could not be
added at all.

**Known cost, named rather than hidden:** the three slot vectors a lane needs
are built once per lane but applied through `multiply_vector`, which encodes
on every call -- so the same plaintext is re-encoded once per head, 8,192
encodes for Q where 192 would do. It is the same defect the SiLU's domain map
has when its fold is off. The fix needs an `HEEncoder` on the constructor; the
arithmetic is unaffected either way.

### 22.10 Coverage, stated plainly

Before this session **nothing in the repo had ever run at the batch-16
shape**: the GPU suite is `N = 4096, d = 8` (i.e. `k = 512, batch = 256`) and
`profile_llama3_batch.cpp` defaults to `HEONGPU_ONEMM_LOGN = 11`, which is
batch **8**. Multi-head and GQA are untested everywhere -- every existing test
sets `heads = 1`, so the head-indexing arithmetic has only ever run at
`h = 0`.

---

## 23. Four branches into one, and the block end to end (2026-08-13)

Sections 19 to 22 are four sessions on the same encoding, each of which
validated its own seam against a host reference and passed. **Nothing had ever
run the seams against each other.** This section is that, and it found two
faults that no unit test in the suite could have caught.

`HEonGPU_LLama3_8B_batch16` is now the single branch; `_ccmm`, `_softmax` and
`_nonlinear` are merged and deleted, each verified an ancestor of the union
first.

### 23.1 The two faults the merge found

**A projection taken on a product was rejected outright.** `pcmm` grew the
guard `relinearization_required_ || cipher_size_ != 2`. The second half is
wrong: `relinearize_inplace` clears the flag and **never writes
`cipher_size_` back to 2**, so every ciphertext that has been through
multiply + relinearize reports three for the rest of its life. The library
already treats the flag as the authority — `operator.cuh` derives the size as
`relinearization_required_ ? 3 : 2` rather than reading the field. Nobody had
hit it because every existing call feeds `project()` from a BRIDGE output,
which is freshly allocated at size two; a slot-resident SwiGLU feeds it a
product. The guard now tests the flag, and both `pcmm` variants stamp their
own output rather than copying the stale size out of `in[0]`.

**RoPE was wired into nothing.** It existed as an entry point that no code
path called, and `BatchAttentionConfig` had no field for it — so every
measurement this path has ever produced was taken with **no positional
information at all**. An attention sublayer without RoPE is not Llama-3's.

### 23.2 What is still not implemented, stated plainly

| gap | status |
|---|---|
| bootstrapping | **absent and unreachable** — no boot key is ever passed, so the two aux-refresh hooks cannot be turned on. One block spends ~56 of a 62-limb chain, so a STACK does not run at all. |
| sequences longer than `d = 128` tokens | **no code path.** There is no token blocking here; the slot path in `llama3.cu` has it and this one does not. |
| the real 8B width | does not fit on a 48 GiB A6000, by the arithmetic in 22.8. The driver is stage-selectable for exactly this reason. |

### 23.3 The block, measured stage by stage

Sicily GPU 2, `N = 4096, d = 128, k = 32, batch 16`, `d_model 128`,
`hidden 256`, 2 heads over 1 KV, 62 limbs. Every stage is compared against a
host reference computed from **that stage's own input**, so a per-stage error
says WHERE and the end-to-end error says WHETHER.

| stage | ms | levels | relative error |
|---|---:|---:|---:|
| norm1 | 6,830 | 0 → 9 | 5.4e-06 |
| attention (QKV, scores, seam, PV, W_o) | 8,773 | 9 → 36 | 2.5e-03 |
| residual | 47 | 36 → 37 | 2.5e-03 |
| norm2 | 1,544 | 37 → 46 | 6.4e-02 |
| SwiGLU | 3,465 | 46 → 55 | 1.4e-01 |
| residual | 30 | 55 → 56 | 1.4e-01 |

**The dataflow is correct**: every stage runs, the levels land exactly where
the ledger predicts (attention is 27 = 1 + 1 + 23 + 1 + 1), and no
orientation, scale or level disagreement appears at any seam. What degrades
is ACCURACY, down a 56-level chain of degree-15 fits — and the attribution is
measured, not asserted. Raising ONLY the degrees, same circuit, same shape:

| stage | deg 15 | deg 31 |
|---|---:|---:|
| norm1 | 5.4e-06 | 1.0e-05 |
| attention (degrees unchanged) | 2.451e-03 | 2.445e-03 |
| norm2 | 6.414e-02 | **8.467e-03** |
| SwiGLU | 1.432e-01 | **1.864e-02** |
| verdict | SUSPECT | **DATAFLOW OK** |

The two stages whose degree moved improve 7.6x and 7.7x; attention, whose
degrees were left alone, does not move at all. That is the signature of fit
error and not of a dataflow fault, and it costs three more levels (56 -> 59).

### 23.4 The optimisation, measured

`slot_resident` on the SwiGLU — the arrangement 22.6's commutation result
unlocks, and which the `pcmm` guard above was silently blocking:

Measured twice, in the whole block and on the sublayer alone:

| | SwiGLU ms | levels | relative error |
|---|---:|---:|---:|
| in-block, three crossings, deg 15 | 3,465 | 46 → 55 | 1.432e-01 |
| in-block, two crossings, deg 15 | **1,601** | 46 → 55 | 1.432e-01 |
| sublayer alone, three crossings, deg 31 | 19,467 | 9 → 19 | 2.264e-04 |
| sublayer alone, two crossings, deg 31 | **9,120** | 9 → 19 | 2.264e-04 |

**2.09x in the block and 2.13x on the sublayer, at identical depth and an
error identical to four significant figures.** The arithmetic is the same
arithmetic; what falls is the number of columns crossed.
At this shape the bridged columns fall 3 x 256 = 768 to 2 x 128 = 256; at the
8B shape the same move is 43,008 to 8,192.

### 23.5 RoPE, priced

| | attention ms | levels | relative error |
|---|---:|---:|---:|
| off | 8,773 | 9 → 36 | 2.451e-03 |
| on | 23,862 | 9 → 39 | 2.405e-03 |

**Exactly +3 levels**, as 22.9 says it must be: two crossings and the
rotation. The error is unchanged, which is what validates the wiring — the
host reference ropes the same Q and K, and so does the score calibration,
which has to see what the circuit will see. The 2.7x in time is the two
crossings over `q_channels + kv_channels`, and it is the price of reaching
slot form from the coefficient encoding. Under a slot-resident stream Q and K
are already in slot form when they are formed and RoPE costs its one level
and nothing else.

### 23.6 The whole block through one call, and the chain it needs

`Llama3Batch16Operator::transformer_block` against the same host reference,
degree 31, 62 limbs:

| | ms | levels | relative error |
|---|---:|---:|---:|
| staged, stage by stage | 20,734 | 59 | 1.887e-02 |
| `transformer_block` | 20,435 | 59 | 1.887e-02 |
| `transformer_block`, slot-resident SwiGLU | **19,106** | 59 | 1.887e-02 |

The wrapper reproduces the staged composition to every digit, which is what
it had to do. Slot residency is **-6.5% on the whole block** — smaller than
the 2.13x on the sublayer, because the SwiGLU is only part of a block.

**The chain is the binding constraint on turning anything on.** A block at
degree 31 spends 59 of 61, and RoPE costs three more, so degree 31 + RoPE
wants a longer chain than 62 limbs and does not run. It does not fail
politely: `evaluate_poly` derives a kernel grid extent from the levels
remaining, so an exhausted chain surfaces as **"CUDA Error ... invalid
configuration argument"** from inside GPU-NTT rather than as anything
mentioning levels. `llama3.cu` warns about exactly this in
`evaluate_chebyshev` and its own guard catches the common case; this path
reaches a different kernel first. Anyone enabling a feature here should raise
`HEONGPU_B16_LIMBS` first, and read that message as "out of chain".

Confirmed by giving it the chain, which is the whole diagnosis:

| | limbs | levels spent | result |
|---|---:|---:|---|
| block_op + RoPE | 62 | — | CUDA "invalid configuration argument" |
| block_op + RoPE | 68 | **62** | **DATAFLOW OK**, 2.039e-02 |

62 = 59 + 3, which is exactly what 23.5 prices RoPE at. **The whole block
runs end to end with rotary embedding**, at 43.0 s for sixteen inputs at this
width.

### 23.7 Two things the driver had to learn


**Calibrate the model, not just the fit.** Random weights at this width put
raw attention scores across a span of hundreds, so the SoftMax was asked to
fit `exp` over `[-300, 0]` — a dynamic range of `e^75`. It did not fail
loudly; it returned values that outgrew int64 at decrypt, which surfaces as
"extracted coefficient does not fit in int64" from the Garner reconstruction
and looks like a library fault. A trained model does not present that, and
that boundedness is the premise Section 4.3's calibration rests on.

**An absolute error means nothing across shapes.** The SwiGLU looked broken
at 6.9e-01 absolute and is ordinary at 1.8e-02 relative: a fit's error grows
with the range it is fitted over, and a wider model gives a wider range. A
fixed absolute threshold reads a correct circuit as broken the moment the
width moves.

## 24. Batch 16 with security on: the batch size IS the ring, and a ring crossing costs 1/22 of an encoding crossing (2026-08-13)

Worked out on `HEonGPU_LLama3_8B_batch16_ringswitch`, forked from
`HEonGPU_LLama3_8B_batch16` @ `c9482fa`. Seven new tests, all green on Sicily
GPU 2. Figures are marked **measured** (this branch, this GPU), **counted**
(exact, read off the source) or **derived** (arithmetic, unvalidated).

§19-§23 priced this path with `sec_level_type::none`. This prices it with the
cap on.

### 24.1 The batch axis is not a cost parameter. It is the security parameter.

`d` is pinned to `head_dim = 128` by Algorithm 4 (`llama3_batch.cu:1211-1224`
needs `q_channels % (heads*d) == 0`, so `d | 128`), and `k = N/d`,
`batch = k/2`. Those three together say

    N = 2 * batch * d = 256 * batch,    and d = N/k = 128 at every batch.

So **choosing the batch size chooses the ring**, and the ring chooses the
128-bit cap. There is no other knob: `MIN_POLY_DEGREE 4096` and
`MAX_POLY_DEGREE 65536` (`kernel/defines.h:14-15`) make the batch axis a
five-valued enum, and every value is a different security level.

| batch | k | N | logN | `heongpu_128bit_std_parms(N)` | Q primes at `q0=41`, 33-bit steps, one 33-bit special | usable levels |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 4096 | 12 | 109 | **2** | **1** |
| 32 | 64 | 8192 | 13 | 218 | 5 | 4 |
| 64 | 128 | 16384 | 14 | 438 | 12 | 11 |
| 128 | 256 | 32768 | 15 | 881 | 25 | 24 |
| 256 | 512 | 65536 | 16 | 1761 | 52 | 51 |

(counted; `log QP = 33L + 41` against the cap. 33-bit primes and `q0 = 41` are
forced from below by bootstrap v2 — §15.4 — and `MIN_USER_DEFINED_MOD_BIT_COUNT
= 30` blocks anything smaller on the generated path.)

**Batch 16 is the smallest batch this encoding admits at all**, and it is the
only one with no room. Everything in §19-§23 — including "batch 16 is cheaper
per input than the rect path" — was measured at `log QP ~ 2,500` against a cap
of 109, i.e. **~23x outside 128-bit**.

**Per-input cost is flat in the batch size, so the security is free.** A
ciphertext is `16·L·N` bytes and holds `N/2` values, so it is **`32·L` bytes
per value at every ring**; likewise every elementary operation on the residual
stream is `O(values · L)` however those values are packed. Raising the batch
therefore costs **working set**, not throughput:

| batch | residual stream at 20 limbs | ciphertexts at the island | at a logN 16 big ring |
|---:|---:|---:|---:|
| 16 | 5.4 GB | 4096 | 256 |
| 32 | 10.7 GB | 4096 | 512 |
| 64 | 21.5 GB | 4096 | 1024 |
| 128 | 43.0 GB | 4096 | 2048 |
| 256 | 85.9 GB | 4096 | 4096 |

So "is B = 16 cheaper than B = 1" is the wrong question. **B is free; take the
largest B whose working set fits** — batch 64 on a 48 GiB A6000, batch 128 on
an 80 GiB A100 — and batch 16 is the one setting with a reason not to pick it.

### 24.2 The descent theorem, and why the crossing must be taken in matrix form

**A matrix encryption at layout `(N, d, k)` descends under `switch_down(m)` to
`m` matrix encryptions at `(N/m, d/m, k)` — the same `k`, hence the same batch
— and small ciphertext `r` holds the rows `i = r (mod m)` at small row `i/m`.**

Definition 2 puts entry `(i,j)` coordinate `t` at coefficient `i + d*t`, and
`switch_down` sends coefficient `p` to position `p/m` of ciphertext `p mod m`.
With `m | d` and `i = m*i' + r`,

    p = i + d*t = m*(i' + (d/m)*t) + r

so `p mod m = i mod m` and `p/m = i' + (d/m)*t`, which is Definition 2 again at
`(N/m, d/m, k)`. **The subring index `t` is never touched, so the batch axis
crosses the ring intact.**

**Measured** (`test/test_ckks_batch_ringswitch.cpp`): worst error **2.69e-07**
at the real island shape (`N` 8192 -> 4096, `d` 256 -> 128, `k` 32, batch 16),
and **6.45e-07** at `m = 4`. Asserted against the ENCODER, not against a second
crossing — two ring-switch paths sharing a convention error would cancel — and
separately that the split is **interleaved, not contiguous**, which symmetric
random data cannot distinguish.

**A slot-form ciphertext does not descend.** `compose_up` of `m` slot-form
ciphertexts gives big slot `s` equal to `sum_j zeta^{j*5^s} *` (small slot `s`
of ciphertext `j`) — an `m`-point twiddled mixture. That mixture alone would be
cheap to invert, `m` diagonals; but it couples the big slots
`{s + (n_small/2)*c}` while the layout wants `{b + (k/2)(m*u' + r)}`, and those
two index sets are a **digit transposition** of one another — a stride
permutation, hence a dense linear map, not `2*sqrt(m)` rotations. **Crossing
rings in slot form is dead.** The crossing is a matrix-form operation, and that
is a placement constraint rather than a cost.

### 24.3 What is actually pinned to the island

Only **Algorithm 4**. Its operands are square at `d`, and `d = N/k = 128` only
at the island ring; at a ring `m` times bigger with the same `k`, `d` is `128m`
and `attention()` throws.

Everything else may sit at either ring:

* **Algorithm 1 is not pinned.** It contracts over CHANNELS, which are the
  ciphertext axis, while the crossing decimates ROWS, which are the token axis
  — disjoint axes, so they commute. **Measured: 6.57e-10** between "project at
  the big ring then descend" and "descend then project at the island", and
  3.44e-10 against a host reference. Checked both ways round, because a host
  reference alone cannot tell "both right" from "both wrong alike".
* **The bridge is not pinned.** `to_slots`/`from_slots` is built for any
  `(N, d, k)`; at the big ring it is `d_big = 128m` diagonals instead of 128.
* **The non-linear layers are not pinned.** They are slot-wise (§22).

### 24.4 A ring crossing costs 1/22 of an encoding crossing

Take one full crossing of the residual stream, `V` values wide.

| crossing | ciphertexts | key switches each | ring | levels | total KS-work |
|---|---:|---:|---|---:|---|
| `to_slots`/`from_slots` at the island | `2V/N` | **22** (BSGS `n1+n2-2` at `d=128`) | `N` | 1 | `22 * 2V * L` |
| `switch_down`/`compose_up` | `2V/(mN)` | **1** | `mN` | **0** | `2V * L` |

The `m` cancels: a ring crossing moves `m` island ciphertexts with one key
switch at a ring `m` times bigger. **The ring crossing is 22x cheaper than the
encoding crossing, at every `m` this family admits**, plus the coefficient move
(~20% of a switch, §13 measured), so call it 18x. It also costs **zero levels**
and needs **no new Galois key** — two Switchkeys of the big context, for the
whole model.

**Measured, this branch:** the crossing costs zero levels in both directions,
and a rescale taken inside the island lands the big ring exactly one level down
(`CrossingIsLevelFreeAndTheIslandSpendsShared`, 3.30e-04). **Measured:** a whole
Algorithm 4 run at the island composes back up into the big-ring matrix
product, worst error **1.88e-03** at `d = 64`, operand scales 2^35/2^25.

So: **yes, ring switching is cheap here — cheaper than on the rect path.** The
batch encoding's Galois set is `d - 1 = 127` shifts however wide the model is
(`llama3_batch.cu:147-161`) and Algorithm 1 needs none, so unlike §7bis the
island is **not** a key-memory refuge. It is a pure level-budget device, and
the crossing that gets you there is the cheapest boundary in the flow.

### 24.5 The placement, and the level ledger that decides it

The split point is **not** "non-linear up, products down". It is "**whatever
does not fit in the island's level budget goes up**" — and the bootstrap is on
that list at every batch in this family, because the island chain never reaches
the ~39 limbs a refresh needs, not even at batch 128.

    big ring, matrix form, full level
      -> bridge to slots (1 big level) -> RMSNorm / SoftMax / SwiGLU -> bridge back
      -> drop to the shared prefix -> switch_down
      -> ISLAND: Algorithm 4                    (1 level)
      -> compose_up -> refresh at the big ring

**Both bridges sit at the big ring on purpose.** A bridge inside the island
spends an island level, and island levels are the scarce resource; at the big
ring it costs `sqrt(m)` more work (`2*sqrt(128m)` rotations on `C/m`
ciphertexts of ring `mN`, against 22 on `C` of ring `N`) but spends a level the
big ring has to spare.

Which makes the island's EXIT the whole question, and it has an answer now:

> **A regular bootstrap carries a batch matrix encryption.** Measured at the
> real island shape (`N = 4096, d = 128, k = 32`, batch 16) on the library's
> own bootstrapping chain: **worst error 3.23e-05, 14.9 bits**, six levels
> restored, scale preserved at 2^50. `regular_bootstrapping` reads no encoding
> tag and its net effect on the plaintext polynomial is the identity with the
> modulus restored; what was NOT obvious is EvalMod, whose bound is on the
> plaintext COEFFICIENTS while this encoding puts an inverse length-`k` DFT
> there rather than the values. It holds. This closes the open question §20
> recorded as "whether bootstrapping carries a matrix encryption at all".

Because the refresh needs no bridge in front of it, the island may exit at one
limb, and the minimum island chain is **two Q primes and a special**:
`41 + 33 + 33 = 107 <= 109`. **Batch 16 has a legal 128-bit parameter set, with
two bits to spare.** Had the refresh wanted slot form, the island would have
needed three Q primes and a special — 140 bits against 109 — and batch 16 would
have had no parameter set at all. One measurement decided that.

**What batch 16 pays instead is forced refreshes.** One island level is exactly
one Algorithm 4 call, so every island visit ends at one limb and must be
refreshed. An attention sublayer has two Algorithm 4 stages (`QK^T` and `PV`)
with the SoftMax between them, so batch 16 buys **two forced bootstrap rounds
per attention sublayer**. Batch 32 (4 island levels) carries a product and a
bridge in one visit and exits with limbs in hand.

The refresh itself is batch-independent per input, and that is worth stating
because it is the other half of "the batch is free": one refresh is `2V/(mN)`
bootstraps at the big ring, which at `d_model = 4096`, 128 tokens and a logN 16
big ring is **16 bootstraps per input at every batch size** (derived, from
`32*L` bytes per value). What the batch buys is not fewer refreshes per input;
it is fewer FORCED ones, by giving the island levels to work with.

### 24.6 What the big ring's extra rows should hold

`d_big = 128m` and the descent decimates that row axis, so the extra capacity
has to hold something or the big ring runs `m`-fold empty. Two consistent
choices, not equivalent:

* **Rows = tokens.** The big ring holds a `128m`-token sequence and the descent
  IS token blocking — which this path has never had (§20: "`d` IS the sequence
  length: 128 tokens, permanently"). The non-linear layers are untouched: the
  channel reduction is still a slot-wise add across ciphertexts. Attention
  across the `m` blocks becomes `m^2` Algorithm 4 products, which is what a
  longer sequence costs anywhere.
* **Rows = (token, channel), channel fastest.** `C/m` ciphertexts at a
  128-token sequence. RMSNorm's channel reduction stops being free — it becomes
  a `log2(m)`-rotation strided reduction inside each ciphertext.

Take the first if the sequence is long, the second if it is not. Either way the
descent is the same index identity; only the labelling of `d_big` changes.

### 24.7 Rejected

**Cross in slot form and un-mix at the big ring.** The mixture is `m`-diagonal,
which looks like `2*sqrt(m)` rotations; the index sets differ by a digit
transposition, so the map is a stride permutation and dense. §24.2.

**Use the island as a key-memory refuge, as the rect path does.** The batch
bridge needs `d - 1 = 127` shifts however wide the model is, and Algorithm 1
needs none. There is no key wall to escape here.

**Get a third Q prime into the batch-16 island.** `3*33 + 41 + 33 = 140 > 109`;
smaller primes are blocked by `MIN_USER_DEFINED_MOD_BIT_COUNT = 30` on the
generated path and by bootstrap v2's `q0 = 41` on the shared-prefix path, which
the small chain must copy verbatim.

**Batch 256 on one card.** The residual stream alone is 86 GB at 20 limbs.

### 24.8 Open

1. **Nothing here is timed.** The seven tests establish algebra, levels and
   precision; every millisecond in §24.4-§24.5 is arithmetic over one measured
   ring-switch datapoint (§13). One `nsys` capture of a crossing at the batch
   shape would settle it.
2. **The `m^2` attention blocking is not written.** Rows-as-tokens gives token
   blocking free at the encoding level, but the causal mask over interleaved
   token classes and the `m^2` product schedule are new code.
3. **Composite security is the MIN of the two rings** and the secrets are tied
   (`s_low(X) = s_high(X^k)`), so the island's 109-bit budget caps the system
   however big the big ring is. And the library's check covers the modulus
   budget only — every driver here uses a SPARSE secret, which
   `heongpu_128bit_std_parms` does not model. Say both together.
4. **14.9 bits is the refresh's ceiling on this chain, not this encoding's.**
   Whether the matrix encoding costs precision relative to a slot encoding at
   the same parameters is one A/B run away and was not made.
