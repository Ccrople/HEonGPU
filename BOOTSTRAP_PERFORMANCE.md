# Making the CKKS bootstrap faster

**This document belongs to both Llama-3 paths, which is why it is its own file
and why the branch is `HEonGPU_LLama3_8B_bootfast` rather than either branch's
name.** It is deliberately kept out of `LLAMA3_8B_LAYER_FLOW.md`: `nobatch` and
`batch16` have each appended sections to that file since they diverged — the
numbering is common to §18, and `batch16` carries §19-§25 that `nobatch` does
not — so anything added at its end conflicts on one of the two merges, and
resolving that conflict by taking a side silently drops a whole session's
write-up. Nothing here touches the flow doc, so this branch merges into
`nobatch` and `batch16` with no conflict at all. Verified by test merge into
both, not assumed.

Section references of the form §14.4, §17, §18, §12 point into
`LLAMA3_8B_LAYER_FLOW.md`, whose numbering is common to both branches up to
§18.

What lands on which path:

- **§6, the constant encoder**, is in `operator.{cu,cuh}`, which is
  byte-identical on `nobatch` and `batch16`. The same win on both, and it
  reaches every polynomial evaluation — the SiLU, the SoftMax exponential and
  reciprocal, the RMSNorm inverse square root — not only the bootstrap.
- **§4, the v2 routing**, is on `Llama3Operator`, the base that the rect, batch
  and batch-16 operators all sit on. Batch 16 has no reachable refresh of its
  own yet, so it inherits the v2 entry point the moment it gets one.

---

Task: the block is bootstrap-bound (53.7% of a two-ring block, §14.4), so make
a bootstrap cheaper — by parameter, by method, or by both.

Everything below is measured on **one idle A6000 (Sicily GPU 2)** with
`benchmark/profile_boot_precision.cpp`, which now takes the non-positional
knobs from the environment so a sweep needs no rebuild:
`HEONGPU_BP_{CTS_PIECE,STC_PIECE,CTS_RATIO,STC_RATIO,SPECIALS,HAMMING,
SPARSE_HW,REPS,EXTRA_L}`. One process per point (the bootstrapping context
locks after generation), 3 reps with the first discarded.

**Baseline: 255.0 ms, 16.76 bits, L = 29, 14 levels returned, 48 Galois keys.**

## 1. Every knob is already at its optimum. The surface is worth ~4%.

| lever | swept | best | vs baseline |
|---|---|---|---|
| DFT pieces CtoS x StoC | all 16 of 2..5 x 2..5 | **4/3, the shipped value** | **0%** |
| special primes (dnum) | 1..20 | 6-8 | 2.2% |
| BSGS ratio | 0.5..16 | 4 | 2.0%, +23% keys |
| model swap to `slim` | — | — | *slower* at equal levels |

`slim` runs EvalMod once and is still slower than v2 (214.7 ms returning 6,
against v2's ~159), because it pays the v1 Taylor exponential and the
un-hoisted `multiply_matrix` path. The pieces trade **key memory**, not time:
5/5 needs **28** Galois keys against 4/3's 48, for +7.7% time — useful wherever
the key ceiling binds, which is most of this project.

## 2. Cost is linear in the chain, and precision is flat along it

`boot_ms ~= 104 + 12.5 x (levels returned)`: 116.6 ms at 2 levels, 256.6 at 14,
341.1 at 20, all at 16.75 bits. `depth_after` is **constant at 15** — the
boot's own `stc_piece + n_sine + cts_piece` — so `nbase` alone sets what comes
back. The ~104 ms intercept is the boot's own depth, and it is what any
algorithmic change has to attack.

Corollary for §17: the five boots that restore 14 levels and let `cross.down`
discard 10 are paying **256.6 ms for what a right-sized chain delivers at
138.6** — 1.85x on those calls, ~10% of a block, at the cost of a second
bootstrapping context and a second 48-key set.

## 3. The EvalMod cliff is real, and it is `h -> K -> degree`

§14.5 recorded (sine_deg 30, dangle 3) as a cliff and closed the 43% bucket.
That verdict holds, and the reason is now measured rather than observed: **K
bounds |I(s)| after ModRaise, |I(s)| grows with the sparse secret's hamming
weight h, and the Chebyshev degree must resolve K oscillations.** So h sets K
sets the degree sets the boot's depth, and none of the three moves alone.

At the shipped h = 32, K = 8 with degree 10 looks like 186.7 ms — 1.37x — and
**fails on 1 of 16 independent secrets**. Degree 14 at K = 8 fails 2 of 12.
The single fast run is a lucky secret; the failure is catastrophic, not
graceful. Matching h to K works: **h = 16, K = 8, degree 10 gives 187.9 ms
against 259.0, 0/16 failures, at BETTER precision (18.09 vs 16.74)** — 1.38x,
bought by halving the sparse bootstrapping secret. That is a security
parameter, so it is recorded as an option and not taken.

**Method note, which cost this section two false starts:** the secret is
redrawn per process and the message is not (fixed seed), so a bootstrap
parameter must be tested across many SECRETS, not many reps. Three reps of one
secret will happily bless a config that fails 1 in 16.

## 4. The module was on v1, and v2 is 2.6-2.8x faster AND more secure

`Llama3Operator::bootstrap` (`llama3.cu:3199`) called `regular_bootstrapping`,
the **v1 Taylor** model, and it is the only boot call in the entire Llama-3
module tree (`llama3.cu`, `llama3_rect.cu`, `llama3_batch.cu`;
`llama3_batch16.cu` has none). `regular_bootstrapping_v2` was reachable only
from benchmarks that go *around* the operator API — `profile_tworing_block.cpp:812`,
`profile_ckks.cpp`, `profile_tworing_stage16.cpp`. So every figure in §14 for
the **rect** path was taken on v1, and the two-ring numbers on v2.

The comparison only means anything at equal **returned** levels, because v1
spends 25 levels of chain depth and v2 spends 15 (`depth_after` is constant per
model, so returned = L - 25 and L - 15):

| returned levels | v1 | v2 | v2 is |
|---:|---:|---:|---:|
| 2 | 328.3 ms | 117.5 ms | **2.79x** |
| 6 | 429.4 ms | 159.6 ms | **2.69x** |
| 10 | 550.5 ms | 207.1 ms | **2.66x** |
| 14 | 682.6 ms | 258.8 ms | **2.64x** |

and v2 gets there on a chain **ten primes shorter**, which every other
operation in the circuit is charged for as well.

**Security runs the same way, which is what settles it.** v1's ModRaise needs
the SCHEME's own secret to be sparse — the shipped examples and this harness's
v1 branch use `Secretkey(context, 16)`. v2 holds a dense h = 192 secret and
switches into a sparse h = 32 one for the ModRaise alone, through two
`Switchkey`s. So v2 is ahead on speed and on security; the only thing that
moves the wrong way is precision, 20.1 -> 16.75 bits, against a block that
already lands near 6 (§12).

Shipped as **`Llama3Operator::use_v2_bootstrapping(swk_d2s, swk_s2d)`** —
opt-in, default still v1, and it refuses a half-configured pair, because half a
pair would ModRaise under the sparse secret and never switch back, which
decrypts as noise rather than throwing. Test:
`test/test_ckks_llama3_bootv2.cpp`, 4/4, 18.13 bits worst-slot at logN 13
through the module API.

All green on a clean card (GPU 2, 4.3 GB used at launch): `ckks_llama3_rect`
38/38 (409 s), `ckks_llama3` 53/53, `ckks_llama3_bootv2` 4/4,
`ckks_llama3_batch` 12/12, `ckks_ringswitch` 9/9, `ckks_tworing_bridge` 2/2.

**A trap that cost an hour here and will cost the next person the same.** While
another session held ~44 GB of that card's 49, those same two suites failed
*every* test with a verbatim RMM `maximum pool size exceeded` at a 3.72 GiB cap
-- 0.9 x the 4.7 GB that was left -- several of them inside gtest `SetUp()` in
4-5 ms. The pool is sized from FREE memory at context creation, so a busy card
does not make a suite slow, it makes it fail, and a 4 ms failure reads exactly
like a real regression. A suite can also come back **rc=143 (SIGTERM) with no
output at all**, which is not a failure either. Read the verbatim error and
re-run on a clean card before believing any of it.

## 5. What is left, priced

1. ~~**The constant-plaintext encode in `evaluate_poly`, ~4-6%.**~~ **Done —
   6.2% at L = 29 and 10.8% at L = 17.** See §6.
2. **Batching the 16 ciphertexts of a refresh call.** `TwoRing::refresh` is a
   literal serial loop on the default stream. Boot phases are 90.5% GPU-busy,
   so stream overlap alone is bounded at ~5% of a block; real sharing of the
   key streaming (~28% of a boot's DRAM traffic) is the larger prize and the
   larger job.
3. **CUDA graphs.** `cudaGraph*` appears zero times in the tree, against
   ~1500 kernel launches per boot.

**Refuted, and worth recording so it is not retried:** a "solo" v2 that runs
EvalMod once, on the theory that the upper coefficient half is an encryption of
zero. It is not. After ModRaise the plaintext is `m + q0*I(s)` and **`I(s)` is
dense over all N coefficients whatever the message is**, so the imaginary
CoeffToSlot output is never zero and dropping it returns noise.

## 6. The constant encoder: a 2^15 FFT to write down a constant (2026-08-14)

`evaluate_poly` encodes every polynomial coefficient from scratch on every
call, ~48 times per bootstrap across the two EvalMod chains, and
`quick_ckks_encoder_constant_complex` (`operator.cu:2733`) did it the long way
each time: build a 32768-entry host `std::vector` by `push_back`, a **blocking**
512 KB `cudaMemcpy`, a 2^15 inverse FFT, the conversion kernel, then `Q_size`
forward NTTs of 2^16 — all on `stream = 0`, ignoring the caller's stream.

**None of that is needed for a real constant, and the reason is exact rather
than approximate.** The inverse FFT of a vector holding `c` in all
`slot_count_` slots puts `slot_count_ * c` at DC and zero elsewhere; times the
`fix = scale / slot_count_` the code already applies, that is `c * scale` in the
constant term and nothing else — the constant POLYNOMIAL. The NTT of a constant
polynomial is that same constant at every evaluation point. So the whole
transform chain evaluates to "write `round(c * scale) mod q_i` everywhere",
which is precisely what `quick_ckks_encoder_constant_double` (`:2778`) already
did for the real-valued API, with one kernel and no FFT.

The encoder now dispatches on `input.imag() == 0.0`. Chebyshev coefficients of
a real function are real, so the bootstrap and every Llama-3 non-linear fit take
the closed form; the two genuinely imaginary constants the encoding transforms
need (`-i/2` and `i`) still go the long way and are encoded once per context.
The call site also passes the operand's live limb count, so limbs above the
level are no longer written at all.

**Measured, five independent processes on an idle A6000 (GPU 2, 4.3 GB used):**

| chain | before | after | gain |
|---|---|---|---|
| L = 29, 14 levels returned | 256.8 ms | **240.9 ms** | **6.2%** |
| L = 17, 2 levels returned | 117.0 ms | **104.3 ms** | **10.8%** |

Run-to-run spread is 0.45 ms, so both are far outside the noise. The gain is
larger on the short chain because the boot's own depth dominates there; in the
linear law of §2 the **intercept falls from ~104 ms to ~92 ms** while the
12.5 ms per returned level is untouched, which is exactly the shape expected
from removing a fixed per-coefficient cost.

**It is not bit-identical, and that is the right outcome.** The two routes are
the same number by different arithmetic — one multiply against 2^15 butterflies
— and the closed form is the more accurate of the pair. `HEONGPU_CONSTENC_CHECK=1`
encodes every real constant BOTH ways and compares them limb by limb, bounding
the *signed* distance modulo `q_i` (a one-ULP disagreement about a negative
residue appears as a gap of `q_i - 1`, so a naive absolute difference would
false-positive on every negative coefficient). Clean at logN 16 deg 30, logN 13
deg 30, and logN 13 deg 62 / dangle 2. Measured precision is unchanged:
16.746-16.760 bits against 16.748-16.765 before.

Suites on the idle card: `ckks_llama3_rect` 38/38, `ckks_llama3` 53/53,
`ckks_llama3_bootv2` 4/4, plus batch, ringswitch, tworing_bridge, encoding,
multiplication, relinearization.

**Where this lands the whole section.** Against the v1 model the module was
using before §4, the Llama-3 refresh is now:

| levels returned | v1 (was) | v2 + closed-form encoder (now) | total |
|---:|---:|---:|---:|
| 2 | 328.3 ms | **104.3 ms** | **3.15x** |
| 14 | 682.6 ms | **240.9 ms** | **2.83x** |

and `evaluate_poly` is the general polynomial evaluator, so the same 6-11%
applies to every Llama-3 non-linear fit — the SiLU, the SoftMax exponential and
reciprocal, the RMSNorm inverse square root — not only to the bootstrap.
