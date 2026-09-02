# Optimising the A2 path for 32-bit ARMv7 (Cortex-A17)

The two previous campaigns in this repository ([SLIMMED-PATH.md](SLIMMED-PATH.md),
[FULL-PATH.md](FULL-PATH.md)) optimised NAM's A2 fast path on Apple Silicon. This
one moves the same question to a part that is nothing like an M2: the **Rockchip
RK3288**, four Cortex-A17 cores at 1.8 GHz, ARMv7-A, 32-bit only. That SoC is in
the HeadRush Core and Prime.

The interesting part is not that the numbers are smaller. It is that **the
ranking is different**. Ideas that won on AArch64 lose here, one idea that lost
there wins here, and the single best AArch64 switch — a very wide frame tile —
inverts completely. ARMv7 has half the vector registers, no by-element FMA, and
a NEON datapath narrower than its own register width, and every one of those
shows up in the measurements.

This is a record of **39 kernels** across two submodels, each measured by the
project's existing benchmark protocol on the real part, against a verbatim-port
control, including the ones written expecting them to lose.

**Constraint: fp32 only.** Every candidate has to stay numerically equivalent to
`a2_fast`, so a winner is a drop-in replacement needing no listening test.
Kernels that declare exactness are held to literal `max|diff| == 0` over a full
render, not to a dB floor. Three kernels deliberately break that rule and are
tagged as such; all three are discussed but none is promoted.

## The target, and one thing to know about the binaries

`-mcpu=cortex-a17` **implies `armv7ve`**, so the binaries this campaign builds
carry hardware integer divide (`sdiv`/`udiv`) and **will not run on a plain
ARMv7-A part**. That is correct for the RK3288 and wrong for, say, a Cortex-A8
or A9. It is recorded here because [`CMakeLists.txt:161`](CMakeLists.txt) points
at this file rather than repeating it: nobody should mistake these results for a
general armhf build.

The other two architecture flags in that block are correctness settings, not
tuning. `-mfpu=neon` alone gives `__ARM_NEON__` but not `__ARM_FEATURE_FMA`, and
Eigen keys `EIGEN_VECTORIZE_FMA` off that macro — so it would silently select
non-fused `vmlaq_f32` for every `pmadd` and change **the reference's own C=8
arithmetic**. `-mfpu=neon-vfpv4` is what makes the comparison meaningful.

## Why it is not a FLOP problem

The A2 shape is fixed: 23 layers, kernel sizes 6 (×21) and 15 (×2), plus a
16-tap head. That gives `179·C² + 40·C` MACs per frame — 1,731 at C=3 and 11,776
at C=8.

| | A2 nano (C=3) | A2 standard (C=8) |
|---|---:|---:|
| Real arithmetic at 48 kHz | 83 MMAC/s | 565 MMAC/s |
| `a2_fast` achieves | 719 MMAC/s | 773 MMAC/s |
| Cortex-A17 scalar VFP FMA peak @1416 MHz | 1.42 GMAC/s | — |
| Cortex-A17 NEON fp32 FMA peak @1416 MHz (64-bit datapath) | ~2.83 GMAC/s | ~2.83 GMAC/s |
| Ring footprint, `a2_fast`'s default sizing, 64-frame blocks | 172.5 KB | ~460 KB |
| Cortex-A17 L1D | 32 KB | 32 KB |

This table is the whole campaign in miniature, and it says something different
from the Apple-Silicon one:

* **At C=3 there is almost no headroom in scalar at all.** `a2_fast`'s 3-channel
  branch is a fully-unrolled *scalar* 3×3 GEMV, and it already runs at **51% of
  the part's scalar FMA issue rate**. No amount of scalar tidying can find more
  than 2×, and realistically far less. The only way up is SIMD — which means the
  planar transform, because vectorising across three channels wastes a quarter
  of every lane before it starts.

* **At C=8 the reference is already vectorised** (Eigen) and reaches 27% of NEON
  peak. There the question is not "can we use SIMD" but "can we keep the working
  set in 16 Q registers", and mostly the answer is no.

* **Both submodels overflow L1D by an order of magnitude.** 172.5 KB of ring
  against 32 KB of L1D is a far harsher ratio than the 172.5-against-128 that
  made the M2 results allocator-sensitive. Anything that reduces ring traffic
  should matter more here — and, as it turns out, it does.

The FLOP peaks above are the published microarchitectural figures for Cortex-A17
(64-bit NEON datapath, so a 128-bit `vfma.f32` issues over two cycles). They were
**not** measured on the part; they are here to size the problem, and no
conclusion below rests on them. Everything else in the document is measured.

## The result

ASUS Tinker Board R2.0 (RK3288, 4× Cortex-A17), Armbian 26.8.3, Release,
cross-built with GCC 13.3 for `armv7ve+neon-vfpv4`, 48 kHz, 64-frame blocks,
`Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam`, 523,808 frames per pass.
Full protocol: 5 s warm-up discarded, 30 s of timed passes, mean of the fastest
70% required to agree within 3%.

**Every run is clock-pinned to 1416 MHz** with the `performance` governor, and
the clock is verified by frequency-residency diff before and after. See
[Measuring on this board](#measuring-on-this-board) for why that number and not
1.8 GHz.

`core_percent` is the cost of one NAM instance as a percentage of one core —
the number that decides how many instances fit in a pedal.

### A2 nano (C=3), block 64

| | Kernel | Mean / pass | core % | vs `a2_fast` | Parity |
|---|---|---:|---:|---:|---|
| | `a2_fast` | 1261.95 ms | 11.56% | — | — |
| 0 | `n_baseline` — control | 1263.14 ms | 11.58% | 0.999× | **bit-identical** |
| | `n_widetile2` | 1421.91 ms | 13.03% | 0.888× | **bit-identical** |
| 2 | `n_planar` (tile 4) | 1130.39 ms | 10.36% | 1.116× | **bit-identical** |
| 3 | `n_widetile8` | 1015.77 ms | 9.31% | **1.242×** | **bit-identical** |
| | `n_widetile12` | 1161.86 ms | 10.65% | 1.086× | **bit-identical** |
| 4 | `n_widetile16` | 1253.84 ms | 11.49% | 1.006× | **bit-identical** |
| 5 | `n_widetile32` | 1322.58 ms | 12.12% | 0.954× | **bit-identical** |
| 6 | `n_ringdirect` | 1190.21 ms | 10.91% | 1.059× | **bit-identical** |
| 7 | `n_skiplast` | 1121.68 ms | 10.28% | 1.123× | **bit-identical** |
| 8 | `n_vext_taps` | 1237.22 ms | 11.34% | 1.019× | **bit-identical** |
| 9 | `n_planar_linear` | 1193.97 ms | 10.94% | 1.055× | **bit-identical** |
| 10 | `n_planar_lazy` | 1186.27 ms | 10.87% | 1.062× | **bit-identical** |
| 11 | `n_stacked8` | 954.24 ms | 8.74% | 1.322× | **bit-identical** |
| 12 | `n_stacked16` | 1191.95 ms | 10.92% | 1.057× | **bit-identical** |
| 13 | `n_vmla` | 2051.89 ms | 18.80% | **0.614×** | 133.7 dB |
| 14 | `n_framemajor` | 1598.11 ms | 14.65% | 0.789× | 133.3 dB |
| 15 | `n_pad4` | 1535.10 ms | 14.07% | 0.821× | **bit-identical** |
| 16 | `n_prefetch` | 1293.75 ms | 11.86% | 0.974× | **bit-identical** |
| | `n_stacked8_ring` | 964.82 ms | 8.84% | 1.304× | **bit-identical** |
| | `n_stacked8_skip` | 1001.59 ms | 9.18% | 1.256× | **bit-identical** |
| | `n_stacked8_lazy` | 954.34 ms | 8.75% | 1.322× | **bit-identical** |
| | **`n_stacked8_linear`** | **915.74 ms** | **8.39%** | **1.378×** | **bit-identical** |

(Index numbers are the kernel's position in the `a32_common.cpp` registry, which
is append-only because the indices appear in conformance labels. Blank means the
kernel was appended after a round's measurements asked for it.)

### A2 standard (C=8), block 64

| | Kernel | Mean / pass | core % | vs `a2_fast` | Parity |
|---|---|---:|---:|---:|---|
| | `a2_fast` | 7982.17 ms | 73.15% | — | — |
| 1 | `s_baseline` — control | 7962.38 ms | 72.97% | 1.002× | **bit-identical** |
| 20 | `s_planar2` | 7608.71 ms | 69.72% | 1.049× | **bit-identical** |
| 21 | `s_planar4` | 7087.15 ms | 64.94% | 1.126× | **bit-identical** |
| 22 | `s_planar8` | 6358.54 ms | 58.27% | **1.255×** | **bit-identical** |
| | `s_planar12` | 7034.50 ms | 64.46% | 1.135× | **bit-identical** |
| | `s_planar16` | 8878.12 ms | 81.36% | 0.899× | **bit-identical** |
| | `s_planar32` | 9709.56 ms | 88.98% | 0.822× | **bit-identical** |
| 23 | `s_ringdirect` | 7497.78 ms | 68.71% | 1.067× | **bit-identical** |
| 24 | `s_skiplast` | 7377.94 ms | 67.61% | 1.084× | **bit-identical** |
| 25 | `s_stacked2` | 7214.60 ms | 66.11% | 1.108× | **bit-identical** |
| 26 | `s_stacked4` | 6826.88 ms | 62.56% | 1.171× | **bit-identical** |
| 27 | `s_head_tile` | 7838.69 ms | 71.83% | 1.020× | **bit-identical** |
| | `s_stacked8` | 6148.09 ms | 56.34% | 1.298× | **bit-identical** |
| | `s_stacked16` | 8699.16 ms | 79.72% | 0.920× | **bit-identical** |
| | `s_stacked8_linear` | 6099.15 ms | 55.89% | 1.309× | **bit-identical** |
| | **`s_stacked8_lazy`** | **6085.81 ms** | **55.77%** | **1.312×** | **bit-identical** |
| 28 | `s_chanmajor` | 5511.87 ms | 50.51% | **1.451×** | 136.6 dB |

Rows come from the round in which each kernel was measured, and **every ratio is
against `a2_fast` measured in that same run**, never against the header row —
so the ratios are the trustworthy column and the millisecond columns are not
strictly comparable across rows. Across the rounds `a2_fast` sat between 1258
and 1262 ms at C=3 and between 7982 and 8015 ms at C=8 — a 0.3% and a 0.4%
spread respectively, which is the run-to-run figure described under the control
below.

**Both winners are bit-identical to `a2_fast` over all 523,808 frames**, and so
is every kernel in both planar families. The only three kernels that are not
bit-identical are `n_vmla`, `n_framemajor` and `s_chanmajor`; two of those are
slower than the code they replace, and the third is discussed at length below
because it is the fastest thing here and cannot be promoted as it stands.

### At the pedal's block size

A pedal runs small blocks. At 32 frames, which is the more realistic setting,
the gap widens rather than narrows:

| Submodel | `a2_fast` | Winner | | |
|---|---:|---:|---:|---|
| A2 standard | **78.5%** of one core | **57.8%** (`s_stacked8_lazy`) | 1.359× | bit-identical |
| A2 nano | 12.32% of one core | 8.87% (`n_stacked8_linear`) | 1.390× | bit-identical |
| A2 standard, if inexactness were acceptable | 78.5% | 51.4% (`s_chanmajor`) | 1.529× | 136.6 dB |

Taking A2 standard from 78.5% to 57.8% of a core, bit-identically, is the
headline. It is the difference between one instance per core with nothing left
over and one instance per core with room for the rest of a signal chain.

The promoted kernels do slightly better than the lab ones, measured through
`a2_planar` itself on the same board at the same pinned clock:

| Submodel | `a2_fast` | `a2_planar` | | |
|---|---:|---:|---:|---|
| A2 standard | 78.79% of one core | **55.63%** | 1.416× | bit-identical |
| A2 nano | 12.29% of one core | **8.36%** | 1.470× | bit-identical |

They are ahead of the lab winners because the promoted kernels carry the
switches *and* a transposed `layer1x1` weight block that the lab kernels do not.
Getting there was not a matter of copying the winner across; see
[What the port cost](#what-the-port-cost).

## What each candidate did

### The control

`n_baseline` and `s_baseline` are verbatim ports of `a2_fast`'s two branches —
the same rings, the same buffer sizing, the same inner loops — compiled into the
a32 lab from the *same* `vendor/upstream` tree through the *same* target
template as `upstream` itself. They exist to answer one question: does the lab
add overhead of its own?

It does not. Both are bit-identical, and across six back-to-back paired runs
they land between 0.993× and 0.999× of `a2_fast` — consistently a hair slower,
never faster, which is the right sign for a port that adds a virtual call and
nothing else. Inside the larger rounds, where more kernels share the window, the
controls span 0.986×–1.008×. **Run-to-run spread on this board is under
0.5%** once the clock is pinned — an order of magnitude tighter than the ±2% the
M2 campaigns had to live with, because the thermal and clock discipline is
enforced rather than hoped for. Differences of 1% here are real.

`BaselineRing` is kept as its own type in `a32_ring.h` rather than being
expressed as a mode of the shared `PlanarRing`, precisely so the controls keep
reproducing the number they were first measured with.

### The ones that were meant to win

**`n_planar` / `s_planar*`** are the core idea, carried over from AArch64:
`a2_fast` keeps channels interleaved and vectorises across *channels*; the
planar kernels give each channel its own plane and vectorise across *frames*, so
one NEON lane runs `a2_fast`'s per-frame scalar reduction verbatim. Nothing is
reassociated, which is what makes bit-identity a property of the design rather
than a lucky accident.

At C=3 the plain kernel is worth 1.116×, at C=8 the tile-8 kernel 1.255×. Both
are real, and both are a long way short of the 1.179× → 1.752× ladder the same
idea climbed on an M2. The reason is the tile sweep below.

**`n_widetile8`** is the best *single* switch at C=3 (1.242×) and the base of
the winner.

**`n_ringdirect` / `s_ringdirect`** write each layer's residual straight into
the next layer's ring instead of into a staging buffer and then copying. On an
M2 this was the biggest single win at C=3 (1.273×). Here it is worth 1.059× and
1.067×. The switch that mattered most there matters least here.

**`n_skiplast` / `s_skiplast`** skip the final `layer1x1`, whose output the head
never reads: 1.123× and 1.084×.

### The ones that were meant to lose

**`n_vmla`** is the planar kernel with `vfma` replaced by non-fused `vmla`. It
exists to price the arithmetic the codegen check forbids, and the price is
brutal: **0.614×**, which is 1.8× slower than the identical kernel with fused
FMAs. On this part `vmla.f32` is not merely less accurate than `vfma.f32`, it is
much slower. `Scripts/a32-codegen-check.sh` was written to protect the
bit-identity claim; it turns out to protect the speed just as much.

**`n_framemajor`** transposes the loop nest to frame-major with nine partials
across two frames. Bespoke, deliberately inexact (133.3 dB), and 0.789×. It lost
on the M2 too, harder (0.606×). Being wrong twice on two different
microarchitectures is a reasonably firm answer.

**`n_pad4`** pads C=3 out to four lanes so the channel-major layout can use full
vectors. Here it is written to pad the *output* side only, looping `j` over the
three real inputs, which makes it **bit-identical** — unlike the M2 version,
which was not. It is still 0.821×. Padding buys lane utilisation and pays for it
in ring footprint, and on a 32 KB L1D that is a bad trade.

**`n_prefetch`** adds `pld` hints ahead of the ring reads: 0.974×, i.e. slightly
negative. The A17's hardware prefetcher is already doing this, and the extra
instructions are not free.

**`s_head_tile`** keeps the control's Eigen layer body verbatim and only
transposes the head: 1.020×. The head is 16 taps out of a 23-layer stack, so
this is close to the arithmetic bound on how much the head alone can be worth,
and it confirms the layer bodies are where the time is.

### The one the plan got backwards

**Wide frame tiles.** On an M2 the tile ladder went 8 → 16 → 32 → 64 with the
peak at 32 (1.752×) and only tile 64 falling back. The plan for this campaign
assumed the same shape and reached for tile 16 and 32 first.

It is exactly inverted. At C=3 the ladder is:

| tile | 2 | 4 | 8 | 12 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| vs `a2_fast` | 0.888× | 1.116× | **1.242×** | 1.086× | 1.006× | 0.954× |
| spills+reloads per FMA in `layer_forward<6>` | — | 0.57 | **0.48** | 1.13 | 1.51 | 3.21 |

and at C=8:

| tile | 2 | 4 | 8 | 12 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| vs `a2_fast` | 1.049× | 1.126× | **1.255×** | 1.135× | 0.899× | 0.822× |
| spills+reloads per FMA | — | — | **1.96** | 3.18 | 4.52 | 6.09 |

The cause is not subtle once you count registers. ARMv7 has **16 Q registers**;
AArch64 has 32. The kernel's live set is

```
z accumulators   C·T/lanes
t accumulators   C·T/lanes
input            T/lanes
weight broadcast 1
```

At C=3, tile 8, 4 lanes that is 6 + 6 + 2 + 1 = **15 of 16**. Tile 12 needs 22.
The measured spill counts agree to the instruction: 0.48 memory ops per FMA at
tile 8, 1.13 at tile 12. **Tile 8 is the last rung that fits, and it is the
peak.**

The model predicts the *collapse* but not the *peak*. At C=8, tile 8 already
needs 35 registers and spends 1.96 memory ops per FMA — it does not fit by more
than a factor of two — and it still wins by a wide margin. Below the point where
the accumulators fit, what matters is how gracefully the spilling degrades, and
that is not something a register count predicts. The sweep had to be run.

This is the campaign's most transferable result: **tile widths tuned on AArch64
are not portable to ARMv7, and the direction of the error is "far too wide".**

### The one the data asked for

Two, in fact, and both are ring strategies rather than kernels.

The composition `n_stacked8` (tile 8 + ring-direct + skip-last) reached 1.322×,
and the two half-stacks say where that came from: `n_stacked8_ring` 1.304×,
`n_stacked8_skip` 1.256×, against `n_widetile8` alone at 1.218× in the same run.
Almost all of the composition's gain is the wide tile; ring-direct adds a little,
skip-last adds a little less.

That made the *ring* the obvious remaining variable, and swapping the
eager-mirror ring for a linear rewind took the C=3 winner from 1.322× to
**1.378×** — the single largest gain available after the tile width. The
eager-mirror ring costs a fixed memcpy per block that is negligible beside a C=8
layer's arithmetic and is not negligible beside a C=3 layer's, and on a 32 KB
L1D it is worse than negligible: it is eviction.

At C=8 the same swap is nearly a wash (1.298× → 1.309× linear, 1.312× lazy),
which is consistent with the same explanation.

### The fastest kernel here, and why it is not the winner

**`s_chanmajor`** is 1.451× at block 64 and 1.529× at block 32 — comfortably
ahead of everything else — and it is **not bit-identical** (136.6 dB below
signal). It is worth being precise about why, because the reason is a finding in
itself.

The two submodels' references do not compute the same *shape* of arithmetic:

* At **C=3**, `a2_fast`'s scalar branch runs a **bias-seeded single chain across
  taps**, with the compiler contracting each `a*b+c` into one FMA. One rounding
  per MAC.
* At **C=8**, Eigen computes a **per-tap partial from zero**, adds the bias
  separately, and rounds the mixin **twice** — once for the product, once for
  the add.

`s_chanmajor` does at C=8 what the reference does at C=3: one chain, one
rounding. It is *better* arithmetic than the thing it is being compared against,
and it is faster precisely because it is a shorter dependency structure with
fewer roundings. It fails the exactness bar for doing the more accurate thing.

That is the shape of the remaining 10%: **matching Eigen's reduction order
bit-for-bit at C=8 costs 1.451× → 1.312×.** Unlocking it is an upstream change
to the reference's arithmetic, not a kernel change, and it is out of scope for a
campaign whose promotion criterion is bit-identity.

## The sweeps

### Tile width

Covered above — it is the campaign's main finding, so it is reported with the
candidates rather than here.

### Ring strategy

Three strategies were implemented in `a32_ring.h`: `Pow2Eager` (what `a2_fast`
ships, mode 1), `Pow2Lazy`, and `LinearRewind` (mode 0).

| | `Pow2Eager` | `LinearRewind` | `Pow2Lazy` |
|---|---:|---:|---:|
| C=3, on `n_planar` | 1.105× | 1.055× | 1.062× |
| C=3, on `n_stacked8` | 1.322× | **1.378×** | 1.322× |
| C=8, on `s_stacked8` | 1.298× | 1.309× | **1.312×** |

The C=3 rows are the interesting ones, because they **disagree**: the linear ring
*loses* on the plain planar kernel and *wins* on the composed one. Ring strategy
is not separable from tile width — a wider tile changes how many times per block
the ring is touched, and therefore which strategy's overhead dominates. Anyone
carrying one of these numbers into another kernel should re-measure rather than
assume.

### Block size

| frames/block | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| A2 standard, `a2_fast` core % | 78.5% | 73.4% | 70.7% | 70.1% | 73.5% |
| `s_stacked8_lazy` | **1.359×** | 1.316× | 1.283× | 1.277× | 1.298× |
| `s_chanmajor` | 1.529× | 1.458× | 1.411× | 1.387× | 1.406× |
| A2 nano, `n_stacked8_linear` | **1.390×** | 1.374× | 1.372× | 1.379× | 1.404× |

**The winners gain more where it matters.** Per-block overhead is a fixed cost
that the kernels amortise better than the reference does, so the advantage is
largest at the smallest block — which is the block a pedal actually runs. Nothing
in this campaign has to be discounted for realistic block sizes; it has to be
*marked up*.

### Compiler

The whole set was rebuilt with clang 18.1.3 (`cmake/toolchain-armv7-linux-gnueabihf-clang.cmake`)
and re-measured. This sweep produced the campaign's two most important caveats.

**At C=8, parity fails under clang.** Twelve planar kernels that are
bit-identical under GCC land at 127.5 dB under clang. The cause is in the
*reference*, not the kernels: Eigen's `gebp_kernel` emits **310 `vmla.f32`**
under clang, so `a2_fast` itself computes different arithmetic — reference
checksum `-17.478711597881365` under GCC against `-17.478718637490147` under
clang. `__ARM_FEATURE_FMA`, `EIGEN_VECTORIZE_FMA` and
`EIGEN_HAS_SINGLE_INSTRUCTION_MADD` are all set correctly, and clang lowers
`vfmaq_f32` correctly in isolation; the mechanism inside Eigen's kernel is not
pinned down. The fact is verified and is enough for the conclusion:

> **Every C=8 bit-identity claim in this document is a GCC claim.**

C=3 is unaffected — it never goes through Eigen — and all C=3 kernels are
bit-identical under both compilers.

**At C=3, the speedups do not survive clang.** Timing, block 64:

| | GCC 13.3 | clang 18.1.3 |
|---|---:|---:|
| `a2_fast` (reference) | 1262 ms | 1319 ms |
| `n_planar` | 1.116× | **0.589×** |
| `n_widetile8` | 1.242× | 1.015× |
| `n_stacked8` | 1.322× | 1.067× |
| `n_stacked8_linear` | **1.378×** | **1.080×** |

The ordering survives, so the design ideas hold, but the margins nearly vanish
and the plain planar kernel becomes *worse than the code it replaces*. The cause
is an inlining decision, visible in the object file: clang leaves `conv<>` and
`frame_scalar` **out of line**, with 11 `bl` calls per `layer_forward`. Because
`conv<>` takes the vector accumulators by reference
(`__simd128_float32_t (&)[3][1]`), a call boundary forces them out of Q registers
and into memory — `frame_scalar` spends **164 loads and 39 stores to do 87
FMAs**. GCC inlines both and keeps the accumulators resident: 52 loads, 9 stores,
168 FMAs. The kernel's entire premise is that the accumulators stay in
registers, and clang's inliner breaks that premise.

This is fixable — force inlining on those two, or pass accumulators by value —
but it is new work rather than a tuning knob, and it is not done here.

At C=8, by contrast, clang changes very little: `s_stacked8_lazy` 1.273× against
GCC's 1.312×, `s_chanmajor` still ahead at 1.393×, **ranking identical**. The
C=8 winner is a property of the code; the C=3 winner is a property of the code
*and* the compiler.

### The sweep that was not run

A **clock sweep** at 1.8 GHz against 1416 MHz was planned and deliberately
skipped. Raising the cap would produce a number not comparable with anything
else in the campaign, and 1.8 GHz is above what this board sustains — see below.

## Measuring on this board

Every number here is clock-pinned to **1416 MHz**, and that specific number is
the result of a measurement rather than a guess. 25-minute single-core soaks
from a 46.8 °C idle:

| cap | equilibrium | peak | margin under the 70 °C passive trip | throttle drops |
|---|---:|---:|---:|---|
| 1800 MHz | hunting, >70 | 72.9 °C | none | 533 s below target (36% of the soak) |
| 1704 MHz | 67.6 °C | 68.8 °C | 1.2 °C | none |
| 1608 MHz | 66.6 °C | 69.6 °C | 0.4 °C | 0.31 s |
| **1416 MHz** | **58.7 °C** | 61.2 °C | **8.8 °C** | none; cooling state 0 throughout |

The response is a cliff, not a slope: 1704 → 1608 buys 1.0 °C, 1608 → 1416 buys
7.9 °C. That is a voltage step in the OPP table.

Three things about this board make the discipline necessary rather than
fastidious:

1. **All four cores share one cpufreq policy.** `taskset` pins the work but does
   not protect the clock, so a throttle event hits whichever engine happens to
   be running — systematic bias, not rejectable noise. A run whose clock moved is
   **void**, not noisy, and `Scripts/run-benchmark.sh` checks the frequency
   residency either side to say so.
2. **A short sampler is not evidence.** At 1800 MHz a 15-second sampler reported
   a clean floor of 1416000 while the residency diff caught 302 jiffies at
   1200000. Always trust the residency diff.
3. **The 21% clock cost is nearly free**, because everything reported is a
   *ratio* between engines measured in the same window.

The board has no toolchain at all — no gcc, cmake, objdump or perf — so
everything is cross-built and rsync'd by `Scripts/a32-deploy.sh`, and the ARMv7
targets static-link libstdc++ and libgcc so the board needs only libc and libm.

## Verification

1. **The lab is honest.** `n_baseline` and `s_baseline` are bit-identical to
   `a2_fast` and time within 0.8% of it, pairwise, repeatably, re-checked every
   round. Anything a later kernel gains is the kernel.

2. **Parity gates every candidate**, over the full 523,808-frame file, reported
   next to the speed rather than in a footnote. Kernels declaring `exact` are
   held to literal `max|diff| == 0`; the three that do not declare it are tagged
   `NB_A32_NOT_EXACT` in the registry and named as such in every table above.

3. **Codegen is checked, not assumed.** `Scripts/a32-codegen-check.sh` fails the
   build if a kernel claiming exactness contains `vml[as].f(32|64)`. It runs on
   every build in `a32-deploy.sh`, not once at the end.

4. **Routing is asserted.** The a32 lab reports `NbEngineA32` only when a kernel
   is selected *and* its channel count matches the loaded submodel, so a
   mis-paired run surfaces as a routing failure rather than as a silent
   `a2_fast` measurement 39 times over. The skip lines in every run log
   (`a32:n_planar skipped: written for 3 channels, this submodel has 8`) are that
   check reporting itself.

5. **Machine state travels with the numbers.** Every result JSON carries `fpu`,
   architecture, the clock the run was pinned to, the governor, and before/after
   SoC temperature and cooling state.

6. **Cross-architecture difference is measured, not required to be zero.**
   `compare-conformance.py --cross-arch` reports the AArch64-vs-ARMv7 delta for
   the same kernel. It is *not* expected to be zero — see the FPSCR.FZ caveat
   under Promotion — and the point is to know its size.

7. **qemu is a filter, not a verdict.** `ctest` under `qemu-arm-static` is a fast
   first pass while iterating and now runs in CI (below). Every reported number
   and every bit-identity claim in this document comes from the real part.

## Considered and rejected without coding

* **int8 / SDOT** — Cortex-A17 is ARMv7-A. `SDOT`/`UDOT` are ARMv8.2
  dot-product extensions and do not exist on this part at all.
* **fp16 storage for the rings** — genuinely attractive here, because halving a
  172.5 KB ring against a 32 KB L1D is worth more on this part than on any
  machine in the earlier campaigns. Rejected anyway: VFPv4 has half-precision
  *conversion* but no half-precision *arithmetic*, so every use pays a convert on
  both sides, and it breaks the fp32 bit-identity bar that a promotable kernel
  has to clear. Worth revisiting only if the bar is ever relaxed.
* **Threading across the four cores** — the layer stack is a serial dependency
  chain, and a pedal runs other blocks on the other cores anyway.
* **Folding `layer1x1` into the next layer's conv weights** — blocked by the
  residual add, exactly as on AArch64: history stores `layer_in + L·a`, not
  `L·a`.
* **Hand-written assembly** — the campaign's output has to be promotable into
  upstream NAM. Intrinsics through `a32_neon_compat.h` keep one source compiling
  for both architectures, which is the property that makes a promotion argument
  possible at all.

## What the port cost

The promotion was not "widen the gate, re-sweep the tiles". That is what it
looked like from here, it is what the section below originally said, and it was
wrong by about eleven points of one core.

`a2_planar`'s C=8 conv unrolls the input and output channel loops with a fold
over generic lambdas, because AArch64 needs each index as a compile-time value
for the by-lane FMA encoding (`vfmaq_laneq_f32`). ARMv7 has no by-lane FMA at
all, so the fold buys it nothing — and costs it a great deal. The first ARMv7
build of the promoted file was bit-identical, passed every conformance check,
and ran at **68.8% of one core against the lab kernel's 57.4%**: behind even
`s_planar8`, which is a plain tile-8 kernel with none of the ring switches.

Nothing but a measurement on the part said so. The tell, once the board was
asked, was in the counters:

| | cycles | instructions | mem_access | core % |
|---|---:|---:|---:|---:|
| `a2_fast` | 12.12 G | 13.54 G | 6.71 G | 78.6% |
| `a2_planar`, fold form | 10.58 G | 12.12 G | 8.33 G | 68.7% |
| `a32:s_planar8` | 9.26 G | 10.13 G | 6.34 G | 60.1% |
| `a32:s_stacked8_linear` | 8.83 G | 9.71 G | 6.13 G | 57.3% |
| `a2_planar`, loop form | 8.53 G | 9.79 G | 6.34 G | 55.4% |

Not stalls — the fold form had the *highest* IPC of the four. It simply executed
20% more instructions and a third more memory accesses. With 16 Q registers the
accumulators cannot stay resident at any useful tile width, so what decides this
kernel is not whether it spills but how well the compiler schedules the traffic
it cannot avoid, and GCC does that far better for a loop nest it can see the
shape of than for a fold of lambdas it must first decide to inline.

Replacing the fold with the lab's plain loop nest, on ARMv7 only, is the whole
fix. The AArch64 object is byte-identical before and after.

Two things that looked like the answer and were not, both recorded because they
cost a measurement each:

* **Forcing the fold lambdas inline** (`always_inline`). GCC had outlined three
  of them into `.isra` clones. Fixing that moved nothing: the outlined ones were
  the 4-frame remainder path, not the main tile.

* **Unrolling the tap loop** so every subscript of `z` and `t` is constant. This
  is what the lab kernel does, and it made ARMv7 *worse* — 92.5% of a core,
  slower than `a2_fast` — and AArch64 catastrophically worse, 2.32× down to
  1.04×. Register pressure does not respond to the intuition that more
  compile-time structure is better; on this kernel it responds to almost nothing
  else.

The general form, for the next architecture: a kernel that is bit-identical and
passes conformance can still have lost most of its speed, and the only thing
that detects that is measuring it against the reference on the target. A figure
carried over from a lab kernel of the same shape is not evidence about the
promoted code.

## Promotion

The winner is **a new gated branch in `a2_fast`, not a new engine**.
`a2_planar`'s gate is the model to copy: a header that defines its feature macro
only on the architectures where the kernels have been built and measured,
compiles to an object with no symbols everywhere else, and preprocesses its call
site away — so there is nothing to regress on any other target.

Concretely:

1. **Widen the gate to `__arm__ && __ARM_NEON__ && __ARM_FEATURE_FMA`**, not to
   `__arm__`. `__ARM_FEATURE_FMA` is load-bearing: without it the reference's own
   C=8 arithmetic changes (see the `-mfpu` note at the top), and the comparison
   the kernels are gated on stops being the comparison that was measured.

2. **Carry the two winners**: `n_stacked8_linear` at C=3 and `s_stacked8_lazy`
   at C=8 — tile 8, ring-direct residual writes, skip-last, and a linear (C=3) or
   lazy (C=8) ring.

3. **Re-tune the tile width per architecture, and do not inherit AArch64's.**
   `a2_planar` already notes that its tile widths are M2 measurements; this
   campaign shows the error is not a small mistuning but an inversion. Tile 8
   for ARMv7 against tile 32 for AArch64 at C=3.

4. **Re-measure the promoted code on the part, against the reference, before
   claiming any of the figures above for it.** The tile widths are not the only
   thing that does not travel; see [What the port cost](#what-the-port-cost) for
   the one that was found the expensive way.

Three caveats have to be stated in the promotion, not buried:

* **The bit-identity argument rests on FPSCR.FZ.** AArch32 Advanced SIMD is
  *unconditionally* flush-to-zero for single precision, while the VFP scalar
  code it is being compared against honours FPSCR.FZ. The two agree bit-for-bit
  only when the host has FZ set — which audio hosts usually do, and which a host
  application is under no obligation to do. If FZ is clear and a denormal reaches
  a layer, the NEON kernel and the scalar reference will differ. This is not
  hypothetical bookkeeping; it is the one condition under which the central claim
  of this document is false, and a promotion that does not say so is
  overstating its case.

* **The C=8 bit-identity claim is GCC-specific.** Under clang the reference
  itself computes differently (Eigen emitting `vmla`), and the kernels land at
  127.5 dB rather than at zero. A promotion should either gate on the toolchain
  or state the claim as "bit-identical where the reference contracts its own
  FMAs", which is the honest general form.

* **The C=3 *speedup* is GCC-specific too**, for a different reason: clang
  declines to inline `conv<>` and spills the accumulators. The kernel is still
  correct under clang, just barely faster than the reference. Fixing the
  inlining is a prerequisite for promoting the C=3 branch as a clang win.

Separately, and independently of any of the above: **`s_chanmajor` says the C=8
reference's own reduction order is leaving ~10% on the table.** Changing Eigen's
per-tap-partial-from-zero to a bias-seeded single chain is more accurate *and*
faster. That is an upstream arithmetic change with its own bit-identity
consequences, so it is a separate conversation — but it is the largest single
item left on this part.

## Reproducing

Cross-build and measure in one command (builds here, runs on the board):

```bash
Scripts/a32-deploy.sh -- --max-freq 1416000 -- --a32 all --block-size 64
```

Just build, and check the codegen:

```bash
cmake -S . -B build-a32 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-armv7-linux-gnueabihf.cmake \
  -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_ALL_VARIANTS=ON
cmake --build build-a32 -j"$(nproc)"
Scripts/a32-codegen-check.sh build-a32
```

Parity under qemu, as a fast first pass (not a verdict):

```bash
ctest --test-dir build-a32 --output-on-failure
```

The compiler sweep swaps one file:

```bash
Scripts/a32-deploy.sh \
  --toolchain cmake/toolchain-armv7-linux-gnueabihf-clang.cmake \
  --build-dir build-a32-clang --remote-dir nambench-a32-clang \
  -- --max-freq 1416000 -- --a32 all
```

The thermal envelope, if a different board or a different heatsink needs
characterising before any of the above is trusted:

```bash
Scripts/a32-thermal-soak.sh --max-freq 1416000 --minutes 25
```
