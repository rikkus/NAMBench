# Optimising the A2 slimmed (3-channel) path

`fused` beats `a2_fast` by 1.9× on the A2 **full** submodel. It cannot touch the
A2 **slimmed** submodel at all, because `parse_spec` rejects it on one line:

```cpp
if (as.channels % 4 != 0 || as.channels > kMaxChannels) return false;   // fused.cpp:142
```

Every other fused requirement is met by the nano config. So nano falls through
to `a2_fast`'s `if constexpr (Channels == 3)` branch — a fully-unrolled
**scalar** 3×3 GEMV
([a2_fast.cpp:447-557](vendor/upstream/NAM/wavenet/a2_fast.cpp)).

This is a record of nineteen kernels — twelve planned ideas plus a tile sweep,
two compositions and one that the measurements asked for — each measured by the
project's existing benchmark protocol rather than a throwaway harness. It
records what each one actually did, including the ones that were expected to
fail and did, the one that was expected to win and came last, and the one where
the plan's reasoning turned out to be backwards.

**Constraint: fp32 only.** Every candidate has to stay numerically equivalent to
`a2_fast`, so a winner is a drop-in replacement needing no listening test.
fp16/bf16 storage and arithmetic are out of scope despite halving the ring
footprint.

## Why it is not a FLOP problem

| | |
|---|---:|
| Real arithmetic in nano | **83 MMAC/s** at 48 kHz (1731 MAC/frame) |
| M2 single-core fp32 FMA peak | ~100 GFLOP/s |
| `z` accumulator load/stores per frame per K=6 layer | **30** (vs 18 history loads) |
| Ring footprint, a2_fast's default sizing | **172.5 KB** (M2 P-core L1D is 128 KB) |

Nano runs about three orders of magnitude below the machine's arithmetic peak.
It is bound by instruction count, `z`-buffer round-trips, dependency-chain
latency, per-block overhead and cache footprint. Every candidate targets one of
those, not the FLOPs.

## The result

Apple M2 (4P + 4E), macOS 27.0, Release, 48 kHz, 64-frame blocks, submodel 0 of
`Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam` (3 channels, 1871 weights),
523,808 frames per pass. Full protocol: 5 s warm-up discarded, 30 s of timed
passes, mean of the fastest 70% required to agree within 3%. Every variant below
was accepted on its first attempt.

| | Kernel | Mean / pass | Fastest | vs `a2_fast` | Parity |
|---|---|---:|---:|---:|---|
| | `upstream` (`a2_fast`) | 60.11 ms | 59.29 ms | — | — |
| 0 | `baseline` — control | 59.57 ms | 58.81 ms | 1.009× | **bit-identical** |
| 1 | `restrict` | 58.72 ms | 58.30 ms | 1.024× | 146.8 dB |
| 2 | `framemajor` | 99.14 ms | 97.97 ms | **0.606×** | 133.3 dB |
| 3 | `pad4` | 71.99 ms | 71.46 ms | **0.835×** | 132.7 dB |
| 4 | `planar` | 50.97 ms | 50.34 ms | 1.179× | **bit-identical** |
| 5 | `planar_ring` | 57.45 ms | 56.74 ms | 1.046× | **bit-identical** |
| 6 | `ringdirect` | 47.21 ms | 46.75 ms | 1.273× | **bit-identical** |
| 7 | `vext_taps` | 50.10 ms | 49.64 ms | 1.200× | **bit-identical** |
| 8 | `ld3` | 56.44 ms | 56.00 ms | 1.065× | **bit-identical** |
| 9 | `skiplast` | 50.02 ms | 49.74 ms | 1.202× | **bit-identical** |
| 10 | `prefetch` | 53.60 ms | 52.84 ms | 1.121× | **bit-identical** |
| 11 | `widetile8` | 43.12 ms | 42.71 ms | 1.394× | **bit-identical** |
| 11 | `widetile16` | 38.52 ms | 38.11 ms | 1.561× | **bit-identical** |
| 11 | `widetile32` | 34.30 ms | 34.05 ms | 1.752× | **bit-identical** |
| 11 | `widetile64` | 41.24 ms | 40.62 ms | 1.458× | **bit-identical** |
| | `stacked16` | 34.71 ms | 34.40 ms | 1.732× | **bit-identical** |
| | `stacked32` | 30.77 ms | 30.41 ms | 1.954× | **bit-identical** |
| | `planar_linear` | 48.58 ms | 47.97 ms | 1.237× | **bit-identical** |
| | **`stacked_linear`** | **28.65 ms** | 28.34 ms | **2.098×** | **bit-identical** |

Rows 5–11 are `planar` plus exactly one switch each, so they read as an
ablation against row 4. `stacked*` compose the switches that paid: a wide frame tile,
residual writes straight into the next layer's ring, skipping the final
`layer1x1`, and — for `stacked_linear` — the linear ring.

**The winner is bit-identical to `a2_fast` over all 523,808 frames.** So is
every kernel in the planar family. Under the fp32-only constraint that was the
point: a drop-in replacement that needs no listening test. The three kernels
that are *not* bit-identical are the three that lost.

Two caveats on the 2.098× headline, both explained below:

- Against `a2_fast` compiled at its own better ring-mode setting rather than the
  one it ships with, the same kernel is **1.734×**. That is the number to carry
  into a promotion argument.
- Run-to-run spread on this machine is around ±2%, so treat differences smaller
  than that as no difference. `restrict` at 1.024× is inside that band.

## What each candidate did

### The control

**`baseline`** is a verbatim port of `a2_fast`'s 3-channel branch — the same
rings, the same buffer sizing, the same inner loops — compiled into a third
framework built from the *same* `vendor/upstream` tree through the *same* target
template as `upstream` itself. It exists to answer one question: does the lab
add overhead of its own?

It does not. Output is **bit-identical**, and pairwise against `upstream` it
lands at 1.006×, 0.995×, 1.011× over three runs.

One caveat worth recording, because it nearly caused a wrong conclusion: in an
early twenty-variant run `baseline` came out 2.9% *faster* than `upstream`,
which looked like a systematic lab artefact. It was not. Repeating the same
run put it back at 1.007×, and the pairwise repeats above bracket it tightly.
Per-process variation of a couple of percent is real here — 172 KB of ring
buffers against a 128 KB L1D makes the result sensitive to where the allocator
happens to put them — so single-run differences under about 2% are not
differences.

### The ones that were meant to win

**`planar`** is the core idea. `a2_fast` keeps three channels interleaved and
vectorises across *channels*, which at C=3 wastes a quarter of every lane before
it starts. The planar layout keeps three separate channel planes and vectorises
across *frames*, so a NEON register holds four consecutive frames of one channel
and every lane does real work:

| | FMA per frame |
|---|---:|
| `a2_fast` (9 scalar FMA/frame) | 9 |
| `pad4` (C=4 vectors, one lane wasted) | 3 |
| `planar` (3 loads + 9 lane-FMAs per 4 frames) | 2.25 |

`z` also stops touching memory: it lives in twelve registers across all K taps,
removing all 30 load/stores per frame per K=6 layer.

It is **bit-identical** to `a2_fast`, which is worth being explicit about. In a
planar FMA the weight is the scalar and the vector is four frames, so the
reduction order per output channel — bias, then tap 0 inputs 0/1/2, then tap 1,
and so on — is exactly `a2_fast`'s scalar chain, one lane per frame. Nothing is
reassociated. The head convolution comes out the same way.

**`widetile8/16/32/64`** sweeps the frame-tile width, and is by a wide margin
the largest single effect in the whole exercise — far more than the layout
change it sits on top of. Two things are happening and the second is probably
the bigger one:

- more independent FMA chains (3 at tile 4, 24 at tile 32), against four 4-cycle
  FMA pipes that want around sixteen;
- **weight-load amortisation**. Each tap loads three weight vectors regardless
  of tile width. At tile 4 that is three loads feeding three FMAs; at tile 32 it
  is three loads feeding twenty-four.

The sweep turns between 32 and 64, exactly where the accumulators (48 vector
registers at tile 64) stop fitting in the register file.

**`skiplast`** is two pieces of work the model never reads. The final layer's
`layer1x1` computes a residual that nothing consumes — there is no layer 24, and
the head reads `head_sum`, not `_layer_in`. `fused` already skips it
(`skip_l1x1_output`); `a2_fast` does not. And the first layer's `head_sum`
accumulate has nothing to accumulate onto, so it can store instead of
load-add-store, which makes the per-block `memset` of `head_sum` unnecessary.

**`ringdirect`** has each layer's tail store its residual straight into the next
layer's ring instead of into `_layer_in`, which the next layer then memcpys in.
That removes 23 memcpys *and* 23 extra passes over a 768-byte buffer per block.
It was genuinely unclear beforehand whether this would win: it trades one small
always-hot buffer for 24 scattered ring windows spread over 172 KB.

The block being written can straddle the ring wrap. Rather than splitting the
frame loop, the write runs off the end of the ring into the mirror region (which
is always at least `max_buffer_size` columns long) and the overhang is folded
back afterwards, so the inner loop stays straight-line.

### The ones that were meant to lose

**`restrict`** — `__restrict` on the hot pointers, `.data()` hoisted out of the
frame loops, `#pragma clang fp contract(fast)`. Close to free and close to
nothing, as expected. It is not bit-identical (146.8 dB below signal): the
contraction pragma adds fused multiply-adds across statements where `a2_fast`
had separate rounding steps. `a2_fast` already gets an FMA for every
`a0 += w * s` because each is a single statement, so there was little left to
find.

**`vext_taps`** — at dilation 1 the six taps of a K=6 layer are six consecutive
frames, so all six 4-frame windows live inside one nine-frame span: three loads
and five `vextq_f32` instead of six loads. Expected to lose, because `vext`
competes with the FMAs for vector issue slots while loads go down the load
pipes. It did not lose; it did not win either — 1.7% ahead of `planar`, inside the
run-to-run band. The loads it removes were not the bottleneck.

**`ld3`** — keep `a2_fast`'s interleaved rings and use `vld3q_f32` to
de-interleave into planar registers at the point of use, testing whether the
layout change is necessary at all. It is: `ld3` recovers only a small part of
what `planar` gets. The reason is structural — the conv reads each ring window
once per tap, so the de-interleave is paid K times per block, whereas the planar
layout pays for it once per layer.

**`prefetch`** — the dilation-239 layers reach about 14 KB behind the write
head, so their taps are a cold miss every block. Explicit prefetching made
things about 5% *worse* than plain `planar`, which is the usual outcome when the
access pattern is a fixed repeating offset that the hardware prefetchers already
handle: the prefetches arrive early enough to evict something that was wanted.

**`framemajor`** — frame-outer/tap-inner with nine partial accumulators summed
at the end, two frames per tile, giving eighteen independent FMA chains instead
of three, with `z` never reaching memory. This was expected to do well and was
by far the worst result in the set, at roughly 0.6× — slower than the scalar
code it was trying to beat. Nine partial accumulators plus two frames plus the
saved inputs is more live scalar state than the FP register file wants to hold,
and the spills cost more than the round-trips they were saving. The vector
version of the same idea (`planar` with a wide tile) gets the ILP without the
spilling, because a vector register holds four frames instead of one.

It is also the only candidate that reassociates the *convolution* reduction
(133.3 dB below signal rather than bit-identical), which was the accepted cost
of the nine partials. Being both slower and less exact, it is a clean rejection.

For completeness on the other two non-exact rows: `pad4`'s 132.7 dB comes from
the head, where `fused`'s `head_conv_block` finishes with a pairwise `vaddvq_f32`
where `a2_fast` sums sequentially; its convolution is exact. `restrict`'s
146.8 dB comes entirely from the extra FMA contraction the pragma allows.

**`pad4`** — pad C=3 to C=4 with zeroed weights and run `fused`'s existing
`conv_tile`/`tail_tile`/`head_conv_block` at Q=1. This is the obvious answer to
"the refusal is one line, just delete it", and it is the wrong answer: it loses
to the scalar code it replaces. A quarter of every lane is multiplied by zero,
and the rings grow from 172.5 KB to 230 KB — further past the L1D they already
did not fit in.

The padding itself is provably safe, and it is worth writing down since it is
the first thing anyone will propose. Every weight touching lane 3 is zero in
both directions at every stage — rechannel, conv (row and column) and bias,
mixin, layer1x1 (row and column) and bias, head conv — so `z[3]` is exactly
zero, `LeakyReLU(0)` is zero, and the pad lane can neither leak into a real
output nor accumulate a value that could denormalise. It is a hard zero for the
whole run.

### The one the plan got backwards

**`planar_ring`** was the footprint argument: `a2_fast` rounds every ring up to
a power of two and mirrors all 24 of them unconditionally on every block —
about 18 KB of memcpy per 64-frame block that almost nothing reads. Sizing each
ring exactly and mirroring lazily (what `fused` does) takes the footprint from
172.5 KB to 110.4 KB, crossing under the M2's 128 KB L1D.

It was a clear loss — roughly back to `baseline` speed, giving up everything
`planar` had gained.

The reason is specific and, in hindsight, obvious. An exactly-sized ring for a
short-lookback layer is barely longer than one block: the dilation-1 layers have
a 5-frame lookback, so capacity is 69 columns and each block advances the write
head by 64. Almost every block wraps. And when a read wraps, the lazy mirror
fires **once per tap** rather than once per layer — up to six memcpys where the
eager scheme did one. Smallest footprint, most copying.

So the footprint was not the problem; the copying was. Which pointed at a third
option the plan did not list.

### The one the data asked for

**`planar_linear`** and **`stacked_linear`** use `a2_fast`'s *other* ring mode —
`NAM_A2_RING_MODE=0`, a linear buffer written forward until it runs out and then
rewound with a `memmove`. No mirror at all, no masking, every read contiguous,
at the price of an occasional large `memmove`.

These were added after the ring-mode sweep below turned up something the plan
did not anticipate: compiling *upstream's own* `a2_fast` at
`NAM_A2_RING_MODE=0` made it **14% faster** on this submodel. `a2_fast` ships
with mode 1.

The footprint barely moves — 166.7 KB against 172.5 KB — so this is not a cache
argument. It is copy traffic:

| ring strategy | footprint | copy traffic per 64-frame block |
|---|---:|---:|
| pow2 + eager mirror (`a2_fast` default) | 172.5 KB | 18.0 KB |
| exact + lazy mirror (`planar_ring`) | 110.4 KB | more than either, and branchy |
| linear + rewind (`planar_linear`) | 166.7 KB | ~10.7 KB amortised |

The linear scheme copies 60 bytes per block on a dilation-1 layer where the
eager mirror copies 768, and pays its big 14 KB `memmove` on the dilation-239
layers only once every nineteen blocks.

## The two sweeps

### Block size

Frames per `process()` call, 2 s warm-up / 5 s timing:

| Block | `a2_fast` | `baseline` | `stacked32` | speedup |
|---:|---:|---:|---:|---:|
| 32 | 62.81 ms | 63.51 ms | 32.62 ms | 1.926× |
| 64 | 59.56 ms | 58.51 ms | 31.36 ms | 1.899× |
| 128 | 59.04 ms | 58.81 ms | 30.49 ms | 1.936× |
| 256 | 59.67 ms | 59.07 ms | 29.63 ms | 2.014× |
| 512 | 59.85 ms | 59.43 ms | 28.56 ms | 2.095× |

Per-block overhead is proportionally much larger at C=3 than at C=8, and it
shows: the optimised kernel keeps improving all the way to 512 frames while
`a2_fast` is flat after 64. `a2_fast` is not paying per-block overhead so much
as drowning it in per-frame scalar work.

The 32-frame column is the one that matters for a plugin: at 32 frames the
optimised kernel still holds its advantage, so this is not a win that only
exists at unrealistic buffer sizes.

### Ring mode

`NAM_A2_RING_MODE` is a compile-time `#define` inside `a2_fast`. Mode 1 (pow2 +
eager tail mirror) is the shipped default; mode 0 is a linear buffer rewound by
`memmove`. The lab kernels do not read the flag, so they are unaffected by it
and act as a control on the rebuild. 3 s warm-up / 10 s timing:

| Kernel | `a2_fast` at mode 1 | `a2_fast` at mode 0 |
|---|---:|---:|
| `upstream` (`a2_fast`) | 59.41 ms | **51.85 ms** |
| `baseline` (unaffected) | 59.44 ms | 59.19 ms |
| `planar` (unaffected) | 51.53 ms | 51.76 ms |
| `stacked32` (unaffected) | 31.47 ms | 31.63 ms |
| `stacked_linear` (unaffected) | 29.56 ms | 29.90 ms |
| **winner vs `a2_fast`** | **2.010×** | **1.734×** |

This is the sweep that changed the shape of the answer, and it is a result about
`a2_fast` rather than about any candidate: **`a2_fast`'s shipped ring mode is
the wrong one for the slimmed model, by 14%.** Changing one `#define` recovers
more than most of the candidates in this document.

That also means the headline speedup has to be quoted carefully. Against
`a2_fast` as shipped the winner is 2.010×; against `a2_fast` compiled at its own
better setting it is 1.734×. The second number is the honest one to carry into a
promotion argument — most of a 14% head start is not something a new kernel
should get credit for.

The rows for the lab kernels are the control here: they do not read the flag, so
their times should not move between the two builds, and they do not (all within
0.7%). Whatever the flag changed, it changed inside `a2_fast`.

## Verification

1. **The lab is honest.** Kernel 0 `baseline` is bit-identical to `upstream` and
   times within 1% of it, pairwise, repeatably. Anything a later kernel gains is
   the kernel.

2. **Parity gates every candidate.** Every kernel is compared against `upstream`
   over the full 523,808-frame file, and the comparison is reported next to the
   speed rather than in a footnote. Sixteen of the nineteen are bit-identical;
   the three that are not are all slower than the code they replace, so nothing
   in this document trades accuracy for speed.

3. **The A2 full comparison is unperturbed.** Re-run after all the harness
   changes: `upstream` 428.34 ms, `fused` 225.67 ms, **1.898×**, parity
   132.6 dB below signal — the same result as before this work started.

4. **The engine is asserted, not assumed.** The slim framework reports
   `NbEngineSlim` only once a kernel has actually been selected; until then it
   routes and reports exactly as `upstream`. A run that forgot to select one
   fails the existing engine check rather than quietly measuring `a2_fast`
   nineteen times.

5. **`fused` is excluded from slimmed runs, by shape not by flag.** On a
   3-channel submodel its detector rejects the shape, and under
   `ScopedEnginePrefer(FusedNeon)` a rejected shape falls through to the
   *generic* engine rather than to `a2_fast`. The runner reads the channel count
   from the file and drops `fused` when it is not a multiple of four.

## Considered and rejected without coding

- **Folding `layer1x1` into the next layer's conv weights** — blocked by the
  residual add: history stores `layer_in + L·a`, not `L·a`.
- **int8/SDOT, fp16, bf16** — out of scope under the fp32-only constraint,
  despite fp16 halving the ring footprint.
- **Accelerate / BNNS / AMX / SME** — call and setup overhead dwarfs a 3×3.
- **Threading** — the layer stack is a serial dependency chain.

## Promotion

The winner is the planar family, not `pad4`, so the production change to
`fused.cpp` is a new kernel family rather than a relaxed detector:

1. **Relax the detector**, but not to "any channel count". `parse_spec` currently
   rejects `channels % 4 != 0`. Replace that with a check that the shape is
   handled by *some* kernel family — the existing `Q`-templated one for multiples
   of four, and a planar family for the rest.

2. **Add the planar family** for `channels % 4 != 0`. It is not specific to C=3;
   the same structure works for any small channel count, with one accumulator
   register per channel per 4-frame lane and a tile width chosen to keep the
   accumulators in registers (`channels × tile / 4 ≲ 24`). For C=3 that is
   tile 32.

3. **Carry the three switches that paid**: a wide frame tile, residual writes
   straight into the next layer's ring, and skipping the final `layer1x1`
   (`fused` already does the last of these).

4. **Reconsider the ring strategy for narrow models generally.** The
   eager-mirror ring `fused` and `a2_fast` mode 1 share costs 18 KB of memcpy
   per block, which is negligible next to a C=8 layer's arithmetic and is not
   negligible next to a C=3 layer's. This is worth re-measuring on the *full*
   submodel before changing anything there — the finding above is a C=3 finding.

The change stays confined to `fused.cpp`'s detector plus one kernel family, so
`a2_fast` remains byte-identical across both checkouts and
`Scripts/fetch-vendor.sh` keeps passing.

Separately, and independently of any of this: **`a2_fast` should default to
`NAM_A2_RING_MODE=0`, or choose per channel count.** That is a one-line change
upstream worth 14% on this model, and it needs no new kernel at all.

## Reproducing

```bash
./Scripts/fetch-vendor.sh && xcodegen generate && xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release build
```

```bash
"$(xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release -showBuildSettings | awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')/nambench" --submodel narrowest --slim all
```

The ring-mode sweep needs a rebuild, because the mode is a compile-time `#define`
inside `a2_fast`:

```bash
xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release OTHER_CPLUSPLUSFLAGS="-DNAM_A2_RING_MODE=0" build
```
