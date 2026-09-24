# Impulse-response convolution: partitioned FFT against direct

AudioDSPTools convolves the plugin's cabinet IR directly: one multiply-add per
tap, per output sample. The
[`partitioned-ir`](https://github.com/rikkus/AudioDSPTools/tree/partitioned-ir)
branch keeps a direct FIR head for the first block, so latency stays at zero,
and moves the rest of the IR into uniformly partitioned FFT convolution. The
README and [BENCHMARKING.md](BENCHMARKING.md#impulse-responses) carry the
current tables. This file records what was investigated around them, what was
changed on the branch as a result, and what was considered and left alone.

Three machines throughout: an M2 MacBook Air, a Raspberry Pi 500 (Cortex-A76,
2.4 GHz) and a Tinker Board (Cortex-A17, held at 1.416 GHz for benchmark runs).
64-frame callbacks at 48 kHz, mono IR, unless stated otherwise.

## Decisions

| | Outcome | Where |
|---|---|---|
| Multiply and store only the half of each spectrum the inverse reads | **Adopted.** Bit-identical output; FFT path's p99 at 8192 taps down 23-29% | `partitioned-ir` [`a09e360`](https://github.com/rikkus/AudioDSPTools/commit/a09e360c374e2774e2906ac8f1a4e90113528390) |
| Raise `kAutoDirectMaxTaps` from 256 to 512 | **Adopted.** FFT lost at 512 taps on both ARM boards | `partitioned-ir` [`e2dc6bc`](https://github.com/rikkus/AudioDSPTools/commit/e2dc6bcaa1cffe6708f93935218829b61eefbf92) |
| Trim IR tails below some threshold to save CPU | **Rejected.** A per-sample -60 dB cut is audible room sound, not silence, and the FFT path already makes full length cheap | [below](#is-a-shorter-ir-good-enough) |
| Spread the partition multiplies across the quiet callbacks | **Adopted.** FFT path's p99 at 8192 taps down 39-55%; output independent of callback size, 145 dB from before | `partitioned-ir` [`b505422`](https://github.com/rikkus/AudioDSPTools/commit/b50542244c2a6c40602eac58a80823c13fb2678d), [below](#spreading-the-multiplies-b505422) |
| Free the three 32 KiB buffers the FFT path never uses (96 KiB) | **Declined for now.** Not worth the effort | [below](#memory) |
| Stop keeping a copy of the raw IR file (94 KiB for a 500 ms file) | **Not pursued.** It is how a sample-rate change rebuilds the convolver; dropping it is a behaviour change | [below](#memory) |
| Aim this convolver at Pico-class boards (264 KiB SRAM, no FPU) | **Out of scope.** Neither the memory nor the arithmetic fits | [below](#memory) |

NAMBench pins the branch at `b505422`, and the published numbers were
measured there.

## Taps, and what `kAutoDirectMaxTaps` chooses

A tap is one sample of the impulse response. Direct convolution costs one
multiply-add per tap for every output sample, so a 4096-tap IR costs twice what
a 2048-tap one does. The plugin resamples the IR to the session rate and
AudioDSPTools then truncates it at 8192 taps: 170.7 ms at 48 kHz, 85 ms at
96 kHz. Upstream `main` always convolves directly.

On the branch, `ConvolutionImplementation::Auto` picks:

- **Direct** at or below `kAutoDirectMaxTaps` taps. The branch's direct path is
  a rewrite (it manages its own history buffer) but is bit-identical to
  upstream's at every length, and within about 3% of it in time.
- **FFT** above it. The first block of taps still runs directly (256 taps for
  IRs up to 2048, 512 up to 8192), which is what keeps latency at zero; the
  rest is split into partitions of that block size, each convolved by FFT.

The threshold was 256 and is now 512. See [below](#auto-threshold-to-512-e2dc6bc).

## The IR library

`ir-study/ir_survey.py` over the 1538 WAVs in the library this was checked
against: Orange 4x12 (954 files), Ampeg 4x10 (573), Deftones (8), Marshall (2),
Mesa (1).

| Rate | Format | Files |
|---|---|---:|
| 44.1 kHz | 24-bit PCM | 509 |
| 48 kHz | 24-bit PCM | 517 |
| 48 kHz | 32-bit float | 3 |
| 96 kHz | 24-bit PCM | 509 |

All mono. **1530 of 1538 reach the 8192-tap cap** once resampled to 48 kHz;
most are 500 ms files. Only the eight Deftones IRs are shorter, at 2040 taps.
So for this library the 8192-tap row is not an extreme: it is nearly every IR
the plugin will ever load.

## Is a shorter IR good enough?

The obvious saving is to convolve less of each IR. The per-sample view makes
that look free, and it is not.

In 48 kHz taps, across all 1538 files:

| Measure | min | median | p90 | max |
|---|---:|---:|---:|---:|
| Last sample within 60 dB of the peak | 762 | 4212 | 5503 | 16477 |
| Point by which 99.99% of the energy has arrived | 540 | 3642 | 4253 | 9884 |

**-60 dB per sample is not silence.** Every tail tap is added into every output
sample, so thousands of individually quiet taps add up. What a cut costs is the
energy it discards: for broadband input that is roughly the level of the error
it adds to the output.

| Cut at | Discarded energy, median | p90 | worst |
|---:|---:|---:|---:|
| 1024 taps | -24.5 dB | -20.8 dB | -3.0 dB |
| 2048 taps | -34.0 dB | -29.5 dB | -11.4 dB |
| 4096 taps | -42.4 dB | -38.9 dB | -21.1 dB |
| 8192 taps (the existing cap) | -58.7 dB | -52.7 dB | -34.3 dB |

The last 10% of each file sits at a median of -109.7 dB relative to its peak,
so what is being cut is decaying room sound and cabinet resonance, not the
recording's noise. The worst cases are the Orange pack's rear, side and room
mic positions, which hear the most room: `REAR 1` keeps half its energy beyond
1024 taps, and even the existing 8192 cap discards about -34 dB of it.

**Decision: no trimming by default.** A 4096-tap cut would shorten the decay
audibly on the room-heavy positions. The FFT path convolves all 8192 taps for a
fraction of what direct convolution spends, so the saving trimming would buy is
small anyway.

A first look at a random 120 files reported worst cases of -12 dB at 1024 taps
and -42 dB at 8192. That sample missed the rear mics, and read the three
float32 files as 24-bit integers. The figures above are from every file.

## The FFT path is bursty, and predictably so

`ir-study/ir_trace.cpp` timed every callback, through the benchmark's own shim
libraries. The page is [`ir-study/burstiness.html`](ir-study/burstiness.html)
and its data `ir-study/data/m2-trace-summary.json`. These traces were taken
before the half-spectrum change.

At 8192 taps, the FFT path collects 512 samples, then transforms them,
multiplies them against all 15 partitions and transforms back, inside the one
callback that completes the 512. With 64-frame callbacks that is every 8th:

| M2, 8192 taps | per callback | share of the 1.33 ms deadline |
|---|---:|---:|
| Shipping (direct), every callback | ~56 µs | 4.2% |
| FFT path, 7 callbacks in 8 | ~2.7 µs | 0.2% |
| FFT path, the 8th | ~30 µs | 2.2% |

- **It is completely predictable.** The spike lands on callbacks 8, 16, 24...
  after a reset. Guitar, white noise, silence and a sine gave the same quiet
  and spike costs to within 0.01% of the deadline. So did the synthetic IR, a single click and
  four real cabinet IRs. Nothing in the path branches on sample values, and
  denormals are flushed.
- **`block_p99_percent` measures the spike.** One callback in eight is a
  transform callback, so the 99th percentile falls inside that cluster: the FFT
  path's p99 is what a transform callback costs, not a rare outlier.
- **The spike grows with the number of partitions.** At 1024 taps and below it
  costs more than a shipping callback, even though the average is lower, which
  is why the FFT path's p99 is worse than shipping's there.
- **Larger host blocks flatten it.** The period is the FFT block over the host
  block. At 32 frames the spike comes once in 16 and reaches 4.4% of the
  shorter deadline; at 512 frames or more every callback runs a transform and
  the p99 sits on the average.
- **The boards show the same shape.** Before the half-spectrum change the FFT
  path's p99 at 8192 taps was 4.9x its average on the M2, 5.1x on the Pi 500
  and 4.6x on the Tinker Board.

Two ways to lower the spike were identified: multiply only half the spectrum,
which was done (next section), and spread 14 of the 15 partition multiplies
across the seven quiet callbacks, since they use input blocks that arrived
earlier. The second was done [later](#spreading-the-multiplies-b505422).

## Listening

`ir-study/ab_render.cpp`, built once against each AudioDSPTools tree, renders
`audio-input/input.wav` through a real IR file exactly as the plugin would:
load, resample, truncate, 64-frame blocks. `ir-study/ab_mix.py` turns the two
renders into level-matched float32 WAVs and a boosted difference.

| IR | Difference RMS, re signal | Difference peak |
|---|---:|---:|
| Orange 4x12 (York Audio 421v-CN) | -142.7 dB | -136.1 dBFS |
| Ampeg 4x10 (York Audio 57-CNT) | -143.0 dB | -137.9 dBFS |
| Mesa OS 4x12 57 | -143.6 dB | -139.0 dBFS |
| Deftones White Pony (2040 taps) | -141.5 dB | -139.3 dBFS |

The difference sits around the bottom of what 24-bit audio can represent. The
renders were made at `ba646f4`; the half-spectrum change was later shown to
produce byte-identical output for the same four IRs, so they stand for the
current branch too.

**The WAVs are not committed.** They are renders of commercial cabinet IRs and
come to about 26 MB. `ir-study/data/ab-samples-README.txt` is the note that
went with them; the tools regenerate them from any IR files.

## Half the spectrum (`a09e360`)

The page is [`ir-study/half-spectrum.html`](ir-study/half-spectrum.html).

A 1024-point FFT of real samples gives 1024 bins, but bin `1024 - k` is always
the complex conjugate of bin `k`: bins 513-1023 carry nothing bins 1-511 do not.
The IR is real too, and a product of two conjugate-symmetric spectra is
conjugate-symmetric, so the inverse transform can rebuild the output from bins
0-512 alone.

Eigen already exploited that inside the transforms. Its real-input forward FFT
computes bins 0-512 with a half-size complex FFT and then copies conjugates into
513-1023 unless `HalfSpectrum` is set; its real-output inverse reads only bins
0-512. In between, the branch multiplied and accumulated all 1024 bins for every
partition, so 511 of every 1024 complex multiply-adds produced values nothing
read.

`a09e360` sets `HalfSpectrum` and stores and multiplies `fft_size / 2 + 1` bins.
Bins 0-512 get exactly the arithmetic they got before, so the output is
**bit-identical**: the same bytes as `ba646f4` for the four IRs above through
the guitar DI, `test_ir_convolution` passes, and the benchmark's accuracy check
still puts the FFT path 136-138 dB below upstream.

The benchmark on all three machines, FFT path, before -> after (`ba646f4` ->
`a09e360`):

| taps | M2 core% | M2 p99 | Pi 500 core% | Pi 500 p99 | Tinker core% | Tinker p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.091 -> 0.090 | 0.12 -> 0.11 | 0.201 -> 0.201 | 0.22 -> 0.22 | 1.810 -> 1.808 | 2.43 -> 2.43 |
| 512 | 0.168 -> 0.159 | 0.40 -> 0.37 | 0.407 -> 0.391 | 1.02 -> 0.96 | 3.276 -> 3.207 | 8.29 -> 8.01 |
| 1024 | 0.188 -> 0.169 | 0.48 -> 0.41 | 0.448 -> 0.413 | 1.18 -> 1.04 | 3.495 -> 3.312 | 9.19 -> 8.44 |
| 2048 | 0.229 -> 0.191 | 0.65 -> 0.49 | 0.535 -> 0.455 | 1.52 -> 1.21 | 3.920 -> 3.557 | 10.89 -> 9.41 |
| 4096 | 0.378 -> 0.338 | 1.75 -> 1.28 | 0.718 -> 0.645 | 3.16 -> 2.56 | 5.211 -> 4.851 | 21.11 -> 18.18 |
| 8192 | 0.459 -> 0.380 | 2.26 -> 1.61 | 0.883 -> 0.727 | 4.48 -> 3.22 | 6.102 -> 5.295 | 28.26 -> 21.74 |

No length got slower on any machine. 256 taps does not move, as it should not:
the whole IR fits in the direct head there and no transform runs. The p99 falls
further than the average because the saving lands entirely on the transform
callback. Against shipping at 8192 taps, the FFT path went from 9.30x to 11.29x
on the M2, 7.50x to 9.13x on the Pi 500 and 6.89x to 7.91x on the Tinker Board.

The branch was not kept on the M2's word. Both ARM boards were re-measured
first, with nothing uploaded, on the grounds that the change was not worth
keeping if it made either of them worse. It did not, and all three machines
were then measured again at the pinned commit for upload.

### Storage or arithmetic?

`a09e360` did two things at once: the loop stops at bin 512, and the spectra are
stored at 513 bins instead of 1024 (240 KiB down to 120 KiB at 8192 taps).
`ir-study/spectrum_storage_ab.cpp` separates them, comparing three builds that
take turns in one process:

- **V0** `ba646f4`: every bin multiplied and stored
- **V1** loop stops at bin 512, storage still full size
- **V2** `a09e360` as committed

Median transform callback at 8192 taps:

| | V0 | V1 | V2 | V2 against V1 |
|---|---:|---:|---:|---:|
| M2 | 29.75 µs | 21.21 µs | 21.25 µs | +0.2% |
| Pi 500 | 59.35 µs | 42.54 µs | 42.48 µs | -0.1% |
| Tinker Board, 1.8 GHz | 272 µs | 207 µs | 205 µs | -1.0%, in two runs |

At 1024-4096 taps V2 was within 1% of V1 everywhere. **Almost all of the gain
is the arithmetic.** Once the loop stops at bin 512 it never touches the upper
half of each buffer, so shrinking the buffers frees memory that was already
unread; the Tinker Board, with the smallest caches, is the only one to show
anything. The smaller buffers stay: they cost nothing.

That comparison ran at the Tinker Board's default 1.8 GHz, not the benchmark's
1.416 GHz cap, so its microseconds do not line up with the published tables;
the comparison between versions is unaffected. Short interleaved runs are
enough for a same-machine A/B like this, because drift lands on all three
builds alike. The full protocol is for numbers that have to stay comparable
across days and uploads.

## `Auto` threshold to 512 (`e2dc6bc`)

At 512 taps, after the half-spectrum change, the FFT path still costs 10% more
than direct on the Pi 500 and 21% more on the Tinker Board, with a p99 about
2.5x direct's on both. From 1024 taps it wins everywhere. The M2 favours FFT
from just above 256, but only by 1.25x at 512, and its p99 is worse there too.

So `kAutoDirectMaxTaps` went from 256 to 512: `Auto` runs direct up to 512 taps
and FFT above, giving up the M2's small win at that one length to stop the loss
where CPU is scarcest. The source comment records the measurements, and
`test_ir_convolution` now pins the boundary at 512 (direct) and 513 (FFT); with
the old threshold the 512 check fails.

A side effect: `Auto` no longer sends an IR that fits entirely inside the direct
head to the FFT path. The shortest IR it now sends there, 513 taps, always has
at least one partition. The benchmark's 256-tap FFT row, which runs no
transform, is now purely a forced configuration; it stays so the ladder starts
in the same place on every machine.

## Spreading the multiplies (`b505422`)

The plan is [`plans/spread-transform.md`](plans/spread-transform.md). First,
what the transform callback is made of. `ir-study/spread_ab.cpp` times an
instrumented copy of `e2dc6bc` with a clock read between the phases; 8192 taps,
64 frames, share of the transform callback's median:

| | transform callback | forward FFT | multiplies | inverse FFT | overlap-add | rest |
|---|---:|---:|---:|---:|---:|---:|
| M2 | 20.9 µs | 22% | 40% | 21% | 3% | 13% |
| Pi 500 | 42.6 µs | 22% | 39% | 21% | 6% | 12% |
| Tinker Board | 265 µs | 16% | 32% | 16% | 22% | 15% |

The overlap-add costs the Tinker Board a fifth of the spike: its output ring
was indexed with a 64-bit `%`, which 32-bit ARM does as a library call, once
per output sample. Three commits followed:

1. [`78fdf96`](https://github.com/rikkus/AudioDSPTools/commit/78fdf96) tests
   that the FFT path's output is identical, bit for bit, whatever the callback
   sizes: fixed sizes from 1 to 1024, a mixed pattern and 200 random ones, at
   1000, 2049, 4096 and 8192 taps.
2. [`40e2329`](https://github.com/rikkus/AudioDSPTools/commit/40e2329) adds the
   overlap-add one output sample at a time from the current and previous
   inverse transforms, with no ring. Byte-identical output on every machine and
   on four real cabinet IRs; the transform callback 3% cheaper on the M2, 6% on
   the Pi 500 and 23% on the Tinker Board.
3. [`b505422`](https://github.com/rikkus/AudioDSPTools/commit/b505422) spreads
   the multiplies for partitions 1 to P-1 evenly across the callbacks between
   transforms, since their input spectra are already known. The transform
   callback keeps the forward FFT, partition 0 and the inverse. The order of
   additions into the accumulator changes, so output is no longer
   byte-identical to `e2dc6bc`: 145 dB below the signal in `spread_ab`,
   142-145 dBFS on the four cabinet IRs. The callback-size test still holds
   exactly.

Median cost at each position in the 8-callback cycle, 8192 taps, 64 frames, µs:

| | positions 1-7 | transform (8th) | mean |
|---|---:|---:|---:|
| M2, `e2dc6bc` | 2.67 | 21.25 | 5.00 |
| M2, `b505422` | 3.75 | 12.88 | 4.92 |
| Pi 500, `e2dc6bc` | 4.85 | 42.43 | 9.53 |
| Pi 500, `b505422` | 7.11 | 24.65 | 9.30 |
| Tinker Board, `e2dc6bc` | 39.1 | 258.1 | 65.8 |
| Tinker Board, `b505422` | 47.0 | 123.4 | 56.6 |

In the benchmark, at 8192 taps (64-frame blocks, the published runs at `b505422`
against those at `a09e360`; see [Records](#records)):

| | p99 before | p99 after | core% change |
|---|---:|---:|---:|
| M2 | 1.61% | 0.99% | -0.4% to -2.4% across the ladder |
| Pi 500 | 3.22% | 1.87% | -2% to -6% |
| Tinker Board | 21.74% | 9.89% | -18% to -31% |

At 32-frame blocks, the pedal case, the Tinker Board's p99 at 8192 taps went
from 40.0% of the deadline to 16.0%. Accuracy against upstream is unchanged at
136-138 dB. The Tinker Board's forced-FFT 256-tap row, which runs no transform
and so only paid the ring's `%`, went from 26% above direct to 2%; the M2's 7%
gap there did not move, so it is not the modulo.

What is left in the spike is the forward FFT, partition 0 and the inverse, all
of which need the block that has just completed: about 60% of the old
transform callback on the M2 and Pi 500, under half on the Tinker Board. Only
a smaller first FFT partition (non-uniform partitioning) would lower it
further.

## Memory

Heap held after building one mono `ImpulseResponse` from a 500 ms, 48 kHz IR and
calling `Reset(64)`, from `ir-study/heap_footprint.cpp` (macOS zone statistics,
so Eigen's allocations are counted too):

| Version | Heap held |
|---|---:|
| upstream `main` (direct) | 449 KiB |
| branch, direct path | 209 KiB |
| branch, FFT path at `ba646f4` | 479 KiB |
| branch, FFT path at `a09e360` and later | 386 KiB |

What the FFT path still holds that it does not need:

- three 8192-float buffers, 32 KiB each: the reversed kernel and the input
  history that only the direct path uses, and the original kernel, which is only
  read while the spectra are built. Freeing them would change no output and
  should bring the FFT path to about 290 KiB (estimated, not measured).
- the raw IR file, 94 KiB for a 500 ms file even though only 8192 samples are
  convolved. `ImpulseResponse` keeps it so it can rebuild the convolver when the
  sample rate changes, and upstream does the same.

**Decision: leave both alone for now.** Not worth the effort at the moment; the
first is a cleanup that can be picked up at any time, the second a behaviour
change.

This does not open the door to Pico-class boards. An RP2040 has 264 KiB of
SRAM in total and a Cortex-M0+ with no FPU. Even trimmed, one mono IR object
would be larger than all of its memory. The FFT path also needs about 16 bytes
per tap by construction (8 for the IR's spectra, 8 for the input history)
against about 8 for direct convolution: it is the choice that saves CPU where
memory is plentiful, not the one that saves memory. And the arithmetic does not
fit either: a transform callback at 8192 taps takes about 205 µs on the Tinker
Board at 1.8 GHz, and software floating point on a 133 MHz M0+ would take far
longer than the whole 1.33 ms callback. A board that small wants a much shorter
IR through a direct path.

## Records

Bencher reports behind the published tables, all 64-frame blocks:

| | Branch at | M2 | Pi 500 | Tinker Board |
|---|---|---|---|---|
| Before | `ba646f4` | `M2-20260921T014503Z` | [report](https://bencher.dev/perf/nambench/reports/01a0cb33-f63a-75f1-b387-a56758166161), `piv-20260922T220743Z` | [report](https://bencher.dev/perf/nambench/reports/01a0cb4e-8e3b-7f90-9c49-dd841b336c34), `tinkerboard-20260922T223553Z` |
| Half spectrum | `a09e360` | [report](https://bencher.dev/perf/nambench/reports/01a0cd74-9fea-76a1-b422-89b40229b09b), `M2-20260923T083527Z` | [report](https://bencher.dev/perf/nambench/reports/01a0cd72-8369-7280-97d0-e5bd48fb1f68), `piv-20260923T083526Z` | [report](https://bencher.dev/perf/nambench/reports/01a0cd7e-b03a-70f1-9fea-85e73875d876), `tinkerboard-20260923T084743Z` |
| Spread multiplies | `b505422` | [report](https://bencher.dev/perf/nam-ir/reports/01a0d4fd-09d6-7920-aa49-1041bb162e71), `M2-20260924T194343Z` | [report](https://bencher.dev/perf/nam-ir/reports/01a0d511-f686-7480-aff7-5dd93d00068a), `piv-20260924T200645Z` | [report](https://bencher.dev/perf/nam-ir/reports/01a0d53f-d194-7810-9cd3-a08c8ef97a66), `tinkerboard-20260924T205600Z` |

The first six runs are in the `nambench` Bencher project, the last three in
`nam-ir`, which has one plot per machine of p99 and core% at 8192 taps. The nine
JSON reports are in `benchmark-results/`, each named as above with
`-ir-block64.json` appended. `ir-study/docs_tables.py` prints the README's
tables from them; the README shows the last row.

Everything used to produce the rest of this file is in
[`ir-study/`](ir-study/README.md), with how to run it.
