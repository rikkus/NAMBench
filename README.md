# NAMBench

Compares NAM A2 WaveNet processing code, macOS and iOS.

| Variant | Repository | Code path |
|---|---|---|
| `a2_fast` | [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) | `a2_fast` (Eigen GEMM) — the reference |
| `a2_planar` | [Core PR #313](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/313) | `a2_planar` (planar NEON, both submodels) |
| `slim:*` | this repo, `Sources/SlimEngines` | experimental kernels for the 3-channel submodel |
| `full:*` | this repo, `Sources/FullEngines` | experimental kernels for the 8-channel submodel |

All are measured in one process, against one in-RAM model and one in-RAM audio
file.

## Build and run

**First, supply a capture.** The `.nam` files are not committed — they are other
people's work and this repository has no right to redistribute them. Put one A2
capture in `nam-files/`; [`nam-files/README.md`](nam-files/README.md) names the
one every published number was measured on, with its checksum. The input DI is
committed, so that is the only missing piece.

```bash
./Scripts/fetch-vendor.sh && xcodegen generate
```

Then either open `NAMBench.xcodeproj` and run the **NAMBench** scheme, or use the CLI:

```bash
xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release build
```

```bash
"$(xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release -showBuildSettings | awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')/nambench"
```

`nambench --help` lists the options. Reports are written as JSON and Markdown —
the Markdown is styled to drop straight into the fork's `benchmark_reports/`.

To run the slimmed-path kernel lab instead of the default A2 full comparison:

```bash
nambench --submodel narrowest --slim all
```

`nambench --list-slim` prints the kernel table; `--slim planar,widetile32`
selects a subset by name or index.

The full-path lab works the same way on the default (8-channel) submodel:

```bash
nambench --full all
```

`--list-full` prints its table, and `--full a2s8_h8,fu_s8_head` selects a subset.

An iOS **device** build needs a team: add
`CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=XXXXXXXXXX` to the `xcodebuild` call.
The Simulator builds unsigned, but its timings are meaningless — the report says
so when it detects one.

## Conformance

There is a second, portable build that does not measure anything. It takes the
same engine sources to Linux, Windows, Android and iOS, and to x86_64 as well as
AArch64, under GCC, Clang, Apple Clang and MSVC, and checks that they still
compile and still compute the same audio — engine routing, cross-variant parity,
and bit-identity for the kernels that claim to be verbatim ports.

```bash
cmake -S . -B build-conformance -DCMAKE_BUILD_TYPE=Release
cmake --build build-conformance --parallel 4
ctest --test-dir build-conformance --output-on-failure
```

It needs no capture: it loads `example_models/A2.nam` from the pinned upstream
checkout. It runs free on GitHub's hosted runners on every push, and no timing
is ever read from it. See [CONFORMANCE.md](CONFORMANCE.md).

## Benchmarking off the Mac

The same build also produces `nam_benchmark`, a port of `BenchCore`'s protocol
that runs where Xcode does not — which currently means a Raspberry Pi:

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
cmake --build build-benchmark --target nam_benchmark --parallel 4
./Scripts/run-benchmark.sh --cpu-set 0-3
```

The default line-up is `a2_fast` against `a2_planar`, on both A2 submodels:

| | M2 Air | Pi 500 (Cortex-A76) |
|---|---|---|
| A2 standard (8 ch) | 2.47x | 2.13x |
| A2 nano (3 ch) | 2.00x | 2.94x |

Bit-identical to `a2_fast` in every case. Results are tracked over time with
Bencher, one testbed per machine. See [BENCHMARKING.md](BENCHMARKING.md).

The same line-up now runs on 32-bit ARMv7 as well, on the RK3288 in the HeadRush
Core: **1.416x** at A2 standard and **1.470x** at A2 nano, bit-identical, at a
pedal's 32-frame blocks and a clock capped to 1416 MHz. Those are not a third
column of the table above, because that table is at 64-frame blocks and this
part gains most at 32 — putting them side by side would flatter the ARMv7 one by
a difference in the measurement rather than in the code.
[A32-PATH.md](A32-PATH.md) has the campaign behind them.

## Impulse responses

A second subject, measured by the same protocol on the same machines:
AudioDSPTools' impulse-response convolution, `main` against the
[`partitioned-ir`](https://github.com/rikkus/AudioDSPTools/tree/partitioned-ir)
branch, which keeps a direct FIR head for the first block — so latency stays at
zero — and moves the tail into partitioned FFT.

```bash
cmake --build build-benchmark --target nam_ir_benchmark --parallel 4
./Scripts/run-benchmark.sh --ir --build-dir build-benchmark
```

Direct convolution costs one multiply-add per tap per output sample, so it
grows with the length of the IR; the FFT path barely does. At 48 kHz, 8192 taps
is 170 ms of impulse response, and it is where AudioDSPTools truncates — the
longest the plugin can load.

M2 MacBook Air, 64-frame blocks, mono IR, all eighteen points accepted (spread
0.4-1.6%). `core%` is the average cost of keeping up with real time; `p99` is
the 99th-percentile block against its own deadline.

| taps | ms of IR | shipping | | partitioned FFT | | |
|---:|---:|---:|---:|---:|---:|---:|
| | | core% | p99 | core% | p99 | |
| 256 | 5.3 | 0.086% | 0.12% | 0.091% | 0.12% | 0.95x |
| 512 | 10.7 | 0.200% | 0.29% | **0.168%** | 0.40% | 1.19x |
| 1024 | 21.3 | 0.437% | 0.59% | **0.188%** | 0.48% | 2.33x |
| 2048 | 42.7 | 0.985% | 1.34% | **0.229%** | 0.65% | 4.29x |
| 4096 | 85.3 | 2.220% | 2.94% | **0.378%** | 1.75% | 5.87x |
| 8192 | 170.7 | 4.268% | 5.18% | **0.459%** | 2.26% | **9.30x** |

The branch's own direct path, measured alongside and not shown above, is
bit-identical to upstream at every length and within 1-7% of it in time. The FFT
path lands 136-138 dB below the signal.

From 1024 taps up, the FFT path is both cheaper on average *and* calmer in its
worst block than the code it replaces — at 8192 taps, 2.26% of a deadline
against 5.18%. The burstiness that `block_p99_percent` exists to catch is real,
but it only costs anything at 512 taps, where the p99 is worse than shipping
(0.40% against 0.29%) while the average is better.

**The 256-tap row is not measuring FFT.** At exactly 256 taps the whole impulse
response fits inside the direct head, so no transform runs and the subject is a
plain FIR — which is why it comes out bit-identical to upstream there, and 5%
slower, that 5% being the ring buffer it carries for a tail it does not have.
The driver says so on the line it prints, and `fftPartitions` in the report is
0. It is the real configuration at the shortest length `Auto` ever chooses the
FFT path for, so it is worth being able to see rather than worth hiding.

The crossover is therefore just above 256 taps, which is where
`kAutoDirectMaxTaps` already puts it. Whether that holds on a Cortex-A76 and a
Cortex-A17 is still to be measured.

It reports a worst-case block alongside the average, which `core_percent` cannot
give: partitioned FFT convolution does a whole partition's transform in one
callback, so an implementation can improve the average and worsen the worst
case, and a plugin that misses one callback clicks. See
[BENCHMARKING.md](BENCHMARKING.md#impulse-responses).

## What it measures, and why it is built this way

**Which submodel, chosen by shape not by position.** The `.nam` is a
`SlimmableContainer` holding a nano submodel (3 channels) and a full one
(8 channels). The shim picks by `max_value` — widest by default, narrowest for
`--submodel narrowest` — rather than by a fixed index, so a reordering in the
trainer cannot silently change which one is measured.

The full-path lab is dropped on the slimmed submodel, and the slim lab is
dropped on the full submodel: each lab's kernels are specialised for one
channel count and refuse the other. The channel count is read from the file,
not inferred from a flag.

**Each variant its own framework, not one binary.** Every variant is built
from its own pinned checkout into its own dynamic framework with
`-fvisibility=hidden`, exporting only a handful of prefixed C symbols. Apple's
two-level namespace keeps each framework bound to its own `nam::` internals.

**The slim lab is built from the same `vendor/upstream` tree as `a2_fast`,**
through the same target template with the same flags — the only differences are
`NB_ENABLE_SLIM_LAB` and the extra sources. That is what makes its kernel 0,
`baseline` (a verbatim port of `a2_fast`'s `Channels == 3` branch), a usable
control: it is bit-identical in output and lands within about 1% of `a2_fast`
in time, so anything a later kernel gains is the kernel and not the lab.

**The engine is asserted, never assumed.** Before anything is measured, the app
asks each build's own public shape detectors which engine the config will route
to, and refuses to run on a mismatch. This matters because a build falls
through to the *generic* engine for a shape it does not accept — a number that
was quietly generic would read as a catastrophic regression rather than a
harness bug.

**One shared Eigen.** `a2_fast` uses Eigen for its GEMM. Building `upstream`
and `planar` against different Eigen versions would put a dependency
difference straight into the measured result, so both compile against a single
`vendor/eigen` tree. The fetch script also asserts both repos pin the same
Eigen commit.

**Optimisation is forced on in Debug too.** A Debug build of the app still
compiles the engines at `-O3`, because a `-O0` benchmark is wrong in a way that
is easy not to notice.

## The protocol

1. **Parity check.** One full pass per variant, comparing the audio. A speed win
   paid for with a correctness regression is not a win.
2. **Warm up.** 5 seconds of full-file passes, discarded.
3. **Time.** 10 further seconds of passes, at least 8 samples.
4. **Accept or retry.** Take the tightest interval containing 70% of the
   samples. If it agrees to within 1%, report the mean of that interval and
   record what was discarded. Otherwise go back to step 2, up to 5 times, then
   report failure *with the samples* rather than a number nobody should trust.

**There is no idle gate.** An earlier version waited for the system to go quiet
before measuring. That does not work on macOS: Spotlight and other background
work are always running and cannot be paused, only disabled outright at the cost
of a full re-index afterwards — so the wait never ends. Interference is
therefore handled where it actually can be, in step 4, which discards the passes
it spoiled. Thermal state, Low Power Mode and system CPU are still *recorded* at
both ends of a run, so a rejected result can be explained after the fact.

### Why 70% and not 90%

Interference is *additive* — it can only ever make a pass slower — so the
contamination is one-sided. That is why the window is chosen by sliding a
fixed-width window over the sorted samples rather than trimming symmetric tails,
and it is why the fastest passes are the trustworthy ones.

90% was the right fraction while the benchmark waited for idle. Without that
wait, more than 10% of passes get hit. Measured on an M2 doing ordinary desktop
things, five consecutive attempts spread 1.4%, 1.2%, 4.5%, 9.4% and 27% at 90%,
but 0.38%, 0.46%, 0.47%, 1.0% and 13% at 70% — while the *fastest* pass in each
of those attempts varied by only 0.4% (412.5–414.1 ms). The signal is not noisy;
the tail is.

Keeping the fastest 70% therefore estimates the uncontended cost, and any
residual bias applies equally to both variants, so the ratio between them is
unaffected. Reports show the fastest pass next to the mean for the same reason.

`--accept-fraction` and `--accept-tolerance` adjust this. A run that fails is
telling you something real about the machine, so look at the recorded samples
before loosening either.

Runs on a `userInteractive` thread — at a lower QoS Apple silicon would schedule
it onto the efficiency cores, which would swamp everything being compared.
Model state is reset before every pass, always outside the timed region, and
denormals are flushed identically in both variants.

## Layout

```
Scripts/fetch-vendor.sh   pinned clones, shared Eigen, integrity checks
Scripts/eigen-order-probe/ what reduction order a2_fast's C=8 Eigen path uses
project.yml               XcodeGen spec — all build flags live here
Sources/Shim/             one C shim, compiled once per variant
Sources/SlimEngines/      experimental kernels for the 3-channel submodel
Sources/FullEngines/      experimental kernels for the 8-channel submodel
Sources/A32Engines/       experimental kernels for 32-bit ARMv7 (Cortex-A17)
Sources/BenchCore/        the protocol, shared by app and CLI
Sources/App/              SwiftUI, macOS + iOS
Sources/CLI/              headless macOS runner
charts/                   presentation graphics, and the script that builds them
benchmark-results/        the reports behind every published number
audio-input/              the input DI, bundled as a resource
nam-files/                where you put a capture — see its README
```

## The slimmed-path kernel lab

`a2_fast` runs the 3-channel submodel through a fully-unrolled *scalar* 3×3
GEMV. `Sources/SlimEngines/` holds candidate replacements, each measured by the
protocol above rather than by a throwaway harness. `benchmark-results/` has the
numbers and
[SLIMMED-PATH.md](SLIMMED-PATH.md) has the analysis: what won, what lost, and
why the ones that lost were worth writing.

Kernels are `nam::DSP` subclasses selected by index at runtime, sharing one
weight loader (`slim_common.h`) so the only thing that varies between two
measurements is the kernel. The planar family shares one templated
implementation (`slim_planar_kernel.h`) parameterised by an options struct, so
each candidate file reads as a diff against the lab's own `planar` kernel
(which is not the `a2_planar` variant — different thing, same idea).

## The full-path kernel lab

`Sources/FullEngines/` does the same job for the 8-channel submodel, where the
thing to beat is `a2_fast` itself. [FULL-PATH.md](FULL-PATH.md) has the
analysis: one question — can `a2_fast` at C=8 be beaten while staying
bit-identical to it — answered by the `a2*` family, which reproduces
`a2_fast`'s arithmetic and is **bit-identical to it**. That is possible because
`a2_fast`'s C=8 path, though it is Eigen, has a reduction order that is
knowable and reproducible — `Scripts/eigen-order-probe/` establishes exactly
which order, bit-for-bit, before any kernel was written.

Kernel 0 (`a2_baseline`) is a verbatim port of `a2_fast`'s `Channels == 8`
branch and has to land on `a2_fast`'s number — the control every other
full-lab kernel is validated against.

(An earlier phase of this lab also carried an `fu*` family reproducing the
arithmetic of a since-retired `fused` engine from a fork. That family, and
`fused` itself, have been removed from this repo — superseded by the planar
kernels now vendored as `vendor/planar` — so the full lab is scoped to the
`a2*` question alone. FULL-PATH.md keeps the historical numbers.)

## The ARMv7 kernel lab

`Sources/A32Engines/` asks the same question on a part none of the above can
reach: the **Rockchip RK3288** — four Cortex-A17 cores, ARMv7-A, 32-bit only —
which is the SoC in the HeadRush Core and Prime.
[A32-PATH.md](A32-PATH.md) has the analysis.

It is not a port of the other two labs' answers, because those answers do not
survive the trip. ARMv7 has 16 Q registers against AArch64's 32, no by-element
FMA, and a NEON datapath narrower than its own register width — so the tile
width that won on an M2 is roughly four times too wide here, and the switch that
mattered most there matters least. Both winners are nonetheless **bit-identical**
to `a2_fast` over a full render, which takes A2 standard from 78.5% to 57.8% of
one core at a pedal's 32-frame block size.

Two structural differences from the other labs, both forced by the target:

- **Measurement happens on real hardware over ssh.** There is no ARMv7 machine
  in the hosted CI fleet and no toolchain on the board, so `Scripts/a32-deploy.sh`
  cross-builds here and rsyncs the binaries, and `Scripts/run-benchmark.sh` runs
  on the board where it can own the governor and the thermal guard. Every
  measured run is clock-pinned to 1416 MHz, which is a soak-measured number, not
  a round one — see A32-PATH.md.
- **Compiler flags are a correctness setting.** On this target `-mfpu` decides
  the *arithmetic*: without `neon-vfpv4`, Eigen silently picks non-fused
  `vmlaq_f32` and `a2_fast`'s own C=8 output changes. The armhf CI entry exists
  for that regression specifically, and `Scripts/a32-codegen-check.sh` fails any
  build where a kernel claiming exactness emits `vmla`.
