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

64-frame blocks, mono IR, all eighteen points accepted on every machine, with
the branch at `a09e360`. `core%` is the average cost of keeping up with real
time; `p99` is the 99th-percentile block against its own deadline.

**M2 MacBook Air** (spread 0.5-2.9%)

| taps | ms of IR | shipping | | partitioned FFT | | |
|---:|---:|---:|---:|---:|---:|---:|
| | | core% | p99 | core% | p99 | |
| 256 | 5.3 | **0.087%** | 0.12% | 0.090% | 0.11% | 0.96x |
| 512 | 10.7 | 0.199% | 0.29% | **0.159%** | 0.37% | 1.25x |
| 1024 | 21.3 | 0.434% | 0.58% | **0.169%** | 0.41% | 2.56x |
| 2048 | 42.7 | 0.988% | 1.49% | **0.191%** | 0.49% | 5.18x |
| 4096 | 85.3 | 2.075% | 2.70% | **0.338%** | 1.28% | 6.14x |
| 8192 | 170.7 | 4.295% | 5.42% | **0.380%** | 1.61% | **11.29x** |

**Raspberry Pi 500, Cortex-A76** (spread 0.05-0.32%)

| taps | ms of IR | shipping | | partitioned FFT | | |
|---:|---:|---:|---:|---:|---:|---:|
| | | core% | p99 | core% | p99 | |
| 256 | 5.3 | **0.189%** | 0.21% | 0.201% | 0.22% | 0.94x |
| 512 | 10.7 | **0.357%** | 0.39% | 0.391% | 0.96% | 0.91x |
| 1024 | 21.3 | 0.687% | 0.73% | **0.413%** | 1.04% | 1.66x |
| 2048 | 42.7 | 1.357% | 1.51% | **0.455%** | 1.21% | 2.98x |
| 4096 | 85.3 | 2.721% | 2.88% | **0.645%** | 2.56% | 4.22x |
| 8192 | 170.7 | 6.630% | 6.91% | **0.727%** | 3.22% | **9.13x** |

**Tinker Board, Cortex-A17 at 1.416 GHz** (spread 0.07-0.45%)

| taps | ms of IR | shipping | | partitioned FFT | | |
|---:|---:|---:|---:|---:|---:|---:|
| | | core% | p99 | core% | p99 | |
| 256 | 5.3 | **1.428%** | 2.03% | 1.808% | 2.43% | 0.79x |
| 512 | 10.7 | **2.653%** | 3.24% | 3.207% | 8.01% | 0.83x |
| 1024 | 21.3 | 5.122% | 5.71% | **3.312%** | 8.44% | 1.55x |
| 2048 | 42.7 | 10.060% | 10.65% | **3.557%** | 9.41% | 2.83x |
| 4096 | 85.3 | 21.023% | 22.95% | **4.851%** | 18.18% | 4.33x |
| 8192 | 170.7 | 41.864% | 47.47% | **5.295%** | 21.74% | **7.91x** |

The branch's own direct path, measured alongside and not shown above, is
bit-identical to upstream at every length on every machine, and within about
3% of it in time. The FFT path lands 136-138 dB below the signal.

At 8192 taps the FFT path is both cheaper on average *and* calmer in its worst
block everywhere. On the Tinker Board that is the difference that matters:
shipping's slowest single block there reached 92% of its deadline, one bad
scheduling moment from a click, where the FFT path's never passed 28%. Below
that, the burstiness that `block_p99_percent` exists to catch is real: a
partition's transform lands in one callback, so on the ARM boards the FFT path's
p99 is *worse* than shipping's at 512 and 1024 taps, even where its average is
far better, and better from 2048 up. On the M2 it is only worse at 512.

**The 256-tap row is not measuring FFT.** At exactly 256 taps the whole impulse
response fits inside the direct head, so no transform runs and the subject is a
plain FIR — which is why it comes out bit-identical to upstream there, and
slower (3% on the M2, 6% on the Pi, 27% on the Tinker Board), that cost being the
ring buffer it carries for a tail it does not have. The driver says so on the
line it prints, and `fftPartitions` in the report is 0. `Auto` never picks the
FFT path at this length: it sends only IRs above 512 taps there, and those
always have at least one partition. So this row is what forcing FFT on a short
IR costs, kept so the ladder starts in the same place on every machine.

**The crossover depends on the machine, and `Auto` is set for the boards.** On
the M2 FFT wins from just above 256 taps. On both ARM boards it wins from
between 512 and 1024: at 512 taps it costs 10% more than direct on the Pi 500
and 21% more on the Tinker Board, with a p99 about two and a half times
shipping's. So since `e2dc6bc` the branch's `kAutoDirectMaxTaps` is 512, up from
256: `Auto` runs direct up to 512 taps and FFT above, giving up the M2's 1.25x at
that one length to stop the loss on both boards, where CPU is scarcest. The
tables force each path, so that change does not move any number in them.

**Only half of each spectrum is multiplied.** Audio and impulse responses are
real, so every spectrum the FFT path forms is conjugate-symmetric: bin
`N - k` is the conjugate of bin `k`, and the real-output inverse transform
only reads bins `0..N/2`. Until `a09e360` the branch multiplied all `N` bins for
every partition anyway. Stopping at `N/2` changes no output bit and makes the
transform callback cheaper, which is why it shows most in the p99: at 8192
taps, 2.26% to 1.61% on the M2, 4.48% to 3.22% on the Pi 500, 28.26% to 21.74%
on the Tinker Board. No length got slower on any machine.

It reports a worst-case block alongside the average, which `core_percent` cannot
give: partitioned FFT convolution does a whole partition's transform in one
callback, so an implementation can improve the average and worsen the worst
case, and a plugin that misses one callback clicks. See
[BENCHMARKING.md](BENCHMARKING.md#impulse-responses) for the protocol, and
[IR-PATH.md](IR-PATH.md) for the investigation around these numbers: the
per-callback traces, the half-spectrum change, the threshold, what trimming an
IR would cost, and what was left alone.

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
ir-study/                 the tools and pages behind IR-PATH.md
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

## The microcontroller experiment (for fun)

`fpi/` and `docs/fpi/` are a side project with no bearing on the benchmarks
above: an RP2350A board — dual Cortex-M33 at 150 MHz, 520 kB of SRAM — and the
question of what a NAM A2-Lite kernel actually costs on one core of it.

**It was for fun, and it is not a plan.** There is no intention of running NAM on
a microcontroller in reality; a laptop, a phone or a pedal SoC is the right home
for it, and that is what the rest of this repository measures. Nothing here should
be read as a proposal, and the experiment's own answer is that an MCU at stock
clock is not close.

It is kept because the *method* transferred: the ARMv7 lab above exists because
the answers do not survive a change of target, and this one is the same lesson
taken to a part three tiers smaller. What it produced:

- a hand-written Q15 kernel with SMLAD over packed tap pairs at **6,362
  cycles/sample, 204% of one core** against a 3,125-cycle budget — about 1.5×
  better per cycle than the tuned `a2_fast` on the same silicon;
- a two-core layer split at 118% of real time, with the finding that the
  second core's ~11% cost is not reachable by moving data, moving code, or doing
  less work per core;
- two rules that transferred intact from the M2 lab — planning beats recomputing,
  and instructions are not the currency (memory traffic is) — plus one that is
  its own: fp32 state is 2.8x slower than fixed point here, because the M33
  retires two integer MACs per instruction and one float MAC;
- and a set of hardware facts worth having: RP2350's SRAM is single-cycle and
  cannot be beaten, cache-as-SRAM is real but 60% slower in practice, and the
  XIP cache is not a second memory, only a second port.

[kernel-progress.png](docs/fpi/kernel-progress.png) plots the whole progression
in measurement order. [docs/fpi/README.md](docs/fpi/README.md) is the analysis and
[TOOLCHAIN.md](docs/fpi/TOOLCHAIN.md) is what had to be installed to get there.
