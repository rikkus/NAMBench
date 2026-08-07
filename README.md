# NAMBench

Compares NAM A2 WaveNet processing code, macOS and iOS.

| Variant | Repository | Code path |
|---|---|---|
| `upstream` | [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) | `a2_fast` (Eigen GEMM) |
| `fused` | [rikkus/OptimisationWorkOnNeuralAmpModelerCore](https://github.com/rikkus/OptimisationWorkOnNeuralAmpModelerCore) | `fused` (hand-written NEON) |
| `slim:*` | this repo, `Sources/SlimEngines` | experimental kernels for the 3-channel submodel |

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

An iOS **device** build needs a team: add
`CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=XXXXXXXXXX` to the `xcodebuild` call.
The Simulator builds unsigned, but its timings are meaningless — the report says
so when it detects one.

## What it measures, and why it is built this way

**Which submodel, chosen by shape not by position.** The `.nam` is a
`SlimmableContainer` holding a nano submodel (3 channels) and a full one
(8 channels). The shim picks by `max_value` — widest by default, narrowest for
`--submodel narrowest` — rather than by a fixed index, so a reordering in the
trainer cannot silently change which one is measured.

On the slimmed submodel the `fused` variant is dropped from the line-up
automatically. Its detector rejects any channel count that is not a multiple of
four, and under `ScopedEnginePrefer(FusedNeon)` a rejected shape falls through
to the *generic* engine rather than to `a2_fast` — so including it would either
abort on the engine assertion or measure the wrong thing. The channel count is
read from the file, not inferred from the flag.

**Three frameworks, not one binary.** The fork is a superset of upstream and
even has an `EnginePrefer` switch, but it has no value that selects `a2_fast`:
fused is tested first, so for an 8-channel model `Auto` and `FusedNeon` both
give fused and `Generic` gives neither. So each variant is built from its own
pinned checkout into its own dynamic framework with `-fvisibility=hidden`,
exporting only a handful of prefixed C symbols. Apple's two-level namespace
keeps each framework bound to its own `nam::` internals.

**The slim lab is a third framework built from the same tree as `upstream`,**
through the same target template with the same flags — the only differences are
`NB_ENABLE_SLIM_LAB` and the extra sources. That is what makes its kernel 0,
`baseline` (a verbatim port of `a2_fast`'s `Channels == 3` branch), a usable
control: it is bit-identical in output and lands within about 1% of `upstream`
in time, so anything a later kernel gains is the kernel and not the lab.

**The engine is asserted, never assumed.** Before anything is measured, the app
asks each build's own public shape detectors which engine the config will route
to, and refuses to run on a mismatch. This matters because the fork falls
through to the *generic* engine for shapes fused does not accept — a "fused"
number that was quietly generic would read as a catastrophic regression rather
than a harness bug.

**One shared Eigen.** `a2_fast` uses Eigen for its GEMM; `fused` is pure NEON.
Building the two against different Eigen versions would put a dependency
difference straight into the measured result, so both compile against a single
`vendor/eigen` tree. The fetch script also asserts both repos pin the same Eigen
commit, and that their `a2_fast` sources are byte-identical.

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
project.yml               XcodeGen spec — all build flags live here
Sources/Shim/             one C shim, compiled once per variant
Sources/SlimEngines/      experimental kernels for the 3-channel submodel
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
GEMV, and `fused` cannot take over because `parse_spec` rejects any channel
count that is not a multiple of four. `Sources/SlimEngines/` holds candidate
replacements, each measured by the protocol above rather than by a throwaway
harness. `benchmark-results/` has the numbers and
[SLIMMED-PATH.md](SLIMMED-PATH.md) has the analysis: what won, what lost, and
why the ones that lost were worth writing.

Kernels are `nam::DSP` subclasses selected by index at runtime, sharing one
weight loader (`slim_common.h`) so the only thing that varies between two
measurements is the kernel. The planar family shares one templated
implementation (`slim_planar_kernel.h`) parameterised by an options struct, so
each candidate file reads as a diff against `planar`.
