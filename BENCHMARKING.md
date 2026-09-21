# Benchmarking on hardware you own

[CONFORMANCE.md](CONFORMANCE.md) covers the half that runs in the cloud and reads
no timing. This is the other half.

Everything here runs on physical machines, one at a time, and the numbers are
kept. Nothing in this document will ever run on a GitHub-hosted runner.

## The drivers

| | `nambench` | `nam_benchmark` | `nam_ir_benchmark` |
|---|---|---|---|
| built by | Xcode, from `project.yml` | CMake, `-DNAMBENCH_BUILD_BENCHMARK=ON` | the same |
| runs on | macOS, iOS | macOS, Linux, Android | the same |
| measures | WaveNet engines | WaveNet engines | impulse-response convolution |
| status | canonical — every published WaveNet number came from it | exists because the Pi cannot run the other one | the only driver for its subject |

The first two are the same measurement by two routes; the third is a different
subject measured the same way. All three run one protocol, which since the IR
work lives in `Tools/nb_protocol.h` and is shared source rather than a
description two files follow: the WAV loader, the warm-up, the timing window,
the tightest-70% selection, the spread test, the retry-and-reject, and what is
recorded about the machine. A driver supplies only what it means to run one
pass. That is what makes a WaveNet number and an IR number safe to put on the
same Bencher project.

`nam_benchmark` is a port of `BenchCore`'s protocol, not an approximation of it:
same warm-up, same timing window, same tightest-70% selection, same
`(max - min) / min` spread test, same retry-and-reject, same defaults. Measured
back to back on one M2, with everything else equal, when this cross-check was
first done (against the now-retired `fused` engine, superseded since by the
planar kernels — the numbers are kept because the validation they establish,
that the two drivers agree, is unaffected by which engine happened to be the
faster one at the time):

| | `nambench` | `nam_benchmark` |
|---|---|---|
| `a2_fast` | 413.96 ms (min 411.13) | 415.00 ms (min 412.15) |
| `fused` | 220.27 ms (min 218.91) | 218.85 ms (min 217.01) |
| ratio | 1.879x | 1.896x |

Within 0.25% on `a2_fast`, 0.6% on `fused`, 0.9% on the ratio — on a machine
that was not idle. Getting there needed one non-obvious thing: `BenchCore` runs
its measurement on a `QOS_CLASS_USER_INTERACTIVE` thread, and until the portable
driver did the same, macOS put it on the efficiency cores and the two disagreed
by 1.4% in opposite directions.

Each testbed uses one WaveNet driver for life, so no history ever contains a
mixture. `nam_ir_benchmark` is the exception that proves the point: every
testbed uses it, including the Macs, because there is no Xcode equivalent — so
the IR series are the one set here measured by a single binary from a single
source on every machine.

The `driver:` field in `.github/workflows/benchmark.yml` names one of these two
*and* where it runs, which is why it has three values rather than two: `xcode`
is `nambench`; `portable` is `nam_benchmark`, built and measured on one machine;
and `cross` is the same `nam_benchmark`, built on one machine and measured on
another over ssh, which is the only way to reach a board that cannot compile it.

## Running one by hand

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
cmake --build build-benchmark --target nam_benchmark --parallel 4
./Scripts/run-benchmark.sh --cpu-set 0-3
```

`run-benchmark.sh` is a wrapper that puts the machine into a state worth
measuring, and puts it back afterwards — including after a Ctrl-C. It is not
optional on Linux:

- **CPU governor.** It sets `performance` and restores the previous setting on
  exit. `ondemand` ramps the clock *during* a pass, so a faster engine spends
  proportionally more of its pass at a low clock than a slower one. That
  compresses the exact ratio the benchmark exists to report, and it is a
  systematic bias, not noise the tightest-70% analysis can reject. If the
  governor cannot be set, the script **refuses to run** rather than quietly
  producing a biased number. `--no-governor` overrides that, deliberately.
- **Thermal.** Three sources are read either side of the run, because no one of
  them exists everywhere. The load-bearing one is **frequency residency**:
  cpufreq records how many jiffies were spent at each frequency, so diffing
  `time_in_state` across the run catches every excursion regardless of how brief
  it was. **Cooling state** names which thermal governor engaged, and
  **temperature** is logged for context. `vcgencmd get_throttled` is used in
  addition where it exists — it does not on the RK3288 board, which is why the
  residency diff replaced it as the primary check.

  Sampling `scaling_cur_freq` on a timer is *not* good enough, and this is
  measured rather than assumed: in a 25-minute soak on the Tinker Board a
  15-second sampler reported a floor of 1416 MHz while residency recorded 302
  jiffies at 1200 MHz — two frequency steps the sampler never once observed.
- **Frequency cap.** `--max-freq KHZ` caps `scaling_max_freq` for the run and
  restores it afterwards. `performance` asks for the top frequency; it does not
  stop the *thermal* governor taking it away again, and on a passively cooled
  board the two interact badly — the part heats past its passive trip and
  cpufreq spends the run hunting between steps. The residency check then
  correctly voids the run, after the whole measurement has been spent. Capping
  to a frequency a soak has shown the board sustains turns that into a run that
  simply does not throttle. Use `Scripts/a32-thermal-soak.sh` to find the
  frequency, and record it in the report note so the history stays
  interpretable. A stable clock matters far more than a high one here: the
  benchmark reports a *ratio* between engines, so absolute frequency barely
  affects the result while a clock that moves mid-run biases it.
- **Pinning.** `--cpu-set` hands the run a fixed set of cores so the scheduler
  cannot migrate it mid-pass onto a cold cache.

Add `--bmf out.json` to also write Bencher Metric Format.

## The machines

| testbed | machine | driver | notes |
|---|---|---|---|
| `m2-air` | M2 MacBook Air 15" | xcode | on macOS 27 beta |
| `m1-air` | M1 MacBook Air | xcode | |
| `pi500` | Raspberry Pi 500, Cortex-A76 | portable | Ubuntu 24.04 aarch64 |
| `tinker` | ASUS Tinker Board, RK3288, Cortex-A17 | cross | 32-bit ARMv7-A, built on the Pi |

**The Tinker Board is the odd one out, in three ways that all matter.** It is
the only 32-bit testbed, and until the planar gate was widened to ARMv7 its
line-up was `a2_fast` against the `a32` lab alone, because `a2_planar` compiled
to nothing there; it now measures all three. Its arithmetic depends on build
flags in a way no other testbed's does — at
`-mfpu=neon` Eigen silently computes `a2_fast`'s 8-channel path with non-fused
`vmlaq_f32`, so the reported `fpu` field must read `neon+fma` for a run to mean
anything. And it is cross-built on the Pi rather than compiled in place, because
an RK3288 with 2 GB of RAM building Eigen at `-O3` is not a sensible use of an
afternoon. Its numbers are not comparable with any other testbed's, only with
its own history.

### How the Tinker Board reaches Bencher

The `cross` driver, which is the `portable` one split across two machines.
`Scripts/a32-deploy.sh` cross-builds on the Pi, rsyncs the static binaries to
the board over ssh, and runs the same `Scripts/run-benchmark.sh` there — so the
governor, the frequency cap and the residency check are the ones every Linux
testbed uses, not a second description of them. The reports come back, and the
conversion and the upload happen on the build host.

That split is deliberate. The board cannot build the thing that measures it, and
it should not hold a Bencher API key or be asked what commit it is on; the build
host owns the compiler, the git history and the credential, and the board owns
nothing but the timing. `--bmf` on `a32-deploy.sh` is that seam — it converts
exactly the reports rsync says it transferred, so an earlier run's results
cannot be uploaded twice as a new one.

By hand:

```bash
./Scripts/track-benchmark.sh --board tib
```

which cross-builds, deploys, measures, converts, uploads as testbed `tinker`,
and syncs the thresholds and plots — the same steps the workflow takes, in the
same order, so runs made either way form one history. `--check` first verifies
the board answers ssh without a prompt as well as that Bencher is reachable.

**Every tinker run is clock-capped to 1416 MHz**, and `track-benchmark.sh`
defaults to it rather than leaving it to the caller. All four A17 cores share
one cpufreq policy, so `--cpu-set` pins the work but not the clock: a thermal
excursion charges whichever engine was running for it, which biases the ratio
the benchmark reports rather than adding noise the tightest-70% analysis can
reject. `run-benchmark.sh` correctly refuses to write BMF for such a run — after
the whole measurement has been spent. `--max-freq none` measures the board as
configured, deliberately. The number itself is soak-measured; see
[A32-PATH.md](A32-PATH.md).

**The tracked series is at 64-frame blocks**, like every other testbed's, so the
four `tinker` series line up structurally with the four on `m2-air`.
[A32-PATH.md](A32-PATH.md)'s headline for this board is at 32 frames instead,
because that is what a pedal runs and because this part gains most there —
which is exactly why the two must not share a series. Block size is not part of
a benchmark name, so a run made with `-- --block-size 32` would land on top of
the 64-frame history as though it were comparable. `bencher-report.py` says so
and names the fix; take it, and convert that run by hand with
`--prefix 'block32/'`.

**The Pi is a Pi 500, not a Pi 5.** Same BCM2712 and the same Cortex-A76, so as
a *core* it is the Pi 5 datapoint — but the 500 is passively cooled inside a
keyboard, and a Pi 5 with a fan will hold a boost clock for longer under
sustained load. The testbed is named `pi500` rather than `pi5` so that nobody
later reads a thermal difference as a code change.

## What is measured

The line-up is `a2_fast` — the reference — against `a2_planar`, the
NEON kernels proposed to Core in
[PR #313](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/313). The
planar checkout is pinned to the head of that draft PR rather than to a copy of
it, so the number and the proposal cannot drift apart.

`fused` has been retired from this repo entirely. The planar kernels
superseded it and cover the 3-channel submodel it never could — its detector
rejected any channel count that was not a multiple of four.

Both A2 submodels are measured on every run, as separate Bencher series:

| | channels | Bencher name |
|---|---|---|
| A2 standard | 8 | `a2_standard/a2_fast`, `a2_standard/a2_planar` |
| A2 nano | 3 | `a2_nano/a2_fast`, `a2_nano/a2_planar` |

They run inside one governor window and between one pair of thermal readings, so
the pair describes the same machine in the same state.

### Results

M2 MacBook Air, 64-frame blocks, against a real capture:

| | `a2_fast` | planar | |
|---|---:|---:|---:|
| A2 standard | 3.69% of a core | **1.49%** | 2.47x |
| A2 nano | 0.515% | **0.251%** | 2.00x |

which independently reproduces the PR's own table (2.43x and 2.01x) on different
audio, through a different harness.

Raspberry Pi 500, Cortex-A76, GCC 13:

| | `a2_fast` | planar | |
|---|---:|---:|---:|
| A2 standard | 10.7% of a core | **5.02%** | 2.13x |
| A2 nano | 2.56% | **0.87%** | 2.94x |

Bit-identical in every one of those runs — `max|diff|` exactly zero, not a
tolerance.

The Pi's profile is not the Mac's. A2 nano wins far more there (2.94x against
2.00x) while A2 standard wins rather less (2.13x against 2.47x). Two different
machines, and only one of them was available to the person who wrote the
kernels.

The Pi is also a far quieter instrument than either Mac: spread across the
accepted samples was 0.04–0.17%, where the M2 routinely lands between 1% and 4%
and sometimes fails all five attempts. A dedicated machine with nothing else
running is worth more than a fast one.

## Impulse responses

A second subject, measured by the same protocol on the same machines:
AudioDSPTools' impulse-response convolution, `main` against the
`partitioned-ir` branch.

Upstream convolves directly — one multiply-add per tap, per output sample — so
its cost is linear in the length of the IR. The branch keeps a direct FIR head
covering the first block, which is what holds latency at zero, and moves
everything behind it into uniformly partitioned FFT convolution, whose cost
barely grows with length at all. The longer the IR, the wider the gap.

"Taps" is that length in samples: at 48 kHz, 512 taps is about 11 ms of
impulse response and 8192 is 170 ms. 8192 is where the ladder stops because
that is where AudioDSPTools truncates, so it is the longest IR the plugin can
load, not an arbitrary ceiling.

### Results

M2 MacBook Air, 64-frame blocks, mono IR. `core%` is the average cost; `p99` is
the 99th-percentile block against its own deadline.

| taps | ms of IR | shipping | | partitioned FFT | | |
|---:|---:|---:|---:|---:|---:|---:|
| | | core% | p99 | core% | p99 | |
| 256 | 5.3 | 0.092% | 0.14% | 0.093% | 0.12% | 0.99x |
| 512 | 10.7 | 0.204% | 0.33% | **0.171%** | 0.43% | 1.19x |
| 1024 | 21.3 | 0.450% | 0.58% | **0.193%** | 0.52% | 2.33x |
| 2048 | 42.7 | 1.009% | 1.18% | **0.237%** | 0.69% | 4.26x |
| 4096 | 85.3 | 2.151% | 2.84% | *rejected* | | |
| 8192 | 170.7 | 4.373% | 5.11% | **0.470%** | 2.31% | **9.31x** |

The branch's own direct path is bit-identical to upstream at every length, and
within 1% of it in time. The FFT path is around 136 dB below the signal.

At 8192 taps it is not only nine times cheaper on average but *calmer* in its
worst block than the code it replaces — 2.31% of a deadline against 5.11%. The
crossover is around 512 taps, which is roughly where `Auto` already switches
(`kAutoDirectMaxTaps = 256`).

Two points are missing because the protocol rejected them: this laptop was not
quiet enough to measure a 10 ms pass to within 3%. They are gaps rather than
numbers nobody should trust. The Pi and the Tinker Board are still to run.

### Three subjects, not two

| Bencher name | what it is |
|---|---|
| `ir_<taps>/adt_upstream` | what the plugin ships |
| `ir_<taps>/adt_partitioned_direct` | the branch's direct path, forced |
| `ir_<taps>/adt_partitioned_fft` | the branch's FFT path, forced |

The middle one is there because the branch's direct path is a *rewrite* of
upstream's, not the same code — it manages its own history buffer so that the
FFT path can share the arrangement. `Auto` selects it for short IRs, so if it
had regressed, every short IR would carry that regression and no amount of
comparing the FFT path against upstream would reveal it. It is measured, and it
comes out bit-identical to upstream at every length: `max|diff|` exactly zero,
not a tolerance.

The FFT path is not bit-identical and cannot be — it is a different order of
arithmetic. It lands around 136 dB below the signal, which is checked as an
accuracy floor before anything is timed, and a point that fails it is not
reported at all.

### Two measures

`core_percent` alone would be misleading here in a way it is not for the
WaveNet engines. Partitioned FFT convolution is bursty by construction: a whole
partition's transform lands in one callback, so the work is spread unevenly
across blocks in a way a direct convolution's never is. An implementation can
lower the average and raise the worst case, and a plugin that misses one
callback clicks however good its average was.

So the IR benchmark also reports **`block_p99_percent`**: the 99th-percentile
block, as a percentage of that block's own real-time budget. 100% is a callback
that used its entire deadline. The shim reads the clock at every block boundary,
so the per-block times and the total they are compared against come from the
same reads and cannot disagree, and the pooled blocks are exactly the blocks of
the passes the window accepted.

A p99 rather than a maximum. A maximum over a million blocks is a measurement of
the operating system — one preemption or one page fault and the number stops
being about the code. The maximum is in the JSON for diagnosis; the p99 is what
Bencher tracks.

### The impulse response is generated

There is no IR asset in this repository. What a convolution costs is set by how
many taps it has and not by what is in them — both implementations touch every
tap of every block regardless of their values — so the shim generates one from
an integer LCG and a multiplicative decay. No libm call, so every platform
produces the same floats; no file, so nothing can differ between two runs being
compared; and nobody's cabinet measurement is being redistributed.

### Running one

```bash
Scripts/track-benchmark.sh --ir
```

or, without uploading:

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
cmake --build build-benchmark --target nam_ir_benchmark --parallel 4
Scripts/run-benchmark.sh --ir --build-dir build-benchmark
```

### A rejected point is a gap, not a number

An IR run is eighteen independent series. If the protocol rejects one of
them — the machine was too noisy to measure a 10 ms pass to within 3% — that
point is omitted from the upload, named on stderr with its reason, and the other
seventeen are uploaded. Its series gets a gap where a run that could not be
trusted would otherwise have left something that could not be trusted either.

This is deliberately *not* what the WaveNet run does. There the line-up is two
variants and what is wanted from it is the ratio between them, so half of it is
worth little and one rejection still refuses the whole upload.

Both refuse everything if the residency check says the clock moved during the
run. That is not one subject being hard to measure; it is every number in the
run describing a machine that was not one machine.

`--taps` sets the ladder (default `256,512,1024,2048,4096,8192`) and `--blocks`
the block sizes. Each block size gets its own report, because block size is not
part of a Bencher benchmark name and two of them in one upload would overwrite
each other; `--bmf` with more than one is refused up front rather than after
the measurement has been spent.

## The planar gate

`a2_planar.h` defines `NAM_A2_PLANAR` wherever `__aarch64__` is defined, and on
32-bit ARM that also has `__ARM_NEON` and `__ARM_FEATURE_FMA`. The FMA half is
not belt-and-braces: without it Eigen computes `a2_fast`'s own 8-channel path
with non-fused `vmlaq_f32`, so the reference the kernels are compared against is
no longer the reference — which is the same trap the `-mfpu` note above
describes, reached from the other side.

Everywhere else the translation unit compiles to an object with no symbols and
the variant is plain `a2_fast` — which the driver reports honestly, and then
**refuses to run**, because measuring the reference against itself would land
within noise and read as the kernels achieving nothing:

```
error: the planar variant routed to a2_fast, not to the planar kernels.
```

The gate was `__APPLE__ && __aarch64__` when the kernels were first proposed,
and NAMBench carried a `NAMBENCH_FORCE_A2_PLANAR` option to open it. Both are
gone: the gate was widened on the evidence that option was added to collect.
Nothing needs forcing, and the Pi needs no special build.

`__aarch64__` specifically, rather than a spelling that would also catch MSVC's
`_M_ARM64`. That is the one part of the old gate worth keeping — MSVC at
`/fp:precise` does not contract `a*b+c` into an FMA, so the reference branch it
would be compared against computes something else and bit-identity would not
hold. clang-cl on ARM64 defines `__aarch64__` and is unaffected.

Conformance holds `a2_planar` to exact bit-identity with `a2_fast`, so the claim
keeps being tested rather than remembered — on Apple Silicon, on the Pi, and on
GitHub's free Linux arm64 runners under both GCC and Clang on every push.

The tile widths remain M2 measurements. They affect speed only, never output,
and the Cortex-A76's rather different profile suggests re-tuning them per part
would be worth someone's afternoon.

## Continuous tracking with Bencher

[`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) runs on
`workflow_dispatch` only — these runners are somebody's laptop — with a global
`concurrency` group so two runs can never share a machine.

Each machine is its own Bencher testbed, so its history is only ever compared
with itself. An M2 and a Cortex-A76 are not two samples of one population.

One measure is reported per variant: **`core_percent`** — what a single NAM
instance costs, as a percentage of one CPU core, while keeping up with real-time
audio.

```
core_percent = (meanMs / 1000) / audioSeconds x 100
```

Real-time audio is constrained by how much work the CPU can do inside each
callback, so that fraction is the number worth tracking. The whole-file render
time it is derived from is not: the test file's length is arbitrary, so a
duration in milliseconds says as much about the signal as about the engine.
Dividing it out leaves a figure invariant to the length of the input and to the
sample rate.

Of one **core**, not of the whole CPU package. `process()` is single-threaded,
so an instance can never use more than one core; dividing by the core count
would give a smaller number that hides how close the audio thread is to its
deadline, and on a heterogeneous part like an M2 it would pretend four
efficiency cores are interchangeable with four performance cores. Keeping the
denominator at one core also keeps the useful inverse honest — `100 /
core_percent` is how many instances fit on one core, under ideal conditions with
no contention, no thermal limit and no headroom.

Two things it does not tell you. It is a mean, so it says nothing about whether a
block misses its deadline — that depends on the worst case, and the protocol
deliberately discards slow outliers as interference. And it is specific to the
block size the run used, since smaller blocks cost more per sample; the
converter warns if a run used anything other than 64 frames, because block size
is not part of the benchmark name.

`latency` and `throughput` were reported until this became the measure. They
were the same information in less useful forms — a render time that depended on
the test file, and a real-time factor that is exactly `100 / core_percent`. Both
were also being fed in the wrong units: Bencher's built-in `latency` is
nanoseconds and its `throughput` is operations per second, so the dashboard was
labelling milliseconds as nanoseconds.

### Hardware counters

`--counters default` brackets **each timed pass** with PMU counters and puts the
results in the report next to the timings; `--list-counters` prints what this
machine's PMU offers. This is not `perf stat`, and not only because Ubuntu ships
no perf binary for 32-bit ARM: `perf stat` counts a whole process, and most of
this process is model loading, wav decoding, parity rendering and warm-up. The
counters are reduced over the passes the tightest-70% window accepted, using the
same median, so they describe the same work `core_percent` does.

Two caveats the tool reports rather than hides. A PMU has a fixed number of
programmable counters — six on the Cortex-A17, and `--list-counters` probes for
the number — and asking for more makes the kernel time-slice them, so every
value becomes an estimate scaled up from part of each pass. And sysfs advertises
the generic architectural event set rather than what the core implements: on
Cortex-A17, `ld_retired` and `st_retired` open successfully and count nothing,
so `mem_access` is the substitute. An event that reads zero on every pass is
flagged as such.

Counters need `kernel.perf_event_paranoid` at 1 or lower. Where it is higher the
run continues without them and says why.

A variant that failed its agreement threshold is **omitted entirely**, never
reported as zero. `Scripts/bencher-report.py` refuses to emit an empty run for
the same reason: an empty result set uploads as "nothing regressed".

### What a series is

A benchmark name is `<model>/<kernel>` — `a2_standard/a2_planar`,
`a2_nano/a2_fast` — because Bencher gives a series exactly one free-text
dimension and there are two things to say with it. The machine is deliberately
not in the name: that is the *testbed* dimension, and keeping it separate is
what stops an M2's history being averaged with a Pi's.

So four names across three testbeds are twelve independent series. Both halves
are spelled the way the Core code path spells them, underscores and all, so a
label on the dashboard and a symbol in the source are the same word.

The IR benchmark keeps the same shape with the impulse-response length where
the model goes — `ir_8192/adt_partitioned_fft`. Length belongs in the name for
the same reason a submodel does: it is the independent variable of that
benchmark, and two lengths are no more comparable to each other than A2
standard is to A2 nano.

### Thresholds, plots, and `bencher-sync.py`

`bencher run` uploads metrics and nothing else. A chart has to be pinned before
anyone can see it, and a threshold has to exist before a regression can raise an
alert — and both are per-dimension, so the newest kernel or the newest machine
is otherwise the one thing nobody is watching.

[`Scripts/bencher-sync.py`](Scripts/bencher-sync.py) derives both from what is
in the project, and runs after every upload from both paths. It is idempotent
and it deletes nothing.

```bash
BENCHER_PROJECT=nambench Scripts/bencher-sync.py --dry-run
```

**Thresholds** — one per (branch, testbed, measure). That is the whole of
Bencher's threshold scope; it is not per benchmark, and does not need to be. One
model per machine is evaluated against each benchmark on that machine
separately, so a single t-test alerts on `a2_nano/a2_planar` regressing while
`a2_standard/a2_fast` holds. Per kernel, per model, per testbed — with one model
to maintain rather than twelve.

The model is a t-test over a rolling 64-sample window, with **both** boundaries
at 0.98 and nothing alerting until there are ten samples to compare against.
Slower is the regression anyone expects. The lower boundary catches the more
dangerous case: a result that is suddenly and impossibly *fast* usually means
the kernel stopped doing the work — a model that failed to load, a routing
change that fell through to a smaller path, a loop the compiler found dead — and
without it that arrives as a win and gets defended.

There are two measures, so two thresholds per testbed. `block_p99_percent` gets
the same t-test one boundary wider, at 0.99. A tail statistic drawn from
millions of blocks moves with a scheduling decision in a way a mean over the
same passes does not, and the protocol's window selection throws out a noisy
*pass* but cannot throw out a noisy block inside an accepted one. A threshold
that cries wolf gets muted, which is worse than not having one.

Thresholds are installed for every testbed as soon as it exists, rather than
being carried along by `bencher run --threshold-*`. Those flags come with
`--thresholds-reset`, so two upload paths differing by one argument take turns
redefining the model and the one that alerts is whichever ran last. That is why
neither the workflow nor `track-benchmark.sh` passes them any more, while both
still pass `--error-on-alert`.

**Plots** — one per (measure, testbed, model), carrying every kernel of that
model.

A pinned plot is a fixed list of UUIDs and cannot follow new data by itself,
which is the reason for the sync rather than a one-off. Grouping this way puts
`a2_fast` and `a2_planar` on one axis at one scale, which is the comparison this
project exists to make. Machines stay apart because the faster one would flatten
the other against the baseline, and `a2_standard` stays away from `a2_nano`
because they differ by roughly seven times. A `core_percent` plot draws the
error bars — the min and max of the accepted set, so you can see how noisy the
machine was before believing a step in the line. A `block_p99_percent` plot does
not: a percentile has no interval around it to draw, and the median below it and
the maximum above it are different statistics rather than uncertainty in this
one. No plot draws the boundary limits: Bencher marks those with a warning
triangle at every point, which reads as a problem when it is only where the
threshold sits.

A plot is recognised by its measure, testbed and model, not its title, so one
renamed on the dashboard keeps its name and is still updated in place. Indices
are one shared 0..64 space across every measure, which is why the sync assigns
them all in one pass and in a fixed measure order: sorting by name would have
put `block_p99_percent` ahead of `core_percent` and shifted every existing plot
by one the first time it ran.

Plots follow `main` only. They are project-wide and capped at 64, so a set per
branch would fill the dashboard with charts nobody asked for and nobody deletes;
branch data is still there to be plotted ad hoc. Thresholds do follow the
branch, so a run on one is still checked.

### By hand, without a runner

`Scripts/track-benchmark.sh` does everything the workflow does except the
runner: it picks the same driver and the same testbed name, so runs made this
way and runs made later by a self-hosted runner form **one continuous history**
rather than two forked ones. It measures this machine, unless `--board HOST`
tells it to cross-build for an ARMv7 board and measure that one instead — see
[How the Tinker Board reaches Bencher](#how-the-tinker-board-reaches-bencher).

```bash
export BENCHER_API_KEY="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
export BENCHER_PROJECT=nambench
```

Or put them in a **`.env` beside the checkout**, which is what the machines that
upload regularly do — the Pi is driven over ssh, and an export forgotten in a
non-interactive shell used to be discovered at the end of a ten-minute
measurement rather than the start of one:

```
BENCHER_API_KEY=bencher_user_...
BENCHER_PROJECT=nambench
```

Both `track-benchmark.sh` and `bencher-sync.py` read it, from the working
directory and from the checkout. It is parsed rather than sourced, and only
`BENCHER_*` is taken from it: sourcing runs whatever is in the file as shell,
and a stray `PATH` or `LD_PRELOAD` left in a file nobody reads any more would
not fail here, it would quietly produce a number. Anything already exported
wins. `.env` is gitignored.

### Check before you measure

```bash
./Scripts/track-benchmark.sh --check
```

Resolves the testbed, the branch and the credential, then makes one real read
against the Bencher API — `bencher project view` — and stops. Measures nothing,
uploads nothing.

```
==> detected testbed: pi500
==> bencher: nambench readable (public)
==> check passed
  testbed  pi500
  project  nambench
  branch   main @ 75660a59201d
  key      /home/rik/NAMBench/.env
  driver   nam_benchmark (portable)
```

A normal run does the same check first, before the build. A run is minutes of a
machine held deliberately quiet, and an expired key, a renamed project or a
missing CLI found at the upload throws all of it away; one GET beforehand costs
nothing. `.github/workflows/benchmark.yml` checks the same way, before its
build, for the same reason. `--dry-run` skips it — that is the one mode meant to
work with no credential at all.

The CLI installs itself into `~/.cargo/bin`, which a non-interactive ssh shell
does not have on `PATH`. The script appends that directory if `bencher` is not
otherwise found — appended, never prepended, because putting a cargo bin
directory ahead of the system one could substitute a different compiler into the
build of the thing being timed.

**`BENCHER_API_KEY`, not `BENCHER_API_TOKEN`.** The CLI reserves `--token` for
JWTs and refuses an API key given to it:

```
error: invalid value (redacted) for '--token <TOKEN>': You supplied a Bencher
API key to `--token`/`BENCHER_API_TOKEN`. Use `--key`/`BENCHER_API_KEY` instead.
```

The script accepts either variable and translates, but it also *unsets* the old
one before calling `bencher` — the CLI reads that variable itself, so a stale
`BENCHER_API_TOKEN` still exported in your shell would sink the run whatever
flag the script passed.

Then, on the M2 Air:

```bash
./Scripts/track-benchmark.sh
```

and on the Pi:

```bash
ssh piv "cd ~/NAMBench && ./Scripts/track-benchmark.sh --cpu-set 0-3 \
    --branch '$(git rev-parse --abbrev-ref HEAD)' \
    --hash '$(git rev-parse HEAD)'"
```

No `BENCHER_API_KEY` and no `PATH` here any more: the key comes from the Pi's own
`.env`, and the script finds `~/.cargo/bin` itself. Passing the key through an
ssh command line put it in the process table of both machines for the duration
of the run, which was the one thing everything else about it was arranged to
avoid. Check first, from here:

```bash
ssh piv "cd ~/NAMBench && ./Scripts/track-benchmark.sh --check"
```

**Pass `--branch` and `--hash` on the Pi.** Its copy is rsync'd without `.git`,
so it cannot read either, and would otherwise record against `main` with no
commit — while the laptop records against whatever branch it is really on. The
two machines' results would then sit in different Bencher branches and stop
being comparable, which is the one thing this arrangement exists to prevent. The
script warns when it has to guess.

The script works out which machine it is on — `m2-air`, `m1-air`, `pi500` — and
refuses to guess if it cannot. That matters more than it looks: a wrong testbed
name is not an error, because Bencher creates testbeds on demand, so it quietly
starts a *second* history for one machine.

Useful flags:

| | |
|---|---|
| `--check` | check this machine can reach Bencher, then stop |
| `--dry-run` | measure and convert, print the BMF, upload nothing |
| `--fail-on-alert` | exit non-zero when Bencher raises one |
| `--no-sync` | skip the threshold and plot sync after uploading |
| `--submodel narrowest` | measure the 3-channel path instead |
| `--timing-seconds N` | shorter window while you are setting things up |

Run `--dry-run` first. It exercises the whole path bar the upload.

There is no `--thresholds` flag any more, and nothing to remember to add once a
machine has a history: the sync installs the model straight away, and its
minimum sample size of ten means it sits inert until there are ten runs for the
t-test to describe.

**Commit before recording anything you care about.** A result is attributed to a
commit, and the script warns if the working tree does not match it — an
attribution that is wrong survives in the history long after the working copy is
gone.

The token is read into the process environment and never passed as an argument:
process arguments are readable by every other process on the machine for as long
as the command runs. Set `BENCHER_OP_REF` to the `op://` reference instead of
exporting the key, and the script will read it from 1Password itself.

### Setup, once

The Bencher CLI is needed on each machine:

```bash
curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh
```

It installs to `~/.cargo/bin`, which is not on a non-interactive ssh session's
PATH — hence the explicit `PATH=` in the Pi command above.

The project slug is whatever you want; `bencher run` creates the project if it
does not exist. To use one that already does:

```bash
bencher organization list --format json | jq -r '.[].slug'
bencher project list --format json | jq -r '.[].slug'
```

### For the GitHub workflow

The API key is yours to install — I have not touched it, and it should not
pass through a terminal, a file in the repo, or a chat window. With the
1Password CLI, it never becomes visible at all:

```bash
op read "op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential" | gh secret set BENCHER_API_KEY --repo rikkus/NAMBench
```

If the field is not called `credential`, this lists the item's fields without
printing their values:

```bash
op item get f2x4p5ymikp25e4hlocah2zexe --vault y6stxkiynyni73l5xoygigzswi --format json | jq '.fields[].label'
```

Once the first upload has created the measure, give it units — Bencher will
otherwise label the axis with nothing, and the whole point of the name is that
nobody reads it as a share of the whole chip:

```bash
bencher measure update --units '% of one core' "$BENCHER_PROJECT" core-percent
```

Then the project slug, which is not a secret:

```bash
gh variable set BENCHER_PROJECT --repo rikkus/NAMBench --body "your-bencher-project-slug"
```

For running by hand on the Pi, put it in the environment rather than in a file:

```bash
export BENCHER_API_KEY="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
```

### Registering a runner

On each machine, from **Settings → Actions → Runners → New self-hosted runner**
in the repository, following GitHub's generated commands. When it asks for
labels, add the one this workflow expects — `nambench-m2air`, `nambench-m1air`
or `nambench-pi500` — alongside the defaults it fills in for you.

**There is no runner on the Tinker Board**, and there is not going to be: the
`tinker` job runs on `nambench-pi500` and reaches the board over ssh. So the Pi
carries both testbeds, and the workflow's `max-parallel: 1` is what keeps a
cross-build from landing in the middle of the Pi measuring itself.

The Pi wants two more things:

```bash
sudo apt-get install -y cmake util-linux
```

and passwordless sudo for the governor, which it already has. Without it,
`run-benchmark.sh` refuses to run rather than producing a biased number.

For the `tinker` job it wants two more again — the armhf cross toolchain, and
key-based ssh to the board under the runner's own user:

```bash
sudo apt-get install -y g++-arm-linux-gnueabihf rsync
ssh-copy-id tib
```

`tib` is the host name the matrix passes as `board`; give it a `Host` entry in
the runner user's `~/.ssh/config` if it needs a user, a port or an address. The
measurement is driven over one non-interactive session, so a board that prompts
for a password fails the run — `track-benchmark.sh --board tib --check` says so
in ten seconds rather than after a cross build.

Install the runner as a service so it survives a reboot:

```bash
sudo ./svc.sh install && sudo ./svc.sh start
```

## What is not automated yet

**The iPhones.** An iOS *device* build needs a development team and signing, and
the results come back through the app's Documents directory rather than stdout.
The `NAMBench` app already writes the same report format and already exposes it
via the Files app, so the path is: run the app on the device, retrieve the JSON,
then

```bash
python3 Scripts/bencher-report.py <report>.json --output bmf.json
bencher run --project "$BENCHER_PROJECT" --testbed iphone-17 --adapter json --file bmf.json
```

which puts a hand-collected iPhone run into the same history as the automated
ones. Automating it needs a Mac runner driving `xcodebuild test` against a
tethered device, which is a bigger piece of work than the rest of this document
combined.

**Android.** Cross-compiles today (see CONFORMANCE.md) but is not measured.
`nam_benchmark` is a static command-line binary, so `adb push` and run works on a
rooted device with locked clocks; on an unrooted one the clocks are not yours to
control and the numbers would not be worth keeping.

**The 2015 Intel MacBook Pro.** Deliberately left alone. The planar kernels
gate on AArch64 (and on 32-bit ARM with NEON and FMA), so an Intel Mac can only
measure `a2_fast` against the generic engine — which CI already does, on
x86_64, for free, without touching a machine that has music-production work on
it.
