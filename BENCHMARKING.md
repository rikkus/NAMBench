# Benchmarking on hardware you own

[CONFORMANCE.md](CONFORMANCE.md) covers the half that runs in the cloud and reads
no timing. This is the other half.

Everything here runs on physical machines, one at a time, and the numbers are
kept. Nothing in this document will ever run on a GitHub-hosted runner.

## The two drivers

| | `nambench` | `nam_benchmark` |
|---|---|---|
| built by | Xcode, from `project.yml` | CMake, `-DNAMBENCH_BUILD_BENCHMARK=ON` |
| runs on | macOS, iOS | macOS, Linux, Android |
| status | canonical — every published number came from it | exists because the Pi cannot run the other one |

`nam_benchmark` is a port of `BenchCore`'s protocol, not an approximation of it:
same warm-up, same timing window, same tightest-70% selection, same
`(max - min) / min` spread test, same retry-and-reject, same defaults. Measured
back to back on one M2, with everything else equal:

| | `nambench` | `nam_benchmark` |
|---|---|---|
| `upstream` | 413.96 ms (min 411.13) | 415.00 ms (min 412.15) |
| `fused` | 220.27 ms (min 218.91) | 218.85 ms (min 217.01) |
| ratio | 1.879x | 1.896x |

Within 0.25% on `upstream`, 0.6% on `fused`, 0.9% on the ratio — on a machine
that was not idle. Getting there needed one non-obvious thing: `BenchCore` runs
its measurement on a `QOS_CLASS_USER_INTERACTIVE` thread, and until the portable
driver did the same, macOS put it on the efficiency cores and the two disagreed
by 1.4% in opposite directions.

Each testbed uses one driver for life, so no history ever contains a mixture.

## Running one by hand

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
cmake --build build-benchmark --target nam_benchmark --parallel
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
- **Thermal.** `vcgencmd get_throttled` is read either side of the run. A
  machine that started throttling partway through did not measure one machine,
  and the script says so.
- **Pinning.** `--cpu-set` hands the run a fixed set of cores so the scheduler
  cannot migrate it mid-pass onto a cold cache.

Add `--bmf out.json` to also write Bencher Metric Format.

## The machines

| testbed | machine | driver | notes |
|---|---|---|---|
| `m2-air` | M2 MacBook Air 15" | xcode | on macOS 27 beta |
| `m1-air` | M1 MacBook Air | xcode | |
| `pi500` | Raspberry Pi 500, Cortex-A76 | portable | Ubuntu 24.04 aarch64 |

**The Pi is a Pi 500, not a Pi 5.** Same BCM2712 and the same Cortex-A76, so as
a *core* it is the Pi 5 datapoint — but the 500 is passively cooled inside a
keyboard, and a Pi 5 with a fan will hold a boost clock for longer under
sustained load. The testbed is named `pi500` rather than `pi5` so that nobody
later reads a thermal difference as a code change.

Measured there, against the same capture as every published run:

```
Raspberry Pi 500 Rev 1.0 — Cortex-A76, 4 cores, Ubuntu 24.04.4 LTS
parity upstream vs fused: 132.5 dB below signal
  upstream   1165.59 ms  (min 1165.14)  RTF   9.4x
  fused       478.65 ms  (min 478.47)  RTF  22.8x   2.435x vs upstream
  SoC 46.3 C -> 49.6 C, throttled=0x0 throughout
```

Two things in there are worth more than the absolute numbers:

**`fused` is worth 2.44x on a Cortex-A76, against 1.90x on an M2.** The same
kernels, the same capture, the same protocol — a much bigger win on the smaller
core. That is the sort of result the whole exercise was for, and no Apple
machine could have produced it.

**The Pi is a far quieter instrument than either Mac.** Spread across the
accepted samples was 0.09% and 0.07%, where the M2 routinely lands between 1%
and 4% and sometimes fails all five attempts. A dedicated machine with nothing
else running is worth more than a fast one.

## Continuous tracking with Bencher

[`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) runs on
`workflow_dispatch` only — these runners are somebody's laptop — with a global
`concurrency` group so two runs can never share a machine.

Each machine is its own Bencher testbed, so its history is only ever compared
with itself. An M2 and a Cortex-A76 are not two samples of one population.

Two measures are reported per variant:

- **`latency`** — milliseconds per pass over the whole input file. `value` is
  the mean of the accepted samples; `lower_value` and `upper_value` are the min
  and max of that same accepted set, so the error bars are the real spread of
  what was kept rather than a symmetric guess.
- **`throughput`** — the real-time factor, in seconds of audio per second of
  wall clock. Bigger is better, which is the direction Bencher already assumes.

A variant that failed its agreement threshold is **omitted entirely**, never
reported as zero. `Scripts/bencher-report.py` refuses to emit an empty run for
the same reason: an empty result set uploads as "nothing regressed".

The threshold is a t-test on `latency` with a 0.98 upper boundary over a rolling
64-sample window, and `--err` makes a breach fail the job.

### By hand, without a runner

`Scripts/track-benchmark.sh` does everything the workflow does except the
runner: it picks the same driver and the same testbed name, so runs made this
way and runs made later by a self-hosted runner form **one continuous history**
rather than two forked ones.

```bash
export BENCHER_API_TOKEN="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
export BENCHER_PROJECT=nambench
```

Then, on the M2 Air:

```bash
./Scripts/track-benchmark.sh
```

and on the Pi:

```bash
ssh piv 'cd ~/NAMBench && PATH="$HOME/.cargo/bin:$PATH" BENCHER_API_TOKEN=... BENCHER_PROJECT=nambench ./Scripts/track-benchmark.sh --cpu-set 0-3'
```

The script works out which machine it is on — `m2-air`, `m1-air`, `pi500` — and
refuses to guess if it cannot. That matters more than it looks: a wrong testbed
name is not an error, because Bencher creates testbeds on demand, so it quietly
starts a *second* history for one machine.

Useful flags:

| | |
|---|---|
| `--dry-run` | measure and convert, print the BMF, upload nothing |
| `--thresholds` | install the regression thresholds (see below) |
| `--fail-on-alert` | exit non-zero when Bencher raises one |
| `--submodel narrowest` | measure the 3-channel path instead |
| `--timing-seconds N` | shorter window while you are setting things up |

Run `--dry-run` first. It exercises the whole path bar the upload.

**Don't add `--thresholds` yet.** A t-test against fewer than ten runs is
arithmetic on nothing; the flag sets `--threshold-min-sample-size 10` so nothing
alerts before then, but there is no reason to install a model until there is a
history for it to describe. Do a handful of runs on each machine first, then add
`--thresholds` once, and drop it again afterwards.

**Commit before recording anything you care about.** A result is attributed to a
commit, and the script warns if the working tree does not match it — an
attribution that is wrong survives in the history long after the working copy is
gone.

The token is read into the process environment and never passed as an argument:
process arguments are readable by every other process on the machine for as long
as the command runs. Set `BENCHER_OP_REF` to the `op://` reference instead of
exporting the token, and the script will read it from 1Password itself.

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

The API token is yours to install — I have not touched it, and it should not
pass through a terminal, a file in the repo, or a chat window. With the
1Password CLI, it never becomes visible at all:

```bash
op read "op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential" | gh secret set BENCHER_API_TOKEN --repo rikkus/NAMBench
```

If the field is not called `credential`, this lists the item's fields without
printing their values:

```bash
op item get f2x4p5ymikp25e4hlocah2zexe --vault y6stxkiynyni73l5xoygigzswi --format json | jq '.fields[].label'
```

Then the project slug, which is not a secret:

```bash
gh variable set BENCHER_PROJECT --repo rikkus/NAMBench --body "your-bencher-project-slug"
```

For running by hand on the Pi, put it in the environment rather than in a file:

```bash
export BENCHER_API_TOKEN="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
```

### Registering a runner

On each machine, from **Settings → Actions → Runners → New self-hosted runner**
in the repository, following GitHub's generated commands. When it asks for
labels, add the one this workflow expects — `nambench-m2air`, `nambench-m1air`
or `nambench-pi500` — alongside the defaults it fills in for you.

The Pi wants two more things:

```bash
sudo apt-get install -y cmake util-linux
```

and passwordless sudo for the governor, which it already has. Without it,
`run-benchmark.sh` refuses to run rather than producing a biased number.

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

**The 2015 Intel MacBook Pro.** Deliberately left alone. `fused` is AArch64-only,
so an Intel Mac can only measure `a2_fast` against the generic engine — which
CI already does, on x86_64, for free, without touching a machine that has
music-production work on it.
