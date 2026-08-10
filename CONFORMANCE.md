# Conformance

This repository measures. That work is done by the Xcode project, on hardware
somebody owns, and none of it is done in the cloud.

Conformance is the other question. The engines are arithmetic, and arithmetic
travels further than `xcodebuild` does. The conformance build takes the same
engine sources to Linux, Windows, Android and iOS, and to x86_64 as well as
AArch64, under GCC, Clang, Apple Clang and MSVC, and asks two things:

1. does it still compile, and
2. does it still compute the same audio.

It runs on every push and pull request, free, on GitHub's hosted runners.

**No timing is read from it.** Every hosted runner is a shared virtual machine
with a neighbour, no thermal headroom and no way to pin a clock. The driver
measures elapsed time, because the shim's `_process` returns it, and then throws
the number away. A millisecond from CI put next to a millisecond from a pinned
M2 would devalue the second one.

## What it checks

### Routing

Each variant must land on the engine it is supposed to land on, and the driver
reports the engine it actually got by asking the same detectors
`wavenet::create_config` asks — never by reading back its own build flags.

| | 8-channel submodel | 3-channel submodel |
|---|---|---|
| `a2_fast` | `a2_fast` | `a2_fast` |
| `a2_planar`, on AArch64 | `a2_planar` | `a2_planar` |
| `a2_planar`, elsewhere | `a2_fast` | `a2_fast` |
| `fused`, on AArch64 | `fused` | `generic` |
| `fused`, elsewhere | `generic` | `generic` |
| slim lab | not built | `slim` |
| full lab | `full` | not built |

The two `generic` cells are the interesting ones. Under
`ScopedEnginePrefer(FusedNeon)` a shape the fused detector rejects falls through
to the *generic* engine and never to `a2_fast` — so a benchmark that assumed
"fused framework means fused engine" would quietly measure the wrong thing.
Off AArch64 the detector rejects everything, and on the 3-channel submodel it
rejects a channel count that is not a multiple of four. Both are asserted rather
than avoided.

That has a useful consequence. On x86_64 the `a2_fast` versus `fused`
comparison *is* `a2_fast` against the generic reference implementation — a
correctness check the Apple-only build cannot perform at all, obtained for free
from runners this project would otherwise have no use for.

### Parity

Every variant is compared against the one it is derived from, as `max|diff|` and
as dB below the reference signal — the same measure `BenchCore` reports.

- `a2_planar` against `a2_fast` — held to **exact bit-identity**, on every
  platform and compiler, because that is precisely what Core PR #313 claims and
  asks Core to rely on: "not within a tolerance, not below the noise floor — the
  same float32 bits, sample for sample"
- `fused` against `a2_fast`
- slim-lab kernels against `a2_fast` on the 3-channel submodel
- full-lab `a2*` kernels against `a2_fast`, `fu*` kernels against `fused`

The floor is 100 dB, about 17 bits down and far below the noise floor of any
capture. Current pairings sit at 119–141 dB.

The verbatim ports are held to a stricter rule. `full_common.h` and
`slim_common.h` both describe their baselines as reproducing their reference's
arithmetic exactly, so `a2_baseline`, `fu_baseline` and slim `baseline` must be
**bit-identical** — `max|diff|` of exactly zero. A verbatim port that has
drifted is the thing most worth catching, because it invalidates every number
the lab produces without looking wrong.

### Finiteness

No NaN, no infinity, no silent output, no diverged output.

## What it does not check

- **Timing.** Deliberately, permanently. See above.
- **The Swift layer.** `BenchCore`, the CLI and the app are Apple-only and are
  built by the Xcode project.
- **The real captures.** CI loads `vendor/upstream/example_models/A2.nam`,
  which is upstream's own file and arrives with the pinned checkout. It is a
  genuine A2 `SlimmableContainer` — 3-channel nano at `max_value` 0.5,
  8-channel full at 1.0, 23 layers each — so it exercises every path the
  benchmark does. The captures under `nam-files/` are other people's work, are
  not committed, and are never needed here.
- **Android and iOS on real hardware.** Android is cross-compiled only. iOS runs
  in the Simulator, which is real arm64 execution of the real kernels and so is
  perfectly good for arithmetic, and worthless for time.
- **MSVC on ARM64.** Marked experimental in the matrix. `fused.cpp` reaches for
  `<arm_neon.h>` under `_M_ARM64`, which MSVC spells differently; if that job is
  red it is a finding for the fork, not a reason to stop looking.
- **The kernel labs under MSVC.** They use `#pragma clang loop`,
  `#pragma clang fp contract` and `__builtin_prefetch`. Those pragmas are how a
  candidate pins down the codegen it is testing, so a version with them stripped
  out would be a different kernel. clang-cl is fine and is not excluded.

## Running it locally

```bash
./Scripts/fetch-vendor.sh
cmake -S . -B build-conformance -DCMAKE_BUILD_TYPE=Release
cmake --build build-conformance --parallel 4
ctest --test-dir build-conformance --output-on-failure
```

For the full parity table rather than a pass/fail:

```bash
python3 Scripts/compare-conformance.py build-conformance/conformance
```

One architecture at a time. A universal build would compile the NEON labs for
the x86_64 slice too, so the configure step refuses it.

Cross-checking another architecture from an Apple Silicon Mac:

```bash
cmake -S . -B build-x86_64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
```

## The matrix

| job | arch | compilers | runs? |
|---|---|---|---|
| linux | x86_64 | gcc-12, gcc-13, gcc-14, clang-18 | yes |
| linux | aarch64 | gcc-14, clang-18 | yes |
| macos 15, macos 26 | arm64 | Apple Clang (M2, M4) | yes |
| macos | x86_64 | Apple Clang | build only |
| ios-simulator | arm64 | Apple Clang | yes, under `simctl` |
| windows | x64 | MSVC, clang-cl | yes |
| windows | arm64 | MSVC | experimental |
| android | arm64-v8a, x86_64 | NDK Clang | build only |

Linux arm64 runners are Azure Cobalt 100 — Neoverse N2, so Armv9 with SVE2.
That is nothing like an Apple P-core or the Cortex-A76 in a Pi 5, which is
precisely why it is worth having: it is the least similar AArch64 core the
kernels are likely to meet.

`Scripts/fetch-vendor.sh` runs once, in its own job, and the result is handed to
every matrix entry. The script asserts more than it downloads — that both
variants pin the same Eigen commit, and that `a2_fast` is byte-identical across
the two checkouts — so running it once means those assertions are made once and
no two matrix entries can be testing different sources.

## Where the hardware fits

Conformance answers "is it correct here". It cannot answer "is it fast here",
and the second question is the one this repository exists for. Those numbers
come from machines with names:

| | what it answers |
|---|---|
| M1 / M2 Macs, iPhone 11 / 14 Pro Max / 17 | the numbers that get published |
| Raspberry Pi 5 (Cortex-A76) | plain NEON, small caches, no SVE — where cloud AArch64 is least representative |
| 2015 Intel MacBook Pro | not needed; CI covers x86_64 better and without disturbing it |

Nothing in this workflow needs any of them, which is the point: the hosted
matrix covers the platforms nobody here owns — Windows, x86_64 Linux, Android —
and leaves the pinned hardware free to do the only job it can do.
