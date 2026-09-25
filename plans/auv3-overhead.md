# Measure what the AUv3 wrapper costs on top of the kernel

## Context

Every number NAMBench publishes is kernel time: one process, one in-RAM model,
one in-RAM input, the shim's block loop timed from inside the shim. That is the
"no wrapper" baseline by construction. The question is what shipping the same
kernel as an AUv3 adds on top, measured rather than estimated.

The expected answer is a cost that is fixed per render call, not per frame,
and dominated by the process boundary when the extension runs out of process
(always, on iOS). The experiment is built to confirm or refute that shape, and
to put a µs-per-call figure on each layer separately rather than one lump.

Nothing like this exists in the repo today: no AU code, no extension target, no
host.

## What gets measured: five arms, each adding one layer

Every arm runs the same engine, the same model, the same input, 48 kHz, fixed
block size, FTZ set, and is driven from one dedicated thread pulling blocks as
fast as it can (offline, no audio device). Each arm differs from the previous
one by exactly one thing, so each difference is attributable.

| Arm | What runs | Adds over the previous arm |
|---|---|---|
| **A** | Existing `P_process` over the whole file | — (today's published number) |
| **A′** | Bare harness calling a new per-block shim entry, one call per block | Per-call shim entry, float↔double conversion |
| **B** | `AUAudioUnit` subclass compiled into the host, registered with `registerSubclass`, driven through its `renderBlock` | AU API plumbing: render block, `AudioBufferList`, pull-input block, event list |
| **C** | Same AU packaged as an app extension, loaded with `.loadInProcess` (macOS only) | Extension packaging and loading, still in-process — expected ≈ B |
| **D** | Same extension, loaded out of process (macOS default) | The process boundary |
| **E** | Arm D on an iOS device | The number that matters: iOS has no in-process option |

Arm C exists so that if D is expensive, the cost is pinned on the boundary and
not on something the extension packaging does.

A′ is needed because A cannot be compared to an AU fairly: `P_process` loops
internally and reads the clock twice per file, while every AU arm calls the
kernel once per block from outside. A′ makes the same per-block call the AU
makes, with nothing around it. A→A′ should be ≈0; if it is not, that is worth
knowing before anything is blamed on the AU.

### Timing points

Host side, per block: `mach_absolute_time()` immediately before and after the
`renderBlock` call. That is the per-block cost each arm reports.

Plugin side, per block: the same clock at render-block entry, before the kernel
and after it, and at render-block exit. `mach_absolute_time` is a single
system-wide timebase, so timestamps taken in the extension process line up
with the host's. For arm D that decomposes one render call into:

```
host call ──hop in──▶ ext entry ──wrapper──▶ kernel ──wrapper──▶ ext exit ──hop out──▶ host return
```

The extension writes these into a preallocated ring (no allocation, no locks on
the render thread) and the host collects it after the run over
`AUMessageChannel` (macOS 13 / iOS 16, both inside the deployment targets
already in `project.yml`). Nothing crosses the boundary during timing except
what the host would send anyway.

### Block-size sweep

16, 32, 64, 128, 256 frames. If the per-call model holds, overhead per block is
flat across the sweep and overhead as a share of the callback period doubles
with every halving. A least-squares fit of `overhead_ns = a + b·frames` per arm
gives `a` (fixed per call) and `b` (per frame, expected ≈0 except A′'s
conversion). This is the check that the headline claim from the earlier
discussion is actually true.

### Engines and submodels

`a2_planar`, both submodels. Nano matters most: its kernel is short (28.6 ms
for the whole file on the M2), so a fixed per-call cost is a larger fraction
of it. `a2_fast` is not needed. Overhead should not depend on the engine, and
running it on one checks that it doesn't, which is enough.

## What gets reported

Per arm, per block size, per submodel:

- `core_percent`, computed exactly as today from the host-side per-block times.
- Wrapper overhead in ns per block, as the arm minus A′, median and p99.
- For D and E: the hop-in / wrapper / kernel / hop-out split, median and p99.

Using the existing protocol from `Tools/nb_protocol.h`: warm-up, the
tightest-70% pass selection, the spread test, and retry-and-reject. p99 is
reported the same way `block_p99_percent` is for the IR benchmark, because an
IPC hop is exactly the kind of cost that shows in the tail before the mean.

A results section in `BENCHMARKING.md` and an `AUV3-PATH.md` alongside the other
`*-PATH.md` files, in their usual form: tables, then decisions, then what was
tried and why.

## Build shape

All new code is under `Sources/AUv3/` and new xcodegen targets. Nothing in the
existing targets changes except one shim addition.

1. **Shim**: add `P_process_block(model, const float* in, float* out, int32_t frames)`
   to `NB_DECLARE_VARIANT`. No timing, no FPCR changes: the caller owns both.
   It converts to `NAM_SAMPLE` (double in these builds), runs one block and
   converts back. This is the conversion the shipping plugin pays too: iPlug2
   works in double.
2. **`NAMBenchAU` framework**: the `AUAudioUnit` subclass in Objective-C++
   (render block is a plain C++ lambda over a kernel struct, capturing no ObjC
   object), linking `NAMEnginePlanar`. One mono in/out bus, float32
   deinterleaved, one parameter (submodel select) so the parameter-tree path
   exists but no smoothing work is added. Model bytes arrive over the message
   channel so the sandboxed extension needs no file access.
3. **`NAMBenchAUExtension`** (app-extension target, `com.apple.AudioUnit`
   extension point, component type `aufx`) wrapping the framework, embedded in
   a small container app. Neither needs a UI.
4. **`nambench-au-host`**: macOS tool (plus a thin iOS app wrapper for arm E)
   that runs arms A′–D and emits the same JSON shape `nambench` does, with arm
   and block size as extra fields.

Container app, extension and host each get the same `-O3` treatment the
`Engine` template enforces. The template's comment about Debug builds applies
here too.

## Phases

**0. Spike: go/no-go.** Pass-through AUv3 extension, ad-hoc signed, registered
on this Mac, loaded out of process by a 20-line host that calls `renderBlock`
in a loop from its own thread. The questions this answers before anything else
is built:

- Does an ad-hoc-signed extension register with `pluginkit` and load out of
  process, or does arm D need the development team the iOS device build
  already needs?
- Does calling `renderBlock` from a plain, non-audio-device thread go through
  the same remote render path a real host uses, or does it take some other
  route when there is no IO unit? If it takes a different route, arms D/E
  measure the wrong thing and the host has to be an `AVAudioEngine` in manual
  rendering mode instead. That is the biggest methodological risk in this
  plan, and the spike exists mainly to settle it.
- Does `AUMessageChannel` work for pulling a few MB of timing data out after a
  run?

**1. Arms A′, B, C, D on macOS**, full sweep, both submodels. First real
numbers.

**2. Arm E on iOS.** `xctrace` lists an iPad and two iPhones (`mu14`, `mu17`),
currently offline. Same host code in the thin app wrapper, timestamps shipped
home the same way the app already exposes reports via the Files app.

**3. Write-up**: `AUV3-PATH.md` and the `BENCHMARKING.md` section.

## Deliberately out of scope

- **Real-time scheduling.** Everything above runs offline, so it measures CPU
  cost, not whether the deadline is met. Workgroup join, thread priority and
  deadline misses under an actual 64-frame IO unit are a separate experiment
  (an arm F on a real `AVAudioEngine` output, counting overruns rather than
  timing). Worth doing, but it answers a different question.
- **Cache contention** from sharing the machine with a host and other plugins.
  Probably bigger than anything measured here for NAM, and needs a different
  design (N co-running instances, or a controlled cache-thrashing neighbour).
- **Sample-rate conversion.** Real, but it's DSP. It belongs in the kernel
  benchmarks as its own subject, not in a wrapper measurement.
- **Real DAWs.** Logic, AUM and friends add their own per-plugin work, which
  varies by host and can't be controlled. The minimal host gives the floor
  Apple's machinery imposes.

## Open decisions

1. **Bencher or one-off.** Wrapper overhead is an OS property more than a
   property of this code, so it won't move with NAMBench commits. It might
   still be worth tracking per OS release, as a new measure on its own
   testbeds. Or it could just be recorded once in `AUV3-PATH.md`.
2. **Arm F now or later.** It's the part that reflects what users actually
   hear, but it roughly doubles the work.
3. **Which iOS devices for arm E**, and whether the older one is worth including
   to see how the hop cost scales with a slower core. Settled: mu17 only.
   mu14 has been retired, so the older-core comparison won't be run.
