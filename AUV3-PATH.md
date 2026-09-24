# What the AUv3 wrapper costs

Every other number in this repository is the cost of the kernel alone: one
process, one in-RAM model, one in-RAM input. This file measures what wrapping
the same kernel in an AUv3 adds, layer by layer. The plan it was run from is
[plans/auv3-overhead.md](plans/auv3-overhead.md).

Two machines: an M2 MacBook Air (macOS 27) and an iPhone 17 (`mu17`, iOS 27).
The `a2_planar` engine, both A2 submodels from the capture named in
[nam-files/README.md](nam-files/README.md), 48 kHz, offline, block sizes 16 to
256.

## Answer

| At 64 frames | M2, nano | M2, standard | iPhone 17, nano | iPhone 17, standard |
|---|---:|---:|---:|---:|
| Bare kernel, % of a core (A1) | 0.266 | 1.495 | 0.266 | 1.629 |
| AU, in-process (B) | 0.271 | 1.502 | 0.275 | 1.644 |
| AUv3 extension, out of process (D) | **0.655** | **2.285** | **0.472** | **1.873** |
| Out-of-process cost per call, median / p99 | 3.5 / 9.8 µs | 3.7 / 13.1 µs | 2.7 / 4.2 µs | 3.0 / 5.8 µs |

- **The AU API itself is essentially free.** In-process, a render call costs
  about 80–120 ns more than calling the kernel directly: under 0.01% of a
  64-frame period. On macOS, loading the unit from an app extension but
  in-process (arm C) is indistinguishable from registering the class directly
  (arm B).
- **The process boundary costs about 3 µs per render call**, split evenly
  between the hop in and the hop out, and fixed per call. The fit of
  `overhead = a + b·frames` gives `a` ≈ 2.6–3.0 µs on the iPhone and
  3.6–4.5 µs on the M2, with `b` at about 1 ns per frame or less. On the
  iPhone it is 0.2% of a core at 64 frames and 0.4% at 32.
- **Relative to the kernel, it is the small model that pays.** A2 standard on
  the iPhone goes from 1.63% to 1.87% of a core at 64 frames (+15%). Nano goes
  from 0.27% to 0.47% (+77%), because the fixed cost is now comparable to the
  kernel itself.
- **Every AU arm's output is bit-identical** to the bare kernel's, on both
  machines, at every block size that ran.
- **No sample-rate conversion anywhere.** No arm has an audio device or an
  `AVAudioEngine`, which are the only places AudioToolbox could insert a
  converter. The unit refuses to allocate at anything but 48 kHz, and the host
  asserts that the bus rate, the model rate and the input rate are all 48 kHz
  before it times anything.

## The arms

| Arm | What runs | Adds |
|---|---|---|
| A | `nb_planar_process` over the whole file | — (the number NAMBench publishes) |
| A1 | `nb_planar_process_block` once per block, with FTZ set per call | per-call entry, float↔double conversion |
| B | `NBAudioUnit` registered in the host with `registerSubclass`, driven through `renderBlock` | the AU API: render block, `AudioBufferList`, pull-input, event list |
| C | the same unit from the app extension, `LoadInProcess` (macOS only) | extension packaging |
| D | the same extension, `LoadOutOfProcess` | the process boundary |

The plan's arm E (arm D on iOS) is arm D in the iPhone results.

A and A1 agree to within noise everywhere except 16-frame nano on the M2 (0.473%
against 0.496%). Calling per block from outside the shim, and converting to
float, costs nothing measurable.

"Wrapper overhead" per block is host-side time around the call minus the
kernel's own time, measured on the same block. Every timestamp is
`mach_absolute_time()`, which is one timebase across processes, so for arm D
the unit's stamps split each call into hop in, wrapper, kernel, wrapper out and
hop out. The 42 ns floor on A1 is one tick of the 24 MHz clock, not a cost.

## iPhone 17: out-of-process call, decomposed (median ns)

| Submodel | Block | hop in | kernel in extension | kernel bare (A1) | hop out |
|---|---:|---:|---:|---:|---:|
| nano | 32 | 1250 | 1708 | 1667 | 1333 |
| nano | 64 | 1292 | 3417 | 3375 | 1375 |
| nano | 256 | 1333 | 13917 | 13833 | 1500 |
| standard | 64 | 1375 | 21333 | 21292 | 1583 |
| standard | 256 | 1500 | 84625 | 84667 | 1667 |

## M2: the kernel runs slower in the extension

| Submodel | Block | hop in | kernel in extension | kernel bare (A1) | hop out |
|---|---:|---:|---:|---:|---:|
| nano | 64 | 1667 | 4667 | 3333 | 1750 |
| nano | 256 | 1708 | 16458 | 12583 | 1750 |
| standard | 64 | 1708 | 25958 | 19292 | 1833 |
| standard | 256 | 1750 | 83875 | 75917 | 1750 |

On the M2 the same kernel, on the same input, takes 10–40% longer inside the
extension process than in the host. On the iPhone it does not. The hops are
similar on both machines, so this is not the boundary. It is where macOS
schedules the extension's render thread. The host's timing thread is
user-interactive, but the extension renders on a thread of its own, and in this
offline setup nothing joins it to an audio workgroup. This is why arm D on the
M2 costs more in `core %` than its per-call hop alone accounts for.

Don't read it as an AUv3 cost. Under a real host's real-time IO thread, the
extension's thread is expected to join the host's workgroup, and it may then
run at full speed. Whether it does is the question for arm F (real-time
scheduling), which was deliberately left out of this round.

## Limits found

- **iOS hosts need the `inter-app-audio` entitlement.** Without it, AudioToolbox
  lists no extension AUs to the app at all. That means not only this one but
  every AU on the device (AUM on the same phone lists all of them), and
  instantiation fails with -3000. The extension itself needs nothing special:
  the UI-less `com.apple.AudioUnit` extension point works on iOS 27, and so does
  the host embedding its own extension. None of these made a difference: an
  active audio session, a UI extension point, a separate container app, the
  `audio` background mode, or launching from the home screen instead of
  `devicectl`.
- **Out of process, `maximumFramesToRender` has a floor.** On macOS allocation
  fails with -10875 at 23 frames and below and succeeds at 24. On iOS it fails
  at 16 (not bisected further). In-process, 16 is fine. The 16-frame arm D
  cells are missing from both sweeps for this reason.
- **The in-process p99 has occasional 8–17 µs outliers on the M2** (arms B and
  C at 128 frames, for example). These are single preemptions over hundreds of
  thousands of calls, and they appear in B and C alike, so they are the machine
  and not the extension. The iPhone shows none: its in-process p99 stays under
  0.9 µs.

## Method

- The host is `NAMBenchAUHost` (`Sources/AUv3/Host`), the unit is `NAMBenchAU`
  (`Sources/AUv3/Unit`), and the extension is `NAMBenchAUExtension`. All three
  are signed with the development team, and nothing else in the project depends
  on them.
- Each subject goes through the protocol in `Tools/nb_protocol.h`: warm-up, a
  timing window, tightest-70% pass selection, the spread test, and
  retry-and-reject. Per-block distributions are pooled over the accepted passes
  only. The window is shorter than `nambench`'s: 2 s of warm-up and 10 s of
  timing, because there are up to fifty subjects. `minSamples` still guarantees
  fifteen passes.
- Every subject asserts that it ran `a2_planar`, at 48 kHz, in the process it
  was meant to run in (unit PID against host PID). Every AU arm is compared
  sample by sample with A1's output.
- The unit's render block captures a pointer to plain C++ state, not `self`,
  sets FTZ around the kernel as A1 does, walks the event list, and pulls its
  input like any effect. The unit exposes one parameter so the parameter tree
  exists, but does no smoothing.
- Timestamps come out of the extension after each pass over an
  `AUMessageChannel`, never during timing.

## Reproducing

```bash
xcodegen generate
xcodebuild -scheme NAMBenchAUHost -configuration Release -destination 'platform=macOS' -allowProvisioningUpdates build
```

```bash
pluginkit -a <products>/NAMBenchAUHost.app/Contents/PlugIns/NAMBenchAUExtension.appex
```

```bash
<products>/NAMBenchAUHost.app/Contents/MacOS/NAMBenchAUHost --model "nam-files/Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam" --input audio-input/input.wav --json out.json
```

On iOS, build the same scheme for the device, install it, and launch it with
`xcrun devicectl device process launch --console cc.hemsley.nambench.auhost`.
The same options can follow the bundle ID. It writes `auv3-overhead.json` to
its Documents folder. `Scripts/auv3-summary.py` turns either JSON into the
tables above.
