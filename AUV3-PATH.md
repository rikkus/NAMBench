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
kernel runs as fast in the extension as in the host. That is arm F, below.

## Arm F: under a real IO thread

Arms A–D run offline and measure CPU cost. Arm F asks whether the deadline is
met: the unit runs in an `AVAudioEngine` whose output is the device's real IO,
at 48 kHz, with the IO buffer set to 32, 64, 128 and 256 frames in turn, 20 s
per subject. **Fi** is the unit registered in the host (arm B's). **F** is the
extension, out of process. The main mixer is muted but still pulled: every
subject's render-call count equals its IO-cycle count.

A miss is counted three ways: a gap in the output's sample time between
consecutive IO cycles, a HAL processor-overload notification (macOS only), and
an IO cycle whose render took longer than its period.

### Deadlines

| | M2 | iPhone 17 |
|---|---|---|
| Missed, nano, any buffer, Fi or F | 0 | 0 |
| Missed, standard, 64–256 frames, Fi or F | 0 | 0 |
| Missed, standard, 32 frames | 0 | **Fi: 41 gaps (1968 frames); F: 101 gaps (5092 frames)** |
| Worst IO cycle, % of period (standard, 32 frames, F) | 14.7% | 23.4% |
| IO cycles longer than their period | 0 | 0 |

The iPhone's 32-frame standard gaps are real dropouts, but this setup does not
explain them. No cycle's render came near its period (the longest took 156 µs
out of 667 µs), so the IO cycle was skipped outside the span the render notify
times: a late wake-up of the IO thread, or the device side of the IO. They are
worse out of process (101 against 41) and absent at every other size.

### The cost of out of process, per IO cycle

F's median IO cycle minus Fi's, in µs:

| IO frames | M2 nano | M2 standard | iPhone nano | iPhone standard |
|---:|---:|---:|---:|---:|
| 32 | 4.4 | 4.7 | 11.9 | 10.3 |
| 64 | 15.4 | 2.3 | 15.9 | 15.8 |
| 128 | 9.9 | 4.4 | 11.3 | 15.3 |
| 256 | 8.4 | 6.5 | 20.6 | 16.4 |

This is 3–7 times the offline hop (3–3.7 µs per call), and it isn't a fixed
cost per call. The two hops wait on a thread in another process being woken,
and under real time that thread is woken from idle every cycle rather than
kept busy. The M2 standard figures are smaller than nano's because the
extension's kernel ran faster than the host's there (next section), not
because the hops were cheaper.

### The kernel is as fast in the extension

Median kernel time per render call, µs, Fi / F:

| IO frames | M2 nano | M2 standard | iPhone nano | iPhone standard |
|---:|---:|---:|---:|---:|
| 32 | 2.8 / 2.7 | 12.1 / 11.8 | 10.4 / 10.5 | 40.5 / 41.2 |
| 64 | 18.1 / 18.3 | 63.3 / 53.9 | 23.4 / 23.0 | 125.0 / 124.3 |
| 128 | 33.5 / 32.4 | 82.5 / 73.9 | 46.0 / 41.9 | 241.8 / 239.7 |
| 256 | 37.6 / 32.0 | 163.2 / 155.0 | 77.0 / 77.8 | 476.4 / 472.9 |

In no pairing is the extension's kernel meaningfully slower than the host's. The
10–40% offline slowdown on the M2 does not survive a real IO thread. The
extension's render thread has the time-constraint (real-time) policy in every
subject on both machines, although the unit's `renderContextObserver` is never
called out of process, so the extension is never handed a workgroup. In process
it is called once and a workgroup arrives.

### The kernel is not as fast as the published numbers

Under real time the kernel takes longer than bare, back-to-back calls to the
same entry point on the same input:

| IO frames | M2 nano | M2 standard | iPhone nano | iPhone standard |
|---:|---:|---:|---:|---:|
| 32 | 1.4× | 1.2× | 6.6× | 5.0× |
| 64 | 4.8× | 3.3× | 7.0× | 8.1× |
| 128 | 4.5× | 2.2× | 6.9× | 7.8× |
| 256 | 2.6× | 2.2× | 5.4× | 7.8× |

(Fi's median over the bare median. F is similar.)

The cause is where and how fast the render runs, not the AU:

- **Which cores.** Each render call records its CPU number, and the host
  classifies CPUs by timing a fixed chunk of work on every one (0–3 are E and
  4–7 P on the M2; 0–3 E and 4–5 P on the iPhone). On the M2 every render call
  ran on a P core. On the iPhone nearly every call ran on an E core. Standard at
  32 frames is the exception: 5–14% of its calls were on a P core.
- **How fast those cores run.** On the M2 the calls are on P cores and still
  2–5× slower at 64 frames and above. A light, periodic load does not raise the
  cluster's clock. At 32 frames the IO thread wakes 1500 times a second, and the
  kernel runs within 1.2–1.4× of bare. This was confirmed by accident: straight
  after a second of all-core load, the same 64-frame nano subject's kernel took
  5.2 µs instead of 18. So the host now idles for 15 s after its cluster probe
  before timing anything.
- **Not AudioToolbox.** A paced reference runs the bare kernel on a thread of the
  host's own, one block per period with the IO thread's time-constraint policy.
  Joined to an audio work interval (`AudioWorkIntervalCreate`, each block marked
  as an interval), it roughly matches the engine on the M2 at 64 and 128 frames
  (nano 18.5 against 18.1, standard 65 against 63). Without the work interval it
  is slower, and it moves by up to 4× between runs. Neither reference matches
  at 32 or 256 frames, so they only bound the answer.

So the offline core % understates the real-time cost, most of all on the
iPhone, where the render runs on E cores. What counts for headroom is the
real-time cycle: standard at 64 frames uses 5.4% of the period on the M2 and
11.1% on the iPhone, out of process, against the 2.3% and 1.9% the offline
numbers imply.

### Method

- `Sources/AUv3/Host/NBAURealtime.mm`. Chain: `AVAudioSourceNode` (the input
  file, looped) → the unit → main mixer (volume 0) → the default output. The
  host checks that the output format is 48 kHz before starting, so no converter
  can be inserted. On macOS the device buffer is set through the output unit's
  `kAudioDevicePropertyBufferFrameSize`. On iOS it is the audio session's
  preferred IO buffer duration, and the observed frames per cycle are what is
  reported.
- A render notify on the output unit stamps each IO cycle's start and end and
  its output sample time. The unit stamps each render call as in arm D, plus
  its CPU number.
- The unit captures its render thread's scheduling policy once, on its first
  call. That is the only syscall the render thread makes.
- The first 2 s of each subject are discarded. The references run after the
  engine has stopped.

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

Arm F: add `--arms Fi,F --blocks 32,64,128,256 --rt-seconds 20` to either
command. On macOS it plays (silently) to the default output device, which must
be at 48 kHz. On iOS put the same options after the bundle ID. Results are in
`benchmark-results/auv3rt_20260925_{m2,mu17}.*`.
