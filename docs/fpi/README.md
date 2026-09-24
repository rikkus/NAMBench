# fpi — the RP2350A-USB-A Mini board

Notes on the physical device ("**fpi**", a fake Pi[co]), and on how a NAM A2-Lite
kernel behaves on one of its Cortex-M33 cores.

> **This one was for fun.** It is a curiosity-driven experiment, not a product
> plan and not a proposal. The question — what does a Cortex-M33 actually do with
> a NAM A2-Lite, and where does the time really go — was worth answering for its
> own sake, and the answer is the measurements, the dead ends and the hardware
> facts collected here.
>
> **There is no intention of running NAM on a microcontroller in reality.** A
> laptop, a phone or a pedal SoC is the right home for it, and the rest of
> NAMBench measures those. Nothing in this directory should be read as a
> suggestion that an MCU is a sensible target, and the conclusion below is that
> at stock clock one is not even close.

## Source material

| File | What it is |
|---|---|
| [`RP2350A-USB-A-Mini-datasheet.pdf`](RP2350A-USB-A-Mini-datasheet.pdf) | The original product PDF, verbatim |
| [`RP-008373-DS-2-rp2350-datasheet.pdf`](RP-008373-DS-2-rp2350-datasheet.pdf) | The full RP2350 datasheet (1380 pages) — used for the SRAM and interpolator sections below |
| [`pdf-text-extract.txt`](pdf-text-extract.txt) | Text layer of pages 1–2 (the only pages with real text) |
| [`pages/`](pages/) | One PNG render per PDF page, plus `contact-sheet.png` |
| [`TOOLCHAIN.md`](TOOLCHAIN.md) | What had to be installed on this laptop, and how to build/flash |

The firmware workspace lives in [`fpi/`](../../fpi/) at the repository root
(`build.sh`, `status.sh`, `src/`), with the Pico SDK and the Arm toolchain
checked out beside it and gitignored.

The PDF is a marketing/product sheet, not a real datasheet: pages 3–30 are
photographs with no text layer, so they had to be rendered as images and read
visually. `pages/contact-sheet.png` is the 30-page overview; `pages/p3.png`
(the GPIO table) and `pages/p5.png` (the board diagram) carry all the
information that actually matters.

## Identification

**Waveshare RP2350A-USB-A Mini Development Board**, 33 × 17.5 mm.

- **MCU: RP2350A** — *not* RP2040. Dual **Arm Cortex-M33** *and* dual
  **Hazard3 RISC-V**, 150 MHz default clock. This is the same silicon as a
  Raspberry Pi Pico 2.
- **520 KB SRAM**, **2 MB on-chip QSPI flash** (W25Q16JVUXIQ).
- **RT9013-33GB** 500 mA LDO, **WS2812** addressable RGB LED.
- 15 GPIO broken out to **stamp holes**, with 1×8 headers included in the box.

The RP2350A part matters: an M33 has an FPU plus DSP/SIMD instructions, which
an RP2040's Cortex-M0+ does not. Any "too slow for NAM" intuition carried over
from RP2040 projects does not transfer directly.

## Connectors and buttons

From `pages/p5.png` (Resource Profile):

| # | Item | Use |
|---|---|---|
| 1 | **USB-C port** | The **device**/slave port: power, flashing, USB audio, CDC console |
| 2 | **USB-A port** | Host/slave, driven via PIOs — *not* for connecting to this laptop |
| 3 | **BOOT button** | Hold at power-up → UF2 bootloader; otherwise a plain GPIO button |
| 4 | **RESET button** | Reboot; hold BOOT + tap RESET to re-enter the bootloader |
| 5 | WS2812 RGB LED | Onboard, on **GPIO16**; `DIN` is the occupied pin |

**Connect the laptop to the USB-C port (#1).** The USB-A port (#2) is for
plugging a keyboard/mouse *into* the board.

## GPIO map

From `pages/p3.png`. Pins are broken out on both edges; the left column is
ADC/bus, the right column is the SPI0/UART0/I2C0 group.

| Left edge | | Right edge | |
|---|---|---|---|
| 5V | power | GP0 | SPI0 RX / I2C0 SDA / UART0 TX |
| GND | ground | GP1 | SPI0 CSn / I2C0 SCL / UART0 RX |
| 3V3 | power | GP2 | SPI0 SCK / I2C1 SDA |
| ADC3 | GP29 | GP3 | SPI0 TX / I2C1 SCL |
| ADC2 | GP28 | GP4 | SPI0 RX / I2C0 SDA / UART1 TX |
| ADC1 | GP27 | GP5 | SPI0 CSn / I2C0 SCL / UART1 RX |
| I2C1 SCL | GP26 | GP6 | SPI0 SCK / I2C1 SDA |
| I2C1 SDA | GP10 (SPI1 SCK) | GP7 | SPI0 TX / I2C1 SCL |
| I2C0 SCL | GP9 (SPI1 CSn) | GP8 | SPI1 RX / I2C0 SDA / UART1 TX |
| UART1 RX | | | |
| — | | **GP16** | **WS2812 `DIN` — occupied** |

There are **12 PIO state machines**, which matters later: I2S in/out for audio
can be done with PIO rather than dedicated hardware.

## How the laptop sees it

Three distinct states, and it is worth knowing which one you are in:

| Board state | macOS evidence | Notes |
|---|---|---|
| **BOOTSEL** (boot ROM, no firmware) | A volume named **`RP2350`** mounts in `/Volumes`; USB device `2e8a:000f` | Drag any `.uf2` onto it; board reboots into that firmware |
| Firmware with `stdio_usb` / TinyUSB CDC | `/dev/cu.usbmodem*` | Console/printf, benchmarks |
| Firmware presenting UAC2 audio | a CoreAudio device; **no `/dev` node at all** | This is the NAM audio path — absence of a serial port is not a fault |

A blank board will **not** produce a serial port. The boot ROM's UF2
mass-storage mode is only entered if **BOOT is held while power is applied**
(or BOOT held while RESET is tapped) — plugging in an unprogrammed RP2350
normally drops straight into the (empty) flash and does nothing visible.

Observed on this laptop (macOS 27.0, Apple silicon, arm64) at the time of
writing: no `RP2350` volume and no `usbmodem` node, i.e. nothing connected yet.

## The connection procedure

1. Use a **USB-C cable that carries data**. Charge-only cables are the single
   most common cause of "the board is dead" — if no volume appears, suspect the
   cable before the board.
2. **Hold BOOT**, plug into the laptop's USB-C port, then release BOOT.
3. Confirm `RP2350` mounted: `ls /Volumes/RP2350`.
4. Flash a `.uf2` — `cp build/foo.uf2 /Volumes/RP2350` (or
   `picotool load -f build/foo.uf2`). The volume disappears as the board
   reboots.
5. For iteration afterwards, **hold BOOT and tap RESET** instead of unplugging.

## The single-core budget (why this is the interesting question)

The reference project is
[oyama/pico-neural-amp-modeler-demo](https://github.com/oyama/pico-neural-amp-modeler-demo),
which is already an **RP2350** project — same silicon as fpi. Its published
per-sample costs, on NAM A2-Lite:

| code path | cycles/sample | CPU @ 300 MHz |
|---|---:|---:|
| generic Eigen | 31,030 | 496% |
| `a2_fast` | 8,396 | 134% |
| `a2_fast` + dual-core | 4,533 | 73% |

A 48 kHz sample budget at 300 MHz is **6,250 cycles**. So:

- `a2_fast` on **one** core is **134% of one core** — it does not fit, and that
  is at 300 MHz, not the 150 MHz default.
- The project's dual-core split is what makes it fit, and it needs 300 MHz at
  1.20 V core to get there.
- The project itself notes it runs *both* M33 cores, so it is a reference for
  the kernel and the tooling, not for a single-core configuration.

Implication for "NAM lite on one core": the target is roughly a **2–3× reduction
in convolution cost** versus `a2_fast`, on a core that does have DSP/SIMD
(`SMLAD`/`SMUAD` over packed 16-bit weights) and an FPU. That is a concrete,
measurable goal rather than a hope — `fpi/bench` exists to measure exactly that.

### The A2-Lite shape, from the model file

Read out of the reference project's committed `example.nam` rather than assumed:

```
layers       23, channels 3
kernel_sizes [6 x14, 15, 15, 6 x7]
dilations    [1, 3, 7, 17, 41, 101, 239] repeating
activation   LeakyReLU, negative_slope 0.01
params       ~1871  (~7.5 KB)
receptive field = 1 + sum((k-1)*d) = 6332 samples = 131.9 ms @ 48 kHz
```

This confirms the "~132 ms" the reference README quotes — the two agree exactly.
(An earlier version of this note claimed they disagreed; that was an error here,
from mis-pairing the kernel sizes with the dilations. Pair them properly and it
is 6332 samples.)

Getting this table right matters, because the per-layer ring sizes are derived
from it:

| layer | dilation | taps | ring samples |
|---|---|---:|---:|
| 0–6 | 1, 3, 7, 17, 41, 101, 239 | 6 | 6, 16, 36, 86, 206, 506, 1196 |
| 7–13 | same again | 6 | 6, 16, 36, 86, 206, 506, 1196 |
| 14, 15 | 1, 13 | **15** | 15, 183 |
| 16–22 | 1, 3, 7, 17, 41, 101, 239 | 6 | 6, 16, 36, 86, 206, 506, 1196 |

Total **6354 samples**. A layer only needs `(k-1)*d + 1` samples, not the whole
6332-sample receptive field, because it only ever reads its own taps.

### Memory: not actually the constraint

At 6354 samples the stack is small:

| representation | ring storage, 3 channels |
|---|---:|
| int16 Q15 | 38,124 bytes |
| float | 76,248 bytes |

Both fit in the RP2350A's 520 KB comfortably, so fixed point here is a *speed*
choice, not a capacity one. (An earlier note claimed a float ring stack needed
~500 KB and would not fit. That was based on a mis-derived shape; the real
figure is 76 KB and it fits fine.)

The measured firmware reflects this: `bench` links with **BSS 77,156 bytes**,
and the model struct is 65,744 bytes.

Hardware-side, note the board has **no audio codec**. The reference project is a
USB-audio *loopback* (host audio in → model → host audio out), which is
driverless and needs no extra parts. A guitar-facing build needs an I2S ADC/DAC
(e.g. PCM5102A for out, PCM1808 for in) wired to the stamp holes.

## Measured, on the device (2026-09-23)

`bench` builds, flashes and runs. RP2350 rev A4, `clk_sys` 150 MHz (stock),
single core, 1407 MACs/sample, 3125 cycles/sample of budget at 48 kHz.
Reproducible across runs (these numbers repeated to within 1 cycle):

| kernel | cycles/sample | % of one core | fits 48 kHz? |
|---|---:|---:|---|
| float scalar (`-O3`) | 21,600 | 691% | no |
| int16 Q15 scalar | 9,306 | 298% | no |
| int16 Q15 + SMLAD, operands packed at use | 10,063 | 322% | no |
| Q15, gather to planar staging | 8,806 | 282% | no |
| Q15, staging + SMLAD | 7,541 | 241% | no |
| **Q15, gather straight to packed pairs + SMLAD** | **6,362** | **204%** | no |

These are the 2026-09-24 figures, after the ring-size lookup fix described
below; the pre-fix numbers were 21,778 / 9,307 / 10,063 / 8,915 / 7,566 / **6,451**.

The whole progression, in the order the measurements were taken, is plotted in
[`kernel-progress.png`](kernel-progress.png) (and `.svg`): the fall from 697% of
one core to 204% during the kernel work, then the plateau at ~206% where every
later experiment was either a regression that got reverted or a fraction of a
percent.

Two negative controls, added later rather than as candidates:

| variant | cycles/sample | % of one core | why it is here |
|---|---:|---:|---|
| duplicated ring (no wrap fixup) | 7,062 | 226% | removes 27% of the instructions and loses 9.5% |
| weights in pinned XIP cache lines | 10,255 | 328% | cache-as-SRAM: works, and costs 60% |

### The ring-size lookup: 1.4%, and a lesson about where the reads come from

`nam_ring_size[]` is `static const`, so it lives in flash and every read of it is
an XIP access through the 16 kB cache both cores share. The same number is already
in SRAM as `layer[l].ring_size`, written there by `nam_model_init`, so the lookup
bought nothing - and the float kernel was doing it *inside* its `(oc, ic)` loops,
nine XIP reads per layer per sample, shadowing an outer variable that already held
the value. Replacing all of them with the struct field gives:

| kernel | before | after |
|---|---:|---:|
| `packed+SMLAD` | 6,451 | **6,362** (-1.4%) |
| staged | 8,915 | 8,806 (-1.2%) |
| staged+SMLAD | 7,566 | 7,541 (-0.3%) |
| float scalar | 21,778 | 21,600 (-0.8%) |
| q15 scalar, q15 SMLAD, framemajor, both ablations | - | unchanged to the digit |

The attribution is clean precisely because of the last row: the kernels that never
read that table did not move at all, so this is the intended effect rather than the
code-layout luck that has moved numbers by a percent before.

**It does not help the two-core case.** The flat split at b = 13 measured 3,679
cycles/sample of wall clock before the fix and 3,676 after: removing 1.4% of each
core's work left the two-core wall where it was. That is worth sitting with,
because it is not what "each core runs at 90% while the other is active" predicts -
that model says the wall should have fallen by 1.4% too. Whatever the second core
costs, it is not proportional to the work each core is doing, which is consistent
with the four placement nulls above and points the same way: a shared limit that
instruction count does not move.

Reference points, measured on different code at 300 MHz with both cores:

| | cycles/sample | % of one core |
|---|---:|---:|
| generic Eigen | 31,030 | 496% |
| `a2_fast` | 8,396 | 134% |
| `a2_fast` + dual core | 4,533 | 73% |

### What the kernel work established

The "planner" idea works, and the attribution is clean because every variant was
measured separately rather than assumed:

- **Addressing, not arithmetic, was the cost.** Resolving each tap's ring
  position per MAC (`p = idx + off[t]`, compare against zero, fix the wrap,
  scale by the channel count) is 4–6 instructions of overhead per
  multiply-accumulate. Pre-multiplying the offsets and specialising on the tap
  count took plain Q15 from 16,693 to 9,255.
- **Gathering to contiguous staging is worth little on its own** (9,255 →
  8,968, 1.03×). Just removing the address math is not enough.
- **The DSP extension only pays when the operands are already adjacent.**
  SMLAD with operands packed at use is 1.08× *slower* than plain scalar — the
  packing costs more than the second MAC saves. Packing during the gather
  instead, so the MAC loop is one 32-bit load per two MACs, gives 7,541 and then
  6,362. That is the whole result: **1.46× over the best scalar version, and
  2.04× over where this started.**
- **2.04× more is needed.** 6,362 cycles/sample implies about 305 MHz for
  48 kHz on one core, against a 150 MHz stock clock.

The Q15 path agrees with the float reference to 4.0e-4 worst case across every
variant, verified at six run lengths from 240 to 8064 samples, so none of this
is trading correctness for speed.

### Honest scope of the conclusion

2.04× short is a long way, and it is worth being clear about what would close
it. This kernel is a straightforward C implementation with no residual stream,
no FiLM conditioning and no input mixin — that is, it is already *less* work per
sample than a faithful A2-Lite, so the real target is somewhat worse than these
numbers suggest. Closing 2× on top of that would need at least one of:

- **A reduced receptive field.** Cost is proportional to taps × dilations, so
  halving the receptive field roughly halves the work. The dilations
  `[1,3,7,17,41,101,239]` are not sacred; 131.9 ms of context is generous for a
  guitar amp.
- **A lower internal sample rate.** Running the stack at 24 kHz and
  interpolating halves the per-sample work at the cost of bandwidth.
- **Both cores**, which is what the reference project did — and this is now
  measured rather than projected: 2.04 / 2 ≈ 1.02 for a perfectly balanced split,
  and **1.18 with the contention two cores actually have**. It does not fit at stock clock. It needs ~177 MHz, or the kernel 15–18%
  faster (15.3% if the second core's 1.80x capacity holds as the kernel shrinks,
  17.7% if its tax scales with the work, which is the likelier of the two).

That last point is the real finding: **the single-core goal is not reachable by
kernel tuning alone, and two stock-clocked cores are not enough either** — the
split is 2.2% short on arithmetic and 16% short in practice. What the kernel does
buy is a 2x clock reduction at the same core count, which is where the remaining
work has to come from.

### Remaining headroom, if this is picked up again

- **The instruction-count lever was tried and it loses.** This is the most useful
  negative result in the project, because it closes off the whole family of
  "make the gather cheaper to *encode*" ideas in one measurement.

  `nam_process_packed_dup` keeps each layer's ring twice, back to back, and
  writes every sample to both copies. The read base then sits one ring ahead, so
  a tap at lag `t*d` lands in `[1, 2N-1]` — always in bounds, never needing a
  wrap fixup, and always the right sample. No mask, no branch, no byte rescale.
  The disassembly confirms it compiled as intended:

  | | instructions | `it` blocks | cond. adds | loads | `lsl` | stores |
  |---|---:|---:|---:|---:|---:|---:|
  | `packed+SMLAD` | 975 | 31 | 22 | 367 | 33 | 80 |
  | `packed+dupring` | **706** | **4** | **2** | **286** | **8** | 98 |

  **27% fewer instructions, 22% fewer loads, all 22 wrap fixups gone — and it
  costs 6,451 → 7,062 cycles/sample, 9.5% *slower*.** It passes `kverify` at
  every run length, so it is a real kernel, just a slower one. The extra memory
  traffic is what did it: six more stores per layer per sample (138 per sample)
  against the instructions saved.

  So the earlier reasoning in this document — "the gather is 68% of the cost and
  costs 7–9 instructions per pair, therefore remove instructions" — was wrong in
  its premise. Removing instructions from this kernel does not make it faster,
  and neither does removing loads. It is bound by memory traffic and access
  latency, not by issue slots. That single fact retro-explains most of the
  negatives already recorded here: the linear ring (3.3x worse; it copied ~5,483
  bytes per sample), SRAM-resident code (instruction fetch competes with data on
  the same port), `framemajor` (accumulator traffic), input-register reuse
  (neutral — it removed loads that were not the bottleneck). The two-core tax
  belongs to the same family — it is memory-side, since the RAM twin changed
  nothing — but it is the one member of it that no amount of moving data or code
  has been able to reach.
- **Therefore the planar power-of-two ring is no longer recommended.** It was
  estimated at ~6,480 → ~4,980 cycles/sample on instruction count alone. It is
  the same *kind* of change as the duplicated ring — fewer instructions, no
  branch, same number of bytes moved — so the honest expectation is neutral to
  worse, not 23% better. The measured 9.5% regression is the evidence; the
  estimate above it is now retracted rather than pending.
- **What is left is reducing bytes moved per sample**, which is a structural
  question rather than a micro-optimisation: fewer taps, a smaller receptive
  field, or a lower internal sample rate. The three original options are
  unchanged and none of them is a rewrite of the inner loop.
- **Two cores are still worth having**, and the requirement is a kernel at or
  under 5,625 cycles/sample — 15% off, not the 3.7% the arithmetic floor alone
  suggests. The second core delivers 1.80x, not 2x, and it is the same memory
  port that limits both.
- **Residual and mixin layers were omitted**, so adding them back costs work;
  the head and any conditioning are currently near-free.
- Layer fusion is the wrong shape here: each layer's output feeds the next
  sample's input at a different dilation, so there is no small tile to fuse.
- **Resolved since this list was written:** a finer multi-boundary split does not
  help (97.7% balance at a single boundary, and the arithmetic floor is above
  budget); the hot loop in RAM does not help, with or without a second core; and
  the SMLAD path in RAM is not a fix for the two-core contention.

## Hardware features investigated, and what they are actually worth

Measured, not reasoned about. Several of these look promising in the datasheet
and are dead ends for this workload.

### Running the kernels from SRAM: **slower**

`__not_in_flash_func` moves code into SRAM, which the datasheet's timing section
suggests should help (SRAM is single-cycle; XIP is not). Measured, it costs
**~2.5%**: packed+SMLAD went 6477 → 6636 cycles/sample, and aligning the entry
points to 16 bytes to dodge the bank-conflict phase changed nothing (6636 again).

The reason is in the same datasheet section (§3.7.4.9.12): the contention is
between instruction fetch and data load *to the same memory*. This kernel's data
— ring pool and weights — already lives in SRAM, so moving the code there puts
both streams behind one port. Executing from flash keeps instruction fetch on the
XIP path with its own 16 KB cache, leaving SRAM entirely to data. The experiment
is preserved as a switchable `NAM_PROCESS` macro in `kernels.c` rather than
deleted, so it can be re-run rather than re-argued.

### SRAM striping and the small 4 KB banks: **automatic, no action available**

Worth knowing because it is often suggested as a tuning knob:

- **Striping is automatic and not selectable.** SRAM0–3 and SRAM4–7 are *always*
  word-striped on address bits 3:2 across four banks each; there is no
  non-striped mirror on RP2350 (RP2040 had one). You cannot place data to get
  striping — you already have it.
- **All 520 KB is single-cycle.** The datasheet is explicit that the dedicated
  SRAM8/9 blocks exist to *guarantee* no contention, which only matters when a
  second manager (a DMA engine, or core 1) is hitting the same bank. On a single
  core doing no DMA, this buys nothing.
- **Bank conflicts with the instruction prefetcher are real but tiny**: the
  prefetcher runs ~8 bytes ahead, so a load that lands 8 (mod 16) bytes ahead of
  the instruction stream can cost one cycle. That is the effect the alignment
  experiment above tried and failed to exploit.

### The hardware interpolators: **dead end for this kernel**

The SIO interpolators do masked/shifted accumulation plus optional blending and
clamping, in one cycle per access — genuinely useful for resampling, LUT
addressing and affine texture mapping, which is what the datasheet lists them
for. This kernel has **no lookup tables and no interpolation**: it is a dense
fixed-shape MAC stream. The natural candidate would be the LeakyReLU
quantisation, but that is already `asr`, `sdiv`, and a compare, executed 69 times
per sample (23 layers × 3 channels) — and reaching the interpolator would mean
writing ACCUMx/BASEx and reading POPx per use, i.e. more memory-mapped register
traffic than the arithmetic it replaces.

### The hardware divider: **already used, and better than a reciprocal**

`pico/divider.h`, and the M33's `SDIV` more generally, looked like an easy win for
the `/ 100` in LeakyReLU. Disassembly shows GCC already emits `SDIV` for that
constant divide, and replacing it with a multiply-by-655-and-shift measured
**57 cycles slower** (6534 vs 6477), reproducibly: there is nothing to win, and
the multiply form lengthens the dependency chain. Left as a documented
`USE_RECIP` switch defaulting to the faster hardware divide.

### Ring-wrap by masking: **abandoned, and the reason is instructive**

Replacing the gather's `p < 0 ? p + wrap : p` with `p & (ring-1)` looks like a
free branch removal — at ~1,400 taps per sample it would be the largest per-tap
saving available. It was implemented twice and abandoned, because the ring
geometry makes it a bad trade rather than a hard problem.

The constraint chain:

- Masking only equals modulo when the mask is `2^k - 1`, and the ring is
  addressed in **interleaved** space (`idx * NAM_CHANNELS + off`). `2^k - 1` is
  never a multiple of 3, so masking an interleaved index lands on the wrong
  channel lane. First attempt: error grew with run length (0.0005 → 0.0058) —
  caught by `kverify`, not by inspection.
- Making the mask `3·2^k - 1` fixes the lane alignment, but then the write index
  must wrap at a *different* modulus than the reads are masked at, so the
  padding slots become readable as stale data. Second attempt: error 0.014.
- The clean version is to make every ring's *minimum* a power of two, so mask and
  wrap coincide. But the 1196-sample rings then round to 2048 and the allocation
  triples: the pool goes from **38 KB to 168 KB**, a 4.4× increase, to remove one
  branch per tap.

That is a bad exchange on a 520 KB part for an unmeasured branch cost, and the
branch is well-predicted anyway (it is taken at the same points every sample).
Reverted, and the numbers confirmed it exactly: back to 0.000502 validation error
and 6,473 cycles/sample. Recorded here so it is not re-attempted.

The cheaper-looking version of this idea — lay each layer's ring out **planar**
(three separate per-channel rings) so the mask applies to a sample index rather
than an interleaved one, with the minimum rounded to a power of two, ~46 KB
rather than 38 KB — is now *retracted rather than pending*. The duplicated ring
tested the same hypothesis more strongly (it removes the branch without any mask
or padding at all, and removes the byte rescale as well) and lost 9.5%. Removing
this branch is not worth anything, because the branch was never the cost. See
"Remaining headroom" at the end of this document.

### The 11% two-core tax is not addressable: four null results

The dominant remaining term is not the split - a perfectly balanced split is
104.0% of real time - but the ~11% the second core costs. `wall` fits
`max(T_A, T_B) + 0.11 * min(T_A, T_B)` across nine boundaries, i.e. each core runs
at about 90% of its solo rate while the other is active. Four separate attempts to
move that number, all measured on the device:

| what was varied | configurations | result |
|---|---|---|
| SRAM **bank phase** of the second core's rings | pool shifted by 0, 1, 2, 3 interleaved slots (6 bytes each; `(6>>2)&3 == 1`, so each value is the next bank) | **3,679 cycles/sample at every phase** |
| **bank group** of the second core's data | same 256 kB region / different region / literally the same words (`mem2`, 3 x 16 kB buffers) | 68.000M, 68.000M, 68.000M cycles - identical |
| **fetch path** (flash vs RAM) | both flash, both RAM, core0 flash + core1 RAM, and the reverse | **3,680 cycles/sample in all four** |
| where the two **stacks** live | - | already separate: the SDK puts core 1's stack at the top of SCRATCH_X (SRAM8) and core 0's at the top of SCRATCH_Y (SRAM9), which are different non-striped banks |

The bank-phase probe is worth a note on its own, because the first version of it
was not a valid test and the null it produced was misleading. A uniform shift
permutes which bank an access lands in, but each core's stream already walks over
all four banks, so two such streams collide with the same probability at every
phase. The only way to test banking is to give the cores memory that stripes over
different banks, which is what `mem2` does - and it comes out flat too, including
the case where both cores hammer the *same words*.

So the datasheet's bandwidth-partitioning advice ("allocate data across two
256 kB blocks of 4-way-striped SRAM") is sound in general and **does not apply
here**: this kernel's data streams are spread across all banks by their own
stride, so what remains is arbitration in the shared bus fabric - which no
address choice can touch.

The one thing this does buy is a sharper statement of the requirement: since the
tax is not addressable, the whole 15-18% has to come out of the kernel itself.

### Cache-as-SRAM in the XIP cache: **works, and is 60% slower**

The datasheet (sections 4.2.1, 4.4.1, 4.4.1.3) offers the XIP cache as extra
memory: "Cache lines can be individually pinned within the XIP address space for
use as SRAM, up to the total cache size of 16 kB", and because the QMI only
decodes the lower half of the 64 MB XIP space, lines can be pinned "outside of
the QMI address range (e.g. at the top of the XIP space)". It is a fair question
whether that memory is *faster* than the main SRAM, and the answer is no - but
the way it fails is worth recording, because the mechanism is not the obvious one.

**It is not faster, and cannot be.** The datasheet gives the cache as "16 kB,
two-way set-associative, **1 cycle hit**", and the SRAM as "single-cycle access
from all managers". Both are already at the ceiling the core can use: the M33 has
one load/store unit and issues at most one data access per cycle. There is no
faster memory on this part to move anything to.

What the cache *is* is a different port - the XIP/AHB path rather than the SRAM
banks - and that is the only reason it was worth testing here, because the two
things this project measured are that the kernel is bound by memory traffic
rather than instructions (the duplicated ring cut instructions 27% and lost 9.5%)
and that a second core loses 11% to the first on a memory-side effect.

`nam_process_packed_wxip` tests it properly: pin 6,624 bytes of cache (828 of the
1024 sets in way 0), copy the packed weight tables into it, and read them from
there. The weights are the right candidate - 6.6 kB, read-only after init, read
621 times per sample, about a third of the kernel's memory operations.

*It works.* The kernel passes `kverify` at all six run lengths with the same
error profile as `packed+SMLAD`, so the weights really are living in pinned cache
lines and the read-back check confirms it.

*It is 10,289 cycles/sample against 6,451 - 60% slower.* The reason is that the
cache is not a separate memory: it is the same 16 kB that is already serving
instruction fetch for a kernel that executes from flash. Pinning 6.6 kB of it
takes those sets away from the code, and every weight load then competes with
instruction fetch on the XIP path. It is a second port, but a port that was
already busy with the thing this kernel can least afford to slow down.

The USB data DPRAM is the other block the datasheet mentions, and it fails for
simpler reasons: 4 kB is too small for the weights, let alone the 38 kB of rings,
it requires USB to be disabled (the console is USB), and it sits in the AHB
peripheral segment where the datasheet allows a wait state, so it is no faster
than SRAM either.

**What the datasheet does offer, and has not been tested here:** the two 256 kB
striped regions stripe over *disjoint* SRAM banks (SRAM0-3 are banks 0-3, SRAM4-7
are banks 4-7), and the datasheet points at this directly - "you can still
achieve some explicit bandwidth partitioning by allocating data across two 256 kB
blocks of 4-way-striped SRAM". That is the documented way to stop two cores
colliding in a bank, which is the shape of the 11% two-core tax. The other is
SRAM8/9, the two non-striped 4 kB banks, documented for "per-core purposes ...
guaranteeing that the processors never stall on these accesses" - though 4 kB is
under the 6.6 kB of weights and far under the rings.

### The double-precision coprocessor (DCP): **exists, but 17 cycles per multiply**

An earlier note here claimed the M33 "has no double-precision FPU so doubles trap
to software". **That was wrong**, and the datasheet is explicit: RP2350 gives each
core a *double-precision coprocessor* (§3.6.2) that accelerates add, subtract,
multiply, divide and square root, and `pico_double` drives it —
`sdk/src/rp2_common/pico_double/double_fma_dcp.S` is real DCP code, not a software
fallback.

The actual reason doubles are wrong here is throughput, from the datasheet's own
table (§3.6.2.10):

| operation | RP2350 DCP | full hardware FPU | software only |
|---|---:|---:|---:|
| `dadd` / `dsub` | **6** | 2–6 | 70–90 |
| `dmul` | **17** | 3–7 | 75–90 |
| `ddiv` | **51** (32 fast) | 13–60 | 135–600 |
| integer ↔ double | **5** | — | — |

Each of those needs a canned sequence of MCRR/CDP/MRRC instructions rather than a
single opcode, and `dmul` at 17 cycles is **17× the 1 cycle** of a single-precision
`VMUL.F32` on the same core. This kernel is multiply-dominated: 1,407 MACs per
sample would cost ~24,000 cycles in double for the multiplies alone, against a
3,125-cycle budget. So fixed-point (or at most `float`) remains right — but for a
throughput reason, not an availability one.

Where the DCP *would* be worth using: anything off the hot path that wants real
double precision — validating the fixed-point kernel against a double reference,
or model conversion — is cheap enough at 5–17 cycles per operation and worth
preferring over software routines.

### Techniques ported from NAMBench's SLIMMED-PATH study: three more negatives

This repo's own M2/NEON kernel study ([`SLIMMED-PATH.md`](../../SLIMMED-PATH.md))
built a dozen kernel variants for Apple silicon and found a clear winner:
`stacked_linear` at 2.098× over `a2_fast`, built on **planar layout (1.18×)** and
**wide frame tiling (1.75× at width 32)**. Those techniques were ported here and
measured. All three lose, and the *reasons* are the useful part — they are all
consequences of the M33 having half the SIMD width and a much smaller register
file than the M2.

| technique | M2 result | RP2350 result | why it inverts |
|---|---:|---:|---|
| wide frame tile (W=4) | **1.75×** (W=32) | **0.61×** (10,573 vs 6,473) | needs 12+ live accumulators plus operands and weights; ARM has ~13 GPRs, so it spills. On M2 a *vector* register holds 4 frames; on M33 a register holds one packed tap pair, so a W-frame tile is W× the live state for no width gain. |
| linear ring (no wrap) | 1.14× | **0.31×** (21,091 vs 6,473) | M2's win was over *lazy mirroring*, which is per-tap and branchy. RP2350's wrapped ring has no mirror and no copy — just a predicated load. The linear ring must copy `sum(lookback) × 3ch × 2B ≈ 5,483 bytes per sample` versus 6 bytes written, so it trades a cheap branch for real memory traffic. |
| framemajor (partial accumulators) | 0.61× | **0.26×** (24,873) | a partial accumulator per (channel, tap) is deliberately too much live state. The M2 study called this its worst case; it is worse here, as predicted from a smaller register file. |
| duplicated ring (added later, and the control for the row above) | — | **0.91×** (7,062 vs 6,451) | the mirror image of the linear-ring result, and it confirms the same rule from the other side: the duplicated ring *removes* all 22 wrap fixups and 27% of the kernel's instructions, and loses 9.5% anyway because it adds 138 stores per sample. Instructions are not the currency here; traffic is. See "Remaining headroom". |

`framemajor` was included precisely as a negative control, and it reproduced the
M2 failure mode at higher severity — which is a useful cross-check that the two
studies are measuring comparable things rather than that one harness is broken.

**Two measurement mistakes were made on the way to the linear-ring number**, both
worth recording because they would have produced a false conclusion:

1. Sizing the buffer as `lookback + tile + 1` gave layer 0 a capacity of 10 with a
   lookback of 5, so it rewound almost every sample: 64,859 cycles/sample, 10×
   worse than it should have looked. A technique has to be sized to amortise
   before it can be fairly judged — the M2 study's slack is implicitly
   `block × 19`.
2. `memmove()` measured *far* worse than a hand-rolled copy loop on this part
   (38,356 vs 21,091 cycles/sample). The first two linear-ring measurements
   therefore overstated the loss by ~2×.

Even after fixing both, the technique loses by 3.3×, so the conclusion is
structural rather than an implementation artefact. The kernel is **removed**;
the reasoning is kept in `src/kernels_exp.c`.

The duplicated ring added later is the same finding from the opposite direction:
it *removes* the branch and reduces instructions, and still loses, because it
adds traffic. Between the two, the rule is now measured in both directions —
what costs on this part is bytes moved, not instructions issued.

### The all-float kernel (from the ESP32-S3 write-up): 2.8x slower, so fixed point stays

[Taurus's ESP32-S3 write-up](https://playtaurus.com/blog/running-nam-a2-lite-natively-on-an-esp32-s3)
is the most useful external datapoint found: they built a 16-bit integer engine,
spent weeks tuning it, and were then beaten by *"writing better floating point"* -
the float build that ships is bit-identical to the desktop render. Their integer
engine was only ~20% faster than "a well-optimised float build", and the qualifier
is the whole point.

That prompted a fair test here, because **this project's earlier float kernel was
not well-optimised**: it kept state in Q15 and converted int16 -> float on every
tap, paying 1,407 conversions per sample for nothing. A proper float kernel -
fp32 ring state, fp32 weights, so a tap is a plain load plus `VFMA.F32` - was
written and measured:

| kernel | cycles/sample | accuracy vs float reference |
|---|---:|---|
| `packed+SMLAD` (fixed point) | **6,362** | 3.99e-4 (Q14 quantisation) |
| float state (fp32 rings, fp32 weights) | 18,104 | **bit-identical** (err 0.000000) |

So on RP2350 the ESP32-S3 result inverts: 2.81x *slower*, not 20% slower.

(This kernel read 18,334 before `packed+dupring` was added to the same file.
Adding a kernel to `kernels_exp.c` moved it by 1.3% with no change to its
source, which is worth knowing when comparing numbers across builds: the
shipped kernels have reproduced to the digit for several rebuilds, but a
*losing* kernel can drift by a percent or so from code layout alone.)

The reason is the mirror image of theirs. The ESP32-S3 has 128-bit **integer**
vector instructions and no vector path into its FPU, so float "goes one multiply
at a time while integers go sixteen". RP2350's M33 has 2-lane integer DSP
(`SMLAD`, two MACs per instruction) *and* a single-precision FPU (`VFMA.F32`, one
per instruction) - but nothing wider for float. So the ratio is 2:1 in
instructions, doubled again by 32-bit loads for float against 16-bit operands on
the packed integer path, and worsened further by floats occupying two registers
each.

The compensation is accuracy: the float path is **bit-identical to the reference
engine**, where Q15 costs 4e-4. Whether that trade is worth 2.84x is a modelling
decision, and 2.84x is too much to pay for it here.

**The transferable lesson from that article is not float-versus-int at all**, it
is fixed-point *scaling discipline*, and it applies directly if this becomes a
scaled-integer engine:

- do not round quantisation scales down to powers of two "because it is a shift" -
  only one shift per layer has to be a power of two; the rest are multiplies and
  can take exact values;
- give each layer its own exponent, driven by a **sliding peak across the history
  the convolution actually reads**, not the current block's peak - scaling on the
  block's peak guarantees clipping immediately after a loud passage;
- let the current block only ever pull the exponent *down* (the direction that
  cannot overflow), and add an explicit retreat rule so a layer cannot park at a
  clipping exponent;
- raise the input by one bit (a guitar DI never reaches full scale, so half the
  input range is reserved for headroom that is never used) but leave the output
  alone (an amp's output does pass unity on transients).

The kernel here is unscaled Q14 with no per-layer exponents, so all of that is
available - worth ~26 dB of quantisation error in their measurement. It is an
accuracy project rather than a speed one.

### Compared with the reference project

Their published numbers against these, like for like:

| | cycles/sample | clock | % of one core |
|---|---:|---:|---:|
| reference generic Eigen | 31,030 | 300 MHz | 496% |
| reference `a2_fast` | 8,396 | 300 MHz | 134% |
| reference `a2_fast` dual-core | 4,533 | 300 MHz | 73% |
| **fpi `packed+SMLAD` (this work)** | **6,362** | **150 MHz** | **204%** |
| fpi dual-core, measured (see below) | 3,676 | 150 MHz | 118% |

The kernel itself is **1.30x better than their tuned `a2_fast`** in the sense that
matters - fewer cycles for the same network on the same silicon. Projected onto
two cores at *stock* 150 MHz it lands at 118% of real time, which is *not* enough -
an earlier note here estimated ~112% by assuming their 1.85x speedup would carry
over, and it does not: the measured speedup is 1.76x, and half of the whole model
is 102.2% of real time before any split inefficiency at all. Their dual-core
result is at 300 MHz, and the honest comparison is that this kernel buys roughly a
2x clock reduction on the same core count, not a pass.

Where the conclusions agree: they tested SRAM placement (2.7% *slower*) and LUTs
(not applicable - LeakyReLU is already inlined) and rejected both, and the same
two measurements here reached the same answers independently. They also
established that 300 MHz at 1.20 V is stable while 400 MHz hard-faults, which is
the useful headroom fact if this kernel is taken to hardware.

## Why two cores: it is throughput, and the split costs no latency at all

The reference project pipelines the layer stack across both cores at block
granularity and notes this "adds one block (about 1 ms) of latency". That is
easy to misread as a per-block penalty that would cause a dropout every
millisecond. It is not, and it is worth being precise about the two different
quantities, because they lead to opposite conclusions about whether one core
could ever do:

- **Latency**: the delay between an input sample arriving and its output
  appearing. This is a *fixed* offset, dominated by the receptive field — 6,332
  samples, 131.9 ms, which no implementation choice removes.
- **Throughput**: samples processed per second. This is what actually decides
  whether audio works. A UAC2 device must deliver one packet per 1 ms USB frame,
  every frame, indefinitely. A core that needs 8,396 cycles per sample at 300 MHz
  has a 6,250-cycle budget and is 34% *over* it - not late once, but permanently
  1.34 ms of work per 1 ms of deadline. That deficit never clears; it accumulates
  as missed packets, which is periodic audible dropout.

So the "1 ms late" is a one-off warm-up, and the reason two cores are needed is
the second effect: the work per sample has to fit the budget. **At 6,362
cycles/sample against a 3,125 budget this kernel is 2.04x over, so the deficit is
real and the fix has to be throughput.**

### The correction: a layer pipeline adds *no* latency

An earlier version of this note said the pipeline's extra block of latency is
"the price" of the throughput. That is wrong, and the measurement is in
`fpi/src/duo.c`: fill the pipeline with one block and time how long that block
takes to come out the far end.

| first result, one 480-sample block | cycles | ms |
|---|---:|---:|
| one core, whole model | 3,111,246 | 20.74 |
| two cores, split at layer 13 | 3,168,652 | 21.12 |
| difference | +57,406 | **+1.8%** |

+1.8% is synchronisation and bus contention. It is **not** a structural cost,
because the first block has to traverse all 23 layers either way: a two-stage
pipeline's fill time is `T_A + T_B = T_total`, exactly what one core takes. The
pipeline changes the *period* between blocks (`T_total` → `max(T_A, T_B)`) and
leaves the latency alone.

The useful generalisation, which resolves the two intuitions:

- **Spatial (within-block) decomposition** — every block is processed by both
  cores, each taking part of the layer stack — leaves latency at `T_total` and
  improves *only* throughput. This is what the split does.
- **Temporal (across-block) decomposition** — core 0 takes even blocks, core 1
  odd — is the one that adds a whole block of latency, and it is the one the
  network forbids: the dilation-1 layers are serially dependent on the previous
  frame.

So "we get throughput from the second core and pay latency for it" is not the
trade being made. There is no latency to pay.

## Is the per-block processing really sequential?

Mostly yes, but not entirely, and the exceptions are informative.

Within one frame, the 23 layers form a genuine chain: layer *l* at frame *n*
needs layer *l-1* at frame *n*. Nothing can break that - it is the network.

Across frames there is real slack, and it depends on dilation. A layer with
dilation *d* has `out(n)` depending on `out(n-d)`, so frames closer together than
*d* are serially dependent *within that layer*:

| layer class | count | share of MACs | frames parallel? |
|---|---:|---:|---|
| dilation == 1 (layers 0, 7, 14, 16) | 4 | **21%** | **no** - `out(n)` needs `out(n-1)`, strictly serial |
| dilation > 1 | 19 | 79% | yes, frames ≥ *d* apart are independent |

So there is a **serial spine of 21% of the work** that no amount of frame-level
parallelism can touch; the other 79% can in principle be spread across frames at
distance ≥ *d*, and the receptive field (6,332 samples ≈ 132 blocks) is how much
history is available to spread over.

That is why the layer-split pipeline is the right shape rather than a
frame-split one, and why its 1.85x is close to the 2x ceiling: the split is
across the *layer* axis, which has no such serial spine. A frame-level split
would hit the dilation-1 layers almost immediately.

The one genuine additional opportunity was a finer split: the pipeline's single
boundary leaves whatever imbalance the layer granularity forces, and the issue's
`process_partitioned` already supports several boundaries with `{6,12,18}`. That
has now been measured — see the next section — and the answer is that balance was
never the problem.

## The fine-grained split, measured

`fpi/src/duo.c` (target `duo`, `cmake --build build --target duo`) implements the
split for real rather than projecting it: the winning kernel restricted to a
layer range `[l0, l1)`, core 0 running `[0,b)` and core 1 running `[b,23)`, a
per-block handoff over shared SRAM flags with `__sev`/`__wfe`, and a sweep over
all 24 possible boundaries. Every split output is compared against the
single-core run bit for bit, and at every boundary it matches.

The split is verifiable here because of a property of this benchmark rather than
of NAM: a layer only ever reads its own ring and the input sample is injected
into every layer, so the 23 layers are independent filters and `[0,b) + [b,23)`
computes exactly what `[0,23)` computes, on disjoint rings. A real chained NAM
would need the same ranges delivered as a double-buffered pipeline; the period is
`max(T_A, T_B)` either way, so the cost measurement carries over, but the
"bit-identical" check does not.

### Choosing the boundary

`T(0,b)` and `T(b,23)` measured independently on one core, 480 samples, best of
5. `per-layer` is the marginal cost of layer *b-1*, which is the more interesting
column. (Taken before the ring-size fix; the post-fix halves read about 1% higher
through the range copy. The balance column, which is the point of the table, is
unaffected.)

| b | A `[0,b)` | B `[b,23)` | max | balance | per-layer |
|---:|---:|---:|---:|---:|---:|
| 10 | 2,525 | 4,049 | 4,049 | 62.4% | 245 |
| 11 | 2,766 | 3,809 | 3,809 | 72.6% | 241 |
| 12 | 3,011 | 3,564 | 3,564 | 84.5% | 245 |
| **13** | **3,249** | **3,325** | **3,325** | **97.7%** | 238 |
| 14 | 3,500 | 3,075 | 3,500 | 87.9% | 251 |
| 15 | 4,128 | 2,447 | 4,128 | 59.3% | **628** |
| 16 | 4,753 | 1,821 | 4,753 | 38.3% | **625** |

Two things fall out of the marginal column. A 6-tap layer costs ~244
cycles/sample and a 15-tap layer ~626 — the 2.5× MAC ratio almost exactly — and
**the cost does not depend on dilation at all**: a 239-dilation ring is further
apart in memory, not more expensive to read. The two 15-tap layers are the only
non-uniformity in the stack.

`b = 13` is very nearly perfectly balanced, and the reason is arithmetic rather
than luck: layers 0–12 are 13 six-tap layers = 702 MACs, and layers 13–22 are 8
six-tap layers plus the two 15-tap ones = 702 MACs. The layer boundary that
halves the MACs halves the time. The residual 2.3% is the per-sample loop
overhead, which the first range pays once and the second does not.

So the answer to "would a fine-grained multi-boundary split help?" is **no**:
layer granularity already reaches 97.7% balance, and the multi-boundary machinery
would be buying the remaining 2.3%.

### The result

32 blocks × 480 samples per run, 150 MHz, all figures cycles of wall clock per
sample. The budget is 3,125. **Taken before the ring-size fix** (see the caveat
under the decomposition: the wall times moved by 3 cycles/sample, so the round
numbers below are still the right ones to read).

| configuration | wall cyc/smp | max(A,B) | wall/max | % of RT | speedup |
|---|---:|---:|---:|---:|---:|
| b=0 (control: core 0 idle) | 6,569 | — | — | 210.2% | 0.99× |
| b=10 | 4,311 | 2,525 | 1.71× | 138.0% | 1.50× |
| b=12 | 3,851 | 3,011 | 1.28× | 123.2% | 1.68× |
| **b=13 (best balance)** | **3,679** | **3,249** | **1.13×** | **117.7%** | **1.76×** |
| b=14 | 3,904 | 3,500 | 1.12× | 124.9% | 1.66× |
| b=16 | 4,984 | 4,753 | 1.05× | 159.5% | 1.30× |
| b=23 (control: core 1 idle) | 6,568 | — | — | 210.2% | 0.99× |
| b=13, hot loop in RAM | 3,679 | 3,249 | 1.13× | 117.7% | 1.76× |
| b=13, pipelined handoff | 3,774 | 3,249 | 1.16× | 120.8% | 1.72× |

The two controls return 0.99× exactly, which is what makes the rest believable:
with one core idle there is no speedup to be had, and the harness does not invent
one. The pipelined variant — the one a chained model needs — costs 2.6% over the
free-running split, which is a fair price for the handoff.

**Two cores at stock 150 MHz land at 118% of real time.** Not enough.

### Why not: the decomposition

Four numbers, each the best case given the one above it.

| | cyc/sample | % of RT | |
|---|---:|---:|---|
| one core, whole model | 6,387 | 204.4% | the kernel as shipped |
| two cores, perfectly balanced | 3,193 | **102.2%** | total work ÷ 2 |
| two cores, best real boundary | 3,286 | 105.2% | see the caveat below |
| two cores, measured | 3,676 | 117.6% | what actually happened |

The first line is the finding. **Half of the whole model is 102.2% of real time,
so no division of the work can reach 100%** — the kernel has to be faster before
any split can help, and that was already the conclusion from the single-core
work. Splitting is not the lever it looked like.

The gap between the last two lines is the second core's cost, and its shape is
informative. `wall / max(A,B)` is not constant across the sweep: it falls from
1.71× at b = 10 to 1.05× at b = 16. A fixed multiplicative tax would give a
constant ratio, so it is not one. The whole sweep fits

```
wall ≈ max(T_A, T_B) + 0.11 · min(T_A, T_B)
```

to within 2% at every boundary: each core loses about 11% of its solo rate while
the other is active, and the *smaller* half's lost time is what appears as excess
wall clock. When the halves are balanced, all of that loss lands on the critical
path, which is the 1.12× at b = 13.

That model describes the sweep well and predicts nothing else. Four separate
attempts to move the number are recorded in the next section and none of them did
— including removing 1.4% of each core's work, which left the wall at 3,676.
Whatever the second core costs, it is not proportional to what the cores are
doing, and it is not located anywhere an address can reach.

**Two cores at 150 MHz need 177 MHz, or a kernel 15–18% faster.**

One caveat on the third row. The wall times compare like with like inside a
single build, but the halves are measured through `nam_process_packed_range`, a
second copy of the kernel whose code generation drifted about 1% the other way
when the ring-size fix changed the file. So 3,286 overstates the split's
inefficiency slightly; the pre-fix pair was 3,249 against a 6,483 whole, and the
98% balance figure comes from the sweep rather than from this row.

### Where the time goes: the gather, not the MACs

An ablation of the winning kernel splits its cost:

| variant | cycles/sample |
|---|---:|
| `packed+SMLAD` (complete) | 6,362 |
| gather only, MAC loop removed | **4,354** |
| MAC loop only, gather removed | 8,517 (confounded) |

The gather is **~68% of the total** — 4,354 of 6,362. That is the single most
useful number produced this session, because it says where any further work must
go: not at the multiply-accumulate, which SMLAD already retires two at a time,
but at producing its operands.

The MAC-only figure is not trustworthy as a cost: with the gather removed the 23
layers' accumulations become one long serial dependency chain, and the empty ring
means no memory traffic to overlap. It is reported for completeness with that
caveat rather than used to claim the MACs "cost 8,492".

### Where the remaining time actually goes

1,407 MACs per sample at 6,362 cycles is **4.5 cycles per MAC**. With SMLAD
retiring two MACs per instruction the arithmetic floor is nearer 1 cycle per MAC,
so the headroom is somewhere in the loop. The obvious answer was instruction
count, and the duplicated-ring experiment shows it is the wrong answer:

- a 6-tap layer costs ~244 cycles/sample for 54 MACs. Its gather is 13
  instructions per 4-byte operand pair, six of which are the wrap fixup.
- `packed+dupring` removes those six and two more, cuts the kernel's instruction
  count by 27% and its loads by 22%, and is **9.5% slower**.

So the loop is not issue-bound. The instruction count is a symptom of the
structure rather than the cost, and the cost tracks memory *traffic*: the one
thing the duplicated ring adds is 138 more stores per sample, and that alone
outweighs everything it removes. Every earlier negative result in this document
lines up behind that reading — the linear ring (3.3x worse, ~5,483 bytes copied
per sample), code in SRAM (fetch and data sharing one port), `framemajor`
(accumulator traffic). The two-core tax is memory-side for the same reason —
moving the code to RAM under two cores changed nothing — but unlike the others it
is not reachable by placing anything anywhere, which is what the four nulls in
"the 11% two-core tax is not addressable" establish.

What that leaves is not a micro-optimisation. Bytes per sample are set by the
network shape — 23 layers x 3 channels x 6 or 15 taps of int16 state — so
reducing them means fewer taps, a smaller receptive field, or a lower internal
sample rate.

Other things this ruled out, so they are not re-tried: a finer multi-boundary
split (balance is already 97.7%, and half the model is 102.2% of real time before
any split at all), running the hot loop from RAM (2.5% slower on one core, and
exactly neutral with two), the planar power-of-two ring (the same class of change
as the duplicated ring, so expected neutral-to-worse rather than the 23% the
instruction count suggested), and the two 15-tap layers as a place to find
balance (they are 1,250 cycles/sample between them, 19% of the total, and no
arrangement of them fixes an 18% deficit).
