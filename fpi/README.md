# fpi — firmware for the RP2350A-USB-A Mini

Firmware workspace for the device documented in [`../docs/fpi/`](../docs/fpi/).
The question it exists to answer, for fun: **can a NAM A2-Lite kernel run on ONE
Cortex-M33 core?**

> **Scope: this was a curiosity-driven experiment, not a plan.** There is no
> intention of running NAM on a microcontroller in reality — a laptop, phone or
> pedal SoC is the right home for it, and the rest of NAMBench measures those.
> The answer this workspace produced is "no, not at stock clock and not with
> tuning alone", and the value of the exercise is the measurements, not a port.

```bash
./build.sh              # configure + build (pass a target to build one)
./status.sh             # is the board attached, and in which state
./serial-probe.py 20    # read the console (the ONLY reliable reader here)
```

## Measured results (2026-09-24, single core, 150 MHz stock clock)

| kernel | cycles/sample | % of one core | fits 48 kHz? |
|---|---:|---:|---|
| float scalar (`-O3`) | 21,600 | 691% | no |
| int16 Q15 scalar | 9,306 | 298% | no |
| int16 Q15 + SMLAD, packed at use | 10,063 | 322% | no |
| Q15, planar staging | 8,806 | 282% | no |
| Q15, staging + SMLAD | 7,541 | 241% | no |
| **Q15, packed staging + SMLAD** | **6,362** | **204%** | no |
| Q15, packed + duplicated ring | 7,062 | 226% | no |
| Q15, packed, weights in XIP cache | 10,255 | 328% | no |
| float state (fp32 rings + weights) | 18,107 | 579% | no |
| widetile x4 (M2 technique) | 10,573 | 338% | no |
| framemajor (M2 study's worst case) | 24,851 | 795% | no |

Budget is 3,125 cycles/sample at 48 kHz. For scale, the reference project's
tuned `a2_fast` path costs 8,396 cycles/sample — but at 300 MHz across two cores.

Plain int16 C on one 150 MHz core now beats the reference's tuned kernel; the
packed-staging variant beats it by 1.3×. Single-core 48 kHz still needs ~311 MHz,
so it is **not** reachable at stock clock by kernel tuning alone.

## Measured results, two cores (2026-09-24, 150 MHz, target `duo`)

The layer stack split down the middle, core 0 taking layers `[0,b)` and core 1
taking `[b,23)`, over 32 blocks of 480 samples:

| | cycles/sample of wall clock | % of real time |
|---|---:|---:|
| one core, whole model | 6,387 | 204.4% |
| two cores, perfectly balanced (total ÷ 2) | 3,193 | 102.2% |
| two cores, best real boundary (b=13) | 3,286 | 105.2% |
| **two cores, measured** | **3,676** | **117.6%** |

**Two stock-clocked cores are not enough either**, and the reason is arithmetic
before it is anything else: half of the whole model is already 102.2% of real
time. The second core then taxes ~11% on top (measured as `wall ≈
max(T_A,T_B) + 0.11·min(T_A,T_B)`, fitted across the seven boundaries where both
halves do real work), so reaching 48 kHz needs ~177 MHz or a kernel 15–18%
faster.

That 15–18% was then attacked directly and did not give: `packed+dupring` removes
every wrap fixup and 27% of the kernel's instructions, passes `kverify`, and is
**9.5% slower** because it moves more bytes. So the requirement is real but it is
not an instruction-count problem, and the planar power-of-two ring — the same
kind of change — is no longer expected to deliver it. See the "Remaining
headroom" section of [`../docs/fpi/README.md`](../docs/fpi/README.md).

What the split does *not* cost is latency: the first block takes 3,111,246 cycles
on one core and 3,168,652 on two, +1.8%, because it traverses all 23 layers
either way. The split changes the period between blocks, not the delay.

And the ~11% is not addressable. Four attempts to move it — SRAM bank phase, bank
group (same region, different region, and literally the same words), fetch path
(flash, RAM, and split across the cores), and stack placement — all measure
identically. See "the 11% two-core tax is not addressable" in
[`../docs/fpi/README.md`](../docs/fpi/README.md). Full
analysis, including the boundary sweep and the per-layer cost profile, is in
[`../docs/fpi/README.md`](../docs/fpi/README.md).

## Targets

| Target | Purpose |
|---|---|
| `hello.uf2` | First contact. Board id, clock, DWT sanity check, and an LED colour cycle. |
| `bench.uf2` | Cycles/sample for all thirteen kernels against the 48 kHz budget at the current clock. |
| `kverify.uf2` | Runs every kernel at six run lengths and checks each against the float path. |
| `duo.uf2` | The two-core layer split: boundary sweep, dual-core runs, contention fit, bank-phase and fetch-path probes, latency check. |
| `mem2.uf2` | Two-core memory contention with the buffers at known addresses (same bank group / different group / same words). |

`kverify` exists because a fault inside the benchmark's last row is otherwise
invisible: the report simply stops, and a hang looks identical to a crash. Run
it after touching kernels — it is what proved the kernels were fine while the
harness was corrupting memory.

Cycle counts come from the DWT cycle counter, whose register access on Armv8-M
uses the peripheral alias — different from the Cortex-M0+/M3 trick.

## Flash and read

```bash
./status.sh                                        # confirm /Volumes/RP2350
cp build/src/hello.uf2 /Volumes/RP2350             # BOOTSEL: hold BOOT while plugging in
picotool load -f build/src/bench.uf2               # or, also reboots into the app
./serial-probe.py 20                               # read the console
```

`picotool reboot -f -u` puts a running board back into BOOTSEL without touching
it. To do it by hand: **hold BOOT, tap RESET, release BOOT.**

**Do not use `screen`, `cat` or `dd` to read the console** — none of them raise
DTR, and rebooting kills the descriptor at exactly the moment output starts.
The result is a working board that looks dead. Read
[`../docs/fpi/TOOLCHAIN.md`](../docs/fpi/TOOLCHAIN.md) before debugging a silent
board; it will save you an hour.

## Layout

| Path | What |
|---|---|
| `src/kernels.{h,c}` | A2-Lite shape tables, ring pool, six shipped kernels |
| `src/kernels_exp.c` | Techniques ported from the NAMBench M2 study and the ESP32-S3 write-up, with the rejected ones documented |
| `src/kernels_internal.h` | Primitives shared by both kernel files |
| `src/bench.c` | Measurement harness and reporting |
| `src/duo.c` | Two-core layer split, boundary sweep, contention fit and latency check (target `duo`) |
| `src/kverify.c` | Per-kernel correctness check at six run lengths |
| `src/hello.c` | First-contact firmware |
| `src/fault.c` | Fault handlers — without these, a crash looks like a dead board |
| `src/ws2812.pio` | Onboard WS2812 bitstream, assembled by pioasm at build time |
| `src/ws2812.c` | LED driver on top of the generated header |
| `boards/fpi.h` | Board definition (RP2350A, 2 MB flash) |
| `build.sh`, `status.sh`, `serial-probe.py`, `monitor.sh` | Build, connection state, console |

`sdk/` (Pico SDK 2.2.0), `toolchain/` (Arm GNU 14.2.rel1) and
`toolchain/pioasm` are built or downloaded beside the sources and gitignored.
See [`../docs/fpi/TOOLCHAIN.md`](../docs/fpi/TOOLCHAIN.md) — in particular, the
Homebrew `arm-none-eabi-gcc` has no newlib and cannot link this.

## Design constraints worth knowing before editing

- **The shape tables are from the real model**, not made up — 23 layers, kernel
  sizes `[6×14, 15, 15, 6×7]` paired with dilations `[1,3,7,17,41,101,239]`
  cycling, receptive field 6332 samples (131.9 ms), ring pool 6354 samples.
  Read them from `example.nam` in the reference project if they change. Getting
  the pairing wrong silently produces wrong ring sizes — `nam_model_init()`
  checks the pool total against the table and refuses to run if they disagree.
- **Specialise the kernel loops on the tap count.** `taps` is a runtime field,
  so the compiler will not unroll it; the `taps == 6` / `taps == 15` split is
  worth ~1.8× on its own.
- **Packed staging is what makes SMLAD pay.** Packing pairs *during the gather*
  so the MAC loop is one 32-bit load per two MACs gives 1.43× over the best
  scalar version. Packing at the point of use is 1.08× *slower* than scalar.
- **A buffer's length must follow the measured run, not the block run.**
  `measure()` writes every sample it processes; passing it a block-sized buffer
  overflowed BSS by 13 KB and corrupted the PIO state. `MAX_SAMPLES` exists for
  this reason.
- **Never hand-size a table that has an initialiser list.** `bench.c` declared
  `g_kernels[13]` with twelve entries when a kernel was added, so `n_kernels` was
  13 and the last iteration called a **NULL function pointer**. The symptom was a
  console that stopped one line short of the end and a board that had to be
  unplugged — picotool could not reboot it, because the app-side reset interface
  is dead once the core is locked up. The array is self-sized now, `main()`
  refuses to run if the count disagrees with the buffers, and `kverify.c` was
  already safe for the same reason.
- **`K_ABL_FIRST` is an index, not a count.** Inserting a kernel in the middle of
  the table shifts every index after it, so the first version of the
  duplicated-ring row left the ablations inside the "best fixed point" ranking
  and `bench` confidently reported that garbage was the fastest correct kernel.
  Re-derive `K_REF_FLOAT`, `K_BEST_Q15` and `K_ABL_FIRST` whenever a row is
  inserted rather than nudging them.
- **Never hand-assemble PIO and never let the LED block.** See
  [`../docs/fpi/TOOLCHAIN.md`](../docs/fpi/TOOLCHAIN.md); a broken WS2812 driver
  hung the whole firmware through `pio_sm_put_blocking`.
- **Do not "optimise" the LeakyReLU divide, and do not move these kernels to
  SRAM.** Both look like wins and both measured worse. GCC already uses the M33
  hardware divider (`SDIV`) for `/ 100`, and SRAM execution puts instruction
  fetch and data load behind one port, costing ~2.5%. The switches (`USE_RECIP`,
  `NAM_PROCESS`) are kept so the experiments can be re-run; see the "hardware
  features investigated" section in [`../docs/fpi/README.md`](../docs/fpi/README.md).
- **The largest remaining per-tap saving turned out to be a bad trade — in the
  interleaved layout only.** Masking the ring wrap instead of branching was
  implemented twice and abandoned: the mask must be `3·2^k - 1` to keep channel
  lanes aligned, and making the geometry consistent needs a power-of-two ring
  minimum, which quadruples the pool (38 KB → 168 KB) to remove one
  well-predicted branch. Reverted; the reasoning and the numbers are in
  [`../docs/fpi/README.md`](../docs/fpi/README.md). The **planar** variant of
  this idea is now retracted rather than pending: the duplicated ring tested the
  same hypothesis more strongly — no mask, no padding, no byte rescale — and
  lost 9.5%, because removing a well-predicted branch was never worth anything
  on this core. Do not spend the 55 KB.
- **The two-core tax is not addressable.** Four attempts to move the 11% - SRAM
  bank phase (0-3 slots), bank group (same region / different region / same
  words, in `mem2`), fetch path (flash vs RAM, and split across the cores), and
  stack placement (already separate banks) - all measure identically. Each core's
  data stream already spreads over all four banks by its own stride, so what is
  left is bus-fabric arbitration, which no address choice reaches. Plan on the
  1.80x, not 2x, and get the 15-18% from the kernel.
- **Measure twice before believing a two-core projection.** The second core does
  not deliver 2×. On this kernel the measured speedup is 1.76×, and the shortfall
  is memory-port contention, not instruction fetch — moving the hot loop to RAM
  changed nothing at all under two cores. Also check the *arithmetic* first: half
  of a 204% single-core cost is still 102.2%, so a split cannot rescue a kernel
  that is more than 2× over budget.
- **The range-split entry point is fenced behind `FPI_DUO` on purpose.** Merely
  adding it to `kernels.c` moved `packed+SMLAD` from 6,451 to 6,539 by giving GCC
  one more call site to clone the body against, and passing the range as two more
  parameters on the shared body moved `staged` from 8,915 to 9,320. `bench` and
  `kverify` compile the original kernel source exactly and still report 6,451 /
  8,915 / 7,566; the experiment carries its own copy.
- **There is no faster memory to move data to, and cache-as-SRAM loses badly.**
  The XIP cache's 16 kB can genuinely be used as SRAM by pinning lines
  (`xip_cache_pin_range`, datasheet 4.4.1.3) at an address in the top half of the
  XIP space that the QMI does not decode; `nam_process_packed_wxip` does exactly
  that with the 6.6 kB of weights and passes `kverify`. It also costs **60%**
  (10,289 vs 6,451 cycles/sample), because that cache is the same 16 kB serving
  instruction fetch for code running from flash - pinning 6.6 kB of it takes
  those sets away from the code. The cache is a second *port*, not a second
  memory, and the port was already busy. Note also that it is "1 cycle hit" and
  the SRAM is "single-cycle access": neither can beat one access per cycle, which
  is the M33's single load/store unit. Do not look for faster RAM on this part.
- **Doubles are not software-emulated on this part.** RP2350 has a
  double-precision coprocessor and `pico_double` uses it, but `dmul` costs 17
  cycles against 1 for single-precision — so fixed point still wins on
  throughput. Do not repeat the claim that doubles trap to software.
- **Float is not competitive on this part.** A proper fp32-state kernel is
  bit-identical to the reference and 2.84x slower than fixed point: the M33 retires
  two integer MACs per instruction via SMLAD and one float MAC via VFMA, and float
  also doubles the load width. The ESP32-S3 write-up's "float beat our integers"
  result does not transfer, because that chip's integer vector unit is 128-bit
  while its FPU has no vector path.
- **Read-only tables belong in the struct, not in flash.** `nam_ring_size[]` is
  `static const` (flash, so every read is an XIP access on the port both cores
  share) while the same value sits in SRAM as `layer[l].ring_size`. Using the
  struct field gave 6,451 -> 6,362 on `packed+SMLAD`, and the kernels that never
  read the table did not move at all. Check for this pattern before adding any
  lookup to a hot loop.
- **The gather is ~68% of the cost** (4,351 of 6,451 cycles/sample), not the
  MACs. Any further optimisation has to attack operand production.
- **But it is *traffic*, not instructions, that costs.** `packed+dupring` removes
  every wrap fixup (13 instructions per tap pair down to 8), cuts the kernel's
  instruction count by 27% and its loads by 22%, and is **9.5% slower** — because
  it adds 138 stores per sample. Reducing instructions from this kernel does not
  make it faster and neither does reducing loads; the same fact explains why the
  linear ring loses 3.3× (it copies ~5,483 bytes/sample), why SRAM-resident code
  is slower, and why the second core only delivers 1.80× rather than 2×. Do not
  start another "fewer instructions in the gather" experiment without a plan for
  bytes moved.
- `bench.c` checks that Q15 agrees with the float path, so a scaling bug shows
  up as a validation failure rather than as a suspiciously good cycle count.
- Every step in `bench.c` that can fail prints a `stage:` marker, and `fault.c`
  reports fault registers. Keep both; a single fault report in the CDC FIFO is
  capped at 64 bytes, so messages are deliberately short.
