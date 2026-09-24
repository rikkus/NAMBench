# Building and flashing for fpi

What is installed on this laptop, what is left to do, and the exact commands.
Device facts (pinout, what macOS shows, the single-core budget) are in
[`README.md`](README.md).

## What is installed

| Piece | Where | Version | Notes |
|---|---|---|---|
| `picotool` | `/opt/homebrew/bin` | 2.3.1 | Homebrew. Flashing + `picotool info` on a `.uf2`. |
| `openocd` | `/opt/homebrew/bin` | 0.12.0 | Homebrew. Only needed for SWD debugging. |
| Pico SDK | `fpi/sdk` | 2.2.0 | `--recurse-submodules`, so TinyUSB is present. |
| `cmake`, `ninja` | `/opt/homebrew/bin` | — | Already present. |
| **Arm GNU toolchain** | `fpi/toolchain/arm-gnu-toolchain-14.2.rel1-...` | 14.2.rel1 | Official Arm build, extracted from the `.pkg` — see below. |

### The toolchain trap

`brew install arm-none-eabi-gcc` does **not** work for this. The Homebrew
formula builds GCC `--without-newlib`, so it has no `nosys.specs` and the very
first link (the RP2350 boot stage 2) fails with:

```
arm-none-eabi-gcc: fatal error: cannot read spec file 'nosys.specs'
```

The bare-metal toolchain has to come from Arm. The Homebrew cask
`gcc-arm-embedded` (15.3.rel1) is the official one, but it is a macOS `.pkg`
that installs to `/Applications` and therefore needs `sudo`. This environment
has no working `sudo` (`/usr/bin/sudo` cannot even be executed here), so the
`.pkg` is **extracted into the workspace instead** — no root, no system-wide
change, and it stays with the project:

```bash
cd fpi/toolchain
curl -L -o arm-gnu-toolchain.pkg \
  "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi.pkg"
pkgutil --expand-full arm-gnu-toolchain.pkg extracted
```

`pkgutil --expand-full` needs no privileges and writes the package tree into
`extracted/`. The compiler then lives at

```
fpi/toolchain/extracted/Payload/bin/arm-none-eabi-gcc
```

and the Pico SDK is pointed at `fpi/toolchain/extracted/Payload` with
`PICO_TOOLCHAIN_PATH`. Both `build.sh` and the commands below do this
explicitly rather than relying on `PATH` order.

The Homebrew `arm-none-eabi-gcc` is still installed and still first on `PATH`.
It is the wrong one for linking, which is why it is set explicitly.

The downloaded `.pkg` (257 MB) was deleted after extraction; the extracted tree
(978 MB) is all that is kept. To rebuild it from scratch, re-run the two
commands above — the URL is the stable Arm release path for 14.2.rel1. The SDK
checkout is a further 725 MB. Both are gitignored, so neither is in the repo.

## Verified state

All five targets build, flash, and run on the device. Verified 2026-09-24:

| | |
|---|---|
| Board | RP2350 rev **A4**, QFN60, chip id `0486A7AD2EF082FC`, `clk_sys` 150 MHz |
| Toolchain | Arm GNU Toolchain 14.2.Rel1 (GCC 14.2.1), darwin-arm64, Apple-notarized `.pkg` |
| DSP path | `__ARM_FEATURE_DSP` set by the SDK's `-march=armv8-m.main+fp+dsp`; `smlad` confirmed in the disassembly |
| Console | `serial-probe.py` receives the full `bench` report (~3.5 kB) |
| `bench` validation | Q15 agrees with the float path at max err 3.99e-4, and the fixed-point path is sane |
| `kverify` | 91 of 91 checks pass across 11 kernels at six run lengths, 240 to 8064 samples |
| `duo` | every split output is bit-identical to the single-core run, at every boundary |

Benchmark results and what they mean are in [`README.md`](README.md).

The `hello.elf` and `bench.elf` size rows that used to be here were stale within a
day of being written — the model struct grew and the kernel table doubled. Sizes
are not worth recording; `arm-none-eabi-size build/src/*.elf` is one command.

## Build

```bash
cd fpi
./build.sh            # configures into build/ and builds every target
```

or by hand:

```bash
cd fpi
export PICO_SDK_PATH="$(pwd)/sdk"
export PICO_TOOLCHAIN_PATH="$(pwd)/toolchain/extracted/Applications/ArmGNUToolchain/14.2.rel1/arm-none-eabi"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Products land in `build/src/`:

| Target | What it is |
|---|---|
| `hello.uf2` | First contact: prints board id, clock, and a heartbeat over USB CDC; breathes the RGB LED. Flash this first. |
| `bench.uf2` | The actual question: cycles/sample for every kernel against the 48 kHz budget, plus validation. |
| `kverify.uf2` | Runs every kernel at six run lengths and checks each against the float path. Run this after touching kernels. |
| `duo.uf2` | The two-core layer split: boundary sweep, dual-core runs, bank-phase and fetch-path probes, latency check. |
| `mem2.uf2` | Two-core memory contention with the buffers at known addresses. Not about NAM at all. |

`nam_bench`-style cycle counts need the DWT cycle counter. On Armv8-M the
registers are reached through the peripheral alias (`0xE0001004` for `DWT_CYCCNT`
here, with `DEMCR.TRCENA` set), which is *not* the same encoding trick used on
Cortex-M0+/M3 — worth knowing if the counts come back zero.

## Flash

The board ships blank, so it is in the boot ROM on first contact. **Hold BOOT,
plug in USB-C, release BOOT**, then:

```bash
ls /Volumes/RP2350
cp fpi/build/src/hello.uf2 /Volumes/RP2350
```

Or, which also reboots the board into the new firmware rather than leaving it in
the bootloader:

```bash
picotool load -f fpi/build/src/hello.uf2
```

## Read the console

**Use `./fpi/serial-probe.py`, not `screen`, `cat` or `dd`.** This is not a
preference; the other tools do not work reliably here, and the failure looks
exactly like a crashed board. See "the silent-console trap" below.

```bash
./fpi/serial-probe.py 20       # watches for 20 s, survives re-enumeration
./fpi/monitor.sh               # interactive, uses screen
```

`hello` and `bench` use `stdio_usb` only (`pico_enable_stdio_uart(..., 0)`),
because the UART pins are on the stamp holes. There is no `/dev/tty.usb*` node
until firmware that actually enumerates CDC is running. Use the `/dev/cu.*`
node, not `/dev/tty.*` — the `tty` variant blocks waiting for carrier, and any
baud rate works because USB CDC ignores it.

Other quirks:

- Both firmwares wait ~2.5 s after reset before their first write, because the
  USB host takes about that long to enumerate and anything written earlier is
  discarded into a disconnected endpoint.
- If the board is running UAC2 audio firmware (not these targets), there is **no
  serial port at all** — it is a sound card. Do not read that as a failure.

## The silent-console trap

A freshly flashed RP2350 can enumerate, run its firmware correctly, and still
print **nothing**. Two independent causes, both hit on this laptop:

1. **The SDK gates USB output on DTR.** `stdio_usb_out_chars()` writes only when
   `stdio_usb_connected()` is true, and that (with the build define unset)
   "actually checks DTR" — see
   `sdk/src/rp2_common/pico_stdio_usb/stdio_usb.c:289`. macOS's CDC-ACM driver
   does not raise DTR just because a program opened `/dev/cu.usbmodem*`, and
   raising it from the host with `TIOCMBIS` does not reliably become a
   `SET_CONTROL_LINE_STATE` on the wire either. The firmware's output is then
   discarded, silently, one byte at a time.
   **Fix:** build with `PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1`, which makes the
   check `tud_ready()` instead. Both targets do this in `src/CMakeLists.txt`;
   note that directory-level `add_compile_definitions()` is not enough, because
   the SDK sources are compiled as part of these targets and the definition has
   to be on the target itself.

2. **A reader that dies at exactly the wrong moment.** Rebooting or reflashing
   invalidates the open descriptor (`OSError: [Errno 6] Device not
   configured`), and firmware that prints its banner once at startup has already
   finished by the time a naive reader reopens the port. `cat` and `dd` also do
   not raise DTR.
   **Fix:** `serial-probe.py` survives re-enumeration, raises DTR/RTS, and keeps
   reading.

The trap is nasty because it is indistinguishable from a crash: `ioreg` shows
the device happily enumerated as `2e8a:0009` (product "Pico", the SDK's default
descriptor), which *proves* `main()` ran far enough to bring TinyUSB up — while
the console shows nothing at all.

Diagnosis order when a board looks dead:

```bash
./fpi/status.sh                                             # enumerated at all?
ioreg -p IOUSB -w0 -l | grep -iE "USB Product Name|idProduct"
./fpi/serial-probe.py 20                                    # the reliable reader
```

`system_profiler SPUSBDataType` returns **nothing** on this macOS build, which is
why `status.sh` uses `ioreg` for USB. `TIOCSDTR` is also not supported on
CDC-ACM; the probe uses `TIOCMBIS` with `TIOCM_DTR`.

If it still says nothing, it is genuinely faulting — which is why `src/fault.c`
exists. It reports the fault status registers (CFSR/HFSR/MMFAR/BFAR) and halts,
and `bench.c` prints a `stage:` marker before every step that can plausibly
fail, so a truncated log localises the fault to a stage rather than leaving you
to guess. That is exactly how the ring-pool size bug was found.

## Six things that cost hours, so they do not cost them again

### 1. A fault handler that disarms itself hides the faults that matter

`fpi_fault_init()` originally reported for the first 5 seconds only, on the
theory that a startup crash is the interesting one. The `bench` run crossed the
5-second mark partway through its last kernel, so the fault it was built to
catch was swallowed by the handler having already turned itself off. The window
is now 10 minutes. If a board goes quiet partway into a long run, check that
your fault reporting is still armed before theorising about the kernel.

### 2. Passing a short buffer to the benchmark's own `measure()`

`measure(fn, out, n_samples)` runs the kernel over `n_samples` and the kernel
writes **every** sample through `out`. `bench` passed
`BLOCK_SAMPLES * N_BLOCKS` = 3840 samples with a buffer sized
`BLOCK_SAMPLES` = 480 floats: a 13 KB overrun straight into BSS. It corrupted
the PIO state used by the LED, and the symptom was a non-deterministic hang
several kernels into a run — which looks exactly like a kernel bug and was not.
`g_out` is now `MAX_SAMPLES` wide, with the length defined by the measured run
rather than the block.

Corollary: a bench harness needs the same scrutiny as the code it measures.
Both of the bugs above were in the harness, not the kernels.

### 3. A hand-assembled PIO program, and a blocking LED write

The WS2812 driver was written with hardcoded opcodes and got three things
wrong: a 2/1/1 timing split where WS2812 needs 3/3/4, no TX FIFO join (4 entries
instead of 8), and no use of pioasm's generated config. The state machine never
advanced, so the LED never lit — and because `pio_sm_put_blocking()` spins until
the FIFO has room, the first few writes filled it and then hung the entire
firmware. A cosmetic status LED took down a benchmark.

Both halves are fixed: the bitstream now comes from `src/ws2812.pio` assembled by
pioasm at build time, and `ws2812_set()` waits with a 2 ms deadline and gives up
rather than blocking forever. Two general rules:

- **Never hand-assemble PIO.** Write a `.pio` file and let pioasm do it; the
  header also gives you `*_program_get_default_config()`, which is where the
  sideset count and wrap are kept consistent with the program.
- **Never let a status LED block.** `pio_sm_put_blocking` with no timeout turns
  a diagnostic into a hang.

pioasm is not in Homebrew and the SDK's own ExternalProject build of it fails
from a clean tree here, so `build.sh` builds and installs one under
`toolchain/pioasm` from the SDK sources (it ships a pre-generated lexer and
parser, so no flex or bison is needed) and points `pioasm_DIR` at it.

### 4. A table sized by hand, and a board that had to be unplugged

`bench.c` declared `g_kernels[13]` with twelve entries after a kernel was added.
C has no objection to that: the thirteenth slot is silently zero-filled, so
`n_kernels` read 13 and the last iteration called a **NULL function pointer**. The
symptom was a console that stopped one line short of the end of the report, and
then a board that picotool could not reboot — `picotool reboot -f -u` reported
success and nothing happened, and the device's USB serial number read back as
garbage. The app-side reset interface is dead once the core is locked up, so the
only recovery is physical: unplug, and replug with BOOT held.

Two guards came out of it. The array is self-sized now (`g_kernels[]`, with
`n_kernels` derived from it), and `main()` refuses to run if the count disagrees
with the buffers it indexes. The same class of bug in `kverify.c` was already
impossible because its table was self-sized from the start.

The sequel is worth knowing too: `K_ABL_FIRST` is an *index*, not a count, so
inserting a kernel in the middle of the table shifts every index after it. The
first version of the duplicated-ring row left the two ablations inside the "best
fixed point" ranking, and `bench` duly reported that garbage was the fastest
correct kernel. Re-derive `K_REF_FLOAT`, `K_BEST_Q15` and `K_ABL_FIRST` whenever a
row is inserted.

### 5. An orphan linker section, and picotool refusing the ELF

`mem2` needs two buffers in known SRAM bank groups, so the obvious move is
`__attribute__((section(".bufa")))` plus `-Wl,--section-start=.bufa=0x20010000`.
That works — GNU ld honours it, and it will tell you loudly if the address
overlaps `.data` or `.bss` — but the section is an *orphan*, so the linker gives
it an LMA in flash and the ELF ends up with initialised contents for an SRAM
address. `picotool` rejects that with

```
ERROR: ELF contains memory contents for uninitialized memory at 0x20010000
```

which is correct of it. The fix used here was to stop fighting the linker: one
large `.bss` arena that happens to straddle the address of interest, cut into
sub-buffers at runtime. If a section really has to be placed, it also has to be
declared `NOLOAD`, which means a linker script rather than a flag.

### 6. Start the console reader before the reset, not after

`serial-probe.py` survives re-enumeration *after* it has opened the port once, but
if it is started while the board is mid-reboot it raises `termios.error: (6,
'Device not configured')` and exits. Firing `picotool load` and the probe off in
the same breath is therefore a coin flip. Either wait for the port to come back
first, or start the probe, let it open, and *then* reboot the board.

## When it does not work

| Symptom | Likely cause |
|---|---|
| No `RP2350` volume when plugged in | Cable is charge-only, or BOOT was not held at power-up. |
| Volume appears but `cp` never finishes | Same cable, or a USB hub. Go direct to the laptop. |
| Freshly flashed firmware prints nothing | Check `ls /dev/cu.usbmodem*` — the CDC node, not a missing UART. |
| `picotool load` says no device | Board is not in BOOTSEL and the running firmware does not expose a reset interface. Hold BOOT + tap RESET. |
| Cycle counts come back 0 | DWT not enabled, or compiled for the wrong core. |
