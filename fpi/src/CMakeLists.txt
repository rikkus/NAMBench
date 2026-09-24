# ---
# fpi: Waveshare RP2350A-USB-A Mini. Targets:
#   hello — flash, enumerate as USB CDC, print. First-contact check.
#   bench — per-sample cycle cost of candidate kernels, printed over USB CDC.
# ---

set(FROM_FPI "${CMAKE_CURRENT_LIST_DIR}/..")

# Console output must not depend on the host raising DTR.
#
# The SDK gates stdio_usb_out_chars() on stdio_usb_connected(), which with
# PICO_STDIO_USB_CONNECTION_WITHOUT_DTR unset "actually checks DTR"
# (sdk/src/rp2_common/pico_stdio_usb/stdio_usb.c:289). macOS's CDC-ACM driver
# does not raise DTR just because a program opened /dev/cu.usbmodem*, and
# raising it from the host with TIOCMBIS is not reliably translated into a
# SET_CONTROL_LINE_STATE on the wire either. The observed failure mode is a
# board that is provably running — it enumerates as 2e8a:0009 and can drive its
# LED — while emitting zero bytes, which is indistinguishable from a crash.
#
# Setting this makes the check tud_ready() (bus configured) instead. The SDK
# sources are compiled as part of these targets (see __/sdk/... in
# DependInfo.cmake), so the definition has to be on every target explicitly;
# directory-level add_compile_definitions() inherits too fragilely to rely on.
set(FPI_CONSOLE_DEFS PICO_STDIO_USB_CONNECTION_WITHOUT_DTR=1)

# The WS2812 bitstream is generated from src/ws2812.pio by pioasm at build time,
# producing ws2812.pio.h next to it. Hand-assembling the opcodes instead is how
# the LED ended up silently broken (see the comment in ws2812.c).
#
# pico_generate_pio_header() takes an existing target as its first argument and
# attaches the generated header to it, so each target calls it after creation.
set(FPI_PIO "${CMAKE_CURRENT_LIST_DIR}/ws2812.pio")

# ---------------------------------------------------------------------------
# hello: first contact. Proves the whole chain works — build, UF2 bootloader,
# USB CDC enumeration, /dev/cu.usbmodem* — before any DSP is involved.
# ---------------------------------------------------------------------------
add_executable(hello
    hello.c
    ws2812.c
    fault.c
)
target_include_directories(hello PRIVATE "${FROM_FPI}/src")
pico_generate_pio_header(hello "${FPI_PIO}")
target_compile_definitions(hello PRIVATE ${FPI_CONSOLE_DEFS})
target_link_libraries(hello PRIVATE
    pico_stdlib
    pico_unique_id
    hardware_pio
)
# Console over USB CDC only: the board's UART pins are on the stamp holes and
# usually not wired up. See docs/fpi/README.md.
pico_enable_stdio_usb(hello 1)
pico_enable_stdio_uart(hello 0)
pico_add_extra_outputs(hello)

# ---------------------------------------------------------------------------
# bench: the actual question. Fixed-point and float convolutions in the shape of
# a NAM A2-Lite, measured against the 48 kHz cycle budget.
# ---------------------------------------------------------------------------
add_executable(bench
    bench.c
    kernels.c
    kernels_exp.c
    ws2812.c
    fault.c
)
target_include_directories(bench PRIVATE "${FROM_FPI}/src")
pico_generate_pio_header(bench "${FPI_PIO}")
target_compile_definitions(bench PRIVATE ${FPI_CONSOLE_DEFS})
target_link_libraries(bench PRIVATE
    hardware_xip_cache
    pico_stdlib
    pico_unique_id
    hardware_pio
)
pico_enable_stdio_usb(bench 1)
pico_enable_stdio_uart(bench 0)
pico_add_extra_outputs(bench)

# ---------------------------------------------------------------------------
# duo: the fine-grained multi-boundary split. The same kernel as bench's winner,
# restricted to a layer range and run on both cores, with a boundary sweep.
#
# FPI_DUO compiles two things bench does not have: the RAM twin of the hot loop
# (so flash-vs-RAM can be re-asked under two-core fetch contention) and the
# range entry point the split needs.
# ---------------------------------------------------------------------------
add_executable(duo
    duo.c
    kernels.c
    ws2812.c
    fault.c
)
target_include_directories(duo PRIVATE "${FROM_FPI}/src")
pico_generate_pio_header(duo "${FPI_PIO}")
target_compile_definitions(duo PRIVATE ${FPI_CONSOLE_DEFS} FPI_DUO=1)
target_link_libraries(duo PRIVATE
    hardware_xip_cache
    pico_stdlib
    pico_unique_id
    pico_multicore
    hardware_pio
)
pico_enable_stdio_usb(duo 1)
pico_enable_stdio_uart(duo 0)
pico_add_extra_outputs(duo)

# ---------------------------------------------------------------------------
# mem2: two-core memory contention, with the buffers at known addresses.
#
# RP2350's 520 kB is two 256 kB regions, each 4-way striped over a disjoint set
# of banks (SRAM0-3 stripe over banks 0-3, SRAM4-7 over banks 4-7). Placing two
# buffers in the same region and two in different regions is the only way to ask
# whether the two cores collide in a bank - a phase shift within one region
# cannot, because both streams already span all four banks.
#
# The buffers are cut out of one .bss arena at runtime rather than placed by the
# linker: an orphan section at a fixed VMA gets flash contents in the ELF, which
# picotool rejects as "memory contents for uninitialized memory". An arena that
# is simply big enough to straddle 0x20040000 does the same job with no linker
# involvement at all.
# ---------------------------------------------------------------------------
add_executable(mem2
    mem2.c
    ws2812.c
    fault.c
)
target_include_directories(mem2 PRIVATE "${FROM_FPI}/src")
pico_generate_pio_header(mem2 "${FPI_PIO}")
target_compile_definitions(mem2 PRIVATE ${FPI_CONSOLE_DEFS})
target_link_libraries(mem2 PRIVATE
    pico_stdlib
    pico_multicore
    hardware_pio
)
pico_enable_stdio_usb(mem2 1)
pico_enable_stdio_uart(mem2 0)
pico_add_extra_outputs(mem2)

# ---------------------------------------------------------------------------
# kverify: run every kernel in isolation and check it against the float path.
# A fault inside the benchmark's last row is otherwise invisible.
# ---------------------------------------------------------------------------
add_executable(kverify
    kverify.c
    kernels.c
    kernels_exp.c
    ws2812.c
    fault.c
)
target_include_directories(kverify PRIVATE "${FROM_FPI}/src")
pico_generate_pio_header(kverify "${FPI_PIO}")
target_compile_definitions(kverify PRIVATE ${FPI_CONSOLE_DEFS})
target_link_libraries(kverify PRIVATE
    hardware_xip_cache
    pico_stdlib
    hardware_pio
)
pico_enable_stdio_usb(kverify 1)
pico_enable_stdio_uart(kverify 0)
pico_add_extra_outputs(kverify)
