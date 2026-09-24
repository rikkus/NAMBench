cmake_minimum_required(VERSION 3.13...3.31)

# fpi — Waveshare RP2350A-USB-A Mini.
#
# Everything fpi/Pico lives in this directory: the SDK is cloned into ./sdk
# (gitignored), the board definition is ./boards/fpi.h, sources are ./src.
#
#   PICO_SDK_PATH=$(pwd)/sdk cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# See ../docs/fpi/TOOLCHAIN.md for the full story.

set(PICO_BOARD_DIRS "${CMAKE_CURRENT_LIST_DIR}/boards" CACHE PATH "Custom board definitions")

if(NOT DEFINED ENV{PICO_SDK_PATH} AND NOT DEFINED PICO_SDK_PATH)
    message(FATAL_ERROR
        "PICO_SDK_PATH is not set. Use: PICO_SDK_PATH=$(pwd)/sdk cmake -S . -B build ...")
endif()

# Default to the Arm Cortex-M33 cores. Override with -DPICO_PLATFORM=rp2350-riscv
# to run the same sources on the Hazard3 RISC-V cores.
if(NOT DEFINED PICO_PLATFORM)
    set(PICO_PLATFORM rp2350)
endif()

# pioasm assembles src/ws2812.pio into a header at build time. The SDK will try
# to build its own copy via ExternalProject if it cannot find one, which fails
# from a clean tree here; build.sh builds and installs one under
# toolchain/pioasm instead and points this at it.
if(NOT DEFINED pioasm_DIR AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/toolchain/pioasm/lib/cmake/pioasm")
    set(pioasm_DIR "${CMAKE_CURRENT_LIST_DIR}/toolchain/pioasm/lib/cmake/pioasm"
        CACHE PATH "pioasm config package")
endif()

include(pico_sdk_import.cmake)

project(fpi C CXX ASM)

pico_sdk_init()

add_subdirectory(src)
