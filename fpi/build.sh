#!/usr/bin/env bash
# Configure and build fpi firmware for the RP2350A-USB-A Mini.
#
#   ./build.sh              configure (if needed) + build both targets
#   ./build.sh hello        build one target
#   ./build.sh clean        wipe build/
#
# The Arm GNU toolchain is the one extracted from the official .pkg into
# toolchain/ — NOT the Homebrew arm-none-eabi-gcc, which has no newlib. See
# ../../docs/fpi/TOOLCHAIN.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TOOLCHAIN="$HERE/toolchain/extracted/Payload"
SDK="$HERE/sdk"
BUILD="$HERE/build"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD"
    echo "removed $BUILD"
    exit 0
fi

if [[ ! -x "$TOOLCHAIN/bin/arm-none-eabi-gcc" ]]; then
    cat >&2 <<EOF
error: Arm GNU toolchain not found at
  $TOOLCHAIN/bin/arm-none-eabi-gcc

Extract it from the official .pkg (no root needed):

  mkdir -p toolchain && cd toolchain
  curl -L -o arm-gnu-toolchain.pkg \\
    "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi.pkg"
  pkgutil --expand-full arm-gnu-toolchain.pkg extracted

Use this toolchain, not 'brew install arm-none-eabi-gcc': the Homebrew build
has no newlib and cannot link the RP2350 boot stage 2.
EOF
    exit 1
fi

if [[ ! -d "$SDK/src" ]]; then
    echo "error: Pico SDK not found at $SDK" >&2
    echo "  git clone --depth 1 --branch 2.2.0 --recurse-submodules \\" >&2
    echo "    https://github.com/raspberrypi/pico-sdk.git sdk" >&2
    exit 1
fi

export PICO_SDK_PATH="$SDK"
export PICO_TOOLCHAIN_PATH="$TOOLCHAIN"

# pioasm turns src/ws2812.pio into a header at build time. There is no Homebrew
# formula, and letting the SDK build its own via ExternalProject fails from a
# clean tree here, so build and install one under toolchain/pioasm. It is a
# standalone C++ tool with a pre-generated lexer and parser, so no flex or bison
# is required.
PIOASM="$HERE/toolchain/pioasm/bin/pioasm"
if [[ ! -x "$PIOASM" ]]; then
    echo "==> building pioasm"
    cmake -S "$SDK/tools/pioasm" -B "$HERE/toolchain/pioasm-build" \
        -DCMAKE_BUILD_TYPE=Release -DPIOASM_VERSION_STRING="pico-sdk-2.2.0" >/dev/null
    cmake --build "$HERE/toolchain/pioasm-build" -j >/dev/null
    cmake --install "$HERE/toolchain/pioasm-build" --prefix "$HERE/toolchain/pioasm" >/dev/null
    echo "    installed $PIOASM"
fi

if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
    echo "==> configuring"
    cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
fi

echo "==> building"
if [[ -n "${1:-}" ]]; then
    cmake --build "$BUILD" --target "$1" -j
else
    cmake --build "$BUILD" -j
fi

echo
echo "==> products"
find "$BUILD/src" -maxdepth 1 -name '*.uf2' -exec ls -lh {} \; 2>/dev/null
echo
echo "Flash:  picotool load -f $BUILD/src/hello.uf2"
echo "        (or, in BOOTSEL: cp $BUILD/src/hello.uf2 /Volumes/RP2350)"
echo "Console: ./serial-probe.py 20   (NOT screen/cat/dd — see docs/fpi/TOOLCHAIN.md)"
