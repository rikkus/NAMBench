# Cross-compile for 32-bit ARMv7-A Linux (armhf).
#
# The target this exists for is the Rockchip RK3288 — quad Cortex-A17, ARMv7-A,
# 32-bit only — which is the SoC in the HeadRush Core and Prime. Building on the
# device is possible but slow (4 slow cores, 2 GB RAM, Eigen at -O3), so the
# supported route is: cross-build here, rsync the tree to the board, measure
# there.
#
#   cmake -S . -B build-a32-conformance \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-armv7-linux-gnueabihf.cmake \
#     -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_ALL_VARIANTS=ON
#
# Note what is deliberately NOT here: -mcpu, -mfpu and -mfloat-abi. Those decide
# the *arithmetic* on this target, not just the scheduling — see the NB_ARMV7
# branch in CMakeLists.txt — so they live in nb_conformance_flags, where one
# place governs both a cross build from here and a native build on the board.
# Putting them here as well would let the two drift.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Lets `ctest` run the conformance binaries here for a first-pass answer, which
# is a fast filter while iterating.
#
# It is NOT the verdict. AArch32 Advanced SIMD is unconditionally flush-to-zero
# for single precision while VFP scalar honours FPSCR.FZ, and a bit-identity
# claim should not rest on qemu's modelling of that. The device run decides.
#
# -L points qemu at the cross toolchain's sysroot for the dynamic loader and the
# armhf runtime libraries, which an AArch64 build host does not otherwise have.
# Without it a dynamically linked ARMv7 binary fails with a bare "no such file or
# directory" — the loader is what is missing, not the binary.
find_program(NB_QEMU_ARM qemu-arm-static qemu-arm)
if(NB_QEMU_ARM)
  set(CMAKE_CROSSCOMPILING_EMULATOR "${NB_QEMU_ARM}" -L "${CMAKE_FIND_ROOT_PATH}")
endif()
