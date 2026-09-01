# Cross-compile for 32-bit ARMv7-A Linux (armhf) with clang.
#
# The second half of the campaign's compiler sweep. GCC 13.3 is the primary
# toolchain — it is what the finished kernels are measured and promoted under —
# and this exists so that "the winner is a property of the code, not of gcc's
# scheduler" is a measured claim rather than a hope.
#
#   cmake -S . -B build-a32-clang \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-armv7-linux-gnueabihf-clang.cmake \
#     -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_ALL_VARIANTS=ON
#
# Note what is deliberately NOT here, exactly as in the gcc toolchain file:
# -mcpu, -mfpu and -mfloat-abi decide the arithmetic on this target, so they live
# in nb_conformance_flags where one place governs both toolchains. If they were
# set here as well the two builds could drift, and a compiler sweep between two
# builds that computed different things would be worthless.
#
# WHAT THIS BUILD IS NOT VALID FOR, established by running it on the board.
#
# At C=8 the reference's own arithmetic changes under clang. a2_fast's C=8 path
# is Eigen, and Eigen's gebp kernel compiled by clang 18.1.3 for this target
# contains 310 non-fused vmla.f32 where the gcc build contains none — so the two
# compilers' a2_fast do not compute the same bits:
#
#   gcc 13.3         a2_fast widest checksum -17.478711597881365
#   clang 18.1.3     a2_fast widest checksum -17.478718637490147
#
# The consequence, measured by Scripts/compare-conformance.py against a clang
# build on the device: every planar C=8 kernel — which uses vfma, as its parity
# claim requires — is 127.5 dB from clang's a2_fast rather than bit-identical,
# while s_baseline and s_head_tile (Eigen layer bodies themselves) still match.
# The C=3 kernels are unaffected: that branch is scalar, no Eigen involved, and
# they stay bit-identical under both compilers.
#
# So: this toolchain is for measuring *speed* under a second compiler. Every
# bit-identity claim this campaign makes at C=8 is a claim about the gcc build,
# and the promotion argument in A32-PATH.md has to say so. It is not a defect in
# the kernels — it is the reference moving.
#
# No --sysroot. Debian's multiarch layout puts the armhf runtime where clang
# already looks; passing --sysroot=/usr/arm-linux-gnueabihf instead makes the
# linker prefix the sysroot onto the absolute paths inside libc.so's linker
# script and fail to find libc at all.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)

# clang finds its own headers and libraries here, but CMake still needs to be
# told not to go looking for host ones.
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Same first-pass filter the gcc toolchain gets, and the same caveat: qemu's
# modelling of AArch32 flush-to-zero is not what a bit-identity claim rests on.
# The device run decides.
find_program(NB_QEMU_ARM qemu-arm-static qemu-arm)
if(NB_QEMU_ARM)
	set(CMAKE_CROSSCOMPILING_EMULATOR "${NB_QEMU_ARM}" -L "${CMAKE_FIND_ROOT_PATH}")
endif()
