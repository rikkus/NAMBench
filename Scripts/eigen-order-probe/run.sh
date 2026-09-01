#!/usr/bin/env bash
#
# Build and run the Eigen reduction-order probe.
#
# Compiled with the same flags the engine frameworks use (project.yml's Engine
# template: -O3, NDEBUG, C++20, the shared vendor/eigen) so that Eigen selects
# the same kernels here as it does inside a2_fast.
#
# The compiler and the architecture flags are overridable, because the ordering
# this probe establishes is not a property of Eigen alone. Eigen's gebp blocking
# is driven by EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS, which
# Eigen/src/Core/arch/NEON/PacketMath.h sets to 32 on ARM64 and 16 everywhere
# else — so an AArch32 build blocks differently and can reduce in a different
# order. The 224/224 result recorded in FULL-PATH.md is an AArch64 result and
# does not transfer; it has to be re-established per target.
#
#   CXX=arm-linux-gnueabihf-g++ \
#   NB_PROBE_FLAGS="-mcpu=cortex-a17 -mfpu=neon-vfpv4 -mfloat-abi=hard" \
#   NB_PROBE_RUNNER=qemu-arm-static ./run.sh
#
# NB_PROBE_RUNNER is how a cross-built probe gets executed; leave it unset for a
# native build. Under qemu the answer is indicative — run it on the device before
# building a kernel on top of it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
EIGEN="${ROOT}/vendor/eigen"

if [[ ! -d "${EIGEN}/Eigen" ]]; then
  echo "error: ${EIGEN} is not a checkout of Eigen. Run ./Scripts/fetch-vendor.sh first." >&2
  exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

CXX="${CXX:-}"
if [[ -z "${CXX}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    CXX=clang++
  else
    CXX=c++
  fi
fi

# Architecture flags. On Apple the historical default was `-arch arm64`; keep it
# so an unadorned run on a Mac reproduces the recorded numbers exactly.
if [[ -n "${NB_PROBE_FLAGS+x}" ]]; then
  read -r -a ARCH_FLAGS <<<"${NB_PROBE_FLAGS}"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  ARCH_FLAGS=(-arch arm64)
else
  ARCH_FLAGS=()
fi

FLAGS=(
  -std=c++20
  -O3
  -DNDEBUG=1
  "${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"}"
  -I "${EIGEN}"
  -Wall
  -Wno-unused-but-set-variable
)

# probe.cpp holds its candidates to the arithmetic they are written with by
# wrapping them in `#pragma clang fp contract(off)`. GCC does not implement that
# pragma — it warns and ignores it — so under GCC the candidates get whatever
# contraction the command line asked for, and a build carrying the engine's
# -ffp-contract=fast silently fuses the very multiply-adds a candidate exists to
# keep apart. That reads as "Eigen orders differently on this target" when what
# actually happened is that the question was never asked.
#
# So on GCC the contraction is turned off for the whole file instead. That does
# not move the reference: Eigen's C=8 path reaches its FMAs through NEON
# intrinsics, not through source-level a*b+c, and the head reference spells its
# fusion std::fmaf. Both stay fused with contraction off, which is exactly what
# the candidates are being compared against.
if ! "${CXX}" --version 2>/dev/null | head -1 | grep -qi clang; then
  FLAGS+=(-ffp-contract=off)
fi

echo "probe: ${CXX} ${ARCH_FLAGS[*]-}" >&2
"${CXX}" "${FLAGS[@]}" "${HERE}/probe.cpp" -o "${OUT}/probe"

if [[ -n "${NB_PROBE_RUNNER:-}" ]]; then
  echo "probe: running under ${NB_PROBE_RUNNER} — indicative only, confirm on the device" >&2
  "${NB_PROBE_RUNNER}" "${OUT}/probe"
else
  "${OUT}/probe"
fi
