#!/usr/bin/env bash
#
# Build and run the Eigen reduction-order probe.
#
# Compiled with the same flags the engine frameworks use (project.yml's Engine
# template: -O3, NDEBUG, C++20, arm64, the shared vendor/eigen) so that Eigen
# selects the same kernels here as it does inside a2_fast.

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

FLAGS=(
  -std=c++20
  -O3
  -DNDEBUG=1
  -arch arm64
  -I "${EIGEN}"
  -Wall
  -Wno-unused-but-set-variable
)

clang++ "${FLAGS[@]}" "${HERE}/probe.cpp" -o "${OUT}/probe"
"${OUT}/probe"
