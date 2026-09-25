#!/usr/bin/env bash
# Build linear_ab (see linear_ab.cpp). Run from the NAMBench root.
#   ir-study/build_linear_ab.sh OUTDIR [extra compiler flags...]
set -euo pipefail
out=$1; shift
cxx=(${CXX:-c++} -std=c++17 -O3 -DNDEBUG "$@")
mkdir -p "$out"
adt=vendor/adt-partitioned
for f in ImpulseResponse dsp wav PartitionedConvolution; do
  "${cxx[@]}" -c -I$adt/dsp -I$adt -isystem vendor/eigen-adt $adt/dsp/$f.cpp -o "$out/adt_$f.o" &
done
"${cxx[@]}" -c -I$adt/dsp -I$adt -Iir-study -isystem vendor/eigen-adt ir-study/linear_ab_adt.cpp -o "$out/linear_ab_adt.o" &
for f in linear dsp registry util; do
  [[ -f vendor/upstream/NAM/$f.cpp ]] && "${cxx[@]}" -c -Ivendor/upstream -Ivendor/upstream/Dependencies/nlohmann -isystem vendor/eigen vendor/upstream/NAM/$f.cpp -o "$out/nam_$f.o" &
done
"${cxx[@]}" -c -Ivendor/upstream -Ivendor/upstream/Dependencies/nlohmann -Iir-study -isystem vendor/eigen ir-study/linear_ab_nam.cpp -o "$out/linear_ab_nam.o" &
"${cxx[@]}" -c -Iir-study ir-study/linear_ab.cpp -o "$out/linear_ab_main.o" &
wait
"${cxx[@]}" "$out"/*.o -o "$out/linear_ab"
