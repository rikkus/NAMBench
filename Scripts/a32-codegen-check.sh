#!/usr/bin/env bash
#
# Check what the A32 lab's kernels actually compiled to.
#
# Two questions, both of which the ARMv7 target makes worth asking of the object
# rather than of the source.
#
# 1. FUSED OR NOT (a correctness check, and this script's reason to exist).
#
#    ARM has two multiply-accumulate instructions and they are not the same:
#
#      VFMA.F32   fused, ONE rounding
#      VMLA.F32   not fused, TWO roundings
#
#    Every a32 kernel that claims bit-identity with a2_fast reproduces a chain of
#    single-rounded FMAs. If a vmla reaches one of those kernels — because the
#    build lost -mfpu=neon-vfpv4, or because someone reached for
#    vmlaq_lane_f32 to save an instruction — the arithmetic changes by one ulp
#    per operation and the parity claim quietly stops being true. This fails the
#    build on that.
#
#    A kernel may opt out by putting the marker NB_A32_NOT_EXACT in its source,
#    which is also what its KernelEntry's `exact = false` says. The vmla variants
#    exist deliberately, to measure what bit-identity costs on this part.
#
# 2. SPILLS (a report, never a failure).
#
#    AArch32 has 16 Q registers against AArch64's 32, and the whole tile ladder
#    is a search for where the working set stops fitting. Counting vldr/vstr in
#    each kernel attributes the collapse point from the object instead of
#    inferring it from a timing curve.
#
# Usage:
#   Scripts/a32-codegen-check.sh <build-dir>
#
# Looks for the object files CMake produced for Sources/A32Engines. Works on a
# cross build (the usual case) as well as a native one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-}"

if [ -z "${BUILD_DIR}" ] || [ ! -d "${BUILD_DIR}" ]; then
	echo "usage: $(basename "$0") <build-dir>" >&2
	exit 2
fi

OBJDUMP="${OBJDUMP:-}"
if [ -z "${OBJDUMP}" ]; then
	for candidate in arm-linux-gnueabihf-objdump llvm-objdump objdump; do
		if command -v "${candidate}" >/dev/null 2>&1; then
			OBJDUMP="${candidate}"
			break
		fi
	done
fi
if [ -z "${OBJDUMP}" ]; then
	echo "error: no objdump found; set OBJDUMP=" >&2
	exit 2
fi

mapfile -t OBJECTS < <(find "${BUILD_DIR}" -path '*A32Engines*' -name '*.cpp.o' | sort)

if [ "${#OBJECTS[@]}" -eq 0 ]; then
	echo "error: no Sources/A32Engines objects under ${BUILD_DIR}." >&2
	echo "  Configure with -DNAMBENCH_ALL_VARIANTS=ON and build nam_conformance_a32." >&2
	exit 2
fi

printf 'A32 codegen check — %d objects, %s\n\n' "${#OBJECTS[@]}" "${OBJDUMP}"
printf '  %-28s %8s %8s %8s  %s\n' "kernel" "vfma" "vmla" "spills" "verdict"

failures=0

for obj in "${OBJECTS[@]}"; do
	base="$(basename "${obj}" .cpp.o)"
	src="${REPO_ROOT}/Sources/A32Engines/${base}.cpp"

	disasm="$("${OBJDUMP}" -d "${obj}" 2>/dev/null || true)"

	# Count the mnemonics. The AArch64 spellings are matched too, so this script
	# says something useful when the lab is built for the host as a cross-check.
	vfma=$(printf '%s\n' "${disasm}" | grep -ciE '\b(vfma|vfms|fmla|fmls)\b' || true)
	vmla=$(printf '%s\n' "${disasm}" | grep -ciE '\bvml[as]\.f(32|64)\b' || true)
	spills=$(printf '%s\n' "${disasm}" | grep -ciE '\b(vldr|vstr)\b' || true)

	opted_out=no
	if [ -f "${src}" ] && grep -q 'NB_A32_NOT_EXACT' "${src}"; then
		opted_out=yes
	fi

	verdict="ok"
	if [ "${vmla}" -gt 0 ]; then
		if [ "${opted_out}" = "yes" ]; then
			verdict="vmla (declared not exact)"
		else
			verdict="FAIL: non-fused vmla"
			failures=$((failures + 1))
		fi
	fi

	printf '  %-28s %8s %8s %8s  %s\n' "${base}" "${vfma}" "${vmla}" "${spills}" "${verdict}"
done

echo

if [ "${failures}" -gt 0 ]; then
	cat >&2 <<-EOF
		error: ${failures} kernel(s) claiming bit-identity contain a non-fused vmla.

		  VMLA.F32 rounds twice where VFMA.F32 rounds once, so those kernels no longer
		  compute what a2_fast computes, whatever the parity run last said.

		  Usual causes:
		    * the build lost -mfpu=neon-vfpv4 (plain -mfpu=neon has NEON but no FMA,
		      and the compiler then has nothing else to emit)
		    * a kernel used vmlaq_lane_f32 to save an instruction

		  If the second is deliberate, mark the source NB_A32_NOT_EXACT and set its
		  KernelEntry's exact flag to false.
	EOF
	exit 1
fi

echo "all kernels claiming bit-identity are fused-only"
