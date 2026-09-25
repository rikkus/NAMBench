#!/usr/bin/env bash
#
# Cross-build for 32-bit ARM here, run it on the board, bring the results back.
#
# The RK3288 has four slow cores and 2 GB of RAM, so building Eigen at -O3 on it
# is not a sensible use of an afternoon. The supported route is to build on this
# machine and copy the binaries: the ARMv7 targets static-link libstdc++ and
# libgcc (see nb_conformance_flags in CMakeLists.txt) precisely so the board
# needs no toolchain, no runtime and no checkout of its own.
#
# The edit-measure loop runs dozens of times across a kernel campaign, so it is
# one command rather than four remembered ones.
#
# What is deliberately NOT here: the measurement. Scripts/run-benchmark.sh owns
# the governor, the thermal guard and the timing protocol, and it runs *on the
# board* — it is copied across with everything else. Duplicating any of that
# here would give two places where a run can be set up differently.
#
# Usage:
#   Scripts/a32-deploy.sh [options] [-- <extra arguments for the remote step>]
#
#   --host HOST         board to deploy to (default: tib)
#   --remote-dir DIR    where to put it (default: ~/nambench-a32)
#   --build-dir DIR     local build tree (default: build-a32)
#   --toolchain FILE    cmake toolchain (default: the gcc one; pass the clang
#                       one for the compiler sweep)
#   --jobs N            build parallelism (default: nproc)
#   --step STEP         what to do on the board, one of:
#                         bench       run-benchmark.sh (default)
#                         conformance nam_conformance_* for every variant
#                         none        deploy only, run nothing
#   --cpu-set LIST      passed through to run-benchmark.sh (default: 0-3)
#   --bmf PATH          convert the reports this run brought back into Bencher
#                       Metric Format, here. --step bench only.
#   --no-build          skip configure+build, deploy what is already there
#   --no-fetch          do not copy results back
#
# Everything after `--` is passed to the remote step verbatim, so a kernel sweep
# is:
#
#   Scripts/a32-deploy.sh -- --a32 all --block-size 64
#
# The BMF is written *here* rather than on the board, which is why --bmf exists
# at all instead of passing run-benchmark.sh's own --bmf through: the board has
# no checkout, so it has no Scripts/bencher-report.py to run, and it has no
# business holding a Bencher API key either. Everything after the measurement —
# conversion, upload, provenance — happens on the machine that owns the git
# history the result is attributed to. Scripts/track-benchmark.sh drives that
# end; this flag is the seam between the two.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HOST="tib"
REMOTE_DIR="nambench-a32"

# A measurement run holds one ssh session open for half an hour with nothing to
# say on it. If the connection dies in that window the remote side is gone but
# the local ssh waits forever, and the run looks like it is still measuring when
# the board has been idle for twenty minutes. Keepalives turn that into an error.
SSH_OPTS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=4)
BUILD_DIR="${REPO_ROOT}/build-a32"
TOOLCHAIN="${REPO_ROOT}/cmake/toolchain-armv7-linux-gnueabihf.cmake"
JOBS="$(nproc 2>/dev/null || echo 4)"
STEP="bench"
CPU_SET="0-3"
BMF=""
DO_BUILD=1
DO_FETCH=1
EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--host) HOST="$2"; shift 2 ;;
		--remote-dir) REMOTE_DIR="$2"; shift 2 ;;
		--build-dir) BUILD_DIR="$2"; shift 2 ;;
		--toolchain) TOOLCHAIN="$2"; shift 2 ;;
		--jobs) JOBS="$2"; shift 2 ;;
		--step) STEP="$2"; shift 2 ;;
		--cpu-set) CPU_SET="$2"; shift 2 ;;
		--bmf) BMF="$2"; shift 2 ;;
		--no-build) DO_BUILD=0; shift ;;
		--no-fetch) DO_FETCH=0; shift ;;
		--) shift; EXTRA=("$@"); break ;;
		# Everything from line 2 up to the first line that is not a comment,
		# the way Scripts/track-benchmark.sh does it. A hardcoded line range
		# silently truncates the help the moment the header grows, which adding
		# --bmf above has just done.
		-h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
			"${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

case "${STEP}" in
	bench|conformance|none) ;;
	*) die "--step must be bench, conformance or none (got '${STEP}')" ;;
esac

if [ -n "${BMF}" ]; then
	[ "${STEP}" = "bench" ] || die "--bmf converts benchmark reports, so it needs --step bench (got '${STEP}')"
	[ "${DO_FETCH}" -eq 1 ] || die "--bmf and --no-fetch ask for opposite things:
  one converts the reports this run brought back, the other leaves them on the board."
	command -v python3 >/dev/null || die "--bmf needs python3 to run Scripts/bencher-report.py"
fi

[ -f "${TOOLCHAIN}" ] || die "no toolchain file at ${TOOLCHAIN}"

# --- Build ------------------------------------------------------------------

if [ "${DO_BUILD}" -eq 1 ]; then
	command -v arm-linux-gnueabihf-g++ >/dev/null 2>&1 \
		|| die "no armhf cross compiler.
  apt install g++-arm-linux-gnueabihf qemu-user-static"

	log "configuring ${BUILD_DIR}"
	cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
		-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
		-DCMAKE_BUILD_TYPE=Release \
		-DNAMBENCH_ALL_VARIANTS=ON \
		-DNAMBENCH_BUILD_BENCHMARK=ON

	log "building with -j${JOBS}"
	cmake --build "${BUILD_DIR}" -j"${JOBS}"

	# Every build, not just at the end. A kernel that claims bit-identity and
	# emits a non-fused vmla is wrong from the moment it is written, and finding
	# that out here costs seconds where finding it out from a parity mismatch
	# after a sweep costs an afternoon.
	if [ -x "${REPO_ROOT}/Scripts/a32-codegen-check.sh" ]; then
		log "codegen check"
		"${REPO_ROOT}/Scripts/a32-codegen-check.sh" "${BUILD_DIR}"
	fi
fi

BINARIES=()
for b in "${BUILD_DIR}"/nam_benchmark "${BUILD_DIR}"/nam_ir_benchmark \
	"${BUILD_DIR}"/nam_conformance_*; do
	[ -x "$b" ] && [ -f "$b" ] && BINARIES=("${BINARIES[@]+${BINARIES[@]}}" "$b")
done
[ "${#BINARIES[@]}" -gt 0 ] || die "nothing executable in ${BUILD_DIR}; build first"

# The variant libraries the benchmarks link against live beside them: the NAM
# engines for nam_benchmark, the Linear and AudioDSPTools ones for
# nam_ir_benchmark.
LIBS=()
while IFS= read -r l; do LIBS=("${LIBS[@]+${LIBS[@]}}" "$l"); done < <(
	find "${BUILD_DIR}" -maxdepth 1 \( -name 'libnam_engine_*' -o -name 'libnam_ir_*' \) \
		-type f 2>/dev/null || true
)

# --- Deploy -----------------------------------------------------------------

log "deploying to ${HOST}:${REMOTE_DIR}"
ssh "${SSH_OPTS[@]}" "${HOST}" "mkdir -p '${REMOTE_DIR}/bin' '${REMOTE_DIR}/Scripts' \
	'${REMOTE_DIR}/nam-files' '${REMOTE_DIR}/audio-input' \
	'${REMOTE_DIR}/benchmark-results' '${REMOTE_DIR}/conformance-out'"

rsync -az --info=stats0 \
	"${BINARIES[@]}" ${LIBS[@]+"${LIBS[@]}"} "${HOST}:${REMOTE_DIR}/bin/"

# run-benchmark.sh runs on the board and owns the machine state there. It is
# copied every time so an edit to the thermal guard cannot be left behind on
# this side while the board keeps measuring with the old one.
rsync -az --info=stats0 \
	"${REPO_ROOT}/Scripts/run-benchmark.sh" \
	"${REPO_ROOT}/Scripts/a32-thermal-soak.sh" "${HOST}:${REMOTE_DIR}/Scripts/"

rsync -az --info=stats0 --delete \
	"${REPO_ROOT}/nam-files/" "${HOST}:${REMOTE_DIR}/nam-files/"
rsync -az --info=stats0 \
	"${REPO_ROOT}/audio-input/" "${HOST}:${REMOTE_DIR}/audio-input/"

# --- Run --------------------------------------------------------------------

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "${STEP}" in
	none)
		log "deployed; running nothing (--step none)"
		exit 0
		;;
	bench)
		log "measuring on ${HOST}"
		# run-benchmark.sh locates the binary as <build-dir>/nam_benchmark and
		# the inputs relative to the repo root it computes from its own path, so
		# the deployed layout mirrors those two expectations rather than
		# teaching the script a second layout.
		# shellcheck disable=SC2029
		ssh -t "${SSH_OPTS[@]}" "${HOST}" "cd '${REMOTE_DIR}' && Scripts/run-benchmark.sh \
			--build-dir bin \
			--cpu-set '${CPU_SET}' \
			--output-dir benchmark-results \
			${EXTRA[*]+${EXTRA[*]}}"
		;;
	conformance)
		# The same capture every variant sees, chosen here rather than on the
		# board so a comparison can never be between two different models.
		# Matches run-benchmark.sh's rule: first capture by name, else
		# upstream's example.
		MODEL_NAME="$(find "${REPO_ROOT}/nam-files" -name '*.nam' -printf '%f\n' 2>/dev/null | sort | head -1 || true)"
		[ -n "${MODEL_NAME}" ] || die "no .nam in nam-files/ to run conformance against"

		log "conformance on ${HOST} against ${MODEL_NAME}"
		# shellcheck disable=SC2029
		ssh "${SSH_OPTS[@]}" "${HOST}" "cd '${REMOTE_DIR}' && rm -rf conformance-out && \
			mkdir -p conformance-out && \
			for v in bin/nam_conformance_*; do \
				echo \"==> \$(basename \"\$v\")\"; \
				\"\$v\" --model 'nam-files/${MODEL_NAME}' \
					--out conformance-out ${EXTRA[*]+${EXTRA[*]}} || exit 1; \
			done"
		;;
esac

# --- Fetch ------------------------------------------------------------------

if [ "${DO_FETCH}" -eq 1 ]; then
	case "${STEP}" in
		bench)
			mkdir -p "${REPO_ROOT}/benchmark-results"
			# --out-format names every file rsync actually transferred, which is
			# the only reliable way to say which reports came from *this* run.
			# The board's results directory accumulates, and its reports are
			# named after the board's hostname and its own clock, so neither the
			# names nor the mtimes (preserved by -a, from a board whose clock is
			# nobody's idea of authoritative) can be trusted to sort this run's
			# from last week's. rsync already knows; ask it.
			#
			# Through a file rather than a pipe or a process substitution,
			# because both of those throw away rsync's exit status: a fetch that
			# failed would arrive here as an empty list and be reported as "the
			# board produced nothing", which is a different problem with a
			# different fix.
			TRANSFERRED="$(mktemp)"
			trap 'rm -f "${TRANSFERRED}"' EXIT
			# --info=stats0,name rather than letting --out-format imply the
			# name category: whether it does is conditional on --info not having
			# been given, and it has been.
			rsync -az --info=stats0,name --out-format='%n' \
				"${HOST}:${REMOTE_DIR}/benchmark-results/" \
				"${REPO_ROOT}/benchmark-results/" > "${TRANSFERRED}"

			FETCHED=()
			while IFS= read -r f; do
				case "${f}" in
					*.json) FETCHED=("${FETCHED[@]+${FETCHED[@]}}" "${REPO_ROOT}/benchmark-results/${f}") ;;
				esac
			done < "${TRANSFERRED}"
			log "results in benchmark-results/"
			for f in ${FETCHED[@]+"${FETCHED[@]}"}; do
				log "report ${f}"
			done

			if [ -n "${BMF}" ]; then
				[ "${#FETCHED[@]}" -gt 0 ] || die "--bmf, but this run brought back no report.

  The board ran, and nothing new came off it. Either the measurement wrote
  nothing, or these results were already fetched by an earlier run — rsync only
  reports what it actually transferred."
				python3 "${REPO_ROOT}/Scripts/bencher-report.py" \
					"${FETCHED[@]}" --output "${BMF}"
			fi
			;;
		conformance)
			OUT="${REPO_ROOT}/conformance-out/${HOST}-${STAMP}"
			mkdir -p "${OUT}"
			rsync -az --info=stats0 \
				"${HOST}:${REMOTE_DIR}/conformance-out/" "${OUT}/"
			log "conformance dumps in ${OUT}"
			log "compare with: Scripts/compare-conformance.py ${OUT}"
			;;
	esac
fi
