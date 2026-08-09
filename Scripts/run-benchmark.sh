#!/usr/bin/env bash
#
# Run the portable benchmark with the machine put into a state worth measuring.
#
# The driver itself is careful about the measurement. It cannot be careful about
# the machine, and on Linux the machine is where the numbers usually go wrong:
#
#   * The `ondemand` governor ramps the clock during a pass. A faster engine
#     therefore spends proportionally more of its pass at a low clock than a
#     slower one, which compresses the very ratio the benchmark exists to
#     report. This is not noise that the tightest-70% analysis can reject: it is
#     a systematic bias that points the wrong way.
#
#   * A Raspberry Pi under sustained load will throttle, and a run that started
#     at 45 C and finished throttled did not measure one machine. `vcgencmd`
#     reports it exactly, so it is checked either side rather than hoped about.
#
#   * Handing the run a fixed set of cores keeps the scheduler from migrating it
#     mid-pass between a core with a warm cache and one without.
#
# Everything this changes is restored on exit, including after a Ctrl-C.
#
# Usage:
#   Scripts/run-benchmark.sh [options] [-- <extra nam_benchmark arguments>]
#
#   --build-dir DIR    where nam_benchmark lives (default: build-benchmark)
#   --model PATH       .nam capture (default: the first one found, see below)
#   --audio PATH       input .wav (default: audio-input/input.wav)
#   --output PATH      JSON report (default: benchmark-results/<host>-<stamp>.json)
#   --bmf PATH         also write Bencher Metric Format alongside
#   --cpu-set LIST     taskset list, e.g. 0-3 (default: no pinning)
#   --no-governor      leave the CPU governor alone
#   --keep-governor    set performance and do NOT restore it on exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build-benchmark"
MODEL=""
AUDIO="${REPO_ROOT}/audio-input/input.wav"
OUTPUT=""
BMF=""
CPU_SET=""
TOUCH_GOVERNOR=1
RESTORE_GOVERNOR=1
EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--build-dir) BUILD_DIR="$2"; shift 2 ;;
		--model) MODEL="$2"; shift 2 ;;
		--audio) AUDIO="$2"; shift 2 ;;
		--output) OUTPUT="$2"; shift 2 ;;
		--bmf) BMF="$2"; shift 2 ;;
		--cpu-set) CPU_SET="$2"; shift 2 ;;
		--no-governor) TOUCH_GOVERNOR=0; shift ;;
		--keep-governor) RESTORE_GOVERNOR=0; shift ;;
		--) shift; EXTRA=("$@"); break ;;
		-h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

BINARY="${BUILD_DIR}/nam_benchmark"
[ -x "${BINARY}" ] || die "no nam_benchmark at ${BINARY}
  Build it with:
    cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
    cmake --build ${BUILD_DIR} --target nam_benchmark --parallel"

# --- Model ------------------------------------------------------------------
#
# Prefer a real capture, because that is what every published number was
# measured on. Fall back to upstream's own example model so the script still
# works on a machine that has no captures on it — with a loud note, since the
# two are not comparable.
if [ -z "${MODEL}" ]; then
	MODEL="$(find "${REPO_ROOT}/nam-files" -name '*.nam' 2>/dev/null | sort | head -1 || true)"
	if [ -z "${MODEL}" ]; then
		MODEL="${REPO_ROOT}/vendor/upstream/example_models/A2.nam"
		warn "no capture in nam-files/; falling back to upstream's example_models/A2.nam.
  That is a real A2 shape and fine for tracking this machine against itself, but
  its numbers are not comparable with any published run."
	fi
fi
[ -f "${MODEL}" ] || die "no such model: ${MODEL}"
[ -f "${AUDIO}" ] || die "no such audio: ${AUDIO}"

HOST="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -z "${OUTPUT}" ]; then
	mkdir -p "${REPO_ROOT}/benchmark-results"
	OUTPUT="${REPO_ROOT}/benchmark-results/${HOST}-${STAMP}.json"
fi

# --- Governor ---------------------------------------------------------------

GOVERNOR_FILES=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
SAVED_GOVERNOR=""

# These are root-owned sysfs files, so an unprivileged `echo x > file` fails in
# the *shell* before the command runs. That failure message is not the command's
# stderr and is not silenced by redirecting it, and — more importantly — writing
# the loop the obvious way lets a failed write scroll past while the run
# continues at the wrong clock. Hence: test first, escalate deliberately, and
# read the value back to confirm.
write_governor() {
	local value="$1" f
	for f in "${GOVERNOR_FILES[@]}"; do
		[ -e "$f" ] || continue
		if [ -w "$f" ]; then
			printf '%s\n' "${value}" > "$f" || return 1
		elif command -v sudo >/dev/null 2>&1; then
			printf '%s\n' "${value}" | sudo tee "$f" >/dev/null || return 1
		else
			return 1
		fi
	done
	[ "$(cat "${GOVERNOR_FILES[0]}")" = "${value}" ]
}

restore_governor() {
	if [ -n "${SAVED_GOVERNOR}" ] && [ "${RESTORE_GOVERNOR}" -eq 1 ]; then
		log "restoring governor to ${SAVED_GOVERNOR}"
		write_governor "${SAVED_GOVERNOR}" || warn "could not restore the governor"
	fi
}
trap restore_governor EXIT INT TERM

if [ "${TOUCH_GOVERNOR}" -eq 1 ] && [ -r "${GOVERNOR_FILES[0]}" ] 2>/dev/null; then
	CURRENT_GOVERNOR="$(cat "${GOVERNOR_FILES[0]}")"
	if [ "${CURRENT_GOVERNOR}" != "performance" ]; then
		log "governor is ${CURRENT_GOVERNOR}; switching to performance for this run"
		if write_governor performance; then
			SAVED_GOVERNOR="${CURRENT_GOVERNOR}"
		else
			die "could not set the CPU governor to performance.

  This is refused rather than warned about, because '${CURRENT_GOVERNOR}' ramps the
  clock during a pass: a faster engine spends proportionally more of its pass at
  a low clock than a slower one, which biases the ratio the benchmark reports.
  That is not noise the analysis can reject.

  Give this user passwordless sudo, or run:
      sudo Scripts/run-benchmark.sh ...
  or, if you really do want to measure the machine as it is configured:
      Scripts/run-benchmark.sh --no-governor ..."
		fi
	else
		log "governor already performance"
	fi
fi

# --- Thermal ----------------------------------------------------------------

throttle_state() {
	command -v vcgencmd >/dev/null 2>&1 && vcgencmd get_throttled 2>/dev/null || true
}

BEFORE_THROTTLED="$(throttle_state)"
if [ -n "${BEFORE_THROTTLED}" ] && [ "${BEFORE_THROTTLED}" != "throttled=0x0" ]; then
	warn "this machine is ALREADY reporting ${BEFORE_THROTTLED} before the run started.
  Bits 0-3 mean it is throttling now; bits 16-19 mean it has since boot.
  Let it cool, or fit a fan, before trusting anything measured here."
fi

# --- Run --------------------------------------------------------------------

RUNNER=()
if [ -n "${CPU_SET}" ]; then
	command -v taskset >/dev/null 2>&1 \
		|| die "--cpu-set needs taskset (apt install util-linux)"
	RUNNER=(taskset -c "${CPU_SET}")
	log "pinned to CPUs ${CPU_SET}"
fi

log "model $(basename "${MODEL}")"
log "audio $(basename "${AUDIO}")"

set +e
"${RUNNER[@]}" "${BINARY}" \
	--model "${MODEL}" \
	--audio "${AUDIO}" \
	--output "${OUTPUT}" \
	--note "${HOST}${CPU_SET:+ cpus=${CPU_SET}}" \
	"${EXTRA[@]}"
STATUS=$?
set -e

AFTER_THROTTLED="$(throttle_state)"
if [ -n "${AFTER_THROTTLED}" ]; then
	log "throttle state: ${BEFORE_THROTTLED:-unknown} -> ${AFTER_THROTTLED}"
	if [ "${AFTER_THROTTLED}" != "throttled=0x0" ] && [ "${BEFORE_THROTTLED}" = "throttled=0x0" ]; then
		warn "the machine started throttling DURING this run (${AFTER_THROTTLED}).
  The numbers in ${OUTPUT} describe a machine that changed speed while being
  measured. Treat them as void."
	fi
fi

# --- Bencher Metric Format --------------------------------------------------

if [ -n "${BMF}" ]; then
	if [ "${STATUS}" -ne 0 ]; then
		warn "not writing BMF: at least one variant produced no trustworthy result"
	else
		python3 "${REPO_ROOT}/Scripts/bencher-report.py" "${OUTPUT}" --output "${BMF}"
	fi
fi

exit "${STATUS}"
