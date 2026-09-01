#!/usr/bin/env bash
#
# Find the clock this board can actually hold, before trusting any number it
# produces.
#
# Runs on the board itself. Puts the machine under the load the benchmark will
# put it under, for long enough to reach thermal equilibrium, and samples what
# happens to the clock. It answers one question: what is the highest
# scaling_max_freq at which this machine never drops a frequency step?
#
# Why this comes before any kernel work
#
#   The benchmark reports a *ratio* between engines. A clock that sags partway
#   through a run does not add noise that the tightest-70% analysis can reject —
#   it charges whichever engine happened to be running at the time. That is a
#   systematic bias, and it points in whichever direction the run order happened
#   to arrange. So a stable clock matters far more here than a high one.
#
#   The Tinker Board makes this pressing rather than theoretical: it is
#   passively cooled, its thermal zone trips passive at 70 C against a ~43 C
#   idle, and all four cores share ONE cpufreq policy, so --cpu-set is no
#   defence. A Pi at least reports throttling through vcgencmd; this board has
#   no such register, which is why Scripts/run-benchmark.sh grew a residency
#   check.
#
# Usage (on the board):
#   Scripts/a32-thermal-soak.sh [options]
#
#   --build-dir DIR   where nam_benchmark lives (default: bin)
#   --minutes N       soak length (default: 20)
#   --interval N      seconds between samples (default: 5)
#   --freq KHZ        cap scaling_max_freq to this first; "none" leaves it
#                     (default: none, i.e. characterise the board as configured)
#   --model PATH      .nam capture (default: first in nam-files/)
#   --audio PATH      input .wav (default: audio-input/input.wav)
#   --csv PATH        where to write samples (default: soak-<stamp>.csv)
#   --load N          concurrent nam_benchmark instances (default: 1)
#
# Why the default is one instance and not one per core
#
#   A measured run loads exactly one core: nam_benchmark is single-threaded, and
#   --cpu-set only bounds where that thread may go. So the soak that decides
#   whether a *measurement* is trustworthy has to reproduce that condition, and
#   a four-core soak would find a cap lower than the lab actually needs.
#
#   --load 4 answers a different and also real question, which is what the
#   HeadRush Core does in service: a pedal runs other DSP blocks on the other
#   cores while NAM runs on one of them, and all four share one cpufreq policy
#   and one thermal zone. If the board holds its clock at --load 1 but not at
#   --load 4, that is a fact about the product rather than about the lab, and
#   it belongs in the write-up either way.
#
# A typical characterisation is three runs: as configured, then capped one and
# two steps down, until one of them holds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="bin"
MINUTES=20
INTERVAL=5
FREQ="none"
MODEL=""
AUDIO="${REPO_ROOT}/audio-input/input.wav"
CSV=""
LOAD=1

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--build-dir) BUILD_DIR="$2"; shift 2 ;;
		--minutes) MINUTES="$2"; shift 2 ;;
		--interval) INTERVAL="$2"; shift 2 ;;
		--freq) FREQ="$2"; shift 2 ;;
		--model) MODEL="$2"; shift 2 ;;
		--audio) AUDIO="$2"; shift 2 ;;
		--csv) CSV="$2"; shift 2 ;;
		--load) LOAD="$2"; shift 2 ;;
		-h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

BINARY="${BUILD_DIR}/nam_benchmark"
[ -x "${BINARY}" ] || die "no nam_benchmark at ${BINARY}"

if [ -z "${MODEL}" ]; then
	MODEL="$(find "${REPO_ROOT}/nam-files" -name '*.nam' 2>/dev/null | sort | head -1 || true)"
	[ -n "${MODEL}" ] || die "no .nam in nam-files/"
fi
[ -f "${AUDIO}" ] || die "no such audio: ${AUDIO}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CSV="${CSV:-soak-${STAMP}.csv}"

POLICY=""
for p in /sys/devices/system/cpu/cpufreq/policy0 /sys/devices/system/cpu/cpu0/cpufreq; do
	[ -d "$p" ] && POLICY="$p" && break
done
[ -n "${POLICY}" ] || die "no cpufreq policy found; nothing to characterise"

# --- Machine state, restored on every exit path -----------------------------

SAVED_GOV=""
SAVED_MAX=""

write_sysfs() {
	local file="$1" value="$2"
	[ -e "${file}" ] || return 1
	if [ -w "${file}" ]; then
		printf '%s\n' "${value}" > "${file}"
	else
		printf '%s\n' "${value}" | sudo tee "${file}" >/dev/null
	fi
}

restore() {
	[ -n "${SAVED_MAX}" ] && write_sysfs "${POLICY}/scaling_max_freq" "${SAVED_MAX}" 2>/dev/null || true
	local f
	if [ -n "${SAVED_GOV}" ]; then
		for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
			[ -e "$f" ] && write_sysfs "$f" "${SAVED_GOV}" 2>/dev/null || true
		done
	fi
	for p in ${LOAD_PIDS:-}; do kill "$p" 2>/dev/null || true; done
}
trap restore EXIT INT TERM

SAVED_GOV="$(cat "${POLICY}/scaling_governor" 2>/dev/null || true)"
if [ "${SAVED_GOV}" != "performance" ]; then
	log "governor ${SAVED_GOV} -> performance"
	for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
		write_sysfs "$f" performance || die "could not set the governor to performance.
  Give this user passwordless sudo for the cpufreq sysfs files, or run under sudo."
	done
else
	SAVED_GOV=""
fi

if [ "${FREQ}" != "none" ]; then
	SAVED_MAX="$(cat "${POLICY}/scaling_max_freq")"
	log "capping scaling_max_freq ${SAVED_MAX} -> ${FREQ}"
	write_sysfs "${POLICY}/scaling_max_freq" "${FREQ}" \
		|| die "could not cap scaling_max_freq"
fi

TARGET_FREQ="$(cat "${POLICY}/scaling_max_freq")"

# --- Load -------------------------------------------------------------------
#
# The load is the benchmark itself, not a synthetic spinner: what matters is the
# thermal behaviour under the instruction mix that will actually be measured.
# NEON-heavy fp32 work does not heat a core the way an integer spinner does.

# Sampling scaling_cur_freq every few seconds can miss a throttle entirely: the
# thermal governor can drop a step and recover between two samples and leave no
# trace in the CSV. The kernel's own time_in_state is a cumulative jiffy count
# per frequency, so diffing it across the soak catches every excursion no matter
# how brief. The sampler stays because it says *when*; this says *whether*.
RESIDENCY_FILE=""
for f in "${POLICY}/stats/time_in_state" \
	/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state; do
	[ -r "$f" ] && RESIDENCY_FILE="$f" && break
done
BEFORE_RESIDENCY=""
[ -n "${RESIDENCY_FILE}" ] && BEFORE_RESIDENCY="$(cat "${RESIDENCY_FILE}")"
[ -n "${RESIDENCY_FILE}" ] || warn "no cpufreq time_in_state on this machine; the
  verdict rests on the 5-second sampler alone, which can miss a brief drop."

log "soaking for ${MINUTES} min at up to ${TARGET_FREQ} kHz"
log "load: ${LOAD} concurrent $(basename "${MODEL}")"

LOAD_PIDS=""
for _n in $(seq 1 "${LOAD}"); do
	"${BINARY}" --model "${MODEL}" --audio "${AUDIO}" \
		--submodel widest \
		--warmup-seconds 0 \
		--timing-seconds $((MINUTES * 60)) \
		--output /dev/null >/dev/null 2>&1 &
	LOAD_PIDS="${LOAD_PIDS}${LOAD_PIDS:+ }$!"
done
# They all run the same work for the same duration, so watching the first is
# enough to know whether the load is still up.
LOAD_PID="${LOAD_PIDS%% *}"

# --- Sample -----------------------------------------------------------------

read_first() {
	local f
	for f in "$@"; do [ -r "$f" ] && cat "$f" && return 0; done
	echo ""
}

{
	echo "elapsed_s,cur_freq_khz,cpu_temp_c,gpu_temp_c,cooling_state,load_running"
} > "${CSV}"

START="$(date +%s)"
DEADLINE=$((START + MINUTES * 60))
MIN_FREQ="${TARGET_FREQ}"
MAX_TEMP=0
THROTTLED=0

while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
	NOW="$(date +%s)"
	ELAPSED=$((NOW - START))

	CUR="$(read_first "${POLICY}/scaling_cur_freq" "${POLICY}/cpuinfo_cur_freq")"
	CPU_T="$(read_first /sys/class/thermal/thermal_zone0/temp)"
	GPU_T="$(read_first /sys/class/thermal/thermal_zone1/temp)"
	COOL="$(read_first /sys/class/thermal/cooling_device0/cur_state)"
	RUNNING=0; kill -0 "${LOAD_PID}" 2>/dev/null && RUNNING=1

	CPU_C="$(awk -v t="${CPU_T:-0}" 'BEGIN { printf "%.1f", t / 1000 }')"
	GPU_C="$(awk -v t="${GPU_T:-0}" 'BEGIN { printf "%.1f", t / 1000 }')"

	printf '%d,%s,%s,%s,%s,%d\n' \
		"${ELAPSED}" "${CUR:-}" "${CPU_C}" "${GPU_C}" "${COOL:-}" "${RUNNING}" >> "${CSV}"

	[ -n "${CUR}" ] && [ "${CUR}" -lt "${MIN_FREQ}" ] && MIN_FREQ="${CUR}"
	awk -v a="${CPU_C}" -v b="${MAX_TEMP}" 'BEGIN { exit !(a > b) }' && MAX_TEMP="${CPU_C}"
	[ -n "${COOL}" ] && [ "${COOL}" -gt 0 ] && THROTTLED=1

	printf '\r  %4ds  %s kHz  %s C  cooling=%s  ' \
		"${ELAPSED}" "${CUR:-?}" "${CPU_C}" "${COOL:-?}"

	if [ "${RUNNING}" -eq 0 ]; then
		echo
		warn "the load exited early at ${ELAPSED}s; soak is incomplete"
		break
	fi

	sleep "${INTERVAL}"
done
echo

for p in ${LOAD_PIDS}; do kill "$p" 2>/dev/null || true; done
for p in ${LOAD_PIDS}; do wait "$p" 2>/dev/null || true; done

# --- Verdict ----------------------------------------------------------------

DROPS=""
if [ -n "${RESIDENCY_FILE}" ] && [ -n "${BEFORE_RESIDENCY}" ]; then
	DROPS="$(awk -v intended="${TARGET_FREQ}" '
		NR == FNR { was[$1] = $2; next }
		$1 + 0 < intended + 0 && $2 - (($1 in was) ? was[$1] : 0) > 0 {
			printf "    %d kHz  +%d jiffies\n", $1, $2 - (($1 in was) ? was[$1] : 0)
		}
	' <(printf '%s\n' "${BEFORE_RESIDENCY}") "${RESIDENCY_FILE}")"
fi

log "samples in ${CSV}"
log "peak CPU temperature: ${MAX_TEMP} C"
log "lowest clock seen:    ${MIN_FREQ} kHz (target ${TARGET_FREQ})"
if [ -n "${DROPS}" ]; then
	log "time spent below ${TARGET_FREQ} kHz:"
	printf '%s\n' "${DROPS}"
elif [ -n "${RESIDENCY_FILE}" ]; then
	log "no jiffies recorded below ${TARGET_FREQ} kHz for the whole soak"
fi

if [ "${THROTTLED}" -eq 1 ] || [ "${MIN_FREQ}" -lt "${TARGET_FREQ}" ] \
	|| [ -n "${DROPS}" ]; then
	warn "this board CANNOT hold ${TARGET_FREQ} kHz under sustained load.

  Nothing measured at this cap is trustworthy: the clock moved while the
  benchmark was timing, which biases the ratio between engines rather than
  merely adding rejectable noise.

  Re-run with --freq set one step lower, until a soak holds. Then use that
  frequency for every measured run and record it in the report note. Or fit a
  fan, which is the better answer if you want the part's real peak."
	exit 1
fi

log "held ${TARGET_FREQ} kHz for the whole soak — this cap is safe to measure at"
