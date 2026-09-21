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
#   --submodels LIST   which to measure (default: widest,narrowest — A2 standard
#                      and A2 nano). Both run inside ONE governor window and one
#                      thermal check, so the machine is in the same state for
#                      each and the pair can be read together.
#
#   --ir               measure impulse-response convolution instead of the
#                      WaveNet engines: nam_ir_benchmark, AudioDSPTools main
#                      against the partitioned-ir branch. Everything this script
#                      does to the machine is the same, which is the point of
#                      putting it here rather than in a script of its own.
#   --taps LIST        --ir only: IR lengths (default 256,512,1024,2048,4096,8192)
#   --blocks LIST      --ir only: block sizes, one report each (default 64).
#                      One report per size because block size is not part of a
#                      Bencher benchmark name, so two of them in one report
#                      would overwrite each other.
#   --output-dir DIR   where reports go (default: benchmark-results)
#   --bmf PATH         merge every submodel into one Bencher Metric Format file
#   --cpu-set LIST     taskset list, e.g. 0-3 (default: no pinning)
#   --max-freq KHZ     cap scaling_max_freq for the run and restore it after.
#                      Use the frequency a thermal soak showed this board holds
#                      indefinitely (Scripts/a32-thermal-soak.sh). "none" (the
#                      default) measures the machine as configured.
#   --no-governor      leave the CPU governor alone
#   --keep-governor    set performance and do NOT restore it on exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build-benchmark"
MODEL=""
AUDIO="${REPO_ROOT}/audio-input/input.wav"
SUBMODELS="widest,narrowest"
IR=0
TAPS="256,512,1024,2048,4096,8192"
BLOCKS="64"
OUTPUT_DIR="${REPO_ROOT}/benchmark-results"
BMF=""
CPU_SET=""
TOUCH_GOVERNOR=1
MAX_FREQ="none"
RESTORE_GOVERNOR=1

# Expanded below as ${arr[@]+"${arr[@]}"}: macOS ships bash 3.2, where under
# `set -u` expanding an empty array is an unbound-variable error.
EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--build-dir) BUILD_DIR="$2"; shift 2 ;;
		--model) MODEL="$2"; shift 2 ;;
		--audio) AUDIO="$2"; shift 2 ;;
		--submodels) SUBMODELS="$2"; shift 2 ;;
		--ir) IR=1; shift ;;
		--taps) TAPS="$2"; shift 2 ;;
		--blocks) BLOCKS="$2"; shift 2 ;;
		--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
		--bmf) BMF="$2"; shift 2 ;;
		--cpu-set) CPU_SET="$2"; shift 2 ;;
		--max-freq) MAX_FREQ="$2"; shift 2 ;;
		--no-governor) TOUCH_GOVERNOR=0; shift ;;
		--keep-governor) RESTORE_GOVERNOR=0; shift ;;
		--) shift; EXTRA=("$@"); break ;;
		-h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

if [ "${IR}" -eq 1 ]; then
	TARGET="nam_ir_benchmark"
	# Every block size produces the same benchmark names, so merging two of them
	# into one BMF would silently keep whichever came last. Caught here rather
	# than after the measurement has been spent.
	case "${BLOCKS}" in
		*,*) [ -z "${BMF}" ] || die "--bmf with more than one --blocks value.
  Each block size produces the same benchmark names, so one upload cannot carry
  two of them. Run each size separately, and give each its own --prefix when
  converting (see Scripts/bencher-report.py)." ;;
	esac
else
	TARGET="nam_benchmark"
fi
BINARY="${BUILD_DIR}/${TARGET}"
[ -x "${BINARY}" ] || die "no ${TARGET} at ${BINARY}
  Build it with:
    cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DNAMBENCH_BUILD_BENCHMARK=ON
    cmake --build ${BUILD_DIR} --target ${TARGET} --parallel 4"

# --- Model ------------------------------------------------------------------
#
# Prefer a real capture, because that is what every published number was
# measured on. Fall back to upstream's own example model so the script still
# works on a machine that has no captures on it — with a loud note, since the
# two are not comparable.
#
# An IR run has no .nam at all: the impulse response is generated inside the
# shim, identically in both variants and on every machine.
if [ "${IR}" -eq 0 ]; then
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
fi
[ -f "${AUDIO}" ] || die "no such audio: ${AUDIO}"

HOST="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTPUT_DIR}"

# --- Frequency cap ----------------------------------------------------------
#
# A governor set to `performance` asks for the highest frequency the policy
# allows; it does not stop the thermal governor taking that away again. On a
# passively cooled board the two interact badly: `performance` pins the request
# at the top, the part heats past its passive trip, and cpufreq spends the run
# hunting between steps. The residency check below then correctly voids the run
# — after the full measurement has been spent.
#
# Capping scaling_max_freq to a frequency a soak has shown the board sustains
# turns that after-the-fact void into a run that simply does not throttle. This
# is set before the residency snapshot is taken, so intended_freq() measures
# against the cap rather than against cpuinfo_max_freq.

MAXFREQ_FILES=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq)
SAVED_MAX_FREQ=""

write_max_freq() {
	local value="$1" f
	for f in "${MAXFREQ_FILES[@]}"; do
		[ -e "$f" ] || continue
		if [ -w "$f" ]; then
			printf '%s\n' "${value}" > "$f" || return 1
		elif command -v sudo >/dev/null 2>&1; then
			printf '%s\n' "${value}" | sudo tee "$f" >/dev/null || return 1
		else
			return 1
		fi
	done
	[ "$(cat "${MAXFREQ_FILES[0]}")" = "${value}" ]
}

restore_max_freq() {
	if [ -n "${SAVED_MAX_FREQ}" ]; then
		log "restoring scaling_max_freq to ${SAVED_MAX_FREQ}"
		write_max_freq "${SAVED_MAX_FREQ}" || warn "could not restore scaling_max_freq"
	fi
}

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

restore_machine() {
	restore_governor
	restore_max_freq
}

restore_governor() {
	if [ -n "${SAVED_GOVERNOR}" ] && [ "${RESTORE_GOVERNOR}" -eq 1 ]; then
		log "restoring governor to ${SAVED_GOVERNOR}"
		write_governor "${SAVED_GOVERNOR}" || warn "could not restore the governor"
	fi
}
trap restore_machine EXIT INT TERM

if [ "${MAX_FREQ}" != "none" ]; then
	[ -r "${MAXFREQ_FILES[0]}" ] || die "--max-freq given but this machine has no cpufreq scaling_max_freq"
	CURRENT_MAX="$(cat "${MAXFREQ_FILES[0]}")"
	if [ "${CURRENT_MAX}" != "${MAX_FREQ}" ]; then
		log "capping scaling_max_freq ${CURRENT_MAX} -> ${MAX_FREQ}"
		# Recorded before the write, not after: a write that succeeds for some
		# cores and fails for others must still leave the trap something to put
		# back, or the machine stays capped after this script exits.
		SAVED_MAX_FREQ="${CURRENT_MAX}"
		if write_max_freq "${MAX_FREQ}"; then
			:
		else
			die "could not cap scaling_max_freq to ${MAX_FREQ}.

  This is refused rather than warned about, for the same reason a wrong governor
  is: the cap was asked for because this machine throttles without it, and a run
  that silently measured at ${CURRENT_MAX} kHz instead would be void.

  Give this user passwordless sudo for the cpufreq sysfs files, or run under sudo."
		fi
	else
		log "scaling_max_freq already ${MAX_FREQ}"
	fi
fi

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
#
# Three sources, because no one of them exists everywhere and they answer
# different questions.
#
#   vcgencmd            Pi only. The most direct statement of intent there.
#   time_in_state       the load-bearing one. cpufreq records how many jiffies
#                       were spent at each frequency, so diffing it across the
#                       run says exactly whether the machine changed speed while
#                       it was being measured. Exact, free, and needs no sampler
#                       running alongside the thing being timed.
#   cooling_device      which thermal governor engaged, when one did. Names the
#                       cause that time_in_state only shows the effect of.
#
# The RK3288 board has no vcgencmd and no throttled-bits register, so before
# this it reported nothing at all and the "treat these numbers as void" warning
# below could never fire. That matters more there than on a Pi: the Tinker Board
# is passively cooled, trips passive at 70 C, and drives ALL FOUR cores from one
# cpufreq policy — so --cpu-set is no defence, and a run that quietly dropped a
# frequency step looks like a slightly slow engine rather than a void result.

throttle_state() {
	command -v vcgencmd >/dev/null 2>&1 && vcgencmd get_throttled 2>/dev/null || true
}

# Jiffies per frequency, as "<kHz> <jiffies>" lines, for the policy that owns
# cpu0. Empty when the kernel does not export cpufreq stats.
freq_residency() {
	local f
	for f in /sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state \
		/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state; do
		[ -r "$f" ] && cat "$f" 2>/dev/null && return 0
	done
	return 0
}

# Highest frequency (kHz) the run is allowed to sit at. Not cpuinfo_max_freq:
# scaling_max_freq is what a deliberate cap sets, and measuring against a cap
# the operator chose is the point.
intended_freq() {
	local f
	for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq \
		/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; do
		[ -r "$f" ] && cat "$f" 2>/dev/null && return 0
	done
	return 0
}

# Which frequencies below `intended` gained jiffies between the two snapshots.
# Prints one "<kHz> +<jiffies>" line per offender, nothing when the clock held.
freq_drops() {
	local before="$1" after="$2" intended="$3"
	[ -n "${before}" ] && [ -n "${after}" ] && [ -n "${intended}" ] || return 0
	awk -v intended="${intended}" '
		NR == FNR { was[$1] = $2; next }
		$1 + 0 < intended + 0 && $2 - (($1 in was) ? was[$1] : 0) > 0 {
			printf "    %d kHz  +%d jiffies\n", $1, $2 - (($1 in was) ? was[$1] : 0)
		}
	' <(printf '%s\n' "${before}") <(printf '%s\n' "${after}")
}

cooling_state() {
	local d out=""
	for d in /sys/class/thermal/cooling_device*; do
		[ -r "$d/cur_state" ] || continue
		out="${out}${out:+, }$(cat "$d/type" 2>/dev/null || basename "$d")=$(cat "$d/cur_state")"
	done
	printf '%s' "${out}"
}

cpu_temp() {
	local z out=""
	for z in /sys/class/thermal/thermal_zone*; do
		[ -r "$z/temp" ] || continue
		out="${out}${out:+, }$(cat "$z/type" 2>/dev/null || basename "$z")=$(awk '{printf "%.1fC", $1 / 1000}' "$z/temp")"
	done
	printf '%s' "${out}"
}

BEFORE_THROTTLED="$(throttle_state)"
BEFORE_RESIDENCY="$(freq_residency)"
BEFORE_COOLING="$(cooling_state)"
INTENDED_FREQ="$(intended_freq)"

if [ -n "${BEFORE_THROTTLED}" ] && [ "${BEFORE_THROTTLED}" != "throttled=0x0" ]; then
	warn "this machine is ALREADY reporting ${BEFORE_THROTTLED} before the run started.
  Bits 0-3 mean it is throttling now; bits 16-19 mean it has since boot.
  Let it cool, or fit a fan, before trusting anything measured here."
fi
if [ -n "${BEFORE_COOLING}" ]; then
	log "cooling state before: ${BEFORE_COOLING}"
	case "${BEFORE_COOLING}" in
		*=0*) ;;
	esac
	if printf '%s' "${BEFORE_COOLING}" | grep -qE '=[1-9]'; then
		warn "a thermal governor is ALREADY capping this machine (${BEFORE_COOLING}).
  Let it cool before measuring: everything below starts from a throttled clock."
	fi
fi
[ -n "$(cpu_temp)" ] && log "temperature before: $(cpu_temp)"
[ -n "${INTENDED_FREQ}" ] && log "intended clock: ${INTENDED_FREQ} kHz"

# --- Run --------------------------------------------------------------------

RUNNER=()
if [ -n "${CPU_SET}" ]; then
	command -v taskset >/dev/null 2>&1 \
		|| die "--cpu-set needs taskset (apt install util-linux)"
	RUNNER=(taskset -c "${CPU_SET}")
	log "pinned to CPUs ${CPU_SET}"
fi

if [ "${IR}" -eq 1 ]; then
	log "impulse responses of ${TAPS} taps"
else
	log "model $(basename "${MODEL}")"
fi
log "audio $(basename "${AUDIO}")"

# The clock the run was held at goes in the report, not just in this terminal.
# A report read months later has to be able to say which frequency it describes:
# on a board that is deliberately capped below its own ceiling, the same engine
# is a different number at 1416 MHz than at 1800, and nothing else in the JSON
# records the cap.
NOTE="${HOST}${CPU_SET:+ cpus=${CPU_SET}}"
[ "${MAX_FREQ}" != "none" ] && NOTE="${NOTE} maxfreq=${MAX_FREQ}kHz"

# Every submodel runs inside the one governor window and between the one pair of
# thermal readings, so A2 standard and A2 nano describe the same machine in the
# same state and can honestly be read side by side.
STATUS=0
REPORTS=()

# In --ir mode the iteration is over block sizes instead of submodels, for the
# same reason the WaveNet run iterates submodels here rather than being invoked
# twice: everything measured inside this loop describes one machine in one
# state. One report per block size, because block size is not part of a Bencher
# benchmark name and two of them in one report would overwrite each other.
if [ "${IR}" -eq 1 ]; then
	LOOP="${BLOCKS}"
else
	LOOP="${SUBMODELS}"
fi

OLD_IFS="${IFS}"
IFS=','
for ITEM in ${LOOP}; do
	IFS="${OLD_IFS}"
	[ -n "${ITEM}" ] || continue

	set +e
	if [ "${IR}" -eq 1 ]; then
		REPORT="${OUTPUT_DIR}/${HOST}-${STAMP}-ir-block${ITEM}.json"
		log "measuring impulse responses, ${ITEM}-frame blocks"
		${RUNNER[@]+"${RUNNER[@]}"} "${BINARY}" \
			--audio "${AUDIO}" \
			--taps "${TAPS}" \
			--blocks "${ITEM}" \
			--output "${REPORT}" \
			--note "${NOTE}" \
			${EXTRA[@]+"${EXTRA[@]}"}
	else
		REPORT="${OUTPUT_DIR}/${HOST}-${STAMP}-${ITEM}.json"
		log "measuring ${ITEM}"
		${RUNNER[@]+"${RUNNER[@]}"} "${BINARY}" \
			--model "${MODEL}" \
			--audio "${AUDIO}" \
			--submodel "${ITEM}" \
			--output "${REPORT}" \
			--note "${NOTE}" \
			${EXTRA[@]+"${EXTRA[@]}"}
	fi
	ONE_STATUS=$?
	set -e

	[ "${ONE_STATUS}" -eq 0 ] || STATUS="${ONE_STATUS}"
	[ -s "${REPORT}" ] && REPORTS=("${REPORTS[@]+${REPORTS[@]}}" "${REPORT}")

	IFS=','
done
IFS="${OLD_IFS}"

AFTER_THROTTLED="$(throttle_state)"
if [ -n "${AFTER_THROTTLED}" ]; then
	log "throttle state: ${BEFORE_THROTTLED:-unknown} -> ${AFTER_THROTTLED}"
	if [ "${AFTER_THROTTLED}" != "throttled=0x0" ] && [ "${BEFORE_THROTTLED}" = "throttled=0x0" ]; then
		warn "the machine started throttling DURING this run (${AFTER_THROTTLED}).
  Everything measured here describes a machine that changed speed while it was
  being measured. Treat these numbers as void."
	fi
fi

AFTER_COOLING="$(cooling_state)"
[ -n "${AFTER_COOLING}" ] && log "cooling state: ${BEFORE_COOLING:-unknown} -> ${AFTER_COOLING}"
[ -n "$(cpu_temp)" ] && log "temperature after: $(cpu_temp)"

# The verdict. A frequency below the cap gaining jiffies during the run means
# the clock moved while the benchmark was timing, which biases the ratio between
# engines rather than merely adding noise the tightest-70% analysis can reject:
# whichever engine happened to be running while the clock was low is charged for
# it. So this is stated as void, not as a caveat.
DROPS="$(freq_drops "${BEFORE_RESIDENCY}" "$(freq_residency)" "${INTENDED_FREQ}")"
if [ -n "${DROPS}" ]; then
	warn "the clock DROPPED below ${INTENDED_FREQ} kHz during this run:
${DROPS}
  The machine changed speed while it was being measured, so the ratio between
  variants is biased towards whichever ran at the lower clock. Treat these
  numbers as void.

  Fix the cause rather than re-running until it passes: fit a fan, or cap
  scaling_max_freq to a frequency this machine can hold indefinitely. A stable
  clock matters far more here than a high one, because what is being reported is
  a ratio."
	STATUS=1
elif [ -n "${BEFORE_RESIDENCY}" ] && [ -n "${INTENDED_FREQ}" ]; then
	log "clock held at ${INTENDED_FREQ} kHz throughout"
fi

for r in ${REPORTS[@]+"${REPORTS[@]}"}; do
	log "report ${r}"
done

# --- Bencher Metric Format --------------------------------------------------

if [ -n "${BMF}" ]; then
	if [ "${STATUS}" -ne 0 ]; then
		warn "not writing BMF: at least one variant produced no trustworthy result"
	elif [ "${#REPORTS[@]}" -eq 0 ]; then
		warn "not writing BMF: no reports were produced"
	else
		python3 "${REPO_ROOT}/Scripts/bencher-report.py" \
			${REPORTS[@]+"${REPORTS[@]}"} --output "${BMF}"
	fi
fi

exit "${STATUS}"
