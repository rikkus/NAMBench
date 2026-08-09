#!/usr/bin/env bash
#
# Measure this machine and send the result to Bencher, by hand.
#
# The same thing .github/workflows/benchmark.yml does, minus the runner. It
# deliberately picks the same driver and the same testbed name the workflow
# would, so runs made this way and runs made later by a self-hosted runner land
# in one continuous history rather than two forked ones.
#
#   macOS  ->  nambench, built by Xcode. The canonical tool: every published
#              number came from it.
#   Linux  ->  nam_benchmark, via Scripts/run-benchmark.sh, which also handles
#              the CPU governor and the thermal check.
#
# The API token is never passed on the command line, never written to a file and
# never echoed. It comes from the environment, or straight out of 1Password into
# this process:
#
#   export BENCHER_API_TOKEN="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
#
# or set BENCHER_OP_REF to that op:// reference and let this script do it.
#
# Usage:
#   Scripts/track-benchmark.sh [options] [-- <extra driver arguments>]
#
#   --project SLUG      Bencher project (default: $BENCHER_PROJECT)
#   --testbed NAME      this machine's testbed (default: detected, see below)
#   --branch NAME       branch to record against (default: the current one)
#   --submodel WHICH    widest (default) or narrowest
#   --timing-seconds N  timing window per variant (default: 30)
#   --cpu-set LIST      Linux only, taskset list, e.g. 0-3
#   --dry-run           measure and convert, but do not upload
#   --fail-on-alert     exit non-zero if Bencher raises an alert
#   --thresholds        install/refresh the regression thresholds

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROJECT="${BENCHER_PROJECT:-}"
TESTBED=""
BRANCH=""
SUBMODEL="widest"
TIMING="30"
CPU_SET=""
DRY_RUN=0
FAIL_ON_ALERT=0
SET_THRESHOLDS=0
EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--project) PROJECT="$2"; shift 2 ;;
		--testbed) TESTBED="$2"; shift 2 ;;
		--branch) BRANCH="$2"; shift 2 ;;
		--submodel) SUBMODEL="$2"; shift 2 ;;
		--timing-seconds) TIMING="$2"; shift 2 ;;
		--cpu-set) CPU_SET="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		--fail-on-alert) FAIL_ON_ALERT=1; shift ;;
		--thresholds) SET_THRESHOLDS=1; shift ;;
		--) shift; EXTRA=("$@"); break ;;
		-h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

# --- Which machine is this? -------------------------------------------------
#
# The names have to match .github/workflows/benchmark.yml exactly. A typo here
# does not fail — Bencher creates testbeds on demand — it silently starts a
# second history for the same machine, which is worse.
detect_testbed() {
	case "$(uname -s)" in
		Darwin)
			case "$(sysctl -n hw.model 2>/dev/null)" in
				Mac14,15) echo "m2-air" ;;
				MacBookAir10,1) echo "m1-air" ;;
				*) echo "" ;;
			esac
			;;
		Linux)
			local model
			model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
			case "${model}" in
				*"Raspberry Pi 500"*) echo "pi500" ;;
				*"Raspberry Pi 5"*) echo "pi5" ;;
				*"Raspberry Pi 4"*) echo "pi4" ;;
				*) echo "" ;;
			esac
			;;
		*) echo "" ;;
	esac
}

if [ -z "${TESTBED}" ]; then
	TESTBED="$(detect_testbed)"
	[ -n "${TESTBED}" ] || die "could not work out which machine this is.

  Pass --testbed explicitly, using the same name the workflow uses for it:
      m2-air, m1-air, pi500
  A new name is not an error — Bencher will create it — which is exactly why
  getting it wrong quietly starts a second history for one machine."
	log "detected testbed: ${TESTBED}"
fi

# --- Provenance -------------------------------------------------------------

if [ -z "${BRANCH}" ]; then
	BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
fi
HASH="$(git rev-parse HEAD 2>/dev/null || true)"

# A measurement is attributed to a commit. If the tree does not match that
# commit, the attribution is a lie, and it is a lie that survives in the history
# long after the working copy is gone.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
	warn "the working tree has uncommitted changes.

  This result will be recorded against ${HASH:0:12}, which is not the code that
  produced it. Fine while you are setting Bencher up; commit before you record
  anything you intend to compare against later."
fi

# --- Measure ----------------------------------------------------------------

REPORT="${REPO_ROOT}/benchmark-results/track-${TESTBED}-$(date -u +%Y%m%dT%H%M%SZ).json"
BMF="${REPORT%.json}.bmf.json"
mkdir -p "${REPO_ROOT}/benchmark-results"

case "$(uname -s)" in
	Darwin)
		log "driver: nambench (Xcode)"
		command -v xcodegen >/dev/null || die "xcodegen is not installed: brew install xcodegen"
		[ -d NAMBench.xcodeproj ] || xcodegen generate
		xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli \
			-configuration Release build >/dev/null \
			|| die "xcodebuild failed"
		PRODUCTS="$(xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli \
			-configuration Release -showBuildSettings \
			| awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')"

		# The CLI names its own output file, so point it at a scratch directory
		# and pick up what it wrote.
		OUTDIR="$(mktemp -d)"
		trap 'rm -rf "${OUTDIR}"' EXIT
		"${PRODUCTS}/nambench" \
			--submodel "${SUBMODEL}" \
			--timing-seconds "${TIMING}" \
			--output "${OUTDIR}" \
			"${EXTRA[@]}"
		cp "$(ls -t "${OUTDIR}"/*.json | head -1)" "${REPORT}"
		python3 Scripts/bencher-report.py "${REPORT}" --output "${BMF}"
		;;

	Linux)
		log "driver: nam_benchmark (portable)"
		cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
			-DNAMBENCH_BUILD_BENCHMARK=ON >/dev/null
		cmake --build build-benchmark --target nam_benchmark --parallel >/dev/null
		./Scripts/run-benchmark.sh \
			--build-dir build-benchmark \
			--output "${REPORT}" \
			--bmf "${BMF}" \
			${CPU_SET:+--cpu-set "${CPU_SET}"} \
			-- --submodel "${SUBMODEL}" --timing-seconds "${TIMING}" "${EXTRA[@]}"
		;;

	*) die "unsupported platform $(uname -s)" ;;
esac

[ -s "${BMF}" ] || die "no Bencher Metric Format was produced; nothing to upload"

log "report ${REPORT}"
log "bmf    ${BMF}"

if [ "${DRY_RUN}" -eq 1 ]; then
	log "--dry-run: not uploading"
	cat "${BMF}"
	exit 0
fi

# --- Upload -----------------------------------------------------------------

command -v bencher >/dev/null || die "the bencher CLI is not on PATH.

  Install it with:
      curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh
  and make sure ~/.cargo/bin is on your PATH."

[ -n "${PROJECT}" ] || die "no Bencher project.

  Pass --project <slug>, or:
      export BENCHER_PROJECT=<slug>"

# Resolve the token last, and only into this process's environment. Never a
# command-line argument: those are visible to every other process on the machine
# for as long as the command runs.
if [ -z "${BENCHER_API_TOKEN:-}" ] && [ -n "${BENCHER_OP_REF:-}" ]; then
	command -v op >/dev/null || die "BENCHER_OP_REF is set but the 1Password CLI is not installed"
	log "reading the API token from 1Password"
	BENCHER_API_TOKEN="$(op read "${BENCHER_OP_REF}")"
	export BENCHER_API_TOKEN
fi

[ -n "${BENCHER_API_TOKEN:-}" ] || die "no API token.

  Either:
      export BENCHER_API_TOKEN=\"\$(op read 'op://Developer/<item>/credential')\"
  or set BENCHER_OP_REF to that op:// reference and let this script read it.
  Do not pass it as an argument — process arguments are world-readable."

THRESHOLD_ARGS=()
if [ "${SET_THRESHOLDS}" -eq 1 ]; then
	# A t-test on latency, and no alert until there are ten runs to compare
	# against — below that the test is arithmetic on nothing.
	THRESHOLD_ARGS=(
		--threshold-measure latency
		--threshold-test t_test
		--threshold-min-sample-size 10
		--threshold-max-sample-size 64
		--threshold-upper-boundary 0.98
		--thresholds-reset
	)
	log "installing/refreshing thresholds on ${BRANCH}/${TESTBED}"
fi

ALERT_ARGS=()
[ "${FAIL_ON_ALERT}" -eq 1 ] && ALERT_ARGS=(--error-on-alert)

log "uploading to ${PROJECT} as ${BRANCH}/${TESTBED}"
bencher run \
	--project "${PROJECT}" \
	--branch "${BRANCH}" \
	${HASH:+--hash "${HASH}"} \
	--testbed "${TESTBED}" \
	--adapter json \
	--file "${BMF}" \
	"${THRESHOLD_ARGS[@]}" \
	"${ALERT_ARGS[@]}"

log "done"
