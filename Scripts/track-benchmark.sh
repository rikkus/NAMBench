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
# The API key is never passed on the command line, never written to a file and
# never echoed. It comes from the environment, or straight out of 1Password into
# this process:
#
#   export BENCHER_API_KEY="$(op read 'op://Developer/f2x4p5ymikp25e4hlocah2zexe/credential')"
#
# or set BENCHER_OP_REF to that op:// reference and let this script do it.
# BENCHER_API_TOKEN is accepted as an alias, since that is what the variable used
# to be called, and is translated to BENCHER_API_KEY before the CLI sees it.
#
# Usage:
#   Scripts/track-benchmark.sh [options] [-- <extra driver arguments>]
#
#   --project SLUG      Bencher project (default: $BENCHER_PROJECT)
#   --testbed NAME      this machine's testbed (default: detected, see below)
#   --branch NAME       branch to record against (default: the current one)
#   --hash SHA          commit to attribute the result to (default: HEAD).
#                       Both are needed on a machine with no git checkout — an
#                       rsync'd copy on a Pi, say — or its results land in a
#                       different Bencher branch from the laptop's.
#   --submodels LIST    which to measure (default: widest,narrowest — A2
#                       standard and A2 nano, uploaded as separate series)
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
HASH=""
SUBMODELS="widest,narrowest"
TIMING="30"
CPU_SET=""
DRY_RUN=0
FAIL_ON_ALERT=0
SET_THRESHOLDS=0

# Arrays below are expanded as ${arr[@]+"${arr[@]}"} rather than "${arr[@]}".
# That is not a typo and not superstition: macOS ships bash 3.2, where under
# `set -u` expanding an *empty* array is an unbound-variable error. bash 4.4
# fixed it, which is why this script ran on the Pi and died on the Mac.
EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
	case "$1" in
		--project) PROJECT="$2"; shift 2 ;;
		--testbed) TESTBED="$2"; shift 2 ;;
		--branch) BRANCH="$2"; shift 2 ;;
		--hash) HASH="$2"; shift 2 ;;
		--submodels) SUBMODELS="$2"; shift 2 ;;
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

# The Pi runs from an rsync'd copy with no .git in it, so neither of these is
# discoverable there. Silently falling back to "main" with no hash was worse than
# it looked: the laptop would report its real branch while the Pi reported main,
# and the two machines' results would land in different Bencher branches and stop
# being comparable — which is the one thing this whole arrangement exists to
# prevent. Say so instead, and take both explicitly.
IN_GIT_REPO=0
git rev-parse --git-dir >/dev/null 2>&1 && IN_GIT_REPO=1

if [ -z "${BRANCH}" ]; then
	if [ "${IN_GIT_REPO}" -eq 1 ]; then
		BRANCH="$(git rev-parse --abbrev-ref HEAD)"
	else
		BRANCH="main"
	fi
fi
if [ -z "${HASH}" ] && [ "${IN_GIT_REPO}" -eq 1 ]; then
	HASH="$(git rev-parse HEAD)"
fi

if [ "${IN_GIT_REPO}" -eq 0 ] && [ -z "${HASH}" ]; then
	warn "this is not a git checkout, so the branch and commit cannot be read here.

  Recording against branch '${BRANCH}'${HASH:+ at ${HASH:0:12}}. If the machine you
  ran the laptop from is on a different branch, the two sets of results land in
  different Bencher branches and stop being comparable. Pass them explicitly:
      --branch \"\$(git -C <checkout> rev-parse --abbrev-ref HEAD)\" \\
      --hash   \"\$(git -C <checkout> rev-parse HEAD)\""
fi

# A measurement is attributed to a commit. If the tree does not match that
# commit, the attribution is a lie, and it is a lie that survives in the history
# long after the working copy is gone.
if [ "${IN_GIT_REPO}" -eq 1 ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
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
		# -destination pins the ambiguity xcodebuild otherwise warns about twice:
		# the scheme matches both "My Mac" and "Any Mac", and it picks the first.
		# generic/platform=macOS says so explicitly without hardcoding an arch.
		XCODE_ARGS=(
			-project NAMBench.xcodeproj
			-scheme nambench-cli
			-configuration Release
			-destination "generic/platform=macOS"
		)
		xcodebuild "${XCODE_ARGS[@]}" build >/dev/null || die "xcodebuild failed"
		PRODUCTS="$(xcodebuild "${XCODE_ARGS[@]}" -showBuildSettings 2>/dev/null \
			| awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')"
		[ -x "${PRODUCTS}/nambench" ] || die "no nambench binary at ${PRODUCTS}"

		# The CLI names its own output file, so point it at a scratch directory
		# and pick up what it wrote.
		OUTDIR="$(mktemp -d)"
		trap 'rm -rf "${OUTDIR}"' EXIT

		REPORTS=()
		OLD_IFS="${IFS}"
		IFS=','
		for SUBMODEL in ${SUBMODELS}; do
			IFS="${OLD_IFS}"
			[ -n "${SUBMODEL}" ] || continue
			log "measuring ${SUBMODEL}"
			"${PRODUCTS}/nambench" \
				--submodel "${SUBMODEL}" \
				--timing-seconds "${TIMING}" \
				--output "${OUTDIR}/${SUBMODEL}" \
				${EXTRA[@]+"${EXTRA[@]}"}
			one="${REPORT%.json}-${SUBMODEL}.json"
			cp "$(ls -t "${OUTDIR}/${SUBMODEL}"/*.json | head -1)" "${one}"
			REPORTS=("${REPORTS[@]+${REPORTS[@]}}" "${one}")
			IFS=','
		done
		IFS="${OLD_IFS}"

		python3 Scripts/bencher-report.py \
			${REPORTS[@]+"${REPORTS[@]}"} --output "${BMF}"
		;;

	Linux)
		log "driver: nam_benchmark (portable)"
		cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
			-DNAMBENCH_BUILD_BENCHMARK=ON >/dev/null
		cmake --build build-benchmark --target nam_benchmark --parallel >/dev/null
		./Scripts/run-benchmark.sh \
			--build-dir build-benchmark \
			--submodels "${SUBMODELS}" \
			--output-dir "$(dirname "${REPORT}")" \
			--bmf "${BMF}" \
			${CPU_SET:+--cpu-set "${CPU_SET}"} \
			-- --timing-seconds "${TIMING}" ${EXTRA[@]+"${EXTRA[@]}"}
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

# Resolve the credential last, and only into this process's environment. Never a
# command-line argument: those are visible to every other process on the machine
# for as long as the command runs.
#
# BENCHER_API_KEY, not BENCHER_API_TOKEN. The CLI now reserves --token for JWTs
# and takes an API key (bencher_user_* or bencher_run_*) via --key; handed a key
# through the old variable it refuses outright:
#
#   error: invalid value (redacted) for '--token <TOKEN>': You supplied a
#   Bencher API key to `--token`/`BENCHER_API_TOKEN`. Use `--key`/`BENCHER_API_KEY`
#
# BENCHER_API_TOKEN is still accepted *here* as an input, because plenty of
# shells and CI configs still export it under that name — but it is translated
# and then removed from the environment below, because the CLI reads it directly
# and would reject the run no matter which flag this script passes.
if [ -z "${BENCHER_API_KEY:-}" ] && [ -n "${BENCHER_API_TOKEN:-}" ]; then
	BENCHER_API_KEY="${BENCHER_API_TOKEN}"
fi

if [ -z "${BENCHER_API_KEY:-}" ] && [ -n "${BENCHER_OP_REF:-}" ]; then
	command -v op >/dev/null || die "BENCHER_OP_REF is set but the 1Password CLI is not installed"
	log "reading the API key from 1Password"
	BENCHER_API_KEY="$(op read "${BENCHER_OP_REF}")"
fi

[ -n "${BENCHER_API_KEY:-}" ] || die "no API key.

  Either:
      export BENCHER_API_KEY=\"\$(op read 'op://Developer/<item>/credential')\"
  or set BENCHER_OP_REF to that op:// reference and let this script read it.
  Do not pass it as an argument — process arguments are world-readable."

export BENCHER_API_KEY
# Whatever the CLI would otherwise find and object to.
unset BENCHER_API_TOKEN

# Checked by prefix only; the value is never printed. A JWT in this slot is a
# real possibility for anyone who set the credential up before the CLI split the
# two, and the resulting error names the wrong fix.
case "${BENCHER_API_KEY}" in
	bencher_user_*|bencher_run_*) ;;
	*) warn "this does not look like a Bencher API key (they begin bencher_user_ or
  bencher_run_). If it is an older JWT, the CLI wants it in --token instead, and
  the better fix is to mint an API key in the Bencher console." ;;
esac

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
	${THRESHOLD_ARGS[@]+"${THRESHOLD_ARGS[@]}"} \
	${ALERT_ARGS[@]+"${ALERT_ARGS[@]}"}

log "done"
