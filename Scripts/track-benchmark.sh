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
# Everything Bencher needs is settled *before* the benchmark runs, not after it.
# A measurement takes minutes of a machine held quiet on purpose, and finding out
# at the end that the key was stale means throwing all of that away. So the
# credential is resolved, and a real read is made against the API, while it still
# costs nothing to fix. `--check` does exactly that and stops.
#
# The API key is never passed on the command line and never echoed. It comes
# from the environment, from a .env beside the checkout, or straight out of
# 1Password into this process:
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
#   --check             check this machine can reach Bencher, then stop.
#                       Measures nothing and uploads nothing.
#   --dry-run           measure and convert, but do not upload
#   --fail-on-alert     exit non-zero if Bencher raises an alert
#   --no-sync           skip the threshold and plot sync after uploading

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Where the caller was standing, before the cd below. A .env is looked for here
# as well as in the checkout, so `cd ~/NAMBench && Scripts/track-benchmark.sh`
# and an absolute path from somewhere else both find the same file.
INVOKED_FROM="${PWD}"
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
SYNC=1
CHECK=0

# Arrays below are expanded as ${arr[@]+"${arr[@]}"} rather than "${arr[@]}".
# That is not a typo and not superstition: macOS ships bash 3.2, where under
# `set -u` expanding an *empty* array is an unbound-variable error. bash 4.4
# fixed it, which is why this script ran on the Pi and died on the Mac.
EXTRA=()

# How many compiler processes to run at once.
#
# `cmake --build --parallel` with no number passes a bare -j to make, which
# means *unlimited*: on an 8-core M2 Air that is 25-odd clang processes, 0% idle
# and kernel_task pinned at 45% while macOS forces the machine to cool. The
# build gets slower, not faster, and everything else on the laptop stops.
#
# Performance cores only on Apple silicon. The efficiency cores add little to a
# memory-bound C++ compile and a great deal to the heat, and this is a machine
# that has to be quiet enough to benchmark on afterwards.
#
# NAMBENCH_BUILD_JOBS overrides.
build_jobs() {
	if [ -n "${NAMBENCH_BUILD_JOBS:-}" ]; then
		printf '%s' "${NAMBENCH_BUILD_JOBS}"
		return
	fi
	local jobs=""
	if [ "$(uname -s)" = "Darwin" ]; then
		jobs="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || true)"
		[ -n "${jobs}" ] || jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
	else
		jobs="$(nproc 2>/dev/null || true)"
	fi
	printf '%s' "${jobs:-4}"
}

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- .env -------------------------------------------------------------------
#
# A machine that uploads keeps its key in a .env beside the checkout, because
# the alternative is remembering to export it into every shell — and the Pi is
# driven over ssh, where a forgotten export used to be discovered at the end of
# a ten-minute measurement rather than at the start of one.
#
# Parsed, not sourced, and only BENCHER_* is taken. Sourcing runs whatever is in
# the file as shell, and this is a script whose entire output is a timing: a
# stray PATH or LD_PRELOAD picked up from a file nobody reads any more would not
# fail, it would quietly produce a number.
#
# Anything already exported wins, so `BENCHER_PROJECT=other ./track-benchmark.sh`
# still overrides the file. Values are never printed.
DOTENV_LOADED=""

load_dotenv() {
	local file="$1" line name value
	[ -f "${file}" ] || return 0

	while IFS= read -r line || [ -n "${line}" ]; do
		line="${line%$'\r'}"
		line="${line#"${line%%[![:space:]]*}"}"
		case "${line}" in
			''|'#'*) continue ;;
			"export "*) line="${line#export }" ;;
		esac

		name="${line%%=*}"
		# No '=' at all: ${line%%=*} is the whole line, and there is nothing to set.
		[ "${name}" != "${line}" ] || continue
		case "${name}" in BENCHER_*) ;; *) continue ;; esac

		value="${line#*=}"
		value="${value%"${value##*[![:space:]]}"}"
		case "${value}" in
			'"'*'"') value="${value#\"}"; value="${value%\"}" ;;
			"'"*"'") value="${value#\'}"; value="${value%\'}" ;;
		esac

		[ -z "${!name:-}" ] || continue
		export "${name}=${value}"
		DOTENV_LOADED="${file}"
	done < "${file}"
}

while [ $# -gt 0 ]; do
	case "$1" in
		--project) PROJECT="$2"; shift 2 ;;
		--testbed) TESTBED="$2"; shift 2 ;;
		--branch) BRANCH="$2"; shift 2 ;;
		--hash) HASH="$2"; shift 2 ;;
		--submodels) SUBMODELS="$2"; shift 2 ;;
		--timing-seconds) TIMING="$2"; shift 2 ;;
		--cpu-set) CPU_SET="$2"; shift 2 ;;
		--check) CHECK=1; shift ;;
		--dry-run) DRY_RUN=1; shift ;;
		--fail-on-alert) FAIL_ON_ALERT=1; shift ;;
		--no-sync) SYNC=0; shift ;;
		--) shift; EXTRA=("$@"); break ;;
		# Everything from line 2 up to the first line that is not a comment, so
		# the help cannot drift out of date with the header the way a hardcoded
		# line range does every time the header grows.
		-h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
			"${BASH_SOURCE[0]}"; exit 0 ;;
		*) die "unknown option $1" ;;
	esac
done

load_dotenv "${INVOKED_FROM}/.env"
[ "${INVOKED_FROM}" = "${REPO_ROOT}" ] || load_dotenv "${REPO_ROOT}/.env"
[ -n "${PROJECT}" ] || PROJECT="${BENCHER_PROJECT:-}"

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
# Not under --check, which measures nothing and so attributes nothing.
if [ "${CHECK}" -eq 0 ] && [ "${IN_GIT_REPO}" -eq 1 ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
	warn "the working tree has uncommitted changes.

  This result will be recorded against ${HASH:0:12}, which is not the code that
  produced it. Fine while you are setting Bencher up; commit before you record
  anything you intend to compare against later."
fi

# --- Bencher preflight ------------------------------------------------------
#
# Before the measurement, not after it. A run is minutes of a machine held
# deliberately quiet, and a key that expired, a project renamed, or an ssh
# session that never had the variable exported are all things that used to
# surface at the upload — with the measurement already made and the machine
# already warm. Every one of them is visible from here for the cost of one GET.
#
# --dry-run skips it, because a dry run is the one mode that is meant to work
# with no credential at all.

# BENCHER_API_KEY, not BENCHER_API_TOKEN. The CLI now reserves --token for JWTs
# and takes an API key (bencher_user_* or bencher_run_*) via --key; handed a key
# through the old variable it refuses outright:
#
#   error: invalid value (redacted) for '--token <TOKEN>': You supplied a
#   Bencher API key to `--token`/`BENCHER_API_TOKEN`. Use `--key`/`BENCHER_API_KEY`
#
# BENCHER_API_TOKEN is still accepted *here* as an input, because plenty of
# shells, CI configs and .env files still spell it that way — but it is
# translated and then removed from the environment, because the CLI reads it
# directly and would reject the run no matter which flag this script passes.
resolve_credential() {
	if [ -z "${BENCHER_API_KEY:-}" ] && [ -n "${BENCHER_API_TOKEN:-}" ]; then
		BENCHER_API_KEY="${BENCHER_API_TOKEN}"
	fi

	if [ -z "${BENCHER_API_KEY:-}" ] && [ -n "${BENCHER_OP_REF:-}" ]; then
		command -v op >/dev/null || die "BENCHER_OP_REF is set but the 1Password CLI is not installed"
		log "reading the API key from 1Password"
		BENCHER_API_KEY="$(op read "${BENCHER_OP_REF}")"
	fi

	[ -n "${BENCHER_API_KEY:-}" ] || die "no API key.

  Any of:
      put BENCHER_API_KEY=... in a .env beside the checkout
      export BENCHER_API_KEY=\"\$(op read 'op://Developer/<item>/credential')\"
      set BENCHER_OP_REF to that op:// reference and let this script read it
  Do not pass it as an argument — process arguments are world-readable."

	# Resolved only into this process's environment, never a command-line
	# argument: those are visible to every other process on the machine for as
	# long as the command runs.
	export BENCHER_API_KEY
	# Whatever the CLI would otherwise find and object to.
	unset BENCHER_API_TOKEN

	# Checked by prefix only; the value is never printed. A JWT in this slot is a
	# real possibility for anyone who set the credential up before the CLI split
	# the two, and the resulting error names the wrong fix.
	case "${BENCHER_API_KEY}" in
		bencher_user_*|bencher_run_*) ;;
		*) warn "this does not look like a Bencher API key (they begin bencher_user_ or
  bencher_run_). If it is an older JWT, the CLI wants it in --token instead, and
  the better fix is to mint an API key in the Bencher console." ;;
	esac
}

preflight() {
	# The Pi is driven over ssh, and a non-interactive shell does not read the
	# profile that puts ~/.cargo/bin on PATH — which is where the Bencher CLI
	# installs itself. *Appended*, never prepended: this is a script whose whole
	# output is a timing, and putting a cargo bin directory ahead of the system
	# one could quietly substitute a different compiler or linker into the build.
	# Appending can only add a tool that was otherwise missing.
	if ! command -v bencher >/dev/null && [ -x "${HOME}/.cargo/bin/bencher" ]; then
		PATH="${PATH}:${HOME}/.cargo/bin"
		export PATH
	fi

	command -v bencher >/dev/null || die "the bencher CLI is not on PATH.

  Install it with:
      curl --proto '=https' --tlsv1.2 -sSfL https://bencher.dev/download/install-cli.sh | sh
  and make sure ~/.cargo/bin is on your PATH.
  Over ssh that is worth checking twice: a non-interactive shell often does not
  read the profile that puts ~/.cargo/bin there."

	command -v python3 >/dev/null || die "python3 is not on PATH; it converts the
  report to Bencher Metric Format and syncs the thresholds and plots."

	[ -n "${PROJECT}" ] || die "no Bencher project.

  Pass --project <slug>, put BENCHER_PROJECT=<slug> in a .env beside the
  checkout, or export it."

	resolve_credential

	# The read. Deliberately the smallest one that proves all three of network,
	# credential and project at once: an unreachable API, a key that is wrong or
	# expired, and a project slug that does not exist each fail here, and each
	# says so in the CLI's own words.
	local output
	if ! output="$(bencher project view "${PROJECT}" 2>&1)"; then
		# Redacted before it is shown. The CLI quotes the key back at you when it
		# fails its own format check — `invalid value 'bencher_user_...' for
		# '--key'` — and this script's one promise about the credential is that it
		# never appears in a terminal or a CI log.
		die "cannot read project '${PROJECT}' from Bencher, so there is no point
  measuring first and finding out afterwards. The CLI said:

$(printf '%s\n' "${output}" \
			| sed -E 's/bencher_(user|run)_[A-Za-z0-9]+/bencher_\1_<redacted>/g' \
			| sed 's/^/    /')"
	fi

	local visibility
	visibility="$(printf '%s' "${output}" \
		| python3 -c 'import json,sys; print(json.load(sys.stdin).get("visibility", "?"))' 2>/dev/null || true)"
	log "bencher: ${PROJECT} readable${visibility:+ (${visibility})}"
}

if [ "${DRY_RUN}" -eq 1 ]; then
	[ "${CHECK}" -eq 0 ] || die "--check and --dry-run ask for opposite things:
  one talks to Bencher and measures nothing, the other measures and talks to
  nothing."
	log "--dry-run: skipping the Bencher preflight"
else
	preflight
fi

if [ "${CHECK}" -eq 1 ]; then
	log "check passed"
	printf '  testbed  %s\n' "${TESTBED}"
	printf '  project  %s\n' "${PROJECT}"
	printf '  branch   %s%s\n' "${BRANCH}" "${HASH:+ @ ${HASH:0:12}}"
	printf '  key      %s\n' "${DOTENV_LOADED:-the environment}"
	printf '  driver   %s\n' \
		"$([ "$(uname -s)" = "Darwin" ] && echo 'nambench (Xcode)' || echo 'nam_benchmark (portable)')"
	log "nothing was measured and nothing was uploaded"
	exit 0
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
		cmake --build build-benchmark --target nam_benchmark --parallel "$(build_jobs)" >/dev/null
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
#
# The CLI, the project, the key and the API were all checked before the
# measurement, and BENCHER_API_KEY is already exported. Nothing to resolve here.

ALERT_ARGS=()
[ "${FAIL_ON_ALERT}" -eq 1 ] && ALERT_ARGS=(--error-on-alert)

# No --threshold-* flags here. The model belongs to Scripts/bencher-sync.py, in
# one place: those flags come with --thresholds-reset, so two upload paths that
# disagree by a single argument would take turns quietly redefining the model
# for a branch and testbed, and the one that alerts is whichever ran last.
log "uploading to ${PROJECT} as ${BRANCH}/${TESTBED}"
bencher run \
	--project "${PROJECT}" \
	--branch "${BRANCH}" \
	${HASH:+--hash "${HASH}"} \
	--testbed "${TESTBED}" \
	--adapter json \
	--file "${BMF}" \
	${ALERT_ARGS[@]+"${ALERT_ARGS[@]}"}

# After the upload, so a machine or a kernel measured here for the first time
# gets its threshold and its chart from this run rather than from whenever
# somebody next remembers. Idempotent, deletes nothing, and cheap enough that
# there is no reason to make it opt-in.
if [ "${SYNC}" -eq 1 ]; then
	log "syncing thresholds and plots"
	SYNC_ARGS=(--project "${PROJECT}" --branch "${BRANCH}")
	# Plots are pinned, project-wide, and capped at 64, so they follow main
	# only; a branch still gets thresholds, and its data can be plotted ad hoc.
	[ "${BRANCH}" = "main" ] || SYNC_ARGS+=(--skip-plots)
	python3 "${REPO_ROOT}/Scripts/bencher-sync.py" "${SYNC_ARGS[@]}"
fi

log "done"
