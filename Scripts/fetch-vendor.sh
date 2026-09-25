#!/usr/bin/env bash
#
# Fetch the code variants under test into vendor/, at pinned commits.
#
# Every NAM checkout is compiled against ONE shared Eigen tree (vendor/eigen).
# a2_fast uses Eigen for its GEMM, and Linear for its dot products and FFTs, so
# an Eigen version difference between two builds would land directly in the
# measured result. Sharing one tree makes that impossible by construction; we
# still assert every repo pins the same Eigen commit, so a future bump that
# diverges is caught here rather than silently skewing a benchmark.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${REPO_ROOT}/vendor"

# --- Pinned commits ----------------------------------------------------------

UPSTREAM_URL="https://github.com/sdatkinson/NeuralAmpModelerCore.git"
UPSTREAM_SHA="0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842"

# The planar NEON kernels, as proposed to Core in
# https://github.com/sdatkinson/NeuralAmpModelerCore/pull/313 — the head of that
# draft PR, not a summary of it. This is the code the benchmark now measures, so
# the number and the proposal cannot drift apart.
#
# Fork branch apple-silicon-a2-planar (the head of PR #313, which now also
# carries the ARMv7 work). That branch is cut from upstream main and carries no
# other engine variants: just a2_fast, plus a2_planar, plus a two-line change in
# A2FastConfig::create that prefers the planar model where one exists.
#
# The gate is __aarch64__, or 32-bit ARM with NEON and FMA -- not
# __APPLE__ && __aarch64__. Where it cannot open the translation unit has no
# symbols in it and the A2 path is byte for byte what upstream ships. The ARMv7
# arm of the gate needs -mfpu=neon-vfpv4: with plain -mfpu=neon,
# __ARM_FEATURE_FMA is undefined, Eigen picks non-fused vmlaq_f32, and the
# reference the kernels are compared against computes different bits.
PLANAR_URL="https://github.com/rikkus/OptimisationWorkOnNeuralAmpModelerCore.git"
PLANAR_SHA="44412fad6ad135d51218785e15dc7418c29a2124"

# linearplus: Core's Linear model with its FFT path replaced by uniform
# partitions whose multiplies are spread across the callbacks between
# transforms. The impulse-response benchmark measures it against upstream's
# Linear, so it is cut from UPSTREAM_SHA exactly and its diff against that
# commit is the whole of what is being measured.
#
# Fork branch linearplus. It was an AudioDSPTools branch (partitioned-ir) until
# Core gained a zero-latency FFT convolver of its own; see IR-PATH.md.
LINEARPLUS_URL="https://github.com/rikkus/OptimisationWorkOnNeuralAmpModelerCore.git"
LINEARPLUS_SHA="79c8009adf1fca8a2dea6cf926dbe7c03cec878f"

# AudioDSPTools, which is what the plugin convolves its IR slot with today: a
# direct FIR. Measured alongside Core's Linear only when asked for, as the
# reference for what the plugin ships. v0.2.0, which is also its main.
#
# It gets its own Eigen tree rather than sharing the NAM one, because
# AudioDSPTools pins a different Eigen than NeuralAmpModelerCore does and
# building either project against the other's pin would measure something
# neither project ships.
ADT_UPSTREAM_URL="https://github.com/sdatkinson/AudioDSPTools.git"
ADT_UPSTREAM_SHA="844680d118f0317565132c3c5e3aca5f5c976e7a"

EIGEN_URL="https://gitlab.com/libeigen/eigen.git"

# Optional local clones to fetch from instead of the network. Much faster, and
# lets this work offline. Each is only used if it already contains the pinned
# commit; otherwise we fall back to the canonical URL.
LOCAL_HINTS=(
	"/Users/rik/code.local/nam/NeuralAmpModelerCore-linearplus"
	"/Users/rik/code.local/nam/NeuralAmpModelerCore-planar"
	"/Users/rik/code.local/nam/NeuralAmpModelerPlugin/NeuralAmpModelerCore"
	"/Users/rik/code.local/nam/OptimisationWorkOnNeuralAmpModelerPlugin/NeuralAmpModelerCore"
	"/Users/rik/code.local/nam/older/OptimisationWorkOnNeuralAmpModelerCore"
	"/Users/rik/code.local/nam/NeuralAmpModelerPlugin/iPlug2/Dependencies/AudioDSPTools"
	"/Users/rik/code.local/nam/OptimisationWorkOnNeuralAmpModelerPlugin/iPlug2/Dependencies/AudioDSPTools"
)

# --- Helpers -----------------------------------------------------------------

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Echo the path of a local clone that already has $1, or nothing.
find_local_with_commit() {
	local sha="$1" hint
	for hint in "${LOCAL_HINTS[@]}"; do
		[ -d "${hint}/.git" ] || continue
		if git -C "${hint}" cat-file -e "${sha}^{commit}" 2>/dev/null; then
			printf '%s' "${hint}"
			return 0
		fi
	done
	return 0
}

# fetch_at <dest> <canonical-url> <sha> [prefer-local-path]
#
# Idempotent: if dest is already checked out at sha, does nothing. Fetches the
# single commit rather than full history where the server permits it.
fetch_at() {
	local dest="$1" url="$2" sha="$3" local_src="${4:-}"

	if [ -d "${dest}/.git" ] && [ "$(git -C "${dest}" rev-parse HEAD 2>/dev/null || true)" = "${sha}" ]; then
		log "$(basename "${dest}") already at ${sha:0:12}"
		return 0
	fi

	mkdir -p "${dest}"
	if [ ! -d "${dest}/.git" ]; then
		git -C "${dest}" init --quiet
	fi
	git -C "${dest}" remote remove origin 2>/dev/null || true
	git -C "${dest}" remote add origin "${url}"

	local fetched=0
	if [ -n "${local_src}" ]; then
		log "$(basename "${dest}"): fetching ${sha:0:12} from local clone ${local_src}"
		if git -C "${dest}" fetch --quiet --depth 1 "${local_src}" "${sha}" 2>/dev/null; then
			fetched=1
		else
			warn "local fetch failed, falling back to ${url}"
		fi
	fi

	if [ "${fetched}" -eq 0 ]; then
		log "$(basename "${dest}"): fetching ${sha:0:12} from ${url}"
		if ! git -C "${dest}" fetch --quiet --depth 1 origin "${sha}" 2>/dev/null; then
			# Server refuses fetch-by-SHA; fall back to full history.
			warn "shallow fetch by SHA refused; fetching full history (slow)"
			git -C "${dest}" fetch --quiet origin
		fi
	fi

	git -C "${dest}" checkout --quiet --detach "${sha}"
	log "$(basename "${dest}") at $(git -C "${dest}" rev-parse --short HEAD)"
}

# Read the Eigen commit a checkout pins, from its git index.
pinned_eigen_sha() {
	git -C "$1" ls-tree HEAD Dependencies/eigen | awk '{print $3}'
}

# --- Fetch the NAM variants --------------------------------------------------

mkdir -p "${VENDOR}"

fetch_at "${VENDOR}/upstream" "${UPSTREAM_URL}" "${UPSTREAM_SHA}" "$(find_local_with_commit "${UPSTREAM_SHA}")"
fetch_at "${VENDOR}/planar" "${PLANAR_URL}" "${PLANAR_SHA}" "$(find_local_with_commit "${PLANAR_SHA}")"
fetch_at "${VENDOR}/linearplus" "${LINEARPLUS_URL}" "${LINEARPLUS_SHA}" "$(find_local_with_commit "${LINEARPLUS_SHA}")"
fetch_at "${VENDOR}/adt-upstream" "${ADT_UPSTREAM_URL}" "${ADT_UPSTREAM_SHA}" "$(find_local_with_commit "${ADT_UPSTREAM_SHA}")"

# --- Eigen: one shared tree, asserted identical across every checkout --------

UPSTREAM_EIGEN="$(pinned_eigen_sha "${VENDOR}/upstream")"
PLANAR_EIGEN="$(pinned_eigen_sha "${VENDOR}/planar")"
LINEARPLUS_EIGEN="$(pinned_eigen_sha "${VENDOR}/linearplus")"

[ -n "${UPSTREAM_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/upstream"
[ -n "${PLANAR_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/planar"
[ -n "${LINEARPLUS_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/linearplus"

if [ "${UPSTREAM_EIGEN}" != "${PLANAR_EIGEN}" ]; then
	die "the checkouts pin different Eigen commits:
    upstream: ${UPSTREAM_EIGEN}
    planar:   ${PLANAR_EIGEN}
  a2_fast uses Eigen for its GEMM, so building against different Eigen versions
  would put a dependency difference into the measured result. It would also
  break the planar kernels' central claim: they are bit-identical to a2_fast
  because they reproduce the order of arithmetic Eigen performs, and a different
  Eigen may not perform the same one. Reconcile the pins before benchmarking."
fi

if [ "${UPSTREAM_EIGEN}" != "${LINEARPLUS_EIGEN}" ]; then
	die "the checkouts pin different Eigen commits:
    upstream:   ${UPSTREAM_EIGEN}
    linearplus: ${LINEARPLUS_EIGEN}
  Both Linear implementations are Eigen code — dot products for the direct
  head and Eigen::FFT for the partitions — so different Eigen versions would put
  a dependency difference into the measured result. Reconcile the pins before
  benchmarking."
fi

log "all three checkouts pin Eigen ${UPSTREAM_EIGEN:0:12} — building against one shared tree"

fetch_at "${VENDOR}/eigen" "${EIGEN_URL}" "${UPSTREAM_EIGEN}" "$(
	for hint in "${LOCAL_HINTS[@]}"; do
		if [ -d "${hint}/Dependencies/eigen/.git" ] || [ -f "${hint}/Dependencies/eigen/.git" ]; then
			if git -C "${hint}/Dependencies/eigen" cat-file -e "${UPSTREAM_EIGEN}^{commit}" 2>/dev/null; then
				printf '%s' "${hint}/Dependencies/eigen"
				break
			fi
		fi
	done
)"

# --- Eigen for AudioDSPTools: a second shared tree ---------------------------
#
# AudioDSPTools pins its own Eigen, and it is not the one NeuralAmpModelerCore
# pins. Forcing either project onto the other's would measure a configuration
# neither of them ships, so there are two trees.

ADT_UPSTREAM_EIGEN="$(pinned_eigen_sha "${VENDOR}/adt-upstream")"

[ -n "${ADT_UPSTREAM_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/adt-upstream"

if [ "${ADT_UPSTREAM_EIGEN}" = "${UPSTREAM_EIGEN}" ]; then
	# Nothing wrong with this; it just means the two projects have converged and
	# the second tree is redundant. Say so rather than fetching it twice.
	log "AudioDSPTools pins the same Eigen as the NAM checkouts — sharing vendor/eigen"
	ADT_EIGEN_DIR="${VENDOR}/eigen"
else
	ADT_EIGEN_DIR="${VENDOR}/eigen-adt"
	log "AudioDSPTools pins Eigen ${ADT_UPSTREAM_EIGEN:0:12} — a separate shared tree"
	fetch_at "${ADT_EIGEN_DIR}" "${EIGEN_URL}" "${ADT_UPSTREAM_EIGEN}" "$(
		for hint in "${LOCAL_HINTS[@]}"; do
			if [ -d "${hint}/Dependencies/eigen/.git" ] || [ -f "${hint}/Dependencies/eigen/.git" ]; then
				if git -C "${hint}/Dependencies/eigen" cat-file -e "${ADT_UPSTREAM_EIGEN}^{commit}" 2>/dev/null; then
					printf '%s' "${hint}/Dependencies/eigen"
					break
				fi
			fi
		done
	)"
fi

# --- Sanity checks -----------------------------------------------------------

[ -f "${VENDOR}/upstream/NAM/wavenet/a2_fast.cpp" ] || die "vendor/upstream missing NAM/wavenet/a2_fast.cpp"
[ -f "${VENDOR}/planar/NAM/wavenet/a2_planar.cpp" ] || die "vendor/planar missing NAM/wavenet/a2_planar.cpp"
[ -f "${VENDOR}/planar/NAM/wavenet/a2_planar.h" ] || die "vendor/planar missing NAM/wavenet/a2_planar.h"
[ -f "${VENDOR}/eigen/Eigen/Dense" ] || die "vendor/eigen missing Eigen/Dense"
[ -f "${VENDOR}/upstream/Dependencies/nlohmann/json.hpp" ] || die "vendor/upstream missing nlohmann/json.hpp"
[ -f "${VENDOR}/planar/Dependencies/nlohmann/json.hpp" ] || die "vendor/planar missing nlohmann/json.hpp"
[ -f "${VENDOR}/linearplus/NAM/linear.cpp" ] || die "vendor/linearplus missing NAM/linear.cpp"
[ -f "${VENDOR}/linearplus/Dependencies/nlohmann/json.hpp" ] || die "vendor/linearplus missing nlohmann/json.hpp"
[ -f "${VENDOR}/adt-upstream/dsp/ImpulseResponse.cpp" ] || die "vendor/adt-upstream missing dsp/ImpulseResponse.cpp"
[ -f "${ADT_EIGEN_DIR}/Eigen/Dense" ] || die "${ADT_EIGEN_DIR} missing Eigen/Dense"

# linearplus is cut from UPSTREAM_SHA, so its diff is the change being measured
# and nothing else. Print it: an unexpectedly large one, or one outside
# NAM/linear.*, means the comparison is no longer isolating the convolution.
# Both are single-commit checkouts, but a diff needs only the two trees, so the
# upstream commit is borrowed from vendor/upstream rather than from the network.
git -C "${VENDOR}/linearplus" fetch --quiet --depth 1 "${VENDOR}/upstream" "${UPSTREAM_SHA}" 2>/dev/null || true
if git -C "${VENDOR}/linearplus" cat-file -e "${UPSTREAM_SHA}^{commit}" 2>/dev/null; then
	log "linearplus changes under NAM/ vs upstream:$(git -C "${VENDOR}/linearplus" diff --shortstat "${UPSTREAM_SHA}" -- NAM/)"
	log "  $(git -C "${VENDOR}/linearplus" diff --name-only "${UPSTREAM_SHA}" -- NAM/ | tr '\n' ' ')"
else
	log "linearplus is a shallow checkout; skipping the diff against upstream ${UPSTREAM_SHA:0:12}"
fi

# The planar branch deliberately DOES touch a2_fast — that is where the two-line
# dispatcher change lives — so byte-identity is the wrong test for it. What must
# hold instead is that the reference implementation it falls back to is still
# upstream's: A2FastModel is the thing planar claims to be bit-identical to, and
# if the branch had also changed the reference, the claim would be circular.
#
# Checked by comparing everything in a2_fast.cpp except the dispatcher body,
# which is the one hunk the PR touches there.
#
# Only possible when the checkout actually has the base commit. These are
# shallow fetches of a single commit, so usually it does not, and saying so is
# better than printing nothing and looking like a clean result.
if git -C "${VENDOR}/planar" cat-file -e "${UPSTREAM_SHA}^{commit}" 2>/dev/null; then
	log "planar branch changes under NAM/ vs upstream:$(git -C "${VENDOR}/planar" diff --shortstat "${UPSTREAM_SHA}" -- NAM/)"
else
	log "planar is a shallow checkout; skipping the diff against upstream ${UPSTREAM_SHA:0:12}"
fi
log "planar at $(git -C "${VENDOR}/planar" rev-parse --short HEAD) (PR #313)"

# --- Record provenance for the report ---------------------------------------

cat > "${VENDOR}/pins.json" <<EOF
{
  "upstream": {
    "url": "${UPSTREAM_URL}",
    "sha": "$(git -C "${VENDOR}/upstream" rev-parse HEAD)"
  },
  "planar": {
    "url": "${PLANAR_URL}",
    "sha": "$(git -C "${VENDOR}/planar" rev-parse HEAD)",
    "pull_request": "https://github.com/sdatkinson/NeuralAmpModelerCore/pull/313"
  },
  "eigen": {
    "url": "${EIGEN_URL}",
    "sha": "$(git -C "${VENDOR}/eigen" rev-parse HEAD)",
    "shared": true
  },
  "linearplus": {
    "url": "${LINEARPLUS_URL}",
    "sha": "$(git -C "${VENDOR}/linearplus" rev-parse HEAD)",
    "branch": "linearplus"
  },
  "adt_upstream": {
    "url": "${ADT_UPSTREAM_URL}",
    "sha": "$(git -C "${VENDOR}/adt-upstream" rev-parse HEAD)"
  },
  "eigen_adt": {
    "url": "${EIGEN_URL}",
    "sha": "$(git -C "${ADT_EIGEN_DIR}" rev-parse HEAD)",
    "shared": true
  }
}
EOF

log "wrote ${VENDOR}/pins.json"
log "done"
