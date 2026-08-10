#!/usr/bin/env bash
#
# Fetch the code variants under test into vendor/, at pinned commits.
#
# Both NAM checkouts are compiled against ONE shared Eigen tree (vendor/eigen).
# a2_fast uses Eigen for its GEMM while the fused engine is pure NEON, so an
# Eigen version difference between the two builds would land directly in the
# measured result. Sharing one tree makes that impossible by construction; we
# still assert both repos pin the same Eigen commit, so a future bump that
# diverges is caught here rather than silently skewing a benchmark.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${REPO_ROOT}/vendor"

# --- Pinned commits ----------------------------------------------------------

UPSTREAM_URL="https://github.com/sdatkinson/NeuralAmpModelerCore.git"
UPSTREAM_SHA="3cde95c354d5ba6da01316cad90b05cfc4855053"

FUSED_URL="https://github.com/rikkus/OptimisationWorkOnNeuralAmpModelerCore.git"
FUSED_SHA="4596b54ce102d3ceef9fd2b4a158978ea794fe9a"

# The planar NEON kernels, as proposed to Core in
# https://github.com/sdatkinson/NeuralAmpModelerCore/pull/313 — the head of that
# draft PR, not a summary of it. This is the code the benchmark now measures, so
# the number and the proposal cannot drift apart.
#
# Same fork as FUSED, different branch (apple-silicon-a2-planar). That branch is
# cut from upstream main and carries no fused engine at all: a2_fast, plus
# a2_planar, plus a two-line change in A2FastConfig::create that prefers the
# planar model where one exists.
#
# The gate is __aarch64__, not __APPLE__ && __aarch64__: the kernels are selected
# on every AArch64 target that defines it. Off AArch64 the translation unit has
# no symbols in it and the A2 path is byte for byte what upstream ships.
PLANAR_URL="${FUSED_URL}"
PLANAR_SHA="ca349f6cfe07200aea9e6767cd29025d7152b33a"

EIGEN_URL="https://gitlab.com/libeigen/eigen.git"

# Optional local clones to fetch from instead of the network. Much faster, and
# lets this work offline. Each is only used if it already contains the pinned
# commit; otherwise we fall back to the canonical URL.
LOCAL_HINTS=(
	"/Users/rik/code.local/nam/NeuralAmpModelerPlugin/NeuralAmpModelerCore"
	"/Users/rik/code.local/nam/OptimisationWorkOnNeuralAmpModelerPlugin/NeuralAmpModelerCore"
	"/Users/rik/code.local/nam/older/OptimisationWorkOnNeuralAmpModelerCore"
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

# --- Fetch the two NAM variants ---------------------------------------------

mkdir -p "${VENDOR}"

fetch_at "${VENDOR}/upstream" "${UPSTREAM_URL}" "${UPSTREAM_SHA}" "$(find_local_with_commit "${UPSTREAM_SHA}")"
fetch_at "${VENDOR}/fused" "${FUSED_URL}" "${FUSED_SHA}" "$(find_local_with_commit "${FUSED_SHA}")"
fetch_at "${VENDOR}/planar" "${PLANAR_URL}" "${PLANAR_SHA}" "$(find_local_with_commit "${PLANAR_SHA}")"

# --- Eigen: one shared tree, asserted identical across every checkout --------

UPSTREAM_EIGEN="$(pinned_eigen_sha "${VENDOR}/upstream")"
FUSED_EIGEN="$(pinned_eigen_sha "${VENDOR}/fused")"
PLANAR_EIGEN="$(pinned_eigen_sha "${VENDOR}/planar")"

[ -n "${UPSTREAM_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/upstream"
[ -n "${FUSED_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/fused"
[ -n "${PLANAR_EIGEN}" ] || die "could not read pinned Eigen commit from vendor/planar"

if [ "${UPSTREAM_EIGEN}" != "${FUSED_EIGEN}" ] || [ "${UPSTREAM_EIGEN}" != "${PLANAR_EIGEN}" ]; then
	die "the checkouts pin different Eigen commits:
    upstream: ${UPSTREAM_EIGEN}
    fused:    ${FUSED_EIGEN}
    planar:   ${PLANAR_EIGEN}
  a2_fast uses Eigen for its GEMM, so building against different Eigen versions
  would put a dependency difference into the measured result. It would also
  break the planar kernels' central claim: they are bit-identical to a2_fast
  because they reproduce the order of arithmetic Eigen performs, and a different
  Eigen may not perform the same one. Reconcile the pins before benchmarking."
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

# --- Sanity checks -----------------------------------------------------------

[ -f "${VENDOR}/upstream/NAM/wavenet/a2_fast.cpp" ] || die "vendor/upstream missing NAM/wavenet/a2_fast.cpp"
[ -f "${VENDOR}/fused/NAM/wavenet/fused.cpp" ] || die "vendor/fused missing NAM/wavenet/fused.cpp"
[ -f "${VENDOR}/fused/NAM/wavenet/engine_prefer.h" ] || die "vendor/fused missing NAM/wavenet/engine_prefer.h"
[ -f "${VENDOR}/planar/NAM/wavenet/a2_planar.cpp" ] || die "vendor/planar missing NAM/wavenet/a2_planar.cpp"
[ -f "${VENDOR}/planar/NAM/wavenet/a2_planar.h" ] || die "vendor/planar missing NAM/wavenet/a2_planar.h"
[ -f "${VENDOR}/eigen/Eigen/Dense" ] || die "vendor/eigen missing Eigen/Dense"
[ -f "${VENDOR}/upstream/Dependencies/nlohmann/json.hpp" ] || die "vendor/upstream missing nlohmann/json.hpp"
[ -f "${VENDOR}/fused/Dependencies/nlohmann/json.hpp" ] || die "vendor/fused missing nlohmann/json.hpp"
[ -f "${VENDOR}/planar/Dependencies/nlohmann/json.hpp" ] || die "vendor/planar missing nlohmann/json.hpp"

# a2_fast must be byte-identical between upstream and the fused fork, otherwise
# "upstream a2_fast vs fork fused" is not the comparison being made.
for f in NAM/wavenet/a2_fast.cpp NAM/wavenet/a2_fast.h; do
	a="$(shasum -a 256 "${VENDOR}/upstream/${f}" | cut -d' ' -f1)"
	b="$(shasum -a 256 "${VENDOR}/fused/${f}" | cut -d' ' -f1)"
	[ "${a}" = "${b}" ] || die "${f} differs between the two checkouts; the fork has diverged from upstream's a2_fast"
done
log "a2_fast sources verified byte-identical across upstream and fused"

# The planar branch deliberately DOES touch a2_fast — that is where the two-line
# dispatcher change lives — so byte-identity is the wrong test for it. What must
# hold instead is that the reference implementation it falls back to is still
# upstream's: A2FastModel is the thing planar claims to be bit-identical to, and
# if the branch had also changed the reference, the claim would be circular.
#
# Checked by comparing everything in a2_fast.cpp except the dispatcher body,
# which is the one hunk the PR touches there.
if command -v git >/dev/null 2>&1; then
	changed="$(git -C "${VENDOR}/planar" diff --stat "${UPSTREAM_SHA}" -- NAM/ 2>/dev/null | tail -1 || true)"
	if [ -n "${changed}" ]; then
		log "planar branch changes under NAM/ vs upstream:${changed}"
	fi
fi
log "planar at $(git -C "${VENDOR}/planar" rev-parse --short HEAD) (PR #313)"

# --- Record provenance for the report ---------------------------------------

cat > "${VENDOR}/pins.json" <<EOF
{
  "upstream": {
    "url": "${UPSTREAM_URL}",
    "sha": "$(git -C "${VENDOR}/upstream" rev-parse HEAD)"
  },
  "fused": {
    "url": "${FUSED_URL}",
    "sha": "$(git -C "${VENDOR}/fused" rev-parse HEAD)"
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
  }
}
EOF

log "wrote ${VENDOR}/pins.json"
log "done"
