# Sync forks and refresh the Core PR

## Context

This repo (NAMBench) vendors optimisation work for `NeuralAmpModelerCore` (sdatkinson's repo, "Core") out of a separate fork, `rikkus/OptimisationWorkOnNeuralAmpModelerCore`. That fork already holds every branch this project has produced: the Apple Silicon planar kernels, the ARMv7 extension of the same kernels, and two earlier, now-superseded approaches (a fused WaveNet engine and an IR optimisation). The user wants confirmation that all of that is safely on their personal fork, and a current, mergeable PR against Core itself.

Investigation found the fork side is already done — nothing to push there. The PR side is not: PR #313 (open, Apple-Silicon-only) is ~179 commits behind Core's current `main` (through v0.5.4) and shows as `CONFLICTING`, and it doesn't include the ARMv7 commits that already exist on the fork's `armv7-a2-planar` branch (which is `apple-silicon-a2-planar` plus 3 more commits: widen the gate to 32-bit ARM, retune tiles, give ARMv7 its own C=8 conv-loop shape, and a doc summary). So the real work is: bring the combined Apple-Silicon-and-ARMv7 branch up to date with Core's current `main`, verify it still holds its claims, and get PR #313 to reflect that.

Per the user's decisions: one combined PR (not two), done via rebase (not merge) onto Core's current `main`, and no update to PR #313 until the post-rebase numbers are recorded in Bencher.

**Execution note:** this plan was written from a session on `piv` (AArch64, no Mac access, no SSH to `tib` configured here). The user will resume execution from a Claude session on their MacBook, which can reach the Apple Silicon testbed directly and SSH to both `piv` and `tib` to run the AArch64 and ARMv7 re-verification and Bencher uploads.

## Verified current state (no action needed)

- `rikkus/OptimisationWorkOnNeuralAmpModelerCore` is a real GitHub fork of `sdatkinson/NeuralAmpModelerCore`.
- Fork branches, all already pushed: `apple-silicon-a2-planar` (PR #313's head), `armv7-a2-planar` (3 commits ahead of the above), `fused-optimisation`, `ir-optimisation`, `optimise-for-apple-silicon`, `main`.
- This repo's vendored copies (`vendor/planar`, `vendor/fused`) are clean and their pinned SHAs (`vendor/pins.json`) exactly match the fork's branch tips — nothing local is unpushed.
- PR #308 (the fused-engine approach) was deliberately closed by the user ("I have a new version which is better") in favour of the planar approach. `fused-optimisation` / `ir-optimisation` / `optimise-for-apple-silicon` stay parked on the fork as history — no PR action for them.

## What needs doing

### 1. Set up a scratch clone for the rebase

Do this outside `vendor/` — the vendored copies here are shallow (grafted) and are NAMBench's pinned build inputs, not a workspace for rewriting history. Clone `rikkus/OptimisationWorkOnNeuralAmpModelerCore` fresh (full history) into a scratch directory, add `sdatkinson/NeuralAmpModelerCore` as an `upstream` remote, fetch both.

### 2. Rebase the combined branch onto Core's current `main`

- Base: `armv7-a2-planar` (tip `80d75c19`, contains all of `apple-silicon-a2-planar`'s 3 commits plus the 3 ARMv7 commits).
- Target: `upstream/main` (currently `2563c0f`, vs. the PR's stale base `3cde95c`).
- `git rebase upstream/main` and resolve conflicts commit-by-commit. These are DSP-path changes (`a2_fast`/`a2_planar` gating, dispatch), so conflict resolution is real engineering, not mechanical — each commit's intent (documented in its own message) has to be preserved against whatever Core changed underneath it.

### 3. Re-verify the claims, not just the build

The PR's whole argument is bit-identity + measured speedup. A 179-commit rebase can silently invalidate either, so don't treat a green build as sufficient:

- Cross-build for AArch64, ARMv7, and x86_64 from the rebased branch; confirm non-target platforms still compile the kernel files to no symbols (the existing gate-verification approach, e.g. `Scripts/a32-codegen-check.sh`, and the parity/bench tools already in the PR: `tools/test/test_a2_planar.cpp`, `tools/bench_a2_planar.cpp`).
- This machine (`piv`) is itself AArch64 (Cortex-A76) — run the parity test and bench tool natively here to re-check the AArch64 bit-identity and speedup claims after rebase, the same way the PR's own commits were verified on this class of hardware.
- Cross-build and deploy to the ARMv7 board (`tib`) via `Scripts/a32-deploy.sh` the same way `A32-PATH.md`'s "Reproducing" section describes, to re-check the ARMv7 claims. Note [[tib-freeze-2026-09]] and [[tinker-board-measurement-constraints]] — use the 1416 MHz cap and expect the board's other constraints.
- Apple Silicon (M2) numbers in the PR description can't be re-verified on this machine — flag this explicitly to the user as something they'd need to spot-check on a Mac, rather than silently leaving stale numbers in the PR.

### 4. Land the post-rebase results on Bencher before touching the PR

The project already tracks every published number in Bencher via `Scripts/track-benchmark.sh` (which calls `Scripts/bencher-report.py` / `bencher-sync.py`). Before PR #313 is updated, run this on every testbed that re-verifies a claim in step 3 — the Mac (Apple Silicon, `nambench`), `piv` (AArch64, `nam_benchmark`), and `tib` (ARMv7, `nam_benchmark`) — so the rebased commit has real recorded history, not just a local console readout, and any regression Bencher's threshold sync would flag is caught before it goes to Core. Use `--hash` for the rebased commit's SHA so results attribute correctly even from machines without a full git checkout (e.g. an rsync'd copy on a board). This is a hard prerequisite for step 5, not a parallel nice-to-have.

### 5. Push and update PR #313

- Force-push the rebased history to the fork's `apple-silicon-a2-planar` branch (PR #313's existing head — do **not** rename it; GitHub PR head branches can't be repointed to a different branch name after creation, but since `armv7-a2-planar` is a strict superset, force-pushing its rebased content onto `apple-silicon-a2-planar` naturally makes #313 combined without needing a new PR).
- Force-push the same rebased history to `armv7-a2-planar` too, so the fork's two branches stay identical/consistent rather than silently diverging (or delete `armv7-a2-planar` post-rebase, since its content is now fully subsumed — decide at execution time based on whether anything else references it).
- Update PR #313's title and description via `gh pr edit` to describe both Apple Silicon and ARMv7 (it currently reads Apple-Silicon-only), and to fold in whatever re-verification results step 3 produced.

### 6. Explicitly out of scope (flag, don't silently do)

- NAMBench's own `vendor/pins.json` / `vendor/planar` pin still points at the pre-rebase SHA. Bumping it to the rebased commit is a separate follow-up (it would mean rebuilding/re-benchmarking NAMBench itself against a newer Core `main`), not required to get the Core PR current. Mention this to the user as a follow-up rather than doing it inline.

## Verification

- `tools/test/test_a2_planar.cpp` and `ctest` pass on the rebased branch, cross-built for AArch64/ARMv7/x86_64.
- `tools/bench_a2_planar.cpp` re-run natively on this AArch64 machine, confirming bit-for-bit parity and the ~2x-class speedup still holds post-rebase.
- ARMv7 parity/bench re-run on `tib` via `Scripts/a32-deploy.sh`, confirming the A32-PATH.md figures still hold post-rebase.
- `gh pr view 313` shows `mergeable: MERGEABLE` (not `CONFLICTING`) against Core's current `main`, and the title/body reflect both platforms.
