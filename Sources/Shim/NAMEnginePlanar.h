// Umbrella header for the planar variant framework.
//
// Built from rikkus/OptimisationWorkOnNeuralAmpModelerCore at the head of the
// apple-silicon-a2-planar branch — the code proposed to Core in PR #313, not a
// restatement of it, so a number measured here belongs to that proposal.
//
// That branch is cut from upstream main and carries no fused engine: it is
// a2_fast, plus a2_planar, plus two lines in A2FastConfig::create that prefer
// the planar model where one exists. So this framework is built exactly as
// NAMEngineUpstream is, with no NAM_ENABLE_FUSED and no ScopedEnginePrefer —
// only NB_VENDOR_HAS_PLANAR, which lets the shim see a2_planar.h and report
// which of the two it actually got.
//
// It reports `planar` only where the kernels exist. NAM_A2_PLANAR follows
// __aarch64__, so an x86 build of this framework honestly reports a2_fast and
// the runner refuses to present that as a comparison.

#ifndef NAM_ENGINE_PLANAR_H
#define NAM_ENGINE_PLANAR_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_planar)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_PLANAR_H
