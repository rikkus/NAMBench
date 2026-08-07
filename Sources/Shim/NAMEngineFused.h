// Umbrella header for the fused variant framework.
//
// Built from rikkus/OptimisationWorkOnNeuralAmpModelerCore with
// NAM_ENABLE_FUSED, and constructed under ScopedEnginePrefer(FusedNeon).

#ifndef NAM_ENGINE_FUSED_H
#define NAM_ENGINE_FUSED_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_fused)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_FUSED_H
