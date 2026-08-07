// Umbrella header for the upstream variant framework.
//
// Built from sdatkinson/NeuralAmpModelerCore, which has no fused engine, so
// wavenet::create_config routes the A2 shape to a2_fast on its own.

#ifndef NAM_ENGINE_UPSTREAM_H
#define NAM_ENGINE_UPSTREAM_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_upstream)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_UPSTREAM_H
