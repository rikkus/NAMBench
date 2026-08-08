// Umbrella header for the full-path kernel lab framework.
//
// Built from the *same* rikkus/OptimisationWorkOnNeuralAmpModelerCore tree as
// the `fused` variant, through the same Engine target template, so the only
// difference between the two builds is NB_ENABLE_FULL_LAB and the extra sources
// under Sources/FullEngines. That tree is a superset: it carries a byte-identical
// a2_fast *and* fused, which is why one lab can hold candidates measured against
// both references.
//
// Two controls justify the arrangement. Kernel 0 ("a2_baseline", a verbatim port
// of a2_fast's Channels==8 branch) has to land on top of the `upstream` number,
// and kernel 1 ("fu_baseline", a verbatim port of fused's C=8 path) has to land
// on top of the `fused` number. If either does not, the lab is measuring itself.

#ifndef NAM_ENGINE_FULL_H
#define NAM_ENGINE_FULL_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_full)
NB_DECLARE_KERNEL_LAB(nb_full)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_FULL_H
