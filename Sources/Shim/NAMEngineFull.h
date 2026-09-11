// Umbrella header for the full-path kernel lab framework.
//
// Built from vendor/upstream, through the same Engine target template as the
// `upstream` variant, so the only difference between the two builds is
// NB_ENABLE_FULL_LAB and the extra sources under Sources/FullEngines.
//
// One control justifies the arrangement. Kernel 0 ("a2_baseline", a verbatim
// port of a2_fast's Channels==8 branch) has to land on top of the `upstream`
// number. If it does not, the lab is measuring itself.

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
