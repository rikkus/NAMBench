// Umbrella header for the slim-lab variant framework.
//
// Built from the *same* sdatkinson/NeuralAmpModelerCore tree as the upstream
// variant, through the same Engine target template, so the only difference
// between the two builds is NB_ENABLE_SLIM_LAB and the extra sources under
// Sources/SlimEngines. That is what makes kernel 0 ("baseline", a verbatim port
// of a2_fast's Channels==3 branch) a valid control: if it does not land on top
// of the upstream number, the lab is measuring itself.

#ifndef NAM_ENGINE_SLIM_H
#define NAM_ENGINE_SLIM_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_slim)
NB_DECLARE_KERNEL_LAB(nb_slim)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_SLIM_H
