// Umbrella header for the 32-bit ARM lab variant framework.
//
// Built from the *same* sdatkinson/NeuralAmpModelerCore tree as the upstream
// variant, through the same target template, so the only difference between the
// two builds is NB_ENABLE_A32_LAB and the extra sources under Sources/A32Engines.
// That is what makes the two controls valid: if `n_baseline` and `s_baseline` —
// verbatim ports of a2_fast's Channels==3 and Channels>=8 branches — do not land
// on top of the upstream numbers, the lab is measuring itself.
//
// Unlike the slim and full labs this one carries kernels for *both* A2
// submodels, because on this target both share one reference (a2_fast), one
// vendor tree, one compat header and one 16-register problem. Which submodel a
// given kernel serves is asked through nb_a32_kernel_channels rather than being
// a property of the lab.

#ifndef NAM_ENGINE_A32_H
#define NAM_ENGINE_A32_H

#include "nam_bench_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

NB_DECLARE_VARIANT(nb_a32)
NB_DECLARE_KERNEL_LAB(nb_a32)

#ifdef __cplusplus
}
#endif

#endif // NAM_ENGINE_A32_H
