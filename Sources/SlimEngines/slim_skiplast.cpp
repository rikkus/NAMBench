// Kernel 9: skiplast — two things that are simply free.
//
// 1. The final layer's layer1x1 computes a residual that nothing ever reads:
//    there is no layer 24, and the head reads `head_sum`, not `_layer_in`. The
//    fused engine already skips it (`skip_l1x1_output`); a2_fast does not. That
//    is one 3×3 GEMM plus a bias and a store per frame, out of 23 layers.
//
// 2. The first layer is the only one whose `head_sum` accumulate has nothing to
//    accumulate onto, so it can store instead of load-add-store — and once it
//    does, the per-block memset of `head_sum` has nothing left to zero.
//
// Neither is an idea about the machine; both are just work the current code
// does for no reason. Together they are worth about 2% over `planar`, which is
// more than the arithmetic alone suggests — dropping the last layer1x1 also
// drops the store traffic that went with it, and the memset removal takes a
// whole pass over head_sum with it.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_skiplast(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
