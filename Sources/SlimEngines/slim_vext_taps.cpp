// Kernel 7: vext_taps — reuse overlapping tap windows instead of reloading.
//
// At dilation 1 the six taps of a K=6 layer are six *consecutive* frames, so in
// planar layout the six 4-frame windows a tile needs all live inside one span
// of nine frames. Three loads and five vextq_f32 can produce all six windows
// where the straightforward path issues six loads. Dilation 3 spans nineteen
// frames: five loads instead of six.
//
// Expected to lose, and included because it is expected to lose. `vext`
// competes with the FMAs for the vector issue slots; loads do not, they go down
// the load pipes. Trading pipeline pressure for load pressure is only a win
// when the loads are the bottleneck, and at 2.25 FMA per frame per tap they
// almost certainly are not.
//
// Applies to the dilation-1 and dilation-3 layers (seven of the twenty-three),
// and only for blocks where the tap span did not straddle the ring wrap; every
// other layer and block falls back to the ordinary planar path.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr bool kVextTaps = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_vext_taps(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
