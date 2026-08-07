// Kernel 4: planar — the expected winner.
//
// Plain structure-of-arrays with a 4-frame tile and nothing else changed: the
// same power-of-two rings a2_fast uses, mirrored eagerly on every write, the
// same scratch-buffer hand-off between layers, the same work at the end. Every
// other candidate in this family is this one plus a single switch, so this is
// the number they are all measured against.
//
// See slim_planar_kernel.h for why the arithmetic comes out bit-identical to
// a2_fast's despite being vectorised.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
};

} // namespace

std::unique_ptr<nam::DSP> make_planar(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
