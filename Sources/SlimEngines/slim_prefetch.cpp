// Kernel 10: prefetch — ask for the far taps before they are wanted.
//
// The dilation-239 layers reach 1195 frames back, which at three channels is
// about 14 KB behind the write head, and there are six such layers plus six at
// dilation 101. Nothing in the block loop touches that memory between one block
// and the next, so by the time a tap is read it is a cold miss every time.
//
// Expected to lose, or to do nothing. Apple's hardware prefetchers handle
// strided access well, the access pattern here is a fixed offset that repeats
// every block, and an explicit prefetch that arrives too early evicts something
// that was wanted. It is in the list because "the far dilations must be missing
// L1" is the kind of plausible story that deserves a measurement rather than a
// paragraph.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr bool kPrefetch = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_prefetch(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
