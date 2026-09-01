// Kernel 16: n_prefetch — ask for the next block's far taps early.
//
// Written expecting it to lose. It lost on AArch64, where an M2's own
// prefetcher had already done the work. It is here anyway because a Cortex-A17
// has a much weaker prefetcher, and a prior loser on a different
// microarchitecture is exactly the kind of result that flips — which is the only
// reason to spend a file on it.
//
// The layers with dilation >= 41 reach thousands of frames back, several
// kilobytes per channel plane, so their taps are certain to be out of L1 by the
// time the next block wants them. The pld goes out one block ahead, sixteen
// floats apart, on those layers only.
//
// It lost again: 0.974x against a2_fast where the same tile-16 kernel without it
// is 1.010x, so the pld costs about 3.4% and buys nothing measurable. The flip
// did not happen. Worth having asked — the A17's prefetcher is nothing like an
// M2's, and "it lost on other hardware" is not an answer about this one.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options : NanoPlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kPrefetch = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_prefetch(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
