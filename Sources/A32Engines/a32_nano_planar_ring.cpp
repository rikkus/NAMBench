// Kernels 9-10: n_planar_linear, n_planar_lazy — the same kernel under the other
// two ring strategies.
//
// Ring strategy was the biggest surprise of the AArch64 nano round: compiling
// a2_fast at NAM_A2_RING_MODE=0 was worth 14%, more than most kernels. The cost
// it removes is a2_fast's default (mode 1, Pow2Eager), which memcpys a mirror of
// the first max_buffer columns after *every* write, on every one of the
// twenty-three layers, whether or not any read goes on to wrap into it.
//
//   linear   no mask, no mirror, every read contiguous, at the price of an
//            occasional large memmove when the buffer runs out
//   lazy     pow2 and masked as the default is, but the mirror is refreshed
//            only on the blocks whose read genuinely wraps
//
// Both are exact: a ring decides where the numbers are, never what they are.
//
// Measured here, on the plain 4-frame planar kernel, both lose: linear 1.055x
// and lazy 1.062x against the default's 1.116x. Do not read that as the answer —
// Phase 5 repeated the comparison on the round's winner, where the scratch-buffer
// copies these kernels still perform have been removed, and there linear *wins*
// by 4%. See a32_nano_stacked.cpp. This file is what the question looks like
// asked on the wrong base.
//
// n_planar_lazy is not in the campaign plan's kernel table; it is here because
// Phase 5's ring sweep names modes 0/1/2 and the lazy mirror is mode 2. Having
// it as a kernel rather than only as a build flag means the sweep can measure it
// without a rebuild.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct OptionsLinear : NanoPlanarOptions
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct OptionsLazy : NanoPlanarOptions
{
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_planar_linear(const std::vector<float>& weights,
                                                  double sampleRate)
{
  return std::make_unique<PlanarModel<OptionsLinear>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_planar_lazy(const std::vector<float>& weights,
                                                double sampleRate)
{
  return std::make_unique<PlanarModel<OptionsLazy>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
