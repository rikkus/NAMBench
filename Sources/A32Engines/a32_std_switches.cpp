// Kernels 23-24: s_ringdirect, s_skiplast — the two switches that paid at C=3,
// at C=8.
//
// Both are layered on the 2-frame tile, which is the width the register budget
// and the spill counts both pick before any of this is timed. The C=3 round is
// the reason that choice is stated rather than assumed: there, kRingDirect lost
// 4% on a 4-frame tile and won 7% on an 8-frame one, so a switch measured on the
// wrong base answers a question nobody asked.
//
// What they should do here, and why the C=3 numbers do not transfer:
//
//   ringdirect  removes 23 block-sized copies per block. At C=8 each copy is
//               2.7x the bytes it was at C=3, so if it pays at all it should pay
//               more — but it also needs C destination pointers live in the
//               layer tail, and at C=8 there are eight of them, not three.
//   skiplast    drops the last layer's layer1x1 (at C=8 that is 64 FMAs per
//               frame, not 9) and the per-block head_sum memset. Strictly less
//               work either way.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct OptionsRingDirect : StdPlanarOptions
{
  static constexpr int kTile = 2;
  static constexpr bool kRingDirect = true;
};

struct OptionsSkipLast : StdPlanarOptions
{
  static constexpr int kTile = 2;
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_std_ringdirect(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<OptionsRingDirect>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_skiplast(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<OptionsSkipLast>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
