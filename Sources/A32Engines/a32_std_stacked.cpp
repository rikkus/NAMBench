// Kernels 25-26: s_stacked{2,4} — the composition.
//
// Two tile widths for the same reason the C=3 round used two: the best tile for
// the composed kernel need not be the best tile for the plain one. kRingDirect
// moves work out of the block-copy and into the layer tail's register pressure,
// and at C=8 the tail is already where this kernel is tightest.
//
// Tiles 2 and 4 were what the register budget said to compose on. The tile sweep
// then found the plain kernel still climbing at 8 (see a32_std_planar.cpp), so 8
// and 16 were added: composing onto a tile that is not the best plain tile
// measures the wrong thing, and the switches were worth 3-6% on the widths they
// were first tried on.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options2 : StdPlanarOptions
{
  static constexpr int kTile = 2;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options4 : StdPlanarOptions
{
  static constexpr int kTile = 4;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options8 : StdPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options16 : StdPlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

/// The winner under the other two ring strategies, for the same reason as at
/// C=3 — and with more at stake here, because a C=8 ring holds 2.7x the bytes a
/// C=3 one does and the mirror memcpy scales with it.
///
/// Measured, there is almost nothing in it: lazy 1.312x, linear 1.309x, eager
/// 1.298x. So the prediction in the sentence above is wrong — the strategy
/// matters *less* at the wider model, not more, even though every copy it avoids
/// is bigger. At C=8 the arithmetic per block is an order of magnitude larger
/// while the ring traffic only doubles, so the same saving is a smaller share of
/// a bigger number. Lazy is kept as the winner on a 1% margin against a 0.04%
/// spread; it is a real difference and a small one.
struct Options8Linear : StdPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct Options8Lazy : StdPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

} // namespace

std::unique_ptr<nam::DSP> make_std_stacked8_linear(const std::vector<float>& weights,
                                                   double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Linear>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_stacked8_lazy(const std::vector<float>& weights,
                                                 double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Lazy>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_stacked2(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options2>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_stacked4(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options4>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_stacked8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_stacked16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
