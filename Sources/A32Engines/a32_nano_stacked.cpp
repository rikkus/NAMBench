// Kernels 11-12: n_stacked{8,16} — everything that paid, together.
//
// The round's order is to measure each switch against `n_planar` on its own and
// then compose the ones that won. Two tile widths rather than one, because the
// best tile for the composed kernel need not be the best tile for `n_planar`
// alone: kRingDirect changes the register pressure in the layer tail, which is
// exactly where a wide tile runs out of registers.
//
// The composition is fixed here from the AArch64 result and the advance
// prediction; if the C=3 measurements below disagree about which switches paid,
// this file is what gets edited, and the write-up records that it was.
//
// They disagreed, and the two half-stacks below are what the disagreement asked
// for. Measured against `n_planar` alone, kRingDirect *lost* (1.059x against
// planar's 1.105x over a2_fast) — but composed onto tile 8 the pair gained 6.6%
// over tile 8 by itself. One of the two switches is carrying that, or the
// interaction is, and with only the full stack measured there is no way to say
// which. So: tile 8 with each switch on its own.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options8 : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options16 : NanoPlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options8Ring : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
};

struct Options8Skip : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kSkipLastL1x1 = true;
};

/// The winner under the other two ring strategies. The plain-planar ring
/// comparison (n_planar_linear, n_planar_lazy) was made on a 4-frame tile with
/// the scratch-buffer copy still in place; on the winner the copies are gone
/// (kRingDirect), which is exactly the traffic a ring strategy is supposed to be
/// judged on. So the sweep is repeated where it now means something.
///
/// It was worth repeating, because the answer inverts:
///
///                     on n_planar (tile 4)   on the winner (tile 8, ringdirect)
///   pow2 + eager      1.116x  (the default)  1.322x
///   pow2 + lazy       1.062x                 1.322x
///   linear + rewind   1.055x                 1.378x  <- the round's best
///
/// Linear loses by 5% measured against the plain kernel and wins by 4% measured
/// against the winner. With the scratch copies still in place they dominate, and
/// the ring strategy is noise on top of them; remove the copies and the mirror
/// memcpy the eager strategy performs on every write of all 23 layers becomes
/// the largest remaining piece of memory traffic in the block.
struct Options8Linear : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct Options8Lazy : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_stacked8_linear(const std::vector<float>& weights,
                                                    double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Linear>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_lazy(const std::vector<float>& weights,
                                                  double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Lazy>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_ring(const std::vector<float>& weights,
                                                  double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Ring>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_skip(const std::vector<float>& weights,
                                                  double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Skip>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
