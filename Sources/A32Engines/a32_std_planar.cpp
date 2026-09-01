// Kernels 20-22: s_planar{2,4,8} — the C=8 tile sweep.
//
// Planar at eight channels is the same idea as at three and a much tighter fit.
// Per tile of T frames the layer body wants, in vector registers:
//
//   z      C * T / lanes        the accumulators, live across every tap
//   t      C * T / lanes        Eigen's per-tap partial (see the Reference note
//                               in a32_planar_kernel.h — at C=8 the taps do not
//                               share a chain, so both arrays exist at once)
//   input  T / lanes
//   weight 1
//
// At T=4 that is 8 + 8 + 1 + 1 = 18 Q registers on a machine with 16. At T=8 it
// is 34. Only T=2 fits, and it fits because AArch32's register file is 32 D
// registers that Q views in pairs: eighteen D registers, comfortably.
//
// The objdump agrees before anything is timed — vldr/vstr in the K=6 layer body:
//
//   T=2   85     T=4  203     T=8  981
//
// So the prediction on the way in is that this sweep turns at 2, which is one
// rung below where FULL-PATH.md's AArch64 equivalent turned (8) and matches the
// campaign plan's own arithmetic for 16 registers. Writing 4 and 8 anyway is
// what makes the turn measurable rather than asserted.
//
// It was wrong, and not by a little:
//
//   T=2    85 spills   1.049x over a2_fast
//   T=4   211 spills   1.132x
//   T=8  1003 spills   1.267x   <- the most spills and the fastest
//
// T=16 was added afterwards, and it does turn there — hard. With Phase 5's
// remaining rungs the completed C=8 ladder is:
//
//   T=2  1.049x     T=8  1.255x  <- the peak
//   T=4  1.126x     T=12 1.135x
//                   T=16 0.899x      T=32 0.822x
//
// so the peak is 8, four rungs above where the budget put it — and the same tile
// the C=3 ladder peaks at, which is the more useful half of the result: the best
// tile is not a function of the channel count at all. What the model got
// wrong is which resource binds. Each tap broadcasts C x C = 64 weights whatever
// the tile is, so per *frame* a 2-frame tile issues four times the weight loads
// an 8-frame one does — and at C=8 that traffic outweighs the accumulator
// spilling the model was counting. Spills land in a hot stack slot; the weight
// broadcasts are the inner loop's real cost. At C=3 the same arithmetic held,
// because there a tap is only 9 weights and the spills dominated instead.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options2 : StdPlanarOptions
{
  static constexpr int kTile = 2;
};

struct Options4 : StdPlanarOptions
{
  static constexpr int kTile = 4;
};

struct Options8 : StdPlanarOptions
{
  static constexpr int kTile = 8;
};

struct Options12 : StdPlanarOptions
{
  static constexpr int kTile = 12;
};

struct Options16 : StdPlanarOptions
{
  static constexpr int kTile = 16;
};

struct Options32 : StdPlanarOptions
{
  static constexpr int kTile = 32;
};

} // namespace

std::unique_ptr<nam::DSP> make_std_planar2(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options2>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_planar4(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options4>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_planar8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_planar12(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options12>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_planar32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_std_planar16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
