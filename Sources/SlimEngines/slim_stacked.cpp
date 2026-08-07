// The composition: everything that won, together.
//
// The plan's run order is "measure 1–4 independently, then stack 5/6/9/10/11
// onto whichever of 2/3/4 wins". `planar` won that first round, and of the
// switches layered onto it these are the ones that paid:
//
//   widetile   the largest single effect by a wide margin
//   ringdirect stop copying each residual through a scratch buffer
//   skiplast   two pieces of work the model never reads
//
// `planar_ring`, `vext_taps` and `prefetch` are left out; see their own files
// and the write-up for what each of them did.
//
// Two tile widths, because the best tile for the composed kernel need not be
// the best tile for `planar` alone: ringdirect changes the register pressure in
// the tail, which is exactly where a wide tile runs out of registers.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options16 : PlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options32 : PlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_stacked16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_stacked32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
