// Kernel 11: widetile — find where ILP stops paying and spilling starts.
//
// `planar` uses a 4-frame tile: one NEON register per channel, three live
// accumulators, three independent FMA chains through the tap loop. Four 4-cycle
// FMA pipes want closer to sixteen chains, so wider tiles should keep going up
// — until 32 registers is not enough to hold them and the compiler starts
// spilling accumulators to the stack, at which point the whole point of keeping
// z in registers is lost.
//
//   tile   live accumulators
//      4   3
//      8   6
//     16   12
//     32   24
//     64   48   (more than the register file has; must spill)
//
// Where the turn is depends on how many registers the tap-weight vectors and
// addressing take up, which is not something to reason about from a manual.
// Four instantiations, one measurement each. 64 is included because it is the
// whole block at the default buffer size, so it is where the sweep has to stop
// being useful even if nothing else stopped it first.
//
// There is a second effect tangled up with the ILP one, and it is probably the
// larger of the two: a wider tile also amortises the *weight* loads. Each tap
// loads three weight vectors regardless of tile width, so at tile 4 that is
// three loads per three FMAs and at tile 32 it is three loads per twenty-four.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options8 : PlanarOptions
{
  static constexpr int kTile = 8;
};

struct Options16 : PlanarOptions
{
  static constexpr int kTile = 16;
};

struct Options32 : PlanarOptions
{
  static constexpr int kTile = 32;
};

struct Options64 : PlanarOptions
{
  static constexpr int kTile = 64;
};

} // namespace

std::unique_ptr<nam::DSP> make_widetile8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_widetile16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_widetile32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_widetile64(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options64>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
