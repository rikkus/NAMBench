// Kernel: planar_linear — the third ring strategy, and the one the data asked
// for.
//
// Not in the original candidate list. It was added after the ring-mode sweep,
// which turned up something the plan did not anticipate: compiling *upstream's
// own* a2_fast with NAM_A2_RING_MODE=0 — a linear buffer rewound by memmove
// instead of a power-of-two ring with an eagerly mirrored tail — made it 14%
// faster on this submodel. a2_fast ships with mode 1.
//
// That is a large enough effect to be worth carrying over: the whole planar
// family inherits mode 1's eager mirror, which at three channels copies
// 23 layers × 3 planes × 64 frames × 4 bytes ≈ 17.7 KB per block for data that
// in most blocks nothing reads. A linear buffer has no mirror at all, no
// masking, and every read contiguous; it pays instead with an occasional large
// memmove — 14 KB on the dilation-239 layers, but only once every nineteen
// blocks, and only 20 bytes per block on the dilation-1 layers where mode 1 was
// copying 768.
//
// `planar_ring` (exactly-sized capacity, lazily mirrored) is the other end of
// the same trade and lost badly, for a reason worth writing down: an exactly
// sized ring for a short-lookback layer is barely longer than one block, so
// almost every block wraps, and the lazy mirror then fires once per *tap*
// rather than once per layer. Smallest footprint, most copying.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct StackedOptions : PlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr RingKind kRing = RingKind::LinearRewind;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_planar_linear(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_stacked_linear(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<StackedOptions>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
