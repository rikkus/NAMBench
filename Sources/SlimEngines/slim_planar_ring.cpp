// Kernel 5: planar_ring — planar with the rings sized to fit the cache.
//
// a2_fast rounds every ring up to a power of two so reads can be masked, and
// then mirrors all 24 of them unconditionally on every block — about 18 KB of
// memcpy per 64-frame block that in the overwhelming majority of blocks nothing
// reads. Sizing each ring exactly and mirroring only when a read genuinely
// wraps (what the fused engine does) takes the footprint from
//
//     172.5 KB  (pow2 capacity + a full mirror on every ring)
//  to 110.4 KB  (exact capacity + a mirror that is usually untouched)
//
// which is the difference between not fitting in the M2 P-core's 128 KB L1D and
// fitting in it, for a workload whose entire per-frame arithmetic is 1731 MACs.
//
// The cost is that reads can no longer be masked: an exactly-sized ring needs a
// compare-and-subtract instead of an AND. That is once per tap per block, not
// per frame, so it should not show up — but "should not" is why this is being
// measured rather than assumed.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr RingKind kRing = RingKind::ExactLazy;
};

} // namespace

std::unique_ptr<nam::DSP> make_planar_ring(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
