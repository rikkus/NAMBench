// Kernel 6: ringdirect — stop laundering the residual through a scratch buffer.
//
// a2_fast writes each layer's residual into `_layer_in`, and the next layer
// starts by memcpy-ing `_layer_in` into its own ring. That is 23 memcpys and 23
// extra passes over a 768-byte buffer per block, for data that was already in
// registers when it was computed. Having the tail store straight into the next
// layer's ring removes both.
//
// Why it is not obviously a win: `_layer_in` is one small, always-hot buffer
// that 23 layers share, whereas the ring windows are 24 scattered locations
// spread across 172 KB. Trading a hot buffer for cold scatter could easily cost
// more than the memcpys saved — which is the whole reason it is a candidate and
// not just a cleanup.
//
// The block being written may straddle the ring wrap. Rather than splitting the
// frame loop, the write runs off the end of the ring into the mirror region
// (which is always at least max_buffer_size columns long) and the overhang is
// folded back afterwards, so the inner loop stays straight-line.

#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_planar_kernel.h"

namespace slimlab
{
namespace
{

struct Options : PlanarOptions
{
  static constexpr bool kRingDirect = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_ringdirect(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
