// Kernel 6: n_ringdirect — write each layer's residual straight into the next
// layer's ring.
//
// Every layer produces a residual, stores it in a scratch buffer, and the next
// layer then memcpys that buffer into its own history ring. Twenty-three layers,
// a block-sized copy each, for data that was in a register moments earlier.
//
// The ring's write region is contiguous by construction — see a32_ring.h — so
// the residual store can address it directly and the copy disappears. On AArch64
// this was worth 1.273× at C=3; a slower memory system relative to its cores
// should make it worth more here, not less.
//
// Measured, it depends on the tile, and it changes sign:
//
//   on tile 4 (this kernel)   0.958x against n_planar    — a loss
//   on tile 8 (n_stacked8_ring) 1.071x against n_widetile8 — the stack's main win
//
// The store into the next ring needs another live destination pointer per
// channel in the layer tail. At tile 4 the tail is where the register pressure
// already is, and paying three more pointers there costs more than the twenty-
// three block copies it saves. At tile 8 the copies dominate. This is the
// clearest case in the round of a switch that cannot be evaluated on its own.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options : NanoPlanarOptions
{
  static constexpr bool kRingDirect = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_ringdirect(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
