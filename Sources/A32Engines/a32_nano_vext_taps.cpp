// Kernel 8: n_vext_taps — build the tap windows by rotation instead of by
// loading each one.
//
// At dilation 1 the K tap windows of a 4-frame tile overlap almost completely:
// taps 0..K-1 of a K=6 layer span nine consecutive frames, which is three vector
// loads, not six. vextq_f32 rotates two loaded vectors into the window between
// them, so the six windows come out of three loads per channel instead of six.
// At dilation 3 the span is 19 frames — five loads for six windows.
//
// Cheap on this core, and the loads it removes are the ones the tap loop is
// otherwise dominated by. The reason it may still lose: those loaded vectors
// have to stay live across the whole tap loop, and at K=15, dilation 1 that is
// five vectors per channel — fifteen registers on a machine with sixteen. This
// candidate is where the 16-register budget gets tested against something other
// than tile width.
//
// The arithmetic is untouched: a window is the same four floats whether they
// arrived by load or by rotation, and a32_planar_kernel.h drives both paths
// through the same reduction.
//
// Measured: 0.922x against n_planar. The register pressure won, as the note
// above suspected — 359 vldr/vstr in the K=6 body against n_planar's 50, for a
// kernel that removes loads. It trades L1 hits, which this core pipelines, for
// stack traffic, which it does not.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options : NanoPlanarOptions
{
  static constexpr bool kVextTaps = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_vext_taps(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
