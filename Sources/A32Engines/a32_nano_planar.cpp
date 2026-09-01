// Kernel 2: n_planar — vectorise across frames instead of across channels.
//
// The core idea of the round, and the one every later candidate is a variation
// on. a2_fast's C=3 branch is scalar: nine weights in registers and three
// accumulator chains, one frame at a time. Cortex-A17's scalar VFP throughput is
// weak next to its NEON, and three chains come nowhere near covering VFP FMA
// latency, so the reference is leaving most of the machine idle.
//
// Planar layout puts each channel in its own plane, so one Q register holds four
// consecutive frames of one channel and every lane does real work:
//
//   a2_fast   9 scalar FMA per frame
//   planar    9 vector FMA per 4 frames = 2.25 FMA/frame
//
// and z lives in three registers across all K taps instead of making 2 × K
// round-trips to memory per frame.
//
// It is also exact. The weight in a planar FMA is a broadcast scalar and the
// vector is four frames, so each lane runs a2_fast's own per-frame reduction in
// a2_fast's own order — bias, then tap 0's inputs 0,1,2, then tap 1, and so on.
// Nothing is reassociated, which is why the parity number for this family is
// zero rather than small.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{

std::unique_ptr<nam::DSP> make_nano_planar(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<NanoPlanarOptions>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
