// Kernel 13: n_vmla — what does bit-identity cost on this part?
//
// NB_A32_NOT_EXACT.
//
// Identical to the tile-16 planar kernel in every respect except one: the conv
// accumulates with a separate multiply and add instead of a fused multiply-add.
// That is two roundings where a2_fast has one, so this kernel is *not*
// bit-identical and is registered exact = false. It exists to put a number on
// what the exactness rule costs, which is otherwise an article of faith.
//
// Measured: 0.614x against a2_fast where the same kernel fused is 1.010x, so
// abandoning bit-identity here costs about 40% rather than buying anything. Two
// causes, and the object file separates them: the conv's instruction count
// doubles (nine multiplies and nine adds per tap-vector where there were nine
// FMAs), and the kernel spills 663 times in the K=6 body against tile 16's 534,
// because the extra products are extra live values. Read it as an upper bound on
// what exactness costs: the round_now barrier that stops GCC folding the pair
// back into a vfma also denies it some scheduling freedom.
//
// A caveat about the name, recorded because it changes what the number means.
// GCC compiles the unfused form to vmul.f32 + vadd.f32 rather than folding it
// into the single vmla.f32 instruction, so what this prices is the second
// rounding and the extra instruction *slot*, not ARM's non-fused
// multiply-accumulate encoding. Reaching for vmlaq_f32 explicitly would price
// that instead; a32_neon_compat.h deliberately makes that a thing a kernel has
// to spell out for itself.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

struct Options : NanoPlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kFused = false;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_vmla(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
