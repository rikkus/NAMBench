// The AArch64-only NEON intrinsics, shimmed for AArch32.
//
// This is the only file in the lab that knows the two instruction sets differ.
// Everything else is written once and compiles for both.
//
// -----------------------------------------------------------------------------
// What is missing, and why
//
// The AArch64 planar kernels are built around multiply-accumulate against a
// *lane* of a loaded weight vector — vfmaq_laneq_f32 and friends. ARMv7's VFPv4
// has no VFMA-by-scalar encoding at all, so none of the by-element forms exist:
//
//   vfmaq_laneq_f32   absent      vdupq_laneq_f32   absent
//   vfmaq_lane_f32    absent      vmulq_laneq_f32   absent
//   vfmaq_n_f32       absent in GCC, present in clang
//
// Present and usable: vfmaq_f32, vld1q_dup_f32, vdupq_lane_f32, vmulq_lane_f32,
// vmulq_n_f32, vextq_f32, vbslq_f32, vcltq_f32, vld1q_f32, vst1q_f32.
//
// Note vfmaq_n_f32 in particular: it is available under clang but not under GCC,
// which is the toolchain this target actually builds with. Probing one compiler
// and generalising would have missed that.
//
// -----------------------------------------------------------------------------
// Why the substitution is identical, not merely similar
//
// The whole campaign's value rests on the replacement computing the same bits,
// so the argument is worth stating precisely. Three parts:
//
//   1. A splat — vdup, vld1q_dup — is a bit-copy. The multiplicand the FMA sees
//      is the identical float32 that the lane-addressed form would have seen.
//      No rounding is introduced.
//   2. vfmaq_f32 is VFMA.F32: a true fused multiply-add, one rounding, exactly
//      as vfmaq_laneq_f32 is on AArch64.
//   3. Lane independence is untouched. Each lane still runs one frame's chain in
//      a2_fast's order, so nothing is reassociated.
//
// There is a second way to break it that has nothing to do with the kernels, and
// it was found the hard way: at C=8 the *reference* is Eigen, and Eigen's gebp
// kernel under clang 18.1.3 on this target emits non-fused vmla.f32 where gcc
// 13.3 emits vfma. a2_fast therefore computes different bits under the two
// compilers, and a kernel that is bit-identical to one is 127 dB from the other.
// Scripts/a32-codegen-check.sh catches this at build time — see the clang
// toolchain file for the numbers. Every C=8 parity claim here is a gcc claim.
//
// The only way to break this is to reach for VMLA, which on ARM is *not* fused —
// two roundings. That is why vmlaq_lane_f32 is deliberately absent from this
// header even though it exists and is one instruction shorter: a kernel wanting
// it must spell it out itself and declare exact = false.
// Scripts/a32-codegen-check.sh fails the build if a kernel claiming exactness
// contains vmla.f32, so this is enforced rather than trusted.

#pragma once

#if defined(NB_ENABLE_A32_LAB)

  #include <arm_neon.h>

// The premise, checked rather than assumed. GCC's default for this target is
// -march=armv7-a+fp (VFPv3-D16): no NEON, no FMA. -mfpu=neon gives NEON but
// still no FMA, at which point Eigen stops defining EIGEN_VECTORIZE_FMA and
// silently computes a2_fast's C=8 path with non-fused vmlaq_f32 — a different
// arithmetic to be bit-identical to. Failing loudly here is much better than
// discovering it as a parity mismatch.
  #if !defined(__aarch64__) && (!defined(__ARM_NEON) || !defined(__ARM_FEATURE_FMA))
    #error "the A32 lab needs NEON with VFPv4 fused multiply-add: build with -mfpu=neon-vfpv4"
  #endif

  #define NB_A32_INLINE inline __attribute__((always_inline))

namespace a32lab
{
namespace compat
{

/// Broadcast lane L of `v` to all four lanes.
///
/// Templated on the lane, never taking it at runtime: the lane is encoded in the
/// vdup.32 opcode, so a runtime index lowers to a spill and reload or a switch
/// ladder. Every call site in the kernels passes a literal, so nothing is lost.
template <int L>
NB_A32_INLINE float32x4_t dupq_laneq(float32x4_t v)
{
  static_assert(L >= 0 && L < 4, "lane out of range");
  #if defined(__aarch64__)
  return vdupq_laneq_f32(v, L);
  #else
  return vdupq_lane_f32(L < 2 ? vget_low_f32(v) : vget_high_f32(v), L & 1);
  #endif
}

/// acc + s * v[L], one rounding.
template <int L>
NB_A32_INLINE float32x4_t fmaq_laneq(float32x4_t acc, float32x4_t s, float32x4_t v)
{
  #if defined(__aarch64__)
  return vfmaq_laneq_f32(acc, s, v, L);
  #else
  return vfmaq_f32(acc, s, dupq_laneq<L>(v));
  #endif
}

/// s * v[L], one rounding. Used to seed a chain where an FMA against zero would
/// otherwise be needed; both are one rounding of w*x, so this is exact either
/// way and just saves the init.
template <int L>
NB_A32_INLINE float32x4_t mulq_laneq(float32x4_t s, float32x4_t v)
{
  #if defined(__aarch64__)
  return vmulq_laneq_f32(s, v, L);
  #else
  return vmulq_f32(s, dupq_laneq<L>(v));
  #endif
}

/// acc + s * w, one rounding, with the scalar loaded and broadcast in one
/// instruction (vld1.32 {d[],d[]}).
///
/// This is the form that replaces a lane-addressed weight without holding the
/// weight vector in a register at all — which on a 16-register machine is the
/// property that matters most, so it is the lab's default weight delivery.
NB_A32_INLINE float32x4_t fmaq_bcast(float32x4_t acc, float32x4_t s, const float* w)
{
  return vfmaq_f32(acc, s, vld1q_dup_f32(w));
}

/// s * w, one rounding, same delivery as fmaq_bcast.
NB_A32_INLINE float32x4_t mulq_bcast(float32x4_t s, const float* w)
{
  return vmulq_f32(s, vld1q_dup_f32(w));
}

/// acc + s * w for a scalar already in a register. GCC lacks vfmaq_n_f32 on
/// ARMv7, so it is spelled out.
NB_A32_INLINE float32x4_t fmaq_n(float32x4_t acc, float32x4_t s, float w)
{
  #if defined(__aarch64__)
  return vfmaq_n_f32(acc, s, w);
  #else
  return vfmaq_f32(acc, s, vdupq_n_f32(w));
  #endif
}

/// Force `v` to be a rounded value before it is used again.
///
/// The vector analogue of a2_planar.cpp's mul_rounded(), and on this target it
/// is mandatory rather than defensive. a2_fast's C=8 layer body computes the
/// mixin as a product and then a separate add — two roundings — and under
/// -ffp-contract=fast (GCC's default) the compiler will happily fold a vector
/// multiply and add into one vfma, which is one rounding and different bits.
///
/// This was measured, not guessed: Scripts/eigen-order-probe under
/// arm-linux-gnueabihf-g++ matches Eigen 224/224 at -ffp-contract=off and only
/// 11/224 at =fast, with the conv stage unaffected either way.
///
/// Turning contraction off globally is the wrong fix — a2_fast's C=3 branch
/// *depends* on the compiler contracting a*b+c, which is the whole bit-identity
/// premise at that width. So contraction stays on and is blocked here, locally,
/// at exactly the points a2_fast rounds.
///
/// Compiles to nothing; "w" is the VFP/NEON register constraint on ARM.
NB_A32_INLINE float32x4_t round_now(float32x4_t v)
{
  __asm__("" : "+w"(v));
  return v;
}

/// Scalar counterpart, for the tail frames a tiled kernel handles one at a time.
NB_A32_INLINE float round_now(float v)
{
  __asm__("" : "+w"(v));
  return v;
}

// -----------------------------------------------------------------------------
// Lane width as a parameter
//
// At C=8 a planar tile of four frames wants eight accumulators plus eight
// per-tap partials plus an input and a weight — eighteen Q registers on a
// machine with sixteen, which is exactly what the measured spill counts show.
// The shape that fits is a *two*-frame tile held in D registers: the same
// eighteen values, but 64 bits each, so eighteen of the thirty-two D registers.
//
// AArch32 NEON is a 32 × D file that a Q register views two at a time, so this
// is not a workaround — it is the register file's own granularity. Everything
// below exists so the layer body can be written once and instantiated at either
// width, because the one thing that must not vary with lane count is the order
// the additions happen in.

/// The 4-lane (Q register) width.
struct Q4
{
  using type = float32x4_t;
  static constexpr int kLanes = 4;

  static NB_A32_INLINE type dup(float v) { return vdupq_n_f32(v); }
  static NB_A32_INLINE type load(const float* p) { return vld1q_f32(p); }
  static NB_A32_INLINE void store(float* p, type v) { vst1q_f32(p, v); }
  static NB_A32_INLINE type add(type a, type b) { return vaddq_f32(a, b); }
  static NB_A32_INLINE type mul(type a, type b) { return vmulq_f32(a, b); }
  static NB_A32_INLINE type fma_bcast(type acc, type s, const float* w)
  {
    return vfmaq_f32(acc, s, vld1q_dup_f32(w));
  }
  static NB_A32_INLINE type mul_bcast(type s, const float* w)
  {
    return vmulq_f32(s, vld1q_dup_f32(w));
  }
  /// LeakyReLU, as a2_fast and Eigen both spell it: negative lanes scaled.
  static NB_A32_INLINE type leaky(type v, type slope)
  {
    return vbslq_f32(vcltq_f32(v, vdupq_n_f32(0.0f)), vmulq_f32(v, slope), v);
  }
  static NB_A32_INLINE type round_now(type v)
  {
    __asm__("" : "+w"(v));
    return v;
  }
};

/// The 2-lane (D register) width. Same operations, half as wide.
struct D2
{
  using type = float32x2_t;
  static constexpr int kLanes = 2;

  static NB_A32_INLINE type dup(float v) { return vdup_n_f32(v); }
  static NB_A32_INLINE type load(const float* p) { return vld1_f32(p); }
  static NB_A32_INLINE void store(float* p, type v) { vst1_f32(p, v); }
  static NB_A32_INLINE type add(type a, type b) { return vadd_f32(a, b); }
  static NB_A32_INLINE type mul(type a, type b) { return vmul_f32(a, b); }
  static NB_A32_INLINE type fma_bcast(type acc, type s, const float* w)
  {
    return vfma_f32(acc, s, vld1_dup_f32(w));
  }
  static NB_A32_INLINE type mul_bcast(type s, const float* w)
  {
    return vmul_f32(s, vld1_dup_f32(w));
  }
  static NB_A32_INLINE type leaky(type v, type slope)
  {
    return vbsl_f32(vclt_f32(v, vdup_n_f32(0.0f)), vmul_f32(v, slope), v);
  }
  static NB_A32_INLINE type round_now(type v)
  {
    __asm__("" : "+w"(v));
    return v;
  }
};

/// Q4 for any tile that is a whole number of Q registers, D2 for a 2-frame one.
template <int LANES>
struct VecWidth;
template <>
struct VecWidth<4>
{
  using type = Q4;
};
template <>
struct VecWidth<2>
{
  using type = D2;
};

} // namespace compat
} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
