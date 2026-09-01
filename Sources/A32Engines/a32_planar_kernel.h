// The planar (structure-of-arrays) engine for both A2 widths, and the switches
// layered on top of it.
//
// One implementation, many candidates: each candidate .cpp instantiates this
// template with a different option set, so what a measured pair of numbers
// compares really is one idea at a time rather than one rewrite at a time.
//
// The core idea, unchanged from SLIMMED-PATH.md. a2_fast keeps channels
// interleaved and vectorises across *channels*; planar keeps one plane per
// channel and vectorises across *frames*, so a Q register holds four consecutive
// frames of one channel and every lane does real work. Each lane then runs
// a2_fast's own per-frame reduction, in a2_fast's own order, which is what makes
// the result bit-identical rather than merely close.
//
// -----------------------------------------------------------------------------
// Why this file has a `Reference` axis where the AArch64 labs did not
//
// The two submodels are bit-identical to two *different* pieces of arithmetic,
// and the difference is not cosmetic:
//
//   C=3   a2_fast's scalar branch. One chain per output channel, seeded with the
//         bias and carried across every tap, with the compiler contracting
//         a*b+c into a fused multiply-add. The mixin is contracted too.
//   C=8   Eigen. `ztile.noalias() += W * input_block` computes a *partial per
//         tap* from zero and adds it in, so the taps do not share a chain; the
//         bias arrives afterwards as its own add; and the mixin is a product and
//         then a separate add — two roundings, not one.
//
// Both orderings were established on this target rather than assumed:
// Scripts/eigen-order-probe/run.sh, cross-built for cortex-a17 and run on the
// board, matches Eigen 224/224 on the conv stage and 224/224 on the whole layer
// body under exactly the ordering written below.
//
// -----------------------------------------------------------------------------
// The register budget this shape is written against
//
// ARMv7 has 16 Q registers, not 32, and no by-element FMA — a weight reaches an
// FMA through vld1q_dup (see a32_neon_compat.h), which costs one register and no
// permanent one. Per tile of T frames (V = T/4 vectors per channel):
//
//   NanoScalar   z C*V  + input V + weight 1
//   StdEigen     z C*V  + t C*V   + input V + weight 1
//
// so at C=3 the ceiling is around T=16 (12 + 4 + 1) and at C=8 it is T=4
// (8 + 8 + 1 + 1 = 18, already over) or T=2. Those are the predictions Phase 3
// and Phase 4 of the campaign check against measured curves and against the
// vldr/vstr counts Scripts/a32-codegen-check.sh reads out of the object file.
// They are written here so the sweep can contradict them.

#pragma once

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <utility>
  #include <vector>

  #include "a32_common.h"
  #include "a32_neon_compat.h"
  #include "a32_ring.h"

namespace a32lab
{

/// Which engine's arithmetic a kernel reproduces bit for bit.
///
/// This selects the conv reduction order *and* the rounding of everything after
/// it; the two are one decision, because both come from the same reference.
enum class Reference
{
  /// a2_fast's `if constexpr (Channels == 3)` scalar branch.
  NanoScalar,
  /// a2_fast's Eigen branch, as it blocks at C=8.
  StdEigen,
};

/// Defaults for a C=3 candidate. Candidates inherit and override what they
/// change, so a candidate .cpp reads as a diff against `n_planar`.
struct NanoPlanarOptions
{
  static constexpr int kChannels = 3;
  static constexpr Reference kReference = Reference::NanoScalar;
  /// Frames per tile; a multiple of 4. 4 is one Q register per channel, higher
  /// buys independent FMA chains until the accumulators stop fitting.
  static constexpr int kTile = 4;
  /// Frames per iteration of the head. Independent of kTile: the head holds one
  /// accumulator, not C of them, so it can go wider than the layer body.
  static constexpr int kHeadTile = 4;
  static constexpr RingKind kRing = RingKind::Pow2Eager;
  /// Write each layer's residual straight into the next layer's ring instead of
  /// into a scratch buffer that the next layer then copies in.
  static constexpr bool kRingDirect = false;
  /// Skip the last layer's layer1x1 — nothing reads its residual — and fold the
  /// first layer's head_sum accumulate into a store, which removes the memset.
  static constexpr bool kSkipLastL1x1 = false;
  /// Accumulate the conv with VFMA (one rounding, bit-identical) or with a
  /// separate multiply and add (two roundings, *not* identical). A kernel
  /// setting this false must register itself exact = false and carry the
  /// NB_A32_NOT_EXACT marker; Scripts/a32-codegen-check.sh enforces the other
  /// direction.
  ///
  /// What this prices is the second rounding, not the VMLA encoding: GCC emits
  /// vmul.f32 + vadd.f32 here rather than folding the pair into vmla.f32, and
  /// a32_neon_compat.h deliberately offers no vmla helper. A candidate that
  /// wants to price the one-instruction-shorter encoding itself has to spell
  /// vmlaq_f32 out in its own source, which is also what makes that choice
  /// visible in review.
  static constexpr bool kFused = true;
  /// On layers whose taps overlap heavily (dilation 1 and 3), build the tap
  /// windows with vextq_f32 from a smaller number of loads.
  static constexpr bool kVextTaps = false;
  /// Prefetch the next block's taps on the far-dilation layers.
  static constexpr bool kPrefetch = false;
  /// Readable columns past the end of each ring plane, for a kernel that builds
  /// its tap windows with vextq_f32 and over-reads the final window. kVextTaps
  /// implies the eight it needs, so this is only for a kernel that wants slack
  /// for some other reason.
  static constexpr int kTailPad = 0;
};

namespace planar_detail
{

/// The 4-frame window starting OFF frames into a loaded span.
///
/// vextq_f32 is a cheap byte-rotate of two registers, so a run of overlapping
/// tap windows can come out of a handful of loads instead of one load each.
template <int OFF, int NW>
NB_A32_INLINE float32x4_t window(const float32x4_t (&v)[NW])
{
  constexpr int I = OFF / 4;
  constexpr int L = OFF % 4;
  if constexpr (L == 0)
    return v[I];
  else
    return vextq_f32(v[I], v[I + 1], L);
}

} // namespace planar_detail

/// Defaults for a C=8 candidate.
struct StdPlanarOptions : NanoPlanarOptions
{
  static constexpr int kChannels = 8;
  static constexpr Reference kReference = Reference::StdEigen;
  /// Eight accumulators and eight per-tap partials leave room for little else;
  /// see the register budget above.
  static constexpr int kTile = 4;
};

template <class Opt>
class PlanarModel : public A32Model<Opt::kChannels>
{
  static constexpr int C = Opt::kChannels;
  static constexpr int T = Opt::kTile;
  /// Four frames per Q register, or two per D register when the tile is 2. See
  /// the lane-width note in a32_neon_compat.h for why a 2-frame tile is a real
  /// shape on this machine rather than a half-used one.
  static constexpr int kLanes = (T % 4 == 0) ? 4 : 2;
  static constexpr int V = T / kLanes;
  static constexpr int HV = Opt::kHeadTile / 4;
  static constexpr bool kPerTapPartial = (Opt::kReference == Reference::StdEigen);
  /// Under kSkipLastL1x1 the first layer stores into head_sum rather than
  /// accumulating into it, which is what makes the per-block memset unnecessary.
  static constexpr bool kFoldHeadInit = Opt::kSkipLastL1x1;

  static_assert(T == 2 || (T % 4 == 0 && T >= 4),
                "kTile is 2 (D registers) or a whole number of Q registers");
  static_assert(Opt::kHeadTile % 4 == 0 && Opt::kHeadTile >= 4, "likewise kHeadTile");

  using VT = typename compat::VecWidth<kLanes>::type;
  using Vec = typename VT::type;

  using Base = A32Model<C>;
  /// The vext path loads whole 4-frame vectors past the last tap window, so the
  /// plane has to stay readable for two vectors beyond it.
  static constexpr int kTailPad = Opt::kVextTaps ? 8 : Opt::kTailPad;

  using Ring = PlanarRing<C, Opt::kRing, kTailPad>;
  using Base::_w;

public:
  using Base::Base;
  using Base::GetMaxBufferSize;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    // Rechannel, plus the float copy of the condition signal every layer's
    // mixin reads. Under kRingDirect this lands straight in layer 0's ring.
    float* r[C];
    if constexpr (Opt::kRingDirect)
    {
      _rings[0].prepare(N);
      for (int c = 0; c < C; c++)
        r[c] = _rings[0].write_ptr(c);
    }
    else
    {
      for (int c = 0; c < C; c++)
        r[c] = lin(c);
    }

    float* cond = _cond.data();
    for (int f = 0; f < N; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      for (int c = 0; c < C; c++)
        r[c][f] = _w.rechannel_w[c] * x;
    }

    if constexpr (Opt::kRingDirect)
      _rings[0].commit(N);

    if constexpr (!kFoldHeadInit)
    {
      for (int c = 0; c < C; c++)
        std::memset(hsum(c), 0, static_cast<size_t>(N) * sizeof(float));
    }

    for (int li = 0; li < kNumLayers; li++)
      dispatch_layer(li, N);

    head_forward(N);

    const float* head_out = _head_out.data();
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);

    _stride = maxBufferSize;
    _layer_in.assign(static_cast<size_t>(C) * _stride, 0.0f);
    _head_sum.assign(static_cast<size_t>(C) * _stride, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_w.layers[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  float* lin(int c) { return _layer_in.data() + static_cast<size_t>(c) * _stride; }
  float* hsum(int c) { return _head_sum.data() + static_cast<size_t>(c) * _stride; }

  // --- arithmetic primitives -------------------------------------------------
  //
  // Every multiply-add in the conv goes through these two, so the exactness
  // switch is one decision in one place rather than a flag threaded through the
  // inner loops.

  /// acc + s * *w. Fused unless the kernel has opted out of bit-identity.
  static NB_A32_INLINE Vec acc_fma(Vec acc, Vec s, const float* w)
  {
    if constexpr (Opt::kFused)
      return VT::fma_bcast(acc, s, w);
    else
    {
      // Two roundings, the VMLA shape. round_now stops GCC contracting the pair
      // straight back into the vfma this candidate exists to avoid.
      const Vec p = VT::round_now(VT::mul_bcast(s, w));
      return VT::add(acc, p);
    }
  }

  /// Scalar counterpart, for the frames a tile does not cover.
  static NB_A32_INLINE float acc_fma(float acc, float s, float w)
  {
    if constexpr (Opt::kFused)
      return __builtin_fmaf(s, w, acc);
    else
      return acc + compat::round_now(s * w);
  }

  // --- layers ---------------------------------------------------------------

  void dispatch_layer(int li, int N)
  {
    const bool first = (li == 0);
    const bool last = (li == kNumLayers - 1);
    const bool do_l1x1 = !(last && Opt::kSkipLastL1x1);
    const bool store_head = first && kFoldHeadInit;

    // The two booleans are loop-invariant for the whole block, so resolving
    // them here keeps them out of the frame loop entirely.
    if (_w.layers[li].kernel_size == 6)
    {
      if (store_head)
        layer_forward<6, true, true>(li, N);
      else if (do_l1x1)
        layer_forward<6, false, true>(li, N);
      else
        layer_forward<6, false, false>(li, N);
    }
    else
    {
      if (store_head)
        layer_forward<15, true, true>(li, N);
      else if (do_l1x1)
        layer_forward<15, false, true>(li, N);
      else
        layer_forward<15, false, false>(li, N);
    }
  }

  template <int K, bool StoreHead, bool DoL1x1>
  void layer_forward(int li, int N)
  {
    Ring& R = _rings[li];
    const LayerWeights<C>& L = _w.layers[li];

    if constexpr (!Opt::kRingDirect)
    {
      R.prepare(N);
      for (int c = 0; c < C; c++)
        std::memcpy(R.write_ptr(c), lin(c), static_cast<size_t>(N) * sizeof(float));
      R.commit(N);
    }

    const float* h[C];
    for (int c = 0; c < C; c++)
      h[c] = R.plane(c);

    int tapb[K];
    for (int k = 0; k < K; k++)
      tapb[k] = R.tap((K - 1 - k) * L.dilation, N);

    // Residual destination: the next layer's ring under kRingDirect, otherwise
    // the scratch planes the next layer will copy in.
    Ring* next = nullptr;
    float* d[C];
    if constexpr (Opt::kRingDirect)
    {
      if (li + 1 < kNumLayers)
      {
        next = &_rings[li + 1];
        next->prepare(N);
        for (int c = 0; c < C; c++)
          d[c] = next->write_ptr(c);
      }
      else
      {
        for (int c = 0; c < C; c++)
          d[c] = lin(c);
      }
    }
    else
    {
      for (int c = 0; c < C; c++)
        d[c] = lin(c);
    }

    float* hs[C];
    for (int c = 0; c < C; c++)
      hs[c] = hsum(c);

    if constexpr (Opt::kPrefetch)
    {
      // The far-dilation layers reach thousands of frames back, so the next
      // block's taps have certainly fallen out of L1 by the time they are
      // wanted. Whether asking for them early helps is a property of this
      // core's prefetcher, which is why it is measured rather than assumed.
      if (L.dilation >= 41)
      {
        for (int k = 0; k < K; k++)
        {
          const int nb = R.wrap(tapb[k] + N);
          const int span = std::min(N, R.cap - nb);
          for (int c = 0; c < C; c++)
            for (int o = 0; o < span; o += 16)
              __builtin_prefetch(h[c] + nb + o);
        }
      }
    }

    if constexpr (Opt::kVextTaps && kLanes == 4)
    {
      // The vext path needs the K tap windows to be one contiguous span, which
      // they are unless this block straddled the ring wrap.
      const bool contiguous = (tapb[K - 1] - tapb[0]) == (K - 1) * L.dilation;
      if (contiguous && L.dilation == 1)
      {
        run_vext<K, 1, StoreHead, DoL1x1>(L, h, tapb[0], d, hs, N);
        finish_layer(next, N);
        return;
      }
      if (contiguous && L.dilation == 3)
      {
        run_vext<K, 3, StoreHead, DoL1x1>(L, h, tapb[0], d, hs, N);
        finish_layer(next, N);
        return;
      }
    }

    int f = 0;
    if constexpr (V > 1)
    {
      for (; f + T <= N; f += T)
        tile<K, V, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);
    }
    for (; f + kLanes <= N; f += kLanes)
      tile<K, 1, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);
    for (; f < N; f++)
      frame_scalar<K, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);

    finish_layer(next, N);
  }

  /// Under kRingDirect the residual was written into the next layer's ring, so
  /// that ring's write has to be committed once the layer is done.
  NB_A32_INLINE void finish_layer(Ring* next, int N)
  {
    if constexpr (Opt::kRingDirect)
    {
      if (next != nullptr)
        next->commit(N);
    }
    else
      (void)next, (void)N;
  }

  /// Seed the accumulators the way the reference this kernel is exact against
  /// starts its own reduction.
  template <int NVEC>
  NB_A32_INLINE void seed(const LayerWeights<C>& L, Vec (&z)[C][NVEC])
  {
    if constexpr (kPerTapPartial)
    {
      // Eigen: ztile.setZero(), then one partial per tap added in.
      for (int i = 0; i < C; i++)
        for (int v = 0; v < NVEC; v++)
          z[i][v] = VT::dup(0.0f);
    }
    else
    {
      // a2_fast's scalar branch seeds the chain with the bias, which is also
      // what saves it a zeroing pass.
      for (int i = 0; i < C; i++)
        for (int v = 0; v < NVEC; v++)
          z[i][v] = VT::dup(L.conv_b[i]);
    }
  }

  /// The conv, in the reference's reduction order, over whatever `load` hands
  /// back for (input channel, tap, vector).
  ///
  /// The ordering lives here once. Both the ordinary path and the vext path
  /// drive it, so the only thing the vext path changes is where the four frames
  /// came from — which is the whole claim it makes, and now the only claim it
  /// *can* make.
  template <int K, int NVEC, class Load>
  NB_A32_INLINE void conv(const LayerWeights<C>& L, Vec (&z)[C][NVEC], Load load)
  {
    auto tap = [&](auto kc) {
      constexpr int k = decltype(kc)::value;
      const float* wk = &L.conv_w[static_cast<size_t>(k) * C * C];

      if constexpr (kPerTapPartial)
      {
        // Eigen computes one tap's product on its own and adds it in, so the
        // taps do not share a chain.
        Vec t[C][NVEC];
        for (int i = 0; i < C; i++)
          for (int v = 0; v < NVEC; v++)
            t[i][v] = VT::dup(0.0f);

        for (int j = 0; j < C; j++)
          for (int v = 0; v < NVEC; v++)
          {
            const Vec sv = load(j, kc, v);
            for (int i = 0; i < C; i++)
              t[i][v] = acc_fma(t[i][v], sv, &wk[j * C + i]);
          }

        for (int i = 0; i < C; i++)
          for (int v = 0; v < NVEC; v++)
            z[i][v] = VT::add(z[i][v], t[i][v]);
      }
      else
      {
        for (int j = 0; j < C; j++)
          for (int v = 0; v < NVEC; v++)
          {
            const Vec sv = load(j, kc, v);
            for (int i = 0; i < C; i++)
              z[i][v] = acc_fma(z[i][v], sv, &wk[j * C + i]);
          }
      }
    };

    // Unrolled with the tap index as a compile-time value, because the vext
    // path's window offset is a template argument.
    [&]<int... Ks>(std::integer_sequence<int, Ks...>) {
      (tap(std::integral_constant<int, Ks>{}), ...);
    }(std::make_integer_sequence<int, K>{});
  }

  /// NVEC × 4 frames, loaded straight from the ring. z stays in registers
  /// across every tap; a weight is broadcast from memory at the point of use,
  /// which on a 16-register machine beats holding it (see a32_neon_compat.h).
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NB_A32_INLINE void tile(const LayerWeights<C>& L, const float* const* h, const int (&tapb)[K],
                          int f0, float* const* d, float* const* hs)
  {
    Vec z[C][NVEC];
    seed<NVEC>(L, z);
    conv<K, NVEC>(L, z, [&](int j, auto kc, int v) {
      return VT::load(h[j] + tapb[decltype(kc)::value] + f0 + kLanes * v);
    });
    post<K, NVEC, StoreHead, DoL1x1>(L, h, tapb[K - 1], f0, z, d, hs);
  }

  /// The vext variant: when the K tap windows all live inside one span — which
  /// they do at small dilations, unless this block straddled the ring wrap — a
  /// handful of loads plus vextq_f32 replaces K × C of them.
  ///
  /// Four frames at a time only. The window offset has to be a compile-time
  /// constant, and a wider tile would need the vector index folded into it as
  /// well, which multiplies the instantiations without answering a new
  /// question: what is under test here is loads, not tile width.
  template <int K, int D, bool StoreHead, bool DoL1x1>
  void run_vext(const LayerWeights<C>& L, const float* const* h, int base, float* const* d,
                float* const* hs, int N)
  {
    constexpr int kSpan = (K - 1) * D + 4;
    constexpr int kLoads = (kSpan + 3) / 4;

    int f = 0;
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t vec[C][kLoads];
      for (int j = 0; j < C; j++)
        for (int i = 0; i < kLoads; i++)
          vec[j][i] = vld1q_f32(h[j] + base + f + 4 * i);

      Vec z[C][1];
      seed<1>(L, z);
      conv<K, 1>(L, z, [&](int j, auto kc, int) {
        return planar_detail::window<decltype(kc)::value * D>(vec[j]);
      });
      post<K, 1, StoreHead, DoL1x1>(L, h, base + (K - 1) * D, f, z, d, hs);
    }

    if (f < N)
    {
      int tapb[K];
      for (int k = 0; k < K; k++)
        tapb[k] = base + k * D;
      for (; f < N; f++)
        frame_scalar<K, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);
    }
  }

  /// Everything after the conv: bias (StdEigen only — NanoScalar has it already),
  /// mixin, LeakyReLU, head_sum and the layer1x1 residual.
  ///
  /// `last_tap` is the base of the offset-0 tap, i.e. this block's own input to
  /// the layer. Reloading the residual from there is cheaper than carrying it
  /// through the tap loop, and it is what decides how wide a tile can usefully
  /// get.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NB_A32_INLINE void post(const LayerWeights<C>& L, const float* const* h, int last_tap, int f0,
                          Vec (&z)[C][NVEC], float* const* d, float* const* hs)
  {
    const Vec slope = VT::dup(kLeakySlope);
    const float* cond = _cond.data();

    for (int v = 0; v < NVEC; v++)
    {
      const Vec cf = VT::load(cond + f0 + kLanes * v);
      for (int i = 0; i < C; i++)
      {
        if constexpr (kPerTapPartial)
        {
          // Eigen rounds three times here and this has to round three times
          // too: z + bias, then the mixin product, then its add.
          Vec a = VT::add(z[i][v], VT::dup(L.conv_b[i]));
          const Vec m = VT::round_now(VT::mul_bcast(cf, &L.mixin_w[i]));
          z[i][v] = VT::add(a, m);
        }
        else
        {
          // a2_fast's scalar branch writes `a += mw * cf`, which the compiler
          // contracts: one rounding.
          z[i][v] = VT::fma_bcast(z[i][v], cf, &L.mixin_w[i]);
        }

        z[i][v] = VT::leaky(z[i][v], slope);
      }
    }

    for (int v = 0; v < NVEC; v++)
    {
      const int o = f0 + kLanes * v;
      for (int i = 0; i < C; i++)
      {
        if constexpr (StoreHead)
          VT::store(hs[i] + o, z[i][v]);
        else
          VT::store(hs[i] + o, VT::add(VT::load(hs[i] + o), z[i][v]));
      }
    }

    if constexpr (!DoL1x1)
      return;

    for (int v = 0; v < NVEC; v++)
    {
      const int o = f0 + kLanes * v;
      for (int i = 0; i < C; i++)
      {
        Vec acc;
        if constexpr (kPerTapPartial)
        {
          // Eigen: lin += l1x1 * z, then the bias as its own pass.
          acc = VT::dup(0.0f);
          for (int j = 0; j < C; j++)
            acc = VT::fma_bcast(acc, z[j][v], &L.l1x1_w[j * C + i]);
          acc = VT::add(VT::load(h[i] + last_tap + o), acc);
          acc = VT::add(acc, VT::dup(L.l1x1_b[i]));
        }
        else
        {
          // a2_fast: `lin += lb + w0*a0 + w1*a1 + w2*a2`, the bias seeding a
          // contracted chain, and the residual added once at the end.
          acc = VT::dup(L.l1x1_b[i]);
          for (int j = 0; j < C; j++)
            acc = VT::fma_bcast(acc, z[j][v], &L.l1x1_w[j * C + i]);
          acc = VT::add(VT::load(h[i] + last_tap + o), acc);
        }
        VT::store(d[i] + o, acc);
      }
    }
  }

  /// The frames of a block that are not a whole tile. Same operation order as
  /// the vector path, one frame at a time — a lane of the vector path *is* this
  /// computation, so the two agree bit for bit by construction.
  template <int K, bool StoreHead, bool DoL1x1>
  void frame_scalar(const LayerWeights<C>& L, const float* const* h, const int (&tapb)[K], int f,
                    float* const* d, float* const* hs)
  {
    float a[C];
    for (int i = 0; i < C; i++)
      a[i] = kPerTapPartial ? 0.0f : L.conv_b[i];

    for (int k = 0; k < K; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * C * C];
      if constexpr (kPerTapPartial)
      {
        float t[C];
        for (int i = 0; i < C; i++)
          t[i] = 0.0f;
        for (int j = 0; j < C; j++)
        {
          const float s = h[j][tapb[k] + f];
          for (int i = 0; i < C; i++)
            t[i] = acc_fma(t[i], s, wk[j * C + i]);
        }
        for (int i = 0; i < C; i++)
          a[i] += t[i];
      }
      else
      {
        for (int j = 0; j < C; j++)
        {
          const float s = h[j][tapb[k] + f];
          for (int i = 0; i < C; i++)
            a[i] = acc_fma(a[i], s, wk[j * C + i]);
        }
      }
    }

    const float cf = _cond[f];
    for (int i = 0; i < C; i++)
    {
      if constexpr (kPerTapPartial)
      {
        a[i] = a[i] + L.conv_b[i];
        a[i] = a[i] + compat::round_now(L.mixin_w[i] * cf);
      }
      else
        a[i] = __builtin_fmaf(L.mixin_w[i], cf, a[i]);

      a[i] = (a[i] < 0.0f) ? a[i] * kLeakySlope : a[i];

      if constexpr (StoreHead)
        hs[i][f] = a[i];
      else
        hs[i][f] += a[i];
    }

    if constexpr (DoL1x1)
    {
      for (int i = 0; i < C; i++)
      {
        float o;
        if constexpr (kPerTapPartial)
        {
          o = 0.0f;
          for (int j = 0; j < C; j++)
            o = __builtin_fmaf(L.l1x1_w[j * C + i], a[j], o);
          o = (h[i][tapb[K - 1] + f] + o) + L.l1x1_b[i];
        }
        else
        {
          o = L.l1x1_b[i];
          for (int j = 0; j < C; j++)
            o = __builtin_fmaf(L.l1x1_w[j * C + i], a[j], o);
          o = h[i][tapb[K - 1] + f] + o;
        }
        d[i][f] = o;
      }
    }
  }

  /// Head rechannel: K=16, dilation 1, C channels down to one, plus bias and
  /// scale. Both baselines run this as a scalar chain per frame — 48 FMAs at
  /// C=3, 128 at C=8 — and in planar layout the identical chain runs four (or
  /// kHeadTile) frames at a time, one per lane.
  void head_forward(int N)
  {
    _head_ring.prepare(N);
    for (int c = 0; c < C; c++)
      std::memcpy(_head_ring.write_ptr(c), hsum(c), static_cast<size_t>(N) * sizeof(float));
    _head_ring.commit(N);

    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* p[C];
    for (int c = 0; c < C; c++)
      p[c] = _head_ring.plane(c);

    const float scale = _w.head_scale;
    float* out = _head_out.data();

    int f = 0;
    if constexpr (HV > 1)
    {
      for (; f + 4 * HV <= N; f += 4 * HV)
        head_tile<HV>(p, hb, f, out, scale);
    }
    for (; f + 4 <= N; f += 4)
      head_tile<1>(p, hb, f, out, scale);
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
        for (int j = 0; j < C; j++)
          y = __builtin_fmaf(_w.head_w[k][j], p[j][hb[k] + f], y);
      out[f] = y * scale;
    }
  }

  template <int NVEC>
  NB_A32_INLINE void head_tile(const float* const* p, const int (&hb)[kHeadKernelSize], int f0,
                               float* out, float scale)
  {
    float32x4_t y[NVEC];
    for (int v = 0; v < NVEC; v++)
      y[v] = vdupq_n_f32(_w.head_b);

    for (int k = 0; k < kHeadKernelSize; k++)
    {
      for (int j = 0; j < C; j++)
      {
        for (int v = 0; v < NVEC; v++)
          y[v] = compat::fmaq_bcast(y[v], vld1q_f32(p[j] + hb[k] + f0 + 4 * v), &_w.head_w[k][j]);
      }
    }

    for (int v = 0; v < NVEC; v++)
      vst1q_f32(out + f0 + 4 * v, vmulq_n_f32(y[v], scale));
  }

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
  int _stride = 0;
};

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
