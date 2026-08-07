// The planar (structure-of-arrays) engine, and the switches layered on top of
// it. One implementation, several candidates: each candidate .cpp instantiates
// this template with a different option set, so what is being compared really
// is one idea at a time rather than one rewrite at a time.
//
// The core idea. a2_fast keeps three channels interleaved and vectorises across
// *channels*, which at C=3 wastes a quarter of every lane before it starts. The
// planar layout keeps three separate channel planes and vectorises across
// *frames* instead, so a NEON register holds four consecutive frames of one
// channel and every lane does real work:
//
//   per conv tap:  3 loads + 9 vfmaq_laneq_f32 per 4 frames  = 2.25 FMA/frame
//   pad4:                                                      3    FMA/frame
//   a2_fast:       9 scalar FMA per frame                    = 9    FMA/frame
//
// and the z accumulator lives in twelve registers across all K taps rather than
// making 30 round-trips to memory per frame per K=6 layer.
//
// It is also bit-exact against a2_fast for the layer stack. The weight in a
// planar FMA is a scalar and the vector is four frames, so the reduction order
// per output channel — bias, then tap 0 inputs 0,1,2, then tap 1, and so on —
// is *identical* to a2_fast's scalar chain, one lane per frame. Nothing is
// reassociated, which is why the parity number for this family is the
// interesting one.

#pragma once

#if defined(NB_ENABLE_SLIM_LAB)

  #include <algorithm>
  #include <cstring>
  #include <utility>
  #include <vector>

  #include "slim_common.h"

  #if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
  #else
    #error "the planar kernels are NEON; this lab is arm64-only"
  #endif

namespace slimlab
{

/// How the history rings are sized and kept readable.
enum class RingKind
{
  /// Power-of-two capacity, tail mirrored on every write. a2_fast's default
  /// (NAM_A2_RING_MODE=1): masked reads, constant work per block, and about
  /// 18 KB of memcpy per 64-frame block that usually nothing reads.
  Pow2Eager,
  /// Exactly-sized capacity, mirrored only when a read genuinely wraps. What
  /// the fused engine does. Smallest footprint.
  ExactLazy,
  /// One linear buffer per ring, written forward until it runs out and then
  /// memmoved back. a2_fast's NAM_A2_RING_MODE=0. No mirror at all, no masking,
  /// every read contiguous — at the price of an occasional large memmove.
  LinearRewind,
};

/// The option set every candidate in this family varies.
///
/// Candidates inherit from this and override the members they change, so a
/// candidate .cpp reads as a diff against `planar`.
struct PlanarOptions
{
  /// Frames per tile. 4 is one NEON register per channel; higher trades
  /// register pressure for independent FMA chains.
  static constexpr int kTile = 4;
  /// Which ring strategy to use. The default matches a2_fast's default.
  static constexpr RingKind kRing = RingKind::Pow2Eager;
  /// Write each layer's residual straight into the next layer's ring instead
  /// of into a scratch buffer that is then copied in.
  static constexpr bool kRingDirect = false;
  /// Skip the last layer's layer1x1 (its residual is never read) and fold the
  /// first layer's head_sum accumulate into a store (removing the memset).
  static constexpr bool kSkipLastL1x1 = false;
  /// Prefetch the next block's taps on the far-dilation layers.
  static constexpr bool kPrefetch = false;
  /// On layers whose taps overlap heavily (dilation 1 and 3), build the tap
  /// windows with vextq_f32 from a smaller number of loads.
  static constexpr bool kVextTaps = false;
};

namespace planar_detail
{

/// Extract the 4-frame window starting OFF frames into a loaded span.
template <int OFF, int NW>
inline float32x4_t window(const float32x4_t (&v)[NW])
{
  constexpr int I = OFF / 4;
  constexpr int L = OFF % 4;
  if constexpr (L == 0)
    return v[I];
  else
    return vextq_f32(v[I], v[I + 1], L);
}

} // namespace planar_detail

/// Planar ring buffer, three channel planes side by side.
///
/// In the two wrapping modes a write is always one contiguous run at `wpos`:
/// the mirror region past the ring proper gives at least `mbs` columns of
/// slack, so the run can overhang the end and be folded back afterwards. That
/// keeps the frame loop straight-line even when a block straddles the wrap,
/// which is what makes writing residuals straight into the ring (kRingDirect)
/// practical at all. In LinearRewind mode there is no wrap to straddle: the
/// buffer is rewound *before* the write if the write would not fit.
///
/// Whichever mode, the contract is the same three calls in the same order:
/// prepare(n), write into write_ptr(c), commit(n).
template <RingKind Kind, int TailPad>
struct PlanarRing
{
  std::vector<float> data;
  int cap = 0; ///< usable columns (LinearRewind: total columns)
  int mask = 0; ///< cap - 1, only meaningful for Pow2Eager
  int stride = 0; ///< distance between channel planes
  int wpos = 0;
  int mbs = 0;
  int lookback = 0;

  void reset(int max_lookback, int max_buffer)
  {
    mbs = max_buffer;
    lookback = max_lookback;
    switch (Kind)
    {
      case RingKind::Pow2Eager:
        cap = next_pow2(max_lookback + max_buffer);
        stride = cap + max_buffer + TailPad;
        break;
      case RingKind::ExactLazy:
        cap = max_lookback + max_buffer;
        stride = cap + max_buffer + TailPad;
        break;
      case RingKind::LinearRewind:
        // Matches a2_fast's ring mode 0 sizing: enough slack that the rewind
        // memmove is amortised over many blocks on the long-lookback layers.
        cap = 2 * max_lookback + max_buffer;
        stride = cap + TailPad;
        break;
    }
    mask = cap - 1;
    data.assign(static_cast<size_t>(kChannels) * stride, 0.0f);
    wpos = max_lookback;
  }

  float* plane(int c) { return data.data() + static_cast<size_t>(c) * stride; }
  const float* plane(int c) const { return data.data() + static_cast<size_t>(c) * stride; }

  int wrap(int v) const
  {
    if constexpr (Kind == RingKind::Pow2Eager)
      return v & mask;
    else if constexpr (Kind == RingKind::ExactLazy)
    {
      if (v >= cap)
        v -= cap;
      if (v < 0)
        v += cap;
      return v;
    }
    else
      return v; // LinearRewind: positions are monotonic, never wrapped
  }

  /// Make room for an n-frame write. Only LinearRewind has anything to do.
  void prepare(int n)
  {
    if constexpr (Kind == RingKind::LinearRewind)
    {
      if (wpos + n > cap)
      {
        for (int c = 0; c < kChannels; c++)
        {
          std::memmove(plane(c), plane(c) + (wpos - lookback), static_cast<size_t>(lookback) * sizeof(float));
        }
        wpos = lookback;
      }
    }
  }

  /// Where an n-frame block is written. Contiguous; see the note above.
  float* write_ptr(int c) { return plane(c) + wpos; }

  /// Fold any overhang back into the ring, refresh the eager mirror, advance.
  void commit(int n)
  {
    if constexpr (Kind != RingKind::LinearRewind)
    {
      const int overflow = wpos + n - cap;
      if (overflow > 0)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c), plane(c) + cap, static_cast<size_t>(overflow) * sizeof(float));
      }
      if constexpr (Kind == RingKind::Pow2Eager)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(mbs) * sizeof(float));
      }
      wpos = wrap(wpos + n);
    }
    else
      wpos += n;
  }

  /// First column of an n-frame read looking `lookback` frames further back
  /// than the block just written.
  int tap(int lookback_frames, int n)
  {
    const int base = wrap(wpos - n - lookback_frames);
    if constexpr (Kind == RingKind::ExactLazy)
    {
      // Lazy mirror: only the blocks whose read actually wraps pay for it.
      const int overflow = base + n - cap;
      if (overflow > 0)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(overflow) * sizeof(float));
      }
    }
    return base;
  }
};

/// One layer's weights, padded to four lanes so each group can be loaded with
/// a single vld1q and addressed by lane.
struct PlanarLayer
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0;
  /// kernel_size × 12 floats: nine weights then three pad, per tap.
  std::vector<float> conv_w;
  std::array<float, 4> conv_b{};
  std::array<float, 4> mixin_w{};
  /// Twelve floats: nine layer1x1 weights then three pad.
  std::array<float, 12> l1x1_w{};
  std::array<float, 4> l1x1_b{};
};

template <class Opt>
class PlanarModel : public SlimModel
{
  static constexpr int kTile = Opt::kTile;
  static constexpr int kVecs = kTile / 4;
  static constexpr int kTailPad = Opt::kVextTaps ? 8 : 0;
  /// Under kSkipLastL1x1 the first layer stores into head_sum rather than
  /// accumulating, which is what makes the per-block memset unnecessary.
  static constexpr bool kFoldHeadInit = Opt::kSkipLastL1x1;

  using Ring = PlanarRing<Opt::kRing, kTailPad>;

public:
  PlanarModel(const std::vector<float>& weights, double sampleRate)
  : SlimModel(weights, sampleRate)
  {
    for (int li = 0; li < kNumLayers; li++)
    {
      const LayerWeights& L = _w.layers[li];
      PlanarLayer& P = _p[li];
      P.kernel_size = L.kernel_size;
      P.dilation = L.dilation;
      P.max_lookback = L.max_lookback;

      P.conv_w.assign(static_cast<size_t>(L.kernel_size) * 12, 0.0f);
      for (int k = 0; k < L.kernel_size; k++)
        for (int e = 0; e < 9; e++)
          P.conv_w[static_cast<size_t>(k) * 12 + e] = L.conv_w[static_cast<size_t>(k) * 9 + e];

      for (int i = 0; i < kChannels; i++)
      {
        P.conv_b[i] = L.conv_b[i];
        P.mixin_w[i] = L.mixin_w[i];
        P.l1x1_b[i] = L.l1x1_b[i];
      }
      for (int e = 0; e < 9; e++)
        P.l1x1_w[e] = L.l1x1_w[e];
    }

    _head_w4.assign(static_cast<size_t>(kHeadKernelSize) * 4, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int j = 0; j < kChannels; j++)
        _head_w4[static_cast<size_t>(k) * 4 + j] = _w.head_w[k][j];
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    // Rechannel + float copy of the condition signal. With kRingDirect this
    // lands straight in layer 0's ring; otherwise in the scratch planes.
    float* cond = _cond.data();
    float* r0;
    float* r1;
    float* r2;
    if constexpr (Opt::kRingDirect)
    {
      _rings[0].prepare(N);
      r0 = _rings[0].write_ptr(0);
      r1 = _rings[0].write_ptr(1);
      r2 = _rings[0].write_ptr(2);
    }
    else
    {
      r0 = lin(0);
      r1 = lin(1);
      r2 = lin(2);
    }
    const float rw0 = _w.rechannel_w[0], rw1 = _w.rechannel_w[1], rw2 = _w.rechannel_w[2];
    for (int f = 0; f < N; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      r0[f] = rw0 * x;
      r1[f] = rw1 * x;
      r2[f] = rw2 * x;
    }
    if constexpr (Opt::kRingDirect)
      _rings[0].commit(N);

    if constexpr (!kFoldHeadInit)
    {
      for (int c = 0; c < kChannels; c++)
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

    _lin_stride = maxBufferSize;
    _hs_stride = maxBufferSize;
    _layer_in.assign(static_cast<size_t>(kChannels) * _lin_stride, 0.0f);
    _head_sum.assign(static_cast<size_t>(kChannels) * _hs_stride, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_p[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  float* lin(int c) { return _layer_in.data() + static_cast<size_t>(c) * _lin_stride; }
  float* hsum(int c) { return _head_sum.data() + static_cast<size_t>(c) * _hs_stride; }

  void dispatch_layer(int li, int N)
  {
    const bool first = (li == 0);
    const bool last = (li == kNumLayers - 1);
    const bool do_l1x1 = !(last && Opt::kSkipLastL1x1);
    const bool store_head = first && kFoldHeadInit;

    // Four compile-time combinations; the booleans are loop-invariant for the
    // whole block, so none of this reaches the inner loop.
    if (_p[li].kernel_size == 6)
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
    const PlanarLayer& P = _p[li];

    if constexpr (!Opt::kRingDirect)
    {
      R.prepare(N);
      for (int c = 0; c < kChannels; c++)
        std::memcpy(R.write_ptr(c), lin(c), static_cast<size_t>(N) * sizeof(float));
      R.commit(N);
    }

    const float* h[kChannels] = {R.plane(0), R.plane(1), R.plane(2)};
    int tapb[K];
    for (int k = 0; k < K; k++)
      tapb[k] = R.tap((K - 1 - k) * P.dilation, N);

    if constexpr (Opt::kPrefetch)
    {
      // The far-dilation layers reach ~14 KB back, so the next block's taps are
      // certain to have fallen out of L1 by the time they are wanted.
      if (P.dilation >= 41)
      {
        for (int k = 0; k < K; k++)
        {
          const int nb = R.wrap(tapb[k] + N);
          const int span = std::min(N, R.cap - nb);
          for (int c = 0; c < kChannels; c++)
            for (int o = 0; o < span; o += 16)
              __builtin_prefetch(h[c] + nb + o);
        }
      }
    }

    // Residual destination: the next layer's ring under kRingDirect, otherwise
    // the scratch planes that the next layer will copy in.
    Ring* next = nullptr;
    float* d[kChannels];
    if constexpr (Opt::kRingDirect)
    {
      if (li + 1 < kNumLayers)
      {
        next = &_rings[li + 1];
        next->prepare(N);
        for (int c = 0; c < kChannels; c++)
          d[c] = next->write_ptr(c);
      }
      else
      {
        for (int c = 0; c < kChannels; c++)
          d[c] = lin(c);
      }
    }
    else
    {
      for (int c = 0; c < kChannels; c++)
        d[c] = lin(c);
    }

    float* hs[kChannels] = {hsum(0), hsum(1), hsum(2)};

    bool done = false;
    if constexpr (Opt::kVextTaps)
    {
      // The vext path needs the K tap windows to be one contiguous span, which
      // they are unless this block straddled the ring wrap.
      const bool contiguous = (tapb[K - 1] - tapb[0]) == (K - 1) * P.dilation;
      if (contiguous && P.dilation == 1)
      {
        run_vext<K, 1, StoreHead, DoL1x1>(P, h, tapb[0], d, hs, N);
        done = true;
      }
      else if (contiguous && P.dilation == 3)
      {
        run_vext<K, 3, StoreHead, DoL1x1>(P, h, tapb[0], d, hs, N);
        done = true;
      }
    }

    if (!done)
    {
      int f = 0;
      if constexpr (kVecs > 1)
      {
        for (; f + kTile <= N; f += kTile)
          tile<K, kVecs, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
      }
      for (; f + 4 <= N; f += 4)
        tile<K, 1, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
      for (; f < N; f++)
        frame_scalar<K, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
    }

    if constexpr (Opt::kRingDirect)
    {
      if (next != nullptr)
        next->commit(N);
    }
  }

  /// NVEC × 4 frames, three channel accumulators per vector. z stays in
  /// registers across every tap.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  inline void tile(const PlanarLayer& P, const float* const* h, const int (&tapb)[K], int f0, float* const* d,
                   float* const* hs)
  {
    const float32x4_t cb = vld1q_f32(P.conv_b.data());
    float32x4_t a0[NVEC], a1[NVEC], a2[NVEC];
    for (int v = 0; v < NVEC; v++)
    {
      a0[v] = vdupq_laneq_f32(cb, 0);
      a1[v] = vdupq_laneq_f32(cb, 1);
      a2[v] = vdupq_laneq_f32(cb, 2);
    }

    const float* cw = P.conv_w.data();
    for (int k = 0; k < K; k++)
    {
      const float* wk = cw + static_cast<size_t>(k) * 12;
      const float32x4_t A = vld1q_f32(wk); // w0 w1 w2 w3
      const float32x4_t B = vld1q_f32(wk + 4); // w4 w5 w6 w7
      const float32x4_t C = vld1q_f32(wk + 8); // w8 . . .
      const int b = tapb[k] + f0;
      for (int v = 0; v < NVEC; v++)
      {
        const float32x4_t s0 = vld1q_f32(h[0] + b + 4 * v);
        const float32x4_t s1 = vld1q_f32(h[1] + b + 4 * v);
        const float32x4_t s2 = vld1q_f32(h[2] + b + 4 * v);
        a0[v] = vfmaq_laneq_f32(a0[v], s0, A, 0);
        a1[v] = vfmaq_laneq_f32(a1[v], s0, A, 1);
        a2[v] = vfmaq_laneq_f32(a2[v], s0, A, 2);
        a0[v] = vfmaq_laneq_f32(a0[v], s1, A, 3);
        a1[v] = vfmaq_laneq_f32(a1[v], s1, B, 0);
        a2[v] = vfmaq_laneq_f32(a2[v], s1, B, 1);
        a0[v] = vfmaq_laneq_f32(a0[v], s2, B, 2);
        a1[v] = vfmaq_laneq_f32(a1[v], s2, B, 3);
        a2[v] = vfmaq_laneq_f32(a2[v], s2, C, 0);
      }
    }

    post<K, NVEC, StoreHead, DoL1x1>(P, h, tapb[K - 1], f0, a0, a1, a2, d, hs);
  }

  /// The vext variant: for small dilations the K tap windows all live inside
  /// one span, so a handful of loads plus vextq_f32 replaces K × 3 loads.
  template <int K, int D, bool StoreHead, bool DoL1x1>
  void run_vext(const PlanarLayer& P, const float* const* h, int base, float* const* d, float* const* hs, int N)
  {
    constexpr int kSpan = (K - 1) * D + 4;
    constexpr int kLoads = (kSpan + 3) / 4;

    const float32x4_t cb = vld1q_f32(P.conv_b.data());
    const float* cw = P.conv_w.data();

    int f = 0;
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t v0[kLoads], v1[kLoads], v2[kLoads];
      for (int i = 0; i < kLoads; i++)
      {
        v0[i] = vld1q_f32(h[0] + base + f + 4 * i);
        v1[i] = vld1q_f32(h[1] + base + f + 4 * i);
        v2[i] = vld1q_f32(h[2] + base + f + 4 * i);
      }

      float32x4_t a0[1] = {vdupq_laneq_f32(cb, 0)};
      float32x4_t a1[1] = {vdupq_laneq_f32(cb, 1)};
      float32x4_t a2[1] = {vdupq_laneq_f32(cb, 2)};

      auto tap = [&](auto kc) {
        constexpr int k = decltype(kc)::value;
        const float* wk = cw + static_cast<size_t>(k) * 12;
        const float32x4_t A = vld1q_f32(wk);
        const float32x4_t B = vld1q_f32(wk + 4);
        const float32x4_t C = vld1q_f32(wk + 8);
        const float32x4_t s0 = planar_detail::window<k * D>(v0);
        const float32x4_t s1 = planar_detail::window<k * D>(v1);
        const float32x4_t s2 = planar_detail::window<k * D>(v2);
        a0[0] = vfmaq_laneq_f32(a0[0], s0, A, 0);
        a1[0] = vfmaq_laneq_f32(a1[0], s0, A, 1);
        a2[0] = vfmaq_laneq_f32(a2[0], s0, A, 2);
        a0[0] = vfmaq_laneq_f32(a0[0], s1, A, 3);
        a1[0] = vfmaq_laneq_f32(a1[0], s1, B, 0);
        a2[0] = vfmaq_laneq_f32(a2[0], s1, B, 1);
        a0[0] = vfmaq_laneq_f32(a0[0], s2, B, 2);
        a1[0] = vfmaq_laneq_f32(a1[0], s2, B, 3);
        a2[0] = vfmaq_laneq_f32(a2[0], s2, C, 0);
      };
      [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (tap(std::integral_constant<int, Is>{}), ...);
      }(std::make_integer_sequence<int, K>{});

      post<K, 1, StoreHead, DoL1x1>(P, h, base + (K - 1) * D, f, a0, a1, a2, d, hs);
    }

    // Whatever is left over runs the ordinary way.
    if (f < N)
    {
      int tapb[K];
      for (int k = 0; k < K; k++)
        tapb[k] = base + k * D;
      for (; f < N; f++)
        frame_scalar<K, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
    }
  }

  /// Everything after the conv: mixin, LeakyReLU, head_sum, layer1x1 residual.
  /// `last_tap` is the base of the offset-0 tap, i.e. this block's own input —
  /// reloading it here is cheaper than carrying it through the tap loop in
  /// registers, which is what decides how wide a tile can usefully get.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  inline void post(const PlanarLayer& P, const float* const* h, int last_tap, int f0, float32x4_t (&a0)[NVEC],
                   float32x4_t (&a1)[NVEC], float32x4_t (&a2)[NVEC], float* const* d, float* const* hs)
  {
    const float32x4_t M = vld1q_f32(P.mixin_w.data());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float* cond = _cond.data();

    for (int v = 0; v < NVEC; v++)
    {
      const float32x4_t cf = vld1q_f32(cond + f0 + 4 * v);
      a0[v] = vfmaq_laneq_f32(a0[v], cf, M, 0);
      a1[v] = vfmaq_laneq_f32(a1[v], cf, M, 1);
      a2[v] = vfmaq_laneq_f32(a2[v], cf, M, 2);
      a0[v] = vbslq_f32(vcltq_f32(a0[v], zero), vmulq_f32(a0[v], slope), a0[v]);
      a1[v] = vbslq_f32(vcltq_f32(a1[v], zero), vmulq_f32(a1[v], slope), a1[v]);
      a2[v] = vbslq_f32(vcltq_f32(a2[v], zero), vmulq_f32(a2[v], slope), a2[v]);
    }

    for (int v = 0; v < NVEC; v++)
    {
      const int o = f0 + 4 * v;
      if constexpr (StoreHead)
      {
        vst1q_f32(hs[0] + o, a0[v]);
        vst1q_f32(hs[1] + o, a1[v]);
        vst1q_f32(hs[2] + o, a2[v]);
      }
      else
      {
        vst1q_f32(hs[0] + o, vaddq_f32(vld1q_f32(hs[0] + o), a0[v]));
        vst1q_f32(hs[1] + o, vaddq_f32(vld1q_f32(hs[1] + o), a1[v]));
        vst1q_f32(hs[2] + o, vaddq_f32(vld1q_f32(hs[2] + o), a2[v]));
      }
    }

    if constexpr (!DoL1x1)
      return;

    const float32x4_t LA = vld1q_f32(P.l1x1_w.data()); // l0 l1 l2 l3
    const float32x4_t LB = vld1q_f32(P.l1x1_w.data() + 4); // l4 l5 l6 l7
    const float32x4_t LC = vld1q_f32(P.l1x1_w.data() + 8); // l8 . . .
    const float32x4_t LBias = vld1q_f32(P.l1x1_b.data());

    for (int v = 0; v < NVEC; v++)
    {
      const int o = f0 + 4 * v;
      float32x4_t o0 = vdupq_laneq_f32(LBias, 0);
      float32x4_t o1 = vdupq_laneq_f32(LBias, 1);
      float32x4_t o2 = vdupq_laneq_f32(LBias, 2);
      o0 = vfmaq_laneq_f32(o0, a0[v], LA, 0);
      o0 = vfmaq_laneq_f32(o0, a1[v], LA, 3);
      o0 = vfmaq_laneq_f32(o0, a2[v], LB, 2);
      o1 = vfmaq_laneq_f32(o1, a0[v], LA, 1);
      o1 = vfmaq_laneq_f32(o1, a1[v], LB, 0);
      o1 = vfmaq_laneq_f32(o1, a2[v], LB, 3);
      o2 = vfmaq_laneq_f32(o2, a0[v], LA, 2);
      o2 = vfmaq_laneq_f32(o2, a1[v], LB, 1);
      o2 = vfmaq_laneq_f32(o2, a2[v], LC, 0);
      vst1q_f32(d[0] + o, vaddq_f32(vld1q_f32(h[0] + last_tap + o), o0));
      vst1q_f32(d[1] + o, vaddq_f32(vld1q_f32(h[1] + last_tap + o), o1));
      vst1q_f32(d[2] + o, vaddq_f32(vld1q_f32(h[2] + last_tap + o), o2));
    }
  }

  /// The last few frames of a block that is not a multiple of four. Same
  /// operation order as the vector path, one frame at a time.
  template <int K, bool StoreHead, bool DoL1x1>
  void frame_scalar(const PlanarLayer& P, const float* const* h, const int (&tapb)[K], int f, float* const* d,
                    float* const* hs)
  {
    float a[3] = {P.conv_b[0], P.conv_b[1], P.conv_b[2]};
    for (int k = 0; k < K; k++)
    {
      const float* wk = P.conv_w.data() + static_cast<size_t>(k) * 12;
      const float s0 = h[0][tapb[k] + f];
      const float s1 = h[1][tapb[k] + f];
      const float s2 = h[2][tapb[k] + f];
      a[0] += wk[0] * s0;
      a[1] += wk[1] * s0;
      a[2] += wk[2] * s0;
      a[0] += wk[3] * s1;
      a[1] += wk[4] * s1;
      a[2] += wk[5] * s1;
      a[0] += wk[6] * s2;
      a[1] += wk[7] * s2;
      a[2] += wk[8] * s2;
    }

    const float cf = _cond[f];
    for (int c = 0; c < kChannels; c++)
    {
      a[c] += P.mixin_w[c] * cf;
      a[c] = (a[c] < 0.0f) ? a[c] * kLeakySlope : a[c];
      if constexpr (StoreHead)
        hs[c][f] = a[c];
      else
        hs[c][f] += a[c];
    }

    if constexpr (DoL1x1)
    {
      for (int c = 0; c < kChannels; c++)
      {
        float o = P.l1x1_b[c];
        o += P.l1x1_w[0 + c] * a[0];
        o += P.l1x1_w[3 + c] * a[1];
        o += P.l1x1_w[6 + c] * a[2];
        d[c][f] = h[c][tapb[K - 1] + f] + o;
      }
    }
  }

  /// Head rechannel: K=16, dilation 1, three channels down to one, plus bias
  /// and scale. In planar layout this is 48 vector FMAs per four frames where
  /// a2_fast does 48 scalar FMAs per frame.
  void head_forward(int N)
  {
    _head_ring.prepare(N);
    for (int c = 0; c < kChannels; c++)
      std::memcpy(_head_ring.write_ptr(c), hsum(c), static_cast<size_t>(N) * sizeof(float));
    _head_ring.commit(N);

    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* p[kChannels] = {_head_ring.plane(0), _head_ring.plane(1), _head_ring.plane(2)};
    const float* hw = _head_w4.data();
    const float scale = _w.head_scale;
    float* out = _head_out.data();

    int f = 0;
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t y = vdupq_n_f32(_w.head_b);
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float32x4_t W = vld1q_f32(hw + static_cast<size_t>(k) * 4);
        y = vfmaq_laneq_f32(y, vld1q_f32(p[0] + hb[k] + f), W, 0);
        y = vfmaq_laneq_f32(y, vld1q_f32(p[1] + hb[k] + f), W, 1);
        y = vfmaq_laneq_f32(y, vld1q_f32(p[2] + hb[k] + f), W, 2);
      }
      vst1q_f32(out + f, vmulq_n_f32(y, scale));
    }
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* w = hw + static_cast<size_t>(k) * 4;
        y += w[0] * p[0][hb[k] + f];
        y += w[1] * p[1][hb[k] + f];
        y += w[2] * p[2][hb[k] + f];
      }
      out[f] = y * scale;
    }
  }

  std::array<PlanarLayer, kNumLayers> _p;
  std::vector<float> _head_w4;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
  int _lin_stride = 0;
  int _hs_stride = 0;
};

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
