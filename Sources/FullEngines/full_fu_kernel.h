// The fu* family: fused's channel-major structure, with the levers the nano lab
// found, keeping fused's arithmetic bit-for-bit.
//
// Everything here vectorises across *channels* — a register holds four channels
// of one frame, and `vfmaq_laneq_f32` broadcasts one input lane against a column
// of weights. That is fused's shape, and it fixes the association: the
// accumulator is seeded with the bias and every tap and every input channel
// folds into the same chain. It is a different association from a2_fast's, which
// is where fused's 132.6 dB comes from, and it is preserved exactly here so the
// fu* family stays bit-identical to `fused`.
//
// What varies:
//
//   kConvTile   fused hardcodes `(Q <= 4) ? 4 : 2`, so C=8 gets 4 frames and
//               8 accumulator chains where four 4-cycle FMA pipes want ~16.
//   kTailTile   fused hardcodes `(Q <= 4) ? 2 : 1`, so C=8 gets 2 frames and
//               *4* chains, plus 16 layer1x1 weight vectors reloaded per pair.
//   kFuseZ      fused runs conv, activation and tail as three passes over the
//               _z buffer. Fusing them keeps z in registers.
//   kRingDirect residual straight into the next layer's ring.
//   kHeadFrames fused's head runs 32 FMAs into a single accumulator per frame.
//   kRing       fused uses pow2 + lazy mirror; the sweep says that is not the
//               best choice at C=8.

#pragma once

#if defined(NB_ENABLE_FULL_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "full_common.h"
  #include "full_ring.h"

  #if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
  #else
    #error "the full-path kernels are NEON; this lab is arm64-only"
  #endif

namespace fulllab
{

struct FuOptions
{
  /// Frames per conv tile. fused's value at C=8 is 4.
  static constexpr int kConvTile = 4;
  /// Frames per tail tile. fused's value at C=8 is 2. Ignored under kFuseZ,
  /// which necessarily shares the conv's tile.
  static constexpr int kTailTile = 2;
  /// Frames per head iteration, i.e. how many of the 32-deep accumulator chains
  /// run at once. fused's value is 1.
  static constexpr int kHeadFrames = 1;
  /// Run conv, mixin, activation, head_sum and layer1x1 as one pass with z in
  /// registers, instead of fused's three passes through the _z buffer.
  static constexpr bool kFuseZ = false;
  /// Write each layer's residual straight into the next layer's ring.
  static constexpr bool kRingDirect = false;
  /// Fold the per-block memset of head_sum into the first layer's accumulate.
  static constexpr bool kFoldHeadInit = false;
  /// Ring strategy. fused's is Pow2Lazy.
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

template <class Opt>
class FuModel : public FullModel
{
  static constexpr int C = kChannels;
  static constexpr int Q = C / 4;
  static constexpr int kConvTile = Opt::kConvTile;
  static constexpr int kTailTile = Opt::kTailTile;
  static constexpr int kHeadFrames = Opt::kHeadFrames;

  using Ring = ChannelRing<Opt::kRing>;

  static_assert(kConvTile >= 1 && kTailTile >= 1 && kHeadFrames >= 1, "tiles must be positive");

public:
  FuModel(const std::vector<float>& weights, double sampleRate)
  : FullModel(weights, sampleRate)
  {
    _head_w.assign(static_cast<size_t>(kHeadKernelSize) * C, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int b = 0; b < C; b++)
        _head_w[static_cast<size_t>(k) * C + b] = _w.head_w[k][b];
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    float* cond = _cond.data();
    for (int f = 0; f < N; f++)
      cond[f] = static_cast<float>(in0[f]);

    if constexpr (Opt::kRingDirect)
      _rings[0].prepare(N);
    float* first_in = Opt::kRingDirect ? _rings[0].write_ptr() : _lin.data();
    rechannel_block(_w.rechannel_w.data(), cond, first_in, N);
    if constexpr (Opt::kRingDirect)
      _rings[0].commit(N);

    if constexpr (Opt::kRingDirect)
    {
      _head_ring.prepare(N);
      _hs = _head_ring.write_ptr();
    }
    else
    {
      _hs = _head_sum.data();
    }

    if constexpr (!Opt::kFoldHeadInit)
      std::memset(_hs, 0, static_cast<size_t>(C) * N * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
      dispatch_layer(li, N);

    if constexpr (Opt::kRingDirect)
      _head_ring.commit(N);

    head_forward(N);

    const float scale = _w.head_scale;
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(scale * _head_out[f]);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_w.layers[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);

    const size_t buf = static_cast<size_t>(C) * maxBufferSize;
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _lin.assign(buf, 0.0f);
    _z.assign(buf, 0.0f);
    _head_sum.assign(buf, 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);
  }

private:
  // --- conv ------------------------------------------------------------------

  template <int T, int LANE>
  static inline void conv_lane(float32x4_t (&acc)[Q][T], const float32x4_t (&iv)[T], const float* wc)
  {
    float32x4_t w[Q];
    for (int q = 0; q < Q; q++)
      w[q] = vld1q_f32(wc + LANE * C + 4 * q);
    for (int t = 0; t < T; t++)
      for (int q = 0; q < Q; q++)
        acc[q][t] = vfmaq_laneq_f32(acc[q][t], w[q], iv[t], LANE);
  }

  /// acc = bias; for each tap, acc += W_k * x_k; then acc += mixin * cond.
  /// One chain per output element across every tap and input channel — fused's
  /// association, and the reason the fu* family is not bit-identical to a2_fast.
  template <int T>
  static inline void conv_acc(float32x4_t (&acc)[Q][T], const float* const* tap_src, int num_taps, const float* W,
                              const float* bias, const float* mixin, const float* cond, int f0)
  {
    for (int q = 0; q < Q; q++)
    {
      const float32x4_t b = vld1q_f32(bias + 4 * q);
      for (int t = 0; t < T; t++)
        acc[q][t] = b;
    }
    for (int k = 0; k < num_taps; k++)
    {
      const float* src = tap_src[k] + static_cast<size_t>(f0) * C;
      const float* wk = W + static_cast<size_t>(k) * C * C;
      for (int j4 = 0; j4 < Q; j4++)
      {
        float32x4_t iv[T];
        for (int t = 0; t < T; t++)
          iv[t] = vld1q_f32(src + t * C + j4 * 4);
        const float* wc = wk + static_cast<size_t>(j4 * 4) * C;
        conv_lane<T, 0>(acc, iv, wc);
        conv_lane<T, 1>(acc, iv, wc);
        conv_lane<T, 2>(acc, iv, wc);
        conv_lane<T, 3>(acc, iv, wc);
      }
    }
    for (int t = 0; t < T; t++)
    {
      const float32x4_t cf = vdupq_n_f32(cond[f0 + t]);
      for (int q = 0; q < Q; q++)
        acc[q][t] = vfmaq_f32(acc[q][t], vld1q_f32(mixin + 4 * q), cf);
    }
  }

  static inline void leaky(float32x4_t& x)
  {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    x = vbslq_f32(vcltq_f32(x, zero), vmulq_f32(x, slope), x);
  }

  template <int T>
  static inline void conv_tile_store(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                                     const float* mixin, const float* cond, float* z, int f0)
  {
    float32x4_t acc[Q][T];
    conv_acc<T>(acc, tap_src, num_taps, W, bias, mixin, cond, f0);
    for (int t = 0; t < T; t++)
    {
      float* zc = z + static_cast<size_t>(f0 + t) * C;
      for (int q = 0; q < Q; q++)
        vst1q_f32(zc + 4 * q, acc[q][t]);
    }
  }

  static void conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                         const float* mixin, const float* cond, float* z, int N)
  {
    int f = 0;
    for (; f + kConvTile <= N; f += kConvTile)
      conv_tile_store<kConvTile>(tap_src, num_taps, W, bias, mixin, cond, z, f);
    for (; f < N; f++)
      conv_tile_store<1>(tap_src, num_taps, W, bias, mixin, cond, z, f);
  }

  static void apply_leaky(float* p, int n)
  {
    for (int i = 0; i < n; i += 4)
    {
      float32x4_t x = vld1q_f32(p + i);
      leaky(x);
      vst1q_f32(p + i, x);
    }
  }

  // --- tail ------------------------------------------------------------------

  template <int T, int LANE>
  static inline void tail_lane(float32x4_t (&out)[Q][T], const float32x4_t (&a)[Q][T], const float* lc, int j4)
  {
    float32x4_t w[Q];
    for (int q = 0; q < Q; q++)
      w[q] = vld1q_f32(lc + LANE * C + 4 * q);
    for (int t = 0; t < T; t++)
      for (int q = 0; q < Q; q++)
        out[q][t] = vfmaq_laneq_f32(out[q][t], w[q], a[j4][t], LANE);
  }

  /// head_sum += a, then (when active) lin += L * a + l1x1_bias.
  template <int T, bool StoreHead, bool DoL1x1>
  static inline void tail_from(const float32x4_t (&a)[Q][T], const float* L, const float* lb, float* head_sum,
                               const float* prev, float* d, int f0)
  {
    for (int t = 0; t < T; t++)
    {
      float* hc = head_sum + static_cast<size_t>(f0 + t) * C;
      for (int q = 0; q < Q; q++)
      {
        if constexpr (StoreHead)
          vst1q_f32(hc + 4 * q, a[q][t]);
        else
          vst1q_f32(hc + 4 * q, vaddq_f32(vld1q_f32(hc + 4 * q), a[q][t]));
      }
    }
    if constexpr (!DoL1x1)
      return;
    float32x4_t out[Q][T];
    for (int q = 0; q < Q; q++)
    {
      const float32x4_t b = vld1q_f32(lb + 4 * q);
      for (int t = 0; t < T; t++)
        out[q][t] = b;
    }
    for (int j4 = 0; j4 < Q; j4++)
    {
      const float* lc = L + static_cast<size_t>(j4 * 4) * C;
      tail_lane<T, 0>(out, a, lc, j4);
      tail_lane<T, 1>(out, a, lc, j4);
      tail_lane<T, 2>(out, a, lc, j4);
      tail_lane<T, 3>(out, a, lc, j4);
    }
    for (int t = 0; t < T; t++)
    {
      const float* pc = prev + static_cast<size_t>(f0 + t) * C;
      float* dc = d + static_cast<size_t>(f0 + t) * C;
      for (int q = 0; q < Q; q++)
        vst1q_f32(dc + 4 * q, vaddq_f32(vld1q_f32(pc + 4 * q), out[q][t]));
    }
  }

  template <int T, bool StoreHead, bool DoL1x1>
  static inline void tail_tile(const float* z, const float* L, const float* lb, float* head_sum, const float* prev,
                               float* d, int f0)
  {
    float32x4_t a[Q][T];
    for (int t = 0; t < T; t++)
    {
      const float* zc = z + static_cast<size_t>(f0 + t) * C;
      for (int q = 0; q < Q; q++)
        a[q][t] = vld1q_f32(zc + 4 * q);
    }
    tail_from<T, StoreHead, DoL1x1>(a, L, lb, head_sum, prev, d, f0);
  }

  template <bool StoreHead, bool DoL1x1>
  static void tail_block(const float* z, const float* L, const float* lb, float* head_sum, const float* prev,
                         float* d, int N)
  {
    int f = 0;
    for (; f + kTailTile <= N; f += kTailTile)
      tail_tile<kTailTile, StoreHead, DoL1x1>(z, L, lb, head_sum, prev, d, f);
    for (; f < N; f++)
      tail_tile<1, StoreHead, DoL1x1>(z, L, lb, head_sum, prev, d, f);
  }

  /// The fused single pass: conv, mixin, activation, head_sum and layer1x1 in
  /// one tile, so z never reaches memory.
  template <int T, bool StoreHead, bool DoL1x1>
  static inline void fused_tile(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                                const float* mixin, const float* cond, const float* L, const float* lb,
                                float* head_sum, const float* prev, float* d, int f0)
  {
    float32x4_t acc[Q][T];
    conv_acc<T>(acc, tap_src, num_taps, W, bias, mixin, cond, f0);
    for (int t = 0; t < T; t++)
      for (int q = 0; q < Q; q++)
        leaky(acc[q][t]);
    tail_from<T, StoreHead, DoL1x1>(acc, L, lb, head_sum, prev, d, f0);
  }

  template <bool StoreHead, bool DoL1x1>
  static void fused_block(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                          const float* mixin, const float* cond, const float* L, const float* lb, float* head_sum,
                          const float* prev, float* d, int N)
  {
    int f = 0;
    for (; f + kConvTile <= N; f += kConvTile)
      fused_tile<kConvTile, StoreHead, DoL1x1>(tap_src, num_taps, W, bias, mixin, cond, L, lb, head_sum, prev, d, f);
    for (; f < N; f++)
      fused_tile<1, StoreHead, DoL1x1>(tap_src, num_taps, W, bias, mixin, cond, L, lb, head_sum, prev, d, f);
  }

  static void rechannel_block(const float* W, const float* x, float* out, int N)
  {
    for (int f = 0; f < N; f++)
    {
      float32x4_t acc[Q];
      for (int q = 0; q < Q; q++)
        acc[q] = vdupq_n_f32(0.0f);
      const float32x4_t s = vdupq_n_f32(x[f]);
      for (int q = 0; q < Q; q++)
        acc[q] = vfmaq_f32(acc[q], vld1q_f32(W + 4 * q), s);
      float* oc = out + static_cast<size_t>(f) * C;
      for (int q = 0; q < Q; q++)
        vst1q_f32(oc + 4 * q, acc[q]);
    }
  }

  // --- layers ----------------------------------------------------------------

  void dispatch_layer(int li, int N)
  {
    const bool first = (li == 0);
    const bool last = (li == kNumLayers - 1);
    const bool store_head = first && Opt::kFoldHeadInit;

    if (_w.layers[li].kernel_size == 6)
    {
      if (store_head)
        layer_forward<6, true, true>(li, N);
      else if (!last)
        layer_forward<6, false, true>(li, N);
      else
        layer_forward<6, false, false>(li, N);
    }
    else
    {
      if (store_head)
        layer_forward<15, true, true>(li, N);
      else if (!last)
        layer_forward<15, false, true>(li, N);
      else
        layer_forward<15, false, false>(li, N);
    }
  }

  template <int K, bool StoreHead, bool DoL1x1>
  void layer_forward(int li, int N)
  {
    Ring& R = _rings[li];
    const LayerWeights& L = _w.layers[li];

    if constexpr (!Opt::kRingDirect)
    {
      R.prepare(N);
      std::memcpy(R.write_ptr(), _lin.data(), static_cast<size_t>(N) * C * sizeof(float));
      R.commit(N);
    }

    const float* tap_src[K];
    for (int k = 0; k < K; k++)
      tap_src[k] = R.tap((K - 1 - k) * L.dilation, N);

    // The layer's own input for the residual add is the offset-0 tap.
    const float* prev = tap_src[K - 1];

    Ring* next = nullptr;
    float* d = nullptr;
    if constexpr (Opt::kRingDirect)
    {
      if (li + 1 < kNumLayers)
      {
        next = &_rings[li + 1];
        next->prepare(N);
        d = next->write_ptr();
      }
      else
      {
        d = _lin.data();
      }
    }
    else
    {
      d = _lin.data();
      prev = _lin.data(); // same values; avoids a dependency on the ring copy
    }

    if constexpr (Opt::kFuseZ)
    {
      fused_block<StoreHead, DoL1x1>(tap_src, K, L.conv_w.data(), L.conv_b.data(), L.mixin_w.data(), _cond.data(),
                                     L.l1x1_w.data(), L.l1x1_b.data(), _hs, prev, d, N);
    }
    else
    {
      conv_block(tap_src, K, L.conv_w.data(), L.conv_b.data(), L.mixin_w.data(), _cond.data(), _z.data(), N);
      apply_leaky(_z.data(), C * N);
      tail_block<StoreHead, DoL1x1>(_z.data(), L.l1x1_w.data(), L.l1x1_b.data(), _hs, prev, d, N);
    }

    if constexpr (Opt::kRingDirect)
    {
      if (next != nullptr)
        next->commit(N);
    }
  }

  /// Head, H = 1: one accumulator per frame across all 16 taps and both channel
  /// groups, finished with a pairwise vaddvq_f32 and then the bias — fused's
  /// order. kHeadFrames of those chains run at once.
  void head_forward(int N)
  {
    if constexpr (!Opt::kRingDirect)
    {
      _head_ring.prepare(N);
      std::memcpy(_head_ring.write_ptr(), _head_sum.data(), static_cast<size_t>(N) * C * sizeof(float));
      _head_ring.commit(N);
    }

    const float* tap_src[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      tap_src[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* hw = _head_w.data();
    const float bias = _w.head_b;
    float* out = _head_out.data();

    int f = 0;
    for (; f + kHeadFrames <= N; f += kHeadFrames)
    {
      float32x4_t acc[kHeadFrames];
      for (int u = 0; u < kHeadFrames; u++)
        acc[u] = vdupq_n_f32(0.0f);
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* w = hw + static_cast<size_t>(k) * C;
        float32x4_t wv[Q];
        for (int q = 0; q < Q; q++)
          wv[q] = vld1q_f32(w + 4 * q);
        for (int u = 0; u < kHeadFrames; u++)
        {
          const float* src = tap_src[k] + static_cast<size_t>(f + u) * C;
          for (int q = 0; q < Q; q++)
            acc[u] = vfmaq_f32(acc[u], wv[q], vld1q_f32(src + 4 * q));
        }
      }
      for (int u = 0; u < kHeadFrames; u++)
        out[f + u] = vaddvq_f32(acc[u]) + bias;
    }
    for (; f < N; f++)
    {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* src = tap_src[k] + static_cast<size_t>(f) * C;
        const float* w = hw + static_cast<size_t>(k) * C;
        for (int q = 0; q < Q; q++)
          acc = vfmaq_f32(acc, vld1q_f32(w + 4 * q), vld1q_f32(src + 4 * q));
      }
      out[f] = vaddvq_f32(acc) + bias;
    }
  }

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _head_w;
  std::vector<float> _cond;
  std::vector<float> _lin;
  std::vector<float> _z;
  std::vector<float> _head_sum;
  std::vector<float> _head_out;
  float* _hs = nullptr;
};

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
