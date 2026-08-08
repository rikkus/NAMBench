// The planar engine for the 8-channel path, reproducing a2_fast's arithmetic
// exactly, and the switches layered on top of it. One implementation, several
// candidates: each candidate .cpp instantiates this template with a different
// option set, so what is being compared really is one idea at a time rather than
// one rewrite at a time.
//
// Why planar at C=8. The obvious objection is that at C=8 a channel-major vector
// wastes no lanes, so the layout change that won at C=3 has nothing to win. Two
// answers:
//
//   1. It is what makes bit-identity practical. A planar register holds four
//      consecutive *frames* of one channel, so every lane independently
//      executes a2_fast's per-frame scalar chain. Nothing is reassociated. The
//      fused engine cannot do this — it vectorises across channels, so its
//      reduction runs across lanes and lands on a different association, which
//      is exactly where its 132.6 dB against a2_fast comes from.
//   2. It costs nothing. Per tap per frame both layouts are 2 input loads,
//      4 weight loads and 16 FMAs.
//
// The arithmetic being reproduced, established by Scripts/eigen-order-probe
// against Eigen itself over 224 (block size, kernel size, trial) combinations:
//
//   per tap k:  t_i = 0; for j = 0..7: t_i = fma(W_k(i,j), x_k(j), t_i)
//   then:       z_i = 0; for k = 0..K-1: z_i = z_i + t_i^(k)
//   then:       z_i = z_i + bias_i
//               z_i = z_i + (mixin_i * cond)      <- product then add, two roundings
//               z_i = z_i < 0 ? z_i * 0.01 : z_i
//               head_sum_i = head_sum_i + z_i
//               u_i = 0; for j: u_i = fma(L(i,j), z_j, u_i)
//               lin_i = (lin_i + u_i) + l1x1_b_i
//
// and for the head, 16 taps × 8 channels of sequential FMA from the bias. The
// probe also showed the head's contraction is load-bearing: the same order with
// the multiply and add rounded separately matches only 6 times in 224.

#pragma once

#if defined(NB_ENABLE_FULL_LAB)

  #include <algorithm>
  #include <cstring>
  #include <utility>
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

/// The option set every candidate in this family varies.
///
/// Candidates inherit from this and override the members they change, so a
/// candidate .cpp reads as a diff against `a2p4`.
struct A2PlanarOptions
{
  /// Frames per conv tile, a multiple of 4. Four is one NEON register per
  /// channel; higher trades register pressure for independent FMA chains and
  /// for weight-load amortisation.
  ///
  /// Note the ceiling is lower here than in the nano lab, because a2_fast's
  /// order needs the running total `z` and the current tap's partial `t` live at
  /// the same time: 2 × 8 × (tile/4) accumulator registers, so tile 4 needs 16
  /// of the 32 and tile 8 needs 32 before inputs and weights. Where that spills
  /// is a measurement, not a guess.
  static constexpr int kTile = 4;
  /// How many passes the conv makes over the output channels. One pass keeps
  /// z and t for all eight channels live at once — 16 registers at tile 4, 32 at
  /// tile 8. Two passes halves what the tap loop needs (z for all eight survives,
  /// but only four channels of t are live), at the cost of loading each input
  /// plane twice per tap. Whether that trade pays is a measurement.
  static constexpr int kOutPasses = 1;
  /// Frames per head tile, in units of 4. The head is a 128-long serial FMA
  /// chain per frame; this many independent chains run at once.
  static constexpr int kHeadVecs = 1;
  /// Hold the layer1x1 weights transposed, so the eight weights feeding one
  /// output channel are contiguous and can be lane-broadcast from two vector
  /// loads instead of eight scalar ones. Same order, same FMAs, fewer loads.
  static constexpr bool kL1x1Lane = false;
  /// Which ring strategy to use. The default matches a2_fast's default.
  static constexpr RingKind kRing = RingKind::Pow2Eager;
  /// Write each layer's residual straight into the next layer's ring instead of
  /// into a scratch buffer that the next layer then copies in.
  static constexpr bool kRingDirect = false;
  /// Skip the last layer's layer1x1. Its residual is never read — there is no
  /// layer 24, and the head reads head_sum — so not computing it is
  /// observationally identical, not an approximation.
  static constexpr bool kSkipLastL1x1 = false;
  /// Fold the per-block memset of head_sum into the first layer's accumulate.
  /// Kept as an add against +0.0 rather than a plain store, because a store
  /// would differ from a2_fast on a signed zero.
  static constexpr bool kFoldHeadInit = false;
  /// Prefetch the next block's taps on the far-dilation layers.
  static constexpr bool kPrefetch = false;
};

template <class Opt>
class A2PlanarModel : public FullModel
{
  static constexpr int C = kChannels;
  static constexpr int kTile = Opt::kTile;
  static constexpr int kVecs = kTile / 4;
  static constexpr int kHeadVecs = Opt::kHeadVecs;

  using Ring = PlanarRing<Opt::kRing>;

  static_assert(kTile % 4 == 0 && kTile >= 4, "kTile must be a positive multiple of 4");
  static_assert(kHeadVecs >= 1, "kHeadVecs must be at least 1");

public:
  A2PlanarModel(const std::vector<float>& weights, double sampleRate)
  : FullModel(weights, sampleRate)
  {
    // Head weights as 8 contiguous floats per tap, so a tap's whole weight row
    // is two vector loads addressed by lane.
    _head_w.assign(static_cast<size_t>(kHeadKernelSize) * C, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int b = 0; b < C; b++)
        _head_w[static_cast<size_t>(k) * C + b] = _w.head_w[k][b];

    // layer1x1 transposed: [i * C + j] is the weight from bottleneck j to output
    // i, so one output's whole row is two contiguous vector loads.
    _l1x1t.assign(static_cast<size_t>(kNumLayers) * C * C, 0.0f);
    for (int li = 0; li < kNumLayers; li++)
      for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
          _l1x1t[(static_cast<size_t>(li) * C + i) * C + j] = _w.layers[li].l1x1_w[j * C + i];
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

    // Rechannel straight into layer 0's destination: its ring under kRingDirect,
    // otherwise the scratch planes it will be copied from.
    if constexpr (Opt::kRingDirect)
      _rings[0].prepare(N);
    for (int c = 0; c < C; c++)
    {
      float* dst = Opt::kRingDirect ? _rings[0].write_ptr(c) : lin(c);
      const float wc = _w.rechannel_w[c];
      int f = 0;
      for (; f + 4 <= N; f += 4)
        vst1q_f32(dst + f, vmulq_n_f32(vld1q_f32(cond + f), wc));
      for (; f < N; f++)
        dst[f] = wc * cond[f];
    }
    if constexpr (Opt::kRingDirect)
      _rings[0].commit(N);

    // Where the layers accumulate head_sum. Under kRingDirect that is the head
    // ring's write window directly, which removes the block-sized copy the head
    // would otherwise make out of a scratch buffer — the same trick as writing
    // each layer's residual into the next layer's ring.
    if constexpr (Opt::kRingDirect)
    {
      _head_ring.prepare(N);
      for (int c = 0; c < C; c++)
        _hs[c] = _head_ring.write_ptr(c);
    }
    else
    {
      for (int c = 0; c < C; c++)
        _hs[c] = hsum(c);
    }

    if constexpr (!Opt::kFoldHeadInit)
    {
      for (int c = 0; c < C; c++)
        std::memset(_hs[c], 0, static_cast<size_t>(N) * sizeof(float));
    }

    for (int li = 0; li < kNumLayers; li++)
      dispatch_layer(li, N);

    if constexpr (Opt::kRingDirect)
      _head_ring.commit(N);

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

  void dispatch_layer(int li, int N)
  {
    const bool first = (li == 0);
    const bool last = (li == kNumLayers - 1);
    const bool do_l1x1 = !(last && Opt::kSkipLastL1x1);
    const bool store_head = first && Opt::kFoldHeadInit;

    // The booleans are loop-invariant for the whole block, so none of this
    // reaches the inner loop.
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
    const LayerWeights& L = _w.layers[li];
    _li = li;

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

    if constexpr (Opt::kPrefetch)
    {
      // The far-dilation layers reach about 38 KB back at C=8, so the next
      // block's taps are certain to have fallen out of L1 by the time they are
      // wanted. The nano lab found this made things worse at C=3; the reach is
      // larger here, so it is worth one measurement rather than an assumption.
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

    // Residual destination: the next layer's ring under kRingDirect, otherwise
    // the scratch planes that the next layer will copy in.
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
        // The final layer's residual feeds nothing. When it is computed at all
        // it goes to scratch.
        for (int c = 0; c < C; c++)
          d[c] = lin(c);
      }
    }
    else
    {
      for (int c = 0; c < C; c++)
        d[c] = lin(c);
    }

    float* const* hs = _hs;

    int f = 0;
    if constexpr (kVecs > 1)
    {
      for (; f + kTile <= N; f += kTile)
        tile<K, kVecs, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);
    }
    for (; f + 4 <= N; f += 4)
      tile<K, 1, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);
    for (; f < N; f++)
      frame_scalar<K, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);

    if constexpr (Opt::kRingDirect)
    {
      if (next != nullptr)
        next->commit(N);
    }
  }

  /// NVEC × 4 frames. `z` is the running total across taps and `t` the current
  /// tap's partial — both live at once, because that separation *is* a2_fast's
  /// association. z never reaches memory.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  inline void tile(const LayerWeights& L, const float* const* h, const int (&tapb)[K], int f0, float* const* d,
                   float* const* hs)
  {
    constexpr int NG = Opt::kOutPasses;
    constexpr int GC = C / NG; // output channels per pass

    float32x4_t z[C][NVEC];
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int i = 0; i < C; i++)
      for (int v = 0; v < NVEC; v++)
        z[i][v] = zero;

    const float* cw = L.conv_w.data();
    for (int g = 0; g < NG; g++)
    {
      const int i0 = g * GC;
      for (int k = 0; k < K; k++)
      {
        const float* wk = cw + static_cast<size_t>(k) * C * C;
        const int base = tapb[k] + f0;
        float32x4_t t[GC][NVEC];

        // Input channel j is unrolled at compile time so that j == 0 can seed
        // the partial with a multiply instead of an FMA against zero. Both are
        // one rounding of w*x, so this is exact either way; it just saves the
        // zero-init.
        const auto do_j = [&](auto jc) {
          constexpr int j = decltype(jc)::value;
          const float* wj = wk + j * C + i0; // W(i0 .. i0+GC-1, j), contiguous
          float32x4_t wv[GC / 4];
          for (int u = 0; u < GC / 4; u++)
            wv[u] = vld1q_f32(wj + 4 * u);
          const float* hp = h[j] + base;
          for (int v = 0; v < NVEC; v++)
          {
            const float32x4_t s = vld1q_f32(hp + 4 * v);
            const auto do_i = [&](auto ic) {
              constexpr int i = decltype(ic)::value;
              if constexpr (j == 0)
                t[i][v] = vmulq_laneq_f32(s, wv[i / 4], i % 4);
              else
                t[i][v] = vfmaq_laneq_f32(t[i][v], s, wv[i / 4], i % 4);
            };
            [&]<int... Js>(std::integer_sequence<int, Js...>) {
              (do_i(std::integral_constant<int, Js>{}), ...);
            }(std::make_integer_sequence<int, GC>{});
          }
        };
        [&]<int... Is>(std::integer_sequence<int, Is...>) {
          (do_j(std::integral_constant<int, Is>{}), ...);
        }(std::make_integer_sequence<int, C>{});

        for (int i = 0; i < GC; i++)
          for (int v = 0; v < NVEC; v++)
            z[i0 + i][v] = vaddq_f32(z[i0 + i][v], t[i][v]);
      }
    }

    post<K, NVEC, StoreHead, DoL1x1>(L, h, tapb[K - 1], f0, z, d, hs);
  }

  /// Everything after the conv: bias, mixin, LeakyReLU, head_sum, layer1x1
  /// residual — in a2_fast's order, with the mixin's multiply and add rounded
  /// separately as Eigen rounds them.
  ///
  /// `last_tap` is the base of the offset-0 tap, i.e. this block's own input.
  /// Reloading it here rather than carrying it through the tap loop is what
  /// keeps the residual add off the register budget.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  inline void post(const LayerWeights& L, const float* const* h, int last_tap, int f0, float32x4_t (&z)[C][NVEC],
                   float* const* d, float* const* hs)
  {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float* cond = _cond.data();

    float32x4_t cf[NVEC];
    for (int v = 0; v < NVEC; v++)
      cf[v] = vld1q_f32(cond + f0 + 4 * v);

    for (int i = 0; i < C; i++)
    {
      const float32x4_t b = vdupq_n_f32(L.conv_b[i]);
      const float m = L.mixin_w[i];
      for (int v = 0; v < NVEC; v++)
      {
        float32x4_t a = vaddq_f32(z[i][v], b);
        a = vaddq_f32(a, vmulq_n_f32(cf[v], m)); // product then add
        z[i][v] = vbslq_f32(vcltq_f32(a, zero), vmulq_f32(a, slope), a);
      }
    }

    for (int i = 0; i < C; i++)
    {
      float* p = hs[i] + f0;
      for (int v = 0; v < NVEC; v++)
      {
        if constexpr (StoreHead)
          vst1q_f32(p + 4 * v, vaddq_f32(zero, z[i][v]));
        else
          vst1q_f32(p + 4 * v, vaddq_f32(vld1q_f32(p + 4 * v), z[i][v]));
      }
    }

    if constexpr (!DoL1x1)
      return;

    const float* lw = L.l1x1_w.data();
    const float* lt = _l1x1t.data() + static_cast<size_t>(_li) * C * C;
    for (int i = 0; i < C; i++)
    {
      const float bi = L.l1x1_b[i];
      // u_i = sum over j in increasing order, from zero. Either form is the same
      // FMA in the same order; they differ only in how the weight reaches the
      // instruction.
      float32x4_t la{}, lb{};
      if constexpr (Opt::kL1x1Lane)
      {
        la = vld1q_f32(lt + i * C); // L(i, 0..3)
        lb = vld1q_f32(lt + i * C + 4); // L(i, 4..7)
      }
      for (int v = 0; v < NVEC; v++)
      {
        float32x4_t u;
        if constexpr (Opt::kL1x1Lane)
        {
          u = vmulq_laneq_f32(z[0][v], la, 0);
          u = vfmaq_laneq_f32(u, z[1][v], la, 1);
          u = vfmaq_laneq_f32(u, z[2][v], la, 2);
          u = vfmaq_laneq_f32(u, z[3][v], la, 3);
          u = vfmaq_laneq_f32(u, z[4][v], lb, 0);
          u = vfmaq_laneq_f32(u, z[5][v], lb, 1);
          u = vfmaq_laneq_f32(u, z[6][v], lb, 2);
          u = vfmaq_laneq_f32(u, z[7][v], lb, 3);
        }
        else
        {
          u = vmulq_n_f32(z[0][v], lw[0 * C + i]);
          u = vfmaq_n_f32(u, z[1][v], lw[1 * C + i]);
          u = vfmaq_n_f32(u, z[2][v], lw[2 * C + i]);
          u = vfmaq_n_f32(u, z[3][v], lw[3 * C + i]);
          u = vfmaq_n_f32(u, z[4][v], lw[4 * C + i]);
          u = vfmaq_n_f32(u, z[5][v], lw[5 * C + i]);
          u = vfmaq_n_f32(u, z[6][v], lw[6 * C + i]);
          u = vfmaq_n_f32(u, z[7][v], lw[7 * C + i]);
        }
        const float32x4_t prev = vld1q_f32(h[i] + last_tap + f0 + 4 * v);
        vst1q_f32(d[i] + f0 + 4 * v, vaddq_f32(vaddq_f32(prev, u), vdupq_n_f32(bi)));
      }
    }
  }

  /// The last few frames of a block that is not a multiple of four. Same
  /// operation order as the vector path, one frame at a time.
  template <int K, bool StoreHead, bool DoL1x1>
  void frame_scalar(const LayerWeights& L, const float* const* h, const int (&tapb)[K], int f, float* const* d,
                    float* const* hs)
  {
    float z[C];
    for (int i = 0; i < C; i++)
      z[i] = 0.0f;

    for (int k = 0; k < K; k++)
    {
      const float* wk = L.conv_w.data() + static_cast<size_t>(k) * C * C;
      float t[C];
      for (int i = 0; i < C; i++)
        t[i] = 0.0f;
      for (int j = 0; j < C; j++)
      {
        const float s = h[j][tapb[k] + f];
        for (int i = 0; i < C; i++)
          t[i] += wk[j * C + i] * s;
      }
      for (int i = 0; i < C; i++)
        z[i] += t[i];
    }

    const float cf = _cond[f];
    for (int i = 0; i < C; i++)
    {
      float a = z[i] + L.conv_b[i];
      a = a + L.mixin_w[i] * cf;
      z[i] = (a < 0.0f) ? a * kLeakySlope : a;
      if constexpr (StoreHead)
        hs[i][f] = 0.0f + z[i];
      else
        hs[i][f] = hs[i][f] + z[i];
    }

    if constexpr (DoL1x1)
    {
      for (int i = 0; i < C; i++)
      {
        float u = 0.0f;
        for (int j = 0; j < C; j++)
          u += L.l1x1_w[j * C + i] * z[j];
        d[i][f] = (h[i][tapb[K - 1] + f] + u) + L.l1x1_b[i];
      }
    }
  }

  /// Head rechannel: K=16, dilation 1, eight channels down to one, plus bias and
  /// scale. a2_fast runs 128 sequential FMAs per frame; in planar layout the
  /// same 128 FMAs cover four frames at a time, and kHeadVecs of those chains
  /// run independently.
  void head_forward(int N)
  {
    // Under kRingDirect the layers accumulated straight into the ring's write
    // window and process() has already committed it.
    if constexpr (!Opt::kRingDirect)
    {
      _head_ring.prepare(N);
      for (int c = 0; c < C; c++)
        std::memcpy(_head_ring.write_ptr(c), hsum(c), static_cast<size_t>(N) * sizeof(float));
      _head_ring.commit(N);
    }

    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* p[C];
    for (int c = 0; c < C; c++)
      p[c] = _head_ring.plane(c);

    const float* hw = _head_w.data();
    const float scale = _w.head_scale;
    const float32x4_t bias = vdupq_n_f32(_w.head_b);
    float* out = _head_out.data();

    int f = 0;
    for (; f + 4 * kHeadVecs <= N; f += 4 * kHeadVecs)
    {
      float32x4_t y[kHeadVecs];
      for (int u = 0; u < kHeadVecs; u++)
        y[u] = bias;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float32x4_t wa = vld1q_f32(hw + static_cast<size_t>(k) * C);
        const float32x4_t wb = vld1q_f32(hw + static_cast<size_t>(k) * C + 4);
        const int base = hb[k] + f;
        for (int u = 0; u < kHeadVecs; u++)
        {
          const int o = base + 4 * u;
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[0] + o), wa, 0);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[1] + o), wa, 1);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[2] + o), wa, 2);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[3] + o), wa, 3);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[4] + o), wb, 0);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[5] + o), wb, 1);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[6] + o), wb, 2);
          y[u] = vfmaq_laneq_f32(y[u], vld1q_f32(p[7] + o), wb, 3);
        }
      }
      for (int u = 0; u < kHeadVecs; u++)
        vst1q_f32(out + f + 4 * u, vmulq_n_f32(y[u], scale));
    }
    if constexpr (kHeadVecs > 1)
    {
      for (; f + 4 <= N; f += 4)
      {
        float32x4_t y = bias;
        for (int k = 0; k < kHeadKernelSize; k++)
        {
          const float32x4_t wa = vld1q_f32(hw + static_cast<size_t>(k) * C);
          const float32x4_t wb = vld1q_f32(hw + static_cast<size_t>(k) * C + 4);
          const int o = hb[k] + f;
          y = vfmaq_laneq_f32(y, vld1q_f32(p[0] + o), wa, 0);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[1] + o), wa, 1);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[2] + o), wa, 2);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[3] + o), wa, 3);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[4] + o), wb, 0);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[5] + o), wb, 1);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[6] + o), wb, 2);
          y = vfmaq_laneq_f32(y, vld1q_f32(p[7] + o), wb, 3);
        }
        vst1q_f32(out + f, vmulq_n_f32(y, scale));
      }
    }
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* wk = hw + static_cast<size_t>(k) * C;
        for (int b = 0; b < C; b++)
          y += wk[b] * p[b][hb[k] + f];
      }
      out[f] = y * scale;
    }
  }

  std::vector<float> _head_w;
  std::vector<float> _l1x1t;
  int _li = 0;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
  /// Where the layers accumulate head_sum this block: the scratch planes, or the
  /// head ring's write window under kRingDirect.
  float* _hs[C] = {};
  int _stride = 0;
};

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
