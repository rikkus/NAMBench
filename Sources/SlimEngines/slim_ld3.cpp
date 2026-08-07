// Kernel 8: ld3 — get planar registers without a planar layout.
//
// `planar` wins (if it wins) by vectorising across frames, which needs each
// channel's four frames in one register. AArch64 can produce exactly that from
// interleaved memory in a single instruction: vld3q_f32 loads twelve floats and
// deals them into three registers by channel. So the layout change may not be
// necessary at all — keep a2_fast's interleaved rings, and de-interleave at the
// point of use.
//
// The question this asks is narrow and worth asking: is the de-interleave
// cheaper than the layout change? ld3/st3 are multi-cycle structured accesses
// and there are K of them per tile, against one extra store pass per layer for
// the planar rings. It is not obvious from the outside which side that lands
// on. Expected to lose — the conv reads each ring window once per tap, so the
// de-interleave is paid K times per block while the planar layout pays for it
// once — but "expected" is not "measured".

#if defined(NB_ENABLE_SLIM_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "slim_common.h"

  #if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
  #else
    #error "ld3 is a NEON kernel; this lab is arm64-only"
  #endif

namespace slimlab
{
namespace
{

class Ld3Model : public SlimModel
{
public:
  Ld3Model(const std::vector<float>& weights, double sampleRate)
  : SlimModel(weights, sampleRate)
  {
    for (int li = 0; li < kNumLayers; li++)
    {
      const LayerWeights& L = _w.layers[li];
      Padded& P = _p[li];
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

    float* cond = _cond.data();
    float* lin = _layer_in.data();
    const float rw0 = _w.rechannel_w[0], rw1 = _w.rechannel_w[1], rw2 = _w.rechannel_w[2];
    for (int f = 0; f < N; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      lin[f * 3 + 0] = rw0 * x;
      lin[f * 3 + 1] = rw1 * x;
      lin[f * 3 + 2] = rw2 * x;
    }

    std::memset(_head_sum.data(), 0, static_cast<size_t>(N) * 3 * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
    {
      if (_w.layers[li].kernel_size == 6)
        layer_forward<6>(li, N);
      else
        layer_forward<15>(li, N);
    }

    head_forward(N);
    const float* head_out = _head_out.data();
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);
    _layer_in.assign(static_cast<size_t>(3) * maxBufferSize, 0.0f);
    _head_sum.assign(static_cast<size_t>(3) * maxBufferSize, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    for (int li = 0; li < kNumLayers; li++)
      reset_ring(_rings[li], _w.layers[li].max_lookback, maxBufferSize);
    reset_ring(_head_ring, kHeadKernelSize - 1, maxBufferSize);
  }

private:
  struct Padded
  {
    std::vector<float> conv_w;
    std::array<float, 4> conv_b{};
    std::array<float, 4> mixin_w{};
    std::array<float, 12> l1x1_w{};
    std::array<float, 4> l1x1_b{};
  };

  /// a2_fast's interleaved ring, unchanged: pow2 capacity, eager tail mirror.
  struct Ring
  {
    std::vector<float> data;
    int pow2 = 0;
    int mask = 0;
    int write_pos = 0;
  };

  static void reset_ring(Ring& R, int max_lookback, int maxBufferSize)
  {
    R.pow2 = next_pow2(max_lookback + maxBufferSize);
    R.mask = R.pow2 - 1;
    R.data.assign(static_cast<size_t>(3) * (R.pow2 + maxBufferSize), 0.0f);
    R.write_pos = max_lookback;
  }

  void ring_write(Ring& R, const float* src, int num_frames)
  {
    const int mbs = GetMaxBufferSize();
    float* const hist = R.data.data();
    const int wp = R.write_pos;
    const int first = std::min(num_frames, R.pow2 - wp);
    std::memcpy(hist + static_cast<size_t>(wp) * 3, src, static_cast<size_t>(first) * 3 * sizeof(float));
    if (first < num_frames)
    {
      std::memcpy(hist, src + static_cast<size_t>(first) * 3,
                  static_cast<size_t>(num_frames - first) * 3 * sizeof(float));
    }
    std::memcpy(hist + static_cast<size_t>(R.pow2) * 3, hist, static_cast<size_t>(mbs) * 3 * sizeof(float));
    R.write_pos = (wp + num_frames) & R.mask;
  }

  template <int K>
  void layer_forward(int li, int N)
  {
    Ring& R = _rings[li];
    const LayerWeights& L = _w.layers[li];
    const Padded& P = _p[li];

    ring_write(R, _layer_in.data(), N);

    const float* hist = R.data.data();
    int tapb[K];
    for (int k = 0; k < K; k++)
      tapb[k] = (R.write_pos - N - (K - 1 - k) * L.dilation) & R.mask;

    const float32x4_t cb = vld1q_f32(P.conv_b.data());
    const float32x4_t M = vld1q_f32(P.mixin_w.data());
    const float32x4_t LA = vld1q_f32(P.l1x1_w.data());
    const float32x4_t LB = vld1q_f32(P.l1x1_w.data() + 4);
    const float32x4_t LC = vld1q_f32(P.l1x1_w.data() + 8);
    const float32x4_t LBias = vld1q_f32(P.l1x1_b.data());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float* cond = _cond.data();
    float* head_sum = _head_sum.data();
    float* lin = _layer_in.data();

    int f = 0;
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t a0 = vdupq_laneq_f32(cb, 0);
      float32x4_t a1 = vdupq_laneq_f32(cb, 1);
      float32x4_t a2 = vdupq_laneq_f32(cb, 2);
      float32x4x3_t cur;

      for (int k = 0; k < K; k++)
      {
        const float* wk = P.conv_w.data() + static_cast<size_t>(k) * 12;
        const float32x4_t A = vld1q_f32(wk);
        const float32x4_t B = vld1q_f32(wk + 4);
        const float32x4_t C = vld1q_f32(wk + 8);
        // One instruction turns twelve interleaved floats into three
        // frame-major registers.
        const float32x4x3_t s = vld3q_f32(hist + static_cast<size_t>(tapb[k] + f) * 3);
        a0 = vfmaq_laneq_f32(a0, s.val[0], A, 0);
        a1 = vfmaq_laneq_f32(a1, s.val[0], A, 1);
        a2 = vfmaq_laneq_f32(a2, s.val[0], A, 2);
        a0 = vfmaq_laneq_f32(a0, s.val[1], A, 3);
        a1 = vfmaq_laneq_f32(a1, s.val[1], B, 0);
        a2 = vfmaq_laneq_f32(a2, s.val[1], B, 1);
        a0 = vfmaq_laneq_f32(a0, s.val[2], B, 2);
        a1 = vfmaq_laneq_f32(a1, s.val[2], B, 3);
        a2 = vfmaq_laneq_f32(a2, s.val[2], C, 0);
        if (k == K - 1)
          cur = s;
      }

      const float32x4_t cf = vld1q_f32(cond + f);
      a0 = vfmaq_laneq_f32(a0, cf, M, 0);
      a1 = vfmaq_laneq_f32(a1, cf, M, 1);
      a2 = vfmaq_laneq_f32(a2, cf, M, 2);
      a0 = vbslq_f32(vcltq_f32(a0, zero), vmulq_f32(a0, slope), a0);
      a1 = vbslq_f32(vcltq_f32(a1, zero), vmulq_f32(a1, slope), a1);
      a2 = vbslq_f32(vcltq_f32(a2, zero), vmulq_f32(a2, slope), a2);

      float32x4x3_t hsv = vld3q_f32(head_sum + static_cast<size_t>(f) * 3);
      hsv.val[0] = vaddq_f32(hsv.val[0], a0);
      hsv.val[1] = vaddq_f32(hsv.val[1], a1);
      hsv.val[2] = vaddq_f32(hsv.val[2], a2);
      vst3q_f32(head_sum + static_cast<size_t>(f) * 3, hsv);

      float32x4_t o0 = vdupq_laneq_f32(LBias, 0);
      float32x4_t o1 = vdupq_laneq_f32(LBias, 1);
      float32x4_t o2 = vdupq_laneq_f32(LBias, 2);
      o0 = vfmaq_laneq_f32(o0, a0, LA, 0);
      o0 = vfmaq_laneq_f32(o0, a1, LA, 3);
      o0 = vfmaq_laneq_f32(o0, a2, LB, 2);
      o1 = vfmaq_laneq_f32(o1, a0, LA, 1);
      o1 = vfmaq_laneq_f32(o1, a1, LB, 0);
      o1 = vfmaq_laneq_f32(o1, a2, LB, 3);
      o2 = vfmaq_laneq_f32(o2, a0, LA, 2);
      o2 = vfmaq_laneq_f32(o2, a1, LB, 1);
      o2 = vfmaq_laneq_f32(o2, a2, LC, 0);
      float32x4x3_t res;
      res.val[0] = vaddq_f32(cur.val[0], o0);
      res.val[1] = vaddq_f32(cur.val[1], o1);
      res.val[2] = vaddq_f32(cur.val[2], o2);
      vst3q_f32(lin + static_cast<size_t>(f) * 3, res);
    }

    for (; f < N; f++)
    {
      float a[3] = {P.conv_b[0], P.conv_b[1], P.conv_b[2]};
      for (int k = 0; k < K; k++)
      {
        const float* wk = P.conv_w.data() + static_cast<size_t>(k) * 12;
        const float* src = hist + static_cast<size_t>(tapb[k] + f) * 3;
        a[0] += wk[0] * src[0];
        a[1] += wk[1] * src[0];
        a[2] += wk[2] * src[0];
        a[0] += wk[3] * src[1];
        a[1] += wk[4] * src[1];
        a[2] += wk[5] * src[1];
        a[0] += wk[6] * src[2];
        a[1] += wk[7] * src[2];
        a[2] += wk[8] * src[2];
      }
      const float cfs = cond[f];
      const float* cur = hist + static_cast<size_t>(tapb[K - 1] + f) * 3;
      for (int c = 0; c < 3; c++)
      {
        a[c] += P.mixin_w[c] * cfs;
        a[c] = (a[c] < 0.0f) ? a[c] * kLeakySlope : a[c];
        head_sum[f * 3 + c] += a[c];
      }
      for (int c = 0; c < 3; c++)
      {
        float o = P.l1x1_b[c];
        o += P.l1x1_w[0 + c] * a[0];
        o += P.l1x1_w[3 + c] * a[1];
        o += P.l1x1_w[6 + c] * a[2];
        lin[f * 3 + c] = cur[c] + o;
      }
    }
  }

  void head_forward(int N)
  {
    ring_write(_head_ring, _head_sum.data(), N);
    const float* hist = _head_ring.data.data();
    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = (_head_ring.write_pos - N - (kHeadKernelSize - 1 - k)) & _head_ring.mask;

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
        const float32x4x3_t s = vld3q_f32(hist + static_cast<size_t>(hb[k] + f) * 3);
        y = vfmaq_laneq_f32(y, s.val[0], W, 0);
        y = vfmaq_laneq_f32(y, s.val[1], W, 1);
        y = vfmaq_laneq_f32(y, s.val[2], W, 2);
      }
      vst1q_f32(out + f, vmulq_n_f32(y, scale));
    }
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* w = hw + static_cast<size_t>(k) * 4;
        const float* src = hist + static_cast<size_t>(hb[k] + f) * 3;
        y += w[0] * src[0];
        y += w[1] * src[1];
        y += w[2] * src[2];
      }
      out[f] = y * scale;
    }
  }

  std::array<Padded, kNumLayers> _p;
  std::vector<float> _head_w4;
  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_ld3(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<Ld3Model>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
