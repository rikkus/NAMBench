// Kernel 3: pad4 — pad C=3 to C=4 and run the existing fused kernels.
//
// This is the obvious answer to "the fused engine already beats a2_fast by
// 1.887× at C=8, so why does it refuse C=3?" — the refusal is one line
// (`as.channels % 4 != 0`), and padding makes it go away. The kernels below are
// fused.cpp's conv_tile / tail_tile / head_conv_block instantiated at Q=1,
// copied rather than shared because vendor/ is fetched at a pinned SHA and must
// not be edited.
//
// What it costs: a quarter of every SIMD lane is multiplied by zero, and the
// ring buffers grow from 172.5 KB to 230 KB — further past the M2's 128 KB L1D
// than they already were.
//
// Why the padding is safe rather than merely plausible. Every weight touching
// lane 3 is zero, in both directions, at every stage:
//   - rechannel:  w[3] = 0            -> lin[3] starts at 0
//   - conv:       row 3 and column 3 of each tap are 0, conv_b[3] = 0
//   - mixin:      mixin[3] = 0        -> z[3] = 0 exactly
//   - activation: LeakyReLU(0) = 0    -> a[3] = 0
//   - head_sum:   accumulates 0
//   - layer1x1:   row 3 and column 3 are 0, bias[3] = 0 -> lin[3] stays 0
//   - head conv:  w[.][3] = 0         -> the pad lane contributes nothing
// So the pad lane can neither be read into a real output nor accumulate a
// value that could later denormalise; it is a hard zero for the whole run.

#if defined(NB_ENABLE_SLIM_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "slim_common.h"

  #if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
  #else
    #error "pad4 is a NEON kernel; this lab is arm64-only"
  #endif

namespace slimlab
{
namespace
{

constexpr int kPadded = 4;

// --- fused.cpp kernels, Q = channels / 4 = 1 ---------------------------------

template <int Q, int T, int LANE>
inline void conv_lane(float32x4_t (&acc)[Q][T], const float32x4_t (&iv)[T], const float* wc)
{
  constexpr int C = 4 * Q;
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(wc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      acc[q][t] = vfmaq_laneq_f32(acc[q][t], w[q], iv[t], LANE);
}

template <int Q, int T>
inline void conv_tile(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                      const float* mixin, const float* cond, float* z, int f0)
{
  constexpr int C = 4 * Q;
  float32x4_t acc[Q][T];
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
      conv_lane<Q, T, 0>(acc, iv, wc);
      conv_lane<Q, T, 1>(acc, iv, wc);
      conv_lane<Q, T, 2>(acc, iv, wc);
      conv_lane<Q, T, 3>(acc, iv, wc);
    }
  }
  for (int t = 0; t < T; t++)
  {
    const float32x4_t cf = vdupq_n_f32(cond[f0 + t]);
    float* zc = z + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(zc + 4 * q, vfmaq_f32(acc[q][t], vld1q_f32(mixin + 4 * q), cf));
  }
}

template <int Q>
void conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias, const float* mixin,
                const float* cond, float* z, int num_frames)
{
  constexpr int T = 4;
  int f = 0;
  for (; f + T <= num_frames; f += T)
    conv_tile<Q, T>(tap_src, num_taps, W, bias, mixin, cond, z, f);
  for (; f < num_frames; f++)
    conv_tile<Q, 1>(tap_src, num_taps, W, bias, mixin, cond, z, f);
}

template <int Q, int T, int LANE>
inline void tail_lane(float32x4_t (&out)[Q][T], const float32x4_t (&a)[Q][T], const float* lc, int j4)
{
  constexpr int C = 4 * Q;
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(lc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      out[q][t] = vfmaq_laneq_f32(out[q][t], w[q], a[j4][t], LANE);
}

template <int Q, int T>
inline void tail_tile(const float* z, const float* L, const float* lb, float* head_sum, float* lin, int f0)
{
  constexpr int C = 4 * Q;
  float32x4_t a[Q][T];
  for (int t = 0; t < T; t++)
  {
    const float* zc = z + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      a[q][t] = vld1q_f32(zc + 4 * q);
  }
  for (int t = 0; t < T; t++)
  {
    float* hc = head_sum + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(hc + 4 * q, vaddq_f32(vld1q_f32(hc + 4 * q), a[q][t]));
  }
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
    tail_lane<Q, T, 0>(out, a, lc, j4);
    tail_lane<Q, T, 1>(out, a, lc, j4);
    tail_lane<Q, T, 2>(out, a, lc, j4);
    tail_lane<Q, T, 3>(out, a, lc, j4);
  }
  for (int t = 0; t < T; t++)
  {
    float* lc = lin + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(lc + 4 * q, vaddq_f32(vld1q_f32(lc + 4 * q), out[q][t]));
  }
}

template <int Q>
void tail_block(const float* z, const float* L, const float* lb, float* head_sum, float* lin, int num_frames)
{
  constexpr int T = 2;
  int f = 0;
  for (; f + T <= num_frames; f += T)
    tail_tile<Q, T>(z, L, lb, head_sum, lin, f);
  for (; f < num_frames; f++)
    tail_tile<Q, 1>(z, L, lb, head_sum, lin, f);
}

/// LeakyReLU over a contiguous buffer, as fused's apply_activation does it.
inline void leaky_relu(float* p, int n)
{
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t slope = vdupq_n_f32(kLeakySlope);
  for (int i = 0; i < n; i += 4)
  {
    const float32x4_t x = vld1q_f32(p + i);
    vst1q_f32(p + i, vbslq_f32(vcltq_f32(x, zero), vmulq_f32(x, slope), x));
  }
}

/// Head rechannel conv, H = 1: y = bias + sum_k w_k . x_k.
inline void head_conv_block(const float* const* tap_src, int num_taps, const float* W, float bias, float* out,
                            int num_frames)
{
  constexpr int C = kPadded;
  for (int f = 0; f < num_frames; f++)
  {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int k = 0; k < num_taps; k++)
    {
      const float* src = tap_src[k] + static_cast<size_t>(f) * C;
      const float* w = W + static_cast<size_t>(k) * C;
      acc = vfmaq_f32(acc, vld1q_f32(w), vld1q_f32(src));
    }
    out[f] = vaddvq_f32(acc) + bias;
  }
}

// --- The model ---------------------------------------------------------------

class Pad4Model : public SlimModel
{
public:
  Pad4Model(const std::vector<float>& weights, double sampleRate)
  : SlimModel(weights, sampleRate)
  {
    _rechannel4.assign(kPadded, 0.0f);
    for (int i = 0; i < kChannels; i++)
      _rechannel4[i] = _w.rechannel_w[i];

    for (int li = 0; li < kNumLayers; li++)
    {
      const LayerWeights& L = _w.layers[li];
      PaddedLayer& P = _padded[li];
      const int K = L.kernel_size;

      P.conv_w.assign(static_cast<size_t>(K) * kPadded * kPadded, 0.0f);
      for (int k = 0; k < K; k++)
        for (int j = 0; j < kChannels; j++)
          for (int i = 0; i < kChannels; i++)
            P.conv_w[static_cast<size_t>(k) * 16 + static_cast<size_t>(j) * 4 + i] =
              L.conv_w[static_cast<size_t>(k) * 9 + static_cast<size_t>(j) * 3 + i];

      P.conv_b.assign(kPadded, 0.0f);
      P.mixin_w.assign(kPadded, 0.0f);
      P.l1x1_b.assign(kPadded, 0.0f);
      P.l1x1_w.assign(kPadded * kPadded, 0.0f);
      for (int i = 0; i < kChannels; i++)
      {
        P.conv_b[i] = L.conv_b[i];
        P.mixin_w[i] = L.mixin_w[i];
        P.l1x1_b[i] = L.l1x1_b[i];
        for (int j = 0; j < kChannels; j++)
          P.l1x1_w[static_cast<size_t>(j) * 4 + i] = L.l1x1_w[static_cast<size_t>(j) * 3 + i];
      }
    }

    _head_w4.assign(static_cast<size_t>(kHeadKernelSize) * kPadded, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int j = 0; j < kChannels; j++)
        _head_w4[static_cast<size_t>(k) * kPadded + j] = _w.head_w[k][j];
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
    const float32x4_t rw = vld1q_f32(_rechannel4.data());
    for (int f = 0; f < N; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      vst1q_f32(lin + static_cast<size_t>(f) * kPadded, vmulq_n_f32(rw, x));
    }

    std::memset(_head_sum.data(), 0, static_cast<size_t>(N) * kPadded * sizeof(float));

    const float* tap_src[kHeadKernelSize];
    for (int li = 0; li < kNumLayers; li++)
    {
      Ring& R = _rings[li];
      const LayerWeights& L = _w.layers[li];
      const PaddedLayer& P = _padded[li];
      const int K = L.kernel_size;

      ring_write(R, lin, N);
      for (int k = 0; k < K; k++)
        tap_src[k] = R.data.data() + static_cast<size_t>(tap_base(R, (K - 1 - k) * L.dilation, N)) * kPadded;

      conv_block<1>(tap_src, K, P.conv_w.data(), P.conv_b.data(), P.mixin_w.data(), cond, _z.data(), N);
      leaky_relu(_z.data(), kPadded * N);
      tail_block<1>(_z.data(), P.l1x1_w.data(), P.l1x1_b.data(), _head_sum.data(), lin, N);
    }

    ring_write(_head_ring, _head_sum.data(), N);
    for (int k = 0; k < kHeadKernelSize; k++)
      tap_src[k] = _head_ring.data.data() + static_cast<size_t>(tap_base(_head_ring, kHeadKernelSize - 1 - k, N)) * kPadded;
    head_conv_block(tap_src, kHeadKernelSize, _head_w4.data(), _w.head_b, _head_out.data(), N);

    const float scale = _w.head_scale;
    const float* head_out = _head_out.data();
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(head_out[f] * scale);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);

    const size_t buf = static_cast<size_t>(kPadded) * maxBufferSize;
    _layer_in.assign(buf, 0.0f);
    _head_sum.assign(buf, 0.0f);
    _z.assign(buf, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      reset_ring(_rings[li], _w.layers[li].max_lookback, maxBufferSize);
    reset_ring(_head_ring, kHeadKernelSize - 1, maxBufferSize);
  }

private:
  struct PaddedLayer
  {
    std::vector<float> conv_w;
    std::vector<float> conv_b;
    std::vector<float> mixin_w;
    std::vector<float> l1x1_w;
    std::vector<float> l1x1_b;
  };

  /// a2_fast's ring, four channels wide: pow2 capacity with an eagerly
  /// refreshed tail mirror. Deliberately not fused's lazy-mirror ring — the
  /// only thing this candidate should be changing is the kernel and the width.
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
    R.data.assign(static_cast<size_t>(kPadded) * (R.pow2 + maxBufferSize), 0.0f);
    R.write_pos = max_lookback;
  }

  static int tap_base(const Ring& R, int lookback, int num_frames)
  {
    return (R.write_pos - num_frames - lookback) & R.mask;
  }

  void ring_write(Ring& R, const float* src, int num_frames)
  {
    const int mbs = GetMaxBufferSize();
    float* const hist = R.data.data();
    const int wp = R.write_pos;
    const int first = std::min(num_frames, R.pow2 - wp);
    std::memcpy(hist + static_cast<size_t>(wp) * kPadded, src, static_cast<size_t>(first) * kPadded * sizeof(float));
    if (first < num_frames)
    {
      std::memcpy(hist, src + static_cast<size_t>(first) * kPadded,
                  static_cast<size_t>(num_frames - first) * kPadded * sizeof(float));
    }
    std::memcpy(hist + static_cast<size_t>(R.pow2) * kPadded, hist,
                static_cast<size_t>(mbs) * kPadded * sizeof(float));
    R.write_pos = (wp + num_frames) & R.mask;
  }

  std::vector<float> _rechannel4;
  std::array<PaddedLayer, kNumLayers> _padded;
  std::vector<float> _head_w4;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _z;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_pad4(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<Pad4Model>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
