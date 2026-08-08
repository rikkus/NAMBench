// Kernel 1: fu_baseline — the control for the fu* family.
//
// A verbatim port of the fused engine's C=8 path: the same pow2 + lazy-mirror
// ring, the same Q=2 register-tiled kernels at their shipped tile widths
// (T=4 for the conv, T=2 for the tail), the same three-pass structure through
// the _z buffer, the same scalar-per-frame head, the same skip of the final
// layer1x1.
//
// Specialised to the A2 shape's single layer array, which removes the array
// loop and the lin/alt swap but changes no arithmetic: with one array the
// rechannel always reads `cond`, `head_sum` is always memset rather than seeded
// from a previous array's head output, and `lin` is never swapped.
//
// Nothing here is meant to be fast. Its whole job is to land on top of `fused`'s
// number: if it does not, the lab is measuring its own overhead and none of the
// fu* candidates mean anything.

#if defined(NB_ENABLE_FULL_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include <arm_neon.h>

  #include "full_common.h"

namespace fulllab
{
namespace
{

constexpr int C = kChannels;
constexpr int Q = C / 4; // 2

// -----------------------------------------------------------------------------
// Ring buffer: power-of-2 capacity with a mirrored tail, refreshed lazily —
// only for the blocks where a read actually wraps, which keeps the common case
// copy-free. This is fused's Ring, not a2_fast's (which mirrors eagerly every
// block).
// -----------------------------------------------------------------------------
struct Ring
{
  std::vector<float> data;
  int pow2 = 0;
  int mask = 0;
  int wpos = 0;
  int mbs = 0;

  void reset(int max_lookback, int max_buffer)
  {
    mbs = max_buffer;
    pow2 = next_pow2(max_lookback + max_buffer);
    mask = pow2 - 1;
    data.assign(static_cast<size_t>(C) * (pow2 + max_buffer), 0.0f);
    wpos = max_lookback & mask;
  }

  void write(const float* src, int n)
  {
    float* h = data.data();
    const int first = std::min(n, pow2 - wpos);
    std::memcpy(h + static_cast<size_t>(wpos) * C, src, static_cast<size_t>(first) * C * sizeof(float));
    if (first < n)
    {
      std::memcpy(h, src + static_cast<size_t>(first) * C, static_cast<size_t>(n - first) * C * sizeof(float));
    }
    wpos = (wpos + n) & mask;
  }

  void mirror(int cols)
  {
    float* h = data.data();
    std::memcpy(h + static_cast<size_t>(pow2) * C, h, static_cast<size_t>(cols) * C * sizeof(float));
  }

  const float* tap(int lookback_frames, int n)
  {
    const int base = (wpos - n - lookback_frames) & mask;
    const int overflow = base + n - pow2;
    if (overflow > 0)
      mirror(overflow);
    return data.data() + static_cast<size_t>(base) * C;
  }
};

// -----------------------------------------------------------------------------
// Register-tiled NEON kernels. A vector spans four *channels*; the tile spans T
// frames. Accumulators live in registers across all taps; the reduction for one
// output element is therefore a single chain over every tap and every input
// channel, seeded with the bias — which is what makes fused's arithmetic differ
// from a2_fast's.
// -----------------------------------------------------------------------------

template <int T, int LANE>
inline void conv_lane(float32x4_t (&acc)[Q][T], const float32x4_t (&iv)[T], const float* wc)
{
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(wc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      acc[q][t] = vfmaq_laneq_f32(acc[q][t], w[q], iv[t], LANE);
}

template <int T>
inline void conv_tile(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                      const float* mixin, const float* cond, float* z, int f0)
{
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
      conv_lane<T, 0>(acc, iv, wc);
      conv_lane<T, 1>(acc, iv, wc);
      conv_lane<T, 2>(acc, iv, wc);
      conv_lane<T, 3>(acc, iv, wc);
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

void conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias, const float* mixin,
                const float* cond, float* z, int num_frames)
{
  constexpr int T = 4; // fused's `(Q <= 4) ? 4 : 2` at Q=2
  int f = 0;
  for (; f + T <= num_frames; f += T)
    conv_tile<T>(tap_src, num_taps, W, bias, mixin, cond, z, f);
  for (; f < num_frames; f++)
    conv_tile<1>(tap_src, num_taps, W, bias, mixin, cond, z, f);
}

/// LeakyReLU over a contiguous buffer. n is always a multiple of 4.
inline void apply_leaky_relu(float* p, int n)
{
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t slope = vdupq_n_f32(kLeakySlope);
  for (int i = 0; i < n; i += 4)
  {
    const float32x4_t x = vld1q_f32(p + i);
    vst1q_f32(p + i, vbslq_f32(vcltq_f32(x, zero), vmulq_f32(x, slope), x));
  }
}

template <int T, int LANE>
inline void tail_lane(float32x4_t (&out)[Q][T], const float32x4_t (&a)[Q][T], const float* lc, int j4)
{
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(lc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      out[q][t] = vfmaq_laneq_f32(out[q][t], w[q], a[j4][t], LANE);
}

template <int T>
inline void tail_tile(const float* z, const float* L, const float* lb, bool has_l1x1, float* head_sum, float* lin,
                      int f0)
{
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
  if (!has_l1x1)
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
    float* lc = lin + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(lc + 4 * q, vaddq_f32(vld1q_f32(lc + 4 * q), out[q][t]));
  }
}

void tail_block(const float* z, const float* L, const float* lb, bool has_l1x1, float* head_sum, float* lin,
                int num_frames)
{
  constexpr int T = 2; // fused's `(Q <= 4) ? 2 : 1` at Q=2
  int f = 0;
  for (; f + T <= num_frames; f += T)
    tail_tile<T>(z, L, lb, has_l1x1, head_sum, lin, f);
  for (; f < num_frames; f++)
    tail_tile<1>(z, L, lb, has_l1x1, head_sum, lin, f);
}

/// Rechannel (1x1, no bias) from the single condition channel.
void rechannel_block(const float* W, const float* x, float* out, int num_frames)
{
  for (int f = 0; f < num_frames; f++)
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

/// Head rechannel conv, H = 1: one accumulator per frame across all 16 taps and
/// both channel groups, finished with a pairwise vaddvq_f32 and then the bias.
void head_conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias, float* out,
                     int num_frames)
{
  for (int f = 0; f < num_frames; f++)
  {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int k = 0; k < num_taps; k++)
    {
      const float* src = tap_src[k] + static_cast<size_t>(f) * C;
      const float* w = W + static_cast<size_t>(k) * C;
      for (int q = 0; q < Q; q++)
        acc = vfmaq_f32(acc, vld1q_f32(w + 4 * q), vld1q_f32(src + 4 * q));
    }
    float y = vaddvq_f32(acc);
    if (bias != nullptr)
      y += bias[0];
    out[f] = y;
  }
}

class FuBaselineModel : public FullModel
{
public:
  FuBaselineModel(const std::vector<float>& weights, double sampleRate)
  : FullModel(weights, sampleRate)
  {
    // fused flattens the head weights to [k * C + j]; the canonical form is
    // head_w[k][j], which is the same bytes in the same order.
    _head_w.assign(static_cast<size_t>(kHeadKernelSize) * C, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int j = 0; j < C; j++)
        _head_w[static_cast<size_t>(k) * C + j] = _w.head_w[k][j];
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    float* cond = _cond.data();
    const NAM_SAMPLE* in0 = input[0];
    for (int f = 0; f < N; f++)
      cond[f] = static_cast<float>(in0[f]);

    float* lin = _lin.data();
    const float* tap_src[kHeadKernelSize > 15 ? kHeadKernelSize : 15];

    rechannel_block(_w.rechannel_w.data(), cond, lin, N);

    std::memset(_head_sum.data(), 0, static_cast<size_t>(C) * N * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
    {
      const LayerWeights& L = _w.layers[li];
      Ring& R = _rings[li];
      const int K = L.kernel_size;

      R.write(lin, N);
      for (int k = 0; k < K; k++)
        tap_src[k] = R.tap((K - 1 - k) * L.dilation, N);

      conv_block(tap_src, K, L.conv_w.data(), L.conv_b.data(), L.mixin_w.data(), cond, _z.data(), N);
      apply_leaky_relu(_z.data(), C * N);
      // The residual output of the very last layer feeds nothing, so its
      // layer1x1 GEMM is skipped — fused's skip_l1x1_output.
      const bool do_l1x1 = (li != kNumLayers - 1);
      tail_block(_z.data(), L.l1x1_w.data(), L.l1x1_b.data(), do_l1x1, _head_sum.data(), lin, N);
    }

    _head_ring.write(_head_sum.data(), N);
    for (int k = 0; k < kHeadKernelSize; k++)
      tap_src[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);
    head_conv_block(tap_src, kHeadKernelSize, _head_w.data(), &_w.head_b, _head_out.data(), N);

    const float scale = _w.head_scale;
    NAM_SAMPLE* out0 = output[0];
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
  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _head_w;
  std::vector<float> _cond;
  std::vector<float> _lin;
  std::vector<float> _z;
  std::vector<float> _head_sum;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_fu_baseline(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<FuBaselineModel>(weights, sampleRate);
}

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
