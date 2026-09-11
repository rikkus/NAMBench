// Kernel 28: s_chanmajor — what does bit-identity cost at C=8?
//
// NB_A32_NOT_EXACT.
//
// The one candidate in this round that is allowed to compute different numbers,
// and it exists to bound what the exactness rule is costing. Two channels of a
// Q register — eight channels is exactly two Q registers per frame — with the
// reduction running across lanes in a2_fast's *storage* order rather than
// Eigen's blocking order.
//
// Why it cannot be exact, by construction. Eigen computes each tap's product on
// its own and adds the result into z (see the Reference note in
// a32_planar_kernel.h); this accumulates every tap into one chain, which is
// candidate C in Scripts/eigen-order-probe and differs from Eigen at 3.75e6 ULP
// in the probe's own measurement. That is not a bug to be fixed — it is the
// shape the (now-retired) `fused` engine used on AArch64, and the question is
// what it buys.
//
// What it should buy: the per-tap partial array disappears, which halves the
// live accumulator count. That is the difference between 18 registers and 10 at
// a 4-frame tile, i.e. between spilling and not.
//
// Measured: 1.450x against a2_fast, where the best *exact* kernel at this width
// reaches 1.305x. So bit-identity costs about 11% at C=8 — worth putting next to
// the C=3 answer, where the inexact candidate was 40% *slower* than its exact
// twin. The two are not in tension: there, abandoning exactness bought nothing
// structural and only added roundings; here it removes an entire live array from
// the inner loop. What exactness costs is a property of the reference's shape,
// not a constant.
//
// Note this kernel keeps the control's scalar head, so the number above is the
// layer body alone. s_head_tile's 1.020x would compose on top of it.

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "a32_common.h"
  #include "a32_neon_compat.h"
  #include "a32_ring.h"

namespace a32lab
{
namespace
{

constexpr int C = 8;
constexpr int Q = C / 4; // Q registers per frame
/// Frames per tile: acc 2 per frame, plus a weight column (2) and a broadcast.
constexpr int kTile = 4;

class StdChanMajorModel : public A32Model<C>
{
public:
  using A32Model<C>::A32Model;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    float* cond = _cond.data();
    for (int f = 0; f < num_frames; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      float* lin = &_layer_in[static_cast<size_t>(f) * C];
      for (int c = 0; c < C; c++)
        lin[c] = _w.rechannel_w[c] * x;
    }

    std::memset(_head_sum.data(), 0, static_cast<size_t>(num_frames) * C * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
    {
      Ring& R = _rings[li];
      const LayerWeights<C>& L = _w.layers[li];
      R.prepare(num_frames);
      std::memcpy(R.write_ptr(), _layer_in.data(),
                  static_cast<size_t>(num_frames) * C * sizeof(float));
      R.commit(num_frames);

      if (L.kernel_size == 6)
        layer_forward<6>(R, L, cond, num_frames);
      else
        layer_forward<15>(R, L, cond, num_frames);
    }

    head_forward(_head_out.data(), num_frames);
    for (int f = 0; f < num_frames; f++)
      out0[f] = static_cast<NAM_SAMPLE>(_head_out[f]);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);

    _layer_in.assign(static_cast<size_t>(C) * maxBufferSize, 0.0f);
    _head_sum.assign(static_cast<size_t>(C) * maxBufferSize, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_w.layers[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  using Ring = ChannelRing<C, RingKind::Pow2Eager>;

  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights<C>& L, const float* cond, int num_frames)
  {
    constexpr int K = KernelSize;

    const float* tap[K];
    for (int k = 0; k < K; k++)
      tap[k] = R.tap((K - 1 - k) * L.dilation, num_frames);

    int f = 0;
    for (; f + kTile <= num_frames; f += kTile)
      tile<K, kTile>(L, tap, cond, f);
    for (; f < num_frames; f++)
      tile<K, 1>(L, tap, cond, f);
  }

  /// T frames, two accumulators each. One chain across every tap — this is the
  /// reassociation.
  template <int K, int T>
  NB_A32_INLINE void tile(const LayerWeights<C>& L, const float* const (&tap)[K],
                          const float* cond, int f0)
  {
    float32x4_t a[T][Q];
    for (int t = 0; t < T; t++)
      for (int q = 0; q < Q; q++)
        a[t][q] = vld1q_f32(&L.conv_b[4 * q]);

    for (int k = 0; k < K; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * C * C];
      const float* src = tap[k] + static_cast<size_t>(f0) * C;
      for (int j = 0; j < C; j++)
      {
        const float32x4_t w0 = vld1q_f32(wk + j * C);
        const float32x4_t w1 = vld1q_f32(wk + j * C + 4);
        for (int t = 0; t < T; t++)
        {
          const float* x = src + static_cast<size_t>(t) * C + j;
          a[t][0] = compat::fmaq_bcast(a[t][0], w0, x);
          a[t][1] = compat::fmaq_bcast(a[t][1], w1, x);
        }
      }
    }

    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    for (int t = 0; t < T; t++)
    {
      const int f = f0 + t;
      for (int q = 0; q < Q; q++)
      {
        a[t][q] = compat::fmaq_bcast(a[t][q], vld1q_f32(&L.mixin_w[4 * q]), cond + f);
        a[t][q] = vbslq_f32(vcltq_f32(a[t][q], zero), vmulq_f32(a[t][q], slope), a[t][q]);
      }

      float* hs = &_head_sum[static_cast<size_t>(f) * C];
      for (int q = 0; q < Q; q++)
        vst1q_f32(hs + 4 * q, vaddq_f32(vld1q_f32(hs + 4 * q), a[t][q]));

      float32x4_t o[Q];
      for (int q = 0; q < Q; q++)
        o[q] = vld1q_f32(&L.l1x1_b[4 * q]);
      alignas(16) float av[C];
      for (int q = 0; q < Q; q++)
        vst1q_f32(av + 4 * q, a[t][q]);
      for (int j = 0; j < C; j++)
        for (int q = 0; q < Q; q++)
          o[q] = compat::fmaq_bcast(o[q], vld1q_f32(&L.l1x1_w[j * C + 4 * q]), av + j);

      const float* res = tap[K - 1] + static_cast<size_t>(f) * C;
      float* lin = &_layer_in[static_cast<size_t>(f) * C];
      for (int q = 0; q < Q; q++)
        vst1q_f32(lin + 4 * q, vaddq_f32(vld1q_f32(res + 4 * q), o[q]));
    }
  }

  /// Unchanged from the control: this kernel is about the layer body, and
  /// leaving the head alone is what keeps the comparison to one variable.
  void head_forward(float* out, int num_frames)
  {
    _head_ring.prepare(num_frames);
    std::memcpy(_head_ring.write_ptr(), _head_sum.data(),
                static_cast<size_t>(num_frames) * C * sizeof(float));
    _head_ring.commit(num_frames);

    const float* hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, num_frames);

    for (int f = 0; f < num_frames; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* src = hb[k] + static_cast<size_t>(f) * C;
        const float* wk = _w.head_w[k].data();
        for (int j = 0; j < C; j++)
          y += wk[j] * src[j];
      }
      out[f] = y * _w.head_scale;
    }
  }

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_std_chanmajor(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<StdChanMajorModel>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
