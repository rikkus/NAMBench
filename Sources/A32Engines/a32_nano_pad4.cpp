// Kernel 15: n_pad4 — pad three channels to four and vectorise across channels.
//
// The other way to fill a Q register. Planar fills it with four frames of one
// channel; this fills it with the four (three real, one padded) channels of one
// frame, which is a2_fast's own layout with a lane of zeroes bolted on.
//
// It lost on an M2 (0.835×), and it is written again here for one reason: on
// AArch64 the channel-major shape pays for lane-addressed weights it can no
// longer amortise, and ARMv7 has no by-element FMA to lose in the first place.
// Every weight arrives through vld1q_dup either way, so the comparison is
// closer here than the AArch64 result suggests — worth one file to find out.
//
// Measured: 0.821x against a2_fast, so it lost here too, and by about as much as
// on AArch64 (0.835x). The by-element FMA was not what decided it. What decides
// it is that this shape does three vector FMAs per frame where planar does nine
// per four frames, so it runs 1.33 vector FMAs per frame against planar's 2.25 —
// and the head, which stays scalar here for exactness, is then a larger share of
// what is left.
//
// Exactness. The padding is only on the *output* side: the accumulator's four
// lanes are output channels 0,1,2 and a dead lane 3, and the reduction loops
// over the three real input channels. Lane i therefore runs
//
//   a_i = conv_b[i]; for k: for j in 0..2: a_i += w_k[j*3+i] * src[j]
//
// which is a2_fast's chain, in a2_fast's order, one rounding per step. Nothing
// is summed across lanes at any point, so no lane can leak into another. The
// pad lane's weights are all zero, so it holds a hard zero for the whole run —
// note that it is never *added* to anything real, which is what keeps the
// signed-zero cases identical too. This kernel claims exactness, and the
// conformance run tests the claim.

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

constexpr int C = 3;
constexpr int P = 4; // padded width: one Q register per frame
/// Frames per tile. Four accumulators, four loaded frames, one weight vector —
/// nine of sixteen registers, with the layer tail's needs still to come.
constexpr int kTile = 4;

class NanoPad4Model : public A32Model<C>
{
public:
  NanoPad4Model(const std::vector<float>& weights, double sampleRate)
  : A32Model<C>(weights, sampleRate)
  {
    // Re-lay the weights once, at construction: [tap][input j][output lane].
    for (int li = 0; li < kNumLayers; li++)
    {
      const LayerWeights<C>& L = _w.layers[li];
      Padded& Q = _p[li];
      Q.kernel_size = L.kernel_size;
      Q.dilation = L.dilation;
      Q.max_lookback = L.max_lookback;

      Q.conv_w.assign(static_cast<size_t>(L.kernel_size) * C * P, 0.0f);
      for (int k = 0; k < L.kernel_size; k++)
        for (int j = 0; j < C; j++)
          for (int i = 0; i < C; i++)
            Q.conv_w[(static_cast<size_t>(k) * C + j) * P + i] = L.conv_w[static_cast<size_t>(k) * C * C + j * C + i];

      for (int i = 0; i < C; i++)
      {
        Q.conv_b[i] = L.conv_b[i];
        Q.mixin_w[i] = L.mixin_w[i];
        Q.l1x1_b[i] = L.l1x1_b[i];
        for (int j = 0; j < C; j++)
          Q.l1x1_w[j * P + i] = L.l1x1_w[j * C + i];
      }
    }
  }

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
      float* lin = &_layer_in[static_cast<size_t>(f) * P];
      for (int c = 0; c < C; c++)
        lin[c] = _w.rechannel_w[c] * x;
      lin[C] = 0.0f;
    }

    std::memset(_head_sum.data(), 0, static_cast<size_t>(num_frames) * P * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
    {
      Ring& R = _rings[li];
      R.prepare(num_frames);
      std::memcpy(R.write_ptr(), _layer_in.data(),
                  static_cast<size_t>(num_frames) * P * sizeof(float));
      R.commit(num_frames);

      if (_p[li].kernel_size == 6)
        layer_forward<6>(R, _p[li], cond, num_frames);
      else
        layer_forward<15>(R, _p[li], cond, num_frames);
    }

    head_forward(_head_out.data(), num_frames);
    for (int f = 0; f < num_frames; f++)
      out0[f] = static_cast<NAM_SAMPLE>(_head_out[f]);
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    nam::DSP::SetMaxBufferSize(maxBufferSize);

    _layer_in.assign(static_cast<size_t>(P) * maxBufferSize, 0.0f);
    _head_sum.assign(static_cast<size_t>(P) * maxBufferSize, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_p[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  using Ring = ChannelRing<P, RingKind::Pow2Eager>;

  /// The A2 weights in padded, lane-per-output form.
  struct Padded
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0;
    std::vector<float> conv_w;   ///< [tap][input j][lane i], P floats per (k, j)
    std::array<float, P> conv_b{};
    std::array<float, P> mixin_w{};
    std::array<float, C * P> l1x1_w{};
    std::array<float, P> l1x1_b{};
  };

  template <int KernelSize>
  void layer_forward(Ring& R, const Padded& Q, const float* cond, int num_frames)
  {
    constexpr int K = KernelSize;

    const float* tap[K];
    for (int k = 0; k < K; k++)
      tap[k] = R.tap((K - 1 - k) * Q.dilation, num_frames);

    int f = 0;
    for (; f + kTile <= num_frames; f += kTile)
      tile<K>(Q, tap, cond, f);
    for (; f < num_frames; f++)
      tile1<K>(Q, tap, cond, f);
  }

  /// kTile frames, one accumulator each. Each frame's four channels are one
  /// load; a weight column is one vld1q; the input scalar is a lane broadcast.
  template <int K>
  NB_A32_INLINE void tile(const Padded& Q, const float* const (&tap)[K], const float* cond, int f0)
  {
    float32x4_t a[kTile];
    const float32x4_t cb = vld1q_f32(Q.conv_b.data());
    for (int t = 0; t < kTile; t++)
      a[t] = cb;

    for (int k = 0; k < K; k++)
    {
      const float* wk = &Q.conv_w[static_cast<size_t>(k) * C * P];
      const float* src = tap[k] + static_cast<size_t>(f0) * P;
      for (int t = 0; t < kTile; t++)
      {
        const float32x4_t s = vld1q_f32(src + static_cast<size_t>(t) * P);
        a[t] = compat::fmaq_laneq<0>(a[t], vld1q_f32(wk + 0 * P), s);
        a[t] = compat::fmaq_laneq<1>(a[t], vld1q_f32(wk + 1 * P), s);
        a[t] = compat::fmaq_laneq<2>(a[t], vld1q_f32(wk + 2 * P), s);
      }
    }

    post<K>(Q, tap, cond, f0, a, kTile);
  }

  template <int K>
  void tile1(const Padded& Q, const float* const (&tap)[K], const float* cond, int f)
  {
    float32x4_t a[1] = {vld1q_f32(Q.conv_b.data())};
    for (int k = 0; k < K; k++)
    {
      const float* wk = &Q.conv_w[static_cast<size_t>(k) * C * P];
      const float32x4_t s = vld1q_f32(tap[k] + static_cast<size_t>(f) * P);
      a[0] = compat::fmaq_laneq<0>(a[0], vld1q_f32(wk + 0 * P), s);
      a[0] = compat::fmaq_laneq<1>(a[0], vld1q_f32(wk + 1 * P), s);
      a[0] = compat::fmaq_laneq<2>(a[0], vld1q_f32(wk + 2 * P), s);
    }
    post<K>(Q, tap, cond, f, a, 1);
  }

  /// Mixin, LeakyReLU, head_sum and the layer1x1 residual, in a2_fast's order.
  template <int K>
  NB_A32_INLINE void post(const Padded& Q, const float* const (&tap)[K], const float* cond, int f0,
                          float32x4_t* a, int n)
  {
    const float32x4_t M = vld1q_f32(Q.mixin_w.data());
    const float32x4_t LB = vld1q_f32(Q.l1x1_b.data());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);

    for (int t = 0; t < n; t++)
    {
      const int f = f0 + t;
      a[t] = compat::fmaq_bcast(a[t], M, cond + f);
      a[t] = vbslq_f32(vcltq_f32(a[t], zero), vmulq_f32(a[t], slope), a[t]);

      float* hs = &_head_sum[static_cast<size_t>(f) * P];
      vst1q_f32(hs, vaddq_f32(vld1q_f32(hs), a[t]));

      float32x4_t o = LB;
      o = compat::fmaq_laneq<0>(o, vld1q_f32(&Q.l1x1_w[0 * P]), a[t]);
      o = compat::fmaq_laneq<1>(o, vld1q_f32(&Q.l1x1_w[1 * P]), a[t]);
      o = compat::fmaq_laneq<2>(o, vld1q_f32(&Q.l1x1_w[2 * P]), a[t]);

      const float32x4_t res = vld1q_f32(tap[K - 1] + static_cast<size_t>(f) * P);
      vst1q_f32(&_layer_in[static_cast<size_t>(f) * P], vaddq_f32(res, o));
    }
  }

  /// The head stays scalar, as a2_fast has it: it reduces across channels, and
  /// summing three lanes of a Q register would reassociate exactly the chain
  /// this kernel is claiming to reproduce.
  void head_forward(float* out, int num_frames)
  {
    _head_ring.prepare(num_frames);
    std::memcpy(_head_ring.write_ptr(), _head_sum.data(),
                static_cast<size_t>(num_frames) * P * sizeof(float));
    _head_ring.commit(num_frames);

    const float* hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, num_frames);

    for (int f = 0; f < num_frames; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* src = hb[k] + static_cast<size_t>(f) * P;
        const float* wk = _w.head_w[k].data();
        for (int j = 0; j < C; j++)
          y += wk[j] * src[j];
      }
      out[f] = y * _w.head_scale;
    }
  }

  std::array<Padded, kNumLayers> _p;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_pad4(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<NanoPad4Model>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
