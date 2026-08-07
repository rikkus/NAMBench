// Kernel 2: framemajor — kill the z round-trips, widen the dependency chain.
//
// Two problems with the baseline, both addressed by loop restructuring alone —
// no intrinsics, no layout change:
//
//   1. `z` is a heap buffer that every tap reads and writes. At K=6 that is 30
//      loads and stores of z per frame per layer, against only 18 history
//      loads. Turning the nest inside out — frame outer, tap inner — lets z
//      live in registers for the whole life of a frame, so it never reaches
//      memory at all.
//
//   2. The baseline's `a0 += w * s` chain offers three independent FMA chains
//      (one per output channel). Four 4-cycle FMA pipes want around sixteen.
//      Splitting the reduction into nine partial accumulators — one per
//      (output, input) pair — and processing two frames per tile gives
//      eighteen.
//
// The second change is not free: summing nine partials at the end reassociates
// the reduction, so this kernel is *not* bit-identical to a2_fast even though
// it computes the same thing. That is exactly what the parity check is for, and
// why the write-up reports parity next to speed for every candidate.

#if defined(NB_ENABLE_SLIM_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "slim_common.h"

namespace slimlab
{
namespace
{

class FrameMajorModel : public SlimModel
{
public:
  using SlimModel::SlimModel;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    float* __restrict cond = _cond.data();
    float* __restrict layer_in = _layer_in.data();
    const float rw0 = _w.rechannel_w[0], rw1 = _w.rechannel_w[1], rw2 = _w.rechannel_w[2];
    for (int f = 0; f < num_frames; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      layer_in[f * 3 + 0] = rw0 * x;
      layer_in[f * 3 + 1] = rw1 * x;
      layer_in[f * 3 + 2] = rw2 * x;
    }

    std::memset(_head_sum.data(), 0, static_cast<size_t>(num_frames) * 3 * sizeof(float));

    for (int li = 0; li < kNumLayers; li++)
    {
      Ring& R = _rings[li];
      const LayerWeights& L = _w.layers[li];
      ring_write(R, layer_in, num_frames);
      if (L.kernel_size == 6)
        layer_forward<6>(R, L, cond, num_frames);
      else
        layer_forward<15>(R, L, cond, num_frames);
    }

    head_forward(_head_out.data(), num_frames);
    const float* __restrict head_out = _head_out.data();
    for (int f = 0; f < num_frames; f++)
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

  /// T frames per tile, nine partial accumulators each: 9 × T independent FMA
  /// chains through the whole tap loop. z never leaves registers.
  template <int K, int T>
  void tile(const float* __restrict hist, const int (&tap_base)[K], const LayerWeights& L,
            const float* __restrict cond, int f0)
  {
    const float* __restrict conv_w = L.conv_w.data();

    // p[i][j][t]: the running sum of contributions to output i from input j.
    float p[3][3][T];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int t = 0; t < T; t++)
          p[i][j][t] = 0.0f;

    float s_last[3][T];

  #pragma clang loop unroll(full)
    for (int k = 0; k < K; k++)
    {
      const float* __restrict wk = conv_w + static_cast<size_t>(k) * 9;
      const float* __restrict src = hist + static_cast<size_t>(tap_base[k] + f0) * 3;
      for (int t = 0; t < T; t++)
      {
        const float s0 = src[t * 3 + 0], s1 = src[t * 3 + 1], s2 = src[t * 3 + 2];
        p[0][0][t] += wk[0] * s0;
        p[1][0][t] += wk[1] * s0;
        p[2][0][t] += wk[2] * s0;
        p[0][1][t] += wk[3] * s1;
        p[1][1][t] += wk[4] * s1;
        p[2][1][t] += wk[5] * s1;
        p[0][2][t] += wk[6] * s2;
        p[1][2][t] += wk[7] * s2;
        p[2][2][t] += wk[8] * s2;
        if (k == K - 1)
        {
          s_last[0][t] = s0;
          s_last[1][t] = s1;
          s_last[2][t] = s2;
        }
      }
    }

    float* __restrict head_sum = _head_sum.data();
    float* __restrict layer_in = _layer_in.data();
    for (int t = 0; t < T; t++)
    {
      const float cf = cond[f0 + t];
      float a[3];
      for (int i = 0; i < 3; i++)
      {
        float v = L.conv_b[i] + p[i][0][t] + p[i][1][t] + p[i][2][t];
        v += L.mixin_w[i] * cf;
        a[i] = (v >= 0.0f) ? v : v * kLeakySlope;
      }

      const size_t o = static_cast<size_t>(f0 + t) * 3;
      for (int i = 0; i < 3; i++)
      {
        head_sum[o + i] += a[i];
        layer_in[o + i] =
          s_last[i][t] + (L.l1x1_b[i] + L.l1x1_w[0 + i] * a[0] + L.l1x1_w[3 + i] * a[1] + L.l1x1_w[6 + i] * a[2]);
      }
    }
  }

  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights& L, const float* __restrict cond, int num_frames)
  {
    constexpr int K = KernelSize;
    const int D = L.dilation;
    const int mask = R.mask;
    const float* __restrict hist = R.data.data();

    int tap_base[K];
    for (int k = 0; k < K; k++)
      tap_base[k] = (R.write_pos - num_frames - (K - 1 - k) * D) & mask;

    int f = 0;
    for (; f + 2 <= num_frames; f += 2)
      tile<K, 2>(hist, tap_base, L, cond, f);
    for (; f < num_frames; f++)
      tile<K, 1>(hist, tap_base, L, cond, f);
  }

  void head_forward(float* __restrict out, int num_frames)
  {
    ring_write(_head_ring, _head_sum.data(), num_frames);
    const int mask = _head_ring.mask;
    const int base = _head_ring.write_pos - num_frames;
    const float* __restrict hist = _head_ring.data.data();
    const float scale = _w.head_scale;
    const float bias = _w.head_b;
    for (int f = 0; f < num_frames; f++)
    {
      float y = bias;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const int col = (base + f - (kHeadKernelSize - 1 - k)) & mask;
        const float* __restrict src = hist + static_cast<size_t>(col) * 3;
        const float* __restrict wk = _w.head_w[k].data();
        for (int b = 0; b < 3; b++)
          y += wk[b] * src[b];
      }
      out[f] = y * scale;
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

std::unique_ptr<nam::DSP> make_framemajor(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<FrameMajorModel>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
