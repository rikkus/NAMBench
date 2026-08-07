// Kernel 1: restrict — the free-lunch check.
//
// `baseline` with three changes and no restructuring at all:
//   - `__restrict` on the pointers into history / z / head_sum / layer_in, so
//     the compiler is told what it cannot prove: that a store to z can never
//     alias the history it is reading from;
//   - the std::vector element accesses hoisted to raw `.data()` pointers
//     outside the frame loops, so the loop body has no vector indirection left;
//   - `#pragma clang fp contract(fast)` across the whole file, allowing
//     contraction across statements rather than only within one.
//
// The last of those is the only change that can alter arithmetic, and it can
// only ever *add* fused multiply-adds where there were separate rounding steps;
// a2_fast already gets FMAs for every `a0 += w * s` because each is a single
// statement. Expected result: nothing, or nearly nothing. Worth one file to
// find out, because if it is not nothing then every later measurement is partly
// measuring aliasing rather than the idea under test.

#if defined(NB_ENABLE_SLIM_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "slim_common.h"

  #pragma clang fp contract(fast)

namespace slimlab
{
namespace
{

class RestrictModel : public SlimModel
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
    _z.assign(static_cast<size_t>(3) * maxBufferSize, 0.0f);
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

  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights& L, const float* __restrict cond, int num_frames)
  {
    constexpr int K = KernelSize;
    const int D = L.dilation;
    const int mask = R.mask;
    const float* __restrict hist = R.data.data();
    const float* __restrict conv_w = L.conv_w.data();
    float* __restrict z = _z.data();
    auto tap_base_phys = [&](int taps_back) { return (R.write_pos - num_frames - taps_back * D) & mask; };

    {
      const float* __restrict wk = conv_w;
      const float* __restrict src0 = hist + static_cast<size_t>(tap_base_phys(K - 1)) * 3;
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2];
      for (int f = 0; f < num_frames; f++)
      {
        const float s0 = src0[f * 3 + 0], s1 = src0[f * 3 + 1], s2 = src0[f * 3 + 2];
        float a0 = cb0 + w0 * s0;
        float a1 = cb1 + w1 * s0;
        float a2 = cb2 + w2 * s0;
        a0 += w3 * s1;
        a1 += w4 * s1;
        a2 += w5 * s1;
        a0 += w6 * s2;
        a1 += w7 * s2;
        a2 += w8 * s2;
        z[f * 3 + 0] = a0;
        z[f * 3 + 1] = a1;
        z[f * 3 + 2] = a2;
      }
    }

    for (int k = 1; k < K - 1; k++)
    {
      const float* __restrict wk = conv_w + static_cast<size_t>(k) * 9;
      const float* __restrict src0 = hist + static_cast<size_t>(tap_base_phys(K - 1 - k)) * 3;
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float s0 = src0[f * 3 + 0], s1 = src0[f * 3 + 1], s2 = src0[f * 3 + 2];
        float a0 = z[f * 3 + 0] + w0 * s0;
        float a1 = z[f * 3 + 1] + w1 * s0;
        float a2 = z[f * 3 + 2] + w2 * s0;
        a0 += w3 * s1;
        a1 += w4 * s1;
        a2 += w5 * s1;
        a0 += w6 * s2;
        a1 += w7 * s2;
        a2 += w8 * s2;
        z[f * 3 + 0] = a0;
        z[f * 3 + 1] = a1;
        z[f * 3 + 2] = a2;
      }
    }

    const float* __restrict wk_last = conv_w + static_cast<size_t>(K - 1) * 9;
    const float* __restrict src_last = hist + static_cast<size_t>(tap_base_phys(0)) * 3;
    float* __restrict head_sum = _head_sum.data();
    float* __restrict layer_in = _layer_in.data();
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2];
    const float cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2];
    const float lw10 = L.l1x1_w[3], lw11 = L.l1x1_w[4], lw12 = L.l1x1_w[5];
    const float lw20 = L.l1x1_w[6], lw21 = L.l1x1_w[7], lw22 = L.l1x1_w[8];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2];
    for (int f = 0; f < num_frames; f++)
    {
      const float s0 = src_last[f * 3 + 0], s1 = src_last[f * 3 + 1], s2 = src_last[f * 3 + 2];
      float a0 = z[f * 3 + 0] + cw0 * s0;
      float a1 = z[f * 3 + 1] + cw1 * s0;
      float a2 = z[f * 3 + 2] + cw2 * s0;
      a0 += cw3 * s1;
      a1 += cw4 * s1;
      a2 += cw5 * s1;
      a0 += cw6 * s2;
      a1 += cw7 * s2;
      a2 += cw8 * s2;
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      head_sum[f * 3 + 0] += a0;
      head_sum[f * 3 + 1] += a1;
      head_sum[f * 3 + 2] += a2;
      layer_in[f * 3 + 0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2;
      layer_in[f * 3 + 1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2;
      layer_in[f * 3 + 2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2;
    }
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
  std::vector<float> _z;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_restrict(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<RestrictModel>(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
