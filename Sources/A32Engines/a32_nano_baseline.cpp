// Kernel 0: n_baseline — the control for the C=3 round.
//
// A verbatim port of a2_fast's `if constexpr (Channels == 3)` branch, along with
// the ring buffers, buffer sizing and process() around it, at ring mode 1 (pow2
// + eager tail mirror), which is a2_fast's default.
//
// Nothing here is meant to be fast. Its whole job is to land on top of
// `upstream`'s number on this target: if it does not, the lab is measuring its
// own overhead and none of the other candidates mean anything.
//
// This is plain scalar C++ with no intrinsics, which is deliberate. a2_fast's
// C=3 branch is scalar VFP on ARMv7 too, and its bit-identity premise is that
// the *compiler* contracts a*b+c into a fused multiply-add. Writing the control
// exactly as upstream writes it is what makes that premise testable rather than
// assumed — Scripts/a32-codegen-check.sh confirms the vfma is really there.

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "a32_common.h"
  #include "a32_ring.h"

namespace a32lab
{
namespace
{

constexpr int C = 3;

class NanoBaselineModel : public A32Model<C>
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
      R.write(_layer_in.data(), num_frames);
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
    _z.assign(static_cast<size_t>(C) * maxBufferSize, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_w.layers[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  /// The lab's shared copy of the ring these two controls used to carry
  /// privately, verbatim. See a32_ring.h for why a control does not use the
  /// parameterised ChannelRing.
  using Ring = BaselineRing<C>;

  /// The 3×3 GEMV fully unrolled: all nine weights lifted into named consts
  /// before the frame loop, the c-reduction kept in scalar temps a0/a1/a2 so the
  /// compiler keeps them in FP registers across the frame loop.
  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights<C>& L, const float* cond, int num_frames)
  {
    constexpr int K = KernelSize;
    const int D = L.dilation;
    const int mask = R.mask;
    auto tap_base_phys = [&](int taps_back) {
      return (R.write_pos - num_frames - taps_back * D) & mask;
    };

    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in the
    // first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &R.data[static_cast<size_t>(tap_base + f) * C];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        float* zf = z + static_cast<size_t>(f) * C;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 9];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &R.data[static_cast<size_t>(tap_base + f) * C];
        float* zf = z + static_cast<size_t>(f) * C;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Final tap (K-1, offset 0) inlined with the post-conv tail, so everything
    // after the conv runs on register-resident scalars:
    //   conv tap K-1 -> mixin -> LeakyReLU -> head_sum += -> layer1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 9];
    const int tap_base_last = tap_base_phys(0);
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
      const float* src = &R.data[static_cast<size_t>(tap_base_last + f) * C];
      const float* zf_mem = z + static_cast<size_t>(f) * C;
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      a0 += cw3 * src[1];
      a1 += cw4 * src[1];
      a2 += cw5 * src[1];
      a0 += cw6 * src[2];
      a1 += cw7 * src[2];
      a2 += cw8 * src[2];
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      float* hsum = &_head_sum[static_cast<size_t>(f) * C];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      float* lin = &_layer_in[static_cast<size_t>(f) * C];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2;
    }
  }

  void head_forward(float* out, int num_frames)
  {
    _head_ring.write(_head_sum.data(), num_frames);
    const int mask = _head_ring.mask;
    const int base = _head_ring.write_pos - num_frames;
    for (int f = 0; f < num_frames; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const int col = (base + f - (kHeadKernelSize - 1 - k)) & mask;
        const float* src = &_head_ring.data[static_cast<size_t>(col) * C];
        const float* wk = _w.head_w[k].data();
        for (int b = 0; b < C; b++)
          y += wk[b] * src[b];
      }
      out[f] = y * _w.head_scale;
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

std::unique_ptr<nam::DSP> make_nano_baseline(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<NanoBaselineModel>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
