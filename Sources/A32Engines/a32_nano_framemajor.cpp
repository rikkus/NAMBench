// Kernel 14: n_framemajor — keep z in registers and widen the chain, without
// changing the layout.
//
// NB_A32_NOT_EXACT.
//
// Written expecting it to lose, and here because confirming *why* it loses is
// what validates the register model the rest of the round is predicted from.
//
// Two things it changes about the control, both by loop restructuring alone —
// no intrinsics, no planar layout:
//
//   1. `z` is a heap buffer every tap reads and writes. At K=6 that is 30 loads
//      and stores of z per frame per layer against only 18 history loads.
//      Turning the nest inside out — frame outer, tap inner — keeps z in
//      registers for the whole life of a frame.
//   2. The control's three accumulator chains cannot cover VFP FMA latency.
//      Splitting the reduction into nine partials, one per (output, input)
//      pair, and running two frames per tile gives eighteen independent chains.
//
// The second change reassociates the reduction — nine partials summed at the
// end is not a2_fast's single chain — so this kernel is not bit-identical and is
// registered exact = false.
//
// The prediction: it lost 0.606× on an M2 with thirty-two vector registers and
// far more scalar FP resource. Eighteen live accumulators plus nine weights plus
// addressing does not fit AArch32's VFP register file comfortably, so it should
// lose by more here. If it does not, the register model is wrong.
//
// Measured: 0.789x against a2_fast. It lost, as predicted — but by *less* than on
// AArch64, which the prediction had backwards. The reason is the reference it is
// being compared against: a2_fast's scalar C=3 branch is much weaker relative to
// NEON on this part than on an M2, so a scalar restructuring that keeps z in
// registers claws back more of the gap even while it spills. The register model
// survives (it does lose); the direction of the *margin* did not.

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

class NanoFrameMajorModel : public A32Model<C>
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

  /// Two frames at a time, nine partial accumulators each. z never reaches
  /// memory: the whole life of a frame's conv, mixin, activation, head_sum and
  /// residual happens in registers.
  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights<C>& L, const float* cond, int num_frames)
  {
    constexpr int K = KernelSize;
    const int D = L.dilation;

    // The mirror past the end of the ring makes every read of up to one block
    // contiguous, so a tap is a plain pointer and the frame loop needs no
    // wrapping test.
    const float* tap[K];
    for (int k = 0; k < K; k++)
      tap[k] = R.tap((K - 1 - k) * D, num_frames);

    int f = 0;
    for (; f + 2 <= num_frames; f += 2)
      pair<K>(tap, L, cond, f);
    for (; f < num_frames; f++)
      one<K>(tap, L, cond, f);
  }

  /// Nine partials for each of two frames. Written out rather than looped so
  /// the compiler is asked for eighteen live chains explicitly.
  template <int K>
  void pair(const float* const (&tap)[K], const LayerWeights<C>& L, const float* cond, int f)
  {
    float p0[C][C] = {};
    float p1[C][C] = {};

    for (int k = 0; k < K; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * C * C];
      const float* s0 = tap[k] + static_cast<size_t>(f) * C;
      const float* s1 = s0 + C;
      for (int j = 0; j < C; j++)
        for (int i = 0; i < C; i++)
        {
          p0[i][j] += wk[j * C + i] * s0[j];
          p1[i][j] += wk[j * C + i] * s1[j];
        }
    }

    tail(p0, L, cond, f, tap[K - 1]);
    tail(p1, L, cond, f + 1, tap[K - 1]);
  }

  template <int K>
  void one(const float* const (&tap)[K], const LayerWeights<C>& L, const float* cond, int f)
  {
    float p[C][C] = {};
    for (int k = 0; k < K; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * C * C];
      const float* s = tap[k] + static_cast<size_t>(f) * C;
      for (int j = 0; j < C; j++)
        for (int i = 0; i < C; i++)
          p[i][j] += wk[j * C + i] * s[j];
    }
    tail(p, L, cond, f, tap[K - 1]);
  }

  /// Sum the partials — this is the reassociation — then the layer tail.
  void tail(const float (&p)[C][C], const LayerWeights<C>& L, const float* cond, int f,
            const float* last_tap)
  {
    float a[C];
    for (int i = 0; i < C; i++)
      a[i] = ((L.conv_b[i] + p[i][0]) + p[i][1]) + p[i][2];

    const float cf = cond[f];
    for (int i = 0; i < C; i++)
    {
      a[i] += L.mixin_w[i] * cf;
      a[i] = (a[i] >= 0.0f) ? a[i] : a[i] * kLeakySlope;
    }

    float* hsum = &_head_sum[static_cast<size_t>(f) * C];
    for (int i = 0; i < C; i++)
      hsum[i] += a[i];

    const float* src = last_tap + static_cast<size_t>(f) * C;
    float* lin = &_layer_in[static_cast<size_t>(f) * C];
    for (int i = 0; i < C; i++)
    {
      float o = L.l1x1_b[i];
      for (int j = 0; j < C; j++)
        o += L.l1x1_w[j * C + i] * a[j];
      lin[i] = src[i] + o;
    }
  }

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

std::unique_ptr<nam::DSP> make_nano_framemajor(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<NanoFrameMajorModel>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
