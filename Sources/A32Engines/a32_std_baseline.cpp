// Kernel 1: s_baseline — the control for the C=8 round.
//
// A verbatim port of a2_fast's `else` branch (the Channels >= 8 path, which is
// Eigen), along with the ring buffers, buffer sizing and process() around it, at
// ring mode 1 (pow2 + eager tail mirror), which is a2_fast's default.
//
// Nothing here is meant to be fast. Its whole job is to land on top of
// `upstream`'s number on this target: if it does not, the lab is measuring its
// own overhead and none of the C=8 candidates mean anything.
//
// On this target the control has a second job the AArch64 one did not need. The
// arithmetic Eigen performs here is decided by the build flags: it keys
// EIGEN_VECTORIZE_FMA off __ARM_FEATURE_FMA, so a -mfpu=neon build (NEON, but
// VFPv3, no FMA) computes every pmadd as a non-fused vmlaq_f32 instead of
// vfmaq_f32 — different bits, in the *reference*. Building this file with the
// same flags as upstream and requiring it to be bit-identical is what proves the
// two agree about which arithmetic they are doing.

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include <Eigen/Dense>

  #include "a32_common.h"
  #include "a32_ring.h"

namespace a32lab
{
namespace
{

constexpr int C = 8;

using MatCC = Eigen::Matrix<float, C, C>;
using MatCDyn = Eigen::Matrix<float, C, Eigen::Dynamic>;
using VecC = Eigen::Matrix<float, C, 1>;
using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

class StdBaselineModel : public A32Model<C>
{
public:
  using A32Model<C>::A32Model;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    // Rechannel: layer_in[c, f] = rechannel_w[c] * input[f]. Also the float copy
    // of the condition signal that every layer's mixin reads.
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

  /// One 8x8 × 8xN GEMM per tap, then bias, mixin, LeakyReLU, head_sum and the
  /// layer1x1 residual — every one of them an Eigen block op, exactly as
  /// a2_fast writes them.
  template <int KernelSize>
  void layer_forward(Ring& R, const LayerWeights<C>& L, const float* cond, int num_frames)
  {
    constexpr int K = KernelSize;
    const int D = L.dilation;
    const int mask = R.mask;
    auto tap_base_phys = [&](int taps_back) {
      return (R.write_pos - num_frames - taps_back * D) & mask;
    };

    Eigen::Map<const VecC> conv_b_vec(L.conv_b.data());
    Eigen::Map<const VecC> mixin_vec(L.mixin_w.data());
    Eigen::Map<const MatCC> l1x1_mat(L.l1x1_w.data());
    Eigen::Map<const VecC> l1x1_b_vec(L.l1x1_b.data());
    Eigen::Map<const RowDyn> cond_row(cond, 1, num_frames);

    Eigen::Map<MatCDyn> ztile(_z.data(), C, num_frames);
    Eigen::Map<MatCDyn> hsum_block(_head_sum.data(), C, num_frames);
    Eigen::Map<MatCDyn> lin_block(_layer_in.data(), C, num_frames);

    ztile.setZero();

    for (int k = 0; k < K; k++)
    {
      const int tap_base = tap_base_phys(K - 1 - k);
      Eigen::Map<const MatCC> W(&L.conv_w[static_cast<size_t>(k) * C * C]);
      Eigen::Map<const MatCDyn> input_block(&R.data[static_cast<size_t>(tap_base) * C], C,
                                            num_frames);
      ztile.noalias() += W * input_block;
    }

    ztile.colwise() += conv_b_vec;
    ztile.noalias() += mixin_vec * cond_row; // rank-1 outer product
    ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
    hsum_block += ztile;
    lin_block.noalias() += l1x1_mat * ztile; // 8x8 × 8xN GEMM
    lin_block.colwise() += l1x1_b_vec;
  }

  /// Head: K=16, dilation 1, eight channels down to one, plus bias and scale.
  /// Scalar, as a2_fast has it — 128 sequential FMAs per frame.
  void head_forward(float* output, int num_frames)
  {
    _head_ring.write(_head_sum.data(), num_frames);
    const int mask = _head_ring.mask;
    auto col_of = [&](int f, int k) {
      return (_head_ring.write_pos - num_frames + f - (kHeadKernelSize - 1 - k)) & mask;
    };

    for (int f = 0; f < num_frames; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const int col = col_of(f, k);
        const float* src = &_head_ring.data[static_cast<size_t>(col) * C];
        const float* wk = _w.head_w[k].data();
        for (int b = 0; b < C; b++)
          y += wk[b] * src[b];
      }
      output[f] = y * _w.head_scale;
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

std::unique_ptr<nam::DSP> make_std_baseline(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<StdBaselineModel>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
