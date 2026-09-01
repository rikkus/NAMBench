// Kernel 27: s_head_tile — the same Eigen layer body, with the head vectorised.
//
// s_baseline's head is 128 sequential scalar FMAs per frame: K=16 taps by eight
// channels, one frame at a time, exactly as a2_fast writes it. At C=3 that was
// 48 per frame and still mattered; at C=8 it is the largest single scalar
// stretch in the model, on a core whose scalar FP throughput is its weakest
// resource. The campaign plan calls it the cheapest large win available at this
// width, and this kernel is the one that tests that in isolation — the layer
// body below is the control's, byte for byte.
//
// Measured: 1.020x. A win, and a small one, and the plan's expectation was wrong
// on the arithmetic rather than on the mechanism. Per frame the head is 16 x 8 =
// 128 FMAs against the layer stack's ~11,500, so even at the penalty scalar code
// pays on this core it is only a few percent of the block. At C=3 the same
// reasoning gives the head a much larger share, which is where the expectation
// came from.
//
// The head reduces across channels, so vectorising it means putting *frames* in
// the lanes, which means the head history has to be planar. The layer body
// produces head_sum channel-major (it is an Eigen C x N block), so this
// transposes C x N once per block on the way into the head ring. That is 8N
// floats moved to save 128N scalar FMAs.
//
// Exact. Each lane runs the reference's own chain — bias, then tap 0's eight
// channels, then tap 1's — in the reference's order. The probe confirmed this
// shape bit-for-bit on the board (Scripts/eigen-order-probe: "planar, one lane
// per frame, FMA", 224/224).

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include <Eigen/Dense>

  #include "a32_common.h"
  #include "a32_neon_compat.h"
  #include "a32_ring.h"

namespace a32lab
{
namespace
{

constexpr int C = 8;
/// Frames per head iteration. The head holds one accumulator, not C of them, so
// it is not competing for registers with anything.
constexpr int kHeadTile = 8;

using MatCC = Eigen::Matrix<float, C, C>;
using MatCDyn = Eigen::Matrix<float, C, Eigen::Dynamic>;
using VecC = Eigen::Matrix<float, C, 1>;
using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

class StdHeadTileModel : public A32Model<C>
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
  /// The control's ring for the layer stack, so the layer body is unchanged in
  /// every respect, and a planar one for the head, which is the point.
  using Ring = BaselineRing<C>;
  using HeadRing = PlanarRing<C, RingKind::Pow2Eager>;

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


  /// Head: K=16, dilation 1, eight channels down to one, plus bias and scale —
  /// kHeadTile frames at a time, one frame per lane.
  void head_forward(float* out, int num_frames)
  {
    // Transpose this block's head_sum into the planar head ring. The layer body
    // wrote it channel-major because Eigen did; the head wants it plane by
    // plane.
    _head_ring.prepare(num_frames);
    for (int c = 0; c < C; c++)
    {
      float* dst = _head_ring.write_ptr(c);
      const float* src = _head_sum.data() + c;
      for (int f = 0; f < num_frames; f++)
        dst[f] = src[static_cast<size_t>(f) * C];
    }
    _head_ring.commit(num_frames);

    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, num_frames);

    const float* p[C];
    for (int c = 0; c < C; c++)
      p[c] = _head_ring.plane(c);

    const float scale = _w.head_scale;
    constexpr int NV = kHeadTile / 4;

    int f = 0;
    for (; f + kHeadTile <= num_frames; f += kHeadTile)
    {
      float32x4_t y[NV];
      for (int v = 0; v < NV; v++)
        y[v] = vdupq_n_f32(_w.head_b);

      for (int k = 0; k < kHeadKernelSize; k++)
        for (int j = 0; j < C; j++)
          for (int v = 0; v < NV; v++)
            y[v] = compat::fmaq_bcast(y[v], vld1q_f32(p[j] + hb[k] + f + 4 * v),
                                      &_w.head_w[k][j]);

      for (int v = 0; v < NV; v++)
        vst1q_f32(out + f + 4 * v, vmulq_n_f32(y[v], scale));
    }
    for (; f + 4 <= num_frames; f += 4)
    {
      float32x4_t y = vdupq_n_f32(_w.head_b);
      for (int k = 0; k < kHeadKernelSize; k++)
        for (int j = 0; j < C; j++)
          y = compat::fmaq_bcast(y, vld1q_f32(p[j] + hb[k] + f), &_w.head_w[k][j]);
      vst1q_f32(out + f, vmulq_n_f32(y, scale));
    }
    for (; f < num_frames; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
        for (int j = 0; j < C; j++)
          y = __builtin_fmaf(_w.head_w[k][j], p[j][hb[k] + f], y);
      out[f] = y * scale;
    }
  }

  std::array<Ring, kNumLayers> _rings;
  HeadRing _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _z;
  std::vector<float> _cond;
  std::vector<float> _head_out;
};

} // namespace

std::unique_ptr<nam::DSP> make_std_head_tile(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<StdHeadTileModel>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
