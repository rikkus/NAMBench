#if defined(NB_ENABLE_FULL_LAB)

  #include "full_common.h"

  #include <iterator>
  #include <sstream>
  #include <stdexcept>
  #include <string>

  #include "NAM/wavenet/a2_fast.h"

namespace fulllab
{

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces A2FastModel::_load_weights, which in turn reproduces the generic
// path's read order:
//   _rechannel (Conv1x1 1 -> 8, no bias)
//   for each layer:
//       _conv (Conv1D 8 -> 8, K × 8 × 8 + 8 bias)
//       _input_mixin (Conv1x1 1 -> 8, no bias)
//       _layer1x1 (Conv1x1 8 -> 8, with bias)
//   _head_rechannel (Conv1D 8 -> 1, K=16, bias)
//   head_scale (trailing float)
//
// Generic Conv1D order is: for i in out_ch, for j in in_ch, for k in taps.
// Generic Conv1x1 order is: for i in out_ch, for j in in_ch. Both are permuted
// into column-major-per-tap storage as they are read.
//
// FusedWaveNet::_load_weights consumes the same stream in the same order into
// the same layouts, so this one loader is faithful to both references. The two
// engines differ in how they *use* these weights, not in how they read them.
//
// For A2 full this consumes exactly 12,146 floats:
//   8 + 21 × (384 + 88) + 2 × (960 + 88) + 128 + 1 + 1.
// -----------------------------------------------------------------------------
Weights parse_weights(const std::vector<float>& weights)
{
  Weights out;

  auto it = weights.begin();
  const auto end = weights.end();
  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("fulllab: weight stream exhausted");
    return *it++;
  };

  for (int i = 0; i < kChannels; i++)
    out.rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    LayerWeights& L = out.layers[li];
    L.kernel_size = kKernelSizes[li];
    L.dilation = kDilations[li];
    L.max_lookback = (L.kernel_size - 1) * L.dilation;
    const int K = L.kernel_size;

    L.conv_w.assign(static_cast<size_t>(K) * kChannels * kChannels, 0.0f);
    for (int i = 0; i < kChannels; i++) // row (out)
      for (int j = 0; j < kChannels; j++) // col (in)
        for (int k = 0; k < K; k++)
          L.conv_w[static_cast<size_t>(k) * kChannels * kChannels + static_cast<size_t>(j) * kChannels + i] = take();
    for (int i = 0; i < kChannels; i++)
      L.conv_b[i] = take();

    for (int i = 0; i < kChannels; i++)
      L.mixin_w[i] = take();

    for (int i = 0; i < kChannels; i++) // row (out)
      for (int j = 0; j < kChannels; j++) // col (in = bottleneck)
        L.l1x1_w[static_cast<size_t>(j) * kChannels + i] = take();
    for (int i = 0; i < kChannels; i++)
      L.l1x1_b[i] = take();
  }

  for (int j = 0; j < kChannels; j++)
    for (int k = 0; k < kHeadKernelSize; k++)
      out.head_w[k][j] = take();
  out.head_b = take();

  out.head_scale = take();

  if (it != end)
  {
    std::stringstream ss;
    ss << "fulllab: weight stream has " << std::distance(it, end) << " trailing values";
    throw std::runtime_error(ss.str());
  }

  return out;
}

// -----------------------------------------------------------------------------
// Receptive field. Matches A2FastModel's and FusedWaveNet's count (both start at
// 1, which is where the generic WaveNet's mPrewarmSamples starts when there is
// no condition DSP), so the lab warms up over the same number of samples as the
// code it stands in for.
// -----------------------------------------------------------------------------
int prewarm_samples()
{
  int prewarm = 1;
  for (int li = 0; li < kNumLayers; li++)
    prewarm += (kKernelSizes[li] - 1) * kDilations[li];
  prewarm += kHeadKernelSize - 1;
  return prewarm;
}

int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

// -----------------------------------------------------------------------------
// Registry
//
// An explicit ordered table rather than self-registration: index 0 must be
// `a2_baseline` and index 1 `fu_baseline`, because those are the two controls
// the whole lab is validated against.
// -----------------------------------------------------------------------------
namespace
{

const KernelEntry kKernels[] = {
  {"a2_baseline", &make_a2_baseline},
  {"fu_baseline", &make_fu_baseline},

  {"a2p4", &make_a2p4},
  {"a2p8", &make_a2p8},
  {"a2p12", &make_a2p12},
  {"a2p16", &make_a2p16},
  {"a2p_headtile", &make_a2p_headtile},
  {"a2p_ringdirect", &make_a2p_ringdirect},
  {"a2p_skiplast", &make_a2p_skiplast},
  {"a2p_ringlazy", &make_a2p_ringlazy},
  {"a2p_ringexact", &make_a2p_ringexact},
  {"a2p_ringlinear", &make_a2p_ringlinear},
  {"a2p_prefetch", &make_a2p_prefetch},
  {"a2p_head2", &make_a2p_head2},
  {"a2p_head8", &make_a2p_head8},

  {"a2s4", &make_a2s4},
  {"a2s8", &make_a2s8},
  {"a2s12", &make_a2s12},
  {"a2s8_linear", &make_a2s8_linear},
  {"a2s12_linear", &make_a2s12_linear},
  {"a2p_split8", &make_a2p_split8},
  {"a2p_split16", &make_a2p_split16},
  {"a2s8_h8", &make_a2s8_h8},
  {"a2p_l1x1lane", &make_a2p_l1x1lane},
  {"a2s8_h8_lane", &make_a2s8_h8_lane},
  {"a2s8_h8_split", &make_a2s8_h8_split},
  {"a2s16_split", &make_a2s16_split},

  {"fu_t4", &make_fu_t4},
  {"fu_t8", &make_fu_t8},
  {"fu_t12", &make_fu_t12},
  {"fu_t16", &make_fu_t16},
  {"fu_tail4", &make_fu_tail4},
  {"fu_tail8", &make_fu_tail8},
  {"fu_fusez", &make_fu_fusez},
  {"fu_ringdirect", &make_fu_ringdirect},
  {"fu_headtile", &make_fu_headtile},
  {"fu_storehead", &make_fu_storehead},
  {"fu_ringeager", &make_fu_ringeager},
  {"fu_ringexact", &make_fu_ringexact},
  {"fu_ringlinear", &make_fu_ringlinear},
  {"fu_t6", &make_fu_t6},
  {"fu_t10", &make_fu_t10},
  {"fu_s6", &make_fu_s6},
  {"fu_s8", &make_fu_s8},
  {"fu_s8_eager", &make_fu_s8_eager},
  {"fu_s8_lazy", &make_fu_s8_lazy},
  {"fu_s8_head", &make_fu_s8_head},
  {"fu_s8_head_lazy", &make_fu_s8_head_lazy},
};

constexpr int kKernelCount = static_cast<int>(sizeof(kKernels) / sizeof(kKernels[0]));

} // namespace

int kernel_count()
{
  return kKernelCount;
}

const char* kernel_name(int index)
{
  if (index < 0 || index >= kKernelCount)
    return nullptr;
  return kKernels[index].name;
}

std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate)
{
  if (index < 0 || index >= kKernelCount)
    throw std::runtime_error("fulllab: no kernel at index " + std::to_string(index));

  // The lab bypasses create_config, so the shape check that would normally gate
  // the fast path has to happen here instead.
  int channels = 0;
  if (!nam::wavenet::a2_fast::is_a2_shape(config, &channels))
    throw std::runtime_error("fulllab: config is not the A2 shape");
  if (channels != kChannels)
  {
    throw std::runtime_error("fulllab: kernels are specialised for " + std::to_string(kChannels)
                             + " channels, this submodel has " + std::to_string(channels));
  }
  if (static_cast<int>(weights.size()) != kWeightCount)
  {
    throw std::runtime_error("fulllab: expected " + std::to_string(kWeightCount) + " weights, got "
                             + std::to_string(weights.size()));
  }

  return kKernels[index].make(weights, sampleRate);
}

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
