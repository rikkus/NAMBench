#if defined(NB_ENABLE_SLIM_LAB)

  #include "slim_common.h"

  #include <iterator>
  #include <sstream>
  #include <stdexcept>
  #include <string>

  #include "NAM/wavenet/a2_fast.h"

namespace slimlab
{

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces A2FastModel::_load_weights, which in turn reproduces the generic
// path's read order:
//   _rechannel (Conv1x1 1 -> 3, no bias)
//   for each layer:
//       _conv (Conv1D 3 -> 3, K × 3 × 3 + 3 bias)
//       _input_mixin (Conv1x1 1 -> 3, no bias)
//       _layer1x1 (Conv1x1 3 -> 3, with bias)
//   _head_rechannel (Conv1D 3 -> 1, K=16, bias)
//   head_scale (trailing float)
//
// Generic Conv1D order is: for i in out_ch, for j in in_ch, for k in taps.
// Generic Conv1x1 order is: for i in out_ch, for j in in_ch. Both are permuted
// into column-major-per-tap storage as they are read.
//
// For A2 nano this consumes exactly 1871 floats:
//   3 + 21 × (54 + 18) + 2 × (135 + 18) + 48 + 1 + 1.
// -----------------------------------------------------------------------------
Weights parse_weights(const std::vector<float>& weights)
{
  Weights out;

  auto it = weights.begin();
  const auto end = weights.end();
  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("slimlab: weight stream exhausted");
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
          L.conv_w[static_cast<size_t>(k) * 9 + static_cast<size_t>(j) * 3 + i] = take();
    for (int i = 0; i < kChannels; i++)
      L.conv_b[i] = take();

    for (int i = 0; i < kChannels; i++)
      L.mixin_w[i] = take();

    for (int i = 0; i < kChannels; i++) // row (out)
      for (int j = 0; j < kChannels; j++) // col (in = bottleneck)
        L.l1x1_w[static_cast<size_t>(j) * 3 + i] = take();
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
    ss << "slimlab: weight stream has " << std::distance(it, end) << " trailing values";
    throw std::runtime_error(ss.str());
  }

  return out;
}

int prewarm_samples()
{
  // Receptive field = 1 (the sample being produced) + the per-layer lookbacks +
  // (head kernel - 1). The leading 1 matches the generic WaveNet's count, which
  // is what A2FastModel matches too.
  int prewarm = 1;
  for (int i = 0; i < kNumLayers; i++)
    prewarm += (kKernelSizes[i] - 1) * kDilations[i];
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
// `baseline`, because it is the control the whole lab is validated against.
// -----------------------------------------------------------------------------
namespace
{

const KernelEntry kKernels[] = {
  {"baseline", &make_baseline},     {"restrict", &make_restrict},     {"framemajor", &make_framemajor},
  {"pad4", &make_pad4},             {"planar", &make_planar},         {"planar_ring", &make_planar_ring},
  {"ringdirect", &make_ringdirect}, {"vext_taps", &make_vext_taps},   {"ld3", &make_ld3},
  {"skiplast", &make_skiplast},     {"prefetch", &make_prefetch},     {"widetile8", &make_widetile8},
  {"widetile16", &make_widetile16}, {"widetile32", &make_widetile32}, {"widetile64", &make_widetile64},
  {"stacked16", &make_stacked16},   {"stacked32", &make_stacked32},
  {"planar_linear", &make_planar_linear},
  {"stacked_linear", &make_stacked_linear},
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
    throw std::runtime_error("slimlab: no kernel at index " + std::to_string(index));

  // The lab bypasses create_config, so the shape check that would normally
  // gate the fast path has to happen here instead.
  int channels = 0;
  if (!nam::wavenet::a2_fast::is_a2_shape(config, &channels))
    throw std::runtime_error("slimlab: config is not the A2 shape");
  if (channels != kChannels)
  {
    throw std::runtime_error("slimlab: kernels are specialised for " + std::to_string(kChannels)
                             + " channels, this submodel has " + std::to_string(channels));
  }

  return kKernels[index].make(weights, sampleRate);
}

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
