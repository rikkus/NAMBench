#if defined(NB_ENABLE_A32_LAB)

  #include "a32_common.h"

  #include <iterator>
  #include <sstream>
  #include <stdexcept>
  #include <string>

  #include "NAM/wavenet/a2_fast.h"

namespace a32lab
{

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces A2FastModel::_load_weights, which in turn reproduces the generic
// path's read order:
//   _rechannel (Conv1x1 1 -> C, no bias)
//   for each layer:
//       _conv (Conv1D C -> C, K × C × C + C bias)
//       _input_mixin (Conv1x1 1 -> C, no bias)
//       _layer1x1 (Conv1x1 C -> C, with bias)
//   _head_rechannel (Conv1D C -> 1, K=16, bias)
//   head_scale (trailing float)
//
// Generic Conv1D order is: for i in out_ch, for j in in_ch, for k in taps.
// Generic Conv1x1 order is: for i in out_ch, for j in in_ch. Both are permuted
// into column-major-per-tap storage as they are read.
//
// One function for both widths, because the order is identical and only the
// strides change. The slim and full labs each carry their own copy of this; that
// duplication is deliberate there (different vendor trees) and pointless here.
// -----------------------------------------------------------------------------
template <int C>
Weights<C> parse_weights(const std::vector<float>& weights)
{
  Weights<C> out;

  auto it = weights.begin();
  const auto end = weights.end();
  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("a32lab: weight stream exhausted");
    return *it++;
  };

  for (int i = 0; i < C; i++)
    out.rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    LayerWeights<C>& L = out.layers[li];
    L.kernel_size = kKernelSizes[li];
    L.dilation = kDilations[li];
    L.max_lookback = (L.kernel_size - 1) * L.dilation;
    const int K = L.kernel_size;

    L.conv_w.assign(static_cast<size_t>(K) * C * C, 0.0f);
    for (int i = 0; i < C; i++) // row (out)
      for (int j = 0; j < C; j++) // col (in)
        for (int k = 0; k < K; k++)
          L.conv_w[static_cast<size_t>(k) * C * C + static_cast<size_t>(j) * C + i] = take();
    for (int i = 0; i < C; i++)
      L.conv_b[i] = take();

    for (int i = 0; i < C; i++)
      L.mixin_w[i] = take();

    for (int i = 0; i < C; i++) // row (out)
      for (int j = 0; j < C; j++) // col (in = bottleneck)
        L.l1x1_w[static_cast<size_t>(j) * C + i] = take();
    for (int i = 0; i < C; i++)
      L.l1x1_b[i] = take();
  }

  for (int j = 0; j < C; j++)
    for (int k = 0; k < kHeadKernelSize; k++)
      out.head_w[k][j] = take();
  out.head_b = take();

  out.head_scale = take();

  if (it != end)
  {
    std::stringstream ss;
    ss << "a32lab: weight stream has " << std::distance(it, end) << " trailing values";
    throw std::runtime_error(ss.str());
  }

  return out;
}

template Weights<3> parse_weights<3>(const std::vector<float>&);
template Weights<8> parse_weights<8>(const std::vector<float>&);

int prewarm_samples()
{
  // Receptive field = 1 (the sample being produced) + the per-layer lookbacks +
  // (head kernel - 1). The leading 1 matches the generic WaveNet's count, which
  // is what A2FastModel matches too. Identical at both widths — the shapes
  // differ only in channel count.
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
// `n_baseline` and index 1 `s_baseline`, because those are the two controls the
// whole lab is validated against — one per submodel.
//
// Append, never insert: the indices appear in published reports and in the
// conformance record labels.
// -----------------------------------------------------------------------------
namespace
{

const KernelEntry kKernels[] = {
  {"n_baseline", &make_nano_baseline, 3, true},
  {"s_baseline", &make_std_baseline, 8, true},

  // The C=3 round, in the order the campaign plan measures it: the idea, the
  // tile sweep, the switches layered on it one at a time, the compositions, and
  // then the three written expecting them to lose.
  {"n_planar", &make_nano_planar, 3, true},
  {"n_widetile8", &make_nano_widetile8, 3, true},
  {"n_widetile16", &make_nano_widetile16, 3, true},
  {"n_widetile32", &make_nano_widetile32, 3, true},
  {"n_ringdirect", &make_nano_ringdirect, 3, true},
  {"n_skiplast", &make_nano_skiplast, 3, true},
  {"n_vext_taps", &make_nano_vext_taps, 3, true},
  {"n_planar_linear", &make_nano_planar_linear, 3, true},
  {"n_planar_lazy", &make_nano_planar_lazy, 3, true},
  {"n_stacked8", &make_nano_stacked8, 3, true},
  {"n_stacked16", &make_nano_stacked16, 3, true},
  {"n_vmla", &make_nano_vmla, 3, false},
  {"n_framemajor", &make_nano_framemajor, 3, false},
  {"n_pad4", &make_nano_pad4, 3, true},
  {"n_prefetch", &make_nano_prefetch, 3, true},

  // Appended after the first round measured: the missing rung of the tile
  // ladder, and the two half-stacks that say which switch the composition's win
  // came from. Appended rather than inserted — the indices are in published
  // reports and conformance labels.
  {"n_widetile12", &make_nano_widetile12, 3, true},
  {"n_stacked8_ring", &make_nano_stacked8_ring, 3, true},
  {"n_stacked8_skip", &make_nano_stacked8_skip, 3, true},

  // The C=8 round.
  {"s_planar2", &make_std_planar2, 8, true},
  {"s_planar4", &make_std_planar4, 8, true},
  {"s_planar8", &make_std_planar8, 8, true},
  {"s_ringdirect", &make_std_ringdirect, 8, true},
  {"s_skiplast", &make_std_skiplast, 8, true},
  {"s_stacked2", &make_std_stacked2, 8, true},
  {"s_stacked4", &make_std_stacked4, 8, true},
  {"s_head_tile", &make_std_head_tile, 8, true},
  {"s_chanmajor", &make_std_chanmajor, 8, false},

  // The rungs the first C=8 measurements asked for.
  {"s_planar16", &make_std_planar16, 8, true},
  {"s_stacked8", &make_std_stacked8, 8, true},
  {"s_stacked16", &make_std_stacked16, 8, true},

  // Phase 5's sweeps.
  {"n_widetile2", &make_nano_widetile2, 3, true},
  {"n_stacked8_linear", &make_nano_stacked8_linear, 3, true},
  {"n_stacked8_lazy", &make_nano_stacked8_lazy, 3, true},
  {"s_planar12", &make_std_planar12, 8, true},
  {"s_planar32", &make_std_planar32, 8, true},
  {"s_stacked8_linear", &make_std_stacked8_linear, 8, true},
  {"s_stacked8_lazy", &make_std_stacked8_lazy, 8, true},
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

int kernel_channels(int index)
{
  if (index < 0 || index >= kKernelCount)
    return -1;
  return kKernels[index].channels;
}

int kernel_exact(int index)
{
  if (index < 0 || index >= kKernelCount)
    return -1;
  return kKernels[index].exact ? 1 : 0;
}

std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate)
{
  if (index < 0 || index >= kKernelCount)
    throw std::runtime_error("a32lab: no kernel at index " + std::to_string(index));

  // The lab bypasses create_config, so the shape check that would normally gate
  // the fast path has to happen here instead.
  int channels = 0;
  if (!nam::wavenet::a2_fast::is_a2_shape(config, &channels))
    throw std::runtime_error("a32lab: this config is not the A2 shape");

  const int want = kKernels[index].channels;
  if (channels != want)
  {
    std::stringstream ss;
    ss << "a32lab: kernel '" << kKernels[index].name << "' is written for " << want
       << " channels, this submodel has " << channels;
    throw std::runtime_error(ss.str());
  }

  return kKernels[index].make(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
