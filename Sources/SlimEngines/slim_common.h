// Shared scaffolding for the slimmed-path kernel lab.
//
// Every candidate under Sources/SlimEngines is a drop-in replacement for
// a2_fast's `if constexpr (Channels == 3)` branch, built from the *same*
// vendor/upstream tree through the same Engine target template. This header
// holds everything that must NOT vary between candidates — the weight loader,
// the prewarm count, the DSP plumbing — so that the only difference between two
// measured numbers is the kernel itself.
//
// The A2 nano shape is fixed and known (checked by a2_fast::is_a2_shape before
// anything here runs): 23 layers, Channels == Bottleneck == 3, kernel sizes
// {6 × 14, 15, 15, 6 × 7}, LeakyReLU(0.01) everywhere, layer1x1 active, head
// rechannel k=16 with bias. 1871 floats in the weight stream.

#pragma once

#if defined(NB_ENABLE_SLIM_LAB)

  #include <array>
  #include <memory>
  #include <vector>

  #include "NAM/dsp.h"
  #include "json.hpp"

namespace slimlab
{

/// \brief Channels == Bottleneck for the slimmed A2 submodel.
constexpr int kChannels = 3;
/// \brief Number of layers in the (single) A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief LeakyReLU negative slope used by every layer.
constexpr float kLeakySlope = 0.01f;

inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// One layer's weights, in a2_fast's canonical column-major-per-tap layout.
///
/// Candidates that want a different arrangement (padded to 4 lanes, planar,
/// interleaved) permute *from* this at construction time. Keeping one canonical
/// form means the loader is written and verified once.
struct LayerWeights
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0; // (kernel_size - 1) * dilation

  /// kernel_size * 9 floats. Tap k, output i, input j lives at [k * 9 + j * 3 + i].
  std::vector<float> conv_w;
  std::array<float, kChannels> conv_b{};
  /// Input mixin (condition size 1 -> 3), no bias.
  std::array<float, kChannels> mixin_w{};
  /// layer1x1 (3 -> 3), column-major: [j * 3 + i] is bottleneck j to output i.
  std::array<float, kChannels * kChannels> l1x1_w{};
  std::array<float, kChannels> l1x1_b{};
};

/// The whole model's weights.
struct Weights
{
  /// Rechannel (input size 1 -> 3), no bias.
  std::array<float, kChannels> rechannel_w{};
  std::array<LayerWeights, kNumLayers> layers;
  /// Head rechannel (3 -> 1), kernel 16. At tap k the matrix is 1 × 3.
  std::array<std::array<float, kChannels>, kHeadKernelSize> head_w{};
  float head_b = 0.0f;
  /// Read from the trailing float of the stream, exactly as the generic
  /// WaveNet does — it overrides the JSON head_scale field.
  float head_scale = 1.0f;
};

/// Consume `weights` in exactly the order A2FastModel::_load_weights does.
/// Throws if the stream is the wrong length.
Weights parse_weights(const std::vector<float>& weights);

/// Receptive field, matching A2FastModel's count so the lab warms up over the
/// same number of samples as the code it stands in for.
int prewarm_samples();

/// Smallest power of two >= v (v > 0).
int next_pow2(int v);

/// Base class for every candidate.
///
/// Holds the parsed weights and the prewarm count and nothing else, so a
/// candidate file contains its kernel and its buffers and no boilerplate.
class SlimModel : public nam::DSP
{
public:
  SlimModel(const std::vector<float>& weights, double expected_sample_rate)
  : nam::DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
  , _w(parse_weights(weights))
  , _prewarm(prewarm_samples())
  {
  }

  int GetPrewarmSamples() override { return _prewarm; }

protected:
  Weights _w;
  int _prewarm = 0;
};

// --- Kernel registry ---------------------------------------------------------

using Factory = std::unique_ptr<nam::DSP> (*)(const std::vector<float>&, double);

struct KernelEntry
{
  const char* name;
  Factory make;
};

/// How many candidates this build carries.
int kernel_count();
/// Name of candidate `index`, or nullptr when out of range.
const char* kernel_name(int index);

/// Build candidate `index` for a config that has already been checked to be
/// the A2 nano shape. Throws with a readable message if it is not.
std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate);

// --- Candidate factories -----------------------------------------------------
//
// Declared here and defined one per .cpp, so the registration table in
// slim_common.cpp is an explicit, ordered list rather than whatever order
// static initialisers happened to run in.

std::unique_ptr<nam::DSP> make_baseline(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_restrict(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_framemajor(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_pad4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_planar(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_planar_ring(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_vext_taps(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_ld3(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_prefetch(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_widetile8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_widetile16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_widetile32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_widetile64(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_stacked16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_stacked32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_planar_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_stacked_linear(const std::vector<float>&, double);

} // namespace slimlab

#endif // NB_ENABLE_SLIM_LAB
