// Shared scaffolding for the full-path kernel lab.
//
// Every candidate under Sources/FullEngines is a drop-in replacement for the
// 8-channel A2 path, built from the *same* vendor/fused tree through the same
// Engine target template. This header holds everything that must NOT vary
// between candidates — the weight loader, the prewarm count, the DSP plumbing —
// so that the only difference between two measured numbers is the kernel.
//
// Two references, two controls. vendor/fused is a superset of upstream: it
// carries a byte-identical a2_fast alongside the fused NEON engine
// (Scripts/fetch-vendor.sh asserts the a2_fast sources match). So this lab holds
// two families:
//
//   a2*  reproduce a2_fast's arithmetic exactly, validated against `upstream`
//   fu*  reproduce fused's arithmetic exactly, validated against `fused`
//
// with kernel 0 (a2_baseline) and kernel 1 (fu_baseline) as verbatim ports that
// have to land on their respective reference's number.
//
// The A2 full shape is fixed and known (checked by a2_fast::is_a2_shape before
// anything here runs): 23 layers, Channels == Bottleneck == 8, kernel sizes
// {6 × 14, 15, 15, 6 × 7}, LeakyReLU(0.01) everywhere, layer1x1 active, head
// rechannel k=16 with bias. 12,146 floats in the weight stream.

#pragma once

#if defined(NB_ENABLE_FULL_LAB)

  #include <array>
  #include <memory>
  #include <vector>

  #include "NAM/dsp.h"
  #include "json.hpp"

namespace fulllab
{

/// \brief Channels == Bottleneck for the full A2 submodel.
constexpr int kChannels = 8;
/// \brief Number of layers in the (single) A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief LeakyReLU negative slope used by every layer.
constexpr float kLeakySlope = 0.01f;
/// \brief Length of the A2 full weight stream. Asserted, not assumed:
/// 8 + 21 × (384 + 88) + 2 × (960 + 88) + 128 + 1 + 1.
constexpr int kWeightCount = 12146;

inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// One layer's weights.
///
/// a2_fast and fused happen to store these identically — column-major per tap
/// for the conv, column-major for the 1x1 — so one canonical form serves both
/// families and the loader is written and verified once. Candidates wanting a
/// different arrangement (planar, padded, transposed) permute *from* this at
/// construction time.
struct LayerWeights
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0; // (kernel_size - 1) * dilation

  /// kernel_size * 64 floats. Tap k, output i, input j lives at [k * 64 + j * 8 + i].
  std::vector<float> conv_w;
  std::array<float, kChannels> conv_b{};
  /// Input mixin (condition size 1 -> 8), no bias.
  std::array<float, kChannels> mixin_w{};
  /// layer1x1 (8 -> 8), column-major: [j * 8 + i] is bottleneck j to output i.
  std::array<float, kChannels * kChannels> l1x1_w{};
  std::array<float, kChannels> l1x1_b{};
};

/// The whole model's weights.
struct Weights
{
  /// Rechannel (input size 1 -> 8), no bias.
  std::array<float, kChannels> rechannel_w{};
  std::array<LayerWeights, kNumLayers> layers;
  /// Head rechannel (8 -> 1), kernel 16. At tap k the matrix is 1 × 8.
  std::array<std::array<float, kChannels>, kHeadKernelSize> head_w{};
  float head_b = 0.0f;
  /// Read from the trailing float of the stream, exactly as the generic
  /// WaveNet does — it overrides the JSON head_scale field.
  float head_scale = 1.0f;
};

/// Consume `weights` in exactly the order A2FastModel::_load_weights does —
/// which is also exactly the order FusedWaveNet::_load_weights does. Throws if
/// the stream is the wrong length.
Weights parse_weights(const std::vector<float>& weights);

/// Receptive field, matching both references' count so the lab warms up over the
/// same number of samples as the code it stands in for.
int prewarm_samples();

/// Smallest power of two >= v (v > 0).
int next_pow2(int v);

/// Base class for every candidate.
///
/// Holds the parsed weights and the prewarm count and nothing else, so a
/// candidate file contains its kernel and its buffers and no boilerplate.
class FullModel : public nam::DSP
{
public:
  FullModel(const std::vector<float>& weights, double expected_sample_rate)
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
/// the A2 full shape. Throws with a readable message if it is not.
std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate);

// --- Candidate factories -----------------------------------------------------
//
// Declared here and defined one per .cpp, so the registration table in
// full_common.cpp is an explicit, ordered list rather than whatever order
// static initialisers happened to run in. Indices 0 and 1 are the two controls
// and must stay where they are.

std::unique_ptr<nam::DSP> make_a2_baseline(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_baseline(const std::vector<float>&, double);

// Family A: planar, a2_fast arithmetic.
std::unique_ptr<nam::DSP> make_a2p4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p12(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_headtile(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_ringlazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_ringexact(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_ringlinear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_prefetch(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_head2(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_head8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s12(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s8_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s12_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_split8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_split16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s8_h8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2p_l1x1lane(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s8_h8_lane(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s8_h8_split(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_a2s16_split(const std::vector<float>&, double);

// Family B: channel-major, fused arithmetic.
std::unique_ptr<nam::DSP> make_fu_t4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_t8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_t12(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_t16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_tail4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_tail8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_fusez(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_headtile(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_storehead(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_ringeager(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_ringexact(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_ringlinear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_t6(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_t10(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s6(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s8_eager(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s8_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s8_head(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_fu_s8_head_lazy(const std::vector<float>&, double);

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
