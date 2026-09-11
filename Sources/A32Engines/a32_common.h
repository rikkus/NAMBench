// Shared scaffolding for the 32-bit ARM kernel lab.
//
// Every candidate under Sources/A32Engines is a drop-in replacement for one of
// a2_fast's two branches, built from the *same* vendor/upstream tree through the
// same target template. This header holds everything that must NOT vary between
// candidates — the weight loader, the prewarm count, the DSP plumbing — so that
// the only difference between two measured numbers is the kernel itself.
//
// Why one lab for both submodels, where SlimEngines and FullEngines are one
// each: those two grew out of historically different reference engines and
// vendor trees (SlimEngines' motivating comparison was against a NEON engine,
// `fused`, that has since been retired; FullEngines briefly carried a second
// family validated against it too, before that family was removed along with
// `fused`), which is what forced them apart at the time. Here both submodels
// share one reference (a2_fast), one vendor tree, one compat header, one set of
// ring strategies and one 16-register problem, so splitting them would duplicate
// all of that for no analytical gain.
//
// The A2 shapes are fixed and known (checked by a2_fast::is_a2_shape before
// anything here runs). They differ only in channel count:
//
//   nano      C == 3, 1871 floats in the weight stream
//   standard  C == 8, 12146 floats
//
// with 23 layers, kernel sizes {6 × 14, 15, 15, 6 × 7}, LeakyReLU(0.01), an
// active layer1x1 and a k=16 head rechannel with bias in both cases.

#pragma once

#if defined(NB_ENABLE_A32_LAB)

  #include <array>
  #include <memory>
  #include <vector>

  #include "NAM/dsp.h"
  #include "json.hpp"

namespace a32lab
{

/// \brief Number of layers in the (single) A2 layer array. Same at both widths.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief LeakyReLU negative slope used by every layer.
constexpr float kLeakySlope = 0.01f;

inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// How long the weight stream is at each width, asserted by the loader so a
/// mis-sized capture fails at construction rather than as a parity mystery.
template <int C>
struct Shape;
template <>
struct Shape<3>
{
  static constexpr int kWeightCount = 1871;
};
template <>
struct Shape<8>
{
  static constexpr int kWeightCount = 12146;
};

/// One layer's weights, in a2_fast's canonical column-major-per-tap layout.
///
/// Candidates that want a different arrangement (planar, splatted, transposed)
/// permute *from* this at construction time. Keeping one canonical form means
/// the loader is written and verified once.
template <int C>
struct LayerWeights
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0; // (kernel_size - 1) * dilation

  /// kernel_size * C * C floats. Tap k, output i, input j at [k*C*C + j*C + i].
  std::vector<float> conv_w;
  std::array<float, C> conv_b{};
  /// Input mixin (condition size 1 -> C), no bias.
  std::array<float, C> mixin_w{};
  /// layer1x1 (C -> C), column-major: [j*C + i] is bottleneck j to output i.
  std::array<float, C * C> l1x1_w{};
  std::array<float, C> l1x1_b{};
};

/// The whole model's weights.
template <int C>
struct Weights
{
  /// Rechannel (input size 1 -> C), no bias.
  std::array<float, C> rechannel_w{};
  std::array<LayerWeights<C>, kNumLayers> layers;
  /// Head rechannel (C -> 1), kernel 16. At tap k the matrix is 1 × C.
  std::array<std::array<float, C>, kHeadKernelSize> head_w{};
  float head_b = 0.0f;
  /// Read from the trailing float of the stream, exactly as the generic WaveNet
  /// does — it overrides the JSON head_scale field.
  float head_scale = 1.0f;
};

/// Consume `weights` in exactly the order A2FastModel::_load_weights does.
/// Throws if the stream is the wrong length.
template <int C>
Weights<C> parse_weights(const std::vector<float>& weights);

extern template Weights<3> parse_weights<3>(const std::vector<float>&);
extern template Weights<8> parse_weights<8>(const std::vector<float>&);

/// Receptive field, matching A2FastModel's count so the lab warms up over the
/// same number of samples as the code it stands in for.
int prewarm_samples();

/// Smallest power of two >= v (v > 0).
int next_pow2(int v);

/// Base class for every candidate.
///
/// Holds the parsed weights and the prewarm count and nothing else, so a
/// candidate file contains its kernel and its buffers and no boilerplate.
template <int C>
class A32Model : public nam::DSP
{
public:
  A32Model(const std::vector<float>& weights, double expected_sample_rate)
  : nam::DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
  , _w(parse_weights<C>(weights))
  , _prewarm(prewarm_samples())
  {
  }

  int GetPrewarmSamples() override { return _prewarm; }

protected:
  Weights<C> _w;
  int _prewarm = 0;
};

// --- Kernel registry ---------------------------------------------------------

using Factory = std::unique_ptr<nam::DSP> (*)(const std::vector<float>&, double);

struct KernelEntry
{
  const char* name;
  Factory make;
  /// 3 (A2 nano) or 8 (A2 standard). This is what lets one lab serve both
  /// submodels: the driver and the conformance runner pair a kernel with its
  /// submodel by reading this, rather than by a lab-wide constant.
  int channels;
  /// Whether this kernel claims to be bit-identical to a2_fast *on this target*.
  ///
  /// The claim lives next to the kernel making it rather than in a name
  /// convention the comparator decodes. On AArch64 only the verbatim ports
  /// claimed exactness and a suffix rule sufficed; here almost every kernel
  /// claims it and a couple deliberately do not, which a suffix rule would not
  /// carry.
  bool exact;
};

/// How many candidates this build carries.
int kernel_count();
/// Name of candidate `index`, or nullptr when out of range.
const char* kernel_name(int index);
/// Channel count candidate `index` is written for, or -1 when out of range.
int kernel_channels(int index);
/// 1 if candidate `index` claims bit-identity, 0 if not, -1 when out of range.
int kernel_exact(int index);

/// Build candidate `index` for a config already checked to be an A2 shape.
/// Throws with a readable message if the shape or the channel count is wrong.
std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate);

// --- Candidate factories -----------------------------------------------------
//
// Declared here and defined one per .cpp, so the registration table in
// a32_common.cpp is an explicit, ordered list rather than whatever order static
// initialisers happened to run in.

std::unique_ptr<nam::DSP> make_nano_baseline(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_baseline(const std::vector<float>&, double);

// The C=3 round.
std::unique_ptr<nam::DSP> make_nano_planar(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_vext_taps(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_planar_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_planar_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_vmla(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_framemajor(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_pad4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_prefetch(const std::vector<float>&, double);

// Added after the first round of C=3 measurements, to pin down where the tile
// ladder turns and which half of the stack is paying.
std::unique_ptr<nam::DSP> make_nano_widetile12(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_ring(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_skip(const std::vector<float>&, double);

// The C=8 round.
std::unique_ptr<nam::DSP> make_std_planar2(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked2(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_head_tile(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_chanmajor(const std::vector<float>&, double);

// Added after the first round of C=8 measurements, because the tile sweep was
// still climbing at its widest rung.
std::unique_ptr<nam::DSP> make_std_planar16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked16(const std::vector<float>&, double);

// Phase 5: the rungs the tile ladders were missing at each width, and the ring
// strategies measured on the winners rather than on the plain kernels.
std::unique_ptr<nam::DSP> make_nano_widetile2(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar12(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8_lazy(const std::vector<float>&, double);

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
