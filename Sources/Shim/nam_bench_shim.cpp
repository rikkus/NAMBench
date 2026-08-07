// One implementation, compiled once per variant framework.
//
// The variant is selected entirely by build settings:
//   NB_PREFIX          symbol prefix, e.g. nb_upstream
//   NB_VARIANT_NAME    display name, e.g. "upstream"
//   NAM_ENABLE_FUSED   defined only for the fork build
//
// Everything else — sources, flags, include paths (including one shared Eigen
// tree) — is identical between the two frameworks. That is the whole point:
// any measured difference has to come from the engine, not the build.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <time.h>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/wavenet/a2_fast.h"

#if defined(NAM_ENABLE_FUSED)
  #include "NAM/wavenet/engine_prefer.h"
  #include "NAM/wavenet/fused.h"
#endif

#if defined(NB_ENABLE_SLIM_LAB)
  #include "slim_common.h"
#endif

#include "nam_bench_shim.h"

#if !defined(NB_PREFIX)
  #error "NB_PREFIX must be defined (e.g. -DNB_PREFIX=nb_upstream)"
#endif
#if !defined(NB_VARIANT_NAME)
  #error "NB_VARIANT_NAME must be defined (e.g. -DNB_VARIANT_NAME=upstream, unquoted)"
#endif

// a2_fast is the upstream variant's whole subject, and the two builds must stay
// identical apart from NAM_ENABLE_FUSED — so this is a build error, not a
// silent fallback to the generic engine.
#if !defined(NAM_ENABLE_A2_FAST)
  #error "NAM_ENABLE_A2_FAST must be defined for both variants"
#endif

#define NB_CAT_(a, b) a##b
#define NB_CAT(a, b) NB_CAT_(a, b)
#define NB_FN(suffix) NB_CAT(NB_PREFIX, suffix)

// Stringized here rather than passed pre-quoted, so no escaped quotes have to
// survive the trip through Xcode build settings.
#define NB_STRINGIFY_(x) #x
#define NB_STRINGIFY(x) NB_STRINGIFY_(x)

// --- Denormal flushing -------------------------------------------------------
//
// Applied identically in both variants around the block loop. A WaveNet's
// decaying tails can drift into denormal range, where the penalty is large and
// unrepresentative of how the plugin actually runs.

namespace
{

#if defined(__aarch64__)
constexpr uint64_t kFpcrFlushToZero = 1ull << 24; // FPCR.FZ

inline uint64_t fpcr_read()
{
  uint64_t value;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
  return value;
}

inline void fpcr_write(uint64_t value)
{
  __asm__ __volatile__("msr fpcr, %0" : : "r"(value));
}

inline uint64_t denormals_disable()
{
  const uint64_t previous = fpcr_read();
  fpcr_write(previous | kFpcrFlushToZero);
  return previous;
}

inline void denormals_restore(uint64_t previous)
{
  fpcr_write(previous);
}
#else
inline uint64_t denormals_disable()
{
  return 0;
}
inline void denormals_restore(uint64_t)
{
}
#endif

inline uint64_t now_ns()
{
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

void set_error(char* err, size_t errLen, const std::string& message)
{
  if (err == nullptr || errLen == 0)
    return;
  std::snprintf(err, errLen, "%s", message.c_str());
}

/// Which submodel of a .nam we are going to run.
struct Selection
{
  const nlohmann::json* model = nullptr;
  int32_t index = -1;
  int32_t count = 0;
  double maxValue = 0.0;
};

/// Pick one submodel out of a .nam.
///
/// A2 captures ship as a SlimmableContainer holding a nano submodel
/// (channels 3, max_value 0.5) and a full submodel (channels 8, max_value 1.0).
/// `Widest`/`Narrowest` select by max_value rather than by position, so
/// ordering changes in the trainer cannot silently switch which one is
/// measured; `Index` is the escape hatch for containers with more than two.
///
/// The chosen submodel's "model" object has exactly the same key set as a
/// top-level .nam (version/metadata/architecture/config/weights/sample_rate),
/// so it can be handed straight to get_dsp. Its layers carry no "slimmable"
/// object, so create_config routes it to the engine detectors rather than back
/// into the slimmable path.
bool select_submodel(const nlohmann::json& root, NbSubmodel mode, int32_t index, Selection& out,
                     std::string& error)
{
  const auto configIt = root.find("config");
  if (configIt != root.end() && configIt->is_object())
  {
    const auto submodelsIt = configIt->find("submodels");
    if (submodelsIt != configIt->end() && submodelsIt->is_array() && !submodelsIt->empty())
    {
      const int32_t count = static_cast<int32_t>(submodelsIt->size());
      int32_t chosen = -1;
      double chosenValue = 0.0;

      if (mode == NbSubmodelIndex)
      {
        if (index < 0 || index >= count)
        {
          error = "submodel index " + std::to_string(index) + " is out of range (container has "
                  + std::to_string(count) + ")";
          return false;
        }
        const auto& submodel = (*submodelsIt)[static_cast<size_t>(index)];
        if (!submodel.is_object() || submodel.find("model") == submodel.end())
        {
          error = "submodel " + std::to_string(index) + " has no \"model\" object";
          return false;
        }
        chosen = index;
        chosenValue = submodel.value("max_value", 0.0);
      }
      else
      {
        const bool widest = (mode != NbSubmodelNarrowest);
        double best = widest ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < submodelsIt->size(); i++)
        {
          const auto& submodel = (*submodelsIt)[i];
          if (!submodel.is_object() || submodel.find("model") == submodel.end())
            continue;
          const double value =
            submodel.value("max_value", widest ? -std::numeric_limits<double>::infinity()
                                               : std::numeric_limits<double>::infinity());
          if (widest ? (value > best) : (value < best))
          {
            best = value;
            chosen = static_cast<int32_t>(i);
          }
        }

        if (chosen < 0)
        {
          error = "container has a submodels array but no entry with a \"model\" object";
          return false;
        }
        chosenValue = best;
      }

      out.model = &(*submodelsIt)[static_cast<size_t>(chosen)]["model"];
      out.index = chosen;
      out.count = count;
      out.maxValue = chosenValue;
      return true;
    }
  }

  // A plain, non-container .nam. Any mode gives the same single model, but an
  // explicit index other than 0 is a caller mistake worth reporting.
  if (mode == NbSubmodelIndex && index > 0)
  {
    error = "this .nam is not a container, so submodel index " + std::to_string(index)
            + " does not exist";
    return false;
  }
  out.model = &root;
  out.index = -1;
  out.count = 0;
  out.maxValue = 0.0;
  return true;
}

#if defined(NB_ENABLE_SLIM_LAB)
/// Which experimental kernel _create will build. Negative means "none chosen
/// yet", in which case this build behaves exactly like `upstream`.
int g_slim_kernel = -1;
#endif

/// Report which engine this build will actually route `modelConfig` to.
///
/// This mirrors wavenet::create_config rather than trusting the build flags.
/// It matters most for the fork: under ScopedEnginePrefer(FusedNeon) a config
/// whose shape the fused engine does not accept falls through to the *generic*
/// engine, never to a2_fast. Measuring that by accident would look like a
/// dramatic regression, so the caller compares this against what it expects and
/// refuses to run on a mismatch.
NbEngine detect_engine(const nlohmann::json& modelConfig, int32_t* channels)
{
  // is_a2_shape is the only detector that yields a channel count, and it is
  // compiled into both variants, so use it for reporting either way.
  int a2Channels = 0;
  const bool isA2 = nam::wavenet::a2_fast::is_a2_shape(modelConfig, &a2Channels);
  if (channels != nullptr)
    *channels = isA2 ? static_cast<int32_t>(a2Channels) : 0;

#if defined(NAM_ENABLE_FUSED)
  if (nam::wavenet::fused::available() && nam::wavenet::fused::is_fused_shape(modelConfig))
    return NbEngineFused;
  return NbEngineGeneric;
#elif defined(NB_ENABLE_SLIM_LAB)
  // _create bypasses get_dsp entirely once a kernel has been chosen, so report
  // what will actually run rather than what create_config would have picked.
  if (g_slim_kernel >= 0 && isA2 && a2Channels == slimlab::kChannels)
    return NbEngineSlim;
  return isA2 ? NbEngineA2Fast : NbEngineGeneric;
#else
  return isA2 ? NbEngineA2Fast : NbEngineGeneric;
#endif
}

} // namespace

// --- Model handle ------------------------------------------------------------

struct NbModel
{
  std::unique_ptr<nam::DSP> dsp;
  NbEngine engine = NbEngineUnknown;
  int32_t channels = 0;
  double sampleRate = 0.0;
  int32_t blockSize = 64;
  std::vector<NAM_SAMPLE> scratch;
};

// --- Exported API ------------------------------------------------------------

extern "C" {

NB_EXPORT const char* NB_FN(_variant_name)(void)
{
  return NB_STRINGIFY(NB_VARIANT_NAME);
}

NB_EXPORT int NB_FN(_has_fused)(void)
{
#if defined(NAM_ENABLE_FUSED)
  return nam::wavenet::fused::available() ? 1 : 0;
#else
  return 0;
#endif
}

NB_EXPORT int NB_FN(_probe)(const uint8_t* namBytes, size_t len, NbSubmodel mode, int32_t index, NbProbe* out,
                            char* err, size_t errLen)
{
  if (namBytes == nullptr || len == 0 || out == nullptr)
  {
    set_error(err, errLen, "probe: null or empty input");
    return 1;
  }

  try
  {
    const auto root = nlohmann::json::parse(namBytes, namBytes + len);

    Selection selection;
    std::string selectionError;
    if (!select_submodel(root, mode, index, selection, selectionError))
    {
      set_error(err, errLen, selectionError);
      return 1;
    }

    const nlohmann::json& model = *selection.model;
    const auto configIt = model.find("config");
    if (configIt == model.end() || !configIt->is_object())
    {
      set_error(err, errLen, "selected model has no \"config\" object");
      return 1;
    }

    int32_t channels = 0;
    out->engine = detect_engine(*configIt, &channels);
    out->channels = channels;
    out->sampleRate = nam::get_sample_rate_from_nam_file(model);
    out->submodelIndex = selection.index;
    out->submodelCount = selection.count;
    out->submodelMaxValue = selection.maxValue;
    return 0;
  }
  catch (const std::exception& e)
  {
    set_error(err, errLen, std::string("probe failed: ") + e.what());
    return 1;
  }
}

NB_EXPORT NbModel* NB_FN(_create)(const uint8_t* namBytes, size_t len, NbSubmodel mode, int32_t index,
                                  int32_t blockSize, char* err, size_t errLen)
{
  if (namBytes == nullptr || len == 0)
  {
    set_error(err, errLen, "create: null or empty input");
    return nullptr;
  }
  if (blockSize <= 0)
  {
    set_error(err, errLen, "create: blockSize must be positive");
    return nullptr;
  }

  try
  {
    const auto root = nlohmann::json::parse(namBytes, namBytes + len);

    Selection selection;
    std::string selectionError;
    if (!select_submodel(root, mode, index, selection, selectionError))
    {
      set_error(err, errLen, selectionError);
      return nullptr;
    }

    const nlohmann::json& model = *selection.model;
    const auto configIt = model.find("config");
    if (configIt == model.end() || !configIt->is_object())
    {
      set_error(err, errLen, "selected model has no \"config\" object");
      return nullptr;
    }

    int32_t channels = 0;
    const NbEngine engine = detect_engine(*configIt, &channels);

    std::unique_ptr<nam::DSP> dsp;
#if defined(NB_ENABLE_SLIM_LAB)
    if (engine == NbEngineSlim)
    {
      // The lab kernels are not reachable through create_config — that is the
      // point of the lab — so build one directly from the same weight stream
      // get_dsp would have handed to A2FastModel.
      const auto weightsIt = model.find("weights");
      if (weightsIt == model.end() || !weightsIt->is_array())
      {
        set_error(err, errLen, "selected model has no \"weights\" array");
        return nullptr;
      }
      dsp = slimlab::create(g_slim_kernel, *configIt, weightsIt->get<std::vector<float>>(),
                            nam::get_sample_rate_from_nam_file(model));
    }
    else
#endif
    {
#if defined(NAM_ENABLE_FUSED)
      // Explicit rather than relying on Auto: Auto also happens to select fused
      // for this shape today, but only because fused is tested first.
      const nam::wavenet::ScopedEnginePrefer prefer(nam::wavenet::EnginePrefer::FusedNeon);
#endif
      // Skip the load-time prewarm; the benchmark's own warm-up loop runs the
      // whole file repeatedly, which settles far more than prewarm() does.
      nam::DspLoadOptions options;
      options.prewarm = false;
      dsp = nam::get_dsp(model, options);
    }

    if (!dsp)
    {
      set_error(err, errLen, "get_dsp returned null");
      return nullptr;
    }

    auto handle = std::make_unique<NbModel>();
    handle->dsp = std::move(dsp);
    handle->engine = engine;
    handle->channels = channels;
    handle->sampleRate = handle->dsp->GetExpectedSampleRate();
    if (handle->sampleRate <= 0.0)
      handle->sampleRate = 48000.0;
    handle->blockSize = blockSize;
    handle->scratch.assign(static_cast<size_t>(blockSize), NAM_SAMPLE(0));

    handle->dsp->SetPrewarmOnReset(false);
    handle->dsp->Reset(handle->sampleRate, blockSize);

    return handle.release();
  }
  catch (const std::exception& e)
  {
    set_error(err, errLen, std::string("create failed: ") + e.what());
    return nullptr;
  }
}

NB_EXPORT void NB_FN(_destroy)(NbModel* model)
{
  delete model;
}

NB_EXPORT NbEngine NB_FN(_engine)(const NbModel* model)
{
  return model != nullptr ? model->engine : NbEngineUnknown;
}

NB_EXPORT int32_t NB_FN(_channels)(const NbModel* model)
{
  return model != nullptr ? model->channels : 0;
}

NB_EXPORT double NB_FN(_sample_rate)(const NbModel* model)
{
  return model != nullptr ? model->sampleRate : 0.0;
}

NB_EXPORT void NB_FN(_reset)(NbModel* model, double sampleRate, int32_t blockSize)
{
  if (model == nullptr || !model->dsp || blockSize <= 0)
    return;

  model->blockSize = blockSize;
  if (model->scratch.size() != static_cast<size_t>(blockSize))
    model->scratch.assign(static_cast<size_t>(blockSize), NAM_SAMPLE(0));

  // SetPrewarmOnReset(false) was set at construction, so this only rewinds
  // state and buffer sizes — no prewarm cost, and callers keep it outside the
  // timed region regardless.
  model->dsp->Reset(sampleRate > 0.0 ? sampleRate : model->sampleRate, blockSize);
}

NB_EXPORT uint64_t NB_FN(_process)(NbModel* model, const double* in, size_t frames, double* out,
                                   double* outChecksum)
{
  if (model == nullptr || !model->dsp || in == nullptr || frames == 0)
    return 0;

  const size_t blockSize = static_cast<size_t>(model->blockSize);
  const bool wantChecksum = outChecksum != nullptr;
  double checksum = 0.0;

  const uint64_t previousFpcr = denormals_disable();
  const uint64_t start = now_ns();

  for (size_t offset = 0; offset < frames;)
  {
    const size_t count = std::min(blockSize, frames - offset);

    // process() takes input[channel][frame]; this model is mono in, mono out,
    // so each "channel array" is a single pointer into the shared buffers.
    NAM_SAMPLE* inputChannel = const_cast<NAM_SAMPLE*>(in + offset);
    NAM_SAMPLE* outputChannel = (out != nullptr) ? (out + offset) : model->scratch.data();
    NAM_SAMPLE* input[1] = {inputChannel};
    NAM_SAMPLE* output[1] = {outputChannel};

    model->dsp->process(input, output, static_cast<int>(count));

    // Only summed when the caller asks. Timed passes pass null so the measured
    // region is process() and nothing else; the work still cannot be elided,
    // because process() is a virtual call across a dylib boundary.
    if (wantChecksum)
    {
      for (size_t i = 0; i < count; i++)
        checksum += static_cast<double>(outputChannel[i]);
    }

    offset += count;
  }

  const uint64_t elapsed = now_ns() - start;
  denormals_restore(previousFpcr);

  if (wantChecksum)
    *outChecksum = checksum;

  return elapsed;
}

#if defined(NB_ENABLE_SLIM_LAB)

NB_EXPORT int NB_FN(_kernel_count)(void)
{
  return slimlab::kernel_count();
}

NB_EXPORT const char* NB_FN(_kernel_name)(int index)
{
  return slimlab::kernel_name(index);
}

NB_EXPORT int NB_FN(_select_kernel)(int index)
{
  if (index < 0 || index >= slimlab::kernel_count())
    return 1;
  g_slim_kernel = index;
  return 0;
}

#endif // NB_ENABLE_SLIM_LAB

} // extern "C"
