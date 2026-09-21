// One implementation, compiled once per AudioDSPTools variant.
//
// The variant is selected entirely by build settings:
//   NB_IR_PREFIX          symbol prefix, e.g. nb_ir_upstream
//   NB_IR_VARIANT_NAME    display name, e.g. adt_upstream
//   NB_IR_HAS_PARTITIONED defined only for the checkout that has
//                         PartitionedConvolution
//
// Everything else — sources, flags, include paths, and one shared Eigen tree —
// is identical between the two, so any measured difference has to come from the
// convolution.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <time.h>

#include "dsp/ImpulseResponse.h"

#include "nam_ir_shim.h"
#include "nb_shim_timing.h"

#if !defined(NB_IR_PREFIX)
  #error "NB_IR_PREFIX must be defined (e.g. -DNB_IR_PREFIX=nb_ir_upstream)"
#endif
#if !defined(NB_IR_VARIANT_NAME)
  #error "NB_IR_VARIANT_NAME must be defined (e.g. -DNB_IR_VARIANT_NAME=adt_upstream, unquoted)"
#endif

#define NB_IR_CAT_(a, b) a##b
#define NB_IR_CAT(a, b) NB_IR_CAT_(a, b)
#define NB_IR_FN(suffix) NB_IR_CAT(NB_IR_PREFIX, suffix)

#define NB_IR_STRINGIFY_(x) #x
#define NB_IR_STRINGIFY(x) NB_IR_STRINGIFY_(x)

namespace
{

void set_error(char* err, size_t errLen, const std::string& message)
{
  if (err == nullptr || errLen == 0)
    return;
  std::snprintf(err, errLen, "%s", message.c_str());
}

// ---------------------------------------------------------------------------
// The impulse response under test.
//
// Generated rather than loaded, because what a convolution costs is set by how
// many taps it has and not by what is in them: both implementations touch every
// tap of every block regardless of their values. A generated IR is identical on
// every machine and in every variant, needs no asset committed to the
// repository, and cannot quietly differ between two runs being compared.
//
// The shape is a decaying, sign-alternating impulse train — roughly the
// envelope of a speaker cabinet measurement, without being one. Built from an
// integer LCG and a multiplicative decay, so there is no libm call in it and
// every platform produces the same floats. The decay floor keeps the tail well
// clear of denormal range on its own, rather than relying on the flush-to-zero
// this shim sets: a denormal tail would be a property of the test signal rather
// than of the implementations, and it would land on whichever of them reads the
// oldest samples.
// ---------------------------------------------------------------------------
std::vector<float> generate_ir(int32_t taps, uint32_t seed)
{
  std::vector<float> ir(static_cast<size_t>(taps), 0.0f);
  uint32_t state = seed;
  float envelope = 1.0f;
  // ~8 ms of decay at 48 kHz per e-fold, floored so the tail stays a normal
  // float no matter how long the IR is.
  const float decay = 0.9975f;
  for (int32_t i = 0; i < taps; i++)
  {
    state = state * 1664525u + 1013904223u;
    // Top 16 bits, mapped to [-1, 1). Exact in float.
    const int32_t bits = static_cast<int32_t>(state >> 16) - 32768;
    const float noise = static_cast<float>(bits) * (1.0f / 32768.0f);
    ir[static_cast<size_t>(i)] = noise * envelope;
    envelope *= decay;
    if (envelope < 1.0e-6f)
      envelope = 1.0e-6f;
  }
  // A leading direct sound, as a real measurement has.
  ir[0] = 1.0f;
  return ir;
}

} // namespace

struct NbIr
{
  std::unique_ptr<dsp::ImpulseResponse> ir;
  int32_t blockSize = 64;
  int32_t irChannels = 1;
  /// Taps after AudioDSPTools' 8192 truncation — what the cost is a function
  /// of, rather than the number asked for.
  int32_t taps = 0;
  double sampleRate = 48000.0;
  /// Mono input, copied per block because Process takes a mutable double**.
  std::vector<double> inputScratch;
  std::vector<double*> inputPointers;
  /// Per-block clock reads, one more than there are blocks. Sized in _reset so
  /// nothing allocates inside the timed loop.
  std::vector<uint64_t> stamps;
};

extern "C" {

// Declared up front so _ir_create can use it to put a new handle into the same
// state a later reset would: one place decides what "ready to process" means.
NB_IR_EXPORT void NB_IR_FN(_ir_reset)(NbIr* ir, int32_t blockSize);

NB_IR_EXPORT const char* NB_IR_FN(_ir_variant_name)(void)
{
  return NB_IR_STRINGIFY(NB_IR_VARIANT_NAME);
}

NB_IR_EXPORT int NB_IR_FN(_ir_can_select)(void)
{
#if defined(NB_IR_HAS_PARTITIONED)
  return 1;
#else
  return 0;
#endif
}

NB_IR_EXPORT NbIr* NB_IR_FN(_ir_create)(int32_t taps, int32_t irChannels, double sampleRate,
                                        NbIrImpl requested, int32_t blockSize, char* err,
                                        size_t errLen)
{
  if (taps <= 0 || blockSize <= 0 || sampleRate <= 0.0 || (irChannels != 1 && irChannels != 2))
  {
    set_error(err, errLen, "invalid IR parameters");
    return nullptr;
  }

#if !defined(NB_IR_HAS_PARTITIONED)
  if (requested == NbIrImplFFT)
  {
    // Refused rather than silently answered with the direct path. A run that
    // asked for FFT and got a direct convolution labelled FFT would be the one
    // failure this whole comparison exists to detect.
    set_error(err, errLen,
              "this variant has only the direct convolution; it cannot be asked for FFT");
    return nullptr;
  }
#endif

  // AudioDSPTools truncates at 8192 taps. The shim generates the IR at the
  // processing rate, so nothing is resampled and this is the only thing between
  // what was asked for and what is convolved.
  const int32_t effectiveTaps = std::min(taps, 8192);

  auto handle = std::make_unique<NbIr>();
  handle->taps = effectiveTaps;
  handle->blockSize = blockSize;
  handle->irChannels = irChannels;
  handle->sampleRate = sampleRate;

  dsp::ImpulseResponse::IRData data;
  data.mRawAudio = generate_ir(taps, 0x9E3779B9u);
  if (irChannels == 2)
    data.mRawAudioRight = generate_ir(taps, 0x85EBCA6Bu);
  // Generated at the processing rate, so _SetWeights does not resample and the
  // tap count measured is the tap count asked for (up to its 8192 truncation).
  data.mRawAudioSampleRate = sampleRate;

  try
  {
#if defined(NB_IR_HAS_PARTITIONED)
    dsp::ConvolutionImplementation implementation = dsp::ConvolutionImplementation::Auto;
    if (requested == NbIrImplDirect)
      implementation = dsp::ConvolutionImplementation::Direct;
    else if (requested == NbIrImplFFT)
      implementation = dsp::ConvolutionImplementation::FFT;
    handle->ir = std::make_unique<dsp::ImpulseResponse>(data, sampleRate, implementation);
#else
    handle->ir = std::make_unique<dsp::ImpulseResponse>(data, sampleRate);
#endif
  }
  catch (const std::exception& e)
  {
    set_error(err, errLen, std::string("could not build the impulse response: ") + e.what());
    return nullptr;
  }

#if defined(NB_IR_HAS_PARTITIONED)
  // This variant can say how many taps it ended up with; upstream cannot, so
  // the driver is told the number the shim derived. Checking the one against
  // the other here is what keeps that derivation honest for both.
  if (static_cast<int32_t>(handle->ir->GetNumTaps()) != effectiveTaps)
  {
    char detail[160];
    std::snprintf(detail, sizeof(detail),
                  "asked for %d taps and expected %d after truncation, but the convolver has %d",
                  taps, effectiveTaps, static_cast<int32_t>(handle->ir->GetNumTaps()));
    set_error(err, errLen, detail);
    return nullptr;
  }
#endif

  NB_IR_FN(_ir_reset)(handle.get(), blockSize);
  return handle.release();
}

NB_IR_EXPORT void NB_IR_FN(_ir_destroy)(NbIr* ir)
{
  delete ir;
}

NB_IR_EXPORT NbIrImpl NB_IR_FN(_ir_implementation)(const NbIr* ir)
{
  if (ir == nullptr || !ir->ir)
    return NbIrImplAuto;
#if defined(NB_IR_HAS_PARTITIONED)
  return ir->ir->GetActiveImplementation() == dsp::ConvolutionImplementation::FFT ? NbIrImplFFT
                                                                                 : NbIrImplDirect;
#else
  return NbIrImplDirect;
#endif
}

NB_IR_EXPORT int32_t NB_IR_FN(_ir_taps)(const NbIr* ir)
{
  if (ir == nullptr || !ir->ir)
    return 0;
  return ir->taps;
}

NB_IR_EXPORT int32_t NB_IR_FN(_ir_channels)(const NbIr* ir)
{
  if (ir == nullptr || !ir->ir)
    return 0;
  return static_cast<int32_t>(ir->ir->GetNumIRChannels());
}

NB_IR_EXPORT void NB_IR_FN(_ir_reset)(NbIr* ir, int32_t blockSize)
{
  if (ir == nullptr || !ir->ir || blockSize <= 0)
    return;
  ir->blockSize = blockSize;
  ir->inputScratch.assign(static_cast<size_t>(blockSize), 0.0);
  ir->inputPointers.assign(1, ir->inputScratch.data());
  // Reset takes the largest block it will be asked for, which is how both
  // variants are told to do their allocating here rather than in Process.
  ir->ir->Reset(static_cast<size_t>(blockSize), 1);
}

NB_IR_EXPORT uint64_t NB_IR_FN(_ir_process)(NbIr* ir, const double* in, size_t frames, double* out,
                                            double* outChecksum, uint64_t* blockNanos)
{
  if (ir == nullptr || !ir->ir || in == nullptr || frames == 0)
    return 0;

  const size_t blockSize = static_cast<size_t>(ir->blockSize);
  const size_t blocks = (frames + blockSize - 1) / blockSize;
  const bool wantChecksum = outChecksum != nullptr;
  const bool wantBlocks = blockNanos != nullptr;
  double checksum = 0.0;

  // One stamp per block boundary, plus the final one. The total returned is the
  // span across these same reads, so a per-block breakdown can never disagree
  // with the total it came from.
  if (wantBlocks && ir->stamps.size() != blocks + 1)
    ir->stamps.assign(blocks + 1, 0);

  const uint64_t previousFpcr = denormals_disable();
  const uint64_t start = now_ns();
  if (wantBlocks)
    ir->stamps[0] = start;

  size_t block = 0;
  for (size_t offset = 0; offset < frames; block++)
  {
    const size_t count = std::min(blockSize, frames - offset);

    // Process takes double** and may write through it, so the input is copied
    // into scratch rather than const_cast away. The copy is inside the timed
    // region for both variants alike; it is the same copy either way.
    std::memcpy(ir->inputScratch.data(), in + offset, count * sizeof(double));
    ir->inputPointers[0] = ir->inputScratch.data();

    double** outputs = ir->ir->Process(ir->inputPointers.data(), 1, count);

    if (out != nullptr)
      std::memcpy(out + offset, outputs[0], count * sizeof(double));

    // Only summed when the caller asks. Timed passes pass null so the measured
    // region is Process() and nothing else; the work still cannot be elided,
    // because Process() is a virtual call across a shared-library boundary.
    if (wantChecksum)
    {
      for (size_t i = 0; i < count; i++)
        checksum += outputs[0][i];
    }

    offset += count;
    if (wantBlocks)
      ir->stamps[block + 1] = now_ns();
  }

  const uint64_t end = wantBlocks ? ir->stamps[blocks] : now_ns();
  denormals_restore(previousFpcr);

  if (wantBlocks)
  {
    for (size_t i = 0; i < blocks; i++)
      blockNanos[i] = ir->stamps[i + 1] - ir->stamps[i];
  }
  if (wantChecksum)
    *outChecksum = checksum;

  return end - start;
}

} // extern "C"
