// The IR shim for NeuralAmpModelerCore's nam::Linear, compiled once per Core
// tree. See nam_ir_shim.h.
//
// The variant is selected entirely by build settings:
//   NB_IR_PREFIX          symbol prefix, e.g. nb_ir_linearplus
//   NB_IR_VARIANT_NAME    display name, e.g. linearplus
//
// Everything else — this file, flags, include paths, the generated impulse
// response and one shared Eigen tree — is identical between the trees, so any
// measured difference has to come from the tree's Linear.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/linear.h"

#include "nam_ir_shim.h"
#include "nb_ir_generate.h"
#include "nb_shim_timing.h"

#if !defined(NB_IR_PREFIX)
  #error "NB_IR_PREFIX must be defined (e.g. -DNB_IR_PREFIX=nb_ir_linear)"
#endif
#if !defined(NB_IR_VARIANT_NAME)
  #error "NB_IR_VARIANT_NAME must be defined (e.g. -DNB_IR_VARIANT_NAME=linear, unquoted)"
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

nam::LinearImplementation to_linear(NbIrImpl impl)
{
  switch (impl)
  {
    case NbIrImplDirect: return nam::LinearImplementation::Direct;
    case NbIrImplFFT: return nam::LinearImplementation::FFT;
    case NbIrImplAuto: break;
  }
  return nam::LinearImplementation::Auto;
}

// A tree whose plan has a tail tier reports its partition size there; one
// without the field has none.
template <typename Plan>
auto tail_partition_size(const Plan& plan, int) -> decltype(plan.tail_partition_size)
{
  return plan.tail_partition_size;
}
template <typename Plan>
int tail_partition_size(const Plan&, long)
{
  return 0;
}

} // namespace

struct NbIr
{
  std::unique_ptr<nam::Linear> linear;
  int32_t blockSize = 64;
  int32_t irChannels = 1;
  /// Core does not truncate, and the IR is generated at the processing rate so
  /// Reset does not resample: every tap asked for is convolved.
  int32_t taps = 0;
  double sampleRate = 48000.0;
  /// Mono input, copied per block for the same reason, and at the same place
  /// in the timed loop, as the AudioDSPTools shim does.
  std::vector<double> inputScratch;
  std::vector<std::vector<double>> outputScratch;
  std::vector<double*> inputPointers;
  std::vector<double*> outputPointers;
  /// Per-block clock reads, one more than there are blocks. Sized in _reset so
  /// nothing allocates inside the timed loop.
  std::vector<uint64_t> stamps;
};

extern "C" {

// Declared up front so _ir_create can use it to put a new handle into the same
// state a later reset would: one place decides what "ready to process" means.
NB_IR_EXPORT void NB_IR_FN(_ir_reset)(NbIr* ir, int32_t blockSize, int32_t maxBlockSize);

NB_IR_EXPORT const char* NB_IR_FN(_ir_variant_name)(void)
{
  return NB_IR_STRINGIFY(NB_IR_VARIANT_NAME);
}

NB_IR_EXPORT int NB_IR_FN(_ir_can_select)(void)
{
  return 1;
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

  auto handle = std::make_unique<NbIr>();
  handle->taps = taps;
  handle->blockSize = blockSize;
  handle->irChannels = irChannels;
  handle->sampleRate = sampleRate;

  // A stereo IR is a one-in, two-out Linear: one impulse response per output,
  // in channel order, which is how Core loads a stereo .wav.
  std::vector<float> weights = nb_generate_ir(taps, kNbIrSeedLeft);
  if (irChannels == 2)
  {
    const std::vector<float> right = nb_generate_ir(taps, kNbIrSeedRight);
    weights.insert(weights.end(), right.begin(), right.end());
  }

  try
  {
    // Trained at the processing rate, so Reset keeps the taps as they are.
    handle->linear = std::make_unique<nam::Linear>(1, irChannels, taps, false, weights, sampleRate,
                                                   to_linear(requested));
  }
  catch (const std::exception& e)
  {
    set_error(err, errLen, std::string("could not build the Linear model: ") + e.what());
    return nullptr;
  }

  NB_IR_FN(_ir_reset)(handle.get(), blockSize, blockSize);
  return handle.release();
}

NB_IR_EXPORT void NB_IR_FN(_ir_destroy)(NbIr* ir)
{
  delete ir;
}

NB_IR_EXPORT NbIrImpl NB_IR_FN(_ir_implementation)(const NbIr* ir)
{
  if (ir == nullptr || !ir->linear)
    return NbIrImplAuto;
  return ir->linear->GetActiveImplementation() == nam::LinearImplementation::FFT ? NbIrImplFFT
                                                                                 : NbIrImplDirect;
}

NB_IR_EXPORT int32_t NB_IR_FN(_ir_taps)(const NbIr* ir)
{
  if (ir == nullptr || !ir->linear)
    return 0;
  return ir->taps;
}

NB_IR_EXPORT int32_t NB_IR_FN(_ir_channels)(const NbIr* ir)
{
  if (ir == nullptr || !ir->linear)
    return 0;
  return ir->linear->NumOutputChannels();
}

// Linear does not expose how it split the IR, but it does expose the plan it
// chose from, and both trees follow their plan: the head is its direct taps,
// capped at the IR's length, and no partition is larger than its maximum.
NB_IR_EXPORT int32_t NB_IR_FN(_ir_head_taps)(const NbIr* ir)
{
  if (ir == nullptr || !ir->linear)
    return 0;
  if (ir->linear->GetActiveImplementation() != nam::LinearImplementation::FFT)
    return ir->taps;
  return std::min(ir->taps, nam::linear::select_fft_plan(ir->taps).direct_taps);
}

NB_IR_EXPORT int32_t NB_IR_FN(_ir_fft_block)(const NbIr* ir)
{
  if (ir == nullptr || !ir->linear || ir->linear->GetActiveImplementation() != nam::LinearImplementation::FFT)
    return 0;
  const nam::LinearFFTPlan plan = nam::linear::select_fft_plan(ir->taps);
  if (ir->taps <= plan.direct_taps)
    return 0;
  const int tail = tail_partition_size(plan, 0);
  return ir->taps > 2 * tail && tail > 0 ? tail : plan.max_partition_size;
}

NB_IR_EXPORT void NB_IR_FN(_ir_reset)(NbIr* ir, int32_t blockSize, int32_t maxBlockSize)
{
  if (ir == nullptr || !ir->linear || blockSize <= 0)
    return;
  ir->blockSize = blockSize;
  ir->inputScratch.assign(static_cast<size_t>(blockSize), 0.0);
  ir->outputScratch.assign(static_cast<size_t>(ir->irChannels), std::vector<double>(static_cast<size_t>(blockSize)));
  ir->inputPointers.assign(1, ir->inputScratch.data());
  ir->outputPointers.clear();
  for (auto& channel : ir->outputScratch)
    ir->outputPointers.push_back(channel.data());
  // Reset takes the largest block it will be asked for, which is how the model
  // is told to do its allocating here rather than in process(). It also clears
  // every kind of history, so each pass starts from the same state.
  ir->linear->Reset(ir->sampleRate, std::max(blockSize, maxBlockSize));
}

NB_IR_EXPORT uint64_t NB_IR_FN(_ir_process)(NbIr* ir, const double* in, size_t frames, double* out,
                                            double* outChecksum, uint64_t* blockNanos)
{
  if (ir == nullptr || !ir->linear || in == nullptr || frames == 0)
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

    // process() takes NAM_SAMPLE** and is entitled to write through it, so the
    // input is copied into scratch rather than const_cast away.
    std::memcpy(ir->inputScratch.data(), in + offset, count * sizeof(double));

    ir->linear->process(ir->inputPointers.data(), ir->outputPointers.data(), static_cast<int>(count));

    if (out != nullptr)
      std::memcpy(out + offset, ir->outputScratch[0].data(), count * sizeof(double));

    // Only summed when the caller asks. Timed passes pass null so the measured
    // region is process() and nothing else; the work still cannot be elided,
    // because process() is a virtual call across a shared-library boundary.
    if (wantChecksum)
    {
      for (size_t i = 0; i < count; i++)
        checksum += ir->outputScratch[0][i];
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
