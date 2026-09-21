// Shared C API exposed by each impulse-response variant.
//
// The same arrangement as nam_bench_shim.h, for the same reason: both
// AudioDSPTools checkouts define dsp::ImpulseResponse, so they can only be
// measured in one process if each is compiled into its own shared library with
// -fvisibility=hidden, exporting nothing but the prefixed functions below.
//
// What is being compared is the convolution, so the shim keeps everything
// around it identical between the variants. The impulse response is generated
// here rather than loaded from a file: the cost of convolving is set by the tap
// count alone and not by what the taps contain, so a synthetic IR measures the
// same thing as a real cabinet while staying byte-identical on every machine
// and carrying nobody's licence.

#ifndef NAM_IR_SHIM_H
#define NAM_IR_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
  #define NB_IR_EXPORT
#else
  #define NB_IR_EXPORT __attribute__((visibility("default")))
#endif

/// Which convolution a variant was asked for, and which it actually chose.
typedef enum NbIrImpl {
  /// Let the variant decide from the tap count. Upstream has only one
  /// implementation, so it always answers Direct.
  NbIrImplAuto = 0,
  /// Time-domain FIR: one dot product of `taps` values per output sample.
  NbIrImplDirect = 1,
  /// Zero-latency partitioned FFT: a direct head covering the first block,
  /// plus uniformly partitioned FFT convolution for the tail.
  NbIrImplFFT = 2,
} NbIrImpl;

typedef struct NbIr NbIr;

/// Declare the exported API for one variant prefix.
#define NB_IR_DECLARE_VARIANT(P)                                                                  \
  /** Human-readable variant name, e.g. "adt_upstream". */                                        \
  NB_IR_EXPORT const char* P##_ir_variant_name(void);                                             \
                                                                                                  \
  /**                                                                                             \
   * Whether this build can be asked for a particular implementation.                             \
   *                                                                                              \
   * 0 for upstream, which has exactly one. The driver asks rather than infers                    \
   * so that requesting FFT from a build that has none is a refusal instead of a                  \
   * direct-convolution measurement wearing an FFT label.                                         \
   */                                                                                             \
  NB_IR_EXPORT int P##_ir_can_select(void);                                                       \
                                                                                                  \
  /**                                                                                             \
   * Build an impulse response of `taps` taps, at `sampleRate`, for blocks of up                  \
   * to `blockSize` frames. `irChannels` is 1 (mono) or 2 (stereo).                                \
   *                                                                                              \
   * The taps are generated, identically in every variant and on every machine —                  \
   * see the note at the top of this header. NULL on failure, with a message in                   \
   * err.                                                                                          \
   */                                                                                             \
  NB_IR_EXPORT NbIr* P##_ir_create(int32_t taps, int32_t irChannels, double sampleRate,           \
                                   NbIrImpl requested, int32_t blockSize, char* err,              \
                                   size_t errLen);                                                \
                                                                                                  \
  NB_IR_EXPORT void P##_ir_destroy(NbIr* ir);                                                     \
                                                                                                  \
  /** Which implementation is actually convolving. */                                             \
  NB_IR_EXPORT NbIrImpl P##_ir_implementation(const NbIr* ir);                                    \
                                                                                                  \
  /**                                                                                             \
   * Taps after resampling and the 8192-tap truncation, which is what the cost                    \
   * is a function of — not the number asked for.                                                 \
   */                                                                                             \
  NB_IR_EXPORT int32_t P##_ir_taps(const NbIr* ir);                                               \
  NB_IR_EXPORT int32_t P##_ir_channels(const NbIr* ir);                                           \
                                                                                                  \
  /** Clear history so every pass starts identically, off the audio thread. */                    \
  NB_IR_EXPORT void P##_ir_reset(NbIr* ir, int32_t blockSize);                                    \
                                                                                                  \
  /**                                                                                             \
   * Run `frames` samples through in fixed blockSize chunks.                                      \
   *                                                                                              \
   * Returns elapsed nanoseconds for the block loop, measured inside the shim so                  \
   * nothing outside the library is in the measured region. `out` may be NULL,                    \
   * in which case output goes to internal scratch; pass a real buffer for the                    \
   * parity pass, where it receives the first IR channel. `outChecksum` receives                  \
   * the sum of all output samples, so the optimiser cannot elide the work.                       \
   *                                                                                              \
   * `blockNanos`, if non-NULL, receives one duration per block and must have                     \
   * room for (frames + blockSize - 1) / blockSize entries. It is what makes the                  \
   * worst-case block visible: partitioned FFT convolution does a whole                           \
   * partition's work in one callback, so a mean that looks good can hide a                       \
   * block that misses its deadline. The total returned is the span across the                    \
   * same clock reads the per-block times come from, so the two always agree.                     \
   */                                                                                             \
  NB_IR_EXPORT uint64_t P##_ir_process(NbIr* ir, const double* in, size_t frames, double* out,    \
                                       double* outChecksum, uint64_t* blockNanos);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NAM_IR_SHIM_H
