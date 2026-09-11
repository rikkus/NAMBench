// Shared C API exposed by each engine framework.
//
// Every variant under test is compiled into its own dynamic framework, built
// from its own checkout with identical flags and -fvisibility=hidden. Only the
// handful of symbols declared here are exported, under a per-variant prefix, so
// the app can link two frameworks that both define nam:: internally without
// their symbols colliding.
//
// This header only defines the shape of that API. Each framework's umbrella
// header instantiates it for its own prefix (see NAMEngineUpstream.h,
// NAMEnginePlanar.h, NAMEngineSlim.h, NAMEngineFull.h).

#ifndef NAM_BENCH_SHIM_H
#define NAM_BENCH_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The Xcode build compiles each variant into its own dynamic framework with
// -fvisibility=hidden, so this attribute is what makes the handful of symbols
// below the only ones exported. The portable conformance build (CMakeLists.txt)
// links one variant per *executable* instead, where nothing needs exporting —
// hence the empty definition under MSVC, which has no such attribute.
#if defined(_MSC_VER)
  #define NB_EXPORT
#else
  #define NB_EXPORT __attribute__((visibility("default")))
#endif

/// Which WaveNet implementation a config will be routed to.
///
/// Reported by asking the engine's own public shape detectors, mirroring the
/// dispatch in wavenet::create_config — not assumed from the build flags.
typedef enum NbEngine {
  NbEngineUnknown = 0,
  /// Generic Eigen WaveNet. Never an expected result here; indicates a
  /// mismatched shape and a run that must be rejected.
  NbEngineGeneric = 1,
  /// Upstream's specialised A2 path (Eigen GEMM).
  NbEngineA2Fast = 2,
  /// One of the in-project experimental kernels for the slimmed (3-channel)
  /// submodel. Only the slim-lab build can report this, and only once a kernel
  /// has been selected — so "I asked for a lab kernel and got a lab kernel" is
  /// still asserted rather than assumed.
  NbEngineSlim = 4,
  /// One of the in-project experimental kernels for the full (8-channel)
  /// submodel. Same rule as NbEngineSlim: only the full-lab build reports it,
  /// and only once a kernel has been selected. Before that the full-lab build
  /// routes and reports as `a2_fast`, which is the engine it is built alongside.
  NbEngineFull = 5,
  /// The planar NEON kernels proposed to Core in PR #313, covering both the
  /// 3-channel and 8-channel A2 submodels.
  ///
  /// Reported only when the build actually has them. `NAM_A2_PLANAR` is defined
  /// wherever `__aarch64__` is, so off AArch64 the planar checkout compiles to
  /// plain `a2_fast` and this build honestly reports `NbEngineA2Fast`. That
  /// distinction is the whole reason the engine is asked for rather than
  /// inferred: a "planar" number that was quietly `a2_fast` would read as the
  /// kernels having achieved nothing.
  NbEnginePlanar = 6,
  /// One of the in-project experimental kernels for 32-bit ARM (ARMv7-A with
  /// NEON and VFPv4). Unlike the other two labs this one covers *both* A2
  /// submodels, so the same rule is enforced one step further: it is reported
  /// only once a kernel has been selected AND that kernel's channel count
  /// matches the submodel actually loaded. Selecting a 3-channel kernel and
  /// running the 8-channel submodel surfaces as a routing failure rather than
  /// as a silent a2_fast measurement.
  NbEngineA32 = 7,
} NbEngine;

/// Which submodel of a SlimmableContainer to run.
///
/// A2 captures ship as a container holding a nano submodel (3 channels,
/// max_value 0.5) and a full one (8 channels, max_value 1.0). Selection is by
/// max_value rather than position, so a reordering in the trainer cannot
/// silently change which one is measured.
typedef enum NbSubmodel {
  /// Greatest max_value — A2 full. The historical behaviour and the default.
  NbSubmodelWidest = 0,
  /// Least max_value — A2 nano, the slimmed path.
  NbSubmodelNarrowest = 1,
  /// A literal index into the submodels array.
  NbSubmodelIndex = 2,
} NbSubmodel;

/// What a .nam turned out to contain, and how it will be run.
typedef struct NbProbe {
  NbEngine engine;
  /// WaveNet channel count. 8 for A2 full, 3 for A2 nano.
  int32_t channels;
  double sampleRate;
  /// Which submodel of a SlimmableContainer was selected, and its max_value.
  /// submodelCount is 0 for a plain (non-container) .nam.
  int32_t submodelIndex;
  int32_t submodelCount;
  double submodelMaxValue;
} NbProbe;

typedef struct NbModel NbModel;

/// Declare the exported API for one variant prefix.
#define NB_DECLARE_VARIANT(P)                                                                     \
  /** Human-readable variant name, e.g. "upstream". */                                            \
  NB_EXPORT const char* P##_variant_name(void);                                                   \
                                                                                                  \
  /** Whether this build compiled in the (retired) fused NEON engine. Always   \
   *  0 now that the fused engine has been removed from every variant; the    \
   *  symbol is kept because it is generated by this shared macro for every   \
   *  variant prefix, and dropping it here would break every variant's        \
   *  linkage rather than just the one that used to report true. */          \
  NB_EXPORT int P##_has_fused(void);                                                              \
                                                                                                  \
  /** Inspect .nam bytes without building a model. 0 on success. */                               \
  NB_EXPORT int P##_probe(const uint8_t* namBytes, size_t len, NbSubmodel mode, int32_t index,    \
                          NbProbe* out, char* err, size_t errLen);                                \
                                                                                                  \
  /** Build the selected submodel. NULL on failure, with a message in err. */                     \
  NB_EXPORT NbModel* P##_create(const uint8_t* namBytes, size_t len, NbSubmodel mode,             \
                                int32_t index, int32_t blockSize, char* err, size_t errLen);      \
                                                                                                  \
  NB_EXPORT void P##_destroy(NbModel* model);                                                     \
  NB_EXPORT NbEngine P##_engine(const NbModel* model);                                            \
  NB_EXPORT int32_t P##_channels(const NbModel* model);                                           \
  NB_EXPORT double P##_sample_rate(const NbModel* model);                                          \
                                                                                                  \
  /** Clear model state so every pass starts identically. Never prewarms. */                      \
  NB_EXPORT void P##_reset(NbModel* model, double sampleRate, int32_t blockSize);                 \
                                                                                                  \
  /**                                                                                             \
   * Run `frames` samples through the model in fixed blockSize chunks.                            \
   *                                                                                              \
   * Returns elapsed nanoseconds for the block loop, measured inside the shim so                  \
   * Swift bridging cost stays outside the measured region. `out` may be NULL,                    \
   * in which case output goes to internal scratch; pass a real buffer only for                   \
   * the parity pass. `outChecksum` receives the sum of all output samples, so                    \
   * the optimiser cannot elide the work.                                                         \
   */                                                                                             \
  NB_EXPORT uint64_t P##_process(NbModel* model, const double* in, size_t frames, double* out,    \
                                 double* outChecksum);

/// The extra API of a kernel-lab framework, declared separately so the
/// `upstream` and `planar` frameworks do not grow symbols they have no
/// implementation for. Both labs — slim (3-channel) and full (8-channel) —
/// expose it under their own prefix.
#define NB_DECLARE_KERNEL_LAB(P)                                                                  \
  /** How many experimental kernels this build carries. */                                        \
  NB_EXPORT int P##_kernel_count(void);                                                           \
                                                                                                  \
  /** Name of kernel `index`, or NULL if out of range. */                                         \
  NB_EXPORT const char* P##_kernel_name(int index);                                               \
                                                                                                  \
  /**                                                                                             \
   * Choose the kernel that subsequent _create calls will build.                                  \
   *                                                                                              \
   * Returns 0 on success. Until a kernel has been selected the build routes and                  \
   * reports as the shipping engine it was built alongside — a2_fast for every                    \
   * lab — so a run that forgot to select one is caught by the engine assertion                   \
   * rather than quietly measuring the wrong thing.                                                \
   */                                                                                             \
  NB_EXPORT int P##_select_kernel(int index);                                                     \
                                                                                                  \
  /**                                                                                             \
   * Channel count kernel `index` is written for: 3 (A2 nano) or 8 (A2 standard).                 \
   * -1 if the index is out of range.                                                             \
   *                                                                                              \
   * The single-submodel labs answer with their one value. It is asked rather                     \
   * than assumed so that the driver and the conformance runner can pair a kernel                 \
   * with its submodel mechanically, which is what lets one lab carry both.                       \
   */                                                                                             \
  NB_EXPORT int P##_kernel_channels(int index);                                                   \
                                                                                                  \
  /**                                                                                             \
   * Whether kernel `index` claims to be bit-identical to its reference engine on                 \
   * this target. 1 = claims exactness, 0 = deliberately reassociates, -1 = out of                \
   * range.                                                                                       \
   *                                                                                              \
   * The claim lives here, next to the kernel that makes it, rather than in a name                \
   * convention the comparator has to decode. On AArch64 only the verbatim ports                  \
   * claimed exactness and a suffix rule sufficed; in the a32 lab almost every                    \
   * kernel claims it and a couple deliberately do not, which a suffix rule would                 \
   * not carry.                                                                                   \
   */                                                                                             \
  NB_EXPORT int P##_kernel_exact(int index);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NAM_BENCH_SHIM_H
