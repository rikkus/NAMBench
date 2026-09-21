// Portable benchmark driver.
//
// This one does measure. It exists because the numbers this project publishes
// come from the Xcode build, and the Xcode build does not run on a Raspberry Pi.
// A Cortex-A76 with 512 KB of L2 and no SVE is the least Apple-like AArch64 core
// the planar kernels are likely to meet, which makes it the most interesting
// place to run them — and the one place none of the existing tooling reaches.
//
// The protocol is a port of BenchCore's, not an approximation of it. Same
// warm-up, same timing window, same tightest-70% selection, same
// (max - min) / min spread test, same retry-and-reject. A number produced here
// and a number produced by `nambench` are meant to be read side by side, so
// anything that differed would have to be a bug.
//
// Unlike Tools/nam_conformance.cpp, this links every engine into ONE process, as
// the app and the CLI do — four shared libraries, each hiding its own nam::
// symbols and exporting only its prefixed C API. Measuring two engines in two
// processes would put the allocator, the page cache and the scheduler between
// them.
//
// Not built under MSVC: the shared-library layout relies on -fvisibility=hidden
// plus a default-visibility attribute on the exported API, which is not how
// Windows DLLs work. Windows gets conformance, which is what it is for.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include "nam_bench_shim.h"
#include "nb_counters.h"
#include "nb_protocol.h"

#if defined(__APPLE__)
  #include <pthread.h>
  #include <sys/sysctl.h>
  #include <sys/types.h>
#endif
#include <unistd.h>

extern "C" {
NB_DECLARE_VARIANT(nb_upstream)
NB_DECLARE_VARIANT(nb_planar)
#if defined(NAMBENCH_HAVE_LABS)
NB_DECLARE_VARIANT(nb_slim)
NB_DECLARE_KERNEL_LAB(nb_slim)
NB_DECLARE_VARIANT(nb_full)
NB_DECLARE_KERNEL_LAB(nb_full)
#endif
#if defined(NAMBENCH_HAVE_A32_LAB)
NB_DECLARE_VARIANT(nb_a32)
NB_DECLARE_KERNEL_LAB(nb_a32)
#endif
}

namespace
{

namespace fs = std::filesystem;

// The protocol, used unqualified throughout: Audio and load_wav, the statistics,
// the environment capture, Attempt/Result/Parity, and measure() itself.
using namespace nbp;

// ---------------------------------------------------------------------------
// One engine's exported API, gathered behind function pointers.
//
// The shim exposes the same set of functions under a different prefix per
// variant. Collecting them here means the measurement loop below is written
// once, and cannot accidentally do something to one variant that it does not do
// to another — which is the same reason project.yml builds all four frameworks
// from one target template.
// ---------------------------------------------------------------------------
struct EngineApi
{
  const char* name = nullptr;
  const char* repository = nullptr;
  const char* codePath = nullptr;

  int (*probe)(const uint8_t*, size_t, NbSubmodel, int32_t, NbProbe*, char*, size_t) = nullptr;
  NbModel* (*create)(const uint8_t*, size_t, NbSubmodel, int32_t, int32_t, char*, size_t) = nullptr;
  void (*destroy)(NbModel*) = nullptr;
  NbEngine (*engine)(const NbModel*) = nullptr;
  int32_t (*channels)(const NbModel*) = nullptr;
  double (*sample_rate)(const NbModel*) = nullptr;
  void (*reset)(NbModel*, double, int32_t) = nullptr;
  uint64_t (*process)(NbModel*, const double*, size_t, double*, double*) = nullptr;

  // Null on the two shipping engines, which carry no kernel lab.
  int (*kernel_count)() = nullptr;
  const char* (*kernel_name)(int) = nullptr;
  int (*select_kernel)(int) = nullptr;
  /// Channel count a kernel is written for. Asked rather than assumed, so one
  /// lab can carry kernels for both A2 submodels.
  int (*kernel_channels)(int) = nullptr;
  /// Whether a kernel claims bit-identity with its reference on this target.
  int (*kernel_exact)(int) = nullptr;
};

#define NB_FILL_BASE(api, P)                                                                       \
  do                                                                                               \
  {                                                                                                \
    (api).probe = &P##_probe;                                                                      \
    (api).create = &P##_create;                                                                    \
    (api).destroy = &P##_destroy;                                                                  \
    (api).engine = &P##_engine;                                                                    \
    (api).channels = &P##_channels;                                                                \
    (api).sample_rate = &P##_sample_rate;                                                          \
    (api).reset = &P##_reset;                                                                      \
    (api).process = &P##_process;                                                                  \
  } while (0)

#define NB_FILL_LAB(api, P)                                                                        \
  do                                                                                               \
  {                                                                                                \
    (api).kernel_count = &P##_kernel_count;                                                        \
    (api).kernel_name = &P##_kernel_name;                                                          \
    (api).select_kernel = &P##_select_kernel;                                                      \
    (api).kernel_channels = &P##_kernel_channels;                                                  \
    (api).kernel_exact = &P##_kernel_exact;                                                        \
  } while (0)

/// One thing to measure: an engine, and for a lab, which kernel of it.
struct Subject
{
  std::string name; // "a2_fast", "a2_planar", "slim:planar", "full:fu_s8_head"
  const EngineApi* api = nullptr;
  int kernel = -1;
};

// ---------------------------------------------------------------------------
// Configuration
//
// The protocol's own settings — warm-up, window, acceptance, attempts — come
// from nbp::ProtocolConfig and are inherited rather than restated, so this
// driver cannot drift away from the standard the IR driver measures to. What is
// added here is only what is specific to timing a WaveNet.
// ---------------------------------------------------------------------------

struct Config : nbp::ProtocolConfig
{
  int blockSize = 64;
  NbSubmodel submodel = NbSubmodelWidest;
  bool checkParity = true;
  double parityWarnRelativeRMS = 1e-4;
  /// PMU events to bracket each timed pass with. Empty means none, which is the
  /// default: opening counters costs two syscalls per event per pass and, more
  /// importantly, the numbers are only meaningful to someone who asked for them.
  std::vector<std::string> counterEvents;
};

const char* engine_name(NbEngine engine)
{
  switch (engine)
  {
    case NbEngineGeneric: return "generic";
    case NbEngineA2Fast: return "a2_fast";
    case NbEngineSlim: return "slim";
    case NbEngineFull: return "full";
    case NbEnginePlanar: return "a2_planar";
    case NbEngineA32: return "a32";
    case NbEngineUnknown: break;
  }
  return "unknown";
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
  // Quality of service, before anything is timed.
  //
  // BenchCore runs its measurement on a QOS_CLASS_USER_INTERACTIVE thread, and
  // the comment there explains why: on Apple silicon a lower-QoS thread is
  // scheduled onto the efficiency cores, and an E-core measurement would swamp
  // every effect this benchmark is trying to resolve. A driver that did not do
  // the same would not be measuring the same machine — which showed up, back
  // when this driver still measured the (now retired) `fused` engine, as this
  // driver reporting it 1.9% faster and `upstream` 0.7% slower than the Swift
  // CLI on the same hardware, in the same minute.
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  Config config;
  std::string modelPath, audioPath;
  fs::path outputPath = "benchmark.json";
  std::string slimSpec, fullSpec, a32Spec;
  std::string testbedNote;

  auto usage = []() {
    std::printf(
      "nam_benchmark — the NAMBench protocol, without Xcode\n"
      "\n"
      "  --model, -m <path>       .nam capture (required)\n"
      "  --audio, -a <path>       input .wav (required)\n"
      "  --output, -o <path>      JSON report to write (default: benchmark.json)\n"
      "\n"
      "  --submodel <which>       widest (default) | narrowest\n"
      "  --slim <list>            slim-lab kernels: all, none, names or indices\n"
      "  --full <list>            full-lab kernels, same format\n"
      "  --a32 <list>             a32-lab kernels (ARMv7), same format\n"
      "  --list-slim/--list-full  print a kernel table and exit\n"
      "  --list-a32               print the a32 kernel table and exit\n"
      "\n"
      "  --block-size <n>         frames per process() call (default 64)\n"
      "  --warmup-seconds <s>     discarded warm-up (default 5)\n"
      "  --timing-seconds <s>     timing window (default 30)\n"
      "  --accept-fraction <r>    fraction of samples kept (default 0.70)\n"
      "  --accept-tolerance <r>   required agreement (default 0.03)\n"
      "  --min-samples <n>        extend the window below this many (default 15)\n"
      "  --max-attempts <n>       attempts before giving up (default 5)\n"
      "  --no-parity              skip the output comparison\n"
      "  --note <text>            free text recorded in the report\n"
      "  --quiet, -q              only print the summary\n"
      "\n"
      "  --counters <list>        PMU events around each timed pass: a\n"
      "                           comma-separated list of event names, or\n"
      "                           'default' for a sensible set for this machine\n"
      "  --list-counters          print this machine's PMU events and exit\n");
  };

  // --- Assemble the engine table -------------------------------------------

  EngineApi upstreamApi;
  upstreamApi.name = "a2_fast";
  upstreamApi.repository = "sdatkinson/NeuralAmpModelerCore";
  upstreamApi.codePath = "a2_fast";
  NB_FILL_BASE(upstreamApi, nb_upstream);

  EngineApi planarApi;
  planarApi.name = "a2_planar";
  planarApi.repository = "rikkus/OptimisationWorkOnNeuralAmpModelerCore@apple-silicon-a2-planar";
  planarApi.codePath = "a2_planar (Core PR #313)";
  NB_FILL_BASE(planarApi, nb_planar);

#if defined(NAMBENCH_HAVE_LABS)
  EngineApi slimApi;
  slimApi.name = "slim";
  slimApi.repository = "this repository";
  slimApi.codePath = "Sources/SlimEngines";
  NB_FILL_BASE(slimApi, nb_slim);
  NB_FILL_LAB(slimApi, nb_slim);

  EngineApi fullApi;
  fullApi.name = "full";
  fullApi.repository = "this repository";
  fullApi.codePath = "Sources/FullEngines";
  NB_FILL_BASE(fullApi, nb_full);
  NB_FILL_LAB(fullApi, nb_full);
#endif

#if defined(NAMBENCH_HAVE_A32_LAB)
  EngineApi a32Api;
  a32Api.name = "a32";
  a32Api.repository = "this repository";
  a32Api.codePath = "Sources/A32Engines";
  NB_FILL_BASE(a32Api, nb_a32);
  NB_FILL_LAB(a32Api, nb_a32);
#endif

  auto list_kernels = [](const EngineApi& api) {
    if (api.kernel_count == nullptr)
      return;
    const int count = api.kernel_count();
    std::printf("%s lab kernels (%d):\n", api.name, count);
    for (int i = 0; i < count; i++)
    {
      // Channel count and the exactness claim are printed because in a lab that
      // carries both submodels they are the two things that decide whether a
      // given kernel is applicable to this run and what it will be held to.
      std::printf("  %2d  %-24s %dch%s\n", i, api.kernel_name(i), api.kernel_channels(i),
                  api.kernel_exact(i) ? "" : "  (not bit-exact)");
    }
  };

  // --- Parse ----------------------------------------------------------------

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];
    const bool has = (i + 1 < argc);
    if ((arg == "--model" || arg == "-m") && has)
      modelPath = argv[++i];
    else if ((arg == "--audio" || arg == "-a") && has)
      audioPath = argv[++i];
    else if ((arg == "--output" || arg == "-o") && has)
      outputPath = argv[++i];
    else if (arg == "--submodel" && has)
    {
      const std::string value = argv[++i];
      config.submodel = (value == "narrowest" || value == "slim" || value == "nano")
                          ? NbSubmodelNarrowest
                          : NbSubmodelWidest;
    }
    else if (arg == "--slim" && has)
      slimSpec = argv[++i];
    else if (arg == "--full" && has)
      fullSpec = argv[++i];
    else if (arg == "--a32" && has)
      a32Spec = argv[++i];
    else if (arg == "--block-size" && has)
      config.blockSize = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    else if (arg == "--warmup-seconds" && has)
      config.warmupSeconds = std::strtod(argv[++i], nullptr);
    else if (arg == "--timing-seconds" && has)
      config.timingWindowSeconds = std::strtod(argv[++i], nullptr);
    else if (arg == "--accept-fraction" && has)
      config.acceptFraction = std::strtod(argv[++i], nullptr);
    else if (arg == "--accept-tolerance" && has)
      config.acceptTolerance = std::strtod(argv[++i], nullptr);
    else if (arg == "--min-samples" && has)
      config.minSamples = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    else if (arg == "--max-attempts" && has)
      config.maxAttempts = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    else if (arg == "--note" && has)
      testbedNote = argv[++i];
    else if (arg == "--no-parity")
      config.checkParity = false;
    else if (arg == "--quiet" || arg == "-q")
      config.quiet = true;
    else if (arg == "--counters" && has)
    {
      const std::string value = argv[++i];
      config.counterEvents.clear();
      if (value == "default")
        config.counterEvents = nb::Counters::default_events();
      else if (value != "none")
      {
        size_t pos = 0;
        while (pos <= value.size())
        {
          const size_t comma = value.find(',', pos);
          const std::string name = value.substr(pos, comma - pos);
          if (!name.empty())
            config.counterEvents.push_back(name);
          if (comma == std::string::npos)
            break;
          pos = comma + 1;
        }
      }
    }
    else if (arg == "--list-counters")
    {
      const std::vector<nb::CounterEvent> events = nb::Counters::available_events();
      if (events.empty())
      {
        std::printf("no PMU events available on this machine\n");
        return 0;
      }
      const int budget = nb::Counters::counter_budget();
      std::printf("PMU events (%zu):\n", events.size());
      for (const nb::CounterEvent& event : events)
        std::printf("  %-28s config=0x%llx\n", event.name.c_str(),
                    static_cast<unsigned long long>(event.config));
      // The budget is the number that decides whether a chosen set is measured
      // or extrapolated, and it is the one number no other tool will tell you.
      if (budget > 0)
        std::printf("\n%d can be counted at once; more than that and the kernel multiplexes,\n"
                    "which makes every value an estimate rather than a count.\n",
                    budget);
      const std::vector<std::string> chosen = nb::Counters::default_events();
      if (!chosen.empty())
      {
        std::printf("\n--counters default would select:\n ");
        for (const std::string& name : chosen)
          std::printf(" %s", name.c_str());
        std::printf("\n");
      }
      return 0;
    }
    else if (arg == "--list-slim")
    {
#if defined(NAMBENCH_HAVE_LABS)
      list_kernels(slimApi);
#else
      std::printf("this build has no kernel labs\n");
#endif
      return 0;
    }
    else if (arg == "--list-full")
    {
#if defined(NAMBENCH_HAVE_LABS)
      list_kernels(fullApi);
#else
      std::printf("this build has no kernel labs\n");
#endif
      return 0;
    }
    else if (arg == "--list-a32")
    {
#if defined(NAMBENCH_HAVE_A32_LAB)
      list_kernels(a32Api);
#else
      std::printf("this build has no a32 lab\n");
#endif
      return 0;
    }
    else if (arg == "--help" || arg == "-h")
    {
      usage();
      return 0;
    }
    else
    {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", arg.c_str());
      return 2;
    }
  }

  if (modelPath.empty() || audioPath.empty())
  {
    std::fprintf(stderr, "error: --model and --audio are both required\n\n");
    usage();
    return 2;
  }

  // --- Load ------------------------------------------------------------------

  const std::vector<uint8_t> nam = read_file(modelPath);
  if (nam.empty())
  {
    std::fprintf(stderr, "error: could not read %s\n", modelPath.c_str());
    return 1;
  }

  Audio audio;
  std::string audioError;
  if (!load_wav(audioPath, audio, audioError))
  {
    std::fprintf(stderr, "error: %s\n", audioError.c_str());
    return 1;
  }

  char err[512] = {0};
  NbProbe probe{};
  if (upstreamApi.probe(nam.data(), nam.size(), config.submodel, 0, &probe, err, sizeof(err)) != 0)
  {
    std::fprintf(stderr, "error: probe failed: %s\n", err);
    return 1;
  }

  if (probe.sampleRate > 0.0 && audio.sampleRate > 0.0 && probe.sampleRate != audio.sampleRate)
  {
    std::fprintf(stderr, "error: model expects %.0f Hz but the audio is %.0f Hz\n", probe.sampleRate,
                 audio.sampleRate);
    return 1;
  }

  const Environment environment = capture_environment();
  const std::string governor = cpu_governor();
  const double temperatureStart = soc_temperature_c();

  if (!config.quiet)
  {
    std::printf("%s — %s, %d cores, %s\n", environment.deviceModel.c_str(),
                environment.cpu.c_str(), environment.totalCores, environment.osVersion.c_str());
    std::printf("%zu frames, %.0f Hz, %.2f s\n", audio.samples.size(), audio.sampleRate,
                audio.durationSeconds());
    if (temperatureStart >= 0.0)
      std::printf("SoC %.1f C at start\n", temperatureStart);
  }

  // A governor that ramps makes a fast engine look slower than it is, because it
  // spends proportionally more of its pass at a low clock. Loud, and not fatal:
  // the run may be a deliberate look at what the machine does as configured.
  if (!governor.empty() && governor != "performance")
  {
    std::fprintf(stderr,
                 "warning: CPU governor is '%s', not 'performance'. Timings from this run are "
                 "not comparable with any other. Fix with:\n"
                 "  sudo cpupower frequency-set -g performance\n",
                 governor.c_str());
  }

  // --- Assemble the line-up --------------------------------------------------
  //
  // Exactly the rule main.swift applies, and for the same reason: read the shape
  // from the file, never infer it from the flag that was passed.

  std::vector<Subject> subjects;
  subjects.push_back({"a2_fast", &upstreamApi, -1});

  // The planar kernels cover both A2 submodels — 3 channels and 8 — so there
  // is no shape gate here. What there is instead is a check that they are
  // actually present: off AArch64 the checkout compiles to plain
  // a2_fast, and measuring that against upstream would produce two identical
  // numbers and the false impression that the kernels achieve nothing.
  //
  // The second condition mirrors a2_planar.h's own gate, and has to be kept in
  // step with it: it opens on AArch64, and on 32-bit ARM with NEON and FMA.
  // Where it cannot open — x86, MSVC/ARM64, ARMv7 built without neon-vfpv4 —
  // the planar checkout compiles to plain a2_fast, and lining it up would
  // measure the reference against itself. That case is caught below with a hard
  // error, but the error exists for a *regression* on a target where the gate
  // should have opened. On a target where it was never going to, the right
  // answer is to say so and carry on with the rest of the line-up, so that e.g.
  // an a32 lab sweep on a Cortex-A17 can still produce a report.
  //
  // Mirroring a gate is a duplicate by construction, and this one has already
  // drifted once: it stayed at __aarch64__ for a release after the header
  // widened, and the effect was not an error but a silently shorter line-up.
  // The runtime check below is what makes the duplicate safe — if these flags
  // and the ones the planar library was built with ever disagree, the run
  // stops rather than reporting a2_fast twice.
  constexpr bool kPlanarBuildable =
#if defined(__aarch64__) || (defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA))
    true;
#else
    false;
#endif

  if (!kPlanarBuildable)
  {
    if (!config.quiet)
      std::printf("planar excluded: a2_planar.h enables the kernels on AArch64 and on 32-bit ARM "
                  "with FMA, so this build is plain a2_fast\n");
  }
  else if (probe.channels == 3 || probe.channels == 8)
  {
    subjects.push_back({"a2_planar", &planarApi, -1});
  }
  else if (!config.quiet)
  {
    std::printf("planar excluded: the kernels cover 3 and 8 channels, this submodel has %d\n",
                probe.channels);
  }

#if defined(NAMBENCH_HAVE_LABS) || defined(NAMBENCH_HAVE_A32_LAB)
  // Which submodel a kernel serves is asked of the kernel, not of the lab. The
  // slim and full labs answer with their one channel count, so this behaves
  // exactly as the old lab-wide check did for them; the a32 lab carries kernels
  // for both submodels and answers per kernel, which is what lets one lab cover
  // both.
  auto add_lab = [&](const EngineApi& api, const std::string& spec) {
    if (spec.empty() || spec == "none")
      return;
    const int count = api.kernel_count();
    int servingThisSubmodel = 0;
    for (int i = 0; i < count; i++)
      if (api.kernel_channels(i) == probe.channels)
        servingThisSubmodel++;
    if (servingThisSubmodel == 0)
    {
      if (!config.quiet)
        std::printf("%s lab excluded: none of its kernels are written for %d channels\n", api.name,
                    probe.channels);
      return;
    }
    std::vector<int> chosen;
    if (spec == "all")
    {
      for (int i = 0; i < count; i++)
        chosen.push_back(i);
    }
    else
    {
      size_t begin = 0;
      while (begin <= spec.size())
      {
        const size_t comma = spec.find(',', begin);
        const std::string piece =
          trimmed(spec.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin));
        if (!piece.empty())
        {
          bool found = false;
          for (int i = 0; i < count; i++)
          {
            if (piece == api.kernel_name(i))
            {
              chosen.push_back(i);
              found = true;
              break;
            }
          }
          if (!found)
          {
            char* endptr = nullptr;
            const long index = std::strtol(piece.c_str(), &endptr, 10);
            if (endptr != piece.c_str() && index >= 0 && index < count)
              chosen.push_back(static_cast<int>(index));
            else
              std::fprintf(stderr, "warning: %s lab has no kernel called %s\n", api.name,
                           piece.c_str());
          }
        }
        if (comma == std::string::npos)
          break;
        begin = comma + 1;
      }
    }
    for (const int k : chosen)
    {
      // Silently skipping a kernel meant for the other submodel would be wrong —
      // "--a32 all" should measure everything applicable and say what it left
      // out — but naming one explicitly and getting nothing back would be worse.
      if (api.kernel_channels(k) != probe.channels)
      {
        if (spec != "all" && !config.quiet)
          std::printf("%s:%s skipped: written for %d channels, this submodel has %d\n", api.name,
                      api.kernel_name(k), api.kernel_channels(k), probe.channels);
        continue;
      }
      subjects.push_back({std::string(api.name) + ":" + api.kernel_name(k), &api, k});
    }
  };

  #if defined(NAMBENCH_HAVE_LABS)
  add_lab(slimApi, slimSpec);
  add_lab(fullApi, fullSpec);
  #endif
  #if defined(NAMBENCH_HAVE_A32_LAB)
  add_lab(a32Api, a32Spec);
  #endif
#endif // NAMBENCH_HAVE_LABS || NAMBENCH_HAVE_A32_LAB

#if !defined(NAMBENCH_HAVE_LABS)
  if (!slimSpec.empty() || !fullSpec.empty())
    std::fprintf(stderr, "warning: this build has no kernel labs; --slim/--full ignored\n");
#endif
#if !defined(NAMBENCH_HAVE_A32_LAB)
  if (!a32Spec.empty())
    std::fprintf(stderr, "warning: this build has no a32 lab; --a32 ignored\n");
#endif

  // --- Build every model up front -------------------------------------------
  //
  // All of them, before anything is measured, so a routing failure aborts the
  // run rather than surfacing halfway through as a mysterious regression.

  std::vector<NbModel*> models(subjects.size(), nullptr);
  for (size_t i = 0; i < subjects.size(); i++)
  {
    const Subject& subject = subjects[i];
    if (subject.kernel >= 0 && subject.api->select_kernel != nullptr
        && subject.api->select_kernel(subject.kernel) != 0)
    {
      std::fprintf(stderr, "error: %s: could not select kernel %d\n", subject.name.c_str(),
                   subject.kernel);
      return 1;
    }

    models[i] = subject.api->create(nam.data(), nam.size(), config.submodel, 0, config.blockSize,
                                    err, sizeof(err));
    if (models[i] == nullptr)
    {
      std::fprintf(stderr, "error: %s: create failed: %s\n", subject.name.c_str(), err);
      return 1;
    }

    if (!config.quiet)
      std::printf("%s: %s, %d channels\n", subject.name.c_str(),
                  engine_name(subject.api->engine(models[i])), subject.api->channels(models[i]));

    // The one failure this driver cannot let pass quietly. a2_planar.h defines
    // NAM_A2_PLANAR only on AArch64 and on 32-bit ARM with NEON and FMA, so
    // anywhere else the planar checkout builds to plain a2_fast. It would then
    // measure the same code as `upstream`, land within noise of it, and read as
    // "the kernels are worth nothing" rather than "the kernels are not in this
    // build".
    if (subject.name == "a2_planar" && subject.api->engine(models[i]) != NbEnginePlanar)
    {
      std::fprintf(stderr,
                   "\nerror: the planar variant routed to %s, not to the planar kernels.\n"
                   "  a2_planar.h enables them on AArch64, and on 32-bit ARM with NEON and\n"
                   "  FMA, so this build is measuring a2_fast twice. Refusing to report that\n"
                   "  as a comparison.\n"
                   "\n"
                   "  Expected on x86, and on MSVC/ARM64, which spells the architecture\n"
                   "  _M_ARM64 and never defines __aarch64__. Unexpected anywhere else. On\n"
                   "  AArch64 it means the gate has regressed; on ARMv7 the likeliest cause\n"
                   "  is -mfpu=neon rather than -mfpu=neon-vfpv4, which leaves\n"
                   "  __ARM_FEATURE_FMA undefined.\n",
                   engine_name(subject.api->engine(models[i])));
      return 1;
    }
  }

  // --- Parity, before timing anything ---------------------------------------

  std::vector<Parity> parities;
  std::vector<double> checksums(subjects.size(), 0.0);

  if (config.checkParity && subjects.size() >= 2)
  {
    std::vector<double> reference;
    for (size_t i = 0; i < subjects.size(); i++)
    {
      std::vector<double> output(audio.samples.size(), 0.0);
      double checksum = 0.0;
      subjects[i].api->reset(models[i], audio.sampleRate, config.blockSize);
      subjects[i].api->process(models[i], audio.samples.data(), audio.samples.size(), output.data(),
                               &checksum);
      checksums[i] = checksum;

      if (i == 0)
      {
        reference = std::move(output);
        continue;
      }

      const Parity p = compare_outputs(reference, output, subjects[0].name, subjects[i].name,
                                       config.parityWarnRelativeRMS);
      parities.push_back(p);
      if (!config.quiet)
      {
        // Rounded, not truncated, and to the same one decimal place BenchCore
        // uses — otherwise the same measurement reads as 132.5 here and 132.6
        // there, and someone spends an afternoon on it.
        char decibels[64];
        if (std::isinf(p.decibelsBelowSignal))
          std::snprintf(decibels, sizeof(decibels), "bit-identical");
        else
          std::snprintf(decibels, sizeof(decibels), "%.1f dB below signal", p.decibelsBelowSignal);
        std::printf("parity %s vs %s: max|diff| %.3e, rms %.3e (%s)%s\n", p.referenceVariant.c_str(),
                    p.comparisonVariant.c_str(), p.maxAbsoluteDifference, p.rmsDifference, decibels,
                    p.withinTolerance ? "" : "  ** OUT OF TOLERANCE **");
      }
    }
  }

  // --- Counters ---------------------------------------------------------------
  //
  // Opened once for the whole run rather than per subject, so every engine is
  // counted through the same file descriptors on the same events. A failure
  // here is reported and stepped over: an unreadable PMU is a reason to lose
  // the attribution, not the measurement.

  nb::Counters counters;
  if (!config.counterEvents.empty())
  {
    if (!counters.open(config.counterEvents))
    {
      std::fprintf(stderr, "warning: counters unavailable, continuing without them: %s\n",
                   counters.reason().c_str());
    }
    else if (!config.quiet)
    {
      const int budget = nb::Counters::counter_budget();
      std::printf("counters:");
      for (const std::string& name : counters.names())
        std::printf(" %s", name.c_str());
      std::printf("\n");
      if (budget > 0 && static_cast<int>(counters.names().size()) > budget)
        std::fprintf(stderr,
                     "warning: %zu events but only %d counters; the kernel will multiplex, so "
                     "every value will be scaled up from a sample of each pass rather than "
                     "counted. Ask for %d or fewer to get counts.\n",
                     counters.names().size(), budget, budget);
    }
  }

  // --- Measure ---------------------------------------------------------------

  std::vector<Result> results;
  for (size_t i = 0; i < subjects.size(); i++)
  {
    const Subject& subject = subjects[i];
    NbModel* model = models[i];

    PassHooks hooks;
    hooks.reset = [&] { subject.api->reset(model, audio.sampleRate, config.blockSize); };
    hooks.run = [&] {
      return subject.api->process(model, audio.samples.data(), audio.samples.size(), nullptr,
                                  nullptr);
    };

    Result result =
      measure(subject.name, config, audio.durationSeconds(), hooks, &counters);
    result.engine = engine_name(subject.api->engine(model));
    result.channels = subject.api->channels(model);
    result.checksum = checksums[i];

    if (!config.quiet)
      report_progress(result);

    results.push_back(std::move(result));
  }

  for (size_t i = 0; i < models.size(); i++)
    subjects[i].api->destroy(models[i]);

  const double temperatureEnd = soc_temperature_c();

  // --- Summary ---------------------------------------------------------------

  std::printf("\n==== Result ====\n");
  const double baselineMs = (!results.empty() && results[0].succeeded) ? results[0].meanMs : 0.0;
  for (const Result& result : results)
  {
    if (!result.succeeded)
    {
      std::printf("  %-24s REJECTED — %s\n", result.variant.c_str(), result.failureReason.c_str());
      continue;
    }
    // CPU percent first: it is the number that answers the question the
    // benchmark exists for. The milliseconds stay because they are the raw
    // measurement and a diagnosis needs them.
    std::printf("  %-24s %7.3f%% of one core  (%8.2f ms, min %.2f)", result.variant.c_str(),
                result.corePercent, result.meanMs, result.minMs);
    if (baselineMs > 0.0 && result.meanMs > 0.0 && &result != &results[0])
      std::printf("  %6.3fx vs %s", baselineMs / result.meanMs, results[0].variant.c_str());
    std::printf("\n");

    // Cycles and instructions get a derived line of their own because IPC is
    // what separates the two explanations for a slow kernel that this campaign
    // most needs to tell apart: one that issues too many instructions, and one
    // that issues the right number and stalls on them.
    if (!result.counters.empty())
    {
      double cycles = 0.0, instructions = 0.0;
      for (const nb::CounterSummary& counter : result.counters)
      {
        if (counter.name == "cpu_cycles" || counter.name == "cycles")
          cycles = counter.median;
        else if (counter.name == "inst_retired" || counter.name == "instructions")
          instructions = counter.median;
      }
      if (cycles > 0.0 && instructions > 0.0)
        std::printf("  %-24s %.0f cycles, %.0f instructions, IPC %.3f\n", "",
                    cycles, instructions, instructions / cycles);
      for (const nb::CounterSummary& counter : result.counters)
      {
        if (counter.name == "cpu_cycles" || counter.name == "cycles"
            || counter.name == "inst_retired" || counter.name == "instructions")
          continue;
        std::printf("  %-24s %-20s %14.0f", "", counter.name.c_str(), counter.median);
        if (cycles > 0.0)
          std::printf("  (%.2f per 1k cycles)", counter.median * 1000.0 / cycles);
        if (counter.alwaysZero)
          std::printf("  [not implemented on this core]");
        else if (counter.worstScaling < 0.999)
          std::printf("  [estimated: counted for %.0f%% of the pass]",
                      counter.worstScaling * 100.0);
        std::printf("\n");
      }
    }
  }
  if (temperatureStart >= 0.0 && temperatureEnd >= 0.0)
    std::printf("\n  SoC %.1f C -> %.1f C\n", temperatureStart, temperatureEnd);

  // --- Report ----------------------------------------------------------------
  //
  // Key names match BenchCore's report where the two overlap, so
  // Scripts/bencher-report.py reads either without knowing which produced it.

  std::string json;
  json.reserve(1 << 16);
  char buffer[1024];

  // Every format string below is a literal at the call site; the compiler just
  // cannot see that through the lambda.
#if defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wformat-security"
#endif
  auto append = [&](const char* format, auto... args) {
    std::snprintf(buffer, sizeof(buffer), format, args...);
    json += buffer;
  };
  auto append_array = [&](const std::vector<double>& values) {
    json += "[";
    for (size_t i = 0; i < values.size(); i++)
    {
      std::snprintf(buffer, sizeof(buffer), "%s%.6f", i ? "," : "", values[i]);
      json += buffer;
    }
    json += "]";
  };

  json += "{\n";
  append("  \"schemaVersion\": 1,\n  \"producer\": \"nam_benchmark\",\n");
  append("  \"environment\": {\"deviceModel\": \"%s\", \"cpu\": \"%s\", \"platform\": \"%s\", "
         "\"architecture\": \"%s\", \"fpu\": \"%s\", \"osVersion\": \"%s\", \"totalCores\": %d, "
         "\"cpuGovernor\": \"%s\"},\n",
         environment.deviceModel.c_str(), environment.cpu.c_str(), environment.platform.c_str(),
         environment.architecture.c_str(), environment.fpu.c_str(), environment.osVersion.c_str(),
         environment.totalCores, governor.c_str());
  append("  \"note\": \"%s\",\n", testbedNote.c_str());
  append("  \"socTemperatureStartC\": %.1f,\n  \"socTemperatureEndC\": %.1f,\n", temperatureStart,
         temperatureEnd);
  append("  \"config\": {\"blockSize\": %d, \"submodel\": \"%s\", \"warmupSeconds\": %.1f, "
         "\"timingWindowSeconds\": %.1f, \"acceptFraction\": %.4f, \"acceptTolerance\": %.4f, "
         "\"minSamples\": %d, \"maxAttempts\": %d},\n",
         config.blockSize, config.submodel == NbSubmodelNarrowest ? "narrowest" : "widest",
         config.warmupSeconds, config.timingWindowSeconds, config.acceptFraction,
         config.acceptTolerance, config.minSamples, config.maxAttempts);
  // Which events were counted, and how many the hardware could hold, belong
  // with the run rather than with each result: a report read months later has
  // to be able to say whether its counter values were counts or extrapolations.
  {
    json += "  \"counters\": {\"requested\": ";
    json += "[";
    for (size_t i = 0; i < config.counterEvents.size(); i++)
      append("%s\"%s\"", i ? "," : "", config.counterEvents[i].c_str());
    json += "], \"opened\": [";
    for (size_t i = 0; i < counters.names().size(); i++)
      append("%s\"%s\"", i ? "," : "", counters.names()[i].c_str());
    append("], \"counterBudget\": %d", config.counterEvents.empty()
                                        ? 0
                                        : nb::Counters::counter_budget());
    if (!config.counterEvents.empty() && !counters.is_open())
      append(", \"unavailableReason\": \"%s\"", counters.reason().c_str());
    json += "},\n";
  }
  append("  \"model\": {\"fileName\": \"%s\", \"channels\": %d, \"sampleRate\": %.1f},\n",
         fs::path(modelPath).filename().string().c_str(), probe.channels, probe.sampleRate);
  append("  \"audio\": {\"fileName\": \"%s\", \"frameCount\": %zu, \"sampleRate\": %.1f, "
         "\"durationSeconds\": %.6f},\n",
         fs::path(audioPath).filename().string().c_str(), audio.samples.size(), audio.sampleRate,
         audio.durationSeconds());

  json += "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); i++)
  {
    const Result& r = results[i];
    append("    {\"variant\": \"%s\", \"engine\": \"%s\", \"channels\": %d, \"succeeded\": %s, ",
           r.variant.c_str(), r.engine.c_str(), r.channels, r.succeeded ? "true" : "false");
    append("\"meanMs\": %.6f, \"medianMs\": %.6f, \"minMs\": %.6f, \"maxMs\": %.6f, "
           "\"spread\": %.6f, \"standardDeviationMs\": %.6f, \"realTimeFactor\": %.6f, "
           "\"corePercent\": %.6f, \"checksum\": %.10g, ",
           r.meanMs, r.medianMs, r.minMs, r.maxMs, r.spread, r.standardDeviationMs,
           r.realTimeFactor, r.corePercent, r.checksum);
    if (!r.failureReason.empty())
      append("\"failureReason\": \"%s\", ", r.failureReason.c_str());
    // A run that stopped on the deadline rather than on a full window is a
    // weaker measurement than one that did not, and on a slow device it is the
    // common case. Recording it makes that visible downstream instead of
    // indistinguishable from a clean run.
    {
      size_t sampleCount = 0;
      bool truncated = false;
      for (const Attempt& a : r.attempts)
      {
        sampleCount += a.samplesMs.size();
        truncated = truncated || a.hitHardDeadline;
      }
      append("\"sampleCount\": %zu, \"hitHardDeadline\": %s, ", sampleCount,
             truncated ? "true" : "false");
    }
    if (!r.counters.empty())
    {
      json += "\"counters\": {";
      for (size_t c = 0; c < r.counters.size(); c++)
      {
        const nb::CounterSummary& counter = r.counters[c];
        // Median per pass, with the range beside it, and the two caveats that
        // decide whether the median means anything: whether the event was
        // multiplexed, and whether it ever moved at all.
        append("%s\"%s\": {\"median\": %.0f, \"min\": %.0f, \"max\": %.0f, "
               "\"config\": %llu, \"worstScaling\": %.4f, \"alwaysZero\": %s}",
               c ? ", " : "", counter.name.c_str(), counter.median, counter.min, counter.max,
               static_cast<unsigned long long>(counter.config), counter.worstScaling,
               counter.alwaysZero ? "true" : "false");
      }
      json += "}, ";
    }
    json += "\"acceptedMs\": ";
    append_array(r.acceptedMs);
    json += ", \"discardedMs\": ";
    append_array(r.discardedMs);
    json += "}";
    json += (i + 1 < results.size()) ? ",\n" : "\n";
  }
  json += "  ],\n";

  json += "  \"parities\": [\n";
  for (size_t i = 0; i < parities.size(); i++)
  {
    const Parity& p = parities[i];
    append("    {\"referenceVariant\": \"%s\", \"comparisonVariant\": \"%s\", "
           "\"maxAbsoluteDifference\": %.6e, \"rmsDifference\": %.6e, \"relativeRMS\": %.6e, "
           "\"decibelsBelowSignal\": %s, \"withinTolerance\": %s}",
           p.referenceVariant.c_str(), p.comparisonVariant.c_str(), p.maxAbsoluteDifference,
           p.rmsDifference, p.relativeRMS,
           std::isinf(p.decibelsBelowSignal) ? "null"
                                             : std::to_string(p.decibelsBelowSignal).c_str(),
           p.withinTolerance ? "true" : "false");
    json += (i + 1 < parities.size()) ? ",\n" : "\n";
  }
  json += "  ]\n}\n";

  std::FILE* file = std::fopen(outputPath.string().c_str(), "wb");
  if (file == nullptr)
  {
    std::fprintf(stderr, "error: could not write %s\n", outputPath.string().c_str());
    return 1;
  }
  std::fwrite(json.data(), 1, json.size(), file);
  std::fclose(file);
#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif
  std::printf("\n  %s\n", outputPath.string().c_str());

  bool allSucceeded = true;
  for (const Result& r : results)
    allSucceeded = allSucceeded && r.succeeded;
  return allSucceeded ? 0 : 1;
}
