// Impulse-response benchmark driver.
//
// AudioDSPTools convolves an impulse response the obvious way: one dot product
// of every tap, per output sample. At the 8192 taps the plugin allows, that is
// a lot of arithmetic to do sample by sample. The partitioned-ir branch keeps a
// direct FIR head for the first block — which is what keeps latency at zero —
// and moves the tail into uniformly partitioned FFT convolution.
//
// This measures both, in one process, from two hidden-visibility shared
// libraries, using the same protocol as Tools/nam_benchmark.cpp: same warm-up,
// same timing window, same tightest-70% selection, same spread test, same
// retry-and-reject. Everything that decides what the numbers mean is in
// Tools/nb_protocol.h and is shared, so an IR number and a WaveNet number sit on
// the same Bencher project honestly.
//
// It reports two things per subject, because for this change a mean on its own
// would be misleading:
//
//   corePercent      — the average cost of keeping up with real time, which is
//                      what the mean pass time says.
//   blockP99Percent  — the 99th-percentile *block*, as a percentage of that
//                      block's real-time budget. Partitioned FFT convolution is
//                      bursty by construction: a whole partition's transform
//                      lands in one callback, so a good average can hide a block
//                      that misses its deadline, and a plugin that misses one
//                      clicks. Above 100% here means it did.
//
// Not built under MSVC, for the same reason nam_benchmark is not: the layout
// relies on -fvisibility=hidden plus a default-visibility attribute on the
// exports, which is not how Windows DLLs work.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nam_ir_shim.h"
#include "nb_counters.h"
#include "nb_protocol.h"

#if defined(__APPLE__)
  #include <pthread.h>
#endif

extern "C" {
NB_IR_DECLARE_VARIANT(nb_ir_upstream)
NB_IR_DECLARE_VARIANT(nb_ir_partitioned)
}

namespace
{

namespace fs = std::filesystem;

using namespace nbp;

// ---------------------------------------------------------------------------
// One variant's exported API, gathered behind function pointers, so the loops
// below cannot do something to one variant they do not do to the other.
// ---------------------------------------------------------------------------
struct IrApi
{
  const char* name = nullptr;
  const char* repository = nullptr;

  int (*can_select)() = nullptr;
  NbIr* (*create)(int32_t, int32_t, double, NbIrImpl, int32_t, char*, size_t) = nullptr;
  void (*destroy)(NbIr*) = nullptr;
  NbIrImpl (*implementation)(const NbIr*) = nullptr;
  int32_t (*taps)(const NbIr*) = nullptr;
  int32_t (*channels)(const NbIr*) = nullptr;
  int32_t (*partitions)(const NbIr*) = nullptr;
  int32_t (*fft_block)(const NbIr*) = nullptr;
  void (*reset)(NbIr*, int32_t) = nullptr;
  uint64_t (*process)(NbIr*, const double*, size_t, double*, double*, uint64_t*) = nullptr;
};

#define NB_IR_FILL(api, P)                                                                        \
  do                                                                                              \
  {                                                                                               \
    (api).can_select = &P##_ir_can_select;                                                        \
    (api).create = &P##_ir_create;                                                                \
    (api).destroy = &P##_ir_destroy;                                                              \
    (api).implementation = &P##_ir_implementation;                                                \
    (api).taps = &P##_ir_taps;                                                                    \
    (api).channels = &P##_ir_channels;                                                            \
    (api).partitions = &P##_ir_partitions;                                                        \
    (api).fft_block = &P##_ir_fft_block;                                                          \
    (api).reset = &P##_ir_reset;                                                                  \
    (api).process = &P##_ir_process;                                                              \
  } while (0)

/// One thing to measure: a variant, and which convolution it was asked for.
struct Subject
{
  std::string name;
  const IrApi* api = nullptr;
  NbIrImpl requested = NbIrImplAuto;
};

const char* impl_name(NbIrImpl impl)
{
  switch (impl)
  {
    case NbIrImplDirect: return "direct";
    case NbIrImplFFT: return "fft";
    case NbIrImplAuto: break;
  }
  return "auto";
}

/// One subject at one (taps, block) point, with everything the report needs.
struct IrResult
{
  Result timing;
  int32_t taps = 0;
  int32_t requestedTaps = 0;
  int32_t blockSize = 0;
  int32_t irChannels = 0;
  std::string requested;
  std::string active;
  /// How the FFT path was configured. Zero partitions on an "fft" subject
  /// means the IR fit inside the direct head and no transform ran — the
  /// subject is a plain FIR, and saying "fft" without this would be reporting
  /// one thing as another.
  int32_t partitions = 0;
  int32_t fftBlock = 0;
  /// Per-block cost as a percentage of that block's real-time budget, pooled
  /// over every pass the window accepted.
  double blockMedianPercent = 0.0;
  double blockP99Percent = 0.0;
  double blockMaxPercent = 0.0;
  size_t blockSampleCount = 0;
};

/// The value at `quantile` of an already-sorted vector, by linear interpolation.
double quantile_of_sorted(const std::vector<double>& sorted, double quantile)
{
  if (sorted.empty())
    return 0.0;
  if (sorted.size() == 1)
    return sorted[0];
  const double position = quantile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

/// Parse "64,128,512" into a list, rejecting anything that is not a positive
/// integer rather than silently contributing a zero.
bool parse_int_list(const std::string& spec, std::vector<int32_t>& out, std::string& error)
{
  size_t start = 0;
  while (start <= spec.size())
  {
    const size_t comma = spec.find(',', start);
    const std::string piece =
      spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (piece.empty())
    {
      error = "empty entry in \"" + spec + "\"";
      return false;
    }
    try
    {
      const int value = std::stoi(piece);
      if (value <= 0)
      {
        error = "\"" + piece + "\" is not a positive integer";
        return false;
      }
      out.push_back(static_cast<int32_t>(value));
    }
    catch (const std::exception&)
    {
      error = "\"" + piece + "\" is not a number";
      return false;
    }
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // Same reason as nam_benchmark: on Apple silicon a lower-QoS thread is
  // scheduled onto the efficiency cores, and an E-core measurement would swamp
  // every effect this is trying to resolve.
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  ProtocolConfig config;
  std::string audioPath;
  fs::path outputPath = "ir-benchmark.json";
  std::string testbedNote;
  std::vector<int32_t> tapsList;
  std::vector<int32_t> blockList;
  int32_t irChannels = 1;
  bool checkParity = true;
  double parityWarnRelativeRMS = 1e-3;
  std::vector<std::string> counterEvents;

  auto usage = []() {
    std::printf(
      "nam_ir_benchmark — impulse-response convolution, NAMBench protocol\n"
      "\n"
      "  --audio, -a <path>       input .wav (required)\n"
      "  --output, -o <path>      JSON report to write (default: ir-benchmark.json)\n"
      "\n"
      "  --taps <list>            IR lengths to measure (default 256,512,1024,2048,4096,8192)\n"
      "  --blocks <list>          block sizes to measure (default 64)\n"
      "  --ir-channels <1|2>      mono or stereo IR (default 1)\n"
      "\n"
      "  --warmup-seconds <s>     discarded warm-up (default 5)\n"
      "  --timing-seconds <s>     timing window (default 30)\n"
      "  --accept-fraction <r>    fraction of samples kept (default 0.70)\n"
      "  --accept-tolerance <r>   required agreement (default 0.03)\n"
      "  --min-samples <n>        extend the window below this many (default 15)\n"
      "  --max-attempts <n>       attempts before giving up (default 5)\n"
      "  --no-parity              skip the output comparison\n"
      "  --parity-tolerance <r>   relative RMS a variant may differ by (default 1e-3)\n"
      "  --note <text>            free text recorded in the report\n"
      "  --quiet, -q              only print the summary\n"
      "\n"
      "  --counters <list>        PMU events around each timed pass, or 'default'\n"
      "  --list-counters          print this machine's PMU events and exit\n");
  };

  // --- Assemble the variant table -------------------------------------------

  IrApi upstreamApi;
  upstreamApi.name = "adt_upstream";
  upstreamApi.repository = "sdatkinson/AudioDSPTools";
  NB_IR_FILL(upstreamApi, nb_ir_upstream);

  IrApi partitionedApi;
  partitionedApi.name = "adt_partitioned";
  partitionedApi.repository = "rikkus/AudioDSPTools@partitioned-ir";
  NB_IR_FILL(partitionedApi, nb_ir_partitioned);

  // --- Parse ----------------------------------------------------------------

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];
    const bool has = (i + 1 < argc);
    std::string error;
    if ((arg == "--audio" || arg == "-a") && has)
      audioPath = argv[++i];
    else if ((arg == "--output" || arg == "-o") && has)
      outputPath = argv[++i];
    else if (arg == "--taps" && has)
    {
      if (!parse_int_list(argv[++i], tapsList, error))
      {
        std::fprintf(stderr, "error: --taps: %s\n", error.c_str());
        return 2;
      }
    }
    else if (arg == "--blocks" && has)
    {
      if (!parse_int_list(argv[++i], blockList, error))
      {
        std::fprintf(stderr, "error: --blocks: %s\n", error.c_str());
        return 2;
      }
    }
    else if (arg == "--ir-channels" && has)
    {
      irChannels = static_cast<int32_t>(std::atoi(argv[++i]));
      if (irChannels != 1 && irChannels != 2)
      {
        std::fprintf(stderr, "error: --ir-channels must be 1 or 2\n");
        return 2;
      }
    }
    else if (arg == "--warmup-seconds" && has)
      config.warmupSeconds = std::atof(argv[++i]);
    else if (arg == "--timing-seconds" && has)
      config.timingWindowSeconds = std::atof(argv[++i]);
    else if (arg == "--accept-fraction" && has)
      config.acceptFraction = std::atof(argv[++i]);
    else if (arg == "--accept-tolerance" && has)
      config.acceptTolerance = std::atof(argv[++i]);
    else if (arg == "--min-samples" && has)
      config.minSamples = std::atoi(argv[++i]);
    else if (arg == "--max-attempts" && has)
      config.maxAttempts = std::atoi(argv[++i]);
    else if (arg == "--no-parity")
      checkParity = false;
    else if (arg == "--parity-tolerance" && has)
      parityWarnRelativeRMS = std::atof(argv[++i]);
    else if (arg == "--note" && has)
      testbedNote = argv[++i];
    else if (arg == "--quiet" || arg == "-q")
      config.quiet = true;
    else if (arg == "--counters" && has)
    {
      const std::string value = argv[++i];
      if (value == "default")
        counterEvents = nb::Counters::default_events();
      else
      {
        size_t start = 0;
        while (start <= value.size())
        {
          const size_t comma = value.find(',', start);
          const std::string piece =
            value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          if (!piece.empty())
            counterEvents.push_back(piece);
          if (comma == std::string::npos)
            break;
          start = comma + 1;
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
      if (budget > 0)
        std::printf("\n%d can be counted at once; more than that and the kernel multiplexes,\n"
                    "which makes every value an estimate rather than a count.\n",
                    budget);
      return 0;
    }
    else if (arg == "--help" || arg == "-h")
    {
      usage();
      return 0;
    }
    else
    {
      std::fprintf(stderr, "error: unrecognised argument %s\n\n", arg.c_str());
      usage();
      return 2;
    }
  }

  if (audioPath.empty())
  {
    usage();
    return 2;
  }
  if (tapsList.empty())
    tapsList = {256, 512, 1024, 2048, 4096, 8192};
  if (blockList.empty())
    blockList = {64};

  // --- Load ------------------------------------------------------------------

  Audio audio;
  std::string error;
  if (!load_wav(audioPath, audio, error))
  {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }

  const Environment environment = capture_environment();
  const std::string governor = cpu_governor();
  if (!governor.empty() && governor != "performance")
    std::fprintf(stderr,
                 "warning: CPU governor is '%s'. Anything but 'performance' spends the start of\n"
                 "  every pass ramping, which measures the scheduler rather than the code.\n",
                 governor.c_str());

  if (!config.quiet)
  {
    std::printf("%s, %s, %s (%s), %d cores\n", environment.deviceModel.c_str(),
                environment.cpu.c_str(), environment.platform.c_str(),
                environment.architecture.c_str(), environment.totalCores);
    std::printf("audio: %s, %.1f s at %.0f Hz\n", fs::path(audioPath).filename().string().c_str(),
                audio.durationSeconds(), audio.sampleRate);
  }

  nb::Counters counters;
  if (!counterEvents.empty() && !counters.open(counterEvents))
    std::fprintf(stderr, "warning: counters unavailable: %s\n", counters.reason().c_str());

  // --- The line-up -----------------------------------------------------------
  //
  // Three subjects per point, because two different questions are being asked.
  // Against upstream, the partitioned branch has to be faster without being
  // worse in its worst block — that is the proposal. Its own direct path is
  // measured as well because it is a rewrite of upstream's, not the same code:
  // if it has regressed, the Auto threshold below which it is chosen is
  // carrying that regression, and no comparison of the FFT path would show it.
  std::vector<Subject> subjects = {
    {"adt_upstream", &upstreamApi, NbIrImplAuto},
    {"adt_partitioned:direct", &partitionedApi, NbIrImplDirect},
    {"adt_partitioned:fft", &partitionedApi, NbIrImplFFT},
  };

  std::vector<IrResult> results;
  std::vector<Parity> parities;
  const double temperatureStart = soc_temperature_c();

  for (const int32_t taps : tapsList)
  {
    for (const int32_t blockSize : blockList)
    {
      if (!config.quiet)
        std::printf("\n---- %d taps, %d-frame blocks ----\n", taps, blockSize);

      // Parity first, before anything is timed, and against the shipping
      // implementation. The FFT path is not expected to be bit-identical —
      // it is a different order of arithmetic — so this is an accuracy floor,
      // not an equality check, and a failure here invalidates every timing at
      // this point rather than being reported next to it.
      std::vector<std::vector<double>> parityOutputs(subjects.size());
      bool parityOk = true;
      if (checkParity)
      {
        for (size_t s = 0; s < subjects.size(); s++)
        {
          char err[256] = {0};
          NbIr* ir = subjects[s].api->create(taps, irChannels, audio.sampleRate,
                                             subjects[s].requested, blockSize, err, sizeof(err));
          if (ir == nullptr)
          {
            std::fprintf(stderr, "error: %s could not be built at %d taps: %s\n",
                         subjects[s].name.c_str(), taps, err);
            return 1;
          }
          parityOutputs[s].assign(audio.samples.size(), 0.0);
          double checksum = 0.0;
          subjects[s].api->process(ir, audio.samples.data(), audio.samples.size(),
                                   parityOutputs[s].data(), &checksum, nullptr);
          subjects[s].api->destroy(ir);
        }

        for (size_t s = 1; s < subjects.size(); s++)
        {
          Parity parity = compare_outputs(parityOutputs[0], parityOutputs[s], subjects[0].name,
                                          subjects[s].name, parityWarnRelativeRMS);
          parity.referenceVariant += " @" + std::to_string(taps);
          parity.comparisonVariant += " @" + std::to_string(taps);
          parities.push_back(parity);
          if (!config.quiet)
          {
            std::printf("parity %s vs %s: %.1f dB below signal, max|diff| %.2e\n",
                        subjects[0].name.c_str(), subjects[s].name.c_str(),
                        parity.decibelsBelowSignal, parity.maxAbsoluteDifference);
          }
          if (!parity.withinTolerance)
          {
            std::fprintf(stderr,
                         "error: %s differs from %s by %.2e relative RMS at %d taps, more than the\n"
                         "  %.0e allowed. Timings for this point are not reported: a faster\n"
                         "  convolution that computes something else is not a faster convolution.\n",
                         subjects[s].name.c_str(), subjects[0].name.c_str(), parity.relativeRMS,
                         taps, parityWarnRelativeRMS);
            parityOk = false;
          }
        }
      }
      if (!parityOk)
        continue;

      for (const Subject& subject : subjects)
      {
        char err[256] = {0};
        NbIr* ir =
          subject.api->create(taps, irChannels, audio.sampleRate, subject.requested, blockSize, err,
                              sizeof(err));
        if (ir == nullptr)
        {
          std::fprintf(stderr, "error: %s could not be built at %d taps: %s\n",
                       subject.name.c_str(), taps, err);
          return 1;
        }

        const size_t blocks =
          (audio.samples.size() + static_cast<size_t>(blockSize) - 1) / static_cast<size_t>(blockSize);

        // Per-pass block times, kept so that the window's selection can be
        // applied to them as well. Reserved up front: allocating inside the
        // timing loop would be measured.
        std::vector<std::vector<uint64_t>> passBlocks;
        std::vector<uint64_t> scratch(blocks, 0);

        PassHooks hooks;
        hooks.reset = [&] { subject.api->reset(ir, blockSize); };
        hooks.run = [&] {
          return subject.api->process(ir, audio.samples.data(), audio.samples.size(), nullptr,
                                      nullptr, scratch.data());
        };
        // measure() retries, and each attempt's pass indices restart at 0.
        // Starting over at pass 0 keeps passBlocks aligned with the attempt
        // whose acceptedIndices are reported, not with a rejected one before it.
        hooks.onTimedPass = [&](size_t passIndex) {
          if (passIndex == 0)
            passBlocks.clear();
          passBlocks.push_back(scratch);
        };

        IrResult record;
        record.timing = measure(subject.name, config, audio.durationSeconds(), hooks, &counters);
        record.requestedTaps = taps;
        record.taps = subject.api->taps(ir);
        record.blockSize = blockSize;
        record.irChannels = subject.api->channels(ir);
        record.requested = impl_name(subject.requested);
        record.active = impl_name(subject.api->implementation(ir));
        record.partitions = subject.api->partitions(ir);
        record.fftBlock = subject.api->fft_block(ir);

        // Pool the accepted passes' blocks. The budget a block has is exactly
        // how long its own audio lasts, so a percentage here is directly
        // readable: 100% is a callback that used its entire deadline.
        {
          std::vector<double> percents;
          for (const size_t index : record.timing.acceptedIndices)
          {
            if (index >= passBlocks.size())
              continue;
            const std::vector<uint64_t>& one = passBlocks[index];
            percents.reserve(percents.size() + one.size());
            for (size_t b = 0; b < one.size(); b++)
            {
              // The last block is short whenever the file does not divide by
              // the block size, so its budget is its own length, not a full
              // block's. Left in rather than dropped: it is a real callback.
              const size_t frames =
                std::min(static_cast<size_t>(blockSize),
                         audio.samples.size() - b * static_cast<size_t>(blockSize));
              const double budgetNs = static_cast<double>(frames) / audio.sampleRate * 1.0e9;
              if (budgetNs > 0.0)
                percents.push_back(static_cast<double>(one[b]) / budgetNs * 100.0);
            }
          }
          std::sort(percents.begin(), percents.end());
          record.blockSampleCount = percents.size();
          record.blockMedianPercent = quantile_of_sorted(percents, 0.50);
          record.blockP99Percent = quantile_of_sorted(percents, 0.99);
          record.blockMaxPercent = percents.empty() ? 0.0 : percents.back();
        }

        if (!config.quiet)
        {
          report_progress(record.timing);
          // Only where there is a window to have pooled blocks from. A rejected
          // attempt has no accepted passes, and printing its zeroes would read
          // as a convolution that took no time at all.
          if (record.timing.succeeded)
          {
            char shape[96] = {0};
            if (record.fftBlock > 0)
              std::snprintf(shape, sizeof(shape), " (%d-point blocks, %d partition%s)",
                            record.fftBlock, record.partitions,
                            record.partitions == 1 ? "" : "s");
            std::printf("    %s%s, %d taps: blocks median %.2f%% / p99 %.2f%% / max %.2f%% of "
                        "deadline\n",
                        record.active.c_str(), shape, record.taps, record.blockMedianPercent,
                        record.blockP99Percent, record.blockMaxPercent);
            // The case this exists to catch: an FFT subject that ran no
            // transform at all, because the whole impulse response fit in the
            // direct head. Its number is real, but it is the direct head's
            // number and comparing it with upstream says nothing about FFT.
            if (record.active == "fft" && record.partitions == 0)
              std::printf("      note: the IR fits in the direct head, so no transform runs "
                          "here — this row measures a plain FIR\n");
          }
        }

        subject.api->destroy(ir);
        results.push_back(std::move(record));
      }
    }
  }

  const double temperatureEnd = soc_temperature_c();

  // --- Summary ---------------------------------------------------------------

  std::printf("\n==== Result ====\n");
  std::printf("  %-8s %-6s %-24s %10s %10s %10s\n", "taps", "block", "variant", "core%", "p99 blk%",
              "vs base");
  for (size_t i = 0; i < results.size(); i++)
  {
    const IrResult& r = results[i];
    if (!r.timing.succeeded)
    {
      std::printf("  %-8d %-6d %-24s REJECTED — %s\n", r.taps, r.blockSize,
                  r.timing.variant.c_str(), r.timing.failureReason.c_str());
      continue;
    }
    // The baseline for a point is upstream at that same point, which is the
    // first subject measured there.
    double baselineMs = 0.0;
    for (const IrResult& other : results)
    {
      if (other.taps == r.taps && other.blockSize == r.blockSize
          && other.timing.variant == "adt_upstream" && other.timing.succeeded)
      {
        baselineMs = other.timing.meanMs;
        break;
      }
    }
    std::printf("  %-8d %-6d %-24s %9.3f%% %9.2f%%", r.taps, r.blockSize, r.timing.variant.c_str(),
                r.timing.corePercent, r.blockP99Percent);
    if (baselineMs > 0.0 && r.timing.meanMs > 0.0 && r.timing.variant != "adt_upstream")
      std::printf("   %6.2fx", baselineMs / r.timing.meanMs);
    std::printf("\n");
  }
  if (temperatureStart >= 0.0 && temperatureEnd >= 0.0)
    std::printf("\n  SoC %.1f C -> %.1f C\n", temperatureStart, temperatureEnd);

  // --- Report ----------------------------------------------------------------
  //
  // Key names match nam_benchmark's where the two overlap, so the Bencher
  // scripts read either without knowing which produced it.

  std::string json;
  json.reserve(1 << 16);
  char buffer[1024];

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
  append("  \"schemaVersion\": 1,\n  \"producer\": \"nam_ir_benchmark\",\n");
  append("  \"environment\": {\"deviceModel\": \"%s\", \"cpu\": \"%s\", \"platform\": \"%s\", "
         "\"architecture\": \"%s\", \"fpu\": \"%s\", \"osVersion\": \"%s\", \"totalCores\": %d, "
         "\"cpuGovernor\": \"%s\"},\n",
         environment.deviceModel.c_str(), environment.cpu.c_str(), environment.platform.c_str(),
         environment.architecture.c_str(), environment.fpu.c_str(), environment.osVersion.c_str(),
         environment.totalCores, governor.c_str());
  append("  \"note\": \"%s\",\n", testbedNote.c_str());
  append("  \"socTemperatureStartC\": %.1f,\n  \"socTemperatureEndC\": %.1f,\n", temperatureStart,
         temperatureEnd);
  append("  \"config\": {\"irChannels\": %d, \"warmupSeconds\": %.1f, "
         "\"timingWindowSeconds\": %.1f, \"acceptFraction\": %.4f, \"acceptTolerance\": %.4f, "
         "\"minSamples\": %d, \"maxAttempts\": %d},\n",
         irChannels, config.warmupSeconds, config.timingWindowSeconds, config.acceptFraction,
         config.acceptTolerance, config.minSamples, config.maxAttempts);
  {
    json += "  \"counters\": {\"requested\": [";
    for (size_t i = 0; i < counterEvents.size(); i++)
      append("%s\"%s\"", i ? "," : "", counterEvents[i].c_str());
    json += "], \"opened\": [";
    for (size_t i = 0; i < counters.names().size(); i++)
      append("%s\"%s\"", i ? "," : "", counters.names()[i].c_str());
    append("], \"counterBudget\": %d",
           counterEvents.empty() ? 0 : nb::Counters::counter_budget());
    if (!counterEvents.empty() && !counters.is_open())
      append(", \"unavailableReason\": \"%s\"", counters.reason().c_str());
    json += "},\n";
  }
  append("  \"audio\": {\"fileName\": \"%s\", \"frameCount\": %zu, \"sampleRate\": %.1f, "
         "\"durationSeconds\": %.6f},\n",
         fs::path(audioPath).filename().string().c_str(), audio.samples.size(), audio.sampleRate,
         audio.durationSeconds());

  json += "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); i++)
  {
    const IrResult& r = results[i];
    append("    {\"variant\": \"%s\", \"taps\": %d, \"requestedTaps\": %d, \"blockSize\": %d, "
           "\"irChannels\": %d, \"requested\": \"%s\", \"implementation\": \"%s\", "
           "\"fftPartitions\": %d, \"fftBlockSize\": %d, \"succeeded\": %s, ",
           r.timing.variant.c_str(), r.taps, r.requestedTaps, r.blockSize, r.irChannels,
           r.requested.c_str(), r.active.c_str(), r.partitions, r.fftBlock,
           r.timing.succeeded ? "true" : "false");
    append("\"meanMs\": %.6f, \"medianMs\": %.6f, \"minMs\": %.6f, \"maxMs\": %.6f, "
           "\"spread\": %.6f, \"standardDeviationMs\": %.6f, \"realTimeFactor\": %.6f, "
           "\"corePercent\": %.6f, ",
           r.timing.meanMs, r.timing.medianMs, r.timing.minMs, r.timing.maxMs, r.timing.spread,
           r.timing.standardDeviationMs, r.timing.realTimeFactor, r.timing.corePercent);
    append("\"blockMedianPercent\": %.6f, \"blockP99Percent\": %.6f, \"blockMaxPercent\": %.6f, "
           "\"blockSampleCount\": %zu, ",
           r.blockMedianPercent, r.blockP99Percent, r.blockMaxPercent, r.blockSampleCount);
    if (!r.timing.failureReason.empty())
      append("\"failureReason\": \"%s\", ", r.timing.failureReason.c_str());
    {
      size_t sampleCount = 0;
      bool truncated = false;
      for (const Attempt& a : r.timing.attempts)
      {
        sampleCount += a.samplesMs.size();
        truncated = truncated || a.hitHardDeadline;
      }
      append("\"sampleCount\": %zu, \"hitHardDeadline\": %s, ", sampleCount,
             truncated ? "true" : "false");
    }
    if (!r.timing.counters.empty())
    {
      json += "\"counters\": {";
      for (size_t c = 0; c < r.timing.counters.size(); c++)
      {
        const nb::CounterSummary& counter = r.timing.counters[c];
        append("%s\"%s\": {\"median\": %.0f, \"min\": %.0f, \"max\": %.0f, "
               "\"config\": %llu, \"worstScaling\": %.4f, \"alwaysZero\": %s}",
               c ? ", " : "", counter.name.c_str(), counter.median, counter.min, counter.max,
               static_cast<unsigned long long>(counter.config), counter.worstScaling,
               counter.alwaysZero ? "true" : "false");
      }
      json += "}, ";
    }
    json += "\"acceptedMs\": ";
    append_array(r.timing.acceptedMs);
    json += ", \"discardedMs\": ";
    append_array(r.timing.discardedMs);
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

  bool allSucceeded = !results.empty();
  for (const IrResult& r : results)
    allSucceeded = allSucceeded && r.timing.succeeded;
  return allSucceeded ? 0 : 1;
}
