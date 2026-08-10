// Portable benchmark driver.
//
// This one does measure. It exists because the numbers this project publishes
// come from the Xcode build, and the Xcode build does not run on a Raspberry Pi.
// A Cortex-A76 with 512 KB of L2 and no SVE is the least Apple-like AArch64 core
// the fused kernels are likely to meet, which makes it the most interesting
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

#if defined(__APPLE__)
  #include <pthread.h>
  #include <sys/sysctl.h>
  #include <sys/types.h>
#endif
#include <unistd.h>

extern "C" {
NB_DECLARE_VARIANT(nb_upstream)
NB_DECLARE_VARIANT(nb_planar)
NB_DECLARE_VARIANT(nb_fused)
#if defined(NAMBENCH_HAVE_LABS)
NB_DECLARE_VARIANT(nb_slim)
NB_DECLARE_KERNEL_LAB(nb_slim)
NB_DECLARE_VARIANT(nb_full)
NB_DECLARE_KERNEL_LAB(nb_full)
#endif
}

namespace
{

namespace fs = std::filesystem;

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
  } while (0)

/// One thing to measure: an engine, and for a lab, which kernel of it.
struct Subject
{
  std::string name; // "upstream", "fused", "slim:planar", "full:fu_s8_head"
  const EngineApi* api = nullptr;
  int kernel = -1;
};

// ---------------------------------------------------------------------------
// WAV
//
// A port of BenchCore's AudioLoader, deliberately hand-rolled for the same
// reason it is there: the input must land in memory as an exact, unresampled
// copy of what is on disk. The committed input.wav is 16-bit mono, but the file
// this reads is an argument, so the other PCM widths are handled too.
// ---------------------------------------------------------------------------
struct Audio
{
  std::vector<double> samples;
  double sampleRate = 0.0;
  int channels = 0;

  double durationSeconds() const
  {
    return sampleRate > 0 ? static_cast<double>(samples.size()) / sampleRate : 0.0;
  }
};

uint16_t read_u16(const std::vector<uint8_t>& b, size_t i)
{
  return static_cast<uint16_t>(b[i] | (static_cast<uint16_t>(b[i + 1]) << 8));
}

uint32_t read_u32(const std::vector<uint8_t>& b, size_t i)
{
  return static_cast<uint32_t>(b[i]) | (static_cast<uint32_t>(b[i + 1]) << 8)
         | (static_cast<uint32_t>(b[i + 2]) << 16) | (static_cast<uint32_t>(b[i + 3]) << 24);
}

std::vector<uint8_t> read_file(const fs::path& path)
{
  std::vector<uint8_t> bytes;
  std::FILE* file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr)
    return bytes;
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  if (size > 0)
  {
    bytes.resize(static_cast<size_t>(size));
    if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size())
      bytes.clear();
  }
  std::fclose(file);
  return bytes;
}

bool load_wav(const fs::path& path, Audio& out, std::string& error)
{
  const std::vector<uint8_t> b = read_file(path);
  if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0
      || std::memcmp(b.data() + 8, "WAVE", 4) != 0)
  {
    error = "not a RIFF/WAVE file: " + path.string();
    return false;
  }

  constexpr int kPCM = 1;
  constexpr int kFloat = 3;
  constexpr int kExtensible = 0xFFFE;

  int formatCode = 0, channels = 0, bits = 0;
  double sampleRate = 0.0;
  size_t dataBegin = 0, dataEnd = 0;

  // Walk the chunk list rather than assuming fmt comes first — the committed
  // input.wav has a JUNK chunk ahead of it.
  size_t cursor = 12;
  while (cursor + 8 <= b.size())
  {
    const char* id = reinterpret_cast<const char*>(b.data() + cursor);
    const size_t size = read_u32(b, cursor + 4);
    const size_t body = cursor + 8;
    if (body > b.size())
      break;
    const size_t end = std::min(body + size, b.size());

    if (std::memcmp(id, "fmt ", 4) == 0 && end - body >= 16)
    {
      formatCode = read_u16(b, body);
      channels = read_u16(b, body + 2);
      sampleRate = static_cast<double>(read_u32(b, body + 4));
      bits = read_u16(b, body + 14);
      if (formatCode == kExtensible && end - body >= 26)
        formatCode = read_u16(b, body + 24);
    }
    else if (std::memcmp(id, "data", 4) == 0)
    {
      dataBegin = body;
      dataEnd = end;
    }

    cursor = body + size + (size & 1); // chunks are word-aligned
  }

  if (channels <= 0 || bits <= 0)
  {
    error = "WAV file has no usable fmt chunk";
    return false;
  }
  if (dataEnd <= dataBegin)
  {
    error = "WAV file has no data chunk";
    return false;
  }

  const size_t stride = static_cast<size_t>(bits / 8) * static_cast<size_t>(channels);
  const size_t frames = stride > 0 ? (dataEnd - dataBegin) / stride : 0;
  if (frames == 0)
  {
    error = "WAV file contains no samples";
    return false;
  }

  out.samples.resize(frames);
  out.sampleRate = sampleRate;
  out.channels = channels;

  // Mono in, mono out: take channel 0 rather than downmixing, so the benchmark
  // feeds exactly the signal that is in the file.
  for (size_t f = 0; f < frames; f++)
  {
    const size_t o = dataBegin + f * stride;
    double value = 0.0;
    if (formatCode == kPCM && bits == 16)
    {
      value = static_cast<double>(static_cast<int16_t>(read_u16(b, o))) / 32768.0;
    }
    else if (formatCode == kPCM && bits == 24)
    {
      uint32_t raw = static_cast<uint32_t>(b[o]) | (static_cast<uint32_t>(b[o + 1]) << 8)
                     | (static_cast<uint32_t>(b[o + 2]) << 16);
      if (raw & 0x00800000u)
        raw |= 0xFF000000u;
      value = static_cast<double>(static_cast<int32_t>(raw)) / 8388608.0;
    }
    else if (formatCode == kPCM && bits == 32)
    {
      value = static_cast<double>(static_cast<int32_t>(read_u32(b, o))) / 2147483648.0;
    }
    else if (formatCode == kFloat && bits == 32)
    {
      const uint32_t raw = read_u32(b, o);
      float f32;
      std::memcpy(&f32, &raw, sizeof(f32));
      value = static_cast<double>(f32);
    }
    else if (formatCode == kFloat && bits == 64)
    {
      const uint64_t raw =
        static_cast<uint64_t>(read_u32(b, o)) | (static_cast<uint64_t>(read_u32(b, o + 4)) << 32);
      std::memcpy(&value, &raw, sizeof(value));
    }
    else
    {
      error = "unsupported WAV format (code " + std::to_string(formatCode) + ", "
              + std::to_string(bits) + "-bit)";
      return false;
    }
    out.samples[f] = value;
  }

  return true;
}

// ---------------------------------------------------------------------------
// Statistics — a port of BenchCore's Statistics.swift.
// ---------------------------------------------------------------------------

struct WindowSelection
{
  std::vector<double> accepted;
  std::vector<double> discarded;
  double spread = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  bool valid = false;
};

double mean_of(const std::vector<double>& v)
{
  if (v.empty())
    return 0.0;
  double sum = 0.0;
  for (const double x : v)
    sum += x;
  return sum / static_cast<double>(v.size());
}

double median_of(std::vector<double> v)
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const size_t mid = v.size() / 2;
  return (v.size() % 2 == 0) ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
}

double stddev_of(const std::vector<double>& v)
{
  if (v.size() < 2)
    return 0.0;
  const double m = mean_of(v);
  double acc = 0.0;
  for (const double x : v)
    acc += (x - m) * (x - m);
  return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

/// Tightest interval containing at least `fraction` of the samples.
///
/// Deliberately not a symmetric trim, for the reason Statistics.swift gives:
/// interference only ever makes a pass slower, so contamination is one-sided and
/// trimming equal tails would throw away good fast samples to balance bad slow
/// ones.
WindowSelection tightest_window(const std::vector<double>& samples, double fraction)
{
  WindowSelection out;
  if (samples.empty() || fraction <= 0.0 || fraction > 1.0)
    return out;

  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());

  const size_t width = std::max<size_t>(
    1, static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size()))));
  if (width > sorted.size())
    return out;

  size_t bestStart = 0;
  double bestSpread = std::numeric_limits<double>::infinity();
  for (size_t start = 0; start + width <= sorted.size(); start++)
  {
    const double low = sorted[start];
    const double high = sorted[start + width - 1];
    if (low <= 0.0)
      continue;
    const double spread = (high - low) / low;
    if (spread < bestSpread)
    {
      bestSpread = spread;
      bestStart = start;
    }
  }
  if (!std::isfinite(bestSpread))
    return out;

  out.accepted.assign(sorted.begin() + static_cast<long>(bestStart),
                      sorted.begin() + static_cast<long>(bestStart + width));
  out.discarded.assign(sorted.begin(), sorted.begin() + static_cast<long>(bestStart));
  out.discarded.insert(out.discarded.end(), sorted.begin() + static_cast<long>(bestStart + width),
                       sorted.end());
  out.spread = bestSpread;
  out.mean = mean_of(out.accepted);
  out.median = median_of(out.accepted);
  out.min = out.accepted.front();
  out.max = out.accepted.back();
  out.valid = true;
  return out;
}

// ---------------------------------------------------------------------------
// Clocks and environment
// ---------------------------------------------------------------------------

double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

std::string trimmed(std::string s)
{
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\0'))
    s.pop_back();
  size_t start = 0;
  while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    start++;
  return s.substr(start);
}

#if defined(__linux__)
/// Read a whole file that will not tell you how big it is.
///
/// Not read_file(): that sizes the file with fseek/ftell, and procfs and sysfs
/// both report st_size 0 for files with real content. Sizing them that way
/// yields an empty string for every one — which is not a loud failure but a
/// quiet one, and cost this driver both its CPU identification and, far worse,
/// its "the governor is not performance" warning. Read until EOF instead.
std::string read_text(const fs::path& path)
{
  std::FILE* file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr)
    return {};

  std::string out;
  char chunk[4096];
  size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0)
    out.append(chunk, got);
  std::fclose(file);

  return trimmed(out);
}
#endif

/// SoC temperature in Celsius, or -1 where the platform does not offer one.
///
/// Recorded either side of a run rather than gated on: a Pi that started at
/// 40 C and finished at 80 C did not measure the same machine twice, and that
/// is worth knowing after the fact even though nothing here can prevent it.
double soc_temperature_c()
{
#if defined(__linux__)
  const std::string raw = read_text("/sys/class/thermal/thermal_zone0/temp");
  if (!raw.empty())
  {
    const double milli = std::strtod(raw.c_str(), nullptr);
    if (milli > 0.0)
      return milli / 1000.0;
  }
#endif
  return -1.0;
}

struct Environment
{
  std::string deviceModel = "unknown";
  std::string cpu = "unknown";
  std::string platform = "unknown";
  std::string architecture = "unknown";
  std::string osVersion = "unknown";
  int totalCores = 0;
};

Environment capture_environment()
{
  Environment env;

#if defined(__aarch64__)
  env.architecture = "arm64";
#elif defined(__x86_64__)
  env.architecture = "x86_64";
#endif

  env.totalCores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));

#if defined(__APPLE__)
  env.platform = "macOS";
  char buffer[256];
  size_t length = sizeof(buffer);
  if (sysctlbyname("hw.model", buffer, &length, nullptr, 0) == 0)
    env.deviceModel = buffer;
  length = sizeof(buffer);
  if (sysctlbyname("machdep.cpu.brand_string", buffer, &length, nullptr, 0) == 0)
    env.cpu = buffer;
  length = sizeof(buffer);
  if (sysctlbyname("kern.osproductversion", buffer, &length, nullptr, 0) == 0)
    env.osVersion = buffer;
#elif defined(__linux__)
  env.platform = "Linux";

  // device-tree/model is what names a Raspberry Pi, and it distinguishes a
  // Pi 500 from a Pi 5 — same SoC, different thermal envelope, so the
  // distinction belongs in the report rather than in someone's memory.
  const std::string model = read_text("/proc/device-tree/model");
  if (!model.empty())
    env.deviceModel = model;

  const std::string cpuinfo = read_text("/proc/cpuinfo");

  auto field = [&cpuinfo](const char* key) -> std::string {
    const size_t at = cpuinfo.find(key);
    if (at == std::string::npos)
      return {};
    const size_t colon = cpuinfo.find(':', at);
    const size_t eol = cpuinfo.find('\n', at);
    if (colon == std::string::npos || eol == std::string::npos || colon > eol)
      return {};
    return trimmed(cpuinfo.substr(colon + 1, eol - colon - 1));
  };

  env.cpu = field("model name");

  // AArch64 Linux does not put a "model name" in /proc/cpuinfo — it reports the
  // MIDR fields instead, and it is lscpu that turns those into "Cortex-A76".
  // Doing the same here matters more than it looks: the whole reason to run on
  // this hardware is that a Cortex-A72 (Pi 4) and a Cortex-A76 (Pi 5, Pi 500)
  // are different machines, and a report that called both "unknown" would lose
  // exactly the distinction it was gathered to make.
  if (env.cpu.empty())
  {
    const std::string part = field("CPU part");
    static const struct
    {
      const char* id;
      const char* name;
    } kParts[] = {
      {"0xd03", "Cortex-A53"},  {"0xd04", "Cortex-A35"}, {"0xd05", "Cortex-A55"},
      {"0xd07", "Cortex-A57"},  {"0xd08", "Cortex-A72"}, {"0xd09", "Cortex-A73"},
      {"0xd0a", "Cortex-A75"},  {"0xd0b", "Cortex-A76"}, {"0xd0d", "Cortex-A77"},
      {"0xd41", "Cortex-A78"},  {"0xd42", "Cortex-A78AE"}, {"0xd44", "Cortex-X1"},
      {"0xd46", "Cortex-A510"}, {"0xd47", "Cortex-A710"}, {"0xd48", "Cortex-X2"},
      {"0xd49", "Neoverse-N2"}, {"0xd0c", "Neoverse-N1"}, {"0xd40", "Neoverse-V1"},
      {"0xd4f", "Neoverse-V2"},
    };
    for (const auto& entry : kParts)
    {
      if (part == entry.id)
      {
        env.cpu = entry.name;
        break;
      }
    }
    if (env.cpu.empty() && !part.empty())
      env.cpu = "AArch64 part " + part;
  }

  if (env.cpu.empty())
    env.cpu = field("Hardware");
  if (env.cpu.empty())
    env.cpu = "unknown";

  const std::string os = read_text("/etc/os-release");
  const size_t at = os.find("PRETTY_NAME=\"");
  if (at != std::string::npos)
  {
    const size_t begin = at + std::strlen("PRETTY_NAME=\"");
    const size_t end = os.find('"', begin);
    if (end != std::string::npos)
      env.osVersion = os.substr(begin, end - begin);
  }
#endif

  return env;
}

/// The CPU frequency governor, on platforms that have one.
///
/// Reported, and warned about, because `ondemand` is the single most effective
/// way to make a Linux benchmark meaningless: the governor spends the first part
/// of every pass ramping, so a faster engine gets measured at a lower clock than
/// a slower one.
std::string cpu_governor()
{
#if defined(__linux__)
  return read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
#else
  return "";
#endif
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

struct Attempt
{
  int attempt = 0;
  int warmupPasses = 0;
  std::vector<double> samplesMs;
  bool accepted = false;
  double spread = 0.0;
  std::string rejectionReason;
};

struct Result
{
  std::string variant;
  std::string engine;
  int32_t channels = 0;
  bool succeeded = false;
  std::string failureReason;
  std::vector<double> acceptedMs;
  std::vector<double> discardedMs;
  double meanMs = 0.0;
  double medianMs = 0.0;
  double minMs = 0.0;
  double maxMs = 0.0;
  double spread = 0.0;
  double standardDeviationMs = 0.0;
  double realTimeFactor = 0.0;
  /// What one instance costs as a percentage of one core while keeping up with
  /// real-time audio. The whole-file render time this comes from is an artefact
  /// of how long the test signal happens to be; this is not.
  double corePercent = 0.0;
  double checksum = 0.0;
  std::vector<Attempt> attempts;
};

struct Parity
{
  std::string referenceVariant;
  std::string comparisonVariant;
  double maxAbsoluteDifference = 0.0;
  double rmsDifference = 0.0;
  double referenceRMS = 0.0;
  double relativeRMS = 0.0;
  double decibelsBelowSignal = 0.0;
  bool withinTolerance = true;
};

Parity compare_outputs(const std::vector<double>& reference, const std::vector<double>& comparison,
                       const std::string& referenceName, const std::string& comparisonName,
                       double tolerance)
{
  Parity p;
  p.referenceVariant = referenceName;
  p.comparisonVariant = comparisonName;

  const size_t count = std::min(reference.size(), comparison.size());
  double sumSqDiff = 0.0, sumSqRef = 0.0;
  for (size_t i = 0; i < count; i++)
  {
    const double d = reference[i] - comparison[i];
    p.maxAbsoluteDifference = std::max(p.maxAbsoluteDifference, std::fabs(d));
    sumSqDiff += d * d;
    sumSqRef += reference[i] * reference[i];
  }

  const double divisor = static_cast<double>(std::max<size_t>(count, 1));
  p.rmsDifference = std::sqrt(sumSqDiff / divisor);
  p.referenceRMS = std::sqrt(sumSqRef / divisor);
  p.relativeRMS = p.referenceRMS > 0.0 ? p.rmsDifference / p.referenceRMS : 0.0;
  p.decibelsBelowSignal =
    p.relativeRMS > 0.0 ? -20.0 * std::log10(p.relativeRMS) : std::numeric_limits<double>::infinity();
  p.withinTolerance = p.relativeRMS <= tolerance;
  return p;
}

// ---------------------------------------------------------------------------
// Configuration — defaults copied from BenchCore's BenchmarkConfig.
// ---------------------------------------------------------------------------

struct Config
{
  int blockSize = 64;
  NbSubmodel submodel = NbSubmodelWidest;
  double warmupSeconds = 5.0;
  double timingWindowSeconds = 30.0;
  double acceptFraction = 0.70;
  double acceptTolerance = 0.03;
  int minSamples = 15;
  int maxAttempts = 5;
  bool resetBetweenPasses = true;
  bool checkParity = true;
  double parityWarnRelativeRMS = 1e-4;
  bool quiet = false;
};

const char* engine_name(NbEngine engine)
{
  switch (engine)
  {
    case NbEngineGeneric: return "generic";
    case NbEngineA2Fast: return "a2_fast";
    case NbEngineFused: return "fused";
    case NbEngineSlim: return "slim";
    case NbEngineFull: return "full";
    case NbEnginePlanar: return "planar";
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
  // the same would not be measuring the same machine — which showed up as this
  // driver reporting `fused` 1.9% faster and `upstream` 0.7% slower than the
  // Swift CLI on the same hardware, in the same minute.
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  Config config;
  std::string modelPath, audioPath;
  fs::path outputPath = "benchmark.json";
  std::string slimSpec, fullSpec;
  std::string testbedNote;
  bool includeFused = false;

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
      "  --list-slim/--list-full  print a kernel table and exit\n"
      "\n"
      "  --block-size <n>         frames per process() call (default 64)\n"
      "  --warmup-seconds <s>     discarded warm-up (default 5)\n"
      "  --timing-seconds <s>     timing window (default 30)\n"
      "  --accept-fraction <r>    fraction of samples kept (default 0.70)\n"
      "  --accept-tolerance <r>   required agreement (default 0.03)\n"
      "  --min-samples <n>        extend the window below this many (default 15)\n"
      "  --max-attempts <n>       attempts before giving up (default 5)\n"
      "  --with-fused             also measure the superseded fused engine\n"
      "  --no-parity              skip the output comparison\n"
      "  --note <text>            free text recorded in the report\n"
      "  --quiet, -q              only print the summary\n");
  };

  // --- Assemble the engine table -------------------------------------------

  EngineApi upstreamApi;
  upstreamApi.name = "upstream";
  upstreamApi.repository = "sdatkinson/NeuralAmpModelerCore";
  upstreamApi.codePath = "a2_fast";
  NB_FILL_BASE(upstreamApi, nb_upstream);

  EngineApi planarApi;
  planarApi.name = "planar";
  planarApi.repository = "rikkus/OptimisationWorkOnNeuralAmpModelerCore@apple-silicon-a2-planar";
  planarApi.codePath = "a2_planar (Core PR #313)";
  NB_FILL_BASE(planarApi, nb_planar);

  // Built, not measured. `fused` left the line-up when the planar kernels
  // superseded it; the API stays bound so a run can still ask for it by name.
  EngineApi fusedApi;
  fusedApi.name = "fused";
  fusedApi.repository = "rikkus/OptimisationWorkOnNeuralAmpModelerCore";
  fusedApi.codePath = "fused";
  NB_FILL_BASE(fusedApi, nb_fused);

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

  auto list_kernels = [](const EngineApi& api) {
    if (api.kernel_count == nullptr)
      return;
    const int count = api.kernel_count();
    std::printf("%s lab kernels (%d):\n", api.name, count);
    for (int i = 0; i < count; i++)
      std::printf("  %2d  %s\n", i, api.kernel_name(i));
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
    else if (arg == "--with-fused")
      includeFused = true;
    else if (arg == "--no-parity")
      config.checkParity = false;
    else if (arg == "--quiet" || arg == "-q")
      config.quiet = true;
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
  subjects.push_back({"upstream", &upstreamApi, -1});

  // The planar kernels cover both A2 submodels — 3 channels and 8 — so unlike
  // `fused` there is no shape gate here. What there is instead is a check that
  // they are actually present: off AArch64 the checkout compiles to plain
  // a2_fast, and measuring that against upstream would produce two identical
  // numbers and the false impression that the kernels achieve nothing.
  if (probe.channels == 3 || probe.channels == 8)
  {
    subjects.push_back({"planar", &planarApi, -1});
  }
  else if (!config.quiet)
  {
    std::printf("planar excluded: the kernels cover 3 and 8 channels, this submodel has %d\n",
                probe.channels);
  }

  if (includeFused)
  {
    if (probe.channels % 4 == 0)
    {
      subjects.push_back({"fused", &fusedApi, -1});
    }
    else if (!config.quiet)
    {
      std::printf("fused excluded: %d channels is not a multiple of 4, so its detector rejects "
                  "this shape and the fork falls through to the generic engine\n",
                  probe.channels);
    }
  }

#if defined(NAMBENCH_HAVE_LABS)
  auto add_lab = [&](const EngineApi& api, const std::string& spec, int wantChannels) {
    if (spec.empty() || spec == "none")
      return;
    if (probe.channels != wantChannels)
    {
      if (!config.quiet)
        std::printf("%s lab excluded: its kernels are specialised for %d channels and this "
                    "submodel has %d\n",
                    api.name, wantChannels, probe.channels);
      return;
    }
    const int count = api.kernel_count();
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
      subjects.push_back({std::string(api.name) + ":" + api.kernel_name(k), &api, k});
  };

  add_lab(slimApi, slimSpec, 3);
  add_lab(fullApi, fullSpec, 8);
#else
  if (!slimSpec.empty() || !fullSpec.empty())
    std::fprintf(stderr, "warning: this build has no kernel labs; --slim/--full ignored\n");
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
    // NAM_A2_PLANAR only where __aarch64__ is defined, so anywhere else the
    // planar checkout builds to plain a2_fast. It would then measure the same
    // code as `upstream`, land within noise of it, and read as "the kernels are
    // worth nothing" rather than "the kernels are not in this build".
    if (subject.name == "planar" && subject.api->engine(models[i]) != NbEnginePlanar)
    {
      std::fprintf(stderr,
                   "\nerror: the planar variant routed to %s, not to the planar kernels.\n"
                   "  a2_planar.h enables them where __aarch64__ is defined, so this build is\n"
                   "  measuring a2_fast twice. Refusing to report that as a comparison.\n"
                   "\n"
                   "  Expected on x86, and on MSVC/ARM64, which spells the architecture\n"
                   "  _M_ARM64 and never defines __aarch64__. Unexpected anywhere else — and\n"
                   "  on AArch64 it would mean the gate has regressed.\n",
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

  // --- Measure ---------------------------------------------------------------

  std::vector<Result> results;
  for (size_t i = 0; i < subjects.size(); i++)
  {
    const Subject& subject = subjects[i];
    NbModel* model = models[i];

    Result result;
    result.variant = subject.name;
    result.engine = engine_name(subject.api->engine(model));
    result.channels = subject.api->channels(model);
    result.checksum = checksums[i];

    for (int attempt = 1; attempt <= config.maxAttempts; attempt++)
    {
      Attempt record;
      record.attempt = attempt;

      // 1. Warm up for a fixed duration, discarding the results. Always
      //    completes at least one pass, and always lets the in-flight pass
      //    finish, so the model is never left mid-buffer going into timing.
      {
        const double start = monotonic_seconds();
        do
        {
          if (config.resetBetweenPasses)
            subject.api->reset(model, audio.sampleRate, config.blockSize);
          subject.api->process(model, audio.samples.data(), audio.samples.size(), nullptr, nullptr);
          record.warmupPasses++;
        } while (monotonic_seconds() - start < config.warmupSeconds);
      }

      // 2. Time for the configured window.
      {
        const double start = monotonic_seconds();
        const double hardDeadline = start + config.timingWindowSeconds * 3.0;
        while (true)
        {
          if (config.resetBetweenPasses)
            subject.api->reset(model, audio.sampleRate, config.blockSize);
          const uint64_t elapsed = subject.api->process(model, audio.samples.data(),
                                                        audio.samples.size(), nullptr, nullptr);
          record.samplesMs.push_back(static_cast<double>(elapsed) / 1.0e6);

          const double now = monotonic_seconds();
          const bool windowDone = (now - start) >= config.timingWindowSeconds;
          const bool enoughSamples =
            static_cast<int>(record.samplesMs.size()) >= config.minSamples;
          if ((windowDone && enoughSamples) || now >= hardDeadline)
            break;
        }
      }

      // 3. Does the required fraction agree?
      const WindowSelection selection = tightest_window(record.samplesMs, config.acceptFraction);
      if (!selection.valid)
      {
        record.rejectionReason = "could not select a window from "
                                 + std::to_string(record.samplesMs.size()) + " samples";
        result.attempts.push_back(record);
        if (!config.quiet)
          std::printf("%s: attempt %d rejected — %s\n", subject.name.c_str(), attempt,
                      record.rejectionReason.c_str());
        continue;
      }

      record.spread = selection.spread;
      record.accepted = selection.spread <= config.acceptTolerance;

      if (!record.accepted)
      {
        char reason[256];
        std::snprintf(reason, sizeof(reason),
                      "tightest %.0f%% of %zu samples spread %.2f%%, want within %.2f%%",
                      config.acceptFraction * 100.0, record.samplesMs.size(),
                      selection.spread * 100.0, config.acceptTolerance * 100.0);
        record.rejectionReason = reason;
        result.attempts.push_back(record);
        if (!config.quiet)
          std::printf("%s: attempt %d rejected — %s\n", subject.name.c_str(), attempt, reason);
        continue;
      }

      result.attempts.push_back(record);
      result.succeeded = true;
      result.acceptedMs = selection.accepted;
      result.discardedMs = selection.discarded;
      result.meanMs = selection.mean;
      result.medianMs = selection.median;
      result.minMs = selection.min;
      result.maxMs = selection.max;
      result.spread = selection.spread;
      result.standardDeviationMs = stddev_of(selection.accepted);
      result.realTimeFactor =
        selection.mean > 0.0 ? audio.durationSeconds() / (selection.mean / 1000.0) : 0.0;
      result.corePercent = audio.durationSeconds() > 0.0
                            ? (selection.mean / 1000.0) / audio.durationSeconds() * 100.0
                            : 0.0;
      break;
    }

    if (!result.succeeded)
    {
      result.failureReason = "no attempt met the agreement threshold in "
                             + std::to_string(config.maxAttempts) + " attempts";
      if (!result.attempts.empty())
        result.discardedMs = result.attempts.back().samplesMs;
    }

    if (!config.quiet)
    {
      if (result.succeeded)
        std::printf("%s: %.3f%% of one core  (%.2f ms, spread %.2f%%, %zu accepted / %zu "
                    "discarded)\n",
                    result.variant.c_str(), result.corePercent, result.meanMs, result.spread * 100.0,
                    result.acceptedMs.size(), result.discardedMs.size());
      else
        std::printf("%s: FAILED — %s\n", result.variant.c_str(), result.failureReason.c_str());
    }

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
         "\"architecture\": \"%s\", \"osVersion\": \"%s\", \"totalCores\": %d, "
         "\"cpuGovernor\": \"%s\"},\n",
         environment.deviceModel.c_str(), environment.cpu.c_str(), environment.platform.c_str(),
         environment.architecture.c_str(), environment.osVersion.c_str(), environment.totalCores,
         governor.c_str());
  append("  \"note\": \"%s\",\n", testbedNote.c_str());
  append("  \"socTemperatureStartC\": %.1f,\n  \"socTemperatureEndC\": %.1f,\n", temperatureStart,
         temperatureEnd);
  append("  \"config\": {\"blockSize\": %d, \"submodel\": \"%s\", \"warmupSeconds\": %.1f, "
         "\"timingWindowSeconds\": %.1f, \"acceptFraction\": %.4f, \"acceptTolerance\": %.4f, "
         "\"minSamples\": %d, \"maxAttempts\": %d},\n",
         config.blockSize, config.submodel == NbSubmodelNarrowest ? "narrowest" : "widest",
         config.warmupSeconds, config.timingWindowSeconds, config.acceptFraction,
         config.acceptTolerance, config.minSamples, config.maxAttempts);
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
