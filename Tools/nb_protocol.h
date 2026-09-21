// The measurement protocol, shared by every benchmark driver in this project.
//
// Tools/nam_benchmark.cpp times WaveNet engines; Tools/nam_ir_benchmark.cpp
// times impulse-response convolution. They measure different things, but they
// have to measure them the same way, because their numbers end up on the same
// Bencher project and get read against each other. Everything that decides what
// a published number means lives here: how the input audio is loaded, how long
// the warm-up runs, which samples the tightest-70% window keeps, what counts as
// agreement, how many attempts a subject gets, and what is recorded about the
// machine that produced it.
//
// It is a port of BenchCore's protocol, not an approximation of it — same
// warm-up, same timing window, same selection, same (max - min) / min spread
// test, same retry-and-reject — so a number from here and a number from
// `nambench` are directly comparable.
//
// What is NOT here is anything that knows what is being timed. The drivers
// supply that through PassHooks: run() does one pass and returns how long it
// took, reset() puts the subject back to a known state between passes. The loop
// below neither knows nor cares whether that pass ran a WaveNet or a
// convolution.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "nb_counters.h"

#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <sys/types.h>
#endif
#include <unistd.h>

namespace nbp
{

namespace fs = std::filesystem;

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

inline uint16_t read_u16(const std::vector<uint8_t>& b, size_t i)
{
  return static_cast<uint16_t>(b[i] | (static_cast<uint16_t>(b[i + 1]) << 8));
}

inline uint32_t read_u32(const std::vector<uint8_t>& b, size_t i)
{
  return static_cast<uint32_t>(b[i]) | (static_cast<uint32_t>(b[i + 1]) << 8)
         | (static_cast<uint32_t>(b[i + 2]) << 16) | (static_cast<uint32_t>(b[i + 3]) << 24);
}

inline std::vector<uint8_t> read_file(const fs::path& path)
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

inline bool load_wav(const fs::path& path, Audio& out, std::string& error)
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

inline double mean_of(const std::vector<double>& v)
{
  if (v.empty())
    return 0.0;
  double sum = 0.0;
  for (const double x : v)
    sum += x;
  return sum / static_cast<double>(v.size());
}

inline double median_of(std::vector<double> v)
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const size_t mid = v.size() / 2;
  return (v.size() % 2 == 0) ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
}

inline double stddev_of(const std::vector<double>& v)
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
inline WindowSelection tightest_window(const std::vector<double>& samples, double fraction)
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

inline double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

inline std::string trimmed(std::string s)
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
inline std::string read_text(const fs::path& path)
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
inline double soc_temperature_c()
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
  /// The floating-point unit the binary was compiled for.
  ///
  /// On 32-bit ARM this is part of the arithmetic's identity, not a tuning note:
  /// -mfpu=neon gives NEON without fused multiply-add, and Eigen then computes
  /// a2_fast's 8-channel path with non-fused vmlaq_f32 instead of vfmaq_f32.
  /// Two runs that disagree here are not comparable, so it is reported rather
  /// than assumed from the architecture.
  std::string fpu = "unknown";
  std::string osVersion = "unknown";
  int totalCores = 0;
};

inline Environment capture_environment()
{
  Environment env;

#if defined(__aarch64__)
  env.architecture = "arm64";
#elif defined(__x86_64__)
  env.architecture = "x86_64";
#elif defined(__arm__) || defined(__ARM_EABI__)
  env.architecture = "armv7";
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #if defined(__ARM_FEATURE_FMA)
  env.fpu = "neon+fma";
  #else
  env.fpu = "neon";
  #endif
#elif defined(__aarch64__)
  env.fpu = "asimd";
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

  // 32-bit ARM Linux DOES put a "model name" in /proc/cpuinfo, but it is useless
  // — "ARMv7 Processor rev 1 (v7l)", the same string for every ARMv7 part ever
  // made. Taking it would hide the MIDR lookup below and label every RK3288
  // result with something that says nothing about which core produced it, which
  // on a testbed that exists to characterise one specific core is worse than no
  // answer at all.
  //
  // Matched on the shape rather than one spelling: an AArch64 kernel running a
  // 32-bit process reports "ARMv8 Processor rev 0 (v8l)" by the same convention,
  // and that is just as uninformative.
  if (env.cpu.rfind("ARMv", 0) == 0 && env.cpu.find(" Processor") != std::string::npos)
    env.cpu.clear();

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
      // 32-bit ARMv7 parts. 0xc0d is the one that matters here: ARM shipped the
      // core as Cortex-A12 and later rebranded it Cortex-A17, and the RK3288 in
      // the HeadRush Core and Prime reports 0xc0d. Naming only one of the two
      // would make every result from that board look like it came from the wrong
      // chip.
      {"0xc05", "Cortex-A5"},   {"0xc07", "Cortex-A7"},  {"0xc08", "Cortex-A8"},
      {"0xc09", "Cortex-A9"},   {"0xc0d", "Cortex-A12/A17"}, {"0xc0e", "Cortex-A17"},
      {"0xc0f", "Cortex-A15"},
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
      env.cpu = std::string(env.architecture == "armv7" ? "ARM part " : "AArch64 part ") + part;
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
inline std::string cpu_governor()
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

/// Fewest samples a spread may be computed from.
///
/// Below this the "tightest 70%" statistic is not measuring dispersion at all —
/// at one sample it reports exactly zero — so an attempt with fewer is rejected
/// rather than accepted on a number that was never tested.
constexpr int kMinSamplesForWindow = 3;

struct Attempt
{
  int attempt = 0;
  int warmupPasses = 0;
  std::vector<double> samplesMs;
  bool accepted = false;
  double spread = 0.0;
  /// The timing loop stopped on the deadline rather than on having both a full
  /// window and minSamples. Recorded so a truncated run is visible downstream
  /// instead of being indistinguishable from a clean one.
  bool hitHardDeadline = false;
  std::string rejectionReason;
  /// One entry per timed pass, in the same order as samplesMs, when counters
  /// were requested. Kept per-pass rather than as a running total so a pass the
  /// window discards can be discarded from the counters too.
  std::vector<nb::CounterSample> counterSamples;
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
  /// Which attempt was accepted, 1-based, or 0 if none was.
  int acceptedAttempt = 0;
  /// Indices into the accepted attempt's samplesMs that the window kept.
  ///
  /// The protocol reduces the counters over exactly these. A driver that
  /// collected its own per-pass detail through PassHooks::onTimedPass reduces
  /// that over them too, so every statistic in a result describes the same set
  /// of passes.
  std::vector<size_t> acceptedIndices;
  /// Reduced from the accepted attempt's passes. Empty unless --counters asked
  /// for them and the machine could provide them.
  std::vector<nb::CounterSummary> counters;
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

inline Parity compare_outputs(const std::vector<double>& reference, const std::vector<double>& comparison,
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
// The measurement protocol itself.
// ---------------------------------------------------------------------------

/// The parts of a driver's configuration that decide what a number means.
///
/// A driver's own Config carries these plus whatever else it needs (which model
/// file, which block size, which impulse response). Only the fields that change
/// the *protocol* live here, so that a driver cannot quietly measure to a
/// different standard than the others by adding a knob of its own.
struct ProtocolConfig
{
  double warmupSeconds = 5.0;
  double timingWindowSeconds = 30.0;
  double acceptFraction = 0.70;
  double acceptTolerance = 0.03;
  int minSamples = 15;
  int maxAttempts = 5;
  bool resetBetweenPasses = true;
  bool quiet = false;
};

/// What the driver has to supply for a subject to be measurable.
struct PassHooks
{
  /// Put the subject back to its initial state. Called before each pass when
  /// resetBetweenPasses is set, and always outside the timed region.
  std::function<void()> reset;

  /// Run one pass over the whole input and return how long it took, in
  /// nanoseconds, measured by the subject itself rather than by this loop — the
  /// shims bracket their own process loop, so the clock reads are inside the
  /// library being measured, not across a shared-library call boundary.
  std::function<uint64_t()> run;

  /// Called after each *timed* pass, with that pass's index into
  /// Attempt::samplesMs. A driver that collects per-pass detail of its own —
  /// the IR driver's per-block times, say — harvests it here, and later uses
  /// Result::acceptedIndices to discard whatever the window discarded.
  std::function<void(size_t passIndex)> onTimedPass;
};

/// Run the protocol against one subject.
///
/// `label` names it in progress output and in the result. `durationSeconds` is
/// how much audio one pass covers, which is what turns a pass time into a
/// real-time factor and a percentage of one core. `counters` may be null.
inline Result measure(const std::string& label, const ProtocolConfig& config,
                      double durationSeconds, const PassHooks& hooks, nb::Counters* counters)
{
  Result result;
  result.variant = label;

  for (int attempt = 1; attempt <= config.maxAttempts; attempt++)
  {
    Attempt record;
    record.attempt = attempt;

    // 1. Warm up for a fixed duration, discarding the results. Always completes
    //    at least one pass, and always lets the in-flight pass finish, so the
    //    subject is never left mid-buffer going into timing.
    {
      const double start = monotonic_seconds();
      do
      {
        if (config.resetBetweenPasses && hooks.reset)
          hooks.reset();
        hooks.run();
        record.warmupPasses++;
      } while (monotonic_seconds() - start < config.warmupSeconds);
    }

    // 2. Time for the configured window.
    //
    // The hard deadline stops a machine so slow that minSamples would never be
    // reached from running forever. It is sized from the work rather than from
    // the window alone: a fixed 3x multiple of a window an operator shortened to
    // make a long sweep tractable defeats minSamples by construction, which is
    // exactly the regime a Cortex-A17 on the 8-channel submodel sits in.
    {
      const double start = monotonic_seconds();
      double hardDeadline = start + config.timingWindowSeconds * 3.0;
      bool deadlineSized = false;
      while (true)
      {
        if (config.resetBetweenPasses && hooks.reset)
          hooks.reset();
        // Counting starts after the reset and stops before the clock is read, so
        // the counters cover the same work the timing does. A reset is real work
        // with real cache traffic, and charging it to the subject would make a
        // bigger working set look like a worse implementation.
        if (counters && counters->is_open())
          counters->begin();
        const uint64_t elapsed = hooks.run();
        if (counters && counters->is_open())
        {
          nb::CounterSample sample;
          counters->end(sample);
          record.counterSamples.push_back(std::move(sample));
        }
        record.samplesMs.push_back(static_cast<double>(elapsed) / 1.0e6);
        if (hooks.onTimedPass)
          hooks.onTimedPass(record.samplesMs.size() - 1);

        const double now = monotonic_seconds();

        if (!deadlineSized)
        {
          // One pass is now measured, so allow enough time for minSamples of
          // them plus half again, if that is longer than the flat multiple.
          const double onePass = now - start;
          const double needed = onePass * config.minSamples * 1.5;
          if (needed > (hardDeadline - start))
            hardDeadline = start + needed;
          deadlineSized = true;
        }

        const bool windowDone = (now - start) >= config.timingWindowSeconds;
        const bool enoughSamples = static_cast<int>(record.samplesMs.size()) >= config.minSamples;
        if ((windowDone && enoughSamples) || now >= hardDeadline)
        {
          record.hitHardDeadline = (now >= hardDeadline) && !(windowDone && enoughSamples);
          break;
        }
      }
    }

    // A window needs samples to be a window. tightest_window takes
    // max(1, ceil(fraction * n)) values, so a single sample yields a width of
    // one, a spread of (x - x) / x = 0, and an attempt that passes the agreement
    // test without ever having been tested. Reject that here, before the spread
    // is consulted, rather than reporting a number no dispersion was ever
    // measured on.
    if (record.samplesMs.size() < kMinSamplesForWindow)
    {
      char reason[256];
      std::snprintf(reason, sizeof(reason),
                    "only %zu sample(s) before the hard deadline; a spread cannot be measured "
                    "from fewer than %d",
                    record.samplesMs.size(), kMinSamplesForWindow);
      record.rejectionReason = reason;
      result.attempts.push_back(record);
      if (!config.quiet)
        std::printf("%s: attempt %d rejected — %s\n", label.c_str(), attempt,
                    record.rejectionReason.c_str());
      continue;
    }

    // 3. Does the required fraction agree?
    const WindowSelection selection = tightest_window(record.samplesMs, config.acceptFraction);
    if (!selection.valid)
    {
      record.rejectionReason =
        "could not select a window from " + std::to_string(record.samplesMs.size()) + " samples";
      result.attempts.push_back(record);
      if (!config.quiet)
        std::printf("%s: attempt %d rejected — %s\n", label.c_str(), attempt,
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
        std::printf("%s: attempt %d rejected — %s\n", label.c_str(), attempt, reason);
      continue;
    }

    // Which passes the window kept, by index into this attempt's samplesMs.
    //
    // Selecting by duration rather than by index is exact up to ties, and a pass
    // tied on time with an accepted one is an equally good pass. Both the
    // counters and whatever the driver collected in onTimedPass are reduced over
    // these and only these: a pass the window threw away was thrown away because
    // something else happened during it, and that something did work the
    // counters faithfully recorded.
    for (size_t s = 0; s < record.samplesMs.size(); s++)
    {
      if (record.samplesMs[s] >= selection.min && record.samplesMs[s] <= selection.max)
        result.acceptedIndices.push_back(s);
    }

    if (!record.counterSamples.empty() && counters)
    {
      std::vector<nb::CounterSample> kept;
      kept.reserve(result.acceptedIndices.size());
      for (const size_t s : result.acceptedIndices)
      {
        if (s < record.counterSamples.size())
          kept.push_back(record.counterSamples[s]);
      }
      result.counters = nb::summarise(counters->names(), counters->events(), kept);
    }

    result.attempts.push_back(record);
    result.succeeded = true;
    result.acceptedAttempt = attempt;
    result.acceptedMs = selection.accepted;
    result.discardedMs = selection.discarded;
    result.meanMs = selection.mean;
    result.medianMs = selection.median;
    result.minMs = selection.min;
    result.maxMs = selection.max;
    result.spread = selection.spread;
    result.standardDeviationMs = stddev_of(selection.accepted);
    result.realTimeFactor = selection.mean > 0.0 ? durationSeconds / (selection.mean / 1000.0) : 0.0;
    result.corePercent =
      durationSeconds > 0.0 ? (selection.mean / 1000.0) / durationSeconds * 100.0 : 0.0;
    break;
  }

  if (!result.succeeded)
  {
    result.failureReason =
      "no attempt met the agreement threshold in " + std::to_string(config.maxAttempts)
      + " attempts";
    if (!result.attempts.empty())
      result.discardedMs = result.attempts.back().samplesMs;
  }

  return result;
}

/// The per-subject progress line, and the counters under it.
inline void report_progress(const Result& result)
{
  if (result.succeeded)
    std::printf("%s: %.3f%% of one core  (%.2f ms, spread %.2f%%, %zu accepted / %zu discarded)\n",
                result.variant.c_str(), result.corePercent, result.meanMs, result.spread * 100.0,
                result.acceptedMs.size(), result.discardedMs.size());
  else
    std::printf("%s: FAILED — %s\n", result.variant.c_str(), result.failureReason.c_str());

  for (const nb::CounterSummary& counter : result.counters)
  {
    std::printf("    %-20s %14.0f", counter.name.c_str(), counter.median);
    if (counter.alwaysZero)
      std::printf("   (read zero on every pass — this core probably does not implement it)");
    else if (counter.worstScaling < 0.999)
      std::printf("   (multiplexed: scaled up from %.0f%% of the pass, so an estimate)",
                  counter.worstScaling * 100.0);
    std::printf("\n");
  }
}

} // namespace nbp
