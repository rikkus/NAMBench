// Portable conformance driver.
//
// One of these is built per variant (see CMakeLists.txt), because linking two
// checkouts of NeuralAmpModelerCore into one process would collide on every
// nam:: symbol. The Xcode build solves that with four hidden-visibility
// frameworks; this build solves it with four executables, which means the
// cross-variant comparison happens between processes rather than inside one.
// Scripts/compare-conformance.py performs it, from the files written here.
//
// What is asserted is arithmetic, not time. The driver runs one deterministic
// pass over a generated signal from a known-reset state and writes the output
// samples verbatim. Nothing here is a benchmark: the elapsed nanoseconds the
// shim returns are read and discarded, because a number measured under CMake on
// a shared cloud runner has no relationship to the numbers this project
// publishes.
//
// Every variant sees a bit-identical input buffer. It is generated rather than
// read from audio-input/input.wav so that the driver needs no WAV parser and no
// transcendental functions, both of which vary across platforms in ways that
// would show up as a spurious parity failure.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "nam_bench_shim.h"

#if !defined(NB_PREFIX)
  #error "NB_PREFIX must be defined (e.g. -DNB_PREFIX=nb_upstream)"
#endif

// One level of indirection so NB_PREFIX expands before the ## in
// NB_DECLARE_VARIANT pastes it. Without it the declarations would come out as
// NB_PREFIX_probe rather than nb_upstream_probe.
#define NB_DECLARE_VARIANT_(P) NB_DECLARE_VARIANT(P)
#define NB_DECLARE_KERNEL_LAB_(P) NB_DECLARE_KERNEL_LAB(P)

extern "C" {
NB_DECLARE_VARIANT_(NB_PREFIX)
#if defined(NB_ENABLE_SLIM_LAB) || defined(NB_ENABLE_FULL_LAB)
NB_DECLARE_KERNEL_LAB_(NB_PREFIX)
#endif
}

#define NB_CAT_(a, b) a##b
#define NB_CAT(a, b) NB_CAT_(a, b)
#define NB_FN(suffix) NB_CAT(NB_PREFIX, suffix)

#if defined(NB_ENABLE_SLIM_LAB) || defined(NB_ENABLE_FULL_LAB)
  #define NB_IS_LAB 1
#else
  #define NB_IS_LAB 0
#endif

// The architecture this binary was *compiled for*, which is what decides
// routing — the fused detector returns false unless these macros are set. It is
// recorded rather than left to the comparator to infer from the host, because
// under a cross-build (or Rosetta) the host and the binary disagree and the
// binary is the one that is right.
#if defined(__aarch64__) || defined(_M_ARM64)
  #define NB_ARCH "aarch64"
#elif defined(__x86_64__) || defined(_M_X64)
  #define NB_ARCH "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
  #define NB_ARCH "x86"
#elif defined(__arm__) || defined(_M_ARM)
  #define NB_ARCH "arm"
#else
  #define NB_ARCH "unknown"
#endif

// Recorded for the report rather than for any decision. The planar kernels now
// follow __aarch64__ alone, so the architecture is what decides whether they are
// present — but knowing an Apple Silicon Mac from a Raspberry Pi is still worth
// having in a result that someone reads six months later.
#if defined(__APPLE__)
  #include <TargetConditionals.h>
  #if TARGET_OS_IPHONE
    #define NB_PLATFORM "ios"
  #else
    #define NB_PLATFORM "macos"
  #endif
#elif defined(__ANDROID__)
  #define NB_PLATFORM "android"
#elif defined(__linux__)
  #define NB_PLATFORM "linux"
#elif defined(_WIN32)
  #define NB_PLATFORM "windows"
#else
  #define NB_PLATFORM "unknown"
#endif

#if defined(__clang__)
  #define NB_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
  #define NB_STR2_(x) #x
  #define NB_STR_(x) NB_STR2_(x)
  #define NB_COMPILER "gcc " NB_STR_(__GNUC__) "." NB_STR_(__GNUC_MINOR__)
#elif defined(_MSC_VER)
  #define NB_STR2_(x) #x
  #define NB_STR_(x) NB_STR2_(x)
  #define NB_COMPILER "msvc " NB_STR_(_MSC_VER)
#else
  #define NB_COMPILER "unknown"
#endif

namespace
{

namespace fs = std::filesystem;

// --- Input signal ------------------------------------------------------------

/// Deterministic pseudo-random excitation, identical on every platform.
///
/// A hand-rolled xorshift rather than <random>: the engines are the subject, and
/// a standard library whose generator or distribution differs by implementation
/// would feed two platforms different audio and make the outputs incomparable
/// for a reason that has nothing to do with the engines.
///
/// The envelope is integer-derived triangle rather than a sine, for the same
/// reason: std::sin is not required to be correctly rounded and differs between
/// libms, which would put a platform difference into the input.
std::vector<double> make_input(size_t frames)
{
  std::vector<double> signal(frames);

  uint64_t state = 0x9E3779B97F4A7C15ull; // any fixed non-zero seed
  const size_t period = 9600;             // 0.2 s at 48 kHz

  for (size_t i = 0; i < frames; i++)
  {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;

    // Top 32 bits to [-1, 1). Exact in binary64, so no rounding to disagree on.
    const uint32_t bits = static_cast<uint32_t>(state >> 32);
    const double noise = (static_cast<double>(bits) * (1.0 / 2147483648.0)) - 1.0;

    // Triangle from 0 to 1 and back, so the run covers quiet passages where a
    // WaveNet's tails decay toward denormal range as well as loud ones.
    const size_t phase = i % period;
    const size_t half = period / 2;
    const size_t rise = phase < half ? phase : (period - phase);
    const double envelope = static_cast<double>(rise) / static_cast<double>(half);

    signal[i] = 0.25 * noise * envelope;
  }

  return signal;
}

// --- Reporting ---------------------------------------------------------------

const char* engine_name(NbEngine engine)
{
  switch (engine)
  {
    case NbEngineGeneric: return "generic";
    case NbEngineA2Fast: return "a2_fast";
    case NbEngineFused: return "fused";
    case NbEngineSlim: return "slim";
    case NbEngineFull: return "full";
    case NbEnginePlanar: return "a2_planar";
    case NbEngineUnknown: break;
  }
  return "unknown";
}

/// One completed run, ready to be written out as a JSON object.
struct Record
{
  std::string label;    // e.g. "widest" or "widest__k03_a2p4"
  std::string submodel; // "widest" | "narrowest"
  std::string kernel;   // lab kernel name, or empty
  int kernelIndex = -1;
  std::string engine;
  int32_t channels = 0;
  double sampleRate = 0.0;
  double checksum = 0.0;
  uint64_t nonFinite = 0;
  double peak = 0.0;
  std::string samplesFile;
};

/// Write the output samples as raw little-endian binary64.
///
/// Raw rather than JSON because the comparison downstream is numerical to the
/// last bit: several lab kernels are verbatim ports that are expected to be
/// bit-identical to the engine they stand in for, and a decimal round-trip
/// through JSON would quietly destroy exactly the property being tested.
///
/// Every platform this builds for is little-endian. If that ever stops being
/// true the comparator's numbers will be visibly wrong rather than subtly so.
bool write_samples(const fs::path& path, const std::vector<double>& samples)
{
  std::FILE* file = std::fopen(path.string().c_str(), "wb");
  if (file == nullptr)
    return false;
  const size_t written = std::fwrite(samples.data(), sizeof(double), samples.size(), file);
  std::fclose(file);
  return written == samples.size();
}

void json_escape(std::string& out, const std::string& text)
{
  for (const char c : text)
  {
    if (c == '"' || c == '\\')
      out += '\\';
    out += c;
  }
}

/// Hand-written rather than via nlohmann, to keep Eigen and the JSON header out
/// of this translation unit. The shape is small and fixed.
bool write_report(const fs::path& path, const std::string& variant, int hasFused,
                  size_t frames, int blockSize, const std::vector<Record>& records)
{
  std::string out;
  char scratch[512];

  out += "{\n  \"variant\": \"";
  json_escape(out, variant);
  out += "\",\n  \"arch\": \"" NB_ARCH "\",\n  \"platform\": \"" NB_PLATFORM "\",\n  \"compiler\": \"";
  json_escape(out, NB_COMPILER);
  out += "\",\n";

  std::snprintf(scratch, sizeof(scratch),
                "  \"has_fused\": %s,\n  \"frames\": %llu,\n  \"block_size\": %d,\n"
                "  \"is_lab\": %s,\n  \"records\": [\n",
                hasFused ? "true" : "false", static_cast<unsigned long long>(frames), blockSize,
                NB_IS_LAB ? "true" : "false");
  out += scratch;

  for (size_t i = 0; i < records.size(); i++)
  {
    const Record& r = records[i];
    out += "    {\"label\": \"";
    json_escape(out, r.label);
    out += "\", \"submodel\": \"";
    json_escape(out, r.submodel);
    out += "\", \"kernel\": ";
    if (r.kernel.empty())
    {
      out += "null";
    }
    else
    {
      out += '"';
      json_escape(out, r.kernel);
      out += '"';
    }
    std::snprintf(scratch, sizeof(scratch),
                  ", \"kernel_index\": %d, \"engine\": \"%s\", \"channels\": %d, "
                  "\"sample_rate\": %.10g, \"checksum\": %.17g, \"non_finite\": %llu, "
                  "\"peak\": %.17g, \"samples_file\": \"",
                  r.kernelIndex, r.engine.c_str(), static_cast<int>(r.channels), r.sampleRate,
                  r.checksum, static_cast<unsigned long long>(r.nonFinite), r.peak);
    out += scratch;
    json_escape(out, r.samplesFile);
    out += "\"}";
    out += (i + 1 < records.size()) ? ",\n" : "\n";
  }

  out += "  ]\n}\n";

  std::FILE* file = std::fopen(path.string().c_str(), "wb");
  if (file == nullptr)
    return false;
  const size_t written = std::fwrite(out.data(), 1, out.size(), file);
  std::fclose(file);
  return written == out.size();
}

// --- Running one case --------------------------------------------------------

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

/// True when `value` is neither NaN nor an infinity.
///
/// Written as arithmetic rather than std::isfinite because this build is
/// compiled -O3 across four toolchains and the check is the one thing that must
/// survive all of them: `v == v` fails only for NaN, and `v - v == 0` fails only
/// for an infinity.
bool is_finite(double value)
{
  return value == value && value - value == 0.0;
}

/// Build, run and record one (submodel, kernel) case.
bool run_case(const std::vector<uint8_t>& nam, NbSubmodel mode, const char* submodelName,
              int kernelIndex, const char* kernelName, const std::vector<double>& input,
              int blockSize, const fs::path& outDir, const std::string& variant,
              std::vector<Record>& records)
{
  char err[512] = {0};

  NbProbe probe{};
  if (NB_FN(_probe)(nam.data(), nam.size(), mode, 0, &probe, err, sizeof(err)) != 0)
  {
    std::fprintf(stderr, "%s/%s: probe failed: %s\n", variant.c_str(), submodelName, err);
    return false;
  }

#if NB_IS_LAB
  if (kernelIndex >= 0 && NB_FN(_select_kernel)(kernelIndex) != 0)
  {
    std::fprintf(stderr, "%s: no kernel %d\n", variant.c_str(), kernelIndex);
    return false;
  }
#else
  (void)kernelIndex;
#endif

  NbModel* model =
    NB_FN(_create)(nam.data(), nam.size(), mode, 0, blockSize, err, sizeof(err));
  if (model == nullptr)
  {
    std::fprintf(stderr, "%s/%s: create failed: %s\n", variant.c_str(), submodelName, err);
    return false;
  }

  Record record;
  record.submodel = submodelName;
  record.engine = engine_name(NB_FN(_engine)(model));
  record.channels = NB_FN(_channels)(model);
  record.sampleRate = NB_FN(_sample_rate)(model);
  record.kernelIndex = kernelIndex;
  if (kernelName != nullptr)
    record.kernel = kernelName;

  record.label = submodelName;
  if (kernelIndex >= 0)
  {
    char suffix[128];
    std::snprintf(suffix, sizeof(suffix), "__k%02d_%s", kernelIndex,
                  kernelName != nullptr ? kernelName : "unnamed");
    record.label += suffix;
  }

  // Reset explicitly, so the pass starts from the same state whether or not
  // _create happened to leave it there.
  NB_FN(_reset)(model, record.sampleRate, blockSize);

  std::vector<double> output(input.size(), 0.0);
  double checksum = 0.0;
  const uint64_t elapsedNs =
    NB_FN(_process)(model, input.data(), input.size(), output.data(), &checksum);
  (void)elapsedNs; // measured, deliberately unused — see the file header

  NB_FN(_destroy)(model);

  record.checksum = checksum;
  for (const double sample : output)
  {
    if (!is_finite(sample))
      record.nonFinite++;
    const double magnitude = sample < 0.0 ? -sample : sample;
    if (magnitude > record.peak)
      record.peak = magnitude;
  }

  record.samplesFile = variant + "__" + record.label + ".f64";
  if (!write_samples(outDir / record.samplesFile, output))
  {
    std::fprintf(stderr, "%s: could not write %s\n", variant.c_str(), record.samplesFile.c_str());
    return false;
  }

  std::printf("  %-28s engine=%-8s channels=%d peak=%.6f checksum=%.6f%s\n",
              record.label.c_str(), record.engine.c_str(), static_cast<int>(record.channels),
              record.peak, record.checksum,
              record.nonFinite != 0 ? "  ** NON-FINITE **" : "");

  records.push_back(std::move(record));
  return true;
}

void print_usage()
{
  std::printf(
    "nam_conformance — run one engine variant over a fixed signal and record the output\n"
    "\n"
    "  --model <path>       .nam to load (required)\n"
    "  --out <dir>          where to write records (default: ./conformance)\n"
    "  --frames <n>         samples to process (default: 48000)\n"
    "  --block-size <n>     frames per process() call (default: 64)\n"
    "\n"
    "Writes <out>/<variant>.json plus one raw binary64 file per case.\n"
    "Scripts/compare-conformance.py checks them against each other.\n");
}

} // namespace

int main(int argc, char** argv)
{
  std::string modelPath;
  fs::path outDir = "conformance";
  size_t frames = 48000;
  int blockSize = 64;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];
    const bool hasValue = (i + 1 < argc);
    if (arg == "--model" && hasValue)
      modelPath = argv[++i];
    else if (arg == "--out" && hasValue)
      outDir = argv[++i];
    else if (arg == "--frames" && hasValue)
      frames = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    else if (arg == "--block-size" && hasValue)
      blockSize = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    else if (arg == "--help" || arg == "-h")
    {
      print_usage();
      return 0;
    }
    else
    {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", arg.c_str());
      return 2;
    }
  }

  if (modelPath.empty())
  {
    std::fprintf(stderr, "error: --model is required\n");
    return 2;
  }
  if (frames == 0 || blockSize <= 0)
  {
    std::fprintf(stderr, "error: --frames and --block-size must be positive\n");
    return 2;
  }

  const std::vector<uint8_t> nam = read_file(modelPath);
  if (nam.empty())
  {
    std::fprintf(stderr, "error: could not read %s\n", modelPath.c_str());
    return 1;
  }

  std::error_code ec;
  fs::create_directories(outDir, ec);
  if (ec)
  {
    std::fprintf(stderr, "error: could not create %s: %s\n", outDir.string().c_str(),
                 ec.message().c_str());
    return 1;
  }

  const std::string variant = NB_FN(_variant_name)();
  const int hasFused = NB_FN(_has_fused)();

  std::printf("%s — %s, %s (fused compiled in: %s)\n", variant.c_str(), NB_ARCH, NB_COMPILER,
              hasFused ? "yes" : "no");

  const std::vector<double> input = make_input(frames);
  std::vector<Record> records;
  bool ok = true;

  // Which cases this variant runs follows the same rule main.swift applies:
  // read the shape, do not trust the flag.
  //
  //   upstream / fused   both submodels. On the 3-channel submodel the fork's
  //                      detector declines the shape and falls through to the
  //                      generic engine; that is asserted downstream rather
  //                      than avoided, because it is the fallback behaviour
  //                      that matters.
  //   slim lab           the 3-channel submodel only, every kernel.
  //   full lab           the 8-channel submodel only, every kernel.
#if defined(NB_ENABLE_SLIM_LAB)
  const int kernels = NB_FN(_kernel_count)();
  for (int k = 0; k < kernels; k++)
    ok &= run_case(nam, NbSubmodelNarrowest, "narrowest", k, NB_FN(_kernel_name)(k), input,
                   blockSize, outDir, variant, records);
#elif defined(NB_ENABLE_FULL_LAB)
  const int kernels = NB_FN(_kernel_count)();
  for (int k = 0; k < kernels; k++)
    ok &= run_case(nam, NbSubmodelWidest, "widest", k, NB_FN(_kernel_name)(k), input, blockSize,
                   outDir, variant, records);
#else
  ok &= run_case(nam, NbSubmodelWidest, "widest", -1, nullptr, input, blockSize, outDir, variant,
                 records);
  ok &= run_case(nam, NbSubmodelNarrowest, "narrowest", -1, nullptr, input, blockSize, outDir,
                 variant, records);
#endif

  const fs::path reportPath = outDir / (variant + ".json");
  if (!write_report(reportPath, variant, hasFused, frames, blockSize, records))
  {
    std::fprintf(stderr, "error: could not write %s\n", reportPath.string().c_str());
    return 1;
  }

  std::printf("  wrote %s (%zu records)\n", reportPath.string().c_str(), records.size());
  return ok ? 0 : 1;
}
