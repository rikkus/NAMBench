// Per-block timing traces for the IR convolution, for charting burstiness.
//
// Two sources of subjects:
//   shim  - the benchmark's own shim dylibs (true upstream vs the branch's FFT)
//   pc    - dsp::PartitionedConvolution compiled in directly, so the IR
//           contents can be chosen (the shim only generates its synthetic IR)
//
// Output: CSV on stdout, one row per timed block. Five experiments: the 8192-tap
// headline, the tap ladder, the input signal, the IR's contents (real cabinet
// IRs listed in a tab-separated "name<TAB>path" file), and the block size.
//
// Build against the benchmark's own shim libraries (cmake target
// nam_ir_benchmark builds them into build-benchmark/):
//
//   c++ -std=c++17 -O3 -ISources/Shim -Ivendor/adt-partitioned -isystem vendor/eigen-adt \
//     ir-study/ir_trace.cpp vendor/adt-partitioned/dsp/PartitionedConvolution.cpp \
//     -Lbuild-benchmark -lnam_ir_upstream -lnam_ir_partitioned \
//     -Wl,-rpath,$PWD/build-benchmark -o ir_trace
//   ./ir_trace audio-input/input.wav irs.txt > trace.csv
//   python3 ir-study/trace_reduce.py trace.csv ir-study/data/m2-trace-summary.json

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#if defined(__APPLE__)
  #include <pthread.h>
  #include <sys/qos.h>
#endif

#include "nam_ir_shim.h"
#include "dsp/PartitionedConvolution.h"

extern "C" {
NB_IR_DECLARE_VARIANT(nb_ir_upstream)
NB_IR_DECLARE_VARIANT(nb_ir_partitioned)
}

static uint64_t now_ns()
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count());
}

// --- WAV reading (PCM 16/24/32, float32; first channel) ----------------------
static bool read_wav(const std::string& path, std::vector<double>& out, double& rate)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4))
    return false;
  size_t pos = 12;
  int fmt = 0, ch = 0, bits = 0;
  const char* data = nullptr;
  size_t dataLen = 0;
  while (pos + 8 <= d.size())
  {
    uint32_t len;
    std::memcpy(&len, d.data() + pos + 4, 4);
    const char* body = d.data() + pos + 8;
    if (!std::memcmp(d.data() + pos, "fmt ", 4))
    {
      uint16_t f16, c16, b16;
      uint32_t r32;
      std::memcpy(&f16, body, 2);
      std::memcpy(&c16, body + 2, 2);
      std::memcpy(&r32, body + 4, 4);
      std::memcpy(&b16, body + 14, 2);
      fmt = f16;
      ch = c16;
      rate = r32;
      bits = b16;
      if (fmt == 0xFFFE)
      {
        uint16_t sub;
        std::memcpy(&sub, body + 24, 2);
        fmt = sub;
      }
    }
    else if (!std::memcmp(d.data() + pos, "data", 4))
    {
      data = body;
      dataLen = std::min<size_t>(len, d.size() - (pos + 8));
    }
    pos += 8 + len + (len & 1);
  }
  if (!data || ch <= 0)
    return false;
  const size_t bps = static_cast<size_t>(bits / 8);
  const size_t frames = dataLen / (bps * ch);
  out.resize(frames);
  for (size_t i = 0; i < frames; i++)
  {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data + i * bps * ch);
    double v = 0;
    if (fmt == 3 && bits == 32)
    {
      float x;
      std::memcpy(&x, p, 4);
      v = x;
    }
    else if (bits == 16)
      v = static_cast<int16_t>(p[0] | (p[1] << 8)) / 32768.0;
    else if (bits == 24)
    {
      int32_t x = (p[0] << 8) | (p[1] << 16) | (p[2] << 24);
      v = (x >> 8) / 8388608.0;
    }
    else if (bits == 32)
    {
      int32_t x;
      std::memcpy(&x, p, 4);
      v = x / 2147483648.0;
    }
    out[i] = v;
  }
  return true;
}

// Same generator as the shim, so "synthetic" here is the benchmark's IR.
static std::vector<float> synthetic_ir(int taps)
{
  std::vector<float> ir(static_cast<size_t>(taps));
  uint32_t state = 0x9E3779B9u;
  float env = 1.0f;
  for (int i = 0; i < taps; i++)
  {
    state = state * 1664525u + 1013904223u;
    const int32_t bits = static_cast<int32_t>(state >> 16) - 32768;
    ir[static_cast<size_t>(i)] = static_cast<float>(bits) * (1.0f / 32768.0f) * env;
    env *= 0.9975f;
    if (env < 1.0e-6f)
      env = 1.0e-6f;
  }
  ir[0] = 1.0f;
  return ir;
}

struct Shim
{
  NbIr* (*create)(int32_t, int32_t, double, NbIrImpl, int32_t, char*, size_t);
  void (*destroy)(NbIr*);
  void (*reset)(NbIr*, int32_t);
  uint64_t (*process)(NbIr*, const double*, size_t, double*, double*, uint64_t*);
  int32_t (*partitions)(const NbIr*);
  int32_t (*fft_block)(const NbIr*);
};

static const Shim kUp = {&nb_ir_upstream_ir_create, &nb_ir_upstream_ir_destroy, &nb_ir_upstream_ir_reset,
                         &nb_ir_upstream_ir_process, &nb_ir_upstream_ir_partitions,
                         &nb_ir_upstream_ir_fft_block};
static const Shim kPart = {&nb_ir_partitioned_ir_create, &nb_ir_partitioned_ir_destroy,
                           &nb_ir_partitioned_ir_reset, &nb_ir_partitioned_ir_process,
                           &nb_ir_partitioned_ir_partitions, &nb_ir_partitioned_ir_fft_block};

static const double kRate = 48000.0;
static int gPasses = 12;
static double gPassSeconds = 2.0;

static void emit(const std::string& exp, const std::string& subject, int taps, int block,
                 const std::string& input, const std::string& ir, int pass, const std::vector<uint64_t>& ns)
{
  for (size_t i = 0; i < ns.size(); i++)
    std::printf("%s,%s,%d,%d,%s,%s,%d,%zu,%llu\n", exp.c_str(), subject.c_str(), taps, block, input.c_str(),
                ir.c_str(), pass, i, static_cast<unsigned long long>(ns[i]));
}

// Run one shim subject: warm up, then gPasses timed passes from a reset.
static void run_shim(const std::string& exp, const Shim& api, NbIrImpl impl, const std::string& subject,
                     int taps, int block, const std::string& inputName, const std::vector<double>& input)
{
  char err[256] = {0};
  NbIr* ir = api.create(taps, 1, kRate, impl, block, err, sizeof(err));
  if (!ir)
  {
    std::fprintf(stderr, "create failed: %s\n", err);
    std::exit(1);
  }
  const size_t frames = std::min(input.size(), static_cast<size_t>(gPassSeconds * kRate));
  const size_t blocks = (frames + block - 1) / block;
  std::vector<uint64_t> ns(blocks);
  // Warm-up: ~1 s of wall time.
  const uint64_t until = now_ns() + 1000000000ull;
  while (now_ns() < until)
  {
    api.reset(ir, block);
    api.process(ir, input.data(), frames, nullptr, nullptr, ns.data());
  }
  for (int p = 0; p < gPasses; p++)
  {
    api.reset(ir, block);
    api.process(ir, input.data(), frames, nullptr, nullptr, ns.data());
    emit(exp, subject, taps, block, inputName, "synthetic", p, ns);
  }
  std::fprintf(stderr, "%s %s taps=%d block=%d input=%s partitions=%d fftBlock=%d\n", exp.c_str(),
               subject.c_str(), taps, block, inputName.c_str(), api.partitions(ir), api.fft_block(ir));
  api.destroy(ir);
}

// Run dsp::PartitionedConvolution directly with a chosen kernel.
static void run_pc(const std::string& exp, dsp::ConvolutionImplementation impl, const std::string& subject,
                   const std::vector<float>& kernel, const std::string& irName, int block,
                   const std::string& inputName, const std::vector<double>& input)
{
  dsp::PartitionedConvolution pc;
  pc.SetKernel(kernel.data(), kernel.size(), impl);
  const size_t frames = std::min(input.size(), static_cast<size_t>(gPassSeconds * kRate));
  std::vector<float> in(frames), out(block);
  for (size_t i = 0; i < frames; i++)
    in[i] = static_cast<float>(input[i]);
  const size_t blocks = (frames + block - 1) / block;
  std::vector<uint64_t> stamps(blocks + 1), ns(blocks);
  volatile float sink = 0;
  auto pass = [&]() {
    pc.Reset(static_cast<size_t>(block));
    stamps[0] = now_ns();
    for (size_t b = 0, off = 0; b < blocks; b++, off += block)
    {
      const size_t n = std::min<size_t>(block, frames - off);
      pc.Process(in.data() + off, out.data(), n);
      stamps[b + 1] = now_ns();
    }
    sink = sink + out[0];
    for (size_t b = 0; b < blocks; b++)
      ns[b] = stamps[b + 1] - stamps[b];
  };
  const uint64_t until = now_ns() + 1000000000ull;
  while (now_ns() < until)
    pass();
  for (int p = 0; p < gPasses; p++)
  {
    pass();
    emit(exp, subject, static_cast<int>(kernel.size()), block, inputName, irName, p, ns);
  }
  std::fprintf(stderr, "%s %s ir=%s taps=%zu partitions=%zu fftBlock=%zu\n", exp.c_str(), subject.c_str(),
               irName.c_str(), kernel.size(), pc.GetNumPartitions(), pc.GetFftBlockSize());
}

int main(int argc, char** argv)
{
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  // argv: guitar.wav irdir-list-file
  if (argc < 3)
  {
    std::fprintf(stderr, "usage: ir_trace guitar.wav irs.txt\n");
    return 2;
  }
#if defined(__aarch64__)
  uint64_t fpcr;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= 1ull << 24;
  __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__arm__)
  uint32_t fpscr;
  __asm__ __volatile__("vmrs %0, fpscr" : "=r"(fpscr));
  fpscr |= 1u << 24;
  __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));
#endif

  std::vector<double> guitar;
  double rate = 0;
  if (!read_wav(argv[1], guitar, rate) || rate != kRate)
  {
    std::fprintf(stderr, "could not read %s as 48 kHz\n", argv[1]);
    return 1;
  }
  const size_t n = guitar.size();
  std::vector<double> noise(n), silence(n, 0.0), sine(n);
  uint32_t s = 12345;
  for (size_t i = 0; i < n; i++)
  {
    s = s * 1664525u + 1013904223u;
    noise[i] = (static_cast<int32_t>(s >> 16) - 32768) / 65536.0;
    sine[i] = 0.5 * std::sin(2.0 * M_PI * 110.0 * static_cast<double>(i) / kRate);
  }

  std::printf("exp,subject,taps,block,input,ir,pass,block_index,ns\n");

  // A: the headline case, 8192 taps, 64-frame blocks, guitar in.
  gPasses = 30;
  run_shim("headline", kUp, NbIrImplAuto, "upstream", 8192, 64, "guitar", guitar);
  run_shim("headline", kPart, NbIrImplFFT, "fft", 8192, 64, "guitar", guitar);

  // B: the tap ladder.
  gPasses = 12;
  for (int taps : {512, 1024, 2048, 4096, 8192})
  {
    run_shim("ladder", kUp, NbIrImplAuto, "upstream", taps, 64, "guitar", guitar);
    run_shim("ladder", kPart, NbIrImplFFT, "fft", taps, 64, "guitar", guitar);
  }

  // C: does the audio going in matter?
  for (auto& [name, sig] : std::vector<std::pair<std::string, const std::vector<double>*>>{
         {"guitar", &guitar}, {"white noise", &noise}, {"silence", &silence}, {"110 Hz sine", &sine}})
  {
    run_shim("input", kUp, NbIrImplAuto, "upstream", 8192, 64, name, *sig);
    run_shim("input", kPart, NbIrImplFFT, "fft", 8192, 64, name, *sig);
  }

  // D: does what is in the IR matter? Kernel set by hand, 8192 taps each.
  std::vector<std::pair<std::string, std::vector<float>>> kernels;
  kernels.push_back({"synthetic", synthetic_ir(8192)});
  {
    std::vector<float> k(8192, 0.0f);
    k[0] = 1.0f;
    kernels.push_back({"single click", k});
  }
  {
    std::ifstream list(argv[2]);
    std::string line;
    while (std::getline(list, line))
    {
      const size_t tab = line.find('\t');
      if (tab == std::string::npos)
        continue;
      std::vector<double> x;
      double r = 0;
      if (!read_wav(line.substr(tab + 1), x, r) || r != kRate)
      {
        std::fprintf(stderr, "skip %s\n", line.c_str());
        continue;
      }
      std::vector<float> k(8192, 0.0f);
      for (size_t i = 0; i < std::min<size_t>(8192, x.size()); i++)
        k[i] = static_cast<float>(x[i]);
      kernels.push_back({line.substr(0, tab), k});
    }
  }
  for (auto& [name, k] : kernels)
  {
    run_pc("ircontent", dsp::ConvolutionImplementation::Direct, "direct", k, name, 64, "guitar", guitar);
    run_pc("ircontent", dsp::ConvolutionImplementation::FFT, "fft", k, name, 64, "guitar", guitar);
  }

  // E: block size, 8192 taps.
  for (int block : {32, 64, 128, 256, 512, 1024})
  {
    run_shim("blocksize", kUp, NbIrImplAuto, "upstream", 8192, block, "guitar", guitar);
    run_shim("blocksize", kPart, NbIrImplFFT, "fft", 8192, block, "guitar", guitar);
  }
  return 0;
}
