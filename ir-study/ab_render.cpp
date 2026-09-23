// Render a WAV through dsp::ImpulseResponse exactly as the plugin would: load
// the IR file, resample to the session rate, truncate to 8192 taps, process in
// fixed blocks. Built once against each AudioDSPTools tree, so the two outputs
// differ only by the convolution. Writes raw float64; ab_mix.py makes WAVs.
//
//   for t in upstream partitioned; do
//     src=(vendor/adt-$t/dsp/{ImpulseResponse,dsp,wav}.cpp)
//     [[ -f vendor/adt-$t/dsp/PartitionedConvolution.cpp ]] && src+=(vendor/adt-$t/dsp/PartitionedConvolution.cpp)
//     c++ -std=c++17 -O3 -Ivendor/adt-$t/dsp -Ivendor/adt-$t -isystem vendor/eigen-adt \
//       ir-study/ab_render.cpp "${src[@]}" -o ab_render_$t
//   done
//   ./ab_render_upstream    cab.wav audio-input/input.wav 64 out/cab_up.f64
//   ./ab_render_partitioned cab.wav audio-input/input.wav 64 out/cab_fft.f64
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "ImpulseResponse.h"
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


int main(int argc, char** argv)
{
  if (argc < 5) { std::fprintf(stderr, "usage: %s ir.wav in.wav block out.f64\n", argv[0]); return 2; }
  std::vector<double> in; double rate = 0;
  if (!read_wav(argv[2], in, rate)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
  const size_t block = std::strtoul(argv[3], nullptr, 10);
  dsp::ImpulseResponse ir(argv[1], rate);
  if (ir.GetWavState() != dsp::wav::LoadReturnCode::SUCCESS) { std::fprintf(stderr, "IR load failed\n"); return 1; }
  // 0.5 s of silence after the input so the IR's tail rings out.
  in.resize(in.size() + static_cast<size_t>(rate / 2), 0.0);
  std::vector<double> out(in.size()), scratch(block);
  for (size_t off = 0; off < in.size(); off += block)
  {
    const size_t n = std::min(block, in.size() - off);
    std::copy(in.begin() + off, in.begin() + off + n, scratch.begin());
    double* p = scratch.data();
    double** o = ir.Process(&p, 1, n);
    std::copy(o[0], o[0] + n, out.begin() + off);
  }
  std::ofstream f(argv[4], std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(double));
  std::fprintf(stderr, "%s: %zu frames at %.0f Hz\n", argv[4], out.size(), rate);
  return 0;
}
