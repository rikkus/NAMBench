// Where does the FFT path's transform callback spend its time, and does
// spreading its work across the quiet callbacks flatten it?
//
// Each variant from make_spread_variants.py (PC0 = the reference, then the
// revisions under test) and PCt (e2dc6bc with a clock read between the phases
// of _RunFftBlock) takes turns in an order that rotates every round, as in
// spectrum_storage_ab.cpp. Same noise input and synthetic IR for all of them,
// FFT path forced, at 1024, 2048, 4096 and 8192 taps.
//
// For each variant: the median cost at each position in the FFT block's cycle
// of callbacks (the profile the change is meant to flatten), the transform
// callback's median, p99 and max over all callbacks, and the mean. When the
// callback size does not divide the FFT block (48), callbacks are classed by
// whether they contain a block boundary instead. Every variant's output is
// compared with PC0's; PCt's phases are reported in us and as a share of its
// transform callback, and their sum is checked against the uninstrumented
// e2dc6bc variant.
//
//   python3 ir-study/make_spread_variants.py <adt clone> OUT e2dc6bc <B> <C>
//   c++ -std=c++17 -O3 -DNDEBUG -IOUT -isystem vendor/eigen-adt \
//     ir-study/spread_ab.cpp OUT/PC*.cpp -o spread_ab
//   ./spread_ab [cpu to pin to, Linux] [rounds] [frames: 32, 48, 64 or 128]
//
// ARMv7 (Tinker Board), cross-built on the Pi with the benchmark's
// nb_conformance_flags:
//   arm-linux-gnueabihf-g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra \
//     -Wno-unused-parameter -Wno-psabi -mcpu=cortex-a17 -mfpu=neon-vfpv4 -mfloat-abi=hard \
//     -ffp-contract=fast -static-libstdc++ -static-libgcc -IOUT \
//     -isystem vendor/eigen-adt ir-study/spread_ab.cpp OUT/PC*.cpp -o spread_ab
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
#include "variants.h"

namespace pct
{
unsigned long long phase_ns[4];
unsigned long long now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace pct

using namespace dsp;
static uint64_t now() { return pct::now_ns(); }
static void ftz()
{
#if defined(__aarch64__)
  uint64_t v; __asm__ __volatile__("mrs %0, fpcr" : "=r"(v)); v |= 1ull << 24; __asm__ __volatile__("msr fpcr, %0" ::"r"(v));
#elif defined(__arm__)
  uint32_t v; __asm__ __volatile__("vmrs %0, fpscr" : "=r"(v)); v |= 1u << 24; __asm__ __volatile__("vmsr fpscr, %0" ::"r"(v));
#endif
}

struct Conv
{
  std::string name, rev;
  bool timed_phases = false;
  virtual ~Conv() = default;
  virtual void Set(const float* k, int taps) = 0;
  virtual void Reset(size_t maxFrames) = 0;
  virtual void Process(const float* in, float* out, size_t n) = 0;
  virtual int Block() = 0;
};
template <class C> struct Wrap : Conv
{
  C c;
  Wrap(const char* n, const char* r, bool t) { name = n; rev = r; timed_phases = t; }
  void Set(const float* k, int taps) override { c.SetKernel(k, (size_t)taps, ConvolutionImplementation::FFT); }
  void Reset(size_t m) override { c.Reset(m); }
  void Process(const float* in, float* out, size_t n) override { c.Process(in, out, n); }
  int Block() override { return (int)c.GetFftBlockSize(); }
};

// Positions in the cycle: B/F of them when F divides B, else two classes
// (0 = no block boundary in the callback, 1 = at least one).
struct Cycle
{
  int B, F, P;
  bool by_boundary;
  Cycle(int b, int f) : B(b), F(f)
  {
    by_boundary = (B % F) != 0 || F >= B;
    P = by_boundary ? 2 : B / F;
  }
  bool boundary(size_t b) const { return ((b + 1) * F) / B > (b * F) / B; }
  int pos(size_t b) const { return by_boundary ? (boundary(b) ? 1 : 0) : (int)(b % P); }
};

struct Stats
{
  std::vector<std::vector<double>> pos;
  std::vector<double> xform, all;
  std::vector<std::array<double, 5>> phases; // fwd, mul, inv, ola, rest
};

static void pass(Conv& c, const std::vector<float>& in, std::vector<float>& out, const Cycle& cy, Stats* st)
{
  c.Reset((size_t)cy.F);
  const size_t n = in.size() / cy.F;
  if (st && st->pos.empty())
    st->pos.resize(cy.P);
  for (size_t b = 0; b < n; b++)
  {
    if (c.timed_phases)
      std::memset(pct::phase_ns, 0, sizeof pct::phase_ns);
    const uint64_t t0 = now();
    c.Process(in.data() + b * cy.F, out.data() + b * cy.F, (size_t)cy.F);
    const uint64_t t1 = now();
    if (!st)
      continue;
    const double us = (t1 - t0) / 1e3;
    st->all.push_back(us);
    st->pos[cy.pos(b)].push_back(us);
    if (cy.boundary(b))
    {
      st->xform.push_back(us);
      if (c.timed_phases)
      {
        std::array<double, 5> p;
        double sum = 0;
        for (int i = 0; i < 4; i++)
          sum += (p[i] = pct::phase_ns[i] / 1e3);
        p[4] = us - sum;
        st->phases.push_back(p);
      }
    }
  }
}

static double quant(std::vector<double> v, double q)
{
  if (v.empty())
    return NAN;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
}
static double med(const std::vector<double>& v) { return quant(v, 0.5); }
static double mean(const std::vector<double>& v)
{
  double s = 0;
  for (double x : v)
    s += x;
  return s / v.size();
}

static std::string compare(const std::vector<float>& ref, const std::vector<float>& x)
{
  if (std::memcmp(ref.data(), x.data(), ref.size() * sizeof(float)) == 0)
    return "byte-identical";
  double e = 0, r = 0, mx = 0, peak = 0;
  for (size_t i = 0; i < ref.size(); i++)
  {
    const double d = (double)x[i] - ref[i];
    e += d * d;
    r += (double)ref[i] * ref[i];
    mx = std::max(mx, std::fabs(d));
    peak = std::max(peak, (double)std::fabs(ref[i]));
  }
  char buf[128];
  snprintf(buf, sizeof buf, "%.1f dB below signal, max|diff| %.3g (%.3g of peak)", 10 * std::log10(e / r), mx, mx / peak);
  return buf;
}

int main(int argc, char** argv)
{
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(argc > 1 ? atoi(argv[1]) : 3, &set);
  sched_setaffinity(0, sizeof(set), &set);
#endif
  ftz();
  const int rounds = argc > 2 ? atoi(argv[2]) : 15;
  const int frames = argc > 3 ? atoi(argv[3]) : 64;

  std::vector<float> in(48000);
  uint32_t s = 12345;
  for (auto& v : in)
  {
    s = s * 1664525u + 1013904223u;
    v = ((int32_t)(s >> 16) - 32768) / 65536.f;
  }
  in.resize(in.size() / frames * frames);

  printf("spread_ab: %d-frame callbacks, %d rounds\n", frames, rounds);
  for (int taps : {1024, 2048, 4096, 8192})
  {
    std::vector<float> k(taps);
    float env = 1;
    uint32_t q = 0x9E3779B9u;
    for (int i = 0; i < taps; i++)
    {
      q = q * 1664525u + 1013904223u;
      k[i] = ((int32_t)(q >> 16) - 32768) / 32768.f * env;
      env *= 0.9975f;
      if (env < 1e-6f)
        env = 1e-6f;
    }
    k[0] = 1;

    std::vector<std::unique_ptr<Conv>> vs;
#define ADD(T, R) vs.emplace_back(new Wrap<T>(#T, R, false));
    SPREAD_VARIANTS(ADD)
#undef ADD
    vs.emplace_back(new Wrap<PCt>("PCt", "e2dc6bc+clock", true));
    for (auto& v : vs)
      v->Set(k.data(), taps);
    const Cycle cy(vs[0]->Block(), frames);

    std::vector<std::vector<float>> outs(vs.size(), std::vector<float>(in.size()));
    for (size_t i = 0; i < vs.size(); i++)
      pass(*vs[i], in, outs[i], cy, nullptr);
    std::vector<Stats> st(vs.size());
    for (int r = 0; r < rounds; r++)
      for (size_t j = 0; j < vs.size(); j++)
      {
        const size_t i = (j + r) % vs.size(); // rotate so no variant always runs first
        pass(*vs[i], in, outs[i], cy, &st[i]);
      }

    printf("\n%d taps, FFT block %d: ", taps, cy.B);
    if (cy.by_boundary)
      printf("callbacks classed as [no boundary | boundary]\n");
    else
      printf("%d callbacks per transform, the transform in the last\n", cy.P);
    for (size_t i = 0; i < vs.size(); i++)
    {
      printf("  %-3s %-13s | profile median (us):", vs[i]->name.c_str(), vs[i]->rev.c_str());
      for (int p = 0; p < cy.P; p++)
        printf(" %6.2f", med(st[i].pos[p]));
      printf(" | transform median %7.2f | p99 %7.2f | max %7.2f | mean %6.3f | %s\n", med(st[i].xform),
             quant(st[i].all, 0.99), quant(st[i].all, 1.0), mean(st[i].all), compare(outs[0], outs[i]).c_str());
    }

    const Stats& t = st.back();
    if (!t.phases.empty())
    {
      static const char* names[5] = {"forward FFT", "multiplies", "inverse FFT", "overlap-add", "rest"};
      const double total = med(t.xform);
      printf("  attribution (PCt, medians over transform callbacks, total %.2f us):", total);
      double sum = 0;
      for (int p = 0; p < 5; p++)
      {
        std::vector<double> col;
        for (auto& a : t.phases)
          col.push_back(a[p]);
        const double m = med(col);
        sum += m;
        printf(" %s %.2f (%.0f%%)%s", names[p], m, 100 * m / total, p < 4 ? "," : "");
      }
      double quiet = NAN;
      if (!cy.by_boundary && cy.P > 1)
      {
        std::vector<double> q;
        for (int p = 0; p < cy.P - 1; p++)
          q.insert(q.end(), t.pos[p].begin(), t.pos[p].end());
        quiet = med(q);
      }
      else if (cy.by_boundary)
        quiet = med(t.pos[0]);
      size_t base = 0;
      for (size_t i = 0; i + 1 < vs.size(); i++)
        if (vs[i]->rev == "e2dc6bc")
          base = i;
      const double un = med(st[base].xform);
      printf("\n    quiet median %.2f us; phases sum %.2f us against uninstrumented %s %.2f us (%+.1f%%)\n", quiet, sum,
             vs[base]->rev.c_str(), un, 100 * (sum / un - 1));
    }
  }
}
