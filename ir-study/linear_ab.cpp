// AudioDSPTools partitioned-ir against NAM core's Linear FFT path, head to head.
//
// Both get the same synthetic 8192-tap IR (decaying noise at 48 kHz, so ADT
// does not resample) and the same noise input, FFT path forced on both. They
// take turns in an order that alternates every round, each round running
// `seconds` of audio through a fresh-warm instance. Every callback is timed;
// reported per implementation are the mean, p50, p99, p99.9 and max in us and
// as a share of the callback deadline at 48 kHz. ADT normalises the IR's gain,
// so outputs are compared after a least-squares scale.
//
//   ir-study/build_linear_ab.sh OUTDIR [extra flags, e.g. -mcpu=cortex-a76]
//   ./linear_ab [frames] [rounds] [seconds per round] [cpu to pin to, Linux]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

#include "ab_iface.h"

static uint64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

int main(int argc, char** argv)
{
  const int frames = argc > 1 ? atoi(argv[1]) : 64;
  const int rounds = argc > 2 ? atoi(argv[2]) : 10;
  const double seconds = argc > 3 ? atof(argv[3]) : 10.0;
#if defined(__linux__)
  if (argc > 4)
  {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(atoi(argv[4]), &s);
    sched_setaffinity(0, sizeof s, &s);
  }
#endif
  ab_ftz();
  const int taps = 8192;
  const double rate = 48000.0;
  std::mt19937 rng(1234);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> ir(taps);
  for (int i = 0; i < taps; i++)
    ir[i] = g(rng) * std::exp(-6.9f * i / taps) * 0.05f;
  const size_t callbacks = (size_t)(seconds * rate / frames);
  std::vector<double> input(callbacks * frames);
  for (auto& x : input)
    x = 0.25 * g(rng);

  const char* names[2] = {"adt partitioned-ir", "nam Linear"};
  std::vector<double> ns[2];
  std::vector<double> firstOut[2];
  for (int r = 0; r < rounds; r++)
    for (int k = 0; k < 2; k++)
    {
      const int which = (r & 1) ? 1 - k : k;
      std::unique_ptr<AbConv> c(which == 0 ? make_adt(ir, rate, frames) : make_nam(ir, rate, frames));
      std::vector<double> out(frames);
      // One second of warm-up so caches and branch predictors settle.
      for (size_t b = 0; b < (size_t)(rate / frames); b++)
        c->Process(&input[(b % callbacks) * frames], out.data(), frames);
      if (r < 2)
        firstOut[which].assign(input.size(), 0.0);
      std::unique_ptr<AbConv> fresh;
      if (r < 2)
      {
        // Output check runs on an instance that has seen only this input.
        fresh.reset(which == 0 ? make_adt(ir, rate, frames) : make_nam(ir, rate, frames));
      }
      for (size_t b = 0; b < callbacks; b++)
      {
        const uint64_t t0 = now_ns();
        c->Process(&input[b * frames], out.data(), frames);
        ns[which].push_back((double)(now_ns() - t0));
        if (fresh)
          fresh->Process(&input[b * frames], &firstOut[which][b * frames], frames);
      }
    }

  const double deadline_ns = frames / rate * 1e9;
  printf("frames %d, deadline %.1f us, %d rounds x %.1f s, 8192 taps\n", frames, deadline_ns / 1e3, rounds, seconds);
  printf("%-20s %8s %8s %8s %8s %8s   %7s %7s %7s\n", "impl", "mean", "p50", "p99", "p99.9", "max", "mean%", "p99%",
         "max%");
  for (int w = 0; w < 2; w++)
  {
    auto v = ns[w];
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v)
      sum += x;
    auto q = [&](double p) { return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
    const double mean = sum / v.size();
    printf("%-20s %8.2f %8.2f %8.2f %8.2f %8.2f   %6.2f%% %6.2f%% %6.2f%%\n", names[w], mean / 1e3, q(0.5) / 1e3,
           q(0.99) / 1e3, q(0.999) / 1e3, v.back() / 1e3, 100 * mean / deadline_ns, 100 * q(0.99) / deadline_ns,
           100 * v.back() / deadline_ns);
  }
  // Output agreement, after fitting ADT's gain normalisation.
  const auto& a = firstOut[0];
  const auto& b = firstOut[1];
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  const double s = ab / aa;
  double err = 0, peak = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    err = std::max(err, std::fabs(s * a[i] - b[i]));
    peak = std::max(peak, std::fabs(b[i]));
  }
  printf("outputs: adt x %.6f vs nam, max abs diff %.3g (%.1f dB below nam peak), correlation %.9f\n", s, err,
         20 * std::log10(err / peak), ab / std::sqrt(aa * bb));
  return 0;
}
