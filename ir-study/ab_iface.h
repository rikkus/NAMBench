// Shared interface for linear_ab.cpp. Each side is compiled in its own
// translation unit against its own headers (ADT's dsp.h and NAM's dsp.h
// clash), so only this header is common.
#pragma once
#include <cstdint>
#include <vector>

struct AbConv
{
  virtual ~AbConv() = default;
  virtual void Process(const double* in, double* out, int n) = 0;
};

AbConv* make_adt(const std::vector<float>& ir, double rate, int maxFrames);
AbConv* make_nam(const std::vector<float>& ir, double rate, int maxFrames);

inline void ab_ftz()
{
#if defined(__aarch64__)
  uint64_t v;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(v));
  v |= 1ull << 24;
  __asm__ __volatile__("msr fpcr, %0" ::"r"(v));
#endif
}
