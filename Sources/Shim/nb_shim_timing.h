// Denormal flushing and the benchmark clock, shared by every shim that times
// a process loop, so that all of them measure the same way.

#pragma once

#include <chrono>
#include <cstdint>

#include <time.h>

// --- Denormal flushing -------------------------------------------------------
//
// Applied identically in both variants around the block loop. A WaveNet's
// decaying tails can drift into denormal range, where the penalty is large and
// unrepresentative of how the plugin actually runs.


namespace
{

#if defined(__aarch64__)
constexpr uint64_t kFpcrFlushToZero = 1ull << 24; // FPCR.FZ

inline uint64_t fpcr_read()
{
  uint64_t value;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
  return value;
}

inline void fpcr_write(uint64_t value)
{
  __asm__ __volatile__("msr fpcr, %0" : : "r"(value));
}

inline uint64_t denormals_disable()
{
  const uint64_t previous = fpcr_read();
  fpcr_write(previous | kFpcrFlushToZero);
  return previous;
}

inline void denormals_restore(uint64_t previous)
{
  fpcr_write(previous);
}
#elif defined(__arm__) || defined(__ARM_EABI__)
// AArch32. FPSCR.FZ is the same idea as AArch64's FPCR.FZ and sits in the same
// bit, but it is reached through the coprocessor move instructions instead.
//
// This branch is not cosmetic, and it is not merely about speed the way the
// AArch64 one is. On AArch32 the two floating-point units disagree by default:
// Advanced SIMD is *unconditionally* flush-to-zero for single precision, while
// VFP scalar honours this bit. a2_fast's Channels == 3 branch is scalar VFP and
// the lab's kernels are NEON, so without setting FZ the reference and the
// candidate treat a decaying tail's denormals differently — and bit-identity
// then fails for a reason that has nothing to do with the kernel.
//
// FZ only. Not DN (bit 25): the AArch64 branch above sets FZ alone, and the two
// must agree about what they are changing or the platforms stop being
// comparable.
constexpr uint32_t kFpscrFlushToZero = 1u << 24; // FPSCR.FZ

inline uint32_t fpscr_read()
{
  uint32_t value;
  __asm__ __volatile__("vmrs %0, fpscr" : "=r"(value));
  return value;
}

inline void fpscr_write(uint32_t value)
{
  __asm__ __volatile__("vmsr fpscr, %0" : : "r"(value));
}

inline uint64_t denormals_disable()
{
  const uint32_t previous = fpscr_read();
  fpscr_write(previous | kFpscrFlushToZero);
  return previous;
}

inline void denormals_restore(uint64_t previous)
{
  fpscr_write(static_cast<uint32_t>(previous));
}
#else
inline uint64_t denormals_disable()
{
  return 0;
}
inline void denormals_restore(uint64_t)
{
}
#endif

// Apple keeps the clock it has always used, unchanged: CLOCK_UPTIME_RAW is
// monotonic, unadjusted, and does not tick while the machine is asleep, which
// is what every published number was measured against.
//
// Elsewhere — the Raspberry Pi and the Tinker Board, whose numbers are
// published alongside the Mac's — steady_clock is the portable equivalent:
// monotonic everywhere, and on Linux backed by the same CLOCK_MONOTONIC the
// driver's own wall clock uses, so the two agree about how long a pass took.
#if defined(__APPLE__)
inline uint64_t now_ns()
{
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
#else
inline uint64_t now_ns()
{
  const auto since_epoch = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch).count());
}
#endif

} // namespace
