// The impulse response every IR variant convolves, shared so that it is the
// same floats in every shim.
//
// Generated rather than loaded, because what a convolution costs is set by how
// many taps it has and not by what is in them: every implementation measured
// touches every tap of every block regardless of their values. A generated IR
// is identical on every machine and in every variant, needs no asset committed
// to the repository, and cannot quietly differ between two runs being compared.
//
// The shape is a decaying, sign-alternating impulse train — roughly the
// envelope of a speaker cabinet measurement, without being one. Built from an
// integer LCG and a multiplicative decay, so there is no libm call in it and
// every platform produces the same floats. The decay floor keeps the tail well
// clear of denormal range on its own, rather than relying on the flush-to-zero
// the shims set: a denormal tail would be a property of the test signal rather
// than of the implementations, and it would land on whichever of them reads the
// oldest samples.

#pragma once

#include <cstdint>
#include <vector>

namespace
{

/// Seeds for the first and second IR channel.
constexpr uint32_t kNbIrSeedLeft = 0x9E3779B9u;
constexpr uint32_t kNbIrSeedRight = 0x85EBCA6Bu;

inline std::vector<float> nb_generate_ir(int32_t taps, uint32_t seed)
{
  std::vector<float> ir(static_cast<size_t>(taps), 0.0f);
  uint32_t state = seed;
  float envelope = 1.0f;
  // ~8 ms of decay at 48 kHz per e-fold, floored so the tail stays a normal
  // float no matter how long the IR is.
  const float decay = 0.9975f;
  for (int32_t i = 0; i < taps; i++)
  {
    state = state * 1664525u + 1013904223u;
    // Top 16 bits, mapped to [-1, 1). Exact in float.
    const int32_t bits = static_cast<int32_t>(state >> 16) - 32768;
    const float noise = static_cast<float>(bits) * (1.0f / 32768.0f);
    ir[static_cast<size_t>(i)] = noise * envelope;
    envelope *= decay;
    if (envelope < 1.0e-6f)
      envelope = 1.0e-6f;
  }
  // A leading direct sound, as a real measurement has.
  ir[0] = 1.0f;
  return ir;
}

} // namespace
