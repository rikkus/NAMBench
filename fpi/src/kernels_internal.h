#ifndef FPI_KERNELS_INTERNAL_H
#define FPI_KERNELS_INTERNAL_H

// Internals shared between kernels.c (the shipped kernels) and kernels_exp.c
// (techniques ported from the NAMBench SLIMMED-PATH study). Not part of the
// public interface.

#include <stdint.h>
#include <math.h>

#include "kernels.h"

#define Q14_SHIFT 14
#define Q14_ONE   (1 << Q14_SHIFT)
#define INV_Q14   (1.0f / 16384.0f)

// Staging geometry shared by the planner kernels: 15 taps per channel, packed
// two per word.
#define NAM_STAGE_STRIDE NAM_TAPS_MAX
#define NAM_PACK_STRIDE  ((NAM_TAPS_MAX + 1) / 2)

// Pack two Q14 taps as [lo | hi << 16] for SMLAD. Both halves are widened as
// unsigned so the value is the bit pattern we mean, independent of how the
// implementation converts a negative int16_t.
static inline int32_t pack2(int16_t lo, int16_t hi) {
    return (int32_t)((uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16));
}

#if defined(__ARM_FEATURE_DSP)
#define FPI_HAS_DSP 1
// (a*b)>>16 + (c*d)>>16, accumulated, in one instruction.
static inline void smlad_acc(int32_t *acc, int32_t a, int32_t b) {
    __asm__ volatile("smlad %0, %1, %2, %0" : "+r"(*acc) : "r"(a), "r"(b));
}
#else
#define FPI_HAS_DSP 0
#endif

static inline int16_t clamp_q15_v(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

// Q28 -> Q14 (rounded), LeakyReLU (slope 0.01), clamped.
// The negative branch is a divide by 100; GCC emits the M33 hardware divider,
// which measured faster than a reciprocal multiply (see docs/fpi/README.md).
static inline int16_t quantize_relu(int32_t acc) {
    int32_t a = (acc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT;
    if (a < 0) a = a / 100;
    return clamp_q15_v(a);
}

static inline int16_t input_to_q15(float x) {
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    long v = lrintf(x * 16384.0f);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

#endif
