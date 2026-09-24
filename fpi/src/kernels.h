#ifndef FPI_KERNELS_H
#define FPI_KERNELS_H

#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// The actual NAM A2-Lite (3-channel) shape, read out of the reference
// project's example.nam rather than guessed:
//
//   23 dilated-conv layers, 3 channels
//   kernel_sizes = [6 x14, 15, 15, 6 x7]
//   dilations    = [1, 3, 7, 17, 41, 101, 239] repeating (23 entries)
//   activation   = LeakyReLU, negative_slope 0.01
//
//   receptive field = 1 + sum((k-1) * d) = 3017 samples = 62.8 ms @ 48 kHz
//
// The two 15-tap layers are the layers where the dilation wraps back to 1,
// which is what keeps the receptive field sane. The full 8-channel A2 has the
// same dilations and 8 channels instead of 3.
//
// Scope: this is the compute core of the network, not a faithful NAM port.
// No input mixin, no FiLM conditioning (A2-Lite has none active), and no
// residual stream. Weights are synthetic and deterministic; per-sample cost is
// data-independent for this shape, so the cycle counts are still meaningful.
// ---------------------------------------------------------------------------

#define NAM_LAYERS   23
#define NAM_CHANNELS 3
#define NAM_TAPS_MAX 15

// --- Shape, from example.nam ----------------------------------------------
static const int nam_dilation[NAM_LAYERS] = {
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239
};
static const int nam_kernel[NAM_LAYERS] = {
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6
};

// Ring capacity per layer. A layer only needs the taps it actually reads, so its
// SRAM cost is (k-1)*d + 1 samples, not the whole receptive field. Only the few
// large-dilation layers are expensive.
//
// Pairing kernel sizes with dilations:
//
//   layer  0..6 : d = 1,3,7,17,41,101,239  k = 6   -> 6,16,36,86,206,506,1196
//   layer  7..13: same again                       -> 6,16,36,86,206,506,1196
//   layer 14,15 : d = 1,13                 k = 15  -> 15,183
//   layer 16..22: d = 1,3,7,17,41,101,239  k = 6   -> 6,16,36,86,206,506,1196
//
// Total 6354 samples = 38124 bytes in int16 x 3 channels. A float equivalent
// would be 76248 bytes, which also fits, so the fixed-point choice is about
// speed rather than capacity.
//
// An attempt to replace the wrap branch with a mask arithmetic (`p & (2^k - 1)`)
// is documented in docs/fpi/README.md and was abandoned: see the ring-wrap note.
static const int nam_ring_size[NAM_LAYERS] = {
    6, 16, 36, 86, 206, 506, 1196, 6, 16, 36, 86, 206, 506, 1196,
    15, 183, 6, 16, 36, 86, 206, 506, 1196
};
#define NAM_RING_TOTAL 6354

typedef struct {
    int16_t  w[NAM_CHANNELS][NAM_CHANNELS][NAM_TAPS_MAX];   // [out][in][tap] Q14
    // The same taps pre-packed two at a time as [lo | hi<<16] for SMLAD.
    // Declared int32_t so every pair is word-aligned: the assembler is
    // entitled to assume that, and a char-array alias would not guarantee it.
    int32_t  wtaps[NAM_CHANNELS][NAM_CHANNELS][(NAM_TAPS_MAX + 1) / 2];
    float    wf[NAM_CHANNELS][NAM_CHANNELS][NAM_TAPS_MAX];  // same, float path
    int32_t  bias_q[NAM_CHANNELS];                          // Q14
    float    bias_f[NAM_CHANNELS];
    int      taps;
    int      dilation;
    int      idx;                    // ring write index
    int      off[NAM_TAPS_MAX];      // per-tap read offsets (negative, samples)
    int      off3[NAM_TAPS_MAX];     // the same, pre-multiplied by NAM_CHANNELS
    int      ring_size;              // nam_ring_size[layer]
    int16_t *ring;                   // nam_ring_size[layer] * NAM_CHANNELS, in pool
    float   *fring;                  // fp32 equivalent, in fpool (f32state kernel)
} nam_layer_t;

// A float ring pool alongside the int16 one, for the all-float kernel ported
// from the ESP32-S3 lesson (see kernels_exp.c). Every other kernel keeps using
// the int16 pool untouched, so the two can be measured against each other
// without either disturbing the other's state.
// One ring's worth of samples for the whole stack, and the pool that holds them.
//
// The pool is *twice* that, because nam_process_packed_dup mirrors every ring
// into the second half so that no tap ever needs a wrap fixup. Layer l's ring is
// the first-half slice nam_model_init assigns; its mirror is the same offset plus
// NAM_POOL_STRIDE. Every other kernel just ignores the second half.
#define NAM_POOL_STRIDE (NAM_RING_TOTAL * NAM_CHANNELS)

typedef struct {
    nam_layer_t layer[NAM_LAYERS];
    float       head_w[NAM_CHANNELS];   // head1x1: 3 -> 1
    int16_t     head_wq[NAM_CHANNELS];
#ifdef FPI_DUO
    // The duo target tests whether the two cores collide in SRAM *banks* by
    // shifting one half of the pool by a whole number of slots. RP2350 SRAM is
    // striped on address bits 3:2, so a 6-byte shift (one interleaved slot of
    // three int16) moves every access in that half to the next bank. The extra
    // storage is 3 pad slots, and only this target pays for it.
    int16_t     pool[NAM_POOL_STRIDE * 2 + 3 * 3];
#else
    int16_t     pool[NAM_POOL_STRIDE * 2];
#endif
    // Float state: same ring sizes, but fp32 and converted to real weights
    // rather than Q14. 2x the int16 pool.
    float       fpool[NAM_RING_TOTAL * NAM_CHANNELS];
    bool        fpool_ready;         // fpool starts zeroed but is not reset per run
} nam_model_t;

// Receptive field in samples (3017 for A2-Lite).
int nam_receptive_field(void);

// Multiply-accumulates per sample across the dilated stack and the head.
int nam_macs_per_sample(void);

// True when the build has the Cortex-M33 DSP extension (__ARM_FEATURE_DSP),
// which the SMLAD path needs. The SDK's RP2350 toolchain sets
// -march=armv8-m.main+fp+dsp, so this should be 1.
int nam_q15_uses_dsp(void);

// Fill synthetic weights, reset the rings, and verify NAM_RING_TOTAL against
// nam_ring_size[]. Returns false if the pool size does not match the shape.
bool nam_model_init(nam_model_t *m);

// Process one block, mono float in/out.
void nam_process_f32(nam_model_t *m, const float *in, float *out, int n);

// Q15 fixed point. nam_process_q15 uses the SMLAD path where the build has it;
// nam_process_q15_nodsp is the same arithmetic without the DSP instructions,
// which is the A/B of what the extension actually buys.
void nam_process_q15(nam_model_t *m, const float *in, float *out, int n);
void nam_process_q15_nodsp(nam_model_t *m, const float *in, float *out, int n);

// ---------------------------------------------------------------------------
// Planner family.
//
// The cost in the naive kernels is not the multiply-accumulate, it is the
// addressing around it: every tap recomputes `p = idx + off[t]`, compares it
// against zero, fixes up the wrap, then scales by the channel count to reach a
// lane inside the interleaved ring. That is 4-6 instructions of overhead per
// MAC, and with only 1407 MACs per sample against a 3125-cycle budget there is
// no room for it.
//
// Instead, gather each layer's taps once per sample into a small contiguous
// planar staging buffer, then run the MACs against that with compile-time
// constant offsets and a fully unrolled loop. Three variants, so the benefit of
// each idea can be attributed:
//
//   nam_process_staged      gather only, scalar MACs
//   nam_process_staged_dsp  gather only, SMLAD over operands packed at use
//   nam_process_packed      gather directly into packed pairs, SMLAD with a
//                           single 32-bit load and no packing at all
//
// All three use the same Q14 arithmetic and state as nam_process_q15, so their
// outputs are directly comparable and validated against the float path.
void nam_process_staged(nam_model_t *m, const float *in, float *out, int n);
void nam_process_staged_dsp(nam_model_t *m, const float *in, float *out, int n);
void nam_process_packed(nam_model_t *m, const float *in, float *out, int n);

#ifdef FPI_DUO
// The same packed kernel restricted to layers [l0, l1), for the two-core split
// experiment. `out` is only written when the range contains the last layer,
// since that is where the head1x1 lives. Only compiled into the duo target;
// see the note in kernels.c on why bench and kverify do not get these.
void nam_process_packed_range(nam_model_t *m, const float *in, float *out, int n,
                              int l0, int l1);

// Identical arithmetic, placed in RAM, so the flash-vs-RAM question can be
// re-asked under two-core instruction-fetch contention.
void nam_process_packed_range_ram(nam_model_t *m, const float *in, float *out, int n,
                                  int l0, int l1);
#endif

// ---------------------------------------------------------------------------
// Experimental kernels, ported from the NAMBench SLIMMED-PATH study's techniques
// (see ../../SLIMMED-PATH.md). Same state and arithmetic as the kernels above, so
// their outputs are directly comparable; kverify checks them all.
//
//   widetile    W frames per pass: weight words load once per W frames, and each
//               output channel gets W independent accumulator chains.
//   framemajor  A partial accumulator per (input channel, tap). Deliberately
//               reproduces the study's worst case as a negative control.
// ---------------------------------------------------------------------------
void nam_process_widetile(nam_model_t *m, const float *in, float *out, int n);
void nam_process_framemajor(nam_model_t *m, const float *in, float *out, int n);

// Ablations, for attribution only - both compute garbage, so kverify skips them.
//   gatheronly  the packed gather with the MAC loop removed
//   maconly     the MAC loop with the gather removed
void nam_process_gatheronly(nam_model_t *m, const float *in, float *out, int n);
void nam_process_maconly(nam_model_t *m, const float *in, float *out, int n);

// All-float kernel: fp32 ring state and fp32 weights, so a tap is a plain load
// and a VFMA with no format conversion. This is the ESP32-S3 article's lesson
// applied - their integer engine was ~20% faster than "a well-optimised float
// build", and the point is the qualifier.
void nam_process_f32state(nam_model_t *m, const float *in, float *out, int n);

// The SMLAD kernel with a *duplicated* ring, so the wrap fixup disappears
// entirely rather than being replaced by a mask. See kernels_exp.c.
void nam_process_packed_dup(nam_model_t *m, const float *in, float *out, int n);

// The SMLAD kernel with its weight tables in pinned XIP cache lines
// (cache-as-SRAM), so weight loads leave the SRAM banks. See kernels_exp.c.
void nam_process_packed_wxip(nam_model_t *m, const float *in, float *out, int n);

// 1 once the weights above are actually resident in pinned cache lines.
extern int nam_weights_in_xip_cache;

#ifdef FPI_DUO
// Bank-phase offset, in interleaved slots, applied to the pool from
// NAM_RING_PAD_AFTER onwards. Set before nam_model_init(); 0 is the shipped
// layout. Only the duo target compiles this.
#define NAM_RING_PAD_AFTER 13
extern int nam_ring_pad_slots;
#endif

#endif
