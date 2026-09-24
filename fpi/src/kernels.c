// Benchmark kernels in the actual NAM A2-Lite (3-channel) shape.
//
// Storage design, which is the whole story on a 520 KB part:
//
//   There is ONE ring layout — int16 Q15, interleaved by channel, one shared
//   pool — and every kernel uses it. There is no float ring. A float ring for
//   this shape is 23 layers x up to 7 KB x 4 B = ~500 KB on its own, which is
//   the entire SRAM of the RP2350A; with 2 MB of flash and 520 KB of SRAM,
//   the practical stack has to be fixed point. The float kernel therefore
//   keeps state in Q15 and computes in float, so the two are directly
//   comparable: same state, same rounding, different arithmetic.
//
//   Layer l's ring is only (k-1)*d + 1 samples, not the whole 3017-sample
//   receptive field — the big dilations are the ones that cost, and there are
//   only a few of them. 41676 samples total across the stack, ~250 KB.
//
// Weights are synthetic; per-sample cost is data-independent for this shape.

#include "kernels.h"
#include "kernels_internal.h"

#include <math.h>
#include <string.h>

#ifdef FPI_DUO
// __not_in_flash_func lives here. Note that "pico/platform.h" must not be
// included directly (it #errors); pico.h is the supported way in.
#include "pico.h"
#endif

// Measured: running these kernels from SRAM is SLOWER, not faster.
//
// The obvious reading of the datasheet (section 3.7.4.9.12: SRAM is single-cycle,
// XIP is not) suggests __not_in_flash_func. In practice it cost ~2.5%
// (packed+SMLAD 6477 -> 6636 cycles/sample), and aligning the entry points to 16
// bytes to dodge the bank-conflict phase made no difference.
//
// The reason is in that same section: contention is between instruction fetch and
// data load *to the same memory*. This kernel's data - ring buffers and weights -
// already lives in SRAM, so moving the code there puts both streams behind one
// port. Executing from flash instead keeps instruction fetch on the XIP path with
// its own 16 KB cache, leaving SRAM entirely to the data. Keeping this as a macro
// so the experiment is repeatable rather than folklore.
#define NAM_PROCESS



// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

int nam_receptive_field(void) {
    int rf = 1;
    for (int l = 0; l < NAM_LAYERS; l++) rf += (nam_kernel[l] - 1) * nam_dilation[l];
    return rf;
}

int nam_macs_per_sample(void) {
    int macs = 0;
    for (int l = 0; l < NAM_LAYERS; l++)
        macs += NAM_CHANNELS * NAM_CHANNELS * nam_kernel[l];
    macs += NAM_CHANNELS;  // head1x1
    return macs;
}

int nam_q15_uses_dsp(void) { return FPI_HAS_DSP; }

// ---------------------------------------------------------------------------
// Weight generation
// ---------------------------------------------------------------------------

static uint32_t rng_state;
static void rng_seed(uint32_t s) { rng_state = s ? s : 1u; }
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float rng_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng_next() >> 8) / 16777216.0f);
}

static inline int16_t f2q14(float x) {
    long v = lrintf(x * (float)Q14_ONE);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

// Pack two Q14 taps as [lo | hi << 16] for SMLAD. Both halves are widened as
// unsigned so the value is the bit pattern we mean, independent of how the

// MEASURED: do not read the ring size through this table in a hot loop.
//
// nam_ring_size[] is `static const`, so it lives in flash and every read of it
// is an XIP access through the cache the two cores share. The same number is
// already in SRAM as layer[l].ring_size, put there by nam_model_init, so the
// lookup buys nothing. The float kernel was doing it *inside* the (oc, ic) loops
// - nine XIP reads per layer per sample, shadowing an outer variable that
// already held the value - and the planned kernel once per layer per sample.
// The table itself is still the source of truth at init.
static inline int nam_ring_for_layer(int l) { return nam_ring_size[l]; }

#ifdef FPI_DUO
// Bank-phase probe for the two-core split experiment; see kernels.h. Only the
// duo target compiles this, so the shipped layout is bit-for-bit unchanged.
int nam_ring_pad_slots;
#endif

bool nam_model_init(nam_model_t *m) {
    memset(m, 0, sizeof(*m));

    // The pool is sized by NAM_RING_TOTAL; make sure that constant still agrees
    // with the per-layer table before handing out slices of it.
    int total = 0;
    for (int l = 0; l < NAM_LAYERS; l++) total += nam_ring_size[l];
    if (total != NAM_RING_TOTAL) return false;

    // Each layer gets 2 * ring_size samples: the ring itself, then a mirror of
    // it, so nam_process_packed_dup can read a window that never wraps. Every
    // other kernel uses only the first half and is unaffected.
    //
    // The doubling has to happen in this stride rather than by putting the mirror
    // at some pool-wide offset. The first attempt placed it at
    // ring + NAM_RING_TOTAL, which silently corrupted the *next* layer's ring:
    // layer l's ring and layer l+1's ring are adjacent in the pool, so a mirror
    // at a fixed global offset lands in the middle of somebody else's data.
    // kverify caught it - correct at 960 samples, wrong from 2016 on, which is
    // exactly where the larger rings first wrap.
    int offset = 0;
#ifdef FPI_DUO
    const int pad = nam_ring_pad_slots;
#else
    const int pad = 0;
#endif
    for (int l = 0; l < NAM_LAYERS; l++) {
#ifdef FPI_DUO
        if (l == NAM_RING_PAD_AFTER) offset += pad;
#endif
        m->layer[l].ring_size = nam_ring_size[l];
        m->layer[l].ring = &m->pool[offset * NAM_CHANNELS];
        offset += 2 * nam_ring_size[l];
    }
    if (offset != NAM_RING_TOTAL * 2 + pad) return false;

    rng_seed(0x5eed1234u);
    for (int l = 0; l < NAM_LAYERS; l++) {
        nam_layer_t *ly = &m->layer[l];
        ly->taps = nam_kernel[l];
        ly->dilation = nam_dilation[l];
        ly->idx = 0;
        for (int t = 0; t < ly->taps; t++) {
            ly->off[t] = -(t * ly->dilation);
            ly->off3[t] = ly->off[t] * NAM_CHANNELS;
        }

        for (int oc = 0; oc < NAM_CHANNELS; oc++) {
            ly->bias_q[oc] = 0;
            ly->bias_f[oc] = 0.0f;
            for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                for (int t = 0; t < ly->taps; t++) {
                    // Scaled by 1/sqrt(k) so a 15-tap layer does not simply
                    // blow up where a 6-tap one does not. Keeps the stack in
                    // range without needing a residual stream.
                    float w = rng_uniform(-0.5f, 0.5f) / sqrtf((float)ly->taps);
                    if (t == ly->taps / 2 && ic == oc) w += 1.0f;  // identity-ish
                    ly->wf[oc][ic][t] = w;
                    ly->w[oc][ic][t] = f2q14(w);
                }
                for (int p = 0; p < (NAM_TAPS_MAX + 1) / 2; p++) {
                    int16_t lo = ly->w[oc][ic][2 * p];
                    int16_t hi = (2 * p + 1 < NAM_TAPS_MAX) ? ly->w[oc][ic][2 * p + 1] : 0;
                    ly->wtaps[oc][ic][p] = pack2(lo, hi);
                }
            }
        }
    }

    for (int c = 0; c < NAM_CHANNELS; c++) {
        float w = (c == 0) ? 1.0f : 0.0f;
        m->head_w[c] = w;
        m->head_wq[c] = f2q14(w);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------


// Q28 -> Q14 (rounded), then LeakyReLU, then clamped to int16 range.
//
// The negative branch is a divide by 100 (the 0.01 slope). There is a USE_RECIP
// switch to replace it with a multiply by 655 and a shift, which is
// mathematically fine (655/65536 = 1/100 to within 1.5e-7).
//
// MEASURED: do not bother. USE_RECIP=1 gives 6534 cycles/sample against 6477 for
// the plain divide - 57 cycles *slower*, reproducible across rebuilds. The reason
// is that GCC already emits the M33 hardware divider (SDIV) for `/ 100`; that is
// a handful of cycles, so there is nothing to win, and the multiply-and-shift
// version lengthens the dependency chain instead. The switch stays so the
// experiment is repeatable, defaulting to the faster form.
#define USE_RECIP 0



// ---------------------------------------------------------------------------
// Kernel 0: float arithmetic over Q15 state
// ---------------------------------------------------------------------------

void NAM_PROCESS(nam_process_f32)(nam_model_t *m, const float *in, float *out, int n) {
    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            int idx = ly->idx;
            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            float res[NAM_CHANNELS];
            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                float acc = ly->bias_f[oc];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const int16_t *base = ly->ring + ic;
                    const float *wt = ly->wf[oc][ic];
                    // k is a runtime value, but it is one of two constants in
                    // practice, so let the compiler duplicate the loop for them
                    // rather than pay a divide per tap.
                    if (ly->taps == 6) {
                        for (int t = 0; t < 6; t++) {
                            int p = idx + ly->off[t];
                            if (p < 0) p += ring;
                            acc += wt[t] * (float)base[p * NAM_CHANNELS];
                        }
                    } else if (ly->taps == 15) {
                        for (int t = 0; t < 15; t++) {
                            int p = idx + ly->off[t];
                            if (p < 0) p += ring;
                            acc += wt[t] * (float)base[p * NAM_CHANNELS];
                        }
                    } else {
                        for (int t = 0; t < ly->taps; t++) {
                            int p = idx + ly->off[t];
                            if (p < 0) p += ring;
                            acc += wt[t] * (float)base[p * NAM_CHANNELS];
                        }
                    }
                }
                acc *= INV_Q14;  // state is Q14, weights are real
                res[oc] = acc > 0.0f ? acc : acc * 0.01f;
                ly->ring[idx * NAM_CHANNELS + oc] = clamp_q15_v((int32_t)lrintf(res[oc] * (float)Q14_ONE));
            }
            ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                float o = 0.0f;
                for (int c = 0; c < NAM_CHANNELS; c++) o += m->head_w[c] * res[c];
                out[s] = o;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: int16 Q14, with the M33 DSP extension where available.
//
// SMLAD retires two multiply-accumulates per instruction from a packed
// [lo | hi<<16] pair. The taps of one channel are `dilation` samples apart in
// the ring, so the pair has to be packed from two loads; that gather is the
// honest cost of using SMLAD here, and the nodsp variant exists to measure
// whether it pays.
// ---------------------------------------------------------------------------


static void NAM_PROCESS(nam_process_q15_impl)(nam_model_t *m, const float *in, float *out, int n, int use_dsp) {
    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            int idx = ly->idx;
            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            // Ring positions for this layer's taps, resolved once per sample,
            // and the layer's output channels. The mask wrap replaces the old
            // `p < 0 ? p + ring : p` branch; see the ring table comment in
            // kernels.h for why masking is equivalent.
            int pos[NAM_TAPS_MAX];
            int16_t res[NAM_CHANNELS];

            // k is a runtime field, but in practice it is always 6 or 15. The
            // compiler cannot unroll a runtime trip count or keep the ring
            // offsets in registers across it, so specialise on the two constants
            // the way the float path already does. This is worth a lot: the
            // taps are the only real work per sample.
#define NAM_Q15_LAYER_BODY(K)                                                 \
            do {                                                              \
                for (int t = 0; t < (K); t++) {                               \
                    int p = idx + ly->off[t];                                 \
                    if (p < 0) p += ring;                                     \
                    pos[t] = p;                                               \
                }                                                             \
                for (int oc = 0; oc < NAM_CHANNELS; oc++) {                   \
                    int32_t acc = ly->bias_q[oc];                             \
                    for (int ic = 0; ic < NAM_CHANNELS; ic++) {               \
                        const int16_t *base = ly->ring + ic;                  \
                        const int32_t *wt = ly->wtaps[oc][ic];                \
                        int t = 0;                                            \
                        if (use_dsp) {                                        \
                            for (; t + 1 < (K); t += 2) {                     \
                                int32_t pr = pack2(base[pos[t] * NAM_CHANNELS], \
                                                   base[pos[t + 1] * NAM_CHANNELS]); \
                                smlad_acc(&acc, pr, wt[t / 2]);               \
                            }                                                 \
                        }                                                     \
                        for (; t < (K); t++) {                                \
                            acc += (int32_t)ly->w[oc][ic][t]                  \
                                 * (int32_t)base[pos[t] * NAM_CHANNELS];      \
                        }                                                     \
                    }                                                         \
                    int16_t r = quantize_relu(acc);                           \
                    res[oc] = r;                                              \
                    ly->ring[idx * NAM_CHANNELS + oc] = r;                    \
                }                                                             \
            } while (0)

            if (ly->taps == 6) {
                NAM_Q15_LAYER_BODY(6);
            } else if (ly->taps == 15) {
                NAM_Q15_LAYER_BODY(15);
            } else {
                // Fallback for any other shape; same arithmetic, runtime count.
                for (int t = 0; t < ly->taps; t++) {
                    int p = idx + ly->off[t];
                    if (p < 0) p += ring;
                    pos[t] = p;
                }
                for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                    int32_t acc = ly->bias_q[oc];
                    for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                        const int16_t *base = ly->ring + ic;
                        for (int t = 0; t < ly->taps; t++) {
                            acc += (int32_t)ly->w[oc][ic][t]
                                 * (int32_t)base[pos[t] * NAM_CHANNELS];
                        }
                    }
                    int16_t r = quantize_relu(acc);
                    res[oc] = r;
                    ly->ring[idx * NAM_CHANNELS + oc] = r;
                }
            }
#undef NAM_Q15_LAYER_BODY

            ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}

void NAM_PROCESS(nam_process_q15)(nam_model_t *m, const float *in, float *out, int n) {
    nam_process_q15_impl(m, in, out, n, 1);
}

void NAM_PROCESS(nam_process_q15_nodsp)(nam_model_t *m, const float *in, float *out, int n) {
    nam_process_q15_impl(m, in, out, n, 0);
}

// ---------------------------------------------------------------------------
// Planner family.
//
// Staging lives on the stack: at most 3 channels x 15 taps = 45 int16, which is
// 90 bytes, comfortably inside the default stack and hot in cache for the whole
// layer sweep. The packed variant needs 3 x 8 = 24 int32 = 96 bytes.
//
// `off3[]` is pre-multiplied by NAM_CHANNELS so the gather is one add and one
// load per tap instead of a shift, an add and a load.
// ---------------------------------------------------------------------------

// Gather one layer's taps for all three input channels into planar staging.
// The ring offsets are already known to be in range, so the only branch here is
// the wrap fixup, which for most layers never triggers.
#define NAM_GATHER(K)                                                          \
    do {                                                                       \
        const int base = idx * NAM_CHANNELS;                                   \
        const int wrap = ring * NAM_CHANNELS;                                  \
        for (int ic = 0; ic < NAM_CHANNELS; ic++) {                            \
            const int16_t *src = ly->ring + ic;                                \
            int16_t *dst = stage + ic * NAM_STAGE_STRIDE;                      \
            for (int t = 0; t < (K); t++) {                                    \
                int p = base + ly->off3[t];                                    \
                dst[t] = src[(p < 0) ? p + wrap : p];                          \
            }                                                                  \
        }                                                                      \
    } while (0)

// Gather straight into packed [lo | hi<<16] pairs, so the MAC loop needs no
// packing at all: one 32-bit load per two taps. This is the whole point of the
// exercise — the DSP instruction is only worth having if its operands are
// already adjacent, and this is what makes them adjacent.
#define NAM_GATHER_PACK(K)                                                     \
    do {                                                                       \
        const int base = idx * NAM_CHANNELS;                                   \
        const int wrap = ring * NAM_CHANNELS;                                  \
        for (int ic = 0; ic < NAM_CHANNELS; ic++) {                            \
            const int16_t *src = ly->ring + ic;                                \
            int32_t *dst = packed + ic * NAM_PACK_STRIDE;                      \
            for (int t = 0; t + 1 < (K); t += 2) {                             \
                int p0 = base + ly->off3[t];                                   \
                int p1 = base + ly->off3[t + 1];                               \
                int16_t lo = src[(p0 < 0) ? p0 + wrap : p0];                   \
                int16_t hi = src[(p1 < 0) ? p1 + wrap : p1];                   \
                dst[t / 2] = pack2(lo, hi);                                    \
            }                                                                  \
        }                                                                      \
    } while (0)

// Scalar MACs from contiguous staging with compile-time constant offsets.
#define NAM_SCALAR_FROM_STAGE(K)                                               \
    do {                                                                       \
        for (int oc = 0; oc < NAM_CHANNELS; oc++) {                            \
            int32_t acc = ly->bias_q[oc];                                      \
            for (int ic = 0; ic < NAM_CHANNELS; ic++) {                        \
                const int16_t *s = stage + ic * NAM_STAGE_STRIDE;              \
                const int16_t *w = ly->w[oc][ic];                              \
                _Pragma("GCC unroll 15")                                       \
                for (int t = 0; t < (K); t++) {                                \
                    acc += (int32_t)w[t] * (int32_t)s[t];                      \
                }                                                              \
            }                                                                  \
            int16_t r = quantize_relu(acc);                                    \
            res[oc] = r;                                                       \
            ly->ring[idx * NAM_CHANNELS + oc] = r;                             \
        }                                                                      \
    } while (0)

// SMLAD over operands packed at use (staged, not pre-packed).
#define NAM_DSP_FROM_STAGE(K)                                                  \
    do {                                                                       \
        for (int oc = 0; oc < NAM_CHANNELS; oc++) {                            \
            int32_t acc = ly->bias_q[oc];                                      \
            for (int ic = 0; ic < NAM_CHANNELS; ic++) {                        \
                const int16_t *s = stage + ic * NAM_STAGE_STRIDE;              \
                const int32_t *wt = ly->wtaps[oc][ic];                         \
                _Pragma("GCC unroll 8")                                        \
                for (int t = 0; t + 1 < (K); t += 2) {                         \
                    smlad_acc(&acc, pack2(s[t], s[t + 1]), wt[t / 2]);         \
                }                                                              \
            }                                                                  \
            int16_t r = quantize_relu(acc);                                    \
            res[oc] = r;                                                       \
            ly->ring[idx * NAM_CHANNELS + oc] = r;                             \
        }                                                                      \
    } while (0)

// Two forms of the packed MAC loop, switchable so the experiment stays repeatable.
//
// USE_REUSE=1 caches the three channels of input words in locals and reuses them
// across output channels: 9 loads per layer instead of 27. In principle that also
// shortens each accumulator's dependency chain.
// USE_REUSE=0 loads per output channel in the straightforward way.
//
// Measured: the reuse form is neutral to slightly worse (6626 vs 6477
// cycles/sample). The saving in loads is cancelled by register pressure - the
// three arrays cannot stay live without spilling the accumulators and weight
// pointers. Kept switchable rather than deleted: the idea is sound and would pay
// on a core with more registers.
#define USE_REUSE 0

#if USE_REUSE
#define NAM_DSP_FROM_PACKED(K)                                                 \
    do {                                                                       \
        const int nw = (K) / 2;                                                \
        int32_t w0[NAM_PACK_STRIDE], w1[NAM_PACK_STRIDE], w2[NAM_PACK_STRIDE];\
        for (int t = 0; t < nw; t++) {                                         \
            w0[t] = packed[0 * NAM_PACK_STRIDE + t];                           \
            w1[t] = packed[1 * NAM_PACK_STRIDE + t];                           \
            w2[t] = packed[2 * NAM_PACK_STRIDE + t];                           \
        }                                                                      \
        for (int oc = 0; oc < NAM_CHANNELS; oc++) {                            \
            int32_t acc = ly->bias_q[oc];                                      \
            const int32_t *wt = ly->wtaps[oc][0];                              \
            _Pragma("GCC unroll 8")                                            \
            for (int t = 0; t < nw; t++) smlad_acc(&acc, w0[t], wt[t]);        \
            wt = ly->wtaps[oc][1];                                             \
            _Pragma("GCC unroll 8")                                            \
            for (int t = 0; t < nw; t++) smlad_acc(&acc, w1[t], wt[t]);        \
            wt = ly->wtaps[oc][2];                                             \
            _Pragma("GCC unroll 8")                                            \
            for (int t = 0; t < nw; t++) smlad_acc(&acc, w2[t], wt[t]);        \
            int16_t r = quantize_relu(acc);                                    \
            res[oc] = r;                                                       \
            ly->ring[idx * NAM_CHANNELS + oc] = r;                             \
        }                                                                      \
    } while (0)
#else
// Straightforward form: one pass per output channel over the three input channels.
#define NAM_DSP_FROM_PACKED(K)                                                 \
    do {                                                                       \
        for (int oc = 0; oc < NAM_CHANNELS; oc++) {                            \
            int32_t acc = ly->bias_q[oc];                                      \
            for (int ic = 0; ic < NAM_CHANNELS; ic++) {                        \
                const int32_t *s = packed + ic * NAM_PACK_STRIDE;              \
                const int32_t *wt = ly->wtaps[oc][ic];                         \
                _Pragma("GCC unroll 8")                                        \
                for (int t = 0; t < (K) / 2; t++) {                            \
                    smlad_acc(&acc, s[t], wt[t]);                              \
                }                                                              \
            }                                                                  \
            int16_t r = quantize_relu(acc);                                    \
            res[oc] = r;                                                       \
            ly->ring[idx * NAM_CHANNELS + oc] = r;                             \
        }                                                                      \
    } while (0)
#endif

typedef enum { STAGED_SCALAR, STAGED_DSP, PACKED_DSP } nam_plan_t;

// The planned kernel: gather each layer's taps into packed [lo | hi<<16] pairs,
// then SMLAD them with no packing at use. `plan` selects which of the three
// staging strategies runs.
//
// The layer loop is over the whole stack. The two-core split experiment needs a
// *range* of layers instead; that lives in its own function further down, next
// to a note on why it is not a parameter here.
static void NAM_PROCESS(nam_process_planned)(nam_model_t *m, const float *in, float *out, int n,
                                nam_plan_t plan) {
    int16_t stage[NAM_CHANNELS * NAM_STAGE_STRIDE];
    int32_t packed[NAM_CHANNELS * NAM_PACK_STRIDE];

    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            int idx = ly->idx;
            int16_t res[NAM_CHANNELS];

            // Input sample, and the two idle lanes.
            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            if (plan == PACKED_DSP) {
                if (ly->taps == 6)       { NAM_GATHER_PACK(6);  NAM_DSP_FROM_PACKED(6);  }
                else if (ly->taps == 15) { NAM_GATHER_PACK(15); NAM_DSP_FROM_PACKED(15); }
                else                     { NAM_GATHER_PACK(6);  NAM_DSP_FROM_PACKED(6);  }
            } else if (plan == STAGED_DSP) {
                if (ly->taps == 6)       { NAM_GATHER(6);  NAM_DSP_FROM_STAGE(6);  }
                else if (ly->taps == 15) { NAM_GATHER(15); NAM_DSP_FROM_STAGE(15); }
                else                     { NAM_GATHER(6);  NAM_DSP_FROM_STAGE(6);  }
            } else {
                if (ly->taps == 6)       { NAM_GATHER(6);  NAM_SCALAR_FROM_STAGE(6);  }
                else if (ly->taps == 15) { NAM_GATHER(15); NAM_SCALAR_FROM_STAGE(15); }
                else                     { NAM_GATHER(6);  NAM_SCALAR_FROM_STAGE(6);  }
            }

            ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}

void NAM_PROCESS(nam_process_staged)(nam_model_t *m, const float *in, float *out, int n) {
    nam_process_planned(m, in, out, n, STAGED_SCALAR);
}
void NAM_PROCESS(nam_process_staged_dsp)(nam_model_t *m, const float *in, float *out, int n) {
    nam_process_planned(m, in, out, n, STAGED_DSP);
}
void NAM_PROCESS(nam_process_packed)(nam_model_t *m, const float *in, float *out, int n) {
    nam_process_planned(m, in, out, n, PACKED_DSP);
}

#ifdef FPI_DUO
// The split experiment's entry point: the packed path over layers [l0, l1).
//
// This is deliberately a *copy* of the packed arm rather than a pair of extra
// parameters on nam_process_planned. It was tried as parameters first, and
// although packed+SMLAD itself was nearly unaffected (6460 against a published
// 6451), the shared body changed the codegen of the two staged arms - staged
// 8915 -> 9320, staged+SMLAD 7566 -> 7802 - purely because GCC clones the body
// per call site and the extra live parameters perturbed two of the three clones.
// Even as a separate function, merely *existing* moved packed+SMLAD to 6539 by
// adding one more call site to clone against. The shipped table is this
// project's measurement record and it should not move because an experiment
// exists nearby, so bench and kverify never see any of this.
//
// A range is interchangeable with the whole loop here because a layer only ever
// reads its own ring and the input sample is injected into every layer: nothing
// carries layer l's output into layer l+1, so [0,b) + [b,23) is bit-identical to
// [0,23) and the two halves touch disjoint rings. That is what makes a static
// two-core split verifiable rather than merely plausible, and it is also the one
// place this benchmark is easier than a real NAM, whose layers form a chain and
// whose split therefore has to be a pipeline.
//
// `out` is written only when the range contains the last layer, since that is
// where the head1x1 lives.
// ---------------------------------------------------------------------------
static void nam_planned_range_body(nam_model_t *m, const float *in, float *out, int n,
                                   int l0, int l1) {
    int32_t packed[NAM_CHANNELS * NAM_PACK_STRIDE];

    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);
        nam_layer_t *ly = &m->layer[l0];   // walked, so it is reset per sample

        for (int l = l0; l < l1; l++, ly++) {
            const int ring = ly->ring_size;
            int idx = ly->idx;
            int16_t res[NAM_CHANNELS];

            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            if (ly->taps == 15) { NAM_GATHER_PACK(15); NAM_DSP_FROM_PACKED(15); }
            else                { NAM_GATHER_PACK(6);  NAM_DSP_FROM_PACKED(6);  }

            ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}

void NAM_PROCESS(nam_process_packed_range)(nam_model_t *m, const float *in, float *out, int n,
                                           int l0, int l1) {
    nam_planned_range_body(m, in, out, n, l0, l1);
}

// The same arithmetic placed in RAM.
//
// Running the hot loop from SRAM measured ~2.5% SLOWER on one core (see the
// NAM_PROCESS comment at the top of this file): the data already sits in SRAM
// and the port is shared, so moving the code there only adds contention. With
// two cores that tradeoff is worth re-measuring rather than assuming, because
// the instruction stream is now fetched twice - and it still loses. The twin is
// kept so the duo target can A/B both in one boot.
void __not_in_flash_func(nam_process_packed_range_ram)(nam_model_t *m, const float *in, float *out,
                                                       int n, int l0, int l1) {
    nam_planned_range_body(m, in, out, n, l0, l1);
}
#endif
