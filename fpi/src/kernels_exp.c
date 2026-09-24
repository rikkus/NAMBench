// Experimental kernels for fpi: techniques ported from the NAMBench
// SLIMMED-PATH study on Apple silicon (see ../../SLIMMED-PATH.md).
//
// They are measured, not assumed, because the M2 study's reasoning does not
// transfer unchanged: it has 4-lane NEON and the M33 has 2-lane DSP, and several
// of its winners won *because* of wide SIMD or a large register file.
//
//   widetile     W frames per pass. M2 won 1.75x at W=32. What transfers is not
//                SIMD width but two latency effects: the weight pairs load once
//                per W frames instead of once per frame, and each output channel
//                gets W independent accumulator chains instead of one serial
//                chain. Both are latency effects, so M33 is a live question.
//   framemajor   A partial accumulator per (input channel, tap) pair, summed at
//                the end. M2: by far the worst result (0.61x) - more live state
//                than the register file holds, so it spilled. Kept as a
//                deliberate negative control; M33 has fewer registers still.
//
// kverify checks every kernel here against the float reference, so a wrong
// variant shows up as a validation failure rather than as a fast wrong answer.

#include "kernels.h"
#include "kernels_internal.h"
#include "hardware/xip_cache.h"

#include <string.h>

#define WT 4                        // frames per widetile pass
#define OPW NAM_PACK_STRIDE         // packed words per channel per frame

// Packed operands for the whole tile: [frame][channel][word].
// Static rather than stack: 4 * 3 * 8 * 4 = 384 bytes, and the stack is 2 KB.
static int32_t g_ops[WT][NAM_CHANNELS][OPW];

// Gather one frame's taps for one layer into slot `f`.
// `linear` selects the wrap rule: 0 = power-of-two ring (wrap at ring_size),
// 1 = linear ring (no wrap; the buffer is long enough by construction).
static inline void gather_frame_ex(nam_layer_t *ly, int idx, int f, int linear) {
    const int base = idx * NAM_CHANNELS;
    const int wrap = ly->ring_size * NAM_CHANNELS;
    #define WRAPOF(p) (linear ? (p) : (((p) < 0) ? (p) + wrap : (p)))

    if (ly->taps == 6) {
        for (int ic = 0; ic < NAM_CHANNELS; ic++) {
            const int16_t *src = ly->ring + ic;
            int32_t *dst = g_ops[f][ic];
            for (int t = 0; t < 6; t += 2) {
                int p0 = base + ly->off3[t];
                int p1 = base + ly->off3[t + 1];
                int16_t lo = src[WRAPOF(p0)];
                int16_t hi = src[WRAPOF(p1)];
                dst[t / 2] = pack2(lo, hi);
            }
        }
    } else {
        for (int ic = 0; ic < NAM_CHANNELS; ic++) {
            const int16_t *src = ly->ring + ic;
            int32_t *dst = g_ops[f][ic];
            int t = 0;
            for (; t + 1 < ly->taps; t += 2) {
                int p0 = base + ly->off3[t];
                int p1 = base + ly->off3[t + 1];
                int16_t lo = src[WRAPOF(p0)];
                int16_t hi = src[WRAPOF(p1)];
                dst[t / 2] = pack2(lo, hi);
            }
            if (t < ly->taps) {   // 15 taps: pair the last with a zero weight
                int p0 = base + ly->off3[t];
                dst[t / 2] = pack2(src[WRAPOF(p0)], 0);
            }
        }
    }
    #undef WRAPOF
}

static inline void gather_frame(nam_layer_t *ly, int idx, int f) {
    gather_frame_ex(ly, idx, f, 0);
}

static inline void gather_frame_lin(nam_layer_t *ly, int idx, int f) {
    gather_frame_ex(ly, idx, f, 1);
}

// MACs for the frames gathered into g_ops[0..w). Each frame's three output
// channels go to its own ring slot; the head reads the last layer back out of
// the ring, so nothing extra needs storing.
static inline void mac_frames(nam_layer_t *ly, const int idxs[WT], int w) {
    const int words = (ly->taps == 6) ? 3 : (ly->taps + 1) / 2;

    for (int oc = 0; oc < NAM_CHANNELS; oc++) {
        int32_t acc[WT];
        for (int f = 0; f < w; f++) acc[f] = ly->bias_q[oc];

        // The point of the experiment: the weight word is loaded once and feeds
        // w independent accumulators, so one chain's SMLAD latency cannot stall
        // the others.
        for (int ic = 0; ic < NAM_CHANNELS; ic++) {
            const int32_t *wt = ly->wtaps[oc][ic];
            for (int t = 0; t < words; t++) {
                int32_t wv = wt[t];
                for (int f = 0; f < w; f++) {
                    smlad_acc(&acc[f], g_ops[f][ic][t], wv);
                }
            }
        }

        for (int f = 0; f < w; f++) {
            ly->ring[idxs[f] * NAM_CHANNELS + oc] = quantize_relu(acc[f]);
        }
    }
}

void nam_process_widetile(nam_model_t *m, const float *in, float *out, int n) {
    int done = 0;
    while (done < n) {
        int w = n - done;
        if (w > WT) w = WT;

        int16_t x[WT];
        for (int f = 0; f < w; f++) x[f] = input_to_q15(in[done + f]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            int idxs[WT];

            // Frames are strictly sequential within a layer: frame f's taps can
            // only be read once frame f-1 has written its outputs to this ring,
            // so gather and MAC interleave frame by frame. The tile can batch the
            // MACs, not the gathers.
            for (int f = 0; f < w; f++) {
                int idx = ly->idx;
                idxs[f] = idx;
                ly->ring[idx * NAM_CHANNELS + 0] = x[f];
                ly->ring[idx * NAM_CHANNELS + 1] = 0;
                ly->ring[idx * NAM_CHANNELS + 2] = 0;
                gather_frame(ly, idx, f);
                ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;
            }

            mac_frames(ly, idxs, w);

            if (l == NAM_LAYERS - 1) {
                for (int f = 0; f < w; f++) {
                    int32_t hacc = 0;
                    for (int c = 0; c < NAM_CHANNELS; c++)
                        hacc += (int32_t)m->head_wq[c]
                              * (int32_t)ly->ring[idxs[f] * NAM_CHANNELS + c];
                    out[done + f] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT)
                                  / (float)Q14_ONE;
                }
            }
        }
        done += w;
    }
}

// ---------------------------------------------------------------------------
// framemajor: a partial accumulator per (input channel, tap), summed at the end.
// Reproduces the M2 study's worst result to check whether the same failure mode
// (spilling) appears here.
// ---------------------------------------------------------------------------
void nam_process_framemajor(nam_model_t *m, const float *in, float *out, int n) {
    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            const int taps = ly->taps;
            int idx = ly->idx;
            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            int16_t res[NAM_CHANNELS];
            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                int32_t partial[NAM_CHANNELS * NAM_TAPS_MAX];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    for (int t = 0; t < taps; t++) {
                        int p = idx + ly->off[t];
                        if (p < 0) p += ring;
                        int32_t v = (int32_t)ly->ring[p * NAM_CHANNELS + ic];
                        partial[ic * NAM_TAPS_MAX + t] = (int32_t)ly->w[oc][ic][t] * v;
                    }
                }
                int32_t acc = ly->bias_q[oc];
                for (int i = 0; i < NAM_CHANNELS * taps; i++) acc += partial[i];
                int16_t r = quantize_relu(acc);
                res[oc] = r;
                ly->ring[idx * NAM_CHANNELS + oc] = r;
            }
            ly->idx = (idx + 1 == ring) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Ablation: run the packed+SMLAD pipeline with the MAC loop compiled out, so the
// cost of the gather alone can be subtracted from the total.
//
// This exists because the obvious next optimisation depends on which half
// dominates: 1407 MACs at 6473 cycles is 4.6 cycles per MAC, which is far above
// what SMLAD throughput allows, so either the MACs are stalling on something or
// the gather is eating the budget. Guessing between those two has already cost
// time once in this project.
//
// The result is not a usable kernel - it computes garbage - so kverify must skip
// it. It is here to be measured, not shipped.
// ---------------------------------------------------------------------------
void nam_process_gatheronly(nam_model_t *m, const float *in, float *out, int n) {
    int16_t sink = 0;
    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);
        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            int idx = ly->idx;
            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            // Same gather as NAM_GATHER_PACK, then discard: the only work left is
            // addressing, the conditional loads and the pair packing.
            int32_t packed[NAM_CHANNELS * NAM_PACK_STRIDE];
            const int base = idx * NAM_CHANNELS;
            const int wrap = ly->ring_size * NAM_CHANNELS;
            for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                const int16_t *src = ly->ring + ic;
                int32_t *dst = packed + ic * NAM_PACK_STRIDE;
                int t = 0;
                for (; t + 1 < ly->taps; t += 2) {
                    int p0 = base + ly->off3[t];
                    int p1 = base + ly->off3[t + 1];
                    int16_t lo = src[(p0 < 0) ? p0 + wrap : p0];
                    int16_t hi = src[(p1 < 0) ? p1 + wrap : p1];
                    dst[t / 2] = pack2(lo, hi);
                }
            }
            sink ^= (int16_t)packed[0];
            ly->idx = (idx + 1 == ly->ring_size) ? 0 : idx + 1;
        }
        out[s] = (float)sink;
    }
}

// ---------------------------------------------------------------------------
// Ablation: the MAC loop without the gather. The operand pairs are read from a
// fixed staging slot so no ring addressing or wrap logic runs.
// ---------------------------------------------------------------------------
void nam_process_maconly(nam_model_t *m, const float *in, float *out, int n) {
    static int32_t ops[NAM_CHANNELS * NAM_PACK_STRIDE];
    int16_t x = input_to_q15(in[0]);
    for (int i = 0; i < NAM_CHANNELS * NAM_PACK_STRIDE; i++) ops[i] = pack2(x, x);

    for (int s = 0; s < n; s++) {
        int32_t total = 0;
        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int words = (ly->taps == 6) ? 3 : (ly->taps + 1) / 2;
            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                int32_t acc = ly->bias_q[oc];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const int32_t *s2 = ops + ic * NAM_PACK_STRIDE;
                    const int32_t *wt = ly->wtaps[oc][ic];
                    for (int t = 0; t < words; t++) smlad_acc(&acc, s2[t], wt[t]);
                }
                total += acc;
            }
        }
        out[s] = (float)(total & 0xffff);
    }
}

// ---------------------------------------------------------------------------
// The linear ring ("no wrap, contiguous reads") was implemented here and REMOVED
// after measuring. It loses by 3.3x, and the reason is structural rather than an
// implementation detail, so it is recorded rather than kept:
//
//   best measured 21,091 cycles/sample against 6,473 for the wrapped ring.
//
// The wrapped ring's per-sample cost is 6 bytes written and no copying at all;
// the linear ring must copy sum(lookback) * 3 channels * 2 bytes per rewind,
// amortised over the slack: ~5,483 bytes per sample at slack 256. The M2 study
// measured linear winning *against lazy mirroring*, which is per-tap and branchy
// - RP2350 has no mirror to remove, so the copy buys nothing.
//
// Two measurement mistakes on the way there are also worth keeping:
//   * slack = tile size gave layer 0 capacity 10 against lookback 5, so it
//     rewound almost every sample: 64,859 cycles, 10x worse. Size for
//     amortisation, not for the minimum.
//   * memmove() measured far worse than a hand-rolled copy loop here
//     (38,356 vs 21,091), so the first two attempts overstated the loss.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// All-float kernel: fp32 ring state, fp32 weights.
//
// Ported from the ESP32-S3 write-up (playtaurus.com), whose integer engine beat
// "a well-optimised float build" by only ~20% - the qualifier being the point.
// This project's earlier float kernel was not well-optimised for the reverse
// reason: it kept state in Q15 and converted int16 -> float on *every tap*, so
// it paid 1,407 conversions per sample for nothing. In fixed point the
// conversion only happens once per layer output.
//
// The M33 has a single-precision FPU, so a float tap can be a plain load plus
// VFMA.F32: one MAC per instruction, no packing, no Q-format bookkeeping. There
// are three times as many independent accumulator chains as taps here (3 output
// channels x 3 input channels if they are kept separate), which is enough to
// hide FMA latency without the register pressure that sank the widetile
// experiment.
//
// State lives in the model's fpool at the same ring sizes as the int16 pool.
// Weights are the real-valued wf[] already held per layer, so this kernel is
// also a check on what the Q14 rounding costs.
// ---------------------------------------------------------------------------
void nam_process_f32state(nam_model_t *m, const float *in, float *out, int n) {
    // Assign the float rings once per call; the layout mirrors the int16 pool.
    int off = 0;
    for (int l = 0; l < NAM_LAYERS; l++) {
        m->layer[l].fring = m->fpool + off * NAM_CHANNELS;
        off += nam_ring_size[l];
    }
    if (!m->fpool_ready) {
        memset(m->fpool, 0, sizeof(m->fpool));
        m->fpool_ready = true;
    }

    for (int s = 0; s < n; s++) {
        float x = in[s];

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            const int taps = ly->taps;
            const int base = ly->idx * NAM_CHANNELS;

            // layer 0's input channel 0 is the live sample; other layers read
            // their own ring. Writing the sample first keeps one code path.
            float *fr = ly->fring;
            fr[base + 0] = x;
            fr[base + 1] = 0.0f;
            fr[base + 2] = 0.0f;

            float res[NAM_CHANNELS];
            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                float acc = ly->bias_f[oc];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const float *w = ly->wf[oc][ic];
                    const float *src = fr + ic;
                    for (int t = 0; t < taps; t++) {
                        int p = base + ly->off[t] * NAM_CHANNELS;
                        if (p < 0) p += ring * NAM_CHANNELS;
                        acc += w[t] * src[p];
                    }
                }
                res[oc] = acc > 0.0f ? acc : acc * 0.01f;
                fr[base + oc] = res[oc];
            }
            ly->idx = (ly->idx + 1 == ring) ? 0 : ly->idx + 1;

            if (l == NAM_LAYERS - 1) {
                float o = 0.0f;
                for (int c = 0; c < NAM_CHANNELS; c++) o += m->head_w[c] * res[c];
                out[s] = o;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nam_process_packed_dup: the winning kernel with a *duplicated* ring.
//
// The motivation is the disassembly of nam_process_packed rather than a theory.
// Each 16-bit tap pair costs 13 instructions there:
//
//     cmp r2,#0 / it lt / addlt r2,r2,r3     wrap fixup, tap t
//     cmp r1,#0 / it lt / addlt r1,r1,r3     wrap fixup, tap t+1
//     mov.w r2,r2,lsl #1 / lsls r1,r1,#1     scale a sample index to bytes
//     ldrsh.w r0,[r9,r2] / ldrh.w r4,[r9,r1] the two taps
//     uxth r0,r0 / orr.w r0,r0,r4,lsl #16    pack to [lo | hi<<16]
//     str r0,[sp,#0]                         spill the pair for the MAC loop
//
// Six of the thirteen are the wrap fixup, and two more rescale an offset that
// was already pre-multiplied once (off3[] is in samples x channels; the compiler
// multiplies by two again for the byte address).
//
// Masking the wrap instead of branching was tried and abandoned: the
// interleaved layout needs `3*2^k - 1` to keep channel lanes aligned, and making
// that geometry work needs a power-of-two ring minimum, which takes the pool
// from 38 KB to 168 KB. Duplicating the ring needs no mask and no branch:
//
//   * the layer's ring occupies 2N samples, with N = ring_size; a byte-for-byte
//     mirror of the whole first half of the pool sits at NAM_POOL_STRIDE;
//   * every write goes to both copies, so ring[j] == ring[j + N] always holds;
//   * the read base is (idx + N), so tap t reads at base - t*d. For idx in
//     [0,N) and t*d in [0, N-1] that lands in [1, 2N-1]: always inside the
//     buffer, never needing a fixup, and always the right sample - the slot at
//     (idx + N - t*d) was last written at step (idx - t*d) mod N.
//
// So the gather keeps two loads, one pack, and an add of a pre-scaled byte
// offset, and loses the compare, the predicated add and the rescale.
//
// What it costs: six extra stores per layer per sample (three for the input
// injection and three for the result, each written twice), 38 KB more pool, and
// more memory traffic overall. That last one matters, because the two-core
// measurement says the shared memory port - not instruction fetch - is what
// taxes the second core by 11%, and this kernel deliberately trades instructions
// for traffic.
//
// Verified against the float path at six run lengths by kverify, like the rest.
// ---------------------------------------------------------------------------

// Per-tap read offsets in BYTES, relative to the read base. -2 * 3 * d * t:
// two bytes per sample, three channels interleaved, t dilations back.
static int32_t g_offb[NAM_LAYERS][NAM_TAPS_MAX];
static int     g_offb_ready;

static void dup_offsets(void) {
    if (g_offb_ready) return;
    for (int l = 0; l < NAM_LAYERS; l++)
        for (int t = 0; t < NAM_TAPS_MAX; t++)
            g_offb[l][t] = -2 * NAM_CHANNELS * nam_dilation[l] * t;
    g_offb_ready = 1;
}

// Write one channel of one slot to both copies of the ring.
static inline void dup_store(nam_layer_t *ly, int idx, int c, int16_t v) {
    ly->ring[idx * NAM_CHANNELS + c] = v;
    ly->ring[(idx + ly->ring_size) * NAM_CHANNELS + c] = v;
}

void nam_process_packed_dup(nam_model_t *m, const float *in, float *out, int n) {
    int32_t packed[NAM_CHANNELS * NAM_PACK_STRIDE];
    dup_offsets();

    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            const int idx = ly->idx;
            const int taps = ly->taps;
            int16_t res[NAM_CHANNELS];

            // Input sample and the two idle lanes, into both copies.
            dup_store(ly, idx, 0, x);
            dup_store(ly, idx, 1, 0);
            dup_store(ly, idx, 2, 0);

            // Branch-free gather. No tap offset can leave the buffer, so there
            // is nothing here to test and nothing to correct.
            {
                const char *base = (const char *)ly->ring
                                 + (size_t)(idx + ring) * NAM_CHANNELS * 2;
                const int32_t *ob = g_offb[l];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const char *src = base + ic * 2;
                    int32_t *dst = packed + ic * NAM_PACK_STRIDE;
                    const int np = (taps == 15) ? 7 : 3;
                    _Pragma("GCC unroll 8")
                    for (int t = 0; t < np; t++) {
                        dst[t] = pack2(*(const int16_t *)(src + ob[2 * t]),
                                       *(const int16_t *)(src + ob[2 * t + 1]));
                    }
                }
            }

            // Same MAC loop as the shipped packed kernel.
            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                int32_t acc = ly->bias_q[oc];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const int32_t *sp = packed + ic * NAM_PACK_STRIDE;
                    const int32_t *wt = ly->wtaps[oc][ic];
                    const int nw = (taps == 15) ? 7 : 3;
                    _Pragma("GCC unroll 8")
                    for (int t = 0; t < nw; t++) smlad_acc(&acc, sp[t], wt[t]);
                }
                int16_t r = quantize_relu(acc);
                res[oc] = r;
                dup_store(ly, idx, oc, r);
            }

            ly->idx = (idx + 1 == ring) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nam_process_packed_wxip: the SMLAD kernel with its weights in pinned XIP
// cache lines, i.e. cache-as-SRAM.
//
// The question this answers, from the datasheet rather than from hope
// (sections 4.2.1, 4.4.1, 4.4.1.3):
//
//   * "Cache lines can be individually pinned within the XIP address space for
//     use as SRAM, up to the total cache size of 16 kB."
//   * "Because the QMI only occupies the lower half of the 64 MB XIP address
//     space, you can pin cache lines outside of the QMI address range (e.g. at
//     the top of the XIP space) to avoid interfering with any QMI accesses."
//   * The cache is "16 kB, two-way set-associative, 1 cycle hit", built as two
//     8 kB banks interleaving odd and even lines, "which allows processors to
//     access multiple cache lines during the same cycle".
//
// A pinned line is *not faster* than SRAM: both are one cycle, and the core's
// single load/store unit issues one access per cycle either way. It is a
// different *port*, physically separate from the SRAM banks, and that is the
// only reason to try it. It follows from what this project measured: the kernel
// is bound by memory traffic rather than by instructions (the duplicated ring
// cut instructions 27% and lost 9.5%), and a second core running it loses 11% to
// the first - a memory-side effect, since moving the code into SRAM changed
// nothing at all.
//
// The weights are the right thing to move: 6,624 bytes, read-only after init,
// and read 621 times per sample, about a third of the kernel's memory
// operations. Pinning 828 of the cache's 1024 sets in way 0 leaves way 1 and the
// remaining 196 sets of way 0 - about 9.4 kB - for instruction fetch, and a miss
// now costs a QSPI access. Whether that trade wins is a measurement.
//
// Kept out of kernels.c on purpose: expressing the same idea by making the
// model's weight array a pointer moved packed+SMLAD from 6,451 to 6,567 and
// q15 SMLAD from 10,063 to 9,069 with no algorithmic change at all. The shipped
// table is this project's record, so this kernel carries its own pointers.
// ---------------------------------------------------------------------------

// XIP offset 0x02000000 is in the top half of the 26-bit downstream space, which
// the QMI does not decode. Its maintenance address (0x18000000 + offset + op) has
// bit 13 clear across the whole range, which is what confines pinning to way 0.
#define WXIP_OFFSET 0x02000000u
#define WXIP_WORDS  (NAM_CHANNELS * NAM_CHANNELS * NAM_PACK_STRIDE)

int nam_weights_in_xip_cache;

static const int32_t *g_wxip[NAM_LAYERS];   // per-layer packed weights
static int            g_wxip_ready;

static void wxip_setup(nam_model_t *m) {
    if (g_wxip_ready) return;

    const uint32_t bytes = NAM_LAYERS * WXIP_WORDS * 4;
    xip_cache_pin_range(WXIP_OFFSET, bytes);

    int32_t *base = (int32_t *)(XIP_BASE + WXIP_OFFSET);
    for (int l = 0; l < NAM_LAYERS; l++) {
        const int32_t *src = &m->layer[l].wtaps[0][0][0];
        for (int i = 0; i < WXIP_WORDS; i++) base[l * WXIP_WORDS + i] = src[i];
    }

    // Read back before trusting it. If the pin did not take, these writes went
    // downstream to an address with nothing behind it, and the kernel would
    // silently compute with whatever came back instead.
    int ok = 1;
    for (int l = 0; l < NAM_LAYERS && ok; l++) {
        const int32_t *src = &m->layer[l].wtaps[0][0][0];
        for (int i = 0; i < WXIP_WORDS; i++)
            if (base[l * WXIP_WORDS + i] != src[i]) { ok = 0; break; }
    }
    if (!ok) return;

    for (int l = 0; l < NAM_LAYERS; l++) g_wxip[l] = base + l * WXIP_WORDS;
    g_wxip_ready = 1;
    nam_weights_in_xip_cache = 1;
}

void nam_process_packed_wxip(nam_model_t *m, const float *in, float *out, int n) {
    int32_t packed[NAM_CHANNELS * NAM_PACK_STRIDE];
    wxip_setup(m);
    if (!g_wxip_ready) return;   // pin failed: produce nothing rather than garbage

    for (int s = 0; s < n; s++) {
        int16_t x = input_to_q15(in[s]);

        for (int l = 0; l < NAM_LAYERS; l++) {
            nam_layer_t *ly = &m->layer[l];
            const int ring = ly->ring_size;
            int idx = ly->idx;
            int16_t res[NAM_CHANNELS];

            ly->ring[idx * NAM_CHANNELS + 0] = x;
            ly->ring[idx * NAM_CHANNELS + 1] = 0;
            ly->ring[idx * NAM_CHANNELS + 2] = 0;

            {
                const int base = idx * NAM_CHANNELS;
                const int wrap = ring * NAM_CHANNELS;
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const int16_t *src = ly->ring + ic;
                    int32_t *dst = packed + ic * NAM_PACK_STRIDE;
                    const int np = (ly->taps == 15) ? 7 : 3;
                    _Pragma("GCC unroll 8")
                    for (int t = 0; t < np; t++) {
                        int p0 = base + ly->off3[2 * t];
                        int p1 = base + ly->off3[2 * t + 1];
                        dst[t] = pack2(src[(p0 < 0) ? p0 + wrap : p0],
                                       src[(p1 < 0) ? p1 + wrap : p1]);
                    }
                }
            }

            for (int oc = 0; oc < NAM_CHANNELS; oc++) {
                int32_t acc = ly->bias_q[oc];
                for (int ic = 0; ic < NAM_CHANNELS; ic++) {
                    const int32_t *sp = packed + ic * NAM_PACK_STRIDE;
                    const int32_t *wt = g_wxip[l] + (oc * NAM_CHANNELS + ic) * NAM_PACK_STRIDE;
                    const int nw = (ly->taps == 15) ? 7 : 3;
                    _Pragma("GCC unroll 8")
                    for (int t = 0; t < nw; t++) smlad_acc(&acc, sp[t], wt[t]);
                }
                int16_t r = quantize_relu(acc);
                res[oc] = r;
                ly->ring[idx * NAM_CHANNELS + oc] = r;
            }

            ly->idx = (idx + 1 == ring) ? 0 : idx + 1;

            if (l == NAM_LAYERS - 1) {
                int32_t hacc = 0;
                for (int c = 0; c < NAM_CHANNELS; c++)
                    hacc += (int32_t)m->head_wq[c] * (int32_t)res[c];
                out[s] = (float)((hacc + (1 << (Q14_SHIFT - 1))) >> Q14_SHIFT) / (float)Q14_ONE;
            }
        }
    }
}
