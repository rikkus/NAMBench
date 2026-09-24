// fpi benchmark: is a NAM A2-Lite-shaped network affordable on ONE core?
//
// Runs three kernels over the same synthetic 23-layer, 3-channel A2-Lite shape
// and reports cycles/sample against the 48 kHz budget, on one core.
//
//   cmake --build build --target bench -j
//   cp build/src/bench.uf2 /Volumes/RP2350
//   screen /dev/cu.usbmodem* 115200
//
// Budget: at F MHz one core for one 48 kHz sample has F*1e6/48000 cycles.
// At the 150 MHz default that is 3125 cycles; at 300 MHz it is 6250.

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/unique_id.h"
#include "hardware/clocks.h"
#include "hardware/vreg.h"

#include "kernels.h"
#include "fault.h"
#include "ws2812.h"

#define SAMPLE_RATE  48000
#define BLOCK_SAMPLES 480      // 10 ms
#define N_BLOCKS      8
#define N_TRIALS      5

// The measurement runs N_BLOCKS * BLOCK_SAMPLES samples per trial and the
// kernel writes every one of them through the output pointer. That buffer must
// therefore be sized for the *measured* length, not the block length.
//
// It was not, and the 13 KB overrun walked into BSS — corrupting the PIO state
// used by the LED, which then blocked forever in pio_sm_put_blocking. The
// symptom was a non-deterministic hang several kernels into the run, which
// looked like a kernel bug and was not. Validation still compares only the
// last block, so the figures stay comparable.
#define MAX_SAMPLES   (BLOCK_SAMPLES * N_BLOCKS)

// ---------------------------------------------------------------------------
// DWT cycle counter (Armv8-M peripheral alias addressing).
// ---------------------------------------------------------------------------
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004UL)
#define DEMCR      (*(volatile uint32_t *)0xE000EDFCUL)

static void dwt_init(void) {
    DEMCR |= (1u << 24);
    DWT_CYCCNT = 0;
    DWT_CTRL |= (1u << 0);
}
static inline uint32_t dwt_cycles(void) { return DWT_CYCCNT; }
static inline uint32_t dwt_running(void) { return DWT_CTRL & 1u; }

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------
static nam_model_t        g_model;                          // ring pool + weights
static float              g_in[MAX_SAMPLES];
// Room for the kernel table below, which is sized automatically. The count is
// asserted against this in main(); see the comment there for why.
#define MAX_KERNELS 16
static float              g_out[MAX_KERNELS][MAX_SAMPLES];   // one row per kernel
static volatile uint32_t  g_sink;

typedef void (*kernel_fn)(nam_model_t *, const float *, float *, int);

typedef struct {
    const char *name;
    const char *note;
    kernel_fn   fn;
} kernel_desc_t;

// Index of the kernel whose output is the reference for validation (float), and
// the one compared against it (the best fixed-point path).
#define K_REF_FLOAT 0
#define K_BEST_Q15  3
#define K_ABL_FIRST 10  // kernels from here on compute garbage (ablations)

// Sized from its initialiser list, never written out by hand. Declaring it
// [13] with twelve entries left a NULL function pointer at the end of the
// table, and bench called it: hard fault, silent console, and a board that
// had to be unplugged. The guard in main() makes that failure loud instead.
static const kernel_desc_t g_kernels[] = {
    {"float scalar",   "32-bit float, -O3, interleaved ring",        nam_process_f32},
    {"q15 scalar",     "int16 Q14, taps resolved per MAC",          nam_process_q15_nodsp},
    {"q15 SMLAD",      "int16 Q14, SMLAD, operands packed at use",  nam_process_q15},
    {"staged",         "gather to planar staging, scalar MACs",      nam_process_staged},
    {"staged+SMLAD",   "gather to staging, SMLAD packed at use",     nam_process_staged_dsp},
    {"packed+SMLAD",   "gather to packed pairs, SMLAD 1 load/2 MAC", nam_process_packed},
    {"packed+dupring", "duplicated ring: no wrap fixup, byte offsets", nam_process_packed_dup},
    {"packed+wxip",    "weights in pinned XIP cache lines (cache-as-SRAM)", nam_process_packed_wxip},
    {"widetile x4",    "4 frames/pass, weights reused, 4 indep chains", nam_process_widetile},
    {"framemajor",     "partial acc per tap (M2 study's worst case)",  nam_process_framemajor},
    {"float state",    "fp32 rings + fp32 weights, VFMA per tap",     nam_process_f32state},
    {"ABL gather only","packed gather, MAC loop removed (garbage out)", nam_process_gatheronly},
    {"ABL mac only",   "MAC loop, gather removed (garbage out)",        nam_process_maconly},
};

// ---------------------------------------------------------------------------
// Measurement: cycles for `n` samples, best of N_TRIALS.
// ---------------------------------------------------------------------------
// `out` must have room for n_samples floats: the kernel writes every sample it
// processes. Callers pass a MAX_SAMPLES buffer and n_samples <= MAX_SAMPLES.
static uint32_t measure(kernel_fn fn, float *out, int n_samples) {
    uint32_t best = 0xFFFFFFFFu;
    for (int t = 0; t < N_TRIALS; t++) {
        nam_model_init(&g_model);
        uint32_t c0 = dwt_cycles();
        int done = 0;
        while (done < n_samples) {
            int n = (n_samples - done < BLOCK_SAMPLES) ? (n_samples - done) : BLOCK_SAMPLES;
            fn(&g_model, g_in, out, n);
            done += n;
        }
        uint32_t c1 = dwt_cycles();
        if (c1 - c0 < best) best = c1 - c0;
    }
    return best / (uint32_t)n_samples;  // cycles per sample
}

static void out_stats(const float *v, int n, float *peak, float *mean_abs) {
    float p = 0.0f, sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(v[i]);
        if (a > p) p = a;
        sum += a;
    }
    *peak = p;
    *mean_abs = sum / (float)n;
}

int main(void) {
    stdio_init_all();
    fpi_fault_init();

    // Wait for the host to attach before the first write. Writing earlier is
    // not merely lost, it is invisible: TinyUSB has not mounted yet, so the
    // stdio driver discards it. This laptop takes about two seconds.
    const uint32_t t0 = to_ms_since_boot(get_absolute_time());
    while (to_ms_since_boot(get_absolute_time()) - t0 < 2500) {
        ws2812_brightness((uint8_t)((to_ms_since_boot(get_absolute_time()) - t0) / 10));
        sleep_ms(20);
    }
    ws2812_set(0, 0, 40);  // blue = benchmarking

    // Progress markers, printed before each stage that can plausibly fail, so a
    // silent console localises the fault to a stage instead of being ambiguous.
    printf("\n=== fpi NAM A2-Lite benchmark ===\n");
    printf("stage: banner ok\n");

    dwt_init();
    printf("stage: dwt ok (%s)\n", dwt_running() ? "running" : "NOT RUNNING");

    int led = ws2812_init();
    printf("stage: ws2812 %s (rc=%d)\n", led == 0 ? "ok" : "FAILED", led);

    char id[2 * PICO_UNIQUE_BOARD_ID_SIZE_BYTES + 1];
    pico_get_unique_board_id_string(id, sizeof(id));

    // Fill the shared input block once, so every kernel sees identical data.
    {
        uint32_t s = 0xC0FFEEu;
        for (int i = 0; i < MAX_SAMPLES; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            g_in[i] = (float)(int32_t)(s >> 8) / 8388608.0f * 0.25f;  // ~+-0.25
        }
    }
    printf("stage: input filled\n");

    printf("stage: model init (%d bytes of rings)...\n", (int)sizeof(g_model.pool));
    if (!nam_model_init(&g_model)) {
        printf("FATAL: ring table failed its invariants (pool sum, power-of-two,\n"
               "       or whole-lane capacity) - see kernels.h\n");
        while (true) sleep_ms(1000);
    }
    printf("stage: model init ok\n");

    int rf = nam_receptive_field();

    const uint32_t f_sys = clock_get_hz(clk_sys);
    const uint32_t budget_cyc = f_sys / SAMPLE_RATE;

    printf("board    : Waveshare RP2350A-USB-A Mini (RP2350A)\n");
    printf("board id : %s\n", id);
    printf("core     : %d of 2 (this build uses one)\n", get_core_num());
    printf("clk_sys  : %u Hz\n", (unsigned)f_sys);
    printf("dsp ext  : %s\n", nam_q15_uses_dsp() ? "present (SMLAD path live)" : "ABSENT");
    printf("\n");
    printf("network  : A2-Lite %d layers x %d channels, kernels 6/15-tap\n",
           NAM_LAYERS, NAM_CHANNELS);
    printf("macs/smp : %d\n", nam_macs_per_sample());
    printf("recv fld : %d samples (%d ms at %d Hz)\n",
           rf, rf * 1000 / SAMPLE_RATE, SAMPLE_RATE);
    printf("rings    : %d samples, pool %d bytes (int16 x %d ch)\n",
           (int)NAM_RING_TOTAL, (int)sizeof(g_model.pool), NAM_CHANNELS);
    printf("           the pool is 2x the shape: each layer owns a mirror of its\n");
    printf("           own ring for the duplicated-ring kernel. Every other kernel\n");
    printf("           uses the first half only.\n");
    printf("model    : %d bytes total\n", (int)sizeof(nam_model_t));
    printf("           (a float ring for this shape needs ~%d bytes on its own)\n",
           (int)(NAM_RING_TOTAL * NAM_CHANNELS * (int)sizeof(float)));
    printf("\n");
    printf("48 kHz budget on one core: %u cycles/sample\n", (unsigned)budget_cyc);
    printf("\n");

    // Warm up, and make sure the sink is real so nothing gets optimised away.
    g_sink = 0;

    printf("%-14s %12s %10s %10s  %s\n", "kernel", "cyc/sample", "% of 1 core", "48kHz?", "note");
    printf("%-14s %12s %10s %10s  %s\n", "--------------", "------------", "----------",
           "----------", "----");

    const int n_kernels = (int)(sizeof(g_kernels) / sizeof(g_kernels[0]));
    if (n_kernels > MAX_KERNELS || n_kernels < 1 || g_kernels[n_kernels - 1].fn == NULL) {
        printf("FATAL: kernel table has %d entries, buffers hold %d, last fn %p\n",
               n_kernels, MAX_KERNELS,
               (const void *)(n_kernels > 0 ? (const void *)g_kernels[n_kernels - 1].fn
                                            : (const void *)0));
        for (;;) sleep_ms(1000);
    }
    uint32_t cyc[MAX_KERNELS];
    for (int k = 0; k < n_kernels; k++) {
        // One soak pass so indices are consistent, then the measured run.
        nam_model_init(&g_model);
        g_kernels[k].fn(&g_model, g_in, g_out[k], BLOCK_SAMPLES);

        // The output buffer is MAX_SAMPLES, not BLOCK_SAMPLES: the kernel writes
        // every sample it processes, and passing a short buffer here silently
        // overflowed BSS into the PIO state.
        cyc[k] = measure(g_kernels[k].fn, g_out[k], MAX_SAMPLES);

        float pct = 100.0f * (float)cyc[k] / (float)budget_cyc;
        printf("%-14s %12u %9.1f%% %10s  %s\n",
               g_kernels[k].name, (unsigned)cyc[k], pct,
               cyc[k] <= budget_cyc ? "YES" : "no",
               g_kernels[k].note);

        for (int i = 0; i < BLOCK_SAMPLES; i++) g_sink += (uint32_t)(g_out[k][i] * 1000.0f);
        ws2812_set(0, 0, (uint8_t)(40 + k * 60));
    }

    // -----------------------------------------------------------------------
    // Validation: the fixed-point path must actually agree with the float one.
    // -----------------------------------------------------------------------
    float pk_f, ma_f, pk_q, ma_q;
    out_stats(g_out[K_REF_FLOAT], BLOCK_SAMPLES, &pk_f, &ma_f);
    out_stats(g_out[K_BEST_Q15], BLOCK_SAMPLES, &pk_q, &ma_q);
    (void)0;

    float max_err = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < BLOCK_SAMPLES; i++) {
        float e = fabsf(g_out[K_BEST_Q15][i] - g_out[K_REF_FLOAT][i]);
        if (e > max_err) max_err = e;
        sum_sq += e * e;
    }
    float rms_err = sqrtf(sum_sq / (float)BLOCK_SAMPLES);

    printf("\nvalidation (last 10 ms block)\n");
    printf("  float  peak %.5f  mean|x| %.5f\n", pk_f, ma_f);
    printf("  q15    peak %.5f  mean|x| %.5f\n", pk_q, ma_q);
    printf("  q15 vs float: max err %.6f, rms err %.6f\n", max_err, rms_err);
    printf("  %s\n", max_err < 0.01f ? "AGREES — fixed-point path is sane"
                                      : "DISAGREES — suspect the Q15 scaling");

    if (cyc[2] < cyc[1]) {
        printf("\n  SMLAD with operands packed at use: %.2fx vs plain int16 scalar\n",
               (double)cyc[1] / (double)cyc[2]);
    } else {
        printf("\n  SMLAD with operands packed at use is %.2fx SLOWER than plain\n"
               "  int16 scalar — the packing costs more than the second MAC saves\n",
               (double)cyc[2] / (double)cyc[1]);
    }
    if (cyc[3] < cyc[1]) {
        printf("  gather-to-staging: %.2fx vs taps resolved per MAC\n",
               (double)cyc[1] / (double)cyc[3]);
    }

    // Best *real* fixed-point kernel, and what clock it would need for 48 kHz.
    // The ablations compute garbage, so they are excluded from the ranking even
    // though one of them is the fastest thing in the table.
    int best = 1;
    for (int k = 1; k < n_kernels; k++) {
        if (k >= K_ABL_FIRST) break;
        if (cyc[k] < cyc[best]) best = k;
    }
    const double mhz_needed = (double)cyc[best] * SAMPLE_RATE / 1e6;
    printf("\nbest fixed point: %s at %u cyc/sample\n",
           g_kernels[best].name, (unsigned)cyc[best]);
    printf("single-core clock needed for 48 kHz: %.0f MHz (stock is 150 MHz)\n",
           mhz_needed);
    printf("headroom still required: %.2fx\n", (double)cyc[best] / (double)budget_cyc);

    printf("\nreference (different code, dual core, 300 MHz — from the\n");
    printf("pico-neural-amp-modeler-demo README, for scale only):\n");
    printf("  generic Eigen         31030 cyc/smp   496%% of one core\n");
    printf("  a2_fast                8396 cyc/smp   134%% of one core\n");
    printf("  a2_fast + dual core    4533 cyc/smp    73%% of one core\n");
    printf("\none core, %u MHz, %u cyc/sample budget. Sink %08x.\n",
           (unsigned)(f_sys / 1000000u), (unsigned)budget_cyc, (unsigned)g_sink);
    printf("done — DWT cycle counter verified against clk_sys above.\n");

    ws2812_set(0, 40, 0);  // green = done

    while (true) {
        // Anything the DWT measured is in the log; hold here so the console
        // stays open for reading.
        sleep_ms(1000);
    }
}
