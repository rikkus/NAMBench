// duo: the fine-grained multi-boundary split, attempted.
//
// ---------------------------------------------------------------------------
// The question
// ---------------------------------------------------------------------------
// One core cannot afford this network at 150 MHz: the best kernel measures 6451
// cycles/sample against a 3125-cycle budget, i.e. 206% of one core. Two cores
// have 6250 cycles of budget between them, and 6451 > 6250. So "use both cores"
// is not automatically sufficient either - it is short by 3.2% *before* any
// split inefficiency. This target exists to find out whether that 3.2% is
// recoverable by splitting well, or whether it is simply gone.
//
// ---------------------------------------------------------------------------
// Why the split is a layer range
// ---------------------------------------------------------------------------
// The obvious serial structure to exploit is the layer chain: layer l needs
// layer l-1. Splitting at a boundary b gives core 0 the range [0,b) and core 1
// the range [b,23), and the two run concurrently - a pipeline, with period
// max(T_A, T_B) and no change in end-to-end latency (the first block still has
// to traverse every layer; see the latency note below).
//
// This benchmark's shape makes the split *easier* than a real NAM in one
// specific way, and it matters for reading the numbers: a layer here reads only
// its own ring, and the input sample is injected into every layer, so nothing
// carries layer l's output into layer l+1. The 23 layers are independent
// filters. That means [0,b) + [b,23) is bit-identical to [0,23) and the two
// halves touch disjoint rings, so the split can be *verified* rather than
// argued about (see the comparison against the single-core run at the end).
//
// For a real chained NAM the same ranges still give the same period, because a
// pipeline's steady-state period is max(T_A, T_B) whether or not the stages
// exchange data; what changes is that the handoff carries an activation block
// and must be double-buffered. So the measurement below is the right cost
// measurement, with one caveat recorded honestly: an unchained model could in
// principle also be split *across samples*, and a chained one cannot.
//
// ---------------------------------------------------------------------------
// Ruled out along the way (see docs/fpi/README.md for the numbers)
// ---------------------------------------------------------------------------
//   * instruction *location*: both cores in RAM measures the same as both in
//     flash, so moving the code is a wash on its own.
//   * SRAM bank phase: shifting one half of the pool by 0-3 interleaved slots
//     changes nothing (3672, 3672, 3673, 3673 cycles/sample).
//   * the two stacks sharing a bank: they do not. The SDK puts core 1's stack at
//     the top of SCRATCH_X (SRAM8) and core 0's at the top of SCRATCH_Y (SRAM9),
//     which are separate non-striped banks already.
//
// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------
// A two-stage pipeline does not add latency. Block 0's output appears at
// T_A + T_B = T_total, exactly as it does on one core; only the *period*
// between subsequent blocks changes, from T_total to max(T_A, T_B). What would
// add a block of latency is splitting across *blocks* (core 0 takes even blocks,
// core 1 odd), which the causal chain forbids anyway. The MODE_LAT measurement
// below checks this rather than asserting it.
//
// ---------------------------------------------------------------------------
// Build and run
// ---------------------------------------------------------------------------
//   cmake --build build --target duo -j
//   cp build/src/duo.uf2 /Volumes/RP2350
//   ./serial-probe.py 30

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"

#include "kernels.h"
#include "fault.h"
#include "ws2812.h"

#define SAMPLE_RATE   48000
#define BLOCK_SAMPLES 480          // 10 ms, the same block the bench target uses
#define N_BLOCKS      32           // enough that a one-block pipeline fill amortises
#define MAX_SAMPLES   (BLOCK_SAMPLES * N_BLOCKS)

// ---------------------------------------------------------------------------
// DWT cycle counter (Armv8-M peripheral alias addressing), as in bench.c.
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

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------
static nam_model_t g_model;
static float       g_in[MAX_SAMPLES];
static float       g_out_split[MAX_SAMPLES];   // written by whichever core owns layer 22
static float       g_out_ref[MAX_SAMPLES];     // single-core reference, same input
static volatile uint32_t g_sink;

// ---------------------------------------------------------------------------
// Core 1 protocol
//
// One persistent server loop, driven by a request counter, so every experiment
// below can be run from a single boot. The flags are ordinary SRAM words rather
// than the SIO FIFO because the FIFO is already carrying the SDK's own core1
// handshake and is not free for reuse; a word plus __sev/__wfe costs a handful
// of cycles and is measured as part of the result rather than assumed away.
// ---------------------------------------------------------------------------
enum { MODE_IDLE = 0, MODE_FLAT, MODE_PIPE, MODE_LAT };

static volatile uint32_t g_req;      // core0 -> core1: run id
static volatile uint32_t g_rel;      // core0 -> core1: blocks released (pipe)
static volatile uint32_t g_fin;      // core1 -> core0: blocks finished
static volatile uint32_t g_ack;      // core1 -> core0: run id completed
static volatile int      g_mode;
static volatile int      g_l0, g_l1, g_n, g_blocks, g_use_ram;
static volatile int      g_use_ram_c1;   // core 1's fetch path, independent of core 0's
static volatile uint32_t g_c1_ready;
// ---------------------------------------------------------------------------
// Core 1 stack placement.
//
// The SDK puts both cores' stacks in SRAM9: __StackOneTop is 0x20081000 and
// __StackBottom is 0x20081800, so they are two halves of the *same* 4 kB
// non-striped bank. SRAM8 (0x20080000) is empty in this build.
//
// That matters because the packed kernel's staging array lives on the stack:
// `int32_t packed[24]` is written 9 times and read 27 times per layer, which is
// 828 of the ~2000 memory operations each core performs per sample. Two cores
// doing that in the same physical bank, in step with each other, is exactly the
// shape of the 11% tax - and it is invisible to a probe that moves the ring
// pool, because the pool is not where that traffic goes.
//
// Mode 0 leaves the SDK's stack alone (the control), 1 moves core 1 to SRAM8 (a
// different non-striped bank), 2 moves it into the striped SRAM (shared with the
// ring pool). The switch happens at the top of each run, on core 1.
// ---------------------------------------------------------------------------


static inline void run_range(int l0, int l1, int n, int use_ram, float *out) {
#ifdef FPI_DUO
    if (use_ram) {
        nam_process_packed_range_ram(&g_model, g_in, out, n, l0, l1);
        return;
    }
#endif
    nam_process_packed_range(&g_model, g_in, out, n, l0, l1);
}

static void core1_serve(void) {
    __dmb();
    g_c1_ready = 1;
    __sev();
    uint32_t last = 0;
    for (;;) {
        while (g_req == last) __wfe();
        last = g_req;
        __dmb();

        const int mode = g_mode, n = g_n, l0 = g_l0, l1 = g_l1, nb = g_blocks;
        const int ram = g_use_ram_c1;

        if (mode == MODE_FLAT) {
            // Free-running: no per-block synchronisation at all. This is the
            // throughput ceiling of the split - both cores simply work through
            // their own range over every block.
            for (int b = 0; b < nb; b++)
                run_range(l0, l1, n, ram, g_out_split + (size_t)b * n);
            __dmb();
            g_fin = (uint32_t)nb;
            __sev();
        } else {
            // Per-block handoff, which is what a chained model needs. MODE_PIPE
            // keeps two blocks in flight so the stages actually overlap;
            // MODE_LAT releases one block and waits, to time the fill.
            for (int b = 0; b < nb; b++) {
                while (g_rel <= (uint32_t)b) __wfe();
                run_range(l0, l1, n, ram, g_out_split + (size_t)b * n);
                __dmb();
                g_fin = (uint32_t)(b + 1);
                __sev();
            }
        }

        __dmb();
        g_ack = last;
        __sev();
    }
}

// ---------------------------------------------------------------------------
// Single-core range measurement: cycles to process n_samples through [l0, l1).
// ---------------------------------------------------------------------------
#define N_TRIALS 5

// The whole model through the *plain* entry point, measured exactly the way
// bench.c measures it. This is here so the split numbers can be compared
// against bench's published 6451 rather than against a different harness: the
// range entry point takes l0/l1 as arguments, and a 4% discrepancy showed up
// between the two harnesses that had to be attributed before anything else in
// this file could be believed.
static uint32_t measure_plain(int n_samples) {
    uint32_t best = 0xFFFFFFFFu;
    for (int t = 0; t < N_TRIALS; t++) {
        nam_model_init(&g_model);
        uint32_t c0 = dwt_cycles();
        int done = 0;
        while (done < n_samples) {
            int n = (n_samples - done < BLOCK_SAMPLES) ? (n_samples - done) : BLOCK_SAMPLES;
            nam_process_packed(&g_model, g_in, g_out_ref, n);
            done += n;
        }
        uint32_t c1 = dwt_cycles();
        if (c1 - c0 < best) best = c1 - c0;
    }
    return best;
}

static uint32_t measure_range(int l0, int l1, int n_samples, int use_ram) {
    uint32_t best = 0xFFFFFFFFu;
    for (int t = 0; t < N_TRIALS; t++) {
        nam_model_init(&g_model);
        uint32_t c0 = dwt_cycles();
        int done = 0;
        while (done < n_samples) {
            int n = (n_samples - done < BLOCK_SAMPLES) ? (n_samples - done) : BLOCK_SAMPLES;
            run_range(l0, l1, n, use_ram, g_out_ref);
            done += n;
        }
        uint32_t c1 = dwt_cycles();
        if (c1 - c0 < best) best = c1 - c0;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Wall-clock time on core 0 for a whole dual-core run, in cycles.
//
// Returns the period for `blocks` blocks, i.e. N_BLOCKS * BLOCK_SAMPLES samples
// of wall clock. Gives back the first-block latency separately for MODE_PIPE.
// ---------------------------------------------------------------------------
static uint32_t dual_run2(int b, int mode, int use_ram, int use_ram_c1, int blocks,
                          uint32_t *first_cycles);

static uint32_t dual_run(int b, int mode, int use_ram, int blocks, uint32_t *first_cycles) {
    return dual_run2(b, mode, use_ram, use_ram, blocks, first_cycles);
}

static uint32_t dual_run2(int b, int mode, int use_ram, int use_ram_c1, int blocks,
                          uint32_t *first_cycles) {
    g_mode     = mode;
    g_l0       = b;
    g_l1       = NAM_LAYERS;
    g_n        = BLOCK_SAMPLES;
    g_blocks   = blocks;
    g_use_ram  = use_ram;
    g_use_ram_c1 = use_ram_c1;
    g_rel      = 0;
    g_fin      = 0;
    g_ack      = 0;

    nam_model_init(&g_model);
    __dmb();

    uint32_t t0 = dwt_cycles();
    g_req++;
    __dmb();
    __sev();

    if (mode == MODE_LAT) {
        // The realistic pipeline fill for a *chained* model: core 0 runs its
        // half of block 0, hands it over, and core 1 cannot start until it
        // arrives. The benchmark's layers are independent so core 1 does not
        // actually consume the handoff, but it still waits for it - which is
        // the conservative ordering, and the one a real NAM is stuck with.
        run_range(0, b, BLOCK_SAMPLES, use_ram, g_out_split);
        g_rel = 1;
        __dmb();
        __sev();
        while (g_fin < 1u) __wfe();
        uint32_t t1 = dwt_cycles();
        if (first_cycles) *first_cycles = t1 - t0;
        while (g_ack != g_req) __wfe();
        return t1 - t0;
    }

    for (int i = 0; i < blocks; i++) {
        // Keep two blocks in flight: A(i) may overlap B(i-1), but not B(i-2),
        // which is the double-buffered handoff a chained model would use. One
        // block in flight would serialise the stages and measure nothing.
        if (mode == MODE_PIPE && i >= 2) {
            while (g_fin < (uint32_t)(i - 1)) __wfe();
        }
        run_range(0, b, BLOCK_SAMPLES, use_ram, g_out_split + (size_t)i * BLOCK_SAMPLES);
        if (mode == MODE_PIPE) {
            g_rel = (uint32_t)(i + 1);
            __dmb();
            __sev();
        }
    }

    while (g_ack != g_req) __wfe();
    uint32_t t1 = dwt_cycles();
    if (first_cycles) *first_cycles = 0;
    return t1 - t0;
}

// ---------------------------------------------------------------------------
int main(void) {
    stdio_init_all();
    fpi_fault_init();

    const uint32_t t_boot = to_ms_since_boot(get_absolute_time());
    while (to_ms_since_boot(get_absolute_time()) - t_boot < 2500) {
        ws2812_brightness((uint8_t)((to_ms_since_boot(get_absolute_time()) - t_boot) / 10));
        sleep_ms(20);
    }
    ws2812_set(0, 0, 40);  // blue

    printf("\n=== fpi dual-core split ===\n");
    printf("stage: banner ok\n");

    dwt_init();
    int led = ws2812_init();
    printf("stage: dwt ok, ws2812 %s\n", led == 0 ? "ok" : "FAILED");

    {
        uint32_t s = 0xC0FFEEu;
        for (int i = 0; i < MAX_SAMPLES; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            g_in[i] = (float)(int32_t)(s >> 8) / 8388608.0f * 0.25f;
        }
    }
    if (!nam_model_init(&g_model)) {
        printf("FATAL: model init failed\n");
        while (true) sleep_ms(1000);
    }
    printf("stage: model ok, %d bytes\n", (int)sizeof(nam_model_t));

    g_c1_ready = 0;
    multicore_launch_core1(core1_serve);
    sleep_ms(50);
    printf("stage: core1 %s\n", g_c1_ready ? "running" : "NOT RUNNING");

    const uint32_t f_sys = clock_get_hz(clk_sys);
    const uint32_t budget = f_sys / SAMPLE_RATE;

    printf("\nclk_sys %u Hz, 48 kHz budget %.0f cycles/sample/core\n",
           (unsigned)f_sys, (double)budget);
    printf("two cores therefore have %.0f cycles/sample between them\n\n",
           2.0 * (double)budget);

    // -----------------------------------------------------------------------
    // Baseline: the whole model on one core, both entry points, and a check
    // that the range entry point agrees with the plain one bit for bit.
    // -----------------------------------------------------------------------
    nam_model_init(&g_model);
    nam_process_packed(&g_model, g_in, g_out_ref, BLOCK_SAMPLES);
    nam_model_init(&g_model);
    nam_process_packed_range(&g_model, g_in, g_out_split, BLOCK_SAMPLES, 0, NAM_LAYERS);
    int id_same = (memcmp(g_out_ref, g_out_split, BLOCK_SAMPLES * sizeof(float)) == 0);
    printf("range[0,23) == packed(whole): %s\n", id_same ? "bit-identical" : "MISMATCH");

    uint32_t whole_plain = measure_plain(MAX_SAMPLES);
    uint32_t whole = measure_range(0, NAM_LAYERS, MAX_SAMPLES, 0);
    printf("single core, whole model, plain entry point : %u cycles/sample (%.1f%%)\n",
           (unsigned)(whole_plain / MAX_SAMPLES),
           100.0 * (double)whole_plain / (double)MAX_SAMPLES / (double)budget);
    printf("single core, whole model, range entry point : %u cycles/sample (%.1f%%)\n\n",
           (unsigned)(whole / MAX_SAMPLES),
           100.0 * (double)whole / (double)MAX_SAMPLES / (double)budget);

    // -----------------------------------------------------------------------
    // Boundary sweep. T(0,b) and T(b,23) measured independently, which also
    // gives the per-layer cost as the marginal difference down the first column.
    // -----------------------------------------------------------------------
    printf("boundary sweep (one core, %d trials, %d samples each)\n", N_TRIALS, BLOCK_SAMPLES);
    printf("%3s %11s %11s %11s %9s %10s %10s\n",
           "b", "A[0,b)", "B[b,23)", "max(A,B)", "balance", "per-layer", "predicted");
    printf("%3s %11s %11s %11s %9s %10s %10s\n",
           "---", "-------", "-------", "--------", "-------", "---------", "---------");

    static uint32_t costA[NAM_LAYERS + 1], costB[NAM_LAYERS + 1];
    int best_b = 1;
    uint32_t best_max = 0xFFFFFFFFu;
    uint32_t prevA = 0;

    for (int b = 0; b <= NAM_LAYERS; b++) {
        uint32_t a = (b == 0) ? 0 : measure_range(0, b, BLOCK_SAMPLES, 0) / BLOCK_SAMPLES;
        uint32_t c = (b == NAM_LAYERS) ? 0 : measure_range(b, NAM_LAYERS, BLOCK_SAMPLES, 0) / BLOCK_SAMPLES;
        costA[b] = a;
        costB[b] = c;
        uint32_t mx = (a > c) ? a : c;
        if (mx < best_max) { best_max = mx; best_b = b; }
        double bal = (a + c) ? 100.0 * (double)((a > c) ? c : a) / (double)((a > c) ? a : c) : 0.0;
        printf("%3d %11u %11u %11u %8.1f%% %10u %9.1f%%\n",
               b, (unsigned)a, (unsigned)c, (unsigned)mx, bal,
               (unsigned)(a - prevA),
               100.0 * (double)mx / (double)budget);
        prevA = a;
    }

    printf("\nbest single boundary: b = %d, max half = %u cycles/sample (%.1f%% of one core)\n",
           best_b, (unsigned)best_max, 100.0 * (double)best_max / (double)budget);
    printf("a perfect split would be %u (%.1f%%) - the gap is split inefficiency,\n",
           (unsigned)(whole / MAX_SAMPLES / 2),
           100.0 * (double)(whole / MAX_SAMPLES / 2) / (double)budget);

    // -----------------------------------------------------------------------
    // Dual-core runs.
    //
    // The point of running several boundaries is not to find the best one - the
    // sweep already knows that - but to separate the two things that can cost:
    //
    //   balance      wall should be max(T_A, T_B), and b=13 makes those equal
    //   contention   wall should be max(T_A, T_B), and it is not
    //
    // If wall/max(T_A,T_B) comes out roughly constant across boundaries, then
    // the second core is being taxed at a fixed rate and balance is not the
    // problem. That distinction decides whether more splitting could ever help.
    // -----------------------------------------------------------------------
    printf("\ndual core, %d blocks x %d samples = %d samples per run\n",
           N_BLOCKS, BLOCK_SAMPLES, N_BLOCKS * BLOCK_SAMPLES);
    printf("%-24s %11s %11s %9s %9s %8s\n",
           "configuration", "wall cyc/smp", "max(A,B)", "wall/max", "% of RT", "speedup");
    printf("%-24s %11s %11s %9s %9s %8s\n",
           "------------------------", "-----------", "--------", "--------", "-------", "-------");

    const int probe_b[] = {0, 10, 11, 12, 13, 14, 15, 16, 23};
    const int n_probe = (int)(sizeof(probe_b) / sizeof(probe_b[0]));
    uint32_t dual_cyc[9];
    double ratio_sum = 0.0;
    int ratio_n = 0;

    for (int t = 0; t < n_probe; t++) {
        int b = probe_b[t];

        // Correctness first: the split output must equal the single-core one.
        nam_model_init(&g_model);
        nam_process_packed(&g_model, g_in, g_out_ref, BLOCK_SAMPLES);
        memset(g_out_split, 0, BLOCK_SAMPLES * sizeof(float));

        uint32_t cyc = dual_run(b, MODE_FLAT, 0, N_BLOCKS, NULL);
        uint32_t n_smp = N_BLOCKS * BLOCK_SAMPLES;
        uint32_t per = cyc / n_smp;
        dual_cyc[t] = per;

        int ok = 1;
        for (int i = 0; i < BLOCK_SAMPLES; i++)
            if (g_out_split[i] != g_out_ref[i]) { ok = 0; break; }
        if (ok && b > 0 && b < NAM_LAYERS) {
            // Only count the boundaries where both halves do real work.
            ratio_sum += (double)per / (double)costA[b];
            ratio_n++;
        }

        uint32_t mx = costA[b];
        double sp = (double)whole_plain / (double)MAX_SAMPLES / (double)per;
        printf("b=%-2d %-18s %11u %11u %8.2fx %8.1f%% %7.2fx%s\n",
               b, "flat flash", (unsigned)per, (unsigned)mx,
               mx ? (double)per / (double)mx : 0.0,
               100.0 * (double)per / (double)budget, sp,
               ok ? "" : "  MISMATCH");
    }

    // One RAM run, as the two-core version of the flash-vs-RAM question. The
    // single-core answer was "RAM is 2.5% slower"; this asks it again when the
    // instruction stream is fetched by two cores at once.
    {
        int b = best_b;
        uint32_t per = dual_run(b, MODE_FLAT, 1, N_BLOCKS, NULL) / (N_BLOCKS * BLOCK_SAMPLES);
        printf("b=%-2d %-18s %11u %11u %8.2fx %8.1f%% %7.2fx\n",
               b, "flat RAM", (unsigned)per, (unsigned)costA[b],
               (double)per / (double)costA[b],
               100.0 * (double)per / (double)budget,
               (double)whole_plain / (double)MAX_SAMPLES / (double)per);
    }

    // -----------------------------------------------------------------------
    // Bank-phase probe.
    //
    // The dominant remaining term is not the split - a perfect split is 104% of
    // real time - but the 11% the second core costs. That is a memory-side
    // effect (moving the code into SRAM changes nothing), and the datasheet
    // offers a specific mechanism for it: RP2350's main SRAM is 4-way striped on
    // address bits 3:2 over banks 0-3 in one 256 kB region and banks 4-7 in the
    // other, so "you can still achieve some explicit bandwidth partitioning by
    // allocating data across two 256 kB blocks". Two cores whose working sets
    // sit at the same bank phase and run the same code at the same rate would
    // collide in a bank on a large fraction of their overlapping accesses.
    //
    // Rather than re-place 76 kB of pool to test that, shift one half by a whole
    // number of interleaved slots: one slot is 3 int16 = 6 bytes, and
    // (6 >> 2) & 3 == 1, so every access in the shifted half moves to the next
    // bank while keeping the channel lanes aligned. Four values of `pad` cover
    // all four bank phases, and 0 is the shipped layout, which is the control.
    //
    // If banks are the mechanism, the tax moves with the phase. If it does not
    // move, they are not, and re-placing the pool would buy nothing.
    // -----------------------------------------------------------------------
    {
        printf("\nbank-phase probe: shift the pool from layer %d by n slots (6 bytes each)\n",
               NAM_RING_PAD_AFTER);
        printf("so the two halves differ in SRAM bank phase by n. b = %d throughout.\n\n",
               NAM_RING_PAD_AFTER);
        printf("%3s %11s %11s %11s %11s %9s %9s\n",
               "pad", "A[0,b)", "B[b,23)", "max", "wall", "wall/max", "% of RT");
        printf("%3s %11s %11s %11s %11s %9s %9s\n",
               "---", "-------", "--------", "---", "----", "--------", "-------");

        double tax_sum = 0.0;
        for (int pad = 0; pad <= 3; pad++) {
            nam_ring_pad_slots = pad;
            uint32_t a = measure_range(0, NAM_RING_PAD_AFTER, BLOCK_SAMPLES, 0) / BLOCK_SAMPLES;
            uint32_t c = measure_range(NAM_RING_PAD_AFTER, NAM_LAYERS, BLOCK_SAMPLES, 0) / BLOCK_SAMPLES;
            uint32_t mx = (a > c) ? a : c;
            uint32_t wall = dual_run(NAM_RING_PAD_AFTER, MODE_FLAT, 0, N_BLOCKS, NULL)
                            / (N_BLOCKS * BLOCK_SAMPLES);
            double tax = (double)wall / (double)mx;
            tax_sum += tax;
            printf("%3d %11u %11u %11u %11u %8.2fx %8.1f%%\n",
                   pad, (unsigned)a, (unsigned)c, (unsigned)mx, (unsigned)wall,
                   tax, 100.0 * (double)wall / (double)budget);
        }
        printf("\nmean wall/max over the four phases: %.3fx\n", tax_sum / 4.0);
        printf("a bank-conflict tax would move with the phase; the spread above is\n");
        printf("what this silicon actually does.\n");

        nam_ring_pad_slots = 0;
        nam_model_init(&g_model);
    }

    // Pipelined (per-block handoff), which is what a chained model is stuck
    // with. Same period is expected; the difference is the sync cost.
    {
        int b = best_b;
        uint32_t per = dual_run(b, MODE_PIPE, 0, N_BLOCKS, NULL) / (N_BLOCKS * BLOCK_SAMPLES);
        printf("b=%-2d %-18s %11u %11u %8.2fx %8.1f%% %7.2fx\n",
               b, "pipe flash", (unsigned)per, (unsigned)costA[b],
               (double)per / (double)costA[b],
               100.0 * (double)per / (double)budget,
               (double)whole_plain / (double)MAX_SAMPLES / (double)per);
    }
    {
        const double us_per_cyc = 1e6 / (double)f_sys;
        uint32_t fill_dual = 0, fill_single = 0;
        int b = best_b;
        dual_run(b, MODE_LAT, 0, 1, &fill_dual);

        nam_model_init(&g_model);
        uint32_t c0 = dwt_cycles();
        run_range(0, NAM_LAYERS, BLOCK_SAMPLES, 0, g_out_ref);
        uint32_t c1 = dwt_cycles();
        fill_single = c1 - c0;

        printf("\npipeline fill / first result, one block of %d samples:\n", BLOCK_SAMPLES);
        printf("  one core, whole model : %8u cycles  (%.3f ms)\n",
               (unsigned)fill_single, (double)fill_single * us_per_cyc / 1000.0);
        printf("  two cores, split b=%d : %8u cycles  (%.3f ms)\n",
               b, (unsigned)fill_dual, (double)fill_dual * us_per_cyc / 1000.0);
        printf("  difference            : %+8d cycles (%.1f%%)\n",
               (int)fill_dual - (int)fill_single,
               100.0 * ((double)fill_dual - (double)fill_single) / (double)fill_single);
        printf("  both are T_total: the first block traverses every layer either\n");
        printf("  way, so the pipeline changes the period, not the latency.\n");
    }

    // -----------------------------------------------------------------------
    // Split fetch paths.
    //
    // The mem2 microbenchmark (src/mem2.c) measured where the two-core penalty
    // actually comes from. Same bank group, different bank groups and literally
    // the same words all give the same answer, so it is not the data. What does
    // change it is where the *loop* lives: with the loop executing from flash
    // through the shared 16 kB XIP cache, the second core costs up to 1.26x, and
    // with the same loop copied into RAM it costs 1.00-1.04x.
    //
    // That is a shared instruction-fetch path, and it explains a result this
    // file recorded earlier without understanding it: both cores in RAM measured
    // exactly the same as both cores in flash, i.e. moving the code is a wash.
    // The wash makes sense if the two configurations trade one contention for
    // another - flash code contends on the XIP port, RAM code contends with data
    // on the SRAM port.
    //
    // Which suggests something neither configuration tries: give the two cores
    // *different* fetch paths. Core 0 keeps executing from flash, core 1 runs the
    // RAM copy, and neither is starved of fetch while only one of them adds to
    // the SRAM data traffic.
    // -----------------------------------------------------------------------
    {
        static const char *names[4] = {
            "both from flash",
            "both from RAM",
            "core0 flash, core1 RAM",
            "core0 RAM, core1 flash",
        };
        printf("\nfetch-path split probe (b = %d, flat split)\n", best_b);
        printf("%-26s %11s %11s %9s\n", "fetch paths", "wall", "max(A,B)", "wall/max");
        printf("%-26s %11s %11s %9s\n", "--------------------------",
               "----", "--------", "--------");
        for (int cfg = 0; cfg < 4; cfg++) {
            int r0 = (cfg == 1 || cfg == 3);
            int r1 = (cfg == 1 || cfg == 2);
            uint32_t wall = dual_run2(best_b, MODE_FLAT, r0, r1, N_BLOCKS, NULL)
                            / (N_BLOCKS * BLOCK_SAMPLES);
            printf("%-26s %11u %11u %8.2fx\n", names[cfg], (unsigned)wall,
                   (unsigned)costA[best_b], (double)wall / (double)costA[best_b]);
        }
        nam_model_init(&g_model);
    }

    // -----------------------------------------------------------------------
    // The decomposition that decides the verdict. Four numbers, each one the
    // best case given the one above it.
    // -----------------------------------------------------------------------
    {
        uint32_t per_plain = whole_plain / MAX_SAMPLES;
        uint32_t ideal     = per_plain / 2;
        uint32_t balanced  = costA[best_b];
        uint32_t measured  = dual_cyc[1 + 3];  // probe_b[] index of b = 13

        printf("\ndecomposition (cycles/sample of wall clock; budget is %u)\n",
               (unsigned)budget);
        printf("  one core, whole model          %6u   %6.1f%%   the kernel as shipped\n",
               (unsigned)per_plain, 100.0 * (double)per_plain / (double)budget);
        printf("  two cores, perfectly balanced  %6u   %6.1f%%   total work / 2 - a\n",
               (unsigned)ideal, 100.0 * (double)ideal / (double)budget);
        printf("  %-30s %6u   %6.1f%%   best boundary this\n",
               "two cores, best real boundary", (unsigned)balanced,
               100.0 * (double)balanced / (double)budget);
        printf("  %-30s %6u   %6.1f%%   shape allows, measured\n",
               "two cores, measured", (unsigned)measured,
               100.0 * (double)measured / (double)budget);
        printf("\n  the first two lines are arithmetic and cannot be split away:\n");
        printf("  half of the *whole* model is already %.1f%% of realtime, so no\n",
               100.0 * (double)ideal / (double)budget);
        printf("  division of the work can reach 100%% until the kernel itself is\n");
        printf("  faster. What the split adds on top is contention: wall/max(A,B)\n");
        printf("  averaged %.2fx over the boundaries where both halves work.\n",
               ratio_n ? ratio_sum / (double)ratio_n : 0.0);
    }

    g_sink = 0;
    for (int i = 0; i < BLOCK_SAMPLES; i++) g_sink += (uint32_t)(g_out_ref[i] * 1000.0f);
    printf("\nsink %08x. done.\n", (unsigned)g_sink);

    ws2812_set(0, 40, 0);  // green
    while (true) sleep_ms(1000);
}
