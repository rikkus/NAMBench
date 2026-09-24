// Kernel verification harness for fpi.
//
// Runs every kernel over the same input and prints pass/fail per kernel against
// the float path, flushing after each so a crash in one kernel names itself
// instead of taking the whole report with it.
//
// This exists because a fault inside the last benchmark row is otherwise
// invisible: the report prints nothing after the previous row, and a hang looks
// identical to a crash.
//
//   cmake --build build --target kverify -j
//   cp build/src/kverify.uf2 /Volumes/RP2350
//   ./serial-probe.py 30

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pico/stdlib.h"

#include "fault.h"
#include "kernels.h"
#include "ws2812.h"

#define N_MAX 8064   // 168 ms at 48 kHz: several ring wraps for every layer

static nam_model_t g_model;
static float       g_in[N_MAX];
static float       g_ref[N_MAX];
static float       g_out[N_MAX];

// Run lengths, chosen to cross every layer's ring wrap point on the way up.
static const int g_lengths[] = {240, 480, 960, 2016, 4032, N_MAX};
#define N_LENGTHS ((int)(sizeof(g_lengths) / sizeof(g_lengths[0])))

typedef void (*kernel_fn)(nam_model_t *, const float *, float *, int);

typedef struct {
    const char *name;
    kernel_fn   fn;
} kernel_desc_t;

static const kernel_desc_t g_kernels[] = {
    {"float scalar",   nam_process_f32},
    {"q15 scalar",     nam_process_q15_nodsp},
    {"q15 SMLAD",      nam_process_q15},
    {"staged",         nam_process_staged},
    {"staged+SMLAD",   nam_process_staged_dsp},
    {"packed+SMLAD",   nam_process_packed},
    {"packed+dupring", nam_process_packed_dup},
    {"packed+wxip",    nam_process_packed_wxip},
    {"widetile x4",    nam_process_widetile},
    {"framemajor",     nam_process_framemajor},
    {"float state",    nam_process_f32state},
};
#define N_KERNELS ((int)(sizeof(g_kernels) / sizeof(g_kernels[0])))

int main(void) {
    stdio_init_all();
    fpi_fault_init();

    // Give the host time to attach. Panic messages go to the CDC endpoint, so
    // without this they are discarded before anyone is listening.
    const uint32_t t0 = to_ms_since_boot(get_absolute_time());
    while (to_ms_since_boot(get_absolute_time()) - t0 < 2500) sleep_ms(20);

    printf("\n=== fpi kernel verification ===\n");
    printf("kernels: %d   run lengths:", N_KERNELS);
    for (int i = 0; i < N_LENGTHS; i++) printf(" %d", g_lengths[i]);
    printf("\n\n");

    uint32_t s = 0xC0FFEEu;
    for (int i = 0; i < N_MAX; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        g_in[i] = (float)(int32_t)(s >> 8) / 8388608.0f * 0.25f;
    }

    if (!nam_model_init(&g_model)) {
        printf("FATAL: ring table failed its invariants (pool sum, power-of-two,\n"
               "       or whole-lane capacity) - see kernels.h\n");
        for (;;) sleep_ms(1000);
    }

    printf("%-14s %6s %10s %10s  %s\n", "kernel", "n", "max err", "rms err", "verdict");
    printf("%-14s %6s %10s %10s  %s\n", "--------------", "-----", "----------",
           "----------", "-------");

    for (int li = 0; li < N_LENGTHS; li++) {
        const int n = g_lengths[li];

        // Reference for this length: the float kernel.
        nam_model_init(&g_model);
        nam_process_f32(&g_model, g_in, g_ref, n);
        printf("%-14s %6d %10s %10s  %s\n", "float scalar", n, "-", "-", "reference");
        stdio_flush();

        for (int k = 1; k < N_KERNELS; k++) {
            // Announce, then force the marker out before running the kernel: if
            // this kernel faults, the last line printed names it.
            printf("%-14s %6d running...", g_kernels[k].name, n);
            stdio_flush();

            memset(g_out, 0, (size_t)n * sizeof(g_out[0]));
            nam_model_init(&g_model);
            g_kernels[k].fn(&g_model, g_in, g_out, n);

            float max_err = 0.0f, sum_sq = 0.0f;
            for (int i = 0; i < n; i++) {
                float e = fabsf(g_out[i] - g_ref[i]);
                if (e > max_err) max_err = e;
                sum_sq += e * e;
            }
            float rms = sqrtf(sum_sq / (float)n);

            // Overwrite the "running..." marker in place.
            printf("\r%-14s %6d %10.6f %10.6f  %s\n", g_kernels[k].name, n, max_err, rms,
                   max_err < 0.01f ? "OK" : "MISMATCH");
            stdio_flush();

            // Also exercise the LED path here. A stalled PIO state machine used
            // to block inside ws2812_set and take the whole firmware with it, so
            // this doubles as a regression check on the WS2812 driver.
            ws2812_set(0, (uint8_t)(k * 30), 0);
        }
    }

    printf("\nall kernels completed\n");
    stdio_flush();
    ws2812_set(0, 40, 0);

    while (true) sleep_ms(1000);
}
