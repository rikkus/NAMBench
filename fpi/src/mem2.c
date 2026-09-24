// mem2: is there two-core contention on the memory system, and is it banking?
//
// ---------------------------------------------------------------------------
// Why this exists
// ---------------------------------------------------------------------------
// The two-core split of the NAM kernel lands at 1.10x the slower half's solo
// time: wall = max(T_A, T_B) + 0.11 * min(T_A, T_B), fitted across nine
// boundaries. That tax is the dominant term in the remaining gap - a *perfect*
// split would be 104% of real time, and the measured split is 117.7% - so it is
// worth knowing what causes it.
//
// What has already been ruled out, by measurement rather than argument:
//
//   * instruction fetch: running the kernel from SRAM instead of flash changes
//     the two-core number by nothing at all (3,679 both ways).
//   * SRAM bank *phase*: shifting one half of the ring pool by 0, 1, 2 or 3
//     interleaved slots - one slot moves every access to the next bank - gives
//     3672, 3672, 3673, 3673 cycles/sample. No effect.
//
// But that last probe was weak, and this file exists because of why: a uniform
// phase shift permutes which bank an access lands in, and a stream that already
// walks over all four banks hits all four either way. Two such streams collide
// with the same probability at every phase. The only way to test "do the two
// cores collide in banks" is to give them memory that stripes over *different*
// banks, which means controlling addresses.
//
// RP2350's main SRAM is two 256 kB regions, each 4-way striped: SRAM0-3 stripe
// over banks 0-3, SRAM4-7 stripe over banks 4-7, disjoint. The datasheet points
// at this directly (section 4.2): "you can still achieve some explicit bandwidth
// partitioning by allocating data across two 256 kB blocks of 4-way-striped
// SRAM". This measures whether that is true and how much it is worth.
//
// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------
// Both cores run an identical fixed workload over their own buffer and count
// their own cycles with their own DWT (the PPB is per-core). The buffers are cut
// out of one large arena at runtime, chosen so that one sits below the
// 0x20040000 boundary and one above it:
//
//   bufA   low  region, banks 0-3
//   bufB   low  region, banks 0-3, 32 kB away from A
//   bufC   high region, banks 4-7
//
// so the same workload can be run with the two cores in the same bank group
// (A+B), in different groups (A+C), or on literally the same words (A+A).
//
// The arena is cut up at runtime rather than placed by the linker because an
// orphan section given a fixed VMA ends up with flash contents in the ELF, and
// picotool rejects that as "memory contents for uninitialized memory".
//
// The access pattern is a stride-7 walk with four loads and one store per
// iteration: spread across banks, and roughly the kernel's mix of memory
// operations to instructions.

#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"

#include "fault.h"
#include "ws2812.h"

#define BUF_WORDS 4096          // 16 kB per buffer
#define ITERS     2000000

// Large enough to reach past 0x20040000 from wherever .bss starts (~0x20002000).
#define ARENA_WORDS (96 * 1024)
static uint32_t arena[ARENA_WORDS];
static uint32_t *bufA, *bufB, *bufC;

static volatile uint32_t g_sink;

// ---------------------------------------------------------------------------
// DWT (Armv8-M peripheral alias addressing), per core.
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
// The workload. `__not_in_flash_func` so the measurement is not entangled with
// instruction fetch, which is already known not to be the issue.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) uint32_t walk_body(uint32_t *buf, int iters) {
    uint32_t a = 1, b = 2, c = 3, d = 4, s = 0;
    for (int i = 0; i < iters; i++) {
        uint32_t j = (uint32_t)(i * 7) & (BUF_WORDS - 1);
        a += buf[j];
        b += buf[(j + 13) & (BUF_WORDS - 1)];
        c += buf[(j + 29) & (BUF_WORDS - 1)];
        d += buf[(j + 61) & (BUF_WORDS - 1)];
        s = a ^ (b << 1) ^ (c << 2) ^ (d << 3) ^ s;
        buf[(j + 5) & (BUF_WORDS - 1)] = s;
    }
    return s;
}

// Two copies of the same loop, differing only in where the *instructions* live.
// This is the variable that matters: the first run showed the two-core penalty is
// 1.26x with the buffers in the same bank group, in different bank groups, and on
// literally the same words - so it is not a data-side effect at all, and the only
// shared resource left that both cores touch every cycle is instruction fetch
// through the 16 kB XIP cache.
static uint32_t memwalk_flash(uint32_t *buf, int iters) { return walk_body(buf, iters); }
static uint32_t __not_in_flash_func(memwalk_ram)(uint32_t *buf, int iters) { return walk_body(buf, iters); }

typedef uint32_t (*walk_fn)(uint32_t *, int);
static volatile walk_fn g_walk;

// ---------------------------------------------------------------------------
// Core 1 server
// ---------------------------------------------------------------------------
static volatile uint32_t g_req, g_ack;
static volatile uint32_t *g_c1_buf;
static volatile uint32_t  g_c1_cycles, g_c1_result;

static void core1_serve(void) {
    dwt_init();
    for (;;) {
        while (g_req == g_ack) __wfe();
        uint32_t *buf = (uint32_t *)g_c1_buf;
        uint32_t t0 = dwt_cycles();
        walk_fn fn = (walk_fn)g_walk;
        uint32_t r = fn(buf, ITERS);
        uint32_t t1 = dwt_cycles();
        g_c1_result = r;
        g_c1_cycles = t1 - t0;
        __dmb();
        g_ack = g_req;
        __sev();
    }
}

// Runs core 0's half and core 1's half on the given buffers.
// Returns core 0's cycles and (through *c1) core 1's.
static uint32_t run_pair(uint32_t *b0, uint32_t *b1, uint32_t *c1, int solo) {
    if (solo) {
        uint32_t t0 = dwt_cycles();
        walk_fn fn = (walk_fn)g_walk;
        uint32_t r = fn(b0, ITERS);
        uint32_t t1 = dwt_cycles();
        g_sink += r;
        if (c1) *c1 = 0;
        return t1 - t0;
    }
    g_c1_buf = b1;
    g_c1_cycles = 0;
    __dmb();
    g_req++;
    __sev();
    uint32_t t0 = dwt_cycles();
    walk_fn fn = (walk_fn)g_walk;
    uint32_t r = fn(b0, ITERS);
    uint32_t t1 = dwt_cycles();
    g_sink += r;
    while (g_ack != g_req) __wfe();
    if (c1) *c1 = g_c1_cycles;
    return t1 - t0;
}

int main(void) {
    stdio_init_all();
    fpi_fault_init();

    const uint32_t t_boot = to_ms_since_boot(get_absolute_time());
    while (to_ms_since_boot(get_absolute_time()) - t_boot < 2500) sleep_ms(20);
    ws2812_set(0, 0, 40);

    printf("\n=== fpi two-core memory contention ===\n");
    dwt_init();

    // Cut the arena up: two 16 kB buffers in the low region and one in the high.
    bufA = arena;
    bufB = arena + 0x2000;                       // 32 kB apart, same region
    {
        uintptr_t base = (uintptr_t)arena;
        uintptr_t hi   = (base < 0x20040000UL) ? 0x20040000UL : base;
        bufC = (uint32_t *)((hi + 0x3FFFUL) & ~(uintptr_t)0x3FFFUL);
        if ((uintptr_t)(bufC + BUF_WORDS) > base + sizeof(arena)) {
            printf("FATAL: arena does not reach the high SRAM region\n");
            for (;;) sleep_ms(1000);
        }
    }
    for (int i = 0; i < BUF_WORDS; i++) { bufA[i] = 0x11111111u; bufB[i] = 0x22222222u; bufC[i] = 0x33333333u; }

    multicore_launch_core1(core1_serve);
    sleep_ms(50);

    const uint32_t f_sys = clock_get_hz(clk_sys);
    printf("clk_sys %u Hz, %d iterations of 4 loads + 1 store\n\n",
           (unsigned)f_sys, ITERS);
    printf("buffer addresses (SRAM stripes on bits 3:2; banks 0-3 below\n");
    printf("0x20040000, banks 4-7 at and above it):\n");
    printf("  bufA %p   bank %u  %s\n", (void *)bufA, (unsigned)(((uintptr_t)bufA >> 2) & 3),
           (uintptr_t)bufA < 0x20040000UL ? "banks 0-3" : "banks 4-7");
    printf("  bufB %p   bank %u  %s\n", (void *)bufB, (unsigned)(((uintptr_t)bufB >> 2) & 3),
           (uintptr_t)bufB < 0x20040000UL ? "banks 0-3" : "banks 4-7");
    printf("  bufC %p   bank %u  %s\n\n", (void *)bufC, (unsigned)(((uintptr_t)bufC >> 2) & 3),
           (uintptr_t)bufC < 0x20040000UL ? "banks 0-3" : "banks 4-7");

    printf("%-38s %12s %12s %9s\n", "configuration", "core0 cyc", "core1 cyc", "vs solo");
    printf("%-38s %12s %12s %9s\n", "--------------------------------------",
           "---------", "---------", "-------");

    for (int place = 0; place < 2; place++) {
        g_walk = place ? memwalk_ram : memwalk_flash;
        const char *pn = place ? "loop in RAM" : "loop in flash";

        uint32_t a_solo = run_pair(bufA, NULL, NULL, 1);

        // core 1 alone on bufC, timed by core 1's own DWT
        g_c1_buf = bufC; g_c1_cycles = 0; __dmb(); g_req++; __sev();
        while (g_ack != g_req) __wfe();
        uint32_t c1solo = g_c1_cycles;

        uint32_t c1same = 0, c1diff = 0, c1self = 0;
        uint32_t a_same = run_pair(bufA, bufB, &c1same, 0);
        uint32_t a_diff = run_pair(bufA, bufC, &c1diff, 0);
        uint32_t a_self = run_pair(bufA, bufA, &c1self, 0);

        printf("%-38s %12u %12s %9s\n", pn, (unsigned)a_solo, "-", "solo");
        printf("%-38s %12s %12u %9s\n", "  core1 alone, bufC", "-", (unsigned)c1solo, "solo");
        printf("%-38s %12u %12u %8.2fx\n", "  both: A + B (same bank group)",
               (unsigned)a_same, (unsigned)c1same, (double)a_same / (double)a_solo);
        printf("%-38s %12u %12u %8.2fx\n", "  both: A + C (different groups)",
               (unsigned)a_diff, (unsigned)c1diff, (double)a_diff / (double)a_solo);
        printf("%-38s %12u %12u %8.2fx\n", "  both: A + A (same words)",
               (unsigned)a_self, (unsigned)c1self, (double)a_self / (double)a_solo);
        printf("\n");
    }

    printf("if the 'both' rows slow down only when the loop is in flash, the shared\n");
    printf("XIP fetch path is what the second core costs - not the data at all.\n");
    printf("\nsink %08x. done.\n", (unsigned)g_sink);

    ws2812_set(0, 40, 0);
    while (true) sleep_ms(1000);
}
