// First-contact firmware for fpi (Waveshare RP2350A-USB-A Mini).
//
// Prints progress from the very first instruction that can print, reports the
// status of each subsystem so a silent terminal localises the fault, then emits
// a steady heartbeat. If the heartbeat is flowing but the LED is dark, that is
// the WS2812 driver; if nothing flows at all, it is USB or the app is faulting.
//
//   cmake --build build --target hello -j
//   cp build/src/hello.uf2 /Volumes/RP2350
//   ../monitor.sh

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/unique_id.h"
#include "hardware/clocks.h"

#include "fault.h"
#include "ws2812.h"

// ---------------------------------------------------------------------------
// Console output that does not depend on the SDK's connection gate.
//
// Diagnosis so far (see docs/fpi/TOOLCHAIN.md for the blow-by-blow): the board
// enumerates as 2e8a:0009, so TinyUSB and main() are up, yet zero bytes reach
// the host. stdio_usb_out_chars() only writes when stdio_usb_connected() is
// true, and that "actually checks DTR" unless built with
// PICO_STDIO_USB_CONNECTION_WITHOUT_DTR. macOS does not raise DTR on open, and
// raising it from the host via TIOCMBIS does not reliably turn into a
// SET_CONTROL_LINE_STATE on the wire.
//
// So this prints through the CDC endpoint directly, gated only on tud_mounted(),
// and falls back to stdio if the direct path is not available. Between the two
// there is no DTR-shaped hole left.
// ---------------------------------------------------------------------------
// tud_cdc_write()/tud_cdc_write_flush() are static inline wrappers in
// cdc_device.h around the interface-indexed functions that actually exist as
// symbols. Declaring those directly avoids pulling the whole TinyUSB class
// header into a file that only needs two calls.
extern bool tud_mounted(void);
extern uint32_t tud_cdc_n_write(uint8_t itf, const void *buffer, uint32_t bufsize);
extern uint32_t tud_cdc_n_write_flush(uint8_t itf);

static void fpi_write(const char *s, int n) {
    if (n <= 0) return;
    if (tud_mounted()) {
        tud_cdc_n_write(0, s, (uint32_t)n);
        tud_cdc_n_write_flush(0);
        return;
    }
    stdio_put_string(s, n, false, false);
}

static void fpi_log(const char *fmt, ...) {
    char buf[192];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fpi_write(buf, n);
}

// Bump this whenever the firmware changes, so the terminal says which build is
// actually on the chip.
#define HELLO_BUILD 2

// ---------------------------------------------------------------------------
// DWT cycle counter. Armv8-M, so registers are reached through the peripheral
// alias — note this is *not* the same encoding trick as on Cortex-M0+/M3.
// ---------------------------------------------------------------------------
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004UL)
#define DEMCR      (*(volatile uint32_t *)0xE000EDFCUL)

static void dwt_init(void) {
    DEMCR |= (1u << 24);        // TRCENA
    DWT_CYCCNT = 0;
    DWT_CTRL |= (1u << 0);      // CYCCNTENA
}
static inline uint32_t dwt_cycles(void) { return DWT_CYCCNT; }
static inline uint32_t dwt_running(void) { return DWT_CTRL & 1u; }

int main(void) {
    stdio_init_all();
    fpi_fault_init();

    // Wait for the host to enumerate the CDC interface before the first write.
    // Writing earlier is not merely lost, it is invisible: tud_mounted() is
    // false, so fpi_write() would hand the text to a stdio driver that discards
    // it. Two seconds is what this laptop takes.
    for (int i = 0; i < 400 && !tud_mounted(); i++) sleep_ms(5);

    fpi_log("\nfpi hello build %d: alive\n", HELLO_BUILD);
    fpi_log("  usb      : %s\n", tud_mounted() ? "mounted" : "NOT MOUNTED (is a cable attached?)");

    dwt_init();
    fpi_log("  dwt      : %s\n", dwt_running() ? "running" : "NOT RUNNING");

    int led = ws2812_init();
    fpi_log("  ws2812   : %s (init returned %d, pin %d)\n",
            led == 0 ? "ok" : "FAILED", led, WS2812_PIN);

    char id[2 * PICO_UNIQUE_BOARD_ID_SIZE_BYTES + 1];
    pico_get_unique_board_id_string(id, sizeof(id));

    fpi_log("  board    : Waveshare RP2350A-USB-A Mini\n");
    fpi_log("  board id : %s\n", id);
    fpi_log("  clk_sys  : %u Hz\n", (unsigned)clock_get_hz(clk_sys));
    fpi_log("  core     : %d of 2 (this build uses one)\n", get_core_num());
    fpi_log("  flash    : %u bytes\n", (unsigned)PICO_FLASH_SIZE_BYTES);
    fpi_log("\nheartbeat every 250 ms (cycles since last beat):\n");

    // Colours cycle forever, so the LED state alone indicates liveness even
    // with no terminal attached.
    static const uint8_t colours[4][3] = {
        {32, 0, 0}, {0, 32, 0}, {0, 0, 32}, {24, 24, 0},
    };

    uint32_t beat = 0;
    uint32_t last = dwt_cycles();
    while (true) {
        uint32_t now = dwt_cycles();
        uint32_t delta = now - last;
        last = now;

        // tud_mounted() appears in every beat on purpose: if the host is seeing
        // a port that never speaks, the question is whether the device thinks
        // it is mounted at all, and this answers it without another rebuild.
        fpi_log("  beat %4u  %10u cyc  (%u us)  usb=%s\n",
                (unsigned)beat, (unsigned)delta,
                (unsigned)(delta / (clock_get_hz(clk_sys) / 1000000u)),
                tud_mounted() ? "mounted" : "none");

        ws2812_set(colours[beat % 4][0], colours[beat % 4][1], colours[beat % 4][2]);
        beat++;
        sleep_ms(250);
    }
}
