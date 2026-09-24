// WS2812 (addressable RGB) driver for the onboard LED on GPIO16.
//
// The bitstream is produced by PIO from src/ws2812.pio, which pioasm assembles
// and encodes at build time (see src/CMakeLists.txt). Do not hand-assemble this
// program: an earlier version did, with 2/1/1 timing instead of 3/3/4 and no
// FIFO join, and the result was a state machine that never advanced. The LED
// stayed dark and pio_sm_put_blocking() spun forever, which looked like a hang
// in a completely unrelated kernel.

#include "ws2812.h"

#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/pio.h"

#include "ws2812.pio.h"

static PIO  ws2812_pio;
static uint ws2812_sm;
static bool ws2812_ready;

static inline uint32_t ws2812_encode(uint8_t r, uint8_t g, uint8_t b) {
    // WS2812 expects GRB, MSB first.
    return ((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b;
}

int ws2812_init(void) {
    if (ws2812_ready) return 0;

    ws2812_pio = pio0;
    if (!pio_can_add_program(ws2812_pio, &ws2812_program)) return -1;

    // pio_claim_unused_sm returns int and -1 for "none free", so it has to be
    // tested as an int before being narrowed.
    int sm = pio_claim_unused_sm(ws2812_pio, false);
    if (sm < 0) return -1;
    ws2812_sm = (uint)sm;

    uint offset = pio_add_program(ws2812_pio, &ws2812_program);

    // Configure from the header pioasm generated. The pieces that matter, and
    // that a hand-written version got wrong:
    //   * the default config comes from ws2812_program_get_default_config(),
    //     so the sideset count and wrap are consistent with the program;
    //   * the FIFOs are joined, giving 8 entries of TX instead of 4, so a burst
    //     of writes cannot fill it;
    //   * the divider is derived from the program's own T1+T2+T3 cycle counts
    //     rather than an assumed 10 cycles per bit.
    pio_gpio_init(ws2812_pio, WS2812_PIN);
    pio_sm_set_consecutive_pindirs(ws2812_pio, ws2812_sm, WS2812_PIN, 1, true);

    pio_sm_config c = ws2812_program_get_default_config(offset);
    sm_config_set_sideset_pins(&c, WS2812_PIN);
    sm_config_set_out_shift(&c, false, true, 24);   // MSB first, autopull at 24 bits
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_TX);

    const int cycles_per_bit = ws2812_T1 + ws2812_T2 + ws2812_T3;
    sm_config_set_clkdiv(&c, (float)clock_get_hz(clk_sys) / (800000.0f * cycles_per_bit));

    pio_sm_init(ws2812_pio, ws2812_sm, offset, &c);
    pio_sm_set_enabled(ws2812_pio, ws2812_sm, true);

    ws2812_ready = true;
    ws2812_set(0, 0, 0);
    return 0;
}

void ws2812_set(uint8_t r, uint8_t g, uint8_t b) {
    if (!ws2812_ready) return;

    // Never block on the LED. The FIFO is joined so there is room for eight
    // frames and this should never wait, but a cosmetic status LED must not be
    // able to wedge the firmware — which is exactly what it did before.
    absolute_time_t deadline = make_timeout_time_ms(2);
    while (pio_sm_is_tx_fifo_full(ws2812_pio, ws2812_sm)) {
        if (time_reached(deadline)) return;
    }

    // Shifted left by 8: the 24-bit value is right-aligned in the OSR.
    pio_sm_put(ws2812_pio, ws2812_sm, ws2812_encode(r, g, b) << 8u);
}

void ws2812_brightness(uint8_t level) {
    // Simple green -> amber -> red ramp, kept dim enough to be comfortable.
    uint8_t g = (uint8_t)(255 - level);
    uint8_t r = level;
    ws2812_set((uint8_t)(r / 8), (uint8_t)(g / 8), 0);
}
