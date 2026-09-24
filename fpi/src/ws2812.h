#ifndef FPI_WS2812_H
#define FPI_WS2812_H

#include <stdint.h>

// Onboard WS2812 on GPIO16 (see docs/fpi/README.md, "GPIO map").
#define WS2812_PIN 16

// Claim a PIO state machine and drive the LED. Idempotent; returns 0 on success.
int ws2812_init(void);

// Set the LED. Components are 0..255. Blocks for ~30 us per call, which is
// fine for status indication but not something to do per audio sample.
void ws2812_set(uint8_t r, uint8_t g, uint8_t b);

// Convenience: scale a single brightness 0..255 across a hue-ish ramp.
void ws2812_brightness(uint8_t level);

#endif
