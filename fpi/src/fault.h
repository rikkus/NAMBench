#ifndef FPI_FAULT_H
#define FPI_FAULT_H

#include <stdbool.h>

// Fault reporting for fpi firmware.
//
// Without this a crash is indistinguishable from a dead board: TinyUSB is
// served from a low-priority interrupt, so the CDC interface stays enumerated
// and the host sees a port that never speaks. These handlers override the SDK's
// weak defaults and print the fault status registers before halting.
//
// Enable early in main(). A one-shot alarm disables the handler after a few
// seconds so that a long-running benchmark that faults is reported once rather
// than filling the console with repeats.
void fpi_fault_init(void);

#endif
