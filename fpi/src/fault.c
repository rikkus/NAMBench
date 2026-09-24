// Fault reporting. See fault.h for why this exists.
//
// Deliberately does not decode the exception stack frame: on a part with an FPU
// the frame is either 8 or 26 words depending on whether floating point state
// was stacked, and getting that wrong yields confidently incorrect register
// values. The status registers (CFSR/HFSR/MMFAR/BFAR) identify the fault
// without guessing, so those are what gets reported.

#include "fault.h"

#include <stdio.h>

#include "pico/stdlib.h"

// System Control Block, Armv8-M peripheral alias addressing.
#define SCB_CFSR    (*(volatile uint32_t *)0xE000ED28UL)  // MMSR|BFSR|UFSR
#define SCB_HFSR    (*(volatile uint32_t *)0xE000ED2CUL)
#define SCB_MMFAR   (*(volatile uint32_t *)0xE000ED34UL)
#define SCB_BFAR    (*(volatile uint32_t *)0xE000ED38UL)
#define SCB_SHCSR   (*(volatile uint32_t *)0xE000ED24UL)

// One report per boot is enough; repeating it forever would bury the output.
static volatile bool fault_armed = true;
static bool fault_reported = false;

static int64_t disarm_faults(alarm_id_t alarm_id, void *user_data) {
    (void)alarm_id;
    (void)user_data;
    fault_armed = false;
    return 0;  // do not reschedule
}

void fpi_fault_init(void) {
    // The window has to cover the whole workload, not just startup. The first
    // version of this used 5 s and the benchmark crossed 5 s partway through
    // its last kernel, so the fault that mattered was swallowed by the handler
    // having already disarmed itself — the exact failure it existed to catch.
    // Keep this comfortably longer than any firmware run.
    add_alarm_in_ms(600000, disarm_faults, NULL, false);
}

__attribute__((noreturn)) void fpi_fault_report(const char *what) {
    if (fault_armed && !fault_reported) {
        fault_reported = true;
        // Kept short on purpose: the TinyUSB CDC FIFO is 64 bytes, so a long
        // report would be truncated mid-line and lose the interesting half.
        // The fault type arrives as `what`; these are the status registers that
        // identify it precisely.
        char buf[100];
        int n = snprintf(buf, sizeof(buf),
                         "\n*** FAULT: %s ***\n"
                         " CFSR=%08x HFSR=%08x\n"
                         " MMFAR=%08x BFAR=%08x\n",
                         what,
                         (unsigned)SCB_CFSR,
                         (unsigned)SCB_HFSR,
                         (unsigned)SCB_MMFAR,
                         (unsigned)SCB_BFAR);
        if (n > 0) {
            if (n > (int)sizeof(buf) - 1) n = (int)sizeof(buf) - 1;
            stdio_put_string(buf, n, false, false);
        }
    }
    for (;;) tight_loop_contents();
}

// The trampolines must be naked: they run before any prologue, so there is no
// clobbered state to preserve and nothing to unwind. The names have external
// linkage on purpose: the assembler cannot reference a function-local static
// (the compiler gives those suffixed local symbols), and `ldr rX, =symbol`
// needs a symbol that actually exists in the object file.
const char fpi_hardfault_name[]  = "HardFault";
const char fpi_busfault_name[]   = "BusFault";
const char fpi_memmanage_name[]  = "MemManage";
const char fpi_usagefault_name[] = "UsageFault";

#define FPI_FAULT_TRAMPOLINE(fn, label)                                       \
    void __attribute__((naked, noreturn)) fn(void) {                          \
        __asm volatile("ldr r0, =" #label "\n\t"                              \
                       "b fpi_fault_report\n\t");                             \
    }

FPI_FAULT_TRAMPOLINE(isr_hardfault,  fpi_hardfault_name)
FPI_FAULT_TRAMPOLINE(isr_busfault,   fpi_busfault_name)
FPI_FAULT_TRAMPOLINE(isr_memmanage,  fpi_memmanage_name)
FPI_FAULT_TRAMPOLINE(isr_usagefault, fpi_usagefault_name)
