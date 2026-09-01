// Hardware performance counters around exactly the passes that are timed.
//
// Why this is in the repo rather than a `perf stat` invocation
//
//   Ubuntu ships no perf binary for 32-bit ARM. `linux-perf` does not exist as
//   a package, the `linux-tools-<ver>` armhf packages contain only cpupower,
//   rtla and usbip, and Debian's armhf linux-perf wants a libpython3.13 that
//   Ubuntu 26.04 does not have. So on the one machine in this project that most
//   needs counter attribution, `perf stat` is simply not available.
//
//   That turned out to be a good thing. `perf stat` counts a whole process, and
//   in this benchmark a whole process is mostly not the measurement: it loads a
//   .nam, decodes a wav, builds four engines, runs a parity render and then
//   warms up for five seconds, all before the first timed pass. Counting that
//   and dividing by the number of passes attributes model loading to the
//   kernel. Bracketing each pass individually — which is what this does — puts
//   the counters on the same footing as the timings they sit beside, and lets
//   the per-pass numbers be reduced with the same robustness the timings get
//   rather than being a single whole-run total.
//
// What it counts, and what it will not
//
//   Events are named, not numbered, and resolved through the kernel's own PMU
//   description in /sys/bus/event_source/devices/<pmu>/. So this compiles and
//   runs unchanged on the Cortex-A17 board, on the Pi's Cortex-A76 and on any
//   other Linux PMU; the event names differ between them and that is the
//   caller's problem, which is why --list-counters exists.
//
//   Two hardware truths this cannot paper over, both learned the hard way on
//   Cortex-A17:
//
//   1. A PMU has a fixed number of programmable counters — six on this part.
//      Ask for more events than that and the kernel time-slices them. It does
//      not compensate: it hands back the raw partial count together with the
//      fraction of the region the event actually held a counter for, and
//      scaling is the reader's job. Skipping that step is not conservative, it
//      is simply wrong — asking for eight events on this board reads 1.34e9
//      cycles for a pass that takes 1.78e9, because the event held a counter
//      for 75% of it. So the scaling is applied here, and every event carries
//      the ratio it was scaled by. A scaled value is an estimate rather than a
//      count, and estimates are not comparable between kernels at the level of
//      difference this campaign cares about, which is why the ratio is
//      reported rather than quietly discarded.
//
//   2. sysfs advertises the generic ARMv7 event set, not the set the core
//      actually implements. On Cortex-A17 ld_retired, st_retired and
//      unaligned_ldst_retired open successfully, count nothing, and read back a
//      confident zero. An event that reads exactly zero across every pass is
//      flagged, because "this kernel performed no loads" and "this core does
//      not implement that counter" are otherwise indistinguishable.
//
// Everything here is a no-op that reports itself unavailable off Linux, or when
// perf_event_open is refused — kernel.perf_event_paranoid above 1 hides even
// userspace counters, and that is a configuration problem to report, not an
// error to abort a benchmark run over.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nb
{

/// One event as the kernel describes it, for --list-counters.
struct CounterEvent
{
  std::string name;
  /// The raw perf_event_attr.config this name resolved to, as documentation:
  /// it is the number that would go into a `perf stat -e rNN` on a machine
  /// where perf existed.
  uint64_t config = 0;
};

/// What one bracketed region cost, one entry per opened event.
struct CounterSample
{
  /// Scaled to the whole region: the kernel's raw count divided by the fraction
  /// of the region the event held a counter for.
  std::vector<uint64_t> values;
  /// time_running / time_enabled for each event. 1.0 means the event held a
  /// counter for the whole region and its value is a count; anything less means
  /// the kernel multiplexed and the value is an extrapolation from a sample of
  /// the region.
  std::vector<double> scaling;
};

/// A set of counters bracketing a region of code.
///
/// Usage is deliberately narrow: open once, then begin()/end() around each
/// region. begin() zeroes and enables; end() disables and reads. The syscalls
/// cost a few microseconds against passes measured in hundreds of milliseconds,
/// but they are still outside the timed region proper — the timing this
/// benchmark reports comes from inside the engine, not from this wrapper.
class Counters
{
public:
  Counters() = default;
  ~Counters();

  Counters(const Counters&) = delete;
  Counters& operator=(const Counters&) = delete;

  /// Events the kernel says this machine's PMU has. Empty off Linux, or when no
  /// core PMU is exposed.
  static std::vector<CounterEvent> available_events();

  /// A reasonable starting set for a kernel campaign, filtered to the events
  /// this machine actually advertises and truncated to the counter budget:
  /// cycles and instructions to normalise everything else, memory accesses and
  /// L1D refills to see spilling, and L2 refills to see the working set leave
  /// the core.
  static std::vector<std::string> default_events();

  /// How many events can be counted at once without multiplexing. Determined by
  /// opening events until one fails to hold a counter, because no interface
  /// reports it: sysfs does not say, and the kernel accepts more than it can
  /// schedule. Returns 0 when counters are unavailable.
  static int counter_budget();

  /// Opens one file descriptor per event. Returns false and sets reason() on
  /// the first failure, having opened nothing.
  bool open(const std::vector<std::string>& eventNames);

  bool is_open() const { return !fds_.empty(); }
  const std::vector<std::string>& names() const { return names_; }
  const std::vector<CounterEvent>& events() const { return events_; }

  /// Why open() failed, or why the counters are unavailable at all.
  const std::string& reason() const { return reason_; }

  /// Zero and start counting.
  void begin();
  /// Stop counting and read. Values are for the region since begin() alone.
  void end(CounterSample& out);

  void close();

private:
  std::vector<int> fds_;
  std::vector<std::string> names_;
  std::vector<CounterEvent> events_;
  std::string reason_;
};

/// Per-event reduction over many passes.
///
/// The median, not the mean. Counter samples inherit the same outliers the
/// timings do — a pass that took a page fault or lost the core to something
/// else counts the extra work it did — and the benchmark's own answer to that
/// is to discard the tails. Reducing counters the same way keeps them
/// consistent with the timing they sit beside.
struct CounterSummary
{
  std::string name;
  uint64_t config = 0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  /// Lowest running/enabled ratio seen across the passes. Below 1.0 the numbers
  /// were scaled up from a partial sample of each pass: estimates, not counts.
  double worstScaling = 1.0;
  /// Read exactly zero on every pass. Almost always means the core does not
  /// implement the event, whatever sysfs claims.
  bool alwaysZero = false;
};

/// Reduces per-pass samples into one summary per event. `samples` may be empty,
/// in which case the result is too.
std::vector<CounterSummary> summarise(const std::vector<std::string>& names,
                                      const std::vector<CounterEvent>& events,
                                      const std::vector<CounterSample>& samples);

} // namespace nb
