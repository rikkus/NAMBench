// See Tools/nb_counters.h for why this exists rather than a `perf stat`.

#include "nb_counters.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
  #include <asm/unistd.h>
  #include <dirent.h>
  #include <fcntl.h>
  #include <linux/perf_event.h>
  #include <sys/ioctl.h>
  #include <sys/syscall.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace nb
{

#if defined(__linux__)

namespace
{

const char* const kPmuRoot = "/sys/bus/event_source/devices";

std::string read_file(const std::string& path)
{
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    return {};
  char buffer[512];
  const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
  std::fclose(f);
  buffer[n] = '\0';
  std::string text(buffer, n);
  while (!text.empty() && (text.back() == '\n' || text.back() == ' '))
    text.pop_back();
  return text;
}

bool path_exists(const std::string& path)
{
  return ::access(path.c_str(), F_OK) == 0;
}

/// Finds the core PMU.
///
/// There is no fixed name for it — it is armv7_cortex_a12 on this Cortex-A17
/// (the kernel's A12 driver covers both), armv8_pmuv3 or armv8_cortex_a76 on
/// AArch64, cpu on x86. What they have in common, and what nothing else under
/// event_source has, is a cpumask: this is the PMU attached to CPUs. Software,
/// tracepoint, kprobe and breakpoint sources have no cpumask, which is exactly
/// the discrimination wanted, and it avoids a name table that would need a new
/// row for every part this project is ever run on.
///
/// Uncore PMUs on server parts also carry a cpumask, so prefer a directory that
/// also has an events/cpu_cycles or events/cycles — a core PMU always does.
std::string find_core_pmu()
{
  DIR* dir = ::opendir(kPmuRoot);
  if (!dir)
    return {};

  std::string best;
  struct dirent* entry = nullptr;
  while ((entry = ::readdir(dir)) != nullptr)
  {
    const std::string name = entry->d_name;
    if (name == "." || name == "..")
      continue;
    const std::string base = std::string(kPmuRoot) + "/" + name;
    if (!path_exists(base + "/cpumask") && !path_exists(base + "/cpus"))
      continue;
    if (!path_exists(base + "/type"))
      continue;
    if (path_exists(base + "/events/cpu_cycles") || path_exists(base + "/events/cycles"))
    {
      best = name;
      break;
    }
    if (best.empty())
      best = name;
  }
  ::closedir(dir);
  return best;
}

struct FormatField
{
  int low = 0;
  int high = 0;
};

/// Parses a format descriptor such as "config:0-7" or "config:16".
///
/// ARM PMUs put the whole event number in config:0-7, so this is arguably more
/// machinery than this project needs. It is here because the alternative is
/// assuming that, and the assumption is wrong on x86 the moment anyone runs
/// this on a desktop with an umask= term.
bool parse_format(const std::string& pmu, const std::string& field, FormatField& out)
{
  const std::string text = read_file(std::string(kPmuRoot) + "/" + pmu + "/format/" + field);
  const size_t colon = text.find(':');
  if (text.rfind("config", 0) != 0 || colon == std::string::npos)
    return false;
  const std::string range = text.substr(colon + 1);
  const size_t dash = range.find('-');
  out.low = std::atoi(range.c_str());
  out.high = (dash == std::string::npos) ? out.low : std::atoi(range.c_str() + dash + 1);
  return out.low >= 0 && out.high >= out.low && out.high < 64;
}

/// Turns the kernel's event description — "event=0x11", or a comma-separated
/// list of terms — into a perf_event_attr.config.
bool resolve_event(const std::string& pmu, const std::string& name, uint64_t& config)
{
  const std::string terms = read_file(std::string(kPmuRoot) + "/" + pmu + "/events/" + name);
  if (terms.empty())
    return false;

  config = 0;
  size_t pos = 0;
  while (pos <= terms.size())
  {
    const size_t comma = terms.find(',', pos);
    const std::string term = terms.substr(pos, comma - pos);
    if (!term.empty())
    {
      const size_t eq = term.find('=');
      const std::string field = term.substr(0, eq);
      const uint64_t value =
        (eq == std::string::npos) ? 1u : std::strtoull(term.c_str() + eq + 1, nullptr, 0);

      FormatField format;
      if (!parse_format(pmu, field, format))
        return false;
      const int width = format.high - format.low + 1;
      const uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
      config |= (value & mask) << format.low;
    }
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return true;
}

int open_counter(uint32_t type, uint64_t config)
{
  perf_event_attr attr;
  std::memset(&attr, 0, sizeof attr);
  attr.size = sizeof attr;
  attr.type = type;
  attr.config = config;
  attr.disabled = 1;
  // The engine's work is what is being measured, not the kernel's. Excluding
  // kernel and hypervisor also keeps this working at perf_event_paranoid 1,
  // which is the least privilege that lets it run at all.
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
  // Count this thread on whatever CPU it is on. Not group-led: a group is
  // scheduled all-or-nothing, so one event too many silently costs the whole
  // set its samples instead of costing each event some of its time, and the
  // latter is at least visible in the scaling ratio.
  return static_cast<int>(::syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
}

} // namespace

Counters::~Counters()
{
  close();
}

std::vector<CounterEvent> Counters::available_events()
{
  std::vector<CounterEvent> events;
  const std::string pmu = find_core_pmu();
  if (pmu.empty())
    return events;

  const std::string dirPath = std::string(kPmuRoot) + "/" + pmu + "/events";
  DIR* dir = ::opendir(dirPath.c_str());
  if (!dir)
    return events;

  struct dirent* entry = nullptr;
  while ((entry = ::readdir(dir)) != nullptr)
  {
    const std::string name = entry->d_name;
    if (name == "." || name == ".." || name.find('.') != std::string::npos)
      continue;
    uint64_t config = 0;
    if (resolve_event(pmu, name, config))
      events.push_back({name, config});
  }
  ::closedir(dir);

  std::sort(events.begin(), events.end(),
            [](const CounterEvent& a, const CounterEvent& b) { return a.name < b.name; });
  return events;
}

int Counters::counter_budget()
{
  // Probed once. It costs a handful of syscalls and a short spin, and it is
  // asked for from three places in a run.
  static int cached = -1;
  if (cached >= 0)
    return cached;
  cached = 0;

  const std::string pmu = find_core_pmu();
  if (pmu.empty())
    return cached;
  const uint32_t type = static_cast<uint32_t>(
    std::strtoul(read_file(std::string(kPmuRoot) + "/" + pmu + "/type").c_str(), nullptr, 10));

  // Probed with the cycle event because it is the one event every PMU has. That
  // makes this an upper bound rather than an exact answer on parts with a
  // dedicated cycle counter alongside the programmable ones, where a set of six
  // non-cycle events might not fit where six cycle events did. It agrees with a
  // hand-probe of six distinct events on Cortex-A17, and in any case the
  // multiplexing is detected per-event at read time — this number exists to
  // warn before a run, not to be relied on instead of that.
  uint64_t config = 0;
  if (!resolve_event(pmu, "cpu_cycles", config) && !resolve_event(pmu, "cycles", config))
    return cached;

  // Open the same event repeatedly and watch for the point where the kernel
  // stops being able to schedule them all simultaneously. Nothing reports the
  // count of programmable counters: sysfs does not expose it, and
  // perf_event_open happily accepts more events than the hardware can hold
  // because multiplexing is its answer rather than an error.
  //
  // A short busy region is needed for time_running to diverge from
  // time_enabled — with no elapsed time both are zero and every set looks
  // schedulable.
  constexpr int kCeiling = 16;
  std::vector<int> fds;
  int budget = 0;
  for (int n = 1; n <= kCeiling; n++)
  {
    const int fd = open_counter(type, config);
    if (fd < 0)
      break;
    fds.push_back(fd);

    for (int fdi : fds)
    {
      ::ioctl(fdi, PERF_EVENT_IOC_RESET, 0);
      ::ioctl(fdi, PERF_EVENT_IOC_ENABLE, 0);
    }
    volatile uint64_t spin = 0;
    for (int i = 0; i < 200000; i++)
      spin += static_cast<uint64_t>(i);
    (void)spin;
    bool allFull = true;
    for (int fdi : fds)
    {
      ::ioctl(fdi, PERF_EVENT_IOC_DISABLE, 0);
      uint64_t buffer[3] = {0, 0, 0};
      if (::read(fdi, buffer, sizeof buffer) != static_cast<ssize_t>(sizeof buffer))
        allFull = false;
      else if (buffer[1] == 0 || buffer[2] < buffer[1])
        allFull = false;
    }
    if (!allFull)
      break;
    budget = n;
  }
  for (int fd : fds)
    ::close(fd);
  cached = budget;
  return cached;
}

std::vector<std::string> Counters::default_events()
{
  // Ordered by how much each one is worth giving a counter to, so truncating to
  // the budget drops the least useful rather than an arbitrary tail.
  //
  // ld_retired and st_retired are deliberately absent even though every ARMv7
  // PMU advertises them: Cortex-A17 does not implement them and returns zero.
  // mem_access is the substitute, and counts both directions.
  static const char* const kPreferred[] = {
    "cpu_cycles",       "cycles",          "inst_retired",    "instructions",
    "mem_access",       "l1d_cache",       "l1d_cache_refill", "l2d_cache_refill",
    "inst_spec",        "l1d_tlb_refill",  "bus_access",      "br_mis_pred",
  };

  std::vector<CounterEvent> have = available_events();
  // When the budget could not be probed — which is what happens when
  // perf_event_paranoid forbids opening a counter at all — take four rather
  // than the whole preferred list. Every PMU this is likely to meet has at
  // least four programmable counters, and the two failure modes are not
  // symmetric: too few loses one event, too many quietly turns every value on
  // the report into an estimate.
  const int probed = counter_budget();
  const int budget = (probed > 0) ? probed : 4;
  std::vector<std::string> chosen;
  bool haveCycles = false;
  bool haveInstructions = false;

  for (const char* candidate : kPreferred)
  {
    if (static_cast<int>(chosen.size()) >= budget)
      break;
    const bool present = std::any_of(have.begin(), have.end(), [&](const CounterEvent& e) {
      return e.name == candidate;
    });
    if (!present)
      continue;
    // cpu_cycles and cycles are the same counter under two names, as are
    // inst_retired and instructions; taking both would waste a slot on a
    // duplicate.
    const std::string name = candidate;
    if (name == "cpu_cycles" || name == "cycles")
    {
      if (haveCycles)
        continue;
      haveCycles = true;
    }
    if (name == "inst_retired" || name == "instructions")
    {
      if (haveInstructions)
        continue;
      haveInstructions = true;
    }
    chosen.push_back(name);
  }
  return chosen;
}

bool Counters::open(const std::vector<std::string>& eventNames)
{
  close();
  reason_.clear();

  const std::string pmu = find_core_pmu();
  if (pmu.empty())
  {
    reason_ = "no core PMU under " + std::string(kPmuRoot);
    return false;
  }
  const uint32_t type = static_cast<uint32_t>(
    std::strtoul(read_file(std::string(kPmuRoot) + "/" + pmu + "/type").c_str(), nullptr, 10));

  for (const std::string& name : eventNames)
  {
    uint64_t config = 0;
    if (!resolve_event(pmu, name, config))
    {
      reason_ = "PMU " + pmu + " has no event called '" + name + "' (try --list-counters)";
      close();
      return false;
    }
    const int fd = open_counter(type, config);
    if (fd < 0)
    {
      char buffer[256];
      std::snprintf(buffer, sizeof buffer,
                    "perf_event_open('%s') failed: %s. "
                    "kernel.perf_event_paranoid must be 1 or lower for userspace counters",
                    name.c_str(), std::strerror(errno));
      reason_ = buffer;
      close();
      return false;
    }
    fds_.push_back(fd);
    names_.push_back(name);
    events_.push_back({name, config});
  }
  return true;
}

void Counters::begin()
{
  for (int fd : fds_)
  {
    ::ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }
}

void Counters::end(CounterSample& out)
{
  out.values.assign(fds_.size(), 0);
  out.scaling.assign(fds_.size(), 0.0);
  for (size_t i = 0; i < fds_.size(); i++)
  {
    ::ioctl(fds_[i], PERF_EVENT_IOC_DISABLE, 0);
    uint64_t buffer[3] = {0, 0, 0};
    if (::read(fds_[i], buffer, sizeof buffer) != static_cast<ssize_t>(sizeof buffer))
      continue;
    const uint64_t value = buffer[0];
    const uint64_t enabled = buffer[1];
    const uint64_t running = buffer[2];
    // Scale a multiplexed count back up to the whole region. The kernel does
    // not do this — it reports the raw count and the time the event was
    // actually scheduled, and leaves the arithmetic to the reader. Skipping it
    // is not the safe option: eight events on a six-counter PMU read three
    // quarters of the true cycle count, which looks like a kernel that got 33%
    // faster rather than a measurement that lost a quarter of its samples.
    const double ratio = (enabled > 0) ? (static_cast<double>(running) / static_cast<double>(enabled))
                                       : 0.0;
    out.values[i] =
      (ratio > 0.0) ? static_cast<uint64_t>(static_cast<double>(value) / ratio + 0.5) : value;
    out.scaling[i] = ratio;
  }
}

void Counters::close()
{
  for (int fd : fds_)
    ::close(fd);
  fds_.clear();
  names_.clear();
  events_.clear();
}

#else // !__linux__

Counters::~Counters() = default;

std::vector<CounterEvent> Counters::available_events()
{
  return {};
}

std::vector<std::string> Counters::default_events()
{
  return {};
}

int Counters::counter_budget()
{
  return 0;
}

bool Counters::open(const std::vector<std::string>&)
{
  reason_ = "hardware counters need Linux perf_event_open; this is not Linux";
  return false;
}

void Counters::begin() {}

void Counters::end(CounterSample& out)
{
  out.values.clear();
  out.scaling.clear();
}

void Counters::close() {}

#endif

std::vector<CounterSummary> summarise(const std::vector<std::string>& names,
                                      const std::vector<CounterEvent>& events,
                                      const std::vector<CounterSample>& samples)
{
  std::vector<CounterSummary> summaries;
  if (names.empty() || samples.empty())
    return summaries;

  for (size_t e = 0; e < names.size(); e++)
  {
    CounterSummary summary;
    summary.name = names[e];
    summary.config = (e < events.size()) ? events[e].config : 0;

    std::vector<double> values;
    values.reserve(samples.size());
    double worst = 1.0;
    for (const CounterSample& sample : samples)
    {
      if (e >= sample.values.size())
        continue;
      values.push_back(static_cast<double>(sample.values[e]));
      if (e < sample.scaling.size())
        worst = std::min(worst, sample.scaling[e]);
    }
    if (values.empty())
      continue;

    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    summary.median = (n % 2) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    summary.min = values.front();
    summary.max = values.back();
    summary.worstScaling = worst;
    summary.alwaysZero = (values.back() == 0.0);
    summaries.push_back(summary);
  }
  return summaries;
}

} // namespace nb
