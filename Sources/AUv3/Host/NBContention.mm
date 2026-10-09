// Cache contention: what the kernel costs when it doesn't have the machine to
// itself. See plans/auv3-overhead.md ("Cache contention") and CONTENTION.md.
//
// Every other NAMBench number runs one instance alone, so its weights and state
// stay in cache from block to block. A real session doesn't: other plugins run
// between its blocks on the same thread, other tracks run on other cores that
// share a cache, and unrelated processes stream through memory. This times
// the bare per-block kernel (arm A1's call, from the same protocol) under
// three kinds of pressure:
//
//   solo      one instance, nothing else: the baseline
//   serial    N instances, one thread, block by block: for each block, every
//             instance processes it in turn, as plugins in a chain or tracks on
//             one audio thread do. Instance 0 is the one timed per block.
//   parallel  N instances on N threads, each flat out over the input: the timed
//             instance on this thread, N-1 neighbours on their own
//   thrash    one instance, and a neighbour thread that streams reads and
//             writes over a buffer of a given size, and does nothing else
//
// Offline, like arms A-D: this measures CPU cost, not deadlines. Instances are
// separate NbModel objects built from the same file, so each has its own
// weights in its own memory.

#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <malloc/malloc.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#import <NAMEnginePlanar/NAMEnginePlanar.h>

#include "nb_protocol.h"
#include "nb_shim_timing.h"

#include "NBContention.h"

namespace
{

double g_ticksToNs = 1.0;

inline double ticks_ns(uint64_t ticks)
{
  return static_cast<double>(ticks) * g_ticksToNs;
}

double quantile(std::vector<double> v, double q)
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

const char* submodel_name(NbSubmodel s)
{
  return s == NbSubmodelNarrowest ? "nano" : "standard";
}

size_t malloc_in_use()
{
  malloc_statistics_t stats{};
  malloc_zone_statistics(nullptr, &stats);
  return stats.size_in_use;
}

NbModel* create(NSData* model, NbSubmodel submodel, int blockSize, std::string& error)
{
  char err[512] = {0};
  NbModel* m = nb_planar_create(static_cast<const uint8_t*>(model.bytes), model.length, submodel, 0, blockSize, err,
                                sizeof(err));
  if (m == nullptr)
    error = err;
  return m;
}

/// A neighbour that runs until told to stop. For parallel, it loops one
/// instance over the input; for thrash, it streams over its buffer.
struct Neighbour
{
  std::thread thread;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> blocks{0};
  std::map<int, uint64_t> cpus; // written by the thread, read after join
};

void thrash_loop(Neighbour& n, size_t bytes)
{
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  std::vector<uint64_t> buffer(bytes / sizeof(uint64_t), 1);
  // One 64-byte line in eight words: touch every line, read and write, so the
  // neighbour both evicts and dirties. A dependent chain would measure latency,
  // not pressure; this is a stream.
  const size_t stride = 8;
  uint64_t sum = 0;
  while (!n.stop.load(std::memory_order_relaxed))
  {
    for (size_t i = 0; i < buffer.size(); i += stride)
    {
      sum += buffer[i];
      buffer[i] = sum;
    }
    n.blocks.fetch_add(1, std::memory_order_relaxed);
  }
  if (sum == 42)
    std::fprintf(stderr, " ");
}

void instance_loop(Neighbour& n, NbModel* m, const std::vector<float>& input, int blockSize)
{
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  std::vector<float> out(static_cast<size_t>(blockSize));
  const size_t bs = static_cast<size_t>(blockSize);
  size_t offset = 0;
  uint64_t count = 0;
  while (!n.stop.load(std::memory_order_relaxed))
  {
    if (offset + bs > input.size())
      offset = 0;
    const uint64_t fpcr = denormals_disable();
    nb_planar_process_block(m, input.data() + offset, out.data(), blockSize);
    denormals_restore(fpcr);
    offset += bs;
    if ((++count & 255) == 0)
    {
      size_t cpu = 0;
      pthread_cpu_number_np(&cpu);
      n.cpus[static_cast<int>(cpu)]++;
    }
  }
  n.blocks.store(count);
}

class Runner
{
public:
  Runner(const ContentionOptions& options, NSData* model, const std::vector<float>& input, double sampleRate,
         FILE* log)
      : options_(options), model_(model), input_(input), sampleRate_(sampleRate), log_(log)
  {
  }

  ContentionResult run(const std::string& scenario, NbSubmodel submodel, int blockSize, int instances,
                       int thrashKiB, const std::vector<float>* reference, std::vector<float>* capture);

private:
  const ContentionOptions& options_;
  NSData* model_;
  const std::vector<float>& input_;
  double sampleRate_;
  FILE* log_;
};

ContentionResult Runner::run(const std::string& scenario, NbSubmodel submodel, int blockSize, int instances,
                             int thrashKiB, const std::vector<float>* reference, std::vector<float>* capture)
{
  ContentionResult r;
  r.scenario = scenario;
  r.submodel = submodel;
  r.blockSize = blockSize;
  r.instances = instances;
  r.thrashKiB = thrashKiB;
  std::string error;

  const bool serial = scenario == "serial";
  const bool parallel = scenario == "parallel";
  const bool thrash = scenario == "thrash";

  // --- Build: instance 0 is timed; serial runs the rest on this thread,
  // parallel on their own.
  std::vector<NbModel*> models;
  const size_t before = malloc_in_use();
  NbModel* first = create(model_, submodel, blockSize, error);
  if (first == nullptr)
  {
    r.failure = error;
    return r;
  }
  r.instanceBytes = malloc_in_use() - before;
  models.push_back(first);
  const int total = (serial || parallel) ? instances : 1;
  for (int i = 1; i < total; i++)
  {
    NbModel* m = create(model_, submodel, blockSize, error);
    if (m == nullptr)
    {
      for (NbModel* x : models)
        nb_planar_destroy(x);
      r.failure = error;
      return r;
    }
    models.push_back(m);
  }
  // Serial runs every instance on this thread; parallel only the first.
  const size_t onThisThread = serial ? models.size() : 1;

  const size_t frames = input_.size();
  const size_t bs = static_cast<size_t>(blockSize);
  const size_t blockCount = frames / bs; // whole blocks only, so every block is the same size
  std::vector<double> blockNs(blockCount);
  std::vector<float> out(frames, 0.0f);
  std::vector<float> scratch(bs);

  auto reset = [&] {
    for (size_t i = 0; i < onThisThread; i++)
      nb_planar_reset(models[i], sampleRate_, blockSize);
  };

  // One pass over the file. Returns the cost of one instance: in serial, the
  // whole pass divided by the instance count, so core % means the same thing
  // in every scenario. Per-block times are instance 0's alone.
  auto pass = [&](float* dst) -> uint64_t {
    const uint64_t start = mach_absolute_time();
    for (size_t b = 0; b < blockCount; b++)
    {
      const size_t offset = b * bs;
      const uint64_t fpcr = denormals_disable();
      const uint64_t t0 = mach_absolute_time();
      nb_planar_process_block(models[0], input_.data() + offset, dst ? dst + offset : scratch.data(), blockSize);
      const uint64_t t1 = mach_absolute_time();
      for (size_t i = 1; i < onThisThread; i++)
        nb_planar_process_block(models[i], input_.data() + offset, scratch.data(), blockSize);
      denormals_restore(fpcr);
      blockNs[b] = ticks_ns(t1 - t0);
    }
    const double elapsed = ticks_ns(mach_absolute_time() - start);
    return static_cast<uint64_t>(elapsed / static_cast<double>(onThisThread));
  };

  // --- Neighbours, started before anything is timed, parity pass included,
  // so the timed instance never sees a moment without them.
  std::vector<std::unique_ptr<Neighbour>> neighbours;
  if (parallel)
  {
    for (size_t i = 1; i < models.size(); i++)
    {
      neighbours.push_back(std::make_unique<Neighbour>());
      Neighbour* n = neighbours.back().get();
      NbModel* m = models[i];
      n->thread = std::thread([n, m, this, blockSize] { instance_loop(*n, m, input_, blockSize); });
    }
  }
  if (thrash)
  {
    neighbours.push_back(std::make_unique<Neighbour>());
    Neighbour* n = neighbours.back().get();
    const size_t bytes = static_cast<size_t>(thrashKiB) * 1024;
    n->thread = std::thread([n, bytes] { thrash_loop(*n, bytes); });
  }

  // --- Parity: instance 0's output, untimed, against solo's.
  reset();
  pass(out.data());
  if (reference != nullptr && reference->size() == frames)
  {
    r.parity = true;
    for (size_t i = 0; i < blockCount * bs; i++)
      if (std::memcmp(&out[i], &(*reference)[i], sizeof(float)) != 0)
        r.mismatchedSamples++;
  }
  if (capture != nullptr)
    *capture = out;

  // --- Timing.
  nbp::ProtocolConfig config;
  config.warmupSeconds = options_.warmupSeconds;
  config.timingWindowSeconds = options_.windowSeconds;
  config.acceptTolerance = options_.acceptTolerance;
  config.quiet = true;
  std::vector<std::vector<double>> passes;
  nbp::PassHooks hooks;
  hooks.reset = reset;
  hooks.run = [&] { return pass(nullptr); };
  hooks.onTimedPass = [&](size_t index) {
    if (index == 0)
      passes.clear();
    passes.push_back(blockNs);
  };
  const std::string label = scenario + "/" + submodel_name(submodel) + "/" + std::to_string(blockSize);
  const double duration = static_cast<double>(blockCount * bs) / sampleRate_;
  const nbp::Result timing = nbp::measure(label, config, duration, hooks, nullptr);

  for (auto& n : neighbours)
  {
    n->stop.store(true);
    n->thread.join();
  }
  if (parallel)
  {
    std::map<int, uint64_t> cpus;
    for (auto& n : neighbours)
      for (auto& [cpu, count] : n->cpus)
        cpus[cpu] += count;
    r.neighbourCpus = "{";
    for (auto& [cpu, count] : cpus)
      r.neighbourCpus += (r.neighbourCpus.size() > 1 ? "," : "") + ("\"" + std::to_string(cpu) + "\":") +
                         std::to_string(count);
    r.neighbourCpus += "}";
  }
  for (NbModel* m : models)
    nb_planar_destroy(m);

  r.ok = timing.succeeded;
  if (!r.ok)
  {
    r.failure = timing.failureReason;
    return r;
  }
  r.corePercent = timing.corePercent;
  r.spread = timing.spread;
  r.acceptedPasses = timing.acceptedIndices.size();
  std::vector<double> pooled;
  for (size_t index : timing.acceptedIndices)
    if (index < passes.size())
      pooled.insert(pooled.end(), passes[index].begin(), passes[index].end());
  r.blockMedianNs = quantile(pooled, 0.5);
  r.blockP99Ns = quantile(pooled, 0.99);
  return r;
}

} // namespace

// -------------------------------------------------------------------------------------

std::vector<ContentionResult> nb_contention_run(const ContentionOptions& options, NSData* model,
                                                const std::vector<float>& input, double sampleRate, FILE* log)
{
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  g_ticksToNs = static_cast<double>(tb.numer) / static_cast<double>(tb.denom);

  Runner runner(options, model, input, sampleRate, log);
  std::vector<ContentionResult> results;
  auto has = [&](const char* s) {
    return std::find(options.scenarios.begin(), options.scenarios.end(), s) != options.scenarios.end();
  };

  std::fprintf(log, "cache contention: agreement tolerance %.0f%%; scenarios", options.acceptTolerance * 100);
  for (const std::string& s : options.scenarios)
    std::fprintf(log, " %s", s.c_str());
  std::fprintf(log, "\n");

  for (NbSubmodel submodel : options.submodels)
  {
    for (int blockSize : options.blockSizes)
    {
      std::fprintf(log, "%s, %d frames:\n", submodel_name(submodel), blockSize);
      std::vector<float> reference;
      std::vector<ContentionResult> here;
      here.push_back(runner.run("solo", submodel, blockSize, 1, 0, nullptr, &reference));

      if (has("serial"))
        for (int n : options.instances)
          if (n > 1)
            here.push_back(runner.run("serial", submodel, blockSize, n, 0, &reference, nullptr));
      if (has("parallel"))
      {
        const int cores = static_cast<int>(std::thread::hardware_concurrency());
        for (int n : options.instances)
          if (n > 1 && n <= cores)
            here.push_back(runner.run("parallel", submodel, blockSize, n, 0, &reference, nullptr));
      }
      if (has("thrash"))
        for (int kib : options.thrashKiB)
          here.push_back(runner.run("thrash", submodel, blockSize, 1, kib, &reference, nullptr));

      const double solo = here.front().ok ? here.front().blockMedianNs : 0;
      for (ContentionResult& r : here)
      {
        r.slowdownMedian = (solo > 0 && r.ok) ? r.blockMedianNs / solo : 0;
        char what[64];
        if (r.scenario == "thrash")
          std::snprintf(what, sizeof(what), "thrash %6d KiB", r.thrashKiB);
        else
          std::snprintf(what, sizeof(what), "%-8s x%-2d", r.scenario.c_str(), r.instances);
        if (!r.ok)
          std::fprintf(log, "  %-16s FAILED: %s\n", what, r.failure.c_str());
        else
          std::fprintf(log,
                       "  %-16s core %7.3f%%  spread %5.2f%%  block med %8.0f p99 %8.0f ns  x%.2f  "
                       "instance %zu KiB  parity %s\n",
                       what, r.corePercent, r.spread * 100.0, r.blockMedianNs, r.blockP99Ns, r.slowdownMedian,
                       r.instanceBytes / 1024,
                       !r.parity ? "-" : (r.mismatchedSamples == 0 ? "bit-identical" : "DIFFERS"));
        std::fflush(log);
      }
      results.insert(results.end(), here.begin(), here.end());
    }
  }
  return results;
}

std::string nb_contention_json(const std::vector<ContentionResult>& results)
{
  std::string j = "[\n";
  for (size_t i = 0; i < results.size(); i++)
  {
    const ContentionResult& r = results[i];
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"scenario\":\"%s\",\"submodel\":\"%s\",\"blockSize\":%d,\"instances\":%d,\"thrashKiB\":%d,"
                  "\"ok\":%s,\"failure\":\"%s\",\"corePercent\":%.4f,\"spread\":%.4f,\"acceptedPasses\":%zu,"
                  "\"blockMedianNs\":%.1f,\"blockP99Ns\":%.1f,\"slowdownMedian\":%.4f,\"instanceBytes\":%zu,"
                  "\"parity\":%s,\"mismatchedSamples\":%zu,\"neighbourCpus\":%s}",
                  r.scenario.c_str(), submodel_name(r.submodel), r.blockSize, r.instances, r.thrashKiB,
                  r.ok ? "true" : "false", r.failure.c_str(), r.corePercent, r.spread, r.acceptedPasses,
                  r.blockMedianNs, r.blockP99Ns, r.slowdownMedian, r.instanceBytes, r.parity ? "true" : "false",
                  r.mismatchedSamples, r.neighbourCpus.c_str());
    j += buf;
    j += (i + 1 < results.size()) ? ",\n" : "\n";
  }
  j += "]";
  return j;
}
