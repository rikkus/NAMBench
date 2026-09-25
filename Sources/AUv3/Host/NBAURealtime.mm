// Arm F of the AUv3 overhead experiment: the same unit under a real host's
// real-time IO thread, rather than driven offline. See plans/auv3-overhead.md
// ("Real-time scheduling") and AUV3-PATH.md.
//
//   Fi  NBAudioUnit registered in this process (arm B's unit), in AVAudioEngine
//   F   the app extension, out of process, in AVAudioEngine
//
// The chain is AVAudioSourceNode (the input file, looped) -> the unit -> the
// main mixer (muted) -> the default output device, at 48 kHz with the device's
// IO buffer set to each block size in turn. Nothing is timed from a harness
// thread: the IO thread runs at the device's pace, and what is recorded is
// what happens to each IO cycle.
//
// Per IO cycle, from a render notify on the output unit: cycle start and end
// (mach ticks) and the output timestamp's sample time. A gap in sample time is
// a cycle the device played without us: an overrun. On macOS the HAL's
// processor-overload notification is counted as well. Per render call, from
// inside the unit: the same stamps arm D records, plus how the unit's render
// thread is scheduled (real-time policy or not) and whether the host handed it
// a workgroup. That last part is the question AUV3-PATH.md left open: on the
// M2 the kernel ran 10-40% slower in the extension offline, and the guess was
// that under a real IO thread the extension's thread joins the host's
// workgroup and runs at full speed.

#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <AudioToolbox/AudioWorkInterval.h>
#import <Foundation/Foundation.h>
#include <os/workgroup.h>
#if TARGET_OS_OSX
#import <CoreAudio/CoreAudio.h>
#endif

#include <mach/mach_time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#import <NAMBenchAU/NBAudioUnit.h>
#import <NAMEnginePlanar/NAMEnginePlanar.h>

#include "nb_protocol.h"
#include "nb_shim_timing.h"

#include "NBAURealtime.h"

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

RtDist dist_of(const std::vector<double>& v)
{
  RtDist d;
  if (v.empty())
    return d;
  double sum = 0;
  for (double x : v)
    sum += x;
  d.mean = sum / static_cast<double>(v.size());
  d.median = quantile(v, 0.5);
  d.p99 = quantile(v, 0.99);
  d.max = *std::max_element(v.begin(), v.end());
  return d;
}

// --- Clusters -----------------------------------------------------------------------
//
// Which CPU numbers are efficiency cores, measured rather than assumed. There
// is no public CPU-to-cluster map, and a background-QoS thread is not confined
// to the E cluster (it was seen on P cores too). So: one busy thread per CPU
// runs the same fixed chunk of dependent floating-point work for a while, and
// a CPU whose mean chunk time is more than 12% above the fastest CPU's is an
// E core. On the M2 that gives 0-3 (about 97 us a chunk against 77 us).

bool g_isE[256] = {};
std::string g_clusterReport;

void learn_e_cores()
{
  const unsigned n = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::atomic<uint64_t>> total(256), count(256);
  std::vector<std::thread> threads;
  const uint64_t until = mach_absolute_time() + static_cast<uint64_t>(1e9 / g_ticksToNs);
  for (unsigned t = 0; t < n; t++)
    threads.emplace_back([&] {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      volatile double x = 1.0;
      while (mach_absolute_time() < until)
      {
        const uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 20000; i++)
          x = x * 1.0000001 + 1e-9;
        const uint64_t t1 = mach_absolute_time();
        size_t cpu = 0;
        pthread_cpu_number_np(&cpu);
        if (cpu < 256)
        {
          total[cpu] += t1 - t0;
          count[cpu]++;
        }
      }
    });
  for (std::thread& t : threads)
    t.join();
  double fastest = 1e300;
  double mean[256] = {};
  for (int c = 0; c < 256; c++)
    if (count[c] > 0)
    {
      mean[c] = ticks_ns(total[c]) / static_cast<double>(count[c]);
      fastest = std::min(fastest, mean[c]);
    }
  for (int c = 0; c < 256; c++)
    if (count[c] > 0)
    {
      g_isE[c] = mean[c] > 1.12 * fastest;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%scpu%d %.0f%s", g_clusterReport.empty() ? "" : ", ", c, mean[c],
                    g_isE[c] ? " E" : " P");
      g_clusterReport += buf;
    }
}

struct ByCluster
{
  size_t onE = 0, onP = 0;
  double medianE = 0, medianP = 0;
};

ByCluster by_cluster(const std::vector<double>& ns, const std::vector<uint32_t>& cpus)
{
  std::vector<double> e, p;
  for (size_t i = 0; i < ns.size(); i++)
    (cpus[i] < 256 && g_isE[cpus[i]] ? e : p).push_back(ns[i]);
  return {e.size(), p.size(), quantile(e, 0.5), quantile(p, 0.5)};
}

// --- IO-thread state ---------------------------------------------------------------
//
// Everything the IO thread touches: preallocated, no locks, no Objective-C.

struct Cycle
{
  uint64_t start = 0;  // output unit pre-render
  uint64_t end = 0;    // output unit post-render
  uint64_t source = 0; // the source node's render block, last call in the cycle
  double sampleTime = 0;
  uint64_t hostTime = 0; // the output timestamp's host time: when this buffer plays
  uint32_t frames = 0;
  uint32_t sourceCalls = 0;
};

struct IoState
{
  std::vector<Cycle> cycles;
  std::atomic<size_t> count{0};

  const float* input = nullptr;
  size_t inputFrames = 0;
  size_t cursor = 0;

  NbThreadInfo ioThread{};
  bool ioThreadCaptured = false;

  std::atomic<uint32_t> overloads{0};
};

constexpr size_t kCycleCapacity = 1u << 18;

OSStatus render_notify(void* ctx, AudioUnitRenderActionFlags* flags, const AudioTimeStamp* ts, UInt32, UInt32 frames,
                       AudioBufferList*)
{
  IoState* io = static_cast<IoState*>(ctx);
  const size_t slot = io->count.load(std::memory_order_relaxed);
  if (slot >= io->cycles.size())
    return noErr;
  Cycle& c = io->cycles[slot];
  if (*flags & kAudioUnitRenderAction_PreRender)
  {
    c.start = mach_absolute_time();
    c.sampleTime = ts->mSampleTime;
    c.hostTime = (ts->mFlags & kAudioTimeStampHostTimeValid) ? ts->mHostTime : 0;
    c.frames = frames;
    c.source = 0;
    c.sourceCalls = 0;
    if (!io->ioThreadCaptured)
    {
      nb_capture_thread(&io->ioThread);
      io->ioThreadCaptured = true;
    }
  }
  else if (*flags & kAudioUnitRenderAction_PostRender)
  {
    c.end = mach_absolute_time();
    io->count.store(slot + 1, std::memory_order_release);
  }
  return noErr;
}

#if TARGET_OS_OSX
OSStatus overload_listener(AudioObjectID, UInt32, const AudioObjectPropertyAddress*, void* ctx)
{
  static_cast<IoState*>(ctx)->overloads.fetch_add(1, std::memory_order_relaxed);
  return noErr;
}
#endif

// --- Setup helpers ------------------------------------------------------------------------

AVAudioUnit* instantiate_node(bool outOfProcess, std::string& error)
{
  AudioComponentDescription desc{};
  desc.componentType = kAudioUnitType_Effect;
  desc.componentManufacturer = NB_AU_MANUFACTURER;
  desc.componentSubType = outOfProcess ? NB_AU_SUBTYPE_EXTENSION : NB_AU_SUBTYPE_INPROC;
  const AudioComponentInstantiationOptions options = outOfProcess ? kAudioComponentInstantiation_LoadOutOfProcess : 0;

  __block AVAudioUnit* node = nil;
  __block NSError* failure = nil;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [AVAudioUnit instantiateWithComponentDescription:desc
                                           options:options
                                 completionHandler:^(AVAudioUnit* au, NSError* err) {
                                   node = au;
                                   failure = err;
                                   dispatch_semaphore_signal(done);
                                 }];
  if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) != 0)
  {
    error = "instantiation timed out";
    return nil;
  }
  if (node == nil)
    error = failure ? failure.description.UTF8String : "instantiation returned nil";
  return node;
}

NSDictionary* call_unit(AUAudioUnit* unit, NSDictionary* message, std::string& error)
{
  id<NSObject, AUMessageChannel> channel = (id<NSObject, AUMessageChannel>)[unit messageChannelFor:kNbAUChannelName];
  if (channel == nil || ![channel respondsToSelector:@selector(callAudioUnit:)])
  {
    error = "unit has no message channel";
    return nil;
  }
  NSDictionary* reply = [channel callAudioUnit:message];
  if (reply == nil || [reply[kNbAUStatus] intValue] != 0)
  {
    error = reply ? [reply[kNbAUError] description].UTF8String : "no reply";
    return nil;
  }
  return reply;
}

RtThread thread_from(const NbThreadInfo& t)
{
  RtThread r;
  r.captured = t.captured != 0;
  r.timeConstraint = t.timeConstraint != 0;
  r.periodNs = ticks_ns(t.periodTicks);
  r.computationNs = ticks_ns(t.computationTicks);
  r.constraintNs = ticks_ns(t.constraintTicks);
  r.qos = t.qos;
  r.priority = t.priority;
  return r;
}

/// Bare kernel on this thread, per block, for reference: the median kernel
/// time of the same call the unit makes, on the same input, at the same block
/// size. Two passes over the file; the second is the one kept.
double bare_kernel_median_ns(NSData* model, NbSubmodel submodel, int blockSize, const std::vector<float>& input)
{
  char err[256] = {0};
  NbModel* m = nb_planar_create(static_cast<const uint8_t*>(model.bytes), model.length, submodel, 0, blockSize, err,
                                sizeof(err));
  if (m == nullptr)
    return 0;
  std::vector<float> out(static_cast<size_t>(blockSize));
  std::vector<double> times;
  for (int passIndex = 0; passIndex < 2; passIndex++)
  {
    times.clear();
    nb_planar_reset(m, 48000.0, blockSize);
    for (size_t offset = 0; offset + static_cast<size_t>(blockSize) <= input.size();
         offset += static_cast<size_t>(blockSize))
    {
      const uint64_t fpcr = denormals_disable();
      const uint64_t t0 = mach_absolute_time();
      nb_planar_process_block(m, input.data() + offset, out.data(), blockSize);
      const uint64_t t1 = mach_absolute_time();
      denormals_restore(fpcr);
      times.push_back(ticks_ns(t1 - t0));
    }
  }
  nb_planar_destroy(m);
  return quantile(times, 0.5);
}

/// The bare kernel paced like an IO thread: our own thread, given the
/// time-constraint policy the output device's IO thread was seen to have, runs
/// one block per period and sleeps until the next. If the kernel is as slow
/// here as under the engine, the slowdown is the duty cycle (clocks, cluster,
/// caches), not anything AudioToolbox does.
void paced_bare(NSData* model, NbSubmodel submodel, int blockSize, const std::vector<float>& input,
                const NbThreadInfo& ioPolicy, double seconds, bool workgroup, std::vector<double>& ns,
                std::vector<uint32_t>& cpus)
{
  std::thread t([&] {
    thread_time_constraint_policy_data_t policy{};
    policy.period = ioPolicy.periodTicks;
    policy.computation = ioPolicy.computationTicks;
    policy.constraint = ioPolicy.constraintTicks;
    policy.preemptible = 1;
    thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                      reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);

    os_workgroup_interval_t wg = nullptr;
    os_workgroup_join_token_s token{};
    if (workgroup)
    {
      wg = AudioWorkIntervalCreate("NAMBench paced", OS_CLOCK_MACH_ABSOLUTE_TIME, nullptr);
      if (wg != nullptr && os_workgroup_join(wg, &token) != 0)
        wg = nullptr;
    }

    char err[256] = {0};
    NbModel* m = nb_planar_create(static_cast<const uint8_t*>(model.bytes), model.length, submodel, 0, blockSize, err,
                                  sizeof(err));
    if (m == nullptr)
      return;
    std::vector<float> out(static_cast<size_t>(blockSize));
    const uint64_t periodTicks = static_cast<uint64_t>(blockSize / 48000.0 * 1e9 / g_ticksToNs);
    const uint64_t start = mach_absolute_time();
    const uint64_t recordFrom = start + static_cast<uint64_t>(1e9 / g_ticksToNs);
    const uint64_t until = recordFrom + static_cast<uint64_t>(seconds * 1e9 / g_ticksToNs);
    uint64_t next = start;
    size_t offset = 0;
    while (true)
    {
      next += periodTicks;
      mach_wait_until(next);
      if (next >= until)
        break;
      if (offset + static_cast<size_t>(blockSize) > input.size())
        offset = 0;
      if (wg != nullptr)
        os_workgroup_interval_start(wg, next, next + periodTicks, nullptr);
      const uint64_t fpcr = denormals_disable();
      const uint64_t t0 = mach_absolute_time();
      nb_planar_process_block(m, input.data() + offset, out.data(), blockSize);
      const uint64_t t1 = mach_absolute_time();
      denormals_restore(fpcr);
      if (wg != nullptr)
        os_workgroup_interval_finish(wg, nullptr);
      offset += static_cast<size_t>(blockSize);
      if (t0 >= recordFrom)
      {
        size_t cpu = 0;
        pthread_cpu_number_np(&cpu);
        ns.push_back(ticks_ns(t1 - t0));
        cpus.push_back(static_cast<uint32_t>(cpu));
      }
    }
    nb_planar_destroy(m);
    if (wg != nullptr)
      os_workgroup_leave(wg, &token);
  });
  t.join();
}

// --- One subject ------------------------------------------------------------------------

RtResult run_subject(const RtOptions& options, bool outOfProcess, NbSubmodel submodel, int blockSize, NSData* model,
                     const std::vector<float>& input, FILE* log)
{
  RtResult r;
  r.outOfProcess = outOfProcess;
  r.submodel = submodel;
  r.requestedFrames = blockSize;
  std::string error;

  r.bareKernelMedianNs = bare_kernel_median_ns(model, submodel, blockSize, input);

#if !TARGET_OS_OSX
  {
    AVAudioSession* session = AVAudioSession.sharedInstance;
    NSError* e = nil;
    [session setPreferredSampleRate:48000.0 error:&e];
    [session setPreferredIOBufferDuration:blockSize / 48000.0 error:&e];
    [session setActive:YES error:&e];
    r.sessionIOBufferFrames = static_cast<int>(std::lround(session.IOBufferDuration * session.sampleRate));
  }
#endif

  AVAudioUnit* node = instantiate_node(outOfProcess, error);
  if (node == nil)
  {
    r.failure = "instantiate: " + error;
    return r;
  }
  AUAudioUnit* unit = node.AUAudioUnit;
  if (!call_unit(unit, @{kNbAUCommand : kNbAUCmdConfigure, @"model" : model, @"submodel" : @(submodel)}, error))
  {
    r.failure = "configure: " + error;
    return r;
  }

  IoState io;
  io.cycles.assign(kCycleCapacity, Cycle{});
  io.input = input.data();
  io.inputFrames = input.size();
  IoState* ioPtr = &io;

  AVAudioFormat* mono = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:48000.0 channels:1];
  AVAudioSourceNode* source = [[AVAudioSourceNode alloc]
      initWithFormat:mono
         renderBlock:^OSStatus(BOOL*, const AudioTimeStamp*, AVAudioFrameCount frameCount, AudioBufferList* out) {
           float* dst = static_cast<float*>(out->mBuffers[0].mData);
           for (AVAudioFrameCount i = 0; i < frameCount; i++)
           {
             dst[i] = ioPtr->input[ioPtr->cursor];
             if (++ioPtr->cursor == ioPtr->inputFrames)
               ioPtr->cursor = 0;
           }
           const size_t slot = ioPtr->count.load(std::memory_order_relaxed);
           if (slot < ioPtr->cycles.size())
           {
             ioPtr->cycles[slot].source = mach_absolute_time();
             ioPtr->cycles[slot].sourceCalls++;
           }
           return noErr;
         }];

  AVAudioEngine* engine = [AVAudioEngine new];
  [engine attachNode:source];
  [engine attachNode:node];
  [engine connect:source to:node format:mono];
  [engine connect:node to:engine.mainMixerNode format:mono];
  // Silent, but still pulled: the mixer renders its inputs whatever its volume.
  // The cycle and render-call counts below confirm that.
  engine.mainMixerNode.outputVolume = 0.0f;

  AudioUnit output = engine.outputNode.audioUnit;
  AVAudioFormat* hw = [engine.outputNode outputFormatForBus:0];
  r.deviceSampleRate = hw.sampleRate;
  if (hw.sampleRate != 48000.0)
  {
    r.failure = "output device is not at 48 kHz (" + std::to_string(hw.sampleRate) + ")";
    return r;
  }

#if TARGET_OS_OSX
  AudioObjectID device = 0;
  UInt32 size = sizeof(device);
  AudioUnitGetProperty(output, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &device, &size);
  UInt32 frames = static_cast<UInt32>(blockSize);
  AudioUnitSetProperty(output, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &frames,
                       sizeof(frames));
  const AudioObjectPropertyAddress overloadAddr = {kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal,
                                                   kAudioObjectPropertyElementMain};
  AudioObjectAddPropertyListener(device, &overloadAddr, overload_listener, &io);
#endif
  AudioUnitAddRenderNotify(output, render_notify, &io);

  [engine prepare];
  NSError* startErr = nil;
  if (![engine startAndReturnError:&startErr])
  {
    r.failure = std::string("engine start: ") + (startErr ? startErr.description.UTF8String : "?");
    AudioUnitRemoveRenderNotify(output, render_notify, &io);
#if TARGET_OS_OSX
    AudioObjectRemovePropertyListener(device, &overloadAddr, overload_listener, &io);
#endif
    return r;
  }

#if TARGET_OS_OSX
  {
    UInt32 actual = 0;
    UInt32 sz = sizeof(actual);
    const AudioObjectPropertyAddress bufAddr = {kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal,
                                                kAudioObjectPropertyElementMain};
    AudioObjectGetPropertyData(device, &bufAddr, 0, nullptr, &sz, &actual);
    r.deviceBufferFrames = static_cast<int>(actual);
  }
#endif

  const uint64_t runStart = mach_absolute_time();
  std::this_thread::sleep_for(std::chrono::duration<double>(options.warmupSeconds));
  const uint64_t measureStart = mach_absolute_time();
  const uint32_t overloadsAtStart = io.overloads.load();
  std::this_thread::sleep_for(std::chrono::duration<double>(options.seconds));
  const uint64_t measureEnd = mach_absolute_time();
  const uint32_t overloadsAtEnd = io.overloads.load();
  (void)runStart;

  NSDictionary* info = call_unit(unit, @{kNbAUCommand : kNbAUCmdInfo}, error);
  NSDictionary* thread = call_unit(unit, @{kNbAUCommand : kNbAUCmdThread}, error);
  [engine stop];
  AudioUnitRemoveRenderNotify(output, render_notify, &io);
#if TARGET_OS_OSX
  AudioObjectRemovePropertyListener(device, &overloadAddr, overload_listener, &io);
#endif
  NSDictionary* stampsReply = call_unit(unit, @{kNbAUCommand : kNbAUCmdStamps}, error);

  if (info == nil || thread == nil || stampsReply == nil)
  {
    r.failure = "unit query after run: " + error;
    return r;
  }
  r.unitPid = [info[@"pid"] intValue];
  r.engine = [info[@"engine"] intValue];
  r.unitMaxFrames = [info[@"blockSize"] intValue];
  r.unitSampleRate = [info[@"busSampleRate"] doubleValue];
  r.ranOutOfProcess = r.unitPid != getpid();

  NbThreadInfo ut{};
  ut.captured = [thread[@"captured"] boolValue];
  ut.timeConstraint = [thread[@"timeConstraint"] intValue];
  ut.periodTicks = [thread[@"periodTicks"] unsignedIntValue];
  ut.computationTicks = [thread[@"computationTicks"] unsignedIntValue];
  ut.constraintTicks = [thread[@"constraintTicks"] unsignedIntValue];
  ut.qos = [thread[@"qos"] intValue];
  ut.priority = [thread[@"priority"] intValue];
  r.unitThread = thread_from(ut);
  r.ioThread = thread_from(io.ioThread);
  r.contextObserverCalls = [thread[@"contextObserverCalls"] unsignedIntValue];
  r.workgroupSeen = [thread[@"workgroup"] boolValue];

  if (r.engine != NbEnginePlanar)
  {
    r.failure = "engine is not a2_planar";
    return r;
  }
  if (r.unitSampleRate != 48000.0)
  {
    r.failure = "unit bus is not at 48 kHz";
    return r;
  }
  if (r.ranOutOfProcess != outOfProcess)
  {
    r.failure = outOfProcess ? "F ran in-process" : "Fi ran out of process";
    return r;
  }

  // --- Cycles inside the measurement window --------------------------------------------
  const size_t cycleCount = io.count.load(std::memory_order_acquire);
  std::vector<const Cycle*> window;
  for (size_t i = 0; i < cycleCount; i++)
  {
    const Cycle& c = io.cycles[i];
    if (c.start >= measureStart && c.end <= measureEnd)
      window.push_back(&c);
  }
  if (window.size() < 2)
  {
    r.failure = "no IO cycles recorded";
    return r;
  }
  r.cycles = window.size();
  r.ioFrames = static_cast<int>(window.front()->frames);
  const double periodNs = r.ioFrames / 48000.0 * 1e9;
  r.periodNs = periodNs;
  r.expectedCycles = ticks_ns(measureEnd - measureStart) / periodNs;
  r.overloads = overloadsAtEnd - overloadsAtStart;

  std::vector<double> cycleNs, startJitterNs, sourceNs, slackNs;
  double prevStartNs = 0;
  for (size_t i = 0; i < window.size(); i++)
  {
    const Cycle& c = *window[i];
    const double d = ticks_ns(c.end - c.start);
    cycleNs.push_back(d);
    if (d > periodNs)
      r.cyclesOverPeriod++;
    if (d > 0.5 * periodNs)
      r.cyclesOverHalfPeriod++;
    if (c.frames != static_cast<uint32_t>(r.ioFrames))
      r.oddFrameCycles++;
    if (c.source != 0)
      sourceNs.push_back(ticks_ns(c.source - c.start));
    if (c.hostTime != 0)
      slackNs.push_back(static_cast<double>(static_cast<int64_t>(c.hostTime - c.start)) * g_ticksToNs);
    const double startNs = ticks_ns(c.start);
    if (i > 0)
    {
      const Cycle& p = *window[i - 1];
      const double expected = p.sampleTime + p.frames;
      if (std::fabs(c.sampleTime - expected) > 0.5)
      {
        r.discontinuities++;
        r.framesSkipped += std::max(0.0, c.sampleTime - expected);
        if (r.gaps.size() < 200)
        {
          RtGap g;
          g.atSeconds = ticks_ns(c.start - measureStart) / 1e9;
          g.frames = c.sampleTime - expected;
          g.startToStartNs = ticks_ns(c.start - p.start);
          g.prevCycleNs = ticks_ns(p.end - p.start);
          g.prevSlackNs = p.hostTime ? static_cast<double>(static_cast<int64_t>(p.hostTime - p.start)) * g_ticksToNs : 0;
          g.slackNs = c.hostTime ? static_cast<double>(static_cast<int64_t>(c.hostTime - c.start)) * g_ticksToNs : 0;
          r.gaps.push_back(g);
        }
      }
      startJitterNs.push_back(startNs - prevStartNs - periodNs);
    }
    prevStartNs = startNs;
  }
  r.cycleNs = dist_of(cycleNs);
  r.cycleStartJitterNs = dist_of(startJitterNs);
  r.sourceAfterStartNs = dist_of(sourceNs);
  r.slackNs = dist_of(slackNs);
  if (!slackNs.empty())
    r.slackMinNs = *std::min_element(slackNs.begin(), slackNs.end());

  // --- Render calls inside the window, matched to their cycles -----------------------------
  NSData* stampBytes = stampsReply[@"stamps"];
  const size_t stampCount = [stampsReply[@"count"] unsignedLongValue];
  const NbAUStamp* s = static_cast<const NbAUStamp*>(stampBytes.bytes);
  std::vector<double> kernel, entryAfterStart, exitBeforeEnd, wrapper;
  std::vector<uint32_t> kernelCpu;
  size_t ci = 0;
  for (size_t i = 0; i < stampCount; i++)
  {
    const NbAUStamp& st = s[i];
    if (st.entry < measureStart || st.exit > measureEnd)
      continue;
    r.renderCalls++;
    kernel.push_back(ticks_ns(st.kernelEnd - st.kernelStart));
    kernelCpu.push_back(st.cpu);
    wrapper.push_back(ticks_ns((st.exit - st.entry) - (st.kernelEnd - st.kernelStart)));
    while (ci < window.size() && window[ci]->end < st.entry)
      ci++;
    if (ci < window.size() && window[ci]->start <= st.entry && st.exit <= window[ci]->end)
    {
      entryAfterStart.push_back(
        static_cast<double>(static_cast<int64_t>(st.entry - window[ci]->start)) * g_ticksToNs);
      exitBeforeEnd.push_back(static_cast<double>(static_cast<int64_t>(window[ci]->end - st.exit)) * g_ticksToNs);
    }
    else
      r.unmatchedCalls++;
  }
  r.kernelNs = dist_of(kernel);
  r.wrapperNs = dist_of(wrapper);
  r.entryAfterCycleStartNs = dist_of(entryAfterStart);
  r.exitBeforeCycleEndNs = dist_of(exitBeforeEnd);
  {
    size_t perCpu[256] = {};
    for (uint32_t c : kernelCpu)
      if (c < 256)
        perCpu[c]++;
    r.callsByCpu = "{";
    for (int c = 0; c < 256; c++)
      if (perCpu[c] > 0)
        r.callsByCpu += (r.callsByCpu.size() > 1 ? "," : "") + ("\"" + std::to_string(c) + "\":") +
                        std::to_string(perCpu[c]);
    r.callsByCpu += "}";
  }
  const ByCluster kc = by_cluster(kernel, kernelCpu);
  r.callsOnE = kc.onE;
  r.callsOnP = kc.onP;
  r.kernelMedianOnE = kc.medianE;
  r.kernelMedianOnP = kc.medianP;

  // The paced reference, after the engine has stopped, with the IO thread's
  // own policy.
  if (io.ioThreadCaptured)
  {
    std::vector<double> pacedNs;
    std::vector<uint32_t> pacedCpu;
    paced_bare(model, submodel, blockSize, input, io.ioThread, options.pacedSeconds, false, pacedNs, pacedCpu);
    r.pacedKernelNs = dist_of(pacedNs);
    const ByCluster pc = by_cluster(pacedNs, pacedCpu);
    r.pacedOnE = pc.onE;
    r.pacedOnP = pc.onP;
    r.pacedMedianOnE = pc.medianE;
    r.pacedMedianOnP = pc.medianP;

    pacedNs.clear();
    pacedCpu.clear();
    paced_bare(model, submodel, blockSize, input, io.ioThread, options.pacedSeconds, true, pacedNs, pacedCpu);
    r.pacedWgKernelNs = dist_of(pacedNs);
    const ByCluster wc = by_cluster(pacedNs, pacedCpu);
    r.pacedWgOnE = wc.onE;
    r.pacedWgOnP = wc.onP;
    r.pacedWgMedianOnE = wc.medianE;
    r.pacedWgMedianOnP = wc.medianP;
  }

  r.ok = true;
  (void)log;
  return r;
}

const char* qos_name(int qos)
{
  switch (qos)
  {
  case QOS_CLASS_USER_INTERACTIVE: return "UI";
  case QOS_CLASS_USER_INITIATED: return "IN";
  case QOS_CLASS_DEFAULT: return "DF";
  case QOS_CLASS_UTILITY: return "UT";
  case QOS_CLASS_BACKGROUND: return "BG";
  case QOS_CLASS_UNSPECIFIED: return "--";
  }
  return "?";
}

std::string thread_json(const RtThread& t)
{
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"captured\":%s,\"timeConstraint\":%s,\"periodNs\":%.0f,\"computationNs\":%.0f,"
                "\"constraintNs\":%.0f,\"qos\":\"%s\",\"priority\":%d}",
                t.captured ? "true" : "false", t.timeConstraint ? "true" : "false", t.periodNs, t.computationNs,
                t.constraintNs, qos_name(t.qos), t.priority);
  return buf;
}

std::string dist_json(const char* name, const RtDist& d)
{
  char buf[256];
  std::snprintf(buf, sizeof(buf), "\"%s\":{\"median\":%.1f,\"p99\":%.1f,\"max\":%.1f,\"mean\":%.1f}", name, d.median,
                d.p99, d.max, d.mean);
  return buf;
}

} // namespace

// -------------------------------------------------------------------------------------------

std::vector<RtResult> nb_au_realtime_run(const RtOptions& options, NSData* model, const std::vector<float>& input,
                                         FILE* log)
{
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  g_ticksToNs = static_cast<double>(tb.numer) / static_cast<double>(tb.denom);

#if !TARGET_OS_OSX
  {
    NSError* e = nil;
    [AVAudioSession.sharedInstance setCategory:AVAudioSessionCategoryPlayback error:&e];
  }
#endif

  learn_e_cores();
  std::fprintf(log, "clusters by chunk time (ns): %s\n", g_clusterReport.c_str());
  // The probe loads every core; let clocks fall back before anything is timed.
  std::this_thread::sleep_for(std::chrono::duration<double>(options.settleSeconds));

  std::fprintf(log, "arm F: real-time AVAudioEngine output, %.0f s per subject after %.0f s warm-up\n",
               options.seconds, options.warmupSeconds);
  std::vector<RtResult> results;
  for (NbSubmodel submodel : options.submodels)
  {
    for (int blockSize : options.blockSizes)
    {
      for (bool oop : options.modes)
      {
        RtResult r = run_subject(options, oop, submodel, blockSize, model, input, log);
        const char* sm = submodel == NbSubmodelNarrowest ? "nano" : "standard";
        if (!r.ok)
          std::fprintf(log, "  %-2s %-8s %4d  FAILED: %s\n", oop ? "F" : "Fi", sm, blockSize, r.failure.c_str());
        else
          std::fprintf(log,
                       "  %-2s %-8s io %4d  cycles %6zu/%6.0f  discont %zu (%.0f frames)  overloads %u  "
                       "over-period %zu  cycle med %6.0f p99 %6.0f max %7.0f ns (%.1f%% / %.1f%% of %.0f)  "
                       "kernel med %6.0f (bare %6.0f, paced %6.0f, paced+wg %6.0f)  calls E/P %zu/%zu  unit RT %d wg %d obs %u  "
                       "io RT %d\n",
                       oop ? "F" : "Fi", sm, r.ioFrames, r.cycles, r.expectedCycles, r.discontinuities,
                       r.framesSkipped, r.overloads, r.cyclesOverPeriod, r.cycleNs.median, r.cycleNs.p99,
                       r.cycleNs.max, r.cycleNs.median / r.periodNs * 100, r.cycleNs.p99 / r.periodNs * 100,
                       r.periodNs, r.kernelNs.median, r.bareKernelMedianNs, r.pacedKernelNs.median, r.pacedWgKernelNs.median, r.callsOnE,
                       r.callsOnP, r.unitThread.timeConstraint,
                       r.workgroupSeen, r.contextObserverCalls, r.ioThread.timeConstraint);
        std::fflush(log);
        results.push_back(std::move(r));
      }
    }
  }
  return results;
}

std::string nb_au_realtime_json(const std::vector<RtResult>& results)
{
  std::string j = "[\n";
  for (size_t i = 0; i < results.size(); i++)
  {
    const RtResult& r = results[i];
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"arm\":\"%s\",\"submodel\":\"%s\",\"requestedFrames\":%d,\"ok\":%s,\"failure\":\"%s\","
                  "\"outOfProcess\":%s,\"ioFrames\":%d,\"deviceBufferFrames\":%d,\"sessionIOBufferFrames\":%d,"
                  "\"unitMaxFrames\":%d,\"deviceSampleRate\":%.0f,\"periodNs\":%.1f,\"cycles\":%zu,"
                  "\"expectedCycles\":%.1f,\"renderCalls\":%zu,\"unmatchedCalls\":%zu,\"discontinuities\":%zu,"
                  "\"framesSkipped\":%.0f,\"overloads\":%u,\"cyclesOverPeriod\":%zu,\"cyclesOverHalfPeriod\":%zu,"
                  "\"oddFrameCycles\":%zu,\"bareKernelMedianNs\":%.1f,\"contextObserverCalls\":%u,"
                  "\"workgroupSeen\":%s,",
                  r.outOfProcess ? "F" : "Fi", r.submodel == NbSubmodelNarrowest ? "nano" : "standard",
                  r.requestedFrames, r.ok ? "true" : "false", r.failure.c_str(), r.outOfProcess ? "true" : "false",
                  r.ioFrames, r.deviceBufferFrames, r.sessionIOBufferFrames, r.unitMaxFrames, r.deviceSampleRate,
                  r.periodNs, r.cycles, r.expectedCycles, r.renderCalls, r.unmatchedCalls, r.discontinuities,
                  r.framesSkipped, r.overloads, r.cyclesOverPeriod, r.cyclesOverHalfPeriod, r.oddFrameCycles,
                  r.bareKernelMedianNs, r.contextObserverCalls, r.workgroupSeen ? "true" : "false");
    j += buf;
    j += "\"unitThread\":" + thread_json(r.unitThread) + ",\"ioThread\":" + thread_json(r.ioThread) + ",";
    j += dist_json("cycleNs", r.cycleNs) + "," + dist_json("cycleStartJitterNs", r.cycleStartJitterNs) + "," +
         dist_json("sourceAfterStartNs", r.sourceAfterStartNs) + "," + dist_json("kernelNs", r.kernelNs) + "," +
         dist_json("wrapperNs", r.wrapperNs) + "," + dist_json("entryAfterCycleStartNs", r.entryAfterCycleStartNs) +
         "," + dist_json("exitBeforeCycleEndNs", r.exitBeforeCycleEndNs) + "," +
         dist_json("pacedKernelNs", r.pacedKernelNs) + "," + dist_json("pacedWgKernelNs", r.pacedWgKernelNs);
    std::snprintf(buf, sizeof(buf),
                  ",\"callsOnE\":%zu,\"callsOnP\":%zu,\"kernelMedianOnE\":%.1f,\"kernelMedianOnP\":%.1f,"
                  "\"pacedOnE\":%zu,\"pacedOnP\":%zu,\"pacedMedianOnE\":%.1f,\"pacedMedianOnP\":%.1f,"
                  "\"pacedWgOnE\":%zu,\"pacedWgOnP\":%zu,\"pacedWgMedianOnE\":%.1f,\"pacedWgMedianOnP\":%.1f",
                  r.callsOnE, r.callsOnP, r.kernelMedianOnE, r.kernelMedianOnP, r.pacedOnE, r.pacedOnP,
                  r.pacedMedianOnE, r.pacedMedianOnP, r.pacedWgOnE, r.pacedWgOnP, r.pacedWgMedianOnE,
                  r.pacedWgMedianOnP);
    j += buf;
    j += ",\"callsByCpu\":" + r.callsByCpu;
    j += "," + dist_json("slackNs", r.slackNs);
    std::snprintf(buf, sizeof(buf), ",\"slackMinNs\":%.0f,\"gaps\":[", r.slackMinNs);
    j += buf;
    for (size_t g = 0; g < r.gaps.size(); g++)
    {
      const RtGap& x = r.gaps[g];
      std::snprintf(buf, sizeof(buf),
                    "%s{\"at\":%.4f,\"frames\":%.0f,\"startToStartNs\":%.0f,\"prevCycleNs\":%.0f,"
                    "\"prevSlackNs\":%.0f,\"slackNs\":%.0f}",
                    g ? "," : "", x.atSeconds, x.frames, x.startToStartNs, x.prevCycleNs, x.prevSlackNs, x.slackNs);
      j += buf;
    }
    j += "]";
    j += (i + 1 < results.size()) ? "},\n" : "}\n";
  }
  j += "]";
  return j;
}
