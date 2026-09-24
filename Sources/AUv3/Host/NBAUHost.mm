// The AUv3 overhead host: runs one kernel through each wrapper layer in turn
// and reports what each layer costs. See plans/auv3-overhead.md.
//
//   A    nb_planar_process over the whole file — the number NAMBench publishes
//   A1   the bare harness calling nb_planar_process_block once per block (A′)
//   B    NBAudioUnit registered inside this process, driven via renderBlock
//   C    the same unit from the app extension, loaded in-process (macOS only)
//   D    the same extension, out of process
//
// On iOS, D is arm E of the plan: the same code on a device.
//
// Everything runs offline on one dedicated thread, pulling blocks as fast as
// they will go. There is no audio device and no AVAudioEngine anywhere, so
// nothing in the chain has a chance to resample: every arm runs 48 kHz end to
// end, and the host checks that rather than assuming it.

#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#import <NAMBenchAU/NBAudioUnit.h>
#import <NAMEnginePlanar/NAMEnginePlanar.h>

#include "nb_protocol.h"
#include "nb_shim_timing.h"

#include "NBAUHost.h"
#include "NBAURealtime.h"

namespace
{

// --- Clock -------------------------------------------------------------------

double g_ticksToNs = 1.0;

void init_timebase()
{
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  g_ticksToNs = static_cast<double>(info.numer) / static_cast<double>(info.denom);
}

inline double ticks_ns(uint64_t ticks)
{
  return static_cast<double>(ticks) * g_ticksToNs;
}

// --- Output ------------------------------------------------------------------

FILE* g_log = stdout;

void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void logf(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  std::vfprintf(g_log, fmt, args);
  va_end(args);
  std::fflush(g_log);
}

// --- Per-block record, common to every arm -------------------------------------
//
// Host stamps bracket the call the host makes: renderBlock for the AU arms, the
// shim call for A1. Unit stamps come from inside: for the AU arms from the
// render block (fetched over the message channel after each pass), for A1 from
// the harness itself around the same calls in the same order.

struct Block
{
  uint64_t hostStart = 0;
  uint64_t hostEnd = 0;
  NbAUStamp unit{};
};

// --- Arms ----------------------------------------------------------------------

enum class Arm
{
  A,
  A1,
  B,
  C,
  D
};

const char* arm_name(Arm arm)
{
  switch (arm)
  {
  case Arm::A: return "A";
  case Arm::A1: return "A1";
  case Arm::B: return "B";
  case Arm::C: return "C";
  case Arm::D: return "D";
  }
  return "?";
}

const char* arm_description(Arm arm)
{
  switch (arm)
  {
  case Arm::A: return "shim whole-file loop (published number)";
  case Arm::A1: return "bare per-block shim call";
  case Arm::B: return "AUAudioUnit subclass in host";
  case Arm::C: return "AUv3 extension, in-process";
  case Arm::D: return "AUv3 extension, out-of-process";
  }
  return "?";
}

// --- Stats ---------------------------------------------------------------------

double quantile(std::vector<double> v, double q)
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

struct Dist
{
  double median = 0, p99 = 0, mean = 0;
};

Dist dist_of(const std::vector<double>& v)
{
  Dist d;
  if (v.empty())
    return d;
  double sum = 0;
  for (double x : v)
    sum += x;
  d.mean = sum / static_cast<double>(v.size());
  d.median = quantile(v, 0.5);
  d.p99 = quantile(v, 0.99);
  return d;
}

// --- AU instantiation -----------------------------------------------------------

AUAudioUnit* instantiate(Arm arm, std::string& error)
{
  AudioComponentDescription desc{};
  desc.componentType = kAudioUnitType_Effect;
  desc.componentManufacturer = NB_AU_MANUFACTURER;
  desc.componentSubType = (arm == Arm::B) ? NB_AU_SUBTYPE_INPROC : NB_AU_SUBTYPE_EXTENSION;

  AudioComponentInstantiationOptions options = 0;
#if TARGET_OS_OSX
  if (arm == Arm::C)
    options = kAudioComponentInstantiation_LoadInProcess;
#else
  if (arm == Arm::C)
  {
    error = "arm C (in-process extension) is macOS-only";
    return nil;
  }
#endif
  if (arm == Arm::D)
    options = kAudioComponentInstantiation_LoadOutOfProcess;

  __block AUAudioUnit* unit = nil;
  __block NSError* failure = nil;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [AUAudioUnit instantiateWithComponentDescription:desc
                                           options:options
                                 completionHandler:^(AUAudioUnit* au, NSError* err) {
                                   unit = au;
                                   failure = err;
                                   dispatch_semaphore_signal(done);
                                 }];
  if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) != 0)
  {
    error = "instantiation timed out";
    return nil;
  }
  if (unit == nil)
    error = failure ? failure.description.UTF8String : "instantiation returned nil";
  return unit;
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

// --- One subject: an arm, a submodel, a block size -------------------------------

struct SubjectResult
{
  Arm arm = Arm::A;
  NbSubmodel submodel = NbSubmodelWidest;
  int blockSize = 64;
  bool ok = false;
  std::string failure;

  // Proof of where it ran and what it ran.
  int unitPid = 0;
  bool outOfProcess = false;
  int engine = 0;
  int channels = 0;
  double sampleRate = 0;

  nbp::Result timing;
  size_t blockSamples = 0;
  Dist hostBlockNs, kernelNs, overheadNs, hopInNs, wrapInNs, wrapOutNs, hopOutNs;

  // Output compared against A1 (the same float kernel, no wrapper).
  bool haveParity = false;
  double maxAbsDiff = 0;
  size_t mismatchedSamples = 0;
};

class Runner
{
public:
  Runner(const HostOptions& options, const nbp::Audio& audio, NSData* model)
      : options_(options), audio_(audio), model_(model)
  {
    inputF_.resize(audio.samples.size());
    for (size_t i = 0; i < audio.samples.size(); i++)
      inputF_[i] = static_cast<float>(audio.samples[i]);
  }

  SubjectResult run(Arm arm, NbSubmodel submodel, int blockSize, const std::vector<float>* reference,
                    std::vector<float>* outputCapture);

private:
  // Per-block timing loop for A1 and the AU arms. `out` null means the timed
  // passes' scratch; non-null captures the whole output for parity.
  uint64_t pass(float* out);

  const HostOptions& options_;
  const nbp::Audio& audio_;
  NSData* model_;
  std::vector<float> inputF_;

  // State for the pass in progress.
  Arm arm_ = Arm::A;
  int blockSize_ = 64;
  NbModel* shimModel_ = nullptr;
  AUAudioUnit* unit_ = nil;
  AURenderBlock renderBlock_ = nil;
  std::vector<Block> blocks_;
  std::vector<float> scratchOut_;
};

uint64_t Runner::pass(float* out)
{
  const size_t frames = inputF_.size();
  const size_t bs = static_cast<size_t>(blockSize_);
  Block* rec = blocks_.data();

  if (arm_ == Arm::A1)
  {
    const uint64_t start = mach_absolute_time();
    size_t b = 0;
    for (size_t offset = 0; offset < frames; offset += bs, b++)
    {
      const int32_t count = static_cast<int32_t>(std::min(bs, frames - offset));
      float* dst = out ? out + offset : scratchOut_.data();
      Block& r = rec[b];
      r.hostStart = mach_absolute_time();
      // The same sequence the render block runs around the same call, so that
      // "kernel" means the same thing in every arm.
      r.unit.entry = r.hostStart;
      const uint64_t previousFpcr = denormals_disable();
      r.unit.kernelStart = mach_absolute_time();
      nb_planar_process_block(shimModel_, inputF_.data() + offset, dst, count);
      r.unit.kernelEnd = mach_absolute_time();
      denormals_restore(previousFpcr);
      r.unit.exit = mach_absolute_time();
      r.hostEnd = r.unit.exit;
    }
    return static_cast<uint64_t>(ticks_ns(mach_absolute_time() - start));
  }

  // AU arms. The pull-input block hands the unit the next slice of the file,
  // exactly as a host hands a unit its input: by copying into the buffer the
  // unit supplies. It captures a raw pointer and an offset cell, nothing
  // Objective-C.
  __block size_t cursor = 0;
  const float* source = inputF_.data();
  AURenderPullInputBlock pull = ^AUAudioUnitStatus(AudioUnitRenderActionFlags*, const AudioTimeStamp*,
                                                   AUAudioFrameCount frameCount, NSInteger,
                                                   AudioBufferList* inputData) {
    AudioBuffer& buf = inputData->mBuffers[0];
    if (buf.mData == nullptr)
      buf.mData = const_cast<float*>(source + cursor);
    else
      std::memcpy(buf.mData, source + cursor, frameCount * sizeof(float));
    buf.mDataByteSize = frameCount * sizeof(float);
    return noErr;
  };

  AudioBufferList outList{};
  outList.mNumberBuffers = 1;
  outList.mBuffers[0].mNumberChannels = 1;

  AudioTimeStamp ts{};
  ts.mFlags = kAudioTimeStampSampleTimeValid;

  const uint64_t start = mach_absolute_time();
  size_t b = 0;
  for (size_t offset = 0; offset < frames; offset += bs, b++)
  {
    const AUAudioFrameCount count = static_cast<AUAudioFrameCount>(std::min(bs, frames - offset));
    cursor = offset;
    outList.mBuffers[0].mData = out ? out + offset : scratchOut_.data();
    outList.mBuffers[0].mDataByteSize = count * sizeof(float);
    ts.mSampleTime = static_cast<Float64>(offset);
    AudioUnitRenderActionFlags flags = 0;

    Block& r = rec[b];
    r.hostStart = mach_absolute_time();
    const AUAudioUnitStatus status = renderBlock_(&flags, &ts, count, 0, &outList, pull);
    r.hostEnd = mach_absolute_time();
    if (status != noErr)
    {
      logf("render failed at block %zu: %d\n", b, static_cast<int>(status));
      return 0;
    }
    // An out-of-process unit may render into its own memory and hand back a
    // pointer instead of filling ours; a host has to honour that.
    if (out && outList.mBuffers[0].mData != out + offset)
      std::memcpy(out + offset, outList.mBuffers[0].mData, count * sizeof(float));
  }
  return static_cast<uint64_t>(ticks_ns(mach_absolute_time() - start));
}

SubjectResult Runner::run(Arm arm, NbSubmodel submodel, int blockSize, const std::vector<float>* reference,
                          std::vector<float>* outputCapture)
{
  SubjectResult result;
  result.arm = arm;
  result.submodel = submodel;
  result.blockSize = blockSize;

  arm_ = arm;
  blockSize_ = blockSize;
  const size_t frames = inputF_.size();
  const size_t blockCount = (frames + static_cast<size_t>(blockSize) - 1) / static_cast<size_t>(blockSize);
  blocks_.assign(blockCount, Block{});
  scratchOut_.assign(static_cast<size_t>(blockSize), 0.0f);

  char err[512] = {0};
  std::string error;
  std::vector<double> reference64;

  // --- Build the subject -------------------------------------------------------
  if (arm == Arm::A || arm == Arm::A1)
  {
    shimModel_ = nb_planar_create(static_cast<const uint8_t*>(model_.bytes), model_.length, submodel, 0, blockSize,
                                  err, sizeof(err));
    if (shimModel_ == nullptr)
    {
      result.failure = err;
      return result;
    }
    result.unitPid = getpid();
    result.engine = nb_planar_engine(shimModel_);
    result.channels = nb_planar_channels(shimModel_);
    result.sampleRate = nb_planar_sample_rate(shimModel_);
  }
  else
  {
    unit_ = instantiate(arm, error);
    if (unit_ == nil)
    {
      result.failure = "instantiate: " + error;
      return result;
    }
    unit_.maximumFramesToRender = static_cast<AUAudioFrameCount>(blockSize);
    AVAudioFormat* format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:audio_.sampleRate channels:1];
    NSError* nsErr = nil;
    if (![unit_.inputBusses[0] setFormat:format error:&nsErr] || ![unit_.outputBusses[0] setFormat:format error:&nsErr])
    {
      result.failure = std::string("setFormat: ") + (nsErr ? nsErr.description.UTF8String : "?");
      unit_ = nil;
      return result;
    }
    if (!call_unit(unit_, @{kNbAUCommand : kNbAUCmdConfigure, @"model" : model_, @"submodel" : @(submodel)}, error))
    {
      result.failure = "configure: " + error;
      unit_ = nil;
      return result;
    }
    if (![unit_ allocateRenderResourcesAndReturnError:&nsErr])
    {
      result.failure = std::string("allocate: ") + (nsErr ? nsErr.description.UTF8String : "?");
      unit_ = nil;
      return result;
    }
    NSDictionary* info = call_unit(unit_, @{kNbAUCommand : kNbAUCmdInfo}, error);
    if (info == nil)
    {
      result.failure = "info: " + error;
      [unit_ deallocateRenderResources];
      unit_ = nil;
      return result;
    }
    result.unitPid = [info[@"pid"] intValue];
    result.engine = [info[@"engine"] intValue];
    result.channels = [info[@"channels"] intValue];
    result.sampleRate = [info[@"busSampleRate"] doubleValue];
    renderBlock_ = unit_.renderBlock;
  }
  result.outOfProcess = result.unitPid != getpid();

  // --- Assertions: what ran, where, at what rate --------------------------------
  auto fail_and_clean = [&](const std::string& why) {
    result.failure = why;
    if (shimModel_)
    {
      nb_planar_destroy(shimModel_);
      shimModel_ = nullptr;
    }
    if (unit_)
    {
      [unit_ deallocateRenderResources];
      unit_ = nil;
      renderBlock_ = nil;
    }
    return result;
  };
  if (result.engine != NbEnginePlanar)
    return fail_and_clean("engine is not a2_planar (" + std::to_string(result.engine) + ")");
  if (result.sampleRate != 48000.0 || audio_.sampleRate != 48000.0)
    return fail_and_clean("not 48 kHz end to end");
  const bool wantOut = (arm == Arm::D);
  if (arm != Arm::A && result.outOfProcess != wantOut)
    return fail_and_clean(wantOut ? "arm D ran in-process" : "in-process arm ran out of process");

  auto reset = [&] {
    if (arm == Arm::A || arm == Arm::A1)
      nb_planar_reset(shimModel_, 48000.0, blockSize);
    else
      call_unit(unit_, @{kNbAUCommand : kNbAUCmdReset}, error);
  };

  // --- Parity: one untimed pass capturing the output -----------------------------
  if (arm != Arm::A)
  {
    std::vector<float> out(frames, 0.0f);
    reset();
    pass(out.data());
    if (reference != nullptr && reference->size() == frames)
    {
      result.haveParity = true;
      for (size_t i = 0; i < frames; i++)
      {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>((*reference)[i]));
        result.maxAbsDiff = std::max(result.maxAbsDiff, d);
        if (std::memcmp(&out[i], &(*reference)[i], sizeof(float)) != 0)
          result.mismatchedSamples++;
      }
    }
    if (outputCapture)
      *outputCapture = std::move(out);
  }

  // --- Timing ------------------------------------------------------------------------
  nbp::ProtocolConfig config;
  config.warmupSeconds = options_.warmupSeconds;
  config.timingWindowSeconds = options_.windowSeconds;
  config.quiet = true;

  // Per timed pass, the per-block records, so the protocol's window selection
  // can be applied to them. A new attempt restarts pass indices at zero, and
  // its passes replace the previous attempt's.
  std::vector<std::vector<Block>> passes;
  nbp::PassHooks hooks;
  hooks.reset = reset;
  if (arm == Arm::A)
  {
    hooks.run = [&] {
      return nb_planar_process(shimModel_, audio_.samples.data(), audio_.samples.size(), nullptr, nullptr);
    };
  }
  else
  {
    hooks.run = [&] { return pass(nullptr); };
    hooks.onTimedPass = [&](size_t index) {
      if (index == 0)
        passes.clear();
      if (arm != Arm::A1)
      {
        NSDictionary* reply = call_unit(unit_, @{kNbAUCommand : kNbAUCmdStamps}, error);
        NSData* stamps = reply[@"stamps"];
        const size_t count = [reply[@"count"] unsignedLongValue];
        const NbAUStamp* s = static_cast<const NbAUStamp*>(stamps.bytes);
        for (size_t b = 0; b < blocks_.size(); b++)
          blocks_[b].unit = (b < count) ? s[b] : NbAUStamp{};
      }
      passes.push_back(blocks_);
    };
  }

  const std::string label = std::string(arm_name(arm)) + "/" +
                            (submodel == NbSubmodelNarrowest ? "nano" : "standard") + "/" +
                            std::to_string(blockSize);
  result.timing = nbp::measure(label, config, audio_.durationSeconds(), hooks, nullptr);

  // --- Pool the accepted passes' blocks -----------------------------------------
  if (arm != Arm::A && result.timing.succeeded)
  {
    std::vector<double> host, kernel, overhead, hopIn, wrapIn, wrapOut, hopOut;
    for (size_t index : result.timing.acceptedIndices)
    {
      if (index >= passes.size())
        continue;
      for (const Block& r : passes[index])
      {
        if (r.unit.entry == 0)
          continue;
        const double h = ticks_ns(r.hostEnd - r.hostStart);
        const double k = ticks_ns(r.unit.kernelEnd - r.unit.kernelStart);
        host.push_back(h);
        kernel.push_back(k);
        overhead.push_back(h - k);
        // Signed: across processes these are differences of one clock read on
        // two cores, and a reading that disagrees with causality is data about
        // the clock rather than something to hide.
        hopIn.push_back(static_cast<double>(static_cast<int64_t>(r.unit.entry - r.hostStart)) * g_ticksToNs);
        wrapIn.push_back(ticks_ns(r.unit.kernelStart - r.unit.entry));
        wrapOut.push_back(ticks_ns(r.unit.exit - r.unit.kernelEnd));
        hopOut.push_back(static_cast<double>(static_cast<int64_t>(r.hostEnd - r.unit.exit)) * g_ticksToNs);
      }
    }
    result.blockSamples = host.size();
    result.hostBlockNs = dist_of(host);
    result.kernelNs = dist_of(kernel);
    result.overheadNs = dist_of(overhead);
    result.hopInNs = dist_of(hopIn);
    result.wrapInNs = dist_of(wrapIn);
    result.wrapOutNs = dist_of(wrapOut);
    result.hopOutNs = dist_of(hopOut);
  }

  result.ok = result.timing.succeeded;
  if (!result.ok)
    result.failure = result.timing.failureReason;

  if (shimModel_)
  {
    nb_planar_destroy(shimModel_);
    shimModel_ = nullptr;
  }
  if (unit_)
  {
    [unit_ deallocateRenderResources];
    unit_ = nil;
    renderBlock_ = nil;
  }
  return result;
}

// --- Reporting -----------------------------------------------------------------------

std::string json_escape(const std::string& s)
{
  std::string o;
  for (char c : s)
  {
    if (c == '"' || c == '\\')
      o += '\\';
    if (static_cast<unsigned char>(c) < 0x20)
      continue;
    o += c;
  }
  return o;
}

void write_dist(std::string& j, const char* name, const Dist& d)
{
  char buf[256];
  std::snprintf(buf, sizeof(buf), "\"%s\":{\"median\":%.1f,\"p99\":%.1f,\"mean\":%.1f}", name, d.median, d.p99,
                d.mean);
  j += buf;
}

std::string to_json(const std::vector<SubjectResult>& results, const std::string& realtime,
                    const nbp::Environment& env, const HostOptions& options)
{
  std::string j = "{\n";
  j += "\"environment\":{\"deviceModel\":\"" + json_escape(env.deviceModel) + "\",\"cpu\":\"" + json_escape(env.cpu) +
       "\",\"os\":\"" + json_escape(env.osVersion) + "\",\"platform\":\"" + json_escape(env.platform) + "\"},\n";
  char buf[512];
  std::snprintf(buf, sizeof(buf), "\"warmupSeconds\":%.1f,\"windowSeconds\":%.1f,\n\"subjects\":[\n",
                options.warmupSeconds, options.windowSeconds);
  j += buf;
  for (size_t i = 0; i < results.size(); i++)
  {
    const SubjectResult& r = results[i];
    std::snprintf(buf, sizeof(buf),
                  "{\"arm\":\"%s\",\"submodel\":\"%s\",\"blockSize\":%d,\"ok\":%s,\"failure\":\"%s\","
                  "\"outOfProcess\":%s,\"engine\":%d,\"channels\":%d,\"sampleRate\":%.0f,",
                  arm_name(r.arm), r.submodel == NbSubmodelNarrowest ? "nano" : "standard", r.blockSize,
                  r.ok ? "true" : "false", json_escape(r.failure).c_str(), r.outOfProcess ? "true" : "false",
                  r.engine, r.channels, r.sampleRate);
    j += buf;
    std::snprintf(buf, sizeof(buf),
                  "\"corePercent\":%.4f,\"medianMs\":%.3f,\"spread\":%.4f,\"acceptedPasses\":%zu,"
                  "\"blockSamples\":%zu,\"parity\":%s,\"mismatchedSamples\":%zu,\"maxAbsDiff\":%.3g,",
                  r.timing.corePercent, r.timing.medianMs, r.timing.spread, r.timing.acceptedIndices.size(),
                  r.blockSamples, r.haveParity ? "true" : "false", r.mismatchedSamples, r.maxAbsDiff);
    j += buf;
    write_dist(j, "hostBlockNs", r.hostBlockNs);
    j += ",";
    write_dist(j, "kernelNs", r.kernelNs);
    j += ",";
    write_dist(j, "overheadNs", r.overheadNs);
    j += ",";
    write_dist(j, "hopInNs", r.hopInNs);
    j += ",";
    write_dist(j, "wrapInNs", r.wrapInNs);
    j += ",";
    write_dist(j, "wrapOutNs", r.wrapOutNs);
    j += ",";
    write_dist(j, "hopOutNs", r.hopOutNs);
    j += (i + 1 < results.size()) ? "},\n" : "}\n";
  }
  j += "]";
  if (!realtime.empty())
    j += ",\n\"realtime\":" + realtime;
  j += "}\n";
  return j;
}

void print_row(const SubjectResult& r)
{
  if (!r.ok)
  {
    logf("  %-3s %-8s %4d  FAILED: %s\n", arm_name(r.arm), r.submodel == NbSubmodelNarrowest ? "nano" : "standard",
         r.blockSize, r.failure.c_str());
    return;
  }
  const double periodNs = r.blockSize / 48000.0 * 1e9;
  logf("  %-3s %-8s %4d  core %7.3f%%  spread %5.2f%%  %s  kernel %8.0f  overhead med %7.0f p99 %7.0f ns "
       "(%.2f%% / %.2f%% of period)  hop in %6.0f out %6.0f  parity %s\n",
       arm_name(r.arm), r.submodel == NbSubmodelNarrowest ? "nano" : "standard", r.blockSize, r.timing.corePercent,
       r.timing.spread * 100.0, r.outOfProcess ? "OOP" : "in ", r.kernelNs.median, r.overheadNs.median,
       r.overheadNs.p99, r.overheadNs.median / periodNs * 100.0, r.overheadNs.p99 / periodNs * 100.0,
       r.hopInNs.median, r.hopOutNs.median,
       !r.haveParity ? "-" : (r.mismatchedSamples == 0 ? "bit-identical" : "DIFFERS"));
}

} // namespace

// ---------------------------------------------------------------------------------

int nb_au_host_run(const HostOptions& options)
{
  init_timebase();

  if (!options.logPath.empty())
  {
    if (FILE* f = std::fopen(options.logPath.c_str(), "w"))
      g_log = f;
  }

  AudioComponentDescription inproc{};
  inproc.componentType = kAudioUnitType_Effect;
  inproc.componentSubType = NB_AU_SUBTYPE_INPROC;
  inproc.componentManufacturer = NB_AU_MANUFACTURER;
  auto has_arm = [&](int a) { return std::find(options.arms.begin(), options.arms.end(), a) != options.arms.end(); };
  if (has_arm(2) || has_arm(6))
    [AUAudioUnit registerSubclass:NBAudioUnit.class
           asComponentDescription:inproc
                             name:@"Hmsl: NAMBench AU (in host)"
                          version:1];

  // Say what AudioToolbox can see before anything tries to instantiate it: a
  // missing extension otherwise surfaces only as an opaque -3000 per subject.
  // On iOS that is what a host without the inter-app-audio entitlement gets:
  // the system lists no extension AUs to it at all.
  {
    AudioComponentDescription any{};
    any.componentType = kAudioUnitType_Effect;
    any.componentManufacturer = NB_AU_MANUFACTURER;
    NSArray<AVAudioUnitComponent*>* found =
      [AVAudioUnitComponentManager.sharedAudioUnitComponentManager componentsMatchingDescription:any];
    logf("components from manufacturer Hmsl: %lu\n", static_cast<unsigned long>(found.count));
    for (AVAudioUnitComponent* c : found)
      logf("  %s (subtype %08x, sandboxSafe %d)\n", c.name.UTF8String, c.audioComponentDescription.componentSubType,
           c.sandboxSafe ? 1 : 0);
  }
  nbp::Audio audio;
  std::string error;
  if (!nbp::load_wav(options.inputPath, audio, error))
  {
    logf("input: %s\n", error.c_str());
    return 1;
  }
  NSData* model = [NSData dataWithContentsOfFile:@(options.modelPath.c_str())];
  if (model == nil)
  {
    logf("model: cannot read %s\n", options.modelPath.c_str());
    return 1;
  }

  const nbp::Environment env = nbp::capture_environment();
  logf("NAMBench AUv3 overhead — %s, %s, %s\n", env.deviceModel.c_str(), env.cpu.c_str(), env.osVersion.c_str());
  std::vector<int> offlineArms;
  for (int armIndex : options.arms)
    if (armIndex <= 4)
    {
      offlineArms.push_back(armIndex);
      logf("  arm %-2s %s\n", arm_name(static_cast<Arm>(armIndex)), arm_description(static_cast<Arm>(armIndex)));
    }
  logf("input %zu frames @ %.0f Hz (%.2f s); warm-up %.1f s, window %.1f s\n", audio.samples.size(),
       audio.sampleRate, audio.durationSeconds(), options.warmupSeconds, options.windowSeconds);

  Runner runner(options, audio, model);
  std::vector<SubjectResult> results;

  for (NbSubmodel submodel : offlineArms.empty() ? std::vector<NbSubmodel>{} : options.submodels)
  {
    for (int blockSize : options.blockSizes)
    {
      logf("%s, %d frames:\n", submodel == NbSubmodelNarrowest ? "nano" : "standard", blockSize);
      // A1 first: its output is the reference every AU arm must reproduce.
      std::vector<float> reference;
      for (int armIndex : offlineArms)
      {
        const Arm arm = static_cast<Arm>(armIndex);
        if (arm == Arm::A1)
        {
          results.push_back(runner.run(arm, submodel, blockSize, nullptr, &reference));
          print_row(results.back());
        }
      }
      for (int armIndex : offlineArms)
      {
        const Arm arm = static_cast<Arm>(armIndex);
        if (arm == Arm::A1)
          continue;
        results.push_back(runner.run(arm, submodel, blockSize, reference.empty() ? nullptr : &reference, nullptr));
        print_row(results.back());
      }
    }
  }

  // Arm F, after the offline arms so that nothing else is running.
  std::vector<RtResult> rtResults;
  if (has_arm(5) || has_arm(6))
  {
    RtOptions rt;
    rt.submodels = options.submodels;
    rt.blockSizes.clear();
    for (int b : options.blockSizes)
      if (b >= 24) // the out-of-process allocation floor
        rt.blockSizes.push_back(b);
    rt.modes.clear();
    if (has_arm(6))
      rt.modes.push_back(false);
    if (has_arm(5))
      rt.modes.push_back(true);
    rt.warmupSeconds = options.warmupSeconds;
    rt.seconds = options.rtSeconds;
    std::vector<float> inputF(audio.samples.begin(), audio.samples.end());
    rtResults = nb_au_realtime_run(rt, model, inputF, g_log);
  }

  const std::string json = to_json(results, rtResults.empty() ? std::string() : nb_au_realtime_json(rtResults), env,
                                   options);
  if (!options.jsonPath.empty())
  {
    if (FILE* f = std::fopen(options.jsonPath.c_str(), "w"))
    {
      std::fputs(json.c_str(), f);
      std::fclose(f);
      logf("wrote %s\n", options.jsonPath.c_str());
    }
  }

  int failures = 0;
  for (const SubjectResult& r : results)
    failures += r.ok ? 0 : 1;
  for (const RtResult& r : rtResults)
    failures += r.ok ? 0 : 1;
  logf("done: %zu subjects, %d failed\n", results.size() + rtResults.size(), failures);
  if (g_log != stdout)
    std::fclose(g_log);
  return failures == 0 ? 0 : 2;
}

bool nb_au_parse_arm(const std::string& s, int& out)
{
  static const char* names[] = {"A", "A1", "B", "C", "D", "F", "Fi"};
  for (int i = 0; i < 7; i++)
  {
    if (s == names[i])
    {
      out = i;
      return true;
    }
  }
  return false;
}
