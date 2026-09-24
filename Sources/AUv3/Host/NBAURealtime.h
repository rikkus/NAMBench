// Arm F: the unit under a real AVAudioEngine output. See NBAURealtime.mm.

#pragma once

#import <Foundation/Foundation.h>

#include <cstdio>
#include <string>
#include <vector>

#import <NAMEnginePlanar/NAMEnginePlanar.h>

struct RtOptions
{
  std::vector<NbSubmodel> submodels = {NbSubmodelNarrowest, NbSubmodelWidest};
  /// IO buffer sizes. 16 is absent: out of process the unit cannot allocate
  /// below 24 frames (AUV3-PATH.md, "Limits found").
  std::vector<int> blockSizes = {32, 64, 128, 256};
  /// false = Fi (in-process unit), true = F (extension, out of process).
  std::vector<bool> modes = {false, true};
  double warmupSeconds = 2.0;
  double seconds = 20.0;
  /// The paced bare-kernel reference, per subject.
  double pacedSeconds = 5.0;
  /// Idle time after the cluster probe, before the first subject.
  double settleSeconds = 15.0;
};

struct RtDist
{
  double median = 0, p99 = 0, max = 0, mean = 0;
};

struct RtThread
{
  bool captured = false;
  bool timeConstraint = false;
  double periodNs = 0, computationNs = 0, constraintNs = 0;
  int qos = 0;
  int priority = 0;
};

struct RtResult
{
  bool outOfProcess = false;
  NbSubmodel submodel = NbSubmodelWidest;
  int requestedFrames = 0;
  bool ok = false;
  std::string failure;

  int unitPid = 0;
  bool ranOutOfProcess = false;
  int engine = 0;
  int unitMaxFrames = 0;
  double unitSampleRate = 0;
  double deviceSampleRate = 0;
  int deviceBufferFrames = 0;    // macOS: the device's buffer after the request
  int sessionIOBufferFrames = 0; // iOS: what the audio session granted

  int ioFrames = 0; // frames per IO cycle, as observed
  double periodNs = 0;
  size_t cycles = 0;
  double expectedCycles = 0;
  size_t renderCalls = 0;
  size_t unmatchedCalls = 0;

  // Deadline misses, three ways.
  size_t discontinuities = 0; // sample-time gaps between consecutive cycles
  double framesSkipped = 0;
  uint32_t overloads = 0;       // macOS HAL processor-overload notifications
  size_t cyclesOverPeriod = 0;  // cycle took longer than its period
  size_t cyclesOverHalfPeriod = 0;
  size_t oddFrameCycles = 0;

  RtDist cycleNs, cycleStartJitterNs, sourceAfterStartNs;
  RtDist kernelNs, wrapperNs, entryAfterCycleStartNs, exitBeforeCycleEndNs;
  double bareKernelMedianNs = 0;

  // Which cluster the unit's render calls ran on, and the kernel on each.
  size_t callsOnE = 0, callsOnP = 0;
  double kernelMedianOnE = 0, kernelMedianOnP = 0;
  /// Render calls per CPU number, as a JSON object: the raw data behind E/P.
  std::string callsByCpu = "{}";

  // The paced reference: the bare kernel on a thread of our own with the IO
  // thread's time-constraint policy, woken once per period. No AU, no engine.
  RtDist pacedKernelNs;
  size_t pacedOnE = 0, pacedOnP = 0;
  double pacedMedianOnE = 0, pacedMedianOnP = 0;
  // The same, joined to an audio work interval (AudioWorkIntervalCreate) with
  // each block marked as an interval ending one period later: how a real audio
  // thread tells the scheduler its deadline.
  RtDist pacedWgKernelNs;
  size_t pacedWgOnE = 0, pacedWgOnP = 0;
  double pacedWgMedianOnE = 0, pacedWgMedianOnP = 0;

  RtThread unitThread, ioThread;
  uint32_t contextObserverCalls = 0;
  bool workgroupSeen = false;
};

std::vector<RtResult> nb_au_realtime_run(const RtOptions& options, NSData* model, const std::vector<float>& input,
                                         FILE* log);

/// A JSON array, one object per subject.
std::string nb_au_realtime_json(const std::vector<RtResult>& results);
