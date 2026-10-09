// Cache contention: the bare kernel with neighbours. See NBContention.mm.

#pragma once

#import <Foundation/Foundation.h>

#include <cstdio>
#include <string>
#include <vector>

#import <NAMEnginePlanar/NAMEnginePlanar.h>

struct ContentionOptions
{
  std::vector<NbSubmodel> submodels = {NbSubmodelNarrowest, NbSubmodelWidest};
  std::vector<int> blockSizes = {64, 256};
  /// Which scenarios: "serial", "parallel", "thrash".
  std::vector<std::string> scenarios = {"serial", "parallel", "thrash"};
  /// Instance counts for serial and parallel. 1 is the solo baseline.
  std::vector<int> instances = {1, 2, 4, 8, 16};
  /// Buffer sizes the thrashing neighbour streams over, in KiB.
  std::vector<int> thrashKiB = {256, 4096, 16384, 65536};
  double warmupSeconds = 2.0;
  double windowSeconds = 10.0;
  /// The protocol's agreement tolerance. 3% elsewhere; contention makes pass
  /// to pass variation part of what is measured, so these runs accept up to
  /// 10% and report each subject's spread.
  double acceptTolerance = 0.10;
};

struct ContentionResult
{
  std::string scenario; // "solo", "serial", "parallel", "thrash"
  NbSubmodel submodel = NbSubmodelWidest;
  int blockSize = 64;
  int instances = 1;     // serial / parallel: instances running, the timed one included
  int thrashKiB = 0;     // thrash: the neighbour's buffer
  bool ok = false;
  std::string failure;

  /// One instance's cost, as nambench reports it: per instance, per core.
  double corePercent = 0;
  double spread = 0;
  size_t acceptedPasses = 0;
  /// The timed instance's per-block kernel time, pooled over accepted passes.
  double blockMedianNs = 0, blockP99Ns = 0;
  /// Against solo at the same submodel and block size.
  double slowdownMedian = 0;

  /// Memory one instance holds (malloc bytes in use after create, minus before).
  size_t instanceBytes = 0;
  /// The timed instance's output against the solo run's: bit-identical or not.
  bool parity = false;
  size_t mismatchedSamples = 0;
  /// Parallel: how many neighbour blocks ran while timing, and on which CPUs.
  std::string neighbourCpus = "{}";
};

std::vector<ContentionResult> nb_contention_run(const ContentionOptions& options, NSData* model,
                                                const std::vector<float>& input, double sampleRate, FILE* log);

/// A JSON array, one object per subject.
std::string nb_contention_json(const std::vector<ContentionResult>& results);
