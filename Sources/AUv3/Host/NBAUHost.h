// Entry point shared by the macOS and iOS front ends of the AUv3 overhead host.

#pragma once

#include <string>
#include <vector>

#import <NAMEnginePlanar/NAMEnginePlanar.h>

struct HostOptions
{
  /// Arm indices: 0 = A, 1 = A1, 2 = B, 3 = C, 4 = D; 5 = F and 6 = Fi run
  /// under a real AVAudioEngine output instead (NBAURealtime.h).
  std::vector<int> arms = {0, 1, 2, 3, 4};
  std::vector<NbSubmodel> submodels = {NbSubmodelNarrowest, NbSubmodelWidest};
  std::vector<int> blockSizes = {16, 32, 64, 128, 256};
  // Shorter than nambench's 5 s / 30 s: this host runs up to fifty subjects,
  // and minSamples still guarantees at least fifteen passes in each window.
  double warmupSeconds = 2.0;
  double windowSeconds = 10.0;
  /// Arm F: seconds measured per subject, after warmupSeconds.
  double rtSeconds = 20.0;
  std::string modelPath;
  std::string inputPath;
  std::string jsonPath;
  /// Empty for stdout.
  std::string logPath;
};

/// 0 when every subject succeeded, 2 when any failed, 1 on setup failure.
int nb_au_host_run(const HostOptions& options);

/// "A", "A1", "B", "C", "D", "F" or "Fi" to an arm index.
bool nb_au_parse_arm(const std::string& s, int& out);
