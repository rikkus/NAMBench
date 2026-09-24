// The extension carries no code of its own. Its principal class and the
// AudioComponents factory are NBAudioUnitFactory in the NAMBenchAU framework,
// which is also what lets macOS load it in-process (AudioComponentBundle).
// This file exists because a target needs at least one source.

#import <NAMBenchAU/NBAudioUnit.h>

__attribute__((used)) static Class nb_force_link(void)
{
  return NBAudioUnitFactory.class;
}
