// The Audio Unit under test: the planar NAM engine behind a plain AUv3 effect.
//
// One class serves every AU arm. Arm B registers it inside the host with
// +[AUAudioUnit registerSubclass:...]; arms C, D and E reach it through the app
// extension, whose factory lives here too so that in-process loading on macOS
// (AudioComponentBundle) finds it in this framework.

#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudioKit/CoreAudioKit.h>
#import <Foundation/Foundation.h>

#import <NAMBenchAU/NBAUShared.h>

NS_ASSUME_NONNULL_BEGIN

@interface NBAudioUnit : AUAudioUnit
@end

/// The extension's principal class and the AudioComponents factoryFunction.
@interface NBAudioUnitFactory : NSObject <AUAudioUnitFactory>
@end

NS_ASSUME_NONNULL_END
