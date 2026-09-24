// What the AUv3 overhead host and the Audio Unit agree on: component
// descriptions, message-channel keys, and the per-block timestamp record.
//
// Included by the Unit framework and by the host. Plain C so both sides see
// the same layout; the timestamp records cross the process boundary as bytes.

#ifndef NB_AU_SHARED_H
#define NB_AU_SHARED_H

#include <stdint.h>

#include <AudioToolbox/AudioToolbox.h>

// 'Hmsl' — shared by both components.
#define NB_AU_MANUFACTURER 0x486d736c
// 'nbIn' — the AUAudioUnit subclass registered inside the host (arm B).
#define NB_AU_SUBTYPE_INPROC 0x6e62496e
// 'nbEx' — the same class, shipped in the app extension (arms C, D, E).
#define NB_AU_SUBTYPE_EXTENSION 0x6e624578

/// One render call, as seen from inside the Audio Unit. All four are
/// mach_absolute_time() ticks. That clock is one system-wide timebase, so
/// these line up with the host's own stamps even when the unit runs in
/// another process.
typedef struct NbAUStamp
{
  uint64_t entry;       // first instruction of the render block
  uint64_t kernelStart; // input pulled, FTZ set, about to call the kernel
  uint64_t kernelEnd;   // kernel returned
  uint64_t exit;        // about to return to the caller
} NbAUStamp;

// Message-channel protocol. Requests carry kNbAUCommand; replies carry
// kNbAUStatus (0 = ok) and kNbAUError on failure.
#define kNbAUChannelName @"cc.hemsley.nambench.au"
#define kNbAUCommand @"cmd"
#define kNbAUStatus @"status"
#define kNbAUError @"error"

/// {model: NSData, submodel: NSNumber(NbSubmodel)} -> {}. Loads the model the
/// next allocateRenderResources will build. Must precede allocation.
#define kNbAUCmdConfigure @"configure"
/// {} -> {pid, engine, channels, sampleRate, blockSize}. After allocation.
#define kNbAUCmdInfo @"info"
/// {} -> {}. Clears model state and the stamp ring, the way the shim's _reset
/// does between passes. Never called while rendering.
#define kNbAUCmdReset @"reset"
/// {} -> {stamps: NSData of NbAUStamp[count], count}. Everything recorded
/// since the last reset.
#define kNbAUCmdStamps @"stamps"

#endif // NB_AU_SHARED_H
