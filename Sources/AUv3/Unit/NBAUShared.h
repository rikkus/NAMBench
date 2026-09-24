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
  uint32_t frames;      // frameCount of this call
  uint32_t cpu;         // pthread_cpu_number_np at exit
} NbAUStamp;

/// How the thread that captured it is scheduled. Captured once, on the first
/// render call (arm F), so the render thread makes one syscall in its life.
typedef struct NbThreadInfo
{
  int32_t captured;
  int32_t timeConstraint; // 1 when the thread has THREAD_TIME_CONSTRAINT_POLICY
  uint32_t periodTicks;
  uint32_t computationTicks;
  uint32_t constraintTicks;
  int32_t preemptible;
  int32_t qos;      // qos_class_t
  int32_t priority; // sched_param priority
} NbThreadInfo;

#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <pthread/qos.h>

static inline void nb_capture_thread(NbThreadInfo* out)
{
  thread_time_constraint_policy_data_t policy = {0};
  mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
  boolean_t isDefault = 0;
  const kern_return_t kr = thread_policy_get(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                                             (thread_policy_t)&policy, &count, &isDefault);
  out->timeConstraint = (kr == KERN_SUCCESS && !isDefault) ? 1 : 0;
  out->periodTicks = policy.period;
  out->computationTicks = policy.computation;
  out->constraintTicks = policy.constraint;
  out->preemptible = policy.preemptible;
  out->qos = (int32_t)qos_class_self();
  int schedPolicy = 0;
  struct sched_param param = {0};
  pthread_getschedparam(pthread_self(), &schedPolicy, &param);
  out->priority = param.sched_priority;
  out->captured = 1;
}

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
/// {} -> {captured, timeConstraint, periodTicks, computationTicks,
/// constraintTicks, qos, priority, contextObserverCalls, workgroup}. How the
/// render thread is scheduled, captured on its first render call.
#define kNbAUCmdThread @"thread"

#endif // NB_AU_SHARED_H
