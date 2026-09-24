#import "NBAudioUnit.h"

#include <mach/mach_time.h>
#include <unistd.h>

#include <atomic>
#include <vector>

#import <NAMEnginePlanar/NAMEnginePlanar.h>

#include "nb_shim_timing.h"

// ---------------------------------------------------------------------------
// Render-side state.
//
// Everything the render block touches lives in this struct, and the block
// captures a raw pointer to it rather than `self`: no Objective-C message send,
// retain or release happens on the render thread. That is how a well-behaved
// AUv3 is written, and a badly behaved one would be measuring itself rather
// than the wrapper.
// ---------------------------------------------------------------------------
namespace
{

struct Kernel
{
  NbModel* model = nullptr;
  int32_t maxFrames = 0;

  // Where pulled input lands. Sized in allocateRenderResources.
  std::vector<float> input;
  AudioBufferList inputList{};

  // Output used when the caller hands in null mData (in-place is allowed by
  // the AU contract; this unit does not rely on the caller choosing either).
  std::vector<float> output;

  // One record per render call since the last reset. Preallocated to hold a
  // generous number of calls, so the render thread never allocates.
  std::vector<NbAUStamp> stamps;
  std::atomic<size_t> stampCount{0};

  // Arm F: how the render thread is scheduled, and whether the host has told
  // the unit about its workgroup.
  NbThreadInfo renderThread{};
  std::atomic<bool> renderThreadCaptured{false};
  std::atomic<uint32_t> contextObserverCalls{0};
  std::atomic<bool> workgroupSeen{false};
};

constexpr size_t kStampCapacity = 1u << 20; // ~33 passes of 16-frame blocks over the input

} // namespace

// The message channel handed to the host. Its own object rather than the unit
// itself: out of process, AudioToolbox sets callHostBlock on whatever
// messageChannelFor: returns, and it must be a real property.
@interface NBAUChannel : NSObject <AUMessageChannel>
@property(NS_NONATOMIC_IOSONLY, copy, nullable) CallHostBlock callHostBlock;
@property(nonatomic, weak) NBAudioUnit* unit;
@end

@interface NBAudioUnit ()
- (NSDictionary*)handleMessage:(NSDictionary*)message;
@end

@implementation NBAUChannel
- (NSDictionary*)callAudioUnit:(NSDictionary*)message
{
  NBAudioUnit* unit = self.unit;
  return unit ? [unit handleMessage:message] : @{kNbAUStatus : @1, kNbAUError : @"unit gone"};
}
@end

@implementation NBAudioUnit
{
  NBAUChannel* _channel;
  AUAudioUnitBusArray* _inputBusArray;
  AUAudioUnitBusArray* _outputBusArray;
  AUAudioUnitBus* _inputBus;
  AUAudioUnitBus* _outputBus;
  Kernel _kernel;

  // Set by the configure message, consumed by allocateRenderResources.
  NSData* _modelBytes;
  NbSubmodel _submodel;
  double _modelSampleRate;
}

- (instancetype)initWithComponentDescription:(AudioComponentDescription)componentDescription
                                     options:(AudioComponentInstantiationOptions)options
                                       error:(NSError**)outError
{
  self = [super initWithComponentDescription:componentDescription options:options error:outError];
  if (self == nil)
    return nil;

  // Mono float32, deinterleaved, at the model's rate. The host sets 48 kHz and
  // allocateRenderResources refuses anything else, so no arm can resample.
  AVAudioFormat* format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:48000.0 channels:1];
  _inputBus = [[AUAudioUnitBus alloc] initWithFormat:format error:outError];
  _outputBus = [[AUAudioUnitBus alloc] initWithFormat:format error:outError];
  if (_inputBus == nil || _outputBus == nil)
    return nil;
  _inputBus.maximumChannelCount = 1;
  _outputBus.maximumChannelCount = 1;
  _inputBusArray = [[AUAudioUnitBusArray alloc] initWithAudioUnit:self
                                                          busType:AUAudioUnitBusTypeInput
                                                           busses:@[ _inputBus ]];
  _outputBusArray = [[AUAudioUnitBusArray alloc] initWithAudioUnit:self
                                                           busType:AUAudioUnitBusTypeOutput
                                                            busses:@[ _outputBus ]];

  // One parameter, so the parameter-tree machinery a real plugin has is present
  // and the render block walks a real event list. It does no DSP: smoothing
  // work would be plugin code, not wrapper.
  AUParameter* level = [AUParameterTree createParameterWithIdentifier:@"level"
                                                                 name:@"Level"
                                                              address:0
                                                                  min:0.0
                                                                  max:1.0
                                                                 unit:kAudioUnitParameterUnit_LinearGain
                                                             unitName:nil
                                                                flags:kAudioUnitParameterFlag_IsReadable |
                                                                      kAudioUnitParameterFlag_IsWritable
                                                         valueStrings:nil
                                                  dependentParameters:nil];
  level.value = 1.0;
  self.parameterTree = [AUParameterTree createTreeWithChildren:@[ level ]];

  _kernel.stamps.resize(kStampCapacity);
  _submodel = NbSubmodelWidest;
  self.maximumFramesToRender = 64;
  return self;
}

- (void)dealloc
{
  if (_kernel.model != nullptr)
    nb_planar_destroy(_kernel.model);
}

- (AUAudioUnitBusArray*)inputBusses
{
  return _inputBusArray;
}

- (AUAudioUnitBusArray*)outputBusses
{
  return _outputBusArray;
}

- (BOOL)canProcessInPlace
{
  return YES;
}

- (BOOL)allocateRenderResourcesAndReturnError:(NSError**)outError
{
  if (![super allocateRenderResourcesAndReturnError:outError])
    return NO;

  const double rate = _outputBus.format.sampleRate;
  if (_modelBytes == nil || rate != 48000.0 || _inputBus.format.sampleRate != rate ||
      _outputBus.format.channelCount != 1 || _inputBus.format.channelCount != 1)
  {
    if (outError)
      *outError = [NSError errorWithDomain:NSOSStatusErrorDomain code:kAudioUnitErr_FormatNotSupported userInfo:nil];
    return NO;
  }

  if (_kernel.model != nullptr)
  {
    nb_planar_destroy(_kernel.model);
    _kernel.model = nullptr;
  }

  const int32_t frames = static_cast<int32_t>(self.maximumFramesToRender);
  char err[512] = {0};
  _kernel.model = nb_planar_create(static_cast<const uint8_t*>(_modelBytes.bytes), _modelBytes.length, _submodel, 0,
                                   frames, err, sizeof(err));
  if (_kernel.model == nullptr || nb_planar_sample_rate(_kernel.model) != rate)
  {
    if (outError)
      *outError = [NSError errorWithDomain:@"NBAudioUnit"
                                      code:1
                                  userInfo:@{NSLocalizedDescriptionKey : @(err[0] ? err : "model rate mismatch")}];
    return NO;
  }
  _modelSampleRate = nb_planar_sample_rate(_kernel.model);

  _kernel.maxFrames = frames;
  _kernel.input.assign(static_cast<size_t>(frames), 0.0f);
  _kernel.output.assign(static_cast<size_t>(frames), 0.0f);
  _kernel.stampCount.store(0);
  _kernel.renderThreadCaptured.store(false);
  return YES;
}

- (void)deallocateRenderResources
{
  [super deallocateRenderResources];
}

- (AUInternalRenderBlock)internalRenderBlock
{
  Kernel* kernel = &_kernel;

  return ^AUAudioUnitStatus(AudioUnitRenderActionFlags* actionFlags, const AudioTimeStamp* timestamp,
                            AUAudioFrameCount frameCount, NSInteger outputBusNumber, AudioBufferList* outputData,
                            const AURenderEvent* realtimeEventListHead, AURenderPullInputBlock pullInputBlock) {
    const uint64_t entry = mach_absolute_time();

    if (!kernel->renderThreadCaptured.load(std::memory_order_relaxed))
    {
      nb_capture_thread(&kernel->renderThread);
      kernel->renderThreadCaptured.store(true, std::memory_order_release);
    }

    if (kernel->model == nullptr || frameCount > static_cast<AUAudioFrameCount>(kernel->maxFrames) ||
        pullInputBlock == nullptr)
      return kAudioUnitErr_Uninitialized;

    // Walk the event list, as any unit with parameters must. Nothing is
    // scheduled in these runs, so this is the cost of looking.
    for (const AURenderEvent* event = realtimeEventListHead; event != nullptr; event = event->head.next)
    {
    }

    AudioBufferList& in = kernel->inputList;
    in.mNumberBuffers = 1;
    in.mBuffers[0].mNumberChannels = 1;
    in.mBuffers[0].mDataByteSize = frameCount * sizeof(float);
    in.mBuffers[0].mData = kernel->input.data();
    AudioUnitRenderActionFlags pullFlags = 0;
    const AUAudioUnitStatus pulled = pullInputBlock(&pullFlags, timestamp, frameCount, 0, &in);
    if (pulled != noErr)
      return pulled;

    AudioBuffer& outBuffer = outputData->mBuffers[0];
    if (outBuffer.mData == nullptr)
      outBuffer.mData = kernel->output.data();
    outBuffer.mDataByteSize = frameCount * sizeof(float);

    const uint64_t previousFpcr = denormals_disable();
    const uint64_t kernelStart = mach_absolute_time();
    nb_planar_process_block(kernel->model, static_cast<const float*>(in.mBuffers[0].mData),
                            static_cast<float*>(outBuffer.mData), static_cast<int32_t>(frameCount));
    const uint64_t kernelEnd = mach_absolute_time();
    denormals_restore(previousFpcr);

    const size_t slot = kernel->stampCount.load(std::memory_order_relaxed);
    if (slot < kernel->stamps.size())
    {
      NbAUStamp& s = kernel->stamps[slot];
      s.entry = entry;
      s.kernelStart = kernelStart;
      s.kernelEnd = kernelEnd;
      s.exit = mach_absolute_time();
      s.frames = frameCount;
      size_t cpu = 0;
      pthread_cpu_number_np(&cpu);
      s.cpu = static_cast<uint32_t>(cpu);
      kernel->stampCount.store(slot + 1, std::memory_order_release);
    }
    return noErr;
  };
}

// Called by the host (on the render thread) with its render context. The
// workgroup in it is what a unit joins its own helper threads to; whether one
// arrives at all, in and out of process, is part of what arm F records.
- (AURenderContextObserver)renderContextObserver
{
  Kernel* kernel = &_kernel;
  return ^(const AudioUnitRenderContext* context) {
    kernel->contextObserverCalls.fetch_add(1, std::memory_order_relaxed);
    if (context != nullptr && context->workgroup != nullptr)
      kernel->workgroupSeen.store(true, std::memory_order_relaxed);
  };
}

// ---------------------------------------------------------------------------
// Message channel: how the host configures the unit and collects its stamps,
// in or out of process, through the same path. None of it runs while timing.
// ---------------------------------------------------------------------------

- (id<AUMessageChannel>)messageChannelFor:(NSString*)channelName
{
  if (![channelName isEqualToString:kNbAUChannelName])
    return nil;
  if (_channel == nil)
  {
    _channel = [NBAUChannel new];
    _channel.unit = self;
  }
  return _channel;
}

static NSDictionary* nb_fail(NSString* why)
{
  return @{kNbAUStatus : @1, kNbAUError : why};
}

- (NSDictionary*)handleMessage:(NSDictionary*)message
{
  NSString* command = message[kNbAUCommand];

  if ([command isEqualToString:kNbAUCmdConfigure])
  {
    NSData* model = message[@"model"];
    NSNumber* submodel = message[@"submodel"];
    if (![model isKindOfClass:NSData.class] || submodel == nil)
      return nb_fail(@"configure needs model and submodel");
    if (self.renderResourcesAllocated)
      return nb_fail(@"configure must precede allocateRenderResources");
    _modelBytes = [model copy];
    _submodel = static_cast<NbSubmodel>(submodel.intValue);
    return @{kNbAUStatus : @0};
  }

  if ([command isEqualToString:kNbAUCmdInfo])
  {
    if (_kernel.model == nullptr)
      return nb_fail(@"not allocated");
    return @{
      kNbAUStatus : @0,
      @"pid" : @(getpid()),
      @"engine" : @(nb_planar_engine(_kernel.model)),
      @"channels" : @(nb_planar_channels(_kernel.model)),
      @"sampleRate" : @(_modelSampleRate),
      @"busSampleRate" : @(_outputBus.format.sampleRate),
      @"blockSize" : @(_kernel.maxFrames),
    };
  }

  if ([command isEqualToString:kNbAUCmdThread])
  {
    const bool captured = _kernel.renderThreadCaptured.load(std::memory_order_acquire);
    const NbThreadInfo& t = _kernel.renderThread;
    return @{
      kNbAUStatus : @0,
      @"captured" : @(captured),
      @"timeConstraint" : @(captured ? t.timeConstraint : 0),
      @"periodTicks" : @(t.periodTicks),
      @"computationTicks" : @(t.computationTicks),
      @"constraintTicks" : @(t.constraintTicks),
      @"qos" : @(t.qos),
      @"priority" : @(t.priority),
      @"contextObserverCalls" : @(_kernel.contextObserverCalls.load()),
      @"workgroup" : @(_kernel.workgroupSeen.load()),
    };
  }

  if ([command isEqualToString:kNbAUCmdReset])
  {
    if (_kernel.model == nullptr)
      return nb_fail(@"not allocated");
    nb_planar_reset(_kernel.model, _modelSampleRate, _kernel.maxFrames);
    _kernel.stampCount.store(0, std::memory_order_release);
    return @{kNbAUStatus : @0};
  }

  if ([command isEqualToString:kNbAUCmdStamps])
  {
    const size_t count = _kernel.stampCount.load(std::memory_order_acquire);
    NSData* bytes = [NSData dataWithBytes:_kernel.stamps.data() length:count * sizeof(NbAUStamp)];
    return @{kNbAUStatus : @0, @"stamps" : bytes, @"count" : @(count)};
  }

  return nb_fail([NSString stringWithFormat:@"unknown command %@", command]);
}

@end

// ---------------------------------------------------------------------------

@implementation NBAudioUnitFactory

- (void)beginRequestWithExtensionContext:(NSExtensionContext*)context
{
}

- (nullable AUAudioUnit*)createAudioUnitWithComponentDescription:(AudioComponentDescription)desc
                                                           error:(NSError**)error
{
  return [[NBAudioUnit alloc] initWithComponentDescription:desc error:error];
}

@end
