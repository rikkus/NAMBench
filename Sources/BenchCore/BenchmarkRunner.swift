import Foundation

/// Progress emitted as a run proceeds.
public enum BenchmarkEvent {
  case status(String)
  case probed(variant: String, probe: VariantProbe)
  case parityMeasured(ParityResult)
  case warmupStarted(variant: String, attempt: Int)
  case warmupPass(variant: String, pass: Int, milliseconds: Double, secondsRemaining: Double)
  case warmupFinished(variant: String, passes: Int)
  case timingStarted(variant: String, attempt: Int)
  case timingPass(variant: String, sample: Int, milliseconds: Double, secondsRemaining: Double)
  case attemptRejected(variant: String, attempt: Int, reason: String)
  case variantFinished(VariantResult)
  case finished(BenchmarkReport)
}

public enum BenchmarkError: LocalizedError {
  case cancelled
  case sampleRateMismatch(model: Double, audio: Double)
  case noVariants
  case channelMismatch(variant: String, expected: Int, actual: Int)

  public var errorDescription: String? {
    switch self {
    case .cancelled:
      return "Benchmark cancelled"
    case let .sampleRateMismatch(model, audio):
      return "Model expects \(Int(model)) Hz but the audio is \(Int(audio)) Hz"
    case .noVariants:
      return "Nothing to measure: the variant list is empty"
    case let .channelMismatch(variant, expected, actual):
      return "\(variant): selected submodel has \(actual) channels, but the first variant reported \(expected)"
    }
  }
}

public final class BenchmarkRunner {
  private let config: BenchmarkConfig
  private let variants: [Variant]
  private let modelURL: URL
  private let audioURL: URL
  private let pinsURL: URL?
  private let isCancelled: () -> Bool

  /// `variants[0]` is the baseline: parity is measured against it, and every
  /// other variant's speedup is relative to it.
  public init(
    config: BenchmarkConfig = BenchmarkConfig(),
    variants: [Variant] = Variant.all,
    modelURL: URL,
    audioURL: URL,
    pinsURL: URL? = nil,
    isCancelled: @escaping () -> Bool = { false }
  ) {
    self.config = config
    self.variants = variants
    self.modelURL = modelURL
    self.audioURL = audioURL
    self.pinsURL = pinsURL
    self.isCancelled = isCancelled
  }

  // MARK: - Run

  public func run(onEvent: (BenchmarkEvent) -> Void) throws -> BenchmarkReport {
    let startedAt = Date()

    // --- Load everything into RAM up front -----------------------------------

    onEvent(.status("Loading \(modelURL.lastPathComponent)"))
    let namBytes = try Data(contentsOf: modelURL)

    onEvent(.status("Loading \(audioURL.lastPathComponent)"))
    let audio = try AudioLoader.loadWAV(at: audioURL)
    onEvent(.status(String(
      format: "%d frames, %.0f Hz, %.2f s",
      audio.frameCount, audio.sampleRate, audio.durationSeconds
    )))

    // --- Probe and build, asserting the engine before anything is measured ---

    guard !variants.isEmpty else { throw BenchmarkError.noVariants }

    var probes: [String: VariantProbe] = [:]
    var models: [String: VariantModel] = [:]
    // Whatever the first variant found in the selected submodel is what every
    // other variant has to agree on. Nothing here hard-codes 8 any more: the
    // channel count is a property of the submodel that was asked for.
    var expectedChannels = 0

    for variant in variants {
      try checkCancelled()
      let probe = try variant.probe(namBytes: namBytes, submodel: config.submodel)
      probes[variant.name] = probe
      onEvent(.probed(variant: variant.name, probe: probe))

      if expectedChannels == 0 {
        expectedChannels = probe.channels
      } else if probe.channels != expectedChannels {
        throw BenchmarkError.channelMismatch(
          variant: variant.name, expected: expectedChannels, actual: probe.channels
        )
      }
      if probe.sampleRate > 0, audio.sampleRate > 0, probe.sampleRate != audio.sampleRate {
        throw BenchmarkError.sampleRateMismatch(model: probe.sampleRate, audio: audio.sampleRate)
      }

      // Throws if the model did not route to the engine this variant exists to
      // measure. Deliberately fatal: the fork silently falls back to the
      // generic engine on a shape mismatch, and reporting that as "fused" would
      // look like a catastrophic regression rather than a harness bug.
      models[variant.name] = try variant.makeModel(
        namBytes: namBytes, config: config, expectedChannels: expectedChannels
      )
      onEvent(.status("\(variant.name): \(probe.engine.displayName), \(probe.channels) channels"))
    }

    // --- Record the conditions, but do not wait on them ----------------------
    //
    // There is deliberately no idle gate. On macOS the system is never idle —
    // Spotlight and other background work cannot be paused, only disabled at
    // the cost of a full re-index — so waiting for quiet just blocks forever.
    // Interference is handled where it can actually be handled: in the
    // tightest-90% analysis, which discards the passes it spoiled.

    let systemAtStart = SystemState.capture(sampleCPU: config.recordSystemCPU)
    onEvent(.status("System at start: \(systemAtStart.summary)"))

    // --- Parity: are we even comparing the same audio? -----------------------

    // Every variant is compared against variants[0] over the whole file. With a
    // dozen candidates in one run this is the only thing standing between "this
    // kernel is 30% faster" and "this kernel reassociated something and is 30%
    // faster at computing different audio".

    var parities: [ParityResult] = []
    var checksums: [String: Double] = [:]

    if config.checkParity, variants.count >= 2 {
      onEvent(.status("Checking output parity"))
      var reference: [Double]?
      for variant in variants {
        try checkCancelled()
        let model = models[variant.name]!
        model.reset()
        let pass = model.process(input: audio.samples, captureOutput: true, wantChecksum: true)
        checksums[variant.name] = pass.checksum ?? 0

        guard let output = pass.output else { continue }
        if reference == nil {
          reference = output
          continue
        }
        let result = Self.compareOutputs(
          reference: reference!,
          comparison: output,
          referenceName: variants[0].name,
          comparisonName: variant.name,
          tolerance: config.parityWarnRelativeRMS
        )
        parities.append(result)
        onEvent(.parityMeasured(result))
      }
    }

    // --- Measure each variant ------------------------------------------------

    var results: [VariantResult] = []
    for variant in variants {
      try checkCancelled()
      let result = try measure(
        variant: variant,
        model: models[variant.name]!,
        probe: probes[variant.name]!,
        audio: audio,
        checksum: checksums[variant.name] ?? 0,
        onEvent: onEvent
      )
      results.append(result)
      onEvent(.variantFinished(result))
    }

    // --- Assemble the report -------------------------------------------------

    // One speedup per non-baseline variant, against variants[0]. Kept as a
    // single scalar too when there are exactly two, so existing two-way reports
    // read as they always have.
    var speedups: [VariantSpeedup] = []
    if let baseline = results.first, baseline.succeeded, baseline.meanMs > 0 {
      for result in results.dropFirst() where result.succeeded && result.meanMs > 0 {
        speedups.append(VariantSpeedup(
          variant: result.variant,
          baseline: baseline.variant,
          factor: baseline.meanMs / result.meanMs
        ))
      }
    }
    let speedup = (results.count == 2) ? speedups.first?.factor : nil

    let firstProbe = probes[variants[0].name]!
    let report = BenchmarkReport(
      schemaVersion: BenchmarkReport.schemaVersion,
      startedAt: startedAt,
      finishedAt: Date(),
      environment: .capture(),
      pins: pinsURL.flatMap(VendorPins.load(from:)),
      config: config,
      model: ModelInfo(
        fileName: modelURL.lastPathComponent,
        byteCount: namBytes.count,
        submodelIndex: firstProbe.submodelIndex,
        submodelCount: firstProbe.submodelCount,
        submodelMaxValue: firstProbe.submodelMaxValue,
        channels: firstProbe.channels,
        sampleRate: firstProbe.sampleRate
      ),
      audio: AudioInfo(
        fileName: audioURL.lastPathComponent,
        frameCount: audio.frameCount,
        sampleRate: audio.sampleRate,
        durationSeconds: audio.durationSeconds
      ),
      systemAtStart: systemAtStart,
      systemAtEnd: .capture(sampleCPU: config.recordSystemCPU),
      results: results,
      parity: parities.first,
      parities: parities,
      speedup: speedup,
      speedups: speedups
    )

    onEvent(.finished(report))
    return report
  }

  // MARK: - Per-variant measurement

  private func measure(
    variant: Variant,
    model: VariantModel,
    probe: VariantProbe,
    audio: AudioBuffer,
    checksum: Double,
    onEvent: (BenchmarkEvent) -> Void
  ) throws -> VariantResult {
    var attempts: [AttemptRecord] = []

    for attempt in 1...config.maxAttempts {
      try checkCancelled()

      // 1. Warm up for a fixed duration, discarding the results.
      onEvent(.warmupStarted(variant: variant.name, attempt: attempt))
      let warmupPasses = try warmUp(variant: variant, model: model, audio: audio, onEvent: onEvent)
      onEvent(.warmupFinished(variant: variant.name, passes: warmupPasses))

      // 2. Time for the configured window.
      onEvent(.timingStarted(variant: variant.name, attempt: attempt))
      let samples = try timeWindow(variant: variant, model: model, audio: audio, onEvent: onEvent)

      // 3. Does 90% of it agree?
      guard let selection = Statistics.tightestWindow(samples, fraction: config.acceptFraction) else {
        let reason = "could not select a \(Int(config.acceptFraction * 100))% window from \(samples.count) samples"
        attempts.append(AttemptRecord(
          attempt: attempt, warmupPasses: warmupPasses, samplesMs: samples,
          accepted: false, spread: nil, rejectionReason: reason
        ))
        onEvent(.attemptRejected(variant: variant.name, attempt: attempt, reason: reason))
        continue
      }

      let accepted = selection.spread <= config.acceptTolerance
      let reason = accepted ? nil : String(
        format: "tightest %.0f%% of %d samples spread %.2f%%, want within %.2f%%",
        config.acceptFraction * 100, samples.count, selection.spread * 100,
        config.acceptTolerance * 100
      )

      attempts.append(AttemptRecord(
        attempt: attempt, warmupPasses: warmupPasses, samplesMs: samples,
        accepted: accepted, spread: selection.spread, rejectionReason: reason
      ))

      if accepted {
        let meanMs = selection.mean
        return VariantResult(
          variant: variant.name,
          repository: variant.repository,
          codePath: variant.codePath,
          engine: model.engine,
          channels: model.channels,
          probe: probe,
          attempts: attempts,
          succeeded: true,
          failureReason: nil,
          acceptedMs: selection.accepted,
          discardedMs: selection.discarded,
          meanMs: meanMs,
          medianMs: selection.median,
          minMs: selection.min,
          maxMs: selection.max,
          spread: selection.spread,
          standardDeviationMs: Statistics.standardDeviation(selection.accepted),
          realTimeFactor: meanMs > 0 ? audio.durationSeconds / (meanMs / 1000) : 0,
          checksum: checksum
        )
      }

      onEvent(.attemptRejected(variant: variant.name, attempt: attempt, reason: reason ?? "rejected"))
    }

    // Every attempt was too noisy. Report the samples rather than a number
    // nobody should trust.
    let lastSamples = attempts.last?.samplesMs ?? []
    return VariantResult(
      variant: variant.name,
      repository: variant.repository,
      codePath: variant.codePath,
      engine: model.engine,
      channels: model.channels,
      probe: probe,
      attempts: attempts,
      succeeded: false,
      failureReason: "no attempt met the \(String(format: "%.1f", config.acceptTolerance * 100))% agreement threshold in \(config.maxAttempts) attempts",
      acceptedMs: [],
      discardedMs: lastSamples,
      meanMs: 0, medianMs: 0, minMs: 0, maxMs: 0, spread: 0, standardDeviationMs: 0,
      realTimeFactor: 0,
      checksum: checksum
    )
  }

  /// Run full-file passes for a fixed duration, discarding the results.
  ///
  /// Always completes at least one pass, and always lets the in-flight pass
  /// finish, so the model is never left mid-buffer going into timing.
  private func warmUp(
    variant: Variant,
    model: VariantModel,
    audio: AudioBuffer,
    onEvent: (BenchmarkEvent) -> Void
  ) throws -> Int {
    var passes = 0
    let start = Date()

    repeat {
      try checkCancelled()

      if config.resetBetweenPasses { model.reset() }
      let elapsed = model.process(input: audio.samples).nanoseconds
      passes += 1

      let spent = Date().timeIntervalSince(start)
      onEvent(.warmupPass(
        variant: variant.name,
        pass: passes,
        milliseconds: Double(elapsed) / 1_000_000,
        secondsRemaining: Swift.max(0, config.warmupSeconds - spent)
      ))
    } while Date().timeIntervalSince(start) < config.warmupSeconds

    return passes
  }

  /// Keep timing passes for the configured window.
  private func timeWindow(
    variant: Variant,
    model: VariantModel,
    audio: AudioBuffer,
    onEvent: (BenchmarkEvent) -> Void
  ) throws -> [Double] {
    var samples: [Double] = []
    let start = Date()
    // Extending for minSamples must not be able to run away on a very slow
    // machine, so cap the extension.
    let hardDeadline = start.addingTimeInterval(config.timingWindowSeconds * 3)

    while true {
      try checkCancelled()

      if config.resetBetweenPasses { model.reset() }
      let elapsed = model.process(input: audio.samples).nanoseconds
      samples.append(Double(elapsed) / 1_000_000)

      let spent = Date().timeIntervalSince(start)
      onEvent(.timingPass(
        variant: variant.name,
        sample: samples.count,
        milliseconds: samples[samples.count - 1],
        secondsRemaining: Swift.max(0, config.timingWindowSeconds - spent)
      ))

      let windowDone = spent >= config.timingWindowSeconds
      // Below minSamples the "90% agree" test says nothing, so extend rather
      // than accept a result built from a handful of numbers.
      let enoughSamples = samples.count >= config.minSamples
      if (windowDone && enoughSamples) || Date() >= hardDeadline { return samples }
    }
  }

  // MARK: - Helpers

  static func compareOutputs(
    reference: [Double],
    comparison: [Double],
    referenceName: String,
    comparisonName: String,
    tolerance: Double
  ) -> ParityResult {
    let count = Swift.min(reference.count, comparison.count)
    var maxAbsolute = 0.0
    var sumSquaredDifference = 0.0
    var sumSquaredReference = 0.0

    for index in 0..<count {
      let difference = reference[index] - comparison[index]
      maxAbsolute = Swift.max(maxAbsolute, abs(difference))
      sumSquaredDifference += difference * difference
      sumSquaredReference += reference[index] * reference[index]
    }

    let divisor = Double(Swift.max(count, 1))
    let rmsDifference = (sumSquaredDifference / divisor).squareRoot()
    let referenceRMS = (sumSquaredReference / divisor).squareRoot()
    let relative = referenceRMS > 0 ? rmsDifference / referenceRMS : 0
    let decibels = relative > 0 ? -20 * log10(relative) : .infinity

    return ParityResult(
      referenceVariant: referenceName,
      comparisonVariant: comparisonName,
      maxAbsoluteDifference: maxAbsolute,
      rmsDifference: rmsDifference,
      referenceRMS: referenceRMS,
      relativeRMS: relative,
      decibelsBelowSignal: decibels,
      withinTolerance: relative <= tolerance
    )
  }

  private func checkCancelled() throws {
    if isCancelled() { throw BenchmarkError.cancelled }
  }
}

// MARK: - Running on a suitable thread

public enum BenchmarkThread {
  /// Run `work` on a dedicated user-interactive thread.
  ///
  /// QoS matters more than usual here: on Apple silicon a background-QoS thread
  /// is scheduled onto the efficiency cores, and an E-core measurement would
  /// swamp every effect this benchmark is trying to resolve.
  public static func run(_ work: @escaping () -> Void) {
    let thread = Thread {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
      work()
    }
    thread.qualityOfService = .userInteractive
    thread.name = "NAMBench"
    thread.stackSize = 1 << 20
    thread.start()
  }
}
