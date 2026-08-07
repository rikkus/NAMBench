import Foundation

// MARK: - Environment

/// Where the numbers were taken. Timings only mean something alongside this.
public struct EnvironmentInfo: Sendable, Codable {
  public let deviceModel: String
  public let cpu: String
  public let performanceCores: Int?
  public let efficiencyCores: Int?
  public let totalCores: Int
  public let memoryBytes: UInt64
  public let osVersion: String
  public let platform: String
  public let architecture: String
  public let isSimulator: Bool

  public static func capture() -> EnvironmentInfo {
    let processInfo = ProcessInfo.processInfo

    #if targetEnvironment(simulator)
      let simulator = true
    #else
      let simulator = false
    #endif

    #if os(macOS)
      let platform = "macOS"
    #elseif os(iOS)
      let platform = "iOS"
    #else
      let platform = "unknown"
    #endif

    #if arch(arm64)
      let architecture = "arm64"
    #elseif arch(x86_64)
      let architecture = "x86_64"
    #else
      let architecture = "unknown"
    #endif

    return EnvironmentInfo(
      deviceModel: sysctlString("hw.model") ?? sysctlString("hw.machine") ?? "unknown",
      cpu: sysctlString("machdep.cpu.brand_string") ?? "unknown",
      performanceCores: sysctlInt("hw.perflevel0.logicalcpu"),
      efficiencyCores: sysctlInt("hw.perflevel1.logicalcpu"),
      totalCores: processInfo.processorCount,
      memoryBytes: processInfo.physicalMemory,
      osVersion: processInfo.operatingSystemVersionString,
      platform: platform,
      architecture: architecture,
      isSimulator: simulator
    )
  }

  private static func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
    return String(cString: buffer)
  }

  private static func sysctlInt(_ name: String) -> Int? {
    var value: Int = 0
    var size = MemoryLayout<Int>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    return value
  }
}

/// The commits the code under test was built from.
public struct VendorPins: Sendable, Codable {
  public struct Pin: Sendable, Codable {
    public let url: String
    public let sha: String
    public let shared: Bool?
  }

  public let upstream: Pin
  public let fused: Pin
  public let eigen: Pin

  public static func load(from url: URL) -> VendorPins? {
    guard let data = try? Data(contentsOf: url) else { return nil }
    return try? JSONDecoder().decode(VendorPins.self, from: data)
  }
}

// MARK: - Results

/// One warm-up + timing attempt for a variant.
public struct AttemptRecord: Sendable, Codable {
  public let attempt: Int
  /// Discarded passes run before timing began.
  public let warmupPasses: Int
  /// Every pass time in the timing window, in milliseconds, in the order taken.
  public let samplesMs: [Double]
  public let accepted: Bool
  public let spread: Double?
  public let rejectionReason: String?
}

public struct VariantResult: Sendable, Codable {
  public let variant: String
  public let repository: String
  public let codePath: String
  public let engine: Engine
  public let channels: Int
  public let probe: VariantProbe
  public let attempts: [AttemptRecord]
  public let succeeded: Bool
  public let failureReason: String?

  /// Samples inside the accepted window, milliseconds.
  public let acceptedMs: [Double]
  /// Samples rejected as outliers, milliseconds.
  public let discardedMs: [Double]

  /// The headline number: mean of the accepted window, milliseconds.
  public let meanMs: Double
  public let medianMs: Double
  public let minMs: Double
  public let maxMs: Double
  public let spread: Double
  public let standardDeviationMs: Double

  /// Audio seconds processed per wall-clock second.
  public let realTimeFactor: Double
  public let checksum: Double
}

/// How much faster one variant was than the run's baseline.
public struct VariantSpeedup: Sendable, Codable {
  public let variant: String
  public let baseline: String
  /// baseline mean ÷ this variant's mean. Above 1 means faster.
  public let factor: Double
}

public struct ParityResult: Sendable, Codable {
  public let referenceVariant: String
  public let comparisonVariant: String
  public let maxAbsoluteDifference: Double
  public let rmsDifference: Double
  public let referenceRMS: Double
  public let relativeRMS: Double
  public let decibelsBelowSignal: Double
  public let withinTolerance: Bool
}

public struct AudioInfo: Sendable, Codable {
  public let fileName: String
  public let frameCount: Int
  public let sampleRate: Double
  public let durationSeconds: Double
}

public struct ModelInfo: Sendable, Codable {
  public let fileName: String
  public let byteCount: Int
  public let submodelIndex: Int
  public let submodelCount: Int
  public let submodelMaxValue: Double
  public let channels: Int
  public let sampleRate: Double
}

/// Everything one run produced.
public struct BenchmarkReport: Sendable, Codable {
  public static let schemaVersion = 1

  public let schemaVersion: Int
  public let startedAt: Date
  public let finishedAt: Date
  public let environment: EnvironmentInfo
  public let pins: VendorPins?
  public let config: BenchmarkConfig
  public let model: ModelInfo
  public let audio: AudioInfo
  /// Conditions around the run. Recorded for diagnosis, not gated on.
  public let systemAtStart: SystemState
  public let systemAtEnd: SystemState
  public let results: [VariantResult]
  /// The first parity comparison, kept so two-variant reports read as before.
  public let parity: ParityResult?
  /// Every variant after the first, compared against the first.
  public let parities: [ParityResult]

  /// Baseline mean ÷ candidate mean, only when the run had exactly two
  /// variants. Multi-variant runs use `speedups`.
  public let speedup: Double?
  /// One entry per variant after the first.
  public let speedups: [VariantSpeedup]
}
