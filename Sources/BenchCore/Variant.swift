import Foundation
import NAMEngineFull
import NAMEngineFused
import NAMEnginePlanar
import NAMEngineSlim
import NAMEngineUpstream

/// Which WaveNet implementation a config gets routed to.
public enum Engine: String, Sendable, Codable {
  case unknown
  case generic
  case a2Fast = "a2_fast"
  case fused
  case planar
  case slim
  case full

  init(_ raw: NbEngine) {
    switch raw.rawValue {
    case NbEngineGeneric.rawValue: self = .generic
    case NbEngineA2Fast.rawValue: self = .a2Fast
    case NbEngineFused.rawValue: self = .fused
    case NbEnginePlanar.rawValue: self = .planar
    case NbEngineSlim.rawValue: self = .slim
    case NbEngineFull.rawValue: self = .full
    default: self = .unknown
    }
  }

  public var displayName: String {
    switch self {
    case .unknown: return "unknown"
    case .generic: return "generic (Eigen WaveNet)"
    case .a2Fast: return "a2_fast (Eigen GEMM)"
    case .fused: return "fused (NEON)"
    case .planar: return "planar NEON (Core PR #313)"
    case .slim: return "slim lab kernel"
    case .full: return "full lab kernel"
    }
  }
}

/// What a `.nam` turned out to contain and how it will be run.
public struct VariantProbe: Sendable, Codable {
  public let engine: Engine
  public let channels: Int
  public let sampleRate: Double
  public let submodelIndex: Int
  public let submodelCount: Int
  public let submodelMaxValue: Double
}

public enum VariantError: LocalizedError {
  case probeFailed(variant: String, message: String)
  case createFailed(variant: String, message: String)
  case wrongEngine(variant: String, expected: Engine, actual: Engine)
  case wrongChannels(variant: String, expected: Int, actual: Int)
  case kernelSelectionFailed(variant: String, lab: KernelLab, kernel: Int)

  public var errorDescription: String? {
    switch self {
    case let .kernelSelectionFailed(variant, lab, kernel):
      return "\(variant): the \(lab.displayName) has no kernel at index \(kernel)"
    case let .probeFailed(variant, message):
      return "\(variant): probe failed — \(message)"
    case let .createFailed(variant, message):
      return "\(variant): could not build model — \(message)"
    case let .wrongEngine(variant, expected, actual):
      return """
        \(variant): expected the \(expected.displayName) engine but the model routed to \
        \(actual.displayName). Refusing to run — a benchmark of the wrong engine is worse \
        than no benchmark. The fused build falls through to the generic engine when a shape \
        is not accepted, which is exactly what this check exists to catch.
        """
    case let .wrongChannels(variant, expected, actual):
      return "\(variant): expected \(expected) channels (A2 full) but got \(actual)"
    }
  }
}

// MARK: - Bridging the two C APIs

/// The exported C entry points of one engine framework.
///
/// Both frameworks expose the same API under different symbol prefixes, so
/// binding them into a struct of function references lets one Swift class serve
/// both without duplicating the call sites.
struct VariantAPI {
  let hasFused: () -> Int32
  let probe: (UnsafePointer<UInt8>?, Int, NbSubmodel, Int32, UnsafeMutablePointer<NbProbe>?, UnsafeMutablePointer<CChar>?, Int) -> Int32
  let create: (UnsafePointer<UInt8>?, Int, NbSubmodel, Int32, Int32, UnsafeMutablePointer<CChar>?, Int) -> OpaquePointer?
  let destroy: (OpaquePointer?) -> Void
  let engine: (OpaquePointer?) -> NbEngine
  let channels: (OpaquePointer?) -> Int32
  let sampleRate: (OpaquePointer?) -> Double
  let reset: (OpaquePointer?, Double, Int32) -> Void
  let process: (OpaquePointer?, UnsafePointer<Double>?, Int, UnsafeMutablePointer<Double>?, UnsafeMutablePointer<Double>?) -> UInt64

  static let upstream = VariantAPI(
    hasFused: nb_upstream_has_fused,
    probe: nb_upstream_probe,
    create: nb_upstream_create,
    destroy: nb_upstream_destroy,
    engine: nb_upstream_engine,
    channels: nb_upstream_channels,
    sampleRate: nb_upstream_sample_rate,
    reset: nb_upstream_reset,
    process: nb_upstream_process
  )

  static let fused = VariantAPI(
    hasFused: nb_fused_has_fused,
    probe: nb_fused_probe,
    create: nb_fused_create,
    destroy: nb_fused_destroy,
    engine: nb_fused_engine,
    channels: nb_fused_channels,
    sampleRate: nb_fused_sample_rate,
    reset: nb_fused_reset,
    process: nb_fused_process
  )

  static let planar = VariantAPI(
    hasFused: nb_planar_has_fused,
    probe: nb_planar_probe,
    create: nb_planar_create,
    destroy: nb_planar_destroy,
    engine: nb_planar_engine,
    channels: nb_planar_channels,
    sampleRate: nb_planar_sample_rate,
    reset: nb_planar_reset,
    process: nb_planar_process
  )

  static let slim = VariantAPI(
    hasFused: nb_slim_has_fused,
    probe: nb_slim_probe,
    create: nb_slim_create,
    destroy: nb_slim_destroy,
    engine: nb_slim_engine,
    channels: nb_slim_channels,
    sampleRate: nb_slim_sample_rate,
    reset: nb_slim_reset,
    process: nb_slim_process
  )

  static let full = VariantAPI(
    hasFused: nb_full_has_fused,
    probe: nb_full_probe,
    create: nb_full_create,
    destroy: nb_full_destroy,
    engine: nb_full_engine,
    channels: nb_full_channels,
    sampleRate: nb_full_sample_rate,
    reset: nb_full_reset,
    process: nb_full_process
  )
}

/// Which in-project kernel lab a variant draws from.
///
/// The two labs target different submodels — `slim` the 3-channel one, `full`
/// the 8-channel one — and live in separate frameworks with separate kernel
/// tables, so a selection is only meaningful together with its lab.
public enum KernelLab: String, Sendable, Codable {
  case slim
  case full

  public var displayName: String {
    switch self {
    case .slim: return "slim lab"
    case .full: return "full lab"
    }
  }
}

/// One lab's kernel table, read out of the framework rather than duplicated
/// here — the C side is the only place either list exists.
public enum LabKernels {
  public static func count(_ lab: KernelLab) -> Int {
    switch lab {
    case .slim: return Int(nb_slim_kernel_count())
    case .full: return Int(nb_full_kernel_count())
    }
  }

  public static func names(_ lab: KernelLab) -> [String] {
    (0..<count(lab)).map { name(lab, $0) }
  }

  public static func name(_ lab: KernelLab, _ index: Int) -> String {
    let raw: UnsafePointer<CChar>?
    switch lab {
    case .slim: raw = nb_slim_kernel_name(Int32(index))
    case .full: raw = nb_full_kernel_name(Int32(index))
    }
    return raw.map(String.init(cString:)) ?? "kernel \(index)"
  }

  public static func index(_ lab: KernelLab, named name: String) -> Int? {
    names(lab).firstIndex(of: name)
  }
}

/// The slim lab's table, kept as its own name because the CLI, the app and the
/// existing reports all speak of "slim kernels".
public enum SlimKernels {
  public static var count: Int { LabKernels.count(.slim) }
  public static var names: [String] { LabKernels.names(.slim) }
  public static func name(_ index: Int) -> String { LabKernels.name(.slim, index) }
  public static func index(named name: String) -> Int? { LabKernels.index(.slim, named: name) }
}

/// The full lab's table.
public enum FullKernels {
  public static var count: Int { LabKernels.count(.full) }
  public static var names: [String] { LabKernels.names(.full) }
  public static func name(_ index: Int) -> String { LabKernels.name(.full, index) }
  public static func index(named name: String) -> Int? { LabKernels.index(.full, named: name) }
}

/// Which lab kernel a variant selects before building, if any.
public struct LabSelection: Sendable, Codable, Equatable {
  public let lab: KernelLab
  public let index: Int

  public init(lab: KernelLab, index: Int) {
    self.lab = lab
    self.index = index
  }

  public var kernelName: String { LabKernels.name(lab, index) }
}

/// One code variation under test.
public final class Variant {
  public let name: String
  public let repository: String
  public let codePath: String
  public let expectedEngine: Engine
  /// Which lab kernel this variant selects before building, if any.
  public let labKernel: LabSelection?

  private let api: VariantAPI

  init(
    name: String,
    repository: String,
    codePath: String,
    expectedEngine: Engine,
    api: VariantAPI,
    labKernel: LabSelection? = nil
  ) {
    self.name = name
    self.repository = repository
    self.codePath = codePath
    self.expectedEngine = expectedEngine
    self.api = api
    self.labKernel = labKernel
  }

  /// Upstream NeuralAmpModelerCore. It has no fused engine, so
  /// `create_config` routes the A2 shape to `a2_fast` on its own.
  public static let upstream = Variant(
    name: "upstream",
    repository: "sdatkinson/NeuralAmpModelerCore",
    codePath: "a2_fast",
    expectedEngine: .a2Fast,
    api: .upstream
  )

  /// The planar NEON kernels, as proposed to Core in PR #313.
  ///
  /// Built from the head of that draft PR rather than from a copy of it, so a
  /// number measured here belongs to the proposal being discussed. Unlike
  /// `fused` these cover both A2 submodels — 3 channels and 8 — which is why
  /// they replaced it in the line-up rather than joining it.
  ///
  /// `expectedEngine` is `.planar`, and `makeModel` refuses to run when the
  /// model routes anywhere else. That check earns its keep here more than
  /// anywhere: `NAM_A2_PLANAR` is gated on `__APPLE__ && __aarch64__`, so on any
  /// other target this framework builds happily and *is* `a2_fast`. Without the
  /// assertion the run would measure a2_fast twice and read as the kernels being
  /// worthless rather than absent.
  public static let planar = Variant(
    name: "planar",
    repository: "rikkus/OptimisationWorkOnNeuralAmpModelerCore@apple-silicon-a2-planar",
    codePath: "a2_planar",
    expectedEngine: .planar,
    api: .planar
  )

  /// The optimisation fork, built with `NAM_ENABLE_FUSED` and constructed
  /// under `ScopedEnginePrefer(FusedNeon)`.
  ///
  /// Superseded by `planar` and no longer in the default line-up. Kept because
  /// the full lab's `fu*` candidates are validated against it, and because a
  /// result that has been published should stay reproducible.
  public static let fused = Variant(
    name: "fused",
    repository: "rikkus/OptimisationWorkOnNeuralAmpModelerCore",
    codePath: "fused",
    expectedEngine: .fused,
    api: .fused
  )

  /// One experimental kernel from the slim lab, presented as a variant so it
  /// goes through exactly the same measurement path as the two shipping
  /// engines. Kernel 0 is `baseline`, the verbatim port of a2_fast's 3-channel
  /// branch that the lab is validated against.
  public static func slim(kernel: Int) -> Variant {
    Variant(
      name: "slim:\(SlimKernels.name(kernel))",
      repository: "NAMBench/Sources/SlimEngines",
      codePath: SlimKernels.name(kernel),
      expectedEngine: .slim,
      api: .slim,
      labKernel: LabSelection(lab: .slim, index: kernel)
    )
  }

  /// One experimental kernel from the full lab. Kernel 0 is `a2_baseline`, the
  /// verbatim port of a2_fast's Channels==8 branch, and kernel 1 is
  /// `fu_baseline`, the verbatim port of fused's C=8 path — the two controls the
  /// lab is validated against.
  public static func full(kernel: Int) -> Variant {
    Variant(
      name: "full:\(FullKernels.name(kernel))",
      repository: "NAMBench/Sources/FullEngines",
      codePath: FullKernels.name(kernel),
      expectedEngine: .full,
      api: .full,
      labKernel: LabSelection(lab: .full, index: kernel)
    )
  }

  /// The default line-up: the reference, and the kernels proposed to replace it.
  ///
  /// `fused` was here until the planar kernels superseded it. It is still
  /// buildable and still selectable by name; it is simply no longer the thing
  /// being proposed, and a line-up is a claim about what is worth measuring.
  public static let all: [Variant] = [.upstream, .planar]

  public var hasFusedEngine: Bool { api.hasFused() != 0 }

  /// Point this variant's framework at its kernel, if it has one. Must happen
  /// before probe or create, because until it does the slim framework reports
  /// (and builds) a2_fast exactly as `upstream` would.
  private func selectKernel() throws {
    guard let labKernel else { return }
    let status: Int32
    switch labKernel.lab {
    case .slim: status = nb_slim_select_kernel(Int32(labKernel.index))
    case .full: status = nb_full_select_kernel(Int32(labKernel.index))
    }
    guard status == 0 else {
      throw VariantError.kernelSelectionFailed(variant: name, lab: labKernel.lab, kernel: labKernel.index)
    }
  }

  public func probe(namBytes: Data, submodel: SubmodelSelection) throws -> VariantProbe {
    try selectKernel()

    var result = NbProbe()
    var error = [CChar](repeating: 0, count: 512)
    let (mode, index) = submodel.cValues

    let status = namBytes.withUnsafeBytes { raw -> Int32 in
      let base = raw.bindMemory(to: UInt8.self).baseAddress
      return api.probe(base, namBytes.count, mode, index, &result, &error, error.count)
    }

    guard status == 0 else {
      throw VariantError.probeFailed(variant: name, message: String(cString: error))
    }

    return VariantProbe(
      engine: Engine(result.engine),
      channels: Int(result.channels),
      sampleRate: result.sampleRate,
      submodelIndex: Int(result.submodelIndex),
      submodelCount: Int(result.submodelCount),
      submodelMaxValue: result.submodelMaxValue
    )
  }

  /// Build the model and verify it routed where we expect.
  ///
  /// The engine assertion is the reason this is not just `create`: the fork
  /// silently falls back to the generic engine for shapes it does not accept,
  /// and a "fused" number that was secretly generic would read as a massive
  /// regression rather than a bug in the harness. The same check catches a slim
  /// variant whose kernel selection did not take.
  public func makeModel(namBytes: Data, config: BenchmarkConfig, expectedChannels: Int) throws -> VariantModel {
    try selectKernel()

    var error = [CChar](repeating: 0, count: 512)
    let (mode, index) = config.submodel.cValues

    let handle = namBytes.withUnsafeBytes { raw -> OpaquePointer? in
      let base = raw.bindMemory(to: UInt8.self).baseAddress
      return api.create(base, namBytes.count, mode, index, Int32(config.blockSize), &error, error.count)
    }

    guard let handle else {
      throw VariantError.createFailed(variant: name, message: String(cString: error))
    }

    let model = VariantModel(handle: handle, api: api, variantName: name, blockSize: config.blockSize)

    guard model.engine == expectedEngine else {
      throw VariantError.wrongEngine(variant: name, expected: expectedEngine, actual: model.engine)
    }
    guard model.channels == expectedChannels else {
      throw VariantError.wrongChannels(variant: name, expected: expectedChannels, actual: model.channels)
    }

    return model
  }
}

private extension SubmodelSelection {
  var cValues: (NbSubmodel, Int32) {
    switch self {
    case .widest: return (NbSubmodelWidest, 0)
    case .narrowest: return (NbSubmodelNarrowest, 0)
    case let .index(value): return (NbSubmodelIndex, Int32(value))
    }
  }
}

/// A live model belonging to one variant.
public final class VariantModel {
  private let handle: OpaquePointer
  private let api: VariantAPI
  private var scratch: [Double]

  public let variantName: String
  public let blockSize: Int

  init(handle: OpaquePointer, api: VariantAPI, variantName: String, blockSize: Int) {
    self.handle = handle
    self.api = api
    self.variantName = variantName
    self.blockSize = blockSize
    self.scratch = []
  }

  deinit { api.destroy(handle) }

  public var engine: Engine { Engine(api.engine(handle)) }
  public var channels: Int { Int(api.channels(handle)) }
  public var sampleRate: Double { api.sampleRate(handle) }

  /// Clear state so the next pass starts identically. Never prewarms, and
  /// callers keep it outside the timed region.
  public func reset() {
    api.reset(handle, sampleRate, Int32(blockSize))
  }

  /// One full pass over `input`, returning the elapsed nanoseconds measured
  /// inside the shim.
  ///
  /// `captureOutput` and `wantChecksum` are off during timed passes so the
  /// measured region contains nothing but `process()`.
  @discardableResult
  public func process(
    input: [Double],
    captureOutput: Bool = false,
    wantChecksum: Bool = false
  ) -> (nanoseconds: UInt64, output: [Double]?, checksum: Double?) {
    var output: [Double] = captureOutput ? [Double](repeating: 0, count: input.count) : []
    var checksum: Double = 0
    var elapsed: UInt64 = 0

    input.withUnsafeBufferPointer { inputBuffer in
      guard let inputBase = inputBuffer.baseAddress else { return }

      func run(_ outputBase: UnsafeMutablePointer<Double>?) {
        if wantChecksum {
          elapsed = api.process(handle, inputBase, input.count, outputBase, &checksum)
        } else {
          elapsed = api.process(handle, inputBase, input.count, outputBase, nil)
        }
      }

      if captureOutput {
        output.withUnsafeMutableBufferPointer { run($0.baseAddress) }
      } else {
        run(nil)
      }
    }

    return (elapsed, captureOutput ? output : nil, wantChecksum ? checksum : nil)
  }
}
