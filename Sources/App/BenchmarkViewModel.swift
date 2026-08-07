import BenchCore
import Combine
import Foundation

/// Live state of one variant while it is being measured.
struct VariantState: Identifiable {
  enum Phase: String {
    case pending = "waiting"
    case warmingUp = "warming up"
    case timing = "timing"
    case done = "done"
    case failed = "rejected"
  }

  var id: String { name }
  let name: String
  let repository: String
  let codePath: String

  var engine: Engine = .unknown
  var channels: Int = 0
  var phase: Phase = .pending
  var attempt: Int = 1
  var passCount: Int = 0
  var lastMilliseconds: Double = 0
  var secondsRemaining: Double?
  /// Recent pass times, for the sparkline.
  var recent: [Double] = []
  var result: VariantResult?

  mutating func record(_ milliseconds: Double) {
    lastMilliseconds = milliseconds
    recent.append(milliseconds)
    if recent.count > 60 { recent.removeFirst(recent.count - 60) }
  }
}

@MainActor
final class BenchmarkViewModel: ObservableObject {
  enum Phase: Equatable {
    case ready
    case preparing
    case checkingParity
    case running
    case finished
    case failed(String)
  }

  @Published private(set) var phase: Phase = .ready
  @Published private(set) var log: [String] = []
  @Published private(set) var variants: [VariantState] = []
  @Published private(set) var parity: ParityResult?
  @Published private(set) var report: BenchmarkReport?
  @Published private(set) var writtenFiles: Reporting.WrittenFiles?
  @Published var config = BenchmarkConfig()

  private var cancelFlag = false

  var isRunning: Bool {
    switch phase {
    case .ready, .finished, .failed: return false
    default: return true
    }
  }

  var modelName: String { modelURL?.lastPathComponent ?? "no model found" }
  var audioName: String { audioURL?.lastPathComponent ?? "no audio found" }

  /// `nam-files` is bundled as a folder reference, so the captures keep their
  /// directory rather than landing at the bundle root — search both.
  private var modelURL: URL? {
    let candidates = (Bundle.main.urls(forResourcesWithExtension: "nam", subdirectory: "nam-files") ?? [])
      + (Bundle.main.urls(forResourcesWithExtension: "nam", subdirectory: nil) ?? [])
    let sorted = candidates.sorted { $0.lastPathComponent < $1.lastPathComponent }
    // The A2 capture this benchmark is specified around, else whatever is there.
    return sorted.first { $0.lastPathComponent.hasPrefix("Ampeg SVT - Gain 10") } ?? sorted.first
  }

  private var audioURL: URL? {
    Bundle.main.url(forResource: "input", withExtension: "wav")
  }

  private var pinsURL: URL? {
    Bundle.main.url(forResource: "pins", withExtension: "json")
  }

  init() {
    resetVariants()
  }

  private func resetVariants() {
    variants = Variant.all.map {
      VariantState(name: $0.name, repository: $0.repository, codePath: $0.codePath)
    }
  }

  func cancel() {
    cancelFlag = true
    append("Cancelling…")
  }

  func start() {
    guard !isRunning else { return }
    guard let modelURL, let audioURL else {
      phase = .failed("Bundled model or audio is missing from the app resources.")
      return
    }

    cancelFlag = false
    log = []
    parity = nil
    report = nil
    writtenFiles = nil
    resetVariants()
    phase = .preparing

    let config = config
    let pinsURL = pinsURL

    // A dedicated user-interactive thread: on Apple silicon a lower QoS lands
    // on the efficiency cores, which would dominate everything being measured.
    BenchmarkThread.run { [weak self] in
      let runner = BenchmarkRunner(
        config: config,
        modelURL: modelURL,
        audioURL: audioURL,
        pinsURL: pinsURL,
        isCancelled: {
          guard let self else { return true }
          return DispatchQueue.main.sync { self.cancelFlag }
        }
      )

      do {
        let finished = try runner.run { event in
          DispatchQueue.main.async { self?.handle(event) }
        }
        DispatchQueue.main.async { self?.finish(with: finished) }
      } catch {
        DispatchQueue.main.async {
          self?.phase = .failed(error.localizedDescription)
          self?.append("Failed: \(error.localizedDescription)")
        }
      }
    }
  }

  // MARK: - Events

  private func handle(_ event: BenchmarkEvent) {
    switch event {
    case let .status(message):
      append(message)

    case let .probed(variant, probe):
      update(variant) {
        $0.engine = probe.engine
        $0.channels = probe.channels
      }
      append("\(variant): \(probe.engine.displayName), \(probe.channels) channels, "
        + "submodel \(probe.submodelIndex + 1) of \(probe.submodelCount)")

    case let .parityMeasured(result):
      phase = .checkingParity
      parity = result
      append(String(format: "Parity: %.1f dB below signal", result.decibelsBelowSignal))

    case let .warmupStarted(variant, attempt):
      phase = .running
      update(variant) {
        $0.phase = .warmingUp
        $0.attempt = attempt
        $0.passCount = 0
        $0.recent = []
        $0.secondsRemaining = nil
      }

    case let .warmupPass(variant, pass, milliseconds, remaining):
      update(variant) {
        $0.passCount = pass
        $0.secondsRemaining = remaining
        $0.record(milliseconds)
      }

    case let .warmupFinished(variant, passes):
      append("\(variant): warmed up over \(passes) discarded passes")

    case let .timingStarted(variant, _):
      update(variant) {
        $0.phase = .timing
        $0.passCount = 0
        $0.recent = []
      }

    case let .timingPass(variant, sample, milliseconds, remaining):
      update(variant) {
        $0.passCount = sample
        $0.secondsRemaining = remaining
        $0.record(milliseconds)
      }

    case let .attemptRejected(variant, attempt, reason):
      append("\(variant): attempt \(attempt) rejected — \(reason)")

    case let .variantFinished(result):
      update(result.variant) {
        $0.phase = result.succeeded ? .done : .failed
        $0.result = result
        $0.secondsRemaining = nil
      }

    case .finished:
      break
    }
  }

  private func finish(with report: BenchmarkReport) {
    self.report = report
    phase = .finished
    do {
      writtenFiles = try Reporting.write(report, to: Reporting.defaultOutputDirectory())
      append("Wrote reports to \(Reporting.defaultOutputDirectory().path)")
    } catch {
      append("Could not write reports: \(error.localizedDescription)")
    }
  }

  private func update(_ name: String, _ mutate: (inout VariantState) -> Void) {
    guard let index = variants.firstIndex(where: { $0.name == name }) else { return }
    mutate(&variants[index])
  }

  private func append(_ message: String) {
    log.append(message)
    if log.count > 200 { log.removeFirst(log.count - 200) }
  }
}
