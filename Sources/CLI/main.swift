import BenchCore
import Foundation

// Headless macOS runner. Same BenchCore, same protocol, same reports as the app.

struct Options {
  var modelPath: String?
  var audioPath: String?
  var outputDirectory: String?
  var pinsPath: String?
  var config = BenchmarkConfig()
  var quiet = false

  static func parse(_ arguments: [String]) throws -> Options {
    var options = Options()
    var index = 0

    func next(_ flag: String) throws -> String {
      index += 1
      guard index < arguments.count else { throw CLIError.missingValue(flag) }
      return arguments[index]
    }

    while index < arguments.count {
      let argument = arguments[index]
      switch argument {
      case "--model", "-m": options.modelPath = try next(argument)
      case "--audio", "-a": options.audioPath = try next(argument)
      case "--output", "-o": options.outputDirectory = try next(argument)
      case "--pins": options.pinsPath = try next(argument)
      case "--block-size": options.config.blockSize = Int(try next(argument)) ?? 64
      case "--submodel": options.config.submodel = try parseSubmodel(try next(argument))
      case "--slim": options.config.slimKernels = try parseSlimKernels(try next(argument))
      case "--list-slim": listSlimKernels(); exit(0)
      case "--timing-seconds": options.config.timingWindowSeconds = Double(try next(argument)) ?? 10
      case "--warmup-seconds": options.config.warmupSeconds = Double(try next(argument)) ?? 5
      case "--accept-tolerance": options.config.acceptTolerance = Double(try next(argument)) ?? 0.01
      case "--accept-fraction": options.config.acceptFraction = Double(try next(argument)) ?? 0.70
      case "--max-attempts": options.config.maxAttempts = Int(try next(argument)) ?? 5
      case "--no-parity": options.config.checkParity = false
      case "--quiet", "-q": options.quiet = true
      case "--help", "-h": printUsage(); exit(0)
      default: throw CLIError.unknownFlag(argument)
      }
      index += 1
    }
    return options
  }
}

enum CLIError: LocalizedError {
  case missingValue(String)
  case unknownFlag(String)
  case resourceNotFound(String)
  case badSubmodel(String)
  case unknownKernel(String)

  var errorDescription: String? {
    switch self {
    case let .missingValue(flag): return "\(flag) needs a value"
    case let .unknownFlag(flag): return "unknown option \(flag)"
    case let .resourceNotFound(what): return "could not find \(what)"
    case let .badSubmodel(value):
      return "--submodel takes widest, narrowest or an index; got \(value)"
    case let .unknownKernel(name):
      return "no slim kernel called \(name). Known: \(SlimKernels.names.joined(separator: ", "))"
    }
  }
}

func parseSubmodel(_ value: String) throws -> SubmodelSelection {
  switch value.lowercased() {
  case "widest", "full": return .widest
  case "narrowest", "slim", "slimmed", "nano": return .narrowest
  default:
    guard let index = Int(value), index >= 0 else { throw CLIError.badSubmodel(value) }
    return .index(index)
  }
}

func parseSlimKernels(_ value: String) throws -> [Int] {
  if value.lowercased() == "all" { return Array(0..<SlimKernels.count) }
  if value.lowercased() == "none" || value.isEmpty { return [] }
  return try value.split(separator: ",").map { piece in
    let name = piece.trimmingCharacters(in: .whitespaces)
    if let index = Int(name), index >= 0, index < SlimKernels.count { return index }
    guard let index = SlimKernels.index(named: name) else { throw CLIError.unknownKernel(name) }
    return index
  }
}

func listSlimKernels() {
  print("slim lab kernels (\(SlimKernels.count)):")
  for (index, name) in SlimKernels.names.enumerated() {
    print(String(format: "  %2d  %@", index, name))
  }
}

func printUsage() {
  print("""
    nambench — compare NAM A2 WaveNet engines

      --model, -m <path>        .nam capture (default: nam-files/*.nam)
      --audio, -a <path>        input .wav  (default: audio-input/input.wav)
      --output, -o <dir>        where to write reports (default: ./benchmark-results)
      --pins <path>             vendor/pins.json for provenance

      --submodel <which>        widest (default, A2 full) | narrowest (A2 nano) | <index>
      --slim <list>             slim-lab kernels to include: all, none, or a
                                comma-separated list of names or indices
      --list-slim               print the kernel table and exit

      --block-size <n>          frames per process() call (default 64)
      --warmup-seconds <s>      discarded warm-up passes (default 5)
      --timing-seconds <s>      timing window (default 30)
      --accept-fraction <r>     fraction of samples kept (default 0.70)
      --accept-tolerance <r>    required agreement, e.g. 0.01 for 1%
      --max-attempts <n>        attempts before giving up (default 5)
      --no-parity               skip the output comparison
      --quiet, -q               only print the summary

    On the slimmed submodel the fused engine is not part of the comparison: its
    detector rejects a channel count that is not a multiple of four, and under
    ScopedEnginePrefer(FusedNeon) that falls through to the *generic* engine
    rather than to a2_fast. The baseline there is `upstream`.
    """)
}

// MARK: - Resource discovery

/// Walk up from `start` looking for a directory containing `marker`.
func findUpwards(_ marker: String, from start: URL) -> URL? {
  var directory = start
  for _ in 0..<8 {
    if FileManager.default.fileExists(atPath: directory.appendingPathComponent(marker).path) {
      return directory
    }
    let parent = directory.deletingLastPathComponent()
    if parent.path == directory.path { break }
    directory = parent
  }
  return nil
}

func defaultModelURL() throws -> URL {
  let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  guard let root = findUpwards("nam-files", from: cwd) else {
    throw CLIError.resourceNotFound("a nam-files directory (pass --model)")
  }
  let directory = root.appendingPathComponent("nam-files")
  let contents = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
  // Prefer the A2 capture the benchmark is specified around.
  let preferred = contents.first { $0.lastPathComponent.hasPrefix("Ampeg SVT - Gain 10") }
  guard let model = preferred ?? contents.first(where: { $0.pathExtension == "nam" }) else {
    throw CLIError.resourceNotFound("a .nam file in \(directory.path)")
  }
  return model
}

func defaultAudioURL() throws -> URL {
  let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  guard let root = findUpwards("audio-input", from: cwd) else {
    throw CLIError.resourceNotFound("an audio-input directory (pass --audio)")
  }
  return root.appendingPathComponent("audio-input/input.wav")
}

// MARK: - Output

let standardError = FileHandle.standardError

func fail(_ message: String) -> Never {
  standardError.write("error: \(message)\n".data(using: .utf8)!)
  exit(1)
}

/// True when stdout is a terminal. Piping to a file or CI log means the
/// in-place progress line would just accumulate escape codes, so it becomes a
/// plain line instead.
let stdoutIsTerminal = isatty(fileno(stdout)) == 1

/// Progress that overwrites itself on a terminal, and is suppressed entirely
/// when redirected — the per-pass detail is in the JSON report either way.
func progress(_ text: String) {
  guard stdoutIsTerminal else { return }
  print("\r\u{1B}[K\(text)", terminator: "")
  fflush(stdout)
}

/// Finish a progress line so the next real line starts cleanly.
func endProgress() {
  if stdoutIsTerminal { print("") }
}

// MARK: - Main

do {
  let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))

  let modelURL = try options.modelPath.map(URL.init(fileURLWithPath:)) ?? defaultModelURL()
  let audioURL = try options.audioPath.map(URL.init(fileURLWithPath:)) ?? defaultAudioURL()

  let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  let pinsURL = options.pinsPath.map(URL.init(fileURLWithPath:))
    ?? findUpwards("vendor", from: cwd)?.appendingPathComponent("vendor/pins.json")
  let outputURL = options.outputDirectory.map(URL.init(fileURLWithPath:))
    ?? cwd.appendingPathComponent("benchmark-results")

  let quiet = options.quiet

  // Assemble the line-up before anything is measured.
  //
  // `fused` is only in it when the selected submodel's channel count is a
  // multiple of four. On a 3-channel submodel its detector rejects the shape,
  // and under ScopedEnginePrefer(FusedNeon) that falls through to the *generic*
  // engine, not to a2_fast — so including it would either abort on the engine
  // assertion or, worse, quietly measure the wrong thing. The channel count is
  // read from the file rather than inferred from the flag.
  let namBytes = try Data(contentsOf: modelURL)
  let shape = try Variant.upstream.probe(namBytes: namBytes, submodel: options.config.submodel)

  var variants: [Variant] = [.upstream]
  if shape.channels % 4 == 0 {
    variants.append(.fused)
  } else if !quiet {
    print("fused excluded: \(shape.channels) channels is not a multiple of 4, so the fused "
      + "detector rejects this shape and the fork falls through to the generic engine")
  }
  variants += options.config.slimKernels.map { Variant.slim(kernel: $0) }

  let runner = BenchmarkRunner(
    config: options.config,
    variants: variants,
    modelURL: modelURL,
    audioURL: audioURL,
    pinsURL: pinsURL
  )

  var lastVariant = ""

  let report = try runner.run { event in
    switch event {
    case let .status(message):
      if !quiet { print(message) }

    case let .probed(variant, probe):
      if !quiet {
        print("\(variant): \(probe.engine.displayName), \(probe.channels) channels, "
          + "submodel \(probe.submodelIndex + 1)/\(probe.submodelCount) "
          + "(max_value \(String(format: "%.2f", probe.submodelMaxValue)))")
      }

    case let .parityMeasured(parity):
      if !quiet {
        print(String(format: "parity %@ vs %@: max|diff| %.3e, rms %.3e (%@)%@",
                     parity.referenceVariant, parity.comparisonVariant,
                     parity.maxAbsoluteDifference, parity.rmsDifference,
                     parity.decibelsBelowSignal.isInfinite
                       ? "bit-identical"
                       : String(format: "%.1f dB below signal", parity.decibelsBelowSignal),
                     parity.withinTolerance ? "" : "  ** OUT OF TOLERANCE **"))
      }

    case let .warmupStarted(variant, attempt):
      lastVariant = variant
      if !quiet { print("\n\(variant): warming up (attempt \(attempt))") }

    case let .warmupPass(_, pass, milliseconds, remaining):
      if !quiet {
        progress(String(format: "  pass %d: %.2f ms  (%.1f s left)", pass, milliseconds, remaining))
      }

    case let .warmupFinished(_, passes):
      if !quiet {
        progress("  warmed up over \(passes) discarded passes")
        endProgress()
      }

    case let .timingStarted(variant, _):
      if !quiet { print("\(variant): timing") }

    case let .timingPass(_, sample, milliseconds, remaining):
      if !quiet {
        progress(String(format: "  sample %d: %.2f ms  (%.1f s left)", sample, milliseconds, remaining))
      }

    case let .attemptRejected(variant, attempt, reason):
      if !quiet { endProgress() }
      print("\(variant): attempt \(attempt) rejected — \(reason)")

    case let .variantFinished(result):
      if !quiet { endProgress() }
      if result.succeeded {
        print(String(format: "%@: %.2f ms  (median %.2f, spread %.2f%%, RTF %.1fx, %d accepted / %d discarded)",
                     result.variant, result.meanMs, result.medianMs, result.spread * 100,
                     result.realTimeFactor, result.acceptedMs.count, result.discardedMs.count))
      } else {
        print("\(result.variant): FAILED — \(result.failureReason ?? "unknown")")
      }

    case .finished:
      break
    }
    _ = lastVariant
  }

  // Summary. Failed variants are listed too — a variant silently missing from
  // the table would read as "not run" rather than "rejected as too noisy".
  print("")
  print("==== Result ====")
  let nameWidth = max(10, report.results.map(\.variant.count).max() ?? 10)
  let speedupByVariant = Dictionary(
    report.speedups.map { ($0.variant, $0.factor) }, uniquingKeysWith: { first, _ in first }
  )
  let parityByVariant = Dictionary(
    report.parities.map { ($0.comparisonVariant, $0) }, uniquingKeysWith: { first, _ in first }
  )
  let baselineName = report.results.first?.variant ?? ""

  for result in report.results {
    let name = result.variant.padding(toLength: nameWidth, withPad: " ", startingAt: 0)
    guard result.succeeded else {
      print("  \(name)   REJECTED — \(result.failureReason ?? "unknown")")
      continue
    }
    // The minimum is shown next to the mean because it is the most stable
    // number available: interference is additive, so the fastest pass is the
    // least contaminated estimate of the real cost.
    var line = String(format: "  %@ %8.2f ms  (min %.2f)  RTF %5.1fx",
                      name, result.meanMs, result.minMs, result.realTimeFactor)
    if let factor = speedupByVariant[result.variant] {
      line += String(format: "  %6.3fx vs %@", factor, baselineName)
    }
    if let parity = parityByVariant[result.variant] {
      line += parity.decibelsBelowSignal.isInfinite
        ? "  parity exact"
        : String(format: "  parity %.1f dB%@", parity.decibelsBelowSignal,
                 parity.withinTolerance ? "" : " **OUT OF TOLERANCE**")
    }
    print(line)
  }
  if !report.results.allSatisfy(\.succeeded) {
    print("  at least one variant produced no usable result")
  }

  let written = try Reporting.write(report, to: outputURL)
  print("")
  print("  \(written.json.path)")
  print("  \(written.markdown.path)")

  let allSucceeded = report.results.allSatisfy(\.succeeded)
  exit(allSucceeded ? 0 : 1)
} catch {
  fail(error.localizedDescription)
}
