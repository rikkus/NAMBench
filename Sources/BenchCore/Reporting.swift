import Foundation

/// Turns a finished run into files worth keeping.
public enum Reporting {
  // MARK: - JSON

  public static func json(for report: BenchmarkReport) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    // A bit-identical variant has zero RMS difference, so its "dB below signal"
    // is genuinely infinite. JSON has no literal for that and JSONEncoder
    // throws rather than guess, which would otherwise lose the whole report at
    // the last step — over the best possible parity result.
    encoder.nonConformingFloatEncodingStrategy = .convertToString(
      positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan"
    )
    return try encoder.encode(report)
  }

  // MARK: - Markdown

  /// Styled to drop straight into the fork's `benchmark_reports/`.
  public static func markdown(for report: BenchmarkReport) -> String {
    var out = ""

    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd"
    dateFormatter.locale = Locale(identifier: "en_US_POSIX")

    out += "# \(workloadName(report)) — \(variantHeadline(report))\n\n"
    out += "Date:       \(dateFormatter.string(from: report.startedAt))  \n"
    out += "Machine:    \(report.environment.cpu) (\(report.environment.architecture))"
    if let performance = report.environment.performanceCores,
       let efficiency = report.environment.efficiencyCores {
      out += ", \(performance)P + \(efficiency)E"
    }
    out += "  \n"
    out += "OS:         \(report.environment.platform) — \(report.environment.osVersion)  \n"
    out += "Workload:   `\(report.model.fileName)`, \(submodelName(report)) submodel "
    out += "(\(report.model.channels) channels, max_value \(format(report.model.submodelMaxValue, 2)))  \n"
    out += "            `\(report.audio.fileName)` — \(report.audio.frameCount) frames, "
    out += "\(format(report.audio.sampleRate, 0)) Hz, \(format(report.audio.durationSeconds, 2)) s  \n"
    out += "            \(format(report.audio.sampleRate / 1000, 0)) kHz, buffer \(report.config.blockSize), Release  \n"
    out += "Tool:       NAMBench\n\n"

    if report.environment.isSimulator {
      out += "> **Simulator run — timings are meaningless.** Numbers below came from the "
      out += "iOS Simulator, which runs host code under an entirely different scheduling and "
      out += "thermal regime. Use a real device.\n\n"
    }

    // --- Results -------------------------------------------------------------

    let baselineName = report.results.first?.variant ?? "baseline"
    let speedupByVariant = Dictionary(
      report.speedups.map { ($0.variant, $0.factor) }, uniquingKeysWith: { first, _ in first }
    )

    out += "## Result\n\n"
    out += "| Variant | Code path | Engine | Mean / pass | Fastest | Median | Spread | RTF | vs `\(baselineName)` |\n"
    out += "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    for (index, result) in report.results.enumerated() {
      let against: String
      if index == 0 {
        against = "—"
      } else if let factor = speedupByVariant[result.variant] {
        against = format(factor, 3) + "×"
      } else {
        against = "—"
      }

      if result.succeeded {
        out += "| \(result.variant) | `\(result.codePath)` | \(result.engine.displayName) "
        out += "| **\(format(result.meanMs, 2)) ms** | \(format(result.minMs, 2)) ms "
        out += "| \(format(result.medianMs, 2)) ms "
        out += "| \(format(result.spread * 100, 2))% | \(format(result.realTimeFactor, 1))× | \(against) |\n"
      } else {
        out += "| \(result.variant) | `\(result.codePath)` | \(result.engine.displayName) "
        out += "| _rejected_ | — | — | — | — | — |\n"
      }
    }
    out += "\n"
    out += "Fastest pass is shown because interference is additive: it can only make a "
    out += "pass slower, so the minimum is the least contaminated estimate of the real cost.\n\n"

    if let speedup = report.speedup, report.results.count == 2 {
      let baseline = report.results[0]
      let candidate = report.results[1]
      out += String(
        format: "**%@ is %.3f× %@** (%.2f ms → %.2f ms, %.1f%% %@).\n\n",
        candidate.variant, speedup >= 1 ? speedup : 1 / speedup,
        speedup >= 1 ? "faster than \(baseline.variant)" : "slower than \(baseline.variant)",
        baseline.meanMs, candidate.meanMs,
        abs(1 - candidate.meanMs / baseline.meanMs) * 100,
        speedup >= 1 ? "less time" : "more time"
      )
    }

    for result in report.results where !result.succeeded {
      out += "> **\(result.variant) produced no usable result:** \(result.failureReason ?? "unknown").\n\n"
    }

    // --- Parity --------------------------------------------------------------

    if !report.parities.isEmpty {
      let references = report.parities.map(\.referenceVariant).reduce(into: [String]()) {
        if !$0.contains($1) { $0.append($1) }
      }
      out += "## Output parity\n\n"
      out += "Over the full file (\(report.audio.frameCount) frames). "
      out += references.count > 1
        ? "Two references, because a candidate can be derived from either shipping engine — "
          + "\"bit-identical\" only means something once it says to what:\n\n"
        : "Every variant against `\(references[0])`:\n\n"
      out += "| Variant | Reference | max \\|difference\\| | RMS difference | Below signal | |\n"
      out += "|---|---|---:|---:|---:|---|\n"
      for parity in report.parities {
        out += "| \(parity.comparisonVariant) | `\(parity.referenceVariant)` "
        out += "| \(scientific(parity.maxAbsoluteDifference)) "
        out += "| \(scientific(parity.rmsDifference)) "
        out += "| \(parity.decibelsBelowSignal.isInfinite ? "**exact**" : format(parity.decibelsBelowSignal, 1) + " dB") "
        out += "| \(parity.withinTolerance ? "ok" : "**out of tolerance**") |\n"
      }
      out += "\nReference RMS \(scientific(report.parities[0].referenceRMS)). "
      out += report.parities.allSatisfy(\.withinTolerance)
        ? "Every variant is numerically equivalent — no speed here was paid for in accuracy.\n\n"
        : "> **At least one variant's output differs more than expected.** A faster engine that "
          + "computes something else is not faster; treat those rows as failures, not wins.\n\n"
    }

    // --- Protocol ------------------------------------------------------------

    out += "## How these numbers were taken\n\n"
    out += "- No idle gate: macOS is never idle (Spotlight and other background work "
    out += "cannot be paused, only disabled at the cost of a re-index), so interference "
    out += "is rejected in the analysis instead of waited out.\n"
    out += "  Conditions at start — \(report.systemAtStart.summary); "
    out += "at end — \(report.systemAtEnd.summary).\n"
    out += "- Warm-up: \(format(report.config.warmupSeconds, 0)) s of full-file passes, discarded.\n"
    out += "- Timing: \(format(report.config.timingWindowSeconds, 0)) s of further passes "
    out += "(at least \(report.config.minSamples) samples).\n"
    out += "- Accepted only when the fastest "
    out += "\(format(report.config.acceptFraction * 100, 0))% of samples agreed to within "
    out += "\(format(report.config.acceptTolerance * 100, 1))%; samples outside that window "
    out += "were discarded and the mean of the rest reported.\n"
    out += "- Model state reset before every pass, outside the timed region. Denormals "
    out += "flushed identically in every variant.\n"
    out += "- Every variant built from a pinned checkout with identical flags against one "
    out += "shared Eigen tree, isolated as its own dynamic framework.\n\n"

    for result in report.results {
      out += "\(result.variant): \(result.attempts.count) attempt"
      out += result.attempts.count == 1 ? "" : "s"
      if let last = result.attempts.last {
        out += ", \(last.warmupPasses) warm-up passes"
        out += ", \(last.samplesMs.count) timed samples"
        out += ", \(result.discardedMs.count) discarded"
      }
      out += ".\n"
    }
    out += "\n"

    // --- Provenance ----------------------------------------------------------

    if let pins = report.pins {
      out += "## Code under test\n\n"
      out += "| Component | Commit |\n|---|---|\n"
      out += "| \(pins.upstream.url) | `\(String(pins.upstream.sha.prefix(12)))` |\n"
      out += "| \(pins.fused.url) | `\(String(pins.fused.sha.prefix(12)))` |\n"
      out += "| Eigen (shared by both builds) | `\(String(pins.eigen.sha.prefix(12)))` |\n\n"
      out += "Eigen is shared deliberately: `a2_fast` uses it for its GEMM while the fused "
      out += "engine is pure NEON, so differing Eigen versions would land straight in the "
      out += "measured difference.\n\n"
    }

    // --- Raw samples ---------------------------------------------------------

    out += "## Raw samples\n\n"
    for result in report.results {
      out += "### \(result.variant)\n\n"
      guard let attempt = result.attempts.last else { out += "_none_\n\n"; continue }
      out += "Accepted (ms): "
      out += result.acceptedMs.isEmpty ? "_none_" : result.acceptedMs.map { format($0, 2) }.joined(separator: ", ")
      out += "\n\n"
      if !result.discardedMs.isEmpty {
        out += "Discarded (ms): \(result.discardedMs.map { format($0, 2) }.joined(separator: ", "))\n\n"
      }
      out += "In order taken (ms): \(attempt.samplesMs.map { format($0, 2) }.joined(separator: ", "))\n\n"
    }

    return out
  }

  // MARK: - Writing

  public struct WrittenFiles: Sendable {
    public let json: URL
    public let markdown: URL
  }

  /// Write both reports into `directory`, named by timestamp.
  @discardableResult
  public static func write(_ report: BenchmarkReport, to directory: URL) throws -> WrittenFiles {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let stamp = DateFormatter()
    stamp.dateFormat = "yyyyMMdd_HHmmss"
    stamp.locale = Locale(identifier: "en_US_POSIX")
    let base = "nambench_\(stamp.string(from: report.startedAt))"

    let jsonURL = directory.appendingPathComponent("\(base).json")
    let markdownURL = directory.appendingPathComponent("\(base).md")

    try json(for: report).write(to: jsonURL)
    try markdown(for: report).data(using: .utf8)?.write(to: markdownURL)

    return WrittenFiles(json: jsonURL, markdown: markdownURL)
  }

  /// Where reports go when the caller has no opinion. On iOS this is the app's
  /// Documents directory, which is reachable over the Files app when the target
  /// declares `UISupportsDocumentBrowser`/`LSSupportsOpeningDocumentsInPlace`.
  public static func defaultOutputDirectory() -> URL {
    let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
      ?? URL(fileURLWithPath: NSTemporaryDirectory())
    return base.appendingPathComponent("NAMBench", isDirectory: true)
  }

  // MARK: - Formatting

  private static func variantHeadline(_ report: BenchmarkReport) -> String {
    // Two variants read as a head-to-head; a lab run of a dozen kernels would
    // produce an unreadable title, so it names the baseline and counts.
    if report.results.count <= 3 {
      return report.results.map { "\($0.variant) (\($0.codePath))" }.joined(separator: " vs ")
    }
    let baseline = report.results[0]
    return "\(baseline.variant) (\(baseline.codePath)) vs \(report.results.count - 1) candidates"
  }

  private static func workloadName(_ report: BenchmarkReport) -> String {
    "NAM A2 \(submodelName(report))"
  }

  /// Named by what the submodel actually is rather than by what was asked for,
  /// so a report cannot claim "full" while holding nano numbers.
  private static func submodelName(_ report: BenchmarkReport) -> String {
    switch report.model.channels {
    case 8: return "full"
    case 3: return "slimmed"
    default: return "\(report.model.channels)-channel"
    }
  }

  private static func format(_ value: Double, _ places: Int) -> String {
    String(format: "%.\(places)f", value)
  }

  private static func scientific(_ value: Double) -> String {
    String(format: "%.3e", value)
  }
}
