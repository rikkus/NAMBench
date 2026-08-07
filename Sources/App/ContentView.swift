import BenchCore
import SwiftUI

struct ContentView: View {
  @StateObject private var model = BenchmarkViewModel()

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        header
        statusCard
        ForEach(model.variants) { VariantCard(state: $0, comparison: comparisonMean) }
        if let report = model.report { ResultsCard(report: report, files: model.writtenFiles) }
        if let parity = model.parity { ParityCard(parity: parity) }
        if !model.log.isEmpty { logCard }
      }
      .padding(20)
      .frame(maxWidth: 720, alignment: .leading)
      .frame(maxWidth: .infinity)
    }
    #if os(macOS)
      .frame(minWidth: 560, minHeight: 640)
    #endif
  }

  /// Slowest successful mean, so the bars share a scale.
  private var comparisonMean: Double? {
    let means = model.variants.compactMap { $0.result?.succeeded == true ? $0.result?.meanMs : nil }
    return means.max()
  }

  private var header: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text("NAM A2 Benchmark").font(.largeTitle.bold())
      Text("A2 full WaveNet — upstream `a2_fast` vs fork `fused`")
        .font(.subheadline).foregroundStyle(.secondary)
      HStack(spacing: 6) {
        Image(systemName: "waveform")
        Text(model.modelName).lineLimit(1).truncationMode(.middle)
        Text("·")
        Text(model.audioName)
      }
      .font(.caption).foregroundStyle(.secondary)
    }
  }

  private var statusCard: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        phaseLabel
        Spacer()
        if model.isRunning {
          Button("Cancel", role: .cancel) { model.cancel() }
        } else {
          Button(model.report == nil ? "Run benchmark" : "Run again") { model.start() }
            .buttonStyle(.borderedProminent)
        }
      }

      if case let .failed(message) = model.phase {
        Label(message, systemImage: "exclamationmark.triangle.fill")
          .font(.callout)
          .foregroundStyle(.orange)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .cardBackground()
  }

  private var phaseLabel: some View {
    let text: String
    let symbol: String
    switch model.phase {
    case .ready: text = "Ready"; symbol = "play.circle"
    case .preparing: text = "Loading model and audio"; symbol = "arrow.down.circle"
    case .checkingParity: text = "Checking output parity"; symbol = "equal.circle"
    case .running: text = "Benchmarking"; symbol = "timer"
    case .finished: text = "Finished"; symbol = "checkmark.circle.fill"
    case .failed: text = "Failed"; symbol = "xmark.circle.fill"
    }
    return Label(text, systemImage: symbol).font(.headline)
  }

  private var logCard: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text("Log").font(.headline)
      ForEach(Array(model.log.suffix(12).enumerated()), id: \.offset) { _, line in
        Text(line)
          .font(.system(.caption, design: .monospaced))
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .cardBackground()
  }
}

// MARK: - Variant

struct VariantCard: View {
  let state: VariantState
  let comparison: Double?

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        VStack(alignment: .leading, spacing: 2) {
          Text(state.name).font(.headline)
          Text(state.repository).font(.caption2).foregroundStyle(.secondary)
        }
        Spacer()
        EngineBadge(engine: state.engine, channels: state.channels)
      }

      if let result = state.result, result.succeeded {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
          Text(String(format: "%.2f", result.meanMs))
            .font(.system(size: 34, weight: .semibold, design: .rounded))
            .monospacedDigit()
          Text("ms / pass").foregroundStyle(.secondary)
          Spacer()
          Text(String(format: "%.1f× real time", result.realTimeFactor))
            .font(.caption).foregroundStyle(.secondary)
        }

        if let comparison, comparison > 0 {
          ProgressView(value: result.meanMs, total: comparison)
            .tint(result.meanMs < comparison ? .green : .secondary)
        }

        HStack(spacing: 14) {
          detail("median", String(format: "%.2f ms", result.medianMs))
          detail("spread", String(format: "%.2f%%", result.spread * 100))
          detail("accepted", "\(result.acceptedMs.count)")
          detail("discarded", "\(result.discardedMs.count)")
        }
      } else if state.phase == .failed {
        Label(state.result?.failureReason ?? "rejected", systemImage: "exclamationmark.triangle")
          .font(.callout).foregroundStyle(.orange)
          .fixedSize(horizontal: false, vertical: true)
      } else {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
          Text(state.lastMilliseconds > 0 ? String(format: "%.2f", state.lastMilliseconds) : "—")
            .font(.system(size: 30, weight: .semibold, design: .rounded))
            .monospacedDigit()
            .foregroundStyle(state.phase == .pending ? .tertiary : .primary)
          Text("ms").foregroundStyle(.secondary)
          Spacer()
          Text(phaseText).font(.caption).foregroundStyle(.secondary)
        }
        Sparkline(values: state.recent)
          .frame(height: 34)
      }
    }
    .cardBackground()
  }

  private var phaseText: String {
    switch state.phase {
    case .pending: return "waiting"
    case .warmingUp:
      if let remaining = state.secondsRemaining {
        return String(format: "warming up · pass %d · %.0f s left", state.passCount, remaining)
      }
      return "warming up · pass \(state.passCount)"
    case .timing:
      if let remaining = state.secondsRemaining {
        return String(format: "timing · sample %d · %.0f s left", state.passCount, remaining)
      }
      return "timing · sample \(state.passCount)"
    case .done: return "done"
    case .failed: return "rejected"
    }
  }

  private func detail(_ title: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 1) {
      Text(title).font(.caption2).foregroundStyle(.secondary)
      Text(value).font(.system(.caption, design: .monospaced))
    }
  }
}

struct EngineBadge: View {
  let engine: Engine
  let channels: Int

  var body: some View {
    VStack(alignment: .trailing, spacing: 2) {
      Text(engine.displayName)
        .font(.caption.weight(.medium))
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(color.opacity(0.15), in: Capsule())
        .foregroundStyle(color)
      if channels > 0 {
        Text("\(channels) channels").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }

  private var color: Color {
    switch engine {
    case .fused: return .green
    case .slim: return .purple
    case .a2Fast: return .blue
    case .generic: return .orange
    case .unknown: return .secondary
    }
  }
}

/// Live pass times. The shape is the point — you can see warm-up settling.
struct Sparkline: View {
  let values: [Double]

  var body: some View {
    GeometryReader { geometry in
      if values.count > 1, let low = values.min(), let high = values.max() {
        let range = Swift.max(high - low, high * 0.001, 0.0001)
        Path { path in
          for (index, value) in values.enumerated() {
            let x = geometry.size.width * Double(index) / Double(values.count - 1)
            let y = geometry.size.height * (1 - (value - low) / range)
            index == 0 ? path.move(to: CGPoint(x: x, y: y)) : path.addLine(to: CGPoint(x: x, y: y))
          }
        }
        .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 1.5, lineJoin: .round))
      }
    }
  }
}

// MARK: - Results

struct ResultsCard: View {
  let report: BenchmarkReport
  let files: Reporting.WrittenFiles?

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Result").font(.headline)

      if let speedup = report.speedup, report.results.count == 2 {
        let faster = speedup >= 1
        HStack(alignment: .firstTextBaseline, spacing: 6) {
          Text(String(format: "%.3f×", faster ? speedup : 1 / speedup))
            .font(.system(size: 40, weight: .bold, design: .rounded))
            .foregroundStyle(faster ? .green : .orange)
          Text("\(report.results[1].variant) is \(faster ? "faster" : "slower") than \(report.results[0].variant)")
            .foregroundStyle(.secondary)
        }
      }

      Divider()

      VStack(alignment: .leading, spacing: 3) {
        Text(report.environment.cpu).font(.caption)
        Text("\(report.environment.platform) \(report.environment.osVersion)")
          .font(.caption2).foregroundStyle(.secondary)
        if report.environment.isSimulator {
          Label("Simulator — timings are not meaningful", systemImage: "exclamationmark.triangle.fill")
            .font(.caption).foregroundStyle(.orange)
        }
      }

      if let pins = report.pins {
        VStack(alignment: .leading, spacing: 2) {
          Text("upstream \(pins.upstream.sha.prefix(12))")
          Text("fused \(pins.fused.sha.prefix(12))")
          Text("eigen \(pins.eigen.sha.prefix(12)) (shared)")
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundStyle(.secondary)
      }

      if let files {
        Text("Reports written to \(files.markdown.deletingLastPathComponent().path)")
          .font(.caption2).foregroundStyle(.secondary)
          .textSelection(.enabled)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .cardBackground()
  }
}

struct ParityCard: View {
  let parity: ParityResult

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text("Output parity").font(.headline)
        Spacer()
        Image(systemName: parity.withinTolerance ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
          .foregroundStyle(parity.withinTolerance ? .green : .orange)
      }
      Text(String(format: "%.1f dB below signal · max |diff| %.2e",
                  parity.decibelsBelowSignal, parity.maxAbsoluteDifference))
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.secondary)
      Text(parity.withinTolerance
        ? "Both engines compute the same audio, so the timing difference is a real speed difference."
        : "The engines disagree more than expected — check correctness before trusting the timings.")
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .cardBackground()
  }
}

// MARK: - Styling

extension View {
  func cardBackground() -> some View {
    padding(14)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
  }
}
