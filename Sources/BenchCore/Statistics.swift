import Foundation

/// The subset of samples judged to agree, and what was thrown away.
public struct WindowSelection: Sendable {
  /// Samples inside the tightest interval holding the required fraction.
  public let accepted: [Double]
  /// Everything else — almost always slow outliers from interference.
  public let discarded: [Double]
  /// `(max - min) / min` across `accepted`.
  public let spread: Double
  public let mean: Double
  public let median: Double
  public let min: Double
  public let max: Double
}

public enum Statistics {
  /// Tightest interval containing at least `fraction` of the samples.
  ///
  /// Sorting then sliding a fixed-width window finds the densest cluster
  /// directly. This is deliberately not a symmetric trim: interference only
  /// ever makes a pass *slower*, so contamination is one-sided and trimming
  /// equal tails would discard good fast samples to balance bad slow ones.
  public static func tightestWindow(_ samples: [Double], fraction: Double) -> WindowSelection? {
    guard !samples.isEmpty, fraction > 0, fraction <= 1 else { return nil }

    let sorted = samples.sorted()
    let width = Swift.max(1, Int((fraction * Double(sorted.count)).rounded(.up)))
    guard width <= sorted.count else { return nil }

    var bestStart = 0
    var bestSpread = Double.infinity
    for start in 0...(sorted.count - width) {
      let low = sorted[start]
      let high = sorted[start + width - 1]
      guard low > 0 else { continue }
      let spread = (high - low) / low
      if spread < bestSpread {
        bestSpread = spread
        bestStart = start
      }
    }
    guard bestSpread.isFinite else { return nil }

    let accepted = Array(sorted[bestStart..<(bestStart + width)])
    var discarded = Array(sorted[0..<bestStart])
    discarded.append(contentsOf: sorted[(bestStart + width)...])

    return WindowSelection(
      accepted: accepted,
      discarded: discarded,
      spread: bestSpread,
      mean: mean(accepted),
      median: median(accepted),
      min: accepted.first ?? 0,
      max: accepted.last ?? 0
    )
  }

  public static func mean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    return values.reduce(0, +) / Double(values.count)
  }

  public static func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
  }

  /// `(max - min) / min`, the spread measure used throughout the protocol.
  public static func relativeSpread(_ values: [Double]) -> Double {
    guard let low = values.min(), let high = values.max(), low > 0 else { return .infinity }
    return (high - low) / low
  }

  public static func standardDeviation(_ values: [Double]) -> Double {
    guard values.count > 1 else { return 0 }
    let m = mean(values)
    let variance = values.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(values.count - 1)
    return variance.squareRoot()
  }
}
