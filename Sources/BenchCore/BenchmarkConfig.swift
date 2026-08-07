import Foundation

/// Which submodel of a `SlimmableContainer` capture to measure.
///
/// A2 captures hold a nano submodel (3 channels, `max_value` 0.5) and a full
/// one (8 channels, `max_value` 1.0). Selection is by `max_value` rather than
/// position, so a reordering in the trainer cannot silently change which one is
/// measured.
public enum SubmodelSelection: Sendable, Codable, Equatable {
  /// Greatest `max_value` — A2 full. The default.
  case widest
  /// Least `max_value` — A2 nano, the slimmed path.
  case narrowest
  /// A literal index into the container's `submodels` array.
  case index(Int)

  public var displayName: String {
    switch self {
    case .widest: return "widest"
    case .narrowest: return "narrowest"
    case let .index(value): return "index \(value)"
    }
  }
}

/// Every tunable in the benchmark protocol, in one place.
///
/// Codable so the exact settings a result was produced under travel with it in
/// the report — a timing is only comparable against another taken the same way.
public struct BenchmarkConfig: Sendable, Codable {
  // MARK: Workload

  /// Audio block size fed to `process()`. 64 matches the methodology used by
  /// the fork's existing `benchmark_reports`, so results stay comparable.
  public var blockSize: Int = 64

  /// Which submodel to build. The slimmed-path work measures `.narrowest`;
  /// everything else has always measured `.widest`.
  public var submodel: SubmodelSelection = .widest

  /// Which slim-lab kernels to include, by index. Empty means none — the run
  /// is then `upstream` (and `fused`, where the shape allows it) alone.
  public var slimKernels: [Int] = []

  /// Reset model state before every pass so each one starts identically.
  /// Always done outside the timed region.
  public var resetBetweenPasses: Bool = true

  // MARK: Warm-up

  /// Run full-file passes for this long before timing anything, discarding the
  /// results. A fixed duration rather than a convergence test: on a machine
  /// with unavoidable background work, "wait until consecutive passes agree"
  /// can fail to settle for reasons that have nothing to do with the code under
  /// test. Rejecting noise is the analysis stage's job instead.
  ///
  /// Measured passes only start once this has elapsed, so the in-flight pass is
  /// always allowed to finish.
  public var warmupSeconds: Double = 5.0

  // MARK: Timing

  /// How long to keep timing passes once warm.
  public var timingWindowSeconds: Double = 30.0

  /// Fraction of samples that must agree for a result to be accepted.
  ///
  /// This was 0.90 while the benchmark waited for an idle system. Without that
  /// wait, more than 10% of passes get hit by background work and the 90%
  /// window is routinely unreachable — measured on an idle-ish M2, five
  /// consecutive attempts spread 1.4%, 1.2%, 4.5%, 9.4% and 27% at 90%, but
  /// 0.38%, 0.46%, 0.47%, 1.0% and 13% at 70%.
  ///
  /// The underlying signal is not noisy; the tail is. Across those same five
  /// attempts the *fastest* pass varied by only 0.4% (412.5–414.1 ms), because
  /// interference is strictly additive — it can only ever make a pass slower.
  /// Keeping the fastest 70% therefore estimates the uncontended cost, and any
  /// residual bias applies equally to both variants, so the ratio between them
  /// is unaffected.
  public var acceptFraction: Double = 0.70

  /// How tightly that fraction must agree: `(max - min) / min`.
  public var acceptTolerance: Double = 0.03

  /// Below this many samples the "90% agree" test is not meaningful, so the
  /// timing window is extended rather than accepting a result built from a
  /// handful of numbers.
  public var minSamples: Int = 15

  /// Attempts (warm-up + timing) before giving up on a variant.
  public var maxAttempts: Int = 5

  // MARK: System state

  /// Sample system-wide CPU utilisation at the start and end of a run and put
  /// it in the report. Purely informational — nothing is gated on it — but it
  /// is what makes a rejected run diagnosable after the fact. Costs a short
  /// sleep at each edge of the run, never during one.
  public var recordSystemCPU: Bool = true

  // MARK: Parity

  /// Compare the two variants' audio output before timing them.
  public var checkParity: Bool = true

  /// Warn above this RMS error relative to the reference signal's RMS.
  public var parityWarnRelativeRMS: Double = 1e-4

  public init() {}
}
