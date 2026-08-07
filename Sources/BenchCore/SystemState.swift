import Foundation

/// What the machine was doing around a run.
///
/// Recorded, not gated on. An earlier version of this benchmark waited for the
/// system to go idle before measuring; on macOS that wait never ends, because
/// Spotlight and friends are always doing something and the only way to stop
/// them is to disable them outright and pay for a full re-index afterwards. So
/// the noise defence moved entirely into the analysis — see the tightest-90%
/// window in `Statistics` — and this type exists only to record the conditions
/// alongside the numbers, which is what makes a rejected run diagnosable.
public struct SystemState: Sendable, Codable {
  public let thermalState: String
  public let lowPowerMode: Bool
  /// System-wide CPU utilisation, 0...1. macOS only — nil on iOS, where
  /// `host_statistics` is not reliably available under the sandbox.
  public let systemCPUUtilisation: Double?

  public var summary: String {
    var parts = ["thermal \(thermalState)"]
    if lowPowerMode { parts.append("Low Power Mode on") }
    if let cpu = systemCPUUtilisation {
      parts.append(String(format: "system CPU %.1f%%", cpu * 100))
    }
    return parts.joined(separator: ", ")
  }

  /// Sample the current state. `sampleCPU` costs a short sleep, so callers
  /// take it at the edges of a run rather than during one.
  public static func capture(sampleCPU: Bool = true, sampleSeconds: Double = 0.25) -> SystemState {
    let processInfo = ProcessInfo.processInfo
    return SystemState(
      thermalState: describe(processInfo.thermalState),
      lowPowerMode: processInfo.isLowPowerModeEnabled,
      systemCPUUtilisation: sampleCPU ? systemCPUUtilisation(sampleSeconds: sampleSeconds) : nil
    )
  }

  public static func describe(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }

  static func systemCPUUtilisation(sampleSeconds: Double) -> Double? {
    #if os(macOS)
      guard let first = readCPUTicks() else { return nil }
      Thread.sleep(forTimeInterval: sampleSeconds)
      guard let second = readCPUTicks() else { return nil }

      let busy = Double((second.user &- first.user) + (second.system &- first.system)
        + (second.nice &- first.nice))
      let idle = Double(second.idle &- first.idle)
      let total = busy + idle
      guard total > 0 else { return nil }
      return busy / total
    #else
      return nil
    #endif
  }

  #if os(macOS)
    private struct CPUTicks {
      var user: UInt32
      var system: UInt32
      var idle: UInt32
      var nice: UInt32
    }

    private static func readCPUTicks() -> CPUTicks? {
      var info = host_cpu_load_info_data_t()
      var count = mach_msg_type_number_t(
        MemoryLayout<host_cpu_load_info_data_t>.stride / MemoryLayout<integer_t>.stride
      )

      let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
          host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, reboundPointer, &count)
        }
      }
      guard result == KERN_SUCCESS else { return nil }

      return CPUTicks(
        user: info.cpu_ticks.0,
        system: info.cpu_ticks.1,
        idle: info.cpu_ticks.2,
        nice: info.cpu_ticks.3
      )
    }
  #endif
}
