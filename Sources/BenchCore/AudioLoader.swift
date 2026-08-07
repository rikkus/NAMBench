import Foundation

/// Audio held entirely in memory, ready to feed to a model.
public struct AudioBuffer: Sendable {
  public let samples: [Double]
  public let sampleRate: Double
  public let sourceChannelCount: Int

  public var frameCount: Int { samples.count }
  public var durationSeconds: Double { sampleRate > 0 ? Double(samples.count) / sampleRate : 0 }
}

public enum AudioLoaderError: LocalizedError {
  case notRIFF
  case missingChunk(String)
  case unsupportedFormat(code: Int, bits: Int)
  case emptyData

  public var errorDescription: String? {
    switch self {
    case .notRIFF: return "not a RIFF/WAVE file"
    case let .missingChunk(name): return "WAV file has no \(name) chunk"
    case let .unsupportedFormat(code, bits): return "unsupported WAV format (code \(code), \(bits)-bit)"
    case .emptyData: return "WAV file contains no samples"
    }
  }
}

/// Minimal WAV reader.
///
/// Deliberately hand-rolled rather than going through AVFoundation: the
/// benchmark input must land in memory as an exact, unresampled copy of what is
/// on disk, with no format negotiation in the way. Handles PCM 16/24/32-bit and
/// IEEE float 32/64-bit.
public enum AudioLoader {
  private static let formatPCM = 1
  private static let formatIEEEFloat = 3
  private static let formatExtensible = 0xFFFE

  public static func loadWAV(at url: URL) throws -> AudioBuffer {
    try loadWAV(data: try Data(contentsOf: url, options: .mappedIfSafe))
  }

  public static func loadWAV(data: Data) throws -> AudioBuffer {
    let bytes = [UInt8](data)
    guard bytes.count >= 12,
          bytes[0...3].elementsEqual("RIFF".utf8),
          bytes[8...11].elementsEqual("WAVE".utf8)
    else { throw AudioLoaderError.notRIFF }

    var formatCode = 0
    var channels = 0
    var sampleRate = 0.0
    var bitsPerSample = 0
    var dataRange: Range<Int>?

    var cursor = 12
    while cursor + 8 <= bytes.count {
      let id = String(decoding: bytes[cursor..<(cursor + 4)], as: UTF8.self)
      let size = Int(readUInt32(bytes, cursor + 4))
      let body = cursor + 8
      guard body <= bytes.count else { break }
      let end = min(body + size, bytes.count)

      switch id {
      case "fmt ":
        guard end - body >= 16 else { break }
        formatCode = Int(readUInt16(bytes, body))
        channels = Int(readUInt16(bytes, body + 2))
        sampleRate = Double(readUInt32(bytes, body + 4))
        bitsPerSample = Int(readUInt16(bytes, body + 14))
        // WAVE_FORMAT_EXTENSIBLE carries the real format code in its GUID.
        if formatCode == formatExtensible, end - body >= 26 {
          formatCode = Int(readUInt16(bytes, body + 24))
        }
      case "data":
        dataRange = body..<end
      default:
        break
      }

      cursor = body + size + (size & 1) // chunks are word-aligned
    }

    guard channels > 0, bitsPerSample > 0 else { throw AudioLoaderError.missingChunk("fmt ") }
    guard let dataRange else { throw AudioLoaderError.missingChunk("data") }

    let bytesPerSample = bitsPerSample / 8
    let frameStride = bytesPerSample * channels
    guard frameStride > 0 else { throw AudioLoaderError.unsupportedFormat(code: formatCode, bits: bitsPerSample) }

    let frameCount = dataRange.count / frameStride
    guard frameCount > 0 else { throw AudioLoaderError.emptyData }

    var samples = [Double](repeating: 0, count: frameCount)

    // The model is mono in, mono out; take channel 0 rather than downmixing so
    // the benchmark feeds exactly the signal that is in the file.
    for frame in 0..<frameCount {
      let offset = dataRange.lowerBound + frame * frameStride
      samples[frame] = try decodeSample(bytes, offset: offset, formatCode: formatCode, bits: bitsPerSample)
    }

    return AudioBuffer(samples: samples, sampleRate: sampleRate, sourceChannelCount: channels)
  }

  private static func decodeSample(_ bytes: [UInt8], offset: Int, formatCode: Int, bits: Int) throws -> Double {
    switch (formatCode, bits) {
    case (formatPCM, 16):
      return Double(Int16(bitPattern: readUInt16(bytes, offset))) / 32768.0

    case (formatPCM, 24):
      let raw = UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8) | (UInt32(bytes[offset + 2]) << 16)
      // Sign-extend 24 bits into 32.
      let signed = Int32(bitPattern: (raw & 0x0080_0000) != 0 ? raw | 0xFF00_0000 : raw)
      return Double(signed) / 8_388_608.0

    case (formatPCM, 32):
      return Double(Int32(bitPattern: readUInt32(bytes, offset))) / 2_147_483_648.0

    case (formatIEEEFloat, 32):
      return Double(Float(bitPattern: readUInt32(bytes, offset)))

    case (formatIEEEFloat, 64):
      return Double(bitPattern: readUInt64(bytes, offset))

    default:
      throw AudioLoaderError.unsupportedFormat(code: formatCode, bits: bits)
    }
  }

  private static func readUInt16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
    UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
  }

  private static func readUInt32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
    UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8)
      | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
  }

  private static func readUInt64(_ bytes: [UInt8], _ offset: Int) -> UInt64 {
    UInt64(readUInt32(bytes, offset)) | (UInt64(readUInt32(bytes, offset + 4)) << 32)
  }
}
