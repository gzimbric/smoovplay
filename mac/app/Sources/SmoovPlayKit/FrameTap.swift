// A frame tap writer for decoded frames (docs/frame-tap.md, layout version 2). The app's window capture hands each
// picture to the core this way, as any app may. The tap is POSIX shared memory named "/" plus the tap's name,
// 4096 + 64 MiB, created by whichever side comes first and never removed. The reader polls write_seq, so there's
// no event.
//
// Header (little-endian): 0 magic 'CFTP', 4 version, 8 slot_count, 12 slot_size, 16 codec, 20 width, 24 height,
// 28 max_fps, 32 u64 write_seq, 40 idr_request, 44 session, 48 u64 qpc_freq, 56 u64 reader_heartbeat,
// 64 features, 88 audio_delay_ms, 92 colour. Slot i at 4096 + i * slot_size: 0 u64 seq, 8 u64 qpc_arrival,
// 16 size, 20 frames_lost, 24 recovered, 28 flags (1 truncated, 2 repeat, 4 new), 64 data.
// Times are nanoseconds of CLOCK_UPTIME_RAW, with qpc_freq 1e9.
import CFrameTap
import Foundation

public final class FrameTapWriter: @unchecked Sendable {
    public static let magic: UInt32 = 0x5054_4643  // 'CFTP'
    public static let headerSize = 4096
    public static let slotHeader = 64
    public static let maxSlots = 32
    public static let capacity = 32 * 2 * 1024 * 1024  // the slots' share, however a writer divides it
    public static let total = headerSize + capacity
    public static let codecNV12: UInt32 = 16, codecBGRA: UInt32 = 17
    public static let versionRaw: UInt32 = 2
    public static let featureVariable: UInt32 = 8  // a frame only when the picture changes
    public static let flagTruncated: UInt32 = 1, flagRepeat: UInt32 = 2, flagNew: UInt32 = 4
    public static let shmNameMax = 31  // macOS's PSHMNAMLEN, with the "/"

    public struct TapError: Error, CustomStringConvertible { public let description: String }

    /// "/" plus what follows the name's last backslash ("Local\smoovplay-tap" becomes "/smoovplay-tap"). It's
    /// checked to leave room for the sound ring's "-audio", as framegen/chiaki_tap.py check_name does.
    public static func posixName(_ name: String) throws -> String {
        let base = String(name.split(separator: "\\", omittingEmptySubsequences: false).last ?? "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !base.isEmpty, !base.contains("/") else {
            throw TapError(description: "the frame tap's name \(name) isn't usable as shared memory (try a plain name)")
        }
        let full = "/" + base
        guard full.utf8.count + "-audio".utf8.count <= shmNameMax else {
            throw TapError(description: "the frame tap's name \(name) is too long: at most \(shmNameMax - 1 - 6) characters after any Local\\")
        }
        return full
    }

    /// (slot_count, slot_size) for frames of this many bytes (framegen.chiaki_tap.raw_layout). Each slot is a whole
    /// number of MiB and holds one frame and its slot header. There are as many as fit, at most 32 and at least 4.
    public static func rawLayout(frameBytes: Int) throws -> (count: Int, size: Int) {
        let need = slotHeader + frameBytes
        let mib = 1 << 20
        let size = (need + mib - 1) / mib * mib
        let count = min(maxSlots, capacity / size)
        guard count >= 4 else {
            throw TapError(description: "a frame of \(frameBytes) bytes is too big for the frame tap (4 have to fit in 64 MiB)")
        }
        return (count, size)
    }

    public let name: String
    public let path: String
    private var base: UnsafeMutableRawPointer?
    public private(set) var slots = 0, slotSize = 0, width = 0, height = 0, frameBytes = 0
    public private(set) var written: UInt64 = 0
    private let lock = NSLock()

    public init(name: String) throws {
        self.name = name
        path = try FrameTapWriter.posixName(name)
        var err: Int32 = 0
        guard let b = path.withCString({ sp_shm_map($0, FrameTapWriter.total, &err) }) else {
            throw TapError(description: err == -1
                ? "the shared memory \(path) is too small for the frame tap: something else made it"
                : "couldn't open the frame tap \(path): \(String(cString: strerror(err)))")
        }
        base = b
    }

    deinit { close() }

    private func u32(_ off: Int, _ v: UInt32) { base!.storeBytes(of: v.littleEndian, toByteOffset: off, as: UInt32.self) }
    private func u64(_ off: Int, _ v: UInt64) { base!.storeBytes(of: v.littleEndian, toByteOffset: off, as: UInt64.self) }
    private func r32(_ off: Int) -> UInt32 { UInt32(littleEndian: base!.load(fromByteOffset: off, as: UInt32.self)) }
    private func r64(_ off: Int) -> UInt64 { UInt64(littleEndian: base!.load(fromByteOffset: off, as: UInt64.self)) }

    /// Starts a stream of BGRA frames (or NV12) of this size.
    public func begin(codec: UInt32 = codecBGRA, width: Int, height: Int, maxFPS: Int = 60, features: UInt32 = 0,
                      colour: UInt32 = 0) throws {
        lock.lock()
        defer { lock.unlock() }
        guard base != nil else { throw TapError(description: "the frame tap is closed") }
        let bytes = codec == FrameTapWriter.codecNV12 ? width * height * 3 / 2 : width * height * 4
        let (count, size) = try FrameTapWriter.rawLayout(frameBytes: bytes)
        slots = count; slotSize = size; self.width = width; self.height = height; frameBytes = bytes
        sp_store_u32(base! + 0, 0)  // magic 0 while the header is written
        sp_fence()
        u32(4, FrameTapWriter.versionRaw)
        u32(8, UInt32(count))
        u32(12, UInt32(size))
        u32(16, codec)
        u32(20, UInt32(width))
        u32(24, UInt32(height))
        u32(28, UInt32(maxFPS))
        u64(32, 0)  // write_seq
        u32(40, 0)
        u64(48, 1_000_000_000)
        u32(64, features)
        u32(92, codec == FrameTapWriter.codecNV12 ? colour : 0)
        u32(44, r32(44) &+ 1)  // session
        sp_fence()
        sp_store_u32(base! + 0, FrameTapWriter.magic)
        written = 0
    }

    /// Writes one frame, `height` rows of `width * 4` bytes (BGRA), each `bytesPerRow` apart in `pixels`. new is
    /// the writer's word on it: true for new content, false for a repeat of the frame before, nil if not known. at
    /// is when the frame was made, in CLOCK_UPTIME_RAW nanoseconds, and defaults to now.
    public func push(pixels: UnsafeRawPointer, bytesPerRow: Int, new: Bool?, at: UInt64? = nil) {
        lock.lock()
        defer { lock.unlock() }
        guard let b = base, slots > 0 else { return }
        let seq = sp_load_u64(b + 32)
        let off = FrameTapWriter.headerSize + Int(seq % UInt64(slots)) * slotSize
        let rowBytes = frameBytes / height
        let size = min(frameBytes, slotSize - FrameTapWriter.slotHeader)
        var flags: UInt32 = size < frameBytes ? FrameTapWriter.flagTruncated : 0
        if let n = new { flags |= n ? FrameTapWriter.flagNew : FrameTapWriter.flagRepeat }
        sp_store_u64(b + off, UInt64.max)  // being rewritten
        sp_fence()
        u64(off + 8, at ?? sp_now_ns())
        u32(off + 16, UInt32(size))
        u32(off + 20, 0)
        u32(off + 24, 0)
        u32(off + 28, flags)
        let dst = b + off + FrameTapWriter.slotHeader
        if bytesPerRow == rowBytes {
            memcpy(dst, pixels, size)
        } else {
            for y in 0..<height { memcpy(dst + y * rowBytes, pixels + y * bytesPerRow, min(rowBytes, bytesPerRow)) }
        }
        sp_fence()
        sp_store_u64(b + off, seq)
        sp_fence()
        sp_store_u64(b + 32, seq &+ 1)
        written += 1
    }

    /// Ends the stream by setting magic to 0. The reader then says "the app stopped sending frames".
    public func end() {
        lock.lock()
        defer { lock.unlock() }
        guard let b = base else { return }
        sp_store_u32(b + 0, 0)
    }

    /// How long ago the reader last said it was reading (seconds), or nil if it never has.
    public var readerAge: Double? {
        lock.lock()
        defer { lock.unlock() }
        guard let b = base else { return nil }
        let hb = sp_load_u64(b + 56)
        guard hb > 0 else { return nil }
        return Double(Int64(bitPattern: sp_now_ns() &- hb)) / 1e9
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        if let b = base { sp_shm_unmap(b, FrameTapWriter.total) }
        base = nil
    }

    /// Removes the shared memory, for tests. Otherwise the tap is left for the next reader or writer.
    public static func unlink(_ name: String) {
        guard let p = try? posixName(name) else { return }
        _ = p.withCString { sp_shm_unlink($0) }
        _ = (p + "-audio").withCString { sp_shm_unlink($0) }
    }
}
