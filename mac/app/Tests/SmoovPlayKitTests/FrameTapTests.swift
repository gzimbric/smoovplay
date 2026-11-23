import CFrameTap
import Foundation
import Testing
@testable import SmoovPlayKit

/// The Swift frame tap writer, read back by the pipeline's own reader (framegen/chiaki_tap.py FrameTap) and by the
/// core itself (run.py --source tap, headless).
@Suite(.serialized) struct FrameTapTests {
    @Test func layoutAsTheDocsSay() throws {
        #expect(try FrameTapWriter.rawLayout(frameBytes: 1920 * 1080 * 4) == (8, 8 << 20))  // docs/frame-tap.md
        #expect(try FrameTapWriter.rawLayout(frameBytes: 1920 * 1080 * 3 / 2) == (21, 3 << 20))
        #expect(try FrameTapWriter.rawLayout(frameBytes: 2560 * 1440 * 4) == (4, 15 << 20))
        #expect(throws: FrameTapWriter.TapError.self) { try FrameTapWriter.rawLayout(frameBytes: 3840 * 2160 * 4) }
        #expect(try FrameTapWriter.posixName("Local\\smoovplay-tap") == "/smoovplay-tap")
        #expect(try FrameTapWriter.posixName("smoovplay-window") == "/smoovplay-window")
        #expect(throws: FrameTapWriter.TapError.self) { try FrameTapWriter.posixName("Local\\a-name-that-is-far-too-long") }
        #expect(throws: FrameTapWriter.TapError.self) { try FrameTapWriter.posixName("Local\\") }
    }

    @Test func captureSizes() {
        #expect(CaptureWindow.captureSize(for: CGRect(x: 0, y: 0, width: 1600, height: 900)) == (1920, 1080))
        #expect(CaptureWindow.captureSize(for: CGRect(x: 0, y: 0, width: 800, height: 600)) == (1440, 1080))
        #expect(CaptureWindow.captureSize(for: CGRect(x: 0, y: 0, width: 1728, height: 1117)) == (1670, 1080))
        #expect(CaptureWindow.captureSize(for: CGRect(x: 0, y: 0, width: 3000, height: 500)) == (1920, 320))
        let list = CaptureWindow.sorted([
            CaptureWindow(id: 1, title: "Notes", app: "Notes", bundleID: "com.apple.Notes", frame: CGRect(x: 0, y: 0, width: 2000, height: 1000), rank: 3),
            CaptureWindow(id: 2, title: "Xbox Cloud Gaming", app: "Safari", bundleID: "com.apple.Safari", frame: CGRect(x: 0, y: 0, width: 800, height: 450),
                          rank: CaptureWindow.rank(bundleID: "com.apple.Safari", title: "Xbox Cloud Gaming")),
            CaptureWindow(id: 3, title: "", app: "PS Remote Play", bundleID: "com.playstation.RemotePlay", frame: CGRect(x: 0, y: 0, width: 640, height: 360),
                          rank: CaptureWindow.rank(bundleID: "com.playstation.RemotePlay", title: "")),
        ])
        #expect(list.map(\.id) == [3, 2, 1])
    }

    /// A BGRA picture of `w`x`h`: pixel (x, y) of frame n is (B, G, R, A) = (n, y, x, 255) mod 256.
    static func picture(_ n: Int, _ w: Int, _ h: Int, stride: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0xEE, count: stride * h)
        for y in 0..<h {
            for x in 0..<w {
                let o = y * stride + x * 4
                b[o] = UInt8(n & 0xFF); b[o + 1] = UInt8(y & 0xFF); b[o + 2] = UInt8(x & 0xFF); b[o + 3] = 255
            }
        }
        return b
    }

    static let reader = """
    import json, sys, time
    import numpy as np
    from framegen.chiaki_tap import FrameTap, FLAG_NEW, FLAG_REPEAT
    tap = FrameTap(sys.argv[1])
    h = tap.header()
    out = {"live": h.live, "version": h.version, "codec": h.codec, "w": h.width, "h": h.height, "fps": h.max_fps,
           "write_seq": h.write_seq, "slots": h.slots, "slot_size": h.slot_size, "features": h.features,
           "freq": h.qpc_freq, "frames": []}
    for seq in range(max(0, h.write_seq - h.slots + 1), h.write_seq):
        buf = np.empty(h.width * h.height * 4, np.uint8)
        p = tap.read(seq, h.qpc_freq, h.slots, h.slot_size, into=buf)
        if p is None:
            out["frames"].append(None); continue
        img = buf.reshape(h.height, h.width, 4)
        n = int(img[0, 0, 0])
        ok = bool((img[:, :, 0] == n).all() and (img[:, :, 1] == (np.arange(h.height)[:, None] % 256)).all()
                  and (img[:, :, 2] == (np.arange(h.width)[None, :] % 256)).all() and (img[:, :, 3] == 255).all())
        out["frames"].append({"seq": p.seq, "size": p.size, "flags": p.flags, "n": n, "ok": ok,
                              "age": time.perf_counter() - p.t_arrival})
    print(json.dumps(out))
    """

    @Test(.enabled(if: Fixture.hasPython)) func pythonReadsTheFrames() throws {
        let name = "sp-test-\(getpid())"
        FrameTapWriter.unlink(name)
        defer { FrameTapWriter.unlink(name) }
        let w = 640, h = 360
        let tap = try FrameTapWriter(name: name)
        try tap.begin(width: w, height: h, maxFPS: 60, features: FrameTapWriter.featureVariable)
        #expect(tap.slots == 32 && tap.slotSize == 1 << 20)
        for n in 0..<5 {
            let stride = n % 2 == 0 ? w * 4 : w * 4 + 64  // rows padded, as a CVPixelBuffer's can be
            let pic = FrameTapTests.picture(n, w, h, stride: stride)
            pic.withUnsafeBytes { tap.push(pixels: $0.baseAddress!, bytesPerRow: stride, new: n == 3 ? false : n == 4 ? nil : true) }
        }
        let script = FileManager.default.temporaryDirectory.appendingPathComponent("sp-tap-reader-\(getpid()).py")
        try FrameTapTests.reader.write(to: script, atomically: true, encoding: .utf8)
        let (rc, out) = run(Fixture.python, [script.path, name], cwd: Fixture.repo,
                            env: ["PYTHONPATH": Fixture.repo.path, "PATH": "/usr/bin:/bin"])
        try? FileManager.default.removeItem(at: script)
        #expect(rc == 0, "\(out)")
        let j = try JSON.parse(String(out.split(separator: "\n").last(where: { $0.hasPrefix("{") }) ?? ""))
        #expect(j["live"] == true)
        #expect(j["version"] == 2 && j["codec"] == 17 && j["w"] == .int(w) && j["h"] == .int(h) && j["fps"] == 60)
        #expect(j["write_seq"] == 5 && j["slots"] == 32 && j["slot_size"] == .int(1 << 20))
        #expect(j["features"] == 8 && j["freq"] == 1_000_000_000)
        let frames = j["frames"]?.array ?? []
        #expect(frames.count == 5)
        for (n, f) in frames.enumerated() {
            #expect(f["seq"] == .int(n) && f["n"] == .int(n) && f["ok"] == true && f["size"] == .int(w * h * 4))
            #expect(f["flags"] == .int(n == 3 ? 2 : n == 4 ? 0 : 4))  // new, repeat, not known
            #expect((f["age"]?.double ?? 99) >= 0 && (f["age"]?.double ?? 99) < 30)  // the same clock as the reader's
        }
        tap.end()
        let (_, out2) = run(Fixture.python, ["-c", "import sys; from framegen.chiaki_tap import FrameTap; print(FrameTap(sys.argv[1]).header().live)", name],
                            cwd: Fixture.repo, env: ["PYTHONPATH": Fixture.repo.path])
        #expect(out2.contains("False"))
        tap.close()
    }

    /// The core plays what the writer sends (run.py --source tap, no window) until the writer ends the stream.
    @Test(.enabled(if: Fixture.hasPython)) func coreReadsTheTap() async throws {
        let name = "sp-core-\(getpid())"
        FrameTapWriter.unlink(name)
        defer { FrameTapWriter.unlink(name) }
        let paths = AppPaths.detect(env: ProcessInfo.processInfo.environment.merging(["SMOOVPLAY_APP": Fixture.repo.path]) { $1 })
        let q = DispatchQueue(label: "core")
        let p = LineProcess(executable: paths.python, arguments: [paths.runPy.path, "--source", "tap", "--tap", name, "--settings", "none"],
                            cwd: paths.repo, env: paths.env, queue: q)
        var lines: [String] = []
        var code: Int32?
        p.onLine = { lines.append($0) }
        p.onExit = { code = $0 }
        try p.start()
        let tap = try FrameTapWriter(name: name)
        let w = 640, h = 360
        try tap.begin(width: w, height: h, maxFPS: 60, features: FrameTapWriter.featureVariable)
        // a 30 fps game, with a new picture every 33 ms (variable rate, so repeats aren't sent), for up to 60 s, and
        // 3 s once the core is playing
        var pics: [[UInt8]] = (0..<8).map { FrameTapTests.picture($0 * 30, w, h, stride: w * 4) }
        for y in 0..<h { for x in 0..<w { for k in 0..<8 { pics[k][(y * w + x) * 4 + 1] = UInt8((x + y + k * 8) & 0xFF) } } }
        let start = Date()
        var playingAt: Date?
        var n = 0
        while Date().timeIntervalSince(start) < 60 {
            pics[n % 8].withUnsafeBytes { tap.push(pixels: $0.baseAddress!, bytesPerRow: w * 4, new: true) }
            n += 1
            try await Task.sleep(nanoseconds: 33_000_000)
            if playingAt == nil, q.sync(execute: { lines.contains(where: { $0.hasPrefix("[run] playing") }) }) { playingAt = Date() }
            if let t = playingAt, Date().timeIntervalSince(t) > 3 { break }
            if q.sync(execute: { code != nil }) { break }
        }
        tap.end()
        for _ in 0..<200 where q.sync(execute: { code == nil }) { try await Task.sleep(nanoseconds: 100_000_000) }
        let all = q.sync { lines.joined(separator: "\n") }
        #expect(q.sync { code } == 0, "\(all.suffix(2000))")
        #expect(all.contains("[tap] frames: BGRA (decoded) 640x360, frames when the picture changes"))
        #expect(all.contains("[run] playing"))
        #expect(all.contains("[tap] the app stopped sending frames"))
        let ticks = all.split(separator: "\n").first(where: { $0.hasPrefix("ticks=") }).map(String.init) ?? ""
        #expect(!ticks.isEmpty && !ticks.hasPrefix("ticks=0 "), "\(ticks)")
        tap.close()
        if p.running { p.kill() }
    }
}
