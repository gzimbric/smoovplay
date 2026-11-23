// App window mode on a Mac: ScreenCaptureKit copies the chosen app's window (PS Remote Play, Moonlight, a browser
// playing a cloud game...) at up to 60 fps as BGRA, and each picture goes into the frame tap as a decoded frame,
// which the core reads (run.py --source tap). The window is copied on its own, so windows over it (SmoovPlay's
// video among them) don't get in the way.
//
// Frames come when the window changes (FEATURE_VARIABLE): ScreenCaptureKit hands over a picture with status
// "complete" when something in the window changed, and only says "idle" when nothing did. So a picture is sent once,
// and the core puts the pictures on a 60 Hz beat, where a beat with no new picture is a repeat for certain. A
// complete picture with nothing marked changed in it is sent as a repeat (FLAG_REPEAT); otherwise the core compares
// pictures, as a streaming app's repeats are decoded again and never quite identical.
//
// The picture is the window's shape at the start, fitted into 1920x1080 (the size the pipeline is tuned for); if
// the window changes size later, the picture is squeezed into the same size (as Windows' window capture always
// does), so the core's video window can be stretched back over it.
//
// Screen Recording permission: macOS asks the first time (System Settings > Privacy & Security > Screen & System
// Audio Recording). Without it the window list is empty and capture fails.
import AppKit
import CFrameTap
import CoreMedia
import CoreVideo
import ScreenCaptureKit

public struct CaptureWindow: Identifiable, Hashable, Sendable {
    public var id: CGWindowID
    public var title: String
    public var app: String  // as people call it ("PS Remote Play")
    public var bundleID: String
    public var frame: CGRect  // points, from the main display's top left
    public var rank: Int  // 0 a known streaming app, 1 a browser showing a cloud game, 2 a browser, 3 anything else

    public var label: String {
        let t = title.isEmpty ? "" : ": " + (title.count > 60 ? String(title.prefix(57)) + "..." : title)
        return app + t + "  (\(Int(frame.width))x\(Int(frame.height)))"
    }

    /// Apps people stream games through, by bundle id (framegen/window.py's PRESETS, for macOS).
    public static let known: [String: String] = [
        "com.playstation.RemotePlay": "PS Remote Play",
        "com.moonlight-stream.Moonlight": "Moonlight",
        "com.valvesoftware.steamlink": "Steam Link",
        "com.nvidia.gfnpc.mall": "GeForce NOW",
        "tv.parsec.www": "Parsec",
        "com.microsoft.xboxapp": "Xbox",
        "com.apple.QuickTimePlayerX": "QuickTime Player",
    ]
    public static let browsers: [String: String] = [
        "com.apple.Safari": "Safari", "com.google.Chrome": "Chrome", "org.mozilla.firefox": "Firefox",
        "com.microsoft.edgemac": "Edge", "company.thebrowser.Browser": "Arc", "com.brave.Browser": "Brave",
    ]
    /// Words in a browser tab's title that mean a cloud game.
    static let cloudWords = ["xbox", "geforce now", "luna", "boosteroid", "cloud gaming", "stadia", "shadow"]

    public static func rank(bundleID: String, title: String) -> Int {
        if known[bundleID] != nil && bundleID != "com.apple.QuickTimePlayerX" { return 0 }
        if browsers[bundleID] != nil {
            let t = title.lowercased()
            return cloudWords.contains(where: t.contains) ? 1 : 2
        }
        return 3
    }

    /// Best first: known apps, cloud games in a browser, other browsers, the rest; then the biggest.
    public static func sorted(_ list: [CaptureWindow]) -> [CaptureWindow] {
        list.sorted { a, b in
            a.rank != b.rank ? a.rank < b.rank : a.frame.width * a.frame.height > b.frame.width * b.frame.height
        }
    }

    /// The picture's size for a window of this shape: fitted into 1920x1080, even, at least 320x180.
    public static func captureSize(for frame: CGRect, box: CGSize = CGSize(width: 1920, height: 1080)) -> (Int, Int) {
        guard frame.width > 0, frame.height > 0 else { return (Int(box.width), Int(box.height)) }
        let s = min(box.width / frame.width, box.height / frame.height)
        func even(_ v: CGFloat, _ lo: Int) -> Int { max(lo, Int((v / 2).rounded()) * 2) }
        return (even(frame.width * s, 320), even(frame.height * s, 180))
    }
}

public enum ScreenPermission {
    /// Granted already (no prompt).
    public static var granted: Bool { CGPreflightScreenCaptureAccess() }

    /// Asks macOS: the first time it shows its prompt; after a "Don't Allow" it shows nothing, and only System
    /// Settings can change it. Returns whether it's granted now.
    @discardableResult
    public static func request() -> Bool { CGRequestScreenCaptureAccess() }

    public static func openSettings() {
        if let u = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
            NSWorkspace.shared.open(u)
        }
    }

    public static let deniedText = "SmoovPlay needs Screen Recording to copy the app's window. In System Settings > Privacy & "
        + "Security > Screen & System Audio Recording, turn on SmoovPlay, then quit and open SmoovPlay again (macOS "
        + "applies it on the next start)."
}

/// The window's place now (points, top-left origin), and whether it's on screen (not minimised or hidden). No
/// permission needed: CGWindowList gives every window's bounds.
public func windowBounds(_ id: CGWindowID) -> (CGRect, Bool)? {
    guard let list = CGWindowListCopyWindowInfo([.optionIncludingWindow], id) as? [[String: Any]],
          let d = list.first, let b = d[kCGWindowBounds as String] as? [String: Any],
          let r = CGRect(dictionaryRepresentation: b as CFDictionary) else { return nil }
    return (r, (d[kCGWindowIsOnscreen as String] as? Bool) ?? false)
}

public final class WindowCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    public struct Stats: Sendable {
        public var complete = 0, idle = 0, repeats = 0, other = 0, sent = 0
        public var started = Date()
    }

    public let tapName: String
    private var tap: FrameTapWriter?
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "smoovplay.capture", qos: .userInteractive)
    private let statsLock = NSLock()
    private var _stats = Stats()
    public private(set) var size = (0, 0)
    public private(set) var window: CaptureWindow?
    /// The title bar left out of the picture, in points (0: none).
    public private(set) var titleBar: CGFloat = 0

    /// A standard title bar's height: 32 points from macOS 26 on (28 for apps built for older versions, which lose 4
    /// points of picture rather than show a strip of title bar), 28 before. Another app's own title bar can't be
    /// measured without the Accessibility permission.
    public static var titleBarHeight: CGFloat {
        ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26 ? 32 : 28
    }
    public var onStop: (String?) -> Void = { _ in }
    public var log: (String) -> Void = { _ in }

    public init(tapName: String = Sources.windowTap) { self.tapName = tapName }

    /// At most this many pictures a second (SMOOVPLAY_CAPTURE_HZ, for testing).
    static let maxRate = Int(ProcessInfo.processInfo.environment["SMOOVPLAY_CAPTURE_HZ"] ?? "") ?? 60

    public var stats: Stats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return _stats
    }

    /// The windows that could be captured, best first (needs Screen Recording; throws without it).
    public static func windows() async throws -> [CaptureWindow] {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let me = Bundle.main.bundleIdentifier ?? ""
        var out: [CaptureWindow] = []
        for w in content.windows {
            guard w.windowLayer == 0, w.frame.width >= 320, w.frame.height >= 180, let app = w.owningApplication else { continue }
            let bid = app.bundleIdentifier
            let appName = app.applicationName.lowercased()
            // (the core's own video window is python's, or the Rust core's smoovplay)
            if bid == me || bid == "com.apple.dock" || bid == "com.apple.WindowManager" || appName == "smoovplay"
                || appName.hasPrefix("python") { continue }
            let title = w.title ?? ""
            let name = CaptureWindow.known[bid] ?? CaptureWindow.browsers[bid] ?? app.applicationName
            out.append(CaptureWindow(id: w.windowID, title: title, app: name, bundleID: bid, frame: w.frame,
                                     rank: CaptureWindow.rank(bundleID: bid, title: title)))
        }
        return CaptureWindow.sorted(out)
    }

    /// Starts copying the window into the tap. Throws if the window is gone or capture can't start.
    /// leaveOutTitleBar: the top of the window, a standard title bar's height, isn't copied (not for a window that
    /// fills a screen, which has none). Apps that draw under their title bar (a browser's tabs, QuickTime) want it off.
    public func start(_ target: CaptureWindow, leaveOutTitleBar: Bool = true) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let w = content.windows.first(where: { $0.windowID == target.id }) else {
            throw FrameTapWriter.TapError(description: "the \(target.app) window isn't there any more. Press Look again")
        }
        let fullScreen = NSScreen.screens.contains { $0.frame.size == w.frame.size }
        let t = leaveOutTitleBar && !fullScreen ? WindowCapture.titleBarHeight : 0
        let source = t > 0 && t < w.frame.height / 3
            ? CGRect(x: 0, y: t, width: w.frame.width, height: w.frame.height - t)
            : CGRect(origin: .zero, size: w.frame.size)
        let (width, height) = CaptureWindow.captureSize(for: source)
        let tap = try FrameTapWriter(name: tapName)
        try tap.begin(codec: FrameTapWriter.codecBGRA, width: width, height: height, maxFPS: 60,
                      features: FrameTapWriter.featureVariable)
        let cfg = SCStreamConfiguration()
        cfg.width = width
        cfg.height = height
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(WindowCapture.maxRate))
        cfg.queueDepth = 5
        cfg.showsCursor = false
        cfg.capturesAudio = false
        cfg.scalesToFit = true
        cfg.preservesAspectRatio = false  // a later size change is squeezed into the same picture
        cfg.ignoreShadowsSingleWindow = true
        cfg.shouldBeOpaque = true
        cfg.colorSpaceName = CGColorSpace.sRGB
        cfg.sourceRect = source  // (points, from the window's top left)
        let filter = SCContentFilter(desktopIndependentWindow: w)
        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        statsLock.withLock { _stats = Stats() }
        self.tap = tap
        self.stream = s
        self.size = (width, height)
        self.window = target
        self.titleBar = source.minY
        do {
            try await s.startCapture()
        } catch {
            tap.end()
            tap.close()
            self.tap = nil
            self.stream = nil
            throw error
        }
        log("[capture] copying \(target.app)'s window (\(Int(w.frame.width))x\(Int(w.frame.height)) points\(source.minY > 0 ? ", without its \(Int(source.minY))-point title bar" : "")) as BGRA \(width)x\(height) into the frame tap \(tapName), when it changes")
    }

    public func stop() async {
        guard let s = stream else { return }
        stream = nil
        try? await s.stopCapture()
        finish()
    }

    private func finish() {
        queue.sync {
            tap?.end()
            tap?.close()
            tap = nil
        }
        let st = stats
        log("[capture] stopped: \(st.sent) pictures sent (\(st.complete) changed, \(st.repeats) unchanged), \(st.idle) idle, in \(Int(Date().timeIntervalSince(st.started))) s")
    }

    // MARK: SCStreamOutput
    public func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let tap else { return }
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = attachments.first, let raw = info[.status] as? Int, let status = SCFrameStatus(rawValue: raw) else { return }
        statsLock.lock()
        defer { statsLock.unlock() }
        switch status {
        case .complete:
            _stats.complete += 1
        case .idle:
            _stats.idle += 1  // nothing changed: nothing to send (the core's beat repeats the last picture)
            return
        default:
            _stats.other += 1
            return
        }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        let dirty = (info[.dirtyRects] as? [NSDictionary]) ?? []
        let unchanged = info[.dirtyRects] != nil && dirty.isEmpty
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb), CVPixelBufferGetWidth(pb) == tap.width,
              CVPixelBufferGetHeight(pb) == tap.height else { return }
        // When the picture was made: ScreenCaptureKit's times are the host clock (mach_absolute_time), which
        // CLOCK_UPTIME_RAW, the reader's clock, counts in nanoseconds. Now, if it doesn't look like one.
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let now = sp_now_ns()
        var at = now
        if pts.isValid, CMTimeGetSeconds(pts) > 0 {
            let t = UInt64(CMTimeGetSeconds(pts) * 1e9)
            if t <= now, now - t < 1_000_000_000 { at = t }
        }
        tap.push(pixels: base, bytesPerRow: CVPixelBufferGetBytesPerRow(pb), new: unchanged ? false : nil, at: at)
        _stats.sent += 1
        if unchanged { _stats.repeats += 1 }
    }

    // MARK: SCStreamDelegate
    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        self.stream = nil
        finish()
        let why = (error as NSError).code == SCStreamError.userDeclined.rawValue ? ScreenPermission.deniedText
            : "the window capture stopped: \(error.localizedDescription)"
        log("[capture] \(why)")
        DispatchQueue.main.async { self.onStop(why) }
    }
}
