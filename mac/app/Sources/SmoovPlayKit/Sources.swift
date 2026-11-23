// The video sources, from the pipeline's own list (python -m framegen.sources --json), and what the core needs to
// play from each on a Mac. App window is the Mac app's own. The app captures the window with ScreenCaptureKit and
// sends the frames into the frame tap, which the core reads (--source tap). On Windows the core captures windows
// itself.
import Foundation

public struct SourceDef: Equatable, Identifiable, Sendable {
    public var id: String { kind }
    public var kind: String
    public var name: String
    public var guide: String
    public var hidden: Bool
    public var testMode: Bool
    public var experimental: Bool
    public var remotePlay: Bool { kind == "chiaki" }

    public static let remotePlay = SourceDef(kind: "chiaki", name: "PS5 Remote Play", guide: "", hidden: false,
                                             testMode: true, experimental: false)

    public static func parse(_ text: String) throws -> [SourceDef] {
        let root = try JSON.parse(text)
        return (root["sources"]?.array ?? []).map { d in
            let kind = d["kind"]?.string ?? ""
            // App window mode sends only new pictures (variable rate), which has no test mode in the core
            return SourceDef(kind: kind, name: d["name"]?.string ?? kind, guide: d["guide"]?.string ?? "",
                             hidden: d["hidden"]?.bool ?? false, testMode: d["test_mode"]?.bool ?? true,
                             experimental: d["experimental"]?.bool ?? false)
        }
    }

    /// The Mac's own wording, where the core's list says what Windows does (Ctrl+Alt, the Xbox app and so on).
    public func forMac() -> SourceDef {
        var s = self
        if kind == "window" {
            s.guide = "Open the app you stream with (PS Remote Play, Moonlight, or a browser for cloud gaming) and start "
                + "the game, pick its window, then press Play. SmoovPlay copies the window with macOS's screen capture, "
                + "and the app keeps the controller and the keyboard. SmoovPlay's own keys use Control-Option "
                + "(Control-Option-M menu, Control-Option-F fullscreen, Control-Option-Esc quit)."
        }
        if kind == "capture" {
            s.guide = "Connect the console's HDMI to the capture card and the card to this Mac. On a PS5, turn off HDCP while "
                + "you use it (Settings > System > HDMI). Set the console to 1080p, 60 Hz, HDR off. On a Mac the card's "
                + "sound isn't played yet, so listen on the card's HDMI output."
        }
        return s
    }
}

public enum Sources {
    /// The tap the app's window capture writes and the core reads (POSIX /smoovplay-window).
    public static let windowTap = "smoovplay-window"

    /// The sources offered. That's the pipeline's list minus hidden ones (SMOOVPLAY_SOURCES=all keeps them), with
    /// Remote Play put first if the list didn't come.
    public static func offered(_ all: [SourceDef], showHidden: Bool) -> [SourceDef] {
        var list = all.filter { showHidden || !$0.hidden }.map { $0.forMac() }
        if !list.contains(where: \.remotePlay) { list.insert(.remotePlay, at: 0) }
        return list
    }

    /// The core's arguments for a source. device is the capture card's number, and file is the Video file source's
    /// file.
    public static func runArgs(_ kind: String, device: String? = nil, file: String? = nil, tap: String? = nil) -> [String] {
        switch kind {
        case "capture": return ["--source", "capture", "--capture", device ?? "0"]
        case "window": return ["--source", "tap", "--tap", windowTap]
        case "tap": return ["--source", "tap"] + (tap.map { ["--tap", $0] } ?? [])
        case "file": return ["--source", "file", file ?? "", "--raw", "--realtime"]
        default: return ["--source", "chiaki"]
        }
    }
}

/// Capture cards, from python -m framegen.capture --list --json. On a Mac the core opens a card by number
/// (OpenCV's AVFoundation input), the "alt" in the list.
public struct CaptureDevice: Equatable, Identifiable, Sendable {
    public var id: String { index }
    public var name: String
    public var index: String

    public static func parse(_ text: String) throws -> [CaptureDevice] {
        let root = try JSON.parse(text)
        return (root["video"]?.array ?? []).enumerated().map { i, d in
            CaptureDevice(name: d["name"]?.string ?? "Device \(i)", index: d["alt"]?.string ?? String(i))
        }
    }
}
