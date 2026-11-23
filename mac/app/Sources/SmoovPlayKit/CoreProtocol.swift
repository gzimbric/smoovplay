// What the core says on stdout, and what the app sends on stdin (framegen/remote.py, run.py --control).
//
// Commands, one per line:
//   set <key> <json value>    change a setting and save it, as if picked in the video window's menu
//   restart                   apply restart-only settings now
//   quit                      stop, as Esc in the video window does
//   state                     print the current state again
//   look-show <look JSON>     show this look (the Looks page is editing it)
//   look-hide                 back to the Look settings' look
//   looks-reload              the user's looks changed on disk
// and, not in the core yet (the app sends them, and stops when the core says "[remote] unknown command"):
//   key <name>                a video-window key, as if pressed there (M, F, Q, S, I, U, Z, K, J, L, R, B; Up,
//                             Down, Left, Right, Enter, Esc, Backspace for the menu): the instant replay keys and
//                             App window mode's shortcuts, which come from other apps on a Mac. "key" alone does
//                             nothing: the app sends it once the video is up, to learn whether the core takes keys
//   overlay <x> <y> <w> <h>   App window mode: sit over the captured window (points, from the main display's top
//                             left), borderless, on top and click-through; "overlay none" while it's minimised
//
// Lines the app reads (the Windows launcher's Session.Line):
//   [settings-state] {json}   values, choices, disabled, restart (SettingsState)
//   [chiaki] waiting for chiaki-ng ...   [tap] waiting for an app ...       -> Waiting (chiaki-ng is opened)
//   [chiaki] stream: ...      [tap] frames: ...                             -> Found
//   [trt] building ...        [run] restarting for <what>: ...              -> Building
//   [run] playing                                                           -> Playing (also after a restart)
//   [chiaki] stream ended     [tap] the app stopped sending frames          -> the stream ended (exit 0: Ended)
//   [chiaki|tap] this stream was already running ...  [chiaki|tap] WARNING: ...   -> a note on the Play page
//   [window-keys] {json}      the shortcuts the video window got
//   [remote] unknown command: ...   [remote] can't set <key>: <why>
import Foundation

public enum PlayState: Equatable, Sendable {
    case idle, starting, waiting, found(String?), building(String?), playing, ended, stopped, error(String)

    public var busy: Bool {
        switch self {
        case .starting, .waiting, .found, .building: return true
        default: return false
        }
    }

    /// Starting or playing: the session is up (Play turns into Stop).
    public var running: Bool { busy || self == .playing }

    public var over: Bool {
        switch self {
        case .ended, .stopped, .error: return true
        default: return false
        }
    }

    public var name: String {
        switch self {
        case .idle: return "idle"
        case .starting: return "starting"
        case .waiting: return "waiting"
        case .found: return "found"
        case .building: return "building"
        case .playing: return "playing"
        case .ended: return "ended"
        case .stopped: return "stopped"
        case .error: return "error"
        }
    }
}

public enum CoreEvent: Equatable, Sendable {
    case state(String)  // the JSON after "[settings-state] "
    case waitingForChiaki
    case waitingForApp
    case found(String)
    case building(String?)  // what a restart is for ("AI engine"), or nil for a TensorRT build
    case playing
    case streamEnded
    case note(String)
    case windowKeys(String)
    case unknownCommand(String)  // the command's name
    case refused(key: String, why: String)
    case log
}

public enum CoreLine {
    public static let statePrefix = "[settings-state] "

    public static func parse(_ line: String) -> CoreEvent {
        if line.hasPrefix(statePrefix) { return .state(String(line.dropFirst(statePrefix.count))) }
        if line.hasPrefix("[chiaki] waiting for chiaki-ng") { return .waitingForChiaki }
        if line.hasPrefix("[tap] waiting for an app") { return .waitingForApp }
        if line.hasPrefix("[chiaki] stream:") { return .found(after(line, "[chiaki] stream:")) }
        if line.hasPrefix("[tap] frames:") { return .found(after(line, "[tap] frames:")) }
        if line.hasPrefix("[trt] building") { return .building(nil) }
        if line.hasPrefix("[run] restarting for ") {  // "[run] restarting for AI engine: RIFE ..."
            let why = after(line, "[run] restarting for ")
            if let r = why.range(of: ": ") { return .building(String(why[..<r.lowerBound])) }
            return .building(why)
        }
        if line.hasPrefix("[run] playing") { return .playing }
        if line.hasPrefix("[chiaki] this stream was already running") {
            return .note("This stream started before SmoovPlay. In chiaki-ng, end the stream and connect again.")
        }
        if line.hasPrefix("[tap] this stream was already running") {
            return .note("This stream started before SmoovPlay. Restart the stream in the app.")
        }
        if line.hasPrefix("[chiaki] WARNING:") || line.hasPrefix("[tap] WARNING:"),
           let r = line.range(of: "WARNING:") {
            return .note(line[r.upperBound...].trimmingCharacters(in: .whitespaces))
        }
        if line.hasPrefix("[chiaki] stream ended") || line.hasPrefix("[tap] the app stopped sending frames") {
            return .streamEnded
        }
        if line.hasPrefix("[window-keys] ") { return .windowKeys(after(line, "[window-keys] ")) }
        if line.hasPrefix("[remote] unknown command: ") {
            let cmd = after(line, "[remote] unknown command: ")
            return .unknownCommand(String(cmd.split(separator: " ").first ?? ""))
        }
        if line.hasPrefix("[remote] can't set ") {
            let rest = after(line, "[remote] can't set ")
            if let r = rest.range(of: ": ") {
                return .refused(key: String(rest[..<r.lowerBound]), why: String(rest[r.upperBound...]))
            }
        }
        return .log
    }

    static func after(_ line: String, _ prefix: String) -> String {
        String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
    }

    /// "set <key> <json>": the command for a setting.
    public static func set(_ key: String, _ value: JSON) -> String { "set \(key) \(value.serialized())" }
}

/// The state the core publishes at the start of every session and after every change.
public struct SettingsState: Equatable, Sendable {
    public var values: [String: JSON]
    public var choices: [String: [JSON]]
    public var disabled: [String: String]
    public var restart: [String]
    public var raw: JSON

    public init(json text: String) throws {
        let j = try JSON.parse(text)
        guard case .object = j else { throw JSON.ParseError(description: "the state isn't an object") }
        raw = j
        values = Dictionary(uniqueKeysWithValues: (j["values"]?.object ?? []).map { ($0.0, $0.1) })
        choices = Dictionary(uniqueKeysWithValues: (j["choices"]?.object ?? []).map { ($0.0, $0.1.array ?? []) })
        disabled = Dictionary(uniqueKeysWithValues: (j["disabled"]?.object ?? []).compactMap { k, v in v.string.map { (k, $0) } })
        restart = (j["restart"]?.array ?? []).compactMap(\.string)
    }

    /// The video window's menu is open (not in the core yet: a "menu_open" field would let the app take the
    /// menu's arrow keys in App window mode).
    public var menuOpen: Bool { raw["menu_open"]?.bool ?? false }
}
