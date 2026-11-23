// Keys that work from any app, through Carbon's RegisterEventHotKey. It needs no Accessibility permission and logs
// no keys, since macOS tells the app only when one of its own shortcuts is pressed.
//
// While a session plays with instant replay on, the app holds Control-Option-R (save the instant replay) and
// Control-Option-B (a before & after video), as the core's Ctrl+Alt+R and Ctrl+Alt+B do on Windows (replay.py,
// Windows-only there). In App window mode it also holds the video window's own keys (WindowKeys.cs,
// framegen/hotkeys.py), since the app being captured keeps the keyboard. Each press becomes a command for the core.
import Carbon.HIToolbox
import Foundation

public struct KeyCombo: Hashable, Sendable, CustomStringConvertible {
    public var mods: UInt32  // Carbon modifiers: controlKey, optionKey, shiftKey, cmdKey
    public var key: String  // "M", "Esc", "Space", "F5", "Up"

    public init(mods: UInt32, key: String) { self.mods = mods; self.key = key }

    public var keyCode: UInt32 { KeyCombo.codes[key] ?? 0 }

    /// The name as people read it, "Control-Option-M", which is how Apple writes shortcuts.
    public var name: String {
        var parts: [String] = []
        if mods & UInt32(controlKey) != 0 { parts.append("Control") }
        if mods & UInt32(optionKey) != 0 { parts.append("Option") }
        if mods & UInt32(shiftKey) != 0 { parts.append("Shift") }
        if mods & UInt32(cmdKey) != 0 { parts.append("Command") }
        return (parts + [key]).joined(separator: "-")
    }

    public var description: String { name }

    public struct Problem: Error, CustomStringConvertible { public let description: String }

    /// From a name: "Control-Option-M", "ctrl+alt+m", "Ctrl+Opt+Esc", "cmd-shift-f5".
    public static func parse(_ text: String) throws -> KeyCombo {
        let parts = text.replacingOccurrences(of: " ", with: "").split(whereSeparator: { $0 == "+" || $0 == "-" }).map(String.init)
        guard !parts.isEmpty else { throw Problem(description: "no keys") }
        var mods: UInt32 = 0
        var key: String?
        for p in parts {
            switch p.lowercased() {
            case "control", "ctrl", "ctl", "⌃": mods |= UInt32(controlKey)
            case "option", "opt", "alt", "⌥": mods |= UInt32(optionKey)
            case "shift", "⇧": mods |= UInt32(shiftKey)
            case "command", "cmd", "⌘": mods |= UInt32(cmdKey)
            default:
                guard key == nil else { throw Problem(description: "more than one key besides Control, Option, Shift and Command") }
                guard let k = codes.keys.first(where: { $0.lowercased() == aliases[p.lowercased(), default: p.lowercased()] }) else {
                    throw Problem(description: "unknown key \(p)")
                }
                key = k
            }
        }
        guard let key else { throw Problem(description: "no key besides Control, Option, Shift and Command") }
        return KeyCombo(mods: mods, key: key)
    }

    /// From a key press in the app (NSEvent's keyCode and Carbon-style modifiers). nil for a key that can't be used.
    public static func from(keyCode: UInt16, mods: UInt32) -> KeyCombo? {
        guard let k = codes.first(where: { $0.value == UInt32(keyCode) })?.key else { return nil }
        return KeyCombo(mods: mods, key: k)
    }

    static let aliases = ["escape": "esc", "return": "enter", "spacebar": "space", "pgup": "pageup", "pgdn": "pagedown",
                          "del": "delete", "backspace": "backspace", "arrowup": "up", "arrowdown": "down",
                          "arrowleft": "left", "arrowright": "right"]

    static let codes: [String: UInt32] = {
        var d: [String: Int] = [
            "A": kVK_ANSI_A, "B": kVK_ANSI_B, "C": kVK_ANSI_C, "D": kVK_ANSI_D, "E": kVK_ANSI_E, "F": kVK_ANSI_F,
            "G": kVK_ANSI_G, "H": kVK_ANSI_H, "I": kVK_ANSI_I, "J": kVK_ANSI_J, "K": kVK_ANSI_K, "L": kVK_ANSI_L,
            "M": kVK_ANSI_M, "N": kVK_ANSI_N, "O": kVK_ANSI_O, "P": kVK_ANSI_P, "Q": kVK_ANSI_Q, "R": kVK_ANSI_R,
            "S": kVK_ANSI_S, "T": kVK_ANSI_T, "U": kVK_ANSI_U, "V": kVK_ANSI_V, "W": kVK_ANSI_W, "X": kVK_ANSI_X,
            "Y": kVK_ANSI_Y, "Z": kVK_ANSI_Z,
            "0": kVK_ANSI_0, "1": kVK_ANSI_1, "2": kVK_ANSI_2, "3": kVK_ANSI_3, "4": kVK_ANSI_4, "5": kVK_ANSI_5,
            "6": kVK_ANSI_6, "7": kVK_ANSI_7, "8": kVK_ANSI_8, "9": kVK_ANSI_9,
            "Esc": kVK_Escape, "Space": kVK_Space, "Enter": kVK_Return, "Tab": kVK_Tab, "Backspace": kVK_Delete,
            "Delete": kVK_ForwardDelete, "Home": kVK_Home, "End": kVK_End, "PageUp": kVK_PageUp, "PageDown": kVK_PageDown,
            "Left": kVK_LeftArrow, "Right": kVK_RightArrow, "Up": kVK_UpArrow, "Down": kVK_DownArrow,
            "F1": kVK_F1, "F2": kVK_F2, "F3": kVK_F3, "F4": kVK_F4, "F5": kVK_F5, "F6": kVK_F6, "F7": kVK_F7,
            "F8": kVK_F8, "F9": kVK_F9, "F10": kVK_F10, "F11": kVK_F11, "F12": kVK_F12, "F13": kVK_F13,
            "F14": kVK_F14, "F15": kVK_F15, "F16": kVK_F16, "F17": kVK_F17, "F18": kVK_F18, "F19": kVK_F19, "F20": kVK_F20,
        ]
        return d.mapValues { UInt32($0) }
    }()
}

/// The app's shortcuts with macOS. Carbon delivers presses on the main thread, through the app's event loop.
public final class Hotkeys: @unchecked Sendable {
    public static let shared = Hotkeys()

    private var handler: EventHandlerRef?
    private var registered: [UInt32: (ref: EventHotKeyRef, combo: KeyCombo, action: () -> Void)] = [:]
    private var nextID: UInt32 = 1
    private static let signature: OSType = 0x534D_5650  // 'SMVP'

    private func installHandler() {
        guard handler == nil else { return }
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let me = Unmanaged.passUnretained(self).toOpaque()
        InstallEventHandler(GetApplicationEventTarget(), { _, event, user in
            guard let event, let user else { return OSStatus(eventNotHandledErr) }
            var hk = EventHotKeyID()
            let err = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                        nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
            guard err == noErr, hk.signature == Hotkeys.signature else { return OSStatus(eventNotHandledErr) }
            let me = Unmanaged<Hotkeys>.fromOpaque(user).takeUnretainedValue()
            me.registered[hk.id]?.action()
            return noErr
        }, 1, &spec, me, &handler)
    }

    /// Takes the shortcut. The action runs on the main thread each time it's pressed. Returns nil if another app or
    /// macOS has it, else an id for unregister.
    @discardableResult
    public func register(_ combo: KeyCombo, action: @escaping () -> Void) -> UInt32? {
        installHandler()
        if Hotkeys.systemShortcuts().contains(combo) { return nil }
        var ref: EventHotKeyRef?
        let id = nextID
        nextID += 1
        let st = RegisterEventHotKey(combo.keyCode, combo.mods, EventHotKeyID(signature: Hotkeys.signature, id: id),
                                     GetApplicationEventTarget(), OptionBits(kEventHotKeyExclusive), &ref)
        guard st == noErr, let ref else { return nil }
        registered[id] = (ref, combo, action)
        return id
    }

    public func unregister(_ id: UInt32) {
        if let r = registered.removeValue(forKey: id) { UnregisterEventHotKey(r.ref) }
    }

    public func unregisterAll() {
        for id in Array(registered.keys) { unregister(id) }
    }

    public var held: [KeyCombo] { registered.values.map(\.combo) }

    /// True if no other app has it and it isn't one of macOS's own (System Settings > Keyboard > Keyboard
    /// Shortcuts). It's taken for a moment to find out, then let go. One the app itself holds counts as free.
    public func free(_ combo: KeyCombo) -> Bool {
        if registered.values.contains(where: { $0.combo == combo }) { return true }
        if Hotkeys.systemShortcuts().contains(combo) { return false }
        var ref: EventHotKeyRef?
        let st = RegisterEventHotKey(combo.keyCode, combo.mods, EventHotKeyID(signature: Hotkeys.signature, id: 0xFFFF),
                                     GetApplicationEventTarget(), OptionBits(kEventHotKeyExclusive), &ref)
        guard st == noErr, let ref else { return false }
        UnregisterEventHotKey(ref)
        return true
    }

    /// macOS's own shortcuts that are turned on (Mission Control, Spotlight, input sources, screenshots and so on).
    public static func systemShortcuts() -> Set<KeyCombo> {
        var out: Set<KeyCombo> = []
        var array: Unmanaged<CFArray>?
        guard CopySymbolicHotKeys(&array) == noErr, let list = array?.takeRetainedValue() as? [[String: Any]] else { return out }
        for d in list {
            guard (d[kHISymbolicHotKeyEnabled as String] as? Bool) == true,
                  let code = (d[kHISymbolicHotKeyCode as String] as? NSNumber)?.uint16Value,
                  let mods = (d[kHISymbolicHotKeyModifiers as String] as? NSNumber)?.uint32Value,
                  let c = KeyCombo.from(keyCode: code, mods: mods & UInt32(controlKey | optionKey | shiftKey | cmdKey))
            else { continue }
            out.insert(c)
        }
        return out
    }
}

/// App window mode's shortcuts. Keep them in step with WindowKeys.cs and framegen/hotkeys.py on Windows. The app
/// being captured keeps the keyboard, so these work from anywhere, as Control-Option with the video window's usual
/// key. People can change them on the Play page. The prefs keep them as "window-key:<action>=<keys>".
public enum WindowKeys {
    public struct Action: Sendable {
        public let id, label: String
        public let defaults: [String]  // the first free one wins
        public let key: String  // the video window's own key it stands for
    }

    public static let actions: [Action] = [
        Action(id: "menu", label: "Menu", defaults: ["Control-Option-M", "Control-Option-Space"], key: "M"),
        Action(id: "fullscreen", label: "Fullscreen", defaults: ["Control-Option-F"], key: "F"),
        Action(id: "quit", label: "Quit", defaults: ["Control-Option-Esc", "Control-Option-Q"], key: "Q"),
        Action(id: "stats", label: "Stats overlay", defaults: ["Control-Option-S"], key: "S"),
        Action(id: "smooth", label: "Smooth on/off", defaults: ["Control-Option-I"], key: "I"),
    ]

    /// Instant replay's keys, held whenever it's on, with any source.
    public static let replay: [(id: String, combo: String, key: String, what: String)] = [
        ("replay", "Control-Option-R", "R", "saving a replay"),
        ("compare", "Control-Option-B", "B", "saving a before & after"),
    ]

    /// The menu's own keys, taken plain while it's open, once the core says when it is (SettingsState.menuOpen).
    public static let nav = ["Up", "Down", "Left", "Right", "Enter", "Esc", "Backspace"]

    static let prefix = "window-key:"

    public static func picked(_ prefs: Prefs) -> [String: String] {
        var d: [String: String] = [:]
        for f in prefs.flags where f.hasPrefix(prefix) {
            let kv = f.dropFirst(prefix.count).split(separator: "=", maxSplits: 1).map(String.init)
            if kv.count == 2, actions.contains(where: { $0.id == kv[0] }), !kv[1].isEmpty { d[kv[0]] = kv[1] }
        }
        return d
    }

    public static func pick(_ prefs: Prefs, _ action: String, _ combo: String?) {
        prefs.flags.filter { $0.hasPrefix(prefix + action + "=") }.forEach { prefs.set($0, false) }
        if let c = combo { prefs.set(prefix + action + "=" + c, true) }
    }

    public static func resetAll(_ prefs: Prefs) {
        prefs.flags.filter { $0.hasPrefix(prefix) }.forEach { prefs.set($0, false) }
    }

    /// Each action's shortcuts to try. That's the one picked, or else its defaults minus any picked for another
    /// action.
    public static func choices(_ prefs: Prefs) -> [String: [String]] {
        let p = picked(prefs)
        let used = Set(p.values)
        var d: [String: [String]] = [:]
        for a in actions { d[a.id] = p[a.id].map { [$0] } ?? a.defaults.filter { !used.contains($0) } }
        return d
    }

    /// What each action gets on this Mac now (the first free one, or nil), and the shortcuts another app has.
    public static func resolve(_ prefs: Prefs, free: (KeyCombo) -> Bool) -> (got: [String: String?], taken: [String]) {
        var got: [String: String?] = [:]
        var taken: [String] = []
        for a in actions {
            got[a.id] = .some(nil)
            for c in choices(prefs)[a.id] ?? [] {
                if let k = try? KeyCombo.parse(c), free(k) { got[a.id] = .some(c); break }
                taken.append(c)
            }
        }
        return (got, taken)
    }

    /// Why `combo` can't be the shortcut for `action`, or nil.
    public static func problem(_ combo: KeyCombo?, action: String, prefs: Prefs, free: (KeyCombo) -> Bool) -> String? {
        guard let c = combo else {
            return "That key can't be used. Pick a letter, a number, F1 to F20, Esc, Space, Return, Tab, an arrow, Home, End, Page Up, Page Down or Delete."
        }
        if c.mods & UInt32(controlKey | optionKey | cmdKey) == 0 {
            return "Add Control, Option or Command: on its own (or with Shift only), the key would stop working in the game and the app."
        }
        if let own = replay.first(where: { (try? KeyCombo.parse($0.combo)) == c }) {
            return "\(c.name) is SmoovPlay's own, for \(own.what)."
        }
        let got = resolve(prefs, free: free).got
        if let other = actions.first(where: { $0.id != action && got[$0.id] == .some(c.name) }) {
            return "\(other.label) already uses \(c.name). Change that one first."
        }
        if !free(c) { return "Another app or macOS already uses \(c.name). Pick another." }
        return nil
    }

    /// "Control-Option-M opens the menu, Control-Option-F goes fullscreen and Control-Option-Esc quits".
    public static func sentence(_ got: [String: String?]) -> String {
        let parts = ["menu", "fullscreen", "quit"].compactMap { a -> String? in
            guard let k = got[a] ?? nil else { return nil }
            return k + " " + (a == "menu" ? "opens the menu" : a == "fullscreen" ? "goes fullscreen" : "quits")
        }
        if parts.count <= 1 { return parts.first ?? "" }
        return parts.dropLast().joined(separator: ", ") + " and " + parts.last!
    }
}
