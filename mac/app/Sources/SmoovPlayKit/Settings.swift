// The Settings page's model. Its rows come from the pipeline's own list (python -m framegen.settings), so a new
// setting appears with no app change, as on Windows (installer/launcher/SettingsPage.cs).
//
// The big choices come first ("quick": frame rate, model, upscaler), then Auto-detect, then the rest in folding
// sections by group. A setting that only matters with another ("applies") is hidden while it doesn't. Two to four
// choices make a row with the picked one's hint under it, more make a dropdown, and an on or off setting is a
// switch.
//
// With nothing playing, a change is written to the settings file, which the next session reads. While playing, a
// change goes to the running core ("set <key> <json>"), which applies it at once. The page follows the core's
// "[settings-state]" lines, so changes made in the video window's menu show up here too. While the session starts,
// before the video is up, the rows are locked, because the core has read the file but can't take commands yet.
import Foundation

public struct SettingChoice: Equatable, Sendable {
    public var value: JSON
    public var name: String
    public var hint: String
}

public struct SettingSpec: Equatable, Sendable {
    public var key, label, group, help: String
    public var later: String
    public var picture, speed: String?
    public var live, saved, quick, experimental: Bool
    public var defaultValue: JSON
    public var choices: [SettingChoice]
    public var appliesKey: String?
    public var appliesValues: [JSON]
    public var sources: [String]
    public var elsewhere: [String: String]

    public var onOff: Bool { choices.count == 2 && choices.allSatisfy { $0.value.bool != nil } }

    public func hint(_ v: JSON) -> String { choices.first(where: { $0.value == v })?.hint ?? "" }
    public func name(_ v: JSON) -> String { choices.first(where: { $0.value == v })?.name ?? v.plain }
    public func has(_ v: JSON) -> Bool { choices.contains(where: { $0.value == v }) }

    /// Why it isn't used with frames from this source (a SourceDef kind), or nil if it is (settings.py unused_with).
    public func unusedWith(_ source: String) -> String? {
        if sources.isEmpty || sources.contains(source) { return nil }
        if let w = elsewhere[source], !w.isEmpty { return w }
        if let w = elsewhere[""], !w.isEmpty { return w }
        return "Not used with this source."
    }
}

public struct SettingSpecs: Sendable {
    public var specs: [SettingSpec]
    public var impactPC: String?
    public var path: String?

    public init(json text: String) throws {
        let root = try JSON.parse(text)
        impactPC = root["impact_pc"]?.string
        path = root["path"]?.string
        specs = (root["specs"]?.array ?? []).map { d in
            let applies = d["applies"]
            var elsewhere: [String: String] = [:]
            for (k, v) in d["elsewhere"]?.object ?? [] { if let s = v.string { elsewhere[k] = s } }
            return SettingSpec(
                key: d["key"]?.string ?? "", label: d["label"]?.string ?? "", group: d["group"]?.string ?? "",
                help: d["help"]?.string ?? "", later: d["later"]?.string ?? "",
                picture: d["impact"]?["picture"]?.string, speed: d["impact"]?["speed"]?.string,
                live: d["live"]?.bool ?? true, saved: d["saved"]?.bool ?? true, quick: d["quick"]?.bool ?? false,
                experimental: d["experimental"]?.bool ?? false, defaultValue: d["default"] ?? .null,
                choices: (d["choices"]?.array ?? []).map {
                    SettingChoice(value: $0["value"] ?? .null, name: $0["name"]?.string ?? "", hint: $0["hint"]?.string ?? "")
                },
                appliesKey: applies?["key"]?.string, appliesValues: applies?["values"]?.array ?? [],
                sources: (d["sources"]?.array ?? []).compactMap(\.string), elsewhere: elsewhere)
        }
    }

    public subscript(key: String) -> SettingSpec? { specs.first(where: { $0.key == key }) }
}

/// The pipeline's settings.json, a flat object rewritten in the core's own layout (indent 2, no byte-order mark).
public struct SettingsFile: Sendable {
    public let url: URL
    public init(_ url: URL) { self.url = url }

    public func read() -> JSON {
        guard let d = try? Data(contentsOf: url), let j = try? JSON.parse(data: d), j.object != nil else { return .object([]) }
        return j
    }

    /// Sets one key, keeping every other as it is.
    public func write(_ key: String, _ value: JSON) throws {
        let j = read().setting(key, value)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = url.appendingPathExtension("part")
        try Data(j.pretty().utf8).write(to: tmp)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
    }

    /// One value as plain text ("90", "fsr", "true"), or nil.
    public func text(_ key: String) -> String? { read()[key]?.plain }
}

// MARK: - the page

public struct ChoiceModel: Equatable, Identifiable, Sendable {
    public var id: String { tag }
    public var tag: String  // "off" for the frame-rate pick's Off, else the value's JSON
    public var value: JSON?  // nil for Off
    public var name: String
    public var hint: String
    public var enabled: Bool
    public var tip: String?
    public var selected: Bool

    public init(tag: String, value: JSON?, name: String, hint: String, enabled: Bool, tip: String?, selected: Bool) {
        self.tag = tag; self.value = value; self.name = name; self.hint = hint
        self.enabled = enabled; self.tip = tip; self.selected = selected
    }
}

public enum RowKind: Equatable, Sendable { case toggle, segments, dropdown }

public struct RowModel: Equatable, Identifiable, Sendable {
    public var id: String { key }
    public var key: String
    public var label: String
    public var help: String
    public var kind: RowKind
    public var choices: [ChoiceModel]
    public var visible: Bool
    public var enabled: Bool
    public var note: String?  // why it can't be changed now
    public var hint: String  // for segments, the picked one's hint
    public var restartTag: String?  // "after a restart" while playing, for settings that aren't live
    public var experimental: Bool
    public var later: String
    public var picture, speed: String?
    public var isOn: Bool  // toggles
    public var isRate: Bool  // the frame-rate pick (Frame generation's Off, then the rates)
}

public struct SectionModel: Equatable, Identifiable, Sendable {
    public var id: String { group }
    public var group: String
    public var rows: [RowModel]
    public var summary: String
    public var changedCount: Int
    public var visible: Bool { rows.contains(where: \.visible) }
}

public struct SettingsPage: Equatable, Sendable {
    public var quick: [RowModel]
    public var sections: [SectionModel]
    public var status: String
    public var live: Bool
    public var locked: Bool
    public var restartBanner: String?
}

/// Builds the Settings page from the specs, the live state (while a session reports one) or the file, and the
/// source picked on the Play page. It's pure, so the same inputs give the same page.
public struct SettingsModel: Sendable {
    public var specs: SettingSpecs
    public var state: SettingsState?
    public var file: JSON
    public var source: String
    public var sessionRunning: Bool  // starting or playing (locked until a state line comes)
    public var building = false  // restarting the AI model
    public var restartSent = false
    public var specsError: String?

    public init(specs: SettingSpecs, state: SettingsState?, file: JSON, source: String, sessionRunning: Bool) {
        self.specs = specs
        self.state = state
        self.file = file
        self.source = source
        self.sessionRunning = sessionRunning
    }

    public var live: Bool { state != nil }
    public var locked: Bool { !live && sessionRunning }

    /// The choices a Mac can use. With nothing playing, the ones an NVIDIA card would add are shown greyed out, as
    /// the core does while playing (run.py's mac_mode rows).
    static let macOnly: [String: Set<String>] = [
        "scaler": ["fsr", "bilinear"], "engine": ["auto", "torch"], "model": ["4.22.lite"], "present": ["gl"],
        "hdr": ["off", "sdr"],
    ]
    static let macWhy: [String: String] = [
        "engine": "On a Mac, RIFE runs through PyTorch on Apple's GPU (Metal).",
        "present": "On a Mac, the window always presents through OpenGL.",
        "vfg_quality": "NVIDIA VFG needs an NVIDIA RTX card.",
        "vsr_quality": "RTX VSR needs an NVIDIA RTX card.",
    ]
    static let macChoiceWhy: [String: String] = [
        "scaler": "RTX VSR needs an NVIDIA RTX card.", "model": "NVIDIA VFG needs an NVIDIA RTX card.",
        "engine": "TensorRT is for NVIDIA cards only.", "present": "Not on a Mac.",
        "hdr": "HDR output isn't on the Mac yet: HDR streams are shown in SDR.",
    ]
    /// Settings the Mac app carries out itself, which the core greys out on a Mac.
    public static let appOwned: [String: String] = [
        "minimize_chiaki": "SmoovPlay hides chiaki-ng while the video plays. Change this when nothing is playing.",
    ]

    public func value(_ spec: SettingSpec) -> JSON {
        if let v = state?.values[spec.key] { return v }
        if let v = file[spec.key], spec.has(v) { return v }
        return spec.defaultValue
    }

    public func value(_ key: String) -> JSON? { specs[key].map { value($0) } }

    func disabled(_ key: String) -> String? {
        if let s = state {
            if let own = SettingsModel.appOwned[key], s.disabled[key] != nil { return own }
            return s.disabled[key]
        }
        return SettingsModel.macWhy[key]
    }

    func allowed(_ key: String) -> [JSON]? {
        if let s = state { return s.choices[key] }
        if let ok = SettingsModel.macOnly[key], let spec = specs[key] {
            return spec.choices.map(\.value).filter { ok.contains($0.plain) }
        }
        return nil
    }

    func applies(_ spec: SettingSpec) -> Bool {
        guard let k = spec.appliesKey else { return true }
        guard let v = value(k) else { return true }
        return spec.appliesValues.contains(v)
    }

    public func page() -> SettingsPage {
        var quick: [RowModel] = []
        let picks = specs.specs.filter { $0.saved && $0.quick }
        let interp = picks.first(where: { $0.key == "interp" })
        let hz = picks.first(where: { $0.key == "hz" })
        let rate = interp?.onOff == true && hz != nil
        for spec in picks {
            if rate, spec.key == "interp" { continue }  // the frame-rate pick's Off
            quick.append(row(spec, off: rate && spec.key == "hz" ? interp : nil))
        }
        var groups: [String] = []
        var bySection: [String: [SettingSpec]] = [:]
        for s in specs.specs where s.saved && !s.quick {
            if bySection[s.group] == nil { groups.append(s.group) }
            bySection[s.group, default: []].append(s)
        }
        let sections = groups.map { g -> SectionModel in
            let rows = bySection[g]!.map { row($0, off: nil) }
            let shown = rows.filter(\.visible)
            let changed = shown.compactMap { r -> String? in
                guard let spec = specs[r.key] else { return nil }
                let v = value(spec)
                if v == spec.defaultValue { return nil }
                return spec.onOff ? spec.label + (v.bool == true ? " on" : " off") : spec.label + ": " + spec.name(v)
            }
            let summary = !changed.isEmpty ? changed.joined(separator: "  ·  ")
                : shown.count == 1 ? "1 setting, at its default" : "\(shown.count) settings, all at their defaults"
            return SectionModel(group: g, rows: rows, summary: summary, changedCount: changed.count)
        }
        let status: String
        if let e = specsError { status = "Couldn't load the settings: \(e)." }
        else if live && building { status = "Restarting the AI model. Other changes still reach the video." }
        else if live { status = "Changes reach the video straight away and are saved for next time. M in the video window opens the same settings." }
        else if locked { status = "SmoovPlay is starting. You can change settings again once the video is on screen." }
        else { status = "Saved for every session. While you play, changes reach the video straight away." }
        var banner: String?
        if let s = state, !s.restart.isEmpty, !restartSent {
            banner = "Restart to apply: " + s.restart.joined(separator: ", ") + ". The video pauses for a few seconds."
        }
        return SettingsPage(quick: quick, sections: sections, status: status, live: live, locked: locked, restartBanner: banner)
    }

    func row(_ spec: SettingSpec, off: SettingSpec?) -> RowModel {
        let v = value(spec)
        let why = disabled(spec.key) ?? off.flatMap { disabled($0.key) } ?? (live ? nil : spec.unusedWith(source))
        let on = !locked && why == nil
        let ok = allowed(spec.key)
        let label = off != nil ? "Frame rate" : spec.label
        let help = off.map { $0.help + "\n\n" + spec.help } ?? spec.help
        let kind: RowKind = spec.onOff && off == nil ? .toggle : (off != nil || spec.choices.count <= 4) ? .segments : .dropdown
        let running = off.map { value($0).bool != false } ?? true  // frame generation on
        var choices: [ChoiceModel] = []
        if let off, let offChoice = off.choices.first(where: { $0.value.bool == false }) {
            choices.append(ChoiceModel(tag: "off", value: nil, name: offChoice.name, hint: offChoice.hint, enabled: on,
                                       tip: offChoice.hint.isEmpty ? nil : offChoice.hint, selected: !running))
        }
        for c in spec.choices {
            var can = ok == nil || ok!.contains(c.value)
            var tip: String? = can ? nil : (live ? "Not available on this Mac right now" : SettingsModel.macChoiceWhy[spec.key] ?? "Not available on this Mac")
            if !live, spec.key == "model", c.value.plain == "vfg" { can = false; tip = "NVIDIA VFG needs an NVIDIA RTX card." }
            choices.append(ChoiceModel(tag: c.value.serialized(), value: c.value, name: c.name, hint: c.hint,
                                       enabled: on && can, tip: tip ?? (c.hint.isEmpty ? nil : c.hint),
                                       selected: running && c.value == v))
        }
        let hint = running ? spec.hint(v) : (off.map { $0.hint(.bool(false)) } ?? "")
        let restartTag = live && !spec.live ? (spec.key == "present" ? "next Play" : "after a restart") : nil
        return RowModel(key: spec.key, label: label, help: help, kind: kind, choices: choices, visible: applies(spec),
                        enabled: on, note: why, hint: hint, restartTag: restartTag, experimental: spec.experimental && off == nil,
                        later: spec.later, picture: (off ?? spec).picture, speed: (off ?? spec).speed,
                        isOn: v.bool == true, isRate: off != nil)
    }

    /// What picking a choice changes, in order, as (key, value) pairs. The frame-rate pick's Off turns frame
    /// generation off. A rate sets the rate first, then turns frame generation back on.
    public func pick(_ key: String, tag: String) -> [(String, JSON)] {
        guard let spec = specs[key] else { return [] }
        let rate = key == "hz" && specs["interp"]?.quick == true && specs["interp"]?.onOff == true
        if rate && tag == "off" { return [("interp", false)] }
        guard let c = spec.choices.first(where: { $0.value.serialized() == tag }) else { return [] }
        var out: [(String, JSON)] = [(key, c.value)]
        if rate, let interp = specs["interp"], value(interp).bool != true { out.append(("interp", true)) }
        return out
    }
}
