// The Looks page's model (installer/launcher/LooksPage.cs, LookPreview.cs). It holds SmoovPlay's looks and the
// user's (framegen/looks.py), and an editor laid out like a color grading app's nodes. A look is a stack of effects,
// applied top to bottom, each with a switch, an amount and its own settings.
//
// The effect types, their settings and SmoovPlay's looks come from the pipeline (python -m framegen.looks schema).
// The user's looks are JSON files in the looks folder ("my:<file name>"), which the app writes itself. Share codes
// and LUT files go through python -m framegen.looks (encode, import, add-lut).
//
// While playing, the video shows the look picked here, live. Changes go to the core as "look-show <JSON>", at most
// 25 times a second, and leaving the page sends "look-hide". SmoovPlay's own looks can be changed freely to try
// things. The changes stay in memory (presetOriginal) until Save as mine makes a copy of yours ("My Cinematic").
import Foundation

public struct LookParam: Equatable, Sendable {
    public var key, label, unit: String
    public var min, max, step, defaultValue: Double
    public var text: Bool  // a file name (the LUT's)
    public var names: [String]  // a choice's names, for min, min + 1 and so on

    public func format(_ v: Double) -> String {
        if key == "mix" { return "\(Int((v * 100).rounded()))%" }
        let places = step >= 1 ? 0 : (step >= 0.05 && abs(step * 10 - (step * 10).rounded()) < 1e-9) ? 1 : 2
        var s = String(format: "%.\(places)f", v)
        if min < 0 && v > 0 { s = "+" + s }
        if unit == "°" || unit == ":1" { return s + unit }
        return unit.isEmpty ? s : s + " " + unit
    }

    public func snap(_ v: Double) -> Double {
        let q = ((Swift.min(Swift.max(v, min), max) - min) / step).rounded()
        return ((min + q * step) * 1e6).rounded() / 1e6
    }
}

public struct LookType: Equatable, Identifiable, Sendable {
    public var id: String { key }
    public var key, label, kind, help: String
    public var params: [LookParam]  // the first is "mix", the amount
}

public struct LookEntry: Equatable, Identifiable, Sendable {
    public var id: String
    public var name: String
    public var builtin: Bool
    public var look: JSON  // {"name", "nodes": [...]}
    public var code: String?  // SmoovPlay's looks only (yours are encoded when copied)
    public var file: URL?

    public var nodes: [JSON] { look["nodes"]?.array ?? [] }
}

public struct LookSample: Equatable, Identifiable, Sendable {
    public var id, game, scene, credit, path: String
}

public struct LooksLibrary: Sendable {
    public var types: [LookType]
    public var entries: [LookEntry]
    public var luts: [String]
    public var folder: URL
    public var lutFolder: URL
    public var maxNodes: Int
    public var samples: [LookSample]
    /// SmoovPlay's looks with changes being tried, by id, each with the look as it was.
    public var presetOriginal: [String: JSON] = [:]

    public init(json text: String) throws {
        let root = try JSON.parse(text)
        if let e = root["error"]?.string { throw JSON.ParseError(description: e) }
        folder = URL(fileURLWithPath: root["folder"]?.string ?? "", isDirectory: true)
        lutFolder = URL(fileURLWithPath: root["lut_folder"]?.string ?? "", isDirectory: true)
        luts = (root["luts"]?.array ?? []).compactMap(\.string)
        maxNodes = root["max_nodes"]?.int ?? 16
        types = (root["types"]?.array ?? []).map { t in
            LookType(key: t["key"]?.string ?? "", label: t["label"]?.string ?? "", kind: t["kind"]?.string ?? "",
                     help: t["help"]?.string ?? "", params: (t["params"]?.array ?? []).map { p in
                         let text = p["text"]?.bool ?? false
                         return LookParam(key: p["key"]?.string ?? "", label: p["label"]?.string ?? "", unit: p["unit"]?.string ?? "",
                                          min: p["min"]?.double ?? 0, max: p["max"]?.double ?? 1, step: p["step"]?.double ?? 0.01,
                                          defaultValue: text ? 0 : p["default"]?.double ?? 0, text: text,
                                          names: (p["names"]?.array ?? []).compactMap(\.string))
                     })
        }
        entries = (root["looks"]?.array ?? []).map { e in
            LookEntry(id: e["id"]?.string ?? "", name: e["name"]?.string ?? "", builtin: e["builtin"]?.bool ?? false,
                      look: e["look"] ?? .object([("name", .string("")), ("nodes", .array([]))]), code: e["code"]?.string,
                      file: e["path"]?.string.map { URL(fileURLWithPath: $0) })
        }
        samples = LooksLibrary.samples(root["samples"])
    }

    public static func samples(_ j: JSON?) -> [LookSample] {
        (j?.array ?? []).map {
            LookSample(id: $0["id"]?.string ?? "", game: $0["game"]?.string ?? "", scene: $0["scene"]?.string ?? "",
                       credit: $0["credit"]?.string ?? "", path: $0["path"]?.string ?? "")
        }
    }

    public func entry(_ id: String?) -> LookEntry? { entries.first(where: { $0.id == id }) }
    public func type(of node: JSON) -> LookType? { types.first(where: { $0.key == node["type"]?.string }) }
    public func changed(_ id: String) -> Bool { presetOriginal[id] != nil }

    /// SmoovPlay's looks first (so they stay where they are), then yours by name.
    public mutating func sort() {
        entries = entries.filter(\.builtin) + entries.filter { !$0.builtin }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    public func uniqueName(_ name: String) -> String {
        var n = name
        var k = 2
        while entries.contains(where: { $0.name.caseInsensitiveCompare(n) == .orderedSame }) { n = "\(name) \(k)"; k += 1 }
        return n
    }

    /// A new file, and so an id, for a look of yours. The file is named after the look, in lower case letters and
    /// digits.
    public func newFile(for name: String) -> (String, URL) {
        var stem = name.lowercased().replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        if stem.count > 40 { stem = String(stem.prefix(40)).trimmingCharacters(in: CharacterSet(charactersIn: "-")) }
        if stem.isEmpty { stem = "look" }
        var s = stem
        var k = 2
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent(s + ".json").path)
            || entries.contains(where: { $0.id == "my:" + s }) { s = "\(stem)-\(k)"; k += 1 }
        return ("my:" + s, folder.appendingPathComponent(s + ".json"))
    }

    /// A node of this type with its defaults (a LUT node gets the first LUT).
    public func newNode(_ t: LookType) -> JSON {
        var o: [(String, JSON)] = [("type", .string(t.key)), ("on", true)]
        for p in t.params { o.append((p.key, p.text ? .string(luts.first ?? "") : .num(p.defaultValue))) }
        return .object(o)
    }

    // MARK: editing (index into entries)
    /// Gets the look at `i` ready to change. A change to one of SmoovPlay's looks is a try-out, and what it was is
    /// kept.
    mutating func editable(_ i: Int) {
        let e = entries[i]
        if e.builtin && presetOriginal[e.id] == nil { presetOriginal[e.id] = e.look }
    }

    mutating func setNodes(_ i: Int, _ nodes: [JSON]) {
        entries[i].look = entries[i].look.setting("nodes", .array(nodes))
        let e = entries[i]
        if e.builtin, let was = presetOriginal[e.id], was == e.look { presetOriginal[e.id] = nil }  // all put back
    }

    public mutating func setValue(_ i: Int, node: Int, key: String, _ v: JSON) {
        guard entries.indices.contains(i), entries[i].nodes.indices.contains(node) else { return }
        editable(i)
        var nodes = entries[i].nodes
        nodes[node] = nodes[node].setting(key, v)
        setNodes(i, nodes)
    }

    public mutating func addNode(_ i: Int, _ t: LookType) {
        guard entries.indices.contains(i), entries[i].nodes.count < maxNodes else { return }
        editable(i)
        setNodes(i, entries[i].nodes + [newNode(t)])
    }

    public mutating func moveNode(_ i: Int, node: Int, by: Int) {
        var nodes = entries[i].nodes
        let j = node + by
        guard nodes.indices.contains(node), nodes.indices.contains(j) else { return }
        editable(i)
        nodes.swapAt(node, j)
        setNodes(i, nodes)
    }

    public mutating func removeNode(_ i: Int, node: Int) {
        var nodes = entries[i].nodes
        guard nodes.indices.contains(node) else { return }
        editable(i)
        nodes.remove(at: node)
        setNodes(i, nodes)
    }

    public mutating func reset(_ i: Int) {
        if let was = presetOriginal.removeValue(forKey: entries[i].id) { entries[i].look = was }
    }

    /// The changes tried on one of SmoovPlay's looks become a look of yours ("My Cinematic"), and SmoovPlay's goes
    /// back. Returns the new look's id.
    public mutating func saveAsMine(_ i: Int) -> String {
        let e = entries[i]
        let name = uniqueName("My " + e.name)
        let (id, file) = newFile(for: name)
        let copy = LookEntry(id: id, name: name, builtin: false, look: e.look.setting("name", .string(name)), code: nil, file: file)
        reset(i)
        entries.append(copy)
        sort()
        return id
    }

    /// A new look of yours with one Color node. Returns its id.
    public mutating func newLook() -> String {
        let name = uniqueName("My look")
        let (id, file) = newFile(for: name)
        let color = types.first(where: { $0.key == "color" }) ?? types.first
        let node = color.map(newNode) ?? .object([])
        let look = JSON.object([("name", .string(name)), ("nodes", .array([node]))])
        entries.append(LookEntry(id: id, name: name, builtin: false, look: look, code: nil, file: file))
        sort()
        return id
    }

    public mutating func rename(_ i: Int, _ name: String) {
        let clean = name.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard !clean.isEmpty, !entries[i].builtin else { return }
        entries[i].name = clean
        entries[i].look = entries[i].look.setting("name", .string(clean))
        sort()
    }

    /// A look of yours, as its file wants it: {"version": 1, "name", "nodes"}.
    public static func fileJSON(_ e: LookEntry) -> JSON {
        .object([("version", 1), ("name", .string(e.name)), ("nodes", .array(e.nodes))])
    }

    public func save(_ e: LookEntry) throws {
        guard !e.builtin, let file = e.file else { return }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(LooksLibrary.fileJSON(e).serialized().utf8).write(to: file, options: .atomic)
    }

    /// A share code in pasted text ("SPL1-..."), even if a chat app wrapped it onto more lines, or nil.
    public static func code(in text: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: #"SPL\d+-[A-Za-z0-9_\-]+([ \t]*\r?\n\s*[A-Za-z0-9_\-]+)*"#),
              let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let r = Range(m.range, in: text) else { return nil }
        return text[r].replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression)
    }
}

/// The preview, drawn by the pipeline itself (python -m framegen.looks_preview), which keeps running while the app
/// runs. One request is out at a time. Changes made meanwhile are sent together once it answers, so dragging a
/// slider redraws as fast as the process keeps up.
public final class LookPreview: @unchecked Sendable {
    private var proc: LineProcess?
    private let paths: AppPaths
    private var ready = false, busy = false
    private var pending: (look: JSON, sample: String, w: Int, h: Int)?
    private var seq = 0, fileNo = 0, starts = 0
    private let dir: URL
    public var onPicture: (URL, String) -> Void = { _, _ in }  // file, sample id
    public var onSamples: ([LookSample]) -> Void = { _ in }
    public var trace: (String) -> Void = { _ in }

    public init(paths: AppPaths) {
        self.paths = paths
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("SmoovPlay-preview", isDirectory: true)
    }

    private func start() {
        guard proc == nil, paths.runtimeReady, starts < 3 else { return }
        starts += 1
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let (exe, args) = PyTool.command(paths, module: "framegen.looks_preview")
        let p = LineProcess(executable: exe, arguments: args, cwd: paths.repo, env: paths.env)
        p.onLine = { [weak self] l in self?.line(l) }
        p.onExit = { [weak self, weak p] _ in
            guard let self, self.proc === p else { return }
            self.proc = nil; self.ready = false; self.busy = false
        }
        proc = p
        do { try p.start(); trace("started") } catch { trace("couldn't start: \(error)") }
    }

    /// Draw this look on the sample, at this size in pixels (now, or once the one being drawn is done).
    public func request(look: JSON, sample: String, width: Int, height: Int) {
        pending = (look, sample, width, height)
        if proc == nil { start(); return }
        send()
    }

    private func send() {
        guard ready, !busy, let p = pending, let proc else { return }
        pending = nil
        seq += 1
        fileNo ^= 1
        let out = dir.appendingPathComponent("preview-\(fileNo).bmp")
        let req = JSON.object([("id", .int(seq)), ("look", p.look), ("sample", .string(p.sample)),
                               ("width", .int(p.w)), ("height", .int(p.h)), ("out", .string(out.path))])
        if proc.send(req.serialized()) { busy = true } else { stop() }
    }

    private func line(_ l: String) {
        guard l.hasPrefix("{"), let d = try? JSON.parse(l) else { if !l.isEmpty { trace(l) }; return }
        if d["ready"] != nil {
            ready = true
            onSamples(LooksLibrary.samples(d["samples"]))
            send()
            return
        }
        busy = false
        if let e = d["error"]?.string { trace(e) }
        if let f = d["file"]?.string, let s = d["sample"]?.string { onPicture(URL(fileURLWithPath: f), s) }
        send()
    }

    public func stop() {
        proc?.closeInput()
        let p = proc
        proc = nil
        ready = false
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.8) { p?.terminate() }
    }
}
