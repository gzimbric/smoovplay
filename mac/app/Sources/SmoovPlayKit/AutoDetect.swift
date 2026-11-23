// Auto-detect, at the top of the Settings page (installer/launcher/AutoDetect.cs). python -m framegen.autodetect
// times the GPU in about 10 s, or a second or two with --timing. It reads the display and what's installed, and
// suggests settings with the reason for each. Apply changes them the same way picking them on the page does. The
// test needs the GPU to itself, so it can't run while playing, and Play waits for it. Its output goes to
// <logs>/autodetect.txt.
import Foundation

public struct DetectFound: Equatable, Sendable {
    public var label, value, detail: String
}

public struct DetectResult: Equatable, Sendable {
    public var settings: [(String, JSON)]
    public var reasons: [String: String]
    public var notes: [String]
    public var found: [DetectFound]

    public static func == (a: DetectResult, b: DetectResult) -> Bool {
        a.settings.map(\.0) == b.settings.map(\.0) && a.settings.map(\.1) == b.settings.map(\.1)
            && a.reasons == b.reasons && a.notes == b.notes && a.found == b.found
    }

    public init(json text: String) throws {
        let j = try JSON.parse(text)
        settings = j["settings"]?.object ?? []
        var r: [String: String] = [:]
        for (k, v) in j["reasons"]?.object ?? [] { if let s = v.string { r[k] = s } }
        reasons = r
        notes = (j["notes"]?.array ?? []).compactMap(\.string)
        found = (j["found"]?.array ?? []).map {
            DetectFound(label: $0["label"]?.string ?? "", value: $0["value"]?.string ?? "", detail: $0["detail"]?.string ?? "")
        }
    }

    /// The suggestions that differ from the current settings, in the page's order: (spec, suggested value).
    public func changes(_ model: SettingsModel) -> [(SettingSpec, JSON)] {
        let suggested = Dictionary(settings, uniquingKeysWith: { a, _ in a })
        return model.specs.specs.compactMap { spec in
            guard let v = suggested[spec.key], spec.has(v), model.value(spec) != v else { return nil }
            return (spec, spec.choices.first(where: { $0.value == v })!.value)
        }
    }
}

public enum AutoDetect {
    public static let resultPrefix = "[detect-result] ", linePrefix = "[detect] "

    /// A "[detect] timing the AI model" line as progress ("Timing the AI model"), or nil for other lines.
    public static func progress(_ line: String) -> String? {
        guard line.hasPrefix(linePrefix), !line.contains("ERROR") else { return nil }
        let step = String(line.dropFirst(linePrefix.count))
        return step.prefix(1).uppercased() + step.dropFirst()
    }

    /// Runs it. It blocks, so call it off the main thread. Returns the result, or why it didn't finish.
    public static func run(_ paths: AppPaths, extra: [String], progress: @escaping (String) -> Void) -> Result<DetectResult, DetectError> {
        try? FileManager.default.createDirectory(at: paths.logs, withIntermediateDirectories: true)
        let logURL = paths.logs.appendingPathComponent("autodetect.txt")
        var logText = ""
        var json: String?
        var last: String?
        let r = PyTool.run(paths, module: "framegen.autodetect", ["--settings", paths.settingsFile.path] + extra, timeout: 180) { line in
            logText += line + "\n"
            if line.hasPrefix(resultPrefix) { json = String(line.dropFirst(resultPrefix.count)); return }
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { last = line }
            if let p = AutoDetect.progress(line) { progress(p) }
        }
        try? logText.write(to: logURL, atomically: true, encoding: .utf8)
        if let json, let res = try? DetectResult(json: json) { return .success(res) }
        let why = (last.map { $0.hasPrefix(linePrefix) ? String($0.dropFirst(linePrefix.count)).replacingOccurrences(of: "ERROR: ", with: "") : $0 })
            ?? "exit code \(r.code)"
        if r.code == 2 { return .failure(DetectError(text: "PyTorch can't use this Mac's GPU (Metal) here. SmoovPlay needs an Apple silicon Mac.")) }
        return .failure(DetectError(text: "The test didn't finish: \(String(why.prefix(180))). Its log is \(logURL.path)."))
    }

    public struct DetectError: Error, Sendable { public let text: String }
}

/// A test-mode report (run.py --results), read for its verdict and a few numbers for the results card, as
/// ShowResults does.
public struct TestResults: Equatable, Sendable {
    public struct Tile: Equatable, Sendable {
        public var label, value, hint: String
        public var accent: Bool
    }

    public var verdict: String
    public var tiles: [Tile]
    public var note: String
    public var text: String
    /// Includes NVIDIA VFG, whose licence forbids sharing its results (run.py PRIVATE_RESULTS).
    public var privateResults: Bool

    public static let vfgPrivate = "These results include NVIDIA VFG."

    public init?(text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        self.text = text
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map { String($0).trimmingCharacters(in: CharacterSet(charactersIn: "\r")) }
        func last(_ p: String) -> String { lines.last(where: { $0.hasPrefix(p) }) ?? "" }
        let v = lines.last(where: { $0.hasPrefix("Verdict: ") }).map { String($0.dropFirst("Verdict: ".count)) } ?? ""
        verdict = v.isEmpty ? "Not enough was played to judge. Next time, play for a minute or more."
            : v.prefix(1).uppercased() + v.dropFirst() + "."
        var t: [Tile] = []
        let quality = last("Quality: ")
        if let m = TestResults.match(quality, #"([\d.]+%) of each AI frame was visibly off.*?against ([\d.]+%)"#), m.count == 2 {
            t.append(Tile(label: "With AI", value: m[0], hint: "visibly off", accent: true))
            t.append(Tile(label: "Without AI", value: m[1], hint: "visibly off (last frame repeated)", accent: false))
        } else if quality.contains("not measured") {
            t.append(Tile(label: "Quality", value: "n/a", hint: "measured at 60 or 120 fps", accent: false))
        }
        if let m = TestResults.match(last("Running behind: "), #"([\d.]+)%"#) {
            t.append(Tile(label: "Running behind", value: m[0] + "%", hint: "of frames", accent: false))
        }
        if let m = TestResults.match(last("AI frame time: "), #"([\d.]+) ms"#) {
            t.append(Tile(label: "AI frame time", value: m[0] + " ms", hint: "out of 16.7 ms", accent: false))
        }
        tiles = t
        var facts: [String] = []
        let output = last("Output: ")
        if !output.isEmpty { facts.append(String(output.dropFirst("Output: ".count)) + " output") }
        if let m = TestResults.match(last("Stream: "), #"played ([\d:]+)"#) { facts.append(m[0] + " played") }
        note = (facts.isEmpty ? "" : facts.joined(separator: ", ") + ". ")
            + "Visibly off: the share of each frame more than about 5% brighter or darker than the real one. Lower is "
            + "better; slow scenes score well either way."
        privateResults = text.contains(TestResults.vfgPrivate)
    }

    static func match(_ s: String, _ pattern: String) -> [String]? {
        guard let re = try? NSRegularExpression(pattern: pattern), let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) else { return nil }
        return (1..<m.numberOfRanges).compactMap { Range(m.range(at: $0), in: s).map { String(s[$0]) } }
    }
}
