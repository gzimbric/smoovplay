// Copy diagnostics (installer/launcher Core.cs Diagnostics and Feedback.cs) collects the Mac, the checks, the
// source, the settings, and the last session's log and test report, ready to paste into a bug report. Bug reports
// get pasted in public places, so the user's home folder becomes "~" however a log spells it, and device names lose
// their owner's name ("Alex's iPhone Camera" becomes "(name)'s iPhone Camera").
import Foundation

public enum Diagnostics {
    public static func anonymize(_ text: String, home: String = AppPaths.home) -> String {
        var t = text
        let h = home.hasSuffix("/") ? String(home.dropLast()) : home
        if !h.isEmpty {
            // the whole folder name, so /Users/Al leaves /Users/Alice alone
            let re = try! NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: h) + #"(?=$|[/"'\s:;,)\]])"#,
                                              options: [.caseInsensitive, .anchorsMatchLines])
            t = re.stringByReplacingMatches(in: t, range: NSRange(t.startIndex..., in: t), withTemplate: "~")
        }
        // any other user's folder, /Users/NAME (not Shared)
        let other = try! NSRegularExpression(pattern: #"/Users/(?!Shared(?:/|"|$))[^/"\s]+"#, options: [.anchorsMatchLines])
        t = other.stringByReplacingMatches(in: t, range: NSRange(t.startIndex..., in: t), withTemplate: "~")
        return noOwner(t)
    }

    /// "Alex's iPhone Camera" becomes "(name)'s iPhone Camera".
    public static func noOwner(_ s: String) -> String {
        let re = try! NSRegularExpression(pattern: #"[^\s()"]+['’]s\b"#)
        return re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "(name)'s")
    }

    /// The last `max` lines of a log.
    public static func tail(_ url: URL, _ max: Int) -> [String] {
        guard let d = try? Data(contentsOf: url) else { return [] }
        let lines = String(decoding: d, as: UTF8.self).split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        return Array(lines.suffix(max))
    }

    public static func newest(_ dir: URL, prefix: String, ext: String) -> URL? {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        return items.filter { $0.lastPathComponent.hasPrefix(prefix) && $0.pathExtension == ext }
            .max { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return da < db
            }
    }

    public static func collect(paths: AppPaths, version: String, source: String, checks: [Check], lastLog: URL?,
                               extra: [String] = []) -> String {
        var out: [String] = []
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        out.append("SmoovPlay for Mac \(version) diagnostics, \(df.string(from: Date()))")
        let os = ProcessInfo.processInfo.operatingSystemVersion
        out.append("macOS: \(os.majorVersion).\(os.minorVersion).\(os.patchVersion), \(MacInfo.chip), \(MacInfo.memoryGB) GB")
        out.append("Source: \(source)")
        for c in checks { out.append("Check [\(c.level.rawValue)] \(c.title): \(c.value). \(c.detail)") }
        out.append("Core: \(paths.coreName)\(paths.coreProblem.map { " (\($0))" } ?? "")")
        out.append("Repo: \(paths.repo.path)")
        out.append("Python: \(paths.python.path)")
        out += extra
        let settings = (try? String(contentsOf: paths.settingsFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        out.append("Settings: " + (settings ?? "none saved"))
        let log = lastLog ?? newest(paths.logs, prefix: "session-", ext: "log")
        if let log, FileManager.default.fileExists(atPath: log.path) {
            let lines = tail(log, 200)
            out.append("Log \(log.lastPathComponent), last \(lines.count) lines:")
            out += lines
        } else {
            out.append("No game session log yet: after playing, press Copy diagnostics again.")
        }
        if let r = newest(paths.logs, prefix: "results-", ext: "txt"), let text = try? String(contentsOf: r, encoding: .utf8) {
            out.append("Test results \(r.lastPathComponent):")
            // NVIDIA's licence forbids publishing VFG's results, and bug reports get pasted in public places.
            out.append(text.contains(TestResults.vfgPrivate)
                ? "(left out: they include NVIDIA VFG, whose licence doesn't allow sharing its results)"
                : text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return anonymize(out.joined(separator: "\n") + "\n")
    }
}
