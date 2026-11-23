// What the launcher remembers between runs, in the Windows launcher's format (one flag a line, next to the
// pipeline's settings): the setup guides' ticks ("guide:<id>"), the source picked on the Play page ("source:<kind>"),
// open Settings sections ("set-open:<group>"), the App window shortcuts ("window-key:<action>=<keys>"), and
// "<prefix>:<value>" choices such as the capture card or the Looks page's sample picture.
import Foundation

public final class Prefs: @unchecked Sendable {
    public let url: URL
    public private(set) var flags: Set<String>

    public init(url: URL) {
        self.url = url
        let text = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        flags = Set(text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })
    }

    public func has(_ f: String) -> Bool { flags.contains(f) }

    public func set(_ f: String, _ on: Bool) {
        if on { flags.insert(f) } else { flags.remove(f) }
        save()
    }

    /// The value after "prefix" ("capture-video:" -> "0"), or nil.
    public func value(_ prefix: String) -> String? {
        flags.first(where: { $0.hasPrefix(prefix) }).map { String($0.dropFirst(prefix.count)) }
    }

    public func choose(_ prefix: String, _ value: String?) {
        flags = flags.filter { !$0.hasPrefix(prefix) }
        if let v = value, !v.isEmpty { flags.insert(prefix + v) }
        save()
    }

    public func save() {
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try (flags.sorted().joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
        } catch {}
    }
}
