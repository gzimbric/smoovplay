// Where everything is. The app runs against a checkout of the repo, as the Windows launcher's dev mode does. The
// repo's .venv Python runs the pipeline's helpers (autodetect, looks, settings and the rest), and the core is
// run.py, or the Rust core (mac/core/target/release/smoovplay) with the same command line.
//
// Environment switches (the Windows launcher's, where they apply):
//   SMOOVPLAY_APP        the repo (by default the one the app was built from, then the folders above the app)
//   SMOOVPLAY_PYTHON     the Python (default <repo>/.venv/bin/python)
//   SMOOVPLAY_CORE       "python" (run.py), "rust" (mac/core/target/release/smoovplay) or a path to a core binary.
//                        It overrides the choice on the Help page. With neither, the core is the Rust one once it's
//                        built, else run.py. With a Rust core, the helpers (Auto-detect, looks, preview and so on)
//                        are its subcommands.
//   SMOOVPLAY_RUN_ARGS   extra arguments for the core
//   SMOOVPLAY_SETTINGS   another settings file (passed to the core and to Auto-detect as --settings)
//   SMOOVPLAY_PREFS      another file for the launcher's own choices (the guide's ticks, the source and so on)
//   SMOOVPLAY_LOGS       another logs folder (default ~/Library/Logs/SmoovPlay)
//   SMOOVPLAY_CHIAKI     another chiaki-ng.app
//   SMOOVPLAY_SOURCES    "all" also offers the hidden sources (Video file)
//   SMOOVPLAY_SOURCE_FILE  the video file for the Video file source
//   SMOOVPLAY_DETECT_ARGS  extra Auto-detect arguments (--timing 9.3,6.8 skips timing the GPU)
//   SMOOVPLAY_SHOTS      a folder for a PNG of the window on every page or state change, drawn by the app itself
import Foundation

public enum CoreKind: String, CaseIterable, Sendable {
    case python, rust, custom
}

public struct AppPaths: Sendable {
    public var repo: URL  // the checkout, or a packaged app's Resources folder
    public var packaged = false  // running from an app made by mac/package.sh, with the core and chiaki-ng inside
    public var python: URL
    public var coreKind: CoreKind
    public var coreBinary: URL?  // rust or custom
    public var runArgs: [String]
    public var settingsFile: URL
    public var settingsOverridden: Bool
    public var prefsFile: URL
    public var logs: URL
    public var chiakiApp: URL
    public var looksDir: URL
    public var env: [String: String]

    public static let chiakiBundleID = "org.streetpea.chiaking"

    /// The framegen data folder (framegen/paths.py data_dir on a Mac).
    public static var dataDir: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/framegen")
    }

    public static func detect(env: [String: String] = ProcessInfo.processInfo.environment,
                              bundle: Bundle = .main, coreChoice: String? = nil) -> AppPaths {
        let fm = FileManager.default
        func dir(_ s: String) -> URL { URL(fileURLWithPath: (s as NSString).expandingTildeInPath, isDirectory: true) }
        // A packaged app (mac/package.sh) carries the Rust core as Contents/Helpers/smoovplay and chiaki-ng under
        // Contents/Helpers, and needs no checkout or Python. SMOOVPLAY_APP still points it at a checkout instead.
        let bundledCore = bundle.bundleURL.appendingPathComponent("Contents/Helpers/smoovplay")
        let packaged = (env["SMOOVPLAY_APP"] ?? "").isEmpty && fm.isExecutableFile(atPath: bundledCore.path)
        var repo: URL? = packaged ? bundle.resourceURL : nil
        if !packaged, let r = env["SMOOVPLAY_APP"], !r.isEmpty { repo = dir(r) }
        if repo == nil, let r = bundle.object(forInfoDictionaryKey: "SmoovPlayRepo") as? String,
           fm.fileExists(atPath: r + "/run.py") { repo = dir(r) }
        if repo == nil {  // the folders above the app (mac/app/build/SmoovPlay.app) or the binary (mac/app/.build/...)
            var u = bundle.bundleURL
            for _ in 0..<8 {
                u = u.deletingLastPathComponent()
                if fm.fileExists(atPath: u.appendingPathComponent("run.py").path) { repo = u; break }
            }
        }
        if repo == nil {
            var u = URL(fileURLWithPath: fm.currentDirectoryPath)
            for _ in 0..<6 {
                if fm.fileExists(atPath: u.appendingPathComponent("run.py").path) { repo = u; break }
                u = u.deletingLastPathComponent()
            }
        }
        let root = repo ?? dir("~/dev/smoovplay")
        let python = env["SMOOVPLAY_PYTHON"].map { URL(fileURLWithPath: $0) } ?? root.appendingPathComponent(".venv/bin/python")

        let rustCore = packaged ? bundledCore : root.appendingPathComponent("mac/core/target/release/smoovplay")
        // There's no Python in a packaged app, so a saved choice of run.py doesn't apply there.
        let saved = packaged && ["python", "run.py"].contains((coreChoice ?? "").lowercased()) ? nil : coreChoice
        let choice = env["SMOOVPLAY_CORE"] ?? saved
            ?? (fm.isExecutableFile(atPath: rustCore.path) ? "rust" : "python")
        var kind = CoreKind.python
        var binary: URL?
        switch choice.lowercased() {
        case "", "python", "run.py": kind = .python
        case "rust", "smoovplay": kind = .rust; binary = rustCore
        default: kind = .custom; binary = URL(fileURLWithPath: (choice as NSString).expandingTildeInPath)
        }

        let settings = env["SMOOVPLAY_SETTINGS"].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let prefs = env["SMOOVPLAY_PREFS"].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? dataDir.appendingPathComponent("smoovplay-launcher.txt")
        let logs = env["SMOOVPLAY_LOGS"].map { dir($0) }
            ?? fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/SmoovPlay", isDirectory: true)
        let chiaki = env["SMOOVPLAY_CHIAKI"].map { dir($0) }
            ?? (packaged ? bundle.bundleURL.appendingPathComponent("Contents/Helpers/chiaki-ng.app", isDirectory: true)
                : root.appendingPathComponent("build/chiaki-ng-frametap/chiaki-ng.app", isDirectory: true))
        // framegen/looks.py looks in $APPDATA/framegen/looks. A Mac has no APPDATA, so that's ~/framegen/looks, not
        // the data folder the settings are in. Every Python process the app starts gets this one instead, so the
        // looks sit beside the settings and the core and the Looks page agree.
        let looks = env["SMOOVPLAY_LOOKS_DIR"].map { dir($0) } ?? dataDir.appendingPathComponent("looks", isDirectory: true)
        var childEnv = env
        childEnv["SMOOVPLAY_LOOKS_DIR"] = looks.path
        childEnv["PYTHONUNBUFFERED"] = "1"  // lines arrive as they're printed
        childEnv["PYTHONIOENCODING"] = "utf-8"
        return AppPaths(repo: root, packaged: packaged, python: python, coreKind: kind, coreBinary: binary,
                        runArgs: split(env["SMOOVPLAY_RUN_ARGS"] ?? ""),
                        settingsFile: settings ?? dataDir.appendingPathComponent("settings.json"),
                        settingsOverridden: settings != nil, prefsFile: prefs, logs: logs, chiakiApp: chiaki,
                        looksDir: looks, env: childEnv)
    }

    public var runPy: URL { repo.appendingPathComponent("run.py") }

    /// What runs the core and its helpers is there, either the Rust core binary or the Python and run.py. This is
    /// the Windows launcher's RuntimeReady, for a dev checkout.
    public var runtimeReady: Bool {
        if coreKind != .python { return coreProblem == nil }
        return FileManager.default.isExecutableFile(atPath: python.path) && FileManager.default.fileExists(atPath: runPy.path)
    }

    /// Why the core can't start, or nil.
    public var coreProblem: String? {
        switch coreKind {
        case .python:
            if !FileManager.default.isExecutableFile(atPath: python.path) {
                return "There's no Python at \(python.path). Make the repo's .venv (docs/DEVELOPING.md), or set SMOOVPLAY_PYTHON."
            }
            if !FileManager.default.fileExists(atPath: runPy.path) { return "There's no run.py in \(repo.path)." }
            return nil
        case .rust, .custom:
            guard let b = coreBinary, FileManager.default.isExecutableFile(atPath: b.path) else {
                return coreKind == .rust
                    ? "The Rust core isn't built yet. Build it with cargo build --release in mac/core, which makes mac/core/target/release/smoovplay."
                    : "There's no core at \(coreBinary?.path ?? "that path")."
            }
            return nil
        }
    }

    /// The core's command, as (executable, the arguments before SmoovPlay's own). run.py's flags follow either way.
    public var coreCommand: (URL, [String]) {
        switch coreKind {
        case .python: return (python, [runPy.path])
        case .rust, .custom: return (coreBinary ?? python, [])
        }
    }

    public var coreName: String {
        switch coreKind {
        case .python: return "Python (run.py)"
        case .rust: return "Rust (smoovplay)"
        case .custom: return coreBinary?.lastPathComponent ?? "custom"
        }
    }

    /// --settings for the core and Auto-detect, when another file is in use (else the core's default, the same file).
    public var settingsArgs: [String] { settingsOverridden ? ["--settings", settingsFile.path] : [] }

    /// Splits SMOOVPLAY_RUN_ARGS as a shell would. Spaces separate, and double or single quotes group.
    public static func split(_ s: String) -> [String] {
        var out: [String] = [], cur = "", quote: Character?
        var any = false
        for c in s {
            if let q = quote {
                if c == q { quote = nil } else { cur.append(c) }
            } else if c == "\"" || c == "'" {
                quote = c; any = true
            } else if c == " " || c == "\t" {
                if !cur.isEmpty || any { out.append(cur); cur = ""; any = false }
            } else {
                cur.append(c)
            }
        }
        if !cur.isEmpty || any { out.append(cur) }
        return out
    }

    /// The user's home, which becomes "~" in text that may be pasted in public (Copy diagnostics).
    public static var home: String { FileManager.default.homeDirectoryForCurrentUser.path }
}

/// SMOOVPLAY_TRACE=<file> turns on developer tracing, like the Windows launcher's switch of the same name.
public enum Trace {
    static let path = ProcessInfo.processInfo.environment["SMOOVPLAY_TRACE"]
    static let lock = NSLock()

    public static func log(_ s: String) {
        guard let path else { return }
        lock.lock()
        defer { lock.unlock() }
        let line = "\(Date().formatted(.iso8601)) \(s)\n"
        if let h = FileHandle(forWritingAtPath: path) {
            h.seekToEndOfFile()
            h.write(Data(line.utf8))
            try? h.close()
        } else {
            try? line.write(toFile: path, atomically: false, encoding: .utf8)
        }
    }
}
