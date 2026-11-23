import Foundation
@testable import SmoovPlayKit

/// Fixtures were captured from the real core (run.py --control, python -m framegen.settings, framegen.autodetect
/// --timing, framegen.looks schema, run.py --test-mode --results), with the home folder replaced by /Users/someone.
enum Fixture {
    static func url(_ name: String) -> URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures").appendingPathComponent(name)
    }

    static func text(_ name: String) -> String { try! String(contentsOf: url(name), encoding: .utf8) }

    static var specs: SettingSpecs { try! SettingSpecs(json: text("specs.json")) }

    /// The captured "[settings-state]" lines, in order: 0 a session's first; 1 after set hz 90, scaler bilinear,
    /// look_on true and a restart; 2 to 5 another session after set motion "half" (restart-only), then scaler
    /// bilinear, hz 90, interp false.
    static var states: [SettingsState] {
        text("states.txt").split(separator: "\n").map { line in
            guard case let .state(json) = CoreLine.parse(String(line)) else { fatalError("not a state line") }
            return try! SettingsState(json: json)
        }
    }

    /// The repo (five folders up from this file: mac/app/Tests/SmoovPlayKitTests/).
    static var repo: URL {
        var u = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { u.deleteLastPathComponent() }
        return u
    }

    static var python: URL { repo.appendingPathComponent(".venv/bin/python") }
    static var hasPython: Bool { FileManager.default.isExecutableFile(atPath: python.path) }
}

/// Runs a command and returns (exit code, stdout + stderr).
func run(_ exe: URL, _ args: [String], cwd: URL? = nil, env: [String: String]? = nil, timeout: TimeInterval = 120) -> (Int32, String) {
    let p = Process()
    p.executableURL = exe
    p.arguments = args
    if let cwd { p.currentDirectoryURL = cwd }
    if let env { p.environment = env }
    let out = Pipe()
    p.standardOutput = out
    p.standardError = out
    var data = Data()
    let lock = NSLock()
    out.fileHandleForReading.readabilityHandler = { h in
        let d = h.availableData
        lock.lock(); data.append(d); lock.unlock()
    }
    try! p.run()
    let end = Date().addingTimeInterval(timeout)
    while p.isRunning && Date() < end { usleep(20_000) }
    if p.isRunning { p.terminate(); p.waitUntilExit() }
    out.fileHandleForReading.readabilityHandler = nil
    lock.lock(); data.append(out.fileHandleForReading.readDataToEndOfFile()); lock.unlock()
    return (p.terminationStatus, String(decoding: data, as: UTF8.self))
}
