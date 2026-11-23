import Foundation
import Testing
@testable import SmoovPlayKit

@Suite struct JSONTests {
    @Test func keepsOrderTypesAndText() throws {
        let j = try JSON.parse(#"{"b": 1.0, "a": true, "c": [1, "x\né", null], "d": {"z": 0.25}}"#)
        #expect(j.keys == ["b", "a", "c", "d"])
        #expect(j["b"] == .int(1))  // numbers compare as numbers
        #expect(j["b"]!.serialized() == "1.0")  // and keep their text
        #expect(j["a"] == .bool(true))
        #expect(j["a"] != .int(1))  // true isn't 1
        #expect(j["c"]?.array?[1].string == "x\né")
        #expect(j.serialized() == "{\"b\": 1.0, \"a\": true, \"c\": [1, \"x\\n\\u00e9\", null], \"d\": {\"z\": 0.25}}")  // ASCII, as json.dumps
    }

    @Test func prettyIsPythonsIndent2() throws {
        let j = JSON.object([("hz", 90), ("scaler", "fsr"), ("fsr_sharpness", .number(0.5, "0.5")), ("interp", false)])
        #expect(j.pretty() == "{\n  \"hz\": 90,\n  \"scaler\": \"fsr\",\n  \"fsr_sharpness\": 0.5,\n  \"interp\": false\n}")
        #expect(j.setting("hz", 120).keys == ["hz", "scaler", "fsr_sharpness", "interp"])
        #expect(j.setting("new", 1).keys.last == "new")
    }

    @Test func rejectsBadJSON() {
        #expect(throws: JSON.ParseError.self) { try JSON.parse("{\"a\": }") }
        #expect(throws: JSON.ParseError.self) { try JSON.parse("[1, 2") }
        #expect(throws: JSON.ParseError.self) { try JSON.parse("{} extra") }
    }
}

@Suite struct CoreLineTests {
    @Test func linesTheLauncherReads() {
        #expect(CoreLine.parse("[chiaki] waiting for chiaki-ng to start a stream (run it with CHIAKI_FRAMETAP=1)...") == .waitingForChiaki)
        #expect(CoreLine.parse("[tap] waiting for an app to start sending frames...") == .waitingForApp)
        #expect(CoreLine.parse("[chiaki] stream: H.265 1920x1080 at 60 fps") == .found("H.265 1920x1080 at 60 fps"))
        #expect(CoreLine.parse("[tap] frames: BGRA (decoded) 1920x1080, frames when the picture changes")
            == .found("BGRA (decoded) 1920x1080, frames when the picture changes"))
        #expect(CoreLine.parse("[trt] building an engine") == .building(nil))
        #expect(CoreLine.parse("[run] restarting for AI engine: RIFE 4.22.lite scale=0.5") == .building("AI engine"))
        #expect(CoreLine.parse("[run] playing") == .playing)
        #expect(CoreLine.parse("[chiaki] stream ended") == .streamEnded)
        #expect(CoreLine.parse("[tap] the app stopped sending frames") == .streamEnded)
        #expect(CoreLine.parse("[chiaki] WARNING: the stream is set to 30 fps. Set chiaki-ng's frame rate to 60")
            == .note("the stream is set to 30 fps. Set chiaki-ng's frame rate to 60"))
        if case .note = CoreLine.parse("[chiaki] this stream was already running, so its codec header went past") {} else { Issue.record("no note") }
        #expect(CoreLine.parse("[window-keys] {\"keys\": {}}") == .windowKeys("{\"keys\": {}}"))
        #expect(CoreLine.parse("[remote] unknown command: key R") == .unknownCommand("key"))
        #expect(CoreLine.parse("[remote] unknown command: overlay 10 20 640 360") == .unknownCommand("overlay"))
        #expect(CoreLine.parse("[display] upscaling with FSR 1") == .log)
    }

    /// Every line of a real --control session parses, and the ones that matter are recognised.
    @Test func realSession() {
        let lines = Fixture.text("session-lines.txt").split(separator: "\n").map(String.init)
        let events = lines.map(CoreLine.parse)
        #expect(events.contains(.playing))
        #expect(events.contains(.building("settings")))
        #expect(events.filter { $0 == .log }.count < events.count)
    }

    @Test func refusedSetting() {
        #expect(CoreLine.parse("[remote] can't set fill: '\"nope\"' isn't one of its choices")
            == .refused(key: "fill", why: "'\"nope\"' isn't one of its choices"))
        #expect(CoreLine.parse("[remote] can't set nonsense: unknown setting") == .refused(key: "nonsense", why: "unknown setting"))
    }

    @Test func commands() {
        #expect(CoreLine.set("hz", 90) == "set hz 90")
        #expect(CoreLine.set("scaler", "bilinear") == "set scaler \"bilinear\"")
        #expect(CoreLine.set("fsr_sharpness", .number(1.0, "1.0")) == "set fsr_sharpness 1.0")
        #expect(CoreLine.set("interp", false) == "set interp false")
    }

    @Test func stateFromTheCore() throws {
        let s = Fixture.states
        #expect(s.count == 6)
        #expect(s[0].values["hz"] == .int(60))
        #expect(s[0].values["scaler"] == "fsr")
        #expect(s[0].choices["hz"] == [60, 90, 120])
        #expect(s[0].choices["scaler"] == ["fsr", "bilinear"])  // no RTX VSR on a Mac
        #expect(s[0].disabled["engine"] == "On a Mac, RIFE runs through PyTorch on Apple's GPU (Metal).")
        #expect(s[0].restart.isEmpty)
        #expect(s[2].values["motion"] == "half")
        #expect(s[2].restart == ["Motion detail"])
        #expect(s[5].values["interp"] == false)
        #expect(!s[0].menuOpen)
    }

    @Test func coreArguments() {
        var p = AppPaths.detect(env: ["SMOOVPLAY_APP": "/r", "SMOOVPLAY_RUN_ARGS": "--hz 60 --chiaki-tap \"Local\\x y\""])
        var o = CoreSession.Options(source: "window", sourceArgs: Sources.runArgs("window"))
        o.testMode = false
        let a = CoreSession.arguments(options: o, paths: p, results: URL(fileURLWithPath: "/l/results-1.txt"))
        #expect(a == ["--source", "tap", "--tap", "smoovplay-window", "--display", "--control", "--results", "/l/results-1.txt",
                      "--hz", "60", "--chiaki-tap", "Local\\x y"])
        p = AppPaths.detect(env: ["SMOOVPLAY_APP": "/r", "SMOOVPLAY_SETTINGS": "/tmp/s.json"])
        o = CoreSession.Options(source: "chiaki", sourceArgs: Sources.runArgs("chiaki"))
        o.testMode = true
        #expect(CoreSession.arguments(options: o, paths: p, results: URL(fileURLWithPath: "/x"))
            == ["--source", "chiaki", "--display", "--control", "--test-mode", "--results", "/x", "--settings", "/tmp/s.json"])
        #expect(Sources.runArgs("capture", device: "2") == ["--source", "capture", "--capture", "2"])
        #expect(Sources.runArgs("file", file: "/v.mp4") == ["--source", "file", "/v.mp4", "--raw", "--realtime"])
    }

    @Test func coreChoice() {
        let py = AppPaths.detect(env: ["SMOOVPLAY_APP": "/r"])
        #expect(py.coreKind == .python)
        #expect(py.coreCommand.0.path == "/r/.venv/bin/python")
        #expect(py.coreCommand.1 == ["/r/run.py"])
        let rust = AppPaths.detect(env: ["SMOOVPLAY_APP": "/r", "SMOOVPLAY_CORE": "rust"])
        #expect(rust.coreCommand.0.path == "/r/mac/core/target/release/smoovplay")
        #expect(rust.coreCommand.1.isEmpty)
        #expect(rust.coreProblem != nil)  // not built at /r
        let custom = AppPaths.detect(env: ["SMOOVPLAY_APP": "/r"], coreChoice: "/opt/core")
        #expect(custom.coreKind == .custom && custom.coreCommand.0.path == "/opt/core")
        // every Python process gets the looks folder beside the settings
        #expect(py.env["SMOOVPLAY_LOOKS_DIR"]?.hasSuffix("Library/Application Support/framegen/looks") == true)
    }

    @Test func splitArgs() {
        #expect(AppPaths.split(#"--a 1  --b "two words" --c 'x y' """#) == ["--a", "1", "--b", "two words", "--c", "x y", ""])
        #expect(AppPaths.split("") == [])
    }

    @Test func sources() throws {
        let all = try SourceDef.parse(Fixture.text("sources.json"))
        #expect(all.map(\.kind) == ["chiaki", "capture", "window", "tap", "file"])
        let offered = Sources.offered(all, showHidden: false)
        #expect(offered.map(\.kind) == ["chiaki", "capture", "window", "tap"])
        #expect(offered.first(where: { $0.kind == "window" })!.guide.contains("Control-Option"))
        #expect(!(offered.first(where: { $0.kind == "window" })!.testMode))
        #expect(Sources.offered([], showHidden: false).map(\.kind) == ["chiaki"])
    }

    /// A real session of the core, driven through CoreSession: video file, --control, a setting changed live, quit.
    @Test(.enabled(if: Fixture.hasPython && ProcessInfo.processInfo.environment["SMOOVPLAY_TEST_DISPLAY"] == "1"))
    func liveSession() async throws {
        let logs = FileManager.default.temporaryDirectory.appendingPathComponent("sp-test-logs-\(getpid())")
        let paths = AppPaths.detect(env: ProcessInfo.processInfo.environment.merging(
            ["SMOOVPLAY_APP": Fixture.repo.path, "SMOOVPLAY_LOGS": logs.path, "SMOOVPLAY_RUN_ARGS": "--settings none"]) { $1 })
        let s = CoreSession(paths: paths, options: CoreSession.Options(
            source: "file", sourceArgs: ["--source", "file", Fixture.repo.appendingPathComponent("testdata/clip_1080p60.mp4").path, "--raw", "--realtime"]))
        let q = DispatchQueue(label: "t")
        s.queue = DispatchQueue(label: "core")
        var states: [PlayState] = []
        var hz: [JSON] = []
        s.onState = { st in q.sync { states.append(st) } }
        s.onEvent = { ev in
            if case let .state(j) = ev, let st = try? SettingsState(json: j) {
                q.sync { hz.append(st.values["hz"] ?? .null) }
                if st.values["hz"] == .int(60) { s.send("set hz 90") } else { s.send("quit") }
            }
        }
        try s.start(header: "[launcher] test")
        for _ in 0..<600 where !(q.sync { states.contains(where: \.over) }) { try await Task.sleep(nanoseconds: 100_000_000) }
        #expect(q.sync { states.contains(.playing) })
        #expect(q.sync { hz.contains(.int(90)) })
        #expect(q.sync { states.last } == .stopped)
        let log = try String(contentsOf: s.logURL, encoding: .utf8)
        #expect(log.contains("[launcher] settings: set hz 90"))
        #expect(log.contains("[launcher] the core exited with code 0"))
        try? FileManager.default.removeItem(at: logs)
    }
}
