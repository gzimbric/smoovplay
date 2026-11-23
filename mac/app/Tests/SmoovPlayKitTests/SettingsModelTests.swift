import Foundation
import Testing
@testable import SmoovPlayKit

/// The Settings page model built from python -m framegen.settings and run.py --control's real state lines.
@Suite struct SettingsModelTests {
    let specs = Fixture.specs
    let states = Fixture.states

    func model(_ state: SettingsState?, file: JSON = .object([]), source: String = "chiaki", running: Bool = false) -> SettingsModel {
        SettingsModel(specs: specs, state: state, file: file, source: source, sessionRunning: running || state != nil)
    }

    func row(_ p: SettingsPage, _ key: String) -> RowModel {
        (p.quick + p.sections.flatMap(\.rows)).first(where: { $0.key == key })!
    }

    @Test func specsParse() {
        #expect(specs.specs.count > 25)
        #expect(specs.impactPC == "an RTX 4070 Ti SUPER")
        let hz = specs["hz"]!
        #expect(hz.choices.map(\.value) == [60, 90, 120])
        #expect(hz.quick && hz.live && hz.saved)
        #expect(specs["fsr_sharpness"]!.appliesKey == "scaler")
        #expect(specs["capture_size"]!.sources == ["capture"])
        #expect(specs["simulate_30"]!.saved == false)
    }

    @Test func liveLayout() {
        let p = model(states[0]).page()
        #expect(p.live && !p.locked)
        #expect(p.quick.map(\.key) == ["hz", "model", "scaler"])  // Frame generation is the rate pick's Off
        let rate = p.quick[0]
        #expect(rate.isRate && rate.label == "Frame rate" && rate.kind == .segments)
        #expect(rate.choices.map(\.tag) == ["off", "60", "90", "120"])
        #expect(rate.choices.filter(\.selected).map(\.tag) == ["60"])
        #expect(rate.hint.hasPrefix("One new frame for each real one"))
        #expect(p.sections.map(\.group) == ["Smoothness", "Picture", "Looks", "Display", "Instant replay", "PS5 Remote Play"])
        // what this Mac can't use is greyed out, with the core's reasons
        let scaler = row(p, "scaler")
        #expect(scaler.choices.first(where: { $0.tag == "\"vsr\"" })!.enabled == false)
        #expect(scaler.choices.first(where: { $0.tag == "\"bilinear\"" })!.enabled)
        #expect(row(p, "model").note == "NVIDIA VFG needs an NVIDIA RTX card.")
        #expect(row(p, "engine").note == "On a Mac, RIFE runs through PyTorch on Apple's GPU (Metal).")
        #expect(!row(p, "engine").enabled)
        // the app hides chiaki-ng on a Mac, so the core's "Not on a Mac yet" becomes the app's own word
        #expect(row(p, "minimize_chiaki").note == SettingsModel.appOwned["minimize_chiaki"])
        #expect(p.status.hasPrefix("Changes reach the video straight away"))
        #expect(p.restartBanner == nil)
    }

    @Test func appliesHidesRows() {
        let fsr = model(states[0]).page()
        #expect(row(fsr, "fsr_sharpness").visible)
        #expect(!row(fsr, "vsr_quality").visible)
        let bilinear = model(states[3]).page()
        #expect(!row(bilinear, "fsr_sharpness").visible)
        #expect(!row(model(states[0]).page(), "vfg_quality").visible)  // only with NVIDIA VFG
    }

    @Test func restartOnlyChanges() {
        let p = model(states[2]).page()
        #expect(p.restartBanner?.hasPrefix("Restart to apply: Motion detail.") == true)
        #expect(row(p, "motion").restartTag == "after a restart")
        #expect(row(p, "hz").restartTag == nil)  // live
        #expect(p.sections.first(where: { $0.group == "Smoothness" })!.summary.contains("Motion detail: Half"))
        var m = model(states[2])
        m.restartSent = true
        #expect(m.page().restartBanner == nil)
    }

    @Test func frameGenerationOff() {
        let p = model(states[5]).page()
        let rate = p.quick[0]
        #expect(rate.choices.filter(\.selected).map(\.tag) == ["off"])
        #expect(rate.hint.hasPrefix("No new frames"))
        #expect(!row(p, "even_out").visible)  // applies only with frame generation on
    }

    @Test func picks() {
        let off = model(states[5])  // frame generation off, 90 fps
        let back = off.pick("hz", tag: "90")
        #expect(back.map(\.0) == ["hz", "interp"])
        #expect(back.map(\.1) == [90, true])
        #expect(model(states[0]).pick("hz", tag: "120").map(\.0) == ["hz"])
        #expect(model(states[0]).pick("hz", tag: "off").map(\.0) == ["interp"])
        #expect(model(states[0]).pick("hz", tag: "off").map(\.1) == [false])
        #expect(model(states[0]).pick("scaler", tag: "\"bilinear\"").map(\.1) == ["bilinear"])
        #expect(model(states[0]).pick("fsr_sharpness", tag: "1.0").map(\.1) == [.number(1.0, "1.0")])
        #expect(model(states[0]).pick("nope", tag: "1").isEmpty)
    }

    @Test func nothingPlayingReadsTheFile() {
        let file = JSON.object([("hz", 120), ("scaler", "bilinear"), ("fill", "nope"), ("look_on", true)])
        let m = model(nil, file: file)
        let p = m.page()
        #expect(!p.live && !p.locked)
        #expect(p.quick[0].choices.filter(\.selected).map(\.tag) == ["120"])
        #expect(m.value("fill") == "fit")  // not one of its choices, so the default
        #expect(m.value("look_on") == true)
        #expect(row(p, "scaler").choices.first(where: { $0.tag == "\"vsr\"" })!.enabled == false)  // a Mac
        #expect(row(p, "model").choices.first(where: { $0.tag == "\"vfg\"" })!.enabled == false)
        #expect(row(p, "minimize_chiaki").enabled)  // the app does it, so it can change with nothing playing
        #expect(row(p, "capture_size").note != nil)  // only with a capture card
        #expect(row(model(nil, source: "capture").page(), "capture_size").note == nil)
        #expect(p.status.hasPrefix("Saved for every session"))
        #expect(p.sections.first(where: { $0.group == "Looks" })!.summary == "Look on")
    }

    @Test func startingLocksTheRows() {
        let p = model(nil, running: true).page()
        #expect(p.locked)
        #expect((p.quick + p.sections.flatMap(\.rows)).allSatisfy { !$0.enabled })
        #expect(p.status.hasPrefix("SmoovPlay is starting"))
    }

    @Test func settingsFileKeepsTheRest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sp-settings-\(getpid())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = SettingsFile(dir.appendingPathComponent("settings.json"))
        try "{\n  \"hz\": 90,\n  \"look\": \"film\"\n}".write(to: f.url, atomically: true, encoding: .utf8)
        try f.write("scaler", "bilinear")
        try f.write("hz", 120)
        #expect(try String(contentsOf: f.url, encoding: .utf8) == "{\n  \"hz\": 120,\n  \"look\": \"film\",\n  \"scaler\": \"bilinear\"\n}")
        #expect(f.text("scaler") == "bilinear")
        try FileManager.default.removeItem(at: dir)
        #expect(SettingsFile(dir.appendingPathComponent("none.json")).read() == .object([]))
    }

    /// The file the app writes is one the pipeline reads (framegen.settings.Settings.load), with the values it set.
    @Test(.enabled(if: Fixture.hasPython)) func pythonReadsWhatTheAppWrites() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sp-settings-py-\(getpid())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let f = SettingsFile(dir.appendingPathComponent("settings.json"))
        let m = model(nil)
        for (k, v) in m.pick("hz", tag: "90") + m.pick("fsr_sharpness", tag: "1.0") + [("look_on", JSON.bool(true))] { try f.write(k, v) }
        let (rc, out) = run(Fixture.python, ["-c", "import json,sys; from framegen.settings import Settings; s=Settings.load(sys.argv[1]); print(json.dumps([s.hz, s.fsr_sharpness, s.look_on, s.scaler]))", f.url.path], cwd: Fixture.repo)
        #expect(rc == 0)
        #expect(out.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n").last == "[90, 1.0, true, \"fsr\"]")
        try? FileManager.default.removeItem(at: dir)
    }

    @Test func autodetect() throws {
        let r = try DetectResult(json: Fixture.text("detect-result.json"))
        #expect(r.settings.map(\.0) == ["hz", "motion", "engine", "scaler"])
        #expect(r.found.first?.label == "Graphics card")
        #expect(r.reasons["hz"]?.contains("60 fps leaves it enough time") == true)
        let m = model(nil, file: .object([("hz", 120)]))
        let changes = r.changes(m)
        #expect(changes.map(\.0.key) == ["hz"])
        #expect(changes.first?.1 == 60)
        #expect(r.changes(model(nil)).isEmpty)  // the defaults already match
        #expect(AutoDetect.progress("[detect] timing the AI model") == "Timing the AI model")
        #expect(AutoDetect.progress("[detect] ERROR: no GPU") == nil)
    }

    @Test func testResults() throws {
        let r = try #require(TestResults(text: Fixture.text("results.txt")))
        #expect(r.verdict == "This PC is falling behind at 60 fps. Try Motion detail: Half.")
        #expect(r.tiles.map(\.label) == ["With AI", "Without AI", "AI frame time"])
        #expect(r.tiles.map(\.value) == ["3.4%", "5.0%", "15.8 ms"])
        #expect(r.note.hasPrefix("60 fps output, 0:12 played. Visibly off"))
        #expect(!r.privateResults)
        #expect(TestResults(text: "  \n") == nil)
        #expect(TestResults(text: "SmoovPlay test results\nStream: 1920x1080, played 0:03")?.verdict.hasPrefix("Not enough was played") == true)
    }
}
