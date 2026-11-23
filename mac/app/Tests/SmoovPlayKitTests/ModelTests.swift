import Carbon.HIToolbox
import Foundation
import Testing
@testable import SmoovPlayKit

@Suite struct LooksTests {
    @Test func schemaAndEditing() throws {
        var lib = try LooksLibrary(json: Fixture.text("looks-schema.json"))
        #expect(lib.types.count >= 16)
        #expect(lib.types.first?.key == "color" && lib.types.first?.params.first?.key == "mix")
        #expect(lib.entries.filter(\.builtin).count == 14)
        #expect(lib.samples.count == 8)
        let i = lib.entries.firstIndex(where: { $0.id == "cinematic" })!
        let was = lib.entries[i].look
        let contrast = lib.entries[i].nodes[0]["contrast"]!
        lib.setValue(i, node: 0, key: "contrast", .num(0.5))
        #expect(lib.changed("cinematic"))
        lib.setValue(i, node: 0, key: "contrast", contrast)  // every change put back, so it's SmoovPlay's look again
        #expect(!lib.changed("cinematic"))
        #expect(lib.entries[i].look == was)
        lib.removeNode(i, node: 0)
        lib.moveNode(i, node: 0, by: 1)
        #expect(lib.changed("cinematic"))
        let id = lib.saveAsMine(i)
        #expect(id == "my:my-cinematic")
        #expect(!lib.changed("cinematic"))
        #expect(lib.entries.first(where: { $0.id == "cinematic" })!.look == was)
        let mine = lib.entry(id)!
        #expect(mine.name == "My Cinematic" && !mine.builtin && mine.nodes.count == was["nodes"]!.array!.count - 1)
        #expect(lib.entries.last?.id == id)  // yours after SmoovPlay's
        let id2 = lib.newLook()
        #expect(lib.entry(id2)?.nodes.first?["type"] == "color")
        #expect(lib.uniqueName("My look") == "My look 2")
        let file = LooksLibrary.fileJSON(mine)
        #expect(file.keys == ["version", "name", "nodes"])
    }

    @Test func params() {
        let p = LookParam(key: "exposure", label: "Exposure", unit: "stops", min: -3, max: 3, step: 0.05, defaultValue: 0, text: false, names: [])
        #expect(p.format(0.5) == "+0.50 stops")  // as LooksPage.cs Format writes it
        #expect(p.snap(0.52) == 0.5)
        #expect(LookParam(key: "mix", label: "Amount", unit: "", min: 0, max: 1, step: 0.01, defaultValue: 1, text: false, names: []).format(0.25) == "25%")
        #expect(LookParam(key: "hue", label: "Hue", unit: "°", min: 0, max: 360, step: 1, defaultValue: 0, text: false, names: []).format(30) == "30°")
    }

    @Test func shareCodes() {
        #expect(LooksLibrary.code(in: "try this: SPL1-AbC_d-ef12 it's nice") == "SPL1-AbC_d-ef12")
        #expect(LooksLibrary.code(in: "SPL1-AbCdef\n  ghIJ-kl thanks!") == "SPL1-AbCdefghIJ-kl")
        #expect(LooksLibrary.code(in: "no code here") == nil)
    }

    /// A code the app copies is one the pipeline decodes back to the same look (python -m framegen.looks).
    @Test(.enabled(if: Fixture.hasPython)) func codesRoundTrip() throws {
        let lib = try LooksLibrary(json: Fixture.text("looks-schema.json"))
        let code = try #require(lib.entry("film")?.code)
        let (rc, out) = run(Fixture.python, ["-m", "framegen.looks", "decode", code], cwd: Fixture.repo)
        #expect(rc == 0)
        let j = try JSON.parse(String(out.split(separator: "\n").last(where: { $0.hasPrefix("{") }) ?? ""))
        #expect(j["look"]?["name"] == "Film")
        #expect(j["look"]?["nodes"]?.array?.count == lib.entry("film")?.nodes.count)
    }
}

@Suite struct KeysTests {
    let ctrlOpt = UInt32(controlKey | optionKey)

    @Test func combos() throws {
        let m = try KeyCombo.parse("Control-Option-M")
        #expect(m.mods == ctrlOpt && m.key == "M" && m.keyCode == UInt32(kVK_ANSI_M))
        #expect(try KeyCombo.parse("ctrl+alt+m") == m)
        #expect(try KeyCombo.parse("Ctrl+Opt+Escape").name == "Control-Option-Esc")
        #expect(try KeyCombo.parse("cmd-shift-f5").name == "Shift-Command-F5")
        #expect(throws: KeyCombo.Problem.self) { try KeyCombo.parse("Control-Option") }
        #expect(throws: KeyCombo.Problem.self) { try KeyCombo.parse("Control-M-N") }
        #expect(throws: KeyCombo.Problem.self) { try KeyCombo.parse("Hyper-M") }
        #expect(KeyCombo.from(keyCode: UInt16(kVK_ANSI_R), mods: ctrlOpt)?.name == "Control-Option-R")
        #expect(KeyCombo.from(keyCode: 0xFF, mods: 0) == nil)
    }

    @Test func windowKeys() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sp-prefs-\(getpid()).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        let prefs = Prefs(url: url)
        let all = WindowKeys.resolve(prefs, free: { _ in true })
        #expect(all.got["menu"] == .some("Control-Option-M") && all.got["quit"] == .some("Control-Option-Esc") && all.taken.isEmpty)
        let menuTaken = WindowKeys.resolve(prefs, free: { $0.name != "Control-Option-M" })
        #expect(menuTaken.got["menu"] == .some("Control-Option-Space") && menuTaken.taken == ["Control-Option-M"])
        WindowKeys.pick(prefs, "menu", "Control-Option-K")
        #expect(WindowKeys.picked(Prefs(url: url)) == ["menu": "Control-Option-K"])  // saved
        #expect(WindowKeys.resolve(prefs, free: { _ in true }).got["menu"] == .some("Control-Option-K"))
        #expect(WindowKeys.problem(try KeyCombo.parse("Control-Option-R"), action: "menu", prefs: prefs, free: { _ in true })?.contains("saving a replay") == true)
        #expect(WindowKeys.problem(KeyCombo(mods: 0, key: "M"), action: "menu", prefs: prefs, free: { _ in true })?.hasPrefix("Add Control") == true)
        #expect(WindowKeys.problem(try KeyCombo.parse("Control-Option-F"), action: "menu", prefs: prefs, free: { _ in true }) == "Fullscreen already uses Control-Option-F. Change that one first.")
        #expect(WindowKeys.problem(try KeyCombo.parse("Control-Option-J"), action: "menu", prefs: prefs, free: { _ in false })?.contains("already uses") == true)
        #expect(WindowKeys.problem(try KeyCombo.parse("Control-Option-J"), action: "menu", prefs: prefs, free: { _ in true }) == nil)
        WindowKeys.resetAll(prefs)
        #expect(WindowKeys.picked(prefs).isEmpty)
        #expect(WindowKeys.sentence(all.got) == "Control-Option-M opens the menu, Control-Option-F goes fullscreen and Control-Option-Esc quits")
    }
}

@Suite struct DiagnosticsTests {
    @Test func leavesOutTheUser() {
        let text = """
        Repo: /Users/someone/dev/smoovplay
        "path": "/Users/someone/Library/Application Support/framegen/settings.json"
        /Users/someoneelse/x and /Users/Shared/clips stay apart
        Card: Alex's iPhone Camera
        """
        let a = Diagnostics.anonymize(text, home: "/Users/someone")
        #expect(a.contains("Repo: ~/dev/smoovplay"))
        #expect(a.contains("\"path\": \"~/Library/Application Support/framegen/settings.json\""))
        #expect(a.contains("~/x and /Users/Shared/clips"))
        #expect(!a.contains("someone"))
        #expect(a.contains("(name)'s iPhone Camera"))
    }

    @Test func prefs() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sp-prefs2-\(getpid()).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Prefs(url: url)
        p.set("guide:psn", true)
        p.choose("source:", "window")
        p.choose("source:", "tap")
        let again = Prefs(url: url)
        #expect(again.has("guide:psn") && again.value("source:") == "tap" && again.flags.count == 2)
    }
}
