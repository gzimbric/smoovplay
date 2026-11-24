// The app's state and what it does (installer/launcher/Ui.cs on Windows): pages, play sessions, settings, looks,
// Auto-detect, window capture, keys from any app, and chiaki-ng.
import AppKit
import Carbon.HIToolbox
import SmoovPlayKit
import SwiftUI

enum Page: String, CaseIterable, Identifiable {
    case play, guide, check, settings, looks, help
    var id: String { rawValue }
}

struct Toast: Equatable {
    var text: String
    var bad: Bool
    var id = UUID()
}

struct Dialog: Identifiable {
    let id = UUID()
    var title: String
    var text: String
    var buttons: [String]  // the first is the default and the last cancels
    var done: (Int) -> Void
}

@MainActor
final class AppModel: ObservableObject {
    nonisolated static let version = (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.2.0-dev"

    @Published var paths: AppPaths
    let prefs: Prefs
    let options: LaunchOptions
    lazy var chiaki = Chiaki(app: paths.chiakiApp)
    let awake = Awake()
    let capture = WindowCapture()

    @Published var page: Page = .play
    @Published var toast: Toast?
    @Published var dialog: Dialog?

    // sources
    @Published var allSources: [SourceDef] = [.remotePlay]
    @Published var source: SourceDef = .remotePlay
    var sources: [SourceDef] { Sources.offered(allSources, showHidden: ProcessInfo.processInfo.environment["SMOOVPLAY_SOURCES"] == "all") }

    // play
    @Published var playState: PlayState = .idle
    @Published var playLog: [String] = []
    @Published var note: String?
    @Published var results: TestResults?
    @Published var showDetails = false
    @Published var overlayNote: String?
    private(set) var session: CoreSession?
    private var minimisedForPlay = false
    private var keysSaid: Set<String> = []

    // settings
    @Published var specs: SettingSpecs?
    @Published var specsError: String?
    @Published var liveState: SettingsState?
    @Published var fileValues: JSON = .object([])
    @Published var restartSent = false
    @Published var openSections: Set<String> = []
    private var loadingSpecs = false
    var settingsFile: SettingsFile { SettingsFile(paths.settingsFile) }

    // Auto-detect
    @Published var detecting = false
    @Published var detectProgress = ""
    @Published var detectResult: DetectResult?
    @Published var detectError: String?
    @Published var detectApplied = false
    @Published var detectChanged: Set<String> = []

    // looks
    @Published var looks: LooksLibrary?
    @Published var looksError: String?
    @Published var lookPicked: String?
    @Published var previewImage: NSImage?
    @Published var previewFor: String?
    @Published var previewHolding = false
    @Published var lookHidden = false
    @Published var lookOpen: [String: Set<Int>] = [:]
    @Published var samples: [LookSample] = []
    lazy var preview: LookPreview = makePreview()
    private var previewMade = false

    /// The preview runs the core's helper, so it's made again when the core choice changes (coreChoice.didSet).
    private func makePreview() -> LookPreview {
        previewMade = true
        let p = LookPreview(paths: paths)
        p.onPicture = { [weak self] url, sample in
            Task { @MainActor in
                // read it now, because the file is written again two pictures later
                guard let self, let d = try? Data(contentsOf: url), let img = NSImage(data: d) else { return }
                let first = self.previewImage == nil
                self.previewImage = img
                self.previewFor = sample
                if first { Snapshots.later("look-preview", ms: 300) }
            }
        }
        p.trace = { Trace.log("look preview: " + $0) }
        p.onSamples = { [weak self] s in Task { @MainActor in if self?.samples.isEmpty == true { self?.samples = s } } }
        return p
    }
    private var lookSendDue = false, lookTimer: Timer?, lookSaveDue: String?, lookSaveAt = Date()
    private weak var lookShownTo: CoreSession?
    var previewPixels = (640, 360)

    // window capture
    @Published var windows: [CaptureWindow] = []
    @Published var windowsLoaded = false
    @Published var windowsError: String?
    @Published var screenGranted = ScreenPermission.granted
    private var overlayTimer: Timer?
    private var overlayAt: String?

    // capture cards
    @Published var devices: [CaptureDevice] = []
    @Published var devicesLoaded = false
    @Published var devicesError: String?

    // keys
    @Published var keysGot: [String: String?] = [:]
    @Published var keysTaken: [String] = []
    @Published var keysEditing = false
    @Published var keyListening: String?
    @Published var keysProblem: String?
    @Published var heldKeys: [String] = []  // what's held while playing, for the Play page
    private var replayKeysTried = false  // takeKeys() went for the replay keys (held or not), so no retry per state line
    private var captureStarts = 0  // counts window-capture starts, so a Stop during one cancels it
    private var keyMonitor: Any?
    private var navKeys: [UInt32] = []  // the menu's own keys, held while it's open (App window mode)

    // the Mac check
    @Published var checks: [Check] = []
    @Published var checking = false

    // the core choice (Help page): python, rust, or a path
    @Published var coreChoice: String {
        didSet {
            UserDefaults.standard.set(coreChoice, forKey: "core")
            paths = AppPaths.detect(coreChoice: coreChoice)
            // chiaki and preview were made from the old paths: the preview runs the old core's helper otherwise
            chiaki.app = paths.chiakiApp
            if previewMade {
                preview.stop()
                preview = makePreview()
            }
        }
    }

    init(options: LaunchOptions) {
        self.options = options
        // With nothing picked on the Help page yet, AppPaths.detect picks the Rust core once it's built, else run.py
        let choice = UserDefaults.standard.string(forKey: "core") ?? AppPaths.detect().coreKind.rawValue
        coreChoice = choice
        paths = AppPaths.detect(coreChoice: choice)
        prefs = Prefs(url: AppPaths.detect(coreChoice: choice).prefsFile)
        openSections = Set(prefs.flags.filter { $0.hasPrefix("set-open:") }.map { String($0.dropFirst("set-open:".count)) })
        fileValues = settingsFile.read()
        if let p = options.page.flatMap(Page.init(rawValue:)) { page = p }
        else if options.play { page = .play }
        else { page = prefs.has("guide-done") ? .play : .guide }  // the PS5 guide on the first run
        chiaki.log = { [weak self] l in self?.addLog(l); self?.session?.write(l) }
        capture.log = { [weak self] l in Task { @MainActor in self?.addLog(l); self?.session?.write(l) } }
        capture.onStop = { [weak self] why in
            Task { @MainActor in
                guard let self, self.session?.running == true, let why else { return }
                self.showToast(why, bad: true)
            }
        }
        loadSources()
        loadSpecs()
        if let r = options.showResults { showResults(URL(fileURLWithPath: r)) }
        if options.detect { DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.runDetect() } }
    }

    // MARK: - pages
    func go(_ p: Page) {
        if page == .looks && p != .looks { onLooksLeave() }
        page = p
        switch p {
        case .settings: loadSpecs(force: specsStale); specsStale = false; fileValues = settingsFile.read()
        case .looks: if looks == nil { loadLooks() }; previewPicked(); requestPreview()
        case .check: runChecks()
        case .play: refreshSourceExtras()
        default: break
        }
        Snapshots.later("page-" + p.rawValue)
    }

    func showToast(_ text: String, bad: Bool = false) {
        let t = Toast(text: text, bad: bad)
        toast = t
        DispatchQueue.main.asyncAfter(deadline: .now() + (bad ? 6 : 3.5)) { [weak self] in
            if self?.toast?.id == t.id { self?.toast = nil }
        }
    }

    func ask(_ title: String, _ text: String, _ buttons: [String], _ done: @escaping (Int) -> Void) {
        dialog = Dialog(title: title, text: text, buttons: buttons, done: done)
    }

    /// A launcher line, shown on the Play page and written to the session's log.
    func launcherLog(_ line: String) {
        addLog(line)
        session?.write(line)
    }

    func addLog(_ line: String) {
        playLog.append(line)
        if playLog.count > 600 { playLog.removeFirst(playLog.count - 500) }
    }

    // MARK: - sources
    func loadSources() {
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.sources", ["--json"], timeout: 30)
            let list = r.json.flatMap { try? SourceDef.parse($0) }
            await MainActor.run {
                if let list { self.allSources = list }
                let chosen = self.options.source ?? self.prefs.value("source:")
                self.source = self.sources.first(where: { $0.kind == chosen }) ?? self.sources.first(where: \.remotePlay) ?? .remotePlay
                self.refreshSourceExtras()
                if self.options.play { self.autoPlay() }
            }
        }
    }

    func pickSource(_ s: SourceDef) {
        if session?.running == true { showToast("Stop playing first, then choose where the video comes from", bad: true); return }
        source = s
        prefs.choose("source:", s.kind == "chiaki" ? nil : s.kind)
        refreshSourceExtras()
        if page == .check { runChecks() }
        Snapshots.later("source-" + s.kind)
    }

    func refreshSourceExtras() {
        if source.kind == "window" {
            screenGranted = ScreenPermission.granted
            if !windowsLoaded { loadWindows() }
            resolveKeys()
        }
        if source.kind == "capture", !devicesLoaded { loadDevices() }
    }

    func loadWindows() {
        windowsLoaded = false
        windowsError = nil
        screenGranted = ScreenPermission.granted
        guard screenGranted else {
            windowsLoaded = true
            windowsError = ScreenPermission.deniedText
            return
        }
        Task {
            do {
                let list = try await WindowCapture.windows()
                self.windows = list
                self.windowsError = list.isEmpty
                    ? "No app window found. Open the app you stream with (PS Remote Play, Moonlight or a browser), then press Look again." : nil
            } catch {
                self.windows = []
                self.windowsError = ScreenPermission.granted ? "Couldn't list the windows: \(error.localizedDescription)" : ScreenPermission.deniedText
            }
            self.windowsLoaded = true
        }
    }

    /// The window picked on the Play page. That's the same window if it's still there, else the same app's biggest.
    var chosenWindow: CaptureWindow? {
        if let id = prefs.value("window-id:").flatMap(UInt32.init), let w = windows.first(where: { $0.id == id }) { return w }
        if let app = prefs.value("window-app:"), let w = windows.first(where: { $0.bundleID == app }) { return w }
        return windows.first
    }

    func chooseWindow(_ id: CGWindowID) {
        guard let w = windows.first(where: { $0.id == id }) else { return }
        prefs.choose("window-id:", String(w.id))
        prefs.choose("window-app:", w.bundleID)
        objectWillChange.send()
    }

    func askScreenPermission() {
        if ScreenPermission.request() {
            screenGranted = true
            loadWindows()
        } else {
            screenGranted = false
            ScreenPermission.openSettings()
            showToast("Turn on SmoovPlay in System Settings, then quit and open SmoovPlay again", bad: true)
        }
    }

    func loadDevices() {
        devicesLoaded = false
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.capture", ["--list", "--json"], timeout: 30)
            let list = r.json.flatMap { try? CaptureDevice.parse($0) }
            await MainActor.run {
                self.devices = list ?? []
                self.devicesError = list == nil ? (r.lastText ?? "exit code \(r.code)") : nil
                self.devicesLoaded = true
            }
        }
    }

    var chosenDevice: String { prefs.value("capture-video:") ?? devices.first?.index ?? "0" }

    // MARK: - play
    var running: Bool { session?.running ?? false }

    func playPressed() {
        if playState.running || running { stop(); return }
        play(test: false)
    }

    func play(test: Bool) {
        if let why = paths.coreProblem { showToast(why, bad: true); return }
        if detecting { showToast("Auto-detect is still testing the GPU. A few more seconds", bad: true); return }
        switch source.kind {
        case "chiaki":
            let others = chiaki.others
            if others.isEmpty { start(test: test); return }
            ask("chiaki-ng is already open",
                "SmoovPlay didn't start that copy, so it may not be passing the video on. Close it so SmoovPlay can open it again? A stream running in it ends.",
                ["Close it and play", "Play anyway", "Cancel"]) { r in
                if r == 0 { self.chiaki.close(others) { self.start(test: test) } }
                else if r == 1 { self.start(test: test) }
            }
        case "window":
            screenGranted = ScreenPermission.granted
            if !screenGranted {
                if ScreenPermission.request() { screenGranted = true; loadWindows(); return }
                ask("Screen Recording is off for SmoovPlay", ScreenPermission.deniedText, ["Open System Settings", "Cancel"]) { r in
                    if r == 0 { ScreenPermission.openSettings() }
                }
                return
            }
            guard let w = chosenWindow else {
                showToast("Pick the app's window first (open the app, then press Look again)", bad: true)
                loadWindows()
                return
            }
            setPlayState(.starting)
            captureStarts += 1
            let thisStart = captureStarts
            Task {
                do {
                    try await capture.start(w, leaveOutTitleBar: !prefs.has("window-keep-title"))
                    // Stop pressed, or Play again, while the capture was starting: this start is off
                    guard thisStart == captureStarts, playState == .starting else {
                        await capture.stop()
                        return
                    }
                    start(test: false)
                } catch {
                    if thisStart == captureStarts { setPlayState(.error("Couldn't copy the window: \(error.localizedDescription)")) }
                }
            }
        default:
            start(test: test)
        }
    }

    private func start(test: Bool) {
        playLog = []
        note = nil
        overlayNote = nil
        results = nil
        keysSaid = []
        var opts = CoreSession.Options(source: source.kind,
                                       sourceArgs: Sources.runArgs(source.kind, device: chosenDevice,
                                                                   file: ProcessInfo.processInfo.environment["SMOOVPLAY_SOURCE_FILE"],
                                                                   tap: ProcessInfo.processInfo.environment["SMOOVPLAY_TAP"]))
        opts.testMode = test
        let s = CoreSession(paths: paths, options: opts)
        session = s
        liveState = nil
        restartSent = false
        minimisedForPlay = false
        s.onLine = { [weak self] l in self?.addLog(l) }
        s.onEvent = { [weak self, weak s] ev in
            guard let self, let s, self.session === s else { return }
            self.coreEvent(ev, s)
        }
        s.onState = { [weak self, weak s] st in
            guard let self, let s, self.session === s else { return }
            self.setPlayState(st)
        }
        let header = "[launcher] SmoovPlay for Mac \(AppModel.version), \(Date().formatted(.iso8601)), \(MacInfo.chip), \(MacInfo.memoryGB) GB"
            + (source.kind != "chiaki" ? ", source \(source.kind)" : "") + (test ? ", test mode (60 fps game played as 30 fps)" : "")
            + ", core \(paths.coreName)"
        do {
            try s.start(header: header)
            if let w = capture.window, source.kind == "window" {
                s.write("[launcher] App window: \(w.app), capture \(capture.size.0)x\(capture.size.1) into /\(Sources.windowTap)")
            }
        } catch {
            setPlayState(.error("Couldn't start: \(error.localizedDescription)"))
            Task { await capture.stop() }
        }
    }

    func stop() {
        guard let s = session, s.running else {
            captureStarts += 1  // a window capture still starting won't start the core when it's done
            if playState.running { setPlayState(.stopped) }
            Task { await capture.stop() }
            return
        }
        s.stop()
    }

    private func coreEvent(_ ev: CoreEvent, _ s: CoreSession) {
        switch ev {
        case let .state(json):
            guard let st = try? SettingsState(json: json) else { return }
            liveState = st
            restartSent = false
            if page == .settings { Snapshots.later("settings-state", ms: 400) }
            afterStateChange()
            onLooksState()
        case .waitingForChiaki:
            if source.kind == "chiaki", !chiaki.oursRunning {
                chiaki.open(env: paths.env) { [weak self] err in if let err { self?.showToast(err, bad: true); self?.addLog("[launcher] " + err) } }
            }
        case let .note(n): note = n
        case let .unknownCommand(c):
            if c == "overlay" {
                overlayNote = "This core can't sit its picture over the app's window yet, so the smooth video is in its own window."
                stopOverlay()
            } else if c == "key", !heldKeys.isEmpty, !keysSaid.contains("key") {
                keysSaid.insert("key")
                launcherLog("[launcher] this core doesn't take keys yet: fullscreen, stats and quit use settings commands, the others wait for it")
            }
        case let .refused(key, why):
            if key != "minimize_chiaki" { showToast("Can't change that now: \(why)", bad: true) }
        default: break
        }
    }

    func setPlayState(_ st: PlayState) {
        let changed = st != playState
        playState = st
        if st == .playing, !minimisedForPlay, session != nil {
            minimisedForPlay = true
            startedPlaying()
        }
        if st.over && changed { sessionOver() }
        updateDock()
        if changed { Snapshots.later("play-" + st.name) }
    }

    /// Runs once per session, when the video is up. It keeps the Mac awake, hides chiaki-ng, takes the keys and moves
    /// this window out of the way.
    private func startedPlaying() {
        awake.hold(true)
        // asks whether the core takes keys. An empty key does nothing, and an older core says "unknown command".
        session?.send("key")
        if source.kind == "chiaki", settingValue("minimize_chiaki")?.bool != false { chiaki.hideForPlay() }
        takeKeys()
        if source.kind == "window" { startOverlay() }
        if options.stopAfter > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + options.stopAfter) { [weak self] in
                guard let self, self.playState == .playing || self.playState.busy else { return }
                self.launcherLog("[launcher] --stop-after: pressing Stop")
                self.stop()
            }
        }
        if !options.keepWindow { NSApp.windows.first(where: { $0.isVisible && $0.canBecomeMain })?.miniaturize(nil) }
    }

    private func sessionOver() {
        awake.hold(false)
        chiaki.showAfterPlay()
        dropKeys()
        stopOverlay()
        Task { await capture.stop() }
        liveState = nil
        restartSent = false
        fileValues = settingsFile.read()
        if let w = NSApp.windows.first(where: { $0.isMiniaturized }) { w.deminiaturize(nil) }
        NSApp.activate(ignoringOtherApps: true)
        if let r = session?.resultsURL, FileManager.default.fileExists(atPath: r.path) { showResults(r) }
        if case .error = playState { showDetails = true }
        resolveKeys()
        if options.quitAfterStop {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { NSApp.terminate(nil) }
        }
    }

    func updateDock() {
        let tile = NSApp.dockTile
        switch playState {
        case .playing: tile.badgeLabel = "Live"
        case .error: tile.badgeLabel = "!"
        default: tile.badgeLabel = playState.busy ? "…" : nil
        }
    }

    func autoPlay() {
        // --play presses Play once the window list is in for App window, and straight away for the other sources
        if source.kind == "window" {
            Task {
                for _ in 0..<40 where !windowsLoaded { try? await Task.sleep(nanoseconds: 100_000_000) }
                if let want = options.window, let w = windows.first(where: {
                    $0.app.localizedCaseInsensitiveContains(want) || $0.bundleID == want || $0.title.localizedCaseInsensitiveContains(want)
                }) { chooseWindow(w.id) }
                play(test: options.test)
            }
        } else {
            play(test: options.test)
        }
    }

    // MARK: - results
    func showResults(_ url: URL) {
        guard let text = try? String(contentsOf: url, encoding: .utf8), let r = TestResults(text: text) else { return }
        results = r
        Snapshots.later("play-results")
    }

    func copyResults() {
        guard let r = results else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(r.text.trimmingCharacters(in: .whitespacesAndNewlines), forType: .string)
        showToast("Results copied. Paste them into your test report")
    }

    // MARK: - keys from any app
    private func takeKeys() {
        Hotkeys.shared.unregisterAll()
        heldKeys = []
        navKeys = []
        guard let s = session else { return }
        let replayOn = (settingValue("replay")?.double ?? 0) > 0
        replayKeysTried = replayOn
        if replayOn {
            for r in WindowKeys.replay {
                guard let c = try? KeyCombo.parse(r.combo) else { continue }
                if Hotkeys.shared.register(c, action: { [weak self] in self?.key(r.key, s) }) != nil { heldKeys.append(c.name) }
                else { launcherLog("[launcher] \(c.name) is taken by another app or macOS; use the video window's \(r.key) key") }
            }
        }
        if source.kind == "window" {
            let (got, taken) = WindowKeys.resolve(prefs, free: { Hotkeys.shared.free($0) })
            keysGot = got
            keysTaken = taken
            for a in WindowKeys.actions {
                guard let name = got[a.id] ?? nil, let c = try? KeyCombo.parse(name) else { continue }
                if Hotkeys.shared.register(c, action: { [weak self] in self?.windowAction(a, s) }) != nil { heldKeys.append(c.name) }
                else { launcherLog("[launcher] \(c.name) is taken by another app or macOS, so \(a.id) has no key from other apps") }
            }
            launcherLog("[launcher] App window keys: " + WindowKeys.actions.map { "\($0.id) \((got[$0.id] ?? nil) ?? "none")" }.joined(separator: ", "))
        }
    }

    private func dropKeys() {
        Hotkeys.shared.unregisterAll()
        heldKeys = []
        navKeys = []
        replayKeysTried = false
    }

    /// In App window mode, while the video window's menu is open, its keys (arrows, Return, Esc, Delete) are taken
    /// plain from any app, and given back when it closes (framegen/hotkeys.py set_nav). This needs the core to say
    /// when the menu is open ("menu_open" in the state).
    private func navKeysFollowMenu(_ s: CoreSession) {
        let want = source.kind == "window" && (liveState?.menuOpen ?? false) && s.supports("key")
        if want == !navKeys.isEmpty { return }
        if want {
            for k in WindowKeys.nav {
                if let id = Hotkeys.shared.register(KeyCombo(mods: 0, key: k), action: { [weak self] in self?.key(k, s) }) { navKeys.append(id) }
            }
        } else {
            navKeys.forEach { Hotkeys.shared.unregister($0) }
            navKeys = []
        }
    }

    /// After every state line, instant replay's keys follow its setting, and the menu's own keys follow the menu.
    private func afterStateChange() {
        guard playState == .playing, let s = session else { return }
        let replayOn = (settingValue("replay")?.double ?? 0) > 0
        // Keys are taken again only when replay was turned on or off, not on every state line while a replay key
        // is held by another app (that re-registered every key and logged the clash each time).
        if replayOn != replayKeysTried { takeKeys() }
        navKeysFollowMenu(s)
    }

    /// A video-window key for the core ("key R"), if it takes them.
    private func key(_ k: String, _ s: CoreSession) {
        guard s.running else { return }
        if k == "R" || k == "B" { s.write("[launcher] key from another app: \(k == "R" ? "save the instant replay" : "save a before & after")") }
        if s.supports("key") { s.send("key \(k)") }
        else if !keysSaid.contains("key") {
            keysSaid.insert("key")
            showToast("This core doesn't take SmoovPlay's keys from other apps yet (it needs the key command)", bad: true)
        }
    }

    private func windowAction(_ a: WindowKeys.Action, _ s: CoreSession) {
        s.write("[launcher] key from another app: \(a.label)")
        switch a.id {
        case "quit": stop()
        case "fullscreen" where !s.supports("key"):  // the same with today's commands
            s.send(CoreLine.set("fullscreen", .bool(!(settingValue("fullscreen")?.bool ?? false))))
        case "stats" where !s.supports("key"):  // Off, Minimal, Detailed, then Off again
            if settingValue("overlay")?.bool != true { s.send(CoreLine.set("overlay_style", "minimal")); s.send(CoreLine.set("overlay", true)) }
            else if settingValue("overlay_style")?.string == "minimal" { s.send(CoreLine.set("overlay_style", "detailed")) }
            else { s.send(CoreLine.set("overlay", false)) }
        default: key(a.key, s)
        }
    }

    /// What App window mode's keys get on this Mac now. It does nothing while playing, when the keys are what was
    /// taken.
    func resolveKeys() {
        guard !running else { return }
        let (got, taken) = WindowKeys.resolve(prefs, free: { Hotkeys.shared.free($0) })
        keysGot = got
        keysTaken = taken
    }

    func listenForKeys(_ action: String) {
        if running { showToast("Stop playing first, then change the keys", bad: true); return }
        keyListening = action
        keysProblem = nil
        if keyMonitor == nil {
            keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] e in
                guard let self, let action = self.keyListening else { return e }
                if e.keyCode == UInt16(kVK_Escape) && e.modifierFlags.intersection([.control, .option, .command]).isEmpty {
                    self.keyListening = nil
                    return nil
                }
                var mods: UInt32 = 0
                if e.modifierFlags.contains(.control) { mods |= UInt32(controlKey) }
                if e.modifierFlags.contains(.option) { mods |= UInt32(optionKey) }
                if e.modifierFlags.contains(.shift) { mods |= UInt32(shiftKey) }
                if e.modifierFlags.contains(.command) { mods |= UInt32(cmdKey) }
                let combo = KeyCombo.from(keyCode: e.keyCode, mods: mods)
                if let why = WindowKeys.problem(combo, action: action, prefs: self.prefs, free: { Hotkeys.shared.free($0) }) {
                    self.keysProblem = why
                } else if let combo {
                    WindowKeys.pick(self.prefs, action, combo.name)
                    self.keyListening = nil
                    self.keysProblem = nil
                    self.resolveKeys()
                }
                return nil
            }
        }
    }

    func defaultKey(_ action: String) { WindowKeys.pick(prefs, action, nil); resolveKeys() }
    func resetKeys() { WindowKeys.resetAll(prefs); resolveKeys() }

    // MARK: - App window: the core's picture over the app's window
    private func startOverlay() {
        overlayAt = nil
        sendOverlay()
        overlayTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sendOverlay() }
        }
    }

    private func stopOverlay() {
        overlayTimer?.invalidate()
        overlayTimer = nil
    }

    private func sendOverlay() {
        guard let s = session, s.running, s.supports("overlay"), let w = capture.window else { return }
        let text: String
        if let (r, onScreen) = windowBounds(w.id), onScreen {
            let t = capture.titleBar  // over the part that's copied
            text = "overlay \(Int(r.minX)) \(Int(r.minY + t)) \(Int(r.width)) \(Int(r.height - t))"
        } else {
            text = "overlay none"
        }
        if text != overlayAt {
            overlayAt = text
            s.send(text)
        }
    }

    // MARK: - settings
    var specsStale = false

    func loadSpecs(force: Bool = false) {
        if loadingSpecs || (specs != nil && !force) || !paths.runtimeReady { return }
        loadingSpecs = true
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.settings", [], timeout: 60)
            let specs = r.json.flatMap { try? SettingSpecs(json: $0) }
            await MainActor.run {
                self.loadingSpecs = false
                if let specs { self.specs = specs; self.specsError = nil }
                else if self.specs == nil { self.specsError = r.lastText ?? "exit code \(r.code)" }
                Snapshots.later("settings-built")
            }
        }
    }

    var settingsModel: SettingsModel? {
        guard let specs else { return nil }
        var m = SettingsModel(specs: specs, state: liveState, file: fileValues, source: source.kind, sessionRunning: playState.running)
        if case .building = playState { m.building = true }
        m.restartSent = restartSent
        return m
    }

    /// A setting's value, from the running session, else the file, else its default.
    func settingValue(_ key: String) -> JSON? {
        if let v = liveState?.values[key] { return v }
        if let m = settingsModel { return m.value(key) }
        return fileValues[key]
    }

    func pick(_ key: String, tag: String) {
        guard let m = settingsModel else { return }
        for (k, v) in m.pick(key, tag: tag) { change(k, v) }
    }

    func change(_ key: String, _ value: JSON) {
        if liveState != nil, let s = session, s.running {
            // the state line that comes back confirms it, or puts the real value back if it couldn't be used
            if !s.send(CoreLine.set(key, value)) { showToast("SmoovPlay isn't running any more", bad: true) }
            return
        }
        if playState.running { return }  // locked meanwhile
        do { try settingsFile.write(key, value) } catch { showToast("Couldn't save the setting: \(error.localizedDescription)", bad: true) }
        fileValues = settingsFile.read()
    }

    func restartNow() {
        guard let s = session, s.send("restart") else { return }
        restartSent = true
    }

    func toggleSection(_ g: String) {
        if openSections.contains(g) { openSections.remove(g) } else { openSections.insert(g) }
        prefs.set("set-open:" + g, openSections.contains(g))
    }

    // MARK: - Auto-detect
    func runDetect() {
        if detecting || !paths.runtimeReady { return }
        if playState.running { showToast("Stop playing first: the test needs the GPU to itself", bad: true); return }
        detecting = true
        detectResult = nil
        detectError = nil
        detectApplied = false
        detectProgress = "Starting the test"
        let paths = paths
        let extra = AppPaths.split(ProcessInfo.processInfo.environment["SMOOVPLAY_DETECT_ARGS"] ?? "")
        Task.detached {
            let r = AutoDetect.run(paths, extra: extra) { p in Task { @MainActor in if self.detecting { self.detectProgress = p } } }
            await MainActor.run {
                self.detecting = false
                switch r {
                case let .success(res): self.detectResult = res
                case let .failure(e): self.detectError = e.text
                }
                Snapshots.later("autodetect")
            }
        }
    }

    var detectChanges: [(SettingSpec, JSON)] {
        guard let r = detectResult, let m = settingsModel else { return [] }
        return r.changes(m)
    }

    func applyDetect() {
        let changes = detectChanges
        detectChanged = Set(changes.map(\.0.key))
        for (spec, v) in changes { change(spec.key, v) }
        detectApplied = true
        showToast(changes.count == 1 ? "1 setting changed" : "\(changes.count) settings changed")
        Snapshots.later("autodetect-applied")
    }

    func dismissDetect() {
        detectResult = nil
        detectError = nil
        detectApplied = false
    }

    // MARK: - the Mac check
    func runChecks() {
        guard !checking else { return }
        checking = true
        let paths = paths, kind = source.kind, installed = chiaki.installed
        Task.detached {
            let c = MacChecks.run(paths: paths, source: kind, chiakiInstalled: installed)
            await MainActor.run { self.checks = c; self.checking = false; Snapshots.later("check") }
        }
    }

    func fix(_ c: Check) {
        switch c.fixID {
        case "displays":
            if let u = URL(string: "x-apple.systempreferences:com.apple.Displays-Settings.extension") { NSWorkspace.shared.open(u) }
        case "screen": askScreenPermission(); runChecks()
        default: break
        }
    }

    // MARK: - chiaki-ng
    func openChiaki() {
        if chiaki.oursRunning || !chiaki.running.isEmpty { showToast("chiaki-ng is already open"); chiaki.running.first?.activate(); return }
        chiaki.open(env: paths.env) { [weak self] err in
            if let err { self?.showToast(err, bad: true) } else { self?.showToast("Opening chiaki-ng...") }
        }
    }

    // MARK: - help
    func copyDiagnostics() {
        showToast("Collecting diagnostics...")
        let paths = paths, kind = source.kind, installed = chiaki.installed, last = session?.logURL
        let extra = ["chiaki-ng: " + (chiaki.oursRunning ? "running (opened by SmoovPlay)" : "not opened by SmoovPlay")
                     + ", \(chiaki.others.count) other chiaki-ng running" + (installed ? "" : ", not built"),
                     "Screen Recording: " + (ScreenPermission.granted ? "allowed" : "not allowed")]
        Task.detached {
            let checks = MacChecks.run(paths: paths, source: kind, chiakiInstalled: installed)
            let text = Diagnostics.collect(paths: paths, version: AppModel.version, source: kind, checks: checks, lastLog: last, extra: extra)
            await MainActor.run {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                self.showToast("Diagnostics copied. Paste them into your bug report")
            }
        }
    }

    func diagnosticsText() -> String {
        Diagnostics.collect(paths: paths, version: AppModel.version, source: source.kind,
                            checks: MacChecks.run(paths: paths, source: source.kind, chiakiInstalled: chiaki.installed),
                            lastLog: nil)
    }

    func openLogs() {
        try? FileManager.default.createDirectory(at: paths.logs, withIntermediateDirectories: true)
        NSWorkspace.shared.open(paths.logs)
    }

    func sendFeedback(_ feature: String) {
        copyDiagnostics()
        let title = feature.isEmpty ? "[Feedback] " : "[Experimental] \(feature) (Mac): "
        var c = URLComponents(string: "https://github.com/smoovplay/smoovplay/issues/new")!
        c.queryItems = [URLQueryItem(name: "template", value: "experimental-feedback.yml"), URLQueryItem(name: "title", value: title)]
        if !feature.isEmpty { c.queryItems?.append(URLQueryItem(name: "feature", value: feature)) }
        if let u = c.url { NSWorkspace.shared.open(u) }
    }

    // MARK: - looks
    func loadLooks(then: (() -> Void)? = nil) {
        guard paths.runtimeReady else { return }
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.looks", ["schema"], timeout: 60)
            await MainActor.run {
                do {
                    guard let j = r.json else { throw JSON.ParseError(description: r.lastText ?? "exit code \(r.code)") }
                    var lib = try LooksLibrary(json: j)
                    // a reload (a pasted code, say) keeps the changes being tried on SmoovPlay's looks
                    if let old = self.looks {
                        lib.presetOriginal = old.presetOriginal.filter { id, _ in lib.entries.contains(where: { $0.builtin && $0.id == id }) }
                        for id in lib.presetOriginal.keys {
                            if let i = lib.entries.firstIndex(where: { $0.id == id }), let was = old.entry(id) { lib.entries[i].look = was.look }
                        }
                    }
                    self.looks = lib
                    self.looksError = nil
                    if self.samples.isEmpty { self.samples = lib.samples }
                    if self.lookPicked == nil || lib.entry(self.lookPicked) == nil {
                        let inUse = self.settingValue("look")?.string
                        self.lookPicked = lib.entry(inUse) != nil ? inUse : lib.entries.first?.id
                    }
                } catch {
                    self.looksError = "\(error)"
                }
                self.requestPreview()
                then?()
                Snapshots.later("looks")
            }
        }
    }

    var looksLive: Bool { liveState != nil && running }
    var pickedLook: LookEntry? { looks?.entry(lookPicked) }
    var pickedIndex: Int? { looks?.entries.firstIndex(where: { $0.id == lookPicked }) }

    func lookInUse(_ id: String) -> Bool {
        settingValue("look_on")?.bool == true && (settingValue("look")?.string ?? "cinematic") == id
    }

    func pickLook(_ id: String) {
        flushLookSave()
        lookPicked = id
        lookHidden = false
        previewPicked()
        requestPreview()
    }

    var currentSample: LookSample? {
        let id = prefs.value("look-sample:")
        return samples.first(where: { $0.id == id }) ?? samples.first
    }

    func stepSample(_ by: Int) {
        guard let s = currentSample, let i = samples.firstIndex(of: s) else { return }
        prefs.choose("look-sample:", samples[(i + by + samples.count) % samples.count].id)
        objectWillChange.send()
        requestPreview()
    }

    func requestPreview() {
        guard page == .looks, let e = pickedLook, let s = currentSample else { return }
        preview.request(look: e.look, sample: s.id, width: previewPixels.0, height: previewPixels.1)
    }

    /// Makes a change to the picked look. A look of yours is saved half a second later, the video gets the change
    /// while playing, and the preview is redrawn.
    func editLook(_ edit: (inout LooksLibrary, Int) -> Void) {
        guard var lib = looks, let i = pickedIndex else { return }
        edit(&lib, i)
        looks = lib
        let e = lib.entries[i]
        lookSaveDue = e.builtin ? nil : e.id
        lookSaveAt = Date().addingTimeInterval(0.5)
        lookHidden = false
        if looksLive { lookSendDue = true }
        lookTimerOn()
        requestPreview()
    }

    private func lookTimerOn() {
        guard lookTimer == nil else { return }
        lookTimer = Timer.scheduledTimer(withTimeInterval: 0.04, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.lookTick() }
        }
    }

    private func lookTick() {
        if lookSendDue { lookSendDue = false; sendLookPreview() }
        if lookSaveDue != nil, Date() >= lookSaveAt { flushLookSave() }
        if !lookSendDue, lookSaveDue == nil { lookTimer?.invalidate(); lookTimer = nil }
    }

    func flushLookSave() {
        guard let id = lookSaveDue, let lib = looks, let e = lib.entry(id) else { lookSaveDue = nil; return }
        lookSaveDue = nil
        do { try lib.save(e) } catch { showToast("Couldn't save the look: \(error.localizedDescription)", bad: true); return }
        if looksLive { session?.send("looks-reload") }  // the running session reads the new version
        specsStale = true
    }

    func previewPicked() {
        guard page == .looks, looksLive, pickedLook != nil, !lookHidden else { return }
        lookSendDue = true
        lookTimerOn()
    }

    private func sendLookPreview() {
        guard let e = pickedLook, looksLive, page == .looks, !lookHidden, let s = session else { return }
        if s.send("look-show " + e.look.serialized()) { lookShownTo = s }
    }

    func onLooksLeave() {
        flushLookSave()
        lookSendDue = false
        lookHidden = false
        hideLookPreview()
    }

    private func hideLookPreview() {
        if let s = lookShownTo, s === session, s.running { s.send("look-hide") }
        lookShownTo = nil
    }

    /// After a state line, or when the session starts or ends, the preview is sent again, because a restart forgets
    /// it.
    private func onLooksState() {
        guard page == .looks else { return }
        if lookShownTo !== session { lookShownTo = nil }
        previewPicked()
    }

    func newLook() {
        guard var lib = looks else { return }
        flushLookSave()
        let id = lib.newLook()
        looks = lib
        if let e = lib.entry(id) { try? lib.save(e) }
        lookOpen[id] = [0]
        lookPicked = id
        specsStale = true
        previewPicked()
        requestPreview()
    }

    func saveAsMine() {
        guard var lib = looks, let i = pickedIndex else { return }
        let open = lookOpen[lib.entries[i].id]
        let id = lib.saveAsMine(i)
        looks = lib
        if let e = lib.entry(id) {
            try? lib.save(e)
            showToast("Saved as your own look: \(e.name)")
        }
        lookOpen[id] = open
        lookPicked = id
        specsStale = true
        if looksLive { session?.send("looks-reload") }
        previewPicked()
        requestPreview()
    }

    func resetLook() {
        guard var lib = looks, let i = pickedIndex else { return }
        lib.reset(i)
        looks = lib
        lookHidden = false
        previewPicked()
        requestPreview()
    }

    func renameLook(_ name: String) {
        guard var lib = looks, let i = pickedIndex, !lib.entries[i].builtin else { return }
        lib.rename(i, name)
        looks = lib
        if let e = lib.entry(lookPicked) { try? lib.save(e) }
        if looksLive { session?.send("looks-reload") }
        specsStale = true
    }

    func deleteLook() {
        guard let e = pickedLook, !e.builtin else { return }
        ask("Delete \(e.name)?", "It's removed from this Mac. Anyone you shared its code with keeps their copy.", ["Delete", "Cancel"]) { r in
            guard r == 0, var lib = self.looks else { return }
            if self.lookSaveDue == e.id { self.lookSaveDue = nil }
            if let f = e.file { try? FileManager.default.removeItem(at: f) }
            let wasUsed = self.settingValue("look")?.string == e.id
            lib.entries.removeAll { $0.id == e.id }
            self.looks = lib
            self.lookOpen[e.id] = nil
            self.specsStale = true
            if self.looksLive { self.session?.send("looks-reload") }
            else if wasUsed && !self.playState.running { try? self.settingsFile.write("look", "cinematic"); self.fileValues = self.settingsFile.read() }
            let at = lib.entries.firstIndex(where: { !$0.builtin }) ?? 0
            self.lookPicked = lib.entries.isEmpty ? nil : lib.entries[min(at, lib.entries.count - 1)].id
            self.previewPicked()
            self.requestPreview()
        }
    }

    func useLook(_ on: Bool) {
        guard let e = pickedLook else { return }
        flushLookSave()
        if looksLive, let s = session {
            s.send("looks-reload")  // so it knows a look that's new
            if on { s.send(CoreLine.set("look", .string(e.id))) }
            s.send(CoreLine.set("look_on", .bool(on)))
            lookHidden = !on
            if !on { lookSendDue = false; hideLookPreview() }
            return
        }
        if playState.running { showToast("SmoovPlay is starting. Try again once the video is on screen.", bad: true); return }
        do {
            if on { try settingsFile.write("look", .string(e.id)) }
            try settingsFile.write("look_on", .bool(on))
        } catch { showToast("Couldn't save the setting: \(error.localizedDescription)", bad: true); return }
        fileValues = settingsFile.read()
        showToast(on ? "\(e.name) is on for your next session. K in the video window turns it off and on." : "Looks are off.")
    }

    func copyLookCode() {
        guard let e = pickedLook, let lib = looks else { return }
        if e.builtin, let code = e.code, !lib.changed(e.id) { copyCode(code); return }
        flushLookSave()
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("SmoovPlay-look-code.json")
        let data = JSON.object([("version", 1), ("name", .string(e.builtin ? "My " + e.name : e.name)), ("nodes", .array(e.nodes))])
        do { try Data(data.serialized().utf8).write(to: tmp) } catch { showToast("Couldn't make its code: \(error.localizedDescription)", bad: true); return }
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.looks", ["encode", tmp.path], timeout: 30)
            let j = PyTool.json(r)
            await MainActor.run {
                if let code = j?["code"]?.string { self.copyCode(code) }
                else { self.showToast("Couldn't make its code: \(j?["error"]?.string ?? r.lastText ?? "it failed")", bad: true) }
            }
        }
    }

    private func copyCode(_ code: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        showToast("Copied the look's code. Anyone with SmoovPlay can paste it on their Looks page.")
    }

    func pasteLookCode() {
        guard looks != nil else { return }
        guard let text = NSPasteboard.general.string(forType: .string), let code = LooksLibrary.code(in: text) else {
            showToast("Copy a look code first (it starts with SPL1-), then press Paste a code", bad: true)
            return
        }
        flushLookSave()
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.looks", ["import", code], timeout: 30)  // letters, digits, - and _ only
            let j = PyTool.json(r)
            await MainActor.run {
                guard let id = j?["id"]?.string else {
                    self.showToast(String((j?["error"]?.string ?? r.lastText ?? "it failed").prefix(160)), bad: true)
                    return
                }
                self.lookPicked = id
                self.specsStale = true
                if self.looksLive { self.session?.send("looks-reload") }
                self.loadLooks { if let e = self.pickedLook { self.showToast("Added \(e.name) to your looks") }; self.previewPicked() }
            }
        }
    }

    func addLut(node: Int) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "cube") ?? .data]
        panel.title = "Add a LUT file"
        guard panel.runModal() == .OK, let file = panel.url else { return }
        let paths = paths
        Task.detached {
            let r = PyTool.run(paths, module: "framegen.looks", ["add-lut", file.path], timeout: 30)
            let j = PyTool.json(r)
            await MainActor.run {
                guard let name = j?["name"]?.string else {
                    self.showToast("Can't use that LUT: \(String((j?["error"]?.string ?? r.lastText ?? "").prefix(140)))", bad: true)
                    return
                }
                if var lib = self.looks, !lib.luts.contains(name) { lib.luts.append(name); lib.luts.sort(); self.looks = lib }
                self.editLook { lib, i in lib.setValue(i, node: node, key: "file", .string(name)) }
            }
        }
    }

    // MARK: - quitting
    /// Quitting. The session is asked to stop as Stop does, and waited for up to 8 s before it's ended outright. The
    /// wait is here on the main thread, because AppKit's quit can come from inside a main-queue block (the SIGTERM
    /// handler), where nothing else on the main queue would run until it returns.
    func shutdown() {
        preview.stop()
        flushLookSave()
        dropKeys()
        awake.hold(false)
        chiaki.showAfterPlay()
        let cap = capture
        Task.detached { await cap.stop() }
        guard let s = session, s.running else { return }
        Trace.log("quitting: asking the core (pid \(s.pid ?? 0)) to stop")
        s.stop()
        let end = Date().addingTimeInterval(8)
        while s.running && Date() < end { usleep(50_000) }
        if s.running { s.stop(); usleep(300_000); s.kill() }  // a second Stop ends it outright
        Trace.log("quitting: the core is \(s.running ? "still running" : "gone")")
    }

}

/// Command-line switches, for testing and scripting (the Windows launcher's --page and SMOOVPLAY_* switches):
///   --page play|guide|check|settings|looks|help    open on a page
///   --source chiaki|window|capture|tap|file        play from this source
///   --window <app, bundle id or title words>       for App window, this app's window
///   --play                                         press Play once ready (--test presses Test with a 60 fps game)
///   --stop-after <seconds>                         press Stop this long after the video is up
///   --quit-after-stop                              quit once the session is over
///   --keep-window                                  don't put the window in the Dock while the video plays
///   --detect                                       run Auto-detect once the settings are in
///   --show-results <results .txt>                  open with that test report on the Play page
///   --diagnostics <file>                           write what Copy diagnostics copies, then quit
struct LaunchOptions {
    var page: String?
    var source: String?
    var window: String?
    var play = false, test = false, quitAfterStop = false, keepWindow = false, detect = false
    var stopAfter: Double = 0
    var showResults: String?
    var diagnostics: String?

    static func parse(_ args: [String]) -> LaunchOptions {
        var o = LaunchOptions()
        var i = 1
        func next() -> String? { i += 1; return i < args.count ? args[i] : nil }
        while i < args.count {
            switch args[i] {
            case "--page": o.page = next()
            case "--source": o.source = next()
            case "--window": o.window = next()
            case "--play": o.play = true
            case "--test": o.test = true; o.play = true
            case "--stop-after": o.stopAfter = Double(next() ?? "") ?? 0
            case "--quit-after-stop": o.quitAfterStop = true
            case "--keep-window": o.keepWindow = true
            case "--detect": o.detect = true
            case "--show-results": o.showResults = next()
            case "--diagnostics": o.diagnostics = next()
            default: break
            }
            i += 1
        }
        return o
    }
}
