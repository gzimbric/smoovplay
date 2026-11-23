// The patched chiaki-ng (build/chiaki-ng-frametap/chiaki-ng.app, chiaki_tap/README.md "Build on macOS"): opened
// with the frame tap on, hidden while SmoovPlay's video plays and shown again when it ends (run.py's
// minimize_chiaki does this on Windows; on a Mac the app does it, with NSRunningApplication), and the Mac kept
// awake meanwhile.
import AppKit
import IOKit.pwr_mgt

public final class Chiaki: @unchecked Sendable {
    public let app: URL
    public let bundleID = AppPaths.chiakiBundleID
    /// The copy SmoovPlay opened (the Windows launcher keeps runtime\chiaki.pid for the same).
    public private(set) var ours: NSRunningApplication?
    private var hiddenByUs = false
    private var hiddenPIDs: [pid_t] = []
    private var keepDownUntil = Date.distantPast
    private var observers: [NSObjectProtocol] = []
    public var log: (String) -> Void = { _ in }

    public init(app: URL) { self.app = app }

    public var installed: Bool { FileManager.default.fileExists(atPath: app.appendingPathComponent("Contents/MacOS/chiaki").path) }

    public var running: [NSRunningApplication] {
        var list = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
        list += NSWorkspace.shared.runningApplications.filter { a in a.bundleIdentifier == bundleID && !list.contains { $0.processIdentifier == a.processIdentifier } }
        if let o = ours, !o.isTerminated, !list.contains(where: { $0.processIdentifier == o.processIdentifier }) { list.append(o) }
        return list.filter { !$0.isTerminated }
    }

    public var oursRunning: Bool { ours.map { !$0.isTerminated } ?? false }

    /// chiaki-ng copies SmoovPlay didn't open: they don't send the video to the tap (unless started with
    /// run_chiaki_frametap.command, which the app can't tell).
    public var others: [NSRunningApplication] {
        running.filter { $0.processIdentifier != ours?.processIdentifier }
    }

    /// Opens chiaki-ng with CHIAKI_FRAMETAP=1 (and the shell's other CHIAKI_* variables, as
    /// run_chiaki_frametap.command passes them), as a new copy.
    public func open(env: [String: String], done: @escaping (String?) -> Void) {
        guard installed else {
            done("chiaki-ng isn't at \(app.path). Build it with chiaki_tap/build_macos.sh (chiaki_tap/README.md), or set SMOOVPLAY_CHIAKI.")
            return
        }
        let cfg = NSWorkspace.OpenConfiguration()
        var e = env.filter { $0.key.hasPrefix("CHIAKI_") }
        e["CHIAKI_FRAMETAP"] = "1"
        cfg.environment = e
        cfg.createsNewApplicationInstance = true
        cfg.activates = true
        NSWorkspace.shared.openApplication(at: app, configuration: cfg) { [weak self] running, error in
            DispatchQueue.main.async {
                if let running { self?.ours = running }
                if let error { done("chiaki-ng didn't open: \(error.localizedDescription)") } else {
                    self?.log("[launcher] opened chiaki-ng with the frame tap on (pid \(running?.processIdentifier ?? 0))")
                    done(nil)
                }
            }
        }
    }

    /// Quits these copies (asked nicely, then forced after 5 s).
    public func close(_ apps: [NSRunningApplication], done: @escaping () -> Void) {
        for a in apps { a.terminate() }
        DispatchQueue.global().async {
            let end = Date().addingTimeInterval(5)
            while Date() < end, apps.contains(where: { !$0.isTerminated }) { usleep(100_000) }
            for a in apps where !a.isTerminated { a.forceTerminate() }
            DispatchQueue.main.async(execute: done)
        }
    }

    /// While the video plays: hidden (it keeps streaming, and stops drawing video nobody sees). For the first 10 s
    /// it's hidden again if it shows itself (its stream view opening); after that, showing it is the player's choice.
    public func hideForPlay() {
        guard let a = running.first(where: { $0.processIdentifier == ours?.processIdentifier }) ?? running.first else { return }
        if !a.isHidden { a.hide() }
        hiddenByUs = true
        hiddenPIDs = [a.processIdentifier]
        keepDownUntil = Date().addingTimeInterval(10)
        log("[launcher] hid chiaki-ng's window while the video plays; it comes back when the session ends")
        if observers.isEmpty {
            let nc = NSWorkspace.shared.notificationCenter
            for name in [NSWorkspace.didUnhideApplicationNotification, NSWorkspace.didActivateApplicationNotification] {
                observers.append(nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] n in
                    guard let self, self.hiddenByUs, Date() < self.keepDownUntil,
                          let app = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                          app.bundleIdentifier == self.bundleID else { return }
                    app.hide()
                    self.log("[launcher] chiaki-ng showed its window again: hid it")
                })
            }
        }
    }

    /// The session ended: chiaki-ng's window back, without taking the keys from whatever has them.
    public func showAfterPlay() {
        observers.forEach { NSWorkspace.shared.notificationCenter.removeObserver($0) }
        observers = []
        guard hiddenByUs else { return }
        hiddenByUs = false
        // by process id, as hidden: the list of running apps by bundle id has come back empty at this point
        var apps = hiddenPIDs.compactMap { NSRunningApplication(processIdentifier: $0) }.filter { !$0.isTerminated }
        apps += running.filter { a in !apps.contains { $0.processIdentifier == a.processIdentifier } }
        for a in apps { a.unhide() }  // (isHidden can lag behind; unhiding a shown app does nothing)
        hiddenPIDs = []
        log("[launcher] chiaki-ng's window is back (\(apps.count) running)")
    }
}

/// Keeps the display and the Mac awake while the video plays (a controller isn't input to macOS, so the display
/// would sleep mid-game).
public final class Awake: @unchecked Sendable {
    private var id: IOPMAssertionID = 0
    private var held = false

    public init() {}

    @discardableResult
    public func hold(_ on: Bool, reason: String = "SmoovPlay is playing a game stream") -> Bool {
        if on == held { return true }
        if on {
            let st = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleDisplaySleep as CFString,
                                                 IOPMAssertionLevel(kIOPMAssertionLevelOn), reason as CFString, &id)
            held = st == kIOReturnSuccess
            return held
        }
        IOPMAssertionRelease(id)
        held = false
        return true
    }

    public var holding: Bool { held }

    deinit { if held { IOPMAssertionRelease(id) } }
}
