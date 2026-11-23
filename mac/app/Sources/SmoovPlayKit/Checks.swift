// The Mac check (installer/launcher/Checks.cs, the parts that apply to a Mac): the chip and memory, the display's
// refresh rate, disk space, the home network, the pipeline, chiaki-ng, and Screen Recording for App window mode.
// NVIDIA's driver, the Visual C++ runtime, DirectML and long paths are Windows' alone.
import AppKit
import CoreGraphics
import Network

public struct Check: Identifiable, Equatable, Sendable {
    public enum Level: String, Sendable { case ok = "OK", info = "Info", warn = "Warn", fail = "Fail" }
    public var id: String
    public var title: String
    public var level: Level
    public var value: String
    public var detail: String
    public var fix: String?  // a button's label
    public var fixID: String?
}

public enum MacInfo {
    public static var chip: String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        let s = String(cString: buf)
        return s.isEmpty ? "unknown chip" : s
    }

    public static var appleSilicon: Bool {
        var v: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("hw.optional.arm64", &v, &size, nil, 0) == 0 && v == 1
    }

    public static var memoryGB: Int { Int((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824).rounded()) }

    /// The main display's refresh rate now, and the fastest it could be set to (0 if macOS doesn't say).
    public static func refresh() -> (now: Double, best: Double, size: String) {
        let d = CGMainDisplayID()
        let mode = CGDisplayCopyDisplayMode(d)
        var now = mode?.refreshRate ?? 0
        let opts = [kCGDisplayShowDuplicateLowResolutionModes: kCFBooleanTrue] as CFDictionary
        let modes = (CGDisplayCopyAllDisplayModes(d, opts) as? [CGDisplayMode]) ?? []
        let best = modes.filter { $0.pixelWidth == mode?.pixelWidth && $0.pixelHeight == mode?.pixelHeight }.map(\.refreshRate).max() ?? 0
        if now == 0, let screen = NSScreen.main { now = Double(screen.maximumFramesPerSecond) }
        let size = mode.map { "\($0.pixelWidth)×\($0.pixelHeight)" } ?? ""
        return (now, max(best, now), size)
    }

    /// The widest screen's shape (width / height), for the Fill the screen drawing.
    public static var widestRatio: Double {
        NSScreen.screens.map { $0.frame.width / max(1, $0.frame.height) }.max() ?? 16.0 / 9.0
    }

    public static func freeGB(_ url: URL) -> Double? {
        let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage.map { Double($0) / 1e9 }
    }
}

public enum MacChecks {
    /// The network the Mac is on now: "Wired", "Wi-Fi", or nil if it can't tell within a second.
    public static func network() -> String? {
        let m = NWPathMonitor()
        let done = DispatchSemaphore(value: 0)
        var kind: String?
        m.pathUpdateHandler = { p in
            kind = p.usesInterfaceType(.wiredEthernet) ? "Wired" : p.usesInterfaceType(.wifi) ? "Wi-Fi"
                : p.status == .satisfied ? "Other" : "Offline"
            done.signal()
        }
        m.start(queue: DispatchQueue(label: "net"))
        _ = done.wait(timeout: .now() + 1)
        m.cancel()
        return kind
    }

    /// Every card. It blocks while it asks the network. source is the Play page's pick.
    public static func run(paths: AppPaths, source: String, chiakiInstalled: Bool) -> [Check] {
        var out: [Check] = []
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let gb = MacInfo.memoryGB
        if !MacInfo.appleSilicon {
            out.append(Check(id: "mac", title: "This Mac", level: .fail, value: MacInfo.chip,
                             detail: "SmoovPlay needs an Apple silicon Mac (M1 or newer), because RIFE runs on its GPU through Metal."))
        } else {
            out.append(Check(id: "mac", title: "This Mac", level: gb < 8 ? .warn : .ok,
                             value: "\(MacInfo.chip), \(gb) GB, macOS \(os.majorVersion).\(os.minorVersion)",
                             detail: gb < 8 ? "With under 8 GB of memory, other apps may slow the video down." : "Fast enough for 60 fps. Auto-detect tells how far it goes."))
        }
        let r = MacInfo.refresh()
        if r.best > r.now + 1 {
            out.append(Check(id: "display", title: "Display", level: .warn, value: "\(r.size) at \(Int(r.now)) Hz",
                             detail: "It can run at \(Int(r.best)) Hz. Set it there in System Settings > Displays for 90 or 120 fps.",
                             fix: "Open Displays", fixID: "displays"))
        } else {
            out.append(Check(id: "display", title: "Display", level: r.now >= 89 ? .ok : .info, value: "\(r.size) at \(Int(r.now)) Hz",
                             detail: r.now >= 119 ? "Fast enough for 120 fps." : r.now >= 89 ? "Fast enough for 90 fps." : "60 fps is the most it shows. That's fine, because frame generation still doubles a 30 fps game."))
        }
        if let free = MacInfo.freeGB(FileManager.default.homeDirectoryForCurrentUser) {
            out.append(Check(id: "disk", title: "Disk space", level: free < 5 ? .warn : .ok, value: String(format: "%.0f GB free", free),
                             detail: free < 5 ? "Instant replay clips go to Movies/SmoovPlay, so make some room first." : "Room for instant replay clips (Movies/SmoovPlay)."))
        }
        if paths.runtimeReady {
            out.append(Check(id: "pipeline", title: "The pipeline", level: paths.coreProblem == nil ? .ok : .fail,
                             value: paths.coreName, detail: paths.coreProblem ?? "From \(paths.repo.path)."))
        } else {
            out.append(Check(id: "pipeline", title: "The pipeline", level: .fail, value: "Not found",
                             detail: paths.coreProblem ?? "No Python or run.py at \(paths.repo.path)."))
        }
        switch source {
        case "chiaki":
            out.append(Check(id: "chiaki", title: "chiaki-ng", level: chiakiInstalled ? .ok : .fail,
                             value: chiakiInstalled ? "With the frame tap" : "Not built",
                             detail: chiakiInstalled ? paths.chiakiApp.path : "Build it with chiaki_tap/build_macos.sh (chiaki_tap/README.md, Build on macOS)."))
            let net = network() ?? "Unknown"
            out.append(Check(id: "lan", title: "Home network", level: net == "Wi-Fi" ? .info : net == "Offline" ? .warn : .ok, value: net,
                             detail: net == "Wi-Fi" ? "Remote Play is smoothest over a cable, or Wi-Fi on 5 or 6 GHz." : net == "Offline" ? "Not connected. Remote Play needs the network the PS5 is on." : "A cable gives the steadiest stream."))
        case "window":
            let ok = ScreenPermission.granted
            out.append(Check(id: "screen", title: "Screen Recording", level: ok ? .ok : .warn, value: ok ? "Allowed" : "Not allowed yet",
                             detail: ok ? "SmoovPlay can copy the app's window." : ScreenPermission.deniedText,
                             fix: ok ? nil : "Allow", fixID: ok ? nil : "screen"))
        default: break
        }
        return out
    }
}
