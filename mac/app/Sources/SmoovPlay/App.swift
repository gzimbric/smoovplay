// SmoovPlay for macOS. The launcher window has the Windows launcher's pages in a rail on the left.
import AppKit
import SmoovPlayKit
import SwiftUI

@main
struct SmoovPlayApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate

    var body: some Scene {
        Window("SmoovPlay", id: "main") {
            RootView()
                .environmentObject(delegate.model)
                .frame(minWidth: 940, minHeight: 640)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1180, height: 780)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(after: .help) {
                Button("Copy Diagnostics") { delegate.model.copyDiagnostics() }
                Button("Open the Logs Folder") { delegate.model.openLogs() }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let options = LaunchOptions.parse(CommandLine.arguments)
    private var sigterm: DispatchSourceSignal?
    lazy var model: AppModel = MainActor.assumeIsolated { AppModel(options: options) }

    func applicationDidFinishLaunching(_ n: Notification) {
        SmoovPlayKit.ignoreSigpipe()
        // SIGTERM (kill, a logout) quits as Quit does, so the core and chiaki-ng's window aren't left behind
        signal(SIGTERM, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { Trace.log("SIGTERM: quitting"); NSApp.terminate(nil) }
        src.resume()
        sigterm = src
        if let out = options.diagnostics {  // --diagnostics FILE writes what Copy diagnostics copies, then quits
            let text = MainActor.assumeIsolated { model.diagnosticsText() }
            try? text.write(toFile: out, atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
            return
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ app: NSApplication) -> NSApplication.TerminateReply {
        Trace.log("quitting")
        MainActor.assumeIsolated { model.shutdown() }
        return .terminateNow
    }
}

struct RootView: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        ZStack(alignment: .bottom) {
            HStack(spacing: 0) {
                Rail()
                    .frame(width: 236)
                Divider().overlay(Color.line)
                Group {
                    switch m.page {
                    case .play: PlayPage()
                    case .guide: GuidePage()
                    case .check: CheckPage()
                    case .settings: SettingsPage()
                    case .looks: LooksPage()
                    case .help: HelpPage()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            if let t = m.toast {
                ToastView(toast: t).padding(.bottom, 22).transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .background(Color.bg)
        .animation(.easeOut(duration: 0.2), value: m.toast)
        .sheet(item: $m.dialog) { d in DialogView(dialog: d) }
        .background(WindowFinder())
        .onAppear { m.go(m.page) }  // loads the first page (--page, or Play or the guide)
    }
}

struct ToastView: View {
    var toast: Toast
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: toast.bad ? "exclamationmark.circle.fill" : "checkmark.circle.fill")
                .foregroundStyle(toast.bad ? Color.bad : Color.good)
            Text(toast.text).font(.system(size: 13)).foregroundStyle(Color.text).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 16).padding(.vertical, 11)
        .frame(maxWidth: 620)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(hex: 0x1E232C)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.line2))
        .shadow(color: .black.opacity(0.4), radius: 18, y: 6)
        .accessibilityIdentifier("toast")
    }
}

struct DialogView: View {
    var dialog: Dialog
    @Environment(\.dismiss) var dismiss
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(dialog.title).font(.system(size: 17, weight: .semibold)).foregroundStyle(Color.text)
            Text(dialog.text).font(.system(size: 13)).foregroundStyle(Color.text2).fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                ForEach(Array(dialog.buttons.enumerated().reversed()), id: \.offset) { i, b in
                    Button(b) { dismiss(); dialog.done(i) }
                        .buttonStyle(SPButton(kind: i == 0 ? .primary : .secondary))
                        .keyboardShortcut(i == 0 ? .defaultAction : i == dialog.buttons.count - 1 ? .cancelAction : nil)
                }
            }
        }
        .padding(22)
        .frame(width: 460)
        .background(Color.card)
    }
}

struct Rail: View {
    @EnvironmentObject var m: AppModel
    var items: [(Page, String, String)] {
        [(.play, "play.fill", "Play"),
         (.guide, "gamecontroller", m.source.remotePlay ? "Connect your PS5" : "Set up"),
         (.check, "checkmark.seal", "Mac check"),
         (.settings, "slider.horizontal.3", "Settings"),
         (.looks, "camera.filters", "Looks"),
         (.help, "questionmark.circle", "Help")]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 40, height: 40)
                VStack(alignment: .leading, spacing: 1) {
                    Text("SmoovPlay").font(.system(size: 19, weight: .semibold)).foregroundStyle(Color.text)
                    Text("v\(AppModel.version) for Mac").font(.system(size: 11.5)).foregroundStyle(Color.muted)
                }
            }
            .padding(.bottom, 26)
            ForEach(items, id: \.0) { page, symbol, label in
                Button { m.go(page) } label: {
                    HStack(spacing: 12) {
                        Image(systemName: symbol).frame(width: 18)
                        Text(label).font(.system(size: 13.5, weight: m.page == page ? .semibold : .regular))
                        Spacer()
                        if page == .play, m.playState.running || m.playState == .playing {
                            Circle().fill(m.playState == .playing ? Color.good : Color.accent).frame(width: 7, height: 7)
                        }
                    }
                    .foregroundStyle(m.page == page ? Color.text : Color.text2)
                    .padding(.horizontal, 12).frame(height: 38)
                    .background(RoundedRectangle(cornerRadius: 9).fill(m.page == page ? Color(hex: 0x1E232C) : Color.clear))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("Nav\(page.rawValue.capitalized)")
            }
            Spacer()
            VStack(alignment: .leading, spacing: 3) {
                Text("Free and open source").font(.system(size: 12)).foregroundStyle(Color.text2)
                Text("MIT licence. Not affiliated with Sony, NVIDIA or AMD.").font(.system(size: 11.5)).foregroundStyle(Color.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 18).padding(.top, 30).padding(.bottom, 22)
        .frame(maxHeight: .infinity)
        .background(Color.rail)
    }
}

/// Keeps a reference to the window, for SMOOVPLAY_SHOTS.
struct WindowFinder: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async {
            Snapshots.window = v.window
            v.window?.backgroundColor = NSColor(Color.bg)
            v.window?.titlebarAppearsTransparent = true
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

/// SMOOVPLAY_SHOTS=<folder> saves a PNG of the window on every page or state change. The app draws it itself,
/// never with a screen capture, so nothing but SmoovPlay's own window is in it.
enum Snapshots {
    static weak var window: NSWindow?
    static let folder = ProcessInfo.processInfo.environment["SMOOVPLAY_SHOTS"]
    static var n = 0

    static func later(_ name: String, ms: Int = 700) {
        guard folder != nil else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(ms)) { snap(name) }
    }

    static func snap(_ name: String) {
        guard let folder, let view = window?.contentView else { return }
        let rect = view.bounds
        guard let rep = view.bitmapImageRepForCachingDisplay(in: rect) else { return }
        view.cacheDisplay(in: rect, to: rep)
        guard let png = rep.representation(using: .png, properties: [:]) else { return }
        try? FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        n += 1
        try? png.write(to: URL(fileURLWithPath: folder).appendingPathComponent(String(format: "app-%02d-%@.png", n, name)))
    }
}
