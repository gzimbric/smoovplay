// Connect your PS5 (Ui.cs BuildGuide) and the other sources' setup guides. Each is a list of steps to tick off,
// remembered in the prefs as "guide:<id>". Once the PS5's is done, the app opens on Play.
import SmoovPlayKit
import SwiftUI

struct GuideStep {
    var id, title, text: String
    var action: String?  // "chiaki" or "play"
}

enum Guides {
    static let ps5: [GuideStep] = [
        GuideStep(id: "remoteplay", title: "Turn on Remote Play", text: "On the PS5: Settings → System → Remote Play → turn on Enable Remote Play."),
        GuideStep(id: "rest", title: "Let it wake from rest mode (optional)", text: "Settings → System → Power Saving → Features Available in Rest Mode: turn on Stay Connected to the Internet and Enable Turning On PS5 from Network."),
        GuideStep(id: "psn", title: "Sign in to PSN in chiaki-ng", text: "Open chiaki-ng. When it asks, sign in to PlayStation Network in the browser window it opens, then paste the address you land on back into chiaki-ng.", action: "chiaki"),
        GuideStep(id: "register", title: "Register your PS5", text: "chiaki-ng finds consoles on your network. Click the one marked Automatic Registration Available and choose Yes. Not offered? Use a PIN from the PS5: Settings → System → Remote Play → Link Device."),
        GuideStep(id: "video", title: "Stream at 60 fps", text: "In chiaki-ng's settings, under Video: resolution 1080p (720p on a weak connection) and 60 fps. Pick a codec without HDR, since on a Mac SmoovPlay shows an HDR stream in SDR. SmoovPlay warns you if it's set differently."),
        GuideStep(id: "hdcp", title: "Black picture? Turn off HDCP", text: "Only if you need it: if the picture stays black, or the PS5 says the content can't be shown, go to Settings → System → HDMI on the PS5, turn off Enable HDCP and connect again. Video apps like Netflix need HDCP, so turn it back on when you're done playing."),
        GuideStep(id: "controller", title: "Connect your controller", text: "Plug it into the Mac with USB or pair it over Bluetooth. It keeps working while chiaki-ng's window is hidden."),
        GuideStep(id: "play", title: "Press Play", text: "Always start from SmoovPlay: press Play, then connect to your PS5 in chiaki-ng. The smooth video opens in its own window, and chiaki-ng's window is hidden until you stop.", action: "play"),
    ]

    static let window: [GuideStep] = [
        GuideStep(id: "window-app", title: "Open the app and start a game", text: "PS Remote Play, Moonlight, Steam Link, GeForce NOW, or a browser for cloud gaming. If the app has a frame rate setting, pick 60 fps."),
        GuideStep(id: "window-permission", title: "Allow Screen Recording", text: "SmoovPlay copies the app's window with macOS's screen capture. macOS asks the first time you pick App window. You can also turn on SmoovPlay in System Settings > Privacy & Security > Screen & System Audio Recording, then open SmoovPlay again."),
        GuideStep(id: "window-screen", title: "Keep its window open", text: "Not minimised. Windows on top of it don't matter, because SmoovPlay copies the window on its own. A window in its own full-screen space works too."),
        GuideStep(id: "window-sound", title: "Sound and controller", text: "The app keeps the controller, the keyboard and its own sound. Its sound comes a little ahead of the smooth picture."),
        GuideStep(id: "window-play", title: "Pick the window, then press Play", text: "Choose the app's window under the source buttons on the Play page. The app keeps the keyboard, so SmoovPlay's keys use Control-Option: Control-Option-M menu, Control-Option-F fullscreen, Control-Option-Esc quit. The Play page lists them all, and you can change them there.", action: "play"),
    ]

    static let capture: [GuideStep] = [
        GuideStep(id: "capture-wire", title: "Connect the card", text: "The console's HDMI output goes into the card's HDMI input, and the card into this Mac (USB 3 or Thunderbolt). A card with an HDMI output can pass the picture on to your TV too."),
        GuideStep(id: "capture-hdcp", title: "Turn off HDCP on a PS5", text: "On the PS5: Settings → System → HDMI → turn off Enable HDCP. With it on, the card only gets a blank picture. Turn it back on when you're done: video apps like Netflix need it."),
        GuideStep(id: "capture-mode", title: "1080p, 60 Hz, HDR off", text: "Set the console's video output to 1080p at 60 Hz, with HDR off. SmoovPlay expects a 60 Hz signal, in which a 30 fps game shows each frame twice."),
        GuideStep(id: "capture-sound", title: "Sound and controller", text: "Your controller stays connected to the console as usual. On a Mac SmoovPlay doesn't play the card's sound yet, so listen through the card's HDMI output."),
        GuideStep(id: "capture-play", title: "Press Play", text: "The smooth video from the card opens in its own window.", action: "play"),
    ]

    static let tap: [GuideStep] = [
        GuideStep(id: "tap-app", title: "For app developers: an app that sends its frames to SmoovPlay", text: "An app built for it can hand its frames straight to SmoovPlay, saying which are repeats, so nothing is captured from the screen. The format is docs/frame-tap.md in SmoovPlay's GitHub repo (on a Mac, POSIX shared memory /smoovplay-tap). For PS Remote Play, Moonlight or a browser, pick App window instead."),
        GuideStep(id: "tap-sound", title: "Sound and controller", text: "The app keeps the controller and plays its own sound."),
        GuideStep(id: "tap-play", title: "Press Play, then start the app", text: "Either can go first: SmoovPlay waits for the app's frames, then the smooth video opens in its own window.", action: "play"),
    ]

    static func steps(_ s: SourceDef) -> [GuideStep] {
        switch s.kind {
        case "chiaki": return ps5
        case "window": return window
        case "capture": return capture
        case "tap": return tap
        default: return [GuideStep(id: s.kind + "-play", title: s.name, text: s.guide, action: "play")]
        }
    }
}

struct GuidePage: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        let steps = Guides.steps(m.source)
        let ps5 = m.source.remotePlay
        let done = steps.filter { m.prefs.has("guide:" + $0.id) }.count
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top) {
                    PageTitle(overline: ps5 || m.source.kind == "capture" ? "ONE-TIME SETUP  ·  ABOUT 5 MINUTES" : m.source.kind == "window" ? "ONE-TIME SETUP  ·  ABOUT 2 MINUTES" : "TESTING",
                              title: ps5 ? "Connect your PS5" : m.source.kind == "capture" ? "Set up your capture card" : m.source.kind == "window" ? "Set up window capture" : m.source.name,
                              sub: ps5 ? "SmoovPlay gets the picture from chiaki-ng, a free, unofficial Remote Play app. Sony doesn't make or support it. Tick each step as you go; SmoovPlay remembers."
                                : "SmoovPlay gets the picture from " + (m.source.kind == "capture" ? "a capture card between your console and this Mac" : m.source.kind == "window" ? "another app's window on this Mac" : "an app or a video file") + ". Tick each step as you go; SmoovPlay remembers.")
                    Spacer()
                    ZStack {
                        Circle().stroke(Color.line2, lineWidth: 5)
                        Circle().trim(from: 0, to: CGFloat(done) / CGFloat(max(1, steps.count)))
                            .stroke(Color.good, style: StrokeStyle(lineWidth: 5, lineCap: .round)).rotationEffect(.degrees(-90))
                            .animation(.easeOut(duration: 0.5), value: done)
                        Text("\(done)/\(steps.count)").font(.system(size: 14, weight: .semibold)).foregroundStyle(Color.text)
                    }
                    .frame(width: 64, height: 64)
                    .accessibilityIdentifier("GuideCount")
                }
                VStack(spacing: 8) {
                    ForEach(Array(steps.enumerated()), id: \.offset) { i, step in StepCard(n: i + 1, step: step, all: steps) }
                }
                HStack(spacing: 10) {
                    if ps5 {
                        Button { NSWorkspace.shared.open(URL(string: "https://streetpea.github.io/chiaki-ng/")!) } label: { Label("chiaki-ng's full guide", systemImage: "arrow.up.right.square") }
                            .buttonStyle(SPButton(kind: .ghost))
                        Button { m.openChiaki() } label: { Label("Open chiaki-ng", systemImage: "arrow.up.forward.app") }
                            .buttonStyle(SPButton(kind: .secondary))
                            .help("Opens the patched chiaki-ng with the frame tap on. A later Play reuses it.")
                    }
                    Spacer()
                    Button { m.go(.play) } label: { Label("Go to Play", systemImage: "arrow.right") }.buttonStyle(SPButton(kind: .primary))
                }
            }
            .padding(.horizontal, 40).padding(.vertical, 34)
            .frame(maxWidth: 900, alignment: .leading)
        }
    }
}

struct StepCard: View {
    @EnvironmentObject var m: AppModel
    var n: Int
    var step: GuideStep
    var all: [GuideStep]

    var body: some View {
        let key = "guide:" + step.id
        let done = m.prefs.has(key)
        HStack(alignment: .top, spacing: 14) {
            Button {
                m.prefs.set(key, !done)
                if m.source.remotePlay { m.prefs.set("guide-done", all.allSatisfy { m.prefs.has("guide:" + $0.id) }) }
                m.objectWillChange.send()
                if all.allSatisfy({ m.prefs.has("guide:" + $0.id) }) && !done { m.showToast("All set. Press Play and enjoy the smoothness") }
            } label: {
                ZStack {
                    Circle().fill(done ? Color.good : Color.clear).frame(width: 30, height: 30)
                    Circle().stroke(done ? Color.good : Color(hex: 0x3A4250), lineWidth: 1.5).frame(width: 30, height: 30)
                    if done { Image(systemName: "checkmark").font(.system(size: 12, weight: .bold)).foregroundStyle(Color.ink) }
                    else { Text("\(n)").font(.system(size: 12.5, weight: .semibold)).foregroundStyle(Color.text2) }
                }
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(done ? "Done. Click to untick." : "Mark as done")
            .accessibilityIdentifier("guide-\(step.id)")
            VStack(alignment: .leading, spacing: 4) {
                Text(step.title).font(.system(size: 14.5, weight: .semibold)).foregroundStyle(done ? Color.text2 : Color.text)
                Small(step.text)
            }
            .padding(.top, 4)
            Spacer()
            if let a = step.action {
                Button(a == "chiaki" ? "Open chiaki-ng" : "Go to Play") { a == "chiaki" ? m.openChiaki() : m.go(.play) }
                    .buttonStyle(SPButton(kind: .pill)).padding(.top, 2)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.card))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(done ? Color(hex: 0x24503A) : Color.line))
    }
}

struct CheckPage: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .bottom) {
                    PageTitle(overline: "MAC CHECK", title: "Is this Mac ready?",
                              sub: "What SmoovPlay needs on a Mac, for the source picked on the Play page (\(m.source.name)).")
                    Spacer()
                    Button(m.checking ? "Checking..." : "Check again") { m.runChecks() }.buttonStyle(SPButton(kind: .secondary)).disabled(m.checking)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 320), spacing: 12)], spacing: 12) {
                    ForEach(m.checks) { c in
                        Card(border: c.level == .fail ? Color(hex: 0xFF6464, alpha: 0.5) : c.level == .warn ? Color(hex: 0xFFC24B, alpha: 0.45) : .line) {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Image(systemName: c.level == .ok ? "checkmark.circle.fill" : c.level == .info ? "info.circle.fill" : c.level == .warn ? "exclamationmark.triangle.fill" : "xmark.octagon.fill")
                                        .foregroundStyle(c.level == .ok ? Color.good : c.level == .info ? Color.text2 : c.level == .warn ? Color.warn : Color.bad)
                                    Text(c.title).font(.system(size: 14, weight: .semibold)).foregroundStyle(Color.text)
                                }
                                Text(c.value).font(.system(size: 13)).foregroundStyle(Color.text)
                                Small(c.detail)
                                if let f = c.fix { Button(f) { m.fix(c) }.buttonStyle(SPButton(kind: .pill)) }
                            }
                        }
                        .accessibilityIdentifier("check-\(c.id)")
                    }
                }
                if m.checks.isEmpty { Small(m.checking ? "Checking..." : "") }
            }
            .padding(.horizontal, 40).padding(.vertical, 34)
        }
        .onAppear { if m.checks.isEmpty { m.runChecks() } }
    }
}

struct HelpPage: View {
    @EnvironmentObject var m: AppModel
    @State private var customCore = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageTitle(overline: "HELP", title: "Help and tools",
                          sub: "Something off? Copy diagnostics and paste them into your bug report. They leave out your user name.")
                HStack(spacing: 12) {
                    HelpCard(symbol: "doc.on.doc", title: "Copy diagnostics", text: "The Mac check, versions and the last log, ready to paste.") { m.copyDiagnostics() }
                    HelpCard(symbol: "folder", title: "Open the logs folder", text: "One log per session, the test reports and the last Auto-detect.") { m.openLogs() }
                }
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Overline(text: "IN THE VIDEO WINDOW")
                        HStack(spacing: 18) {
                            ForEach([("M", "Menu"), ("F", "Fullscreen"), ("I", "Frame generation on/off"), ("S", "Stats"), ("U", "Upscaler"), ("Q", "Stop")], id: \.0) { k, t in
                                HStack(spacing: 8) { KeyCap(key: k); Text(t).font(.system(size: 12.5)).foregroundStyle(Color.text2) }
                            }
                        }
                        Small("From any app, Control-Option-R saves the instant replay and Control-Option-B a before & after, while instant replay is on (Settings). Playing from an app's window? The app keeps the keyboard, so the video window's keys use Control-Option there: Control-Option-M menu, Control-Option-F fullscreen, Control-Option-Esc quit. The Play page lists them all and lets you change them.")
                        Small("Only 60 fps games? Test with a 60 fps game, on the Play page, plays one as if it were 30 fps.")
                        Small("chiaki-ng is an unofficial Remote Play app. SmoovPlay and chiaki-ng are independent projects, not made, supported or endorsed by Sony Interactive Entertainment. PlayStation and PS5 are its trademarks.", color: .muted)
                    }
                }
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Overline(text: "THE CORE")
                        Small("The Python pipeline (run.py) plays the video today, or the Rust core once it's built. Both take the same command line and the same commands.")
                        Picker("", selection: Binding(get: { ["python", "rust"].contains(m.coreChoice) ? m.coreChoice : "custom" },
                                                      set: { v in if v == "custom" { if !customCore.isEmpty { m.coreChoice = customCore } } else { m.coreChoice = v } })) {
                            Text("Python (run.py)").tag("python")
                            Text("Rust (mac/core/target/release/smoovplay)").tag("rust")
                            Text("Another core binary").tag("custom")
                        }
                        .pickerStyle(.radioGroup)
                        .disabled(m.running)
                        HStack {
                            TextField("Path to a core binary", text: $customCore).textFieldStyle(.roundedBorder).frame(maxWidth: 420)
                            Button("Use it") { if !customCore.isEmpty { m.coreChoice = customCore } }.buttonStyle(SPButton(kind: .ghost)).disabled(m.running)
                        }
                        Small(m.paths.coreProblem ?? "Using \(m.paths.coreName), from \(m.paths.repo.path).", color: m.paths.coreProblem == nil ? .text2 : .warn)
                        if ProcessInfo.processInfo.environment["SMOOVPLAY_CORE"] != nil { Small("SMOOVPLAY_CORE is set, and overrides this.", color: .muted) }
                    }
                }
                HStack(spacing: 10) {
                    Button("Send feedback") { m.sendFeedback("") }.buttonStyle(SPButton(kind: .ghost))
                    Button("Licences") { NSWorkspace.shared.open(m.paths.repo.appendingPathComponent("installer/THIRD_PARTY_NOTICES.txt")) }.buttonStyle(SPButton(kind: .ghost))
                    Button("chiaki-ng source") { NSWorkspace.shared.open(URL(string: "https://github.com/streetpea/chiaki-ng")!) }.buttonStyle(SPButton(kind: .ghost))
                    Button("chiaki-ng help") { NSWorkspace.shared.open(URL(string: "https://streetpea.github.io/chiaki-ng/")!) }.buttonStyle(SPButton(kind: .ghost))
                    Spacer()
                    Text("SmoovPlay for Mac \(AppModel.version)").font(.system(size: 12)).foregroundStyle(Color.muted)
                }
            }
            .padding(.horizontal, 40).padding(.vertical, 34)
            .frame(maxWidth: 980, alignment: .leading)
        }
        .onAppear { if !["python", "rust"].contains(m.coreChoice) { customCore = m.coreChoice } }
    }
}

struct HelpCard: View {
    var symbol, title, text: String
    var action: () -> Void
    @State private var hover = false
    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: symbol).font(.system(size: 16)).foregroundStyle(Color.accent).frame(width: 32, height: 32)
                    .background(RoundedRectangle(cornerRadius: 9).fill(Color(hex: 0x1E222B)))
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.system(size: 14, weight: .semibold)).foregroundStyle(Color.text)
                    Small(text)
                }
                Spacer()
            }
            .padding(14)
            .background(RoundedRectangle(cornerRadius: 12).fill(hover ? Color.cardHover : Color.card))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.line))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}
