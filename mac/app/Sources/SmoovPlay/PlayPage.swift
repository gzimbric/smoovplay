// The Play page (Ui.cs SetPlayState, BuildSources, BuildDevices, BuildWindowKeys, ShowResults): the Play button,
// what's happening while it starts, where the video comes from, the test-mode results card, and the log.
import SmoovPlayKit
import SwiftUI

struct PlayPage: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                PlayDisc()
                    .scaleEffect(m.results == nil ? 1 : 0.72)
                    .frame(height: m.results == nil ? 210 : 160)
                    .padding(.top, 20)
                VStack(spacing: 8) {
                    Text(title).font(.system(size: 28, weight: .semibold)).foregroundStyle(Color.text)
                        .accessibilityIdentifier("PlayTitle")
                    Text(sub).font(.system(size: 13.5)).foregroundStyle(Color.text2).multilineTextAlignment(.center)
                        .frame(maxWidth: 560).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("PlaySub")
                }
                if m.sources.count > 1 { SourcePills() }
                SourceExtras()
                if m.source.kind == "window" { KeysCard() }
                if m.results == nil { Chips() }
                if let n = m.note { NoteLine(text: n, color: .warn) }
                if let n = m.overlayNote, m.playState.running { NoteLine(text: n, color: .text2) }
                if !m.playState.running && m.source.testMode {
                    Button { m.play(test: true) } label: {
                        Label("Test with a 60 fps game", systemImage: "checkmark.seal")
                    }
                    .buttonStyle(SPButton(kind: .ghost))
                    .help("Plays a 60 fps game as if it were 30 fps, rates each generated frame against the real one that was left out, and shows a report when you stop.")
                    .accessibilityIdentifier("BtnPlayTest")
                }
                if m.results != nil { ResultsCard() }
                Details()
            }
            .padding(.horizontal, 40).padding(.bottom, 30)
            .frame(maxWidth: .infinity)
        }
    }

    var title: String {
        switch m.playState {
        case .starting: return "Starting SmoovPlay"
        case .waiting: return "Waiting for your stream"
        case .found: return "Stream found"
        case let .building(d): return d != nil ? "Applying your settings" : "Optimising for this stream"
        case .playing: return m.session?.options.testMode == true ? "Testing" : "Playing"
        case .ended: return "The stream ended"
        case .stopped: return "Stopped"
        case .error: return "SmoovPlay stopped"
        case .idle: return "Ready when you are"
        }
    }

    var sub: String {
        let kind = m.source.kind
        switch m.playState {
        case .starting: return "Waking up the AI model..."
        case .waiting:
            return kind == "tap" ? "Start the app that sends its frames to SmoovPlay. Its smooth video opens as soon as they come."
                : kind == "window" ? "Waiting for the window's pictures."
                : "chiaki-ng is open. Connect to your PS5 there and start Remote Play."
        case let .found(d): return (d.map { $0 + ". " } ?? "") + "Loading the AI model..."
        case let .building(d):
            return d.map { "Restarting the AI model for \($0). A few seconds; the stream keeps going." }
                ?? "Once per stream size: a minute or two. Then it's instant."
        case .playing:
            if m.session?.options.testMode == true {
                return "Your 60 fps game is playing as if it were 30 fps. Play a few minutes, with some fast motion, then press Q in the video window for the results."
            }
            if kind == "window" {
                let keys = WindowKeys.sentence(m.keysGot)
                let where_ = m.overlayNote == nil ? "The smooth picture is over the app's window" : "The smooth video is in its own window"
                return where_ + ", and the app keeps the keyboard and the controller. "
                    + (keys.isEmpty ? "Stop here ends it." : keys + ", from anywhere (keys below).") + " Settings here apply straight away."
            }
            return "The smooth video is in its own window. There, M opens the menu, F goes fullscreen and Esc stops. Settings here apply to it straight away."
        case .ended: return kind == "chiaki" ? "Press Play to connect again." : "Press Play to start again."
        case .stopped: return "Press Play to start again."
        case let .error(d):
            let s = d.count > 180 ? String(d.prefix(177)) + "..." : d
            return s + "  Copy diagnostics puts the details on the clipboard for a bug report."
        case .idle:
            switch kind {
            case "chiaki": return "Press Play. chiaki-ng opens: connect to your PS5 there, and the smooth video appears in its own window."
            case "capture": return "Press Play. The smooth video from your capture card appears in its own window."
            case "window": return "Press Play. SmoovPlay copies the app's window and plays it smoothly, and the app keeps the keyboard and the controller. SmoovPlay's own keys are below."
            case "tap": return "Press Play, then start the app. Its smooth video appears in its own window."
            default: return "Press Play to play the test video."
            }
        }
    }
}

struct PlayDisc: View {
    @EnvironmentObject var m: AppModel
    @State private var spin = false
    @State private var glow = false

    var body: some View {
        let running = m.playState.running
        ZStack {
            Circle().fill(Color(hex: 0xFF9A1F, alpha: glow ? 0.28 : 0.14)).frame(width: 220, height: 220).blur(radius: 30)
            if m.playState.busy {
                Circle().trim(from: 0, to: 0.28)
                    .stroke(Color.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .frame(width: 172, height: 172)
                    .rotationEffect(.degrees(spin ? 360 : 0))
                    .onAppear { withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) { spin = true } }
                    .onDisappear { spin = false }
            }
            Button { m.playPressed() } label: {
                ZStack {
                    Circle().fill(Color.accentGrad).frame(width: 150, height: 150)
                        .shadow(color: Color(hex: 0xFF8A1F, alpha: 0.45), radius: 22, y: 8)
                    if running {
                        RoundedRectangle(cornerRadius: 6).fill(Color.ink).frame(width: 40, height: 40)
                    } else {
                        Image(systemName: "play.fill").font(.system(size: 54)).foregroundStyle(Color.ink).offset(x: 5)
                    }
                }
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(running ? (m.source.remotePlay ? "Stop SmoovPlay (chiaki-ng stays open)" : "Stop SmoovPlay") : "Play")
            .accessibilityLabel(running ? "Stop" : "Play")
            .accessibilityIdentifier("PlayButton")
            .keyboardShortcut(.return, modifiers: [])
            if m.playState == .playing {
                HStack(spacing: 6) {
                    Circle().fill(Color.good).frame(width: 7, height: 7).opacity(glow ? 1 : 0.35)
                    Text("LIVE").font(.system(size: 11, weight: .bold)).foregroundStyle(Color.good)
                }
                .padding(.horizontal, 9).padding(.vertical, 3)
                .background(Capsule().fill(Color(hex: 0x0E1014, alpha: 0.85)))
                .overlay(Capsule().stroke(Color(hex: 0x3DDC8E, alpha: 0.5)))
                .offset(y: 92)
            }
        }
        .onAppear { withAnimation(.easeInOut(duration: 1.6).repeatForever()) { glow = true } }
    }
}

struct SourcePills: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        FlowLayout(spacing: 8) {
            ForEach(m.sources) { s in
                let on = s.kind == m.source.kind
                Button { m.pickSource(s) } label: {
                    HStack(spacing: 0) {
                        Text(s.name).font(.system(size: 12.5, weight: on ? .semibold : .regular))
                            .foregroundStyle(on ? Color.accent : Color.text2)
                        if s.experimental { ExperimentalBadge().padding(.leading, 7) }
                    }
                    .padding(.horizontal, 14).padding(.vertical, 6)
                    .background(Capsule().fill(on ? Color(hex: 0xFFB02E, alpha: 0.15) : Color.chip))
                    .overlay(Capsule().stroke(on ? Color.accent : Color.line2))
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .help(s.guide)
                .accessibilityIdentifier("source-\(s.kind)")
            }
            if m.source.experimental {
                Button("Send feedback") { m.sendFeedback(m.source.name) }.buttonStyle(SPButton(kind: .pill))
                    .help("Copies diagnostics and opens a GitHub page for your report. Nothing is sent until you post it.")
            }
        }
    }
}

/// What goes under the source pills: which capture card, or which app window and Screen Recording.
struct SourceExtras: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        switch m.source.kind {
        case "capture":
            HStack(spacing: 10) {
                if !m.devicesLoaded {
                    Small("Looking for capture devices…")
                } else if m.devices.isEmpty {
                    Small(m.devicesError.map { "Couldn't list capture devices: \($0)" } ?? "No capture device found. Connect the card, then press Look again.", color: .warn)
                } else {
                    Text("Card").font(.system(size: 12.5)).foregroundStyle(Color.text2)
                    Picker("", selection: Binding(get: { m.chosenDevice }, set: { m.prefs.choose("capture-video:", $0); m.objectWillChange.send() })) {
                        ForEach(m.devices) { d in Text(Diagnostics.noOwner(d.name)).tag(d.index) }
                    }
                    .labelsHidden().frame(maxWidth: 280)
                    .disabled(m.running)
                }
                LinkButton(text: "Look again") { m.loadDevices() }
            }
        case "window":
            if !m.screenGranted {
                Card(border: Color(hex: 0xFFC24B, alpha: 0.4)) {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Screen Recording", systemImage: "rectangle.dashed.badge.record").font(.system(size: 13.5, weight: .semibold)).foregroundStyle(Color.text)
                        Small("SmoovPlay copies the app's window with macOS's screen capture, so it needs Screen Recording. macOS asks once. If you said Don't Allow, turn on SmoovPlay in System Settings > Privacy & Security > Screen & System Audio Recording, then quit and open SmoovPlay again.")
                        HStack {
                            Button("Allow") { m.askScreenPermission() }.buttonStyle(SPButton(kind: .primary))
                            Button("Open System Settings") { ScreenPermission.openSettings() }.buttonStyle(SPButton(kind: .secondary))
                            LinkButton(text: "Check again") { m.loadWindows() }
                        }
                    }
                }
                .frame(maxWidth: 560)
                .accessibilityIdentifier("screen-permission")
            } else {
                VStack(spacing: 6) {
                    HStack(spacing: 10) {
                        if !m.windowsLoaded {
                            Small("Looking for app windows…")
                        } else if m.windows.isEmpty {
                            Small(m.windowsError ?? "No app window found.", color: .warn).frame(maxWidth: 460)
                        } else {
                            Text("App").font(.system(size: 12.5)).foregroundStyle(Color.text2)
                            Picker("", selection: Binding(get: { m.chosenWindow?.id ?? 0 }, set: { m.chooseWindow($0) })) {
                                ForEach(m.windows) { w in Text(w.label).tag(w.id) }
                            }
                            .labelsHidden().frame(maxWidth: 440)
                            .disabled(m.running)
                            .accessibilityIdentifier("window-app")
                        }
                        LinkButton(text: "Look again") { m.loadWindows() }
                    }
                    Toggle("Leave out its title bar", isOn: Binding(get: { !m.prefs.has("window-keep-title") },
                                                                    set: { m.prefs.set("window-keep-title", !$0); m.objectWillChange.send() }))
                        .toggleStyle(.checkbox).font(.system(size: 12.5)).foregroundStyle(Color.text2)
                        .disabled(m.running)
                        .help("On for apps with a standard title bar (PS Remote Play, Moonlight). Turn it off for an app that draws under its title bar, like a browser's tabs.")
                }
            }
        default:
            EmptyView()
        }
    }
}

/// App window mode's keys, with what each action gets on this Mac (another app can have one). Clicking one and
/// pressing new keys changes it.
struct KeysCard: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        Card(padding: 14) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("SmoovPlay's keys while you play").font(.system(size: 13.5, weight: .semibold)).foregroundStyle(Color.text)
                    Spacer()
                    LinkButton(text: m.keysEditing ? "Done" : "Change keys") {
                        if m.running { m.showToast("Stop playing first, then change the keys", bad: true); return }
                        m.keysEditing.toggle()
                        m.keyListening = nil
                        m.keysProblem = nil
                    }
                }
                Small("The app keeps the keyboard, so M, F and Esc go to it. These work from anywhere:")
                ForEach(WindowKeys.actions, id: \.id) { a in
                    let k = m.keysGot[a.id] ?? nil
                    let listening = m.keyListening == a.id
                    HStack(spacing: 10) {
                        Button { if m.keysEditing { m.listenForKeys(a.id) } } label: {
                            Text(listening ? "Press the new keys…" : k ?? "None free")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(listening ? Color.accent : k == nil ? Color.warn : Color.text)
                                .padding(.horizontal, 9).frame(minWidth: 170, minHeight: 26)
                                .background(RoundedRectangle(cornerRadius: 6).fill(Color(hex: 0x20252E)))
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(listening ? Color.accent : Color(hex: 0x323A47)))
                        }
                        .buttonStyle(.plain)
                        .help(m.keysEditing ? "Click, then press the keys you'd like (Esc cancels)" : "")
                        Text(a.label).font(.system(size: 12.5)).foregroundStyle(Color.text2)
                        if m.keysEditing, WindowKeys.picked(m.prefs)[a.id] != nil { LinkButton(text: "Default") { m.defaultKey(a.id) } }
                    }
                }
                if !m.keysTaken.isEmpty {
                    Small(m.keysTaken.joined(separator: " and ") + (m.keysTaken.count == 1 ? " is" : " are") + " used by another app or macOS, so the next choice is used.", color: .warn)
                }
                if let p = m.keysProblem { Small(p, color: .bad) }
                if m.keysEditing {
                    HStack {
                        Small("Click one, then press its new keys with Control, Option or Command. If nothing happens, another app has those keys.", color: .muted)
                        LinkButton(text: "Reset all") { m.resetKeys() }
                    }
                } else {
                    Small("Control-Option-R saves the instant replay and Control-Option-B a before & after, from any app, while instant replay is on. Stop here ends it too.", color: .muted)
                }
            }
        }
        .frame(maxWidth: 560)
        .accessibilityIdentifier("window-keys")
    }
}

struct Chips: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        let hz = m.settingValue("hz")?.plain ?? "60"
        let on = m.settingValue("interp")?.bool != false
        let scaler = m.settingValue("scaler")?.plain ?? "fsr"
        FlowLayout(spacing: 8) {
            Chip(symbol: "speedometer", text: on ? hz + " fps" : "Frame generation off")
            Chip(symbol: "sparkles.tv", text: scaler == "vsr" ? "NVIDIA RTX VSR" : scaler == "bilinear" ? "Bilinear" : "AMD FSR 1")
            Chip(symbol: "cpu", text: "RIFE on Metal")
            if m.settingValue("look_on")?.bool == true {
                Chip(symbol: "camera.filters", text: m.looks?.entry(m.settingValue("look")?.string)?.name ?? (m.settingValue("look")?.plain.capitalized ?? "Look"))
            }
        }
        .help("Change these on the Settings page, or press M in the video window.")
    }
}

struct NoteLine: View {
    var text: String
    var color: Color
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle").foregroundStyle(color)
            Text(text).font(.system(size: 12.5)).foregroundStyle(Color.text).fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 560)
    }
}

struct ResultsCard: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        if let r = m.results {
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Overline(text: "TEST RESULTS")
                        Spacer()
                        Button { m.copyResults() } label: { Label("Copy results", systemImage: "doc.on.doc") }.buttonStyle(SPButton(kind: .ghost))
                        if let u = m.session?.resultsURL ?? m.options.showResults.map({ URL(fileURLWithPath: $0) }) {
                            Button("Full report") { NSWorkspace.shared.open(u) }.buttonStyle(SPButton(kind: .ghost))
                        }
                        Button { m.results = nil } label: { Image(systemName: "xmark") }.buttonStyle(SPButton(kind: .ghost))
                    }
                    Text(r.verdict).font(.system(size: 15, weight: .semibold)).foregroundStyle(Color.text).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("ResultsVerdict")
                    if !r.tiles.isEmpty {
                        HStack(spacing: 8) {
                            ForEach(r.tiles, id: \.label) { t in
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(t.label).font(.system(size: 11.5)).foregroundStyle(Color.muted)
                                    Text(t.value).font(.system(size: 19, weight: .semibold)).foregroundStyle(t.accent ? Color.accent : Color.text)
                                    Text(t.hint).font(.system(size: 11.5)).foregroundStyle(Color.text2).lineLimit(1)
                                }
                                .padding(.horizontal, 11).padding(.vertical, 8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(RoundedRectangle(cornerRadius: 9).fill(Color.chip))
                                .overlay(RoundedRectangle(cornerRadius: 9).stroke(Color.line2))
                            }
                        }
                    }
                    Small(r.note)
                }
            }
            .frame(maxWidth: 640)
            .accessibilityIdentifier("PlayResults")
        }
    }
}

struct Details: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button { m.showDetails.toggle() } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.down").rotationEffect(.degrees(m.showDetails ? 180 : 0))
                        Text("Show details")
                    }
                    .font(.system(size: 12.5)).foregroundStyle(Color.text2)
                }
                .buttonStyle(.plain)
                Spacer()
                if case .error = m.playState {
                    Button { m.copyDiagnostics() } label: { Label("Copy diagnostics", systemImage: "doc.on.doc") }.buttonStyle(SPButton(kind: .ghost))
                }
                Button { m.openLogs() } label: { Label("Logs", systemImage: "folder") }.buttonStyle(SPButton(kind: .ghost))
            }
            if m.showDetails {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 1) {
                            ForEach(Array(m.playLog.suffix(400).enumerated()), id: \.offset) { i, l in
                                Text(l).font(.system(size: 11, design: .monospaced)).foregroundStyle(Color.text2)
                                    .textSelection(.enabled).id(i)
                            }
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 220)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color(hex: 0x0A0C10)))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.line))
                    .onChange(of: m.playLog.count) { _, _ in proxy.scrollTo(min(399, m.playLog.count - 1), anchor: .bottom) }
                }
                .accessibilityIdentifier("PlayLog")
            }
        }
        .frame(maxWidth: 760)
    }
}

/// Lays its views out in rows, centred, starting a new row when one is full (the source pills, the chips).
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func rows(_ proposal: ProposedViewSize, _ subviews: Subviews) -> [[(Int, CGSize)]] {
        let width = proposal.width ?? .infinity
        var rows: [[(Int, CGSize)]] = [[]]
        var x: CGFloat = 0
        for (i, v) in subviews.enumerated() {
            let size = v.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > width { rows.append([]); x = 0 }
            rows[rows.count - 1].append((i, size))
            x += size.width + spacing
        }
        return rows
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let r = rows(proposal, subviews)
        let w = r.map { row in row.map(\.1.width).reduce(0, +) + spacing * CGFloat(max(0, row.count - 1)) }.max() ?? 0
        let h = r.map { $0.map(\.1.height).max() ?? 0 }.reduce(0, +) + spacing * CGFloat(max(0, r.count - 1))
        return CGSize(width: min(w, proposal.width ?? w), height: h)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in rows(ProposedViewSize(width: bounds.width, height: nil), subviews) {
            let rowW = row.map(\.1.width).reduce(0, +) + spacing * CGFloat(max(0, row.count - 1))
            let rowH = row.map(\.1.height).max() ?? 0
            var x = bounds.minX + max(0, (bounds.width - rowW) / 2)
            for (i, size) in row {
                subviews[i].place(at: CGPoint(x: x, y: y + (rowH - size.height) / 2), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += rowH + spacing
        }
    }
}
