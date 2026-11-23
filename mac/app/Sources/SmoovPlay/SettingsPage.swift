// The Settings page (SettingsPage.cs, AutoDetect.cs, FillPreview.cs), drawn from SettingsModel. That model is built
// from the pipeline's own list of settings and the core's live state.
import SmoovPlayKit
import SwiftUI

struct SettingsPage: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top) {
                    PageTitle(overline: "SETTINGS", title: "Settings", sub: nil)
                    Spacer()
                    if m.liveState != nil {
                        HStack(spacing: 6) {
                            Circle().fill(Color.good).frame(width: 7, height: 7)
                            Text("LIVE").font(.system(size: 11, weight: .bold)).foregroundStyle(Color.good)
                        }
                        .padding(.horizontal, 9).padding(.vertical, 4)
                        .overlay(Capsule().stroke(Color(hex: 0x3DDC8E, alpha: 0.5)))
                        .accessibilityIdentifier("SetLive")
                    }
                }
                if let model = m.settingsModel {
                    let page = model.page()
                    Small(page.status).accessibilityIdentifier("SetStatus")
                    if let banner = page.restartBanner {
                        Card(border: Color(hex: 0xFFB02E, alpha: 0.5)) {
                            HStack {
                                Small(banner, color: .text)
                                Spacer()
                                Button("Restart now") { m.restartNow() }.buttonStyle(SPButton(kind: .primary))
                            }
                        }
                        .accessibilityIdentifier("SetRestart")
                    }
                    Card(padding: 0) {
                        VStack(spacing: 0) {
                            ForEach(Array(page.quick.filter(\.visible).enumerated()), id: \.element.id) { i, row in
                                if i > 0 { Divider().overlay(Color.line).padding(.horizontal, 18) }
                                SettingRowView(row: row, quick: true)
                            }
                        }
                    }
                    DetectCard()
                    let shown = page.sections.filter(\.visible)
                    let cols = [GridItem(.adaptive(minimum: 400), spacing: 14, alignment: .top)]
                    LazyVGrid(columns: cols, alignment: .leading, spacing: 14) {
                        ForEach(shown) { sec in SectionCard(section: sec) }
                    }
                } else {
                    Small(m.specsError.map { "Couldn't load the settings: \($0)." } ?? (m.paths.runtimeReady ? "Loading..." : "There's no pipeline at \(m.paths.repo.path). Make the repo's .venv first."),
                          color: m.specsError == nil ? .text2 : .bad)
                }
            }
            .padding(.horizontal, 40).padding(.vertical, 34)
            .frame(maxWidth: 1100, alignment: .leading)
        }
    }
}

struct SectionCard: View {
    @EnvironmentObject var m: AppModel
    var section: SectionModel
    static let icons = ["Smoothness": "speedometer", "Picture": "sparkles.tv", "Display": "display", "Instant replay": "gobackward",
                        "Looks": "camera.filters", "PS5 Remote Play": "gamecontroller"]

    var body: some View {
        let open = m.openSections.contains(section.group)
        Card(padding: 0) {
            VStack(spacing: 0) {
                Button { withAnimation(.easeOut(duration: 0.18)) { m.toggleSection(section.group) } } label: {
                    HStack(spacing: 14) {
                        Image(systemName: SectionCard.icons[section.group] ?? "slider.horizontal.3").font(.system(size: 14))
                            .foregroundStyle(Color.accent).frame(width: 32, height: 32)
                            .background(RoundedRectangle(cornerRadius: 9).fill(Color(hex: 0x1E222B)))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(section.group).font(.system(size: 14, weight: .semibold)).foregroundStyle(Color.text)
                            Text(section.summary).font(.system(size: 12)).foregroundStyle(section.changedCount > 0 ? Color.text2 : Color.muted).lineLimit(1)
                        }
                        Spacer()
                        Image(systemName: "chevron.down").font(.system(size: 12)).foregroundStyle(Color.text2).rotationEffect(.degrees(open ? 180 : 0))
                    }
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("SetFold_" + section.group.replacingOccurrences(of: " ", with: ""))
                if open {
                    Divider().overlay(Color.line)
                    ForEach(Array(section.rows.filter(\.visible).enumerated()), id: \.element.id) { i, row in
                        if i > 0 { Divider().overlay(Color.line).padding(.horizontal, 18) }
                        SettingRowView(row: row, quick: false)
                    }
                }
            }
        }
    }
}

struct SettingRowView: View {
    @EnvironmentObject var m: AppModel
    var row: RowModel
    var quick: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(row.label).font(.system(size: quick ? 14.5 : 14, weight: .semibold)).foregroundStyle(Color.text)
                        if quick {
                            Image(systemName: "info.circle").font(.system(size: 12)).foregroundStyle(Color.muted)
                                .help(row.help + impact + (row.later.isEmpty ? "" : "\n\n" + row.later))
                        }
                        if row.experimental {
                            ExperimentalBadge()
                            Button("Send feedback") { m.sendFeedback(row.label) }.buttonStyle(SPButton(kind: .pill))
                        }
                        if let t = row.restartTag {
                            Text(t).font(.system(size: 10.5)).foregroundStyle(Color.muted).padding(.horizontal, 7).padding(.vertical, 1)
                                .overlay(Capsule().stroke(Color.line2))
                                .help(t == "next Play" ? "While playing, this one changes the next time you press Play." : "While playing, this one changes after a restart (Restart now, at the top).")
                        }
                    }
                    if quick, row.kind == .segments, !row.hint.isEmpty { Small(row.hint) }
                }
                Spacer(minLength: 12)
                if quick || row.kind != .segments { control }
            }
            if !quick {
                Small(row.help)
                if row.picture != nil || row.speed != nil {
                    VStack(alignment: .leading, spacing: 3) {
                        if let p = row.picture { impactLine("sparkles.tv", "Picture", p) }
                        if let s = row.speed {
                            impactLine("speedometer", "Speed", s)
                                .help(m.specs?.impactPC.map { "Rough figures, measured on \($0). Yours will differ." } ?? "")
                        }
                    }
                    .padding(.top, 2)
                }
                if row.kind == .segments {
                    control.padding(.top, 4)
                    if row.key == "fill" {
                        HStack(alignment: .top, spacing: 18) {
                            FillPreview(mode: row.choices.first(where: \.selected)?.value?.string ?? "fit")
                            Small(row.hint, color: .text).padding(.top, 8)
                        }
                        .padding(.top, 4)
                    } else if !row.hint.isEmpty {
                        Small(row.hint, color: .text).padding(.top, 2)
                    }
                }
            }
            if let n = row.note { Small(n, color: .warn) }
        }
        .padding(.horizontal, 18).padding(.vertical, quick ? 16 : 14)
        .accessibilityIdentifier("Row_" + row.key)
    }

    var impact: String {
        (row.picture.map { "\n\nPicture: " + $0 } ?? "") + (row.speed.map { (row.picture != nil ? "\n" : "\n\n") + "Speed: " + $0 } ?? "")
    }

    func impactLine(_ symbol: String, _ name: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: symbol).font(.system(size: 11)).foregroundStyle(Color.accent).frame(width: 14)
            (Text(name + "  ").font(.system(size: 12, weight: .semibold)).foregroundColor(.text) + Text(text).font(.system(size: 12)).foregroundColor(.text2))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder var control: some View {
        switch row.kind {
        case .toggle:
            Toggle("", isOn: Binding(get: { row.isOn }, set: { m.change(row.key, .bool($0)) }))
                .toggleStyle(.switch).labelsHidden().tint(Color.accent)
                .disabled(!row.enabled)
                .accessibilityIdentifier("Set_" + row.key)
        case .segments:
            Segments(choices: row.choices, enabled: row.enabled) { tag in m.pick(row.key, tag: tag) }
                .disabled(!row.enabled)
                .accessibilityIdentifier("Set_" + row.key)
        case .dropdown:
            Menu {
                ForEach(row.choices) { c in
                    Button { m.pick(row.key, tag: c.tag) } label: {
                        Text(c.name + (c.selected ? "  ✓" : ""))
                    }
                    .disabled(!c.enabled)
                    .help(c.hint)
                }
                if !row.later.isEmpty {
                    Divider()
                    Text(row.later).italic()
                }
            } label: {
                Text(row.choices.first(where: \.selected)?.name ?? "").font(.system(size: 13)).foregroundStyle(Color.text)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .padding(.horizontal, 10).frame(height: 30)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0x10131A)))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.line2))
            .disabled(!row.enabled)
            .help(row.choices.first(where: \.selected)?.hint ?? "")
            .accessibilityIdentifier("Set_" + row.key)
        }
    }
}

struct DetectCard: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Auto-detect").font(.system(size: 14.5, weight: .semibold)).foregroundStyle(Color.text)
                        Small("Times this Mac's GPU (about 10 seconds), reads the display and suggests settings, with the reason for each. Nothing changes until you press Apply.")
                    }
                    Spacer()
                    Button(m.detecting ? "Testing..." : (m.detectResult != nil || m.detectError != nil) ? "Run again" : "Auto-detect") { m.runDetect() }
                        .buttonStyle(SPButton(kind: .secondary))
                        .disabled(m.detecting || m.playState.running || m.specs == nil)
                        .help(m.playState.running ? "Stop playing first: the test needs the GPU to itself." : "")
                        .accessibilityIdentifier("BtnDetect")
                }
                if m.detecting {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView().progressViewStyle(.linear).tint(Color.accent).frame(width: 260)
                        Small(m.detectProgress + "...").accessibilityIdentifier("DetectProgress")
                    }
                } else if let e = m.detectError {
                    Small(e, color: .bad)
                } else if let r = m.detectResult {
                    results(r)
                }
            }
        }
    }

    @ViewBuilder func results(_ r: DetectResult) -> some View {
        let changes = m.detectChanges
        let changedKeys = Set(changes.map(\.0.key))
        VStack(alignment: .leading, spacing: 8) {
            Overline(text: "FOUND ON THIS MAC")
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                ForEach(r.found, id: \.label) { f in
                    GridRow {
                        Text(f.label).font(.system(size: 12.5)).foregroundStyle(Color.muted)
                        (Text(f.value).foregroundColor(.text) + Text(f.detail.isEmpty ? "" : "  ·  " + f.detail).foregroundColor(.text2))
                            .font(.system(size: 12.5)).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            Overline(text: m.detectApplied ? "APPLIED" : "SUGGESTED").padding(.top, 8)
            ForEach(m.specs?.specs.filter { s in r.settings.contains(where: { $0.0 == s.key }) } ?? [], id: \.key) { spec in
                let suggested = r.settings.first(where: { $0.0 == spec.key })!.1
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(spec.label).font(.system(size: 13, weight: .semibold)).foregroundStyle(Color.text).frame(width: 150, alignment: .leading)
                        if changedKeys.contains(spec.key), let cur = m.settingsModel?.value(spec) {
                            (Text(spec.name(cur)).foregroundColor(.muted).strikethrough() + Text("  →  ").foregroundColor(.muted)
                                + Text(spec.name(suggested)).foregroundColor(.accent).bold()).font(.system(size: 13))
                        } else {
                            let changed = m.detectApplied && m.detectChanged.contains(spec.key)
                            (Text(spec.name(suggested)).foregroundColor(.text) + Text(changed ? "   changed" : "   already set").foregroundColor(changed ? .good : .muted).font(.system(size: 12)))
                                .font(.system(size: 13))
                        }
                    }
                    .accessibilityIdentifier("Detect_" + spec.key)
                    if let why = r.reasons[spec.key] { Small(why) }
                }
            }
            ForEach(r.notes, id: \.self) { n in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(Color.warn)
                    Small(n, color: .text)
                }
            }
            HStack(spacing: 10) {
                if m.detectApplied {
                    Image(systemName: "checkmark").foregroundStyle(Color.good)
                    Small("Done. Fine-tune anything below; Auto-detect won't change it again unless you run it.", color: .text)
                    Button("Hide") { m.dismissDetect() }.buttonStyle(SPButton(kind: .ghost)).accessibilityIdentifier("BtnDetectDismiss")
                } else if changes.isEmpty {
                    Small("Your settings already match this Mac.", color: .text)
                    Button("Hide") { m.dismissDetect() }.buttonStyle(SPButton(kind: .ghost)).accessibilityIdentifier("BtnDetectDismiss")
                } else {
                    Button(changes.count == 1 ? "Apply this change" : "Apply \(changes.count) changes") { m.applyDetect() }
                        .buttonStyle(SPButton(kind: .primary))
                        .disabled(m.playState.running && m.liveState == nil)
                        .accessibilityIdentifier("BtnDetectApply")
                    Button("Not now") { m.dismissDetect() }.buttonStyle(SPButton(kind: .ghost)).accessibilityIdentifier("BtnDetectDismiss")
                }
            }
            .padding(.top, 6)
        }
    }
}

/// The Fill the screen preview. It shows what the picked choice does with the console's 16:9 picture on this Mac's
/// widest screen, or on an example 21:9 one if none is wider than 16:9. The maths is fill.py's. Zoom fills the width
/// and cuts the top and bottom, and Panorama bends the picture across the screen with u = atan(t*x) / atan(t).
struct FillPreview: View {
    var mode: String
    let screenH: CGFloat = 46

    var body: some View {
        let own = MacInfo.widestRatio > 1.85
        let ratio = own ? MacInfo.widestRatio : 21.0 / 9.0
        let fw = (screenH * ratio).rounded()
        let pad = max(9, ((fw * 9 / 16 - screenH) / 2).rounded(.up) + 3)
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Canvas { ctx, size in
                    let sw: CGFloat = 53, sh: CGFloat = 30
                    let top = pad + (screenH - sh) / 2
                    FillPreview.picture(ctx) { p in CGPoint(x: p.x * sw, y: top + p.y * sh) }
                    ctx.stroke(Path(CGRect(x: 0, y: top, width: sw, height: sh)), with: .color(.text2), lineWidth: 1)
                }
                .frame(width: 53, height: screenH + 2 * pad)
                Text("From the console: 16:9").font(.system(size: 10.5)).foregroundStyle(Color.muted)
            }
            Text("→").font(.system(size: 16)).foregroundStyle(Color.muted).padding(.top, pad + screenH / 2 - 12)
            VStack(alignment: .leading, spacing: 1) {
                Canvas { ctx, _ in draw(ctx, fw: fw, ratio: ratio, pad: pad) }
                    .frame(width: fw, height: screenH + 2 * pad)
                let shape = ratio >= 3.2 ? "32:9" : ratio >= 2.2 ? "21:9" : nil
                Text(own ? "Your \(shape ?? "wide") screen" : "On a 21:9 ultrawide").font(.system(size: 10.5)).foregroundStyle(Color.muted)
            }
        }
        .accessibilityLabel("Fill the screen preview")
    }

    func draw(_ ctx: GraphicsContext, fw: CGFloat, ratio: CGFloat, pad: CGFloat) {
        let frame = CGRect(x: 0, y: pad, width: fw, height: screenH)
        ctx.fill(Path(frame), with: .color(Color(hex: 0x050608)))
        var map: (CGPoint) -> CGPoint
        if mode == "zoom" {
            let ph = fw * 9 / 16, top = pad + (screenH - ph) / 2
            map = { p in CGPoint(x: p.x * fw, y: top + p.y * ph) }
            var faint = ctx
            faint.opacity = 0.4
            FillPreview.picture(faint, map)
            ctx.stroke(Path(CGRect(x: 0, y: top, width: fw, height: ph)), with: .color(.muted), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
        } else if mode == "stretch" {
            map = { p in CGPoint(x: p.x * fw, y: pad + p.y * self.screenH) }
        } else if mode == "panorama" && ratio > 16.0 / 9 * 1.01 {
            let t = FillPreview.panoramaT(ratio / (16.0 / 9)), at = atan(t)
            map = { p in CGPoint(x: fw / 2 * (1 + tan((2 * p.x - 1) * at) / t), y: pad + p.y * self.screenH) }
        } else {
            let pw = screenH * 16 / 9, left = (fw - pw) / 2
            map = { p in CGPoint(x: left + p.x * pw, y: pad + p.y * self.screenH) }
        }
        var inside = ctx
        inside.clip(to: Path(frame))
        FillPreview.picture(inside, map)
        ctx.stroke(Path(frame), with: .color(.text2), lineWidth: 1)
    }

    /// fill.py's panorama_t, the curve's t that makes the middle 15% of the extra width wider than true.
    static func panoramaT(_ k: Double) -> Double {
        let goal = (1.0 + 0.15 * (k - 1.0)) / k
        var lo = 1e-6, hi = 1e3
        for _ in 0..<80 {
            let mid = (lo + hi) / 2
            if atan(mid) / mid > goal { lo = mid } else { hi = mid }
        }
        return (lo + hi) / 2
    }

    /// A stand-in game picture: sky and ground, evenly spaced lines, a HUD in the top left (which Zoom cuts) and a
    /// round sun (which Stretch makes oval).
    static func picture(_ ctx: GraphicsContext, _ map: (CGPoint) -> CGPoint) {
        func quad(_ x0: CGFloat, _ y0: CGFloat, _ x1: CGFloat, _ y1: CGFloat, _ c: Color) {
            var p = Path()
            let n = 24
            for i in 0...n { let pt = map(CGPoint(x: x0 + (x1 - x0) * CGFloat(i) / CGFloat(n), y: y0)); i == 0 ? p.move(to: pt) : p.addLine(to: pt) }
            for i in stride(from: n, through: 0, by: -1) { p.addLine(to: map(CGPoint(x: x0 + (x1 - x0) * CGFloat(i) / CGFloat(n), y: y1))) }
            p.closeSubpath()
            ctx.fill(p, with: .color(c))
        }
        quad(0, 0, 1, 0.62, Color(hex: 0x2B3B55))
        quad(0, 0.62, 1, 1, Color(hex: 0x3A342A))
        for i in 1..<8 {
            var l = Path()
            l.move(to: map(CGPoint(x: CGFloat(i) / 8, y: 0)))
            l.addLine(to: map(CGPoint(x: CGFloat(i) / 8, y: 1)))
            ctx.stroke(l, with: .color(.white.opacity(0.27)), lineWidth: 1)
        }
        var sun = Path()
        for i in 0..<48 {
            let a = CGFloat(i) * .pi * 2 / 48
            let pt = map(CGPoint(x: 0.5 + 0.2 * 9 / 16 * cos(a), y: 0.5 + 0.2 * sin(a)))
            i == 0 ? sun.move(to: pt) : sun.addLine(to: pt)
        }
        sun.closeSubpath()
        ctx.fill(sun, with: .color(Color.text.opacity(0.92)))
        quad(0.04, 0.07, 0.24, 0.17, .accent)
    }
}
