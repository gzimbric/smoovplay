// The Looks page (LooksPage.cs, LookPreview.cs): the looks on the left, the picked one's name, buttons, preview
// and effects on the right. Holding the mouse on the preview shows the picture without the look.
import SmoovPlayKit
import SwiftUI

struct LooksPage: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                PageTitle(overline: "LOOKS", title: "Looks", sub: status)
                Spacer()
                if m.looksLive {
                    HStack(spacing: 6) {
                        Circle().fill(Color.good).frame(width: 7, height: 7)
                        Text("LIVE").font(.system(size: 11, weight: .bold)).foregroundStyle(Color.good)
                    }
                    .padding(.horizontal, 9).padding(.vertical, 4).overlay(Capsule().stroke(Color(hex: 0x3DDC8E, alpha: 0.5)))
                }
                if m.looks != nil {
                    Button { m.newLook() } label: { Label("New look", systemImage: "plus") }.buttonStyle(SPButton(kind: .secondary))
                    Button { m.pasteLookCode() } label: { Label("Paste a code", systemImage: "doc.on.clipboard") }.buttonStyle(SPButton(kind: .ghost))
                }
            }
            if let lib = m.looks {
                HStack(alignment: .top, spacing: 18) {
                    LookList(lib: lib).frame(width: 230)
                    if let e = m.pickedLook, let i = m.pickedIndex {
                        LookEditor(lib: lib, entry: e, index: i)
                    }
                }
            }
        }
        .padding(.horizontal, 40).padding(.vertical, 34)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    var status: String {
        if let e = m.looksError { return "Couldn't load the looks: \(e)." }
        if !m.paths.runtimeReady { return "There's no pipeline at \(m.paths.repo.path)." }
        if m.looks == nil { return "Loading..." }
        if m.looksLive { return "The video shows the look you pick here, as you change it. Use this look keeps it on after you leave this page." }
        return "Effects on top of the picture, applied top to bottom. Press Play to see changes live. While you play, K turns the look on and off and J goes to the next one."
    }
}

struct LookList: View {
    @EnvironmentObject var m: AppModel
    var lib: LooksLibrary

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 2) {
                heading("SMOOVPLAY'S LOOKS", top: 0)
                ForEach(lib.entries.filter(\.builtin)) { e in item(e) }
                heading("YOURS", top: 14)
                let mine = lib.entries.filter { !$0.builtin }
                ForEach(mine) { e in item(e) }
                if mine.isEmpty { Small("Change a look and press Save as mine, press New look, or paste a code someone shared.", color: .muted).padding(.horizontal, 10) }
            }
        }
    }

    func heading(_ t: String, top: CGFloat) -> some View {
        Text(t).font(.system(size: 10.5, weight: .semibold)).foregroundStyle(Color.muted).padding(.leading, 10).padding(.top, top).padding(.bottom, 4)
    }

    func item(_ e: LookEntry) -> some View {
        let picked = e.id == m.lookPicked
        return Button { m.pickLook(e.id) } label: {
            HStack {
                Text(e.name).font(.system(size: 13.5, weight: picked ? .semibold : .regular)).foregroundStyle(picked ? Color.text : Color.text2).lineLimit(1)
                Spacer()
                if lib.changed(e.id) { Text("changed").font(.system(size: 11)).italic().foregroundStyle(Color.muted) }
                if m.lookInUse(e.id) { Circle().fill(Color.good).frame(width: 7, height: 7).help("On now") }
            }
            .padding(.horizontal, 10).padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 8).fill(picked ? Color(hex: 0x232A36) : Color.clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("Look_" + e.id)
    }
}

struct LookEditor: View {
    @EnvironmentObject var m: AppModel
    var lib: LooksLibrary
    var entry: LookEntry
    var index: Int
    @State private var name = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                head
                LookPreviewCard(entry: entry)
                ForEach(Array(entry.nodes.enumerated()), id: \.offset) { i, node in
                    if let t = lib.type(of: node) { NodeCard(lib: lib, entry: entry, node: node, i: i, type: t) }
                }
                if entry.nodes.isEmpty { Small("No effects yet. Add one below.") }
                Menu {
                    ForEach(lib.types) { t in Button(t.label) { m.editLook { lib, i in lib.addNode(i, t) }; m.lookOpen[entry.id, default: []].insert(entry.nodes.count) }.help(t.help) }
                } label: { Label("Add an effect", systemImage: "plus") }
                    .menuStyle(.borderlessButton).fixedSize()
                    .disabled(entry.nodes.count >= lib.maxNodes)
                    .help(entry.nodes.count >= lib.maxNodes ? "A look can have up to \(lib.maxNodes) effects." : "")
            }
            .padding(.trailing, 6)
        }
        .onAppear { name = entry.name }
        .onChange(of: entry.id) { _, _ in name = entry.name }
    }

    var head: some View {
        let changed = lib.changed(entry.id)
        let inUse = m.lookInUse(entry.id)
        return HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                if entry.builtin {
                    Text(entry.name).font(.system(size: 21, weight: .semibold)).foregroundStyle(Color.text)
                } else {
                    TextField("Look name", text: $name).textFieldStyle(.roundedBorder).font(.system(size: 16)).frame(maxWidth: 320)
                        .onSubmit { m.renameLook(name) }
                }
                Small(changed ? "Changed here, not saved. Save as mine keeps it as a look of your own; Reset puts SmoovPlay's back."
                      : entry.builtin ? "One of SmoovPlay's looks. Try any change: Save as mine keeps it as a look of your own."
                      : "Yours. Changes are saved as you make them.")
            }
            Spacer()
            if changed {
                Button("Save as mine") { m.saveAsMine() }.buttonStyle(SPButton(kind: .primary))
                    .help("Keeps this as a look of your own (My \(entry.name)), to use, change more and share. \(entry.name) stays as it was.")
                Button("Reset") { m.resetLook() }.buttonStyle(SPButton(kind: .ghost))
            } else {
                Button(inUse ? "Turn the look off" : "Use this look") { m.useLook(!inUse) }
                    .buttonStyle(SPButton(kind: inUse ? .secondary : .primary))
                    .disabled(m.playState.running && !m.looksLive)
                    .help(inUse ? "Plays without a look. K in the video window turns it back on." : "Turns this look on, now and next time. K in the video window turns it off and on.")
            }
            Button("Copy code") { m.copyLookCode() }.buttonStyle(SPButton(kind: .ghost))
                .help("Copies a short code for this look. Anyone with SmoovPlay can paste it on their Looks page to get the same look.")
            if !entry.builtin {
                Button { m.deleteLook() } label: { Image(systemName: "trash") }.buttonStyle(SPButton(kind: .ghost)).help("Delete this look")
            }
        }
    }
}

struct LookPreviewCard: View {
    @EnvironmentObject var m: AppModel
    var entry: LookEntry

    var body: some View {
        if let s = m.currentSample {
            VStack(alignment: .leading, spacing: 6) {
                GeometryReader { g in
                    let look = !m.previewHolding && m.previewImage != nil && m.previewFor == s.id
                    ZStack(alignment: .topLeading) {
                        if look, let img = m.previewImage {
                            Image(nsImage: img).resizable().aspectRatio(16 / 9, contentMode: .fit)
                        } else if let img = NSImage(contentsOfFile: s.path) {
                            Image(nsImage: img).resizable().aspectRatio(16 / 9, contentMode: .fit)
                        }
                        Text(look ? entry.name.uppercased() : "ORIGINAL").font(.system(size: 10.5, weight: .bold)).foregroundStyle(Color.text)
                            .padding(.horizontal, 7).padding(.vertical, 2).background(Capsule().fill(.black.opacity(0.55))).padding(8)
                    }
                    .frame(width: g.size.width, height: g.size.width * 9 / 16)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .gesture(DragGesture(minimumDistance: 0).onChanged { _ in m.previewHolding = true }.onEnded { _ in m.previewHolding = false })
                    .onAppear { setSize(g.size.width) }
                    .onChange(of: g.size.width) { _, w in setSize(w) }
                }
                .aspectRatio(16 / 9, contentMode: .fit)
                .frame(maxWidth: 720)
                HStack {
                    Text("\(s.game)  ·  \(s.scene)").font(.system(size: 12)).foregroundStyle(Color.text2)
                    Spacer()
                    Text(s.credit.contains("©") ? String(s.credit[s.credit.firstIndex(of: "©")!...]) : s.credit).font(.system(size: 11)).foregroundStyle(Color.muted)
                        .help("\(s.credit). A screenshot from the game, to show what the look does; not part of SmoovPlay's licence.")
                    Button { m.stepSample(-1) } label: { Image(systemName: "chevron.left") }.buttonStyle(.plain).foregroundStyle(Color.text2)
                    Text("\((m.samples.firstIndex(of: s) ?? 0) + 1) / \(m.samples.count)").font(.system(size: 11.5)).foregroundStyle(Color.muted)
                    Button { m.stepSample(1) } label: { Image(systemName: "chevron.right") }.buttonStyle(.plain).foregroundStyle(Color.text2)
                }
                .frame(maxWidth: 720)
            }
        }
    }

    func setSize(_ w: CGFloat) {
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let pw = max(64, min(1920, Int((w * scale / 2).rounded()) * 2))
        let size = (pw, Int((Double(pw) * 9 / 16).rounded()))
        if size != m.previewPixels { m.previewPixels = size; m.requestPreview() }
    }
}

struct NodeCard: View {
    @EnvironmentObject var m: AppModel
    var lib: LooksLibrary
    var entry: LookEntry
    var node: JSON
    var i: Int
    var type: LookType

    var body: some View {
        let on = node["on"]?.bool ?? true
        let open = m.lookOpen[entry.id]?.contains(i) ?? (entry.nodes.count <= 2)
        Card(padding: 12) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 12) {
                    Text("\(i + 1)").font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Color.text2).frame(width: 24, height: 24)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Color(hex: 0x1E222B)))
                    Toggle("", isOn: Binding(get: { on }, set: { v in m.editLook { lib, ei in lib.setValue(ei, node: i, key: "on", .bool(v)) } }))
                        .toggleStyle(.switch).labelsHidden().controlSize(.small).tint(Color.accent).help("This effect on or off")
                    Button {
                        var s = m.lookOpen[entry.id] ?? (entry.nodes.count <= 2 ? Set(entry.nodes.indices) : [])
                        if s.contains(i) { s.remove(i) } else { s.insert(i) }
                        m.lookOpen[entry.id] = s
                    } label: {
                        HStack(spacing: 8) {
                            Text(type.label).font(.system(size: 14, weight: .semibold)).foregroundStyle(Color.text)
                            Image(systemName: open ? "chevron.up" : "chevron.down").font(.system(size: 10)).foregroundStyle(Color.muted)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(type.help)
                    Spacer()
                    Button { m.editLook { lib, ei in lib.moveNode(ei, node: i, by: -1) } } label: { Image(systemName: "arrow.up") }
                        .buttonStyle(.plain).disabled(i == 0).help("Move up")
                    Button { m.editLook { lib, ei in lib.moveNode(ei, node: i, by: 1) } } label: { Image(systemName: "arrow.down") }
                        .buttonStyle(.plain).disabled(i >= entry.nodes.count - 1).help("Move down")
                    Button { m.editLook { lib, ei in lib.removeNode(ei, node: i) } } label: { Image(systemName: "xmark") }
                        .buttonStyle(.plain).help("Remove this effect")
                }
                .foregroundStyle(Color.text2)
                if open {
                    VStack(alignment: .leading, spacing: 6) {
                        Small(type.help, color: .muted)
                        ForEach(type.params, id: \.key) { p in ParamRow(p: p, node: node, i: i, lib: lib) }
                    }
                    .padding(.leading, 36)
                }
            }
        }
        .opacity(on ? 1 : 0.6)
    }
}

struct ParamRow: View {
    @EnvironmentObject var m: AppModel
    var p: LookParam
    var node: JSON
    var i: Int
    var lib: LooksLibrary

    var body: some View {
        HStack(spacing: 10) {
            Text(p.label).font(.system(size: 12.5)).foregroundStyle(Color.text2).frame(width: 150, alignment: .leading).lineLimit(1)
            if p.text {
                let current = node[p.key]?.string ?? ""
                Menu {
                    ForEach(lib.luts, id: \.self) { n in Button(n) { set(.string(n)) } }
                    if !current.isEmpty && !lib.luts.contains(current) { Button(current + " (missing)") {} }
                    Divider()
                    Button("Add a .cube file...") { m.addLut(node: i) }
                } label: { Text(current.isEmpty ? "Pick a .cube file" : current).font(.system(size: 12.5)) }
                    .menuStyle(.borderlessButton).fixedSize()
            } else if !p.names.isEmpty {
                let v = node[p.key]?.double ?? p.defaultValue
                Segments(choices: p.names.enumerated().map { k, n in
                    ChoiceModel(tag: String(k), value: .num(p.min + Double(k)), name: n, hint: "", enabled: true, tip: nil,
                                selected: abs(p.min + Double(k) - v) < 1e-6)
                }, enabled: true) { tag in set(.num(p.min + Double(Int(tag) ?? 0))) }
            } else {
                let v = node[p.key]?.double ?? p.defaultValue
                Slider(value: Binding(get: { v }, set: { set(.num(p.snap($0))) }), in: p.min...p.max)
                    .tint(Color.accent)
                    .onTapGesture(count: 2) { set(.num(p.defaultValue)) }
                    .help("Double-click: back to " + p.format(p.defaultValue))
                HStack(spacing: 6) {
                    if p.unit == "°" { Circle().fill(Color(hue: (v.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360) / 360, saturation: 1, brightness: 1)).frame(width: 10, height: 10) }
                    Text(p.format(v)).font(.system(size: 12.5, design: .monospaced)).foregroundStyle(Color.text)
                }
                .frame(width: 86, alignment: .trailing)
            }
        }
    }

    func set(_ v: JSON) { m.editLook { lib, ei in lib.setValue(ei, node: i, key: p.key, v) } }
}
