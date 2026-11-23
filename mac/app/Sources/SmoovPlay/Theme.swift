// The look of the Windows launcher (installer/launcher/Ui.xaml's brushes), in SwiftUI: a dark window, cards, and
// SmoovPlay's amber accent.
import AppKit
import SmoovPlayKit
import SwiftUI

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255, opacity: alpha)
    }

    static let bg = Color(hex: 0x0E1014)
    static let rail = Color(hex: 0x0B0D11)
    static let card = Color(hex: 0x161A21)
    static let cardHover = Color(hex: 0x1B2029)
    static let chip = Color(hex: 0x171B22)
    static let line = Color(hex: 0x222833)
    static let line2 = Color(hex: 0x2D3440)
    static let text = Color(hex: 0xF3F5F9)
    static let text2 = Color(hex: 0xA9B1BF)
    static let muted = Color(hex: 0x6F788A)
    static let accent = Color(hex: 0xFFB02E)
    static let ink = Color(hex: 0x1A1206)
    static let good = Color(hex: 0x3DDC8E)
    static let warn = Color(hex: 0xFFC24B)
    static let bad = Color(hex: 0xFF6464)
    static let accentGrad = LinearGradient(colors: [Color(hex: 0xFFCB5C), Color(hex: 0xFF8A1F)], startPoint: .topLeading, endPoint: .bottomTrailing)
}

struct Card<Content: View>: View {
    var padding: CGFloat = 16
    var border: Color = .line
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 12).fill(Color.card))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(border, lineWidth: 1))
    }
}

struct Overline: View {
    var text: String
    var body: some View {
        Text(text).font(.system(size: 11, weight: .semibold)).foregroundStyle(Color.muted).tracking(0.6)
    }
}

struct PageTitle: View {
    var overline: String?
    var title: String
    var sub: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let overline { Overline(text: overline) }
            Text(title).font(.system(size: 28, weight: .semibold)).foregroundStyle(Color.text)
            if let sub { Text(sub).font(.system(size: 13.5)).foregroundStyle(Color.text2).fixedSize(horizontal: false, vertical: true) }
        }
    }
}

struct Small: View {
    var text: String
    var color: Color = .text2
    init(_ text: String, color: Color = .text2) { self.text = text; self.color = color }
    var body: some View {
        Text(text).font(.system(size: 12)).foregroundStyle(color).fixedSize(horizontal: false, vertical: true)
    }
}

enum ButtonKind { case primary, secondary, ghost, pill }

struct SPButton: ButtonStyle {
    var kind: ButtonKind = .secondary
    @Environment(\.isEnabled) var enabled
    func makeBody(configuration c: Configuration) -> some View {
        let label = c.label.font(.system(size: kind == .pill ? 12.5 : 13, weight: .semibold))
        Group {
            switch kind {
            case .primary:
                label.foregroundStyle(Color.ink).padding(.horizontal, 16).frame(height: 34)
                    .background(Capsule().fill(Color.accentGrad))
            case .secondary:
                label.foregroundStyle(Color.text).padding(.horizontal, 14).frame(height: 34)
                    .background(Capsule().fill(Color(hex: 0x20252E))).overlay(Capsule().stroke(Color.line2))
            case .ghost:
                label.foregroundStyle(Color.text2).padding(.horizontal, 12).frame(height: 32)
                    .background(Capsule().fill(c.isPressed ? Color.cardHover : Color.clear)).overlay(Capsule().stroke(Color.line2))
            case .pill:
                label.foregroundStyle(Color.accent).padding(.horizontal, 12).frame(height: 28)
                    .background(Capsule().fill(Color(hex: 0xFFB02E, alpha: 0.12))).overlay(Capsule().stroke(Color(hex: 0xFFB02E, alpha: 0.35)))
            }
        }
        .opacity(enabled ? (c.isPressed ? 0.8 : 1) : 0.4)
        .contentShape(Capsule())
    }
}

struct LinkButton: View {
    var text: String
    var action: () -> Void
    var body: some View {
        Button(action: action) { Text(text).font(.system(size: 12.5)).foregroundStyle(Color.accent).underline() }
            .buttonStyle(.plain)
    }
}

struct ExperimentalBadge: View {
    var body: some View {
        Text("EXPERIMENTAL").font(.system(size: 9.5, weight: .semibold)).foregroundStyle(Color.warn)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(RoundedRectangle(cornerRadius: 4).fill(Color(hex: 0xFFC24B, alpha: 0.15)))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color(hex: 0xFFC24B, alpha: 0.4)))
            .fixedSize()
            .help("Experimental: built from the best information we have, without the hardware to test every setup. Tell us how it goes with Send feedback.")
    }
}

/// A row of two to four choices, the picked one lit (the Settings page's SegBar).
struct Segments: View {
    var choices: [ChoiceModel]
    var enabled: Bool
    var pick: (String) -> Void
    var body: some View {
        HStack(spacing: 2) {
            ForEach(choices) { c in
                Button { pick(c.tag) } label: {
                    Text(c.name).font(.system(size: 12.5, weight: c.selected ? .semibold : .regular))
                        .foregroundStyle(c.selected ? Color.ink : Color.text2)
                        .padding(.horizontal, 12).frame(height: 28)
                        .background(RoundedRectangle(cornerRadius: 7).fill(c.selected ? AnyShapeStyle(Color.accentGrad) : AnyShapeStyle(Color.clear)))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(!c.enabled)
                .opacity(c.enabled || c.selected ? 1 : 0.4)
                .help(c.tip ?? "")
                .accessibilityIdentifier("Set_\(c.tag)")
            }
        }
        .padding(3)
        .background(RoundedRectangle(cornerRadius: 9).fill(Color(hex: 0x10131A)))
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(Color.line2))
        .opacity(enabled ? 1 : 0.7)
    }
}

struct Chip: View {
    var symbol: String
    var text: String
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: symbol).font(.system(size: 11)).foregroundStyle(Color.accent)
            Text(text).font(.system(size: 12.5)).foregroundStyle(Color.text)
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(Capsule().fill(Color.chip)).overlay(Capsule().stroke(Color.line2))
    }
}

/// A key cap, for the keys legend.
struct KeyCap: View {
    var key: String
    var body: some View {
        Text(key).font(.system(size: 12, weight: .semibold)).foregroundStyle(Color.text)
            .padding(.horizontal, 7).frame(minWidth: 26, minHeight: 26)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color(hex: 0x20252E)))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(hex: 0x323A47)))
    }
}
