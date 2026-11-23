// JSON as the core writes it. Objects keep their key order, so the settings file is rewritten in the core's own
// layout. true and 1 stay different, where JSONSerialization makes both an NSNumber. A number keeps the text it came
// as, so a choice goes back to the core exactly as it was given ("set fsr_sharpness 1.0"). Numbers compare as
// numbers, so 1 and 1.0 are the same choice, as remote.py treats them.
import Foundation

public enum JSON: Equatable, Hashable, CustomStringConvertible, Sendable {
    case null
    case bool(Bool)
    case number(Double, String)  // the value, and its text
    case string(String)
    case array([JSON])
    case object([(String, JSON)])

    public static func num(_ v: Double) -> JSON { .number(v, JSON.format(v)) }
    public static func int(_ v: Int) -> JSON { .number(Double(v), String(v)) }

    static func format(_ v: Double) -> String {
        if v.isFinite, v == v.rounded(), abs(v) < 1e15 { return String(Int64(v)) }
        return String(v)
    }

    // MARK: equality (numbers by value, objects regardless of order)
    public static func == (a: JSON, b: JSON) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.number(x, _), .number(y, _)): return x == y
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)):
            guard x.count == y.count else { return false }
            let d = Dictionary(y, uniquingKeysWith: { a, _ in a })
            return x.allSatisfy { k, v in d[k] == v }
        default: return false
        }
    }

    public func hash(into h: inout Hasher) {
        switch self {
        case .null: h.combine(0)
        case let .bool(b): h.combine(1); h.combine(b)
        case let .number(v, _): h.combine(2); h.combine(v)
        case let .string(s): h.combine(3); h.combine(s)
        case let .array(a): h.combine(4); h.combine(a)
        case let .object(o): h.combine(5); h.combine(o.count)
        }
    }

    // MARK: access
    public subscript(key: String) -> JSON? {
        if case let .object(o) = self { return o.first(where: { $0.0 == key })?.1 }
        return nil
    }

    public var string: String? { if case let .string(s) = self { return s }; return nil }
    public var bool: Bool? { if case let .bool(b) = self { return b }; return nil }
    public var double: Double? { if case let .number(v, _) = self { return v }; return nil }
    public var int: Int? { double.map { Int($0) } }
    public var array: [JSON]? { if case let .array(a) = self { return a }; return nil }
    public var object: [(String, JSON)]? { if case let .object(o) = self { return o }; return nil }
    public var isNull: Bool { if case .null = self { return true }; return false }
    public var keys: [String] { object?.map(\.0) ?? [] }

    /// A copy with `key` set (added at the end if it's new, kept in its place if not).
    public func setting(_ key: String, _ value: JSON) -> JSON {
        var o = object ?? []
        if let i = o.firstIndex(where: { $0.0 == key }) { o[i].1 = value } else { o.append((key, value)) }
        return .object(o)
    }

    /// Plain text for a value, as the launcher shows and compares it ("90", "fsr", "true").
    public var plain: String {
        switch self {
        case let .string(s): return s
        default: return serialized()
        }
    }

    public var description: String { serialized() }

    // MARK: writing
    /// Compact JSON, as one line (what the core reads on stdin). ASCII only, like Python's json.dumps.
    public func serialized() -> String {
        var out = ""
        write(&out, indent: nil, level: 0)
        return out
    }

    /// Python's json.dumps(indent=2) layout (the settings file, the looks' files).
    public func pretty() -> String {
        var out = ""
        write(&out, indent: 2, level: 0)
        return out
    }

    private func write(_ out: inout String, indent: Int?, level: Int) {
        switch self {
        case .null: out += "null"
        case let .bool(b): out += b ? "true" : "false"
        case let .number(_, text): out += text
        case let .string(s): JSON.quote(s, into: &out)
        case let .array(a):
            if a.isEmpty { out += "[]"; return }
            out += "["
            for (i, v) in a.enumerated() {
                if i > 0 { out += indent == nil ? ", " : "," }
                if let n = indent { out += "\n" + String(repeating: " ", count: n * (level + 1)) }
                v.write(&out, indent: indent, level: level + 1)
            }
            if let n = indent { out += "\n" + String(repeating: " ", count: n * level) }
            out += "]"
        case let .object(o):
            if o.isEmpty { out += "{}"; return }
            out += "{"
            for (i, (k, v)) in o.enumerated() {
                if i > 0 { out += indent == nil ? ", " : "," }
                if let n = indent { out += "\n" + String(repeating: " ", count: n * (level + 1)) }
                JSON.quote(k, into: &out)
                out += ": "
                v.write(&out, indent: indent, level: level + 1)
            }
            if let n = indent { out += "\n" + String(repeating: " ", count: n * level) }
            out += "}"
        }
    }

    static func quote(_ s: String, into out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if u.value < 0x20 || u.value > 0x7E {
                    for unit in String(u).utf16 { out += String(format: "\\u%04x", unit) }
                } else {
                    out.unicodeScalars.append(u)
                }
            }
        }
        out += "\""
    }

    // MARK: reading
    public struct ParseError: Error, CustomStringConvertible {
        public let description: String
        public init(description: String) { self.description = description }
    }

    public static func parse(_ text: String) throws -> JSON {
        var p = Parser(Array(text.utf8))
        p.space()
        let v = try p.value()
        p.space()
        if p.i != p.b.count { throw ParseError(description: "extra text after the JSON at \(p.i)") }
        return v
    }

    public static func parse(data: Data) throws -> JSON {
        guard let s = String(data: data, encoding: .utf8) else { throw ParseError(description: "not UTF-8") }
        return try parse(s.hasPrefix("\u{FEFF}") ? String(s.dropFirst()) : s)
    }

    private struct Parser {
        let b: [UInt8]
        var i = 0
        init(_ b: [UInt8]) { self.b = b }

        mutating func space() {
            while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 }
        }

        func fail(_ what: String) -> ParseError { ParseError(description: "\(what) at \(i)") }

        mutating func value() throws -> JSON {
            guard i < b.count else { throw fail("unexpected end") }
            switch b[i] {
            case UInt8(ascii: "{"): return try object()
            case UInt8(ascii: "["): return try array()
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try word("true"); return .bool(true)
            case UInt8(ascii: "f"): try word("false"); return .bool(false)
            case UInt8(ascii: "n"): try word("null"); return .null
            case UInt8(ascii: "N"): try word("NaN"); return .number(.nan, "NaN")
            case UInt8(ascii: "I"): try word("Infinity"); return .number(.infinity, "Infinity")
            default: return try number()
            }
        }

        mutating func word(_ w: String) throws {
            let u = Array(w.utf8)
            guard i + u.count <= b.count, Array(b[i..<i + u.count]) == u else { throw fail("bad word") }
            i += u.count
        }

        mutating func number() throws -> JSON {
            let start = i
            if i < b.count, b[i] == UInt8(ascii: "-") { i += 1 }
            if i < b.count, b[i] == UInt8(ascii: "I") { try word("Infinity"); return .number(-.infinity, "-Infinity") }
            while i < b.count, (b[i] >= 0x30 && b[i] <= 0x39) || b[i] == UInt8(ascii: ".") || b[i] == UInt8(ascii: "e")
                || b[i] == UInt8(ascii: "E") || b[i] == UInt8(ascii: "+") || b[i] == UInt8(ascii: "-") { i += 1 }
            let text = String(decoding: b[start..<i], as: UTF8.self)
            guard !text.isEmpty, let v = Double(text) else { throw fail("bad number") }
            return .number(v, text)
        }

        mutating func string() throws -> String {
            i += 1  // the quote
            var bytes: [UInt8] = []
            while i < b.count {
                let c = b[i]
                if c == UInt8(ascii: "\"") { i += 1; return String(decoding: bytes, as: UTF8.self) }
                if c == UInt8(ascii: "\\") {
                    i += 1
                    guard i < b.count else { break }
                    let e = b[i]
                    i += 1
                    switch e {
                    case UInt8(ascii: "n"): bytes.append(0x0A)
                    case UInt8(ascii: "r"): bytes.append(0x0D)
                    case UInt8(ascii: "t"): bytes.append(0x09)
                    case UInt8(ascii: "b"): bytes.append(0x08)
                    case UInt8(ascii: "f"): bytes.append(0x0C)
                    case UInt8(ascii: "u"):
                        var unit = try hex4()
                        var scalar: Unicode.Scalar?
                        if unit >= 0xD800 && unit < 0xDC00, i + 1 < b.count, b[i] == UInt8(ascii: "\\"), b[i + 1] == UInt8(ascii: "u") {
                            i += 2
                            let low = try hex4()
                            unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                        }
                        scalar = Unicode.Scalar(unit)
                        bytes.append(contentsOf: Array(String(scalar ?? "\u{FFFD}").utf8))
                    default: bytes.append(e)
                    }
                    continue
                }
                bytes.append(c)
                i += 1
            }
            throw fail("unterminated string")
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= b.count, let v = UInt32(String(decoding: b[i..<i + 4], as: UTF8.self), radix: 16) else {
                throw fail("bad \\u escape")
            }
            i += 4
            return v
        }

        mutating func array() throws -> JSON {
            i += 1
            var out: [JSON] = []
            space()
            if i < b.count, b[i] == UInt8(ascii: "]") { i += 1; return .array(out) }
            while true {
                space()
                out.append(try value())
                space()
                guard i < b.count else { throw fail("unterminated array") }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "]") { i += 1; return .array(out) }
                throw fail("expected , or ]")
            }
        }

        mutating func object() throws -> JSON {
            i += 1
            var out: [(String, JSON)] = []
            space()
            if i < b.count, b[i] == UInt8(ascii: "}") { i += 1; return .object(out) }
            while true {
                space()
                guard i < b.count, b[i] == UInt8(ascii: "\"") else { throw fail("expected a key") }
                let k = try string()
                space()
                guard i < b.count, b[i] == UInt8(ascii: ":") else { throw fail("expected :") }
                i += 1
                space()
                let v = try value()
                if let j = out.firstIndex(where: { $0.0 == k }) { out[j].1 = v } else { out.append((k, v)) }
                space()
                guard i < b.count else { throw fail("unterminated object") }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "}") { i += 1; return .object(out) }
                throw fail("expected , or }")
            }
        }
    }
}

extension JSON: ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral, ExpressibleByNilLiteral {
    public init(stringLiteral v: String) { self = .string(v) }
    public init(booleanLiteral v: Bool) { self = .bool(v) }
    public init(integerLiteral v: Int) { self = .int(v) }
    public init(floatLiteral v: Double) { self = .number(v, String(v)) }
    public init(nilLiteral: ()) { self = .null }
}
