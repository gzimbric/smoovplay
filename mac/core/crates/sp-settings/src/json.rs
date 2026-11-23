//! JSON as Python's json module reads and writes it, so files and protocol lines match the Python byte for byte.
//!
//! - [`Value`] keeps what Python keeps: ints and floats apart (1 and 1.0 are different on the wire), dict order as
//!   written, any size of int.
//! - [`loads`] is json.loads (NaN and Infinity allowed, Python's error messages and positions).
//! - [`dumps`] is json.dumps with its defaults (", " and ": ", ASCII only with \u escapes, floats as repr()), and
//!   [`dumps_indent`] is json.dumps(indent=n).
//! - [`Value::py_eq`], [`Value::truthy`], [`Value::py_str`], [`Value::py_repr`] are ==, bool(), str() and repr().

use std::fmt;

use crate::py;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    /// An int outside i64, as its decimal digits (with a leading "-" if negative).
    BigInt(String),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Dict(Map),
}

/// A dict, with keys in insertion order. Setting a key that's there keeps its place, as in Python.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Map(Vec<(String, Value)>);

impl Map {
    pub fn new() -> Self {
        Map(Vec::new())
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        let key = key.into();
        let value = value.into();
        match self.get_mut(&key) {
            Some(v) => *v = value,
            None => self.0.push((key, value)),
        }
    }
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let i = self.0.iter().position(|(k, _)| k == key)?;
        Some(self.0.remove(i).1)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

impl<K: Into<String>, V: Into<Value>> FromIterator<(K, V)> for Map {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut m = Map::new();
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v as i64)
    }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Int(v as i64)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Int(v as i64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<&String> for Value {
    fn from(v: &String) -> Self {
        Value::Str(v.clone())
    }
}
impl From<Map> for Value {
    fn from(v: Map) -> Self {
        Value::Dict(v)
    }
}
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        v.map_or(Value::Null, Into::into)
    }
}

/// A dict from (key, value) pairs, in that order.
pub fn obj<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    Value::Dict(pairs.into_iter().collect())
}

/// Exact comparison of an integer (as digits or i64) with a float, as Python compares int and float.
fn int_eq_float(i: &Value, f: f64) -> bool {
    if !f.is_finite() || f.fract() != 0.0 {
        return false;
    }
    match i {
        Value::Int(n) => {
            if f.abs() < 9.2e18 {
                f as i64 == *n && (*n as f64) == f
            } else {
                false
            }
        }
        Value::BigInt(s) => format!("{:.0}", f) == *s,
        _ => false,
    }
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_dict(&self) -> Option<&Map> {
        match self {
            Value::Dict(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v),
            _ => None,
        }
    }
    /// A number as a float (bool, int or float), as Python arithmetic would use it.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Int(n) => Some(*n as f64),
            Value::BigInt(s) => s.parse().ok(),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
    /// A whole number (bool, int, or a float with no fraction).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(*b as i64),
            Value::Int(n) => Some(*n),
            Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.2e18 => Some(*f as i64),
            _ => None,
        }
    }
    /// dict.get(key) on a dict, and None for anything else.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict().and_then(|m| m.get(key))
    }

    fn is_number(&self) -> bool {
        matches!(self, Value::Bool(_) | Value::Int(_) | Value::BigInt(_) | Value::Float(_))
    }

    /// Python's ==: True == 1 == 1.0, dicts whatever their order.
    pub fn py_eq(&self, other: &Value) -> bool {
        use Value::*;
        if self.is_number() && other.is_number() {
            let norm = |v: &Value| match v {
                Bool(b) => Int(*b as i64),
                other => other.clone(),
            };
            return match (norm(self), norm(other)) {
                (Int(a), Int(b)) => a == b,
                (BigInt(a), BigInt(b)) => a == b,
                (Int(_), BigInt(_)) | (BigInt(_), Int(_)) => false,
                (Float(a), Float(b)) => a == b,
                (i, Float(f)) | (Float(f), i) => int_eq_float(&i, f),
                _ => false,
            };
        }
        match (self, other) {
            (Null, Null) => true,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y)),
            (Dict(a), Dict(b)) => a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| v.py_eq(w))),
            _ => false,
        }
    }

    /// `self in values`, by ==.
    pub fn py_in(&self, values: &[Value]) -> bool {
        values.iter().any(|v| self.py_eq(v))
    }

    /// bool(v).
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::BigInt(_) => true,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Dict(m) => !m.is_empty(),
        }
    }

    /// str(v).
    pub fn py_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            other => other.py_repr(),
        }
    }

    /// repr(v).
    pub fn py_repr(&self) -> String {
        match self {
            Value::Null => "None".into(),
            Value::Bool(true) => "True".into(),
            Value::Bool(false) => "False".into(),
            Value::Int(n) => n.to_string(),
            Value::BigInt(s) => s.clone(),
            Value::Float(f) => py::float_repr(*f),
            Value::Str(s) => py::str_repr(s),
            Value::List(v) => format!("[{}]", v.iter().map(Value::py_repr).collect::<Vec<_>>().join(", ")),
            Value::Dict(m) => format!(
                "{{{}}}",
                m.iter().map(|(k, v)| format!("{}: {}", py::str_repr(k), v.py_repr())).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    /// The Python type's name, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) | Value::BigInt(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
        }
    }
}

// ------------------------------------------------------------------------------------------------ writing
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
        }
    }
    out.push('"');
}

fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
    } else if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        out.push_str(&py::float_repr(f));
    }
}

fn write(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(n) => out.push_str(&n.to_string()),
        Value::BigInt(s) => out.push_str(s),
        Value::Float(f) => write_float(out, *f),
        Value::Str(s) => write_str(out, s),
        Value::List(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                sep(out, i, indent, level + 1);
                write(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push(']');
        }
        Value::Dict(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in m.iter().enumerate() {
                sep(out, i, indent, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push('}');
        }
    }
}

fn sep(out: &mut String, i: usize, indent: Option<usize>, level: usize) {
    match indent {
        None => {
            if i > 0 {
                out.push_str(", ");
            }
        }
        Some(n) => {
            if i > 0 {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

/// json.dumps(v).
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write(&mut out, v, None, 0);
    out
}

/// json.dumps(v, indent=indent).
pub fn dumps_indent(v: &Value, indent: usize) -> String {
    let mut out = String::new();
    write(&mut out, v, Some(indent), 0);
    out
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&dumps(self))
    }
}

// ------------------------------------------------------------------------------------------------ reading
/// A json.JSONDecodeError (or the ValueError json.loads raises for an int too long to read), with Python's text.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonError {
    pub msg: String,
    /// Where, in characters. None for an error without a position.
    pub pos: Option<usize>,
    lineno: usize,
    colno: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pos {
            Some(pos) => write!(f, "{}: line {} column {} (char {})", self.msg, self.lineno, self.colno, pos),
            None => f.write_str(&self.msg),
        }
    }
}

impl std::error::Error for JsonError {}

enum Fail {
    /// StopIteration(pos): "Expecting value" there.
    Stop(usize),
    At(&'static str, usize),
    Plain(String),
}

const MAX_INT_DIGITS: usize = 4300; // sys.get_int_max_str_digits()
const MAX_DEPTH: usize = 990; // about where CPython gives up with a RecursionError

struct Parser<'a> {
    s: &'a [char],
    depth: usize,
}

fn is_json_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl Parser<'_> {
    fn ws(&self, mut i: usize) -> usize {
        while i < self.s.len() && is_json_ws(self.s[i]) {
            i += 1;
        }
        i
    }

    fn word(&self, i: usize, w: &str) -> bool {
        let n = w.chars().count();
        i + n <= self.s.len() && self.s[i..i + n].iter().copied().eq(w.chars())
    }

    fn scan_once(&mut self, idx: usize) -> Result<(Value, usize), Fail> {
        let len = self.s.len();
        if idx >= len {
            return Err(Fail::Stop(idx));
        }
        match self.s[idx] {
            '"' => {
                let (s, end) = self.scanstring(idx + 1)?;
                return Ok((Value::Str(s), end));
            }
            '{' | '[' => {
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return Err(Fail::Plain("maximum recursion depth exceeded while decoding a JSON document".into()));
                }
                let r = if self.s[idx] == '{' { self.object(idx + 1) } else { self.array(idx + 1) };
                self.depth -= 1;
                return r;
            }
            'n' if idx + 3 < len && self.word(idx, "null") => return Ok((Value::Null, idx + 4)),
            't' if idx + 3 < len && self.word(idx, "true") => return Ok((Value::Bool(true), idx + 4)),
            'f' if idx + 4 < len && self.word(idx, "false") => return Ok((Value::Bool(false), idx + 5)),
            'N' if idx + 2 < len && self.word(idx, "NaN") => return Ok((Value::Float(f64::NAN), idx + 3)),
            'I' if idx + 7 < len && self.word(idx, "Infinity") => {
                return Ok((Value::Float(f64::INFINITY), idx + 8));
            }
            '-' if idx + 8 < len && self.word(idx, "-Infinity") => {
                return Ok((Value::Float(f64::NEG_INFINITY), idx + 9));
            }
            _ => {}
        }
        self.number(idx)
    }

    fn number(&self, start: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let end_idx = s.len() - 1;
        let digit = |i: usize| s[i].is_ascii_digit();
        let mut idx = start;
        if s[idx] == '-' {
            idx += 1;
            if idx > end_idx {
                return Err(Fail::Stop(start));
            }
        }
        if ('1'..='9').contains(&s[idx]) {
            idx += 1;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        } else if s[idx] == '0' {
            idx += 1;
        } else {
            return Err(Fail::Stop(start));
        }
        let mut is_float = false;
        if idx < end_idx && s[idx] == '.' && digit(idx + 1) {
            is_float = true;
            idx += 2;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        }
        if idx < end_idx && (s[idx] == 'e' || s[idx] == 'E') {
            let e_start = idx;
            idx += 1;
            if idx < end_idx && (s[idx] == '-' || s[idx] == '+') {
                idx += 1;
            }
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
            if digit(idx - 1) {
                is_float = true;
            } else {
                idx = e_start;
            }
        }
        let text: String = s[start..idx].iter().collect();
        if is_float {
            return Ok((Value::Float(text.parse().unwrap_or(f64::NAN)), idx));
        }
        let digits = text.trim_start_matches('-').len();
        if digits > MAX_INT_DIGITS {
            return Err(Fail::Plain(format!(
                "Exceeds the limit ({MAX_INT_DIGITS} digits) for integer string conversion: value has {digits} \
                 digits; use sys.set_int_max_str_digits() to increase the limit"
            )));
        }
        Ok((
            match text.parse::<i64>() {
                Ok(n) => Value::Int(n),
                Err(_) => Value::BigInt(text),
            },
            idx,
        ))
    }

    fn hex4(&self, from: usize, to: usize) -> Option<u32> {
        self.s[from..to].iter().try_fold(0u32, |c, d| Some((c << 4) | d.to_digit(16)?))
    }

    fn scanstring(&self, mut end: usize) -> Result<(String, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let begin = end - 1;
        let mut out = String::new();
        loop {
            let mut next = end;
            let mut c = '\0';
            while next < len {
                c = s[next];
                if c == '"' || c == '\\' {
                    break;
                }
                if (c as u32) <= 0x1f {
                    return Err(Fail::At("Invalid control character at", next));
                }
                next += 1;
            }
            if next >= len {
                return Err(Fail::At("Unterminated string starting at", begin));
            }
            out.extend(&s[end..next]);
            next += 1;
            if c == '"' {
                end = next;
                break;
            }
            if next == len {
                return Err(Fail::At("Unterminated string starting at", begin));
            }
            let e = s[next];
            let ch = if e != 'u' {
                end = next + 1;
                match e {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(Fail::At("Invalid \\escape", end - 2)),
                }
            } else {
                next += 1;
                end = next + 4;
                if end >= len {
                    return Err(Fail::At("Invalid \\uXXXX escape", next - 1));
                }
                let mut code = self.hex4(next, end).ok_or(Fail::At("Invalid \\uXXXX escape", end - 5))?;
                next = end;
                if (0xD800..=0xDBFF).contains(&code) && end + 6 < len && s[next] == '\\' && s[next + 1] == 'u' {
                    end += 6;
                    let c2 = self.hex4(next + 2, end).ok_or(Fail::At("Invalid \\uXXXX escape", end - 5))?;
                    if (0xDC00..=0xDFFF).contains(&c2) {
                        code = 0x10000 + (((code - 0xD800) << 10) | (c2 - 0xDC00));
                    } else {
                        end -= 6;
                    }
                }
                // A lone surrogate, which a Python str can hold and a Rust one can't, reads as U+FFFD.
                char::from_u32(code).unwrap_or('\u{FFFD}')
            };
            out.push(ch);
        }
        Ok((out, end))
    }

    fn object(&mut self, mut idx: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let mut m = Map::new();
        idx = self.ws(idx);
        if idx >= len || s[idx] != '}' {
            loop {
                if idx >= len || s[idx] != '"' {
                    return Err(Fail::At("Expecting property name enclosed in double quotes", idx));
                }
                let (key, next) = self.scanstring(idx + 1)?;
                idx = self.ws(next);
                if idx >= len || s[idx] != ':' {
                    return Err(Fail::At("Expecting ':' delimiter", idx));
                }
                idx = self.ws(idx + 1);
                let (val, next) = self.scan_once(idx)?;
                m.insert(key, val);
                idx = self.ws(next);
                if idx < len && s[idx] == '}' {
                    break;
                }
                if idx >= len || s[idx] != ',' {
                    return Err(Fail::At("Expecting ',' delimiter", idx));
                }
                idx = self.ws(idx + 1);
            }
        }
        Ok((Value::Dict(m), idx + 1))
    }

    fn array(&mut self, mut idx: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let mut items = Vec::new();
        idx = self.ws(idx);
        if idx >= len || s[idx] != ']' {
            loop {
                let (val, next) = self.scan_once(idx)?;
                items.push(val);
                idx = self.ws(next);
                if idx < len && s[idx] == ']' {
                    break;
                }
                if idx >= len || s[idx] != ',' {
                    return Err(Fail::At("Expecting ',' delimiter", idx));
                }
                idx = self.ws(idx + 1);
            }
        }
        Ok((Value::List(items), idx + 1))
    }
}

fn error_at(s: &[char], msg: &str, pos: usize) -> JsonError {
    let before = &s[..pos.min(s.len())];
    let lineno = before.iter().filter(|c| **c == '\n').count() + 1;
    let colno = match before.iter().rposition(|c| *c == '\n') {
        Some(i) => pos - i,
        None => pos + 1,
    };
    JsonError { msg: msg.to_string(), pos: Some(pos), lineno, colno }
}

/// json.loads(text).
pub fn loads(text: &str) -> Result<Value, JsonError> {
    let s: Vec<char> = text.chars().collect();
    if s.first() == Some(&'\u{feff}') {
        return Err(error_at(&s, "Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    let mut p = Parser { s: &s, depth: 0 };
    let start = p.ws(0);
    let (v, end) = match p.scan_once(start) {
        Ok(x) => x,
        Err(Fail::Stop(pos)) => return Err(error_at(&s, "Expecting value", pos)),
        Err(Fail::At(msg, pos)) => return Err(error_at(&s, msg, pos)),
        Err(Fail::Plain(msg)) => return Err(JsonError { msg, pos: None, lineno: 0, colno: 0 }),
    };
    let end = p.ws(end);
    if end != s.len() {
        return Err(error_at(&s, "Extra data", end));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_like_python() {
        let (e, smile) = (char::from_u32(0xe9).unwrap(), char::from_u32(0x1F600).unwrap());
        let text = format!(r#"{{"a": 1, "b": 1.0, "c": [true, null, "x{e}{smile}"], "a": 2, "d": -0.0, "e": 1e400}}"#);
        let v = loads(&text).unwrap();
        let bs = '\\';
        let want = format!(
            r#"{{"a": 2, "b": 1.0, "c": [true, null, "x{bs}u00e9{bs}ud83d{bs}ude00"], "d": -0.0, "e": Infinity}}"#
        );
        assert_eq!(dumps(&v), want);
        let esc = format!(r#""{bs}u00e9{bs}ud83d{bs}ude00""#);
        assert_eq!(loads(&esc).unwrap(), Value::Str(format!("{e}{smile}")));
        assert_eq!(
            dumps_indent(&loads("{\"a\": [1, {}], \"b\": []}").unwrap(), 2),
            "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": []\n}"
        );
        assert_eq!(loads("12345678901234567890123").unwrap(), Value::BigInt("12345678901234567890123".into()));
    }

    #[test]
    fn errors_read_like_pythons() {
        let e = |s: &str| loads(s).unwrap_err().to_string();
        assert_eq!(e("{not json"), "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)");
        assert_eq!(e(""), "Expecting value: line 1 column 1 (char 0)");
        assert_eq!(e("[1,]"), "Expecting value: line 1 column 4 (char 3)");
        assert_eq!(e("{\"a\" 1}"), "Expecting ':' delimiter: line 1 column 6 (char 5)");
        assert_eq!(e("[1 2]"), "Expecting ',' delimiter: line 1 column 4 (char 3)");
        assert_eq!(e("\"abc"), "Unterminated string starting at: line 1 column 1 (char 0)");
        assert_eq!(e("1 2"), "Extra data: line 1 column 3 (char 2)");
        assert_eq!(e("{\n\"a\": tru}"), "Expecting value: line 2 column 6 (char 7)");
    }

    #[test]
    fn python_equality() {
        assert!(Value::Bool(true).py_eq(&Value::Int(1)));
        assert!(Value::Int(1).py_eq(&Value::Float(1.0)));
        assert!(!Value::Str("1".into()).py_eq(&Value::Int(1)));
        assert!(Value::Float(0.0).py_eq(&Value::Bool(false)));
    }
}
