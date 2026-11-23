//! The pieces of CPython's behaviour the ported code leans on, so it reads and writes exactly what the Python does:
//! str.split()/strip() whitespace, float() and int() of a string, round(), "%.1f" formatting, repr() of a float and
//! of a string, float %, sum() (Neumaier, as 3.12 does it) and the text of a UnicodeDecodeError.

use crate::pyuni;

fn in_table(c: u32, table: &[(u32, u32)]) -> bool {
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// str.isspace().
pub fn is_space(c: char) -> bool {
    pyuni::SPACE.binary_search(&(c as u32)).is_ok()
}

/// str.split() with no arguments.
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|p| !p.is_empty()).collect()
}

/// s.split(None, 1): the first word and the rest (its leading whitespace skipped), or None for no words.
pub fn split_ws_once(s: &str) -> Option<(&str, Option<&str>)> {
    let s = s.trim_start_matches(is_space);
    if s.is_empty() {
        return None;
    }
    match s.find(is_space) {
        None => Some((s, None)),
        Some(i) => {
            let rest = s[i..].trim_start_matches(is_space);
            Some((&s[..i], if rest.is_empty() { None } else { Some(rest) }))
        }
    }
}

/// str.strip() with no arguments.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// The value of a Unicode decimal digit (category Nd), as \d, float() and int() read it.
pub fn decimal_digit(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    let c = c as u32;
    let i = match pyuni::ND_ZEROS.binary_search(&c) {
        Ok(i) => i,
        Err(0) => return None,
        Err(i) => i - 1,
    };
    let d = c - pyuni::ND_ZEROS[i];
    (d < 10).then_some(d)
}

/// str.isdigit() for one character.
pub fn is_digit(c: char) -> bool {
    in_table(c as u32, pyuni::ISDIGIT)
}

/// str.isprintable() for one character.
pub fn is_printable(c: char) -> bool {
    !in_table(c as u32, pyuni::NONPRINTABLE)
}

/// What _PyUnicode_TransformDecimalAndSpaceToASCII does before float() and int() parse: Unicode digits to ASCII,
/// Unicode whitespace to a space; None if another non-ASCII character is in it.
fn to_ascii_number(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) < 127 {
            out.push(c); // (ASCII stays as it is: "\x1c" is whitespace to str.split(), not to float())
        } else if is_space(c) {
            out.push(' ');
        } else {
            out.push(char::from(b'0' + decimal_digit(c)? as u8));
        }
    }
    Some(out)
}

fn ascii_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// Underscores only between digits (PEP 515), then taken out; None if they're misplaced.
fn drop_underscores(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut prev = '\0';
    for c in s.chars() {
        if c == '_' {
            if !prev.is_ascii_digit() {
                return None;
            }
        } else {
            if prev == '_' && !c.is_ascii_digit() {
                return None;
            }
            out.push(c);
        }
        prev = c;
    }
    if prev == '_' {
        return None;
    }
    Some(out)
}

/// float(s) for a str: None where Python raises ValueError.
pub fn float_from_str(s: &str) -> Option<f64> {
    let t = to_ascii_number(s)?;
    let t = t.trim_matches(ascii_space);
    let t = drop_underscores(t)?;
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, &t[..]),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(sign * f64::INFINITY);
    }
    if lower == "nan" {
        return Some(if sign < 0.0 { -f64::NAN } else { f64::NAN });
    }
    // [digits][.digits][(e|E)[sign]digits], with at least one digit before the exponent
    let b = body.as_bytes();
    let mut i = 0;
    let mut mantissa_digits = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        mantissa_digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            mantissa_digits += 1;
        }
    }
    if mantissa_digits == 0 {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != b.len() {
        return None;
    }
    let v: f64 = body.parse().ok()?;
    Some(sign * v)
}

/// int(s) for a str (base 10): None where Python raises ValueError. Too big for an i128: i128::MAX or MIN.
pub fn int_from_str(s: &str) -> Option<i128> {
    let t = to_ascii_number(s)?;
    let t = t.trim_matches(ascii_space);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() || body.starts_with('_') {
        return None;
    }
    let body = drop_underscores(body)?;
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // (int("0_0") is 0, and leading zeros are fine in int(), unlike in Python source)
    match body.parse::<i128>() {
        Ok(v) => Some(if neg { -v } else { v }),
        Err(_) => Some(if neg { i128::MIN } else { i128::MAX }),
    }
}

/// round(x) for a float: to the nearest whole number, halves to even.
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// round(x, ndigits) for a float: correctly rounded to ndigits decimals (halves to even, on the exact binary value).
pub fn round_to(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let s = format!("{:.*}", ndigits, x);
    s.parse().unwrap_or(x)
}

/// f"{x:.{prec}f}".
pub fn fmt_fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{:.*}", prec, x)
}

/// repr(x) (and str(x)) for a float.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let e = format!("{:e}", x.abs()); // the shortest digits that read back as x, as dtoa mode 0 gives
    let (mant, exp) = e.split_once('e').expect("exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt: i32 = exp.parse::<i32>().expect("exponent") + 1;
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let m = if n == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let ex = decpt - 1;
        format!("{sign}{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
    } else if decpt <= 0 {
        format!("{sign}0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt < n {
        format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    } else {
        format!("{sign}{}{}.0", digits, "0".repeat((decpt - n) as usize))
    }
}

/// x % y for floats: the result has the divisor's sign.
pub fn fmod(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m != 0.0 { if (y < 0.0) != (m < 0.0) { m + y } else { m } } else { 0.0f64.copysign(y) }
}

/// sum() of floats, as CPython 3.12 adds them (Neumaier's compensated sum), from a start of int 0.
pub fn sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let mut it = items.into_iter();
    let Some(first) = it.next() else { return 0.0 };
    let mut f = 0.0 + first;
    let mut c = 0.0f64;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// repr(s) for a str.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x7f && (c as u32) >= 0x20 => out.push(c),
            c if (c as u32) < 0x80 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if is_printable(c) => out.push(c),
            c if (c as u32) <= 0xff => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

/// Python's text for bytes that aren't UTF-8 ("'utf-8' codec can't decode byte 0xff in position 0: invalid start
/// byte"), as read_text(encoding="utf-8") raises it.
pub fn utf8_error_text(bytes: &[u8], e: &std::str::Utf8Error) -> String {
    let start = e.valid_up_to();
    let (len, why) = match e.error_len() {
        Some(n) => {
            let lead = bytes[start];
            let why = if (0xC2..=0xF4).contains(&lead) { "invalid continuation byte" } else { "invalid start byte" };
            (n, why)
        }
        None => (bytes.len() - start, "unexpected end of data"),
    };
    if len == 1 {
        format!("'utf-8' codec can't decode byte 0x{:02x} in position {}: {}", bytes[start], start, why)
    } else {
        format!("'utf-8' codec can't decode bytes in position {}-{}: {}", start, start + len - 1, why)
    }
}

/// Text mode's universal newlines: "\r\n" and "\r" read as "\n".
pub fn universal_newlines(s: &str) -> String {
    if !s.contains('\r') {
        return s.to_string();
    }
    s.replace("\r\n", "\n").replace('\r', "\n")
}

/// Why a file couldn't be read or written, as Python's OSError.strerror says it ("No such file or directory").
pub fn strerror(e: &std::io::Error) -> String {
    let text = e.to_string();
    match text.rfind(" (os error ") {
        Some(i) if e.raw_os_error().is_some() => text[..i].to_string(),
        _ => text,
    }
}

/// str(OSError) for an error about a path: "[Errno 2] No such file or directory: '/x'".
pub fn oserror_text(e: &std::io::Error, path: &std::path::Path) -> String {
    match e.raw_os_error() {
        Some(n) => format!("[Errno {}] {}: {}", n, strerror(e), str_repr(&path.to_string_lossy())),
        None => strerror(e),
    }
}

/// Path.read_text(encoding="utf-8"): strict UTF-8, universal newlines. Err: the OSError or UnicodeDecodeError text.
pub fn read_text(path: &std::path::Path) -> Result<String, ReadError> {
    let bytes = std::fs::read(path).map_err(ReadError::Os)?;
    match std::str::from_utf8(&bytes) {
        Ok(s) => Ok(universal_newlines(s)),
        Err(e) => Err(ReadError::Decode(utf8_error_text(&bytes, &e))),
    }
}

#[derive(Debug)]
pub enum ReadError {
    /// An OSError.
    Os(std::io::Error),
    /// A UnicodeDecodeError (a ValueError), with Python's text.
    Decode(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (0.1 + 0.2, "0.30000000000000004"),
            (2.0, "2.0"),
            (-0.0, "-0.0"),
            (1.5e300, "1.5e+300"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.25, "0.25"),
        ] {
            assert_eq!(float_repr(x), want);
        }
    }

    #[test]
    fn float_from_str_matches_python() {
        assert_eq!(float_from_str(" 1_000.5 "), Some(1000.5));
        assert_eq!(float_from_str("1__0"), None);
        assert_eq!(float_from_str("_1"), None);
        assert_eq!(float_from_str("5."), Some(5.0));
        assert_eq!(float_from_str(".5"), Some(0.5));
        assert_eq!(float_from_str("."), None);
        assert_eq!(float_from_str("1e"), None);
        assert_eq!(float_from_str("-iNfInItY"), Some(f64::NEG_INFINITY));
        assert!(float_from_str("nan").unwrap().is_nan());
        assert_eq!(float_from_str("\u{0661}.5"), Some(1.5)); // Arabic-Indic one
        assert_eq!(float_from_str("1.5\u{2003}"), Some(1.5));
        assert_eq!(float_from_str("x"), None);
        assert_eq!(float_from_str(""), None);
        assert_eq!(int_from_str("+9"), Some(9));
        assert_eq!(int_from_str("1_7"), Some(17));
        assert_eq!(int_from_str("9.0"), None);
    }

    #[test]
    fn neumaier_sum() {
        assert_eq!(sum([0.1, 0.2, 0.3]), 0.6); // plain left-to-right gives 0.6000000000000001
        assert_eq!(sum([1e100, 1.0, -1e100]), 1.0);
    }

    #[test]
    fn repr_of_strings() {
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(str_repr("a\nb\u{a0}"), "'a\\nb\\xa0'");
        assert_eq!(str_repr("caf\u{e9}"), "'caf\u{e9}'");
    }
}
