//! Share codes: "SPL1-" and URL-safe base64 (no padding) of
//!
//! ```text
//! flags (1 byte: bit 0 = the rest is raw deflate)  body  crc32(flags + body) & 0xFFFF, big-endian
//! body: varint name length, name (UTF-8, at most 48 bytes), varint node count,
//!       then per node: varint (code << 1 | off), varint mask of the fields that differ from their defaults,
//!       and for each of those: zigzag varint of the difference in steps, or (a LUT's file) varint length + UTF-8.
//! ```
//!
//! Python's encode() deflates the body at level 9 and keeps it if it's shorter. [`crate::zlib`] does it bit for bit,
//! so the same look gets the same code.

use super::{Look, LookError, MAX_NAME, normalize, type_by_code};
use crate::json::{Map, Value};
use crate::{py, pyuni, zlib};

pub const PREFIX: &str = "SPL1-";

/// A share code that can't be read. The message says why, for the user.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct CodeError(pub String);

const NOT_A_CODE: &str = "That isn't a SmoovPlay look code. Look codes start with SPL1-.";
const NEWER: &str = "That look code is from a newer version of SmoovPlay. Update SmoovPlay to use it.";
const TYPO: &str = "That look code is incomplete or has a typo. Copy it again, all of it.";
const DAMAGED: &str = "That look code is damaged. Copy it again, all of it.";
const UNKNOWN_EFFECT: &str =
    "That look uses an effect this version of SmoovPlay doesn't have. Update SmoovPlay to use it.";

fn varint(out: &mut Vec<u8>, mut v: u128) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        out.push(b | if v != 0 { 0x80 } else { 0 });
        if v == 0 {
            return;
        }
    }
}

fn zz(v: i64) -> u128 {
    if v >= 0 { (v as u128) * 2 } else { ((-(v as i128)) * 2 - 1) as u128 }
}

fn unzz(v: u128) -> i128 {
    let v = v as i128;
    if v % 2 == 0 { v / 2 } else { -(v + 1) / 2 }
}

/// text.encode("utf-8")[:limit], cut back to a whole character.
fn cut_utf8(text: &str, limit: usize) -> &[u8] {
    let mut end = limit.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text.as_bytes()[..end]
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let n = chunk.len();
        let b = [chunk[0], if n > 1 { chunk[1] } else { 0 }, if n > 2 { chunk[2] } else { 0 }];
        let v = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..(n + 1) {
            out.push(B64[((v >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

/// base64.urlsafe_b64decode(text + padding): None where Python raises (a length 1 more than a multiple of 4, or a
/// character outside ASCII). Leftover bits are ignored, as binascii does.
fn b64_decode(text: &str) -> Option<Vec<u8>> {
    if !text.is_ascii() || text.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for c in text.bytes() {
        let v = B64.iter().position(|&x| x == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// The look's share code (normalized first, as the Python does).
pub fn encode(look: &Value) -> Result<String, LookError> {
    Ok(encode_look(&normalize(look)?))
}

/// The share code of a look that's already normalized.
pub fn encode_look(look: &Look) -> String {
    let mut out = Vec::new();
    let name = cut_utf8(&look.name, MAX_NAME);
    varint(&mut out, name.len() as u128);
    out.extend_from_slice(name);
    varint(&mut out, look.nodes.len() as u128);
    for n in &look.nodes {
        varint(&mut out, (n.ty.code as u128) << 1 | if n.on { 0 } else { 1 });
        let mut mask: u128 = 0;
        let mut vals = Vec::new();
        for (i, (f, v)) in n.ty.fields().zip(&n.values).enumerate() {
            if f.text {
                let t = v.text();
                if !t.is_empty() {
                    mask |= 1 << i;
                    let b = cut_utf8(t, 120);
                    varint(&mut vals, b.len() as u128);
                    vals.extend_from_slice(b);
                }
            } else {
                let d = f.q(v.num()) - f.q(f.default_num());
                if d != 0 {
                    mask |= 1 << i;
                    varint(&mut vals, zz(d));
                }
            }
        }
        varint(&mut out, mask);
        out.extend(vals);
    }
    let packed = zlib::deflate_raw(&out);
    let mut data = if packed.len() < out.len() {
        let mut d = vec![1u8];
        d.extend(packed);
        d
    } else {
        let mut d = vec![0u8];
        d.extend(&out);
        d
    };
    let crc = (zlib::crc32(&data) & 0xFFFF) as u16;
    data.extend(crc.to_be_bytes());
    format!("{PREFIX}{}", b64_encode(&data))
}

fn in_list(c: char, list: &[u32]) -> bool {
    list.contains(&(c as u32))
}

/// (?i)SPL(\d+)-([A-Za-z0-9_-]+), matched as Python's re does (IGNORECASE lets in a few letters beyond ASCII).
fn match_code(text: &str) -> Option<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let letter = |c: char, up: char, extra: &[u32]| c.eq_ignore_ascii_case(&up) || in_list(c, extra);
    if chars.len() < 5
        || !letter(chars[0], 'S', pyuni::S_EXTRA)
        || !letter(chars[1], 'P', pyuni::P_EXTRA)
        || !letter(chars[2], 'L', pyuni::L_EXTRA)
    {
        return None;
    }
    let mut i = 3;
    let start = i;
    while i < chars.len() && py::decimal_digit(chars[i]).is_some() {
        i += 1;
    }
    if i == start || i >= chars.len() || chars[i] != '-' {
        return None;
    }
    let version: String = chars[start..i].iter().collect();
    let body: String = chars[i + 1..].iter().collect();
    let ok = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-' || in_list(c, pyuni::CODE_EXTRA);
    if body.is_empty() || !body.chars().all(ok) {
        return None;
    }
    Some((version, body))
}

/// The look in a share code. Err holds a message for the user.
pub fn decode(code: &str) -> Result<Look, CodeError> {
    let text: String = py::split_ws(code).concat(); // chat apps wrap long lines
    let Some((version, body)) = match_code(&text) else {
        return Err(CodeError(NOT_A_CODE.into()));
    };
    if version != "1" {
        return Err(CodeError(NEWER.into()));
    }
    let data = b64_decode(&body).unwrap_or_default();
    if data.len() < 4 {
        return Err(CodeError(TYPO.into()));
    }
    let (head, tail) = data.split_at(data.len() - 2);
    if ((zlib::crc32(head) & 0xFFFF) as u16).to_be_bytes() != tail {
        return Err(CodeError(TYPO.into()));
    }
    let flags = head[0];
    if flags & !1 != 0 {
        return Err(CodeError(NEWER.into()));
    }
    let body = if flags & 1 != 0 {
        zlib::inflate_raw(&head[1..]).map_err(|_| CodeError(DAMAGED.into()))?
    } else {
        head[1..].to_vec()
    };
    match parse(&body) {
        Ok(look) => Ok(look),
        Err(Fail::Code(m)) => Err(CodeError(m.into())),
        Err(Fail::Damaged) => Err(CodeError(DAMAGED.into())),
    }
}

enum Fail {
    Code(&'static str),
    Damaged,
}

struct Reader<'a> {
    body: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn varint(&mut self) -> Result<u128, Fail> {
        let (mut v, mut shift) = (0u128, 0u32);
        loop {
            let b = *self.body.get(self.pos).ok_or(Fail::Damaged)?;
            self.pos += 1;
            v |= ((b & 0x7F) as u128) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            if shift > 63 {
                return Err(Fail::Damaged); // varint too long
            }
        }
    }

    fn text(&mut self) -> Result<String, Fail> {
        let n = self.varint()?;
        let end = self.pos.saturating_add(usize::try_from(n).unwrap_or(usize::MAX)).min(self.body.len());
        let s = std::str::from_utf8(&self.body[self.pos.min(end)..end]).map_err(|_| Fail::Damaged)?.to_string();
        if (self.body.len() as u128) < self.pos as u128 + n {
            return Err(Fail::Damaged); // cut short
        }
        self.pos += n as usize;
        Ok(s)
    }
}

fn parse(body: &[u8]) -> Result<Look, Fail> {
    let mut r = Reader { body, pos: 0 };
    let name = r.text()?;
    let mut nodes: Vec<Value> = Vec::new();
    let count = r.varint()?;
    let mut k: u128 = 0;
    while k < count {
        k += 1;
        let head = r.varint()?;
        let t = type_by_code(head >> 1).ok_or(Fail::Code(UNKNOWN_EFFECT))?;
        let mut n = Map::new();
        n.insert("type", t.key);
        n.insert("on", head & 1 == 0);
        n.insert("mix", 1.0);
        let mask = r.varint()?;
        let mut i = 0u32;
        while i < 128 && (mask >> i) != 0 {
            if (mask >> i) & 1 == 1 {
                let f = t.field(i as usize);
                if let Some(f) = f.filter(|f| f.text) {
                    n.insert(f.key, r.text()?);
                } else {
                    let d = unzz(r.varint()?); // a setting from a newer version, skipped
                    if let Some(f) = f {
                        let steps = (f.q(f.default_num()) as i128 + d) as f64;
                        n.insert(f.key, f.lo + steps * f.step);
                    }
                }
            }
            i += 1;
        }
        nodes.push(Value::Dict(n));
    }
    if r.pos != body.len() {
        return Err(Fail::Damaged); // extra bytes
    }
    let mut look = Map::new();
    look.insert("name", name);
    look.insert("nodes", Value::List(nodes));
    normalize(&Value::Dict(look)).map_err(|_| Fail::Damaged)
}
