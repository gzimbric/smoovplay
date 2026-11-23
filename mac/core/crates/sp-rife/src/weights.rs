//! RIFE checkpoints as vs-rife publishes them (flownet_v{version}.pkl). Each is a `torch.save` zip archive holding a
//! pickled dict of tensor names to tensors (data.pkl) and each tensor's bytes (data/N). This reads them without
//! Python. It has a reader for the zip's stored (uncompressed) entries, which is all torch.save writes, and a small
//! unpickler that knows the opcodes and the two callables such a dict uses, `collections.OrderedDict` and
//! `torch._utils._rebuild_tensor_v2`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Error, Result};

/// One tensor from the checkpoint, as f32 whatever it was stored as, contiguous in its own (row-major) shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub type Weights = HashMap<String, Tensor>;

/// Where framegen/interpolator.py downloads checkpoints from (its WEIGHTS_URL).
pub const WEIGHTS_URL: &str = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{version}.pkl";

/// The repo's models/ folder (framegen/interpolator.py DEFAULT_MODEL_DIR), found from this crate's location, or
/// $SMOOVPLAY_MODELS if set.
pub fn default_model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SMOOVPLAY_MODELS") {
        return PathBuf::from(dir);
    }
    // four levels up from mac/core/crates/sp-rife is the repo root
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../models")
}

/// The checkpoint for `version` in `dir`, downloaded first if it isn't there. The download uses curl, as the Python
/// does when its own download fails, so the Mac's TLS inspection proxy and system certificates apply.
pub fn ensure_weights(version: &str, dir: &Path) -> Result<PathBuf> {
    ensure_weights_from(version, dir, WEIGHTS_URL)
}

/// ensure_weights with another URL template ("{version}" is replaced), for tests and mirrors.
pub fn ensure_weights_from(version: &str, dir: &Path, url_template: &str) -> Result<PathBuf> {
    let path = dir.join(format!("flownet_v{version}.pkl"));
    if path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(dir)?;
    let url = url_template.replace("{version}", version);
    eprintln!("[rife] downloading {url}");
    let part = path.with_extension("part");
    let status = Command::new("curl").arg("-sfL").arg("-o").arg(&part).arg(&url).status()?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(Error::Other(format!("couldn't download {url} (curl: {status})")));
    }
    std::fs::rename(&part, &path)?;
    Ok(path)
}

/// Every tensor in a torch.save checkpoint, by name.
pub fn load(path: &Path) -> Result<Weights> {
    let bytes = std::fs::read(path)?;
    let zip = Zip::parse(&bytes).map_err(|e| Error::Other(format!("{}: {e}", path.display())))?;
    let pkl_name = zip
        .names()
        .find(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
        .ok_or_else(|| Error::Other(format!("{}: no data.pkl (not a torch.save archive)", path.display())))?
        .to_string();
    let prefix = &pkl_name[..pkl_name.len() - "data.pkl".len()];
    let pkl = zip.get(&pkl_name).map_err(|e| Error::Other(format!("{}: {e}", path.display())))?;
    let root = unpickle(pkl).map_err(|e| Error::Other(format!("{}: data.pkl: {e}", path.display())))?;
    let Value::Dict(items) = root else {
        return Err(Error::Other(format!("{}: the checkpoint isn't a dict of tensors", path.display())));
    };
    let mut out = Weights::new();
    for (k, v) in items {
        let (Value::Str(name), Value::Tensor(t)) = (k, v) else { continue };
        let raw = zip.get(&format!("{prefix}data/{}", t.key)).map_err(|e| Error::Other(format!("{name}: {e}")))?;
        out.insert(name.clone(), t.materialise(raw).map_err(|e| Error::Other(format!("{name}: {e}")))?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------- zip

struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<(String, usize, usize)>, // name, local header offset, size
}

fn u16_at(b: &[u8], o: usize) -> std::result::Result<usize, String> {
    b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]) as usize).ok_or_else(|| "truncated zip".into())
}

fn u32_at(b: &[u8], o: usize) -> std::result::Result<usize, String> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize).ok_or_else(|| "truncated zip".into())
}

impl<'a> Zip<'a> {
    fn parse(bytes: &'a [u8]) -> std::result::Result<Zip<'a>, String> {
        // The end of central directory record is 22 bytes plus a comment of up to 64 KiB, at the very end.
        let lo = bytes.len().saturating_sub(22 + 65535);
        let eocd = (lo..bytes.len().saturating_sub(21))
            .rev()
            .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .ok_or("not a zip file")?;
        let count = u16_at(bytes, eocd + 10)?;
        let mut at = u32_at(bytes, eocd + 16)?;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(bytes, at)? != 0x02014b50 {
                return Err("bad zip central directory".into());
            }
            let method = u16_at(bytes, at + 10)?;
            let size = u32_at(bytes, at + 24)?;
            let (n, m, k) = (u16_at(bytes, at + 28)?, u16_at(bytes, at + 30)?, u16_at(bytes, at + 32)?);
            let local = u32_at(bytes, at + 42)?;
            let name = String::from_utf8_lossy(bytes.get(at + 46..at + 46 + n).ok_or("truncated zip")?).into_owned();
            if method != 0 {
                return Err(format!("{name} is compressed (torch.save stores entries uncompressed)"));
            }
            if size == 0xffff_ffff || local == 0xffff_ffff {
                return Err(format!("{name}: zip64 entries aren't supported"));
            }
            entries.push((name, local, size));
            at += 46 + n + m + k;
        }
        Ok(Zip { bytes, entries })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.0.as_str())
    }

    fn get(&self, name: &str) -> std::result::Result<&'a [u8], String> {
        let &(_, local, size) = self.entries.iter().find(|e| e.0 == name).ok_or(format!("no {name} in the archive"))?;
        if u32_at(self.bytes, local)? != 0x04034b50 {
            return Err(format!("{name}: bad local header"));
        }
        let start = local + 30 + u16_at(self.bytes, local + 26)? + u16_at(self.bytes, local + 28)?;
        self.bytes.get(start..start + size).ok_or(format!("{name}: truncated"))
    }
}

// ----------------------------------------------------------------------------------------------------- unpickler

#[derive(Clone, Copy, Debug, PartialEq)]
enum Storage {
    F32,
    F16,
    BF16,
    F64,
}

#[derive(Clone, Debug)]
struct TensorRef {
    key: String,
    storage: Storage,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl TensorRef {
    fn materialise(&self, raw: &[u8]) -> std::result::Result<Tensor, String> {
        let size = match self.storage {
            Storage::F32 => 4,
            Storage::F16 | Storage::BF16 => 2,
            Storage::F64 => 8,
        };
        let read = |i: usize| -> std::result::Result<f32, String> {
            let b = raw.get(i * size..(i + 1) * size).ok_or("tensor runs past its storage")?;
            Ok(match self.storage {
                Storage::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                Storage::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                Storage::BF16 => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
                Storage::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            })
        };
        let n: usize = self.shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..n {
            let off = self.offset + idx.iter().zip(&self.stride).map(|(i, s)| i * s).sum::<usize>();
            data.push(read(off)?);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Tensor { shape: self.shape.clone(), data })
    }
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(#[allow(dead_code)] bool),
    Int(i64),
    Float(#[allow(dead_code)] f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    /// A persistent id's tuple: ('storage', storage type, key, location, numel).
    Persistent(Vec<Value>),
    Tensor(TensorRef),
    /// Anything else a callable made (ignored).
    Opaque,
    Mark,
}

fn as_ints(v: &Value) -> std::result::Result<Vec<usize>, String> {
    match v {
        Value::Tuple(xs) | Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Int(i) if *i >= 0 => Ok(*i as usize),
                _ => Err("expected a tuple of sizes".to_string()),
            })
            .collect(),
        _ => Err("expected a tuple".into()),
    }
}

fn rebuild_tensor(args: &[Value]) -> std::result::Result<Value, String> {
    let [Value::Persistent(p), Value::Int(offset), size, stride, ..] = args else {
        return Err("unexpected _rebuild_tensor_v2 arguments".into());
    };
    let (Some(Value::Global(_, ty)), Some(Value::Str(key))) = (p.get(1), p.get(2)) else {
        return Err("unexpected storage id".into());
    };
    let storage = match ty.as_str() {
        "FloatStorage" => Storage::F32,
        "HalfStorage" => Storage::F16,
        "BFloat16Storage" => Storage::BF16,
        "DoubleStorage" => Storage::F64,
        other => return Err(format!("tensors of {other} aren't supported")),
    };
    Ok(Value::Tensor(TensorRef {
        key: key.clone(),
        storage,
        offset: *offset as usize,
        shape: as_ints(size)?,
        stride: as_ints(stride)?,
    }))
}

fn unpickle(b: &[u8]) -> std::result::Result<Value, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<usize, Value> = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> std::result::Result<&[u8], String> {
        let s = b.get(*i..*i + n).ok_or("truncated pickle")?;
        *i += n;
        Ok(s)
    };
    let line = |i: &mut usize| -> std::result::Result<String, String> {
        let end = b[*i..].iter().position(|&c| c == b'\n').ok_or("truncated pickle")? + *i;
        let s = String::from_utf8_lossy(&b[*i..end]).into_owned();
        *i = end + 1;
        Ok(s)
    };
    fn pop_mark(stack: &mut Vec<Value>) -> std::result::Result<Vec<Value>, String> {
        let m = stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or("no mark")?;
        let items = stack.split_off(m + 1);
        stack.pop();
        Ok(items)
    }
    loop {
        let op = *b.get(i).ok_or("pickle has no STOP")?;
        i += 1;
        match op {
            0x80 => i += 1, // PROTO
            0x95 => i += 8, // FRAME
            b'.' => return stack.pop().ok_or_else(|| "empty stack at STOP".into()),
            b'(' => stack.push(Value::Mark),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => stack.push(Value::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => {
                let s = take(&mut i, 2)?;
                stack.push(Value::Int(u16::from_le_bytes([s[0], s[1]]) as i64))
            }
            b'J' => {
                let s = take(&mut i, 4)?;
                stack.push(Value::Int(i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as i64))
            }
            0x8a => {
                // LONG1
                let n = take(&mut i, 1)?[0] as usize;
                let s = take(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &c) in s.iter().enumerate().take(8) {
                    v |= (c as i64) << (8 * k);
                }
                if n > 0 && n < 8 && s[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Value::Int(v))
            }
            b'G' => {
                let s = take(&mut i, 8)?;
                stack.push(Value::Float(f64::from_be_bytes(s.try_into().unwrap())))
            }
            b'X' | 0x8c | b'B' | b'C' => {
                // BINUNICODE, SHORT_BINUNICODE, BINBYTES, SHORT_BINBYTES
                let n = if op == b'X' || op == b'B' {
                    let s = take(&mut i, 4)?;
                    u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize
                } else {
                    take(&mut i, 1)?[0] as usize
                };
                let s = take(&mut i, n)?;
                stack.push(Value::Str(String::from_utf8_lossy(s).into_owned()))
            }
            b'c' => {
                let module = line(&mut i)?;
                let name = line(&mut i)?;
                stack.push(Value::Global(module, name))
            }
            0x93 => {
                // STACK_GLOBAL
                let name = stack.pop();
                let module = stack.pop();
                match (module, name) {
                    (Some(Value::Str(m)), Some(Value::Str(n))) => stack.push(Value::Global(m, n)),
                    _ => return Err("bad STACK_GLOBAL".into()),
                }
            }
            b')' => stack.push(Value::Tuple(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Value::Tuple(items))
            }
            b'l' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Value::List(items))
            }
            0x85..=0x87 => {
                // TUPLE1..3
                let n = (op - 0x84) as usize;
                if stack.len() < n {
                    return Err("stack underflow".into());
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(Value::Tuple(items))
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as usize;
                memo.insert(k, stack.last().cloned().ok_or("empty stack")?);
            }
            b'r' => {
                let s = take(&mut i, 4)?;
                memo.insert(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize, stack.last().cloned().ok_or("empty stack")?);
            }
            0x94 => {
                // MEMOIZE
                let k = memo.len();
                memo.insert(k, stack.last().cloned().ok_or("empty stack")?);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as usize;
                stack.push(memo.get(&k).cloned().ok_or("bad memo")?)
            }
            b'j' => {
                let s = take(&mut i, 4)?;
                let k = u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
                stack.push(memo.get(&k).cloned().ok_or("bad memo")?)
            }
            b'Q' => {
                // BINPERSID
                match stack.pop() {
                    Some(Value::Tuple(t)) => stack.push(Value::Persistent(t)),
                    _ => return Err("bad persistent id".into()),
                }
            }
            b'R' => {
                let args = stack.pop().ok_or("stack underflow")?;
                let f = stack.pop().ok_or("stack underflow")?;
                let Value::Tuple(args) = args else { return Err("REDUCE without a tuple".into()) };
                let v = match &f {
                    Value::Global(m, n) if m == "collections" && n == "OrderedDict" => Value::Dict(Vec::new()),
                    Value::Global(m, n) if m == "torch._utils" && n == "_rebuild_tensor_v2" => rebuild_tensor(&args)?,
                    _ => Value::Opaque,
                };
                stack.push(v)
            }
            b'b' => {
                stack.pop().ok_or("stack underflow")?; // BUILD, whose state (a dict's metadata) isn't needed
            }
            b's' => {
                let v = stack.pop().ok_or("stack underflow")?;
                let k = stack.pop().ok_or("stack underflow")?;
                match stack.last_mut() {
                    Some(Value::Dict(d)) => d.push((k, v)),
                    _ => return Err("SETITEM on a non-dict".into()),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                let Some(Value::Dict(d)) = stack.last_mut() else { return Err("SETITEMS on a non-dict".into()) };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    d.push((k, v));
                }
            }
            b'a' => {
                let v = stack.pop().ok_or("stack underflow")?;
                if let Some(Value::List(l)) = stack.last_mut() {
                    l.push(v)
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Value::List(l)) = stack.last_mut() {
                    l.extend(items)
                }
            }
            b'0' => {
                stack.pop();
            }
            b'2' => {
                let v = stack.last().cloned().ok_or("stack underflow")?;
                stack.push(v)
            }
            other => return Err(format!("unsupported pickle opcode 0x{other:02x} at {}", i - 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_422_lite_checkpoint() {
        let path = default_model_dir().join("flownet_v4.22.lite.pkl");
        if !path.exists() {
            eprintln!("skipped: {} not downloaded", path.display());
            return;
        }
        let w = load(&path).unwrap();
        assert_eq!(w.len(), 168);
        let t = &w["module.block0.conv0.0.0.weight"];
        assert_eq!(t.shape, vec![96, 15, 3, 3]);
        assert_eq!(w["module.encode.cnn3.weight"].shape, vec![16, 4, 4, 4]);
        assert_eq!(w["module.block3.lastconv.0.bias"].numel(), 52);
        assert!(t.data.iter().all(|v| v.is_finite()) && t.data.iter().any(|&v| v != 0.0));
    }
}
