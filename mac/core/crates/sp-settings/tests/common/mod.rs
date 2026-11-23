//! Shared by the integration tests: temporary folders (inside the target folder, never the real settings or looks),
//! the Python fixtures, and a buffer to collect a Remote's output.
#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sp_settings::json::{self, Value};

static N: AtomicUsize = AtomicUsize::new(0);

/// A new, empty folder for one test.
pub fn tmp_dir(name: &str) -> PathBuf {
    let n = N.fetch_add(1, Ordering::SeqCst);
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("sp-settings-{}-{name}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

/// mac/core/fixtures/settings/<name>.json, written by tools/settings_fixtures.py. None, with a note, if it isn't there.
pub fn fixture(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/settings").join(format!("{name}.json"));
    match std::fs::read_to_string(&p) {
        Ok(text) => Some(json::loads(&text).expect("fixture JSON")),
        Err(_) => {
            eprintln!("SKIPPED: no fixture {} (run .venv/bin/python mac/core/tools/settings_fixtures.py)", p.display());
            None
        }
    }
}

/// The repo's framegen/samples folder, as the Python resolves it.
pub fn samples_dir() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../framegen/samples");
    std::fs::canonicalize(&p).unwrap_or(p)
}

pub fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or_else(|| panic!("{key} in {}", json::dumps(v)))
}

pub fn list<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_list).unwrap_or_else(|| panic!("{key} in {}", json::dumps(v)))
}

pub fn items(v: &Value) -> &[Value] {
    v.as_list().expect("a list")
}

/// A Write that keeps what's written, for a Remote's output.
#[derive(Clone, Default)]
pub struct Buf(pub Arc<Mutex<Vec<u8>>>);

impl Buf {
    pub fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Writes files, given as name and text, into a new folder for a looks library.
pub fn library_from(files: &Value) -> PathBuf {
    let d = tmp_dir("lib");
    if let Value::Dict(m) = files {
        for (name, text) in m.iter() {
            std::fs::write(d.join(name), text.as_str().unwrap()).unwrap();
        }
    }
    d
}

/// Bytes from hex.
pub fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
