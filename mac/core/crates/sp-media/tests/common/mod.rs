//! Shared by sp-media's tests, for the reference data mac/core/tools/media_fixtures.py makes.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/media")
}

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf()
}

pub fn python() -> Option<PathBuf> {
    let p = repo().join(".venv/bin/python");
    p.exists().then_some(p)
}

/// The access units of fixtures/media/<name>.annexb, or None if the fixtures haven't been made.
pub fn packets(name: &str) -> Option<Vec<Vec<u8>>> {
    let b = std::fs::read(fixtures().join(format!("{name}.annexb"))).ok()?;
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        let n = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        out.push(b[i + 4..i + 4 + n].to_vec());
        i += 4 + n;
    }
    Some(out)
}

pub fn read(name: &str) -> Vec<u8> {
    std::fs::read(fixtures().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// IEEE half to f32.
pub fn half(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// The max, the mean and the share over 1 of |a - b| over bytes.
pub fn diff(a: &[u8], b: &[u8]) -> (u8, f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut max = 0u8;
    let mut sum = 0u64;
    let mut over = 0usize;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += d as u64;
        over += (d > 1) as usize;
    }
    (max, sum as f64 / a.len() as f64, over as f64 / a.len() as f64)
}
