//! The codec header saved for a later run that joins the same stream (framegen/chiaki_tap.py `_save_header`,
//! `_load_header`): the file `<temp dir>/<name after its last backslash>-header.bin`, holding u32 session, u32 codec
//! (little-endian), then the parameter sets as [`crate::nal::param_sets`] gives them. The temp dir is found the way
//! Python's `tempfile.gettempdir()` finds it, so Python and Rust readers share the file.

use std::path::{Path, PathBuf};

/// Python's `tempfile.gettempdir()` on macOS: the first of $TMPDIR, $TEMP, $TMP that's a writable folder, else
/// /tmp, /var/tmp, /usr/tmp, else the current folder.
pub fn temp_dir() -> PathBuf {
    let mut dirs: Vec<PathBuf> = ["TMPDIR", "TEMP", "TMP"]
        .iter()
        .filter_map(std::env::var_os)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .collect();
    dirs.extend(["/tmp", "/var/tmp", "/usr/tmp"].iter().map(PathBuf::from));
    for d in dirs {
        if writable(&d) {
            return absolute(&d);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn writable(d: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    if !d.is_dir() {
        return false;
    }
    let c = CString::new(d.as_os_str().as_bytes()).unwrap();
    unsafe { libc::access(c.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

/// os.path.abspath: absolute and normalised (no trailing "/", "." or "..") without resolving links.
fn absolute(d: &Path) -> PathBuf {
    let d = if d.is_absolute() { d.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(d) };
    let mut out = PathBuf::from("/");
    for c in d.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::Normal(p) => out.push(p),
            _ => {}
        }
    }
    out
}

/// The file for the tap `tap_name`.
pub fn header_path(tap_name: &str) -> PathBuf {
    let base = tap_name.rsplit('\\').next().unwrap_or("").trim_start_matches('/');
    temp_dir().join(format!("{base}-header.bin"))
}

pub fn save(tap_name: &str, session: u32, codec: u32, params: &[u8]) {
    let mut b = Vec::with_capacity(8 + params.len());
    b.extend_from_slice(&session.to_le_bytes());
    b.extend_from_slice(&codec.to_le_bytes());
    b.extend_from_slice(params);
    let _ = std::fs::write(header_path(tap_name), b);
}

/// The codec header saved for this same stream (session and codec), or empty.
pub fn load(tap_name: &str, session: u32, codec: u32) -> Vec<u8> {
    match std::fs::read(header_path(tap_name)) {
        Ok(b) if b.len() > 8
            && u32::from_le_bytes(b[0..4].try_into().unwrap()) == session
            && u32::from_le_bytes(b[4..8].try_into().unwrap()) == codec =>
        {
            b[8..].to_vec()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load() {
        let name = format!("Local\\sp-rs-h{}", std::process::id());
        save(&name, 7, 1, b"\x00\x00\x01\x40\x01vps");
        assert_eq!(load(&name, 7, 1), b"\x00\x00\x01\x40\x01vps");
        assert!(load(&name, 8, 1).is_empty() && load(&name, 7, 0).is_empty());
        assert!(header_path(&name).ends_with(format!("sp-rs-h{}-header.bin", std::process::id())));
        let _ = std::fs::remove_file(header_path(&name));
    }
}
