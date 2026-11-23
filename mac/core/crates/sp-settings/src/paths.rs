//! Where SmoovPlay keeps its own files (framegen/paths.py): settings, card timing, clips being saved.
//!
//! macOS: ~/Library/Application Support/framegen. Windows: %APPDATA%\framegen, as the launcher expects. Elsewhere:
//! $APPDATA/framegen if it's set, else ~/framegen. (The user's looks are elsewhere, in looks::looks_dir().)

use std::path::PathBuf;

/// The user's home folder, as Path.home() finds it.
pub fn home() -> PathBuf {
    std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn appdata() -> Option<PathBuf> {
    std::env::var_os("APPDATA").filter(|a| !a.is_empty()).map(PathBuf::from)
}

/// data_dir().
pub fn data_dir() -> PathBuf {
    if cfg!(target_os = "macos") && appdata().is_none() {
        return home().join("Library").join("Application Support").join("framegen");
    }
    appdata().unwrap_or_else(home).join("framegen")
}
