//! Where a packaged copy finds its files. When `smoovplay` runs from inside SmoovPlay.app (as
//! `Contents/Helpers/smoovplay`), RIFE's weights and the looks preview's pictures come from the app's `Resources`
//! folder. Nothing is ever written into the app, since that would break its signature, so anything downloaded
//! later goes to the data folder beside the settings instead.

use std::path::{Path, PathBuf};

/// The app's `Contents` folder, when this program runs from inside an app bundle.
fn contents_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let contents = exe.parent()?.parent()?;
    let is_bundle = contents.file_name()? == "Contents" && contents.join("Info.plist").is_file();
    is_bundle.then(|| contents.to_path_buf())
}

/// The app's `Resources` folder, or None when this program isn't running from inside an app.
pub fn resources_dir() -> Option<PathBuf> {
    contents_dir().map(|c| c.join("Resources"))
}

/// Whether `path` is inside the app this program runs from.
pub fn contains(path: &Path) -> bool {
    match contents_dir() {
        Some(contents) => {
            let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            let contents = std::fs::canonicalize(&contents).unwrap_or(contents);
            path.starts_with(contents)
        }
        None => false,
    }
}

/// SmoovPlay's data folder, `~/Library/Application Support/framegen`, where the settings live.
pub fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join("Library").join("Application Support").join("framegen")
}
