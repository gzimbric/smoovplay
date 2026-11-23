//! Test mode's report file (run.py write_results): a header, then each tested session's engine line and
//! Stats::results(). The launcher shows it after a session and adds it to Copy diagnostics. Written whole each time,
//! via a temporary file, so it's never seen half-written.

use std::path::Path;

use crate::stats::{Stats, TICK};

/// run.py PRIVATE_RESULTS.
pub const PRIVATE_RESULTS: &str = "These results include NVIDIA VFG. NVIDIA's licence doesn't allow sharing them, so \
they stay on this PC: Copy diagnostics leaves them out.";

/// What a session ran with (run.py's session config).
#[derive(Clone, Debug, PartialEq)]
pub struct SessionInfo {
    /// "metal", "trt", "torch", "dml"...
    pub engine: String,
    /// RIFE model ("4.22.lite"), or "vfg".
    pub model: String,
    /// Motion detail: 1.0 Full, 0.5 Half, 0.25 Quarter.
    pub motion: f64,
    /// NVIDIA VFG's quality's display name (model "vfg" only).
    pub vfg_quality: String,
}

/// What the header says about this machine and run.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportHeader {
    /// time.strftime('%Y-%m-%d %H:%M').
    pub stamp: String,
    /// "1920x1080".
    pub size: String,
    /// "Apple M5 Max".
    pub gpu: String,
    /// The upscaler setting's display name.
    pub upscaler: String,
}

fn mmss(seconds: f64) -> String {
    let s = seconds as i64;
    format!("{}:{:02}", s.div_euclid(60), s.rem_euclid(60))
}

/// Python's f"{x:g}".
fn g(x: f64) -> String {
    let s = format!("{x}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

/// Python's str(float): 1.0, 0.5.
fn py_str(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 { format!("{x:.1}") } else { format!("{x}") }
}

/// The report's lines, or None when no session was tested (none simulated a 30 fps game or scored a frame).
pub fn results_lines(header: &ReportHeader, sessions: &[(SessionInfo, &Stats)]) -> Option<Vec<String>> {
    let sessions: Vec<&(SessionInfo, &Stats)> =
        sessions.iter().filter(|(_, st)| st.simulated > 0 || !st.psnr_interp.is_empty()).collect();
    if sessions.is_empty() {
        return None;
    }
    let played = sessions.iter().map(|(_, st)| st.busy_ms.len()).sum::<usize>() as f64 * TICK;
    let simulated = sessions.iter().map(|(_, st)| st.simulated).sum::<u64>() as f64 * TICK;
    let mut lines = vec![
        format!("SmoovPlay test results, {}", header.stamp),
        format!(
            "Stream: {}, played {}{}",
            header.size,
            mmss(played),
            if simulated != 0.0 { format!(", {} of it as a simulated 30 fps game", mmss(simulated)) } else { String::new() }
        ),
        format!("Graphics card: {}", header.gpu),
        format!("Upscaler: {}", header.upscaler),
    ];
    if sessions.iter().any(|(cfg, _)| cfg.model == "vfg") {
        lines.push(PRIVATE_RESULTS.into());
    }
    let many = sessions.len() > 1;
    for (i, (cfg, st)) in sessions.iter().enumerate() {
        let head = if many { format!("Part {}: ", i + 1) } else { "AI engine: ".into() };
        if cfg.engine == "dml" {
            lines.push(String::new());
            lines.push(format!("{head}DirectML (Experimental), motion detail {}, RIFE {}", py_str(cfg.motion), cfg.model));
            continue;
        }
        if cfg.model == "vfg" {
            lines.push(String::new());
            lines.push(format!("{head}NVIDIA VFG, quality {}", cfg.vfg_quality));
            lines.extend(st.results(Some(false)));
            continue;
        }
        let engine = match cfg.engine.as_str() {
            "trt" => "TensorRT",
            "metal" => "Metal",
            _ => "PyTorch",
        };
        let motion = if cfg.motion == 1.0 {
            "Full".to_string()
        } else if cfg.motion == 0.5 {
            "Half".to_string()
        } else if cfg.motion == 0.25 {
            "Quarter".to_string()
        } else {
            format!("{}x", g(cfg.motion))
        };
        lines.push(String::new());
        lines.push(format!("{head}{engine}, motion detail {motion}, RIFE {}", cfg.model));
        lines.extend(st.results(Some(cfg.motion >= 1.0)));
    }
    Some(lines)
}

/// Writes the report to `path` (through path.tmp, then renamed). False when nothing was tested or it couldn't be
/// written.
pub fn write_results(path: &Path, header: &ReportHeader, sessions: &[(SessionInfo, &Stats)]) -> bool {
    let Some(lines) = results_lines(header, sessions) else { return false };
    let write = || -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        let tmp = path.with_file_name(name);
        std::fs::write(&tmp, lines.join("\n") + "\n")?;
        std::fs::rename(&tmp, path)
    };
    match write() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[results] couldn't write {}: {e}", path.display());
            false
        }
    }
}
