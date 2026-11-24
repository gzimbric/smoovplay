//! Opening the source run.py's flags ask for, and the handles the session needs besides the stream itself. The
//! pipeline owns the source while it plays, so chiaki-ng's reader is shared with the delay test's presses, the frame
//! log and the restarts' hold(). Its mutex is taken a tick at a time, always on the pipeline's thread.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use sp_core::{Event, Frame, Source, SourceInfo, Sound};
use sp_media::{ChiakiOptions, ChiakiSource, FileSource, OpenedTap, Ps5SimSource, SharedTestMode, SimOptions, TapSource};
use sp_sched::delaytest::Press;
use sp_sched::Remote;

use crate::adapt::lock_any;
use crate::args::Args;

/// A source behind a mutex, for the pipeline, with its info kept beside it (Source::info hands out a reference,
/// which can't come through the lock).
pub struct WithInfo<S: Source> {
    pub inner: Arc<Mutex<S>>,
    pub info: SourceInfo,
}

impl<S: Source> Source for WithInfo<S> {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        lock_any(&self.inner).size()
    }
    fn next_event(&mut self) -> Option<Event> {
        lock_any(&self.inner).next_event()
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        lock_any(&self.inner).ground_truth(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        lock_any(&self.inner).read_sound()
    }
    fn request_keyframe(&mut self) {
        lock_any(&self.inner).request_keyframe()
    }
    fn set_audio_delay(&mut self, ms: f64) {
        lock_any(&self.inner).set_audio_delay(ms)
    }
    fn close(&mut self) {
        lock_any(&self.inner).close()
    }
}

/// --max-ticks for a live source (a file's own reader counts them itself, as run.py's do).
pub struct Limit {
    pub inner: Box<dyn Source>,
    pub left: u64,
}

impl Source for Limit {
    fn info(&self) -> &SourceInfo {
        self.inner.info()
    }
    fn size(&self) -> (usize, usize) {
        self.inner.size()
    }
    fn next_event(&mut self) -> Option<Event> {
        if self.left == 0 {
            return None;
        }
        let e = self.inner.next_event()?;
        if matches!(e, Event::Tick(_)) {
            self.left -= 1;
        }
        Some(e)
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.inner.ground_truth(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        self.inner.read_sound()
    }
    fn request_keyframe(&mut self) {
        self.inner.request_keyframe()
    }
    fn set_audio_delay(&mut self, ms: f64) {
        self.inner.set_audio_delay(ms)
    }
    fn close(&mut self) {
        self.inner.close()
    }
}

/// chiaki-ng's presses and sound delay for the pipeline (the delay test, Sound in sync).
pub struct ChiakiRemote(pub Arc<Mutex<ChiakiSource>>);

impl Press for ChiakiRemote {
    fn press(&mut self, buttons: u32, hold_ms: u32) -> u64 {
        lock_any(&self.0).press(buttons, hold_ms) as u64
    }
    fn press_sent(&mut self, token: u64) -> Option<f64> {
        lock_any(&self.0).press_sent(token as u32)
    }
}

impl Remote for ChiakiRemote {
    fn can_press(&self) -> bool {
        lock_any(&self.0).can_press()
    }
    fn has_press(&self) -> bool {
        true
    }
    fn can_delay_audio(&self) -> bool {
        lock_any(&self.0).can_delay_audio()
    }
    fn set_audio_delay(&mut self, ms: i64) {
        lock_any(&self.0).set_audio_delay(ms as f64);
    }
}

/// A capture card's side for the pipeline. Sound in sync holds its sound back. It has no presses, because the delay
/// test is Remote Play's.
pub struct CardRemote(pub Arc<Mutex<sp_capture::CardSource>>);

impl Press for CardRemote {
    fn press(&mut self, _buttons: u32, _hold_ms: u32) -> u64 {
        0
    }
    fn press_sent(&mut self, _token: u64) -> Option<f64> {
        None
    }
}

impl Remote for CardRemote {
    fn has_press(&self) -> bool {
        false
    }
    fn can_delay_audio(&self) -> bool {
        lock_any(&self.0).can_delay_audio()
    }
    fn set_audio_delay(&mut self, ms: i64) {
        lock_any(&self.0).set_audio_delay(ms as f64);
    }
}

/// The source of a run and what the session needs from it besides its ticks.
pub struct Opened {
    pub src: Box<dyn Source>,
    pub info: SourceInfo,
    pub w: usize,
    pub h: usize,
    /// "chiaki" (also an app's compressed frames on the tap), "tap" (decoded frames) or "file".
    pub kind: String,
    /// chiaki-ng's reader (or an app's compressed frames), for log_tick, set_hdr, the delay test and the stream check.
    pub chiaki: Option<Arc<Mutex<ChiakiSource>>>,
    /// Stop decoding while a restart rebuilds the pipeline (on), and start from a fresh frame (off).
    pub hold: Option<Box<dyn FnMut(bool) + Send>>,
    /// HDR10 from the PS5 (chiaki-ng's H.265 HDR codec).
    pub hdr_stream: bool,
    /// The game's sound since the last call, for instant replay (run.py's src.read_sound).
    pub sound: Option<Box<dyn FnMut() -> Vec<Sound> + Send>>,
    /// The other way, for the pipeline: chiaki-ng's presses and sound delay, or the card's sound delay.
    pub remote: Option<Box<dyn Remote + Send>>,
}

fn chiaki(src: ChiakiSource, kind: &str) -> Opened {
    let info = src.info().clone();
    let (w, h) = src.size();
    let hdr_stream = src.hdr_stream;
    let arc = Arc::new(Mutex::new(src));
    let (a2, a3) = (arc.clone(), arc.clone());
    Opened {
        sound: Some(Box::new(move || lock_any(&a3).read_sound())),
        remote: Some(Box::new(ChiakiRemote(arc.clone()))),
        src: Box::new(WithInfo { inner: arc.clone(), info: info.clone() }),
        info,
        w,
        h,
        kind: kind.into(),
        chiaki: Some(arc),
        hold: Some(Box::new(move |on| lock_any(&a2).hold(on))),
        hdr_stream,
    }
}

fn tap(src: TapSource) -> Opened {
    let info = src.info().clone();
    let (w, h) = src.size();
    let arc = Arc::new(Mutex::new(src));
    let (a2, a3) = (arc.clone(), arc.clone());
    Opened {
        sound: Some(Box::new(move || lock_any(&a3).read_sound())),
        remote: None,
        src: Box::new(WithInfo { inner: arc, info: info.clone() }),
        info,
        w,
        h,
        kind: "tap".into(),
        chiaki: None,
        hold: Some(Box::new(move |on| lock_any(&a2).hold(on))),
        hdr_stream: false,
    }
}

/// run.py frame_log(). When played from the launcher (--results), a per-frame log of the stream goes next to the
/// session's own, as frames-<the session's time>.csv. The last 10 are kept.
pub fn frame_log(results: Option<&str>) -> Option<PathBuf> {
    const KEPT: usize = 10;
    let r = PathBuf::from(results?);
    let stem = r.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let stamp = stem.split_once('-').map_or(stem.clone(), |(_, s)| s.to_string());
    let dir = r.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let dir_r = if dir.as_os_str().is_empty() { PathBuf::from(".") } else { dir.clone() };
    if let Ok(rd) = std::fs::read_dir(&dir_r) {
        let mut old: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name().map(|n| n.to_string_lossy().into_owned()).is_some_and(|n| n.starts_with("frames-") && n.ends_with(".csv"))
            })
            .collect();
        old.sort();
        let n = old.len().saturating_sub(KEPT - 1);
        for p in &old[..n] {
            let _ = std::fs::remove_file(p);
        }
    }
    let path = dir.join(format!("frames-{stamp}.csv"));
    println!("[run] frame log -> {}", path.file_name().unwrap().to_string_lossy());
    Some(path)
}

/// Opens what run.py's flags ask for. Err holds the message and the exit code.
pub fn open(a: &Args, kind: &str, test: &Option<SharedTestMode>) -> Result<Opened, (String, i32)> {
    match kind {
        "tap" => {
            let name = a.tap.clone().unwrap_or_else(|| sp_tap::DEFAULT_APP_NAME.to_string());
            let log = a.chiaki_log.as_ref().map(PathBuf::from);
            match sp_media::open_tap(&name, 600.0, test.clone(), log) {
                Err(e) => Err((format!("[tap] {e}"), 1)),
                Ok(OpenedTap::Chiaki(s)) => Ok(chiaki(s, "chiaki")),
                Ok(OpenedTap::Tap(s)) => Ok(tap(s)),
                Ok(OpenedTap::Variable(c)) => {
                    let info = c.info().clone();
                    let (w, h) = c.size();
                    let arc = Arc::new(Mutex::new(c));
                    let (a2, a3) = (arc.clone(), arc.clone());
                    Ok(Opened {
                        sound: Some(Box::new(move || lock_any(&a3).read_sound())),
                        remote: None,
                        src: Box::new(WithInfo { inner: arc, info: info.clone() }),
                        info,
                        w,
                        h,
                        kind: "tap".into(),
                        chiaki: None,
                        hold: Some(Box::new(move |on| lock_any(&a2).source.hold(on))),
                        hdr_stream: false,
                    })
                }
            }
        }
        "chiaki" => {
            let log = a.chiaki_log.as_ref().map(PathBuf::from).or_else(|| frame_log(a.results.as_deref()));
            let opts = ChiakiOptions {
                tap_name: a.chiaki_tap.clone().unwrap_or_else(|| sp_tap::NAME.to_string()),
                wait_s: 600.0,
                log_path: log,
                test: test.clone(),
                app: false,
                stale_s: sp_media::STALE_S,
            };
            match ChiakiSource::open(opts) {
                Ok(s) => Ok(chiaki(s, "chiaki")),
                Err(e) => Err((format!("[chiaki] {e}"), 1)),
            }
        }
        "file" => {
            let path = PathBuf::from(a.video.as_deref().unwrap_or(""));
            let src: Box<dyn Source> = if a.raw {
                Box::new(FileSource::open(&path, a.realtime, a.max_ticks, test.clone()).map_err(|e| (format!("[run] {e}"), 1))?)
            } else {
                let opts = SimOptions { realtime: a.realtime, hitch: a.hitch, noise: a.noise, max_ticks: a.max_ticks, seed: 0 };
                Box::new(Ps5SimSource::open(&path, opts).map_err(|e| (format!("[run] {e}"), 1))?)
            };
            let info = src.info().clone();
            let (w, h) = src.size();
            Ok(Opened { src, info, w, h, kind: "file".into(), chiaki: None, hold: None, hdr_stream: false, sound: None, remote: None })
        }
        "capture" => {
            let (cw, ch) = match &a.capture_size {
                Some(s) => {
                    let (x, y) = s.split_once('x').unwrap_or((s, ""));
                    match (x.trim().parse::<u32>(), y.trim().parse::<u32>()) {
                        (Ok(x), Ok(y)) => (x, y),
                        _ => crate::args::error(&format!("--capture-size {s}: give WIDTHxHEIGHT, like 1920x1080")),
                    }
                }
                None => {
                    // Capture card resolution (the settings are loaded again later, with the command line's overrides)
                    let saved = match a.settings_path() {
                        None => sp_settings::Settings::defaults(),
                        Some(p) => sp_settings::Settings::load(p, sp_settings::LookChoices::builtin()),
                    };
                    sp_settings::settings::capture_size(saved.get_str("capture_size")).unwrap_or((1920, 1080))
                }
            };
            println!(
                "[capture] asking the card for {cw}x{ch}{}",
                if (cw, ch) == (1920, 1080) {
                    ""
                } else {
                    " (Capture card resolution: every new frame is drawn at this size, so it costs more than 1080p)"
                }
            );
            let ca = sp_capture::CaptureArgs {
                capture: a.capture.clone().unwrap_or_default(),
                w: cw,
                h: ch,
                fps: a.capture_fps,
                format: a.capture_format.clone(),
                colour: a.capture_colour.clone(),
                audio: a.capture_audio.clone(),
                test: test.clone(),
            };
            let card = sp_capture::open_capture(&ca, Some(sp_capture::player_factory())).map_err(|e| (format!("[capture] {e}"), 1))?;
            let info = card.info().clone();
            let (w, h) = card.size();
            let arc = Arc::new(Mutex::new(card));
            let a3 = arc.clone();
            Ok(Opened {
                src: Box::new(WithInfo { inner: arc.clone(), info: info.clone() }),
                info,
                w,
                h,
                kind: "capture".into(),
                chiaki: None,
                hold: None,
                hdr_stream: false,
                sound: Some(Box::new(move || lock_any(&a3).read_sound())),
                remote: Some(Box::new(CardRemote(arc))),
            })
        }
        _ => unreachable!("{kind}"),
    }
}
