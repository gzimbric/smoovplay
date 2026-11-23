//! The video window as the pipeline's presenter: sp-present's VideoPresenter with the launcher's link polled on the
//! window's thread before each frame and wait (io.py GLWindowSink polls framegen/remote.py in its events), so its
//! changes act exactly like the menu's. Besides the Python's commands, the Mac app's:
//!
//! - `key <name>`: a key as if pressed in the window (M, F, I, S, U, Z, K, J, L, R, B, Q; Up, Down, Left, Right,
//!   Enter, Esc, Backspace, Tab; a letter is the same in either case, "Shift+J" is the previous look); `key` alone
//!   does nothing (the app's probe).
//! - `overlay <x> <y> <w> <h>` / `overlay none`: sit over the copied app window, or be a normal window again.
//! - "menu_open" in the state, published again whenever the menu opens or closes.

use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sp_core::{Frame, Overlay, Presenter, SinkReply, Waiting};
use sp_present::{Key, VideoPresenter};
use sp_settings::remote::{LineSender, Remote, Request};

use crate::adapt::{SharedLooks, SharedSettings, lock, lock_any};

/// What a command asked the window for, applied once the settings are let go.
#[derive(Clone, Debug, PartialEq)]
pub enum Cmd {
    Key(Key, bool),
    Overlay(Option<(f64, f64, f64, f64)>),
}

/// `key <name>`'s names.
pub fn key_of(name: &str) -> Result<Option<(Key, bool)>, String> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    let lower = name.to_ascii_lowercase();
    let (shift, base) = match lower.strip_prefix("shift+").or_else(|| lower.strip_prefix("shift-")) {
        Some(b) => (true, b.to_string()),
        None => (false, lower.clone()),
    };
    let key = match base.as_str() {
        "up" => Key::Up,
        "down" => Key::Down,
        "left" => Key::Left,
        "right" => Key::Right,
        "enter" | "return" => Key::Enter,
        "esc" | "escape" => Key::Esc,
        "backspace" | "delete" => Key::Backspace,
        "tab" => Key::Tab,
        b if b.chars().count() == 1 && b.chars().all(|c| c.is_ascii_alphanumeric()) => Key::Char(b.chars().next().unwrap()),
        _ => return Err(format!("unknown key {name:?}")),
    };
    Ok(Some((key, shift)))
}

/// `overlay x y w h` / `overlay none`.
pub fn overlay_of(text: &str) -> Result<Option<(f64, f64, f64, f64)>, String> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let v: Vec<f64> = t.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    match v[..] {
        [x, y, w, h] if v.len() == t.split_whitespace().count() && w > 0.0 && h > 0.0 => Ok(Some((x, y, w, h))),
        _ => Err(format!("overlay needs x y w h (points) or none, not {:?}", t.chars().take(60).collect::<String>())),
    }
}

/// The launcher's link: the Remote, and what its handlers left for the window.
pub struct Control {
    pub remote: Remote,
    pub cmds: Arc<Mutex<Vec<Cmd>>>,
    /// looks-reload: publish again once the settings are free (the new list of looks).
    pub republish: Arc<AtomicBool>,
}

impl Control {
    /// The Remote with run.py's handlers (the window's remote_commands()) and the Mac app's.
    pub fn new(mut remote: Remote, looks: SharedLooks) -> Control {
        let cmds = Arc::new(Mutex::new(Vec::new()));
        let republish = Arc::new(AtomicBool::new(false));
        let l1 = looks.clone();
        remote.add_handler("look-show", move |_s, text| {
            let r = match sp_settings::json::loads(text) {
                Ok(v) => lock_any(&l1).state.show(Some(&v)).map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            if let Err(e) = r {
                println!("[looks] can't show that look: {e}");
            }
            Ok(())
        });
        let l2 = looks.clone();
        remote.add_handler("look-hide", move |_s, _text| {
            let _ = lock_any(&l2).state.show(None);
            Ok(())
        });
        let l3 = looks;
        let rp = republish.clone();
        remote.add_handler("looks-reload", move |s, _text| {
            let mut g = lock_any(&l3);
            let crate::adapt::Looks { state, library } = &mut *g;
            state.reload(s, library);
            rp.store(true, Ordering::SeqCst);
            Ok(())
        });
        let c1 = cmds.clone();
        remote.add_handler("key", move |_s, text| {
            if let Some((k, shift)) = key_of(text)? {
                lock_any(&c1).push(Cmd::Key(k, shift));
            }
            Ok(())
        });
        let c2 = cmds.clone();
        remote.add_handler("overlay", move |_s, text| {
            lock_any(&c2).push(Cmd::Overlay(overlay_of(text)?));
            Ok(())
        });
        Control { remote, cmds, republish }
    }
}

/// Reads the launcher's lines from `input` (stdin) on a thread; when it closes (the app went), asks to quit, so the
/// core never outlives the app.
pub fn read_commands(tx: LineSender, input: impl Read + Send + 'static) {
    let _ = std::thread::Builder::new().name("remote".into()).spawn(move || {
        let mut r = BufReader::new(input);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => tx.send(&String::from_utf8_lossy(&buf)),
            }
        }
        println!("[remote] the launcher closed its link: quitting");
        tx.send("quit");
    });
}

/// The window the pipeline presents to.
pub struct Window {
    pub p: VideoPresenter,
    pub settings: SharedSettings,
    pub control: Option<Control>,
    pub menu_open: Arc<AtomicBool>,
    /// Messages from other threads (instant replay's), shown at the next frame.
    pub toasts: Arc<Mutex<Vec<(String, f64)>>>,
    /// The look the window shows now (MenuAdapter::display), for "Looks in clips".
    pub shown_look: Arc<Mutex<Option<Arc<sp_present::looks::Look>>>>,
}

impl Window {
    /// The state line (session start, as run.py's remote.publish()).
    pub fn publish(&self) {
        if let Some(c) = &self.control {
            c.remote.publish(&lock(&self.settings));
        }
    }

    /// The launcher's commands since the last call, then the window's own (keys, overlay).
    fn poll(&mut self) -> Option<Request> {
        for (text, secs) in std::mem::take(&mut *lock_any(&self.toasts)) {
            self.p.toast(&text, secs);
        }
        let c = self.control.as_mut()?;
        let req = {
            let mut s = lock(&self.settings);
            let req = c.remote.poll(&mut s);
            if c.republish.swap(false, Ordering::SeqCst) {
                c.remote.publish(&s);
            }
            req
        };
        let cmds = std::mem::take(&mut *lock_any(&c.cmds));
        for cmd in cmds {
            match cmd {
                Cmd::Key(k, shift) => self.p.press(k, shift),
                Cmd::Overlay(r) => self.p.set_overlay(r),
            }
        }
        req
    }

    /// The menu opened or closed: say so (menu_open).
    fn after(&mut self) {
        let open = self.p.menu_open();
        if open != self.menu_open.swap(open, Ordering::SeqCst) {
            self.publish();
        }
    }

    pub fn busy<T: Send>(&mut self, text: &str, f: impl FnOnce() -> T + Send) -> T {
        let out = self.p.busy(text, f);
        self.after();
        out
    }
}

/// The launcher's request with the window's reply: quit first, then what the window said, then a restart.
fn combine(req: Option<Request>, reply: SinkReply) -> SinkReply {
    match (req, reply) {
        (Some(Request::Quit), _) => SinkReply::Quit,
        (_, r) if r != SinkReply::Continue => r,
        (Some(Request::Restart), _) => SinkReply::Restart,
        _ => SinkReply::Continue,
    }
}

impl Presenter for Window {
    fn write(&mut self, frame: &Frame, overlay: Option<&Overlay>) -> SinkReply {
        let req = self.poll();
        let r = self.p.write(frame, overlay);
        self.after();
        combine(req, r)
    }
    fn wait(&mut self, w: &Waiting) -> SinkReply {
        let req = self.poll();
        let r = self.p.wait(w);
        self.after();
        combine(req, r)
    }
    fn toast(&mut self, text: &str, seconds: f64) {
        self.p.toast(text, seconds);
    }
    fn close(&mut self) {
        self.p.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names() {
        assert_eq!(key_of("M").unwrap(), Some((Key::Char('m'), false)));
        assert_eq!(key_of("j").unwrap(), Some((Key::Char('j'), false)));
        assert_eq!(key_of("Shift+J").unwrap(), Some((Key::Char('j'), true)));
        assert_eq!(key_of("Escape").unwrap(), Some((Key::Esc, false)));
        assert_eq!(key_of("Esc").unwrap(), Some((Key::Esc, false)));
        assert_eq!(key_of("Backspace").unwrap(), Some((Key::Backspace, false)));
        assert_eq!(key_of("Enter").unwrap(), Some((Key::Enter, false)));
        assert_eq!(key_of("").unwrap(), None);
        assert!(key_of("F13").is_err());
    }

    #[test]
    fn overlay_args() {
        assert_eq!(overlay_of("none").unwrap(), None);
        assert_eq!(overlay_of("10 20 640 360").unwrap(), Some((10.0, 20.0, 640.0, 360.0)));
        assert!(overlay_of("10 20 640").is_err());
        assert!(overlay_of("10 20 0 360").is_err());
    }
}
