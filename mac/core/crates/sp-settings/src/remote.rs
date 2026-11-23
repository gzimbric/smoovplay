//! The launcher's live link to a running session (framegen/remote.py, `run.py --control`), so its Settings page works
//! while you play. The Swift app talks to the Rust core exactly this way.
//!
//! The launcher writes one command per line:
//!
//! ```text
//! set <key> <json value>    change a setting and save it, as if it were picked in the video window's menu
//! restart                   apply restart-only settings now (the menu's "Restart now")
//! quit                      stop, as Esc in the video window does
//! state                     print the current state again
//! look-show <look JSON>     show this look (handlers: the caller's)
//! look-hide                 back to the Look settings' look
//! looks-reload              the user's looks changed on disk
//! ```
//!
//! and reads a `[settings-state] {json}` line at the start of every session and after every change, whoever made it:
//! the current values, which choices can be picked right now, rows that are unavailable and why, and the
//! restart-only changes still waiting. Lines are read on a thread ([`Remote::read_from`]) but applied in
//! [`Remote::poll`], on the thread the menu's keys run on, so they act exactly like menu changes.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use crate::json::{self, Map, Value};
use crate::settings::{Settings, WatchId, spec, specs};

pub const PREFIX: &str = "[settings-state] ";

/// The window's rules, the same ones as its menu. The Python passes three callables; these take the settings too, as
/// a Rust closure can't reach into them while they're being changed.
pub trait Rules: Send {
    /// Whether this choice can be picked right now.
    fn allowed(&self, _settings: &Settings, _key: &str, _value: &Value) -> bool {
        true
    }
    /// Why this row is unavailable (greyed out), or None.
    fn disabled(&self, _settings: &Settings, _key: &str) -> Option<String> {
        None
    }
    /// The labels of restart-only settings changed since the session started.
    fn restart_reasons(&self, _settings: &Settings) -> Vec<String> {
        Vec::new()
    }
    /// Fields the Python's state doesn't have, put after "restart" (the Mac app's "menu_open"). None by default, so
    /// the line stays the Python's byte for byte.
    fn extra(&self, _settings: &Settings) -> Vec<(String, Value)> {
        Vec::new()
    }
}

/// Rules from three closures.
pub struct FnRules<A, D, R> {
    pub allowed: A,
    pub disabled: D,
    pub restart_reasons: R,
}

impl<A, D, R> Rules for FnRules<A, D, R>
where
    A: Fn(&Settings, &str, &Value) -> bool + Send,
    D: Fn(&Settings, &str) -> Option<String> + Send,
    R: Fn(&Settings) -> Vec<String> + Send,
{
    fn allowed(&self, s: &Settings, key: &str, value: &Value) -> bool {
        (self.allowed)(s, key, value)
    }
    fn disabled(&self, s: &Settings, key: &str) -> Option<String> {
        (self.disabled)(s, key)
    }
    fn restart_reasons(&self, s: &Settings) -> Vec<String> {
        (self.restart_reasons)(s)
    }
}

/// Everything allowed, nothing disabled, no restart pending.
pub struct OpenRules;
impl Rules for OpenRules {}

/// What poll() was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Quit,
    Restart,
}

/// A command handler: gets the settings and the rest of the line. An Err is reported, never raised (a bad command
/// must never take the video down).
pub type Handler = Box<dyn FnMut(&mut Settings, &str) -> Result<(), String> + Send>;

type Log = Box<dyn FnMut(&str) + Send>;

struct Shared {
    out: Box<dyn Write + Send>,
    log: Option<Log>,
    rules: Box<dyn Rules>,
}

impl Shared {
    fn state(&self, settings: &Settings) -> Value {
        let looks = settings.look_choices();
        let choices: Map = specs()
            .iter()
            .map(|s| {
                let vs: Vec<Value> =
                    s.choices(looks).into_iter().filter(|v| self.rules.allowed(settings, s.key, v)).collect();
                (s.key, Value::List(vs))
            })
            .collect();
        let disabled: Map = specs()
            .iter()
            .filter_map(|s| {
                self.rules.disabled(settings, s.key).filter(|why| !why.is_empty()).map(|why| (s.key, Value::from(why)))
            })
            .collect();
        let mut state = json::obj([
            ("values", Value::Dict(settings.values())),
            ("choices", Value::Dict(choices)),
            ("disabled", Value::Dict(disabled)),
            ("restart", Value::List(self.rules.restart_reasons(settings).into_iter().map(Value::from).collect())),
        ]);
        if let Value::Dict(m) = &mut state {
            for (k, v) in self.rules.extra(settings) {
                m.insert(k, v);
            }
        }
        state
    }

    fn publish(&mut self, settings: &Settings) {
        let line = format!("{PREFIX}{}\n", json::dumps(&self.state(settings)));
        let _ = self.out.write_all(line.as_bytes()); // one write, so other threads' lines don't split it
        let _ = self.out.flush();
    }

    fn say(&mut self, msg: &str) {
        match &mut self.log {
            Some(f) => f(msg),
            None => {
                let _ = self.out.write_all(format!("{msg}\n").as_bytes());
                let _ = self.out.flush();
            }
        }
    }
}

/// Queues command lines for a [`Remote`] from another thread.
#[derive(Clone)]
pub struct LineSender(Sender<String>);

impl LineSender {
    /// One line as read from the launcher: whitespace and a leading byte-order mark taken off; blank lines dropped.
    pub fn send(&self, line: &str) {
        for part in line.split(['\n', '\r']) {
            let part = crate::py::strip(part).trim_start_matches('\u{feff}');
            if !part.is_empty() {
                let _ = self.0.send(part.to_string());
            }
        }
    }
}

pub struct Remote {
    tx: Sender<String>,
    rx: Receiver<String>,
    shared: Arc<Mutex<Shared>>,
    handlers: Vec<(String, Handler)>,
    watch: Option<WatchId>,
}

impl Remote {
    /// Watches `settings` (each change publishes a state line to `out`) and waits for lines: from
    /// [`Remote::read_from`] (the launcher's pipe, stdin), [`Remote::sender`] or [`Remote::push_line`].
    pub fn new(settings: &mut Settings, rules: impl Rules + 'static, out: impl Write + Send + 'static) -> Remote {
        let (tx, rx) = channel();
        let shared = Arc::new(Mutex::new(Shared { out: Box::new(out), log: None, rules: Box::new(rules) }));
        let s2 = Arc::clone(&shared);
        let watch = settings.watch(move |st, _key| s2.lock().unwrap_or_else(|e| e.into_inner()).publish(st));
        Remote { tx, rx, shared, handlers: Vec::new(), watch: Some(watch) }
    }

    /// Where "[remote] ..." messages go (by default, `out`, as the Python prints them to the same stdout).
    pub fn set_log(&mut self, f: impl FnMut(&str) + Send + 'static) {
        self.lock().log = Some(Box::new(f));
    }

    /// More commands: name -> fn(settings, the rest of the line), e.g. the looks' look-show, look-hide, looks-reload.
    pub fn add_handler(
        &mut self,
        name: &str,
        f: impl FnMut(&mut Settings, &str) -> Result<(), String> + Send + 'static,
    ) {
        self.handlers.retain(|(n, _)| n != name);
        self.handlers.push((name.to_string(), Box::new(f)));
    }

    pub fn sender(&self) -> LineSender {
        LineSender(self.tx.clone())
    }

    /// Queues a line, as if the launcher had written it.
    pub fn push_line(&self, line: &str) {
        self.sender().send(line);
    }

    /// Reads lines from `input` on a thread until it closes (the launcher closed our stdin, or exited). Universal
    /// newlines, as Python's text-mode stdin reads them; bytes that aren't UTF-8 are replaced.
    pub fn read_from(&self, input: impl Read + Send + 'static) -> std::thread::JoinHandle<()> {
        let tx = self.sender();
        std::thread::Builder::new()
            .name("remote".into())
            .spawn(move || {
                let mut r = BufReader::new(input);
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    match r.read_until(b'\n', &mut buf) {
                        Ok(0) | Err(_) => return,
                        Ok(_) => tx.send(&String::from_utf8_lossy(&buf)),
                    }
                }
            })
            .expect("remote reader thread")
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Applies the commands that arrived since the last call (on the window's thread). Returns Quit or Restart if one
    /// asked for it (Quit first).
    pub fn poll(&mut self, settings: &mut Settings) -> Option<Request> {
        let mut out = None;
        while let Ok(line) = self.rx.try_recv() {
            let (cmd, rest) = line.split_once(' ').unwrap_or((&line, ""));
            match cmd {
                "quit" => out = Some(Request::Quit),
                "restart" => out = out.or(Some(Request::Restart)),
                "state" => self.publish(settings),
                "set" => {
                    let (key, text) = rest.split_once(' ').unwrap_or((rest, ""));
                    self.set(settings, key, text);
                }
                _ => {
                    let res = self.handlers.iter_mut().find(|(n, _)| n == cmd).map(|(_, h)| h(settings, rest));
                    match res {
                        Some(Ok(())) => {}
                        Some(Err(e)) => {
                            let e: String = e.chars().take(200).collect();
                            self.lock().say(&format!("[remote] {cmd} failed: {e}"));
                        }
                        None => {
                            let l: String = line.chars().take(80).collect();
                            self.lock().say(&format!("[remote] unknown command: {l}"));
                        }
                    }
                }
            }
        }
        out
    }

    fn set(&mut self, settings: &mut Settings, key: &str, text: &str) {
        let sp = spec(key);
        let value = json::loads(text).unwrap_or(Value::Null);
        let looks = settings.look_choices().clone();
        let why = {
            let sh = self.lock();
            match sp {
                None => Some("unknown setting".to_string()),
                Some(s) if !s.is_choice(&value, &looks) => {
                    let t: String = text.chars().take(40).collect();
                    Some(format!("{} isn't one of its choices", crate::py::str_repr(&t)))
                }
                Some(_) => sh.rules.disabled(settings, key).filter(|w| !w.is_empty()).or_else(|| {
                    (!sh.rules.allowed(settings, key, &value)).then(|| "not available right now".to_string())
                }),
            }
        };
        if let Some(why) = why {
            let mut sh = self.lock();
            sh.say(&format!("[remote] can't set {key}: {why}"));
            sh.publish(settings); // so the launcher shows the real value again
            return;
        }
        let s = sp.expect("known");
        let value = s.canonical(&value, &looks).expect("a choice"); // the spec's own type (JSON 1 -> 1.0)
        if settings.get(key).is_some_and(|cur| value.py_eq(cur)) {
            self.publish(settings); // nothing changes, so no watcher would answer
        } else {
            let _ = settings.set(key, value); // saved; the watchers apply it and publish
        }
    }

    /// The state the launcher reads: {"values", "choices", "disabled", "restart"}.
    pub fn state(&self, settings: &Settings) -> Value {
        self.lock().state(settings)
    }

    /// The line publish() writes (without its newline).
    pub fn state_line(&self, settings: &Settings) -> String {
        format!("{PREFIX}{}", json::dumps(&self.state(settings)))
    }

    pub fn publish(&self, settings: &Settings) {
        self.lock().publish(settings);
    }

    /// Stops watching the settings.
    pub fn close(mut self, settings: &mut Settings) {
        if let Some(w) = self.watch.take() {
            settings.unwatch(w);
        }
    }
}
