//! Puts a variable-rate source on a steady beat, 60 Hz by default (framegen/sources.py Clock60): on each beat the
//! newest frame is handed on, or the previous one again if nothing came since. The source's frames are taken from
//! its queue on a thread of their own, so a late frame never holds up the beat.
//!
//! Each beat's tick: a fresh frame keeps the source's repeat hint; a beat with no fresh frame is a repeat for certain
//! (new_hint false). t_arrival stays when the frame really arrived; t_ready is when the beat handed it on. Frames
//! overtaken by a newer one before their beat are counted in `overtaken`. Until the first frame, it yields Waitings.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use sp_core::clock::now;
use sp_core::tick::WAIT_POLL_S;
use sp_core::{Event, Frame, Source, SourceInfo, Sound, Tick, Waiting};

use crate::queue::{Got, TickQueue};

/// A source whose reader thread fills a TickQueue: Clock60 takes frames from the queue directly.
pub trait QueuedSource: Source {
    fn queue(&self) -> Arc<TickQueue>;
    /// The pipeline is ready: start reading.
    fn start(&mut self);
    /// The source's hint for a Waiting.
    fn hint(&self) -> String;
}

struct Latest {
    tick: Option<Tick>,
    /// Frames received so far, and the serial of the one the beat handed on last.
    serial: u64,
    taken: u64,
    done: bool,
}

pub struct Clock60<S: QueuedSource> {
    pub source: S,
    period: f64,
    shared: Arc<(Mutex<Latest>, Condvar)>,
    stop: Arc<AtomicBool>,
    overtaken: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
    taken: u64,
    beat: Option<f64>,
    index: u64,
    t_wait: f64,
}

impl<S: QueuedSource> Clock60<S> {
    pub fn new(source: S, hz: f64) -> Clock60<S> {
        Clock60 {
            source,
            period: 1.0 / hz,
            shared: Arc::new((Mutex::new(Latest { tick: None, serial: 0, taken: 0, done: false }), Condvar::new())),
            stop: Arc::new(AtomicBool::new(false)),
            overtaken: Arc::new(AtomicU64::new(0)),
            thread: None,
            taken: 0,
            beat: None,
            index: 0,
            t_wait: now(),
        }
    }

    pub fn overtaken(&self) -> u64 {
        self.overtaken.load(Ordering::Relaxed)
    }

    fn start(&mut self) {
        self.source.start();
        let (q, sh, stop, over) = (self.source.queue(), self.shared.clone(), self.stop.clone(), self.overtaken.clone());
        self.thread = Some(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match q.get(WAIT_POLL_S) {
                    Got::Tick(t) => {
                        let mut l = sh.0.lock().unwrap();
                        if l.tick.is_some() && l.serial > l.taken {
                            over.fetch_add(1, Ordering::Relaxed);
                        }
                        l.tick = Some(t);
                        l.serial += 1;
                        sh.1.notify_all();
                    }
                    Got::Timeout => {}
                    Got::Ended => break,
                }
            }
            let mut l = sh.0.lock().unwrap();
            l.done = true;
            sh.1.notify_all();
        }));
    }
}

impl<S: QueuedSource> Source for Clock60<S> {
    fn info(&self) -> &SourceInfo {
        self.source.info()
    }
    fn size(&self) -> (usize, usize) {
        self.source.size()
    }
    fn next_event(&mut self) -> Option<Event> {
        if self.thread.is_none() {
            self.start();
            self.t_wait = now();
        }
        if self.beat.is_none() {
            // the first beat is the first frame
            let (lock, cv) = &*self.shared;
            let l = lock.lock().unwrap();
            let (l, _) = cv.wait_timeout_while(l, Duration::from_secs_f64(WAIT_POLL_S), |l| l.tick.is_none() && !l.done).unwrap();
            if l.tick.is_none() {
                if l.done {
                    return None;
                }
                drop(l);
                return Some(Event::Waiting(Waiting { since: now() - self.t_wait, what: "the app".into(), why: self.source.hint() }));
            }
            self.beat = Some(now());
        } else {
            let beat = self.beat.unwrap() + self.period;
            let t = now();
            let beat = if beat < t - self.period { t } else { beat }; // fell a whole beat behind: restart from now
            if beat > t {
                std::thread::sleep(Duration::from_secs_f64(beat - t));
            }
            self.beat = Some(beat);
        }
        let (latest, serial, done) = {
            let mut l = self.shared.0.lock().unwrap();
            l.taken = l.serial;
            (l.tick.clone(), l.serial, l.done)
        };
        let fresh = serial != self.taken;
        self.taken = serial;
        let latest = latest?;
        if done && !fresh {
            return None;
        }
        let handed = now();
        let mut t = Tick::new(latest.frame.clone(), if fresh { latest.t_arrival } else { handed }, self.index);
        t.content = latest.content;
        t.truth_new = if fresh { latest.truth_new } else { None };
        t.new_hint = if fresh { latest.new_hint } else { Some(false) };
        t.damaged = latest.damaged && fresh;
        t.t_ready = Some(handed);
        t.t_source = latest.t_source;
        if fresh {
            t.extra = latest.extra.clone();
        }
        self.index += 1;
        Some(Event::Tick(t))
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.source.ground_truth(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        self.source.read_sound()
    }
    fn request_keyframe(&mut self) {
        self.source.request_keyframe()
    }
    fn set_audio_delay(&mut self, ms: f64) {
        self.source.set_audio_delay(ms)
    }
    fn close(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.source.close();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl<S: QueuedSource> Drop for Clock60<S> {
    fn drop(&mut self) {
        self.close();
    }
}
