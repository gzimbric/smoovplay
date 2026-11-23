//! A live source's ticks between its reader thread and the pipeline (framegen/io.py queue_ticks). At most a few are
//! queued, and the oldest is dropped when the pipeline falls behind. They're numbered from 0 as they're taken, with a
//! Waiting whenever none has come for WAIT_POLL_S.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use sp_core::tick::WAIT_POLL_S;
use sp_core::{Event, Tick, Waiting};

pub struct TickQueue {
    state: Mutex<State>,
    cv: Condvar,
    cap: usize,
}

struct State {
    items: VecDeque<Tick>,
    ended: bool,
}

pub enum Got {
    Tick(Tick),
    Timeout,
    Ended,
}

impl TickQueue {
    pub fn new(cap: usize) -> TickQueue {
        TickQueue { state: Mutex::new(State { items: VecDeque::new(), ended: false }), cv: Condvar::new(), cap }
    }

    /// Queue a tick. If the queue is full the oldest goes. Returns true if one was dropped.
    pub fn put(&self, t: Tick) -> bool {
        let mut s = self.state.lock().unwrap();
        let dropped = s.items.len() >= self.cap;
        if dropped {
            s.items.pop_front();
        }
        s.items.push_back(t);
        self.cv.notify_all();
        dropped
    }

    /// No more ticks (after the ones queued).
    pub fn end(&self) {
        self.state.lock().unwrap().ended = true;
        self.cv.notify_all();
    }

    pub fn clear(&self) {
        self.state.lock().unwrap().items.clear();
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, timeout_s: f64) -> Got {
        let mut s = self.state.lock().unwrap();
        let t_end = std::time::Instant::now() + Duration::from_secs_f64(timeout_s);
        loop {
            if let Some(t) = s.items.pop_front() {
                return Got::Tick(t);
            }
            if s.ended {
                return Got::Ended;
            }
            let now = std::time::Instant::now();
            if now >= t_end {
                return Got::Timeout;
            }
            s = self.cv.wait_timeout(s, t_end - now).unwrap().0;
        }
    }
}

/// queue_ticks' numbering and waiting.
pub struct Ticker {
    index: u64,
    last: f64,
    pub what: String,
}

impl Ticker {
    pub fn new(what: &str) -> Ticker {
        Ticker { index: 0, last: sp_core::clock::now(), what: what.to_string() }
    }

    /// The next tick (numbered), a Waiting after WAIT_POLL_S without one, or None at the end. `why` gives the source's
    /// hint for the Waiting.
    pub fn next(&mut self, q: &TickQueue, why: impl FnOnce() -> String) -> Option<Event> {
        match q.get(WAIT_POLL_S) {
            Got::Tick(mut t) => {
                t.index = self.index;
                self.index += 1;
                self.last = sp_core::clock::now();
                Some(Event::Tick(t))
            }
            Got::Timeout => {
                Some(Event::Waiting(Waiting { since: sp_core::clock::now() - self.last, what: self.what.clone(), why: why() }))
            }
            Got::Ended => None,
        }
    }
}
