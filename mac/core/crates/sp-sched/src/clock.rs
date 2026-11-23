//! The pipeline's clock. It is injectable so the parity traces and the tests can replay a run deterministically.
//! [`SystemClock`] is sp_core::clock (CLOCK_UPTIME_RAW, the frame tap's clock). [`FakeClock`] only moves when told.

use std::sync::atomic::{AtomicU64, Ordering};

pub trait Clock: Send + Sync {
    /// Seconds (Python's time.perf_counter()).
    fn now(&self) -> f64;
    /// time.sleep(secs).
    fn sleep(&self, secs: f64);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        sp_core::clock::now()
    }
    fn sleep(&self, secs: f64) {
        if secs > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
    }
}

/// A clock that moves only when told. sleep() and advance() add to it, and at_least() moves it forward to a given
/// time. The arithmetic is exactly the Python trace clock's (tools/sched_traces.py FakeClock), so both sides
/// read the same times.
#[derive(Debug)]
pub struct FakeClock {
    bits: AtomicU64,
}

impl FakeClock {
    pub fn new(t: f64) -> FakeClock {
        FakeClock { bits: AtomicU64::new(t.to_bits()) }
    }
    pub fn get(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::SeqCst))
    }
    pub fn set(&self, t: f64) {
        self.bits.store(t.to_bits(), Ordering::SeqCst);
    }
    pub fn advance(&self, d: f64) {
        self.set(self.get() + d);
    }
    pub fn at_least(&self, t: f64) {
        if t > self.get() {
            self.set(t);
        }
    }
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        self.get()
    }
    fn sleep(&self, secs: f64) {
        if secs > 0.0 {
            self.advance(secs);
        }
    }
}
