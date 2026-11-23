//! The clock, in seconds or nanoseconds on `CLOCK_UPTIME_RAW`. On macOS chiaki-ng's frame tap stamps frames with it
//! (chiaki_tap/frametap.c, `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`) and Python's `time.monotonic()` and
//! `time.perf_counter()` read it, so times from either side compare directly. It doesn't count time asleep.
//! `CLOCK_MONOTONIC` does, and is days ahead after sleeps.

unsafe extern "C" {
    fn clock_gettime_nsec_np(clock: u32) -> u64;
}

const CLOCK_UPTIME_RAW: u32 = 8; // <time.h>

/// Nanoseconds since boot, not counting sleep.
pub fn now_ns() -> u64 {
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

/// Seconds since boot, not counting sleep (Python's `time.perf_counter()` on macOS).
pub fn now() -> f64 {
    now_ns() as f64 * 1e-9
}

/// The frame tap's tick rate on macOS (its header's `qpc_freq`). A tick is a nanosecond.
pub const TAP_FREQ: u64 = 1_000_000_000;

#[cfg(test)]
mod tests {
    #[test]
    fn it_moves_forward() {
        let a = super::now_ns();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = super::now_ns();
        assert!(b - a >= 2_000_000 && b - a < 200_000_000);
    }
}
