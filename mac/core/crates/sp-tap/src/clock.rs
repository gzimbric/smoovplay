//! `CLOCK_UPTIME_RAW`, the clock frametap.c stamps frames with on macOS. Python's `time.perf_counter()` and
//! `time.monotonic_ns()` read it there too. This repeats sp_core::clock so the tap needs nothing from the GPU side.

unsafe extern "C" {
    fn clock_gettime_nsec_np(clock: u32) -> u64;
}

const CLOCK_UPTIME_RAW: u32 = 8; // <time.h>

/// Now, in the tap's ticks, which are nanoseconds (Python's `qpc()` on a Mac).
pub fn qpc() -> u64 {
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

/// The tap's tick rate here (the header's `qpc_freq`).
pub fn qpc_freq() -> u64 {
    1_000_000_000
}

/// Now, in seconds on the same clock (Python's `time.perf_counter()` on a Mac).
pub fn now() -> f64 {
    qpc() as f64 * 1e-9
}

/// A tap time (`q` ticks at `freq` a second) on [`now`]'s timebase. Like the Python reader, it takes now less the
/// time since `q`. With the writer on this clock (freq 1e9) that's `q` in seconds.
pub fn to_seconds(q: u64, freq: u64) -> f64 {
    let (nq, nt) = (qpc(), now());
    if freq == 0 {
        return nt;
    }
    nt - (nq as i128 - q as i128) as f64 / freq as f64
}
