//! What instant replay costs at 1080p: the test clip's frames as the pipeline has them (GPU frames from sp-media's
//! decoder), shown in real time at --hz (90) for --seconds (65) with a --length (60) s replay, the game's frames at
//! 30 fps and 48 kHz stereo sound alongside. Prints the pipeline thread's time per frame, the GPU's, the encoder's
//! latency, frames dropped, and the memory held per minute (against the settings' hint).
//!
//!     cargo run --release -p sp-replay --example replay_bench -- [--hz 90] [--seconds 65] [--length 60] [--clip FILE]

use std::path::Path;
use std::time::Instant;

use sp_core::{Sound, clock};
use sp_media::VideoFile;
use sp_replay::{Options, Recorder};

fn arg(name: &str, default: f64) -> f64 {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn footprint_mb() -> f64 {
    // phys_footprint, as Activity Monitor's Memory column
    #[repr(C)]
    struct VmInfo {
        _pad: [u64; 18],
        phys_footprint: u64,
        _rest: [u64; 40],
    }
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: u32, info: *mut VmInfo, count: *mut u32) -> i32;
    }
    let mut info = VmInfo { _pad: [0; 18], phys_footprint: 0, _rest: [0; 40] };
    let mut count = (std::mem::size_of::<VmInfo>() / 4) as u32;
    let st = unsafe { task_info(mach_task_self(), 22 /* TASK_VM_INFO */, &mut info, &mut count) };
    if st == 0 { info.phys_footprint as f64 / 1e6 } else { f64::NAN }
}

fn main() {
    let hz = arg("--hz", 90.0) as usize;
    let seconds = arg("--seconds", 65.0);
    let length = arg("--length", 60.0) as u32;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf();
    let a: Vec<String> = std::env::args().collect();
    let clip = a.iter().position(|x| x == "--clip").and_then(|i| a.get(i + 1)).map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo.join("testdata/clip_1080p60.mp4"));
    let out = std::env::temp_dir().join(format!("sp-replay-bench-{}", std::process::id()));
    let rec = Recorder::new(Options { length_s: length, out_dir: Some(out.clone()), size: Some((1920, 1080)), ..Default::default() });
    rec.set_output_hz(hz as u32);
    rec.set_delay(50.0);
    rec.warm_up();
    let mem0 = footprint_mb();
    let mut file = VideoFile::open(&clip).expect("the test clip");
    let mut cur = file.next_frame().unwrap();
    let mut c_have = 0usize;
    let mut seed = 12345u64;
    let mut noise = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 6000 - 3000) as i16
    };
    let t0 = clock::now() + 0.2;
    let mut k = 0usize;
    let mut late = 0usize;
    let started = Instant::now();
    let n = (seconds * hz as f64) as usize;
    for i in 0..n {
        let due = t0 + i as f64 / hz as f64;
        while clock::now() < due - 0.002 {
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
        while clock::now() < due {}
        if clock::now() > due + 0.004 {
            late += 1;
        }
        // Sound up to now, in 10 ms chunks
        while t0 + k as f64 * 0.01 <= due {
            let pcm: Vec<i16> = (0..960).map(|_| noise()).collect();
            rec.add_sound(vec![Sound { t: t0 + k as f64 * 0.01 - 0.05, pcm, rate: 48_000, channels: 2 }]);
            k += 1;
        }
        let c = i * 60 / hz;
        let mut new = false;
        while c_have < c {
            cur = match file.next_frame() {
                Some(f) => f,
                None => {
                    file = VideoFile::open(&clip).unwrap();
                    file.next_frame().unwrap()
                }
            };
            c_have += 1;
            new = true;
        }
        let t = clock::now();
        rec.frame(&cur, t);
        if (new || i == 0) && c.is_multiple_of(2) {
            rec.real_frame(&cur, t - 0.05);
        }
    }
    let wall = started.elapsed().as_secs_f64();
    rec.flush(5.0);
    let held = rec.held();
    let st = rec.stats();
    let mem1 = footprint_mb();
    let per_min = |(s, mb): (f64, f64)| if s > 0.0 { mb / s * 60.0 } else { 0.0 };
    let total_min = per_min(held.after) + per_min(held.before) + per_min(held.sound);
    println!("{hz} Hz output, {n} frames in {wall:.1} s ({late} shown late by the bench itself), length {length} s");
    println!("pipeline thread per frame: median {:.0} us, p99 {:.0} us, max {:.0} us ({} frames)", st.put_us.p50, st.put_us.p99, st.put_us.max, st.put_us.n);
    println!("GPU conversion per frame: median {:.0} us, p99 {:.0} us", st.gpu_us.p50, st.gpu_us.p99);
    println!("encoder latency: median {:.1} ms, p99 {:.1} ms, max {:.1} ms", st.encode_ms.p50, st.encode_ms.p99, st.encode_ms.max);
    println!("dropped frames: {}", rec.dropped());
    println!(
        "held: after {:.1} s {:.1} MB ({:.1} Mbit/s), before {:.1} s {:.1} MB ({:.1} Mbit/s), sound {:.1} s {:.2} MB",
        held.after.0,
        held.after.1,
        held.after.1 * 8.0 / held.after.0.max(1e-9),
        held.before.0,
        held.before.1,
        held.before.1 * 8.0 / held.before.0.max(1e-9),
        held.sound.0,
        held.sound.1
    );
    println!(
        "memory per minute: {total_min:.0} MB (the settings' hint for 1 min at 1080p90: {:.0} MB); process footprint {mem0:.0} -> {mem1:.0} MB",
        sp_replay::memory_mb_1080p90(60.0)
    );
    let t = Instant::now();
    let saved = rec.save(None, None, true);
    println!("saved {} in {:.2} s", saved.first().map(|p| p.display().to_string()).unwrap_or_default(), t.elapsed().as_secs_f64());
    rec.close(0.0);
    let _ = std::fs::remove_dir_all(&out);
}
