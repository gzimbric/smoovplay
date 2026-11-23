//! Runs a real clip through the recorder at 60 and 90 Hz with sound. The plan is in tools/replay_fixtures.py, which
//! also has the Python recorder save the same. The saved MP4 is checked with ffprobe for durations, frame rate,
//! keyframes, the sound's length and its sync with the picture to within a frame. Then it's compared with the
//! Python's clip, which should have the same cut and frame count, by the PSNR of matching frames and of each against
//! the source. Results go to fixtures/replay/results_<hz>.json.
//!
//! Needs the fixtures (`.venv/bin/python mac/core/tools/replay_fixtures.py`) and is skipped without them. It's faster
//! with --release because of the PSNR sums.

mod common;

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

use common::*;
use sp_core::{Frame, Sound};
use sp_replay::Kind;

const W: usize = 1920;
const H: usize = 1080;

/// Frames from an FFmpeg pipe, one at a time.
struct Pipe {
    child: Child,
    out: ChildStdout,
    size: usize,
}

impl Pipe {
    fn video(path: &Path, pix_fmt: &str, bpp: usize, frames: Option<usize>) -> Pipe {
        let mut c = Command::new("ffmpeg");
        c.args(["-v", "error", "-i"]).arg(path).args(["-map", "0:v:0"]);
        if let Some(n) = frames {
            c.args(["-frames:v", &n.to_string()]);
        }
        let mut child = c.args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]).stdout(Stdio::piped()).spawn().expect("ffmpeg");
        let out = child.stdout.take().unwrap();
        Pipe { child, out, size: W * H * bpp }
    }
    fn next(&mut self) -> Option<Vec<u8>> {
        let mut b = vec![0u8; self.size];
        self.out.read_exact(&mut b).ok()?;
        Some(b)
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn stamp(img: &mut [u8], bpp: usize, i: usize) {
    for b in 0..16 {
        let v = if (i >> b) & 1 == 1 { 255 } else { 0 };
        for y in 0..16 {
            for x in 16 * b..16 * b + 16 {
                let o = (y * W + x) * bpp;
                img[o..o + 3].fill(v);
            }
        }
    }
}

fn read_stamp(bgr: &[u8]) -> usize {
    let mut i = 0;
    for b in 0..16 {
        let mut s = 0u32;
        for y in 4..12 {
            for x in 16 * b + 4..16 * b + 12 {
                s += bgr[(y * W + x) * 3 + 1] as u32;
            }
        }
        if s / 64 > 128 {
            i |= 1 << b;
        }
    }
    i
}

struct Plan {
    t0: f64,
    delay: f64,
    seconds: f64,
    length: u32,
    pcm: Vec<i16>,
}

fn plan() -> Option<Plan> {
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("manifest.json")).ok()?).ok()?;
    let raw = std::fs::read(fixtures().join("clip_audio.s16")).ok()?;
    Some(Plan {
        t0: m["t0"].as_f64()?,
        delay: m["delay"].as_f64()?,
        seconds: m["seconds"].as_f64()?,
        length: m["length"].as_u64()? as u32,
        pcm: raw.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect(),
    })
}

/// Runs the plan through the Rust recorder. Returns the saved clip, which is also copied to
/// fixtures/replay/rust_<hz>.mp4.
fn feed(p: &Plan, hz: usize, dir: &Path) -> (std::path::PathBuf, serde_json::Value) {
    let (rec, said) = recorder(dir, p.length);
    rec.set_delay(p.delay * 1000.0);
    rec.set_output_hz(hz as u32);
    let clip = repo().join("testdata/clip_1080p60.mp4");
    let mut src = Pipe::video(&clip, "bgra", 4, Some((p.seconds * 60.0).round() as usize));
    let mut cur: Option<(usize, Vec<u8>)> = None;
    let mut next_c = 0usize;
    let mut k = 0usize;
    let mut seen = HashSet::new();
    let mut frame_us = Vec::new();
    for i in 0..(p.seconds * hz as f64).round() as usize {
        let t = p.t0 + p.delay + i as f64 / hz as f64;
        while (k + 1) * 480 <= p.pcm.len() && p.t0 + k as f64 / 100.0 + p.delay <= t {
            rec.add_sound(vec![Sound { t: p.t0 + k as f64 / 100.0, pcm: p.pcm[k * 480..(k + 1) * 480].to_vec(), rate: 48_000, channels: 1 }]);
            k += 1;
        }
        let c = i * 60 / hz;
        while cur.as_ref().is_none_or(|x| x.0 < c) {
            match src.next() {
                Some(f) => cur = Some((next_c, f)),
                None => break,
            }
            next_c += 1;
        }
        let Some((cc, base)) = cur.as_ref() else { break };
        if *cc != c {
            break;
        }
        let mut img = base.clone();
        stamp(&mut img, 4, i);
        let f = Frame::from_bgra(&img, W, H);
        let t1 = std::time::Instant::now();
        rec.frame(&f, t);
        frame_us.push(t1.elapsed().as_secs_f64() * 1e6);
        if c.is_multiple_of(2) && seen.insert(c) {
            rec.real_frame(&f, p.t0 + c as f64 / 60.0);
        }
    }
    assert!(rec.flush(60.0));
    let held = rec.held();
    let t1 = std::time::Instant::now();
    let paths = rec.save(Some(Kind::Smoovplay), None, true);
    let save_s = t1.elapsed().as_secs_f64();
    assert_eq!(paths.len(), 1, "{}", said.log());
    let t1 = std::time::Instant::now();
    let compare = rec.save(Some(Kind::Compare), None, true);
    let compare_s = t1.elapsed().as_secs_f64();
    assert_eq!(compare.len(), 1, "{}", said.log());
    let dropped = rec.dropped();
    rec.close(0.0);
    let keep = fixtures().join(format!("rust_{hz}.mp4"));
    std::fs::copy(&paths[0], &keep).unwrap();
    std::fs::copy(&compare[0], fixtures().join(format!("rust_{hz}_compare.mp4"))).unwrap();
    frame_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let info = serde_json::json!({
        "save_s": save_s,
        "compare_s": compare_s,
        "dropped": dropped,
        "held_mb": held.mb(),
        "held": format!("{held:?}"),
        "frame_call_us_median": frame_us[frame_us.len() / 2],
        "frame_call_us_p99": frame_us[frame_us.len() * 99 / 100],
    });
    (keep, info)
}

/// Where the clip's sound is against the original, as the original's time (s) at the clip's time 0.
fn audio_at_zero(path: &Path, original: &[i16], expect: f64) -> f64 {
    let (_, a) = probe(path);
    let a = a.expect("no sound");
    let got = decode_audio(path);
    let (w0, n) = (48_000usize, 12_000usize);
    let win = &got[w0..w0 + n];
    let orig: Vec<f32> = original.iter().map(|&s| s as f32 / 32768.0).collect();
    // The original index that decoded sample w0 is. Search 100 ms either side of where it should be.
    let centre = ((expect + a.start) * 48_000.0).round() as i64 + w0 as i64;
    let mut best = (f32::MIN, 0i64);
    for lag in centre - 4800..=centre + 4800 {
        if lag < 0 || lag as usize + n > orig.len() {
            continue;
        }
        let o = &orig[lag as usize..lag as usize + n];
        let (mut xy, mut yy) = (0f32, 0f32);
        for j in 0..n {
            xy += win[j] * o[j];
            yy += o[j] * o[j];
        }
        let c = xy / yy.sqrt().max(1e-9);
        if c > best.0 {
            best = (c, lag);
        }
    }
    (best.1 - w0 as i64) as f64 / 48_000.0 - a.start
}

#[test]
fn a_clip_at_60_and_90_hz_matches_the_python() {
    let Some(p) = plan() else {
        eprintln!("no fixtures: run .venv/bin/python mac/core/tools/replay_fixtures.py");
        return;
    };
    let tmp = TempDir::new("clip");
    let clip = repo().join("testdata/clip_1080p60.mp4");
    for hz in [60usize, 90] {
        let (rust, mut info) = feed(&p, hz, tmp.path());
        let python = fixtures().join(format!("python_{hz}.mp4"));
        let (v, a) = probe(&rust);
        let (v, a) = (v.unwrap(), a.expect("no sound"));
        let (pv, pa) = probe(&python);
        let (pv, pa) = (pv.unwrap(), pa.unwrap());
        let frame = 1.0 / hz as f64;
        // What ffprobe says
        assert!(v.width == W as u64 && v.height == H as u64 && v.profile == "High", "{v:?}");
        assert!((p.length as f64..=p.length as f64 + 1.0 + frame).contains(&v.duration), "{v:?}");
        assert!((v.fps - hz as f64).abs() < 0.01 && (v.frames as f64 - v.duration * hz as f64).abs() <= 1.0, "{v:?}");
        let f = frames(&rust);
        assert!(f[0].1 && f.len() as u64 == v.frames, "first frame not a keyframe, or {} frames", f.len());
        let keys: Vec<f64> = f.iter().filter(|x| x.1).map(|x| x.0).collect();
        assert!(keys.windows(2).all(|k| k[1] - k[0] <= 1.0 + frame + 1e-6), "keyframes {keys:?}");
        // The sound from the first frame to within the AAC encoder's hold (2112 samples) and a chunk of the end
        assert!(a.start < 1024.0 / 48_000.0 + 1e-3 && (a.start + a.duration - v.duration).abs() < 0.1, "sound {a:?}, video {v:?}");
        // For sync, the first frame's stamp is its shown index, and the sound at the clip's 0 should be the same moment
        let mut rp = Pipe::video(&rust, "bgr24", 3, None);
        let mut pp = Pipe::video(&python, "bgr24", 3, None);
        let mut sp = Pipe::video(&clip, "bgr24", 3, None);
        let (mut src, mut src_c) = (sp.next().unwrap(), 0usize);
        let (mut n, mut p_rs, mut p_ps, mut p_rp) = (0usize, Vec::new(), Vec::new(), Vec::new());
        let mut i0 = None;
        let mut j = 0usize;
        let mut last = 0;
        while let Some(r) = rp.next() {
            let py = pp.next();
            let sr = read_stamp(&r);
            if let Some(py) = &py {
                // The same cut, with the same shown frames frame for frame, for as long as the Python's goes
                let spy = read_stamp(py);
                assert_eq!(sr, spy, "frame {j}: rust shows {sr}, python {spy}");
            }
            i0.get_or_insert(sr);
            last = sr;
            let c = sr * 60 / hz;
            while src_c < c {
                src = sp.next().unwrap();
                src_c += 1;
            }
            if j.is_multiple_of(3)
                && let Some(py) = &py {
                    let mut s = src.clone();
                    stamp(&mut s, 3, sr);
                    p_rs.push(psnr(&r, &s));
                    p_ps.push(psnr(py, &s));
                    p_rp.push(psnr(&r, py));
                    n += 1;
                }
            j += 1;
        }
        assert_eq!(j as u64, v.frames);
        let i0 = i0.unwrap();
        // Every frame shown up to the save is there, because the encoder is emptied first. FFmpeg's VideoToolbox
        // encoder still holds the last few when the Python saves, so its clip ends that much earlier.
        assert_eq!(last + 1, (p.seconds * hz as f64).round() as usize, "the clip ends at shown frame {last}");
        assert!(v.frames >= pv.frames && v.frames - pv.frames <= 12, "rust {} frames, python {}", v.frames, pv.frames);
        let shown = i0 as f64 / hz as f64;
        let rust_audio = audio_at_zero(&rust, &p.pcm, shown);
        let python_audio = audio_at_zero(&python, &p.pcm, shown);
        let (sync, py_sync) = (rust_audio - shown, python_audio - shown);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        // The before & after has every frame of the clip, with the game's frame held on the left. It's compared with
        // the Python's.
        let rc = fixtures().join(format!("rust_{hz}_compare.mp4"));
        let pc = fixtures().join(format!("python_{hz}_compare.mp4"));
        let (cv, ca) = probe(&rc);
        let (cv, ca) = (cv.unwrap(), ca.expect("no sound in the before & after"));
        assert!(cv.frames == v.frames && (cv.duration - v.duration).abs() < 1e-3 && cv.width == W as u64, "{cv:?} vs {v:?}");
        assert!((ca.start + ca.duration - cv.duration).abs() < 0.1, "{ca:?}");
        let (pcv, _) = probe(&pc);
        let pcv = pcv.unwrap();
        let (mut rcp, mut pcp) = (Pipe::video(&rc, "bgr24", 3, None), Pipe::video(&pc, "bgr24", 3, None));
        let (mut rvp, mut pvp) = (Pipe::video(&rust, "bgr24", 3, None), Pipe::video(&python, "bgr24", 3, None));
        // SmoovPlay's half against the clip's own frame (under the labels, right of the line)
        let right = |a: &[u8], b: &[u8]| {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for row in 100..H {
                x.extend_from_slice(&a[(row * W + W / 2 + 8) * 3..(row * W + W) * 3]);
                y.extend_from_slice(&b[(row * W + W / 2 + 8) * 3..(row * W + W) * 3]);
            }
            psnr(&x, &y)
        };
        let (mut cj, mut c_rust, mut c_python) = (0usize, Vec::new(), Vec::new());
        while let (Some(r), Some(rv)) = (rcp.next(), rvp.next()) {
            let py = pcp.next();
            let pv = pvp.next();
            if let Some(py) = &py {
                // The left half's stamp is the game frame's, the same one the Python holds. The last few can differ,
                // because the Python's before track is missing what FFmpeg's encoder still held.
                if (cj as u64) + 12 < pcv.frames {
                    assert_eq!(read_stamp(&r), read_stamp(py), "before & after frame {cj}: different game frames");
                }
            }
            if cj % 6 == 0 {
                c_rust.push(right(&r, &rv));
                if let (Some(py), Some(pv)) = (&py, &pv) {
                    c_python.push(right(py, pv));
                }
            }
            cj += 1;
        }
        assert_eq!(cj as u64, cv.frames);
        let res = serde_json::json!({
            "hz": hz,
            "rust": {"frames": v.frames, "duration": v.duration, "bit_rate_mbps": file_mbps(&rust, v.duration),
                     "audio_start": a.start, "audio_duration": a.duration, "sync_ms": sync * 1000.0,
                     "psnr_vs_source": {"mean": mean(&p_rs), "min": min(&p_rs)}, "keyframes": keys},
            "python": {"frames": pv.frames, "duration": pv.duration, "bit_rate_mbps": file_mbps(&python, pv.duration),
                       "audio_start": pa.start, "audio_duration": pa.duration, "sync_ms": py_sync * 1000.0,
                       "psnr_vs_source": {"mean": mean(&p_ps), "min": min(&p_ps)}},
            "psnr_rust_vs_python": {"mean": mean(&p_rp), "min": min(&p_rp)},
            "compare": {"rust_frames": cv.frames, "python_frames": pcv.frames, "duration": cv.duration,
                        "rust_mbps": file_mbps(&rc, cv.duration), "python_mbps": file_mbps(&pc, pcv.duration),
                        "smoovplay_half_vs_its_clip_psnr": {"rust": mean(&c_rust), "python": mean(&c_python)},
                        "python_compare_s": python_info(hz)["compare_s"].clone()},
            "frames_compared": n,
            "first_shown_index": i0,
            "last_shown_index": last,
            "feed": info.take(),
        });
        std::fs::write(fixtures().join(format!("results_{hz}.json")), serde_json::to_string_pretty(&res).unwrap()).unwrap();
        assert!(mean(&c_rust) > 40.0, "the before & after's SmoovPlay half isn't its clip: {res}");
        println!("{hz} Hz: {}", serde_json::to_string_pretty(&res).unwrap());
        assert!(sync.abs() < frame, "{hz} Hz: the sound is {:.1} ms off the picture", sync * 1000.0);
        assert!(mean(&p_rs) > 35.0 && mean(&p_rs) > mean(&p_ps) - 1.5, "{res}");
    }
}

fn python_info(hz: usize) -> serde_json::Value {
    std::fs::read(fixtures().join(format!("python_{hz}.json"))).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn file_mbps(path: &Path, secs: f64) -> f64 {
    std::fs::metadata(path).unwrap().len() as f64 * 8.0 / secs / 1e6
}
