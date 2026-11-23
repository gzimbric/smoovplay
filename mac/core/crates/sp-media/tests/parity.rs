//! The Rust ChiakiSource against the Python one (framegen/chiaki_tap.py, VideoToolbox through PyAV), both reading the
//! same fake PS5 stream (scripts/frametap_fake.py) at the same time; the Python side's ticks are dumped by
//! mac/core/tools/media_ticks.py. Compared frame by frame (by the tap's sequence number): which frames became ticks,
//! and what each says (size, the size check's call, keyframe, loss, damaged, test mode's content and truth, hints).
//!
//! Slow (a few seconds of real-time stream per case) and needs the repo's .venv, so ignored by default:
//!
//!     cargo test --release -p sp-media --test parity -- --ignored --nocapture

mod common;

use std::collections::BTreeMap;
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use sp_core::{Event, Source};
use sp_media::{ChiakiOptions, ChiakiSource, HdrOut, HdrSettings, TestMode};

const FIELDS: [&str; 10] =
    ["size", "size_new", "keyframe", "frames_lost", "recovered", "damaged", "content", "truth_new", "new_hint", "game_new"];

fn rust_ticks(name: &str, test: bool, hdr: bool) -> (Vec<Value>, (u64, u64, u64)) {
    let opts = ChiakiOptions {
        tap_name: name.into(),
        wait_s: 30.0,
        test: test.then(|| TestMode::shared(true)),
        ..Default::default()
    };
    let mut src = ChiakiSource::open(opts).expect("open");
    if hdr {
        src.set_hdr(Some(HdrSettings::new(HdrOut::Sdr8, None)), "");
    }
    let mut out = Vec::new();
    while let Some(ev) = src.next_event() {
        let Event::Tick(t) = ev else { continue };
        let e = &t.extra;
        let g = |k: &str| e.get(k).cloned().unwrap_or(Value::Null);
        out.push(json!({
            "index": t.index, "seq": g("seq"), "size": g("size"), "size_new": g("size_new"), "keyframe": g("keyframe"),
            "frames_lost": g("frames_lost"), "recovered": g("recovered"), "damaged": t.damaged, "content": t.content,
            "truth_new": t.truth_new, "new_hint": t.new_hint, "game_new": g("game_new"), "t_arrival": t.t_arrival,
            "decode_ms": g("decode_ms"),
        }));
    }
    let counts = src.counts();
    src.close();
    (out, counts)
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { f64::NAN } else { v[((v.len() - 1) as f64 * p).round() as usize] }
}

fn case(tag: &str, fake_args: &[&str], test: bool, hdr: bool) {
    let Some(py) = common::python() else { return eprintln!("skipped: no .venv") };
    let repo = common::repo();
    let name = format!("Local\\sp-rs-par{tag}{}", std::process::id() % 1000);
    sp_tap::unlink(&name);
    let dir = std::env::temp_dir();
    let out = dir.join(format!("sp-parity-{tag}-{}.jsonl", std::process::id()));
    let mut pyargs = vec![
        repo.join("mac/core/tools/media_ticks.py").display().to_string(),
        name.clone(),
        out.display().to_string(),
        "--wait".into(),
        "40".into(),
    ];
    if test {
        pyargs.push("--test-mode".into());
    }
    if hdr {
        pyargs.extend(["--hdr".into(), "sdr8".into()]);
    }
    let py_reader = Command::new(&py).args(&pyargs).current_dir(&repo).stdout(Stdio::null()).spawn().unwrap();
    let n2 = name.clone();
    let rust = std::thread::spawn(move || rust_ticks(&n2, test, hdr));
    // Both readers wait for the stream; it starts after they're up (--delay), so both see its codec header.
    let mut fargs = vec![
        repo.join("scripts/frametap_fake.py").display().to_string(),
        repo.join("testdata/clip_1080p60.mp4").display().to_string(),
        "--chiaki-like".into(),
        "--tap".into(),
        name.clone(),
        "--delay".into(),
        if hdr { "10" } else { "4" }.into(),
    ];
    fargs.extend(fake_args.iter().map(|s| s.to_string()));
    let fake = Command::new(&py).args(&fargs).current_dir(&repo).output().unwrap();
    let fake_out = String::from_utf8_lossy(&fake.stdout).to_string();
    assert!(fake.status.success(), "fake PS5: {fake_out}{}", String::from_utf8_lossy(&fake.stderr));
    let py_status = py_reader.wait_with_output().unwrap();
    assert!(py_status.status.success(), "python reader failed");
    let (rs, rs_counts) = rust.join().unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    let mut pyt: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let py_done = pyt.pop().unwrap();
    sp_tap::unlink(&name);
    let _ = std::fs::remove_file(sp_tap::saved::header_path(&name)); // (both readers saved the stream's header)
    let _ = std::fs::remove_file(&out);

    let by_seq = |v: &[Value]| -> BTreeMap<u64, Value> { v.iter().map(|t| (t["seq"].as_u64().unwrap(), t.clone())).collect() };
    let (p, r) = (by_seq(&pyt), by_seq(&rs));
    let only_py: Vec<_> = p.keys().filter(|k| !r.contains_key(k)).collect();
    let only_rs: Vec<_> = r.keys().filter(|k| !p.contains_key(k)).collect();
    let mut mism: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    let mut worst_t = 0f64;
    for (seq, a) in &p {
        let Some(b) = r.get(seq) else { continue };
        for f in FIELDS {
            // (a whole number may come back from JSON as an integer on one side and a float on the other)
            let same = a[f] == b[f] || (a[f].as_f64().is_some() && a[f].as_f64() == b[f].as_f64());
            if !same {
                mism.entry(f).or_default().push(*seq);
            }
        }
        worst_t = worst_t.max((a["t_arrival"].as_f64().unwrap() - b["t_arrival"].as_f64().unwrap()).abs());
        if a["index"] != b["index"] {
            mism.entry("index").or_default().push(*seq);
        }
    }
    let dec = |v: &[Value]| v.iter().filter_map(|t| t["decode_ms"].as_f64()).collect::<Vec<_>>();
    let (dp, dr) = (dec(&pyt), dec(&rs));
    let count = |v: &[Value], f: &str, val: Value| v.iter().filter(|t| t[f] == val).count();
    eprintln!(
        "parity {tag}: python {} ticks (dropped/skipped/idr {}/{}/{}), rust {} ticks ({}/{}/{}); only python {:?}, only rust \
         {:?}; damaged py {} rs {}; size_new true py {} rs {}, null py {} rs {}; t_arrival max diff {:.3} ms; decode ms \
         p50/p95 python {:.2}/{:.2}, rust {:.2}/{:.2}\n  fake: {}",
        pyt.len(),
        py_done["dropped"],
        py_done["skipped"],
        py_done["idr_requests"],
        rs.len(),
        rs_counts.0,
        rs_counts.1,
        rs_counts.2,
        only_py,
        only_rs,
        count(&pyt, "damaged", json!(true)),
        count(&rs, "damaged", json!(true)),
        count(&pyt, "size_new", json!(true)),
        count(&rs, "size_new", json!(true)),
        count(&pyt, "size_new", Value::Null),
        count(&rs, "size_new", Value::Null),
        worst_t * 1e3,
        pct(dp.clone(), 0.5),
        pct(dp, 0.95),
        pct(dr.clone(), 0.5),
        pct(dr, 0.95),
        fake_out.lines().last().unwrap_or(""),
    );
    for (f, seqs) in &mism {
        eprintln!("  {f} differs at {} frames, first {:?}", seqs.len(), &seqs[..seqs.len().min(10)]);
    }
    assert!(pyt.len() > 100, "too few ticks");
    assert!(only_py.is_empty() && only_rs.is_empty() && mism.is_empty(), "{tag}: the tick sequences differ");
    assert!(worst_t < 1e-3, "t_arrival differs by {worst_t}");
}

#[test]
#[ignore]
fn parity_30fps_game_hevc() {
    case("a", &["--codec", "hevc", "--seconds", "6"], false, false);
}

#[test]
#[ignore]
fn parity_test_mode_native_60() {
    case("b", &["--codec", "hevc", "--native", "--seconds", "6"], true, false);
}

#[test]
#[ignore]
fn parity_loss_and_jitter() {
    case("c", &["--codec", "hevc", "--seconds", "8", "--loss", "0.02", "--jitter", "2", "--seed", "3"], false, false);
}

#[test]
#[ignore]
fn parity_h264() {
    case("d", &["--codec", "h264", "--seconds", "6", "--loss", "0.01", "--seed", "5"], false, false);
}

#[test]
#[ignore]
fn parity_hdr10_sdr8() {
    case("e", &["--hdr", "--seconds", "6"], false, true);
}

/// The saved codec header (sp_tap::saved, framegen/chiaki_tap.py _save_header/_load_header): a reader that saw the
/// stream start saves it; one that joins later loads it and decodes, whichever language each is. And with nobody to
/// have saved it, a late reader decodes nothing and says why.
#[test]
#[ignore]
fn saved_header_shared_between_python_and_rust() {
    let Some(py) = common::python() else { return eprintln!("skipped: no .venv") };
    let repo = common::repo();
    for (k, first) in ["python", "rust", "none"].iter().enumerate() {
        let name = format!("Local\\sp-rs-hdr{k}{}", std::process::id() % 1000);
        sp_tap::unlink(&name);
        let _ = std::fs::remove_file(sp_tap::saved::header_path(&name));
        let out = std::env::temp_dir().join(format!("sp-hdr-{k}-{}.jsonl", std::process::id()));
        let fake = Command::new(&py)
            .args([
                repo.join("scripts/frametap_fake.py").display().to_string(),
                repo.join("testdata/clip_1080p60.mp4").display().to_string(),
                "--chiaki-like".into(),
                "--codec".into(),
                "hevc".into(),
                "--tap".into(),
                name.clone(),
                "--delay".into(),
                "4".into(),
                "--seconds".into(),
                "9".into(),
                "--no-sound".into(),
            ])
            .current_dir(&repo)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let py_args = |wait: &str| {
            vec![
                repo.join("mac/core/tools/media_ticks.py").display().to_string(),
                name.clone(),
                out.display().to_string(),
                "--wait".into(),
                wait.into(),
            ]
        };
        // The early reader sees the stream start (and saves its header); the late one comes 4 s in.
        let early_py = (*first == "python").then(|| Command::new(&py).args(py_args("30")).current_dir(&repo).stdout(Stdio::null()).spawn().unwrap());
        let n2 = name.clone();
        let early_rs = (*first == "rust").then(|| std::thread::spawn(move || rust_ticks(&n2, false, false)));
        std::thread::sleep(std::time::Duration::from_secs(8));
        let saved = std::fs::read(sp_tap::saved::header_path(&name)).unwrap_or_default();
        let (late_n, late_damaged, why) = if *first == "rust" {
            let o = Command::new(&py).args(py_args("5")).current_dir(&repo).output().unwrap();
            let said = String::from_utf8_lossy(&o.stdout).to_string();
            let text = std::fs::read_to_string(&out).unwrap_or_default();
            let ticks: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).filter(|v: &Value| v.get("seq").is_some()).collect();
            assert!(said.contains("using its saved codec header"), "python: {said}");
            (ticks.len(), ticks.iter().filter(|t| t["damaged"] == json!(true)).count(), String::new())
        } else {
            let mut src = ChiakiSource::open(ChiakiOptions { tap_name: name.clone(), wait_s: 5.0, ..Default::default() }).unwrap();
            let (mut n, mut dmg, mut why) = (0, 0, String::new());
            while let Some(ev) = src.next_event() {
                match ev {
                    Event::Tick(t) => {
                        n += 1;
                        dmg += t.damaged as usize;
                    }
                    Event::Waiting(w) => {
                        if !w.why.is_empty() {
                            why = w.why;
                        }
                    }
                }
                if *first == "none" && why.contains("started before SmoovPlay") {
                    break;
                }
            }
            if *first == "none" {
                assert!(src.counts().2 >= 1, "no keyframe request");
            }
            eprintln!("  rust late reader: decode errors {}, keyframe requests {}", src.decoder_state().0, src.counts().2);
            src.close();
            (n, dmg, why)
        };
        if let Some(p) = early_py {
            p.wait_with_output().unwrap();
        }
        if let Some(t) = early_rs {
            t.join().unwrap();
        }
        let _ = Command::new("kill").arg(fake.id().to_string()).status();
        let _ = fake.wait_with_output();
        eprintln!(
            "saved header, {first} first: file {} bytes; the late reader got {late_n} ticks ({late_damaged} held as damaged){}",
            saved.len(),
            if why.is_empty() { String::new() } else { format!("; why: {why:?}") }
        );
        if *first == "none" {
            assert!(saved.is_empty() && late_n == 0 && why.contains("started before SmoovPlay"));
        } else {
            // u32 session, u32 codec (1), then VPS/SPS/PPS after 3-byte start codes
            assert!(saved.len() > 8 && saved[4..8] == 1u32.to_le_bytes() && saved[8..11] == [0, 0, 1], "{saved:?}");
            assert!(late_n > 100, "the late reader decoded {late_n} ticks");
            assert!(late_damaged >= 55, "the first ~60 ticks after joining are held");
        }
        sp_tap::unlink(&name);
        let _ = std::fs::remove_file(sp_tap::saved::header_path(&name));
        let _ = std::fs::remove_file(&out);
    }
}
