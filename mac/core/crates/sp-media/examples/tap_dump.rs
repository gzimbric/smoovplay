//! Reads a live frame tap and prints what each tick says, with timings.
//!
//!     cargo run --release -p sp-media --example tap_dump -- 'Local\NAME' [--test-mode] [--ticks N] [--wait S]
//!         [--hdr sdr8|pq16] [--jsonl OUT] [--quiet]
//!
//! Compressed frames (chiaki-ng, the fake PS5) go through ChiakiSource and decoded frames through TapSource, as
//! open_tap picks. --jsonl writes one JSON line per tick with the fields mac/core/tools/media_ticks.py writes for the
//! Python reader, for comparing the two. At the end it prints decode time and delay (arrival to the pipeline) at p50
//! and p95, and counts.

use std::io::Write;

use serde_json::{json, Value};
use sp_core::{clock, Event, Source};
use sp_media::{open_tap, HdrOut, HdrSettings, OpenedTap, TestMode};

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(name) = args.first().cloned() else {
        eprintln!("usage: tap_dump NAME [--test-mode] [--ticks N] [--wait S] [--hdr sdr8|pq16] [--jsonl OUT] [--quiet]");
        std::process::exit(2);
    };
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let ticks: u64 = opt("--ticks").and_then(|v| v.parse().ok()).unwrap_or(0);
    let wait: f64 = opt("--wait").and_then(|v| v.parse().ok()).unwrap_or(600.0);
    let quiet = flag("--quiet");
    let test = flag("--test-mode").then(|| TestMode::shared(true));
    let mut jsonl = opt("--jsonl").map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("--jsonl")));

    let opened = match open_tap(&name, wait, test, None) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("tap_dump: {e}");
            std::process::exit(1);
        }
    };
    let kind = opened.kind();
    let mut src: Box<dyn Source> = match opened {
        OpenedTap::Chiaki(mut c) => {
            if let Some(h) = opt("--hdr").and_then(|v| HdrOut::parse(&v)) {
                c.set_hdr(Some(HdrSettings::new(h, None)), "");
            }
            Box::new(c)
        }
        o => o.into_source(),
    };
    let (w, h) = src.size();
    println!("tap_dump: {kind} source, {w}x{h}, {:?}", src.info().name);
    let (mut decode, mut delay, mut gaps) = (Vec::new(), Vec::new(), Vec::new());
    let (mut n, mut waits, mut damaged) = (0u64, 0u64, 0u64);
    let mut last_arrival: Option<f64> = None;
    let t_start = clock::now();
    while let Some(ev) = src.next_event() {
        let t = match ev {
            Event::Waiting(w) => {
                waits += 1;
                if !quiet && waits % 10 == 1 {
                    println!("  waiting {:.1} s for {}{}", w.since, w.what, if w.why.is_empty() { String::new() } else { format!(": {}", w.why) });
                }
                continue;
            }
            Event::Tick(t) => t,
        };
        let now = clock::now();
        let e = &t.extra;
        let g = |k: &str| e.get(k).cloned().unwrap_or(Value::Null);
        let dms = e.get("decode_ms").and_then(|v| v.as_f64());
        if let Some(d) = dms {
            decode.push(d);
        }
        delay.push((now - t.t_arrival) * 1000.0);
        if let Some(l) = last_arrival {
            gaps.push((t.t_arrival - l) * 1000.0);
        }
        last_arrival = Some(t.t_arrival);
        damaged += t.damaged as u64;
        if let Some(out) = jsonl.as_mut() {
            let line = json!({
                "index": t.index, "seq": g("seq"), "size": g("size"), "size_new": g("size_new"), "size_ref": g("size_ref"),
                "big": g("big"), "keyframe": g("keyframe"), "frames_lost": g("frames_lost"), "recovered": g("recovered"),
                "damaged": t.damaged, "content": t.content, "truth_new": t.truth_new, "new_hint": t.new_hint,
                "game_new": g("game_new"), "t_arrival": t.t_arrival, "t_ready": t.t_ready, "decode_ms": dms,
            });
            let _ = writeln!(out, "{line}");
        }
        if !quiet {
            let b = |v: Value| match v {
                Value::Bool(true) => "y".to_string(),
                Value::Bool(false) => "n".to_string(),
                Value::Null => "-".to_string(),
                v => v.to_string(),
            };
            println!(
                "tick {:5} seq {:>5} {:>7} B  size_new {} key {} lost {} rec {} dmg {}  content {:>5} truth_new {} hint {}  \
                 decode {:5.2} ms  delay {:6.2} ms",
                t.index,
                b(g("seq")),
                b(g("size")),
                b(g("size_new")),
                b(g("keyframe")),
                b(g("frames_lost")),
                b(g("recovered")),
                if t.damaged { "y" } else { "n" },
                t.content.map(|c| c.to_string()).unwrap_or("-".into()),
                t.truth_new.map(|v| if v { "y" } else { "n" }).unwrap_or("-"),
                t.new_hint.map(|v| if v { "y" } else { "n" }).unwrap_or("-"),
                dms.unwrap_or(f64::NAN),
                (now - t.t_arrival) * 1000.0
            );
        }
        n += 1;
        if ticks > 0 && n >= ticks {
            break;
        }
    }
    let secs = clock::now() - t_start;
    src.close();
    println!(
        "tap_dump: {n} ticks in {secs:.1} s ({:.1}/s), {damaged} damaged, {waits} waits; decode p50 {:.2} ms p95 {:.2} ms; \
         arrival to here p50 {:.2} ms p95 {:.2} ms; arrival gaps p50 {:.2} ms p95 {:.2} ms",
        n as f64 / secs.max(1e-9),
        pct(&mut decode.clone(), 0.5),
        pct(&mut decode, 0.95),
        pct(&mut delay.clone(), 0.5),
        pct(&mut delay, 0.95),
        pct(&mut gaps.clone(), 0.5),
        pct(&mut gaps, 0.95),
    );
}
