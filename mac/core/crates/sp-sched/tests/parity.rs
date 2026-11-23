//! Parity with the Python pipeline. Every trace tools/sched_traces.py recorded (fixtures/sched/trace_*.json) is
//! replayed through the Rust pipeline with the same inputs, stand-ins and fake clock, and must come out the same.
//! That covers every event (frames shown and when, overlay texts, toasts, log lines, presses, the sound delay, replay
//! calls, per-tick decisions), Stats' lists and counts, summary(), results(), and the results report built from them.
//!
//! Every number is compared exactly, bit for bit, the detector's block differences and the PSNRs included. Two
//! numbers cross an f32 boundary on the Rust side and are compared to 1e-6 instead: the AI frame's fraction t
//! (sp_core's Interpolator takes f32) and the Steady HUD threshold (f32).
//!
//! Without fixtures (they're generated and git-ignored) the test says so and passes.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::*;
use serde_json::{Value, json};
use sp_core::{Presenter, Source};
use sp_sched::report::{ReportHeader, SessionInfo, results_lines};
use sp_sched::{
    DuplicateDetector, FakeClock, GpuProbe, LiveSettings, Pipeline, Probe, Reference, Remote, Replay, RunOptions,
    SharedLive, Stats, Tuning,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sched")
}

struct Ran {
    events: Vec<Value>,
    stats: Stats,
    settings_after: Option<Value>,
    replay_counts: Option<(u64, u64)>,
}

fn run_trace(fx: &Value) -> Ran {
    let cfg = &fx["config"];
    let clock = Arc::new(FakeClock::new(START));
    let ev: Events = Arc::new(Mutex::new(Vec::new()));
    let size = cfg
        .get("size")
        .map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize))
        .unwrap_or((128, 96));
    let settings: Option<Arc<SharedLive>> =
        cfg.get("settings").filter(|s| !s.is_null()).map(|s| Arc::new(SharedLive::new(live_from(s))));
    let mut remote = cfg.get("remote").filter(|r| !r.is_null()).map(|r| TraceRemote::new(r, clock.clone(), ev.clone()));
    let mut replay = cfg.get("replay").and_then(|r| r.as_bool()).unwrap_or(false).then(|| TraceReplay {
        ev: ev.clone(),
        frames: 0,
        real: 0,
    });
    let tag_probe = cfg.get("probe").and_then(|p| p.as_str()) == Some("tag");
    let ticks = fx["ticks"].as_array().unwrap().clone();
    let src = TraceSource::new(
        ticks,
        clock.clone(),
        settings.clone(),
        remote.as_ref().map(|r| r.st.clone()),
        tag_probe,
        size,
        ev.clone(),
    );
    let mut source: Box<dyn Source> = if cfg.get("testmode").and_then(|t| t.as_bool()).unwrap_or(false) {
        Box::new(TestMode::new(src, settings.clone().unwrap()))
    } else {
        Box::new(src)
    };
    let mut sink = cfg
        .get("sink")
        .and_then(|s| s.as_bool())
        .unwrap_or(true)
        .then(|| TraceSink::new(clock.clone(), ev.clone(), cfg.get("sink_cfg").unwrap_or(&json!({}))));
    let interp = TraceInterp::new(clock.clone(), ev.clone(), cfg.get("interp").unwrap_or(&json!({})), size);
    let d = cfg.get("detector").cloned().unwrap_or(json!({}));
    let det = DuplicateDetector::new(
        d.get("threshold").and_then(|v| v.as_f64()).unwrap_or(2.0),
        if d.get("reference").and_then(|v| v.as_str()) == Some("new") { Reference::New } else { Reference::Previous },
        d.get("learn").and_then(|v| v.as_bool()).unwrap_or(false),
    );
    let probe: Box<dyn Probe> = if tag_probe { Box::new(TagProbe) } else { Box::new(GpuProbe::new()) };
    let log_ev = ev.clone();
    let mut p = Pipeline::new(interp, Some(det))
        .with_clock(clock.clone())
        .with_probe(probe)
        .with_log(Box::new(move |line| push(&log_ev, json!(["log", line]))));
    let t = cfg.get("tuning").cloned().unwrap_or(json!({}));
    let def = Tuning::default();
    let tf = |k: &str, d: f64| t.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    p.tuning = Tuning {
        rate_note_s: tf("rate_note_s", def.rate_note_s),
        rate_toast_s: tf("rate_toast_s", def.rate_toast_s),
        delay_settle_s: tf("delay_settle_s", def.delay_settle_s),
        delay_gap_s: tf("delay_gap_s", def.delay_gap_s),
    };
    let r = cfg.get("run").cloned().unwrap_or(json!({}));
    let tick_ev = ev.clone();
    let mut on_tick = move |tick: &sp_core::Tick, diff: &sp_sched::FrameDiff, action: sp_sched::Action, proc_ms: f64, t0: f64| {
        push(
            &tick_ev,
            json!(["tick", tick.index, action.as_str(), diff.is_new, num(diff.max_diff), num(diff.mean_diff), num(proc_ms), num(t0)]),
        );
    };
    let opts = RunOptions {
        enabled: r.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
        score: r.get("score").and_then(|v| v.as_bool()).unwrap_or(false),
        label: r.get("label").and_then(|v| v.as_str()).unwrap_or("RIFE").to_string(),
        hz: r.get("hz").and_then(|v| v.as_u64()).unwrap_or(60) as u32,
        settings: settings.as_deref().map(|s| s as &dyn LiveSettings),
        remote: remote.as_mut().map(|r| r as &mut dyn Remote),
        replay: replay.as_mut().map(|r| r as &mut dyn Replay),
        on_tick: Some(&mut on_tick),
        gt: None,
    };
    let stats = p.run(&mut *source, sink.as_mut().map(|s| s as &mut dyn Presenter), opts);
    drop(source);
    let events = ev.lock().unwrap().clone();
    Ran {
        events,
        stats,
        settings_after: settings.map(|s| live_json(&s.live())),
        replay_counts: replay.map(|r| (r.frames, r.real)),
    }
}

fn close(a: &Value, b: &Value, tol: f64) -> bool {
    match (a, b) {
        (Value::Number(_), Value::Number(_)) => {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            x == y || (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs()))
        }
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| close(p, q, tol)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| close(v, w, tol)))
        }
        _ => a == b,
    }
}

/// An event, with the two numbers that cross an f32 boundary compared to 1e-6.
fn event_close(py: &Value, rs: &Value) -> bool {
    let kind = py.get(0).and_then(|k| k.as_str()).unwrap_or("");
    match kind {
        "write" => {
            let (p, r) = (py.as_array().unwrap(), rs.as_array().unwrap());
            p.len() == r.len()
                && close(&p[0], &r[0], 0.0)
                && close(&p[1], &r[1], 0.0)
                && close(&p[2], &r[2], 1e-6)
                && close(&p[3], &r[3], 0.0)
        }
        "steady" => close(py, rs, 1e-6),
        _ => close(py, rs, 0.0),
    }
}

fn list(st: &Stats, k: &str) -> Vec<f64> {
    match k {
        "proc_interp_ms" => st.proc_interp_ms.clone(),
        "proc_other_ms" => st.proc_other_ms.clone(),
        "sink_ms" => st.sink_ms.clone(),
        "queue_ms" => st.queue_ms.clone(),
        "score_ms" => st.score_ms.clone(),
        "busy_ms" => st.busy_ms.clone(),
        "real_delay_ms" => st.real_delay_ms.clone(),
        "psnr_interp" => st.psnr_interp.clone(),
        "psnr_repeat" => st.psnr_repeat.clone(),
        "psnr_blend" => st.psnr_blend.clone(),
        "off_interp" => st.off_interp.clone(),
        "off_repeat" => st.off_repeat.clone(),
        "psnr_still" => st.psnr_still.clone(),
        "psnr_still_repeat" => st.psnr_still_repeat.clone(),
        "still_frac" => st.still_frac.clone(),
        "psnr_hud" => st.psnr_hud.clone(),
        "hud_off" => st.hud_off.clone(),
        "hud_frac" => st.hud_frac.clone(),
        "new_gaps_ms" => st.new_gaps_ms.clone(),
        _ => panic!("{k}"),
    }
}

fn count(st: &Stats, k: &str) -> u64 {
    match k {
        "quarter_shown" => st.quarter_shown,
        "quarter_skipped" => st.quarter_skipped,
        "behind" => st.behind,
        "simulated" => st.simulated,
        "test_game_new" => st.test_game_new,
        "test_game_repeat" => st.test_game_repeat,
        "uneven_ticks" => st.uneven_ticks,
        "uneven_waits" => st.uneven_waits,
        "dip_ticks" => st.dip_ticks,
        "stale" => st.stale,
        _ => panic!("{k}"),
    }
}

/// What differs between the Python's trace and the Rust run. Empty when they match.
fn compare(fx: &Value, ran: &Ran) -> Vec<String> {
    let mut bad = Vec::new();
    let (py, rs) = (fx["events"].as_array().unwrap(), &ran.events);
    let mut shown = 0;
    for (i, (p, r)) in py.iter().zip(rs).enumerate() {
        if !event_close(p, r) {
            if shown < 4 {
                let ctx: Vec<String> = py[i.saturating_sub(2)..i].iter().map(|e| e.to_string()).collect();
                bad.push(format!("event {i}: python {p}\n      rust   {r}\n      after  {}", ctx.join(" | ")));
            }
            shown += 1;
        }
    }
    if shown > 4 {
        bad.push(format!("...{} events differ in all", shown));
    }
    if py.len() != rs.len() {
        bad.push(format!("events: python {} rust {}", py.len(), rs.len()));
    }
    let s = &fx["stats"];
    for (k, v) in s["lists"].as_object().unwrap() {
        let r: Vec<Value> = list(&ran.stats, k).into_iter().map(num).collect();
        if !close(v, &Value::Array(r.clone()), 0.0) {
            let n = v.as_array().unwrap().len();
            let first = v.as_array().unwrap().iter().zip(&r).position(|(a, b)| !close(a, b, 0.0));
            bad.push(format!("stats.{k}: python {n} rust {} first difference at {first:?}", r.len()));
        }
    }
    for (k, v) in s["counts"].as_object().unwrap() {
        if v.as_u64().unwrap() != count(&ran.stats, k) {
            bad.push(format!("stats.{k}: python {v} rust {}", count(&ran.stats, k)));
        }
    }
    let pairs = |c: &sp_sched::stats::Counts| Value::Array(c.0.iter().map(|(k, n)| json!([k, n])).collect());
    if s["actions"] != pairs(&ran.stats.actions) {
        bad.push(format!("actions: python {} rust {}", s["actions"], pairs(&ran.stats.actions)));
    }
    if s["dedup"] != pairs(&ran.stats.dedup) {
        bad.push(format!("dedup: python {} rust {}", s["dedup"], pairs(&ran.stats.dedup)));
    }
    let hz: Vec<u32> = ran.stats.hz_used.iter().copied().collect();
    if s["hz_used"] != json!(hz) {
        bad.push(format!("hz_used: python {} rust {hz:?}", s["hz_used"]));
    }
    if s["uneven_fps"] != json!(ran.stats.uneven_fps) {
        bad.push("uneven_fps differ".into());
    }
    if s["delay_tests"] != json!(ran.stats.delay_tests) {
        bad.push(format!("delay_tests: python {} rust {:?}", s["delay_tests"], ran.stats.delay_tests));
    }
    if s["exit"] != json!(ran.stats.exit.as_str()) {
        bad.push(format!("exit: python {} rust {}", s["exit"], ran.stats.exit.as_str()));
    }
    if !close(&s["wall_s"], &json!(ran.stats.wall_s), 0.0) {
        bad.push(format!("wall_s: python {} rust {}", s["wall_s"], ran.stats.wall_s));
    }
    let worst = ran.stats.worst.as_ref();
    if !close(&s["worst_psnr"], &worst.map_or(Value::Null, |w| num(w.psnr)), 0.0)
        || s["worst_tick"] != worst.map_or(Value::Null, |w| json!(w.tick))
    {
        bad.push(format!("worst: python {} at {} rust {:?}", s["worst_psnr"], s["worst_tick"], worst.map(|w| (w.psnr, w.tick))));
    }
    let text_diff = |name: &str, py: &[String], rs: &[String], bad: &mut Vec<String>| {
        if py != rs {
            let i = py.iter().zip(rs).position(|(a, b)| a != b).unwrap_or(py.len().min(rs.len()));
            bad.push(format!(
                "{name} line {i}:\n      python {:?}\n      rust   {:?}",
                py.get(i).map(String::as_str).unwrap_or("<none>"),
                rs.get(i).map(String::as_str).unwrap_or("<none>")
            ));
        }
    };
    let lines = |v: &Value| -> Vec<String> { v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
    let py_summary: Vec<String> = fx["summary"].as_str().unwrap().split('\n').map(String::from).collect();
    let rs_summary: Vec<String> = ran.stats.summary().split('\n').map(String::from).collect();
    text_diff("summary", &py_summary, &rs_summary, &mut bad);
    text_diff("results", &lines(&fx["results"]), &ran.stats.results(None), &mut bad);
    text_diff("results(full_motion=False)", &lines(&fx["results_no_full_motion"]), &ran.stats.results(Some(false)), &mut bad);
    if !fx["settings_after"].is_null() && Some(&fx["settings_after"]) != ran.settings_after.as_ref() {
        bad.push(format!("settings after: python {} rust {:?}", fx["settings_after"], ran.settings_after));
    }
    if !fx["replay_counts"].is_null() {
        let (f, r) = ran.replay_counts.unwrap();
        if fx["replay_counts"] != json!([f, r]) {
            bad.push(format!("replay counts: python {} rust [{f}, {r}]", fx["replay_counts"]));
        }
    }
    bad
}

#[test]
fn traces_match_python() {
    let dir = fixtures();
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(d) => d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| {
            p.file_name().unwrap().to_string_lossy().starts_with("trace_")
        }).collect(),
        Err(_) => {
            eprintln!("no fixtures in {}: run tools/sched_traces.py first", dir.display());
            return;
        }
    };
    files.sort();
    let only = std::env::var("TRACE").ok();
    let mut failed = Vec::new();
    let mut stats: BTreeMap<String, Stats> = BTreeMap::new();
    for f in &files {
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        let name = fx["name"].as_str().unwrap().to_string();
        if only.as_ref().is_some_and(|o| !o.split(',').any(|x| x == name)) {
            continue;
        }
        let ran = run_trace(&fx);
        let bad = compare(&fx, &ran);
        let n = fx["events"].as_array().unwrap().len();
        if bad.is_empty() {
            eprintln!("MATCH  {name:<26} {n:>5} events");
        } else {
            eprintln!("DIFFER {name:<26} {n:>5} events\n  {}", bad.join("\n  "));
            failed.push(name.clone());
        }
        stats.insert(name, ran.stats);
    }
    // The results report (run.py write_results) from the same sessions.
    if only.is_none()
        && let Ok(text) = std::fs::read_to_string(dir.join("results_report.json")) {
            let rep: Value = serde_json::from_str(&text).unwrap();
            let h = &rep["header"];
            let header = ReportHeader {
                stamp: h["stamp"].as_str().unwrap().into(),
                size: h["size"].as_str().unwrap().into(),
                gpu: h["gpu"].as_str().unwrap().into(),
                upscaler: h["upscaler"].as_str().unwrap().into(),
            };
            for r in rep["reports"].as_array().unwrap() {
                let sessions: Vec<(SessionInfo, &Stats)> = r["sessions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let cfg = &s[1];
                        let q = cfg.get("vfg_quality").and_then(|v| v.as_str()).unwrap_or("");
                        let info = SessionInfo {
                            engine: cfg["engine"].as_str().unwrap().into(),
                            model: cfg["model"].as_str().unwrap().into(),
                            motion: cfg["motion"].as_f64().unwrap(),
                            vfg_quality: r["vfg_quality_names"].get(q).and_then(|v| v.as_str()).unwrap_or("").into(),
                        };
                        (info, &stats[s[0].as_str().unwrap()])
                    })
                    .collect();
                let got = results_lines(&header, &sessions);
                let want: Option<Vec<String>> = r["lines"]
                    .as_array()
                    .map(|l| l.iter().map(|x| x.as_str().unwrap().to_string()).collect());
                let name = r["name"].as_str().unwrap();
                if got == want {
                    eprintln!("MATCH  report {name}");
                } else {
                    eprintln!("DIFFER report {name}\n  python {want:#?}\n  rust   {got:#?}");
                    failed.push(format!("report {name}"));
                }
            }
        }
    assert!(failed.is_empty(), "differ from the Python: {failed:?}");
}
