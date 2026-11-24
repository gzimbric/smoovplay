//! Interpolation turned off and on again while Even out or Dip smoothing is timing frames. The timeline wasn't fed
//! while interpolation was off, so turning it on served frames from before (a seconds-old picture, then a blend of
//! it with the current one), and at a repeat tick while the game was already dipping it asked place() for frames it
//! didn't have and panicked. framegen/pipeline.py had the same two faults; both now start the timeline afresh at the
//! toggle and don't time frames before it has one.
mod common;

use std::sync::{Arc, Mutex};

use common::*;
use serde_json::{Value, json};
use sp_core::{Presenter, Source};
use sp_sched::{DuplicateDetector, FakeClock, LiveSettings, Pipeline, RunOptions, SharedLive};

const TICK: f64 = 1.0 / 60.0;

/// A 30 fps game that holds every 4th frame for 3 ticks (a dip), as 60 Hz ticks with the source's new hint, and
/// `set interp true` at the first tick at or after `toggle_at` whose hint is `toggle_on_new`.
fn dipping(n_frames: usize, toggle_at: usize, toggle_on_new: bool) -> Vec<Value> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut tag = 0.0;
    for f in 0..n_frames {
        let hold = if f % 4 == 3 { 3 } else { 2 };
        for k in 0..hold {
            out.push(json!({"at": (i + 1) as f64 * TICK, "tag": tag, "hint": k == 0}));
            i += 1;
        }
        tag += 1.0;
    }
    let idx = (toggle_at..out.len()).find(|&j| out[j]["hint"].as_bool() == Some(toggle_on_new)).unwrap();
    out[idx]["set"] = json!({"interp": true});
    out
}

fn run(ticks: Vec<Value>, settings: Value) -> Vec<Value> {
    let clock = Arc::new(FakeClock::new(START));
    let ev: Events = Arc::new(Mutex::new(Vec::new()));
    let settings = Arc::new(SharedLive::new(live_from(&settings)));
    let mut source = TraceSource::new(ticks, clock.clone(), Some(settings.clone()), None, true, (8, 8), ev.clone());
    let mut sink = TraceSink::new(clock.clone(), ev.clone(), &json!({}));
    let interp = TraceInterp::new(clock.clone(), ev.clone(), &json!({}), (8, 8));
    let log_ev = ev.clone();
    let mut p = Pipeline::new(interp, Some(DuplicateDetector::default()))
        .with_clock(clock.clone())
        .with_probe(Box::new(TagProbe))
        .with_log(Box::new(move |line| push(&log_ev, json!(["log", line]))));
    let opts = RunOptions { settings: Some(&*settings as &dyn LiveSettings), ..RunOptions::default() };
    let _ = p.run(&mut source as &mut dyn Source, Some(&mut sink as &mut dyn Presenter), opts);
    let e = ev.lock().unwrap().clone();
    e
}

/// Every "tag" number in a written frame's tag: a source frame's {"tag": n}, or {"ai": [a, b, t]} with a and b tags.
fn tags(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Object(m) => {
            if let Some(t) = m.get("tag").and_then(|t| t.as_f64()) {
                out.push(t);
            }
            for (k, x) in m {
                if k != "tag" {
                    tags(x, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| tags(x, out)),
        _ => {}
    }
}

/// From `from_t` seconds in, no written frame is more than `back` game frames behind the newest one shown so far.
fn no_old_frames(ev: &[Value], from_t: f64, back: f64) {
    let mut newest = f64::MIN;
    let mut checked = 0;
    for e in ev.iter().filter(|e| e[0] == "write") {
        let mut t = Vec::new();
        tags(&e[2], &mut t);
        let Some(max) = t.iter().copied().reduce(f64::max) else { continue };
        newest = newest.max(max);
        let at = e[1].as_f64().unwrap() - START;
        if at >= from_t {
            checked += 1;
            for x in &t {
                assert!(*x >= newest - back, "a frame {} behind the newest ({newest}) was shown at {at:.3} s: {}", newest - x, e[2]);
            }
        }
    }
    assert!(checked > 50, "only {checked} writes after {from_t:.2} s");
}

#[test]
fn interp_on_at_a_repeat_tick_while_dipping_doesnt_panic() {
    let ev = run(dipping(120, 200, false), json!({"hz": 60, "interp": false, "even_dips": true, "overlay": false}));
    assert!(ev.iter().any(|e| e[0] == "log" && e[1].as_str().is_some_and(|s| s.contains("interpolation ON"))));
    assert!(ev.iter().filter(|e| e[0] == "write").count() > 250, "the run ended early");
}

#[test]
fn interp_off_then_on_while_dipping_shows_no_old_frames() {
    let mut ticks = dipping(200, 300, false);
    ticks[0]["set"] = json!({"interp": true});
    ticks[100]["set"] = json!({"interp": false});
    let ev = run(ticks, json!({"hz": 60, "interp": false, "even_dips": true, "overlay": false}));
    no_old_frames(&ev, 300.0 * TICK, 6.0);
}

#[test]
fn even_out_interp_off_then_on_shows_no_old_frames() {
    // a 45 fps game (new on 3 ticks of 4) with Even out, the default
    let mut out = Vec::new();
    let mut tag = -1.0;
    for i in 0..600usize {
        let new = i % 4 != 3;
        if new {
            tag += 1.0;
        }
        out.push(json!({"at": (i + 1) as f64 * TICK, "tag": tag, "hint": new}));
    }
    out[200]["set"] = json!({"interp": false});
    let idx = (400..600).find(|&j| out[j]["hint"].as_bool() == Some(false)).unwrap();
    out[idx]["set"] = json!({"interp": true});
    let ev = run(out, json!({"hz": 60, "interp": true, "overlay": false}));
    no_old_frames(&ev, 400.0 * TICK, 6.0);
}
