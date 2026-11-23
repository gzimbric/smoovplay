//! tests/test_streamcheck.py, ported: the same made-up streams (through the real size check, which is sp-tap's, so
//! the rows it gave come from fixtures/sched/streamcheck.json), the same assertions, and the report must be the
//! Python's line for line.

use std::sync::OnceLock;

use serde_json::Value;
use sp_core::{Frame, Tick};
use sp_sched::FrameDiff;
use sp_sched::streamcheck::StreamCheck;

fn cases() -> Option<&'static Value> {
    static C: OnceLock<Option<Value>> = OnceLock::new();
    C.get_or_init(|| {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sched/streamcheck.json");
        std::fs::read_to_string(p).ok().map(|t| serde_json::from_str(&t).unwrap())
    })
    .as_ref()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| match v.as_str() {
        Some("inf") => f64::INFINITY,
        Some("-inf") => f64::NEG_INFINITY,
        _ => f64::NAN,
    })
}

/// The case's rows through the Rust StreamCheck; its report, checked against the Python's.
fn play(name: &str) -> Option<Vec<String>> {
    let Some(c) = cases() else {
        eprintln!("no fixtures: run tools/sched_traces.py first");
        return None;
    };
    let c = &c[name];
    let r = &c["rows"];
    let mut sc = StreamCheck::default();
    for i in 0..r["t"].as_array().unwrap().len() {
        sc.add(
            f(&r["t"][i]),
            r["seq"][i].as_i64().unwrap(),
            r["size"][i].as_i64().unwrap(),
            r["ref"][i].as_f64(),
            r["dec"][i].as_bool(),
            r["key"][i].as_bool().unwrap(),
            r["lost"][i].as_bool().unwrap(),
            f(&r["md"][i]),
            f(&r["mx"][i]),
        );
    }
    let lines = sc.report();
    let want: Vec<String> = c["report"].as_array().unwrap().iter().map(|l| l.as_str().unwrap().to_string()).collect();
    assert_eq!(lines, want, "{name}: the report differs from the Python's");
    Some(lines)
}

fn verdict(lines: &[String]) -> &str {
    lines.iter().find(|l| l.starts_with("Verdict:")).unwrap()
}

fn breaks(lines: &[String]) -> &str {
    lines.iter().find(|l| l.trim().starts_with("breaks:")).unwrap()
}

#[test]
fn test_locked_30_reads_correctly() {
    let Some(lines) = play("locked_30") else { return };
    assert!(verdict(&lines).contains("read this 30 fps game correctly"));
    assert!(lines.iter().any(|l| l.starts_with("Rhythm") && l.contains("100.0%")));
    assert!(lines.iter().any(|l| l.contains("30 fps (median)")));
    assert!(lines.iter().any(|l| l.contains("pixel-identical to the frame before: 100.0% of \"repeats\"")));
}

#[test]
fn test_game_hitches_are_the_games_not_misreads() {
    let Some(lines) = play("game_hitches") else { return };
    assert!(verdict(&lines).contains("read this 30 fps game correctly"));
    let b = breaks(&lines);
    let causes: Vec<(String, u64)> = b
        .split_once(':')
        .unwrap()
        .1
        .trim()
        .split(", ")
        .map(|p| {
            let (k, v) = p.rsplit_once(' ').unwrap();
            (k.to_string(), v.parse().unwrap())
        })
        .collect();
    let game = causes.iter().find(|(k, _)| k == "the game itself (the picture agrees)").map_or(0, |c| c.1);
    let total: u64 = causes.iter().map(|c| c.1).sum();
    assert!(game as f64 >= 0.9 * total as f64, "{b}");
    assert!(lines.iter().any(|l| l.contains("held frames:") && l.contains("longest")));
}

#[test]
fn test_the_size_check_recovers_after_a_long_hold() {
    let Some(lines) = play("long_hold") else { return };
    assert!(verdict(&lines).contains("read this 30 fps game correctly") && !verdict(&lines).contains("undecided"));
}

#[test]
fn test_misreads_are_caught_by_the_picture() {
    let Some(lines) = play("misreads") else { return };
    assert!(verdict(&lines).contains("misread this 30 fps game too often"));
    assert!(breaks(&lines).contains("picture says misread"));
}

#[test]
fn test_a_few_misreads_is_mostly_right() {
    let Some(lines) = play("few_misreads") else { return };
    assert!(verdict(&lines).contains("mostly read this 30 fps game correctly"));
}

#[test]
fn test_packet_loss_explains_breaks() {
    let Some(lines) = play("packet_loss") else { return };
    assert!(breaks(&lines).contains("packet loss"));
}

#[test]
fn test_missing_frames_explain_breaks() {
    let Some(lines) = play("missing") else { return };
    assert!(lines.iter().any(|l| l.contains("never did")));
    assert!(verdict(&lines).contains("read this 30 fps game correctly"));
}

#[test]
fn test_60_fps_game() {
    let Some(lines) = play("sixty") else { return };
    assert!(verdict(&lines).contains("about 60 fps"));
}

#[test]
fn test_varying_frame_rate() {
    let Some(lines) = play("varying") else { return };
    assert!(verdict(&lines).contains("frame rate varied"));
}

#[test]
fn test_30_fps_stream() {
    let Some(lines) = play("stream_30") else { return };
    assert!(verdict(&lines).contains("stream itself is 30 fps"));
}

#[test]
fn test_too_short() {
    let Some(lines) = play("too_short") else { return };
    assert!(verdict(&lines).contains("too short"));
}

#[test]
fn test_still_scene_cant_be_confirmed() {
    let Some(lines) = play("still") else { return };
    assert!(lines.iter().any(|l| l.starts_with("Picture check: too few")));
    assert!(verdict(&lines).contains("couldn't confirm"));
}

#[test]
fn test_test_mode_frames_are_left_out() {
    let Some(c) = cases() else { return };
    let c = &c["test_mode_left_out"];
    let mut sc = StreamCheck::default();
    let frame = Frame::from_bgr(&[0u8; 12], 2, 2);
    for (i, t) in c["ticks"].as_array().unwrap().iter().enumerate() {
        let mut tick = Tick::new(frame.clone(), t["t"].as_f64().unwrap(), i as u64);
        tick.truth_new = t["truth"].as_bool();
        tick.extra = t["extra"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let diff = FrameDiff { is_new: false, max_diff: t["mx"].as_f64().unwrap(), mean_diff: t["md"].as_f64().unwrap(), cut: None };
        sc.add_tick(&tick, &diff);
    }
    assert_eq!(sc.simulated, 400);
    let lines = sc.report();
    assert!(lines[0].contains("left out"));
    let want: Vec<String> = c["report"].as_array().unwrap().iter().map(|l| l.as_str().unwrap().to_string()).collect();
    assert_eq!(lines, want);
}

#[test]
fn test_empty_and_capped() {
    assert!(StreamCheck::default().report().is_empty());
    let mut sc = StreamCheck::new(10);
    for i in 0..20 {
        sc.add(i as f64 / 60.0, i, 1000, None, None, false, false, 0.0, 0.0);
    }
    assert!(sc.extra == 10 && sc.t.len() == 10);
}
