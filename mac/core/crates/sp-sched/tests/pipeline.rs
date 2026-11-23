//! The pipeline tests of tests/test_timeline.py, test_120fps.py, test_game_rate.py, test_dips.py, test_delaytest.py,
//! test_waiting.py, test_watchdog.py and test_steady.py, ported: same names, same assertions. The Python runs them
//! on the real clock with sped-up ticks; here a FakeClock moves as the ticks would.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use common::pyrandom::PyRandom;
use common::*;
use serde_json::{Value, json};
use sp_core::{Event, Frame, Interpolator, SinkReply, Tick, Waiting};
use sp_sched::stats::simulate_off;
use sp_sched::{
    Clock, Exit, FakeClock, GpuProbe, Live, LiveSettings, OverlayStyle, Pipeline, Remote, Replay, RunOptions,
    STEADY_HUD_THRESHOLD, SharedLive, Stats, SystemClock, Tuning, rate_advice,
};

type Logs = Arc<Mutex<Vec<String>>>;

fn pipeline<I: Interpolator>(interp: I, clock: &Arc<FakeClock>, logs: &Logs) -> Pipeline<I> {
    let logs = logs.clone();
    Pipeline::new(interp, None)
        .with_clock(clock.clone())
        .with_probe(Box::new(TagProbe))
        .with_log(Box::new(move |l| logs.lock().unwrap().push(l.to_string())))
}

fn settings(f: impl FnOnce(&mut Live)) -> Arc<SharedLive> {
    let mut l = Live::default();
    f(&mut l);
    Arc::new(SharedLive::new(l))
}

fn run<I: Interpolator, S: sp_core::Source>(p: &mut Pipeline<I>, src: &mut S, sink: &mut RecSink, s: &Arc<SharedLive>) -> Stats {
    let opts = RunOptions { label: "RIFE".into(), settings: Some(s.as_ref() as &dyn LiveSettings), ..Default::default() };
    p.run(src, Some(sink), opts)
}

fn no_waits() -> Tuning {
    Tuning { rate_note_s: 0.0, rate_toast_s: 0.0, ..Tuning::default() }
}

/// test_game_rate.py _game: a live 60 Hz stream, sped up: a tick every 2 ms there, plus the Python pipeline's own
/// time per tick, which a fake clock doesn't have (2.5 ms here), so the overlay sees a second of frames go by.
fn game(clock: Arc<FakeClock>, pattern: impl Fn(u64) -> bool + Send + 'static, n: u64) -> impl sp_core::Source {
    let (mut i, mut content) = (0u64, 0u64);
    FnSource::new(move || {
        if i >= n {
            return None;
        }
        clock.advance(0.0025);
        let new = pattern(i);
        content += new as u64;
        let t = tag_tick((content % 250) as f64, clock.now(), i, Some(new));
        i += 1;
        Some(Event::Tick(t))
    })
}

fn text(o: &Option<sp_core::Overlay>) -> &str {
    o.as_ref().map_or("", |o| o.text.as_str())
}

// ------------------------------------------------------------------------------------------------ test_game_rate.py
#[test]
fn test_overlay_shows_the_games_rate_and_advice_for_a_45_fps_game() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let mut sink = RecSink::default();
    let s = settings(|l| {
        l.even_out = false;
        l.overlay_style = OverlayStyle::Detailed;
    });
    run(&mut p, &mut game(clock.clone(), |i| i % 4 != 3, 600), &mut sink, &s);
    let last = text(sink.texts.last().unwrap());
    assert!(last.starts_with("RIFE  INTERP ON  game 45 fps, showing "), "{last}");
    assert_eq!(last.split('\n').nth(1).unwrap(), rate_advice(Some(45.0), false, false));
    assert_eq!(sink.toasts, [rate_advice(Some(45.0), false, false)]); // once
}

#[test]
fn test_minimal_overlay_is_a_few_numbers_and_a_dot() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let mut sink = RecSink::default();
    let s = settings(|l| l.even_out = false);
    run(&mut p, &mut game(clock.clone(), |i| i % 4 != 3, 600), &mut sink, &s);
    let last = sink.texts.last().unwrap().clone().unwrap();
    assert!(last.minimal && !last.text.contains('\n')); // the default style; the advice is a message, not a line
    let texts: Vec<&str> = last.parts.iter().map(|(t, _)| t.as_str()).collect();
    assert!(texts[0].ends_with(" fps") && texts[1] == "game 45" && texts[2].starts_with("delay "));
    assert_eq!(last.parts[1].1, "warn"); // 45 fps isn't what frame generation is for: amber
    assert_eq!(last.health, "good"); // the stand-in interpolator takes no time
    assert_eq!(sink.toasts, [rate_advice(Some(45.0), false, false)]);
    // (the drawn size of the box is sp-present's)
}

#[test]
fn test_a_45_fps_game_is_evened_out() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let mut sink = RecSink::default();
    let s = settings(|l| l.overlay_style = OverlayStyle::Detailed);
    let st = run(&mut p, &mut game(clock.clone(), |i| i % 4 != 3, 600), &mut sink, &s);
    let last = text(sink.texts.last().unwrap());
    assert!(last.starts_with("RIFE  INTERP ON  game 45 fps evened out, showing ") && !last.contains('\n'), "{last}");
    assert!(sink.toasts.len() == 1 && sink.toasts[0].contains("spacing its frames evenly"));
    let fps: Vec<f64> = st.uneven_fps.iter().map(|&x| x as f64).collect();
    assert!(st.uneven_ticks > 400 && sp_sched::np::median(&fps) == 45.0 && st.uneven_waits == 0);
    assert!(st.actions.get("interp") > 300 && st.summary().contains("even timing"));
    assert!(st.results(None).iter().any(|l| l.contains("evened out")));
    assert!(rate_advice(Some(45.0), false, true).is_empty() && !rate_advice(Some(58.0), false, true).is_empty());
}

#[test]
fn test_a_30_fps_game_gets_no_advice() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let mut sink = RecSink::default();
    let s = settings(|l| l.overlay_style = OverlayStyle::Detailed);
    run(&mut p, &mut game(clock.clone(), |i| i % 2 == 0, 600), &mut sink, &s);
    let last = text(sink.texts.last().unwrap());
    assert!(last.contains("game 30 fps") && !last.contains('\n') && sink.toasts.is_empty());
    let mut sink = RecSink::default(); // the minimal overlay: the game's rate in plain grey
    let s = settings(|_| {});
    run(&mut p, &mut game(clock.clone(), |i| i % 2 == 0, 600), &mut sink, &s);
    let last = sink.texts.last().unwrap().clone().unwrap();
    assert!(last.parts.iter().any(|(t, k)| t == "game 30" && k == "dim") && sink.toasts.is_empty());
}

fn tm_tick(i: u64, new: Option<bool>) -> Tick {
    tag_tick(i as f64, 0.0, i, new)
}

#[test]
fn test_test_mode_notes_the_games_own_repeats_and_scores_only_fair_frames() {
    let s = settings(|l| l.simulate_30 = true);
    let mk = || TestMode::new(FnSource::new(|| None), s.clone());
    // a 60 fps game: every frame new, so every left-out frame can be scored
    let mut tm = mk();
    let ticks: Vec<Tick> = (0..8).map(|i| tm.apply(tm_tick(i, Some(true))).unwrap()).collect();
    assert!(ticks.iter().all(|t| t.extra["game_new"] == json!(true)));
    assert!(tm.gt(5).is_some());
    // a 30 fps game (A A B B): the frame left out is just the kept one again
    let mut tm = mk();
    let ticks: Vec<Tick> = (0..8).map(|i| tm.apply(tm_tick(i, Some(i % 2 == 0))).unwrap()).collect();
    let game: Vec<Value> = ticks.iter().map(|t| t.extra["game_new"].clone()).collect();
    assert_eq!(game, [true, false, true, false, true, false, true, false].map(Value::Bool));
    assert!([1, 3, 5].iter().all(|&i| tm.gt(i).is_none()));
    // the other phase: the left-out frame is new, but the kept ones around it are repeats
    let mut tm = mk();
    (0..8).for_each(|i| drop(tm.apply(tm_tick(i, Some(i % 2 == 1)))));
    assert!([1, 3, 5].iter().all(|&i| tm.gt(i).is_none()));
    // no word from the source (it compares pictures): scored as before
    let mut tm = mk();
    (0..8).for_each(|i| drop(tm.apply(tm_tick(i, None))));
    assert!(tm.gt(5).is_some());
}

#[test]
fn test_test_mode_on_a_30_fps_game_turns_itself_off() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let s = settings(|l| {
        l.overlay_style = OverlayStyle::Detailed;
        l.simulate_30 = true;
    });
    let mut tm = TestMode::new(game(clock.clone(), |i| i % 2 == 0, 600), s.clone());
    let mut sink = RecSink::default();
    let st = run(&mut p, &mut tm, &mut sink, &s);
    assert!(!s.live().simulate_30 && !tm.on);
    assert_eq!(sink.toasts, [simulate_off(30)]);
    let last = text(sink.texts.last().unwrap());
    assert!(!last.contains("TEST") && last.contains("game 30 fps") && !last.contains('\n'), "{last}");
    assert!(st.psnr_interp.is_empty()); // nothing fair to score against
    assert!(0 < st.test_game_new && st.test_game_new < 300 && st.actions.get("interp") > 100);
}

#[test]
fn test_test_mode_stays_on_through_a_60_fps_games_30_fps_scene() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let s = settings(|l| l.simulate_30 = true);
    let mut tm = TestMode::new(game(clock.clone(), |i| i < 300 || i % 2 == 0, 900), s.clone());
    let mut sink = RecSink::default();
    run(&mut p, &mut tm, &mut sink, &s);
    assert!(s.live().simulate_30 && tm.on);
    assert!(sink.toasts.len() == 1 && sink.toasts[0].starts_with("Simulate") && !sink.toasts[0].contains("turned off"));
}

#[test]
fn test_a_still_picture_doesnt_turn_test_mode_off() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    p.tuning = no_waits();
    let s = settings(|l| l.simulate_30 = true);
    let mut tm = TestMode::new(game(clock.clone(), |i| i % 40 == 0, 600), s.clone());
    run(&mut p, &mut tm, &mut RecSink::default(), &s);
    assert!(s.live().simulate_30);
}

#[test]
fn test_a_session_logs_its_new_frames_arrival() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let st = run(&mut p, &mut game(clock.clone(), |i| i % 2 == 0, 600), &mut RecSink::default(), &settings(|_| {}));
    assert!(st.new_gaps_ms.len() == 299 && st.summary().contains("game frames as they arrived: 300 new"));
}

// ------------------------------------------------------------------------------------------------ test_timeline.py
fn game_times(fps: f64, ticks: i64) -> BTreeMap<i64, f64> {
    let (mut t, mut arr) = (0.37f64, BTreeMap::new());
    while t < ticks as f64 {
        arr.insert(t.ceil() as i64, t);
        t += 60.0 / fps;
    }
    arr
}

/// Ticks showing each new frame's game time, arriving as they're taken.
fn stream(clock: Arc<FakeClock>, arr: BTreeMap<i64, f64>, ticks: i64) -> impl sp_core::Source {
    let (mut n, mut shown) = (0i64, 0.0);
    FnSource::new(move || {
        if n >= ticks {
            return None;
        }
        let new = arr.contains_key(&n);
        if new {
            shown = arr[&n];
        }
        let t = tag_tick(shown, clock.now(), n as u64, Some(new));
        n += 1;
        Some(Event::Tick(t))
    })
}

fn diff(xs: &[f64]) -> Vec<f64> {
    xs.windows(2).map(|w| w[1] - w[0]).collect()
}

#[test]
fn test_the_pipeline_shows_even_motion() {
    for hz in [60u32, 90] {
        let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
        let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
        let mut sink = RecSink::default();
        let ticks = 260;
        let st = run(&mut p, &mut stream(clock.clone(), game_times(45.0, ticks), ticks), &mut sink, &settings(|l| l.hz = hz));
        assert!(st.uneven_ticks > 100 && st.uneven_waits == 0);
        let from = sink.values.len() - (st.uneven_ticks as f64 * hz as f64 / 60.0) as usize + 30;
        let worst = diff(&sink.values[from..]).iter().map(|d| (d - 60.0 / hz as f64).abs()).fold(0.0, f64::max);
        assert!(worst < 0.1, "{hz}: {worst}");
        // and the even timing off: shown as they come, unevenly
        let mut sink2 = RecSink::default();
        let mut p2 = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
        let st2 = run(&mut p2, &mut stream(clock.clone(), game_times(45.0, ticks), ticks), &mut sink2, &settings(|l| l.even_out = false));
        let tail = &sink2.values[sink2.values.len() - 100..];
        assert!(st2.uneven_ticks == 0 && diff(tail).iter().map(|d| (d - 1.0).abs()).fold(0.0, f64::max) > 0.3);
    }
}

// ------------------------------------------------------------------------------------------------ test_120fps.py
const AI: i64 = 200;

/// test_120fps.py's _Interp: an AI frame at t shows AI + round(4 t).
fn quarter_value(tag: &Value) -> i64 {
    match tag.get("ai") {
        Some(ai) => AI + (4.0 * ai[2].as_f64().unwrap()).round() as i64,
        None => tag["tag"].as_f64().unwrap() as i64,
    }
}

/// A 30 fps game on a 60 Hz stream; each tick arrives as it's taken, so none is late.
fn game30(clock: Arc<FakeClock>, n: u64) -> impl sp_core::Source {
    let mut i = 0u64;
    FnSource::new(move || {
        if i >= n {
            return None;
        }
        let t = tag_tick(((i / 2) % 150) as f64, clock.now(), i, Some(i.is_multiple_of(2)));
        i += 1;
        Some(Event::Tick(t))
    })
}

#[derive(Default)]
struct CountReplay {
    hz: Vec<u32>,
    n: usize,
}

impl Replay for CountReplay {
    fn set_output_hz(&mut self, hz: u32) {
        self.hz.push(hz);
    }
    fn frame(&mut self, _: &Frame, _: f64) {
        self.n += 1;
    }
    fn real_frame(&mut self, _: &Frame, _: f64) {}
    fn save(&mut self, _: Option<&str>) {}
}

#[test]
fn test_120_fps_shows_each_quarter_once_and_in_order() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let (mut sink, mut replay) = (RecSink::default(), CountReplay::default());
    let s = settings(|l| {
        l.hz = 120;
        l.even_out = false;
        l.overlay = false;
    });
    let opts = RunOptions {
        label: "RIFE".into(),
        settings: Some(s.as_ref() as &dyn LiveSettings),
        replay: Some(&mut replay),
        ..Default::default()
    };
    let st = p.run(&mut game30(clock.clone(), 120), Some(&mut sink), opts);
    assert!(st.quarter_shown > 40 && st.quarter_skipped == 0);
    let shown: Vec<i64> = sink.tags.iter().map(quarter_value).collect();
    let first = shown.iter().position(|&v| v == AI + 1).unwrap();
    let steady = &shown[first..first + 40];
    for k in (0..36).step_by(4) {
        assert_eq!(steady[k..k + 3], [AI + 1, AI + 2, AI + 3], "{:?}", &steady[k..k + 4]);
        assert!(steady[k + 3] < AI);
    }
    let single = &p.interp.single;
    assert_eq!(single.iter().map(|t| t.to_bits()).collect::<BTreeSet<_>>(), [0.25f32, 0.5, 0.75].map(f32::to_bits).into());
    let n = |t: f32| single.iter().filter(|&&x| x == t).count();
    assert!(n(0.25) == n(0.5) && n(0.5) == n(0.75)); // each once a pair
    assert!(replay.hz == [120] && replay.n == shown.len());
}

#[test]
fn test_90_fps_is_unchanged() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let s = settings(|l| {
        l.hz = 90;
        l.even_out = false;
        l.overlay = false;
    });
    run(&mut p, &mut game30(clock.clone(), 120), &mut RecSink::default(), &s);
    let set: BTreeSet<u32> = p.interp.single.iter().map(|t| t.to_bits()).collect();
    assert_eq!(set, [1.0f32 / 3.0, 2.0 / 3.0].map(f32::to_bits).into());
}

// ------------------------------------------------------------------------------------------------ test_dips.py
/// A 30 fps game that misses about held3 of its frames (each then stays up `hold` ticks), after `steady` frames that
/// it doesn't, with Remote Play's sizes and frame numbers. Python's random, so the same games.
fn dipping_game(clock: Arc<FakeClock>, frames: usize, held3: f64, seed: u64, hold: usize, steady: usize) -> impl sp_core::Source {
    let mut rnd = PyRandom::new(seed);
    let mut ticks = Vec::new();
    let (mut i, mut t) = (0u64, 0u64);
    for f in 0..frames {
        let n = if f >= steady && rnd.random() < held3 { hold } else { 2 };
        for k in 0..n {
            ticks.push((t as f64, i, k == 0));
            i += 1;
        }
        t += n as u64;
    }
    let mut it = ticks.into_iter();
    FnSource::new(move || {
        let (v, i, new) = it.next()?;
        let mut tick = tag_tick(v, clock.now(), i, Some(new));
        tick.extra.insert("size".into(), json!(if new { 9000 } else { 300 }));
        tick.extra.insert("seq".into(), json!(i));
        Some(Event::Tick(tick))
    })
}

fn backward_steps(shown: &[f64]) -> usize {
    shown.windows(2).filter(|w| w[1] < w[0] - 1e-6).count()
}

fn unevenness(shown: &[f64]) -> f64 {
    let steps = diff(shown);
    let m = steps.iter().sum::<f64>() / steps.len() as f64;
    (steps.iter().map(|s| (s - m).powi(2)).sum::<f64>() / steps.len() as f64).sqrt() / m
}

fn play_dips(frames: usize, held3: f64, hold: usize, steady: usize, f: impl FnOnce(&mut Live)) -> (Vec<f64>, Stats) {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let mut sink = RecSink::default();
    let s = settings(|l| {
        l.overlay = false;
        f(l);
    });
    let st = run(&mut p, &mut dipping_game(clock.clone(), frames, held3, 1, hold, steady), &mut sink, &s);
    (sink.values, st)
}

#[test]
fn test_the_30_fps_setting_never_steps_back_when_the_game_dips() {
    for hz in [60, 90, 120] {
        let (shown, _) = play_dips(150, 0.3, 3, 0, |l| {
            l.hz = hz;
            l.even_out = false;
            l.game_fps_30 = true;
        });
        assert!(shown.len() > 300);
        assert_eq!(backward_steps(&shown), 0, "{hz}");
    }
}

#[test]
fn test_auto_never_steps_back_when_the_game_dips() {
    let (shown, _) = play_dips(150, 0.3, 3, 0, |l| {
        l.hz = 90;
        l.even_out = false;
    });
    assert_eq!(backward_steps(&shown), 0);
}

#[test]
fn test_a_steady_30_fps_game_is_unchanged_by_the_30_fps_setting() {
    let shown: Vec<Vec<f64>> = [false, true]
        .iter()
        .map(|&lock| {
            play_dips(150, 0.0, 3, 0, |l| {
                l.hz = 90;
                l.even_out = false;
                l.game_fps_30 = lock;
            })
            .0
        })
        .collect();
    assert_eq!(shown[1][40..], shown[0][40..]); // (after the lock has found the rhythm)
}

#[test]
fn test_even_out_dips_spreads_a_dip_instead_of_freezing() {
    for hz in [60, 90, 120] {
        for hold in [3, 4] {
            let (off, _) = play_dips(300, 0.15, hold, 40, |l| l.hz = hz);
            let (on, st) = play_dips(300, 0.15, hold, 40, |l| {
                l.hz = hz;
                l.even_dips = true;
            });
            let skip = (200 * hz / 60) as usize; // (after the dips have switched it on)
            assert!(st.dip_ticks > 300);
            assert_eq!(backward_steps(&on), 0);
            let (u_on, u_off) = (unevenness(&on[skip..]), unevenness(&off[skip..]));
            assert!(u_on < 0.5 * u_off, "{hz} Hz, hold {hold}: {u_on} vs {u_off}");
        }
    }
}

#[test]
fn test_even_out_dips_works_with_the_30_fps_setting() {
    let (off, _) = play_dips(300, 0.15, 4, 40, |l| {
        l.hz = 90;
        l.game_fps_30 = true;
    });
    let (on, st) = play_dips(300, 0.15, 4, 40, |l| {
        l.hz = 90;
        l.game_fps_30 = true;
        l.even_dips = true;
    });
    assert!(st.dip_ticks > 300 && backward_steps(&on) == 0);
    assert!(unevenness(&on[300..]) < 0.5 * unevenness(&off[300..]));
}

#[test]
fn test_even_out_dips_changes_nothing_while_a_game_holds_30_fps() {
    for lock in [false, true] {
        let (off, _) = play_dips(200, 0.0, 3, 0, |l| {
            l.hz = 90;
            l.game_fps_30 = lock;
        });
        let (on, st) = play_dips(200, 0.0, 3, 0, |l| {
            l.hz = 90;
            l.game_fps_30 = lock;
            l.even_dips = true;
        });
        assert!(on == off && st.dip_ticks == 0);
    }
}

#[test]
fn test_even_out_dips_is_off_by_default() {
    assert!(!Live::default().even_dips); // (that it's marked experimental is sp-settings')
    let (_, st) = play_dips(300, 0.15, 4, 40, |l| l.hz = 90);
    assert_eq!(st.dip_ticks, 0);
}

// ------------------------------------------------------------------------------------------------ test_delaytest.py
/// A 60 Hz stream (sped up 4x) of a still scene whose menu the remote's presses toggle.
fn menu_game(clock: Arc<FakeClock>, remote: Arc<Mutex<RemoteState>>, seconds: f64) -> impl sp_core::Source {
    let t_end = clock.now() + seconds;
    let mut i = 0u64;
    FnSource::new(move || {
        if clock.now() >= t_end {
            return None;
        }
        clock.advance(0.004);
        let now = clock.now();
        let open = remote.lock().unwrap().menu_open(now);
        let f = Frame::from_bgr(&content(&json!({"kind": "menu", "open": open}), 8, 8), 8, 8);
        let mut t = Tick::new(f, now, i);
        t.new_hint = Some(i.is_multiple_of(2));
        i += 1;
        Some(Event::Tick(t))
    })
}

#[test]
fn test_the_pipeline_runs_it_and_keeps_the_sound_in_step() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let ev: Events = Default::default();
    let mut remote = TraceRemote::new(&json!({"answer_s": 0.05}), clock.clone(), ev.clone());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs).with_probe(Box::new(GpuProbe::new()));
    p.tuning = Tuning { delay_settle_s: 0.1, delay_gap_s: 0.1, ..Tuning::default() };
    let mut sink = RecSink { cmds: vec![SinkReply::DelayTest], ..Default::default() };
    let s = settings(|_| {});
    let mut src = menu_game(clock.clone(), remote.st.clone(), 2.5);
    let opts = RunOptions {
        label: "RIFE".into(),
        settings: Some(s.as_ref() as &dyn LiveSettings),
        remote: Some(&mut remote as &mut dyn Remote),
        ..Default::default()
    };
    let st = p.run(&mut src, Some(&mut sink), opts);
    assert!(st.delay_tests.len() == 1 && st.delay_tests[0].contains("PS5 and Remote Play"), "{:?}", st.delay_tests);
    assert!(sink.toasts.iter().any(|t| t.starts_with("PS5 and Remote Play")));
    assert!(sink.texts.iter().any(|t| text(t).contains("DELAY TEST")));
    assert!(st.summary().contains(&st.delay_tests[0]) && st.results(None).join("\n").contains(&st.delay_tests[0]));
    assert!(logs.lock().unwrap().iter().any(|l| l.contains("delay test started")));
    assert!(!remote.st.lock().unwrap().audio.is_empty()); // told chiaki-ng how long to wait (whole ms, by type)
    // Sound in sync off: chiaki-ng is told 0
    let mut remote2 = TraceRemote::new(&json!({}), clock.clone(), ev.clone());
    let s2 = settings(|l| l.audio_sync = false);
    let mut src = menu_game(clock.clone(), remote2.st.clone(), 0.3);
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs).with_probe(Box::new(GpuProbe::new()));
    let opts = RunOptions {
        settings: Some(s2.as_ref() as &dyn LiveSettings),
        remote: Some(&mut remote2 as &mut dyn Remote),
        ..Default::default()
    };
    p.run(&mut src, Some(&mut RecSink::default()), opts);
    assert_eq!(remote2.st.lock().unwrap().audio, [0]);
}

#[test]
fn test_without_remote_play_it_says_so() {
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let remote = TraceRemote::new(&json!({}), clock.clone(), Default::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs).with_probe(Box::new(GpuProbe::new()));
    let mut sink = RecSink { cmds: vec![SinkReply::DelayTest], ..Default::default() };
    run(&mut p, &mut menu_game(clock.clone(), remote.st.clone(), 0.3), &mut sink, &settings(|_| {}));
    assert!(!sink.toasts.is_empty() && sink.toasts[0].contains("works with PS5 Remote Play"));
}

// ------------------------------------------------------------------------------------------------ test_waiting.py, test_watchdog.py
#[test]
fn test_pipeline_quits_during_a_stall() {
    // Q (or closing the window) while no frame comes: the session ends instead of hanging. On the real clock.
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let mut p = Pipeline::new(BlendInterp::new(Arc::new(FakeClock::new(0.0))), None).with_clock(clock).with_probe(Box::new(TagProbe));
    let status = p.status();
    let before = status.beat();
    let mut sink = RecSink { wait_quit_at: Some(3), ..Default::default() };
    let mut stalled = FnSource::new(|| Some(Event::Waiting(Waiting { since: 1.0, what: "the PS5".into(), why: String::new() })));
    let st = p.run(&mut stalled, Some(&mut sink), RunOptions::default());
    assert!(st.exit == Exit::Quit && sink.waits == 3);
    assert!(status.beat() > before); // (the watchdog reads it)
    assert_eq!(status.waiting().unwrap().what, "the PS5");
}

#[test]
fn test_the_watchdogs_beat_moves_while_frames_come_and_waiting_clears() {
    let (clock, logs) = (Arc::new(FakeClock::new(5.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let status = p.status();
    let mut n = 0;
    let c = clock.clone();
    let mut src = FnSource::new(move || {
        n += 1;
        c.advance(0.5);
        match n {
            1 => Some(Event::Waiting(Waiting { since: 0.5, what: "the PS5".into(), why: String::new() })),
            2..=4 => Some(Event::Tick(tag_tick(n as f64, c.now(), n as u64, Some(true)))),
            _ => None,
        }
    });
    p.run(&mut src, Some(&mut RecSink::default()), RunOptions::default());
    assert_eq!(status.beat(), 7.0); // the last tick's
    assert!(status.waiting().is_none()); // frames came again
    assert_eq!(status.stats().proc_other_ms.len() + status.stats().proc_interp_ms.len(), 3); // readable after
}

// ------------------------------------------------------------------------------------------------ test_steady.py
#[test]
fn test_threshold_is_the_shared_one() {
    assert_eq!(STEADY_HUD_THRESHOLD, 0.03);
    // and the pipeline applies it as the setting changes: off at the start, on, off
    let (clock, logs) = (Arc::new(FakeClock::new(1.0)), Logs::default());
    let mut p = pipeline(BlendInterp::new(clock.clone()), &clock, &logs);
    let s = settings(|_| {});
    let s2 = s.clone();
    let mut inner = game(clock.clone(), |i| i % 2 == 0, 30);
    let mut i = 0;
    let mut src = FnSource::new(move || {
        i += 1;
        if i == 10 {
            s2.update(|l| l.steady_hud = true);
        }
        if i == 20 {
            s2.update(|l| l.steady_hud = false);
        }
        sp_core::Source::next_event(&mut inner)
    });
    run(&mut p, &mut src, &mut RecSink::default(), &s);
    assert_eq!(p.interp.steady, [0.0, STEADY_HUD_THRESHOLD, 0.0]);
    let logs = logs.lock().unwrap();
    assert_eq!(logs.iter().filter(|l| l.starts_with("[pipeline] steady HUD")).count(), 3);
}
