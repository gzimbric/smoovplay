//! tests/test_game_rate.py (GameRate, ShownRate, rate_advice, results(), the arrival line) and test_settings.py's
//! results report, ported with the same names and assertions, plus run.py write_results' file handling.

use std::collections::BTreeSet;

use sp_sched::report::{ReportHeader, SessionInfo, results_lines, write_results};
use sp_sched::{GameRate, ShownRate, Stats, rate_advice};

#[test]
fn test_game_rate_counts_new_frames_over_two_seconds_of_ticks() {
    let mut r = GameRate::new(120);
    for i in 0..119 {
        r.add(Some(i % 2 == 0));
    }
    assert_eq!(r.fps(), None); // not a whole window yet
    r.add(Some(false));
    assert_eq!(r.fps(), Some(30.0));
    r.add(None); // the source can't say, so it's left out, not counted as a repeat
    assert_eq!(r.fps(), Some(30.0));
    for i in 0..120 {
        r.add(Some(i % 4 != 3)); // three new frames in every four ticks
    }
    assert_eq!(r.fps(), Some(45.0));
    r.clear();
    assert_eq!(r.fps(), None);
}

#[test]
fn test_shown_rate_is_frames_on_screen_in_the_last_second() {
    let mut r = ShownRate::new(1.0);
    for i in 0..91 {
        r.add(i as f64 / 90.0);
    }
    assert_eq!(r.fps(), Some(90.0));
    for i in 1..61 {
        r.add(1.0 + i as f64 / 60.0); // down to 60 a second
    }
    assert_eq!(r.fps(), Some(60.0));
    assert_eq!(ShownRate::new(1.0).fps(), None);
}

#[test]
fn test_rate_advice() {
    assert!(rate_advice(None, false, false).is_empty() && rate_advice(None, true, false).is_empty());
    assert!(rate_advice(Some(30.5), false, false).is_empty() && rate_advice(Some(34.0), false, false).is_empty());
    let a46 = rate_advice(Some(46.0), false, false);
    assert!(a46.contains("about 46 fps, not 30") && a46.contains("30 fps mode"));
    assert!(rate_advice(Some(59.6), false, false).contains("Game at 60 fps: no gaps to fill"));
    assert!(rate_advice(Some(59.0), true, false).is_empty()); // test mode with a 60 fps game is as it should be
    assert!(rate_advice(Some(30.0), true, false).contains("at 30 fps already: turn Simulate off"));
    assert!(rate_advice(Some(44.0), true, false).contains("about 44, so quality isn't measured"));
    for g in [30.0, 44.0, 46.0, 60.0] {
        assert!(rate_advice(Some(g), false, false).len() <= 100 && rate_advice(Some(g), true, false).len() <= 100);
    }
}

fn last(st: &Stats) -> String {
    st.results(None).last().unwrap().clone()
}

#[test]
fn test_results_say_when_test_mode_had_no_60_fps_game_and_judge_by_extra_frames() {
    let mut st = Stats {
        proc_interp_ms: vec![12.0; 100],
        busy_ms: vec![9.0; 1000],
        real_delay_ms: vec![30.0; 100],
        hz_used: BTreeSet::from([90]),
        quarter_shown: 96,
        quarter_skipped: 4,
        behind: 177,
        simulated: 1000,
        test_game_new: 500,
        test_game_repeat: 500,
        ..Default::default()
    };
    let text = st.results(None).join("\n");
    assert!(text.contains("about 30 fps, not a steady 60") && text.contains("doesn't need it"));
    assert!(text.contains("Quality: not measured, as the game wasn't at a steady 60 fps"));
    // 18% of ticks came in while busy, but 96% of the extra frames were shown, and that's keeping up at 90
    assert_eq!(last(&st), "Verdict: this PC keeps up at 90 fps");
    (st.quarter_shown, st.quarter_skipped) = (85, 15);
    assert!(last(&st).contains("mostly keeps up at 90 fps"));
    (st.quarter_shown, st.quarter_skipped) = (70, 30);
    assert!(last(&st).contains("falling behind at 90 fps"));
    // a steady 60 fps game in test mode gets no note and scores as usual
    let st = Stats {
        proc_interp_ms: vec![8.0; 10],
        busy_ms: vec![9.0; 10],
        psnr_interp: vec![34.0; 10],
        psnr_repeat: vec![26.0; 10],
        psnr_blend: vec![30.0; 10],
        off_interp: vec![0.05; 10],
        off_repeat: vec![0.2; 10],
        hz_used: BTreeSet::from([60]),
        test_game_new: 1990,
        test_game_repeat: 10,
        ..Default::default()
    };
    let text = st.results(None).join("\n");
    assert!(!text.contains("Game:") && text.contains("5.0% of each AI frame was visibly off"));
    assert!(text.contains("(10 frames measured, worst 5.0%)"));
    assert!(st.summary().contains("10 of 2000 frames were repeats"));
    // 90 fps output with a game at about 37 fps, where 44 frames from 60 fps fallbacks are too few to say anything
    let mut st = Stats {
        proc_interp_ms: vec![18.0; 100],
        busy_ms: vec![9.0; 100],
        psnr_interp: vec![40.0; 44],
        psnr_repeat: vec![60.0; 44],
        psnr_blend: vec![50.0; 44],
        off_interp: vec![0.034; 44],
        off_repeat: vec![0.0; 44],
        hz_used: BTreeSet::from([90]),
        quarter_shown: 99,
        quarter_skipped: 1,
        test_game_new: 4590,
        test_game_repeat: 2760,
        ..Default::default()
    };
    let text = st.results(None).join("\n");
    assert!(text.contains("Quality: not measured, as the game wasn't at a steady 60 fps") && !text.contains("visibly off"));
    (st.test_game_new, st.test_game_repeat) = (7340, 10); // a steady 60, still only 90 fps output
    assert!(st.results(None).join("\n").contains("Quality: not measured yet"));
}

#[test]
fn test_the_log_says_how_evenly_the_games_frames_arrived() {
    let mut st = Stats::default();
    assert!(st.arrival_line().is_empty() && !st.summary().contains("game frames"));
    st.new_gaps_ms.extend(std::iter::repeat_n(33.3, 90).chain(std::iter::repeat_n(66.7, 6)).chain(std::iter::repeat_n(2.0, 4)));
    let line = st.arrival_line();
    assert_eq!(
        line,
        "game frames as they arrived: 101 new, usually 33.3 ms apart; 6.0% late (over 1.5x that), 4.0% bunched \
         (under half); longest gap 67 ms"
    );
    assert!(st.summary().contains(&line) && Stats { private: true, ..st.clone() }.summary().contains(&line));
}

// ------------------------------------------------------------------------------------------------ test_settings.py
#[test]
fn test_results_report_reads_plainly() {
    let mut st = Stats {
        proc_interp_ms: vec![8.0; 100],
        busy_ms: vec![9.0; 200],
        real_delay_ms: vec![20.0; 100],
        psnr_interp: [vec![34.0; 50], vec![f64::INFINITY]].concat(),
        psnr_repeat: [vec![26.0; 50], vec![f64::INFINITY]].concat(),
        off_interp: vec![0.062; 50],
        off_repeat: vec![0.12; 50],
        hz_used: BTreeSet::from([60, 120]),
        quarter_shown: 99,
        quarter_skipped: 1,
        simulated: 200,
        ..Default::default()
    };
    let text = st.results(None).join("\n");
    assert!(text.contains("Output: 60 fps and 120 fps"));
    // Quality in plain terms (the share of each frame visibly off), not dB. The dB stay in the log.
    assert!(text.contains("6.2% of each AI frame was visibly off") && text.contains("against 12% when the last frame is repeated"));
    assert!(text.contains("50 frames measured") && !text.contains(" dB"));
    assert!(st.summary().contains("PSNR") && st.summary().contains("visibly off (more than 12 levels): RIFE 6.20% | repeat 12.00%"));
    assert!(text.contains("Verdict: this PC keeps up at 120 fps"));
    st.behind = 50; // a quarter of the frames waited, but nearly all extra frames were shown, so it's still keeping up
    assert!(last(&st).contains("keeps up at 120 fps"));
    (st.quarter_shown, st.quarter_skipped) = (60, 40);
    assert!(last(&st).contains("falling behind at 120 fps"));
    // at 60 fps, frames that waited say nothing, but frames that took longer to make than a frame lasts do
    (st.hz_used, st.quarter_shown, st.quarter_skipped) = (BTreeSet::from([60]), 0, 0);
    assert_eq!(last(&st), "Verdict: this PC keeps up at 60 fps");
    assert!(!st.results(None).join("\n").contains("Running behind"));
    st.proc_interp_ms = [vec![8.0; 90], vec![20.0; 10]].concat();
    assert!(st.results(None).join("\n").contains("Slow frames: 10.0% (10 of 100) took longer to make than a 60 fps frame lasts"));
    assert!(last(&st).contains("falling behind at 60 fps"));
    (st.proc_interp_ms, st.proc_other_ms) = ([vec![8.0; 97], vec![20.0; 3]].concat(), vec![1.0; 100]); // 1.5%
    assert!(last(&st).contains("mostly keeps up at 60 fps"));
    (st.proc_interp_ms, st.proc_other_ms) = (vec![8.0; 100], vec![]);
    (st.hz_used, st.quarter_shown, st.quarter_skipped) = (BTreeSet::from([90]), 0, 0);
    (st.psnr_interp, st.off_interp, st.off_repeat) = (vec![], vec![], vec![]);
    assert!(st.results(None).iter().any(|l| l.contains("not measured yet")));
}

// ------------------------------------------------------------------------------------------------ run.py write_results
#[test]
fn write_results_writes_whole_files_and_skips_untested_sessions() {
    let dir = std::env::temp_dir().join(format!("sp-sched-results-{}", std::process::id()));
    let path = dir.join("results-20261122-120000.txt");
    let header = ReportHeader { stamp: "2026-11-22 12:00".into(), size: "1920x1080".into(), gpu: "Apple M5 Max".into(), upscaler: "AMD FSR 1".into() };
    let info = SessionInfo { engine: "metal".into(), model: "4.22.lite".into(), motion: 1.0, vfg_quality: String::new() };
    let untested = Stats { busy_ms: vec![9.0; 100], ..Default::default() };
    assert!(!write_results(&path, &header, &[(info.clone(), &untested)]) && !path.exists());
    let tested = Stats { busy_ms: vec![9.0; 3600], simulated: 1800, proc_interp_ms: vec![8.0; 10], ..Default::default() };
    assert!(write_results(&path, &header, &[(info.clone(), &tested)]));
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("SmoovPlay test results, 2026-11-22 12:00\nStream: 1920x1080, played 1:00, 0:30 of it as a simulated 30 fps game\n"));
    assert!(text.contains("\nAI engine: Metal, motion detail Full, RIFE 4.22.lite\n") && text.ends_with('\n'));
    assert!(!dir.join("results-20261122-120000.txt.tmp").exists());
    let two = results_lines(&header, &[(info.clone(), &tested), (SessionInfo { motion: 0.5, ..info }, &tested)]).unwrap();
    assert!(two.contains(&"Part 2: Metal, motion detail Half, RIFE 4.22.lite".to_string()));
    std::fs::remove_dir_all(&dir).ok();
}
