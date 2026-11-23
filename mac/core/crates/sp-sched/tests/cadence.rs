//! tests/test_cadence.py, test_rhythm_lock.py, test_timeline.py (the timeline on its own) and test_dips.py
//! (DipWatch and the timeline timed as frames arrived), ported: same names, same assertions.

mod common;

use std::collections::BTreeMap;

use common::pyrandom::PyRandom;
use sp_core::Frame;
use sp_core::stats::BlockMeans;
use sp_sched::cadence::cut_features;
use sp_sched::{Action, CadenceScheduler, CutDetector, CutFeatures, DipWatch, DuplicateDetector, FrameTimeline, GpuProbe, Probe, RhythmLock};

use Action::{Current as C, Hold as H, Interp as I, Pending as P};

// ------------------------------------------------------------------------------------------------ test_cadence.py
fn run(pattern: &str, cuts: &[usize]) -> Vec<Action> {
    let mut s = CadenceScheduler::new();
    pattern.chars().enumerate().map(|(i, ch)| s.step(ch == 'N', cuts.contains(&i))).collect()
}

#[test]
fn test_steady_30fps_alternates_interp_and_real() {
    assert_eq!(run("NdNdNdNd", &[]), [C, C, I, C, I, C, I, C]);
}

#[test]
fn test_hitch_repeats_extra_duplicate() {
    assert_eq!(run("NddNdNd", &[]), [C, C, C, I, C, I, C]);
}

#[test]
fn test_60fps_section_passes_through_one_tick_late() {
    assert_eq!(run("NdNNNNdNd", &[]), [C, C, I, P, P, P, C, I, C]);
}

#[test]
fn test_scene_cut_holds_instead_of_interpolating() {
    assert_eq!(run("NdNdNd", &[2]), [C, C, H, C, I, C]);
}

/// A small xorshift for pictures (the Python uses torch's generator; only the kind of picture matters).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn means(bgr: &[u8], w: usize, h: usize) -> BlockMeans {
    GpuProbe::new().block_means(&Frame::from_bgr(bgr, w, h))
}

#[test]
fn test_detector_ignores_exact_and_noisy_repeats_but_sees_small_motion() {
    let (w, h) = (1920, 1080);
    let mut g = Rng(0x9E3779B97F4A7C15);
    let base: Vec<u8> = (0..w * h * 3).map(|_| g.byte()).collect();
    let mut det = DuplicateDetector::new(2.0, sp_sched::Reference::Previous, false);
    assert!(det.update(&means(&base, w, h)).is_new); // first frame
    assert!(!det.update(&means(&base.clone(), w, h)).is_new); // bit-exact repeat
    let noisy: Vec<u8> = base.iter().map(|&v| (v as i32 + (g.next() % 7) as i32 - 3).clamp(0, 255) as u8).collect();
    assert!(!det.update(&means(&noisy, w, h)).is_new); // +-3 per-pixel noise averages out in 16x16 blocks
    let mut moved = noisy.clone();
    for y in 500..540 {
        for x in 900..940 {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                moved[i] = 255 - moved[i]; // one small object changes
            }
        }
    }
    let r = det.update(&means(&moved, w, h));
    assert!(r.is_new && r.mean_diff < 1.0); // global mean barely moves; block max catches it
}

#[test]
fn test_cut_detector_flags_spikes_not_sustained_fast_motion() {
    let mut d = CutDetector::new(25.0, true);
    let seq = [8.0, 9.0, 7.0, 60.0, 10.0, 9.0, 28.0, 33.0, 35.0, 34.0, 32.0];
    let flags: Vec<bool> = seq.iter().map(|&m| d.update(m, None)).collect();
    assert_eq!(flags, [false, false, false, true, false, false, false, false, false, false, false]);
}

/// A dark 1080p frame (brightness ~5-40) with its own layout of light and shadow: 9x16 random blobs, bilinear
/// (torch's interpolate, align_corners=False).
fn dark_scene(seed: u64) -> Vec<u8> {
    let (w, h, bw, bh) = (1920usize, 1080usize, 16usize, 9usize);
    let mut g = Rng(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1);
    let blobs: Vec<f32> = (0..bw * bh).map(|_| g.unit()).collect();
    let src = |d: usize, n_in: usize, n_out: usize| -> (usize, usize, f32) {
        let s = ((d as f32 + 0.5) * (n_in as f32 / n_out as f32) - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(n_in - 1);
        (i0, (i0 + 1).min(n_in - 1), s - i0 as f32)
    };
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let (y0, y1, ly) = src(y, bh, h);
        for x in 0..w {
            let (x0, x1, lx) = src(x, bw, w);
            let top = blobs[y0 * bw + x0] * (1.0 - lx) + blobs[y0 * bw + x1] * lx;
            let bot = blobs[y1 * bw + x0] * (1.0 - lx) + blobs[y1 * bw + x1] * lx;
            let v = (5.0 + 35.0 * (top * (1.0 - ly) + bot * ly)) as u8;
            out[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(v);
        }
    }
    out
}

#[test]
fn test_cut_between_dark_shots_is_caught_by_structure() {
    let (a, b) = (dark_scene(1), dark_scene(2));
    let mut det = DuplicateDetector::default();
    det.update(&means(&a, 1920, 1080));
    let same = det.update(&means(&a.clone(), 1920, 1080));
    let cut = det.update(&means(&b, 1920, 1080));
    assert!(cut.mean_diff < 25.0); // too small for the plain-difference test on its own
    assert!(cut.cut.unwrap().corr_dist > 0.75 && 0.75 > same.cut.unwrap().corr_dist);
    assert!(!CutDetector::new(25.0, false).update(cut.mean_diff, cut.cut.as_ref()));
    assert!(CutDetector::default().update(cut.mean_diff, cut.cut.as_ref()));
}

#[test]
fn test_structure_needs_a_jump_and_a_real_change() {
    let mut d = CutDetector::default();
    let noise = CutFeatures { corr_dist: 1.2, hist_dist: 0.1, luma: 3.0, contrast: 0.5 }; // near-black frame noise
    assert!(!d.update(2.0, Some(&noise))); // layout "changes" completely, but the picture barely changed
    let mut d = CutDetector::default();
    let moving: Vec<CutFeatures> =
        [0.9, 0.95, 0.85, 0.9].iter().map(|&c| CutFeatures { corr_dist: c, hist_dist: 0.1, luma: 40.0, contrast: 20.0 }).collect();
    let flags: Vec<bool> = moving.iter().map(|f| d.update(12.0, Some(f))).collect();
    assert_eq!(flags, [true, false, false, false]); // only the start can look like a cut
}

#[test]
fn test_flat_frames_are_alike_and_flat_to_detail_is_not() {
    let (rows, cols) = (68, 120);
    let black = BlockMeans { rows, cols, data: vec![0.0; rows * cols * 3] };
    let mut g = Rng(3);
    let detail = BlockMeans { rows, cols, data: (0..rows * cols * 3).map(|_| g.unit() * 255.0).collect() };
    assert_eq!(cut_features(&black, &black.clone()).corr_dist, 0.0);
    assert_eq!(cut_features(&black, &detail).corr_dist, 1.0);
}

// ------------------------------------------------------------------------------------------------ test_rhythm_lock.py
/// A 30 fps game's compressed frame sizes at 60 Hz, as tiny as in a still scene (0.6 KB new, 0.2 KB repeats).
fn sizes(n: usize, phase: usize, seed: u64) -> Vec<i64> {
    let mut rnd = PyRandom::new(seed);
    (0..n).map(|i| ((if i % 2 == phase { 650.0 } else { 200.0 }) * rnd.uniform(0.75, 1.25)) as i64).collect()
}

#[test]
fn test_tiny_frames_that_fool_the_size_line_still_alternate() {
    let mut lock = RhythmLock::new();
    let mut rnd = PyRandom::new(2);
    let sz = sizes(600, 0, 1);
    // the size check's own calls, with 40% of repeats misread as new
    let calls: Vec<bool> = (0..600).map(|i| i % 2 == 0 || rnd.random() < 0.4).collect();
    let out: Vec<bool> = calls.iter().zip(&sz).map(|(&c, &s)| lock.update(Some(c), Some(s), false, None)).collect();
    assert_eq!(out[40..], (40..600).map(|i| i % 2 == 0).collect::<Vec<_>>()[..]);
    assert_eq!(lock.moves, 0);
    assert_eq!(out[40..].iter().filter(|&&x| x).count(), 280); // exactly 30 fps
}

#[test]
fn test_a_hitch_moves_the_rhythm_by_a_tick() {
    let mut lock = RhythmLock::new();
    let mut sz = sizes(200, 0, 1);
    sz.extend_from_slice(&sizes(300, 1, 3)[200..]); // a frame held for 3 ticks at 200
    let out: Vec<bool> = sz.iter().map(|&s| lock.update(None, Some(s), false, None)).collect();
    assert_eq!(out[150..200], (150..200).map(|i| i % 2 == 0).collect::<Vec<_>>()[..]);
    assert_eq!(out[260..], (260..300).map(|i| i % 2 == 1).collect::<Vec<_>>()[..]);
    assert_eq!(lock.moves, 1);
}

#[test]
fn test_without_sizes_the_checks_calls_vote() {
    let mut lock = RhythmLock::new();
    let mut rnd = PyRandom::new(4);
    let calls: Vec<bool> = (0..300).map(|i| if rnd.random() > 0.2 { i % 2 == 1 } else { i % 2 == 0 }).collect();
    let out: Vec<bool> = calls.iter().map(|&c| lock.update(Some(c), None, false, None)).collect();
    assert_eq!(out[40..], (40..300).map(|i| i % 2 == 1).collect::<Vec<_>>()[..]);
}

#[test]
fn test_keyframes_say_nothing() {
    let mut lock = RhythmLock::new();
    let sz = sizes(100, 0, 1);
    let out: Vec<bool> = sz
        .iter()
        .enumerate()
        .map(|(i, &s)| lock.update(None, Some(if i % 20 == 7 { 50_000 } else { s }), i % 20 == 7, None))
        .collect();
    assert_eq!(out[30..], (30..100).map(|i| i % 2 == 0).collect::<Vec<_>>()[..]);
}

#[test]
fn test_the_first_ticks_go_by_the_checks() {
    let mut lock = RhythmLock::new();
    let out: Vec<bool> = [true, true, false].iter().map(|&c| lock.update(Some(c), None, false, None)).collect();
    assert_eq!(out, [true, true, false]);
    assert_eq!(lock.phase, None);
}

#[test]
fn test_lost_frames_dont_move_the_rhythm_when_counted_by_the_streams_numbers() {
    let mut lock = RhythmLock::new();
    let sz = sizes(400, 0, 1);
    let keep: Vec<usize> = (0..400).filter(|i| ![101, 151, 152, 153, 251].contains(i)).collect();
    let out: Vec<(usize, bool)> = keep.iter().map(|&i| (i, lock.update(None, Some(sz[i]), false, Some(i as i64)))).collect();
    assert!(out.iter().filter(|(i, _)| *i >= 40).all(|&(i, new)| new == (i % 2 == 0)));
    assert_eq!(lock.moves, 0);
}

#[test]
fn test_a_capture_cards_frames_dropped_from_the_queue_dont_move_the_rhythm() {
    let mut rnd = PyRandom::new(7);
    let calls: Vec<bool> = (0..600).map(|i| if rnd.random() > 0.15 { i % 2 == 0 } else { i % 2 == 1 }).collect();
    let keep: Vec<usize> = (0..600).filter(|i| ![100, 230, 231, 377, 501].contains(i)).collect();
    let mut lock = RhythmLock::new();
    let out: Vec<(usize, bool)> = keep.iter().map(|&i| (i, lock.update(Some(calls[i]), None, false, Some(i as i64)))).collect();
    assert!(out.iter().filter(|(i, _)| *i >= 40).all(|&(i, new)| new == (i % 2 == 0)));
    assert_eq!(lock.moves, 0);
    let mut as_they_came = RhythmLock::new();
    for &i in &keep {
        as_they_came.update(Some(calls[i]), None, false, None);
    }
    assert!(as_they_came.moves >= 2); // what it did before
}

// ------------------------------------------------------------------------------------------------ test_timeline.py
/// {arrival tick: the time (in ticks) the game drew the frame shown from it}.
fn game_times(fps: f64, ticks: i64, phase: f64, jitter: f64, seed: u64) -> BTreeMap<i64, f64> {
    let mut rng = PyRandom::new(seed + 100);
    let (mut t, mut arr) = (phase, BTreeMap::new());
    while t < ticks as f64 {
        arr.insert(t.ceil() as i64, t); // a later frame done in the same tick replaces it
        t += 60.0 / fps * (1.0 + jitter * rng.normal());
    }
    arr
}

/// Where in the game's time each output frame is (linear motion), and how many waited for a late frame.
fn play(arr: &BTreeMap<i64, f64>, ticks: i64, hz: u32) -> (Vec<f64>, u32, FrameTimeline<f64>) {
    let mut tl = FrameTimeline::new(12);
    let (mut pos, mut waits) = (Vec::new(), 0);
    for n in 0..ticks {
        tl.tick(arr.contains_key(&n), false, arr.get(&n).copied(), n as f64);
        if tl.uneven {
            for phi in tl.slots(hz) {
                let p = tl.place(phi).unwrap();
                waits += p.waiting as u32;
                let a = p.a.frame.unwrap();
                pos.push(match &p.b {
                    None => a,
                    Some(b) => a + p.t * (b.frame.unwrap() - a),
                });
            }
        }
    }
    (pos, waits, tl)
}

fn diff(xs: &[f64]) -> Vec<f64> {
    xs.windows(2).map(|w| w[1] - w[0]).collect()
}

fn std(xs: &[f64]) -> f64 {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

#[test]
fn test_motion_is_even_at_any_rate_in_between() {
    for fps in [40.0, 45.0, 48.0, 50.0, 55.0] {
        for hz in [60, 90, 120] {
            let (pos, waits, tl) = play(&game_times(fps, 900, 0.37, 0.0, 0), 900, hz);
            let steps = diff(&pos[60..]);
            assert!(tl.uneven && waits == 0, "{fps} {hz}");
            let worst = steps.iter().map(|s| (s - 60.0 / hz as f64).abs()).fold(0.0, f64::max);
            assert!(worst < 0.1, "{fps} fps at {hz}: {worst}"); // shown as they come, a step is off by a whole tick
            assert!((1.0..=1.6).contains(&tl.delay), "{fps} {hz}: delay {}", tl.delay);
        }
    }
}

#[test]
fn test_30_and_60_fps_games_are_left_to_the_cadence_scheduler() {
    for fps in [30.0, 60.0] {
        let (_, _, tl) = play(&game_times(fps, 600, 0.0, 0.0, 0), 600, 60);
        assert!(!tl.uneven);
    }
}

#[test]
fn test_a_menu_at_60_in_a_30_fps_game_doesnt_switch_it_on() {
    let new = |n: i64| (n < 300 && n % 2 == 0) || (300..330).contains(&n) || (n >= 330 && n % 2 == 0);
    let mut tl: FrameTimeline<f64> = FrameTimeline::new(12);
    let mut on = 0;
    for n in 0..600 {
        tl.tick(new(n), false, None, 0.0);
        on += tl.uneven as u32;
    }
    assert_eq!(on, 0);
}

#[test]
fn test_a_jittery_game_and_a_changing_rate_stay_close() {
    for (fps, jitter) in [(45.0, 0.05), (50.0, 0.1)] {
        let (pos, waits, _) = play(&game_times(fps, 900, 0.37, jitter, 0), 900, 90);
        let d: Vec<f64> = diff(&pos[60..]).iter().map(|s| s - 2.0 / 3.0).collect();
        assert!(std(&d) < 0.1 && waits < 30, "{fps}: std {} waits {waits}", std(&d));
    }
    let (mut t, mut arr) = (0.3f64, BTreeMap::new());
    while t < 900.0 {
        arr.insert(t.ceil() as i64, t);
        t += 60.0 / (45.0 + 10.0 * t / 900.0); // 45 to 55 fps
    }
    let (pos, waits, _) = play(&arr, 900, 120);
    let d: Vec<f64> = diff(&pos[60..]).iter().map(|s| s - 0.5).collect();
    assert!(std(&d) < 0.05 && waits < 20);
}

#[test]
fn test_a_hitch_waits_then_carries_on() {
    let mut arr = game_times(45.0, 600, 0.37, 0.0, 0);
    arr.retain(|&n, _| !(300..306).contains(&n)); // the game stops for 6 ticks
    let (pos, waits, tl) = play(&arr, 600, 60);
    assert!((1..=8).contains(&waits) && tl.uneven, "waits {waits}");
    assert!(diff(&pos).iter().all(|&d| d >= -1e-9)); // never back in time
    let tail = &pos[pos.len() - 120..];
    assert!(diff(tail).iter().all(|d| (d - 1.0).abs() < 0.1)); // and even again after
}

#[test]
fn test_no_blend_across_a_cut() {
    let mut tl = FrameTimeline::new(12);
    let arr = game_times(45.0, 300, 0.37, 0.0, 0);
    let mut blended = Vec::new();
    for n in 0..300 {
        let cut = n == 200 && arr.contains_key(&n);
        tl.tick(arr.contains_key(&n), cut || (n == 201 && !arr.contains_key(&200)), Some(n as f64), 0.0);
        if tl.uneven {
            for phi in tl.slots(60) {
                let p = tl.place(phi).unwrap();
                if p.b.as_ref().is_some_and(|b| b.cut) {
                    blended.push(n);
                }
            }
        }
    }
    assert!(blended.is_empty());
}

#[test]
fn test_slots_follow_the_output_rate() {
    let mut tl: FrameTimeline<f64> = FrameTimeline::new(12);
    let mut counts: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for hz in [60, 90, 120] {
        for _ in 0..12 {
            tl.tick(true, false, None, 0.0);
            counts.entry(hz).or_default().push(tl.slots(hz).len());
        }
    }
    assert_eq!(counts[&60][1..], [1; 11]);
    assert!([16, 17].contains(&counts[&90][1..].iter().sum::<usize>()));
    assert_eq!(counts[&120][1..], [2; 11]);
}

// ------------------------------------------------------------------------------------------------ test_dips.py
#[test]
fn test_dip_watch_switches_on_while_the_game_dips_and_off_after_it_stops() {
    let mut w = DipWatch::new();
    let calls: Vec<bool> = (0..60).flat_map(|_| [true, false]).collect(); // 2 s at a steady 30 fps
    assert!(!calls.iter().any(|&c| w.update(c)));
    let dip: Vec<bool> = [true, false, false, false].into_iter().chain([true, false].repeat(4)).collect();
    for &c in &dip {
        w.update(c);
    }
    assert!(!w.dipping); // one dip isn't enough
    for &c in &dip {
        w.update(c);
    }
    assert!(w.dipping); // two within a second
    for _ in 0..85 {
        w.update(true);
        w.update(false); // 2.9 s steady again: still on (the dips may come back)
    }
    assert!(w.dipping);
    for _ in 0..3 {
        w.update(true);
        w.update(false); // 3 s since the last dip
    }
    assert!(!w.dipping);
}

#[test]
fn test_dip_watch_leaves_pauses_and_fast_games_alone() {
    let mut w = DipWatch::new();
    let pauses: Vec<bool> = (0..20).flat_map(|_| (0..12).map(|k| k == 0)).collect(); // a frame every 200 ms
    let sixty: Vec<bool> = [true; 50].into_iter().chain([true, false, false]).chain([true; 50]).collect();
    let all: Vec<bool> = pauses.into_iter().chain(sixty.repeat(3)).collect();
    assert!(!all.iter().any(|&c| w.update(c)));
}

#[test]
fn test_a_frame_counts_as_shown_when_it_is_drawn_as_itself() {
    for hz in [60, 90, 120] {
        let mut tl = FrameTimeline::new(12);
        tl.set_at_arrival(true);
        let mut rnd = PyRandom::new(3);
        let mut calls = Vec::new();
        for _ in 0..300 {
            calls.push(true);
            let k = if rnd.random() < 0.3 { 2 } else { 1 };
            calls.extend(std::iter::repeat_n(false, k));
        }
        let (mut drawn, mut counted): (BTreeMap<i64, usize>, BTreeMap<i64, usize>) = (BTreeMap::new(), BTreeMap::new());
        for (n, &new) in calls.iter().enumerate() {
            tl.tick(new, false, Some(n as i64), 0.0);
            for phi in tl.slots(hz) {
                let p = tl.place(phi).unwrap();
                if p.b.is_none() {
                    drawn.entry(p.a.frame.unwrap()).or_insert(n);
                }
                for f in &p.reached {
                    let fr = f.frame.unwrap();
                    assert!(!counted.contains_key(&fr)); // each frame once
                    counted.insert(fr, n);
                }
            }
        }
        assert!(counted.len() > 280);
        let late: Vec<i64> = drawn.iter().filter(|&(&a, &n)| a > 60 && counted[&a] != n).map(|(&a, _)| a).collect();
        assert!(late.is_empty(), "{hz}: {late:?}");
    }
}
