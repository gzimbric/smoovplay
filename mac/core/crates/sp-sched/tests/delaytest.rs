//! tests/test_delaytest.py's delay test on its own and small(), ported: same names, same assertions (the pipeline's
//! part is in tests/pipeline.rs).

mod common;

use common::pyrandom::PyRandom;
use sp_core::Frame;
use sp_sched::delaytest::{self as dt, DelayTest, Press};
use sp_sched::{CpuProbe, GpuProbe, Probe};

/// chiaki-ng as the delay test sees it, on a clock the test moves.
struct Remote {
    answer_s: f64,
    send_s: f64,
    answers: bool,
    sends: bool,
    asked: Vec<f64>,
    sent: Vec<f64>,
    now: f64,
}

impl Remote {
    fn new(answer_s: f64, answers: bool, sends: bool) -> Remote {
        Remote { answer_s, send_s: 0.004, answers, sends, asked: vec![], sent: vec![], now: 0.0 }
    }
    fn menu_open(&self, t: f64) -> bool {
        self.answers && self.sent.iter().filter(|&&s| t >= s + self.answer_s).count() % 2 == 1
    }
}

impl Press for Remote {
    fn press(&mut self, buttons: u32, hold_ms: u32) -> u64 {
        assert!(buttons == dt::BUTTON_OPTIONS && hold_ms == dt::HOLD_MS);
        self.asked.push(self.now);
        (self.asked.len() - 1) as u64
    }
    fn press_sent(&mut self, token: u64) -> Option<f64> {
        let t = self.asked[token as usize] + self.send_s;
        if !self.sends || self.now < t {
            return None;
        }
        if self.sent.len() <= token as usize {
            self.sent.push(t);
        }
        Some(self.sent[token as usize])
    }
}

/// A still scene, with a little noise, and the menu when it's open.
fn blocks(menu: bool, rng: &mut PyRandom) -> Vec<f32> {
    let mut b: Vec<f32> = (0..18 * 32).map(|_| (60.0 + 0.5 * rng.normal()) as f32).collect();
    if menu {
        for y in 2..16 {
            for x in 4..28 {
                b[y * 32 + x] = 200.0;
            }
        }
    }
    b
}

#[test]
fn test_times_each_press_from_sending_to_the_answer() {
    let mut rng = PyRandom::new(0);
    let mut remote = Remote::new(0.1, true, true);
    remote.now = 100.0;
    let mut test = DelayTest::new(dt::PRESSES, dt::BUTTON_OPTIONS, 100.0);
    while !test.done() && remote.now < 140.0 {
        remote.now += 1.0 / 60.0;
        let (now, b) = (remote.now, blocks(remote.menu_open(remote.now), &mut rng));
        test.feed(&mut remote, &b, now, now);
    }
    assert!(test.done() && test.error.is_empty() && test.pressed == dt::PRESSES && test.ms.len() == dt::PRESSES as usize);
    assert!(test.ms.iter().all(|&ms| (100.0..100.0 + 1000.0 / 60.0 + 1.0).contains(&ms))); // the first frame after
    let text = test.result(Some(30.0));
    assert!(text.contains("PS5 and Remote Play 1") && text.contains("SmoovPlay 30 ms") && text.contains("from button to"));
    assert!(test.status().contains("press 6 of 6"));
}

#[test]
fn test_says_when_nothing_answers_or_nothing_is_sent() {
    let mut rng = PyRandom::new(1);
    for (answers, sends, words) in [(false, true, "nothing changed on screen"), (true, false, "didn't send")] {
        let mut remote = Remote::new(0.1, answers, sends);
        let mut test = DelayTest::new(dt::PRESSES, dt::BUTTON_OPTIONS, 0.0);
        while !test.done() && remote.now < 30.0 {
            remote.now += 1.0 / 60.0;
            let (now, b) = (remote.now, blocks(false, &mut rng));
            test.feed(&mut remote, &b, now, now);
        }
        assert!(test.done() && test.error.contains(words) && test.result(None).contains(words));
    }
}

struct Never;
impl Press for Never {
    fn press(&mut self, _: u32, _: u32) -> u64 {
        0
    }
    fn press_sent(&mut self, _: u64) -> Option<f64> {
        None
    }
}

#[test]
fn test_a_moving_picture_raises_the_bar() {
    let mut test = DelayTest::new(dt::PRESSES, dt::BUTTON_OPTIONS, 0.0);
    let mut rng = PyRandom::new(2);
    for i in 0..80 {
        // a second and a bit of a scene that moves by about 10 levels over SPAN frames
        let b: Vec<f32> = (0..18 * 32).map(|_| (60.0 + i as f64 + 0.5 * rng.normal()) as f32).collect();
        test.feed(&mut Never, &b, i as f64 / 60.0, i as f64 / 60.0);
    }
    assert!(test.threshold > 2.0 * dt::SPAN as f64);
}

#[test]
fn test_small_is_18x32_brightness() {
    let (w, h) = (64, 36);
    let bgr: Vec<u8> = (0..w * h).flat_map(|_| [0u8, 0, 255]).collect(); // red
    for b in [GpuProbe::new().small(&Frame::from_bgr(&bgr, w, h)), CpuProbe.small(&Frame::from_bgr(&bgr, w, h))] {
        assert!(b.len() == 18 * 32 && b.iter().all(|&v| (v - 0.299 * 255.0).abs() < 0.01));
    }
}

#[test]
fn test_small_reads_every_4th_pixel_of_a_1080p_frame_and_still_gets_each_block() {
    assert!(dt::step(1080, 1920) == 4 && dt::step(2160, 3840) == 8 && dt::step(720, 1280) == 2);
    assert_eq!(dt::step(36, 64), 1);
    let mut g = PyRandom::new(0);
    let block: Vec<[u8; 3]> = (0..18 * 32).map(|_| [0, 1, 2].map(|_| (g.random() * 256.0) as u8)).collect();
    let (w, h) = (1920, 1080);
    let mut bgr = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let b = block[(y / 60) * 32 + x / 60];
            for c in 0..3 {
                bgr[(y * w + x) * 3 + c] = (b[c] as i32 + (g.random() * 7.0) as i32 - 3).clamp(0, 255) as u8;
            }
        }
    }
    let mut every = vec![0f64; 18 * 32];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 3;
            every[(y / 60) * 32 + x / 60] += bgr[p] as f64 * 0.114 + bgr[p + 1] as f64 * 0.587 + bgr[p + 2] as f64 * 0.299;
        }
    }
    let small = GpuProbe::new().small(&Frame::from_bgr(&bgr, w, h));
    let worst = small.iter().zip(&every).map(|(&s, &e)| (s as f64 - e / 3600.0).abs()).fold(0.0, f64::max);
    assert!(worst < 1.0, "{worst}"); // within a brightness level of reading every pixel
    let cpu = dt::small_bgr(&bgr, w, h);
    let gpu_cpu = small.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(gpu_cpu < 1e-3, "{gpu_cpu}"); // and the GPU reads what the CPU reference does
}
