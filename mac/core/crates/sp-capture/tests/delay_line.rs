//! The card's sound held back to meet the picture (tests/test_audio.py's DelayLine and Player tests), and the
//! Python's DelayLine replayed. The same pushes and pulls must give the same bytes (fixtures/capture/delay_*).

mod common;

use common::{card, PyRng};
use sp_capture::audio::{DelayLine, PlayerCore};
use sp_capture::{sound_report, Player, SoundStats};

const RATE: u32 = 48000;
const CH: usize = 2;
const BLOCK: usize = 480; // the output's 10 ms

fn pcm(seconds: f64, value: i16) -> Vec<i16> {
    vec![value; (seconds * RATE as f64).round() as usize * CH]
}

/// A loud 220 Hz tone, continuous across packets as a card's sound is, 20 ms a packet.
struct Sine {
    n: usize,
}

impl Sine {
    fn next(&mut self) -> Vec<i16> {
        let n = (0.02 * RATE as f64) as usize;
        let mut out = Vec::with_capacity(n * CH);
        for i in self.n..self.n + n {
            let v = (12000.0 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / RATE as f64).sin()) as i16;
            out.extend([v; CH]);
        }
        self.n += n;
        out
    }
}

/// Packets in and blocks out, at their times. Returns the output as played.
fn play(line: &mut DelayLine, events: &[(f64, bool)], mut packet: impl FnMut() -> Vec<i16>) -> Vec<i16> {
    let mut out = Vec::new();
    for &(t, inn) in events {
        if inn {
            line.push_at(t, &packet());
        } else {
            out.extend(line.pull(BLOCK));
        }
    }
    out
}

#[test]
fn hold_ups_reading_the_card_leave_no_gaps() {
    // Issue 17 again. The thread that reads the card is held up now and then, and the packets meanwhile come at
    // once. With only a packet and 10 ms queued, each hold-up of 20 ms or more was a gap. The cushion grows to the
    // longest recent wait.
    for hold in [20.0, 40.0, 60.0, 100.0] {
        let mut line = DelayLine::new(RATE, CH);
        line.set_delay(0.015);
        play(&mut line, &card(120.0, hold, 5.0, 2, None), || pcm(0.02, 100));
        assert!(line.underruns <= 3, "{hold}: {}", line.underruns); // the first hold-up only
        assert_eq!(line.jumps(), 0, "{hold}: {} {}", line.skips, line.refills);
        assert!(line.kept_s() <= hold / 1000.0 + 0.035 && line.kept_s() <= 0.15 + 1e-3);
        assert!((line.longest_wait_s - (hold + 20.0) / 1000.0).abs() < 0.008, "{hold}: {}", line.longest_wait_s);
    }
}

#[test]
fn the_cushion_forgets_old_hold_ups_and_a_stop_isnt_one() {
    let mut line = DelayLine::new(RATE, CH);
    play(&mut line, &card(70.0, 60.0, 100.0, 2, Some(2.0)), || pcm(0.02, 100)); // one hold-up, near the start
    assert!(line.kept_s() < 0.04); // a minute later, back to a packet and the cushion
    let mut line = DelayLine::new(RATE, CH);
    let ev: Vec<(f64, bool)> = card(10.0, 0.0, 5.0, 2, None).into_iter().filter(|e| !(4.0..6.0).contains(&e.0) || !e.1).collect();
    play(&mut line, &ev, || pcm(0.02, 100)); // the sound stops for 2 s
    assert!(line.stops == 1 && line.kept_s() < 0.04);
}

#[test]
fn a_change_in_the_pictures_delay_is_eased_not_jumped() {
    let mut line = DelayLine::new(RATE, CH);
    line.set_delay(0.04);
    let ev = card(30.0, 0.0, 5.0, 2, None);
    let part = |a: f64, b: f64| ev.iter().copied().filter(|e| a <= e.0 && e.0 < b).collect::<Vec<_>>();
    play(&mut line, &part(0.0, 5.0), || pcm(0.02, 100));
    line.set_delay(0.12); // 80 ms more, as when a 30 fps game starts to dip
    play(&mut line, &part(5.0, 20.0), || pcm(0.02, 100));
    assert!((line.queued_s() - 0.12).abs() < 0.02 && line.jumps() == 0 && line.underruns == 0);
    line.set_delay(0.04); // and back
    play(&mut line, &part(20.0, 1e9), || pcm(0.02, 100));
    assert!((line.queued_s() - 0.04).abs() < 0.02 && line.jumps() == 0);
}

#[test]
fn jumps_and_gaps_are_eased_so_they_dont_click() {
    let natural = 12000.0 * 2.0 * std::f64::consts::PI * 220.0 / RATE as f64; // the tone's own biggest step (about 350)
    let mut line = DelayLine::new(RATE, CH);
    line.set_delay(0.3);
    let mut tone = Sine { n: 0 };
    let ev = card(6.0, 0.0, 5.0, 2, None);
    let part = |a: f64, b: f64| ev.iter().copied().filter(|e| a <= e.0 && e.0 < b).collect::<Vec<_>>();
    let mut out = play(&mut line, &part(0.0, 3.0), || tone.next());
    line.set_delay(0.0); // far less, so the oldest sound is skipped
    out.extend(play(&mut line, &part(3.0, 4.0), || tone.next()));
    let dry: Vec<(f64, bool)> = part(4.0, 6.0).into_iter().filter(|e| !e.1).collect();
    out.extend(play(&mut line, &dry, || unreachable!())); // nothing comes, so it runs dry
    let back: Vec<(f64, bool)> = card(2.0, 0.0, 5.0, 2, None).into_iter().map(|(t, i)| (6.0 + t, i)).collect();
    out.extend(play(&mut line, &back, || tone.next()));
    assert!(line.skips >= 1 && line.underruns == 1); // one gap while nothing came, not one per block
    let left: Vec<i32> = out.iter().step_by(CH).map(|&v| v as i32).collect();
    let step = left.windows(2).map(|w| (w[1] - w[0]).abs()).max().unwrap() as f64;
    assert!(step < natural + 24000.0 / 96.0 + 50.0, "a step of {step}"); // eased, never a step like a click
}

#[test]
fn silence_until_the_delay_then_the_sound_in_order() {
    let mut line = DelayLine::new(RATE, CH);
    line.set_delay(0.1);
    let mut out = Vec::new();
    for i in 0..30 {
        // 10 ms in, 10 ms out
        line.push_at(i as f64 * 0.01, &pcm(0.01, i as i16 + 1));
        out.extend(line.pull(BLOCK));
    }
    let left: Vec<i16> = out.iter().step_by(CH).copied().collect();
    let first = left.iter().position(|&v| v != 0).unwrap() as f64 / RATE as f64;
    assert!((0.06..=0.11).contains(&first), "{first}"); // silence while it fills (to within a jump), then
    let sound: Vec<i16> = left.into_iter().filter(|&v| v != 0).collect();
    assert!(sound.windows(2).all(|w| w[1] >= w[0]) && sound[0] == 1); // the sound from the start, in order
    assert!((line.queued_s() - 0.1).abs() < 0.045);
}

#[test]
fn the_delay_holds_when_the_card_clock_runs_fast_or_slow() {
    for speed in [1.002, 0.998] {
        // 0.2%, far more than real clocks differ
        let mut line = DelayLine::new(RATE, CH);
        line.set_delay(0.05);
        let (mut got, mut queued) = (0.0f64, 0.0);
        let r4 = |x: f64| (x * 1e4).round_ties_even() / 1e4;
        for i in 0..3000 {
            got += 0.01 * speed;
            line.push_at(i as f64 * 0.01, &pcm(r4(got) - r4(got - 0.01 * speed), 100));
            queued = line.queued_s(); // before the output takes its block
            line.pull(BLOCK);
        }
        assert!((queued - 0.05).abs() < 0.006, "{speed}: {queued}");
        assert_eq!(line.jumps(), 0); // nudged back, by playing a little faster or slower
    }
}

#[test]
fn a_short_delay_never_runs_dry_between_packets() {
    // Issue 17. A card's sound came in 20 ms packets, a little late now and then, with less delay asked for than a
    // packet lasts. The queue ran dry every packet or so, a click each time.
    let mut rnd = PyRng::new(1);
    for asked in [0.0, 0.005, 0.015] {
        let mut line = DelayLine::new(RATE, CH);
        line.set_delay(asked);
        let mut ev: Vec<(f64, bool)> = (0..1500).map(|i| (i as f64 * 0.02 + rnd.gauss(0.0, 0.004).abs(), true)).collect();
        ev.extend((0..3000).map(|i| (i as f64 * 0.010001, false)));
        ev.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        play(&mut line, &ev, || pcm(0.02, 100));
        assert!(line.underruns <= 1, "{asked}: {}", line.underruns); // the first block, before anything came
        // the longest wait between packets and the cushion, so sound at most about 45 ms behind the picture
        assert!((0.025..=0.045).contains(&line.kept_s()), "{}", line.kept_s());
        assert_eq!(line.jumps(), 0);
    }
    let mut line = DelayLine::new(RATE, CH);
    line.set_delay(0.08);
    line.push_at(0.0, &pcm(0.02, 0));
    assert!((line.kept_s() - 0.08).abs() < 1e-3); // a longer delay is kept as asked
}

#[test]
fn a_big_change_jumps() {
    let mut line = DelayLine::new(RATE, CH);
    line.set_delay(0.3);
    for i in 0..60 {
        line.push_at(i as f64 * 0.01, &pcm(0.01, 7));
        line.pull(BLOCK);
    }
    line.set_delay(0.03);
    line.push_at(0.6, &pcm(0.01, 7));
    line.pull(BLOCK);
    assert!(line.queued_s() < 0.05 && line.jumps() >= 1);
    assert_eq!(line.pull(BLOCK).len(), BLOCK * CH);
}

#[test]
fn the_log_says_what_went_wrong() {
    let s = SoundStats {
        ran_dry: 3,
        output_short: 1,
        skips: 0,
        refills: 1,
        stops: 0,
        longest_wait_ms: 46,
        kept_ms: 56,
        longest_call_ms: Some(31),
        output_ms: Some(40),
    };
    let full = sound_report(&s, None).unwrap();
    assert!(full.contains("ran dry 3 times") && full.contains("output ran short 1 times") && full.contains("46 ms"));
    assert!(full.contains("output 40 ms") && !full.contains("stopped"));
    assert_eq!(sound_report(&s, Some(&s)), None); // nothing new, so no line
    assert!(sound_report(&SoundStats { ran_dry: 4, ..s.clone() }, Some(&s)).unwrap().contains("ran dry 1 times"));
}

#[test]
fn the_player_counts_the_output_running_short() {
    let p = PlayerCore::new(RATE, CH, 0.04); // no sound device, only the callback
    let mut buf = vec![0i16; BLOCK * CH];
    p.render(&mut buf, false);
    std::thread::sleep(std::time::Duration::from_millis(20));
    p.render(&mut buf, true); // the output says so
    assert_eq!(p.stats().output_short, 1);
    std::thread::sleep(std::time::Duration::from_millis(60));
    p.render(&mut buf, false); // it doesn't, but the call came later than the 40 ms buffer lasts
    let st = p.stats();
    assert_eq!(st.output_short, 2);
    assert!(st.longest_call_ms.unwrap() >= 50 && st.output_ms == Some(40));
    // the delay asked for less the output's own
    p.set_delay(0.1);
    assert_eq!(p.line.lock().unwrap().target(), (0.06 * RATE as f64).round() as usize * CH);
}

/// The Python's DelayLine, replayed. Every byte it played must match, and so must its counts at the end.
#[test]
fn the_same_bytes_as_the_python() {
    let names = ["holdups", "eased", "change", "fast", "slow", "fill", "odd"];
    let mut ran = 0;
    for name in names {
        let Some(j) = common::fixture(&format!("delay_{name}.json")) else { return };
        let info: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(j).unwrap()).unwrap();
        let read = |ext: &str| -> Vec<i16> {
            let b = std::fs::read(common::fixtures().join(format!("delay_{name}.{ext}"))).unwrap();
            b.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect()
        };
        let (pin, pout) = (read("in.s16"), read("out.s16"));
        let ch = info["channels"].as_u64().unwrap() as usize;
        let mut line = DelayLine::new(info["rate"].as_u64().unwrap() as u32, ch);
        let mut out: Vec<i16> = Vec::with_capacity(pout.len());
        for op in info["ops"].as_array().unwrap() {
            match op[0].as_str().unwrap() {
                "delay" => line.set_delay(op[1].as_f64().unwrap()),
                "in" => {
                    let (o, n) = (op[2].as_u64().unwrap() as usize, op[3].as_u64().unwrap() as usize);
                    line.push_at(op[1].as_f64().unwrap(), &pin[o..o + n]);
                }
                _ => {
                    let at = out.len();
                    out.extend(line.pull(op[2].as_u64().unwrap() as usize));
                    if out[at..] != pout[at..out.len().min(pout.len())] {
                        let i = (at..out.len()).find(|&i| out[i] != pout[i]).unwrap();
                        panic!("{name}: differs from the Python at sample {i} (block from {at}): {} vs {}", out[i], pout[i]);
                    }
                }
            }
        }
        assert_eq!(out.len(), pout.len(), "{name}");
        let st = line.stats();
        let want = &info["stats"];
        assert_eq!(
            (st.ran_dry, st.skips, st.refills, st.stops, st.longest_wait_ms, st.kept_ms),
            (
                want["ran_dry"].as_u64().unwrap(),
                want["skips"].as_u64().unwrap(),
                want["refills"].as_u64().unwrap(),
                want["stops"].as_u64().unwrap(),
                want["longest_wait_ms"].as_i64().unwrap(),
                want["kept_ms"].as_i64().unwrap()
            ),
            "{name}"
        );
        assert!((line.queued_s() - info["queued_s"].as_f64().unwrap()).abs() < 1e-12, "{name}");
        ran += 1;
    }
    assert_eq!(ran, names.len());
}

/// The real output, playing nothing but silence for a moment. It opens, calls for sound, knows its latency, and
/// closes. Skipped with no sound device.
#[test]
fn the_player_plays_on_the_macs_output() {
    let mut p = match Player::open(RATE, CH as u32) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("(skipped: no output: {e})");
            return;
        }
    };
    p.set_delay(0.05);
    for _ in 0..30 {
        p.push(&vec![0i16; BLOCK * CH]);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let st = p.stats();
    println!("output {:?}: latency {:.1} ms, {} calls, {st:?}", p.device_name, p.latency() * 1000.0, p.calls());
    assert!(p.calls() > 5, "the output never asked for sound");
    assert!(p.latency() > 0.0 && p.latency() < 0.5, "{}", p.latency());
    p.close();
    let calls = p.calls();
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert_eq!(p.calls(), calls, "still called after closing");
}
