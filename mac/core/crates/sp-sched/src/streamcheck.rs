//! Remote Play stream check (framegen/streamcheck.py): how well the size-based repeat check read a real game, in
//! plain words. It keeps what the stream and the picture check already measured, one row per frame, and report()
//! turns them into lines: the game's frame rate, the rhythm and its breaks (each tied to what was around it), the
//! picture's view of likely misreads, the sizes, the arrival times and a verdict. The arithmetic is numpy's
//! (crate::np), so the text matches the Python's.

use sp_core::Tick;

use crate::cadence::FrameDiff;
use crate::np;

pub const TICK_MS: f64 = 1000.0 / 60.0;
/// The first 30 minutes; a check needs about 5.
pub const MAX_TICKS: usize = 30 * 60 * 60;
/// size / typical new frame, around the size check's 0.4: where size can't really tell.
pub const AMBIGUOUS: (f64, f64) = (0.25, 0.6);
/// Frames calibrated together for the picture's view (1 s).
pub const STRETCH: usize = 60;

fn mmss(seconds: f64) -> String {
    let s = seconds as i64;
    format!("{}:{:02}", s.div_euclid(60), s.rem_euclid(60))
}

fn pct(n: usize, d: usize) -> String {
    if d > 0 { format!("{}%", np::f(100.0 * n as f64 / d as f64, 1)) } else { "n/a".into() }
}

fn count(m: impl IntoIterator<Item = bool>) -> usize {
    m.into_iter().filter(|&b| b).count()
}

/// numpy's histogram of `xs` over `edges` (searchsorted, the last bin closed).
fn histogram(xs: &[f64], edges: &[f64]) -> Vec<i64> {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = edges.len();
    let cum: Vec<usize> = edges
        .iter()
        .enumerate()
        .map(|(i, &e)| if i + 1 < m { s.partition_point(|&v| v < e) } else { s.partition_point(|&v| v <= e) })
        .collect();
    cum.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect()
}

/// Lengths of runs of true in mask, split where frames are missing.
fn runs(mask: &[bool], seq: &[i64]) -> Vec<usize> {
    let (mut out, mut run) = (Vec::new(), 0usize);
    for (i, &m) in mask.iter().enumerate() {
        if i > 0 && seq[i] != seq[i - 1] + 1 && run > 0 {
            out.push(run);
            run = 0;
        }
        if m {
            run += 1;
        } else if run > 0 {
            out.push(run);
            run = 0;
        }
    }
    if run > 0 {
        out.push(run);
    }
    out
}

/// (miss, extra_new, checked) per frame, judged by the picture within each stretch (streamcheck.py _picture_flags).
fn picture_flags(dec: &[i8], md: &[f64], mx: &[f64], key: &[bool]) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let n = dec.len();
    let (mut miss, mut extra, mut checked) = (vec![false; n], vec![false; n], vec![false; n]);
    let fin: Vec<bool> = (0..n).map(|i| md[i].is_finite() && mx[i].is_finite()).collect();
    let mut s = 0;
    while s < n {
        let e = (s + STRETCH).min(n);
        if count(fin[s..e].iter().copied()) < STRETCH / 3 {
            s += STRETCH;
            continue;
        }
        let m: Vec<f64> = (s..e).filter(|&i| fin[i]).map(|i| md[i]).collect();
        let (lo, typ_new) = (np::percentile(&m, 25.0), np::percentile(&m, 75.0));
        let typ_rep = lo.max(0.02);
        if typ_new < 0.5 || typ_new < 5.0 * typ_rep {
            s += STRETCH;
            continue;
        }
        let quiet: Vec<f64> = (s..e).filter(|&i| fin[i] && md[i] <= 2.0 * typ_rep).map(|i| mx[i]).collect();
        let rep_max = (if quiet.is_empty() { 0.0 } else { np::median(&quiet) }).max(1.0);
        for i in s..e {
            let o = fin[i] && !key[i] && dec[i] >= 0;
            checked[i] = o;
            miss[i] = o && dec[i] == 0 && md[i] >= 0.5 * typ_new;
            extra[i] = o && dec[i] == 1 && md[i] <= 2.0 * typ_rep && mx[i] <= 2.0 * rep_max;
        }
        s += STRETCH;
    }
    (miss, extra, checked)
}

#[allow(clippy::too_many_arguments)]
fn verdict(
    span: f64,
    stream_fps: f64,
    game_fps: f64,
    steady: usize,
    seconds: usize,
    alt: usize,
    pairs: usize,
    looks: usize,
    checked: usize,
    undecided: f64,
) -> String {
    if span < 60.0 || seconds == 0 {
        return "Verdict: too short to judge; play for at least a minute (about 5 is best)".into();
    }
    if stream_fps < 45.0 {
        return "Verdict: the stream itself is 30 fps, so there are no repeats to check. Set chiaki-ng's frame rate to \
                60 and play again"
            .into();
    }
    let reads = if checked < 600 {
        "the picture couldn't confirm it (too little movement)".to_string()
    } else if looks > 0 {
        format!("{} of moving frames look misread", pct(looks, checked))
    } else {
        "no frame looks misread".into()
    };
    if game_fps >= 55.0 {
        return format!(
            "Verdict: this game ran at about 60 fps here, so there were few repeats to fill; {reads}. For this check, \
             play a game (or a mode) that runs at 30 fps"
        );
    }
    if (steady as f64 / seconds as f64) < 0.8 {
        return format!(
            "Verdict: this game's frame rate varied (about {} fps), so its rhythm can't be checked; {reads}",
            np::f(game_fps, 0)
        );
    }
    let rate = if checked >= 600 { looks as f64 / checked as f64 } else { 0.0 };
    if undecided > 0.05 {
        return format!(
            "Verdict: the size check left {}% of this 30 fps game's frames undecided (the picture check stood in; \
             {reads}). Please send this report",
            np::f(100.0 * undecided, 0)
        );
    }
    if rate <= 0.005 && alt as f64 / pairs.max(1) as f64 >= 0.97 {
        return format!("Verdict: the size check read this 30 fps game correctly ({} on rhythm; {reads})", pct(alt, pairs));
    }
    if rate <= 0.02 {
        return format!(
            "Verdict: the size check mostly read this 30 fps game correctly ({} on rhythm; {reads}). Please send this \
             report",
            pct(alt, pairs)
        );
    }
    format!("Verdict: the size check misread this 30 fps game too often ({reads}). Please send this report")
}

/// One row per Remote Play frame the pipeline saw; report() turns them into plain-language lines.
#[derive(Clone, Debug)]
pub struct StreamCheck {
    pub max_ticks: usize,
    /// Arrival, s.
    pub t: Vec<f64>,
    /// The tap's frame number: a jump means frames never reached the pipeline.
    pub seq: Vec<i64>,
    /// Compressed bytes.
    pub size: Vec<i64>,
    /// The typical new frame size the check compared with (NaN: keyframe, warm-up).
    pub reference: Vec<f64>,
    /// Size check: 1 new, 0 repeat, -1 undecided (the picture check decided).
    pub decision: Vec<i8>,
    /// Keyframe, or a frame so big it was treated as one.
    pub key: Vec<bool>,
    /// Frames were lost before it, or it's the PS5's recovery frame.
    pub lost: Vec<bool>,
    /// The picture check: average change from the frame before (0-255)...
    pub mean_diff: Vec<f64>,
    /// ...and the biggest change anywhere in the picture.
    pub max_diff: Vec<f64>,
    /// Frames played with "Simulate a 30 fps game" on: not part of the check.
    pub simulated: usize,
    /// Frames past max_ticks.
    pub extra: usize,
}

impl Default for StreamCheck {
    fn default() -> Self {
        Self::new(MAX_TICKS)
    }
}

impl StreamCheck {
    pub fn new(max_ticks: usize) -> Self {
        StreamCheck {
            max_ticks,
            t: Vec::new(),
            seq: Vec::new(),
            size: Vec::new(),
            reference: Vec::new(),
            decision: Vec::new(),
            key: Vec::new(),
            lost: Vec::new(),
            mean_diff: Vec::new(),
            max_diff: Vec::new(),
            simulated: 0,
            extra: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        t: f64,
        seq: i64,
        size: i64,
        reference: Option<f64>,
        decision: Option<bool>,
        key: bool,
        lost: bool,
        mean_diff: f64,
        max_diff: f64,
    ) {
        if self.t.len() >= self.max_ticks {
            self.extra += 1;
            return;
        }
        self.t.push(t);
        self.seq.push(seq);
        self.size.push(size);
        self.reference.push(reference.unwrap_or(f64::NAN));
        self.decision.push(decision.map_or(-1, |d| d as i8));
        self.key.push(key);
        self.lost.push(lost);
        self.mean_diff.push(if mean_diff.is_finite() { mean_diff } else { f64::NAN });
        self.max_diff.push(if max_diff.is_finite() { max_diff } else { f64::NAN });
    }

    /// From the pipeline's on_tick: a Remote Play tick (its extra: seq, size, size_ref, size_new, keyframe, big,
    /// frames_lost, recovered) and the picture check's FrameDiff.
    pub fn add_tick(&mut self, tick: &Tick, diff: &FrameDiff) {
        if tick.truth_new.is_some() {
            self.simulated += 1; // test mode replaced this frame
            return;
        }
        let e = &tick.extra;
        let int = |k: &str| e.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))).unwrap_or(0);
        let truthy = |k: &str| match e.get(k) {
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::Number(n)) => n.as_f64() != Some(0.0),
            _ => false,
        };
        self.add(
            tick.t_arrival,
            int("seq"),
            int("size"),
            e.get("size_ref").and_then(|v| v.as_f64()),
            e.get("size_new").and_then(|v| v.as_bool()),
            truthy("keyframe") || truthy("big"),
            truthy("frames_lost") || truthy("recovered"),
            diff.mean_diff,
            diff.max_diff,
        );
    }

    pub fn report(&self) -> Vec<String> {
        let n = self.t.len();
        if n < 2 {
            return Vec::new();
        }
        let (t, seq, size, dec, key, lost) = (&self.t, &self.seq, &self.size, &self.decision, &self.key, &self.lost);
        let (md, mx) = (&self.mean_diff, &self.max_diff);
        let span = t[n - 1] - t[0];
        let new: Vec<bool> = dec.iter().map(|&d| d == 1).collect();
        let rep: Vec<bool> = dec.iter().map(|&d| d == 0).collect();
        let mut lines = vec![format!(
            "Stream check (Remote Play), {} checked{}",
            mmss(span),
            if self.extra > 0 { " (the first 30 minutes; later frames weren't recorded)" } else { "" }
        )];
        if self.simulated > 0 {
            lines[0] += &format!(
                "; {} with \"Simulate a 30 fps game\" on left out",
                mmss(self.simulated as f64 * TICK_MS / 1000.0)
            );
        }

        // Frame rates, per whole second of play.
        let edges: Vec<f64> = (0..=(span as i64)).map(|i| t[0] + i as f64).collect();
        let t_new: Vec<f64> = (0..n).filter(|&i| new[i]).map(|i| t[i]).collect();
        let (per_s, pkt_s) =
            if edges.len() > 1 { (histogram(&t_new, &edges), histogram(t, &edges)) } else { (Vec::new(), Vec::new()) };
        let step: Vec<i64> = seq.windows(2).map(|w| w[1] - w[0]).collect();
        let missing: i64 = step.iter().filter(|&&s| s > 1).map(|s| s - 1).sum();
        let as_f = |v: &[i64]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let stream_fps = if per_s.is_empty() { f64::NAN } else { np::median(&as_f(&pkt_s)) };
        let game_fps = if per_s.is_empty() { f64::NAN } else { np::median(&as_f(&per_s)) };
        let steady = count(per_s.iter().map(|&p| (28..=32).contains(&p)));
        if !per_s.is_empty() {
            lines.push(format!(
                "Stream: {} frames a second reached SmoovPlay{}",
                np::f(stream_fps, 0),
                if missing > 0 { format!("; {missing} never did (packet loss, or SmoovPlay fell behind)") } else { String::new() }
            ));
            lines.push(format!(
                "Game frame rate, by the size check: {} fps (median); {} of seconds at 28-32 fps",
                np::f(game_fps, 0),
                pct(steady, per_s.len())
            ));
            let bands: [(&str, fn(i64) -> bool); 6] = [
                ("under 25", |p| p < 25),
                ("25-27", |p| (25..28).contains(&p)),
                ("28-32", |p| (28..=32).contains(&p)),
                ("33-45", |p| p > 32 && p <= 45),
                ("46-55", |p| p > 45 && p <= 55),
                ("56+", |p| p > 55),
            ];
            let parts: Vec<String> = bands
                .iter()
                .filter_map(|(name, f)| {
                    let c = count(per_s.iter().map(|&p| f(p)));
                    (c > 0).then(|| format!("{name} fps {c}"))
                })
                .collect();
            lines.push(format!("  seconds at: {}", parts.join(", ")));
        }
        let thirty = (25.0..=35.0).contains(&game_fps);

        // The picture's view, calibrated per stretch.
        let (miss, extra_new, checked) = picture_flags(dec, md, mx, key);
        let looks: Vec<bool> = (0..n).map(|i| miss[i] || extra_new[i]).collect();

        // The rhythm: consecutive frames, both decided, nothing missing between them.
        let ok: Vec<bool> = (0..n - 1).map(|i| step[i] == 1 && dec[i + 1] >= 0 && dec[i] >= 0).collect();
        let pairs = count(ok.iter().copied());
        let alt = count((0..n - 1).map(|i| ok[i] && dec[i] != dec[i + 1]));
        let breaks: Vec<usize> = (0..n - 1).filter(|&i| ok[i] && dec[i] == dec[i + 1]).map(|i| i + 1).collect();
        let nn = count(breaks.iter().map(|&i| dec[i] == 1));
        let rr = count(breaks.iter().map(|&i| dec[i] == 0));
        lines.push(format!(
            "Rhythm (a 30 fps game alternates new / repeat): {} of {pairs} frame pairs; {nn} new-new, {rr} \
             repeat-repeat{}",
            pct(alt, pairs),
            if thirty { "" } else { " (this game isn't at 30 fps, so breaks are expected)" }
        ));
        let holds = runs(&rep, seq);
        if !holds.is_empty() {
            let longest = holds.iter().max().unwrap() + 1;
            let n_long = count(holds.iter().map(|&r| r >= 2));
            lines.push(if n_long > 0 {
                format!(
                    "  held frames: {n_long} times a frame stayed up 3 or more ticks (50 ms+), longest {longest} ticks \
                     ({} ms)",
                    np::f(longest as f64 * TICK_MS, 0)
                )
            } else {
                "  held frames: none longer than 2 ticks".into()
            });
        }
        let ratio: Vec<f64> = (0..n).map(|i| size[i] as f64 / self.reference[i]).collect();
        let near: Vec<bool> = ratio.iter().map(|&r| r >= AMBIGUOUS.0 && r <= AMBIGUOUS.1).collect();
        if thirty && !breaks.is_empty() {
            let names = [
                "frames missing",
                "packet loss",
                "keyframe",
                "size close to the threshold",
                "picture says misread",
                "the game itself (the picture agrees)",
                "can't tell",
            ];
            let mut causes = [0usize; 7];
            for &i in &breaks {
                let (lo, hi) = (i.saturating_sub(2), (i + 3).min(n));
                let (s0, s1) = (i.saturating_sub(3), (i + 2).min(n - 1));
                let k = if step[s0.min(s1)..s1].iter().any(|&s| s != 1) {
                    0
                } else if lost[lo..hi].iter().any(|&l| l) {
                    1
                } else if key[i] || key[i - 1] {
                    2
                } else if looks[i] || looks[i - 1] {
                    4
                } else if near[i] || near[i - 1] {
                    3
                } else if checked[i] && checked[i - 1] {
                    5
                } else {
                    6 // a near-still stretch: the picture can't separate them either
                };
                causes[k] += 1;
            }
            let parts: Vec<String> =
                names.iter().zip(causes).filter(|(_, v)| *v > 0).map(|(k, v)| format!("{k} {v}")).collect();
            lines.push(format!("  breaks: {}", parts.join(", ")));
        }

        let n_checked = count(checked.iter().copied());
        let n_looks = count(looks.iter().copied());
        if n_checked >= 600 {
            lines.push(format!(
                "Picture check, in stretches where the picture moves ({} of frames): {n_looks} look misread ({}): {} \
                 \"repeats\" changed like a new frame, {} \"new\" frames changed no more than a repeat",
                pct(n_checked, n),
                pct(n_looks, n_checked),
                count(miss.iter().copied()),
                count(extra_new.iter().copied())
            ));
        } else {
            lines.push(format!(
                "Picture check: too few stretches where new frames and repeats clearly differ on screen ({n_checked} \
                 frames; a still scene, or few repeats), so it can't confirm the size check"
            ));
        }
        // Does the PS5 send true copies?
        let same: Vec<bool> = mx.iter().map(|&x| x.is_finite() && x <= 0.01).collect();
        let n_rep = count((0..n).map(|i| rep[i] && !key[i]));
        if n_rep > 0 {
            lines.push(format!(
                "  pixel-identical to the frame before: {} of \"repeats\", {} of \"new\" frames",
                pct(count((0..n).map(|i| same[i] && rep[i] && !key[i])), n_rep),
                pct(count((0..n).map(|i| same[i] && new[i] && !key[i])), count((0..n).map(|i| new[i] && !key[i])))
            ));
            if count((0..n).map(|i| same[i] && rep[i] && !key[i])) as f64 >= 0.9 * n_rep as f64 {
                let agree = count((0..n).map(|i| (new[i] && !same[i]) || (rep[i] && same[i])));
                lines.push(format!(
                    "  the picture alone (pixel-identical = repeat) agrees with the size check on {} of frames, and \
                     would also decide the {} the size check left undecided",
                    pct(agree, count(dec.iter().map(|&d| d >= 0))),
                    count(dec.iter().map(|&d| d < 0))
                ));
            }
        }

        // How far apart new frames and repeats are by size.
        let sized: Vec<bool> = (0..n).map(|i| !key[i] && ratio[i].is_finite()).collect();
        let rn: Vec<f64> = (0..n).filter(|&i| sized[i] && new[i]).map(|i| ratio[i]).collect();
        let rp: Vec<f64> = (0..n).filter(|&i| sized[i] && rep[i]).map(|i| ratio[i]).collect();
        if !rn.is_empty() && !rp.is_empty() {
            let sn: Vec<f64> = (0..n).filter(|&i| sized[i] && new[i]).map(|i| size[i] as f64).collect();
            let sr: Vec<f64> = (0..n).filter(|&i| sized[i] && rep[i]).map(|i| size[i] as f64).collect();
            lines.push(format!(
                "Sizes: new frames {} KB, repeats {} KB (medians). Against a typical new frame, the smallest 5% of new \
                 frames were {}x and the biggest 5% of repeats {}x (threshold 0.40x); {} of frames were close to it \
                 ({}-{}x)",
                np::f(np::median(&sn) / 1000.0, 1),
                np::f(np::median(&sr) / 1000.0, 1),
                np::f(np::percentile(&rn, 5.0), 2),
                np::f(np::percentile(&rp, 95.0), 2),
                pct(count((0..n).map(|i| near[i] && sized[i])), count(sized.iter().copied())),
                np::f(AMBIGUOUS.0, 2),
                np::f(AMBIGUOUS.1, 2)
            ));
        }
        let key_runs = runs(key, seq);
        lines.push(format!(
            "Undecided by size (the picture check decided): {} frames; keyframes (or frames far bigger than recent ones) \
             {} ({}){}",
            count(dec.iter().map(|&d| d < 0)),
            count(key.iter().copied()),
            pct(count(key.iter().copied()), n),
            match key_runs.iter().max() {
                Some(m) => format!(", longest run {m} in a row"),
                None => String::new(),
            }
        ));

        // When frames arrived.
        let dt: Vec<f64> = (0..n - 1).filter(|&i| step[i] == 1).map(|i| (t[i + 1] - t[i]) * 1000.0).collect();
        if !dt.is_empty() {
            let slowest = dt.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            lines.push(format!(
                "Arrival: a frame every {} ms (median), 95% within {} ms, slowest {} ms; {} over 25 ms",
                np::f(np::median(&dt), 1),
                np::f(np::percentile(&dt, 95.0), 1),
                np::f(slowest, 0),
                pct(count(dt.iter().map(|&d| d > 25.0)), dt.len())
            ));
        }
        if t_new.len() > 2 {
            let dn: Vec<f64> = t_new.windows(2).map(|w| (w[1] - w[0]) * 1000.0).collect();
            lines.push(format!(
                "  new frames: every {} ms (median), 95% within {} ms",
                np::f(np::median(&dn), 1),
                np::f(np::percentile(&dn, 95.0), 1)
            ));
        }

        lines.push(verdict(
            span,
            stream_fps,
            game_fps,
            steady,
            per_s.len(),
            alt,
            pairs,
            n_looks,
            n_checked,
            count(dec.iter().map(|&d| d < 0)) as f64 / n as f64,
        ));
        lines
    }
}
