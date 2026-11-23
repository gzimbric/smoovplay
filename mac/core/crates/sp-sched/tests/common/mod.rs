//! Stand-ins for the pipeline's tests, as tests/ in the Python repo uses them: a source from a list of ticks, test
//! mode (framegen/sources.py TestMode), an interpolator whose frames say which two frames and which fraction they
//! are, a presenter that records what it got, chiaki-ng for the delay test and the sound, instant replay, and the
//! pictures tools/sched_traces.py makes (content()).
#![allow(dead_code)]

pub mod pyrandom;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{Value, json};
use sp_core::stats::BlockMeans;
use sp_core::{Event, Frame, Interpolator, Overlay, Presenter, SinkReply, Source, SourceInfo, Tick, Waiting};
use sp_sched::delaytest::Press;
use sp_sched::score::ScoreSums;
use sp_sched::{FakeClock, Live, LiveSettings, OverlayStyle, Probe, Remote, Replay, SharedLive};

pub const START: f64 = 1000.0;
pub type Events = Arc<Mutex<Vec<Value>>>;

pub fn push(ev: &Events, v: Value) {
    ev.lock().unwrap().push(v);
}

/// A float for JSON as the Python trace writes it (inf and nan as strings).
pub fn num(x: f64) -> Value {
    if x.is_nan() {
        json!("nan")
    } else if x.is_infinite() {
        json!(if x > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(x)
    }
}

// ------------------------------------------------------------------------------------------------ pictures
fn h32(xs: &[i64]) -> u64 {
    let mut v: u64 = 2166136261;
    for &x in xs {
        v = ((v ^ ((x as u64) & 0xFFFF_FFFF)) * 16777619) & 0xFFFF_FFFF;
    }
    v
}

/// HxWx3 BGR from a recipe, as tools/sched_traces.py content() makes it.
pub fn content(c: &Value, w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0i64; w * h * 3];
    if c.get("kind").and_then(|k| k.as_str()) == Some("menu") {
        let open = c["open"].as_bool().unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = if open && (1..h - 1).contains(&y) && (1..w - 1).contains(&x) { 200 } else { 60 };
                for ch in 0..3 {
                    img[(y * w + x) * 3 + ch] = v;
                }
            }
        }
        return img.into_iter().map(|v| v as u8).collect();
    }
    let s = c["s"].as_i64().unwrap();
    let k = c["k"].as_i64().unwrap();
    for y in 0..h {
        for x in 0..w {
            let base = 30 + (h32(&[(x / 16) as i64, (y / 16) as i64, s]) % 150) as i64;
            for ch in 0..3 {
                img[(y * w + x) * 3 + ch] = base + 20 * ch as i64;
            }
        }
    }
    let x0 = (5 * k).rem_euclid(w as i64 - 12) as usize;
    let y0 = h / 2 - 6;
    let fill = |img: &mut Vec<i64>, ys: std::ops::Range<usize>, xs: std::ops::Range<usize>, v: i64| {
        for y in ys.clone() {
            for x in xs.clone() {
                if y < h && x < w {
                    for ch in 0..3 {
                        img[(y * w + x) * 3 + ch] = v;
                    }
                }
            }
        }
    };
    fill(&mut img, y0..y0 + 12, x0..x0 + 12, 250);
    if c.get("hud").and_then(|v| v.as_bool()).unwrap_or(false) {
        fill(&mut img, 8..24, 8..56, 255);
        for x in [12, 24, 36] {
            fill(&mut img, 12..20, x..x + 4, 0);
        }
    }
    let add = c.get("add").and_then(|v| v.as_i64()).unwrap_or(0);
    let noise = c.get("noise").and_then(|v| v.as_i64()).unwrap_or(-1);
    let amp = c.get("amp").and_then(|v| v.as_i64()).unwrap_or(2);
    for y in 0..h {
        for x in 0..w {
            let n = if noise >= 0 { (h32(&[x as i64, y as i64, noise + 1000]) % (2 * amp as u64 + 1)) as i64 - amp } else { 0 };
            for ch in 0..3 {
                let i = (y * w + x) * 3 + ch;
                img[i] += add + n;
            }
        }
    }
    img.into_iter().map(|v| v.clamp(0, 255) as u8).collect()
}

// ------------------------------------------------------------------------------------------------ frames
/// What a frame is, for the trace: {"tag": v} or {"ai": [a, b, t]}.
#[derive(Debug)]
pub struct Tag(pub Value);

pub fn tag_of(f: &Frame) -> Value {
    f.owner.as_ref().and_then(|o| o.downcast_ref::<Tag>()).map_or(json!({"tag": null}), |t| t.0.clone())
}

pub fn tag_value(f: &Frame) -> Option<f64> {
    tag_of(f).get("tag").and_then(|v| v.as_f64())
}

fn dummy() -> Frame {
    static D: OnceLock<Frame> = OnceLock::new();
    D.get_or_init(|| Frame::from_bgr(&[0u8; 8 * 8 * 3], 8, 8)).clone()
}

/// A frame whose pixels don't matter, only its tag.
pub fn tagged(tag: Value) -> Frame {
    let mut f = dummy();
    f.owner = Some(Arc::new(Tag(tag)));
    f
}

pub fn with_tag(mut f: Frame, tag: Value) -> Frame {
    f.owner = Some(Arc::new(Tag(tag)));
    f
}

/// Block means from the tag's value (an 8x8 float frame of that value, as tests/test_dips.py's frames).
pub struct TagProbe;

impl Probe for TagProbe {
    fn block_means(&mut self, frame: &Frame) -> BlockMeans {
        let v = tag_value(frame).unwrap_or(0.0) as f32;
        BlockMeans { rows: 1, cols: 1, data: vec![v, v, v] }
    }
    fn small(&mut self, frame: &Frame) -> Vec<f32> {
        vec![tag_value(frame).unwrap_or(0.0) as f32; 18 * 32]
    }
    fn score(&mut self, _: &Frame, _: &Frame, _: &Frame, _: &Frame) -> ScoreSums {
        ScoreSums { pixels: 64.0, ..Default::default() }
    }
}

// ------------------------------------------------------------------------------------------------ settings
pub fn live_from(v: &Value) -> Live {
    let mut l = Live::default();
    apply_settings(&mut l, v);
    l
}

pub fn apply_settings(l: &mut Live, v: &Value) {
    let Some(m) = v.as_object() else { return };
    for (k, x) in m {
        match k.as_str() {
            "interp" => l.interp = x.as_bool().unwrap(),
            "hz" => l.hz = x.as_u64().unwrap() as u32,
            "simulate_30" => l.simulate_30 = x.as_bool().unwrap(),
            "steady_hud" => l.steady_hud = x.as_bool().unwrap(),
            "overlay" => l.overlay = x.as_bool().unwrap(),
            "overlay_style" => {
                l.overlay_style = if x.as_str() == Some("minimal") { OverlayStyle::Minimal } else { OverlayStyle::Detailed }
            }
            "even_out" => l.even_out = x.as_bool().unwrap(),
            "even_dips" => l.even_dips = x.as_bool().unwrap(),
            "game_fps" => l.game_fps_30 = x.as_str() == Some("30"),
            "audio_sync" => l.audio_sync = x.as_bool().unwrap(),
            other => panic!("unknown setting {other}"),
        }
    }
}

pub fn live_json(l: &Live) -> Value {
    json!({
        "interp": l.interp, "hz": l.hz, "simulate_30": l.simulate_30, "steady_hud": l.steady_hud,
        "overlay": l.overlay, "overlay_style": if l.overlay_style == OverlayStyle::Minimal { "minimal" } else { "detailed" },
        "even_out": l.even_out, "even_dips": l.even_dips, "game_fps": if l.game_fps_30 { "30" } else { "auto" },
        "audio_sync": l.audio_sync,
    })
}

// ------------------------------------------------------------------------------------------------ chiaki-ng
#[derive(Debug, Default)]
pub struct RemoteState {
    pub asked: Vec<f64>,
    pub sent: Vec<f64>,
    pub answer_s: f64,
    pub send_s: f64,
    pub answers: bool,
    pub sends: bool,
    pub press: bool,
    /// A capture card, with sound to hold back and nothing to press (no press at all).
    pub card: bool,
    pub audio: Vec<i64>,
}

impl RemoteState {
    pub fn menu_open(&self, t: f64) -> bool {
        !self.card && self.answers && self.sent.iter().filter(|&&s| t >= s + self.answer_s).count() % 2 == 1
    }
}

pub struct TraceRemote {
    pub st: Arc<Mutex<RemoteState>>,
    pub clock: Arc<FakeClock>,
    pub ev: Events,
}

impl TraceRemote {
    pub fn new(cfg: &Value, clock: Arc<FakeClock>, ev: Events) -> TraceRemote {
        let f = |k: &str, d: f64| cfg.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let b = |k: &str| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(true);
        let st = RemoteState {
            answer_s: f("answer_s", 0.1),
            send_s: f("send_s", 0.004),
            answers: b("answers"),
            sends: b("sends"),
            press: b("press"),
            card: cfg.get("card").and_then(|v| v.as_bool()).unwrap_or(false),
            ..Default::default()
        };
        TraceRemote { st: Arc::new(Mutex::new(st)), clock, ev }
    }
}

impl Press for TraceRemote {
    fn press(&mut self, buttons: u32, hold_ms: u32) -> u64 {
        push(&self.ev, json!(["press", buttons, hold_ms]));
        let mut s = self.st.lock().unwrap();
        s.asked.push(self.clock.get());
        (s.asked.len() - 1) as u64
    }
    fn press_sent(&mut self, token: u64) -> Option<f64> {
        let mut s = self.st.lock().unwrap();
        let t = s.asked[token as usize] + s.send_s;
        if !s.sends || self.clock.get() < t {
            return None;
        }
        if s.sent.len() <= token as usize {
            s.sent.push(t);
        }
        Some(s.sent[token as usize])
    }
}

impl Remote for TraceRemote {
    fn can_press(&self) -> bool {
        let s = self.st.lock().unwrap();
        s.press && !s.card
    }
    fn has_press(&self) -> bool {
        !self.st.lock().unwrap().card
    }
    fn can_delay_audio(&self) -> bool {
        true
    }
    fn set_audio_delay(&mut self, ms: i64) {
        self.st.lock().unwrap().audio.push(ms);
        push(&self.ev, json!(["audio", ms]));
    }
}

// ------------------------------------------------------------------------------------------------ the source
pub struct TraceSource {
    pub ticks: Vec<Value>,
    pub i: usize,
    pub clock: Arc<FakeClock>,
    pub settings: Option<Arc<SharedLive>>,
    pub remote: Option<Arc<Mutex<RemoteState>>>,
    pub tag_probe: bool,
    pub size: (usize, usize),
    pub ev: Events,
    info: SourceInfo,
}

impl TraceSource {
    pub fn new(
        ticks: Vec<Value>,
        clock: Arc<FakeClock>,
        settings: Option<Arc<SharedLive>>,
        remote: Option<Arc<Mutex<RemoteState>>>,
        tag_probe: bool,
        size: (usize, usize),
        ev: Events,
    ) -> TraceSource {
        TraceSource { ticks, i: 0, clock, settings, remote, tag_probe, size, ev, info: SourceInfo::of("sim") }
    }
}

impl Source for TraceSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        self.size
    }
    fn next_event(&mut self) -> Option<Event> {
        let i = self.i;
        let s = self.ticks.get(i)?.clone();
        self.i += 1;
        if let Some(set) = s.get("set") {
            let st = self.settings.as_ref().expect("settings for a change");
            st.update(|l| apply_settings(l, set));
        }
        let at = START + s["at"].as_f64().unwrap();
        self.clock.at_least(at);
        if let Some(w) = s.get("waiting") {
            return Some(Event::Waiting(Waiting {
                since: w[0].as_f64().unwrap(),
                what: w[1].as_str().unwrap().into(),
                why: w[2].as_str().unwrap_or("").into(),
            }));
        }
        let tag = json!({"tag": s["tag"].as_f64()});
        let (w, h) = self.size;
        let frame = if self.tag_probe {
            tagged(tag)
        } else {
            let mut c = s["c"].clone();
            if c.get("kind").and_then(|k| k.as_str()) == Some("menu") {
                let open = self.remote.as_ref().is_some_and(|r| r.lock().unwrap().menu_open(at));
                c["open"] = json!(open);
                push(&self.ev, json!(["menu", open]));
            }
            with_tag(Frame::from_bgr(&content(&c, w, h), w, h), tag)
        };
        let mut t = Tick::new(frame, at, s.get("index").and_then(|v| v.as_u64()).unwrap_or(i as u64));
        t.content = s.get("content").and_then(|v| v.as_i64());
        t.truth_new = s.get("truth").and_then(|v| v.as_bool());
        t.new_hint = s.get("hint").and_then(|v| v.as_bool());
        t.damaged = s.get("damaged").and_then(|v| v.as_bool()).unwrap_or(false);
        t.t_ready = s.get("ready").and_then(|v| v.as_f64()).map(|r| START + r);
        if let Some(e) = s.get("extra").and_then(|e| e.as_object()) {
            t.extra = e.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<BTreeMap<_, _>>();
        }
        Some(Event::Tick(t))
    }
}

/// "Simulate a 30 fps game" over any 60 Hz source, following the setting (framegen/sources.py TestMode).
pub struct TestMode<S: Source> {
    pub inner: S,
    pub settings: Arc<SharedLive>,
    pub on: bool,
    n: i64,
    last: Option<Frame>,
    truth: HashMap<i64, Frame>,
    game: HashMap<i64, Option<bool>>,
}

impl<S: Source> TestMode<S> {
    pub const KEEP: i64 = 8;

    pub fn new(inner: S, settings: Arc<SharedLive>) -> TestMode<S> {
        let on = settings.live().simulate_30;
        TestMode { inner, settings, on, n: 0, last: None, truth: HashMap::new(), game: HashMap::new() }
    }

    pub fn set_on(&mut self, on: bool) {
        if on && !self.on {
            self.last = None;
            self.truth.clear();
            self.game.clear();
        }
        self.on = on;
    }

    fn follow(&mut self) {
        let want = self.settings.live().simulate_30;
        if want != self.on {
            self.set_on(want);
        }
    }

    pub fn gt(&self, content: i64) -> Option<Frame> {
        if (content - 1..=content + 1).any(|i| self.game.get(&i) == Some(&Some(false))) {
            return None;
        }
        self.truth.get(&content).cloned()
    }

    pub fn apply(&mut self, mut tick: Tick) -> Option<Tick> {
        if !self.on {
            return Some(tick);
        }
        let i = self.n;
        self.n += 1;
        let game_new = match tick.extra.get("size_new") {
            Some(v) => v.as_bool(),
            None => tick.new_hint,
        };
        self.game.insert(i, game_new);
        tick.extra.insert("game_new".into(), game_new.map_or(Value::Null, Value::Bool));
        self.game.remove(&(i - Self::KEEP));
        if i % 2 == 0 {
            self.last = Some(tick.frame.clone());
            tick.content = Some(i);
            tick.truth_new = Some(true);
            tick.new_hint = Some(true);
        } else if let Some(last) = &self.last {
            self.truth.insert(i, tick.frame.clone());
            self.truth.remove(&(i - Self::KEEP));
            tick.frame = last.clone();
            tick.content = Some(i - 1);
            tick.truth_new = Some(false);
            tick.new_hint = Some(false);
        } else {
            return None; // switched on at an odd frame, so there's nothing to repeat yet
        }
        Some(tick)
    }
}

impl<S: Source> Source for TestMode<S> {
    fn info(&self) -> &SourceInfo {
        self.inner.info()
    }
    fn size(&self) -> (usize, usize) {
        self.inner.size()
    }
    fn next_event(&mut self) -> Option<Event> {
        loop {
            // The Python's settings.watch switches it the moment the setting changes. Here it follows the setting
            // before the next frame (the pipeline turned it off) and after the inner source's own changes.
            self.follow();
            let ev = self.inner.next_event();
            self.follow();
            match ev? {
                Event::Waiting(w) => return Some(Event::Waiting(w)),
                Event::Tick(t) => {
                    if let Some(t) = self.apply(t) {
                        return Some(Event::Tick(t));
                    }
                }
            }
        }
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.gt(content)
    }
}

// ------------------------------------------------------------------------------------------------ interpolator
pub struct TraceInterp {
    pub clock: Arc<FakeClock>,
    pub ev: Events,
    pub blend8: bool,
    pub cost_interp: f64,
    pub cost_prepare: f64,
    pub size: (usize, usize),
    input: Option<Frame>,
    slots: [Option<f64>; 2],
    px: [Option<Frame>; 2],
}

impl TraceInterp {
    pub fn new(clock: Arc<FakeClock>, ev: Events, cfg: &Value, size: (usize, usize)) -> TraceInterp {
        let f = |k: &str| cfg.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
        TraceInterp {
            clock,
            ev,
            blend8: cfg.get("kind").and_then(|v| v.as_str()) == Some("blend8"),
            cost_interp: f("cost_interp"),
            cost_prepare: f("cost_prepare"),
            size,
            input: None,
            slots: [None, None],
            px: [None, None],
        }
    }
}

impl Interpolator for TraceInterp {
    fn label(&self) -> String {
        "RIFE".into()
    }
    fn size(&self) -> (usize, usize) {
        self.size
    }
    fn load(&mut self, frame: &Frame) {
        self.input = Some(frame.clone());
    }
    fn prepare(&mut self, slot: usize) {
        self.slots[slot] = self.input.as_ref().and_then(tag_value);
        self.px[slot] = self.input.clone();
        self.clock.advance(self.cost_prepare);
    }
    fn interpolate(&mut self, newest: usize, t: f32) -> Frame {
        let (a, b) = (self.slots[1 - newest], self.slots[newest]);
        self.clock.advance(self.cost_interp);
        let tag = json!({"ai": [a, b, t as f64]});
        match (&self.px[1 - newest], &self.px[newest]) {
            (Some(pa), Some(pb)) if self.blend8 => {
                let (va, vb) = (pa.to_bgr(), pb.to_bgr());
                let out: Vec<u8> = va.iter().zip(&vb).map(|(&x, &y)| (x as u32 + y as u32).div_ceil(2) as u8).collect();
                with_tag(Frame::from_bgr(&out, pa.width, pa.height), tag)
            }
            _ => tagged(tag),
        }
    }
    fn set_steady(&mut self, threshold: f32) {
        push(&self.ev, json!(["steady", threshold as f64]));
    }
}

// ------------------------------------------------------------------------------------------------ presenter
pub fn overlay_json(o: &Overlay) -> Value {
    json!({"text": o.text, "parts": o.parts.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
           "health": o.health, "second": o.second, "minimal": o.minimal})
}

pub fn reply(cmd: Option<&str>) -> SinkReply {
    match cmd {
        Some("quit") => SinkReply::Quit,
        Some("restart") => SinkReply::Restart,
        Some("save_replay") => SinkReply::SaveReplay,
        Some("save_compare") => SinkReply::SaveCompare,
        Some("delay_test") => SinkReply::DelayTest,
        None => SinkReply::Continue,
        Some(other) => panic!("no SinkReply for {other}"),
    }
}

pub struct TraceSink {
    pub clock: Arc<FakeClock>,
    pub ev: Events,
    pub cost_write: f64,
    pub cmds: HashMap<usize, String>,
    pub wait_cmds: HashMap<usize, String>,
    pub n: usize,
    pub nw: usize,
}

impl TraceSink {
    pub fn new(clock: Arc<FakeClock>, ev: Events, cfg: &Value) -> TraceSink {
        let map = |k: &str| -> HashMap<usize, String> {
            cfg.get(k)
                .and_then(|m| m.as_object())
                .map(|m| m.iter().map(|(k, v)| (k.parse().unwrap(), v.as_str().unwrap().to_string())).collect())
                .unwrap_or_default()
        };
        TraceSink {
            clock,
            ev,
            cost_write: cfg.get("cost_write").and_then(|v| v.as_f64()).unwrap_or(0.0),
            cmds: map("cmds"),
            wait_cmds: map("wait_cmds"),
            n: 0,
            nw: 0,
        }
    }
}

impl Presenter for TraceSink {
    fn write(&mut self, frame: &Frame, overlay: Option<&Overlay>) -> SinkReply {
        let i = self.n;
        self.n += 1;
        self.clock.advance(self.cost_write);
        push(&self.ev, json!(["write", self.clock.get(), tag_of(frame), overlay.map(overlay_json)]));
        reply(self.cmds.get(&i).map(String::as_str))
    }
    fn wait(&mut self, w: &Waiting) -> SinkReply {
        let i = self.nw;
        self.nw += 1;
        push(&self.ev, json!(["wait", w.since, w.what, w.why]));
        reply(self.wait_cmds.get(&i).map(String::as_str))
    }
    fn toast(&mut self, text: &str, seconds: f64) {
        push(&self.ev, json!(["toast", text, seconds]));
    }
}

// ------------------------------------------------------------------------------------------------ instant replay
pub struct TraceReplay {
    pub ev: Events,
    pub frames: u64,
    pub real: u64,
}

impl Replay for TraceReplay {
    fn set_output_hz(&mut self, hz: u32) {
        push(&self.ev, json!(["replay_hz", hz]));
    }
    fn set_delay(&mut self, ms: f64) {
        push(&self.ev, json!(["replay_delay", ms]));
    }
    fn frame(&mut self, _frame: &Frame, _t: f64) {
        self.frames += 1;
    }
    fn real_frame(&mut self, _frame: &Frame, _t: f64) {
        self.real += 1;
    }
    fn save(&mut self, kind: Option<&str>) {
        push(&self.ev, json!(["replay_save", kind]));
    }
}

// ------------------------------------------------------------------------------------------------ simple ones
/// A source from a closure (the Python tests' generators).
pub struct FnSource<F: FnMut() -> Option<Event> + Send> {
    pub f: F,
    info: SourceInfo,
}

impl<F: FnMut() -> Option<Event> + Send> FnSource<F> {
    pub fn new(f: F) -> Self {
        FnSource { f, info: SourceInfo::of("sim") }
    }
}

impl<F: FnMut() -> Option<Event> + Send> Source for FnSource<F> {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (8, 8)
    }
    fn next_event(&mut self) -> Option<Event> {
        (self.f)()
    }
}

/// The tests' _Sink. It records what was written (the frame's tag value, or the AI frame's blend of its two tags),
/// the overlay and the toasts, and returns `cmds` in turn.
#[derive(Default)]
pub struct RecSink {
    pub values: Vec<f64>,
    pub tags: Vec<Value>,
    pub texts: Vec<Option<Overlay>>,
    pub toasts: Vec<String>,
    pub cmds: Vec<SinkReply>,
    pub waits: usize,
    pub wait_quit_at: Option<usize>,
}

/// The value an output frame shows: a real frame's tag, or a + t (b - a) for an AI frame of tags a and b.
pub fn shown_value(tag: &Value) -> f64 {
    if let Some(v) = tag.get("tag") {
        return v.as_f64().unwrap_or(f64::NAN);
    }
    let ai = &tag["ai"];
    let (a, b, t) = (ai[0].as_f64().unwrap_or(0.0), ai[1].as_f64().unwrap_or(0.0), ai[2].as_f64().unwrap());
    a + t * (b - a)
}

impl Presenter for RecSink {
    fn write(&mut self, frame: &Frame, overlay: Option<&Overlay>) -> SinkReply {
        let tag = tag_of(frame);
        self.values.push(shown_value(&tag));
        self.tags.push(tag);
        self.texts.push(overlay.cloned());
        self.cmds.pop().unwrap_or(SinkReply::Continue)
    }
    fn wait(&mut self, _w: &Waiting) -> SinkReply {
        self.waits += 1;
        if Some(self.waits) == self.wait_quit_at { SinkReply::Quit } else { SinkReply::Continue }
    }
    fn toast(&mut self, text: &str, _seconds: f64) {
        self.toasts.push(text.to_string());
    }
}

/// The tests' stand-in interpolator. It blends the tags of its two slots, so an output frame's value says which game
/// time it shows, and costs `cost` seconds on the clock.
pub struct BlendInterp {
    pub clock: Arc<FakeClock>,
    pub cost: f64,
    input: Option<Frame>,
    slots: [Option<f64>; 2],
    pub single: Vec<f32>,
    pub steady: Vec<f32>,
}

impl BlendInterp {
    pub fn new(clock: Arc<FakeClock>) -> Self {
        BlendInterp { clock, cost: 0.0, input: None, slots: [None, None], single: Vec::new(), steady: Vec::new() }
    }
}

impl Interpolator for BlendInterp {
    fn label(&self) -> String {
        "RIFE".into()
    }
    fn size(&self) -> (usize, usize) {
        (8, 8)
    }
    fn load(&mut self, frame: &Frame) {
        self.input = Some(frame.clone());
    }
    fn prepare(&mut self, slot: usize) {
        self.slots[slot] = self.input.as_ref().and_then(tag_value);
    }
    fn interpolate(&mut self, newest: usize, t: f32) -> Frame {
        self.single.push(t);
        self.clock.advance(self.cost);
        tagged(json!({"ai": [self.slots[1 - newest], self.slots[newest], t as f64]}))
    }
    fn set_steady(&mut self, threshold: f32) {
        self.steady.push(threshold);
    }
}

/// A tick of an 8x8 frame whose tag (and, for the tag probe, picture) is `v`.
pub fn tag_tick(v: f64, t: f64, i: u64, hint: Option<bool>) -> Tick {
    let mut tick = Tick::new(tagged(json!({"tag": v})), t, i);
    tick.new_hint = hint;
    tick
}
