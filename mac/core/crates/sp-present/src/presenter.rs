//! The video window as the pipeline sees it. [`VideoPresenter`] implements sp_core::Presenter, which is
//! framegen/io.py GLWindowSink's write, wait, toast and close, plus busy() and baked(). It draws on whatever thread
//! calls it, while the window itself stays on the main thread (crate::window).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::MainThreadMarker;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDrawable, MTLPixelFormat, MTLTexture};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use sp_core::clock;
use sp_core::{Frame, Gpu, Overlay, PixelKind, Presenter, Result, SinkReply, Waiting};

use crate::display::{DisplaySettings, MenuModel, Pacing};
use crate::fill::Rect;
use crate::gpu::{self, Blend, Load, Tex};
use crate::looks::LutStore;
use crate::looks_gpu::LookPlan;
use crate::menu::{Menu, MenuResult, Nav};
use crate::renderer::{Drawn, Panel, Picture, Renderer};
use crate::ui::{self, MenuDrawer};
use crate::window::{self, Key, KeyPress, ScreenInfo, Shared, WindowCmd};

/// Shown when the window opens, and again when the mouse moves over it (io.py KEYS_HINT).
pub const KEYS_HINT: &str = "M: menu    F: fullscreen    Esc: quit";

/// How the window opens.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    /// CAMetalLayer maximumDrawableCount. 3 is Apple's default and the smoothest. 2 gives a refresh less delay when
    /// the GPU keeps up.
    pub drawables: usize,
    /// For HDR10 output. `hdr_white_nits` is the PQ level that is SDR white, which macOS shows at the desktop's white
    /// (203 nits in BT.2408, hdr.py SDR_WHITE_NITS). `hdr_peak_nits` is the brightest the content goes (hdr.py
    /// SOURCE_PEAK_NITS), for EDR's roll-off.
    pub hdr_white_nits: f32,
    pub hdr_peak_nits: f32,
    /// The user's looks folder. LUTs are read from its "luts" folder.
    pub looks_folder: Option<PathBuf>,
    /// Print "[display]" lines on stdout, like the Python window.
    pub log: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig { title: "SmoovPlay".into(), drawables: 3, hdr_white_nits: 203.0, hdr_peak_nits: 1000.0,
                       looks_folder: None, log: true }
    }
}

/// When each frame was handed over and when it reached the screen, and the GPU time of each frame's drawing.
#[derive(Default)]
pub struct Timing {
    gpu: Mutex<BTreeMap<String, Vec<f64>>>,
    /// The GPU's last 64 frame times in ms.
    gpu_recent: Mutex<std::collections::VecDeque<f64>>,
    /// (written at, presented at) on sp_core::clock, which is CACurrentMediaTime's clock. Presented at is 0 if it
    /// never was.
    presents: Mutex<Vec<(f64, f64)>>,
}

/// Percentiles of a list (ms).
#[derive(Clone, Debug, Default)]
pub struct Spread {
    pub n: usize,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

impl Spread {
    pub fn of(v: &[f64]) -> Spread {
        if v.is_empty() {
            return Spread::default();
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
        Spread { n: s.len(), p50: q(0.5), p95: q(0.95), p99: q(0.99), max: s[s.len() - 1], mean: s.iter().sum::<f64>() / s.len() as f64 }
    }
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "p50 {:.2} / p95 {:.2} / p99 {:.2} / max {:.2} ms (n={})", self.p50, self.p95, self.p99, self.max, self.n)
    }
}

/// How frames reached the screen.
#[derive(Clone, Debug, Default)]
pub struct PresentStats {
    pub frames: usize,
    /// Frames the display never showed (a newer one replaced them first).
    pub dropped: usize,
    /// Time between consecutive frames shown (presentedTime), in ms.
    pub shown: Spread,
    /// Time between consecutive write() calls, in ms.
    pub written: Spread,
    /// Time from write() to the frame being on screen, in ms.
    pub latency: Spread,
    /// The shown intervals themselves in ms, in order.
    pub intervals: Vec<f64>,
}

impl Timing {
    pub fn gpu(&self) -> BTreeMap<String, Spread> {
        self.gpu.lock().unwrap().iter().map(|(k, v)| (k.clone(), Spread::of(v))).collect()
    }

    /// Present timing over the frames since `from` (an index into the frames written so far).
    pub fn presents(&self, from: usize) -> PresentStats {
        let p = self.presents.lock().unwrap();
        let p = &p[from.min(p.len())..];
        let shown: Vec<f64> = p.iter().filter(|x| x.1 > 0.0).map(|x| x.1).collect();
        let intervals: Vec<f64> = shown.windows(2).map(|w| (w[1] - w[0]) * 1e3).collect();
        let written: Vec<f64> = p.windows(2).map(|w| (w[1].0 - w[0].0) * 1e3).collect();
        let latency: Vec<f64> = p.iter().filter(|x| x.1 > 0.0).map(|x| (x.1 - x.0) * 1e3).collect();
        PresentStats {
            frames: p.len(),
            dropped: p.iter().filter(|x| x.1 == 0.0).count(),
            shown: Spread::of(&intervals),
            written: Spread::of(&written),
            latency: Spread::of(&latency),
            intervals,
        }
    }

    pub fn frames(&self) -> usize {
        self.presents.lock().unwrap().len()
    }
}

struct LayerRef(Retained<CAMetalLayer>);
// CAMetalLayer's nextDrawable may be called from any thread, and Apple's own samples render on a display-link thread.
// Every property change goes through the main thread (crate::window::WindowCmd).
unsafe impl Send for LayerRef {}

/// The video window, for the pipeline. It implements [`Presenter`]. [`open`] makes it on the main thread, and after
/// that any one thread at a time can use it (it's Send).
pub struct VideoPresenter {
    shared: Arc<Shared>,
    layer: LayerRef,
    renderer: Renderer,
    model: Box<dyn MenuModel>,
    cfg: WindowConfig,
    /// The stream's size.
    w: usize,
    h: usize,
    pub display: DisplaySettings,
    rev: Option<u64>,
    menu: Menu,
    drawer: MenuDrawer,
    menu_panel: Panel,
    menu_state: Option<String>,
    menu_top: Option<(i64, Rect)>,
    toast_panel: Panel,
    toast_state: Option<String>,
    toast_text: String,
    toast_until: f64,
    stats_panel: Panel,
    stats_state: Option<String>,
    detail_w: usize,
    detail_ui: i64,
    stats_shown: bool,
    hint_next: f64,
    waited: i64,
    cmd: Option<SinkReply>,
    last: Option<Frame>,
    frame_no: u64,
    layer_cfg: Option<(bool, bool, usize)>,
    fullscreen: Option<bool>,
    screen: ScreenInfo,
    screen_at: f64,
    visible: bool,
    timing: Arc<Timing>,
    last_label: String,
    slot_at: f64,
    steady: bool,
    closed: bool,
}

/// Opens the video window for a stream of `w` x `h` frames. Call it on the main thread. `model` is the settings, for
/// the menu, hotkeys and the live display settings. The presenter then draws from whichever thread the pipeline runs
/// on. The crate's docs cover driving the event loop.
pub fn open(mtm: MainThreadMarker, w: usize, h: usize, cfg: WindowConfig, model: Box<dyn MenuModel>) -> Result<VideoPresenter> {
    let shared = Arc::new(Shared::default());
    let display = model.display();
    let layer = window::open(mtm, w, h, &cfg.title, cfg.drawables, display.vsync, shared.clone());
    let luts = Arc::new(LutStore::new(cfg.looks_folder.as_ref().map(|f| f.join("luts"))));
    let renderer = Renderer::new(luts)?;
    let screen = shared.inbox.lock().unwrap().screen.clone();
    let mut p = VideoPresenter {
        shared,
        layer: LayerRef(layer),
        renderer,
        model,
        w,
        h,
        display,
        rev: None,
        menu: Menu::new(),
        drawer: MenuDrawer::default(),
        menu_panel: Panel::new(),
        menu_state: None,
        menu_top: None,
        toast_panel: Panel::new(),
        toast_state: None,
        toast_text: KEYS_HINT.into(),
        toast_until: clock::now() + 8.0,
        stats_panel: Panel::new(),
        stats_state: None,
        detail_w: 0,
        detail_ui: 0,
        stats_shown: false,
        hint_next: 0.0,
        waited: -1,
        cmd: None,
        last: None,
        frame_no: 0,
        layer_cfg: None,
        fullscreen: None,
        screen,
        screen_at: 0.0,
        visible: true,
        timing: Arc::new(Timing::default()),
        last_label: String::new(),
        slot_at: 0.0,
        steady: false,
        closed: false,
        cfg,
    };
    p.sync_settings();
    if p.cfg.log {
        let s = &p.screen;
        println!(
            "[display] Metal window on {} ({:.0}x{:.0} pt, x{}), {} refresh{}, EDR headroom {:.2} (up to {:.2}); upscaling with {}; m opens the settings menu",
            s.name, s.size.0, s.size.1, s.scale,
            if s.variable_refresh() { format!("variable {:.0}-{:.0} Hz", 1.0 / s.refresh_max, 1.0 / s.refresh_min) } else { format!("{:.0} Hz", 1.0 / s.refresh_min.max(1e-6)) },
            if s.variable_refresh() { "" } else { " (fixed)" },
            s.edr_now, s.edr_potential, p.display.scaler.name()
        );
    }
    Ok(p)
}

/// Runs the main thread's event loop until the video window closes. It's for a pipeline on another thread when
/// nothing else, such as a Swift app, runs one. Call it on the main thread after [`open`], once the presenter has moved
/// to the pipeline's thread. It returns once the presenter has been closed.
pub fn run(mtm: MainThreadMarker) {
    window::run(mtm);
}

fn nav_of(k: Key) -> Option<Nav> {
    Some(match k {
        Key::Up => Nav::Up,
        Key::Down => Nav::Down,
        Key::Left => Nav::Left,
        Key::Right => Nav::Right,
        Key::Enter => Nav::Enter,
        Key::Esc | Key::Backspace => Nav::Back, // Esc goes back a page, and on the main page it closes the menu
        _ => return None,
    })
}

impl VideoPresenter {
    /// The screen the window is on (refresh, EDR headroom), as last seen.
    pub fn screen(&self) -> &ScreenInfo {
        &self.screen
    }

    /// The window's number (for `screencapture -l`).
    pub fn window_number(&self) -> isize {
        self.shared.inbox.lock().unwrap().window_number
    }

    pub fn timing(&self) -> Arc<Timing> {
        self.timing.clone()
    }

    /// What drew the last frame ("FSR 1 3200x1800 + look").
    pub fn label(&self) -> &str {
        &self.last_label
    }

    /// Whether the menu is open, for tests and the example.
    pub fn menu_open(&self) -> bool {
        self.menu.open
    }

    /// Presses a key as if on the keyboard. The launcher and tests use it.
    pub fn press(&mut self, key: Key, shift: bool) {
        self.on_key(KeyPress { key, shift, command: false, repeat: false });
    }

    /// The pacing in effect. Auto with vsync on means Steady while a frame's lead (a refresh plus the GPU's time) fits
    /// in one output frame, and Immediate otherwise, because holding frames for their slots would run the layer out of
    /// drawables. Auto with vsync off means Immediate.
    pub fn pacing(&self) -> Pacing {
        match self.display.pacing {
            Pacing::Auto if self.display.vsync => {
                let period = 1.0 / self.display.hz.max(1.0);
                let margin = if self.steady { 0.0005 } else { -0.0005 }; // a little hysteresis
                if self.lead() <= period + margin { Pacing::Steady } else { Pacing::Immediate }
            }
            Pacing::Auto => Pacing::Immediate,
            p => p,
        }
    }

    /// How far ahead of now Steady presents. It's a refresh plus the GPU's recent time (90th percentile of the last
    /// 64 frames, at least 2 ms). SP_PRESENT_LEAD_MS overrides it.
    fn lead(&self) -> f64 {
        let period = 1.0 / self.display.hz.max(1.0);
        let refresh = if self.screen.refresh_min > 0.0 { self.screen.refresh_min } else { 1.0 / 60.0 };
        if let Some(ms) = std::env::var("SP_PRESENT_LEAD_MS").ok().and_then(|v| v.parse::<f64>().ok()) {
            return ms / 1e3;
        }
        let gpu = {
            let mut v: Vec<f64> = self.timing.gpu_recent.lock().unwrap().iter().copied().collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v.get(v.len() * 9 / 10).copied().unwrap_or(2.0) / 1e3
        };
        refresh + (1.2 * gpu + 0.0005).clamp(0.002, period)
    }

    /// The next slot on the steady grid for a frame written at `now` (Pacing::Steady).
    fn slot(&mut self, now: f64) -> f64 {
        let period = 1.0 / self.display.hz.max(1.0);
        let lead = self.lead();
        let next = self.slot_at + period;
        self.slot_at = if next < now + lead * 0.5 || next > now + lead + period {
            now + lead // late, or after a pause, so start the grid again from here
        } else {
            next
        };
        self.slot_at
    }

    // ---- settings
    fn sync_settings(&mut self) {
        let rev = self.model.revision();
        if self.rev == Some(rev) {
            return;
        }
        self.rev = Some(rev);
        self.display = self.model.display();
        if self.fullscreen != Some(self.display.fullscreen) {
            if self.fullscreen.is_some() || self.display.fullscreen {
                window::on_main(WindowCmd::Fullscreen(self.display.fullscreen));
            }
            self.fullscreen = Some(self.display.fullscreen);
        }
    }

    fn ensure_layer(&mut self, hdr: bool) {
        let want = (hdr, self.display.vsync, self.cfg.drawables);
        if self.layer_cfg != Some(want) {
            if self.cfg.log && self.layer_cfg.map(|c| c.0) != Some(hdr) {
                println!("[display] {}", if hdr {
                    format!("HDR10 frames: RGBA16F drawable, PQ (BT.2100) colour space, EDR on; SDR white at {} nits; headroom now {:.2}", self.cfg.hdr_white_nits, self.screen.edr_now)
                } else {
                    "SDR frames: BGRA8 drawable, sRGB".to_string()
                });
            }
            self.layer_cfg = Some(want);
            window::on_main(WindowCmd::Layer { hdr, vsync: want.1, drawables: want.2, white_nits: self.cfg.hdr_white_nits,
                                               peak_nits: self.cfg.hdr_peak_nits });
        }
    }

    fn say(&mut self, key: &str) {
        if let Some(r) = self.model.row(key) {
            self.toast(&format!("{}: {}", r.label, r.value), 2.0);
        }
    }

    fn get_bool(&self, key: &str) -> bool {
        self.model.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    // ---- keys (io.py _on_key)
    fn on_key(&mut self, k: KeyPress) {
        let nav = nav_of(k.key);
        if k.repeat && !(self.menu.open && nav.is_some()) {
            return;
        }
        if self.menu.open
            && let Some(n) = nav {
                match self.menu.key(&mut *self.model, n) {
                    Some(MenuResult::Close) => self.menu.open = false,
                    Some(MenuResult::Restart) => {
                        self.menu.open = false;
                        self.cmd = Some(SinkReply::Restart);
                    }
                    None => {}
                }
                self.sync_settings();
                return;
            }
        if k.command {
            if k.key == Key::Char('q') {
                self.cmd = Some(SinkReply::Quit); // Cmd+Q
            }
            return;
        }
        let m = &mut self.model;
        match k.key {
            Key::Char('m') | Key::Tab => {
                self.menu.open = !self.menu.open;
                self.toast_until = 0.0;
            }
            Key::Char('q') | Key::Esc => self.cmd = Some(SinkReply::Quit),
            Key::Char('i') => {
                let on = self.get_bool("interp");
                self.model.set("interp", (!on).into(), false); // a quick comparison, not saved
                self.say("interp");
            }
            Key::Char('s') => {
                // Off, then Minimal, then Detailed, then Off again
                let style = m.get("overlay_style").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
                if !self.get_bool("overlay") {
                    self.model.set("overlay_style", "minimal".into(), true);
                    self.model.set("overlay", true.into(), true);
                    self.say("overlay");
                } else if style == "minimal" {
                    self.model.set("overlay_style", "detailed".into(), true);
                    self.say("overlay_style");
                } else {
                    self.model.set("overlay", false.into(), true);
                    self.say("overlay");
                }
            }
            Key::Char('u') => {
                m.step("scaler", 1);
                self.say("scaler");
            }
            Key::Char('z') => {
                m.step("fill", 1);
                self.sync_settings();
                let fill = self.display.fill;
                self.toast(&format!("Fill the screen: {}{}", fill.name(),
                    if fill == crate::fill::Fill::Fit { "" } else { "\nThe same 16:9 picture: consoles send 16:9 only" }), 3.0);
            }
            Key::Char('k') => {
                let on = self.get_bool("look_on");
                self.model.set("look_on", (!on).into(), true);
                self.say_look();
            }
            Key::Char('j') => {
                if !self.get_bool("look_on") {
                    self.model.set("look_on", true.into(), true);
                } else {
                    self.model.step("look", if k.shift { -1 } else { 1 });
                }
                self.say_look();
            }
            Key::Char('f') => {
                let on = self.get_bool("fullscreen");
                self.model.set("fullscreen", (!on).into(), true);
                self.say("fullscreen");
            }
            Key::Char('l') => self.cmd = Some(SinkReply::DelayTest), // the pipeline runs it (PS5 Remote Play only)
            Key::Char('r') => self.cmd = Some(SinkReply::SaveReplay),
            Key::Char('b') => self.cmd = Some(SinkReply::SaveCompare),
            _ => {}
        }
        self.sync_settings();
    }

    fn say_look(&mut self) {
        self.sync_settings();
        let name = self.display.look.as_ref().map_or("Off".to_string(), |l| l.name.clone());
        self.toast(&format!("Look: {name}"), 2.0);
    }

    /// Handles the window's events. It pumps them here when this is the main thread, then acts on the keys, mouse and
    /// close it saw.
    fn events(&mut self) {
        if let Some(mtm) = MainThreadMarker::new() {
            window::pump(mtm);
        }
        let (keys, moved, close, visible) = {
            let mut inbox = self.shared.inbox.lock().unwrap();
            let keys = std::mem::take(&mut inbox.keys);
            let r = (keys, inbox.moved, inbox.close, inbox.visible);
            inbox.moved = false;
            inbox.close = false;
            r
        };
        self.visible = visible;
        for k in keys {
            self.on_key(k);
        }
        let now = clock::now();
        if moved && now >= self.hint_next && !self.menu.open && now >= self.toast_until {
            self.hint_next = now + 20.0;
            self.toast(KEYS_HINT, 3.0);
        }
        if close {
            self.cmd = Some(SinkReply::Quit);
        }
        if now >= self.screen_at {
            self.screen_at = now + 1.0;
            window::on_main(WindowCmd::RefreshScreen);
            self.screen = self.shared.inbox.lock().unwrap().screen.clone();
        }
        self.sync_settings();
    }

    // ---- drawing
    fn stats_ui(&self, dh: i64) -> f64 {
        0.6f64.max(0.5f64.max(dh as f64 / 1080.0) * self.display.overlay_size.factor())
    }

    /// Draws one frame to the screen: the picture (or black), the stats, and the menu or a message.
    fn render(&mut self, frame: Option<&Frame>, overlay: Option<&Overlay>, plain: bool) {
        if !self.visible {
            return; // minimised or hidden, so there's nothing to show and the layer may not hand out drawables
        }
        let hdr = frame.is_some_and(|f| f.kind == PixelKind::Rgba16f);
        self.ensure_layer(hdr);
        let plan: Option<Arc<LookPlan>> = match (&self.display.look, hdr) {
            (Some(look), false) if !plain => {
                let luts = self.renderer.luts.clone();
                self.renderer.looks.plan(look, &luts)
            }
            _ => None,
        };
        let t_write = clock::now();
        autoreleasepool(|_| {
            let Some(drawable) = self.layer.0.nextDrawable() else { return };
            let target = drawable.texture();
            let Some(cmd) = Gpu::shared().queue.commandBuffer() else { return };
            let hdr_target = target.pixelFormat() == MTLPixelFormat::RGBA16Float;
            self.frame_no += 1;
            let drawn = match frame {
                Some(f) => {
                    let pic = Picture {
                        src: &f.texture,
                        src_size: (f.width, f.height),
                        scaler: self.display.scaler,
                        sharpness: self.display.fsr_sharpness,
                        fill: self.display.fill,
                        look: plan.as_deref(),
                        frame_no: self.frame_no,
                        hdr,
                        plain,
                    };
                    match self.renderer.draw(&cmd, &pic, &target) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("[display] {e}");
                            return;
                        }
                    }
                }
                None => {
                    if let Ok(ps) = self.renderer.pipes.get("bilinear", target.pixelFormat(), Blend::Replace) {
                        gpu::pass(&cmd, &target, Load::Clear, &ps, Rect::new(0, 0, 0, 0), &[], &[]);
                    }
                    let r = gpu::whole(&target);
                    Drawn { ui: r, picture: r, name: "nothing".into(), label: "nothing".into() }
                }
            };
            if let Some(o) = overlay.filter(|_| self.display.overlay) {
                self.draw_stats(&cmd, &target, o, &drawn, hdr_target);
            } else {
                self.stats_shown = false;
            }
            self.draw_ui(&cmd, &target, drawn.ui, hdr_target);
            self.last_label = drawn.label.clone();

            let timing = self.timing.clone();
            let label = drawn.label.clone();
            let done = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let cb = unsafe { cb.as_ref() };
                let ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
                if ms > 0.0 {
                    timing.gpu.lock().unwrap().entry(label.clone()).or_default().push(ms);
                    let mut r = timing.gpu_recent.lock().unwrap();
                    if r.len() == 64 {
                        r.pop_front();
                    }
                    r.push_back(ms);
                }
            });
            unsafe { cmd.addCompletedHandler(RcBlock::as_ptr(&done)) };
            let idx = {
                let mut p = self.timing.presents.lock().unwrap();
                p.push((t_write, 0.0));
                p.len() - 1
            };
            let timing = self.timing.clone();
            let shown = RcBlock::new(move |d: NonNull<ProtocolObject<dyn MTLDrawable>>| {
                let t = unsafe { d.as_ref() }.presentedTime();
                timing.presents.lock().unwrap()[idx].1 = t;
            });
            let d: &ProtocolObject<dyn MTLDrawable> = ProtocolObject::from_ref(&*drawable);
            unsafe { d.addPresentedHandler(RcBlock::as_ptr(&shown)) };
            let pacing = self.pacing();
            self.steady = pacing == Pacing::Steady;
            match pacing {
                Pacing::MinDuration => cmd.presentDrawable_afterMinimumDuration(d, 1.0 / self.display.hz.max(1.0)),
                Pacing::Steady => {
                    let at = self.slot(t_write);
                    cmd.presentDrawable_atTime(d, at)
                }
                _ => cmd.presentDrawable(d),
            }
            cmd.commit();
        });
    }

    /// The stats overlay in the corner picked, tucked into its edges (io.py _draw_stats). Minimal draws the frame's
    /// parts. Detailed draws its text, with the upscaler on the first line. It's drawn again only when it changes.
    fn draw_stats(&mut self, cmd: &gpu::Cmd, target: &Tex, o: &Overlay, drawn: &Drawn, hdr: bool) {
        let ui_rect = drawn.ui;
        let ui = self.stats_ui(ui_rect.h);
        let uik = (ui * 1000.0).round() as i64;
        if o.minimal {
            let state = format!("mini|{:?}|{}|{}|{uik}", o.parts, o.health, o.second);
            if self.stats_state.as_ref() != Some(&state) {
                let img = ui::mini(&o.parts, &o.health, &o.second, ui);
                self.stats_panel.update(&img.bgra, img.w, img.h);
                self.stats_state = Some(state);
            }
        } else {
            // The upscaler goes on the first line. A second line is the advice or the test line.
            let (first, rest) = o.text.split_once('\n').map_or((o.text.as_str(), None), |(a, b)| (a, Some(b)));
            let text = match rest {
                Some(r) => format!("{first}  {}\n{r}", drawn.name),
                None => format!("{first}  {}", drawn.name),
            };
            let state = format!("detail|{text}|{uik}");
            if self.stats_state.as_ref() != Some(&state) {
                if self.detail_ui != uik {
                    self.detail_w = 0;
                    self.detail_ui = uik;
                }
                let img = ui::detail(&text, ui, self.detail_w);
                self.detail_w = img.w;
                self.stats_panel.update(&img.bgra, img.w, img.h);
                self.stats_state = Some(state);
            }
        }
        self.stats_shown = true;
        let (pw, ph) = (self.stats_panel.w as i64, self.stats_panel.h as i64);
        let m = (4.0 * ui).floor().max(2.0) as i64;
        let c = self.display.overlay_corner;
        let x = if c.right() { ui_rect.x + ui_rect.w - pw - m } else { ui_rect.x + m };
        let y = if c.bottom() { ui_rect.y + ui_rect.h - ph - m } else { ui_rect.y + m };
        let _ = self.renderer.panel(cmd, target, &self.stats_panel, x, y, hdr, self.cfg.hdr_white_nits);
    }

    /// The menu, or a message, on top of everything else (io.py _draw_ui). Images are redrawn only when they change.
    fn draw_ui(&mut self, cmd: &gpu::Cmd, target: &Tex, r: Rect, hdr: bool) {
        let ui = 0.5f64.max(r.h as f64 / 1080.0);
        if !self.menu.open {
            self.menu_top = None;
        }
        if self.menu.open {
            let rows = self.menu.rows(&*self.model);
            self.menu.follow(&rows);
            let state = format!("{:?}|{:?}|{}|{}|{:?}|{:.3}", self.menu.page, self.menu.index, self.model.revision(),
                                rows.len(), self.model.restart_reasons(), ui);
            if self.menu_state.as_ref() != Some(&state) {
                let img = self.drawer.render(&mut self.menu, &*self.model, ui);
                self.menu_panel.update(&img.bgra, img.w, img.h);
                self.menu_state = Some(state);
            }
            let ph = self.menu_panel.h as i64;
            // Centred when it opens. Then its top stays put so a new row grows it downwards, and it stays inside the
            // picture.
            if self.menu_top.map(|t| t.1) != Some(r) {
                self.menu_top = Some((r.y + (r.h - ph) / 2, r));
            }
            let top = self.menu_top.unwrap().0.min(r.y + r.h - ph).max(r.y);
            let _ = self.renderer.panel(cmd, target, &self.menu_panel, r.x + (32.0 * ui) as i64, top, hdr, self.cfg.hdr_white_nits);
        } else if clock::now() < self.toast_until {
            let state = format!("{}|{:.3}", self.toast_text, ui);
            if self.toast_state.as_ref() != Some(&state) {
                let img = ui::pill(&self.toast_text, ui);
                self.toast_panel.update(&img.bgra, img.w, img.h);
                self.toast_state = Some(state);
            }
            let m = (16.0 * ui) as i64;
            let mut below = 0; // with the stats overlay in the top right, messages go under it
            if self.stats_shown && self.display.overlay_corner == crate::display::Corner::TopRight {
                below = self.stats_panel.h as i64 + (4.0 * self.stats_ui(r.h)).floor().max(2.0) as i64;
            }
            let (pw, _) = (self.toast_panel.w as i64, self.toast_panel.h as i64);
            let _ = self.renderer.panel(cmd, target, &self.toast_panel, r.x + r.w - pw - m, r.y + m + below, hdr, self.cfg.hdr_white_nits);
        }
    }

    /// Runs `f` on a worker thread while the window stays responsive, showing the last frame and `text` with moving
    /// dots (io.py busy()). Keys pressed meanwhile take effect afterwards, so Q quits at the next write().
    pub fn busy<T: Send>(&mut self, text: &str, f: impl FnOnce() -> T + Send) -> T {
        self.menu.open = false;
        let last = self.last.clone();
        let out = std::thread::scope(|s| {
            let worker = s.spawn(f);
            let t0 = clock::now();
            while !worker.is_finished() {
                let dots = ".".repeat(1 + ((clock::now() - t0) as usize) % 3);
                self.toast(&format!("{text}{dots:<3}"), 1.0);
                self.events();
                self.render(last.as_ref(), None, true);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            worker.join()
        });
        self.toast_until = 0.0;
        match out {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }

    /// The frame just shown, with the look drawn on at the stream's size, for instant replay with "Looks in clips"
    /// (io.py baked()). Returns the frame itself when no look is on, and for HDR frames. The new frame's GPU work is
    /// queued on the shared queue, so anything reading it there comes after. A CPU reader waits for the queue first.
    pub fn baked(&mut self, frame: &Frame) -> Frame {
        let Some(look) = self.display.look.clone() else { return frame.clone() };
        if frame.kind != PixelKind::Bgra8 {
            return frame.clone();
        }
        let luts = self.renderer.luts.clone();
        let Some(plan) = self.renderer.looks.plan(&look, &luts) else { return frame.clone() };
        let out = Frame::new(frame.width, frame.height, PixelKind::Bgra8);
        autoreleasepool(|_| {
            if let Some(cmd) = Gpu::shared().queue.commandBuffer()
                && self.renderer.bake(&cmd, &plan, &frame.texture, &out.texture, self.frame_no).is_ok() {
                    cmd.commit();
                }
        });
        out
    }

    /// For window capture. Puts the window over the app's window at (x, y, w, h), in points from the main display's
    /// top left as ScreenCaptureKit gives it. The window is then borderless, floating and click-through, and never
    /// takes the keys. None makes it a normal window again (crate::window::WindowCmd::Overlay).
    pub fn set_overlay(&mut self, rect: Option<(f64, f64, f64, f64)>) {
        window::on_main(WindowCmd::Overlay(rect));
        if let Some(mtm) = MainThreadMarker::new() {
            window::pump(mtm);
        }
    }

    /// Takes the window off the screen now, before the rest of stopping. close() follows.
    pub fn hide(&mut self) {
        window::on_main(WindowCmd::Hide);
        if let Some(mtm) = MainThreadMarker::new() {
            window::pump(mtm);
        }
    }

    /// The menu's model, for the caller to reach its settings.
    pub fn model(&mut self) -> &mut dyn MenuModel {
        &mut *self.model
    }

    fn take_reply(&mut self, all: bool) -> SinkReply {
        match self.cmd.take() {
            None => SinkReply::Continue,
            Some(c) if all || matches!(c, SinkReply::Quit | SinkReply::Restart) => c,
            Some(c) => {
                self.cmd = Some(c); // kept for write()
                SinkReply::Continue
            }
        }
    }

    pub fn log_summary(&self) {
        let gpu = self.timing.gpu();
        let mut v: Vec<_> = gpu.iter().collect();
        v.sort_by_key(|(_, s)| std::cmp::Reverse(s.n));
        if !v.is_empty() {
            let parts: Vec<String> = v.iter().map(|(k, s)| format!("{k} p50 {:.2} / p95 {:.2} ms (n={})", s.p50, s.p95, s.n)).collect();
            println!("[display] GPU time to draw each frame: {}", parts.join("; "));
        }
        let p = self.timing.presents(0);
        if p.shown.n > 0 {
            println!("[display] between frames on screen: {} ({} not shown of {})", p.shown, p.dropped, p.frames);
        }
    }
}

impl Presenter for VideoPresenter {
    fn write(&mut self, frame: &Frame, overlay: Option<&Overlay>) -> SinkReply {
        if self.closed {
            return SinkReply::Quit;
        }
        if self.waited >= 0 {
            // a picture again after a wait, so its note goes
            self.waited = -1;
            self.toast_until = 0.0;
        }
        self.events();
        if (frame.width, frame.height) != (self.w, self.h) {
            self.w = frame.width;
            self.h = frame.height;
        }
        self.render(Some(frame), overlay, false);
        self.last = Some(frame.clone());
        self.take_reply(true)
    }

    fn wait(&mut self, w: &Waiting) -> SinkReply {
        if self.closed {
            return SinkReply::Quit;
        }
        self.events();
        let n = w.since as i64;
        if n >= 2 && (n != self.waited || self.menu.open) {
            self.waited = n;
            let why = if w.why.is_empty() { String::new() } else { format!("\n{}", w.why) };
            self.toast(&format!("No picture from {} for {n} s. Esc quits{why}", w.what), 1.5);
            let last = self.last.clone();
            self.render(last.as_ref(), None, true); // the last frame, plain (no upscaler or look)
        }
        self.take_reply(false)
    }

    fn toast(&mut self, text: &str, seconds: f64) {
        self.toast_text = text.to_string();
        self.toast_until = clock::now() + seconds;
    }

    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Some(cmd) = Gpu::shared().queue.commandBuffer() {
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        if self.cfg.log {
            self.log_summary();
        }
        window::on_main(WindowCmd::Close);
        if let Some(mtm) = MainThreadMarker::new() {
            window::pump(mtm);
        }
    }
}

// Everything the presenter holds is Metal state, the layer (see LayerRef), plain data and the Send MenuModel. Metal
// devices, queues, pipelines and textures are thread-safe, and command buffers live for one call. It's used from one
// thread at a time (&mut self), so it can move to the pipeline's thread.
unsafe impl Send for VideoPresenter {}

impl Drop for VideoPresenter {
    fn drop(&mut self) {
        self.close();
    }
}
