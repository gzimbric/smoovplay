//! The AppKit side of the video window. That's an NSWindow whose content view hosts a CAMetalLayer, plus its keys and
//! mouse, Retina sizing and the screen's refresh and EDR facts. Fullscreen is a borderless window over the monitor,
//! with the menu bar and Dock hidden while SmoovPlay is in front, as in io.py's _mac_hide_bars.
//!
//! AppKit wants all of this on the main thread, so it lives there in a thread-local. The presenter, on any thread,
//! reaches it through [`Shared`] and [`on_main`]. [`Shared`] holds what the window saw, such as keys, a close and the
//! screen. [`on_main`] sends what it should do, such as fullscreen, the layer's format and sync, or closing. On the
//! main thread [`on_main`] runs at once. Elsewhere it's queued on the main dispatch queue, which the main thread's
//! run loop serves (NSApp.run, the Swift app's, or [`pump`]).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dispatch2::DispatchQueue;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSApplicationPresentationOptions, NSBackingStoreType, NSCursor,
    NSEvent, NSEventMask, NSEventModifierFlags, NSFloatingWindowLevel, NSNormalWindowLevel, NSResponder, NSScreen,
    NSView, NSWindow, NSWindowCollectionBehavior, NSWindowDelegate, NSWindowOcclusionState, NSWindowStyleMask,
};
use objc2_core_foundation::CGSize;
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceITUR_2100_PQ, kCGColorSpaceSRGB};
use objc2_foundation::{MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString};
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::{CAEDRMetadata, CAMetalLayer};
use sp_core::Gpu;

/// A key the window got.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A letter or other character key, lower case, as the keyboard layout has it.
    Char(char),
    Up,
    Down,
    Left,
    Right,
    Enter,
    Esc,
    Backspace,
    Tab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyPress {
    pub key: Key,
    pub shift: bool,
    pub command: bool,
    pub repeat: bool,
}

/// What the screen the window is on can do.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScreenInfo {
    pub name: String,
    /// Points to pixels (2 on a Retina display).
    pub scale: f64,
    /// The screen in points, and its usable part (no menu bar or Dock).
    pub size: (f64, f64),
    pub visible: (f64, f64),
    /// EDR headroom now, and the most it can give (maximumPotentialExtendedDynamicRangeColorComponentValue). 1.0 means
    /// none, on an SDR display or with HDR off.
    pub edr_now: f64,
    pub edr_potential: f64,
    /// Refresh intervals in seconds. They're equal on a fixed-rate display. min is below max when it runs a variable
    /// rate (Adaptive-Sync set to Variable, or ProMotion).
    pub refresh_min: f64,
    pub refresh_max: f64,
    pub max_fps: i64,
}

impl ScreenInfo {
    pub fn variable_refresh(&self) -> bool {
        self.refresh_max > self.refresh_min * 1.05
    }
}

/// What the main thread tells the presenter.
#[derive(Debug, Default)]
pub struct Inbox {
    pub keys: Vec<KeyPress>,
    pub moved: bool,
    pub close: bool,
    pub screen: ScreenInfo,
    /// Some part of the window is on screen (not minimised or hidden behind others).
    pub visible: bool,
    /// The drawable's size in pixels and the window's number (screencapture -l).
    pub pixels: (usize, usize),
    pub window_number: isize,
    pub fullscreen: bool,
}

#[derive(Debug, Default)]
pub struct Shared {
    pub inbox: Mutex<Inbox>,
}

/// What the presenter asks the main thread to do.
#[derive(Clone, Debug)]
pub enum WindowCmd {
    Fullscreen(bool),
    /// The layer's format for the frames coming, either HDR10 (RGBA16Float, PQ BT.2100, EDR) or SDR (BGRA8, sRGB).
    Layer { hdr: bool, vsync: bool, drawables: usize, white_nits: f32, peak_nits: f32 },
    RefreshScreen,
    HideCursor,
    Hide,
    Close,
    /// Sit over another app's window, for window capture (io.py's overlay_of). The rectangle is (x, y, w, h) in points
    /// from the main display's top left, as ScreenCaptureKit gives it. The window is borderless, floats above the app,
    /// lets clicks through and never takes the keys, which stay with the app. None makes it a normal window again.
    Overlay(Option<(f64, f64, f64, f64)>),
}

/// The window is an overlay (WindowCmd::Overlay), so it mustn't take the keys from the app under it.
static OVERLAY: AtomicBool = AtomicBool::new(false);

pub(crate) struct Ivars {
    shared: Arc<Shared>,
    layer: Retained<CAMetalLayer>,
}

define_class!(
    // SAFETY: NSView has no subclassing requirements, and VideoView doesn't implement Drop.
    #[unsafe(super(NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SmoovPlayVideoView"]
    #[ivars = Ivars]
    pub(crate) struct VideoView;

    impl VideoView {
        #[unsafe(method(acceptsFirstResponder))]
        fn accepts_first_responder(&self) -> bool {
            true
        }

        #[unsafe(method(isOpaque))]
        fn is_opaque(&self) -> bool {
            true
        }

        #[unsafe(method(keyDown:))]
        fn key_down(&self, event: &NSEvent) {
            if let Some(k) = key_of(event) {
                self.ivars().shared.inbox.lock().unwrap().keys.push(k);
            }
        }

        #[unsafe(method(mouseMoved:))]
        fn mouse_moved(&self, _event: &NSEvent) {
            self.ivars().shared.inbox.lock().unwrap().moved = true;
        }

        #[unsafe(method(setFrameSize:))]
        fn set_frame_size(&self, size: NSSize) {
            let _: () = unsafe { msg_send![super(self), setFrameSize: size] };
            self.update_drawable();
        }

        #[unsafe(method(viewDidChangeBackingProperties))]
        fn did_change_backing(&self) {
            let _: () = unsafe { msg_send![super(self), viewDidChangeBackingProperties] };
            self.update_drawable();
        }
    }

    unsafe impl NSObjectProtocol for VideoView {}
);

impl VideoView {
    fn new(mtm: MainThreadMarker, frame: NSRect, shared: Arc<Shared>, layer: Retained<CAMetalLayer>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Ivars { shared, layer });
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }

    /// The drawable follows the view's size in pixels. On Retina that's points times the backing scale.
    fn update_drawable(&self) {
        let px = self.convertSizeToBacking(self.bounds().size);
        let scale = self.window().map_or(1.0, |w| w.backingScaleFactor());
        let layer = &self.ivars().layer;
        layer.setContentsScale(scale);
        let (w, h) = (px.width.round().max(1.0), px.height.round().max(1.0));
        layer.setDrawableSize(CGSize::new(w, h));
        self.ivars().shared.inbox.lock().unwrap().pixels = (w as usize, h as usize);
    }
}

define_class!(
    // SAFETY: NSWindow's subclassing rules are met (init goes through its designated initialiser), and there's no Drop.
    #[unsafe(super(NSWindow, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SmoovPlayVideoWindow"]
    pub(crate) struct VideoWindow;

    impl VideoWindow {
        // A borderless window (fullscreen) takes the keys too. An overlay over another app never does.
        #[unsafe(method(canBecomeKeyWindow))]
        fn can_become_key(&self) -> bool {
            !OVERLAY.load(Ordering::Relaxed)
        }
        #[unsafe(method(canBecomeMainWindow))]
        fn can_become_main(&self) -> bool {
            !OVERLAY.load(Ordering::Relaxed)
        }
    }

    unsafe impl NSObjectProtocol for VideoWindow {}
);

pub(crate) struct DelegateIvars {
    shared: Arc<Shared>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SmoovPlayVideoWindowDelegate"]
    #[ivars = DelegateIvars]
    pub(crate) struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl NSWindowDelegate for Delegate {
        // The close button. The presenter returns Quit and the pipeline closes the window, as in io.py's
        // window_should_close.
        #[unsafe(method(windowShouldClose:))]
        fn window_should_close(&self, _sender: &NSWindow) -> bool {
            self.ivars().shared.inbox.lock().unwrap().close = true;
            false
        }

        // These can come while the main thread's MainWindow is in use, from inside a call that changes the window,
        // so they work only from the notification's window and their own `shared`.
        #[unsafe(method(windowDidChangeScreen:))]
        fn did_change_screen(&self, n: &NSNotification) {
            if let Some(w) = n.object().and_then(|o| o.downcast::<NSWindow>().ok()) {
                screen_info(self.mtm(), &w, &self.ivars().shared);
            }
        }

        #[unsafe(method(windowDidChangeOcclusionState:))]
        fn did_change_occlusion(&self, n: &NSNotification) {
            if let Some(w) = n.object().and_then(|o| o.downcast::<NSWindow>().ok()) {
                let v = w.occlusionState().contains(NSWindowOcclusionState::Visible);
                self.ivars().shared.inbox.lock().unwrap().visible = v;
            }
        }
    }
);

impl Delegate {
    fn new(mtm: MainThreadMarker, shared: Arc<Shared>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DelegateIvars { shared });
        unsafe { msg_send![super(this), init] }
    }
}

fn key_of(e: &NSEvent) -> Option<KeyPress> {
    let flags = e.modifierFlags();
    let key = match e.keyCode() {
        36 | 76 => Key::Enter,
        48 => Key::Tab,
        51 => Key::Backspace,
        53 => Key::Esc,
        123 => Key::Left,
        124 => Key::Right,
        125 => Key::Down,
        126 => Key::Up,
        _ => {
            let s = e.charactersIgnoringModifiers()?.to_string();
            Key::Char(s.chars().next()?.to_lowercase().next()?)
        }
    };
    Some(KeyPress {
        key,
        shift: flags.contains(NSEventModifierFlags::Shift),
        command: flags.contains(NSEventModifierFlags::Command),
        repeat: e.isARepeat(),
    })
}

/// The window, on the main thread.
pub(crate) struct MainWindow {
    pub window: Retained<VideoWindow>,
    pub view: Retained<VideoView>,
    pub layer: Retained<CAMetalLayer>,
    _delegate: Retained<Delegate>,
    pub shared: Arc<Shared>,
    windowed: Option<NSRect>,
    borderless: bool,
    hdr: Option<bool>,
    /// For WindowCmd::Overlay, the app window's rectangle (AppKit's coordinates) and the frame from before it.
    overlay: Option<NSRect>,
    before_overlay: Option<NSRect>,
}

thread_local! {
    static MAIN: RefCell<Option<MainWindow>> = const { RefCell::new(None) };
}

fn with_main(f: impl FnOnce(&mut MainWindow)) {
    MAIN.with(|m| match m.try_borrow_mut() {
        Ok(mut w) => {
            if let Some(w) = w.as_mut() {
                f(w)
            }
        }
        // a command from inside another one's AppKit call, which should run once that's done
        Err(_) => eprintln!("[display] window busy: a window change was skipped"),
    });
}

/// Whether this process set up NSApplication itself (no Swift app around it), so it should finish launching it.
static LAUNCHED: AtomicBool = AtomicBool::new(false);

const WINDOWED: NSWindowStyleMask = NSWindowStyleMask(
    NSWindowStyleMask::Titled.0 | NSWindowStyleMask::Closable.0 | NSWindowStyleMask::Miniaturizable.0 | NSWindowStyleMask::Resizable.0,
);

/// The window's first size in points (io.py _first_size). It's the frame's size, at most 1600x900 and at most 90% of
/// the screen's usable area, keeping the frame's shape.
pub fn first_size(w: usize, h: usize, visible: (f64, f64)) -> (f64, f64) {
    let (ww, wh) = ((w as f64).min(1600.0), (h as f64).min(900.0));
    let k = 1f64.min(0.9 * visible.0 / ww).min(0.9 * visible.1 / wh);
    ((ww * k).floor(), (wh * k).floor())
}

/// Opens the window, on the main thread. Returns the layer, for the presenter.
pub(crate) fn open(mtm: MainThreadMarker, w: usize, h: usize, title: &str, drawables: usize, vsync: bool,
                   shared: Arc<Shared>) -> Retained<CAMetalLayer> {
    let app = NSApplication::sharedApplication(mtm);
    if !app.isRunning() && !LAUNCHED.load(Ordering::Relaxed) {
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    }
    let screen = NSScreen::mainScreen(mtm);
    let visible = screen.as_ref().map_or((1440.0, 900.0), |s| (s.visibleFrame().size.width, s.visibleFrame().size.height));
    let (ww, wh) = first_size(w, h, visible);
    let rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(ww, wh));

    let layer = CAMetalLayer::new();
    layer.setDevice(Some(&Gpu::shared().device));
    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    layer.setFramebufferOnly(true);
    layer.setMaximumDrawableCount(drawables.clamp(2, 3));
    layer.setDisplaySyncEnabled(vsync);
    layer.setOpaque(true);
    if let Some(cs) = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB })) {
        layer.setColorspace(Some(&cs));
    }

    let window = unsafe {
        let w: Retained<VideoWindow> = msg_send![VideoWindow::alloc(mtm),
            initWithContentRect: rect, styleMask: WINDOWED, backing: NSBackingStoreType::Buffered, defer: false];
        w.setReleasedWhenClosed(false);
        w
    };
    window.setTitle(&NSString::from_str(title));
    let view = VideoView::new(mtm, rect, shared.clone(), layer.clone());
    view.setLayer(Some(&layer));
    view.setWantsLayer(true);
    window.setContentView(Some(&view));
    window.setAcceptsMouseMovedEvents(true);
    let delegate = Delegate::new(mtm, shared.clone());
    window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    window.center();
    window.makeKeyAndOrderFront(None);
    window.makeFirstResponder(Some(&view));
    #[allow(deprecated)]
    app.activateIgnoringOtherApps(true);
    view.update_drawable();
    {
        let mut inbox = shared.inbox.lock().unwrap();
        inbox.window_number = window.windowNumber();
        inbox.visible = true;
    }
    MAIN.with(|m| {
        *m.borrow_mut() = Some(MainWindow {
            window,
            view,
            layer: layer.clone(),
            _delegate: delegate,
            shared,
            windowed: None,
            borderless: false,
            hdr: None,
            overlay: None,
            before_overlay: None,
        })
    });
    refresh_screen(mtm);
    layer
}

/// Runs `cmd` on the main thread. It runs now if this is the main thread, or else it's queued on the main dispatch
/// queue.
pub fn on_main(cmd: WindowCmd) {
    if let Some(mtm) = MainThreadMarker::new() {
        apply(mtm, cmd);
    } else {
        DispatchQueue::main().exec_async(move || {
            if let Some(mtm) = MainThreadMarker::new() {
                apply(mtm, cmd);
            }
        });
    }
}

fn apply(mtm: MainThreadMarker, cmd: WindowCmd) {
    match cmd {
        WindowCmd::Fullscreen(on) => with_main(|m| set_fullscreen(mtm, m, on)),
        WindowCmd::Layer { hdr, vsync, drawables, white_nits, peak_nits } => with_main(|m| {
            m.layer.setDisplaySyncEnabled(vsync);
            m.layer.setMaximumDrawableCount(drawables.clamp(2, 3));
            if m.hdr == Some(hdr) {
                return;
            }
            m.hdr = Some(hdr);
            unsafe {
                if hdr {
                    // HDR10 passes the frames' PQ values as they are, and macOS maps them to the display's EDR
                    // headroom. The metadata says SDR white is `white_nits` (BT.2408's 203) and the content peaks at
                    // `peak_nits`, so on a display with less headroom (or none) the highlights are rolled off, not
                    // clipped.
                    m.layer.setPixelFormat(MTLPixelFormat::RGBA16Float);
                    m.layer.setColorspace(CGColorSpace::with_name(Some(kCGColorSpaceITUR_2100_PQ)).as_deref());
                    m.layer.setWantsExtendedDynamicRangeContent(true);
                    // SP_PRESENT_EDR=none leaves the metadata out, and a number sets the peak (for measuring)
                    match std::env::var("SP_PRESENT_EDR").ok().as_deref() {
                        Some("none") => m.layer.setEDRMetadata(None),
                        other => {
                            let peak = other.and_then(|v| v.parse::<f32>().ok()).unwrap_or(peak_nits);
                            let md = CAEDRMetadata::HDR10MetadataWithMinLuminance_maxLuminance_opticalOutputScale(
                                0.005, peak, white_nits);
                            m.layer.setEDRMetadata(Some(&md));
                        }
                    }
                } else {
                    m.layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                    m.layer.setColorspace(CGColorSpace::with_name(Some(kCGColorSpaceSRGB)).as_deref());
                    m.layer.setWantsExtendedDynamicRangeContent(false);
                    m.layer.setEDRMetadata(None);
                }
            }
        }),
        WindowCmd::Overlay(rect) => with_main(|m| set_overlay(mtm, m, rect)),
        WindowCmd::RefreshScreen => refresh_screen(mtm),
        WindowCmd::HideCursor => NSCursor::setHiddenUntilMouseMoves(true),
        WindowCmd::Hide => with_main(|m| m.window.orderOut(None)),
        WindowCmd::Close => {
            let gone = MAIN.with(|m| m.borrow_mut().take());
            if let Some(mut m) = gone {
                if m.borderless {
                    set_fullscreen(mtm, &mut m, false);
                }
                m.window.setDelegate(None);
                m.window.close();
            }
            let app = NSApplication::sharedApplication(mtm);
            if app.isRunning() && LAUNCHED.load(Ordering::Relaxed) {
                // run() was ours (sp_present::run), and it returns once the window has gone
                app.stop(None);
                wake(mtm);
            }
        }
    }
}

/// Fullscreen as a borderless window over the monitor the window is mostly on (io.py _set_borderless). The menu bar
/// and the Dock are hidden while SmoovPlay is in front. They slide in at the edges and come back by themselves when
/// another app is in front. There's no fullscreen Space, so it's instant and Cmd+Tab works as usual.
fn set_fullscreen(mtm: MainThreadMarker, m: &mut MainWindow, on: bool) {
    if m.borderless == on {
        return;
    }
    if let Some(r) = m.overlay {
        // Over another app, go to its monitor or back over its window. It stays floating, click-through and without
        // the keys.
        let screen = m.window.screen().or_else(|| NSScreen::mainScreen(mtm));
        let frame = if on { screen.map_or(r, |s| s.frame()) } else { r };
        m.window.setFrame_display(frame, true);
        m.borderless = on;
        m.shared.inbox.lock().unwrap().fullscreen = on;
        m.view.update_drawable();
        return;
    }
    let app = NSApplication::sharedApplication(mtm);
    if on {
        m.windowed = Some(m.window.frame());
        let screen = m.window.screen().or_else(|| NSScreen::mainScreen(mtm));
        // this first, or macOS keeps the window under the menu bar
        app.setPresentationOptions(NSApplicationPresentationOptions::AutoHideDock | NSApplicationPresentationOptions::AutoHideMenuBar);
        m.window.setStyleMask(NSWindowStyleMask::Borderless);
        if let Some(s) = screen {
            m.window.setFrame_display(s.frame(), true);
        }
        NSCursor::setHiddenUntilMouseMoves(true);
    } else {
        app.setPresentationOptions(NSApplicationPresentationOptions::Default);
        m.window.setStyleMask(WINDOWED);
        m.window.setTitle(&NSString::from_str("SmoovPlay"));
        if let Some(r) = m.windowed {
            m.window.setFrame_display(r, true);
        }
    }
    m.window.makeKeyAndOrderFront(None);
    m.window.makeFirstResponder(Some(&m.view));
    m.borderless = on;
    m.shared.inbox.lock().unwrap().fullscreen = on;
    m.view.update_drawable();
}

/// WindowCmd::Overlay on the main thread.
fn set_overlay(mtm: MainThreadMarker, m: &mut MainWindow, rect: Option<(f64, f64, f64, f64)>) {
    match rect {
        Some((x, y, w, h)) => {
            // ScreenCaptureKit's points run down from the main display's top left. AppKit's run up from its bottom
            // left.
            let screens = NSScreen::screens(mtm);
            let main_h = screens.firstObject().map_or(0.0, |s| s.frame().size.height);
            let r = NSRect::new(NSPoint::new(x, main_h - y - h), NSSize::new(w.max(1.0), h.max(1.0)));
            if m.overlay.is_none() {
                let fullscreen = m.borderless;
                if fullscreen {
                    set_fullscreen(mtm, m, false); // its frame comes back below, over the app's monitor
                }
                m.before_overlay = Some(m.window.frame());
                OVERLAY.store(true, Ordering::Relaxed);
                m.window.setStyleMask(NSWindowStyleMask::Borderless);
                m.window.setLevel(NSFloatingWindowLevel);
                m.window.setIgnoresMouseEvents(true);
                m.window.setHasShadow(false);
                m.window.setCollectionBehavior(
                    NSWindowCollectionBehavior::CanJoinAllSpaces
                        | NSWindowCollectionBehavior::Stationary
                        | NSWindowCollectionBehavior::IgnoresCycle
                        | NSWindowCollectionBehavior::FullScreenAuxiliary,
                );
                m.overlay = Some(r);
                m.window.setFrame_display(r, true);
                m.window.orderFrontRegardless();
                if fullscreen {
                    set_fullscreen(mtm, m, true);
                }
            } else {
                m.overlay = Some(r);
                if !m.borderless {
                    m.window.setFrame_display(r, true);
                }
            }
            m.view.update_drawable();
        }
        None => {
            if m.overlay.take().is_none() {
                return;
            }
            OVERLAY.store(false, Ordering::Relaxed);
            let fullscreen = m.borderless;
            m.borderless = false;
            m.window.setLevel(NSNormalWindowLevel);
            m.window.setIgnoresMouseEvents(false);
            m.window.setHasShadow(true);
            m.window.setCollectionBehavior(NSWindowCollectionBehavior::Default);
            m.window.setStyleMask(WINDOWED);
            m.window.setTitle(&NSString::from_str("SmoovPlay"));
            if let Some(r) = m.before_overlay.take() {
                m.window.setFrame_display(r, true);
            }
            m.window.makeKeyAndOrderFront(None);
            m.window.makeFirstResponder(Some(&m.view));
            m.shared.inbox.lock().unwrap().fullscreen = false;
            if fullscreen {
                set_fullscreen(mtm, m, true);
            }
            m.view.update_drawable();
        }
    }
}

fn refresh_screen(mtm: MainThreadMarker) {
    with_main(|m| screen_info(mtm, &m.window, &m.shared));
}

fn screen_info(mtm: MainThreadMarker, window: &NSWindow, shared: &Shared) {
    {
        let Some(s) = window.screen().or_else(|| NSScreen::mainScreen(mtm)) else { return };
        let info = ScreenInfo {
            name: s.localizedName().to_string(),
            scale: s.backingScaleFactor(),
            size: (s.frame().size.width, s.frame().size.height),
            visible: (s.visibleFrame().size.width, s.visibleFrame().size.height),
            edr_now: s.maximumExtendedDynamicRangeColorComponentValue(),
            edr_potential: s.maximumPotentialExtendedDynamicRangeColorComponentValue(),
            refresh_min: s.minimumRefreshInterval(),
            refresh_max: s.maximumRefreshInterval(),
            max_fps: s.maximumFramesPerSecond() as i64,
        };
        shared.inbox.lock().unwrap().screen = info;
    }
}

/// Finishes launching NSApplication, once, when nothing else runs it (a Swift app or [`crate::run`]).
pub(crate) fn launch_if_needed(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    if !app.isRunning() && !LAUNCHED.swap(true, Ordering::Relaxed) {
        app.finishLaunching();
        #[allow(deprecated)]
        app.activateIgnoringOtherApps(true);
    }
}

/// Handles the window's waiting events without blocking, like glfw.poll_events. It's for a pipeline on the main
/// thread.
pub fn pump(mtm: MainThreadMarker) {
    launch_if_needed(mtm);
    let app = NSApplication::sharedApplication(mtm);
    let past = NSDate::distantPast();
    loop {
        let e = unsafe {
            app.nextEventMatchingMask_untilDate_inMode_dequeue(NSEventMask::Any, Some(&past), NSDefaultRunLoopMode, true)
        };
        match e {
            Some(e) => app.sendEvent(&e),
            None => break,
        }
    }
    app.updateWindows();
}

/// Runs the main thread's event loop until the video window closes. It's for a pipeline on another thread with no
/// Swift app.
pub fn run(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    if MAIN.with(|m| m.borrow().is_none()) {
        return;
    }
    LAUNCHED.store(true, Ordering::Relaxed);
    app.run();
}

/// Posts an empty event, so a run loop waiting for one returns after stop:.
fn wake(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    if let Some(e) = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
        objc2_app_kit::NSEventType::ApplicationDefined, NSPoint::new(0.0, 0.0), NSEventModifierFlags::empty(), 0.0, 0,
        None, 0, 0, 0,
    ) {
        app.postEvent_atStart(&e, true);
    }
}

/// Posts a key press to the window as a keyDown event in this app's own event queue, for checks. It goes the way a
/// real one does, through AppKit's dispatch, the view's keyDown: and the key map. Call it on the main thread.
pub fn post_key(mtm: MainThreadMarker, key: Key, shift: bool) {
    let (code, ch): (u16, String) = match key {
        Key::Up => (126, "\u{F700}".into()),
        Key::Down => (125, "\u{F701}".into()),
        Key::Left => (123, "\u{F702}".into()),
        Key::Right => (124, "\u{F703}".into()),
        Key::Enter => (36, "\r".into()),
        Key::Esc => (53, "\u{1b}".into()),
        Key::Tab => (48, "\t".into()),
        Key::Backspace => (51, "\u{7f}".into()),
        Key::Char(c) => {
            // ANSI positions of the letters the window uses. The event's characters decide, as for a real key.
            let code = match c {
                'a' => 0, 's' => 1, 'd' => 2, 'f' => 3, 'h' => 4, 'g' => 5, 'z' => 6, 'x' => 7, 'c' => 8, 'v' => 9,
                'b' => 11, 'q' => 12, 'w' => 13, 'e' => 14, 'r' => 15, 'y' => 16, 't' => 17, 'o' => 31, 'u' => 32,
                'i' => 34, 'p' => 35, 'l' => 37, 'j' => 38, 'k' => 40, 'n' => 45, 'm' => 46,
                _ => 0,
            };
            (code, c.to_string())
        }
    };
    let Some(num) = MAIN.with(|m| m.try_borrow().ok().and_then(|w| w.as_ref().map(|w| w.window.windowNumber()))) else { return };
    let flags = if shift { NSEventModifierFlags::Shift } else { NSEventModifierFlags::empty() };
    let chars = NSString::from_str(&if shift { ch.to_uppercase() } else { ch.clone() });
    let plain = NSString::from_str(&ch);
    if let Some(e) = NSEvent::keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode(
        objc2_app_kit::NSEventType::KeyDown, NSPoint::new(0.0, 0.0), flags, 0.0, num, None, &chars, &plain, false, code,
    ) {
        NSApplication::sharedApplication(mtm).postEvent_atStart(&e, false);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn first_size_fits_the_screen() {
        assert_eq!(super::first_size(1920, 1080, (3008.0, 1568.0)), (1600.0, 900.0));
        let (w, h) = super::first_size(1920, 1080, (1470.0, 880.0)); // a 13-inch MacBook
        assert!(w <= 0.9 * 1470.0 && h <= 0.9 * 880.0 && (w / h - 16.0 / 9.0).abs() < 0.01);
        assert_eq!(super::first_size(1280, 720, (3008.0, 1568.0)), (1280.0, 720.0));
    }
}
