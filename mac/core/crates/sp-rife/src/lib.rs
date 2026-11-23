//! RIFE frame interpolation on Apple's GPU, for SmoovPlay's macOS core: framegen/interpolator.py's
//! RifeInterpolator as it runs on a Mac (framegen/metalrife.py's MetalIFNet), without Python.
//!
//! * Weights are read straight from vs-rife's checkpoint (weights.rs), downloaded first if missing.
//! * The convolutions run as MPSGraph executables, one per IFBlock and one for the encoder, compiled once for the
//!   stream's size (graph.rs); everything between them is metalrife.py's Metal kernels (kernels.rs): building each
//!   block's input from the frames warped by the flow so far, the last pass that warps, blends and writes the frame,
//!   the encoder's last layer, and Steady HUD.
//! * Frames come in and go out as [`sp_core::Frame`] textures (BGRA8, or RGBA16F PQ for HDR), with no CPU copies.
//!
//! A stream's real frames go through [`Interpolator::load`] then [`Interpolator::prepare`] into one of two
//! ping-pong slots, once each; [`Interpolator::interpolate`] draws the frame at a fraction t between the two slots.

use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use objc2_metal_performance_shaders::MPSCommandBuffer;
use sp_core::{Frame, Gpu, Interpolator, PixelKind};

pub mod graph;
pub mod kernels;
pub mod net;
pub mod weights;

pub use net::{MODELS, Model};
pub use sp_core::{Error, Result};

use kernels::Buffer;
use net::Net;

/// framegen/steady.py STEADY_HUD_THRESHOLD: the default Steady HUD threshold.
pub const STEADY_HUD_THRESHOLD: f32 = 0.03;

/// What to build (framegen/interpolator.py RifeInterpolator's arguments that apply on a Mac).
#[derive(Clone, Debug)]
pub struct Options {
    /// "4.22.lite" (the default; also "4.25", "4.26", "4.25.lite").
    pub version: String,
    /// Motion detail (framegen's `scale`): 1.0, 0.5 or 0.25.
    pub scale: f32,
    /// Half precision, as the Python Mac engine. Only true is supported.
    pub fp16: bool,
    /// HDR output: interpolate() returns RGBA16F frames (PQ BT.2020 0..1), as framegen's pq16. Loaded frames may
    /// be either kind; the model sees 0..1 either way.
    pub hdr: bool,
    /// Where checkpoints live (None: the repo's models/ folder, or $SMOOVPLAY_MODELS).
    pub model_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Options {
        Options { version: "4.22.lite".into(), scale: 1.0, fp16: true, hdr: false, model_dir: None }
    }
}

/// How many output frames interpolate() cycles through, so the frame from one call is still intact while the next
/// one or two are drawn (two in-between frames per real frame at 90 Hz, three at 120 Hz).
const OUTPUTS: usize = 4;

/// framegen/interpolator.py's RifeInterpolator for the Mac (see the crate docs). interpolate() returns one of four
/// output frames, used in turn, so a frame stays intact while the next three are drawn; with
/// [`RifeInterpolator::set_blocking`] off it returns before the GPU has drawn it.
pub struct RifeInterpolator {
    net: Net,
    opts: Options,
    slots: [Retained<Buffer>; 2],
    edges: [Retained<Buffer>; 2],
    hud: Retained<Buffer>,
    steady: f32,
    newest: Option<usize>,
    input: Option<Frame>,
    // What each slot was prepared from, kept until the slot is prepared again (the GPU may still be reading it).
    prepared: [Option<Frame>; 2],
    outputs: Vec<Frame>,
    next_out: usize,
    blocking: bool,
    last: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

// Metal objects are thread-safe; the interpolator is used from one thread at a time (&mut self).
unsafe impl Send for RifeInterpolator {}

impl RifeInterpolator {
    /// An interpolator for `width` x `height` frames. Errors (not panics) for an unknown model, a motion detail the
    /// Metal engine doesn't take (2.0), fp32, a checkpoint that can't be found, downloaded or read, or a Metal
    /// failure.
    pub fn new(width: usize, height: usize, opts: Options) -> Result<RifeInterpolator> {
        if !opts.fp16 {
            return Err(Error::Other(
                "the Mac's RIFE engine runs in half precision only (fp16 = true), as framegen's does".into(),
            ));
        }
        if width == 0 || height == 0 {
            return Err(Error::Other(format!("bad frame size {width}x{height}")));
        }
        let m = net::model(&opts.version)?;
        net::block_scales(&m, opts.scale)?; // the motion detail, before reading any weights
        let dir = opts.model_dir.clone().unwrap_or_else(weights::default_model_dir);
        let path = weights::ensure_weights(&opts.version, &dir)?;
        let wts = weights::load(&path)?;
        let net = Net::new(&wts, &m, opts.scale, width, height)?;
        let kind = if opts.hdr { PixelKind::Rgba16f } else { PixelKind::Bgra8 };
        let outputs = (0..OUTPUTS).map(|_| Frame::new(width, height, kind)).collect();
        let mut it = RifeInterpolator {
            slots: [net.new_slot(), net.new_slot()],
            edges: [net.new_plane(1), net.new_plane(1)],
            hud: net.new_plane(2),
            net,
            opts,
            steady: 0.0,
            newest: None,
            input: None,
            prepared: [None, None],
            outputs,
            next_out: 0,
            blocking: true,
            last: None,
        };
        it.warm_up();
        Ok(it)
    }

    /// One run through every kernel and executable on a blank frame: MPSGraph specialises an executable the first
    /// time it's encoded (10 to 20 ms here), which shouldn't land on the stream's first frames.
    fn warm_up(&mut self) {
        let blank = Frame::from_bgra(&vec![0u8; self.net.w * self.net.h * 4], self.net.w, self.net.h);
        for slot in 0..2 {
            self.load(&blank);
            self.prepare(slot);
        }
        self.steady = STEADY_HUD_THRESHOLD;
        let cb = self.command_buffer();
        self.encode_steady(&cb, 1, true);
        self.commit(cb);
        self.interpolate(1, 0.5);
        self.steady = 0.0;
        self.interpolate(1, 0.5);
        self.wait();
        self.newest = None;
        self.input = None;
        self.prepared = [None, None];
        self.next_out = 0;
    }

    pub fn options(&self) -> &Options {
        &self.opts
    }

    /// The padded size the network runs at (framegen's pw, ph).
    pub fn padded_size(&self) -> (usize, usize) {
        (self.net.pw, self.net.ph)
    }

    /// Whether interpolate() waits for the GPU (the default: the frame it returns is finished). With false it
    /// returns once the work is committed to the shared queue (sp_core::Gpu): anything else on that queue runs
    /// after it, and [`RifeInterpolator::wait`] waits for it.
    pub fn set_blocking(&mut self, blocking: bool) {
        self.blocking = blocking;
    }

    /// Waits for everything committed so far.
    pub fn wait(&mut self) {
        if let Some(cb) = self.last.take() {
            cb.waitUntilCompleted();
        }
    }

    fn command_buffer(&self) -> Retained<MPSCommandBuffer> {
        unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&Gpu::shared().queue) }
    }

    fn commit(&mut self, cb: Retained<MPSCommandBuffer>) {
        cb.commit();
        // The buffer the MPSCommandBuffer is on now: MPSGraph may have committed earlier ones, which run first.
        self.last = Some(unsafe { cb.commandBuffer() });
    }

    /// Steady HUD's mask between the older slot and `newest` (and, if `both`, the older slot's edges first).
    fn encode_steady(&self, cb: &MPSCommandBuffer, newest: usize, both: bool) {
        let old = 1 - newest;
        if both {
            self.net.encode_edges(cb, &self.slots[old], &self.edges[old]);
        }
        self.net.encode_edges(cb, &self.slots[newest], &self.edges[newest]);
        self.net.encode_mask(
            cb,
            &self.slots[old],
            &self.slots[newest],
            &self.edges[old],
            &self.edges[newest],
            &self.hud,
            self.steady,
        );
    }

    /// The real frame in a slot as fp16 RGB (PH x PW x 3, padded), for tests. Waits for the GPU first.
    pub fn slot_rgb(&mut self, slot: usize) -> Vec<f32> {
        self.wait();
        let n = self.net.ph * self.net.pw;
        let s = unsafe { std::slice::from_raw_parts(self.slots[slot].contents().as_ptr() as *const half::f16, n * 8) };
        s.chunks_exact(8).flat_map(|p| [p[0].to_f32(), p[1].to_f32(), p[2].to_f32()]).collect()
    }

    /// Per-stage GPU times (net.rs profile), for the bench. Needs a loaded frame.
    pub fn profile(&mut self) -> Vec<(String, f64)> {
        self.wait();
        let frame = self.input.clone().expect("load a frame first");
        let out = self.outputs[0].clone();
        let slot = Retained::clone(&self.slots[1]);
        let r = self.net.profile(&frame, &self.slots[0], &slot, &out);
        // (profile wrote encode_out into slot 1 from the loaded frame: prepare it again so the slots stay right)
        self.prepare(1);
        self.wait();
        r
    }

    /// Steady HUD's current mask (PH x PW, 0..1), for tests. Waits for the GPU first.
    pub fn steady_mask(&mut self) -> Vec<f32> {
        self.wait();
        let n = self.net.ph * self.net.pw;
        let s = unsafe { std::slice::from_raw_parts(self.hud.contents().as_ptr() as *const half::f16, n) };
        s.iter().map(|v| v.to_f32()).collect()
    }
}

impl Interpolator for RifeInterpolator {
    fn label(&self) -> String {
        format!("RIFE {}", self.opts.version)
    }

    fn size(&self) -> (usize, usize) {
        (self.net.w, self.net.h)
    }

    /// Takes the frame (no copy: frames are immutable); prepare() reads it on the GPU.
    fn load(&mut self, frame: &Frame) {
        assert_eq!(
            (frame.width, frame.height),
            (self.net.w, self.net.h),
            "RIFE was built for {}x{} frames",
            self.net.w,
            self.net.h
        );
        self.input = Some(frame.clone());
    }

    fn prepare(&mut self, slot: usize) {
        assert!(slot < 2, "slot is 0 or 1");
        let frame = self.input.clone().expect("prepare() before load()");
        let cb = self.command_buffer();
        self.net.encode_prepare(&cb, &frame, &self.slots[slot]);
        if self.steady > 0.0 {
            self.encode_steady(&cb, slot, false);
        }
        self.commit(cb);
        self.prepared[slot] = Some(frame);
        self.newest = Some(slot);
    }

    fn interpolate(&mut self, newest: usize, t: f32) -> Frame {
        assert!(newest < 2, "slot is 0 or 1");
        let out = self.outputs[self.next_out].clone();
        self.next_out = (self.next_out + 1) % OUTPUTS;
        let cb = self.command_buffer();
        let hud = if self.steady > 0.0 { Some(&*self.hud) } else { None };
        self.net.encode_interpolate(&cb, &self.slots[1 - newest], &self.slots[newest], t, &out, hud);
        self.commit(cb);
        if self.blocking {
            self.wait();
        }
        out
    }

    fn set_steady(&mut self, threshold: f32) {
        let was_off = self.steady <= 0.0;
        self.steady = threshold.max(0.0);
        if threshold > 0.0 && was_off {
            // Turned on mid-pair: the current pair's mask now (the older frame's edges weren't worked out while it
            // was off).
            if let Some(newest) = self.newest {
                let cb = self.command_buffer();
                self.encode_steady(&cb, newest, true);
                self.commit(cb);
            }
        }
    }
}

/// Helpers for tests and examples.
pub mod testing {
    use std::ptr::NonNull;

    use objc2_metal::{MTLOrigin, MTLRegion, MTLSize, MTLTexture};
    use sp_core::{Frame, PixelKind};

    /// An RGBA16F frame from RGBA halves.
    pub fn frame_rgba16f(rgba: &[half::f16], w: usize, h: usize) -> Frame {
        assert_eq!(rgba.len(), w * h * 4);
        let f = Frame::new(w, h, PixelKind::Rgba16f);
        let region = MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w, height: h, depth: 1 } };
        unsafe {
            f.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new(rgba.as_ptr() as *mut _).unwrap(),
                w * 8,
            );
        }
        f
    }
}
