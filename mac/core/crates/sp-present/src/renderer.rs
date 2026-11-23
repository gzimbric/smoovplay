//! Drawing a frame into a target texture (the window's drawable, or an offscreen texture for checks and for
//! instant replay's "Looks in clips"): the upscaler (FSR 1 or bilinear), the look, Fill the screen (Panorama's extra
//! pass), then the panels (menu, messages, stats) on top. framegen/io.py GLWindowSink.write's drawing, in Metal.
//! Everything reads the frame's texture as it is: no CPU copies.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2_metal::{MTLPixelFormat, MTLTexture};
use sp_core::Result;

use crate::display::Scaler;
use crate::fill::{self, Fill, Rect};
use crate::gpu::{self, Blend, Cmd, Load, Pipelines, Tex, bytes_of};
use crate::looks::LutStore;
use crate::looks_gpu::{LookPlan, LookRenderer};

#[repr(C)]
#[derive(Clone, Copy)]
struct Place {
    origin: [f32; 2],
    size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FsrP {
    in_size: [f32; 2],
    out_size: [f32; 2],
    origin: [f32; 2],
    sharpness: f32,
    pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Pano {
    screen: [f32; 2],
    t: f32,
    pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PanelP {
    origin: [f32; 2],
    hdr: u32,
    white: f32,
}

/// What to draw: a picture of `src_size` with these settings.
#[derive(Clone)]
pub struct Picture<'a> {
    pub src: &'a Tex,
    pub src_size: (usize, usize),
    pub scaler: Scaler,
    pub sharpness: f32,
    pub fill: Fill,
    pub look: Option<&'a LookPlan>,
    pub frame_no: u64,
    /// HDR10 (PQ) values: intermediates stay half floats.
    pub hdr: bool,
    /// Plain: bilinear only, no look (the "no picture" screen's last frame, as io.py's wait()).
    pub plain: bool,
}

/// Where the picture went, for the panels: the UI rectangle (the picture as seen on screen) and what drew it.
#[derive(Clone, Debug)]
pub struct Drawn {
    pub ui: Rect,
    pub picture: Rect,
    /// "FSR 1 + look", "no upscale", ... (the detailed overlay's)
    pub name: String,
    /// The name and the picture's size, for GPU timings ("FSR 1 3200x1800").
    pub label: String,
}

/// A premultiplied BGRA image on the GPU, drawn 1:1 (menu, messages, the stats overlay).
pub struct Panel {
    texs: Vec<Retained<Tex>>,
    turn: usize,
    pub w: usize,
    pub h: usize,
}

impl Panel {
    pub fn new() -> Panel {
        Panel { texs: Vec::new(), turn: 0, w: 0, h: 0 }
    }

    /// New contents (w x h premultiplied BGRA, rows top first). Three textures take turns, so one the GPU may still
    /// be reading is never written.
    pub fn update(&mut self, bgra: &[u8], w: usize, h: usize) {
        if (w, h) != (self.w, self.h) || self.texs.is_empty() {
            self.texs = (0..3).map(|_| gpu::texture_with(w, h, MTLPixelFormat::BGRA8Unorm, objc2_metal::MTLStorageMode::Shared)).collect();
            self.w = w;
            self.h = h;
        }
        self.turn = (self.turn + 1) % self.texs.len();
        let t = &self.texs[self.turn];
        unsafe {
            t.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: w, height: h, depth: 1 },
                },
                0,
                std::ptr::NonNull::new(bgra.as_ptr() as *mut _).unwrap(),
                w * 4,
            );
        }
    }

    pub fn texture(&self) -> Option<&Tex> {
        self.texs.get(self.turn).map(|t| &**t)
    }
}

impl Default for Panel {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Renderer {
    pub pipes: Pipelines,
    pub looks: LookRenderer,
    pub luts: Arc<LutStore>,
    mid: Option<Retained<Tex>>,
    pano: Option<Retained<Tex>>,
}

impl Renderer {
    pub fn new(luts: Arc<LutStore>) -> Result<Renderer> {
        Ok(Renderer { pipes: Pipelines::new()?, looks: LookRenderer::new(), luts, mid: None, pano: None })
    }

    fn sized(slot: &mut Option<Retained<Tex>>, w: usize, h: usize, f: MTLPixelFormat) -> Retained<Tex> {
        match slot {
            Some(t) if t.width() == w && t.height() == h && t.pixelFormat() == f => t.clone(),
            _ => {
                let t = gpu::texture(w, h, f);
                *slot = Some(t.clone());
                t
            }
        }
    }

    /// FSR 1 from `src` (sw x sh) into `target`'s rectangle `rect` (EASU into an image of the rectangle's size, then
    /// RCAS into the target).
    #[allow(clippy::too_many_arguments)]
    pub fn fsr(&mut self, cmd: &Cmd, src: &Tex, sw: usize, sh: usize, target: &Tex, rect: Rect, load: Load,
               sharpness: f32, hdr: bool) -> Result<()> {
        let (dw, dh) = (rect.w as usize, rect.h as usize);
        // The EASU result: 8-bit for SDR as in fsr.py, half floats for HDR10 values.
        let mf = if hdr { MTLPixelFormat::RGBA16Float } else { MTLPixelFormat::RGBA8Unorm };
        let mid = Self::sized(&mut self.mid, dw, dh, mf);
        let p = FsrP { in_size: [sw as f32, sh as f32], out_size: [dw as f32, dh as f32], origin: [0.0, 0.0], sharpness, pad: 0.0 };
        let easu = self.pipes.get("fsr_easu", mf, Blend::Replace)?;
        gpu::pass(cmd, &mid, Load::DontCare, &easu, gpu::whole(&mid), &[(src, 0)], &[(bytes_of(&p), 0)]);
        let p = FsrP { origin: [rect.x as f32, rect.y as f32], ..p };
        let rcas = self.pipes.get("fsr_rcas", target.pixelFormat(), Blend::Replace)?;
        gpu::pass(cmd, target, load, &rcas, rect, &[(&mid, 0)], &[(bytes_of(&p), 0)]);
        Ok(())
    }

    /// The picture resized with plain bilinear filtering into `rect`.
    pub fn bilinear(&mut self, cmd: &Cmd, src: &Tex, target: &Tex, rect: Rect, load: Load) -> Result<()> {
        let p = Place { origin: [rect.x as f32, rect.y as f32], size: [rect.w as f32, rect.h as f32] };
        let ps = self.pipes.get("bilinear", target.pixelFormat(), Blend::Replace)?;
        gpu::pass(cmd, target, load, &ps, rect, &[(src, 0)], &[(bytes_of(&p), 0)]);
        Ok(())
    }

    /// The whole picture: upscaler, look, fill. Clears the target (black bars).
    pub fn draw(&mut self, cmd: &Cmd, pic: &Picture, target: &Tex) -> Result<Drawn> {
        let (fw, fh) = (target.width() as i64, target.height() as i64);
        let (sw, sh) = (pic.src_size.0 as i64, pic.src_size.1 as i64);
        let (dw, dh, x0, y0) = fill::layout(pic.fill, fw, fh, sw, sh);
        let (vx, vy, vw, vh) = fill::visible(dw, dh, x0, y0, fw, fh);
        let pano = pic.fill == Fill::Panorama && fill::wider(fw, fh, sw, sh);
        let ui = if pano { Rect::new(0, 0, fw, fh) } else { Rect::from_gl(vw, vh, vx, vy, fh) };
        let picture = Rect::from_gl(dw, dh, x0, y0, fh);
        let upscale = dw > sw || dh > sh;
        let mut label = if !upscale || pic.plain { "no upscale".to_string() } else { pic.scaler.name().to_string() };
        if pic.plain && upscale {
            label = "bilinear".into();
        }

        // Panorama: drawn as for Fit into its own texture, then bent across the screen.
        let pano_tex = pano.then(|| Self::sized(&mut self.pano, dw as usize, dh as usize, MTLPixelFormat::RGBA16Float));
        let (dst, rect): (&Tex, Rect) = match &pano_tex {
            Some(t) => (t, Rect::new(0, 0, dw, dh)),
            None => (target, picture),
        };
        let load = if pano { Load::DontCare } else { Load::Clear };
        let look = if pic.plain || pic.hdr { None } else { pic.look };
        if let Some(plan) = look {
            let src: Retained<Tex>;
            let input = if upscale && pic.scaler == Scaler::Fsr {
                let inp = self.looks.input_target("screen", dw as usize, dh as usize);
                self.fsr(cmd, pic.src, pic.src_size.0, pic.src_size.1, &inp, Rect::new(0, 0, dw, dh), Load::DontCare,
                         pic.sharpness, false)?;
                src = inp;
                &*src
            } else {
                pic.src
            };
            self.looks.draw(&mut self.pipes, cmd, plan, input, dst, rect, load, pic.frame_no, "screen")?;
            label += " + look";
        } else if upscale && pic.scaler == Scaler::Fsr && !pic.plain {
            self.fsr(cmd, pic.src, pic.src_size.0, pic.src_size.1, dst, rect, load, pic.sharpness, pic.hdr)?;
        } else {
            self.bilinear(cmd, pic.src, dst, rect, load)?;
        }
        if let Some(t) = &pano_tex {
            let p = Pano { screen: [fw as f32, fh as f32], t: fill::panorama_t(fw, fh, sw, sh) as f32, pad: 0.0 };
            let ps = self.pipes.get("panorama", target.pixelFormat(), Blend::Replace)?;
            gpu::pass(cmd, target, Load::DontCare, &ps, gpu::whole(target), &[(t, 0)], &[(bytes_of(&p), 0)]);
            label += " + panorama";
        }
        Ok(Drawn { ui, picture, label: format!("{label} {dw}x{dh}"), name: label })
    }

    /// The look alone, at the frame's own size, into `target` (instant replay's "Looks in clips").
    pub fn bake(&mut self, cmd: &Cmd, plan: &LookPlan, src: &Tex, target: &Tex, frame_no: u64) -> Result<()> {
        self.looks.draw(&mut self.pipes, cmd, plan, src, target, gpu::whole(target), Load::DontCare, frame_no, "bake")
    }

    /// A panel at (x, y) (its top-left corner, in target pixels), blended over what's there.
    pub fn panel(&mut self, cmd: &Cmd, target: &Tex, panel: &Panel, x: i64, y: i64, hdr: bool, white_nits: f32) -> Result<()> {
        let Some(tex) = panel.texture() else { return Ok(()) };
        let p = PanelP { origin: [x as f32, y as f32], hdr: hdr as u32, white: white_nits };
        let ps = self.pipes.get("panel", target.pixelFormat(), Blend::Over)?;
        gpu::pass(cmd, target, Load::Keep, &ps, Rect::new(x, y, panel.w as i64, panel.h as i64), &[(tex, 0)], &[(bytes_of(&p), 0)]);
        Ok(())
    }
}
