//! Draws a look (crate::looks) with Metal (framegen/looks_gl.py's LookRenderer). looks::stages() splits a look into
//! passes; each pass is one fragment function made for that run of nodes, so a run of per-pixel nodes costs one pass
//! however many there are; a node that reads other pixels starts a new pass, which reads the one before it from a
//! half-float texture of the output's size. Bloom, halation, blur and clarity also build a blurred copy of that input
//! (six halvings, then back up). The first pass reads the picture (FSR's result at the output's size, or the frame
//! itself, which it resizes) and the last writes to the target's rectangle.
//!
//! Passes are compiled once per layout of nodes (a few ms); setting changes only change the float buffer.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use objc2::Message;
use objc2::rc::Retained;
use objc2_metal::{MTLPixelFormat, MTLTexture};
use sp_core::gpu::Library;
use sp_core::{Gpu, Result};

use crate::fill::Rect;
use crate::gpu::{self, Blend, Cmd, Load, Pipelines, Tex, bytes_of, floats_bytes};
use crate::looks::{self, Look, Lut, LutStore, Stage};

const HEAD: &str = include_str!("shaders/looks_head.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct LookK {
    out_size: [f32; 2],
    origin: [f32; 2],
    frame_no: u32,
    pad: [u32; 3],
}

/// One pass of a look, ready to draw.
struct PassPlan {
    lib: Retained<Library>,
    lib_id: usize,
    params: Vec<f32>,
    luts: Vec<Retained<Tex>>,
    /// The blurred copy: (bright-pass threshold or -1, how much wider levels count), for bloom, blur...
    pyramid: Option<(f32, f32)>,
}

/// A look's passes. Cheap to keep; made again when the look changes.
pub struct LookPlan {
    passes: Vec<PassPlan>,
    pub look: Arc<Look>,
}

impl LookPlan {
    pub fn passes(&self) -> usize {
        self.passes.len()
    }
}

/// What a look needs at one output size: two half-float images to pass between passes, and the blur levels.
struct Targets {
    size: (usize, usize),
    ping: [Retained<Tex>; 2],
    down: Vec<Retained<Tex>>,
    up: Vec<Retained<Tex>>,
}

impl Targets {
    fn new(w: usize, h: usize) -> Targets {
        let f = MTLPixelFormat::RGBA16Float;
        Targets { size: (w, h), ping: [gpu::texture(w, h, f), gpu::texture(w, h, f)], down: Vec::new(), up: Vec::new() }
    }
    fn levels(&mut self) {
        if self.down.is_empty() {
            let sizes = looks::pyramid_sizes(self.size.0, self.size.1);
            self.down = sizes.iter().map(|&(w, h)| gpu::texture(w, h, MTLPixelFormat::RGBA16Float)).collect();
            self.up = sizes.iter().take(sizes.len().saturating_sub(1))
                .map(|&(w, h)| gpu::texture(w, h, MTLPixelFormat::RGBA16Float)).collect();
        }
    }
}

/// The fragment function's source for one pass, and how its settings are laid out.
pub fn stage_source(stage: &Stage, first: bool) -> String {
    let mut decl = String::new();
    let mut body = String::new();
    let mut off = 0usize;
    let mut luts = 0usize;
    for (i, n) in stage.all().enumerate() {
        let start = i == 0 && stage.start.is_some();
        let mut args = vec!["X".to_string(), "c".to_string()];
        if start {
            args.push("src".into());
            args.push("glow".into());
        }
        if n.ty.key == "lut" {
            let _ = write!(decl, ", texture3d<float> lut{luts} [[texture({})]]", 2 + luts);
            args.push(format!("lut{luts}"));
            luts += 1;
        }
        for (name, v) in looks::uniforms(n, None) {
            args.push(match v.len() {
                1 => format!("P[{off}]"),
                3 => format!("v3(P + {off})"),
                _ if name == "m" => format!("P + {off}"),
                _ => unreachable!(),
            });
            off += v.len();
        }
        let f = if start { "s_" } else { "n_" };
        let _ = writeln!(body, "    {{ float3 o = {f}{}({}); c = clamp(mix(c, o, P[{off}]), 0.0, 1.0); }}", n.ty.key, args.join(", "));
        off += 1;
    }
    let read = if first { "src.sample(LIN, X.UV).rgb" } else { "src.read(uint2(X.Q)).rgb" };
    format!(
        "{HEAD}\nfragment float4 look_pass(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], \
         texture2d<float> glow [[texture(1)]]{decl}, constant float* P [[buffer(0)]], constant LookK& K [[buffer(1)]]) {{\n\
         \x20   Ctx X;\n    X.Q = in.pos.xy - K.origin;\n    X.UV = X.Q / K.out_size;\n    X.out_size = K.out_size;\n\
         \x20   X.frame_no = K.frame_no;\n    float3 c = {read};\n{body}    return float4(c, 1.0);\n}}\n"
    )
}

/// The settings of a pass's nodes, in its shader's order.
fn stage_params(stage: &Stage, lut_of: &dyn Fn(&str) -> Option<Arc<Lut>>) -> Vec<f32> {
    let mut out = Vec::new();
    for n in stage.all() {
        let lut = if n.ty.key == "lut" { lut_of(&n.file) } else { None };
        for (_, v) in looks::uniforms(n, lut.as_deref()) {
            out.extend(v.iter().map(|&x| x as f32));
        }
        out.push(n.mix as f32);
    }
    out
}

fn layout_key(stage: &Stage, first: bool) -> String {
    let mut k = format!("{first}:");
    for n in stage.all() {
        k += n.ty.key;
        k.push(',');
    }
    if stage.start.is_some() {
        k.push('s');
    }
    k
}

pub struct LookRenderer {
    libs: HashMap<String, (Retained<Library>, usize)>,
    next_id: usize,
    lut_tex: HashMap<String, (Arc<Lut>, Retained<Tex>)>,
    targets: HashMap<&'static str, Targets>,
    plan_for: Option<(usize, Look)>,
    plan: Option<Arc<LookPlan>>,
    /// Why looks can't be drawn here (a shader didn't compile), or None.
    pub broken: Option<String>,
}

impl Default for LookRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl LookRenderer {
    pub fn new() -> LookRenderer {
        LookRenderer { libs: HashMap::new(), next_id: 1000, lut_tex: HashMap::new(), targets: HashMap::new(),
                       plan_for: None, plan: None, broken: None }
    }

    /// The passes for this look (kept until the look changes), or None if it does nothing.
    pub fn plan(&mut self, look: &Arc<Look>, luts: &LutStore) -> Option<Arc<LookPlan>> {
        if self.broken.is_some() {
            return None;
        }
        let key = Arc::as_ptr(look) as usize;
        let same = matches!(&self.plan_for, Some((k, l)) if *k == key && l == &**look);
        if !same {
            self.plan = self.make_plan(look, luts).map(Arc::new);
            self.plan_for = Some((key, (**look).clone()));
        }
        self.plan.clone()
    }

    fn make_plan(&mut self, look: &Arc<Look>, store: &LutStore) -> Option<LookPlan> {
        let lut_of = |name: &str| store.get(name);
        let st = looks::stages(look, &|name| store.get(name).is_some());
        if st.is_empty() {
            return None;
        }
        let mut passes = Vec::new();
        for (i, stage) in st.iter().enumerate() {
            let key = layout_key(stage, i == 0);
            let (lib, lib_id) = match self.libs.get(&key) {
                Some(l) => l.clone(),
                None => match Gpu::shared().library(&stage_source(stage, i == 0)) {
                    Ok(lib) => {
                        self.next_id += 1;
                        self.libs.insert(key.clone(), (lib.clone(), self.next_id));
                        (lib, self.next_id)
                    }
                    Err(e) => {
                        let why = format!("a shader didn't compile ({})", e.to_string().chars().take(300).collect::<String>());
                        eprintln!("[looks] can't draw looks: {why}");
                        self.broken = Some(why);
                        return None;
                    }
                },
            };
            let mut luts = Vec::new();
            for n in stage.all().filter(|n| n.ty.key == "lut") {
                if let Some(l) = lut_of(&n.file) {
                    luts.push(self.lut_texture(&n.file, &l));
                }
            }
            let pyramid = stage.start.as_ref().filter(|n| n.ty.uses_pyramid()).map(|n| {
                let (thr, r) = looks::glow(n);
                (thr.map_or(-1.0, |t| t as f32), r as f32)
            });
            passes.push(PassPlan { lib, lib_id, params: stage_params(stage, &lut_of), luts, pyramid });
        }
        Some(LookPlan { passes, look: look.clone() })
    }

    fn lut_texture(&mut self, name: &str, lut: &Arc<Lut>) -> Retained<Tex> {
        if let Some((l, t)) = self.lut_tex.get(name)
            && Arc::ptr_eq(l, lut) {
                return t.clone();
            }
        let t = gpu::texture_3d(lut.size, &lut.table);
        self.lut_tex.insert(name.to_string(), (lut.clone(), t.clone()));
        t
    }

    fn targets(&mut self, purpose: &'static str, w: usize, h: usize) -> &mut Targets {
        let t = self.targets.entry(purpose).or_insert_with(|| Targets::new(w, h));
        if t.size != (w, h) {
            *t = Targets::new(w, h);
        }
        t
    }

    /// An image at the output's size for the upscaler to draw into; draw() then reads it as `src`.
    pub fn input_target(&mut self, purpose: &'static str, w: usize, h: usize) -> Retained<Tex> {
        self.targets(purpose, w, h).ping[0].clone()
    }

    /// The look from `src` (any size; the first pass resizes it) into `target`'s rectangle `rect` (which may go past
    /// its edges), the first pass into the target clearing it (`load`).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, pipes: &mut Pipelines, cmd: &Cmd, plan: &LookPlan, src: &Tex, target: &Tex, rect: Rect,
                load: Load, frame: u64, purpose: &'static str) -> Result<()> {
        let (w, h) = (rect.w as usize, rect.h as usize);
        let tf = target.pixelFormat();
        let tg = self.targets(purpose, w, h);
        let mut out_i = if std::ptr::eq(src as *const Tex, &*tg.ping[0] as *const Tex) { 1 } else { 0 };
        let pings = tg.ping.clone();
        let last = plan.passes.len() - 1;
        let mut cur: Retained<Tex> = src.retain();
        for (i, p) in plan.passes.iter().enumerate() {
            let glow = match p.pyramid {
                Some((thr, r)) => self.pyramid(pipes, cmd, purpose, w, h, &cur, thr, r)?,
                None => cur.clone(),
            };
            let (dst, fmt, place, ld) = if i == last {
                (target, tf, rect, load)
            } else {
                (&*pings[out_i], MTLPixelFormat::RGBA16Float, Rect::new(0, 0, rect.w, rect.h), Load::DontCare)
            };
            let k = LookK {
                out_size: [w as f32, h as f32],
                origin: [place.x as f32, place.y as f32],
                frame_no: (frame & 0x3FFF_FFFF) as u32,
                pad: [0; 3],
            };
            let ps = pipes.get_in(&p.lib, p.lib_id, "look_pass", fmt, Blend::Replace)?;
            let mut tex: Vec<(&Tex, usize)> = vec![(&cur, 0), (&glow, 1)];
            for (j, t) in p.luts.iter().enumerate() {
                tex.push((t, 2 + j));
            }
            let params = if p.params.is_empty() { vec![0.0] } else { p.params.clone() };
            gpu::pass(cmd, dst, ld, &ps, place, &tex, &[(floats_bytes(&params), 0), (bytes_of(&k), 1)]);
            if i != last {
                cur = pings[out_i].clone();
                out_i ^= 1;
            }
        }
        Ok(())
    }

    /// The blurred copy of `src` (output-sized): its first level, half the size.
    #[allow(clippy::too_many_arguments)]
    fn pyramid(&mut self, pipes: &mut Pipelines, cmd: &Cmd, purpose: &'static str, w: usize, h: usize, src: &Tex,
               thr: f32, r: f32) -> Result<Retained<Tex>> {
        let tg = self.targets(purpose, w, h);
        tg.levels();
        if tg.down.is_empty() {
            return Ok(src.retain());
        }
        let (down, up) = (tg.down.clone(), tg.up.clone());
        let f = MTLPixelFormat::RGBA16Float;
        let ps = pipes.get("pyr_down", f, Blend::Replace)?;
        let mut cur: &Tex = src;
        for (k, lv) in down.iter().enumerate() {
            let t: f32 = if k == 0 { thr } else { -1.0 };
            gpu::pass(cmd, lv, Load::DontCare, &ps, gpu::whole(lv), &[(cur, 0)], &[(bytes_of(&t), 0)]);
            cur = lv;
        }
        let ps = pipes.get("pyr_up", f, Blend::Replace)?;
        let mut below: &Tex = &down[down.len() - 1];
        for k in (0..up.len()).rev() {
            gpu::pass(cmd, &up[k], Load::DontCare, &ps, gpu::whole(&up[k]), &[(&down[k], 0), (below, 1)], &[(bytes_of(&r), 0)]);
            below = &up[k];
        }
        Ok(below.retain())
    }
}
