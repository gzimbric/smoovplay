//! Small Metal helpers for the window's passes: render pipelines made once per (function, pixel format, blending),
//! textures, and one call that encodes a full-viewport pass.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLFunction, MTLLibrary,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
use sp_core::gpu::Library;
use sp_core::{Error, Gpu, Result};

use crate::fill::Rect;

pub type Tex = ProtocolObject<dyn MTLTexture>;
pub type Cmd = ProtocolObject<dyn MTLCommandBuffer>;
pub type Rps = ProtocolObject<dyn MTLRenderPipelineState>;
pub type Func = ProtocolObject<dyn MTLFunction>;

pub const PRESENT_MSL: &str = include_str!("shaders/present.metal");

/// The bytes of a plain-data value, for setFragmentBytes.
pub fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

pub fn floats_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Blending for a pass: none (it replaces), or premultiplied alpha over what's there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Blend {
    Replace,
    Over,
}

/// Render pipelines by (library, function, target format, blending), made on first use. Every pass uses the same
/// vertex function (one triangle over the viewport).
pub struct Pipelines {
    pub present: Retained<Library>,
    vertex: Retained<Func>,
    cache: HashMap<(usize, String, MTLPixelFormat, Blend), Retained<Rps>>,
}

impl Pipelines {
    pub fn new() -> Result<Pipelines> {
        let gpu = Gpu::shared();
        let present = gpu.library(PRESENT_MSL)?;
        let vertex = present
            .newFunctionWithName(&NSString::from_str("fullscreen"))
            .ok_or_else(|| Error::Metal("no vertex function".into()))?;
        Ok(Pipelines { present, vertex, cache: HashMap::new() })
    }

    /// A pipeline for `function` in the window's own shaders.
    pub fn get(&mut self, function: &str, format: MTLPixelFormat, blend: Blend) -> Result<Retained<Rps>> {
        let lib = self.present.clone();
        self.get_in(&lib, 0, function, format, blend)
    }

    /// A pipeline for `function` in another library (`lib_id` tells libraries apart in the cache).
    pub fn get_in(&mut self, lib: &Library, lib_id: usize, function: &str, format: MTLPixelFormat, blend: Blend)
        -> Result<Retained<Rps>> {
        let key = (lib_id, function.to_string(), format, blend);
        if let Some(p) = self.cache.get(&key) {
            return Ok(p.clone());
        }
        let f = lib
            .newFunctionWithName(&NSString::from_str(function))
            .ok_or_else(|| Error::Metal(format!("no shader function {function}")))?;
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&self.vertex));
        desc.setFragmentFunction(Some(&f));
        let att = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        att.setPixelFormat(format);
        if blend == Blend::Over {
            att.setBlendingEnabled(true);
            att.setSourceRGBBlendFactor(MTLBlendFactor::One);
            att.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            att.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            att.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        let p = Gpu::shared()
            .device
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|e| Error::Metal(format!("pipeline {function}: {}", e.localizedDescription())))?;
        self.cache.insert(key, p.clone());
        Ok(p)
    }

    /// Drops the pipelines made from library `lib_id` (a look's passes being remade).
    pub fn forget(&mut self, lib_id: usize) {
        self.cache.retain(|k, _| k.0 != lib_id);
    }
}

/// A 2D texture the GPU draws into and reads (private storage).
pub fn texture(w: usize, h: usize, format: MTLPixelFormat) -> Retained<Tex> {
    texture_with(w, h, format, MTLStorageMode::Private)
}

pub fn texture_with(w: usize, h: usize, format: MTLPixelFormat, storage: MTLStorageMode) -> Retained<Tex> {
    unsafe {
        let desc = MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format,
            w.max(1),
            h.max(1),
            false,
        );
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
        desc.setStorageMode(storage);
        Gpu::shared().device.newTextureWithDescriptor(&desc).expect("couldn't make a texture")
    }
}

/// A 3D RGBA16F texture from RGB floats (a LUT: `n`^3 triples, red fastest).
pub fn texture_3d(n: usize, rgb: &[[f32; 3]]) -> Retained<Tex> {
    let mut halves: Vec<u16> = Vec::with_capacity(n * n * n * 4);
    for c in rgb {
        for v in c {
            halves.push(half::f16::from_f32(*v).to_bits());
        }
        halves.push(half::f16::ONE.to_bits());
    }
    unsafe {
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(MTLTextureType::Type3D);
        desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        desc.setWidth(n);
        desc.setHeight(n);
        desc.setDepth(n);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Shared);
        let t = Gpu::shared().device.newTextureWithDescriptor(&desc).expect("couldn't make a 3D texture");
        t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
            MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: n, height: n, depth: n } },
            0,
            0,
            NonNull::new(halves.as_mut_ptr() as *mut _).unwrap(),
            n * 8,
            n * n * 8,
        );
        t
    }
}

/// Whether a pass keeps what's in the target or starts from black.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Load {
    Clear,
    Keep,
    /// Everything gets drawn over.
    DontCare,
}

/// Encodes one pass: `pipeline` over `viewport` of `target`, with fragment textures by index and fragment buffers
/// (bytes, index).
pub fn pass(cmd: &Cmd, target: &Tex, load: Load, pipeline: &Rps, viewport: Rect, textures: &[(&Tex, usize)],
            buffers: &[(&[u8], usize)]) {
    let rp = MTLRenderPassDescriptor::new();
    let att = unsafe { rp.colorAttachments().objectAtIndexedSubscript(0) };
    att.setTexture(Some(target));
    att.setLoadAction(match load {
        Load::Clear => MTLLoadAction::Clear,
        Load::Keep => MTLLoadAction::Load,
        Load::DontCare => MTLLoadAction::DontCare,
    });
    att.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0 });
    att.setStoreAction(MTLStoreAction::Store);
    let Some(enc) = cmd.renderCommandEncoderWithDescriptor(&rp) else { return };
    let (tw, th) = (target.width() as i64, target.height() as i64);
    if let Some(v) = viewport.clip(tw, th) {
        enc.setRenderPipelineState(pipeline);
        enc.setViewport(MTLViewport {
            originX: v.x as f64,
            originY: v.y as f64,
            width: v.w as f64,
            height: v.h as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        unsafe {
            for (t, i) in textures {
                enc.setFragmentTexture_atIndex(Some(t), *i);
            }
            for (b, i) in buffers {
                enc.setFragmentBytes_length_atIndex(NonNull::new(b.as_ptr() as *mut _).unwrap(), b.len(), *i);
            }
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
    }
    enc.endEncoding();
}

/// A whole target's rectangle.
pub fn whole(t: &Tex) -> Rect {
    Rect::new(0, 0, t.width() as i64, t.height() as i64)
}
