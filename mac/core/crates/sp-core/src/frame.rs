//! A picture on the GPU. It's a Metal texture, BGRA8 for SDR video or RGBA16F for HDR10 (PQ, BT.2020, as
//! framegen/hdr.py's "pq16"). Like Python's Tick.frame, it's never written to once a source has handed it on.
//!
//! Textures use shared storage. On Apple silicon the GPU and CPU share memory, so a test or a CPU-side check can read
//! one without a blit. A frame made from a decoder's IOSurface keeps that surface alive through `owner`.
//!
//! Hold the `Frame`, not only its texture, until GPU work reading it has finished. A producer (sp-media's Converter,
//! sp-rife's output ring) reuses a texture once nothing holds its Frame any more.

use std::any::Any;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};

use crate::gpu::Gpu;

pub type Texture = ProtocolObject<dyn MTLTexture>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelKind {
    /// 8-bit BGRA (alpha unused), full-range SDR. Every SDR source gives this and RIFE takes it.
    Bgra8,
    /// Half-float RGBA, PQ-encoded BT.2020 (HDR output).
    Rgba16f,
}

impl PixelKind {
    pub fn format(self) -> MTLPixelFormat {
        match self {
            PixelKind::Bgra8 => MTLPixelFormat::BGRA8Unorm,
            PixelKind::Rgba16f => MTLPixelFormat::RGBA16Float,
        }
    }
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelKind::Bgra8 => 4,
            PixelKind::Rgba16f => 8,
        }
    }
}

#[derive(Clone)]
pub struct Frame {
    pub texture: Retained<Texture>,
    pub width: usize,
    pub height: usize,
    pub kind: PixelKind,
    /// Whatever must outlive the texture (a decoder's CVPixelBuffer, say).
    pub owner: Option<Arc<dyn Any + Send + Sync>>,
}

// Metal textures are safe to share between threads, and a Frame is never written to once handed on.
unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frame({}x{} {:?})", self.width, self.height, self.kind)
    }
}

impl Frame {
    /// A new, uninitialised frame (shader read and write).
    pub fn new(width: usize, height: usize, kind: PixelKind) -> Frame {
        let gpu = Gpu::shared();
        let texture = unsafe {
            let desc = MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                kind.format(),
                width,
                height,
                false,
            );
            desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::RenderTarget);
            desc.setStorageMode(MTLStorageMode::Shared);
            gpu.device.newTextureWithDescriptor(&desc).expect("couldn't make a texture")
        };
        Frame { texture, width, height, kind, owner: None }
    }

    /// Wraps a texture someone else made (an IOSurface-backed one from a decoder, say).
    pub fn wrap(texture: Retained<Texture>, kind: PixelKind, owner: Option<Arc<dyn Any + Send + Sync>>) -> Frame {
        let (width, height) = (texture.width(), texture.height());
        Frame { texture, width, height, kind, owner }
    }

    fn region(&self) -> MTLRegion {
        MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: self.width, height: self.height, depth: 1 },
        }
    }

    /// An SDR frame from packed 8-bit BGR (OpenCV's layout, as the Python side and test clips use).
    pub fn from_bgr(bgr: &[u8], width: usize, height: usize) -> Frame {
        assert_eq!(bgr.len(), width * height * 3, "BGR data doesn't match {width}x{height}");
        let mut bgra = vec![255u8; width * height * 4];
        for (d, s) in bgra.chunks_exact_mut(4).zip(bgr.chunks_exact(3)) {
            d[..3].copy_from_slice(s);
        }
        Self::from_bgra(&bgra, width, height)
    }

    /// An SDR frame from packed 8-bit BGRA.
    pub fn from_bgra(bgra: &[u8], width: usize, height: usize) -> Frame {
        assert_eq!(bgra.len(), width * height * 4);
        let f = Frame::new(width, height, PixelKind::Bgra8);
        unsafe {
            f.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                f.region(),
                0,
                NonNull::new(bgra.as_ptr() as *mut _).unwrap(),
                width * 4,
            );
        }
        f
    }

    /// An HDR frame from packed RGBA half floats, given as their 16-bit patterns (`half::f16::to_bits`).
    pub fn from_rgba16f_bits(rgba: &[u16], width: usize, height: usize) -> Frame {
        assert_eq!(rgba.len(), width * height * 4);
        let f = Frame::new(width, height, PixelKind::Rgba16f);
        unsafe {
            f.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                f.region(),
                0,
                NonNull::new(rgba.as_ptr() as *mut _).unwrap(),
                width * 8,
            );
        }
        f
    }

    /// The frame's bytes as the texture holds them (BGRA8, or RGBA16F as little-endian halves). Waits for nothing,
    /// so call it after the GPU work that wrote it has completed.
    pub fn read_raw(&self) -> Vec<u8> {
        let row = self.width * self.kind.bytes_per_pixel();
        let mut out = vec![0u8; row * self.height];
        unsafe {
            self.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(out.as_mut_ptr() as *mut _).unwrap(),
                row,
                self.region(),
                0,
            );
        }
        out
    }

    /// An SDR frame as packed 8-bit BGR (for tests, scoring against the Python reference, and saving images).
    pub fn to_bgr(&self) -> Vec<u8> {
        assert_eq!(self.kind, PixelKind::Bgra8, "to_bgr is for SDR frames");
        let raw = self.read_raw();
        let mut out = Vec::with_capacity(self.width * self.height * 3);
        for p in raw.chunks_exact(4) {
            out.extend_from_slice(&p[..3]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgr_round_trips() {
        let (w, h) = (37, 11);
        let bgr: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let f = Frame::from_bgr(&bgr, w, h);
        assert_eq!((f.width, f.height), (w, h));
        assert_eq!(f.to_bgr(), bgr);
    }

    #[test]
    fn rgba16f_round_trips() {
        let (w, h) = (5, 3);
        let bits: Vec<u16> = (0..w * h * 4).map(|i| 0x3c00 + i as u16).collect(); // 1.0 and up
        let raw = Frame::from_rgba16f_bits(&bits, w, h).read_raw();
        let back: Vec<u16> = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        assert_eq!(back, bits);
    }
}
