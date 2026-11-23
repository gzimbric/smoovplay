//! VideoToolbox decoding of H.264 and HEVC access units as the frame tap delivers them. They come as Annex-B. The
//! codec header (parameter sets) arrives on its own once per stream, as the PS5 sends it out of band, or in-band
//! with keyframes.
//!
//! The decoder keeps the stream's parameter sets and makes its format description from them. Each access unit's
//! other NAL units go to VideoToolbox as one AVCC sample with 4-byte big-endian lengths. Decoding is synchronous,
//! with no asynchronous or temporal-order flags, so each picture comes out before `decode` returns. They come in
//! decode order because the PS5 and the fake PS5 send no B-frames. Pictures are IOSurface-backed, Metal-compatible
//! CVPixelBuffers. That means NV12 ('420v' or '420f') for 8-bit streams and P010 ('x420' or 'xf20') for 10-bit
//! ones, never VideoToolbox's compressed formats, which Metal can't read.
//!
//! No sample is marked as a sync sample or not. A PS5 answers a keyframe request with an intra frame whose NAL type
//! isn't a keyframe's (chiaki_tap/README.md, "First PS5 session"), and VideoToolbox decodes it like any other.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2_core_foundation::{CFBoolean, CFRetained, CFType};
use objc2_core_media::{
    kCMBlockBufferAssureMemoryNowFlag, kCMFormatDescriptionExtension_FullRangeVideo, CMBlockBuffer,
    CMFormatDescription, CMSampleBuffer, CMTime, CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets,
};
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey,
    CVImageBuffer, CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
};
use objc2_video_toolbox::{
    kVTDecompressionPropertyKey_RealTime, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
    kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder, VTDecodeFrameFlags, VTDecodeInfoFlags,
    VTDecompressionOutputCallbackRecord, VTDecompressionSession, VTSessionCopyProperty, VTSessionSetProperty,
};
use sp_tap::nal::{self, Codec};

use crate::cf::{self, fourcc};
use crate::colour::Tags;
use crate::params;

pub const NV12_VIDEO: u32 = fourcc(b"420v");
pub const NV12_FULL: u32 = fourcc(b"420f");
pub const P010_VIDEO: u32 = fourcc(b"x420");
pub const P010_FULL: u32 = fourcc(b"xf20");

/// Why an access unit didn't decode.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// A picture came before any codec header. Nothing decodes without one.
    #[error("no codec header (parameter sets) yet")]
    NoParams,
    /// The parameter sets didn't make a format description.
    #[error("bad codec header: {0} ({1})")]
    BadParams(i32, &'static str),
    /// VideoToolbox refused or failed the picture (or the session).
    #[error("{what}: {status} ({})", cf::status_name(*.status))]
    Status { what: &'static str, status: i32 },
}

/// A decoded picture.
pub struct Decoded {
    pub buffer: CFRetained<CVPixelBuffer>,
    pub width: usize,
    pub height: usize,
    /// P010 (10 bits in the top of 16), else NV12.
    pub ten_bit: bool,
    pub tags: Tags,
}

// CVPixelBuffers are CF objects, safe to hand between threads.
unsafe impl Send for Decoded {}

impl Decoded {
    pub fn bits(&self) -> u32 {
        if self.ten_bit { 10 } else { 8 }
    }
}

enum Out {
    Picture(CFRetained<CVImageBuffer>),
    Dropped,
    Failed(i32),
}

/// Where the output callback leaves pictures. It's boxed because its address is the callback's refcon.
struct Sink {
    out: Mutex<Vec<Out>>,
}

unsafe extern "C-unwind" fn on_output(
    refcon: *mut c_void,
    _frame_refcon: *mut c_void,
    status: i32,
    _flags: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    _pts: CMTime,
    _duration: CMTime,
) {
    let sink = unsafe { &*(refcon as *const Sink) };
    let o = if status != 0 {
        Out::Failed(status)
    } else if let Some(img) = NonNull::new(image) {
        Out::Picture(unsafe { CFRetained::retain(img) })
    } else {
        Out::Dropped
    };
    sink.out.lock().unwrap().push(o);
}

pub struct VtDecoder {
    codec: Codec,
    /// The stream's parameter sets as NAL units without start codes or trailing zeros, in the order they came.
    params: Vec<Vec<u8>>,
    format: Option<CFRetained<CMFormatDescription>>,
    format_stale: bool,
    session: Option<CFRetained<VTDecompressionSession>>,
    pixel_format: u32,
    sink: Box<Sink>,
    /// Whether the current session decodes on the media engine, as VideoToolbox reports it.
    pub hardware: bool,
    /// Sessions made so far (a new one for a new codec header that the old session can't take, or after `reset`).
    pub sessions: u32,
    sample: Vec<u8>,
}

// The session and format are CF objects, and a decoder is used from one thread at a time.
unsafe impl Send for VtDecoder {}

impl VtDecoder {
    pub fn new(codec: Codec) -> VtDecoder {
        VtDecoder {
            codec,
            params: Vec::new(),
            format: None,
            format_stale: true,
            session: None,
            pixel_format: NV12_VIDEO,
            sink: Box::new(Sink { out: Mutex::new(Vec::new()) }),
            hardware: false,
            sessions: 0,
            sample: Vec::new(),
        }
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    pub fn has_params(&self) -> bool {
        !self.params.is_empty()
    }

    /// Start over with a new session at the next picture (the parameter sets are kept).
    pub fn reset(&mut self) {
        self.drop_session();
        self.format_stale = true;
    }

    fn drop_session(&mut self) {
        if let Some(s) = self.session.take() {
            unsafe {
                s.wait_for_asynchronous_frames();
                s.invalidate();
            }
        }
    }

    /// Takes the parameter sets in `annexb` (an access unit, or the codec header on its own) as the stream's.
    /// Returns whether it held any.
    pub fn set_params(&mut self, annexb: &[u8]) -> bool {
        let units: Vec<Vec<u8>> = nal::nal_units(annexb, self.codec)
            .into_iter()
            .filter(|(t, _)| self.codec.is_param(*t))
            .map(|(_, u)| nal::trim(u).to_vec())
            .collect();
        if units.is_empty() {
            return false;
        }
        if units != self.params {
            self.params = units;
            self.format_stale = true;
        }
        true
    }

    fn make_format(&self) -> Result<CFRetained<CMFormatDescription>, DecodeError> {
        let want: &[u8] = match self.codec {
            Codec::H264 => &[7, 8],
            Codec::Hevc => &[32, 33, 34],
        };
        // At least one of each kind, the latest of each. A stream normally has exactly one.
        let mut sets: Vec<&[u8]> = Vec::new();
        for &t in want {
            let u = self.params.iter().rev().find(|u| !u.is_empty() && self.codec.unit_type(u[0]) == t);
            match u {
                Some(u) => sets.push(u),
                None => return Err(DecodeError::NoParams),
            }
        }
        let ptrs: Vec<NonNull<u8>> = sets.iter().map(|s| NonNull::new(s.as_ptr() as *mut u8).unwrap()).collect();
        let sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();
        let mut out: *const CMFormatDescription = std::ptr::null();
        let status = unsafe {
            match self.codec {
                Codec::H264 => CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    None,
                    sets.len(),
                    NonNull::new(ptrs.as_ptr() as *mut _).unwrap(),
                    NonNull::new(sizes.as_ptr() as *mut _).unwrap(),
                    4,
                    NonNull::from(&mut out),
                ),
                Codec::Hevc => CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    None,
                    sets.len(),
                    NonNull::new(ptrs.as_ptr() as *mut _).unwrap(),
                    NonNull::new(sizes.as_ptr() as *mut _).unwrap(),
                    4,
                    None,
                    NonNull::from(&mut out),
                ),
            }
        };
        if status != 0 || out.is_null() {
            return Err(DecodeError::BadParams(status, cf::status_name(status)));
        }
        Ok(unsafe { CFRetained::from_raw(NonNull::new(out as *mut CMFormatDescription).unwrap()) })
    }

    /// 8 or 10 bits from the SPS, and full range from the format description's colour description if it has one.
    fn pixel_format_for(&self, format: &CMFormatDescription) -> u32 {
        let sps_type = if self.codec == Codec::H264 { 7 } else { 33 };
        let bits = self
            .params
            .iter()
            .find(|u| !u.is_empty() && self.codec.unit_type(u[0]) == sps_type)
            .and_then(|u| params::bit_depth(u, self.codec))
            .unwrap_or(8);
        let full = unsafe { format.extension(kCMFormatDescriptionExtension_FullRangeVideo) }
            .and_then(|v| v.downcast::<CFBoolean>().ok())
            .map(|b| b.as_bool())
            .unwrap_or(false);
        match (bits > 8, full) {
            (false, false) => NV12_VIDEO,
            (false, true) => NV12_FULL,
            (true, false) => P010_VIDEO,
            (true, true) => P010_FULL,
        }
    }

    fn make_session(&mut self, format: &CMFormatDescription) -> Result<(), DecodeError> {
        self.drop_session();
        let pf = self.pixel_format_for(format);
        let empty = cf::dict(&[]);
        let pf_num = cf::num(pf as i64);
        let attrs = unsafe {
            cf::dict(&[
                (kCVPixelBufferPixelFormatTypeKey, &pf_num),
                (kCVPixelBufferMetalCompatibilityKey, cf::yes()),
                (kCVPixelBufferIOSurfacePropertiesKey, &empty),
            ])
        };
        let spec = unsafe { cf::dict(&[(kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder, cf::yes())]) };
        let record = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(on_output),
            decompressionOutputRefCon: &*self.sink as *const Sink as *mut c_void,
        };
        let mut out: *mut VTDecompressionSession = std::ptr::null_mut();
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                format,
                Some(&spec),
                Some(&attrs),
                &record,
                NonNull::from(&mut out),
            )
        };
        if status != 0 || out.is_null() {
            return Err(DecodeError::Status { what: "VTDecompressionSessionCreate", status });
        }
        let session = unsafe { CFRetained::from_raw(NonNull::new(out).unwrap()) };
        unsafe {
            VTSessionSetProperty(&session, kVTDecompressionPropertyKey_RealTime, Some(cf::yes()));
            let mut hw: *const CFType = std::ptr::null();
            if VTSessionCopyProperty(
                &session,
                kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                None,
                &mut hw as *mut _ as *mut c_void,
            ) == 0
                && !hw.is_null()
            {
                let v: CFRetained<CFType> = CFRetained::from_raw(NonNull::new(hw as *mut CFType).unwrap());
                self.hardware = v.downcast::<CFBoolean>().map(|b| b.as_bool()).unwrap_or(false);
            }
        }
        self.pixel_format = pf;
        self.session = Some(session);
        self.sessions += 1;
        Ok(())
    }

    fn ensure_session(&mut self) -> Result<(), DecodeError> {
        if !self.format_stale && self.session.is_some() {
            return Ok(());
        }
        if self.params.is_empty() {
            return Err(DecodeError::NoParams);
        }
        let format = self.make_format()?;
        // A new bit depth or range needs a new pixel format, so a new session.
        let same_pixels = self.pixel_format_for(&format) == self.pixel_format;
        let keep = match (&self.session, &self.format) {
            (Some(s), Some(_)) => same_pixels && unsafe { s.can_accept_format_description(&format) },
            _ => false,
        };
        if !keep {
            self.make_session(&format)?;
        }
        self.format = Some(format);
        self.format_stale = false;
        Ok(())
    }

    /// Decodes one access unit (Annex-B). Parameter sets in it become the stream's. A unit without a picture, such
    /// as the codec header on its own, gives no pictures. Err for a picture before any parameter sets, or when
    /// VideoToolbox fails it. The session is kept after a failure because it holds the reference pictures. A new
    /// one is made only when VideoToolbox says the session itself is gone.
    pub fn decode(&mut self, annexb: &[u8]) -> Result<Vec<Decoded>, DecodeError> {
        let units = nal::nal_units(annexb, self.codec);
        if units.iter().any(|(t, _)| self.codec.is_param(*t)) {
            self.set_params(annexb);
        }
        // The sample is every unit but the parameter sets, AVCC-style
        self.sample.clear();
        let mut pictures = 0;
        for (t, u) in &units {
            if self.codec.is_param(*t) {
                continue;
            }
            pictures += self.codec.is_vcl(*t) as usize;
            let u = nal::trim(u);
            self.sample.extend_from_slice(&(u.len() as u32).to_be_bytes());
            self.sample.extend_from_slice(u);
        }
        if pictures == 0 {
            return Ok(Vec::new());
        }
        self.ensure_session()?;
        let result = self.decode_sample();
        if let Err(DecodeError::Status { status: -12903, .. }) = result {
            // The session is gone because the GPU or media engine was reset or the Mac slept. Make a new one from
            // the same header.
            self.reset();
            self.ensure_session()?;
            return self.decode_sample();
        }
        result
    }

    fn decode_sample(&mut self) -> Result<Vec<Decoded>, DecodeError> {
        let session = self.session.as_ref().unwrap();
        let format = self.format.as_ref().unwrap();
        let len = self.sample.len();
        unsafe {
            let mut bb: *mut CMBlockBuffer = std::ptr::null_mut();
            let st = CMBlockBuffer::create_with_memory_block(
                None,
                std::ptr::null_mut(),
                len,
                None,
                std::ptr::null(),
                0,
                len,
                kCMBlockBufferAssureMemoryNowFlag,
                NonNull::from(&mut bb),
            );
            if st != 0 || bb.is_null() {
                return Err(DecodeError::Status { what: "CMBlockBufferCreateWithMemoryBlock", status: st });
            }
            let bb = CFRetained::from_raw(NonNull::new(bb).unwrap());
            let st = CMBlockBuffer::replace_data_bytes(
                NonNull::new(self.sample.as_ptr() as *mut c_void).unwrap(),
                &bb,
                0,
                len,
            );
            if st != 0 {
                return Err(DecodeError::Status { what: "CMBlockBufferReplaceDataBytes", status: st });
            }
            let mut sb: *mut CMSampleBuffer = std::ptr::null_mut();
            let st = CMSampleBuffer::create_ready(
                None,
                Some(&bb),
                Some(format),
                1,
                0,
                std::ptr::null(),
                1,
                &len,
                NonNull::from(&mut sb),
            );
            if st != 0 || sb.is_null() {
                return Err(DecodeError::Status { what: "CMSampleBufferCreateReady", status: st });
            }
            let sb = CFRetained::from_raw(NonNull::new(sb).unwrap());
            self.sink.out.lock().unwrap().clear();
            let mut info = VTDecodeInfoFlags::empty();
            let st = session.decode_frame(&sb, VTDecodeFrameFlags::empty(), std::ptr::null_mut(), &mut info);
            session.wait_for_asynchronous_frames();
            if st != 0 {
                return Err(DecodeError::Status { what: "VTDecompressionSessionDecodeFrame", status: st });
            }
        }
        let outs = std::mem::take(&mut *self.sink.out.lock().unwrap());
        let format = self.format.as_ref().unwrap();
        let mut pictures = Vec::with_capacity(outs.len());
        for o in outs {
            match o {
                Out::Picture(buf) => {
                    let pf = CVPixelBufferGetPixelFormatType(&buf);
                    let full = pf == NV12_FULL || pf == P010_FULL;
                    let tags = Tags::of_format(format, full);
                    pictures.push(Decoded {
                        width: CVPixelBufferGetWidth(&buf),
                        height: CVPixelBufferGetHeight(&buf),
                        ten_bit: pf == P010_VIDEO || pf == P010_FULL,
                        tags,
                        buffer: buf,
                    });
                }
                Out::Dropped => {}
                Out::Failed(status) => return Err(DecodeError::Status { what: "decoding", status }),
            }
        }
        Ok(pictures)
    }
}

impl Drop for VtDecoder {
    fn drop(&mut self) {
        self.drop_session(); // before the sink the callback writes to goes
    }
}

impl VtDecoder {
    /// The current format description's extensions (colour description, bit depth, the hvcC or avcC), for debugging.
    pub fn format_description(&self) -> String {
        match &self.format {
            Some(f) => {
                let ext = unsafe { f.extensions() };
                format!("{ext:?}")
            }
            None => "no format".into(),
        }
    }
}
