//! H.264 encoding with a VTCompressionSession on the media engine, in BT.709 limited range. It's set up like
//! replay.py's h264_videotoolbox, with realtime, prio_speed, allow_sw 0, bf 0, the bit rate for the nominal rate with
//! maxrate 1.5x, and a keyframe at most every two nominal seconds. The caller also forces one every KEY_EVERY_S.
//!
//! Frames go in as NV12 pixel buffers from the session's own pool. They're IOSurface-backed and Metal-compatible, so
//! the GPU writes them in place. Packets come out on VideoToolbox's thread to the callback given, in AVCC form with
//! the session's avcC.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use objc2_core_foundation::{CFArray, CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMFormatDescription, CMSampleBuffer, CMTime, CMTimeFlags, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
    kCMSampleAttachmentKey_NotSync, kCMTimeInvalid, kCMVideoCodecType_H264,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferPool, kCVImageBufferColorPrimaries_ITU_R_709_2, kCVImageBufferTransferFunction_ITU_R_709_2,
    kCVImageBufferYCbCrMatrix_ITU_R_709_2, kCVPixelBufferHeightKey, kCVPixelBufferIOSurfacePropertiesKey,
    kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferPoolAllocationThresholdKey,
    kCVPixelBufferWidthKey,
};
use objc2_video_toolbox::{
    VTCompressionSession, VTEncodeInfoFlags, VTSessionCopyProperty, VTSessionSetProperty,
    kVTCompressionPropertyKey_AllowFrameReordering, kVTCompressionPropertyKey_AverageBitRate,
    kVTCompressionPropertyKey_ColorPrimaries, kVTCompressionPropertyKey_DataRateLimits,
    kVTCompressionPropertyKey_ExpectedFrameRate, kVTCompressionPropertyKey_MaxKeyFrameInterval,
    kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality, kVTCompressionPropertyKey_ProfileLevel,
    kVTCompressionPropertyKey_RealTime, kVTCompressionPropertyKey_TransferFunction,
    kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder, kVTCompressionPropertyKey_YCbCrMatrix,
    kVTEncodeFrameOptionKey_ForceKeyFrame, kVTProfileLevel_H264_High_AutoLevel,
    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
};
use sp_media::cf;

use crate::track::Format;
use crate::{Error, MAX_BITRATE, Result, TB};

/// NV12, limited range ('420v'), which is the encoder's own format.
pub const NV12: u32 = cf::fourcc(b"420v");

#[derive(Clone, Copy, Debug)]
pub struct VideoConfig {
    pub width: usize,
    pub height: usize,
    /// The rate the bit rate is given for (replay.py AFTER_NOMINAL or BEFORE_NOMINAL).
    pub nominal: u32,
    /// Bits per pixel per frame at that rate (AFTER_BPP or BEFORE_BPP).
    pub bpp: f64,
    /// Real time (instant replay's tracks), or as fast as it likes (the before & after, behind them).
    pub realtime: bool,
    /// The frame rate it'll really get, as a hint to the rate control. None gives no hint, like FFmpeg's
    /// h264_videotoolbox, and the rate control then gives each frame about the same share at any rate, as replay.py
    /// means it to.
    pub expected_fps: Option<f64>,
}

impl VideoConfig {
    /// replay.py _configure_video's rate, which is bits per frame times the nominal rate, at most MAX_BITRATE.
    pub fn bit_rate(&self) -> i64 {
        (self.bpp * (self.width * self.height) as f64 * self.nominal as f64).min(MAX_BITRATE) as i64
    }
}

/// A packet, which is an access unit in AVCC form.
#[derive(Debug)]
pub struct Encoded {
    /// On the 90 kHz clock, as given.
    pub pts: i64,
    pub key: bool,
    pub data: Vec<u8>,
    pub format: Arc<Format>,
    /// Seconds from encode() to here.
    pub latency: f64,
}

pub type OnOutput = Box<dyn Fn(std::result::Result<Encoded, i32>) + Send + Sync>;

struct Ctx {
    on_output: OnOutput,
    width: u32,
    height: u32,
    format: Mutex<Option<(CFRetained<CMFormatDescription>, Arc<Format>)>>,
}

pub struct VtEncoder {
    session: CFRetained<VTCompressionSession>,
    pool: CFRetained<CVPixelBufferPool>,
    ctx: *mut Ctx,
    force_key: CFRetained<CFDictionary>,
    pub width: usize,
    pub height: usize,
    pub hardware: bool,
    pub config: VideoConfig,
    invalidated: Mutex<bool>,
}

// The session, its pool and the dictionaries are CF objects, safe between threads. Encoding happens from one thread
// at a time.
unsafe impl Send for VtEncoder {}
unsafe impl Sync for VtEncoder {}

fn avcc_of(format: &CMFormatDescription) -> Option<Vec<u8>> {
    unsafe {
        let atoms = format.extension(kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms)?;
        let atoms = atoms.downcast::<CFDictionary>().ok()?;
        let key = CFString::from_str("avcC");
        let v = atoms.value(&*key as *const CFString as *const c_void);
        if v.is_null() {
            return None;
        }
        let data = &*(v as *const CFData);
        Some(data.to_vec())
    }
}

fn is_key(sb: &CMSampleBuffer) -> bool {
    unsafe {
        let Some(arr) = sb.sample_attachments_array(false) else { return true };
        if arr.count() < 1 {
            return true;
        }
        let d = arr.value_at_index(0) as *const CFDictionary;
        if d.is_null() {
            return true;
        }
        let v = (*d).value(kCMSampleAttachmentKey_NotSync as *const CFString as *const c_void);
        if v.is_null() {
            return true;
        }
        !(*(v as *const CFBoolean)).as_bool()
    }
}

unsafe extern "C-unwind" fn output_callback(
    refcon: *mut c_void,
    frame: *mut c_void,
    status: i32,
    _flags: VTEncodeInfoFlags,
    sb: *mut CMSampleBuffer,
) {
    let ctx = unsafe { &*(refcon as *const Ctx) };
    // The frame's refcon is when it was handed over
    let latency = if frame.is_null() { f64::NAN } else { sp_core::clock::now() - unsafe { *Box::from_raw(frame as *mut f64) } };
    if status != 0 {
        (ctx.on_output)(Err(status));
        return;
    }
    let Some(sb) = NonNull::new(sb) else {
        (ctx.on_output)(Err(-1)); // dropped
        return;
    };
    let sb = unsafe { sb.as_ref() };
    let out = unsafe {
        let t = sb.presentation_time_stamp();
        let pts = if t.timescale as i64 == TB { t.value } else { (t.value as i128 * TB as i128 / t.timescale.max(1) as i128) as i64 };
        let Some(bb) = sb.data_buffer() else {
            (ctx.on_output)(Err(-2));
            return;
        };
        let len = bb.data_length();
        let mut data = vec![0u8; len];
        let st = bb.copy_data_bytes(0, len, NonNull::new(data.as_mut_ptr() as *mut c_void).unwrap());
        if st != 0 {
            (ctx.on_output)(Err(st));
            return;
        }
        let format = match sb.format_description() {
            Some(fd) => {
                let mut slot = ctx.format.lock().unwrap();
                match &*slot {
                    Some((p, f)) if CFRetained::as_ptr(p) == CFRetained::as_ptr(&fd) => f.clone(),
                    _ => {
                        let avcc = avcc_of(&fd).unwrap_or_default();
                        let f = Arc::new(Format::H264 { width: ctx.width, height: ctx.height, avcc });
                        *slot = Some((fd, f.clone()));
                        f
                    }
                }
            }
            None => match &*ctx.format.lock().unwrap() {
                Some((_, f)) => f.clone(),
                None => {
                    (ctx.on_output)(Err(-3));
                    return;
                }
            },
        };
        Encoded { pts, key: is_key(sb), data, format, latency }
    };
    (ctx.on_output)(Ok(out));
}

fn set(session: &VTCompressionSession, key: &CFString, value: &CFType) -> i32 {
    unsafe { VTSessionSetProperty(session, key, Some(value)) }
}

impl VtEncoder {
    pub fn new(config: VideoConfig, on_output: OnOutput) -> Result<VtEncoder> {
        let (w, h) = (config.width & !1, config.height & !1);
        if w < 2 || h < 2 {
            return Err(Error::Other(format!("frames too small to encode ({w}x{h})")));
        }
        let ctx = Box::into_raw(Box::new(Ctx { on_output, width: w as u32, height: h as u32, format: Mutex::new(None) }));
        let spec = unsafe { cf::dict(&[(kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder, cf::yes())]) };
        let empty = cf::dict(&[]);
        let (pf, wn, hn) = (cf::num(NV12 as i64), cf::num(w as i64), cf::num(h as i64));
        let attrs = unsafe {
            cf::dict(&[
                (kCVPixelBufferPixelFormatTypeKey, &pf),
                (kCVPixelBufferWidthKey, &wn),
                (kCVPixelBufferHeightKey, &hn),
                (kCVPixelBufferMetalCompatibilityKey, cf::yes()),
                (kCVPixelBufferIOSurfacePropertiesKey, &empty),
            ])
        };
        let mut out: *mut VTCompressionSession = std::ptr::null_mut();
        let st = unsafe {
            VTCompressionSession::create(
                None,
                w as i32,
                h as i32,
                kCMVideoCodecType_H264,
                Some(&spec),
                Some(&attrs),
                None,
                Some(output_callback),
                ctx as *mut c_void,
                NonNull::from(&mut out),
            )
        };
        if st != 0 || out.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
            return Err(Error::Status { what: "VTCompressionSessionCreate (H.264, hardware)", status: st });
        }
        let session = unsafe { CFRetained::from_raw(NonNull::new(out).unwrap()) };
        let rate = config.bit_rate();
        let no: &CFType = CFBoolean::new(false).as_ref();
        let yes: &CFType = cf::yes().as_ref();
        let real: &CFType = CFBoolean::new(config.realtime).as_ref();
        let max_bytes = cf::num(rate * 3 / 2 / 8);
        let one = cf::num(1);
        let limits = CFArray::<CFNumber>::from_retained_objects(&[max_bytes, one]);
        let (rate_n, gop_n, fps_n) = (cf::num(rate), cf::num(2 * config.nominal as i64), CFNumber::new_f64(config.expected_fps.unwrap_or(0.0)));
        let props: [(&CFString, &CFType, &str); 11] = unsafe {
            [
                (kVTCompressionPropertyKey_RealTime, real, "RealTime"),
                (kVTCompressionPropertyKey_AllowFrameReordering, no, "AllowFrameReordering"),
                (kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_H264_High_AutoLevel.as_ref(), "ProfileLevel"),
                (kVTCompressionPropertyKey_AverageBitRate, rate_n.as_ref(), "AverageBitRate"),
                (kVTCompressionPropertyKey_DataRateLimits, limits.as_ref(), "DataRateLimits"),
                (kVTCompressionPropertyKey_MaxKeyFrameInterval, gop_n.as_ref(), "MaxKeyFrameInterval"),
                (kVTCompressionPropertyKey_ExpectedFrameRate, fps_n.as_ref(), "ExpectedFrameRate"),
                (kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality, if config.realtime { yes } else { no }, "PrioritizeEncodingSpeedOverQuality"),
                (kVTCompressionPropertyKey_ColorPrimaries, kCVImageBufferColorPrimaries_ITU_R_709_2.as_ref(), "ColorPrimaries"),
                (kVTCompressionPropertyKey_TransferFunction, kCVImageBufferTransferFunction_ITU_R_709_2.as_ref(), "TransferFunction"),
                (kVTCompressionPropertyKey_YCbCrMatrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2.as_ref(), "YCbCrMatrix"),
            ]
        };
        for (k, v, name) in props {
            if name == "ExpectedFrameRate" && config.expected_fps.is_none() {
                continue;
            }
            let st = set(&session, k, v);
            if st != 0 && name != "ExpectedFrameRate" {
                eprintln!("[replay] VideoToolbox didn't take {name} ({st})");
            }
        }
        let mut hardware = false;
        unsafe {
            session.prepare_to_encode_frames();
            let mut hw: *const CFType = std::ptr::null();
            if VTSessionCopyProperty(
                &session,
                kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                None,
                &mut hw as *mut _ as *mut c_void,
            ) == 0
                && !hw.is_null()
            {
                let v: CFRetained<CFType> = CFRetained::from_raw(NonNull::new(hw as *mut CFType).unwrap());
                hardware = v.downcast::<CFBoolean>().map(|b| b.as_bool()).unwrap_or(false);
            }
        }
        let Some(pool) = (unsafe { session.pixel_buffer_pool() }) else {
            unsafe { session.invalidate() };
            drop(unsafe { Box::from_raw(ctx) });
            return Err(Error::Other("VideoToolbox gave no pixel buffer pool".into()));
        };
        let force_key = unsafe { cf::dict(&[(kVTEncodeFrameOptionKey_ForceKeyFrame, cf::yes())]) };
        Ok(VtEncoder { session, pool, ctx, force_key, width: w, height: h, hardware, config, invalidated: Mutex::new(false) })
    }

    /// A buffer from the session's pool to draw the next frame into. None if `limit` are already out because the
    /// encoder is behind, or if the pool can't make one.
    pub fn pixel_buffer(&self, limit: usize) -> Option<CFRetained<CVPixelBuffer>> {
        let n = cf::num(limit as i64);
        let aux = unsafe { cf::dict(&[(kCVPixelBufferPoolAllocationThresholdKey, &n)]) };
        let mut out: *mut CVPixelBuffer = std::ptr::null_mut();
        let st = unsafe { CVPixelBufferPool::create_pixel_buffer_with_aux_attributes(None, &self.pool, Some(&aux), NonNull::from(&mut out)) };
        if st != 0 || out.is_null() {
            return None;
        }
        Some(unsafe { CFRetained::from_raw(NonNull::new(out).unwrap()) })
    }

    /// Hands a frame to the encoder, shown at `pts` (90 kHz). `key` forces a keyframe.
    pub fn encode(&self, buf: &CVPixelBuffer, pts: i64, key: bool) -> Result<()> {
        let t = CMTime { value: pts, timescale: TB as i32, flags: CMTimeFlags::Valid, epoch: 0 };
        let when = Box::into_raw(Box::new(sp_core::clock::now()));
        let st = unsafe {
            self.session.encode_frame(
                buf,
                t,
                kCMTimeInvalid,
                if key { Some(&self.force_key) } else { None },
                when as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        if st != 0 {
            drop(unsafe { Box::from_raw(when) }); // no callback comes for a frame it refused
        }
        crate::status("VTCompressionSessionEncodeFrame", st)
    }

    /// Every frame given so far comes out (through the callback) before this returns.
    pub fn complete(&self) {
        if !*self.invalidated.lock().unwrap() {
            unsafe {
                self.session.complete_frames(kCMTimeInvalid);
            }
        }
    }

    /// Tears the session down now, finishing the frames still in it first. No callbacks come after this.
    pub fn invalidate(&self) {
        let mut done = self.invalidated.lock().unwrap();
        if !*done {
            unsafe {
                self.session.complete_frames(kCMTimeInvalid);
                self.session.invalidate();
            }
            *done = true;
        }
    }
}

impl Drop for VtEncoder {
    fn drop(&mut self) {
        self.invalidate();
        drop(unsafe { Box::from_raw(self.ctx) });
    }
}
