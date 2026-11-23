//! Files standing in for a card (capture.py CardSource.video_file and raw_file), played at their own rate from when
//! the pipeline starts reading, through the same per-frame path as a card's frames ([`crate::pipe`]).
//!
//! - [`CardSource::video_file`] takes any video AVFoundation reads, such as H.264, HEVC or ProRes in .mp4 or .mov.
//!   AVAssetReader decodes it straight into NV12, read in the file's own colours from its matrix and range tags.
//!   Untagged is BT.601 limited range, as FFmpeg takes it. That's the picture the Python's stand-in ends up with. It
//!   reformats the recording into the raw path's colours and reads it back with them, a round trip, so
//!   --capture-colour doesn't change a stand-in there either. Each picture is handed on as a card's would be. Its
//!   sound, 16-bit PCM at the file's own rate (two channels at most), waits until the picture gets to it. It goes in
//!   pieces of at most 1024 sample frames, an AAC frame and what PyAV hands the Python, so it comes out with its
//!   picture.
//! - [`CardSource::raw_file`] reads raw frames as a card sends them (nv12, yuyv422, uyvy422, bgr0), at `fps`.
//! - [`CardSource::raw_frames`] does the same from memory, with None for a frame the card sent damaged. It's for
//!   tests.

use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVAssetTrack, AVMediaTypeAudio, AVMediaTypeVideo,
    AVURLAsset,
};
use objc2_avf_audio::{
    AVFormatIDKey, AVLinearPCMBitDepthKey, AVLinearPCMIsBigEndianKey, AVLinearPCMIsFloatKey,
    AVLinearPCMIsNonInterleaved, AVNumberOfChannelsKey, AVSampleRateKey,
};
use objc2_core_audio_types::kAudioFormatLinearPCM;
use objc2_core_foundation::{CFBoolean, CFRetained, CFString};
use objc2_core_media::{
    kCMFormatDescriptionExtension_FullRangeVideo, kCMFormatDescriptionExtension_YCbCrMatrix, CMFormatDescription,
    CMSampleBuffer,
};
use objc2_core_video::{kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
use sp_core::clock::now;
use sp_media::cf;

use crate::pcm::{asbd, pcm_of};
use crate::pipe::{convert_pixels, convert_raw, Pipe, PipeConfig, NV12_FULL, NV12_VIDEO};
use crate::source::{CardOptions, CardSource};

/// The most sample frames of a file's sound handed on at once.
const SOUND_PIECE: usize = 1024;

pub(crate) struct Meta {
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub sound: bool,
    pub bt709: bool,
    pub full_range: bool,
}

fn err(e: impl std::fmt::Display) -> sp_core::Error {
    sp_core::Error::Other(e.to_string())
}

/// AVAssetReader's output settings for video. NV12, IOSurface-backed, in the range the raw path reads it as.
fn video_settings(full_range: bool) -> CFRetained<objc2_core_foundation::CFDictionary> {
    let empty = cf::dict(&[]);
    let pf = cf::num(if full_range { NV12_FULL } else { NV12_VIDEO } as i64);
    unsafe {
        cf::dict(&[
            (kCVPixelBufferPixelFormatTypeKey, &pf),
            (kCVPixelBufferMetalCompatibilityKey, cf::yes()),
            (kCVPixelBufferIOSurfacePropertiesKey, &empty),
        ])
    }
}

/// 16-bit interleaved linear PCM, `channels` channels, at `rate` (0 for the source's).
pub(crate) fn pcm_settings(channels: u32, rate: f64) -> Retained<NSDictionary<NSString, AnyObject>> {
    unsafe {
        let mut keys: Vec<&NSString> = vec![
            AVFormatIDKey.unwrap(),
            AVLinearPCMBitDepthKey.unwrap(),
            AVLinearPCMIsFloatKey.unwrap(),
            AVLinearPCMIsBigEndianKey.unwrap(),
            AVLinearPCMIsNonInterleaved.unwrap(),
        ];
        let mut values: Vec<Retained<NSNumber>> = vec![
            NSNumber::new_u32(kAudioFormatLinearPCM),
            NSNumber::new_u32(16),
            NSNumber::new_bool(false),
            NSNumber::new_bool(false),
            NSNumber::new_bool(false),
        ];
        if channels > 0 {
            keys.push(AVNumberOfChannelsKey.unwrap());
            values.push(NSNumber::new_u32(channels));
        }
        if rate > 0.0 {
            keys.push(AVSampleRateKey.unwrap());
            values.push(NSNumber::new_f64(rate));
        }
        let objs: Vec<&AnyObject> = values.iter().map(|v| &*****v).collect();
        NSDictionary::from_slices(&keys, &objs)
    }
}

pub(crate) struct Readers {
    pub video: (Retained<AVAssetReader>, Retained<AVAssetReaderTrackOutput>),
    pub sound: Option<(Retained<AVAssetReader>, Retained<AVAssetReaderTrackOutput>)>,
}

/// A video track's own colours as (BT.709, full range), from its format description's tags.
fn track_colours(track: &AVAssetTrack) -> (bool, bool) {
    unsafe {
        let first = track.formatDescriptions().firstObject();
        let Some(desc) = first.as_ref().map(|o| &*(Retained::as_ptr(o) as *const CMFormatDescription)) else {
            return (false, false);
        };
        let matrix: Option<String> = desc
            .extension(kCMFormatDescriptionExtension_YCbCrMatrix)
            .and_then(|v| v.downcast::<CFString>().ok())
            .map(|s| s.to_string());
        let full = desc
            .extension(kCMFormatDescriptionExtension_FullRangeVideo)
            .and_then(|v| v.downcast::<CFBoolean>().ok())
            .is_some_and(|b| b.as_bool());
        (matches!(matrix.as_deref(), Some("ITU_R_709_2") | Some("ITU_R_2020")), full)
    }
}

#[allow(deprecated)] // tracksWithMediaType, because the async loader needs a run loop and this thread has none
pub(crate) fn open_readers(path: &Path, want_sound: bool) -> Result<(Readers, Meta), String> {
    unsafe {
        let abs = std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(&abs.to_string_lossy()));
        let asset = AVURLAsset::URLAssetWithURL_options(&url, None);
        let track = asset
            .tracksWithMediaType(AVMediaTypeVideo.ok_or("no AVMediaTypeVideo")?)
            .firstObject()
            .ok_or_else(|| format!("{}: no video track AVFoundation can read", path.display()))?;
        let size = track.naturalSize();
        let fps = track.nominalFrameRate() as f64;
        let (bt709, full_range) = track_colours(&track);
        let reader = |t: &AVAssetTrack, settings: &NSDictionary<NSString, AnyObject>| -> Result<_, String> {
            let r = AVAssetReader::assetReaderWithAsset_error(&asset).map_err(|e| e.localizedDescription().to_string())?;
            let o = AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(t, Some(settings));
            o.setAlwaysCopiesSampleData(false);
            r.addOutput(&o);
            if !r.startReading() {
                return Err(r.error().map(|e| e.localizedDescription().to_string()).unwrap_or("can't read it".into()));
            }
            Ok((r, o))
        };
        let vs = video_settings(full_range);
        let vs: &NSDictionary<NSString, AnyObject> = &*(CFRetained::as_ptr(&vs).as_ptr() as *const _);
        let video = reader(&track, vs)?;
        let mut sound = None;
        let audio_track = if want_sound {
            asset.tracksWithMediaType(AVMediaTypeAudio.ok_or("no AVMediaTypeAudio")?).firstObject()
        } else {
            None
        };
        if let Some(at) = audio_track {
            let first = at.formatDescriptions().firstObject();
            let desc = first.as_ref().map(|o| &*(Retained::as_ptr(o) as *const CMFormatDescription));
            if let Some(a) = desc.and_then(asbd) {
                let settings = pcm_settings(a.mChannelsPerFrame.clamp(1, 2), a.mSampleRate);
                match reader(&at, &settings) {
                    Ok(r) => sound = Some(r),
                    Err(e) => println!("[capture] the recording's sound can't be read ({e}); playing without it"),
                }
            }
        }
        let meta = Meta {
            w: size.width.round() as usize,
            h: size.height.round() as usize,
            fps: if fps > 0.0 { fps } else { 60.0 },
            sound: sound.is_some(),
            bt709,
            full_range,
        };
        Ok((Readers { video, sound }, meta))
    }
}

fn seconds(t: objc2_core_media::CMTime) -> Option<f64> {
    let s = unsafe { t.seconds() };
    s.is_finite().then_some(s)
}

/// Hands the picture's sound to the pipe, up to `until`. It's read from the file as far as needed, in pieces.
struct SoundReader {
    out: Retained<AVAssetReaderTrackOutput>,
    _reader: Retained<AVAssetReader>,
    /// The end of what's been read so far (its time in the file).
    read_to: f64,
    done: bool,
}

impl SoundReader {
    fn fill(&mut self, pipe: &Pipe, until: f64) {
        while !self.done && self.read_to <= until {
            let Some(sb) = (unsafe { self.out.copyNextSampleBuffer() }) else {
                self.done = true;
                break;
            };
            let sb: &CMSampleBuffer = &sb;
            let Some((pcm, rate, ch)) = pcm_of(sb) else { continue };
            let t0 = seconds(unsafe { sb.presentation_time_stamp() }).unwrap_or(self.read_to);
            let frames = pcm.len() / ch as usize;
            for (k, piece) in pcm.chunks(SOUND_PIECE * ch as usize).enumerate() {
                pipe.hold_sound(t0 + (k * SOUND_PIECE) as f64 / rate as f64, piece.to_vec(), rate, ch);
            }
            self.read_to = t0 + frames as f64 / rate as f64;
        }
    }
}

impl CardSource {
    /// A recording standing in for a card, decoded and sent through the raw NV12 path at its own rate. With
    /// `opts.audio`, its sound is held until the picture gets to it.
    pub fn video_file(path: &Path, opts: CardOptions) -> sp_core::Result<CardSource> {
        if !path.exists() {
            return Err(sp_core::Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, path.display().to_string())));
        }
        let want_sound = opts.audio.is_some();
        let (meta_tx, meta_rx) = sync_channel::<Result<Meta, String>>(1);
        let (pipe_tx, pipe_rx) = sync_channel::<Arc<Pipe>>(1);
        let p: PathBuf = path.to_path_buf();
        let thread = std::thread::Builder::new()
            .name("capture-file".into())
            .spawn(move || {
                let (readers, meta) = match open_readers(&p, want_sound) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = meta_tx.send(Err(e));
                        return;
                    }
                };
                let fps = meta.fps;
                let _ = meta_tx.send(Ok(meta));
                let Ok(pipe) = pipe_rx.recv() else { return };
                play_file(&pipe, readers, fps);
            })
            .map_err(sp_core::Error::from)?;
        let meta = meta_rx.recv().map_err(|_| err("the file reader stopped"))?.map_err(err)?;
        let (bt709, full_range) = (meta.bt709, meta.full_range);
        let pipe = Arc::new(Pipe::new(PipeConfig {
            w: meta.w,
            h: meta.h,
            fps: meta.fps,
            bt709,
            full_range,
            fmt: "nv12".into(),
            live: false,
            queue_size: opts.queue_size,
            test: opts.test,
            give_up_s: opts.damaged_give_up_s,
            sound_out: opts.audio,
            has_sound: meta.sound,
        })?);
        pipe_tx.send(pipe.clone()).map_err(|_| err("the file reader stopped"))?;
        let src = CardSource::new(pipe, "Capture card (a recording standing in)", true, None, Some(thread));
        if meta.sound {
            src.pipe.say("[capture] sound: the recording's, played on this Mac, held back to meet the picture".into());
        }
        Ok(src)
    }

    /// Raw frames read exactly as a card's are (capture.py CardSource.raw_file). `fmt` is nv12, yuyv422, uyvy422 or
    /// bgr0, the size is `w` x `h`, and they play at `fps`.
    pub fn raw_file(path: &Path, w: usize, h: usize, fmt: &str, fps: f64, opts: CardOptions) -> sp_core::Result<CardSource> {
        let data = std::fs::read(path)?;
        let n = frame_bytes(fmt, w, h)?;
        let frames: Vec<Option<Vec<u8>>> = data.chunks(n).map(|c| Some(c.to_vec())).collect();
        Self::raw_source(frames, w, h, fmt, fps, opts, "Capture card (a raw file standing in)")
    }

    /// Raw frames from memory, as raw_file reads them. None stands for a frame the card sent that can't be read.
    pub fn raw_frames(frames: Vec<Option<Vec<u8>>>, w: usize, h: usize, fmt: &str, fps: f64, opts: CardOptions) -> sp_core::Result<CardSource> {
        frame_bytes(fmt, w, h)?;
        Self::raw_source(frames, w, h, fmt, fps, opts, "Capture card (raw frames standing in)")
    }

    fn raw_source(frames: Vec<Option<Vec<u8>>>, w: usize, h: usize, fmt: &str, fps: f64, opts: CardOptions, name: &str) -> sp_core::Result<CardSource> {
        let (bt709, full_range) = opts.colour.resolve(h, "");
        let pipe = Arc::new(Pipe::new(PipeConfig {
            w,
            h,
            fps,
            bt709,
            full_range,
            fmt: fmt.into(),
            live: false,
            queue_size: opts.queue_size,
            test: opts.test,
            give_up_s: opts.damaged_give_up_s,
            sound_out: None,
            has_sound: false,
        })?);
        let (p, fmt2) = (pipe.clone(), fmt.to_string());
        let thread = std::thread::Builder::new()
            .name("capture-raw".into())
            .spawn(move || play_raw(&p, frames, &fmt2))
            .map_err(sp_core::Error::from)?;
        Ok(CardSource::new(pipe, name, true, None, Some(thread)))
    }
}

fn frame_bytes(fmt: &str, w: usize, h: usize) -> sp_core::Result<usize> {
    let bpp = crate::devices::bytes_per_pixel(fmt).filter(|_| matches!(fmt, "nv12" | "yuyv422" | "uyvy422" | "bgr0"));
    Ok((w as f64 * h as f64 * bpp.ok_or_else(|| err(format!("can't read raw {fmt} frames")))?) as usize)
}

/// A card delivers on the HDMI clock, so frame n of something standing in goes out at t0 + n / fps.
fn pace(t0: f64, n: u64, fps: f64) -> Option<f64> {
    let wait = t0 + n as f64 / fps - now();
    if wait > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(wait));
        return Some(now());
    }
    None
}

fn play_raw(pipe: &Pipe, frames: Vec<Option<Vec<u8>>>, fmt: &str) {
    pipe.wait_go();
    let t0 = now();
    for (n, f) in frames.into_iter().enumerate() {
        let n = n as u64;
        if pipe.stopped() {
            break;
        }
        let mut t_arrival = now();
        if let Some(t) = pace(t0, n, pipe.fps) {
            t_arrival = t;
        }
        let t_src = Some(n as f64 / pipe.fps); // the clock of the card it stands in for
        let (w, h, bt709, full) = (pipe.w, pipe.h, pipe.bt709, pipe.full_range);
        let r = pipe.frame(t_arrival, t_src, |conv| match &f {
            Some(data) => convert_raw(conv, data, fmt, w, h, bt709, full),
            None => Err("a frame the decoder couldn't read".into()),
        });
        if let Err(e) = r {
            pipe.stop_with(&e);
            break;
        }
    }
    pipe.end();
}

fn play_file(pipe: &Pipe, readers: Readers, fps: f64) {
    let Readers { video: (vreader, vout), sound } = readers;
    let mut sound = sound.map(|(r, o)| SoundReader { out: o, _reader: r, read_to: 0.0, done: false });
    pipe.wait_go();
    let t0 = now();
    let mut n = 0u64;
    while !pipe.stopped() {
        let Some(sb) = (unsafe { vout.copyNextSampleBuffer() }) else { break };
        let sb: &CMSampleBuffer = &sb;
        if unsafe { sb.num_samples() } == 0 {
            continue; // a marker, no picture
        }
        let t_src = seconds(unsafe { sb.presentation_time_stamp() }).unwrap_or(n as f64 / fps);
        let mut t_arrival = now();
        if let Some(t) = pace(t0, n, fps) {
            t_arrival = t;
        }
        // a file's sound goes out as the picture gets to it
        let until = t_src + 1.0 / fps;
        if let Some(s) = sound.as_mut() {
            s.fill(pipe, until);
        }
        pipe.release_sound(until);
        let img = unsafe { sb.image_buffer() };
        let (bt709, full) = (pipe.bt709, pipe.full_range);
        let r = pipe.frame(t_arrival, Some(t_src), |conv| match &img {
            Some(img) => convert_pixels(conv, img, bt709, full),
            None => Err("a sample with no picture in it".into()),
        });
        if let Err(e) = r {
            pipe.stop_with(&e);
            break;
        }
        n += 1;
    }
    if unsafe { vreader.status() } == AVAssetReaderStatus::Failed {
        let why = unsafe { vreader.error() }.map(|e| e.localizedDescription().to_string()).unwrap_or("read failed".into());
        pipe.stop_with(&why);
    }
    pipe.end();
}
