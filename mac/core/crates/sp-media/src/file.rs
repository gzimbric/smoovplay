//! Video files as sources (framegen/io.py FileSource and Ps5SimSource), decoded by AVAssetReader (VideoToolbox
//! underneath) on a thread of their own into NV12 pictures. They're converted like the stream's with
//! [`Converter::sdr`], by the file's own colour tags, or BT.601 when it has none, as FFmpeg's converter takes an
//! untagged picture.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::JoinHandle;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_foundation::{CFDictionary, CFRetained};
use objc2_core_media::CMFormatDescription;
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey,
    CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
};
use objc2_foundation::{NSDictionary, NSString, NSURL};
use sp_core::clock::now;
use sp_core::{Event, Frame, Source, SourceInfo, Tick};

use crate::cf;
use crate::colour::Tags;
use crate::convert::Converter;
use crate::pyrandom::PyRandom;
use crate::test_mode::SharedTestMode;
use crate::vt::{Decoded, NV12_FULL, NV12_VIDEO};

const TICK: f64 = 1.0 / 60.0;

/// A video file's frames in order, decoded on a background thread (framegen/io.py _Decoder), up to 16 ahead.
pub struct VideoFile {
    pub width: usize,
    pub height: usize,
    /// Frames a second, the track's nominal rate, or 60 if it doesn't say.
    pub fps: f64,
    rx: Receiver<Option<Frame>>,
    thread: Option<JoinHandle<()>>,
    ended: bool,
}

struct Meta {
    width: usize,
    height: usize,
    fps: f64,
}

impl VideoFile {
    pub fn open(path: &Path) -> sp_core::Result<VideoFile> {
        if !path.exists() {
            return Err(sp_core::Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, path.display().to_string())));
        }
        let (meta_tx, meta_rx) = sync_channel::<Result<Meta, String>>(1);
        let (tx, rx) = sync_channel::<Option<Frame>>(16);
        let path = path.to_path_buf();
        let thread = std::thread::Builder::new()
            .name("file-decoder".into())
            .spawn(move || {
                if let Err(e) = decode_file(&path, &meta_tx, &tx) {
                    let _ = meta_tx.try_send(Err(e.clone()));
                    eprintln!("[file] {}: {e}", path.display());
                }
                let _ = tx.send(None);
            })
            .map_err(sp_core::Error::from)?;
        let meta = meta_rx
            .recv()
            .map_err(|_| sp_core::Error::Other("the file decoder stopped".into()))?
            .map_err(sp_core::Error::Other)?;
        Ok(VideoFile { width: meta.width, height: meta.height, fps: meta.fps, rx, thread: Some(thread), ended: false })
    }

    /// The next frame, or None at the end.
    pub fn next_frame(&mut self) -> Option<Frame> {
        if self.ended {
            return None;
        }
        match self.rx.recv() {
            Ok(Some(f)) => Some(f),
            _ => {
                self.ended = true;
                None
            }
        }
    }
}

impl Drop for VideoFile {
    fn drop(&mut self) {
        // Without a receiver the decoder's next send fails, and it stops
        drop(std::mem::replace(&mut self.rx, sync_channel(1).1));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[allow(deprecated)] // tracksWithMediaType, because the async loader needs a run loop and this thread has none
fn decode_file(
    path: &Path,
    meta: &std::sync::mpsc::SyncSender<Result<Meta, String>>,
    tx: &std::sync::mpsc::SyncSender<Option<Frame>>,
) -> Result<(), String> {
    let mut conv = Converter::new().map_err(|e| e.to_string())?;
    unsafe {
        let abs = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(&abs.to_string_lossy()));
        let asset = AVURLAsset::URLAssetWithURL_options(&url, None);
        let tracks = asset.tracksWithMediaType(AVMediaTypeVideo.ok_or("no AVMediaTypeVideo")?);
        let track = tracks.firstObject().ok_or("no video track")?;
        let size = track.naturalSize();
        let fps = track.nominalFrameRate() as f64;
        let fps = if fps > 0.0 { fps } else { 60.0 };
        // The file's own colour tags, from its format description. Its pictures' attachments carry defaults.
        let descs = track.formatDescriptions();
        let first = descs.firstObject();
        let format: Option<&CMFormatDescription> = first.as_ref().map(|o| &*(Retained::as_ptr(o) as *const CMFormatDescription));
        let tags_of = |full: bool| match format {
            Some(f) => Tags::of_format(f, full),
            None => Tags { matrix: 2, transfer: 2, primaries: 2, full_range: full },
        };
        let reader = AVAssetReader::assetReaderWithAsset_error(&asset).map_err(|e| e.localizedDescription().to_string())?;
        let empty = cf::dict(&[]);
        let pf = cf::num(NV12_VIDEO as i64);
        let settings: CFRetained<CFDictionary> = cf::dict(&[
            (kCVPixelBufferPixelFormatTypeKey, &pf),
            (kCVPixelBufferMetalCompatibilityKey, cf::yes()),
            (kCVPixelBufferIOSurfacePropertiesKey, &empty),
        ]);
        // CFDictionary and NSDictionary are toll-free bridged
        let settings: &NSDictionary<NSString, AnyObject> = &*(CFRetained::as_ptr(&settings).as_ptr() as *const _);
        let output = AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(&track, Some(settings));
        output.setAlwaysCopiesSampleData(false);
        reader.addOutput(&output);
        if !reader.startReading() {
            return Err(reader.error().map(|e| e.localizedDescription().to_string()).unwrap_or("can't read it".into()));
        }
        let _ = meta.send(Ok(Meta { width: size.width.round() as usize, height: size.height.round() as usize, fps }));
        while let Some(sb) = output.copyNextSampleBuffer() {
            let Some(img) = sb.image_buffer() else { continue };
            let pf = CVPixelBufferGetPixelFormatType(&img);
            let d = Decoded {
                width: CVPixelBufferGetWidth(&img),
                height: CVPixelBufferGetHeight(&img),
                ten_bit: false,
                tags: tags_of(pf == NV12_FULL),
                buffer: img,
            };
            let f = conv.sdr(&d).map_err(|e| e.to_string())?;
            if tx.send(Some(f)).is_err() {
                return Ok(()); // the reader went
            }
        }
        if reader.status() == AVAssetReaderStatus::Failed {
            return Err(reader.error().map(|e| e.localizedDescription().to_string()).unwrap_or("read failed".into()));
        }
    }
    Ok(())
}

/// A file's frames as-is (a raw recording, say), test mode optionally applied as it reads.
pub struct FileSource {
    info: SourceInfo,
    dec: VideoFile,
    realtime: bool,
    max_ticks: Option<u64>,
    test: Option<SharedTestMode>,
    i: u64,
    t0: Option<f64>,
}

impl FileSource {
    pub fn open(path: &Path, realtime: bool, max_ticks: Option<u64>, test: Option<SharedTestMode>) -> sp_core::Result<FileSource> {
        let dec = VideoFile::open(path)?;
        let mut info = SourceInfo::of("file");
        info.live = realtime;
        info.hz = dec.fps;
        Ok(FileSource { info, dec, realtime, max_ticks, test, i: 0, t0: None })
    }
}

impl Source for FileSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (self.dec.width, self.dec.height)
    }
    fn next_event(&mut self) -> Option<Event> {
        let t0 = *self.t0.get_or_insert_with(now);
        loop {
            if self.max_ticks.is_some_and(|m| self.i >= m) {
                return None;
            }
            let f = self.dec.next_frame()?;
            let i = self.i;
            self.i += 1;
            if self.realtime {
                let wait = t0 + i as f64 * TICK - now();
                if wait > 0.0 {
                    std::thread::sleep(Duration::from_secs_f64(wait));
                }
            }
            let mut tick = Tick::new(f, now(), i);
            tick.t_source = Some(i as f64 / self.dec.fps);
            match &self.test {
                None => return Some(Event::Tick(tick)),
                Some(t) => {
                    if let Some(tick) = t.lock().unwrap().apply(tick) {
                        return Some(Event::Tick(tick));
                    }
                }
            }
        }
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.test.as_ref()?.lock().unwrap().gt(content)
    }
}

/// Options for [`Ps5SimSource`].
#[derive(Clone, Debug, Default)]
pub struct SimOptions {
    /// Deliver ticks at 60 Hz, with latency measured from the scheduled arrival. Otherwise as fast as they're taken.
    pub realtime: bool,
    /// The chance a game frame stays up three ticks instead of two (a dropped game frame).
    pub hitch: f64,
    /// Noise added to each tick's picture, sd in levels. 0 for none.
    pub noise: f64,
    pub max_ticks: Option<u64>,
    /// Python's random.Random seed. The same seed gives the same hitches as framegen/io.py.
    pub seed: u64,
}

/// A native 60 fps recording played as what a PS5 sends for a 30 fps game. Source frames 0, 2, 4 and so on are the
/// game's, each shown for two ticks (three with probability `hitch`). The odd frames are never shown and serve as
/// ground truth for the generated midpoints ([`Source::ground_truth`]). A file of about 30 fps is used frame for
/// frame instead, with every frame a game frame and no ground truth.
pub struct Ps5SimSource {
    info: SourceInfo,
    dec: VideoFile,
    opts: SimOptions,
    step: u64,
    rng: PyRandom,
    history: VecDeque<(i64, Frame)>,
    conv: Option<Converter>,
    // iteration state
    i: i64,
    tick: u64,
    current: Option<(i64, Frame)>,
    reps_left: u32,
    rep: u32,
    t0: Option<f64>,
}

impl Ps5SimSource {
    pub fn open(path: &Path, opts: SimOptions) -> sp_core::Result<Ps5SimSource> {
        let dec = VideoFile::open(path)?;
        let mut info = SourceInfo::of("sim");
        info.live = opts.realtime;
        let step = if dec.fps > 45.0 { 2 } else { 1 };
        if step == 1 {
            println!("[sim] {:.2} fps source: using every frame as a game frame (no ground truth)", dec.fps);
        }
        let conv = if opts.noise > 0.0 { Some(Converter::new()?) } else { None };
        let rng = PyRandom::new(opts.seed);
        Ok(Ps5SimSource {
            info,
            dec,
            opts,
            step,
            rng,
            history: VecDeque::new(),
            conv,
            i: -1,
            tick: 0,
            current: None,
            reps_left: 0,
            rep: 0,
            t0: None,
        })
    }

    pub fn fps(&self) -> f64 {
        self.dec.fps
    }
}

impl Source for Ps5SimSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (self.dec.width, self.dec.height)
    }
    fn next_event(&mut self) -> Option<Event> {
        let t0 = *self.t0.get_or_insert_with(now);
        while self.reps_left == 0 {
            // the next game frame
            let f = self.dec.next_frame()?;
            self.i += 1;
            self.history.push_back((self.i, f.clone()));
            while self.history.len() > 8 {
                self.history.pop_front();
            }
            if self.i % self.step as i64 != 0 {
                continue; // odd frames of a 60 fps source, standing for frames the game never rendered
            }
            self.reps_left = if self.rng.random() < self.opts.hitch { 3 } else { 2 };
            self.rep = 0;
            self.current = Some((self.i, f));
        }
        if self.opts.max_ticks.is_some_and(|m| self.tick >= m) {
            return None;
        }
        let (i, f) = self.current.clone().unwrap();
        let r = self.rep;
        self.rep += 1;
        self.reps_left -= 1;
        let mut out = f;
        if let Some(conv) = self.conv.as_mut() {
            let plane = self.rng.randrange(6);
            out = conv.add_noise(&out, plane, self.opts.noise as f32);
        }
        let t_arr = if self.opts.realtime {
            // A capture card delivers on the HDMI clock whether or not we keep up, so latency is measured from the
            // scheduled arrival (backlog included).
            let t = t0 + self.tick as f64 * TICK;
            let wait = t - now();
            if wait > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(wait));
            }
            t
        } else {
            now()
        };
        let mut tick = Tick::new(out, t_arr, self.tick);
        if self.step == 2 {
            tick.content = Some(i);
            tick.truth_new = Some(r == 0);
        }
        tick.t_source = Some(i as f64 / self.dec.fps);
        self.tick += 1;
        Some(Event::Tick(tick))
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.history.iter().find(|(i, _)| *i == content).map(|(_, f)| f.clone())
    }
}
