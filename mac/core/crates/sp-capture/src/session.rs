//! A real card through AVFoundation (capture.py CardSource.open): one AVCaptureSession with the card's video and its
//! sound, so both are timed by the session's clock (as DirectShow's one graph times both on Windows).
//!
//! - The card's mode is picked from its formats ([`crate::pick_mode`]), set as the device's active format and frame
//!   duration, and held across startRunning (AVFoundation keeps it then).
//! - The video data output hands over IOSurface-backed CVPixelBuffers: NV12 in the card's own format when it sends
//!   NV12 (no conversion), otherwise NV12 converted by macOS (4:2:2 and MJPEG, which macOS decodes). Each one is
//!   copied tightly into the converter's buffer and converted on the GPU with capture.py raw_to_bgr's maths, on the
//!   output's own queue, so the sample buffer goes straight back to the card's pool.
//! - Timestamps are the session's synchronisation clock converted to host time (sp_core::clock), so `t_source` and
//!   `t_arrival` compare directly, and a sound packet's lateness against the picture's is measured on one clock.
//! - Frames macOS drops before handing them over are numbered all the same; a drop for a break in the card's stream
//!   counts as a damaged frame.
//! - Sound: the card's own audio port when it has one (a muxed device), else the sound device given, as a second
//!   input in the same session, asked for 16-bit PCM (two channels at most) at the device's rate, on a queue of its
//!   own so the picture's work never holds it up.
//!
//! Untested with a real card (see the crate docs for what to try).

use std::sync::{Arc, OnceLock};

use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{define_class, msg_send, AnyThread, DefinedClass, Message};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureAudioDataOutput, AVCaptureAudioDataOutputSampleBufferDelegate, AVCaptureConnection,
    AVCaptureDevice, AVCaptureDeviceInput, AVCaptureInput, AVCaptureOutput, AVCaptureSession,
    AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaType, AVMediaTypeAudio,
    AVMediaTypeVideo,
};
use objc2_core_foundation::{CFRetained, CFString, CFType};
use objc2_core_media::{
    kCMSampleBufferAttachmentKey_DroppedFrameReason, kCMSampleBufferDroppedFrameReason_Discontinuity, CMClock,
    CMGetAttachment, CMSampleBuffer, CMSyncConvertTime, CMTime,
};
use objc2_core_video::kCVPixelBufferPixelFormatTypeKey;
use objc2_foundation::{NSDictionary, NSObject, NSObjectProtocol, NSString};
use sp_core::clock::now;
use sp_media::cf;

use crate::devices::{self, device_modes, g, has_sound, open_rate, pick_mode, DeviceMode, Mode};
use crate::file::pcm_settings;
use crate::pcm::{asbd, pcm_of};
use crate::pipe::{clip, convert_pixels, Pipe, PipeConfig, NV12_FULL, NV12_VIDEO};
use crate::source::{CardOptions, CardSource};

/// The session's clock, shared with the outputs' delegates once the session runs.
pub(crate) struct SyncClock(Retained<CMClock>);
// CMClocks are thread-safe.
unsafe impl Send for SyncClock {}
unsafe impl Sync for SyncClock {}

pub(crate) struct DelegateIvars {
    pipe: Arc<Pipe>,
    audio: bool,
    clock: Arc<OnceLock<SyncClock>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and OutputDelegate doesn't implement Drop.
    #[unsafe(super(NSObject))]
    #[name = "SPCaptureOutputDelegate"]
    #[ivars = DelegateIvars]
    pub(crate) struct OutputDelegate;

    unsafe impl NSObjectProtocol for OutputDelegate {}

    // (the audio output's delegate protocol has the same two methods: `audio` says which output this one serves)
    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for OutputDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(&self, _output: &AVCaptureOutput, sample: &CMSampleBuffer, _connection: Option<&AVCaptureConnection>) {
            let t_arrival = now();
            let iv = self.ivars();
            if iv.audio {
                iv.sound(sample, t_arrival);
            } else {
                iv.picture(sample, t_arrival);
            }
        }

        #[unsafe(method(captureOutput:didDropSampleBuffer:fromConnection:))]
        fn did_drop(&self, _output: &AVCaptureOutput, sample: &CMSampleBuffer, _connection: Option<&AVCaptureConnection>) {
            let iv = self.ivars();
            if !iv.audio {
                iv.dropped(sample);
            }
        }
    }

    unsafe impl AVCaptureAudioDataOutputSampleBufferDelegate for OutputDelegate {}
);

impl OutputDelegate {
    pub(crate) fn new(pipe: Arc<Pipe>, audio: bool, clock: Arc<OnceLock<SyncClock>>) -> Retained<OutputDelegate> {
        let this = Self::alloc().set_ivars(DelegateIvars { pipe, audio, clock });
        unsafe { msg_send![super(this), init] }
    }

    /// What AVFoundation calls with each sample buffer (tests call it with a file's).
    #[cfg(test)]
    fn deliver(&self, output: &AVCaptureOutput, sample: &CMSampleBuffer) {
        unsafe {
            let _: () = msg_send![self, captureOutput: output, didOutputSampleBuffer: sample, fromConnection: std::ptr::null::<AnyObject>()];
        }
    }

    /// What AVFoundation calls for a frame it dropped (tests).
    #[cfg(test)]
    fn deliver_drop(&self, output: &AVCaptureOutput, sample: &CMSampleBuffer) {
        unsafe {
            let _: () = msg_send![self, captureOutput: output, didDropSampleBuffer: sample, fromConnection: std::ptr::null::<AnyObject>()];
        }
    }
}

impl DelegateIvars {
    /// A timestamp on the session's clock as host time (sp_core::clock), in seconds.
    fn host_seconds(&self, t: CMTime) -> Option<f64> {
        let t = match self.clock.get() {
            Some(c) => unsafe {
                let host = CMClock::host_time_clock();
                let from: &CFType = &c.0;
                let to: &CFType = &host;
                CMSyncConvertTime(t, from, to)
            },
            None => t,
        };
        let s = unsafe { t.seconds() };
        s.is_finite().then_some(s)
    }

    fn picture(&self, sample: &CMSampleBuffer, t_arrival: f64) {
        let pipe = &self.pipe;
        let t_src = self.host_seconds(unsafe { sample.presentation_time_stamp() });
        let img = unsafe { sample.image_buffer() };
        let (bt709, full) = (pipe.bt709, pipe.full_range);
        let r = pipe.frame(t_arrival, t_src, |conv| match &img {
            Some(img) => convert_pixels(conv, img, bt709, full),
            None => Err("a sample with no picture in it".into()),
        });
        if let Err(e) = r {
            pipe.stop_with(&e);
        }
    }

    fn dropped(&self, sample: &CMSampleBuffer) {
        let reason = unsafe {
            CMGetAttachment(sample.as_ref(), kCMSampleBufferAttachmentKey_DroppedFrameReason, std::ptr::null_mut())
        };
        let reason: Option<String> =
            reason.and_then(|r| r.downcast::<CFString>().ok()).map(|s| s.to_string());
        let discontinuity = unsafe { kCMSampleBufferDroppedFrameReason_Discontinuity.to_string() };
        if reason.as_deref() == Some(discontinuity.as_str()) {
            if let Err(e) = self.pipe.os_damaged("macOS reported a break in the card's frames") {
                self.pipe.stop_with(&e);
            }
        } else {
            self.pipe.os_dropped();
        }
    }

    fn sound(&self, sample: &CMSampleBuffer, t_arrival: f64) {
        let Some((pcm, rate, ch)) = pcm_of(sample) else { return };
        let frames = pcm.len() / ch as usize;
        let t_end = self
            .host_seconds(unsafe { sample.presentation_time_stamp() })
            .map(|t| t + frames as f64 / rate.max(1) as f64);
        self.pipe.sound_live(pcm, rate, ch, t_end, t_arrival);
    }
}

/// The running session, for the source to stop and ask after.
pub(crate) struct Live {
    session: Retained<AVCaptureSession>,
    device: Retained<AVCaptureDevice>,
    _outputs: Vec<Retained<AVCaptureOutput>>,
    _delegates: Vec<Retained<OutputDelegate>>,
    _queues: Vec<DispatchRetained<DispatchQueue>>,
}

// AVCaptureSession and AVCaptureDevice may be used from any thread (startRunning and stopRunning block, and
// Apple has them called off the main thread).
unsafe impl Send for Live {}

impl Live {
    pub fn connected(&self) -> bool {
        unsafe { self.device.isConnected() }
    }
    pub fn running(&self) -> bool {
        unsafe { self.session.isRunning() || self.session.isInterrupted() }
    }
    pub fn in_use_elsewhere(&self) -> bool {
        unsafe { self.device.isInUseByAnotherApplication() }
    }
    pub fn stop(&self) {
        unsafe { self.session.stopRunning() };
    }
}

/// Why a card wouldn't open, in plain words (capture.py open_failure, for the Mac).
pub fn open_failure(device: &str, error: &str, in_use: bool) -> String {
    let low = error.to_lowercase();
    if in_use || low.contains("in use") || low.contains("busy") {
        return format!(
            "{device} is in use by another app. Close OBS, Elgato's apps, a browser tab using the camera, or anything \
             else that shows the card, then press Play again: many cards work with one app at a time"
        );
    }
    if low.contains("not authorized") || low.contains("permission") || low.contains("denied") {
        return format!(
            "macOS wouldn't let SmoovPlay open {device}. Turn SmoovPlay on in System Settings > Privacy & Security > \
             Camera (and Microphone, for its sound), then press Play again"
        );
    }
    if low.contains("disconnected") || low.contains("no longer") || low.contains("not found") {
        return format!("macOS no longer lists {device}: reconnect the card, then press Look again on the Play page");
    }
    if low.contains("format") {
        return format!("{device} refused the video format SmoovPlay asked for");
    }
    String::new()
}

/// Waits for macOS's answer on using cameras or microphones (asking the person the first time); Err in plain words
/// when it's no.
fn authorize(media: &AVMediaType, what: &str, device: &str) -> Result<(), String> {
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(media) };
    if status == AVAuthorizationStatus::Authorized {
        return Ok(());
    }
    let denied = || {
        format!(
            "macOS wouldn't let SmoovPlay use {device}. Turn SmoovPlay on in System Settings > Privacy & Security > \
             {what}, then press Play again"
        )
    };
    if status != AVAuthorizationStatus::NotDetermined {
        return Err(denied());
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<bool>(1);
    let block = block2::RcBlock::new(move |granted: Bool| {
        let _ = tx.try_send(granted.as_bool());
    });
    unsafe { AVCaptureDevice::requestAccessForMediaType_completionHandler(media, &block) };
    match rx.recv_timeout(std::time::Duration::from_secs(300)) {
        Ok(true) => Ok(()),
        _ => Err(denied()),
    }
}

/// The video output's settings: just the pixel format (the only key AVFoundation promises to take everywhere; its
/// buffers are IOSurface-backed anyway, and they're copied into the converter's buffer).
fn video_settings(code: u32) -> Retained<NSDictionary<NSString, AnyObject>> {
    let pf = cf::num(code as i64);
    let d = unsafe { cf::dict(&[(kCVPixelBufferPixelFormatTypeKey, &pf)]) };
    // (CFDictionary and NSDictionary are toll-free bridged)
    unsafe { Retained::retain(CFRetained::as_ptr(&d).as_ptr() as *mut NSDictionary<NSString, AnyObject>).unwrap() }
}

/// What to ask the video output for: the card's own NV12 as it is, anything else as NV12 converted by macOS, in
/// the range the mode says (limited unless it says full).
pub fn output_format(mode: &Mode, code: u32) -> u32 {
    if code == NV12_VIDEO || code == NV12_FULL {
        code
    } else if mode.range == "pc" {
        NV12_FULL
    } else {
        NV12_VIDEO
    }
}

/// The rate to ask a format's frame rate range for: open_rate's, within what the range allows (a 120-only mode
/// picked for 60 runs at 120).
fn range_rate(dm: &DeviceMode, rate: f64) -> f64 {
    let (lo, hi) = unsafe { (dm.rates.minFrameRate(), dm.rates.maxFrameRate()) };
    if rate > hi - 0.01 { hi } else if rate < lo + 0.01 { lo } else { rate }
}

/// The frame duration for that rate: the range's own at either end (exact: 1001/60000 for 59.94), else 1/rate.
fn frame_duration(dm: &DeviceMode, rate: f64) -> CMTime {
    unsafe {
        if rate >= dm.rates.maxFrameRate() {
            dm.rates.minFrameDuration()
        } else if rate <= dm.rates.minFrameRate() {
            dm.rates.maxFrameDuration()
        } else {
            CMTime::new(1000, (rate * 1000.0).round() as i32)
        }
    }
}

fn queue(label: &str) -> DispatchRetained<DispatchQueue> {
    DispatchQueue::new(label, None)
}

impl CardSource {
    /// A capture card by unique ID, name or number (capture.py CardSource.open). `mode`: the one to use (None: the
    /// best for `w` x `h` at `fps`, pick_mode). `audio_device`: the card's sound device (pick_audio), its own sound
    /// when it names the card itself; opened with the video, played by `opts.audio`.
    pub fn open(
        device: &str,
        w: u32,
        h: u32,
        fps: f64,
        mode: Option<Mode>,
        audio_device: Option<&str>,
        opts: CardOptions,
    ) -> Result<CardSource, String> {
        let dev = devices::find_video(device).ok_or_else(|| {
            format!("macOS no longer lists {device}: reconnect the card, then press Look again on the Play page")
        })?;
        let name = unsafe { dev.localizedName().to_string() };
        let uid = unsafe { dev.uniqueID().to_string() };
        let modes = device_modes(&dev);
        let all: Vec<Mode> = modes.iter().map(|m| m.mode.clone()).collect();
        let (mode, note) = match mode {
            Some(m) => (m, String::new()),
            None => {
                let (m, note) = pick_mode(&all, w, h, fps);
                (m.ok_or_else(|| format!("capture card {name:?}: {note}"))?, note)
            }
        };
        if !note.is_empty() {
            println!("[capture] {note}");
        }
        let dm = modes.iter().find(|m| m.mode == mode).cloned().ok_or_else(|| format!("{name} refused the video format SmoovPlay asked for"))?;
        let rate = range_rate(&dm, open_rate(&mode, fps));
        let code = output_format(&mode, dm.code);
        let (bt709, full_range) = opts.colour.resolve(mode.h as usize, if code == NV12_FULL { "pc" } else { &mode.range });

        let video_media = unsafe { AVMediaTypeVideo.unwrap() };
        authorize(video_media, "Camera", &name)?;
        // the sound: the card's own port, another device, or none
        let wants_sound = opts.audio.is_some() && audio_device.is_some();
        let own_sound = audio_device.is_some_and(|a| a == name || a == uid);
        let mut audio_dev: Option<Retained<AVCaptureDevice>> = None;
        let mut sound_note: Option<String> = None;
        if wants_sound {
            if own_sound {
                if !has_sound(&dev) {
                    sound_note = Some(
                        "[capture] no sound from the card itself (macOS lists no sound on it): playing without sound; \
                         pick a sound device in the launcher, or with --capture-audio (see --list-devices)"
                            .into(),
                    );
                }
            } else {
                match devices::find_audio(audio_device.unwrap()) {
                    Some(a) => audio_dev = Some(a),
                    None => {
                        sound_note = Some(format!(
                            "[capture] couldn't open the sound device ({} isn't listed); playing without sound",
                            audio_device.unwrap()
                        ))
                    }
                }
            }
        }
        let sound_on = wants_sound && sound_note.is_none();
        if sound_on
            && let Err(e) = authorize(unsafe { AVMediaTypeAudio.unwrap() }, "Microphone", &name)
        {
            sound_note = Some(format!("[capture] couldn't open the sound device ({}); playing without sound", clip(&e, 120)));
        }
        let sound_on = sound_on && sound_note.is_none();

        let pipe = Arc::new(
            Pipe::new(PipeConfig {
                w: mode.w as usize,
                h: mode.h as usize,
                fps: rate,
                bt709,
                full_range,
                fmt: mode.fmt.clone(),
                live: true,
                queue_size: opts.queue_size,
                test: opts.test,
                give_up_s: opts.damaged_give_up_s,
                sound_out: if sound_on { opts.audio } else { None },
                has_sound: sound_on,
            })
            .map_err(|e| e.to_string())?,
        );
        if let Some(n) = sound_note {
            pipe.say(n);
        }
        let live = start_session(&pipe, &dev, &dm, rate, code, if sound_on { Some(audio_dev.as_deref().unwrap_or(&dev)) } else { None }, &name)?;
        let src = CardSource::new(pipe, &format!("Capture card ({name})"), mode.raw, Some(live), None);
        if !mode.raw {
            src.pipe.say(format!(
                "[capture] {} is compressed: macOS decodes each frame first, which can stutter. If the card offers NV12 \
                 or YUY2 at this size, that's better.",
                mode.fmt.to_uppercase()
            ));
        }
        if src.pipe.can_delay_audio() {
            src.pipe.say("[capture] sound: the card's, played on this Mac, held back to meet the picture".into());
        }
        Ok(src)
    }
}

fn start_session(
    pipe: &Arc<Pipe>,
    dev: &AVCaptureDevice,
    dm: &DeviceMode,
    rate: f64,
    code: u32,
    sound: Option<&AVCaptureDevice>,
    name: &str,
) -> Result<Live, String> {
    unsafe {
        let in_use = dev.isInUseByAnotherApplication();
        let fail = |e: String| {
            let why = open_failure(name, &e, in_use);
            if why.is_empty() { format!("{name}: {e}") } else { why }
        };
        let session = AVCaptureSession::new();
        session.beginConfiguration();
        let input = AVCaptureDeviceInput::deviceInputWithDevice_error(dev).map_err(|e| fail(e.localizedDescription().to_string()))?;
        let input_ref: &AVCaptureInput = &input;
        if !session.canAddInput(input_ref) {
            session.commitConfiguration();
            return Err(fail("the capture session can't take it".into()));
        }
        session.addInput(input_ref);
        // the format, once the input is in the session (adding it sets the session's preset), held across
        // startRunning so the session keeps it
        dev.lockForConfiguration().map_err(|e| fail(format!("format: {}", e.localizedDescription())))?;
        dev.setActiveFormat(&dm.format);
        let d = frame_duration(dm, rate);
        dev.setActiveVideoMinFrameDuration(d);
        dev.setActiveVideoMaxFrameDuration(d);

        let clock: Arc<OnceLock<SyncClock>> = Arc::new(OnceLock::new());
        let mut outputs: Vec<Retained<AVCaptureOutput>> = Vec::new();
        let mut delegates = Vec::new();
        let mut queues = Vec::new();

        let vout = AVCaptureVideoDataOutput::new();
        vout.setVideoSettings(Some(&video_settings(code)));
        vout.setAlwaysDiscardsLateVideoFrames(true);
        let vdel = OutputDelegate::new(pipe.clone(), false, clock.clone());
        let vq = queue("smoovplay.capture.video");
        vout.setSampleBufferDelegate_queue(Some(ProtocolObject::from_ref(&*vdel)), Some(&vq));
        let vout_ref: &AVCaptureOutput = &vout;
        if !session.canAddOutput(vout_ref) {
            dev.unlockForConfiguration();
            session.commitConfiguration();
            return Err(fail("the capture session can't hand its frames over".into()));
        }
        session.addOutput(vout_ref);
        outputs.push(Retained::into_super(vout));
        delegates.push(vdel);
        queues.push(vq);

        let mut sound_added = false;
        if let Some(sdev) = sound {
            let same = sdev.uniqueID().to_string() == dev.uniqueID().to_string();
            let mut sound_input: Option<Retained<AVCaptureDeviceInput>> = same.then(|| input.clone());
            let added = if same {
                true
            } else {
                // the card's own audio port, if it has one, is turned off: the sound comes from the device given
                for p in input.ports().iter() {
                    if AVMediaTypeAudio.is_some_and(|a| p.mediaType().to_string() == a.to_string()) {
                        p.setEnabled(false);
                    }
                }
                match AVCaptureDeviceInput::deviceInputWithDevice_error(sdev) {
                    Ok(ain) => {
                        let ain_ref: &AVCaptureInput = &ain;
                        if session.canAddInput(ain_ref) {
                            session.addInput(ain_ref);
                            sound_input = Some(ain.clone());
                            true
                        } else {
                            false
                        }
                    }
                    Err(e) => {
                        pipe.say(format!(
                            "[capture] couldn't open the sound device ({}); playing without sound",
                            clip(&e.localizedDescription().to_string(), 120)
                        ));
                        false
                    }
                }
            };
            if added {
                let ch = sound_input.as_deref().and_then(audio_channels).map(|c| c.clamp(1, 2)).unwrap_or(2);
                let aout = AVCaptureAudioDataOutput::new();
                aout.setAudioSettings(Some(&pcm_settings(ch, 0.0)));
                let adel = OutputDelegate::new(pipe.clone(), true, clock.clone());
                let aq = queue("smoovplay.capture.sound");
                aout.setSampleBufferDelegate_queue(Some(ProtocolObject::from_ref(&*adel)), Some(&aq));
                let aout_ref: &AVCaptureOutput = &aout;
                if session.canAddOutput(aout_ref) {
                    session.addOutput(aout_ref);
                    sound_added = true;
                    outputs.push(Retained::into_super(aout));
                    delegates.push(adel);
                    queues.push(aq);
                } else {
                    pipe.say("[capture] couldn't open the sound device (the session can't take its sound); playing without sound".into());
                }
            }
        }
        if sound.is_some() && !sound_added {
            pipe.sound_off();
        }
        session.commitConfiguration();
        session.startRunning();
        dev.unlockForConfiguration();
        if let Some(c) = session.synchronizationClock() {
            let _ = clock.set(SyncClock(c));
        }
        if !session.isRunning() {
            return Err(fail("the capture session didn't start".into()));
        }
        let active = dev.activeFormat();
        let kind = |f: &objc2_av_foundation::AVCaptureDeviceFormat| {
            let d = f.formatDescription();
            (d.media_sub_type(), objc2_core_media::CMVideoFormatDescriptionGetDimensions(&d))
        };
        let (a, b) = (kind(&active), kind(&dm.format));
        if (a.0, a.1.width, a.1.height) != (b.0, b.1.width, b.1.height) {
            pipe.say(format!(
                "[capture] macOS didn't keep the format asked for ({} at {} fps)",
                dm.mode.label(),
                g(rate)
            ));
        }
        Ok(Live { session, device: dev.retain(), _outputs: outputs, _delegates: delegates, _queues: queues })
    }
}

/// The channels of an input's sound port, as its format says.
fn audio_channels(input: &AVCaptureDeviceInput) -> Option<u32> {
    unsafe {
        let audio = AVMediaTypeAudio?;
        input
            .ports()
            .iter()
            .filter(|p| p.mediaType().to_string() == audio.to_string())
            .find_map(|p| p.formatDescription().and_then(|d| asbd(&d)))
            .map(|a| a.mChannelsPerFrame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sp_core::Tick;
    use sp_media::queue::Got;

    /// A file's sample buffers handed to the outputs' delegates the way AVFoundation hands a card's (through the
    /// Objective-C method), as a live card's: numbering (a dropped frame included), the session clock's timestamps,
    /// the conversion, and the sound with how late it comes.
    #[test]
    fn a_files_samples_go_through_the_delegates_as_a_cards() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/capture/onset.mov");
        if !path.exists() {
            eprintln!("(skipped: run mac/core/tools/capture_fixtures.py)");
            return;
        }
        let (readers, meta) = crate::file::open_readers(&path, true).unwrap();
        let pipe = Arc::new(
            Pipe::new(PipeConfig {
                w: meta.w,
                h: meta.h,
                fps: 60.0,
                bt709: false,
                full_range: false,
                fmt: "nv12".into(),
                live: true,
                queue_size: 500,
                test: None,
                give_up_s: 5.0,
                sound_out: None,
                has_sound: false,
            })
            .unwrap(),
        );
        let clock: Arc<OnceLock<SyncClock>> = Arc::new(OnceLock::new());
        let (vdel, adel) = (OutputDelegate::new(pipe.clone(), false, clock.clone()), OutputDelegate::new(pipe.clone(), true, clock.clone()));
        let (vout, aout) = unsafe { (AVCaptureVideoDataOutput::new(), AVCaptureAudioDataOutput::new()) };
        let (_, vo) = &readers.video;
        let (_, ao) = readers.sound.as_ref().unwrap();
        let mut n = 0;
        while let Some(sb) = unsafe { vo.copyNextSampleBuffer() } {
            if n == 10 {
                vdel.deliver_drop(&vout, &sb); // macOS dropped this one: numbered all the same
            } else {
                vdel.deliver(&vout, &sb);
            }
            if n == 5 {
                // from here on, timestamps go through the session's clock (host time here: the same seconds)
                let host = unsafe { CMClock::host_time_clock() };
                let _ = clock.set(SyncClock(unsafe { Retained::retain(CFRetained::as_ptr(&host).as_ptr()).unwrap() }));
            }
            n += 1;
        }
        while let Some(sb) = unsafe { ao.copyNextSampleBuffer() } {
            adel.deliver(&aout, &sb);
        }
        pipe.end();
        let mut ticks: Vec<Tick> = Vec::new();
        while let Got::Tick(t) = pipe.q.get(0.0) {
            ticks.push(t);
        }
        assert_eq!(n, 120);
        assert_eq!(ticks.len(), 119);
        let seqs: Vec<u64> = ticks.iter().map(|t| t.extra["seq"].as_u64().unwrap()).collect();
        assert_eq!(seqs, (0..120).filter(|&i| i != 10).collect::<Vec<u64>>());
        assert_eq!(pipe.os_dropped.load(std::sync::atomic::Ordering::Relaxed), 1);
        for t in &ticks {
            let s = t.extra["seq"].as_u64().unwrap() as f64;
            assert!((t.t_source.unwrap() - s / 60.0).abs() < 1e-6, "{:?}", t.t_source);
        }
        let bright = |t: &Tick| t.frame.to_bgr().iter().map(|&v| v as f64).sum::<f64>() / (meta.w * meta.h * 3) as f64 > 120.0;
        assert_eq!(ticks.iter().position(bright), Some(59)); // frame 60, one tick earlier for the dropped one
        // the sound, kept for instant replay, and how late it came against the picture
        let sound = pipe.read_sound();
        let total: usize = sound.iter().map(|s| s.pcm.len()).sum();
        assert_eq!(total, 2 * 48000 * 2);
        assert!(pipe.sound_late_s().is_finite());
        assert!(pipe.messages().is_empty(), "{:?}", pipe.messages());
    }
}
