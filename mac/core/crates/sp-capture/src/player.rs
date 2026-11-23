//! Plays the card's sound on the Mac's output (framegen/audio.py Player). It uses Core Audio's default output unit,
//! which follows the output picked in System Settings and converts the card's rate to the device's, as WASAPI's
//! auto_convert does on Windows. The unit pulls from a [`PlayerCore`] on its real-time thread.
//!
//! The output's own latency (the device's, its safety offset, its buffer, its stream's, and the unit's rate
//! converter) is taken off the delay asked for, so the total is what the pipeline asked for. Core Audio's overload
//! notice counts as the output running short, as do calls that come later than the latency lasts.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_audio_toolbox::{
    kAudioOutputUnitProperty_CurrentDevice, kAudioUnitManufacturer_Apple, kAudioUnitProperty_Latency,
    kAudioUnitProperty_SetRenderCallback, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Global,
    kAudioUnitScope_Input, kAudioUnitSubType_DefaultOutput, kAudioUnitType_Output, AURenderCallbackStruct,
    AudioComponentDescription, AudioComponentFindNext, AudioComponentInstanceDispose, AudioComponentInstanceNew,
    AudioOutputUnitStart, AudioOutputUnitStop, AudioUnit, AudioUnitGetProperty, AudioUnitInitialize,
    AudioUnitRenderActionFlags, AudioUnitSetProperty, AudioUnitUninitialize,
};
use objc2_core_audio::{
    kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyLatency, kAudioDevicePropertyNominalSampleRate,
    kAudioDevicePropertySafetyOffset, kAudioDevicePropertyStreams, kAudioDeviceProcessorOverload,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeOutput, AudioObjectAddPropertyListener, AudioObjectGetPropertyData,
    AudioObjectGetPropertyDataSize, AudioObjectID, AudioObjectPropertyAddress, AudioObjectRemovePropertyListener,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM, AudioBufferList,
    AudioStreamBasicDescription, AudioTimeStamp,
};
use objc2_core_foundation::{CFRetained, CFString};

use crate::audio::{PlayerCore, SoundStats};
use crate::pipe::{SoundFactory, SoundOut};

/// The card's sound on the Mac's default output, held back to meet the picture.
pub struct Player {
    /// What the callbacks see (the core and the channel count), boxed so its address stays put as the Player moves.
    cb: Box<(Arc<PlayerCore>, usize)>,
    unit: AudioUnit,
    device: AudioObjectID,
    listening: bool,
    closed: bool,
    /// The output's name, for the log.
    pub device_name: String,
}

// The unit is only touched from the thread that owns the Player (its callback has its own Arc).
unsafe impl Send for Player {}

fn check(st: i32, what: &str) -> Result<(), String> {
    if st == 0 { Ok(()) } else { Err(format!("{what} failed ({st})")) }
}

fn address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress { mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain }
}

fn get_u32(obj: AudioObjectID, selector: u32, scope: u32) -> Option<u32> {
    let mut v: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as u32;
    let a = address(selector, scope);
    let st = unsafe {
        AudioObjectGetPropertyData(obj, NonNull::from(&a), 0, std::ptr::null(), NonNull::from(&mut size), NonNull::from(&mut v).cast())
    };
    (st == 0).then_some(v)
}

fn get_f64(obj: AudioObjectID, selector: u32, scope: u32) -> Option<f64> {
    let mut v: f64 = 0.0;
    let mut size = std::mem::size_of::<f64>() as u32;
    let a = address(selector, scope);
    let st = unsafe {
        AudioObjectGetPropertyData(obj, NonNull::from(&a), 0, std::ptr::null(), NonNull::from(&mut size), NonNull::from(&mut v).cast())
    };
    (st == 0).then_some(v)
}

fn device_name(obj: AudioObjectID) -> String {
    let mut s: *const CFString = std::ptr::null();
    let mut size = std::mem::size_of::<*const CFString>() as u32;
    let a = address(kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal);
    let st = unsafe {
        AudioObjectGetPropertyData(obj, NonNull::from(&a), 0, std::ptr::null(), NonNull::from(&mut size), NonNull::from(&mut s).cast())
    };
    match NonNull::new(s as *mut CFString) {
        Some(p) if st == 0 => unsafe { CFRetained::from_raw(p) }.to_string(),
        _ => String::new(),
    }
}

/// The output's latency in seconds. That's the device's own, its safety offset, its IO buffer and its first output
/// stream's, at the device's rate, which is what PortAudio reports as an output stream's latency. The unit's is
/// added on top.
fn output_latency(unit: AudioUnit, device: AudioObjectID) -> f64 {
    let out = kAudioObjectPropertyScopeOutput;
    let rate = get_f64(device, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal).filter(|r| *r > 0.0);
    let Some(rate) = rate else { return 0.0 };
    let mut frames = 0u64;
    for sel in [kAudioDevicePropertyLatency, kAudioDevicePropertySafetyOffset] {
        frames += get_u32(device, sel, out).unwrap_or(0) as u64;
    }
    frames += get_u32(device, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal).unwrap_or(0) as u64;
    // the first output stream's latency
    let a = address(kAudioDevicePropertyStreams, out);
    let mut size = 0u32;
    if unsafe { AudioObjectGetPropertyDataSize(device, NonNull::from(&a), 0, std::ptr::null(), NonNull::from(&mut size)) } == 0
        && size >= 4
    {
        let mut ids = vec![0u32; size as usize / 4];
        let ok = unsafe {
            AudioObjectGetPropertyData(device, NonNull::from(&a), 0, std::ptr::null(), NonNull::from(&mut size), NonNull::new(ids.as_mut_ptr().cast()).unwrap())
        } == 0;
        if let (true, Some(&s)) = (ok, ids.first()) {
            frames += get_u32(s, kAudioDevicePropertyLatency, kAudioObjectPropertyScopeGlobal).unwrap_or(0) as u64;
        }
    }
    let mut unit_latency = 0.0f64;
    let mut size = std::mem::size_of::<f64>() as u32;
    unsafe {
        AudioUnitGetProperty(
            unit,
            kAudioUnitProperty_Latency,
            kAudioUnitScope_Global,
            0,
            NonNull::from(&mut unit_latency).cast(),
            NonNull::from(&mut size),
        );
    }
    frames as f64 / rate + unit_latency.max(0.0)
}

unsafe extern "C-unwind" fn render(
    refcon: NonNull<c_void>,
    _flags: NonNull<AudioUnitRenderActionFlags>,
    _ts: NonNull<AudioTimeStamp>,
    _bus: u32,
    frames: u32,
    data: *mut AudioBufferList,
) -> i32 {
    let core = unsafe { &*(refcon.as_ptr() as *const (Arc<PlayerCore>, usize)) };
    let Some(list) = (unsafe { data.as_mut() }) else { return 0 };
    if list.mNumberBuffers == 0 {
        return 0;
    }
    let buf = &mut list.mBuffers[0];
    let n = (frames as usize * core.1).min(buf.mDataByteSize as usize / 2);
    if buf.mData.is_null() {
        return 0;
    }
    let out = unsafe { std::slice::from_raw_parts_mut(buf.mData as *mut i16, n) };
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| core.0.render(out, false))).is_err() {
        out.fill(0);
    }
    0
}

unsafe extern "C-unwind" fn overload(
    _obj: AudioObjectID,
    _n: u32,
    _addresses: NonNull<AudioObjectPropertyAddress>,
    refcon: *mut c_void,
) -> i32 {
    if let Some(core) = unsafe { (refcon as *const (Arc<PlayerCore>, usize)).as_ref() } {
        core.0.ran_short();
    }
    0
}

impl Player {
    /// Opens the default output for 16-bit PCM at the card's rate and channels, and starts it (silent until sound
    /// comes and the delay has filled).
    pub fn open(rate: u32, channels: u32) -> Result<Player, String> {
        let ch = channels.clamp(1, 2) as usize;
        let desc = AudioComponentDescription {
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_DefaultOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        let comp = unsafe { AudioComponentFindNext(std::ptr::null_mut(), NonNull::from(&desc)) };
        if comp.is_null() {
            return Err("this Mac has no default output unit".into());
        }
        let mut unit: AudioUnit = std::ptr::null_mut();
        check(unsafe { AudioComponentInstanceNew(comp, NonNull::from(&mut unit)) }, "opening the output")?;
        let mut p = Player {
            cb: Box::new((Arc::new(PlayerCore::new(rate, ch, 1.0)), ch)),
            unit,
            device: 0,
            listening: false,
            closed: false,
            device_name: String::new(),
        };
        p.start(rate)?;
        Ok(p)
    }

    fn refcon(&self) -> *mut c_void {
        // the boxed pair lives as long as the Player, at an address that doesn't move
        &*self.cb as *const (Arc<PlayerCore>, usize) as *mut c_void
    }

    fn start(&mut self, rate: u32) -> Result<(), String> {
        let ch = self.cb.1 as u32;
        let asbd = AudioStreamBasicDescription {
            mSampleRate: rate as f64,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2 * ch,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2 * ch,
            mChannelsPerFrame: ch,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        unsafe {
            check(
                AudioUnitSetProperty(
                    self.unit,
                    kAudioUnitProperty_StreamFormat,
                    kAudioUnitScope_Input,
                    0,
                    (&asbd as *const AudioStreamBasicDescription).cast(),
                    std::mem::size_of::<AudioStreamBasicDescription>() as u32,
                ),
                "setting the output's format",
            )?;
            let cb = AURenderCallbackStruct { inputProc: Some(render), inputProcRefCon: self.refcon() };
            check(
                AudioUnitSetProperty(
                    self.unit,
                    kAudioUnitProperty_SetRenderCallback,
                    kAudioUnitScope_Input,
                    0,
                    (&cb as *const AURenderCallbackStruct).cast(),
                    std::mem::size_of::<AURenderCallbackStruct>() as u32,
                ),
                "setting the output's callback",
            )?;
            check(AudioUnitInitialize(self.unit), "starting the output")?;
            let mut dev: AudioObjectID = 0;
            let mut size = std::mem::size_of::<AudioObjectID>() as u32;
            AudioUnitGetProperty(
                self.unit,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global,
                0,
                NonNull::from(&mut dev).cast(),
                NonNull::from(&mut size),
            );
            self.device = dev;
            self.device_name = device_name(dev);
            self.cb.0.set_latency(output_latency(self.unit, dev));
            if dev != 0 {
                let a = address(kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal);
                self.listening = AudioObjectAddPropertyListener(dev, NonNull::from(&a), Some(overload), self.refcon()) == 0;
            }
            check(AudioOutputUnitStart(self.unit), "playing")?;
        }
        Ok(())
    }

    /// The output's own latency in seconds, which is taken off the delay.
    pub fn latency(&self) -> f64 {
        self.cb.0.latency()
    }

    /// How many times the output has asked for sound.
    pub fn calls(&self) -> u64 {
        self.cb.0.calls()
    }

    pub fn push(&self, pcm: &[i16]) {
        self.cb.0.push(pcm)
    }

    pub fn set_delay(&self, seconds: f64) {
        self.cb.0.set_delay(seconds)
    }

    pub fn stats(&self) -> SoundStats {
        self.cb.0.stats()
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        unsafe {
            AudioOutputUnitStop(self.unit);
            if self.listening {
                let a = address(kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal);
                AudioObjectRemovePropertyListener(self.device, NonNull::from(&a), Some(overload), self.refcon());
            }
            AudioUnitUninitialize(self.unit);
            AudioComponentInstanceDispose(self.unit);
        }
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.close();
    }
}

impl SoundOut for Player {
    fn push(&mut self, pcm: &[i16]) {
        Player::push(self, pcm)
    }
    fn set_delay(&mut self, seconds: f64) {
        Player::set_delay(self, seconds)
    }
    fn stats(&self) -> Option<SoundStats> {
        Some(Player::stats(self))
    }
    fn close(&mut self) {
        Player::close(self)
    }
}

/// What plays a capture card's sound, a [`Player`] on the Mac's default output (run.py sound_player).
pub fn player_factory() -> SoundFactory {
    Box::new(|rate, channels| {
        let p = Player::open(rate, channels)?;
        println!(
            "[capture] sound: {rate} Hz, {channels} channel(s), played on {} ({:.0} ms of output latency), held back \
             to meet the picture",
            if p.device_name.is_empty() { "this Mac".to_string() } else { p.device_name.clone() },
            p.latency() * 1000.0
        );
        Ok(Box::new(p) as Box<dyn SoundOut>)
    })
}
