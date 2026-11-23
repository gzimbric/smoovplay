//! AAC-LC encoding (48 kHz stereo, 160 kbit/s: replay.py's sound track) with AudioToolbox's AudioConverter, fed a
//! sample at a time as the sound comes, with each packet's timestamp worked out from where its samples were put.
//!
//! The encoder holds back `priming` samples at the start (2112 for Apple's): packet k decodes to input samples from
//! k*1024 - priming, so that's the time it's given, and a clip cut from anywhere has its sound in place.

use std::collections::VecDeque;
use std::ffi::c_void;

use crate::{Error, Result, SOUND_BITRATE, SOUND_RATE};

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(non_snake_case)]
struct AudioStreamBasicDescription {
    mSampleRate: f64,
    mFormatID: u32,
    mFormatFlags: u32,
    mBytesPerPacket: u32,
    mFramesPerPacket: u32,
    mBytesPerFrame: u32,
    mChannelsPerFrame: u32,
    mBitsPerChannel: u32,
    mReserved: u32,
}

#[repr(C)]
#[allow(non_snake_case)]
struct AudioBuffer {
    mNumberChannels: u32,
    mDataByteSize: u32,
    mData: *mut c_void,
}

#[repr(C)]
#[allow(non_snake_case)]
struct AudioBufferList {
    mNumberBuffers: u32,
    mBuffers: [AudioBuffer; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(non_snake_case)]
struct AudioStreamPacketDescription {
    mStartOffset: i64,
    mVariableFramesInPacket: u32,
    mDataByteSize: u32,
}

#[repr(C)]
#[derive(Default)]
#[allow(non_snake_case)]
struct AudioConverterPrimeInfo {
    leadingFrames: u32,
    trailingFrames: u32,
}

type AudioConverterRef = *mut c_void;
type InputProc = unsafe extern "C" fn(
    AudioConverterRef,
    *mut u32,
    *mut AudioBufferList,
    *mut *mut AudioStreamPacketDescription,
    *mut c_void,
) -> i32;

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioConverterNew(
        src: *const AudioStreamBasicDescription,
        dst: *const AudioStreamBasicDescription,
        out: *mut AudioConverterRef,
    ) -> i32;
    fn AudioConverterDispose(c: AudioConverterRef) -> i32;
    fn AudioConverterSetProperty(c: AudioConverterRef, id: u32, size: u32, data: *const c_void) -> i32;
    fn AudioConverterGetProperty(c: AudioConverterRef, id: u32, size: *mut u32, data: *mut c_void) -> i32;
    fn AudioConverterFillComplexBuffer(
        c: AudioConverterRef,
        proc_: InputProc,
        user: *mut c_void,
        packets: *mut u32,
        out: *mut AudioBufferList,
        desc: *mut AudioStreamPacketDescription,
    ) -> i32;
}

const fn fourcc(s: &[u8; 4]) -> u32 {
    (s[0] as u32) << 24 | (s[1] as u32) << 16 | (s[2] as u32) << 8 | s[3] as u32
}

const LPCM: u32 = fourcc(b"lpcm");
const AAC: u32 = fourcc(b"aac ");
const SIGNED_PACKED: u32 = (1 << 2) | (1 << 3);
const BIT_RATE: u32 = fourcc(b"brat");
const PRIME_INFO: u32 = fourcc(b"prim");
const MAX_PACKET: u32 = fourcc(b"xops");
const QUALITY: u32 = fourcc(b"cdqu");
/// What the input callback says when it has nothing more for now (any non-zero status).
const NO_DATA: i32 = fourcc(b"nodt") as i32;

/// AAC-LC, 48 kHz, stereo.
pub const ASC: [u8; 2] = [0x11, 0x90];
pub const FRAME: i64 = 1024;

struct Input {
    fifo: VecDeque<i16>,
    handed: Vec<i16>,
}

unsafe extern "C" fn input_proc(
    _c: AudioConverterRef,
    packets: *mut u32,
    data: *mut AudioBufferList,
    _desc: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> i32 {
    let input = unsafe { &mut *(user as *mut Input) };
    let frames = (input.fifo.len() / 2).min(unsafe { *packets } as usize);
    if frames == 0 {
        unsafe { *packets = 0 };
        return NO_DATA;
    }
    input.handed.clear();
    input.handed.extend(input.fifo.drain(..frames * 2));
    unsafe {
        (*data).mNumberBuffers = 1;
        (*data).mBuffers[0] = AudioBuffer {
            mNumberChannels: 2,
            mDataByteSize: (frames * 4) as u32,
            mData: input.handed.as_mut_ptr() as *mut c_void,
        };
        *packets = frames as u32;
    }
    0
}

/// An AAC packet and when it's heard (48 kHz samples).
pub struct Packet {
    pub pts: i64,
    pub data: Vec<u8>,
}

pub struct AacEncoder {
    conv: AudioConverterRef,
    input: Box<Input>,
    pub priming: i64,
    max_packet: usize,
    /// Samples put in so far.
    fed: i64,
    /// Where the timeline jumps: (input sample index, its timestamp), oldest first.
    segments: VecDeque<(i64, i64)>,
    packets: i64,
}

// The converter is used from one thread at a time (the replay's).
unsafe impl Send for AacEncoder {}

impl AacEncoder {
    pub fn new() -> Result<AacEncoder> {
        let src = AudioStreamBasicDescription {
            mSampleRate: SOUND_RATE as f64,
            mFormatID: LPCM,
            mFormatFlags: SIGNED_PACKED,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 2,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        let dst = AudioStreamBasicDescription {
            mSampleRate: SOUND_RATE as f64,
            mFormatID: AAC,
            mFormatFlags: 0,
            mFramesPerPacket: FRAME as u32,
            mChannelsPerFrame: 2,
            ..Default::default()
        };
        let mut conv: AudioConverterRef = std::ptr::null_mut();
        let st = unsafe { AudioConverterNew(&src, &dst, &mut conv) };
        if st != 0 || conv.is_null() {
            return Err(Error::Status { what: "AudioConverterNew (AAC)", status: st });
        }
        let mut enc = AacEncoder {
            conv,
            input: Box::new(Input { fifo: VecDeque::new(), handed: Vec::new() }),
            priming: 2112,
            max_packet: 6144,
            fed: 0,
            segments: VecDeque::new(),
            packets: 0,
        };
        unsafe {
            let quality: u32 = 0x7f; // kAudioConverterQuality_Max
            AudioConverterSetProperty(conv, QUALITY, 4, &quality as *const u32 as *const c_void);
            let rate: u32 = SOUND_BITRATE;
            crate::status("AudioConverter bit rate", AudioConverterSetProperty(conv, BIT_RATE, 4, &rate as *const u32 as *const c_void))?;
            let mut prime = AudioConverterPrimeInfo::default();
            let mut size = std::mem::size_of::<AudioConverterPrimeInfo>() as u32;
            if AudioConverterGetProperty(conv, PRIME_INFO, &mut size, &mut prime as *mut _ as *mut c_void) == 0 {
                enc.priming = prime.leadingFrames as i64;
            }
            let mut max: u32 = 0;
            let mut size = 4u32;
            if AudioConverterGetProperty(conv, MAX_PACKET, &mut size, &mut max as *mut u32 as *mut c_void) == 0 && max > 0 {
                enc.max_packet = max as usize;
            }
        }
        Ok(enc)
    }

    /// The timestamp the next sample put in would have if the timeline carried on.
    fn next_pts(&self) -> Option<i64> {
        self.segments.back().map(|&(i, p)| p + (self.fed - i))
    }

    fn pts_of(&mut self, index: i64) -> i64 {
        while self.segments.len() > 1 && self.segments[1].0 <= index {
            self.segments.pop_front();
        }
        let (i, p) = self.segments[0];
        p + (index - i)
    }

    /// Interleaved stereo 16-bit samples at 48 kHz, the first of them at `pts`; returns the packets now done.
    pub fn write(&mut self, pcm: &[i16], pts: i64) -> Vec<Packet> {
        let frames = (pcm.len() / 2) as i64;
        if frames == 0 {
            return Vec::new();
        }
        if self.next_pts() != Some(pts) {
            self.segments.push_back((self.fed, pts));
        }
        self.input.fifo.extend(&pcm[..frames as usize * 2]);
        self.fed += frames;
        let mut out = Vec::new();
        let mut buf = vec![0u8; self.max_packet];
        loop {
            let mut n: u32 = 1;
            let mut desc = AudioStreamPacketDescription::default();
            let mut list = AudioBufferList {
                mNumberBuffers: 1,
                mBuffers: [AudioBuffer { mNumberChannels: 2, mDataByteSize: buf.len() as u32, mData: buf.as_mut_ptr() as *mut c_void }],
            };
            let st = unsafe {
                AudioConverterFillComplexBuffer(
                    self.conv,
                    input_proc,
                    &mut *self.input as *mut Input as *mut c_void,
                    &mut n,
                    &mut list,
                    &mut desc,
                )
            };
            if n == 0 {
                break;
            }
            let size = if desc.mDataByteSize > 0 { desc.mDataByteSize } else { list.mBuffers[0].mDataByteSize } as usize;
            let first = self.packets * FRAME - self.priming;
            let pts = self.pts_of(first.max(0)) + first.min(0);
            self.packets += 1;
            out.push(Packet { pts, data: buf[..size.min(buf.len())].to_vec() });
            if st != 0 && st != NO_DATA {
                break;
            }
        }
        out
    }
}

impl Drop for AacEncoder {
    fn drop(&mut self) {
        unsafe {
            AudioConverterDispose(self.conv);
        }
    }
}

/// Interleaved 16-bit samples, any rate and channel count, as stereo at 48 kHz (mono to both sides, surround's
/// front left and right; other rates linearly), keeping the resampler's place between chunks.
#[derive(Default)]
pub struct ToStereo48 {
    rate: u32,
    phase: f64,
    last: Option<(i16, i16)>,
}

impl ToStereo48 {
    pub fn convert(&mut self, pcm: &[i16], channels: u32, rate: u32) -> Vec<i16> {
        let ch = channels.max(1) as usize;
        let frames = pcm.len() / ch;
        let pair = |i: usize| -> (i16, i16) {
            let a = pcm[i * ch];
            let b = if ch >= 2 { pcm[i * ch + 1] } else { a };
            (a, b)
        };
        if rate == SOUND_RATE {
            let mut out = Vec::with_capacity(frames * 2);
            for i in 0..frames {
                let (a, b) = pair(i);
                out.push(a);
                out.push(b);
            }
            return out;
        }
        if rate != self.rate {
            self.rate = rate;
            self.phase = 0.0;
            self.last = None;
        }
        // Positions in the input (with the previous chunk's last frame at -1) for each output sample
        let step = rate as f64 / SOUND_RATE as f64;
        let mut out = Vec::new();
        let get = |k: isize, last: Option<(i16, i16)>| -> (i16, i16) {
            if k < 0 { last.unwrap_or_else(|| pair(0)) } else { pair(k as usize) }
        };
        let mut pos = self.phase;
        while frames > 0 && pos <= (frames - 1) as f64 {
            let k = pos.floor();
            let f = pos - k;
            let (a0, b0) = get(k as isize, self.last);
            let (a1, b1) = get(k as isize + 1, self.last);
            out.push((a0 as f64 + (a1 as f64 - a0 as f64) * f).round() as i16);
            out.push((b0 as f64 + (b1 as f64 - b0 as f64) * f).round() as i16);
            pos += step;
        }
        if frames > 0 {
            self.phase = pos - frames as f64;
            self.last = Some(pair(frames - 1));
        }
        out
    }
}
