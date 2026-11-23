//! Sound from AVFoundation's sample buffers as interleaved 16-bit PCM, at most two channels. capture.py `_sound`
//! does the same with PyAV's resampler to s16, stereo or mono, at the source's rate. The outputs are asked for
//! 16-bit interleaved PCM already. Other linear PCM layouts (float, 32-bit, planar) are converted here, and more
//! than two channels keep the first two.

use std::ptr::NonNull;

use objc2_core_audio_types::{
    kAudioFormatFlagIsBigEndian, kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved, kAudioFormatLinearPCM,
    AudioStreamBasicDescription,
};
use objc2_core_media::{CMAudioFormatDescriptionGetStreamBasicDescription, CMFormatDescription, CMSampleBuffer};

/// The stream description of an audio format description.
pub(crate) fn asbd(desc: &CMFormatDescription) -> Option<AudioStreamBasicDescription> {
    let p = unsafe { CMAudioFormatDescriptionGetStreamBasicDescription(desc) };
    if p.is_null() { None } else { Some(unsafe { *p }) }
}

/// (pcm, rate, channels) of a sample buffer of linear PCM. None for anything else, or for an empty buffer.
pub(crate) fn pcm_of(sb: &CMSampleBuffer) -> Option<(Vec<i16>, u32, u32)> {
    unsafe {
        let desc = sb.format_description()?;
        let a = asbd(&desc)?;
        if a.mFormatID != kAudioFormatLinearPCM || a.mChannelsPerFrame == 0 {
            return None;
        }
        let frames = sb.num_samples().max(0) as usize;
        let block = sb.data_buffer()?;
        let len = block.data_length();
        if frames == 0 || len == 0 {
            return None;
        }
        let mut bytes = vec![0u8; len];
        if block.copy_data_bytes(0, len, NonNull::new(bytes.as_mut_ptr().cast()).unwrap()) != 0 {
            return None;
        }
        let ch_in = a.mChannelsPerFrame as usize;
        let planar = a.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0;
        let float = a.mFormatFlags & kAudioFormatFlagIsFloat != 0;
        let big = a.mFormatFlags & kAudioFormatFlagIsBigEndian != 0;
        let bits = a.mBitsPerChannel as usize;
        let bps = bits.div_ceil(8);
        if !matches!((float, bits), (false, 16) | (false, 32) | (true, 32) | (false, 24)) {
            return None;
        }
        let frames = frames.min(len / (bps * ch_in));
        let ch = ch_in.min(2);
        let sample = |i: usize| -> i16 {
            let b = &bytes[i * bps..i * bps + bps];
            match float {
                // integers keep their top 16 bits
                false if big => i16::from_be_bytes([b[0], b[1]]),
                false => i16::from_le_bytes([b[bps - 2], b[bps - 1]]),
                true => {
                    let v = [b[0], b[1], b[2], b[3]];
                    let f = if big { f32::from_be_bytes(v) } else { f32::from_le_bytes(v) };
                    (f * 32768.0).round().clamp(-32768.0, 32767.0) as i16
                }
            }
        };
        let mut out = Vec::with_capacity(frames * ch);
        for i in 0..frames {
            for c in 0..ch {
                out.push(sample(if planar { c * frames + i } else { i * ch_in + c }));
            }
        }
        Some((out, a.mSampleRate.round() as u32, ch as u32))
    }
}
