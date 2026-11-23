//! The layout's numbers (chiaki_tap/frametap.h, docs/frame-tap.md, framegen/chiaki_tap.py).

/// chiaki-ng's tap (on a Mac, the shared memory "/chiaki-frametap").
pub const NAME: &str = "Local\\chiaki-frametap";
/// framegen/tap.py's default for other apps (on a Mac, "/smoovplay-tap").
pub const DEFAULT_APP_NAME: &str = "Local\\smoovplay-tap";

pub const MAGIC: u32 = 0x5054_4643; // 'CFTP'
/// Compressed frames, fixed slots (chiaki-ng).
pub const VERSION: u32 = 1;
/// Decoded frames, slots of the writer's own size.
pub const VERSION_RAW: u32 = 2;
pub const HEADER: usize = 4096;
pub const SLOT_HEADER: usize = 64;
pub const SLOTS: usize = 32;
pub const SLOT_SIZE: usize = 2 * 1024 * 1024;
pub const TOTAL: usize = HEADER + SLOTS * SLOT_SIZE;
/// The slots' share of the mapping, however a writer divides it.
pub const CAPACITY: usize = SLOTS * SLOT_SIZE;

// Header offsets
pub const OFF_MAGIC: usize = 0;
pub const OFF_VERSION: usize = 4;
pub const OFF_SLOT_COUNT: usize = 8;
pub const OFF_SLOT_SIZE: usize = 12;
pub const OFF_CODEC: usize = 16;
pub const OFF_WIDTH: usize = 20;
pub const OFF_HEIGHT: usize = 24;
pub const OFF_MAX_FPS: usize = 28;
pub const OFF_WRITE_SEQ: usize = 32;
pub const OFF_IDR_REQUEST: usize = 40;
pub const OFF_SESSION: usize = 44;
pub const OFF_QPC_FREQ: usize = 48;
pub const OFF_HEARTBEAT: usize = 56;
pub const OFF_FEATURES: usize = 64;
pub const OFF_PRESS_BUTTONS: usize = 68;
pub const OFF_PRESS_HOLD_MS: usize = 72;
pub const OFF_PRESS_COUNT: usize = 76;
pub const OFF_PRESS_QPC: usize = 80;
pub const OFF_AUDIO_DELAY_MS: usize = 88;
pub const OFF_COLOUR: usize = 92;

// Codecs (the header's `codec`): ChiakiCodec's, then decoded frames
pub const CODEC_H264: u32 = 0;
pub const CODEC_H265: u32 = 1;
pub const CODEC_H265_HDR: u32 = 2;
pub const CODEC_NV12: u32 = 16;
pub const CODEC_BGRA: u32 = 17;

/// What to call a codec in messages (framegen/chiaki_tap.py CODEC_NAMES).
pub fn codec_name(codec: u32) -> String {
    match codec {
        CODEC_H264 => "H.264".into(),
        CODEC_H265 => "H.265".into(),
        CODEC_H265_HDR => "H.265 HDR".into(),
        CODEC_NV12 => "NV12 (decoded)".into(),
        CODEC_BGRA => "BGRA (decoded)".into(),
        c => c.to_string(),
    }
}

/// A decoded format (version 2), as framegen/chiaki_tap.py RAW_CODECS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawFormat {
    Nv12,
    Bgra,
}

impl RawFormat {
    pub fn of_codec(codec: u32) -> Option<RawFormat> {
        match codec {
            CODEC_NV12 => Some(RawFormat::Nv12),
            CODEC_BGRA => Some(RawFormat::Bgra),
            _ => None,
        }
    }
    /// Bytes of one tightly packed frame.
    pub fn frame_bytes(self, width: usize, height: usize) -> usize {
        match self {
            RawFormat::Nv12 => width * height * 3 / 2,
            RawFormat::Bgra => width * height * 4,
        }
    }
}

/// NV12's colours from the header's `colour`: (bt709, full_range); None for 0 (auto: BT.709 from 720 lines up,
/// else BT.601, limited range).
pub fn colour_of(colour: u32) -> Option<(bool, bool)> {
    match colour {
        1 => Some((true, false)),
        2 => Some((true, true)),
        3 => Some((false, false)),
        4 => Some((false, true)),
        _ => None,
    }
}

/// Frames held (not interpolated) after a gap in the stream, while the picture repairs itself.
pub const DAMAGE_HOLD: u32 = 6;

// The header's `features`
pub const FEATURE_PRESS: u32 = 1;
pub const FEATURE_AUDIO_DELAY: u32 = 2;
pub const FEATURE_AUDIO_TAP: u32 = 4;
/// Frames come when the picture changes, not once a refresh.
pub const FEATURE_VARIABLE: u32 = 8;

// A slot's `flags`
pub const FLAG_TRUNCATED: u32 = 1;
pub const FLAG_REPEAT: u32 = 2;
pub const FLAG_NEW: u32 = 4;

// The sound ring (its own mapping: the name + AUDIO_SUFFIX)
pub const AUDIO_SUFFIX: &str = "-audio";
pub const AUDIO_MAGIC: u32 = 0x4154_4643; // 'CFTA'
pub const AUDIO_HEADER: usize = 64;
pub const AUDIO_SLOT_HEADER: usize = 32;
pub const AUDIO_SLOTS: usize = 512;
pub const AUDIO_SLOT_SIZE: usize = 8192;
pub const AUDIO_TOTAL: usize = AUDIO_HEADER + AUDIO_SLOTS * AUDIO_SLOT_SIZE;

// ChiakiControllerButton
pub const BUTTON_OPTIONS: u32 = 1 << 12;
pub const BUTTON_PS: u32 = 1 << 15;

/// POSIX shared memory names: "/" and the name, at most this long with the "/" (macOS's PSHMNAMLEN).
pub const SHM_NAME_MAX: usize = 31;
/// How often a waiting reader looks at the header, in seconds.
pub const POLL_S: f64 = 0.0005;

/// (slot_count, slot_size) for decoded frames of this format and size: slots a whole number of MiB that hold one
/// frame and its slot header, as many as fit in CAPACITY (at most 32). Err if fewer than 4 fit.
pub fn raw_layout(fmt: RawFormat, width: usize, height: usize) -> crate::Result<(usize, usize)> {
    let need = SLOT_HEADER + fmt.frame_bytes(width, height);
    let size = need.div_ceil(1 << 20) << 20;
    let count = SLOTS.min(CAPACITY / size);
    if count < 4 {
        let name = match fmt {
            RawFormat::Nv12 => "nv12",
            RawFormat::Bgra => "bgra",
        };
        return Err(crate::Error::Layout(format!(
            "a {width}x{height} {name} frame is too big for the frame tap ({:.1} MB; 4 have to fit in {} MiB)",
            need as f64 / 1e6,
            CAPACITY >> 20
        )));
    }
    Ok((count, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_as_documented() {
        assert_eq!(raw_layout(RawFormat::Bgra, 1920, 1080).unwrap(), (8, 8 << 20));
        assert_eq!(raw_layout(RawFormat::Nv12, 1920, 1080).unwrap(), (21, 3 << 20));
        assert_eq!(raw_layout(RawFormat::Bgra, 2560, 1440).unwrap(), (4, 15 << 20));
        assert!(raw_layout(RawFormat::Bgra, 3840, 2160).is_err());
        assert_eq!(TOTAL, 67_112_960);
        assert_eq!(AUDIO_TOTAL, 4_194_368);
    }
}
