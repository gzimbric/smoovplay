//! Annex-B access units as the tap carries them (framegen/chiaki_tap.py `nal_units`, `is_keyframe`, `param_sets`).
//!
//! A unit runs from just after a 3-byte start code (00 00 01) to the next one, as Python's `data.find` splits them:
//! a 4-byte start code's leading zero stays at the end of the unit before it. [`param_sets`] joins units with 3-byte
//! start codes exactly as Python does, so a saved codec header is the same bytes from either side.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

impl Codec {
    /// The tap header's codec (0 H.264, 1 H.265, 2 H.265 HDR); None for a decoded format or one nobody knows.
    pub fn of_tap(codec: u32) -> Option<Codec> {
        match codec {
            0 => Some(Codec::H264),
            1 | 2 => Some(Codec::Hevc),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
        }
    }
    /// A unit's type from its first header byte.
    pub fn unit_type(self, b: u8) -> u8 {
        match self {
            Codec::H264 => b & 0x1F,
            Codec::Hevc => (b >> 1) & 0x3F,
        }
    }
    /// VPS/SPS/PPS (HEVC 32-34), SPS/PPS (H.264 7, 8).
    pub fn is_param(self, t: u8) -> bool {
        match self {
            Codec::H264 => t == 7 || t == 8,
            Codec::Hevc => (32..=34).contains(&t),
        }
    }
    /// H.264: an IDR slice (5). HEVC: IRAP pictures, BLA/IDR/CRA (16-21).
    pub fn is_key(self, t: u8) -> bool {
        match self {
            Codec::H264 => t == 5,
            Codec::Hevc => (16..=21).contains(&t),
        }
    }
    /// A coded slice (a picture's data): H.264 1-5, HEVC 0-31.
    pub fn is_vcl(self, t: u8) -> bool {
        match self {
            Codec::H264 => (1..=5).contains(&t),
            Codec::Hevc => t < 32,
        }
    }
}

fn find3(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < 3 {
        return None;
    }
    let mut i = from;
    while i + 3 <= data.len() {
        // (memchr for the 01, then look back: start codes are rare in a slice's bytes)
        {
            let k = data[i + 2..].iter().position(|&b| b == 1)?;
            let j = i + 2 + k; // data[j] == 1
            if data[j - 1] == 0 && data[j - 2] == 0 {
                return Some(j - 2);
            }
            i = j - 1;
        }
    }
    None
}

/// (type, unit without its start code) for each NAL unit, as Python's nal_units (whose units are b"\0\0\1" + these).
pub fn nal_units(data: &[u8], codec: Codec) -> Vec<(u8, &[u8])> {
    let mut starts = Vec::new();
    let mut i = 0;
    while let Some(s) = find3(data, i) {
        starts.push(s);
        i = s + 3;
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        if s + 3 >= data.len() {
            break;
        }
        let end = starts.get(k + 1).copied().unwrap_or(data.len());
        out.push((codec.unit_type(data[s + 3]), &data[s + 3..end]));
    }
    out
}

pub fn is_keyframe(data: &[u8], codec: Codec) -> bool {
    nal_units(data, codec).iter().any(|&(t, _)| codec.is_key(t))
}

/// The codec header's units in an access unit, each after a 3-byte start code, or empty if none. chiaki-ng sends
/// these once, as a packet of their own at the start of a stream (the PS5 delivers them out of band).
pub fn param_sets(data: &[u8], codec: Codec) -> Vec<u8> {
    let mut out = Vec::new();
    for (t, u) in nal_units(data, codec) {
        if codec.is_param(t) {
            out.extend_from_slice(&[0, 0, 1]);
            out.extend_from_slice(u);
        }
    }
    out
}

/// True if every unit in the access unit is a parameter set (the header on its own).
pub fn only_params(data: &[u8], codec: Codec) -> bool {
    nal_units(data, codec).iter().all(|&(t, _)| codec.is_param(t))
}

/// A unit without the trailing zero bytes Annex-B allows after it (a 4-byte start code's first byte, or
/// trailing_zero_8bits): a NAL unit itself never ends in a zero byte.
pub fn trim(unit: &[u8]) -> &[u8] {
    let mut n = unit.len();
    while n > 0 && unit[n - 1] == 0 {
        n -= 1;
    }
    &unit[..n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_and_units() {
        let h264 = b"\x00\x00\x00\x01\x67sps\x00\x00\x01\x68pps\x00\x00\x01\x65idr";
        assert!(is_keyframe(h264, Codec::H264));
        assert!(!is_keyframe(b"\x00\x00\x01\x41p-slice", Codec::H264));
        assert!(is_keyframe(b"\x00\x00\x01\x40\x01vps\x00\x00\x01\x26\x01idr", Codec::Hevc));
        assert!(!is_keyframe(b"\x00\x00\x01\x02\x01trail", Codec::Hevc));
        let u = nal_units(h264, Codec::H264);
        assert_eq!(u.iter().map(|x| x.0).collect::<Vec<_>>(), vec![7, 8, 5]);
        assert_eq!(u[0].1, b"\x67sps");
        assert_eq!(param_sets(h264, Codec::H264), b"\x00\x00\x01\x67sps\x00\x00\x01\x68pps");
        assert!(!only_params(h264, Codec::H264));
        assert!(only_params(&param_sets(h264, Codec::H264), Codec::H264));
        // A 4-byte start code's leading zero stays with the unit before it, as in Python
        let two = b"\x00\x00\x01\x67a\x00\x00\x00\x01\x68b";
        assert_eq!(nal_units(two, Codec::H264)[0].1, b"\x67a\x00");
        assert_eq!(trim(b"\x67a\x00"), b"\x67a");
        assert!(nal_units(b"\x00\x00\x01", Codec::H264).is_empty());
        assert!(nal_units(b"", Codec::H264).is_empty());
    }
}
