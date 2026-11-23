//! What a decoded picture's samples mean, with FFmpeg's numbering (AVColorSpace, AVColorTransferCharacteristic,
//! AVColorPrimaries), as framegen/hdr.py's FrameInfo and PyAV's to_ndarray read them. Also the colour maths'
//! constants from hdr.py for PQ, BT.2390 and the BT.2020 and BT.709 matrices.

use objc2_core_foundation::{CFRetained, CFString};
use objc2_core_media::CMFormatDescription;
use objc2_core_video::{
    kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey, CVBuffer,
};

// FFmpeg's values (pixfmt.h), as hdr.py names them
pub const UNSPECIFIED: u32 = 2;
pub const TRC_BT709: u32 = 1;
pub const TRC_PQ: u32 = 16;
pub const TRC_HLG: u32 = 18;
pub const PRI_BT709: u32 = 1;
pub const PRI_BT2020: u32 = 9;
pub const SPC_BT709: u32 = 1;
pub const SPC_BT2020_NCL: u32 = 9;
pub const SPC_BT2020_CL: u32 = 10;

/// A decoded picture's colour tags and range. The tags are FFmpeg's numbers, with 2 for unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tags {
    pub matrix: u32,
    pub transfer: u32,
    pub primaries: u32,
    pub full_range: bool,
}

fn attachment(buf: &CVBuffer, key: &CFString) -> Option<String> {
    let v = unsafe { buf.attachment(key, std::ptr::null_mut()) }?;
    let s: CFRetained<CFString> = v.downcast().ok()?;
    Some(s.to_string())
}

fn extension(f: &CMFormatDescription, key: &CFString) -> Option<String> {
    let v = unsafe { f.extension(key) }?;
    let s: CFRetained<CFString> = v.downcast().ok()?;
    Some(s.to_string())
}

impl Tags {
    /// From a format description made from a stream's parameter sets. That gives its colour description if the
    /// stream has one, or unspecified as FFmpeg reports it. VideoToolbox's pictures carry BT.709 attachments even for
    /// an untagged stream, so those don't tell what the stream says.
    pub fn of_format(f: &CMFormatDescription, full_range: bool) -> Tags {
        // The format description's keys are the same strings as the image buffer's.
        let (m, t, p) = unsafe {
            (
                extension(f, kCVImageBufferYCbCrMatrixKey),
                extension(f, kCVImageBufferTransferFunctionKey),
                extension(f, kCVImageBufferColorPrimariesKey),
            )
        };
        Self::of_names(m, t, p, full_range)
    }

    /// From a CVPixelBuffer's attachments (what CoreVideo thinks the picture is, defaults included) and its pixel
    /// format's range.
    pub fn of_buffer(buf: &CVBuffer, full_range: bool) -> Tags {
        let (m, t, p) = unsafe {
            (
                attachment(buf, kCVImageBufferYCbCrMatrixKey),
                attachment(buf, kCVImageBufferTransferFunctionKey),
                attachment(buf, kCVImageBufferColorPrimariesKey),
            )
        };
        Self::of_names(m, t, p, full_range)
    }

    fn of_names(m: Option<String>, t: Option<String>, p: Option<String>, full_range: bool) -> Tags {
        let matrix = match m.as_deref() {
            Some("ITU_R_709_2") => 1,
            Some("ITU_R_601_4") => 6,
            Some("SMPTE_240M_1995") => 7,
            Some("ITU_R_2020") => 9,
            _ => UNSPECIFIED,
        };
        let transfer = match t.as_deref() {
            Some("ITU_R_709_2") => 1,
            Some("SMPTE_240M_1995") => 7,
            Some("Linear") => 8,
            Some("IEC_sRGB") | Some("sRGB") => 13,
            Some("ITU_R_2020") => 14,
            Some("SMPTE_ST_2084_PQ") => 16,
            Some("SMPTE_ST_428_1") => 17,
            Some("ITU_R_2100_HLG") => 18,
            _ => UNSPECIFIED,
        };
        let primaries = match p.as_deref() {
            Some("ITU_R_709_2") => 1,
            Some("EBU_3213") => 5,
            Some("SMPTE_C") => 6,
            Some("ITU_R_2020") => 9,
            Some("DCI_P3") => 11,
            Some("P3_D65") => 12,
            Some("P22") => 22,
            _ => UNSPECIFIED,
        };
        Tags { matrix, transfer, primaries, full_range }
    }

    /// The Y'CbCr matrix, as (kr, kb), that PyAV's to_ndarray("bgr24") uses for these tags through swscale. That's
    /// BT.709 or BT.2020 if tagged so, else BT.601, untagged included.
    pub fn swscale_kr_kb(&self) -> (f64, f64) {
        match self.matrix {
            SPC_BT709 => (0.2126, 0.0722),
            SPC_BT2020_NCL | SPC_BT2020_CL => (0.2627, 0.0593),
            7 => (0.212, 0.087),
            _ => (0.299, 0.114),
        }
    }
}

/// hdr.py's FrameInfo, which says what an HDR path's samples mean, with its defaults for untagged streams. 10-bit
/// HEVC from a PS5 in HDR mode is HDR10, and untagged 8-bit is BT.709.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameInfo {
    pub bits: u32,
    pub full_range: bool,
    pub matrix: u32,
    pub transfer: u32,
    pub primaries: u32,
}

impl FrameInfo {
    pub fn of(tags: &Tags, bits: u32) -> FrameInfo {
        let hdr_default = bits > 8;
        let mut trc = tags.transfer;
        if ![TRC_PQ, TRC_HLG, TRC_BT709, 6, 14, 15].contains(&trc) {
            trc = if hdr_default { TRC_PQ } else { TRC_BT709 };
        }
        let mut pri = tags.primaries;
        if ![PRI_BT709, PRI_BT2020, 5, 6].contains(&pri) {
            pri = if trc == TRC_PQ || trc == TRC_HLG { PRI_BT2020 } else { PRI_BT709 };
        }
        let mut spc = tags.matrix;
        if ![SPC_BT709, SPC_BT2020_NCL, SPC_BT2020_CL, 5, 6].contains(&spc) {
            spc = if pri == PRI_BT2020 { SPC_BT2020_NCL } else { SPC_BT709 };
        }
        FrameInfo { bits, full_range: tags.full_range, matrix: spc, transfer: trc, primaries: pri }
    }
    pub fn hdr(&self) -> bool {
        self.transfer == TRC_PQ || self.transfer == TRC_HLG
    }
    /// hdr.py KR_KB.get(matrix, KR_KB_601).
    pub fn kr_kb(&self) -> (f64, f64) {
        match self.matrix {
            SPC_BT709 => (0.2126, 0.0722),
            SPC_BT2020_NCL | SPC_BT2020_CL => (0.2627, 0.0593),
            _ => (0.299, 0.114),
        }
    }
    /// For the log, as in "10-bit PQ BT.2020" (StreamConverter.describe).
    pub fn describe(&self) -> String {
        let trc = match self.transfer {
            TRC_PQ => "PQ",
            TRC_HLG => "HLG",
            _ => "SDR gamma",
        };
        let pri = match self.primaries {
            PRI_BT2020 => "BT.2020".to_string(),
            PRI_BT709 => "BT.709".to_string(),
            p => format!("primaries {p}"),
        };
        format!("{}-bit {trc} {pri}{}", self.bits, if self.full_range { " full range" } else { "" })
    }
}

// ------------------------------------------------------------------------------------------- hdr.py's constants
pub const M1: f64 = 2610.0 / 16384.0;
pub const M2: f64 = 2523.0 / 4096.0 * 128.0;
pub const C1: f64 = 3424.0 / 4096.0;
pub const C2: f64 = 2413.0 / 4096.0 * 32.0;
pub const C3: f64 = 2392.0 / 4096.0 * 32.0;
pub const PQ_MAX_NITS: f64 = 10000.0;
pub const HLG_A: f64 = 0.17883277;
pub const HLG_B: f64 = 0.28466892;
pub const HLG_C: f64 = 0.55991073;
pub const HLG_DISPLAY_NITS: f64 = 1000.0;
pub const HLG_GAMMA: f64 = 1.2;
pub const SDR_GAMMA: f64 = 2.4;

/// BT.2408's SDR white (FRAMEGEN_HDR_SDR_NITS overrides it, as in Python).
pub fn sdr_white_nits() -> f64 {
    std::env::var("FRAMEGEN_HDR_SDR_NITS").ok().and_then(|v| v.parse().ok()).unwrap_or(203.0)
}

/// The brightest a game's highlights go when the stream doesn't say (FRAMEGEN_HDR_PEAK overrides it).
pub fn source_peak_nits() -> f64 {
    std::env::var("FRAMEGEN_HDR_PEAK").ok().and_then(|v| v.parse().ok()).unwrap_or(1000.0)
}

pub fn pq_oetf(y: f64) -> f64 {
    let p = y.clamp(0.0, 1.0).powf(M1);
    ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
}

pub fn pq_eotf(e: f64) -> f64 {
    let p = e.clamp(0.0, 1.0).powf(1.0 / M2);
    ((p - C1).max(0.0) / (C2 - C3 * p)).powf(1.0 / M1)
}

pub fn pq_of_nits(nits: f64) -> f64 {
    pq_oetf(nits / PQ_MAX_NITS)
}

type M3 = [[f64; 3]; 3];

const BT709_XY: [(f64, f64); 3] = [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)];
const BT2020_XY: [(f64, f64); 3] = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)];
const D65: (f64, f64) = (0.3127, 0.3290);

#[allow(clippy::needless_range_loop)] // the cofactor formula, as written
fn inv(m: &M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            r[i][j] = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det;
        }
    }
    r
}

fn mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

/// As hdr.py rgb_to_xyz, linear RGB with these primaries to CIE XYZ (Y = 1 for white).
fn rgb_to_xyz(primaries: &[(f64, f64); 3]) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for (j, &(x, y)) in primaries.iter().enumerate() {
        m[0][j] = x / y;
        m[1][j] = 1.0;
        m[2][j] = (1.0 - x - y) / y;
    }
    let (wx, wy) = D65;
    let w = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    let mi = inv(&m);
    let s: Vec<f64> = (0..3).map(|i| (0..3).map(|k| mi[i][k] * w[k]).sum()).collect();
    for row in m.iter_mut() {
        for j in 0..3 {
            row[j] *= s[j];
        }
    }
    m
}

/// Linear BT.2020 RGB to linear BT.709 (hdr.py BT2020_TO_709).
pub fn bt2020_to_709() -> M3 {
    mul(&inv(&rgb_to_xyz(&BT709_XY)), &rgb_to_xyz(&BT2020_XY))
}

pub fn bt709_to_2020() -> M3 {
    inv(&bt2020_to_709())
}

/// BT.2020's luma weights (0.2627, 0.6780, 0.0593).
pub fn bt2020_luma() -> [f64; 3] {
    rgb_to_xyz(&BT2020_XY)[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_and_pq() {
        let m = bt2020_to_709();
        // numpy's values for hdr.BT2020_TO_709
        assert!((m[0][0] - 1.6605).abs() < 1e-3 && (m[0][1] + 0.5876).abs() < 1e-3 && (m[2][2] - 1.1187).abs() < 1e-3);
        let l = bt2020_luma();
        assert!((l[0] - 0.2627).abs() < 1e-3 && (l[1] - 0.6780).abs() < 1e-3 && (l[2] - 0.0593).abs() < 1e-3);
        assert!((pq_eotf(pq_oetf(0.0203)) - 0.0203).abs() < 1e-9);
        assert!((pq_of_nits(10000.0) - 1.0).abs() < 1e-9);
        let i = mul(&bt2020_to_709(), &bt709_to_2020());
        assert!((i[0][0] - 1.0).abs() < 1e-9 && i[0][1].abs() < 1e-9);
    }

    #[test]
    fn frame_info_defaults() {
        let untagged = Tags { matrix: 2, transfer: 2, primaries: 2, full_range: false };
        let f = FrameInfo::of(&untagged, 10);
        assert_eq!((f.transfer, f.primaries, f.matrix), (TRC_PQ, PRI_BT2020, SPC_BT2020_NCL));
        let f = FrameInfo::of(&untagged, 8);
        assert_eq!((f.transfer, f.primaries, f.matrix), (TRC_BT709, PRI_BT709, SPC_BT709));
        assert_eq!(untagged.swscale_kr_kb(), (0.299, 0.114));
    }
}
