//! BT.709, limited range, for BGR input (replay.py's colour section): Y = 16 + 219 (kr R + kg G + kb B), Cb/Cr =
//! 128 + 224 (...), with R..B in 0..1. The GPU kernels use these weights; [`bgr_to_i420`] is the same sums on the CPU
//! (for the labels, the divider and the tests).

pub const KR: f32 = 0.2126;
pub const KB: f32 = 0.0722;
pub const KG: f32 = 1.0 - KR - KB;

/// Rows Y, Cb, Cr of weights on (B, G, R) in 0..255, giving 0..255 (before the +16 / +128): replay.py _M709.
pub fn m709() -> [[f32; 3]; 3] {
    let s = [219.0f32 / 255.0, 224.0 / 255.0, 224.0 / 255.0];
    let rows = [
        [KB, KG, KR],
        [0.5, -KG / (1.0 - KB) / 2.0, -KR / (1.0 - KB) / 2.0],
        [-KB / (1.0 - KR) / 2.0, -KG / (1.0 - KR) / 2.0, 0.5],
    ];
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = rows[i][j] * s[i];
        }
    }
    out
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn u8_of(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// H x W packed BGR (even size) -> I420 (BT.709, limited range): Y (h*w), then U, then V (each h/2 * w/2), as
/// replay.py bgr_to_i420.
pub fn bgr_to_i420(bgr: &[u8], w: usize, h: usize) -> Vec<u8> {
    assert!(w.is_multiple_of(2) && h.is_multiple_of(2) && bgr.len() >= w * h * 3);
    let m = m709();
    let mut out = vec![0u8; w * h * 3 / 2];
    let px = |x: usize, y: usize| {
        let i = (y * w + x) * 3;
        [bgr[i] as f32, bgr[i + 1] as f32, bgr[i + 2] as f32]
    };
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = u8_of(dot(px(x, y), m[0]) + 16.0);
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    for y in 0..ch {
        for x in 0..cw {
            let mut s = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = px(2 * x + dx, 2 * y + dy);
                for k in 0..3 {
                    s[k] += p[k];
                }
            }
            let s = [s[0] / 4.0, s[1] / 4.0, s[2] / 4.0];
            out[w * h + y * cw + x] = u8_of(dot(s, m[1]) + 128.0);
            out[w * h + cw * ch + y * cw + x] = u8_of(dot(s, m[2]) + 128.0);
        }
    }
    out
}

/// One colour's (Y, U, V).
pub fn yuv_of(bgr: [u8; 3]) -> [u8; 3] {
    let px: Vec<u8> = bgr.iter().cycle().take(12).copied().collect();
    let i = bgr_to_i420(&px, 2, 2);
    [i[0], i[4], i[5]]
}
