//! --score and test mode's quality measures (framegen/pipeline.py _score, hud_mask, psnr). Each generated frame is
//! compared with the real one it stood in for, as are the last real frame repeated and a 50/50 blend, over the whole
//! picture, still areas and HUD-like areas. A probe ([`crate::probe`]) adds up the per-pixel numbers, on the GPU or
//! on the CPU from `Frame::to_bgr` with [`score_sums_bgr`]. [`add_scores`] turns the sums into Stats' figures with
//! the same float32 and float64 steps as the torch code, so the reports read the same.

use crate::stats::{HUD_EDGE, STILL_LEVELS, Stats, VISIBLE_LEVELS};

/// PQ (SMPTE ST 2084), framegen/hdr.py.
const M1: f64 = 2610.0 / 16384.0;
const M2: f64 = 2523.0 / 4096.0 * 128.0;
const C1: f64 = 3424.0 / 4096.0;
const C2: f64 = 2413.0 / 4096.0 * 32.0;
const C3: f64 = 2392.0 / 4096.0 * 32.0;
pub const SDR_WHITE_NITS: f64 = 203.0;

pub fn pq_of_nits(nits: f64) -> f64 {
    let p = (nits / 10000.0).clamp(0.0, 1.0).powf(M1);
    ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
}

/// hdr.py LEVEL8_SCALE. A PQ value times this is in 8-bit levels, with SDR white at 255.
pub fn level8_scale() -> f64 {
    255.0 / pq_of_nits(SDR_WHITE_NITS)
}

/// The per-pixel numbers of one scored frame, added up over the picture. Pixels are in 8-bit levels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScoreSums {
    /// Squared error summed over pixels and colours, for the AI frame, the last frame repeated and the 50/50 blend.
    pub se_out: f64,
    pub se_rep: f64,
    pub se_blend: f64,
    /// Pixels within STILL_LEVELS in both real frames.
    pub n_still: f64,
    pub se_out_still: f64,
    pub se_rep_still: f64,
    /// Pixels more than VISIBLE_LEVELS off in any colour, in the AI frame and in the last frame repeated.
    pub n_off: f64,
    pub n_off_rep: f64,
    /// HUD-like pixels (still in a, gt and b, and on an edge of gt), then the AI frame's squared error and its
    /// visibly-off pixels there.
    pub n_hud: f64,
    pub se_out_hud: f64,
    pub n_off_hud: f64,
    /// Pixels in the picture.
    pub pixels: f64,
}

impl ScoreSums {
    pub fn add(&mut self, o: &ScoreSums) {
        self.se_out += o.se_out;
        self.se_rep += o.se_rep;
        self.se_blend += o.se_blend;
        self.n_still += o.n_still;
        self.se_out_still += o.se_out_still;
        self.se_rep_still += o.se_rep_still;
        self.n_off += o.n_off;
        self.n_off_rep += o.n_off_rep;
        self.n_hud += o.n_hud;
        self.se_out_hud += o.se_out_hud;
        self.n_off_hud += o.n_off_hud;
    }
}

/// 10 log10(255^2 / mse), inf for a perfect match.
pub fn db(mse: f64) -> f64 {
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64.powi(2) / mse).log10() }
}

/// PSNR of two packed 8-bit images.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum();
    db((se / a.len().max(1) as f64) as f32 as f64)
}

/// Brightness (BT.601 from BGR) as torch has it, float32.
fn luma(p: [f32; 3]) -> f32 {
    p[0] * 0.114 + p[1] * 0.587 + p[2] * 0.299
}

/// Sobel L1 of gt's brightness above HUD_EDGE, edges replicated (hud_mask's edge test), for every pixel.
fn edges(gt: &[[f32; 3]], w: usize, h: usize) -> Vec<bool> {
    let l: Vec<f32> = gt.iter().map(|&p| luma(p)).collect();
    let at = |x: isize, y: isize| l[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let mut out = vec![false; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y) + 2.0 * at(x + 1, y) - at(x - 1, y + 1)
                + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)
                + at(x - 1, y + 1)
                + 2.0 * at(x, y + 1)
                + at(x + 1, y + 1);
            out[y as usize * w + x as usize] = gx.abs() + gy.abs() > HUD_EDGE;
        }
    }
    out
}

fn max_abs(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

/// HUD-like pixels for scoring. These are unchanged within STILL_LEVELS across the real frames a, gt (the real
/// in-between one) and b, and on an edge of gt's brightness. Images are BGR levels from 0 to 255, one [f32; 3] per
/// pixel.
pub fn hud_mask(a: &[[f32; 3]], b: &[[f32; 3]], gt: &[[f32; 3]], w: usize, h: usize) -> Vec<bool> {
    let e = edges(gt, w, h);
    (0..w * h).map(|i| max_abs(b[i], a[i]) <= STILL_LEVELS && max_abs(gt[i], a[i]) <= STILL_LEVELS && e[i]).collect()
}

fn px(bgr: &[u8]) -> Vec<[f32; 3]> {
    bgr.as_chunks::<3>().0.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect()
}

/// The sums from packed 8-bit BGR. `a` is the previous real frame, `b` the next one, `out` the AI frame and `gt` the
/// real one it stood in for. This is the CPU route and the reference for the GPU's.
pub fn score_sums_bgr(a: &[u8], b: &[u8], out: &[u8], gt: &[u8], w: usize, h: usize) -> ScoreSums {
    score_sums_levels(&px(a), &px(b), &px(out), &px(gt), w, h)
}

pub fn score_sums_levels(a: &[[f32; 3]], b: &[[f32; 3]], out: &[[f32; 3]], gt: &[[f32; 3]], w: usize, h: usize) -> ScoreSums {
    let e = edges(gt, w, h);
    let mut s = ScoreSums { pixels: (w * h) as f64, ..Default::default() };
    for i in 0..w * h {
        let (va, vb, vo, vg) = (a[i], b[i], out[i], gt[i]);
        let (mut se_o, mut se_r, mut se_b) = (0f64, 0f64, 0f64);
        for c in 0..3 {
            let blend = ((va[c] + vb[c]) * 0.5).round_ties_even();
            se_o += ((vo[c] - vg[c]) as f64).powi(2);
            se_r += ((va[c] - vg[c]) as f64).powi(2);
            se_b += ((blend - vg[c]) as f64).powi(2);
        }
        let still = max_abs(vb, va) <= STILL_LEVELS;
        let off = max_abs(vo, vg) > VISIBLE_LEVELS;
        let off_rep = max_abs(va, vg) > VISIBLE_LEVELS;
        let hud = still && max_abs(vg, va) <= STILL_LEVELS && e[i];
        s.se_out += se_o;
        s.se_rep += se_r;
        s.se_blend += se_b;
        if still {
            s.n_still += 1.0;
            s.se_out_still += se_o;
            s.se_rep_still += se_r;
        }
        s.n_off += off as u8 as f64;
        s.n_off_rep += off_rep as u8 as f64;
        if hud {
            s.n_hud += 1.0;
            s.se_out_hud += se_o;
            s.n_off_hud += off as u8 as f64;
        }
    }
    s
}

fn f32div(a: f64, b: f64) -> f64 {
    (a as f32 / b as f32) as f64
}

/// The figures one scored frame adds to Stats, from its sums. The arithmetic is pipeline.py _score's, with torch's
/// float32 means and then Python floats. Returns the frame's PSNR.
pub fn add_scores(st: &mut Stats, s: &ScoreSums) -> f64 {
    let n = s.pixels * 3.0;
    let mse = [f32div(s.se_out, n), f32div(s.se_rep, n), f32div(s.se_blend, n)];
    let n_still = s.n_still.max(1.0) * 3.0;
    let still_out = f32div(s.se_out_still, n_still);
    let still_rep = f32div(s.se_rep_still, n_still);
    let still_frac = f32div(s.n_still, s.pixels);
    let off = f32div(s.n_off, s.pixels);
    let off_rep = f32div(s.n_off_rep, s.pixels);
    let (p, p_repeat, p_blend) = (db(mse[0]), db(mse[1]), db(mse[2]));
    if still_frac > 0.0 {
        st.psnr_still.push(db(still_out));
        st.psnr_still_repeat.push(db(still_rep));
        st.still_frac.push(still_frac);
    }
    st.psnr_interp.push(p);
    st.psnr_repeat.push(p_repeat);
    st.psnr_blend.push(p_blend);
    st.off_interp.push(off);
    st.off_repeat.push(off_rep);
    let n_hud = s.n_hud as f32 as f64;
    if n_hud != 0.0 {
        st.psnr_hud.push(db(s.se_out_hud as f32 as f64 / (3.0 * n_hud)));
        st.hud_off.push(s.n_off_hud as f32 as f64 / n_hud);
        st.hud_frac.push(n_hud / s.pixels);
    }
    p
}
