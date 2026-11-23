//! What the decoder needs from a stream's sequence parameter set before VideoToolbox starts. That's the bit depth, 8
//! or 10, which sets the pixel format to ask for. VideoToolbox reads everything else itself.

use sp_tap::Codec;

/// A NAL unit's payload with emulation prevention bytes (00 00 03) removed.
fn rbsp(unit: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(unit.len());
    let mut zeros = 0;
    for &b in unit {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn u(&mut self, n: usize) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.d.get(self.pos / 8)?;
            v = (v << 1) | ((byte >> (7 - self.pos % 8)) & 1) as u32;
            self.pos += 1;
        }
        Some(v)
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.pos += n;
        (self.pos <= self.d.len() * 8).then_some(())
    }
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.u(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1u32 << zeros) - 1 + self.u(zeros)?)
    }
}

/// Luma bit depth from a sequence parameter set (the unit without its start code, header bytes included).
pub fn bit_depth(sps: &[u8], codec: Codec) -> Option<u32> {
    let d = rbsp(sps);
    match codec {
        Codec::H264 => {
            let mut b = Bits { d: &d, pos: 8 };
            let profile = b.u(8)?;
            b.skip(16)?;
            b.ue()?; // seq_parameter_set_id
            if [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].contains(&profile) {
                if b.ue()? == 3 {
                    b.skip(1)?;
                }
                return Some(8 + b.ue()?);
            }
            Some(8)
        }
        Codec::Hevc => {
            let mut b = Bits { d: &d, pos: 16 };
            b.skip(4)?; // sps_video_parameter_set_id
            let max_sub = b.u(3)? as usize;
            b.skip(1)?;
            b.skip(96)?; // general profile, tier, flags, level
            let mut present = Vec::new();
            for _ in 0..max_sub {
                present.push((b.u(1)?, b.u(1)?));
            }
            if max_sub > 0 {
                b.skip(2 * (8 - max_sub))?;
            }
            for (p, l) in present {
                if p == 1 {
                    b.skip(88)?;
                }
                if l == 1 {
                    b.skip(8)?;
                }
            }
            b.ue()?; // sps_seq_parameter_set_id
            if b.ue()? == 3 {
                b.skip(1)?;
            }
            b.ue()?;
            b.ue()?; // width, height
            if b.u(1)? == 1 {
                for _ in 0..4 {
                    b.ue()?;
                }
            }
            Some(8 + b.ue()?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_and_emulation_prevention() {
        assert_eq!(rbsp(&[0, 0, 3, 1, 0, 0, 3]), vec![0, 0, 1, 0, 0]);
        let d = [0b1010_0110, 0b0100_0000];
        let mut b = Bits { d: &d, pos: 0 };
        assert_eq!((b.ue(), b.ue(), b.ue(), b.ue()), (Some(0), Some(1), Some(2), Some(3)));
    }
}
