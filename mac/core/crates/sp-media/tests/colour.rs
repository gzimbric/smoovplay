//! VideoToolbox and the conversion kernels against FFmpeg (PyAV) on the same packets, fixtures from
//! mac/core/tools/media_fixtures.py. The SDR path is checked against to_ndarray("bgr24"), and the HDR paths against
//! framegen/hdr.py's StreamConverter and its exact maths. Prints the differences, and fails past 1 level where the
//! Python path is meant to match.

mod common;

use common::*;
use sp_media::*;
use sp_tap::Codec;

const REF: [usize; 5] = [0, 1, 2, 31, 89];

fn decode_all(name: &str, codec: Codec) -> Option<Vec<Decoded>> {
    let packets = packets(name)?;
    let mut d = VtDecoder::new(codec);
    let mut out = Vec::new();
    for p in &packets {
        out.extend(d.decode(p).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    assert_eq!(out.len(), packets.len(), "{name}: a picture per access unit");
    Some(out)
}

#[test]
fn sdr_matches_pyav() {
    let mut conv = Converter::new().unwrap();
    for (name, codec) in [("h264", Codec::H264), ("hevc", Codec::Hevc)] {
        let Some(pics) = decode_all(name, codec) else { return eprintln!("skipped: run tools/media_fixtures.py") };
        let mut worst = 0;
        for i in REF {
            let f = conv.sdr(&pics[i]).unwrap();
            let (max, mean, over) = diff(&f.to_bgr(), &read(&format!("{name}_{i}.bgr")));
            eprintln!("{name} frame {i}: SDR vs PyAV bgr24: max {max}, mean {mean:.4}, over 1: {:.4}%", over * 100.0);
            worst = worst.max(max);
        }
        assert!(worst <= 1, "{name}: {worst} levels from PyAV");
    }
}

#[test]
fn hdr_matches_hdr_py() {
    let Some(pics) = decode_all("hevc10", Codec::Hevc) else { return eprintln!("skipped: run tools/media_fixtures.py") };
    let mut conv = Converter::new().unwrap();
    let sdr8 = HdrSettings::new(HdrOut::Sdr8, None);
    let pq16 = HdrSettings::new(HdrOut::Pq16, None);
    let (mut worst_exact, mut worst_over, mut worst_pq) = (0u8, 0f64, 0f32);
    for i in REF {
        let d = &pics[i];
        assert!(d.ten_bit);
        // With HDR off it's washed out, as in Python, which warns about it. swscale's 10-bit path isn't the 8-bit
        // one's maths (it seems to dither), so this is only reported.
        let off = conv.sdr(d).unwrap();
        let (max, mean, _) = diff(&off.to_bgr(), &read(&format!("hevc10_{i}.bgr")));
        eprintln!("hevc10 frame {i}: HDR off vs PyAV bgr24: max {max}, mean {mean:.4}");
        // sdr8 against the maths per pixel, and against Python's 3D LUT
        let (f, info) = conv.hdr(d, &sdr8).unwrap();
        assert_eq!(info.describe(), "10-bit PQ BT.2020");
        let ours = f.to_bgr();
        let (max_e, mean_e, over_e) = diff(&ours, &read(&format!("hevc10_{i}.exact.bgr")));
        let (max_l, mean_l, over_l) = diff(&ours, &read(&format!("hevc10_{i}.sdr8.bgr")));
        eprintln!(
            "hevc10 frame {i}: sdr8 vs hdr.py maths: max {max_e}, mean {mean_e:.4}, over 1 {:.4}%; vs its LUT: max \
             {max_l}, mean {mean_l:.4}, over 1 {:.3}%",
            over_e * 100.0,
            over_l * 100.0
        );
        worst_exact = worst_exact.max(max_e);
        worst_over = worst_over.max(over_e);
        // pq16 is half floats, RGBA here and BGR there
        let (f, _) = conv.hdr(d, &pq16).unwrap();
        let raw = f.read_raw();
        let theirs = read(&format!("hevc10_{i}.pq16.f16"));
        let mut max_pq = 0f32;
        for (p, q) in raw.as_chunks::<8>().0.iter().zip(theirs.as_chunks::<6>().0) {
            let h = |b: &[u8], k: usize| half(u16::from_le_bytes([b[2 * k], b[2 * k + 1]]));
            for (ours, theirs) in [(h(p, 0), h(q, 2)), (h(p, 1), h(q, 1)), (h(p, 2), h(q, 0))] {
                max_pq = max_pq.max((ours - theirs).abs());
            }
        }
        eprintln!("hevc10 frame {i}: pq16 vs StreamConverter: max {max_pq:.6} (of 0..1; a 10-bit step is {:.6})", 1.0 / 876.0);
        worst_pq = worst_pq.max(max_pq);
        // The 8-bit stand-in (hdr.py to_u8, v * 255 / PQ(203 nits) truncated) against the same sum on the CPU
        let level = conv.pq16_to_8(&f, false, &pq16).to_bgr();
        let scale = 255.0 / sp_media::colour::pq_of_nits(pq16.sdr_nits) as f32;
        let mut worst_level = 0u8;
        for (o, p) in level.as_chunks::<3>().0.iter().zip(raw.as_chunks::<8>().0) {
            let h = |k: usize| half(u16::from_le_bytes([p[2 * k], p[2 * k + 1]]));
            for (c, k) in [(0, 2), (1, 1), (2, 0)] {
                let want = (h(k) * scale).clamp(0.0, 255.0).floor() as u8;
                worst_level = worst_level.max(o[c].abs_diff(want));
            }
        }
        assert!(worst_level <= 1, "level8 {worst_level}");
        // pq16 back to sdr8, for instant replay's clips. It's the direct sdr8's maths, but from half floats. Where
        // the BT.2020-to-709 matrix leaves a channel of a bright, saturated colour near 0, SDR gamma is steep and the
        // half-float rounding of the PQ value shows as up to about 20 levels. Elsewhere it's 0 to 4.
        let back = conv.pq16_to_8(&f, true, &pq16).to_bgr();
        let (max_b, mean_b, _) = diff(&back, &ours);
        eprintln!("hevc10 frame {i}: pq16 to sdr8 vs sdr8 directly: max {max_b}, mean {mean_b:.4}, level8 within {worst_level}");
        assert!(mean_b < 1.0 && max_b <= 32);
    }
    // float32 on both sides, but different pow and exp. A few pixels a frame near black, where SDR gamma is
    // steepest, land 2 to 4 levels apart. Everything else is within 1.
    assert!(worst_exact <= 4 && worst_over < 1e-5, "sdr8 {worst_exact} levels ({worst_over}) from hdr.py's maths");
    assert!(worst_pq < 1.0 / 876.0, "pq16 {worst_pq} from StreamConverter");
}
