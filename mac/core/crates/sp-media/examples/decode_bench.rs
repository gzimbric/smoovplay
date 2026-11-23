//! Decode time per frame at 1080p for VideoToolbox, then the conversion to a frame, on the fixture streams from
//! mac/core/tools/media_fixtures.py (H.264, HEVC 8-bit, HEVC 10-bit HDR10).
//!
//!     cargo run --release -p sp-media --example decode_bench [-- --rounds 3]
//!
//! Each stream is decoded `rounds` times, with a new session each round. Its first frame, which starts the session,
//! is left out of the times and shown on its own. Frames are fed one at a time, as the tap delivers them, so these are
//! latencies, not throughput.

use std::path::Path;

use sp_core::clock::now;
use sp_media::{Converter, HdrOut, HdrSettings, VtDecoder};
use sp_tap::Codec;

fn packets(name: &str) -> Option<Vec<Vec<u8>>> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/media/{name}.annexb"));
    let b = std::fs::read(p).ok()?;
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        let n = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        out.push(b[i + 4..i + 4 + n].to_vec());
        i += 4 + n;
    }
    Some(out)
}

fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms (n={})", p(0.5), p(0.95), p(1.0), v.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.iter().position(|a| a == "--rounds").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(3);
    let mut conv = Converter::new().expect("Metal");
    for (name, codec, label) in
        [("h264", Codec::H264, "H.264 8-bit"), ("hevc", Codec::Hevc, "HEVC 8-bit"), ("hevc10", Codec::Hevc, "HEVC 10-bit HDR10")]
    {
        let Some(pk) = packets(name) else {
            eprintln!("no fixtures: run mac/core/tools/media_fixtures.py");
            return;
        };
        let ten = name == "hevc10";
        let (mut dec_t, mut conv_t, mut total, mut hdr_t, mut first) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..rounds {
            let mut d = VtDecoder::new(codec);
            for (i, p) in pk.iter().enumerate() {
                let t0 = now();
                let pics = d.decode(p).expect("decode");
                let t1 = now();
                let f = conv.sdr(&pics[0]).expect("convert");
                let t2 = now();
                if ten {
                    let s = HdrSettings::new(HdrOut::Sdr8, None);
                    let _ = conv.hdr(&pics[0], &s).expect("hdr");
                    hdr_t.push((now() - t2) * 1e3);
                }
                drop(f);
                if i == 0 {
                    first.push((t2 - t0) * 1e3);
                    continue;
                }
                dec_t.push((t1 - t0) * 1e3);
                conv_t.push((t2 - t1) * 1e3);
                total.push((t2 - t0) * 1e3);
            }
            assert!(d.hardware, "{name}: not decoding on the media engine");
        }
        println!("{label} 1920x1080 ({} frames x {rounds}):", pk.len());
        println!("  decode (VideoToolbox)   {}", stats(dec_t));
        println!("  convert to BGRA8        {}", stats(conv_t));
        println!("  decode + convert        {}", stats(total));
        if ten {
            println!("  HDR10 to sdr8 instead   {}", stats(hdr_t));
        }
        println!("  first frame (new session) {}", stats(first));
    }
}
