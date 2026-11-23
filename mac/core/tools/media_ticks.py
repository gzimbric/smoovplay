"""Dump the Python ChiakiSource's ticks from a live tap, one JSON line per tick, for comparing with the Rust one
(mac/core/crates/sp-media/tests/parity.rs, or `cargo run -p sp-media --example tap_dump -- NAME --jsonl OUT`).

    .venv/bin/python mac/core/tools/media_ticks.py 'Local\\NAME' OUT.jsonl [--test-mode] [--decoder videotoolbox]
        [--wait 30] [--hdr sdr8|pq16]

Reads until the stream ends (or --ticks N). Each line holds index, seq, size, size_new, size_ref, big, keyframe,
frames_lost, recovered, damaged, content, truth_new, new_hint, game_new, t_arrival, t_ready, decode_ms, and the
frame's mean level (a cheap check that a picture came out). The last line is {"done": ..., counts}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from framegen.chiaki_tap import ChiakiSource  # noqa: E402
from framegen.io import Waiting  # noqa: E402
from framegen.sources import TestMode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tap")
    ap.add_argument("out")
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--decoder", default="videotoolbox", choices=["videotoolbox", "sw"])
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--hdr", choices=["sdr8", "pq16"], default=None)
    a = ap.parse_args()
    test = TestMode(True) if a.test_mode else None
    if a.hdr:  # torch and the GPU first, so the reader isn't held up by them once the stream starts
        from framegen import gpu
        from framegen import hdr as H
        gpu.device()
    src = ChiakiSource(a.decoder, wait_s=a.wait, tap_name=a.tap, test=test)
    if a.hdr:
        src.set_hdr(a.hdr)
        if src.convert is not None:  # its 3D LUT for an HDR10 stream, made now rather than at the first frame
            c = src.convert
            fi = H.FrameInfo(10, False, H.SPC_BT2020_NCL, H.TRC_PQ, H.PRI_BT2020)
            c.info, c.lut = fi, H._yuv_lut(fi, c.out, c.src_peak, c.display_peak, H.SDR_WHITE_NITS, c.device)
    n = 0
    with open(a.out, "w") as f:
        for t in src:
            if isinstance(t, Waiting):
                continue
            e = t.extra
            img = t.frame
            mean = float(img.float().mean()) if hasattr(img, "float") else float(img.mean())
            f.write(json.dumps({
                "index": t.index, "seq": e.get("seq"), "size": e.get("size"), "size_new": e.get("size_new"),
                "size_ref": e.get("size_ref"), "big": e.get("big"), "keyframe": e.get("keyframe"),
                "frames_lost": e.get("frames_lost"), "recovered": e.get("recovered"), "damaged": t.damaged,
                "content": t.content, "truth_new": t.truth_new, "new_hint": t.new_hint, "game_new": e.get("game_new"),
                "t_arrival": t.t_arrival, "t_ready": t.t_ready, "decode_ms": e.get("decode_ms"), "mean": mean,
            }) + "\n")
            n += 1
            if a.ticks and n >= a.ticks:
                break
        f.write(json.dumps({"done": n, "dropped": src.dropped, "skipped": src.skipped,
                            "idr_requests": src.idr_requests}) + "\n")
    src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
