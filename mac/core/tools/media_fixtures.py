"""Reference data for sp-media's decoder and colour tests (mac/core/fixtures/media/, git-ignored).

    .venv/bin/python mac/core/tools/media_fixtures.py [--frames 240] [--hdr-frames 90]

Encodes testdata/clip_1080p60.mp4 the way scripts/frametap_fake.py does on a Mac (its own encoder() and make_hdr(),
on the CPU), as three streams of Annex-B access units:

  h264.annexb     H.264 (libx264), tagged BT.709 limited range (as an encoder that writes its colours does)
  hevc.annexb     H.265 8-bit (libx265), untagged (what the fake PS5 sends)
  hevc10.annexb   H.265 Main10 HDR10 (libx265 10-bit, PQ, BT.2020), what a PS5 in HDR mode sends

Each file holds, per access unit, a little-endian u32 length and the bytes. Then the references, decoded by FFmpeg in
software from the same packets, for a few frames (REF):

  <stream>_<i>.bgr      PyAV's to_ndarray("bgr24") (what framegen/chiaki_tap.py's VideoToolbox path hands on)
  hevc10_<i>.sdr8.bgr   framegen.hdr.StreamConverter("sdr8") (its 3D LUT, as the Python pipeline runs it)
  hevc10_<i>.exact.bgr  framegen.hdr.convert_yuv(..., "sdr8"), the same maths per pixel with no LUT
  hevc10_<i>.pq16.f16   StreamConverter("pq16"), HxWx3 float16 BGR in PQ BT.2020

and manifest.json describing them. For the file sources:

  clip_30.bgr     frame 30 of the clip itself, PyAV to_ndarray("bgr24") (FileSource's colours)
  sim_ticks.json  framegen.io.Ps5SimSource's (content, truth_new) for the first 120 ticks with hitch 0.3, seed 0

--sources-only remakes only those two.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import frametap_fake as fake  # noqa: E402

OUT = ROOT / "mac/core/fixtures/media"
REF = [0, 1, 2, 31, 89]


def encode(codec: str, frames: int, hdr: bool, tag709: bool) -> list[bytes]:
    src = av.open(str(ROOT / "testdata/clip_1080p60.mp4"))
    vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    enc = fake.encoder(codec, w, h, 15.0, sw=True, hdr=hdr)
    if tag709:  # encoder() sets these before open(), so it's reopened here with the tags
        enc = av.CodecContext.create(enc.name, "w")
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        from fractions import Fraction
        enc.time_base, enc.framerate, enc.bit_rate, enc.gop_size, enc.max_b_frames = (
            Fraction(1, 60), Fraction(60), 15_000_000, 100000, 0)
        enc.color_primaries, enc.color_trc, enc.colorspace, enc.color_range = 1, 1, 1, 1
        enc.options = {"preset": "veryfast", "tune": "zerolatency", "forced-idr": "1"}
        enc.open()
    to_hdr = fake.make_hdr(enc, w, h, 1000.0, True) if hdr else None
    out = []
    for i, fr in enumerate(src.decode(vs)):
        if i >= frames:
            break
        img = to_hdr(fr) if hdr else fr.reformat(format="yuv420p")
        if tag709:
            img.colorspace, img.color_range = 1, 1
        img.pts = i
        if i == 60:  # one forced intra frame mid-stream, as after a keyframe request
            img.pict_type = av.video.frame.PictureType.I
        data = b"".join(bytes(p) for p in enc.encode(img))
        if data:
            out.append(data)
        print(f"\r{codec}{' hdr' if hdr else ''}: {i + 1}/{frames}", end="", flush=True)
    out += [bytes(p) for p in enc.encode(None)]
    print()
    return out


def write_stream(name: str, packets: list[bytes]) -> None:
    with open(OUT / f"{name}.annexb", "wb") as f:
        for p in packets:
            f.write(struct.pack("<I", len(p)) + p)


def references(name: str, codec: str, packets: list[bytes], hdr: bool) -> dict:
    dec = av.CodecContext.create(codec, "r")
    frames = []
    for p in packets:
        frames += dec.decode(av.Packet(p))
    frames += dec.decode(None)
    info = {"decoded": len(frames), "refs": REF}
    fr0 = frames[0]
    info["tags"] = {"colorspace": int(fr0.colorspace), "range": int(fr0.color_range),
                    "trc": int(fr0.color_trc), "primaries": int(fr0.color_primaries), "format": fr0.format.name}
    conv = {}
    if hdr:
        from framegen import hdr as H
        conv = {"sdr8": H.StreamConverter("sdr8", "cpu"), "pq16": H.StreamConverter("pq16", "cpu")}
    for i in REF:
        fr = frames[i]
        fr.to_ndarray(format="bgr24").tofile(OUT / f"{name}_{i}.bgr")
        if hdr:
            conv["sdr8"](fr).numpy().tofile(OUT / f"{name}_{i}.sdr8.bgr")
            conv["pq16"](fr).numpy().tofile(OUT / f"{name}_{i}.pq16.f16")
            bits, (y, u, v) = conv["sdr8"]._planes(fr)
            fi = H.FrameInfo.of(fr, bits)
            H.convert_yuv(y, u, v, fi, "sdr8").numpy().tofile(OUT / f"{name}_{i}.exact.bgr")
            info["frame_info"] = {"bits": fi.bits, "full_range": fi.full_range, "matrix": fi.matrix,
                                  "transfer": fi.transfer, "primaries": fi.primaries}
    return info


def source_refs() -> None:
    clip = str(ROOT / "testdata/clip_1080p60.mp4")
    c = av.open(clip)
    for i, fr in enumerate(c.decode(c.streams.video[0])):
        if i == 30:
            fr.to_ndarray(format="bgr24").tofile(OUT / "clip_30.bgr")
            break
    c.close()
    from framegen.io import Ps5SimSource
    sim = Ps5SimSource(clip, hitch=0.3, seed=0, max_ticks=120)
    ticks = [[t.content, t.truth_new] for t in sim]
    (OUT / "sim_ticks.json").write_text(json.dumps(ticks))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--hdr-frames", type=int, default=90)
    ap.add_argument("--sources-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source_refs()
    if a.sources_only:
        return 0
    manifest = {"width": 1920, "height": 1080, "streams": {}}
    for name, codec, hdr, tag, n in [("h264", "h264", False, True, a.frames), ("hevc", "hevc", False, False, a.frames),
                                     ("hevc10", "hevc", True, False, a.hdr_frames)]:
        packets = encode(codec, n, hdr, tag)
        write_stream(name, packets)
        info = references(name, codec, packets, hdr)
        info.update(codec=codec, packets=len(packets), hdr=hdr, bytes=sum(map(len, packets)))
        manifest["streams"][name] = info
        print(name, json.dumps(info))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
