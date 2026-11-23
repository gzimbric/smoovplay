"""Reference data for sp-rife's parity tests: real frame pairs from the test clip and what the Python Mac engine
(framegen/interpolator.py RifeInterpolator on Apple's GPU, i.e. framegen/metalrife.py's MetalIFNet) draws between
them, plus the real in-between frames for scoring.

Run from the repo root with the project's venv:

    .venv/bin/python mac/core/tools/rife_fixtures.py

It writes mac/core/fixtures/rife/ (git-ignored): raw frames and index.json. Frames are packed HxWx3 BGR, uint8
(".bgr") or, for the HDR case, little-endian float16 PQ (".bgr16", framegen/hdr.py's pq16). index.json lists the
cases with their settings, the files, and the Python output's PSNR against the real frame, which the Rust tests
must match within 0.05 dB.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from framegen import hdr, metalrife  # noqa: E402
from framegen.interpolator import MODELS, RifeInterpolator  # noqa: E402

CLIP = ROOT / "testdata" / "clip_1080p60.mp4"
OUT = ROOT / "mac" / "core" / "fixtures" / "rife"
MODEL = "4.22.lite"
STEADY = 0.03  # framegen/steady.py STEADY_HUD_THRESHOLD

# (name, first frame, last frame): the real frames between them are the truth for t = k / (last - first).
PAIRS = [("p120", 120, 122), ("p210", 210, 213), ("p440", 440, 442)]
# (pair, scale, t's, steady, hdr)
CASES = [
    ("p120", 1.0, [1 / 2], False, False),
    ("p120", 0.5, [1 / 2], False, False),
    ("p210", 1.0, [1 / 3, 2 / 3], False, False),
    ("p210", 0.5, [1 / 3, 2 / 3], False, False),
    ("p120", 1.0, [1 / 2], True, False),
    ("p440", 0.5, [1 / 2], True, False),
    ("p440", 1.0, [1 / 2], False, True),
    ("p120", 0.25, [1 / 2], False, False),
]
# RIFE versions with five blocks, on random weights (no download needed; checkpoints written here in vs-rife's
# format): (version, scale, crop size). As tests/test_metalrife.py, the blocks' last layers are scaled down so the
# random flow stays a few pixels.
RANDOM = [("4.25", 1.0, (320, 180)), ("4.25.lite", 0.5, (320, 180))]


def read_frames(want: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(CLIP))
    got, i = {}, 0
    while len(got) < len(want):
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"{CLIP} ended at frame {i}")
        if i in want:
            got[i] = f
        i += 1
    return got


def to_pq16(bgr: np.ndarray) -> np.ndarray:
    """An SDR frame as an HDR10 one (framegen/hdr.py's test-stream look), HxWx3 float16 BGR PQ."""
    nits = hdr.expand_sdr(torch.from_numpy(bgr))
    pq = hdr.pq_oetf(nits.clamp_min(0.0) / hdr.PQ_MAX_NITS)
    return pq.flip(-1).to(torch.float16).numpy()


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(peak**2 / mse)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("needs Apple's GPU (MPS)")
    OUT.mkdir(parents=True, exist_ok=True)
    want = {i for _, a, b in PAIRS for i in range(a, b + 1)}
    frames = read_frames(want)
    h, w = frames[PAIRS[0][1]].shape[:2]
    files: dict[tuple[int, bool], str] = {}

    def frame_file(i: int, is_hdr: bool) -> str:
        key = (i, is_hdr)
        if key not in files:
            name = f"f{i:04d}.bgr16" if is_hdr else f"f{i:04d}.bgr"
            data = to_pq16(frames[i]) if is_hdr else frames[i]
            (OUT / name).write_bytes(np.ascontiguousarray(data).tobytes())
            files[key] = name
        return files[key]

    pairs = {name: (a, b) for name, a, b in PAIRS}
    cases = []
    for pair, scale, ts, steady, is_hdr in CASES:
        a, b = pairs[pair]
        it = RifeInterpolator(w, h, MODEL, scale, fp16=True, device="mps", hdr=is_hdr)
        assert it.metal is not None, "the Python Mac engine didn't start"
        if steady:
            it.set_steady(STEADY)
        for slot, i in ((0, a), (1, b)):
            f = to_pq16(frames[i]) if is_hdr else frames[i]
            it.load(torch.from_numpy(np.ascontiguousarray(f)).to("mps"))
            it.prepare(slot)
        steady_share = float((it.hud.w > 0).float().mean().item()) if steady else 0.0
        for t in ts:
            out = it.interpolate(1, t).cpu().numpy()
            torch.mps.synchronize()
            truth_i = a + round(t * (b - a))
            tag = f"{pair}_s{scale:g}_t{round(t * 1000):03d}{'_steady' if steady else ''}{'_hdr' if is_hdr else ''}"
            name = f"py_{tag}.bgr16" if is_hdr else f"py_{tag}.bgr"
            (OUT / name).write_bytes(np.ascontiguousarray(out).tobytes())
            if is_hdr:  # scored on 0..1 PQ values, peak 1
                truth = to_pq16(frames[truth_i]).astype(np.float32)
                score = psnr(out.astype(np.float32), truth, 1.0)
            else:
                score = psnr(out, frames[truth_i], 255.0)
            cases.append({
                "name": tag, "a": frame_file(a, is_hdr), "b": frame_file(b, is_hdr),
                "truth": frame_file(truth_i, is_hdr), "python": name, "scale": scale, "t": t,
                "steady": STEADY if steady else 0.0, "hdr": is_hdr, "psnr_python_vs_truth": score,
                "steady_share": steady_share,
            })
            print(f"{tag}: Python vs real frame {score:.3f} dB"
                  + (f" (Steady HUD acts on {steady_share * 100:.2f}% of pixels)" if steady else ""))
        del it
    # Five-block models on random weights, straight through MetalIFNet (RifeInterpolator would download weights).
    a, b = pairs["p120"]
    for version, scale, (cw, ch) in RANDOM:
        torch.manual_seed(0)
        mod = importlib.import_module(f"framegen.rife.{MODELS[version][0]}")
        net = mod.IFNet(scale=scale)
        for i in range(len(net.scale_list)):
            getattr(net, f"block{i}").lastconv[0].weight.data.mul_(0.1)
        rdir = OUT / f"random_{version}"
        rdir.mkdir(exist_ok=True)
        torch.save({f"module.{k}": v.clone() for k, v in net.state_dict().items()}, rdir / f"flownet_v{version}.pkl")
        net = net.eval().to("mps", torch.half)
        modulo = MODELS[version][1]
        step = max(modulo, int(modulo / scale))
        pw, ph = math.ceil(cw / step) * step, math.ceil(ch / step) * step
        mt = metalrife.MetalIFNet(net, net.encode, ch, cw, ph, pw, torch.device("mps"))
        crops = {}
        for i in (a, b):
            name = f"c{i:04d}_{cw}x{ch}.bgr"
            crop = np.ascontiguousarray(frames[i][300:300 + ch, 800:800 + cw])
            (OUT / name).write_bytes(crop.tobytes())
            crops[i] = (name, crop)
        s0, s1 = mt.new_slot(), mt.new_slot()
        with torch.no_grad():
            mt.prepare(torch.from_numpy(crops[a][1]).to("mps"), s0)
            mt.prepare(torch.from_numpy(crops[b][1]).to("mps"), s1)
            out = torch.zeros((ch, cw, 3), dtype=torch.uint8, device="mps")
            mt.interpolate(s0, s1, 0.5, out)
        torch.mps.synchronize()
        tag = f"random_{version}_s{scale:g}_t500"
        (OUT / f"py_{tag}.bgr").write_bytes(out.cpu().numpy().tobytes())
        cases.append({
            "name": tag, "a": crops[a][0], "b": crops[b][0], "truth": None, "python": f"py_{tag}.bgr",
            "scale": scale, "t": 0.5, "steady": 0.0, "hdr": False, "psnr_python_vs_truth": None,
            "version": version, "model_dir": rdir.name, "width": cw, "height": ch,
        })
        print(f"{tag}: written")
    index = {"model": MODEL, "width": w, "height": h, "cases": cases}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
