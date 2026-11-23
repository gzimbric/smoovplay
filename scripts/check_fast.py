"""Verify FastIFNet matches the reference vs-rife forward on real frame pairs, on this machine's GPU (CUDA, or Apple's
through Metal, framegen/gpu.py). On Apple's GPU it also checks MetalIFNet (framegen/metalrife.py), the engine the
Mac runs, against the same reference.

  python scripts/check_fast.py testdata/clip_1080p60.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from framegen import gpu  # noqa: E402
from framegen.interpolator import MODELS, RifeInterpolator  # noqa: E402
from framegen.pipeline import psnr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--pairs", type=int, default=12)
    a = ap.parse_args()

    dev = gpu.device()
    cap = cv2.VideoCapture(a.video)
    frames = []
    while len(frames) < a.pairs * 20 + 3:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    h, w = frames[0].shape[:2]
    starts = range(0, len(frames) - 2, max(2, (len(frames) - 2) // a.pairs))
    gt = [torch.from_numpy(frames[s + 1]).to(dev) for s in starts]
    # what's checked against the reference: (label, RifeInterpolator keywords)
    variants = [("fast", {"metal": False})]
    if dev == "mps":
        variants.append(("metal", {"metal": True}))
    print(f"{gpu.name()} ({dev})")

    ok_all = True
    for model in a.models:
        for scale in a.scales:
            outs = {}
            for label, kw in [("ref", {"reference": True})] + variants:
                it = RifeInterpolator(w, h, model, scale, cuda_graph=False, **kw)
                if label == "metal" and it.metal is None:
                    print(f"{model:<10} scale {scale}: MetalIFNet doesn't apply here, skipped")
                    continue
                res = []
                for s in starts:
                    it.load(torch.from_numpy(frames[s]).to(dev)); it.prepare(0)
                    it.load(torch.from_numpy(frames[s + 2]).to(dev)); it.prepare(1)
                    res.append(it.interpolate(1).clone())
                outs[label] = res
                del it
            q_ref = sum(psnr(o, g) for o, g in zip(outs["ref"], gt)) / len(gt)
            for label in outs:
                if label == "ref":
                    continue
                agree = [psnr(r, f) for r, f in zip(outs["ref"], outs[label])]
                maxdiff = max((r.int() - f.int()).abs().max().item() for r, f in zip(outs["ref"], outs[label]))
                q = sum(psnr(o, g) for o, g in zip(outs[label], gt)) / len(gt)
                good = min(agree) > 45
                ok_all &= good
                print(f"{model:<10} scale {scale} {label:>5}-vs-ref PSNR min {min(agree):6.2f} dB, max |diff| "
                      f"{maxdiff:3d}/255 | vs ground truth: ref {q_ref:.3f} dB, {label} {q:.3f} dB  "
                      f"{'OK' if good else 'MISMATCH'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
