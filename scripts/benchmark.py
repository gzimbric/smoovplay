"""GPU timing per stage for each RIFE model / resolution (synthetic frames, no decode).

On NVIDIA each setting runs with and without CUDA graphs. On Apple's GPU (Metal) it runs eagerly ("eager",
FastIFNet) and with the Metal kernels ("metal", framegen/metalrife.py), which is what the Mac runs.

  python scripts/benchmark.py
  python scripts/benchmark.py --models 4.25 4.22.lite --res 1080p 4k --scales 1.0 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from framegen import gpu  # noqa: E402
from framegen.interpolator import MODELS, RifeInterpolator  # noqa: E402

RES = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}


def gpu_ms(fn, iters: int) -> float:
    start, stop = gpu.timer()
    gpu.synchronize()
    start()
    for _ in range(iters):
        fn()
    return stop() / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--res", nargs="+", default=["1080p", "4k"], choices=list(RES))
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.5])
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()

    dev = gpu.device()
    # (label, RifeInterpolator keywords) per setting
    variants = ([("False", {"cuda_graph": False}), ("True", {"cuda_graph": True})] if dev == "cuda" else
                [("eager", {"metal": False}), ("metal", {"metal": True})] if dev == "mps" else
                [("eager", {})])
    print(f"{gpu.name()}  torch {torch.__version__}  ({dev})")
    print(f"{'model':<10} {'res':<6} {'scale':>5} {'graph' if dev == 'cuda' else 'run':>5} | {'prep ms':>7} "
          f"{'interp ms':>9} {'new-frame tick':>14} {'wall/pair':>9}")
    for res in a.res:
        w, h = RES[res]
        frame = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device=dev)
        for model in a.models:
            for scale in a.scales:
                for label, kw in variants:
                    it = RifeInterpolator(w, h, model, scale, **kw)
                    if label == "metal" and it.metal is None:
                        del it
                        continue  # MetalIFNet doesn't apply at this scale
                    it.load(frame)
                    it.prepare(0)
                    for _ in range(5):  # warm-up
                        it.prepare(1)
                        it.interpolate(1)
                    prep = gpu_ms(lambda: it.prepare(1), a.iters)
                    interp = gpu_ms(lambda: it.interpolate(1), a.iters)
                    gpu.synchronize()
                    t = time.perf_counter()
                    for _ in range(a.iters):
                        it.prepare(1)
                        it.interpolate(1)
                    gpu.synchronize()
                    wall = (time.perf_counter() - t) * 1000 / a.iters
                    print(f"{model:<10} {res:<6} {scale:>5} {label:>5} | {prep:7.2f} {interp:9.2f} "
                          f"{prep + interp:14.2f} {wall:9.2f}")
                    del it
                    gpu.empty_cache()


if __name__ == "__main__":
    main()
