"""The GPU PyTorch runs on here, NVIDIA's through CUDA, Apple's through Metal ("mps") on a Mac, or else the CPU.

Everything NVIDIA-specific (CUDA graphs, TensorRT, NVDEC, NVIDIA VFG and RTX VSR) stays on "cuda". On a Mac, RIFE
runs on "mps" with framegen/metalrife.py's kernels.
"""
from __future__ import annotations

import functools
import subprocess
import sys
import time

import torch

MAC = sys.platform == "darwin"


@functools.cache
def device() -> str:
    """The best PyTorch device here, "cuda", "mps" or "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if MAC and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize() -> None:
    """Wait for the GPU's queued work (a no-op on the CPU)."""
    kind = device()
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def empty_cache() -> None:
    """Hand cached GPU memory back, so the next interpolator has room."""
    kind = device()
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


@functools.cache
def name() -> str:
    """The graphics card's name from CUDA, or on a Mac the chip's (for example "Apple M5 Max"). "unknown" if it can't
    be told."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if MAC:
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=2).stdout.strip()
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def timer():
    """A GPU timer, returning (start, stop). stop() gives the milliseconds since start(). It uses CUDA events on
    NVIDIA. Elsewhere it reads the wall clock around a synchronize, since PyTorch can't time MPS's command buffers."""
    if device() == "cuda":
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def start() -> None:
            a.record()

        def stop() -> float:
            b.record()
            b.synchronize()
            return a.elapsed_time(b)
        return start, stop
    t0 = [0.0]

    def start() -> None:
        synchronize()
        t0[0] = time.perf_counter()

    def stop() -> float:
        synchronize()
        return (time.perf_counter() - t0[0]) * 1000.0
    return start, stop
