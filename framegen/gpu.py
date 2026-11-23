"""Which GPU PyTorch runs on: NVIDIA's CUDA, Apple's GPU through Metal (MPS) on a Mac, or the CPU.

Everything NVIDIA-specific (CUDA graphs, TensorRT, NVDEC, CUDA-OpenGL interop, NVIDIA VFG and RTX VSR) stays on
"cuda". On a Mac, RIFE runs eagerly on "mps". Frames reach the window through the CPU, which is cheap because the
memory is shared with the GPU, and VideoToolbox decodes the stream.
"""
from __future__ import annotations

import functools
import subprocess
import sys

import torch

MAC = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"


@functools.cache
def device() -> str:
    """The best PyTorch device here: "cuda", "mps" or "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if MAC and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_mps() -> bool:
    return device() == "mps"


def synchronize(dev: str | torch.device | None = None) -> None:
    """Wait for the GPU's queued work (a no-op on the CPU)."""
    kind = torch.device(dev).type if dev is not None else device()
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def empty_cache() -> None:
    """Hand cached GPU memory back (between sessions, so a rebuilt engine has room)."""
    kind = device()
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


@functools.cache
def name() -> str:
    """The graphics chip's name from CUDA, or on a Mac the chip's (for example "Apple M5 Max"). "unknown" if it can't
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
    """A GPU timer. It uses CUDA events on NVIDIA, else wall time around a synchronize, because MPS's command buffers
    can't be timed from PyTorch. Returns (start, stop), and stop() returns the milliseconds since start()."""
    if device() == "cuda":
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def start():
            a.record()

        def stop() -> float:
            b.record()
            b.synchronize()
            return a.elapsed_time(b)
        return start, stop
    import time
    t0 = [0.0]

    def start():
        synchronize()
        t0[0] = time.perf_counter()

    def stop() -> float:
        synchronize()
        return (time.perf_counter() - t0[0]) * 1000.0
    return start, stop
