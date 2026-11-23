"""MetalIFNet (framegen/metalrife.py), the Mac's RIFE engine, must draw what eager FastIFNet draws on Apple's GPU.
That goes for RIFE 4.22's four blocks and the five of 4.25 and 4.26, at each motion detail it takes, with Steady HUD,
HDR frames and timesteps other than the midpoint. Needs Apple's GPU (MPS) and is skipped elsewhere."""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    pytest.skip("needs Apple's GPU (MPS)", allow_module_level=True)

from framegen import metalrife  # noqa: E402
from framegen.fastrife import FastIFNet  # noqa: E402
from framegen.interpolator import DEFAULT_MODEL_DIR, MODELS, RifeInterpolator  # noqa: E402

DEV = torch.device("mps")
MODEL = "4.22.lite"
needs_weights = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / f"flownet_v{MODEL}.pkl").exists(),
                                   reason="RIFE weights not downloaded")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def two_frames(w: int, h: int, shift: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """A smooth coloured pattern and the same pattern moved `shift` pixels right, HxWx3 uint8 BGR on the GPU."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def at(x):
        return np.stack([np.sin(x / 9) * 70 + np.cos(yy / 7) * 50 + 128,
                         np.cos(x / 13) * 80 + 128 + 0 * yy,
                         np.sin((x + yy) / 11) * 90 + 128], -1).clip(0, 255).astype(np.uint8)

    return torch.from_numpy(at(xx)).to(DEV), torch.from_numpy(at(xx - shift)).to(DEV)


def eager_frame(fast: FastIFNet, encode, frames, pad, h: int, w: int, t: float) -> torch.Tensor:
    """FastIFNet's forward as RifeInterpolator runs it eagerly, as HxWx3 uint8 BGR."""
    src = []
    for f in frames:
        x = F.pad(f.permute(2, 0, 1).flip(0).unsqueeze(0).half() / 255.0, pad)
        src.append(FastIFNet.make_src(x, encode(x)))
    fast.set_timestep(t)
    mid = fast(*src)[0, :, :h, :w].flip(0).permute(1, 2, 0)
    return mid.clamp(0, 1).mul(255).round().to(torch.uint8)


@pytest.mark.parametrize("version,scale", [("4.22.lite", 1.0), ("4.22.lite", 0.5), ("4.22.lite", 0.25),
                                           ("4.25", 1.0), ("4.25.lite", 0.5), ("4.26", 1.0)])
@torch.no_grad()
def test_it_draws_what_eager_fastifnet_draws(version, scale):
    """Random weights (so no download is needed), small enough that the flow stays a few pixels."""
    torch.manual_seed(0)
    mod = importlib.import_module(f"framegen.rife.{MODELS[version][0]}")
    net, encode = mod.IFNet(scale=scale), mod.Head()
    for i in range(len(net.scale_list)):  # tame the random flow
        getattr(net, f"block{i}").lastconv[0].weight.mul_(0.1)
    net, encode = net.eval().to(DEV, torch.half), encode.eval().to(DEV, torch.half)
    w, h = 200, 120  # not a multiple of the padding
    step = max(MODELS[version][1], int(MODELS[version][1] / scale))
    pw, ph = math.ceil(w / step) * step, math.ceil(h / step) * step
    a, b = two_frames(w, h)
    for t in (0.5, 0.25):
        want = eager_frame(FastIFNet(net, ph, pw, torch.half, DEV), encode, (a, b), (0, pw - w, 0, ph - h), h, w, t)
        mt = metalrife.MetalIFNet(net, encode, h, w, ph, pw, DEV)
        s0, s1 = mt.new_slot(), mt.new_slot()
        mt.prepare(a, s0)
        mt.prepare(b, s1)
        got = torch.zeros_like(a)
        mt.interpolate(s0, s1, t, got)
        assert psnr(got, want) > 45, (version, scale, t, psnr(got, want))


@needs_weights
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_rife_interpolator_uses_it_on_mps(scale):
    w, h = 320, 180
    a, b = two_frames(w, h)
    outs, rgb = {}, {}
    for metal in (False, True):
        it = RifeInterpolator(w, h, MODEL, scale, metal=metal)
        assert (it.metal is not None) == metal
        it.load(a)
        it.prepare(0)
        it.load(b)
        it.prepare(1)
        outs[metal] = [it.interpolate(1, t).clone() for t in (0.5, 1 / 3)]
        it.set_steady(0.02)  # Steady HUD takes the metal engine's fp32 path, with the real frames from its slots
        outs[metal].append(it.interpolate(1, 0.5).clone())
        rgb[metal] = it._rgb(0)
    for e, m in zip(outs[False], outs[True]):
        assert psnr(m, e) > 45
    assert rgb[True].shape == rgb[False].shape and (rgb[True] - rgb[False]).abs().max().item() < 1e-3


@needs_weights
def test_load_is_safe_to_overwrite_on_mps():
    """The pipeline reuses one CPU buffer for every frame, and even timing loads two in a row. What load() was given
    must be what prepare() encodes, even if the buffer is written again before the GPU gets to it."""
    w, h = 320, 180
    a, b = (f.cpu() for f in two_frames(w, h))
    it = RifeInterpolator(w, h, MODEL, 1.0)
    buf = a.clone()
    big = torch.rand((2048, 2048), device=DEV, dtype=torch.half)
    for _ in range(8):  # keep the GPU busy, so an asynchronous copy would still be waiting
        big = big @ big * 1e-3
    it.load(buf)
    buf.copy_(b)  # CPU only, so nothing here waits for the GPU
    assert torch.equal(it.inp.cpu(), a)


@needs_weights
def test_hdr_frames():
    w, h = 320, 180
    a, b = two_frames(w, h)
    outs = {}
    for metal in (False, True):
        it = RifeInterpolator(w, h, MODEL, 1.0, metal=metal, hdr=True)
        it.load(a.half() / 255)
        it.prepare(0)
        it.load(b.half() / 255)
        it.prepare(1)
        out = it.interpolate(1)
        assert out.dtype == torch.float16
        outs[metal] = out.float().mul(255).clone()
    assert psnr(outs[True], outs[False]) > 45


def test_whole_number_scales_only():
    assert metalrife.supported([8, 4, 2, 1]) and metalrife.supported([32.0, 16.0, 8.0, 4.0, 1.0])
    assert not metalrife.supported([4, 2, 1, 0.5])  # motion detail 2.0 runs eager FastIFNet


@needs_weights
def test_falls_back_to_eager_when_it_cant_run(monkeypatch, capsys):
    def broken():
        raise RuntimeError("no Metal compiler")
    monkeypatch.setattr(metalrife, "library", broken)
    it = RifeInterpolator(320, 180, MODEL, 1.0)
    assert it.metal is None and "running eagerly" in capsys.readouterr().out
    a, b = two_frames(320, 180)
    it.load(a)
    it.prepare(0)
    it.load(b)
    it.prepare(1)
    assert it.interpolate(1).shape == (180, 320, 3)
    assert RifeInterpolator(320, 180, MODEL, 2.0).metal is None  # motion detail 2.0 has half-pixel scales
