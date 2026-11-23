"""GPU time of each built-in look (framegen/looks_gl.py), as the video window draws it: from a 1920x1080 frame onto a
2560x1440 window (FSR's result, the usual path), and at 1920x1080 for instant replay ("Looks in clips").

    python scripts/looks_timing.py [--frames 300]

Uses a hidden window and GL timer queries; prints the median and 95th percentile per look. The numbers in the Look
settings' "speed" lines (framegen/settings.py LOOK_SPEED) come from here. Run it with nothing else on the graphics
card.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import glfw
    from OpenGL import GL

    from framegen import glsl, looks
    from framegen.fsr import GpuTimer
    from framegen.looks_gl import LookRenderer

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    if not glfw.init():
        raise SystemExit("glfw init failed")
    glsl.context_hints(glfw)  # a 4.1 core context on a Mac
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "looks timing", None, None)
    glfw.make_context_current(win)

    def texture(w, h):
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        img = (np.random.default_rng(0).random((h, w, 4)) * 255).astype(np.uint8)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img)
        f = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
        return t, f

    src, _ = texture(1920, 1080)
    _, screen = texture(2560, 1440)
    _, clip = texture(1920, 1080)
    r = LookRenderer(GL, looks.Library().lut)
    timer = GpuTimer(GL, ring=8)
    print(f"{'look':28s} {'1440p window':>18s} {'1080p clip':>18s}   (GPU ms: median / 95th percentile)")
    for rev, (key, look) in enumerate(looks.BUILTIN.items()):
        plan = r.plan(look, rev)
        for name, (fbo, w, h, purpose) in (("screen", (screen, 2560, 1440, "screen")),
                                           ("clip", (clip, 1920, 1080, "bake"))):
            for i in range(a.frames + 20):
                q = timer.begin() if i >= 20 else None
                r.draw(plan, src, False, fbo, 0, 0, w, h, purpose == "screen", i, purpose)
                timer.end(q, f"{key}:{name}")
                if i % 30 == 0:
                    GL.glFinish()
        GL.glFinish()
        timer._collect()
        cells = []
        for name in ("screen", "clip"):
            v = timer.ms.get(f"{key}:{name}", [0])
            cells.append(f"{np.median(v):.2f} / {np.percentile(v, 95):.2f}")
        print(f"{look['name']:28s} {cells[0]:>18s} {cells[1]:>18s}")
    r.close()
    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
