"""Reference data for sp-present's FSR 1 and bilinear passes. It's what the Python window's OpenGL versions draw on
this Mac's OpenGL 4.1, which framegen/glsl.py rewrites the shaders for. Those versions are framegen/fsr.py FsrScaler
and GLWindowSink's glBlitFramebuffer with GL_LINEAR. crates/sp-present/tests/shaders.rs draws the same with Metal and
compares.

    .venv/bin/python mac/core/tools/present_fsr.py   (from the repository root)

Writes mac/core/fixtures/present/fsr/: <name>.in.bgr, <name>.<w>x<h>.<what>.bgr (top row first) and cases.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "mac" / "core" / "fixtures" / "present" / "fsr"


def _pattern(w: int, h: int) -> np.ndarray:
    """Edges, diagonals, fine detail and smooth gradients, which are what FSR's edge detection and RCAS react to."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.stack([128 + 120 * np.sin(xx / 9), 128 + 120 * np.sin(yy / 7 + 1), 128 + 120 * np.cos((xx + yy) / 13)], -1)
    a[(xx // 8 + yy // 8) % 2 == 0] *= 0.6  # a checkerboard
    r = np.hypot(xx - w * 0.7, yy - h * 0.4)
    a[r < h * 0.25] = 255 * ((np.sin(r[r < h * 0.25] * 1.3) > 0)[:, None])  # rings, circles that are prone to aliasing
    a[h // 10: h // 10 + 3, :] = 255  # thin lines
    a[:, w // 7: w // 7 + 1] = 0
    return np.clip(a, 0, 255).astype(np.uint8)


def _clip_frame() -> np.ndarray | None:
    import cv2
    cap = cv2.VideoCapture(str(ROOT / "testdata" / "clip_1080p60.mp4"))
    for _ in range(30):  # a frame from a little way in
        ok, f = cap.read()
        if not ok:
            return None
    cap.release()
    return f


def main() -> None:
    import glfw
    from OpenGL import GL

    from framegen.fsr import FsrScaler
    from framegen.glsl import context_hints

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.bgr"):
        old.unlink()
    if not glfw.init():
        raise SystemExit("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    context_hints(glfw)
    win = glfw.create_window(64, 64, "fsr reference", None, None)
    glfw.make_context_current(win)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    fsr = FsrScaler(GL, 0.5)

    def tex(img):
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        h, w = img.shape[:2]
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, img)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
        return t, fbo

    def target(w, h):
        return tex(np.zeros((h, w, 3), np.uint8))

    def read(fbo, w, h):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
        px = GL.glReadPixels(0, 0, w, h, GL.GL_BGR, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(px, np.uint8).reshape(h, w, 3)[::-1].copy()  # GL row 0 is the bottom

    inputs = {"pattern": _pattern(320, 180)}
    clip = _clip_frame()
    if clip is not None:
        inputs["clip"] = clip
    plan = {"pattern": [(640, 360, (0.0, 0.5, 2.0)), (517, 293, (0.5,)), (1200, 675, (0.25,))],
            "clip": [(2560, 1440, (0.5,)), (3840, 2160, (0.0,))]}
    cases = []
    for name, img in inputs.items():
        h, w = img.shape[:2]
        (OUT / f"{name}.in.bgr").write_bytes(np.ascontiguousarray(img).tobytes())
        src, src_fbo = tex(img)
        for dw, dh, sharps in plan[name]:
            out_t, out_fbo = target(dw, dh)
            for s in sharps:
                fsr.sharpness = s
                fsr.draw(src, w, h, out_fbo, 0, 0, dw, dh)
                file = f"{name}.{dw}x{dh}.fsr{s:g}.bgr"
                (OUT / file).write_bytes(read(out_fbo, dw, dh).tobytes())
                cases.append({"input": f"{name}.in.bgr", "w": w, "h": h, "dw": dw, "dh": dh, "what": "fsr",
                              "sharpness": s, "file": file})
            # bilinear is the window's plain blit. Image row 0 is at the top, so the destination's Y is flipped.
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, out_fbo)
            GL.glBlitFramebuffer(0, 0, w, h, 0, dh, dw, 0, GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
            file = f"{name}.{dw}x{dh}.bilinear.bgr"
            (OUT / file).write_bytes(read(out_fbo, dw, dh).tobytes())
            cases.append({"input": f"{name}.in.bgr", "w": w, "h": h, "dw": dw, "dh": dh, "what": "bilinear",
                          "file": file})
            GL.glDeleteFramebuffers(1, [out_fbo])
            GL.glDeleteTextures([out_t])
    (OUT / "cases.json").write_text(json.dumps({"renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                                                "cases": cases}, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases -> {OUT} ({GL.glGetString(GL.GL_RENDERER).decode()})")
    fsr.close()
    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
