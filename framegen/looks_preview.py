"""The launcher's Looks page preview: a look drawn on a sample game picture, the way the video window draws it.

`python -m framegen.looks_preview` runs while the Looks page is open. It reads one JSON request per line on stdin and
answers with one JSON line on stdout:

    -> {"id": 7, "look": {...}, "sample": "gta6-beach", "width": 640, "height": 360, "out": "C:\\...\\preview-1.bmp"}
    <- {"id": 7, "file": "C:\\...\\preview-1.bmp", "sample": "gta6-beach", "ms": 2.1}   (or {"id": 7, "error": "..."})

Its first line is {"ready": true, "gl": true|false, "samples": [...]}. It ends when stdin closes (the launcher went).
A request is drawn at the size the launcher shows it, so effects measured in pixels (grain, scanlines) look as they
will on screen.

The drawing is the video window's own (framegen/looks_gl.py) in a hidden OpenGL context: under a millisecond.
Without OpenGL it falls back to looks.render(), the NumPy reference: the same picture, about 0.1 to 0.2 s at 640x360.

The sample pictures (framegen/samples, listed in samples.json) are screenshots from games, credited in samples.json,
on the Looks page and in THIRD_PARTY_NOTICES.txt. They belong to their publishers and aren't covered by SmoovPlay's
MIT licence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import looks

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_SIZE = (1920, 1080)


def samples(folder: Path = SAMPLES_DIR) -> list[dict]:
    """The sample pictures: id, game, scene, credit, and path (absolute). Missing files are left out."""
    try:
        listed = json.loads((folder / "samples.json").read_text(encoding="utf-8"))["samples"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for s in listed:
        path = folder / s["file"]
        if path.is_file():
            out.append({"id": s["id"], "game": s["game"], "scene": s["scene"], "credit": s["credit"],
                        "path": str(path)})
    return out


class _Gl:
    """A hidden OpenGL context with the video window's look renderer, drawing into an offscreen picture."""

    def __init__(self, lut_of) -> None:
        import glfw
        from OpenGL import GL

        from .glsl import context_hints
        from .looks_gl import LookRenderer
        if not glfw.init():
            raise RuntimeError("no OpenGL")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        context_hints(glfw)
        self.win = glfw.create_window(64, 64, "SmoovPlay look preview", None, None)
        glfw.default_window_hints()
        if not self.win:
            glfw.terminate()
            raise RuntimeError("no OpenGL context")
        glfw.make_context_current(self.win)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        self.glfw, self.GL = glfw, GL
        self.r = LookRenderer(GL, lut_of)
        self.rev = 0
        self.src: dict[tuple, int] = {}  # (sample, w, h) -> texture
        self.dst: tuple | None = None  # (w, h, texture, framebuffer)

    def _texture(self, img: np.ndarray | None, w: int, h: int) -> int:
        GL = self.GL
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                        None if img is None else np.ascontiguousarray(img))
        return t

    def render(self, look: dict, img: np.ndarray, key: tuple, frame: int = 0) -> np.ndarray:
        GL = self.GL
        h, w = img.shape[:2]
        self.rev += 1
        plan = self.r.plan(look, self.rev)
        if plan is None:
            if self.r.broken:
                raise RuntimeError("the look shaders didn't build")
            return img
        if key not in self.src:
            if len(self.src) > 8:  # (a few sizes of a few pictures)
                GL.glDeleteTextures(list(self.src.values()))
                self.src.clear()
            self.src[key] = self._texture(img, w, h)
        if self.dst is None or self.dst[:2] != (w, h):
            if self.dst is not None:
                GL.glDeleteFramebuffers(1, [self.dst[3]])
                GL.glDeleteTextures([self.dst[2]])
            t = self._texture(None, w, h)
            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
            self.dst = (w, h, t, fbo)
        # Row 0 of the picture is the top, in and out (as tests/test_looks.py checks against the reference).
        self.r.draw(plan, self.src[key], False, self.dst[3], 0, 0, w, h, flip_out=False, frame=frame,
                    purpose="preview")
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.dst[3])
        data = GL.glReadPixels(0, 0, w, h, GL.GL_BGR, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()

    def close(self) -> None:
        try:
            self.r.close()
            self.glfw.destroy_window(self.win)
            self.glfw.terminate()
        except Exception:  # noqa: BLE001 - closing anyway
            pass


class Previewer:
    """Looks drawn on the sample pictures, at any size."""

    def __init__(self, library: looks.Library | None = None, use_gl: bool = True, folder: Path = SAMPLES_DIR):
        self.lib = library or looks.Library()
        self.samples = {s["id"]: s for s in samples(folder)}
        self.gl: _Gl | None = None
        if use_gl:
            try:
                self.gl = _Gl(self.lib.lut)
            except Exception as e:  # noqa: BLE001 - no OpenGL here: the NumPy path draws the same picture
                print(f"[looks] preview without OpenGL ({str(e)[:120]}): slower", file=sys.stderr)
        self._pictures: dict[tuple, np.ndarray] = {}

    def picture(self, sample: str, w: int, h: int) -> np.ndarray:
        import cv2

        from .imagefile import read_image
        key = (sample, w, h)
        if key not in self._pictures:
            if sample not in self.samples:
                raise ValueError(f"there's no sample picture called {sample!r}")
            img = read_image(self.samples[sample]["path"])
            if img is None:
                raise ValueError(f"couldn't read the sample picture {sample!r}")
            if len(self._pictures) > 16:
                self._pictures.clear()
            self._pictures[key] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return self._pictures[key]

    def render(self, look: dict, sample: str, w: int, h: int, frame: int = 0) -> np.ndarray:
        w = max(16, min(MAX_SIZE[0], int(w)))
        h = max(16, min(MAX_SIZE[1], int(h)))
        img = self.picture(sample, w, h)
        look = looks.normalize(look)
        if self.gl is not None:
            try:
                return self.gl.render(look, img, (sample, w, h), frame)
            except Exception as e:  # noqa: BLE001 - fall back for good
                print(f"[looks] OpenGL preview failed ({str(e)[:160]}): using the slower one", file=sys.stderr)
                self.gl.close()
                self.gl = None
        return looks.render(look, img, frame, self.lib)

    def close(self) -> None:
        if self.gl is not None:
            self.gl.close()
            self.gl = None


def serve(inp=None, out=None, previewer: Previewer | None = None) -> int:
    from .imagefile import write_image
    inp = inp or sys.stdin
    if out is None:
        out = sys.stdout
        sys.stdout = sys.stderr  # stdout carries the answers: anything else printed goes to stderr
    pv = previewer or Previewer()

    def say(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")  # (ASCII)
        out.flush()

    say({"ready": True, "gl": pv.gl is not None, "samples": list(pv.samples.values())})
    try:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            rid = None
            try:
                req = json.loads(line)
                rid = req.get("id")
                t = time.perf_counter()
                img = pv.render(req["look"], req["sample"], req.get("width", 640), req.get("height", 360),
                                int(req.get("frame", 0)))
                if not write_image(req["out"], img):
                    raise OSError(f"couldn't write {req['out']}")
                say({"id": rid, "file": req["out"], "sample": req["sample"],
                     "ms": round((time.perf_counter() - t) * 1000, 1)})
            except (ValueError, KeyError, TypeError, OSError) as e:
                say({"id": rid, "error": str(e)[:300]})
    finally:
        pv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
