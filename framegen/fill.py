"""Filling a screen that isn't the picture's shape (Settings: Picture, Fill the screen; key Z in the video window).

Consoles only send a 16:9 picture: a PS5 or Xbox Series can't put out anything wider, to a TV or to a capture card,
and Remote Play streams what it puts out. So on an ultrawide (21:9, 32:9) or a 16:10 screen there is no more of the
game to show, and none of these is real ultrawide. They're different ways of laying out the same 16:9 picture:

- fit: all of it, true to shape, with black bars at the sides (or top and bottom). The default.
- zoom: fills the screen by cutting off the top and bottom (or the sides), which can cut off the HUD.
- stretch: fills the screen by making everything wider (or taller).
- panorama: screens wider than the picture only. The middle stays close to its true shape and the stretch grows
  towards the sides, like a TV's wide zoom: usually easier on the eye than stretch. One extra pass over the screen
  (Panorama), after the upscaler and the look.

layout() and visible() are the arithmetic (tested without a GPU); Panorama is the GL pass.
"""
from __future__ import annotations

FILLS = ("fit", "zoom", "stretch", "panorama")
PANORAMA_MIDDLE = 0.15  # Panorama: the middle is drawn this share of the extra width wider than true (5% on a 21:9
# screen, 15% on a 32:9); the sides take the rest, more towards the edges (about 2x on 21:9, 5x on 32:9)


def wider(fw: int, fh: int, w: int, h: int) -> bool:
    """Whether the screen is wider than the picture (by more than a pixel or so)."""
    return fw * h > (fh + 1) * w


def layout(fill: str, fw: int, fh: int, w: int, h: int) -> tuple[int, int, int, int]:
    """The picture's rectangle (dw, dh, x0, y0) on an fw x fh screen, GL style (x0, y0 the bottom left). Zoom's goes
    past the screen's edges. Panorama's is fit's: that's drawn first, then bent across the screen (Panorama)."""
    if fill == "stretch":
        return fw, fh, 0, 0
    s = max(fw / w, fh / h) if fill == "zoom" else min(fw / w, fh / h)
    dw, dh = int(round(w * s)), int(round(h * s))
    if fill != "zoom":
        dw, dh = min(dw, fw), min(dh, fh)
    return dw, dh, (fw - dw) // 2, (fh - dh) // 2


def visible(dw: int, dh: int, x0: int, y0: int, fw: int, fh: int) -> tuple[int, int, int, int]:
    """The part of the picture's rectangle that's on the screen (x, y, w, h): where the menu and messages go."""
    x, y = max(0, x0), max(0, y0)
    return x, y, min(fw, x0 + dw) - x, min(fh, y0 + dh) - y


def panorama_t(fw: int, fh: int, w: int, h: int) -> float:
    """Panorama's curve: u = atan(t*x) / atan(t) maps the screen (x from -1 to 1) onto the picture (u from -1 to 1).
    Drawn across a screen k times as wide as the picture at the same height, the picture is k / u'(x) times wider than
    true at x: k*atan(t)/t in the middle, growing as 1 + t*t*x*x towards the sides. t is picked so the middle is
    1 + PANORAMA_MIDDLE*(k - 1); 0 (no bending) when the screen isn't wider."""
    import math

    k = (fw / fh) / (w / h)
    if k <= 1.0:
        return 0.0
    goal = (1.0 + PANORAMA_MIDDLE * (k - 1.0)) / k  # atan(t)/t, which falls from 1 (t = 0) towards 0
    lo, hi = 1e-6, 1e3
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if math.atan(mid) / mid > goal else (lo, mid)
    return (lo + hi) / 2


def panorama_u(x: float, t: float) -> float:
    """Where on the picture (-1 to 1) a point across the screen (-1 to 1) comes from."""
    import math

    return x if t <= 0.0 else math.atan(t * x) / math.atan(t)


_FRAG = """#version 450
layout(binding = 0) uniform sampler2D src;  // the picture, fit to the screen's height, as on screen
uniform vec2 screen;                         // pixels
uniform float t;  // panorama_t
out vec4 color;
void main() {
    vec2 p = gl_FragCoord.xy / screen;
    float x = p.x * 2.0 - 1.0;
    float u = atan(t * x) / atan(t);
    color = texture(src, vec2(u * 0.5 + 0.5, p.y));
}
"""


class Panorama:
    """The picture is drawn as usual, fit to the screen's height, into this one's texture (RGBA16F, so HDR keeps its
    precision), then bent across the whole screen. Needs a current GL 4.5 context (on a Mac, the 4.1 core one
    framegen/glsl.py asks for)."""

    def __init__(self, GL) -> None:
        from .fsr import _VERT
        from .glsl import program
        self.GL = GL
        self.prog = program(GL, _VERT, _FRAG)
        self.u_screen = GL.glGetUniformLocation(self.prog, "screen")
        self.u_t = GL.glGetUniformLocation(self.prog, "t")
        self.vao = GL.glGenVertexArrays(1)
        self.tex = self.fbo = None
        self.size = (0, 0)

    def target(self, w: int, h: int) -> int:
        """The framebuffer to draw the picture into (w x h at 0, 0), re-made when the size changes."""
        GL = self.GL
        if self.size != (w, h):
            self.free()
            self.tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
            for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                         (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None)
            self.fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)
            self.size = (w, h)
        return self.fbo

    def draw(self, dst_fbo: int, fw: int, fh: int, t: float) -> None:
        """The picture in target(), across all of dst_fbo (fw x fh). Leaves dst_fbo bound."""
        GL = self.GL
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst_fbo)
        GL.glViewport(0, 0, fw, fh)
        GL.glBindVertexArray(self.vao)
        GL.glUseProgram(self.prog)
        GL.glUniform2f(self.u_screen, float(fw), float(fh))
        GL.glUniform1f(self.u_t, float(t))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def free(self) -> None:
        GL = self.GL
        if self.fbo is not None:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteTextures([self.tex])
        self.tex = self.fbo = None
        self.size = (0, 0)
