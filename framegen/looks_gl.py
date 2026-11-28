"""Draws a look (framegen/looks.py) with OpenGL, after upscaling, in the video window's last step.

looks.stages() splits the look into passes. Each pass is one fragment shader made for that run of nodes, so a run
of per-pixel nodes (color, wheels, grain, vignette...) costs one full-screen pass however many there are; a node
that reads other pixels (blur, bloom, sharpen...) starts a new pass, which reads the pass before it from a
half-float texture of the output's size. Bloom, halation, blur and clarity also build a blurred copy of that
input: six halvings (a 4x4 box each, the first with bloom's bright pass), then back up, mixing each level with the
one below (looks.glow()). The first pass reads the picture (FSR's or VSR's result, or the stream's own frame,
which it resizes) and the last one writes to the window, upright, or to a texture for instant replay ("Looks in
clips", at the stream's size).

Programs are made once per look's layout (adding, removing or switching off a node remakes them, in a few ms);
setting changes only change uniforms. Everything here is held to looks.render(), the NumPy reference, by
tests/test_looks.py.
"""
from __future__ import annotations

from . import looks

_VERT = """#version 450
void main() {  // one triangle covering the viewport
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Every pass: where this pixel is in the output image (Q: pixel center, row 0 at the top; UV: 0..1), then the nodes.
_HEAD = """#version 450
layout(binding = 0) uniform sampler2D src;   // the pass's input
layout(binding = 1) uniform sampler2D glow;  // its blurred copy (bloom, halation, blur, clarity)
uniform vec2 out_size;  // the output image, pixels
uniform ivec2 origin;   // the framebuffer pixel of its first column and its bottom row
uniform int flip_out;   // 1: the framebuffer's row 0 is the image's bottom (the window); 0: its top (a texture)
uniform int flip_in;    // 1: the input's row 0 is the image's bottom (FSR's result)
uniform uint frame_no;  // moves the grain
out vec4 color;
const vec3 W = vec3(0.2126, 0.7152, 0.0722);
vec2 Q;
vec2 UV;

uint pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float hash3(uint x, uint y, uint z) { return float(pcg(x + pcg(y + pcg(z)))) * (1.0 / 4294967296.0); }
float vnoise(uint x, uint y, vec2 f, uint z) {
    float a = hash3(x, y, z), b = hash3(x + 1u, y, z), c = hash3(x, y + 1u, z), d = hash3(x + 1u, y + 1u, z);
    float top = a * (1.0 - f.x) + b * f.x;
    float bot = c * (1.0 - f.x) + d * f.x;
    return (top * (1.0 - f.y) + bot * f.y - 0.5) * 2.0;
}

vec3 n_color(vec3 c, vec3 wb, float gain, float k, float sh, float hi, float fade, float sat, float vib) {
    c = clamp(c * wb, 0.0, 1.0);
    c = clamp(c * gain, 0.0, 1.0);
    vec3 s = c * c * (3.0 - 2.0 * c);
    c = c + k * (s - c);
    vec3 x1 = 1.0 - c;
    c = c + sh * c * x1 * x1 + hi * c * c * x1;
    x1 = 1.0 - c;
    c = c + fade * x1 * x1;
    float y = dot(c, W);
    float spread = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    float f = sat * max(0.0, 1.0 + vib * (1.0 - spread));
    return y + (c - y) * f;
}
vec3 n_wheels(vec3 c, vec3 lift, vec3 gain, vec3 power) {
    return pow(max(gain * (c + lift * (1.0 - c)), 0.0), power);
}
vec3 n_split(vec3 c, vec3 sh, vec3 hi, float bal) {
    float y = dot(c, W);
    float t = clamp(y + 0.5 * bal, 0.0, 1.0);
    float wh = t * t * (3.0 - 2.0 * t);
    return c + (1.0 - wh) * smoothstep(0.0, 0.25, y) * sh + wh * (1.0 - smoothstep(0.9, 1.0, y)) * hi;
}
vec3 n_mono(vec3 c, vec3 w, vec3 tone) {
    float y = clamp(dot(c, w), 0.0, 1.0);
    return vec3(y) + tone * (4.0 * y * (1.0 - y));
}
vec3 n_lut(vec3 c, sampler3D t, vec3 dmin, vec3 dscale, float n) {
    vec3 x = clamp((c - dmin) * dscale, 0.0, 1.0);
    return texture(t, (x * (n - 1.0) + 0.5) / n).rgb;
}
vec3 n_grain(vec3 c, float a, float size, float col) {
    float px = max(0.5, size * out_size.y / 1080.0);
    vec2 g = Q / px;
    vec2 fl = floor(g);
    vec2 f = g - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint x = uint(int(fl.x)), y = uint(int(fl.y)), z = frame_no * 4u;
    vec3 n3 = vec3(vnoise(x, y, f, z), vnoise(x, y, f, z + 1u), vnoise(x, y, f, z + 2u));
    float nm = vnoise(x, y, f, z + 3u);
    vec3 nn = nm * (1.0 - col) + n3 * col;
    float yl = clamp(dot(c, W), 0.0, 1.0);
    return c + a * nn * (0.25 + 3.0 * yl * (1.0 - yl));
}
vec3 n_vignette(vec3 c, float a, float r0, float soft, float rnd) {
    float ax = 1.0 + (out_size.x / out_size.y - 1.0) * rnd;
    vec2 d2 = (UV - 0.5) * 2.0 * vec2(ax, 1.0);
    float d = length(d2) / length(vec2(ax, 1.0));
    return c * (1.0 - a * smoothstep(r0 - soft, r0 + soft, d));
}
vec3 n_scanlines(vec3 c, float a, float size) {
    float period = max(2.0, size * out_size.y / 1080.0);  // (lines need 2 pixels or more: at 1 every row is alike)
    return c * (1.0 - a * (0.5 + 0.5 * cos(6.2831853 * fract(Q.y / period))));
}
vec3 n_letterbox(vec3 c, float aspect) {
    float e = out_size.x / out_size.y < aspect ? abs(Q.y - 0.5 * out_size.y) - 0.5 * out_size.x / aspect
                                               : abs(Q.x - 0.5 * out_size.x) - 0.5 * out_size.y * aspect;
    return c * (1.0 - clamp(e + 0.5, 0.0, 1.0));
}
vec3 n_assist(vec3 c, mat3 m) {
    return pow(clamp(m * pow(c, vec3(2.2)), 0.0, 1.0), vec3(1.0 / 2.2));
}

vec3 s_bloom(vec3 c, float amount, vec3 tint) { return c + amount * texture(glow, UV).rgb * tint; }
vec3 s_halation(vec3 c, float amount, vec3 tint) { return c + amount * texture(glow, UV).rgb * tint; }
vec3 s_blur(vec3 c) { return texture(glow, UV).rgb; }
vec3 s_clarity(vec3 c, float amount) { return c + amount * (c - texture(glow, UV).rgb); }
vec3 s_ca(vec3 c, float k) {
    vec2 off = (UV - 0.5) * k;
    return vec3(texture(src, UV + off).r, c.g, texture(src, UV - off).b);
}
vec3 s_sharpen(vec3 c, float amount) {
    ivec2 p = ivec2(Q), m = textureSize(src, 0) - 1;
    vec3 nb = texelFetch(src, clamp(p + ivec2(1, 0), ivec2(0), m), 0).rgb
            + texelFetch(src, clamp(p - ivec2(1, 0), ivec2(0), m), 0).rgb
            + texelFetch(src, clamp(p + ivec2(0, 1), ivec2(0), m), 0).rgb
            + texelFetch(src, clamp(p - ivec2(0, 1), ivec2(0), m), 0).rgb;
    return c + amount * (c - 0.25 * nb);
}
"""

_DOWN = """#version 450
layout(binding = 0) uniform sampler2D src;
uniform float thr;  // bloom's bright pass; below 0: none
out vec4 color;
vec3 bright(vec3 c) {
    float br = max(max(c.r, c.g), c.b);
    float knee = 0.1;
    float soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    return c * (max(soft, br - thr) / max(br, 1e-4));
}
void main() {  // a 4x4 box, centered on this pixel's 2x2 of the level above
    ivec2 p = ivec2(gl_FragCoord.xy) * 2 - 1, m = textureSize(src, 0) - 1;
    vec3 s = vec3(0.0);
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            vec3 c = texelFetch(src, clamp(p + ivec2(x, y), ivec2(0), m), 0).rgb;
            s += thr >= 0.0 ? bright(c) : c;
        }
    color = vec4(s / 16.0, 1.0);
}
"""

_UP = """#version 450
layout(binding = 0) uniform sampler2D level;
layout(binding = 1) uniform sampler2D below;  // the next smaller level, already mixed
uniform float r;
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(p) + 0.5) / vec2(textureSize(level, 0));
    color = vec4(mix(texelFetch(level, p, 0).rgb, texture(below, uv).rgb, r), 1.0);
}
"""


def _gl_type(v) -> str:
    if isinstance(v, (int, float)):
        return "float"
    return {3: "vec3", 9: "mat3"}[len(v)]


def stage_source(stage: looks.Stage, first: bool) -> str:
    """The fragment shader for one pass (uniforms u<i>_<name> for its i-th node, LUTs lut<j> from binding 2)."""
    decl, body = [], []
    luts = 0
    nodes = ([stage.start] if stage.start is not None else []) + list(stage.nodes)
    for i, n in enumerate(nodes):
        args = ["c"]
        if n["type"] == "lut":
            decl.append(f"layout(binding = {2 + luts}) uniform sampler3D lut{luts};")
            args.append(f"lut{luts}")
            luts += 1
        for name, v in looks.uniforms(n):
            decl.append(f"uniform {_gl_type(v)} u{i}_{name};")
            args.append(f"u{i}_{name}")
        decl.append(f"uniform float u{i}_mix;")
        fn = ("s_" if i == 0 and stage.start is not None else "n_") + n["type"]
        body.append(f"    {{ vec3 o = {fn}({', '.join(args)}); c = clamp(mix(c, o, u{i}_mix), 0.0, 1.0); }}")
    read = ("texture(src, flip_in == 1 ? vec2(UV.x, 1.0 - UV.y) : UV).rgb" if first
            else "texelFetch(src, ivec2(Q), 0).rgb")
    return (_HEAD + "\n".join(decl) + "\nvoid main() {\n"
            "    Q = gl_FragCoord.xy - vec2(origin);\n"
            "    if (flip_out == 1) Q.y = out_size.y - Q.y;\n"
            "    UV = Q / out_size;\n"
            f"    vec3 c = {read};\n" + "\n".join(body) + "\n    color = vec4(c, 1.0);\n}\n")


class _Tex:
    """A 2D texture with its framebuffer."""

    def __init__(self, GL, w: int, h: int, fmt) -> None:
        self.GL, self.w, self.h = GL, w, h
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)

    def close(self) -> None:
        self.GL.glDeleteFramebuffers(1, [self.fbo])
        self.GL.glDeleteTextures([self.tex])


class _Targets:
    """What a look needs at one output size: two half-float images to pass between passes, and the blur levels."""

    def __init__(self, GL, w: int, h: int) -> None:
        self.GL, self.size = GL, (w, h)
        self.ping = [_Tex(GL, w, h, GL.GL_RGBA16F) for _ in range(2)]
        self.down: list[_Tex] = []
        self.up: list[_Tex] = []

    def levels(self) -> tuple[list[_Tex], list[_Tex]]:
        if not self.down:
            sizes = looks.pyramid_sizes(*self.size)
            self.down = [_Tex(self.GL, w, h, self.GL.GL_RGBA16F) for w, h in sizes]
            self.up = [_Tex(self.GL, w, h, self.GL.GL_RGBA16F) for w, h in sizes[:-1]]
        return self.down, self.up

    def close(self) -> None:
        for t in self.ping + self.down + self.up:
            t.close()


class LookRenderer:
    """Needs a current GL 4.5 context (on a Mac, the 4.1 core one framegen/glsl.py asks for), the one the window
    draws with. lut_of(name) -> looks.Lut or None."""

    def __init__(self, GL, lut_of) -> None:
        self.GL, self.lut_of = GL, lut_of
        self.vao = GL.glGenVertexArrays(1)
        self._programs: dict[tuple, tuple[int, dict]] = {}
        self._down = self._compile(_DOWN)
        self._up = self._compile(_UP)
        self._u_thr = GL.glGetUniformLocation(self._down, "thr")
        self._u_r = GL.glGetUniformLocation(self._up, "r")
        self._targets: dict[str, _Targets] = {}
        self._lut_tex: dict[str, tuple[object, int]] = {}  # name -> (the Lut it was made from, 3D texture)
        self._plan_for: tuple | None = None  # (look id, rev) the plan below is for
        self._plan: list | None = None
        self._set: dict[int, int] = {}  # program -> the plan serial its node uniforms were last set for
        self._serial = 0
        self.broken: str | None = None  # why looks can't be drawn here (a shader didn't compile), or None

    def _compile(self, frag: str) -> int:
        from .glsl import program
        return program(self.GL, _VERT, frag)

    # ---- what to draw
    def plan(self, look: dict | None, rev: int):
        """The passes for this look (cached until `rev` changes), or None if it does nothing."""
        if look is None or self.broken:
            return None
        key = (id(look), rev)
        if key != self._plan_for:
            self._plan_for, self._plan = key, self._make_plan(look)
            self._serial += 1
        return self._plan

    def _make_plan(self, look: dict):
        passes = looks.stages(look, lambda name: self.lut_of(name) is not None)
        if not passes:
            return None
        plan = []
        for i, st in enumerate(passes):
            nodes = ([st.start] if st.start is not None else []) + list(st.nodes)
            layout = (i, st.start["type"] if st.start else None, tuple(n["type"] for n in st.nodes))
            try:
                prog, locs = self._program(layout, st, i == 0)
            except Exception as e:  # noqa: BLE001 - a driver that can't compile it: no looks, the video goes on
                self.broken = f"a shader didn't compile ({str(e)[:300]})"
                print(f"[looks] can't draw looks on this graphics card: {self.broken}")
                return None
            values, luts = [], []
            for k, n in enumerate(nodes):
                lut = self.lut_of(n["file"]) if n["type"] == "lut" else None
                if lut is not None:
                    luts.append(self._lut_texture(n["file"], lut))
                for name, v in looks.uniforms(n, lut):
                    values.append((locs.get(f"u{k}_{name}", -1), v))
                values.append((locs.get(f"u{k}_mix", -1), float(n["mix"])))
            plan.append((prog, locs, values, luts, st.start))
        return plan

    def _program(self, layout: tuple, st: looks.Stage, first: bool):
        got = self._programs.get(layout)
        if got is None:
            GL = self.GL
            prog = self._compile(stage_source(st, first))
            n_uniforms = GL.glGetProgramiv(prog, GL.GL_ACTIVE_UNIFORMS)
            locs = {}
            for j in range(n_uniforms):
                name = GL.glGetActiveUniform(prog, j)[0]
                name = name.decode() if isinstance(name, bytes) else str(name)
                locs[name] = GL.glGetUniformLocation(prog, name)
            got = self._programs[layout] = (prog, locs)
        return got

    def _lut_texture(self, name: str, lut) -> int:
        import numpy as np
        GL = self.GL
        got = self._lut_tex.get(name)
        if got is not None and got[0] is lut:
            return got[1]
        if got is not None:
            GL.glDeleteTextures([got[1]])
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                     (GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, p, v)
        n = lut.size
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA16F, n, n, n, 0, GL.GL_RGB, GL.GL_FLOAT,
                        np.ascontiguousarray(lut.table, np.float32))
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        self._lut_tex[name] = (lut, tex)
        return tex

    # ---- drawing
    def targets(self, purpose: str, w: int, h: int) -> _Targets:
        t = self._targets.get(purpose)
        if t is None or t.size != (w, h):
            if t is not None:
                t.close()
            t = self._targets[purpose] = _Targets(self.GL, w, h)
        return t

    def input_target(self, w: int, h: int, purpose: str = "screen") -> tuple[int, int]:
        """(texture, framebuffer) at the output's size for the upscaler to draw into; draw() then reads it."""
        t = self.targets(purpose, w, h).ping[0]
        return t.tex, t.fbo

    def draw(self, plan, src_tex: int, flip_in: bool, dst_fbo: int, x0: int, y0: int, w: int, h: int,
             flip_out: bool = True, frame: int = 0, purpose: str = "screen") -> None:
        """The look from src_tex (any size; flip_in: its row 0 is the image's bottom) into dst_fbo's rectangle at
        (x0, y0), w x h: upright on screen with flip_out, else image row 0 in framebuffer row 0. Leaves dst_fbo bound;
        the caller sets its viewport back."""
        GL = self.GL
        tg = self.targets(purpose, w, h)
        GL.glBindVertexArray(self.vao)
        cur = src_tex
        # the passes alternate between the two images; one that's being read is never written
        out_i = 1 if src_tex == tg.ping[0].tex else 0
        last = len(plan) - 1
        for i, (prog, locs, values, luts, start) in enumerate(plan):
            if start is not None and looks.TYPES[start["type"]].kind in ("glow", "blur", "clarity"):
                glow_tex = self._pyramid(tg, cur, *looks.glow(start))
            else:
                glow_tex = cur
            if i == last:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst_fbo)
                GL.glViewport(x0, y0, w, h)
                origin, flip = (x0, y0), flip_out
            else:
                out = tg.ping[out_i]
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, out.fbo)
                GL.glViewport(0, 0, w, h)
                origin, flip = (0, 0), False
            GL.glUseProgram(prog)
            if self._set.get(prog) != self._serial:
                for loc, v in values:
                    self._uniform(loc, v)
                self._set[prog] = self._serial
            if "out_size" in locs:
                GL.glUniform2f(locs["out_size"], float(w), float(h))
            if "origin" in locs:
                GL.glUniform2i(locs["origin"], *origin)
            if "flip_out" in locs:
                GL.glUniform1i(locs["flip_out"], int(flip))
            if "flip_in" in locs:
                GL.glUniform1i(locs["flip_in"], int(flip_in and i == 0))
            if "frame_no" in locs:
                GL.glUniform1ui(locs["frame_no"], frame & 0x3FFFFFFF)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, cur)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, glow_tex)
            for j, tex in enumerate(luts):
                GL.glActiveTexture(GL.GL_TEXTURE2 + j)
                GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            if i != last:
                cur = tg.ping[out_i].tex
                out_i ^= 1
        for j in range(max((len(p[3]) for p in plan), default=0)):
            GL.glActiveTexture(GL.GL_TEXTURE2 + j)
            GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)

    def _uniform(self, loc: int, v) -> None:
        if loc < 0:
            return
        GL = self.GL
        if isinstance(v, (int, float)):
            GL.glUniform1f(loc, float(v))
        elif len(v) == 3:
            GL.glUniform3f(loc, *(float(c) for c in v))
        else:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, [float(c) for c in v])

    def _pyramid(self, tg: _Targets, src: int, thr: float | None, r: float) -> int:
        """The blurred copy of src (output-sized): its first level, half the size, as a texture."""
        GL = self.GL
        down, up = tg.levels()
        if not down:
            return src
        GL.glUseProgram(self._down)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        cur = src
        for k, lv in enumerate(down):
            GL.glUniform1f(self._u_thr, thr if (k == 0 and thr is not None) else -1.0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, lv.fbo)
            GL.glViewport(0, 0, lv.w, lv.h)
            GL.glBindTexture(GL.GL_TEXTURE_2D, cur)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            cur = lv.tex
        GL.glUseProgram(self._up)
        GL.glUniform1f(self._u_r, r)
        below = down[-1].tex
        for k in range(len(up) - 1, -1, -1):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, up[k].fbo)
            GL.glViewport(0, 0, up[k].w, up[k].h)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, down[k].tex)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, below)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            below = up[k].tex
        GL.glActiveTexture(GL.GL_TEXTURE0)
        return below

    def close(self) -> None:
        GL = self.GL
        for t in self._targets.values():
            t.close()
        for _, tex in self._lut_tex.values():
            GL.glDeleteTextures([tex])
        for prog, _ in self._programs.values():
            GL.glDeleteProgram(prog)
        GL.glDeleteProgram(self._down)
        GL.glDeleteProgram(self._up)
        GL.glDeleteVertexArrays(1, [self.vao])
