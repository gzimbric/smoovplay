"""The OpenGL context and GLSL for the video window's shaders (framegen/fsr.py, fill.py, looks_gl.py) on each system.

The shaders are written for GLSL 4.50, as on Windows. macOS stops at OpenGL 4.1, core profile only, and only when a
program asks for it (otherwise it gets 2.1, with no vertex arrays). So on a Mac context_hints() asks GLFW for a 4.1
core context, and source() rewrites each shader for GLSL 4.10. It sets "#version 410 core", takes layout(binding = N)
off the samplers because that needs 4.20, and writes packHalf2x16 and unpackHalf2x16 out in GLSL. Those are 4.20 too,
and FSR's ffx_a.h uses them. program() then sets each sampler's texture unit with glUniform1i, as the binding did.
Elsewhere the shaders are compiled as they are.
"""
from __future__ import annotations

import re
import sys

# The GLSL version shaders are rewritten for here. 410 on a Mac, and None elsewhere, where they stay as written (450).
TARGET = 410 if sys.platform == "darwin" else None

_VERSION = re.compile(r"^[ \t]*#version[ \t]+\d+[ \t]*(core|compatibility)?[ \t]*$", re.M)
_BINDING = re.compile(r"layout\s*\(\s*binding\s*=\s*(\d+)\s*\)\s*uniform\s+(\w*sampler\w+)\s+(\w+)\s*;")

# GLSL 4.20's half-float packing, for 4.10. It rounds to nearest even and keeps infinities and NaNs.
_HALF_PACKING = """uint smoov_f32tof16(float v) {
    uint x = floatBitsToUint(v);
    uint s = (x >> 16) & 0x8000u, ex = (x >> 23) & 0xFFu, m = x & 0x7FFFFFu;
    if (ex == 0xFFu) return s | 0x7C00u | (m != 0u ? 0x200u : 0u);
    int e = int(ex) - 112;
    if (e >= 31) return s | 0x7C00u;
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000u;
        uint t = uint(14 - e);
        uint r = m >> t, rem = m & ((1u << t) - 1u), hf = 1u << (t - 1u);
        return s | (r + uint(rem > hf || (rem == hf && (r & 1u) != 0u)));
    }
    uint h = (uint(e) << 10) | (m >> 13), rem = m & 0x1FFFu;
    return s | (h + uint(rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)));
}
float smoov_f16tof32(uint h) {
    uint s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) return uintBitsToFloat(s | floatBitsToUint(float(m) * 5.9604645e-8));
    if (e == 31u) return uintBitsToFloat(s | 0x7F800000u | (m << 13));
    return uintBitsToFloat(s | ((e + 112u) << 23) | (m << 13));
}
uint smoov_packHalf2x16(vec2 v) { return smoov_f32tof16(v.x) | (smoov_f32tof16(v.y) << 16); }
vec2 smoov_unpackHalf2x16(uint u) { return vec2(smoov_f16tof32(u & 0xFFFFu), smoov_f16tof32(u >> 16)); }
#define packHalf2x16 smoov_packHalf2x16
#define unpackHalf2x16 smoov_unpackHalf2x16
"""


def context_hints(glfw) -> None:
    """GLFW window hints for a context these shaders compile in. Call before each glfw.create_window that draws
    with them (glfw.default_window_hints() clears them). On a Mac that's OpenGL 4.1 core, forward-compatible.
    Elsewhere there are none, so GLFW's default applies, the driver's newest compatibility context."""
    if sys.platform != "darwin":
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)


def source(src: str, target: int | None = TARGET) -> tuple[str, dict[str, int]]:
    """A GLSL 4.50 shader as `target` takes it, and the texture unit of each sampler whose layout(binding) was taken
    off ({name: unit}, for program() to set). With target None (not a Mac), the source as it is and {}."""
    if target is None or target >= 420:
        return src, {}
    units: dict[str, int] = {}

    def unbind(m: re.Match) -> str:
        units[m.group(3)] = int(m.group(1))
        return f"uniform {m.group(2)} {m.group(3)};"

    out = _BINDING.sub(unbind, src)
    head = f"#version {target} core" + ("\n" + _HALF_PACKING.rstrip("\n") if "packHalf2x16" in out else "")
    out, n = _VERSION.subn(lambda m: head, out, count=1)
    if n == 0:
        out = head + "\n" + out
    return out, units


def program(GL, vert: str, frag: str) -> int:
    """A linked program from GLSL 4.50 sources (source() rewrites them for this system), with each sampler on the
    texture unit its layout(binding) named. Raises OpenGL.GL.shaders' errors, with the compiler's log, if either
    doesn't compile or they don't link."""
    from OpenGL.GL import shaders

    vs, v_units = source(vert)
    fs, f_units = source(frag)
    prog = shaders.compileProgram(shaders.compileShader(vs, GL.GL_VERTEX_SHADER),
                                  shaders.compileShader(fs, GL.GL_FRAGMENT_SHADER), validate=False)
    units = {**v_units, **f_units}
    if units:
        was = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
        GL.glUseProgram(prog)
        for name, unit in units.items():
            loc = GL.glGetUniformLocation(prog, name)
            if loc >= 0:  # a sampler the compiler dropped as unused has none
                GL.glUniform1i(loc, unit)
        GL.glUseProgram(int(was))
    return prog
