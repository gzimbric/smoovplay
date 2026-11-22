"""Fill the screen (framegen/fill.py): the same 16:9 picture fit, zoomed, stretched or bent across a wider screen."""
import math

import numpy as np
import pytest

from framegen.fill import FILLS, layout, panorama_t, panorama_u, visible, wider
from framegen.settings import BY_KEY

W, H = 1920, 1080


def test_fit_keeps_all_of_the_picture_with_bars():
    assert layout("fit", 3440, 1440, W, H) == (2560, 1440, 440, 0)  # 21:9: bars at the sides
    assert layout("fit", 1920, 1200, W, H) == (1920, 1080, 0, 60)  # 16:10: bars top and bottom
    assert layout("fit", 2560, 1440, W, H) == (2560, 1440, 0, 0)  # 16:9: the whole screen
    assert layout("panorama", 3440, 1440, W, H) == layout("fit", 3440, 1440, W, H)  # drawn as fit, then bent


def test_zoom_fills_by_cropping_and_stretch_by_widening():
    dw, dh, x0, y0 = layout("zoom", 3440, 1440, W, H)
    assert (dw, x0) == (3440, 0) and abs(dw / dh - W / H) < 0.002  # true to shape...
    assert y0 < 0 and y0 + dh > 1440  # ...past the top and bottom: about a quarter of the picture cut off
    assert 0.24 < 1 - 1440 / dh < 0.27
    dw, dh, x0, y0 = layout("zoom", 1920, 1200, W, H)
    assert (dh, y0) == (1200, 0) and x0 < 0  # 16:10: the sides cut instead
    assert layout("stretch", 3440, 1440, W, H) == (3440, 1440, 0, 0)
    for fill in FILLS:  # a 16:9 screen: every one is the whole picture, unchanged
        assert layout(fill, 2560, 1440, W, H) == (2560, 1440, 0, 0), fill


def test_messages_and_the_menu_go_on_the_part_of_the_picture_on_screen():
    assert visible(*layout("zoom", 3440, 1440, W, H), 3440, 1440) == (0, 0, 3440, 1440)
    assert visible(*layout("fit", 3440, 1440, W, H), 3440, 1440) == (440, 0, 2560, 1440)


def test_panorama_keeps_the_middle_close_to_true_and_never_folds():
    assert not wider(2560, 1440, W, H) and not wider(1920, 1200, W, H) and wider(3440, 1440, W, H)
    assert panorama_t(2560, 1440, W, H) == 0.0 and panorama_u(0.3, 0.0) == 0.3
    for fw, fh, middle, edge in ((3440, 1440, 1.06, 2.5), (2560, 1080, 1.06, 2.5), (5120, 1440, 1.16, 5.5)):
        t = panorama_t(fw, fh, W, H)
        k = (fw / fh) / (W / H)
        xs = np.linspace(-1, 1, 2001)
        us = np.array([panorama_u(x, t) for x in xs])
        assert abs(us[0] + 1) < 1e-9 and abs(us[-1] - 1) < 1e-9 and abs(us[1000]) < 1e-9  # edge to edge, centred
        assert (np.diff(us) > 0).all()  # always moving on across the picture: nothing repeated or flipped
        widen = k * (xs[1] - xs[0]) / np.diff(us)  # how much wider than true, across the screen
        assert 1.0 < widen[1000] < middle and widen[0] < edge, (fw, fh, widen[1000], widen[0])
        assert widen[1000] < widen[750] < widen[500] < widen[0]  # more towards the sides
        assert abs(widen[1000] - k * math.atan(t) / t) < 0.01


def test_the_setting_says_it_is_the_same_16_9_picture():
    s = BY_KEY["fill"]
    assert s.choices == FILLS and s.default == "fit" and s.live
    assert "16:9" in s.help and "none of these is real ultrawide" in s.help and "Key: Z" in s.help
    assert "16:9" in s.impact["picture"]


# ---- Panorama's pass, held to panorama_u (needs OpenGL 4.5, or 4.1 on a Mac: skipped without a graphics card)
@pytest.fixture(scope="module")
def gl():
    glfw = pytest.importorskip("glfw")
    from OpenGL import GL

    from framegen.glsl import context_hints
    if not glfw.init():
        pytest.skip("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    context_hints(glfw)  # (a Mac: OpenGL 4.1 core)
    win = glfw.create_window(64, 64, "test", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("no OpenGL 4.5 (Mac: 4.1) context")
    glfw.make_context_current(win)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    yield GL
    glfw.destroy_window(win)
    glfw.terminate()


def test_the_panorama_pass_bends_the_picture_as_the_curve_says(gl):
    from framegen.fill import Panorama
    GL = gl
    fw, fh = 860, 360  # 21:9
    dw, dh, _, _ = layout("panorama", fw, fh, W, H)
    p = Panorama(GL)
    ramp = np.tile(np.linspace(0, 255, dw).astype(np.uint8)[None, :, None], (dh, 1, 4))  # red rises left to right
    ramp[: dh // 2, :, 1] = 200  # and the top half is green, to check it stays upright
    ramp[dh // 2:, :, 1] = 0
    GL.glBindTexture(GL.GL_TEXTURE_2D, (p.target(dw, dh), p.tex)[1])
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, dw, dh, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ramp[::-1].copy())
    out_tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, out_tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, fw, fh, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    out = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, out)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, out_tex, 0)
    t = panorama_t(fw, fh, W, H)
    p.draw(out, fw, fh, t)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, out)
    img = np.frombuffer(GL.glReadPixels(0, 0, fw, fh, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(fh, fw, 4)
    img = img[::-1]  # image row 0 at the top
    assert img[10, :, 1].min() > 150 and img[-10, :, 1].max() < 50  # upright, and the full height
    red = img[fh // 4, :, 0].astype(float)
    for col in (0, fw // 8, fw // 4, fw // 2, 3 * fw // 4, fw - 1):
        x = (col + 0.5) / fw * 2 - 1
        want = (panorama_u(x, t) + 1) / 2 * 255
        assert abs(red[col] - want) < 3, (col, red[col], want)
    assert (np.diff(red) >= -1).all()  # never runs backwards
    p.free()
    GL.glDeleteFramebuffers(1, [out])
    GL.glDeleteTextures([out_tex])
