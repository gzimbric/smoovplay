"""docs/frame-tap-writer.c, the C example in docs/frame-tap.md, against SmoovPlay's reader (framegen/tap.py): built
with MSYS2's gcc when it's there (C:\\msys64 or D:\\msys64) on Windows, or clang on macOS, run on a tap name of its
own, read on the CPU."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NAME = "Local\\smoovplay-tap-ctest"
if sys.platform == "win32":
    GCC = next((p for p in (Path(r"C:\msys64\mingw64\bin\gcc.exe"), Path(r"D:\msys64\mingw64\bin\gcc.exe"))
                if p.exists()), None)
elif sys.platform == "darwin":  # Apple's clang
    GCC = next((Path(p) for p in (shutil.which("clang"), shutil.which("cc")) if p), None)
else:  # the example is for Windows and macOS
    GCC = None


@pytest.mark.skipif(GCC is None, reason={"win32": "no MSYS2 gcc", "darwin": "no C compiler"}.get(
    sys.platform, "the example writer is for Windows and macOS"))
def test_c_writer_frames_read_back():
    from framegen.chiaki_tap import FrameTap, unlink
    from framegen.io import Waiting
    from framegen.tap import TapSource, wait_for_writer

    exe = Path(tempfile.gettempdir()) / ("frame_tap_writer_test.exe" if sys.platform == "win32" else
                                         "frame_tap_writer_test")
    env = dict(os.environ, PATH=str(GCC.parent) + os.pathsep + os.environ["PATH"])
    subprocess.run([str(GCC), "-O2", "-Wall", "-Werror", "-o", str(exe), str(ROOT / "docs" / "frame-tap-writer.c")],
                   check=True, env=env)
    tap = FrameTap(NAME, NAME + "-frame")
    tap.end()
    proc = subprocess.Popen([str(exe), "120", NAME], env=env, stdout=subprocess.PIPE, text=True)
    try:
        h = wait_for_writer(tap, 10)
        assert (h.version, h.codec, h.width, h.height, h.max_fps) == (2, 17, 1920, 1080, 60)
        assert (h.slots, h.slot_size) == (8, 8 << 20)
        src = TapSource(FrameTap(NAME, NAME + "-frame"), NAME, h, device=None)
        got, t_end = [], time.perf_counter() + 10
        for t in src:
            assert time.perf_counter() < t_end, "no frames came"
            if isinstance(t, Waiting):
                continue
            got.append(t)
            if len(got) == 20:
                break
        src.close()
        assert all(t.frame.shape == (1080, 1920, 3) for t in got)
        # Each picture's number is in red; new pictures say so, and repeats show the picture before again.
        for a, b in zip(got, got[1:]):
            if b.new_hint:
                assert int(b.frame[0, 0, 2]) == int(a.frame[0, 0, 2]) + 1
            else:
                assert b.new_hint is False and (b.frame == a.frame).all()
        assert {t.new_hint for t in got} == {True, False}
    finally:
        proc.wait(timeout=30)
        tap.close()
        unlink(NAME)  # POSIX shared memory lasts until it's removed
    assert proc.returncode == 0
