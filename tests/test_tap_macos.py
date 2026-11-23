"""The frame tap on macOS (framegen/tapsys.py, chiaki_tap/frametap_posix.h): shared-memory names, either side first,
the polled wait, and chiaki-ng's C writer built with clang."""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS's shared memory")

from framegen.chiaki_tap import AUDIO_SUFFIX, FrameTap  # noqa: E402
from framegen.tapsys import SHM_NAME_MAX, shm_name, unlink  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_names():
    assert shm_name("Local\\chiaki-frametap") == shm_name("/chiaki-frametap") == shm_name("chiaki-frametap")
    assert shm_name("Local\\chiaki-frametap") == "/chiaki-frametap"  # what chiaki-ng's tap opens on a Mac
    assert len(shm_name("Local\\chiaki-frametap-selftest" + AUDIO_SUFFIX)) == SHM_NAME_MAX  # just fits
    for bad in ("", "Local\\", "Local\\a/b", "Local\\" + "x" * SHM_NAME_MAX):
        with pytest.raises(ValueError):
            shm_name(bad)
    with pytest.raises(ValueError, match="31 characters at most"):
        FrameTap("Local\\" + "x" * SHM_NAME_MAX)  # fails before any memory is made


def _after(s: float, f, *args) -> threading.Timer:
    t = threading.Timer(s, f, args)
    t.start()
    return t


def test_either_side_first_and_the_polled_wait():
    name = f"Local\\sp-mac-{os.getpid()}"
    reader = FrameTap(name)
    writer = FrameTap(name)  # finds it already sized (macOS sizes an object only once)
    try:
        writer.begin(1, 64, 64, 60)
        reader.wait(0)
        t0 = time.perf_counter()
        reader.wait(0.05)  # nothing comes, so it times out
        assert 0.045 < time.perf_counter() - t0 < 0.2
        t, t0 = _after(0.05, writer.push, b"frame"), time.perf_counter()
        reader.wait(2.0)  # woken by the frame, not the timeout
        waited = time.perf_counter() - t0
        t.join()
        assert waited < 0.5 and reader.header().write_seq == 1
        writer.push(b"another")  # lands between two waits, so the next returns at once, as with the event
        t0 = time.perf_counter()
        reader.wait(2.0)
        assert time.perf_counter() - t0 < 0.05
        p = reader.read(1, reader.header().qpc_freq)
        assert p.data == b"another" and 0 <= time.perf_counter() - p.t_arrival < 1.0
        t, t0 = _after(0.05, writer.end), time.perf_counter()
        reader.wait(2.0)  # the stream's end wakes it too
        waited = time.perf_counter() - t0
        t.join()
        assert waited < 0.5 and not reader.header().live
    finally:
        reader.close()
        writer.close()
        unlink(name, name + AUDIO_SUFFIX)


@pytest.mark.skipif(not shutil.which("clang"), reason="needs clang (Xcode's command line tools)")
def test_the_c_writer_against_the_reader():
    """chiaki_tap/selftest.py: frametap.c built with clang, checked byte for byte against framegen/chiaki_tap.py."""
    r = subprocess.run([sys.executable, str(ROOT / "chiaki_tap" / "selftest.py")], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("OK:"), r.stdout + r.stderr
