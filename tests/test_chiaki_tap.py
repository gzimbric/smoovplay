import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from framegen.chiaki_tap import (SHM_NAME_MAX, SLOTS, WINDOWS, FrameTap, SizeRepeatDetector, check_name,  # noqa: E402
                                 is_keyframe, posix_name, unlink)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tap():
    t = FrameTap("Local\\chiaki-frametap-test", "Local\\chiaki-frametap-test-frame")
    yield t
    t.close()
    unlink("Local\\chiaki-frametap-test")  # POSIX shared memory lasts until it's removed


def test_roundtrip(tap):
    tap.begin(1, 1920, 1080, 60)
    h = tap.header()
    assert h.live and (h.codec, h.width, h.height, h.max_fps, h.write_seq) == (1, 1920, 1080, 60, 0)
    tap.push(b"\x00\x00\x01frame-zero", frames_lost=0)
    tap.push(b"second" * 1000, frames_lost=2, recovered=True)
    h = tap.header()
    assert h.write_seq == 2
    p0, p1 = tap.read(0, h.qpc_freq), tap.read(1, h.qpc_freq)
    assert p0.data == b"\x00\x00\x01frame-zero" and p0.frames_lost == 0 and not p0.recovered
    assert p1.data == b"second" * 1000 and p1.frames_lost == 2 and p1.recovered and not p1.truncated


def test_overwritten_slot_reads_none(tap):
    tap.begin(0, 64, 64, 60)
    for i in range(SLOTS + 1):
        tap.push(bytes([i % 256]) * 10)
    h = tap.header()
    assert tap.read(0, h.qpc_freq) is None  # slot 0 now holds frame SLOTS
    assert tap.read(SLOTS, h.qpc_freq).data == bytes([SLOTS % 256]) * 10


def test_new_session_and_idr_flag(tap):
    tap.begin(0, 64, 64, 60)
    s = tap.header().session
    tap.request_idr()
    assert tap.take_idr_request() and not tap.take_idr_request()
    tap.end()
    assert not tap.header().live
    tap.begin(0, 64, 64, 60)
    assert tap.header().session == s + 1 and tap.header().write_seq == 0


def test_keyframes():
    assert is_keyframe(b"\x00\x00\x00\x01\x67sps\x00\x00\x01\x68pps\x00\x00\x01\x65idr", "h264")
    assert not is_keyframe(b"\x00\x00\x01\x41p-slice", "h264")
    assert is_keyframe(b"\x00\x00\x01\x40\x01vps\x00\x00\x01\x26\x01idr", "hevc")  # type 32 VPS, 19 IDR_W_RADL
    assert not is_keyframe(b"\x00\x00\x01\x02\x01trail", "hevc")  # type 1 TRAIL_R


def test_size_detector_30fps_cadence():
    d = SizeRepeatDetector()
    sizes = [60000, 3000] * 10
    out = [d.update(s, False) for s in sizes]
    assert out[:3] == [None, None, None]
    assert out[4:] == [True, False] * 8
    # a keyframe in steady 30-in-60 cadence takes the opposite of the frame before it
    assert d.update(150000, True) is True  # last decision was False (a repeat)
    assert d.update(2500, False) is False


def test_size_detector_60fps_all_new():
    d = SizeRepeatDetector()
    out = [d.update(s, False) for s in [50000, 52000, 47000, 61000, 55000, 49000, 58000, 51000]]
    assert all(out[4:])


def test_size_detector_recovers_after_a_still_stretch():
    # A menu or loading screen sends tiny frames; then a 30 fps game resumes. Its new frames are far bigger
    # than anything in the window, so they were treated like keyframes and left out of it: the window stayed full
    # of tiny frames and the detector stuck. Many big frames recently are now the new normal. (Real 30 fps logs,
    # 17-18 Nov: silent on 55-91% of ticks before, 1-4% after.)
    rng = np.random.default_rng(3)
    d = SizeRepeatDetector()
    for _ in range(180):
        d.update(max(50, int(rng.normal(600, 200))), False)
    out = [d.update(int(rng.normal(50000, 8000)) if i % 2 == 0 else int(rng.normal(2000, 300)), False)
           for i in range(300)]
    assert out[20:] == [True, False] * 140  # back on track within a third of a second


# ---- the game's sound, for instant replay (frametap.h "Sound ring"; chiaki_tap/selftest.py checks the C writer)
def test_sound_ring_roundtrip_split_and_times():
    from framegen.chiaki_tap import AUDIO_SLOT_HEADER, AUDIO_SLOT_SIZE, AUDIO_SLOTS, AudioTap, qpc
    import time
    a = AudioTap(r"Local\chiaki-frametap-test-audio")
    try:
        a.end()
        assert not a.live() and a.read_new() == []
        a.begin(48000, 2)
        assert a.live()
        a.reset(from_start=True)
        big = bytes(range(256)) * 80  # 20480 bytes: more than a slot holds, so it's split by whole samples
        q0 = qpc()
        a.push(b"\x01\x02\x03\x04" * 480, q0)
        a.push(big, q0)
        got = a.read_new()
        cap = (AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER) // 4 * 4
        assert [len(c.pcm) for c in got] == [1920, cap, cap, len(big) - 2 * cap]
        assert b"".join(c.pcm for c in got[1:]) == big and all((c.rate, c.channels) == (48000, 2) for c in got)
        assert got[2].t - got[1].t == pytest.approx(cap / 4 / 48000, abs=1e-4)  # timed by the samples before them
        assert abs(time.perf_counter() - got[0].t) < 1.0
        assert a.read_new() == []  # nothing new
        for _ in range(AUDIO_SLOTS + 10):  # a reader a whole ring behind gets what's left, not garbage
            a.push(b"\x00" * 64)
        assert len(a.read_new()) == AUDIO_SLOTS - 1
        a.begin(48000, 2)  # the writer started over: the reader follows it
        a.push(b"\x05" * 64)
        assert a.read_new() == [] and a.next == 1  # from where it is now
        a.end()
        assert a.read_new() == [] and a.next is None
    finally:
        a.close()
        unlink("Local\\chiaki-frametap-test")


# ---- POSIX (macOS): names, either side first, and waiting without an event
@pytest.mark.skipif(WINDOWS, reason="POSIX shared memory names")
def test_posix_names():
    assert posix_name("Local\\chiaki-frametap") == "/chiaki-frametap"  # chiaki-ng's, from frametap.h's FRAMETAP_NAME
    assert posix_name("Local\\smoovplay-tap") == posix_name("/smoovplay-tap") == posix_name("smoovplay-tap")
    check_name("Local\\chiaki-frametap-selftest")  # the self-test's name, whose sound ring is exactly 31 on macOS
    room = SHM_NAME_MAX - 1 - len("-audio")
    check_name("x" * room)
    with pytest.raises(ValueError, match="sound ring"):
        check_name("Local\\" + "x" * (room + 1))  # fits on its own, but not with "-audio"
    with pytest.raises(ValueError, match="sound ring"):
        FrameTap("Local\\" + "x" * (room + 1))  # fails at once, before any memory is made
    for bad in ("", "Local\\", "a/b"):
        with pytest.raises(ValueError):
            posix_name(bad)


@pytest.mark.skipif(WINDOWS, reason="POSIX shared memory")
def test_posix_either_side_first_sees_the_same_memory():
    name = f"sp-first-{os.getpid()}"
    unlink(name)
    try:
        reader = FrameTap(name)  # made it (and sized it)
        writer = FrameTap(name)  # found it sized (macOS refuses a second ftruncate, which is fine)
        writer.begin(1, 64, 64, 60)
        writer.push(b"\x00\x00\x01x")
        h = reader.header()
        assert h.live and h.write_seq == 1 and h.qpc_freq == 1_000_000_000
        assert reader.read(0, h.qpc_freq).data == b"\x00\x00\x01x"
        assert 0 <= time.perf_counter() - reader.read(0, h.qpc_freq).t_arrival < 1.0  # the same clock both sides
        writer.close()
        reader.close()
    finally:
        unlink(name)


@pytest.mark.skipif(WINDOWS, reason="POSIX, where the reader polls instead of waiting on an event")
def test_posix_wait_wakes_on_a_frame_and_times_out(tap):
    tap.begin(1, 64, 64, 60)
    writer = FrameTap("Local\\chiaki-frametap-test")
    try:
        tap.header()
        t0 = time.perf_counter()
        tap.wait(0.05)  # nothing comes, so it times out
        assert 0.045 < time.perf_counter() - t0 < 0.2
        timer = threading.Timer(0.05, writer.push, (b"frame",))
        timer.start()
        tap.header()
        t0 = time.perf_counter()
        tap.wait(2.0)
        waited = time.perf_counter() - t0
        timer.join()
        assert tap.header().write_seq == 1 and waited < 0.5  # woken by the frame, not the timeout
        writer.push(b"another")  # a frame since this thread's last header(), so wait() returns at once
        t0 = time.perf_counter()
        tap.wait(2.0)
        assert time.perf_counter() - t0 < 0.05
    finally:
        writer.close()


@pytest.mark.skipif(sys.platform != "darwin" or not (shutil.which("clang") or shutil.which("cc")),
                    reason="builds with clang on macOS. On Windows run chiaki_tap/selftest.py with MSYS2's gcc")
def test_c_writer_selftest():
    """chiaki_tap/frametap.c, built with clang, against this reader byte for byte (chiaki_tap/selftest.py)."""
    r = subprocess.run([sys.executable, str(ROOT / "chiaki_tap" / "selftest.py")], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("OK:"), r.stdout + r.stderr


def test_the_patch_asks_chiaki_ng_for_opengl_once_and_only_when_told(tmp_path):
    # chiaki_tap/apply_patch.py's edit to chiaki-ng's renderer choice (issue 19: Vulkan crashed on a Radeon): the
    # launcher sets CHIAKI_FRAMETAP_RENDERER=opengl without an NVIDIA card. Applied to the anchor, twice: one edit.
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "chiaki_tap"))
    import apply_patch as a
    old, new = a.RENDERER_EDIT
    f = tmp_path / "qmlmainwindow.cpp"
    f.write_bytes(("void x()\r\n{\r\n" + old + "            reason = 1;\r\n").replace("\n", "\r\n").replace("\r\r", "\r")
                  .encode())
    a.patch_file(f, [a.RENDERER_EDIT], "CHIAKI_FRAMETAP_RENDERER")
    a.patch_file(f, [a.RENDERER_EDIT], "CHIAKI_FRAMETAP_RENDERER")
    text = f.read_bytes().decode()
    assert text.count("CHIAKI_FRAMETAP_RENDERER") == 2 and text.count("CHIAKI_FORCE_VULKAN_FALLBACK") == 1
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")  # chiaki-ng's line endings kept
    assert text.index('== QLatin1String("opengl")') < text.index("CHIAKI_FORCE_VULKAN_FALLBACK")
    assert "pending_renderer_fallback_reason.clear();" in text  # no dialog
