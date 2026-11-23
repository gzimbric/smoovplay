"""Build selftest.c against frametap.c with MSYS2's gcc (Windows) or clang (macOS) and check its frames with the
Python reader.

    .venv\\Scripts\\python chiaki_tap\\selftest.py [MSYS2 folder, default C:\\msys64]
    .venv/bin/python chiaki_tap/selftest.py [C compiler, default clang]                  # macOS

Verifies the C writer (what runs inside chiaki-ng) and framegen/chiaki_tap.py agree on every byte: header
fields, frame data, sizes, loss/recovered flags, the keyframe-request flag, the delay test's button press, the
sound delay (and that it lapses without SmoovPlay's heartbeat), the game's sound for instant replay (data, format
and times, including a chunk split over several slots) and the end-of-stream marker. Uses a tap name of
its own (CHIAKI_FRAMETAP_NAME), so it can run while chiaki-ng is streaming.
"""
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from framegen.chiaki_tap import (AUDIO_SUFFIX, BUTTON_OPTIONS, FEATURE_AUDIO_DELAY, FEATURE_AUDIO_TAP,  # noqa: E402
                                 FEATURE_PRESS, WINDOWS, AudioTap, FrameTap, qpc_freq, unlink)

NAME = r"Local\chiaki-frametap-selftest"


def main() -> int:
    if WINDOWS:
        msys = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\msys64")  # MSYS2's default install folder
        gcc = msys / "mingw64" / "bin" / "gcc.exe"
        exe = Path(tempfile.gettempdir()) / "frametap_selftest.exe"
        env = dict(os.environ, PATH=str(gcc.parent) + os.pathsep + os.environ["PATH"], CHIAKI_FRAMETAP_NAME=NAME)
    else:  # macOS: Apple's clang (POSIX shared memory, frametap.c's other half)
        gcc = Path(sys.argv[1] if len(sys.argv) > 1 else shutil.which("clang") or "cc")
        exe = Path(tempfile.gettempdir()) / "frametap_selftest"
        env = dict(os.environ, CHIAKI_FRAMETAP_NAME=NAME)
    subprocess.run([str(gcc), "-O2", "-Wall", "-o", str(exe), str(HERE / "selftest.c"), str(HERE / "frametap.c")],
                   check=True, env=env)

    tap = FrameTap(NAME, NAME + "-frame")
    tap.end()  # clear any old state
    sound = AudioTap(NAME + AUDIO_SUFFIX)
    sound.end()
    proc = subprocess.Popen([str(exe)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_end = time.perf_counter() + 10
    ages = []  # how long after the C writer stamped each frame (its clock) we read it (ours), to show the clocks agree
    for i in range(40):  # read each frame as it arrives, like ChiakiSource (the ring holds only 32)
        while time.perf_counter() < t_end and (not tap.header().live or tap.header().write_seq <= i):
            tap.wait(0.1)
        h = tap.header()
        assert h.live and (h.version, h.codec, h.width, h.height, h.max_fps) == (1, 1, 1920, 1080, 60), h
        assert h.qpc_freq == qpc_freq(), (h.qpc_freq, qpc_freq())
        p = tap.read(i, h.qpc_freq)
        ages.append(time.perf_counter() - p.t_arrival if p is not None else 0.0)
        size = 1000 + i if i % 2 else 50000 + i * 1000
        want = bytes((i * 31 + j) & 0xFF for j in range(size))
        assert p is not None and p.data == want, f"frame {i}: data mismatch"
        assert p.frames_lost == (2 if i == 17 else 0) and p.recovered == (i == 18) and not p.truncated, i
        assert 0 < time.perf_counter() - p.t_arrival < 5, f"frame {i}: arrival time off"
    tap.request_idr()
    # The game's sound: every byte, in order, with its format, and times that move on with it
    assert sound.live(), "sound ring not started"
    sound.reset(from_start=True)
    chunks = sound.read_new()
    want = b"".join(bytes((k * 7 + j) & 0xFF for j in range(1920 if k < 20 else 20000)) for k in range(21))
    assert b"".join(c.pcm for c in chunks) == want, f"sound data mismatch ({len(chunks)} chunks)"
    assert len(chunks) == 23 and all((c.rate, c.channels) == (48000, 2) for c in chunks), [len(c.pcm) for c in chunks]
    assert all(b.t >= a.t for a, b in zip(chunks, chunks[1:])), "sound times go backwards"
    assert abs((chunks[-1].t - chunks[-2].t) - 2040 / 48000) < 1e-3, "a split chunk's slots aren't timed by their samples"
    assert 0 < time.perf_counter() - chunks[0].t < 10, "sound arrival time off"
    # The delay test's press: asked for, sent by the writer, timed in our clock
    assert tap.features() == FEATURE_PRESS | FEATURE_AUDIO_DELAY | FEATURE_AUDIO_TAP, tap.features()
    t_ask = time.perf_counter()
    token = tap.press(BUTTON_OPTIONS, 120)
    while (sent := tap.press_sent(token, h.qpc_freq)) is None and time.perf_counter() - t_ask < 5:
        time.sleep(0.005)
    assert sent is not None and t_ask < sent < time.perf_counter(), "press not sent, or its time is off"
    # The sound delay holds while our heartbeat is fresh, then lapses
    tap.set_audio_delay(42)
    for _ in range(20):
        tap.heartbeat()
        time.sleep(0.05)
    out, err = proc.communicate(timeout=15)
    assert proc.returncode == 0, (proc.returncode, out, err)
    assert f"press {BUTTON_OPTIONS} for 120 ms sent; audio delay 42 ms" in out, out
    assert "audio delay lapsed: yes" in out, out
    assert not tap.header().live, "end-of-stream not marked"
    assert not sound.live(), "sound ring's end not marked"
    print(f"OK: 40 frames and 21 chunks of sound byte-identical, flags and timestamps right, keyframe request, "
          f"press and sound delay "
          f"({out.strip()}). Frames read {statistics.median(ages) * 1000:.2f} ms (median) after the writer stamped them")
    tap.close()
    sound.close()
    unlink(NAME)  # on POSIX the test's shared memory would otherwise stay until a restart
    return 0


if __name__ == "__main__":
    sys.exit(main())
