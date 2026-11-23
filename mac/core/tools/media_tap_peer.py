"""The Python side of sp-tap's interop tests (mac/core/crates/sp-tap/tests/interop.rs): framegen/chiaki_tap.py's
FrameTap and AudioTap reading what the Rust writer wrote, writing what the Rust reader checks, and racing it to create
the shared memory.

    python mac/core/tools/media_tap_peer.py read NAME      # check the Rust writer's frames and sound, then ask back
    python mac/core/tools/media_tap_peer.py write NAME     # write frames and sound, answer the Rust reader's asks
    python mac/core/tools/media_tap_peer.py race NAME T    # open NAME at time T (perf_counter), write, wait for an ask

Frame i (both directions): size 50000 + 1000 i (even i) or 1000 + i (odd), byte j = (31 i + j) & 255, frames_lost 2
at i == 7, recovered at i == 8, flags new (even) / repeat (odd). Sound: 20 chunks of 1920 bytes and one of 20000, byte
j of chunk k = (7 k + j) & 255, 48 kHz stereo. Prints one line starting "OK" on success.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from framegen.chiaki_tap import (BUTTON_OPTIONS, FLAG_NEW, FLAG_REPEAT, AudioTap, FrameTap)  # noqa: E402

FRAMES = 24  # fewer than the ring's 32, so the reader can read them all after the writer is done


def frame(i: int) -> bytes:
    size = 1000 + i if i % 2 else 50000 + i * 1000
    return bytes((i * 31 + j) & 0xFF for j in range(size))


def chunk(k: int) -> bytes:
    return bytes((k * 7 + j) & 0xFF for j in range(1920 if k < 20 else 20000))


def wait_live(tap: FrameTap, seconds: float = 10.0):
    t_end = time.perf_counter() + seconds
    while not (h := tap.header()).live:
        if time.perf_counter() > t_end:
            raise SystemExit("FAIL: no stream")
        tap.wait(0.05)
    return h


def read(name: str) -> int:
    tap, sound = FrameTap(name), AudioTap(name + "-audio")
    h = wait_live(tap)
    t_end = time.perf_counter() + 10
    while tap.header().write_seq < FRAMES and time.perf_counter() < t_end:
        tap.wait(0.05)
    h = tap.header()
    assert (h.version, h.codec, h.width, h.height, h.max_fps, h.write_seq) == (1, 1, 1920, 1080, 60, FRAMES), h
    for i in range(FRAMES):
        p = tap.read(i, h.qpc_freq)
        assert p is not None and p.data == frame(i), f"frame {i}"
        assert p.frames_lost == (2 if i == 7 else 0) and p.recovered == (i == 8) and not p.truncated, i
        assert p.flags == (FLAG_NEW if i % 2 == 0 else FLAG_REPEAT), (i, p.flags)
        assert 0 <= time.perf_counter() - p.t_arrival < 10, f"frame {i}: arrival time off"
    sound.reset(from_start=True)
    chunks = sound.read_new()
    assert b"".join(c.pcm for c in chunks) == b"".join(chunk(k) for k in range(21)), len(chunks)
    assert len(chunks) == 23 and all((c.rate, c.channels) == (48000, 2) for c in chunks)
    assert abs((chunks[-1].t - chunks[-2].t) - 2040 / 48000) < 1e-3
    # Asks the other way, for the Rust writer to check
    tap.request_idr()
    tap.press(BUTTON_OPTIONS, 120)
    tap.set_audio_delay(42)
    tap.heartbeat()
    print(f"OK read {FRAMES} frames and {len(chunks)} sound slots")
    return 0


def write(name: str) -> int:
    tap, sound = FrameTap(name), AudioTap(name + "-audio")
    tap.end()
    sound.begin(48000, 2)
    tap.begin(1, 1920, 1080, 60, features=7)
    for k in range(21):
        sound.push(chunk(k))
    for i in range(FRAMES):
        tap.push(frame(i), frames_lost=2 if i == 7 else 0, recovered=i == 8, new=i % 2 == 0)
        time.sleep(0.004)
    t_end = time.perf_counter() + 10
    while not tap.take_idr_request():
        if time.perf_counter() > t_end:
            raise SystemExit("FAIL: no keyframe request")
        time.sleep(0.005)
    while not (b := tap.take_press_request()):
        if time.perf_counter() > t_end:
            raise SystemExit("FAIL: no press request")
        time.sleep(0.005)
    hold = __import__("struct").unpack_from("<I", tap.mm, 72)[0]
    time.sleep(0.02)
    tap.press_done()
    while tap.audio_delay_ms() == 0:
        if time.perf_counter() > t_end:
            raise SystemExit("FAIL: no sound delay")
        time.sleep(0.005)
    delay = tap.audio_delay_ms()
    time.sleep(0.2)  # (the reader checks press_sent)
    tap.end()
    sound.end()
    print(f"OK wrote {FRAMES} frames; press {b} for {hold} ms; audio delay {delay} ms")
    return 0


def race(name: str, t_start: float) -> int:
    while time.perf_counter() < t_start:
        pass
    tap = FrameTap(name)
    tap.begin(1, 64, 64, 60)
    tap.push(b"from-python")
    t_end = time.perf_counter() + 5
    while not tap.take_idr_request():
        if time.perf_counter() > t_end:
            raise SystemExit("FAIL: the Rust side never asked (it didn't see our memory)")
        time.sleep(0.001)
    print("OK race")
    return 0


def main() -> int:
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "read":
        return read(name)
    if mode == "write":
        return write(name)
    if mode == "race":
        return race(name, float(sys.argv[3]))
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main())
