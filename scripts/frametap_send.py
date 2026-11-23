"""Sends a video's decoded frames into the frame tap, as another app would (docs/frame-tap.md): a reference writer
for decoded frames, and a stand-in for an app, to test SmoovPlay's frame-tap source without one.

    .venv\\Scripts\\python scripts\\frametap_send.py testdata\\clip_1080p60.mp4 --seconds 20
    .venv\\Scripts\\python run.py --source tap --display          # in another terminal, before or after

Paced at 60 frames a second, as a 60 Hz output delivers them. --game-fps 30 (the default) shows each picture twice,
as a 30 fps game does; --hints marks each frame new or a repeat (FLAG_NEW / FLAG_REPEAT), as a writer that knows
would, and without it SmoovPlay compares the pictures. --variable sends a picture only when it changes
(FEATURE_VARIABLE), as an app that draws when it likes does. --format bgra sends BGRA instead of NV12.
Runs on the CPU only (PyAV decoding), so it leaves the graphics card to SmoovPlay. On a Mac it's the same with
.venv/bin/python. The tap is POSIX shared memory there, so --tap NAME can have at most 24 characters after any
Local\\.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from framegen.chiaki_tap import FEATURE_VARIABLE, RAW_CODECS, FrameTap  # noqa: E402
from framegen.tap import DEFAULT_NAME  # noqa: E402

CODEC = {fmt: codec for codec, fmt in RAW_CODECS.items()}  # "nv12" -> 16, "bgra" -> 17


def pictures(path: str, start: float, game_fps: float):
    """The clip's pictures at game_fps, from `start`: (index, frame) for each, decoded on the CPU."""
    cont = av.open(path)
    stream = cont.streams.video[0]
    stream.thread_type = "AUTO"
    want = 0
    try:
        for fr in cont.decode(stream):
            t = (fr.time or 0.0) - start
            if t < want / game_fps - 1e-3:
                continue
            yield want, fr
            want += 1
    finally:
        cont.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--tap", default=DEFAULT_NAME, help="shared-memory name (default %(default)s)")
    ap.add_argument("--format", default="nv12", choices=sorted(CODEC))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--game-fps", type=float, default=30.0, help="the game's own frame rate (60 or below)")
    ap.add_argument("--hints", action="store_true", help="say which frames are new and which are repeats")
    ap.add_argument("--variable", action="store_true", help="send a picture only when it changes")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before starting")
    a = ap.parse_args()
    if not 0 < a.game_fps <= 60:
        ap.error("--game-fps: 60 or below")

    tap = FrameTap(a.tap, a.tap + "-frame")
    time.sleep(a.delay)
    started, sent, t0 = False, 0, 0.0
    tick = 0  # 60 Hz refreshes so far
    try:
        for i, fr in pictures(a.video, a.start, a.game_fps):
            data = fr.to_ndarray(format=a.format)
            if not started:
                tap.begin(CODEC[a.format], fr.width, fr.height, 60,
                          features=FEATURE_VARIABLE if a.variable else 0)
                print(f"[send] {a.format} {fr.width}x{fr.height} into {a.tap}, a {a.game_fps:g} fps game"
                      + (", only new pictures" if a.variable else " at 60 Hz") + (", with hints" if a.hints else ""))
                started, t0 = True, time.perf_counter()
            # Every 60 Hz refresh until the next picture is due shows this one (once, if --variable).
            refreshes = 0
            while tick < round((i + 1) * 60 / a.game_fps) or refreshes == 0:
                if (wait := t0 + tick / 60 - time.perf_counter()) > 0:
                    time.sleep(wait)
                if refreshes == 0 or not a.variable:
                    tap.push(data, new=(refreshes == 0) if a.hints else None)
                    sent += 1
                refreshes += 1
                tick += 1
            if tick >= a.seconds * 60:
                break
    except KeyboardInterrupt:
        pass
    finally:
        tap.end()
        tap.close()
    print(f"[send] {sent} frames in {tick / 60:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
