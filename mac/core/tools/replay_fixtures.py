"""Reference data for sp-replay's clip checks (mac/core/fixtures/replay/, git-ignored).

    .venv/bin/python mac/core/tools/replay_fixtures.py [--seconds 10] [--length 5]

Feeds framegen/replay.py's Recorder (the Python reference, on this Mac: h264_videotoolbox) a decoded clip as
SmoovPlay would, at 60 and 90 Hz, with its sound, and saves what R saves. sp-replay's tests (tests/clip.rs) feed the
Rust recorder the same plan and compare.

The plan (both sides build it from manifest.json the same way):
  - frames: testdata/clip_1080p60.mp4 as FFmpeg's bgr24 (the same bytes on both sides), the first `seconds`;
  - shown frame i at t = T0 + DELAY + i/hz shows content frame floor(i*60/hz), with i stamped in its top-left corner
    (16 blocks of 16x16, bit b of i white at x = 16b, rows 0..16), so a saved clip tells which frame is which;
  - the game's own frames: the even content frames (a 30 fps game), each the first time it's shown, arriving at
    T0 + c/60;
  - sound: the clip's own (mono, 48 kHz, s16; a steady 440 Hz tone) with seeded white noise added, so a saved clip's
    sound can be lined up with the original to the sample; in 10 ms chunks, chunk k arriving at T0 + k/100, given
    before the frame shown at or after its time plus the delay.

Writes: manifest.json, clip_audio.s16 (the clip's sound), python_<hz>.mp4 and python_<hz>_compare.mp4 (the Python
recorder's saved clip and before & after video), and python_<hz>.json (its log and how long each part took).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from framegen import replay  # noqa: E402
from framegen.io import Sound  # noqa: E402

OUT = ROOT / "mac/core/fixtures/replay"
CLIP = ROOT / "testdata/clip_1080p60.mp4"
T0, DELAY = 1000.0, 0.05
W, H = 1920, 1080


def frames(seconds: float):
    """The clip's first `seconds` as bgr24 frames (FFmpeg)."""
    n = round(seconds * 60)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(CLIP), "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = W * H * 3
    out = []
    while len(out) < n:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
    p.wait()
    return out


def stamp(img: np.ndarray, i: int) -> np.ndarray:
    a = img.copy()
    for b in range(16):
        a[0:16, 16 * b:16 * b + 16] = 255 if (i >> b) & 1 else 0
    return a


def audio() -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(CLIP), "-map", "0:a:0", "-ac", "1", "-ar", "48000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    tone = np.frombuffer(raw, np.int16).astype(np.int32)
    noise = np.random.default_rng(7).integers(-3000, 3000, len(tone))
    return np.clip(tone + noise, -32768, 32767).astype(np.int16)


def run_python(hz: int, seconds: float, length: int, pics, pcm: np.ndarray, out: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="replay-py-"))
    log = []
    r = replay.Recorder(length, out_dir=tmp, encoder="h264_videotoolbox", device="cpu", drop_when_behind=False,
                        notify=lambda t, s: log.append(t))
    r.set_delay(DELAY * 1000)
    r.set_output_hz(hz)
    k = 0
    seen = set()
    t_start = time.perf_counter()
    for i in range(round(seconds * hz)):
        t = T0 + DELAY + i / hz
        while k * 480 + 480 <= len(pcm) and T0 + k / 100 + DELAY <= t:
            r.add_sound([Sound(T0 + k / 100, pcm[k * 480:(k + 1) * 480].tobytes(), 48000, 1)])
            k += 1
        c = (i * 60) // hz
        if c >= len(pics):
            break
        img = stamp(pics[c], i)
        r.frame(img, t)
        if c % 2 == 0 and c not in seen:
            seen.add(c)
            r.real_frame(img, T0 + c / 60)
    feed_s = time.perf_counter() - t_start
    assert r.flush(60)
    t1 = time.perf_counter()
    paths = r.save(wait=True)
    save_s = time.perf_counter() - t1
    t1 = time.perf_counter()
    compare = r.save("compare", wait=True)
    compare_s = time.perf_counter() - t1
    held = r.held()
    r.close()
    shutil.copy(paths[0], out)
    if compare:
        shutil.copy(compare[0], out.with_name(out.stem + "_compare.mp4"))
    shutil.rmtree(tmp, ignore_errors=True)
    return {"hz": hz, "feed_s": feed_s, "save_s": save_s, "compare_s": compare_s, "held": held, "toasts": log,
            "dropped": r.dropped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--length", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pcm = audio()
    (OUT / "clip_audio.s16").write_bytes(pcm.tobytes())
    pics = frames(a.seconds)
    manifest = {"clip": str(CLIP.relative_to(ROOT)), "t0": T0, "delay": DELAY, "seconds": a.seconds,
                "length": a.length, "width": W, "height": H, "hz": [60, 90], "audio_rate": 48000,
                "audio_channels": 1, "chunk": 480}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for hz in (60, 90):
        info = run_python(hz, a.seconds, a.length, pics, pcm, OUT / f"python_{hz}.mp4")
        (OUT / f"python_{hz}.json").write_text(json.dumps(info, indent=1, default=str))
        print(f"python {hz} Hz: fed in {info['feed_s']:.1f} s, saved in {info['save_s']:.2f} s, before & after in "
              f"{info['compare_s']:.1f} s, held {info['held']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
