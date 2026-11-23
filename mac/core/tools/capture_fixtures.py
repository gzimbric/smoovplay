"""Reference data for sp-capture's tests (mac/core/fixtures/capture/, git-ignored), made by the Python itself.

    .venv/bin/python mac/core/tools/capture_fixtures.py [--skip-frames] [--skip-onset] [--check-devices]

  clip_<i>.bgr      frames of testdata/clip_1080p60.mp4 decoded by PyAV and read with capture.raw_to_bgr (CPU, torch)
                    in the file's own colours (untagged: BT.601, limited range, as FFmpeg takes it): HxWx3 uint8
                    BGR. The Rust stand-in must match these within a level.
  clip_<i>.standin.bgr  the same frames as the Python's CardSource.video_file hands them on: reformatted by swscale
                    into NV12 BT.709 (from the file's BT.601: a round trip, so the picture stays the file's own)
                    and read back with raw_to_bgr. Within a level of the above but where swscale resamples chroma
                    at colour edges.
  silent.mov        12 frames of the same, no sound track
  onset.mov         320x180 at 60 fps, 2 s, H.264 and 16-bit PCM (48 kHz stereo): dark until frame 60 (1.0 s), then
                    bright; silent until 1.0 s, then a 440 Hz tone. What the sound timing check plays.
  delay_<name>.*    framegen.audio.DelayLine runs on a simulated clock: the pushes and pulls (.json), the PCM pushed
                    (.in.s16) and what came out (.out.s16), for the Rust to replay byte for byte
  picks.json        capture.pick_mode and pick_audio on many inputs, audio.sound_report on a few stats
  manifest.json     what's here, and the Python's own measurements (onset: how far the tone's start came from the
                    flash's tick, in the Python's CardSource.video_file)

--check-devices also lists the Mac's devices through FFmpeg's AVFoundation input (names only, nothing opened) and
compares them with sp-capture's own list (target-capture/debug/examples/capture --list --json, if built).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "mac" / "core" / "fixtures" / "capture"
CLIP = ROOT / "testdata" / "clip_1080p60.mp4"
REF = (0, 1, 2, 30, 31, 359, 719)
RATE, CH = 48000, 2


def clip_frames(manifest: dict) -> None:
    import torch
    from av.video.reformatter import ColorRange, Colorspace

    from framegen import capture as c
    con = av.open(str(CLIP))
    vs = con.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    bt709 = full = None
    for i, fr in enumerate(con.decode(vs)):
        if i > max(REF):
            break
        if i not in REF:
            continue
        # the file's own colours (AVCOL_SPC_BT709 = 1; AVCOL_RANGE_JPEG = 2), as swscale takes them
        bt709, full = fr.colorspace == 1, fr.color_range == 2
        nv12 = fr.reformat(format="nv12").to_ndarray().ravel().copy()  # (the decoder's samples, as they are)
        bgr = c.raw_to_bgr(torch.from_numpy(nv12), "nv12", w, h, bt709, full).numpy()
        (OUT / f"clip_{i}.bgr").write_bytes(bgr.tobytes())
        # what CardSource.video_file does (_frames, decode_to="nv12"), the raw path's colours for 1080p
        raw = fr.reformat(format="nv12", dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG)
        stand = c.raw_to_bgr(torch.from_numpy(raw.to_ndarray().ravel().copy()), "nv12", w, h, True, False).numpy()
        (OUT / f"clip_{i}.standin.bgr").write_bytes(stand.tobytes())
    con.close()
    manifest["clip"] = {"path": str(CLIP.relative_to(ROOT)), "w": w, "h": h, "frames": list(REF),
                        "bt709": bt709, "full_range": full}
    print(f"[fixtures] clip frames {REF} ({w}x{h}), the file's colours: {'BT.709' if bt709 else 'BT.601'}, "
          f"{'full' if full else 'limited'} range")


def make_onset(path: Path, seconds=2.0, fps=60, at=1.0, w=320, h=180, sound=True) -> None:
    out = av.open(str(path), "w", format="mov")
    vs = out.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    vs.options = {"crf": "18", "g": "30", "bf": "0"}
    aus = out.add_stream("pcm_s16le", rate=RATE, layout="stereo") if sound else None
    per = RATE // fps
    k = 0
    for i in range(int(seconds * fps)):
        img = np.full((h, w, 3), 230 if i >= int(at * fps) else 20, np.uint8)
        f = av.VideoFrame.from_ndarray(img, format="bgr24")
        f.pts = i
        for p in vs.encode(f):
            out.mux(p)
        if aus is None:
            continue
        n = np.arange(k, k + per)
        k += per
        tone = np.where(n >= int(at * RATE), 12000 * np.sin(2 * np.pi * 440 * n / RATE), 0).astype(np.int16)
        a = np.repeat(tone, CH)[None]
        af = av.AudioFrame.from_ndarray(a, format="s16", layout="stereo")
        af.sample_rate, af.pts = RATE, i * per
        for p in aus.encode(af):
            out.mux(p)
    for st in (vs, aus):
        for p in st.encode(None) if st is not None else []:
            out.mux(p)
    out.close()


class _Out:
    """Stands in for framegen.audio.Player (tests/test_audio.py's)."""

    def __init__(self, rate, channels):
        self.rate, self.channels, self.pushed, self.delays = rate, channels, [], []

    def push(self, pcm):
        self.pushed.append((time.perf_counter(), pcm))

    def set_delay(self, s):
        self.delays.append(s)

    def close(self):
        pass


def onset(manifest: dict) -> None:
    from framegen import capture as c
    from framegen.io import Waiting
    path = OUT / "onset.mov"
    make_onset(path)
    make_onset(OUT / "silent.mov", seconds=0.2, sound=False)  # 12 frames, no sound track
    outs = []
    src = c.CardSource.video_file(str(path), device="cpu", audio=lambda r, ch: outs.append(_Out(r, ch)) or outs[-1])
    ticks = [t for t in src if not isinstance(t, Waiting)]
    src.close()
    flash = next(t for t in ticks if t.frame.float().mean() > 120)
    t_on = None
    for t, pcm in outs[0].pushed:
        a = np.abs(np.frombuffer(pcm, np.int16))
        if a.max() > 1000:
            t_on = t + int(np.argmax(a > 1000)) // CH / RATE
            break
    total = sum(len(p) for _, p in outs[0].pushed) / CH / 2 / RATE
    manifest["onset"] = {"path": "onset.mov", "flash_frame": 60, "tone_s": 1.0, "ticks": len(ticks),
                         "python_offset_s": t_on - flash.t_arrival, "python_flash_seq": flash.extra["seq"],
                         "python_sound_s": total}
    print(f"[fixtures] onset.mov: the Python's tone came {1000 * (t_on - flash.t_arrival):+.1f} ms from its flash's "
          f"tick; {len(ticks)} ticks, {total:.3f} s of sound")


# ----------------------------------------------------------------------------- DelayLine traces
class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


def _card(secs, hold_ms=0.0, every_s=5.0, seed=2, first_at=None):
    """tests/test_audio.py _card: a card's 20 ms packets, held up now and then, and the output's 10 ms blocks."""
    rnd = random.Random(seed)
    ev, held, nxt = [], 0, every_s if first_at is None else first_at
    for i in range(int(secs / 0.02)):
        t = i * 0.02 + abs(rnd.gauss(0, 0.002))
        if hold_ms and nxt <= t < nxt + hold_ms / 1000:
            held += 1
            continue
        if held:
            ev += [(nxt + hold_ms / 1000, "in")] * held
            held, nxt = 0, nxt + every_s
        ev.append((t, "in"))
    ev += [(i * 0.01, "out") for i in range(int(secs / 0.01))]
    return sorted(ev, key=lambda e: e[0])


def _sine(channels, freq=220, amp=12000, per_s=0.02):
    state = {"n": 0}

    def make():
        n = int(round(per_s * RATE))
        i = np.arange(state["n"], state["n"] + n)
        state["n"] += n
        return np.repeat((amp * np.sin(2 * np.pi * freq * i / RATE)).astype(np.int16), channels).tobytes()
    return make


def trace(name: str, channels: int, steps) -> dict:
    """Runs a DelayLine through `steps` ((t, "in", pcm) / (t, "out", frames) / (None, "delay", s)), saving it all."""
    from framegen.audio import DelayLine
    clock = _Clock()
    line = DelayLine(RATE, channels, clock=clock)
    ops, pin, pout = [], bytearray(), bytearray()
    for t, what, arg in steps:
        if what == "delay":
            line.set_delay(arg)
            ops.append(["delay", arg])
            continue
        clock.t = t
        if what == "in":
            ops.append(["in", t, len(pin) // 2, len(arg) // 2])
            pin += arg
            line.push(arg)
        else:
            b = line.pull(arg * channels * 2)
            ops.append(["out", t, arg])
            pout += b
    (OUT / f"delay_{name}.in.s16").write_bytes(bytes(pin))
    (OUT / f"delay_{name}.out.s16").write_bytes(bytes(pout))
    info = {"rate": RATE, "channels": channels, "ops": ops, "stats": line.stats(), "kept_s": line.kept_s,
            "queued_s": line.queued_s, "jumps": line.jumps, "underruns": line.underruns}
    (OUT / f"delay_{name}.json").write_text(json.dumps(info))
    print(f"[fixtures] delay_{name}: {len(ops)} steps, {line.stats()}")
    return {"channels": channels, "stats": line.stats()}


def traces(manifest: dict) -> None:
    out = {}
    blk = int(0.01 * RATE)
    # hold-ups reading the card (test_hold_ups_reading_the_card_leave_no_gaps), a 220 Hz tone
    tone = _sine(CH)
    out["holdups"] = trace("holdups", CH, [(None, "delay", 0.015)] + [
        (t, w, tone() if w == "in" else blk) for t, w in _card(20, 40, every_s=4)])
    # a jump down, a gap, and back (test_jumps_and_gaps_are_eased_so_they_dont_click)
    tone = _sine(CH)
    ev = _card(6)
    steps = [(None, "delay", 0.3)] + [(t, w, tone() if w == "in" else blk) for t, w in ev if t < 3]
    steps += [(None, "delay", 0.0)] + [(t, w, tone() if w == "in" else blk) for t, w in ev if 3 <= t < 4]
    steps += [(t, "out", blk) for t, w in ev if 4 <= t < 6 and w == "out"]
    steps += [(6 + t, w, tone() if w == "in" else blk) for t, w in _card(2)]
    out["eased"] = trace("eased", CH, steps)
    # the picture's delay changing (test_a_change_in_the_pictures_delay_is_eased_not_jumped)
    tone = _sine(CH, freq=330)
    ev = _card(30)
    steps = [(None, "delay", 0.04)] + [(t, w, tone() if w == "in" else blk) for t, w in ev if t < 5]
    steps += [(None, "delay", 0.12)] + [(t, w, tone() if w == "in" else blk) for t, w in ev if 5 <= t < 20]
    steps += [(None, "delay", 0.04)] + [(t, w, tone() if w == "in" else blk) for t, w in ev if t >= 20]
    out["change"] = trace("change", CH, steps)
    # the card's clock 0.2% fast, then slow (test_the_delay_holds_when_the_card_clock_runs_fast_or_slow), mono
    for name, speed in (("fast", 1.002), ("slow", 0.998)):
        tone = _sine(1, freq=440, per_s=0.01)
        steps, got = [(None, "delay", 0.05)], 0.0
        for i in range(1500):
            got += 0.01 * speed
            n = int(round((round(got, 4) - round(got - 0.01 * speed, 4)) * RATE))
            pcm = tone()[: n * 2] if n <= blk else tone() + tone()[: (n - blk) * 2]
            steps += [(i * 0.01, "in", pcm), (i * 0.01, "out", blk)]
        out[name] = trace(name, 1, steps)
    # filling: silence until the delay, then the sound in order (test_silence_until_the_delay_then_the_sound_in_order)
    steps = [(None, "delay", 0.1)]
    for i in range(30):
        steps += [(i * 0.01, "in", np.full(blk * CH, i + 1, np.int16).tobytes()), (i * 0.01, "out", blk)]
    out["fill"] = trace("fill", CH, steps)
    # odd block sizes, as Core Audio asks when the card's rate isn't the output's (470/471 frames, say)
    tone = _sine(CH, freq=1000)
    rnd = random.Random(5)
    steps = [(None, "delay", 0.03)]
    t_in = t_out = 0.0
    while t_in < 8:
        if t_in <= t_out:
            steps.append((t_in, "in", tone()))
            t_in += 0.02 + abs(rnd.gauss(0, 0.003))
        else:
            n = rnd.choice((470, 471, 512, 1, 2, 4096))
            steps.append((t_out, "out", n))
            t_out += n / RATE
    out["odd"] = trace("odd", CH, steps)
    manifest["delay"] = out


# ----------------------------------------------------------------------------- picks
def picks(manifest: dict) -> None:
    from framegen import audio as au
    from framegen import capture as c
    rnd = random.Random(7)
    fmts = ["nv12", "yuyv422", "uyvy422", "mjpeg", "h264", "bgr0", "yuv420p", "p010", "hevc"]
    sizes = [(1920, 1080), (1280, 720), (3840, 2160), (640, 480), (2560, 1440), (720, 480), (1024, 768)]
    rates = [60.0, 59.94, 30.0, 29.97, 120.0, 50.0, 25.0, 24.0, 15.0, 7.5]
    cases = []
    tests_options = c.parse_modes(open(ROOT / "tests" / "test_capture.py").read().split("OPTIONS = '''")[1]
                                  .split("'''")[0])
    lists = [tests_options, [], [c.Mode("h264", False, 1920, 1080, 60, 60)],
             [c.Mode("nv12", True, 1280, 720, 30.0, 30.0), c.Mode("nv12", True, 640, 480, 30.0, 30.0)]]
    for _ in range(300):
        modes = []
        for _ in range(rnd.randint(1, 8)):
            f = rnd.choice(fmts)
            w, h = rnd.choice(sizes)
            hi = rnd.choice(rates)
            lo = hi if rnd.random() < 0.7 else rnd.choice([r for r in rates if r <= hi])
            m = c.Mode(f, f not in ("mjpeg", "h264", "p010", "hevc"), w, h, lo, hi, rnd.choice(["", "tv", "pc"]),
                       rnd.choice(["", "bt709"]))
            if m not in modes:
                modes.append(m)
        lists.append(modes)
    asks = [(1920, 1080, 60.0), (1280, 720, 60.0), (3840, 2160, 30.0), (1920, 1080, 120.0), (2560, 1440, 59.94),
            (800, 600, 30.0)]
    for modes in lists:
        for w, h, fps in (asks if len(cases) < 40 else [rnd.choice(asks)]):
            best, note = c.pick_mode(modes, w, h, fps)
            # (sizes of equal area: Python's set order picks; the Rust takes the first in the card's order)
            areas = [m.w * m.h for m in modes]
            if len(set(areas)) != len(set((m.w, m.h) for m in modes)):
                continue
            cases.append({"modes": [m.__dict__ for m in modes], "w": w, "h": h, "fps": fps,
                          "pick": best.__dict__ if best else None, "note": note,
                          "label": best.label() if best else None})
    D = c.Device
    groups = [
        [D("USB3.0 Capture", "video"), D("Digital Audio Interface (USB3.0 Capture)", "audio"),
         D("Microphone (Realtek Audio)", "audio"), D("Cam Link 4K", "video"), D("Cam Link 4K", "audio"),
         D("Game Capture HD60 X", "video"), D("Game Capture HD60 X Audio", "audio")],
        [D("Elgato 4K X", "video"), D("Line (Elgato 4K X)", "audio"), D("Line (Elgato Wave)", "audio")],
        [D("Live Gamer Ultra 2.1-Video", "video"), D("HDMI/Line In (Live Gamer Ultra 2.1-Audio)", "audio"),
         D("Virtual Microphone (AVerMedia Streaming Center)", "audio"),
         D("Not To Use (Live Gamer Ultra 2.1-Audio)", "audio")],
        [D("Live Gamer Ultra 2.1-Video", "video"), D("Virtual Microphone (AVerMedia Streaming Center)", "audio"),
         D("Not To Use (Live Gamer Ultra 2.1-Audio)", "audio")],
        [D("MacBook Pro Camera", "video"), D("MacBook Pro Microphone", "audio"), D("Cam Link 4K", "video"),
         D("Cam Link 4K", "audio"), D("Game Capture HD60 S+", "video"), D("Game Capture HD60 S+", "audio"),
         D("USB Video", "video"), D("USB Audio", "audio"), D("UGREEN 25854", "video"),
         D("UGREEN 25854 Audio", "audio"), D("Capture Card Pro", "audio"), D("Pro Capture Card", "video")],
    ]
    audio_cases = []
    for devs in groups:
        for v in [d.name for d in devs if d.kind == "video"] + ["USB Video", "Elgato", "HD60", "Unknown Card"]:
            audio_cases.append({"video": v, "devices": [d.__dict__ for d in devs], "pick": c.pick_audio(v, devs)})
    stats = [{"ran_dry": 3, "output_short": 1, "skips": 0, "refills": 1, "stops": 0, "longest_wait_ms": 46,
              "kept_ms": 56, "longest_call_ms": 31, "output_ms": 40},
             {"ran_dry": 0, "output_short": 0, "skips": 2, "refills": 0, "stops": 4, "longest_wait_ms": 21,
              "kept_ms": 31}]
    reports = []
    for s in stats:
        reports.append({"stats": s, "since": None, "line": au.sound_report(s)})
        reports.append({"stats": s, "since": dict(s), "line": au.sound_report(s, dict(s))})
        reports.append({"stats": {**s, "ran_dry": s["ran_dry"] + 2, "stops": s["stops"] + 1}, "since": s,
                        "line": au.sound_report({**s, "ran_dry": s["ran_dry"] + 2, "stops": s["stops"] + 1}, s)})
        reports.append({"stats": s, "since": {}, "line": au.sound_report(s, {})})
    (OUT / "picks.json").write_text(json.dumps({"modes": cases, "audio": audio_cases, "reports": reports}))
    manifest["picks"] = {"modes": len(cases), "audio": len(audio_cases), "reports": len(reports)}
    print(f"[fixtures] picks.json: {len(cases)} pick_mode cases, {len(audio_cases)} pick_audio, {len(reports)} reports")


def check_devices() -> int:
    from framegen import capture as c
    py = c.avfoundation_devices()
    exe = ROOT / "mac" / "core" / "target-capture" / "debug" / "examples" / "capture"
    if not exe.exists():
        print(f"[devices] build sp-capture's example first ({exe})")
        return 1
    rust = json.loads(subprocess.run([str(exe), "--list", "--json"], capture_output=True, text=True).stdout)
    for kind in ("video", "audio"):
        a = sorted(d.name for d in py if d.kind == kind)
        b = sorted(d["name"] for d in rust[kind])
        print(f"[devices] {kind}: FFmpeg {a}; sp-capture {b}" + ("" if a == b else "  <- differ"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--skip-onset", action="store_true")
    ap.add_argument("--check-devices", action="store_true")
    a = ap.parse_args()
    if a.check_devices:
        return check_devices()
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    if not a.skip_frames:
        clip_frames(manifest)
    if not a.skip_onset:
        onset(manifest)
    traces(manifest)
    picks(manifest)
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[fixtures] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
