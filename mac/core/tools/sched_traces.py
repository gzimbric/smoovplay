"""Parity traces for sp-sched: drives the Python pipeline (framegen/) with deterministic inputs and records what it
decided, tick by tick, to JSON in mac/core/fixtures/sched/. tests/parity.rs replays the same inputs through the Rust
pipeline and must come out the same.

Deterministic: the pipeline's clock (time.perf_counter / time.sleep in framegen.pipeline and framegen.delaytest) is
replaced by a fake one that moves only when something says so: a tick arriving (the source moves it to the tick's
arrival time, as a live source waiting for a frame), the stand-in interpolator and presenter (each costs a fixed
time), and the pacer's sleeps. Stand-ins as tests/ uses them: a source yielding Ticks, an interpolator whose frames
say which two frames and which fraction they are, a presenter recording what it was given, chiaki-ng for the delay
test and the sound, instant replay.

Frames carry a .tag (a number) so the trace can say which frame was shown. Pictures come from a small integer recipe
(content(), reproduced exactly in crates/sp-sched/tests/common/mod.rs) so both sides see the same pixels; "tag"
scenarios use 8x8 float frames whose value is the tag, as tests/test_dips.py does.

Run: .venv/bin/python mac/core/tools/sched_traces.py   (writes fixtures/sched/*.json; about a minute)
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import sys
import time as _time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "sched"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from framegen import delaytest as dt  # noqa: E402
from framegen import pipeline as pl  # noqa: E402
from framegen.io import MiniText, Tick, Waiting  # noqa: E402
from framegen.settings import Settings  # noqa: E402
from framegen.sources import TestMode  # noqa: E402

START = 1000.0
TICK = 1 / 60


class FakeClock:
    def __init__(self, t: float = START) -> None:
        self.t = t

    def perf_counter(self) -> float:
        return self.t

    def sleep(self, d: float) -> None:
        if d > 0:
            self.t += d

    def advance(self, d: float) -> None:
        self.t += d

    def at_least(self, t: float) -> None:
        if t > self.t:
            self.t = t


# ------------------------------------------------------------------------------------------------ pictures
def h32(*xs):
    """FNV-1a over 32-bit words, on numpy arrays (uint64 holding 32-bit values)."""
    v = np.uint64(2166136261)
    for x in xs:
        v = ((v ^ (np.asarray(x, dtype=np.uint64) & np.uint64(0xFFFFFFFF))) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    return v


def content(c: dict, w: int, h: int) -> np.ndarray:
    """HxWx3 uint8 BGR from a recipe: {"s": scene, "k": where the square is, "noise": seed or -1, "amp": noise
    amplitude, "add": brightness added, "hud": a still label with text}."""
    if c.get("kind") == "menu":
        f = np.full((h, w, 3), 60, np.uint8)
        if c["open"]:
            f[1:h - 1, 1:w - 1] = 200
        return f
    ys, xs = np.mgrid[0:h, 0:w]
    base = (30 + h32(xs // 16, ys // 16, c["s"]) % np.uint64(150)).astype(np.int64)
    img = np.stack([base + 20 * ch for ch in range(3)], -1)
    x0 = (5 * c["k"]) % (w - 12)
    y0 = h // 2 - 6
    img[y0:y0 + 12, x0:x0 + 12] = 250
    if c.get("hud"):
        img[8:24, 8:56] = 255
        for x in (12, 24, 36):
            img[12:20, x:x + 4] = 0
    img = img + c.get("add", 0)
    if c.get("noise", -1) >= 0:
        amp = c.get("amp", 2)
        n = (h32(xs, ys, c["noise"] + 1000) % np.uint64(2 * amp + 1)).astype(np.int64) - amp
        img = img + n[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def tag_of(x):
    t = getattr(x, "tag", None)
    return None if t is None else float(t)


def fid(frame) -> dict:
    """Which frame this is, for the trace."""
    t = getattr(frame, "tag", None)
    if isinstance(t, tuple):
        return {"ai": [t[1], t[2], t[3]]}
    return {"tag": None if t is None else float(t)}


def num(x):
    """A float for JSON (inf and nan as strings, which JSON can't hold)."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


# ------------------------------------------------------------------------------------------------ stand-ins
class Interp:
    device = "cpu"
    private = False

    def __init__(self, clock, ev, w, h, kind, cost_interp, cost_prepare):
        self.clock, self.ev, self.kind = clock, ev, kind
        self.h, self.w = h, w
        self.cost_interp, self.cost_prepare = cost_interp, cost_prepare
        self.in_u8 = self.out_u8 = torch.zeros(h, w, 3, dtype=torch.uint8)
        self.slots, self.px = [None, None], [None, None]

    def load(self, frame):
        self.in_u8 = frame

    def prepare(self, slot):
        self.slots[slot], self.px[slot] = tag_of(self.in_u8), self.in_u8
        self.clock.advance(self.cost_prepare)

    def interpolate(self, slot, t=0.5):
        a, b = self.slots[1 - slot], self.slots[slot]
        self.clock.advance(self.cost_interp)
        if self.kind == "blend8" and self.px[0] is not None and self.px[1] is not None:
            out = ((self.px[1 - slot].int() + self.px[slot].int() + 1) // 2).to(torch.uint8)
        else:
            out = torch.zeros(self.h, self.w, 3, dtype=torch.uint8)
        out.tag = ("ai", a, b, float(t))
        self.out_u8 = out
        return out

    def set_steady(self, threshold):
        self.ev.append(["steady", float(threshold)])


def overlay_json(text):
    if isinstance(text, MiniText):
        return {"text": str(text), "parts": [list(p) for p in text.parts], "health": text.health,
                "second": text.second, "minimal": True}
    return {"text": text, "parts": [], "health": "", "second": "", "minimal": False}


class Sink:
    gpu = False

    def __init__(self, clock, ev, settings, cost_write, cmds, wait_cmds):
        self.clock, self.ev, self.settings = clock, ev, settings
        self.cost_write, self.cmds, self.wait_cmds = cost_write, cmds, wait_cmds
        self.n = self.nw = 0

    def write(self, frame, text=None):
        i = self.n
        self.n += 1
        self.clock.advance(self.cost_write)
        on = self.settings is None or self.settings.overlay
        self.ev.append(["write", self.clock.t, fid(frame), overlay_json(text) if on else None])
        return self.cmds.get(str(i))

    def wait(self, w):
        i = self.nw
        self.nw += 1
        self.ev.append(["wait", w.since, w.what, w.why])
        return self.wait_cmds.get(str(i))

    def toast(self, text, seconds=2.0):
        self.ev.append(["toast", text, float(seconds)])


class Remote:
    """chiaki-ng: presses go out send_s after they're asked for; the game toggles its menu answer_s later."""

    def __init__(self, clock, ev, answer_s, send_s, answers, sends, press):
        self.clock, self.ev = clock, ev
        self.answer_s, self.send_s, self.answers, self.sends = answer_s, send_s, answers, sends
        self.asked, self.sent = [], []
        if not press:
            self.press = None  # (hasattr(remote, "press") stays true; can_press says no)
        self._can = press

    def can_press(self):
        return self._can

    def can_delay_audio(self):
        return True

    def press(self, buttons, hold_ms=100):
        self.ev.append(["press", buttons, hold_ms])
        self.asked.append(self.clock.t)
        return len(self.asked) - 1

    def press_sent(self, token):
        t = self.asked[token] + self.send_s
        if not self.sends or self.clock.t < t:
            return None
        if len(self.sent) <= token:
            self.sent.append(t)
        return self.sent[token]

    def set_audio_delay(self, ms):
        self.ev.append(["audio", ms])

    def menu_open(self, t):
        return self.answers and sum(t >= s + self.answer_s for s in self.sent) % 2 == 1


class CardRemote:
    """A capture card's sound, which the pipeline can hold back, but no buttons to press (no press at all)."""

    def __init__(self, ev):
        self.ev = ev

    def can_press(self):
        return False

    def can_delay_audio(self):
        return True

    def set_audio_delay(self, ms):
        self.ev.append(["audio", ms])

    def menu_open(self, t):
        return False


class Replay:
    def __init__(self, ev):
        self.ev, self.frames, self.real = ev, 0, 0

    def set_output_hz(self, hz):
        self.ev.append(["replay_hz", hz])

    def set_delay(self, ms):
        self.ev.append(["replay_delay", float(ms)])

    def frame(self, frame, t=None):
        self.frames += 1

    def real_frame(self, frame, t):
        self.real += 1

    def save(self, kind=None):
        self.ev.append(["replay_save", kind])


class LogLines(io.TextIOBase):
    """stdout while the pipeline runs: each "[pipeline] ..." line goes into the trace where it happened."""

    def __init__(self, ev):
        self.ev, self.buf = ev, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.ev.append(["log", line])
        return len(s)


SETTING_KEYS = ("interp", "hz", "simulate_30", "steady_hud", "overlay", "overlay_style", "even_out", "even_dips",
                "game_fps", "audio_sync")


# ------------------------------------------------------------------------------------------------ one run
def run(name: str, cfg: dict, ticks: list) -> dict:
    clock = FakeClock(START)
    ev: list = []
    w, h = cfg.get("size", (128, 96))
    settings = None
    if cfg.get("settings") is not None:
        settings = Settings({k: v for k, v in cfg["settings"].items() if k != "simulate_30"})
        for k, v in cfg["settings"].items():
            settings.override(k, v)
    remote = None
    if cfg.get("remote") is not None:
        r = cfg["remote"]
        remote = (CardRemote(ev) if r.get("card") else
                  Remote(clock, ev, r.get("answer_s", 0.1), r.get("send_s", 0.004), r.get("answers", True),
                         r.get("sends", True), r.get("press", True)))
    replay = Replay(ev) if cfg.get("replay") else None
    tag_probe = cfg.get("probe") == "tag"

    def source():
        for i, s in enumerate(ticks):
            for k, v in s.get("set", {}).items():
                settings.override(k, v)
            at = START + s["at"]
            clock.at_least(at)
            if "waiting" in s:
                yield Waiting(*s["waiting"])
                continue
            if tag_probe:
                frame = torch.full((h, w, 3), float(s["tag"]))
            else:
                c = s["c"]
                if c.get("kind") == "menu":
                    c = {**c, "open": remote is not None and remote.menu_open(at)}
                    ev.append(["menu", c["open"]])
                frame = torch.from_numpy(content(c, w, h))
            frame.tag = s["tag"]
            yield Tick(frame, at, s.get("index", i), content=s.get("content"), truth_new=s.get("truth"),
                       new_hint=s.get("hint"), damaged=s.get("damaged", False),
                       t_ready=START + s["ready"] if "ready" in s else None, extra=dict(s.get("extra", {})))

    src = source()
    gt = None
    if cfg.get("testmode"):
        tm = TestMode(settings.simulate_30)
        settings.watch(lambda key: tm.set_on(settings.simulate_30) if key == "simulate_30" else None)
        src, gt = tm.wrap(src), tm.gt
    sink = None
    if cfg.get("sink", True):
        s = cfg.get("sink_cfg", {})
        sink = Sink(clock, ev, settings, s.get("cost_write", 0.0), s.get("cmds", {}), s.get("wait_cmds", {}))
    ic = cfg.get("interp", {})
    interp = Interp(clock, ev, w, h, ic.get("kind", "tag"), ic.get("cost_interp", 0.0), ic.get("cost_prepare", 0.0))
    d = cfg.get("detector", {})
    p = pl.Pipeline.__new__(pl.Pipeline)
    p.interp, p.cut_threshold, p.stats = interp, 25.0, None
    p.detector = pl.DuplicateDetector(d.get("threshold", 2.0), reference=d.get("reference", "previous"),
                                      learn=d.get("learn", False))
    p.beat, p.waiting = clock.t, None

    def on_tick(tick, diff, action, proc_ms, t0):
        ev.append(["tick", tick.index, action.value, bool(diff.is_new), num(diff.max_diff), num(diff.mean_diff),
                   num(proc_ms), num(t0)])

    tuning = cfg.get("tuning", {})
    # The generated frame as the stand-in made it (with its tag), not copied into pinned memory.
    pl.Pipeline._generated = lambda self, frame, gpu_out: frame
    saved = (pl.RATE_NOTE_S, pl.RATE_TOAST_S, dt.SETTLE_S, dt.GAP_S, pl.time, dt.time)
    pl.RATE_NOTE_S = tuning.get("rate_note_s", pl.RATE_NOTE_S)
    pl.RATE_TOAST_S = tuning.get("rate_toast_s", pl.RATE_TOAST_S)
    dt.SETTLE_S = tuning.get("delay_settle_s", dt.SETTLE_S)
    dt.GAP_S = tuning.get("delay_gap_s", dt.GAP_S)
    fake = SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep)
    pl.time = dt.time = fake
    r = cfg.get("run", {})
    try:
        with contextlib.redirect_stdout(LogLines(ev)):
            st = p.run(src, sink, enabled=r.get("enabled", True), score=r.get("score", False),
                       label=r.get("label", "RIFE"), on_tick=on_tick, hz=r.get("hz", 60), settings=settings, gt=gt,
                       remote=remote, replay=replay)
    finally:
        pl.RATE_NOTE_S, pl.RATE_TOAST_S, dt.SETTLE_S, dt.GAP_S, pl.time, dt.time = saved

    lists = {k: [num(x) for x in getattr(st, k)] for k in (
        "proc_interp_ms", "proc_other_ms", "sink_ms", "queue_ms", "score_ms", "busy_ms", "real_delay_ms",
        "psnr_interp", "psnr_repeat", "psnr_blend", "off_interp", "off_repeat", "psnr_still", "psnr_still_repeat",
        "still_frac", "psnr_hud", "hud_off", "hud_frac", "new_gaps_ms")}
    counts = {k: getattr(st, k) for k in (
        "quarter_shown", "quarter_skipped", "behind", "simulated", "test_game_new", "test_game_repeat",
        "uneven_ticks", "uneven_waits", "dip_ticks", "stale")}
    return {
        "name": name,
        "config": cfg,
        "ticks": ticks,
        "events": ev,
        "stats": {
            "lists": lists, "counts": counts, "actions": [[k, v] for k, v in st.actions.items()],
            "dedup": [[k, v] for k, v in st.dedup.items()], "hz_used": sorted(st.hz_used),
            "uneven_fps": list(st.uneven_fps), "delay_tests": list(st.delay_tests), "exit": st.exit,
            "wall_s": st.wall_s, "worst_psnr": num(st.worst[0]) if st.worst else None,
            "worst_tick": st.worst[1] if st.worst else None,
        },
        "summary": st.summary(),
        "results": st.results(),
        "results_no_full_motion": st.results(full_motion=False),
        "settings_after": None if settings is None else {k: getattr(settings, k) for k in SETTING_KEYS},
        "replay_counts": None if replay is None else [replay.frames, replay.real],
        "_stats": st,
    }


# ------------------------------------------------------------------------------------------------ scenarios
def game(pattern, n, scene=0, hint=False, noise=-1, amp=2, hud=False, cut_every=0, damaged=(), period=TICK,
         start_k=0):
    """n ticks of a game on a 60 Hz stream; pattern(i) says if tick i shows a new frame. The picture moves on each
    new frame (k counts them); with noise, each tick has its own noise (a decoded stream's repeats aren't identical)."""
    out, k = [], start_k - 1
    for i in range(n):
        new = bool(pattern(i))
        k += new
        s = scene + (k // cut_every if cut_every else 0)
        c = {"s": s, "k": k, "noise": (i if noise >= 0 else -1), "amp": amp, "hud": hud}
        t = {"at": (i + 1) * period, "tag": float(k), "c": c}
        if hint:
            t["hint"] = new
        if i in damaged:
            t["damaged"] = True
        out.append(t)
    return out


def dipping_game(frames=150, held3=0.3, seed=1, hold=3, steady=0):
    """tests/test_dips.py _dipping_game: values are the game's time in ticks, with Remote Play's sizes and numbers."""
    rnd = random.Random(seed)
    out, i, t = [], 0, 0
    for f in range(frames):
        ticks = hold if f >= steady and rnd.random() < held3 else 2
        for k in range(ticks):
            out.append({"at": (i + 1) * TICK, "tag": float(t), "hint": k == 0,
                        "extra": {"size": 9000 if k == 0 else 300, "seq": i}})
            i += 1
        t += ticks
    return out


def game_times(fps, ticks, phase=0.37, jitter=0.0, seed=0):
    """tests/test_timeline.py _game_times: {arrival tick: the game time of the frame shown from it}."""
    rng = np.random.default_rng(seed)
    t, arr = phase, {}
    while t < ticks:
        arr[math.ceil(t)] = t
        t += 60 / fps * (1 + jitter * rng.normal())
    return arr


def timed_stream(arr, ticks):
    out, k = [], -1
    for n in range(ticks):
        new = n in arr
        k += new
        out.append({"at": (n + 1) * TICK, "tag": float(k), "hint": new, "c": {"s": 3, "k": k}})
    return out


def menu_stream(seconds, period=TICK):
    return [{"at": (i + 1) * period, "tag": float(i), "hint": i % 2 == 0, "c": {"kind": "menu"}}
            for i in range(int(seconds / period))]


def scenarios():
    S = {}
    det = {"overlay_style": "detailed"}
    S["p30_steady"] = ({"settings": {"hz": 60, **det}}, game(lambda i: i % 2 == 0, 300))
    S["p30_steady_minimal"] = ({"settings": {"hz": 60}, "interp": {"cost_interp": 0.004}},
                               game(lambda i: i % 2 == 0, 300))
    hitch = lambda i: (i % 2 == 0 and not 100 <= i < 103) or i == 103 or (200 <= i < 230)  # noqa: E731
    S["p30_hitches"] = ({"settings": {"hz": 60, **det}, "interp": {"cost_interp": 0.003, "cost_prepare": 0.001}},
                        game(hitch, 400))
    S["p60_passthrough"] = ({"settings": {"hz": 60, **det, "even_out": False}}, game(lambda i: True, 800))
    S["p60_minimal"] = ({"settings": {"hz": 60}}, game(lambda i: True, 800))
    S["cuts"] = ({"settings": {"hz": 60, **det}}, game(lambda i: i % 2 == 0, 400, cut_every=20, damaged=(151, 152)))
    S["hz120"] = ({"settings": {"hz": 120, "even_out": False, **det}, "interp": {"cost_interp": 0.002},
                   "replay": True}, game(lambda i: i % 2 == 0, 240))
    S["hz120_behind"] = ({"settings": {"hz": 120, "even_out": False}, "interp": {"cost_interp": 0.009},
                          "sink_cfg": {"cost_write": 0.001}}, game(lambda i: i % 2 == 0, 300))
    S["hz90"] = ({"settings": {"hz": 90, "even_out": False, **det}, "interp": {"cost_interp": 0.003}},
                 game(lambda i: i % 2 == 0, 240))
    S["hz90_behind"] = ({"settings": {"hz": 90, "even_out": False}, "interp": {"cost_interp": 0.015},
                         "sink_cfg": {"cost_write": 0.005}}, game(lambda i: i % 2 == 0, 300))
    S["p30_hitches_120"] = ({"settings": {"hz": 120, "even_out": False, **det}, "interp": {"cost_interp": 0.002}},
                            game(hitch, 400))
    S["p60_120"] = ({"settings": {"hz": 120, "even_out": False}}, game(lambda i: True, 300))
    for hz in (60, 90, 120):
        S[f"even45_{hz}"] = ({"settings": {"hz": hz, **det}, "interp": {"cost_interp": 0.001}},
                             game(lambda i: i % 4 != 3, 600, hint=True))
    S["even_jitter50_90"] = ({"settings": {"hz": 90}}, timed_stream(game_times(50, 700, jitter=0.1), 700))
    S["even_rate_change_120"] = ({"settings": {"hz": 120}}, timed_stream(
        {math.ceil(t): t for t in _changing_rate()}, 900))
    S["even_off_45"] = ({"settings": {"hz": 60, "even_out": False, **det}}, game(lambda i: i % 4 != 3, 600, hint=True))
    S["dips_auto_90"] = ({"probe": "tag", "size": (8, 8), "settings": {"hz": 90, "even_dips": True, "overlay": False}},
                         dipping_game(300, 0.15, hold=4, steady=40))
    S["dips_auto_120"] = ({"probe": "tag", "size": (8, 8),
                           "settings": {"hz": 120, "even_dips": True, "overlay": False}},
                          dipping_game(300, 0.15, hold=3, steady=40))
    S["dips_lock30_60"] = ({"probe": "tag", "size": (8, 8),
                            "settings": {"hz": 60, "even_dips": True, "game_fps": "30", **det}},
                           dipping_game(300, 0.15, hold=4, steady=40))
    S["lock30_dips_off"] = ({"probe": "tag", "size": (8, 8),
                             "settings": {"hz": 90, "even_out": False, "game_fps": "30", "overlay": False}},
                            dipping_game(150, 0.3))
    S["testmode_60"] = ({"settings": {"hz": 60, "simulate_30": True, **det}, "testmode": True,
                         "interp": {"kind": "blend8", "cost_interp": 0.005}},
                        game(lambda i: True, 400, hud=True, hint=True))
    S["testmode_120_minimal"] = ({"settings": {"hz": 120, "simulate_30": True, "even_out": False}, "testmode": True,
                                  "interp": {"kind": "blend8", "cost_interp": 0.003}},
                                 game(lambda i: True, 400, hud=True, hint=True))
    S["testmode_90"] = ({"settings": {"hz": 90, "simulate_30": True, **det}, "testmode": True,
                         "interp": {"kind": "blend8", "cost_interp": 0.003}},
                        game(lambda i: True, 300, hud=True, hint=True))
    S["testmode_30_game_off"] = ({"settings": {"hz": 60, "simulate_30": True, **det}, "testmode": True,
                                  "interp": {"kind": "blend8"}},
                                 game(lambda i: i % 2 == 0, 1000, hud=True, hint=True))
    S["testmode_60_then_30"] = ({"settings": {"hz": 60, "simulate_30": True}, "testmode": True,
                                 "interp": {"kind": "blend8"}},
                                game(lambda i: i < 300 or i % 2 == 0, 1200, hud=True, hint=True))
    S["score_flag"] = ({"settings": {"hz": 60, **det}, "run": {"score": True}, "testmode": True,
                        "interp": {"kind": "blend8"}}, _turn_on(game(lambda i: True, 200, hud=True), 0))
    S["simulate_on_mid"] = ({"settings": {"hz": 60, **det}, "testmode": True, "interp": {"kind": "blend8"}},
                            _toggles(game(lambda i: True, 500, hud=True, hint=True),
                                     {100: {"simulate_30": True}, 350: {"simulate_30": False}}))
    S["toggles"] = ({"settings": {"hz": 60, **det}, "interp": {"cost_interp": 0.004}},
                    _toggles(game(lambda i: i % 2 == 0, 700), {
                        50: {"steady_hud": True}, 100: {"hz": 120}, 160: {"interp": False}, 200: {"interp": True},
                        260: {"hz": 90}, 300: {"overlay_style": "minimal"}, 340: {"overlay": False},
                        380: {"overlay": True, "steady_hud": False}, 420: {"hz": 60}, 500: {"game_fps": "30"},
                        600: {"game_fps": "auto"}}))
    S["learn_window"] = ({"settings": {"hz": 60, **det}, "detector": {"learn": True}},
                         game(lambda i: i % 2 == 0 and not 150 <= i < 153, 400, noise=0, amp=20))
    S["learn_window_amp8"] = ({"settings": {"hz": 60}, "detector": {"learn": True}},
                              game(lambda i: i % 2 == 0 and not 150 <= i < 153, 400, noise=0, amp=8))
    S["capture_new_ref"] = ({"settings": {"hz": 60, **det}, "detector": {"reference": "new"}}, _drift(300))
    S["prev_ref_drift"] = ({"settings": {"hz": 60, **det}}, _drift(300))
    S["headless"] = ({"sink": False, "run": {"hz": 120}}, game(lambda i: i % 2 == 0, 200))
    S["no_settings_120"] = ({"run": {"hz": 120, "label": "RIFE 4.22"}, "interp": {"cost_interp": 0.002}},
                            game(lambda i: i % 2 == 0, 240))
    S["no_settings_off"] = ({"run": {"enabled": False, "label": "X"}}, game(lambda i: i % 2 == 0, 200))
    S["delay_test"] = ({"settings": {"hz": 60, **det}, "size": (8, 8), "remote": {"answer_s": 0.1},
                        "sink_cfg": {"cmds": {"30": "delay_test"}}}, menu_stream(14.0))
    S["delay_test_fast_minimal"] = ({"settings": {"hz": 90}, "size": (8, 8), "remote": {"answer_s": 0.05},
                                     "sink_cfg": {"cmds": {"5": "delay_test"}},
                                     "tuning": {"delay_settle_s": 0.1, "delay_gap_s": 0.1}}, menu_stream(4.0))
    S["delay_test_no_answer"] = ({"settings": {"hz": 60}, "size": (8, 8), "remote": {"answers": False},
                                  "sink_cfg": {"cmds": {"3": "delay_test"}}}, menu_stream(8.0))
    S["delay_test_not_sent"] = ({"settings": {"hz": 60}, "size": (8, 8), "remote": {"sends": False},
                                 "sink_cfg": {"cmds": {"3": "delay_test"}}}, menu_stream(4.0))
    S["delay_test_no_remote"] = ({"settings": {"hz": 60}, "size": (8, 8), "sink_cfg": {"cmds": {"3": "delay_test"}}},
                                 menu_stream(1.0))
    S["delay_test_cant_press"] = ({"settings": {"hz": 60}, "size": (8, 8), "remote": {"press": False},
                                   "sink_cfg": {"cmds": {"3": "delay_test"}}}, menu_stream(1.0))
    S["delay_test_card"] = ({"settings": {"hz": 60}, "size": (8, 8), "remote": {"card": True},
                             "sink_cfg": {"cmds": {"3": "delay_test"}}}, menu_stream(3.0))
    S["audio_sync_off"] = ({"settings": {"hz": 60, "audio_sync": False}, "size": (8, 8), "remote": {}},
                           menu_stream(5.0))
    S["waiting_quit"] = ({"settings": {"hz": 60}, "sink_cfg": {"wait_cmds": {"2": "quit"}}},
                         _stale(game(lambda i: i % 2 == 0, 60)) + [
                             {"at": 1.1 + 0.1 * j, "waiting": [0.1 * (j + 1), "the PS5", "hint"]} for j in range(5)])
    S["restart_cmd"] = ({"settings": {"hz": 60}, "sink_cfg": {"cmds": {"40": "restart"}}},
                        game(lambda i: i % 2 == 0, 100))
    S["save_cmds"] = ({"settings": {"hz": 60}, "replay": True,
                       "sink_cfg": {"cmds": {"10": "save_replay", "20": "save_compare"}}},
                      game(lambda i: i % 2 == 0, 300))
    S["save_cmds_no_replay"] = ({"settings": {"hz": 60}, "sink_cfg": {"cmds": {"10": "save_replay"}}},
                                game(lambda i: i % 2 == 0, 60))
    S["queued_behind_60"] = ({"settings": {"hz": 60, **det}, "interp": {"cost_interp": 0.02}},
                             _ready(game(lambda i: i % 2 == 0, 200)))
    return S


def _changing_rate():
    t, out = 0.3, []
    while t < 900:
        out.append(t)
        t += 60 / (45 + 10 * t / 900)
    return out


def _toggles(ticks, at):
    for i, changes in at.items():
        ticks[i] = {**ticks[i], "set": changes}
    return ticks


def _turn_on(ticks, i):
    ticks[i] = {**ticks[i], "set": {"simulate_30": True}}
    for t in ticks:
        t["hint"] = True
    return ticks


def _drift(n):
    """A still scene that brightens a level a tick, with a new frame every 30 ticks."""
    out, k = [], 0
    for i in range(n):
        k += i % 30 == 0 and i > 0
        out.append({"at": (i + 1) * TICK, "tag": float(i), "c": {"s": 5, "k": k * 3, "add": (i % 30)}})
    return out


def _stale(ticks):
    """Three ticks queued before the run started (a restart), then the stream."""
    return [{"at": -0.05 + 0.01 * j, "tag": -1.0 - j, "c": {"s": 9, "k": j}} for j in range(3)] + ticks


def _ready(ticks):
    """Remote Play: each frame ready (decoded) 5 ms after it arrived."""
    for t in ticks:
        t["ready"] = t["at"] + 0.005
    return ticks


# ------------------------------------------------------------------------------------------------ the others
def results_reports(runs: dict) -> dict:
    """run.py write_results for a few sessions, with the clock, the GPU's name and the upscaler fixed."""
    import run as runpy
    saved = (runpy.gpu_name, runpy.time)
    runpy.gpu_name = lambda: "Apple M5 Max"
    runpy.time = SimpleNamespace(strftime=lambda fmt: "2026-11-22 12:00")
    settings = Settings({})
    out = {"header": {"stamp": "2026-11-22 12:00", "gpu": "Apple M5 Max",
                      "upscaler": runpy.BY_KEY["scaler"].name(settings.scaler), "size": "1920x1080"}, "reports": []}
    tmp = OUT / "results.tmp.txt"
    try:
        combos = [
            ("one", [("testmode_60", {"engine": "trt", "model": "4.22.lite", "motion": 1.0})]),
            ("half", [("testmode_120_minimal", {"engine": "torch", "model": "4.22.lite", "motion": 0.5})]),
            ("two", [("testmode_60", {"engine": "trt", "model": "4.22.lite", "motion": 0.25}),
                     ("testmode_90", {"engine": "torch", "model": "4.25", "motion": 0.75})]),
            ("dml", [("testmode_60", {"engine": "dml", "model": "4.22.lite", "motion": 0.5})]),
            ("dml_full", [("testmode_60", {"engine": "dml", "model": "4.22.lite", "motion": 1.0})]),
            ("vfg", [("testmode_120_minimal", {"engine": "trt", "model": "vfg", "motion": 1.0, "vfg_quality": "high"})]),
            ("untested", [("p30_steady", {"engine": "trt", "model": "4.22.lite", "motion": 1.0})]),
        ]
        for name, sessions in combos:
            ss = [(cfg, runs[r]["_stats"]) for r, cfg in sessions]
            ok = runpy.write_results(str(tmp), ss, "1920x1080", settings)
            vfgq = {cfg.get("vfg_quality"): runpy.BY_KEY["vfg_quality"].name(cfg["vfg_quality"])
                    for cfg, _ in ss if cfg.get("vfg_quality")}
            out["reports"].append({"name": name, "sessions": [[r, cfg] for r, cfg in sessions], "written": ok,
                                   "vfg_quality_names": vfgq,
                                   "lines": tmp.read_text(encoding="utf-8").split("\n")[:-1] if ok else None})
            tmp.unlink(missing_ok=True)
    finally:
        runpy.gpu_name, runpy.time = saved
    return out


def streamcheck_cases() -> dict:
    """tests/test_streamcheck.py's streams, through the real size check: the rows StreamCheck got and its report."""
    from framegen.chiaki_tap import SizeRepeatDetector
    from framegen.streamcheck import StreamCheck

    def play(pattern, seconds=300, fps_packets=60, lost_at=(), missing_at=(), md_new=5.0, md_rep=0.0, seed=0,
             size_says=None):
        rng = np.random.default_rng(seed)
        sc, det = StreamCheck(), SizeRepeatDetector()
        rows = {k: [] for k in ("t", "seq", "size", "ref", "dec", "key", "lost", "md", "mx")}
        seq = 0
        for i in range(int(seconds * fps_packets)):
            seq += 1 + (i in missing_at)
            new = pattern(i)
            size = int(rng.normal(50_000, 8_000)) if new else int(rng.normal(2_000, 300))
            dec = det.update(size, False)
            if size_says is not None:
                dec = size_says(i, new, dec)
            md = md_new * rng.uniform(0.5, 1.5) if new else md_rep
            mx = md * 20 if new else md_rep * 5
            row = (i / fps_packets + rng.normal(0, 0.0005), seq, size, det.ref, dec, det.big, i in lost_at, md, mx)
            sc.add(*row)
            for k, v in zip(rows, row):
                rows[k].append(None if v is None else (bool(v) if isinstance(v, (bool, np.bool_)) else
                                                       (int(v) if isinstance(v, (int, np.integer)) else float(v))))
        return {"rows": rows, "report": sc.report()}

    def hitch(i):
        k = i % 600
        return k % 2 == 0 and not 589 < k < 592

    def hold(i):
        k = i % 600
        return k % 2 == 0 if k < 590 else k == 594

    lost = set(range(1000, 18000, 900))
    pattern45 = np.random.default_rng(1).random(18000) < 0.75
    cases = {
        "locked_30": play(lambda i: i % 2 == 0),
        "game_hitches": play(hitch),
        "long_hold": play(hold),
        "misreads": play(lambda i: i % 2 == 0, size_says=lambda i, new, dec: True if (not new and i % 40 == 1) else dec),
        "few_misreads": play(lambda i: i % 2 == 0, size_says=lambda i, new, dec: False if (new and i % 120 == 0) else dec),
        "packet_loss": play(lambda i: i % 2 == 0, lost_at=lost,
                            size_says=lambda i, new, dec: (not new) if i in lost else dec),
        "missing": play(lambda i: i % 2 == 0, missing_at=set(range(500, 18000, 1000))),
        "sixty": play(lambda i: True),
        "varying": play(lambda i: bool(pattern45[i])),
        "stream_30": play(lambda i: True, fps_packets=30),
        "too_short": play(lambda i: i % 2 == 0, seconds=30),
        "still": play(lambda i: i % 2 == 0, md_new=0.1, md_rep=0.0),
    }
    # test_test_mode_frames_are_left_out: add_tick, with test mode's frames
    sc, det = StreamCheck(), SizeRepeatDetector()
    ticks = []
    for i in range(4000):
        size = 50_000 if i % 2 == 0 else 2_000
        dec = det.update(size, False)
        tick = SimpleNamespace(truth_new=(i % 2 == 0) if i >= 3600 else None, t_arrival=i * TICK,
                               extra={"seq": i, "size": size, "size_ref": det.ref, "size_new": dec,
                                      "keyframe": False, "big": det.big, "frames_lost": 0, "recovered": False})
        diff = SimpleNamespace(mean_diff=5.0 if i % 2 == 0 else 0.0, max_diff=90.0 if i % 2 == 0 else 0.0)
        sc.add_tick(tick, diff)
        ticks.append({"truth": tick.truth_new, "t": tick.t_arrival, "extra": tick.extra,
                      "md": diff.mean_diff, "mx": diff.max_diff})
    cases["test_mode_left_out"] = {"ticks": ticks, "report": sc.report(), "simulated": sc.simulated}
    return cases


def sanitize(x):
    if isinstance(x, float):
        return num(x)
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items() if k != "_stats"}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return num(float(x))
    return x


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    runs = {}
    t0 = _time.perf_counter()
    for name, (cfg, ticks) in scenarios().items():
        if only and name not in only and not only & {"results", "all"}:
            continue
        runs[name] = run(name, cfg, ticks)
        (OUT / f"trace_{name}.json").write_text(json.dumps(sanitize(runs[name])), encoding="utf-8")
        print(f"{name}: {len(runs[name]['events'])} events, {runs[name]['stats']['actions']}", file=sys.stderr)
    if not only or only & {"results", "all"}:
        (OUT / "results_report.json").write_text(json.dumps(sanitize(results_reports(runs))), encoding="utf-8")
    if not only or only & {"streamcheck", "all"}:
        (OUT / "streamcheck.json").write_text(json.dumps(sanitize(streamcheck_cases())), encoding="utf-8")
    print(f"done in {_time.perf_counter() - t0:.1f} s -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
