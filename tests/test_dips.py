"""30 fps games that dip: a frame the game misses stays on the 60 Hz stream for a third tick (50 ms, not 33), or a
fourth (67 ms, waiting for the next 30 fps slot). CPU only: a stand-in interpolator blends its two frames' numbers,
and each game frame's number is its time in ticks, so on screen the numbers should only ever go up."""
import random
import statistics
import time

import pytest
import torch

from framegen.cadence import DipWatch, FrameTimeline
from framegen.io import Tick
from framegen.pipeline import Pipeline
from framegen.settings import Settings


class _Interp:
    h = w = 8
    device = "cpu"
    private = False

    def __init__(self):
        self.in_u8 = torch.zeros(8, 8, 3)
        self.slots = [0.0, 0.0]

    def load(self, frame):
        self.in_u8 = frame

    def prepare(self, slot):
        self.slots[slot] = float(self.in_u8[0, 0, 0])

    def interpolate(self, slot, t=0.5):
        a, b = self.slots[1 - slot], self.slots[slot]
        return torch.full((8, 8, 3), a + t * (b - a))


class _Sink:
    gpu = False

    def __init__(self):
        self.shown = []

    def write(self, frame, text=None):
        f = frame if torch.is_tensor(frame) else torch.as_tensor(frame)
        self.shown.append(float(f[0, 0, 0]))


def _pipeline(interp):
    from framegen import pipeline as pl
    p = Pipeline.__new__(Pipeline)  # no pinned memory: that would start CUDA
    p.interp, p.detector, p.cut_threshold, p.stats = interp, pl.DuplicateDetector(), 25.0, None
    p._out_pinned = torch.empty((8, 8, 3))  # float, so the blends aren't rounded
    p._out_np = p._out_pinned
    return p


def _dipping_game(frames=150, held3=0.3, seed=1, hold=3, steady=0):
    """A 30 fps game on a 60 Hz stream that misses about held3 of its frames (each then stays up `hold` ticks), after
    `steady` frames that it doesn't, as Remote Play sends it: compressed sizes (a new frame big, a repeat small) and
    the stream's frame numbers."""
    rnd = random.Random(seed)
    i = t = 0
    for f in range(frames):
        ticks = hold if f >= steady and rnd.random() < held3 else 2
        for k in range(ticks):
            yield Tick(torch.full((8, 8, 3), float(t)), time.perf_counter(), i, new_hint=(k == 0),
                       extra={"size": 9000 if k == 0 else 300, "seq": i})
            i += 1
        t += ticks


def _backward_steps(shown):
    return sum(1 for a, b in zip(shown, shown[1:]) if b < a - 1e-6)


def test_the_30_fps_setting_never_steps_back_when_the_game_dips():
    for hz in (60, 90, 120):
        it, sink = _Interp(), _Sink()
        _pipeline(it).run(_dipping_game(), sink, label="RIFE",
                          settings=Settings({"hz": hz, "even_out": False, "overlay": False, "game_fps": "30"}))
        assert len(sink.shown) > 300
        assert _backward_steps(sink.shown) == 0, hz


def test_auto_never_steps_back_when_the_game_dips():
    it, sink = _Interp(), _Sink()
    _pipeline(it).run(_dipping_game(), sink, label="RIFE",
                      settings=Settings({"hz": 90, "even_out": False, "overlay": False}))
    assert _backward_steps(sink.shown) == 0


def test_a_steady_30_fps_game_is_unchanged_by_the_30_fps_setting():
    shown = {}
    for fps in ("auto", "30"):
        it, sink = _Interp(), _Sink()
        _pipeline(it).run(_dipping_game(held3=0.0), sink, label="RIFE",
                          settings=Settings({"hz": 90, "even_out": False, "overlay": False, "game_fps": fps}))
        shown[fps] = sink.shown
    assert shown["30"][40:] == shown["auto"][40:]  # (after the lock has found the rhythm)


# ------------------------------------------------------------------------------------------ Dip smoothing
def _unevenness(shown):
    """How unevenly the game's time moves from one output frame to the next: 0 when every step is the same."""
    steps = [b - a for a, b in zip(shown, shown[1:])]
    return statistics.pstdev(steps) / statistics.mean(steps)


def _play(game, hz, **settings):
    it, sink = _Interp(), _Sink()
    st = _pipeline(it).run(game, sink, label="RIFE", settings=Settings({"hz": hz, "overlay": False, **settings}))
    return sink.shown, st


def test_dip_watch_switches_on_while_the_game_dips_and_off_after_it_stops():
    w = DipWatch()
    calls = [k == 0 for _ in range(60) for k in range(2)]  # 2 s at a steady 30 fps
    assert not any(w.update(c) for c in calls)
    dip = [True, False, False, False] + [True, False] * 4  # a frame held 4 ticks, then 4 steady ones
    for c in dip:
        w.update(c)
    assert not w.dipping  # one dip isn't enough
    for c in dip:
        w.update(c)
    assert w.dipping  # two within a second
    for c in [True, False] * 85:  # 2.9 s steady again: still on (the dips may come back)
        w.update(c)
    assert w.dipping
    for c in [True, False] * 3:  # 3 s since the last dip
        w.update(c)
    assert not w.dipping


def test_dip_watch_leaves_pauses_and_fast_games_alone():
    w = DipWatch()
    pauses = [k == 0 for _ in range(20) for k in range(12)]  # a frame every 200 ms: a load or a still scene
    sixty = [True] * 50 + [True, False, False] + [True] * 50  # a 60 fps game's hitches
    assert not any(w.update(c) for c in pauses + sixty * 3)


@pytest.mark.parametrize("hz", [60, 90, 120])
@pytest.mark.parametrize("hold", [3, 4])
def test_even_out_dips_spreads_a_dip_instead_of_freezing(hz, hold):
    off, _ = _play(_dipping_game(300, 0.15, hold=hold, steady=40), hz)
    on, st = _play(_dipping_game(300, 0.15, hold=hold, steady=40), hz, even_dips=True)
    skip = int(200 * hz / 60)  # (after the dips have switched it on)
    assert st.dip_ticks > 300
    assert _backward_steps(on) == 0
    assert _unevenness(on[skip:]) < 0.5 * _unevenness(off[skip:]), (_unevenness(on[skip:]), _unevenness(off[skip:]))


def test_even_out_dips_works_with_the_30_fps_setting():
    off, _ = _play(_dipping_game(300, 0.15, hold=4, steady=40), 90, game_fps="30")
    on, st = _play(_dipping_game(300, 0.15, hold=4, steady=40), 90, game_fps="30", even_dips=True)
    assert st.dip_ticks > 300 and _backward_steps(on) == 0
    assert _unevenness(on[300:]) < 0.5 * _unevenness(off[300:])


def test_even_out_dips_changes_nothing_while_a_game_holds_30_fps():
    for fps in ("auto", "30"):
        off, _ = _play(_dipping_game(200, 0.0), 90, game_fps=fps)
        on, st = _play(_dipping_game(200, 0.0), 90, game_fps=fps, even_dips=True)
        assert on == off and st.dip_ticks == 0


def test_even_out_dips_is_off_by_default():
    from framegen.settings import SPECS
    spec = next(s for s in SPECS if s.key == "even_dips")
    assert spec.default is False and spec.experimental
    _, st = _play(_dipping_game(300, 0.15, hold=4, steady=40), 90)
    assert st.dip_ticks == 0


@pytest.mark.parametrize("hz", [60, 90, 120])
def test_a_frame_counts_as_shown_when_it_is_drawn_as_itself(hz):
    """The delay figures, and the sound sync that follows them, count a frame from when it's first on screen as
    itself. Timed as it arrived, that's a display time before its own time passes (which is where it used to be
    counted: a tick late, so the sound was held back 17 ms too long)."""
    tl = FrameTimeline()
    tl.set_at_arrival(True)
    rnd = random.Random(3)
    calls = []
    for _ in range(300):
        calls += [True] + [False] * (2 if rnd.random() < 0.3 else 1)
    drawn, counted = {}, {}
    for n, new in enumerate(calls):
        tl.tick(new, frame=n)
        for phi in tl.slots(hz):
            p = tl.place(phi)
            if p.b is None:
                drawn.setdefault(p.a.frame, n)
            for f in p.reached:
                assert f.frame not in counted  # each frame once
                counted[f.frame] = n
    assert len(counted) > 280
    assert [a for a, n in drawn.items() if a > 60 and counted[a] != n] == []


def _toggling(game, settings, at):
    """The game's ticks, with interpolation switched (off, on, off...) at the tick numbers in `at`."""
    for n, tick in enumerate(game):
        if n in at:
            settings.override("interp", not settings.interp)
        yield tick


@pytest.mark.parametrize("hz", [60, 90])
@pytest.mark.parametrize("even_dips", [False, True])
def test_interpolation_switched_off_and_on_while_the_game_dips(hz, even_dips):
    """The timeline (and which of its frames are in the interpolator's slots) starts afresh when interpolation is
    toggled. It used to keep the frames from before it went off: switched on again at a repeat tick while the game
    dipped, the pipeline asked it for frames it didn't have and raised, or showed a seconds-old picture."""
    s = Settings({"hz": hz, "overlay": False, "even_dips": even_dips})
    at = {300, 333, 410, 425, 500, 561, 640, 655}  # off, then on again 15-61 ticks later (most of them repeats)
    sink = _Sink()
    st = _pipeline(_Interp()).run(_toggling(_dipping_game(400, 0.15, hold=3, steady=40), s, at), sink,
                                  label="RIFE", settings=s)
    assert s.interp  # (an even number of switches)
    assert not even_dips or st.dip_ticks > 100
    assert _backward_steps(sink.shown) == 0
