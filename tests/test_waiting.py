"""A live source that stalls must not stall the video loop: it yields Waiting, and the loop keeps the window
answering (its keys, and Windows' "not responding" check, which used to close SmoovPlay mid-session)."""
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from framegen.io import WAIT_POLL_S, MultiSink, Tick, Waiting, queue_ticks


def test_queue_ticks_says_waiting_while_nothing_comes():
    q = queue.Queue()
    got = []

    def feed():
        q.put(Tick(np.zeros((2, 2, 3), np.uint8), time.perf_counter(), 99))
        time.sleep(WAIT_POLL_S * 8)  # a stall, long enough for two waits on a slow machine (one at 3.5 polls, 28 Nov)
        q.put(Tick(np.zeros((2, 2, 3), np.uint8), time.perf_counter(), 99))
        q.put(None)

    threading.Thread(target=feed, daemon=True).start()
    t0 = time.perf_counter()
    for x in queue_ticks(q, "the PS5"):
        got.append(x)
        assert time.perf_counter() - t0 < 5
    ticks = [x for x in got if isinstance(x, Tick)]
    waits = [x for x in got if isinstance(x, Waiting)]
    assert [t.index for t in ticks] == [0, 1]  # numbered, Waiting not counted
    assert 2 <= len(waits) <= 10
    assert all(w.what == "the PS5" for w in waits)
    assert [round(w.since, 1) for w in waits] == sorted(round(w.since, 1) for w in waits)  # time since the last frame
    kinds = ["T" if isinstance(x, Tick) else "W" for x in got]
    assert kinds[0] == "T" and kinds[-1] == "T" and "W" in kinds[1:-1]  # the waits come in the stall


def test_multisink_wait_asks_the_sinks_that_can():
    class Win:
        def __init__(self):
            self.seen = []

        def write(self, frame, text=None):
            return None

        def wait(self, w):
            self.seen.append(w)
            return "quit"

    class File:
        def write(self, frame, text=None):
            return None

    win = Win()
    assert MultiSink(win, File()).wait(Waiting(3.0, "the PS5")) == "quit"
    assert len(win.seen) == 1


def test_pipeline_quits_during_a_stall():
    """Q (or closing the window) while no frame comes: the session ends instead of hanging."""
    import torch

    from framegen.pipeline import Pipeline
    interp = SimpleNamespace(h=8, w=8, device=torch.device("cpu"), model="rife")

    class Sink:
        sinks = [1]
        calls = 0

        def write(self, frame, text=None):
            return None

        def wait(self, w):
            Sink.calls += 1
            return "quit" if Sink.calls == 3 else None

    def stalled():
        while True:
            yield Waiting(1.0, "the PS5")

    pipe = Pipeline(interp)
    before = pipe.beat
    try:
        st = pipe.run(stalled(), Sink())
    except AttributeError as e:  # a fake interpolator that's too thin for run()'s set-up: say what it needs
        raise AssertionError(f"run() needs more of the interpolator before its loop: {e}")
    assert st.exit == "quit" and Sink.calls == 3
    assert pipe.beat > before
    assert pipe.waiting is not None and pipe.waiting.what == "the PS5"  # (run.py's watchdog reads it)


def test_the_note_can_take_a_second_line():
    from framegen.io import _pill
    one, two = _pill("No picture from the PS5 for 3 s. Q quits", 1.0), _pill("No picture\nconnect again", 1.0)
    assert two.shape[0] > one.shape[0] * 1.4 and two.shape[1] < one.shape[1]
