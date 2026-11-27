"""Instant replay (framegen/replay.py): the tracks it keeps, what it saves, the before & after video, the sound's
timing, and the pipeline's side. x264 on the CPU and small frames, so it runs anywhere; one check of the graphics
card's YUV against the CPU's when there's CUDA."""
import os
import re
import sys
import time

import av
import numpy as np
import pytest
import torch

from framegen import replay
from framegen.io import Sound, Tick
from framegen.settings import BY_KEY, Settings
from test_game_rate import _pipeline  # a pipeline with a stand-in interpolator

W, H = 160, 96
SR = 48000


def _img(k: int, bgr) -> np.ndarray:
    a = np.full((H, W, 3), bgr, np.uint8)
    a[:, (k * 3) % W:(k * 3) % W + 8] = 255  # a bar that moves
    return a


def _tone(k: int, n: int) -> bytes:
    s = np.arange(n) + k * n
    return np.repeat((np.sin(2 * np.pi * 440 * s / SR) * 8000).astype(np.int16), 2).tobytes()


def _feed(rec, seconds: float, hz: int = 90, t0: float = 1000.0, jitter: float = 0.02, seed: int = 0):
    """`seconds` of play: a frame shown at `hz`, a new game frame every third, 100 ms of sound every 0.1 s."""
    rng = np.random.default_rng(seed)
    rec.set_delay(40)
    for i in range(round(seconds * hz)):
        t = t0 + i / hz
        rec.frame(_img(i, (40, 160, 40)), t)
        if i % 3 == 0:
            rec.real_frame(_img(i, (160, 40, 40)), t - 0.04)
        if i % (hz // 10) == 0:
            k = i // (hz // 10)
            rec.add_sound([Sound(t0 + k * 0.1 - 0.04 + rng.uniform(0, jitter), _tone(k, SR // 10), SR, 2)])
    assert rec.flush(30)


def _probe(path):
    c = av.open(str(path))
    try:
        out = {}
        for s in c.streams:
            out[s.type] = (s.frames, float(s.duration * s.time_base) if s.duration else 0.0,
                           getattr(s.codec_context, "width", None))
        return out
    finally:
        c.close()


@pytest.fixture
def rec(tmp_path):
    toasts = []
    r = replay.Recorder(3, out_dir=tmp_path, encoder="libx264", device="cpu", drop_when_behind=False,
                        notify=lambda t, s: toasts.append(t))
    r.toasts = toasts
    yield r
    r.close()


def test_yuv_is_bt709_limited_range():
    for bgr, want in (((255, 255, 255), (235, 128, 128)), ((0, 0, 0), (16, 128, 128)),
                      ((0, 0, 255), (63, 102, 240)), ((255, 0, 0), (32, 240, 118))):
        y, u, v = replay._planes(replay.bgr_to_i420(np.full((4, 4, 3), bgr, np.uint8)), 4, 4)
        assert (int(y[0, 0]), int(u[0, 0]), int(v[0, 0])) == want, bgr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_graphics_cards_yuv_matches():
    r = replay.Recorder(0, device="cuda")
    try:
        img = np.random.default_rng(1).integers(0, 256, (H + 1, W + 1, 3), np.uint8)  # odd: cut to even
        r._prepare()
        r._make_pool("after", W, H)
        w, h, (arr, done, _), release = r._gpu_i420("after", torch.from_numpy(img).cuda())
        done.synchronize()
        got = arr.reshape(h * 3 // 2, w).astype(int)
        release()
        assert (w, h) == (W, H)
        assert np.abs(got - replay.bgr_to_i420(img[:H, :W]).astype(int)).max() <= 1
    finally:
        r.close()


def test_saves_what_was_shown_with_the_sound(rec, tmp_path):
    _feed(rec, 6)
    paths = rec.save(wait=True)
    assert len(paths) == 1 and paths[0].parent == tmp_path and paths[0].suffix == ".mp4"
    p = _probe(paths[0])
    frames, secs, w = p["video"]
    # the last 3 s, from the keyframe before: 3 to 4 s, every frame shown
    assert 3.0 <= secs <= 4.1 and w == W and abs(frames / secs - 90) < 2
    assert abs(p["audio"][1] - secs) < 0.15  # the sound covers the clip
    assert any(t.startswith("Saved the last") for t in rec.toasts)


def test_keeps_only_the_length_asked_for(rec):
    _feed(rec, 9)
    held = rec.held()
    limit = rec.length + replay.KEEP_EXTRA_S + replay.KEY_EVERY_S + 0.1
    assert all(secs <= limit for secs, _ in held.values()), held
    assert held["after"][0] >= rec.length
    rec.set_length(0)  # off: everything goes
    assert rec.flush(5) and all(mb == 0 for _, mb in rec.held().values())


def test_before_and_after_split_and_side_by_side(rec, tmp_path):
    _feed(rec, 4)
    paths = rec.save("both", wait=True)
    assert len(paths) == 2 and "before and after" in paths[1].name
    p = _probe(paths[1])
    assert p["video"][2] == W and abs(p["video"][1] - _probe(paths[0])["video"][1]) < 0.05 and "audio" in p
    c = av.open(str(paths[1]))
    img = next(f for i, f in enumerate(c.decode(video=0)) if i == 30).to_ndarray(format="bgr24")
    c.close()
    left, right = img[60:90, 4:30].reshape(-1, 3).mean(0), img[60:90, -30:-4].reshape(-1, 3).mean(0)
    assert left[0] > 120 and left[1] < 80  # the game's own frames (blue) on the left
    assert right[1] > 120 and right[0] < 80  # SmoovPlay's (green) on the right
    side = rec.save("compare", layout="side", wait=True)
    assert len(side) == 1 and _probe(side[0])["video"][2] == 2 * W
    assert not list(tmp_path.glob("*.part")) and not list(tmp_path.glob("*.ts"))


def test_labels_name_the_frame_rates(rec, monkeypatch):
    seen = []
    real_popen = replay.subprocess.Popen

    def popen(cmd, **kw):
        seen.append(cmd)
        return real_popen(cmd, **kw)
    monkeypatch.setattr(replay.subprocess, "Popen", popen)
    _feed(rec, 3)
    rec.save("compare", wait=True)
    labels = seen[0][seen[0].index("--labels") + 1:][:2]
    assert labels == ["Original 30 fps", "SmoovPlay 90 fps"]


def test_stopping_waits_for_the_clip_but_not_the_before_and_after(rec, tmp_path, monkeypatch, capsys):
    """Esc: the clip is finished first, but a before & after video carries on in its own process, so the video
    window never waits on it (it once looked frozen for half a minute, and Windows offered to end it)."""
    real_popen, procs = replay.subprocess.Popen, []

    def slow(cmd, **kw):  # stands in for making the before & after video
        procs.append(real_popen([sys.executable, "-c", "import time; time.sleep(4)"], **kw))
        return procs[-1]
    monkeypatch.setattr(replay.subprocess, "Popen", slow)
    _feed(rec, 3)
    rec.save("both")
    t0 = time.perf_counter()
    while not procs and time.perf_counter() - t0 < 10:
        time.sleep(0.02)
    t1 = time.perf_counter()
    rec.close(wait_saves=30)
    took = time.perf_counter() - t1
    assert procs and procs[0].poll() is None and took < 2.5
    assert len(list(tmp_path.glob("SmoovPlay *.mp4"))) == 1  # the clip itself is there
    assert "still being made: they carry on in the background" in capsys.readouterr().out
    procs[0].wait(10)


def test_the_before_and_after_process_leaves_word_for_the_launcher(tmp_path):
    mark = tmp_path / "saving"
    out = tmp_path / "x before and after.mp4"
    with pytest.raises(Exception):
        replay.main(["compose", "--after", str(tmp_path / "none.ts"), "--before", str(tmp_path / "none.ts"),
                     "--out", str(out), "--mark", str(mark)])
    files = list(mark.glob("*.txt"))
    assert len(files) == 1 and files[0].stem == str(os.getpid())
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert lines[0] == str(out) and lines[1].startswith("failed: ")
    assert not out.exists() and not (tmp_path / (out.name + ".part")).exists()


def test_a_finished_before_and_after_tidies_its_word_away(rec, tmp_path):
    rec.mark_dir = tmp_path / "saving"
    _feed(rec, 3)
    paths = rec.save("compare", wait=True)
    assert len(paths) == 1 and paths[0].exists()
    assert not list(rec.mark_dir.glob("*"))


def test_off_says_so_and_takes_nothing(tmp_path):
    toasts = []
    r = replay.Recorder(0, out_dir=tmp_path, encoder="libx264", device="cpu", notify=lambda t, s: toasts.append(t))
    try:
        r.frame(_img(0, (0, 0, 0)), 1.0)
        assert r.save(wait=True) == [] and "off" in toasts[-1]
        assert not r._q.unfinished_tasks and not list(tmp_path.iterdir())
    finally:
        r.close()


def test_nothing_yet(rec):
    assert rec.save(wait=True) == [] and "nothing" in rec.toasts[-1]


# ---- the sound's timing: arrival times jitter, the sound plays on; gaps and running ahead are put right
def _sound_rec():
    r = replay.Recorder(10, encoder="libx264", device="cpu", drop_when_behind=False)
    r.set_delay(0)
    return r


def _chunk(t, n=480):
    return Sound(t, bytes(4 * n), SR, 2)


def test_jitter_doesnt_move_the_sound():
    r = _sound_rec()
    try:
        rng = np.random.default_rng(2)
        for k in range(300):  # 3 s of 10 ms chunks, each up to 40 ms late
            r._encode_sound(_chunk(100 + k * 0.01 + rng.uniform(0, 0.04)))
        written = (r._a_pts - replay.OFFSET * SR // replay.TB) / SR  # from the clock's zero, 5 s before the first
        assert written == pytest.approx(5.0 + 3.0, abs=0.03)  # exactly the sound given: nothing put in or skipped
    finally:
        r.close()


def test_a_gap_becomes_silence_and_running_ahead_is_skipped():
    r = _sound_rec()
    try:
        r.set_delay(200)
        t = 100.0
        for k in range(100):
            r._encode_sound(_chunk(t + k * 0.01))
        t_gap = t + 1.5  # half a second with no sound
        for k in range(50):
            r._encode_sound(_chunk(t_gap + k * 0.01))
        assert r._a_next == pytest.approx(t_gap + 0.5 + 0.2, abs=0.02)  # silence put in, then the sound on time
        # SmoovPlay's delay drops by 100 ms and stays there: after a second the sound skips 100 ms, back in step
        r.set_delay(100)
        for k in range(150):
            r._encode_sound(_chunk(t_gap + 0.5 + k * 0.01))
        assert r._a_next == pytest.approx(t_gap + 0.5 + 1.5 + 0.1, abs=0.02)
    finally:
        r.close()


# ---- the pipeline's side
class _Rec:
    def __init__(self):
        self.shown, self.real, self.delays, self.saves = 0, 0, [], []

    def frame(self, frame, t):
        self.shown += 1

    def real_frame(self, frame, t):
        self.real += 1

    def set_delay(self, ms):
        self.delays.append(ms)

    def save(self, kind=None):
        self.saves.append(kind)


class _Sink:
    gpu = False

    def __init__(self, cmds):
        self.cmds, self.toasts = list(cmds), []

    def write(self, frame, text=None):
        return self.cmds.pop(0) if self.cmds else None

    def toast(self, text, seconds=2.0):
        self.toasts.append(text)


def _stream(seconds):
    i, t_end = 0, time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        while time.perf_counter() - t < 0.004:
            pass
        f = torch.full((8, 8, 3), 60 + (i // 2) % 100, dtype=torch.uint8)
        yield Tick(f, time.perf_counter(), i, new_hint=i % 2 == 0)
        i += 1


def test_the_pipeline_feeds_it_and_r_and_b_save():
    r = _Rec()
    sink = _Sink([None, "save_replay", None, "save_compare"])
    st = _pipeline().run(_stream(2.5), sink, label="RIFE", settings=Settings({"hz": 60}), replay=r)
    n = st.actions and sum(st.actions.values())
    assert r.shown >= n  # every frame shown
    assert abs(r.real - n / 2) <= 2  # each new game frame once
    assert r.delays and all(d > 0 for d in r.delays)
    assert r.saves == [None, "compare"]
    sink = _Sink(["save_replay"])
    _pipeline().run(_stream(0.2), sink, label="RIFE", settings=Settings({"hz": 60}))
    assert sink.toasts and "isn't available" in sink.toasts[0]


# ---- settings
def test_settings_and_their_memory_hints():
    s = BY_KEY["replay"]
    assert s.default == 0 and s.choices == replay.LENGTHS and s.group == "Instant replay"
    for secs in s.choices[1:]:
        m = re.search(r"about ([\d.]+) (MB|GB)", s.hints[secs])
        mb = float(m.group(1)) * (1000 if m.group(2) == "GB" else 1)
        assert abs(mb - replay.memory_mb(secs)) / replay.memory_mb(secs) < 0.05, secs
    assert BY_KEY["replay_save"].default == "smoovplay"  # what was shown, unless asked for more
    for k in ("replay_save", "replay_layout"):
        assert BY_KEY[k].applies[0] == "replay" and 0 not in BY_KEY[k].applies[1]


def test_durations_and_names():
    assert replay._dur(30) == "30 s" and replay._dur(60) == "60 s" and replay._dur(120) == "2 min"
    assert replay._dur(300) == "5 min" and replay._dur(150) == "2.5 min"
    assert replay._snap_fps(29.6) == 30 and replay._snap_fps(88.9) == 90 and replay._snap_fps(37) == 37


def test_two_saves_in_one_second_get_different_names(tmp_path):
    """Both saves checked only that the clip's name was free, while the writing goes to name.part: the second one
    now finds the first's .part and takes "(2)"."""
    from framegen.replay import _unique
    p = tmp_path / "SmoovPlay 2026-11-23 01-02-03.mp4"
    first, second = _unique(p), _unique(p)
    assert first == p
    assert second == tmp_path / "SmoovPlay 2026-11-23 01-02-03 (2).mp4"
    assert (tmp_path / (first.name + ".part")).exists() and (tmp_path / (second.name + ".part")).exists()
    (tmp_path / (first.name + ".part")).rename(first)  # the first save finished
    assert _unique(p) == tmp_path / "SmoovPlay 2026-11-23 01-02-03 (3).mp4"


def test_a_save_that_fails_leaves_no_part_file(tmp_path):
    """_unique makes the clip's .part before anything is written: when the remux fails (here, on video that isn't
    MPEG-TS), that empty .part goes too, rather than staying beside the clips."""
    from framegen.replay import _unique, to_mp4
    p = _unique(tmp_path / "SmoovPlay 2026-11-27 01-02-03.mp4")
    assert (tmp_path / (p.name + ".part")).exists()
    with pytest.raises(Exception):
        to_mp4(p, (b"not a video", [(0, 0, b"nor this")]), None, 0, 1)
    assert list(tmp_path.iterdir()) == []
