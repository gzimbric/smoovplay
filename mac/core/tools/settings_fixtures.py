"""Writes the parity fixtures for crates/sp-settings (mac/core/fixtures/settings/*.json, git-ignored) from the Python
reference. They hold what framegen/settings.py, remote.py, looks.py, fill.py and autodetect.py produce for a set of
inputs, so the Rust tests (crates/sp-settings/tests/parity.rs) can check the port gives exactly the same.

    .venv/bin/python mac/core/tools/settings_fixtures.py

Everything runs in a temporary folder (APPDATA and SMOOVPLAY_LOOKS_DIR point there before framegen is imported), so
the real settings file and looks are never read or written. Paths that depend on that folder are written as
placeholders ({dir}, {src}, {samples}) that the Rust tests fill in with their own.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "settings"
TMP = Path(tempfile.mkdtemp(prefix="sp-settings-fixtures-"))
os.environ["APPDATA"] = str(TMP / "appdata")
os.environ["SMOOVPLAY_LOOKS_DIR"] = str(TMP / "looks-at-import")
sys.path.insert(0, str(ROOT))

from framegen import autodetect, fill, looks, remote, settings  # noqa: E402
from framegen.looks import BUILTIN, TYPES_LIST, Library, decode, encode, node, normalize  # noqa: E402
from framegen.settings import BY_KEY, SPECS, Settings  # noqa: E402

DEFAULTS = {s.key: s.default for s in SPECS}
SOURCES = ("capture", "chiaki", "window", "tap", "file", "")
EMPTY_LIB = TMP / "lib-empty"


def fresh_dir(name: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=name + "-", dir=TMP))
    return d


def reset_looks() -> None:
    Library(EMPTY_LIB)  # the setting's choices back to the built-in looks only


def write(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  {name}.json")


def err(e: BaseException) -> dict:
    return {"error": type(e).__name__, "message": str(e)}


USER_LOOKS = {  # each file's text by name, for cases that need looks of the user's
    "moody-blues.json": json.dumps({"version": 1, "name": "Moody blues", "nodes": [node("color", temperature=-0.4)]}),
    "alpha.json": json.dumps({"name": "alpha", "nodes": [node("grain", amount=0.5)]}),
    "zeta.json": json.dumps({"name": "Alpha", "nodes": []}),
    "caf\u00e9.json": json.dumps({"name": "Caf\u00e9 cr\u00e8me \u2615", "nodes": [node("vignette")]}),
    "broken.json": "{nope",
}


def user_library(files: dict) -> Path:
    d = fresh_dir("lib")
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------------------ settings
def gen_specs() -> None:
    saved = settings.DEFAULT_PATH
    settings.DEFAULT_PATH = Path("/fixture/framegen/settings.json")
    try:
        reset_looks()
        builtin = json.dumps(settings.specs_json())
        with contextlib.redirect_stdout(io.StringIO()):
            Library(user_library(USER_LOOKS))
        with_user = json.dumps(settings.specs_json())
    finally:
        settings.DEFAULT_PATH = saved
        reset_looks()
    probes = [True, False, 0, 1, 2, 60, 75, 90.0, 0.0, 0.25, 1.0, "fsr", "nope", "auto", None, "1080p", "cinematic"]
    names = {s.key: [[json.dumps(v), s.name(v), s.hints.get(v, "") if _hashable(v) else ""] for v in probes]
             for s in SPECS if s.key != "look"}
    applies = []
    for vals in ({}, {"model": "vfg"}, {"model": 1}, {"scaler": "vsr", "replay": 60, "overlay": True},
                 {"interp": 1, "fullscreen": 0, "look_on": 1.0}, DEFAULTS):
        applies.append({"values": json.dumps(vals), "applies": [s.applies_to(vals) for s in SPECS]})
    write("specs", {
        "path": "/fixture/framegen/settings.json",
        "builtin": builtin,
        "user_looks": USER_LOOKS,
        "with_user": with_user,
        "unused_with": {s.key: {src: s.unused_with(src) for src in SOURCES} for s in SPECS},
        "names": names,
        "applies": applies,
        "capture_sizes": settings.CAPTURE_SIZES,
    })


def _hashable(v) -> bool:
    try:
        hash(v)
        return True
    except TypeError:
        return False


SETTINGS_FILES = [
    ("missing", None, [["set", "scaler", "bilinear"]]),
    ("empty", "", [["set", "hz", 90]]),
    ("not json", "{not json", [["set", "overlay", False]]),
    ("a list", "[1, 2]", [["set", "overlay", False]]),
    ("a number", "90", []),
    ("null", "null", []),
    ("byte-order mark", "\ufeff{\"hz\": 90}", []),
    ("test from the Python", json.dumps({"hz": 90, "scaler": "nope", "unknown": 1, "fsr_sharpness": 0.25}),
     [["set", "scaler", "bilinear"]]),
    ("old keys", json.dumps({"upscaler": "fsr", "frame_rate": 90, "rife_model": "4.6", "hz": 120,
                             "window_scale": 1.5}), [["set", "vsync", True]]),
    ("wrong types", json.dumps({"hz": "90", "interp": 1, "overlay": 0, "fsr_sharpness": 1, "vsr_quality": 2.0,
                                "replay": False, "vsync": "true", "fill": ["fit"], "look_on": 1.0,
                                "even_out": {"on": True}, "steady_hud": None}),
     [["set", "fullscreen", True], ["set", "interp", True], ["set", "hz", 120.0]]),
    ("floats kept as floats", json.dumps({"hz": 90.0, "replay": 30.0, "vsr_quality": 4.0, "fsr_sharpness": 2}),
     [["set", "overlay", False]]),
    ("bools as numbers", json.dumps({"fsr_sharpness": True, "vsr_quality": True, "replay": False, "hz": True}),
     [["set", "overlay", False]]),
    ("NaN and Infinity", '{"fsr_sharpness": NaN, "hz": Infinity, "replay": -Infinity, "vsync": true}',
     [["set", "overlay", False]]),
    ("huge numbers", '{"hz": 90000000000000000000000000, "replay": 1e400, "vsr_quality": -0, '
                     '"fsr_sharpness": -0.0}', [["set", "overlay", False]]),
    ("duplicate keys", '{"hz": 60, "scaler": "vsr", "hz": 120}', [["set", "overlay", False]]),
    ("unicode keys and values", json.dumps({"h\u00fcz": 90, "scaler": "fsr\u00e9", "look": "caf\u00e9"}),
     [["set", "overlay", False]]),
    ("test mode in the file", json.dumps({"simulate_30": True, "hz": 90}),
     [["set", "simulate_30", True], ["set", "overlay", False]]),
    ("every setting", json.dumps({s.key: s.choices[-1] for s in SPECS if s.key != "look"}),
     [["set", "hz", 60]]),
    ("pretty and odd whitespace", '\n\t{ "hz" :\r\n 90 ,"scaler":"vsr"}\n\n', [["set", "overlay", False]]),
    ("override is not saved", None, [["override", "hz", 120], ["set", "overlay", False]]),
    ("override then set", None, [["override", "hz", 120], ["set", "hz", 120], ["override", "hz", 90],
                                 ["set", "overlay", False]]),
    ("bad values", None, [["set", "hz", 75], ["set", "nope", 1], ["set", "scaler", None],
                          ["set", "interp", "yes"], ["set", "fill", "wide"], ["set", "look", "my:none"],
                          ["override", "fsr_sharpness", 0.3]]),
    ("set to the same value", None, [["set", "hz", 60], ["set", "interp", 1], ["set", "fsr_sharpness", 0]]),
    ("step", None, [["step", "hz", 1], ["step", "hz", 1], ["step", "hz", 1], ["step", "hz", -1],
                    ["step", "scaler", 1, [["scaler", "vsr"]]], ["step", "scaler", 1, [["scaler", "vsr"]]],
                    ["step", "fsr_sharpness", -3], ["step", "look", 1], ["step", "look", -2],
                    ["step", "interp", 5], ["step", "overlay_corner", 2, [["overlay_corner", "bottom_left"]]],
                    ["step", "nope", 1]]),
    ("step where nothing is allowed", None, [["step", "hz", 1, [["hz", 60], ["hz", 90], ["hz", 120]]]]),
    ("step from a value of another type", json.dumps({"interp": 0, "hz": 90.0}),
     [["step", "interp", 1], ["step", "hz", 1]]),
    ("user looks", json.dumps({"look": "my:moody-blues", "look_on": True}), [["set", "look", "my:alpha"]]),
]


def gen_settings_files() -> None:
    cases = []
    for name, text, ops in SETTINGS_FILES:
        reset_looks()
        lib = None
        if name == "user looks" or any(op[1] == "look" for op in ops if len(op) > 1):
            with contextlib.redirect_stdout(io.StringIO()):
                lib = Library(user_library(USER_LOOKS))
        d = fresh_dir("settings")
        path = d / "sub" / "settings.json"
        if text is not None:
            path.parent.mkdir(parents=True)
            path.write_bytes(text.encode("utf-8"))
        st = Settings.load(path)
        seen = []
        st.watch(seen.append)
        values = json.dumps(st.values())
        results = []
        log = io.StringIO()
        for op in ops:
            try:
                with contextlib.redirect_stdout(log):
                    if op[0] == "set":
                        st.set(op[1], op[2])
                        r = None
                    elif op[0] == "override":
                        st.override(op[1], op[2])
                        r = None
                    else:
                        deny = op[3] if len(op) > 3 else None
                        r = st.step(op[1], op[2], (lambda k, v, deny=deny: [k, v] not in deny) if deny else None)
                results.append({"ok": json.dumps(r)})
            except (ValueError, KeyError) as e:
                results.append(err(e))
        cases.append({
            "name": name, "file": text, "ops": ops, "user_looks": USER_LOOKS if lib else None,
            "values": values, "results": results, "seen": seen,
            "values_after": json.dumps(st.values()),
            "file_after": path.read_text(encoding="utf-8") if path.exists() else None,
            "log": log.getvalue(),
        })
    reset_looks()
    write("settings_files", cases)


STATE_CASES = [
    {"name": "defaults", "commands": []},
    {"name": "test: set applies, saves and reports", "commands": ["set hz 90", "set fsr_sharpness 1",
                                                                  "set overlay false"]},
    {"name": "test: bad and unavailable values",
     "deny": [["scaler", "vsr"]], "disabled": {"vsr_quality": "no RTX VSR here"},
     "commands": ["set hz 75", "set nope 1", "set scaler \"vsr\"", "set vsr_quality 4", "set hz {broken",
                  "bogus"]},
    {"name": "test: restart and state", "restart": ["AI engine"],
     "commands": ["set engine \"torch\"", "restart", "state"]},
    {"name": "quit wins over restart", "commands": ["restart", "quit", "restart"]},
    {"name": "unavailable rows", "deny": [["hz", 120], ["hz", 90], ["model", "vfg"], ["fsr_sharpness", 0]],
     "disabled": {"capture_size": "Only used with a capture card.", "minimize_chiaki": "",
                  "vfg_quality": "Only used with NVIDIA VFG."},
     "commands": ["set hz 120", "set fsr_sharpness 0.0", "set fsr_sharpness 0.5", "set capture_size \"1440p\"",
                  "set minimize_chiaki false"]},
    {"name": "restart pending", "restart": ["AI engine", "Motion detail", "Window presentation"],
     "commands": ["set motion \"half\"", "set present \"dxgi\"", "state", "restart"]},
    {"name": "overrides", "overrides": [["hz", 120], ["interp", False], ["simulate_30", True]],
     "commands": ["set interp true", "set hz 120", "state"]},
    {"name": "odd file", "file": json.dumps({"hz": 90.0, "interp": 1, "fsr_sharpness": 2, "replay": 30.0}),
     "commands": ["set interp true", "set hz 90", "set replay 60", "set fsr_sharpness 2.0", "set interp 0"]},
    {"name": "odd lines", "commands": ["\ufeffset hz 90", "  set hz 120  ", "set hz 60\r", "set", "set hz",
                                       "set  hz 90", "set hz  90", "set hz 9e1", "set hz 90.0", "set hz true",
                                       "set overlay 1", "set overlay 0.0", "set overlay \"false\"",
                                       "SET hz 90", "set\thz 90", "", "   ", "state extra words",
                                       "quit now"]},
    {"name": "long and unicode values", "commands": [
        "set scaler \"" + "x" * 60 + "\"", "set scaler \"fsr\u00e9\u2603\"", "set look \"my:caf\u00e9\"",
        "set look \"cinematic\"", "set look_on true", "set fill \"panorama\"", "set nope " + "y" * 100,
        "unknown-command " + "z" * 100]},
    {"name": "user looks", "user_looks": True,
     "commands": ["set look \"my:moody-blues\"", "set look \"my:broken\"", "set look \"my:caf\u00e9\"",
                  "set look_on true"]},
    {"name": "handlers", "commands": ['look-show {"name": "x", "nodes": []}', "look-hide", "look-show oops",
                                      'look-show {"name": "y", "nodes": "no"}', "looks-reload"]},
    {"name": "changes from the menu publish", "commands": ["state"],
     "after": [["set", "hz", 90], ["set", "hz", 90], ["override", "interp", False], ["override", "interp", False],
               ["set", "simulate_30", True], ["set", "fsr_sharpness", 0]]},
]


def gen_state() -> None:
    import threading  # noqa: F401

    out_cases = []
    for case in STATE_CASES:
        reset_looks()
        if case.get("user_looks"):
            with contextlib.redirect_stdout(io.StringIO()):
                Library(user_library(USER_LOOKS))
        d = fresh_dir("state")
        path = d / "settings.json"
        if case.get("file") is not None:
            path.write_text(case["file"], encoding="utf-8")
        st = Settings.load(path)
        for k, v in case.get("overrides", []):
            st.override(k, v)
        deny = case.get("deny", [])
        dis = case.get("disabled", {})
        restart = case.get("restart", [])
        shown = []

        def look_show(text, shown=shown):
            shown.append(json.dumps(normalize(json.loads(text))))

        handlers = {"look-show": look_show, "look-hide": lambda t, shown=shown: shown.append("hide"),
                    "looks-reload": lambda t, shown=shown: shown.append("reload")}
        out = io.StringIO()
        cmds = case.get("commands", [])
        r = remote.Remote(st, lambda k, v: [k, v] not in deny, lambda k: dis.get(k), lambda: list(restart),
                          commands=io.StringIO("".join(c + "\n" for c in cmds)), out=out, handlers=handlers)
        want = sum(1 for c in cmds if c.strip().lstrip("\ufeff"))
        for _ in range(400):
            if r._q.qsize() == want:
                break
            time.sleep(0.005)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            r.publish()  # the session's first line
            got = r.poll()
            for op in case.get("after", []):
                (st.set if op[0] == "set" else st.override)(op[1], op[2])
        out_cases.append({
            **{k: v for k, v in case.items() if k != "user_looks"},
            "user_looks": USER_LOOKS if case.get("user_looks") else None,
            "lines": out.getvalue().split("\n"),
            "log": log.getvalue().split("\n"),
            "poll": got,
            "shown": shown,
            "values": json.dumps(st.values()),
            "file_after": path.read_text(encoding="utf-8") if path.exists() else None,
        })
    reset_looks()
    write("state", out_cases)


# ------------------------------------------------------------------------------------------------ looks
NAMES = ["", "   ", "A look", "  A   look ", "x" * 70, "caf\u00e9 \u00e9t\u00e9", "\u00e9" * 30, "\U0001F600" * 13,
         "tab\tand\nnewline\u2003em space\u00a0nbsp\x1cfs", "\u4e2d\u6587\u540d\u5b57" * 5, "Kelvin \u212a", None,
         0, 1, 2.5, True, False, [], ["a", 1], {"k": "v"}, "My look", "\u200b", "a\x00b"]


def random_value(rng: random.Random, depth: int = 0):
    r = rng.random()
    if r < 0.25:
        return rng.uniform(-1000, 1000) if rng.random() < 0.5 else rng.uniform(-2, 3)
    if r < 0.35:
        return rng.randint(-500, 500)
    if r < 0.45:
        return rng.choice([True, False, None])
    if r < 0.6:
        return rng.choice(["0.5", " 1.25 ", "1_000", "nan", "inf", "-1e9", "x", "", "\u0661.5", "1e-3", "+.5",
                           "5.", "1__0", "0x10", "Infinity", "-iNf", "3\u2003"])
    if r < 0.7 and depth < 2:
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 2))]
    if r < 0.75 and depth < 2:
        return {"a": random_value(rng, depth + 1)}
    if r < 0.8:
        return rng.choice([float("nan"), float("inf"), -float("inf"), 1e300, -0.0, 10 ** 20])
    return rng.uniform(-5, 400)


def messy_look(rng: random.Random) -> dict:
    nodes = []
    for _ in range(rng.randint(0, 20)):
        r = rng.random()
        if r < 0.05:
            nodes.append(random_value(rng))
            continue
        t = rng.choice(TYPES_LIST).key if r < 0.9 else rng.choice(["warp-drive", "", "COLOR", 5, None])
        n = {"type": t}
        if rng.random() < 0.7:
            n["on"] = random_value(rng) if rng.random() < 0.3 else rng.random() > 0.2
        if rng.random() < 0.7:
            n["mix"] = random_value(rng)
        tt = looks.TYPES.get(t) if isinstance(t, str) else None
        for f in (tt.params if tt else ()):
            if rng.random() < 0.7:
                if f.text:
                    n[f.key] = rng.choice(["My grade.cube", "..\\..\\x\\grade.CUBE", "C:grade.cube", "a/b/c.cube",
                                           "notalut.png", "", None, 5, " spaced .cube ", "\\\\server\\share",
                                           "\\\\server\\share\\x.cube", "x.cube/", "x.cube/.", "." * 3 + ".cube",
                                           "y" * 130 + ".cube", "\\\\?\\UNC\\s\\h\\deep.cube", "Gr\u00e4de.cube"])
                else:
                    n[f.key] = random_value(rng) if rng.random() < 0.4 else rng.uniform(f.lo - 1, f.hi + 1)
        if rng.random() < 0.1:
            n["extra"] = 1
        nodes.append(n)
    look = {"name": rng.choice(NAMES), "nodes": nodes}
    if rng.random() < 0.1:
        del look["name"]
    if rng.random() < 0.05:
        look["version"] = 1
    return look


def try_normalize(v) -> dict:
    try:
        return {"look": json.dumps(normalize(v))}
    except Exception as e:  # noqa: BLE001 - TypeError and OverflowError too, which the Rust port reports
        return err(e)


def gen_normalize() -> None:
    rng = random.Random(7)
    inputs = [
        "\"not a look\"", "null", "[]", "{}", "{\"nodes\": null}", "{\"nodes\": {}}", "{\"nodes\": \"x\"}",
        "{\"name\": \"x\"}", "{\"nodes\": [{\"type\": [\"color\"]}]}", "{\"nodes\": [{\"type\": {\"a\": 1}}]}",
        "{\"nodes\": [{\"type\": \"color\", \"contrast\": 1" + "0" * 400 + "}]}",
        "{\"nodes\": [{\"type\": \"color\", \"contrast\": 123456789012345678901234567890}]}",
        json.dumps({"name": "  A   look ", "nodes": [
            {"type": "color", "contrast": 7, "saturation": "x", "exposure": 0.123},
            {"type": "warp-drive"}, "junk",
            {"type": "lut", "file": "..\\..\\secret\\grade.cube"},
            {"type": "lut", "file": "notalut.png"}] + [{"type": "grain"}] * 30}),
        json.dumps({"nodes": [{"type": t.key} for t in TYPES_LIST]}),
        json.dumps({"nodes": [{"type": t.key, **{f.key: f.hi + 10 for f in t.params if not f.text}}
                              for t in TYPES_LIST]}),
        json.dumps({"nodes": [{"type": t.key, **{f.key: f.lo - 10 for f in t.params if not f.text}}
                              for t in TYPES_LIST]}),
        json.dumps({"nodes": [{"type": "color", "exposure": x} for x in
                              (0.025, 0.075, 0.125, -0.025, 2.975, 0.1 + 0.2, 1 / 3, 0.0078125, 0.005, 0.015)]}),
        json.dumps({"nodes": [{"type": "wheels", "lift_hue": x} for x in (0.5, 1.5, 2.5, 359.5, 360.4, -0.5)]}),
        json.dumps({"nodes": [{"type": "assist", "kind": x} for x in (0.5, 1.5, 2.5, -1, 3, "1", True)]}),
    ]
    for name in NAMES:
        inputs.append(json.dumps({"name": name, "nodes": []}))
    for _ in range(400):
        inputs.append(json.dumps(messy_look(rng)))
    write("normalize", [{"input": text, **try_normalize(json.loads(text))} for text in inputs])


def random_look(rng: random.Random) -> dict:
    nodes = []
    for _ in range(rng.randint(0, 18)):
        t = rng.choice(TYPES_LIST)
        n = node(t.key, on=bool(rng.random() > 0.2), mix=float(rng.random()) if rng.random() < 0.6 else 1.0)
        for f in t.params:
            if rng.random() < 0.6:
                n[f.key] = (rng.choice(["My grade.cube", "Gr\u00e4de \u2603.cube", "z" * 150 + ".cube", ""])
                            if f.text else float(rng.uniform(f.lo, f.hi)))
        nodes.append(n)
    return {"name": rng.choice([n for n in NAMES if isinstance(n, str)] + ["Random", "x" * 47, "\u00e9" * 24]),
            "nodes": nodes}


def make_code(body: bytes, flags: int | None = None, compress: bool = False) -> str:
    if compress:
        p = zlib.compressobj(9, zlib.DEFLATED, -15)
        body = p.compress(body) + p.flush()
    data = bytes([(1 if compress else 0) if flags is None else flags]) + body
    data += (zlib.crc32(data) & 0xFFFF).to_bytes(2, "big")
    return "SPL1-" + base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def try_decode(code: str) -> dict:
    try:
        return {"code": code, "look": json.dumps(decode(code))}
    except looks.CodeError as e:
        return {"code": code, "error": str(e)}


def gen_codes() -> None:
    rng = random.Random(11)
    builtin = [{"id": k, "look": json.dumps(v), "code": encode(v)} for k, v in BUILTIN.items()]
    edited = []
    for _ in range(300):
        raw = random_look(rng)
        edited.append({"input": json.dumps(raw), "look": json.dumps(normalize(raw)), "code": encode(raw)})
    for raw in ({"name": "x", "nodes": [node("lut", file="a" * 200 + ".cube")]},
                {"name": "\u00e9" * 40, "nodes": []}, {"nodes": [node(t.key) for t in TYPES_LIST]}):
        edited.append({"input": json.dumps(raw), "look": json.dumps(normalize(raw)), "code": encode(raw)})
    codes = [e["code"] for e in builtin] + [e["code"] for e in edited[:40]]
    bad = []
    film = encode(BUILTIN["film"])
    bad += [f"  {film[:20]}\n{film[20:]} \n", film.lower().replace("spl1-", "SPL1-", 1)[:5] + film[5:],
            "spl1-" + film[5:], "\u017fPL1-" + film[5:], "SPL01-" + film[5:], "SPL\u0661-" + film[5:],
            "hello", "SPL2-AAAA", "SPL1-", "SPL1-garbage", "SPL-AAAA", "SPL1-AAAA=", "SPL1-AA\u212aA", film[:-3],
            film + "A", film + "AA", film[:12] + ("A" if film[12] != "A" else "B") + film[13:], "SPL1-AAAAA",
            film + "\u2003", "SPL1-" + "A" * 3, "SPL1-" + "_-" * 10]
    for code in codes[:30]:
        for _ in range(4):
            i = rng.randrange(5, len(code))
            c = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
            bad.append(code[:i] + c + code[i + 1:])
        bad.append(code[:rng.randrange(5, len(code))])
    # hand-made bodies (test_codes_from_a_newer_version and more), with a valid checksum
    vig = bytes([13 << 1])
    bodies = [
        (b"\x01X\x01" + vig + bytes([1 << 6]) + b"\x04", None, False),  # a setting this version doesn't have
        (b"\x01X\x01" + bytes([99 << 1]) + b"\x00", None, False),  # an effect it doesn't have
        (b"\x01X\x01" + vig + b"\x00", 2, False),  # flags from a newer version
        (b"\x01X\x01" + vig + b"\x00", 3, False),
        (b"\x01X\x01" + vig + b"\x00\x00", None, False),  # extra bytes
        (b"\x01X\x02" + vig + b"\x00", None, False),  # cut short
        (b"\x05X", None, False),  # a name longer than the body
        (b"\x02\xff\xfe\x00", None, False),  # a name that isn't UTF-8
        (b"\x00\x00", None, False),
        (b"\x00" + b"\x80" * 12, None, False),  # a varint that's too long
        (b"\x00\x01" + vig + b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01" + b"\x00" * 80, None, False),  # a huge mask
        (b"\x00\x01" + vig + bytes([1 << 1]) + b"\xff\xff\xff\xff\x0f", None, False),  # a huge step count
        (b"\x00\x01" + bytes([5 << 1]) + bytes([1 << 1]) + b"\x0aMy.cube.x", None, False),
        (b"\x00\x01" + bytes([5 << 1]) + bytes([1 << 1]) + b"\x08..\\a.CUBE", None, False),
        (b"\x00\x01" + bytes([5 << 1 | 1]) + bytes([1]) + b"\x63", None, False),  # off, mix
        (b"\x03abc\x14" + (vig + b"\x00") * 20, None, False),  # 20 nodes, 16 kept
        (b"\x03abc\x01" + vig + b"\x00", None, True),  # compressed
        (b"\x03abc\x01" + vig + b"\x00", 1, False),  # says compressed, isn't
        (b"\x00\x00", 0, True),  # compressed, says it isn't
    ]
    for body, flags, comp in bodies:
        bad.append(make_code(body, flags, comp))
    decode_cases = [try_decode(c) for c in codes + bad]
    write("codes", {"builtin": builtin, "edited": edited, "decode": decode_cases})


def gen_uniforms() -> None:
    rng = random.Random(5)
    nodes = [n for look in BUILTIN.values() for n in look["nodes"]]
    for t in TYPES_LIST:
        for _ in range(6):
            n = {"type": t.key, "mix": rng.random()}
            for f in t.params:
                n[f.key] = "Test.cube" if f.text else rng.uniform(f.lo, f.hi)
            nodes.append(normalize({"nodes": [n]})["nodes"][0])
    lut = looks.Lut(17, None, (0.0, -0.25, 0.5), (1.0, 2.0, 0.5))
    lut2 = looks.Lut(33, None, (0.1, 0.2), (0.9, 0.8))
    cases = []
    for n in nodes:
        entry = {"node": json.dumps(n), "uniforms": json.dumps(looks.uniforms(n))}
        if n["type"] == "lut":
            entry["with_lut"] = json.dumps(looks.uniforms(n, lut))
            entry["with_lut2"] = json.dumps(looks.uniforms(n, lut2))
        if TYPES_LIST[[t.key for t in TYPES_LIST].index(n["type"])].kind in ("glow", "blur", "clarity"):
            entry["glow"] = json.dumps(looks.glow(n))
        cases.append(entry)
    stage_looks = list(BUILTIN.values())
    for _ in range(60):
        raw = random_look(rng)
        for n in raw["nodes"]:
            if n["type"] == "lut" and rng.random() < 0.5:
                n["file"] = rng.choice(["Missing.cube", "Here.cube"])
        stage_looks.append(raw)
    stages = []
    for raw in stage_looks:  # normalize() once, as it isn't idempotent for a LUT name cut at 120 characters
        look = normalize(raw)
        st = looks.stages(look, lambda name: name != "Missing.cube")
        stages.append({"look": json.dumps(raw), "stages": [
            [s.start["type"] if s.start else None, [n["type"] for n in s.nodes]] for s in st]})
    hues = [x * 7.5 for x in range(-10, 60)] + [0.1, 359.9, 1e-9, 720.0, -0.0]
    write("uniforms", {
        "nodes": cases, "stages": stages,
        "hue_vec": [[h, json.dumps(looks.hue_vec(h))] for h in hues],
        "assist": [json.dumps(looks.assist_matrix(k)) for k in range(3)],
        "lut": {"size": 17, "dmin": [0.0, -0.25, 0.5], "dmax": [1.0, 2.0, 0.5]},
        "lut2": {"size": 33, "dmin": [0.1, 0.2], "dmax": [0.9, 0.8]},
    })


CUBES = [
    ("identity 2", "TITLE \"test\"\nLUT_3D_SIZE 2\n# a comment\n" + "\n".join(
        f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1))),
    ("1D", "LUT_1D_SIZE 4\n0 0 0\n"),
    ("too few rows", "LUT_3D_SIZE 2\n0 0 0\n"),
    ("no size", "hello\n"),
    ("size 1", "LUT_3D_SIZE 1\n0 0 0\n"),
    ("size 130", "LUT_3D_SIZE 130\n"),
    ("bad size", "LUT_3D_SIZE two\n"),
    ("size with underscore", "LUT_3D_SIZE 0_2\n" + "0 0 0\n" * 8),
    ("domain", "DOMAIN_MIN 0 -0.5 0.25\nDOMAIN_MAX 1 2.5 0.75 9\nlut_3d_size 2\n" + "0.5 0.25 1e-3\n" * 8),
    ("short domain", "DOMAIN_MIN 0.1\nDOMAIN_MAX\nLUT_3D_SIZE 2\n" + "+1 -1 .5\n" * 8),
    ("bad domain", "DOMAIN_MIN 0 x 0\nLUT_3D_SIZE 2\n" + "0 0 0\n" * 8),
    ("two numbers", "LUT_3D_SIZE 2\n" + "0 0 0\n" * 7 + "0 0\n"),
    ("words", "LUT_3D_SIZE 2\n" + "0 0 0\n" * 7 + "0 0 zero\n"),
    ("nan", "LUT_3D_SIZE 2\n" + "0 0 0\n" * 7 + "nan 0 0\n"),
    ("float32 overflow", "LUT_3D_SIZE 2\n" + "0 0 0\n" * 7 + "1e39 0 0\n"),
    ("extra columns and comments", "TITLE\tTabbed title\nLUT_3D_SIZE 2 # size\n" +
     "0.1 0.2 0.3 0.4 # four\n" * 8),
    ("CRLF and CR", "TITLE \"crlf\"\r\nLUT_3D_SIZE 2\r" + "0.3333333 0.6666667 0.1\r\n" * 8),
    ("title quoting", "TITLE   \"  Spaced  \"  \nLUT_3D_SIZE 2\n" + "1 1 1\n" * 8),
    ("keywords ignored", "LUT_3D_INPUT_RANGE 0 1\nLUT_3D_SIZE 2\n" + "0 0 0\n" * 8),
    ("unicode digits", "LUT_3D_SIZE \u0662\n" + "\u0661 0 0\n" * 8),
    ("underscores in colors", "LUT_3D_SIZE 2\n" + "1_0 0 0\n" * 8),
    ("random 5", None),
]


def gen_cube() -> None:
    rng = random.Random(3)
    d = fresh_dir("cube")
    cases = []
    for name, text in CUBES:
        if text is None:
            text = "LUT_3D_SIZE 5\n" + "\n".join(
                " ".join(f"{rng.uniform(-0.2, 1.2):.{rng.randint(1, 9)}f}" for _ in range(3)) for _ in range(125))
        p = d / "x.cube"
        p.write_text(text, encoding="utf-8", newline="")
        try:
            lut = looks.load_cube(p)
            got = {"size": lut.size, "title": lut.title, "dmin": list(lut.dmin), "dmax": list(lut.dmax),
                   "table": [float(x) for x in lut.table.reshape(-1)]}
        except ValueError as e:
            got = err(e)
        except Exception as e:  # noqa: BLE001 - IndexError, for example, which the Python doesn't catch
            got = err(e)
        cases.append({"name": name, "text": text, **got})
    write("cube", cases)


def run_cli(argv: list[str], stdin: str = "") -> dict:
    out = io.StringIO()
    old = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = looks.main(argv)
            except SystemExit as e:
                rc = e.code
    finally:
        sys.stdin = old
    return {"rc": rc, "out": out.getvalue()}


def gen_library_and_cli() -> None:
    from framegen.looks_preview import SAMPLES_DIR

    reset_looks()
    lib_dir = fresh_dir("library")
    src_dir = fresh_dir("src")
    subst = [(str(lib_dir), "{dir}"), (str(src_dir), "{src}"), (str(SAMPLES_DIR), "{samples}")]

    def place(s):
        for a, b in subst:
            s = s.replace(a, b)
        return s

    def unplace(s):
        for a, b in subst:
            s = s.replace(b, a)
        return s

    def snapshot(lib) -> dict:
        files = {}
        for p in sorted(lib_dir.iterdir()) if lib_dir.exists() else []:
            files[p.name] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
        return {"choices": list(looks.CHOICES), "names": [[k, v] for k, v in looks.NAMES.items()], "files": files}

    # ---- the library, step by step
    steps = []
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lib = Library(lib_dir)
    script = [
        ["save", {"name": "Moody blues", "nodes": [node("color", temperature=-0.4)]}, None],
        ["save", {"name": "Moody blues", "nodes": []}, None],
        ["save", {"name": "Renamed", "nodes": []}, "my:moody-blues"],
        ["save", {"name": "\u00c9t\u00e9 \u2603 caf\u00e9!!", "nodes": [node("grain")]}, None],
        ["save", {"name": "\u2603\u2603", "nodes": []}, None],
        ["save", {"name": "x" * 70, "nodes": []}, None],
        ["save", {"name": "a" * 39 + " b", "nodes": []}, None],
        ["save", {"name": "From builtin id", "nodes": []}, "cinematic"],
        ["save", {"name": "Dots", "nodes": []}, "my:a.b"],
        ["save", {"name": "Empty id", "nodes": []}, ""],
        ["save", "not a look", None],
        ["write", "broken.json", "{nope"],
        ["write", "list.json", "[1, 2]"],
        ["write", "bom.json", "\ufeff{\"name\": \"bom\", \"nodes\": []}"],
        ["write_bytes", "latin.json", "7b226e616d65223a2022e9227d"],
        ["write", ".hidden.json", "{\"name\": \"Hidden\", \"nodes\": []}"],
        ["write", "UPPER.JSON", "{\"name\": \"Upper\", \"nodes\": []}"],
        ["write", "zz-first.json", "{\"name\": \"aaa first by name\", \"nodes\": []}"],
        ["mkdir", "folder.json"],
        ["reload"],
        ["delete", "my:moody-blues"],
        ["delete", "my:moody-blues"],
        ["delete", "cinematic"],
        ["delete", "my:..\\..\\x"],
        ["delete", "my:folder"],
        ["path", "my:..\\..\\x"], ["path", "my:a/b"], ["path", "cinematic"], ["path", "my:"], ["path", "my:x"],
    ]
    for step in script:
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                if step[0] == "save":
                    r = lib.save(step[1], step[2])
                elif step[0] == "write":
                    (lib_dir / step[1]).write_text(step[2], encoding="utf-8")
                    r = None
                elif step[0] == "write_bytes":
                    (lib_dir / step[1]).write_bytes(bytes.fromhex(step[2]))
                    r = None
                elif step[0] == "mkdir":
                    (lib_dir / step[1]).mkdir()
                    r = None
                elif step[0] == "reload":
                    lib.reload()
                    r = None
                elif step[0] == "delete":
                    r = lib.delete(step[1])
                else:
                    p = lib.path(step[1])
                    r = None if p is None else place(str(p))
            res = {"ok": r}
        except Exception as e:  # noqa: BLE001
            res = {"error": type(e).__name__, "message": place(str(e))}
        steps.append({"step": step, **res, **snapshot(lib), "log": place(log.getvalue())})
    (lib_dir / "folder.json").rmdir()

    # ---- LUTs
    good = "LUT_3D_SIZE 2\n" + "0 0 0\n" * 8
    (src_dir / "Warm.cube").write_text(good, encoding="utf-8")
    (src_dir / "Shout.CUBE").write_text(good, encoding="utf-8")
    (src_dir / "bad.cube").write_text("hello\n", encoding="utf-8")
    (src_dir / "x.png").write_text(good, encoding="utf-8")
    lut_steps = []
    for name in ("Warm.cube", "Shout.CUBE", "bad.cube", "x.png", "missing.cube"):
        try:
            r = {"ok": lib.add_lut(src_dir / name)}
        except ValueError as e:
            r = err(e)
        lut_steps.append({"src": name, **r, "luts": lib.luts(),
                          "lut": [lib.lut(n).size if lib.lut(n) else None for n in ("Warm.cube", "Shout.CUBE",
                                                                                      "Missing.cube", "..\\Warm.cube")]})

    # ---- the command line, as the launcher runs it
    os.environ["SMOOVPLAY_LOOKS_DIR"] = str(lib_dir)
    shutil.rmtree(lib_dir)
    lib_dir.mkdir()
    calls = [
        (["schema"], ""),
        (["encode"], json.dumps(BUILTIN["noir"])),
        (["decode", encode(BUILTIN["noir"])], ""),
        (["decode", "SPL1-garbage"], ""),
        (["decode", "hello"], ""),
        (["save"], json.dumps({"name": "From the launcher", "nodes": []})),
        (["save"], "{not json"),
        (["save"], "\"not a look\""),
        (["save", "my:from-the-launcher"], json.dumps({"name": "Renamed", "nodes": [node("ca")]})),
        (["schema"], ""),
        (["encode", "{dir}/from-the-launcher.json"], ""),
        (["encode", "{dir}/missing.json"], ""),
        (["encode"], "[1, 2]"),
        (["encode"], ""),
        (["import", encode(BUILTIN["vhs"])], ""),
        (["import", "SPL1-AAAA"], ""),
        (["add-lut", "{src}/Warm.cube"], ""),
        (["add-lut", "{src}/bad.cube"], ""),
        (["add-lut", "{src}/x.png"], ""),
        (["add-lut", "{src}/missing.cube"], ""),
        (["delete", "my:from-the-launcher"], ""),
        (["delete", "my:from-the-launcher"], ""),
        (["schema"], ""),
        ([], ""),
        (["frobnicate"], ""),
        (["decode"], ""),
    ]
    cli = []
    for argv, stdin in calls:
        r = run_cli([unplace(a) for a in argv], stdin)
        cli.append({"args": argv, "stdin": stdin, "rc": r["rc"], "out": place(r["out"])})
    os.environ["SMOOVPLAY_LOOKS_DIR"] = str(TMP / "looks-at-import")
    reset_looks()
    write("library", {"steps": steps, "luts": lut_steps, "cli": cli})


# ------------------------------------------------------------------------------------------------ fill
def gen_fill() -> None:
    sizes = [(1920, 1080), (2560, 1440), (3440, 1440), (2560, 1080), (5120, 1440), (1920, 1200), (2560, 1600),
             (1280, 1024), (3840, 2160), (6016, 3384), (860, 360), (1, 1), (1081, 607), (3024, 1964), (800, 1280),
             (5120, 2160), (1366, 768), (1921, 1080), (1920, 1081)]
    pics = [(1920, 1080), (1280, 720), (2560, 1440), (640, 480)]
    cases = []
    for fw, fh in sizes:
        for w, h in pics:
            row = {"screen": [fw, fh], "picture": [w, h], "wider": fill.wider(fw, fh, w, h),
                   "panorama_t": fill.panorama_t(fw, fh, w, h)}
            for f in fill.FILLS:
                lay = fill.layout(f, fw, fh, w, h)
                row[f] = [list(lay), list(fill.visible(*lay, fw, fh))]
            t = row["panorama_t"]
            row["u"] = [fill.panorama_u(x, t) for x in (-1.0, -0.73, -0.5, -0.1, 0.0, 0.3, 0.999, 1.0)]
            cases.append(row)
    write("fill", cases)


# ------------------------------------------------------------------------------------------------ Auto-detect
def timing(full: float, half: float, full_engine: str = "trt", estimated: bool = False) -> dict:
    return {"full": {"ms": full, "engine": full_engine, "estimated": estimated},
            "half": {"ms": half, "engine": "torch", "estimated": False}}


D = autodetect.Display
UW120 = [3440, 1440, 120, 120]
THIS_PC = timing(9.3, 6.8)
NOT_SET_UP = "NVIDIA VFG isn't set up on this PC. SmoovPlay's Settings page downloads it from NVIDIA."

RECOMMEND = [  # name, current, timing, display, kw (test_autodetect's pick() passes trt=True)
    ("tested card", {}, THIS_PC, UW120, {"trt": True}),
    ("60 Hz display", {}, THIS_PC, [2560, 1440, 60, 60], {"trt": True}),
    ("100 Hz display", {}, THIS_PC, [2560, 1440, 100, 100], {"trt": True}),
    ("119 Hz", {}, timing(5.0, 3.5), [2560, 1440, 119, 119], {"trt": True}),
    ("set below what it can do", {}, THIS_PC, [2560, 1440, 60, 144], {"trt": True}),
    ("fast card at 144", {}, timing(5.0, 3.5), [2560, 1440, 144, 144], {"trt": True}),
    ("mid card", {}, timing(12.0, 8.5), [2560, 1440, 144, 144], {"trt": True}),
    ("slow card", {}, timing(20.0, 13.0), [2560, 1440, 144, 144], {"trt": True}),
    ("too slow", {}, timing(40.0, 30.0), UW120, {"trt": True}),
    ("desktop margin", {}, timing(8.9, 6.6), UW120, {"trt": True}),
    ("laptop", {}, timing(8.9, 6.6), UW120, {"trt": True, "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU"}),
    ("vsr doesn't fit", {"scaler": "vsr"}, THIS_PC, UW120, {"trt": True}),
    ("vsr fits", {"scaler": "vsr"}, THIS_PC, [2560, 1440, 60, 60], {"trt": True}),
    ("vsr not set up", {"scaler": "vsr"}, THIS_PC, [2560, 1440, 60, 60],
     {"trt": True, "vsr": "RTX Video SDK not set up"}),
    ("vsr empty reason", {"scaler": "vsr"}, THIS_PC, [2560, 1440, 60, 60], {"trt": True, "vsr": ""}),
    ("vsr on a 1080p display", {"scaler": "vsr"}, THIS_PC, [1920, 1080, 60, 60], {"trt": True}),
    ("fsr never swapped", {}, timing(3.0, 2.0), [2560, 1440, 60, 60], {"trt": True}),
    ("no TensorRT, no display", {}, timing(10.4, 6.8, "torch"), None, {"trt": False}),
    ("estimated", {}, timing(9.3, 6.8, estimated=True), UW120, {"trt": True}),
    ("busy", {}, THIS_PC, UW120, {"trt": True, "busy": 85}),
    ("not busy", {}, THIS_PC, UW120, {"trt": True, "busy": 3}),
    ("DirectML", {}, timing(14.0, 9.0, "dml"), [2560, 1440, 144, 144], {"gpu": "AMD Radeon RX 7800 XT"}),
    ("Mac: M5 Max on a 160 Hz 6K", {}, timing(7.6, 5.0, "torch"), [6016, 3384, 160, 160], {"gpu": "Apple M5 Max"}),
    ("Mac: M5 Max on a 120 Hz MacBook", {}, timing(7.6, 5.0, "torch"), [3024, 1964, 120, 120],
     {"gpu": "Apple M5 Max"}),
    ("Mac: M5 Max on a 60 Hz 4K", {}, timing(7.6, 5.0, "torch"), [3840, 2160, 60, 60], {"gpu": "Apple M5 Max"}),
    ("Mac: slower chip", {}, timing(19.0, 12.5, "torch"), [2560, 1440, 144, 144], {"gpu": "Apple M3"}),
    ("Mac: no display", {}, timing(7.6, 5.0, "torch"), None, {"gpu": "Apple M5 Max"}),
    ("Mac: was set to vfg", {"model": "vfg"}, timing(7.6, 5.0, "torch"), [6016, 3384, 160, 160],
     {"gpu": "Apple M5 Max", "vfg_why": "NVIDIA VFG needs an NVIDIA RTX card"}),
    ("vfg picked", {}, THIS_PC, UW120, {"trt": True, "vfg": {"low": {"ms": 5.0}, "medium": {"ms": 6.0},
                                                              "high": {"ms": 7.4}}}),
    ("vfg slower than rife", {}, THIS_PC, UW120, {"trt": True, "vfg": {"low": {"ms": 9.0}, "medium": {"ms": 12.0},
                                                                        "high": {"ms": 15.0}}}),
    ("vfg can't keep up", {}, THIS_PC, UW120, {"trt": True, "vfg": {"low": {"ms": 30.0}, "medium": {"ms": 40.0},
                                                                     "high": {"ms": 50.0}}}),
    ("vfg at 90 on a 144", {}, timing(12.0, 8.5), [2560, 1440, 144, 144],
     {"trt": True, "vfg": {"low": {"ms": 8.5}, "medium": {"ms": 9.5}, "high": {"ms": 12.0}}}),
    ("vfg on a 60 Hz", {"scaler": "vsr"}, THIS_PC, [2560, 1440, 60, 60],
     {"trt": True, "vfg": {"low": {"ms": 5.0}, "medium": {"ms": 6.0}, "high": {"ms": 7.4}}}),
    ("was vfg, not set up", {"model": "vfg"}, THIS_PC, UW120, {"trt": True, "vfg_why": NOT_SET_UP}),
    ("was vfg, no reason", {"model": "vfg"}, THIS_PC, UW120, {"trt": True}),
]


def recommend_case(name, current, t, display, kw) -> dict:
    disp = D(*display) if display else None
    r = autodetect.recommend({**DEFAULTS, **current}, t, disp, **kw)
    return {"name": name, "current": current, "timing": t, "display": display, "kw": kw, "result": json.dumps(r)}


def gen_recommend() -> None:
    cases = [recommend_case(*c) for c in RECOMMEND]
    rng = random.Random(9)
    displays = [None, [1920, 1080, 60, 60], [1920, 1080, 144, 144], [2560, 1440, 60, 144], [2560, 1440, 165, 165],
                [3440, 1440, 120, 120], [3440, 1440, 100, 100], [3840, 2160, 60, 60], [3840, 2160, 120, 120],
                [3840, 2160, 144, 144], [6016, 3384, 160, 160], [5120, 2880, 60, 60], [3024, 1964, 120, 120],
                [2560, 1600, 120, 120], [1366, 768, 60, 60], [1920, 1200, 75, 75], [2560, 1080, 200, 240],
                [7680, 2160, 240, 240]]
    gpus = ["NVIDIA GeForce RTX 4070 Ti SUPER", "NVIDIA GeForce RTX 4080 Laptop GPU", "Apple M5 Max", "Apple M2",
            "AMD Radeon RX 7800 XT", "", "NVIDIA GeForce RTX 3060 LAPTOP"]
    for _ in range(400):
        full = round(rng.uniform(2.0, 45.0), rng.choice([1, 2]))
        half = round(full * rng.uniform(0.5, 0.85), 2)
        engine = rng.choice(["trt", "torch", "dml"])
        t = timing(full, half, engine, rng.random() < 0.15)
        current = {}
        if rng.random() < 0.3:
            current["scaler"] = rng.choice(["vsr", "fsr", "bilinear"])
        if rng.random() < 0.2:
            current["model"] = "vfg"
        kw = {"gpu": rng.choice(gpus), "trt": rng.random() < 0.5}
        if rng.random() < 0.3:
            kw["vsr"] = rng.choice([None, "RTX Video SDK not set up", ""])
        if rng.random() < 0.3:
            kw["busy"] = rng.choice([None, 0, 29, 30, 99])
        if rng.random() < 0.25:
            lo = rng.uniform(3, 30)
            kw["vfg"] = {"low": {"ms": lo}, "medium": {"ms": lo * 1.2}, "high": {"ms": lo * rng.uniform(1.3, 2)}}
        if rng.random() < 0.2:
            kw["vfg_why"] = rng.choice([NOT_SET_UP, "NVIDIA VFG needs an RTX 40 or 50 series card.", ""])
        cases.append(recommend_case(f"sweep {len(cases)}", current, t, rng.choice(displays), kw))
    write("recommend", cases)


def gen_detect() -> None:
    from types import SimpleNamespace

    import torch

    from framegen import gpu as gpus
    from framegen import rtxvsr, trt, vfg

    saved = {"is_available": torch.cuda.is_available, "get_device_name": torch.cuda.get_device_name,
             "get_device_properties": torch.cuda.get_device_properties, "primary_display": autodetect.primary_display,
             "nvidia_smi": autodetect.nvidia_smi, "trt_available": trt.trt_available,
             "unavailable_reason": rtxvsr.unavailable_reason, "vfg_unavailable": vfg.vfg_unavailable,
             "MAC": gpus.MAC, "name": gpus.name, "mem": autodetect._mac_memory_gb}
    cases = []
    try:
        nvidia = [
            ("RTX 4070, VFG not set up", "NVIDIA GeForce RTX 4070", 12, UW120, "621.24", 0, True, None, NOT_SET_UP,
             THIS_PC, None),
            ("RTX 4070, old driver", "NVIDIA GeForce RTX 4070", 12, UW120, "566.36", 0, True, None, NOT_SET_UP,
             THIS_PC, None),
            ("RTX 3080, no VFG", "NVIDIA GeForce RTX 3080", 10, [2560, 1440, 60, 144], "621.24", 45, False,
             "RTX Video SDK not set up", "NVIDIA VFG needs an RTX 40 or 50 series card.", timing(12.0, 8.0), None),
            ("RTX 4090, VFG set up", "NVIDIA GeForce RTX 4090", 24, [3840, 2160, 144, 144], None, None, True, None,
             "", timing(5.5, 3.9, estimated=True), {"low": {"ms": 4.0}, "medium": {"ms": 4.6}, "high": {"ms": 6.1}}),
            ("unknown display", "NVIDIA GeForce RTX 4060 Laptop GPU", 8, None, "r570", None, False, None,
             "NVIDIA VFG couldn't be checked: boom", timing(14.0, 9.5), None),
        ]
        for (name, gpu, vram, display, driver, busy, has_trt, vsr, why, t, vf) in nvidia:
            torch.cuda.is_available = lambda: True
            torch.cuda.get_device_name = lambda i=0, gpu=gpu: gpu
            torch.cuda.get_device_properties = lambda i=0, vram=vram: SimpleNamespace(total_memory=vram * 2**30)
            autodetect.primary_display = lambda display=display: D(*display) if display else None
            smi = {"driver": driver, "busy": busy} if busy is not None else ({"driver": driver} if driver else {})
            autodetect.nvidia_smi = lambda samples=3, smi=smi: smi
            trt.trt_available = lambda has_trt=has_trt: has_trt
            rtxvsr.unavailable_reason = lambda vsr=vsr: vsr
            vfg.vfg_unavailable = lambda why=why: why
            r = autodetect.detect(DEFAULTS, log=lambda m: None, timing=t, vfg=vf)
            cases.append({"name": name, "machine": {
                "display": display, "driver": smi.get("driver"), "busy": smi.get("busy"), "gpu": gpu,
                "vram_gb": float(vram), "mac_mode": False, "dml_mode": False, "trt": has_trt, "vsr": vsr,
                "vfg_why": why}, "current": {}, "timing": t, "vfg": vf, "result": json.dumps(r)})
        mac = [
            ("Mac: M5 Max on a 160 Hz 6K", "Apple M5 Max", 64.0, [6016, 3384, 160, 160], timing(7.6, 5.0, "torch"),
             {}),
            ("Mac: M5 Max on its own screen", "Apple M5 Max", 128.0, [3024, 1964, 120, 120],
             timing(7.6, 5.0, "torch"), {"model": "vfg", "scaler": "vsr"}),
            ("Mac: unknown display", "Apple M1", 8.0, None, timing(30.0, 20.0, "torch"), {}),
        ]
        torch.cuda.is_available = lambda: False
        gpus.MAC = True
        for name, gpu, mem, display, t, current in mac:
            gpus.name = lambda gpu=gpu: gpu
            autodetect._mac_memory_gb = lambda mem=mem: mem
            autodetect.primary_display = lambda display=display: D(*display) if display else None
            autodetect.nvidia_smi = lambda samples=3: {}
            r = autodetect.detect({**DEFAULTS, **current}, log=lambda m: None, timing=t)
            cases.append({"name": name, "machine": {
                "display": display, "driver": None, "busy": None, "gpu": gpu, "vram_gb": mem, "mac_mode": True,
                "dml_mode": False, "trt": False, "vsr": "it needs an NVIDIA RTX card",
                "vfg_why": "NVIDIA VFG needs an NVIDIA RTX card"}, "current": current, "timing": t, "vfg": None,
                "result": json.dumps(r)})
    finally:
        torch.cuda.is_available = saved["is_available"]
        torch.cuda.get_device_name = saved["get_device_name"]
        torch.cuda.get_device_properties = saved["get_device_properties"]
        autodetect.primary_display = saved["primary_display"]
        autodetect.nvidia_smi = saved["nvidia_smi"]
        trt.trt_available = saved["trt_available"]
        rtxvsr.unavailable_reason = saved["unavailable_reason"]
        vfg.vfg_unavailable = saved["vfg_unavailable"]
        gpus.MAC = saved["MAC"]
        gpus.name = saved["name"]
        autodetect._mac_memory_gb = saved["mem"]

    # ---- the rest: auto_scale, the VFG offer, the timing file
    scale = []
    for w, h in ((1920, 1080), (2560, 1440), (3840, 2160), (1280, 720), (1920, 1081)):
        for hz in (60, 90, 120):
            for full in (None, 5.0, 9.3, 10.0, 10.5, 14.4, 15.0, 19.7, 40.0):
                for share in (autodetect.SHARE, autodetect.LAPTOP_SHARE):
                    scale.append([w, h, hz, full, share, autodetect.auto_scale(w, h, hz, full, share)])
    offers = []
    for gpu in ("NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 4070", "NVIDIA ", "  GeForce  ", "Apple M5 Max", ""):
        for driver in ("621.24", "566.36", "570.65", "570.64", "570", "571", None, "", "r570", " 570 . 64 ",
                       "570.64.01", "1_000.0", "\u0665\u0667\u0660.1"):
            offers.append([gpu, driver, autodetect.vfg_offer(True, gpu, driver), autodetect.vfg_driver_too_old(driver)])
    offers.append(["NVIDIA GeForce RTX 3080", "621.24", autodetect.vfg_offer(False, "NVIDIA GeForce RTX 3080", "621.24"),
                   False])
    d = fresh_dir("timing")
    sp = d / "new" / "settings.json"
    autodetect.save_timing(sp, "NVIDIA GeForce RTX 2080 Ti", timing(19.7, 14.5))
    saved_text = autodetect.timing_path(sp).read_text(encoding="utf-8")
    reads = []
    for text in (saved_text, "{not json", "[1]", json.dumps({"gpu": "X", "timing": {"full": {"ms": "19.5"}}}),
                 json.dumps({"gpu": "X", "timing": {"full": {"ms": None}}}),
                 json.dumps({"gpu": "X", "timing": {"full": {"ms": True}}}),
                 json.dumps({"gpu": "X", "timing": {"full": {}}}), json.dumps({"gpu": "X", "timing": []}),
                 json.dumps({"gpu": "X", "timing": {"full": {"ms": 12}}}), json.dumps({"timing": {}})):
        autodetect.timing_path(sp).write_text(text, encoding="utf-8")
        row = [text]
        for gpu in ("NVIDIA GeForce RTX 2080 Ti", "X"):
            try:
                row.append(autodetect.saved_full_ms(sp, gpu))
            except Exception as e:  # noqa: BLE001 - AttributeError on a list, which the Python doesn't catch
                row.append(type(e).__name__)
        reads.append(row)
    write("detect", {"detect": cases, "auto_scale": scale, "offers": offers, "timing_file": saved_text,
                     "timing_reads": reads})


# ------------------------------------------------------------------------------------------------ JSON and Python
def gen_pyjson() -> None:
    rng = random.Random(13)
    texts = ["{}", "[]", "0", "-0", "-0.0", "1.5e300", "1E400", "-1e-400", "123456789012345678901234567890",
             "1e16", "1e15", "0.1", "[1, 2.0, true, false, null]", "{\"a\": {\"b\": [{}, []]}}", "\"\\u00e9\"",
             "\"\\ud83d\\ude00\"", "\"\\ud83d\"", "\"\\ude00\\ud83d\"", "\"\\ud83dx\"", "\"a\\/b\\\\c\\\"d\\b\\f\\n\\r\\t\"",
             "\"\\u0000\\u001f\\u007f\"", "NaN", "Infinity", "-Infinity", "[NaN, -Infinity]", "  {\"a\" : 1 }  ",
             "{\"a\": 1, \"a\": 2, \"b\": 3, \"a\": 4}", "{\"\": 0}", "\"" + "\u00e9\u2603\U0001F600" * 3 + "\"",
             "{not json", "", " ", "[1,]", "{\"a\": 1,}", "{\"a\" 1}", "[1 2]", "\"abc", "\"a\\x\"", "\"\\u12\"",
             "\"\\u12g4\"", "\"\\u0041", "\"a\tb\"", "01", "1.", ".5", "-", "-a", "1e", "1e+", "tru", "nul",
             "{\"a\": tru}", "\ufeff{}", "{\n\"a\": [1,\n 2,, 3]}", "[\"x\"\n,\n]", "1 2", "{} {}", "[]]",
             "{\"a\": 1}}", "\"\\uD83D\\uDE00\"", "\"\\u00E9\"", "[1e5, 1E-5, 2.50, 100.0, 1e22, 1e21]",
             "[1.0000000000000002, 0.30000000000000004, 5e-324, 1.7976931348623157e308]", "[" * 5 + "]" * 5,
             "{\"a\":\"b\"" + "," * 1 + "}", "\"\\\"", "[-0, 0e0, -0e-0]", "9007199254740993", "-9223372036854775809",
             "9223372036854775807", "[\"\\u2028\\u2029\"]", "\"\x7f\""]
    loads = []
    for t in texts:
        try:
            v = json.loads(t)
            loads.append({"text": t, "dumps": json.dumps(v), "indent2": json.dumps(v, indent=2),
                          "indent1": json.dumps(v, indent=1), "repr": repr(v), "str": str(v)})
        except (ValueError, RecursionError) as e:
            loads.append({"text": t, **err(e)})
    floats = [0.0, -0.0, 1.0, 0.1, 1e16, 1e15, 1234567890123456.0, 12345678901234567.0, 1e-4, 1e-5, 0.00012345,
              5e-324, 1.7976931348623157e308, 2.5, 1 / 3, 2 / 3, 100.0, 1e22, 1e21, 123.456, -7.25e-7]
    for _ in range(500):
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(x):
            floats.append(x)
    for _ in range(300):
        floats.append(rng.uniform(-1e6, 1e6))
        floats.append(round(rng.uniform(-10, 10), rng.randint(0, 8)))
    float_cases = [[struct.pack("<d", x).hex(), repr(x), f"{x:.1f}", f"{x:.0f}", f"{x:.2f}", repr(round(x, 6)),
                    repr(round(x, 2)), int(round(x)) if abs(x) < 1e15 else None] for x in floats]
    strs = ["1", " 1.5 ", "1_000", "1__0", "_1", "1_", "5.", ".5", ".", "1e", "1e5", "1E+05", "-1e-5", "+.5e2",
            "inf", "-Infinity", "iNf", "nan", "-nan", "+NaN", "infinity ", "in_f", "0x10", "1e999", "-1e-999",
            "\u0661.5", "\u0967\u0968", "1.5\u2003", "\u20031.5", "1.5\u00a0", "\x1c1\x1d", "\u00bd", "\u00b2",
            "", " ", "1 2", "1.2.3", "--1", "+-1", "e5", "1e5.5", "0001", "1_0.5_5", "1._5", "1_.5", "١٢٣",
            "1" * 30, "0.1" + "0" * 30 + "1"]
    float_strs = []
    for s in strs:
        try:
            float_strs.append([s, repr(float(s))])
        except ValueError:
            float_strs.append([s, None])
    int_strs = []
    for s in ["1", "+9", "-0", "0_0", "1_7", " 12 ", "9.0", "", "_1", "1_", "\u0662", "1" * 40, "0x10", "1__2"]:
        try:
            int_strs.append([s, str(int(s))])
        except ValueError:
            int_strs.append([s, None])
    splits = []
    for s in ["  a  b ", "a\x1cb\x1dc\x1ed\x1fe", "a\u2028b\u2029c\x85d", "a\u200bb", "\u3000x\u3000", "",
              "tab\tand\nnew\rline\x0bvt\x0cff", "a\u00a0b\u2007c\u202fd\u205fe\u1680f"]:
        splits.append([s, s.split(), s.strip()])
    reprs = []
    for s in ["", "a", "it's", "\"q\"", "both ' and \"", "back\\slash", "\t\n\r", "\x00\x1f\x7f\x80\x9f\xa0\xad",
              "caf\u00e9", "\u2028\u2029", "\U0001F600", "\ue000", "\U000e0001", "\u0378", "\ufeff", "\u200b\u200e",
              "\u3000", "\uffff", "\U0010ffff", "\u0300"]:
        reprs.append([s, repr(s)])
    for _ in range(300):
        s = "".join(chr(rng.choice([rng.randint(0, 0x7f), rng.randint(0x80, 0x2fff), rng.randint(0x3000, 0xd7ff),
                                    rng.randint(0xe000, 0x10ffff)])) for _ in range(rng.randint(1, 6)))
        reprs.append([s, repr(s)])
    sums = []
    for _ in range(200):
        xs = [rng.uniform(-10, 10) * 10 ** rng.randint(-8, 8) for _ in range(rng.randint(1, 6))]
        sums.append([[struct.pack("<d", x).hex() for x in xs], struct.pack("<d", sum(xs)).hex()])
    deflate = []
    for _ in range(200):
        n = rng.randint(0, 3000)
        alpha = bytes(rng.sample(range(256), rng.randint(1, 60)))
        data = bytes(rng.choice(alpha) for _ in range(n))
        p = zlib.compressobj(9, zlib.DEFLATED, -15)
        deflate.append([data.hex(), (p.compress(data) + p.flush()).hex()])
    utf8 = []
    for b in ["ff", "c3", "e282", "e228a1", "e282 28".replace(" ", ""), "f09f98", "eda080", "f490", "e080", "c0af",
              "41c3a9ff", "4142e2"]:
        try:
            bytes.fromhex(b).decode("utf-8")
            utf8.append([b, None])
        except UnicodeDecodeError as e:
            utf8.append([b, str(e)])
    write("pyjson", {"loads": loads, "floats": float_cases, "float_strs": float_strs, "int_strs": int_strs,
                     "splits": splits, "reprs": reprs, "sums": sums, "deflate": deflate, "utf8": utf8})


def main() -> None:
    print(f"writing {OUT} (from {sys.executable}, zlib {zlib.ZLIB_RUNTIME_VERSION})")
    try:
        gen_specs()
        gen_settings_files()
        gen_state()
        gen_normalize()
        gen_codes()
        gen_uniforms()
        gen_cube()
        gen_library_and_cli()
        gen_fill()
        gen_recommend()
        gen_detect()
        gen_pyjson()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
