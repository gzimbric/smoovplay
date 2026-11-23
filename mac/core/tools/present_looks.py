"""Reference data for sp-present's looks: framegen/looks.py's NumPy render() of the built-in looks, each node type
alone at random settings, and random looks, on tests/test_looks.py's test picture (the same cases and seed as its
test_the_shaders_draw_what_the_reference_does). crates/sp-present/tests/shaders.rs draws them with Metal and holds them
to the same tolerances.

    .venv/bin/python mac/core/tools/present_looks.py   (from the repository root)

Writes mac/core/fixtures/present/looks/: input.bgr (H x W x 3 uint8), luts/Test.cube, cases.json (name, look, frame,
file) and one .bgr per case.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from framegen import looks  # noqa: E402
from framegen.looks import BUILTIN, TYPES_LIST, Library, node, normalize, render  # noqa: E402
from test_looks import H, W, _cube, _img, _random_look  # noqa: E402

OUT = ROOT / "mac" / "core" / "fixtures" / "present" / "looks"


def main() -> None:
    (OUT / "luts").mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.bgr"):
        old.unlink()
    _cube(OUT / "luts" / "Test.cube", 17, lambda r, g, b: (np.sqrt(r), g ** 1.5, 0.5 * b + 0.25 * r))
    lib = Library(OUT)
    img = _img()
    (OUT / "input.bgr").write_bytes(np.ascontiguousarray(img).tobytes())
    cases = dict(BUILTIN)
    rng = np.random.default_rng(7)
    for t in TYPES_LIST:  # each node alone, at random settings
        n = node(t.key, mix=0.8)
        for f in t.params:
            n[f.key] = "Test.cube" if f.text else float(rng.uniform(f.lo, f.hi))
        cases["only " + t.key] = normalize({"nodes": [n]})
    for i in range(30):
        cases[f"random {i}"] = _random_look(rng, lut="Test.cube")
    out = []
    for i, (key, look) in enumerate(cases.items()):
        ref = render(look, img, i, lib)
        name = f"{i:03d}.bgr"
        (OUT / name).write_bytes(np.ascontiguousarray(ref).tobytes())
        out.append({"name": key, "look": look, "frame": i, "file": name,
                    "passes": len(looks.stages(look, lambda n: lib.lut(n) is not None))})
    (OUT / "cases.json").write_text(json.dumps({"width": W, "height": H, "cases": out}, indent=1), encoding="utf-8")
    print(f"{len(out)} looks -> {OUT}")


if __name__ == "__main__":
    main()
