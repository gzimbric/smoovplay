"""framegen/tapsys.py on a system with no frame tap (FreeBSD, say): it still imports, with a clock, so framegen/chiaki_tap.py
and everything that imports it (the repeat detector, the stream check) work there; only opening the tap fails."""
import importlib.util
import sys
from pathlib import Path

import pytest

TAPSYS = Path(__file__).resolve().parent.parent / "framegen" / "tapsys.py"


def test_the_tap_module_imports_where_there_is_no_tap(monkeypatch):
    monkeypatch.setattr(sys, "platform", "freebsd14")
    spec = importlib.util.spec_from_file_location("tapsys_elsewhere", TAPSYS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.SUPPORTED
    assert mod.qpc_freq() == 1_000_000_000
    first = mod.qpc()
    assert mod.qpc() >= first > 0
    with pytest.raises(RuntimeError, match="Windows, Linux and macOS only"):
        mod.open_map("Local\\chiaki-frametap-test", 4096)
