"""Gate A review BT8: real-cache tests fail rather than skip in a gate run; paths come from the checkout root."""
import os

import pytest

from hcd_analysis.paths import REPO_ROOT
from tests import gate_helpers as G


def test_real_cache_path_is_inside_this_checkout():
    p = G.real_cache_path("lf")
    assert os.path.isabs(p) and p.startswith(str(REPO_ROOT) + os.sep)


def test_missing_cache_fails_in_a_gate_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HCD_GATE_RUN", "1")
    with pytest.raises(pytest.fail.Exception):
        G.require_real_cache(str(tmp_path / "absent.h5"))


def test_missing_cache_skips_outside_a_gate_run(monkeypatch, tmp_path):
    monkeypatch.delenv("HCD_GATE_RUN", raising=False)
    with pytest.raises(pytest.skip.Exception):
        G.require_real_cache(str(tmp_path / "absent.h5"))
