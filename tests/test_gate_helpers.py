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


# external_path (scrub delta review m4): an unset or empty variable gives a path that never exists and names it
@pytest.mark.parametrize("value", [None, ""])
def test_external_path_unset_or_empty_names_the_variable_and_never_exists(monkeypatch, value):
    var = "HCD_TEST_EXTERNAL_ROOT"
    if value is None:
        monkeypatch.delenv(var, raising=False)
    else:
        monkeypatch.setenv(var, value)
    for parts in [(), ("tmp",), ("usr", "bin")]:          # parts that exist under "/" on any POSIX host
        p = G.external_path(var, *parts)
        assert f"<{var} unset>" in p and not os.path.isabs(p) and not os.path.exists(p)
        assert p == os.path.join(f"<{var} unset>", *parts)


def test_external_path_set_joins_onto_the_variable(monkeypatch, tmp_path):
    var = "HCD_TEST_EXTERNAL_ROOT"
    monkeypatch.setenv(var, str(tmp_path))
    assert G.external_path(var) == str(tmp_path)
    assert G.external_path(var, "a", "b.h5") == os.path.join(str(tmp_path), "a", "b.h5")
