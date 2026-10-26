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


class _Mod:
    def __init__(self, f):
        self.__file__ = f


def test_checkout_layout_requires_the_tests_tree(monkeypatch, tmp_path):
    """From a checkout, a project module outside the tests' tree is foreign, whichever top-level package it is."""
    monkeypatch.setattr(G, "installed_root", lambda: None)
    other = str(tmp_path.resolve() / "other_checkout")
    inside = {"hcd_analysis.emulator.data": _Mod(G.TESTS_ROOT + "/hcd_analysis/emulator/data.py"),
              "tests.test_x": _Mod(G.TESTS_ROOT + "/tests/test_x.py")}
    assert G.foreign_project_modules(inside) == []
    assert len(G.foreign_project_modules({"hcd_analysis.emulator.data": _Mod(other + "/hcd_analysis/emulator/data.py"),
                                          "scripts.run": _Mod(other + "/scripts/run.py")})) == 2


def test_installed_layout_splits_package_and_repository_modules(monkeypatch, tmp_path):
    """Against an installed wheel, hcd_analysis and cli must come from site-packages and tests from the tests' tree;
    the same files the other way round are foreign."""
    site = str(tmp_path.resolve() / "site-packages")
    monkeypatch.setattr(G, "installed_root", lambda: site)
    assert G.foreign_project_modules({"hcd_analysis.emulator.data": _Mod(site + "/hcd_analysis/emulator/data.py"),
                                      "cli.run": _Mod(site + "/cli/run.py"),
                                      "tests.test_x": _Mod(G.TESTS_ROOT + "/tests/test_x.py")}) == []
    assert len(G.foreign_project_modules({"hcd_analysis.emulator.data": _Mod(G.TESTS_ROOT + "/hcd_analysis/x.py"),
                                          "tests.test_x": _Mod(site + "/tests/test_x.py")})) == 2


def test_symlinked_path_to_the_tree_is_not_foreign(monkeypatch, tmp_path):
    """A module imported through a symlink (a venv's lib64 -> lib, a linked checkout) is compared after resolving."""
    monkeypatch.setattr(G, "installed_root", lambda: None)
    link = tmp_path / "link"
    link.symlink_to(G.TESTS_ROOT, target_is_directory=True)
    assert G.foreign_project_modules({"tests.gate_helpers": _Mod(str(link / "tests" / "gate_helpers.py"))}) == []


def test_installed_root_matches_where_hcd_analysis_was_imported():
    """Run from a checkout, hcd_analysis is the tests' sibling and there is no installed root; run against the wheel
    (CI's wheel tier), the installed root is the directory hcd_analysis was imported from."""
    import hcd_analysis
    pkg_parent = os.path.dirname(os.path.dirname(os.path.realpath(hcd_analysis.__file__)))
    if pkg_parent == G.TESTS_ROOT:
        assert G.installed_root() is None
    else:
        assert G.installed_root() == pkg_parent
