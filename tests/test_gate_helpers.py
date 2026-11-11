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
    for parts in [(), ("tmp",), ("usr", "bin")]:  # parts that exist under "/" on any POSIX host
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
    inside = {
        "hcd_analysis.emulator.data": _Mod(G.TESTS_ROOT + "/hcd_analysis/emulator/data.py"),
        "tests.test_x": _Mod(G.TESTS_ROOT + "/tests/test_x.py"),
    }
    assert G.foreign_project_modules(inside) == []
    assert (
        len(
            G.foreign_project_modules(
                {
                    "hcd_analysis.emulator.data": _Mod(other + "/hcd_analysis/emulator/data.py"),
                    "scripts.run": _Mod(other + "/scripts/run.py"),
                }
            )
        )
        == 2
    )


def test_installed_layout_splits_package_and_repository_modules(monkeypatch, tmp_path):
    """Against an installed wheel, hcd_analysis (its cli included) must come from site-packages and tests from the
    tests' tree; the same files the other way round are foreign."""
    site = str(tmp_path.resolve() / "site-packages")
    monkeypatch.setattr(G, "installed_root", lambda: site)
    assert (
        G.foreign_project_modules(
            {
                "hcd_analysis.emulator.data": _Mod(site + "/hcd_analysis/emulator/data.py"),
                "hcd_analysis.cli.run": _Mod(site + "/hcd_analysis/cli/run.py"),
                "tests.test_x": _Mod(G.TESTS_ROOT + "/tests/test_x.py"),
            }
        )
        == []
    )
    assert (
        len(
            G.foreign_project_modules(
                {
                    "hcd_analysis.emulator.data": _Mod(G.TESTS_ROOT + "/hcd_analysis/x.py"),
                    "tests.test_x": _Mod(site + "/tests/test_x.py"),
                }
            )
        )
        == 2
    )


def test_symlinked_path_to_the_tree_is_not_foreign(monkeypatch, tmp_path):
    """A module imported through a symlink (a venv's lib64 -> lib, a linked checkout) is compared after resolving."""
    monkeypatch.setattr(G, "installed_root", lambda: None)
    link = tmp_path / "link"
    link.symlink_to(G.TESTS_ROOT, target_is_directory=True)
    assert (
        G.foreign_project_modules(
            {"tests.gate_helpers": _Mod(str(link / "tests" / "gate_helpers.py"))}
        )
        == []
    )


def test_installed_root_matches_where_hcd_analysis_was_imported():
    """Run from a checkout, hcd_analysis is the tests' sibling and there is no installed root; run against the wheel
    (CI's wheel tier), the installed root is the directory hcd_analysis was imported from."""
    import hcd_analysis

    pkg_parent = os.path.dirname(os.path.dirname(os.path.realpath(hcd_analysis.__file__)))
    if pkg_parent == G.TESTS_ROOT:
        assert G.installed_root() is None
    else:
        assert G.installed_root() == pkg_parent


class _Dist:
    def __init__(self, location, record):
        self._location, self._record = location, record

    def read_text(self, name):
        return self._record if name == "RECORD" else None

    def locate_file(self, path):
        return os.path.join(self._location, path)


@pytest.mark.parametrize(
    "record, at_root, installed",
    [
        (
            "hcd_analysis/paths.py,sha256=x,1\nhcd_analysis/__init__.py,sha256=y,2\n",
            True,
            True,
        ),  # wheel install
        (None, True, False),  # a checkout with a legacy egg-info: no RECORD (PU-0241's tree)
        ("__editable__.hcd_priya-0.2.0.pth,sha256=x,1\n", True, False),  # editable install
        (
            "hcd_analysis/paths.py,sha256=x,1\n",
            False,
            False,
        ),  # installed elsewhere, imported from here
    ],
)
def test_installed_root_requires_a_wheel_record_where_the_package_was_imported(
    monkeypatch, tmp_path, record, at_root, installed
):
    root = tmp_path.resolve() / "site"
    root.mkdir()
    location = root if at_root else tmp_path.resolve() / "elsewhere"
    monkeypatch.setattr(G, "REPO_ROOT", root)
    monkeypatch.setattr(
        G, "TESTS_ROOT", str(tmp_path.resolve() / "testsdir")
    )  # tests outside any checkout
    monkeypatch.setattr(G.md, "distribution", lambda name: _Dist(str(location), record))
    assert G.installed_root() == (str(root) if installed else None)


def test_tests_inside_a_checkout_are_never_in_installed_mode(monkeypatch, tmp_path):
    """A stale `pip install .` in the venv must not excuse hcd_analysis resolving there while the tests come from a
    checkout: the tests' tree holds the package, so the checkout rule applies and the mix is flagged.
    """
    site = tmp_path.resolve() / "site"
    checkout = tmp_path.resolve() / "checkout"
    (checkout / "hcd_analysis").mkdir(parents=True)
    (checkout / "hcd_analysis" / "__init__.py").write_text("")
    monkeypatch.setattr(G, "REPO_ROOT", site)
    monkeypatch.setattr(G, "TESTS_ROOT", str(checkout))
    monkeypatch.setattr(
        G.md, "distribution", lambda name: _Dist(str(site), "hcd_analysis/paths.py,sha256=x,1\n")
    )
    assert G.installed_root() is None
    assert G.foreign_project_modules(
        {"hcd_analysis.paths": _Mod(str(site / "hcd_analysis" / "paths.py"))}
    )
