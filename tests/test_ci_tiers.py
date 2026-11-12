"""tests/ci_tiers.toml classifies every test file exactly once, so no file can drop out of CI unnoticed.

CI (.github/workflows/tests.yml, tools/ci/run_test_tier.py) runs the "wheel" and "source" tiers and refuses any skip
there except of a test named in its file's asset_tests or skipped_by_design, which is reported as not run, and any
expected failure except a deferred_legacy deferral or one whose exact message is in the file's deferred_xfails; every
other tier is reported as not run too, with the reason recorded here."""

import ast
import glob
import os
import tomllib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TIERS = {"wheel", "source", "assets", "deferred", "external", "no_tests"}
CI_TIERS = {"wheel", "source"}


def _doc():
    with open(os.path.join(TESTS_DIR, "ci_tiers.toml"), "rb") as f:
        return tomllib.load(f)


def _test_files():
    root = os.path.dirname(TESTS_DIR)
    pats = ("test_*.py", "regression/test_*.py", "integration/test_*.py", "external/test_*.py")
    found = (p for pat in pats for p in glob.glob(os.path.join(TESTS_DIR, pat)))
    return {os.path.relpath(p, root).replace(os.sep, "/") for p in found}


def test_tier_names_are_the_known_ones():
    assert set(_doc()["tiers"]) == TIERS


def test_every_test_file_is_classified_exactly_once():
    listed = set(_doc()["files"])
    present = _test_files()
    assert not present - listed, f"unclassified test files: {sorted(present - listed)}"
    assert not listed - present, f"classified files that do not exist: {sorted(listed - present)}"


def test_each_entry_has_a_known_tier_and_out_of_ci_entries_say_why():
    for path, entry in _doc()["files"].items():
        assert set(entry) <= {
            "tier",
            "reason",
            "asset_tests",
            "skipped_by_design",
            "deferred_xfails",
        }, path
        assert entry["tier"] in TIERS, path
        if entry["tier"] not in CI_TIERS or set(entry) & {"asset_tests", "skipped_by_design"}:
            assert entry.get(
                "reason", ""
            ).strip(), f"{path}: a file or test outside CI needs its reason"


def test_declared_skips_are_test_functions_of_their_file():
    """asset_tests and skipped_by_design (allowed only in a CI tier) name test functions of the file, or single
    parameter cases of them ("test_x[a]"), each once, so a rename leaves no stale entry."""
    root = os.path.dirname(TESTS_DIR)
    for path, entry in _doc()["files"].items():
        names = entry.get("asset_tests", []) + entry.get("skipped_by_design", [])
        if not names:
            continue
        assert entry["tier"] in CI_TIERS, path
        assert len(set(names)) == len(names), path
        with open(os.path.join(root, path), encoding="utf-8") as f:
            defined = {
                n.name for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.FunctionDef)
            }
        bases = {n.split("[")[0] for n in names}
        assert bases <= defined, f"{path}: not test functions here: {sorted(bases - defined)}"


def test_declared_deferrals_are_exact_messages_in_a_ci_tier():
    """deferred_xfails lists full expected-failure messages (exact matches, never prefixes) of a CI-tier file."""
    for path, entry in _doc()["files"].items():
        if "deferred_xfails" in entry:
            assert entry["tier"] in CI_TIERS and entry.get("reason", "").strip(), path
            msgs = entry["deferred_xfails"]
            assert msgs and all(isinstance(m, str) and len(m) > 40 for m in msgs), path


def test_external_tier_is_exactly_tests_external():
    for path, entry in _doc()["files"].items():
        assert (entry["tier"] == "external") == path.startswith("tests/external/"), path
