"""Tests of the CI machinery itself: the scripts in tools/ci/ and the GitHub Actions workflows in .github/workflows/.

What is covered:
1. tools/ci/check_dist.py, which checks that a built wheel or sdist holds exactly the committed package;
2. tools/ci/run_test_tier.py, the runner CI uses for the tests: its outcome counts and verdicts, and which files and
   tests it runs at each level (tests/ci_tiers.toml defines tiers and levels in its header). Note: in CI a skip is
   not a pass; a skip fails its file unless the test is declared there, and a declared test that runs fails it too;
3. tools/ci/nightly_gate.py, which refuses a release unless the latest nightly run on its commit passed;
4. the pinned development-environment versions against pyproject.toml's lower bounds;
5. the workflow files themselves. Many tests here read the text or the parsed YAML of a workflow and check a rule of
   it, such as "no step may hide a failure" or "the nightly job runs the same steps as the pull-request jobs". They
   test configuration, not Python behaviour, so that nobody can weaken CI by accident.

Most inputs are small synthetic archives, junit reports and run lists; the workflow, tier and level tests read the
real files of this checkout.

Terms: see the header of tests/ci_tiers.toml.

Run: pytest tests/test_ci_tools.py (CI runs it at the pr level, in the source tier; the workflow tests need PyYAML).
"""

import importlib.util
import io
import json
import os
import re
import tarfile
import tomllib
import zipfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """A script of tools/ci, loaded from its file path as a module (tools/ci is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        f"_ci_{name}", os.path.join(REPO, "tools", "ci", f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The three CI scripts under test, by short names: CD the distribution check, RT the test-tier runner, NG the
# nightly gate. (XV, the HDF5 cross-version check, is loaded in the one test that uses it.)
CD = _load("check_dist")
RT = _load("run_test_tier")
NG = _load("nightly_gate")
# A fake "committed package" for the archive tests: {path: git blob id}. check_dist compares each file of an archive
# with the git blob id (the hash git gives a file's bytes) of the committed file. Every fake file holds the bytes
# "x", so every id is X. PACKAGE is what a wheel must hold; TOP the extra top-level files an sdist must hold;
# DIST_INFO the metadata files of the wheel's .dist-info folder.
X = CD.git_blob_id(b"x")
PACKAGE = {
    n: X
    for n in (
        "hcd_analysis/__init__.py",
        "hcd_analysis/emulator/__init__.py",
        "hcd_analysis/cli/__init__.py",
        "hcd_analysis/cli/run.py",
        *CD.PACKAGE_DATA,
    )
}
TOP = {n: X for n in CD.SDIST_COMMITTED}
DIST_INFO = ["METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"]


def _wheel(tmp_path, extra=(), drop=(), top="hcd_analysis\n", content=None):
    """Write a small fake wheel of PACKAGE and return its path. extra: names added beyond PACKAGE; drop: names left
    out; top: the text of top_level.txt (the top-level packages the wheel declares); content: {name: text} for files
    whose bytes are not "x" (an uncommitted edit)."""
    path = tmp_path / "hcd_priya-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted((set(PACKAGE) | set(extra)) - set(drop)):
            zf.writestr(name, (content or {}).get(name, "x"))
        for name in DIST_INFO:
            zf.writestr(f"hcd_priya-0.0.0.dist-info/{name}", "x")
        zf.writestr("hcd_priya-0.0.0.dist-info/top_level.txt", top)
    return str(path)


def _sdist(tmp_path, extra=(), drop=(), link=None, content=None):
    """Write a small fake sdist of PACKAGE and TOP (plus the files a build adds) and return its path. extra, drop and
    content as in _wheel; link: a name to add as a symbolic link pointing outside the archive."""
    path = tmp_path / "hcd_priya-0.0.0.tar.gz"
    names = (
        set(PACKAGE)
        | set(TOP)
        | {"PKG-INFO", "setup.cfg", "hcd_priya.egg-info/SOURCES.txt"}
        | set(extra)
    )
    with tarfile.open(path, "w:gz") as tf:
        for name in sorted(names - set(drop)):
            data = (content or {}).get(name, "x").encode()
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{link}")
            info.type = tarfile.SYMTYPE
            # a link to a file outside the archive: check_sdist must refuse any link
            info.linkname = "/etc/passwd"
            tf.addfile(info)
    return str(path)


def test_git_blob_id_is_the_id_git_gives():
    """check_dist hashes a file as git does: the id of the empty file is the one `git hash-object /dev/null` prints."""
    assert CD.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_check_wheel_accepts_exactly_the_committed_package(tmp_path):
    """A wheel holding exactly the committed package files (and its metadata) has no problem."""
    assert CD.check_wheel(_wheel(tmp_path), PACKAGE) == []


@pytest.mark.parametrize(
    "extra",
    [
        "hcd_analysis/_emulator_data/observables_tau0_lf.h5",
        "hcd_analysis/emulator/member.norm.pkl",
        "tests/test_x.py",
        "hcd_analysis/scratch.py",
        "hcd_analysis/emulator/README.md",
        "hcd_priya-0.0.0.dist-info/extra.txt",
        "other-0.0.0.dist-info/METADATA",
        "hcd_analysis-0.0.0.dist-info/METADATA",
    ],
)
def test_check_wheel_refuses_any_other_file(tmp_path, extra):
    """Data, pickles, tests, an uncommitted module, docs, extra metadata, or the metadata of another distribution
    (also the old name hcd_analysis) never pass."""
    problems = CD.check_wheel(_wheel(tmp_path, extra=[extra]), PACKAGE)
    assert any(
        extra in p for p in problems
    ), f"the check did not name {extra}; it reported {problems}"


@pytest.mark.parametrize(
    "drop", ["hcd_analysis/cli/run.py", "hcd_analysis/comparison/phase0_v1.json"]
)
def test_check_wheel_refuses_a_missing_file(tmp_path, drop):
    """A committed module or package data file missing from the wheel is reported."""
    problems = CD.check_wheel(_wheel(tmp_path, drop=[drop]), PACKAGE)
    assert any(
        f"missing {drop}" in p for p in problems
    ), f"no 'missing {drop}'; it reported {problems}"


def test_check_wheel_refuses_a_file_that_differs_from_head(tmp_path):
    """An uncommitted edit (or a stale build/ directory) ships different bytes under the right name."""
    problems = CD.check_wheel(
        _wheel(tmp_path, content={"hcd_analysis/cli/run.py": "edited"}), PACKAGE
    )
    assert problems == ["wheel: hcd_analysis/cli/run.py differs from HEAD"]


def test_check_wheel_refuses_another_top_level_package(tmp_path):
    """A generic top-level name such as cli, which another distribution could also install, never ships."""
    for top in ("cli\nhcd_analysis\n", "hcd_analysis\ntests\n"):
        problems = CD.check_wheel(_wheel(tmp_path, top=top), PACKAGE)
        assert any(
            "top_level" in p for p in problems
        ), f"top_level {top!r} passed; it reported {problems}"


def test_check_sdist_accepts_exactly_the_committed_package(tmp_path):
    """An sdist holding exactly the committed package, the committed top-level files and what a build adds has no
    problem."""
    assert CD.check_sdist(_sdist(tmp_path), PACKAGE, TOP) == []


@pytest.mark.parametrize(
    "extra",
    [
        "tests/test_x.py",
        "checkpoints/manifest.json",
        "scripts/run.py",
        "hcd_analysis/_emulator_data/observables_tau0_lf.h5",
        "hcd_analysis/scratch.py",
    ],
)
def test_check_sdist_refuses_any_other_file(tmp_path, extra):
    """Tests, checkpoints, scripts, data or an uncommitted module in the sdist are reported."""
    problems = CD.check_sdist(_sdist(tmp_path, extra=[extra]), PACKAGE, TOP)
    assert any(
        extra in p for p in problems
    ), f"the check did not name {extra}; it reported {problems}"


def test_check_sdist_refuses_links_missing_and_edited_files(tmp_path):
    """In one sdist, a symbolic link, a missing committed file and an edited committed file are each reported."""
    problems = CD.check_sdist(
        _sdist(
            tmp_path,
            drop=["pyproject.toml"],
            link="hcd_analysis/x.py",
            content={"MANIFEST.in": "graft tests"},
        ),
        PACKAGE,
        TOP,
    )
    assert any("link" in p for p in problems), f"the link was not reported; it reported {problems}"
    assert any(
        "missing pyproject.toml" in p for p in problems
    ), f"the missing pyproject.toml was not reported; it reported {problems}"
    assert (
        "sdist: MANIFEST.in differs from HEAD" in problems
    ), f"the edited MANIFEST.in was not reported; it reported {problems}"


def test_committed_files_cover_the_packages_and_nothing_else():
    """The real list check_dist builds from git: the package's modules and package data, and the sdist's committed
    top-level files; no other kind of file."""
    package, top = CD.committed_files()
    assert {"hcd_analysis/__init__.py", "hcd_analysis/paths.py", "hcd_analysis/cli/run.py"} <= set(
        package
    )
    assert set(CD.PACKAGE_DATA) <= set(package) and set(top) == set(CD.SDIST_COMMITTED)
    assert all(
        p.split("/")[0] in CD.PACKAGES and (p.endswith(".py") or p in CD.PACKAGE_DATA)
        for p in package
    )


# A fake pytest junit report (the XML file pytest writes with --junitxml) with one test case of each kind:
#   a      passes;
#   b      fails;
#   c      errors;
#   d[p1]  skips (a plain skip; "[p1]" is a parameter case of the test d);
#   e      an expected failure whose message is the declared deferral DEFERRAL below;
#   f      an expected failure with another message (not a declared deferral).
# pytest writes an expected failure as a <skipped> element of type pytest.xfail.
_JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="t" name="a"/>
<testcase classname="t" name="b"><failure message="x"/></testcase>
<testcase classname="t" name="c"><error message="x"/></testcase>
<testcase classname="t" name="d[p1]"><skipped type="pytest.skip" message="absent"/></testcase>
<testcase classname="t" name="e"><skipped type="pytest.xfail" message="deferred_legacy X: y"/></testcase>
<testcase classname="t" name="f"><skipped type="pytest.xfail" message="some other reason"/></testcase>
</testsuite></testsuites>"""


def _counts(**changes):
    """The outcome counts (RT.COUNT_KEYS) of a clean file of 3 tests that all passed, with `changes` setting single
    counts."""
    out = dict.fromkeys(RT.COUNT_KEYS, 0)
    out.update(tests=3, passed=3)
    out.update(changes)
    return out


# the one deferral message the tests below declare: the message of row e of _JUNIT
DEFERRAL = {"deferred_legacy X: y"}


def test_junit_counts_separate_skips_deferrals_and_other_expected_failures(tmp_path):
    """Each row of _JUNIT lands in its own count; an expected failure counts as a deferral (xfailed) only when its
    message is exactly a declared one."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), deferrals=DEFERRAL)
    assert c == _counts(
        tests=6, passed=1, failures=1, errors=1, skipped=1, xfailed=1, bad_xfail=1
    ), f"each row of _JUNIT must land in its own count (rows a to f), with e a declared deferral; got {c}"
    # no deferral declared: both expected failures, e and f, are undeclared (bad_xfail)
    c = RT.junit_counts(str(p))
    assert (
        c["bad_xfail"] == 2
    ), f"with no deferral declared, rows e and f must both count as bad_xfail; got {c}"


def test_junit_counts_declared_skips(tmp_path):
    """A declared skip that skips is NOT RUN, not a skip; a declared test that runs anyway is flagged."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    # each tuple below: (undeclared skips, declared skips that skipped, declared skips that ran)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"])
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 1), (
        "declaring d and a: d's skip must count as a declared skip (asset_skipped), not an undeclared one, and a, "
        f"which passed, as a declared skip that ran (asset_ran); got {c}"
    )
    c = RT.junit_counts(str(p), declared_skips=["d[p1]"])  # one parameter case, named exactly
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0), (
        "declaring the case d[p1] by its full name must cover the skip of d[p1], which then counts as a "
        f"declared skip; got {c}"
    )
    # another case: the skip of d[p1] stays undeclared
    c = RT.junit_counts(str(p), declared_skips=["d[p2]"])
    assert (c["skipped"], c["asset_skipped"]) == (1, 0), (
        "declaring d[p2] must not cover the skip of d[p1], which stays an undeclared skip; "
        f"got {c}"
    )


def test_junit_counts_in_an_assets_job_a_required_test_must_run(tmp_path):
    """In a job meant to have the assets, a declared asset test that skips means its asset is missing; one that runs
    is what that job is for. A skipped_by_design test (declared, not required) may still skip."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    # (required tests not run, declared skips that skipped, declared skips that ran, undeclared skips)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["d", "a"])
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"], c["skipped"]) == (1, 0, 0, 0), (
        "requiring d and a: d's skip must count as a required test not run (asset_missing), and a, which ran, as "
        f"nothing wrong; got {c}"
    )
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["a"])  # d skipped by design
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0), (
        "d declared but not required (skipped by design) may skip: it must count as asset_skipped; "
        f"got {c}"
    )


def test_junit_counts_in_an_assets_job_a_required_test_absent_from_the_report_is_missing(tmp_path):
    """A required asset test that never reaches the report (not collected: a renamed test or parameter; or
    deselected) is missing too; a required name matched by its full id or by its function name is not.
    """
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    # zz is not in the report: (required tests not run, passed) = (1, 1)
    c = RT.junit_counts(str(p), declared_skips=["a", "zz"], required=["a", "zz"])
    assert (c["asset_missing"], c["passed"]) == (1, 1), (
        "the required name zz, absent from the report, must count as asset_missing; a still passes; "
        f"got {c}"
    )
    c = RT.junit_counts(
        str(p), declared_skips=["d[p1]", "d[p9]", "a"], required=["d[p1]", "d[p9]", "a"]
    )
    assert (
        c["asset_missing"] == 2
    ), f"d[p1] (skipped) and d[p9] (absent) must both count as asset_missing; got {c}"
    c = RT.junit_counts(str(p), declared_skips=["a"], required=["a"])
    assert c["asset_missing"] == 0, f"a required test that ran (a) is not missing; got {c}"
    # zz declared but not required: its absence is not counted
    c = RT.junit_counts(str(p), declared_skips=["zz"])
    assert (
        c["asset_missing"] == 0
    ), f"zz is declared but not required: its absence is not counted; got {c}"


def test_the_assets_job_runs_the_assets_tier_and_every_file_with_asset_or_release_tests():
    """The files the assets job picks: all of the assets tier, plus the wheel and source files that declare asset or
    release tests, and never a deferred, external or no-tests file."""
    _, files = RT.load_tiers()
    picked = RT.asset_job_files(files)
    assert set(RT.files_of(files, "assets")) <= set(picked)
    for path in picked:
        assert (
            files[path]["tier"] == "assets"
            or files[path].get("asset_tests")
            or files[path].get("release_tests")
        ), f"{path}: the assets job picked it, but it is not in the assets tier and declares no asset or release tests"
    never = {p for p in picked if files[p]["tier"] in ("deferred", "external", "no_tests")}
    assert not never, f"the assets job picked files of tiers it must not run: {sorted(never)}"
    # once the assets job has run, the list of files not run names neither the assets tier nor a CI tier
    listed = set(RT.not_run(files, ran=("assets",))) & {"assets", *RT.CI_TIERS}
    assert not listed, f"the not-run list names tiers that ran: {sorted(listed)}"


# The tests below read RT.tier_plan(tier, files, level); its docstring in tools/ci/run_test_tier.py describes the plan.


def test_the_plan_of_a_ci_tier_runs_its_files_without_required_tests_or_added_environment():
    """A CI tier (wheel or source) at the default level, nightly: every file of the tier, no added environment
    variable, slow tests deselected, the declared skips of each file, and no test required to run (only the assets
    job requires tests)."""
    _, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        plan = RT.tier_plan(tier, files)
        assert plan["todo"] == sorted(p for p, e in files.items() if e["tier"] == tier)
        assert plan["env"] == {} and plan["marker"] == "not slow"  # slow tests listed as NOT RUN
        assert set(plan["declared"]) == set(plan["required"]) == set(plan["todo"])
        for rel in plan["todo"]:
            assert plan["declared"][rel] == files[rel].get("asset_tests", []) + files[rel].get(
                "skipped_by_design", []
            ), f"{rel}: the declared skips must be the file's asset_tests and skipped_by_design"
            assert plan["required"][rel] == [], f"{rel}: a CI tier requires no test to run"


def test_the_plan_of_the_assets_job_runs_every_qualifying_file_with_the_gate_on_and_requires_its_asset_and_release_tests():
    """Every file of the assets tier and every CI-tier file with asset_tests or release_tests, and nothing else;
    HCD_GATE_RUN=1 (a missing real cache fails instead of skipping); each file's asset_tests and release_tests are
    required to run, skipped_by_design is not.
    """
    _, files = RT.load_tiers()
    plan = RT.tier_plan("assets", files)
    want = sorted(
        p
        for p, e in files.items()
        if e["tier"] == "assets"
        or (e["tier"] in ("wheel", "source") and (e.get("asset_tests") or e.get("release_tests")))
    )
    # the assets job runs more than the assets tier: also the CI-tier files that declare asset or release tests
    assert plan["todo"] == want and len(want) > len(RT.files_of(files, "assets"))
    assert plan["env"] == {"HCD_GATE_RUN": "1"}
    assert plan["marker"] is None  # slow tests run too: an asset test may be marked slow
    for rel in want:
        assert plan["required"][rel] == files[rel].get("asset_tests", []) + files[rel].get(
            "release_tests", []
        ), f"{rel}: the assets job must require exactly the file's asset_tests and release_tests"
        # every test of the file: nothing selected by name, nothing left out
        assert plan["select"][rel] == [rel] and plan["deselect"][rel] == [], (
            f"{rel}: the assets job must run every test of the file, but it selects {plan['select'][rel]} "
            f"and leaves out {plan['deselect'][rel]}"
        )
        both = set(plan["required"][rel]) & set(files[rel].get("skipped_by_design", []))
        assert not both, f"{rel}: tests both required and skipped by design: {sorted(both)}"


def test_deferred_legacy_messages_are_exactly_what_conftest_writes():
    """The runner reads the deferral messages from the source of tests/gate_helpers.py without importing the tests;
    they must equal the messages the imported DEFERRED_CAUSES gives (the text tests/conftest.py writes).
    """
    from tests.gate_helpers import DEFERRED_CAUSES

    assert RT.deferred_legacy_messages() == {
        f"deferred_legacy {cause}: {reason}" for cause, (_, reason) in DEFERRED_CAUSES.items()
    }


# Each row: (pytest's exit status, the file's outcome counts, whether the file passes).
@pytest.mark.parametrize(
    "rc, counts, ok",
    [
        (0, _counts(), True),
        (0, _counts(tests=5, xfailed=2), True),  # declared deferrals beside passes
        (0, _counts(tests=4, asset_skipped=1), True),  # a declared skip beside passes
        (0, _counts(skipped=1), False),  # a skip is never a pass
        (0, _counts(bad_xfail=1), False),  # only declared deferrals may be expected failures
        (0, _counts(asset_ran=1), False),  # a declared skip must skip without its asset
        (0, _counts(asset_missing=1), False),  # an assets job: a missing asset fails, never skips
        (0, _counts(tests=0, passed=0), False),  # nothing ran
        (0, _counts(tests=2, passed=0, xfailed=2), False),  # deferrals only: nothing passed
        (0, _counts(tests=2, passed=0, asset_skipped=2), False),  # declared skips only
        # pytest exited 1 although no test failed: for example tests/conftest.py's check at the end of every session,
        # which fails the session when a project module was imported from another checkout
        (1, _counts(), False),
        (1, _counts(failures=1), False),
        (None, None, False),  # timeout
        # no readable junit report (any non-zero exit status; 4 is only an example)
        (4, None, False),
    ],
)
def test_file_verdict(rc, counts, ok):
    """RT.file_verdict(exit status, counts) says whether one test file passes: only if pytest exited 0, at least one
    test passed, and nothing failed, errored, skipped without being declared, ran when it was declared to skip,
    was an undeclared expected failure, or (in the assets job) was required and did not run. An exit status of None
    is a timeout; counts None is a missing report.
    """
    verdict, reason = RT.file_verdict(rc, counts)
    assert verdict is ok, f"exit {rc}, counts {counts}: the verdict was {verdict} ({reason})"


def test_the_runner_reads_the_real_tier_map():
    """The runner's view of tests/ci_tiers.toml: the files of each CI tier, and a not-run list without CI tiers."""
    tiers, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        assert RT.files_of(files, tier) == sorted(p for p, e in files.items() if e["tier"] == tier)
    assert not set(RT.not_run(files)) & set(RT.CI_TIERS)


def _lower_bounds():
    """{package name: its lower bound} of the dependencies and test dependencies in pyproject.toml, for example
    {"numpy": "2.0"} from "numpy>=2.0"."""
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        proj = tomllib.load(f)["project"]
    reqs = proj["dependencies"] + proj["optional-dependencies"]["test"]
    out = {}
    for r in reqs:
        # "numpy>=2.0" -> ("numpy", ">=", "2.0"); a requirement without ">=" gets an empty bound
        name, _, bound = r.partition(">=")
        out[name.strip().lower()] = bound.strip()
    return out


# Direct dependencies whose lower bound is deliberately below the pinned release series, with the reason.
BOUND_BELOW_PIN = {
    "numpy": "2.0: jax 0.10.1 requires numpy 2.0 or later and no newer numpy interface is used"
}


def test_devenv_pins_sit_in_the_release_series_of_the_lower_bounds():
    """tools/ci/constraints-devenv.txt pins the development environment; each pinned direct dependency's lower
    bound in pyproject.toml is the pin's release series (CI's development-environment job keeps the bounds tested),
    except the bounds named in BOUND_BELOW_PIN, which must still lie at or below the pin. A release series is the
    leading part of a version: a bound of "0.10" and a pin of "0.10.1" agree; a bound of "0.9" does not.
    """
    bounds = _lower_bounds()
    pins = {}
    for line in open(os.path.join(REPO, "tools", "ci", "constraints-devenv.txt")):
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[name.lower()] = version
    direct = set(pins) & set(bounds)
    # the direct dependencies that must be both pinned and bounded
    must_be_pinned = {
        "jax",
        "equinox",
        "optax",
        "numpyro",
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "pytest",
    }
    assert direct >= must_be_pinned, (
        f"{sorted(must_be_pinned - direct)}: missing from tools/ci/constraints-devenv.txt or without a lower "
        "bound in pyproject.toml; add the pin or the bound"
    )
    for name in direct:
        # the versions as lists of their parts: "0.10.1" -> ["0", "10", "1"]
        bound_parts, pin_parts = bounds[name].split("."), pins[name].split(".")
        if name in BOUND_BELOW_PIN:
            assert BOUND_BELOW_PIN[name].startswith(bounds[name] + ":"), (
                f"{name}: BOUND_BELOW_PIN says {BOUND_BELOW_PIN[name]!r}, which does not start with the lower bound "
                f"in pyproject.toml, {bounds[name]!r}; keep the two in step"
            )
            assert tuple(map(int, bound_parts)) <= tuple(
                map(int, pin_parts[: len(bound_parts)])
            ), f"{name}: bound {bounds[name]} above the pin"
            continue
        assert (
            pin_parts[: len(bound_parts)] == bound_parts
        ), f"{name}: bound {bounds[name]} is not the series of the pin {pins[name]}"


def test_the_workflow_calls_the_ci_helpers_that_exist():
    """tests.yml names each CI helper it runs, and each of those files exists, so a rename cannot go unnoticed; it
    also runs both CI tiers and the list of what is not run."""
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml")).read()
    for helper in (
        "tools/ci/check_dist.py",
        "tools/ci/run_test_tier.py",
        "tools/ci/constraints-devenv.txt",
        "tools/ci/h5_cross_version.py",
    ):
        assert helper in text and os.path.isfile(
            os.path.join(REPO, helper)
        ), f"{helper}: tests.yml must name it and the file must exist"
    for tier in ("--tier wheel", "--tier source", "--list-not-run"):
        assert tier in text, f"tests.yml must run the runner with {tier}"


# Shell text that hides a failure. The first branch matches "set +e" and "set +o ..." (such as "set +o pipefail"),
# which turn off stopping on an error; the second matches "|| true", "|| :" and "|| exit 0", which make a failing
# command count as a success. (?![\w-]) stops the second branch matching the start of a longer word ("|| true-x").
# Matches: "set +e", "set +o pipefail", "set +eux", "cmd || true", "cmd || exit 0". Does not match: "set -e",
# "cmd || true-x".
UNDO_FAILURE = re.compile(r"set\s+\+[a-zA-Z]*[eo]|\|\|\s*(true|:|exit\s+0)(?![\w-])")


def test_every_run_step_fails_when_a_command_in_a_pipe_fails():
    """The workflow names bash as the shell of every run step, so GitHub runs each with `bash -eo pipefail` and
    `pytest ... | tee log` fails its step when pytest fails. With no shell named, GitHub runs `bash -e`, where a pipe's
    status is that of its last command (tee). No job or step may set another shell or its own defaults, and no
    run step may turn -e or pipefail off (set +e, set +o pipefail) or swallow a failure (|| true, || :, || exit 0).
    """
    for flow in ("tests.yml", "nightly.yml", "release-validation.yml", "publish.yml"):
        wf = _workflow(flow)
        bash_everywhere = wf["defaults"] == {"run": {"shell": "bash"}}
        assert bash_everywhere, f"{flow}: the workflow must start with defaults: run: shell: bash"
        for name, job in wf["jobs"].items():
            assert "defaults" not in job, f"{flow} job {name}: a job may not set its own defaults"
            for step in job["steps"]:
                where = f"{flow} job {name}, step {step.get('name')!r}"
                assert "shell" not in step, f"{where}: a step may not set its own shell"
                # nor may a step undo -e or pipefail, or swallow a failing command
                assert not UNDO_FAILURE.search(step.get("run", "")), (
                    f"{where}: the run command hides a failure "
                    f"({UNDO_FAILURE.search(step.get('run', '')).group(0)!r}); let the step fail instead"
                )


def _workflow(name="tests.yml"):
    """A workflow file of .github/workflows/, parsed from YAML into a dict."""
    import yaml  # PyYAML: needed only by the workflow tests, so imported where it is used

    return yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", name)))


# the pr level runs as two jobs of tests.yml: the wheel tier's job and the source tier's
PR_JOBS = (
    "fixture-tests",
    "fixture-tests-source",
)


def test_only_the_pinned_job_declares_the_pinned_stack():
    """A workflow's "matrix" runs one job once per stack (tests/ci_tiers.toml defines a stack). The pinned stack is the
    only supported one; "pinned", "development-environment" (its deps value) and "supported" (its label) all name it.

    The test checks both workflows that run tests:
    1. HCD_PINNED_STACK is "1" in the development-environment (supported) jobs and empty in the others, and it is set
       nowhere else: in tests.yml the name occurs exactly twice outside comments (the env of the two pr jobs, one per
       tier), so no other env, run command or write to $GITHUB_ENV (the file a step writes to set a variable for
       later steps) can set it.
    2. Each pr job's matrix in tests.yml is exactly its include list (no other axis) and holds the one supported
       stack.
    3. nightly.yml's matrix holds the supported stack and the two newest-dependency stacks, and sets
       HCD_PINNED_STACK to "1" only on the supported one.
    """
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml"), encoding="utf-8").read()
    # drop the YAML comments (a "#" at the start of a line or after a space), so a mention in a comment is not counted
    code = "\n".join(re.sub(r"(^|\s)#.*", "", line) for line in text.splitlines())
    n_named = code.count("HCD_PINNED_STACK")
    assert n_named == len(
        PR_JOBS
    ), "tests.yml: HCD_PINNED_STACK may appear only in the env of the pr jobs"
    wf = _workflow()
    assert "HCD_PINNED_STACK" not in (wf.get("env") or {}), "tests.yml: not in the workflow's env"
    for name, job in wf["jobs"].items():
        if name not in PR_JOBS:
            assert "HCD_PINNED_STACK" not in (
                job.get("env") or {}
            ), f"tests.yml job {name}: only the pr jobs may set HCD_PINNED_STACK"
        for step in job["steps"]:
            assert "HCD_PINNED_STACK" not in (
                step.get("env") or {}
            ), f"tests.yml job {name}, step {step.get('name')!r}: no step may set HCD_PINNED_STACK"
    for name in PR_JOBS:
        job = wf["jobs"][name]
        # GitHub's expression syntax: "1" when the job's stack is development-environment, empty otherwise. The test
        # pins the exact text, so the variable cannot be set to 1 on another stack.
        assert (
            job["env"]["HCD_PINNED_STACK"]
            == "${{ matrix.deps == 'development-environment' && '1' || '' }}"
        ), f"tests.yml job {name}: HCD_PINNED_STACK must be 1 exactly in the development-environment stack"
        axes = set(job["strategy"]["matrix"])
        assert axes == {"include"}, f"tests.yml job {name}: a matrix axis other than include"
        include = job["strategy"]["matrix"]["include"]
        # the PR level runs on the supported stack only; the newer stacks run at the nightly level
        assert [(e["python"], e["deps"], e["label"]) for e in include] == [
            ("3.11", "development-environment", "Python 3.11, pinned (supported)")
        ], f"tests.yml job {name}: the matrix must hold the one pinned stack, Python 3.11"
    nightly = _workflow("nightly.yml")
    stacks = nightly["jobs"]["nightly"]["strategy"]["matrix"]["stack"]
    assert [(e["python"], e["deps"]) for e in stacks] == [
        ("3.11", "development-environment"),
        ("3.11", "newest"),
        ("3.12", "newest"),
    ], "nightly.yml: the stacks must be the pinned one and the two newest-dependency ones"
    assert [e["label"].endswith("(supported)") for e in stacks] == [
        True,
        False,
        False,
    ], "nightly.yml: only the pinned stack's label may say (supported)"
    assert (
        nightly["jobs"]["nightly"]["env"]["HCD_PINNED_STACK"]
        == "${{ matrix.stack.deps == 'development-environment' && '1' || '' }}"
    ), "nightly.yml: HCD_PINNED_STACK must be 1 exactly in the development-environment stack"


# The `with` inputs each action may take: an input such as setup-python's pip-install would install packages
# outside the pinned steps.
ACTION_INPUTS = {
    "actions/checkout@v4": {"persist-credentials"},
    "actions/setup-python@v5": {"python-version"},
    "actions/upload-artifact@v4": {"name", "path", "retention-days"},
}


def test_every_action_takes_only_its_known_inputs():
    """Each action a workflow uses (a `uses:` step) is one of ACTION_INPUTS and takes only the inputs listed there."""
    for flow in ("tests.yml", "nightly.yml", "release-validation.yml"):
        for name, job in _workflow(flow)["jobs"].items():
            for step in job["steps"]:
                if "uses" in step:
                    assert step["uses"] in ACTION_INPUTS, (
                        f"{flow} job {name}: {step['uses']} is not a known action (ACTION_INPUTS); add it "
                        "there, with the inputs it may take, only after review"
                    )
                    given = set(step.get("with") or {})
                    assert given <= ACTION_INPUTS[step["uses"]], (
                        f"{flow} job {name}: {step['uses']} is given the inputs "
                        f"{sorted(given - ACTION_INPUTS[step['uses']])}, which ACTION_INPUTS does not allow"
                    )
                    # no workflow checks out another branch (ref): each tests the commit that started it
                    assert (
                        "ref" not in given
                    ), f"{flow} job {name}: a workflow may not check out another branch"


def test_the_docs_build_is_pinned_and_treats_warnings_as_errors():
    """Read the Docs and the CI docs job install the same fully pinned requirements and fail on any Sphinx warning."""
    import yaml

    rtd = yaml.safe_load(open(os.path.join(REPO, ".readthedocs.yaml")))
    assert rtd["sphinx"] == {
        "configuration": "docs/conf.py",
        "fail_on_warning": True,
    }, ".readthedocs.yaml: Sphinx must use docs/conf.py and fail on any warning (fail_on_warning: true)"
    assert rtd["python"]["install"] == [
        {"requirements": "docs/requirements.txt"}
    ], ".readthedocs.yaml must install exactly docs/requirements.txt, as the CI docs job does"
    job = _workflow()["jobs"]["docs"]
    assert [step.get("uses") for step in job["steps"] if "uses" in step] == [
        "actions/checkout@v4",
        "actions/setup-python@v5",
    ], "tests.yml docs job: only the checkout and setup-python actions"
    runs = [step["run"] for step in job["steps"] if "run" in step]
    # nothing else installed, and -W turns every Sphinx warning into an error
    assert runs == [
        "python -m pip install -r docs/requirements.txt",
        'python -m sphinx -W --keep-going -b html docs "$RUNNER_TEMP/html"',
    ], "tests.yml docs job: install docs/requirements.txt, then build with sphinx -W, and nothing else"
    lines = [ln.strip() for ln in open(os.path.join(REPO, "docs", "requirements.txt"))]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    # every requirement is pinned exactly, "name==version" and nothing else (no range, no extra, no URL)
    pinned = re.compile(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*")
    assert reqs and all(
        pinned.fullmatch(r) for r in reqs
    ), f"docs/requirements.txt: every line must be name==version: {reqs}"
    names = {r.split("==")[0].lower().replace("_", "-") for r in reqs}
    needed = {
        "sphinx",
        "myst-parser",
        "sphinx-autoapi",
        "sphinx-rtd-theme",
        "sphinxcontrib-mermaid",
    }
    assert (
        needed <= names
    ), f"docs/requirements.txt lacks {sorted(needed - names)}, which the docs build uses"


def test_h5_cross_version_check_reads_back_and_catches_a_change_or_a_new_library(tmp_path, capsys):
    """tools/ci/h5_cross_version.py: a synthetic bundle reads back with every digest equal; an edited digest fails, of
    a model leaf or of a normalizer array; and --max-hdf5 fails on a library that is not older than the limit, at the
    limit too (the running library's own MAJOR.MINOR), so the older-libhdf5 job (in tests.yml) cannot pass on a new
    library.
    """
    import json
    import h5py

    # XV: the HDF5 cross-version check; "write" makes a synthetic bundle, "read" reads it back and compares digests
    XV = _load("h5_cross_version")
    out = tmp_path / "x"
    assert XV.main(["write", str(out)]) == 0, "writing the synthetic bundle failed"
    assert (
        XV.main(["read", str(out)]) == 0
    ), "the bundle written here did not read back with every digest equal"
    # the version of the HDF5 C library that h5py runs on, e.g. "1.14.2": its MAJOR.MINOR
    major, minor = (int(x) for x in h5py.version.hdf5_version.split(".")[:2])
    # --max-hdf5 L: the read must fail unless the running library is OLDER than L
    assert (
        XV.main(["read", str(out), "--max-hdf5", "1.0"]) == 1
    ), "passed with a limit far below the library"
    assert (
        XV.main(["read", str(out), "--max-hdf5", f"{major}.{minor}"]) == 1
    ), "passed with the running library's own version as the limit"
    assert (
        XV.main(["read", str(out), "--max-hdf5", f"{major}.{minor + 1}"]) == 0
    ), "failed with a limit above the running library"
    capsys.readouterr()
    pristine = (out / "expected.json").read_text()
    exp = json.loads(pristine)
    # expected.json holds one digest per array of each ensemble member of the bundle: edit one model-leaf digest of
    # member 1 and one normalizer digest of member 0
    model_key = sorted(k for k in exp["members"][1] if k.startswith("model"))[0]
    norm_key = sorted(k for k in exp["members"][0] if k.startswith("norm/"))[0]
    for member, key in ((1, model_key), (0, norm_key)):
        exp = json.loads(pristine)
        # a SHA-256 digest is 64 hexadecimal digits; all zeros is a wrong one, which the read must report
        exp["members"][member][key] = "0" * 64
        (out / "expected.json").write_text(json.dumps(exp))
        assert (
            XV.main(["read", str(out)]) == 1
        ), f"an edited digest of member {member} ({key}) passed"
        assert f"MISMATCH member {member} {key}" in capsys.readouterr().out


# ---- the levels: pr, nightly (tests/ci_tiers.toml); the nightly gate of a release


def test_the_pr_level_runs_the_pr_files_and_the_pr_tests_of_nightly_files_and_leaves_out_their_exceptions():
    """The pr level's plan, file by file: it runs every pr file whole except its nightly_tests and release_tests,
    only the pr_tests of a nightly file, and no other file."""
    _, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        plan = RT.tier_plan(tier, files, "pr")
        for rel, e in files.items():
            if e["tier"] != tier:
                continue
            # a file runs at pr if its level is pr, or if it is a nightly file that names pr_tests
            runs = e["level"] == "pr" or (e["level"] == "nightly" and bool(e.get("pr_tests")))
            assert (rel in plan["todo"]) is runs, (
                f"{rel} (level {e['level']}): expected to {'run' if runs else 'not run'} at pr, "
                f"but the plan {'leaves it out' if runs else 'runs it'}"
            )
            if not runs:
                continue
            if e["level"] == "pr":
                assert plan["select"][rel] == [rel], f"{rel}: a pr file runs whole"
            else:
                assert plan["select"][rel] == [
                    f"{rel}::{n}" for n in e["pr_tests"]
                ], f"{rel}: a nightly file runs only its pr_tests at pr"
            assert sorted(plan["deselect"][rel]) == sorted(
                f"{rel}::{n}"
                for n in e.get("release_tests", [])
                + (e.get("nightly_tests", []) if e["level"] == "pr" else [])
            ), f"{rel}: at pr, the release_tests (and a pr file's nightly_tests) are left out"
    # the loop's branches are not empty: a pr file with nightly_tests (17), a nightly file with pr_tests, and a
    # nightly file without them
    model = RT.tier_plan("wheel", files, "pr")
    left_out = model["deselect"]["tests/test_emulator_model.py"]
    assert len(left_out) == 17, (
        f"tests/ci_tiers.toml now lists {len(left_out)} nightly_tests for test_emulator_model.py, not 17; if that "
        "was intended, change 17 here"
    )
    assert model["select"]["tests/test_emulator_data.py"][0].startswith(
        "tests/test_emulator_data.py::"
    ), "test_emulator_data.py (a nightly file) must run only its named pr_tests at pr"
    assert (
        "tests/test_emulator_multifidelity.py" not in model["todo"]
    ), "test_emulator_multifidelity.py (a nightly file without pr_tests) must not run at pr"


def test_the_nightly_level_runs_every_ci_file_whole_except_its_release_tests():
    """The nightly level runs every pr and nightly file of the wheel and source tiers whole, except the release_tests
    (they need an asset); the external tier runs at nightly only; there is no "release" level for a CI tier.
    """
    _, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        plan = RT.tier_plan(tier, files, "nightly")
        assert plan["todo"] == sorted(
            p for p, e in files.items() if e["tier"] == tier and e["level"] in ("pr", "nightly")
        ), f"{tier} tier: the nightly level must run every pr and nightly file of the tier"
        for rel in plan["todo"]:
            assert plan["select"][rel] == [rel], f"{rel}: the nightly level runs every file whole"
            assert plan["deselect"][rel] == [
                f"{rel}::{n}" for n in files[rel].get("release_tests", [])
            ], f"{rel}: at nightly only the release_tests are left out"
    # the 19 release_tests of test_emulator_multifidelity.py (they need the multi-fidelity resolution table)
    multi = RT.tier_plan("wheel", files, "nightly")["deselect"][
        "tests/test_emulator_multifidelity.py"
    ]
    assert len(multi) == 19, (
        f"tests/ci_tiers.toml now lists {len(multi)} release_tests for test_emulator_multifidelity.py, not 19; if "
        "that was intended, change 19 here"
    )
    ext = RT.tier_plan("external", files, "nightly")
    assert ext["todo"] == [
        "tests/external/test_safetensors_interop.py"
    ], "the external tier at nightly must run exactly tests/external/test_safetensors_interop.py"
    # a run lists as not run only what it did not run: the external file, once run, is not among them
    listed = [p for p, _ in RT.not_run(files, ran_files=ext["todo"]).get("external", [])]
    assert ext["todo"][0] not in listed, "the external file that ran is still listed as not run"
    assert listed, "the other external files (not run) are missing from the not-run list"
    # refused: the external tier at pr, and any CI tier at "release" (release tests run only in the assets job)
    with pytest.raises(SystemExit):
        RT.tier_plan("external", files, "pr")
    with pytest.raises(SystemExit):
        RT.tier_plan("wheel", files, "release")


def test_a_named_test_that_is_the_start_of_another_name_is_refused(tmp_path, monkeypatch):
    """pytest selects and deselects by node-id prefix: a listed name that starts another function's name would also
    catch that function, so the runner refuses it; a name that is not a function of the file is refused too.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_a():\n    pass\n\n\ndef test_a_more():\n    pass\n\n\ndef test_b():\n    pass\n"
    )
    monkeypatch.setattr(RT, "REPO", str(tmp_path))
    base = {"tier": "wheel", "level": "pr"}
    RT.check_named_tests("tests/test_x.py", dict(base, nightly_tests=["test_b"]))
    with pytest.raises(SystemExit, match="is the start of"):
        RT.check_named_tests("tests/test_x.py", dict(base, nightly_tests=["test_a"]))
    with pytest.raises(SystemExit, match="not a function"):
        RT.check_named_tests("tests/test_x.py", dict(base, release_tests=["test_c"]))


def _fake_run(sha, status="completed", conclusion="success", started="2026-11-17T06:00:00Z", rid=1):
    """A made-up entry of the GitHub Actions list of workflow runs, with the keys nightly_gate.decide reads; by
    default a finished, passed run."""
    return dict(
        id=rid,
        head_sha=sha,
        status=status,
        conclusion=conclusion,
        run_started_at=started,
        html_url="u",
    )


SHA = "a" * 40  # a made-up full commit id (40 hexadecimal digits)


@pytest.mark.parametrize(
    "runs, passed",
    [
        ([], False),  # no nightly run on the commit
        ([_fake_run("b" * 40)], False),  # a run of another commit does not count
        ([_fake_run(SHA)], True),
        ([_fake_run(SHA, conclusion="failure")], False),
        ([_fake_run(SHA, conclusion="cancelled")], False),
        # the latest run decides: an older pass does not outweigh a newer failure, and the other way round
        (
            [
                _fake_run(SHA, rid=1),
                _fake_run(SHA, conclusion="failure", started="2026-11-17T07:00:00Z", rid=2),
            ],
            False,
        ),
        (
            [
                _fake_run(SHA, conclusion="failure", rid=1),
                _fake_run(SHA, started="2026-11-17T07:00:00Z", rid=2),
            ],
            True,
        ),
        # a newer run still going is a refusal until it has passed
        (
            [
                _fake_run(SHA, rid=1),
                _fake_run(
                    SHA,
                    status="in_progress",
                    conclusion=None,
                    started="2026-11-17T07:00:00Z",
                    rid=2,
                ),
            ],
            False,
        ),
    ],
)
def test_the_nightly_gate_passes_only_when_the_latest_run_on_the_commit_passed(runs, passed):
    """NG.decide(runs, commit) on made-up lists of nightly runs: it passes only when the latest run on that commit is
    finished and passed (each case above says what it shows)."""
    verdict, message = NG.decide(runs, SHA)  # decide returns (passed, the gate's own explanation)
    assert verdict is passed, f"the gate said {message!r}"


def test_the_nightly_gate_refuses_when_the_runs_cannot_be_read(monkeypatch, capsys):
    """The gate fails closed (when it cannot be sure, it refuses): if the GitHub API cannot be read, or the commit id
    is not a full 40-digit id, it refuses (exit status 1)."""

    def broken(*a, **k):
        raise OSError("no network")

    # fetch_runs replaced by a function that fails as an unreachable API would
    monkeypatch.setattr(NG, "fetch_runs", broken)
    assert NG.main(["--repo", "o/r", "--sha", SHA]) == 1, "the gate passed without reading the runs"
    assert "REFUSED" in capsys.readouterr().out
    # a short commit id is refused before any request
    assert NG.main(["--repo", "o/r", "--sha", "abc"]) == 1, "the gate accepted a short commit id"


def _steps_comparable(job):
    """The steps of a test job, made comparable between the pr jobs (tests.yml) and the nightly job (nightly.yml):
    the parts that may differ between the two are set to one neutral value, so that every other difference shows.
    """
    text = json.dumps(job["steps"])
    for a, b in (
        # the level a step runs at (pr in tests.yml, nightly in nightly.yml)
        ("--level nightly", "--level L"),
        ("--level pr", "--level L"),
        # nightly.yml's matrix entries sit under "stack" (matrix.stack.python); tests.yml's at the top (matrix.python)
        ("matrix.stack.", "matrix."),
    ):
        text = text.replace(a, b)
    steps = json.loads(text)
    for step in steps:
        # the name of the uploaded results differs by workflow and job; it is not compared
        if step.get("uses") == "actions/upload-artifact@v4":
            step["with"]["name"] = "NAME"
    return steps


def test_the_pr_workflow_runs_the_pr_level_and_the_nightly_workflow_the_same_steps_at_the_nightly_level():
    """tests.yml (pull requests, pushes to main) runs the pr level on the pinned stack; nightly.yml runs the same
    steps at the nightly level on the three stacks, plus the external tests of that level, once a day on the default
    branch and by hand. Why the same steps: if the two workflows drifted apart, a green pull request would no longer
    predict a green nightly run.

    It checks, in this order: 1. the levels each workflow passes; 2. nightly.yml's triggers and permissions;
    3. its external step; 4. the two pr jobs against each other; 5. the nightly job against the pr jobs.
    """
    # 1. the levels each workflow passes (and 5. ends with nightly.yml's count)
    jobs = _workflow("tests.yml")["jobs"]
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml"), encoding="utf-8").read()
    # the three places tests.yml passes "--level pr": the wheel tier, the source tier and the list of what is not run
    assert (
        text.count("--level pr") == 3 and "--level nightly" not in text
    ), "tests.yml must pass the pr level three times and never the nightly level"
    # 2. nightly.yml's triggers and permissions
    flow = _workflow("nightly.yml")
    # YAML 1.1 reads the workflow key `on` as the boolean True, so the parsed dict may hold it under either name
    on = flow.get("on", flow.get(True))
    assert set(on) == {
        "schedule",
        "workflow_dispatch",
    }, f"nightly.yml must start on a schedule and by hand only, not {sorted(on)}"
    assert len(on["schedule"]) == 1, "nightly.yml: one daily schedule"
    assert flow["permissions"] == {"contents": "read"}, "nightly.yml may only read the repository"
    job = flow["jobs"]["nightly"]
    assert set(job["strategy"]["matrix"]) == {
        "stack"
    }, "nightly.yml: the matrix holds the stacks only (no branch: it tests the commit that started it)"
    # 3. the external step: the one step nightly.yml has beyond the pr jobs. Its tests need an extra package,
    # safetensors (tests/external), which the step installs first.
    nightly_steps = _steps_comparable(job)
    extra = [s for s in nightly_steps if "--tier external" in s.get("run", "")]
    assert (
        len(extra) == 1 and "pip install safetensors" in extra[0]["run"]
    ), "nightly.yml: one external-tier step, which installs the extra package (safetensors) first"
    # 4. the two pr jobs, one per tier: the wheel job holds the fresh install, the wheel tier and the list of what is
    # not run; the source job holds the same preparation (build, checks, scan, install) and the source tier
    wheel = _steps_comparable(jobs["fixture-tests"])
    source = _steps_comparable(jobs["fixture-tests-source"])
    i_install = next(i for i, s in enumerate(wheel) if s.get("id") == "install")
    i_wheel = next(i for i, s in enumerate(wheel) if "--tier wheel" in s.get("run", ""))
    src_tier = [s for s in source if "--tier source" in s.get("run", "")]
    assert len(src_tier) == 1 and not [
        s for s in wheel if "--tier source" in s.get("run", "")
    ], "tests.yml: the source tier runs once, in the source job only"
    # the source job: the wheel job's steps up to and including the install, then the source tier, then the wheel
    # job's last step (the upload of the results)
    assert (
        source == wheel[: i_install + 1] + src_tier + wheel[-1:]
    ), "tests.yml: the source job must repeat the wheel job's preparation step for step, then run the source tier"
    # the README's first example (tools/ci/readme_first_example.py, in the fresh-install step) runs in the wheel job
    # only
    in_wheel = "readme_first_example" in json.dumps(wheel)
    in_source = "readme_first_example" in json.dumps(source)
    assert in_wheel and not in_source, "tests.yml: the README's example runs in the wheel job only"
    # 5. the nightly job: the wheel job's steps with the source tier inserted right after the wheel tier, plus the
    # external step
    up_to_wheel_tier, after_wheel_tier = wheel[: i_wheel + 1], wheel[i_wheel + 1 :]
    nightly_without_external = [s for s in nightly_steps if s not in extra]
    assert nightly_without_external == up_to_wheel_tier + src_tier + after_wheel_tier, (
        "nightly.yml's steps must equal the pr jobs' steps; " "change both files together"
    )
    ntext = open(os.path.join(REPO, ".github", "workflows", "nightly.yml"), encoding="utf-8").read()
    # the four places nightly.yml passes the nightly level: the wheel, source and external tiers and the list
    assert (
        ntext.count("--level nightly") == 4 and "--level pr" not in ntext
    ), "nightly.yml must pass the nightly level four times and never the pr level"


def _gate_steps(job):
    """The steps of a job that run the nightly gate (tools/ci/nightly_gate.py)."""
    return [s for s in job["steps"] if "tools/ci/nightly_gate.py" in s.get("run", "")]


def test_a_release_is_refused_unless_the_latest_nightly_run_on_its_commit_passed():
    """The release validation workflow starts only by hand, reads Actions with its own token (read access only),
    and checks the gate before anything else. The publishing workflow (publish.yml), disabled until the maintainer
    approves a release, checks the same gate first in its build job, without a token (the repository is public), so its
    permissions stay as they are."""
    flow = _workflow("release-validation.yml")
    # YAML 1.1 reads the workflow key `on` as the boolean True, so the parsed dict may hold it under either name
    on = flow.get("on", flow.get(True))
    assert list(on) == ["workflow_dispatch"], "release-validation.yml must start only by hand"
    assert flow["permissions"] == {
        "contents": "read",
        "actions": "read",
    }, "release-validation.yml may only read the repository and the Actions runs"
    # the release validation workflow has one job; its first step with a run command must be the gate
    (job,) = flow["jobs"].values()
    first_run = next(s for s in job["steps"] if "run" in s)
    assert _gate_steps(job) == [
        first_run
    ], "release-validation.yml: the gate must be the first step that runs a command, and the only gate step"
    # GitHub's expression for the workflow's own token, with the read access above
    assert first_run["env"] == {
        "GH_TOKEN": "${{ github.token }}"
    }, "release-validation.yml: the gate reads the runs with the workflow's own token, and nothing else"
    assert (
        '--sha "$GITHUB_SHA"' in first_run["run"]
    ), "release-validation.yml: the gate must check the commit the workflow runs on ($GITHUB_SHA)"
    text = open(
        os.path.join(REPO, ".github", "workflows", "release-validation.yml"), encoding="utf-8"
    ).read()
    assert (
        "check_dist_content.py dist --release" in text and "--list-not-run --level nightly" in text
    ), "release-validation.yml must run the release scan and list what CI does not run"
    publish = _workflow("publish.yml")["jobs"]["build"]
    gate = _gate_steps(publish)
    assert (
        len(gate) == 1 and next(s for s in publish["steps"] if "run" in s) == gate[0]
    ), "publish.yml: the gate must be the first step of the build job that runs a command"
    assert (
        "env" not in gate[0]
    ), "publish.yml: the gate step takes no token (its permissions stay unchanged)"
