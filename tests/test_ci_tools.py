"""The CI helpers in tools/ci: the distribution-content check, the tier runner's verdict, and the pinned
development-environment versions against pyproject.toml's lower bounds. Synthetic inputs only."""
import importlib.util
import io
import os
import tarfile
import tomllib
import zipfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_ci_{name}", os.path.join(REPO, "tools", "ci", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CD = _load("check_dist")
RT = _load("run_test_tier")
MODULES = {"hcd_analysis/__init__.py", "hcd_analysis/emulator/__init__.py", "cli/__init__.py", "cli/run.py"}
DIST_INFO = ["METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"]


def _wheel(tmp_path, extra=(), drop=(), top="cli\nhcd_analysis\n"):
    path = tmp_path / "hcd_analysis-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted((MODULES | set(CD.PACKAGE_DATA) | set(extra)) - set(drop)):
            zf.writestr(name, "x")
        for name in DIST_INFO:
            zf.writestr(f"hcd_analysis-0.0.0.dist-info/{name}", "x")
        zf.writestr("hcd_analysis-0.0.0.dist-info/top_level.txt", top)
    return str(path)


def _sdist(tmp_path, extra=(), drop=(), link=None):
    path = tmp_path / "hcd_analysis-0.0.0.tar.gz"
    names = (MODULES | set(CD.PACKAGE_DATA) | {"pyproject.toml", "LICENSE", "PKG-INFO", "setup.cfg"} | set(extra))
    with tarfile.open(path, "w:gz") as tf:
        for name in sorted(names - set(drop)):
            info = tarfile.TarInfo(f"hcd_analysis-0.0.0/{name}")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        if link:
            info = tarfile.TarInfo(f"hcd_analysis-0.0.0/{link}")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
    return str(path)


def test_check_wheel_accepts_exactly_the_package(tmp_path):
    assert CD.check_wheel(_wheel(tmp_path), MODULES) == []


@pytest.mark.parametrize("extra", ["hcd_analysis/_emulator_data/observables_tau0_lf.h5",
                                   "hcd_analysis/emulator/member.norm.pkl", "tests/test_x.py",
                                   "hcd_analysis/emulator/README.md", "hcd_analysis-0.0.0.dist-info/extra.txt"])
def test_check_wheel_refuses_any_other_file(tmp_path, extra):
    assert any(extra in p for p in CD.check_wheel(_wheel(tmp_path, extra=[extra]), MODULES))


@pytest.mark.parametrize("drop", ["cli/run.py", "hcd_analysis/comparison/phase0_v1.json"])
def test_check_wheel_refuses_a_missing_file(tmp_path, drop):
    assert any(f"missing {drop}" in p for p in CD.check_wheel(_wheel(tmp_path, drop=[drop]), MODULES))


def test_check_wheel_refuses_another_top_level_package(tmp_path):
    assert any("top_level" in p for p in CD.check_wheel(_wheel(tmp_path, top="cli\nhcd_analysis\ntests\n"), MODULES))


def test_check_sdist_accepts_exactly_the_package(tmp_path):
    assert CD.check_sdist(_sdist(tmp_path), MODULES) == []


@pytest.mark.parametrize("extra", ["tests/test_x.py", "checkpoints/manifest.json", "scripts/run.py",
                                   "hcd_analysis/_emulator_data/observables_tau0_lf.h5"])
def test_check_sdist_refuses_any_other_file(tmp_path, extra):
    assert any(extra in p for p in CD.check_sdist(_sdist(tmp_path, extra=[extra]), MODULES))


def test_check_sdist_refuses_links_and_missing_files(tmp_path):
    problems = CD.check_sdist(_sdist(tmp_path, drop=["pyproject.toml"], link="hcd_analysis/x.py"), MODULES)
    assert any("link" in p for p in problems) and any("missing pyproject.toml" in p for p in problems)


def test_source_modules_cover_the_packages_and_nothing_else():
    mods = CD.source_modules()
    assert {"hcd_analysis/__init__.py", "hcd_analysis/emulator/safe_io.py", "cli/run.py"} <= mods
    assert all(m.split("/")[0] in CD.PACKAGES and m.endswith(".py") for m in mods)
    assert not any("_emulator_data" in m for m in mods)


_JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="t" name="a"/>
<testcase classname="t" name="b"><failure message="x"/></testcase>
<testcase classname="t" name="c"><error message="x"/></testcase>
<testcase classname="t" name="d[p1]"><skipped type="pytest.skip" message="absent"/></testcase>
<testcase classname="t" name="e"><skipped type="pytest.xfail" message="deferred_legacy X: y"/></testcase>
<testcase classname="t" name="f"><skipped type="pytest.xfail" message="some other reason"/></testcase>
</testsuite></testsuites>"""


def _c(**kw):
    out = dict(tests=3, passed=3, failures=0, errors=0, skipped=0, xfailed=0, bad_xfail=0, asset_skipped=0,
               asset_ran=0)
    out.update(kw)
    return out


def test_junit_counts_separate_skips_deferrals_and_other_expected_failures(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    assert RT.junit_counts(str(p)) == _c(tests=6, passed=1, failures=1, errors=1, skipped=1, xfailed=1, bad_xfail=1)


def test_junit_counts_declared_asset_tests(tmp_path):
    """A declared asset test that skips is NOT RUN, not a skip; one that runs anyway is flagged."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), asset_tests=["d", "a"])
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 1)
    c = RT.junit_counts(str(p), asset_tests=["d[p1]"])          # one parameter case, named exactly
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0)
    c = RT.junit_counts(str(p), asset_tests=["d[p2]"])          # another case: the skip of d[p1] stays undeclared
    assert (c["skipped"], c["asset_skipped"]) == (1, 0)


@pytest.mark.parametrize("rc, counts, ok", [
    (0, _c(), True),
    (0, _c(xfailed=2), True),
    (0, _c(tests=4, asset_skipped=1), True),
    (0, _c(skipped=1), False),                      # a skip is never a pass
    (0, _c(bad_xfail=1), False),                    # only deferred_legacy deferrals may be expected failures
    (0, _c(asset_ran=1), False),                    # a declared asset test must skip without its asset
    (0, _c(tests=0, passed=0), False),              # nothing ran
    (0, _c(tests=2, passed=0, asset_skipped=2), False),
    (1, _c(), False),                               # e.g. the session-end origin check
    (1, _c(failures=1), False),
    (None, None, False),                            # timeout
    (4, None, False),                               # no junit report
])
def test_file_verdict(rc, counts, ok):
    assert RT.file_verdict(rc, counts)[0] is ok


def test_the_runner_reads_the_real_tier_map():
    tiers, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        assert RT.files_of(files, tier) == sorted(p for p, e in files.items() if e["tier"] == tier)
    assert not set(RT.not_run(files)) & set(RT.CI_TIERS)


def _bounds():
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        proj = tomllib.load(f)["project"]
    reqs = proj["dependencies"] + proj["optional-dependencies"]["test"]
    out = {}
    for r in reqs:
        name, _, bound = r.partition(">=")
        out[name.strip().lower()] = bound.strip()
    return out


def test_devenv_pins_sit_in_the_release_series_of_the_lower_bounds():
    """tools/ci/constraints-devenv.txt pins the development environment; each pinned direct dependency's lower
    bound in pyproject.toml is the pin's release series (CI's development-environment job keeps the bounds tested)."""
    bounds = _bounds()
    pins = {}
    for line in open(os.path.join(REPO, "tools", "ci", "constraints-devenv.txt")):
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[name.lower()] = version
    direct = set(pins) & set(bounds)
    assert direct >= {"jax", "equinox", "optax", "numpyro", "numpy", "scipy", "h5py", "matplotlib", "pytest"}
    for name in direct:
        b, p = bounds[name].split("."), pins[name].split(".")
        assert p[:len(b)] == b, f"{name}: bound {bounds[name]} is not the series of the pin {pins[name]}"


def test_the_workflow_calls_the_ci_helpers_that_exist():
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml")).read()
    for helper in ("tools/ci/check_dist.py", "tools/ci/run_test_tier.py", "tools/ci/constraints-devenv.txt"):
        assert helper in text and os.path.isfile(os.path.join(REPO, helper))
    for tier in ("--tier wheel", "--tier source", "--list-not-run"):
        assert tier in text
