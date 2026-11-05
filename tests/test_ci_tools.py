"""The CI helpers in tools/ci: the distribution-content check, the tier runner's verdict, and the pinned
development-environment versions against pyproject.toml's lower bounds. Synthetic inputs only."""
import importlib.util
import io
import os
import re
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
X = CD.git_blob_id(b"x")
PACKAGE = {n: X for n in ("hcd_analysis/__init__.py", "hcd_analysis/emulator/__init__.py",
                          "hcd_analysis/cli/__init__.py", "hcd_analysis/cli/run.py", *CD.PACKAGE_DATA)}
TOP = {n: X for n in CD.SDIST_COMMITTED}
DIST_INFO = ["METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"]


def _wheel(tmp_path, extra=(), drop=(), top="hcd_analysis\n", content=None):
    path = tmp_path / "hcd_priya-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted((set(PACKAGE) | set(extra)) - set(drop)):
            zf.writestr(name, (content or {}).get(name, "x"))
        for name in DIST_INFO:
            zf.writestr(f"hcd_priya-0.0.0.dist-info/{name}", "x")
        zf.writestr("hcd_priya-0.0.0.dist-info/top_level.txt", top)
    return str(path)


def _sdist(tmp_path, extra=(), drop=(), link=None, content=None):
    path = tmp_path / "hcd_priya-0.0.0.tar.gz"
    names = set(PACKAGE) | set(TOP) | {"PKG-INFO", "setup.cfg", "hcd_priya.egg-info/SOURCES.txt"} | set(extra)
    with tarfile.open(path, "w:gz") as tf:
        for name in sorted(names - set(drop)):
            data = (content or {}).get(name, "x").encode()
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(f"hcd_priya-0.0.0/{link}")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
    return str(path)


def test_git_blob_id_is_gits():
    assert CD.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"   # git hash-object /dev/null


def test_check_wheel_accepts_exactly_the_committed_package(tmp_path):
    assert CD.check_wheel(_wheel(tmp_path), PACKAGE) == []


@pytest.mark.parametrize("extra", ["hcd_analysis/_emulator_data/observables_tau0_lf.h5",
                                   "hcd_analysis/emulator/member.norm.pkl", "tests/test_x.py",
                                   "hcd_analysis/scratch.py", "hcd_analysis/emulator/README.md",
                                   "hcd_priya-0.0.0.dist-info/extra.txt"])
def test_check_wheel_refuses_any_other_file(tmp_path, extra):
    """Data, pickles, tests, an uncommitted module, docs or extra metadata never pass."""
    assert any(extra in p for p in CD.check_wheel(_wheel(tmp_path, extra=[extra]), PACKAGE))


@pytest.mark.parametrize("drop", ["hcd_analysis/cli/run.py", "hcd_analysis/comparison/phase0_v1.json"])
def test_check_wheel_refuses_a_missing_file(tmp_path, drop):
    assert any(f"missing {drop}" in p for p in CD.check_wheel(_wheel(tmp_path, drop=[drop]), PACKAGE))


def test_check_wheel_refuses_a_file_that_differs_from_head(tmp_path):
    """An uncommitted edit (or a stale build/ directory) ships different bytes under the right name."""
    problems = CD.check_wheel(_wheel(tmp_path, content={"hcd_analysis/cli/run.py": "edited"}), PACKAGE)
    assert problems == ["wheel: hcd_analysis/cli/run.py differs from HEAD"]


def test_check_wheel_refuses_another_top_level_package(tmp_path):
    """A generic top-level name such as cli, which another distribution could also install, never ships."""
    for top in ("cli\nhcd_analysis\n", "hcd_analysis\ntests\n"):
        assert any("top_level" in p for p in CD.check_wheel(_wheel(tmp_path, top=top), PACKAGE))


def test_check_sdist_accepts_exactly_the_committed_package(tmp_path):
    assert CD.check_sdist(_sdist(tmp_path), PACKAGE, TOP) == []


@pytest.mark.parametrize("extra", ["tests/test_x.py", "checkpoints/manifest.json", "scripts/run.py",
                                   "hcd_analysis/_emulator_data/observables_tau0_lf.h5", "hcd_analysis/scratch.py"])
def test_check_sdist_refuses_any_other_file(tmp_path, extra):
    assert any(extra in p for p in CD.check_sdist(_sdist(tmp_path, extra=[extra]), PACKAGE, TOP))


def test_check_sdist_refuses_links_missing_and_edited_files(tmp_path):
    problems = CD.check_sdist(_sdist(tmp_path, drop=["pyproject.toml"], link="hcd_analysis/x.py",
                                     content={"MANIFEST.in": "graft tests"}), PACKAGE, TOP)
    assert any("link" in p for p in problems) and any("missing pyproject.toml" in p for p in problems)
    assert "sdist: MANIFEST.in differs from HEAD" in problems


def test_committed_files_cover_the_packages_and_nothing_else():
    package, top = CD.committed_files()
    assert {"hcd_analysis/__init__.py", "hcd_analysis/paths.py", "hcd_analysis/cli/run.py"} <= set(package)
    assert set(CD.PACKAGE_DATA) <= set(package) and set(top) == set(CD.SDIST_COMMITTED)
    assert all(p.split("/")[0] in CD.PACKAGES and (p.endswith(".py") or p in CD.PACKAGE_DATA) for p in package)


_JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="t" name="a"/>
<testcase classname="t" name="b"><failure message="x"/></testcase>
<testcase classname="t" name="c"><error message="x"/></testcase>
<testcase classname="t" name="d[p1]"><skipped type="pytest.skip" message="absent"/></testcase>
<testcase classname="t" name="e"><skipped type="pytest.xfail" message="deferred_legacy X: y"/></testcase>
<testcase classname="t" name="f"><skipped type="pytest.xfail" message="some other reason"/></testcase>
</testsuite></testsuites>"""


def _c(**kw):
    out = dict.fromkeys(RT.COUNT_KEYS, 0)
    out.update(tests=3, passed=3)
    out.update(kw)
    return out


DEFERRAL = {"deferred_legacy X: y"}


def test_junit_counts_separate_skips_deferrals_and_other_expected_failures(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    assert RT.junit_counts(str(p), deferrals=DEFERRAL) == _c(tests=6, passed=1, failures=1, errors=1, skipped=1,
                                                             xfailed=1, bad_xfail=1)
    assert RT.junit_counts(str(p))["bad_xfail"] == 2        # only exactly declared deferral messages count


def test_junit_counts_declared_skips(tmp_path):
    """A declared skip that skips is NOT RUN, not a skip; a declared test that runs anyway is flagged."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"])
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 1)
    c = RT.junit_counts(str(p), declared_skips=["d[p1]"])          # one parameter case, named exactly
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0)
    c = RT.junit_counts(str(p), declared_skips=["d[p2]"])          # another case: the skip of d[p1] stays undeclared
    assert (c["skipped"], c["asset_skipped"]) == (1, 0)


def test_junit_counts_in_an_assets_job_a_required_test_must_run(tmp_path):
    """In a job meant to have the assets, a declared asset test that skips means its asset is missing; one that runs
    is what that job is for. A skipped_by_design test (declared, not required) may still skip."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["d", "a"])
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"], c["skipped"]) == (1, 0, 0, 0)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["a"])          # d skipped by design
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0)


def test_the_assets_job_runs_the_assets_tier_and_every_file_with_asset_tests():
    _, files = RT.load_tiers()
    picked = RT.asset_job_files(files)
    assert set(RT.files_of(files, "assets")) <= set(picked)
    for path in picked:
        assert files[path]["tier"] == "assets" or files[path].get("asset_tests")
    assert not {p for p in picked if files[p]["tier"] in ("deferred", "external", "no_tests")}
    assert not set(RT.not_run(files, ran=("assets",))) & {"assets", *RT.CI_TIERS}


def test_deferred_legacy_messages_are_exactly_what_conftest_writes():
    from tests.gate_helpers import DEFERRED_CAUSES
    assert RT.deferred_legacy_messages() == {f"deferred_legacy {cause}: {reason}"
                                             for cause, (_, reason) in DEFERRED_CAUSES.items()}


@pytest.mark.parametrize("rc, counts, ok", [
    (0, _c(), True),
    (0, _c(tests=5, xfailed=2), True),
    (0, _c(tests=4, asset_skipped=1), True),
    (0, _c(skipped=1), False),                      # a skip is never a pass
    (0, _c(bad_xfail=1), False),                    # only declared deferrals may be expected failures
    (0, _c(asset_ran=1), False),                    # a declared skip must skip without its asset
    (0, _c(asset_missing=1), False),                # an assets job: a missing asset fails, never skips
    (0, _c(tests=0, passed=0), False),              # nothing ran
    (0, _c(tests=2, passed=0, xfailed=2), False),   # deferrals only: nothing passed
    (0, _c(tests=2, passed=0, asset_skipped=2), False),
    (1, _c(), False),                               # e.g. the session-end origin check
    (1, _c(failures=1), False),
    (None, None, False),                            # timeout
    (4, None, False),                               # no readable junit report
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


# Direct dependencies whose lower bound is deliberately below the pinned release series, with the reason.
BOUND_BELOW_PIN = {"numpy": "2.0: jax 0.10.1 requires numpy 2.0 or later and no newer numpy interface is used"}


def test_devenv_pins_sit_in_the_release_series_of_the_lower_bounds():
    """tools/ci/constraints-devenv.txt pins the development environment; each pinned direct dependency's lower
    bound in pyproject.toml is the pin's release series (CI's development-environment job keeps the bounds tested),
    except the bounds named in BOUND_BELOW_PIN, which must still lie at or below the pin."""
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
        if name in BOUND_BELOW_PIN:
            assert BOUND_BELOW_PIN[name].startswith(bounds[name] + ":"), f"{name}: the stated bound is not the bound"
            assert tuple(map(int, b)) <= tuple(map(int, p[:len(b)])), f"{name}: bound {bounds[name]} above the pin"
            continue
        assert p[:len(b)] == b, f"{name}: bound {bounds[name]} is not the series of the pin {pins[name]}"


def test_the_workflow_calls_the_ci_helpers_that_exist():
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml")).read()
    for helper in ("tools/ci/check_dist.py", "tools/ci/run_test_tier.py", "tools/ci/constraints-devenv.txt",
                   "tools/ci/h5_cross_version.py"):
        assert helper in text and os.path.isfile(os.path.join(REPO, helper))
    for tier in ("--tier wheel", "--tier source", "--list-not-run"):
        assert tier in text


def test_every_run_step_fails_when_a_command_in_a_pipe_fails():
    """The workflow names bash as the shell of every run step, so GitHub runs each with `bash -eo pipefail` and
    `pytest ... | tee log` fails its step when pytest fails. With no shell named, GitHub runs `bash -e`, where a pipe's
    status is that of its last command (tee). No job or step may set another shell or its own defaults."""
    import yaml
    wf = yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", "tests.yml")))
    assert wf["defaults"] == {"run": {"shell": "bash"}}
    for name, job in wf["jobs"].items():
        assert "defaults" not in job, name
        for step in job["steps"]:
            assert "shell" not in step, (name, step.get("name"))


def test_only_the_pinned_job_declares_the_pinned_stack():
    """HCD_PINNED_STACK is "1" in the development-environment (supported) job and empty in the others."""
    import yaml
    job = yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", "tests.yml")))["jobs"]["fixture-tests"]
    expr = job["env"]["HCD_PINNED_STACK"]
    assert expr == "${{ matrix.deps == 'development-environment' && '1' || '' }}"
    assert {e["deps"] for e in job["strategy"]["matrix"]["include"]} == {"development-environment", "newest"}


def test_the_docs_build_is_pinned_and_treats_warnings_as_errors():
    """Read the Docs and the CI docs job install the same fully pinned requirements and fail on any Sphinx warning."""
    import yaml
    rtd = yaml.safe_load(open(os.path.join(REPO, ".readthedocs.yaml")))
    assert rtd["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}
    assert rtd["python"]["install"] == [{"requirements": "docs/requirements.txt"}]
    job = yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", "tests.yml")))["jobs"]["docs"]
    runs = [step.get("run", "") for step in job["steps"]]
    assert "python -m pip install -r docs/requirements.txt" in runs
    assert any(r.startswith("python -m sphinx -W --keep-going -b html docs ") for r in runs)
    lines = [ln.strip() for ln in open(os.path.join(REPO, "docs", "requirements.txt"))]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert reqs and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*", r) for r in reqs), reqs
    names = {r.split("==")[0].lower().replace("_", "-") for r in reqs}
    assert {"sphinx", "myst-parser", "sphinx-autoapi", "sphinx-rtd-theme", "sphinxcontrib-mermaid"} <= names


def test_h5_cross_version_check_reads_back_and_catches_a_change_or_a_new_library(tmp_path, capsys):
    """tools/ci/h5_cross_version.py: a synthetic bundle reads back with every digest equal; an edited digest fails; and
    --max-hdf5 fails on a library that is not older than the limit (here the running one), so the older-libhdf5 job
    cannot pass on a new library."""
    import json
    XV = _load("h5_cross_version")
    out = tmp_path / "x"
    assert XV.main(["write", str(out)]) == 0
    assert XV.main(["read", str(out)]) == 0
    assert XV.main(["read", str(out), "--max-hdf5", "1.0"]) == 1
    exp = json.loads((out / "expected.json").read_text())
    key = sorted(exp["members"][1])[0]
    exp["members"][1][key] = "0" * 64
    (out / "expected.json").write_text(json.dumps(exp))
    assert XV.main(["read", str(out)]) == 1
    assert f"MISMATCH member 1 {key}" in capsys.readouterr().out
