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
    spec = importlib.util.spec_from_file_location(
        f"_ci_{name}", os.path.join(REPO, "tools", "ci", f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CD = _load("check_dist")
RT = _load("run_test_tier")
NG = _load("nightly_gate")
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
            info.linkname = "/etc/passwd"
            tf.addfile(info)
    return str(path)


def test_git_blob_id_is_gits():
    assert (
        CD.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    )  # git hash-object /dev/null


def test_check_wheel_accepts_exactly_the_committed_package(tmp_path):
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
    assert any(extra in p for p in CD.check_wheel(_wheel(tmp_path, extra=[extra]), PACKAGE))


@pytest.mark.parametrize(
    "drop", ["hcd_analysis/cli/run.py", "hcd_analysis/comparison/phase0_v1.json"]
)
def test_check_wheel_refuses_a_missing_file(tmp_path, drop):
    assert any(
        f"missing {drop}" in p for p in CD.check_wheel(_wheel(tmp_path, drop=[drop]), PACKAGE)
    )


def test_check_wheel_refuses_a_file_that_differs_from_head(tmp_path):
    """An uncommitted edit (or a stale build/ directory) ships different bytes under the right name."""
    problems = CD.check_wheel(
        _wheel(tmp_path, content={"hcd_analysis/cli/run.py": "edited"}), PACKAGE
    )
    assert problems == ["wheel: hcd_analysis/cli/run.py differs from HEAD"]


def test_check_wheel_refuses_another_top_level_package(tmp_path):
    """A generic top-level name such as cli, which another distribution could also install, never ships."""
    for top in ("cli\nhcd_analysis\n", "hcd_analysis\ntests\n"):
        assert any("top_level" in p for p in CD.check_wheel(_wheel(tmp_path, top=top), PACKAGE))


def test_check_sdist_accepts_exactly_the_committed_package(tmp_path):
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
    assert any(extra in p for p in CD.check_sdist(_sdist(tmp_path, extra=[extra]), PACKAGE, TOP))


def test_check_sdist_refuses_links_missing_and_edited_files(tmp_path):
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
    assert any("link" in p for p in problems) and any(
        "missing pyproject.toml" in p for p in problems
    )
    assert "sdist: MANIFEST.in differs from HEAD" in problems


def test_committed_files_cover_the_packages_and_nothing_else():
    package, top = CD.committed_files()
    assert {"hcd_analysis/__init__.py", "hcd_analysis/paths.py", "hcd_analysis/cli/run.py"} <= set(
        package
    )
    assert set(CD.PACKAGE_DATA) <= set(package) and set(top) == set(CD.SDIST_COMMITTED)
    assert all(
        p.split("/")[0] in CD.PACKAGES and (p.endswith(".py") or p in CD.PACKAGE_DATA)
        for p in package
    )


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
    assert RT.junit_counts(str(p), deferrals=DEFERRAL) == _c(
        tests=6, passed=1, failures=1, errors=1, skipped=1, xfailed=1, bad_xfail=1
    )
    assert (
        RT.junit_counts(str(p))["bad_xfail"] == 2
    )  # only exactly declared deferral messages count


def test_junit_counts_declared_skips(tmp_path):
    """A declared skip that skips is NOT RUN, not a skip; a declared test that runs anyway is flagged."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"])
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 1)
    c = RT.junit_counts(str(p), declared_skips=["d[p1]"])  # one parameter case, named exactly
    assert (c["skipped"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0)
    c = RT.junit_counts(
        str(p), declared_skips=["d[p2]"]
    )  # another case: the skip of d[p1] stays undeclared
    assert (c["skipped"], c["asset_skipped"]) == (1, 0)


def test_junit_counts_in_an_assets_job_a_required_test_must_run(tmp_path):
    """In a job meant to have the assets, a declared asset test that skips means its asset is missing; one that runs
    is what that job is for. A skipped_by_design test (declared, not required) may still skip."""
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["d", "a"])
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"], c["skipped"]) == (1, 0, 0, 0)
    c = RT.junit_counts(str(p), declared_skips=["d", "a"], required=["a"])  # d skipped by design
    assert (c["asset_missing"], c["asset_skipped"], c["asset_ran"]) == (0, 1, 0)


def test_junit_counts_in_an_assets_job_a_required_test_absent_from_the_report_is_missing(tmp_path):
    """A required asset test that never reaches the report (not collected: a renamed test or parameter; or
    deselected) is missing too; a required name matched by its full id or by its function name is not.
    """
    p = tmp_path / "r.xml"
    p.write_text(_JUNIT)
    c = RT.junit_counts(str(p), declared_skips=["a", "zz"], required=["a", "zz"])
    assert (c["asset_missing"], c["passed"]) == (1, 1)
    c = RT.junit_counts(
        str(p), declared_skips=["d[p1]", "d[p9]", "a"], required=["d[p1]", "d[p9]", "a"]
    )
    assert c["asset_missing"] == 2  # d[p1] skipped, d[p9] absent
    c = RT.junit_counts(str(p), declared_skips=["a"], required=["a"])
    assert c["asset_missing"] == 0
    assert (
        RT.junit_counts(str(p), declared_skips=["zz"])["asset_missing"] == 0
    )  # not required: not counted


def test_the_assets_job_runs_the_assets_tier_and_every_file_with_asset_or_release_tests():
    _, files = RT.load_tiers()
    picked = RT.asset_job_files(files)
    assert set(RT.files_of(files, "assets")) <= set(picked)
    for path in picked:
        assert (
            files[path]["tier"] == "assets"
            or files[path].get("asset_tests")
            or files[path].get("release_tests")
        ), f"{path}: the assets job picked it, but it is not in the assets tier and declares no asset or release tests"
    assert not {p for p in picked if files[p]["tier"] in ("deferred", "external", "no_tests")}
    assert not set(RT.not_run(files, ran=("assets",))) & {"assets", *RT.CI_TIERS}


def test_the_plan_of_a_ci_tier_runs_its_files_without_required_tests_or_added_environment():
    _, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        plan = RT.tier_plan(tier, files)
        assert plan["todo"] == sorted(p for p, e in files.items() if e["tier"] == tier)
        assert plan["env"] == {} and plan["marker"] == "not slow"  # slow tests listed as NOT RUN
        assert set(plan["declared"]) == set(plan["required"]) == set(plan["todo"])
        for rel in plan["todo"]:
            assert plan["declared"][rel] == files[rel].get("asset_tests", []) + files[rel].get(
                "skipped_by_design", []
            )
            assert plan["required"][rel] == []


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
        assert not set(plan["required"][rel]) & set(files[rel].get("skipped_by_design", []))


def test_deferred_legacy_messages_are_exactly_what_conftest_writes():
    from tests.gate_helpers import DEFERRED_CAUSES

    assert RT.deferred_legacy_messages() == {
        f"deferred_legacy {cause}: {reason}" for cause, (_, reason) in DEFERRED_CAUSES.items()
    }


@pytest.mark.parametrize(
    "rc, counts, ok",
    [
        (0, _c(), True),
        (0, _c(tests=5, xfailed=2), True),
        (0, _c(tests=4, asset_skipped=1), True),
        (0, _c(skipped=1), False),  # a skip is never a pass
        (0, _c(bad_xfail=1), False),  # only declared deferrals may be expected failures
        (0, _c(asset_ran=1), False),  # a declared skip must skip without its asset
        (0, _c(asset_missing=1), False),  # an assets job: a missing asset fails, never skips
        (0, _c(tests=0, passed=0), False),  # nothing ran
        (0, _c(tests=2, passed=0, xfailed=2), False),  # deferrals only: nothing passed
        (0, _c(tests=2, passed=0, asset_skipped=2), False),
        (1, _c(), False),  # e.g. the session-end origin check
        (1, _c(failures=1), False),
        (None, None, False),  # timeout
        (4, None, False),  # no readable junit report
    ],
)
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
BOUND_BELOW_PIN = {
    "numpy": "2.0: jax 0.10.1 requires numpy 2.0 or later and no newer numpy interface is used"
}


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
    assert direct >= {
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
    for name in direct:
        b, p = bounds[name].split("."), pins[name].split(".")
        if name in BOUND_BELOW_PIN:
            assert BOUND_BELOW_PIN[name].startswith(
                bounds[name] + ":"
            ), f"{name}: the stated bound is not the bound"
            assert tuple(map(int, b)) <= tuple(
                map(int, p[: len(b)])
            ), f"{name}: bound {bounds[name]} above the pin"
            continue
        assert (
            p[: len(b)] == b
        ), f"{name}: bound {bounds[name]} is not the series of the pin {pins[name]}"


def test_the_workflow_calls_the_ci_helpers_that_exist():
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml")).read()
    for helper in (
        "tools/ci/check_dist.py",
        "tools/ci/run_test_tier.py",
        "tools/ci/constraints-devenv.txt",
        "tools/ci/h5_cross_version.py",
    ):
        assert helper in text and os.path.isfile(os.path.join(REPO, helper))
    for tier in ("--tier wheel", "--tier source", "--list-not-run"):
        assert tier in text


# Shell text that hides a failure. The first branch matches "set +e" and "set +o ..." (such as "set +o pipefail"),
# which turn off stopping on an error; the second matches "|| true", "|| :" and "|| exit 0", which make a failing
# command count as a success. (?![\w-]) stops the second branch matching the start of a longer word ("|| true-x").
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
    import yaml

    return yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", name)))


# the pr level runs as two jobs of tests.yml: the wheel tier's job and the source tier's
PR_JOBS = (
    "fixture-tests",
    "fixture-tests-source",
)


def test_only_the_pinned_job_declares_the_pinned_stack():
    """In a workflow, a "matrix" runs the same job several times with different settings; here each setting is a
    "stack", a Python version and a set of dependency versions. The pinned stack, Python 3.11 with the exact versions
    of tools/ci/constraints-devenv.txt, is the only supported environment (docs/compatibility.md).

    HCD_PINNED_STACK is "1" in the development-environment (supported) jobs and empty in the others, and it is set
    nowhere else: in tests.yml the name occurs exactly twice outside comments (the env of the two pr jobs, one per
    tier), so no other env, run command or write to $GITHUB_ENV (the file a step writes to set a variable for later
    steps) can set it. Each pr job's matrix is exactly its include list (no other axis) and holds the one supported
    stack; the nightly workflow holds it and the two newest-dependency stacks.
    """
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml"), encoding="utf-8").read()
    code = "\n".join(re.sub(r"(^|\s)#.*", "", line) for line in text.splitlines())
    assert code.count("HCD_PINNED_STACK") == len(PR_JOBS)
    wf = _workflow()
    assert "HCD_PINNED_STACK" not in (wf.get("env") or {})
    for name, job in wf["jobs"].items():
        if name not in PR_JOBS:
            assert "HCD_PINNED_STACK" not in (job.get("env") or {}), name
        for step in job["steps"]:
            assert "HCD_PINNED_STACK" not in (step.get("env") or {}), (name, step.get("name"))
    for name in PR_JOBS:
        job = wf["jobs"][name]
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
        ], name
    nightly = _workflow("nightly.yml")
    stacks = nightly["jobs"]["nightly"]["strategy"]["matrix"]["stack"]
    assert [(e["python"], e["deps"]) for e in stacks] == [
        ("3.11", "development-environment"),
        ("3.11", "newest"),
        ("3.12", "newest"),
    ]
    assert [e["label"].endswith("(supported)") for e in stacks] == [True, False, False]
    assert (
        nightly["jobs"]["nightly"]["env"]["HCD_PINNED_STACK"]
        == "${{ matrix.stack.deps == 'development-environment' && '1' || '' }}"
    ), "nightly.yml: HCD_PINNED_STACK must be 1 exactly in the development-environment stack"


# The `with` inputs each action may take: an input such as setup-python's pip-install would install packages
# outside the pinned steps.
ACTION_INPUTS = {
    "actions/checkout@v4": {"persist-credentials", "ref"},
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
                    # only nightly.yml checks out a branch by name ("ref": the release stack's branch on the daily
                    # schedule); the other workflows test the commit that started them
                    if flow != "nightly.yml":
                        assert (
                            "ref" not in given
                        ), f"{flow} job {name}: only nightly.yml may check out another branch (ref)"


def test_the_docs_build_is_pinned_and_treats_warnings_as_errors():
    """Read the Docs and the CI docs job install the same fully pinned requirements and fail on any Sphinx warning."""
    import yaml

    rtd = yaml.safe_load(open(os.path.join(REPO, ".readthedocs.yaml")))
    assert rtd["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}
    assert rtd["python"]["install"] == [{"requirements": "docs/requirements.txt"}]
    job = _workflow()["jobs"]["docs"]
    assert [step.get("uses") for step in job["steps"] if "uses" in step] == [
        "actions/checkout@v4",
        "actions/setup-python@v5",
    ]
    runs = [step["run"] for step in job["steps"] if "run" in step]
    assert runs == [
        "python -m pip install -r docs/requirements.txt",
        'python -m sphinx -W --keep-going -b html docs "$RUNNER_TEMP/html"',
    ]  # nothing else installed
    lines = [ln.strip() for ln in open(os.path.join(REPO, "docs", "requirements.txt"))]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert reqs and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*", r) for r in reqs), reqs
    names = {r.split("==")[0].lower().replace("_", "-") for r in reqs}
    assert {
        "sphinx",
        "myst-parser",
        "sphinx-autoapi",
        "sphinx-rtd-theme",
        "sphinxcontrib-mermaid",
    } <= names


def test_h5_cross_version_check_reads_back_and_catches_a_change_or_a_new_library(tmp_path, capsys):
    """tools/ci/h5_cross_version.py: a synthetic bundle reads back with every digest equal; an edited digest fails, of
    a model leaf or of a normalizer array; and --max-hdf5 fails on a library that is not older than the limit, at the
    limit too (the running library's own MAJOR.MINOR), so the older-libhdf5 job cannot pass on a new library.
    """
    import json
    import h5py

    XV = _load("h5_cross_version")
    out = tmp_path / "x"
    assert XV.main(["write", str(out)]) == 0
    assert XV.main(["read", str(out)]) == 0
    major, minor = (int(x) for x in h5py.version.hdf5_version.split(".")[:2])
    assert XV.main(["read", str(out), "--max-hdf5", "1.0"]) == 1
    assert XV.main(["read", str(out), "--max-hdf5", f"{major}.{minor}"]) == 1
    assert XV.main(["read", str(out), "--max-hdf5", f"{major}.{minor + 1}"]) == 0
    capsys.readouterr()
    pristine = (out / "expected.json").read_text()
    exp = json.loads(pristine)
    model_key = sorted(k for k in exp["members"][1] if k.startswith("model"))[0]
    norm_key = sorted(k for k in exp["members"][0] if k.startswith("norm/"))[0]
    for member, key in ((1, model_key), (0, norm_key)):
        exp = json.loads(pristine)
        exp["members"][member][key] = "0" * 64
        (out / "expected.json").write_text(json.dumps(exp))
        assert XV.main(["read", str(out)]) == 1
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
            runs = e["level"] == "pr" or (e["level"] == "nightly" and bool(e.get("pr_tests")))
            assert (rel in plan["todo"]) is runs, rel
            if not runs:
                continue
            if e["level"] == "pr":
                assert plan["select"][rel] == [rel]
            else:
                assert plan["select"][rel] == [f"{rel}::{n}" for n in e["pr_tests"]]
            assert sorted(plan["deselect"][rel]) == sorted(
                f"{rel}::{n}"
                for n in e.get("release_tests", [])
                + (e.get("nightly_tests", []) if e["level"] == "pr" else [])
            )
    # The loop above checks every file against its entry; these lines make sure its branches are not empty, using
    # files whose entries in tests/ci_tiers.toml carry per-test lists: test_emulator_model.py, a pr file, names 17
    # nightly_tests (its training-loss tests; it has no release_tests), so the pr level leaves out exactly 17 of its
    # tests; test_emulator_data.py, a nightly file, names pr_tests; test_emulator_multifidelity.py, a nightly file
    # without pr_tests, does not run at pr. If you change those entries, change the number here too.
    model = RT.tier_plan("wheel", files, "pr")
    assert len(model["deselect"]["tests/test_emulator_model.py"]) == 17
    assert model["select"]["tests/test_emulator_data.py"][0].startswith(
        "tests/test_emulator_data.py::"
    )
    assert "tests/test_emulator_multifidelity.py" not in model["todo"]


def test_the_nightly_level_runs_every_ci_file_whole_except_its_release_tests():
    """The nightly level runs every pr and nightly file of the wheel and source tiers whole, except the release_tests
    (they need an asset); the external tier runs at nightly only; there is no "release" level for a CI tier.
    """
    _, files = RT.load_tiers()
    for tier in RT.CI_TIERS:
        plan = RT.tier_plan(tier, files, "nightly")
        assert plan["todo"] == sorted(
            p for p, e in files.items() if e["tier"] == tier and e["level"] in ("pr", "nightly")
        )
        for rel in plan["todo"]:
            assert plan["select"][rel] == [rel]
            assert plan["deselect"][rel] == [
                f"{rel}::{n}" for n in files[rel].get("release_tests", [])
            ]
    # test_emulator_multifidelity.py names 19 release_tests in tests/ci_tiers.toml (the tests that need the
    # multi-fidelity resolution table); the nightly level leaves out exactly those. If you change that entry, change
    # the number here too.
    multi = RT.tier_plan("wheel", files, "nightly")["deselect"][
        "tests/test_emulator_multifidelity.py"
    ]
    assert len(multi) == 19
    ext = RT.tier_plan("external", files, "nightly")
    assert ext["todo"] == ["tests/external/test_safetensors_interop.py"]
    # a run lists as not run only what it did not run: the external file, once run, is not among them
    listed = [p for p, _ in RT.not_run(files, ran_files=ext["todo"]).get("external", [])]
    assert ext["todo"][0] not in listed and listed
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


def _run(sha, status="completed", conclusion="success", started="2026-11-17T06:00:00Z", rid=1):
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
        ([_run("b" * 40)], False),  # a run of another commit does not count
        ([_run(SHA)], True),
        ([_run(SHA, conclusion="failure")], False),
        ([_run(SHA, conclusion="cancelled")], False),
        # the latest run decides: an older pass does not outweigh a newer failure, and the other way round
        (
            [
                _run(SHA, rid=1),
                _run(SHA, conclusion="failure", started="2026-11-17T07:00:00Z", rid=2),
            ],
            False,
        ),
        (
            [
                _run(SHA, conclusion="failure", rid=1),
                _run(SHA, started="2026-11-17T07:00:00Z", rid=2),
            ],
            True,
        ),
        # a newer run still going is a refusal until it has passed
        (
            [
                _run(SHA, rid=1),
                _run(
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
    verdict, message = NG.decide(runs, SHA)  # decide returns (passed, the gate's own explanation)
    assert verdict is passed, f"the gate said {message!r}"


def test_the_nightly_gate_refuses_when_the_runs_cannot_be_read(monkeypatch, capsys):
    def broken(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(NG, "fetch_runs", broken)
    assert NG.main(["--repo", "o/r", "--sha", SHA]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert NG.main(["--repo", "o/r", "--sha", "abc"]) == 1  # not a full sha


def _steps_without_level(job):
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
        # the branch to check out: only nightly.yml names one
        (step.get("with") or {}).pop("ref", None)
        # the name of the uploaded results differs by workflow and job; it is not compared
        if step.get("uses") == "actions/upload-artifact@v4":
            step["with"]["name"] = "NAME"
    return steps


def test_the_pr_workflow_runs_the_pr_level_and_the_nightly_workflow_the_same_steps_at_the_nightly_level():
    """tests.yml (pull requests, pushes to main) runs the pr level on the pinned stack; nightly.yml runs the same
    steps at the nightly level on the three stacks, plus the external tests of that level, once a day (and on the
    release stack's branch while it is open), on each push to that branch and by hand. Why the same steps: if the two
    workflows drifted apart, a green pull request would no longer predict a green nightly run."""
    jobs = _workflow("tests.yml")["jobs"]
    text = open(os.path.join(REPO, ".github", "workflows", "tests.yml"), encoding="utf-8").read()
    # the three places tests.yml passes "--level pr": the wheel tier, the source tier and the list of what is not run
    assert text.count("--level pr") == 3 and "--level nightly" not in text
    flow = _workflow("nightly.yml")
    # YAML 1.1 reads the workflow key `on` as the boolean True, so the parsed dict may hold it under either name
    on = flow.get("on", flow.get(True))
    assert set(on) == {"schedule", "push", "workflow_dispatch"}
    # TEMPORARY: nightly.yml names the release stack's branch, ci/joss-stack-optb, until pull request #23 merges.
    # Remove the branch from nightly.yml and from the two checks below together, when #23 merges.
    assert len(on["schedule"]) == 1 and on["push"] == {"branches": ["ci/joss-stack-optb"]}
    assert flow["permissions"] == {"contents": "read"}
    job = flow["jobs"]["nightly"]
    assert (
        "ci/joss-stack-optb" in job["strategy"]["matrix"]["ref"]
        and "schedule" in job["strategy"]["matrix"]["ref"]
    )
    nightly_steps = _steps_without_level(job)
    extra = [s for s in nightly_steps if "--tier external" in s.get("run", "")]
    assert len(extra) == 1 and "pip install safetensors" in extra[0]["run"]
    # the pr level runs as two jobs, one per tier: the wheel job holds the fresh install, the wheel tier and the list
    # of what is not run; the source job holds the same preparation (build, checks, scan, install) and the source tier
    wheel = _steps_without_level(jobs["fixture-tests"])
    source = _steps_without_level(jobs["fixture-tests-source"])
    i_install = next(i for i, s in enumerate(wheel) if s.get("id") == "install")
    i_wheel = next(i for i, s in enumerate(wheel) if "--tier wheel" in s.get("run", ""))
    src_tier = [s for s in source if "--tier source" in s.get("run", "")]
    assert len(src_tier) == 1 and not [s for s in wheel if "--tier source" in s.get("run", "")]
    # the source job: the wheel job's steps up to and including the install, then the source tier, then the wheel
    # job's last step (the upload of the results)
    assert source == wheel[: i_install + 1] + src_tier + wheel[-1:]
    # the README's first example (tools/ci/readme_first_example.py, in the fresh-install step) runs in the wheel job
    # only
    assert "readme_first_example" not in json.dumps(
        source
    ) and "readme_first_example" in json.dumps(wheel)
    # the nightly job: the wheel job's steps with the source tier inserted right after the wheel tier, plus the
    # external step
    assert [s for s in nightly_steps if s not in extra] == wheel[: i_wheel + 1] + src_tier + wheel[
        i_wheel + 1 :
    ]
    ntext = open(os.path.join(REPO, ".github", "workflows", "nightly.yml"), encoding="utf-8").read()
    # the four places nightly.yml passes the nightly level: the wheel, source and external tiers and the list
    assert ntext.count("--level nightly") == 4 and "--level pr" not in ntext


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
    assert list(on) == ["workflow_dispatch"]
    assert flow["permissions"] == {"contents": "read", "actions": "read"}
    # the release validation workflow has one job; its first step with a run command must be the gate
    (job,) = flow["jobs"].values()
    first_run = next(s for s in job["steps"] if "run" in s)
    assert _gate_steps(job) == [first_run]
    assert first_run["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert '--sha "$GITHUB_SHA"' in first_run["run"]
    text = open(
        os.path.join(REPO, ".github", "workflows", "release-validation.yml"), encoding="utf-8"
    ).read()
    assert (
        "check_dist_content.py dist --release" in text and "--list-not-run --level nightly" in text
    )
    publish = _workflow("publish.yml")["jobs"]["build"]
    gate = _gate_steps(publish)
    assert len(gate) == 1 and next(s for s in publish["steps"] if "run" in s) == gate[0]
    assert "env" not in gate[0]
