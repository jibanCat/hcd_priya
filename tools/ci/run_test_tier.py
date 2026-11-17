"""Run one CI test tier of tests/ci_tiers.toml at one level, one pytest process per file, and refuse any undeclared skip.

Terms: a tier says HOW a file runs (wheel: against the installed package; source: from the checkout), a level says
WHEN it runs (pr: on every pull request; nightly: once a day). Both, and the per-file keys used below (asset_tests,
skipped_by_design, deferred_xfails, pr_tests, nightly_tests, release_tests), are defined in the header of
tests/ci_tiers.toml. A junit report is the XML file pytest writes with --junitxml. The header of tests/ci_tiers.toml
also defines an asset, NOT RUN and the skip rules; a test marked slow is one with @pytest.mark.slow (registered in
tests/conftest.py); a deferral and HCD_GATE_RUN are described below and in tests/gate_helpers.py.

usage:
  python tools/ci/run_test_tier.py --tier wheel  --level LEVEL --workdir NEW_DIR --out NEW_DIR [--wheel WHEEL]
  python tools/ci/run_test_tier.py --tier source --level LEVEL --out NEW_DIR
  python tools/ci/run_test_tier.py --tier external --level nightly --out NEW_DIR
  python tools/ci/run_test_tier.py --tier assets --out NEW_DIR
  python tools/ci/run_test_tier.py --list-not-run [--level LEVEL]

Typical local run, as the pr level's source job runs it: python tools/ci/run_test_tier.py --tier source --level pr
--out results_pr  (a folder that does not exist yet)

NEW_DIR must not exist yet; the runner creates it (--out: each file's log and junit report, and summary.json;
--workdir: the copy of tests/ that the wheel tier runs from).

LEVEL is pr (every pull request) or nightly (the pr level and the nightly files); without --level a tier runs at
nightly, everything CI runs. At pr, a nightly file runs only the tests its entry names in pr_tests, and a pr file
does not run the tests in its nightly_tests. At every CI level the tests in a file's release_tests are deselected:
they need an asset and fail without it; the assets job runs them, and there they must run.

Run it with the interpreter under test (CI: the fresh venv holding the installed wheel).

- wheel tier: tests/ is copied into NEW_DIR, and pytest runs from there, so hcd_analysis can only come from the
  interpreter's site-packages; the runner checks that before running anything, and with --wheel also that the
  installed files are exactly that wheel's (the RECORD entries of its package files).
- source tier: pytest runs from the repository checkout this script belongs to, so hcd_analysis comes from it.
- external tier, at nightly only: the files of tests/external whose level is nightly, from the checkout, in an
  interpreter that has their extra package (the nightly workflow installs it); a skip fails, as everywhere.
- assets: a job meant to test the real model, never run by the CI workflow (it has no assets). From the checkout,
  it runs the files of the "assets" tier and every wheel or source file that declares asset_tests or
  release_tests, with HCD_GATE_RUN=1 (tests/gate_helpers.py: a missing real cache fails instead of skipping), and
  with the tests marked slow. There a declared asset test or release test MUST run: if it skips or is not in the
  junit report at all (not collected, or deselected), its file fails.

A file passes only if pytest exits 0 and its junit report has no failure, no error and no undeclared skip, and at least
one test passed. Besides passes, only these outcomes are allowed; each is counted and listed, never as a pass:
- an expected failure whose message is exactly that of an explicit deferral. A deferral is a test known to fail,
  recorded in tests/DEFERRED.md under a deferred_legacy cause of tests/gate_helpers.py DEFERRED_CAUSES (tests/conftest.py
  marks such a test with its exact cause), or a reason that tests/ci_tiers.toml lists in the file's deferred_xfails (a
  strict, exact-exception xfail);
- a skip of a test that tests/ci_tiers.toml names in the file's asset_tests (it needs a real asset that CI does not
  have) or skipped_by_design (skipped unconditionally until a later step). Such a test must skip here: if it passes
  or fails instead, the file fails, so the lists stay exact.
In the wheel and source tiers, tests marked slow are deselected (-m "not slow") and listed as NOT RUN. The exit status
is 1 if any file did not pass. The summary also lists the files of the tiers that CI does not run, with their reasons.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import tomllib
import xml.etree.ElementTree as ET
import zipfile

# the repository root: this file is tools/ci/run_test_tier.py, three folders below it
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TIERS_FILE = os.path.join(REPO, "tests", "ci_tiers.toml")
CI_TIERS = ("wheel", "source")
ASSET_JOB = "assets"
EXTERNAL = "external"
LEVELS = ("pr", "nightly", "release", "not run")
# the file levels that run at each CI level
RUN_LEVELS = {"pr": ("pr",), "nightly": ("pr", "nightly")}
# The keys that name single tests whose level differs from their file's (the header of tests/ci_tiers.toml):
# pr_tests sits in a nightly file (these tests also run at pr); nightly_tests in a pr file (these run at nightly
# only); release_tests in either (CI leaves them out; the assets job runs them).
PER_TEST_KEYS = ("pr_tests", "nightly_tests", "release_tests")
# a file whose pytest run takes longer than this (30 minutes) fails as a timeout
FILE_TIMEOUT_S = 1800
# The outcome counts kept for each file, read from its junit report by junit_counts. Plain pytest outcomes: tests,
# passed, failures, errors, and skipped (an UNDECLARED skip: the file fails). The others:
#   xfailed        expected failures whose message is a declared deferral: allowed, listed, never a pass;
#   bad_xfail      expected failures that are not a declared deferral: the file fails;
#   asset_skipped  declared skips (asset_tests, skipped_by_design) that did skip: allowed, reported as NOT RUN;
#   asset_ran      declared skips that ran instead of skipping: the file fails (its list in ci_tiers.toml is stale);
#   asset_missing  in the assets job only: tests it must run that skipped or are not in the report: the file fails.
COUNT_KEYS = (
    "tests",
    "passed",
    "failures",
    "errors",
    "skipped",
    "xfailed",
    "bad_xfail",
    "asset_skipped",
    "asset_ran",
    "asset_missing",
)


def load_tiers(path=TIERS_FILE):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    return doc["tiers"], doc["files"]


def files_of(files, tier):
    return sorted(p for p, entry in files.items() if entry["tier"] == tier)


def asset_job_files(files):
    """The files an assets job runs: the "assets" tier, and every CI-tier file that declares asset_tests or
    release_tests."""
    return sorted(
        p
        for p, entry in files.items()
        if entry["tier"] == ASSET_JOB
        or (entry["tier"] in CI_TIERS and (entry.get("asset_tests") or entry.get("release_tests")))
    )


def defined_functions(rel):
    """The names of the functions defined in a test file of this checkout."""
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return {n.name for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.FunctionDef)}


def check_named_tests(rel, entry):
    """The names of pr_tests, nightly_tests and release_tests must be test functions of the file, and none may be the
    start of another function's name: pytest selects and deselects by node-id prefix, so such a name would also catch
    the other function."""
    names = [n for key in PER_TEST_KEYS for n in entry.get(key, [])]
    if not names:
        return
    defined = defined_functions(rel)
    for n in names:
        if n not in defined:
            sys.exit(
                f"{rel}: {n} is named in tests/ci_tiers.toml but is not a function of the file; correct the "
                "name or delete it from the file's entry"
            )
        clash = sorted(d for d in defined if d != n and d.startswith(n))
        if clash:
            sys.exit(
                f"{rel}: {n} is the start of {clash}: a node-id prefix would select those too; rename one of "
                "the functions"
            )


def level_selection(rel, entry, level):
    """How a CI level runs one file: (run, select, deselect). run is False when the file does not run at this level;
    select is the pytest arguments that choose its tests (the file, or node ids of its pr_tests); deselect is the node
    ids to leave out (nightly_tests at pr, release_tests at every CI level)."""
    check_named_tests(rel, entry)
    deselect = [f"{rel}::{n}" for n in entry.get("release_tests", [])]
    if entry["level"] in RUN_LEVELS[level]:
        if level == "pr":
            deselect += [f"{rel}::{n}" for n in entry.get("nightly_tests", [])]
        return True, [rel], deselect
    if level == "pr" and entry["level"] == "nightly" and entry.get("pr_tests"):
        return True, [f"{rel}::{n}" for n in entry["pr_tests"]], deselect
    return False, [], []


def tier_plan(tier, files, level="nightly"):
    """What a run of `tier` does, as a dict (a CI tier runs at `level`; the assets job runs every test of its files):
    - todo: the files it runs;
    - level: the CI level, or None for the assets job;
    - env: the environment variables it adds (HCD_GATE_RUN=1 in the assets job);
    - marker: the pytest -m expression of each file's run ("not slow" in a CI tier; None: every test);
    - declared: per file, the tests allowed to skip (asset_tests and skipped_by_design);
    - required: per file, the tests that must run (in the assets job, asset_tests and release_tests; none in a CI
      tier);
    - select: per file, the pytest arguments that choose its tests (the file, or node ids of its pr_tests);
    - deselect: per file, the node ids left out.
    """
    is_assets_job = tier == ASSET_JOB
    select, deselect = {}, {}
    if is_assets_job:
        todo = asset_job_files(files)
        for rel in todo:
            check_named_tests(rel, files[rel])
            select[rel], deselect[rel] = [rel], []
    else:
        if level not in RUN_LEVELS:
            sys.exit(f"level {level!r}: a CI tier runs at {' or '.join(RUN_LEVELS)}")
        if tier == EXTERNAL and level != "nightly":
            sys.exit("the external tier runs at the nightly level only")
        todo = []
        for rel in files_of(files, tier):
            run, sel, desel = level_selection(rel, files[rel], level)
            if run:
                todo.append(rel)
                select[rel], deselect[rel] = sel, desel
    return dict(
        todo=todo,
        level=None if is_assets_job else level,
        env={"HCD_GATE_RUN": "1"} if is_assets_job else {},
        marker=None if is_assets_job else "not slow",
        declared={
            rel: files[rel].get("asset_tests", []) + files[rel].get("skipped_by_design", [])
            for rel in todo
        },
        required={
            rel: (
                list(files[rel].get("asset_tests", [])) + list(files[rel].get("release_tests", []))
                if is_assets_job
                else []
            )
            for rel in todo
        },
        select=select,
        deselect=deselect,
    )


def deferred_legacy_messages(path=os.path.join(REPO, "tests", "gate_helpers.py")):
    """The exact junit messages of the deferred_legacy causes, as tests/conftest.py writes them, read from the
    DEFERRED_CAUSES literal without importing the tests."""
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "DEFERRED_CAUSES" for t in node.targets
        ):
            return {
                f"deferred_legacy {cause}: {reason}"
                for cause, (_, reason) in ast.literal_eval(node.value).items()
            }
    raise SystemExit(f"no DEFERRED_CAUSES in {path}")


def junit_counts(xml_path, declared_skips=(), deferrals=(), required=()):
    """Outcome counts of a pytest junit report (see the module docstring). declared_skips are the file's asset_tests
    and skipped_by_design together; a test matches one by its full name (with parameters, e.g. "test_x[a]") or by
    its function name (all parameters). deferrals are the exact expected-failure messages allowed. required (an
    assets job: the file's asset_tests and release_tests) are tests that must run here: each one that skips, and each
    required name that no test of the report matches (not collected, or deselected), counts as asset_missing.

    Each test lands in exactly one outcome, the first that applies in this order: failures, errors, xfailed or
    bad_xfail (an expected failure), asset_missing (a required test that skipped), asset_skipped or skipped (any
    other skip), passed. asset_ran is counted besides the outcome: a declared skip that did not skip.
    """
    declared_names, allowed_messages, required_names = (
        set(declared_skips),
        set(deferrals),
        set(required),
    )
    counts = dict.fromkeys(COUNT_KEYS, 0)
    seen = set()  # the names in the report, full and function-only
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        counts["tests"] += 1
        name = case.get("name", "")
        base = name.split("[")[0]  # the function name: "test_x[a]" -> "test_x"
        seen |= {name, base}
        is_declared = name in declared_names or base in declared_names
        is_required = name in required_names or base in required_names
        skip_tag = case.find("skipped")  # the report's <skipped> element, or None
        if case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif skip_tag is not None and skip_tag.get("type") == "pytest.xfail":
            # pytest writes an expected failure as a skip of type pytest.xfail; only an allowed message is a deferral
            counts["xfailed" if skip_tag.get("message") in allowed_messages else "bad_xfail"] += 1
        elif skip_tag is not None and is_required:
            # a test that this job must run skipped: its asset is missing
            counts["asset_missing"] += 1
        elif skip_tag is not None:
            # any other skip: declared (reported as NOT RUN) or undeclared (the file fails)
            counts["asset_skipped" if is_declared else "skipped"] += 1
        else:
            counts["passed"] += 1
        if is_declared and not is_required and skip_tag is None:
            # a declared skip that ran instead of skipping (outside the tests a job requires): the file fails
            counts["asset_ran"] += 1
    # a required test that is not in the report at all (not collected, or deselected)
    counts["asset_missing"] += len(required_names - seen)
    return counts


def file_verdict(returncode, counts):
    """(passed, reason) for one file's pytest run. The checks run in this order and the first that fails gives the
    reason: a timeout; no readable report; any count in the list below; no test passed; a non-zero pytest exit.
    """
    if returncode is None:
        return False, f"timeout after {FILE_TIMEOUT_S} s"
    if counts is None:
        return False, f"no readable junit report (exit {returncode})"
    for key, what in (
        ("failures", "failed"),
        ("errors", "errors"),
        ("skipped", "undeclared skips"),
        ("bad_xfail", "expected failures that are not declared deferrals"),
        ("asset_ran", "declared skips that did not skip"),
        (
            "asset_missing",
            "asset tests not run (skipped, not collected or deselected) in a job that "
            "must have their assets",
        ),
    ):
        if counts[key]:
            return False, f"{counts[key]} {what}"
    if counts["passed"] == 0:
        return False, "no test passed"
    if returncode != 0:
        return False, f"pytest exit {returncode}"
    return True, "ok"


def package_origin(python, cwd):
    """The real path of the hcd_analysis/__init__.py that `python` imports when started in `cwd`."""
    out = subprocess.run(
        [python, "-c", "import hcd_analysis; print(hcd_analysis.__file__)"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return os.path.realpath(out.stdout.strip())


def check_origin(tier, origin):
    """Refuse unless hcd_analysis is imported from where this tier must take it: the wheel tier from this interpreter's
    site-packages, the other tiers from the checkout. `origin` is the path of hcd_analysis/__init__.py, so two
    dirnames give the folder that holds the package."""
    pkg_parent = os.path.dirname(os.path.dirname(origin))
    if tier == "wheel":
        site = {os.path.realpath(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")}
        if pkg_parent not in site:
            sys.exit(
                f"wheel tier: hcd_analysis imports from {origin}, not from this interpreter's site-packages; "
                "install the wheel into this interpreter (python -m pip install <the wheel>) and run the runner "
                "with that interpreter"
            )
    elif pkg_parent != REPO:
        sys.exit(
            f"{tier} tier: hcd_analysis imports from {origin}, not from the checkout {REPO}; run this checkout's "
            "runner with an interpreter whose import path does not put another hcd_analysis first (check "
            "PYTHONPATH and any .pth file in its site-packages)"
        )


def check_installed_is(wheel):
    """Every package file the wheel's RECORD lists is installed with the same sha256 (read back from disk)."""
    with zipfile.ZipFile(wheel) as zf:
        # a wheel has exactly one RECORD file (the unpacking fails otherwise)
        (rec,) = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
        # a RECORD line is "path,digest,size": split at the last two commas, so a comma in a path does no harm
        want = {
            line.rsplit(",", 2)[0]: line.rsplit(",", 2)[1]
            for line in zf.read(rec).decode().splitlines()
            if line.split("/")[0] == "hcd_analysis"
        }
    dist = md.distribution("hcd-priya")  # the installed distribution's name, as in pyproject.toml
    bad = []
    for path, digest in sorted(want.items()):
        data = open(dist.locate_file(path), "rb").read()
        # a RECORD digest is "sha256=" and the SHA-256 in URL-safe base64 without the "=" padding
        got = (
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        )
        if got != digest:
            bad.append(path)
    if bad or not want:
        sys.exit(
            f"the installed hcd_analysis is not {wheel}: {bad[:10] or 'empty RECORD'}; install that wheel into a "
            "fresh virtual environment and run again"
        )
    return len(want)


def slow_tests(cmd_base, cwd, env, args):
    """The tests marked slow among the tests a run selects (deselected by the runs), from one collection-only pass;
    args are the selections and the --deselect options of every file of the run. Returns (the sorted node ids, "")
    or, when the collection fails, (None, its output)."""
    proc = subprocess.run(
        cmd_base + ["--collect-only", "-q", "-m", "slow", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=FILE_TIMEOUT_S,
    )
    if proc.returncode not in (0, 5):  # 5: nothing selected (no slow test in the tier)
        return None, proc.stdout + proc.stderr
    return sorted(line.strip() for line in proc.stdout.splitlines() if "::" in line), ""


def prepare_cwd(tier, workdir):
    """The folder pytest runs from. The wheel tier: `workdir`, a new folder holding a copy of tests/ only, so that
    hcd_analysis can be imported only from the installed package; every other tier: the checkout."""
    if tier != "wheel":
        return REPO
    if workdir is None or os.path.exists(workdir):
        sys.exit("--workdir must name a directory that does not exist yet")
    os.makedirs(workdir)
    shutil.copytree(
        os.path.join(REPO, "tests"),
        os.path.join(workdir, "tests"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return os.path.realpath(workdir)


def pytest_env(extra):
    """The environment of every pytest run: this process's, without the variables that would add code or options
    from outside the run (PYTHONPATH, PYTEST_ADDOPTS, PYTEST_PLUGINS), with plugin autoloading off (no installed
    pytest plugin loads unasked), plus the tier's own variables (`extra`)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.update(extra)
    return env


def pytest_command(cwd):
    """The start of every pytest command: this interpreter, the repository's pytest settings (pyproject.toml), `cwd`
    as the root (no conftest.py above it is read), and no cache folder."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.path.join(REPO, "pyproject.toml"),
        "--rootdir",
        cwd,
        "--confcutdir",
        cwd,
        "-p",
        "no:cacheprovider",
    ]


def run_file(rel, plan, cmd_base, cwd, env, out, deferrals, deferred_xfails):
    """Run one file of the plan in its own pytest process, write its log and junit report into `out`, print one PASS
    or FAIL line, and return the file's result: the verdict and its reason, the exit status, the time, what was
    selected and left out, and the outcome counts (COUNT_KEYS)."""
    slug = rel.replace("/", "__")
    xml = os.path.join(out, slug + ".xml")
    t0 = time.time()
    try:
        marker = ["-m", plan["marker"]] if plan["marker"] else []
        # one "--deselect NODE_ID" pair per test left out
        deselect = [a for d in plan["deselect"][rel] for a in ("--deselect", d)]
        # -q: short output; -rA: a summary line for every outcome, so the log shows each skip and its reason
        proc = subprocess.run(
            cmd_base + marker + deselect + ["-q", "-rA", f"--junitxml={xml}"] + plan["select"][rel],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=FILE_TIMEOUT_S,
        )
        rc, log = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        # the output caught before the timeout: bytes whenever some was captured, even with text=True (as the
        # subprocess documentation says), or None
        partial = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        rc, log = None, f"timeout after {FILE_TIMEOUT_S} s; output so far:\n{partial}"
    with open(os.path.join(out, slug + ".log"), "w") as f:
        f.write(log)
    declared, required = plan["declared"][rel], plan["required"][rel]
    counts = None
    if rc is not None and os.path.exists(xml):
        try:
            counts = junit_counts(xml, declared, deferrals | set(deferred_xfails), required)
        except ET.ParseError:
            counts = None
    ok, why = file_verdict(rc, counts)
    result = dict(
        file=rel,
        ok=ok,
        reason=why,
        exit=rc,
        seconds=round(time.time() - t0, 1),
        declared_skips=declared,
        selected=plan["select"][rel],
        deselected=plan["deselect"][rel],
        **(counts or {}),
    )
    print(f"{'PASS' if ok else 'FAIL'} {rel} ({why}; {result['seconds']} s)", flush=True)
    return result


def print_summary(summary):
    """Print a run's totals, then what it did not run: the declared skips, the tests marked slow, the files it ran
    only in part, and the files of the other tiers and levels."""
    tier, level = summary["tier"], summary["level"]
    results, totals = summary["files"], summary["totals"]
    slow, slow_error = summary["slow_not_run"], summary["slow_listing_error"]
    print(
        f"\n{tier} tier{'' if level is None else ', level ' + level}: {len(results)} files, {len(summary['failed_files'])} failed; {totals['tests']} tests: "
        f"{totals['passed']} passed, {totals['failures']} failed, {totals['errors']} errors, "
        f"{totals['skipped']} undeclared skips, {totals['bad_xfail']} undeclared expected failures, "
        f"{totals['asset_missing']} asset tests not run in an assets job; "
        f"{totals['xfailed']} declared deferrals (expected failures), "
        f"{totals['asset_skipped']} declared skips NOT RUN (an asset absent here, or skipped by design)"
    )
    for r in results:
        if r.get("asset_skipped"):
            print(
                f"  NOT RUN here (tests/ci_tiers.toml says why): {r['file']}: {', '.join(r['declared_skips'])}"
            )
    if slow is None:
        print(f"  could not list the slow tests:\n{slow_error[-2000:]}")
    else:
        print(f"  NOT RUN here, marked slow ({len(slow)}): {', '.join(slow) or 'none'}")
    for r in results:
        if r["deselected"] or r["selected"] != [r["file"]]:
            print(
                f"  this level runs only part of {r['file']}: {' '.join(r['selected'])}; left out: {', '.join(r['deselected']) or 'none'}"
            )
    print_not_run(summary["not_run"])
    if summary["not_run_at_level"] is not None:
        print_not_run_at_level(summary["not_run_at_level"], level)


def run_tier(tier, out, workdir=None, wheel=None, level="nightly"):
    """Run the files of one tier at one level (tier_plan says which tests of which files), one pytest process per
    file; write each file's log and junit report and the run's summary.json into `out`, a new folder, and print the
    summary. Returns 1 if any file did not pass or the slow tests could not be listed, else 0."""
    # 1. What to run.
    _, files = load_tiers()
    plan = tier_plan(tier, files, level)
    todo = plan["todo"]
    if not todo:
        sys.exit(
            f"tier {tier!r} has no file at level {level!r}; see the files' levels in tests/ci_tiers.toml"
        )
    # 2. Where to run it, and a check that the package under test comes from where this tier must take it.
    cwd = prepare_cwd(tier, workdir)
    os.makedirs(out, exist_ok=False)
    origin = package_origin(sys.executable, cwd)
    check_origin(tier, origin)
    n_checked = check_installed_is(wheel) if wheel else None
    # 3. How to run it: the allowed deferral messages, a clean environment, the pytest command, and the list of the
    # selected tests that are marked slow (deselected below, and reported as not run).
    deferrals = deferred_legacy_messages()
    env = pytest_env(plan["env"])
    cmd_base = pytest_command(cwd)
    every = [a for rel in todo for a in plan["select"][rel]]
    every += [a for rel in todo for d in plan["deselect"][rel] for a in ("--deselect", d)]
    slow, slow_error = slow_tests(cmd_base, cwd, env, every) if plan["marker"] else ([], "")
    # 4. Run every file and judge it from its junit report.
    results = [
        run_file(
            rel, plan, cmd_base, cwd, env, out, deferrals, files[rel].get("deferred_xfails", [])
        )
        for rel in todo
    ]
    # 5. The summary: written to summary.json, then printed.
    totals = {k: sum(r.get(k, 0) for r in results) for k in COUNT_KEYS}
    failed = [r["file"] for r in results if not r["ok"]]
    summary = dict(
        tier=tier,
        level=plan["level"],
        python=sys.version.split()[0],
        hcd_analysis=origin,
        wheel=wheel,
        wheel_files_checked=n_checked,
        cwd=cwd,
        files=results,
        totals=totals,
        failed_files=failed,
        slow_not_run=slow,
        slow_listing_error=slow_error,
        not_run=not_run(files, ran=(ASSET_JOB,) if tier == ASSET_JOB else (), ran_files=todo),
        not_run_at_level=None if tier == ASSET_JOB else not_run_at_level(files, level),
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print_summary(summary)
    return 1 if failed or slow is None else 0


def not_run(files, ran=(), ran_files=()):
    """{tier: [(file, reason)]} for the files outside the CI tiers (and outside `ran`, the tiers this job ran, and
    `ran_files`, the files it ran: the external tier's file of the nightly level, when the nightly job ran it).
    """
    out = {}
    for path, entry in sorted(files.items()):
        if entry["tier"] in CI_TIERS or entry["tier"] in ran or path in ran_files:
            continue
        out.setdefault(entry["tier"], []).append((path, entry.get("reason", "")))
    return out


def not_run_at_level(files, level):
    """What a CI level does not run, as {"files": {file level: [(file, reason)]}, "tests": [(file, test, level)]}:
    "files" groups by their level the files the level does not run, or runs only in part (a nightly file whose
    pr_tests run at pr); "tests" lists the single tests left out of the files it runs, each with the level that runs
    it (nightly for a pr file's nightly_tests at pr; release for release_tests)."""
    groups, tests = {}, []
    for path, entry in sorted(files.items()):
        lv = entry["level"]
        if lv in RUN_LEVELS[level]:
            tests += [(path, n, "nightly") for n in entry.get("nightly_tests", []) if level == "pr"]
        elif level == "pr" and lv == "nightly" and entry.get("pr_tests"):
            groups.setdefault("nightly (only its pr_tests run at pr)", []).append(
                (path, entry.get("reason", ""))
            )
        else:
            groups.setdefault(lv, []).append((path, entry.get("reason", "")))
        tests += [(path, n, "release") for n in entry.get("release_tests", [])]
    return dict(files=groups, tests=tests)


def print_not_run_at_level(groups, level):
    """Print not_run_at_level's result. Files are counted per level; only the release and not-run files, which no CI
    level runs, are listed one by one with their reasons. Every left-out test is listed."""
    for lv, rows in groups["files"].items():
        print(f"NOT RUN at level {level}: files of level {lv!r} ({len(rows)})")
        if lv in ("release", "not run"):
            for path, reason in rows:
                print(f"  {path}: {reason}")
    if groups["tests"]:
        print(
            f"NOT RUN at level {level}: tests left out of the files it runs ({len(groups['tests'])})"
        )
        for path, name, lv in groups["tests"]:
            print(f"  {path}::{name} (level {lv})")


def print_not_run(groups):
    """Print not_run's result: per tier, every file CI does not run, with its reason."""
    for tier, rows in groups.items():
        print(f"NOT RUN in CI, tier {tier!r} ({len(rows)} files):")
        for path, reason in rows:
            print(f"  {path}: {reason}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=CI_TIERS + (EXTERNAL, ASSET_JOB))
    ap.add_argument(
        "--level",
        choices=tuple(RUN_LEVELS),
        help="a CI tier's level (default nightly: everything CI runs)",
    )
    ap.add_argument("--workdir")
    ap.add_argument("--wheel", help="wheel tier: check that the installed package is this wheel")
    ap.add_argument("--out")
    ap.add_argument("--list-not-run", action="store_true")
    args = ap.parse_args(argv)
    if args.list_not_run:
        files = load_tiers()[1]
        print_not_run(not_run(files))
        if args.level:
            print_not_run_at_level(not_run_at_level(files, args.level), args.level)
        return 0
    if args.tier is None or args.out is None:
        ap.error("--tier and --out are required")
    if args.wheel and args.tier != "wheel":
        ap.error("--wheel goes with --tier wheel")
    if args.level and args.tier == ASSET_JOB:
        ap.error("the assets job runs every test of its files: no --level")
    return run_tier(
        args.tier, os.path.realpath(args.out), args.workdir, args.wheel, args.level or "nightly"
    )


if __name__ == "__main__":
    sys.exit(main())
