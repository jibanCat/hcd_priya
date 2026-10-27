"""Run one CI test tier of tests/ci_tiers.toml, one pytest process per file, and refuse any undeclared skip.

usage:
  python tools/ci/run_test_tier.py --tier wheel  --workdir NEW_DIR --out NEW_DIR [--wheel WHEEL]
  python tools/ci/run_test_tier.py --tier source --out NEW_DIR
  python tools/ci/run_test_tier.py --tier assets --out NEW_DIR
  python tools/ci/run_test_tier.py --list-not-run

Run it with the interpreter under test (CI: the fresh venv holding the installed wheel).

- wheel tier: tests/ is copied into NEW_DIR, and pytest runs from there, so hcd_analysis can only come from the
  interpreter's site-packages; the runner checks that before running anything, and with --wheel also that the
  installed files are exactly that wheel's (the RECORD entries of its package files).
- source tier: pytest runs from the repository checkout this script belongs to, so hcd_analysis comes from it.
- assets: a job meant to test the real model, never run by the CI workflow (it has no assets). From the checkout,
  it runs the files of the "assets" tier and every wheel or source file that declares asset_tests, with
  HCD_GATE_RUN=1 (tests/gate_helpers.py: a missing real cache fails instead of skipping). There a declared asset
  test MUST run: if it skips, its asset is missing and the file fails.

A file passes only if pytest exits 0 and its junit report has no failure, no error and no undeclared skip, and at least
one test passed. Besides passes, only these outcomes are allowed; each is counted and listed, never as a pass:
- an expected failure whose message is exactly that of an explicit deferral: a deferred_legacy cause of
  tests/gate_helpers.py DEFERRED_CAUSES (the exact-cause marker of tests/conftest.py, tests/DEFERRED.md), or a reason
  that tests/ci_tiers.toml lists in the file's deferred_xfails (a strict, exact-exception xfail);
- a skip of a test that tests/ci_tiers.toml names in the file's asset_tests (it needs a real asset that CI does not
  have) or skipped_by_design (skipped unconditionally until a later step). Such a test must skip here: if it passes
  or fails instead, the file fails, so the lists stay exact.
Tests marked slow are deselected (-m "not slow") and listed as NOT RUN. The exit status is 1 if any file did not
pass. The summary also lists the files of the tiers that CI does not run, with their reasons.
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TIERS_FILE = os.path.join(REPO, "tests", "ci_tiers.toml")
CI_TIERS = ("wheel", "source")
ASSET_JOB = "assets"
FILE_TIMEOUT_S = 1800
COUNT_KEYS = ("tests", "passed", "failures", "errors", "skipped", "xfailed", "bad_xfail", "asset_skipped",
              "asset_ran", "asset_missing")


def load_tiers(path=TIERS_FILE):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    return doc["tiers"], doc["files"]


def files_of(files, tier):
    return sorted(p for p, entry in files.items() if entry["tier"] == tier)


def asset_job_files(files):
    """The files an assets job runs: the "assets" tier, and every CI-tier file that declares asset_tests."""
    return sorted(p for p, entry in files.items()
                  if entry["tier"] == ASSET_JOB or (entry["tier"] in CI_TIERS and entry.get("asset_tests")))


def deferred_legacy_messages(path=os.path.join(REPO, "tests", "gate_helpers.py")):
    """The exact junit messages of the deferred_legacy causes, as tests/conftest.py writes them, read from the
    DEFERRED_CAUSES literal without importing the tests."""
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFERRED_CAUSES" for t in node.targets):
            return {f"deferred_legacy {cause}: {reason}" for cause, (_, reason) in ast.literal_eval(node.value).items()}
    raise SystemExit(f"no DEFERRED_CAUSES in {path}")


def junit_counts(xml_path, declared_skips=(), deferrals=(), required=()):
    """Outcome counts of a pytest junit report (see the module docstring). declared_skips are the file's asset_tests
    and skipped_by_design together; a test matches one by its full name (with parameters, e.g. "test_x[a]") or by
    its function name (all parameters). deferrals are the exact expected-failure messages allowed. required (an
    assets job: the file's asset_tests) are tests that must run here: one that skips counts as asset_missing."""
    declared, allowed, must = set(declared_skips), set(deferrals), set(required)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        counts["tests"] += 1
        name = case.get("name", "")
        is_declared = name in declared or name.split("[")[0] in declared
        is_required = name in must or name.split("[")[0] in must
        sk = case.find("skipped")
        if case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif sk is not None and sk.get("type") == "pytest.xfail":
            counts["xfailed" if sk.get("message") in allowed else "bad_xfail"] += 1
        elif sk is not None and is_required:
            counts["asset_missing"] += 1
        elif sk is not None:
            counts["asset_skipped" if is_declared else "skipped"] += 1
        else:
            counts["passed"] += 1
        if is_declared and not is_required and sk is None:
            counts["asset_ran"] += 1
    return counts


def file_verdict(returncode, counts):
    """(passed, reason) for one file's pytest run."""
    if returncode is None:
        return False, f"timeout after {FILE_TIMEOUT_S} s"
    if counts is None:
        return False, f"no readable junit report (exit {returncode})"
    for key, what in (("failures", "failed"), ("errors", "errors"), ("skipped", "undeclared skips"),
                      ("bad_xfail", "expected failures that are not declared deferrals"),
                      ("asset_ran", "declared skips that did not skip"),
                      ("asset_missing", "asset tests skipped in a job that must have their assets")):
        if counts[key]:
            return False, f"{counts[key]} {what}"
    if counts["passed"] == 0:
        return False, "no test passed"
    if returncode != 0:
        return False, f"pytest exit {returncode}"
    return True, "ok"


def package_origin(python, cwd):
    out = subprocess.run([python, "-c", "import hcd_analysis; print(hcd_analysis.__file__)"], cwd=cwd,
                         capture_output=True, text=True, check=True)
    return os.path.realpath(out.stdout.strip())


def check_origin(tier, origin):
    pkg_parent = os.path.dirname(os.path.dirname(origin))
    if tier == "wheel":
        site = {os.path.realpath(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")}
        if pkg_parent not in site:
            sys.exit(f"wheel tier: hcd_analysis imports from {origin}, not from this interpreter's site-packages")
    elif pkg_parent != REPO:
        sys.exit(f"source tier: hcd_analysis imports from {origin}, not from the checkout {REPO}")


def check_installed_is(wheel):
    """Every package file the wheel's RECORD lists is installed with the same sha256 (read back from disk)."""
    with zipfile.ZipFile(wheel) as zf:
        (rec,) = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
        want = {line.rsplit(",", 2)[0]: line.rsplit(",", 2)[1] for line in zf.read(rec).decode().splitlines()
                if line.split("/")[0] == "hcd_analysis"}
    dist = md.distribution("hcd-priya")
    bad = []
    for path, digest in sorted(want.items()):
        data = open(dist.locate_file(path), "rb").read()
        got = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if got != digest:
            bad.append(path)
    if bad or not want:
        sys.exit(f"the installed hcd_analysis is not {wheel}: {bad[:10] or 'empty RECORD'}")
    return len(want)


def slow_tests(cmd_base, cwd, env, todo):
    """The tests marked slow in the tier's files (deselected by the runs), from one collection-only pass."""
    proc = subprocess.run(cmd_base + ["--collect-only", "-q", "-m", "slow", *todo], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=FILE_TIMEOUT_S)
    if proc.returncode not in (0, 5):                 # 5: nothing selected (no slow test in the tier)
        return None, proc.stdout + proc.stderr
    return sorted(line.strip() for line in proc.stdout.splitlines() if "::" in line), ""


def run_tier(tier, out, workdir=None, wheel=None):
    tiers, files = load_tiers()
    todo = asset_job_files(files) if tier == ASSET_JOB else files_of(files, tier)
    if not todo:
        sys.exit(f"tier {tier!r} has no file")
    if tier == "wheel":
        if workdir is None or os.path.exists(workdir):
            sys.exit("--workdir must name a directory that does not exist yet")
        os.makedirs(workdir)
        shutil.copytree(os.path.join(REPO, "tests"), os.path.join(workdir, "tests"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        cwd = os.path.realpath(workdir)
    else:
        cwd = REPO
    os.makedirs(out, exist_ok=False)
    origin = package_origin(sys.executable, cwd)
    check_origin(tier, origin)
    n_checked = check_installed_is(wheel) if wheel else None
    deferrals = deferred_legacy_messages()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if tier == ASSET_JOB:
        env["HCD_GATE_RUN"] = "1"
    cmd_base = [sys.executable, "-m", "pytest", "-c", os.path.join(REPO, "pyproject.toml"), "--rootdir", cwd,
                "--confcutdir", cwd, "-p", "no:cacheprovider"]
    slow, slow_error = slow_tests(cmd_base, cwd, env, todo)
    results = []
    for rel in todo:
        slug = rel.replace("/", "__")
        xml = os.path.join(out, slug + ".xml")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd_base + ["-m", "not slow", "-q", "-rA", f"--junitxml={xml}", rel], cwd=cwd,
                                  env=env, capture_output=True, text=True, timeout=FILE_TIMEOUT_S)
            rc, log = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            rc, log = None, f"timeout after {FILE_TIMEOUT_S} s; output so far:\n{partial}"
        with open(os.path.join(out, slug + ".log"), "w") as f:
            f.write(log)
        declared = files[rel].get("asset_tests", []) + files[rel].get("skipped_by_design", [])
        required = files[rel].get("asset_tests", []) if tier == ASSET_JOB else []
        counts = None
        if rc is not None and os.path.exists(xml):
            try:
                counts = junit_counts(xml, declared, deferrals | set(files[rel].get("deferred_xfails", [])), required)
            except ET.ParseError:
                counts = None
        ok, why = file_verdict(rc, counts)
        results.append(dict(file=rel, ok=ok, reason=why, exit=rc, seconds=round(time.time() - t0, 1),
                            declared_skips=declared, **(counts or {})))
        print(f"{'PASS' if ok else 'FAIL'} {rel} ({why}; {results[-1]['seconds']} s)", flush=True)
    totals = {k: sum(r.get(k, 0) for r in results) for k in COUNT_KEYS}
    failed = [r["file"] for r in results if not r["ok"]]
    summary = dict(tier=tier, python=sys.version.split()[0], hcd_analysis=origin, wheel=wheel,
                   wheel_files_checked=n_checked, cwd=cwd, files=results, totals=totals, failed_files=failed,
                   slow_not_run=slow, slow_listing_error=slow_error,
                   not_run=not_run(files, ran=(ASSET_JOB,) if tier == ASSET_JOB else ()))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{tier} tier: {len(results)} files, {len(failed)} failed; {totals['tests']} tests: "
          f"{totals['passed']} passed, {totals['failures']} failed, {totals['errors']} errors, "
          f"{totals['skipped']} undeclared skips, {totals['bad_xfail']} undeclared expected failures, "
          f"{totals['asset_missing']} asset tests skipped in an assets job; "
          f"{totals['xfailed']} declared deferrals (expected failures), "
          f"{totals['asset_skipped']} declared skips NOT RUN (an asset absent here, or skipped by design)")
    for r in results:
        if r.get("asset_skipped"):
            print(f"  NOT RUN here (tests/ci_tiers.toml says why): {r['file']}: {', '.join(r['declared_skips'])}")
    if slow is None:
        print(f"  could not list the slow tests:\n{slow_error[-2000:]}")
    else:
        print(f"  NOT RUN here, marked slow ({len(slow)}): {', '.join(slow) or 'none'}")
    print_not_run(summary["not_run"])
    return 1 if failed or slow is None else 0


def not_run(files, ran=()):
    """{tier: [(file, reason)]} for the files outside the CI tiers (and outside `ran`, the tiers this job ran)."""
    out = {}
    for path, entry in sorted(files.items()):
        if entry["tier"] in CI_TIERS or entry["tier"] in ran:
            continue
        out.setdefault(entry["tier"], []).append((path, entry.get("reason", "")))
    return out


def print_not_run(groups):
    for tier, rows in groups.items():
        print(f"NOT RUN in CI, tier {tier!r} ({len(rows)} files):")
        for path, reason in rows:
            print(f"  {path}: {reason}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=CI_TIERS + (ASSET_JOB,))
    ap.add_argument("--workdir")
    ap.add_argument("--wheel", help="wheel tier: check that the installed package is this wheel")
    ap.add_argument("--out")
    ap.add_argument("--list-not-run", action="store_true")
    args = ap.parse_args(argv)
    if args.list_not_run:
        print_not_run(not_run(load_tiers()[1]))
        return 0
    if args.tier is None or args.out is None:
        ap.error("--tier and --out are required")
    if args.wheel and args.tier != "wheel":
        ap.error("--wheel goes with --tier wheel")
    return run_tier(args.tier, os.path.realpath(args.out), args.workdir, args.wheel)


if __name__ == "__main__":
    sys.exit(main())
