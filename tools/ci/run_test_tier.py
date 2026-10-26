"""Run one CI test tier of tests/ci_tiers.toml, one pytest process per file, and refuse any skip.

usage:
  python tools/ci/run_test_tier.py --tier wheel  --workdir NEW_DIR --out DIR
  python tools/ci/run_test_tier.py --tier source --out DIR
  python tools/ci/run_test_tier.py --list-not-run

Run it with the interpreter under test (CI: the fresh venv holding the installed wheel).

- wheel tier: tests/ is copied into NEW_DIR (which must not exist), and pytest runs from there, so hcd_analysis can
  only come from the interpreter's site-packages; the runner checks that before running anything.
- source tier: pytest runs from the repository checkout this script belongs to, so hcd_analysis comes from it.

A file passes only if pytest exits 0 and its junit report has no failure, no error and no skip, and at least one test
ran. Two outcomes are allowed besides a pass, and both are counted and listed, never as passes:
- an expected failure of an existing deferred_legacy deferral (tests/DEFERRED.md, the exact-cause marker of
  tests/conftest.py); any other expected failure fails the file;
- a skip of a test that tests/ci_tiers.toml names in the file's asset_tests: it needs a real asset (a cache, a trained
  member, observational data, or the separately installed fake_spectra) that CI does not have, or that the file's
  skipped_by_design names (a test skipped unconditionally until a later step), and is reported as NOT RUN. Such a test
  must skip here: if it passes or fails instead, the file fails, so the lists stay exact.
Every file runs (-m "not slow"), and the exit status is 1 if any file did not pass. The summary also lists the files
of the tiers that CI does not run, with their reasons.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import tomllib
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TIERS_FILE = os.path.join(REPO, "tests", "ci_tiers.toml")
CI_TIERS = ("wheel", "source")
FILE_TIMEOUT_S = 1800


def load_tiers(path=TIERS_FILE):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    return doc["tiers"], doc["files"]


def files_of(files, tier):
    return sorted(p for p, entry in files.items() if entry["tier"] == tier)


def junit_counts(xml_path, asset_tests=()):
    """Outcome counts of a pytest junit report (see the module docstring). asset_tests are the file's declared skips
    (its asset_tests and skipped_by_design together); a test matches one by its full name (with parameters, e.g.
    "test_x[a]") or by its function name (all parameters)."""
    asset = set(asset_tests)
    counts = dict(tests=0, passed=0, failures=0, errors=0, skipped=0, xfailed=0, bad_xfail=0, asset_skipped=0,
                  asset_ran=0)
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        counts["tests"] += 1
        name = case.get("name", "")
        is_asset = name in asset or name.split("[")[0] in asset
        sk = case.find("skipped")
        if case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif sk is not None and sk.get("type") == "pytest.xfail":
            counts["xfailed" if (sk.get("message") or "").startswith("deferred_legacy ") else "bad_xfail"] += 1
        elif sk is not None:
            counts["asset_skipped" if is_asset else "skipped"] += 1
        else:
            counts["passed"] += 1
        if is_asset and sk is None:
            counts["asset_ran"] += 1
    return counts


def file_verdict(returncode, counts):
    """(passed, reason) for one file's pytest run."""
    if returncode is None:
        return False, f"timeout after {FILE_TIMEOUT_S} s"
    if counts is None:
        return False, f"no junit report (exit {returncode})"
    for key, what in (("failures", "failed"), ("errors", "errors"), ("skipped", "undeclared skips"),
                      ("bad_xfail", "expected failures that are not deferred_legacy deferrals"),
                      ("asset_ran", "declared asset tests that did not skip")):
        if counts[key]:
            return False, f"{counts[key]} {what}"
    if counts["tests"] - counts["asset_skipped"] == 0:
        return False, "no test ran"
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


def run_tier(tier, out, workdir=None):
    tiers, files = load_tiers()
    todo = files_of(files, tier)
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
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    results = []
    for rel in todo:
        slug = rel.replace("/", "__")
        xml = os.path.join(out, slug + ".xml")
        cmd = [sys.executable, "-m", "pytest", "-c", os.path.join(REPO, "pyproject.toml"), "--rootdir", cwd,
               "--confcutdir", cwd, "-p", "no:cacheprovider", "-m", "not slow", "-q", "-rA", f"--junitxml={xml}", rel]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=FILE_TIMEOUT_S)
            rc, log = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            rc, log = None, f"timeout after {FILE_TIMEOUT_S} s\n"
        with open(os.path.join(out, slug + ".log"), "w") as f:
            f.write(log)
        asset_tests = files[rel].get("asset_tests", []) + files[rel].get("skipped_by_design", [])
        counts = junit_counts(xml, asset_tests) if rc is not None and os.path.exists(xml) else None
        ok, why = file_verdict(rc, counts)
        results.append(dict(file=rel, ok=ok, reason=why, exit=rc, seconds=round(time.time() - t0, 1),
                            asset_tests=asset_tests, **(counts or {})))
        print(f"{'PASS' if ok else 'FAIL'} {rel} ({why}; {results[-1]['seconds']} s)", flush=True)
    keys = ("tests", "passed", "failures", "errors", "skipped", "xfailed", "bad_xfail", "asset_skipped", "asset_ran")
    totals = {k: sum(r.get(k, 0) for r in results) for k in keys}
    summary = dict(tier=tier, python=sys.version.split()[0], hcd_analysis=origin, cwd=cwd, files=results,
                   totals=totals, failed_files=[r["file"] for r in results if not r["ok"]],
                   not_run=not_run(files))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{tier} tier: {len(results)} files, {len(summary['failed_files'])} failed; {totals['tests']} tests: "
          f"{totals['passed']} passed, {totals['failures']} failed, {totals['errors']} errors, "
          f"{totals['skipped']} undeclared skips, {totals['xfailed']} expected failures (deferred_legacy), "
          f"{totals['asset_skipped']} declared skips NOT RUN (an asset absent here, or skipped by design)")
    for r in results:
        if r.get("asset_skipped"):
            print(f"  NOT RUN here (tests/ci_tiers.toml says why): {r['file']}: {', '.join(r['asset_tests'])}")
    print_not_run(summary["not_run"])
    return 1 if summary["failed_files"] else 0


def not_run(files):
    """{tier: [(file, reason)]} for the files outside the CI tiers."""
    out = {}
    for path, entry in sorted(files.items()):
        if entry["tier"] in CI_TIERS:
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
    ap.add_argument("--tier", choices=CI_TIERS)
    ap.add_argument("--workdir")
    ap.add_argument("--out")
    ap.add_argument("--list-not-run", action="store_true")
    args = ap.parse_args(argv)
    if args.list_not_run:
        print_not_run(not_run(load_tiers()[1]))
        return 0
    if args.tier is None or args.out is None:
        ap.error("--tier and --out are required")
    return run_tier(args.tier, os.path.realpath(args.out), args.workdir)


if __name__ == "__main__":
    sys.exit(main())
