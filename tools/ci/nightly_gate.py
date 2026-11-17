"""Refuse a release unless the latest nightly run on its commit passed (the release validation and publishing workflows).

usage: python tools/ci/nightly_gate.py --repo OWNER/NAME --sha COMMIT [--workflow nightly.yml]

It reads the runs of the nightly workflow on that commit from the GitHub Actions API and passes only if the most
recent of them (by its start time) is completed with conclusion "success". A newer run that is still going, a
failure, a cancellation, no run at all, or any error reading the API is a refusal: nightly failures block a release.
With the environment variable GH_TOKEN set, the request carries it (a workflow's own token with read access to
Actions); without it, the request is anonymous, which works for a public repository only. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

API = "https://api.github.com"


def decide(runs, sha):
    """(passed, message) for the runs of the nightly workflow (the API's workflow_runs list) on commit `sha`."""
    mine = [r for r in runs if r.get("head_sha") == sha]
    if not mine:
        return False, f"no nightly run on {sha}: start the nightly workflow on this commit first"
    latest = max(
        mine, key=lambda r: (r.get("run_started_at") or r.get("created_at") or "", r.get("id", 0))
    )
    where = f"run {latest.get('id')} ({latest.get('html_url', 'no url')})"
    if latest.get("status") != "completed":
        return (
            False,
            f"the latest nightly run on {sha} is not finished ({latest.get('status')}): {where}",
        )
    if latest.get("conclusion") != "success":
        return (
            False,
            f"the latest nightly run on {sha} did not pass ({latest.get('conclusion')}): {where}",
        )
    return True, f"the latest nightly run on {sha} passed: {where}"


def fetch_runs(repo, workflow, sha, token=None):
    """The runs of `workflow` on `sha`, read from the Actions API (one page of up to 100, newest first)."""
    url = f"{API}/repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}&per_page=100"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
        return json.load(resp)["workflow_runs"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--workflow", default="nightly.yml")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        print(f"REFUSED: {args.sha!r} is not a full commit sha")
        return 1
    try:
        runs = fetch_runs(args.repo, args.workflow, args.sha, os.environ.get("GH_TOKEN") or None)
    except Exception as exc:  # noqa: BLE001  any failure to read the runs is a refusal
        print(f"REFUSED: the nightly runs could not be read ({type(exc).__name__}: {exc})")
        return 1
    passed, message = decide(runs, args.sha)
    print(("PASSED: " if passed else "REFUSED: ") + message)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
