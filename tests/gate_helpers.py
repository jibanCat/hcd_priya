"""Helpers for gate runs of the emulator-debug campaign (gate A review BT8).

Set HCD_GATE_RUN=1 in a gate run: a test that needs the real caches then FAILS when they are absent instead of
skipping, so a gate can never pass by silently not running its load-bearing tests."""
import os

import pytest

from hcd_analysis.paths import REPO_ROOT

PROJECT_TOP = ("hcd_analysis", "scripts", "tests", "cli", "config")


def real_cache_path(which="lf"):
    """Path of a real cache inside THIS checkout (never a CWD-relative or absolute literal)."""
    return os.path.join(str(REPO_ROOT), "hcd_analysis", "_emulator_data", f"observables_tau0_{which}.h5")


def require_real_cache(path):
    if os.path.exists(path):
        return path
    msg = f"real cache absent: {path}"
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1: gate runs require the real caches)")
    pytest.skip(msg)


def foreign_project_modules(modules):
    """Project modules (top-level package in PROJECT_TOP) whose file lies outside this checkout."""
    root = str(REPO_ROOT)
    foreign = []
    for name, mod in list(modules.items()):
        if name.split(".")[0] not in PROJECT_TOP:
            continue
        f = getattr(mod, "__file__", None)
        if f and not os.path.abspath(f).startswith(root + os.sep):
            foreign.append(f"{name}: {f}")
    return foreign
