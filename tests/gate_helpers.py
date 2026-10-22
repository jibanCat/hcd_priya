"""Helpers for gate runs of the emulator-debug campaign (gate A review BT8).

Set HCD_GATE_RUN=1 in a gate run: a test that needs the real caches then FAILS when they are absent instead of
skipping, so a gate can never pass by silently not running its load-bearing tests."""
import os
import re

import pytest

from hcd_analysis.paths import REPO_ROOT

PROJECT_TOP = ("hcd_analysis", "scripts", "tests", "cli", "config")

# deferred_legacy causes (tests/conftest.py, tests/DEFERRED.md): cause -> (exact error pattern, reason).
DEFERRED_CAUSES = {
    "CTX_SIGNATURE": (
        r"build_legb_ctx\(\) (got an unexpected keyword argument '(use_xclass|with_mf|ckpt)'"
        r"|missing 1 required keyword-only argument: 'ensemble_ckpts')",
        "the pre-2026-10 build_legb_ctx signature (use_xclass / with_mf / a single ckpt) is retired at gate E; the "
        "gate E context takes ensemble_ckpts"),
    "REAL_FIT_CKPT_DIR": (
        r"MISSING pinned production ensemble file: \S*/checkpoints/prod_repaired_seed\d\.eqx",
        "scripts/run_real_fit.build_real_ctx verifies the ensemble against <repo>/checkpoints (the schema-1 layout); "
        "the schema-2 members live in the manifest's checkpoints_dir"),
    "SCHEMA1_CHECKPOINT": (
        r"checkpoint \S+ has schema None, expected 2\.0: pre-2026-10 checkpoints carry a single velocity grid",
        "the test loads a pre-2026-10 (schema-1) checkpoint, which train.load_checkpoint refuses by design"),
    "RETIRED_BUILDER": (
        r"\w+: the pre-2026-10 mock/truth builder is retired; rebuilt at gate F",
        "the pre-2026-10 mock/truth builder is retired at gate E (NotImplementedError stub)"),
}


def raise_if_deferred(exc):
    """Re-raise ``exc`` when it is a deferred_legacy cause, so a guard that skips on absent data cannot hide a
    retired interface; return otherwise (the caller then skips)."""
    if any(re.search(pattern, f"{type(exc).__name__}: {exc}") for pattern, _ in DEFERRED_CAUSES.values()):
        raise exc


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
