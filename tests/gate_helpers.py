"""Helpers for gate runs of the emulator-debug campaign (gate A review BT8).

Set HCD_GATE_RUN=1 in a gate run: a test that needs the real caches then FAILS when they are absent instead of
skipping, so a gate can never pass by silently not running its load-bearing tests."""
import importlib.metadata as md
import os
import re

import pytest

from hcd_analysis.paths import REPO_ROOT

PROJECT_TOP = ("hcd_analysis", "scripts", "tests", "cli", "config")
PACKAGE_TOP = ("hcd_analysis", "cli")      # the top-level packages the wheel installs (pyproject.toml)
TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # the tree these tests belong to

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
    "RETIRED_STUB_SOURCE": (
        r"(?s)def run_legb\(\*args, \*\*kwargs\):.{0,40}PRE-2026-10 INTERFACE \(retired at gate E; rebuilt at gate F\)",
        "a source inspection of run_legb, which gate E retired to a NotImplementedError stub"),
}


def raise_if_deferred(exc):
    """Re-raise ``exc`` when it is a deferred_legacy cause, so a guard that skips on absent data cannot hide a
    retired interface; return otherwise (the caller then skips)."""
    if any(re.search(pattern, f"{type(exc).__name__}: {exc}") for pattern, _ in DEFERRED_CAUSES.values()):
        raise exc


def real_cache_path(which="lf"):
    """Path of a real cache inside THIS checkout (never a CWD-relative or absolute literal)."""
    return os.path.join(str(REPO_ROOT), "hcd_analysis", "_emulator_data", f"observables_tau0_{which}.h5")


# Data that lives outside the repository is located by environment variables, never by an absolute literal. An unset
# variable gives a path that does not exist, so the test's own guard skips (or fails under HCD_GATE_RUN=1) and names
# the variable. The variables, each a directory:
#   HCD_CHECKPOINTS_DIR      the production ensemble members (also read by prod_ensemble.verify_manifest)
#   HCD_EMULATOR_V2_DIR      the emulator products: gateC_eval/, mf/gateD/, gateE/products/, caches/s4_build_shards/
#   HCD_LF_RAW_DIR           the LF simulation outputs (config data_root: <sim>/output/SPECTRA_<snap>/...)
#   HCD_HR_RAW_DIR           the HR simulation outputs
#   HCD_OUTPUTS_DIR          the HCD catalogue outputs (<sim>/snap_<snap>/..., hires/...)
#   HCD_UPSTREAM_HR_DIR      the upstream PRIYA HR emulator directory (mf_emulator_flux_vectors_*.hdf5)
#   HCD_P1D_DATA_DIR         the P1D data products (desi_dr1_p1d/, eboss_dr14_p1d/)
#   HCD_HISTORICAL_CHECKOUT  the checkout holding the frozen pre-2026-10 products (checkpoints/, _emulator_data/)
#   HCD_LYAEMU_PARENT, HCD_KS_BASEDIR   as in tests/external/test_lyaemu_adapter_guards.py
def external_path(var, *parts):
    """``os.path.join($var, *parts)``; ``<var unset>/...`` (a path that never exists) when the variable is unset."""
    root = os.environ.get(var) or f"<{var} unset>"
    return os.path.join(root, *parts)


def require_real_cache(path):
    if os.path.exists(path):
        return path
    msg = f"real cache absent: {path}"
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1: gate runs require the real caches)")
    pytest.skip(msg)


def installed_root():
    """The directory hcd_analysis was imported from when an installed wheel put it there, else None (hcd_analysis was
    imported from a checkout). Installed means: the hcd_analysis distribution found on sys.path has a RECORD (only a
    pip-installed .dist-info has one), the RECORD lists hcd_analysis/paths.py, and the distribution lives in the
    directory paths.py was imported from (a venv, a user site, a --target directory). A checkout carrying a legacy
    hcd_analysis.egg-info (no RECORD) and an editable install (its RECORD lists no package module) stay checkouts, so
    the cross-checkout guard below keeps its full strength there. Tests that sit in a checkout (their tree holds the
    package) are never in installed mode, so a stale installed copy cannot excuse a checkout's mixed origins."""
    if os.path.isfile(os.path.join(TESTS_ROOT, "hcd_analysis", "__init__.py")):
        return None
    try:
        dist = md.distribution("hcd_analysis")
    except md.PackageNotFoundError:
        return None
    record = dist.read_text("RECORD") or ""
    if not any(line.split(",")[0] == "hcd_analysis/paths.py" for line in record.splitlines()):
        return None
    root = os.path.realpath(str(REPO_ROOT))
    return root if os.path.realpath(str(dist.locate_file(""))) == root else None


def foreign_project_modules(modules):
    """Project modules (top-level package in PROJECT_TOP) whose file lies outside the code under test.

    From a checkout, every project module must come from the tree these tests belong to. Against an installed wheel
    (hcd_analysis imported from this interpreter's site-packages), the installed packages (PACKAGE_TOP) must come from
    that site-packages and the repository modules (tests, scripts, config) from the tests' tree. Both sides of each
    comparison are resolved, so a symlinked path (a venv's lib64, a linked checkout) is neither foreign nor hidden."""
    site = installed_root()
    foreign = []
    for name, mod in list(modules.items()):
        top = name.split(".")[0]
        if top not in PROJECT_TOP:
            continue
        f = getattr(mod, "__file__", None)
        root = site if site is not None and top in PACKAGE_TOP else TESTS_ROOT
        if f and not os.path.realpath(f).startswith(root + os.sep):
            foreign.append(f"{name}: {f}")
    return foreign
