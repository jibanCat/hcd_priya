"""OPTIONAL external test of the lyaemu reference adapter's guards, on a COPY of a reference basedir.

Needs the reference's own interpreter (GPy, emukit, lyaemu) and two paths:
  HCD_KS_BASEDIR=<basedir>  HCD_LYAEMU_PARENT=<directory holding lyaemu/>
Run it there without the suite's jax conftest:
  <ref-venv>/bin/python -m pytest --noconftest -p no:cacheprovider tests/external/test_lyaemu_adapter_guards.py
Skipped otherwise (a skip is not a pass). The masters are never touched: every test works on a tmp copy."""
import json
import os
import shutil
import sys

import pytest

BASEDIR = os.environ.get("HCD_KS_BASEDIR")
LYA = os.environ.get("HCD_LYAEMU_PARENT")
pytest.importorskip("GPy", reason="reference interpreter (GPy) not active")
if not (BASEDIR and LYA and os.path.isdir(BASEDIR) and os.path.isdir(LYA)):
    pytest.skip("HCD_KS_BASEDIR / HCD_LYAEMU_PARENT not set", allow_module_level=True)
sys.path.insert(0, LYA)

from hcd_analysis.comparison.adapters import lyaemu_priya as A  # noqa: E402


@pytest.fixture
def copy_basedir(tmp_path):
    dst = tmp_path / "basedir"
    shutil.copytree(BASEDIR, dst, ignore=shutil.ignore_patterns(".git", "*.pyc", "__pycache__"))
    for dp, ds, fs in os.walk(dst):                        # the copy must be writable so the test can tamper with it
        os.chmod(dp, 0o755)
        for f in fs:
            os.chmod(os.path.join(dp, f), 0o644)
    return str(dst)


def test_class_strings_outside_the_allowlist_are_refused(copy_basedir):
    td = os.path.join(copy_basedir, "trained_mf")
    A.check_class_strings(td)
    f = os.path.join(td, "zbin3.0.json")
    with open(f) as fh:
        d = json.load(fh)
    d["class"] = "__import__('os').system"
    with open(f, "w") as fh:
        json.dump(d, fh)
    with pytest.raises(A.ReferenceRefused, match="allowlist"):
        A.check_class_strings(td)


def test_pin_mismatch_is_refused(copy_basedir):
    pins = {"artifacts_sha256": {"emulator_params.json": "0" * 64}, "trained_mf_tree_sha256": "",
            "implementation": {"lyaemu_code_md5": ""}}
    with pytest.raises(A.ReferenceRefused, match="sha256"):
        A.check_pins(copy_basedir, os.path.join(LYA, "lyaemu"), pins)


def test_a_missing_trained_file_is_refused_never_refit(copy_basedir):
    """lyaemu catches the load failure and would OPTIMIZE and SAVE; the adapter's refusal must stop it, and nothing
    may be written back into the trained directory."""
    td = os.path.join(copy_basedir, "trained_mf")
    os.remove(os.path.join(td, "zbin3.0"))
    before = sorted(os.listdir(td))
    A.install_refit_refusal()
    with pytest.raises(A.ReferenceRefused, match="fit or save"):
        A.load_reference(copy_basedir, 13)
    assert sorted(os.listdir(td)) == before
