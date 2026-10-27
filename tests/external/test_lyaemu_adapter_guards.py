"""OPTIONAL external test of the lyaemu reference adapter's guards, on a COPY of a reference basedir.

Needs the reference's own interpreter (GPy, emukit, lyaemu) and two paths:
  HCD_KS_BASEDIR=<basedir>  HCD_LYAEMU_PARENT=<directory holding lyaemu/>
Run it there without the suite's jax conftest:
  <ref-venv>/bin/python -B -m pytest --noconftest -p no:cacheprovider tests/external/test_lyaemu_adapter_guards.py
Skipped otherwise (a skip is not a pass). The masters are never touched: every test works on a tmp copy."""
import json
import os
import shutil
import sys

import pytest

sys.dont_write_bytecode = True                  # never write bytecode into the reference package

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


def test_nested_class_strings_are_checked(copy_basedir):
    td = os.path.join(copy_basedir, "trained_mf")
    f = os.path.join(td, "zbin3.0.json")
    with open(f) as fh:
        d = json.load(fh)
    d["kernel"] = {"parts": [{"class": "os.system", "x": 1}]}
    with open(f, "w") as fh:
        json.dump(d, fh)
    with pytest.raises(A.ReferenceRefused, match="allowlist"):
        A.check_class_strings(td)


def _true_pins(basedir):
    arts = ("emulator_params.json", "mf_emulator_flux_vectors_tau1000000.hdf5")
    return {"artifacts_sha256": {a: A.sha256_file(os.path.join(basedir, a)) for a in arts},
            "trained_mf_tree_sha256": A.tree_sha256(os.path.join(basedir, "trained_mf")),
            "implementation": {"lyaemu_code_md5": A.lyaemu_md5(os.path.join(LYA, "lyaemu"))}}


def test_pin_mismatches_are_refused(copy_basedir):
    pkg = os.path.join(LYA, "lyaemu")
    pins = _true_pins(copy_basedir)
    A.check_pins(copy_basedir, pkg, pins)                                   # the true pins pass
    for key, bad, match in (("artifacts_sha256", {"emulator_params.json": "0" * 64}, "sha256"),
                            ("trained_mf_tree_sha256", "0" * 64, "trained_mf"),
                            ("implementation", {"lyaemu_code_md5": "0" * 32}, "md5")):
        with pytest.raises(A.ReferenceRefused, match=match):
            A.check_pins(copy_basedir, pkg, dict(pins, **{key: bad}))
    with open(os.path.join(copy_basedir, "trained_mf", "zbin3.0.json"), "a") as fh:
        fh.write(" ")
    with pytest.raises(A.ReferenceRefused, match="trained_mf"):
        A.check_pins(copy_basedir, pkg, pins)


def test_symlinks_in_a_reference_tree_are_refused(copy_basedir, tmp_path):
    os.symlink(str(tmp_path), os.path.join(copy_basedir, "link"))
    with pytest.raises(A.ReferenceRefused, match="symlink"):
        A.refuse_symlinks(copy_basedir)


def test_a_missing_trained_file_is_refused_never_refit(copy_basedir):
    """The adapter must refuse any fit or save while loading a reference whose trained file is missing, and nothing
    may be written back into the trained directory."""
    td = os.path.join(copy_basedir, "trained_mf")
    os.remove(os.path.join(td, "zbin3.0"))
    before = sorted(os.listdir(td))
    A.install_refit_refusal()
    with pytest.raises(A.ReferenceRefused, match="fit or save"):
        A.load_reference(copy_basedir, 13)
    assert sorted(os.listdir(td)) == before


def test_compiled_bytecode_in_the_reference_package_is_refused(tmp_path):
    pkg = tmp_path / "parent" / "lyaemu"
    shutil.copytree(os.path.join(LYA, "lyaemu"), pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for dp, ds, fs in os.walk(pkg):                         # the tmp copy must be writable (the source is read-only)
        os.chmod(dp, 0o755)
    A.refuse_compiled(str(pkg))
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "gp_wrap.cpython-311.pyc").write_bytes(b"\0")
    with pytest.raises(A.ReferenceRefused, match="bytecode"):
        A.refuse_compiled(str(pkg))



def test_extension_modules_in_the_reference_package_are_refused(tmp_path):
    import importlib.machinery
    pkg = tmp_path / "parent2" / "lyaemu"
    shutil.copytree(os.path.join(LYA, "lyaemu"), pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for dp, ds, fs in os.walk(pkg):
        os.chmod(dp, 0o755)
    (pkg / ("gp_wrap" + importlib.machinery.EXTENSION_SUFFIXES[0])).write_bytes(b"\0")
    with pytest.raises(A.ReferenceRefused, match="extension module"):
        A.refuse_compiled(str(pkg))
