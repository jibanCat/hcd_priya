"""TDD suite for the deterministic NUTS seed derivation (P0, plan-to-unblind section 3A).

THE BUG THIS FIXES. `scripts/run_real_fit.py:241` derived the per-survey NUTS key with
`jax.random.fold_in(key0, hash(survey) & 0x7fffffff)`. Python's builtin `hash()` on a str is
SipHash-salted PER PROCESS, so a recorded seed does NOT reproduce the chain: re-running the
same command in a fresh interpreter draws a different stream. `scripts/run_joint_fit.py:149`
already does the correct `zlib.crc32(...)` and its comment names run_real_fit as the pending
fix. This is the single hard blocker on every blind fit, and it must be the LAST
forward-affecting change so `forward_signature` freezes cleanly afterwards.

Contract under test:
  * `nuts_fold_int` is exactly `zlib.crc32(label.encode()) & 0x7fffffff`
  * it is STABLE across interpreter processes started with different PYTHONHASHSEED values
  * the test harness can actually DETECT instability (the builtin-hash control), so the
    stability test above is not vacuous
  * distinct labels give distinct folds, so two legs never share a chain stream
  * it accepts the joint driver's '+'-joined multi-leg label unchanged, so both drivers can
    share one implementation without either changing its stream
"""
import importlib.util
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEEDING = ROOT / "hcd_analysis" / "emulator" / "seeding.py"


def _load_direct():
    """Import seeding.py BY PATH, bypassing the package __init__ (which pulls jax). Keeps the
    subprocess control below fast and proves the helper is pure-python with no jax dependency."""
    spec = importlib.util.spec_from_file_location("_seeding_direct", SEEDING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _in_subprocess(expr, hashseed):
    """Evaluate `expr` in a FRESH interpreter at a given PYTHONHASHSEED and return stdout."""
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    code = (f"import importlib.util;"
            f"spec=importlib.util.spec_from_file_location('m', r'{SEEDING}');"
            f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            f"print({expr})")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# ---------------------------------------------------------------- the derivation itself

def test_nuts_fold_int_is_crc32_masked():
    m = _load_direct()
    for label in ("eboss", "desi", "ks", "DESI+KS"):
        assert m.nuts_fold_int(label) == zlib.crc32(label.encode()) & 0x7FFFFFFF


def test_nuts_fold_int_fits_the_fold_in_domain():
    """fold_in takes a non-negative int32; the mask must keep it in range for every label."""
    m = _load_direct()
    for label in ("eboss", "desi", "ks", "DESI+KS", "a" * 512):
        v = m.nuts_fold_int(label)
        assert 0 <= v <= 0x7FFFFFFF


def test_distinct_surveys_get_distinct_folds():
    """Two legs must never share a chain stream."""
    m = _load_direct()
    folds = {s: m.nuts_fold_int(s) for s in ("eboss", "desi", "ks")}
    assert len(set(folds.values())) == 3, folds


def test_joint_multileg_label_is_accepted_unchanged():
    """run_joint_fit folds on '+'.join(leg_names). Sharing one helper must not change the
    joint driver's existing stream, so the helper must reproduce crc32 of that exact label."""
    m = _load_direct()
    label = "+".join(["DESI", "KS"])
    assert m.nuts_fold_int(label) == zlib.crc32(b"DESI+KS") & 0x7FFFFFFF


# ---------------------------------------------------------------- cross-process stability

def test_fold_is_stable_across_pythonhashseed_values():
    """The whole point: a recorded seed must reproduce the chain in a fresh interpreter."""
    a = _in_subprocess("m.nuts_fold_int('eboss')", hashseed=0)
    b = _in_subprocess("m.nuts_fold_int('eboss')", hashseed=12345)
    assert a == b, f"fold differs across PYTHONHASHSEED: {a} vs {b}"


def test_builtin_hash_control_proves_the_harness_detects_instability():
    """ANTI-VACUITY CONTROL. If the subprocess harness could not detect a per-process salt, the
    stability test above would pass for the WRONG reason. This pins that builtin hash() really
    does vary across PYTHONHASHSEED under the same harness, i.e. the harness has teeth."""
    a = _in_subprocess("hash('eboss') & 0x7fffffff", hashseed=0)
    b = _in_subprocess("hash('eboss') & 0x7fffffff", hashseed=12345)
    assert a != b, ("builtin hash() did not vary across PYTHONHASHSEED; the stability test "
                    "above is vacuous on this interpreter")


# ---------------------------------------------------------------- driver wiring

def test_run_real_fit_uses_the_helper_not_builtin_hash():
    """The deployed driver must not carry the SipHash idiom any more. Source-level check so it
    cannot regress silently under a refactor that keeps the tests green."""
    src = (ROOT / "scripts" / "run_real_fit.py").read_text()
    assert "hash(survey)" not in src, "run_real_fit.py still derives the NUTS key from builtin hash()"
    assert "nuts_fold_int" in src, "run_real_fit.py does not use the shared seed helper"


def test_no_builtin_hash_on_config_values_in_deployed_drivers():
    """Sweep the deployed driver path for any other builtin hash( on a config value, which
    would reintroduce the same non-reproducibility elsewhere."""
    import re
    bad = []
    for name in ("run_real_fit.py", "run_joint_fit.py"):
        for i, line in enumerate((ROOT / "scripts" / name).read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"(^|[^_A-Za-z0-9.])hash\(", code):
                bad.append(f"{name}:{i}: {line.strip()}")
    assert not bad, "builtin hash() on a config value in a deployed driver:\n" + "\n".join(bad)


def test_seed_derivation_string_is_recorded_for_the_real_fit():
    """A recorded seed is only reproducible if the DERIVATION is recorded with it (run_joint_fit
    already stamps one). Without this the number in the export is ambiguous."""
    src = (ROOT / "scripts" / "run_real_fit.py").read_text()
    assert "seed_derivation" in src, "run_real_fit.py does not record seed_derivation in its meta"
    assert "crc32" in src, "the recorded seed_derivation must name crc32, not hash()"
