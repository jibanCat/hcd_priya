"""TDD for hcd_analysis.emulator.blinding — the REAL-data parameter-blind (A_p, n_s ONLY).

Pins the locked blinding-strategy contract:
  1. ROUND-TRIP:    unblind(apply_blind(x)) == x   (exact inverse, no drift).
  2. DETERMINISM:   same seed_str → same offset; different seed → (almost surely) different.
  3. WITHIN ±3σ:    every offset lies in (−3σ_prior, +3σ_prior); σ_prior = (hi−lo)/√12 uniform.
  4. ONLY A_p/n_s:  no other column is touched by apply_blind/unblind.
  5. INDEPENDENCE:  the A_p and n_s offsets are derived from independent salted hashes.
  6. blind.lock:    write stores the SEED (not the offset) + round-trips through read; refuses
                    to overwrite a frozen lock.
  7. SAFETY:        apply/unblind on a matrix MISSING a blinded column raises (no silent no-op).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_blinding.py -q
"""
import json
import os

import numpy as np
import pytest

from hcd_analysis.emulator import blinding as B
from hcd_analysis.emulator.inference import PARAM_NAMES
from hcd_analysis.emulator.data import SAMPLING_LIMITS

SEED = "hcd_priya_real_fit_v1@deadbeef"


def _matrix(n=64, seed=0):
    rng = np.random.default_rng(seed)
    # an (N, 9) θ matrix with realistic A_p / n_s magnitudes (so a relative tolerance is meaningful).
    x = rng.standard_normal((n, len(PARAM_NAMES)))
    x[:, PARAM_NAMES.index("ns")] = 0.96 + 0.01 * rng.standard_normal(n)
    x[:, PARAM_NAMES.index("Ap")] = 1.9e-9 + 1e-11 * rng.standard_normal(n)
    return x


# ---- 1. round-trip ------------------------------------------------------------------------- #
def test_roundtrip_identity():
    off = B.blind_offset(SEED)
    x = _matrix()
    back = B.unblind(B.apply_blind(x, off), off)
    assert np.allclose(back, x, rtol=0, atol=1e-300 + 1e-12 * np.abs(x).max())
    # and exactly equal where untouched.
    assert np.array_equal(back, x) or np.allclose(back, x, rtol=1e-15, atol=0)


def test_roundtrip_other_order():
    off = B.blind_offset(SEED)
    x = _matrix(seed=3)
    # apply then unblind, and unblind then apply, both recover x.
    assert np.allclose(B.unblind(B.apply_blind(x, off), off), x, rtol=1e-14, atol=0)
    assert np.allclose(B.apply_blind(B.unblind(x, off), off), x, rtol=1e-14, atol=0)


# ---- 2. determinism ------------------------------------------------------------------------ #
def test_determinism_same_seed():
    assert B.blind_offset(SEED) == B.blind_offset(SEED)


def test_different_seed_different_offset():
    a = B.blind_offset(SEED)
    b = B.blind_offset(SEED + "x")
    assert any(a[p] != b[p] for p in B.BLIND_PARAMS)


# ---- 3. within ±3σ_prior ------------------------------------------------------------------- #
def test_offset_within_3sigma():
    # sweep many seeds; every offset stays strictly inside (−3σ, +3σ).
    for s in range(500):
        off = B.blind_offset(f"{SEED}/{s}")
        for p in B.BLIND_PARAMS:
            half = B.OFFSET_SIGMA_MULTIPLE * B.prior_sigma(p)
            assert -half <= off[p] <= half


def test_prior_sigma_matches_uniform_std():
    for p in B.BLIND_PARAMS:
        i = PARAM_NAMES.index(p)
        lo, hi = float(SAMPLING_LIMITS[i, 0]), float(SAMPLING_LIMITS[i, 1])
        assert B.prior_sigma(p) == pytest.approx((hi - lo) / np.sqrt(12.0))


def test_offset_spans_window():
    # over many seeds the offsets should roughly fill (−3σ, +3σ): max |offset| > 2σ for each param.
    seen = {p: 0.0 for p in B.BLIND_PARAMS}
    for s in range(2000):
        off = B.blind_offset(f"span/{s}")
        for p in B.BLIND_PARAMS:
            seen[p] = max(seen[p], abs(off[p]))
    for p in B.BLIND_PARAMS:
        assert seen[p] > 2.0 * B.prior_sigma(p)


# ---- 4. ONLY A_p / n_s touched ------------------------------------------------------------- #
def test_only_blind_params_touched():
    off = B.blind_offset(SEED)
    x = _matrix(seed=7)
    y = B.apply_blind(x, off)
    for j, nm in enumerate(PARAM_NAMES):
        if nm in B.BLIND_PARAMS:
            assert not np.array_equal(y[:, j], x[:, j])           # moved
            assert np.allclose(y[:, j] - x[:, j], off[nm])        # by exactly the offset
        else:
            assert np.array_equal(y[:, j], x[:, j])               # untouched, byte-exact


def test_extra_nuisance_columns_passthrough():
    # a packed matrix with τ₀ / α columns after θ9: only ns/Ap move, the rest pass through.
    names = list(PARAM_NAMES) + ["tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, len(names)))
    off = B.blind_offset(SEED)
    y = B.apply_blind(x, off, columns=names)
    for j, nm in enumerate(names):
        if nm in B.BLIND_PARAMS:
            assert np.allclose(y[:, j] - x[:, j], off[nm])
        else:
            assert np.array_equal(y[:, j], x[:, j])
    assert np.allclose(B.unblind(y, off, columns=names), x, rtol=1e-14, atol=0)


# ---- 5. A_p / n_s offsets independent ------------------------------------------------------ #
def test_offsets_independent_across_params():
    # the salted hashes make the two offsets uncorrelated; a crude check: over seeds the
    # normalized offsets are not identical.
    diffs = []
    for s in range(200):
        off = B.blind_offset(f"indep/{s}")
        diffs.append((off["ns"] / B.prior_sigma("ns")) - (off["Ap"] / B.prior_sigma("Ap")))
    assert np.std(diffs) > 0.1     # they genuinely differ


# ---- 6. blind.lock ------------------------------------------------------------------------- #
def test_blind_lock_roundtrip(tmp_path):
    path = str(tmp_path / "blind.lock")
    rec = B.write_blind_lock(path, "hcd_priya_real_fit_v1", commit="deadbeef")
    assert os.path.exists(path)
    got = B.read_blind_lock(path)
    assert got["seed_str"] == rec["seed_str"] == "hcd_priya_real_fit_v1@deadbeef"
    # the offset must NOT be stored in the lock (it is derived at view time).
    blob = json.dumps(got)
    assert "offset" not in {k.lower() for k in got} or "prior_sigma" in got
    assert "offset_value" not in blob.lower()
    # offset_from_lock reproduces blind_offset(seed_str).
    assert B.offset_from_lock(path) == B.blind_offset("hcd_priya_real_fit_v1@deadbeef")


def test_blind_lock_refuses_overwrite(tmp_path):
    path = str(tmp_path / "blind.lock")
    B.write_blind_lock(path, "proj", commit="c0ffee")
    with pytest.raises(FileExistsError):
        B.write_blind_lock(path, "proj2", commit="c0ffee")


# ---- 7. safety: refuse a partially-blind matrix -------------------------------------------- #
def test_refuses_missing_blind_column():
    off = B.blind_offset(SEED)
    # a matrix whose columns drop "Ap" → must raise (no silent unblinded write).
    names = [nm for nm in PARAM_NAMES if nm != "Ap"]
    x = np.zeros((4, len(names)))
    with pytest.raises(ValueError):
        B.apply_blind(x, off, columns=names)


def test_refuses_shape_mismatch():
    off = B.blind_offset(SEED)
    x = np.zeros((4, len(PARAM_NAMES) + 1))
    with pytest.raises(ValueError):
        B.apply_blind(x, off, columns=PARAM_NAMES)
