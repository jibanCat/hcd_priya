"""Phase-C T2 — τ₀-band error vector: make_tau0_bands + aggregate_error_vector (4D).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_tau0_bands.py -v
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hcd_analysis.emulator.data import (
    make_tau0_bands, tau0_ladder_factor, KIM_AMP, KIM_SLOPE,
)
from hcd_analysis.emulator.train import aggregate_error_vector


def _synthetic_ladder(n_alpha=20, alpha_lo=0.6556, alpha_hi=1.3312):
    """tau0 = alpha * Kim(z) on a PRIYA-like ladder × z grid (the real cache shape)."""
    alphas = np.linspace(alpha_lo, alpha_hi, n_alpha)
    z = np.arange(2.2, 4.61, 0.2)
    A, Z = np.meshgrid(alphas, z, indexing="ij")
    kim = KIM_AMP * (1.0 + Z) ** KIM_SLOPE
    tau0 = (A * kim).ravel()
    zg = Z.ravel()
    return tau0, zg, alphas


def test_tau0_ladder_factor_is_z_independent():
    tau0, zg, alphas = _synthetic_ladder()
    a = tau0_ladder_factor(tau0, zg).reshape(len(alphas), -1)
    # each ladder rung has the SAME factor across all z (to numerical precision)
    assert np.allclose(a.std(axis=1), 0, atol=1e-10)
    assert np.allclose(a.mean(axis=1), alphas, rtol=1e-10)


def test_make_tau0_bands_isolates_extremes():
    tau0, zg, alphas = _synthetic_ladder()
    band, centres = make_tau0_bands(tau0, zg, n_tb=4)
    a = tau0_ladder_factor(tau0, zg)
    # band 0 = ONLY the least-absorption rung; band 3 = ONLY the most-absorption rung
    assert np.allclose(a[band == 0].max(), alphas.min())
    assert np.allclose(a[band == 3].min(), alphas.max())
    # every row assigned to a band in [0,4)
    assert set(np.unique(band)).issubset({0, 1, 2, 3})
    assert (band >= 0).all()
    # centres monotone increasing in α, finite
    assert np.all(np.diff(centres) > 0)
    assert np.isfinite(centres).all()
    assert np.isclose(centres[0], alphas.min()) and np.isclose(centres[-1], alphas.max())


def test_make_tau0_bands_interior_split():
    tau0, zg, alphas = _synthetic_ladder()
    band, _ = make_tau0_bands(tau0, zg, n_tb=4)
    # interior bands 1,2 hold only interior rungs (strictly between the extremes)
    a = tau0_ladder_factor(tau0, zg)
    interior = (band == 1) | (band == 2)
    assert a[interior].min() > alphas.min()
    assert a[interior].max() < alphas.max()


def test_make_tau0_bands_n_tb_lt_3_fallback():
    tau0, zg, _ = _synthetic_ladder()
    band, centres = make_tau0_bands(tau0, zg, n_tb=2)
    assert set(np.unique(band)).issubset({0, 1})
    assert (band >= 0).all() and len(centres) == 2


def test_aggregate_error_vector_4d_rms_over_folds():
    rng = np.random.default_rng(0)
    F, C, K, Zb, Tb = 3, 4, 6, 2, 4
    resid = [rng.uniform(0.01, 0.1, (C, K, Zb, Tb)) for _ in range(F)]
    neff = [rng.uniform(5e3, 1e4, (C, K, Zb, Tb)) for _ in range(F)]
    out = aggregate_error_vector(resid, neff)
    assert out["sigma"].shape == (C, K, Zb, Tb)
    expect = np.sqrt(np.mean(np.stack(resid) ** 2, axis=0))
    assert np.allclose(out["sigma"], expect)
    assert out["dla_shot_flag"].shape == (K,)


def test_aggregate_error_vector_dla_flag_worst_over_all_bands():
    F, C, K, Zb, Tb = 2, 4, 5, 2, 3
    resid = [np.full((C, K, Zb, Tb), 0.05) for _ in range(F)]
    neff = [np.full((C, K, Zb, Tb), 1e4) for _ in range(F)]
    # drive ONE DLA (class 3) cell at k=2 below threshold in a single (fold,zb,tb)
    neff[0][3, 2, 1, 0] = 0.5
    out = aggregate_error_vector(resid, neff, shot_thresh=1.0)
    flag = out["dla_shot_flag"]
    assert flag[2] and flag.sum() == 1   # only k=2 flagged (worst-case over all bands)


def test_aggregate_error_vector_3d_backcompat():
    """Old (4,K,Zb) inputs still work (trailing axis optional)."""
    F, C, K, Zb = 2, 4, 5, 3
    resid = [np.full((C, K, Zb), 0.05) for _ in range(F)]
    neff = [np.full((C, K, Zb), 1e4) for _ in range(F)]
    out = aggregate_error_vector(resid, neff)
    assert out["sigma"].shape == (C, K, Zb)
    assert out["dla_shot_flag"].shape == (K,)
