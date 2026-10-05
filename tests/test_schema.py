"""Gate A, task 1 and 3: cache schema registry, collapse guard, validated GLOBAL_STATIC k_com.
Incident: hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md"""
import os

import numpy as np
import pytest

from hcd_analysis.emulator import schema as S

R, K = 3, 8
LF = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def _synthetic_cache(n_rows=R, n_k=K, break_kcom=False):
    """Rows differ in EVERY per-row field; v_box differs per row so kfkms differs per row;
    k_com = kfkms * v_box / L is identical across rows unless break_kcom. params are chosen so that
    nbins*dv equals the analytic v_box = L 100 E(z)/(1+z) (hub 0.735, omegamh2 0.1405)."""
    rng = np.random.default_rng(0)
    hub, omh2 = 0.735, 0.1405
    om = omh2 / hub ** 2
    z = np.array([2.2, 3.0, 4.6])[:n_rows]
    E = np.sqrt(om * (1 + z) ** 3 + 1 - om)
    vbox = S.L_BOX_HMPC * 100.0 * E / (1 + z)
    nbins = np.floor(vbox / 10.0).astype(int)
    dv = vbox / nbins
    n = np.arange(1, n_k + 1)
    kfkms = 2 * np.pi * n[None, :] / vbox[:, None]
    if break_kcom:
        kfkms[1] *= 1.01
    params = np.tile([0.95, 1.9e-9, 3.7, 2.9, 1.9, hub, omh2, 7.0, 0.05], (n_rows, 1))
    params[:, 0] += 0.01 * np.arange(n_rows)            # rows differ (different sims)
    return dict(
        kfkms=kfkms, nbins_native=nbins, dv_kms=dv, params=params, z_grid=z, z_meta=z + 1e-5,
        alpha_idx=np.arange(n_rows), alpha_slope=0.7 + 0.03 * np.arange(n_rows),
        target_F=0.8 - 0.1 * np.arange(n_rows), scale=1.0 + 0.1 * np.arange(n_rows),
        P_tier_p=rng.uniform(size=(n_rows, n_k)), P_tier_c=rng.uniform(size=(n_rows, 15, n_k)),
        P_tier_c_filtered=rng.uniform(size=(n_rows, 15, n_k)),
        tier_c_counts=rng.integers(1, 9, size=(n_rows, 15)), mean_F_by_bin=rng.uniform(size=(n_rows, 15)),
        sim_name=np.array([f"sim{i}" for i in range(n_rows)]), snap=np.arange(n_rows),
        snap_group_idx=np.arange(n_rows), snap_dNdX=rng.uniform(size=(n_rows, 3)),
        snap_f_nhi=rng.uniform(size=(n_rows, 30)), snap_total_path_dX=rng.uniform(size=n_rows),
        snap_n_absorbers=rng.integers(0, 5, size=(n_rows, 30)),
    )


def test_every_registered_key_has_a_class():
    for k, spec in S.CACHE_SCHEMA_V33.items():
        assert isinstance(spec.cls, S.KeyClass), k


def test_validator_accepts_a_consistent_cache():
    rep = S.validate_cache_schema(_synthetic_cache())
    assert rep["n_rows"] == R
    assert "kfkms" in rep["checked"]
    assert np.allclose(rep["derived"]["k_com_hmpc"], 2 * np.pi * np.arange(1, K + 1) / S.L_BOX_HMPC)


def test_validator_refuses_a_collapsed_per_row_key():
    d = _synthetic_cache()
    d["kfkms"] = np.repeat(d["kfkms"][0:1], R, axis=0)       # the 2026 failure, applied to the cache
    with pytest.raises(S.SchemaCollapseError, match="PER_ROW"):
        S.validate_cache_schema(d)


def test_validator_refuses_a_global_static_that_varies():
    with pytest.raises(S.SchemaCollapseError, match="k_com_hmpc"):
        S.validate_cache_schema(_synthetic_cache(break_kcom=True))


def test_validator_refuses_unregistered_row_indexed_key():
    d = _synthetic_cache()
    d["mystery"] = np.zeros((R, 2))
    with pytest.raises(S.SchemaCollapseError, match="unregistered"):
        S.validate_cache_schema(d)


def test_validator_ignores_nan_padded_bins():
    d = _synthetic_cache()
    d["kfkms"][2, -2:] = np.nan
    d["P_tier_p"][2, -2:] = np.nan                            # row above its Nyquist
    rep = S.validate_cache_schema(d)
    assert np.all(np.isfinite(rep["derived"]["k_com_hmpc"]))


@pytest.mark.skipif(not os.path.exists(LF), reason="real cache absent")
def test_real_caches_satisfy_schema():
    """The validator on the RAW products (every stored dataset, read with h5py; independent of load_cache)."""
    import h5py
    for p in (LF, LF.replace("_lf", "_hr")):
        with h5py.File(p, "r") as h:
            raw = {k: h[k][...] for k in h.keys() if isinstance(h[k], h5py.Dataset) and k != "sim_name"}
            raw["sim_name"] = np.array([s.decode() if isinstance(s, bytes) else s for s in h["sim_name"][...]])
        rep = S.validate_cache_schema(raw)
        assert rep["derived"]["vbox_rtol_measured"] < S.VBOX_RTOL
        assert len(rep["derived"]["k_com_hmpc"]) in (172, 525)
        assert np.allclose(rep["derived"]["k_com_hmpc"], 2 * np.pi * np.arange(1, len(rep["derived"]["k_com_hmpc"]) + 1) / S.L_BOX_HMPC, rtol=1e-9)


@pytest.mark.skipif(not os.path.exists(LF), reason="real cache absent")
def test_load_cache_exposes_k_com_and_report():
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(LF)
    assert d["k_com_hmpc"].shape == (172,) and d["schema_report"]["n_rows"] == 21440
    hr = load_cache(LF.replace("_lf", "_hr"))
    assert hr["k_com_hmpc"].shape == (525,)
    assert np.allclose(hr["k_com_hmpc"][:172], d["k_com_hmpc"], rtol=1e-9)   # same box modes, longer grid
