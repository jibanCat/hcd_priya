"""Gate D: the LF->HF correction measured mode by mode at equal physical k (spec hcd_priya_notes
docs/superpowers/emulator-debug-2026-10/GATE_D_SPEC.md). Synthetic inputs; no caches."""
import numpy as np
import pytest

from hcd_analysis.emulator import mf_modes as MM

K = 6


def _pair_caches(n_sims=3, zs=(2.8, 3.0, 3.2), n_rung=4, k_shift=0.0):
    """Minimal LF/HR dicts with the keys the module reads; HR has more modes than LF."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n_sims):
        p = rng.random(9)
        for z in zs:
            for a in range(n_rung):
                rows.append((s, p, z, a))
    R = len(rows)
    params = np.stack([r[1] for r in rows])
    z = np.array([r[2] for r in rows])
    a = np.array([r[3] for r in rows])
    vmax = 13000.0 + 100 * z + 50 * params[:, 5]
    n = np.arange(1, K + 1)
    lf = dict(sim_name=np.array([f"s{r[0]}" for r in rows]), params=params, z_grid=z, alpha_idx=a,
              tau0=0.3 + 0.1 * a + 0.05 * z, kfkms=2 * np.pi * n[None, :] / vmax[:, None])
    nh = np.arange(1, 2 * K + 1)
    hr = dict(sim_name=lf["sim_name"].copy(), params=params.copy(), z_grid=z.copy(), alpha_idx=a.copy(),
              tau0=lf["tau0"].copy(), kfkms=2 * np.pi * nh[None, :] / (vmax[:, None] * (1 + k_shift)))
    return lf, hr


def test_pairs_match_identical_design_points():
    lf, hr = _pair_caches()
    pairs = MM.match_pairs(lf, hr)
    assert len(pairs) == lf["z_grid"].size and all(h == l for h, l in pairs)


def test_coordinate_mapping_error_is_zero_for_equal_physical_modes_and_detects_a_shift():
    lf, hr = _pair_caches()
    assert MM.mode_mapping_error(lf, hr, MM.match_pairs(lf, hr), n_modes=K) < 1e-15
    lf2, hr2 = _pair_caches(k_shift=1e-6)
    assert abs(MM.mode_mapping_error(lf2, hr2, MM.match_pairs(lf2, hr2), n_modes=K) - 1e-6) < 1e-9


def test_targets_are_log_ratios_mode_by_mode_with_no_interpolation():
    lf, hr = _pair_caches()
    pairs = MM.match_pairs(lf, hr)
    M = len(pairs)
    P_lf = np.full((M, 4, K), 2.0)
    P_hr_full = np.full((lf["z_grid"].size, 4, 2 * K), 3.0)
    P_hr_full[:, :, K:] = 99.0                        # HR-only modes must be ignored
    P_hr_full[0, 2, 1] = 0.0                          # empty class -> NaN target
    hr["P_filt"] = P_hr_full
    x = np.zeros((lf["z_grid"].size, 10))
    t = MM.measure_mode_targets(lf, hr, pairs, P_lf, x=x, n_modes=K)
    assert t["g"].shape == (M, 4, K)
    assert np.isnan(t["g"][0, 2, 1]) and np.allclose(np.delete(t["g"].ravel(), np.ravel_multi_index((0, 2, 1), (M, 4, K))), np.log(1.5))


def _separable_targets(lf, truth_fn):
    pairs = MM.match_pairs(lf, lf)
    z = lf["z_grid"]
    x = np.zeros((z.size, 10))
    from hcd_analysis.emulator.data import Z_LIMITS
    x[:, 9] = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    g = np.stack([truth_fn(zz, aa) for zz, aa in zip(z, lf["alpha_idx"])])
    return dict(x=x, tau0=lf["tau0"], alpha_idx=lf["alpha_idx"], g=g, sim=lf["sim_name"],
                hr_row=np.arange(z.size), lf_row=np.arange(z.size)), pairs


def test_fit_and_apply_recover_a_separable_theta_independent_correction():
    lf, _ = _pair_caches()
    kk = np.arange(K)
    truth = lambda zz, aa: np.ones((4, 1)) * (0.01 * kk + 0.02 * (zz - 3.0) + 0.003 * aa)[None, :]
    t, _ = _separable_targets(lf, truth)
    tables = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != "s0")[0])
    for i in np.where(t["sim"] == "s0")[0]:
        g = MM.apply_mode_mf(tables, t["x"][i], t["tau0"][i])
        assert np.allclose(g, truth(lf["z_grid"][i], lf["alpha_idx"][i]), atol=1e-10)


def test_held_out_rows_do_not_influence_the_fit_including_log_rho():
    lf, _ = _pair_caches()
    t, _ = _separable_targets(lf, lambda zz, aa: np.full((4, K), 0.05))
    train = np.where(t["sim"] != "s0")[0]
    a = MM.fit_mode_mf(t, train_rows=train)
    t2 = dict(t, g=t["g"].copy())
    t2["g"][t["sim"] == "s0"] += 7.0                  # held-out rows changed wildly
    b = MM.fit_mode_mf(t2, train_rows=train)
    for k in a:
        assert np.allclose(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k


def test_tables_round_trip_with_k_com_labels_and_refuse_velocity_keys(tmp_path):
    lf, _ = _pair_caches()
    t, _ = _separable_targets(lf, lambda zz, aa: np.full((4, K), 0.05))
    tables = MM.fit_mode_mf(t)
    k_com = 2 * np.pi * np.arange(1, K + 1) / 120.0
    p = tmp_path / "mf.npz"
    MM.save_mode_mf(p, tables, k_com_hmpc=k_com, provenance={"lf": "x", "hr": "y"})
    back, kc, prov = MM.load_mode_mf(p)
    assert np.allclose(kc, k_com) and prov["lf"] == "x"
    for k in tables:
        assert np.allclose(np.asarray(back[k]), np.asarray(tables[k]))
    with pytest.raises(ValueError, match="velocity"):
        MM.save_mode_mf(tmp_path / "bad.npz", dict(tables, kfkms=np.ones(K)), k_com_hmpc=k_com, provenance={})
    with pytest.raises(ValueError, match="2 pi n / L"):
        MM.save_mode_mf(tmp_path / "bad2.npz", tables, k_com_hmpc=k_com * 1.01, provenance={})
