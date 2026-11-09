"""Matched leave-one-simulation-out helpers (hcd_analysis/emulator/loo_matched.py). Synthetic cache only."""
import importlib.util
import os

import numpy as np
import pytest

from hcd_analysis.emulator import loo_matched as L
from hcd_analysis.emulator.data import load_cache_historical
from hcd_analysis.emulator.dndx_wc import _DELTA_C_COEFFS, w_c_corrected
from tests.emulator._fixture import write_synthetic_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def d(tmp_path):
    p = tmp_path / "obs.h5"
    write_synthetic_cache(p, n_sims=5, snaps_per_sim=4, n_alpha=4, n_k=8)
    return load_cache_historical(p, reason='test: a cache in the pre-lineage 3.3 format')


def test_matched_mask_selects_the_priya_design(d):
    m = L.matched_mask(d)
    assert set(np.asarray(d["alpha_idx"])[m]) <= set(L.PRIYA_ALPHA_IDX)
    assert np.all(np.round(np.asarray(d["z_grid"])[m], 1) >= 2.2) and m.any() and not m.all()


def test_fold_split_holds_out_the_whole_simulation(d):
    sims = np.asarray(d["sim_name"])
    fit, val, held = L.fold_split(d, "sim2")
    assert set(sims[held]) == {"sim2"} and "sim2" not in set(sims[fit]) | set(sims[val])
    assert not set(fit) & set(val) and not set(held) & (set(fit) | set(val))
    m = L.matched_mask(d)
    assert set(fit) | set(val) | set(held) == set(np.flatnonzero(m))
    assert np.all(m[held]) and set(held) == set(np.flatnonzero(m & (sims == "sim2")))
    n_train = fit.size + val.size
    assert val.size == int(L.VAL_FRAC * n_train)
    f2, v2, h2 = L.fold_split(d, "sim2")
    assert np.array_equal(fit, f2) and np.array_equal(val, v2) and np.array_equal(held, h2)
    with pytest.raises(ValueError, match="no matched rows"):
        L.fold_split(d, "nosuchsim")


def _calibrate_module():
    spec = importlib.util.spec_from_file_location("calibrate_delta_c", os.path.join(REPO, "scripts", "calibrate_delta_c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_delta_c_is_the_calibration_method_on_the_given_sims_only(d):
    cal = _calibrate_module()
    t = L.group_table(d)
    keep = t["ok"] & (t["sim"] != "sim1")
    from hcd_analysis.emulator.dndx_wc import mu_from_dndx, w_c_from_mu
    wm = np.asarray(w_c_from_mu(mu_from_dndx(t["dndx"][keep], t["xbar_geom"][keep])))
    ref_coeffs, _, ref_std, _ = cal.fit(t["z"][keep], t["w_counted"][keep], wm, deg=2)
    coeffs, std = L.fit_delta_c(d, [s for s in set(t["sim"]) if s != "sim1"])
    assert np.allclose(coeffs, ref_coeffs, rtol=0, atol=1e-14) and np.allclose(std, ref_std, rtol=0, atol=1e-14)
    all_coeffs, _ = L.fit_delta_c(d, set(t["sim"]))
    assert not np.allclose(all_coeffs, coeffs)                      # the held-out simulation matters, and is left out


def test_geometric_path_is_total_path_over_sightlines(d):
    t = L.group_table(d)
    cc = np.asarray(d["coarse_counts"])
    gid = np.asarray(d["snap_group_idx"])
    for g in range(t["z"].size):
        r = np.flatnonzero(gid == g)[0]
        assert t["xbar_geom"][g] == pytest.approx(np.asarray(d["snap_total_path_dX"])[g] / cc[r].sum(), rel=1e-15)


def test_weights_with_explicit_coefficients_equal_the_deployed_function():
    rng = np.random.default_rng(0)
    dndx = rng.uniform(0.01, 0.3, (20, 3))
    xbar = rng.uniform(0.5, 3.0, 20)
    z = np.linspace(1.5, 6.0, 20)                                    # includes the clamped ends
    a = L.w_c_with_coeffs(dndx, xbar, z, np.asarray(_DELTA_C_COEFFS))
    b = np.asarray(w_c_corrected(dndx, xbar, z))
    assert np.allclose(a, b, rtol=0, atol=1e-13) and np.allclose(a.sum(-1), 1.0)


def test_step_matched_epochs_scale_the_production_epochs_rounding_down():
    assert L.PRODUCTION_STEPS == 6840
    assert L.step_matched_epochs(38 * 512) == 180                   # 38 steps per epoch: the production budget
    n = L.step_matched_epochs(14 * 512 - 100)                       # 14 steps per epoch
    assert n == 6840 // 14 == 488 and n * 14 <= L.PRODUCTION_STEPS < (n + 1) * 14


def test_the_matched_design_constants():
    assert L.PRIYA_Z == (2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6)
    assert L.PRIYA_ALPHA_IDX == (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)


def test_the_member_recipe_is_the_production_recipe():
    def load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    assert load("run_matched_loo_member").RECIPE == load("train_production_emulator").RECIPE


# ----------------------------------------------------------------------------- the GP side of a fold
def _gp_cache(n_sims=4, n_k=6, rng=None):
    rng = rng or np.random.default_rng(1)
    sims, zs, aidx = np.meshgrid(np.arange(n_sims), np.asarray(L.PRIYA_Z), np.arange(20), indexing="ij")
    sims, zs, aidx = sims.ravel(), zs.ravel(), aidx.ravel()
    order = rng.permutation(sims.size)                               # the cache's row order must not matter
    sims, zs, aidx = sims[order], zs[order], aidx[order]
    theta = rng.uniform(0.0, 1.0, (n_sims, 9))
    P = rng.uniform(0.5, 2.0, (sims.size, n_k))
    P[:, -1] = np.nan                                                # beyond the modes used
    return {"params": theta[sims], "param_names": np.array([f"p{i}" for i in range(9)]),
            "sim_name": np.array([f"sim{s}" for s in sims]), "z_grid": zs, "alpha_idx": aidx,
            "alpha_slope": 0.6 + 0.03 * aidx, "P_tier_p": P}


def test_gp_fold_inputs_hold_out_the_simulation_and_share_one_design_across_z():
    c = _gp_cache()
    g = L.gp_fold_inputs(c, "sim2", 5)
    nz, n_rung = len(L.PRIYA_Z), len(L.PRIYA_ALPHA_IDX)
    assert g["X_train"].shape == (nz, 3 * n_rung, 10) and g["Y_train"].shape == (nz, 3 * n_rung, 5)
    assert g["X_held"].shape == (nz, n_rung, 10) and g["rows_held"].shape == (nz, n_rung)
    assert "sim2" not in set(g["train_sims"]) and set(g["train_alpha_idx"]) == set(L.PRIYA_ALPHA_IDX)
    sims = c["sim_name"]
    for i, zz in enumerate(L.PRIYA_Z):
        r = g["rows_held"][i]
        assert set(sims[r]) == {"sim2"} and np.allclose(c["z_grid"][r], zz) and set(c["alpha_idx"][r]) == set(L.PRIYA_ALPHA_IDX)
        assert np.array_equal(g["Y_held"][i], c["P_tier_p"][r, :5])
        assert np.array_equal(g["X_held"][i][:, 0], c["alpha_slope"][r])
        assert np.array_equal(g["X_held"][i][:, 1:], c["params"][r])
    assert np.array_equal(g["X_train"][0][:, 1:], g["X_train"][-1][:, 1:])  # the same rows at every z
    rt = g["rows_train"]
    assert rt.shape == (nz, 3 * n_rung) and "sim2" not in set(sims[rt.ravel()])
    m = L.matched_mask(c)
    assert sorted(rt.ravel().tolist()) == sorted(np.flatnonzero(m & (sims != "sim2")).tolist())


def test_gp_fold_inputs_refuse_a_broken_design_or_non_finite_values():
    c = _gp_cache()
    drop = np.flatnonzero((c["sim_name"] == "sim1") & (c["alpha_idx"] == 4) & np.isclose(c["z_grid"], 3.0))
    bad = {k: np.delete(v, drop, axis=0) if k != "param_names" else v for k, v in c.items()}
    with pytest.raises(ValueError, match="training rows differ"):
        L.gp_fold_inputs(bad, "sim2", 5)
    with pytest.raises(ValueError, match="held-out rows"):
        L.gp_fold_inputs(bad, "sim1", 5)
    with pytest.raises(ValueError, match="non-finite"):
        L.gp_fold_inputs(c, "sim2", 6)


def test_the_matched_design_must_be_complete():
    c = _gp_cache()
    assert L.check_matched_design(c) == ["sim0", "sim1", "sim2", "sim3"]
    drop = np.flatnonzero((c["sim_name"] == "sim1") & (c["alpha_idx"] == 4) & np.isclose(c["z_grid"], 3.0))
    with pytest.raises(ValueError, match="incomplete"):
        L.check_matched_design({k: np.delete(v, drop, axis=0) if k != "param_names" else v for k, v in c.items()})
    dup = np.flatnonzero((c["sim_name"] == "sim1") & (c["alpha_idx"] == 4))[:1]
    with pytest.raises(ValueError, match="appears twice"):
        L.check_matched_design({k: np.concatenate([v, v[dup]]) if k != "param_names" else v for k, v in c.items()})


def test_gp_fold_inputs_are_written_once_and_load_without_pickle(tmp_path):
    g = L.gp_fold_inputs(_gp_cache(), "sim0", 5)
    p = tmp_path / "fold.npz"
    L.write_gp_fold_inputs(p, g)
    with pytest.raises(FileExistsError):
        L.write_gp_fold_inputs(p, g)
    with np.load(p, allow_pickle=False) as f:
        assert set(f.files) == set(g) and all(np.array_equal(f[k], g[k]) for k in g)


# ----------------------------------------------------------------------------- the C1 side of a fold
def test_predict_fold_averages_members_after_reconstruction_and_uses_the_per_fold_weights(d):
    from hcd_analysis.emulator.data import Z_LIMITS, normalize_params
    K = np.asarray(d["P_tier_p"]).shape[1]
    calls = []

    def fake(model, ns, x, tau0):                                    # member "model" = a scale; ns = a dN/dX offset
        calls.append((model, np.array(x), tau0))
        return np.full((4, K), model) * np.arange(1, 5)[:, None], np.array([0.05, 0.02, 0.01]) + ns
    _, _, held = L.fold_split(d, "sim3")
    rows = held[:4]
    coeffs, _ = L.fit_delta_c(d, ["sim0", "sim1", "sim2", "sim4"])
    P, info = L.predict_fold([(1.0, 0.0), (3.0, 0.01)], d, rows, coeffs, outputs=fake)
    t = L.group_table(d)
    for i, r in enumerate(rows):
        pf = 2.0 * np.arange(1, 5)[:, None] * np.ones((4, K))       # mean of 1x and 3x
        dn = np.array([0.055, 0.025, 0.015])
        w = L.w_c_with_coeffs(dn, t["xbar_geom"][d["snap_group_idx"][r]], d["z_grid"][r], coeffs)
        assert np.allclose(info["P_filt"][i], pf) and np.allclose(info["dndx"][i], dn) and np.allclose(info["w_c"][i], w)
        assert np.allclose(P[i], w @ pf, rtol=1e-12)
    m, x, tau0 = calls[0]
    z_unit = (d["z_grid"][rows[0]] - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    assert np.allclose(x, np.concatenate([normalize_params(d["params"][rows[0]]), [z_unit]]))
    assert tau0 == pytest.approx(float(d["tau0"][rows[0]]))
    Pt, it = L.predict_fold([(1.0, 0.0)], d, rows, coeffs, truth_weights=True, outputs=fake)
    assert it["weights"] == "truth" and np.allclose(it["w_c"], np.asarray(d["w_c_cache"])[rows])


def test_fold_errors_pair_both_methods_on_identical_points_and_apply_the_not_run_rules(d):
    from hcd_analysis.emulator.kcoord import k_skm_from_kcom
    _, _, held = L.fold_split(d, "sim3")
    P = np.asarray(d["P_tier_p"], dtype=np.float64)
    fin = np.all(np.isfinite(P[held]), axis=0)
    n = int(np.argmin(fin)) if not fin.all() else fin.size
    perm = held[::-1]                                                # the GP record lists the rows in another order
    rec = {"fold_sim": "sim3", "status": "OK", "reason": "", "rows_held": perm.reshape(1, -1).tolist(),
           "mean": (P[perm, :n] * 0.98)[None].tolist()}
    k_com = np.arange(1, P.shape[1] + 1) * 2 * np.pi / 120.0
    out = L.fold_errors(d, "sim3", held, P[held] * 1.01, rec, k_com)
    assert out["status"] == "OK" and out["e_c1"].shape == out["e_gp"].shape == out["k_skm"].shape == (held.size * n,)
    assert np.allclose(out["e_c1"], 0.01) and np.allclose(out["e_gp"], -0.02)
    r0 = held[0]
    k0 = np.asarray(k_skm_from_kcom(k_com[:n], float(d["z_grid"][r0]), float(d["params"][r0][5]), float(d["params"][r0][6])))
    assert np.allclose(out["k_skm"][:n], k0) and np.all(out["z"][:n] == d["z_grid"][r0])
    nr = L.fold_errors(d, "sim3", held, P[held], dict(rec, status="NOT_RUN", reason="LinAlgError at z=3.0"), k_com)
    assert nr["status"] == "NOT_RUN" and "LinAlgError" in nr["reason"]
    few = L.fold_errors(d, "sim3", held, P[held], rec, k_com, member_ok=[True, True, True, False, False])
    assert few["status"] == "NOT_RUN" and "3 usable members" in few["reason"]
    four = L.fold_errors(d, "sim3", held, P[held] * 1.01, rec, k_com, member_ok=[True, True, True, True, False])
    assert four["status"] == "OK"                                    # at least 4 usable members suffice
    with pytest.raises(ValueError, match="different rows"):
        L.fold_errors(d, "sim3", held[1:], P[held[1:]], rec, k_com)
    with pytest.raises(ValueError, match="not 'sim2'"):
        L.fold_errors(d, "sim2", held, P[held], rec, k_com)


# ----------------------------------------------------------------------------- one fold assembled (eval script)
def _assembly(d, holdout="sim3", n=5, nan_at=(), arms=None, cache="c" * 64):
    fit, val, held = L.fold_split(d, holdout)
    coeffs, _ = L.fit_delta_c(d, [s for s in set(np.asarray(d["sim_name"]).tolist()) if s != holdout])
    fj = {"holdout_sim": holdout, "held_rows": held.tolist(), "fit_rows": fit.tolist(), "val_rows": val.tolist(),
          "delta_c_coeffs": coeffs.tolist()}
    P = np.asarray(d["P_tier_p"], dtype=np.float64)
    fin = np.all(np.isfinite(P[held]), axis=0)
    nk = int(np.argmin(fin)) if not fin.all() else fin.size
    k_com = (np.arange(1, P.shape[1] + 1) * 2 * np.pi / 120.0).tolist()
    members = []
    for i in range(n):
        arm = (arms or ["verbatim"] * n)[i]
        meta = {"cache_sha256": cache, "recipe": {"step_matched": arm == "step_matched"}, "k_com_hmpc": k_com}
        members.append((f"/synthetic/m{i}", dict(fj), meta, ("nan" if i in nan_at else 1.0, None)))
    rec = {"fold_sim": holdout, "status": "OK", "reason": "", "rows_held": held.reshape(1, -1).tolist(),
           "mean": (P[held, :nk] * 0.98)[None].tolist(), "rows_train": [sorted(fit.tolist() + val.tolist())],
           "per_z": [{"z": 3.0, "converged_final": False}, {"z": 3.2, "converged_final": True}]}
    return members, rec


def _fake_predict(nets, d, held, coeffs, truth_weights=False):
    """A stand-in for predict_fold: finite on every mode, as a network is (1% above the truth where it is finite)."""
    P = np.nan_to_num(np.asarray(d["P_tier_p"], dtype=np.float64)[held], nan=1.0) * 1.01
    return (P * np.nan if any(n[0] == "nan" for n in nets) else P), {}


def test_assemble_fold_applies_the_ensemble_and_failure_rules(d):
    members, rec = _assembly(d)
    kw = dict(cache_sha256="c" * 64, predict=_fake_predict)
    out = L.assemble_fold(d, "sim3", members, rec, **kw)
    assert out["status"] == "OK" and out["member_ok"] == [True] * 5 and out["arm"] == "verbatim"
    assert np.allclose(out["e_c1"], 0.01) and np.allclose(out["e_gp"], -0.02) and out["gp_unconverged_z"] == [3.0]
    four = L.assemble_fold(d, "sim3", members[:4], rec, **kw)          # one member missing (time limit): dropped
    assert four["status"] == "OK" and four["member_ok"] == [True] * 4 + [False]
    m_nan, _ = _assembly(d, nan_at=(2,))
    dropped = L.assemble_fold(d, "sim3", m_nan, rec, **kw)
    assert dropped["status"] == "OK" and dropped["member_ok"] == [True, True, False, True, True]
    three = L.assemble_fold(d, "sim3", members[:3], rec, **kw)
    assert three["status"] == "NOT_RUN" and "3 usable members" in three["reason"]
    gp_out = L.assemble_fold(d, "sim3", members, {"fold_sim": "sim3", "status": "NOT_RUN", "reason": "time limit"}, **kw)
    assert gp_out["status"] == "NOT_RUN" and "time limit" in gp_out["reason"]
    none = L.assemble_fold(d, "sim3", [], rec, **kw)
    assert none["status"] == "NOT_RUN" and none["member_ok"] == [False] * 5


def test_assemble_fold_refuses_a_mixed_or_foreign_ensemble(d):
    members, rec = _assembly(d)
    kw = dict(cache_sha256="c" * 64, predict=_fake_predict)
    with pytest.raises(ValueError, match="given twice"):
        L.assemble_fold(d, "sim3", members[:4] + [members[0]], rec, **kw)
    mixed, _ = _assembly(d, arms=["verbatim"] * 4 + ["step_matched"])
    with pytest.raises(ValueError, match="mix training arms"):
        L.assemble_fold(d, "sim3", mixed, rec, **kw)
    with pytest.raises(ValueError, match="another cache"):
        L.assemble_fold(d, "sim3", members, rec, cache_sha256="d" * 64, predict=_fake_predict)
    with pytest.raises(ValueError, match="holds out"):
        L.assemble_fold(d, "sim2", members, rec, **kw)
    bad_rows = dict(rec, rows_train=[rec["rows_train"][0][1:]])
    with pytest.raises(ValueError, match="training rows"):
        L.assemble_fold(d, "sim3", members, bad_rows, **kw)
    with pytest.raises(ValueError, match="more than the expected"):
        L.assemble_fold(d, "sim3", members, rec, expected_members=4, **kw)


def test_assemble_fold_refuses_another_fold_and_passes_the_weights_choice(d):
    members, rec = _assembly(d)
    kw = dict(cache_sha256="c" * 64, predict=_fake_predict)
    md, fj, meta, net = members[4]
    for bad in (dict(fj, val_rows=fj["val_rows"][1:]),
                dict(fj, delta_c_coeffs=(np.asarray(fj["delta_c_coeffs"]) + 1.0).tolist())):
        with pytest.raises(ValueError, match="not of the same fold"):
            L.assemble_fold(d, "sim3", members[:4] + [(md, bad, meta, net)], rec, **kw)
    with pytest.raises(ValueError, match="not of the same fold"):
        L.assemble_fold(d, "sim3", members[:4] + [(md, fj, dict(meta, k_com_hmpc=meta["k_com_hmpc"][::-1]), net)],
                        rec, **kw)
    seen = []

    def spy(nets, d_, held, coeffs, truth_weights=False):
        seen.append(truth_weights)
        return _fake_predict(nets, d_, held, coeffs, truth_weights=truth_weights)
    ctx = L.assemble_fold(d, "sim3", members, rec, cache_sha256="c" * 64, predict=spy, truth_weights=True)
    assert set(seen) == {True} and ctx["weights"] == "truth (context only)"
    seen.clear()
    prim = L.assemble_fold(d, "sim3", members, rec, cache_sha256="c" * 64, predict=spy)
    assert set(seen) == {False} and prim["weights"] == "head_a_per_fold"


def test_the_ensemble_is_the_usable_members_and_a_swapped_training_row_is_refused(d):
    members, rec = _assembly(d)
    scales = [1.00, 1.01, None, 1.03, 1.04]                         # member 2 predicts NaN
    members = [(md, fj, meta, (s if s is not None else "nan", None)) for (md, fj, meta, _), s in zip(members, scales)]

    def mean_of_members(nets, d_, held, coeffs, truth_weights=False):
        T = np.nan_to_num(np.asarray(d_["P_tier_p"], dtype=np.float64)[held], nan=1.0)
        if any(n[0] == "nan" for n in nets):
            return T * np.nan, {}
        return T * np.mean([n[0] for n in nets]), {}
    out = L.assemble_fold(d, "sim3", members, rec, cache_sha256="c" * 64, predict=mean_of_members)
    assert out["member_ok"] == [True, True, False, True, True] and np.allclose(out["e_c1"], 0.02)
    fit = members[0][1]["fit_rows"]
    leak = sorted(fit[1:] + members[0][1]["val_rows"] + [members[0][1]["held_rows"][0]])   # same size, one held row
    with pytest.raises(ValueError, match="training rows"):
        L.assemble_fold(d, "sim3", members, dict(rec, rows_train=[leak]), cache_sha256="c" * 64,
                        predict=mean_of_members)


def test_a_gp_record_without_the_convergence_key_is_flagged(d):
    members, rec = _assembly(d)
    rec = dict(rec, per_z=[{"z": 3.0}, {"z": 3.2, "converged_final": True}])   # a per-z entry without the key
    out = L.assemble_fold(d, "sim3", members, rec, cache_sha256="c" * 64, predict=_fake_predict)
    assert out["gp_unconverged_z"] == [3.0]                        # fails closed
