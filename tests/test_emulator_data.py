import h5py, numpy as np
from tests.emulator._fixture import write_synthetic_cache

def test_fixture_is_valid_v33(tmp_path):
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3"
        R = h["P_tier_p"].shape[0]
        assert R == 3 * 2 * 4
        cnt = h["tier_c_counts"][:].astype(float)          # (R,15)
        Pf = h["P_tier_c_filtered"][:]                      # (R,15,n_k)
        Pp = h["P_tier_p"][:]                               # (R,n_k)
        w = cnt / cnt.sum(1, keepdims=True)
        recon = np.einsum("rc,rck->rk", w, Pf)
        ok = np.isfinite(Pp)
        assert np.allclose(recon[ok], Pp[ok], rtol=0, atol=1e-12)
        assert np.isnan(Pp).any()

from hcd_analysis.emulator.data import load_cache, COARSE_SLICES, _collapse_p1d

def test_empty_class_collapses_to_zero_not_nan():
    # Only the clean class (fine bin 0) is populated; LLS/subDLA/DLA are empty.
    counts15 = np.zeros((1, 15), dtype=np.int64)
    counts15[0, 0] = 100
    p15 = np.zeros((1, 15, 4))                     # finite zeros (cache convention)
    out = _collapse_p1d(p15, counts15)             # (1, 4, 4)
    # Empty LLS class -> 0.0 and finite, NOT NaN (would poison structural sum).
    assert np.isfinite(out[0, 1]).all()
    assert np.allclose(out[0, 1], 0.0)
    # Genuine all-NaN segment (above native Nyquist) -> NaN, for masking.
    p15b = p15.copy()
    p15b[0, 0, :] = np.nan                          # clean class entirely NaN
    outb = _collapse_p1d(p15b, counts15)
    assert np.isnan(outb[0, 0]).all()

def test_load_collapses_15_to_4_and_builds_tau0_masks_weights(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_k=8)
    d = load_cache(path)
    R = d["P_tier_p"].shape[0]
    assert d["P_filt"].shape == (R, 4, 8)
    assert d["delta"].shape == (R, 3, 8)
    assert np.allclose(d["tau0"], -np.log(d["target_F"]))
    assert COARSE_SLICES == (slice(0,1), slice(1,8), slice(8,13), slice(13,15))
    assert d["mask"].shape == (R, 8)
    assert d["mask"].dtype == bool
    assert (d["mask"] == np.isfinite(d["P_tier_p"])).all()
    assert d["inv_nc"].shape == (R, 4)
    assert np.all(np.isfinite(d["inv_nc"]))

from hcd_analysis.emulator.data import signed_log, signed_log_inv, fit_norm, apply_norm

def test_signed_log_roundtrip_through_zero():
    x = np.array([-5.0, -1e-9, 0.0, 1e-9, 3.0])
    assert np.allclose(signed_log_inv(signed_log(x)), x, atol=1e-12)

def test_norm_uses_only_train_rows():
    x = np.arange(20.0).reshape(10, 2)
    stats = fit_norm(x, train_idx=np.arange(5))
    assert np.allclose(stats["mean"], x[:5].mean(0))
    z = apply_norm(x, stats)
    assert np.allclose(z[:5].mean(0), 0.0, atol=1e-9)

from hcd_analysis.emulator.data import kfold_loso, tau0_edge_holdout

def test_kfold_loso_every_sim_held_once(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_sims=6, snaps_per_sim=2, n_alpha=4)
    d = load_cache(path)
    folds = kfold_loso(d["sim_name"], n_folds=3)
    held = set()
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        held |= set(d["sim_name"][va])
    assert held == set(d["sim_name"])

def test_tau0_edge_holdout_picks_extremes(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path, n_sims=4, snaps_per_sim=3, n_alpha=4)
    d = load_cache(path)
    tr, ho = tau0_edge_holdout(d["tau0"], frac=0.2)
    assert d["tau0"][ho].min() <= d["tau0"][tr].min()
    assert d["tau0"][ho].max() >= d["tau0"][tr].max()

from hcd_analysis.emulator.data import PARAM_LIMITS, normalize_params

def test_param_limits_cover_nine_params_in_cache_order():
    # PARAM_LIMITS is (9,2) lo/hi aligned to the cache param order
    assert PARAM_LIMITS.shape == (9, 2)
    assert np.all(PARAM_LIMITS[:, 1] > PARAM_LIMITS[:, 0])

def test_normalize_params_maps_to_unit_cube(tmp_path):
    path = tmp_path / "obs.h5"; write_synthetic_cache(path)   # fixture params are uniform(0.5,1.5)
    d = load_cache(path)
    # load_cache must now expose normalized model input x = [params_unit (9), z_unit (1)]
    assert "x" in d and d["x"].shape == (d["params"].shape[0], 10)
    # real-cache params (when in-domain) normalize into [0,1]; synthetic may be out-of-domain ->
    # normalize_params must still be finite and monotonic. Check a known in-range point maps right:
    lo, hi = PARAM_LIMITS[:, 0], PARAM_LIMITS[:, 1]
    mid = lo + 0.5 * (hi - lo)
    assert np.allclose(normalize_params(mid[None, :])[0], 0.5)
    assert np.allclose(normalize_params(lo[None, :])[0], 0.0)
    assert np.allclose(normalize_params(hi[None, :])[0], 1.0)


# --- A1: make_batch target-transform + train-split normalization --------------
from hcd_analysis.emulator.data import (
    fit_target_norm, make_batch, untransform_prediction, safe_log,
)


def _load_fixture(tmp_path, **kw):
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, **kw)
    return load_cache(path)


def test_make_batch_keys_and_shapes(tmp_path):
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    train_idx = np.arange(R)[: R - 4]            # hold out the last snap-block's alphas
    norm = fit_target_norm(d, train_idx)
    idx = np.array([0, 1, 5, 9, 13])
    b = make_batch(d, idx, norm)
    K = d["P_filt"].shape[2]
    n = len(idx)
    assert b["x"].shape == (n, 10)
    assert b["tau0"].shape == (n,)
    assert b["t_f_nhi"].shape == (n, 30)
    assert b["t_dndx"].shape == (n, 3)
    assert b["t_P_filt"].shape == (n, 4, K)
    assert b["t_delta"].shape == (n, 3, K)
    assert b["mask"].shape == (n, K)
    assert b["mask"].dtype == bool
    assert b["inv_nc"].shape == (n, 4)
    assert b["inv_nalpha"].shape == (n,)
    assert b["mean_F_clean"].shape == (n,)
    # all float64
    for k in ("x", "tau0", "t_f_nhi", "t_dndx", "t_P_filt", "t_delta",
              "inv_nc", "inv_nalpha", "mean_F_clean"):
        assert b[k].dtype == np.float64, k
    # x must be the unit-cube input untouched
    assert np.allclose(b["x"], d["x"][idx])
    # f_nhi/dndx (per-block) finite everywhere
    assert np.isfinite(b["t_f_nhi"]).all()
    assert np.isfinite(b["t_dndx"]).all()
    # t_P_filt / t_delta finite where mask True
    m3 = b["mask"][:, None, :]
    assert np.isfinite(b["t_P_filt"][np.broadcast_to(m3, b["t_P_filt"].shape)]).all()
    assert np.isfinite(b["t_delta"][np.broadcast_to(m3, b["t_delta"].shape)]).all()


def test_target_norm_roundtrip_to_physical(tmp_path):
    d = _load_fixture(tmp_path, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    # raw transform round-trip is exact
    from hcd_analysis.emulator.data import (
        signed_log, signed_log_inv, apply_norm, invert_norm,
    )
    P = d["P_filt"][:5]                                   # may contain NaN above Nyquist
    z = apply_norm(safe_log(P), norm["P_filt"])
    Prt = np.exp(invert_norm(z, norm["P_filt"]))
    fin = np.isfinite(P)
    assert np.allclose(Prt[fin], P[fin], atol=1e-10, rtol=1e-10)
    Dl = d["delta"][:5]
    zd = apply_norm(signed_log(Dl), norm["delta"])
    Drt = signed_log_inv(invert_norm(zd, norm["delta"]))
    find = np.isfinite(Dl)
    assert np.allclose(Drt[find], Dl[find], atol=1e-10, rtol=1e-10)
    # untransform_prediction recovers physical P_filt / delta from standardized targets
    idx = np.arange(5)
    b = make_batch(d, idx, norm)
    pred = {"f_nhi": b["t_f_nhi"], "dndx": b["t_dndx"],
            "P_filt": b["t_P_filt"], "delta": b["t_delta"]}
    phys = untransform_prediction(pred, norm)
    okP = np.isfinite(d["P_filt"][idx])
    assert np.allclose(phys["P_filt"][okP], d["P_filt"][idx][okP], atol=1e-9)
    okD = np.isfinite(d["delta"][idx])
    assert np.allclose(phys["delta"][okD], d["delta"][idx][okD], atol=1e-9)


def test_inv_nalpha_counts_alpha_siblings(tmp_path):
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    b = make_batch(d, np.arange(R), norm)
    assert np.allclose(b["inv_nalpha"], 0.25)


def test_train_norm_uses_only_train_blocks(tmp_path):
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    full = fit_target_norm(d, np.arange(R))
    # take a subset of snap-blocks (drop the last sim's rows): stats must differ
    sub = np.arange(R)[: R // 2]
    part = fit_target_norm(d, sub)
    # at least one channel's mean differs (train-only normalization)
    assert not np.allclose(part["P_filt"]["mean"], full["P_filt"]["mean"])
    assert not np.allclose(part["f_nhi"]["mean"], full["f_nhi"]["mean"])
    assert not np.allclose(part["delta"]["mean"], full["delta"]["mean"])
