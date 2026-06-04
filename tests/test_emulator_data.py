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


def test_fit_norm_relative_std_floor_lifts_collapsed_columns():
    """min_std_frac floors a near-degenerate column's std at a fraction of the
    channel median, so a standardized value of an off-train sample no longer
    explodes — while well-conditioned columns stay BIT-IDENTICAL (floor only
    raises, never lowers). Mirrors the Δ_c std-collapse fix in fit_target_norm."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 200, 5
    x = rng.normal(scale=1.0, size=(n_rows, n_cols))
    # column 0 is a COLLAPSED bin: ~constant on train -> std ~1e-7
    x[:, 0] = 1.0 + 1e-7 * rng.normal(size=n_rows)
    tr = np.arange(n_rows)

    base = fit_norm(x, tr, min_std_frac=0.0)
    floored = fit_norm(x, tr, min_std_frac=0.01)

    # the collapsed column's std is lifted to the relative floor (1% of the
    # median over the populated columns), well above its raw ~1e-7.
    median_std = float(np.median(base["std"]))
    assert base["std"][0] < 1e-5                       # raw std is collapsed
    assert floored["std"][0] >= 0.01 * median_std * (1 - 1e-9)
    assert floored["std"][0] > base["std"][0] * 100    # genuinely raised
    # well-conditioned columns (std >> floor) are untouched, bit-for-bit
    wc = base["std"] > 10 * 0.01 * median_std
    assert wc.sum() >= 3
    assert np.array_equal(floored["std"][wc], base["std"][wc])
    assert np.array_equal(floored["mean"], base["mean"])  # mean never touched

    # a normal off-train sample standardizes sanely under the floor (no blow-up).
    probe = np.zeros(n_cols)                            # ~3.5e6 sigma raw on col 0
    z_raw = np.abs((probe - base["mean"]) / base["std"])[0]
    z_floored = np.abs((probe - floored["mean"]) / floored["std"])[0]
    assert z_raw > 1e4 and z_floored < 1e3 and z_floored < z_raw

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


from hcd_analysis.emulator.data import make_splits


def test_make_splits_holdout_disjoint_from_train_and_val(tmp_path):
    """make_splits: holdout disjoint from BOTH train and val; train/val disjoint;
    union covers all rows; LOSO (no sim straddles train/val)."""
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=8, snaps_per_sim=2, n_alpha=4)
    d = load_cache(path)
    n_rows = len(d["tau0"])
    for fold in range(8):
        tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
        s_tr, s_va, s_ho = set(tr), set(va), set(ho)
        assert s_ho.isdisjoint(s_tr), fold
        assert s_ho.isdisjoint(s_va), fold
        assert s_tr.isdisjoint(s_va), fold
        # union covers every row exactly once
        assert s_tr | s_va | s_ho == set(range(n_rows)), fold
        assert len(tr) + len(va) + len(ho) == n_rows, fold
        # LOSO: no sim appears in both train and val
        assert set(d["sim_name"][tr]).isdisjoint(set(d["sim_name"][va])), fold


import os
import pytest

_REAL_LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


@pytest.mark.skipif(not os.path.exists(_REAL_LF_CACHE),
                    reason="real merged LF cache not present")
def test_make_splits_disjoint_on_real_lf_cache():
    """A-list referee guard: run make_splits' disjointness/LOSO/union invariants on
    the REAL merged LF cache (not just the synthetic fixture), since the production
    sweep splits this exact cache. Spot-checks a few folds across the LOSO set."""
    d = load_cache(_REAL_LF_CACHE)
    n_rows = len(d["tau0"])
    n_folds = len(kfold_loso(d["sim_name"], n_folds=8))
    for fold in (0, n_folds // 2, n_folds - 1):
        tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
        s_tr, s_va, s_ho = set(tr.tolist()), set(va.tolist()), set(ho.tolist())
        assert s_ho.isdisjoint(s_tr), fold
        assert s_ho.isdisjoint(s_va), fold
        assert s_tr.isdisjoint(s_va), fold
        assert s_tr | s_va | s_ho == set(range(n_rows)), fold
        assert len(tr) + len(va) + len(ho) == n_rows, fold
        assert set(d["sim_name"][tr]).isdisjoint(set(d["sim_name"][va])), fold


def test_make_splits_matches_inline_cli_logic(tmp_path):
    """make_splits reproduces the old inline tau0-holdout x LOSO composition."""
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=6, snaps_per_sim=2, n_alpha=4)
    d = load_cache(path)
    fold, n_folds, frac = 1, 4, 0.15
    # old inline logic
    tr_pool, holdout = tau0_edge_holdout(d["tau0"], frac=frac)
    pool_mask = np.zeros(len(d["tau0"]), bool); pool_mask[tr_pool] = True
    folds = kfold_loso(d["sim_name"], n_folds=n_folds)
    tr0, va0 = folds[fold]
    tr0 = tr0[pool_mask[tr0]]; va0 = va0[pool_mask[va0]]
    # new helper
    tr1, va1, ho1 = make_splits(d, fold, n_folds=n_folds, holdout_frac=frac)
    assert np.array_equal(np.sort(tr0), np.sort(tr1))
    assert np.array_equal(np.sort(va0), np.sort(va1))
    assert np.array_equal(np.sort(holdout), np.sort(ho1))


from hcd_analysis.emulator.data import _valid_target_mask


def test_valid_target_mask_shared(tmp_path):
    """M2: make_batch's f_nhi/dndx mask == fit_target_norm's masking predicate
    (both go through the single _valid_target_mask helper)."""
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    d["snap_f_nhi"][:, 0] = 0.0        # structural zero -> invalid
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.arange(R)
    b = make_batch(d, idx, norm)
    grp = d["snap_group_idx"][idx]
    # the predicate make_batch emits must equal _valid_target_mask on the same data
    assert np.array_equal(b["t_f_nhi_mask"], _valid_target_mask(d["snap_f_nhi"][grp]))
    assert np.array_equal(b["t_dndx_mask"], _valid_target_mask(d["snap_dNdX"][grp]))
    # and the helper is the one fit_target_norm uses for f_nhi/dndx validity
    assert np.array_equal(_valid_target_mask(d["snap_f_nhi"]),
                          np.isfinite(d["snap_f_nhi"]) & (d["snap_f_nhi"] > 0))

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
    fit_baseline_residual_norm, reconstruct_P_filt, cell_id,
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
    # REDESIGN: t_P_filt replaced by t_p_base + t_p_resid (both (n,4,K)).
    assert b["t_p_base"].shape == (n, 4, K)
    assert b["t_p_resid"].shape == (n, 4, K)
    assert "t_P_filt" not in b
    assert b["t_delta"].shape == (n, 3, K)
    assert b["mask"].shape == (n, K)
    assert b["mask"].dtype == bool
    assert b["inv_nc"].shape == (n, 4)
    assert b["inv_nalpha"].shape == (n,)
    assert b["mean_F_clean"].shape == (n,)
    # all float64
    for k in ("x", "tau0", "t_f_nhi", "t_dndx", "t_p_base", "t_p_resid", "t_delta",
              "inv_nc", "inv_nalpha", "mean_F_clean"):
        assert b[k].dtype == np.float64, k
    # x must be the unit-cube input untouched
    assert np.allclose(b["x"], d["x"][idx])
    # f_nhi/dndx (per-block) finite everywhere
    assert np.isfinite(b["t_f_nhi"]).all()
    assert np.isfinite(b["t_dndx"]).all()
    # t_p_base / t_p_resid / t_delta finite where mask True
    m3 = b["mask"][:, None, :]
    assert np.isfinite(b["t_p_base"][np.broadcast_to(m3, b["t_p_base"].shape)]).all()
    assert np.isfinite(b["t_p_resid"][np.broadcast_to(m3, b["t_p_resid"].shape)]).all()
    assert np.isfinite(b["t_delta"][np.broadcast_to(m3, b["t_delta"].shape)]).all()


def test_edge_emphasis_k_weight_shape_and_profile():
    """edge_emphasis_k_weight: mean-1 over finite bins, U-shaped (edges > mid in log k),
    all-ones when both gains are 0, and a noop on non-finite/zero k bins."""
    from hcd_analysis.emulator.data import edge_emphasis_k_weight
    kf = np.geomspace(4e-4, 7e-2, 172)
    w = edge_emphasis_k_weight(kf, edge_gain=3.0, lowk_extra=2.0)
    assert w.shape == kf.shape
    assert np.isclose(w.mean(), 1.0, atol=1e-12)         # mean-1 normalization
    # both EDGES out-weight the geometric-centre bin (U-shape in log k)
    mid = int(np.argmin(np.abs(np.log10(kf) - np.log10(np.sqrt(kf[0] * kf[-1])))))
    assert w[0] > w[mid] and w[-1] > w[mid]
    # the lowest-k bin (where A_p's coherent residual lives) is up-weighted vs mid
    assert w[0] > 1.0
    # uniform (both gains 0) -> all ones
    w0 = edge_emphasis_k_weight(kf, edge_gain=0.0, lowk_extra=0.0)
    assert np.allclose(w0, 1.0)
    # non-finite / zero k bins fall back to weight 1
    kbad = kf.copy(); kbad[3] = np.nan; kbad[7] = 0.0
    wb = edge_emphasis_k_weight(kbad, edge_gain=3.0, lowk_extra=2.0)
    assert wb[3] == 1.0 and wb[7] == 1.0


def test_make_batch_carries_k_weight_per_row(tmp_path):
    """make_batch(k_weight=...) tiles the (K,) profile to a per-row (n,K) array so it
    pads/batches like every other array; absent the arg there is no k_weight key."""
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.array([0, 1, 5, 9])
    K = d["P_filt"].shape[2]
    # without k_weight: key absent (back-compat)
    assert "k_weight" not in make_batch(d, idx, norm)
    # with k_weight: tiled to (n, K), each row identical
    kw = np.linspace(0.5, 2.0, K)
    b = make_batch(d, idx, norm, k_weight=kw)
    assert b["k_weight"].shape == (len(idx), K)
    assert b["k_weight"].dtype == np.float64
    for r in range(len(idx)):
        assert np.allclose(b["k_weight"][r], kw)


def test_target_norm_roundtrip_to_physical(tmp_path):
    d = _load_fixture(tmp_path, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    # raw delta transform round-trip is exact (delta keeps the flat-norm path)
    from hcd_analysis.emulator.data import (
        signed_log, signed_log_inv, apply_norm, invert_norm,
    )
    Dl = d["delta"][:5]
    zd = apply_norm(signed_log(Dl), norm["delta"])
    Drt = signed_log_inv(invert_norm(zd, norm["delta"]))
    find = np.isfinite(Dl)
    assert np.allclose(Drt[find], Dl[find], atol=1e-10, rtol=1e-10)
    # untransform_prediction recovers physical P_filt (from baseline+residual targets)
    # and delta from standardized targets.
    idx = np.arange(5)
    b = make_batch(d, idx, norm)
    pred = {"f_nhi": b["t_f_nhi"], "dndx": b["t_dndx"],
            "P_filt_base": b["t_p_base"], "P_filt_resid": b["t_p_resid"],
            "delta": b["t_delta"]}
    phys = untransform_prediction(pred, norm)
    okP = np.isfinite(d["P_filt"][idx])
    assert np.allclose(phys["P_filt"][okP], d["P_filt"][idx][okP], atol=1e-9)
    okD = np.isfinite(d["delta"][idx])
    assert np.allclose(phys["delta"][okD], d["delta"][idx][okD], atol=1e-9)


def test_delta_channel_std_floor_tames_collapsed_bin(tmp_path):
    """Δ_c std-collapse fix: fit_target_norm floors the delta-channel std at
    DELTA_STD_FLOOR_FRAC * median std (via fit_norm's min_std_frac), so a bin
    whose train std collapses toward zero (the LOSO low-k behaviour: ~2.5e-7,
    standardizing a normal val delta to ~9e4) no longer explodes the standardized
    target — while the well-conditioned bins stay untouched."""
    from hcd_analysis.emulator.data import (
        DELTA_STD_FLOOR_FRAC, signed_log, apply_norm,
    )
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    # Force a COLLAPSED delta bin (class 0, k 0): nearly identical across all rows
    # (the LOSO low-k delta-std-collapse the fix targets). delta is (R,3,K).
    d["delta"] = d["delta"].copy()
    d["delta"][:, 0, 0] = 1.0 + 1e-7 * np.arange(d["delta"].shape[0])
    R = d["x"].shape[0]

    norm = fit_target_norm(d, np.arange(R))
    std = norm["delta"]["std"]
    finite = np.isfinite(std) & (std >= 1e-12)
    floor = DELTA_STD_FLOOR_FRAC * float(np.median(std[finite]))

    # the collapsed bin's std is lifted to (at least) the relative floor.
    coll = std.reshape(3, -1)[0, 0]
    assert coll >= floor * (1 - 1e-9)
    assert coll > 1e-6                                  # well above the raw ~1e-7 collapse
    # every populated std now respects the floor (no near-zero std survives).
    assert np.all(std[finite] >= floor * (1 - 1e-9))

    # a normal off-train delta standardizes sanely (no ~9e4 blow-up) on that bin.
    probe = np.zeros((1,) + d["delta"].shape[1:])      # delta=0 everywhere
    z = apply_norm(signed_log(probe), norm["delta"])
    assert np.abs(z.reshape(1, 3, -1)[0, 0, 0]) < 1e4
    assert np.isfinite(z).all()


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
    # at least one channel's mean differs (train-only normalization). P_filt now
    # holds the structured baseline/residual stats; check its marginal mean (mu_marg).
    assert not np.allclose(part["P_filt"]["mu_marg"], full["P_filt"]["mu_marg"])
    assert not np.allclose(part["f_nhi"]["mean"], full["f_nhi"]["mean"])
    assert not np.allclose(part["delta"]["mean"], full["delta"]["mean"])


def test_target_norm_no_val_leak(tmp_path):
    """STRICT leak guard: corrupting val rows (and their snap-blocks) must NOT
    move any train-only norm stat. Pins both the per-row (P_filt/delta) and the
    per-block (f_nhi/dndx via snap_group_idx) channels against val/test leakage."""
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    val = d["sim_name"] == "sim3"
    train_idx = np.where(~val)[0]
    norm = fit_target_norm(d, train_idx)

    # stats must equal an independent train-only computation (no val rows used).
    # P_filt now holds the structured baseline/residual stats; mu_marg is the
    # marginal per-(c,k) mean of logP over train rows.
    exp_pf = np.nanmean(safe_log(d["P_filt"][train_idx]), axis=0)
    assert np.allclose(norm["P_filt"]["mu_marg"], exp_pf, equal_nan=True)
    train_blocks = np.unique(d["snap_group_idx"][train_idx])
    exp_f = np.nanmean(safe_log(d["snap_f_nhi"][train_blocks]), axis=0)
    assert np.allclose(norm["f_nhi"]["mean"], exp_f, equal_nan=True)

    # corrupt ONLY val rows + the snap-blocks they (and only they) map into
    d2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
    d2["P_filt"][val] *= 1e6
    d2["delta"][val] *= 1e6
    val_blocks = np.unique(d["snap_group_idx"][np.where(val)[0]])
    train_only_blocks = np.setdiff1d(val_blocks, train_blocks)  # blocks no train row touches
    d2["snap_f_nhi"][train_only_blocks] *= 1e6
    d2["snap_dNdX"][train_only_blocks] *= 1e6
    norm2 = fit_target_norm(d2, train_idx)
    for ch in ("f_nhi", "dndx", "delta"):
        assert np.allclose(norm[ch]["mean"], norm2[ch]["mean"], equal_nan=True), ch
        assert np.allclose(norm[ch]["std"], norm2[ch]["std"], equal_nan=True), ch
    # P_filt structured stats: marginal + cosmo-scale unchanged by val corruption
    for stat in ("mu_marg", "sig_marg", "sig_cosmo"):
        assert np.allclose(norm["P_filt"][stat], norm2["P_filt"][stat], equal_nan=True), stat
    # the per-cell means dict is over the SAME train cells with the SAME values
    assert set(norm["P_filt"]["cell_mean"]) == set(norm2["P_filt"]["cell_mean"])
    for cid, cm in norm["P_filt"]["cell_mean"].items():
        assert np.allclose(cm, norm2["P_filt"]["cell_mean"][cid], equal_nan=True), cid


def test_fnhi_zero_bin_floor_is_bounded_and_invertible(tmp_path):
    """The real cache has STRUCTURAL zeros in f_nhi (CDDF): bin 0 is all-zero and
    the high-NHI tail bins are partially zero. safe_log sends those to the floor
    log(1e-30) ~= -69.08, which (a) must stay finite through normalization and
    (b) must invert to ~0 (not NaN/inf). This pins the floor behaviour so a future
    floor change can't silently corrupt the CDDF target norm.

    A4b: the floor is now MASKED out of the norm fit (the zero bins are excluded
    from mean/std), so a fully-zero bin's norm falls back to the neutral 0/1
    guard and is NOT floor-contaminated. See
    test_fnhi_zero_bin_masked_out_of_norm_and_loss for the masking proof."""
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    # inject the real-cache structure: bin 0 fully zero, last bin partially zero
    d["snap_f_nhi"][:, 0] = 0.0
    d["snap_f_nhi"][::2, -1] = 0.0
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    # all norm stats finite (floor did not produce NaN/inf)
    assert np.isfinite(norm["f_nhi"]["mean"]).all()
    assert np.isfinite(norm["f_nhi"]["std"]).all() and np.all(norm["f_nhi"]["std"] > 0)
    # round-trip of a zero bin returns ~0 (floor 1e-30), never NaN/inf
    b = make_batch(d, np.arange(R), norm)
    phys = untransform_prediction({"f_nhi": b["t_f_nhi"]}, norm)
    zero_mask = d["snap_f_nhi"][d["snap_group_idx"]] == 0.0
    assert np.all(np.abs(phys["f_nhi"][zero_mask]) <= 1e-29)
    assert np.isfinite(phys["f_nhi"]).all()


def test_fnhi_zero_bin_masked_out_of_norm_and_loss(tmp_path):
    """A4b: a fully-zero f_nhi bin is (a) MASKED out of the norm fit so its
    mean/std are NOT floor-contaminated (mean != log(1e-30)), and (b) emitted in
    make_batch's t_f_nhi_mask as False so it contributes ZERO gradient to the
    Head-A loss. Compare against the pre-fix floor-contaminated values to prove
    the mask actually changed the norm."""
    import jax, jax.numpy as jnp
    from hcd_analysis.emulator.model import joint_loss, Emulator
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    d["snap_f_nhi"][:, 0] = 0.0                     # bin 0 structurally all-zero
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    floor = safe_log(0.0)

    # (a) bin 0 norm is the neutral fallback (no valid data), NOT the floor value
    assert not np.isclose(norm["f_nhi"]["mean"][0], floor)
    assert np.isclose(norm["f_nhi"]["mean"][0], 0.0)   # fit_norm neutral fallback
    assert np.isclose(norm["f_nhi"]["std"][0], 1.0)
    # a nonzero bin's norm is a real (non-floor, non-neutral) statistic
    assert norm["f_nhi"]["mean"][1] != 0.0

    # (b) make_batch masks bin 0 out of the Head-A loss
    b = make_batch(d, np.arange(R), norm)
    assert b["t_f_nhi_mask"].shape == b["t_f_nhi"].shape
    assert not b["t_f_nhi_mask"][:, 0].any()           # bin 0 masked everywhere
    assert b["t_f_nhi_mask"][:, 1:].all()              # other bins unmasked

    # the masked bin contributes zero gradient: corrupt the bin-0 TARGET only;
    # loss & model-grad must be unchanged (masked out).
    bj = {k: (jnp.asarray(v) if hasattr(v, "dtype") else v) for k, v in b.items()}
    m = Emulator(in_dim=10, n_k=d["P_filt"].shape[2], key=jax.random.PRNGKey(0))
    v0, g0 = jax.value_and_grad(lambda mm: joint_loss(mm, bj))(m)
    t2 = bj["t_f_nhi"].at[:, 0].set(1e6)
    bj2 = dict(bj); bj2["t_f_nhi"] = t2
    v1, g1 = jax.value_and_grad(lambda mm: joint_loss(mm, bj2))(m)
    import equinox as eqx
    assert jnp.array_equal(v0, v1)
    for a, c in zip(jax.tree_util.tree_leaves(eqx.filter(g0, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array))):
        assert jnp.array_equal(a, c)


# ---------------------------------------------------------------------------
# Phase-2b normalization REDESIGN: θ-blind baseline + σ_cosmo-whitened residual.
# ---------------------------------------------------------------------------

def test_cell_id_groups_same_z_alpha(tmp_path):
    """cell_id encodes (round(z,4), alpha_idx): two rows share a cell iff same
    rounded z AND same alpha_idx; distinct (z,α) -> distinct ids."""
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    cid = cell_id(d)
    R = len(d["z_grid"])
    assert cid.shape == (R,)
    # n distinct cells == n distinct (round(z,4), alpha_idx) pairs
    pairs = set(zip(np.round(d["z_grid"], 4).tolist(), d["alpha_idx"].tolist()))
    assert len(np.unique(cid)) == len(pairs)
    # rows with the same (z,α) get the same id; different (z,α) differ
    for i in range(R):
        for j in range(R):
            same = (round(float(d["z_grid"][i]), 4) == round(float(d["z_grid"][j]), 4)
                    and d["alpha_idx"][i] == d["alpha_idx"][j])
            assert (cid[i] == cid[j]) == same


def test_residual_target_unit_variance(tmp_path):
    """REDESIGN core claim: the σ_cosmo-whitened residual target has ~O(1) std (the
    re-whitening works) and sig_cosmo < sig_marg per (c,k) (the burial factor — the
    cosmology signal is a small slice of the marginal spread)."""
    d = _load_fixture(tmp_path, n_sims=6, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    pf = norm["P_filt"]
    # pf computed standalone must match the one inside fit_target_norm
    assert np.allclose(pf["sig_cosmo"],
                       fit_baseline_residual_norm(d, np.arange(R))["sig_cosmo"],
                       equal_nan=True)
    b = make_batch(d, np.arange(R), norm)
    tpr = np.asarray(b["t_p_resid"])
    fin = np.isfinite(tpr)
    std = np.nanstd(tpr[fin])
    # ~unit variance: O(1), within a generous band (the fixture is random, not
    # physical, but the whitening still pins it to ~1).
    assert 0.2 < std < 5.0, std
    # sig_cosmo < sig_marg per (c,k) wherever there is genuine within-cell spread
    # (excludes the neutral sig_cosmo==1 fallback bins with no cosmology signal).
    sc, sm = pf["sig_cosmo"], pf["sig_marg"]
    real = (sc != 1.0) & np.isfinite(sm) & (sm > 1e-9)
    assert real.any()
    assert np.all(sc[real] < sm[real] + 1e-9)
    # median burial ratio < 1 (sig_cosmo is a slice of sig_marg)
    ratio = sc[real] / sm[real]
    assert np.median(ratio) < 1.0


def test_cell_mean_covers_all_cells_under_loso(tmp_path):
    """Under LOSO every VAL row's (z,τ₀)-cell exists in the TRAIN cells (LOSO holds
    out SIMS, not cells), so make_batch never has to fall back to mu_marg."""
    d = _load_fixture(tmp_path, n_sims=8, snaps_per_sim=2, n_alpha=4, n_k=8)
    for fold in range(8):
        tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
        pf = fit_baseline_residual_norm(d, tr)
        train_cells = set(pf["cell_mean"].keys())
        val_cells = set(int(c) for c in cell_id(d, va))
        assert val_cells <= train_cells, (fold, val_cells - train_cells)


def test_reconstruction_roundtrip(tmp_path):
    """REDESIGN: given t_p_base, t_p_resid (built from a known logP) and the stats,
    reconstruct_P_filt / untransform_prediction recover the original LINEAR P_filt to
    ~1e-10. Pins logP̂ = (m̂·sig_marg+mu_marg) + sig_cosmo·r̂ -> exp as the exact
    inverse of make_batch's split."""
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.arange(R)
    b = make_batch(d, idx, norm)
    # direct reconstruct_P_filt
    P_rt = reconstruct_P_filt(b["t_p_base"], b["t_p_resid"], norm["P_filt"])
    ok = np.isfinite(d["P_filt"][idx])
    assert np.allclose(P_rt[ok], d["P_filt"][idx][ok], atol=1e-10, rtol=1e-10)
    # via untransform_prediction (the inference entry point)
    phys = untransform_prediction(
        {"P_filt_base": b["t_p_base"], "P_filt_resid": b["t_p_resid"]}, norm)
    assert np.allclose(phys["P_filt"][ok], d["P_filt"][idx][ok], atol=1e-10, rtol=1e-10)


def test_loso_assert_fires_off_cells():
    """fit_baseline_residual_norm asserts every cell has a train sim under LOSO.
    Here we feed a NON-LOSO split (a whole cell missing from train) and confirm the
    cell_mean dict simply omits it (make_batch then warns + falls back to mu_marg).
    This pins the documented fallback contract."""
    import warnings as _w
    import tempfile, os
    from tests.emulator._fixture import write_synthetic_cache
    from hcd_analysis.emulator.data import load_cache
    p = os.path.join(tempfile.mkdtemp(), "obs.h5")
    write_synthetic_cache(p, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    d = load_cache(p)
    cid = cell_id(d)
    # train = all rows EXCEPT one whole cell; val = that cell -> not in train_cells
    target_cell = int(cid[0])
    train_idx = np.where(cid != target_cell)[0]
    val_idx = np.where(cid == target_cell)[0]
    norm = fit_target_norm(d, train_idx)
    assert target_cell not in norm["P_filt"]["cell_mean"]
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        make_batch(d, val_idx, norm)
    assert any("fell back to mu_marg" in str(w.message) for w in rec)


# --- FINALIZED RECIPE: cell/n_cells emission + data-range scoping --------------

def test_make_batch_emits_cell_and_n_cells(tmp_path):
    """make_batch must carry the global (z,τ₀)-cell id per row + the STATIC scalar
    n_cells = n_z·n_alpha (the segment count for the coherent de-bias term). cell ids
    must be in [0, n_cells) and match cell_id(d, idx)."""
    from hcd_analysis.emulator.data import cell_id
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.array([0, 1, 5, 9, 13, 20])
    b = make_batch(d, idx, norm)
    assert "cell" in b and "n_cells" in b
    assert b["cell"].shape == (len(idx),)
    assert np.array_equal(b["cell"], cell_id(d, idx).astype(np.int32))
    n_z = len(np.unique(np.round(d["z_grid"], 4)))
    n_alpha = int(d["alpha_idx"].max()) + 1
    assert int(b["n_cells"]) == n_z * n_alpha
    assert b["cell"].max() < int(b["n_cells"]) and b["cell"].min() >= 0
    # n_cells MUST be a PLAIN PYTHON int (segment count, static), not np.int64:
    # coherent_debias_term feeds it to segment_sum as num_segments, which must be a
    # static python int under jax.jit. np.int64 traces as an int64[] leaf and raises
    # ConcretizationTypeError when a raw make_batch dict is jitted (jax-traps #25).
    # NB the old `np.asarray(...).ndim == 0` assert PASSED on np.int64 — it did not
    # catch this; the isinstance(int) check does.
    assert isinstance(b["n_cells"], int) and not isinstance(b["n_cells"], bool)


def test_coherent_debias_term_jits_on_raw_make_batch(tmp_path):
    """coherent_debias_term under eqx.filter_jit on a RAW make_batch dict (NO
    _to_jnp_batch).

    REGRESSION (jax-traps #25): the upcoming likelihood/sampler will jit a raw
    make_batch batch directly — it does NOT route through train._to_jnp_batch /
    _pad_batch, which special-case n_cells to a python int. eqx.filter_jit keeps
    non-array leaves STATIC, so a PYTHON-int n_cells stays a static num_segments and
    the `int(batch["n_cells"])` inside coherent_debias_term is legal. An np.int64 is an
    array leaf (eqx.is_array True), so it would be TRACED and raise
    ConcretizationTypeError here — which is exactly the bug make_batch's
    `int(n_cells)` fix prevents. (Verified: this body raises under the old np.int64
    emission and passes with the python-int fix.)"""
    import jax, equinox as eqx
    from hcd_analysis.emulator.model import Emulator, coherent_debias_term
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.array([0, 1, 5, 9, 13, 20])
    b = make_batch(d, idx, norm)                 # RAW dict (np arrays + python-int n_cells)
    n_k = d["P_filt"].shape[2]
    model = Emulator(in_dim=10, n_k=n_k, n_basis=4, key=jax.random.PRNGKey(0))
    preds = jax.vmap(model)(b["x"], b["tau0"])   # P_filt_resid etc. (B,4,K)
    # filter_jit WITHOUT _to_jnp_batch — python-int n_cells stays a static arg.
    val = eqx.filter_jit(coherent_debias_term)(preds, b)
    assert np.isfinite(float(val))


def test_datarange_mask_z_and_k_cut(tmp_path):
    """datarange_mask: in-range iff z∈[z_lo,z_hi] AND k≥k_min. Defaults z∈[2.2,4.6],
    k≥1e-3. NaN k -> out-of-range. Custom bounds honored."""
    from hcd_analysis.emulator.data import datarange_mask, DATA_RANGE
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    # fixture z_grid is {2.0, 2.4}; k grid linspace(1e-3, 0.1, 8) (all >= 1e-3)
    m = datarange_mask(d)                                # defaults
    z = d["z_grid"]; kf = d["kfkms"]
    in_z = (z >= DATA_RANGE["z_lo"] - 1e-9) & (z <= DATA_RANGE["z_hi"] + 1e-9)
    in_k = np.isfinite(kf) & (kf >= DATA_RANGE["k_min"])
    assert np.array_equal(m, in_z[:, None] & in_k)
    # z=2.0 rows are out-of-range (below 2.2); z=2.4 rows in-range
    assert not m[z < 2.2].any()
    assert m[np.isclose(z, 2.4)].any()
    # custom k_min above all k -> all out-of-range
    m_hik = datarange_mask(d, k_min=1.0)
    assert not m_hik.any()
    # custom z range that includes 2.0 -> those rows now in-range (k passes)
    m_loz = datarange_mask(d, z_lo=1.9, z_hi=5.0)
    assert m_loz[np.isclose(z, 2.0)].any()


def test_datarange_loss_weight_soft_factor(tmp_path):
    """datarange_loss_weight: in-range bins -> 1.0, out-of-range -> oor_weight (~0.2,
    a SOFT factor, not 0). Shape (n,K); configurable factor."""
    from hcd_analysis.emulator.data import (
        datarange_loss_weight, datarange_mask, DATARANGE_OOR_WEIGHT)
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    idx = np.arange(d["x"].shape[0])
    w = datarange_loss_weight(d, idx)
    K = d["kfkms"].shape[1]
    assert w.shape == (len(idx), K) and w.dtype == np.float64
    in_range = datarange_mask(d)[idx]
    assert np.allclose(w[in_range], 1.0)
    assert np.allclose(w[~in_range], DATARANGE_OOR_WEIGHT)
    assert DATARANGE_OOR_WEIGHT > 0.0          # SOFT, not a hard zero
    # custom factor
    w2 = datarange_loss_weight(d, idx, oor_weight=0.5)
    assert np.allclose(w2[~in_range], 0.5)


def test_make_batch_datarange_weight_emission(tmp_path):
    """make_batch(datarange=True) carries datarange_weight (n,K); absent by default."""
    from hcd_analysis.emulator.data import datarange_loss_weight
    d = _load_fixture(tmp_path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    norm = fit_target_norm(d, np.arange(d["x"].shape[0]))
    idx = np.array([0, 1, 5, 9, 13])
    b_off = make_batch(d, idx, norm)
    assert "datarange_weight" not in b_off
    b_on = make_batch(d, idx, norm, datarange=True)
    assert b_on["datarange_weight"].shape == (len(idx), d["kfkms"].shape[1])
    assert np.array_equal(b_on["datarange_weight"], datarange_loss_weight(d, idx))
