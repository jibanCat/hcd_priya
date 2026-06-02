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


def test_target_norm_no_val_leak(tmp_path):
    """STRICT leak guard: corrupting val rows (and their snap-blocks) must NOT
    move any train-only norm stat. Pins both the per-row (P_filt/delta) and the
    per-block (f_nhi/dndx via snap_group_idx) channels against val/test leakage."""
    d = _load_fixture(tmp_path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=8)
    val = d["sim_name"] == "sim3"
    train_idx = np.where(~val)[0]
    norm = fit_target_norm(d, train_idx)

    # stats must equal an independent train-only computation (no val rows used)
    exp_pf = np.nanmean(safe_log(d["P_filt"][train_idx]), axis=0)
    assert np.allclose(norm["P_filt"]["mean"], exp_pf, equal_nan=True)
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
    for ch in ("f_nhi", "dndx", "P_filt", "delta"):
        assert np.allclose(norm[ch]["mean"], norm2[ch]["mean"], equal_nan=True), ch
        assert np.allclose(norm[ch]["std"], norm2[ch]["std"], equal_nan=True), ch


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
