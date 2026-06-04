"""v3.3 cache loader for the Phase-2b emulator. Returns numpy arrays (host side).

15->4 coarse-class collapse, tau0 coordinate, native-Nyquist NaN masks, and the
analytic 1/n_c sample-variance weights. See spec sec.4, sec.7, sec.9.

Empty-class contract: a coarse class with zero counts collapses to 0.0 (the
cache stores finite zeros for empty classes), which keeps the structural sum
``P_tier_p = Σ_c w_c·P_filt`` finite (its ``w_c=0`` would turn any NaN into NaN
via ``0·NaN``). Only a bin that is NaN across all fine sub-bins (above native
Nyquist) stays NaN, so it can be masked out downstream.
"""
from __future__ import annotations
import warnings
import h5py, numpy as np

COARSE_SLICES = (slice(0, 1), slice(1, 8), slice(8, 13), slice(13, 15))
COARSE_NAMES = ("clean", "LLS", "subDLA", "DLA")

# --- Unit-cube input normalization (spec sec.10 / preprocessing audit) --------
# PRIYA's design box for the production grid, pulled verbatim from the saved
# emulator config that THIS cache reproduces:
#   /home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json
#   ("param_names":{ns:0,Ap:1,herei:2,heref:3,alphaq:4,hub:5,omegamh2:6,
#                    hireionz:7,bhfeedback:8}, "param_limits":[...])
# This is the actual Emulator.param_limits serialized for this grid (it widens
# the coarse_grid.py code defaults at L93-124, e.g. ns_hi=1.05, herei_hi=4.5).
# All 60 PRIYA design points map into [0,1] under these limits (verified).
# Aligned BY NAME to OUR cache param order
# [ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback]
# (== PRIYA's param_names order, so the alignment is the identity).
PARAM_LIMITS = np.array([
    [0.8,    1.05],     # ns        (emulator_params.json param_limits[0])
    [1.2e-9, 2.6e-9],   # Ap        (param_limits[1])
    [3.5,    4.5],      # herei     (param_limits[2])
    [2.2,    3.2],      # heref     (param_limits[3])
    [1.3,    3.0],      # alphaq    (param_limits[4])
    [0.65,   0.75],     # hub       (param_limits[5])
    [0.14,   0.146],    # omegamh2  (param_limits[6])
    [6.5,    8.0],      # hireionz  (param_limits[7])
    [0.03,   0.07],     # bhfeedback(param_limits[8])
], dtype=np.float64)

# PRIYA zout grid range (coarse_grid.py L153-154: max_z=5.4, min_z=2.0).
Z_LIMITS = (2.0, 5.4)

# DESI DR1 P1D DATA RANGE (the modes the data actually constrain). The cache
# footprint is wider (z∈{2.0..5.4}, k∈[4.4e-4..0.076]); z=2.0 and z>4.6 are
# out-of-range, and the lowest k bins (<1e-3) are below DESI's k_min. The training
# loss SOFT down-weights out-of-range bins (keeps their regularizing signal but
# focuses capacity in-range), and the emulator-error budget (C_emu) is RESTRICTED to
# this range so it only covers modes the data constrain. Configurable everywhere.
DATA_RANGE = {"z_lo": 2.2, "z_hi": 4.6, "k_min": 1e-3}
# soft out-of-range down-weight factor (NOT a hard zero — z>4.6's signal still
# regularizes the encoder, just gets ~5× less say; validated factor).
DATARANGE_OOR_WEIGHT = 0.2


def normalize_params(params):
    """Map raw params (...,9) to the unit cube (...,9) via PRIYA's design box.

    Vectorized, finite, and monotonic. Values outside the design box map
    outside [0,1] (caller can flag via the in-domain check)."""
    lo = PARAM_LIMITS[:, 0]
    hi = PARAM_LIMITS[:, 1]
    return (np.asarray(params) - lo) / (hi - lo)

def _collapse_counts(counts15):
    return np.stack([counts15[:, s].sum(1) for s in COARSE_SLICES], axis=1)

def _collapse_p1d(p15, counts15):
    """Count-weighted collapse of fine-bin P1D to coarse classes (R,15,K)->(R,4,K).

    NaN-safe over fine bins. An *empty* coarse class (zero counts) collapses to
    0.0, not NaN: the production cache stores finite zeros for empty classes
    (see ``priya_p1d._per_class_p1d_at_scale``), and the downstream structural
    sum ``P_tier_p = Σ_c w_c·P_filt`` weights an empty class by ``w_c=0`` — a
    NaN there would poison the whole total via ``0·NaN=NaN``. Only a bin that is
    NaN across *all* fine sub-bins (genuinely above native Nyquist) stays NaN so
    it can be masked.

    Corner case: when a single k mixes finite and NaN fine sub-bins within one
    coarse class, NaN sub-bins contribute 0 while their weight stays in the
    denominator, slightly under-weighting the surviving finite sub-bins. This is
    a documented, accepted approximation (no behaviour change)."""
    R, _, K = p15.shape
    out = np.full((R, 4, K), np.nan)
    for ci, s in enumerate(COARSE_SLICES):
        c = counts15[:, s].astype(float)
        w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
        seg = p15[:, s, :]
        contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
        summ = contrib.sum(1)
        # Fires only for the Nyquist-NaN case (ALL fine sub-bins NaN), NOT for
        # zero-count empties (those have finite zeros and correctly stay 0.0).
        allnan = (~np.isfinite(seg)).all(1)
        summ[allnan] = np.nan
        out[:, ci, :] = summ
    return out

def load_cache(path):
    with h5py.File(path, "r") as h:
        assert h.attrs["cache_version"] == "3.3", h.attrs.get("cache_version")
        g = lambda k: h[k][:]
        counts15 = g("tier_c_counts")
        Pf15, Pu15 = g("P_tier_c_filtered"), g("P_tier_c")
        out = dict(
            params=g("params").astype(np.float64),
            kfkms=g("kfkms").astype(np.float64),
            P_tier_p=g("P_tier_p").astype(np.float64),
            target_F=g("target_F").astype(np.float64),
            scale=g("scale").astype(np.float64),
            z_grid=g("z_grid").astype(np.float64),
            z_meta=g("z_meta").astype(np.float64),
            alpha_idx=g("alpha_idx").astype(np.int32),
            snap_group_idx=g("snap_group_idx").astype(np.int32),
            sim_name=np.array([s.decode() if isinstance(s, bytes) else s
                               for s in h["sim_name"][:]]),
            snap_dNdX=np.stack([g("snap_dNdX_LLS"), g("snap_dNdX_subDLA"),
                                g("snap_dNdX_DLA")], axis=1),
            snap_f_nhi=g("snap_f_nhi").astype(np.float64),
            snap_total_path_dX=g("snap_total_path_dX").astype(np.float64),
            snap_n_absorbers=g("snap_n_absorbers").astype(np.float64),
        )
    coarse_counts = _collapse_counts(counts15)
    Pf4 = _collapse_p1d(Pf15, counts15)
    Pu4 = _collapse_p1d(Pu15, counts15)
    out["P_filt"] = Pf4
    out["delta"] = (Pu4 - Pf4)[:, 1:, :]
    out["coarse_counts"] = coarse_counts
    out["tau0"] = -np.log(out["target_F"])
    out["mask"] = np.isfinite(out["P_tier_p"])
    N = coarse_counts.sum(1, keepdims=True)
    # Guard 0/0 (a row with zero total counts is unreachable in practice but
    # must not silently produce NaN), mirroring the inv_nc guard below.
    out["w_c_cache"] = np.where(N > 0, coarse_counts / np.maximum(N, 1), 0.0)
    out["inv_nc"] = np.where(coarse_counts > 0, 1.0 / np.maximum(coarse_counts, 1), 0.0)

    # --- Normalized encoder input x = [params_unit (9), z_unit (1)] -----------
    params_unit = normalize_params(out["params"])                  # (R,9)
    z_unit = (out["z_grid"] - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    out["params_unit"] = params_unit
    out["x"] = np.concatenate([params_unit, z_unit[:, None]], axis=1)  # (R,10)
    out["in_domain"] = np.all((params_unit >= -1e-9)
                              & (params_unit <= 1 + 1e-9), axis=1)
    n_oob = int((~out["in_domain"]).sum())
    if n_oob:
        print(f"WARN load_cache: {n_oob}/{len(out['in_domain'])} rows "
              f"out of PRIYA param domain (params_unit outside [0,1])")
    return out


# --- Channel transforms + train-split normalisation (spec sec.10) -------------

def signed_log(x):                 # sign-safe transform for Delta_c (smooth through 0)
    return np.arcsinh(x)

def signed_log_inv(y):
    return np.sinh(y)

def safe_log(x, floor=1e-30):
    return np.log(np.maximum(x, floor))

def fit_norm(x, train_idx, valid_mask=None, min_std_frac=0.0):
    """Per-column {mean,std} over train rows, NaN-aware.

    valid_mask (same shape as x, optional): elements that are False are excluded
    from the mean/std (A4b: structural-zero bins floored by safe_log must not
    contaminate the f_nhi/dN/dX norm). NaN entries are always ignored too.

    ``min_std_frac`` (default 0 == off): floor each column's std at
    ``min_std_frac * median(finite column stds)`` — a RELATIVE floor that guards
    against a near-degenerate column whose train-row std COLLAPSES toward zero
    (which would otherwise blow up the standardized target by ~1/std). The hard
    ``std >= 1e-12 -> 1.0`` fallback (an empty/constant column) is applied FIRST
    and is unaffected. The relative floor only ever RAISES a too-small std toward
    the channel scale; well-conditioned columns (std >> the floor) are untouched.
    The median is taken over the columns whose std passed the hard 1e-12 guard
    (the populated columns), so an all-empty channel falls back cleanly to 1.0.
    See the ``delta`` channel in ``fit_target_norm``: under LOSO its low-k std
    collapses to ~2.5e-7 (172/516 bins < 1e-3), standardizing a normal val delta
    to ~9e4 and corrupting the Δ_c term that feeds C_emu."""
    xt = x[train_idx]
    if valid_mask is not None:
        mt = valid_mask[train_idx]
        xt = np.where(mt, xt, np.nan)   # invalid -> NaN -> ignored by nanmean/nanstd
    with warnings.catch_warnings():
        # all-invalid columns are handled by the finite-fallback below; the
        # "empty slice" / "ddof<=0" warnings from nanmean/nanstd are expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(xt, axis=0)
        std = np.nanstd(xt, axis=0)
    # a column with no valid entries -> NaN mean/std; fall back to neutral 0/1.
    mean = np.where(np.isfinite(mean), mean, 0.0)
    populated = np.isfinite(std) & (std >= 1e-12)
    std = np.where(populated, std, 1.0)
    if min_std_frac > 0.0 and np.any(populated):
        # RELATIVE floor: a small fraction of the channel's typical (median) std.
        # Reference scale is the median over POPULATED columns only, so collapsed
        # bins don't drag the reference down and empty bins (now 1.0) don't drag
        # it up. Floor never lowers a std, only raises a collapsed one.
        floor = min_std_frac * float(np.median(std[populated]))
        std = np.maximum(std, floor)
    return {"mean": mean, "std": std}

def apply_norm(x, stats):
    return (x - stats["mean"]) / stats["std"]

def invert_norm(z, stats):
    return z * stats["std"] + stats["mean"]


# --- A1: per-channel transform registry + train-split target normalization ----
#
# joint_loss compares model predictions to standardized targets t_* that live in
# log/arcsinh space; this section is the missing PRODUCER of those targets. Each
# channel has a (forward, inverse) transform pair applied BEFORE standardization:
#   f_nhi, dndx, P_filt -> safe_log / exp          (strictly-positive amplitudes)
#   delta               -> signed_log / signed_log_inv  (arcsinh, smooth through 0)
TARGET_TRANSFORMS = {
    "f_nhi":  (safe_log, np.exp),
    "dndx":   (safe_log, np.exp),
    "P_filt": (safe_log, np.exp),
    "delta":  (signed_log, signed_log_inv),
}
TARGET_CHANNELS = ("f_nhi", "dndx", "P_filt", "delta")

# Δ_c (delta channel) std floor: under LOSO the delta std collapses at low-k
# (~2.5e-7 in ~1/3 of bins), so we floor each (c,k) std at this fraction of the
# channel's median std (see fit_norm's min_std_frac / fit_target_norm). 1% is a
# conservative relative floor: it lifts collapsed bins toward the channel scale
# without perturbing the well-conditioned bins (whose std is ≫ the floor).
DELTA_STD_FLOOR_FRAC = 0.01


def _valid_target_mask(arr):
    """Shared f_nhi/dN/dX validity predicate (M2): finite AND strictly positive.

    Single source of truth used by BOTH fit_target_norm (to exclude structural
    zeros from the norm fit) and make_batch (to mask them out of the Head-A
    loss), so the two can never desync. The CDDF/dN/dX have STRUCTURAL zeros
    (e.g. f_nhi bin 0) that safe_log floors to log(1e-30) ~= -69; this predicate
    flags exactly those (and NaN bins) as invalid."""
    return np.isfinite(arr) & (arr > 0)


def cell_id(d, idx=None):
    """Encode each row's (z,τ₀)-cell = (round(z_grid,4), alpha_idx) -> int.

    A "cell" is the conditioning coordinate the baseline head is blind-to-θ over:
    the SAME (z, τ₀-slope) shared across the 60 cosmologies. Two rows share a cell
    iff they have the same rounded z_grid AND the same alpha_idx (the τ₀-slope grid
    index). We pack (z_key, alpha_idx) into a single int64 via a stable lexicographic
    encoding (z dominates) so cells are comparable/hashable across calls.

    ``idx`` selects rows (default all). Returns an int64 array of shape (len(idx),).
    """
    if idx is None:
        idx = np.arange(len(d["z_grid"]))
    idx = np.asarray(idx)
    z_key = np.round(np.asarray(d["z_grid"])[idx], 4)
    a_idx = np.asarray(d["alpha_idx"])[idx].astype(np.int64)
    # z grid is discrete (PRIYA zout); map each distinct rounded z to a dense rank
    # so the packed key is a small, stable integer (round-off-proof vs hashing floats).
    uniq_z = np.unique(np.round(np.asarray(d["z_grid"]), 4))
    z_rank = np.searchsorted(uniq_z, z_key).astype(np.int64)   # 0..(n_z-1)
    n_alpha = int(np.asarray(d["alpha_idx"]).max()) + 1
    return z_rank * n_alpha + a_idx


def fit_baseline_residual_norm(d, train_idx):
    """Per-(c,k) baseline/residual normalization for P_filt (spec: redesign).

    Decomposes ``logP = safe_log(P_filt)`` (R,4,K) over (z,τ₀)-CELLS into:
      - ``cell_mean[cell] -> (4,K)`` : the within-cell mean of logP over TRAIN rows
        (the (z,τ₀)-CONDITIONAL MEAN, ~99.5% of the per-k log-variance).
      - ``mu_marg, sig_marg (4,K)``  : global per-(c,k) mean/std of logP over train
        rows (to σ_marg-standardize the BASELINE head output, which targets the
        cell-mean spectrum).
      - ``sig_cosmo (4,K)``          : sqrt(mean over cells of the within-cell variance
        of logP) — a per-(c,k) CONSTANT = the cosmology signal scale (the ~0.4%).

    The residual ``(logP − m[cell]) / sig_cosmo`` is ~unit variance and IS the
    cosmology signal; the baseline target ``(m[cell] − mu_marg)/sig_marg`` is the
    σ_marg-standardized cell-mean. All stats are NaN-aware (above-Nyquist NaN bins
    are ignored, reusing fit_norm's nanmean/nanstd machinery).

    Under LOSO every (z,α) cell has ≥1 train sim (LOSO holds out SIMS, not cells),
    so cells are populated by construction. This is NOT asserted — an all-NaN
    (c,k) bin within a populated cell silently falls back to ``mu_marg`` (the
    global per-(c,k) mean), so a missing/empty bin degrades gracefully rather than
    erroring. Returns ``norm_stats_pf`` =
    ``{"mu_marg","sig_marg","sig_cosmo", "cell_mean": {cell_id:(4,K)}}``.
    """
    train_idx = np.asarray(train_idx)
    logP = safe_log(d["P_filt"])                 # (R,4,K); NaN above Nyquist
    logP_tr = logP[train_idx]                    # (n,4,K)

    # marginal (global) per-(c,k) mean/std over train rows -> standardize baseline head.
    marg = fit_norm(logP_tr.reshape(len(train_idx), -1),
                    np.arange(len(train_idx)))
    shp = logP.shape[1:]                          # (4,K)
    mu_marg = marg["mean"].reshape(shp)
    sig_marg = marg["std"].reshape(shp)

    # per-cell mean over train rows + within-cell variance accumulation.
    cells_tr = cell_id(d, train_idx)
    cell_mean = {}
    # within-cell variance: average over cells of Var_within(logP) per (c,k).
    within_var_sum = np.zeros(shp)                # Σ_cells Var_within (per (c,k))
    within_var_cnt = np.zeros(shp)                # # cells contributing (per (c,k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for cid in np.unique(cells_tr):
            sel = train_idx[cells_tr == cid]
            block = logP[sel]                     # (m,4,K)
            cm = np.nanmean(block, axis=0)        # (4,K) cell mean (NaN where all-NaN)
            cm = np.where(np.isfinite(cm), cm, mu_marg)  # all-NaN bin -> fall back to mu_marg
            cell_mean[int(cid)] = cm
            wv = np.nanvar(block, axis=0)         # within-cell variance (4,K)
            ok = np.isfinite(wv)
            within_var_sum += np.where(ok, wv, 0.0)
            within_var_cnt += ok.astype(float)
    sig_cosmo = np.sqrt(np.where(within_var_cnt > 0,
                                 within_var_sum / np.maximum(within_var_cnt, 1),
                                 0.0))
    # a (c,k) with no within-cell spread (e.g. a single train sim per cell, or an
    # empty class) -> sig_cosmo 0; floor to a neutral 1.0 so the residual whitening
    # never divides by zero (those bins carry no cosmology signal anyway).
    sig_cosmo = np.where(sig_cosmo >= 1e-12, sig_cosmo, 1.0)

    return {
        "mu_marg": mu_marg,
        "sig_marg": sig_marg,
        "sig_cosmo": sig_cosmo,
        "cell_mean": cell_mean,
    }


def fit_target_norm(d, train_idx):
    """Per-channel {mean,std} in TRANSFORMED space, TRAIN ROWS ONLY (spec sec.10).

    f_nhi/dndx are per-snap-block (G rows); we restrict to the blocks that any
    train row maps into (via snap_group_idx) so block-level stats are train-only
    too. P_filt/delta are per-row (R rows). All stats are NaN-aware (NaN bins
    above native Nyquist are ignored) via fit_norm's nanmean/nanstd.

    Returns ``norm_stats`` with keys f_nhi, dndx, P_filt, delta, each a dict
    {mean, std} broadcastable against the transformed channel array.
    """
    train_idx = np.asarray(train_idx)
    train_blocks = np.unique(d["snap_group_idx"][train_idx])
    fwd = lambda ch: TARGET_TRANSFORMS[ch][0]
    # A4b: f_nhi (CDDF) and dN/dX have STRUCTURAL zeros (e.g. CDDF bin 0 is all-
    # zero) that safe_log floors to log(1e-30) ~= -69. Fit the f_nhi/dndx norm
    # over the NONZERO (and finite) bins ONLY so the floor never contaminates
    # mean/std.
    f_nhi_valid = _valid_target_mask(d["snap_f_nhi"])
    dndx_valid = _valid_target_mask(d["snap_dNdX"])
    stats = {
        "f_nhi":  fit_norm(fwd("f_nhi")(d["snap_f_nhi"]), train_blocks, f_nhi_valid),
        "dndx":   fit_norm(fwd("dndx")(d["snap_dNdX"]), train_blocks, dndx_valid),
        # P_filt: the normalization REDESIGN. norm_stats["P_filt"] is the structured
        # baseline/residual dict (mu_marg, sig_marg, sig_cosmo, cell_mean), NOT the
        # old flat {mean,std}. The θ-blind baseline head targets the σ_marg-
        # standardized cell-mean; the residual head targets the σ_cosmo-whitened
        # within-cell cosmology signal. See fit_baseline_residual_norm.
        "P_filt": fit_baseline_residual_norm(d, train_idx),
        # delta (Δ_c = P_unfiltered − P_filtered): under LOSO the low-k std
        # COLLAPSES (~2.5e-7 in 172/516 bins), so a normal val delta standardizes
        # to ~9e4 and corrupts the Δ_c → C_emu term (the fold-3 val_loss spike).
        # Floor the per-(c,k) std at 1% of the channel's median std (DELTA_STD_FLOOR_FRAC)
        # — a principled relative floor that lifts the collapsed bins to the channel
        # scale while leaving the well-conditioned bins (std ≫ floor) untouched.
        "delta":  fit_norm(fwd("delta")(d["delta"]), train_idx,
                           min_std_frac=DELTA_STD_FLOOR_FRAC),
    }
    return stats


def edge_emphasis_k_weight(kfkms, edge_gain=3.0, lowk_extra=1.0, mid_frac=0.5):
    """Per-k RESIDUAL loss weight (K,): a U-shaped low/high-k EDGE emphasis.

    The deployed residual head is flat mid-band but biased at the band EDGES — low-k
    (where the Lyα amplitude Δ²_* pivots, so A_p maps onto a coherent low-k residual)
    and high-k. This builds a smooth weight in log10(k) that is ~1 in the mid-band and
    rises toward both edges, so the optimizer (and the early-stop metric) attend MORE
    to the edges where the COHERENT bias lives.

    Profile: let u = (log10 k − log10 k_lo)/(log10 k_hi − log10 k_lo) ∈ [0,1] over the
    finite k-range. The weight is ``1 + edge_gain·g(u)`` where g(u) is a symmetric
    parabola ``(2u−1)²`` (0 at mid, 1 at both edges), PLUS an extra low-k ramp
    ``lowk_extra·(1−u)`` so the A_p-relevant low-k edge gets additional pull (A_p is
    the priority param). ``mid_frac`` is unused historically; kept for signature
    stability. The weight is normalized to mean 1 over finite bins so the overall
    p_resid term scale (hence its balance against term_w) is unchanged — only the
    RELATIVE per-k attention shifts. Non-finite/zero k bins get weight 1.

    Returns a float64 (K,) array. With edge_gain=0 and lowk_extra=0 it is all-ones
    (the uniform/back-compat case).
    """
    k = np.asarray(kfkms, dtype=np.float64)
    if k.ndim == 2:
        k = k[0]
    w = np.ones_like(k)
    good = np.isfinite(k) & (k > 0)
    if good.sum() < 2 or (edge_gain == 0.0 and lowk_extra == 0.0):
        return w
    lk = np.log10(k[good])
    u = (lk - lk.min()) / max(lk.max() - lk.min(), 1e-12)      # 0 at low-k .. 1 at high-k
    g = (2.0 * u - 1.0) ** 2                                    # symmetric edge bump
    wg = 1.0 + edge_gain * g + lowk_extra * (1.0 - u)          # +extra low-k ramp
    wg = wg / wg.mean()                                        # mean-1 over finite bins
    w[good] = wg
    return w


def datarange_mask(d, z_lo=None, z_hi=None, k_min=None):
    """Per-(row,k) boolean DATA-RANGE mask: in-range iff z∈[z_lo,z_hi] AND k≥k_min.

    The modes the DESI DR1 P1D data constrain (defaults from ``DATA_RANGE``:
    z∈[2.2,4.6], k≥1e-3 s/km). Used to RESTRICT the emulator-error budget (the
    error vector / C_emu) to the data range, and as the basis for the training
    SOFT down-weight (``datarange_loss_weight``). The z test is per-ROW (each row's
    z_grid), the k test is per-row (each row's kfkms grid).

    Returns a ``(R, K)`` bool array (R = len(d["z_grid"])). NaN k bins are treated
    as out-of-range (False)."""
    z_lo = DATA_RANGE["z_lo"] if z_lo is None else z_lo
    z_hi = DATA_RANGE["z_hi"] if z_hi is None else z_hi
    k_min = DATA_RANGE["k_min"] if k_min is None else k_min
    z = np.asarray(d["z_grid"])                                  # (R,)
    kf = np.asarray(d["kfkms"])                                  # (R,K)
    in_z = (z >= z_lo - 1e-9) & (z <= z_hi + 1e-9)               # (R,)
    in_k = np.isfinite(kf) & (kf >= k_min)                      # (R,K)
    return in_z[:, None] & in_k                                  # (R,K)


def datarange_loss_weight(d, idx, z_lo=None, z_hi=None, k_min=None,
                          oor_weight=None):
    """Per-(row,k) SOFT data-range loss weight (n,K) for rows ``idx``.

    In-range (z,k) bins get weight 1.0; OUT-of-range bins (z∉[z_lo,z_hi] OR k<k_min)
    get the soft factor ``oor_weight`` (~0.2, NOT a hard zero — keeps z>4.6's signal
    regularizing the encoder while focusing capacity in-range). Carried into the
    batch as ``datarange_weight`` and applied by ``joint_loss`` to the P_filt-channel
    (baseline/residual/delta) MSE terms. Defaults from ``DATA_RANGE`` /
    ``DATARANGE_OOR_WEIGHT``.

    Returns a float64 (n,K) array."""
    oor_weight = DATARANGE_OOR_WEIGHT if oor_weight is None else oor_weight
    idx = np.asarray(idx)
    in_range = datarange_mask(d, z_lo=z_lo, z_hi=z_hi, k_min=k_min)[idx]   # (n,K)
    return np.where(in_range, 1.0, oor_weight).astype(np.float64)


def make_batch(d, idx, norm_stats, k_weight=None, datarange=False,
               z_lo=None, z_hi=None, k_min=None, oor_weight=None):
    """Assemble the exact dict joint_loss consumes for rows ``idx`` (spec sec.4).

    Targets t_* are in standardized log/arcsinh space; bins above native Nyquist
    stay NaN (joint_loss is NaN-safe). Inputs x are the already-unit-cube encoder
    inputs and are NOT re-transformed. mean_F_clean is a documented placeholder
    (see below). Everything is float64.

    ``k_weight`` (K,), optional: a per-k RESIDUAL loss emphasis (see
    ``edge_emphasis_k_weight``). When given it is carried in the batch under
    ``k_weight`` and ``joint_loss``/``p_resid_loss`` apply it to the cosmology
    (p_resid) term. Absent -> the loss is the uniform inv_nc-weighted MSE.

    ``datarange`` (bool), optional: when True, carry the per-(row,k) SOFT data-range
    down-weight (``datarange_loss_weight``) in the batch under ``datarange_weight``,
    which ``joint_loss`` applies to the P_filt-channel terms (out-of-range bins get
    ``oor_weight``≈0.2). The z/k cut + soft factor are configurable
    (``z_lo``/``z_hi``/``k_min``/``oor_weight``; defaults from ``DATA_RANGE``).

    The batch ALSO carries the global (z,τ₀)-``cell`` id (per row, int) and the
    static scalar ``n_cells`` (= n_z·n_alpha = the segment count) so the coherent
    de-bias term (``model.coherent_debias_term``) can segment-sum per cell.
    """
    idx = np.asarray(idx)
    grp = d["snap_group_idx"][idx]                         # (n,) block index per row

    t_f_nhi = apply_norm(safe_log(d["snap_f_nhi"][grp]), norm_stats["f_nhi"])   # (n,30)
    t_dndx = apply_norm(safe_log(d["snap_dNdX"][grp]), norm_stats["dndx"])      # (n,3)
    t_delta = apply_norm(signed_log(d["delta"][idx]), norm_stats["delta"])      # (n,3,K)

    # P_filt normalization REDESIGN: split into the θ-blind BASELINE target
    # t_p_base (σ_marg-standardized cell-mean) and the σ_cosmo-whitened RESIDUAL
    # target t_p_resid (the cosmology signal). See fit_baseline_residual_norm.
    pf = norm_stats["P_filt"]
    logP = safe_log(d["P_filt"][idx])                                          # (n,4,K)
    cells = cell_id(d, idx)                                                    # (n,)
    m_cell = np.empty_like(logP)                                              # (n,4,K) cell-mean per row
    cell_mean = pf["cell_mean"]
    n_fallback = 0
    for r, cid in enumerate(cells):
        cm = cell_mean.get(int(cid))
        if cm is None:
            # Shouldn't happen under LOSO (it holds out SIMS, not cells); fall back
            # to mu_marg (a zero-cosmology-signal row) and warn.
            cm = pf["mu_marg"]
            n_fallback += 1
        m_cell[r] = cm
    if n_fallback:
        warnings.warn(
            f"make_batch: {n_fallback}/{len(idx)} rows in a cell absent from "
            f"train_cells; fell back to mu_marg (unexpected under LOSO)",
            RuntimeWarning)
    t_p_base = (m_cell - pf["mu_marg"]) / pf["sig_marg"]                        # (n,4,K)
    t_p_resid = (logP - m_cell) / pf["sig_cosmo"]                              # (n,4,K)

    # A4b: Head-A masks = finite AND (cache value > 0). The CDDF/dN/dX structural
    # zeros (e.g. f_nhi bin 0) are safe_log-floored to ~-69; mask them out of the
    # Head-A loss (joint_loss uses these instead of jnp.ones_like).
    t_f_nhi_mask = _valid_target_mask(d["snap_f_nhi"][grp])
    t_dndx_mask = _valid_target_mask(d["snap_dNdX"][grp])

    # inv_nalpha: 1 / (#rows sharing this row's snap-block) over the FULL cache,
    # so alpha-siblings (same snap, different alpha rescale) each get weight 1/n_a
    # and a snap-block is not over-counted relative to its number of alpha samples.
    block_counts = np.bincount(d["snap_group_idx"],
                               minlength=int(d["snap_group_idx"].max()) + 1)
    inv_nalpha = 1.0 / block_counts[grp].astype(np.float64)                     # (n,)

    # mean_F_clean: kept (unused by the loss) for batch-contract stability. The
    # A3 meanF loss term was removed (it had zero model gradient — see joint_loss);
    # mean-flux consistency is structural (tau0 = -ln(target_F) is an input). We
    # use exp(-tau0), the definitional mean flux tau0 encodes, until a dedicated
    # mean-F head is wired (deferred, out of scope).
    mean_F_clean = np.exp(-d["tau0"][idx]).astype(np.float64)                   # (n,)

    # global (z,τ₀)-cell id per row + the static segment count n_cells = n_z·n_alpha
    # (cell_id packs z_rank·n_alpha + alpha_idx, so the max id is n_z·n_alpha-1). The
    # coherent de-bias term segment-sums the residual error per cell over num_segments
    # = n_cells (a fold-level constant; traced once).
    n_z = int(np.unique(np.round(np.asarray(d["z_grid"]), 4)).shape[0])
    n_alpha = int(np.asarray(d["alpha_idx"]).max()) + 1
    n_cells = n_z * n_alpha

    batch = {
        "x": d["x"][idx].astype(np.float64),
        "tau0": d["tau0"][idx].astype(np.float64),
        "t_f_nhi": t_f_nhi.astype(np.float64),
        "t_dndx": t_dndx.astype(np.float64),
        "t_p_base": t_p_base.astype(np.float64),
        "t_p_resid": t_p_resid.astype(np.float64),
        "t_delta": t_delta.astype(np.float64),
        "t_f_nhi_mask": t_f_nhi_mask.astype(bool),
        "t_dndx_mask": t_dndx_mask.astype(bool),
        "mask": d["mask"][idx].astype(bool),
        "inv_nc": d["inv_nc"][idx].astype(np.float64),
        "inv_nalpha": inv_nalpha,
        "mean_F_clean": mean_F_clean,
        "cell": cells.astype(np.int32),
        # n_cells is the STATIC segment count for the coherent de-bias term — emit a
        # PLAIN PYTHON int (NOT np.int64). coherent_debias_term feeds it to
        # segment_sum as num_segments, which MUST be a static python int under
        # jax.jit; an np.int64 traces as an int64[] leaf and raises
        # ConcretizationTypeError when a raw make_batch dict is jitted (jax-traps #25).
        "n_cells": int(n_cells),
    }
    if k_weight is not None:
        # carried as a PER-ROW (n,K) tile so it pads/batches like every other array
        # (_pad_batch zero-pads it; padded rows are masked out anyway). The loss reads
        # row 0 (all rows identical) — see model._k_weight_from_batch.
        kw = np.asarray(k_weight, dtype=np.float64).ravel()
        batch["k_weight"] = np.broadcast_to(kw, (len(idx), kw.shape[0])).astype(np.float64)
    if datarange:
        # per-(row,k) SOFT down-weight; out-of-range bins get oor_weight (~0.2).
        # Per-ROW (depends on z), so it pads/batches like every other array.
        batch["datarange_weight"] = datarange_loss_weight(
            d, idx, z_lo=z_lo, z_hi=z_hi, k_min=k_min, oor_weight=oor_weight)
    return batch


def reconstruct_P_filt(P_filt_base, P_filt_resid, pf_stats):
    """Reconstruct LINEAR P_filt from the two-head outputs (normalization REDESIGN).

    logP̂ = (m̂·sig_marg + mu_marg) + sig_cosmo·r̂, where m̂ = baseline-head output
    (σ_marg-standardized cell-mean) and r̂ = residual-head output (σ_cosmo-whitened
    cosmology signal); then exp -> linear P_filt. ``pf_stats`` is the structured
    P_filt norm dict (mu_marg, sig_marg, sig_cosmo). Differentiable; the only θ
    dependence is through r̂ (the baseline is θ-blind), so
    ∂logP̂/∂θ = sig_cosmo·∂r̂/∂θ.
    """
    base = np.asarray(P_filt_base)
    resid = np.asarray(P_filt_resid)
    logP = (base * pf_stats["sig_marg"] + pf_stats["mu_marg"]) \
        + pf_stats["sig_cosmo"] * resid
    return np.exp(logP)


def untransform_prediction(pred_dict, norm_stats):
    """Standardized prediction dict -> physical-space dict.

    For f_nhi/dndx/delta: inverse of make_batch's transform (invert_norm then the
    channel's inverse exp/sinh). For P_filt (the normalization REDESIGN): supply
    BOTH ``P_filt_base`` and ``P_filt_resid`` and the structured norm_stats["P_filt"]
    dict; reconstruction is ``reconstruct_P_filt`` -> linear P_filt under key
    ``P_filt``. Channels absent from pred_dict are skipped.
    """
    out = {}
    if "P_filt_base" in pred_dict and "P_filt_resid" in pred_dict:
        out["P_filt"] = reconstruct_P_filt(
            pred_dict["P_filt_base"], pred_dict["P_filt_resid"], norm_stats["P_filt"])
    for ch, arr in pred_dict.items():
        if ch in ("P_filt_base", "P_filt_resid"):
            continue
        inv = TARGET_TRANSFORMS[ch][1]
        out[ch] = inv(invert_norm(np.asarray(arr), norm_stats[ch]))
    return out


# --- Cross-validation / extrapolation splits (spec sec.7) ---------------------

def kfold_loso(sim_name, n_folds=8):
    """Return [(train_row_idx, val_row_idx), ...]; sims partitioned into n_folds
    disjoint groups, each group held out as validation once."""
    sims = np.array(sorted(set(sim_name)))
    groups = np.array_split(sims, n_folds)
    all_idx = np.arange(len(sim_name))
    folds = []
    for held in groups:
        if len(held) == 0:
            continue
        is_val = np.isin(sim_name, held)
        folds.append((all_idx[~is_val], all_idx[is_val]))
    return folds

def tau0_edge_holdout(tau0, frac=0.15):
    """Hold out the low+high tau0 tails (the extrapolation probe in tau0-space)."""
    order = np.argsort(tau0)
    k = max(1, int(round(frac * len(tau0) / 2)))
    ho = np.concatenate([order[:k], order[-k:]])
    tr = np.setdiff1d(np.arange(len(tau0)), ho)
    return tr, ho


def make_splits(d, fold, n_folds=8, holdout_frac=0.15):
    """Compose the tau0-edge holdout x LOSO split for one fold (spec sec.7).

    Returns ``(train_idx, val_idx, holdout_idx)`` where:
      - ``holdout_idx`` is the tau0-edge extrapolation probe (``tau0_edge_holdout``),
        excluded from BOTH train and val so it stays entirely unseen;
      - the remaining pool is LOSO-partitioned (``kfold_loso``) into ``n_folds``,
        and ``train_idx``/``val_idx`` are this ``fold``'s train/val rows with the
        holdout removed from each.

    Mirrors the inline CLI logic in scripts/train_emulator.py so the composition
    is unit-testable (disjointness + LOSO no-straddle)."""
    _, holdout_idx = tau0_edge_holdout(d["tau0"], frac=holdout_frac)
    pool_mask = np.ones(len(d["tau0"]), bool)
    pool_mask[holdout_idx] = False

    folds = kfold_loso(d["sim_name"], n_folds=n_folds)
    if fold >= len(folds):
        raise IndexError(f"fold {fold} out of range (have {len(folds)} folds)")
    tr, va = folds[fold]
    train_idx = tr[pool_mask[tr]]   # drop holdout rows from train
    val_idx = va[pool_mask[va]]     # and from val
    return train_idx, val_idx, holdout_idx
