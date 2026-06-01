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

def fit_norm(x, train_idx):
    xt = x[train_idx]
    mean = np.nanmean(xt, axis=0)
    std = np.nanstd(xt, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}

def apply_norm(x, stats):
    return (x - stats["mean"]) / stats["std"]

def invert_norm(z, stats):
    return z * stats["std"] + stats["mean"]


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
