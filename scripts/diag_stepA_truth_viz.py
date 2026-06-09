"""STEP-A closure-mock RAW-DATA visualization (forward cache reads + plots; NO training/NUTS).

For each STEP-A fiducial TRUTH sim (a held-out PRIYA sim used as a closure mock), show where its
mock truth sits relative to the full 60-sim LF suite ensemble:

  1. stepA_truth_p1d.png        — mock TRUTH P1D vs k (z=2.4/3.0/3.8), one panel per fiducial,
                                  overlaid on the suite-mean P1D + 16-84% suite band.
  2. stepA_truth_dndx.png       — per-class dN/dX (LLS/subDLA/DLA) vs z + CDDF f_nhi(N_HI), for
                                  the fiducials vs the suite mean.
  3. stepA_truth_perclass_p1d.png — for ONE fiducial (M1): per-class decomposition P_clean +
                                  the HCD excess R_c = P_c - P_clean (LLS/subDLA/DLA) on the cache grid.
  4. stepA_hr_vs_lf_truth.png   — for M4: the HR-resolution truth P1D vs the LF truth at the SAME
                                  sim (the resolution gap the MF correction must bridge), per z.

The mock TRUTH is the cache's MEASURED contaminated P1D of the held-out sim (P_obs_true =
Sum_c coef_c * P_filt_c with the sim's own structural w_c, DLA core add-back on the DLA class) at
the becker13-anchored tau0 rung per z — EXACTLY closure_legb.make_truth_from_sim (LF, no MF).

"Sim-suite mean" = mean over ALL 60 LF sims at matched z, each built the SAME way (becker13 rung),
interpolated onto a common canonical LF k-grid before averaging.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_stepA_truth_viz.py
"""
from __future__ import annotations

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp

from hcd_analysis.emulator.data import load_cache, Z_LIMITS
from hcd_analysis.emulator.closure_legb import (
    CACHE_PATH, held_out_sims, make_truth_from_sim, build_legb_ctx,
)
from hcd_analysis.emulator.meanflux_prior import becker13_tau0
from hcd_analysis.emulator import multifidelity as MF

OUT = "/home/mfho/hcd_priya/figures/analysis/05_likelihood"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"

# Fiducial targets: (label, fold, target n_s). M4 is the HR truth (n_s ~= 0.979).
FIDUCIALS = [
    ("L1a-lo", 0, 0.81),
    ("L1a-mid", 4, 0.92),
    ("L1a-hi", 7, 1.00),
    ("M1", 7, 1.019),
    ("M2", 7, 1.040),
]
M4_NS = 0.979           # HR truth (matched in BOTH HR and LF caches by params)

# representative z for the P1D panels.
Z_REP = [2.4, 3.0, 3.8]

# common canonical LF k-grid for interp/averaging (log-spaced inside the LF band).
K_GRID = np.geomspace(5e-3, 0.069, 40)


# ============================================================================ #
#  Resolve the exact held-out sim closest to each target n_s.
# ============================================================================ #
def resolve_fiducials(d):
    names = np.asarray(d["sim_name"])
    ns = d["params"][:, 0]
    resolved = {}
    for label, fold, target in FIDUCIALS:
        sims, _va = held_out_sims(d, fold=fold)
        best, best_ns, best_dn = None, None, np.inf
        for s in sims:
            r = np.where(names == s)[0][0]
            dn = abs(ns[r] - target)
            if dn < best_dn:
                best, best_ns, best_dn = str(s), float(ns[r]), dn
        resolved[label] = dict(sim=best, ns=best_ns, fold=fold, target=target)
    return resolved


def resolve_m4(d):
    """The HR truth sim closest to n_s~=0.979, also present in the LF cache (for the LF
    comparison). Returns dict(sim, ns_hr, ns_lf, lf_fold)."""
    with h5py.File(HR_CACHE, "r") as f:
        hr_names = np.array([s.decode() if isinstance(s, bytes) else s
                             for s in f["sim_name"][:]])
        hr_ns = f["params"][:, 0]
    uniq = {}
    for s, n in zip(hr_names, hr_ns):
        uniq.setdefault(str(s), float(n))
    sim = min(uniq, key=lambda s: abs(uniq[s] - M4_NS))
    ns_hr = uniq[sim]
    lf_names = np.asarray(d["sim_name"])
    lf_rows = np.where(lf_names == sim)[0]
    ns_lf = float(d["params"][lf_rows[0], 0]) if lf_rows.size else None
    lf_fold = None
    for fold in range(8):
        sims, _ = held_out_sims(d, fold=fold)
        if sim in sims:
            lf_fold = fold
            break
    return dict(sim=sim, ns_hr=ns_hr, ns_lf=ns_lf, lf_fold=lf_fold)


# ============================================================================ #
#  Suite-mean (LF, all 60 sims) P1D, dN/dX, CDDF on matched z (+ becker13 rung).
# ============================================================================ #
def _sim_truth_rows(d, sim):
    """The becker13-anchored ladder row per in-range z for a sim (mirrors make_truth_from_sim)."""
    names = np.asarray(d["sim_name"])
    z_grid = d["z_grid"]
    P_filt = d["P_filt"]
    tau0 = d["tau0"]
    rows = np.where(names == sim)[0]
    cand = np.array([int(r) for r in rows
                     if (2.2 - 1e-6 <= z_grid[r] <= 4.6 + 1e-6) and np.isfinite(P_filt[r]).all()])
    if cand.size == 0:
        return np.array([], int)
    z_of = np.round(z_grid[cand], 4)
    keep = []
    for zz in np.unique(z_of):
        sub = cand[z_of == zz]
        target = float(becker13_tau0(jnp.asarray(float(zz))))
        keep.append(int(sub[int(np.argmin(np.abs(tau0[sub] - target)))]))
    return np.array(sorted(keep, key=lambda r: z_grid[r]))


def _Pobs_of_row(d, r):
    """The contaminated truth P1D of one cache row (LF, no MF) on that row's own k-grid."""
    P_filt = d["P_filt"]
    delta = d["delta"]
    w_c = d["w_c_cache"]
    a = w_c[r, 1:]
    coef = np.concatenate([[1.0 - a.sum()], a])
    core = delta[r, 2]
    P_cls = np.stack([P_filt[r, 0], P_filt[r, 1], P_filt[r, 2], P_filt[r, 3] + core])
    return np.einsum("c,ck->k", coef, P_cls)


def suite_stats(d):
    """Per-z suite statistics over all 60 LF sims at the becker13 rung, on K_GRID.

    Returns dict z -> dict(P_mean, P_lo, P_hi, n_sim) [P1D on K_GRID];
            and dN/dX + CDDF suite means keyed by z / by N_HI.
    """
    names = np.asarray(d["sim_name"])
    z_grid = d["z_grid"]
    cache_k = d["kfkms"]
    gidx = d["snap_group_idx"]
    snap_dNdX = d["snap_dNdX"]            # (1072,3)
    snap_f_nhi = d["snap_f_nhi"]          # (1072,30)
    all_sims = sorted(set(str(s) for s in names))

    from collections import defaultdict
    p1d_by_z = defaultdict(list)
    dndx_by_z = defaultdict(list)         # z -> list of (3,) dN/dX per sim
    cddf_by_z = defaultdict(list)         # z -> list of (30,) f_nhi per sim

    for s in all_sims:
        rows = _sim_truth_rows(d, s)
        for r in rows:
            zz = round(float(z_grid[r]), 2)
            P = _Pobs_of_row(d, r)
            k = cache_k[r]
            m = np.isfinite(k) & (k > 0) & np.isfinite(P) & (P > 0)
            if m.sum() < 5:
                continue
            P_on = np.exp(np.interp(np.log(K_GRID), np.log(k[m]), np.log(P[m]),
                                    left=np.nan, right=np.nan))
            p1d_by_z[zz].append(P_on)
            g = int(gidx[r])
            dndx_by_z[zz].append(snap_dNdX[g])
            cddf_by_z[zz].append(snap_f_nhi[g])

    p1d_stats = {}
    for zz, lst in p1d_by_z.items():
        arr = np.array(lst)
        p1d_stats[zz] = dict(
            P_mean=np.nanmean(arr, 0),
            P_lo=np.nanpercentile(arr, 16, axis=0),
            P_hi=np.nanpercentile(arr, 84, axis=0),
            n_sim=arr.shape[0])
    dndx_stats = {zz: np.nanmean(np.array(lst), 0) for zz, lst in dndx_by_z.items()}
    cddf_stats = {zz: np.nanmean(np.array(lst), 0) for zz, lst in cddf_by_z.items()}
    return dict(p1d=p1d_stats, dndx=dndx_stats, cddf=cddf_stats)


# ============================================================================ #
#  Per-fiducial truth packs (LF) + the M4 HR/LF truth.
# ============================================================================ #
def truth_pack(d, sim, fold):
    """make_truth_from_sim + per-z dN/dX + CDDF (snap-level) + interp P1D onto K_GRID."""
    t = make_truth_from_sim(d, sim, fold=fold)        # LF truth, no MF
    z = np.asarray(t["z"])
    P = np.asarray(t["P_obs_true"])                   # (nZ, K)
    cache_k = d["kfkms"]
    rows = np.asarray(t["rows"])
    gidx = d["snap_group_idx"]
    snap_dNdX = d["snap_dNdX"]
    snap_f_nhi = d["snap_f_nhi"]
    P_on = np.full((len(z), len(K_GRID)), np.nan)
    dndx = np.full((len(z), 3), np.nan)
    cddf = np.full((len(z), 30), np.nan)
    for i, r in enumerate(rows):
        k = cache_k[r]
        m = np.isfinite(k) & (k > 0) & np.isfinite(P[i]) & (P[i] > 0)
        if m.sum() >= 5:
            P_on[i] = np.exp(np.interp(np.log(K_GRID), np.log(k[m]), np.log(P[i][m]),
                                       left=np.nan, right=np.nan))
        g = int(gidx[r])
        dndx[i] = snap_dNdX[g]
        cddf[i] = snap_f_nhi[g]
    return dict(z=z, P_on=P_on, dndx=dndx, cddf=cddf, w_c=np.asarray(t["w_c"]),
                P_native=P, rows=rows, cache_k=cache_k)


def m4_hr_lf_truth(d, m4):
    """The M4 sim's LF truth (from the LF cache) and HR truth (from the HR cache), both built
    the SAME way: becker13 rung per z, contaminated P_obs = Sum coef*P_filt + DLA core. The HR
    cache is collapsed to the 4 coarse classes the SAME way load_cache does for LF, so the only
    difference is the simulation RESOLUTION (1536^3 vs 3072^3), not the construction."""
    sim = m4["sim"]
    # LF truth pack (sim is held-out in m4['lf_fold']).
    lf = truth_pack(d, sim, m4["lf_fold"])
    # HR truth: load HR cache, collapse to coarse classes, build the contaminated combination.
    hr = _hr_truth_from_cache(sim)
    return lf, hr


def _hr_truth_from_cache(sim):
    """Build the HR contaminated truth P1D per z for one sim from the HR cache (mirrors
    load_cache's coarse collapse + make_truth_from_sim's becker13 rung + contamination)."""
    from hcd_analysis.emulator.data import _collapse_counts, _collapse_p1d
    with h5py.File(HR_CACHE, "r") as f:
        names = np.array([s.decode() if isinstance(s, bytes) else s for s in f["sim_name"][:]])
        sel = np.where(names == sim)[0]
        counts15 = f["tier_c_counts"][sel]
        Pf15 = f["P_tier_c_filtered"][sel]
        Pu15 = f["P_tier_c"][sel]
        kfkms = f["kfkms"][sel]
        z = f["z_grid"][sel]
        tF = f["target_F"][sel]
    Pf4 = _collapse_p1d(Pf15, counts15)               # (n,4,K)
    Pu4 = _collapse_p1d(Pu15, counts15)
    cc = _collapse_counts(counts15)                   # (n,4)
    delta = (Pu4 - Pf4)[:, 1:, :]                     # (n,3,K) DLA core add-back source
    N = cc.sum(1, keepdims=True)
    w_c = np.where(N > 0, cc / np.maximum(N, 1), 0.0)  # (n,4)
    tau0 = -np.log(tF)
    # becker13 rung per z (HR cache also has multiple tau0 rungs per z).
    z_of = np.round(z, 4)
    keep = []
    for zz in np.unique(z_of):
        sub = np.where((z_of == zz) & (2.2 - 1e-6 <= z) & (z <= 4.6 + 1e-6)
                       & np.isfinite(Pf4[:, 0, :]).all(1))[0]
        if sub.size == 0:
            continue
        target = float(becker13_tau0(jnp.asarray(float(zz))))
        keep.append(int(sub[int(np.argmin(np.abs(tau0[sub] - target)))]))
    keep = np.array(sorted(keep, key=lambda r: z[r]))
    K = Pf4.shape[-1]
    P_on = np.full((len(keep), len(K_GRID)), np.nan)
    z_keep = z[keep]
    for i, r in enumerate(keep):
        a = w_c[r, 1:]
        coef = np.concatenate([[1.0 - a.sum()], a])
        core = delta[r, 2]
        P_cls = np.stack([Pf4[r, 0], Pf4[r, 1], Pf4[r, 2], Pf4[r, 3] + core])
        Pobs = np.einsum("c,ck->k", coef, P_cls)
        k = kfkms[r]
        m = np.isfinite(k) & (k > 0) & np.isfinite(Pobs) & (Pobs > 0)
        if m.sum() >= 5:
            P_on[i] = np.exp(np.interp(np.log(K_GRID), np.log(k[m]), np.log(Pobs[m]),
                                       left=np.nan, right=np.nan))
    return dict(z=z_keep, P_on=P_on)


# ============================================================================ #
#  Figures.
# ============================================================================ #
def _nearest_z(zarr, ztarget, tol=0.15):
    j = int(np.argmin(np.abs(np.asarray(zarr) - ztarget)))
    return j if abs(zarr[j] - ztarget) <= tol else None


def fig1_truth_p1d(resolved, packs, suite):
    """Per fiducial x z: a P1D panel (truth solid, suite mean dashed, 16-84% band) with a
    thin RATIO strip below (truth/suite-mean and the band-as-ratio) so the deviation of each
    mock's truth from the ensemble is legible (on log-log P1D it is otherwise invisible)."""
    labels = list(resolved.keys())
    nrow = len(labels)
    ncol = len(Z_REP)
    fig = plt.figure(figsize=(4.2 * ncol, 3.5 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.30)
    h0 = l0 = None
    for ri, label in enumerate(labels):
        pk = packs[label]
        info = resolved[label]
        for ci, zt in enumerate(Z_REP):
            inner = outer[ri, ci].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
            zz = round(min(suite["p1d"].keys(), key=lambda z: abs(z - zt)), 2)
            sst = suite["p1d"][zz]
            mean = sst["P_mean"]
            ax.fill_between(K_GRID, sst["P_lo"], sst["P_hi"], color="0.78", alpha=0.7,
                            label="suite 16-84%")
            ax.plot(K_GRID, mean, "--", color="0.35", lw=1.6, label="suite mean")
            j = _nearest_z(pk["z"], zt)
            if j is not None:
                ax.plot(K_GRID, pk["P_on"][j], "-", color="C3", lw=2.2, label="mock truth")
                axr.plot(K_GRID, pk["P_on"][j] / mean, "-", color="C3", lw=1.8)
            axr.fill_between(K_GRID, sst["P_lo"] / mean, sst["P_hi"] / mean,
                             color="0.78", alpha=0.7)
            axr.axhline(1.0, color="0.35", ls="--", lw=1.0)
            ax.set_xscale("log"); ax.set_yscale("log"); axr.set_xscale("log")
            ax.grid(alpha=0.3, which="both"); axr.grid(alpha=0.3, which="both")
            ax.tick_params(labelbottom=False)
            axr.set_ylim(0.9, 1.1)
            if h0 is None:
                h0, l0 = ax.get_legend_handles_labels()
            if ri == 0:
                ax.set_title(f"z = {zt:.1f}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{label} (n_s={info['ns']:.3f})\nP1D [km/s]", fontsize=8.5)
                axr.set_ylabel("/suite", fontsize=8)
            if ri == nrow - 1:
                axr.set_xlabel("k [s/km]")
    fig.legend(h0, l0, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("STEP-A mock TRUTH P1D vs the 60-sim LF suite (contaminated P_obs, becker13 rung)"
                 "  —  lower strip: truth / suite-mean",
                 fontsize=12, y=1.025)
    p = f"{OUT}/stepA_truth_p1d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2_dndx(resolved, packs, suite, d):
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 4, wspace=0.32)
    classes = ["LLS", "subDLA", "DLA"]
    colors = {lab: c for lab, c in zip(resolved, ["C0", "C1", "C2", "C3", "C4"])}
    # suite dN/dX vs z.
    z_suite = np.array(sorted(suite["dndx"].keys()))
    dndx_suite = np.array([suite["dndx"][z] for z in z_suite])   # (nz,3)
    for ic, cls in enumerate(classes):
        ax = fig.add_subplot(gs[0, ic])
        ax.plot(z_suite, dndx_suite[:, ic], "k--o", lw=1.8, ms=4, label="suite mean", zorder=5)
        for label, pk in packs.items():
            order = np.argsort(pk["z"])
            ax.plot(pk["z"][order], pk["dndx"][order, ic], "-", color=colors[label],
                    lw=1.5, alpha=0.9, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_ylabel(f"dN/dX  ({cls})")
        ax.set_title(f"dN/dX: {cls}")
        ax.grid(alpha=0.3)
        if ic == 0:
            ax.legend(fontsize=7, ncol=2)
    # CDDF f_nhi vs log N_HI for one representative sim (M1) vs suite mean, at z~3.0.
    ax = fig.add_subplot(gs[0, 3])
    with h5py.File(CACHE_PATH, "r") as f:
        lognhi = f["log_nhi_centres"][:]
    zrep = round(min(suite["cddf"].keys(), key=lambda z: abs(z - 3.0)), 2)
    cddf_suite = suite["cddf"][zrep]
    ax.plot(lognhi, cddf_suite, "k--o", lw=1.8, ms=3, label=f"suite mean (z~{zrep})", zorder=5)
    pk = packs["M1"]
    j = _nearest_z(pk["z"], 3.0)
    if j is not None:
        ax.plot(lognhi, pk["cddf"][j], "-", color="C3", lw=1.8, label=f"M1 (z={pk['z'][j]:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("log10 N_HI [cm^-2]")
    ax.set_ylabel("f(N_HI)  [CDDF]")
    ax.set_title("CDDF f(N_HI): M1 vs suite (z~3)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle("STEP-A fiducials: per-class dN/dX(z) + CDDF vs the 60-sim suite mean", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{OUT}/stepA_truth_dndx.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3_perclass(d, resolved):
    """For M1: per-class decomposition on the cache grid. P_clean and the HCD excess
    R_c = P_c - P_clean (LLS/subDLA/DLA), at the becker13 rung, per representative z."""
    sim = resolved["M1"]["sim"]
    fold = resolved["M1"]["fold"]
    rows = _sim_truth_rows(d, sim)
    z_grid = d["z_grid"]
    P_filt = d["P_filt"]                 # (R,4,K) [clean,LLS,sub,DLA]
    delta = d["delta"]                   # (R,3,K) [LLS,sub,DLA]
    cache_k = d["kfkms"]
    fig, axes = plt.subplots(1, len(Z_REP), figsize=(4.6 * len(Z_REP), 4.2), squeeze=False)
    cls_names = ["LLS", "subDLA", "DLA"]
    cls_colors = ["C0", "C1", "C2"]
    for ci, zt in enumerate(Z_REP):
        ax = axes[0][ci]
        # nearest becker13-rung row to zt.
        zr = z_grid[rows]
        ri = rows[int(np.argmin(np.abs(zr - zt)))]
        k = cache_k[ri]
        m = np.isfinite(k) & (k > 0)
        P_clean = P_filt[ri, 0]
        ax.plot(k[m], P_clean[m], "k-", lw=2.2, label="P_clean")
        # R_c = P_c - P_clean for c in LLS/sub/DLA. P_c here is the per-class P_filt (clean-baseline
        # contamination component); the DLA class also carries the additive core (delta[,2]).
        for jc, (cn, cc) in enumerate(zip(cls_names, cls_colors)):
            P_c = P_filt[ri, 1 + jc].copy()
            if cn == "DLA":
                P_c = P_c + delta[ri, 2]
            R_c = P_c - P_clean
            # plot |R_c| (signed shown via style: positive solid, negative dashed)
            pos = R_c > 0
            ax.plot(k[m & pos], np.abs(R_c)[m & pos], "-", color=cc, lw=1.6,
                    label=f"R_{cn} (+)")
            if (m & ~pos).any():
                ax.plot(k[m & ~pos], np.abs(R_c)[m & ~pos], ":", color=cc, lw=1.6,
                        label=f"R_{cn} (-)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k [s/km]")
        if ci == 0:
            ax.set_ylabel("P1D  /  |R_c = P_c - P_clean|  [km/s]")
        ax.set_title(f"z = {z_grid[ri]:.1f}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"STEP-A M1 per-class decomposition: clean baseline + HCD excess R_c "
                 f"(n_s={resolved['M1']['ns']:.3f}, becker13 rung)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{OUT}/stepA_truth_perclass_p1d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig4_hr_vs_lf(m4, lf, hr):
    fig, axes = plt.subplots(1, len(Z_REP), figsize=(4.6 * len(Z_REP), 4.4), squeeze=False)
    for ci, zt in enumerate(Z_REP):
        ax = axes[0][ci]
        axr = ax.twinx()
        jl = _nearest_z(lf["z"], zt)
        jh = _nearest_z(hr["z"], zt)
        if jl is not None:
            ax.plot(K_GRID, lf["P_on"][jl], "-", color="C0", lw=2.2, label="LF truth (1536^3)")
        if jh is not None:
            ax.plot(K_GRID, hr["P_on"][jh], "-", color="C3", lw=2.2, label="HR truth (3072^3)")
        if jl is not None and jh is not None:
            ratio = lf["P_on"][jl] / hr["P_on"][jh] - 1.0
            axr.plot(K_GRID, 100 * ratio, "--", color="0.4", lw=1.4, label="(LF/HR - 1)")
            axr.axhline(0, color="0.6", lw=0.7)
            axr.set_ylabel("LF/HR - 1 [%]", color="0.3")
            axr.set_ylim(-25, 25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k [s/km]")
        if ci == 0:
            ax.set_ylabel("P_obs truth [km/s]")
        ax.set_title(f"z = {zt:.1f}")
        ax.grid(alpha=0.3, which="both")
        if ci == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axr.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    fig.suptitle(f"STEP-A M4 HR-vs-LF truth P1D (the resolution gap the MF correction bridges)\n"
                 f"sim n_s_HR={m4['ns_hr']:.3f} / n_s_LF={m4['ns_lf']:.3f}", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.84, bottom=0.13, wspace=0.45)
    p = f"{OUT}/stepA_hr_vs_lf_truth.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    print("loading LF cache ...")
    d = load_cache(CACHE_PATH)

    resolved = resolve_fiducials(d)
    m4 = resolve_m4(d)
    print("\n=== resolved fiducials (exact held-out sims closest to target n_s) ===")
    for label, info in resolved.items():
        print(f"  {label:8s}  fold={info['fold']}  target n_s={info['target']:.3f}  "
              f"-> n_s={info['ns']:.4f}  {info['sim']}")
    print(f"  {'M4(HR)':8s}  HR-truth   target n_s={M4_NS:.3f}  "
          f"-> n_s_HR={m4['ns_hr']:.4f} / n_s_LF={m4['ns_lf']:.4f} (LF fold {m4['lf_fold']})")
    print(f"           {m4['sim']}")

    print("\nbuilding per-fiducial truth packs ...")
    packs = {label: truth_pack(d, info["sim"], info["fold"])
             for label, info in resolved.items()}

    print("computing 60-sim suite stats (P1D / dN/dX / CDDF) ...")
    suite = suite_stats(d)
    # sanity-check the suite mean.
    nbad = 0
    for zz, st in suite["p1d"].items():
        if not np.isfinite(st["P_mean"]).any() or (st["P_mean"][np.isfinite(st["P_mean"])] <= 0).any():
            nbad += 1
    print(f"  suite P1D: {len(suite['p1d'])} z-bins, n_sim~{suite['p1d'][2.4]['n_sim']}, "
          f"{nbad} bad-mean z-bins")
    print(f"  suite dN/dX z-bins: {len(suite['dndx'])}  CDDF z-bins: {len(suite['cddf'])}")

    print("\nbuilding M4 HR/LF truth ...")
    lf_m4, hr_m4 = m4_hr_lf_truth(d, m4)

    print("\nrendering figures ...")
    p1 = fig1_truth_p1d(resolved, packs, suite)
    p2 = fig2_dndx(resolved, packs, suite, d)
    p3 = fig3_perclass(d, resolved)
    p4 = fig4_hr_vs_lf(m4, lf_m4, hr_m4)
    print(f"  {p1}\n  {p2}\n  {p3}\n  {p4}")

    # M4 HR-vs-LF resolution gap summary (for the txt + return).
    gap_lines = []
    for zt in Z_REP:
        jl = _nearest_z(lf_m4["z"], zt)
        jh = _nearest_z(hr_m4["z"], zt)
        if jl is None or jh is None:
            continue
        ratio = lf_m4["P_on"][jl] / hr_m4["P_on"][jh] - 1.0
        klo = K_GRID < 0.02
        khi = K_GRID > 0.04
        gap_lines.append(
            f"  z={zt:.1f}: LF/HR-1 low-k(k<0.02) mean={np.nanmean(ratio[klo])*100:+.2f}%  "
            f"high-k(k>0.04) mean={np.nanmean(ratio[khi])*100:+.2f}%")

    # write the txt sidecar.
    txt = f"{OUT}/stepA_truth_viz.txt"
    with open(txt, "w") as f:
        f.write("STEP-A closure-mock truth visualization — exact sims used\n")
        f.write("=" * 70 + "\n")
        f.write("Mock TRUTH = cache MEASURED contaminated P_obs of the held-out sim\n")
        f.write("  (P_obs = Sum_c coef_c * P_filt_c, coef=[1-Sum w, w_LLS, w_sub, w_DLA],\n")
        f.write("   + DLA core delta[,2] on the DLA class), becker13-anchored tau0 rung per z.\n")
        f.write("Suite mean = mean over all 60 LF sims, built the SAME way, at matched z.\n\n")
        for label, info in resolved.items():
            f.write(f"{label:8s}  fold={info['fold']}  target n_s={info['target']:.3f}  "
                    f"-> n_s={info['ns']:.4f}\n           {info['sim']}\n")
        f.write(f"{'M4':8s}  HR truth  target n_s={M4_NS:.3f}  "
                f"-> n_s_HR={m4['ns_hr']:.4f} / n_s_LF={m4['ns_lf']:.4f} (LF held-out fold {m4['lf_fold']})\n")
        f.write(f"           {m4['sim']}\n\n")
        f.write("M4 HR-vs-LF resolution gap (LF/HR - 1, on the shared LF band):\n")
        f.write("\n".join(gap_lines) + "\n")
    print(f"  {txt}")
    print("\n".join(gap_lines))
    return resolved, m4, gap_lines


if __name__ == "__main__":
    main()
