# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# Step-A target visualization diagnostic.
"""STEP-A closure TARGET-MOCK vs TRUTH visualization on the CORRECTED §0c DLA construction.

Forward cache reads + Leg-B mock build + plots ONLY. NO training, NO NUTS, NO checkpoint writes.

The closure TARGET MOCK is built by ``closure_legb.make_legb_mock`` (which calls
``make_truth_from_sim``), now §0c (PI-confirmed final intent 2026-06-09, commit 1e1b278):

  - DESI target = DLA-MASKED filtered Tier-P baseline  +  0.10·(full DLA excess)
                  (the 10% unmasked-DLA residual; TRUTH_DLA_FRAC["DESI"]=0.10).
  - KS   target = DLA-MASKED filtered Tier-P baseline,  0% DLA (TRUTH_DLA_FRAC["KS"]=0.0).
  - LLS/sub-DLA at the sim's structural w_c.

``make_truth_from_sim`` returns the masked baseline ``P_obs_true`` and the FULL DLA excess
``dla_excess_true`` (= w_DLA·(P_DLA^unf − P_clean)) on the cache grid; ``make_legb_mock`` adds
TRUTH_DLA_FRAC[leg]·excess per leg and exposes the noiseless per-leg target via
``info["truth_on_leg"]`` (the actual target the closure now fits, BEFORE cosmic noise).

Figures (figures/analysis/05_likelihood/, dpi=150) — REPLACE the stale ones:
  1. stepA_target_vs_truth.png — per fiducial, per leg (DESI vs KS): the noiseless TARGET P1D
     overlaid on the filtered-Tier-P baseline (no DLA residual), a few z. A ratio strip
     (target/baseline − 1) makes the 10% DLA residual visible as a small bump on DESI and FLAT
     (zero) on KS — the key contrast.
  2. stepA_dla_residual.png — the DLA-residual component itself, 0.10·(full DLA excess) on the
     cache grid vs k per z (DESI); KS = 0 shown as a flat line — exactly what the HCD α_DLA
     nuisance must marginalize.
  3. stepA_truth_p1d.png — REGENERATED on the corrected construction: the mock TARGET P1D vs the
     60-sim LF suite (target now = masked baseline + 0.10·DLA DESI residual / 0% KS).

Sidecar stepA_target_viz.txt: per fiducial, the exact sim + n_s, and the size of the DESI 10%
DLA residual (% of P1D at k~0.02, z~2.4) vs KS=0.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_stepA_target_viz.py
"""
from __future__ import annotations

# import the package FIRST (sets jax config) before any jax import.
import hcd_analysis.emulator  # noqa: F401

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import Z_LIMITS  # noqa: F401
from hcd_analysis.emulator.closure_legb import (
    CACHE_PATH, build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock,
    make_splits, TRUTH_DLA_FRAC,
)

OUT = "/home/mfho/hcd_priya/figures/analysis/05_likelihood"

# Fiducial targets: (label, fold, target n_s) — the L1a-lo/-mid/-hi + M1 the prompt names.
FIDUCIALS = [
    ("L1a-lo", 0, 0.81),
    ("L1a-mid", 4, 0.92),
    ("L1a-hi", 7, 1.00),
    ("M1", 7, 1.019),
]

# representative z for the P1D panels.
Z_REP = [2.4, 3.0, 3.8]

# common canonical LF k-grid for the suite interp/averaging (fig-3, suite mean).
K_GRID = np.geomspace(5e-3, 0.069, 40)

MOCK_SEED = 0  # the cosmic-noise draw seed (only used to instantiate the mock; we plot the
               # NOISELESS truth_on_leg target, so the seed does not affect the curves shown).


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


# ============================================================================ #
#  Per-fiducial: build the truth + the per-leg noiseless TARGET (truth_on_leg).
# ============================================================================ #
def build_target_pack(ctx, d, sim, fold):
    """make_truth_from_sim → make_legb_mock; return everything needed to draw the
    target-vs-baseline contrast on each leg (the NOISELESS target, the filtered baseline,
    the per-leg DLA-residual component) PLUS the cache-grid pieces for the suite figure."""
    t = make_truth_from_sim(d, sim, fold=fold)               # §0c masked baseline + dla_excess
    _ml, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(MOCK_SEED))
    cache_k = np.asarray(ctx.cache_k)
    z_sim = np.asarray(t["z"])
    P_base_cache = np.asarray(t["P_obs_true"])               # (nZs, K) masked baseline (no DLA resid)
    dla_excess_cache = np.asarray(t["dla_excess_true"])      # (nZs, K) FULL DLA excess

    legs = {}
    for leg in ctx.legs:
        truth_frac = float(TRUTH_DLA_FRAC.get(leg.name, 0.0))
        k_flat = np.asarray(leg.k)
        z_idx = np.asarray(leg.z_idx)
        target_flat = np.asarray(info["truth_on_leg"][leg.name])  # noiseless target (with resid)
        per_z = {}
        for iz in range(leg.n_z):
            zz = float(leg.z[iz])
            rows = np.where(z_idx == iz)[0]
            if rows.size == 0:
                continue
            j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) > 0.15:    # dropped z (no sim z) → skip
                continue
            ks_ = k_flat[rows]
            order = np.argsort(ks_)
            ks_ = ks_[order]
            rows = rows[order]
            base = np.interp(ks_, cache_k, P_base_cache[j])           # filtered baseline (no resid)
            resid = np.interp(ks_, cache_k, truth_frac * dla_excess_cache[j])  # the DLA residual
            target = target_flat[rows]                                 # = base + resid (noiseless)
            per_z[zz] = dict(k=ks_, target=target, base=base, resid=resid,
                             ratio=target / base - 1.0)
        legs[leg.name] = dict(truth_frac=truth_frac, per_z=per_z)

    # cache-grid pieces (for the DLA-residual fig + the suite-mean fig).
    return dict(t=t, tp=tp, z_sim=z_sim, cache_k=cache_k,
                P_base_cache=P_base_cache, dla_excess_cache=dla_excess_cache, legs=legs)


# ============================================================================ #
#  Suite mean (LF, all sims) of the §0c DESI TARGET on K_GRID (for fig-3).
#  Per sim, per z: masked baseline + 0.10·(full DLA excess), interp onto K_GRID.
# ============================================================================ #
def suite_target_stats(d, dla_frac):
    """Per-z suite mean/16-84% of the §0c TARGET (masked baseline + dla_frac·DLA excess) over
    ALL sims, on K_GRID. ``dla_frac`` selects which leg's target convention to mean (DESI 0.10
    for the DESI-target suite band; the suite is a context, the per-fiducial target is the focus)."""
    from collections import defaultdict
    names_arr = np.asarray(d["sim_name"])
    names = sorted(set(str(s) for s in names_arr))
    cache_k = np.asarray(d["kfkms"])
    # make_truth_from_sim selects rows from fold's VALIDATION split, so a sim only yields rows in
    # the fold whose val split contains it. Map each sim → a fold that holds it out (full 60-sim
    # coverage). The truth construction within that sim's rows is otherwise fold-independent.
    fold_of_sim = {}
    n_folds = 8
    for fold in range(n_folds):
        for s in held_out_sims(d, fold=fold)[0]:
            fold_of_sim.setdefault(str(s), fold)
    p1d_by_z = defaultdict(list)
    for s in names:
        fold = fold_of_sim.get(s)
        if fold is None:
            continue
        try:
            t = make_truth_from_sim(d, s, fold=fold)
        except Exception:
            continue
        z_sim = np.asarray(t["z"])
        Pb = np.asarray(t["P_obs_true"])
        Ex = np.asarray(t["dla_excess_true"])
        rows = np.asarray(t["rows"])
        for i, r in enumerate(rows):
            zz = round(float(z_sim[i]), 2)
            k = cache_k[r]
            P = Pb[i] + dla_frac * Ex[i]
            m = np.isfinite(k) & (k > 0) & np.isfinite(P) & (P > 0)
            if m.sum() < 5:
                continue
            P_on = np.exp(np.interp(np.log(K_GRID), np.log(k[m]), np.log(P[m]),
                                    left=np.nan, right=np.nan))
            p1d_by_z[zz].append(P_on)
    stats = {}
    for zz, lst in p1d_by_z.items():
        arr = np.array(lst)
        stats[zz] = dict(P_mean=np.nanmean(arr, 0),
                         P_lo=np.nanpercentile(arr, 16, axis=0),
                         P_hi=np.nanpercentile(arr, 84, axis=0),
                         n_sim=arr.shape[0])
    return stats


# ============================================================================ #
#  Fig 1 — target vs filtered baseline, DESI (left col) vs KS (right col), ratio strip.
# ============================================================================ #
def fig1_target_vs_truth(resolved, packs):
    labels = list(resolved.keys())
    leg_names = ["DESI", "KS"]
    ncol = len(leg_names) * len(Z_REP)        # DESI(z1,z2,z3)  KS(z1,z2,z3)
    nrow = len(labels)
    fig = plt.figure(figsize=(3.4 * ncol, 3.2 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.46, wspace=0.34)
    h0 = l0 = None
    for ri, label in enumerate(labels):
        info = resolved[label]
        pk = packs[label]
        for lj, lname in enumerate(leg_names):
            legpz = pk["legs"][lname]["per_z"]
            tf = pk["legs"][lname]["truth_frac"]
            for zi, zt in enumerate(Z_REP):
                ci = lj * len(Z_REP) + zi
                inner = outer[ri, ci].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.06)
                ax = fig.add_subplot(inner[0])
                axr = fig.add_subplot(inner[1], sharex=ax)
                # nearest available leg z to zt.
                if legpz:
                    zz = min(legpz.keys(), key=lambda z: abs(z - zt))
                    if abs(zz - zt) <= 0.15:
                        rec = legpz[zz]
                        ax.plot(rec["k"], rec["base"], "--", color="0.4", lw=1.7,
                                label="filtered Tier-P baseline (no DLA)")
                        ax.plot(rec["k"], rec["target"], "-", color="C3", lw=2.0,
                                label="target mock (noiseless)")
                        axr.plot(rec["k"], 100 * rec["ratio"], "-", color="C3", lw=1.8)
                axr.axhline(0.0, color="0.5", ls="--", lw=1.0)
                ax.set_xscale("log"); ax.set_yscale("log"); axr.set_xscale("log")
                ax.grid(alpha=0.3, which="both"); axr.grid(alpha=0.3, which="both")
                ax.tick_params(labelbottom=False)
                axr.set_ylim(-0.5, 0.5)
                if h0 is None:
                    h0, l0 = ax.get_legend_handles_labels()
                if ri == 0:
                    ax.set_title(f"{lname}  z={zt:.1f}\n(DLA frac={tf:.2f})", fontsize=9.5)
                if ci == 0:
                    ax.set_ylabel(f"{label}\n(n_s={info['ns']:.3f})\nP1D [km/s]", fontsize=8.5)
                if zi == 0:
                    axr.set_ylabel("targ/base\n-1 [%]", fontsize=7.5)
                if ri == nrow - 1:
                    axr.set_xlabel("k [s/km]", fontsize=8)
    fig.legend(h0, l0, loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.006))
    fig.suptitle("STEP-A TARGET MOCK vs filtered Tier-P baseline (corrected §0c): "
                 "DESI = baseline + 0.10·DLA excess (bump in ratio strip), KS = baseline + 0 (flat)",
                 fontsize=12.5, y=1.028)
    p = f"{OUT}/stepA_target_vs_truth.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================ #
#  Fig 2 — the DLA-residual component itself (0.10·full DLA excess) on DESI; KS = 0.
# ============================================================================ #
def fig2_dla_residual(resolved, packs):
    labels = list(resolved.keys())
    ncol = len(Z_REP)
    nrow = len(labels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ri, label in enumerate(labels):
        pk = packs[label]
        z_sim = pk["z_sim"]
        cache_k = pk["cache_k"]
        base = pk["P_base_cache"]
        excess = pk["dla_excess_cache"]
        fdesi = float(TRUTH_DLA_FRAC["DESI"])
        for ci, zt in enumerate(Z_REP):
            ax = axes[ri][ci]
            j = int(np.argmin(np.abs(z_sim - zt)))
            k = cache_k
            m = np.isfinite(k) & (k > 0)
            resid_desi = fdesi * excess[j]                   # 0.10·(full DLA excess)
            P = base[j]
            # plot the residual as % of P1D (the physically interpretable axis for the nuisance).
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_desi = 100.0 * resid_desi / P
            # signed: positive solid, negative dotted.
            pos = resid_desi >= 0
            ax.plot(k[m & pos], np.abs(pct_desi)[m & pos], "-", color="C3", lw=2.0,
                    label="DESI: 0.10·DLA excess (+)")
            if (m & ~pos).any():
                ax.plot(k[m & ~pos], np.abs(pct_desi)[m & ~pos], ":", color="C3", lw=2.0,
                        label="DESI: 0.10·DLA excess (-)")
            # KS = 0 reference line (drawn just above the axis floor so it reads as the zero level).
            ax.plot([k[m].min(), k[m].max()], [3e-4, 3e-4], "-", color="C0", lw=2.2,
                    label="KS: 0% (no DLA residual)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylim(2e-4, 2.0)
            ax.grid(alpha=0.3, which="both")
            if ri == 0:
                ax.set_title(f"z = {zt:.1f}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{label} (n_s={resolved[label]['ns']:.3f})\n"
                              f"|DLA residual| / P1D [%]", fontsize=8.5)
            if ri == nrow - 1:
                ax.set_xlabel("k [s/km]")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("STEP-A DLA-residual component the HCD α_DLA nuisance marginalizes "
                 "(corrected §0c): DESI = 0.10·w_DLA·(P_DLA^unf − P_clean), KS = 0",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{OUT}/stepA_dla_residual.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================ #
#  Fig 3 — regenerate stepA_truth_p1d.png on the corrected §0c construction.
#  The mock TARGET (DESI convention: masked baseline + 0.10·DLA excess) vs the suite mean.
# ============================================================================ #
def _nearest_z(zarr, ztarget, tol=0.15):
    j = int(np.argmin(np.abs(np.asarray(zarr) - ztarget)))
    return j if abs(zarr[j] - ztarget) <= tol else None


def fig3_truth_p1d(resolved, packs, suite):
    labels = list(resolved.keys())
    nrow = len(labels)
    ncol = len(Z_REP)
    fig = plt.figure(figsize=(4.2 * ncol, 3.5 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.30)
    h0 = l0 = None
    for ri, label in enumerate(labels):
        pk = packs[label]
        info = resolved[label]
        # the per-fiducial DESI-convention target on K_GRID (masked baseline + 0.10·excess).
        z_sim = pk["z_sim"]
        cache_k = pk["cache_k"]
        fdesi = float(TRUTH_DLA_FRAC["DESI"])
        tgt_on = {}
        for j, zz in enumerate(z_sim):
            P = pk["P_base_cache"][j] + fdesi * pk["dla_excess_cache"][j]
            m = np.isfinite(cache_k) & (cache_k > 0) & np.isfinite(P) & (P > 0)
            if m.sum() >= 5:
                tgt_on[round(float(zz), 2)] = np.exp(
                    np.interp(np.log(K_GRID), np.log(cache_k[m]), np.log(P[m]),
                              left=np.nan, right=np.nan))
        for ci, zt in enumerate(Z_REP):
            inner = outer[ri, ci].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
            zz = round(min(suite.keys(), key=lambda z: abs(z - zt)), 2)
            sst = suite[zz]
            mean = sst["P_mean"]
            ax.fill_between(K_GRID, sst["P_lo"], sst["P_hi"], color="0.78", alpha=0.7,
                            label="suite 16-84%")
            ax.plot(K_GRID, mean, "--", color="0.35", lw=1.6, label="suite mean (§0c target)")
            zkey = min(tgt_on.keys(), key=lambda z: abs(z - zt)) if tgt_on else None
            if zkey is not None and abs(zkey - zt) <= 0.15:
                ax.plot(K_GRID, tgt_on[zkey], "-", color="C3", lw=2.2, label="mock target")
                axr.plot(K_GRID, tgt_on[zkey] / mean, "-", color="C3", lw=1.8)
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
    fig.suptitle("STEP-A mock TARGET P1D vs the LF suite (corrected §0c: DESI-convention "
                 "target = masked baseline + 0.10·DLA excess)  —  lower strip: target / suite-mean",
                 fontsize=12, y=1.025)
    p = f"{OUT}/stepA_truth_p1d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================ #
def _desi_resid_readout(pk, zt=2.4, kt=0.02):
    """The DESI 10% DLA residual as % of P1D at (k~kt, z~zt) on the DESI leg, + the max over k."""
    legpz = pk["legs"]["DESI"]["per_z"]
    if not legpz:
        return None
    zz = min(legpz.keys(), key=lambda z: abs(z - zt))
    rec = legpz[zz]
    pct_at = float(np.interp(kt, rec["k"], 100 * rec["ratio"]))
    imax = int(np.argmax(np.abs(rec["ratio"])))
    return dict(z=zz, k=kt, pct_at_kt=pct_at,
                pct_max=float(100 * rec["ratio"][imax]), k_at_max=float(rec["k"][imax]))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("building Leg-B context (DESI + KS) ...")
    ctx, d = build_legb_ctx()
    for leg in ctx.legs:
        print(f"  leg {leg.name}: n_z={leg.n_z} N={leg.k.shape[0]} "
              f"k=[{float(leg.k.min()):.4g},{float(leg.k.max()):.4g}] "
              f"dla_forward_frac={getattr(leg, 'dla_forward_frac', 1.0)} "
              f"TRUTH_DLA_FRAC={TRUTH_DLA_FRAC.get(leg.name, 0.0)}")

    resolved = resolve_fiducials(d)
    print("\n=== resolved fiducials (exact held-out sims closest to target n_s) ===")
    for label, info in resolved.items():
        print(f"  {label:8s}  fold={info['fold']}  target n_s={info['target']:.3f}  "
              f"-> n_s={info['ns']:.4f}  {info['sim']}")

    print("\nbuilding per-fiducial target packs (make_truth_from_sim + make_legb_mock) ...")
    packs = {label: build_target_pack(ctx, d, info["sim"], info["fold"])
             for label, info in resolved.items()}

    print("computing suite mean of the §0c DESI-convention target (all sims) ...")
    suite = suite_target_stats(d, dla_frac=float(TRUTH_DLA_FRAC["DESI"]))
    print(f"  suite z-bins: {len(suite)}  n_sim(z~2.4)~{suite.get(2.4, {}).get('n_sim', '?')}")

    print("\nrendering figures ...")
    p1 = fig1_target_vs_truth(resolved, packs)
    p2 = fig2_dla_residual(resolved, packs)
    p3 = fig3_truth_p1d(resolved, packs, suite)
    print(f"  {p1}\n  {p2}\n  {p3}")

    # per-fiducial readout of the DESI 10% DLA residual.
    readouts = {label: _desi_resid_readout(pk) for label, pk in packs.items()}

    txt = f"{OUT}/stepA_target_viz.txt"
    with open(txt, "w") as f:
        f.write("STEP-A closure TARGET-MOCK vs TRUTH visualization (corrected §0c DLA, commit 1e1b278)\n")
        f.write("=" * 78 + "\n")
        f.write("TARGET MOCK (make_legb_mock + make_truth_from_sim, info['truth_on_leg']):\n")
        f.write("  DESI = DLA-MASKED filtered Tier-P baseline + 0.10·(full DLA excess)\n")
        f.write("         [TRUTH_DLA_FRAC['DESI']=0.10; the 10% unmasked-DLA residual]\n")
        f.write("  KS   = DLA-MASKED filtered Tier-P baseline + 0.0  [TRUTH_DLA_FRAC['KS']=0.0]\n")
        f.write("  full DLA excess = w_DLA·(P_DLA^unf − P_clean), P_DLA^unf = P_filt[DLA] + DLA core.\n")
        f.write("  LLS/subDLA at the sim's structural w_c. The forward marginalizes α_DLA over the\n")
        f.write("  DESI residual (dla_forward_frac: DESI 1.0 / KS 0.0).\n\n")
        for label, info in resolved.items():
            r = readouts[label]
            f.write(f"{label:8s}  fold={info['fold']}  n_s={info['ns']:.4f}  {info['sim']}\n")
            if r is not None:
                f.write(f"           DESI 10% DLA residual (target/baseline − 1): "
                        f"{r['pct_at_kt']:+.3f}% at k~{r['k']:.3g}, z~{r['z']:.2f}; "
                        f"max |{r['pct_max']:+.3f}%| at k~{r['k_at_max']:.4g}\n")
            f.write("           KS DLA residual: 0.000% (target ≡ filtered baseline, all k/z)\n")
        f.write("\nAll three figures regenerated on the corrected §0c construction "
                "(DESI carries 10%, KS 0).\n")
    print(f"  {txt}")

    print("\n=== DESI 10% DLA-residual readout (target/baseline − 1) ===")
    for label in resolved:
        r = readouts[label]
        if r is not None:
            print(f"  {label:8s}: {r['pct_at_kt']:+.3f}% at k~0.02 z~{r['z']:.2f}; "
                  f"peak {r['pct_max']:+.3f}% at k~{r['k_at_max']:.4g} | KS=0")
    return resolved, packs, readouts


if __name__ == "__main__":
    main()
