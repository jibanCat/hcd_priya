"""DECOMPOSE the deployed pred/true residual + MULTI-FOLD coherent-tilt audit.

Answers the PI's three concerns about the deployed LF emulator residual
   δ(sim, z, τ₀, k) = logP̂ − logP_true   over held-out (LOSO val) rows,
where logP̂ = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂  (data.reconstruct_P_filt):
  m̂ = θ-BLIND baseline head (z,τ₀) — pre-fit + FROZEN;
  r̂ = θ-dependent residual head (the tuned cosmology head, commit 73009c5/d0a008d).

CONCERN 1 (coherent k-tilt): clean/subDLA FAIL at low-k (k<0.01), DLA at high-k
  (k>0.03). We attribute each failing band to a component (a)-(f) below.
CONCERN 2 ("τ₀ & z or just cosmology?"): the deployed |P̂/P−1| / fig07 / the
  coherent-vs-CV plot are stated precisely (they marginalise OVER τ₀ and z).
CONCERN 3 (fold-0 only): we train/load all 8 LOSO folds and report the per-fold
  coherent tilt + A_p Fisher-bias.

DECOMPOSITION of δ (exact algebra, all in logP units unless whitened):
  δ = (m̂ − m_cell)              ... (a) BASELINE / (z,τ₀) mis-fit   [θ-independent]
    + σ_cosmo·(r̂ − t_resid)     ... (b) RESIDUAL-HEAD cosmology error within-cell
  where t_resid = (logP_true − m_cell)/σ_cosmo, m_cell = TRAIN cell-mean of logP.
  (c) ⟨r̂⟩_θ  : the residual head's θ-MEAN per (z,τ₀)-cell (should be ~0; the
      un-enforced-zero-mean concern — a coherent systematic NOT from cosmology).
  (d) τ₀-dependence of ⟨δ⟩ at fixed (z,k) : an uncaptured mean-flux systematic.
  (e) z-dependence  of ⟨δ⟩ at fixed (cls,k).
  (f) CV floor per (z,k) from diag_lfhf_tilt_and_cv.json (irreducible).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_residual_decomposition.py
        [--reuse] [--folds N] [--nbasis-test]   (writes figures + JSON to 04_emulator)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (
    load_cache, make_splits, make_batch, cell_id, safe_log, reconstruct_P_filt,
)
from hcd_analysis.emulator import train as T

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
CV_JSON = f"{OUT}/diag_lfhf_tilt_and_cv.json"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")

# DESI-DR1-like diagonal per-mode covariance (same assumption as the gate script).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
KPIVOT = 0.009                  # Lyα amplitude pivot Δ²_* (A_p maps onto low-k)

# The PI's named coherent-failure bands.
LOWK = (0.0, 0.01)              # clean & subDLA FAIL here  (k < 0.01)
HIGHK = (0.03, 0.07)           # DLA FAILS here             (k > 0.03)
MIDK = (0.01, 0.03)

# DEPLOYED production recipe (matches scripts/diag_ap_fisher_bias + walkthrough).
DEPLOYED = dict(n_basis=24, epochs=180, lr=1e-3, batch=512, patience=25,
                seed=0, early_stop_metric="auto", weight_decay=1e-4)


# --------------------------------------------------------------------------- #
# Train / load one fold at the deployed config.
# --------------------------------------------------------------------------- #
def train_or_load_fold(d, fold, n_k, *, n_basis=24, reuse=True, tag="decomp"):
    ckpt = f"/home/mfho/hcd_priya/checkpoints/{tag}_nb{n_basis}_fold{fold}"
    tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
    if reuse and Path(ckpt + ".eqx").exists():
        model, meta, norm = T.load_checkpoint(ckpt)
        return model, norm, (tr, va, ho)
    model, norm, hist = T.train_fold(
        d, tr, va, n_basis=n_basis, lr=DEPLOYED["lr"], epochs=DEPLOYED["epochs"],
        batch_size=DEPLOYED["batch"], seed=DEPLOYED["seed"],
        key=jax.random.PRNGKey(DEPLOYED["seed"]), patience=DEPLOYED["patience"],
        n_k=n_k, early_stop_metric=DEPLOYED["early_stop_metric"],
        weight_decay=DEPLOYED["weight_decay"])
    T.save_checkpoint(ckpt, model, {"in_dim": 10, "n_k": n_k, "n_basis": n_basis},
                      norm, seed=DEPLOYED["seed"], kfkms=d["kfkms"], cache_path=CACHE)
    print(f"  [train] fold {fold} nb{n_basis}: {len(hist['train_loss'])} epochs -> {ckpt}")
    return model, norm, (tr, va, ho)


# --------------------------------------------------------------------------- #
# Core: predict the deployed pieces on a set of rows.
# --------------------------------------------------------------------------- #
def deployed_pieces(model, d, idx, norm):
    """Return the per-row arrays needed for the full decomposition.

    logP̂, logP_true, m̂ (deployed baseline, logP units), r̂ (whitened residual
    head), m_cell (TRAIN cell-mean of logP), t_resid (training residual target),
    plus σ_cosmo (4,K), the (z,τ₀)-cell id, z_grid, tau0, kf — all over rows ``idx``.
    """
    pf = norm["P_filt"]
    sig_marg, mu_marg, sig_cosmo = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    b = make_batch(d, idx, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    base = np.asarray(pred["P_filt_base"])               # (n,4,K) standardized m̂
    resid = np.asarray(pred["P_filt_resid"])             # (n,4,K) standardized r̂
    m_hat = base * sig_marg + mu_marg                    # deployed baseline (logP)
    logP_true = safe_log(d["P_filt"][idx])               # (n,4,K)
    logP_hat = m_hat + sig_cosmo * resid                 # deployed logP̂

    cv = cell_id(d, idx)
    m_cell = np.stack([pf["cell_mean"].get(int(c), mu_marg) for c in cv])
    t_resid = (logP_true - m_cell) / sig_cosmo           # training residual target
    finite = np.isfinite(logP_true)
    return dict(logP_hat=logP_hat, logP_true=logP_true, m_hat=m_hat, r_hat=resid,
                m_cell=m_cell, t_resid=t_resid, sig_cosmo=sig_cosmo, cv=cv,
                z=d["z_grid"][idx], tau0=d["tau0"][idx], alpha=d["alpha_idx"][idx],
                kf=np.nanmedian(np.where(np.isfinite(d["kfkms"][idx]),
                                         d["kfkms"][idx], np.nan), 0),
                finite=finite)


def _band_mask(kf, band):
    return np.isfinite(kf) & (kf >= band[0]) & (kf < band[1])


def _coh_rms(arr_ck, kf, ci, band):
    """Fractional-coherent RMS over the k-bins of a band for class ci, given a
    per-(class,k) fractional COHERENT array (mean over sims)."""
    kb = _band_mask(kf, band)
    v = arr_ck[ci, kb]
    return float(np.sqrt(np.nanmean(v ** 2))), float(np.nanmean(v))


# --------------------------------------------------------------------------- #
# CV floor loader (per-k, interpolated to the cache k-grid).
# --------------------------------------------------------------------------- #
def load_cv_floor(kf):
    """Per-k fractional CV floor on the cache k-grid (nearest-band step function)."""
    if not Path(CV_JSON).exists():
        return None, None
    cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
    bands, vals = [], []
    for band, v in cvj.items():
        lo, hi = (float(x) for x in band.split("-"))
        bands.append((lo, hi)); vals.append(v["median_pct"] / 100.0)
    floor = np.full_like(kf, np.nan, dtype=float)
    for i, k in enumerate(kf):
        if not np.isfinite(k):
            continue
        j = int(np.argmin([abs((lo + hi) / 2 - k) for lo, hi in bands]))
        floor[i] = vals[j]
    band_lookup = {b: v for b, v in zip(bands, vals)}
    return floor, band_lookup


def cv_band(band_lookup, band):
    """Median CV floor over the named band (the nearest stored bands)."""
    if band_lookup is None:
        return np.nan
    sel = [v for (lo, hi), v in band_lookup.items()
           if (lo + hi) / 2 >= band[0] and (lo + hi) / 2 < band[1]]
    if not sel:  # fall back to nearest
        best = min(band_lookup, key=lambda b: abs((b[0]+b[1])/2 - (band[0]+band[1])/2))
        return float(band_lookup[best])
    return float(np.median(sel))


# --------------------------------------------------------------------------- #
# Full decomposition on one fold's val rows (all z & τ₀; cosmology fold-out).
# --------------------------------------------------------------------------- #
def decompose_fold(P, kf):
    """Given deployed_pieces P, compute the (a)-(f) component arrays.

    All components are reported as a per-(class,k) COHERENT FRACTIONAL signal =
    mean over the held-out (sim,z,τ₀) rows of exp(component_logP)-1, so they live
    on the SAME |P̂/P−1| scale the PI reads. Returns dict of (4,K) arrays."""
    fin = P["finite"]
    sc = P["sig_cosmo"]

    # exact logP-space pieces (per row, 4, K):
    e_base = P["m_hat"] - P["m_cell"]                       # (a) baseline mis-fit (logP)
    e_resid = sc * (P["r_hat"] - P["t_resid"])              # (b) residual-head err (logP)
    e_tot = P["logP_hat"] - P["logP_true"]                  # δ (logP) == (a)+(b)
    # sanity: (a)+(b) should equal δ to machine precision.
    recon_err = float(np.nanmax(np.abs((e_base + e_resid) - e_tot)))

    def coh_frac(elog):
        """Per-(class,k) coherent fractional bias = mean over rows of exp(elog)-1."""
        f = np.where(fin, np.exp(elog) - 1.0, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(f, axis=0)                    # (4,K)

    def scat_frac(elog):
        f = np.where(fin, np.exp(elog) - 1.0, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanstd(f, axis=0)                     # (4,K)

    # (c) residual-head θ-MEAN per cell, then RMS of that cell-mean over cells per
    # (class,k). r̂ should average to ~0 over θ within a cell (un-enforced zero-mean).
    # We ALSO measure (i) the sampling floor σ_within/√n_sim (is ⟨r̂⟩_θ significant?)
    # and (ii) the DEVIATION of ⟨r̂⟩_θ from the true held-out-sim cell-mean ⟨t_resid⟩_θ
    # (this deviation, NOT the full ⟨r̂⟩_θ, is what leaks coherently into δ).
    cv = P["cv"]
    cell_means, cell_stds, cell_n, cell_dev = [], [], [], []
    for c in np.unique(cv):
        sel = cv == c
        n = int(sel.sum())
        with np.errstate(invalid="ignore"):
            cm = np.nanmean(np.where(fin[sel], P["r_hat"][sel], np.nan), axis=0)
            cstd = np.nanstd(np.where(fin[sel], P["r_hat"][sel], np.nan), axis=0)
            tm = np.nanmean(np.where(fin[sel], P["t_resid"][sel], np.nan), axis=0)
        cell_means.append(cm); cell_stds.append(cstd); cell_n.append(n)
        cell_dev.append(cm - tm)                            # ⟨r̂⟩_θ − ⟨t_resid⟩_θ
    cell_means = np.stack(cell_means)                       # (Ncell,4,K)
    cell_stds = np.stack(cell_stds); cell_n = np.array(cell_n)
    cell_dev = np.stack(cell_dev)
    with np.errstate(invalid="ignore"):
        rhat_cellmean_rms = np.sqrt(np.nanmean(cell_means ** 2, axis=0))   # (4,K) whitened
        sampling_floor = np.sqrt(np.nanmean(cell_stds ** 2 / cell_n[:, None, None], axis=0))
        rhat_dev_rms = np.sqrt(np.nanmean(cell_dev ** 2, axis=0))          # leak into δ
    # convert the residual θ-mean to a fractional-logP scale via σ_cosmo so it is
    # comparable to (a)/(b): a non-zero ⟨r̂⟩_θ injects σ_cosmo·⟨r̂⟩_θ into logP̂.
    rhat_cellmean_frac = np.expm1(sc * rhat_cellmean_rms)   # (4,K) ~fractional

    out = dict(
        coh_total=coh_frac(e_tot),       # deployed coherent bias (the headline)
        coh_base=coh_frac(e_base),       # (a)
        coh_resid=coh_frac(e_resid),     # (b)
        scat_total=scat_frac(e_tot),     # sim-to-sim scatter (CV-like)
        rhat_theta_mean=rhat_cellmean_frac,   # (c) on fractional/σ_cosmo scale
        rhat_theta_mean_whit=rhat_cellmean_rms,   # (c) whitened (σ_cosmo units)
        rhat_sampling_floor_whit=sampling_floor,  # σ_within/√n_sim (whitened)
        rhat_theta_dev_whit=rhat_dev_rms,         # ⟨r̂⟩_θ−⟨t_resid⟩_θ (the δ leak)
        recon_err=recon_err,
    )
    return out


# --------------------------------------------------------------------------- #
# (d) τ₀-dependence and (e) z-dependence of ⟨δ⟩ at fixed (cls,k).
# --------------------------------------------------------------------------- #
def tau0_z_dependence(P, kf, band_lookup):
    """At fixed (class,k): does ⟨δ⟩ (coherent frac bias) vary with τ₀? with z?

    Returns, per class and per named band, the SLOPE of the coherent bias vs τ₀
    (mean-flux systematic) and vs z, plus the spread (max-min over the grid)."""
    fin = P["finite"]
    e_tot = P["logP_hat"] - P["logP_true"]
    frac = np.where(fin, np.exp(e_tot) - 1.0, np.nan)       # (n,4,K)
    tau0 = P["tau0"]; z = P["z"]
    bands = {"lowk_<0.01": LOWK, "mid_0.01-0.03": MIDK, "highk_>0.03": HIGHK}

    def grid_dependence(coord, ci, band):
        kb = _band_mask(kf, band)
        # per-row band-averaged coherent frac for this class
        with np.errstate(invalid="ignore"):
            v = np.nanmean(frac[:, ci, :][:, kb], axis=1)   # (n,)
        g = np.isfinite(v) & np.isfinite(coord)
        if g.sum() < 5 or np.nanstd(coord[g]) < 1e-9:
            return dict(slope=np.nan, spread=np.nan, corr=np.nan)
        # bin by the coordinate grid to get ⟨δ⟩(coord); the cache coord is discrete.
        uc = np.unique(coord[g])
        means = np.array([np.nanmean(v[g][coord[g] == u]) for u in uc])
        slope = float(np.polyfit(uc, means, 1)[0]) if len(uc) >= 2 else np.nan
        spread = float(np.nanmax(means) - np.nanmin(means))
        corr = float(np.corrcoef(coord[g], v[g])[0, 1]) if g.sum() >= 3 else np.nan
        return dict(slope=slope, spread=spread, corr=corr)

    res = {}
    for bname, band in bands.items():
        res[bname] = {}
        cvf = cv_band(band_lookup, band)
        for ci, nm in enumerate(CLS):
            res[bname][nm] = dict(
                tau0=grid_dependence(tau0, ci, band),
                z=grid_dependence(z, ci, band),
                cv_floor_frac=cvf,
            )
    return res


# --------------------------------------------------------------------------- #
# Fisher-bias (deployed model) — reused per fold. (condensed from gate script.)
# --------------------------------------------------------------------------- #
def fisher_bias_ap(model, d, va, norm, *, z_fid=3.0):
    pf = norm["P_filt"]; sc = pf["sig_cosmo"]; n_k = sc.shape[1]
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - z_fid))], atol=1e-6)
    tau0_fid = float(np.median(d["tau0"][zsel]))
    z_unit_fid = float(np.median(d["x"][zsel, 9]))
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    mask_va = np.isfinite(safe_log(d["P_filt"][va]))
    kvalid = (mask_va[vz].mean(0) > 0.5)
    z_j = jnp.asarray(z_unit_fid)

    def rhat(theta9_tau):
        x = jnp.concatenate([theta9_tau[:9], z_j[None]])
        return model(x, theta9_tau[9])["P_filt_resid"]
    p0 = jnp.asarray(np.concatenate([np.full(9, 0.5), [tau0_fid]]), dtype=jnp.float64)
    Jr = np.asarray(jax.jacfwd(rhat)(p0))                  # (4,n_k,10)
    J_logP = Jr * sc[:, :, None]

    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])
    logP_true = safe_log(d["P_filt"][va][vz])
    m_slice = np.isfinite(logP_true)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j],
                          (logP_hat - logP_true)[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])
    rJ, rd, rC = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
               or not np.isfinite(delta[ci, j]):
                continue
            rJ.append(J_logP[ci, j]); rd.append(delta[ci, j]); rC.append(SIGMA_FRAC[nm] ** 2)
    J = np.array(rJ); dv = np.array(rd); Cinv = 1.0 / np.array(rC)
    J9 = J[:, :9]
    F = (J9.T * Cinv) @ J9
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J9.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    return dict(ap_bias_sigma=float(bias_sigma[1]),
                ap_dtheta_unit=float(dtheta[1]),
                bias_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
                fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
                n_modes=int(len(dv)))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_decomposition(comp, kf, band_lookup, path):
    """Per-class: deployed coherent bias decomposed into (a) baseline, (b) residual
    head, (c) residual θ-mean, vs the CV floor; with the failing bands shaded."""
    floor, _ = load_cv_floor(kf)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    # (c) lives on the σ_cosmo-amplified scale; give it a shared twin axis so the
    # (a)/(b)/δ fractional curves stay readable while (c) is still visible.
    ax_lim = max(2.0, float(np.nanpercentile(np.abs(100 * comp["coh_total"]), 99)) * 1.2)
    clim = max(2.0, float(np.nanpercentile(np.abs(100 * comp["rhat_theta_mean"]), 99)) * 1.1)
    for ci, nm in enumerate(CLS):
        ax = axes[ci]
        g = np.isfinite(kf)
        ax.axhline(0, color="k", lw=0.6)
        ax.semilogx(kf[g], 100 * comp["coh_total"][ci, g], "k-", lw=2.2, label="δ deployed (a+b)")
        ax.semilogx(kf[g], 100 * comp["coh_base"][ci, g], "C0--", lw=1.6, label="(a) baseline/(z,τ₀)")
        ax.semilogx(kf[g], 100 * comp["coh_resid"][ci, g], "C1-.", lw=1.6, label="(b) residual-head")
        if floor is not None:
            ax.fill_between(kf[g], -100 * floor[g], 100 * floor[g], color="grey",
                            alpha=0.18, label="±CV floor (f)")
        # (c) on a twin y-axis (σ_cosmo-amplified scale).
        axc = ax.twinx()
        axc.semilogx(kf[g], 100 * comp["rhat_theta_mean"][ci, g], "C2:", lw=1.6)
        axc.set_ylim(0, clim)
        axc.tick_params(axis="y", labelcolor="C2", labelsize=7)
        if ci == 3:
            axc.set_ylabel("(c) ⟨r̂⟩_θ cell-RMS [%, σ_cosmo scale]", color="C2", fontsize=8)
        # shade the failing bands relevant to this class
        if nm in ("clean", "subDLA"):
            ax.axvspan(1e-4, LOWK[1], color="red", alpha=0.06)
        if nm == "DLA":
            ax.axvspan(HIGHK[0], HIGHK[1], color="red", alpha=0.06)
        ax.axvline(KPIVOT, color="purple", ls=":", lw=0.8)
        ax.set_title(nm); ax.set_xlabel("k [s/km]"); ax.grid(alpha=0.3, which="both")
        ax.set_ylim(-ax_lim, ax_lim)
        if ci == 0:
            ax.set_ylabel("coherent bias ⟨P̂/P−1⟩ [%]")
            # combined legend (left axis lines + the (c) twin line)
            h, l = ax.get_legend_handles_labels()
            h.append(plt.Line2D([], [], color="C2", ls=":", lw=1.6)); l.append("(c) ⟨r̂⟩_θ (right axis)")
            ax.legend(h, l, fontsize=7.5)
    fig.suptitle("Residual decomposition (deployed LF emulator, all held-out z & τ₀): "
                 "δ = (a) baseline mis-fit + (b) residual-head cosmology error; "
                 "(c) residual θ-mean (right axis); (f) CV floor")
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def fig_perfold_tilt(perfold, path):
    """Per-fold coherent tilt for the failing bands + A_p Fisher-bias per fold."""
    folds = sorted(perfold)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # panel 1: clean low-k coherent RMS per fold
    for key, nm, c in (("clean_lowk", "clean low-k (<0.01)", "C0"),
                       ("subdla_lowk", "subDLA low-k (<0.01)", "C2"),
                       ("dla_highk", "DLA high-k (>0.03)", "C3")):
        vals = [perfold[f][key] * 100 for f in folds]
        axes[0].plot(folds, vals, "-o", color=c, label=nm)
        m, s = np.nanmean(vals), np.nanstd(vals)
        axes[0].axhline(m, color=c, ls=":", lw=0.8)
    axes[0].set_xlabel("LOSO fold"); axes[0].set_ylabel("coherent |⟨P̂/P−1⟩| RMS [%]")
    axes[0].set_title("Coherent tilt in the failing bands, per fold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # panel 2: signed band-mean (the TILT direction) per fold
    for key, nm, c in (("clean_lowk_mean", "clean low-k", "C0"),
                       ("subdla_lowk_mean", "subDLA low-k", "C2"),
                       ("dla_highk_mean", "DLA high-k", "C3")):
        vals = [perfold[f][key] * 100 for f in folds]
        axes[1].plot(folds, vals, "-o", color=c, label=nm)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("LOSO fold"); axes[1].set_ylabel("signed coherent ⟨P̂/P−1⟩ [%]")
    axes[1].set_title("Signed coherent tilt (sign consistency = real systematic)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # panel 3: A_p Fisher-bias per fold
    apb = [perfold[f]["ap_bias_sigma"] for f in folds]
    axes[2].plot(folds, apb, "-o", color="C1")
    axes[2].axhline(0.2, color="r", ls="--", lw=1, label="0.2σ gate")
    axes[2].axhline(-0.2, color="r", ls="--", lw=1)
    m, s = np.nanmean(apb), np.nanstd(apb)
    axes[2].axhline(m, color="C1", ls=":", lw=0.9, label=f"mean={m:+.3f}±{s:.3f}σ")
    axes[2].set_xlabel("LOSO fold"); axes[2].set_ylabel("A_p Fisher-bias [σ]")
    axes[2].set_title("A_p bias per fold (does fold-0's 0.03σ hold?)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached per-fold checkpoints if present")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--nbasis-test", action="store_true",
                    help="also train fold-0 at n_basis=48 (representation-rank test)")
    args = ap.parse_args()

    d = load_cache(CACHE)
    n_k = d["P_tier_p"].shape[1]
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"]), d["kfkms"], np.nan), 0)
    floor, band_lookup = load_cv_floor(kf)
    print(f"cache: {len(d['z_grid'])} rows, n_k={n_k}, "
          f"{len(set(d['sim_name']))} sims; CV floor loaded={floor is not None}")

    # ---- per-fold loop: deployed model, decomposition, Fisher A_p ----
    perfold = {}
    fold0_comp = fold0_dep = None
    for fold in range(args.folds):
        model, norm, (tr, va, ho) = train_or_load_fold(
            d, fold, n_k, n_basis=24, reuse=args.reuse)
        P = deployed_pieces(model, d, va, norm)
        comp = decompose_fold(P, kf)
        dep = tau0_z_dependence(P, kf, band_lookup)
        fb = fisher_bias_ap(model, d, va, norm, z_fid=3.0)

        clean_lowk_rms, clean_lowk_mean = _coh_rms(comp["coh_total"], kf, 0, LOWK)
        subdla_lowk_rms, subdla_lowk_mean = _coh_rms(comp["coh_total"], kf, 2, LOWK)
        dla_highk_rms, dla_highk_mean = _coh_rms(comp["coh_total"], kf, 3, HIGHK)
        perfold[fold] = dict(
            clean_lowk=clean_lowk_rms, clean_lowk_mean=clean_lowk_mean,
            subdla_lowk=subdla_lowk_rms, subdla_lowk_mean=subdla_lowk_mean,
            dla_highk=dla_highk_rms, dla_highk_mean=dla_highk_mean,
            ap_bias_sigma=fb["ap_bias_sigma"], ap_dtheta_unit=fb["ap_dtheta_unit"],
            fisher_cond=fb["fisher_cond"], n_modes=fb["n_modes"],
            recon_err=comp["recon_err"])
        print(f"  fold {fold}: clean<0.01={clean_lowk_mean*100:+.2f}% "
              f"subDLA<0.01={subdla_lowk_mean*100:+.2f}% DLA>0.03={dla_highk_mean*100:+.2f}% "
              f"| A_p bias={fb['ap_bias_sigma']:+.3f}σ | recon_err={comp['recon_err']:.1e}")
        if fold == 0:
            fold0_comp, fold0_dep, fold0_P = comp, dep, P

    # ---- representation-rank test: n_basis 24 -> 48 on fold 0 ----
    nbasis_result = None
    if args.nbasis_test:
        print("\n[rank test] training fold-0 at n_basis=48 ...")
        model48, norm48, (tr, va, ho) = train_or_load_fold(
            d, 0, n_k, n_basis=48, reuse=args.reuse, tag="decomp")
        P48 = deployed_pieces(model48, d, va, norm48)
        comp48 = decompose_fold(P48, kf)
        c24 = fold0_comp; c48 = comp48
        nbasis_result = {}
        for label, ci, band in (("clean_lowk", 0, LOWK), ("subdla_lowk", 2, LOWK),
                                 ("dla_highk", 3, HIGHK)):
            r24, _ = _coh_rms(c24["coh_total"], kf, ci, band)
            r48, _ = _coh_rms(c48["coh_total"], kf, ci, band)
            # the (b) residual-head share of the band-edge coherent residual
            b24, _ = _coh_rms(c24["coh_resid"], kf, ci, band)
            b48, _ = _coh_rms(c48["coh_resid"], kf, ci, band)
            nbasis_result[label] = dict(total_rms_nb24=r24, total_rms_nb48=r48,
                                        resid_rms_nb24=b24, resid_rms_nb48=b48,
                                        reduced=bool(r48 < r24 - 1e-4))
            print(f"  {label}: total {r24*100:.2f}% -> {r48*100:.2f}%  "
                  f"(resid-head {b24*100:.2f}% -> {b48*100:.2f}%) "
                  f"{'REDUCED' if r48 < r24 - 1e-4 else 'no improvement'}")

    # ---- figures (fold 0) ----
    fig_decomposition(fold0_comp, kf, band_lookup,
                      f"{OUT}/resid_decomposition_perclass.png")
    fig_perfold_tilt(perfold, f"{OUT}/resid_perfold_tilt.png")

    # ---- per-fold consistency summary ----
    def stat(key):
        v = np.array([perfold[f][key] for f in perfold])
        return dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)),
                    min=float(np.nanmin(v)), max=float(np.nanmax(v)))

    # ---- assemble JSON ----
    def comp_band(comp, ci, band):
        rms, mean = _coh_rms(comp["coh_total"], kf, ci, band)
        rms_a, mean_a = _coh_rms(comp["coh_base"], kf, ci, band)
        rms_b, mean_b = _coh_rms(comp["coh_resid"], kf, ci, band)
        rms_c, mean_c = _coh_rms(comp["rhat_theta_mean"], kf, ci, band)
        scat, _ = _coh_rms(comp["scat_total"], kf, ci, band)
        return dict(total_coh_rms=rms, total_coh_mean=mean,
                    a_baseline_rms=rms_a, a_baseline_mean=mean_a,
                    b_residhead_rms=rms_b, b_residhead_mean=mean_b,
                    c_rhat_thetamean_rms=rms_c,
                    scatter_rms=scat, cv_floor_frac=cv_band(band_lookup, band))

    decomp_json = {}
    for nm, ci, band in (("clean@lowk", 0, LOWK), ("subDLA@lowk", 2, LOWK),
                         ("DLA@highk", 3, HIGHK), ("clean@mid", 0, MIDK),
                         ("DLA@mid", 3, MIDK), ("clean@highk", 0, HIGHK)):
        decomp_json[nm] = comp_band(fold0_comp, ci, band)

    # (c) significance: is ⟨r̂⟩_θ a real systematic (vs sampling floor), and how much
    # of it actually LEAKS into δ (deviation from the true held-out-sim cell-mean)?
    c_sig = {}
    for ci, cnm in enumerate(CLS):
        whit = float(np.sqrt(np.nanmean(fold0_comp["rhat_theta_mean_whit"][ci] ** 2)))
        floor_c = float(np.sqrt(np.nanmean(fold0_comp["rhat_sampling_floor_whit"][ci] ** 2)))
        dev = float(np.sqrt(np.nanmean(fold0_comp["rhat_theta_dev_whit"][ci] ** 2)))
        c_sig[cnm] = dict(rhat_thetamean_whit=whit, sampling_floor_whit=floor_c,
                          significance_ratio=whit / max(floor_c, 1e-9),
                          deviation_from_true_cellmean_whit=dev,
                          note="whit=σ_cosmo units; significance>1 => real non-zero "
                               "θ-mean; deviation = what leaks into δ (small => head "
                               "tracks the held-out-sim cell-mean).")
    decomp_json["c_rhat_theta_mean_significance"] = c_sig

    out = {
        "what_residual_averages_over": {
            "deployed_delta": "δ(sim,z,τ₀,k)=logP̂−logP_true over LOSO-held-out rows; "
                              "rows span ALL 18 z × 20 τ₀(=alpha) of every held-out sim.",
            "fig07_perclass_frac_error": "pools ALL val rows -> averages OVER z AND τ₀.",
            "deployed_coherent_vs_cv_plot": "SLICES z=3.0, averages over τ₀(=alpha) at z=3; "
                                            "coherent=mean over val sims, scatter=std over val sims.",
            "coherent_vs_cv_split_def": "coherent = mean over val SIMS of (P̂/P−1) at fixed (cls,k); "
                                        "scatter = std over val sims; (a)=baseline mis-fit "
                                        "(m̂−m_cell), (b)=σ_cosmo·(r̂−t_resid).",
            "answer": "These plots DO include τ₀ and z variation — they MARGINALISE over them. "
                      "They are NOT pure cosmology errors; the τ₀/z systematic and the cosmology "
                      "error are entangled in the headline ⟨P̂/P−1⟩. This decomposition separates them.",
        },
        "fold0_decomposition_by_band": decomp_json,
        "fold0_tau0_z_dependence": fold0_dep,
        "perfold": perfold,
        "multifold_consistency": {
            "clean_lowk_mean": stat("clean_lowk_mean"),
            "subdla_lowk_mean": stat("subdla_lowk_mean"),
            "dla_highk_mean": stat("dla_highk_mean"),
            "clean_lowk_rms": stat("clean_lowk"),
            "dla_highk_rms": stat("dla_highk"),
            "ap_bias_sigma": stat("ap_bias_sigma"),
        },
        "nbasis_rank_test": nbasis_result,
        "config": {"deployed": DEPLOYED, "n_folds": args.folds,
                   "bands": {"lowk": LOWK, "mid": MIDK, "highk": HIGHK}},
    }
    jpath = f"{OUT}/resid_decomposition.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jpath}")

    # ---- console summary ----
    print("\n===== MULTI-FOLD CONSISTENCY (signed coherent tilt, % ; mean±std over folds) =====")
    for key, lab in (("clean_lowk_mean", "clean  k<0.01"),
                     ("subdla_lowk_mean", "subDLA k<0.01"),
                     ("dla_highk_mean", "DLA    k>0.03")):
        s = stat(key)
        print(f"  {lab:14}: {s['mean']*100:+.2f}% ± {s['std']*100:.2f}%  "
              f"[{s['min']*100:+.2f}, {s['max']*100:+.2f}]")
    s = stat("ap_bias_sigma")
    print(f"  A_p Fisher-bias: {s['mean']:+.3f}σ ± {s['std']:.3f}σ  "
          f"[{s['min']:+.3f}, {s['max']:+.3f}]   (fold-0 gate was 0.03σ)")


if __name__ == "__main__":
    main()
