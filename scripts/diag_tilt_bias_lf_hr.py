"""Does the emulator's k-TILT prediction error bias n_s / A_p — and is the
HIGH-FIDELITY (HR) suite clean across the KODIAQ k-range so the multi-fidelity
result is safe?

This is a CORRECTNESS diagnostic, not a production-model change. It rebuilds the
deeper-baseline + hard-residual recipe IN THIS SCRIPT (reusing the model/training
approach proven in scripts/feasibility_subpercent.py) and runs it on BOTH the LF
and the HR τ₀ caches, fold-0 LOSO. Three pieces:

  1. TRAIN (per fidelity): ≥3-layer (w256) θ-blind baseline head + Σ(weight·mask)
     loss (no inv_nc shrink) + hard residual head (encoder+HeadB, rank 24, ~250 ep).
     This is the post-recipe deployed emulator whose error we interrogate.

  2. TILT diagnostic (per fidelity, per class): deployed fractional error
     δ_frac = P̂/P − 1 on held-out (LOSO val) sims, as a function of k. The TILT is
     slope(⟨δ_frac⟩ vs log10 k) plus the low-k vs high-k means. Does the recipe
     REMOVE the LF tilt? Is there a residual tilt in HR across the KODIAQ band
     0.07–0.2 s/km? LF and HR plotted side by side.

  3. FISHER-BIAS projection (the decisive check). J = the EMULATOR autodiff
     ∂(logP̂)/∂θ (jax.jacfwd over the 9 unit-cube params, through the trained
     residual model; the baseline is θ-blind so ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ) at a
     fiducial cosmology+z. δ = the deployed log-error (logP̂ − logP_true) on
     held-out sims. C = a DESI-DR1-like diagonal per-mode covariance (per-mode %
     error on the clean channel; the HCD channels carry the 15% DLA systematic
     floor from docs/superpowers/2026-06-01-hcd-marginalization-literature.md,
     Karaçaylı+2025). Bias δθ_i = [(JᵀC⁻¹J)⁻¹JᵀC⁻¹δ]_i in units of the Fisher
     posterior σ_i = sqrt(diag (JᵀC⁻¹J)⁻¹). Reported for ALL 9 params, focus on
     ns & Ap, for LF and HR separately over their own k-ranges.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_tilt_bias_lf_hr.py
Writes figures + a JSON summary to figures/analysis/04_emulator/.
"""
from __future__ import annotations

import json
import time

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (load_cache, make_splits, cell_id, safe_log,
                                        fit_baseline_residual_norm)
from hcd_analysis.emulator.model import Encoder, HeadB, svd_basis_init

OUT = "figures/analysis/04_emulator"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")

# Survey k-band of interest: the KODIAQ high-k that the MF result rests on.
KODIAQ_BAND = (0.07, 0.20)        # s/km
# Low/high split for the tilt means: split each fidelity at the geometric centre
# of its OWN finite k-range (set per-fidelity at runtime).

# DESI-DR1-like diagonal per-mode covariance assumption (Fisher C).
#   clean P1D: 3% per-mode statistical+continuum error (DESI DR1 ~few-% per (k,z)).
#   HCD channels (LLS/subDLA/DLA): the 15% DLA systematic FLOOR
#     (Karaçaylı+2025, σ_DLA = 0.15·r_DLA·P_smooth; see
#      docs/superpowers/2026-06-01-hcd-marginalization-literature.md L54).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}

CACHES = {
    "LF": ("hcd_analysis/_emulator_data/observables_tau0_lf.h5", 8, 172),
    "HR": ("hcd_analysis/_emulator_data/observables_tau0_hr.h5", 6, 525),
}


# ===========================================================================
# Recipe models (rebuilt here; reuse the feasibility_subpercent.py approach)
# ===========================================================================
class DeepBaseline(eqx.Module):
    """θ-blind (z,τ₀)→(4,n_k) baseline, ≥3 hidden layers, low-rank decode.

    Same construction as feasibility_subpercent.DeepBaseline (the deeper baseline
    that the recipe needs to drop term (b) from ~0.84·σ_cosmo to << σ_cosmo)."""
    layers: list
    basis: jax.Array
    nb: int = eqx.field(static=True)
    n_k: int = eqx.field(static=True)

    def __init__(self, nb, width, depth, n_k, binit, key):
        ks = jax.random.split(key, depth + 1)
        dims = [2] + [width] * depth
        self.layers = ([eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(depth)]
                       + [eqx.nn.Linear(width, 4 * nb, key=ks[depth])])
        self.basis = jnp.asarray(binit)
        self.nb = nb
        self.n_k = n_k

    def __call__(self, z, tau0):
        x = jnp.stack([jnp.atleast_1d(z)[0], jnp.atleast_1d(tau0)[0]])
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        c = self.layers[-1](x).reshape(4, self.nb)
        return c @ self.basis


class ResidModel(eqx.Module):
    """encoder + HeadB residual path (the production architecture, θ-dependent)."""
    enc: Encoder
    head: HeadB
    n_k: int = eqx.field(static=True)

    def __init__(self, nb, n_k, binit, key):
        k1, k2 = jax.random.split(key)
        self.enc = Encoder(in_dim=10, key=k1)
        self.head = HeadB(latent=64, n_k=n_k, n_basis=nb, p_filt_basis_init=binit, key=k2)
        self.n_k = n_k

    def __call__(self, x, tau0):
        return self.head(self.enc(x), tau0)["P_filt_resid"]


def _cell_table(d, idx, cm, mu, sm, n_k):
    """Per distinct (z,τ₀)-cell: input (z_unit, τ₀) + σ_marg-standardized cell-mean."""
    c = cell_id(d, idx)
    z_unit = d["x"][:, 9]
    uc = np.unique(c)
    cz, ct, cy = [], [], []
    for cc in uc:
        rows = idx[c == cc]
        cz.append(z_unit[rows][0]); ct.append(d["tau0"][rows][0]); cy.append(cm[int(cc)])
    cz, ct, cy = np.array(cz), np.array(ct), np.stack(cy)
    tb = (cy - mu) / sm
    return cz, ct, tb, np.isfinite(tb), uc


def _train_baseline(cz, ct, tb, mask, nb, width, depth, n_k, epochs, lr, seed=0):
    M = tb.reshape(-1, n_k); M = M[np.all(np.isfinite(M), 1)]
    binit = np.asarray(svd_basis_init(jnp.asarray(M), nb))
    head = DeepBaseline(nb, width, depth, n_k, binit, jax.random.PRNGKey(seed))
    Z, T = jnp.asarray(cz), jnp.asarray(ct)
    Y = jnp.asarray(np.where(mask, tb, 0.0)); Mk = jnp.asarray(mask.astype(jnp.float64))

    def loss(h):                                   # inv_nc FIX: uniform Σ(mask) norm
        pred = jax.vmap(h)(Z, T)
        diff = jnp.where(Mk > 0, pred - Y, 0.0)
        return jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(Mk), 1.0)

    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs), weight_decay=1e-7)
    st = opt.init(eqx.filter(head, eqx.is_array))

    @eqx.filter_jit
    def step(h, st):
        l, g = eqx.filter_value_and_grad(loss)(h)
        u, st = opt.update(g, st, eqx.filter(h, eqx.is_array))
        return eqx.apply_updates(h, u), st, l
    for _ in range(epochs):
        head, st, _ = step(head, st)
    return head


def _resid_target(d, idx, cm, sc, n_k):
    logP = safe_log(d["P_filt"])
    c = cell_id(d, idx)
    T = np.empty((len(idx), 4, n_k))
    for i, (r, ci) in enumerate(zip(idx, c)):
        T[i] = (logP[r] - cm[int(ci)]) / sc
    return T


def _train_resid(d, tr, Ttr, mtr, nb, n_k, epochs, lr, bs=512, seed=0, wd=1e-5):
    M = Ttr.reshape(-1, n_k); M = M[np.all(np.isfinite(M), 1)]
    binit = np.asarray(svd_basis_init(jnp.asarray(M), nb))
    m = ResidModel(nb, n_k, binit, jax.random.PRNGKey(seed))
    Xt = jnp.asarray(d["x"][tr]); Tt = jnp.asarray(d["tau0"][tr])
    Yt = jnp.asarray(np.where(mtr, Ttr, 0.0)); Mt = jnp.asarray(mtr.astype(jnp.float64))
    n = len(tr); bs = min(bs, n)
    spe = int(np.ceil(n / bs)); tot = spe * epochs
    opt = optax.adamw(optax.cosine_decay_schedule(lr, tot), weight_decay=wd)
    st = opt.init(eqx.filter(m, eqx.is_array))

    def loss(m, xb, tb, yb, mb):
        pred = jax.vmap(m)(xb, tb)
        diff = jnp.where(mb > 0, pred - yb, 0.0)
        return jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(mb), 1.0)

    @eqx.filter_jit
    def step(m, st, xb, tb, yb, mb):
        l, g = eqx.filter_value_and_grad(loss)(m, xb, tb, yb, mb)
        u, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), st, l

    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < bs:
                b = np.concatenate([b, perm[:bs - len(b)]])
            m, st, _ = step(m, st, Xt[b], Tt[b], Yt[b], Mt[b])
    return m


# ===========================================================================
# Train fold-0 on a fidelity and return the deployed prediction on val rows
# ===========================================================================
def train_fold0(fid):
    path, n_folds, n_k = CACHES[fid]
    d = load_cache(path)
    tr, va, ho = make_splits(d, 0, n_folds)
    pf = fit_baseline_residual_norm(d, tr)
    mu, sm, sc, cm = pf["mu_marg"], pf["sig_marg"], pf["sig_cosmo"], pf["cell_mean"]

    print(f"\n[{fid}] fold-0: train={len(tr)} val={len(va)} holdout={len(ho)} "
          f"| n_sim={len(set(d['sim_name']))} n_k={n_k} "
          f"| val sims={len(set(d['sim_name'][va]))}")

    # --- recipe: deep baseline (uniform-mask loss) + hard residual (rank 24) ----
    cz, ct, tb, bmask, _ = _cell_table(d, tr, cm, mu, sm, n_k)
    t0 = time.time()
    head_b = _train_baseline(cz, ct, tb, bmask, nb=48, width=256, depth=3,
                             n_k=n_k, epochs=15000, lr=2e-3)
    Ttr = _resid_target(d, tr, cm, sc, n_k); mtr = np.isfinite(Ttr)
    m_r = _train_resid(d, tr, Ttr, mtr, nb=24, n_k=n_k, epochs=250, lr=1e-3)
    print(f"[{fid}] trained baseline+residual in {time.time()-t0:.0f}s")

    # --- deployed reconstruction on val rows -----------------------------------
    z_va = jnp.asarray(d["x"][va, 9]); tau_va = jnp.asarray(d["tau0"][va])
    Xva = jnp.asarray(d["x"][va])
    m_hat = np.asarray(jax.vmap(head_b)(z_va, tau_va)) * sm + mu      # (n,4,K) logP
    r_hat = np.asarray(jax.vmap(m_r)(Xva, tau_va))                   # (n,4,K)
    logP_hat = m_hat + sc * r_hat
    logP_true = safe_log(d["P_filt"][va])
    mask_va = np.isfinite(logP_true)
    kf = d["kfkms"][va]                                             # (n,K) per-row k

    return dict(fid=fid, d=d, tr=tr, va=va, pf=pf, head_b=head_b, m_r=m_r,
                logP_hat=logP_hat, logP_true=logP_true, mask_va=mask_va,
                kf=kf, n_k=n_k)


# ===========================================================================
# TILT diagnostic
# ===========================================================================
def tilt_diagnostic(R):
    """Per-class deployed frac error vs k; tilt = slope of ⟨frac⟩ vs log10 k.

    Builds a per-class median frac-error profile on a common finite k-grid (the
    cache k-grid is the same across rows up to the Nyquist mask), then fits a
    line in log10 k and reports low-/high-k band means + the KODIAQ-band mean."""
    fid = R["fid"]
    frac = np.exp(R["logP_hat"] - R["logP_true"]) - 1.0    # (n,4,K) deployed frac err
    kf = R["kf"]; mask = R["mask_va"]
    # representative k per bin = median finite k across rows (cache grid is ~fixed)
    kbin = np.nanmedian(np.where(np.isfinite(kf), kf, np.nan), 0)   # (K,)
    out = {}
    profiles = {}
    # per-fidelity low/high split at geometric centre of the finite k-range
    kfin = kbin[np.isfinite(kbin)]
    ksplit = np.sqrt(kfin.min() * kfin.max())
    for ci, nm in enumerate(CLS):
        mm = mask[:, ci, :]
        with np.errstate(invalid="ignore"):
            prof = np.array([np.median(frac[:, ci, j][mm[:, j]]) if mm[:, j].any()
                             else np.nan for j in range(R["n_k"])])
        profiles[nm] = prof
        good = np.isfinite(prof) & np.isfinite(kbin) & (kbin > 0)
        if good.sum() < 3:
            out[nm] = dict(slope=np.nan, slope_datak=np.nan, lowk_mean=np.nan,
                           highk_mean=np.nan, kodiaq_mean=np.nan,
                           kodiaq_absmean=np.nan, n_kodiaq=0)
            continue
        x = np.log10(kbin[good]); y = prof[good]
        slope = float(np.polyfit(x, y, 1)[0])              # per dex of k (full range)
        # data-k slope: exclude the extreme low-k bins (k<1e-3 s/km) where the known
        # below-data-k_min spike lives, so the headline tilt reflects the survey band.
        gdat = good & (kbin >= 1e-3)
        slope_datak = (float(np.polyfit(np.log10(kbin[gdat]), prof[gdat], 1)[0])
                       if gdat.sum() >= 3 else np.nan)
        lo = good & (kbin < ksplit); hi = good & (kbin >= ksplit)
        kod = good & (kbin >= KODIAQ_BAND[0]) & (kbin <= KODIAQ_BAND[1])
        out[nm] = dict(
            slope=slope,
            slope_datak=slope_datak,
            lowk_mean=float(np.mean(prof[lo])) if lo.any() else np.nan,
            highk_mean=float(np.mean(prof[hi])) if hi.any() else np.nan,
            kodiaq_mean=float(np.mean(prof[kod])) if kod.any() else np.nan,
            kodiaq_absmean=float(np.mean(np.abs(prof[kod]))) if kod.any() else np.nan,
            n_kodiaq=int(kod.sum()))
    return dict(kbin=kbin, profiles=profiles, stats=out, ksplit=float(ksplit),
                kfin_min=float(kfin.min()), kfin_max=float(kfin.max()))


# ===========================================================================
# FISHER-BIAS projection (the decisive correctness check)
# ===========================================================================
def fisher_bias(R, fiducial_param_unit, z_fid=3.0):
    """δθ = (JᵀC⁻¹J)⁻¹ JᵀC⁻¹ δ in units of the Fisher posterior σ.

    J  = ∂logP̂/∂θ over the 9 unit-cube params (jax.jacfwd through the trained
         residual model; baseline is θ-blind so ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ),
         stacked over the 4 classes × finite-k bins at the fiducial (θ, z, τ₀).
    δ  = the deployed log error (logP̂ − logP_true), per (class,k), averaged over
         held-out val rows in the fiducial z-slice (the coherent k-tilt).
    C  = diagonal per-mode covariance, (σ_frac·1)² in log-P space per class
         (a fractional per-mode error ≈ a log-P std), DESI-DR1-like.

    Done over the fidelity's OWN finite k-range (LF: ≤0.098; HR: ≤0.29 incl KODIAQ).
    """
    fid = R["fid"]; d = R["d"]; pf = R["pf"]; m_r = R["m_r"]
    sc = pf["sig_cosmo"]                                    # (4,K)
    n_k = R["n_k"]

    # fiducial τ₀ at z_fid: median τ₀ over cache rows at that z (the data anchor).
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - z_fid))], atol=1e-6)
    tau0_fid = float(np.median(d["tau0"][zsel]))
    z_unit_fid = float(np.median(d["x"][zsel, 9]))

    # finite-k mask at the fiducial slice: bins finite for >50% of val rows in z.
    va = R["va"]; vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    mask_va = R["mask_va"]
    kvalid = (mask_va[vz].mean(0) > 0.5)                   # (4,K) per class
    kf = np.nanmedian(np.where(np.isfinite(R["kf"][vz]), R["kf"][vz], np.nan), 0)

    # ----- J = σ_cosmo · ∂r̂/∂θ at fiducial (autodiff over the 9 unit params) ---
    z_unit_j = jnp.asarray(z_unit_fid)
    tau_j = jnp.asarray(tau0_fid)

    def rhat_of_theta(theta9):
        # x = [params_unit(9), z_unit(1)]; only the 9 params vary.
        x = jnp.concatenate([theta9, z_unit_j[None]])
        return m_r(x, tau_j)                                # (4, n_k) residual

    theta0 = jnp.asarray(fiducial_param_unit, dtype=jnp.float64)
    # jacfwd: (4, n_k, 9)
    Jr = np.asarray(jax.jacfwd(rhat_of_theta)(theta0))
    J_logP = Jr * sc[:, :, None]                            # ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ

    # ----- δ = deployed log error at the fiducial z-slice, per (class,k) --------
    dlog = R["logP_hat"][vz] - R["logP_true"][vz]           # (m,4,K)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(mask_va[vz][:, ci, j],
                                               dlog[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])  # (4,K)

    # ----- stack the finite modes, build C diag (σ_frac in log-P) ---------------
    rows_J, rows_d, rows_C, mode_class = [], [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j]:
                continue
            if not np.all(np.isfinite(J_logP[ci, j])):
                continue
            if not np.isfinite(delta[ci, j]):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2); mode_class.append(ci)
    J = np.array(rows_J)                                   # (M, 9)
    dv = np.array(rows_d)                                  # (M,)
    Cinv = 1.0 / np.array(rows_C)                          # (M,) diagonal
    mode_class = np.array(mode_class)

    # ----- Fisher F = JᵀC⁻¹J, posterior σ, bias δθ ------------------------------
    F = (J.T * Cinv) @ J                                   # (9,9)
    # tiny ridge for numerical stability of the inverse (degenerate IGM dirs)
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))                         # Fisher posterior σ_i
    JtCd = (J.T * Cinv) @ dv                               # (9,)
    dtheta = Finv @ JtCd                                   # bias in unit-cube θ
    bias_sigma = dtheta / sigma                            # bias in σ units

    # how many of the stacked modes fall in the KODIAQ high-k band (HR's leverage)
    kod_mode_count = int(sum(
        (kf[j] >= KODIAQ_BAND[0]) and (kf[j] <= KODIAQ_BAND[1])
        for _ci, j in _iter_modes(kvalid, J_logP, delta, n_k)))

    return dict(
        fiducial_unit=list(map(float, fiducial_param_unit)),
        z_fid=z_fid, tau0_fid=tau0_fid,
        n_modes=int(len(dv)),
        n_modes_kodiaq=kod_mode_count,
        n_modes_per_class={CLS[ci]: int((mode_class == ci).sum()) for ci in range(4)},
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)},
        dtheta_unit={PARAMS[i]: float(dtheta[i]) for i in range(9)},
        bias_in_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        # condition number flags how degenerate the 9-param Fisher is over this k-range
        fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
    )


def _iter_modes(kvalid, J_logP, delta, n_k):
    for ci in range(4):
        for j in range(n_k):
            if kvalid[ci, j] and np.all(np.isfinite(J_logP[ci, j])) and np.isfinite(delta[ci, j]):
                yield ci, j


# ===========================================================================
# Figures
# ===========================================================================
def fig_tilt(T_lf, T_hr):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for col, (fid, T) in enumerate((("LF", T_lf), ("HR", T_hr))):
        a = ax[col]
        for ci, nm in enumerate(CLS):
            prof = T["profiles"][nm]; kbin = T["kbin"]
            g = np.isfinite(prof) & np.isfinite(kbin)
            a.plot(kbin[g], prof[g] * 100, lw=1.4, label=nm)
        a.axhline(0, color="k", lw=0.7)
        a.axvspan(*KODIAQ_BAND, color="C1", alpha=0.10,
                  label="KODIAQ 0.07–0.2")
        a.set_xscale("log")
        a.set_xlabel("k [s/km]")
        a.set_title(f"{fid}: deployed ⟨P̂/P−1⟩ vs k (post-recipe)\n"
                    f"k∈[{T['kfin_min']:.1e},{T['kfin_max']:.1e}]")
        a.grid(alpha=0.3)
        # annotate clean-class tilt (full-range and data-k≥1e-3 slope)
        sc = T["stats"]["clean"]
        a.text(0.04, 0.96,
               f"clean tilt: {sc['slope']*100:+.2f}%/dex (full)\n"
               f"            {sc['slope_datak']*100:+.2f}%/dex (k≥1e-3)\n"
               f"KODIAQ |⟨frac⟩| = {sc['kodiaq_absmean']*100:.2f}%",
               transform=a.transAxes, va="top", fontsize=8.5,
               bbox=dict(fc="w", alpha=0.75, ec="0.6"))
    ax[0].set_ylabel("median fractional error [%]")
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/diag_tilt_lf_vs_hr.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/diag_tilt_lf_vs_hr.png")


def fig_bias(B_lf, B_hr):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xs = np.arange(9); bw = 0.38
    bl = [abs(B_lf["bias_in_sigma"][p]) for p in PARAMS]
    bh = [abs(B_hr["bias_in_sigma"][p]) for p in PARAMS]
    ax.bar(xs - bw / 2, bl, bw, label="LF (k≤0.098)", color="C0")
    ax.bar(xs + bw / 2, bh, bw, label="HR (k≤0.29, incl KODIAQ)", color="C3")
    ax.axhline(0.2, color="k", ls="--", lw=0.9, label="0.2σ threshold")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, label="1σ")
    ax.set_xticks(xs); ax.set_xticklabels(PARAMS, rotation=30, ha="right")
    ax.set_ylabel("|implied bias δθ| / σ_Fisher")
    ax.set_title("Fisher-projected emulator-error bias per parameter (fold-0)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
    # highlight ns & Ap
    for i in (0, 1):
        ax.get_xticklabels()[i].set_color("C3")
        ax.get_xticklabels()[i].set_fontweight("bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/diag_bias_per_param.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/diag_bias_per_param.png")


def main():
    t0 = time.time()
    R_lf = train_fold0("LF")
    R_hr = train_fold0("HR")

    print("\n=== TILT diagnostic (deployed ⟨P̂/P−1⟩ vs k) ===")
    T_lf = tilt_diagnostic(R_lf)
    T_hr = tilt_diagnostic(R_hr)
    for fid, T in (("LF", T_lf), ("HR", T_hr)):
        print(f"  [{fid}] k∈[{T['kfin_min']:.2e},{T['kfin_max']:.2e}], "
              f"low/high split @ {T['ksplit']:.3e} s/km")
        for nm in CLS:
            s = T["stats"][nm]
            print(f"     {nm:7}: slope={s['slope']*100:+.3f}%/dex "
                  f"(data-k {s['slope_datak']*100:+.3f}%/dex)  "
                  f"lowk={s['lowk_mean']*100:+.3f}%  highk={s['highk_mean']*100:+.3f}%  "
                  f"KODIAQ⟨⟩={s['kodiaq_mean']*100:+.3f}% (|·|={s['kodiaq_absmean']*100:.3f}%, "
                  f"n={s['n_kodiaq']})")

    # Fisher bias at a fiducial = the centre of the unit cube (mid-design),
    # z_fid=3.0 (the KODIAQ-relevant redshift used in the τ0-response figs).
    fiducial = np.full(9, 0.5)
    print("\n=== FISHER-BIAS projection (δθ in σ units; fiducial=cube centre, z=3) ===")
    B_lf = fisher_bias(R_lf, fiducial, z_fid=3.0)
    B_hr = fisher_bias(R_hr, fiducial, z_fid=3.0)
    for fid, B in (("LF", B_lf), ("HR", B_hr)):
        print(f"  [{fid}] {B['n_modes']} modes "
              f"(per-class {B['n_modes_per_class']}), Fisher cond={B['fisher_cond']:.2e}")
        for p in PARAMS:
            print(f"     {p:11}: bias={B['bias_in_sigma'][p]:+.3f}σ  "
                  f"(σ_Fisher={B['sigma_fisher'][p]:.4f}, δθ_unit={B['dtheta_unit'][p]:+.4e})")

    fig_tilt(T_lf, T_hr)
    fig_bias(B_lf, B_hr)

    summary = {
        "kmax_confirm": {"LF": [T_lf["kfin_min"], T_lf["kfin_max"]],
                         "HR": [T_hr["kfin_min"], T_hr["kfin_max"]]},
        "sigma_frac_assumed": SIGMA_FRAC,
        "tilt": {"LF": T_lf["stats"], "HR": T_hr["stats"]},
        "fisher_bias": {"LF": B_lf, "HR": B_hr},
    }
    with open(f"{OUT}/diag_tilt_bias_lf_hr.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {OUT}/diag_tilt_bias_lf_hr.json  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
