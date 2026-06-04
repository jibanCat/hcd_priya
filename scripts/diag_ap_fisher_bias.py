"""A_p (and 9-param) Fisher-bias of the DEPLOYED PRODUCTION emulator + the low-k
coherent-vs-CV split + the A_p–τ₀ amplitude/mean-flux degeneracy (the gate).

Unlike scripts/diag_tilt_bias_lf_hr.py (which rebuilds its own recipe model), this
trains the ACTUAL production model via train.train_fold (frozen θ-blind baseline,
the redesign two-phase) on the LF fold-0 LOSO split, then interrogates THAT model.
It is the gate for the residual-head tuning: A_p Fisher-bias < 0.2σ.

Pieces (LF fold-0):
  1. TRAIN/LOAD the production train_fold model (honors term_w / k_weight / early
     stop). A checkpoint is cached for reuse.
  2. FISHER-BIAS over the 9 unit-cube params: J = ∂logP̂/∂θ (jacfwd through the
     deployed model; the baseline is θ-blind so ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ),
     δ = the deployed log-error averaged over held-out val rows at the fiducial
     z-slice, C = a DESI-DR1-like diagonal per-mode covariance. Report ALL 9 in σ
     units PLUS the absolute unit-cube shift vs the prior width and the Fisher cond
     (trap #22: σ-units are only physical when |dθ_unit| ≪ 1 and cond is moderate).
  3. A_p–τ₀ Fisher correlation: extend J to a 10th column ∂logP̂/∂τ₀ and report the
     (A_p, τ₀) entry of the normalized Fisher inverse (the amplitude–mean-flux
     degeneracy). Confirms ∂P/∂A_p and ∂P/∂τ₀ are both well-resolved.
  4. LOW-k coherent-vs-CV split: at the fiducial z-slice, decompose the per-(class,k)
     deployed log-error into the COHERENT bias (mean over val sims) vs the SCATTER
     (std over val sims), compared to the per-(z,k) cosmic-variance floor
     (figures/analysis/04_emulator/diag_lfhf_tilt_and_cv.json). Reports what is a
     fixable coherent bias vs the irreducible CV/sampling scatter.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_ap_fisher_bias.py [--reuse]
Writes figures + a JSON summary to figures/analysis/04_emulator/.
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
    load_cache, make_splits, make_batch, cell_id, safe_log,
    edge_emphasis_k_weight,
)
from hcd_analysis.emulator import train as T

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
CKPT = "/home/mfho/hcd_priya/checkpoints/ap_fisher_fold0"
CV_JSON = "/home/mfho/hcd_priya/figures/analysis/04_emulator/diag_lfhf_tilt_and_cv.json"
CLS = ("clean", "LLS", "subDLA", "DLA")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
# DESI-DR1-like diagonal per-mode covariance (same assumption as diag_tilt_bias_lf_hr).
SIGMA_FRAC = {"clean": 0.03, "LLS": 0.15, "subDLA": 0.15, "DLA": 0.15}
# The Lyα amplitude pivot (Δ²_* at k_* ≈ 0.009 s/km) — A_p maps onto a coherent
# residual at/below this; "low-k" band for the coherent/CV split.
KPIVOT = 0.009
LOWK_BAND = (0.0, 0.005)        # the lowest-k band (where CV is largest, A_p lives)


# ---------------------------------------------------------------------------- #
# Training (the production model)
# ---------------------------------------------------------------------------- #
def train_or_load(d, *, fold=0, n_basis=24, epochs=180, lr=1e-3, batch=512, seed=0,
                  patience=25, term_w=None, k_weight=None, reuse=False,
                  early_stop_metric="auto", weight_decay=1e-4):
    n_k = d["P_tier_p"].shape[1]
    tr, va, ho = make_splits(d, fold, n_folds=8, holdout_frac=0.15)
    if reuse and Path(CKPT + ".eqx").exists():
        model, meta, norm = T.load_checkpoint(CKPT)
        print(f"[reuse] {CKPT} (n_basis={meta['arch_cfg']['n_basis']})")
        return model, norm, (tr, va, ho), n_k
    print(f"[train] fold={fold} epochs={epochs} n_basis={n_basis} term_w={term_w} "
          f"k_weight={'edge' if k_weight is not None else None} "
          f"es={early_stop_metric} wd={weight_decay} "
          f"train={len(tr)} val={len(va)} holdout={len(ho)}")
    model, norm, history = T.train_fold(
        d, tr, va, n_basis=n_basis, lr=lr, epochs=epochs, batch_size=batch,
        seed=seed, key=jax.random.PRNGKey(seed), patience=patience, n_k=n_k,
        staged=False, term_w=term_w, k_weight=k_weight,
        early_stop_metric=early_stop_metric, weight_decay=weight_decay)
    arch = {"in_dim": 10, "n_k": n_k, "n_basis": n_basis}
    T.save_checkpoint(CKPT, model, arch, norm, seed=seed,
                      kfkms=d["kfkms"], cache_path=CACHE)
    n_ep = len(history["train_loss"])
    argmin = int(np.argmin(history["val_resid_loss"]))
    print(f"[train] {n_ep} epochs (val_resid min @ ep {argmin+1}); -> {CKPT}")
    return model, norm, (tr, va, ho), n_k


# ---------------------------------------------------------------------------- #
# Fisher-bias (deployed production model)
# ---------------------------------------------------------------------------- #
def fisher_bias(model, d, va, norm, *, z_fid=3.0, fiducial=None):
    """δθ = (JᵀC⁻¹J)⁻¹JᵀC⁻¹δ in σ_Fisher units, over the 9 params PLUS a 10th τ₀
    column (for the A_p–τ₀ correlation). Returns the per-param bias + the absolute
    unit-cube shift + the (A_p,τ₀) Fisher correlation + cond."""
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]                                    # (4,K)
    n_k = sc.shape[1]
    if fiducial is None:
        fiducial = np.full(9, 0.5)

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
    kvalid = (mask_va[vz].mean(0) > 0.5)                   # (4,K)
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"][va][vz]),
                               d["kfkms"][va][vz], np.nan), 0)

    # J over the 9 params AND τ₀ (10 columns). The model's residual head is
    # r̂(x, τ₀); x = [params_unit(9), z_unit(1)]. ∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ.
    z_j = jnp.asarray(z_unit_fid)

    def rhat(theta9_tau):
        theta9 = theta9_tau[:9]
        tau = theta9_tau[9]
        x = jnp.concatenate([theta9, z_j[None]])
        return model(x, tau)["P_filt_resid"]               # (4, n_k)

    p0 = jnp.asarray(np.concatenate([fiducial, [tau0_fid]]), dtype=jnp.float64)
    Jr = np.asarray(jax.jacfwd(rhat)(p0))                  # (4, n_k, 10)
    J_logP = Jr * sc[:, :, None]                           # ∂logP̂/∂(θ,τ₀)

    # deployed log error at the fiducial z-slice (the coherent k-tilt)
    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    resid = np.asarray(pred["P_filt_resid"])
    logP_hat = base + sc * resid
    logP_true = safe_log(d["P_filt"][va][vz])
    m_slice = np.isfinite(logP_true)
    with np.errstate(invalid="ignore"):
        delta = np.array([[np.nanmean(np.where(m_slice[:, ci, j], (logP_hat - logP_true)[:, ci, j], np.nan))
                           for j in range(n_k)] for ci in range(4)])  # (4,K)

    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(n_k):
            if not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j])) \
               or not np.isfinite(delta[ci, j]):
                continue
            rows_J.append(J_logP[ci, j]); rows_d.append(delta[ci, j])
            rows_C.append(SIGMA_FRAC[nm] ** 2)
    J = np.array(rows_J)                                   # (M, 10)
    dv = np.array(rows_d)                                  # (M,)
    Cinv = 1.0 / np.array(rows_C)

    # 9-param Fisher (the gate). τ₀ column kept separate for the correlation block.
    J9 = J[:, :9]
    F = (J9.T * Cinv) @ J9                                  # (9,9)
    ridge = 1e-12 * np.trace(F) / 9.0
    Finv = np.linalg.inv(F + ridge * np.eye(9))
    sigma = np.sqrt(np.diag(Finv))
    dtheta = Finv @ ((J9.T * Cinv) @ dv)
    bias_sigma = dtheta / sigma
    prior_width = np.ones(9)                                # unit-cube prior width

    # A_p–τ₀ correlation: 10-param Fisher (9 + τ₀), normalized inverse-covariance corr.
    F10 = (J.T * Cinv) @ J                                  # (10,10)
    ridge10 = 1e-12 * np.trace(F10) / 10.0
    F10inv = np.linalg.inv(F10 + ridge10 * np.eye(10))
    s10 = np.sqrt(np.diag(F10inv))
    corr10 = F10inv / np.outer(s10, s10)
    iAp, itau = 1, 9
    ap_tau_corr = float(corr10[iAp, itau])
    # are ∂P/∂A_p and ∂P/∂τ₀ both well-resolved? (their J-column norms, weighted)
    jnorm = np.sqrt((J ** 2 * Cinv[:, None]).sum(0))        # (10,) C^{-1/2}-weighted
    return dict(
        z_fid=z_fid, tau0_fid=tau0_fid, n_modes=int(len(dv)),
        fisher_cond=float(np.linalg.cond(F + ridge * np.eye(9))),
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)},
        dtheta_unit={PARAMS[i]: float(dtheta[i]) for i in range(9)},
        bias_in_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        dtheta_over_prior={PARAMS[i]: float(abs(dtheta[i]) / prior_width[i]) for i in range(9)},
        ap_tau0_corr=ap_tau_corr,
        J_colnorm_Cinv={**{PARAMS[i]: float(jnorm[i]) for i in range(9)},
                        "tau0": float(jnorm[9])},
        fisher_cond10=float(np.linalg.cond(F10 + ridge10 * np.eye(10))),
    )


# ---------------------------------------------------------------------------- #
# Low-k coherent-vs-CV split
# ---------------------------------------------------------------------------- #
def coherent_vs_cv(model, d, va, norm, *, z_fid=3.0):
    """Per-(class,k) at the fiducial z: COHERENT bias = mean over val sims of the
    deployed log-error; SCATTER = std over val sims. Compare to the CV floor."""
    pf = norm["P_filt"]
    sc = pf["sig_cosmo"]; n_k = sc.shape[1]
    vz = np.isclose(d["z_grid"][va], z_fid, atol=0.05)
    if not vz.any():
        vz = np.ones(len(va), bool)
    x_va = jnp.asarray(d["x"][va][vz]); tau_va = jnp.asarray(d["tau0"][va][vz])
    pred = jax.vmap(model)(x_va, tau_va)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    resid = np.asarray(pred["P_filt_resid"])
    logP_hat = base + sc * resid
    logP_true = safe_log(d["P_filt"][va][vz])
    frac = np.exp(logP_hat - logP_true) - 1.0              # (m,4,K) fractional err
    m = np.isfinite(logP_true)
    frac = np.where(m, frac, np.nan)
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"][va][vz]),
                               d["kfkms"][va][vz], np.nan), 0)   # (K,)

    with np.errstate(invalid="ignore"):
        coherent = np.nanmean(frac, axis=0)                # (4,K) mean over sims
        scatter = np.nanstd(frac, axis=0)                  # (4,K) std over sims
        nsim = np.sum(np.isfinite(frac), axis=0)           # (4,K) # val sims

    # CV floor per k-band from the cosmic-variance diagnostic (clean P1D proxy).
    cv = None
    if Path(CV_JSON).exists():
        cvj = json.load(open(CV_JSON))["measurement_2_cosmic_variance"]["cv_frac_per_kbin"]
        cv = {}
        for band, v in cvj.items():
            lo, hi = (float(x) for x in band.split("-"))
            cv[(lo, hi)] = float(v["median_pct"]) / 100.0  # fractional

    def band_stat(lo, hi):
        kb = (kf >= lo) & (kf < hi) & np.isfinite(kf)
        out = {}
        for ci, nm in enumerate(CLS):
            c = coherent[ci, kb]; s = scatter[ci, kb]
            out[nm] = dict(coherent_rms=float(np.sqrt(np.nanmean(c ** 2))),
                           coherent_mean=float(np.nanmean(c)),
                           scatter_rms=float(np.sqrt(np.nanmean(s ** 2))),
                           n_kbins=int(np.isfinite(c).sum()))
        # CV floor for this band (nearest)
        cvfloor = np.nan
        if cv is not None:
            best = min(cv, key=lambda b: abs((b[0]+b[1])/2 - (lo+hi)/2))
            cvfloor = cv[best]
        out["cv_floor_frac"] = float(cvfloor)
        return out

    bands = {"lowk_0-0.005": band_stat(0.0, 0.005),
             "lowk_0.005-0.01": band_stat(0.005, 0.01),
             "mid_0.01-0.03": band_stat(0.01, 0.03),
             "high_0.03-0.07": band_stat(0.03, 0.07)}
    return dict(kf=kf, coherent=coherent, scatter=scatter, nsim=nsim, bands=bands,
                cv=cv)


# ---------------------------------------------------------------------------- #
# Figure: per-k residual (coherent + scatter) per class + CV floor
# ---------------------------------------------------------------------------- #
def fig_perk_residual(split, tag, path):
    kf = split["kf"]; coh = split["coherent"]; sca = split["scatter"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for ci, nm in enumerate(CLS):
        g = np.isfinite(kf) & np.isfinite(coh[ci])
        ax[0].semilogx(kf[g], coh[ci][g] * 100, lw=1.5, label=nm)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].axvline(KPIVOT, color="grey", ls=":", lw=1, label=f"A_p pivot k*≈{KPIVOT}")
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("COHERENT bias ⟨P̂/P−1⟩ [%]")
    ax[0].set_title(f"{tag}: coherent (mean over val sims) per-k bias")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for ci, nm in enumerate(CLS):
        g = np.isfinite(kf) & np.isfinite(sca[ci])
        ax[1].loglog(kf[g], sca[ci][g] * 100, lw=1.5, label=nm)
    # CV floor overlay
    if split["cv"] is not None:
        for (lo, hi), v in split["cv"].items():
            ax[1].hlines(v * 100, lo, hi, color="k", ls="--", lw=1.2,
                         label="CV floor" if (lo, hi) == list(split["cv"])[0] else None)
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("SCATTER std(P̂/P−1) over sims [%]")
    ax[1].set_title(f"{tag}: scatter vs the cosmic-variance floor")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--n-basis", type=int, default=24)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--p-resid-w", type=float, default=1.0,
                    help="term_w['p_resid'] up-weight (others stay 1)")
    ap.add_argument("--edge-gain", type=float, default=0.0,
                    help="k-weight edge emphasis gain (0 = uniform)")
    ap.add_argument("--lowk-extra", type=float, default=0.0,
                    help="k-weight extra low-k ramp")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--early-stop", default="auto", choices=["auto", "resid", "joint"])
    ap.add_argument("--tag", default="prod")
    args = ap.parse_args()

    d = load_cache(CACHE)
    term_w = None
    if args.p_resid_w != 1.0:
        term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0,
                  "p_resid": args.p_resid_w, "delta": 1.0}
    k_weight = None
    if args.edge_gain != 0.0 or args.lowk_extra != 0.0:
        k_weight = edge_emphasis_k_weight(d["kfkms"][0], edge_gain=args.edge_gain,
                                          lowk_extra=args.lowk_extra)
        print(f"  k_weight edge-emphasis: min={k_weight.min():.3f} "
              f"max={k_weight.max():.3f} mean={k_weight.mean():.3f}")

    model, norm, (tr, va, ho), n_k = train_or_load(
        d, n_basis=args.n_basis, epochs=args.epochs, patience=args.patience,
        term_w=term_w, k_weight=k_weight, reuse=args.reuse,
        weight_decay=args.weight_decay, early_stop_metric=args.early_stop)

    print("\n=== FISHER-BIAS (deployed production model, fiducial=cube centre, z=3) ===")
    B = fisher_bias(model, d, va, norm, z_fid=3.0)
    print(f"  {B['n_modes']} modes, Fisher cond(9)={B['fisher_cond']:.2e}, "
          f"cond(10)={B['fisher_cond10']:.2e}")
    print(f"  A_p–τ₀ Fisher correlation = {B['ap_tau0_corr']:+.3f}")
    print(f"  {'param':11} {'bias/σ':>9} {'σ_Fisher':>9} {'|dθ|/prior':>11} {'J·Cinv':>9}")
    for p in PARAMS:
        flag = "  <-- A_p" if p == "Ap" else ""
        print(f"  {p:11} {B['bias_in_sigma'][p]:+9.3f} {B['sigma_fisher'][p]:9.4f} "
              f"{B['dtheta_over_prior'][p]:11.4f} {B['J_colnorm_Cinv'][p]:9.3e}{flag}")
    print(f"  {'tau0':11} {'':9} {'':9} {'':11} {B['J_colnorm_Cinv']['tau0']:9.3e}")

    print("\n=== LOW-k COHERENT vs CV split (fiducial z=3) ===")
    split = coherent_vs_cv(model, d, va, norm, z_fid=3.0)
    for band, st in split["bands"].items():
        cvf = st["cv_floor_frac"]
        print(f"  [{band}] CV floor ~{cvf*100:.2f}%")
        for nm in CLS:
            s = st[nm]
            verdict = "COHERENT>CV (fixable)" if s["coherent_rms"] > cvf else "within CV scatter"
            print(f"     {nm:7}: coherent_rms={s['coherent_rms']*100:6.2f}%  "
                  f"⟨coh⟩={s['coherent_mean']*100:+6.2f}%  scatter={s['scatter_rms']*100:6.2f}%  "
                  f"({s['n_kbins']} k)  -> {verdict}")

    fig_perk_residual(split, args.tag, f"{OUT}/ap_lowk_coherent_vs_cv_{args.tag}.png")

    out = {"tag": args.tag, "n_basis": args.n_basis,
           "term_w": term_w, "edge_gain": args.edge_gain, "lowk_extra": args.lowk_extra,
           "weight_decay": args.weight_decay,
           "fisher": B,
           "lowk_split": {b: {nm: split["bands"][b][nm] for nm in CLS} |
                          {"cv_floor_frac": split["bands"][b]["cv_floor_frac"]}
                          for b in split["bands"]}}
    jpath = f"{OUT}/ap_fisher_bias_{args.tag}.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
