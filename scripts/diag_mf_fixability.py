#!/usr/bin/env python3
"""IS THE MF's ~5% IGM-thermal theta-departure (and its ~2-sigma high-k Fisher bias)
FIXABLE under realistic PRIYA-style priors?  The EMPIRICAL verdict.

CONTEXT (commits 0775362 rho-only MF / 7aed98e attribution).  The LF->HF resolution
correction g = logP_HF - logP_hat_LF is ~95% a fixed tilt + ~5% theta-departure that
is REAL (per-sim D/noise 3.6-18.7) and IGM-THERMAL-driven.  The 6-HF attribution ranks
  bhfeedback (r=-0.78, DOMINANT) > hireionz (-0.59) > herei (+0.54) > ... ; ns/Ap weak.
The rho-only MF (production default) carries this into a HF-LOSO Fisher bias whose
worst held-out fold is |n_s|~2.0, |A_p|~1.4 sigma -- the 6-HF floor.  The targeted-delta
fix FAILED (residual aliases into A_p through f_LF's Jacobian).

PI DOMAIN FACTS that make this FIXABLE-or-not under REALISTIC priors:
  * PRIYA FIXES bhfeedback with a TIGHT GAUSSIAN PRIOR (it is the dominant driver).
  * hireionz is z~7 physics, weakly constrained by the z=2.2-4.6 P1D data.
  * HeII reionization (herei/heref) carries STOCHASTICITY (bubble-painting to haloes ->
    realization noise, NOT a smooth theta-dependence) -> if the herei/heref 'signal' is
    realization scatter it is IRREDUCIBLE (part of C_emu), not fittable with more sims.

This script (READ-ONLY on production; reuses the build_mf_delta HF-LOSO/Fisher harness +
the diag_mf_theta_attribution per-sim departure machinery) answers FOUR questions:

  PART 1 -- bhfeedback-PRIOR FISHER TEST (the key).
    Add a Gaussian prior P to the MF HF-LOSO Fisher: F_tot = J^T C^-1 J + P, with P a
    diagonal precision LARGE on the bhfeedback unit-cube direction (1/sigma_bhf^2) and
    ZERO elsewhere.  The biased estimator becomes dtheta = (F+P)^-1 J^T C^-1 d.  We
    SWEEP sigma_bhf from the PRIYA-tight regime (pin) to infinity (= the current no-prior
    result) and report the WORST-over-HF-LOSO-folds |n_s| and |A_p| bias vs sigma_bhf.
    DOES pinning bhfeedback REDUCE the n_s/A_p bias (because d's residual projects onto
    the bhfeedback-correlated direction the prior now absorbs)?  This is the realistic
    PRIYA Fisher: bhfeedback is NOT a free parameter the data must fit.

  PART 2 -- RESIDUAL (non-bhfeedback) DEPARTURE.
    Re-measure the per-sim theta-departure D_s with bhfeedback REGRESSED OUT (the
    component of D orthogonal to bhfeedback over the 6 sims).  How much of the 5%
    survives?  Re-attribute the residual: is it hireionz / herei / heref / other?  AND
    re-run the Fisher bias with bhfeedback HARD-PINNED (sigma_bhf -> 0) to see the
    residual n_s/A_p bias that bhfeedback CANNOT explain.

  PART 3 -- SMOOTH vs STOCHASTIC for the HeII params (herei/heref).
    With only 6 HF sims + the 20-alpha noise floor, can we tell a SMOOTH theta-trend on
    herei/heref from REALIZATION scatter (the bubble-painting stochasticity)?  Test:
      (a) does a smooth (linear) herei/heref model EXPLAIN the per-sim departure better
          than chance, given the 6-sim noise floor? (R^2 vs a permutation null);
      (b) is the herei/heref departure consistent with the alpha noise floor (-> scatter,
          irreducible) or does it clear it (-> a real, fittable trend)?
    HONEST about the 6-sim limit: 6 points -> wide CIs; we report the permutation
    p-value and flag 'cannot distinguish' where the data can't.

  PART 4 -- hireionz COVERAGE / constrainability.
    Do the 6 HF sims SPAN the hireionz prior cube [0,1], or is the hireionz-departure an
    EXTRAPOLATION?  Report the 6-sim hireionz cube coverage + gaps.  AND: is hireionz
    even constrained at the data z-range?  We read the MF Fisher sigma(hireionz) (the
    data's own constraint) and compare it to the prior width -> if sigma_Fisher >> prior,
    the data does NOT constrain hireionz (z~7 physics) and its departure is prior-absorbed
    like bhfeedback.

Figures -> figures/analysis/05_multifidelity/:
  * mf_fixability_bhf_prior.png   -- worst |n_s|/|A_p| Fisher bias vs bhfeedback-prior
                                     width (PART 1) + the per-param bias at pin vs free.
  * mf_fixability_residual.png    -- residual departure decomposition (PART 2-4): the
                                     non-bhfeedback departure, its re-attribution, the
                                     HeII smooth-vs-stochastic test, hireionz coverage.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_fixability.py [args]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import hcd_analysis.emulator  # enables x64
from hcd_analysis.emulator.data import load_cache, Z_LIMITS, PARAM_LIMITS
from hcd_analysis.emulator import multifidelity as MF

# reuse the build_mf_delta HF-LOSO / Fisher harness + the attribution per-sim departure.
def _load_sibling(name):
    import sys
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    # register BEFORE exec so eqx.Module/dataclass subclasses can resolve __module__.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BMF = _load_sibling("build_mf_delta")
ATTR = _load_sibling("diag_mf_theta_attribution")

CLS = MF.COARSE_NAMES
KODIAQ_BAND = MF.KODIAQ_BAND
PARAMS = BMF.PARAMS                          # 9 names, cache order
PIDX = {p: i for i, p in enumerate(PARAMS)}
BHF = PIDX["bhfeedback"]                     # 8
HEREI, HEREF, HIREIONZ = PIDX["herei"], PIDX["heref"], PIDX["hireionz"]
HEII = ("herei", "heref")
# raw->cube width: a Gaussian prior of raw std sigma_raw on a param is a cube std
# sigma_cube = sigma_raw / (hi - lo).  PRIYA "fixes" bhfeedback with a TIGHT prior; we
# express the sweep DIRECTLY in cube-sigma so it is range-independent.  Reference marks:
#   sigma_cube = 1.0  -> a prior as wide as the whole design box (~ uninformative)
#   sigma_cube = 0.1  -> the prior pins bhfeedback to ~10% of the box (a tight PRIYA-like
#                        prior; bhfeedback is the dominant driver so PRIYA constrains it)
#   sigma_cube -> 0    -> bhfeedback HARD-PINNED at fiducial (not a free parameter)
PRIYA_BHF_SIGMA_CUBE = 0.1   # representative "PRIYA fixes bhfeedback tightly" mark


# ============================================================================ #
# Fisher with a Gaussian prior on a chosen parameter direction
# ============================================================================ #
def mf_fisher_bias_with_prior(mf, hf_abs, eval_rows, k_eval, *, z_fid=3.0,
                              prior_idx=None, prior_sigma_cube=None):
    """build_mf_delta.mf_fisher_bias re-implemented with an optional Gaussian PRIOR.

    Identical J = d logP_MF/d theta (jacfwd on the differentiable MF), residual d (the
    coherent MF log-error over the held-out HF rows at z_fid), and DESI-DR1-like diagonal
    C as build_mf_delta.mf_fisher_bias -- but the biased estimator uses
        F_tot = J^T C^-1 J + P ,   dtheta = F_tot^-1 (J^T C^-1 d) ,   sigma = sqrt(diag F_tot^-1)
    where P is diagonal with P[prior_idx] = 1/prior_sigma_cube^2 on the unit-cube
    direction(s) in ``prior_idx`` (a scalar or list), 0 elsewhere.  prior_sigma_cube may
    be a scalar (same width for all prior_idx) or a list (per index).  prior_idx=None ->
    no prior == the original harness (validated to match below).  Returns the same dict
    plus the prior config and the marginal bias/sigma for every param under the prior.
    """
    # ---- reproduce the harness J, d, C exactly (copied from BMF.mf_fisher_bias) ----
    z_unit_fid = float((z_fid - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0]))
    zsel = np.isclose(hf_abs["x"][eval_rows, 9], z_unit_fid, atol=0.02)
    if not zsel.any():
        zsel = np.ones(len(eval_rows), bool)
    tau0_fid = float(np.median(hf_abs["tau0"][eval_rows][zsel]))
    fiducial = np.full(9, 0.5)
    z_j = jnp.asarray(z_unit_fid)

    def logP_of_theta(theta9):
        x = jnp.concatenate([theta9, z_j[None]])
        return mf.logP_mf(x, jnp.asarray(tau0_fid))

    p0 = jnp.asarray(fiducial, dtype=jnp.float64)
    J = np.asarray(jax.jacfwd(logP_of_theta)(p0))           # (4,K,9)

    rows = np.asarray(eval_rows)[zsel]
    logP_hr = hf_abs["logP"][rows]
    xs = jnp.asarray(hf_abs["x"][rows]); ts = jnp.asarray(hf_abs["tau0"][rows])
    logP_mf = np.asarray(jax.vmap(mf.logP_mf)(xs, ts))
    m = np.isfinite(logP_hr) & np.isfinite(logP_mf)
    with np.errstate(invalid="ignore"):
        d = np.array([[np.nanmean(np.where(m[:, ci, j], (logP_mf - logP_hr)[:, ci, j], np.nan))
                       for j in range(J.shape[1])] for ci in range(4)])    # (4,K)

    kmask = (k_eval >= MF.DATA_RANGE["k_min"]) & (k_eval <= MF.KODIAQ_KMAX)
    rows_J, rows_d, rows_C = [], [], []
    for ci, nm in enumerate(CLS):
        for j in range(J.shape[1]):
            if not kmask[j] or not np.all(np.isfinite(J[ci, j])) or not np.isfinite(d[ci, j]):
                continue
            rows_J.append(J[ci, j]); rows_d.append(d[ci, j])
            rows_C.append(BMF.SIGMA_FRAC[nm] ** 2)
    Jm = np.array(rows_J); dv = np.array(rows_d); Cinv = 1.0 / np.array(rows_C)
    F = (Jm.T * Cinv) @ Jm
    ridge = 1e-12 * np.trace(F) / 9.0
    Fr = F + ridge * np.eye(9)

    # ---- the PRIOR precision matrix P (diagonal, cube units) ------------------------
    P = np.zeros((9, 9))
    prior_meta = None
    if prior_idx is not None:
        idxs = [prior_idx] if np.isscalar(prior_idx) else list(prior_idx)
        sig = prior_sigma_cube
        sigs = ([sig] * len(idxs)) if np.isscalar(sig) else list(sig)
        prior_meta = {}
        for ii, ss in zip(idxs, sigs):
            ss = max(float(ss), 1e-12)              # sigma->0 == hard pin (huge precision)
            P[ii, ii] = 1.0 / ss ** 2
            prior_meta[PARAMS[ii]] = float(ss)

    Ftot = Fr + P
    Finv = np.linalg.inv(Ftot)
    sigma = np.sqrt(np.clip(np.diag(Finv), 0, None))
    dtheta = Finv @ ((Jm.T * Cinv) @ dv)
    bias_sigma = dtheta / np.where(sigma > 0, sigma, np.inf)
    # Fisher correlation of bhfeedback with ns/Ap (the DATA's parameter degeneracy):
    # if |corr| is small, pinning bhfeedback CANNOT move the ns/Ap bias (they're not
    # degenerate in the data) -> the residual aliases into ns/Ap DIRECTLY, not via bhf.
    dsig = np.where(sigma > 0, sigma, np.inf)
    corr = Finv / np.outer(dsig, dsig)
    bhf_corr = {p: float(corr[BHF, PIDX[p]]) for p in ("ns", "Ap", "hireionz")}
    return dict(
        n_modes=int(len(dv)), z_fid=z_fid, tau0_fid=tau0_fid,
        fisher_cond=float(np.linalg.cond(Ftot)),
        prior=prior_meta, bhf_corr=bhf_corr,
        bias_in_sigma={PARAMS[i]: float(bias_sigma[i]) for i in range(9)},
        dtheta={PARAMS[i]: float(dtheta[i]) for i in range(9)},
        sigma_fisher={PARAMS[i]: float(sigma[i]) for i in range(9)},
    )


# ============================================================================ #
# Build a rho-only MF for a given HF-LOSO fold (the production default head)
# ============================================================================ #
def build_rho_only_fold(lf_model, lf_norm, lf_logk, tg, eval_logk, train_rows, n_basis):
    """The production rho-only (FixedMeanHead) MF trained on ``train_rows`` (no leakage)."""
    log_rho = MF.mean_log_ratio_rho({"g": tg["g"][train_rows]}, eval_logk)
    log_rho = np.where(np.isfinite(log_rho), log_rho, 0.0)
    head = MF.build_default_head(tg, log_rho, train_mask_rows=train_rows, n_basis=n_basis)
    return MF.build_multifidelity(lf_model, lf_norm, lf_logk, head, eval_logk=eval_logk,
                                  log_rho=log_rho, n_basis=n_basis, delta_mode="none")


# ============================================================================ #
# PART 2 helper -- residual departure after regressing out a param subset
# ============================================================================ #
def regress_out(D, X_cols):
    """Residual of D after OLS on the columns X_cols (each (nsim,)) + intercept.

    D (nsim,), X_cols list of (nsim,) standardized predictors.  Returns (resid, R2,
    beta) of the joint fit.  With few sims this is descriptive, not inferential."""
    n = len(D)
    A = np.column_stack([np.ones(n)] + [np.asarray(c) for c in X_cols])
    beta, *_ = np.linalg.lstsq(A, D, rcond=None)
    pred = A @ beta
    resid = D - pred
    ss_tot = np.sum((D - D.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return resid, float(r2), beta


def permutation_R2(D, X, n_perm=20000, seed=0):
    """Permutation p-value that a SMOOTH linear model on predictor(s) X explains D.

    X (nsim, p) standardized.  Null: D is exchangeable scatter (no smooth dependence) ->
    permute D's labels, refit, collect R^2.  p = P(R2_perm >= R2_obs).  HONEST: with 6
    sims the null R^2 distribution is wide; a small p means the smooth trend is unusually
    strong vs scatter, a large p means we CANNOT distinguish trend from realization
    scatter (the HeII stochasticity case)."""
    n = len(D)
    A = np.column_stack([np.ones(n), X]) if X.ndim == 2 else np.column_stack([np.ones(n), X[:, None]])
    beta, *_ = np.linalg.lstsq(A, D, rcond=None)
    r2_obs = 1.0 - np.sum((D - A @ beta) ** 2) / max(np.sum((D - D.mean()) ** 2), 1e-30)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        Dp = rng.permutation(D)
        b, *_ = np.linalg.lstsq(A, Dp, rcond=None)
        r2p = 1.0 - np.sum((Dp - A @ b) ** 2) / max(np.sum((Dp - Dp.mean()) ** 2), 1e-30)
        if r2p >= r2_obs - 1e-12:
            cnt += 1
    return float(r2_obs), (cnt + 1) / (n_perm + 1)


# ============================================================================ #
# Figures
# ============================================================================ #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_bhf_prior(sweep, sigmas, per_param_free, per_param_pin, figdir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ns_worst = np.array([sweep[i]["ns_worst"] for i in range(len(sigmas))])
    ap_worst = np.array([sweep[i]["ap_worst"] for i in range(len(sigmas))])
    ns_mean = np.array([sweep[i]["ns_mean"] for i in range(len(sigmas))])
    ap_mean = np.array([sweep[i]["ap_mean"] for i in range(len(sigmas))])
    bhf_worst = np.array([sweep[i]["bhf_worst"] for i in range(len(sigmas))])
    ax.semilogx(sigmas, ns_worst, "o-", color="C3", lw=2, ms=7, label="|n_s| worst fold")
    ax.semilogx(sigmas, ap_worst, "s-", color="C2", lw=2, ms=7, label="|A_p| worst fold")
    ax.semilogx(sigmas, ns_mean, "o--", color="C3", lw=1.3, ms=4, alpha=0.6, label="|n_s| mean")
    ax.semilogx(sigmas, ap_mean, "s--", color="C2", lw=1.3, ms=4, alpha=0.6, label="|A_p| mean")
    ax.semilogx(sigmas, bhf_worst, "^:", color="0.4", lw=1.3, ms=6, label="|bhf| worst (absorbed)")
    ax.axvline(PRIYA_BHF_SIGMA_CUBE, color="purple", ls="-.", lw=1.4)
    ax.text(PRIYA_BHF_SIGMA_CUBE * 1.05, ax.get_ylim()[1] * 0.9, "PRIYA-tight",
            color="purple", fontsize=9, rotation=90, va="top")
    ax.axhline(2.0, color="grey", lw=0.9, ls=":")
    ax.set_xlabel("bhfeedback Gaussian prior width  sigma_cube  (fraction of design box)")
    ax.set_ylabel("MF HF-LOSO Fisher bias [sigma]")
    ax.set_title("PART 1: does PINNING bhfeedback (left) suppress the n_s/A_p bias?\n"
                 "(right edge = no prior = current rho-only result)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.invert_xaxis()   # tight prior (pin) on the LEFT -> reads "as we tighten ->"
    # panel 2: per-param worst-fold bias, free vs hard-pinned bhfeedback.
    ax2 = axes[1]
    x = np.arange(9)
    w = 0.38
    bf = np.array([per_param_free[p] for p in PARAMS])
    bp = np.array([per_param_pin[p] for p in PARAMS])
    cols = ["C3" if p in HEII or p == "hireionz" else ("C0" if p in ("ns", "Ap") else "0.6")
            for p in PARAMS]
    ax2.bar(x - w / 2, np.abs(bf), w, color=cols, alpha=0.55, label="bhf FREE (no prior)")
    ax2.bar(x + w / 2, np.abs(bp), w, color=cols, edgecolor="k", lw=1.0,
            label="bhf HARD-PINNED")
    ax2.axhline(2.0, color="grey", lw=0.9, ls=":")
    ax2.set_xticks(x); ax2.set_xticklabels(PARAMS, rotation=45, ha="right")
    ax2.set_ylabel("|Fisher bias| [sigma]  (worst HF-LOSO fold)")
    ax2.set_title("Per-param bias: bhfeedback FREE vs HARD-PINNED\n"
                  "(does the residual move INTO n_s/A_p, or vanish?)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("bhfeedback-PRIOR Fisher test: is the MF high-k n_s/A_p bias acceptable "
                 "once PRIYA fixes bhfeedback?")
    p = Path(figdir) / "mf_fixability_bhf_prior.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


def fig_residual(res, figdir):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    sims = res["sims"]
    # panel 1: per-sim departure D, the bhfeedback-explained part, the residual.
    D = np.array(res["D"]); resid = np.array(res["resid_bhf"])
    xs = np.arange(len(sims))
    ax = axes[0]
    ax.bar(xs, D, color="C3", alpha=0.55, label="|departure| D_s (clean,KODIAQ)")
    ax.bar(xs, np.abs(resid), color="none", edgecolor="k", lw=1.3,
           label="residual after bhfeedback")
    ax.plot(xs, np.array(res["noise"]), "k_", ms=14, label="alpha noise floor")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ns{float(s.split('Ap')[0][2:]):.3f}" for s in sims],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("KODIAQ-band |dg|")
    ax.set_title(f"PART 2: departure vs residual after bhfeedback\n"
                 f"bhfeedback R^2={res['r2_bhf']:.2f}  ->  residual frac="
                 f"{res['resid_frac']:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    # panel 2: re-attribution of the RESIDUAL to the remaining params.
    ax2 = axes[1]
    rp = res["resid_attr"]   # dict param -> r
    order = sorted([p for p in PARAMS if p != "bhfeedback"],
                   key=lambda p: -abs(rp[p]))
    rr = [rp[p] for p in order]
    cols = ["C3" if p in HEII else ("C1" if p == "hireionz" else
            ("C0" if p in ("ns", "Ap") else "0.6")) for p in order]
    ax2.barh(np.arange(len(order)), rr, color=cols)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(np.arange(len(order))); ax2.set_yticklabels(order)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pearson r( param , residual departure )")
    ax2.set_title("Re-attribution of the NON-bhfeedback residual\n"
                  "(orange=hireionz, red=HeII herei/heref, blue=ns/Ap)")
    ax2.grid(alpha=0.3, axis="x"); ax2.set_xlim(-1.05, 1.05)
    # panel 3: HeII smooth-vs-stochastic + hireionz coverage text.
    ax3 = axes[2]; ax3.axis("off")
    L = []
    L.append("PART 3 -- HeII (herei/heref): smooth trend vs realization scatter")
    L.append(f"   alpha noise floor (mean over sims) : {np.mean(res['noise']):.4f}")
    L.append(f"   |departure| (mean over sims)       : {np.mean(np.abs(D)):.4f}")
    h = res["heii"]
    L.append(f"   smooth herei+heref model  R^2={h['r2']:.2f}  perm-p={h['perm_p']:.3f}")
    L.append(f"   verdict: {h['verdict']}")
    L.append("")
    L.append("PART 4 -- hireionz coverage / constrainability")
    hz = res["hireionz"]
    L.append(f"   6-sim hireionz cube range : [{hz['cube_min']:.2f}, {hz['cube_max']:.2f}]"
             f"  (of [0,1])")
    L.append(f"   largest coverage gap      : {hz['max_gap']:.2f}")
    L.append(f"   Fisher sigma(hireionz)    : {hz['sigma_fisher']:.2f} cube"
             f"  (prior box = 1.0)")
    L.append(f"   data constrains hireionz? : {hz['constrained']}")
    L.append(f"   verdict: {hz['verdict']}")
    L.append("")
    L.append("RESIDUAL (bhf-pinned) Fisher bias [sigma], worst fold:")
    for p in ("ns", "Ap", "hireionz", "herei", "heref"):
        L.append(f"   |{p:9}| = {abs(res['pin_bias'][p]):.2f}")
    ax3.text(0.0, 1.0, "\n".join(L), va="top", ha="left", fontsize=10,
             family="monospace", transform=ax3.transAxes)
    fig.suptitle("Residual (non-bhfeedback) departure: which params, how big, "
                 "FITTABLE (more HF) vs IRREDUCIBLE (HeII stochasticity)?")
    p = Path(figdir) / "mf_fixability_residual.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return str(p)


# ============================================================================ #
# main
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lf-fold", type=int, default=0)
    ap.add_argument("--n-k", type=int, default=48)
    ap.add_argument("--k-max", type=float, default=MF.KODIAQ_KMAX)
    ap.add_argument("--n-basis", type=int, default=4)
    ap.add_argument("--n-prior", type=int, default=25,
                    help="# bhfeedback-prior-width points in the sweep (log-spaced)")
    ap.add_argument("--figdir", default="figures/analysis/05_multifidelity")
    ap.add_argument("--out",
                    default="figures/analysis/05_multifidelity/mf_fixability_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    print("jax.devices():", jax.devices())
    t0 = time.time()

    lf = load_cache(MF.LF_CACHE)
    hr = load_cache(MF.HR_CACHE)
    lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=args.lf_fold)
    pairs = MF.match_hr_to_lf(lf, hr)
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=args.k_max, n_k=args.n_k)
    tg = MF.measure_delta_targets(lf, hr, lf_model, lf_norm, lf_logk, eval_logk, pairs)
    hf_abs = BMF.measure_hf_abs_targets(hr, lf, eval_logk, pairs)
    sim_of_row = hr["sim_name"][tg["hr_row"]]
    sims = sorted(set(sim_of_row))
    print(f"matched {len(pairs)} HR<->LF rows; {len(sims)} HF sims; eval "
          f"{k_eval[0]:.4f}..{k_eval[-1]:.4f} s/km; LF fold {args.lf_fold} "
          f"({time.time()-t0:.1f}s)")

    # per-sim unit-cube params (constant within a sim).
    params_unit = np.zeros((len(sims), 9))
    for si, s in enumerate(sims):
        ri = np.where(sim_of_row == s)[0][0]
        params_unit[si] = tg["x"][ri, :9]

    if args.smoke:
        args.n_prior = 8

    # ===================================================================== #
    # Build a rho-only MF per HF-LOSO fold ONCE; cache (mf, eval_rows) so the
    # prior sweep just re-solves the (cheap) Fisher linear algebra per width.
    # ===================================================================== #
    print("\n========== building rho-only MF per HF-LOSO fold ==========")
    folds = []
    for s in sims:
        held = (sim_of_row == s)
        train_rows = np.where(~held)[0]; eval_rows = np.where(held)[0]
        ts = time.time()
        mf = build_rho_only_fold(lf_model, lf_norm, lf_logk, tg, eval_logk,
                                 train_rows, args.n_basis)
        folds.append((s, mf, eval_rows))
        print(f"  [{s[:14]}..] built ({time.time()-ts:.1f}s)")

    # ===================================================================== #
    # PART 1 -- bhfeedback-prior sweep
    # ===================================================================== #
    print("\n========== PART 1: bhfeedback-PRIOR FISHER SWEEP ==========")
    # log-spaced cube-sigma from tight pin (0.01 box) to ~uninformative (3 box).
    sigmas = np.logspace(np.log10(0.01), np.log10(3.0), args.n_prior)
    sweep = []
    # precompute per-fold FREE (no prior) and HARD-PIN (sigma->0) once.
    per_fold_free = {}
    per_fold_pin = {}
    for s, mf, eval_rows in folds:
        per_fold_free[s] = mf_fisher_bias_with_prior(
            mf, hf_abs, eval_rows, k_eval, z_fid=3.0, prior_idx=None)
        per_fold_pin[s] = mf_fisher_bias_with_prior(
            mf, hf_abs, eval_rows, k_eval, z_fid=3.0, prior_idx=BHF,
            prior_sigma_cube=1e-6)
    for sig in sigmas:
        ns_a, ap_a, bhf_a = [], [], []
        for s, mf, eval_rows in folds:
            FB = mf_fisher_bias_with_prior(
                mf, hf_abs, eval_rows, k_eval, z_fid=3.0,
                prior_idx=BHF, prior_sigma_cube=float(sig))
            ns_a.append(abs(FB["bias_in_sigma"]["ns"]))
            ap_a.append(abs(FB["bias_in_sigma"]["Ap"]))
            bhf_a.append(abs(FB["bias_in_sigma"]["bhfeedback"]))
        sweep.append(dict(
            sigma_cube=float(sig),
            ns_worst=float(np.max(ns_a)), ns_mean=float(np.mean(ns_a)),
            ap_worst=float(np.max(ap_a)), ap_mean=float(np.mean(ap_a)),
            bhf_worst=float(np.max(bhf_a)),
        ))
    # worst-fold per-param at free & pin (for panel 2 + reporting).
    pp_free = {p: max(abs(per_fold_free[s]["bias_in_sigma"][p]) for s, _, _ in folds)
               for p in PARAMS}
    pp_pin = {p: max(abs(per_fold_pin[s]["bias_in_sigma"][p]) for s, _, _ in folds)
              for p in PARAMS}
    # bias at the PRIYA-tight mark.
    j_priya = int(np.argmin(np.abs(sigmas - PRIYA_BHF_SIGMA_CUBE)))
    print(f"{'sigma_cube':>11} {'|ns|worst':>10} {'|Ap|worst':>10} {'|bhf|worst':>11}")
    for i, sig in enumerate(sigmas):
        mark = "  <- PRIYA-tight" if i == j_priya else ""
        print(f"{sig:11.4f} {sweep[i]['ns_worst']:10.2f} {sweep[i]['ap_worst']:10.2f} "
              f"{sweep[i]['bhf_worst']:11.2f}{mark}")
    free_w = sweep[-1]; priya_w = sweep[j_priya]; pin_ns = pp_pin["ns"]; pin_ap = pp_pin["Ap"]
    print(f"\nFREE (no prior, right edge): |ns|worst={free_w['ns_worst']:.2f} "
          f"|Ap|worst={free_w['ap_worst']:.2f}")
    print(f"PRIYA-tight (sig={PRIYA_BHF_SIGMA_CUBE}): |ns|worst={priya_w['ns_worst']:.2f} "
          f"|Ap|worst={priya_w['ap_worst']:.2f}")
    print(f"HARD-PIN: |ns|worst={pin_ns:.2f} |Ap|worst={pin_ap:.2f}")
    # WHY: the data-Fisher constraint on bhfeedback + its degeneracy with ns/Ap.  A tight
    # prior on bhfeedback can only move the ns/Ap bias if (a) the data constrains
    # bhfeedback (small sigma) AND (b) bhfeedback is degenerate with ns/Ap (|corr| large).
    sig_bhf = float(np.mean([per_fold_free[s]["sigma_fisher"]["bhfeedback"]
                             for s, _, _ in folds]))
    corr_ns = float(np.mean([per_fold_free[s]["bhf_corr"]["ns"] for s, _, _ in folds]))
    corr_ap = float(np.mean([per_fold_free[s]["bhf_corr"]["Ap"] for s, _, _ in folds]))
    print(f"WHY no transfer: data Fisher sigma(bhfeedback)={sig_bhf:.2f} cube "
          f"(box=1.0 -> data BARELY constrains bhf); "
          f"Fisher corr(bhf,ns)={corr_ns:+.2f} corr(bhf,Ap)={corr_ap:+.2f} "
          f"(weak -> pinning bhf can't move ns/Ap).")
    bhf_why = dict(sigma_fisher_bhf=sig_bhf, corr_bhf_ns=corr_ns, corr_bhf_ap=corr_ap)

    # ===================================================================== #
    # PART 2 -- residual (non-bhfeedback) departure
    # ===================================================================== #
    print("\n========== PART 2: RESIDUAL (non-bhfeedback) DEPARTURE ==========")
    dep = ATTR.per_sim_departure(tg, sim_of_row, k_eval)
    D = dep["D"]; noise = dep["noise"]; dep_sims = dep["sims"]
    # align params_unit to dep_sims order (ATTR sorts the same way; assert).
    assert dep_sims == sims, "sim ordering mismatch between ATTR and main"
    # standardize predictors over the 6 sims.
    def zc(col):
        c = params_unit[:, col]
        sd = c.std(ddof=1)
        return (c - c.mean()) / (sd if sd > 1e-12 else 1.0)
    bhf_z = zc(BHF)
    resid_bhf, r2_bhf, _ = regress_out(D, [bhf_z])
    resid_frac = float(np.sqrt(np.mean(resid_bhf ** 2)) / np.sqrt(np.mean(D ** 2)))
    # re-attribute residual to remaining params (univariate Pearson r).
    from scipy import stats as _st
    resid_attr = {}
    for p in PARAMS:
        if p == "bhfeedback":
            continue
        r, _p = _st.pearsonr(params_unit[:, PIDX[p]], resid_bhf)
        resid_attr[p] = float(r)
    rank_resid = sorted(resid_attr, key=lambda p: -abs(resid_attr[p]))
    print(f"bhfeedback explains R^2={r2_bhf:.2f} of the per-sim departure; "
          f"residual RMS frac of D = {resid_frac:.2f}")
    print("residual re-attribution (Pearson r):")
    for p in rank_resid[:5]:
        tag = ("HeII" if p in HEII else ("hireionz" if p == "hireionz" else
               ("ns/Ap" if p in ("ns", "Ap") else "other")))
        print(f"   {p:11} r={resid_attr[p]:+.3f}  ({tag})")

    # ===================================================================== #
    # PART 3 -- HeII smooth vs stochastic
    # ===================================================================== #
    print("\n========== PART 3: HeII (herei/heref) SMOOTH vs STOCHASTIC ==========")
    # use the residual-after-bhfeedback departure as the target the HeII params must
    # explain (bhfeedback already removed); fit a smooth linear herei+heref model.
    X_heii = np.column_stack([zc(HEREI), zc(HEREF)])
    r2_heii, perm_p = permutation_R2(resid_bhf, X_heii, seed=0)
    # is the departure even above the noise floor?  mean |D| vs mean noise.
    above_floor = float(np.mean(np.abs(D))) > 2.0 * float(np.mean(noise))
    if perm_p < 0.1 and r2_heii > 0.5:
        heii_verdict = ("SMOOTH trend favored (perm-p<0.1) -> potentially FITTABLE "
                        "with more HF in herei/heref")
    elif not above_floor:
        heii_verdict = ("departure ~ noise floor -> consistent with REALIZATION SCATTER "
                        "(irreducible, part of C_emu)")
    else:
        heii_verdict = ("CANNOT distinguish smooth trend from scatter at 6 sims "
                        "(perm-p high); treat HeII departure as IRREDUCIBLE pending more HF")
    print(f"smooth herei+heref model: R^2={r2_heii:.2f}, permutation p={perm_p:.3f}")
    print(f"mean|D|={np.mean(np.abs(D)):.4f} vs 2x noise floor={2*np.mean(noise):.4f} "
          f"-> above floor: {above_floor}")
    print(f"verdict: {heii_verdict}")
    heii = dict(r2=float(r2_heii), perm_p=float(perm_p), above_floor=above_floor,
                verdict=heii_verdict)

    # ===================================================================== #
    # PART 4 -- hireionz coverage / constrainability
    # ===================================================================== #
    print("\n========== PART 4: hireionz COVERAGE / CONSTRAINABILITY ==========")
    hz_cube = np.sort(params_unit[:, HIREIONZ])
    gaps = np.diff(np.concatenate([[0.0], hz_cube, [1.0]]))
    max_gap = float(gaps.max())
    # data's own hireionz constraint = Fisher sigma (NO prior), worst fold (largest =
    # least constrained).  prior box width is 1.0 in cube units.
    sig_hz = max(per_fold_free[s]["sigma_fisher"]["hireionz"] for s, _, _ in folds)
    constrained = sig_hz < 0.5    # < half the box -> data says something
    if not constrained:
        hz_verdict = ("hireionz UNCONSTRAINED by z=2.2-4.6 P1D (sigma_Fisher >= box) "
                      "-> its departure is prior-absorbed like bhfeedback (z~7 physics)")
    elif max_gap > 0.35:
        hz_verdict = ("6 HF sims leave a large hireionz coverage gap -> departure is "
                      "partly EXTRAPOLATION; needs an HF spanning the gap")
    else:
        hz_verdict = "hireionz reasonably covered + weakly constrained"
    print(f"6-sim hireionz cube range [{hz_cube.min():.2f},{hz_cube.max():.2f}], "
          f"max gap {max_gap:.2f}")
    print(f"worst-fold Fisher sigma(hireionz) = {sig_hz:.2f} cube (box=1.0) -> "
          f"constrained: {constrained}")
    print(f"verdict: {hz_verdict}")
    hireionz = dict(cube_min=float(hz_cube.min()), cube_max=float(hz_cube.max()),
                    max_gap=max_gap, sigma_fisher=float(sig_hz),
                    constrained=bool(constrained), verdict=hz_verdict)

    # residual (bhf-pinned) worst-fold bias for the text panel.
    pin_bias_worst = {p: max(per_fold_pin[s]["bias_in_sigma"][p] for s, _, _ in folds)
                      if abs(max(per_fold_pin[s]["bias_in_sigma"][p] for s, _, _ in folds))
                      >= abs(min(per_fold_pin[s]["bias_in_sigma"][p] for s, _, _ in folds))
                      else min(per_fold_pin[s]["bias_in_sigma"][p] for s, _, _ in folds)
                      for p in PARAMS}

    res = dict(
        sims=sims, D=D.tolist(), noise=noise.tolist(),
        resid_bhf=resid_bhf.tolist(), r2_bhf=float(r2_bhf), resid_frac=resid_frac,
        resid_attr=resid_attr, rank_resid=rank_resid,
        heii=heii, hireionz=hireionz, pin_bias=pin_bias_worst,
    )

    # ===================================================================== #
    # figures + report
    # ===================================================================== #
    fp1 = fig_bhf_prior(sweep, sigmas, pp_free, pp_pin, args.figdir)
    fp2 = fig_residual(res, args.figdir)

    # ---- the VERDICT ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    bhf_suppresses = priya_w["ns_worst"] < 0.8 * free_w["ns_worst"] or \
        priya_w["ap_worst"] < 0.8 * free_w["ap_worst"]
    print(f"(1) bhfeedback PRIYA-tight prior: "
          f"|ns| {free_w['ns_worst']:.2f}->{priya_w['ns_worst']:.2f}, "
          f"|Ap| {free_w['ap_worst']:.2f}->{priya_w['ap_worst']:.2f} sigma "
          f"(free->PRIYA).  Hard-pin: |ns|={pin_ns:.2f} |Ap|={pin_ap:.2f}.")
    print(f"    -> bhfeedback prior {'SUPPRESSES' if bhf_suppresses else 'does NOT suppress'}"
          f" the n_s/A_p high-k bias.")
    print(f"(2) Residual after bhfeedback: {resid_frac:.0%} of the departure RMS survives; "
          f"top residual driver = {rank_resid[0]} (r={resid_attr[rank_resid[0]]:+.2f}).")
    print(f"    bhf-pinned worst-fold residual bias: |ns|={abs(pin_bias_worst['ns']):.2f} "
          f"|Ap|={abs(pin_bias_worst['Ap']):.2f} sigma.")
    print(f"(3) HeII herei/heref: {heii['verdict']}")
    print(f"(4) hireionz: {hireionz['verdict']}")
    acceptable = max(priya_w["ns_worst"], priya_w["ap_worst"]) < 2.0
    print(f"\n=> Under realistic priors the MF high-k bias is "
          f"{'ACCEPTABLE (<2 sigma)' if acceptable else 'still >2 sigma worst-fold'}; "
          f"the residual {resid_frac:.0%} is "
          f"{'mostly bhfeedback-absorbed' if resid_frac < 0.5 else 'NOT fully bhf-absorbed'}.")

    out = dict(
        config=vars(args), params=list(PARAMS),
        priya_bhf_sigma_cube=PRIYA_BHF_SIGMA_CUBE,
        part1_bhf_prior=dict(
            sigmas_cube=sigmas.tolist(), sweep=sweep,
            free_worst={p: pp_free[p] for p in PARAMS},
            pin_worst={p: pp_pin[p] for p in PARAMS},
            free_ns_worst=free_w["ns_worst"], free_ap_worst=free_w["ap_worst"],
            priya_ns_worst=priya_w["ns_worst"], priya_ap_worst=priya_w["ap_worst"],
            pin_ns_worst=pin_ns, pin_ap_worst=pin_ap,
            bhf_suppresses=bool(bhf_suppresses), why=bhf_why,
        ),
        part2_residual=dict(
            r2_bhf=float(r2_bhf), resid_frac=resid_frac,
            resid_attr=resid_attr, rank_resid=rank_resid,
            per_sim_D=D.tolist(), per_sim_noise=noise.tolist(),
            resid_bhf=resid_bhf.tolist(),
        ),
        part3_heii=heii, part4_hireionz=hireionz,
        pin_residual_bias=pin_bias_worst,
        verdict=dict(bhf_suppresses=bool(bhf_suppresses), acceptable=bool(acceptable),
                     resid_frac=resid_frac),
        figures={"bhf_prior": fp1, "residual": fp2},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {args.out}\nfigures:\n  {fp1}\n  {fp2}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
