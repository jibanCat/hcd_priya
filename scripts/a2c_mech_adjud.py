#!/usr/bin/env python3
"""A2c MECHANISM ADJUDICATION (PI #14; prereg 2026-08-07-A2C-MECHANISM-ADJUDICATION-PREREG.md).

POPULATION-ONLY. Reads committed A2c pkls, the frozen PRIYA design, and committed emulator
validation artifacts. Performs NO simulation, NO emulator training, NO likelihood fit, NO
sampler run, NO mock generation. Descriptive/adjudicative only: nothing here can alter
Outcome D, ARM ROW 3, or any certification statement.

SCOPE DETERMINATIONS FROZEN IN THE PREREG (not re-litigated here):
  * Layer O is INADMISSIBLE -- the stored loglik IS ll_rank_frac_mean (PI #9 excluded) and
    the corrected statistic is uncomputable. This module REFUSES to compute it.
  * H2 Condition A is NON-ADJUDICABLE -- no P1D residuals were stored. Only F_lowk exposure.
  * The H1-H2 interaction is NOT TESTABLE (it requires Condition A).
"""
import numpy as np

# --- frozen constants (prereg) ---------------------------------------------------------
NS_SPLIT_PRIMARY = 1.00        # PI-frozen; NEVER optimized
NS_SPLIT_SECONDARY = 0.995     # the code's OWN pre-existing sparse/C_emu-inflation boundary
KNN_K = 3                      # frozen local-support neighbour order
LOWK_CUT = 0.0102              # the committed C_emu coverage floor ("low-k hole")
PERM_B, PERM_SEED = 50_000, 20260811
MDE_RHO = 0.40                 # frozen H1 effect-size floor
H3_COMPAT_FLOOR = 0.30         # frozen H3 material-fraction floor
HOLM_ALPHA = 0.05

Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])
Z_HI_DESI = 4.2


class AdjudRefusal(RuntimeError):
    """A frozen scope boundary was violated: no output."""


def layer_o_refuse():
    """Layer O is INADMISSIBLE (prereg section 1). This function exists so that any attempt
    to compute the omnibus loglik rank fails LOUDLY rather than silently bypassing the PI #9
    exclusion by renaming an equivalent statistic (PI #14 section 9)."""
    raise AdjudRefusal(
        "LAYER O IS INADMISSIBLE. loglik_rank(ll_true, ll_draws) is identically "
        "ll_rank_frac_mean (closure_legb.py:4045), and both call sites producing ll_true "
        "(:3889) and ll_draws (:3974) pass metal_nodes=None, alpha_res=None, "
        "b_res_global=None while this arm sets metal_prior='flatlog2node' and "
        "fres_selfdraw=True -- the scorer omits the sectors the arm made prior-predictive "
        "(the PI #9 Q3 defect). The corrected statistic cannot be substituted because the "
        "per-mock P1D data vectors were never stored and regenerating them is prohibited "
        "mock generation. DO NOT INSPECT.")


def h2_condition_a_refuse():
    """H2 Condition A is NON-ADJUDICABLE (prereg section 3.1)."""
    raise AdjudRefusal(
        "H2 CONDITION A IS NON-ADJUDICABLE. The A2c pkls store no P1D data vector, model "
        "prediction or residual, so empirical low-k residual variance across the 48 mocks "
        "cannot be computed. Reconstructing it is mock generation (prohibited). Only the "
        "F_lowk EXPOSURE fraction is adjudicable, and exposure is NOT evidence for H2.")


# --- physical coordinate helpers (verbatim deployed conventions) ------------------------

def physical_ns(theta_unit0, lo=0.8, hi=1.05):
    """Physical n_s = 0.8 + 0.25*theta_unit[0] (data_likelihood.py:1039)."""
    return lo + (hi - lo) * np.asarray(theta_unit0, float)


def tau_eff_coefficients(z_grid=Z_TAU0, pivot=4.0):
    """c_i = ln((1+z_i)/pivot). ln tau_eff(z_i) = ln(amp) + c_i*dtau0 + const, EXACTLY."""
    z = np.asarray(z_grid, float)
    lx = np.log(1.0 + z)
    zbar = float(np.exp(lx.mean()) - 1.0)
    c = np.log((1.0 + z) / pivot)
    desi = z <= Z_HI_DESI
    lx_d = lx[desi]
    zbar_d = float(np.exp(lx_d.mean()) - 1.0)
    return dict(z=z, c=c, zbar=zbar, c_bar=float(np.log((1.0 + zbar) / pivot)),
                desi_mask=desi, zbar_desi=zbar_d,
                c_bar_desi=float(np.log((1.0 + zbar_d) / pivot)),
                admixture_ratio=float(np.log((1.0 + zbar) / pivot)
                                      / np.log((1.0 + zbar_d) / pivot)))


# --- H1: local design support ------------------------------------------------------------

def load_design(cache_path, param_limits):
    """The frozen PRIYA design: unique sim design points, normalized to the unit cube by
    PARAM_LIMITS. Read-only; no training, no emulator evaluation."""
    import h5py
    with h5py.File(cache_path, "r") as h:
        P = h["params"][:].astype(float)
        nm = np.array([s.decode() if isinstance(s, bytes) else s for s in h["sim_name"][:]])
    _, idx = np.unique(nm, return_index=True)
    D = P[np.sort(idx)]
    lo, hi = np.asarray(param_limits)[:, 0], np.asarray(param_limits)[:, 1]
    U = (D - lo) / (hi - lo)
    if not np.all((U >= -1e-9) & (U <= 1 + 1e-9)):
        raise AdjudRefusal("design points fall outside PARAM_LIMITS after normalization")
    return dict(physical=D, unit=U, n=int(D.shape[0]))


def local_support(truth_unit, design_unit, k=KNN_K):
    """S_m = Euclidean distance in the frozen 9-D unit cube from each truth to its k-th
    nearest design point. FROZEN metric; k is not tuned."""
    T = np.atleast_2d(np.asarray(truth_unit, float))
    D = np.asarray(design_unit, float)
    if T.shape[1] != D.shape[1]:
        raise AdjudRefusal(f"dimension mismatch: truths {T.shape[1]} vs design {D.shape[1]}")
    if k > D.shape[0]:
        raise AdjudRefusal(f"k={k} exceeds the {D.shape[0]}-point design")
    d = np.sqrt(((T[:, None, :] - D[None, :, :]) ** 2).sum(-1))
    d.sort(axis=1)
    return d[:, k - 1], d[:, 0]          # (k-th NN distance, 1-NN distance)


def spearman_partial(x, y, z):
    """Partial Spearman of x,y controlling z (rank-residual form)."""
    from scipy.stats import spearmanr
    r = [np.argsort(np.argsort(np.asarray(v, float))).astype(float) for v in (x, y, z)]
    def resid(a, b):
        B = np.column_stack([np.ones_like(b), b])
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return float(spearmanr(resid(r[0], r[2]), resid(r[1], r[2])).statistic)


def permutation_p(stat_fn, a, b, B=PERM_B, seed=PERM_SEED):
    """Permutation null over realizations, with the (r+1)/(B+1) estimator and MC se."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = float(stat_fn(a, b))
    null = np.empty(B)
    for i in range(B):
        null[i] = stat_fn(a, rng.permutation(b))
    c = float(np.sum(np.abs(null - np.median(null)) >= abs(obs - np.median(null))))
    p = (c + 1.0) / (B + 1.0)
    return dict(observed=obs, p=p, B=int(B),
                null_median=float(np.median(null)), null_sd=float(null.std(ddof=1)),
                mc_se=float(np.sqrt(max(p * (1 - p), 1e-12) / B)))


def holm(pvals, alpha=HOLM_ALPHA):
    m = len(pvals)
    order = np.argsort(pvals)
    sig = [False] * m
    for j, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - j):
            sig[idx] = True
        else:
            break
    return sig


# --- H2: exposure only (Condition B side) -------------------------------------------------

def f_lowk_exposure(k, score_u, cut=LOWK_CUT):
    """F_lowk = the fraction of the total nominal n_s Fisher information carried by the
    frozen low-k block, from the COMMITTED whitened score. EXPOSURE ONLY: this quantifies how
    exposed n_s is to a low-k variance error; it is NOT evidence that such an error exists
    (Condition A is non-adjudicable)."""
    k = np.asarray(k, float); u = np.asarray(score_u, float)
    good = np.isfinite(k) & np.isfinite(u)
    k, u = k[good], u[good]
    tot = float(np.sum(u ** 2))
    if tot <= 0:
        raise AdjudRefusal("f_lowk_exposure: degenerate score vector")
    lo = k < cut
    return dict(cut=cut, n_bins=int(k.size), n_lowk=int(lo.sum()),
                F_lowk=float(np.sum(u[lo] ** 2) / tot), total_info=tot,
                interpretation="EXPOSURE ONLY. Condition A (does a low-k variance mismatch "
                               "exist) is NON-ADJUDICABLE; F_lowk is not evidence for H2.")


# --- H3: absorber -> tau_eff projection ---------------------------------------------------

def absorber_projection(x_amp, x_dtau, absorber_block, coeffs):
    """Local projected counterfactual APPROXIMATION (prereg 5.2). NOT an intervention.

    Regresses the sampled mean-flux coordinates x = (ln amp, dtau0) on the absorber block
    WITHIN one realization's own draws, forms the absorber-associated displacement of the
    posterior mean, and projects it through the exact [1, c_i] Jacobian into ln tau_eff(z_i).
    """
    la = np.log(np.clip(np.asarray(x_amp, float), 1e-300, None))
    dt = np.asarray(x_dtau, float)
    A = np.asarray(absorber_block, float)
    if A.ndim != 2 or A.shape[0] != la.size:
        raise AdjudRefusal("absorber_projection: block/draw length mismatch")
    Ac = A - A.mean(0, keepdims=True)
    X = np.column_stack([np.ones(la.size), Ac])
    beta_a, *_ = np.linalg.lstsq(X, la, rcond=None)
    beta_d, *_ = np.linalg.lstsq(X, dt, rcond=None)
    # Absorber-associated direction: response of (ln amp, dtau0) to a unit absorber shift,
    # contracted with the absorber block's own posterior-mean offset from its prior centre.
    return dict(dln_amp_dabs=beta_a[1:], ddtau_dabs=beta_d[1:],
                c=np.asarray(coeffs["c"], float), c_bar=float(coeffs["c_bar"]),
                r2_amp=float(1 - np.var(la - X @ beta_a) / max(np.var(la), 1e-300)),
                r2_dtau=float(1 - np.var(dt - X @ beta_d) / max(np.var(dt), 1e-300)))


def project_to_tau_eff(d_amp, d_dtau, c):
    """Exact Jacobian row [1, c_i]: displacement in ln tau_eff(z_i)."""
    return float(d_amp) + np.asarray(c, float) * float(d_dtau)
