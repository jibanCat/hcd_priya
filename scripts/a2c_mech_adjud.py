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


# =====================================================================================
# ANALYSIS DRIVER (single invocation; frozen scope)
# =====================================================================================

ABSORBERS = ["alpha_lls", "alpha_subdla", "alpha_dla"]


def run_adjudication(recs, design_unit, param_limits, lowk_npz=None,
                     B=PERM_B, seed=PERM_SEED):
    """Assemble Layer-O status, H1, H2-exposure and H3 into one result dict.

    Layer O and H2 Condition A are NOT computed -- they are recorded as frozen scope
    determinations with their refusal reasons, per the prereg.
    """
    from scipy.stats import spearmanr
    out = {}
    n = len(recs)
    co = tau_eff_coefficients()
    lo, hi = np.asarray(param_limits)[:, 0], np.asarray(param_limits)[:, 1]

    # --- frozen scope determinations (recorded, never computed) ---
    try:
        layer_o_refuse()
    except AdjudRefusal as e:
        out["LayerO"] = dict(status="INADMISSIBLE", computed=False, reason=str(e))
    try:
        h2_condition_a_refuse()
    except AdjudRefusal as e:
        h2_cond_a = dict(status="NON_ADJUDICABLE", computed=False, reason=str(e))
    out["H1_H2_interaction"] = dict(
        status="NOT_TESTABLE",
        reason="PI #14 section 12 condition 2 requires low-k information mismatch to be "
               "INDEPENDENTLY PRESENT; H2 Condition A is non-adjudicable, so the "
               "interaction cannot be tested. No interaction model is fitted.")

    # --- deployed n_s pull/rank (recomputed with the frozen arithmetic) ---
    ns_pull, ns_rank, theta0, tru_unit = [], [], [], []
    for r in recs:
        j = r["names"].index("ns")
        col = np.asarray(r["draws"], float)[:, j]
        t = float(np.asarray(r["truth"], float)[j])
        ns_pull.append((float(col.mean()) - t) / float(col.std(ddof=1)))
        ns_rank.append(float(np.mean(col < t)))
        theta0.append(t)
        tru_unit.append(np.asarray(r["truth"], float)[:9])
    ns_pull = np.asarray(ns_pull); ns_rank = np.asarray(ns_rank)
    theta0 = np.asarray(theta0); TU = np.asarray(tru_unit)
    ns_phys = physical_ns(theta0)

    # ================================ H1 ================================
    S_k, S_1 = local_support(TU, design_unit, k=KNN_K)
    h1 = dict(metric=f"{KNN_K}rd-NN distance in the frozen 9-D unit cube",
              n=int(n), design_n=int(np.asarray(design_unit).shape[0]))
    for lab, thr in (("primary_1.00", NS_SPLIT_PRIMARY),
                     ("secondary_0.995", NS_SPLIT_SECONDARY)):
        hi_m = ns_phys > thr
        grp = {}
        for gname, m in (("above", hi_m), ("below", ~hi_m)):
            if m.sum() == 0:
                grp[gname] = dict(n=0); continue
            grp[gname] = dict(
                n=int(m.sum()),
                pull_mean=float(ns_pull[m].mean()),
                pull_sd=float(ns_pull[m].std(ddof=1)) if m.sum() > 1 else None,
                rank_mean=float(ns_rank[m].mean()),
                support_median=float(np.median(S_k[m])),
                support_range=[float(S_k[m].min()), float(S_k[m].max())])
        h1[lab] = grp
    # collinearity guard (frozen abort rule)
    rho_S_ns = float(spearmanr(S_k, ns_phys).statistic)
    h1["collinearity"] = dict(spearman_support_vs_truth_ns=rho_S_ns,
                              abort_threshold=0.8,
                              identified=bool(abs(rho_S_ns) <= 0.8))
    h1["marginal"] = permutation_p(lambda a, b: spearmanr(a, b).statistic,
                                   S_k, np.abs(ns_pull), B=B, seed=seed)
    if h1["collinearity"]["identified"]:
        pp = permutation_p(lambda a, b: spearman_partial(a, b, ns_phys),
                           S_k, np.abs(ns_pull), B=B, seed=seed + 1)
        pp["material"] = bool(abs(pp["observed"]) >= MDE_RHO)
        h1["partial_controlling_truth_ns"] = pp
        h1["status"] = "ADJUDICATED"
    else:
        h1["partial_controlling_truth_ns"] = dict(
            computed=False,
            reason=f"support and truth n_s are collinear (|rho| {abs(rho_S_ns):.3f} > 0.8); "
                   "the partial correlation is uninformative")
        h1["status"] = "NON_IDENTIFIED"
    # within-group stratified check (removes the group contrast by construction)
    strat = {}
    for gname, m in (("above_1.00", ns_phys > NS_SPLIT_PRIMARY),
                     ("below_1.00", ns_phys <= NS_SPLIT_PRIMARY)):
        if m.sum() >= 8:
            strat[gname] = dict(n=int(m.sum()),
                                rho=float(spearmanr(S_k[m], np.abs(ns_pull[m])).statistic))
        else:
            strat[gname] = dict(n=int(m.sum()), rho=None,
                                note="fewer than 8 realizations: not evaluable")
    h1["stratified"] = strat
    h1["extrapolation"] = dict(
        max_truth_ns=float(ns_phys.max()),
        n_truths_above_max_design=int(np.sum(ns_phys > float(
            (np.asarray(design_unit)[:, 0] * (hi[0] - lo[0]) + lo[0]).max()))),
        max_design_ns=float((np.asarray(design_unit)[:, 0] * (hi[0] - lo[0]) + lo[0]).max()))
    out["H1"] = h1

    # ================================ H2 ================================
    h2 = dict(condition_A=h2_cond_a)
    if lowk_npz is not None:
        d = np.load(lowk_npz, allow_pickle=True)
        h2["condition_B_exposure"] = f_lowk_exposure(d["DESI_k"], d["DESI_u"])
    h2["status"] = "UNRESOLVED_EXPOSURE_ONLY"
    h2["establishment"] = ("PI #14 section 11.4 requires BOTH Condition A and Condition B. "
                           "A is non-adjudicable, so H2 can be neither established nor "
                           "excluded under this authorization.")
    out["H2"] = h2

    # ================================ H3 ================================
    tau_r = []
    for r in recs:
        tidx = [r["names"].index(f"tau0_z{i}") for i in range(13)]
        D = np.asarray(r["draws"], float)[:, tidx]
        amp = np.exp(np.log(np.clip(D, 1e-8, None)).mean(axis=1))
        T = np.asarray(r["truth"], float)[tidx]
        amp_t = float(np.exp(np.log(np.clip(T, 1e-8, None)).mean()))
        tau_r.append(float(np.mean(amp < amp_t)))
    tau_r = np.asarray(tau_r)

    proj, compat, r2a, r2d = [], [], [], []
    for r in recs:
        se = r["sites_extra"]
        amp_d = np.asarray(se["tau0_amp"]["draws"], float)
        dt_d = np.asarray(se["dtau0"]["draws"], float)
        A = np.column_stack([np.asarray(r["draws"], float)[:, r["names"].index(a)]
                             for a in ABSORBERS])
        pr = absorber_projection(amp_d, dt_d, A, co)
        # absorber-associated displacement of the posterior mean, contracted with the
        # absorber block's own mean offset (local first-order approximation)
        a_off = A.mean(0) - np.asarray(r["truth"], float)[
            [r["names"].index(a) for a in ABSORBERS]]
        d_amp = float(pr["dln_amp_dabs"] @ a_off)
        d_dt = float(pr["ddtau_dabs"] @ a_off)
        pred = project_to_tau_eff(d_amp, d_dt, co["c"])
        obs_bar = (float(np.log(np.asarray(se["tau0_amp"]["draws"], float)).mean())
                   - float(np.log(max(float(se["tau0_amp"]["truth"]), 1e-300)))
                   + co["c_bar"] * (float(dt_d.mean()) - float(se["dtau0"]["truth"])))
        pred_bar = d_amp + co["c_bar"] * d_dt
        proj.append(pred)
        compat.append(pred_bar / obs_bar if abs(obs_bar) > 1e-12 else np.nan)
        r2a.append(pr["r2_amp"]); r2d.append(pr["r2_dtau"])
    proj = np.asarray(proj); compat = np.asarray(compat, float)
    fin = np.isfinite(compat)
    h3 = dict(
        coefficients=dict(zbar=co["zbar"], c_bar=co["c_bar"], c_bar_desi=co["c_bar_desi"],
                          admixture_ratio=co["admixture_ratio"],
                          n_desi_constrained=int(co["desi_mask"].sum())),
        regression_quality=dict(median_r2_ln_amp=float(np.median(r2a)),
                                median_r2_dtau=float(np.median(r2d))),
        compatibility=dict(median=float(np.median(compat[fin])) if fin.any() else None,
                           mean=float(np.mean(compat[fin])) if fin.any() else None,
                           n_finite=int(fin.sum()),
                           floor=H3_COMPAT_FLOOR,
                           material=bool(fin.any() and
                                         abs(float(np.median(compat[fin]))) >= H3_COMPAT_FLOOR)),
        projected_by_rung=[dict(z=float(co["z"][i]), c=float(co["c"][i]),
                                desi_constrained=bool(co["desi_mask"][i]),
                                median_projected=float(np.median(proj[:, i])))
                           for i in range(13)],
        rank_extremeness_assoc=permutation_p(
            lambda a, b: spearmanr(a, b).statistic,
            np.abs(compat[fin]) if fin.any() else np.zeros(n),
            2 * np.abs(tau_r[fin] - 0.5) if fin.any() else np.zeros(n),
            B=min(B, 20000), seed=seed + 2),
        caveat="LOCAL PROJECTED APPROXIMATION, NOT AN INTERVENTION. The compatibility "
               "fraction is GEOMETRIC, never causal. No absorber prior was removed, "
               "weakened or changed.")
    out["H3"] = h3

    # ---- primary Holm family (m = 2: H1, H3) ----
    prim = []
    if h1["status"] == "ADJUDICATED":
        prim.append(("H1", h1["partial_controlling_truth_ns"]["p"]))
    if h3["rank_extremeness_assoc"] is not None:
        prim.append(("H3", h3["rank_extremeness_assoc"]["p"]))
    if prim:
        fl = holm([p for _, p in prim])
        out["primary_holm_family"] = dict(
            m=len(prim),
            tests={k: dict(p=p, holm_significant=bool(f)) for (k, p), f in zip(prim, fl)})
    out["meta"] = dict(n=n, B=B, seed=seed, prereg="2026-08-07 mechanism adjudication",
                       mde_rho=MDE_RHO, compat_floor=H3_COMPAT_FLOOR)
    return out
