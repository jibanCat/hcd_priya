#!/usr/bin/env python3
"""desi_stage1.py (v2 after Reviewers C and D, 2026-09-24) -- DESI mechanism follow-up, Stage 1: four stored-product
diagnostics in the 4-space C4 = (ns, tau0_amp, dtau0, k_SiIII_DESI_z1). READ-ONLY analysis layer over the FROZEN WS-A
modules (scripts/a2c_joint_calib.py = JC, scripts/a2c_joint_calib_ext.py = EX), imported by path. No sampling.

Preregistration v2: hcd_priya_notes/desi_mechanism_followup_2026-09/2026-09-24-DESI-STAGE1-PREREGISTRATION-v2.md.
Null: posterior-draw exchangeability with the LOO rank-1 moment downdate and ONE shared draw index per mock per
replicate (WS-A section 6), B = 20000; p = (r+1)/(B+1). Seeds: SeedSequence(20260925).spawn(6) in the order
("null","DESI"), ("null","eBOSS"), ("boot"), ("dirs"), ("perm","DESI"), ("perm","eBOSS").
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

REPO = "/home/mfho/hcd_priya"
SEED_ROOT = 20260925
B_NULL = 20000
B_BOOT = 2000
B_DIRS = 2000
B_PERM = 20000
ALPHA = 0.05
CHUNK = 2000
C4 = ("ns", "tau0_amp", "dtau0", "k_SiIII_DESI_z1")
P3 = C4[:3]
BOUNDS = {"ns": (0.0, 1.0), "tau0_amp": (0.75, 1.25), "dtau0": (-0.40, 0.25), "k_SiIII_DESI_z1": (-3.0, -1.0)}   # k in log10 s/km
LOG10_COORDS = ("k_SiIII_DESI_z1",)
TAIL_ABS_PROJ = 2.0
D1_DEGENERATE_SPREAD = 0.01       # reference q95 - q05 below this: every direction equally coherent, percentile uninformative (descriptive only)
WSA_PREREG = "2026-08-11-JOINT-CALIBRATION-PREREG-v2"
WSA_SEED_ROOT = 20260811


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


JC = _load("a2c_joint_calib", os.path.join(REPO, "scripts", "a2c_joint_calib.py"))
EX = _load("a2c_joint_calib_ext", os.path.join(REPO, "scripts", "a2c_joint_calib_ext.py"))


class Stage1Refusal(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- #
#  coordinates and moments
# --------------------------------------------------------------------------------------------- #
def coord_matrix(rec, coords):
    """(L, d) draws and (d,) truth for the named coordinates, k-node in log10 (finite-checked)."""
    params = list(rec["params"])
    cols = [params.index(c) for c in coords]
    X = np.array(rec["X"][:, cols], float)
    t = np.array(rec["t"][cols], float)
    for j, c in enumerate(coords):
        if c in LOG10_COORDS:
            if np.any(X[:, j] <= 0) or t[j] <= 0:
                raise Stage1Refusal(f"non-positive value for log10 coordinate {c}")
            X[:, j] = np.log10(X[:, j]); t[j] = np.log10(t[j])
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(t))):
        raise Stage1Refusal("non-finite coordinate after transformation")
    return X, t


def _sqrtm_sym(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 1e-300, None))) @ V.T


def whitened_z(X, t):
    mu = X.mean(axis=0)
    S = np.cov(X.T, ddof=1).reshape(X.shape[1], X.shape[1])
    return JC.sym_inv_sqrt(S) @ (mu - t)


def loo_z(X):
    return JC.loo_z_table(X, None, list(range(X.shape[1])))


def loo_e_table(X, cols):
    """(L, len(cols)): row j = (mu_j - x_j) / sd_j with LOO mean and sd (rank-1 downdate; verified against direct LOO)."""
    Y = X[:, cols]
    L = Y.shape[0]
    mu = Y.mean(axis=0)
    A = ((Y - mu) ** 2).sum(axis=0)
    dev = Y - mu
    mu_j = (L * mu - Y) / (L - 1)
    A_j = A - (L / (L - 1)) * dev ** 2
    return (mu_j - Y) / np.sqrt(A_j / (L - 2))


def _corr_from_S(S):
    dd = np.sqrt(np.diagonal(S, axis1=-2, axis2=-1))
    return S / (dd[..., :, None] * dd[..., None, :])


def _partial(C, i, j, given):
    idx = [i, j] + list(given)
    sub = C[..., idx, :][..., :, idx]
    P = np.linalg.inv(sub)
    return -P[..., 0, 1] / np.sqrt(P[..., 0, 0] * P[..., 1, 1])


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def _spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _partial_spearman(a, b, c):
    """Rank-based partial correlation of a and b given c."""
    C = np.corrcoef(np.stack([_rank(a), _rank(b), _rank(c)]))
    return float(_partial(C, 0, 1, [2]))


def _p_two_sided(null, obs):
    return float(min(1.0, 2.0 * min(JC.p_upper(null, obs), JC.p_lower(null, obs))))


# --------------------------------------------------------------------------------------------- #
#  D1 physical-space coherence
# --------------------------------------------------------------------------------------------- #
def d1_coherence(Xs, v_max, v_min, rng, axes=("ns", "tau0_amp", "dtau0")):
    """Physical coordinates are scaled by the across-mock median posterior sd per coordinate (fixed diagonal D), so
    cosines between mocks are scale-free. Degenerate case: when every posterior shares one orientation every direction
    is coherent, the percentile is arbitrary and the label says so."""
    Sig = [np.cov(X.T, ddof=1) for X in Xs]
    sd_med = np.median([np.sqrt(np.diag(S)) for S in Sig], axis=0)
    roots = [_sqrtm_sym(S) / sd_med[:, None] for S in Sig]
    def coh(v):
        U = np.stack([R @ v for R in roots]); U /= np.linalg.norm(U, axis=1, keepdims=True)
        w, V = np.linalg.eigh(U.T @ U / len(U)); ubar = V[:, -1]
        return float(np.mean(np.abs(U @ ubar))), ubar.tolist()
    out = {}
    d = len(axes)
    for name, v in [("v_max", v_max), ("v_min", v_min)] + [(f"axis_{a}", np.eye(d)[i]) for i, a in enumerate(axes)]:
        c, ub = coh(np.asarray(v, float) / np.linalg.norm(v)); out[name] = dict(coherence=c, physical_direction_scaled=ub)
    ref = np.empty(B_DIRS)
    for b in range(B_DIRS):
        v = rng.standard_normal(d); v /= np.linalg.norm(v); ref[b], _ = coh(v)
    q05, q50, q95 = (float(np.quantile(ref, q)) for q in (0.05, 0.5, 0.95))
    pct = float(np.mean(ref <= out["v_max"]["coherence"]) * 100.0)
    out["reference"] = dict(n_dirs=B_DIRS, q05=q05, q50=q50, q95=q95, q99=float(np.quantile(ref, 0.99)), spread_q95_q05=q95 - q05)
    out["v_max_percentile"] = pct
    if q95 - q05 < D1_DEGENERATE_SPREAD:
        out["label"] = "DEGENERATE: every direction equally coherent (shared orientation or near-isotropic posteriors); percentile uninformative"
    elif pct >= 95.0:
        out["label"] = "coherent physical direction"
    else:
        out["label"] = "not more coherent than a generic whitened direction"
    out["sd_scale_median"] = sd_med.tolist()
    return out


# --------------------------------------------------------------------------------------------- #
#  D2 orientation mismatch: realized coupling against the exchangeability null of the same statistic
# --------------------------------------------------------------------------------------------- #
def d2_orientation(Xs, ts, J, pairs=((0, 2), (0, 1), (2, 1)), names=("ns", "tau0_amp", "dtau0")):
    N = len(Xs)
    out = {}
    e_tabs = [loo_e_table(X, list(range(X.shape[1]))) for X in Xs]
    E_obs = np.stack([(X.mean(axis=0) - t) / X.std(axis=0, ddof=1) for X, t in zip(Xs, ts)])
    for (a, b) in pairs:
        rho = np.array([np.corrcoef(X[:, a], X[:, b])[0, 1] for X in Xs])
        r_obs = float(np.corrcoef(E_obs[:, a], E_obs[:, b])[0, 1])
        r_b = np.empty(J.shape[0])
        for s in range(0, J.shape[0], CHUNK):
            Js = J[s:s + CHUNK]
            Ea = np.stack([e_tabs[i][Js[:, i], a] for i in range(N)], axis=1)
            Eb = np.stack([e_tabs[i][Js[:, i], b] for i in range(N)], axis=1)
            Ea = Ea - Ea.mean(axis=1, keepdims=True); Eb = Eb - Eb.mean(axis=1, keepdims=True)
            r_b[s:s + CHUNK] = (Ea * Eb).sum(axis=1) / np.sqrt((Ea ** 2).sum(axis=1) * (Eb ** 2).sum(axis=1))
        null_med = float(np.median(r_b))
        p = _p_two_sided(r_b, r_obs)
        out[f"{names[a]}_{names[b]}"] = dict(r_raw_realized=r_obs, null_median=null_med, D_orient=float(r_obs - null_med), p_two_sided=p,
                                              fires=bool(p < ALPHA), rho_posterior_mean=float(rho.mean()), rho_posterior_median=float(np.median(rho)),
                                              rho_iqr=[float(np.quantile(rho, .25)), float(np.quantile(rho, .75))],
                                              null_q=[float(np.quantile(r_b, q)) for q in (0.025, 0.5, 0.975)])
    return out


# --------------------------------------------------------------------------------------------- #
#  D3 mediation versus projection in C4 (4-D whitening, k in log10; the frozen 2-D values are reported separately)
# --------------------------------------------------------------------------------------------- #
def d3_mediation(X4s, t4s, J, rng_boot):
    N = len(X4s)
    Z = np.stack([whitened_z(X, t) for X, t in zip(X4s, t4s)])
    S = Z.T @ Z / N
    C = _corr_from_S(S)
    c_nd, c_nk = float(C[0, 2]), float(C[0, 3])
    pc_nd_k = float(_partial(C, 0, 2, [3]))
    pc_nk_mf = float(_partial(C, 0, 3, [2, 1]))
    tabs = [loo_z(X) for X in X4s]
    stats = {k: [] for k in ("c_nd", "c_nk", "pc_nd_k", "pc_nk_mf")}
    for S_b in EX._null_moments(tabs, J):
        Cb = _corr_from_S(S_b)
        stats["c_nd"].append(Cb[:, 0, 2]); stats["c_nk"].append(Cb[:, 0, 3])
        stats["pc_nd_k"].append(_partial(Cb, 0, 2, [3])); stats["pc_nk_mf"].append(_partial(Cb, 0, 3, [2, 1]))
    stats = {k: np.concatenate(v) for k, v in stats.items()}
    p = {k: JC.p_upper(np.abs(stats[k]), abs(v)) for k, v in (("c_nd", c_nd), ("c_nk", c_nk), ("pc_nd_k", pc_nd_k), ("pc_nk_mf", pc_nk_mf))}
    fires_nd_k, fires_nk_mf = (bool(x) for x in EX.holm([p["pc_nd_k"], p["pc_nk_mf"]], alpha=ALPHA))
    # bootstrap (mock resampling) intervals for the four statistics, descriptive
    idx = rng_boot.integers(0, N, size=(B_BOOT, N))
    bs = {k: np.empty(B_BOOT) for k in stats}
    for b in range(B_BOOT):
        Zb = Z[idx[b]]; Cb = _corr_from_S(Zb.T @ Zb / N)
        bs["c_nd"][b] = Cb[0, 2]; bs["c_nk"][b] = Cb[0, 3]; bs["pc_nd_k"][b] = _partial(Cb, 0, 2, [3]); bs["pc_nk_mf"][b] = _partial(Cb, 0, 3, [2, 1])
    ci = {k: [float(np.quantile(v, 0.16)), float(np.quantile(v, 0.84))] for k, v in bs.items()}
    mediated = bool(p["c_nd"] < ALPHA and (not fires_nd_k) and abs(pc_nd_k) < 0.5 * abs(c_nd))
    projection = bool(p["c_nk"] < ALPHA and (not fires_nk_mf) and abs(pc_nk_mf) < 0.5 * abs(c_nk))
    if mediated and projection:
        label = "JOINT, NOT SEPARABLE"
    elif mediated:
        label = "MEDIATED"
    elif projection:
        label = "PROJECTION"
    elif fires_nd_k and fires_nk_mf:
        label = "INDEPENDENT"
    else:
        label = "UNRESOLVED"
    # the smallest |partial| compatible with the observed marginals (partial = (c_nk - c_nd c_dk) / sqrt((1-c_nd^2)(1-c_dk^2)) family):
    return dict(S4=S.tolist(), corr4=C.tolist(), coords=list(C4), whitening="4-D, k in log10 (NOT the frozen 2-D plane values)",
                c_ns_dtau0=dict(observed=c_nd, p=p["c_nd"], boot68=ci["c_nd"]), c_ns_k=dict(observed=c_nk, p=p["c_nk"], boot68=ci["c_nk"]),
                pc_ns_dtau0_given_k=dict(observed=pc_nd_k, p=p["pc_nd_k"], holm_fires=fires_nd_k, boot68=ci["pc_nd_k"], threshold_half_marginal=0.5 * abs(c_nd)),
                pc_ns_k_given_dtau0_tau0amp=dict(observed=pc_nk_mf, p=p["pc_nk_mf"], holm_fires=fires_nk_mf, boot68=ci["pc_nk_mf"], threshold_half_marginal=0.5 * abs(c_nk)),
                mediated_rule=mediated, projection_rule=projection, label=label,
                label_semantics=dict(MEDIATED="k-sufficient: the k-node co-displacement accounts for the (ns, dtau0) structure; a common factor that k measures best is NOT excluded",
                                     PROJECTION="the k-node association is accounted for by the (ns, mean-flux) structure; requires the marginal c(ns, k) to fire",
                                     note="'not PROJECTION' is weak evidence against M-D when the smallest partial compatible with the marginals exceeds the threshold"),
                eigenvalues_desc=np.linalg.eigvalsh(S)[::-1].tolist())


# --------------------------------------------------------------------------------------------- #
#  D4 tail and rail attribution (any coordinate set; eBOSS control on P3)
# --------------------------------------------------------------------------------------------- #
def d4_tail_rail(Xs, ts, v_max, rng_perm, J=None, coords=C4, frozen_proj=None):
    """Xs, ts on the coordinate set `coords` (the first three = P3 define the projection space; rails use all `coords`)."""
    N = len(Xs)
    X3 = [X[:, :3] for X in Xs]; t3 = [t[:3] for t in ts]
    Z = np.stack([whitened_z(X, t) for X, t in zip(X3, t3)])
    S = Z.T @ Z / N
    tmax = float(np.linalg.eigvalsh(S)[-1])
    shares = []
    for i in range(N):
        S_i = (Z.T @ Z - np.outer(Z[i], Z[i])) / (N - 1)
        shares.append((tmax - float(np.linalg.eigvalsh(S_i)[-1])) / tmax)
    v = np.asarray(v_max, float); v /= np.linalg.norm(v)
    proj = Z @ v
    if frozen_proj is not None and np.max(np.abs(proj - np.asarray(frozen_proj))) > 1e-6:
        raise Stage1Refusal("recomputed projections on v_max differ from the frozen WS-A descriptive")
    tail = [int(i) for i in np.where(np.abs(proj) > TAIL_ABS_PROJ)[0]]
    d = np.empty((N, len(coords)))
    sd_ns = np.empty(N)
    for i, X in enumerate(Xs):
        mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1); sd_ns[i] = sd[0]
        for c, name in enumerate(coords):
            lo, hi = BOUNDS[name]
            d[i, c] = min(mu[c] - lo, hi - mu[c]) / sd[c]
    r = d.min(axis=1)
    ap = np.abs(proj)
    rho_obs = _spearman(r, ap)
    perm = np.empty(B_PERM)
    for b in range(B_PERM):
        perm[b] = _spearman(r[rng_perm.permutation(N)], ap)
    p_one = JC.p_lower(perm, rho_obs)
    # shared-index alternative null: the projections change with the pseudo-truth, the rail distances do not
    p_shared = None; rho_shared_q = None
    if J is not None:
        tabs = [loo_z(X) for X in X3]
        rho_b = np.empty(J.shape[0])
        for b in range(J.shape[0]):
            pb = np.abs(np.stack([tabs[i][J[b, i]] for i in range(N)]) @ v)
            rho_b[b] = _spearman(r, pb)
        p_shared = JC.p_lower(rho_b, rho_obs); rho_shared_q = [float(np.quantile(rho_b, q)) for q in (0.025, 0.5, 0.975)]
    tail_mask = np.zeros(N, bool); tail_mask[tail] = True
    frac = {}
    for c, name in enumerate(coords):
        frac[name] = dict(tail_lt1=float(np.mean(d[tail_mask, c] < 1.0)) if tail_mask.any() else None, rest_lt1=float(np.mean(d[~tail_mask, c] < 1.0)),
                          tail_lt1p5=float(np.mean(d[tail_mask, c] < 1.5)) if tail_mask.any() else None, rest_lt1p5=float(np.mean(d[~tail_mask, c] < 1.5)))
    return dict(coords=list(coords), T_max=tmax, loo_shares=[float(s) for s in shares], loo_shares_sorted_desc=[int(i) for i in np.argsort(shares)[::-1]],
                projection_on_vmax=proj.tolist(), tail_set=tail, n_tail=len(tail),
                rail_distance_sd_units={coords[c]: d[:, c].tolist() for c in range(len(coords))}, rail_min=r.tolist(),
                spearman_rail_vs_absproj=rho_obs, p_one_sided_negative=p_one, fires=bool(p_one < ALPHA),
                p_shared_index_null=p_shared, shared_index_null_q=rho_shared_q,
                width_confound=dict(spearman_absproj_vs_sd_ns=_spearman(ap, sd_ns), spearman_rail_vs_sd_ns=_spearman(r, sd_ns),
                                    partial_spearman_rail_vs_absproj_given_sd_ns=_partial_spearman(r, ap, sd_ns)),
                frac_within_bound=frac)


# --------------------------------------------------------------------------------------------- #
#  decision (preregistration v2 section 2: every combination routed explicitly)
# --------------------------------------------------------------------------------------------- #
def decide(d2_desi, d2_eboss, d3, d4_desi):
    fires_desi = bool(d2_desi["ns_dtau0"]["fires"]); fires_eboss = bool(d2_eboss["ns_dtau0"]["fires"])
    rail = bool(d4_desi["fires"]); lab = d3["label"]
    if rail and lab == "MEDIATED":
        return dict(outcome="RAIL-ASSOCIATED AND k-SUFFICIENT (both fire)", mechanism_class="M-B (rail / parameterization) or M-A local to the rails, together with M-C; not separable at N = 48",
                    stage2="NOT RUN", next_="checkpoint; PI decision between the parameterization change and the pinned-node arm")
    if rail:
        return dict(outcome="RAIL-ASSOCIATED TAILS", mechanism_class="M-B (rail / parameterization), or M-A local to the rails", stage2="NOT RUN",
                    next_="parameterization change (Stage 3 class; PI decision); note that this stop skips the M-A versus M-B separation")
    if lab == "MEDIATED":
        return dict(outcome="k-SUFFICIENT (MEDIATED BY THE k-NODE)", mechanism_class="M-C (metal k-node channel) localized; common factor not excluded", stage2="NOT RUN",
                    next_="pinned-node arm (PI decision)")
    if lab == "JOINT, NOT SEPARABLE":
        return dict(outcome="JOINT DIRECTION SPANNING ns, dtau0 AND k; NOT SEPARABLE", mechanism_class="unresolved among M-A/M-B/M-C at N = 48", stage2="NOT RECOMMENDED",
                    next_="checkpoint; PI decision")
    if fires_desi and fires_eboss:
        return dict(outcome="ORIENTATION PATTERN SHARED BY THE eBOSS CONTROL (not DESI-specific)", mechanism_class="machinery-generic feature or chance; not a DESI mechanism",
                    stage2="NOT RECOMMENDED", next_="checkpoint; PI decision")
    if fires_desi and lab == "PROJECTION":
        return dict(outcome="ORIENTATION ERROR WITH THE k-NODE ASSOCIATION PROJECTED ONTO THE MEAN-FLUX STRUCTURE", mechanism_class="M-D context; M-A versus M-B-orientation fork recorded",
                    stage2="NOT RECOMMENDED (three-lanes 4.2 as adopted in PI #24)", next_="checkpoint; PI decision on the M-A versus M-B fork")
    if fires_desi and lab in ("UNRESOLVED", "INDEPENDENT"):
        return dict(outcome="ORIENTATION ERROR, NO RAIL, NO MEDIATION", mechanism_class="M-A versus M-B-orientation (live)",
                    stage2="TRIGGERED: preregister Stage 2 (1-mock pilot first); LAUNCHING it needs a new PI decision (PI #24 item 4)", next_="Stage 2 preregistration, then PI decision")
    return dict(outcome="NOT FURTHER LOCALIZABLE WITH STORED DRAWS", mechanism_class="unresolved among M-A/M-B/M-D/M-E", stage2="NOT RECOMMENDED", next_="checkpoint; PI decision")


def _rngs():
    kids = np.random.SeedSequence(SEED_ROOT).spawn(6)
    return dict(null_desi=np.random.default_rng(kids[0]), null_eboss=np.random.default_rng(kids[1]), boot=np.random.default_rng(kids[2]),
                dirs=np.random.default_rng(kids[3]), perm_desi=np.random.default_rng(kids[4]), perm_eboss=np.random.default_rng(kids[5]))


def analyze(recs_d, recs_e, wsa, rngs, B_null=None):
    B = B_NULL if B_null is None else B_null
    X3, t3 = map(list, zip(*[coord_matrix(r, P3) for r in recs_d])); X4, t4 = map(list, zip(*[coord_matrix(r, C4) for r in recs_d]))
    v_max = np.asarray(wsa["DESI"]["directions"]["v_max"], float); v_min = np.asarray(wsa["DESI"]["directions"]["v_min"], float)
    Jd = EX._shared_index(recs_d, rngs["null_desi"], B)
    d1 = d1_coherence(X3, v_max, v_min, rngs["dirs"])
    d2 = d2_orientation(X3, t3, Jd)
    d3 = d3_mediation(X4, t4, Jd, rngs["boot"])
    d4 = d4_tail_rail(X4, t4, v_max, rngs["perm_desi"], J=Jd, coords=C4, frozen_proj=wsa["DESI"]["descriptive"]["proj_vmax"])
    Xe, te = map(list, zip(*[coord_matrix(r, P3) for r in recs_e]))
    Je = EX._shared_index(recs_e, rngs["null_eboss"], B)
    d2e = d2_orientation(Xe, te, Je)
    d4e = d4_tail_rail(Xe, te, np.asarray(wsa["eBOSS"]["directions"]["v_max"], float), rngs["perm_eboss"], J=Je, coords=P3,
                       frozen_proj=wsa["eBOSS"]["descriptive"]["proj_vmax"])
    dec = decide(d2, d2e, d3, d4)
    frozen = dict(note="frozen WS-A values for reference; NOT the D3 statistics (those use one 4-D whitening with k in log10)",
                  wsa_S3_corr_ns_dtau0_3D=float(np.asarray(wsa["DESI"]["S3"])[0, 2] / np.sqrt(np.asarray(wsa["DESI"]["S3"])[0, 0] * np.asarray(wsa["DESI"]["S3"])[2, 2])),
                  ext_plane_ns_dtau0_c_2D=0.3752, ext_plane_ns_dtau0_note="EXT-1 Level B (ns, dtau0) plane, 2-D whitening, p 0.0084, not counted (P3-internal)",
                  ext_N_plane_ns_k_c_2D=0.502, ext_N_plane_note="EXT-1 Level B, 2-D plane whitening with linear k (a2c_joint_calib_ext_result.json)")
    return dict(D1=d1, D2_DESI=d2, D2_eBOSS_control=d2e, D3=d3, D4_DESI=d4, D4_eBOSS_control=d4e, decision=dec, frozen_reference=frozen,
                B_null=B, B_boot=B_BOOT, B_dirs=B_DIRS, B_perm=B_PERM, alpha=ALPHA, N_DESI=len(recs_d), N_eBOSS=len(recs_e),
                mock_ids_DESI=[int(r["m"]) for r in recs_d], mock_ids_eBOSS=[int(r["m"]) for r in recs_e],
                L_DESI=[int(r["L"]) for r in recs_d], L_eBOSS=[int(r["L"]) for r in recs_e], bounds=BOUNDS, coords=list(C4), tail_abs_proj=TAIL_ABS_PROJ)


def preflight(desi_dir, desi_sha, eboss_dir, eboss_sha, wsa_json):
    recs_d = EX.load_arm_ext(desi_dir, desi_sha, "DESI")
    recs_e = EX.load_arm_ext(eboss_dir, eboss_sha, "eBOSS")
    wsa = json.load(open(wsa_json))
    if wsa.get("prereg") != WSA_PREREG or int(wsa.get("seed_root", -1)) != WSA_SEED_ROOT:
        raise Stage1Refusal("WS-A JSON identity fields are not the frozen ones")
    for c in C4:
        if c not in recs_d[0]["params"]:
            raise Stage1Refusal(f"coordinate {c} absent from the DESI live set")
    rep = {}
    for arm, recs in (("DESI", recs_d), ("eBOSS", recs_e)):
        X3, t3 = zip(*[coord_matrix(r, P3) for r in recs])
        Z = np.stack([whitened_z(X, t) for X, t in zip(X3, t3)]); S = Z.T @ Z / len(Z)
        w, V = np.linalg.eigh(S); v = EX._sign_fix(V[:, -1])
        if np.max(np.abs(v - np.asarray(wsa[arm]["directions"]["v_max"]))) > 1e-6:
            raise Stage1Refusal(f"recomputed v_max differs from the frozen WS-A v_max ({arm})")
        proj = Z @ v
        if np.max(np.abs(proj - np.asarray(wsa[arm]["descriptive"]["proj_vmax"]))) > 1e-6:
            raise Stage1Refusal(f"recomputed projections differ from the frozen WS-A descriptive ({arm})")
        rep[arm] = dict(n=len(recs), L=[int(r["L"]) for r in recs], tail_set=[int(i) for i in np.where(np.abs(proj) > TAIL_ABS_PROJ)[0]])
    for r in recs_d:
        coord_matrix(r, C4)      # finite / positivity check on the k-node
    return dict(v_max_reproduced=True, projections_reproduced=True, wsa_identity_ok=True, DESI=rep["DESI"], eBOSS=rep["eBOSS"],
                n_desi=len(recs_d), n_eboss=len(recs_e), bounds=BOUNDS)


def _strict_json_dump(obj, path):
    s = json.dumps(obj, indent=1, sort_keys=True, allow_nan=False)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(s + "\n")
    os.replace(tmp, path)


def run(desi_dir, desi_sha, eboss_dir, eboss_sha, wsa_json, out_json):
    if os.path.exists(out_json) or os.path.exists(out_json + ".tmp"):
        raise Stage1Refusal(f"output exists: {out_json}")
    pf = preflight(desi_dir, desi_sha, eboss_dir, eboss_sha, wsa_json)
    recs_d = EX.load_arm_ext(desi_dir, desi_sha, "DESI"); recs_e = EX.load_arm_ext(eboss_dir, eboss_sha, "eBOSS")
    wsa = json.load(open(wsa_json))
    res = analyze(recs_d, recs_e, wsa, _rngs())
    res["preflight"] = pf; res["prereg"] = "2026-09-24-DESI-STAGE1-PREREGISTRATION-v2"; res["seed_root"] = SEED_ROOT
    res["seed_streams"] = ["null:DESI", "null:eBOSS", "boot", "dirs", "perm:DESI", "perm:eBOSS"]
    _strict_json_dump(res, out_json)
    return res
