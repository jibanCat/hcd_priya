#!/usr/bin/env python3
"""Expanded nuisance analysis for the WS-A execution (PREREG AMENDMENT EXT-1, 2026-09-22;
PI DECISIONS #20 sections 3, 4, 7, 8, 9). READ-ONLY diagnostic/readout layer.

This module NEVER modifies the frozen primary tool ``scripts/a2c_joint_calib.py``; it imports
it by file path and reuses its whitening (``sym_inv_sqrt``, ``observed_z``), LOO exchangeability
null (``loo_z_table``), estimators (``p_upper``, ``p_lower``, ``holm_pair``, ``eig22_batch``)
and its census/sha loader (``load_arm``). No generative or inference code is touched; no
sampling, no likelihood evaluation, no model code.

Frozen design (amendment EXT-1):
  Level A  per-parameter calibration (pull mean/sd, rank KS p, Beta-band count) on every live
           parameter with a propagated truth; Holm over the arm's parameters for the rank claim;
           consistency conjuncts against the committed gate JSON / esc-diag values at rtol 1e-9.
  Level B  (ns, j) planes for every live j != ns: c_j = whitened-displacement correlation, |c_j|
           upper tail vs the exchangeability null, Holm over the planes; B2 = plane-extreme pair
           (T_pmax_B, T_pmin_B) Holm pair; descriptive Spearman(pull_ns, pull_j).
  Level C  11 blocks per arm (COSMO, EMU, MF, HCD, METALS, RES and COSMO+each): T_max/T_min
           (ONE Holm family of 2K p-values), T_det band, T_corr, bootstrap stability of v_max/v_min.
  FULL     all live parameters: descriptive, stability only; nothing counted.
Seeds: SeedSequence(20260922).spawn(4) in the FROZEN order [null-DESI, null-eBOSS, boot-DESI,
boot-eBOSS]; ONE shared draw index per mock per null replicate reused across every subspace of the
arm. B_null = 20000, B_boot = 2000, alpha = 0.05, stability threshold 0.7. Independent of the
primary's SeedSequence(20260811), so this layer cannot perturb the primary result.
"""
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np
from scipy import stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("a2c_joint_calib", os.path.join(_HERE, "a2c_joint_calib.py"))
JC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(JC)

EXT_PREREG = "2026-09-22-JOINT-CALIBRATION-PREREG-v2-EXT-AMENDMENT-1"
EXT_SEED_ROOT = 20260922
B_NULL_EXT = 20_000
B_BOOT_EXT = 2_000
ALPHA = 0.05
STAB_THRESH = 0.7
RTOL = 1e-9
CHUNK = 1000                       # null replicates per einsum chunk (memory bound)
N_TOTAL = 48

MAIN_NAMES_25 = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback",
                 "tau0_z0", "tau0_z1", "tau0_z2", "tau0_z3", "tau0_z4", "tau0_z5", "tau0_z6",
                 "tau0_z7", "tau0_z8", "tau0_z9", "tau0_z10", "tau0_z11", "tau0_z12",
                 "alpha_lls", "alpha_subdla", "alpha_dla")
TAU_RUNGS = tuple(f"tau0_z{i}" for i in range(13))          # derived, excluded from B/C/full
COSMO = ("ns", "Ap")
EMU = ("herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
MF = ("tau0_amp", "dtau0")
HCD = ("alpha_lls", "alpha_subdla", "alpha_dla", "s_lls", "s_subdla", "s_dla")
RES = ("f_res_amp", "f_res_slope")
METALS = {
    "DESI": ("f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1",
             "k_SiIII_DESI_z0", "k_SiIII_DESI_z1", "k_SiII_DESI_z0", "k_SiII_DESI_z1"),
    "eBOSS": ("f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1", "k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1"),
}
SITES_IN_TRUTH_VEC = set(COSMO) | set(EMU) | {"alpha_lls", "alpha_subdla", "alpha_dla"}
CASE_A = tuple(COSMO + EMU + MF + HCD)                       # paired truths, same semantics
PRIMARY_TARGETS = ("ns", "tau0_amp", "dtau0")                # Case D check


class ExtRefusal(RuntimeError):
    """Integrity/schema/consistency failure of the extension: no output."""


def live_params(arm):
    return tuple(COSMO + EMU + MF + HCD + METALS[arm] + RES)


def expected_sites_extra(arm):
    return tuple(sorted(MF + ("s_lls", "s_subdla", "s_dla") + METALS[arm] + RES))


def case_of(name):
    return "A" if name in CASE_A else "C"


def block_map(arm):
    """Ordered blocks (frozen coordinate order inside each block)."""
    b = {"COSMO": COSMO, "EMU": EMU, "MF": MF, "HCD": HCD, "METALS": METALS[arm], "RES": RES}
    for k in ("EMU", "MF", "HCD", "METALS", "RES"):
        b["COSMO+" + k] = tuple(COSMO + b[k])
    return b


# ------------------------------ general Holm (step-down, strict <) ------------------------------

def holm(pvals, alpha=ALPHA):
    p = np.asarray(pvals, float)
    m = p.size
    fires = np.zeros(m, bool)
    order = np.argsort(p, kind="stable")
    for r, idx in enumerate(order):
        if p[idx] < alpha / (m - r):
            fires[idx] = True
        else:
            break
    return fires


# ------------------------------ loading + schema + pairing ------------------------------

def load_arm_ext(outdir, sha_file, arm):
    """Census + sha via the frozen loader, then the FULL live-parameter matrix per mock."""
    JC.load_arm(outdir, sha_file)          # census + sha + P7 alignment conjuncts (refuse on fail)
    params = live_params(arm)
    exp_se = expected_sites_extra(arm)
    recs = []
    for m in range(N_TOTAL):
        with open(os.path.join(outdir, f"mock_{m:04d}.pkl"), "rb") as f:
            z = pickle.load(f)
        names = tuple(z["names"])
        if names != MAIN_NAMES_25:
            raise ExtRefusal(f"mock {m}: names != frozen 25-list")
        draws = np.asarray(z["draws"], float)
        truth = np.asarray(z["truth_vec"], float)
        L = int(z["L"])
        if draws.shape != (L, 25) or truth.shape != (25,):
            raise ExtRefusal(f"mock {m}: draws/truth shape {draws.shape}/{truth.shape}")
        se = z["sites_extra"]
        if tuple(sorted(se.keys())) != exp_se:
            raise ExtRefusal(f"mock {m}: sites_extra keys != inventory for {arm}")
        X = np.empty((L, len(params)))
        t = np.empty(len(params))
        for c, p in enumerate(params):
            if p in SITES_IN_TRUTH_VEC:
                j = names.index(p)
                X[:, c] = draws[:, j]
                t[c] = truth[j]
            else:
                col = np.asarray(se[p]["draws"], float)
                if col.shape != (L,):
                    raise ExtRefusal(f"mock {m}: sites_extra[{p}] rows {col.shape} != L {L}")
                X[:, c] = col
                t[c] = float(se[p]["truth"])
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(t))):
            raise ExtRefusal(f"mock {m}: non-finite draws/truth in the live set")
        if np.any(np.ptp(X, axis=0) <= 0):
            raise ExtRefusal(f"mock {m}: a live parameter has constant draws")
        recs.append(dict(m=m, X=X, t=t, L=L, params=params, truth_full=truth,
                         se_truth={k: float(se[k]["truth"]) for k in se}))
    return recs


def check_pairing_ext(recs_d, recs_e):
    """Case A truths bit-identical across arms on every mock. The Case D primary targets (ns,
    tau0_amp, dtau0) are checked FIRST so a primary-target mismatch is always reported as CASE D."""
    pd_, pe_ = recs_d[0]["params"], recs_e[0]["params"]
    for ra, rb in zip(recs_d, recs_e):
        for p in PRIMARY_TARGETS:
            if ra["t"][pd_.index(p)] != rb["t"][pe_.index(p)]:
                raise ExtRefusal(f"CASE D (primary target) truth pairing FAIL on mock {ra['m']} for {p}")
        if not np.array_equal(ra["truth_full"], rb["truth_full"]):
            raise ExtRefusal(f"truth_vec pairing FAIL on mock {ra['m']}")
        for p in CASE_A:
            if ra["t"][pd_.index(p)] != rb["t"][pe_.index(p)]:
                raise ExtRefusal(f"Case A truth pairing FAIL on mock {ra['m']} for {p}")
    return True


def whitening_feasibility(recs, arm):
    """Deterministic PREFLIGHT pass (amendment EXT-1a): evaluate the observed whitening and the LOO
    whitening table for EVERY subspace run_ext will use (all (ns, j) planes, the 11 blocks, the full
    space) on every mock, discard the results, return only the count of (subspace, mock) passes.
    Any non-positive-definite covariance raises here, BEFORE the primary has consumed the execution."""
    params = recs[0]["params"]
    i_ns = params.index("ns")
    subspaces = [(i_ns, params.index(p)) for p in params if p != "ns"]
    subspaces += [tuple(params.index(p) for p in m) for m in block_map(arm).values()]
    subspaces += [tuple(range(len(params)))]
    n = 0
    for cols in subspaces:
        for r in recs:
            JC.observed_z(r["X"], r["t"], list(cols))
            JC.loo_z_table(r["X"], None, list(cols))
            n += 1
    return n


# ------------------------------ Level A ------------------------------

def _pulls_ranks(recs, c):
    pulls = np.array([(r["X"][:, c].mean() - r["t"][c]) / r["X"][:, c].std(ddof=1) for r in recs])
    ranks = np.array([float(np.mean(r["X"][:, c] < r["t"][c])) for r in recs])
    post_sd = np.array([r["X"][:, c].std(ddof=1) for r in recs])
    return pulls, ranks, post_sd


def rank_uniformity(u):
    """Exactly analyze_sbc_perleg.rank_uniformity: KS p vs U(0,1) + Beta(k, N+1-k) band count."""
    u = np.asarray(u, float)
    n = u.size
    ks_p = float(sps.kstest(u, "uniform").pvalue)
    us = np.sort(u)
    out = int(sum(not (sps.beta.ppf(0.025, k, n + 1 - k) <= us[k - 1] <= sps.beta.ppf(0.975, k, n + 1 - k))
                  for k in range(1, n + 1)))
    return dict(n=int(n), mean=float(u.mean()), ks_p=ks_p, n_outside_band=out,
                verdict=("UNIFORM" if ks_p > 0.05 else "NON-UNIFORM"))


def finite_L_ref(Ls):
    Ls = np.asarray(Ls, float)
    return float(np.mean(np.sqrt((1.0 + 1.0 / Ls) * (Ls - 1.0) / (Ls - 3.0))))


def level_a(recs, arm):
    params = recs[0]["params"]
    Ls = [r["L"] for r in recs]
    ref = finite_L_ref(Ls)
    out, ks_ps = {}, []
    for c, p in enumerate(params):
        pulls, ranks, post_sd = _pulls_ranks(recs, c)
        ru = rank_uniformity(ranks)
        out[p] = dict(block=_block_of(p, arm), case=case_of(p),
                      pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
                      pull_sem=float(pulls.std(ddof=1) / np.sqrt(len(pulls))),
                      rank_mean=ru["mean"], rank_ks_p=ru["ks_p"], n_outside_band=ru["n_outside_band"],
                      rank_verdict_raw=ru["verdict"], post_sd_median=float(np.median(post_sd)),
                      pulls=pulls.tolist(), ranks=ranks.tolist())
        ks_ps.append(ru["ks_p"])
    fires = holm(ks_ps)
    for (p, f) in zip(params, fires):
        out[p]["rank_nonuniform_holm"] = bool(f)
    return dict(finite_L_pull_sd_reference=ref, m_A=len(params), holm_alpha=ALPHA, params=out)


def _block_of(p, arm):
    for k, v in (("COSMO", COSMO), ("EMU", EMU), ("MF", MF), ("HCD", HCD), ("METALS", METALS[arm]), ("RES", RES)):
        if p in v:
            return k
    return "other"


def check_consistency_ext(recs, gate_json, arm, esc_diag_json=None):
    """Level A conjuncts vs the committed gate JSON (and esc-diag for DESI), rtol 1e-9. Returns
    only labels + pass flags (the compared values are already public, but nothing new is exposed)."""
    with open(gate_json) as f:
        leg = json.load(f)["legs"][arm]
    params = recs[0]["params"]
    if list(map(int, leg["L_all"])) != [r["L"] for r in recs]:
        raise ExtRefusal("L census != gate JSON L_all")
    checks = []
    for p in ("ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"):
        pulls, ranks, _ = _pulls_ranks(recs, params.index(p))
        checks.append((f"{p} pull mean", float(pulls.mean()), float(leg["pulls"][p]["mean"])))
        checks.append((f"{p} pull sd", float(pulls.std(ddof=1)), float(leg["pulls"][p]["std"])))
        checks.append((f"{p} rank ks_p", rank_uniformity(ranks)["ks_p"], float(leg["rank_uniformity"][p]["ks_p"])))
    sites = leg.get("repaired_sectors", {}).get("sites", {})
    for p in tuple(METALS[arm]) + RES:
        if p not in sites:
            raise ExtRefusal(f"gate JSON repaired_sectors lacks {p}")
        pulls, ranks, _ = _pulls_ranks(recs, params.index(p))
        checks.append((f"{p} pull mean", float(pulls.mean()), float(sites[p]["pull_mean"])))
        checks.append((f"{p} pull sd", float(pulls.std(ddof=1)), float(sites[p]["pull_sd"])))
        checks.append((f"{p} rank ks_p", rank_uniformity(ranks)["ks_p"], float(sites[p]["rank_ks_p"])))
    if esc_diag_json is not None:
        with open(esc_diag_json) as f:
            esc = json.load(f)
        for p, key in (("tau0_amp", "tau0_amp_sampled"), ("dtau0", "dtau0_sampled")):
            pulls, _, _ = _pulls_ranks(recs, params.index(p))
            checks.append((f"{p} sampled pull sd", float(pulls.std(ddof=1)),
                           float(esc["P1"]["sampled_sites"][key]["pull_sd"])))
    for label, got, want in checks:
        if not np.isclose(got, want, rtol=RTOL, atol=0.0):
            raise ExtRefusal(f"consistency FAIL on {label}: {got!r} vs {want!r}")
    return {label: True for label, _, _ in checks}


# ------------------------------ shared null machinery ------------------------------

def _shared_index(recs, rng, B):
    Ls = np.array([r["L"] for r in recs])
    return np.stack([rng.integers(0, L, size=B) for L in Ls], axis=1)      # (B, N)


def _null_moments(loo_tabs, J, chunk=CHUNK):
    """Yield stacked S_b (chunk, d, d) for the shared index J over the mocks' LOO tables."""
    B, N = J.shape
    for s in range(0, B, chunk):
        Js = J[s:s + chunk]
        G = np.stack([loo_tabs[i][Js[:, i]] for i in range(N)], axis=1)      # (b, N, d)
        yield np.einsum("bni,bnj->bij", G, G) / N


def _sign_fix(v):
    v = v.copy()
    if v[np.argmax(np.abs(v))] < 0:
        v *= -1.0
    return v


def _corr_max_offdiag(S):
    d = S.shape[-1]
    dd = np.sqrt(np.diagonal(S, axis1=-2, axis2=-1))
    C = S / (dd[..., :, None] * dd[..., None, :])
    iu = np.triu_indices(d, 1)
    return np.max(np.abs(C[..., iu[0], iu[1]]), axis=-1)


def _subspace_analysis(recs, cols, J, rng_boot, B_boot, want_boot=True):
    """Observed + null statistics for one subspace; returns dict + null arrays."""
    N = len(recs)
    Z = np.stack([JC.observed_z(r["X"], r["t"], list(cols)) for r in recs])   # (N, d)
    S = (Z.T @ Z) / N
    w, V = np.linalg.eigh(S)
    d = len(cols)
    tmax_o, tmin_o = float(w[-1]), float(w[0])
    tdet_o = float(np.sum(np.log(w)))
    tcorr_o = float(_corr_max_offdiag(S)) if d > 1 else 0.0
    v_max, v_min = _sign_fix(V[:, -1]), _sign_fix(V[:, 0])
    tabs = [JC.loo_z_table(r["X"], None, list(cols)) for r in recs]
    tmax_b, tmin_b, tdet_b, tcorr_b, c_b = [], [], [], [], []
    for S_b in _null_moments(tabs, J):
        wb = np.linalg.eigvalsh(S_b)
        tmax_b.append(wb[:, -1]); tmin_b.append(wb[:, 0]); tdet_b.append(np.sum(np.log(wb), axis=1))
        if d > 1:
            tcorr_b.append(_corr_max_offdiag(S_b))
        if d == 2:
            c_b.append(S_b[:, 0, 1] / np.sqrt(S_b[:, 0, 0] * S_b[:, 1, 1]))
    tmax_b, tmin_b, tdet_b = map(np.concatenate, (tmax_b, tmin_b, tdet_b))
    tcorr_b = np.concatenate(tcorr_b) if d > 1 else np.zeros_like(tmax_b)
    res = dict(
        coords=list(cols), d=d, S=S.tolist(),
        T_max=dict(observed=tmax_o, p=JC.p_upper(tmax_b, tmax_o),
                   null_q=[float(np.quantile(tmax_b, q)) for q in (0.05, 0.5, 0.95)]),
        T_min=dict(observed=tmin_o, p=JC.p_lower(tmin_b, tmin_o),
                   null_q=[float(np.quantile(tmin_b, q)) for q in (0.05, 0.5, 0.95)]),
        T_det=dict(observed=tdet_o,
                   central_band=[float(np.quantile(tdet_b, 0.025)), float(np.quantile(tdet_b, 0.975))]),
        T_corr=dict(observed=tcorr_o, p=(JC.p_upper(tcorr_b, tcorr_o) if d > 1 else None)),
        v_max=v_max.tolist(), v_min=v_min.tolist(), eigenvalues_desc=w[::-1].tolist(),
        S_diag=[float(S[i, i]) for i in range(d)],
    )
    res["T_det"]["inside"] = bool(res["T_det"]["central_band"][0] <= tdet_o <= res["T_det"]["central_band"][1])
    if d == 2:
        c_o = float(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]))
        c_b = np.concatenate(c_b)
        res["c"] = dict(observed=c_o, p_abs=JC.p_upper(np.abs(c_b), abs(c_o)),
                        null_q_abs=[float(np.quantile(np.abs(c_b), q)) for q in (0.5, 0.95)])
        lo_o, hi_o = JC.eig22_batch(S[None])
        res["lambda_lo"], res["lambda_hi"] = float(lo_o[0]), float(hi_o[0])
    if want_boot:
        idx = rng_boot.integers(0, N, size=(B_boot, N))
        cmax = np.empty(B_boot); cmin = np.empty(B_boot)
        for b in range(B_boot):
            Zb = Z[idx[b]]
            Sb = (Zb.T @ Zb) / N
            _, Vb = np.linalg.eigh(Sb)
            cmax[b] = abs(float(Vb[:, -1] @ v_max)); cmin[b] = abs(float(Vb[:, 0] @ v_min))
        res["stability"] = dict(
            v_max=dict(median=float(np.median(cmax)), q05=float(np.quantile(cmax, 0.05)),
                       q95=float(np.quantile(cmax, 0.95)), frac_ge_thresh=float(np.mean(cmax >= STAB_THRESH))),
            v_min=dict(median=float(np.median(cmin)), q05=float(np.quantile(cmin, 0.05)),
                       q95=float(np.quantile(cmin, 0.95)), frac_ge_thresh=float(np.mean(cmin >= STAB_THRESH))),
            threshold=STAB_THRESH, B_boot=int(B_boot))
        res["proj_vmax"] = (Z @ v_max).tolist()
    return res, dict(tmax=tmax_b, tmin=tmin_b)


# ------------------------------ Level B ------------------------------

P3_INTERNAL_PLANES = ("tau0_amp", "dtau0")     # (ns, j) planes inside the primary's P3: NOT counted (EXT-1a)


def level_b(recs, arm, J, rng_boot):
    """Level B. Every (ns, j) plane is computed and reported; the COUNTED family (Holm) and the B2
    extremes use only the planes OUTSIDE P3 (prereg v2 section 2 excluded the P3-internal planes
    from any sweep so that no post-hoc localization inside P3 is possible; the primary's T_corr is
    the frozen statistic for those two planes). Per-plane eigen p-values are descriptive only."""
    params = recs[0]["params"]
    i_ns = params.index("ns")
    others = [p for p in params if p != "ns"]
    counted = [p for p in others if p not in P3_INTERNAL_PLANES]
    planes, hi_b, lo_b = {}, None, None
    pulls_ns = _pulls_ranks(recs, i_ns)[0]
    for p in others:
        cols = (i_ns, params.index(p))
        res, nulls = _subspace_analysis(recs, cols, J, rng_boot, B_boot=1, want_boot=False)
        res["case"] = case_of(p); res["block"] = _block_of(p, arm)
        res["counted_in_family_B"] = p in counted
        res["in_primary"] = p in ("tau0_amp", "dtau0", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla")
        if p in P3_INTERNAL_PLANES:
            res["not_counted_reason"] = "P3-internal plane (prereg v2 section 2); the primary's T_corr governs"
        for key in ("T_max", "T_min", "T_corr"):
            res[key]["p_DESCRIPTIVE_not_counted"] = res[key].pop("p")
        pulls_j = _pulls_ranks(recs, cols[1])[0]
        res["spearman_pull_ns_vs_pull_j_DESCRIPTIVE"] = float(sps.spearmanr(pulls_ns, pulls_j).statistic)
        planes[p] = res
        if p in counted:
            hi_b = nulls["tmax"] if hi_b is None else np.maximum(hi_b, nulls["tmax"])
            lo_b = nulls["tmin"] if lo_b is None else np.minimum(lo_b, nulls["tmin"])
    p_abs = [planes[p]["c"]["p_abs"] for p in counted]
    fires = holm(p_abs)
    for p in others:
        planes[p]["c"]["holm_fires"] = bool(fires[counted.index(p)]) if p in counted else None
    hi_o = [planes[p]["lambda_hi"] for p in counted]; lo_o = [planes[p]["lambda_lo"] for p in counted]
    tpmax_o, tpmin_o = float(np.max(hi_o)), float(np.min(lo_o))
    p_pmax, p_pmin = JC.p_upper(hi_b, tpmax_o), JC.p_lower(lo_b, tpmin_o)
    f_pmax, f_pmin = JC.holm_pair(p_pmax, p_pmin)
    amax, amin = counted[int(np.argmax(hi_o))], counted[int(np.argmin(lo_o))]
    return dict(m_B=len(counted), counted_planes=counted, holm_alpha=ALPHA, planes=planes,
                implicated_EXT_N=[p for p, f in zip(counted, fires) if f],
                B2=dict(T_pmax_B=dict(observed=tpmax_o, p=p_pmax, holm_fires=f_pmax,
                                      argmax_plane_DESCRIPTIVE=amax,
                                      argmax_plane_c_holm_fires=bool(planes[amax]["c"]["holm_fires"]),
                                      interpretation_rule=("EXT-1a: a T_pmax_B detection is a re-expression of the "
                                                           "ratified n_s marginal unless the argmax plane's c_j "
                                                           "survives Holm; it licenses no plane naming otherwise")),
                        T_pmin_B=dict(observed=tpmin_o, p=p_pmin, holm_fires=f_pmin,
                                      argmin_plane_DESCRIPTIVE=amin,
                                      argmin_plane_c_holm_fires=bool(planes[amin]["c"]["holm_fires"]))))


# ------------------------------ Level C + full space ------------------------------

def level_c(recs, arm, J, rng_boot, B_boot):
    params = recs[0]["params"]
    blocks = block_map(arm)
    out, labels, pvals = {}, [], []
    for name, members in blocks.items():
        cols = tuple(params.index(p) for p in members)
        res, _ = _subspace_analysis(recs, cols, J, rng_boot, B_boot=B_boot, want_boot=True)
        res["members"] = list(members)
        res["case"] = "A" if all(case_of(p) == "A" for p in members) else "C"
        res["holm_pair_within_block_DESCRIPTIVE"] = list(JC.holm_pair(res["T_max"]["p"], res["T_min"]["p"]))
        out[name] = res
        labels += [(name, "T_max"), (name, "T_min")]
        pvals += [res["T_max"]["p"], res["T_min"]["p"]]
    fires = holm(pvals)
    for (name, stat), f in zip(labels, fires):
        out[name][stat]["holm_family_fires"] = bool(f)
    for name, res in out.items():
        both = res["T_max"]["holm_family_fires"] and res["T_min"]["holm_family_fires"]
        tcorr_fires = bool(res["T_corr"]["p"] is not None and res["T_corr"]["p"] < ALPHA)
        rot = bool(both and res["T_det"]["inside"] and tcorr_fires
                   and res["stability"]["v_max"]["median"] >= STAB_THRESH
                   and res["stability"]["v_min"]["median"] >= STAB_THRESH)
        res["implicated_EXT_B"] = bool(res["T_max"]["holm_family_fires"] or res["T_min"]["holm_family_fires"])
        res["T_corr"]["fires"] = tcorr_fires
        res["rotation_word_conjuncts_met_within_arm"] = rot
        res["marginal_pattern_class"] = MARGINAL_PATTERN_CLASS[name]
    # EXT-1a mechanical labels (frozen BEFORE execution; the readout quotes these, never re-derives them)
    for name, res in out.items():
        res["ext_b_label"] = _ext_b_label(name, res, out)
    return dict(K=len(blocks), family_size=len(pvals), holm_alpha=ALPHA, blocks=out,
                implicated_EXT_B=[n for n, r in out.items() if r["implicated_EXT_B"]],
                ext_b_labels={n: r["ext_b_label"] for n, r in out.items()})


# Blocks whose members already carry DISCLOSED non-unit marginal pull sds (ns 1.27, dtau0 0.79, tau0_amp 1.10,
# alpha_dla 1.30, alpha_lls 1.11, alpha_subdla 1.09, Ap 1.06): a T_max/T_min survival there is expected under the
# pure-marginal pattern alone (EXT-1a operating-characteristic table). EMU, METALS, RES carry no disclosed non-unit
# marginal (unread or read out as healthy).
MARGINAL_PATTERN_CLASS = {"COSMO": "known_nonunit", "MF": "known_nonunit", "HCD": "known_nonunit",
                          "EMU": "unit_or_unread", "METALS": "unit_or_unread", "RES": "unit_or_unread",
                          "COSMO+EMU": "known_nonunit", "COSMO+MF": "known_nonunit", "COSMO+HCD": "known_nonunit",
                          "COSMO+METALS": "known_nonunit", "COSMO+RES": "known_nonunit"}


def _ext_b_label(name, res, allblocks):
    """Frozen label logic (EXT-1a rules ii and iii):
    not implicated                                   -> 'none'
    implicated and T_corr fires                      -> 'EXT-B-joint' (joint structure in the block)
    implicated, class unit_or_unread, no T_corr      -> 'EXT-B-localization' (genuine new localization to the block)
    implicated, COSMO+X with X not implicated alone  -> 'attributed-to-ns-marginal'
    implicated, class known_nonunit, no T_corr       -> 'EXT-B-anisotropy-consistent-with-disclosed-marginals'"""
    if not res["implicated_EXT_B"]:
        return "none"
    if res["T_corr"]["fires"]:
        return "EXT-B-joint"
    if res["marginal_pattern_class"] == "unit_or_unread":
        return "EXT-B-localization"
    if name.startswith("COSMO+"):
        x = name.split("+", 1)[1]
        if x in allblocks and not allblocks[x]["implicated_EXT_B"]:
            return "attributed-to-ns-marginal"
    return "EXT-B-anisotropy-consistent-with-disclosed-marginals"


def full_space(recs, arm, J, rng_boot, B_boot):
    params = recs[0]["params"]
    cols = tuple(range(len(params)))
    res, _ = _subspace_analysis(recs, cols, J, rng_boot, B_boot=B_boot, want_boot=True)
    res["members"] = list(params)
    for key in ("T_max", "T_min", "T_corr"):      # EXT-1a: NO p-value in the full space, null quantiles only
        res[key].pop("p", None)
    res["stability_screen_passed_v_max"] = bool(res["stability"]["v_max"]["median"] >= STAB_THRESH)
    res["stability_screen_passed_v_min"] = bool(res["stability"]["v_min"]["median"] >= STAB_THRESH)
    res["stability_null_reference_note"] = ("EXT-1a section 4: under the null the bootstrap median |cos| of v_max is itself "
                                            "about 0.55 at d = 27 (0.71 at d = 10, 0.83 at d = 6, 0.92 at d = 3), and the 0.7 "
                                            "screen passes about 7 percent of null datasets at d = 27 (nearly all for d <= 6); "
                                            "read the statement as a screen against that reference")
    res["statement"] = ("full-space orientation passes the 0.7 stability SCREEN for v_max (see null reference)"
                        if res["stability_screen_passed_v_max"] else "full-space orientation not resolved at N = 48")
    return res


# ------------------------------ drivers ------------------------------

def analyze_arm_ext(recs, arm, rng_null, rng_boot, B_null=None, B_boot=None):
    B_null = B_NULL_EXT if B_null is None else B_null
    B_boot = B_BOOT_EXT if B_boot is None else B_boot
    J = _shared_index(recs, rng_null, B_null)
    return dict(
        arm=arm, n=len(recs), params=list(recs[0]["params"]),
        L_census=[r["L"] for r in recs], B_null=int(B_null), B_boot=int(B_boot),
        level_A=level_a(recs, arm),
        level_B=level_b(recs, arm, J, rng_boot),
        level_C=level_c(recs, arm, J, rng_boot, B_boot),
        full_space=full_space(recs, arm, J, rng_boot, B_boot),
    )


def cross_arm(desi, eboss):
    """Case-aware cross-arm flags (no paired statistic; flags only)."""
    out = dict(level_B={}, level_C={})
    for p, r in desi["level_B"]["planes"].items():
        if r["case"] == "A" and p in eboss["level_B"]["planes"]:
            out["level_B"][p] = dict(desi_fires=r["c"]["holm_fires"],
                                     eboss_fires=eboss["level_B"]["planes"][p]["c"]["holm_fires"])
        else:
            out["level_B"][p] = dict(case="C", note="survey-specific; within-arm only; no cross-arm rule")
    for b, r in desi["level_C"]["blocks"].items():
        rb = eboss["level_C"]["blocks"][b]
        if r["case"] == "A":
            eboss_clean = bool((not rb["implicated_EXT_B"]) and rb["T_det"]["inside"])
            out["level_C"][b] = dict(desi_implicated=r["implicated_EXT_B"], desi_label=r["ext_b_label"],
                                     eboss_implicated=rb["implicated_EXT_B"], eboss_block_clean=eboss_clean,
                                     rotation_word_final=bool(r["rotation_word_conjuncts_met_within_arm"] and eboss_clean),
                                     rotation_rule="EXT-1a: rotation word for a Case A block requires the within-arm conjuncts AND the eBOSS block clean (prereg v2 section 11 v)")
        else:
            out["level_C"][b] = dict(case="C", desi_implicated=r["implicated_EXT_B"], desi_label=r["ext_b_label"],
                                     rotation_word_final=bool(r["rotation_word_conjuncts_met_within_arm"]),
                                     note="contains survey-specific parameters; within-arm only; no eBOSS conjunct available")
    return out


def preflight_ext(desi_dir, desi_sha, desi_gate, esc_diag_json, eboss_dir, eboss_sha, eboss_gate):
    """Loader + schema + pairing + consistency conjuncts for BOTH arms. Returns pass flags only.
    Intended to run BEFORE the primary run() so that a refusal here consumes nothing."""
    recs_d = load_arm_ext(desi_dir, desi_sha, "DESI")
    recs_e = load_arm_ext(eboss_dir, eboss_sha, "eBOSS")
    check_pairing_ext(recs_d, recs_e)
    cons_d = check_consistency_ext(recs_d, desi_gate, "DESI", esc_diag_json=esc_diag_json)
    cons_e = check_consistency_ext(recs_e, eboss_gate, "eBOSS")
    wf_d = whitening_feasibility(recs_d, "DESI")
    wf_e = whitening_feasibility(recs_e, "eBOSS")
    return dict(schema_ok=True, pairing_ok=True, case_D_fires=False,
                consistency_checks=dict(DESI=len(cons_d), eBOSS=len(cons_e)),
                whitening_feasibility_passes=dict(DESI=int(wf_d), eBOSS=int(wf_e)),
                n_params=dict(DESI=len(recs_d[0]["params"]), eBOSS=len(recs_e[0]["params"])))


def run_ext(desi_dir, desi_sha, desi_gate, esc_diag_json,
            eboss_dir, eboss_sha, eboss_gate, out_json, B_null=None, B_boot=None):
    if os.path.exists(out_json):
        raise ExtRefusal(f"refusing to overwrite existing output {out_json}")
    ss = np.random.SeedSequence(EXT_SEED_ROOT)
    kids = ss.spawn(4)     # FROZEN order: null-DESI, null-eBOSS, boot-DESI, boot-eBOSS
    rngs = [np.random.default_rng(k) for k in kids]
    recs_d = load_arm_ext(desi_dir, desi_sha, "DESI")
    recs_e = load_arm_ext(eboss_dir, eboss_sha, "eBOSS")
    check_pairing_ext(recs_d, recs_e)
    cons_d = check_consistency_ext(recs_d, desi_gate, "DESI", esc_diag_json=esc_diag_json)
    cons_e = check_consistency_ext(recs_e, eboss_gate, "eBOSS")
    desi = analyze_arm_ext(recs_d, "DESI", rngs[0], rngs[2], B_null, B_boot)
    eboss = analyze_arm_ext(recs_e, "eBOSS", rngs[1], rngs[3], B_null, B_boot)
    out = dict(prereg=EXT_PREREG, seed_root=EXT_SEED_ROOT, alpha=ALPHA, stability_threshold=STAB_THRESH,
               consistency=dict(DESI=cons_d, eBOSS=cons_e),
               inventory=dict(DESI=dict(params=list(recs_d[0]["params"]), case={p: case_of(p) for p in recs_d[0]["params"]}),
                              eBOSS=dict(params=list(recs_e[0]["params"]), case={p: case_of(p) for p in recs_e[0]["params"]}),
                              derived_excluded=list(TAU_RUNGS)),
               DESI=desi, eBOSS=eboss, cross_arm=cross_arm(desi, eboss))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    return out
