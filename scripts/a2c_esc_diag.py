#!/usr/bin/env python3
"""A2c ROW-3 BOUNDED ESCALATION DIAGNOSTIC (PI #13; prereg
2026-08-07-A2C-ESCALATION-DIAG-PREREGISTRATION.md).

POPULATION-ONLY: reads the 48 committed A2c pkls, the committed gate/selfdraw JSONs and
(for the matched comparison) the committed KS N=96 pkls/JSONs. Performs NO sampling, NO
likelihood evaluation, NO refitting, NO inference change. Descriptive only: nothing here
can alter the frozen ARM ROW 3 result, the frozen gate, or any certification statement.

DEPLOYMENT-CONSISTENCY CONJUNCTS (binding, prereg section 4): the deployed statistics are
recomputed with identical arithmetic and MUST match the committed gate JSON (rtol 1e-9) --
ns/Ap/tau0amp pull mean+sd and rank KS p, plus the finite-L null sd -- else the run
refuses with NO diagnostic output. Population integrity: census 48/48, sha256 inventory
match, frozen 23-key run_cfg equality, selfdraw anchor identity, L census identity.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya python3 scripts/a2c_esc_diag.py \\
      OUTDIR GATE_JSON SELFDRAW_JSON SHA_FILE --out-json OUT.json --fig-prefix FIGPREFIX
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np

N_TOTAL = 48
RTOL = 1e-9
LEG = "DESI"

# --- frozen constants (prereg sections 3, 5, 9) -------------------------------------
LEVELS = (0.68, 0.95)
# Literal band edges: the float expression (1-0.95)/2 lands ABOVE the double for 0.025
# and would exclude an exactly-boundary rank, violating the frozen inclusive-endpoint rule.
BAND_EDGES = {0.68: (0.16, 0.84), 0.95: (0.025, 0.975)}
WINSOR_K = 3                      # ceil(0.05 * 48); the RULE is reused from n=96/k=5
TOP_K = (1, 2, 3, 5)
QUANTS = (0.025, 0.05, 0.16, 0.50, 0.84, 0.95, 0.975)
NORMREF_SEED, NORMREF_NSIM = 20260808, 200_000
BOOT_B, BOOT_SEED = 10_000, 20260807
A2C_NULL_SEED, B_NULL = 20260810, 50_000
HOLM_ALPHA = 0.05

# tau0 ladder z-grid, COPIED VERBATIM from analyze_sbc_perleg.py so numbers match
Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])
_lx = np.log(1.0 + Z_TAU0)
_lx_c = _lx - _lx.mean()


class DiagRefusal(RuntimeError):
    """Integrity or deployment-consistency failure: no diagnostic output."""


# --- deployed quantity definitions (VERBATIM; no redefinition permitted) -------------

def tau0_amp_slope(tau0_ladder):
    """COPIED VERBATIM from analyze_sbc_perleg.py (itself from analyze_sbc_heldout)."""
    y = np.log(np.clip(tau0_ladder, 1e-8, None))
    slope = np.sum(_lx_c * (y - y.mean())) / np.sum(_lx_c ** 2)
    intercept = y.mean()
    return np.exp(intercept), slope


def tau0_amp_vec(ladders):
    """Vectorized amp over an (n, 13) ladder block; identical arithmetic to the scalar."""
    a = np.asarray(ladders, float)
    y = np.log(np.clip(a, 1e-8, None))
    return np.exp(y.mean(axis=-1))


def finite_L_null_sd(L_list):
    """The deployed arm-level finite-L reference; refuses L <= 3."""
    L = np.asarray(list(L_list), float)
    if L.size == 0 or np.any(L <= 3):
        raise DiagRefusal("finite_L_null_sd: empty or L <= 3 present")
    return float(np.sqrt(((1.0 + 1.0 / L) * (L - 1.0) / (L - 3.0)).mean()))


def rank_ks_p(rank_fracs):
    from scipy import stats as _st
    u = np.asarray([x for x in rank_fracs if np.isfinite(x)], float)
    return float(_st.kstest(u, "uniform").pvalue)


def pull_of(col, truth):
    c = np.asarray(col, float)
    return (float(c.mean()) - float(truth)) / float(c.std(ddof=1))


def rank_of(col, truth):
    return float(np.mean(np.asarray(col, float) < float(truth)))


def holm(pvals, alpha=HOLM_ALPHA):
    """Step-down Holm: returns per-test significance flags in input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    sig = [False] * m
    for j, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - j):
            sig[idx] = True
        else:
            break
    return sig


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frozen_cfg(leg=LEG):
    """The frozen per-leg registry cfg, read from the deployed analyzer (single source)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analyze_firstarm_selfdraw.py")
    spec = importlib.util.spec_from_file_location("_afs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.REGISTRY[leg]["frozen_cfg"])


# --- population load + integrity ------------------------------------------------------

def load_population(outdir, sha_file, n_total=N_TOTAL, leg=LEG, require_cfg=True):
    """Census + sha + cfg conjuncts; returns per-mock records carrying the raw draw
    blocks needed downstream. Refuses on any integrity failure."""
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "mock_*.pkl"))
                   if not p.endswith(".smoke.pkl"))
    expect = [f"mock_{m:04d}.pkl" for m in range(n_total)]
    if found != expect:
        raise DiagRefusal(
            f"census != mock_0000..mock_{n_total - 1:04d} (got {len(found)} files)")

    recorded = {}
    with open(sha_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                digest, name = line.split(None, 1)
            except ValueError:
                raise DiagRefusal(f"malformed sha inventory line {lineno}: {line!r}")
            base = os.path.basename(name.strip().lstrip("*"))
            if base in recorded:
                raise DiagRefusal(f"duplicate sha inventory entry: {base}")
            recorded[base] = digest
    if set(recorded) != set(expect):
        raise DiagRefusal(f"sha inventory names != the {n_total}-mock census")

    frozen = _frozen_cfg(leg) if require_cfg else None
    recs = []
    for name in expect:
        path = os.path.join(outdir, name)
        if _sha256(path) != recorded[name]:
            raise DiagRefusal(f"sha256 mismatch vs the committed inventory: {name}")
        with open(path, "rb") as f:
            d = pickle.load(f)
        if require_cfg and dict(d["run_cfg"]) != frozen:
            raise DiagRefusal(f"run_cfg != frozen registry cfg: {name}")

        names = list(d["names"])
        dr = np.asarray(d["draws"], float)
        t = np.asarray(d["truth_vec"], float)
        L = int(d["L"])
        if dr.shape[0] != L:
            raise DiagRefusal(f"{name}: draws rows {dr.shape[0]} != stored L {L}")
        tidx = [names.index(f"tau0_z{z}") for z in range(13)]
        recs.append(dict(
            m=int(name[5:9]), L=L, names=names,
            ns=dr[:, names.index("ns")], ns_truth=float(t[names.index("ns")]),
            Ap=dr[:, names.index("Ap")], Ap_truth=float(t[names.index("Ap")]),
            tau0amp=tau0_amp_vec(dr[:, tidx]),
            tau0amp_truth=float(tau0_amp_slope(t[tidx])[0]),
            draws=dr, truth=t, sites_extra=d.get("sites_extra", {}),
            n_div=int(d.get("n_div", 0)),
        ))
    return recs


def _chan(recs, key):
    """Deployed pull/rank vectors for a channel present as a per-draw column."""
    pulls = np.array([pull_of(r[key], r[f"{key}_truth"]) for r in recs])
    ranks = np.array([rank_of(r[key], r[f"{key}_truth"]) for r in recs])
    return pulls, ranks


def check_consistency(recs, gate_json, selfdraw_json, leg=LEG, n_total=N_TOTAL):
    """The binding deployment-consistency conjuncts (prereg section 4). Refuse-on-fail."""
    with open(gate_json) as f:
        legrec = json.load(f)["legs"][leg]
    with open(selfdraw_json) as f:
        sd = json.load(f)

    Ls = [r["L"] for r in recs]
    checks = []
    for key, jkey in (("ns", "ns"), ("Ap", "Ap"), ("tau0amp", "tau0amp")):
        p, rk = _chan(recs, key)
        checks += [
            (f"{jkey} pull mean", float(p.mean()), float(legrec["pulls"][jkey]["mean"])),
            (f"{jkey} pull sd", float(p.std(ddof=1)), float(legrec["pulls"][jkey]["std"])),
            (f"{jkey} rank KS p", rank_ks_p(rk),
             float(legrec["rank_uniformity"][jkey]["ks_p"])),
        ]
    checks.append(("finite-L null sd", finite_L_null_sd(Ls),
                   float(legrec["finite_L_null"]["sd"])))

    for label, got, want in checks:
        if not np.isclose(got, want, rtol=RTOL, atol=0.0):
            raise DiagRefusal(f"deployment-consistency FAIL on {label}: recomputed {got!r} "
                              f"vs committed {want!r} (rtol {RTOL})")

    if int(sd.get("n", -1)) != n_total or sd.get("survey") != leg \
            or sd.get("conjuncts_ok") is not True or sd.get("health_ok") is not True:
        raise DiagRefusal(
            f"selfdraw anchor is not the passed n={n_total} {leg} record "
            f"(n {sd.get('n')!r}, survey {sd.get('survey')!r}, "
            f"conjuncts_ok {sd.get('conjuncts_ok')!r}, health_ok {sd.get('health_ok')!r})")
    if float(np.median(Ls)) != float(sd["L_median"]) or \
            [min(Ls), max(Ls)] != [int(x) for x in sd["L_range"]]:
        raise DiagRefusal("L census != selfdraw JSON median/range")
    if int(legrec["div_total"]) != sum(r["n_div"] for r in recs):
        raise DiagRefusal("divergence total != committed gate JSON")

    return {label: got for label, got, _ in checks}


# --- P2 core: scale / tail / coverage (VERBATIM reuse of the frozen KS suite) ---------

def discrete_null_coverage(Ls, lo, hi):
    """Exact null occupancy of the INCLUSIVE band for the discrete rank r = k/L
    (k uniform on 0..L), averaged over the arm's own L census. Uses only fit metadata."""
    vals = []
    for L in Ls:
        k = np.arange(0, int(L) + 1)
        vals.append(np.mean((k / L >= lo) & (k / L <= hi)))
    return float(np.mean(vals))


def coverage(ranks, Ls=None):
    from scipy.stats import binomtest
    out = {}
    r = np.asarray(ranks, float)
    for q in LEVELS:
        lo, hi = BAND_EDGES[q]
        k = int(np.sum((r >= lo) & (r <= hi)))
        ci = binomtest(k, len(r)).proportion_ci(confidence_level=0.95, method="exact")
        row = dict(covered=k, n=len(r), fraction=k / len(r), nominal=q,
                   diff=k / len(r) - q, ci95=[float(ci.low), float(ci.high)])
        if Ls is not None:
            row["discrete_null_expectation"] = discrete_null_coverage(Ls, lo, hi)
        out[f"{q:.2f}"] = row
    return out


def winsorized_sd(x, k=WINSOR_K):
    """Explicit symmetric winsorization, k per tail (k = ceil(0.05 n))."""
    s = np.sort(np.asarray(x, float))
    w = s.copy()
    w[:k] = s[k]
    w[-k:] = s[-(k + 1)]
    return float(np.std(w, ddof=1))


def normal_references(n=N_TOTAL, k=WINSOR_K, seed=NORMREF_SEED, nsim=NORMREF_NSIM):
    """Seeded finite-sample NORMAL references for the scale/tail ratios, regenerated at
    THIS arm's (n, k). E[winsorized/sd] is NOT 1 -- winsorizing shrinks a normal sample's
    sd by construction -- and the n=96/k=5 constant 0.907 must never be carried over.
    Synthetic only: no real-data contact."""
    from scipy.stats import median_abs_deviation
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nsim, n))
    sd = x.std(ddof=1, axis=1)
    s = np.sort(x, axis=1)
    w = s.copy()
    w[:, :k] = s[:, [k]]
    w[:, -k:] = s[:, [-(k + 1)]]
    wins = w.std(ddof=1, axis=1)
    mad = median_abs_deviation(x, scale="normal", axis=1)
    xc = x - x.mean(axis=1, keepdims=True)
    sq = np.sort(xc ** 2, axis=1)[:, ::-1]
    ss = sq.sum(axis=1)
    return dict(winsorized_over_sd=float(np.mean(wins / sd)),
                sd_over_mad=float(np.mean(sd / mad)),
                variance_contrib_topk={str(kk): float(np.mean(sq[:, :kk].sum(axis=1) / ss))
                                       for kk in TOP_K},
                n=n, k=k, seed=seed, nsim=nsim)


def scales_and_tail(pulls, n=N_TOTAL, k=WINSOR_K, normref_nsim=NORMREF_NSIM):
    from scipy.stats import median_abs_deviation
    x = np.asarray(pulls, float)
    nn = len(x)
    sd = float(np.std(x, ddof=1))
    xc = x - x.mean()
    ss = float(np.sum(xc ** 2))
    order = np.argsort(-np.abs(xc))
    contrib = {str(kk): float(np.sum(xc[order[:kk]] ** 2) / ss) for kk in TOP_K}
    loo = np.array([np.std(np.delete(x, i), ddof=1) for i in range(nn)])
    top5 = [dict(i=int(i), pull=float(x[i])) for i in np.argsort(-np.abs(x))[:5]]
    mad = float(median_abs_deviation(x, scale="normal"))
    wins = winsorized_sd(x, k=k)
    return dict(
        sd=sd, mean=float(x.mean()), median=float(np.median(x)), mad_scale=mad,
        winsorized_sd=wins, winsor_k=k,
        quantiles={f"{q:g}": float(np.quantile(x, q)) for q in QUANTS},
        top5_abs_pulls=top5, variance_contrib_topk=contrib,
        loo_sd_min=float(loo.min()), loo_sd_max=float(loo.max()),
        loo_sd_max_abs_change=float(np.max(np.abs(loo - sd))),
        loo_sd_all=[float(v) for v in loo],
        sd_over_mad=sd / mad, winsorized_over_sd=wins / sd,
        normal_references=normal_references(n=n, k=k, nsim=normref_nsim))


def loo_tables(x, y):
    """EXACT leave-one-out tables for a realization's paired draws (x=ns, y=tau0_amp).

    For every draw index j, treats draw j as the pseudo-truth and the remaining L-1 draws
    as the reference set, and returns the deployed statistics computed on that LOO set.
    Closed-form updates (no O(L^2) loop) make the whole null cheap and arithmetically
    identical to deleting the row.

    Under the SBC null the pseudo-truth IS a posterior draw, so these arrays ARE the exact
    finite-L null for this realization; a null replicate is one uniformly chosen j.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    L = x.size
    if y.size != L:
        raise DiagRefusal("loo_tables: paired length mismatch")
    if L < 5:
        raise DiagRefusal(f"loo_tables: L={L} too small for a LOO covariance")
    n = L - 1.0

    Sx, Sy = x.sum(), y.sum()
    Sxx, Syy, Sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()

    mx = (Sx - x) / n
    my = (Sy - y) / n
    cxx = (Sxx - x * x - n * mx * mx) / (n - 1.0)
    cyy = (Syy - y * y - n * my * my) / (n - 1.0)
    cxy = (Sxy - x * y - n * mx * my) / (n - 1.0)
    cxx = np.clip(cxx, 1e-300, None)
    cyy = np.clip(cyy, 1e-300, None)

    sdx, sdy = np.sqrt(cxx), np.sqrt(cyy)
    pull_x = (mx - x) / sdx
    pull_y = (my - y) / sdy

    # LOO rank of draw j among the other L-1 draws (exactly uniform on {0..L-2}/(L-1)
    # under the null when there are no ties).
    rank_x = (np.argsort(np.argsort(x, kind="stable"), kind="stable")).astype(float) / n
    rank_y = (np.argsort(np.argsort(y, kind="stable"), kind="stable")).astype(float) / n

    corr = np.clip(cxy / (sdx * sdy), -1.0, 1.0)
    det = np.clip(cxx * cyy - cxy * cxy, 1e-300, None)

    # Eigen-decomposition of the 2x2 LOO covariance, closed form.
    tr = cxx + cyy
    disc = np.sqrt(np.clip(tr * tr - 4.0 * det, 0.0, None))
    lam1 = 0.5 * (tr + disc)          # major
    lam2 = np.clip(0.5 * (tr - disc), 1e-300, None)   # minor
    # Major-axis eigenvector (cxy, lam1 - cxx), normalized; falls back to (1,0) if isotropic.
    v1x, v1y = cxy, lam1 - cxx
    nrm = np.sqrt(v1x * v1x + v1y * v1y)
    iso = nrm < 1e-12 * np.sqrt(np.abs(lam1) + 1e-300)
    v1x = np.where(iso, 1.0, v1x / np.where(nrm > 0, nrm, 1.0))
    v1y = np.where(iso, 0.0, v1y / np.where(nrm > 0, nrm, 1.0))

    dx, dy = mx - x, my - y                     # truth displacement (mean - truth)
    d_par = (dx * v1x + dy * v1y) / np.sqrt(lam1)
    d_perp = (-dx * v1y + dy * v1x) / np.sqrt(lam2)
    D2 = d_par ** 2 + d_perp ** 2

    return dict(pull_x=pull_x, pull_y=pull_y, rank_x=rank_x, rank_y=rank_y,
                corr=corr, area=np.pi * np.sqrt(det), lam1=lam1, lam2=lam2,
                sdx=sdx, sdy=sdy, d_par=d_par, d_perp=d_perp, D2=D2,
                v1x=v1x, v1y=v1y, isotropic=iso, L=L)


def ks_influence(ranks, alpha=0.05):
    """P1 influence block (prereg section 8 / PI #13 section 10).

    INTERPRETATION BOUNDARY (binding): the leave-one-out p-values here are INFLUENCE
    DIAGNOSTICS ONLY. They are not n separate hypothesis tests, must not be counted as
    such, and must never be used to label a realization invalid. No realization may be
    removed, down-weighted or excluded from the formal result. `min_omission_greedy` is a
    DESCRIPTIVE fragility statement about the KS statistic, never a proposal.
    """
    from scipy import stats as _st
    r = np.asarray(ranks, float)
    n = r.size
    ks = _st.kstest(r, "uniform")
    s = np.sort(r)
    i = np.arange(1, n + 1)
    d_plus = i / n - s
    d_minus = s - (i - 1) / n
    per = np.maximum(d_plus, d_minus)                 # per-order-statistic contribution
    order = np.argsort(r)
    contrib = np.empty(n)
    contrib[order] = per                              # back to realization order

    loo_stat, loo_p = np.empty(n), np.empty(n)
    for j in range(n):
        k = _st.kstest(np.delete(r, j), "uniform")
        loo_stat[j], loo_p[j] = k.statistic, k.pvalue

    # Greedy (NOT exhaustive) minimum-omission count; an UPPER BOUND, descriptive only.
    keep = list(range(n))
    removed, cur_p = 0, float(ks.pvalue)
    while cur_p <= alpha and len(keep) > 5:
        best_j, best_p = None, cur_p
        for j in list(keep):
            p = _st.kstest(np.delete(r[keep], keep.index(j)), "uniform").pvalue
            if p > best_p:
                best_p, best_j = p, j
        if best_j is None:
            break
        keep.remove(best_j)
        removed += 1
        cur_p = best_p

    return dict(
        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), n=int(n),
        mean_rank=float(r.mean()), median_rank=float(np.median(r)),
        empirical_cdf=[[float(v), float((k + 1) / n)] for k, v in enumerate(s)],
        per_realization_contrib=[float(v) for v in contrib],
        loo_ks_stat=[float(v) for v in loo_stat],
        loo_ks_p=[float(v) for v in loo_p],
        loo_p_max=float(loo_p.max()), loo_p_min=float(loo_p.min()),
        most_influential=int(np.argmax(loo_p)),
        min_omission_greedy=int(removed) if cur_p > alpha else None,
        min_omission_is_greedy_upper_bound=True,
        min_omission_reached_p=float(cur_p),
        interpretation="LOO p-values are influence diagnostics ONLY -- not n tests, never "
                       "grounds to exclude a realization; min_omission is descriptive.")


def tail_occupancy(ranks, edge=0.1):
    r = np.asarray(ranks, float)
    n = r.size
    lo = int(np.sum(r < edge))
    hi = int(np.sum(r > 1.0 - edge))
    at0 = int(np.sum(r <= 0.0))
    at1 = int(np.sum(r >= 1.0))
    return dict(edge=edge, n=int(n), lower=lo, upper=hi, both=lo + hi,
                lower_frac=lo / n, upper_frac=hi / n, both_frac=(lo + hi) / n,
                asymmetry=(lo - hi) / n, at_boundary_0=at0, at_boundary_1=at1)


def shape_stats(ranks):
    """The frozen P1 shape family S1-S4 (prereg section 8). Values only; significance is
    assigned against the section-5 null by the caller, with Holm inside the P1 family."""
    r = np.asarray(ranks, float)
    n = r.size
    lo = float(np.sum(r < 0.1))
    hi = float(np.sum(r > 0.9))
    return dict(
        S1_mean_minus_half=float(r.mean() - 0.5),
        S2_var_minus_uniform=float(r.var(ddof=1) - 1.0 / 12.0),
        S3_tail_asymmetry=float((lo - hi) / n),
        S4_both_tail_occupancy=float(lo + hi))


def null_replicates(tables, B=B_NULL, seed=A2C_NULL_SEED):
    """Section 5 primary reference: B population-level replicates. Each replicate picks ONE
    draw index per realization (the same index for ns and tau0_amp, preserving the realized
    joint geometry) and gathers the per-realization LOO statistics into a synthetic
    population of the same size and the same L census as the real arm."""
    rng = np.random.default_rng(seed)
    keys = ("pull_x", "pull_y", "rank_x", "rank_y", "corr", "area",
            "d_par", "d_perp", "D2", "sdx", "sdy")
    n = len(tables)
    out = {k: np.empty((B, n)) for k in keys}
    for i, t in enumerate(tables):
        idx = rng.integers(0, t["L"], size=B)
        for k in keys:
            out[k][:, i] = np.asarray(t[k])[idx]
    return out


def calibrated_p(observed, null_samples, two_sided=True):
    """Calibrated p-value from the section-5 null with the standard (r+1)/(B+1) estimator,
    so a p-value is never reported as exactly zero."""
    s = np.asarray(null_samples, float)
    B = s.size
    if two_sided:
        c = float(np.sum(np.abs(s - s.mean()) >= abs(observed - s.mean())))
    else:
        c = float(np.sum(s >= observed))
    return dict(p=(c + 1.0) / (B + 1.0), B=int(B),
                null_mean=float(s.mean()), null_sd=float(s.std(ddof=1)),
                null_q025=float(np.quantile(s, 0.025)),
                null_q975=float(np.quantile(s, 0.975)),
                observed=float(observed), resolution=1.0 / (B + 1.0))


def rank_implied_z_scale(ranks, L_list):
    """The rank-implied z-score scale (physics review SHOULD-FIX 3): sd of Phi^-1(rank).

    DECISIVE for separating a CALIBRATION defect from a SUMMARY artefact. The Gaussian pull
    (mean - truth)/sd is only a calibrated z-score when the posterior is Gaussian. If the
    posterior is skewed or truncated (n_s is sampled under a HARD BOX on theta_unit, so
    prior-drawn truths land near the boundary a fixed fraction of the time), the pull sd can
    exceed 1 while the RANKS stay uniform. sd[Phi^-1(rank)] ~ 1 alongside pull sd >> 1 is
    the signature of a summary artefact, not of an over-concentrated posterior.

    Ranks are continuity-corrected to (k + 0.5)/(L + 1) so the transform stays finite at the
    empirical extremes; the correction uses only fit metadata (L).
    """
    from scipy.stats import norm
    r = np.asarray(ranks, float)
    L = np.asarray(list(L_list), float)
    if r.size != L.size:
        raise DiagRefusal("rank_implied_z_scale: rank/L length mismatch")
    k = np.rint(r * L)
    rr = (k + 0.5) / (L + 1.0)
    z = norm.ppf(rr)
    return dict(z_sd=float(np.std(z, ddof=1)), z_mean=float(np.mean(z)),
                z_median=float(np.median(z)), n=int(r.size),
                continuity="(k + 0.5)/(L + 1)",
                note="sd[Phi^-1(rank)] ~ 1 with a Gaussian pull sd >> 1 indicates a "
                     "NON-GAUSSIAN POSTERIOR SUMMARY artefact, not over-concentration.")


def anisotropy_contrast(d_par, d_perp):
    """Physics review SHOULD-FIX 7. mean|d_perp| alone cannot separate 'narrow across the
    degeneracy' from 'narrow overall', because both components are Mahalanobis-normalised by
    the same covariance. This contrast is scale-free in the overall width and isolates
    ONE-DIRECTIONAL over-concentration (PI #13 section 16)."""
    a = np.asarray(d_par, float) ** 2
    b = np.asarray(d_perp, float) ** 2
    den = a + b
    good = den > 0
    c = np.zeros_like(den)
    c[good] = (b[good] - a[good]) / den[good]
    return dict(mean=float(c[good].mean()), median=float(np.median(c[good])),
                n=int(good.sum()),
                note="+1 = displacement entirely PERPENDICULAR to the degeneracy axis; "
                     "-1 = entirely PARALLEL; 0 = isotropic. Invariant to overall width.")


def permutation_null(a, b, stat, B=B_NULL, seed=A2C_NULL_SEED):
    """Permutation null over realizations for associations between TWO DRAWS-ONLY
    quantities (physics review MUST-FIX 4).

    The section-5 LOO null is DEGENERATE for statistics that do not involve the truth: it
    resamples the pseudo-truth but leaves the draw matrix essentially unchanged, so a
    draws-only statistic has a null that is a point mass at the observed value. Associations
    such as (candidate posterior mean) vs (posterior correlation / ellipse area) must
    therefore be calibrated by permuting the pairing ACROSS realizations instead.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.size != y.size:
        raise DiagRefusal("permutation_null: length mismatch")
    rng = np.random.default_rng(seed)
    obs = float(stat(x, y))
    null = np.array([float(stat(x, rng.permutation(y))) for _ in range(B)])
    return dict(observed=obs, **{k: v for k, v in calibrated_p(obs, null).items()
                                 if k != "observed"})


def tau_eff_rung_coefficients(z_grid=Z_TAU0, pivot=4.0):
    """c_i = ln((1+z_i)/pivot): the EXACT dtau0 admixture in ln tau_eff(z_i).

    Because the deployed ladder is tau0(z) = amp * ((1+z)/4)^dtau0 * Kim(z), ln tau_eff(z_i)
    = ln(amp) + c_i*dtau0 + const EXACTLY (verified to machine precision). The deployed
    'tau0_amp' summary is therefore ln(amp) + c_bar*dtau0 with c_bar = ln((1+zbar)/4) at the
    log-mean redshift of the 13-rung UNION grid -- a rotated coordinate in the mean-flux
    plane, not the sampled amplitude.
    """
    z = np.asarray(z_grid, float)
    lx = np.log(1.0 + z)
    zbar = float(np.exp(lx.mean()) - 1.0)
    return dict(z=[float(v) for v in z],
                c=[float(v) for v in np.log((1.0 + z) / pivot)],
                zbar=zbar, c_bar=float(np.log((1.0 + zbar) / pivot)), pivot=pivot)


def r_scale_bootstrap(pulls, Ls, B=BOOT_B, seed=BOOT_SEED):
    """Finite-L-adjusted scale ratio with a PAIRED bootstrap over (pull, L) pairs, so the
    numerator and the finite-L denominator are resampled together (the frozen KS form)."""
    x = np.asarray(pulls, float)
    L = np.asarray(Ls, float)
    if x.size != L.size:
        raise DiagRefusal("r_scale_bootstrap: pull/L length mismatch")
    point = float(np.std(x, ddof=1) / finite_L_null_sd(L))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    vals = []
    for row in idx:
        xb, Lb = x[row], L[row]
        if np.any(Lb <= 3):
            continue
        vals.append(np.std(xb, ddof=1) / finite_L_null_sd(Lb))
    v = np.sort(np.asarray(vals, float))
    return dict(point=point, B=int(v.size), seed=seed,
                ci68=[float(np.quantile(v, 0.16)), float(np.quantile(v, 0.84))],
                ci95=[float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))])


# =====================================================================================
# v2 AMENDMENT MACHINERY (prereg v2; all referee MUST-FIX)
# =====================================================================================

def exact_rank_null(Ls, B, seed):
    """EXACT discrete rank null (prereg v2 B.2). Under the SBC null the deployed rank
    k/L has k ~ Uniform{0..L}; only the L CENSUS enters, not the realized posteriors.
    This is a draw from the exact null, not an approximation."""
    rng = np.random.default_rng(seed)
    L = np.asarray(list(Ls), int)
    return np.stack([rng.integers(0, Lm + 1, size=B) / Lm for Lm in L], axis=1)


def check_ties(recs, cols=("ns", "Ap", "tau0amp")):
    """Prereg v2 B.4: duplicate draw rows break the LOO bijection and bias the strict
    rank downward. The deployed statistic is immune (truth is never a draw); the null is
    not. Refuse-on-fail."""
    for r in recs:
        M = np.column_stack([np.asarray(r[c], float) for c in cols])
        if np.unique(M, axis=0).shape[0] != M.shape[0]:
            raise DiagRefusal(f"mock {r['m']}: duplicate draw rows in {cols} -- the LOO "
                              f"bijection is broken (prereg v2 B.4)")
    return True


def check_sites_alignment(recs, keys):
    """Prereg v2 B.5: every sites_extra array must have exactly L rows so the shared
    pseudo-truth index j is applicable. Refuse-on-fail."""
    for r in recs:
        for k in keys:
            se = r["sites_extra"].get(k)
            if se is None:
                raise DiagRefusal(f"mock {r['m']}: sites_extra['{k}'] absent")
            d = np.asarray(se["draws"], float)
            if d.shape != (r["L"],):
                raise DiagRefusal(f"mock {r['m']}: sites_extra['{k}'] has {d.shape} rows, "
                                  f"expected ({r['L']},) -- row misalignment")
            if not np.all(np.isfinite(d)) or not np.isfinite(float(se["truth"])):
                raise DiagRefusal(f"mock {r['m']}: sites_extra['{k}'] non-finite")
    return True


def site_pull_rank(recs, key):
    """Deployed pull/rank for a sites_extra channel (sampled coordinates)."""
    p, rk = [], []
    for r in recs:
        d = np.asarray(r["sites_extra"][key]["draws"], float)
        t = float(r["sites_extra"][key]["truth"])
        p.append(pull_of(d, t))
        rk.append(rank_of(d, t))
    return np.asarray(p), np.asarray(rk)


def identity_check(recs, rtol=1e-9):
    """Prereg v2 P1-I (refuse-on-fail). The deployed ladder-regressed amp/slope MUST equal
    the closed forms in the SAMPLED coordinates. A mismatch IS the derived-summary artefact."""
    co = tau_eff_rung_coefficients()
    zbar, z = co["zbar"], np.asarray(co["z"])
    worst_a = worst_s = 0.0
    for r in recs:
        amp = np.asarray(r["sites_extra"]["tau0_amp"]["draws"], float)
        dt = np.asarray(r["sites_extra"]["dtau0"]["draws"], float)
        want_a = amp * ((1 + zbar) / 4.0) ** dt * 0.0023 * (1 + zbar) ** 3.65
        got_a = np.asarray(r["tau0amp"], float)
        y = np.log(np.clip(np.asarray(r["draws"], float)[:, [r["names"].index(f"tau0_z{i}")
                                                             for i in range(13)]], 1e-8, None))
        lxc = np.log(1 + z) - np.log(1 + z).mean()
        got_s = (y - y.mean(axis=1, keepdims=True)) @ lxc / np.sum(lxc ** 2)
        worst_a = max(worst_a, float(np.max(np.abs(got_a / want_a - 1.0))))
        worst_s = max(worst_s, float(np.max(np.abs(got_s - (dt + 3.65)))))
    ok = (worst_a < rtol) and (worst_s < 1e-7)
    return dict(max_rel_err_amp=worst_a, max_abs_err_slope=worst_s, rtol=rtol,
                identity_holds=bool(ok), zbar=zbar, c_bar=co["c_bar"],
                note="amp_reg == tau0_amp*((1+zbar)/4)^dtau0*2.3e-3*(1+zbar)^3.65 and "
                     "slope_reg == dtau0 + 3.65. Holding EXCLUDES the derived-summary "
                     "curvature/projection artefact class a priori.")


def per_rung_scan(recs, z_hi_desi=4.2):
    """Prereg v2 P1-R: rank-KS p, mean rank and S1/S2 per tau_eff(z) rung. rank of
    tau_eff(z_i) is the rank of ln(amp) + c_i*dtau0, so the scan reads WHICH DIRECTION in
    the mean-flux plane is miscalibrated, in the physically interpretable coordinate."""
    co = tau_eff_rung_coefficients()
    z, c = np.asarray(co["z"]), np.asarray(co["c"])
    rows = []
    for i in range(13):
        col = f"tau0_z{i}"
        ranks = np.array([rank_of(np.asarray(r["draws"], float)[:, r["names"].index(col)],
                                  float(np.asarray(r["truth"], float)[r["names"].index(col)]))
                          for r in recs])
        pulls = np.array([pull_of(np.asarray(r["draws"], float)[:, r["names"].index(col)],
                                  float(np.asarray(r["truth"], float)[r["names"].index(col)]))
                          for r in recs])
        sh = shape_stats(ranks)
        rows.append(dict(rung=i, z=float(z[i]), c=float(c[i]),
                         desi_constrained=bool(z[i] <= z_hi_desi),
                         rank_ks_p=rank_ks_p(ranks), mean_rank=float(ranks.mean()),
                         pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
                         S1=sh["S1_mean_minus_half"], S2=sh["S2_var_minus_uniform"]))
    return dict(rungs=rows, zbar=co["zbar"], c_bar=co["c_bar"],
                n_desi_constrained=int(sum(r["desi_constrained"] for r in rows)),
                note="DESCRIPTIVE SCAN, not 13 confirmatory tests. Pre-declared readings: "
                     "HCD-mediated degrades toward HIGH z; metal-mediated toward LOW z; "
                     "z-flat is amplitude-like and should have shown in A_p.")


def null_relative_material(obs, null_samples, thresh=0.30, k_sd=2.0):
    """Prereg v2 D.2: materiality as a deviation from the NULL, not from zero. v1's
    |rho|>=0.30 fired on 92.1% of PERFECTLY HEALTHY arms because the healthy null is
    centred near -0.47, not 0."""
    s = np.asarray(null_samples, float)
    med, sd = float(np.median(s)), float(np.std(s, ddof=1))
    dev = float(obs) - med
    return dict(observed=float(obs), null_median=med, null_sd=sd, deviation=dev,
                material=bool(abs(dev) >= thresh and abs(dev) >= k_sd * sd),
                thresh=thresh, k_sd=k_sd)


def mc_se(p, B):
    return float(np.sqrt(max(p * (1.0 - p), 1e-12) / B))


def partial_corr(x, y, z):
    """Partial Spearman of x,y controlling z (prereg v2 D.3: the frozen test of the PI's
    own premise, since the committed record reports partial corr(n_s, tau0 | A_p) = -0.047)."""
    from scipy.stats import spearmanr, norm
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rz = np.argsort(np.argsort(np.asarray(z, float))).astype(float)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(spearmanr(resid(rx, rz), resid(ry, rz)).statistic)


def autocorr_lag1(x):
    a = np.asarray(x, float)
    a = a - a.mean()
    d = float(np.dot(a, a))
    return float(np.dot(a[:-1], a[1:]) / d) if d > 0 else 0.0


# =====================================================================================
# ANALYSIS DRIVER
# =====================================================================================

TIER3_COSMO = ["herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]
TIER3_SITES = ["f_res_slope", "s_lls", "s_subdla", "s_dla"]
METAL_NODES = ["f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1",
               "k_SiIII_DESI_z0", "k_SiIII_DESI_z1", "k_SiII_DESI_z0", "k_SiII_DESI_z1"]
ALPHAS = ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _spear(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _pc1(mat):
    """Leading principal component score of a (n, k) pull block (prereg v2 F.1/F.3)."""
    X = np.asarray(mat, float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[0], float(S[0] ** 2 / np.sum(S ** 2))


def run_analysis(recs, B_null=50_000, seed=A2C_NULL_SEED):
    from scipy.stats import norm, skew
    out = {}
    Ls = [r["L"] for r in recs]
    n = len(recs)

    # --- integrity additions (refuse-on-fail) ---
    check_ties(recs)
    check_sites_alignment(recs, ["tau0_amp", "dtau0", "f_res_amp"] + METAL_NODES)
    out["identity_check"] = identity_check(recs)
    if not out["identity_check"]["identity_holds"]:
        raise DiagRefusal("P1-I identity FAILED: the deployed tau0_amp is not the closed "
                          "form in sampled coordinates -- THIS IS the derived-summary "
                          f"artefact ({out['identity_check']})")

    # --- deployed channels ---
    ns_p, ns_r = _chan(recs, "ns")
    ap_p, ap_r = _chan(recs, "Ap")
    t0_p, t0_r = _chan(recs, "tau0amp")

    # --- LOO tables (truth-involving null) ---
    tabs_nt = [loo_tables(r["ns"], r["tau0amp"]) for r in recs]
    tabs_na = [loo_tables(r["ns"], r["Ap"]) for r in recs]
    tabs_at = [loo_tables(r["Ap"], r["tau0amp"]) for r in recs]
    rep = null_replicates(tabs_nt, B=B_null, seed=seed)
    rep_na = null_replicates(tabs_na, B=B_null, seed=seed + 1)
    rep_at = null_replicates(tabs_at, B=B_null, seed=seed + 2)

    # --- exact discrete rank null (rank-only statistics) ---
    rnull = exact_rank_null(Ls, B_null, seed + 10)

    # =============================== P1 ===============================
    p1 = dict(deployed_rank_ks_p=rank_ks_p(t0_r),
              influence=ks_influence(t0_r),
              tails=tail_occupancy(t0_r),
              shape=shape_stats(t0_r))
    shp_null = {k: np.array([shape_stats(rnull[b])[k] for b in range(B_null)])
                for k in p1["shape"]}
    p1["shape_calibrated"] = {}
    for k, v in p1["shape"].items():
        cp = calibrated_p(v, shp_null[k])
        cp["mc_se"] = mc_se(cp["p"], len(shp_null[k]))
        p1["shape_calibrated"][k] = cp
    keys4 = list(p1["shape"].keys())
    p1["holm_within_P1"] = dict(zip(keys4,
                                    holm([p1["shape_calibrated"][k]["p"] for k in keys4])))
    # fragility calibration (prereg v2 C.3)
    # DISCLOSED CAP: ks_influence is O(n^2) per replicate (48 LOO KS tests), so the
    # fragility null runs at 400 replicates, not B_null. Reported with the result.
    mo_null = []
    for b in range(min(B_null, 400)):
        mo_null.append(ks_influence(rnull[b])["min_omission_greedy"])
    p1["fragility_null"] = dict(
        min_omission_null_median=float(np.median([x for x in mo_null if x is not None])
                                       if any(x is not None for x in mo_null) else np.nan),
        frac_null_needing_ge1=float(np.mean([(x or 0) >= 1 for x in mo_null])),
        n_null=len(mo_null),
        note="At a BOUNDARY-level KS p the min-omission count is ~1 BY DEFINITION; this "
             "calibration shows how often a HEALTHY arm needs the same omission.")
    # sampled-site control + per-rung scan + L confounder
    amp_p, amp_r = site_pull_rank(recs, "tau0_amp")
    dt_p, dt_r = site_pull_rank(recs, "dtau0")
    p1["sampled_sites"] = {
        "tau0_amp_sampled": dict(pull_mean=float(amp_p.mean()), pull_sd=float(amp_p.std(ddof=1)),
                                 rank_ks_p=rank_ks_p(amp_r), mean_rank=float(amp_r.mean()),
                                 **shape_stats(amp_r)),
        "dtau0_sampled": dict(pull_mean=float(dt_p.mean()), pull_sd=float(dt_p.std(ddof=1)),
                              rank_ks_p=rank_ks_p(dt_r), mean_rank=float(dt_r.mean()),
                              **shape_stats(dt_r)),
        "deployed_tau0amp": dict(pull_mean=float(t0_p.mean()), pull_sd=float(t0_p.std(ddof=1)),
                                 rank_ks_p=rank_ks_p(t0_r), mean_rank=float(t0_r.mean())),
    }
    p1["per_rung"] = per_rung_scan(recs)
    p1["ln_coordinate_pull"] = dict(
        mean=float(np.mean([(np.log(r["tau0amp"]).mean() - np.log(r["tau0amp_truth"]))
                            / np.log(r["tau0amp"]).std(ddof=1) for r in recs])),
        note="Jensen check: pull in ln coordinates (ranks unaffected, exp monotone).")
    p1["L_confounder"] = dict(
        spearman_rankextreme_vs_L=_spear(2 * np.abs(t0_r - 0.5), Ls),
        spearman_abs_pull_ns_vs_L=_spear(np.abs(ns_p), Ls),
        L_floor_note="closure_sbc.L_FLOOR = 99; realizations below it: "
                     f"{int(np.sum(np.asarray(Ls) < 99))}")
    out["P1"] = p1

    # =============================== P2 ===============================
    p2 = dict(scales=scales_and_tail(ns_p, n=n, k=WINSOR_K),
              coverage=coverage(ns_r, Ls=Ls),
              r_scale=r_scale_bootstrap(ns_p, Ls),
              finite_L_null=finite_L_null_sd(Ls))
    p2["rank_implied_z"] = rank_implied_z_scale(ns_r, Ls)
    p2["summary_artefact_contrast"] = dict(
        gaussian_pull_sd=float(ns_p.std(ddof=1)),
        rank_implied_z_sd=p2["rank_implied_z"]["z_sd"],
        ratio=float(ns_p.std(ddof=1) / p2["rank_implied_z"]["z_sd"]),
        note="rank-implied z-sd ~1 with a Gaussian pull sd >> 1 indicates a NON-GAUSSIAN "
             "SUMMARY artefact rather than posterior over-concentration.")
    sd_null = rep["pull_x"].std(axis=1, ddof=1)
    cp = calibrated_p(float(ns_p.std(ddof=1)), sd_null, two_sided=False)
    cp["mc_se"] = mc_se(cp["p"], sd_null.size)
    p2["headline_sd_calibrated_p"] = cp
    p2["posterior_skew"] = dict(
        ns_median_skew=float(np.median([skew(r["ns"]) for r in recs])),
        Ap_median_skew=float(np.median([skew(r["Ap"]) for r in recs])),
        tau0_median_skew=float(np.median([skew(r["tau0amp"]) for r in recs])))
    # KS five-variable family restored (V1..V5)
    postsd = np.array([float(np.std(r["ns"], ddof=1)) for r in recs])
    nullint = np.sqrt((1 + 1 / np.asarray(Ls, float)) * (np.asarray(Ls, float) - 1)
                      / (np.asarray(Ls, float) - 3))
    fres_t = np.array([abs(float(r["sites_extra"]["f_res_amp"]["truth"])) for r in recs])
    V = [("V1 truth_ns", _spear(np.array([r["ns_truth"] for r in recs]), np.abs(ns_p))),
         ("V2 post_sd_ns", _spear(postsd, np.abs(ns_p))),
         ("V3 L", _spear(np.asarray(Ls, float), np.abs(ns_p) / nullint)),
         ("V4 |f_res truth|", _spear(fres_t, np.abs(ns_p))),
         ("V5 |tau0amp pull|", _spear(np.abs(t0_p), np.abs(ns_p)))]
    pv = []
    for (lab, rho), src in zip(V, [np.array([r["ns_truth"] for r in recs]), postsd,
                                   np.asarray(Ls, float), fres_t, np.abs(t0_p)]):
        tgt = np.abs(ns_p) / nullint if lab.startswith("V3") else np.abs(ns_p)
        pn = permutation_null(src, tgt, _spear, B=B_null, seed=seed + 20)
        pn["mc_se"] = mc_se(pn["p"], B_null)
        pv.append(dict(variable=lab, rho=rho, **{k: pn[k] for k in
                                                 ("p", "null_mean", "null_sd", "mc_se")}))
    flags = holm([r["p"] for r in pv])
    for r_, f_ in zip(pv, flags):
        r_["holm_flagged"] = bool(f_)
    p2["KS_five_variable_family"] = pv
    out["P2"] = p2

    # =============================== P3 ===============================
    # observed geometry with the REAL truths
    def real_geom(xkey, ykey, xt, yt):
        dpar, dperp, D2, rho_m, area, iso = [], [], [], [], [], []
        for r in recs:
            x = np.asarray(r[xkey], float); y = np.asarray(r[ykey], float)
            C = np.cov(np.vstack([x, y]), ddof=1)
            d = np.array([x.mean() - r[xt], y.mean() - r[yt]])
            w, Vv = np.linalg.eigh(C)
            order = np.argsort(-w); w = w[order]; Vv = Vv[:, order]
            dpar.append(float(d @ Vv[:, 0] / np.sqrt(max(w[0], 1e-300))))
            dperp.append(float(d @ Vv[:, 1] / np.sqrt(max(w[1], 1e-300))))
            D2.append(dpar[-1] ** 2 + dperp[-1] ** 2)
            rho_m.append(float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
            area.append(float(np.pi * np.sqrt(max(np.linalg.det(C), 1e-300))))
            iso.append(bool(abs(rho_m[-1]) < 2 / np.sqrt(max(r["L"] - 3, 1))))
        return (np.array(dpar), np.array(dperp), np.array(D2), np.array(rho_m),
                np.array(area), np.array(iso))

    def p3_for(name, tabs, replic, xp, yp, yr, xkey, ykey, xt, yt):
        dpar, dperp, D2, rho_m, area, iso = real_geom(xkey, ykey, xt, yt)
        e = 2 * np.abs(yr - 0.5)
        J1o = _spear(np.abs(xp), e)
        J2o = _spear(xp, yp)
        J3o = float(np.log(np.mean(dperp ** 2) / np.mean(dpar ** 2)))
        J1n = np.array([_spear(np.abs(replic["pull_x"][b]), 2 * np.abs(replic["rank_y"][b] - 0.5))
                        for b in range(B_null)])
        J2n = np.array([_spear(replic["pull_x"][b], replic["pull_y"][b])
                        for b in range(B_null)])
        J3n = np.array([float(np.log(np.mean(replic["d_perp"][b] ** 2)
                                     / np.mean(replic["d_par"][b] ** 2)))
                        for b in range(B_null)])
        blk = {}
        for lab, o, nl in (("J1_overlap", J1o, J1n), ("J2_association", J2o, J2n),
                           ("J3p_log_anisotropy", J3o, J3n)):
            cp = calibrated_p(o, nl); cp["mc_se"] = mc_se(cp["p"], nl.size)
            blk[lab] = dict(**cp, materiality=null_relative_material(o, nl))
        blk["mean_D2"] = dict(observed=float(np.mean(D2)),
                              null_mean=float(np.mean(replic["D2"])),
                              note="SCALE limb, largely determined by the disclosed marginals.")
        blk["anisotropy_contrast"] = anisotropy_contrast(dpar, dperp)
        blk["rho_m"] = dict(mean=float(rho_m.mean()), median=float(np.median(rho_m)),
                            sd=float(rho_m.std(ddof=1)),
                            n_sign_unresolved=int(iso.sum()),
                            deconvolved_spread=float(
                                np.var(np.arctanh(np.clip(rho_m, -0.999, 0.999)), ddof=1)
                                - np.mean(1.0 / np.maximum(np.asarray(Ls, float) - 3, 1))),
                            note="DRAWS-ONLY: no calibrated reference (prereg v2 A.3). "
                                 "Spread is Fisher-z noise-deconvolved.")
        blk["area"] = dict(mean=float(area.mean()),
                           note="DRAWS-ONLY: descriptive only, no calibrated reference.")
        return blk

    P3 = {"ns_tau0": p3_for("ns_tau0", tabs_nt, rep, ns_p, t0_p, t0_r,
                            "ns", "tau0amp", "ns_truth", "tau0amp_truth"),
          "ns_Ap_control": p3_for("ns_Ap", tabs_na, rep_na, ns_p, ap_p, ap_r,
                                  "ns", "Ap", "ns_truth", "Ap_truth"),
          "Ap_tau0_control": p3_for("Ap_tau0", tabs_at, rep_at, ap_p, t0_p, t0_r,
                                    "Ap", "tau0amp", "Ap_truth", "tau0amp_truth")}
    # frozen premise test: partial corr(n_s, tau0 | A_p)
    pc_o = partial_corr(ns_p, t0_p, ap_p)
    pc_n = np.array([partial_corr(rep["pull_x"][b], rep["pull_y"][b], rep_na["pull_y"][b])
                     for b in range(B_null)])
    cp = calibrated_p(pc_o, pc_n); cp["mc_se"] = mc_se(cp["p"], pc_n.size)
    P3["premise_partial_corr_ns_tau0_given_Ap"] = dict(
        **cp, materiality=null_relative_material(pc_o, pc_n),
        marginal_spearman=_spear(ns_p, t0_p),
        committed_record_partial="-0.047 (sbc-ns-subdla-coupling; marginal -0.81)",
        note="FROZEN TEST OF THE PI'S OWN PREMISE (prereg v2 A.1/D.3).")
    out["P3"] = P3

    # =============================== M-block ===============================
    from scipy import stats as _st
    ns_t = np.array([r["ns_truth"] for r in recs])
    ap_t = np.array([r["Ap_truth"] for r in recs])
    M = {}
    M["M1_truth_law"] = dict(
        ns_truth_uniform_ks_p=float(_st.kstest(ns_t, "uniform").pvalue),
        Ap_truth_uniform_ks_p=float(_st.kstest(ap_t, "uniform").pvalue),
        ns_truth_range=[float(ns_t.min()), float(ns_t.max())],
        Ap_truth_range=[float(ap_t.min()), float(ap_t.max())],
        note="var_prior = 1/12 confirms Uniform unit priors, so this KS test is EXACT. "
             "The health block certifies only the TEN self-drawn sites; ns/Ap were "
             "unverified. A break here is a generator-vs-likelihood mismatch.")
    ac = {k: [autocorr_lag1(r[k]) for r in recs] for k in ("ns", "Ap", "tau0amp")}
    M["M2_sampler"] = dict(
        lag1_median={k: float(np.median(v)) for k, v in ac.items()},
        lag1_max={k: float(np.max(v)) for k, v in ac.items()},
        thin_step_median=float(np.median([600.0 / r["L"] for r in recs])),
        null_sd_shift_note="Residual autocorrelation at the observed L shifts the null pull "
                           "sd only ~1.010 -> ~1.014, so this class can be EXCLUDED, not "
                           "confirmed. The rank null ABSORBS autocorrelation by construction.")
    m3 = permutation_null(ns_t, np.abs(ns_p), _spear, B=B_null, seed=seed + 30)
    m3["mc_se"] = mc_se(m3["p"], B_null)
    M["M3_prior_edge"] = dict(
        spearman_abs_pull_vs_truth_ns=_spear(ns_t, np.abs(ns_p)), **m3,
        spearman_postsd_vs_truth_ns=_spear(ns_t, postsd),
        committed_prediction="the 2026-06 record states the n_s bias was NOT a prior-edge "
                             "artefact (pull-vs-truth flat) -- this re-tests it",
        note="Proxy for the named 'stop_gradient n_s-edge MF-floor term that n_s-dependently "
             "widens C'. This is the restored KS variable V1.")
    M["loglik_rank_NOT_RUN"] = (
        "ll_true/ll_draws are stored and the Modrak loglik rank is the one omnibus statistic "
        "sensitive to a C_mock != C_like break, but PI #9 Q3 ruled ll_rank_frac_mean "
        "UNINTERPRETABLE on a self-draw arm. NOT RUN; returned as a section-19 question.")
    out["M_mechanism_class"] = M

    # =============================== Tier-2 ===============================
    def pulls_for_sites(keys):
        return np.column_stack([site_pull_rank(recs, k)[0] for k in keys])
    alpha_block = np.column_stack([
        np.array([pull_of(np.asarray(r["draws"], float)[:, r["names"].index(a)],
                          float(np.asarray(r["truth"], float)[r["names"].index(a)]))
                  for r in recs]) for a in ALPHAS])
    hcd_pc1, hcd_var = _pc1(alpha_block)
    metal_pc1, metal_var = _pc1(pulls_for_sites(METAL_NODES))
    fres_p, _ = site_pull_rank(recs, "f_res_amp")
    t2 = {}
    cands = [("dtau0", dt_p), ("HCD_PC1", hcd_pc1), ("metal_PC1", metal_pc1),
             ("f_res_amp", fres_p)]
    t2_tests = []
    for name, v in cands:
        for tgt_name, tgt in (("pull_ns", ns_p), ("pull_tau0amp", t0_p)):
            pn = permutation_null(v, tgt, _spear, B=B_null, seed=seed + 40)
            pn["mc_se"] = mc_se(pn["p"], B_null)
            t2_tests.append(dict(candidate=name, target=tgt_name, rho=_spear(v, tgt), **pn))
    fl = holm([t["p"] for t in t2_tests])
    for t_, f_ in zip(t2_tests, fl):
        t_["holm_flagged"] = bool(f_)
        t_["materiality"] = dict(note="null-relative; see null_mean/null_sd",
                                 deviation=t_["rho"] - t_["null_mean"],
                                 material=bool(abs(t_["rho"] - t_["null_mean"]) >= 0.30
                                               and abs(t_["rho"] - t_["null_mean"])
                                               >= 2 * t_["null_sd"]))
    t2["tests"] = t2_tests
    t2["holm_family_size"] = len(t2_tests)
    t2["hcd_pc1_var_explained"] = hcd_var
    t2["metal_pc1_var_explained"] = metal_var
    t2["frozen_directional_predictions"] = dict(
        subdla_expected=0.73, lls_expected=0.25, ordering="subDLA > LLS",
        observed_subdla=_spear(alpha_block[:, 1], ns_p),
        observed_lls=_spear(alpha_block[:, 0], ns_p),
        observed_dla=_spear(alpha_block[:, 2], ns_p),
        note="Signed ordered replication predictions from the committed record. "
             "+0.73 is above the n=48 MDE ~0.40; +0.25 is NOT.")
    # joint mean-flux 2-D calibration in SAMPLED coordinates
    mf_tabs = [loo_tables(np.asarray(r["sites_extra"]["tau0_amp"]["draws"], float),
                          np.asarray(r["sites_extra"]["dtau0"]["draws"], float))
               for r in recs]
    mf_rep = null_replicates(mf_tabs, B=B_null, seed=seed + 50)
    mfD2 = []
    for r in recs:
        x = np.asarray(r["sites_extra"]["tau0_amp"]["draws"], float)
        y = np.asarray(r["sites_extra"]["dtau0"]["draws"], float)
        C = np.cov(np.vstack([x, y]), ddof=1)
        d = np.array([x.mean() - float(r["sites_extra"]["tau0_amp"]["truth"]),
                      y.mean() - float(r["sites_extra"]["dtau0"]["truth"])])
        mfD2.append(float(d @ np.linalg.inv(C) @ d))
    cp = calibrated_p(float(np.mean(mfD2)), mf_rep["D2"].mean(axis=1), two_sided=False)
    cp["mc_se"] = mc_se(cp["p"], mf_rep["D2"].shape[0])
    t2["meanflux_joint_2d"] = dict(**cp,
                                   note="Mahalanobis D2 in the SAMPLED (tau0_amp, dtau0) "
                                        "coordinates: a rotation in the mean-flux plane need "
                                        "not show in either marginal.")
    out["Tier2"] = t2

    # =============================== Tier-3 ===============================
    t3 = {}
    for nm in TIER3_COSMO:
        j = recs[0]["names"].index(nm)
        p = np.array([pull_of(np.asarray(r["draws"], float)[:, j],
                              float(np.asarray(r["truth"], float)[j])) for r in recs])
        rk = np.array([rank_of(np.asarray(r["draws"], float)[:, j],
                               float(np.asarray(r["truth"], float)[j])) for r in recs])
        t3[nm] = dict(pull_mean=float(p.mean()), pull_sd=float(p.std(ddof=1)),
                      rank_ks_p=rank_ks_p(rk),
                      spearman_vs_pull_ns=_spear(p, ns_p),
                      spearman_vs_pull_tau0=_spear(p, t0_p),
                      flagged=bool(abs(p.mean()) > 0.5 or p.std(ddof=1) > 1.5
                                   or p.std(ddof=1) < 0.6 or rank_ks_p(rk) < 1e-3))
    for nm in TIER3_SITES + METAL_NODES:
        if nm not in recs[0]["sites_extra"]:
            continue
        p, rk = site_pull_rank(recs, nm)
        t3[nm] = dict(pull_mean=float(p.mean()), pull_sd=float(p.std(ddof=1)),
                      rank_ks_p=rank_ks_p(rk),
                      spearman_vs_pull_ns=_spear(p, ns_p),
                      spearman_vs_pull_tau0=_spear(p, t0_p),
                      flagged=bool(abs(p.mean()) > 0.5 or p.std(ddof=1) > 1.5
                                   or p.std(ddof=1) < 0.6 or rank_ks_p(rk) < 1e-3))
    out["Tier3"] = dict(channels=t3, n_flagged=int(sum(v["flagged"] for v in t3.values())),
                        power_disclosure="At n=48 sd has sampling sd ~0.10, so sd>1.5 is "
                                         "+4.9 sigma and |mean|>0.5 is 3.5 sigma; expected "
                                         "false flags across ~19 channels ~0.03. A CLEAN "
                                         "SWEEP IS UNINFORMATIVE; 'everything else is "
                                         "healthy' is BARRED.")

    # =============================== primary Holm family ===============================
    prim = [("S1", p1["shape_calibrated"]["S1_mean_minus_half"]["p"]),
            ("S2", p1["shape_calibrated"]["S2_var_minus_uniform"]["p"]),
            ("S3", p1["shape_calibrated"]["S3_tail_asymmetry"]["p"]),
            ("S4", p1["shape_calibrated"]["S4_both_tail_occupancy"]["p"]),
            ("J1", P3["ns_tau0"]["J1_overlap"]["p"]),
            ("J2", P3["ns_tau0"]["J2_association"]["p"]),
            ("J3p", P3["ns_tau0"]["J3p_log_anisotropy"]["p"]),
            ("premise_partial", P3["premise_partial_corr_ns_tau0_given_Ap"]["p"])]
    fl = holm([p for _, p in prim])
    out["primary_holm_family"] = dict(
        m=len(prim), alpha=HOLM_ALPHA,
        tests={k: dict(p=p, holm_significant=bool(f)) for (k, p), f in zip(prim, fl)},
        realized_fwer_note="ONE family (v1's three families gave FWER 14.3%).")
    out["meta"] = dict(n=n, B_null=B_null, seed=seed, L_median=float(np.median(Ls)),
                       L_range=[int(min(Ls)), int(max(Ls))],
                       winsor_k=WINSOR_K, prereg="v2 amendment 2026-08-07")
    return out
