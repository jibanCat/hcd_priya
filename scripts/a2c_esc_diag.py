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
A2C_NULL_SEED, B_NULL = 20260810, 2000
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
