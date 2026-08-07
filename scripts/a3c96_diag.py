#!/usr/bin/env python3
"""A3c N=96 BOUNDED DIAGNOSTIC (PI #11 sec 4-6; prereg
2026-08-06-A3C96-DIAG-PREREGISTRATION.md). N=96-population-ONLY: reads the 96
committed pkls and committed summary JSONs; performs NO sampling, NO likelihood
evaluation, NO inference change. Descriptive only: nothing here can alter the
Branch-B verdict, the frozen gate, or any certification statement.

DEPLOYMENT-CONSISTENCY CONJUNCTS (binding): the deployed statistics are recomputed
with identical arithmetic and MUST match the committed gate JSON (rel. tol 1e-9) --
n_s pull mean/sd, n_s rank KS-p, tau0amp pull mean/sd, finite-L null sd -- else the
run refuses with no diagnostic output. Population integrity: census 96/96, sha256
inventory match, frozen 23-key run_cfg equality.

Usage:
  ... scripts/a3c96_diag.py OUTDIR GATE_JSON SELFDRAW_JSON SHA96_FILE \
        --out-json OUT.json --fig-prefix FIGPREFIX
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

N_TOTAL = 96
RTOL = 1e-9
LEVELS = (0.68, 0.95)
WINSOR_K = 5                     # ceil(0.05 * 96), prereg section 5
TOP_K = (1, 2, 3, 5)
QUANTS = (0.025, 0.05, 0.16, 0.50, 0.84, 0.95, 0.975)
BOOT_B = 10_000
BOOT_SEED = 20260807
HOLM_ALPHA = 0.05

# tau0 ladder constants, COPIED VERBATIM from analyze_sbc_perleg.py so numbers match
Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])
_lx = np.log(1.0 + Z_TAU0)
_lx_c = _lx - _lx.mean()


class DiagRefusal(RuntimeError):
    """Integrity or deployment-consistency failure: no diagnostic output."""


def tau0_amp_slope(tau0_ladder):
    """COPIED VERBATIM from analyze_sbc_perleg.py (itself from analyze_sbc_heldout)."""
    y = np.log(np.clip(tau0_ladder, 1e-8, None))
    slope = np.sum(_lx_c * (y - y.mean())) / np.sum(_lx_c ** 2)
    intercept = y.mean()
    return np.exp(intercept), slope


def finite_L_null_sd(L_list):
    """The deployed arm-level finite-L reference (analyze_sbc_perleg.finite_L_null_sd):
    root-mean-variance of (1 + 1/L)(L-1)/(L-3); refuses L <= 3."""
    L = np.asarray(list(L_list), float)
    if L.size == 0 or np.any(L <= 3):
        raise DiagRefusal("finite_L_null_sd: empty or L <= 3 present")
    return float(np.sqrt(((1.0 + 1.0 / L) * (L - 1.0) / (L - 3.0)).mean()))


def rank_ks_p(rank_fracs):
    from scipy import stats as _st
    u = np.asarray([x for x in rank_fracs if np.isfinite(x)], float)
    return float(_st.kstest(u, "uniform").pvalue)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frozen_cfg():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analyze_firstarm_selfdraw.py")
    spec = importlib.util.spec_from_file_location("fsd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.REGISTRY["KS"]["frozen_cfg"])


def load_population(outdir, sha_file):
    """Census + sha + cfg conjuncts; returns the per-mock record list with the deployed
    statistics recomputed by the identical arithmetic."""
    names_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "mock_*.pkl"))
                         if not p.endswith(".smoke.pkl"))
    expect = [f"mock_{m:04d}.pkl" for m in range(N_TOTAL)]
    if names_files != expect:
        raise DiagRefusal(f"census != mock_0000..%04d (got {len(names_files)} files)" % (N_TOTAL - 1))
    recorded = {}
    with open(sha_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(None, 1)
            recorded[os.path.basename(name.strip().lstrip("*"))] = digest
    if set(recorded) != set(expect):
        raise DiagRefusal("sha inventory names != 96-mock census")
    frozen = _frozen_cfg()
    out = []
    for name in expect:
        path = os.path.join(outdir, name)
        if _sha256(path) != recorded[name]:
            raise DiagRefusal(f"sha256 mismatch vs the committed inventory: {name}")
        with open(path, "rb") as f:
            d = pickle.load(f)
        if dict(d["run_cfg"]) != frozen:
            raise DiagRefusal(f"run_cfg != frozen registry cfg: {name}")
        names = list(d["names"])
        dr = np.asarray(d["draws"], float)
        t = np.asarray(d["truth_vec"], float)
        j = names.index("ns")
        col = dr[:, j]
        L = int(d["L"])
        pull = (float(col.mean()) - float(t[j])) / float(col.std(ddof=1))
        rank = float(np.mean(col < t[j]))
        tidx = [names.index(f"tau0_z{z}") for z in range(13)]
        amps = np.array([tau0_amp_slope(row[tidx])[0] for row in dr])
        amp_t = tau0_amp_slope(t[tidx])[0]
        tau0_pull = (float(amps.mean()) - amp_t) / float(amps.std(ddof=1))
        out.append(dict(
            m=int(name[5:9]), pull=pull, rank=rank, L=L,
            post_sd=float(col.std(ddof=1)), truth_ns=float(t[j]),
            fres_abs=abs(float(d["sites_extra"]["f_res_amp"]["truth"])),
            tau0_pull=tau0_pull))
    return out


def check_consistency(recs, gate_json, selfdraw_json):
    """The binding deployment-consistency conjuncts (prereg section 1)."""
    with open(gate_json) as f:
        leg = json.load(f)["legs"]["KS"]
    with open(selfdraw_json) as f:
        sd = json.load(f)
    pulls = np.array([r["pull"] for r in recs])
    t0 = np.array([r["tau0_pull"] for r in recs])
    Ls = [r["L"] for r in recs]
    checks = [
        ("ns pull mean", float(np.mean(pulls)), float(leg["pulls"]["ns"]["mean"])),
        ("ns pull sd", float(np.std(pulls, ddof=1)), float(leg["pulls"]["ns"]["std"])),
        ("ns rank KS p", rank_ks_p([r["rank"] for r in recs]),
         float(leg["rank_uniformity"]["ns"]["ks_p"])),
        ("tau0amp pull mean", float(np.mean(t0)), float(leg["pulls"]["tau0amp"]["mean"])),
        ("tau0amp pull sd", float(np.std(t0, ddof=1)), float(leg["pulls"]["tau0amp"]["std"])),
        ("finite-L null sd", finite_L_null_sd(Ls), float(leg["finite_L_null"]["sd"])),
    ]
    for label, got, want in checks:
        if not np.isclose(got, want, rtol=RTOL, atol=0.0):
            raise DiagRefusal(f"deployment-consistency FAIL on {label}: recomputed {got!r} "
                              f"vs committed {want!r} (rtol {RTOL})")
    if float(np.median(Ls)) != float(sd["L_median"]) or \
            [min(Ls), max(Ls)] != [int(x) for x in sd["L_range"]]:
        raise DiagRefusal("L census != selfdraw JSON median/range")
    return {label: got for label, got, _ in checks}


def coverage(ranks):
    from scipy.stats import binomtest
    out = {}
    r = np.asarray(ranks, float)
    for q in LEVELS:
        lo, hi = (1.0 - q) / 2.0, (1.0 + q) / 2.0
        k = int(np.sum((r >= lo) & (r <= hi)))
        ci = binomtest(k, len(r)).proportion_ci(confidence_level=0.95, method="exact")
        out[f"{q:.2f}"] = dict(covered=k, n=len(r), fraction=k / len(r), nominal=q,
                               diff=k / len(r) - q, ci95=[float(ci.low), float(ci.high)])
    return out


def winsorized_sd(x, k=WINSOR_K):
    """Prereg section 5: explicit symmetric winsorization, k per tail."""
    s = np.sort(np.asarray(x, float))
    w = s.copy()
    w[:k] = s[k]
    w[-k:] = s[-(k + 1)]
    return float(np.std(w, ddof=1))


def scales_and_tail(pulls):
    from scipy.stats import median_abs_deviation
    x = np.asarray(pulls, float)
    n = len(x)
    sd = float(np.std(x, ddof=1))
    xc = x - x.mean()
    ss = float(np.sum(xc ** 2))
    order = np.argsort(-np.abs(xc))
    contrib = {str(k): float(np.sum(xc[order[:k]] ** 2) / ss) for k in TOP_K}
    loo = np.array([np.std(np.delete(x, i), ddof=1) for i in range(n)])
    top5 = [dict(m=int(i), pull=float(x[i])) for i in np.argsort(-np.abs(x))[:5]]
    mad = float(median_abs_deviation(x, scale="normal"))
    wins = winsorized_sd(x)
    return dict(
        sd=sd, median=float(np.median(x)), mad_scale=mad,
        mad_norm_constant=1.4826022185056018,
        winsorized_sd=wins, winsor_k=WINSOR_K,
        quantiles={f"{q:g}": float(np.quantile(x, q)) for q in QUANTS},
        top5_abs_pulls=top5, variance_contrib_topk=contrib,
        loo_sd_min=float(loo.min()), loo_sd_max=float(loo.max()),
        loo_sd_max_abs_change=float(np.max(np.abs(loo - sd))),
        loo_sd_all=[float(v) for v in loo],
        sd_over_mad=sd / mad, winsorized_over_sd=wins / sd)


def holm(pvals, alpha=HOLM_ALPHA):
    """Step-down Holm: returns the per-test significance flags (input order)."""
    m = len(pvals)
    order = np.argsort(pvals)
    sig = [False] * m
    for j, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - j):
            sig[idx] = True
        else:
            break
    return sig


def metadata_dependence(recs):
    from scipy.stats import spearmanr
    pulls = np.array([r["pull"] for r in recs])
    ab = np.abs(pulls)
    Ls = np.array([r["L"] for r in recs], float)
    null_i = np.sqrt((1.0 + 1.0 / Ls) * (Ls - 1.0) / (Ls - 3.0))
    variables = [
        ("V1 truth_ns", np.array([r["truth_ns"] for r in recs]), ab),
        ("V2 post_sd_ns", np.array([r["post_sd"] for r in recs]), ab),
        ("V3 L (target |pull|/null_L)", Ls, ab / null_i),
        ("V4 |f_res_amp truth|", np.array([r["fres_abs"] for r in recs]), ab),
        ("V5 |tau0amp pull|", np.abs(np.array([r["tau0_pull"] for r in recs])), ab),
    ]
    rows = []
    for label, v, target in variables:
        rho, p = spearmanr(v, target)
        rows.append(dict(variable=label, rho=float(rho), p=float(p)))
    flags = holm([r["p"] for r in rows])
    for r, f in zip(rows, flags):
        r["holm_flagged"] = bool(f)
    return rows, variables


def rscale_bootstrap(pulls, Ls, B=BOOT_B, seed=BOOT_SEED):
    x = np.asarray(pulls, float)
    L = np.asarray(Ls, float)
    n = len(x)
    s_obs = float(np.std(x, ddof=1))
    s_ref = finite_L_null_sd(L)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    xs = x[idx]
    s_star = xs.std(ddof=1, axis=1)
    Lb = L[idx]
    ref_star = np.sqrt(((1.0 + 1.0 / Lb) * (Lb - 1.0) / (Lb - 3.0)).mean(axis=1))
    r_star = s_star / ref_star

    def pct(a, lo, hi):
        return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]

    def basic(a, point, lo, hi):
        # reverse-percentile: 2*point - upper/lower percentiles
        p = pct(a, lo, hi)
        return [2 * point - p[1], 2 * point - p[0]]

    out = dict(
        s_obs=s_obs, s_ref=s_ref, r_scale=s_obs / s_ref, B=B, seed=seed,
        s_obs_ci68=pct(s_star, 16, 84), s_obs_ci95=pct(s_star, 2.5, 97.5),
        r_ci68=pct(r_star, 16, 84), r_ci95=pct(r_star, 2.5, 97.5),
        s_obs_basic68=basic(s_star, s_obs, 16, 84),
        s_obs_basic95=basic(s_star, s_obs, 2.5, 97.5),
        r_basic68=basic(r_star, s_obs / s_ref, 16, 84),
        r_basic95=basic(r_star, s_obs / s_ref, 2.5, 97.5),
    )
    # material sensitivity: basic-vs-percentile discrepancy > 10% of interval width
    def sens(a, b):
        w = b[1] - b[0]
        return bool(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 0.10 * w) if w > 0 else False
    out["sensitivity_material"] = dict(
        s68=sens(out["s_obs_basic68"], out["s_obs_ci68"]),
        s95=sens(out["s_obs_basic95"], out["s_obs_ci95"]),
        r68=sens(out["r_basic68"], out["r_ci68"]),
        r95=sens(out["r_basic95"], out["r_ci95"]))
    return out


def make_figures(recs, scales, variables, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pulls = np.array([r["pull"] for r in recs])
    Ls = [r["L"] for r in recs]
    s_ref = finite_L_null_sd(Ls)
    # F1 pull histogram + N(0, s_ref)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pulls, bins=24, density=True, alpha=0.6, label="n_s pulls (N=96)")
    g = np.linspace(pulls.min() - 0.5, pulls.max() + 0.5, 300)
    ax.plot(g, np.exp(-g**2 / (2 * s_ref**2)) / np.sqrt(2 * np.pi * s_ref**2),
            label=f"N(0, {s_ref:.4f}) finite-L ref")
    ax.set_xlabel("n_s pull"); ax.legend(); fig.tight_layout()
    fig.savefig(prefix + "_f1_hist.png", dpi=130); plt.close(fig)
    # F2 LOO strip
    fig, ax = plt.subplots(figsize=(6, 3))
    loo = scales["loo_sd_all"]
    ax.plot(range(96), loo, ".", ms=4)
    ax.axhline(scales["sd"], ls="--", lw=1, label=f"full sd {scales['sd']:.4f}")
    ax.axhline(1.1, ls=":", lw=1, label="frozen gate 1.1")
    ax.set_xlabel("omitted mock"); ax.set_ylabel("LOO sd"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(prefix + "_f2_loo.png", dpi=130); plt.close(fig)
    # F3 metadata scatters
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, (label, v, target) in zip(axes, variables):
        ax.plot(v, target, ".", ms=4)
        ax.set_title(label, fontsize=8); ax.set_ylabel("target", fontsize=7)
    fig.tight_layout(); fig.savefig(prefix + "_f3_meta.png", dpi=130); plt.close(fig)
    # F4 rank histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist([r["rank"] for r in recs], bins=12, range=(0, 1), alpha=0.7)
    ax.axhline(96 / 12, ls="--", lw=1)
    ax.set_xlabel("rank of truth (deployed statistic)")
    fig.tight_layout(); fig.savefig(prefix + "_f4_ranks.png", dpi=130); plt.close(fig)


def run(outdir, gate_json, selfdraw_json, sha_file):
    recs = load_population(outdir, sha_file)
    consistency = check_consistency(recs, gate_json, selfdraw_json)
    pulls = [r["pull"] for r in recs]
    Ls = [r["L"] for r in recs]
    cov = coverage([r["rank"] for r in recs])
    sc = scales_and_tail(pulls)
    meta_rows, variables = metadata_dependence(recs)
    rs = rscale_bootstrap(pulls, Ls)
    return dict(consistency=consistency, coverage=cov, scales=sc,
                metadata=meta_rows, rscale=rs,
                per_mock=[{k: r[k] for k in ("m", "pull", "rank", "L")} for r in recs]), \
        recs, sc, variables


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("gate_json")
    ap.add_argument("selfdraw_json"); ap.add_argument("sha96_file")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fig-prefix", required=True)
    a = ap.parse_args(argv[1:])
    out, recs, sc, variables = run(a.outdir, a.gate_json, a.selfdraw_json, a.sha96_file)
    make_figures(recs, sc, variables, a.fig_prefix)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("=== A3c N=96 BOUNDED DIAGNOSTIC (descriptive only; Branch B unchanged) ===")
    print("deployment-consistency conjuncts: ALL PASS (gate-JSON equality at rtol 1e-9)")
    print("\nT1 COVERAGE (equal-tailed empirical band on the deployed rank statistic):")
    for q, c in out["coverage"].items():
        print(f"  {float(q):.0%}: {c['covered']}/96 = {c['fraction']:.4f} "
              f"(nominal {c['nominal']:.2f}, diff {c['diff']:+.4f}, "
              f"exact CI95 [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}])")
    print("\nT2 SCALE/TAIL:")
    print(f"  sd {sc['sd']:.4f} | median {sc['median']:+.4f} | MAD-scale {sc['mad_scale']:.4f} "
          f"| winsorized(k=5) {sc['winsorized_sd']:.4f}")
    print(f"  sd/MAD {sc['sd_over_mad']:.4f} | winsorized/sd {sc['winsorized_over_sd']:.4f}")
    print(f"  quantiles: " + "  ".join(f"{k}:{v:+.3f}" for k, v in sc["quantiles"].items()))
    print(f"  top-5 |pulls|: " + ", ".join(f"m{t['m']}:{t['pull']:+.3f}" for t in sc["top5_abs_pulls"]))
    print(f"  variance contrib top-k: " + ", ".join(f"k={k}:{v:.3f}"
          for k, v in sc["variance_contrib_topk"].items()))
    print(f"\nT3 LOO sd: min {sc['loo_sd_min']:.4f} | max {sc['loo_sd_max']:.4f} | "
          f"max |change| {sc['loo_sd_max_abs_change']:.4f}")
    print("\nT4 METADATA (Spearman vs preregistered target; Holm alpha 0.05):")
    for r in out["metadata"]:
        print(f"  {r['variable']:32s} rho {r['rho']:+.4f}  p {r['p']:.4f}  "
              f"{'FLAGGED' if r['holm_flagged'] else 'not flagged'}")
    rs = out["rscale"]
    print(f"\nT5 r_scale: s_obs {rs['s_obs']:.4f} / s_ref {rs['s_ref']:.4f} = "
          f"{rs['r_scale']:.4f}")
    print(f"  s_obs CI68 [{rs['s_obs_ci68'][0]:.4f}, {rs['s_obs_ci68'][1]:.4f}] "
          f"CI95 [{rs['s_obs_ci95'][0]:.4f}, {rs['s_obs_ci95'][1]:.4f}]")
    print(f"  r     CI68 [{rs['r_ci68'][0]:.4f}, {rs['r_ci68'][1]:.4f}] "
          f"CI95 [{rs['r_ci95'][0]:.4f}, {rs['r_ci95'][1]:.4f}]  "
          f"(percentile, B={rs['B']}, seed {rs['seed']}; paired finite-L denominator)")
    print(f"  basic-interval material sensitivity: {rs['sensitivity_material']}")
    print(f"\njson -> {a.out_json}")


if __name__ == "__main__":
    main(sys.argv)
