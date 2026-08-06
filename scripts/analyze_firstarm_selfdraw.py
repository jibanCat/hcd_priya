#!/usr/bin/env python3
"""FIRST-ARM self-draw conjuncts + health block (A3c gate G1; serves A2c unchanged).

A FIRST certification arm (corrected KS A3c, corrected DESI A2c) has no defective
reference arm, so the A1c paired repair machinery does not apply. What replaces it, per
the 2026-08-05/06 pre-registrations:

  * CONJUNCTS (fail-loud; on any failure this module raises FirstArmError and prints NO
    statistics): census; every self-drawn truth finite/nonzero (Normal sites) or strictly
    in-bracket (LogUniform sites); across-mock distinctness per site; law-consistency per
    site at alpha 1e-3; `truth_site_semantics.not_self_drawn == []` on every pkl; frozen
    run_cfg equality across the arm (incl. the sampler-population stamps).
  * HEALTH BLOCK (binding EVIDENCE, never a refusal and never a gate): per-site
    rank-of-truth uniformity (KS test, healthy iff p >= 1e-3), plus on DESI the f_res_amp
    across-mock scatter ratio in [0.5, 2.0] (EXCLUDED on KS with the reason recorded: the
    sector is prior-dominated there, shrinkage ~0.98, so the Gaussian-linear expectation
    is ill-conditioned -- A3d measured 0.0463 observed vs 0.0053 linear on a healthy drawn
    arm; on KS the scatter is REPORTED, never thresholded). A health failure sets
    healthy=False and feeds the disposition's row 2; the arm still reads out.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/analyze_firstarm_selfdraw.py OUTDIR N --leg KS|DESI [--json OUT.json]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats as _st

LAW_ALPHA = 1e-3
HEALTH_RANK_ALPHA = 1e-3

_SIG = "50befc941edfc4c789286d2054d0107f1eb19a5672bcb86131426fb101eea216"

_COMMON_CFG = dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, fold=0,
                   tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=False,
                   sample_res=True, fres_selfdraw=True, diag_no_sample_metals=False,
                   hcd_prior_signature=_SIG, single_member=False,
                   seed=20260614, n_warmup=250, n_samples=600, max_tree_depth=10)

# Per-leg registry. Site spec: ("normal", sigma) or ("loguniform", lo, hi).
# The DESI frozen_cfg is pinned from the design record and MUST be verified against the
# A2c smoke pkl before any DESI launch (A2c pre-registration gate G4-equivalent); a
# mismatch there is a registry bug to fix WITH the panel, never a reason to relax C6.
REGISTRY = {
    "KS": dict(
        sites={"f_res_amp": ("normal", 0.15), "f_res_slope": ("normal", 0.5)},
        scatter_site=None, scatter_band=None,
        frozen_cfg=dict(_COMMON_CFG, leg="KS", survey="KS", metal_prior="uniform",
                        metal_selfdraw=False, f_res_amp_sigma=0.15, ks_kmax=0.065,
                        hcd_parameterization="dndx_mapped_v2"),
    ),
    "DESI": dict(
        sites={
            "f_SiIII_DESI_z0": ("loguniform", 0.003, 0.03),
            "f_SiIII_DESI_z1": ("loguniform", 0.003, 0.03),
            "f_SiII_DESI_z0": ("loguniform", 0.003, 0.03),
            "f_SiII_DESI_z1": ("loguniform", 0.003, 0.03),
            "k_SiIII_DESI_z0": ("loguniform", 1e-3, 1e-1),
            "k_SiIII_DESI_z1": ("loguniform", 1e-3, 1e-1),
            "k_SiII_DESI_z0": ("loguniform", 1e-3, 1e-1),
            "k_SiII_DESI_z1": ("loguniform", 1e-3, 1e-1),
            "f_res_amp": ("normal", 0.02),
            "f_res_slope": ("normal", 0.5),
        },
        scatter_site="f_res_amp", scatter_band=(0.5, 2.0),
        frozen_cfg=dict(_COMMON_CFG, leg="DESI", survey="DESI",
                        metal_prior="flatlog2node", metal_selfdraw=True,
                        f_res_amp_sigma=0.02, ks_kmax=None,
                        hcd_parameterization="alpha_pivot_powerlaw_v1"),
    ),
}


class FirstArmError(RuntimeError):
    """A conjunct failed: the arm is not the pre-registered population. NO statistics."""


def _fail(msg):
    raise FirstArmError(msg + " -- REFUSING to print any statistic.")


def run(outdir, n, leg):
    n = int(n)
    reg = REGISTRY[leg]
    files = sorted(glob.glob(os.path.join(outdir, "mock_*.pkl")))
    want = [os.path.join(outdir, f"mock_{m:04d}.pkl") for m in range(n)]
    if files != want:
        _fail(f"census: found {len(files)} mock pkls, expected exactly mock_0000..{n-1:04d} "
              f"(missing {sorted(set(map(os.path.basename, want)) - set(map(os.path.basename, files)))[:4]}, "
              f"extra {sorted(set(map(os.path.basename, files)) - set(map(os.path.basename, want)))[:4]})")
    recs = []
    for p in want:
        with open(p, "rb") as f:
            recs.append(pickle.load(f))

    # C2/C3/C4: per-site truth checks
    truths = {}
    for k, spec in reg["sites"].items():
        vals = []
        for m, rec in enumerate(recs):
            se = rec.get("sites_extra", {})
            if k not in se:
                _fail(f"mock {m}: self-drawn site {k} absent from sites_extra")
            vals.append(float(se[k]["truth"]))
        vals = np.array(vals)
        if spec[0] == "normal":
            if not (np.all(np.isfinite(vals)) and np.all(vals != 0.0)):
                _fail(f"site {k}: truth not finite/nonzero on some mock (pinned?)")
            law_p = _st.kstest(vals / spec[1], "norm").pvalue
        else:
            lo, hi = spec[1], spec[2]
            if not (np.all(np.isfinite(vals)) and np.all(vals > lo) and np.all(vals < hi)):
                _fail(f"site {k}: truth outside the open bracket ({lo}, {hi}) on some mock")
            u = (np.log(vals) - np.log(lo)) / (np.log(hi) - np.log(lo))
            law_p = _st.kstest(u, "uniform").pvalue
        if np.unique(vals).size != n:
            _fail(f"site {k}: truths not distinct across mocks (decoy population)")
        if law_p < LAW_ALPHA:
            _fail(f"site {k}: truths inconsistent with the deployed law (KS p {law_p:.2e} "
                  f"< {LAW_ALPHA})")
        truths[k] = (vals, law_p)

    # C5: runner provenance
    for m, rec in enumerate(recs):
        nsd = rec.get("truth_site_semantics", {}).get("not_self_drawn", "MISSING")
        if nsd != []:
            _fail(f"mock {m}: not_self_drawn = {nsd!r} (expected [])")

    # C6: frozen run_cfg
    for m, rec in enumerate(recs):
        cfg = rec.get("run_cfg")
        if cfg != reg["frozen_cfg"]:
            diff = {kk: (None if not cfg else cfg.get(kk), reg["frozen_cfg"][kk])
                    for kk in reg["frozen_cfg"]
                    if cfg is None or cfg.get(kk) != reg["frozen_cfg"][kk]}
            extra = set(cfg or {}) - set(reg["frozen_cfg"])
            _fail(f"mock {m}: run_cfg != the frozen registry cfg (diff {diff}, "
                  f"extra {sorted(extra)})")

    # HEALTH BLOCK (evidence, not a refusal)
    health = {}
    for k, spec in reg["sites"].items():
        ranks, mus, sds = [], [], []
        for rec in recs:
            dr = np.asarray(rec["sites_extra"][k]["draws"], float)
            tr = float(rec["sites_extra"][k]["truth"])
            ranks.append(float(np.mean(dr < tr)))
            mus.append(float(np.mean(dr))); sds.append(float(np.std(dr, ddof=1)))
        ranks, mus, sds = map(np.array, (ranks, mus, sds))
        rank_p = float(_st.kstest(ranks, "uniform").pvalue)
        healthy = rank_p >= HEALTH_RANK_ALPHA
        scatter = float(mus.std(ddof=1))
        if spec[0] == "normal":
            expected = float(np.sqrt(max(spec[1] ** 2 - float(np.mean(sds ** 2)), 0.0)))
        else:
            expected = None            # LogUniform: no Gaussian-linear expectation
        ratio = (scatter / expected if expected else None)
        if reg["scatter_site"] == k:
            lo, hi = reg["scatter_band"]
            healthy = healthy and (ratio is not None and lo <= ratio <= hi)
        health[k] = dict(rank_ks_p=rank_p, healthy=bool(healthy), scatter=scatter,
                         scatter_expected=expected, scatter_ratio=ratio,
                         pull_mean=float(((mus - truths[k][0]) / sds).mean()),
                         pull_sd=float(((mus - truths[k][0]) / sds).std(ddof=1)),
                         law_ks_p=truths[k][1],
                         thresholded_on_scatter=(reg["scatter_site"] == k))

    ndiv = sum(int(r.get("n_div", 0)) for r in recs)
    Ls = [int(r.get("L", 0)) for r in recs]
    return dict(survey=leg, n=n, conjuncts_ok=True, health=health,
                health_ok=all(h["healthy"] for h in health.values()),
                n_div_total=ndiv, L_median=float(np.median(Ls)),
                L_range=[min(Ls), max(Ls)],
                health_rank_alpha=HEALTH_RANK_ALPHA)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("n", type=int)
    ap.add_argument("--leg", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--json", default=None, help="write the result JSON here")
    a = ap.parse_args(argv[1:])
    out = run(a.outdir, a.n, a.leg)
    print(f"=== FIRST-ARM SELF-DRAW READOUT ({a.leg}, n={out['n']}) ===")
    print(f"conjuncts: ALL PASS; n_div total {out['n_div_total']}; "
          f"L median {out['L_median']:.0f} range {out['L_range']}")
    for k, h in out["health"].items():
        extra = (f", scatter ratio {h['scatter_ratio']:.2f} in {REGISTRY[a.leg]['scatter_band']}"
                 if h["thresholded_on_scatter"] else
                 f", scatter {h['scatter']:.4f} (reported, not thresholded)")
        print(f"  {k:18s} rank KS-p {h['rank_ks_p']:.4f} -> "
              f"{'HEALTHY' if h['healthy'] else 'UNHEALTHY'}{extra}")
    print(f"HEALTH BLOCK: {'PASS' if out['health_ok'] else 'FAIL (row-2 evidence)'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main(sys.argv)
