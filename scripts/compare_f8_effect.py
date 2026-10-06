#!/usr/bin/env python3
"""Gate C report R3: the effect of the F8 cache repair on emulator PREDICTIONS (GATE_C_SPEC.md section 4).

Same recipe, same seed: production member seed 0 trained on the HISTORICAL cache vs on the REPAIRED cache; both predict
per-class P_filt at every row of the repaired cache (every simulation's theta, every z, every rung, incl. the restored
ns0.907 z = 3.0 rows). The difference is compared with the seed-to-seed scatter (seeds 0 and 1 on the repaired cache).
Reported overall, by class and z, at ns0.907 (z 2.8, 3.0), at its nearest neighbours in the unit cube, and elsewhere.
No claim about historical inference is made (S4 ruling).

Usage: PYTHONPATH=<repo> JAX_PLATFORMS=cpu python3 scripts/compare_f8_effect.py --rep0 CK --rep1 CK --hist0 CK --out-json J
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, _REPO_ROOT)
import hcd_analysis.emulator  # noqa: F401,E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import load_cache, normalize_params  # noqa: E402
from hcd_analysis.emulator.predict import predict_P_filt  # noqa: E402

CLS = ("clean", "LLS", "subDLA", "DLA")
NS0907 = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
CACHE = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def predict_all(ckpt, d):
    model, meta, norm = T.load_checkpoint(ckpt)
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    x, tau0 = jnp.asarray(d["x"]), jnp.asarray(d["tau0"])
    f = jax.jit(jax.vmap(lambda xi, ti: predict_P_filt(model, xi[:9], xi[9], ti, pf)))
    out = [np.asarray(f(x[i:i + 2048], tau0[i:i + 2048])) for i in range(0, x.shape[0], 2048)]
    return np.concatenate(out), meta


def summary(diff, sel, keep):
    r = np.where(keep[sel][:, None, :], np.abs(diff[sel]), np.nan)
    return {c: dict(median=float(np.nanmedian(r[:, i])), p99=float(np.nanpercentile(r[:, i], 99)),
                    max=float(np.nanmax(r[:, i]))) for i, c in enumerate(CLS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep0", required=True)
    ap.add_argument("--rep1", required=True)
    ap.add_argument("--hist0", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--n-neighbours", type=int, default=5)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    d = load_cache(a.cache)
    P0, m0 = predict_all(a.rep0, d)
    P1, _ = predict_all(a.rep1, d)
    PH, mh = predict_all(a.hist0, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        dF8 = PH / P0 - 1.0
        dseed = P1 / P0 - 1.0
    keep = d["mask"].astype(bool)
    names = np.asarray(d["sim_name"])
    u = normalize_params(d["params"])
    sims = sorted(set(names))
    cen = {s: u[names == s][0] for s in sims}
    dist = sorted((float(np.linalg.norm(cen[s] - cen[NS0907])), s) for s in sims if s != NS0907)
    neigh = [s for _, s in dist[:a.n_neighbours]]
    z = np.round(d["z_grid"], 1)
    groups = {"all_rows": np.ones(names.size, bool),
              "ns0907_z2.8": (names == NS0907) & (z == 2.8), "ns0907_z3.0": (names == NS0907) & (z == 3.0),
              "ns0907_other_z": (names == NS0907) & (z != 2.8) & (z != 3.0),
              f"nearest_{a.n_neighbours}_neighbours": np.isin(names, neigh),
              "elsewhere": ~np.isin(names, neigh + [NS0907])}
    out = dict(rep0=a.rep0, rep1=a.rep1, hist0=a.hist0, rep_cache_sha=m0["cache_sha256"], hist_cache_sha=mh["cache_sha256"],
               neighbours=neigh, groups={})
    for g, sel in groups.items():
        out["groups"][g] = dict(n_rows=int(sel.sum()), F8_effect=summary(dF8, sel, keep), seed_scatter=summary(dseed, sel, keep))
    out["by_z_all_rows_F8_median"] = {str(zz): {c: float(np.nanmedian(np.where(keep[z == zz][:, None, :],
                                                                                np.abs(dF8[z == zz]), np.nan)[:, i]))
                                                for i, c in enumerate(CLS)} for zz in np.unique(z)}
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("| group | rows | F8 effect median / p99 (clean, DLA) | seed scatter median / p99 (clean, DLA) |")
    print("|---|---|---|---|")
    for g, v in out["groups"].items():
        fe, ss = v["F8_effect"], v["seed_scatter"]
        print(f"| {g} | {v['n_rows']} | {100 * fe['clean']['median']:.3f}/{100 * fe['clean']['p99']:.3f}%, "
              f"{100 * fe['DLA']['median']:.3f}/{100 * fe['DLA']['p99']:.3f}% | {100 * ss['clean']['median']:.3f}/"
              f"{100 * ss['clean']['p99']:.3f}%, {100 * ss['DLA']['median']:.3f}/{100 * ss['DLA']['p99']:.3f}% |")


if __name__ == "__main__":
    main()
