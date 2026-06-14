"""Merge production-ensemble Leg-A SBC shards -> aggregate -> rank-uniformity pass/fail + figure.

Verdict basis = rank uniformity (the ECDF simultaneous-band test, Säilynoja+2022) at
L_eff ≥ L_FLOOR, per quantity (A_p, n_s primary). This is the §6 inference-calibration gate.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import numpy as np
import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator.closure_sbc import aggregate_leg_a, L_FLOOR


def _fig(res, figpath, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = res["names"]
    n = len(names)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[j // ncol][j % ncol]
        band = res["ecdf_bands"].get(nm)
        if band is None:
            ax.set_visible(False); continue
        lower, upper, ecdf, grid = band
        ax.fill_between(grid, lower, upper, color="0.85", label="95% band")
        ax.plot(grid, grid, "k--", lw=0.8)
        ok = res["passed"].get(nm, False)
        ax.plot(grid, ecdf, color=("tab:green" if ok else "tab:red"), lw=1.4)
        ax.set_title(f"{nm}  [{'PASS' if ok else 'FAIL'}]", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    fig.savefig(figpath, dpi=130)
    return figpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--prob", type=float, default=0.95)
    ap.add_argument("--figdir", default="/home/mfho/hcd_priya/figures/analysis/06_validation_summary")
    a = ap.parse_args()

    per_mock, seen, n_z = [], set(), None
    for fn in sorted(glob.glob(os.path.join(a.shard_dir, "shard_*.pkl"))):
        with open(fn, "rb") as f:
            dd = pickle.load(f)
        n_z = dd["n_z"] if n_z is None else n_z
        assert dd["n_z"] == n_z, f"shard {fn} n_z={dd['n_z']} != {n_z}"
        for rec, mi in zip(dd["per_mock"], dd["idxs"]):
            if mi in seen:
                continue
            seen.add(mi); per_mock.append(rec)
    if not per_mock:
        raise SystemExit(f"no shards in {a.shard_dir}")
    res = aggregate_leg_a(per_mock, n_z, prob=a.prob)

    print("\n===== Production-ensemble Leg-A SBC (rank uniformity) =====")
    print(f"mocks merged={len(per_mock)}  kept(rankable)={res['n_kept']}  "
          f"divergent={res['n_divergent']}  excluded={res['n_excluded']}  "
          f"L_eff(thinned)={res['L']}  gate_valid(L>={L_FLOOR})={res['gate_valid']}")
    if res["passed"]:
        n_pass = sum(res["passed"].values())
        print(f"\nper-quantity ECDF rank-uniformity (prob={a.prob}):")
        for nm in res["names"]:
            print(f"  {nm:14s}: {'PASS' if res['passed'][nm] else 'FAIL'}")
        verdict = ("CALIBRATION verdict" if res["gate_valid"]
                   else f"PATH-only — L={res['L']}<{L_FLOOR} (bands over-reject; bump n_samples)")
        print(f"\n{n_pass}/{len(res['names'])} quantities PASS ({verdict})")
    else:
        print("no rankable mocks -> no band test (raise N or n_samples / reduce divergences)")

    p = _fig(res, os.path.join(a.figdir, "prod_sbc_pilot.png"),
             f"Production-ensemble Leg-A SBC (N={len(per_mock)}, L_eff={res['L']}) — "
             f"rank uniformity {'VALID' if res['gate_valid'] else 'PATH-only'}")
    print(f"[fig] {p}")


if __name__ == "__main__":
    main()
