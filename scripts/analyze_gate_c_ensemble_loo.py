#!/usr/bin/env python3
"""S7 diagnostic read-out (PI ruling 2026-10-06; read-out pre-specified in PU-0049): the C4 comparison for the
production-equivalent 5-member ensemble of the 60-fold leave-one-simulation-out models.

Members: seed 0 (``<eval-dir>/eval_loo60_s{NN}.npz``) and seeds 1-4 (``<eval-dir>/loo60_ensemble/eval_loo60_s{NN}_seed{S}.npz``).
Ensemble residual per entry = mean of the members' residuals (equal to the residual of the mean prediction: common
truth, linear interpolation onto the KS bins). Same 7790 matched entries and same comparison as C4.

Usage: PYTHONPATH=<repo> python3 scripts/analyze_gate_c_ensemble_loo.py --eval-dir DIR --out-json OUT
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, _REPO_ROOT)
from hcd_analysis.emulator import gate_c as G  # noqa: E402

UP_LOO = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/loo_fps.hdf5"
CACHE = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def stats(e):
    a = np.abs(e)
    return dict(median=float(np.median(a)), rms=float(np.sqrt(np.mean(e ** 2))), mean_signed=float(np.mean(e)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)), p999=float(np.percentile(a, 99.9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    with h5py.File(a.cache, "r") as f:
        sims, prm = f["sim_name"].asstr()[...], f["params"][...]
    sim_params = {s: prm[i] for i, s in enumerate(sims)}
    with h5py.File(UP_LOO, "r") as f:
        up = {k: f[k][...] for k in ("flux_predict", "flux_true", "params", "zout")}
    members, ens, upe, zs = [[] for _ in range(5)], [], [], []

    def member_evals():
        for n in range(60):
            files = [f"{a.eval_dir}/eval_loo60_s{n:02d}.npz"] + [
                f"{a.eval_dir}/loo60_ensemble/eval_loo60_s{n:02d}_seed{s}.npz" for s in range(1, 5)]
            yield [dict(np.load(p, allow_pickle=True)) for p in files]
    entries, res, kks = G.c4_collect(member_evals(), sim_params)              # res[key]: (5, 11)
    z_of = {e["key"]: e["z"] for e in entries}
    # checked comparison set (BT-C1): every upstream entry outside F1 matched once, all members finite, 7790 entries
    for key, (ur, zi) in G.c4_matched_set(entries, res, up["params"], up["zout"]):
        for m in range(5):
            members[m].append(res[key][m])
        ens.append(G.ensemble_residual(list(res[key])))
        upe.append(up["flux_predict"][ur, zi] / up["flux_true"][ur, zi] - 1.0)
        zs.append(z_of[key])
    ens, upe, zs = np.array(ens), np.array(upe), np.array(zs)
    out = dict(n_entries=int(ens.shape[0]), ensemble=stats(ens), upstream=stats(upe),
               members=[stats(np.array(m)) for m in members])
    out["ensemble_meets_C4_inequality"] = bool(out["ensemble"]["median"] <= out["upstream"]["median"]
                                              and out["ensemble"]["rms"] <= out["upstream"]["rms"])
    out["per_k_median"] = {f"{k:.4f}": [float(np.median(np.abs(ens[:, j]))), float(np.median(np.abs(upe[:, j])))]
                           for j, k in enumerate(kks)}
    out["per_z_median"] = {str(z): [float(np.median(np.abs(ens[np.round(zs, 1) == z]))),
                                    float(np.median(np.abs(upe[np.round(zs, 1) == z]))), int((np.round(zs, 1) == z).sum())]
                           for z in np.unique(np.round(zs, 1))}
    json.dump(out, open(a.out_json, "w"), indent=1)
    e, u = out["ensemble"], out["upstream"]
    print(f"entries {out['n_entries']}")
    print(f"ensemble: median {100 * e['median']:.3f}% RMS {100 * e['rms']:.3f}% mean {100 * e['mean_signed']:+.3f}% "
          f"p90/99/99.9 {100 * e['p90']:.2f}/{100 * e['p99']:.2f}/{100 * e['p999']:.2f}%")
    print(f"upstream: median {100 * u['median']:.3f}% RMS {100 * u['rms']:.3f}% mean {100 * u['mean_signed']:+.3f}% "
          f"p90/99/99.9 {100 * u['p90']:.2f}/{100 * u['p99']:.2f}/{100 * u['p999']:.2f}%")
    print("members (median / RMS %): " + ", ".join(f"{100 * m['median']:.3f}/{100 * m['rms']:.3f}" for m in out["members"]))
    print(f"ensemble meets the C4 inequality (median AND RMS <= upstream): {out['ensemble_meets_C4_inequality']}")
    print("per k (ensemble / upstream median %): " + ", ".join(f"{k}: {100 * v[0]:.3f}/{100 * v[1]:.3f}" for k, v in out["per_k_median"].items()))
    print("per z (ensemble / upstream median %): " + ", ".join(f"{z}: {100 * v[0]:.3f}/{100 * v[1]:.3f}" for z, v in out["per_z_median"].items()))


if __name__ == "__main__":
    main()
