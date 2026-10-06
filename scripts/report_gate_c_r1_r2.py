#!/usr/bin/env python3
"""Gate C reports completed in the review fix pass (gate C review concern 5; GATE_C_SPEC.md R1, R2):

R1 by fold n_s range: the 8-fold folds hold out contiguous blocks of simulations sorted by name, i.e. by n_s. Per fold
(repaired cache, seed 0; C2's rows: non-tau0-edge held-out rows, in-range finite modes): the held-out n_s range,
per-class RMS and the median |total residual|. And C4 (single member, seed 0, the checked 7790 entries) by the held-out
simulation's n_s in five equal-count groups, ours vs upstream.

R2 per-mode vs interpolated: on the same C4 entries, |residual| per comoving mode (each row's own grid, modes inside
the KS bin span) vs |residual| interpolated onto the 11 KS bins; pooled and per z.

Usage: PYTHONPATH=<repo> python3 scripts/report_gate_c_r1_r2.py --eval-dir DIR --out-json OUT
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, _REPO_ROOT)
from hcd_analysis.emulator import gate_c as G  # noqa: E402

UP_LOO = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/loo_fps.hdf5"
CACHE = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def load(path):
    e = np.load(path, allow_pickle=True)
    return {k: e[k] for k in e.files}


def med_rms(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(np.abs(v))), rms=float(np.sqrt(np.mean(v ** 2))), n=int(v.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out_json):
        raise SystemExit(f"refusing to overwrite {a.out_json}")
    with h5py.File(a.cache, "r") as f:
        sims, prm, kfkms = f["sim_name"].asstr()[...], f["params"][...], f["kfkms"][...]
    sim_params = dict(zip(sims, prm))
    with h5py.File(UP_LOO, "r") as f:
        up = {k: f[k][...] for k in ("flux_predict", "flux_true", "params", "zout")}
    out = {}

    # ---------------- R1: 8-fold by fold n_s range
    folds = {}
    for fold in range(8):
        e = load(f"{a.eval_dir}/eval_loso8_repaired_fold{fold}.npz")
        v = ~e["is_tau0_edge"]
        keep = e["in_range"][v] & np.isfinite(e["res_tot"][v])
        ns = sorted({float(sim_params[str(s)][0]) for s in e["sim_name"]})
        rms = G.class_rms(e["res_cls"][v], keep)
        folds[str(fold)] = dict(ns_min=ns[0], ns_max=ns[-1], n_sims=len(ns),
                                class_rms=dict(zip(("clean", "LLS", "subDLA", "DLA"), rms.tolist())),
                                median_abs_total=float(np.median(np.abs(e["res_tot"][v][keep]))))
    out["R1_8fold_by_fold_ns_range"] = folds

    # ---------------- C4 entries (single member, seed 0), checked set
    files = [f"{a.eval_dir}/eval_loo60_s{n:02d}.npz" for n in range(60)]
    evals = [load(p) for p in files]
    entries, res, kks = G.c4_collect(([e] for e in evals), sim_params)
    matched = G.c4_matched_set(entries, res, up["params"], up["zout"])
    ent = {e["key"]: e for e in entries}

    # R1: C4 by held-out n_s, five equal-count groups of simulations
    sim_ns = np.array([float(sim_params[str(e["sim_name"][0])][0]) for e in evals])
    order = np.argsort(sim_ns)
    groups = np.array_split(order, 5)
    by_ns = {}
    for g in groups:
        sel = [(key, ue) for key, ue in matched if key[0] in set(g.tolist())]
        ours = np.array([res[key][0] for key, _ in sel])
        ups = np.array([up["flux_predict"][ur, zi] / up["flux_true"][ur, zi] - 1.0 for _, (ur, zi) in sel])
        by_ns[f"[{sim_ns[g].min():.4f},{sim_ns[g].max():.4f}]"] = dict(ours=med_rms(ours), upstream=med_rms(ups),
                                                                      n_sims=int(g.size))
    out["R1_C4_by_heldout_ns"] = by_ns

    # ---------------- R2: per-mode vs KS-interpolated, same entries
    lo, hi = float(kks.min()), float(kks.max())
    per_mode, interp, z_pm, z_ks = [], [], {}, {}
    for key, _ in matched:
        n, i = key
        e = evals[n]
        k_row = kfkms[int(e["rows"][i])]
        m = np.isfinite(k_row) & (k_row >= lo) & (k_row <= hi) & np.isfinite(e["res_tot"][i])
        z = str(round(float(ent[key]["z"]), 1))
        per_mode.append(e["res_tot"][i][m]); interp.append(res[key][0])
        z_pm.setdefault(z, []).append(e["res_tot"][i][m]); z_ks.setdefault(z, []).append(res[key][0])
    out["R2_per_mode_vs_interpolated"] = dict(
        k_span_skm=[lo, hi], per_mode=med_rms(np.concatenate(per_mode)), interpolated=med_rms(np.concatenate(interp)),
        per_z={z: dict(per_mode=med_rms(np.concatenate(z_pm[z])), interpolated=med_rms(np.concatenate(z_ks[z])))
               for z in sorted(z_pm, key=float)})
    json.dump(out, open(a.out_json, "w"), indent=1)

    for f, v in folds.items():
        print(f"fold {f}: n_s [{v['ns_min']:.4f}, {v['ns_max']:.4f}] ({v['n_sims']} sims)  RMS % "
              + "/".join(f"{100 * x:.3f}" for x in v["class_rms"].values())
              + f"  median |total| {100 * v['median_abs_total']:.3f}%")
    for g, v in by_ns.items():
        print(f"C4 n_s {g}: ours median {100 * v['ours']['median']:.3f}% RMS {100 * v['ours']['rms']:.3f}%; upstream "
              f"{100 * v['upstream']['median']:.3f}% / {100 * v['upstream']['rms']:.3f}% (n {v['ours']['n']})")
    r2 = out["R2_per_mode_vs_interpolated"]
    print(f"R2 per-mode median {100 * r2['per_mode']['median']:.3f}% RMS {100 * r2['per_mode']['rms']:.3f}% "
          f"(n {r2['per_mode']['n']}) vs interpolated {100 * r2['interpolated']['median']:.3f}% / "
          f"{100 * r2['interpolated']['rms']:.3f}% (n {r2['interpolated']['n']})")
    for z, v in r2["per_z"].items():
        print(f"  z {z}: per-mode {100 * v['per_mode']['median']:.3f}% interpolated {100 * v['interpolated']['median']:.3f}%")


if __name__ == "__main__":
    main()
