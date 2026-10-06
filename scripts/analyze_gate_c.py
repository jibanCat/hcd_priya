#!/usr/bin/env python3
"""Gate C criteria C1-C5 and reports R1-R2 from the per-checkpoint evaluations of validate_gate_c.py
(spec GATE_C_SPEC.md in the notes repository; criteria registered in PU-0043 before evaluation).

Inputs in --eval-dir: eval_loso8_<tag>_fold{F}.npz (tag repaired | hist), eval_loo60_s{NN}.npz,
eval_prod_repaired_seed{S}.npz; --fisher-json: the ported historical Fisher diagnostic's output for the repaired
8-fold models. Writes --out-json (every number) and prints a markdown summary.

Usage: PYTHONPATH=<repo> python3 scripts/analyze_gate_c.py --eval-dir DIR --fisher-json J --out-json OUT
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import glob
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, _REPO_ROOT)
from hcd_analysis.emulator import gate_c as G  # noqa: E402

CLS = ("clean", "LLS", "subDLA", "DLA")
HIST_C2 = {"clean": 0.01048, "LLS": 0.01081, "subDLA": 0.01288, "DLA": 0.01878}     # GATE_C_SPEC C2
HIST_C5 = {"clean": 0.0054, "LLS": 0.0059, "subDLA": 0.0079, "DLA": 0.0206}         # GATE_C_SPEC C5
UP_LOO = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/loo_fps.hdf5"
F1_SIM = "ns0.959Ap2.34e-09herei3.81heref2.99alphaq1.77hub0.725omegamh20.144hireionz6.83bhfeedback0.0467"
CACHE = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def load(path):
    e = np.load(path, allow_pickle=True)
    return {k: e[k] for k in e.files}


def keep_of(e, rows_sel):
    return e["in_range"][rows_sel] & np.isfinite(e["res_tot"][rows_sel])


def fold_metrics(e):
    v = ~e["is_tau0_edge"]
    keep = keep_of(e, v)
    rms = G.class_rms(e["res_cls"][v], keep)
    med = [float(np.nanmedian(np.abs(np.where(keep, e["res_cls"][v][:, c], np.nan)))) for c in range(4)]
    return rms, np.array(med)


def strat(e, sel, keep, by, edges=None):
    """Median |residual| (total P1D, per mode) stratified by a per-row variable."""
    out = {}
    vals = by[sel]
    groups = np.unique(np.round(vals, 3)) if edges is None else range(len(edges) - 1)
    r = np.where(keep, np.abs(e["res_tot"][sel]), np.nan)
    for g in groups:
        m = (np.round(vals, 3) == g) if edges is None else ((vals >= edges[g]) & (vals < edges[g + 1]))
        if m.any():
            key = float(g) if edges is None else f"[{edges[g]:.3g},{edges[g + 1]:.3g})"
            out[str(key)] = dict(median=float(np.nanmedian(r[m])), rms=float(np.sqrt(np.nanmean(r[m] ** 2))), n_rows=int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--fisher-json", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R = {"criteria": {}, "reports": {}}

    # ---------------- C1 coordinate agreement over every evaluated row of every protocol
    allf = sorted(glob.glob(f"{a.eval_dir}/eval_*.npz"))
    c1 = max(float(np.nanmax(load(p)["coord"])) for p in allf)
    R["criteria"]["C1"] = dict(max_rel=c1, threshold=1e-12, pass_=c1 <= 1e-12, n_files=len(allf))

    # ---------------- C2 8-fold historical protocol, repaired cache (and the same code on the historical cache)
    for tag in ("repaired", "hist"):
        files = sorted(glob.glob(f"{a.eval_dir}/eval_loso8_{tag}_fold*.npz"))
        per = [fold_metrics(load(p)) for p in files]
        rms = np.array([p[0] for p in per]); med = np.array([p[1] for p in per])
        R["reports"][f"loso8_{tag}_perfold_rms"] = rms.tolist()
        R["reports"][f"loso8_{tag}_perfold_median_abs"] = med.tolist()
        if tag == "repaired":
            mrms, mmed = np.median(rms, 0), np.median(med, 0)
            ok = all(mrms[c] <= HIST_C2[n] + 0.002 for c, n in enumerate(CLS)) and all(mmed < 0.01)
            R["criteria"]["C2"] = dict(n_folds=len(files), median_rms=dict(zip(CLS, mrms.tolist())),
                                       historical=HIST_C2, tolerance_pp=0.2,
                                       median_abs=dict(zip(CLS, mmed.tolist())), pass_=bool(ok and len(files) == 8))
    # ---------------- C3 Fisher-bias (ported historical diagnostic)
    fj = json.load(open(a.fisher_json))
    nf = int(fj["summary"]["gate_failures"]); nfo = int(fj["summary"]["n_folds"])
    R["criteria"]["C3"] = dict(gate_failures=nf, n_folds=nfo, ap=fj["summary"]["ap_inrange"], ns=fj["summary"]["ns_inrange"],
                               pass_=bool(nf <= 1 and nfo == 8))

    # ---------------- C4 upstream-matched leave-one-simulation-out vs upstream's LOO product
    with h5py.File(a.cache, "r") as f:
        sims, prm = f["sim_name"].asstr()[...], f["params"][...]
    sim_params = {s: prm[i] for i, s in enumerate(sims)}
    with h5py.File(UP_LOO, "r") as f:
        up = {k: f[k][...] for k in ("flux_predict", "flux_true", "params", "zout")}
    ours_e, ups_e, zs_e, entries = [], [], [], []
    for p in sorted(glob.glob(f"{a.eval_dir}/eval_loo60_s*.npz")):
        e = load(p)
        for i in range(e["rows"].size):
            if not np.isfinite(e["res_ks"][i]).all():
                continue
            entries.append(dict(params=sim_params[str(e["sim_name"][i])], alpha=float(e["alpha"][i]), z=float(e["z"][i]),
                                key=(p, i)))
    match = G.match_upstream_loo(entries, up["params"], up["zout"])
    cache_ev = {}
    for (p, i), (ur, zi) in match.items():
        e = cache_ev.setdefault(p, load(p))
        if str(e["sim_name"][i]) == F1_SIM and abs(float(e["z"][i]) - 2.2) < 1e-6:
            continue
        ours_e.append(e["res_ks"][i]); zs_e.append(float(e["z"][i]))
        ups_e.append(up["flux_predict"][ur, zi] / up["flux_true"][ur, zi] - 1.0)
    ours_e, ups_e, zs_e = np.array(ours_e), np.array(ups_e), np.array(zs_e)
    o_med, o_rms = float(np.median(np.abs(ours_e))), float(np.sqrt(np.mean(ours_e ** 2)))
    u_med, u_rms = float(np.median(np.abs(ups_e))), float(np.sqrt(np.mean(ups_e ** 2)))
    perz = {}
    for z in np.unique(np.round(zs_e, 1)):
        m = np.round(zs_e, 1) == z
        om, um = float(np.median(np.abs(ours_e[m]))), float(np.median(np.abs(ups_e[m])))
        perz[str(z)] = dict(ours_median=om, upstream_median=um, flag=om > 1.25 * um, n=int(m.sum()))
    R["criteria"]["C4"] = dict(n_entries=int(ours_e.shape[0]), ours_median=o_med, ours_rms=o_rms, upstream_median=u_med,
                               upstream_rms=u_rms, per_z=perz, pass_=bool(o_med <= u_med and o_rms <= u_rms))

    # ---------------- C5 production ensemble on its row-val rows
    files = sorted(glob.glob(f"{a.eval_dir}/eval_prod_repaired_seed*.npz"))
    ev = [load(p) for p in files]
    if ev:
        assert all(np.array_equal(ev[0]["rows"], e["rows"]) for e in ev)
        keep = keep_of(ev[0], slice(None))
        ens = G.ensemble_residual([e["res_cls"] for e in ev])
        ens_rms = G.class_rms(ens, keep)
        mem = [G.class_rms(e["res_cls"], keep).tolist() for e in ev]
        ok = all(ens_rms[c] <= HIST_C5[n] + 0.002 for c, n in enumerate(CLS)) and ens_rms.max() < 0.05
        R["criteria"]["C5"] = dict(n_members=len(ev), ensemble_rms=dict(zip(CLS, ens_rms.tolist())), members_rms=mem,
                                   historical=HIST_C5, tolerance_pp=0.2, pass_=bool(ok and len(ev) == 5))

    # ---------------- R1 stratification (8-fold repaired: in-range val rows; tau0-edge rows separately; 60-fold)
    st = {}
    for tag, pat in (("loso8_repaired", "eval_loso8_repaired_fold*.npz"), ("loo60", "eval_loo60_s*.npz")):
        es = [load(p) for p in sorted(glob.glob(f"{a.eval_dir}/{pat}"))]
        if not es:
            continue
        cat = {k: np.concatenate([e[k] for e in es]) for k in ("res_tot", "res_cls", "z", "alpha", "is_tau0_edge",
                                                                "in_range", "theta_unit")}
        v = ~cat["is_tau0_edge"]
        keep = cat["in_range"][v] & np.isfinite(cat["res_tot"][v])
        face = np.min(np.minimum(cat["theta_unit"], 1 - cat["theta_unit"]), axis=1)
        st[tag] = dict(by_z=strat(cat, v, keep, cat["z"]),
                       by_alpha_band=strat(cat, v, keep, cat["alpha"], edges=[0.0, 0.70, 0.80, 1.20, 1.30, 2.0]),
                       by_distance_to_box_face=strat(cat, v, keep, face, edges=[0.0, 0.05, 0.1, 0.2, 0.5]),
                       by_class_median_abs=dict(zip(CLS, [float(np.nanmedian(np.abs(np.where(keep, cat["res_cls"][v][:, c], np.nan))))
                                                          for c in range(4)])))
        kb = es[0]["k_com_hmpc"]
        r = np.where(keep, np.abs(cat["res_tot"][v]), np.nan)
        st[tag]["by_mode_band"] = {f"modes {lo + 1}-{hi}": float(np.nanmedian(r[:, lo:hi])) for lo, hi in
                                   ((0, 10), (10, 40), (40, 100), (100, kb.size))}
        if cat["is_tau0_edge"].any():
            ed = cat["is_tau0_edge"]
            keep_e = np.isfinite(cat["res_tot"][ed])
            st[tag]["tau0_edge_extrapolation_rows"] = dict(
                n_rows=int(ed.sum()), median_abs_total=float(np.nanmedian(np.where(keep_e, np.abs(cat["res_tot"][ed]), np.nan))),
                by_z=strat(cat, ed, keep_e, cat["z"]))
    R["reports"]["R1"] = st
    json.dump(R, open(a.out_json, "w"), indent=1, default=float)

    print("| criterion | value | threshold | pass |\n|---|---|---|---|")
    c = R["criteria"]
    print(f"| C1 coordinate | {c['C1']['max_rel']:.2e} | 1e-12 | {c['C1']['pass_']} |")
    if "C2" in c:
        print("| C2 8-fold RMS (median over folds) | " + ", ".join(f"{n} {100 * v:.3f}%" for n, v in c["C2"]["median_rms"].items())
              + " | historical + 0.2 pp; median abs < 1% | " + str(c["C2"]["pass_"]) + " |")
    print(f"| C3 Fisher-bias | {c['C3']['gate_failures']}/{c['C3']['n_folds']} folds fail | <= 1/8 | {c['C3']['pass_']} |")
    print(f"| C4 vs upstream (n={c['C4']['n_entries']}) | ours median {100 * o_med:.3f}% RMS {100 * o_rms:.3f}%; "
          f"upstream {100 * u_med:.3f}% / {100 * u_rms:.3f}% | ours <= upstream | {c['C4']['pass_']} |")
    if "C5" in c:
        print("| C5 ensemble | " + ", ".join(f"{n} {100 * v:.3f}%" for n, v in c["C5"]["ensemble_rms"].items())
              + f" | historical + 0.2 pp, max < 5% | {c['C5']['pass_']} |")


if __name__ == "__main__":
    main()
