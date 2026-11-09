#!/usr/bin/env python3
"""Gate D read-outs requested by the blind review (PU-0063/PU-0064), computed from the frozen gate D arrays (no
re-measurement): (1) D3 per held-out simulation and on a common mode-index band (``gate_d.d3_per_simulation``);
(2) R-D1 k-resolved: the all-6 correction g against k_skm at a representative theta (the HR simulation nearest the
box centre) per class at z 2.2, 3.0, 4.6, mean over mean-flux rungs, plus the spread over the 6 simulations.
Writes NEW files only (refuses to overwrite).

Usage: PYTHONPATH=<repo> python3 scripts/report_gate_d_review.py --gate-d-dir DIR
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, _REPO_ROOT)
from hcd_analysis.emulator import gate_d as GD  # noqa: E402
from hcd_analysis.emulator import mf_modes as MM  # noqa: E402
from hcd_analysis.emulator.data import load_cache  # noqa: E402

LF = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
K_REPORT = (0.002, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06)
Z_REPORT = (2.2, 3.0, 4.6)


def _write(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    json.dump(obj, open(path, "w"), indent=1)


def rd1_k_resolved(a, x_unit):
    sims = sorted(set(a["sim"].astype(str)))
    rep = min(sims, key=lambda s: float(np.linalg.norm(x_unit[a["sim"] == s][0, :9] - 0.5)))
    z = np.round(a["z"], 1)
    out = {"k_skm": list(K_REPORT), "representative": {"sim": rep, "table": {}}, "spread_over_sims": {}}
    for c, name in enumerate(GD.CLS):
        out["representative"]["table"][name] = {}
        out["spread_over_sims"][name] = {}
        for zz in Z_REPORT:
            per_sim = {}
            for s in sims:
                rows = np.where((a["sim"] == s) & (z == zz))[0]
                if rows.size == 0:
                    continue
                vals = []
                for r in rows:
                    k, g = a["k_hr"][r], a["g_all"][r, c]
                    m = np.isfinite(k) & np.isfinite(g)
                    vals.append(np.where((np.array(K_REPORT) >= k[m].min()) & (np.array(K_REPORT) <= k[m].max()),
                                         np.interp(K_REPORT, k[m], g[m]), np.nan))
                per_sim[s] = np.nanmean(np.array(vals), axis=0)
            if rep in per_sim:
                out["representative"]["table"][name][str(zz)] = [float(v) for v in per_sim[rep]]
            arr = np.array(list(per_sim.values()))
            out["spread_over_sims"][name][str(zz)] = dict(min=[float(v) for v in np.nanmin(arr, 0)],
                                                          max=[float(v) for v in np.nanmax(arr, 0)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-d-dir", required=True)
    a_ = ap.parse_args()
    a = np.load(f"{a_.gate_d_dir}/gate_d_arrays.npz", allow_pickle=True)
    l_mf, l_lf = np.log1p(a["r_mf"]), np.log1p(a["r_lf"])
    keep = a["in_range"][:, None, :] & np.isfinite(a["r_mf"]) & np.isfinite(a["r_lf"])
    d3 = GD.d3_per_simulation(l_mf, l_lf, a["sim"], a["z"], a["k_hr"], keep)
    lf, hr = load_cache(LF), load_cache(HR)
    l_rows = np.array([p[1] for p in MM.match_pairs(lf, hr)])
    assert np.allclose(np.asarray(lf["z_grid"])[l_rows], a["z"])
    rd1 = rd1_k_resolved(a, np.asarray(lf["x"])[l_rows])
    _write(f"{a_.gate_d_dir}/gate_d_d3_per_simulation.json", d3)
    _write(f"{a_.gate_d_dir}/gate_d_rd1_k_resolved.json", rd1)
    for c in GD.CLS:
        print(f"D3 {c}: pooled registered MF {d3['pooled_registered_band'][c]['mf']:+.2e} LF "
              f"{d3['pooled_registered_band'][c]['lf']:+.2e}; common-mode-band MF {d3['pooled_common_mode_band'][c]['mf']:+.1e}; "
              f"RMS over sims MF {d3['rms_over_simulations'][c]['mf']:.3e} LF {d3['rms_over_simulations'][c]['lf']:.3e}")
    print("first common mode:", d3["common_mode_band_first_mode"])
    for s, v in d3["per_simulation"].items():
        print(f"  {s[:10]}: " + "  ".join(f"{c} {100 * v[c]['mf']:+.2f}/{100 * v[c]['lf']:+.2f}%" for c in GD.CLS))
    print("R-D1 representative:", rd1["representative"]["sim"][:24])
    for zz in Z_REPORT:
        print(f"  clean z {zz}: " + " ".join(f"{100 * v:+.1f}" for v in rd1["representative"]["table"]["clean"][str(zz)])
              + "  (spread k<=0.005: " + f"{100 * rd1['spread_over_sims']['clean'][str(zz)]['min'][0]:+.1f}.."
              f"{100 * rd1['spread_over_sims']['clean'][str(zz)]['max'][0]:+.1f}%)")


if __name__ == "__main__":
    main()
