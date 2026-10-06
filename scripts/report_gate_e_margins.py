#!/usr/bin/env python3
"""Gate E criterion E3(a) read-out (GATE_E_SPEC v1): per leg and z, the kept data bins' k range against the simulated
mode range over the whole sampling box, [max_theta k_skm,1, min_theta k_skm,K] (kcoord.kbounds_over_box), as margins
k_min / k1_max and kK_min / k_max (both > 1 means every kept bin is inside for every theta). Production legs and cuts.

Usage: PYTHONPATH=<repo> python3 scripts/report_gate_e_margins.py --out-json OUT
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
from hcd_analysis.emulator import data_likelihood as DL  # noqa: E402
from hcd_analysis.emulator import kcoord as KC  # noqa: E402
from hcd_analysis.emulator.data import sampling_unit_bounds  # noqa: E402
from hcd_analysis.emulator.schema import L_BOX_HMPC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out_json):
        raise SystemExit(f"refusing to overwrite {a.out_json}")
    lo, hi = sampling_unit_bounds()
    k_com = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC
    legs = [DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
            DL.load_ks_leg(resolution_float=True, k_max=0.065),
            DL.load_eboss_leg(resolution_float=True, resolution_coherent=False)]
    out = {}
    for leg in legs:
        keep = np.isfinite(np.asarray(leg.P_data))
        rows_out = {}
        for iz, z in enumerate(leg.z):
            r = np.where((np.asarray(leg.z_idx) == iz) & keep)[0]
            if r.size == 0:
                continue
            k = np.asarray(leg.k)[r]
            k1, kK = KC.kbounds_over_box(k_com, float(z), np.asarray(lo), np.asarray(hi))
            rows_out[f"{float(z):.2f}"] = dict(k_min=float(k.min()), k_max=float(k.max()), k1_max_box=k1, kK_min_box=kK,
                                               low_margin=float(k.min() / k1), high_margin=float(kK / k.max()))
        out[leg.name] = rows_out
    json.dump(out, open(a.out_json, "w"), indent=1)
    for name, rows in out.items():
        lowm = min(v["low_margin"] for v in rows.values()); him = min(v["high_margin"] for v in rows.values())
        zl = min(rows, key=lambda z: rows[z]["low_margin"]); zh = min(rows, key=lambda z: rows[z]["high_margin"])
        print(f"{name}: smallest low margin {lowm:.4f} (z {zl}), smallest high margin {him:.4f} (z {zh})")


if __name__ == "__main__":
    main()
