#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 section 1, step 3: the T1 regularization selection by 10-fold simulation CV on the 60-fold
leave-one-simulation-out ensemble residuals (gate C). Builds no product. Writes, write-once: the selection summary
(--out-json, private notes) and the per-simulation scores (--out-npz, private storage).

Rows: every cache row of every simulation at the 13 data z cells (2.2-4.6), all 20 tau0 rungs. Scored modes per z
cell: the union over the production legs (DESI, KS k <= 0.065, eBOSS) of the modes bracketing the leg's kept bins at
that z over the whole sampling box. k band of a mode: its physical k at the box centre (< 0.01, 0.01-0.031,
0.031-0.050, >= 0.050 s/km). Class coefficients: production, main arm (DLA masked), the row's own w_c.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/run_gate_e_t1_select.py --eval-dir <gateC_eval> --out-json <..> --out-npz <..>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.data import (PARAM_LIMITS, load_cache, make_tau0_bands, sampling_unit_bounds,
                                        tau0_ladder_factor)
from hcd_analysis.emulator.kcoord import k_skm_from_kcom
from hcd_analysis.emulator.schema import L_BOX_HMPC

H_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)          # amendment A1 rev 1 section 1 (analyst choice)
SZ_GRID = (0.0, 0.2)
Z_CELLS = np.round(np.arange(2.2, 4.61, 0.2), 1)
K_BANDS = (0.01, 0.031, 0.050)
I_HUB, I_OMH2 = 5, 6


def candidates():
    return [(h, sz, pooled) for pooled in (True, False) for h in H_GRID for sz in SZ_GRID]


def legs():
    return [DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
            DL.load_ks_leg(resolution_float=True, k_max=0.065),
            DL.load_eboss_leg(resolution_float=True, resolution_coherent=False)]


def mode_masks(k_com, leg_list, lo, hi):
    """(n_z, K) scored modes: union over legs of the modes bracketing the leg's kept bins at the z cell over the box."""
    K = k_com.size
    mask = np.zeros((Z_CELLS.size, K), bool)
    for leg in leg_list:
        keep = np.isfinite(np.asarray(leg.P_data))
        for iz, z in enumerate(np.asarray(leg.z, float)):
            j = np.where(np.abs(Z_CELLS - z) < 0.05)[0]
            r = np.where((np.asarray(leg.z_idx) == iz) & keep)[0]
            if r.size == 0:
                continue
            if j.size != 1:
                raise ValueError(f"{leg.name} z {z} is not a data z cell")
            k = np.asarray(leg.k)[r]
            first, last = CB.bracket_modes(k_com, float(z), float(k.min()), float(k.max()), lo, hi)
            mask[j[0], first - 1:last] = True
    return mask


def kband_of_modes(k_com, lo, hi):
    lim = np.asarray(PARAM_LIMITS, float)
    c = 0.5 * (np.asarray(lo) + np.asarray(hi))
    hub = lim[I_HUB, 0] + c[I_HUB] * np.ptp(lim[I_HUB])
    om = lim[I_OMH2, 0] + c[I_OMH2] * np.ptp(lim[I_OMH2])
    return np.stack([np.searchsorted(K_BANDS, np.asarray(k_skm_from_kcom(k_com, float(z), hub, om)), side="right")
                     for z in Z_CELLS])


def build_t1_data(d, R, mask, kband):
    z = np.asarray(R["z"], float)
    zc = np.full(z.size, -1)
    for i, zz in enumerate(Z_CELLS):
        zc[np.abs(z - zz) < 0.05] = i
    keep = zc >= 0
    rows = np.asarray(R["rows"])[keep]
    tau0 = np.asarray(d["tau0"], float)[rows]
    zr = np.asarray(d["z_grid"], float)[rows]
    band, centres = make_tau0_bands(tau0, zr, 4)
    return CB.T1Data(r=np.asarray(R["r"])[keep], P=np.asarray(d["P_filt"], float)[rows],
                     coef=CB.class_coef(np.asarray(d["w_c_cache"], float)[rows], masked=True),
                     alpha=tau0_ladder_factor(tau0, zr), band=band, centres=centres, zc=zc[keep],
                     sim=np.asarray(R["sim"])[keep], z_cells=Z_CELLS.astype(float), mask=mask, kband=kband), rows


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--sims", default=None, help="comma-separated held-out simulation indices (smoke runs only)")
    a = ap.parse_args(argv)
    for p in (a.out_json, a.out_npz):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    d = load_cache(a.cache)
    k_com = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC
    np.testing.assert_allclose(np.asarray(d["k_com_hmpc"]), k_com, rtol=1e-13)
    lo, hi = sampling_unit_bounds()
    mask = mode_masks(k_com, legs(), lo, hi)
    kband = kband_of_modes(k_com, lo, hi)
    R = CB.load_loo_ensemble_residuals(a.eval_dir, d, sims=None if a.sims is None else [int(s) for s in a.sims.split(",")])
    T, rows = build_t1_data(d, R, mask, kband)
    cands = candidates()
    cv = CB.t1_cv(T, cands, n_folds=10)
    sel = CB.t1_select(cv, cands, n_boot=a.n_boot, seed=0)
    raw_i = cands.index((0.0, 0.0, cands[sel["chosen"]][2]))
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), code_commit=commit, cache=a.cache, cache_sha256=_sha(a.cache),
        eval_dir=a.eval_dir, n_rows=int(rows.size), n_sims=len(cv["sims"]), z_cells=Z_CELLS.tolist(),
        tau0_centres=T.centres.tolist(), modes_scored_per_z=mask.sum(axis=1).tolist(),
        candidates=[list(c) for c in cands], chosen=list(cands[sel["chosen"]]), tau0_rule=sel["tau0"],
        tau0_gain=sel["tau0_gain"], tau0_gain_se=sel["tau0_gain_se"],
        best_banded=list(cands[sel["best_banded"]]), best_pooled=list(cands[sel["best_pooled"]]),
        h_of_z=sel["h_of_z"], guard_triggered_z=[float(Z_CELLS[i]) for i in sel["guard_triggered"]],
        mean_score_per_sim=dict(zip([str(c) for c in cands], (np.asarray(sel["means"])).tolist())),
        chosen_minus_raw=dict(mean=float(np.mean(cv["main"][:, sel["chosen"]] - cv["main"][:, raw_i])),
                              paired_se=CB.paired_se(cv["main"][:, sel["chosen"]], cv["main"][:, raw_i], a.n_boot, 0)),
        per_class_mean_score=dict(zip([str(c) for c in cands], cv["per_class"].mean(axis=0).tolist())),
        stop=(sel["tau0"] == "banded"),
        stop_reason=("banded tau0 T1 beats pooled by > 2 paired SE: STOP for a PI ruling (amendment A1 rev 1 section 1)"
                     if sel["tau0"] == "banded" else None),
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out_npz)), exist_ok=True)
    np.savez(a.out_npz, main=cv["main"], cell=cv["cell"], per_class=cv["per_class"], sims=np.array(cv["sims"]),
             candidates=np.array(cands, dtype=float), mask=mask, kband=kband, rows=rows)
    out["scores_npz"] = a.out_npz
    out["scores_npz_sha256"] = _sha(a.out_npz)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("chosen", "tau0_rule", "h_of_z", "guard_triggered_z", "stop")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
