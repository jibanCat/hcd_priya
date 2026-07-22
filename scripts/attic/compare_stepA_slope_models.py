#!/usr/bin/env python3
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# Step-A slope-model comparison diagnostic.
"""HCD-SLOPE-MODEL SELECTION compare: 1D re-center (stepA_slfix) vs 2D amplitude×tilt (stepA_2dtilt).

For each of the 4 per-survey closure mocks, pool the 4 chains and print, per cosmology param
(n_s, A_p), the TRUTH value vs the pooled posterior mean±std of each HCD-slope model, plus which
model recovers truth closest. Pick the production HCD-slope model = whichever recovers n_s closest
to truth with the least overshoot, per survey.

Run (any env; CPU-only, no jax needed):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/compare_stepA_slope_models.py
"""
import os
import numpy as np

REPO = "/home/mfho/hcd_priya"
DIR_1D = f"{REPO}/checkpoints/stepA_slfix"     # 1D re-centered power-law
DIR_2D = f"{REPO}/checkpoints/stepA_2dtilt"    # 2D amplitude×tilt + hierarchical
MOCKS = ["D_f3", "K_f4", "XS_f6_s0", "E_f5"]   # DESI / KS / DESI+KS / eBOSS
SURVEY = {"D_f3": "DESI", "K_f4": "KS", "XS_f6_s0": "DESI+KS", "E_f5": "eBOSS"}
PARAMS = ["ns", "Ap"]
N_CHAINS = 4


def pooled(directory, mock):
    """Pool the mock's chains: return {param: (truth, mean, std)} or None if not all present."""
    paths = [f"{directory}/{mock}_c{c}.npz" for c in range(N_CHAINS)]
    if not all(os.path.exists(p) for p in paths):
        return None
    packs, names, truth = [], None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"]))
        names = list(map(str, z["names"])) if names is None else names
        truth = np.asarray(z["truth_vec"]) if truth is None else truth
    nmin = min(p.shape[0] for p in packs)
    pool = np.concatenate([p[:nmin] for p in packs], axis=0)   # (N*chains, P)
    out = {}
    for prm in PARAMS:
        if prm not in names:
            continue
        j = names.index(prm)
        out[prm] = (float(truth[j]), float(pool[:, j].mean()), float(pool[:, j].std()))
    return out


def main():
    hdr = (f"{'mock':10s} {'survey':8s} {'param':5s} {'truth':>9s} "
           f"{'1D mean':>10s} {'1D z':>7s} {'2D mean':>10s} {'2D z':>7s}  closer")
    print(hdr)
    print("-" * len(hdr))
    for mock in MOCKS:
        a = pooled(DIR_1D, mock)
        b = pooled(DIR_2D, mock)
        for prm in PARAMS:
            t = ma = sa = mb = sb = None
            if a and prm in a:
                t, ma, sa = a[prm]
            if b and prm in b:
                t2, mb, sb = b[prm]
                t = t if t is not None else t2
            # bias-z (truth - mean)/std for each model; |z| smaller = closer to truth
            za = (t - ma) / sa if (ma is not None and sa and sa > 0) else None
            zb = (t - mb) / sb if (mb is not None and sb and sb > 0) else None
            if za is not None and zb is not None:
                closer = "1D" if abs(za) < abs(zb) else ("2D" if abs(zb) < abs(za) else "tie")
            elif za is not None:
                closer = "1D(only)"
            elif zb is not None:
                closer = "2D(only)"
            else:
                closer = "PENDING"
            def f(x, w=10, p=4):
                return f"{x:>{w}.{p}f}" if x is not None else f"{'--':>{w}s}"
            print(f"{mock:10s} {SURVEY[mock]:8s} {prm:5s} {f(t,9)} "
                  f"{f(ma)} {f(za,7,2)} {f(mb)} {f(zb,7,2)}  {closer}")
    print("\nNOTE: 1D = checkpoints/stepA_slfix (re-centered 1D power-law); "
          "2D = checkpoints/stepA_2dtilt (amplitude×tilt). "
          "Pick the model with the smaller |bias z| on n_s (closest to truth, least overshoot) per survey.")


if __name__ == "__main__":
    main()
