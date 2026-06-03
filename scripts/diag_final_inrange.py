"""IN-RANGE per-fold A_p/ns Fisher-bias + deployed median + coherent low-k tilt for
the FINALIZED LF emulator (de-bias w_coh=80 + data-range down-weight).

Loads the 8 production checkpoints `checkpoints/final_fold{0..7}` (trained by
scripts/run_loso_sweep.py with the finalized recipe) and scores each, RESTRICTED to
the DESI DATA RANGE (z∈[2.2,4.6], k≥1e-3 s/km):

  * A_p & n_s Fisher-bias at the fiducial z=3.0 slice, k≥1e-3 (reuses
    diag_datarange_mask.build_fisher_rows + fisher_project + slice_masks);
  * deployed median |P̂/P−1| per class over all DATA-RANGE val bins;
  * the coherent low-k tilt vs the CV floor (in-range z), reusing
    diag_datarange_mask.coherent_tilt + band_rms + load_cv_band.

Reports per fold + mean±scatter, and confirms vs the pre-finalization numbers
(A_p in-range RMS ~0.12σ, ns RMS ≤0.13σ, gate failures ≤1/8, clean deployed
median <1%). READ-ONLY on production code.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_final_inrange.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import (
    load_cache, make_splits, safe_log, datarange_mask, cell_id, DATA_RANGE,
)
from hcd_analysis.emulator import train as T

# reuse the validated, apples-to-apples scoring from the data-range diagnostic
from scripts.diag_datarange_mask import (
    build_fisher_rows, fisher_project, slice_masks, coherent_tilt,
    load_cv_band, band_rms, cv_band_value, load_per_class_floor,
    CLS, PARAMS, Z_DATA, K_DATA_MIN, LOWK, MIDK, HIGHK,
)

CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
CKPT = "/home/mfho/hcd_priya/checkpoints/final_fold{}"
Z_FID = 3.0


def deployed_median_inrange(model, d, va, norm):
    """Deployed median |P̂/P−1| per class over the DATA-RANGE val bins (z∈[2.2,4.6],
    k≥1e-3). The deployed (in-range) median the PI tracks."""
    pf = norm["P_filt"]; sc = pf["sig_cosmo"]
    x = jnp.asarray(d["x"][va]); tau = jnp.asarray(d["tau0"][va])
    pred = jax.vmap(model)(x, tau)
    base = np.asarray(pred["P_filt_base"]) * pf["sig_marg"] + pf["mu_marg"]
    logP_hat = base + sc * np.asarray(pred["P_filt_resid"])
    logP_true = safe_log(d["P_filt"][va])
    frac = np.abs(np.exp(logP_hat - logP_true) - 1.0)
    fin = np.isfinite(logP_true)
    in_range = datarange_mask(d)[va]                       # (Nval,K)
    keep = fin & in_range[:, None, :]
    out = {}
    for ci, nm in enumerate(CLS):
        f = frac[:, ci, :][keep[:, ci, :]]
        out[nm] = float(np.median(f)) if f.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=8)
    args = ap.parse_args()

    d = load_cache(CACHE)
    n_k = d["P_tier_p"].shape[1]
    kf = np.nanmedian(np.where(np.isfinite(d["kfkms"]), d["kfkms"], np.nan), 0)
    cv = load_cv_band()
    per_class_floor = load_per_class_floor()

    perfold = {}
    coh0 = None
    for fold in range(args.folds):
        ckpt = CKPT.format(fold)
        if not Path(ckpt + ".eqx").exists():
            print(f"  [skip] fold {fold}: no checkpoint {ckpt}.eqx")
            continue
        model, meta, norm = T.load_checkpoint(ckpt)
        tr, va, ho = make_splits(d, fold, n_folds=args.folds, holdout_frac=0.15)

        # IN-RANGE Fisher (z=3.0 slice, k≥1e-3) — reuse the apples-to-apples scoring.
        rows = build_fisher_rows(model, d, va, norm)
        masked = fisher_project(rows, slice_masks(rows, Z_FID, mask_k=True))
        full = fisher_project(rows, slice_masks(rows, Z_FID, mask_k=False))

        med = deployed_median_inrange(model, d, va, norm)
        perfold[fold] = dict(
            ap_inrange=masked["ap_bias_sigma"], ns_inrange=masked["ns_bias_sigma"],
            ap_full=full["ap_bias_sigma"], ns_full=full["ns_bias_sigma"],
            ap_dtheta_inrange=masked["ap_dtheta_unit"],
            ns_dtheta_inrange=masked["ns_dtheta_unit"],
            sigma_ap=masked["sigma_fisher"]["Ap"], cond=masked["fisher_cond"],
            n_modes=masked["n_modes"], median=med)
        print(f"  fold {fold}: A_p(in)={masked['ap_bias_sigma']:+.3f}σ "
              f"ns(in)={masked['ns_bias_sigma']:+.3f}σ  "
              f"clean med={med['clean']*100:.2f}%  modes={masked['n_modes']}")
        if fold == 0:
            coh0 = coherent_tilt(model, d, va, norm, kf)

    # ---- mean±scatter over folds ----
    def stat(key):
        v = np.array([perfold[f][key] for f in perfold], float)
        v = v[np.isfinite(v)]
        return (float(np.mean(v)), float(np.std(v)),
                float(np.sqrt(np.mean(v ** 2))), float(np.min(v)), float(np.max(v)))

    def stat_med(nm):
        v = np.array([perfold[f]["median"][nm] for f in perfold], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)), float(np.mean(v)), float(np.max(v))

    ap_m, ap_s, ap_rms, ap_lo, ap_hi = stat("ap_inrange")
    ns_m, ns_s, ns_rms, ns_lo, ns_hi = stat("ns_inrange")
    gate_fail = sum(
        1 for f in perfold
        if abs(perfold[f]["ap_inrange"]) > 0.2 or abs(perfold[f]["ns_inrange"]) > 0.2)

    print("\n===== IN-RANGE (z∈[2.2,4.6], k≥1e-3) per-fold A_p / ns Fisher-bias =====")
    print(f"  {'fold':5} {'A_p σ':>8} {'ns σ':>8} {'clean med%':>11} {'gate':>6}")
    for f in sorted(perfold):
        r = perfold[f]
        gate = "FAIL" if (abs(r["ap_inrange"]) > 0.2 or abs(r["ns_inrange"]) > 0.2) else "ok"
        print(f"  {f:5} {r['ap_inrange']:+8.3f} {r['ns_inrange']:+8.3f} "
              f"{r['median']['clean']*100:11.2f} {gate:>6}")
    print(f"\n  A_p in-range: mean {ap_m:+.3f}σ  RMS {ap_rms:.3f}σ  scatter {ap_s:.3f}σ  "
          f"[{ap_lo:+.3f},{ap_hi:+.3f}]")
    print(f"  ns  in-range: mean {ns_m:+.3f}σ  RMS {ns_rms:.3f}σ  scatter {ns_s:.3f}σ  "
          f"[{ns_lo:+.3f},{ns_hi:+.3f}]")
    print(f"  gate failures (|A_p| or |ns| > 0.2σ): {gate_fail}/{len(perfold)}")
    print("  deployed IN-RANGE median |P̂/P−1| per class (median / mean / max over folds):")
    for nm in CLS:
        mm, me, mx = stat_med(nm)
        print(f"    {nm:7}: {mm*100:.2f}% / {me*100:.2f}% / {mx*100:.2f}%")

    # ---- coherent low-k tilt vs CV (fold 0, in-range z) ----
    tilt = {}
    print("\n===== COHERENT LOW-k TILT vs CV floor (fold 0, in-range z, k≥1e-3) =====")
    for ci, nm in enumerate(CLS):
        rm, mm, nk = band_rms(coh0["coh_inz"], kf, ci, LOWK, k_floor=K_DATA_MIN)
        cvf = cv_band_value(cv, LOWK); pcf = per_class_floor.get(nm, float("nan"))
        verdict = "ABOVE CV" if rm > cvf else "within CV"
        tilt[nm] = dict(lowk_rms=rm, lowk_mean=mm, cv_floor=cvf,
                        per_class_floor=pcf, nk=nk, verdict=verdict)
        print(f"    {nm:7} low-k: coherent rms={rm*100:5.2f}% mean={mm*100:+5.2f}%  "
              f"({verdict} {cvf*100:.2f}%) [class floor {pcf*100:.2f}%] (nk {nk})")

    out = {
        "data_range": {"z": list(Z_DATA), "k_min": K_DATA_MIN},
        "perfold": {str(f): perfold[f] for f in perfold},
        "summary": {
            "ap_inrange": {"mean": ap_m, "rms": ap_rms, "scatter": ap_s,
                           "min": ap_lo, "max": ap_hi},
            "ns_inrange": {"mean": ns_m, "rms": ns_rms, "scatter": ns_s,
                           "min": ns_lo, "max": ns_hi},
            "gate_failures": gate_fail, "n_folds": len(perfold),
            "deployed_median_inrange": {nm: dict(zip(("median", "mean", "max"),
                                                     stat_med(nm))) for nm in CLS},
        },
        "coherent_lowk_tilt_fold0": tilt,
    }
    jpath = f"{OUT}/final_inrange_metrics.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
