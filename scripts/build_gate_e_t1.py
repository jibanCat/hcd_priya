#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 section 1, step 4: the T1 product from all 60 simulations with the recorded selection
(``--selection``, the step-3 JSON: smoothing h per z cell, s_z, pooled or banded), and the raw-covariance report
(R-E-T1raw, S11): per z cell the raw (h = 0) and selected per-class diagonals against mode with 68% simulation-bootstrap
bands, and the mode-to-mode correlation of the ensemble residuals across rows at lags 1, 2, 5, 10 (per class).
Write-once: the ``cemu_t1`` product (private storage), the report JSON and figure (private notes).

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/build_gate_e_t1.py --eval-dir <..> --cache <..> --selection <t1_selection.json> --out <product.npz> \
     --report-json <..> --report-fig <..>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator.data import load_cache, sampling_unit_bounds
from hcd_analysis.emulator.products import save_product
from hcd_analysis.emulator.schema import L_BOX_HMPC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gate_e_t1_select as SEL  # noqa: E402  (the step-3 data definitions: z cells, scored modes, k bands)

LAGS = (1, 2, 5, 10)
CLASSES = ("clean", "LLS", "subDLA", "DLA")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lag_correlation(r, lag):
    """Correlation across rows of r[:, c, n] and r[:, c, n + lag], averaged over n, per class."""
    a, b = r[:, :, :-lag], r[:, :, lag:]
    a = a - a.mean(axis=0); b = b - b.mean(axis=0)
    num = (a * b).mean(axis=0)
    den = np.sqrt((a ** 2).mean(axis=0) * (b ** 2).mean(axis=0))
    return np.nanmean(num / den, axis=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--report-fig", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)
    for p in (a.out, a.report_json, a.report_fig):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    sel_rec = json.load(open(a.selection))
    if sel_rec.get("stop"):
        raise SystemExit("the recorded T1 selection is a STOP; no product")
    h_sel, s_z, pooled = sel_rec["chosen"]
    d = load_cache(a.cache)
    k_com = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC
    lo, hi = sampling_unit_bounds()
    mask = SEL.mode_masks(k_com, SEL.legs(), lo, hi)
    kband = SEL.kband_of_modes(k_com, lo, hi)
    R = CB.load_loo_ensemble_residuals(a.eval_dir, d)
    T, rows = SEL.build_t1_data(d, R, mask, kband)
    cands = [tuple(c) for c in sel_rec["candidates"]]
    sel = dict(chosen=cands.index((h_sel, s_z, pooled)), h_of_z=sel_rec["h_of_z"])
    all_rows = np.arange(len(T.sim))
    rho = CB.t1_build(T, all_rows, cands, sel)                           # (n_z, B, 4, 4, K)
    raw = CB.t1_raw_cells(T, all_rows, pooled)
    # simulation bootstrap of the raw per-class diagonals (68% bands)
    sims = np.array(sorted(set(T.sim)))
    rng = np.random.default_rng(0)
    boots = []
    by_sim = {s: np.where(T.sim == s)[0] for s in sims}
    for _ in range(a.n_boot):
        pick = rng.choice(sims, sims.size, replace=True)
        idx = np.concatenate([by_sim[s] for s in pick])
        Tb = T._replace(r=T.r[idx], zc=T.zc[idx], band=T.band[idx], sim=np.arange(idx.size).astype(str))
        boots.append(np.einsum("zbccn->zcn", CB.t1_raw_cells(Tb, np.arange(idx.size), pooled)[:, :1]))
    boots = np.array(boots)
    band_lo, band_hi = np.percentile(boots, [16, 84], axis=0)
    lagc = {int(L): {c: float(v) for c, v in zip(CLASSES, lag_correlation(T.r, L))} for L in LAGS}
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    prov = dict(code_commit=commit, cache_sha256=_sha(a.cache),
                inputs={"eval_dir": a.eval_dir, "selection": a.selection, "selection_sha256": _sha(a.selection)},
                row_rule=("all 60 simulations' LOO ensemble residual rows at the 13 data z cells (2.2-4.6), all rungs; "
                          "uncentered second moment per z cell" + (" pooled over tau0" if pooled else " and tau0 band")
                          + f"; smoothing along ln mode h per z {sel_rec['h_of_z']}, along z s_z {s_z}"),
                amendment="GATE_E_AMENDMENT_A1 rev 1 section 1 (PU-0068); selection PU-0070")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_product(a.out, "cemu_t1", k_com_hmpc=k_com, provenance=prov, rho=rho, z_cells=T.z_cells,
                 alpha_centres=(np.array([1.0]) if pooled else T.centres))
    diag_raw = np.einsum("zbccn->zcn", raw[:, :1])
    diag_sel = np.einsum("zbccn->zcn", rho[:, :1])
    rep = dict(product=a.out, product_sha256=_sha(a.out), selection=sel_rec["chosen"], lag_correlation=lagc,
               z_cells=T.z_cells.tolist(),
               sel_over_raw_median_per_z={f"{z:.1f}": float(np.median(diag_sel[i, 0, mask[i]] / diag_raw[i, 0, mask[i]]))
                                          for i, z in enumerate(T.z_cells)},
               raw_outside_band_frac_per_z={f"{z:.1f}": float(np.mean((diag_sel[i, 0, mask[i]] < band_lo[i, 0, mask[i]])
                                                                        | (diag_sel[i, 0, mask[i]] > band_hi[i, 0, mask[i]])))
                                            for i, z in enumerate(T.z_cells)})
    with open(a.report_json, "w") as f:
        json.dump(rep, f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
    n = np.arange(1, 173)
    for ax, i in zip(axes.ravel(), range(T.z_cells.size)):
        m = mask[i]
        for c, col in zip(range(4), ("k", "C0", "C1", "C3")):
            ax.fill_between(n[m], np.sqrt(band_lo[i, c, m]), np.sqrt(band_hi[i, c, m]), color=col, alpha=0.2, lw=0)
            ax.plot(n[m], np.sqrt(diag_raw[i, c, m]), color=col, lw=0.6, alpha=0.7)
            ax.plot(n[m], np.sqrt(diag_sel[i, c, m]), color=col, lw=1.4, label=CLASSES[c] if i == 0 else None)
        ax.set_xscale("log"); ax.set_title(f"z {T.z_cells[i]:.1f}", fontsize=9)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[T.z_cells.size:]:
        ax.axis("off")
    fig.suptitle("T1 per-class RMS fractional residual vs mode: raw (thin), selected (thick), 68% simulation bootstrap "
                 f"(band); selection h {sel_rec['h_of_z'][0]}, s_z {s_z}, {'pooled' if pooled else 'banded'} tau0")
    fig.supxlabel("mode n"); fig.supylabel("sqrt(rho_cc)")
    fig.tight_layout()
    fig.savefig(a.report_fig, dpi=110)
    print(json.dumps({"product_sha256": rep["product_sha256"], "lag1": lagc[1]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
