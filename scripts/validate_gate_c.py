#!/usr/bin/env python3
"""Gate C held-out evaluation in PHYSICAL coordinates (spec GATE_C_SPEC.md in the notes repository, sections 2-4).

For every held-out row of one checkpoint: the prediction comes through the production API
``predict.predict_on_physical_grid`` (k_skm derived from the query (z, theta)); the truth is the held-out simulation's
own source product in the cache (per-class P_filt, total P_tier_p) on the row's header-exact grid ``kfkms``. Writes one
npz per checkpoint with per-row coordinate agreement (C1), per-mode residuals per class and for the total, and the
total P1D residual after interpolating BOTH sides onto fixed physical grids (upstream's 11 KODIAQ-SQUAD bins and the
eBOSS DR14 bins), so a coordinate error would appear as an amplitude error.

Protocols (which rows are held out):
  loso8  checkpoint ``..._fold{F}`` of run_loso_sweep.py: the fold's validation rows plus the fold's simulations'
         tau0-edge rows (never trained on; an extrapolation in mean flux, scored separately)
  loo60  checkpoint trained with train_production_emulator.py --holdout-sim S: every row of S
  prod   a production member: its row-val rows

Usage (from the checkout root):
  PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu python3 scripts/validate_gate_c.py \
      --protocol loso8 --ckpt checkpoints/gateC/loso8_fold3 --out <dir>/eval_loso8_fold3.npz
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import re
import sys
import time

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, _REPO_ROOT)
import hcd_analysis.emulator  # noqa: F401,E402  (x64 before jax arrays)
from hcd_analysis.emulator import data as D  # noqa: E402
from hcd_analysis.emulator import gate_c as G  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.predict import predict_on_physical_grid  # noqa: E402

CACHE = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def ks_bins():
    """Upstream's LOO k values: the KODIAQ-SQUAD conservative bins its likelihood keeps (0.005 <= k <= 0.064,
    lyaemu/likelihood.py, kf_old), read from the same table."""
    path = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/final-conservative-p1d-karacayli_etal2021.txt"
    k = np.array(sorted({float(ln.split("|")[2]) for ln in open(path).readlines()[1:] if ln.strip()}))   # "| z | k | P | e |"
    k = k[(k >= 0.005) & (k <= 0.064)]
    if k.size != 11:
        raise ValueError(f"expected upstream's 11 LOO k bins, found {k.size}")
    return k


def eboss_bins():
    d = np.load("/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz", allow_pickle=True)
    return np.unique(np.asarray(d["k"], float))


def held_out_rows(d, protocol, ckpt, meta):
    names = np.asarray(d["sim_name"])
    if protocol == "loso8":
        fold = int(re.search(r"_fold(\d+)$", ckpt).group(1))
        tr, va, ho = D.make_splits(d, fold, n_folds=8)
        sims = np.unique(names[va])
        edge = ho[np.isin(names[ho], sims)]
        return np.concatenate([va, edge]), np.concatenate([np.zeros(va.size, bool), np.ones(edge.size, bool)])
    if protocol == "loo60":
        sim = (meta.get("recipe") or {}).get("holdout_sim")
        if not sim:
            raise ValueError(f"{ckpt}: checkpoint recipe names no holdout_sim")
        rows = np.where(names == sim)[0]
        return rows, np.zeros(rows.size, bool)
    if protocol == "prod":
        rec = meta.get("recipe") or {}
        tr, va, _ = D.production_row_split(names, val_seed=rec.get("val_seed", 12345), val_frac=rec.get("val_frac", 0.1))
        return va, np.zeros(va.size, bool)
    raise ValueError(protocol)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("loso8", "loo60", "prod"), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    d = D.load_cache(a.cache)
    model, meta, norm = T.load_checkpoint(a.ckpt)
    if meta.get("cache_sha256") and "cache_path" in meta:
        from hcd_analysis.emulator.train import _sha256_or_none
        sha = _sha256_or_none(a.cache)
        if sha != meta["cache_sha256"]:
            raise SystemExit(f"cache {a.cache} sha256 {sha} != the checkpoint's training cache {meta['cache_sha256']}")
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    rows, is_edge = held_out_rows(d, a.protocol, a.ckpt, meta)
    K = d["P_tier_p"].shape[1]
    kks, keb = ks_bins(), eboss_bins()
    in_range = D.datarange_mask(d)

    n = rows.size
    coord = np.full(n, np.nan)
    res_cls = np.full((n, 4, K), np.nan)
    res_tot = np.full((n, K), np.nan)
    res_ks = np.full((n, kks.size), np.nan)
    res_eb = np.full((n, keb.size), np.nan)
    theta_u = D.normalize_params(d["params"][rows])
    for i, r in enumerate(rows):
        z = float(d["z_grid"][r])
        pred = predict_on_physical_grid(model, meta, jnp.asarray(theta_u[i]), z, jnp.asarray(d["tau0"][r]),
                                        jnp.zeros(3), pf, jnp.zeros(K))
        k_pred = np.asarray(pred.k_skm)
        k_row = np.asarray(d["kfkms"][r], float)
        coord[i] = G.coordinate_agreement(k_pred, k_row)
        P_pred = np.asarray(pred.P_filt)                                   # (4,K)
        P_true = np.asarray(d["P_filt"][r])
        with np.errstate(invalid="ignore", divide="ignore"):
            res_cls[i] = np.where(np.isfinite(P_true) & (P_true != 0), P_pred / P_true - 1.0, np.nan)
            tot_pred = np.einsum("c,ck->k", np.asarray(d["w_c_cache"][r]), P_pred)
            tot_true = np.asarray(d["P_tier_p"][r], float)
            res_tot[i] = np.where(np.isfinite(tot_true) & (tot_true != 0), tot_pred / tot_true - 1.0, np.nan)
        for kt, out in ((kks, res_ks), (keb, res_eb)):
            fin = np.isfinite(k_row) & np.isfinite(tot_true)
            if kt.max() <= k_row[fin].max() and kt.min() >= k_row[fin].min():
                out[i] = G.interp_to_grid(kt, k_pred[fin], tot_pred[fin]) / G.interp_to_grid(kt, k_row[fin], tot_true[fin]) - 1.0
    np.savez_compressed(
        a.out, rows=rows, is_tau0_edge=is_edge, sim_name=np.asarray(d["sim_name"])[rows].astype(str),
        z=d["z_grid"][rows], alpha=D.tau0_ladder_factor(d["tau0"][rows], d["z_grid"][rows]),
        alpha_idx=d["alpha_idx"][rows], theta_unit=theta_u, coord=coord, res_cls=res_cls, res_tot=res_tot,
        res_ks=res_ks, res_eb=res_eb, k_ks=kks, k_eb=keb, in_range=in_range[rows], k_com_hmpc=d["k_com_hmpc"],
        protocol=a.protocol, ckpt=a.ckpt, cache=a.cache, meta=json.dumps(meta))
    print(f"[gateC eval] {a.protocol} {a.ckpt}: {n} rows ({int(is_edge.sum())} tau0-edge); max C1 {np.nanmax(coord):.2e}; "
          f"median |res| clean/LLS/subDLA/DLA "
          + "/".join(f"{np.nanmedian(np.abs(res_cls[~is_edge, c][in_range[rows][~is_edge]])):.4f}" for c in range(4))
          + f"; total {np.nanmedian(np.abs(res_tot[~is_edge][in_range[rows][~is_edge]])):.4f}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
