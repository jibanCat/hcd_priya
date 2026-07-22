#!/usr/bin/env python3
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# retrain-ensemble evaluation; not the deployed emulator.
"""Ensemble headroom probe: does averaging seeds REDUCE the held-out per-class error?

Loads the saved single-member checkpoints checkpoints/retrain/ens_baseline_fold{f}_seed{s}
(s=0,1,2), and per fold compares:
  - single-member held-out rms(k<0.06) per class (mean over the 3 members)
  - the 3-member ENSEMBLE (mean of reconstructed P_filt over members) held-out rms

Ensemble pred path == predict.predict_P_filt with an EnsembleEmulator (mean of post-exp
P_filt over members), the SAME definition production uses.

ENV: PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import hcd_analysis.emulator  # x64
import jax, jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache, datarange_mask, make_splits
from hcd_analysis.emulator.predict import predict_P_filt

CLASSES = ["clean", "LLS", "subDLA", "DLA"]
REPO = "/home/mfho/hcd_priya"
CK = f"{REPO}/checkpoints/retrain"


class EnsembleEmulator:
    """Duck-typed ensemble: predict.predict_P_filt detects .members and means post-exp."""
    def __init__(self, members): self.members = members


def member_pred_P(model, norm, d, idx):
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    def one(xi, ti):
        return predict_P_filt(model, xi[:9], xi[9], ti, pf)
    out = []
    for s in range(0, len(idx), 2048):
        out.append(np.asarray(jax.vmap(one)(x[s:s + 2048], tau0[s:s + 2048])))
    return np.concatenate(out, axis=0)


def rms_per_class(P_emu, Pt, km, kv, kcut=0.06):
    res = {}
    for c in range(4):
        denom = np.where(Pt[:, c, :] != 0, Pt[:, c, :], np.nan)
        e = P_emu[:, c, :] / denom - 1.0
        good = km & np.isfinite(e) & np.isfinite(kv) & (kv > 0) & (kv < kcut)
        ee = e[good]
        res[CLASSES[c]] = float(np.sqrt(np.mean(ee ** 2))) if ee.size else np.nan
    return res


def main():
    d = load_cache(f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5")
    keep = datarange_mask(d) & d["mask"]
    kf = np.asarray(d["kfkms"]); P_true = np.asarray(d["P_filt"])
    folds = [0, 1, 2]; seeds = [0, 1, 2]
    rows = []
    print("# ===== ENSEMBLE vs single-member HELD-OUT rms(k<0.06) per class =====")
    single_acc = {c: [] for c in CLASSES}; ens_acc = {c: [] for c in CLASSES}
    for f in folds:
        _tr, va, _ho = make_splits(d, f)
        Pt = P_true[va]; km = keep[va]; kv = kf[va]
        members = []
        singles = []
        ok = True
        for s in seeds:
            ck = f"{CK}/ens_baseline_fold{f}_seed{s}"
            if not Path(ck + ".eqx").exists():
                print(f"  [skip] fold{f} seed{s}: no checkpoint"); ok = False; continue
            model, meta, norm = T.load_checkpoint(ck)
            members.append((model, norm))
            P = member_pred_P(model, norm, d, va)
            singles.append(rms_per_class(P, Pt, km, kv))
        if not members:
            continue
        # ensemble: members must share the SAME norm to mean P_filt with one pf_stats.
        # Each member has its OWN norm (train-split-identical here: same fold => same train rows
        # => same norm), so use member-0's norm for the shared pf_stats in predict_P_filt.
        ens = EnsembleEmulator([m for m, _ in members])
        norm0 = members[0][1]
        Pe = member_pred_P(ens, norm0, d, va)
        ens_rms = rms_per_class(Pe, Pt, km, kv)
        single_mean = {c: float(np.mean([s[c] for s in singles])) for c in CLASSES}
        print(f"\nfold {f}  ({len(members)} members):")
        for c in CLASSES:
            d_pct = 100 * (ens_rms[c] - single_mean[c])
            print(f"   {c:7s}: single(mean)={100*single_mean[c]:.3f}%  ensemble={100*ens_rms[c]:.3f}%  "
                  f"Δ={d_pct:+.3f}%")
            single_acc[c].append(single_mean[c]); ens_acc[c].append(ens_rms[c])
    print("\n# ===== POOLED over folds =====")
    for c in CLASSES:
        if not single_acc[c]:
            continue
        sm = np.mean(single_acc[c]); em = np.mean(ens_acc[c])
        print(f"  {c:7s}: single {100*sm:.3f}%  ->  ensemble {100*em:.3f}%   "
              f"Δ {100*(em-sm):+.3f}%  ({100*(em-sm)/sm:+.1f}% rel)")
    print("\n(ensemble Δ NEGATIVE => ensembling reduces held-out error = real headroom)")


if __name__ == "__main__":
    main()
