#!/usr/bin/env python3
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# headroom-experiment pilot; not the deployed emulator.
"""Emulator-retraining HEADROOM study: train ONE variant on ONE LOSO fold, eval held-out.

THE QUESTION: is the current emulator at its accuracy floor, or is there headroom?
This harness trains a single (variant, fold) and writes its HELD-OUT per-class/per-k/
per-z fractional error so we can compare a variant against the deployed baseline
(checkpoints/final_fold{f}) on the SAME held-out sims (apples-to-apples).

A "variant" is a delta on the FINAL_RECIPE / arch. Levers exposed:
  - capacity:   --enc-widths (encoder), --headb-trunk, --baseline-width/--baseline-layers
  - budget:     --epochs --patience --lr
  - loss:       --p-resid-w --w-coh --edge-gain --lowk-extra --no-datarange
  - output rep: --n-basis (SVD/basis components)
The harness reuses hcd_analysis.emulator.train.train_fold for the BASELINE arch, and
a thin custom loop for capacity variants that change the Emulator constructor (the
encoder/headb widths are hard-coded in model.py, so a capacity variant monkeypatches
those module classes before constructing — see _build_model).

HELD-OUT metric == the SAME masking/pred path as scripts/diag_emu_loso_perk.py:
fractional error e=(P_emu/P_true-1) over datarange_mask & d['mask'], per class, binned
in (k,z). Reported: per-class in-range RMS, k<0.06 RMS, worst-(k,z) cell, and a
per-parameter Fisher-sensitivity proxy (gradient of logP wrt each of the 9 params,
held-out-row-averaged) so we can see which params the variant resolves better.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/retrain_headroom_pilot.py \
        --fold 0 --variant baseline --tag pilot
"""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
import numpy as np

import hcd_analysis.emulator  # x64 hard-assert BEFORE jax
import jax, jax.numpy as jnp
import equinox as eqx

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import model as M
from hcd_analysis.emulator.data import (
    load_cache, datarange_mask, edge_emphasis_k_weight, make_splits, DATA_RANGE,
)
from hcd_analysis.emulator.predict import predict_P_filt

REPO = "/home/mfho/hcd_priya"
CACHE = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]
PARAMS = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]

FINAL_RECIPE = dict(n_basis=24, p_resid_w=8.0, edge_gain=3.0, lowk_extra=2.0,
                    w_coh=80.0, weight_decay=3e-4, datarange=True,
                    epochs=180, patience=25, lr=1e-3, batch=512)


def _patched_widths(enc_widths, headb_trunk):
    """Context-managed monkeypatch of Encoder/HeadB default widths.

    model.py hard-codes the encoder widths (256,128,64) and HeadB trunk (256). A
    capacity variant rebinds those defaults around the Emulator construction. We patch
    the __init__ defaults via wrappers so the rest of train_fold is untouched."""
    import contextlib
    @contextlib.contextmanager
    def ctx():
        orig_enc = M.Encoder.__init__
        orig_hb = M.HeadB.__init__
        def enc_init(self, in_dim=10, widths=enc_widths, key=None):
            orig_enc(self, in_dim=in_dim, widths=widths, key=key)
        def hb_init(self, latent=64, n_k=172, n_basis=None, p_filt_basis_init=None, key=None,
                    _trunk=headb_trunk):
            # replicate HeadB.__init__ but with a configurable trunk width
            import jax as _jax
            k1, k2, k3 = _jax.random.split(key, 3)
            self.n_k = n_k; self.n_basis = n_basis
            self.trunk = eqx.nn.Linear(latent + 1, _trunk, key=k1)
            if n_basis is None:
                self.out = eqx.nn.Linear(_trunk, 7 * n_k, key=k2); self.p_filt_basis = None
            else:
                self.out = eqx.nn.Linear(_trunk, 4 * n_basis + 3 * n_k, key=k2)
                if p_filt_basis_init is not None:
                    self.p_filt_basis = jnp.asarray(p_filt_basis_init)
                else:
                    self.p_filt_basis = _jax.nn.initializers.orthogonal()(k3, (n_basis, n_k))
        M.Encoder.__init__ = enc_init
        M.HeadB.__init__ = hb_init
        try:
            yield
        finally:
            M.Encoder.__init__ = orig_enc
            M.HeadB.__init__ = orig_hb
    return ctx()


VARIANTS = {
    # name -> dict of (recipe overrides) + (arch overrides: enc_widths, headb_trunk)
    "baseline":      dict(),
    # --- capacity ---
    # NOTE: the encoder's FINAL width is the latent dim, hard-wired to 64 in HeadA/HeadB
    # (model.py). Capacity variants must keep the last enc width == 64; only the HIDDEN
    # widths and the HeadB trunk may grow. (A non-64 final width => head matmul shape error.)
    "wider_enc":     dict(enc_widths=(384, 192, 64), headb_trunk=384),
    "deeper_enc":    dict(enc_widths=(256, 192, 128, 64), headb_trunk=256),
    # --- output representation ---
    "nbasis48":      dict(n_basis=48),
    "nbasis12":      dict(n_basis=12),
    # --- training budget ---
    "longer":        dict(epochs=300, patience=40),
    # --- loss weighting ---
    "presid_hi":     dict(p_resid_w=16.0),
    "wcoh_hi":       dict(w_coh=160.0),
    "no_datarange":  dict(datarange=False),
    "no_edge":       dict(edge_gain=1.0, lowk_extra=1.0),
}


def _build_and_train(d, tr, va, *, recipe, enc_widths, headb_trunk, seed):
    n_k = d["P_tier_p"].shape[1]
    term_w = {"f_nhi": 1.0, "dndx": 1.0, "p_base": 1.0,
              "p_resid": recipe["p_resid_w"], "delta": 1.0}
    k_weight = (edge_emphasis_k_weight(d["kfkms"][0], edge_gain=recipe["edge_gain"],
                                       lowk_extra=recipe["lowk_extra"])
                if recipe["edge_gain"] != 1.0 or recipe["lowk_extra"] != 1.0 else
                edge_emphasis_k_weight(d["kfkms"][0], edge_gain=recipe["edge_gain"],
                                       lowk_extra=recipe["lowk_extra"]))
    kw = dict(d=d, train_idx=tr, val_idx=va, n_basis=recipe["n_basis"],
              lr=recipe["lr"], epochs=recipe["epochs"], batch_size=recipe["batch"],
              seed=seed, key=jax.random.PRNGKey(seed), patience=recipe["patience"],
              n_k=n_k, term_w=term_w, k_weight=k_weight, w_coh=recipe["w_coh"],
              datarange=recipe["datarange"], weight_decay=recipe["weight_decay"])
    if enc_widths is None and headb_trunk is None:
        return T.train_fold(**kw)
    ew = enc_widths or (256, 128, 64)
    ht = headb_trunk or 256
    with _patched_widths(ew, ht):
        return T.train_fold(**kw)


def member_pred(model, norm, d, idx):
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    def one(xi, ti):
        return predict_P_filt(model, xi[:9], xi[9], ti, pf)
    out = []
    for s in range(0, len(idx), 2048):
        out.append(np.asarray(jax.vmap(one)(x[s:s + 2048], tau0[s:s + 2048])))
    return np.concatenate(out, axis=0)


def eval_heldout(model, norm, d, va):
    """Per-class held-out fractional error summary on val rows va.
    Returns dict of per-class metrics (all-k RMS, k<0.06 RMS, worst cell)."""
    keep = datarange_mask(d) & d["mask"]                       # (R,K) in-range finite
    kf = np.asarray(d["kfkms"]); P_true = np.asarray(d["P_filt"])
    P_emu = member_pred(model, norm, d, va)                    # (n,4,K)
    Pt = P_true[va]; km = keep[va]; kv = kf[va]
    zv = np.asarray(d["z_grid"])[va]                           # (n,)
    res = {}
    kcut = 0.06
    # per-class flat errors
    for c in range(4):
        denom = np.where(Pt[:, c, :] != 0, Pt[:, c, :], np.nan)
        e = P_emu[:, c, :] / denom - 1.0                       # (n,K)
        good = km & np.isfinite(e) & np.isfinite(kv) & (kv > 0)
        e_all = e[good]; k_all = kv[good]
        z_all = np.broadcast_to(zv[:, None], e.shape)[good]
        rms_all = float(np.sqrt(np.mean(e_all ** 2))) if e_all.size else np.nan
        med_all = float(np.median(np.abs(e_all))) if e_all.size else np.nan
        sub06 = k_all < kcut
        rms06 = float(np.sqrt(np.mean(e_all[sub06] ** 2))) if sub06.any() else np.nan
        # worst (k,z) cell: bin into 12 logk x 8 z, find max |signed-mean| ... use rms
        kb = np.geomspace(max(1e-3, k_all.min()), k_all.max(), 13)
        zb = np.linspace(zv.min(), zv.max(), 9)
        ki = np.clip(np.digitize(k_all, kb) - 1, 0, 11)
        zi = np.clip(np.digitize(z_all, zb) - 1, 0, 7)
        cell_rms = np.full((12, 8), np.nan)
        for i in range(12):
            for j in range(8):
                sel = (ki == i) & (zi == j)
                if sel.sum() >= 20:
                    cell_rms[i, j] = np.sqrt(np.mean(e_all[sel] ** 2))
        worst = float(np.nanmax(cell_rms)) if np.isfinite(cell_rms).any() else np.nan
        res[CLASSES[c]] = dict(rms_all=rms_all, med_all=med_all, rms_k06=rms06,
                               worst_cell=worst, npts=int(e_all.size))
    return res


def fisher_sensitivity(model, norm, d, va, n_rows=300):
    """Per-parameter held-out Fisher-sensitivity proxy: mean over held-out rows of
    ||d logP_clean / d theta_p||^2 (unit-cube params), a relative knob on how strongly
    the emulator's output responds to each of the 9 params. Higher == more constraining.
    A coarse proxy (no data covariance), but apples-to-apples across variants."""
    rng = np.random.default_rng(0)
    sel = va if len(va) <= n_rows else va[rng.choice(len(va), n_rows, replace=False)]
    x = jnp.asarray(d["x"][sel]); tau0 = jnp.asarray(d["tau0"][sel])
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    keep = (datarange_mask(d) & d["mask"])[sel]                # (n,K)

    def logP_clean(theta9, z_unit, ti):
        Pf = predict_P_filt(model, theta9, z_unit, ti, pf)     # (4,K)
        return jnp.log(Pf[0])                                  # clean (K,)
    jac = jax.vmap(lambda xi, ti: jax.jacfwd(lambda th: logP_clean(th, xi[9], ti))(xi[:9]))
    J = np.asarray(jac(x, tau0))                               # (n,K,9)
    km = np.asarray(keep)[:, :, None]                          # (n,K,1)
    Jm = np.where(km, J, np.nan)
    # per-param mean-square sensitivity over in-range (row,k)
    sens = np.nanmean(Jm ** 2, axis=(0, 1))                    # (9,)
    return {PARAMS[p]: float(sens[p]) for p in range(9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--no-fisher", action="store_true")
    ap.add_argument("--outdir", default=f"{REPO}/checkpoints/retrain")
    args = ap.parse_args()

    t0 = time.time()
    d = load_cache(args.cache)
    tr, va, ho = make_splits(d, args.fold)
    recipe = dict(FINAL_RECIPE)
    spec = VARIANTS[args.variant]
    enc_widths = spec.pop("enc_widths", None) if isinstance(spec, dict) else None
    headb_trunk = spec.pop("headb_trunk", None) if isinstance(spec, dict) else None
    recipe.update(spec)
    # re-insert (we popped from a copy? no — VARIANTS is module global; restore)
    if enc_widths is not None: VARIANTS[args.variant]["enc_widths"] = enc_widths
    if headb_trunk is not None: VARIANTS[args.variant]["headb_trunk"] = headb_trunk

    print(f"[retrain] variant={args.variant} fold={args.fold} seed={args.seed}", flush=True)
    print(f"[retrain] recipe={recipe}  enc_widths={enc_widths} headb_trunk={headb_trunk}", flush=True)
    print(f"[retrain] train={len(tr)} val={len(va)}  (held-out sims = fold {args.fold})", flush=True)

    tf = time.time()
    model, norm, hist = _build_and_train(
        d, tr, va, recipe=recipe, enc_widths=enc_widths, headb_trunk=headb_trunk, seed=args.seed)
    train_s = time.time() - tf
    n_ep = len(hist["train_loss"])
    print(f"[retrain] trained {n_ep} epochs in {train_s:.0f}s "
          f"({train_s/max(n_ep,1):.1f}s/epoch)", flush=True)

    held = eval_heldout(model, norm, d, va)
    print("[retrain] HELD-OUT per-class fractional error:", flush=True)
    for c in CLASSES:
        m = held[c]
        print(f"   {c:7s}: rms_all={100*m['rms_all']:.3f}%  rms(k<0.06)={100*m['rms_k06']:.3f}%  "
              f"med={100*m['med_all']:.3f}%  worst_cell={100*m['worst_cell']:.3f}%  n={m['npts']}",
              flush=True)
    fisher = None
    if not args.no_fisher:
        fisher = fisher_sensitivity(model, norm, d, va)
        print("[retrain] per-param Fisher-sensitivity proxy (mean dlogP/dtheta^2):", flush=True)
        for p in PARAMS:
            print(f"   {p:11s}: {fisher[p]:.4g}", flush=True)

    out = dict(variant=args.variant, fold=args.fold, seed=args.seed, recipe=recipe,
               enc_widths=enc_widths, headb_trunk=headb_trunk,
               n_epochs=n_ep, train_s=train_s, s_per_epoch=train_s / max(n_ep, 1),
               held=held, fisher=fisher,
               val_resid_min=float(np.min(hist["val_resid_loss"])))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    op = f"{args.outdir}/{args.tag}_{args.variant}_fold{args.fold}_seed{args.seed}.json"
    Path(op).write_text(json.dumps(out, indent=2, default=float))
    print(f"[retrain] -> {op}  (total {time.time()-t0:.0f}s)", flush=True)
    if args.save_model:
        ck = f"{args.outdir}/{args.tag}_{args.variant}_fold{args.fold}_seed{args.seed}"
        T.save_checkpoint(ck, model, {"in_dim": 10, "n_k": d["P_tier_p"].shape[1],
                          "n_basis": recipe["n_basis"]}, norm, seed=args.seed,
                          kfkms=d["kfkms"], cache_path=args.cache, recipe=recipe)
        print(f"[retrain] checkpoint -> {ck}.eqx", flush=True)


if __name__ == "__main__":
    main()
