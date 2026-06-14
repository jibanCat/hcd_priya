"""Per-k (wavenumber) LOSO prediction-error diagnostic for the HCD P1D emulator.

This is the HONEST held-out generalization error of the LF emulator AS A FUNCTION OF k,
broken down per HCD class (clean / LLS / subDLA / DLA). It complements the scalar per-fold
val-RMS table (final_loso_train.log / notes Doc A §2a) and the (A_p,n_s) Fisher-bias gate
by showing WHERE in k the held-out error lives.

Protocol (reuses the deployed LOSO machinery — NOT the all-sims production ensemble, which
is in-sample):
  - For each of the 8 group-k LOSO folds f, load that fold's OWN held-out emulator
    checkpoints/final_fold{f} (the loader + fold/held-out logic from
    scripts/diag_emu_bias_allfolds.py).
  - Get the fold's held-out VAL rows via closure_legb.held_out_sims(d, f) (== make_splits's
    val_idx with the tau0-edge holdout removed) — whole sims the emulator never trained on.
  - Predict per-class LINEAR P_filt (4,K) for every held-out row with predict_P_filt (the
    exact pred path of scripts/validate_production_ensemble.py::member_pred), vmapped.
  - Fractional prediction error e = P_emu/P_true - 1 per (row, k, class), restricted to the
    in-range, finite mask (datarange_mask & d["mask"], same masking as
    validate_production_ensemble.py).
  - Aggregate over ALL held-out rows across ALL 8 folds. Because the cache's kfkms grid is
    NOT row-uniform (each row's physical k depends on z/scale), we flatten every in-range
    (row, k-bin) point and re-bin onto a COMMON log-k grid, then per common-k bin per class
    report median|e|, RMS(e), and the 16-84 percentile band of |e|.

x-axis = k in s/km, ANGULAR (k = 2π/λ_v); kfkms is fed directly, NO /(2π).

Forward-only; no NUTS, no training. Runs in ~minutes on CPU.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_loso_perk.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 hard-assert BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache, datarange_mask
from hcd_analysis.emulator.predict import predict_P_filt
from hcd_analysis.emulator.closure_legb import CACHE_PATH, held_out_sims

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
FIG = f"{FIGDIR}/loso_perk_pred_error.png"
NPZ = f"{FIGDIR}/loso_perk_pred_error.npz"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]
COLORS = ["C0", "C1", "C2", "C3"]
N_FOLDS = 8
# In-range reference RMS per class from the deployed 8-fold LOSO (notes Doc A §2a / §2b,
# final_loso_train.log) — drawn as a faint horizontal reference on each panel.
INRANGE_RMS_REF = {"clean": 0.0058, "LLS": 0.0060, "subDLA": 0.0067, "DLA": 0.0098}  # MEDIAN |e|, in-range


def member_pred(model, norm, d, idx):
    """(N,4,K) predicted LINEAR P_filt for rows idx — exact pred path of
    validate_production_ensemble.py::member_pred, vmapped over rows."""
    x = jnp.asarray(d["x"][idx]); tau0 = jnp.asarray(d["tau0"][idx])
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    def one(xi, ti):
        return predict_P_filt(model, xi[:9], xi[9], ti, pf)
    # chunk the vmap so we never blow memory on a big fold
    out = []
    B = 2048
    for s in range(0, len(idx), B):
        out.append(np.asarray(jax.vmap(one)(x[s:s + B], tau0[s:s + B])))
    return np.concatenate(out, axis=0)


def main():
    d = load_cache(CACHE_PATH)
    keep_all = datarange_mask(d) & d["mask"]          # (R,K) in-range finite
    kf = np.asarray(d["kfkms"])                        # (R,K) ANGULAR k [s/km], per-row grid
    P_true_all = np.asarray(d["P_filt"])              # (R,4,K)

    # ----- gather every held-out (row,k,class) fractional error across all 8 folds -----
    # store flat arrays: k value, signed error, class index
    flat_k = {c: [] for c in range(4)}
    flat_e = {c: [] for c in range(4)}
    n_rows_per_fold = []
    for f in range(N_FOLDS):
        ckpt = f"{REPO}/checkpoints/final_fold{f}"
        model, meta, norm = T.load_checkpoint(ckpt)
        _sims, va = held_out_sims(d, f)               # val_idx (tau0-edge holdout removed)
        va = np.asarray(va)
        n_rows_per_fold.append(len(va))
        P_emu = member_pred(model, norm, d, va)        # (n,4,K)
        Pt = P_true_all[va]                            # (n,4,K)
        km = keep_all[va]                              # (n,K)
        kv = kf[va]                                     # (n,K)
        for c in range(4):
            denom = np.where(Pt[:, c, :] != 0, Pt[:, c, :], np.nan)
            e = P_emu[:, c, :] / denom - 1.0           # (n,K) fractional error
            good = km & np.isfinite(e) & np.isfinite(kv) & (kv > 0)
            flat_k[c].append(kv[good]); flat_e[c].append(e[good])
        print(f"  fold {f}: {len(_sims)} held-out sims, {len(va)} val rows "
              f"({np.asarray(_sims)[:3]}...)", flush=True)
    for c in range(4):
        flat_k[c] = np.concatenate(flat_k[c]); flat_e[c] = np.concatenate(flat_e[c])
    print(f"\ntotal held-out rows over 8 folds: {sum(n_rows_per_fold)} "
          f"(in-range points/class ~{flat_k[0].size})", flush=True)

    # ----- common log-k binning (the per-row grid is non-uniform) -----
    # in-range k spans ~1e-3 .. ~0.096; the dominant Nyquist of the n_k=172 grid is ~0.069.
    kmin = max(1e-3, min(flat_k[c].min() for c in range(4)))
    kmax = max(flat_k[c].max() for c in range(4))
    n_kbin = 22
    edges = np.geomspace(kmin, kmax, n_kbin + 1)
    kcen = np.sqrt(edges[:-1] * edges[1:])

    med = np.full((4, n_kbin), np.nan); rms = np.full((4, n_kbin), np.nan)
    p16 = np.full((4, n_kbin), np.nan); p84 = np.full((4, n_kbin), np.nan)
    npts = np.zeros((4, n_kbin), dtype=int)
    for c in range(4):
        which = np.digitize(flat_k[c], edges) - 1
        for b in range(n_kbin):
            sel = which == b
            n = int(sel.sum()); npts[c, b] = n
            if n < 20:
                continue
            ae = np.abs(flat_e[c][sel])
            med[c, b] = np.median(ae)
            rms[c, b] = np.sqrt(np.mean(flat_e[c][sel] ** 2))
            p16[c, b] = np.percentile(ae, 16); p84[c, b] = np.percentile(ae, 84)

    # ----- headline numbers -----
    NYQ = 0.069  # nominal Nyquist of the n_k=172 angular grid (kfkms top ~0.0694)
    def at_k(arr_c, ktarget):
        bb = np.nanargmin(np.abs(kcen - ktarget))
        return arr_c[bb], kcen[bb]
    print("\n# ===== headline: median |P_emu/P_true - 1| vs angular k (per class) =====")
    head = {}
    bbnyq = int(np.nanargmin(np.abs(kcen - NYQ)))      # bin straddling the Nyquist k
    for c in range(4):
        valid = np.isfinite(med[c])
        klo = kcen[valid][0]; lo = med[c][valid][0]
        khi = kcen[bbnyq]; hi = med[c][bbnyq]           # at the Nyquist bin
        worst = np.nanmax(med[c]); kworst = kcen[np.nanargmax(med[c])]
        best = np.nanmin(med[c]); kbest = kcen[np.nanargmin(med[c])]
        head[CLASSES[c]] = dict(lowk=lo, klo=klo, nyqk=khi, nyq=hi,
                                worst=worst, kworst=kworst, best=best, kbest=kbest)
        print(f"  {CLASSES[c]:7s}: low-k({klo:.4f}) {100*lo:.2f}%  ->  Nyq-k({khi:.4f}) {100*hi:.2f}%"
              f"   | best {100*best:.2f}% @k={kbest:.4f}  worst {100*worst:.2f}% @k={kworst:.4f} (above-Nyq tail)")

    # ----- figure: 2x2 panels, one per class -----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)
    axs = axs.ravel()
    for c in range(4):
        ax = axs[c]; col = COLORS[c]; v = np.isfinite(med[c])
        ax.fill_between(kcen[v], 100 * p16[c][v], 100 * p84[c][v], color=col, alpha=0.18,
                        label="16-84 pct of |e|")
        ax.plot(kcen[v], 100 * med[c][v], "-o", color=col, ms=4, lw=1.8,
                label="median |P_emu/P_true - 1|")
        ax.plot(kcen[v], 100 * rms[c][v], "--", color=col, alpha=0.7, lw=1.3, label="RMS(e)")
        ref = INRANGE_RMS_REF[CLASSES[c]]
        ax.axhline(100 * ref, color="k", ls=":", lw=1.0, alpha=0.7,
                   label=f"in-range LOSO median {100*ref:.2f}%")
        ax.axvline(0.069, color="grey", ls="-", lw=0.7, alpha=0.5)
        ax.text(0.069, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
                " Nyq k≈0.069", rotation=90, va="top", ha="right", fontsize=7, color="grey")
        ax.set_xscale("log")
        ax.set_title(f"{CLASSES[c]}  (held-out LOSO, all 8 folds)")
        ax.set_ylabel("|P_emu/P_true - 1|  [%]")
        if c >= 2:
            ax.set_xlabel("angular wavenumber k  [s/km]   (k = 2π/λ_v)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Per-k held-out LOSO prediction error of the LF P1D emulator\n"
                 "(8 group-k folds, each fold's OWN held-out checkpoint final_fold{0..7}; "
                 "honest generalization, NOT in-sample)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(FIGDIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n[fig] {FIG}")

    np.savez(NPZ, kcen=kcen, edges=edges, classes=np.array(CLASSES),
             median_abs_err=med, rms_err=rms, p16_abs=p16, p84_abs=p84, npts=npts,
             n_rows_per_fold=np.array(n_rows_per_fold),
             nyquist_k=NYQ, k_axis_note="angular k [s/km], k=2pi/lambda_v, fed direct (no /2pi)")
    print(f"[npz] {NPZ}")
    print("[done]")


if __name__ == "__main__":
    main()
