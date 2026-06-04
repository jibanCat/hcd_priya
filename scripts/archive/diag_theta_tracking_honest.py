"""HONEST within-cell θ-tracking + deployed-error decomposition (Bayesian referee C1).

The walkthrough/diag θ-tracking demeans pred & true EACH by its OWN val-row
empirical mean (`lt[m]-lt[m].mean(0)`, `lp[m]-lp[m].mean(0)`) — two different
references — which removes the baseline mis-fit from both in lockstep and inflates
the correlation. At inference the deployed model uses the θ-blind baseline head
m̂(z,τ₀), NOT a val-estimated mean.

This re-derives the metric referenced to the DEPLOYED baseline m̂ (the SAME for pred
and true) and decomposes the deployed whitened cosmology-signal error
   r_pred − r_true,  r ≡ (logP − m̂)/σ_cosmo
into two pieces:
   (a) RESIDUAL-head fit error  : r̂ − t_resid,  t_resid = (logP − m_cell_train)/σ_cosmo
   (b) BASELINE-head mis-fit    : (m̂ − m_cell_train)/σ_cosmo   [σ_cosmo amplifies it]
so that  r_pred − r_true = (a) + (b).  Because σ_cosmo ≈ 0.077·σ_marg, a baseline
error that is tiny in σ_marg units is large in cosmology-signal units — (b) is the
quantity that decides whether the deployed emulator actually resolves cosmology.

Reports, per class and overall:
  * FLATTERED corr/spread-ratio (val-mean reference — the old headline)
  * HONEST pooled corr/spread-ratio (m̂ reference — includes baseline mis-fit)
  * MEDIAN per-cell corr (pure within-cell ranking, pooling-independent)
  * whitened RMS of (a), (b), total; and deployed |P̂/P−1| frac-RMS.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_theta_tracking_honest.py
"""
from __future__ import annotations
import json
import numpy as np
import jax
import jax.numpy as jnp
from hcd_analysis.emulator.data import (load_cache, make_batch, make_splits,
                                        cell_id, safe_log)
from hcd_analysis.emulator.train import load_checkpoint

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CKPT = "checkpoints/walkthrough_fold0"
CLS = ("clean", "LLS", "subDLA", "DLA")


def _corr_ratio(dt, dp):
    """Pooled Pearson corr + spread-ratio over finite entries of two 1-D arrays."""
    g = np.isfinite(dt) & np.isfinite(dp)
    if g.sum() < 3:
        return np.nan, np.nan
    c = float(np.corrcoef(dt[g], dp[g])[0, 1])
    r = float(np.nanstd(dp[g]) / np.nanstd(dt[g]))
    return c, r


def _percell_corr(dt_c, dp_c, cv):
    """Median within-cell Pearson corr (each cell scored alone, then median)."""
    out = []
    for cl in np.unique(cv):
        m = cv == cl
        if m.sum() < 2:
            continue
        a = dt_c[m].ravel()
        b = dp_c[m].ravel()
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() < 3 or np.nanstd(a[g]) < 1e-12 or np.nanstd(b[g]) < 1e-12:
            continue
        out.append(float(np.corrcoef(a[g], b[g])[0, 1]))
    return float(np.nanmedian(out)) if out else np.nan, len(out)


def _whit_rms(x):
    g = np.isfinite(x)
    return float(np.sqrt(np.nanmean(x[g] ** 2)))


def main():
    d = load_cache(CACHE)
    model, meta, norm = load_checkpoint(CKPT)
    pf = norm["P_filt"]
    sig_marg, mu_marg, sig_cosmo = pf["sig_marg"], pf["mu_marg"], pf["sig_cosmo"]
    fold = int(meta.get("fold", 0))
    nfold = int(meta.get("n_folds", 8))
    tr, va, ho = make_splits(d, fold, nfold)

    b = make_batch(d, va, norm)
    pred = jax.vmap(model)(jnp.asarray(b["x"]), jnp.asarray(b["tau0"]))
    base = np.asarray(pred["P_filt_base"])               # (n,4,K) standardized m̂
    resid = np.asarray(pred["P_filt_resid"])             # (n,4,K) standardized r̂

    m_hat = base * sig_marg + mu_marg                    # deployed baseline, logP units
    lt = safe_log(d["P_filt"][va])                       # true logP
    lp = m_hat + sig_cosmo * resid                       # deployed prediction logP
    cv = cell_id(d, va)
    m_cell = np.stack([pf["cell_mean"].get(int(c), mu_marg) for c in cv])  # train cell-mean

    # whitened cosmology-signal residuals (reference = deployed m̂)
    r_pred = resid
    r_true = (lt - m_hat) / sig_cosmo
    t_resid = (lt - m_cell) / sig_cosmo                  # training target residual
    err_resid = r_pred - t_resid                         # (a) residual-head fit error
    err_base = (m_hat - m_cell) / sig_cosmo              # (b) baseline mis-fit (whitened)
    err_tot = r_pred - r_true                            # deployed total = (a)+(b)

    # deployed fractional P error
    frac = np.exp(lp - lt) - 1.0                         # P̂/P − 1

    # cosmology-signal deviations (logP units) for corr metrics
    dev_t_hat = lt - m_hat                               # honest: true dev from deployed m̂
    dev_p_hat = lp - m_hat                               # = sig_cosmo·r̂
    print(f"fold={fold}/{nfold}  n_val={len(va)}  cells_with>=2={int(sum(np.bincount(cv.astype(int))>=2))}\n")

    rows = []
    for ci, nm in enumerate(CLS):
        # FLATTERED (val-mean ref, per-cell demean) — reproduce the old headline
        ft, fp = [], []
        for cl in np.unique(cv):
            m = cv == cl
            if m.sum() < 2:
                continue
            ft.append((lt[m, ci, :] - np.nanmean(lt[m, ci, :], 0)).ravel())
            fp.append((lp[m, ci, :] - np.nanmean(lp[m, ci, :], 0)).ravel())
        fl_corr, fl_ratio = _corr_ratio(np.concatenate(ft), np.concatenate(fp))

        # HONEST pooled (m̂ ref, NO re-demean) — keeps baseline mis-fit in dev_t
        keep = np.isin(cv, [cl for cl in np.unique(cv) if (cv == cl).sum() >= 2])
        hon_corr, hon_ratio = _corr_ratio(dev_t_hat[keep, ci, :].ravel(),
                                          dev_p_hat[keep, ci, :].ravel())

        # MEDIAN per-cell corr (pure within-cell ranking)
        pc_corr, n_cells = _percell_corr(dev_t_hat[:, ci, :], dev_p_hat[:, ci, :], cv)

        sig_true = np.nanstd(r_true[:, ci, :][np.isfinite(r_true[:, ci, :])])
        a = _whit_rms(err_resid[:, ci, :]) / sig_true
        bb = _whit_rms(err_base[:, ci, :]) / sig_true
        tot = _whit_rms(err_tot[:, ci, :]) / sig_true
        fr = _whit_rms(frac[:, ci, :])
        rows.append(dict(cls=nm, flat_corr=fl_corr, flat_ratio=fl_ratio,
                         honest_corr=hon_corr, honest_ratio=hon_ratio,
                         percell_corr=pc_corr, n_cells=n_cells,
                         resid_fit=a, base_misfit=bb, deployed_whit=tot, fracP_rms=fr))

    hdr = (f"{'class':7} | {'FLAT corr':>9} {'ratio':>6} | {'HONEST corr':>11} {'ratio':>6} "
           f"{'medcell':>7} | {'(a)resid':>8} {'(b)base':>8} {'tot/sig':>8} | {'|fracP|':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['cls']:7} | {r['flat_corr']:9.3f} {r['flat_ratio']:6.3f} | "
              f"{r['honest_corr']:11.3f} {r['honest_ratio']:6.3f} {r['percell_corr']:7.3f} | "
              f"{r['resid_fit']:8.3f} {r['base_misfit']:8.3f} {r['deployed_whit']:8.3f} | "
              f"{r['fracP_rms']:8.3f}")

    # overall (all classes pooled)
    g = np.isin(cv, [cl for cl in np.unique(cv) if (cv == cl).sum() >= 2])
    oc, orr = _corr_ratio(dev_t_hat[g].ravel(), dev_p_hat[g].ravel())
    ft = [(lt[cv == cl] - np.nanmean(lt[cv == cl], 0)).ravel()
          for cl in np.unique(cv) if (cv == cl).sum() >= 2]
    fp = [(lp[cv == cl] - np.nanmean(lp[cv == cl], 0)).ravel()
          for cl in np.unique(cv) if (cv == cl).sum() >= 2]
    fc, frr = _corr_ratio(np.concatenate(ft), np.concatenate(fp))
    sig_true_all = np.nanstd(r_true[np.isfinite(r_true)])
    print("-" * len(hdr))
    print(f"{'ALL':7} | {fc:9.3f} {frr:6.3f} | {oc:11.3f} {orr:6.3f} {'':7} | "
          f"{_whit_rms(err_resid)/sig_true_all:8.3f} {_whit_rms(err_base)/sig_true_all:8.3f} "
          f"{_whit_rms(err_tot)/sig_true_all:8.3f} | {_whit_rms(frac):8.3f}")

    print("\nLEGEND: FLAT=old headline (val-mean ref, inflated). HONEST=deployed m̂ ref. "
          "medcell=median per-cell corr (pure ranking).\n"
          "(a)resid=residual-head fit err / σ_signal; (b)base=baseline mis-fit / σ_signal "
          "(σ_cosmo amplifies); tot/sig=deployed whitened err / σ_signal; |fracP|=RMS|P̂/P−1|.")

    out = {"fold": fold, "n_folds": nfold, "rows": rows,
           "ALL": {"flat_corr": fc, "honest_corr": oc,
                   "resid_fit": _whit_rms(err_resid) / sig_true_all,
                   "base_misfit": _whit_rms(err_base) / sig_true_all,
                   "deployed_whit": _whit_rms(err_tot) / sig_true_all,
                   "fracP_rms": _whit_rms(frac)}}
    with open("figures/analysis/04_emulator/honest_theta_tracking.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote figures/analysis/04_emulator/honest_theta_tracking.json")


if __name__ == "__main__":
    main()
