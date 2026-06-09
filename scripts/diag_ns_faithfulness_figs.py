"""n_s (tilt) FAITHFULNESS figure pack — forward-only, NO retraining.

Produces 4 PNGs under figures/analysis/04_emulator/ for the PI's visual review of whether the
emulator's n_s response is faithful AND where the coherent −0.65σ prediction RESIDUAL lives, with
the linear-k-grid vs log-k contrast made explicit throughout.

  FIG 1  nsfaith_response.png        — n_s RESPONSE faithfulness (∂logP/∂ns, β(k)), log + linear k.
                                        REUSES figures/.../ns_slope_attenuation.npz (no recompute).
  FIG 2  nsfaith_coherent_residual.png — the actual −0.65σ object: coherent held-out ⟨ΔlogP⟩ =
                                        ⟨logP_emu − logP_truth⟩ (clean class), vs log-k, z-banded;
                                        decomposed baseline-head vs residual-head; in-sample vs held-out.
  FIG 3  nsfaith_spectrum_overlay.png — logP_clean emu-vs-truth for ONE held-out sim at 3 z + residual.
  FIG 4  nsfaith_logk_contrast.png    — the near-power-law logP(k) + how 172 LINEAR-k bins distribute
                                        in log-k (sparse at the low-k pivot, dense at high-k).

ΔlogP CONVENTION (clean class, index 0): ΔlogP = logP_emu − logP_truth (emu minus truth). β<1 ⇒
emu under-predicts the tilt; the Fisher reads a coherent ΔlogP that aligns with +∂logP/∂ns as a
NEGATIVE Δns. The decomposition splits ΔlogP into the θ-blind baseline-head term and the
cosmology residual-head term so the carrier is visible.

Run (import hcd_analysis.emulator BEFORE jax; x64):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_ns_faithfulness_figs.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

assert jax.config.read("jax_enable_x64"), "x64 must be on (import hcd_analysis.emulator first)"

from hcd_analysis.emulator.closure_legb import build_legb_ctx, held_out_sims
from hcd_analysis.emulator.data import load_cache, make_splits, safe_log
from hcd_analysis.emulator.predict import predict_P_filt, reconstruct_P_filt_jax
from hcd_analysis.emulator.inference import PARAM_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
ATTN_NPZ = f"{FIGDIR}/ns_slope_attenuation.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
CACHE_PATH = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
RESP_SHAPES = "/tmp/resp_shapes.npz"

NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
Z_LO, Z_HI = 2.2, 4.6
MIN_SIMS = 12          # cell-mean stability (matches diag_ns_slope_attenuation truth OLS)
K_PIVOT_LO = 0.005     # low-k pivot edge (the n_s zero-crossing sits here)

DPI = 150


# ============================================================================ #
#  Truth cell-means (for the baseline-head vs residual-head ΔlogP decomposition).
#  m_cell = mean of logP_clean over the in-range sims in each (z, alpha_idx) cell;
#  the same cells the truth OLS used. The truth log-P then splits as
#  logP_truth = m_cell + (logP_truth − m_cell), the baseline + cosmology parts.
# ============================================================================ #
def truth_cell_means(d):
    z = np.round(np.asarray(d["z_grid"]), 4)
    a = np.asarray(d["alpha_idx"]).astype(int)
    P_filt = np.asarray(d["P_filt"])
    fin = np.isfinite(P_filt).all(axis=(1, 2))
    inr = (z >= Z_LO - 1e-6) & (z <= Z_HI + 1e-6)
    cells = {}
    for r in np.where(fin & inr)[0]:
        cells.setdefault((z[r], a[r]), []).append(int(r))
    m_cell = {}
    for cell, rows in cells.items():
        if len(rows) < MIN_SIMS:
            continue
        m_cell[cell] = np.mean(safe_log(P_filt[np.array(rows), 0, :]), axis=0)  # (K,) clean logP
    return m_cell


# ============================================================================ #
#  All-folds coherent held-out residual (clean class) + decomposition + in-sample arm.
# ============================================================================ #
def collect_coherent_residual(d, m_cell):
    """For every honestly-held-out sim row (all 8 folds, each its own emulator) AND a matched
    in-sample arm (rows the fold trained on, same sims-count budget), collect the clean-class:
        logP_emu   = full reconstruction (m̂·sig_marg+mu_marg + sig_cosmo·r̂)
        logP_truth = safe_log(cache P_filt[row,0])
        base_term  = m̂·sig_marg + mu_marg          (θ-blind baseline head)
        resid_term = sig_cosmo·r̂                    (cosmology residual head)
    Returns dicts of stacked arrays for held-out and in-sample, tagged by z & ns_unit.
    """
    z_round = np.round(np.asarray(d["z_grid"]), 4)
    names = np.asarray(d["sim_name"])

    HO = dict(dlogP=[], base_err=[], resid_err=[], z=[], ns=[])
    IS = dict(dlogP=[], z=[], ns=[])

    for fold in range(8):
        ckpt = f"{REPO}/checkpoints/final_fold{fold}"
        ctx, _dd = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0,
                                  ks_kwargs={"k_max": 0.06})
        model, pf = ctx.model, ctx.pf_stats
        sig_marg = np.asarray(pf["sig_marg"])[0]    # clean class
        mu_marg = np.asarray(pf["mu_marg"])[0]
        sig_cosmo = np.asarray(pf["sig_cosmo"])[0]

        @jax.jit
        def heads(th, zu, t0):
            x = jnp.concatenate([th, jnp.asarray(zu)[None]])
            pred = model(x, t0)
            P = reconstruct_P_filt_jax(pred["P_filt_base"], pred["P_filt_resid"], pf)
            return jnp.log(P[0]), pred["P_filt_base"][0], pred["P_filt_resid"][0]

        tr, va, _ho = make_splits(d, fold)
        sims_ho, _ = held_out_sims(_dd, fold)

        def rows_for(split_rows, sim):
            rs = split_rows[names[split_rows] == sim]
            out = []
            for r in rs:
                r = int(r)
                zz = z_round[r]
                if not (Z_LO - 1e-6 <= zz <= Z_HI + 1e-6):
                    continue
                if not np.isfinite(d["P_filt"][r]).all():
                    continue
                out.append(r)
            return out

        def eval_row(r):
            th = jnp.asarray(d["params_unit"][r])
            zu = float((d["z_grid"][r] - 2.0) / 3.4)
            t0 = float(d["tau0"][r])
            logP_emu, base_hat, resid_hat = heads(th, zu, t0)
            logP_emu = np.asarray(logP_emu)
            base_term = np.asarray(base_hat) * sig_marg + mu_marg
            resid_term = sig_cosmo * np.asarray(resid_hat)
            logP_truth = safe_log(d["P_filt"][r, 0, :])
            return logP_emu, logP_truth, base_term, resid_term

        # --- HELD-OUT arm (this fold's val sims, this fold's emulator never saw them) ---
        for sim in sims_ho:
            for r in rows_for(va, sim):
                cell = (z_round[r], int(d["alpha_idx"][r]))
                mc = m_cell.get(cell)
                if mc is None:
                    continue
                logP_emu, logP_truth, base_term, resid_term = eval_row(r)
                HO["dlogP"].append(logP_emu - logP_truth)
                # baseline-head error: emu baseline term vs truth cell-mean (θ-blind part)
                HO["base_err"].append(base_term - mc)
                # residual-head error: emu cosmology term vs truth deviation-from-cell-mean
                HO["resid_err"].append(resid_term - (logP_truth - mc))
                HO["z"].append(z_round[r]); HO["ns"].append(float(d["params_unit"][r, NS_I]))

        # --- IN-SAMPLE arm (sims this fold TRAINED on) — cap to ~same #sims as held-out for a
        #     balanced comparison (the first len(sims_ho) train sims, in name order). ---
        train_sims = sorted(set(names[tr]))[:len(sims_ho)]
        for sim in train_sims:
            for r in rows_for(tr, sim):
                cell = (z_round[r], int(d["alpha_idx"][r]))
                if m_cell.get(cell) is None:
                    continue
                logP_emu, logP_truth, _b, _r = eval_row(r)
                IS["dlogP"].append(logP_emu - logP_truth)
                IS["z"].append(z_round[r]); IS["ns"].append(float(d["params_unit"][r, NS_I]))

        print(f"  fold {fold}: held-out rows so far={len(HO['dlogP'])}, "
              f"in-sample rows so far={len(IS['dlogP'])}", flush=True)

    for D in (HO, IS):
        for k in D:
            D[k] = np.asarray(D[k])
    return HO, IS


# ============================================================================ #
#  One held-out spectrum (for FIG 3) — fold 5 mid-n_s sim if available.
# ============================================================================ #
def one_holdout_spectrum(d, fold=5):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    ctx, _dd = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0, ks_kwargs={"k_max": 0.06})
    model, pf = ctx.model, ctx.pf_stats
    tr, va, _ho = make_splits(d, fold)
    names = np.asarray(d["sim_name"])
    sims_ho, _ = held_out_sims(_dd, fold)
    z_round = np.round(np.asarray(d["z_grid"]), 4)

    # pick the held-out sim with n_s_unit closest to 0.5 (mid-n_s)
    best_sim, best_d = None, np.inf
    for sim in sims_ho:
        rs = va[names[va] == sim]
        rs = [int(r) for r in rs if Z_LO - 1e-6 <= z_round[r] <= Z_HI + 1e-6
              and np.isfinite(d["P_filt"][r]).all()]
        if not rs:
            continue
        ns_u = float(d["params_unit"][rs[0], NS_I])
        if abs(ns_u - 0.5) < best_d:
            best_d, best_sim = abs(ns_u - 0.5), sim
    rows = [int(r) for r in va[names[va] == best_sim]
            if Z_LO - 1e-6 <= z_round[r] <= Z_HI + 1e-6 and np.isfinite(d["P_filt"][r]).all()]
    rows = sorted(rows, key=lambda r: d["z_grid"][r])
    zs = np.array([z_round[r] for r in rows])
    # 3 representative z: low, mid, high
    targets = [2.2, 3.0, 3.8]
    pick = [rows[int(np.argmin(np.abs(zs - tz)))] for tz in targets]

    out = []
    for r in pick:
        th = jnp.asarray(d["params_unit"][r])
        zu = float((d["z_grid"][r] - 2.0) / 3.4)
        t0 = float(d["tau0"][r])
        logP_emu = np.log(np.asarray(predict_P_filt(model, th, zu, t0, pf))[0])
        logP_truth = safe_log(d["P_filt"][r, 0, :])
        out.append((float(z_round[r]), logP_emu, logP_truth))
    return best_sim, float(d["params_unit"][pick[0], NS_I]), fold, out


# ============================================================================ #
#  FIG 1 — n_s response faithfulness (reuse attenuation npz).
# ============================================================================ #
def fig1_response(k, slope_true_mean, rec_emu, rec_true, rec_z, beta_k, mask, thresh, abs_true):
    # representative z bands
    zbands = [(2.2, "z≈2.2", 2.2, 2.5), (3.0, "z≈3.0", 2.9, 3.1), (3.8, "z≈3.8", 3.7, 4.0)]
    cols = ["C0", "C1", "C2"]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # zero-crossing of ensemble truth slope
    sgn = np.sign(slope_true_mean)
    xc = k[np.where(np.diff(sgn) != 0)[0]]
    pivot_k = float(xc[0]) if xc.size else K_PIVOT_LO

    # the per-z response means (bounded ~[-0.3,0.2]); fix the shared y-limits explicitly so the
    # divergent first-bin regression value can't blow up the linear-x autoscale.
    zmeans = []
    for (zc, lab, zlo, zhi), c in zip(zbands, cols):
        sel = (rec_z >= zlo) & (rec_z < zhi)
        if sel.sum() == 0:
            zmeans.append(None); continue
        zmeans.append((lab, c, np.nanmean(rec_true[sel], axis=0), np.nanmean(rec_emu[sel], axis=0)))
    finite_vals = np.concatenate([np.concatenate([t, e]) for z in zmeans if z for t, e in [z[2:]]])
    ylo = float(np.nanpercentile(finite_vals, 0.5)) - 0.02
    yhi = float(np.nanpercentile(finite_vals, 99.5)) + 0.02
    n_below = int((k < K_PIVOT_LO).sum())

    # (a) log-x: emu vs truth ∂logP/∂ns per z
    a = ax[0]
    a.axhline(0, color="k", lw=0.6)
    for z in zmeans:
        if z is None:
            continue
        lab, c, tru_z, emu_z = z
        a.plot(k, tru_z, "-", color=c, lw=2.0, label=f"truth {lab}")
        a.plot(k, emu_z, "--", color=c, lw=1.6, label=f"emu {lab}")
    a.axvline(pivot_k, color="gray", ls=":", alpha=0.8)
    a.set_ylim(ylo, yhi)
    a.annotate(f"pivot (zero-cross)\nk≈{pivot_k:.4f}", xy=(pivot_k, 0),
               xytext=(pivot_k * 1.6, yhi * 0.55), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="gray"))
    a.set_xscale("log"); a.set_xlabel("k [s/km] (log)"); a.set_ylabel(r"$\partial \log P_{clean}/\partial n_s$")
    a.set_title("(a) n_s response: emu (dashed) vs truth (solid), log-k")
    a.legend(fontsize=7, ncol=1); a.grid(alpha=0.3)

    # (b) linear-x: SAME data, annotate # bins below the pivot (the low-k sampling sparsity)
    b = ax[1]
    b.axhline(0, color="k", lw=0.6)
    for z in zmeans:
        if z is None:
            continue
        lab, c, tru_z, emu_z = z
        b.plot(k, tru_z, "-", color=c, lw=2.0, label=f"truth {lab}")
        b.plot(k, emu_z, "--", color=c, lw=1.6, label=f"emu {lab}")
    b.set_ylim(ylo, yhi)
    # rug of the 172 linear-k bin positions along the bottom
    rug_y = ylo + 0.03 * (yhi - ylo)
    b.plot(k, np.full_like(k, rug_y), "|", color="0.35", ms=10, mew=0.5)
    b.axvline(K_PIVOT_LO, color="r", ls=":", alpha=0.7)
    b.annotate(f"only {n_below} of {k.size} bins\nbelow k<{K_PIVOT_LO}\n(the pivot region)",
               xy=(K_PIVOT_LO, yhi * 0.4), xytext=(0.018, yhi * 0.45), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="r"))
    b.set_xlabel("k [s/km] (LINEAR — the training grid)")
    b.set_ylabel(r"$\partial \log P_{clean}/\partial n_s$")
    b.set_title("(b) SAME, linear-k: pivot/low-k is sparsely sampled")
    b.legend(fontsize=7); b.grid(alpha=0.3)

    # (c) β(k) vs log-k with z-scatter IQR band + masked pivot shading
    c = ax[2]
    c.axhline(1.0, color="k", ls="--", lw=0.9, label="β=1 (faithful)")
    # per-z β(k) for an IQR band: compute β(k) per z-band and show spread
    beta_zk = []
    for (zc, lab, zlo, zhi), col in zip(zbands, cols):
        sel = (rec_z >= zlo) & (rec_z < zhi)
        if sel.sum() == 0:
            continue
        e = rec_emu[sel]; t = rec_true[sel]
        bz = np.full(k.size, np.nan)
        for ik in range(k.size):
            cm = abs_true[sel][:, ik] > thresh
            if cm.sum() >= 5:
                bz[ik] = np.sum(e[cm, ik] * t[cm, ik]) / np.sum(t[cm, ik] ** 2)
        beta_zk.append(bz)
        c.plot(k, bz, "-", color=col, lw=1.0, alpha=0.8, label=f"β(k) {lab}")
    c.plot(k, beta_k, "o-", color="k", ms=3, lw=1.4, label="β(k) all-z pooled")
    # shade the masked (|slope_true|<p40) bins where β is divide-by-tiny-noisy (the pivot + the
    # high-k response turnover): the same p40 mask the attenuation diagnostic applied.
    from matplotlib.patches import Patch
    abs_true_mean = np.nanmean(abs_true, axis=0)
    masked_k = abs_true_mean <= np.percentile(abs_true_mean, 40)
    for kk in k[masked_k]:
        c.axvspan(kk * 0.99, kk * 1.01, color="orange", alpha=0.10)
    c.set_xscale("log"); c.set_xlabel("k [s/km] (log)"); c.set_ylabel("β(k) = slope_emu / slope_true")
    c.set_ylim(0, 1.8); c.set_title("(c) β(k): response ratio (≈1 = faithful)")
    handles, labels = c.get_legend_handles_labels()
    handles.append(Patch(facecolor="orange", alpha=0.3))
    labels.append("masked (|slope_true|<p40): noisy β")
    c.legend(handles, labels, fontsize=7); c.grid(alpha=0.3)

    fig.suptitle("FIG 1 — n_s RESPONSE faithfulness: ∂logP_clean/∂n_s (emu vs truth) and β(k). "
                 "Reuses ns_slope_attenuation.npz (all-folds held-out).", fontsize=12)
    fig.tight_layout()
    p = f"{FIGDIR}/nsfaith_response.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, pivot_k, n_below


# ============================================================================ #
#  FIG 2 — the coherent residual (the −0.65σ object).
# ============================================================================ #
def _ns_projected_residual(dlogP, ns):
    """Per-k regression slope of ΔlogP on (ns_unit − mean): r_ns(k) = ∂⟨ΔlogP⟩/∂ns across sims.
    This is the n_s-CORRELATED part of the residual — the piece that projects onto Δn_s in the
    Fisher (a plain mean over the symmetric n_s box cancels and is NOT what the −0.65σ reads).
    If the response is attenuated by β, this slope ≈ (β−1)·slope_true."""
    x = ns - ns.mean()
    denom = float(np.sum(x * x))
    return (x[:, None] * dlogP).sum(axis=0) / denom    # (K,)


def fig2_coherent(k, HO, IS, slope_true_mean):
    zbands = [("z<2.6", lambda z: z < 2.6, "C0"),
              ("2.6≤z<3.4", lambda z: (z >= 2.6) & (z < 3.4), "C1"),
              ("z≥3.4", lambda z: z >= 3.4, "C3")]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # normalized truth n_s response shape (for overlay alignment check)
    resp = slope_true_mean / np.max(np.abs(slope_true_mean))

    # (a) the n_s-PROJECTED residual r_ns(k)=∂⟨ΔlogP⟩/∂ns (the −0.65σ object) per z-band,
    #     overlaid with the truth n_s response. ALSO show the plain mean ⟨ΔlogP⟩ (faint) to be
    #     honest that the mean cancels — the Fisher reads the n_s-correlated slope, not the mean.
    a = ax[0]
    a.axhline(0, color="k", lw=0.6)
    r_ns_all = _ns_projected_residual(HO["dlogP"], HO["ns"])
    for lab, fn, c in zbands:
        sel = fn(HO["z"])
        if sel.sum() < 20:
            continue
        r_ns = _ns_projected_residual(HO["dlogP"][sel], HO["ns"][sel])
        a.plot(k, 100 * r_ns, "-", color=c, lw=1.8,
               label=f"$r_{{n_s}}$(k) {lab} (n={int(sel.sum())})")
    # plain mean (faint, for honesty): it ~cancels
    a.plot(k, 100 * np.nanmean(HO["dlogP"], axis=0), ":", color="0.55", lw=1.2,
           label="plain ⟨ΔlogP⟩ (cancels over box)")
    a.set_xscale("log"); a.set_xlabel("k [s/km] (log)")
    a.set_ylabel(r"$r_{n_s}(k)=\partial\langle\Delta\log P\rangle/\partial n_s$  [%/unit-$n_s$]")
    a2 = a.twinx()
    a2.plot(k, resp, color="0.5", ls="--", lw=1.5, label="truth n_s response (norm.)")
    a2.axhline(0, color="0.7", lw=0.5)
    a2.set_ylabel("normalized $\\partial\\log P/\\partial n_s$", color="0.4")
    a2.set_ylim(-1.2, 1.2)
    a.set_title("(a) n_s-PROJECTED residual vs the n_s tilt shape")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left"); a.grid(alpha=0.3)

    # (b) decomposition: baseline-head error vs residual-head error (pooled all-z)
    b = ax[1]
    b.axhline(0, color="k", lw=0.6)
    tot = np.nanmean(HO["dlogP"], axis=0)
    base = np.nanmean(HO["base_err"], axis=0)
    res = np.nanmean(HO["resid_err"], axis=0)
    b.plot(k, 100 * tot, "-", color="k", lw=2.2, label="total ⟨ΔlogP⟩")
    b.plot(k, 100 * base, "-", color="C0", lw=1.6, label="baseline-head error (m̂ − cell-mean)")
    b.plot(k, 100 * res, "-", color="C3", lw=1.6, label="residual-head error (σ_cosmo·r̂ − dev)")
    b.set_xscale("log"); b.set_xlabel("k [s/km] (log)"); b.set_ylabel(r"⟨error⟩ [%]")
    b.set_title("(b) decomposition: who carries the coherent residual?")
    b.legend(fontsize=8); b.grid(alpha=0.3)

    # (c) generalization gap: in-sample vs held-out coherent residual
    c = ax[2]
    c.axhline(0, color="k", lw=0.6)
    ho_all = np.nanmean(HO["dlogP"], axis=0)
    is_all = np.nanmean(IS["dlogP"], axis=0)
    c.plot(k, 100 * ho_all, "-", color="C3", lw=2.0, label=f"HELD-OUT (n={HO['dlogP'].shape[0]})")
    c.plot(k, 100 * is_all, "-", color="C2", lw=2.0, label=f"IN-SAMPLE (n={IS['dlogP'].shape[0]})")
    c.set_xscale("log"); c.set_xlabel("k [s/km] (log)"); c.set_ylabel(r"⟨ΔlogP⟩ [%]")
    c.set_title("(c) generalization gap: in-sample vs held-out")
    c.legend(fontsize=8); c.grid(alpha=0.3)

    fig.suptitle("FIG 2 — the COHERENT residual ⟨ΔlogP⟩ (emu−truth, clean class) that the Fisher reads "
                 "as Δn_s. All 8 folds, honestly held out.", fontsize=12)
    fig.tight_layout()
    p = f"{FIGDIR}/nsfaith_coherent_residual.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # diagnostics for the report
    klo = k < 0.005; kmid = (k >= 0.005) & (k < 0.02); khi = k >= 0.02
    align_mean = float(np.corrcoef(tot, slope_true_mean)[0, 1])
    # the n_s-PROJECTED residual is the object the Fisher actually reads:
    align_ns = float(np.corrcoef(r_ns_all, slope_true_mean)[0, 1])
    # implied β from the projection: r_ns = (β−1)·slope_true ⇒ β = 1 + <r·s>/<s·s>
    beta_implied = 1.0 + float(np.sum(r_ns_all * slope_true_mean) / np.sum(slope_true_mean ** 2))
    return p, dict(
        tot=tot, base=base, res=res, ho_all=ho_all, is_all=is_all, r_ns=r_ns_all,
        lowk_pct=100 * tot[klo].mean(), midk_pct=100 * tot[kmid].mean(), hik_pct=100 * tot[khi].mean(),
        align_mean=align_mean, align_ns=align_ns, beta_implied=beta_implied,
        rns_lowk=100 * r_ns_all[klo].mean(), rns_midk=100 * r_ns_all[kmid].mean(),
        rns_hik=100 * r_ns_all[khi].mean(),
        base_lowk=100 * base[klo].mean(), res_lowk=100 * res[klo].mean(),
        base_rms=100 * np.sqrt(np.mean(base ** 2)), res_rms=100 * np.sqrt(np.mean(res ** 2)),
        is_rms=100 * np.sqrt(np.mean(is_all ** 2)), ho_rms=100 * np.sqrt(np.mean(ho_all ** 2)))


# ============================================================================ #
#  FIG 3 — direct spectrum overlay for one held-out sim.
# ============================================================================ #
def fig3_spectrum(k, sim, ns_u, fold, spectra):
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.4]},
                           sharex=True)
    cols = ["C0", "C1", "C3"]
    a, r = ax
    for (zz, lp_emu, lp_tru), c in zip(spectra, cols):
        a.plot(k, lp_tru, "-", color=c, lw=2.0, label=f"truth z={zz:.1f}")
        a.plot(k, lp_emu, "--", color=c, lw=1.5, label=f"emu z={zz:.1f}")
        r.plot(k, 100 * (lp_emu - lp_tru), "-", color=c, lw=1.5, label=f"z={zz:.1f}")
    a.set_xscale("log"); a.set_ylabel(r"$\log P_{clean}$"); a.grid(alpha=0.3)
    a.legend(fontsize=8, ncol=3); a.set_title(
        f"FIG 3 — held-out spectrum: sim {sim[:24]} (n_s_unit={ns_u:.2f}, fold {fold})")
    r.axhline(0, color="k", lw=0.6)
    r.set_xscale("log"); r.set_xlabel("k [s/km] (log)")
    r.set_ylabel(r"$\log P_{emu}-\log P_{truth}$ [%]")
    r.grid(alpha=0.3); r.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    p = f"{FIGDIR}/nsfaith_spectrum_overlay.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================================ #
#  FIG 4 — the linear-vs-log-k question, made explicit.
# ============================================================================ #
def fig4_logk(k, spectra, tot_resid):
    # use the mid-z spectrum from FIG 3 as the near-power-law display
    zz, lp_emu, lp_tru = spectra[1]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))

    # (left) near-power-law logP(k) + bin-density rug + per-decade histogram
    a = ax[0]
    a.plot(k, lp_tru, "-", color="C0", lw=2.0, label=f"logP_clean (truth, z={zz:.1f})")
    a.set_xscale("log"); a.set_xlabel("k [s/km] (log)"); a.set_ylabel(r"$\log P_{clean}$")
    # rug marks at each of the 172 linear-k bins
    ylo, yhi = a.get_ylim()
    rug_y = ylo + 0.03 * (yhi - ylo)
    a.plot(k, np.full_like(k, rug_y), "|", color="0.3", ms=8, mew=0.6,
           label="172 linear-k bins (rug)")
    a.set_title("(a) logP is near-power-law; the 172 bins are LINEAR in k")
    a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)
    # inset: histogram of bins per log-k decade
    ai = a.inset_axes([0.12, 0.12, 0.42, 0.34])
    kpos = k[k > 0]
    logk = np.log10(kpos)
    bins = np.linspace(logk.min(), logk.max(), 9)
    ai.hist(logk, bins=bins, color="0.5", edgecolor="k", lw=0.4)
    ai.set_xlabel("log10 k", fontsize=7); ai.set_ylabel("# bins", fontsize=7)
    ai.tick_params(labelsize=6); ai.set_title("bins per log-k", fontsize=7)

    # (right) coherent residual vs log-k with the bin-density rug
    b = ax[1]
    b.axhline(0, color="k", lw=0.6)
    b.plot(k, 100 * tot_resid, "-", color="C3", lw=2.0, label="all-z coherent ⟨ΔlogP⟩")
    ylo2, yhi2 = b.get_ylim()
    rug_y2 = ylo2 + 0.04 * (yhi2 - ylo2)
    b.plot(k, np.full_like(k, rug_y2), "|", color="0.3", ms=8, mew=0.6,
           label="172 linear-k bins (rug)")
    b.axvspan(k[k > 0].min(), 0.005, color="orange", alpha=0.12, label="sparse low-k (pivot) region")
    b.set_xscale("log"); b.set_xlabel("k [s/km] (log)")
    b.set_ylabel(r"⟨ΔlogP⟩ [%]"); b.set_title("(b) residual structure vs the sparse-bin region")
    b.legend(fontsize=8); b.grid(alpha=0.3)

    n_decade_low = int((logk < (logk.min() + (logk.max() - logk.min()) / 8)).sum())
    n_decade_high = int((logk >= (logk.max() - (logk.max() - logk.min()) / 8)).sum())

    fig.suptitle("FIG 4 — linear-k grid vs log-k: low-k (the n_s pivot) is sparsely sampled, "
                 "high-k densely. Does the residual live in the sparse region?", fontsize=12)
    fig.tight_layout()
    p = f"{FIGDIR}/nsfaith_logk_contrast.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, n_decade_low, n_decade_high


# ============================================================================ #
#  Main.
# ============================================================================ #
def main():
    print("[load] attenuation npz + cache", flush=True)
    A = np.load(ATTN_NPZ)
    k = np.asarray(A["k"])
    slope_true_mean = np.asarray(A["slope_true_mean"])
    rec_emu = np.asarray(A["rec_emu"]); rec_true = np.asarray(A["rec_true"])
    rec_z = np.asarray(A["rec_z"]); beta_k = np.asarray(A["beta_k"])
    mask = np.asarray(A["mask"]); thresh = float(A["thresh"])
    abs_true = np.abs(rec_true)

    # --- SANITY: truth response shape vs /tmp/resp_shapes.npz::ns_resp ---
    if Path(RESP_SHAPES).exists():
        rs = np.load(RESP_SHAPES)
        ns_resp = np.asarray(rs["ns_resp"])
        cc = float(np.corrcoef(slope_true_mean, ns_resp)[0, 1])
        sgn = np.sign(slope_true_mean)
        xc = k[np.where(np.diff(sgn) != 0)[0]]
        print(f"[sanity] truth-slope vs resp_shapes::ns_resp shape corr={cc:+.3f} "
              f"(expect strongly +); zero-cross k≈{xc[:1]} (expect ~0.0044); "
              f"low-k(k<0.005) mean={slope_true_mean[k<0.005].mean():+.3f} (expect <0), "
              f"high-k(k>0.02) mean={slope_true_mean[k>0.02].mean():+.3f} (expect >0)", flush=True)
        assert cc > 0.5, f"truth slope shape does not match resp_shapes (corr={cc})"
        assert slope_true_mean[k < 0.005].mean() < 0, "low-k truth slope not negative"
        assert slope_true_mean[k > 0.02].mean() > 0, "high-k truth slope not positive"
        print("[sanity] PASS", flush=True)
    else:
        print("[sanity] resp_shapes.npz absent; relying on sign check", flush=True)

    print("[cache] loading + truth cell-means", flush=True)
    d = load_cache(CACHE_PATH)
    m_cell = truth_cell_means(d)
    print(f"[cache] {len(m_cell)} (z,alpha) cells with >= {MIN_SIMS} sims", flush=True)

    # --- FIG 1 (no recompute) ---
    print("[fig1] response faithfulness", flush=True)
    p1, pivot_k, n_below = fig1_response(k, slope_true_mean, rec_emu, rec_true, rec_z,
                                         beta_k, mask, thresh, abs_true)

    # --- coherent residual (all folds) ---
    print("[collect] coherent residual over all 8 folds (held-out + in-sample)", flush=True)
    HO, IS = collect_coherent_residual(d, m_cell)
    print(f"[collect] held-out rows={HO['dlogP'].shape[0]}, in-sample rows={IS['dlogP'].shape[0]}",
          flush=True)

    # sanity: coherent residual should be sub-percent / percent level (not 10s of %)
    ho_rms = 100 * np.sqrt(np.mean(np.nanmean(HO["dlogP"], axis=0) ** 2))
    print(f"[sanity] held-out coherent ⟨ΔlogP⟩ RMS = {ho_rms:.3f}% (expect ~1-2% level)", flush=True)
    assert ho_rms < 15.0, f"coherent residual implausibly large ({ho_rms}%)"

    # --- FIG 2 ---
    print("[fig2] coherent residual + decomposition + gen-gap", flush=True)
    p2, diag = fig2_coherent(k, HO, IS, slope_true_mean)

    # --- FIG 3 ---
    print("[fig3] one held-out spectrum", flush=True)
    sim, ns_u, fold, spectra = one_holdout_spectrum(d, fold=5)
    p3 = fig3_spectrum(k, sim, ns_u, fold, spectra)

    # --- FIG 4 ---
    print("[fig4] linear-vs-log-k contrast", flush=True)
    p4, ndl, ndh = fig4_logk(k, spectra, diag["tot"])

    # --- save small npz for reproducibility ---
    np.savez(f"{FIGDIR}/nsfaith_coherent_residual.npz",
             k=k, ho_dlogP_mean=np.nanmean(HO["dlogP"], axis=0),
             is_dlogP_mean=np.nanmean(IS["dlogP"], axis=0),
             base_err_mean=np.nanmean(HO["base_err"], axis=0),
             resid_err_mean=np.nanmean(HO["resid_err"], axis=0),
             r_ns=diag["r_ns"], beta_implied=diag["beta_implied"],
             ho_ns=HO["ns"], ho_z=HO["z"],
             slope_true_mean=slope_true_mean, n_ho=HO["dlogP"].shape[0], n_is=IS["dlogP"].shape[0])

    # ---- report block ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"FIG1 {p1}")
    print(f"  pivot zero-cross k≈{pivot_k:.4f}; {n_below} of {k.size} bins below k<{K_PIVOT_LO}")
    print(f"FIG2 {p2}")
    print(f"  PLAIN mean ⟨ΔlogP⟩ (emu−truth) cancels over the box: low-k={diag['lowk_pct']:+.3f}%, "
          f"mid-k={diag['midk_pct']:+.3f}%, high-k={diag['hik_pct']:+.3f}% (RMS {diag['ho_rms']:.3f}%)")
    print(f"  n_s-PROJECTED residual r_ns(k) (the −0.65σ object): low-k={diag['rns_lowk']:+.3f}, "
          f"mid-k={diag['rns_midk']:+.3f}, high-k={diag['rns_hik']:+.3f} %/unit-ns")
    print(f"  alignment corr(plain mean, n_s resp)={diag['align_mean']:+.3f}; "
          f"corr(r_ns, n_s resp)={diag['align_ns']:+.3f}; implied β={diag['beta_implied']:.3f}")
    print(f"  baseline-head RMS={diag['base_rms']:.3f}% vs residual-head RMS={diag['res_rms']:.3f}%; "
          f"at low-k: base={diag['base_lowk']:+.3f}%, resid={diag['res_lowk']:+.3f}%")
    print(f"  gen-gap: in-sample RMS={diag['is_rms']:.3f}% vs held-out RMS={diag['ho_rms']:.3f}%")
    print(f"FIG3 {p3}  (sim {sim}, fold {fold}, n_s_unit={ns_u:.2f})")
    print(f"FIG4 {p4}  (bins in lowest log-k bin={ndl}, in highest={ndh})")
    print("[done]")


if __name__ == "__main__":
    main()
