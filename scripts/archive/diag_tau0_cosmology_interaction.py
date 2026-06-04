"""EMPIRICAL τ₀ × cosmology INTERACTION in the filtered P1D, and whether the
two-stage emulator captures it.

The production cache is a (z, α) grid of CELLS: at fixed z the α (mean-flux
rescale, τ→α·τ) index IS the τ₀ axis, and within one (z,α) cell τ₀ is
bit-identical across the ~60 cosmologies (verified std ~9e-16). So the
within-(z,α)-cell spread of lnP across sims IS the COSMOLOGY response, measured
SEPARATELY at each τ₀ level. Comparing that cosmology response across α (τ₀) at
fixed z is a direct, model-free probe of the ∂²lnP/∂θ∂τ₀ interaction.

We measure four things and emit figures to figures/analysis/04_emulator/:

1. AMPLITUDE interaction. Per (z, class, k): σ_cosmo(k | z, α) = within-cell std
   of lnP across cosmologies, as a function of α (τ₀). If the cosmology response
   amplitude were τ₀-independent this is flat in α. We report the spread
   max_α/min_α of σ_cosmo per k (the amplitude part of ∂²lnP/∂θ∂τ₀).

2. SHAPE / RANKING interaction. Per cell, the per-sim deviation δ_s(k) = lnP_s −
   cellmean is the cosmology imprint of sim s at that τ₀. We match sims between
   the lowest-α cell and the highest-α cell (same z) and correlate the pooled
   (sim,k) deviation patterns. corr≈1 ⇒ the τ₀ change only RESCALES the cosmology
   imprint (a scalar interaction the per-k σ_cosmo would mostly absorb); corr<1 ⇒
   the SHAPE of the cosmology response changes with τ₀ (a genuine non-separable
   interaction). This is the decisive test.

3. LINEAR-MODEL variance decomposition. Per (z, k, class) regress lnP on the 9
   unit-cube cosmology params x, on τ₀, and on the x×τ₀ cross terms; report the
   fraction of the COSMOLOGY-response sum-of-squares carried by the x×τ₀
   interaction vs the pure-x main effect. A non-trivial x×τ₀ fraction ⇒ the
   residual head MUST be jointly non-separable in (θ, τ₀).

4. ARCHITECTURE CHECK. With the fold-0 checkpoint, jax.jacfwd the reconstructed
   lnP̂ wrt the 9 unit-cube params at a FIXED cosmology+z while SWEEPING τ₀ across
   the ladder. The trained cosmology response is σ_cosmo(k)·∂r̂/∂θ (the baseline is
   θ-blind). If the model learned the interaction this Jacobian's MAGNITUDE varies
   with τ₀; we compare its τ₀-dependence (max/min over α of ‖∂lnP̂/∂θ‖) to the
   empirical σ_cosmo(k|z,α) τ₀-dependence from step 1, per class.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_tau0_cosmology_interaction.py
"""
from __future__ import annotations
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache, safe_log, normalize_params, Z_LIMITS
from hcd_analysis.emulator.train import load_checkpoint

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
CKPT = "checkpoints/walkthrough_fold0"
OUTDIR = "figures/analysis/04_emulator"
CLS = ("clean", "LLS", "subDLA", "DLA")
NPARAM = 9
Z_PROBE = (2.4, 3.0, 4.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cell_rows(d, zz, aa):
    """Row indices whose rounded z == zz and alpha_idx == aa."""
    z = np.round(d["z_grid"], 4)
    return np.where(np.isclose(z, zz) & (d["alpha_idx"] == aa))[0]


def _within_cell_std(lp_cell):
    """Across-cosmology std of lnP per (class,k), NaN-aware. lp_cell: (n,4,K)."""
    with np.errstate(invalid="ignore"):
        return np.nanstd(lp_cell, axis=0, ddof=1)        # (4,K)


def _safe_corr(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 3 or np.nanstd(a[g]) < 1e-15 or np.nanstd(b[g]) < 1e-15:
        return np.nan, int(g.sum())
    return float(np.corrcoef(a[g], b[g])[0, 1]), int(g.sum())


# ---------------------------------------------------------------------------
# Step 1: amplitude interaction  σ_cosmo(k | z, α)
# ---------------------------------------------------------------------------

def step1_amplitude(d, kf):
    """For each probe z and each α, the within-cell cosmology std of lnP per
    (class,k). Returns sigma[z][α] = (4,K) and the per-(z,class,k) max_α/min_α
    spread ratio. Plots σ_cosmo(k) vs α per class."""
    lp = safe_log(d["P_filt"])                            # (R,4,K)
    alphas = np.unique(d["alpha_idx"])
    out = {}                                              # z -> (n_alpha,4,K)
    tau0_of = {}                                          # z -> (n_alpha,)
    for zz in Z_PROBE:
        stk = np.full((len(alphas), 4, lp.shape[2]), np.nan)
        t0 = np.full(len(alphas), np.nan)
        for ai, aa in enumerate(alphas):
            rows = _cell_rows(d, zz, aa)
            if len(rows) < 3:
                continue
            stk[ai] = _within_cell_std(lp[rows])
            t0[ai] = d["tau0"][rows].mean()
        out[zz] = stk
        tau0_of[zz] = t0

    # spread ratio max_α/min_α of σ_cosmo per (z,class,k)
    spread = {}
    for zz in Z_PROBE:
        stk = out[zz]                                     # (n_alpha,4,K)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmax(stk, axis=0) / np.nanmax(
                np.stack([np.nanmin(stk, axis=0),
                          np.full_like(stk[0], 1e-30)]), axis=0)   # (4,K)
        spread[zz] = ratio

    # ---- figure: σ_cosmo(k) vs α, per class, one panel-row per z -----------
    fig, axes = plt.subplots(len(Z_PROBE), 4, figsize=(17, 3.2 * len(Z_PROBE)),
                             sharex=True)
    for zi, zz in enumerate(Z_PROBE):
        stk = out[zz]
        t0 = tau0_of[zz]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
        for ci in range(4):
            ax = axes[zi, ci]
            for ai in range(len(alphas)):
                if not np.isfinite(t0[ai]):
                    continue
                ax.plot(kf, stk[ai, ci], color=cmap[ai], lw=0.9, alpha=0.85)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if zi == 0:
                ax.set_title(CLS[ci])
            if ci == 0:
                ax.set_ylabel(f"z={zz}\nσ_cosmo(k) [lnP]")
            if zi == len(Z_PROBE) - 1:
                ax.set_xlabel("k [s/km]")
            ax.grid(alpha=0.25, which="both")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(alphas.min(), alphas.max()))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("α index (low→high τ₀)")
    fig.suptitle("Step 1 — amplitude interaction: within-cell cosmology std of lnP "
                 "vs τ₀ (α). Curves spreading apart ⇒ amplitude depends on τ₀.",
                 y=0.995)
    p1 = os.path.join(OUTDIR, "tau0xcosmo_amplitude_sigma_vs_alpha.png")
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- figure: spread ratio (max_α/min_α of σ_cosmo) vs k ----------------
    fig2, ax2 = plt.subplots(1, len(Z_PROBE), figsize=(5 * len(Z_PROBE), 4),
                             sharey=True)
    for zi, zz in enumerate(Z_PROBE):
        for ci in range(4):
            ax2[zi].plot(kf, spread[zz][ci], lw=1.2, label=CLS[ci])
        ax2[zi].set_xscale("log")
        ax2[zi].axhline(1.0, color="k", lw=0.6, ls=":")
        ax2[zi].set_title(f"z={zz}")
        ax2[zi].set_xlabel("k [s/km]")
        ax2[zi].grid(alpha=0.25, which="both")
        if zi == 0:
            ax2[zi].set_ylabel("max_α σ_cosmo / min_α σ_cosmo")
            ax2[zi].legend(fontsize=8)
    fig2.suptitle("Step 1 — amplitude τ₀-dependence of the cosmology response "
                  "(1.0 = no amplitude interaction)")
    p2 = os.path.join(OUTDIR, "tau0xcosmo_amplitude_spread_ratio.png")
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    return out, tau0_of, spread, (p1, p2)


# ---------------------------------------------------------------------------
# Step 2: shape / ranking interaction  corr(δ@α_lo, δ@α_hi)
# ---------------------------------------------------------------------------

def step2_shape(d, kf):
    """Match sims between the lowest-α and highest-α cells (same z) and correlate
    the pooled (sim,k) cosmology-deviation patterns δ_s = lnP_s − cellmean. corr≈1
    ⇒ pure scalar rescale; corr<1 ⇒ τ₀-dependent SHAPE change. Also compute the
    correlation as a smooth function of α (vs the lowest-α reference)."""
    lp = safe_log(d["P_filt"])
    alphas = np.unique(d["alpha_idx"])
    a_lo, a_hi = int(alphas.min()), int(alphas.max())
    sim = d["sim_name"]

    results = {}        # z -> per-class corr(lo,hi)
    curves = {}         # z -> (class, n_alpha) corr vs reference α_lo
    for zz in Z_PROBE:
        rows_lo = _cell_rows(d, zz, a_lo)
        # build a sim->row map for each α at this z
        per_alpha_rows = {aa: {sim[r]: r for r in _cell_rows(d, zz, aa)}
                          for aa in alphas}
        sims_lo = [sim[r] for r in rows_lo]

        # corr(lo, hi) per class, pooled over (sim,k)
        cc = np.full(4, np.nan)
        # deviation at α_lo and α_hi for the COMMON sims
        common = [s for s in sims_lo if s in per_alpha_rows[a_hi]]
        rlo = np.array([per_alpha_rows[a_lo][s] for s in common])
        rhi = np.array([per_alpha_rows[a_hi][s] for s in common])
        dev_lo = lp[rlo] - np.nanmean(lp[rlo], axis=0)    # (m,4,K)
        dev_hi = lp[rhi] - np.nanmean(lp[rhi], axis=0)
        for ci in range(4):
            cc[ci], _ = _safe_corr(dev_lo[:, ci, :].ravel(),
                                   dev_hi[:, ci, :].ravel())
        results[zz] = cc

        # smooth corr vs α (reference = α_lo) per class
        crv = np.full((4, len(alphas)), np.nan)
        for ai, aa in enumerate(alphas):
            common_a = [s for s in sims_lo if s in per_alpha_rows[aa]]
            r0 = np.array([per_alpha_rows[a_lo][s] for s in common_a])
            ra = np.array([per_alpha_rows[aa][s] for s in common_a])
            d0 = lp[r0] - np.nanmean(lp[r0], axis=0)
            da = lp[ra] - np.nanmean(lp[ra], axis=0)
            for ci in range(4):
                crv[ci, ai], _ = _safe_corr(d0[:, ci, :].ravel(),
                                            da[:, ci, :].ravel())
        curves[zz] = crv

    # ---- figure: pattern-corr vs α (per class, per z) ----------------------
    fig, axes = plt.subplots(1, len(Z_PROBE), figsize=(5 * len(Z_PROBE), 4),
                             sharey=True)
    for zi, zz in enumerate(Z_PROBE):
        for ci in range(4):
            axes[zi].plot(alphas, curves[zz][ci], "-o", ms=3, label=CLS[ci])
        axes[zi].set_title(f"z={zz}")
        axes[zi].set_xlabel("α index (τ₀)")
        axes[zi].axhline(1.0, color="k", lw=0.6, ls=":")
        axes[zi].grid(alpha=0.25)
        if zi == 0:
            axes[zi].set_ylabel("corr(δ@α_lo, δ@α)  [pooled sim×k]")
            axes[zi].legend(fontsize=8)
    fig.suptitle("Step 2 — shape interaction: cosmology-deviation pattern corr vs τ₀.\n"
                 "1.0 = pure scalar rescale; drop = τ₀-dependent SHAPE change.", y=1.06)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "tau0xcosmo_shape_pattern_corr.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return results, curves, p


# ---------------------------------------------------------------------------
# Step 3: linear-model variance decomposition  (pure-x vs x×τ₀)
# ---------------------------------------------------------------------------

def step3_linear(d, kf):
    """Per (z, class, k): regress lnP on [1, x(9), τ0c, x*τ0c] and split the
    explained cosmology-response sum-of-squares into the pure-x main effect and the
    x×τ₀ interaction. τ0c is the τ₀ deviation from its mean over the rows in the
    regression (so the main-effect block is the response at mean τ₀). x is centered
    likewise. Returns per-class median fractions over k and a (4,K) interaction map.

    Method: fit the full OLS once. SS_full = ‖ŷ − ȳ‖². Then the contribution of a
    coefficient block is taken as the drop in fitted SS when that block is zeroed
    (a hierarchical/sequential-style attribution; because x and τ₀ are centered and
    nearly orthogonal on this near-factorial grid the blocks are close to
    independent, so frac_x + frac_xtau ≈ 1 for the cosmology-driven part)."""
    lp = safe_log(d["P_filt"])
    z = np.round(d["z_grid"], 4)
    Xall = normalize_params(d["params"])                  # (R,9) unit cube
    K = lp.shape[2]
    frac_x = {}
    frac_xt = {}
    R2 = {}
    for zz in Z_PROBE:
        mz = np.isclose(z, zz)
        rows = np.where(mz)[0]
        xx = Xall[rows]                                   # (n,9)
        t0 = d["tau0"][rows]                              # (n,)
        # center
        xc = xx - xx.mean(0)
        tc = t0 - t0.mean()
        # design: [1, xc(9), tc, xc*tc(9)]
        cross = xc * tc[:, None]                          # (n,9)
        D = np.concatenate([np.ones((len(rows), 1)), xc, tc[:, None], cross], axis=1)
        # column block indices
        b_int = [0]
        b_x = list(range(1, 1 + NPARAM))
        b_t = [1 + NPARAM]
        b_xt = list(range(2 + NPARAM, 2 + 2 * NPARAM))

        fx = np.full((4, K), np.nan)
        fxt = np.full((4, K), np.nan)
        r2 = np.full((4, K), np.nan)
        for ci in range(4):
            Y = lp[rows, ci, :]                           # (n,K)
            good = np.isfinite(Y).all(0)
            if not good.any():
                continue
            Yg = Y[:, good]
            # OLS: beta = pinv(D) Y
            beta, *_ = np.linalg.lstsq(D, Yg, rcond=None)  # (P,Kg)
            yhat = D @ beta
            ybar = Yg.mean(0)
            ss_tot = np.sum((Yg - ybar) ** 2, axis=0)
            ss_full = np.sum((yhat - ybar) ** 2, axis=0)
            # block contributions: SS drop when zeroing a block
            def ss_without(blocks):
                bz = beta.copy()
                bz[blocks] = 0.0
                yh = D @ bz
                return np.sum((yh - ybar) ** 2, axis=0)
            ss_no_x = ss_without(b_x)
            ss_no_xt = ss_without(b_xt)
            contrib_x = np.maximum(ss_full - ss_no_x, 0.0)
            contrib_xt = np.maximum(ss_full - ss_no_xt, 0.0)
            denom = contrib_x + contrib_xt
            with np.errstate(invalid="ignore", divide="ignore"):
                fx[ci, good] = contrib_x / np.where(denom > 0, denom, np.nan)
                fxt[ci, good] = contrib_xt / np.where(denom > 0, denom, np.nan)
                r2[ci, good] = ss_full / np.where(ss_tot > 0, ss_tot, np.nan)
        frac_x[zz] = fx
        frac_xt[zz] = fxt
        R2[zz] = r2

    # ---- figure: interaction fraction vs k, per class/z --------------------
    fig, axes = plt.subplots(1, len(Z_PROBE), figsize=(5 * len(Z_PROBE), 4),
                             sharey=True)
    for zi, zz in enumerate(Z_PROBE):
        for ci in range(4):
            axes[zi].plot(kf, frac_xt[zz][ci], lw=1.2, label=CLS[ci])
        axes[zi].set_xscale("log")
        axes[zi].set_ylim(-0.02, 1.02)
        axes[zi].set_title(f"z={zz}")
        axes[zi].set_xlabel("k [s/km]")
        axes[zi].grid(alpha=0.25, which="both")
        if zi == 0:
            axes[zi].set_ylabel("frac of cosmology SS in x×τ₀ interaction")
            axes[zi].legend(fontsize=8)
    fig.suptitle("Step 3 — linear decomposition: fraction of the cosmology response "
                 "carried by the x×τ₀ interaction (vs pure-x main effect)")
    p = os.path.join(OUTDIR, "tau0xcosmo_linear_interaction_fraction.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return frac_x, frac_xt, R2, p


# ---------------------------------------------------------------------------
# Step 4: does the trained residual head reproduce the τ₀-dependence?
# ---------------------------------------------------------------------------

def step4_architecture(d, kf, sigma_emp, tau0_of):
    """jax.jacfwd the reconstructed lnP̂ wrt the 9 unit-cube params at a FIXED
    cosmology+z, SWEEPING τ₀ across the ladder. The trained cosmology response is
    σ_cosmo(k)·∂r̂/∂θ (baseline is θ-blind). We report, per class:
      ρ_model(α) = ‖∂lnP̂/∂θ‖(z,α)  (RMS over the 9 params and k),
    its max_α/min_α spread, and compare to the empirical σ_cosmo τ₀-spread.
    Also returns the per-k model response magnitude vs α for the amplitude
    overlay."""
    model, meta, norm = load_checkpoint(CKPT)
    pf = norm["P_filt"]
    sig_marg = jnp.asarray(pf["sig_marg"])               # (4,K)
    mu_marg = jnp.asarray(pf["mu_marg"])
    sig_cosmo = jnp.asarray(pf["sig_cosmo"])             # (4,K) CONSTANT in τ₀

    # pick a representative cosmology: the cache's median cosmology by params
    Xunit = normalize_params(d["params"])
    centroid = np.median(Xunit, axis=0)
    sim_dist = np.sum((Xunit - centroid) ** 2, axis=1)
    ref_row = int(np.argmin(sim_dist))
    theta0 = jnp.asarray(Xunit[ref_row])                 # (9,) unit cube

    alphas = np.unique(d["alpha_idx"])

    def lnP_hat(theta, z_unit, tau0):
        """Reconstructed lnP̂ (4,K) for unit-cube theta(9), scalar z_unit, scalar τ₀."""
        x = jnp.concatenate([theta, jnp.atleast_1d(z_unit)])     # (10,)
        out = model(x, tau0)
        base = out["P_filt_base"]                                # (4,K) standardized m̂
        resid = out["P_filt_resid"]                              # (4,K) standardized r̂
        logP = (base * sig_marg + mu_marg) + sig_cosmo * resid
        return logP                                              # (4,K)

    jac_fn = jax.jacfwd(lnP_hat, argnums=0)   # d lnP̂ / d theta -> (4,K,9)

    model_resp = {}      # z -> (n_alpha,4,K)  RMS over params of |∂lnP̂/∂θ_j|
    model_norm = {}      # z -> (n_alpha,4)    RMS over params&k
    for zz in Z_PROBE:
        z_unit = (zz - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        respk = np.full((len(alphas), 4, len(kf)), np.nan)
        respn = np.full((len(alphas), 4), np.nan)
        t0 = tau0_of[zz]
        for ai, aa in enumerate(alphas):
            if not np.isfinite(t0[ai]):
                continue
            J = np.asarray(jac_fn(theta0, jnp.asarray(z_unit),
                                  jnp.asarray(t0[ai])))          # (4,K,9)
            # response magnitude per (class,k): RMS over the 9 params
            respk[ai] = np.sqrt(np.mean(J ** 2, axis=2))         # (4,K)
            respn[ai] = np.sqrt(np.mean(J ** 2, axis=(1, 2)))    # (4,)
        model_resp[zz] = respk
        model_norm[zz] = respn

    # ---- figure: model response τ₀-spread vs empirical σ_cosmo τ₀-spread ----
    fig, axes = plt.subplots(2, len(Z_PROBE), figsize=(5 * len(Z_PROBE), 8),
                             sharex=True)
    for zi, zz in enumerate(Z_PROBE):
        respn = model_norm[zz]                            # (n_alpha,4)
        t0 = tau0_of[zz]
        ok = np.isfinite(t0)
        # top row: model τ₀-dependence of the response NORM, per class
        for ci in range(4):
            axes[0, zi].plot(t0[ok], respn[ok, ci], "-o", ms=3, label=CLS[ci])
        axes[0, zi].set_title(f"z={zz}")
        axes[0, zi].grid(alpha=0.25)
        if zi == 0:
            axes[0, zi].set_ylabel("MODEL ‖∂lnP̂/∂θ‖ (RMS over θ,k)")
            axes[0, zi].legend(fontsize=8)
        # bottom row: empirical σ_cosmo NORM (RMS over k) per class vs τ₀
        semp = sigma_emp[zz]                              # (n_alpha,4,K)
        emp_norm = np.sqrt(np.nanmean(semp ** 2, axis=2))  # (n_alpha,4)
        for ci in range(4):
            axes[1, zi].plot(t0[ok], emp_norm[ok, ci], "-s", ms=3, label=CLS[ci])
        axes[1, zi].set_xlabel("τ₀")
        axes[1, zi].grid(alpha=0.25)
        if zi == 0:
            axes[1, zi].set_ylabel("EMPIRICAL σ_cosmo (RMS over k)")
            axes[1, zi].legend(fontsize=8)
    fig.suptitle("Step 4 — trained cosmology-response τ₀-dependence (top) vs empirical "
                 "σ_cosmo τ₀-dependence (bottom). Matched τ₀-trend ⇒ model learned the "
                 "interaction.")
    p = os.path.join(OUTDIR, "tau0xcosmo_model_vs_empirical_response.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return model_resp, model_norm, ref_row, p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = load_cache(CACHE)
    kf = d["kfkms"][0]

    print("=" * 78)
    print("τ₀ × COSMOLOGY INTERACTION in filtered P1D  (cache:", CACHE, ")")
    print("=" * 78)

    sigma_emp, tau0_of, spread1, figs1 = step1_amplitude(d, kf)
    shape_lohi, shape_curves, fig2 = step2_shape(d, kf)
    frac_x, frac_xt, R2, fig3 = step3_linear(d, kf)
    model_resp, model_norm, ref_row, fig4 = step4_architecture(
        d, kf, sigma_emp, tau0_of)

    # ----- numeric summary --------------------------------------------------
    summary = {"cache": CACHE, "ckpt": CKPT, "z_probe": list(Z_PROBE),
               "ref_cosmology_row": ref_row, "per_z": {}}

    print("\n--- STEP 1: AMPLITUDE interaction (max_α/min_α of σ_cosmo, median over k) ---")
    print(f"{'z':>5} | " + " ".join(f"{c:>9}" for c in CLS))
    for zz in Z_PROBE:
        med = [float(np.nanmedian(spread1[zz][ci])) for ci in range(4)]
        p95 = [float(np.nanpercentile(spread1[zz][ci], 95)) for ci in range(4)]
        print(f"{zz:>5} | " + " ".join(f"{m:9.3f}" for m in med)
              + "   (median)")
        summary["per_z"].setdefault(str(zz), {})["amp_spread_median"] = dict(zip(CLS, med))
        summary["per_z"][str(zz)]["amp_spread_p95"] = dict(zip(CLS, p95))

    print("\n--- STEP 2: SHAPE interaction  corr(δ@α_lo, δ@α_hi)  (1=pure rescale) ---")
    print(f"{'z':>5} | " + " ".join(f"{c:>9}" for c in CLS))
    for zz in Z_PROBE:
        cc = shape_lohi[zz]
        print(f"{zz:>5} | " + " ".join(f"{v:9.4f}" for v in cc))
        summary["per_z"][str(zz)]["shape_corr_lo_hi"] = dict(
            zip(CLS, [float(v) for v in cc]))

    print("\n--- STEP 3: LINEAR x×τ₀ interaction fraction of cosmology SS (median over k) ---")
    print(f"{'z':>5} | " + " ".join(f"{c:>9}" for c in CLS) + "   | median R²(full)")
    for zz in Z_PROBE:
        med_xt = [float(np.nanmedian(frac_xt[zz][ci])) for ci in range(4)]
        med_r2 = [float(np.nanmedian(R2[zz][ci])) for ci in range(4)]
        print(f"{zz:>5} | " + " ".join(f"{m:9.3f}" for m in med_xt)
              + f"   | " + " ".join(f"{r:.3f}" for r in med_r2))
        summary["per_z"][str(zz)]["linear_xtau_frac_median"] = dict(zip(CLS, med_xt))
        summary["per_z"][str(zz)]["linear_R2_full_median"] = dict(zip(CLS, med_r2))

    print("\n--- STEP 4: MODEL response τ₀-spread vs EMPIRICAL σ_cosmo τ₀-spread ---")
    print(f"   (max_α/min_α of the τ₀-trend; MODEL ‖∂lnP̂/∂θ‖ vs EMPIRICAL σ_cosmo, RMS over k)")
    print(f"{'z':>5} {'class':>8} | {'MODEL':>8} {'EMPIR':>8}  ratio(model/empir)")
    for zz in Z_PROBE:
        respn = model_norm[zz]                            # (n_alpha,4)
        semp = sigma_emp[zz]
        emp_norm = np.sqrt(np.nanmean(semp ** 2, axis=2))  # (n_alpha,4)
        rec = {}
        for ci in range(4):
            mr = respn[:, ci]
            er = emp_norm[:, ci]
            with np.errstate(invalid="ignore", divide="ignore"):
                m_spread = float(np.nanmax(mr) / np.nanmin(mr))
                e_spread = float(np.nanmax(er) / np.nanmin(er))
            ratio = m_spread / e_spread if e_spread > 0 else np.nan
            print(f"{zz:>5} {CLS[ci]:>8} | {m_spread:8.3f} {e_spread:8.3f}  {ratio:8.3f}")
            rec[CLS[ci]] = {"model_tau0_spread": m_spread,
                            "empir_tau0_spread": e_spread,
                            "model_over_empir": ratio}
        summary["per_z"][str(zz)]["step4_response_tau0_spread"] = rec

    with open(os.path.join(OUTDIR, "tau0xcosmo_interaction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFIGURES:")
    for p in (*figs1, fig2, fig3, fig4):
        print("  ", p)
    print("  ", os.path.join(OUTDIR, "tau0xcosmo_interaction_summary.json"))


if __name__ == "__main__":
    main()
