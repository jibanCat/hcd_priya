# Phase-2b emulator — performance walkthrough (BOTH heads, finalized models)

A figure-by-figure walkthrough of the **deployed** Phase-2b emulator's performance on
**held-out** data, covering **both heads**:

- **Head B** — the Lyα **P1D / cosmology** channel (the θ-blind baseline `m̂(z,τ₀)` +
  the σ_cosmo-whitened cosmology residual `r̂(θ,z,τ₀)`), plus the **multi-fidelity**
  LF→HF high-k correction and the `Δ_c` HCD templates.
- **Head A** — the τ₀-invariant **CDDF `f_NHI`** + **`dN/dX`** incidence channel, which
  feeds the structural class weights `w_c` (→ `P_tier_p`) and the `α_c` prior.

Every panel is generated from the **8 finalized LOSO checkpoints**
`checkpoints/final_fold{0..7}.eqx` (the `FINAL_RECIPE` in `scripts/run_loso_sweep.py`,
loaded via `train.load_checkpoint`) evaluated on each fold's held-out sims, plus the
multi-fidelity layer `hcd_analysis/emulator/multifidelity.py`. The MF figure (B6) uses
the **validated `ρ(k,z)`-only `FixedMeanHead`** (the production default — the learned-MLP
delta-head is the deprecated ablation). Read-only on the production caches; classes are
`clean / LLS / subDLA / DLA`.

**Reconstruction.** `logP_filt = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂`, so the only θ
dependence is through the residual: `∂lnP/∂θ = σ_cosmo·∂r̂/∂θ`.

**Regenerate everything:**
```
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_performance_walkthrough.py
# --no-mf skips the multi-fidelity figure B6 (needs the HR cache + res_corr table)
```

## Headline numbers

| head | metric | value | target / note |
|---|---|---|---|
| **B** | deployed in-range frac-P1D RMS (clean / LLS / subDLA / DLA) | **1.1 / 1.2 / 1.3 / 2.5 %** | clean ≈ CV floor; DLA shot-limited |
| **B** | honest within-cell θ-tracking **corr / spread-ratio** | **0.98 / 1.02** | ≳0.95 / →1.0 redesign gate |
| **B** | deployed whitened cosmology-signal error (tot / σ_signal) | **0.19–0.22 σ** | (a) residual-fit dominated; (b) baseline ≤0.04 |
| **B** | per-fold **A_p** Fisher-bias RMS / max | **0.067 / 0.131 σ** | gate \|bias\| < 0.2σ — all 8 folds pass |
| **B** | per-fold **n_s** Fisher-bias RMS / max | **0.071 / 0.131 σ** | gate \|bias\| < 0.2σ — all 8 folds pass |
| **B** | MF KODIAQ-band frac err: MF vs LF-extrapolated | **0.18 vs 0.77** | MF cuts high-k error ~4× |
| **A** | `dN/dX` median \|frac err\| (LLS / subDLA / DLA) | **1.6 / 1.6 / 1.3 %** | <2.5% target |
| **A** | CDDF `f_NHI` median / p95 \|frac err\| | **2.3 % / 34 %** | p95 driven by the shot-noise tail ≥21.5 |
| **A** | structural `P_tier_p` faithfulness (emu vs true `w_c`) median / p95 | **0.04 % / 0.22 %** | ≤2.5% target — comfortably met |

Raw numbers: `headline_numbers.json`. Source: `scripts/plot_performance_walkthrough.py`.

---

# HEAD B — P1D / cosmology

## B1 — Predicted vs true per-class P1D

![](B1_pred_vs_true_p1d.png)

Reconstructed **linear** P1D (`reconstruct_P_filt`) vs the cache for the 3 best-sampled
held-out fold-0 sims per class, log-log, with a `pred/true − 1` subpanel; the grey band
marks below the DESI data `k_min` (1e-3 s/km). **How to read:** dashed (pred) should
overlay solid (true) and the subpanel should hug zero within the resolved range.
**Good/bad:** here the residuals sit within ±0.1 and turn over only at the per-row
Nyquist edge — small, roughly k-flat residuals are good; a coherent k-tilt would flag a
shape bias.

## B2 — Deployed per-class fractional P1D error vs k (in-range, all folds)

![](B2_deployed_frac_err_vs_k.png)

The **deployed** `|pred/true − 1|` (median line, IQR shaded) over **all 8 folds'**
held-out rows, restricted to the DESI data range (z∈[2.2,4.6], k≥k_min), with the
**cosmic-variance floor** overlaid (dashed black). **How to read:** the in-range RMS is
clean **1.1%**, LLS **1.2%**, subDLA **1.3%**, DLA **2.5%**. **Good/bad:** the clean
error sits essentially **at the CV floor** through the mid-k forest band (the irreducible
sampling scatter, not a model defect); the larger DLA error is the shot-noise-limited
3%-of-sightlines class. Error growing only at the band edges (continuum / Nyquist) is the
expected, benign behavior.

## B3 — Learned cosmology response ∂lnP/∂θ vs k (all 9 params)

![](B3_cosmology_response.png)

`σ_cosmo·∂r̂/∂θ` via `jax.jacfwd` for **all 9 cosmology params**, per class, at a
representative held-out point — the differentiable readout HMC consumes. **How to read:**
the panel title shows the in-range RMS response amplitude (how strongly the P1D moves
with that param). **Good/bad:** `n_s` (tilt) **flips sign across k** — pivoting the P1D
shape — while `A_p` and `hub` carry the largest broadly-coherent amplitude responses; the
slow-reionization / feedback params (`herei`, `heref`, `bhfeedback`) have small, smooth
responses. Smooth, structured, k-resolved sensitivities (not flat or noisy) are exactly
what a gradient-based likelihood needs; which params the P1D constrains, and where in k,
is read directly off the amplitudes and shapes.

## B4 — Within-(z,τ₀)-cell θ-tracking + honest (a)/(b) decomposition

![](B4_theta_tracking.png)

**Left:** the **honest** within-cell θ-tracking — predicted vs true log-P1D deviation
**referenced to the deployed baseline `m̂(z,τ₀)`** (NOT a val-estimated mean, which would
inflate the metric); corr **0.98**, spread-ratio **1.02**. **Right:** the honest
decomposition of the deployed whitened cosmology-signal error into **(a)** the
residual-head fit error and **(b)** the σ_cosmo-amplified baseline mis-fit, stacked per
class. **How to read:** the total bar (black diamond) is the deployed error in units of
the cosmology-signal σ; the `|P̂/P−1|` annotation is the corresponding fractional P error.
**Good/bad:** the total is **0.19–0.22 σ_signal** and is **(a)-dominated** — the baseline
mis-fit (b) is ≤0.04 σ, i.e. the θ-blind baseline is not leaking into the cosmology
budget. A tight diagonal + an (a)-dominated, small total is the redesign working.

## B5 — Per-fold A_p & n_s Fisher-bias (the 8-fold spread)

![](B5_fisher_bias_perfold.png)

The inference **gate**: the deployed model's A_p and n_s Fisher-bias (in-range modes, a
DESI-DR1-like diagonal per-mode covariance, fiducial z=3) for **each of the 8 LOSO
folds**. **How to read:** every bar should sit inside the green ±0.2σ gate; the dotted
lines are the across-fold RMS. **Good/bad:** **all 8 folds pass** for both params — A_p
RMS **0.067σ** (max 0.131σ), n_s RMS **0.071σ** (max 0.131σ). A robustly small, sign-mixed
spread (no systematic offset) is the win the residual-head tuning was designed for; a fold
poking outside the gate, or a coherent same-sign bias, would flag a deployment risk.

## B6 — Multi-fidelity high-k: MF vs LF-extrapolated vs HF-standalone

![](B6_mf_high_k.png)

The **multi-fidelity** payoff at high k. HF-LOSO RMS fractional error vs k (RMS over the 6
held-out HF sims), comparing the **MF** (`ρ(k,z)`-only `FixedMeanHead`, blue) to two
baselines: the **LF backbone extrapolated** past its Nyquist (red dashed) and a
**HF-standalone** 6-sim emulator surrogate (green dotted). The shaded band is the **KODIAQ
reach** 0.07–0.2 s/km; the dash-dot line is the LF Nyquist (~0.069). **How to read:** in
the KODIAQ band the MF should sit well below both baselines. **Good/bad:** it does — MF
KODIAQ-band frac err ≈ **0.18** vs LF-extrapolated **0.77** and HF-standalone **0.59–0.82**
(a ~4× reduction). The MF turns the unreliable LF tail extrapolation into a usable high-k
prediction; the HF-standalone is the hopeless few-sim alternative MF beats.

## B7 — Δ_c HCD class templates vs k / z

![](B7_delta_c_templates.png)

The `Δ_c` HeadB output — the per-HCD-class (LLS/subDLA/DLA) template that re-weights the
structural `P_tier_p` — shown across 4 held-out z slices, **symlog-y** so both the
large low-k structure and the small in-range tail are legible (grey = below data `k_min`).
**How to read:** each curve is one z; the amplitude grows with class (DLA ≫ LLS) and the
template has a **physical low-k sign-flip**. **Good/bad:** smooth, z-ordered templates that
decay into the resolved forest band are healthy; the structure is concentrated at the
lowest (out-of-range) k where the HCD contamination is largest, exactly as expected.

---

# HEAD A — dN/dX + CDDF

## A1 — dN/dX predicted vs true, per class vs z

![](A1_dndx_pred_vs_true.png)

Head-A `dN/dX` fractional accuracy (median + p95 over the held-out rows of all 8 folds)
vs redshift, per HCD class, with the 2.5% target line. **How to read:** the median
(blue) should sit at/under 2.5% across the DESI z range. **Good/bad:** median ≈ **1.3–1.7%**
in-range for all three classes; the error rises only at the z extremes (z<2.2 and z>4.6,
outside the DESI window) where the sims are count-limited — the expected, benign edge
behavior.

## A2 — f_NHI / CDDF predicted vs true vs logN_HI

![](A2_cddf_pred_vs_true.png)

Head-A `f_NHI` (the column-density distribution function) fractional accuracy per
log-N_HI bin (median + p95), with the **class boundaries** marked (LLS onset 17.2,
LLS|subDLA 19.0, subDLA|DLA 20.3) and the **shot-noise tail** (logN_HI ≥ 21.5) shaded.
The gray curve (right axis) is the valid-row fraction. **How to read:** the median should
be a few % across the well-sampled range and is allowed to blow up only in the shaded
tail. **Good/bad:** median **2.3%** overall, ~1–2% across 17.5–21; the p95 of **34%** is
**driven entirely by the shaded shot-noise tail** (≥21.5), where the rarest absorbers are
under-sampled — flagged, not hidden.

## A3 — w_c → P_tier_p coupling + faithfulness

![](A3_wc_ptierp_coupling.png)

The structural coupling: **left** — `w_c` from **emulated** `dN/dX` vs `w_c` from **true**
`dN/dX` (the round-trip); **middle** — the `|Δw_c|` distribution per class against the 2.5%
line; **right** — the resulting **structural `P_tier_p` faithfulness** (`P_tier_p` built
with emulated vs true `w_c`, same true P_filt). **How to read:** the round-trip should lie
on the diagonal and the `P_tier_p` error should sit inside ±2.5% (green). **Good/bad:**
`P_tier_p` frac-err median **0.04%**, p95 **0.22%** — ~10× inside the ≤2.5% target, so the
emulated `dN/dX` propagates into the observable with negligible structural distortion.

## A4 — Head-A error heatmap: class × z × fold

![](A4_head_a_error_heatmap.png)

The per-(class, z, fold) `dN/dX` median \|frac err\| (%) as a heatmap — one panel per HCD
class, fold on the y-axis, z on the x-axis, shared color scale. **How to read:** a uniform
cool field means the error is well-balanced across folds and redshifts; a hot **row** flags
a weak fold, a hot **column** a weak z. **Good/bad:** the field is mostly cool (~1–2%) with
hot cells only at the z>5 / z<2.2 extremes (outside the DESI range) and a few low-z
subDLA/DLA cells (count-limited) — no systematically bad fold, which is the cross-validation
robustness check.

## A5 — dN/dX → α_c prior-center sanity

![](A5_alpha_prior_sanity.png)

Is the emulated `dN/dX` a sensible **center for the α_c prior**? **Left** — the emulated
`w_c` (the prior center, via `w_c_corrected`) vs the cache's empirical **CDDF-integral**
`w_c`. **Right** — the per-class `|w_c(emu) − w_c(cache)|` distribution against the 2.5%
line. **How to read:** the scatter should track the diagonal and the deviations should sit
under 2.5%. **Good/bad:** they do — the emulated prior center matches the independent CDDF
integral to median ≈ 0.1–0.4% per class (almost all rows under 2.5%), so the HCD prior is
anchored to the data, not the model's own bias.

---

## Notes / caveats

- **No figure was skipped** — all 12 generated cleanly from the finalized models + the MF
  layer.
- **B1, B3, B4, B7** use **fold-0** held-out sims as the representative single-fold view;
  **B2, B5, A1–A5** aggregate **all 8 LOSO folds**' held-out rows.
- The MF figure (**B6**) uses the **validated `ρ(k,z)`-only `FixedMeanHead`** — the
  production default. The learned-MLP delta-head over-fits the θ-gradient on 6 HF sims and
  is the deprecated ablation (see `hcd_analysis/emulator/multifidelity.py`).
- Fisher bias (**B5**) and the cosmology response (**B3**) assume a **DESI-DR1-like
  diagonal per-mode covariance** (clean 3%, HCD 15%); they are a deployment sanity gate,
  not the final survey-covariance inference.
- The Head-A high-N_HI tail (**A2** ≥21.5) and the off-DESI z extremes (**A1/A4**) are the
  known count-limited regions — flagged, not structural.
- Source: `scripts/plot_performance_walkthrough.py`; raw numbers in
  `headline_numbers.json`; finalized checkpoints `checkpoints/final_fold{0..7}.{eqx,meta.json,norm.pkl}`.
