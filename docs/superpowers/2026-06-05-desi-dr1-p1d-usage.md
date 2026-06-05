# DESI DR1 Lyα P1D — likelihood usage recipe

From Karaçaylı 2025 (measurement, arXiv:2505.07974, §4.1–4.5, §5.4, §D) + the cosmology
companion (arXiv:2601.21432, §2.1, §4.1, §4.3). Data: `/home/mfho/data/desi_dr1_p1d/`
(npz via `scripts/convert_desi_dr1_p1d.py`). KS (Karaçaylı 2021) is the same pipeline.

## Decisions that change the FORWARD MODEL

1. **Metals — KEEP a SiIII (+SiII) oscillation term; add NO extra covariance.**
   - SB1-subtraction only removes metals at λ≳1300 Å (redward of Lyα). SiIII (1206.5 Å), SiII,
     and inter-forest metals are NOT removed → must be forward-modelled.
   - Form (companion Eq. 4.3): multiply theory P1D by `1 + f_metals(k)·exp(−k²/2k_s²)`,
     `k_s = 0.009 s/km`, with the Lyα–SiIII term `a²_SiIII + 2 a_SiIII·cos(k·Δv_LyαSiIII)`
     (McDonald form — exactly the planned term). Add SiII similarly; optionally MgII/CIV doublets
     `a_d(1.25+cos(k·v_d))exp(−k²/2k_s²)`. Strong detections: a_SiIII∈[0.03,0.065] z<3.9,
     a_SiII∈[0.025,0.045] z<3.5. a_SiIII, a_SiII are free nuisances (per z or smooth).
   - **C_metal: do NOT add our own.** The SB1 statistical error is already folded in
     (`C_final = C_Lyα + C_SB1`); residual metals are handled by the model term, not by C.

2. **Resolution — the data is DECONVOLVED; do NOT convolve the model.**
   - QMLE deconvolves the per-spectrum resolution matrix + survey window internally. Compare
     theory directly (no window matrix).
   - The residual resolution UNCERTAINTY is `E_RESOLUTION` (a ~1.5% bias dominating k≳0.01),
     already a correlated mode in `COVARIANCE_SYST`. Two options:
     (a) keep it in C (simplest), OR (b) template-marginalize: remove the resolution mode from C
     and multiply theory by `exp(2·b_res·k²·R_z²)`, `b_res ~ N(0, 1.5%)`, `R_z = c·0.8Å/((1+z)·1215.67Å)`
     (companion Eq. 4.8: `C_res = 1 + 2 f_res R_z² k²`, one f_res per z). Recommended for tightest constraints.

3. **Continuum** — already applied (`contcorr`); add NO continuum nuisance. Residual uncertainty is in
   C via `E_CONTINUUM`/`E_CONTINUUM_ADD`. It sets the low-k cut.

## Covariance + cuts

- **Use the full `COVARIANCE` (= STAT + SYST).** The systematics are correlated error-mode templates
  already summed into `COVARIANCE_SYST` → one matrix inversion, no separate handling (unless you
  template-marginalize a mode — then subtract it from C + add the model term, e.g. resolution above).
- **`cov_diag_inflation` is ADDED to the diagonal** (variance units) to reach χ²_ν∼1 (no Hartlap; C is
  bootstrap-regularized). Alternative: a flat 5% inflation of the STAT errors (companion §2.1). Pick one.
- **k cut:** `1e-3 < k < 0.5π/R_z` (z-dependent high cut = half-Nyquist resolution). Drop k<1e-3 (continuum
  floor) and the highest-k bins at high z.
- **z:** fit z = 2.2–4.2 (11 bins); optionally drop z=4.4.

## Columns (the npz keys)
- `plya` — THE DATA VECTOR (fit it directly). "Blind" = analysis-blind provenance, NOT a numerically
  blinded vector; `pinput≠plya` is expected (`pinput` is the injected fiducial). **No unblinding step.**
- `pfid` — estimator fiducial (for inv-cov weighting); cancels, model need not match.
- `psmooth` — smooth fit used to scale syst templates; not a data vector.
- `praw`,`pnoise` — diagnostics (`plya ≈ praw − pnoise − P_SB1`, contcorr applied).
- `e_pk/e_stat/e_syst` — diagonal errors (full matrices = the cov HDUs).
- `syst_*` — the 7 systematic modes (DLA/BAL completeness, resolution, continuum×2, noise×2), all already in `cov_syst`.

## One-line
Drop k<1e-3 & k>0.5π/R_z & (opt) z=4.4 → Gaussian likelihood with **full COVARIANCE** (+ `cov_diag_inflation`
on the diagonal) → compare **deconvolved** theory directly (no window) → ×`(1+f_metals·exp(−k²/2k_s²))`
(SiIII+SiII McDonald) → resolution either in C or ×`exp(2 b_res k² R_z²)`. **No separate C_metal.**

## For Leg-B (the closure)
Only the COVARIANCE is needed (mock noise ~ N(0, C_desi); `plya` not used). The metal/resolution MODEL
terms still matter for the A4 (metals) / resolution stress arms (mock injects them, fit marginalizes).
KS (conservative mode) already SUBTRACTS metals/continuum/resolution + inflates C → its treatment differs
from DESI's (model-the-systematic); reconcile per-leg when wiring both.
