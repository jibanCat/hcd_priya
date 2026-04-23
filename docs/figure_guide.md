# Figure reading guide

A collaborator's index of every figure under `figures/analysis/`.
Grouped by the subdirectory organisation introduced in session 3 of 2026-04-22/23:

```
figures/analysis/
├── 01_catalog_obs/        — CDDF, dN/dX, Ω_HI vs observations
├── 02_param_sensitivity/  — 9-parameter sensitivity grids + partial correlation
├── 03_templates_and_p1d/  — Rogers per-class templates + P1D HR/LF convergence
├── 04_hcd_mf/             — HCD multi-fidelity development (14 figures)
└── data/                  — summary HDF5s + CSVs
```

Science narrative is in [`analysis.md`](analysis.md) (§1–6) and
[`hcd_mf_analysis.md`](hcd_mf_analysis.md) (the MF story).  Audit
evidence is in [`bugs_found.md`](bugs_found.md) and
[`masking_strategy.md`](masking_strategy.md).  This doc is just a viewer
index with one short block per figure.

---

## TL;DR tours

**2 minutes (core science)**:
1. `01_catalog_obs/cddf_per_z.png` — does PRIYA match P+14 CDDF?
2. `01_catalog_obs/dndx_hr_vs_lf_vs_obs_per_class.png` + `04_hcd_mf/bootstrap_dndx_per_sim.png` — dN/dX vs obs, LF vs HR, fiducial slice bootstrap.
3. `03_templates_and_p1d/convergence_Tk.png` — P1D resolution convergence.
4. `04_hcd_mf/matched_pair_dndx_hr_vs_lf.png` — HR/LF dN/dX at matched cosmology.

**10 minutes (adds MF machinery)**:
5. `01_catalog_obs/omega_hi_hr_vs_lf_vs_obs.png` — PRIYA vs PW09/N12/C15/Berg+19 Ω_HI overlays.
6. `02_param_sensitivity/param_sens_{dndx,omega_hi}.png` — 9-parameter scan, A_p dominance.
7. `04_hcd_mf/mf_global_fit_visual.png` — G1/G2 global fit at 3 representative z.
8. `04_hcd_mf/mf_slope_stability.png` — the 4th-HR-sim argument.
9. `04_hcd_mf/mf_fourth_sim_test.png` — held-out validation on ns0.914.

---

## 01_catalog_obs/ — catalog vs observations

### `nhi_distribution.png`
Histogram of all 501 M catalog entries across 60 LF sims × 19 z-bins,
colour-coded by class (LLS/subDLA/DLA).  Look for smooth monotonic
decay; no cliffs at the class boundaries 17.2 / 19.0 / 20.3.  Red flag:
gaps or spikes at boundaries (pre-audit signature of the
`_SIGMA_PREFACTOR` bug).

### `cddf_per_z.png`
f(N_HI, X) per z-bin, 60 LF sims stacked, Prochaska+2014 overlaid.
Look for agreement within 0.1–0.3 dex across log N ∈ [17, 22] at every
z.  Red flag: systematic > 0.5 dex offset (pre-audit signature of the
dX bug).

### `dndx_vs_z.png`
dN/dX vs z per class, 60 LF sims.  Red = DLA; PW09 (■), N12 (▲),
Ho21 (⬥) overlaid.  Look at the red curve vs the three obs markers at
z ≈ 2–3.5.  PRIYA LF sits ~1.3–1.6× below obs — real feature, tested
with bootstrap (see 04_hcd_mf/bootstrap_dndx_per_sim.png).

### `dndx_hr_vs_lf_vs_obs_per_class.png`
LLS / subDLA / DLA vs z, both LF suite (circles, 60 sims) and HR suite
(dashed squares, 4 sims), with all observational overlays.  LLS panel
adds Prochaska+10, Fumagalli+13, Crighton+18 etc. from
`~/data/lofz_literature.txt` (dN/dz converted to dN/dX with Planck
cosmology).  HR sits ~20–40 % above LF across all classes; HR DLA
reaches the obs band at z ≈ 3.

### `omega_hi_hr_vs_lf_vs_obs.png`
Ω_HI per class vs z, LF + HR suite, with Berg+19 (XQ-100) as filled
black band and PW09 / N12 / Crighton+15 as data points.  All Ω_HI
here is μ=1 (pure HI mass, no helium).  Literature span:
~0.4–1.0 × 10⁻³.

### `cddf_bugfix_comparison.png`
Audit evidence for bug #7: CDDF at z=3 with old (broken-dX) vs new
(fixed-dX) normalisation, overlaid with P+14.  The old/new offset is
exactly `(1+z)·h ≈ 2.8` at z=3 — predicted by the missing factors.

---

## 02_param_sensitivity/ — 9-parameter scan at z = 3

### `param_sensitivity_LLS.png` / `_subDLA.png` / `_DLA.png`
3×3 grids: raw class counts per 60 LF sims vs each of 9 PRIYA params at
z ≈ 3.  Spearman ρ + p-value in each panel.  The A_p panel always has
ρ ≈ +0.84, p ≈ 10⁻¹⁶.  n_s secondary (+0.35–0.41).  Others near zero.

### `param_sens_dndx.png`
Same 3×9 grid but y-axis = dN/dX (normalised by ΔX).  Signs and ρ
magnitudes nearly identical to the raw-count grids; confirms the A_p
dominance isn't a path-length artefact.

### `param_sens_omega_hi.png`
Same 3×9 grid but y-axis = Ω_HI per class.  **Look at the hub column**
(3rd from left): ρ(hub) = −0.41 to −0.50 across classes — *much*
stronger than the hub signal in dN/dX.  Two compounding causes: the
Ω_HI prefactor contains 1/h; and the prior covaries h with structure
growth.

### `hypothesis_partial_corr.png`
Bar chart, three series per class target.  Blue = raw Spearman(A_p, y);
orange = partial Spearman controlling for n_s; red = Spearman after
dropping highest-A_p sim.  Orange bars are *higher* than blue
(stronger correlation after partialling n_s) — A_p dominance is
robust, not an A_p/n_s covariance artefact.

---

## 03_templates_and_p1d/ — per-class templates + P1D convergence

### `per_class_ratio_vs_z.png`
T_class(k) = P_class_only / P_clean for sim `ns0.803` across 18 z
snapshots, colour-coded by z, in PRIYA angular k.  Dashed black:
Rogers+2018 template with α fit.  Shape agreement good across
emulator range; DLA rises to ~7× at k ≈ 0.001 rad·s/km.

### `per_class_ratio_vs_sim.png`
Same axes but at z = 3, overlaid across 54 LF sims colour-coded by A_p.
Bundles tight — template shape is roughly A_p-independent; most of
the A_p signal is in amplitude (counts/Ω_HI), not in template shape.

### `convergence_Tk.png`
T(k) = P1D_HR / P1D_LF across 3 matched sims × 18 z each, using the
z-matched pairs fix (`|Δz| ≤ 0.05`).  Mid-k: T ≈ 1 ± 0.05 (converged).
High-k: T ≈ 0.65–0.87 (LF can't resolve b ≈ 30 km/s; standard
resolution fall-off).

---

## 04_hcd_mf/ — HCD multi-fidelity development

All 14 figures in this subdir are catalogued in detail in
[`hcd_mf_analysis.md`](hcd_mf_analysis.md).  Compact index here:

### Bootstrap + hypothesis tests
- `hypothesis_dndx_bootstrap.png` — across-sim bootstrap (original
  test, now deprecated in favour of per-sim).  Kept for history.
- `bootstrap_dndx_per_sim.png` — per-sim cosmic-variance bootstrap
  with fiducial-slice highlighting (10 sims nearest eBOSS BF).  **Main
  figure for dN/dX under-prediction significance**.

### HR/LF convergence (matched-pair)
- `matched_pair_dndx_hr_vs_lf.png` — 3 sims, per-class dN/dX vs z, LF
  (solid circle) and HR (dashed square) colour-matched per sim.  DLA
  obs overlaid.
- `matched_pair_omega_hi_hr_vs_lf.png` — same layout, Ω_HI.

### HR/LF convergence (ratio)
- `hf_lf_ratio_scalars.png` — R(z) = Q_HR / Q_LF for 6 scalar
  quantities, per-sim lines with Poisson error bars.  LLS has strong
  z-evolution; subDLA roughly flat; DLA intermediate.
- `hf_lf_template_sim_spread.png` — per-class T_HR / T_LF at z = 3, 3
  sims overlaid.  ±1 % at mid-k, 3–7 % at low-k.
- `hf_lf_template_vs_k_and_z.png` — full z-evolution of
  per-class R(k) across 3 sims × ~18 z, colour-coded by z.

### MF fit development
- `mf_fit_templates_per_k.png` — flat (black) vs linear-in-A_p (red)
  fit of per-class template R(k, z≈3), with RMS improvement in
  bottom row.
- `mf_fit_vs_residuals_z3.png` — 6 scalar quantities at z = 3, scatter
  + flat + linear, residuals below.  Linear literally interpolates
  n = 3 points — 0 % residual by construction, illustrates the
  per-z DOF limit.
- `mf_slope_stability.png` — per-z linear slope vs z for each
  quantity, with leave-one-out envelope.  Every envelope crosses zero
  — the 4th-HR-sim case.

### MF fit production
- `mf_z2_hypothesis_test.png` — σ_observed vs σ_Poisson per quantity
  per z.  Diagnoses parameter-dependent signal vs shot noise.
- `mf_low_z_breakdown.png` — step-by-step MF strategies (flat / linear
  / extrapolated-slope / none) at z-extremes for Ω_HI(DLA).  Table of
  fractional residuals.
- `mf_global_fit_visual.png` — global G1 (1-slope) vs G2 (+ z-drift)
  fit at 3 representative z.  Shows the stable, over-determined fit.
- `mf_fourth_sim_test.png` — held-out validation on ns0.914 (no LF
  counterpart).  `Q_HR_pred = Q_LF_nearest · R_MF` recovers all six
  quantities to 1.5–4.3 %.

---

## data/ — HDF5 summaries + CSVs

### HDF5
- `hcd_summary_lf.h5` — 1076 LF (sim, snap) records, schema in
  `scripts/build_hcd_summary.py`.  Contains per-class counts,
  ΣN_HI, dN/dX, Ω_HI, plus 9 PRIYA params per record.
- `hcd_summary_hr.h5` — 70 HR records, same schema.

### CSVs (machine-readable tables backing figures)

| CSV | Backing figure / script |
|---|---|
| `param_sens_summary.csv` | `02_param_sensitivity/param_sens_*.png` |
| `bootstrap_dndx_per_sim.csv` | `04_hcd_mf/bootstrap_dndx_per_sim.png` |
| `matched_pair_hr_vs_lf_table.csv` | `04_hcd_mf/matched_pair_*.png` |
| `mf_necessity_scalars.csv` | `04_hcd_mf/hf_lf_ratio_scalars.png` verdicts |
| `mf_necessity_templates.csv` | `04_hcd_mf/hf_lf_template_*.png` verdicts |
| `mf_deviation_from_flat.csv` | per-(quantity, z) flat-MF deviation |
| `mf_z2_hypothesis_test.csv` | `04_hcd_mf/mf_z2_hypothesis_test.png` |
| `mf_linear_coefficients_omega_dla.csv` | per-z linear slopes, Ω_HI(DLA) only |
| `mf_linear_coefficients_all.csv` | per-z linear slopes, all 6 quantities |
| `mf_fit_scalars.csv` / `mf_fit_templates.csv` | analytical-fit summaries |
| `mf_global_fit_coefficients.csv` | G1/G2 coefficients per quantity |
| `mf_global_fit_predictions.csv` | G1/G2 predictions per (sim, z) |
| `mf_low_z_breakdown.csv` | z-extreme strategy comparison |
| `mf_fourth_sim_test.csv` | held-out test residuals per (quantity, z, method) |

---

## Figure-lookup cheat sheet

| I want to know… | Look at |
|---|---|
| Does the sim match obs CDDF? | `01_catalog_obs/cddf_per_z.png` |
| Is PRIYA's dN/dX under-prediction significant? | `01_catalog_obs/dndx_vs_z.png` + `04_hcd_mf/bootstrap_dndx_per_sim.png` |
| How does HR compare to LF vs obs? | `01_catalog_obs/dndx_hr_vs_lf_vs_obs_per_class.png`, `01_catalog_obs/omega_hi_hr_vs_lf_vs_obs.png` |
| What drives HCD abundance? | `02_param_sensitivity/param_sens_dndx.png` |
| Is A_p dominance robust? | `02_param_sensitivity/hypothesis_partial_corr.png` |
| Are per-class P1D templates Rogers-compatible? | `03_templates_and_p1d/per_class_ratio_vs_z.png` |
| Is P1D resolution-converged? | `03_templates_and_p1d/convergence_Tk.png` |
| Are HCD scalars resolution-converged? | `04_hcd_mf/hf_lf_ratio_scalars.png` |
| Are HCD templates resolution-converged? | `04_hcd_mf/hf_lf_template_sim_spread.png` + `hf_lf_template_vs_k_and_z.png` |
| Does HCD need multi-fidelity? | `04_hcd_mf/mf_global_fit_visual.png`, `mf_fit_vs_residuals_z3.png` |
| Is the MF correction generalisable? | `04_hcd_mf/mf_fourth_sim_test.png` |
| Why do we need a 4th HR sim? | `04_hcd_mf/mf_slope_stability.png` |
