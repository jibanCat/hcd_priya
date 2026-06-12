# figures/ — reader's guide (code repo)

A map of the **foundational, notebook-referenced** figures that live in this code repo
(`github.com/jibanCat/hcd_priya`). These are the figures the README/notebooks point at.

> **The working/closure diagnostic figures and ALL superpowers notes now live in the
> PRIVATE notes repo** `/home/mfho/hcd_priya_notes` (github.com/jibanCat/hcd_priya_notes):
> `figures/analysis/{05_likelihood, 05_multifidelity, 05_truth_validation}`,
> `figures/analysis/06_clustering`, `figures/analysis/review`, and the whole
> `docs/superpowers/` tree (plans, specs, onboarding/reviews, reflections, walkthroughs).
>
> **Cross-repo link convention** (the two repos are siblings under `/home/mfho/`): from a
> notes doc, foundational figures here are linked `../../../hcd_priya/figures/analysis/<dir>/...`;
> figures that moved to the notes repo are linked `../../figures/analysis/<dir>/...`.

---

## Top-level summary PNGs
- `discovery_summary.png`, `discovery_summary_hires.png` — project headline / discovery summary figures.

## `analysis/01_catalog_obs/` — HCD catalog vs observations
CDDF, dN/dX, N_HI distribution, Ω_HI: simulation (HR/LF) vs observed.

## `analysis/02_param_sensitivity/` — parameter sensitivity
How dN/dX, Ω_HI, and per-class HCD templates respond to cosmological / reionization / astro params.

## `analysis/03_templates_and_p1d/` — HCD templates & P1D
Per-class P1D template ratios vs z and vs sim, Rogers-α template calibration, convergence, Tier-C/τ₀ response.

## `analysis/04_emulator/` — Phase-2b emulator diagnostics (large)
The deployed two-head emulator: LOSO folds, pred-vs-true, CV/C_emu floors, A_p Fisher-bias, Nyquist
masking, convergence, per-fold structure. **NEVER overwrite `emu_bias_allfolds.txt`** (the load-bearing
all-folds emulator-bias reference).

## `analysis/04_hcd_mf/` — HCD multi-fidelity (LF→HR)
Matched-pair LF vs HR dN/dX & Ω_HI, HF/LF template ratios, MF fit/residuals, bootstrap spread.

## `analysis/06_performance_walkthrough/` — finalized emulator walkthrough
Figure-by-figure performance of the deployed Phase-2b emulator (both heads) on held-out data. Has its
own `README.md` (panel-by-panel) and `headline_numbers.json`.

## `diagnostics/` — notebook-referenced diagnostics
Pipeline / recovery / masking / real-space–Fourier sanity checks, plus a `clustering/` subdir.

---

### Not tracked / elsewhere
- `figures/private/` — DESI REAL-DATA cosmology results (gitignored, local only).
- `figures/intermediate/` — scratch / intermediate render output.
- Working diagnostics (`05_*`, `06_clustering`, `review`) + all `docs/superpowers/` notes →
  the private notes repo `/home/mfho/hcd_priya_notes`.
