# Local grounding: head design + loss + tau0-n_s correlation (feedback #1, #4)

Date: 2026-06-08. Agent: local grounding (no web). Scope: read the actual code +
find the tau0-vs-n_s correlation in the lya_emulator_full MCMC chains.

## (A) Head design (feedback #1) — IS it "first predict (tau0,z) mean, then cosmology around it"? YES.

Files: `hcd_analysis/emulator/model.py`, `hcd_analysis/emulator/predict.py`.

- `BaselineHead(z, tau0)` (model.py L65-135): input is ONLY `[z, tau0]` (2-dim),
  theta-BLIND by construction (never sees the latent / the 9 cosmology+IGM params).
  Output = sigma_marg-standardized (z,tau0)-CONDITIONAL-MEAN spectrum m_hat for the 4
  P_filt classes. Because theta never enters, d(m_hat)/d(theta) = 0 exactly.
- `HeadB(concat(latent, tau0))` (L138-208): output is the theta-DEPENDENT RESIDUAL
  r_hat (4,K) in sigma_cosmo-whitened units, PLUS 3 HCD deltas. Key returned:
  `P_filt_resid`.
- Reconstruction (`predict.reconstruct_P_filt_jax`, predict.py L29-39):
  `logP = (m_hat*sig_marg + mu_marg) + sig_cosmo * r_hat ; P_filt = exp(logP)`.
  So ALL cosmology flows through r_hat: `dlogP/dtheta = sig_cosmo * dr_hat/dtheta`.

So the user's framing is EXACTLY right: the design is a two-stage
(theta-blind tau0,z CONDITIONAL MEAN) + (cosmology residual ON TOP). tau0 enters BOTH
stages (BaselineHead gets it directly; HeadB gets it concatenated to the latent).

### Is there a tau0-conditional (low vs high tau0) accuracy diagnostic? YES — one.

`scripts/embias_arch_tau0regime.py` -> `figures/analysis/04_emulator/embias_arch_tau0regime.{npz,png}`.
It bins the CLEAN-class LOW-k (kf<3e-3) coherent fractional pred-true error by the
tau0-LADDER FACTOR (the rung alpha), for fold-0 held-out sims, AND decomposes the
log error into baseline-head vs residual-head. Saved results:

- clean low-k frac vs tau0 rung (alpha center -> %):
  0.69 -> -0.275 | 0.78 -> -0.204 | 0.89 -> -0.194 | 0.98 -> -0.202 |
  1.05 -> -0.115 | 1.14 -> -0.128 | 1.23 -> -0.115 | 1.30 -> -0.076
  => MONOTONE-ish: the (under)error is WORST at LOW tau0 rung (~-0.28%) and best at
  HIGH tau0 rung (~-0.08%). This is direct evidence FOR the user's worry: accuracy is
  tau0-DEPENDENT and degrades at LOW tau0.
- the becker13 closure uses only INTERIOR rungs (alpha 0.976-1.118); the full ladder
  spans 0.656-1.331. So the closure is anchored where the error is mid-range, not at
  the worst low-tau0 edge.
- baseline-head vs residual-head decomposition (becker rows, clean low-k LOG err):
  total -0.113% | baseline-head +0.024% | residual-head -0.136%.
  => the LOW-k error is dominated by the RESIDUAL head (HeadB), NOT the baseline head.
  The theta-blind baseline is essentially exact (+0.024%); the cosmology residual r_hat
  carries the bias.

CAVEAT: this diagnostic is about CLEAN-class LOW-k (the A_p / amplitude investigation),
NOT the n_s TILT directly. There is NO diagnostic that splits the n_s bias by low-vs-high
tau0. So the *mechanism* "low-tau0 baseline inaccuracy -> (tau0-n_s degeneracy) -> n_s bias"
is plausible and partially supported (accuracy IS worse at low tau0), but it is the RESIDUAL
head, not the baseline, that carries the low-k error in the one diagnostic that exists.

## (B) Loss (feedback #4) — explicit (z,tau0)-mean loss? YES, and it is FROZEN by default.

File: `hcd_analysis/emulator/train.py`.

- EXPLICIT baseline (conditional-mean) loss EXISTS and is SEPARATE from the residual:
  - `joint_loss` (model.py L387-456) splits the P_filt term into TWO:
    - `p_base` = masked MSE of BaselineHead vs `t_p_base` (the sigma_marg-standardized
      (z,tau0)-conditional CELL-MEAN), inv_nc-weighted (model.py L437).
    - `p_resid` = masked MSE of HeadB residual vs `t_p_resid` (sigma_cosmo-whitened
      cosmology signal), inv_nc + optional k_weight + datarange weight (L441-447).
  - five terms total: f_nhi, dndx, p_base, p_resid, delta; default term_w UNIFORM.
- STAGED SCHEDULE (train.py `train_fold` L271-..., `_prefit_baseline` L165-209):
  1. `_baseline_cell_table` (L137-162) collapses train rows to ONE entry per distinct
     (z,tau0) cell (~306 cells on LF); target = sigma_marg-standardized train-cell-mean.
  2. `_prefit_baseline` (L165-209): trains ONLY `head_base` (encoder/HeadA/HeadB frozen)
     for `prefit_baseline_epochs` (default 8000) on that cell table, with a plain
     masked-mean MSE (UNIFORM weight, no inv_nc shrink), cosine-decay LR.
  3. JOINT loop (L429+): by default `freeze_baseline = did_prefit` (L370-371) -> since a
     prefit ran, the baseline is FROZEN (partitioned out via eqx, L405-408) during the
     joint fine-tune. The joint loop then trains ONLY encoder / HeadA / HeadB residual.
  - Early stop: "auto" -> on the val RESIDUAL (p_resid) loss when the baseline is frozen
    (L376-377).
  - Production driver `scripts/run_loso_sweep.py`: w_coh=80.0, weight_decay=3e-4,
    datarange=True, k_weight (gain3/lowk2) — i.e. prefit+freeze baseline ON.

- DOES the design GUARANTEE the (tau0,z) conditional mean is correct, or can fine-tune
  pull it off?
  - In the DEFAULT (production) path the baseline is FROZEN during fine-tune, so the
    joint loss CANNOT pull it off its prefit floor. The docstring (L288-296) is explicit:
    "if the baseline kept training jointly it DRIFTS off its floor" — that is precisely
    why they freeze it. So the conditional mean is pinned, NOT guaranteed-correct: it is
    only as correct as the 8000-epoch prefit on the ~306-cell table got it (claimed
    ~0.03-0.05*sig_cosmo term-(b) floor). Under LOSO the cell-mean target is the
    TRAIN-cell-mean (shared across 60 cosmologies), so the held-out-sim true conditional
    mean can differ from the train-cell-mean — a finite-60 generalization gap that the
    freeze does NOT close (it just stops fine-tune from making it worse).
  - There IS a non-default path (`freeze_baseline=False`, or `prefit_baseline_epochs=0`)
    where the joint loop trains the baseline jointly and CAN pull it off — the inv_nc
    weighting + the much larger p_resid/f_nhi terms would drive it. This is the failure
    mode the freeze guards against.
  - `coherent_debias_term` (model.py L342-384, w_coh=80 in prod) penalizes the
    per-(z,tau0)-CELL theta-MEAN of the RESIDUAL error -> flattens the coherent k-tilt of
    HeadB. It operates on the residual, not the baseline, but it IS a (z,tau0)-cell-mean
    constraint and is the closest thing to a tilt-tied loss term.

## (C) tau0-n_s correlation magnitude (feedback #1) — STRONG NEGATIVE, ~-0.6 to -0.8.

The user's "mcmc chains of simdat in lya_emu_full" are cobaya chains. They are NOT under
`/home/mfho/lya_emulator_full/` (the `chaindir="chains/"` is relative; the runners' commented
`simdat/mf-48-48-z2.2-4.6` lines confirm the naming). The actual saved chains are at:

- `/home/mfho/student_projects/InferenceLyaData/Chains/{fps-only,fps-meant,meant-only}/`
- `/home/mfho/Latex/kodiaq_emu/figures/chains_xq100_only/...` (xq100)

These are PRIYA mean-flux ("s") closure/data cobaya chains. Columns (from `.1.txt` header
and `.updated.yaml` `input_params`): weight, minuslogpost, **dtau0, tau0, ns, Ap**, herei,
heref, alphaq, hub, omegamh2, hireionz, bhfeedback, a_lls, a_dla, fSiIII. Here `tau0` = the
mean-flux AMPLITUDE (prior [0.75,1.25]) and `dtau0` = the mean-flux SLOPE; `ns` prior is
[0.8, 1.05] — IDENTICAL to data.py PARAM_LIMITS. `Ap` prior [1.2e-9, 2.6e-9].

WEIGHTED correlations (computed from the existing chain files, NO new MCMC run):

| chain (mean-flux 's')                      | corr(ns,tau0) | corr(ns,dtau0) | corr(ns,Ap) | corr(Ap,tau0) | ns_mean |
|--------------------------------------------|---------------|----------------|-------------|---------------|---------|
| fps-meant/mf-48-48-z2.2-4.6 (simdat,meanT) | **-0.562**    | +0.172         | +0.180      | -0.668        | 0.896   |
| fps-meant/mf-48-48-z2.6-4.6 (simdat,meanT) | **-0.747**    | +0.302         | +0.422      | -0.707        | 0.984   |
| fps-meant/mf-48-48-z2.6-4.6-gpemu (+emuerr)| **-0.696**    | +0.249         | +0.447      | -0.808        | 0.991   |
| fps-only/mf-48-dr9-z2.2-4.4                | **-0.608**    | +0.156         | +0.255      | -            | 0.900   |
| fps-only/mf-48-dr9-z2.6-4.4                | **-0.658**    | +0.329         | +0.279      | -            | 1.017   |
| fps-only/mf-48-z2.2-4.6                    | **-0.637**    | +0.146         | +0.328      | -            | 0.898   |
| fps-only/mf-48-z2.6-4.6                    | **-0.781**    | +0.485         | +0.451      | -            | 1.008   |

So corr(ns, tau0) is robustly **negative, ~-0.56 to -0.78** (median ~-0.66) across all
7 chains. This is the field-standard mean-flux-AMPLITUDE / spectral-TILT degeneracy: raising
the mean-flux amplitude (tau0) flattens P1D, which the fit compensates by lowering n_s, and
vice versa. corr(ns,dtau0) (the SLOPE) is mildly POSITIVE (+0.15..+0.49). corr(Ap,tau0) is
even stronger negative (-0.67..-0.81).

DIRECT BEARING on the -0.65sigma n_s under-prediction: the emulator's coherent n_s UNDER-
prediction (the bias the all-folds LOSO found) maps, via this ~-0.66 ns-tau0 degeneracy, onto
a coherent mean-flux/tau0 mis-set of the SAME order in the orthogonal direction. A residual-
head that is biased at LOW tau0 (per the tau0regime diagnostic) tilts P1D, and with
corr(ns,tau0)~-0.66 that tilt is absorbed (in inference) ~2/3 into a wrong n_s. The
magnitude (-0.65sigma n_s) is entirely consistent with a sub-percent tau0-regime tilt
bias being re-expressed through this degeneracy.

How to recompute (already done): load the `.1.txt`, take weight=col0, and a weighted Pearson
of cols `ns` vs `tau0`. The `.covmat` files also store the cobaya covariance directly.

## (D) PARAM_LIMITS + how ns>1.0 was sampled.

- `hcd_analysis/emulator/data.py` L31-32: `PARAM_LIMITS[0] = [0.8, 1.05]` (ns), matching
  `emulator_params.json` param_limits[0]. Confirmed. (Doc: `docs/superpowers/2026-06-01-param-limits.md`.)
- HOW ns>1.0 was sampled (feedback #2 relevance): the 60 PRIYA design points
  (`/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json` -> `sample_params`,
  shape (60,9), ns = col 0) have ns in **[0.8032, 1.0396]** but ONLY **2 sims with ns>1.0**
  (1.0188, 1.0396) and **3 with ns>0.995**. The top values are
  0.9788, 0.9788, 0.9822, 0.9852, 0.9918, 0.9979, **1.0188, 1.0396** — note the visible GAP
  between 0.9979 and 1.0188.
- The grid generator (`lya_emulator_full/lyaemu/coarse_grid.py` `build_params` L314-328 ->
  `latin_hypercube.get_hypercube_samples` with `prior_points`) is a MAXIMIN Latin hypercube
  that SUPPORTS augmentation: `prior_points` lets you ADD new sims that respect existing
  strata. The repo ALSO ships `lyaemu/bayesian_opt.py` ("Acquisition functions for selecting
  the next simulation point", per `docs/REPO_MAP.md` L57/L70). So the design is a base LHS
  that CAN BE / WAS extended by Bayesian-optimization-selected refinement points. The sparse,
  gap-separated 2 points above ns=1.0 are consistent with BO-extension corner points, NOT a
  space-filling LHS draw (a uniform LHS over [0.8,1.05] would put ~10-12 of 60 above 1.0; only
  2 are). => feedback #2's concern is WELL-GROUNDED: the ns>1.0 region is barely populated and
  likely BO-extension, so "backed by training points by construction" (the C1 claim) is weak
  exactly where the box is widest. No repo doc records the BO-vs-LHS provenance of these 2
  points explicitly; it must be inferred from sample_params + bayesian_opt.py's existence.

## Key file paths

- model.py: `/home/mfho/hcd_priya/hcd_analysis/emulator/model.py` (BaselineHead L65, HeadB L138,
  joint_loss L387, coherent_debias_term L342)
- predict.py: `/home/mfho/hcd_priya/hcd_analysis/emulator/predict.py` (reconstruct L29)
- train.py: `/home/mfho/hcd_priya/hcd_analysis/emulator/train.py` (_baseline_cell_table L137,
  _prefit_baseline L165, train_fold L271, freeze L370/L405)
- tau0 diagnostic: `/home/mfho/hcd_priya/scripts/embias_arch_tau0regime.py` +
  `/home/mfho/hcd_priya/figures/analysis/04_emulator/embias_arch_tau0regime.npz`
- chains: `/home/mfho/student_projects/InferenceLyaData/Chains/fps-meant/mf-48-48-z2.2-4.6.1.txt`
  (+ .updated.yaml, .covmat), `.../fps-only/*.1.txt`
- PRIYA design box: `/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json`
- grid gen: `/home/mfho/lya_emulator_full/lyaemu/coarse_grid.py`,
  `/home/mfho/lya_emulator_full/lyaemu/latin_hypercube.py`,
  `/home/mfho/lya_emulator_full/lyaemu/bayesian_opt.py`
