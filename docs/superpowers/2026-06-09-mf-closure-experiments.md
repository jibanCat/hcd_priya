# MF closure experiments — running reference log

**Purpose:** a tracked, figure-rich log of the multi-fidelity (MF) Leg-B closure experiments, so each run
(setup, cost, recovery, verdict) is reproducible and comparable later. Append new runs to §Log.
Image paths are relative to `docs/superpowers/` (i.e. `../../figures/...`).

**Branch:** `phase2c-likelihood`. **Env:**
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## 0. What "MF closure" means here

The Leg-B closure fits a held-out **sim** (truth) on the real DESI+KS grids and asks whether NUTS recovers
(A_p, n_s, τ₀) with **coverage ≥ nominal + |bias| < 0.2σ**. "MF closure" = the same, but the forward goes
through the **multi-fidelity correction** (`MultiFidelity.logP_mf`: frozen LF backbone + the τ₀+z-resolved
`g(z,τ₀,k)` + `res_corr`) instead of the bare LF emulator, with the **small-scale C_emu floor** added to
`C_total`. This is the real arbiter that supersedes all the forward-only Fisher scouting.

**Wiring (committed):**
- `data_likelihood.predict_P_obs_on_leg(..., mf=, mf_floor=)` — opt-in MF forward + floor; `mf=None` is
  byte-identical to the LF path (golden/closure unperturbed).
- `data_likelihood`: `MFFloor` + `load_mf_floor()` + `_mf_floor_var_on_k` — adds `(σ_floor·P_obs)² +
  (σ_edge·P_obs)²` to the C_emu diagonal. **Applied on the small-scale leg only** (KS `mf_floor_on=True`,
  DESI `False`). **PI invariant:** the legs only reach z=4.6 (DESI 4.2, KS 4.6), so the floor is indexed
  only on z≤4.6 — the z>4.6 floor cells (incl. the flagged single-sim z=5.4 spike) are **never read**
  (a runtime `assert leg.z.max() ≤ 4.6` guards it). The n_s-edge term's `ns` is `stop_gradient`'d.
- `closure_legb`: `LegBCtx.mf/mf_floor`, `build_mf_correction(fold)` (reuses the certified gate
  construction; `final_fold0` model verified byte-identical to `load_lf_backbone(0)`), MF applied to the
  mock **truth** as well as the forward (the gate invariant, so the correction cancels in the closure ΔP).
  CLI: `--mf` / `--no-floor`.

---

## 1. Run MF-SMOKE-01 — profiling mock (2026-06-09)

**Goal:** profile one MF closure mock to (a) confirm the MF+floor closure path runs end-to-end and is
NUTS-healthy, and (b) extrapolate the cost of a full coverage run before requesting compute. **SMOKE scale**
(few warmup/samples, max_tree_depth capped) — explicitly **not** a coverage verdict.

**Config:** fold 0, held-out sim n_s=0.803; 1 chain, ~60 warmup + 60 samples, dense mass (dim 25 =
θ9 + τ₀-ladder(13) + α(3)), mtd=5 (smoke). Script: `scripts/profile_mf_closure_smoke.py`.

### 1.1 Sampler health + cost

| metric | value |
|---|---|
| NUTS wall (smoke, compile-dominated) | 83.5 s |
| divergences | **0 / 60 (0.0%)** |
| mean accept_prob | 0.986 |
| dense-mass dim | 25 |
| ESS / sample | n_s 0.30, A_p 0.35, τ₀(z≈3) 0.99 |
| leapfrogs/sample | 31 (≈22.5 ms/leapfrog) |

**Cost extrapolation** (target ESS≈400/param; production mtd=10 → leapfrog inflation ×4.7; wall≈CPU-h;
embarrassingly parallel over mocks): **≈2.4 CPU-h/mock → N=99 ≈ 245 CPU-h, N=600 ≈ 1490 CPU-h** (incl. the
postprocess fix below). **Both are well within the ~4000 CPU-h cavestru0 budget.** 0 divergences + 0.99
accept ⇒ the 25-dim dense-mass geometry is healthy through the MF forward.

> **Efficiency flag for the production run:** numpyro `mcmc.get_samples()` JIT-replays the full MF
> likelihood to recover the `deterministic` sites (`tau0_vec`, `alpha_hcd_z`, `alpha_dla`) — **237.6 s/mock**,
> ~3% of the production per-mock cost but pure waste at smoke scale. Fix before the big run: drop the
> `numpyro.deterministic` sites and reconstruct host-side (`tau0_vec = alpha_ladder·Kim(z)`), or cache
> `postprocess_fn`. The likelihood/floor code is unaffected.

### 1.2 Whitened residual at truth — C_total is correctly sized

![whitened residual at truth](../../figures/analysis/05_likelihood/mf_closure_whitened_resid.png)

DESI χ²/dof = **1.24**, KS χ²/dof = **0.83**. Both ≈1 ⇒ the C_total (incl. the MF floor on KS) fits the
data-vs-model residual within the covariance at the truth. KS slightly <1 ⇒ the floor is mildly
conservative (acceptable). No coherent k-trend ⇒ the MF correction left no structured residual at truth.

### 1.3 The C_emu floor lifts only the KS small-scale leg

![C_emu floor on each leg](../../figures/analysis/05_likelihood/mf_closure_cemu_on_leg.png)

DESI (left, `floor_on=False`): MF+floor (red) overlies LF (grey) — **no floor on DESI**, as intended. KS
(right, `floor_on=True`): the floor raises C_emu/C_data to the spec'd ~1.2% (LF-resolvable) → few-% level
across the KS k-band — the small-scale generalization budget, only where it belongs.

### 1.4 Recovery (n=1, ESS≈20 — NOT a verdict)

![smoke posterior vs truth](../../figures/analysis/05_likelihood/mf_closure_smoke_posterior.png)

n_s 0.808±0.003 (truth 0.803), A_p 1.89e-9±7e-11 (truth 2.20e-9), τ₀(z≈3) 0.404±0.004 (truth 0.405). The
posteriors are visibly **multimodal/unconverged** (ESS≈18–21 from 60 samples) — the σ is under-estimated,
so the apparent A_p −4.5σ / n_s +1.7σ are **low-ESS artifacts, not bias**. This run is a path + cost check
only; the bias/coverage verdict requires the full run below.

**MF-SMOKE-01 verdict:** ✅ path healthy (0 divergences), ✅ floor + C_total sane at truth (χ²/dof≈1), ✅
affordable (~245 CPU-h @ N=99). Recovery is not interpretable at n=1/ESS≈20.

---

## 2. Planned: Run MF-COVERAGE-01 — the actual verdict (NEEDS PI compute sign-off)

The certification run. **Gates:** coverage ≥ nominal (per-param rank-uniformity is diagnostic-only for
Leg-B) **and** |bias| < 0.2σ on **A_p AND n_s**, through the MF forward + floor, KS z_lo=2.4.
- **Scale:** N ≥ 99 (L_FLOOR=99 for the ECDF diagnostic; the budget memory wants N≥99 with production
  samples). N=99 ≈ 245 CPU-h; N=600 ≈ 1490 CPU-h. SLURM-only, sharded per-mock (`fold_in` seeding).
- **Pre-launch:** (i) the get_samples postprocess fix (§1.1); (ii) confirm SLURM account + N with the PI;
  (iii) bump n_samples so the thinned L ≥ 99 given ESS≈0.3/sample (so ~1300+ samples/param → profile the
  real per-mock wall at production mtd before the full fan-out).
- Compare the MF coverage to the LF closure (the LF Leg-B bias the z-cut already fixed) to show MF does not
  regress coverage while removing the high-k real-fit systematic (gate G3).

**Status:** awaiting PI sign-off on scale + account (per `cavestru0-compute-budget`: profile [done] + ask).

---

## Log
- **MF-SMOKE-01** (2026-06-09): profiling mock, fold 0. Healthy, affordable (~245 CPU-h @ N=99), floor sane
  (χ²/dof≈1). Not a verdict (n=1). Artifacts: `figures/analysis/05_likelihood/mf_closure_smoke.{txt,npz}` +
  the 3 figures above; `scripts/profile_mf_closure_smoke.py`.
