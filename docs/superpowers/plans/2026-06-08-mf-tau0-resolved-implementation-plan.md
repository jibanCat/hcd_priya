# Multi-fidelity (LF→HR) τ₀+z-resolved resolution correction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans.
> Steps use checkbox (`- [ ]`). TDD where a behaviour is testable; forward-only diagnostic gates otherwise.

**Goal:** Wire the multi-fidelity emulator into the production likelihood with a resolution correction
**resolved in both z and τ₀** (the current `FixedMeanHead` pools over τ₀), so the real-data fit corrects the
LF's ~−6% low-z high-k power deficit (vs HR) — the confirmed real-fit n_s systematic — and re-measure the
n_s bias *through the MF forward*.

**Architecture:** `P_MF = (ρ(k)·f_LF(θ,z,τ₀) + δ)·res_corr(z,k)`; log form
`log P_MF = log P̂_LF + g(z,τ₀,k) + log res_corr` (`hcd_analysis/emulator/multifidelity.py:5,44`). Default
`delta_mode='none'` ⇒ `g = gbar(z,k)`, a **θ-independent** z-resolved table (`FixedMeanHead:309`,
`gbar_tab (Nz, n_classes, K)`, reads `cond[9]=z_unit`, **ignores `cond[10]=τ₀`**). We make `g` τ₀-resolved.

**Tech stack:** JAX/Equinox; LF backbone frozen via `_freeze`/`stop_gradient` (multifidelity.py:154). Env:
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## Design decisions LOCKED by the 4-referee review (2026-06-08, `onboarding/2026-06-08-review2-meta.md`)

1. **Correction FORM (resolves the only substantive dissent):** build `g` **separable + ONE rank-1
   interaction**:
   `g(z,τ₀,k) = gbar_z(z,k) + gbar_tau(τ₀,k) + a(k)·u_z(z)·u_τ(τ₀)`
   — NOT a free 2-D `(z,τ₀,k)` table. Rationale: the LF→HR deficit has a real **(z,τ₀) sign-flip**
   interaction (cosmology lens: ρ<1 at high-τ₀ for z≈5) that a pure separable form cannot represent, BUT a
   free 2-D table over only **6 clustered HR sims** would **alias the n_s tilt into the τ₀ axis** and
   recreate the retired `delta_mode='mlp'` over-fit (4σ n_s Fisher bias). A single rank-1 interaction term
   captures the sign-flip with too few DOF to alias the tilt. **Adoption gate (HARD):** the n_s
   Fisher-bias check must be run **THROUGH `MultiFidelity.logP_mf`** (the production forward incl. LF
   tail-extrapolation), not the raw cache, and come back in-gate before this form is adopted.
2. **Single clean ρ across HCD classes** (`scripts/diag_mf_rescorr_per_class.py`, 2026-06-08): the LF→HR
   correction is class-independent to ≤1.1pp (< DLA's 1.28% 6-sim noise); the excess templates R_c are
   correctly corrected by the clean ρ where they carry weight. Use **one clean-class ρ** for all 4 classes.
   (Re-check DLA only if HR sims are added.)
3. **Small-scale-leg C_emu floor:** size from the **worst-per-sim coherent LOSO residual (~0.9%)**, NOT the
   pooled signed mean (+0.01%, which cancels); measure it **z-resolved** (the deficit sign-flips with z),
   **through the production forward** (incl. tail extrapolation), reported **per-band** (k<0.069
   LF-resolvable vs k>0.07 extrapolated), with **×1.2–1.5 finite-sample inflation** (n=6). It covers
   **LF→HR generalization ONLY**; HR→truth non-convergence is covered by the **k<0.06 analysis cap**
   (do not claim MF "resolves to truth").
4. **n_s-edge extrapolation budget:** the 6 HR sims span only **ns∈[0.86,0.98]**; eBOSS lands at n_P≈1.009.
   Budget the **MF-correction extrapolation** uncertainty for ns outside [0.86,0.98] **separately** from the
   LF-emulator sparse-edge step-C_emu above 0.995 (two different objects). Flag in the blinding/real-fit
   report that any posterior mass above ns≈0.98 rests on MF extrapolation.
5. **gbar vs res_corr:** confirm `gbar` (LF→HR box-resolution ratio) and `res_corr` (L15 particle-convergence
   table) correct **different** physical effects and are not double-applied to the same convergence gap.

---

## Pre-MF CRITICAL items (cheap, do FIRST — from the review)

### Task C1: All-folds certification of the shipped KS z_lo
**Files:** reuse `scripts/diag_emu_bias_allfolds.py` (ks_kwargs); append to `figures/analysis/04_emulator/nsbias_kscut_scan.txt`.
- [ ] Run `diag_emu_bias_allfolds.py` over **all 60 sims** at the **shipped** `z_lo` (see C2 decision),
  forward-only. Record the n_s bias next to the committed `z_lo=2.6` all-folds row (+0.038σ). Confirm in-gate.
- [ ] Commit the number with the z_lo change.

### Task C2: Reconcile the z_lo default across code/docstring/plan/memory  ⚠️ PI DECISION FIRST
The review (cosmology+Lyα, meta) recommends defaulting to **z_lo=2.8** (the PI's published KODIAQ-SQUAD cut)
because the closure is **blind** to the DLA-finder incompleteness that degrades continuously toward low z
(so z=2.4/2.6 KS are also data-compromised), and the closure cannot distinguish 2.4 vs 2.8 (marginal
constraining power ≈0). The PI instructed 2.4. **Resolve with the PI**, then:
- [ ] Set `load_ks_leg` default to the chosen value; expose the other as opt-in.
- [ ] Make the docstring rationale, this plan, the diagnostics plan, and MEMORY all consistent (currently the
  docstring cites "excludes z<2.8" while defaulting to 2.4).
- [ ] Regenerate the golden + re-run the 29 tests for the chosen z_lo.

---

## MF implementation tasks

### Task T1: τ₀+z-resolved fixed-mean table (the correction itself)
**Files:** Modify `hcd_analysis/emulator/multifidelity.py` (`fixed_mean_table`, `FixedMeanHead`, `make_cond`);
Test `tests/test_multifidelity.py`.
- [ ] **Step 1 (test first):** add a test that `g(z,τ₀,k)` for the separable+rank-1 head reproduces the
  measured ρ on the 6 HR sims to within their 6-sim noise (per `diag_mf_rescorr_vs_tau0_z`), AND that
  `g` is **differentiable in τ₀** (`jax.grad` finite, nonzero) — the current `FixedMeanHead` has ∂g/∂τ₀≡0.
- [ ] **Step 2:** extend `fixed_mean_table` to bin the LF→HR log-ratio by **(z, alpha_idx)** (not pooled
  over alpha), then fit the **separable + rank-1** decomposition `gbar_z(z,k)+gbar_tau(τ₀,k)+a(k)u_z u_τ`
  (e.g. SVD-rank-1 of the residual-after-separable). Store the components.
- [ ] **Step 3:** rewrite `FixedMeanHead.__call__` to read `cond[10]=τ₀` and evaluate the separable+rank-1
  `g` with **clamped** z- and τ₀-interpolation (mirror `interp_res_corr`'s clamped bilinear, multifidelity.py:122).
- [ ] **Step 4:** run the test → pass. Confirm `_freeze`/stop_gradient on the LF backbone is untouched.

### Task T2: gbar vs res_corr non-double-counting check (design item 5)
- [ ] Read `load_res_corr`/`interp_res_corr` + the gbar construction; verify (and document in
  `multifidelity.py` + this plan) that gbar = box-resolution (1536³→3072³) and res_corr = particle-load
  (L15 384/512) correct **distinct** gaps. If they overlap, drop/rescale one. No code change if disjoint.

### Task T3: n_s-Fisher-bias THROUGH the MF forward (the HARD adoption gate)
**Files:** new `scripts/diag_emu_bias_allfolds_mf.py` (clone `diag_emu_bias_allfolds.py`, swap the forward to
`MultiFidelity.logP_mf`).
- [ ] Re-run the all-folds n_s/A_p Fisher MAP-shift with the **MF forward** (τ₀+z-resolved g + res_corr),
  KS at the shipped z_lo, **z-resolved** output. Gate: n_s bias in ±0.2σ AND the z=2.8–3.4 high-k residual
  reduced vs LF. If the separable+rank-1 g leaves a coherent n_s residual or shows aliasing, STOP and
  revisit the form (do not adopt).

### Task T4: z-resolved small-scale C_emu floor through the MF forward (design item 3)
**Files:** new `scripts/diag_mf_cemu_floor.py`; the C_emu assembly in `data_likelihood.py`.
- [ ] 6-fold LOSO of the correction **through `logP_mf`**, worst-per-sim coherent residual, **z-resolved**,
  per-band (k<0.069 vs >0.07), ×1.2–1.5 inflation. Add it as a small-scale-leg C_emu floor term.
- [ ] Add the n_s-edge MF-extrapolation inflation for ns outside [0.86,0.98] (design item 4), separate from
  the existing step-C_emu>0.995.

### Task T5: Wire MF into `data_loglik` (the production likelihood)
**Files:** `hcd_analysis/emulator/data_likelihood.py` (`predict_P_obs_on_leg`, `data_loglik`); tests.
- [ ] Add an `mf=` path so `data_loglik` calls the MF forward (LF stays the reference/opt-in). Carry the
  dropped PRIYA δ(θ) in C_emu and show it is incoherent-in-cosmology (per the onboarding R4).
- [ ] Re-run the closure (Leg-B) through the MF forward; confirm coverage + |bias|<0.2σ on A_p AND n_s.
- [ ] Golden + 29 tests pass (regenerate golden for the MF path with explicit reason).

---

### Task T6: dN/dX + CDDF resolution correction (Head A — PI ask, currently absent)
MF is **P1D-only**; Head A (CDDF `f_nhi`, per-class incidence `dN/dX`) has no resolution correction, yet
HR≠LF dN/dX. PI directive: **MF the dN/dX/CDDF via a FIXED-SLOPE resolution correction, KEEP the existing
HCD error estimates** (the incidence-prior widths are unchanged).
**Files:** caches `observables_tau0_{lf,hr}.h5` datasets `snap_dNdX_{LLS,subDLA,DLA}` + `snap_f_nhi` (30
N_HI bins); `hcd_analysis/emulator/multifidelity.py` (Head-A correction); `hcd_analysis/emulator/inference.py`
(the incidence prior — DO NOT widen). NOTE: dN/dX rows index differently from P1D (τ₀-invariant → ~1072 LF /
103 HR rows = sims × z, no τ₀ ladder) — match on (params, z), not the P1D row map.
- [ ] **Step 1:** measure HR/LF for `dN/dX` (per class) and the CDDF `f_nhi` on the matched overlap sims,
  as a function of z and log N_HI. Characterise it as a **fixed (θ-independent) slope-in-logN_HI** (or a
  smooth z-resolved ratio) — confirm θ-independence over the 6 HR sims (as for ρ) and quantify the noise.
- [ ] **Step 2 (test first):** test that the Head-A correction is θ-independent (∂/∂θ≡0, like the P1D
  baseline), differentiable in z, and reproduces the HR dN/dX/CDDF from LF×correction to within 6-sim noise.
- [ ] **Step 3:** implement the fixed-slope correction on Head A outputs (multiply dN/dX / shift the CDDF
  slope by the fixed resolution factor); leave the incidence-prior widths (`inference.py`) UNCHANGED per the
  directive.
- [ ] **Step 4:** verify the HCD-template forward (which consumes dN/dX via `g_fixed = w_c_from_mu(dN/dX·X̄)`)
  is consistent after the correction, and that the α-incidence closure bias (<0.17σ, [[phase2c-hcd-redesign]])
  is not regressed. Run the golden + closure tests.

## Self-review
- Covers all 5 review design constraints (form, single-ρ, floor, ns-edge, double-count) + the 2 CRITICAL
  pre-items. The HARD gate (T3, n_s-Fisher-through-MF) precedes adoption. Compute: forward-only diagnostics
  + one closure re-run; no LF retrain. Profile before any NUTS at scale (budget ~4000 CPU-h).
- Open: the separable-vs-2D form is decided as **separable+rank-1**, arbitrated by T3 — if T3 fails, the
  fallback is documented (revisit the interaction rank, never a free 2-D table over 6 sims).
