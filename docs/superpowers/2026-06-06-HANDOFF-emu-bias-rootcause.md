# HANDOFF — the emulator coherent-bias investigation (for PI root-cause review)

**Date:** 2026-06-06. **Branch:** `phase2c-likelihood` (pushed; commits `23744a2`, `6e91112`, +this).
**Purpose:** you asked for time to think about the root cause of a coherent emulator bias. This is the
self-contained map: what's settled, what the honest number now is, the candidate root causes, and the
exact figures/docs/scripts to read — ordered.

---

## 1. One-paragraph state

The HCD-incidence **slope-prior question is SETTLED and safe** (marginalizing it costs ×1.04 on
σ(A_p); its own bias <0.17σ; golden-guarded; the per-leg amplitude anchor is the absorber). While
measuring that, a **coherent emulator-sourced bias** surfaced. After three rounds of correcting my own
diagnostics and finally re-measuring honestly over all 8 LOSO folds, the real signal is:

> **The emulator coherently UNDER-predicts n_s** by **−0.65σ** (95% CI [−0.85, −0.43], t=−5.84, 50 of
> 60 held-out sims negative). The **A_p bias is consistent with ZERO** (+0.03σ, CI [−0.22,+0.29]).

This **reverses** my earlier headline (which, from the fold-0-only sample — all low-n_s box-edge — had
mis-attributed it as a +0.6σ **A_p** bias). The investigation was right that a coherent, diagonal-C_emu-
unwhitenable emulator bias exists; it is **on n_s, negative, and not A_p**. This is the open root-cause
question. **No decision is needed from you yet** — this is for you to think about; I have NOT started a
C_emu build or retrain.

---

## 2. What is SETTLED (do not re-litigate)

1. **Slope-prior width** = literature dN/dX-slope 1σ (0.52/0.53/0.33 LLS/subDLA/DLA) + z-edge inflation
   ×1.25; slope center 0, fixed, no hyper-prior; per-leg amplitude anchor (KS 0.27, DESI 0.12),
   amplitude-only (not slope). Marginalizing costs ×1.04 σ(A_p); own-bias <0.17σ on all sims. Verified
   4/4 lenses (`checkpoint-meta` + `checkpoint2-meta`). **One loose end:** the σ_anchor(0.27) > σ_α(0.15)
   unit question (could the anchor erode A_p identifiability?) — carried over, not yet resolved.
2. **Golden guard** (`tests/test_legb_golden.py` + `tests/golden/legb_lf_golden.npz`): the legacy
   (3,)-α forward is pinned bit-for-bit (rtol 1e-12, verified sensitive). The factorized-α refactor is
   gated on this and may proceed independently of the bias question.
3. **The factorized α(z) model** (`α_c(z)=a_pivot·exp(δ_leg)·g_fixed,c(z)·((1+z)/(1+z_p))^{s_c}`, with
   g_fixed = lit-derived `w_c_from_mu(dN/dX_lit·X̄(z))`, NEVER dN/dX alone) is the agreed production form.

## 3. The OPEN question — the coherent n_s under-prediction (−0.65σ)

### 3.1 The honest number + its structure (a root-cause lead)
All-folds, each fold's own held-out emulator, KS<0.06, 60 sims spanning n_s_unit 0.013–0.958
(`emu_bias_allfolds.{png,txt}`). The n_s bias is **fold-structured**, not flat:

| fold (n_s_unit span) | A_p bias | **n_s bias** |
|---|---|---|
| 0 (0.01–0.12) | +0.59 | −0.47 |
| 1 (0.12–0.22) | +1.18 | **+0.47** |
| 2 (0.24–0.34) | −0.25 | −0.52 |
| 3 (0.35–0.44) | −0.21 | −0.84 |
| 4 (0.44–0.51) | +0.02 | −1.04 |
| 5 (0.53–0.61) | −1.77 | **−1.56** |
| 6 (0.62–0.71) | +0.39 | −0.75 |
| 7 (0.71–0.96) | +0.13 | −0.63 |

- **corr(n_s bias, n_s_unit) = −0.35** — mild **shrink-to-centre** (high-n_s sims pulled down more;
  n_s_unit<0.3 → −0.19, n_s_unit>0.6 → −0.76).
- **Fold 5 (mid-n_s) is the worst** on BOTH params — the only fold with a large A_p bias too.

### 3.2 Candidate root causes (for you to weigh — I have NOT picked one)
1. **Finite-training-set LOSO generalization floor on the n_s tilt.** The fold-structure + shrink-to-
   centre is the textbook signature: holding out an n_s slice, the emulator interpolates the tilt from
   neighbours and systematically regresses it toward the training mean. This is a **data/coverage**
   limit (60 LF sims), not a bug — it would only be cured by more sims or a tilt-aware loss/prior, and
   it is exactly the kind of thing C_emu is meant to MARGINALIZE, not fix. The Phase-2 design memo
   already flagged a "finite-60-sim generalization floor at VAL 0.16–0.21 σ_cosmo" — this may be that
   floor, now seen as a *coherent* (not scatter) n_s pull. **My leading hypothesis.**
2. **High-k → tilt leakage via the global SVD basis** (your Q1). The earlier per-band work showed the
   coherent error is high-k-sourced and the global basis couples bands (r=−0.61, n=8 — suggestive,
   not significant). A mis-fit high-k shape IS a tilt error (n_s is the k-slope), so this and (1) are
   not exclusive: the basis is the *channel*, finite-sims is the *cause*. The trained basis is far from
   orthonormal (||BBᵀ−I||≈2.7) — a concrete lead, though cosmology-lens judged it weak (cond 12.8).
3. **Fold-5 specificity** — is there something about the mid-n_s sims (a bad sim, a τ₀-ladder gap, an
   HeII-reion cell) driving the worst fold? Worth a targeted look before generalizing.
4. **NOT the ruler, NOT box-edge, NOT truth-noise:** the covariance choice explains only 1.26×;
   edge-proximity corr is −0.04; high-k truth P_filt is the *smoothest* band (so it's genuine emulator
   error, not noise leaking from the cache truth). These are ruled out.

### 3.3 The eventual options (NOT to decide now)
- (a) **Correlated-in-k/z C_emu** to marginalize the coherent mode — the planned path (§0b-2);
  3-of-4 lenses favor it over a retrain. **Precondition the lenses insist on:** measure ρ (alignment of
  the bias mode with the n_s/A_p response) → the σ inflation 1/√(1−ρ²); gate at ~1.15×. Unmeasured.
- (b) **Targeted retrain** (tilt-aware loss, or fix basis orthonormality) — removes it at source but
  invalidates every cert and fights a finite-sim floor; lenses judged premature.
- (c) **MF-wire first** (Lyα dissent) — the bias is for a forward production won't run; MF reshapes the
  high-k band. Re-measure post-MF before designing C_emu.
- (d) **A cheap untested lever:** a z<2.3 (or n_s-range) cut — does it collapse the n_s bias?

---

## 4. READING LIST (ordered — for the root-cause investigation)

**Read first (the story + the honest numbers):**
1. `docs/superpowers/2026-06-06-slope-prior-and-emu-lowk-investigation.md` — the main narrative (has the
   ⚠️ checkpoint-2 corrections banner at the top; read that banner carefully — several §2–§3 framings
   are corrected). 3 inline figures.
2. **THIS doc** — the handoff + the all-folds reversal (n_s −0.65σ, A_p ~0).
3. `docs/superpowers/specs/2026-06-06-hcd-slope-prior-spec.md` — §7 (checkpoint resolution), §8 (rerun),
   **§8b (the all-folds reversal)**, §9 (PI calls). The settled-vs-open ledger.

**The figures to study (all regenerate from their scripts in minutes):**
4. `figures/analysis/04_emulator/emu_bias_allfolds.png` — **the key figure**: A_p & n_s EMU bias vs n_s
   over all 60 honestly-held-out sims, with the bootstrap mean. Shows A_p~0, n_s coherent-negative.
5. `figures/analysis/04_emulator/emu_lowk_investigation.png` — the 3-panel (rank-floor / band-attribution
   / leakage); **read with the caveat** that its A_p framing is the fold-0 (mis-attributed) sample —
   the *mechanism* panels (rank floor, leakage) are still informative, the A_p magnitude is superseded.
6. `figures/analysis/04_emulator/loso_error_vs_k.png` + `figures/analysis/05_likelihood/grad_fidelity.png`
   — the EXISTING per-k / gradient diagnostics (full-k pooled; the blind spot was per-z-band coherence).
7. `figures/analysis/05_likelihood/legb_slope_prior_rerun.png` — the slope-prior gating (FULL vs EMU).

**The text outputs (the raw per-sim numbers):**
8. `figures/analysis/04_emulator/emu_bias_allfolds.txt` — per-fold, per-sim A_p/n_s bias + the pooled
   bootstrap + the box-edge confound check.
9. `figures/analysis/04_emulator/emu_lowk_investigation.txt` — the per-k-band recon-vs-floor table.

**The deep-dive referee reports (if you want the adversarial detail):**
10. `docs/superpowers/onboarding/2026-06-06-checkpoint2-meta.md` — the synthesis that broke my overclaims
    (z-localized not z-coherent; n=8 significance; the fork is not ready). Plus the four
    `checkpoint2-{bayesian,cs,cosmology,lya}.md` (esp. **cosmology** for the per-z localization and the
    σ(A_p)-inflation precondition, and **lya** for the in-scope-band + MF-gating argument).
11. `docs/superpowers/onboarding/2026-06-06-onboarding-{cosmology,lya,meta}.md` — the original deep reads
    (normalization redesign, θ-blind baseline identifiability, PRIYA convergence, the finite-60-sim floor).

**The code (to re-run / extend):**
12. `scripts/diag_emu_bias_allfolds.py` — the all-folds measurement (the authoritative bias number).
13. `scripts/diag_emu_lowk_investigation.py` — per-k-band recon, rank-floor, band-attribution, leakage.
14. `scripts/diag_ap_fisher_bias.py` — the ORIGINAL Phase-2 0.067σ (single-z=3, diagonal cov) — compare
    its metric to the all-folds one to see the single-z-vs-full-z difference directly.
15. `scripts/run_loso_sweep.py` — the LOSO sweep + per-k error vector (the finite-sim floor lives here).
16. `hcd_analysis/emulator/model.py` (HeadB + the global SVD `p_filt_basis`, the coupling channel) +
    `predict.py::reconstruct_P_filt_jax` (the θ-blind-baseline + σ_cosmo·residual reconstruction).

**Background memory (already in your context):** `phase2c-likelihood-progress`,
`phase2-hr-phase1-and-bins` (the finite-60-sim floor + the 0.067σ), `phase2-tau0-cosmology-interaction`.

---

## 5. Where to resume (when you've decided a direction)

- If **(a) C_emu**: first run the ρ / σ-inflation measurement (forward-only, cheap) + the z<2.3 cut test;
  THEN design the C_emu z-resolved/edge-weighted (NOT global rank-1 — the bias is z-localized).
- If **(c) MF-first**: wire MF (plan `2026-06-05-mf-likelihood-wiring-plan.md`), re-run
  `diag_emu_bias_allfolds.py` post-MF, then revisit.
- Independently, the **slope-prior refactor** can proceed now (golden-guarded), if you want forward
  progress decoupled from the bias question.
- Resolve the **σ_anchor < σ_α** unit question before locking the slope-SAFE verdict.

Every number here is **forward-only Fisher (MAP-shift)** — the eventual arbiter is the NUTS Leg-B
coverage run, where the box-edge n_s and any A_p–anchor degeneracy are exactly where MAP ≠ posterior mean.
