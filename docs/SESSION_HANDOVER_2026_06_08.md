# Session handover — 2026-06-08 (Phase-C: slope prior settled; a coherent n_s emulator bias is the open thread)

Branch: **`phase2c-likelihood`** (off `main`). Read this + the two companion docs first:
- **Walkthrough (the full debugging story, 4 inline figures):** `docs/superpowers/2026-06-08-walkthrough-emu-bias-debugging.md`
- **PI root-cause handoff (candidate causes + ordered reading list):** `docs/superpowers/2026-06-06-HANDOFF-emu-bias-rootcause.md`
- **Spec (settled-vs-open ledger):** `docs/superpowers/specs/2026-06-06-hcd-slope-prior-spec.md` (§7 checkpoint, §8 rerun, §8b the all-folds reversal, §9 PI calls)

Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## ⭐ RESUME HERE

**One-paragraph state.** The HCD-incidence **slope-prior question is SETTLED** (marginalizing the
z-slope costs ×1.04 on σ(A_p); its own bias <0.17σ; the per-leg amplitude anchor is the absorber;
golden-guarded). While settling it, a **coherent emulator-sourced bias** surfaced and was chased through
four reframings (full story in the walkthrough). The honest, all-folds result: **A_p EMU bias is
consistent with ZERO (+0.03σ, 95% CI [−0.22,+0.29]); the real coherent bias is −0.65σ on n_s**
(t=−5.84, 50/60 held-out sims negative, NOT a box-edge artifact). The earlier "+0.62σ A_p" was a
low-n_s box-edge fluctuation of the fold-0-only sample. **The open thread is the n_s bias** — root cause
not yet pinned, no fix started; it is waiting on PI root-cause review (the user asked for thinking time).

**The single most important fact for the next session:** do NOT reintroduce "the emulator has a +0.6σ
A_p bias." That is superseded. The real object is **−0.65σ on n_s** (`scripts/diag_emu_bias_allfolds.py`,
`figures/analysis/04_emulator/emu_bias_allfolds.{png,txt}`).

**Leading root-cause hypothesis (for the PI to confirm/refute):** a **finite-60-sim LOSO generalization
floor on the n_s tilt** — the bias is fold-structured (worst fold-5 mid-n_s −1.56σ) with a mild
shrink-to-centre (corr(n_s-bias, n_s_unit)=−0.35), the signature of the emulator regressing a held-out
n_s slice's tilt toward the training mean. The global SVD output basis's high-k→tilt coupling is the
plausible *channel* (basis is non-orthonormal ‖BBᵀ−I‖≈2.7). This is likely a floor to MARGINALIZE
(correlated C_emu), not a bug to fix — but that is the PI's call.

**WHEN THE PI RETURNS WITH A DIRECTION, the options (none started):**
1. **(a) correlated-in-k/z C_emu** — the planned path (plan §0b-2). PREREQUISITES the checkpoint lenses
   insist on, both cheap + forward-only, do them FIRST: (i) measure ρ = alignment of the bias mode with
   the n_s (and A_p) response → the σ inflation 1/√(1−ρ²), gate at ~1.15×; (ii) try a z<2.3 (or n_s-range)
   cut — may collapse a big fraction of the bias and moot the rebuild. Then build the C_emu
   **z-resolved/edge-weighted** (the bias is z-LOCALIZED at z≈2.0–2.3 + z≈3.5–3.7, NOT a global rank-1
   mode — see walkthrough §5).
2. **(b) targeted retrain** (tilt-aware loss; fix basis orthonormality) — removes it at source but
   invalidates every cert and fights a finite-sim floor; lenses judged premature.
3. **(c) MF-first** (Lyα dissent) — the bias is for an LF-only forward production won't run; wire MF
   (plan `2026-06-05-mf-likelihood-wiring-plan.md`), re-run `diag_emu_bias_allfolds.py` post-MF, then revisit.

**Decoupled forward progress available now (NOT gated by the n_s question):** the **factorized-α
refactor** can proceed — golden guard is in place (`tests/test_legb_golden.py`, rtol 1e-12). The spec §7
has the locked design (factorize α(z) into a_pivot·exp(δ_leg)·g_fixed·(slope); g_fixed = lit-derived
`w_c_from_mu(dN/dX_lit·X̄(z))`, NEVER dN/dX alone; slope center 0; per-leg AMPLITUDE anchor only). CS
flagged it's a ~40–60 line contract change to `_data_loglik_legcore` (move α-assembly into the per-leg
loop), non-centered log-space reparam to avoid the amplitude×slope funnel. Run the golden test after.

**LOOSE END to resolve before locking the slope-SAFE verdict:** σ_anchor (KS 0.27, DESI 0.12) vs
σ_α (0.15) — are the units matched (log-offset vs fractional)? An anchor wider than the global incidence
prior could itself erode A_p identifiability. Carried over from checkpoint-1, still open.

---

## Conventions in force (do not drop)
- **Per-checkpoint 4-agent Workflow review** (Bayesian/PPL + CS + cosmology + Lyα → meta-reviewer report
  → surface to PI). Code with Bayesian+CS, consult Lyα on science Qs. Emit diagnostics figures along the
  way. (Memory `feedback-checkpoint-review-convention`.) This convention CAUGHT the A_p→n_s mis-attribution.
- **Forward-only Fisher is a scout, not the arbiter** — every bias number here is a MAP-shift surrogate;
  the eventual judge is the NUTS Leg-B coverage run (coverage≥nominal + |bias|<0.2σ on A_p AND n_s).
- **Cobaya/GetDist output** for any MCMC chains (memory `feedback-cobaya-chain-output`).
- **DESI real-data cosmology stays LOCAL** (gitignored); mocks/closure/KS/code committable
  (memory `desi-results-privacy`). Everything this session is mock/closure → committed + pushed.

## This session's commits (on `phase2c-likelihood`, pushed to origin)
- `23744a2` HCD slope-prior tradeoff + the (then-A_p) low-k bias investigation + golden guard + the
  5 onboarding + 5 checkpoint-1 referee reports + the spec.
- `6e91112` checkpoint-2 corrections (z-localized not z-coherent; n=8 significance; framing fixes) +
  the 5 checkpoint-2 referee reports.
- `00f0b34` all-folds re-measurement (the A_p→n_s reversal) + the PI root-cause handoff.
- (+ this handover + the walkthrough.)

## Artifacts map (regenerate any figure in minutes)
- `scripts/diag_legb_slope_prior_tradeoff.py` — first sweep (×1.04 variance; A_p framing superseded).
- `scripts/diag_legb_slope_prior_rerun.py` — corrected rerun (FULL vs EMU baseline; slope <0.17σ).
- `scripts/diag_emu_lowk_investigation.py` — per-k-band recon/rank-floor/leakage (the *channel* finding).
- `scripts/diag_emu_bias_allfolds.py` — **the authoritative bias number** (all folds, bootstrap, n_s −0.65σ).
- `scripts/make_legb_golden.py` + `tests/test_legb_golden.py` + `tests/golden/legb_lf_golden.npz` — the guard.
- Figures: `figures/analysis/05_likelihood/legb_slope_prior_{tradeoff,rerun}.png`,
  `figures/analysis/04_emulator/emu_{lowk_investigation,bias_allfolds}.png`.
- Memory pointers: `phase2c-likelihood-progress` (live state), `phase2-hr-phase1-and-bins` (the finite-60-sim
  floor + the 0.067σ origin), `phase2c-hcd-redesign`, `feedback-checkpoint-review-convention`.
