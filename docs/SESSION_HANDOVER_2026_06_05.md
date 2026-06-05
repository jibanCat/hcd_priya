# Session handover — 2026-06-05 (Phase-C T4: closure/SBC + inference)

Branch: **`phase2c-likelihood`** (off `main`; Phase-2 emulator merged in PR #10, `8a980c2`).
Read this + `docs/superpowers/plans/2026-06-04-phase-c-t4-closure-plan.md` first.

---

## 1. Where we are
Phase-C = turn the merged emulator into a **differentiable likelihood** (numpyro/NUTS) and certify it
via SBC closure before fitting real data. T4a (foundation) + T4b (C_emu validation) done; the cross-class
C_emu fix is mid-build; the real DESI/KS data is acquired; the blinding strategy is decided.

## 2. This session — by commit (on `phase2c-likelihood`)
- `30db793` diag(phase-c): 4-agent checkpoint review (all GO) + HCD-template/Rogers/DLA/ECDF diagnostics.
  Resolved the user's HCD-template questions: offsets are REAL (global-⟨F⟩, matches Rogers); k is ANGULAR
  community-wide (feed `kfkms` directly, no /2π); DLA boost matches Rogers ~5% at k_min (band-median misread);
  Rogers Eq-6 has a `+c(z)` plateau the repo omitted. ECDF/PIT explained + illustrated.
- `6f00de7` / `aff3677` docs(emulator): `hcd_analysis/emulator/README.md` (env/architecture/forward-model/
  usage/conventions) + reviewer-flagged cleanups (doc numbers, `class_ratio` Eq-6, closure-diag notes).
- `e94ec88` feat(closure): **T4a foundation** — `closure_ctx.py` (Ctx + pack/unpack), `sampler_numpyro.py`
  (numpyro model, α-ladder τ₀, softplus α_dla, factor-loglik), `meanflux_prior.py`, `closure_mocks.py`
  (Leg-A), `closure_sbc.py` (Leg-A SBC + `--smoke`). `potential==loglik+Σlogprior` ✓; CS + Lyα GO.
- `d25cb57` fix(closure): T4a CS follow-ups — ECDF **L_FLOOR=99 gate guard** (the band over-rejects at
  small L: type-I 0.20@L49, 0.05@L99), **divergent-mock re-run** (escalate target_accept), seeded tie-break,
  x64 assert.

## 3. IN FLIGHT (not yet committed)
- **Cross-class C_emu build** (background agent, started this session). T4b found the diagonal C_emu
  **under-sizes the held-out residual ~2.5×** (whitening var=2.46), source = **class-correlation (+1.18)**
  + fold-variance (×1.25), NOT k-correlation/binning (CS-verified). Fix = a **4×4 cross-class covariance**
  `ρ_cc'(k,z,τ₀)` pooled over all 8 folds → `emu_var = Σ coef_c coef_c' ρ_cc' P_c P_c'` (α-propagating;
  diagonal recovers today's form). Agent is editing `inference.py`/`likelihood.py`/`closure_ctx.py`/
  `closure_sbc.py` (+ `scripts/build_xclass_error_vector.py`, `tests/test_xclass_cemu.py`). **On return:
  CS-review + re-validate whitening (target var→~1) + commit.** Diagnostics already committed-adjacent:
  `scripts/diag_cemu_validation.py`, figures `cemu_whitening_qq.png` / `cemu_over_cdata_heatmap.png`.
- **Untracked this session**: `scripts/convert_desi_dr1_p1d.py`, `docs/superpowers/2026-06-05-desi-dr1-p1d-usage.md`,
  this handoff, the T4-plan §8 edit. (No DESI data committed — it lives outside the repo, see §5.)

## 4. NEXT STEPS (in order)
1. Land the cross-class C_emu (CS-review + whitening re-validate + commit).
2. **Data-binding layer** — wire the real `cosmic_cov`: bin the emulator `P_obs` + `C_emu` onto the DESI
   (12z×85k) and KS grids; use the full correlated covariance; add the **SiIII+SiII metal** McDonald term
   and the **resolution template** to the forward model (per the usage doc); Becker+2013 μ_z in `meanflux_prior`.
3. **Leg-B coverage gate** — held-out-sim mocks (ε~N(0,C_cosmic)), coverage + bias-vs-{interior-τ₀, θ-hull},
   split by HCD class × k-band (DLA-low-k panel is binding); calibrate `cemu_inflate`.
4. The degeneracy / τ₀-param / HCD-metal-HeII arms + the PRIYA overlay (plan §1–4).
5. **Phase-D real-data BLIND fit** — see §6.

## 5. Data locations (outside the repo, gitignored)
- **DESI DR1 P1D** (Karaçaylı 2025, Zenodo 16943723): `/home/mfho/data/desi_dr1_p1d/` — raw `data_points.tar`
  + the npz `desi_dr1_p1d.npz` (12z×85k angular, full 1020×1020 cov, via `scripts/convert_desi_dr1_p1d.py`,
  SYSTEM python + astropy). Central `plya` is the TRUE measurement ("blind"=analysis-blind only).
- **KODIAQ-SQUAD** (Karaçaylı 2021, HR high-k leg): `/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/`
  (`final-conservative-p1d/-covariance`, SiIV cov; loaded by `lyaemu/lyman_data.py::KSData`; drop first 4 k-bins).
- Usage recipe (metals/resolution/cov/cuts): `docs/superpowers/2026-06-05-desi-dr1-p1d-usage.md`.

## 6. BLINDING STRATEGY (decided 2026-06-05 — for the Phase-D real-data fit)
Statistician agent, field-grounded; mirrors DESI DR1 Lyα cosmology (arXiv:2601.21432). PRIYA had NO blinding
→ this is a genuine improvement. (Full rationale: memory `blinding-strategy.md`; also T4-plan §8.)

- **Parameter-blind ONLY (A_p, n_s) — NOT a data-vector cosmology shift.** A data cosmology-shift is a smooth
  (k,z) distortion, which the τ₀/thermal/HCD/metal/resolution nuisances are built to absorb → it partially
  self-unblinds AND biases nuisance recovery (Muir-2020 data-shift is safe for 3×2pt, not P1D). Use a hidden
  additive offset `θ_shown = θ_inferred + δ`, `δ_{Ap,ns} ~ U(−3σ_prior,+3σ_prior)` from a
  `SHA256(project+commit)` seed COMMITTED to `blind.lock` but never de-hashed (email the hash to a colleague).
  Leave ALL nuisances FREE + VISIBLE (watch for negative τ₀, runaway resolution, prior-rail pile-up).
- **Freeze first (does most of the work):** pre-register (`analysis.lock`) the k/z cuts, covariance choice,
  `cemu_inflate`, nuisance-model complexity, outlier/χ² rules, priors, decision tree — locked on the SBC
  closure + mocks BEFORE `plya` is ever fit.
- **Sequence:** Leg-A/Leg-B SBC *unblinded on mocks* (catches pipeline bugs blinding can't) → freeze + commit
  seed → blind real-data NUTS fit (view only `θ_shown`) → pre-registered unblinding checklist → **unblind
  ONCE, NO changes after** (the load-bearing rule across DES Y3 / DESI BAO / DESI DR1 Lyα).
- **Single-PI sealing**: commit the seed BEFORE the first real-data fit so post-hoc reconstruction is
  git-detectable; encrypt nothing (only the 2-number offset is sealed). Nothing to build during the closure.

## 7. Key conventions / gotchas
- Env (MANDATORY): `PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3`. x64 hard-asserted on `import hcd_analysis.emulator`.
- **k is ANGULAR** (`k=2π/λ_v`, s/km) everywhere — cache `kfkms`, DESI, KS, Rogers all agree; never /2π.
- Production model = `checkpoints/final_fold0` (matched to `error_vector.npz`); single fold for inference (LOSO
  spread is in the error vector — don't ensemble).
- ECDF SBC gate is only valid at L≳99 (`closure_sbc.L_FLOOR`); the `--smoke` is PATH-only.
- astropy is in SYSTEM python only, NOT emu-jax → convert FITS→npz once.

## 8. How to pick up
1. `cd /home/mfho/hcd_priya && git status` (expect `phase2c-likelihood`).
2. Check the cross-class C_emu agent result; CS-review + re-validate whitening + commit (§3).
3. `PYTHONNOUSERSITE=1 … pytest tests/test_closure_sbc.py tests/test_likelihood_driver.py -q` — should pass.
4. Then §4 step 2 (data-binding) → Leg-B gate.
