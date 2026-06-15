# Session handover — 2026-06-15 (real-fit validation arc: blind fits running, two gates RED)

Branch: **`phase2c-likelihood`** (off `main`; merge-base `8a980c2` = Phase-2 emulator PR #10).
HEAD: **`f71c86a`** — all pushed to `origin` (0 ahead of origin, 110 ahead of `main`).
Written: **2026-06-15 ~05:18 UTC** (≈01:18 EDT). A scheduled refresh is set for **3:30 AM EDT (07:30 UTC)** to
capture the then-current job results — this version is the safety-net snapshot.

Env (MANDATORY; x64 hard-asserted on import):
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`
Private notes repo (read+write): `/home/mfho/hcd_priya_notes` — start at `docs/superpowers/INDEX.md`.
Prior handover: `docs/SESSION_HANDOVER_2026_06_14.md` (architecture-validated + N=5 ensemble trained).

---

## ⭐ RESUME HERE — one paragraph

The real-data **blind fits are running** on cavestru1 (eBOSS DONE + clean health; KS + DESI still going). But
**two validation gates went RED this session and both block the real-fit cosmology headline:** (1) the
**MF n_s high-k cert FAILED** (worst n_s bias_z **+3.26σ** on DESI+KS, A_p also FAIL) — the production
multi-fidelity correction does not control the high-k n_s tilt, so the **KS fit (and a re-assessed DESI)
cannot be believed on n_s until a θ-resolved res_corr or a wider MF σ is in place**; and (2) the new
**data-nuisance bias gate** (built + reviewed this session) needs a blocker bug fix + a power redesign
before it can certify the data nuisances leave cosmology unbiased. The **production-ensemble SBC** (rank
uniformity) is still mid-flight (core 20/24 shards). **Blinding/privacy is intact**: I have NOT read any
unblind cosmology; only blind-safe `.health.json` diagnostics. The PI runs the unblind notebook
(`hcd_priya_notes/notebooks/2026-06-15-unblind_eboss_ks.ipynb`) on their end.

---

## RED gates (these block trusting the real fit)

### 1. MF n_s high-k cert — FAIL  ⛔ (blocks KS n_s; reassess DESI)
`scripts/analyze_mf_nscert_dk.py` (array 51793562, done). Genuine HF-LOSO Test B on DESI+KS:
```
 HR n_s | DESI+KS n_s bias_z | DESI-only n_s
  0.859 |   -0.37            |  -0.64
  0.885 |   +3.26  OUT       |  +1.14  OUT
  0.909 |   +0.07            |  -0.62
  0.972 |   +1.83  OUT       |  +2.80  OUT
  0.979 |   +1.32  OUT       |  +1.08  OUT
 worst |n_s bias_z| = 3.26σ; VERDICT n_s FAIL, A_p FAIL  (gate |bias_z|<1)
```
Figure: `hcd_priya_notes/figures/analysis/05_truth_validation/mf_nscert_dk_closure.png`.
**Meaning:** the diagonal C_emu floor cannot whiten the coherent high-k n_s tilt the LF→HR ρ-correction
leaves (consistent with the scoping note: HF-LOSO +2.8σ at ns0.972). **eBOSS is low-k (k_max 0.0195) so it's
least exposed; KS is the high-k leg and is most exposed; DESI (k_max 0.041) is intermediate.**
**Fix options (PI decision needed):** (a) θ-resolved res_corr in `multifidelity.py`; (b) widen the MF shape σ
(the conservative floor) until the tilt is covered; (c) cap k for KS. This is the headline blocker for KS.

### 2. Data-nuisance bias gate — BUILT, REVIEWED, needs fixes before submit  ⛔
Goal (PI ask): mock-verify that wiring the additional DATA nuisances (metals, resolution, per-survey LLS
pin) leaves A_p/n_s **unbiased** on the production config (N=5 ensemble + per-survey pin). **This had never
been done** — the running production SBC uses `survey=None` (cosmic-avg, not the per-survey pin) and is
`leg_a=True` self-draw, which by construction can't catch a misspecification bias.
PI directive (2026-06-15 04:04): **all four arms as a gate, run in parallel via SLURM.**

Built TDD this session (8 tests green). Files:
- `hcd_analysis/emulator/closure_legb.py` — NEW `apply_lls_truth_boost`, `_meanflux_on_leg`; `make_leg_a_legmock(..., inject_metal_misspec=, inject_resolution=)`; `inject_spec=` threaded through `run_legb` leg_a branch (default None = byte-identical no-op, verified).
- `scripts/run_dnuis_bias_shard.py` (arms: metal_misspec, resolution, lls_excess, metal_matched; per-survey), `scripts/analyze_dnuis_bias.py` (bias_z gate), `scripts/batch_dnuis_bias.sh` (cavestru1 array, NOT submitted), `tests/test_dnuis_inject.py`.
- **UNCOMMITTED** (in working tree).

Two-lens review verdict — **architecture sound, clean-null logic correct, but MUST FIX before it's a believable gate:**
- **BLOCKER (CS/JAX):** `_meanflux_on_leg` double-applies Kim — `tau0_global` is ALREADY τ_eff=α·Kim, but the code does `α·Kim` again → ⟨F⟩=exp(−α·Kim²), so the injected metal amplitude `A_X=f_X/(1−⟨F⟩)` is up to **~6× too large at low z**. Fix: `tau_eff = truth_pack["tau0_global"][sel]` directly (drop `*kimz`); fix docstring; tighten the test to pin ⟨F⟩=exp(−tau0_global[sel]). Corrupts the metal_misspec arm only.
- **POWER (Bayesian):** 16 mocks → SE≈0.22 on mean bias_z, which **cannot certify <0.3σ**. Recommended (high-leverage): a **PAIRED clean-vs-injected estimator** (same `fold_in(seed,m)`; shared noise cancels → tiny SD on Δbias_z, the way the eBOSS no-leak cert was done). Also gate on `|mean|+2·SE < 0.30` (not `|mean|`), and bump n. The clean set can be shared across arms per survey (amortize).
- metal_misspec: **KS is `metals_on=False` → silent no-op** (a PASS that tests nothing) — guard/skip KS×metal_misspec. Add `form="eboss"` for the eBOSS arm.
- resolution: justify `b_res` vs the *residual* (run a {0.005,0.01,0.02} ladder; `b_res=0.02` ≈ the FULL `syst_e_resolution`, pessimistic). KS `R_z` is a placeholder (DESI proxy) — flag, since KS high-k is where it bites.
- lls_excess: KS boost (2.5) == the KS pin center → near-null; add a KS arm offset OFF the pin.

**COST REALITY (must clear with PI before launch):** one NUTS fit ≈ **27 CPU-h** (eBOSS blind fit = 96296 CPU-s / 8 cpu = 26.7 CPU-h, 3:20 wall). Paired sweep over ~6–8 arm×survey cells at N≈12 → **~2000–3900 CPU-h**, a large fraction of the **5000 CPU-h cavestru1 cap** (memory `cavestru1-compute-budget`). Options to shrink: paired estimator (fewer mocks), drop n_samples 600→~300 (eBOSS ESS_bulk was 540 — ample margin), trim cells. **Profile + present the budget to the PI before sbatch.**

---

## In-flight cavestru1 jobs (snapshot ~05:18 UTC)

| Job | id | status |
|---|---|---|
| Production SBC pilot (core, N=24) | 51785809 | ~20/24 shards in `/scratch/cavestru_root/cavestru1/mfho/prod_sbc_core/`; merge with `scripts/merge_prod_sbc_shards.py` → rank uniformity |
| Production SBC (full, N=12) | 51786439 | ~3/12 shards in `prod_sbc_full/` |
| Blind real fits (eBOSS/KS/DESI) | 51786430 | **eBOSS DONE** (health clean, below); KS (_1) + DESI (_2) running ~3:41 |
| KS z2.8 diagnostic | 51793556 | running |
| MF n_s high-k cert | 51793562 | **DONE → FAIL** (gate #1 above) |

**eBOSS blind-fit health (blind-safe — no cosmology):** R̂max **1.013**, ESS_bulk_min **540**, ESS_tail_min
**192**, **0 divergences** (all chains), E-BFMI 0.83, treedepth-sat 0.0, n_params 26, 455 rows. Sampler is healthy.
Output `results/real_fit/real_eboss.*` (chains BLINDED on A_p/n_s by default).

**Background monitor `bt6fs39im`** (persistent) emits a line as each tracked job finishes (SBC merge / MF-cert /
blind-fit health). MF-cert event already fired. It will fire for the SBC arrays and the KS/DESI blind fits.

---

## What was COMPLETED + committed this session

- **`ef40225`** Demo C reworked → the per-fold A_p/n_s Fisher-bias plot (`B5_fisher_bias_perfold.png`), 0/8-fail at a glance; per-sim cloud demoted.
- **`f71c86a`** (and prior) Emulator README §9 **Blinding section** (parameter-blind A_p/n_s, seed-only blind.lock, SBC→freeze→blind→unblind-once protocol, artifact layout, unblind recipe pointing at the PI's private notebook). Reviewed PASS (faithful + PI voice + privacy correct). Module map → §10.
- DESI metals/resolution **audit** (answering the PI's defer question): **both wired** — `a_SiIII~U[0,0.15]` enters via `_metal_factor` (DESI-paper SiIII oscillation, gated on `leg.metals_on`); resolution handled by QMLE deconvolution (E_RESOLUTION lives in C_data, no window to apply). **Do not defer DESI on those grounds.** One nuance: we keep resolution in covariance vs the DESI-paper f_res marginalization (both valid).

---

## NEXT ACTIONS (in priority order)

1. **PI decisions needed:**
   - MF gate #1: which fix (θ-resolved res_corr / wider σ / KS k-cap)? Blocks KS n_s.
   - dnuis budget: approve the ~2–4k CPU-h sweep (or a trimmed/paired version)?
2. **dnuis fixes before submit:** apply the `_meanflux_on_leg` Kim fix; switch to the paired clean-vs-injected estimator + `|mean|+2SE` gate; guard KS×metal_misspec; add eBOSS metal form; resolution b_res ladder. Then profile 1 mock at production settings, estimate, submit the 4 arms in parallel. (Continue impl agent `ad31ccab863badca5` via SendMessage, or do inline.)
3. **SBC merge** when 51785809/51786439 finish → `merge_prod_sbc_shards.py` → rank-uniformity verdict (the pending inference-calibration gate). Note the pilot ran mf=False/eboss=False/survey=None — it certifies sampler calibration of the ensemble, not the per-survey/data-nuisance config.
4. **Blind-fit health** for KS + DESI when they finish (read `.health.json` ONLY; report R̂/ESS/div; **do NOT unblind**).
5. Commit the dnuis build once fixed + green.

---

## INVARIANTS / guardrails (do not violate)

- **Blinding/privacy:** never read unblind cosmology; never fold unblind results into memory. PI runs the unblind notebook (private notes repo) on their end. `.health.json` is blind-safe; chains are blinded on A_p/n_s by default. DESI real results stay private (`results_local/desi_production/`, gitignored); eBOSS/KS chains + all code + mocks/closures are committable.
- **Compute:** cavestru1 capped 5000 CPU-h; profile before big NUTS sweeps; ~27 CPU-h per fit/mock.
- **Build convention:** caveats → design pair (Bayesian+CS) → TDD → step-review pair → close with a validation gate; JAX/Equinox gets an independent JAX review.
- **Baseline (locked, `analysis.lock`):** 2-param τ₀, lit-pinned HCD `hierarchical=false`, emucoh + cross-class C_emu, MF P1D+dN/dX (off-diag cov), eBOSS+DESI metals, per-survey LLS pin (DESI 1.0×/σ0.30, KS 2.5×/σ0.40, eBOSS cosmic 1.0×/σ0.15). DESI and eBOSS fit SEPARATELY (not joint). KS z_lo=2.4 baseline / 2.8 diagnostic. MF high-k cert ON.
