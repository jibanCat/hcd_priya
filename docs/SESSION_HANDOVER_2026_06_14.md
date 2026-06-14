# Session handover — 2026-06-14 (Phase-C: production ensemble trained; merge gate + production-SBC pending)

Branch: **`phase2c-likelihood`** (off `main`; merge-base `8a980c2` = Phase-2 emulator PR #10).
HEAD: **`245b0c1`** "test/hygiene: merge-readiness fixes from the 4-lens Phase-C review".
State: **89 commits ahead of `main`, 13 ahead of `origin` (UNPUSHED)**. Full suite **457 passed, 6 skipped**
(verified this session; the `245b0c1` commit message recorded 459 before the joblib `importorskip` collapsed 2→skip).

> **Why this doc exists:** the previous session disconnected before a handoff was written for the
> 2026-06-08 → 2026-06-14 arc (phase4d/5a/5b/5c). This was reconstructed from the git history + the
> private notes repo by three parallel agents. The last hand-written handover was `SESSION_HANDOVER_2026_06_08.md`.

Env (MANDATORY; x64 hard-asserted on import):
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`
Private notes repo (read+write): `/home/mfho/hcd_priya_notes` — start at `docs/superpowers/INDEX.md`.

---

## ⭐ RESUME HERE

**One-paragraph state.** Phase-C (the differentiable numpyro/NUTS likelihood) is **architecture-validated** —
held-out-sim (Leg-B) closures recover cosmology in-gate (n_s 8/8 within 1σ on the DESI-only HCD closure;
eBOSS n_s ±0.5σ / A_p ±0.74σ), with **0 divergences across every closure run**; emucoh C_emu, the HCD↔n_s
systematic, and the SiIII metal nuisance are each separately certified. The **N=5 all-sims production
ensemble** (`checkpoints/final_prod_seed0..4`) is trained + row-validated. **Two things are pending:**
(1) a **merge-to-`main` decision** (the branch is merge-ready but has a small pre-merge cleanup, below — this
is PARKED awaiting your choice of integration method, per the "present before touching main" gate); and
(2) the **production-ensemble SBC** — the one inference-calibration gate still open before the blind real fit.

**The immediate decisions waiting on the PI:**
- **Merge method** — direct merge to `main` vs push+PR vs keep-branch-open (parked; see §Merge gate).
- **Production-SBC budget** — approve a ~530 CPU-h N≈24 pilot, then the full N=128 (~2400–3800 CPU-h) needs sign-off.
- Three freeze-time calls: **KS z_lo 2.4 vs 2.8**, subDLA add-back for KS, a_SiIII real-fit wiring (see Forward plan §5).

---

## Merge gate — pre-merge cleanup (do before touching `main`)

Tests are green, but the **working tree** (not the committed tree) needs a small cleanup; a blind `git add -A` is wrong.

| item | action |
|---|---|
| `hcd_analysis/emulator/closure_legb_figs.py` (untracked) | **COMMIT** — tracked `closure_legb.py:1777` imports it (lazy, under `--smoke --figures`), so the suite passes 457/457 while a fresh `main` checkout would `ImportError` on that path. The one correctness-relevant item. |
| `checkpoints/final_prod_seed{0..4}.hist.json` (untracked) | **COMMIT** — production ensemble histories; `.gitignore` keeps `*.hist.json` intentionally; they back `2e51707`. |
| `figures/analysis/04_emulator/emu_bias_allfolds_mf.txt` (modified → 0 bytes) | **RESTORE** (`git checkout --`) — a tracked deliverable (the T3 MF adoption-gate table) accidentally emptied; do NOT commit as-is. |
| `docs/SESSION_HANDOVER_2026_06_05.md` (modified) | **RESTORE** — stale mid-edit, superseded by the committed `_06_08`. |
| `checkpoints/stepA*/` log/pid/txt strays, ~56 diagnostic figures + .npz | **LEAVE / route to notes repo** — not merge-blocking; per convention working diagnostics go to `/home/mfho/hcd_priya_notes`. |
| ~45 untracked `scripts/*.py` | per-file decision; none block the merge. Commit the closure-shard pipeline (`run_legb_shard.py`, `merge_legb_shards.py`, `legb_chains_to_cobaya.py`, `batch_legb_pilot.sh`) if you want it preserved. |

After the first three rows + `git status` check, the branch is clean to integrate.

---

## Where we are (state ledger, 2026-06-14)

### Validated + adopted
- **Emulator recipe** (FINAL_RECIPE) passes the LOSO gate at **0/8 Fisher-bias failures** (A_p RMS 0.067σ / n_s 0.071σ).
- **Cache v3.3 + 20-rung τ₀ ladder** — frozen, normalization-corrected (Δ=P_c−P_clean); loader hard-asserts version.
- **Per-survey legs/k-cuts:** DESI [0.00125,0.041] (carries A_p/n_s); KS [0.0055,0.063] (high-k degeneracy-breaker);
  eBOSS [0.0011,0.0195] (low-k shakedown); k_min=1e-3; separate per-survey inference.
- **Cross-class ρ + emucoh C_emu, emucoh-ON pinned** — off-diagonal right-sizes pooled whitening 1.28→0.93;
  emucoh widens σ through nominal (std z_Ap 1.085→0.917), 4-lens GO_WITH_CHANGES at infl=1; hard production requirement.
- **Per-survey LLS pin DESI 1.0×/σ0.30, KS 2.5×/σ0.40**; **subDLA tight at center 1.00** (TruncatedNormal).
- **SiIII metals no-leak** — free a_SiIII recovers the 0.045 injection with ≤0.11σ n_s / ≤0.10σ A_p leak, 24/24 chains 0 div.
- **ρ-only MF (LF→HR), A_p-certified** — θ-independent ρ·res_corr; T3 adoption-gate passed (n_s +0.026σ).
- **Parameter-blind blinding** (A_p/n_s posterior-offset only + freeze-on-mocks).
- **N=5 production ensemble** — in-range RMS clean 0.54 / LLS 0.59 / subDLA 0.79 / DLA 2.06 pct; seed scatter ~0.2%.

### Refuted / not adopted
- **#9 HCD-class coherent C_emu** — the α_subDLA −1 to −3σ miscoverage is a posterior-MEAN bias from the
  subDLA↔DLA degeneracy (a covariance term can only widen, never move a biased mean); coherence already covered
  by clean-class emucoh; saved ~400 CPU-h.
- **Option B / 2D HCD reparam** — validated byte-exact opt-ins but NOT adopted: they RELOCATE rather than reduce
  the HCD→n_s coupling (corr stays 0.79–0.93 across all parametrizations).
- **MLP MF δ-head** — retired at **4σ** n_s bias (over-fits the ~5% LF→HR departure on 6 clustered HR sims).

### Open / pending
- **Production-ensemble SBC** — the one inference-calibration gate, PENDING (rank-uniformity + ESS at L≥99,
  unblinded-on-mocks). [the gate]
- **MF n_s high-k cert** — genuine HF-LOSO gives +2.8σ n_s tilt at ns0.972 the shape-floor can only widen;
  needs a θ(n_s/τ₀)-resolved res_corr OR accept the wider σ; re-run Test B on DESI+KS. [DECISION + WORK]
- **KS z_lo 2.4 (live) vs 2.8 (published, referee-recommended).** [PI DECISION]
- **subDLA add-back for the KS low-k leg** (KS has 0 modes <k0.005). [WORK]
- **a_SiIII real-fit wiring** for DESI/eBOSS production. [WORK]
- **Blinding-seed commit** (after SBC reads clean, before first real fit). [WORK]
- **HCD prior-center lit-dN/dX→w_c mapping** (slope ~2.4 = w_c, not raw dN/dX ~1.4; use MF/HR-converged
  ratios) + budget the ~0.5σ n_s prior-center systematic. [WORK]

### Data + conventions
- **Data:** DESI DR1 P1D `/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz` (gitignored); KODIAQ-SQUAD
  `/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/`; eBOSS DR14 (Chabanier 2019).
- **k is angular** (k=2π/λ_v, s/km) on every leg — no /2π.
- **Results privacy:** real-DESI cosmology headlines stay LOCAL (`figures/private/desi_production/`,
  `results_local/`, gitignored); mocks/closure/blinding-infra/all-KS/all-code are committable.
- **Compute:** cavestru0 ~4000 CPU-h, cavestru1 5000 CPU-h — profile + ask before big NUTS runs.
- **4-lens checkpoint-review convention** (Bayesian/PPL + CS + cosmology + Lyα → meta → ask PI); close every build with a validation gate.

---

## This session arc (2026-06-08 → 2026-06-14)

The 06-08 handover closed with the open n_s-bias thread (a coherent −0.65σ n_s emulator-LOSO bias). The arc
became a sequence of build → 4-lens-review → closure-gate cycles, each leaving an adopt/refute decision.

- **Late phase-4c/4d — real-fit prior hardening + LLS-pin closure (Jun 10–11).** LLS prior CENTER (not width)
  carries ~1σ DESI A_p; per-survey LLS-pin closure validated; DESI σ_LLS 0.15→0.30; IGM box restriction;
  τ₀ → PRIYA 2-param (τ₀,dτ₀) mean-flux model replacing 13 per-z rungs; phase-4d SiIII metal groundwork.
- **emucoh — 60-sim k-coherent C_emu** (`2e6d1ae`,`53c1c8f`,`1ca4f7a`). Low-rank term from the 60-sim LOSO
  clean-class residual (within-band cross-k coherence 0.59–0.79). Matched-pair EC0/EC1 closure → GO_WITH_CHANGES,
  adopt at infl=1; it's an A_p term, NOT a de-biaser; **emucoh-ON now a hard production requirement.**
- **MF/HF shape-floor + genuine HF-LOSO** (`ce290b8`,`f2c6522`,`e50cbf3`). MF is **A_p-certified but NOT
  n_s-certified** (+2.8σ n_s tilt at ns0.972). Built a shape-aware low-rank eps-outer-product floor; caught +
  fixed a **θ-dependent-covariance pathology** (scaling C_shape by live P_model⊗P_model). Floor widens σ but is
  degenerate with n_s → can't de-bias; **left open before the KS real fit.**
- **eBOSS DR14 leg + SiIII metals** (`f0c6cce`,`0131ece`,`8800390`). Low-k shakedown recovers n_s ±0.5σ; SiIII
  certified end-to-end (≤0.11σ leak, 24/24 chains 0 div). PI correction: **MF not negligible at eBOSS k**
  (Δn_s≈+0.01) → real eBOSS fit must carry MF.
- **#9 HCD-class coherent C_emu — REFUTED** (`fc7efb9`,`ade6ba0`). Gate A passed (coherent) but Gate B fails:
  the α_subDLA miscoverage is a posterior-MEAN bias a covariance term can't fix; 4/4 panel; harmless to cosmology;
  saved ~400 CPU-h.
- **Hierarchical (Option B) + 2D amplitude×tilt HCD prior — TRIED, NOT ADOPTED** (`d457276`,`104e30b`,
  `309164d`,`5f51fc6`). Collapses the subDLA mean bias but RELOCATES the n_s coupling onto the A_HCD center
  (+0.51–0.63σ per ±1σ). `5f51fc6` also carried the z-slope-center fix (HCD_INCIDENCE_SLOPE ~2.4). Kept as opt-ins.
- **HCD↔n_s degeneracy — MEASURED** (`wf_4e8d25e2-909`). The LLS excess is **91% in-plane with the forest
  (A_p,n_s) on DESI** (cos(LLS,A_p)=+0.81) — real physics, irreducible by HCD reparam or a tighter prior
  (HCD-free inflates σ_n_s 3.2×); 0.5σ = prior_width × degeneracy; broken only by high-k KS + the damping-wing shape.
- **Production N=5 all-sims ensemble trained** (`2e51707`). FINAL_RECIPE reused byte-for-byte on all 60 LF sims;
  generalization inherited from the LOSO C_emu. **Architecture validated; production-ensemble SBC the one gate left.**
- **Merge-readiness hygiene** (`245b0c1`). Stale RED test fixed, 64MB regenerable blob un-cached, joblib
  `importorskip` guard; no inference/production code touched.

---

## Forward plan (ordered)

0. **Merge `phase2c-likelihood` → `main`** — do the §Merge-gate cleanup, then integrate. PI DECISION: integration
   method (parked). ~30–60 min, no compute.
1. **Production-SBC — the PENDING inference gate.**
   - 1a. Build the new code: an **ensemble-mean predict wrapper** (mean of P_filt over the 5 members — NOT
     weight-averaging) + a `mock_indices`/`return_per_mock` shard split for the Leg-A SBC driver. Reuse
     `closure_sbc.py` (Leg-A logic) + `scripts/{run_legb_shard,merge_legb_shards}.py` + `batch_legb_pilot.sh`.
     Gate = Leg-A rank-uniformity SBC on the N=5 ensemble + production covariance (emucoh-ON), unblinded on mocks;
     Leg-B-on-ensemble is a cheap confirmation arm. ~1–2 days TDD (+ independent JAX-specialist test).
   - 1b. **N≈24 pilot first** (~530 CPU-h) — measures the true ensemble forward overhead + confirms L_eff≥99
     (NUTS ESS≈0.18/sample ⇒ n_samples≥600). No PI sign-off needed but profile + report `sacct` before scaling.
   - 1c. **Full SBC, N=128** (~2400–3800 CPU-h depending on overhead). **PI DECISION: budget sign-off + partition**
     (this is the bulk of either allocation). Deliverable → 4-lens checkpoint review → PI.
2. **Freeze + commit the blind seed** — wire the 2-param posterior-offset infra; write `blind.lock` (SHA256,
   never de-hashed) + `analysis.lock` (k/z cuts, covariance incl. emucoh-ON, priors, decision tree). Prereq:
   SBC passed-and-unblinded-on-mocks. ~½ day. PI DECISION: sign off the frozen `analysis.lock`.
3. **MF wiring for KS/eBOSS + n_s high-k cert** — wire the frozen θ-independent ρ-only MF; certify n_s through MF
   (re-run Test B on DESI+KS, |z|<1). Can FOLLOW step 1 (MF is KS-only/θ-independent/gate-invariant). PI DECISION:
   θ-resolved res_corr vs accept the wider σ.
4. **HCD prior-center: map lit dN/dX → w_c** (slope ~2.4 = w_c; MF/HR-converged ratios) + budget the ~0.5σ n_s
   systematic. Must be in `analysis.lock` before any real fit. ~1 day, no NUTS. PI DECISION: confirm lit sources + pin centers.
5. **Resolve the open PI calls (fold into the freeze):** KS z_lo 2.4 vs 2.8; subDLA add-back for KS; a_SiIII
   real-fit wiring. All PI DECISIONS; small config edits.
6. **Blind real-data fit + unblinding checklist** — parameter-blind production NUTS (DESI → eBOSS → KS),
   nuisances free+visible; walk the pre-registered checklist; **unblind ONCE**. Output chains in cobaya/GetDist
   format. Real-DESI cosmology stays LOCAL. Prereq: steps 1–5 complete. PI DECISION: authorize the one-time unblind.

### Immediate next action (recommended)
Step 0's cleanup is the smallest concrete move: `git add hcd_analysis/emulator/closure_legb_figs.py` + the 5
`final_prod_seed*.hist.json`, restore the emptied `emu_bias_allfolds_mf.txt`, then choose the integration method.
Zero compute, unblocks the merge.

---

## Production-SBC scoping (appendix — done this session, do not redo)

- **Gate = Leg-A rank-uniformity SBC** (draw θ from the inference prior → simulate with the production forward →
  NUTS → check rank uniformity), run on the **N=5 production ensemble + production covariance (emucoh-ON)**,
  unblinded on mocks. This is what validation Doc B §6 literally asks for ("rank uniformity; ESS at L≥99").
  Leg-B's null is NOT rank-uniform → keep Leg-B-on-ensemble as a coverage confirmation arm only.
- **The one real code gap:** the likelihood takes a single Equinox model and both closure builders hardcode
  `checkpoints/final_fold0`; nothing ensembles `final_prod_seed0..4` at inference yet (`validate_production_ensemble.py`
  is the only place that means-over-members). Add an `EnsembleEmulator` (vmap-over-members → mean of P_filt;
  differentiable) behind a new `ckpt="ensemble"`/`ensemble_ckpts=[...]` kwarg in `build_ctx`/`build_legb_ctx`.
- **Cost:** ~17 CPU-h/mock single-member at n_samples≥600; ensemble forward ~1.5–2× (the 681×681 Cholesky-VJP is
  shared across members, only the cheap MLP is ×5) → ~25–38 CPU-h/mock. N=128 ≈ 2400–3800 CPU-h. **Pilot N≈24 ≈
  530 CPU-h** first (measures the true overhead + L_eff≥99). Levers: mtd 8→6, diagonal-mass warmup, just-clear-L≥99.
- **SLURM:** array modeled on `batch_legb_pilot.sh` (8 cpu / 24G / 24h; scratch OUTDIR; per-mock `jax.random.fold_in`).
- **Prereqs:** SBC is LF-backbone DESI(+eBOSS); MF can FOLLOW (KS-only, θ-independent, gate-invariant). Map
  dN/dX→w_c before freezing the real-fit prior so SBC + real fit share one frozen prior.
