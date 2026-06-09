# Validation-plan review — CS / ML / JAX + sampler-practice lens (2026-06-09)

**Referee:** CS / ML / JAX + sampler-practice. **Object:** how to validate the Phase-C MF Leg-B
likelihood before the real DESI+KODIAQ-SQUAD fit. **Question:** is the main agent's
convergence-first ladder right, and is the SBC coverage ensemble NECESSARY to certify the
real-fit (A_p, n_s) error bars?

**VERDICT: ENDORSE_WITH_CHANGES.** The convergence-first ladder (STEP A few-fiducial long
multichain → STEP B optional ensemble) is the correct ordering and the better use of the
~4000 CPU-h budget. The SBC coverage ensemble is **CONDITIONAL**, not mandatory: convergence +
per-mock bias on a few fiducials is the *binding* check; a *modest* coverage ensemble
(N≈40–60, not 99–600) is a cheap calibration cert worth running AFTER STEP A passes, but it is
not on the critical path to a defensible real fit. The changes below are about making STEP A
actually diagnostic (multi-chain R-hat is necessary but not sufficient as configured) and about
right-sizing STEP B so the budget is not spent certifying the wrong thing.

---

## 0. The single most important finding (read this first)

The PI's intuition is statistically correct, and the project's own machinery confirms it.

**The Leg-B "coverage" headline gate is the per-param normalized bias** `z̄ = mean_mocks[(truth −
post_mean)/post_sd]`, reported with `se = std/√N` (`closure_legb._aggregate_legb`, lines 798-808),
plus empirical interval coverage. To **resolve** `|bias| < 0.2σ` from the ENSEMBLE bias estimator
you need `se < 0.1σ` ⟹ **N ≥ 100 mocks** (verified: `se(N=99)=0.101`, `se(N=300)=0.058`). So an
ensemble below ~100 cannot even tighten the bias claim, and N=300–600 only buys you `se` 0.06→0.04.

But a **single fiducial mock run to ESS≈400** measures the per-mock bias `z = (truth −
post_mean)/post_sd` with a Monte-Carlo error on `post_mean` of `≈ post_sd/√ESS = 1/√400 = 0.05σ`.
**One long chain resolves a 0.2σ bias on that truth 2× better than the entire N=99 ensemble
resolves the AVERAGE bias** — and at a fraction of the cost. The ensemble's only added value over a
few long fiducials is (a) averaging over *which* truth (the bias could be truth-dependent), and (b)
the ECDF rank-uniformity / interval-coverage calibration statement. Both are real but secondary,
and (a) is far cheaper with ~10–20 truths at moderate ESS than 99–600.

This is why the ladder is right: **convergence + per-mock bias on a few long fiducials is the
high-information-per-CPU-h object; the ensemble is a calibration garnish, not the certificate.**

---

## (a) Is 4-chain R-hat on 2–3 fiducials the right convergence check for this 25-dim dense-mass NUTS?

**Mostly yes, with three required upgrades.** R-hat is necessary; as the plan states it, it is not
sufficient, and 2–3 fiducials is too few given the known geometry hazards.

### What R-hat over 4 chains catches and what it misses here
- **Catches:** multimodality (chains stuck in different modes → R-hat ≫ 1.01), the slope×amplitude
  (n_s–A_p) and τ₀–n_s degeneracy ridges if a single chain fails to traverse them, and gross
  non-stationarity. This is exactly the failure the PI named ("cosmology chains are hard to
  converge"). 4 chains from `init_to_median` is the standard minimum; I would use **at least 4,
  prefer 6–8**, started from **over-dispersed** inits (NOT all `init_to_median` — identical inits
  defeat the between-chain variance that R-hat relies on; see CHANGE-1).
- **MISSES (this is the load-bearing gap):** R-hat is a *between-chain/within-chain variance ratio*.
  It is **blind to a coherent bias that every chain shares** — and the entire point of this analysis
  is a coherent ≤0.2σ n_s/A_p bias. Four chains can all converge to R-hat=1.000 on a posterior whose
  MEAN sits 0.5σ from the truth, and R-hat will say "converged." **Convergence ≠ unbiasedness.** So
  STEP A MUST report, per fiducial, BOTH (i) R-hat/ESS (sampler health) AND (ii) the per-mock bias
  `z = (truth − post_mean)/post_sd` for A_p and n_s (recovery). The walkthrough's MF gate already
  measures the *Fisher* bias (+0.026σ); STEP A's job is to confirm that survives in the *full NUTS
  posterior* (non-Gaussian τ₀–n_s curvature, prior edges) — that is the thing the Fisher cannot see
  and the ensemble would otherwise certify.

### The specific red flags STEP A must trip on
1. **R-hat > 1.01** on any of (A_p, n_s, the τ₀-ladder, α) — use rank-normalized split-R-hat
   (arviz `az.rhat(method="rank")`), NOT the classic R-hat (the classic one is fooled by heavy tails
   / non-stationary variance, both plausible for τ₀ in the α-ladder coordinate).
2. **min bulk-ESS < ~400 and min tail-ESS < ~400** per parameter (tail-ESS matters for the 95%
   credible intervals that the coverage claim rests on; bulk-ESS alone is not enough for interval
   edges).
3. **Any divergences after the target_accept escalation ladder** (0.9→0.95→0.99 is already wired,
   `closure_legb.py:697`). The mtd=10 profile shows **0/60 divergences** — excellent — but that is
   n=1 at one truth. A high-n_s fiducial (near the eBOSS/HR-edge ns≈0.98–1.01) is the worst case for
   the τ₀–n_s funnel + the n_s-edge C_emu ramp; it MUST be one of the fiducials (see CHANGE-2).
4. **max_tree_depth saturation > ~1–2%.** The mtd=10 profile reports **0/60 max-depth hits, mean
   depth 6.28, max 8** — trees are NOT saturating, so mtd=10 is adequate and the funnel is not
   forcing pathologically deep trees at this truth. STEP A must re-confirm this at the high-n_s
   fiducial; if depth saturates there, that is the funnel signature (CHANGE-3).
5. **The τ₀–n_s funnel directly:** plot the bivariate (n_s, log σ_τ0-ladder) and (A_p, n_s)
   pairwise; a funnel shows as a narrowing-variance neck the sampler under-explores (low ESS
   localized at the neck). A divergence-free run with low ESS *only* near the neck is the classic
   non-centered-reparam tell (CHANGE-4 below).

### Is 2–3 fiducials enough?
**No — bump to a small grid (CHANGE-2).** 2–3 covers the n_s lever but not the joint hazard
surface. The geometry that breaks 25-dim dense-mass NUTS is not at the n_s extremes per se; it is at
(high n_s) × (high τ₀, where the resolution sign-flip lives) × (HCD-edge α). Use ~6 fiducials
spanning {low/mid/high n_s} × {low/high A_p} plus one deliberately at the **n_s-edge ns≈0.99–1.01**
(extrapolated MF + sigma_edge active) and one at the **τ₀ ladder extreme**. That is ~6 mocks ×
4 long chains ≈ 24 chain-units — still cheap (see (c)).

---

## (b) ESS/sample 0.44 at mtd=10 — practical implications

ESS/sample ≈ 0.44 (n_s, A_p), 0.63 (τ₀) is **healthy for a 25-dim dense-mass NUTS** — that is the
expected range for a moderately correlated posterior; it is not a pathology. Implications:

### Chain length / thinning for the ECDF L≥99
- The ECDF gate's `L_FLOOR=99` is **L = thinned-to-near-independence draws** (`thin_to_ess`,
  `closure_diagnostics.py:76`), NOT raw samples. At ESS/sample 0.44, **99 independent draws ⟹ ~225
  raw post-warmup samples MINIMUM**. The profile targets ESS≥400 (902 samples) — comfortably above
  the L=99 floor — but note the gate sizing: if STEP B uses a *moderate* chain (ESS≈100–150/mock to
  save budget), 99 ≤ L is exactly at the floor and the ECDF band is only marginally calibrated
  (the `L_FLOOR` comment in `closure_sbc.py:49` measured type-I = 0.05 at L=99, rising fast below).
  **Do not run STEP B below ~250 raw samples post-warmup**, or the per-mock thinned L drops under 99
  and the ECDF becomes a path check, not a calibration verdict.
- **Important interaction:** `thin_to_ess` thins by the WORST parameter's ESS. The τ₀-ladder (13
  dims) or the α-block can have lower ESS than (A_p, n_s); the profile only reports A_p/n_s/τ₀(z≈3).
  STEP A MUST report **min-ESS over ALL 25 params**, because the thinning (and thus L for the
  per-mock coverage) is set by the worst one, not by n_s.

### Dense vs diagonal mass
- **Keep dense_mass=True for production** (the [θ,τ₀] correlation is real physics, per
  `sampler_numpyro.py:80` and the MF rotation review's −6% σ(n_s) tightening from the n_s–τ₀
  coupling). Diagonal mass would inflate leapfrogs/sample on the correlated ridge. The smoke's
  `--diag-mass` is a *speed hack for the path check only* (`closure_legb.py:895`), not production.
- **Dense-mass warmup is the cost driver, not sampling.** The profile: warmup ≈ 833 s (40 warmup),
  sampling 20.8 s/sample. Dense-mass adaptation on 25 dims needs enough warmup to condition the
  full mass matrix; **40 warmup is too few for a robust production run** (it worked at one easy
  truth). The first mtd=10 attempt was *killed after ~19 CPU-h on a pathological deep-tree warmup*
  (experiment log §1.5) — that is exactly the dense-mass-warmup-underconditioned failure mode.
  Use **≥150, prefer ~250–300 warmup** for STEP A/B (CHANGE-5); the firm cost table already
  budgets warmup=150–250.

### Non-centered reparam for the τ₀-ladder / α — the highest-leverage efficiency lever
- The τ₀-ladder is currently a **CENTERED** Normal: `alpha_ladder ~ Normal(tau0_mu/kim,
  tau0_sigma/kim)` (`closure_legb.py:537-538`). The α-DLA is `softplus(Normal)` (one-sided, fine).
  A 13-dim centered hierarchical-ish ladder with a tight data-driven prior is a **classic funnel
  source** when the data is weak at some z (the high-z τ₀ bins, where KS/DESI have few rows). If
  STEP A shows low ESS or depth saturation localized in the high-z τ₀ bins, the fix is a
  **non-centered reparam**: `alpha_raw ~ Normal(0,1); alpha_ladder = mu/kim + (sigma/kim)*alpha_raw`.
  This is a near-zero-risk JAX change (the change of variable is linear, Jacobian-free, exactly as
  the centered version's docstring already notes the Kim(z) constant is). **I recommend pre-emptively
  testing the non-centered ladder as a STEP-A arm** even if the centered version converges — it
  typically halves leapfrogs on funnel-prone hierarchies and would cut the ~6 CPU-h/mock materially,
  which is what makes a larger STEP B affordable. (CHANGE-4.) It must be golden-tested for posterior
  identity vs the centered version on a fixed seed before adoption.

---

## (c) The cost trade: few-long vs many-moderate — best use of ~4000 CPU-h

**Few-long-chains wins decisively for the certification question, with a small ensemble as a
follow-on.** The arithmetic (grounded in the firm mtd=10 profile, `legb_mtd10_profile.txt`):

| arm | config | cost | what it certifies |
|---|---|---|---|
| **STEP A** (recommended) | ~6 fiducials × 4 chains × ESS≈400 | ~6 × 4 × 5.45 ≈ **130 CPU-h** | R-hat convergence + per-mock bias (A_p,n_s) at the truths that matter, incl. the n_s-edge + τ₀-extreme funnel |
| STEP B small | N≈48 mocks × ESS≈400, 1 chain | ~48 × 5.45 ≈ **260 CPU-h** | + ensemble-averaged bias (se≈0.14σ) + ECDF/interval coverage at L≈400 |
| STEP B medium | N≈99 × ESS≈400 | ~540 CPU-h | + bias se≈0.10σ, the L_FLOOR-valid ECDF |
| (original) N=300 | | 1.6–2.0k CPU-h | bias se≈0.06σ |
| (original) N=600 | | 3.65–4.0k CPU-h (AT ceiling) | bias se≈0.04σ |

**STEP A + STEP B-medium ≈ 670 CPU-h** — ~17% of budget — and certifies everything the real fit
needs: sampler health (R-hat, ESS, divergences, no funnel) AND unbiasedness at the truths that
matter AND a properly-sized ensemble coverage statement. **N=300–600 spends 1.6–4.0k CPU-h to push
the bias se from 0.10→0.04σ — a refinement well below the 0.2σ gate, on the WRONG axis** (it
averages over truth-realization noise, which a few long fiducials already pin per-truth). That is
the budget mistake the PI is right to resist.

**The single empirical caveat that could flip this:** if STEP A reveals the per-mock bias is
strongly *truth-dependent* (e.g. clean at mid-n_s, drifting to 0.3σ at the n_s-edge), then a
many-mock ensemble IS warranted to map the bias-vs-truth surface — but you would then size it to
the surface (a denser grid near the edge), not a flat N=600. STEP A is the gate that tells you
whether STEP B needs to be big. **Sequence them; don't pre-commit the 4000.**

### SLURM sharding (either path)
The code is already correctly built for this: each mock is seeded by `jax.random.fold_in(seed, m)`
(`run_legb`, `closure_legb.py:692`) and `mock_indices` lets a SLURM array split `range(n_mocks)`
across tasks with the merged set bit-identical to a single run (`return_per_mock=True` for the
cross-shard merge, lines 663-743). **For STEP A** (few mocks × many chains), shard over
**(mock, chain) pairs** — run `num_chains=1` per SLURM task at distinct seeds and merge chains
host-side for R-hat (do NOT use numpyro `num_chains=4` in one task: on CPU it vectorizes/loops the
chains in one process and you lose the embarrassing parallelism + a single task holds 4× the wall).
One core per task, ~5.45 CPU-h each, ~24 tasks for STEP A. **For STEP B**, one task per mock as the
code already supports. Cap concurrent tasks per the cavestru0 etiquette and checkpoint per-mock npz
so a preempted task resumes without recompute.

---

## (d) Pitfalls in running the coverage ensemble (seeding, fast_postprocess, reproducibility)

1. **`fast_postprocess` reproducibility — VERIFY, don't trust.** `fast_postprocess=True` is now the
   default (`_run_nuts_legb`, `closure_legb.py:596`) and reconstructs the 3 deterministic sites
   host-side (`_legb_reconstruct_deterministics`) instead of the 237.6 s/mock full-model replay.
   The docstring claims byte-identical (rtol=0). **CS requirement: pin this with a golden test
   BEFORE the fan-out** — run ONE mock both ways (`fast_postprocess=True/False`) and assert
   `allclose(rtol=0)` on `tau0_vec`, `alpha_dla`, `alpha_hcd_z`. The CS onboarding note already flags
   (R1) that **no golden-regression guard exists on disk** (`tests/golden/` absent); a postprocess
   refactor that silently changed the reconstruction would corrupt the τ₀ coverage of the ENTIRE
   ensemble and only show up as a coverage anomaly, not a crash. This is the highest-severity
   ensemble pitfall.

2. **Per-mock seeding splits the SAME key for mock-noise AND NUTS** (`closure_legb.py:692`:
   `k_mock, k_nuts = split(fold_in(key0, m), 2)`). This is correct and reproducible, BUT the
   divergence-retry ladder does `seed=base_seed + attempt` (line 702). `base_seed` is drawn from
   `k_nuts` (line 696), so it is per-mock-deterministic — good. **Pitfall:** for STEP A's multichain,
   chains MUST use genuinely independent seeds (`fold_in(k_nuts, chain_id)`), not `base_seed+chain`,
   to avoid correlated streams; `+attempt` is fine for the retry (different geometry, not a
   between-chain estimate) but `+chain` for R-hat risks subtle seed-collision with the retry path.
   Use a separate `fold_in` axis for chains.

3. **The thinned-L mismatch across mocks is handled but watch the floor.** `_aggregate_legb` uses
   `L_eff = min over mocks` and subsamples every mock's ECDF ranks to `L_eff`
   (`closure_legb.py:773, 816`). If ONE mock thins to L=40 (a low-ESS chain), the ENTIRE ensemble's
   ECDF is sized at L_eff=40 < L_FLOOR=99 and silently becomes path-only (`gate_valid=False`,
   line 820). **STEP B must guarantee every mock clears ESS so its thinned L ≥ 99** — i.e. budget
   the per-mock samples for the WORST-ESS parameter, not n_s. A single hard mock can demote the
   whole ensemble's calibration verdict. Mitigation: set `target_per_param` in `thin_to_ess` and
   pre-screen / re-run any mock with min-ESS < target before aggregating.

4. **Mock reuse beyond the sim count.** `sim = sims[m % len(sims)]` (line 688) cycles the held-out
   sims with fresh noise when `n_mocks > #sims`. Fold-0 has ~8 held-out sims (the CS onboarding
   note). So **N=99 mocks reuses each truth ~12×** — the ensemble's "different truth from the prior"
   framing in the handoff is **NOT what the code does**: it is the SAME ~8 sim-truths with different
   cosmic-noise draws. This materially changes the interpretation: the ensemble measures
   coverage/bias *over noise realizations of 8 fixed truths*, not over the prior. **This is actually
   an argument FOR the few-fiducial approach** (the ensemble is not sampling truth-space densely
   anyway) and AGAINST sizing to N=600 (you are re-drawing noise on 8 truths 75× — diminishing
   returns past ~3–5 noise draws per truth, i.e. N≈30–40 already saturates the noise-averaging).
   **Flag for the meta-reviewer: the N≥99 "different mocks from the prior" premise is not met by the
   8-sim fold-0 pool; either pool held-out sims across folds for more truths, or accept that STEP B
   is a noise-coverage check over a handful of truths — in which case N≈40 suffices and N=300–600 is
   wasted.**

5. **x64 / import-order across SLURM tasks.** Every entry point must `import hcd_analysis.emulator`
   before `import jax` (CS onboarding §x64). The closure modules do; any thin SLURM wrapper script
   must too, or the whole ensemble silently runs float32 and the Cholesky/identities degrade. Add
   the `assert jax.config.read("jax_enable_x64")` to the wrapper.

---

## Recommended compute-efficient path (the deliverable)

1. **STEP A (≈130 CPU-h):** ~6 fiducials (low/mid/high n_s × low/high A_p, + one n_s-edge ns≈0.99–1.01,
   + one τ₀-ladder extreme) × 4–8 over-dispersed-init chains × ESS≥400, warmup≥150, mtd=10,
   dense_mass. Report per fiducial: rank-normalized split-R-hat (gate <1.01), min bulk+tail-ESS over
   ALL 25 params (gate ≥400), divergences after the 0.9→0.99 ladder (gate 0), max-tree-depth
   saturation (gate <~2%), the (n_s, A_p) and (n_s, log σ_τ0) funnel pair-plots, and the per-mock
   bias z for A_p and n_s (gate |z|<0.2σ, MC error ~0.05σ at ESS=400). **Also run the non-centered
   τ₀-ladder as a parallel arm** and adopt it if it cuts leapfrogs without changing the posterior
   (golden-pinned).
2. **GATE on STEP A.** If R-hat clean + bias <0.2σ at all fiducials INCLUDING the edge + no funnel →
   the real-fit error bars are convergence-certified and per-truth-unbiased. If the edge fiducial
   drifts → STEP B sized to map the bias-vs-n_s surface near the edge (not a flat N).
3. **STEP B (CONDITIONAL, ≈260–540 CPU-h):** N≈48–99 single-chain ESS≥400 mocks (NOT 300–600). This
   is the calibration garnish: ensemble-averaged bias (se 0.10–0.14σ) + the L≥99 ECDF/interval
   coverage. Pool truths across folds if a denser truth-space is wanted. Pre-screen min-ESS so every
   mock clears L≥99. **Total STEP A+B ≈ 400–670 CPU-h (≤17% of budget).**
4. **Pre-flight (mandatory, CS):** the `fast_postprocess` golden assert + a golden `(P_model,
   C_total)` freeze on the LF and MF paths (CS onboarding R1) BEFORE any fan-out.

---

## The subfield standard (Lya P1D — what these analyses ACTUALLY run)

The directly-comparable Lyα-forest P1D cosmology analyses do **NOT** run an SBC coverage ensemble.
Across the three closest references the validation is uniformly **a few fiducial closure tests +
leave-one-out emulator validation + many systematic-variation runs**, with convergence handled
by the (auto-converging) sampler's internal R-hat threshold and rarely reported explicitly:

- **PRIYA KODIAQ-SQUAD / XQ100 (Bird et al, arXiv:2509.18271)** — the *same data, same PRIYA
  multi-fidelity (60 LF + 3 HF) setup* as this analysis. Validation = a **single closure test** on
  one simulated spectrum with the KODIAQ-SQUAD covariance ("all parameters within the 1σ range of
  the correct input values", their Fig. 2) + LOO-CV emulator accuracy (~1% at k=0.01–0.06 s/km).
  Sampler: **Cobaya / Metropolis** with a learned-covariance proposal; **no R-hat, chain count, or
  ESS reported.** No coverage ensemble.
- **eBOSS / Lyssa GP emulator (arXiv:2412.05372, IOP JCAP 2025)** — closure on the fiducial sim +
  one external B13 sim; LOO emulator tests (percent-level). Sampler **MontePython v3.0**; **no
  explicit R-hat / chain / ESS.** No SBC ensemble.
- **DESI DR1 Lyα P1D cosmology (arXiv:2601.21432, 2026)** — the modern bar. Validation = closure on
  **~2 held-out validation sims** (mpg-central/seed, lyssa-central) across code/resolution/IC
  variants + **a large number of alternative data-analysis variations** with **parameters BLINDED
  throughout** (their robustness philosophy). The validation effort is in the *systematic
  variations* (estimator, covariance, emulator variant, contaminant model), NOT in an SBC coverage
  ensemble — which is not described. `cup1d` likelihood; convergence specifics not detailed in the
  text.

**Implication for this lens:** the convergence-first, few-fiducial ladder IS the subfield standard,
and this analysis would *exceed* it by adding (i) explicit rank-R-hat/ESS/divergence/funnel
reporting (which the published papers do not even show) and (ii) an optional modest coverage
ensemble. The original N=99–600 SBC ensemble would be *above* the subfield norm — defensible and
admirable as over-validation, but **not necessary** to certify the (A_p, n_s) error bars, and the
budget is better spent on the systematic-variation runs the DESI DR1 paper treats as the real
robustness evidence (MF-on/off, KS-in/out, C_emu inflation, the n_s-edge guard). NB: the SBC
literature (Talts+2018, Säilynoja+2022, Modrak+2023, all cited in `closure_diagnostics.py`) is the
right machinery IF you do run an ensemble — the code implements it correctly — but SBC is a
sampler+model *self-consistency* check (rank-uniformity under the prior), which the Leg-B null is
explicitly NOT (truth is a held-out sim, not a prior draw; the code correctly tags the ECDF
DIAGNOSTIC-ONLY, `_aggregate_legb` line 810-813). So even a full ensemble here is a *coverage* check,
not true SBC — which further weakens the case that it is mandatory.
