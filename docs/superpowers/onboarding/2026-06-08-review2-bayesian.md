# Review-2 (Bayesian / PPL / inference referee) — the n_s EMU-bias resolution, z_lo=2.4 fix, MF plan

Date: 2026-06-08. Charge: scrutinize the −0.65σ n_s resolution as RESOLVED, the z_lo=2.4 default,
and the MF plan. Reproduce one cheap check; do NOT rubber-stamp. Lens = inference geometry,
decomposition validity, calibration/coverage, p-hacking risk.

Plot: `figures/analysis/review/2026-06-08-review2-bayesian.png` (4 panels: the exact decomposition
gate; the z_lo triangulation; the MF-LOSO per-sim residual + ns-coverage gap; the coverage ledger).

## Cheap reproductions I actually ran (no retrain)

1. **z-attribution gate** (loaded `nsbias_z_attribution.npz`): `Σ_z bias_z_mean = −0.646136σ`,
   `mean over 60 sims = −0.646136σ`, `|diff| = 2.10e-13`. Exactly the claimed 2.1e-13. Leg additivity
   `max|(DESI+KS) − total| = 1.1e-15`. z=2.0+2.2 KS contribution `−0.684σ = 106% of total`. Signs 10+/50−.
2. **MF-LOSO worst per-sim coherent residual**, recomputed independently from the raw HR/LF caches with
   my own match+LOSO code path: per-sim `[+0.05, −0.34, +0.26, +0.91, −0.70, −0.12]%`, **worst +0.91%**,
   pooled +0.01%. Exact match to `mf_rescorr_loso.txt`. corr(ns, residual) = −0.33; the worst sim
   (ns=0.914) is **mid-cluster**, not an edge.
3. **z_lo triangulation**: subset[0,2,5,7] z_lo=2.4 = +0.041σ (committed); all-folds z_lo=2.6 = +0.038σ
   (committed); attribution-removal of z=2.0+2.2 from the exact full fit = +0.038σ. Three independent
   estimators, all inside ±0.2σ.

## (a) Is the z-attribution math a VALID per-z decomposition? — YES, exactly.

The MAP shift is `dx = C_post · g` with `g = Σ_z g_z`, where `g_z` is the **gradient** block accumulated
over the rows whose redshift falls in bin z (`g_z[z] = Σ_{rows∈z} J[row]ᵀ (C⁻¹ΔP)[row]`). Because
`dx = C_post (Σ_z g_z)` is **linear** in g, and `bias_z = (g_z · C_post[:,ns])/σ_ns`, then
`Σ_z bias_z = ((Σ_z g_z)·C_post[:,ns])/σ_ns = (g·C_post[:,ns])/σ_ns = dx[ns]/σ_ns = bNs` identically.
This is an *exact* additive split of the Fisher MAP shift, NOT a per-z re-fit (a per-z re-fit would not
sum to the total). The script even asserts the per-sim gate at 1e-6 and the pooled gate at 1e-6; I
reproduced the pooled gate at 2e-13. **Critically, C_post is the FULL information matrix** — so each
bias_z is "the n_s MAP shift attributable to z-bin z's residual gradient, correctly propagated through
the joint posterior (τ₀ marginalized)." That is the right object; the τ₀–n_s degeneracy is handled by
C_post, not swept under. The decomposition is sound.

One honest caveat I verified: the decomposition is exact for the **full** fit, but using it to predict a
**cut** (drop z<2.4) is only *approximate* because removing rows re-inverts F → C_post changes. The
approximate removal (+0.038σ) nonetheless lands on top of the directly-measured cut numbers (+0.041,
+0.038), so the approximation is benign here. The team did NOT rely on the approximation — they ran the
actual cut on the subset and the all-folds z_lo=2.6 — which is the correct discipline.

## (b) Closure-vs-real distinction: p-hacking by deleting data? — NO. SOUND, with one missing run.

Both sides, then the rule:

**The p-hacking worry (steelman):** "−0.65σ is a real closure bias; you made it vanish by deleting the
two z-bins that carried it (z=2.0, 2.2 → 106% of the bias). That is selecting the cut that gives the
answer you want." If the z<2.4 cut had been chosen *because* it zeroed the bias, this would be a textbook
garden-of-forking-paths.

**Why it is NOT p-hacking (the decisive points):**
- The cut is **pre-registered by an external, physically-motivated criterion that predates this bias**:
  the PI's own KODIAQ-SQUAD analysis excludes z<2.8 for DLA-finder incompleteness; Karaçaylı-2021's low-z
  P1D is independently known unreliable. The bias localization (86% at KS z=2.0/2.2) is *post-hoc
  confirmation* of a *pre-existing* data-quality flag, not the reason for the cut. That direction —
  independent reason first, bias-confirmation second — is the opposite of forking-path selection.
- It is **out-of-sample, not data-deletion-to-fit-a-parameter**: the closure here is LF-emu-vs-LF-truth;
  dropping a leg's z-bins removes a *known-bad* data block from the likelihood, it does not tune a free
  parameter to the residual. The z=2.4 bin (kept) is independently CLEAN (+0.012σ) — the cut is not
  shaving until zero, it stops exactly where the physical flag says to and the kept bins are clean.
- The k-scan is the real anti-p-hacking control: at z_lo=2.0 the bias is **broadband/low-k** (survives
  removing k>0.04: −0.756σ), so it is NOT a high-k/resolution feature you could "cut away" selectively —
  it is the whole low-z KS leg being miscalibrated, exactly what the published z-cut targets.
- The DESI primary leg is +0.04σ throughout (clean), so the cosmology-driving data was never biased.

**Rule: the z_lo cut is a legitimate, externally-justified data-quality exclusion, not p-hacking.** The
one thing that keeps this from a clean GO is that the *real-fit* systematic (LF-vs-HR, −6%) is a
**different object** that the cut does NOT fix — and the resolution correctly says so. So the closure is
clean AND there is a separately-acknowledged real bias; that is the honest, non-circular framing.

## (c) MF-LOSO with n=6: honestly stated? — Mostly yes; "θ-stable" is over-strong, "0.9% residual" is honest.

- "**θ-stable (0.6%)**" is the inter-sim CoV of ρ across 6 sims. With **n=6 clustered sims** (ns∈[0.859,
  0.979], max NN gap 0.058), this is a **weak (necessary, not sufficient)** test — the script's own
  coverage caveat says exactly this, which is to its credit. But the *summary verb* "θ-stable" overstates
  what 6 LOSO points can certify: it shows the correction is self-consistent *inside* a tight cluster, not
  that it is θ-insensitive over the prior box. I would downgrade "θ-stable" → "θ-insensitive within the
  6-sim cluster; globally untested."
- "**~0.9% LOSO residual**" is honest and I reproduced it exactly (+0.91%). The team correctly reports the
  **worst per-sim** number, not the pooled signed mean (+0.01%, which cancels opposite-sign sims and would
  badly understate the per-cosmology residual). Reporting the worst-per-sim is the right conservative
  choice. The +0.91% sim is mid-cluster (ns=0.914), so it is a genuine in-cluster generalization residual,
  not an edge artifact — this *strengthens* the case that 0.9% is irreducible-at-this-fidelity, not a
  coverage hole.
- The two numbers are not in tension: CoV 0.6% (raw inter-sim scatter of ρ) < 0.9% (residual a *fixed*
  θ-independent gbar leaves) is internally consistent — a fixed mean leaves more than the raw scatter
  because it also eats z-interpolation error and the held-out point's pull on the mean.
- **Honest gap:** there is a real τ₀-dependence (~4.3pp rung0→rung19) and z-dependence (~12pp swing)
  that the *current* α-pooled gbar smears. The plan's "z+τ₀-resolved correction" is the right response;
  the LOSO residual would shrink under it. So 0.9% is the *current pooled* floor, likely reducible.

## (d) Coverage of the z_lo=2.4 + keep-[0.8,1.05]+step-C_emu-above-0.995 decision.

- **z_lo=2.4 closure:** the all-folds (60-sim) closure is directly measured only at z_lo=2.0 (−0.646) and
  **z_lo=2.6 (+0.038)**. z_lo=2.4 is the *default* but its all-folds number is **only inferred** (subset
  +0.041, attribution-removal +0.038). z_lo=2.4 and z_lo=2.6 keep the same KS z-grid except the clean
  z=2.4 bin (+0.012σ), so the all-folds z_lo=2.4 number is recoverable as ≈+0.05σ — but it should be **run
  once** before certification (cheap; 60-sim forward, no retrain). MINOR but worth closing.
- **The ~0.9% small-scale C_emu floor — in-sample vs out-of-sample:** this is the load-bearing calibration
  risk. The 0.9% is a 6-fold LOSO worst residual, which is *out-of-sample within the cluster* (good), but
  it is **not** a held-out estimate of the floor's *amplitude for a C_emu* — if you set the C_emu floor
  to the in-sample-fit LOSO residual you risk the same mild-circularity my onboarding R1 flagged for the
  zero-mean-C_emu path. The floor should be sized to the **worst-per-sim** (0.9%), not the pooled (0.01%),
  and ideally inflated to the *upper* end of the per-sim spread, since n=6 gives a very noisy variance
  estimate. As stated, sizing C_emu at ~0.9% is reasonable IF it uses the worst-per-sim and a small
  finite-sample inflation; do NOT size it to the pooled signed mean.
- **keep-[0.8,1.05] + step-C_emu above 0.995:** the right call (a hard cap would rail the posterior; eBOSS
  n_P=1.009 lands in the sparse band). The step-inflated C_emu honestly down-weights the 2/60-point
  extrapolation. Coverage there is *guarded*, not *nominal* — and that is the honest state given 2 training
  points. The MF correction itself is **untested above ns=0.979 and below 0.859** (no HR sims), so the MF
  forward inherits the same edge-untrust; the step-C_emu should also cover the MF-correction extrapolation
  uncertainty at the ns edges, not only the LF-emulator's. This is the IMPORTANT coverage point.

## Verdict

**GO_WITH_CHANGES.** The decomposition is mathematically exact (reproduced to 2e-13). The z_lo=2.4 fix is
a legitimate externally-justified data-quality cut, not p-hacking, and triangulates cleanly. The MF
diagnosis is honest about its n=6 weakness and the 0.9% residual. The changes: (1) run the all-folds
z_lo=2.4 closure once; (2) z+τ₀-resolve the MF correction and size the small-scale C_emu floor to the
worst-per-sim LOSO residual with a finite-sample inflation (out-of-sample, not pooled); (3) extend the
step-C_emu to cover the MF-correction extrapolation at the ns edges (no HR sims below 0.86 / above 0.98).
