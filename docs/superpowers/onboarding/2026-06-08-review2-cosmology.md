# Cosmology / P1D-emulator review #2 — the n_s-bias resolution + z_lo=2.4 + MF plan (2026-06-08)

Reviewer: cosmology / P1D-emulator lens (standing per-checkpoint adversarial review).
Reproduced from disk this session (caches, npz, code). Env:
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

Plot: `figures/analysis/review/2026-06-08-review2-cosmology.png` (script `scripts/review2_cosmology_plot.py`).

## VERDICT: GO_WITH_CHANGES

The physics of the resolution is sound and I reproduced its load-bearing numbers independently. The
z_lo=2.4 fix is correct as a *bias closure* but I recommend the published z_lo=2.8 as the *default* for
the real fit (see below). The MF plan is on the right track but has three physics gaps that must be closed
before the real-fit cert.

---

## Cheap reproductions I ran (independent of the diag bookkeeping)

1. **z-attribution linear-Fisher consistency (npz).** Σ_z bias_z = −0.6461 vs pooled mean −0.6461,
   |diff|=2.1e−13 (exact, as it must be for a linear Fisher decomposition). DESI total = **+0.0388σ**
   (matches the "+0.04σ clean" claim); KS z≤2.2 = −0.6763σ = **105% of total**. Predicted bias after a
   KS z<2.4 cut (DESI all-z + KS z≥2.4) = **+0.030σ** — and the actual all-folds rerun at z_lo=2.6 gave
   +0.038σ, so the linearization is faithful (the cut genuinely closes it, not a Fisher artifact).

2. **Independent LF−HR clean-P1D deficit (my own cache read, NOT the diag script).** Reading
   `observables_tau0_{hr,lf}.h5` directly, matching on (params, z, alpha_idx), interpolating HR(525)→LF k:
   - Using the **filtered clean class** (the emulator's actual P_filt target), low-z (z=2.0,2.4) high-k
     (k>0.04) deficit z-trend: z=2.0 **−9.5%**, z=2.2 **−7.2%**, z=2.4 **−4.5%**, z=2.6 −2.5%, → ~0 at
     z=3.4, **+0.9% at z=4.0** (sign flip). k-growth at low z: +4.8% (k~0.005) → +4.2% → +0.8% →
     **−4.4%** (k>0.04) — a clean tilt pivoting near k~0.025. **6/6 low-z sims negative.** This matches the
     diag txt `(2c)` table to within my tighter overlap-band restriction (diag: z=2.0 −7.6%, z=3.4 +0.08%).
   - The diag's headline −6.46% uses the filtered product over a slightly wider band; both are the same
     physical effect. **Sign, k-growth, z-trend, and 6/6 coherence all independently reproduced.**

3. **tau0-dependence physics (cache read).** At z=2.2 the τ₀ rungs map alpha_slope 0.656/0.976/1.331 →
   target_F 0.900/0.855/0.808. The MF correction ρ is largest at rung0 (low τ₀, **high ⟨F⟩**, +8.1%) and
   smallest at rung19 (+3.9%). Physically coherent (see §C).

4. **The 2.4-vs-2.8 difference is closure-invisible.** KS z=2.4 contributes +0.0094σ, z=2.6 −0.0014σ; the
   two bins where z_lo=2.4 and z_lo=2.8 *differ* sum to **+0.008σ**. So the cut choice is NOT a
   closure-bias decision — it is a data-systematics decision.

---

## (a) Is "−0.65σ = low-z KS data + LF resolution" coherent, and is DESI-clean robust?

**Coherent — and DESI-clean is robust.** The −0.65σ is two physically distinct objects in the SAME
low-z/small-scale corner, correctly separated:
- **Closure object (LF-emu vs LF-truth):** A coherent ΔP=P_truth−P_emu residual localized to KS z=2.0+2.2,
  amplified into n_s by the KS low-z covariance. The β≈0.98 slope-faithfulness (D1) rules out a response
  defect; the residual is a *misprediction shape*, not an attenuation. Dropping z<2.4 collapses it.
- **Real-fit object (LF vs HR/reality):** The −6% low-z high-k LF resolution deficit. The LF-only emulator
  faithfully reproduces the LF truth, so it inherits this systematic vs reality. Fixed by MF, NOT by the
  z-cut (the deficit persists, weaker, at z=2.8–3.4 high-k: my recompute gives −1.1% at z=2.8, −0.3% at
  z=3.2 — small but nonzero in the KS band).

**DESI-clean (+0.04σ) is robust** because the DESI leg is capped at k≤0.02 and z≥2.2 (loader default
z_lo=2.2): it never touches the high-k tilt band where the LF deficit lives, and it sums to +0.039σ across
all z (Panel A). The DESI primary cosmology was never biased. This is the single most reassuring result in
the package: the survey that drives the cosmology constraint is clean by construction (low-k, and its low-z
bins carry ≤+0.025σ each).

**One nuance I flag:** the closure used *sim truth* for the KS leg. The closure cannot "see" that real KS
z=2.0/2.2 are DLA-contaminated — it can only see that those bins, with their real covariance, amplify the
LF-emu residual. So the closure result ("z=2.0/2.2 drive the bias") and the data-quality argument ("those
bins are DLA-incomplete in the real data") are *consistent and mutually reinforcing* but are NOT the same
measurement. The honest framing: the closure tells you the low-z KS *covariance* over-weights a regime the
LF emulator mispredicts; the published z<2.8 cut tells you that same regime is *also* data-unreliable. Both
point to dropping it. Good.

## (b) Is the LF−HR −6% deficit consistent with PRIYA's published ~7% convergence?

**Yes, in sign and k-growth — with one important caveat the package under-states.** PRIYA (Bird+2023)
reports the LF (1536³) converges to percent-level at k<0.05 and ~7% at k~0.1, worst at He-II reionization
z≈3–4. My recompute confirms:
- **Sign:** LF is power-DEFICIENT at high k (the higher-resolution HR has MORE small-scale power) — correct
  for an under-resolved box.
- **k-growth:** monotone from +5% (low-k, LF excess) through a pivot at k~0.025 to −9% near Nyquist at
  low z — a genuine tilt, exactly the shape that lands on n_s.
- **Magnitude:** −4 to −9% at low-z high-k, bracketing the published ~7% (Panel B gold band).

**CAVEAT (raised to IMPORTANT below): the z-trend SIGN FLIPS.** My recompute shows the deficit is
strongly NEGATIVE at low z (−9.5% at z=2.0) but goes through zero at z≈3.4 and becomes POSITIVE (LF
*excess*, +0.9% at z=4.0, growing further at z>4.6). The package's framing ("−6% at low-z high-k, matching
PRIYA worst at z~3–4") is half-right: the *magnitude* peaks at low z in MY (and the diag's) measurement,
NOT at z~3–4. PRIYA's "worst at z~3–4" is the He-II reionization thermal-broadening regime; what I see
dominating at z=2.0–2.2 is a DIFFERENT, larger effect (the low-z IGM is at very high ⟨F⟩≈0.9 where
small-scale density structure is maximally resolution-sensitive). The two are not in conflict — they are
different physical drivers at different z — but the MF correction must handle BOTH the low-z negative
deficit AND the high-z positive excess, and a correction tuned only on the low-z magnitude will mis-sign
the high-z bins. The gbar(z,k) table (FixedMeanHead) does interpolate in z, so it can represent a sign
flip — but the LOSO residual and the C_emu floor must be measured z-resolved, NOT pooled (a pooled coherent
mean cancels the opposite-sign z-bins, hiding the per-z residual — exactly the "+0.01% pooled vs +0.91%
worst-per-sim" cancellation the LOSO txt already warns about).

## (c) Is the z+τ₀-resolved MF correction physically sensible? Does τ₀-dependence of a *resolution*
correction make sense?

**Yes — the τ₀-dependence is physical, not a bug, and the plan is right to make the correction τ₀-resolved.**
A "resolution" correction is the ratio of P1D(δ_F) between two particle loads, and P1D depends on the flux
field δ_F = F/⟨F⟩−1, not the raw density. Which structures survive into δ_F is set by the mean flux (τ₀):
- At **low τ₀ / high ⟨F⟩** (z=2.2: ⟨F⟩≈0.90), the forest is in the weakly-absorbed, low-density regime
  where small-scale density fluctuations map nearly linearly into transmission — exactly the small-scale
  structure the LF box under-resolves. So the correction is LARGEST here (+8.1%).
- At **high τ₀ / low ⟨F⟩** (⟨F⟩≈0.81), saturation in the denser regions washes out small-scale transmission
  structure, so P1D is less resolution-sensitive → smaller correction (+3.9%).

So a ~4pp τ₀-swing in a *resolution* correction is the expected mean-flux modulation, NOT a contradiction.
The plan's instinct ("mean-flux × small-scale") is correct. The repo's current default (`delta_mode='none'`,
FixedMeanHead) carries gbar(z,k) **pooled over alpha/τ₀**, which averages +8.1% and +3.9% into a single
number per (z,k) — it WILL mis-correct the τ₀ extremes by ~±2pp. The plan to make it τ₀-resolved is
warranted. **The (z,τ₀) sign-flip interaction** (z=5.0 shows ρ<1 at high τ₀, i.e. LF *excess* — Panel C
navy curve) is also real in the cache and means a separable g(z)·g(τ₀) form is insufficient; the correction
needs a genuine 2D (z,τ₀) table or a low-order interaction term. This is the strongest single argument for
going beyond the α-pooled gbar.

**IMPORTANT subtlety on the MF architecture (this lens):** the existing `multifidelity.py` ALREADY contains
two separate factors — `gbar(z,k)` (the LF→HR clean-P1D ratio measured here) AND a `res_corr(z,k)` table
(the L15n512/L15n384 particle-convergence factor from PRIYA's own convergence runs). These are NOT the same
object and must not be double-counted. The 6-HR-sim LF→HR ratio (gbar) is a *box-resolution* (1536³→3072³)
correction; res_corr is a *particle-load* convergence factor at fixed box. The diagnostics in this package
(lf_vs_hr_highk, mf_rescorr_*) all measure the gbar object (6 HR sims = 6 LF design points). Whether the
real-fit forward should apply gbar AND res_corr, or whether gbar already subsumes the L15 convergence, is
NOT addressed in the resolution and is a live double-counting hazard (see concerns).

## (d) The ns>0.995 edge + the MF ns∈[0.86,0.98] coverage gap

**The keep-[0.8,1.05]+step-C_emu decision is defensible and I agree with it over a hard cap.** A hard cap
at 0.995 would rail the posterior against a wall right where eBOSS n_P=1.009 lands, biasing n_s low —
exactly the failure mode we are trying to remove. Carrying a step-inflated C_emu above 0.995 is the correct
honest treatment of a sparse extension (2/60 design points >1.0, gapped, α_q-corner-pinned per the prior
report).

**BUT the MF coverage gap is a genuine and under-appreciated problem (raised to IMPORTANT).** Panel D makes
it visceral: the 6 HR sims span n_s∈[0.859, 0.979], and the real fit is expected to land near n_P≈1.0
(Planck 0.965, eBOSS 1.009) — i.e. **at or above the upper edge of where the MF correction was ever
measured.** The MF LOSO certifies θ-stability ONLY inside [0.86,0.98]; the ~0.9% worst-coherent residual
is an *interpolation* number. At n_s~1.0 the τ₀-resolved correction is EXTRAPOLATING, and the n_s-direction
of the LF→HR ratio is precisely the tilt direction we cannot afford to get wrong. The step-C_emu above
0.995 was sized for the *emulator's* sparse-extension uncertainty (the LF interpolation), NOT for the *MF
correction's* extrapolation uncertainty. These are two different objects and the C_emu must budget BOTH.
The fact that the real fit lands in BOTH the MF coverage gap AND the sparse LF extension simultaneously is
the highest-risk corner of the whole analysis, and the package treats them as one.

---

## Agreements
- The z-attribution is exact and decisive; DESI (the cosmology driver) is clean.
- β≈0.98 correctly retires the tilt-attenuation / normalization-loss (#3/#4) mechanism as the *cause*.
- The LF−HR deficit is real, coherent (6/6), tilt-shaped, and consistent with PRIYA in sign and k-growth.
- The τ₀-dependence of the resolution correction is physical (mean-flux modulation of small-scale
  resolution sensitivity), so τ₀-resolving it is the right call.
- keep-[0.8,1.05]+step-C_emu beats a hard ns<1.0 cap.
- The MF LOSO caveat (6 clustered sims, weak test, edges untested) is stated honestly in the txt.

## Concerns (severity + fix) — see structured output.

## z_lo assessment
z_lo=2.4 is the *minimal* cut that closes the closure bias (z=2.0+2.2 carry 105%; z=2.4/2.6 add only
+0.008σ, so 2.4 and 2.8 are closure-indistinguishable). But the choice is a DATA-systematics decision the
closure cannot adjudicate: the published KODIAQ-SQUAD analysis excludes z<2.8 for DLA-finder incompleteness,
and the plan body + memory both say z_lo=2.8. The implemented default (2.4) is the *less conservative* of
the two and is NOT the published cut. Since dropping the extra two bins (z=2.4,2.6) costs essentially zero
constraining power (they add +0.008σ of closure signal and a handful of high-k KS points), I recommend
**defaulting to z_lo=2.8 to match the published analysis** and exposing 2.4 as an opt-in for sensitivity.
At minimum the docstring's "the KODIAQ-SQUAD analysis excludes z<2.8" while the default is 2.4 is an
internal inconsistency that must be reconciled (the docstring justifies 2.8 then sets 2.4). This is
GO_WITH_CHANGES, not NO_GO: 2.4 is scientifically safe for closure; the change is to align the *default*
with the published data cut for the real fit.

## mf_plan_assessment
The z+τ₀-resolved MF correction is the right design and the τ₀-dependence is physically justified. Three
changes before the real-fit cert: (1) the LF−HR deficit SIGN-FLIPS with z (negative low-z, positive
z>3.6) — the correction and especially the LOSO residual + C_emu floor must be measured Z-RESOLVED, never
pooled (pooling cancels the sign and hides the per-z residual); a separable g(z)·g(τ₀) is insufficient
because of the (z,τ₀) interaction — use a 2D table or an interaction term. (2) The 6 HR sims span only
n_s∈[0.86,0.98] but the real fit lands near 1.0 — the MF correction EXTRAPOLATES in the tilt direction at
the landing point; the step-C_emu above 0.995 must budget the MF *extrapolation* uncertainty separately
from the LF *interpolation* uncertainty (they are distinct objects). (3) Clarify gbar vs res_corr
double-counting: the forward already has both an LF→HR box-resolution ratio AND an L15 particle-convergence
res_corr — confirm they are not both applied to the same physical effect. Then re-run
diag_emu_bias_allfolds through the full MF forward (z-resolved residual) and carry the worst per-z coherent
residual (~0.9%) as a small-scale-leg C_emu floor.
