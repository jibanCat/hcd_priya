# Lyα-forest / IGM / PRIYA checkpoint review — HCD slope-prior spec — 2026-06-06

Lens: is the HCD-incidence physics in the slope-prior spec FAITHFUL — to the PRIYA sim's
measured w_c(z), to the observed dN/dX literature, to the telescoping-Poisson w_c map, and
to the data's DLA-masked state? Everything below is either read from code (file:line),
read from the sweep results (`legb_slope_prior_tradeoff.txt`), or reproduced LIVE this
session (the lit-derived g_fixed slope vs the sim w_c slope — `/tmp/q1_gfixed.py`, group-mapped).

## Verdict summary
**GO-WITH-CHANGES.** The factorized α(z) = α_pivot·exp(δ_leg)·g_fixed(z)·((1+z)/(1+z_p))^s_c
is the right physics and the LOCKED width is sound. But the spec must (1) adopt the
**production-faithful lit-derived g_fixed** (Q1: yes — and I verified it carries BOTH the
incidence AND the path-length growth correctly); (2) **re-derive the bias number on that
g_fixed** — the sweep's +0.167σ mixes two different g_fixed framings (closure-only g_fixed,
production-framing bias offset), so the headline bias is computed against a construction the
closure does not run; (3) **re-center s_c on the differential against the chosen g_fixed**
(Q2: WLS lit values, but the center depends on which g_fixed you pick). Q3's exact-bias
recompute is REQUIRED and cheap. Q4: amplitude float is correct; do NOT also float the slope.

---

## Q1 — Should the closure use the PRODUCTION (lit-derived) g_fixed? **YES. This is the load-bearing call and it is mine.**

**The physics walk (verified live).** Production has no sim w_c(z). The honest g_fixed must
be built from the observed dN/dX through the SAME telescoping-Poisson map the amplitude prior
uses (`dndx_wc.w_c_from_mu`, `dndx_wc.py:12-19`): w_c = (1−e^{−μ})·e^{−(higher classes)},
μ = dN/dX·X̄(z). The question the onboarding flagged is whether a lit-derived g_fixed
correctly carries BOTH the incidence growth AND the absorption-path growth X̄(z). **I built it
and checked** (`/tmp/q1_gfixed.py`: literature power-law dN/dX_c(z) from O'Meara/Zafar/PW09 ×
the sim's implied X̄(z), pushed through `w_c_from_mu`):

```
            w_c abs slope d ln w_c/d ln(1+z)
class    sim (empirical w_c)   lit-derived g_fixed   GAP(sim−lit)
LLS         +2.60                  +3.20               −0.60
subDLA      +2.72                  +2.75               −0.03
X̄(z) slope d ln X̄/d ln(1+z) = +2.02   (the path-length growth, carried by both)
```

So the lit-derived g_fixed **does** carry both: ~+2.02 of the LLS slope is the path-length
X̄(z) growth (which is θ-/incidence-independent geometry, identical in sim and lit), and the
remaining ~+1.18 is the incidence dN/dX growth. The map is faithful — w_c is NOT just an
incidence, and the telescoping-Poisson form correctly composes incidence × path × class
shadowing. **This confirms the production-faithful g_fixed is well-defined and physical.**

**The shape gap coverage must survive is physically bounded by the dN/dX spread — YES.**
The sim-vs-lit shape gap on the LLS w_c slope is **−0.60** (lit shape steeper: γ_lit^LLS=+2.31
vs γ_sim^LLS=+1.36, a +0.95 dN/dX ratio-slope, telescoping-damped to +0.60 in w_c-slope units).
That gap is exactly the literature ratio-slope `HCD_LIT_OVER_SIM_SLOPE[0]=0.95` (`inference.py:74`)
that the amplitude prior already uses, and the WLS 1σ on it is 0.52 (LOCKED width). So the
gap the closure must absorb is ~1.15σ of the slope prior for LLS — comparable scale, within
the literature dN/dX spread. The physics is consistent: the same dN/dX uncertainty that sets
the prior width also bounds the sim-vs-lit shape gap. **The slope nuisance + per-leg anchor
have the right magnitude to absorb it.**

**Recommendation: option (B-production), as the meta-review and onboarding R1 converge on.**
g_fixed,c(z) = w_c-shape derived from the observed dN/dX power-law × X̄(z) through `w_c_from_mu`
(NOT the sim's own w_c). Then the closure forward is byte-identical to the real fit, the held-out
sim truth (slope +2.60) differs from g_fixed (slope +3.20) by −0.60, and the slope nuisance must
ride at s_LLS ≈ −0.60 to recover it. Certify coverage with s_c MARGINALIZED. The sim-w_c-fixed
"+0.03σ EXACT" case is a NECESSARY sampler sanity check, NOT sufficient — it certifies a shape
production never runs.

---

## Q2 — Where does s_c center? **WLS lit (0.76/0.05) over the code values (0.95/0.15) — BUT the center is conditional on the chosen g_fixed, and that interaction is not yet clean in the spec.**

Two things are conflated and must be separated:

1. **The honest dN/dX ratio-slope center.** The code constants (0.95, 0.15, 0.40 — `inference.py:74`)
   are an UNWEIGHTED `np.polyfit` of log dN/dX vs log(1+z) (`plot_dndx_vs_literature.py:102-108`) +
   a hand-down-weighted DLA (raw fit +1.08 → forced 0.40, `inference.py:71-73`). The WLS values
   (0.76, 0.05) re-fit with the quoted error bars. **The WLS is the physically honest center** —
   the literature dN/dX points have very unequal errors (LLS: ±0.05 at z=2.4 vs ±0.19 at z=4.23;
   `plot_dndx_vs_literature.py:36-37`), and an unweighted fit lets the noisy high-z points drag the
   slope. The z=2.4–2.8 high-precision LLS points anchor a shallower slope than the unweighted +0.95.

2. **But "center of s_c" depends on what g_fixed already absorbs.** This is the spec's internal
   inconsistency (see Errors below). If g_fixed = lit-shape (the recommended Q1 production framing,
   slope +3.20), then s_c centers near **0** by construction (g_fixed already encodes the full lit
   z-shape; s_c is only the residual lit-fit uncertainty). The 0.76/0.05 numbers are the
   sim-vs-lit *gap*, which is what the closure-truth offset is, NOT where the prior centers in the
   production framing. The spec's §4 last sentence ("CENTER then sits at the lit-vs-(forward-g_fixed)
   differential — likely ~0 if g_fixed already encodes the lit shape") is the correct statement, and
   it CONTRADICTS the sweep, which centers the prior at s=0 but injects the bias offset as if the
   center were the sim slope.

**Verdict:** with the recommended production g_fixed, **s_c centers at 0** (the lit shape is in
g_fixed; s_c carries only the lit-fit slope uncertainty, width 0.52/0.53/0.33). Keep it a FIXED
constant at 0, NOT a hyper-prior — there is no second-level information to justify a hyperprior,
and a hyper-prior on a slope that the sweep shows "barely competes with A_p" only adds funnel risk
([BAY]/[CS] territory). If instead the PI keeps g_fixed = sim w_c (the closure-only framing), the
center must be the WLS lit gap 0.76/0.05 — but then the closure is NOT production-faithful and Q1
is violated. **The two choices are coupled: pick production g_fixed → center s_c at 0; pick sim
g_fixed → center at WLS 0.76/0.05 and accept the closure is a sampler-only check.**

**Does the z=2.5–3.5 trust window change the center?** Yes, slightly, and the spec only applies
it to the WIDTH (the z-edge inflation, §2). Physically the window should also down-weight the
center: the LLS dN/dX slope is purity-limited below z=2.5 (LLS blends into the forest) and
completeness-limited above z=3.5 (the +0.78 point at z=4.23 with ±0.19 is the noisy one). A
window-restricted WLS LLS slope would be SHALLOWER than the full-window 0.76 (the steep high-z
points get the edge-inflation down-weight). **Recommendation: report the in-window (z=2.5–3.5) WLS
center alongside the full-window one; if they differ by >0.2 the center is being set by the
untrusted edges.** This is cheap (it's a re-fit of the same `LIT` arrays).

---

## Q3 — Is the +0.167σ Fisher bias trustworthy enough to SET the prior? **NO — recompute the exact non-linearized construction bias on the production g_fixed first. Required, cheap.**

Two independent reasons, one physics one methodological:

1. **The linearization is 13% (DESI) / 15% (KS) off** (`legb_slope_prior_tradeoff.txt:7-8`,
   `||J_s·gap − (P(gap)−P(0))||/||...|| = 0.129 / 0.152`). The bias offset (slope step 0.76 LLS /
   1.16 DLA) is LARGE, and (1+z)^s is convex over it, so J_s·gap underestimates the true ΔP on the
   far-z bins where (1+z)^s curves up. 13–15% on a 0.167σ headline that sits under a 0.2σ gate is
   NOT a comfortable margin — the exact bias could be 0.19σ, AT the gate.

2. **The bias is computed against the WRONG g_fixed.** The sweep sets g_fixed = sim w_c(z)
   (`diag_legb_slope_prior_tradeoff.py:90`, `g_fixed = wc_perz[sel_sim]/wc_pivot`) — the
   closure-only framing — and then injects `SIM_LIT_GAP=[0.76,0.05,1.16]` as the truth-slope offset.
   But if g_fixed IS the sim's own w_c, the sim truth is recovered at s=0 EXACTLY (that is literally
   the +0.03σ case), so there is NO construction bias to absorb — the +0.167σ is the bias you'd get
   IF the prior centered at s=0 but the truth slope were offset by 0.76, which is the PRODUCTION
   scenario (lit-centered prior, sim truth), not the closure scenario the same script's g_fixed
   encodes. **The variance test and the bias test use two different g_fixed framings.** The honest
   number is: build g_fixed from the LIT dN/dX shape (slope +3.20 LLS), set the truth to the sim
   (slope +2.60), and compute the EXACT ΔP = P(s=s_truth) − P(s=0) where s_truth ≈ −0.60 (LLS) is
   the residual the prior must center on / absorb. That is the construction bias the real fit
   inherits, and it must be recomputed non-linearly.

**Recommendation:** before committing the width, run the forward-only exact bias on the production
g_fixed: P_obs with g_fixed=lit-shape at s=0 (prior center) vs the held-out-sim truth, whitened by
C_total, projected onto the A_p Fisher direction. This is the same machinery as
`diag_legb_zresolved_alpha_check.py` (which already does the whitened-Δ projection) + one extra α(z)
construction — ~minutes, no NUTS. If the exact bias stays < 0.2σ, the width is confirmed; if not,
widen (the sweep shows free headroom — at m=2 the bias halves to ~0.08σ for +0.0015 in σ(A_p)).

---

## Q4 — Per-leg anchor float on AMPLITUDE or also SLOPE? **AMPLITUDE only. The slope float is physically wrong AND unnecessary.**

Physics view: the per-leg float δ_leg exists because **KS-SQUAD and DESI sample different absorber
populations** — KS is high-S/N, absorber-targeted (SQUAD = high-z quasar spectra selected partly for
absorption studies), so its LLS/subDLA *incidence* (the amplitude) is plausibly biased high vs the
large-area DESI dN/dX. That is an **amplitude** effect: a different number density of absorbers per
sightline at a given z. It is NOT a **z-shape** effect: the dN/dX *slope* with redshift is a property
of the absorber population's cosmological evolution (self-shielding + cosmic abundance), which is the
SAME physics in both surveys' lines of sight — KS does not evolve its absorbers differently with z,
it just has more of them. So a per-leg slope float would be modeling a z-evolution difference that
has no physical mechanism.

Methodological view confirms it: the sweep shows the slope "barely competes with A_p" (σ(A_p)
0.390→0.405 marginalizing the GLOBAL slope; per-leg slopes would be even weaker, split across legs)
and the bias is LLS-AMPLITUDE-driven (`legb_slope_prior_tradeoff.txt:14`, A_p-bias-by-class LLS
+0.119 dominated by the amplitude block via the slope coupling). A per-leg slope float adds 3–6
near-degenerate parameters into the exact A_p-degenerate low-k mode → funnel risk ([CS]/[BAY]) for
zero physics gain. **Amplitude float is the right and sufficient lever; confirm via Q4 as specced.**

**Is σ_anchor 0.25–0.30 (KS) physically defensible?** Borderline-generous but acceptable as a
conservative prior. The KS LLS/subDLA incidence-excess from absorber-targeted selection is real but
hard to quantify; exp(0.30) ≈ 1.35, i.e. allowing KS incidence up to ~35% above the DESI dN/dX
center. That is the right order for a selection-function excess (the LLS definitional offset below is
of similar size). DESI 0.10–0.15 (≈10–16%) is appropriate for the large-area dN/dX measurement
uncertainty. **Defensible — but the spec correctly flags these are expectations, not measured
(§3).** One caution: with KS effectively capped at k<0.06 (memory `phase2-analysis-scope-dla`) and
the KS loader currently near-redundant with DESI (onboarding R4), the KS anchor float barely moves
the result today — it is insurance for when the HR/MF emulator reaches KS's unique small scales.

**The LLS definitional offset — should it ride δ_leg or the amplitude center?** The sim LLS floor
is log N_HI≥17.2 (τ≥1) while the literature LLS is τ≥2 / N≥17.5, so PRIYA's LLS is EXPECTED ~25–40%
higher (`plot_dndx_vs_literature.py:11-13`). This is a **global sim-vs-lit definitional offset, NOT
a per-leg one** — it applies identically to both legs (it is a property of the sim's class
definition vs the literature's, not of the survey). **It belongs in the amplitude CENTER**
(`HCD_LIT_OVER_SIM[0]=1.06`, `inference.py:70`, which already partly absorbs it — the 0.98 median
ratio in `plot_dndx_vs_literature` reflects the sim being close after this offset is folded in), NOT
in δ_leg. Putting it in δ_leg would double-count it across legs and let it masquerade as a survey
difference. **Recommendation: keep the LLS definitional offset in the amplitude center; reserve
δ_leg for the genuine per-survey selection excess.**

---

## Errors or risks in the LOCKED parts

**E1 [REAL — must fix before committing the width] The sweep mixes two g_fixed framings, so the
LOCKED bias headline (+0.167σ) is not the bias the production config will have.** The variance
columns use g_fixed = sim w_c (closure-only, `diag_legb_slope_prior_tradeoff.py:90`); the bias
columns inject `SIM_LIT_GAP=[0.76,0.05,1.16]` as if the prior centered at s=0 against a
sim-slope-offset truth (the production scenario). These are inconsistent: under the script's actual
g_fixed=sim-w_c, s=0 recovers the truth exactly (+0.03σ), so the +0.167σ is a hypothetical, not the
construction's bias. The CORRECT computation (production g_fixed=lit-shape, sim truth, exact ΔP) is
Q3's recompute. **This does not invalidate the WIDTH conclusion (×1.04 is real and framing-robust —
it is a variance-only statement), but it does mean the bias number that "keeps bias < gate" is
provisional.** Re-derive on the production g_fixed.

**E2 [REAL — spec self-inconsistency on DLA] §2 says DLA width 0.33 and "DLA MOOT (§0c masks
α_DLA→0)", but the sweep's `SIM_LIT_GAP[2]=1.16` (DLA) is the LARGEST gap and the bias table shows a
persistent DLA contribution (+0.062σ, `legb_slope_prior_tradeoff.txt:14`, the third per-class
number).** The sweep is NOT masking the DLA — it carries a +0.062σ A_p bias from the DLA slope at
every width. If §0c truly zeros α_DLA (Gaussian-at-0 + DLA-masked mock truth), that +0.062σ should
VANISH from the sweep, and the total bias drops from +0.167σ to ~+0.105σ (LLS+subDLA only). **The
sweep was run WITHOUT the §0c DLA mask, so it OVER-states the bias.** This is reassuring (the real
bias is lower), but it means the spec's "DLA MOOT" and the sweep's DLA contribution disagree —
confirm by re-running the sweep with the §0c-masked truth, or at least subtract the DLA column when
quoting the gated bias. (See §0c coupling below — masking DOES remove the DLA slope from mattering,
so the spec's "moot" is physically correct; the sweep just predates the mask.)

**E3 [confirm, not error] The DLA raw dN/dX slope is +1.08 (`plot_dndx_vs_literature.py` live output:
`DLA ratio_slope=+1.08`), and `SIM_LIT_GAP[2]=1.16` ≈ that, NOT the code's weakened 0.40.** So the
sweep correctly uses the raw DLA slope as the gap (honest), while the production `inference.py:74`
uses the weakened 0.40. These serve different purposes (gap-to-survive vs prior-center) and are
consistent with the documented "weaken DLA because z>3.5 unreliable" reasoning — but since §0c masks
DLA, neither matters for the cosmology. No action beyond E2.

Otherwise: the LOCKED **width = literature dN/dX-slope 1σ (0.52/0.53/0.33)** is faithful — I
reproduced the underlying γ_lit−γ_sim slopes (LLS +0.95, subDLA +0.15, DLA +1.08) live, and the WLS
widths are the honest 1σ on those. The **×1.04 variance cost** is framing-robust (it is a marginal
σ(A_p) statement independent of where the truth sits). The **z-edge inflation** (β=1, mean ×1.25) is
physically motivated (purity below z=2.5, completeness above z=3.5) and correctly reproduces the
existing DLA `dla_inflate` slope (`inference.py:104`). **LLS as the binding class is correct** —
tightest amp prior (0.15, `inference.py:58`) + biggest gap (+0.95).

---

## Additional findings from this lens

**A1 [§0c coupling — confirmed, masking DOES make the DLA slope moot].** The spec defers the DLA
mask to its own step but asserts the DLA slope is "moot." I confirm this physically: §0c (plan
lines 50-58) sets α_DLA residual ≈ 0 (KS ~0% — damping wings trivially masked; DESI ≥90% masked,
missed ~15% misclassified as subDLA → small subDLA residual, not DLA). With α_DLA pinned at ~0 by a
Gaussian-at-0 prior + DLA-masked mock truth, the DLA slope multiplies a ~0 amplitude → its
contribution to P_obs and to ∂P/∂s_DLA → 0. So the DLA slope cannot bias cosmology once §0c lands.
**The dependency chain is: §0c must land BEFORE the slope-prior width is finalized**, because (a) the
sweep's +0.062σ DLA bias (E2) only disappears under the mask, and (b) `make_truth_from_sim`
currently builds the truth with the FULL sim w_c INCLUDING DLA (`closure_legb.py:273-278`,
`a = w_c[r,1:]` is all 3 classes) — so today's closure mock carries a DLA contamination the data
does not have. This is onboarding R2/R3, and it directly couples to this spec. **Recommendation:
sequence §0c before the slope cert; re-run the sweep DLA-masked to confirm the gated bias.**

**A2 [g_fixed needs X̄(z), and the spec does not say where X̄(z) comes from in production].** My Q1
construction needed X̄(z) (the mean absorption path per sightline) to map lit dN/dX → μ → w_c. In the
sim cache this is `snap_total_path_dX / n_skewers` (`data.py:121`, `cddf.py:154-157`). In production
the lit-derived g_fixed must use a consistent X̄(z) — either the sim's (defensible: X̄ is geometry,
θ-/incidence-weakly-dependent, slope +2.02 dominated by the (1+z) line-element) or computed from the
survey's path-length. **The spec's §1 g_fixed bullet does not specify the X̄(z) source.** Since X̄(z)
is the SAME +2.02 geometric growth in sim and data (it is dz→dX cosmology, near-identical inside the
PRIYA box prior), using the sim's X̄(z) is faithful — but DOCUMENT it, because if a future reader
builds g_fixed from dN/dX alone (slope ~+1.2 incidence-only) instead of dN/dX×X̄ (slope +3.20), the
g_fixed would be 2σ too shallow and the −0.24σ low-k pull reappears. **This is exactly the original
bug in a new disguise.** The production g_fixed builder must go through `w_c_from_mu(dN/dX·X̄)`, not
dN/dX directly.

**A3 [the slope nuisance is a RATIO-slope correction, and that must be enforced in the model].** In
the production framing, g_fixed carries slope +3.20 and s_c carries the residual lit-fit uncertainty
(center 0, width 0.52). The current `_legb_model` (`closure_legb.py:453`) builds
`shape_zg = ((1+z)/(1+z_p))^HCD_LIT_OVER_SIM_SLOPE` — i.e. it uses the lit RATIO-slope (0.95) as the
FULL shape, with NO g_fixed factor at all. That is the −0.24σ partial-fix code. The spec's α(z) =
α_pivot·exp(δ)·g_fixed·(1+z)^s_c is the correct replacement, but the implementation MUST multiply in
g_fixed (slope +3.20) AND sample s_c around 0 — NOT reuse the existing `(1+z)^HCD_LIT_OVER_SIM_SLOPE`
as g_fixed (that would double-apply the lit slope and over-steepen to +4.15). The golden guard
(onboarding R2) must freeze the legacy path before this refactor.

**A4 [subDLA gap is genuinely ~0, so the subDLA slope is essentially data-set, as the spec says].**
Live: subDLA sim w_c slope +2.72 vs lit-derived +2.75, GAP −0.03 — and the sweep's subDLA A_p bias
is +0.004σ (`legb_slope_prior_tradeoff.txt:14`). The spec's "subDLA essentially data-set, gap ~0" is
correct. The subDLA width 0.53 is broad (the Zafar-vs-O'Meara factor-2 measurement disagreement,
`inference.py:59`) — fine to keep broad; it costs nothing (the gap is ~0 so a broad prior adds no
bias and the sweep shows σ(A_p) flat in the subDLA width).

**A5 [the closure fiducial is ONE sim — the gap is sim-dependent].** The sweep uses
`ns0.803Ap2.2e-09herei4.05h` (`legb_slope_prior_tradeoff.txt:3`). The sim-vs-lit w_c gap (−0.60 LLS)
is a property of THIS sim's measured w_c(z). Different held-out sims have different w_c(z) (the
incidence is θ-weakly-dependent but not zero). **The coverage cert must run over multiple held-out
sims (the N≥99 Leg-B coverage), not just this fiducial** — otherwise the slope prior is tuned to one
sim's gap. The spec's width is set on one fiducial; confirm the gap distribution across the held-out
folds brackets within the locked width (cheap: compute the LLS w_c slope per held-out sim).

---

## Concrete recommendations (numbered, actionable)

1. **Adopt the production-faithful g_fixed (Q1=YES).** Build g_fixed,c(z) from the observed dN/dX
   power-law × X̄(z) through `dndx_wc.w_c_from_mu`, NOT the sim's w_c. I verified it carries both
   incidence and path-length growth (slope +3.20 LLS, of which +2.02 is X̄). Document the X̄(z) source
   (use the sim's `snap_total_path_dX/n_skewers`; it is θ-/survey-near-invariant geometry) — and warn
   future readers that g_fixed MUST go through w_c_from_mu(dN/dX·X̄), never dN/dX alone (A2: the
   incidence-only shape is +1.2, the original bug in disguise).

2. **Re-center s_c at 0 with the production g_fixed (Q2).** With g_fixed = lit-shape, s_c carries
   only the residual lit-fit slope uncertainty → center 0, width 0.52/0.53/0.33, FIXED constant (no
   hyper-prior — adds funnel risk for no information). If the PI instead keeps g_fixed = sim w_c
   (closure-only), center s_c at the WLS lit gap 0.76/0.05 and explicitly mark the closure as a
   sampler-only check, not production-faithful. Pick ONE framing and make the center match it.

3. **Recompute the EXACT (non-linearized) construction bias on the production g_fixed BEFORE
   committing (Q3, E1).** g_fixed=lit-shape at s=0 vs the held-out-sim truth, exact ΔP whitened and
   projected onto the A_p direction — reuse `diag_legb_zresolved_alpha_check.py`'s whitened-Δ
   machinery + one α(z) construction (~minutes). The current +0.167σ is (a) 13–15% linearization-off
   and (b) computed against the wrong g_fixed. Confirm < 0.2σ exact; if not, widen (free headroom per
   the sweep).

4. **Re-run the slope sweep with the §0c DLA mask applied (E2, A1).** The current sweep carries a
   +0.062σ DLA-slope bias that the mask removes; the gated bias drops to ~+0.105σ (LLS+subDLA). This
   also fixes `make_truth_from_sim` (`closure_legb.py:273-278`) building a full-w_c (un-masked) truth.
   Sequence §0c BEFORE finalizing the width.

5. **Anchor float on AMPLITUDE only (Q4=confirm as specced).** No per-leg slope float — there is no
   physical mechanism for a per-survey z-evolution difference, and it adds funnel risk into the
   A_p-degenerate low-k mode. Keep the LLS definitional offset (sim τ≥1 vs lit τ≥2) in the amplitude
   CENTER (`HCD_LIT_OVER_SIM[0]`), NOT in δ_leg (it is global, not per-survey).

6. **Report the in-window (z=2.5–3.5) WLS center alongside the full-window one (Q2).** If they differ
   by >0.2, the full-window LLS center 0.76 is being set by the untrusted high-z edge (the ±0.19
   z=4.23 point). Cheap re-fit of the same `LIT` arrays.

7. **Certify coverage with s_c MARGINALIZED over multiple held-out sims (A5), gated on A_p coverage
   ≥ nominal + |bias| < gate** — NOT the single-fiducial +0.03σ exact case (necessary, not
   sufficient). The +0.03σ sim-w_c-fixed run remains a useful sampler sanity check; keep it as a
   diagnostic, not the verdict.

8. **In `_legb_model`, multiply in g_fixed and sample s_c around 0 (A3).** Do NOT reuse the existing
   `(1+z)^HCD_LIT_OVER_SIM_SLOPE` as g_fixed (double-counts the lit slope → +4.15). Land the golden
   guard (onboarding R2) before this refactor.
