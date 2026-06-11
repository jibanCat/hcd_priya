# HCD referee audit — "filtered subDLA/DLA but correct dN/dX?" + the fixes

*2026-06-11. The PI asked: "How come subDLA and DLA are filtered but the dN/dX of subDLA and DLA
are correct?" and to spawn referee agents on the HCD part. Three adversarial referees (Lyα-physics,
code-trace, Bayesian/PPL) audited the HCD contamination model. This is the meta-report + the fixes.*

---

## The question, answered

**dN/dX (a count) and the per-class P1D template (a power) are different observables, and the code
keeps them consistent.** The τ=1e⁶ filter rewrites only *flux / optical-depth* pixels (it trough-fills
saturated cores, Chabanier-2019 algorithm) — it never touches the absorber catalog. The cache builder
**asserts** `n_by_bin == n_by_bin_filt` (counts identical, filtered vs unfiltered;
`build_emulator_cache_tau0.py:328`). The emulator's dndx head targets that full, unfiltered count, so
"correct dN/dX + filtered power" is not a contradiction — they come from different sources.

The filter's **in-band P1D power** impact, by class (verified directly):

| class | in-band filtered-power deficit `δ/P_filt` | add-back wired? |
|---|---|---|
| LLS | **0.0%** (below the filter floor) | n/a |
| subDLA | **0.8%** | no — and negligible |
| DLA | **3.1%** (≈13.5% incl. saturated cores) | **yes**, `dla_core = delta[:,2]` |

So **only DLA is materially filtered, and DLA is the class that gets the unfiltered add-back.** The
subDLA add-back is designed but unimplemented (`delta[:,1]` is cached, never consumed) — but at 0.8%
in the DESI band it is **not load-bearing** (deferred; KS reaches lower k, check there before the KS fit).

## Referee meta-verdict

| lens | dN/dX↔power consistent? | issues raised |
|---|---|---|
| Lyα-physics | ✅ consistent (closure self-consistent; forward↔data subDLA gap is <0.2% in-band) | subDLA add-back unimplemented (Important, but small) |
| code-trace | ✅ A–E confirmed; counts asserted identical; only DLA add-back exists | subDLA `delta[:,1]` never consumed |
| Bayesian/PPL | ✅ `P_tier_p ≡ Σ w_c·P_filt` to 1e-16; α=w_c reproduces the sim's own contaminated P1D | **prior-center** mis-match drives "subDLA recovered low"; **KS α support**; **linearity at the boost** |

**The closure is valid.** The forward and mock both use the same filtered subDLA, so the filter-removed
power cancels in the closure residual. The genuine to-dos are **prior-side**, and the referees corrected
my initial "subDLA power deficit" framing: the subDLA excess is *negative* (adding subDLA *lowers*
small-scale power), so there is no "inflate α to add missing power" channel — the bias is prior-centering.

## Fixes applied (PI-selected 2026-06-11)

1. **α_LLS / α_subDLA bounded ≥ 0** (`closure_legb._legb_model` + `_legb_priors_only`):
   `Normal → TruncatedNormal(low=0)`, matching the DLA softplus. A plain Normal at the **KS-boosted**
   LLS center (`N(0.727, 0.291)`) put ~6% mass at α<0 and a tail at Σα>1 → a *negative clean fraction*
   (unphysical P_obs). TruncatedNormal keeps the (μ,σ) interpretation and removes that tail. DESI is
   unaffected in practice (its α<0 mass was ~1e-11/0.5%); the change is for consistency + the KS leg.

2. **subDLA prior center 0.76 → 1.00** (`inference.HCD_LIT_OVER_SIM[1]`): the old Zafar+2013 0.76 sat
   −0.8σ below the sim subDLA incidence → pulled α_subDLA low and leaked into n_s (corr≈+0.3). The
   literature is factor-2 uncertain (Zafar vs O'Meara; Berg+2019 revises) and **PRIYA produces subDLAs
   in-situ** (Rahmati+2013 self-shielding) → center on the sim with the broad σ/μ=0.40 marginalizing the
   residual abundance uncertainty, rather than imposing an offset literature center.

3. **Linearity / template-stability check** (`scripts/diag_lls_template_linearity.py`,
   `figures/analysis/05_likelihood/lls_template_linearity.png`): the forward is exactly linear in α, so
   the question is whether the filtered LLS excess *template shape* is incidence-independent up to the KS
   ~2.5× regime. **It is — shape corr(low, high w_LLS) = 0.97.** The excess *amplitude* shows a mild
   anti-correlation with abundance (r = −0.42, partly cosmology-confounded across the 60
   single-cosmology sims), which the **broad KS σ=0.40 absorbs** — confirming "high center, broad σ" was
   the right call. The ×2.5 KS boost is the correct lever (scale incidence α, not template power) and
   does **not** double-count `HCD_LIT_OVER_SIM` (cosmic ratio × selection multiplier are distinct).

![LLS template linearity](../../figures/analysis/05_likelihood/lls_template_linearity.png)

## Deferred (referee-rated low)

- **subDLA add-back** (`delta[:,1]`): only ~0.8% in the DESI band → deferred; revisit for the KS leg
  (lower k) before the KS real fit.

## Provenance

Referee agents: Lyα-physics `ab48622695d69d440`, code-trace `a905a2e9754bf3e10`, Bayesian `a9fbbbfbd52052895`
(resumable). Companion: the per-survey LLS pin + the box restriction
([`2026-06-11-reflection-igm-prior-restriction.md`](2026-06-11-reflection-igm-prior-restriction.md)).
