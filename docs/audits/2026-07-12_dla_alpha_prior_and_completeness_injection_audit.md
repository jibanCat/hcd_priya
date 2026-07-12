# DLA alpha_dla prior + completeness-injection deep audit (SYNTHESIS, authoritative)

Date: 2026-07-12. Repo: /home/mfho/hcd_priya @ f9d157000e29b9b52038aab448a70577dc4e1474
(branch gate-b-dla-desi-rerun). PI-commissioned pre-freeze audit; 5 independent evidence roles
+ 1 adversarial referee + this synthesis. Blind-safe throughout: only mock differentials,
templates, whitened projections, prior/mock-truth quantities. No real-data posteriors touched.

Evidence tags: MEASURED (recomputed by at least one agent this session; load-bearing numbers
additionally re-reproduced by the synthesis auditor from the saved deployed arrays and pkls),
RECORD (verbatim design record), INFERRED, ASSUMED. Input reports and re-run commands are in
the scratchpad dla_deep_audit directory (commands.md collates every command verbatim).

---

## 1. Executive verdict

1. **Is the alpha_dla prior correct?** YES, as designed and as documented. Deployed (legacy
   branch): alpha_dla = softplus(raw), raw ~ Normal(softplus^-1(0.00441) = -5.4218, 1.0)
   (closure_legb.py:1993-2000). No Jacobian bug (sample site is raw, alpha is a numpyro
   deterministic), no index swap (full chain verified), no railing (posterior floor mass <= 6.7%,
   prior-to-posterior contraction ~ 0), truth drawn from the identical distribution (self-draw
   consistent). One labeling hazard: the deployed width is the hardcoded latent sigma = 1.0
   (effective sigma/mu ~ 1.28), NOT the named HCD_PRIOR_FRAC_SIGMA[2] = 0.50; the z > 3.5
   dla_inflate is inert on this branch. Deliberate and mirrored in the truth draw (RECORD,
   commit 686b046, in-code note :1991-1997), but any text quoting "0.50" for the deployed DLA
   prior is wrong by ~2.5x. Confidence: high.

2. **Is the forward implementation correct?** YES. P_obs = P_clean + sum_c alpha_c (P_c - P_clean)
   with P_DLA^unf = P_filt[DLA] + dla_core (predict.py:82-116, data_likelihood.py:657-663);
   sign, index order, z-resolution, per-leg dla_forward_frac (DESI 1.0), and the NORC stamp all
   verified at file:line; mu exactly linear in alpha_dla; symmetric finite differences match
   jax.jacfwd to <= 9e-12 relative. Three documented conventions (z-mean per-leg core, core not
   MF-corrected, per-row kfkms averaged on a common grid) cancel exactly in this Leg-A gate and
   remain real-fit caveats (arm A3), not bugs. Confidence: high.

3. **Is the injection correct?** YES, mechanically. e_dla = the DESI-shipped
   syst_e_dla_completeness column (strictly positive, P units km/s, quadrature-verified against
   the shipped cov_syst), mapped EXACTLY onto the 681 leg rows (681/681 exact, dz = dk = 0),
   added additively once (run_dla_completeness_shard.py:40). The referee's likelihood-level
   identity diff(+1) + diff(-1) = -e^T C^-1 e holds per mock (measured -2.58 to -3.11 vs
   predicted -2.967; synthesis re-reproduced all 8 values), which excludes double-apply, missing
   injection, halving, and arm mislabels at the level the likelihood actually consumed. Pairing
   exact (clean truths bit-identical across arms, seed 20260615, 0 divergences). ONE severity
   caveat, not a bug: the column is an UNSIGNED per-(z,k) envelope and DESI's own cov_syst is
   exactly z-block-diagonal, so the z-COHERENT +/-1 injection is jointly a chi2 = 10.7
   (~3.3 sigma) excursion under DESI's stated systematic model (1.72 sigma whitened under the
   deployed C_total). The "+/-1 sigma" label is per-z true, joint-z overstated; the PI must
   adjudicate which convention the ledger row is denominated in. Confidence: high (mechanics),
   PI-adjudication pending (denomination).

4. **Is the tau = 1e6 + DLA-core restoration correct?** YES for this gate. The tau = 1e6
   trough-fill machinery (masking.py:419,463; p1d.py:366; priya_p1d.py:96,162) executes only at
   cache-build time; no tau = 1e6 code runs at fit time. The forward's core add-back
   P_filt[3] + delta[:,2] was previously verified bit-identical to the unfiltered coarse DLA
   power (RECORD 2026-06-04). This campaign is a Leg-A SELF-DRAW (run_dla_completeness_shard.py:106):
   mock and fit are built by the SAME predict_P_obs_on_leg with the SAME z-mean core
   (closure_legb.py:2627-2632), so core errors cancel exactly; the sim-truth add-back path
   (restored once, no double count, closure_legb.py:1089-1095,1336) is not even exercised.
   Real-fit core fidelity (z-mean MVP, un-MF-corrected core) stays open as arm A3. Confidence: high.

5. **Does the Gate-B fail remain scientifically valid?** YES, valid AFTER reinterpretation of
   its mechanism. The headline numbers stand (n_s ~ 0.21-0.28 sigma, A_p ~ 0.21-0.24 sigma per
   arm; tau0_amp ~ 0.27-0.43; f_res null), independently recomputed three ways. But the quoted
   mechanism "alpha_dla ANTI-absorbs (cos(e_dla, R_DLA) ~ -0.53, wrong projection sign)" is
   REFUTED and must be retired: the referee REPRODUCED -0.538 to two decimals as a NO-CORE
   template in a DIAG-whitened metric, i.e. a diagnostic-construction error (wrong template AND
   wrong metric), while in the deployed C_total^-1 metric the cosine is +0.323 (correct sign)
   and the measured posterior alpha_dla moves WITH the injection in both arms (+0.62 / -0.36
   sigma_clean). The true mechanism, demonstrated numerically: alpha_dla is a correct-sign but
   WEAK lever (cos^2 = 10%; its positive limb is only the 2 lowest kept k bins, and the required
   per-z coefficient falls ~6x by z = 3 and flips sign at z >= 4 against the rigid rising
   (1+z)^2.366 slope prior); under the designed HCD-sector priors ~58% of the whitened mode is
   stranded and relocates onto uniform-prior tau0_amp, dragging n_s/A_p along the amplitude
   degeneracy. With flat HCD priors the mode WOULD be ~92% absorbed, but only at physically
   absurd incidence excursions (alpha_dla 3.1x fiducial plus wrong-signed alpha_lls), which the
   priors correctly forbid. So the FAIL = the model honestly refusing to launder a
   completeness-shaped mode through incidence dials: physics + parameterization + prior
   geometry jointly; NOT code, NOT injection construction. Confidence: high.

6. **Are expensive reruns justified yet?** NO. The existing +/-1 campaign is implementation-clean
   and its numbers are reusable; nothing found here invalidates them. Reruns become justified
   only AFTER two PI decisions: (i) the severity denomination of "+/-1 sigma" (a z-block-scaled
   or z-incoherent arm would be a NEW test, not a repair), and (ii) whether a completeness-shaped
   response term (floated e_DLA nuisance, or a z-resolved DLA amplitude) enters the model, which
   is a design question requiring a design pair + small mock test before any campaign. The eBOSS
   arm should not be built before (i) is decided (Section 12). Confidence: high.

---

## 2. Provenance

Commit: f9d157000e29b9b52038aab448a70577dc4e1474 (gate-b-dla-desi-rerun). Working tree at audit
time: 2 modified tracked files (both figures/), untracked scripts/figures/checkpoints only; no
modified tracked source (verified by the prior auditor and re-checked at synthesis).

Campaign config (MEASURED from pkl meta, all 16 shards): meta.forward = {res_corr_on: False,
fix_alpha_res: True, sample_res: True, f_res_amp_sigma: 0.02, metal_prior: flatlog2node,
metal_node_z: (2.2, 4.2)}; seed 20260615; strength stamped -1.0 / +1.0; survey DESI;
hierarchical_hcd = False (legacy alpha_dla branch; run_dnuis_bias_shard.py:209);
use_prod_forward = True with runtime parity asserts (run_dla_completeness_shard.py:87-98);
0 divergences in all 32 fits (recomputed, not quoted).

| Artifact | Path | Stamp | Produced by | Consumed by | Verified? |
|---|---|---|---|---|---|
| DESI DR1 P1D npz | /home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz | none (derived file) | scripts/convert_desi_dr1_p1d.py from DESI QMLE FITS (Karacayli 2025, Zenodo 16943723) | build_e_dla, leg loader | YES: column copied verbatim; quadrature identities e_syst^2 = sum e_t^2, diag(cov_syst) = e_syst^2 exact; FITS TUNIT km/s from raw header |
| Injected arm +1 | /scratch/cavestru_root/cavestru0/mfho/dla_deployed_p1/ (8 pkls) | NORC forward stamp + strength +1.0 + seed | run_dla_completeness_shard.py | analyze_dnuis_bias.py, this audit | YES: ll identity, pairing, pulls recomputed |
| Injected arm -1 | /scratch/cavestru_root/cavestru0/mfho/dla_deployed_m1/ (8 pkls) | NORC stamp + strength -1.0 + seed | same | same | YES: same checks |
| STALE thin-forward run | /scratch/cavestru_root/cavestru1/mfho/dla_completeness | pre-NORC | superseded runner | none | FLAGGED: do not quote; superseded by dla_deployed_{m1,p1} |
| LF tau0 cache | hcd_analysis/_emulator_data/observables_tau0_lf.h5 | v3.3 | Phase-2 cache build | forward, template audits | YES (tier structure re-read; core identity RECORD 2026-06-04) |
| Golden forward npz | tests/golden/legb_mf_golden.npz | byte-identity ref | tests/test_norc_forward.py | parity asserts | YES; NOTE golden alpha3[2] = 0.0426 is a byte-identity reference, 9.7x the deployed prior center 0.0044; never use it as "the fiducial" |
| Deployed J, C_total, e_comp | scratchpad dla_deep_audit/dla_projection_results.npz | script-documented | numerical verifier reproducer (one ~350s deployed-forward chain) | referee T2-T4, synthesis re-check | YES: FD-vs-jacfwd 1e-12; synthesis reproduced cos/beta/GLS exactly |
| Prior/posterior recomputes | scratchpad prior_numbers.npz, posterior_recompute.npz, edla_vs_template.npz, pkl_arm_consistency.npz, fwd_projection.npz | script-documented | evidence agents | referee, synthesis | YES (cross-agent agreement + synthesis spot checks) |
| Earlier audit docs | hcd_priya_notes docs/superpowers/2026-07-12-dla-completeness-audit.md, dla-rerun-artifacts/audit-meta-report.md, 4lens-panel-verdict-wh13njw20.md | n/a | earlier panel | corrected by this audit | PARTIALLY: -0.53 reproduced as a defect (H8); per-z run -0.59 to -0.40 unreproducible, retired |

Ambiguous/unstamped: the earlier panel's -0.53 computation is not on disk (its construction is
INFERRED from an exact 2-decimal reproduction); the stale cavestru1 run carries no NORC stamp.

## 3. Intended mathematics

Per leg, per z, on the cache k grid (RECORD, verified against live code):

    P_model(k) = P_clean(k; theta, z, tau0)
               + alpha_LLS(z)    * [P_LLS^filt(k)  - P_clean(k)]
               + alpha_subDLA(z) * [P_sub^filt(k)  - P_clean(k)]
               + f_leg * alpha_DLA(z) * [P_DLA^unf(k) - P_clean(k)]

    P_DLA^unf = P_filt[DLA] + dla_core,   dla_core = cache delta[:,2] = P_DLA^unf - P_DLA^filt
    alpha_c(z) = alpha_pivot_c * ((1+z)/(1+3))^{s_c},  s_c ~ Normal(zslope_mu_c, sigma_s)
    f_leg = dla_forward_frac: DESI 1.0, KS 0.0, eBOSS 0.0

alpha_c is an ABSOLUTE effective post-masking per-class sightline-incidence weight (the w_c
object), NOT (alpha - 1) and NOT a shape-floated template multiplier; algebraically identical
to the mixture (1 - sum alpha) P_clean + sum alpha_c P_c. alpha_dla UP = MORE residual
unmasked-DLA contamination; alpha_dla = 0 = perfect masking. Prior center (DESI):
mu_DLA = HCD_DLA_RESIDUAL_FRAC(0.10) x lit/sim(1.34) x w_DLA(z=3)(0.03291) = 0.0044097,
encoded as softplus(Normal(-5.4218, 1.0)) (MC: median 0.00441, mean 0.00722, sd/mean 1.28,
quantiles 2.5/16/50/84/97.5% = 0.00062/0.00164/0.00441/0.01187/0.03100; P(alpha_dla > full
incidence w_DLA = 0.033) = 2.2%). Closure-mock design intent (RECORD 1e1b278, inference.py:58-74,
closure_legb.py:1080-1088): alpha_dla IS the DLA-finder completeness-residual dial, marginalized;
completeness DOWN => alpha_dla UP, with the response SHAPE fixed to the in-situ R_DLA.

Completeness injection: P_data(z,k) += strength x e_dla(z,k), strength = +/-1, e_dla = the
DESI-shipped syst_e_dla_completeness (unsigned 1-sigma envelope; the sign is a hand-chosen
coherent convention; the +/- pair is the test).

Sign conventions: residual r = P_data - P_model (closure_legb.py:1752-1755); k angular s/km;
P velocity power km/s. Key measured shapes: e_dla > 0 at all 1020 rows (min 1.47e-3), e/P median
0.117%, low-k weighted (e steeper than P); R_DLA bipolar, positive only at k <~ 0.002-0.003 s/km
(damping-wing limb, core-dominated), ~ -0.3 to -0.4 x P_clean broadband above, negative lobe
deepening with z. Both are Rogers-consistent in-situ physics (RECORD 2026-06-04), not cache bugs.

## 4. Executed code path

Trace (legacy branch, the one this campaign runs):

    closure_legb.py:1998  alpha_dla_raw ~ Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0)
    sampler_numpyro.py:37-40  _dla_raw_mu = softplus^-1 (log(expm1(clip(mu,1e-6))))
    closure_legb.py:2000  alpha_dla = deterministic(softplus(raw))          [one-sided > 0]
    closure_legb.py:2062-2070  s_c ~ Normal(zslope_mu, sigma)  [DESI zslope_mu = (2.127, 2.758, 2.366)]
    closure_legb.py:1903-1904  alpha_hcd_z[iz,c] = alpha_pivot_c * ((1+z)/4)^{s_c}
    closure_legb.py:1716-1721  per-leg z slice (nearest index); tau0_vec slice
    data_likelihood.py:1062-1066  dla_scale = [1,1,dla_forward_frac]  [DESI 1.0: no-op]
    predict.py:42-63  P_filt(4,K) = emulated per-class LINEAR power, tau0 an INPUT (cond[10])
    data_likelihood.py:616-660  MF class correction exp(g); NORC: res_corr absent, alpha_res=(1,0) pinned (closure_legb.py:1933-1934)
    predict.py:82-86  R = [P1-P0, P2-P0, (P3 + dla_core) - P0]   [core added raw, un-MF-corrected]
    data_likelihood.py:663  P_obs = P_clean + einsum("c,ck->k", alpha_leg[iz], R)
    data_likelihood.py:1106  interp to leg k (no extrapolation: leg k in [0.00125, 0.041] inside cache span)
    data_likelihood.py:1123-1140  x metal factor x resolution factor (HCD excess INSIDE the product)
    data_likelihood.py:1184-1219  C_total = C_data + diag(emu) + shape terms (alpha enters the emu coef)
    closure_legb.py:1752-1755  r = P_data - P_model; gaussian_loglik(r, C_total)

Mock and injection:

    closure_legb.py:1418-1444  truth = prior self-draw (identical sites via _legb_priors_only)
    closure_legb.py:1579-1655  mock = SAME predict_P_obs_on_leg at truth + N(0, C_total(truth))
    closure_legb.py:2627-2632  z-MEAN fiducial dla_core fed identically to mock and fit
    run_dla_completeness_shard.py:28-43  monkeypatch: P_data[fin] += strength * e_dla[fin]
    run_dla_completeness_shard.py:46-57  build_e_dla: nearest (z,k) map of the shipped column [measured exact]

Verified linear/derivative behavior: mu exactly linear in alpha_dla; FD vs jacfwd <= 9e-12
relative across h = 0.02 to 2e-4; deterministic sign test: +0.02 raises P at all 16
template-positive rows, lowers at all others; self-test injecting +1 x t_dla returns
beta = +1.000000, cos = +1.000000.

## 5. Prior audit

- Definition sites verified (closure_legb.py:1987-2000, sampler_numpyro.py:37-40,
  inference.py:58,70,83,133,217-251; ctx build closure_legb.py:700-711). alpha_lls/alpha_subdla:
  TruncatedNormal(mu, sigma, low=0) at (0.19383, 0.0291) / (0.06218, 0.0249). alpha_dla: the
  softplus(Normal) above.
- No Jacobian or double-density bug: exactly one likelihood factor; alpha_dla is deterministic;
  truth draw traces the identical sites; analyzer name-keyed and reproduced from raw draws to
  the third decimal.
- No railing: posterior mass below the prior 2.5% quantile <= 6.7% (typically <= 2%);
  raw-space posterior sd 0.84-1.15 vs prior 1.0, contraction ~ 0. alpha_dla is essentially
  data-UNCONSTRAINED on DESI; the marginal is close to the prior.
- Asymmetric support consequence (real, not a bug): support alpha > 0 bounds the maximum
  downward template contribution by the truth's own level; the -1 arm's LS-required move
  (~ -0.0039 to -0.0064 depending on span) is floor-compressed for low-truth mocks. Measured
  physical-space asymmetry +0.621 vs -0.357 sigma_clean; RAW-space arm sums symmetric within
  noise (+0.103 +/- 0.055), so the softplus convexity explains the physical-space asymmetry
  without any sign flip.
- Named-vs-deployed width: HCD_PRIOR_FRAC_SIGMA[2] = 0.50 is NOT deployed; latent sigma
  hardcoded 1.0 (effective sd/mean 1.28); dla_inflate inert. Deliberate, mirrored in the truth
  draw (any width change must be mirrored at closure_legb.py:1998, closure_mocks.py:56,
  sampler_numpyro.py); documentation must stop quoting 0.50 for the deployed DLA prior.
- Prior-geometry attribution (referee T3, synthesis-reproduced): flat priors on all 8 linearized
  directions absorb the mode to an 8.4% residual with near-zero cosmology response
  (alpha_dla + 0.0137 = 3.1x fiducial, alpha_lls wrong-signed -0.8 sigma_prior); deployed
  priors leave a 58.0% residual with ns -0.0281 / Ap -0.0313 (unit-box units). Removing
  alpha_dla entirely under deployed priors moves ns only to -0.0319: alpha_dla's OWN prior owns
  ~10-15% of the cosmology bias; the HCD SECTOR priors jointly own it. Widening alpha_dla alone
  buys little mean bias and converts it into cosmology variance; tightening pushes more onto
  tau0/cosmology.

## 6. Spectrum/template-construction audit (tau = 1e6 + DLA-core restoration)

- Cache tiers: P_filt = tau = 1e6 trough-filled per-class P1D (priya_p1d.py:27-44), classes by
  highest-NHI sightline (priya_p1d.py:110-122); delta = (P_unfiltered - P_filtered)[:, 1:]
  (data.py:164), so delta[:,2] = the DLA-class power the trough-fill removed (saturated cores +
  affected wing), i.e. exactly what undetected DLAs restore. tau = 1e6 constants:
  masking.py:418-419,463; p1d.py:366.
- Core restored ONCE in the sim-truth path (baseline holds the DLA class at clean,
  closure_legb.py:1091; the excess added once at TRUTH_DLA_FRAC 0.10, :1323,1336): no double
  count. That path is NOT exercised by this Leg-A campaign.
- Forward core: P_filt[3] + delta[:,2] previously verified bit-identical to the unfiltered
  coarse DLA power (RECORD 2026-06-04). In this gate the mock uses the identical z-mean core:
  exact cancellation.
- Real-fit (non-gate) caveats, all documented, none a bug here: (A4) z-MEAN core MVP flattens
  the z-structure of the deployed DLA response (closure_legb.py:1683-1687, 2631-2632; arm A3);
  (A5) core not MF-corrected (predict.py:83 vs data_likelihood.py:660); (A6) per-row kfkms
  (~4% within-z spread) averaged as if on a common grid (closure_legb.py:917).
- The injected mode most resembles the CORE alone: cos_wh(e, core_only) = +0.844 (stable across
  metrics). The PI's physical picture (the completeness envelope is tau = 1e6-removed
  saturated-DLA power that undetected DLAs restore) is supported in shape space, even though the
  injection is mechanically the DESI-shipped column, not a sim-built spectrum perturbation.

## 7. Injection audit

- Semantics: deterministic additive data-space mode, s x e added to every mock after noise;
  linear in s, no clipping/log; clean arm is the exact midpoint; noise cancels in the paired
  differential. The finite-row guard is inert (0 nonfinite rows).
- Ordering: injection sits OUTSIDE the metal/resolution product while the forward's HCD excess
  sits INSIDE; mismatch O(|mfac-1|, |resfac-1|) x 0.1-1.6% ~ 1e-5. Immaterial.
- Sign: the column is unsigned; the +/- bracket signs it by hand (correct treatment for an
  unsigned budget per the +/-1 sigma injection convention). "Anti-absorption of one arm" can
  never be more than a statement about the hand-chosen sign; the PAIR is the test.
- Linearity/pairing: ll identity diff(+1) + diff(-1) = -e^T C^-1 e verified per mock
  (-2.58 to -3.11 vs -2.967 predicted; theta-dependence of C explains the spread); clean fits
  bit-identical across arm directories; strengths stamped; single-variable strength wiring makes
  a stamp/injection sign inconsistency structurally impossible (:31,40,77,125).
- Mapping: exact 681/681, dz = dk = 0, byte-equal to the shipped column at those rows (verified
  independently by three agents plus an exact-equality lookup with no nearest metric).
- Severity convention (H10, load-bearing caveat): DESI cov_syst is exactly z-block-diagonal;
  within-block the injected e is ~ a 1-sigma direction (per-block Mahalanobis chi2 0.87-1.00),
  but the z-coherent joint excursion is chi2 = 10.7 (~3.3 sigma) under DESI's own systematic
  model, |e|_wh = 1.72 under the deployed C_total (1.83 under C_data alone). Deliberate
  worst-case convention matching the resolution arm's construction; the FAIL magnitude is
  conditional on it; the PI must adjudicate the denomination of the 0.50 waiver row (per stop
  rules, no ceiling recommendation here).
- Latent path split (inert): build_e_dla hardcodes the baseline npz while the leg loader can
  switch to SNR>3 via env; batch scripts set no env; snr3 e within ~3.5% and grids identical.

## 8. Numerical verification

All numbers below MEASURED; load-bearing ones re-reproduced at synthesis from
dla_projection_results.npz (deployed jacfwd J + deployed C_total + e_comp) and the pkls.

- Derivative: mu exactly linear in alpha_dla; FD vs jacfwd <= 9e-12 rel (h = 0.02 to 2e-4);
  other directions verified at O(h^2) truncation.
- Deployed template t_dla = dmu/dalpha_dla: positive at exactly 16/681 rows (k = 0.00125 and,
  z <= 3.0, 0.00175), zero-crossing k ~ 0.002 s/km, negative at all 665 remaining rows
  (deepening with z, -2.6 at z = 2.2 to -48.6 at z = 4.2).
- Whitened cosines with e_comp (deployed C_total, full leg): alpha_lls +0.495,
  alpha_subdla +0.543, alpha_dla +0.323, tau0_amp +0.406, dtau0 +0.107, ns -0.464, Ap +0.423,
  f_res_amp -0.081. |e|_wh = 1.723 (|e|_wh^2 = 2.967).
- Template-convention grid (referee T2): cos(e, template) = -0.538 for the NO-CORE template in
  the DIAG-whitened metric, reproducing the panel's -0.53 to two decimals; no with-core
  convention reaches it in any metric; core_only = +0.844 in every metric. The panel's per-z run
  (-0.59 to -0.40) is reproduced by none of 12 conventions: retired as unreproducible.
- Fragility (referee T4, synthesis-reproduced): dropping the k < 0.002 rows flips the deployed
  cosine +0.323 to -0.628 (659 rows; -0.95 by k >= 0.005); the 16 template-positive rows carry
  the great majority of the whitened injected power (block-whitened low-k |e|_wh^2 = 2.85 of
  2.97). "Correct-sign absorber" is a property of the deployed k-band edge (k_min = 1e-3), not a
  robust structural property.
- Captured fraction: single-direction beta_dla = +0.0033, required alpha_dla 0.0077 (latent
  z +0.56, inside the prior), captured whitened power cos^2 = 10.4%, |residual|/|e| = 0.946.
- GLS 8-direction decomposition (linearized; synthesis-reproduced): flat priors residual 8.4%
  with cosmology ~ 0 (alpha_dla +0.0137); deployed priors residual 58.0% with alpha_dla +0.0015,
  tau0_amp +0.0076 (+0.58 pull), ns -0.0281 (-0.57), Ap -0.0313 (-0.38). Signs match the
  measured campaign on 8/8 parameters; factor ~2 magnitude gaps expected (25+-dim non-Gaussian
  posterior vs 8-dim linear model).
- Per-z: block-whitened cos runs +0.78 (z = 2.2) to -0.25 (z = 4.2); required per-z coefficient
  +0.0103 (z = 2.2) to +0.0006 (z = 3.6) to -0.0016 (z = 4.2): a falling, sign-flipping profile
  the rigid rising (1+z)^{s_dla}, s_dla ~ N(2.366, 0.33), cannot track (matching would need
  s_dla ~ -8, ~31 prior sigma). z >= 3.6 whitened share 12.8-16.7% (block-convention dependent):
  the high-z-leverage premise is refuted either way.
- Measured paired pulls (pooled mean +/- SEM over 8 mocks, sigma_clean units; synthesis
  re-reproduced alpha_dla): ns -0.210 +/- 0.066 (+1) / +0.280 +/- 0.081 (-1);
  Ap -0.240 +/- 0.059 / +0.237 +/- 0.068; tau0_amp +0.388 +/- 0.052 / -0.270 +/- 0.078;
  alpha_dla +0.621 +/- 0.129 / -0.357 +/- 0.063; f_res_amp -0.050 +/- 0.064 / +0.034 +/- 0.050.
  Analyzer-convention (per-fit posterior sd denominator) alpha_dla = +0.250 / -0.229: the same
  draws, a different denominator (injected posteriors 1.04-1.82x wider); both true; every quote
  must state its denominator.
- Antisymmetry: all parameters antisymmetric within ~1.5 SEM except alpha_dla (+2.3 SEM,
  physical space), fully explained by softplus convexity (raw-space sums symmetric).
- Entanglement: within-posterior corr(alpha_dla, tau0_amp) = -0.061 (partial -0.013): the
  tau0-A_p ridge does NOT flow through alpha_dla; the alpha_dla response is a direct weak
  absorption at its whitened-LS optimum.

## 9. Hypothesis adjudication

| Hyp | Statement | Key evidence | Verdict | Confidence |
|---|---|---|---|---|
| H1 | Implementation sign bug (forward or injection) | single-variable strength wiring; ll identity exact per mock; 8/8 GLS sign match; deterministic row test; FD = jacfwd | REFUTED | high |
| H2 | Fiducial-centering / softplus-transform bug | sample site raw + deterministic alpha (no Jacobian to misapply); identical truth-draw prior; analyzer name-keyed, reproduced from raw draws | REFUTED | high |
| H3 | Correct code; the template genuinely points elsewhere than the injected mode | cos^2 = 10% deployed; residual 58% under priors; per-z coefficient sign-flips at z >= 4 vs rigid rising slope; mode resembles core_only (+0.84) while the lever ties the core to a negative filtered body | SUPPORTED (dominant, jointly with H7) | high |
| H4 | DLA core restoration incomplete/doubled | code: core bit-identity RECORD + ll magnitude excludes doubling/halving. TWIST: the core was omitted in the PANEL'S DIAGNOSTIC template, which manufactured the -0.53 | REFUTED in code; CONFIRMED as the panel-diagnostic defect | high |
| H5 | Injection ordering wrong (vs noise/metals/resolution) | ll identity exact; ordering effect ~ 1e-5 | REFUTED | high |
| H6 | tau = 1e6 mismatch mock-vs-forward | Leg-A self-draw: same function, same core; no tau = 1e6 code at fit time | REFUTED for this gate (real-fit core fidelity stays arm A3) | high |
| H7 | All correct; model lacks dof to absorb the mode | flat priors WOULD absorb 92% but at physically absurd incidence values the priors correctly forbid; fits land AT the whitened-LS optimum; relocation onto uniform-prior tau0_amp as geometry predicts | SUPPORTED with the qualifier (= the surviving explanation with H3) | high |
| H8 (new) | The -0.53 was manufactured by a no-core template x diag-whitened metric | reproduced -0.538 to two decimals; no with-core convention comes close | CONFIRMED (global); per-z run retired as unreproducible | high (global) |
| H9 (new) | "Correct-sign absorption" is knife-edge on the k < 0.002 rows | cos +0.32 flips to -0.63 dropping 22 rows; 16 positive rows carry ~ all whitened e-power | CONFIRMED as a fragility caveat (not a defect) | high |
| H10 (new) | The +/-1 sigma label is joint-z overstated (coherent injection of a z-block-diagonal-budgeted envelope) | cov_syst exactly z-block-diagonal; joint chi2 = 10.7 (~3.3 sigma) under DESI's model; per-block ~1 sigma | CONFIRMED as a severity convention; PI adjudicates the denomination | high |

## 10. Impact on previous conclusions

| Item | Classification | Reason |
|---|---|---|
| 4 gate cells (n_s, A_p x two arms) | remains-valid | Independently recomputed (ns -0.21/+0.28, Ap -0.24/+0.24 sigma; analyzer table reproduced to +/-0.001); implementation clean |
| alpha_dla anti-absorption claim (cos ~ -0.53, "wrong projection sign") | invalid, retire (no recompute needed; correction of record) | Manufactured by a no-core template in a diag-whitened metric (H8); deployed-metric cosine +0.323, posterior moves WITH the injection in both arms; the per-z -0.59 to -0.40 run unreproducible |
| f_res-null claim | remains-valid | Pulls -0.050/+0.034 (null); cos_wh(e, f_res) = -0.081; contraction shows f_res genuinely data-constrained elsewhere |
| tau0 relocation (~0.43) | remains-valid (mechanism sharpened) | Measured +0.39/-0.27 (or +0.43 analyzer convention); GLS predicts +0.58 via the uniform-prior amplitude sink; the ridge does NOT flow through alpha_dla |
| Widening-vs-absorption split | valid-after-reinterpretation | Injected posteriors 1.04-1.82x wider deflate analyzer-normalized shifts; +0.62 (sigma_clean) vs +0.25 (analyzer) are the same draws; all quotes must state the denominator |
| DESI budget bracket | valid-after-reinterpretation | Numbers stand; must carry the H10 severity-denomination caveat, the H9 k-band fragility caveat, and the retirement of the anti-absorb wording; per-z panel numbers retired |
| Proxied eBOSS row | undecidable | Not directly tested by any agent; structurally different question (EBOSS_DLA_FORWARD_FRAC = 0, data_likelihood.py:202-203: no alpha_dla lever on eBOSS; completeness lives in eBOSS C_data); needs its own design before any proxy number is quoted |
| Combined ledger | valid-after-reinterpretation | Row values reusable; wording changes required (mechanism, denominator, H9/H10 caveats); do not quote the stale cavestru1 thin-forward run |
| Floated-e_DLA recommendation | valid-after-reinterpretation | Still a coherent candidate, but the rationale changes: not fixing a wrong-sign lever; adding the unspanned completeness direction (mode ~ core_only, cos +0.84; 58% whitened remainder under priors). Alternative: z-resolved DLA amplitude. Design decision, not a repair |
| Freeze / re-SBC readiness | undecidable by this audit | No DLA-sector implementation blocker found, but freeze is gated on the PI decisions in Section 12 and on the Gate-B row disposition; per stop rules this audit declares nothing budget-ready |

## 11. Corrective actions (ranked)

Applicable outcome: **Outcome A** (implementation clean; injection mechanically correct; the
gate failure is genuine model inadequacy under the chosen severity convention). All proposed
changes are default-off and byte-identical when unused; audit and patch are kept logically
separate (nothing below was implemented).

1. Record corrections (necessity: HIGH; information gain: prevents wrong downstream decisions;
   risk: none; compute: none). Retire "anti-absorbs / wrong projection sign / cos = -0.53" and
   the per-z -0.59 to -0.40 run from the audit doc, meta-report, and panel-derived text; quote
   the deployed-metric +0.323 with the H9 fragility caveat; state the pull denominator on every
   table; fix "0.50" quotes of the deployed DLA prior width (deployed sd/mean ~ 1.28); reconcile
   the e/P 0.7%-vs-0.3% and 12.8%-vs-16.7% quoting differences with one line each.
2. PI adjudication of the severity denomination (necessity: HIGH, blocks ledger wording; risk:
   none; compute: none). Decide whether the DLA row is denominated in the z-coherent worst case
   (current, joint ~3.3 sigma under DESI's own cov_syst) or a z-block-consistent 1 sigma. No
   ceiling recommendation is made here (stop rule).
3. Pin the diagnostic-deployment-consistency lesson (necessity: MEDIUM-HIGH; risk: low; compute:
   trivial). Add a small pinned test asserting the DLA diagnostic template includes dla_core and
   that projection diagnostics are computed in the deployed C_total metric (the H8 failure
   class, same family as the feedback-diagnostic-deployment-consistency memory); plus a unit
   test pinning P_filt[3] + delta[:,2] == unfiltered coarse DLA power (currently RECORD only).
4. Design decision on a completeness-shaped response term (necessity: MEDIUM, conditional on 2;
   information gain: HIGH if the PI wants the DESI row absorbed rather than budgeted; risk:
   MEDIUM (new nuisance, SBC implications); compute: small mock test first, campaign only with
   sign-off). Two candidates: (a) floated e_DLA nuisance with the DESI-shipped shape (spans the
   mode by construction; prior width = the DESI budget); (b) z-resolved DLA amplitude (frees the
   z-profile the current rigid slope cannot track; does not fix the k-shape mismatch alone).
   Both default-off, byte-identical when off, mirrored in truth draws if adopted.
5. eBOSS arm design (necessity: LOW now; compute: deferred). Define what an eBOSS completeness
   arm tests given EBOSS_DLA_FORWARD_FRAC = 0 before building anything.
6. Arm A3 (real-fit DLA core fidelity: z-mean MVP, un-MF-corrected core, kfkms smearing)
   remains open with unchanged priority; not a gate blocker.

13 validation tests, mapped:

| # | Test | Status |
|---|---|---|
| 1 | Injection ll identity diff(+1)+diff(-1) = -e^T C^-1 e per mock | EXISTS (referee T1; synthesis re-ran) |
| 2 | Exact (z,k) mapping of e_dla onto the leg (no nearest-metric trust) | EXISTS (3 agents + exact lookup) |
| 3 | FD-vs-jacfwd derivative convergence on the deployed forward | EXISTS (reproducer) |
| 4 | Deployed-metric whitened cosine + beta for all 8 directions | EXISTS (reproducer; synthesis re-ran) |
| 5 | k-band fragility scan of the cosine (H9) | EXISTS (referee T4; synthesis re-ran) |
| 6 | Per-z required-coefficient profile vs the slope prior | EXISTS (referee T3/bayesian) |
| 7 | GLS flat-vs-deployed-priors attribution (incl. leave-out variants) | EXISTS (referee T3; synthesis re-ran) |
| 8 | Prior MC quantiles + railing/floor-mass + contraction | EXISTS (prior auditor) |
| 9 | Pairing/bit-identity of clean arms + stamp/strength consistency | EXISTS (3 agents) |
| 10 | Template-convention grid reproducing the -0.53 (H8) | EXISTS (referee T2) |
| 11 | Pinned unit test: core add-back bit-identity to unfiltered coarse DLA power | MUST BE BUILT (currently RECORD 2026-06-04 only) |
| 12 | Documented severity-convention check vs DESI cov_syst block structure | EXISTS as measurement; the convention NOTE must be written into the ledger row |
| 13 | Live-emulated per-mock-theta MF-corrected template projection (removes the cache-proxy caveat); prerequisite for any floated-e_DLA design test | MUST BE BUILT (one ~350s chain per theta; only needed if action 4 proceeds) |

Plus, if action 4 is adopted: a default-off byte-identity test (flag off => bit-identical
posterior pipeline) and truth-draw mirroring tests.

## 12. PI DECISION BOX

| Decision | Recommended action | Evidence | What remains uncertain | Compute consequence |
|---|---|---|---|---|
| Repair alpha_dla (prior or forward)? | NO. No defect exists; widening buys variance not bias (contraction ~ 0), tightening pushes more onto tau0/cosmology | Sections 5, 8; H1/H2 refuted; T3 leave-out (own prior owns ~10-15% of the bias) | Whether DESI's true completeness signature looks like their shipped red mode or like in-situ R_DLA (physics, unresolved) | none |
| Regenerate the injection? | NO mechanically. Only if the severity convention changes (z-block-scaled or z-incoherent arm), which is a NEW test, not a fix | Section 7; ll identity; exact mapping; H10 | The denomination choice itself (PI call) | new arm ~ same cost as the existing campaign; not justified before the denomination decision |
| Rerun the DESI DLA gate? | NO. The +/-1 results are valid and reusable after the Section 10 wording changes | Sections 8, 10 | Only the convention question above | none now |
| Still add a floated e_DLA template? | DEFER TO DESIGN (action 4). Coherent candidate with a changed rationale (span the unspanned completeness direction; mode ~ core_only +0.84); alternative = z-resolved DLA amplitude | Sections 8, 9 (H3/H7); GLS residual 58% | Whether the PI prefers absorbing (nuisance) vs budgeting (ledger row); SBC implications of a new nuisance | design pair + one small mock test first; campaign only with sign-off |
| Build the eBOSS arm now? | NO. eBOSS has no alpha_dla lever (EBOSS_DLA_FORWARD_FRAC = 0; completeness sits in eBOSS C_data), so the arm tests a structurally different question that needs its own design | Section 10 (proxied eBOSS row undecidable); data_likelihood.py:199-203 | What an eBOSS completeness injection should even perturb | deferred |
| Can the NORC refactor merge independently? | YES from this audit's standpoint: NORC stamps verified in all campaign meta, runtime parity asserts pass, nothing DLA-sector blocks it | Section 2; forward auditor Hop 4; parity asserts run_dla_completeness_shard.py:87-98 | This audit exercised only the DESI DLA path; other legs rely on the earlier NORC reviews | none |
| Can the forward be frozen? | NO DLA-sector implementation blocker found, but freeze should follow the two PI decisions above (denomination; floated-template) and the Gate-B row disposition. This audit does NOT declare the DLA row budget-ready (stop rule) | Sections 1, 10, 11 | The waiver disposition itself; arm A3 (real-fit core fidelity) remains open but was never a freeze gate | none |
| Can Wave-2 re-SBC begin? | NOT YET. Re-SBC must run on the FROZEN forward; starting before the floated-template decision risks a second stale-forward SBC (the exact failure mode that made Wave-2 mandatory) | Section 11 action 4; production-emulator/blinding roadmap | Timing of the PI decisions | re-SBC cost unchanged; no new cost from this audit |

---

Stop-condition compliance (all seven roles + synthesis): no production campaigns, no lock
regeneration, no code modification (plans only), no waiver-ceiling recommendation, no
budget-ready declaration, blind-safe throughout, audit and patch kept logically separate.

Synthesis-level uncertainty statement: HIGH confidence on every implementation verdict (each
rests on at least two independent measurements, and the synthesis re-reproduced the whitened
cosine, k-band flip, GLS table, ll identity, and alpha_dla pulls from the raw artifacts).
MEDIUM confidence on the physical reading of the DESI column (unsigned catalog-variation
envelope; the DR1 paper's exact estimator was not verified locally). The panel's -0.53
construction is INFERRED from a 2-decimal reproduction (its computation is not on disk); its
per-z run stays unexplained and is retired rather than reinterpreted. The GLS mechanism is
linearized at one fiducial; sign-level conclusions are robust (8/8 measured agreement), factor
~2 magnitudes are indicative only.
