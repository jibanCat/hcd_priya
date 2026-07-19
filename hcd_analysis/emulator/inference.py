"""Phase-C differentiable likelihood driver — the object HMC (numpyro/blackjax) and the
Cobaya adapter both wrap (design §1.4).

Mirrors PRIYA's (Ho 2023/2024) parameter contract so the cosmology community drives it
as before: the 9 cosmo/IGM params (``data.PARAM_LIMITS`` order, identical to PRIYA's
``coarse_grid``), mean-flux as per-z ``tau0`` (PRIYA's ``mean_flux="per_z"``), and HCD
nuisance as per-class ``alpha_hcd`` (LLS, subDLA, DLA — the single-amplitude analog of
PRIYA's ``a_lls/a_dla``). Wires ``predict.predict_P_obs`` + the τ₀-aware per-class C_emu
(``likelihood.sigma_at_tau0`` interp, then ``emu_var = Σ_c coef_c²·σ_c²·P_c²`` assembled
INLINE in ``log_lik_single_z`` — NOT the legacy ``likelihood.assemble_covariance``, which is
superseded) + the logdet Gaussian (``likelihood.gaussian_loglik``) + a SMOOTH bounded prior
(replacing PRIYA's ``-inf`` wall so NUTS gets finite gradients).

Everything is JAX-pure and differentiable in (θ_unit, tau0, alpha_hcd).
"""
from __future__ import annotations

import hashlib
import json
import os

import jax
import jax.numpy as jnp

from .predict import predict_P_filt
from .likelihood import sigma_at_tau0, rho_at_tau0, gaussian_loglik

# PRIYA / coarse_grid order — the names the Cobaya/numpyro adapters expose.
PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
               "hireionz", "bhfeedback")


def unit_box_logprior(theta_unit, sharpness=1e3):
    """Smooth soft-wall log-prior on the unit cube [0,1]^9 (design: replaces PRIYA's
    ``return -inf`` hard wall, which has zero/inf gradient and breaks NUTS).

    0 inside the box; a smooth quadratic penalty outside so the gradient stays finite
    everywhere (HMC can be pushed back in). ``sharpness`` sets the wall steepness.
    """
    t = jnp.asarray(theta_unit)
    below = jnp.clip(-t, 0.0, None)            # >0 where t<0
    above = jnp.clip(t - 1.0, 0.0, None)       # >0 where t>1
    return -sharpness * jnp.sum(below ** 2 + above ** 2)


def meanflux_logprior(tau0, mu, sigma):
    """Per-z Gaussian prior on τ₀(z) at the measurement width (design §4); the
    τ_eff–cosmology degeneracy can bias cosmology, so anchor τ₀ on the observed ⟨F⟩(z).
    ``tau0``/``mu``/``sigma`` broadcast over z."""
    return -0.5 * jnp.sum(((jnp.asarray(tau0) - jnp.asarray(mu)) / jnp.asarray(sigma)) ** 2)


def gaussian_logprior(value, mu, sigma):
    """Generic additive Gaussian prior (mirror PRIYA's optional hub/omega/bhfeedback)."""
    return -0.5 * (((jnp.asarray(value) - mu) / sigma) ** 2)


# HCD incidence priors — literature-calibrated (2026-06-04 Lyα agent, corrected-law
# re-derivation 2026-07-18; sources: Prochaska/O'Meara/Worseck+2010 + O'Meara+2013 +
# Fumagalli+2013 (LLS, kernel-corrected to the binned class), Zafar+2013 Table 3 counts
# (subDLA), Prochaska&Wolfe2009 (DLA)). Fractional widths σ/μ per class; LLS TIGHT
# (cosmology-degenerate, DESI DR1). DLA on the per-leg unmasked-DLA residual
# (PI-confirmed final intent 2026-06-09).
HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.50)   # σ/μ (survey=None closure/SBC path): LLS TIGHT
#   (cosmology-degenerate; the closure mock truth is the SIM itself, so the lit-kernel
#   systematic width 0.287 of the survey path deliberately does NOT cascade here); subDLA
#   BROAD 0.40 as a residual-abundance marginalization — the old rationale "poor
#   measurement, Zafar-vs-O'Meara factor-2" DISSOLVED 2026-07-18: the apparent factor-2
#   was the wrong-object bug (the deployed 'subDLA' points were Zafar Table 3's
#   Peroux+2003b DLA column); the corrected Poisson-GLM subDLA law is HEALTHY
#   (deviance/dof 0.73 vs the old chi2/dof 5.41) and the 0.40 width is kept as a
#   deliberate marginalization hedge, not a measurement statement; DLA WIDE 0.50 = the
#   masking-completeness uncertainty (the DLA-finder misses ~10% with a broad completeness
#   width), widened further above z=3.5 (see HCD_DLA_Z_RELIABLE).
# §0c DLA prior (PI-confirmed final intent 2026-06-09): the DLA-finder masking is INCOMPLETE —
# the finder misses ~10% of DLAs (completeness ~90%), so those ~10% REMAIN as full unmasked DLA
# systems in the DESI data (and the closure TARGET MOCK). The α_DLA prior is therefore centered
# on that 10% residual incidence and α_DLA is MARGINALIZED (sampled, NOT fixed) over it: the
# closure tests whether HCD-marginalization recovers cosmology DESPITE the DLA residual. The
# residual is leg-specific — KS fully masks DLAs (0% residual), DESI carries the 10% — handled
# by the per-leg DataLeg.dla_forward_frac in data_likelihood (this prior is the DESI center).
# α_DLA is one-sided (softplus, sampler_numpyro) → α_DLA ≥ 0 (a completeness fraction).
HCD_DLA_RESIDUAL_FRAC = 0.10                 # DLA residual incidence = 0.10 × data incidence
#   (the DESI 10% unmasked-DLA residual; was 0.30 for the WRONG ~70%-masking model, then 0.05
#   for the WRONG "α_DLA≈0 / masked-to-clean" revert). With w_DLA≈0.04 the α_DLA prior center is
#   ≈0.004 (σ/μ=0.50, one-sided softplus → a 10%-scale residual the forward marginalizes); the
#   closure truth's α_DLA is 0.10·w_DLA on DESI / 0 on KS (per-leg DLA-forward axis).
HCD_DLA_Z_RELIABLE = 3.5                     # DLA dN/dX unreliable beyond this z → widen σ_DLA
HCD_Z_PIVOT = 3.0
# PRIYA-sim-vs-observed dN/dX offset (literature / PRIYA-sim) per class, as a POWER-LAW in
# (1+z) — mirroring the τ₀ Kim-curve+slope model. (lit/sim)@z_pivot + the slope
# d ln(lit/sim)/d ln(1+z). LLS slots REFIT 2026-07-18 against the CORRECTED
# definition-matched laws (the deployed K1a-constrained binned-LLS law vs the PRIYA LLS
# class column, the plot_dndx_vs_literature.py construction executed inside
# scripts/derive_hcd_dndx_corrected.py): the old 1.06 was cumulative-tau>=2-vs-binned-class
# (wrong estimand pair); definition-matched, PRIYA's LLS incidence matches the corrected
# literature at z=3 to 0.5% (ratio 0.995). subDLA/DLA slots unchanged (re-verified same run).
HCD_LIT_OVER_SIM = (0.995, 1.00, 1.34)       # (LLS, subDLA, DLA): data/sim at z_pivot=3.0
# subDLA centered on the SIM (1.00) — KEPT by PI decision 9 (2026-07-18): PRIYA produces
# subDLAs IN-SITU (Rahmati+2013 self-shielding) so the sim is the faithful prior here, and
# the correction VINDICATES the sim anchor: the corrected lit/sim subDLA ratio is 1.40
# (recorded as EVIDENCE in hcd_lit_dndx_corrected.json), inside the broad σ/μ=0.40
# (HCD_PRIOR_FRAC_SIGMA[1]) that marginalizes the residual subDLA-abundance uncertainty
# rather than imposing an offset center. (The old "Zafar 0.76 / factor-2" story was the
# wrong-object DLA-column bug; see the HCD_LIT_DNDX_LAW provenance block below.)
# DLA slope deliberately WEAK (0.4, the conservative Ω_DLA∝(1+z)^0.4): the raw fit (+1.15
# corrected, same class of number as the old +1.08) is dominated by z>3.5 DLA dN/dX that the
# literature does not measure reliably — do not impose a strong DLA z-evolution; let the
# data set it (σ widened above z=3.5).
# LLS ratio SLOPE refit 2026-07-18: 0.764 = γ_LLS(corrected, constrained 2.127) − γ_sim
# (1.363, PRIYA LLS class column fit) — replaces the old 0.95 (which paired the cumulative
# compilation slope with the sim class slope). Under the 1.5 forward-zslope guard floor.
HCD_LIT_OVER_SIM_SLOPE = (0.764, 0.15, 0.40)  # d ln(lit/sim) / d ln(1+z)
# ============================ READ THIS BEFORE USING THIS CONSTANT ============================
# This is the lit/sim RATIO slope — d ln[(literature dN/dX)/(PRIYA dN/dX)]/d ln(1+z). It is the
# PRIOR CENTER at the z=3 PIVOT ONLY (where the slope CANCELS — zero production effect), consumed
# EXCLUSIVELY by lit_over_sim_at_z(z=z_pivot). It is NEVER a forward z-exponent.
#   ⇒ The FORWARD HCD incidence-weight z-slope s_c in α_c(z)=α_pivot·((1+z)/4)^s_c is a DISTINCT
#     object: closure_legb.HCD_INCIDENCE_SLOPE=(2.465,2.758,2.366) (~2.4 — the SIM w_c(z) slope
#     d ln w_c/d ln(1+z) the mock truth carries and the forward α_c(z) must track).
# Using THIS (0.764) ratio slope as the forward exponent makes the predicted dN/dX(z) FALL with z
# (truth + literature RISE) and puts the mock-truth slope σ's off-center — the wrong-object bug
# that has recurred 3+ times (at its old value 0.95, 2.9–6σ off). See hcd-dndx-zslope-bug (notes)
# + tests/test_zslope_center.py.
# =============================================================================================

# --- PER-SURVEY effective LLS-abundance pin (real-fit prior; 2026-06-11 Lyα-agent + PI) --------
# The LLS prior CENTER is the dominant DESI-A_p risk (a tight prior at an offset center moves A_p
# ~1σ; see the Phase-4b headline). The effective LLS incidence is SURVEY-SPECIFIC:
#   DESI DR1 — large, homogeneous, magnitude/redshift-selected forest sample → the cosmic-average
#     literature dN/dX is appropriate (boost 1.0), and TIGHT (cosmology-degenerate).
#   KODIAQ-SQUAD — archival high-res echelle, deliberately includes DLA/absorber-rich sightlines →
#     the data prefer ~2–3× the PRIYA LLS (arXiv:2509.18271 §4.3.3: α_LLS≈2 ⇒ "triple the LLS in
#     PRIYA"; ≈2.5× the cosmic average). The excess is SELECTION, not cosmic — and ↑α_LLS mimics
#     ↑A_p — so use a HIGH CENTER but a BROAD width (PI 2026-06-11): the KS data set it within an
#     informative window rather than the prior imposing a possibly-wrong tight number.
# A multiplier on the cosmic-average (lit/sim) LLS center, applied ONLY when ``survey`` is given
# (the closure's sim-mean cert passes survey=None and is unaffected).
HCD_LLS_SURVEY_BOOST = {"DESI": 1.0, "eBOSS": 1.0, "DESI+KS": 1.0, "KS": 2.5}
# --- PI WIDTH RULE (2026-06-17 re-determination; 1× value RE-DERIVED 2026-07-18) ---------------
# per-survey LLS fractional width σ/μ (overrides HCD_PRIOR_FRAC_SIGMA[0] when survey given). The PI
# rule: set σ_LLS to 1–2× the LITERATURE dN/dX MEASUREMENT error (1× ideal; 2× = cosmic-variance
# hedge). The 1× value is DERIVED on the CORRECTED binned-LLS points (kernel-corrected POW10 +
# O'Meara13 + Fumagalli13; PI decision 4, 2026-07-18): the measurement-only piece is
# max(GLS norm err @z=3 χ²-inflated, per-point scatter) = 0.174, and the ADOPTED width adds the
# K1a kernel COMMON-MODE systematic s_r3=0.227 in quadrature → σ/μ = 0.287 (the exact computed
# value, 3 dp; scripts/derive_hcd_dndx_corrected.py prints the breakdown; the superseded
# derive_hcd_lls_width.py 0.16→0.15 knob was measurement-only ON THE WRONG-OBJECT cumulative
# points). 2× = 0.574, the cosmic-variance hedge arm.
#   DESI / eBOSS / DESI+KS  — 1× = 0.287 (the lit-anchored real-fit primary; wider than the old
#     0.15 because the estimand-correction kernel's common-mode uncertainty is now carried
#     honestly in the width, not hidden).
#   KS — stays 0.40 (broad; its width is the SELECTION-driven PI rule — archival absorber-rich
#     sightlines — NOT the lit-error 1×/2× machinery; z<2.4 cut → corr(LLS,n_s)≈0.07, no leak).
#     FLAGGED consequence: the 2× hedge (0.574) now EXCEEDS the KS 0.40 — the old "KS already
#     broad" premise is weakened; KS deliberately keeps its own rule (PI cascade reading
#     2026-07-18, see the implementation report).
# The PRIMARY (1×) is HCD_LLS_SURVEY_FRAC_SIGMA; the 2× cosmic-variance HEDGE arm is the easily-
# toggled HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X (use_lls_width_hedge2x=True in hcd_incidence_prior).
HCD_LLS_SURVEY_FRAC_SIGMA = {"DESI": 0.287, "eBOSS": 0.287, "DESI+KS": 0.287, "KS": 0.40}
# 2× cosmic-variance hedge (double the 1× corrected lit width); KS unchanged (its own rule).
HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X = {"DESI": 0.574, "eBOSS": 0.574, "DESI+KS": 0.574, "KS": 0.40}

# --- REAL-FIT LLS forward z-slope (2026-06-17 PI re-determination; RE-JUSTIFIED 2026-07-18) -----
# The LLS prior CENTER's z-EVOLUTION is the dominant low-z LLS→n_s leak lever (NOT the width). On
# the REAL FIT (data; PRIYA≠data, the forest follows the literature dN/dX) the LLS forward z-slope
# tracks the literature LLS law slope γ_LLS=2.127 — KEPT under the 2026-07-18 corrected-law
# re-derivation (PI decision 1c): the corrected K1a binned-LLS FREE-gamma fit gives 2.137±0.67,
# consistent with the deployed 2.127, so the deployed law is the corrected points refit with gamma
# CONSTRAINED to 2.127 and this constant stays identically equal to HCD_LIT_DNDX_LAW["LLS"][1]
# (test-enforced, tests/test_zslope_center.py). NOT the sim incidence slope 2.465 (which
# over-predicts low-z LLS vs lit/truth; the sim-revert stays a PI physics-dialogue option). The
# CLOSURE/SBC path (survey=None, sim-truth mocks) STAYS on closure_legb.HCD_INCIDENCE_SLOPE=2.465
# (the slope the mock carries). subDLA/DLA keep the sim incidence slope on BOTH paths (only LLS is
# lit-anchored on the real fit). γ=2.127 > the forward z-slope guard floor 1.5, so the swap passes
# _assert_forward_zslope_center (the ratio-slope guard intent is preserved).
HCD_LLS_REALFIT_ZSLOPE = 2.127

# === LITERATURE dN/dX power-laws (A, γ) per HCD class — CORRECTED LAWS OF RECORD ================
# dN/dX_c(z) = A_c·(1+z)^γ_c. The LLS law is the REAL-FIT LLS prior CENTER anchor:
# hcd_lls_realfit_alpha_center builds α_LLS at the z-pivot from this law DIRECTLY (alt-(b)),
# round-tripping the lit dN/dX to <0.2% at the pivot (re-measured 2026-07-18); subDLA/DLA laws are
# display/round-trip references. PROVENANCE (in-house corrected-law re-derivation, 2026-07-18):
#
# THE BUGS the 2026-07-18 re-derivation retired (both verified at source by two independent
# designers + the implementer; tombstoned in hcd_analysis/emulator/lit_dndx.py):
#   1. The old "LLS" law (0.0201, 2.127) was fit on the CUMULATIVE tau912>=2 compilation
#      l(X)(N>=10^17.5) — which INCLUDES subDLAs and DLAs — while the consumer
#      (dndx_wc.alpha_from_dndx_law -> w_c_from_mu, a telescoping Poisson over DISJOINT classes)
#      requires the BINNED [17.2, 19.0) class rate: a cumulative rate fed to a disjoint-class
#      telescoping consumer (double-counts the subDLA+DLA share; also the O'Meara abscissa was
#      mis-stated, 2.4 = the f(N) pivot, not the measurement z ~2.21).
#   2. The old "subDLA" law (0.0211, 0.937) was fit on rows of Zafar+2013 Table 3's logN>=20.3
#      block = the Peroux+2003b DLA rates (a column mislabel): the wrong object entirely.
# THE FIX: estimand-corrected source arrays (lit_dndx.py) + the PRIYA-CDDF estimand-correction
# kernel (lit_dndx_kernel.py, pinned K1 recipe) + per-class estimators (Poisson GLM for counts,
# GLS with the kernel-covariance blocks for the LLS compilation), derivation of record in
# scripts/derive_hcd_dndx_corrected.py -> hcd_analysis/emulator/hcd_lit_dndx_corrected.json.
# KERNEL CHOICE = K1a (PRIYA-CDDF cache kernel, per-point b(z_i); r(3)=0.8884). PI reasoning
# (2026-07-18, verbatim): "the simulations evolve the Lya forest, LLSs, subDLAs, and DLAs jointly
# under the same structure formation, so the relative decomposition across absorber classes is
# physically self-consistent; the observational decomposition may still contain unresolved
# Eddington-bias and classification issues and is not trusted at the 30-50% level over a
# physically consistent simulation; the absolute incidence stays anchored to observations."
# (PI, 2026-07-18). K3 (lit-anchored telescoping-consistent) and K2 (POW10-own-words) remain in
# the JSON as the one-sided minus bracket arms; the K1a telescoping pulls (+1.7..+2.4σ at z>=3)
# are EXPECTED by design (PRIYA-shape vs literature class-decomposition tension, PI-accepted).
# DEPLOYED LAWS:
#   LLS    = the K1a-corrected 8-point compilation refit with gamma CONSTRAINED to the deployed
#            z-slope 2.127 (amplitude-only GLS under the same covariance; PI decision 1c — the
#            free-gamma fit 2.137±0.67 is the consistency evidence, recorded in the JSON), at
#            full precision so the reproduction test is bit-level.
#   subDLA = the Poisson GLM on the verified Zafar+2013 Table 3 >=19.0-block counts
#            (n=(4,11,24,23,19,8), sum 89; deviance/dof 0.73 — healthy where the old
#            wrong-object law showed chi2/dof 5.41), full precision.
#   DLA    = (0.0076, 1.592) KEPT: the shared-WLS refit of the PW09 Table 1 points reproduces it
#            exactly at the deployed rounding (machinery anchor; refit documented in the JSON).
# REFERENCES: Prochaska, O'Meara & Worseck 2010 (arXiv:0912.0292, Table 4); Zafar+2013
# (arXiv:1307.0602, Table 3); Prochaska & Wolfe 2009 (arXiv:0811.2003, Table 1); O'Meara+2013
# (arXiv:1204.3093); Fumagalli+2013 (arXiv:1308.1101). (Two arXiv IDs circulating in older
# notes were WRONG: 0912.0562 is a graphene paper, 1306.0333 is not Zafar.)
# ARTIFACT: hcd_analysis/emulator/hcd_lit_dndx_corrected.json (committed; sha256
# 55249943310021091183a0f170625e672368875d866c8400b2e1d5748e699cca — also carried live in
# hcd_prior_constants_payload()["derivation_json_sha256"]). Spec + decision record:
# hcd_priya_notes/docs/superpowers/2026-07-18-corrected-law-spec.md (incl. ADDENDUM) and the
# 2026-07-18 PI decision bundle. Tests: tests/test_lit_dndx_corrected.py (bit-level
# reproduction, estimand regression, telescoping record), tests/test_lit_dndx{,_fits,_kernel}.py.
HCD_LIT_DNDX_LAW = {"LLS": (0.01840091965972645, 2.127),
                    "subDLA": (0.004832763139114936, 2.438007767664778),
                    "DLA": (0.0076, 1.592)}
# The estimand each deployed law is a law OF (disjoint binned classes — NEVER cumulative; the
# consumer telescopes disjoint classes). Guarded at the consumption boundary by
# assert_dndx_law_estimand; test-enforced equal to the JSON's hcd_lit_dndx_estimand.
HCD_LIT_DNDX_ESTIMAND = {"LLS": "binned_17.2_19.0", "subDLA": "binned_19.0_20.3",
                         "DLA": "binned_ge20.3"}


def assert_dndx_law_estimand(cls, expected, where):
    """ESTIMAND GUARD (2026-07-18 corrected-law re-derivation): assert the deployed law for
    ``cls`` is a law of the ``expected`` estimand (a CONSTANT-dict check — trace-safe, mirrors
    the _btilt_site guard pattern; never touches traced values). Fires if a future edit
    reinstates a wrong-object law under a re-labeled estimand (the cumulative-vs-binned LLS bug
    or the Zafar DLA-column subDLA bug)."""
    got = HCD_LIT_DNDX_ESTIMAND.get(cls)
    assert got == expected, (
        f"HCD dN/dX law estimand mismatch [{where}]: HCD_LIT_DNDX_ESTIMAND[{cls!r}] = {got!r}, "
        f"consumer requires {expected!r}. The consumer telescopes DISJOINT binned classes — a "
        f"cumulative (or wrong-object) law here is the 2026-07-18 corrected-law bug recurring. "
        f"See the HCD_LIT_DNDX_LAW provenance block.")

# --- HCD prior PIVOT GUARD (PI 2026-06-17, the CENTER-construction bug; band re-derived
# 2026-07-18 at the corrected laws) ---------------------------------------------------------------
# THE BUG: the LLS/subDLA pivot AMPLITUDE was built from w_c_med = nanmedian(w_c_cache[:,1:], axis=0)
# — the MEDIAN over ALL z-groups (z=2.0–5.4). Since w_c rises monotonically with z, that all-z median
# (LLS 0.274) equals the z≈3.6 value, but it is consumed as the z=3 PIVOT → the LLS α-center came out
# ~1.45× too high (0.291 instead of the z=3-consistent value), overshooting the lit dN/dX law worst
# at low z = the LLS→n_s leak. The FIX builds the pivot from the z=3 STRUCTURAL w_c (closure/SBC) or
# the lit dN/dX law directly (real fit). These bands let a future revert to nanmedian(...all z...)
# TRIP at runtime. BAND (2026-07-18): adopted corrected center × (0.83, 1.21) (the same relative
# margins as before), 2-dp → (0.14, 0.21): contains the corrected lit-law center 0.172 (K1a,
# constrained slope) AND the sim z=3 w_c center 0.2004 (closure-guard reuse), still EXCLUDES the
# all-z-median 0.2909 (×boost). ALLZ_MEDIAN + GUARD_REL unchanged.
HCD_PIVOT_LLS_ALPHA_Z3_BAND = (0.14, 0.21)   # z=3-consistent LLS α-pivot (DESI/cosmic-avg boost 1.0)
HCD_PIVOT_LLS_ALLZ_MEDIAN = 0.2909           # the BUGGY all-z-median LLS α-pivot (z≈3.6) — must NOT recur
HCD_PIVOT_GUARD_REL = 0.05                   # |α − all-z-median| must exceed this·all-z-median


def hcd_lls_realfit_alpha_center(Xbar_z, z=HCD_Z_PIVOT, boost=1.0):
    """REAL-FIT LLS α-pivot CENTER built from the CORRECTED literature dN/dX power-law DIRECTLY
    (the validated alt-(b)): dN/dX_LLS(z) = A·(1+z)^γ (HCD_LIT_DNDX_LAW["LLS"], the K1a-corrected
    binned [17.2,19.0) law constrained to γ=2.127; PI 2026-07-18) → α_LLS(z) via the EXACT
    telescoping w_c map (dndx_wc.alpha_from_dndx_law), round-tripping the lit dN/dX to <0.2% at
    the pivot (re-measured at the corrected laws). ``Xbar_z`` = the cache
    mean-absorption-path-per-sightline at ``z`` (the z=3 pivot value). Returns the scalar α_LLS
    center (× ``boost`` for the per-survey selection excess). At z=3, Xbar≈0.632 → α_LLS≈0.172 —
    the z=3-consistent corrected center, NOT the all-z-median ~0.291 (the CENTER-construction
    bug), and NOT the pre-correction 0.194 (the cumulative-estimand bug)."""
    assert_dndx_law_estimand("LLS", "binned_17.2_19.0", "hcd_lls_realfit_alpha_center")
    from hcd_analysis.emulator.dndx_wc import alpha_from_dndx_law
    A = jnp.asarray([HCD_LIT_DNDX_LAW[c][0] for c in ("LLS", "subDLA", "DLA")])
    g = jnp.asarray([HCD_LIT_DNDX_LAW[c][1] for c in ("LLS", "subDLA", "DLA")])
    z_arr = jnp.atleast_1d(jnp.asarray(z, float))
    Xb_arr = jnp.atleast_1d(jnp.asarray(Xbar_z, float))
    alpha = alpha_from_dndx_law(A, g, Xb_arr, z_arr)          # (...,3) (LLS,subDLA,DLA)
    return float(boost) * float(jnp.asarray(alpha).reshape(-1, 3)[0, 0])


def assert_hcd_pivot_z3(alpha_mu, z, where, *, boost=1.0):
    """PIVOT GUARD (PI 2026-06-17): the LLS α-PIVOT center ``alpha_mu`` (scalar, the LLS slot of the
    hcd_incidence_prior μ — already ×survey boost) must be the z=3 value, NOT the all-z median (=z≈3.6).
    Only checked at the z=3 pivot (the slope-cancellation point the center is built at). FIRES if the
    LLS α-center is ≈ the buggy all-z-median 0.291 (×boost) → a future revert to
    nanmedian(w_c_cache[...all z...]) trips here. See the HCD-pivot dN/dX low-z overshoot bug."""
    if abs(float(z) - float(HCD_Z_PIVOT)) > 1e-6:
        return                                                # only meaningful at the z=3 pivot
    a = float(alpha_mu)
    allz = float(HCD_PIVOT_LLS_ALLZ_MEDIAN) * float(boost)    # the buggy z≈3.6 value (per boost)
    lo, hi = (b * float(boost) for b in HCD_PIVOT_LLS_ALPHA_Z3_BAND)
    assert abs(a - allz) > HCD_PIVOT_GUARD_REL * allz, (
        f"HCD LLS α-PIVOT center [{where}] = {a:.4f} ≈ the all-z-median (z≈3.6) value {allz:.4f} — "
        f"the HCD pivot MUST use the z=3 w_c, NOT the median-over-all-z (= z≈3.6). See the dN/dX "
        f"low-z overshoot bug (CENTER-construction fix 2026-06-17).")
    assert lo <= a <= hi, (
        f"HCD LLS α-PIVOT center [{where}] = {a:.4f} outside the z=3-consistent band [{lo:.4f},{hi:.4f}] "
        f"(boost={boost}). Expected ≈0.172 (corrected lit-law, PI 2026-07-18) / ≈0.200 (sim z=3 w_c). "
        f"A center near the all-z-median {allz:.4f} (= z≈3.6) is the CENTER-construction bug. See the "
        f"dN/dX low-z overshoot + the HCD_LIT_DNDX_LAW provenance block.")


# --- HCD prior-constants freeze payload + signature (2026-07-18; closes the tripwire gap
# forward_signature's docstring flags: prior constants were NOT covered by any signature) -------
_HCD_DERIVATION_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "hcd_lit_dndx_corrected.json")


def hcd_prior_constants_payload():
    """JSON-native dict of every deployed HCD prior constant + the corrected-law derivation
    provenance (adopted kernel, r(3), the per-kernel alpha bracket endpoints, and the sha256 of
    the committed derivation JSON). The freeze task MUST insert this as
    analysis.lock["prior_constants"] alongside forward_signature() so a prior-constant change can
    never again be invisible to the freeze artifact (spec 2026-07-18 sec 5.8; test-pinned by
    tests/test_lit_dndx_corrected.py case 8). SCOPE: this payload covers the inference.py HCD
    prior constants only — closure_legb.ZSLOPE_PRIOR_SIGMA and HCD_INCIDENCE_SLOPE (and any
    other closure_legb-side prior constant) are covered by NO signature; the freeze checklist
    must record them separately."""
    with open(_HCD_DERIVATION_JSON, "rb") as fh:
        raw = fh.read()
    j = json.loads(raw)
    return dict(
        HCD_LIT_DNDX_LAW={c: list(v) for c, v in HCD_LIT_DNDX_LAW.items()},
        HCD_LIT_DNDX_ESTIMAND=dict(HCD_LIT_DNDX_ESTIMAND),
        HCD_LIT_OVER_SIM=list(HCD_LIT_OVER_SIM),
        HCD_LIT_OVER_SIM_SLOPE=list(HCD_LIT_OVER_SIM_SLOPE),
        HCD_LLS_REALFIT_ZSLOPE=float(HCD_LLS_REALFIT_ZSLOPE),
        HCD_PIVOT_LLS_ALPHA_Z3_BAND=list(HCD_PIVOT_LLS_ALPHA_Z3_BAND),
        HCD_PIVOT_LLS_ALLZ_MEDIAN=float(HCD_PIVOT_LLS_ALLZ_MEDIAN),
        HCD_PIVOT_GUARD_REL=float(HCD_PIVOT_GUARD_REL),
        HCD_PRIOR_FRAC_SIGMA=list(HCD_PRIOR_FRAC_SIGMA),
        HCD_LLS_SURVEY_BOOST=dict(HCD_LLS_SURVEY_BOOST),
        HCD_LLS_SURVEY_FRAC_SIGMA=dict(HCD_LLS_SURVEY_FRAC_SIGMA),
        HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X=dict(HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X),
        HCD_DLA_RESIDUAL_FRAC=float(HCD_DLA_RESIDUAL_FRAC),
        HCD_DLA_Z_RELIABLE=float(HCD_DLA_Z_RELIABLE),
        HCD_Z_PIVOT=float(HCD_Z_PIVOT),
        adopted_kernel=j["kernel_chosen"],
        adopted_kernel_r3=j["kernel_table"]["budget"]["r3_k1"],
        adopted_alpha_lls_z3=j["alpha_lls_z3"]["adopted"],
        bracket_alpha_lls_z3=j["alpha_lls_z3"]["per_kernel"],
        derivation_json_sha256=hashlib.sha256(raw).hexdigest(),
    )


def hcd_prior_signature():
    """Stable sha256 hex digest over the canonical-JSON (sorted-keys) prior-constants payload —
    the prior-constant analog of closure_legb.forward_signature (freeze/audit artifact; consumed
    by NOTHING in the deployed inference path)."""
    payload = json.dumps(hcd_prior_constants_payload(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lit_over_sim_at_z(z, ratio_pivot=HCD_LIT_OVER_SIM, slope=HCD_LIT_OVER_SIM_SLOPE,
                      z_pivot=HCD_Z_PIVOT):
    """z-dependent (literature/sim) dN/dX ratio per class — a power-law in (1+z):
    r_c(z) = r_c(z_p)·((1+z)/(1+z_p))^s_c. The α-prior center tracks the OBSERVED dN/dX_c(z)
    evolution this way (the τ₀-analog: a fixed curve + slope, not a single number)."""
    r = jnp.asarray(ratio_pivot); s = jnp.asarray(slope)
    return r * ((1.0 + jnp.asarray(z)) / (1.0 + z_pivot)) ** s


def hcd_incidence_prior(w_c_fid, z=HCD_Z_PIVOT, lit_over_sim=None, survey=None,
                        use_lls_width_hedge2x=False):
    """Per-class HCD incidence prior (μ, σ) on α_c (the effective per-class sightline weight
    in ``predict_P_obs``), centered on the OBSERVED incidence AT the data redshift ``z``
    (NOT the sim's), from the fiducial sim weights ``w_c_fid`` = (w_LLS, w_subDLA, w_DLA) ×
    the z-SLOPE literature/sim ratio (``lit_over_sim_at_z(z)``; override with
    ``lit_over_sim``). LLS/subDLA largely UNMASKED → center = (lit/sim)(z)·w_c; the DLA class is
    INCOMPLETELY masked (the finder misses ~10%, PI-confirmed final intent 2026-06-09) → center =
    HCD_DLA_RESIDUAL_FRAC·(lit/sim)(z)·w_DLA = 0.10·(...) (the unmasked-DLA residual the forward
    MARGINALIZES over; leg-specific via data_likelihood.dla_forward_frac — this is the DESI
    center, KS forward DLA term is 0). α=w_c (the sim) is NOT the LLS/subDLA center: PRIYA
    mis-predicts subDLA/DLA dN/dX AND its z-slope (see scripts/plot_dndx_vs_literature.py).
    Widths = literature fractional σ/μ × center (DLA σ/μ=0.50, masking-completeness).
    Returns (alpha_mu (3,), alpha_sigma (3,)); DLA should additionally be one-sided
    (half-normal/softplus) in the sampler — α_DLA ≥ 0 (a completeness fraction), sampled
    (marginalized) over the 10% residual center.
    """
    w = jnp.asarray(w_c_fid)                                    # (3,)
    r = lit_over_sim_at_z(z) if lit_over_sim is None else jnp.asarray(lit_over_sim)
    fl, fs, fd = HCD_PRIOR_FRAC_SIGMA
    # PER-SURVEY effective-LLS pin (real fit only; survey=None leaves the closure cert untouched):
    # boost the LLS CENTER (KS selection excess) and override its fractional WIDTH (DESI tight / KS
    # broad). subDLA/DLA are survey-agnostic here (DLA is masked; subDLA tracks the cosmic average).
    lls_boost = 1.0
    if survey is not None:
        lls_boost = HCD_LLS_SURVEY_BOOST.get(survey, 1.0)
        # PI WIDTH RULE: 1× lit measurement error (primary) or the 2× cosmic-variance hedge.
        _fsig = HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X if use_lls_width_hedge2x else HCD_LLS_SURVEY_FRAC_SIGMA
        fl = _fsig.get(survey, fl)
    mu = jnp.stack([lls_boost * r[0] * w[0], r[1] * w[1], HCD_DLA_RESIDUAL_FRAC * r[2] * w[2]])
    # DLA dN/dX is unreliable beyond z≈3.5 → widen σ_DLA above it (weak high-z prior) so the
    # data, not the prior, sets the high-z DLA incidence.
    dla_inflate = 1.0 + jnp.clip(jnp.asarray(z) - HCD_DLA_Z_RELIABLE, 0.0, None)
    sigma = jnp.stack([fl * mu[0], fs * mu[1], fd * mu[2] * dla_inflate])
    return mu, sigma


def predict_P_obs_and_cov_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                                   pf_stats, sigma_zb, alpha_centres, cosmic_cov,
                                   dla_core, dla_shot_flag, shot_inflate=10.0,
                                   cemu_inflate=1.0, rho_zb=None):
    """Per-z (P_obs (K,), C (K,K)) — the EXACT forward model + covariance the
    likelihood uses, factored out so the closure mock can draw its noise from the
    IDENTICAL C (Leg A's contract: C_mock == C_like; closure_mocks calls this).

      P_obs = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)
      C     = cosmic_cov + diag(emu_var).

    emu_var is C_emu's per-k variance, in ONE of two forms (C stays DIAGONAL IN k either way
    — the CS decomposition confirmed no k-correlation; only the per-k class structure changes):

      DIAGONAL (default, ``rho_zb=None``):
        emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c²,  coef = [1−Σα, α_LLS, α_subDLA, α_DLA].
      CROSS-CLASS (opt-in, ``rho_zb`` (C,C,K,Tb) given):
        emu_var = Σ_{c,c'} coef_c·coef_c'·ρ_cc'(k,z,τ₀)·P_c·P_c',
        ρ_cc'(k,z,τ₀) = the τ₀-interp'd 4×4 cross-class second moment of the held-out
        FRACTIONAL residual (``build_xclass_error_vector``). The DIAGONAL ρ_cc=σ_c² recovers
        the diagonal form; the OFF-diagonals add the class-coupling (the 4 per-class residuals
        share ONE network → coherently correlated). ρ is a sample covariance ⇒ SPD ⇒
        emu_var = coefᵀ(P∘ρ∘P)coef ≥ 0 GUARANTEED. Differentiable in (θ,τ₀,α): ∂/∂α now
        carries the cross terms (∂emu_var/∂α_c = 2 Σ_c' coef_c'·ρ_cc'·P_c·P_c' · ∂coef_c/∂α).

    No data, no residual, no jitter — JUST the model mean and covariance (the jitter is
    added inside ``gaussian_loglik`` / the mock's Cholesky). Differentiable in (θ9,τ₀,α).
    """
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)          # (4,K) emulated
    P_clean = P_filt[0]
    P_dla_unf = P_filt[3] + jnp.asarray(dla_core)                           # unfiltered DLA
    P_cls = jnp.stack([P_clean, P_filt[1], P_filt[2], P_dla_unf])           # (4,K) class powers
    a = jnp.asarray(alpha_hcd)                                              # (3,)
    coef = jnp.concatenate([jnp.atleast_1d(1.0 - jnp.sum(a)), a])          # (4,) [clean,LLS,sub,DLA]
    P_obs = jnp.einsum("c,ck->k", coef, P_cls)                             # = P_clean + Σ α_c(P_c−P_clean)
    # C_emu per-k variance: cross-class 4×4 block (opt-in) OR the per-class diagonal (default).
    if rho_zb is not None:
        rho_ck = rho_at_tau0(rho_zb, alpha_centres, z, tau0)               # (4,4,K) τ₀-interp'd
        # emu_var(k) = Σ_{c,c'} coef_c·coef_c'·ρ_cc'(k)·P_c·P_c'  (≥0: ρ SPD ⇒ coefᵀ(P∘ρ∘P)coef≥0).
        emu_var = jnp.einsum("c,d,cdk,ck,dk->k", coef, coef,
                             jnp.nan_to_num(rho_ck), P_cls, P_cls)
    else:
        sigma_ck = sigma_at_tau0(sigma_zb, alpha_centres, z, tau0)          # (4,K) fractional
        emu_var = jnp.einsum("c,ck,ck->k", coef ** 2,
                             jnp.nan_to_num(sigma_ck) ** 2, P_cls ** 2)     # Σ coef²·σ²·P_c²
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var) * cemu_inflate
    cosmic = jnp.asarray(cosmic_cov)
    C = (jnp.diag(cosmic) if cosmic.ndim == 1 else cosmic) + jnp.diag(emu_var)
    return P_obs, C


def log_lik_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                     pf_stats, sigma_zb, alpha_centres, cosmic_cov, P_data, dla_core,
                     dla_shot_flag, shot_inflate=10.0, cemu_inflate=1.0,
                     valid_k=None, include_logdet=True, rho_zb=None):
    """Per-z Gaussian log-likelihood for the CORRECTED HCD forward model + the logdet term.

      P_obs = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)          (clean-forest baseline)
      logL_z = −½ rᵀC⁻¹r − ½ logdet C,  r = P_data − P_obs,  C = cosmic_cov + C_emu(θ,τ₀).

    α_c = the effective post-masking per-class incidence (LLS, subDLA, DLA); α_c = w_c
    reproduces the sim's contaminated P_tier_p. ``P_c`` are LIVE-emulated per-class P_filt:
    filtered for LLS/subDLA, UNFILTERED for DLA via ``dla_core`` (K,) = the DLA-core add-back
    P_DLA^unf − P_DLA^filt. ∂P_obs/∂α_c = (P_c − P_clean) (≠0 for LLS — fixes the old
    Δ_LLS≡0 bug).

    C_emu is the per-class emulator error propagated through P_obs's class coefficients
    coef = [1−Σα, α_LLS, α_subDLA, α_DLA] over (clean, LLS, subDLA, DLA):
    emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c². The (4,K) FRACTIONAL error ``sigma_zb`` is
    τ₀-banded (Tb axis); ``cemu_inflate`` is the conservative inflation (review I1);
    ``valid_k`` (bool, K; FIXED, not traced) neutralises out-of-range/Nyquist bins (σ
    all-NaN, P_data NaN) so they carry no info and no NaN gradient.
    ``rho_zb`` (opt-in): a (4,4,K,Tb) cross-class block → C_emu uses the cross-class form
    emu_var = Σ_cc' coef_c·coef_c'·ρ_cc'·P_c·P_c' (the off-diagonals capture the coherent
    cross-class correlation the single network induces; recovers the diagonal when ρ=diag(σ²)).
    ``include_logdet=False`` = the negative control. Differentiable in (θ9, τ₀, α).

    The (P_obs, C) assembly is shared with ``predict_P_obs_and_cov_single_z`` so the
    closure mock draws noise from the IDENTICAL covariance.
    """
    P_obs, C = predict_P_obs_and_cov_single_z(
        model, theta9, z_unit, z, tau0, alpha_hcd, pf_stats=pf_stats, sigma_zb=sigma_zb,
        alpha_centres=alpha_centres, cosmic_cov=cosmic_cov, dla_core=dla_core,
        dla_shot_flag=dla_shot_flag, shot_inflate=shot_inflate, cemu_inflate=cemu_inflate,
        rho_zb=rho_zb)
    # NaN-safe residual: sanitise P_data (out-of-range bins NaN) BEFORE the subtract so
    # the where-branch can't poison the gradient; valid_k zeroes those bins' residual.
    r = jnp.nan_to_num(jnp.asarray(P_data), nan=0.0) - P_obs
    if valid_k is not None:
        r = jnp.where(jnp.asarray(valid_k), r, 0.0)
    if include_logdet:
        return gaussian_loglik(r, C)
    # negative control: chi2 only, no logdet (still SPD-jittered for a fair comparison)
    K = C.shape[-1]
    Cj = C + (1e-10 * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    L = jnp.linalg.cholesky(Cj)
    sol = jax.scipy.linalg.cho_solve((L, True), r)
    return -0.5 * (r @ sol)


def log_lik_multiz(model, theta9, tau0_vec, alpha_hcd, *, pf_stats, z, z_unit, sigma_zb,
                   alpha_centres, cosmic_cov, P_data, dla_core, dla_shot_flag, valid_k,
                   shot_inflate=10.0, cemu_inflate=1.0, include_logdet=True, rho_zb=None):
    """LIKELIHOOD-ONLY multi-z log-likelihood = Σ_z log_lik_single_z (no priors).

    The data-bin sum is **vmap'd over the z-axis** (K is fixed across bins → no padding; CS
    review M2): θ9 + α are shared (closed over), ``tau0_vec`` is per-z, and the per-z arrays
    (``z, z_unit, sigma_zb, cosmic_cov, P_data, dla_core, dla_shot_flag, valid_k``) carry a
    leading z-axis. This is the LIKELIHOOD-ONLY payload for numpyro's ``factor`` / Cobaya's
    ``logp`` — priors are added separately (CS review M3: never double-count). Differentiable.

    ``rho_zb`` (opt-in cross-class C_emu): None → the diagonal σ path (default, unchanged);
    a (n_z,4,4,K,Tb) array → the cross-class 4×4 block per z (vmapped over the leading z-axis
    like the other per-z leaves). Whether None is decided STATICALLY (python), so the vmap
    in_axes never traces over a None leaf and the diagonal path never recompiles."""
    use_xclass = rho_zb is not None
    def one(tau0_z, z_z, zu_z, sz, cc, pd, dc, flag, vk, rz):
        return log_lik_single_z(
            model, theta9, zu_z, z_z, tau0_z, alpha_hcd, pf_stats=pf_stats, sigma_zb=sz,
            alpha_centres=alpha_centres, cosmic_cov=cc, P_data=pd, dla_core=dc,
            dla_shot_flag=flag, valid_k=vk, shot_inflate=shot_inflate,
            cemu_inflate=cemu_inflate, include_logdet=include_logdet, rho_zb=rz)
    # rho_zb maps over the z-axis when present (in_axes 0), else is broadcast as None
    # (in_axes None — a non-array leaf vmap passes through unmapped).
    in_axes = (0,) * 9 + (0 if use_xclass else None,)
    rz_arg = jnp.asarray(rho_zb) if use_xclass else None
    per_z = jax.vmap(one, in_axes=in_axes)(
        jnp.asarray(tau0_vec), jnp.asarray(z), jnp.asarray(z_unit),
        jnp.asarray(sigma_zb), jnp.asarray(cosmic_cov), jnp.asarray(P_data),
        jnp.asarray(dla_core), jnp.asarray(dla_shot_flag),
        jnp.asarray(valid_k), rz_arg)
    return jnp.sum(per_z)


def log_posterior_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                           pf_stats, sigma_zb, alpha_centres, cosmic_cov, P_data, dla_core,
                           dla_shot_flag, tau0_mu, tau0_sigma, alpha_mu, alpha_sigma,
                           shot_inflate=10.0, cemu_inflate=1.0, valid_k=None,
                           include_logdet=True, box_sharpness=1e3, rho_zb=None):
    """Single-z log-POSTERIOR = log-likelihood + smooth unit-box prior + τ₀ mean-flux
    Gaussian + the per-class HCD **incidence prior** on α (TIGHT informative Gaussian
    centered on the structural w_c(dN/dX); LLS especially tight — it's cosmology-degenerate,
    DESI DR1; DLA centered on the 10% unmasked-DLA residual, marginalized). The differentiable scalar
    a single-z-bin NUTS run targets (the multi-z posterior sums ``log_lik_single_z`` over
    data bins + ONE box prior + the per-z τ₀ Gaussian + the α prior — T4 driver).

    PRIOR NOTE (review M1): ``unit_box_logprior`` is a SMOOTH soft-wall, used only for the
    raw-``log_prob`` (blackjax) path; production numpyro uses Uniform(0,1)+auto-bijector
    (exactly-flat in-box, finite grads) — set box_sharpness=0 there."""
    ll = log_lik_single_z(
        model, theta9, z_unit, z, tau0, alpha_hcd, pf_stats=pf_stats, sigma_zb=sigma_zb,
        alpha_centres=alpha_centres, cosmic_cov=cosmic_cov, P_data=P_data, dla_core=dla_core,
        dla_shot_flag=dla_shot_flag, shot_inflate=shot_inflate, cemu_inflate=cemu_inflate,
        valid_k=valid_k, include_logdet=include_logdet, rho_zb=rho_zb)
    lp = unit_box_logprior(theta9, sharpness=box_sharpness)
    lp += meanflux_logprior(tau0, tau0_mu, tau0_sigma)
    lp += jnp.sum(gaussian_logprior(alpha_hcd, alpha_mu, alpha_sigma))     # HCD incidence prior
    return ll + lp
