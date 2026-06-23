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


# HCD incidence priors — literature-calibrated (2026-06-04 Lyα agent; sources: O'Meara+2013
# / Prochaska+2010 / Fumagalli+2013 (LLS), Zafar+2013 (subDLA), Prochaska&Wolfe2009 /
# Noterdaeme+2012 (DLA)). Fractional widths σ/μ per class; LLS TIGHT (cosmology-degenerate,
# DESI DR1). DLA on the per-leg unmasked-DLA residual (PI-confirmed final intent 2026-06-09).
HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.50)   # σ/μ: LLS TIGHT (cosmology-degenerate); subDLA
#   BROAD (poor measurement — the Zafar-vs-O'Meara factor-2); DLA WIDE 0.50 = the
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
# (1+z) — mirroring the τ₀ Kim-curve+slope model. PRIYA does NOT match the data: the
# observed dN/dX evolves FASTER with z than the sim (γ_lit > γ_sim), so a single
# z-independent ratio mis-centers the prior at the z edges. (lit/sim)@z_pivot + the slope
# d ln(lit/sim)/d ln(1+z), fit by scripts/plot_dndx_vs_literature.py (PRIYA vs
# Prochaska&Wolfe09 / Zafar13 / O'Meara13):
HCD_LIT_OVER_SIM = (1.06, 1.00, 1.34)        # (LLS, subDLA, DLA): data/sim at z_pivot=3.0
# subDLA centered on the SIM (1.00), NOT the old Zafar+2013 0.76: that literature value is
# factor-2 uncertain (Zafar-vs-O'Meara; Berg+2019 XQ-100 revises it), and PRIYA produces subDLAs
# IN-SITU (Rahmati+2013 self-shielding) so the sim is the faithful prior here. The old 0.76 sat
# −0.8σ below the sim subDLA incidence → pulled α_subDLA low and leaked into n_s (corr≈+0.3; HCD
# referee 2026-06-11). The broad σ/μ=0.40 (HCD_PRIOR_FRAC_SIGMA[1]) marginalizes the residual
# subDLA-abundance uncertainty rather than imposing an offset center.
# DLA slope deliberately WEAK (0.4, the conservative Ω_DLA∝(1+z)^0.4): the raw fit (+1.08,
# or +1.90 on z≤3.5) is dominated by z>3.5 DLA dN/dX that the literature does not measure
# reliably — so do not impose a strong DLA z-evolution; let the data set it (σ widened above).
HCD_LIT_OVER_SIM_SLOPE = (0.95, 0.15, 0.40)  # d ln(lit/sim) / d ln(1+z)
# ============================ READ THIS BEFORE USING THIS CONSTANT ============================
# This is the lit/sim RATIO slope — d ln[(literature dN/dX)/(PRIYA dN/dX)]/d ln(1+z). It is the
# PRIOR CENTER at the z=3 PIVOT ONLY (where the slope CANCELS — zero production effect), consumed
# EXCLUSIVELY by lit_over_sim_at_z(z=z_pivot). It is NEVER a forward z-exponent.
#   ⇒ The FORWARD HCD incidence-weight z-slope s_c in α_c(z)=α_pivot·((1+z)/4)^s_c is a DISTINCT
#     object: closure_legb.HCD_INCIDENCE_SLOPE=(2.465,2.758,2.366) (~2.4 — the SIM w_c(z) slope
#     d ln w_c/d ln(1+z) the mock truth carries and the forward α_c(z) must track).
# Using THIS (0.95) ratio slope as the forward exponent makes the predicted dN/dX(z) FALL with z
# (truth + literature RISE) and puts the mock-truth slope 2.9–6σ off-center — the wrong-object bug
# that has recurred 3+ times. See hcd-dndx-zslope-bug (notes) + tests/test_zslope_center.py.
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
# --- PI WIDTH RULE (2026-06-17 re-determination, PI-approved) ----------------------------------
# per-survey LLS fractional width σ/μ (overrides HCD_PRIOR_FRAC_SIGMA[0] when survey given). The PI
# rule: set σ_LLS to 1–2× the LITERATURE dN/dX MEASUREMENT error (1× ideal; 2× = cosmic-variance
# hedge). The 1× value is DERIVED from the lit dN/dX measurement uncertainty for LLS (O'Meara13 /
# Fumagalli13 / Prochaska10): the WLS-fit normalization error at z=3 (χ²-inflated) AND the per-point
# scatter about the WLS power-law, taken as max → fractional σ/μ ≈ 0.16 → the clean 1× knob 0.15
# (scripts/derive_hcd_lls_width.py prints the breakdown; 2× = 0.30, the cosmic-variance hedge arm).
#   DESI / eBOSS / DESI+KS  — 1× = 0.15 (TIGHT, the litWLS-anchored real-fit primary; the
#     litWLS CENTER's z-evolution — not the width — was the dominant low-z LLS→n_s leak, so a 0.15
#     width at the lit-anchored center now sits ON the data-truth, zero prior-pull leak BY
#     CONSTRUCTION; gated on the litWLS-σ0.15 closure NUTS reproducing the historical width isolator).
#   KS — stays 0.40 (broad; selection-driven LLS excess, z<2.4 cut → corr(LLS,n_s)≈0.07, no leak).
# The PRIMARY (1×) is HCD_LLS_SURVEY_FRAC_SIGMA; the 2× cosmic-variance HEDGE arm is the easily-
# toggled HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X (use_lls_width_hedge2x=True in hcd_incidence_prior).
HCD_LLS_SURVEY_FRAC_SIGMA = {"DESI": 0.15, "eBOSS": 0.15, "DESI+KS": 0.15, "KS": 0.40}
# 2× cosmic-variance hedge (double the 1× lit measurement error); KS unchanged (already broad).
HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X = {"DESI": 0.30, "eBOSS": 0.30, "DESI+KS": 0.30, "KS": 0.40}

# --- REAL-FIT LLS forward z-slope (litWLS, 2026-06-17 PI re-determination) ----------------------
# The LLS prior CENTER's z-EVOLUTION is the dominant low-z LLS→n_s leak lever (NOT the width). On
# the REAL FIT (data; PRIYA≠data, the forest follows the literature dN/dX) the LLS forward z-slope
# must track the literature WLS power-law slope γ_LLS=+2.127 (lit dN/dX_LLS(z)=A·(1+z)^γ, A≈0.0201,
# WLS over O'Meara13/Fumagalli13/Prochaska10), NOT the sim incidence slope 2.465 (which over-predicts
# low-z LLS by +62–87% vs lit/truth). The CLOSURE/SBC path (survey=None, sim-truth mocks) STAYS on
# closure_legb.HCD_INCIDENCE_SLOPE=2.465 (the slope the mock carries). subDLA/DLA keep the sim
# incidence slope on BOTH paths (only LLS is lit-anchored on the real fit). γ=2.127 > the forward
# z-slope guard floor 1.5, so the swap passes _assert_forward_zslope_center (the 0.95 ratio-slope
# guard is preserved). See the spec 2026-06-17-hcd-prior-redetermination.
HCD_LLS_REALFIT_ZSLOPE = 2.127

# --- LITERATURE dN/dX power-laws (A, γ) per HCD class -------------------------------------------
# dN/dX_c(z) = A_c·(1+z)^γ_c, the WLS over the literature points (derive_hcd_lls_width.py /
# plot_dndx_vs_literature.py). The LLS law (O'Meara13/Fumagalli13/Prochaska10; A=0.0201, γ=2.127,
# χ²/dof=0.72) is the REAL-FIT LLS prior CENTER anchor: hcd_lls_realfit_alpha_center builds α_LLS at
# the z-pivot from this law DIRECTLY (alt-(b)), round-tripping the lit dN/dX to <0.34% — see the
# HCD-pivot CENTER-construction fix (2026-06-17). subDLA/DLA laws are display/round-trip references.
HCD_LIT_DNDX_LAW = {"LLS": (0.0201, 2.127), "subDLA": (0.0211, 0.937), "DLA": (0.0076, 1.592)}

# --- HCD prior PIVOT GUARD (PI 2026-06-17, the CENTER-construction bug) -------------------------
# THE BUG: the LLS/subDLA pivot AMPLITUDE was built from w_c_med = nanmedian(w_c_cache[:,1:], axis=0)
# — the MEDIAN over ALL z-groups (z=2.0–5.4). Since w_c rises monotonically with z, that all-z median
# (LLS 0.274) equals the z≈3.6 value, but it is consumed as the z=3 PIVOT → the LLS α-center came out
# ~1.45× too high (0.291 instead of the z=3-consistent ~0.194–0.200), overshooting the lit dN/dX law
# 2.05× at z=2.4 / 1.63× at z=3 (worst at low z) = the LLS→n_s leak. The FIX builds the pivot from the
# z=3 STRUCTURAL w_c (closure/SBC) or the lit dN/dX law directly (real fit). These bands let a future
# revert to nanmedian(...all z...) TRIP at runtime: the z=3-consistent LLS α-center is ≈0.19–0.21
# (sim z=3 w_c·1.06 = 0.2004; lit-law alt-(b) = 0.1938), the all-z-median value is ≈0.29 (DESI boost).
HCD_PIVOT_LLS_ALPHA_Z3_BAND = (0.16, 0.24)   # z=3-consistent LLS α-pivot (DESI/cosmic-avg boost 1.0)
HCD_PIVOT_LLS_ALLZ_MEDIAN = 0.2909           # the BUGGY all-z-median LLS α-pivot (z≈3.6) — must NOT recur
HCD_PIVOT_GUARD_REL = 0.05                   # |α − all-z-median| must exceed this·all-z-median


def hcd_lls_realfit_alpha_center(Xbar_z, z=HCD_Z_PIVOT, boost=1.0):
    """REAL-FIT LLS α-pivot CENTER built from the literature dN/dX power-law DIRECTLY (the validated
    alt-(b)): dN/dX_LLS(z) = A·(1+z)^γ (HCD_LIT_DNDX_LAW["LLS"]) → α_LLS(z) via the EXACT telescoping
    w_c map (dndx_wc.alpha_from_dndx_law), round-tripping the lit dN/dX to <0.34%. ``Xbar_z`` = the
    cache mean-absorption-path-per-sightline at ``z`` (the z=3 pivot value). Returns the scalar
    α_LLS center (× ``boost`` for the per-survey selection excess). At z=3, Xbar≈0.632 → α_LLS≈0.194
    — the z=3-consistent center, NOT the all-z-median ~0.291 (the CENTER-construction bug)."""
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
        f"(boost={boost}). Expected ≈0.194 (lit-law) / ≈0.200 (sim z=3 w_c·1.06). A center near the "
        f"all-z-median {allz:.4f} (= z≈3.6) is the CENTER-construction bug. See the dN/dX low-z overshoot.")


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
