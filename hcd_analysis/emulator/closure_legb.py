"""Phase-C T4b — Leg-B closure-under-misspecification on the REAL DESI+KS grids.

Leg B (plan §0) certifies the EMULATOR-AS-LIKELIHOOD: the mock truth is a HELD-OUT-SIM
P1D (NOT the emulator's own model), the noise is cosmic-ONLY (``ε ~ N(0, C_data)`` — the
emulator error IS the sim≠emu discrepancy, not added noise; CS M5), and the likelihood
uses ``C = C_data + C_emu`` (cross-class). Coverage then tests whether C_emu correctly
sizes the sim-emu discrepancy ON THE REAL GRID. Judged on one-sided empirical coverage
(≥ nominal) + bias, NOT rank uniformity (uniformity is false-by-construction here).

Differences from Leg A (``closure_sbc`` / ``closure_mocks``):
  - truth P1D is the CACHE's measured contaminated power (held-out sim), interpolated onto
    each leg's k — NOT ``predict_P_obs`` of the production model;
  - noise = jittered-Cholesky of C_data ALONE (cosmic-only), per leg;
  - the likelihood is the multi-leg ``data_likelihood.data_loglik`` (real C_data + C_emu),
    pointed at the mock legs, sampled by a numpyro model that mirrors ``sampler_numpyro``'s
    priors (θ~Uniform, τ₀ α-ladder Normal, α HCD priors) — a thin adapter so the FACTOR is
    ``data_loglik`` instead of the cache-grid ``log_lik_from_ctx``.

``--smoke`` runs the FULL path at tiny size (N≈4–8 mocks, small warmup/samples) end to end
and prints per-param coverage + #divergences + L + the cross-class-C_emu-on-leg check.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    -m hcd_analysis.emulator.closure_legb --smoke
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from typing import NamedTuple

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.diagnostics as npd
from numpyro.infer import NUTS, MCMC, init_to_median, init_to_sample
from numpyro.infer.util import constrain_fn
from scipy.stats import norm as _scipy_norm

from . import train as T
from .data import load_cache, make_splits, KIM_AMP, KIM_SLOPE, Z_LIMITS, sampling_unit_bounds
from .meanflux_prior import (meanflux_tau0_prior, becker13_tau0, tau0_alpha_priya,
                             fit_tau0_alpha_priya, TAU0_AMP_RANGE, DTAU0_RANGE, TAU0_PIVOT_Z)
from .inference import (PARAM_NAMES, hcd_incidence_prior,
                        HCD_LIT_OVER_SIM_SLOPE, HCD_Z_PIVOT, HCD_DLA_RESIDUAL_FRAC,
                        HCD_LLS_REALFIT_ZSLOPE, HCD_LLS_SURVEY_BOOST,
                        HCD_LLS_SURVEY_FRAC_SIGMA, HCD_PRIOR_FRAC_SIGMA,
                        hcd_lls_realfit_alpha_center, assert_hcd_pivot_z3)
from .sampler_numpyro import _dla_raw_mu
from . import data_likelihood as DL
from .closure_diagnostics import (
    thin_to_ess, central_interval, empirical_coverage, sbc_rank,
    ecdf_pit_bands, loglik_rank,
)

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"

REPO = "/home/mfho/hcd_priya"
# The ONE production model + its matched cross-class error vector (plan §5).
CKPT = f"{REPO}/checkpoints/final_fold0"
ERROR_VECTOR = f"{REPO}/checkpoints/error_vector.npz"
XCLASS_ERROR_VECTOR = f"{REPO}/checkpoints/error_vector_xclass.npz"
CACHE_PATH = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"

# The ECDF simultaneous-band gate is only correctly sized for L≳99 thinned draws (the
# L_FLOOR rule; see closure_sbc). Below it the ECDF is PATH-only (the --smoke regime).
L_FLOOR = 99
DIVERGENCE_RETRY_TARGET_ACCEPT = (0.95, 0.99)

# STEP-A M3 (z-slope MARGINALIZED, real-fit config): the HCD per-class incidence z-slope s_c
# is SAMPLED (not fixed to HCD_LIT_OVER_SIM_SLOPE). Prior is centered on the forward's own
# fixed slope (so M3 isolates the marginalization COST, not a center-shift bias) at the
# literature WLS 1σ width (scripts/diag_legb_slope_prior_tradeoff: WLS fit of dN/dX vs PRIYA
# with the quoted literature dN/dX error bars; class order LLS, subDLA, DLA).
ZSLOPE_PRIOR_SIGMA = (0.52, 0.53, 0.33)
# res_corr AMPLITUDE nuisance α(z) = α₀·((1+z)/(1+Z_PIVOT))^s (Task 1.3), marginalized
# FORWARD-ONLY through the MF chokepoint. Priors LOCKED by the Phase-0 Fisher gate:
#   α₀ ~ TruncatedNormal(loc=1.0, scale=SIGMA_A0, low=0.0)  — wide symmetric, truncated >0
#                                                             so P_MF = exp(α·log_rc)·… > 0;
#   s  ~ Normal(0.0, SIGMA_S)                                — modest z-slope (Phase-0: α is
#                                                             NOT degenerate with dτ₀ → keep s).
# Z_PIVOT (z=3) is owned by data_likelihood (DL.Z_PIVOT) — the α(z) pivot used at the chokepoint.
SIGMA_A0 = 1.0
SIGMA_S = 0.5
# SPECTRAL-RESOLUTION nuisance (option-b, Gate B): b_res(z) = f_res_amp·((1+z)/(1+F_RES_PIVOT_Z))^f_res_slope,
# threaded FORWARD-ONLY into _resolution_factor = exp(2·b_res·k²·R_z²) (data_likelihood). DISTINCT from
# alpha_res (res_corr / Gate A). Prior TIGHT on physics (4-lens 2026-07-02): DESI is deconvolved to a ~few-%
# residual (implied b_res~0.02 from syst_e_resolution) and a k² tilt is ~degenerate with the thermal cutoff, so
# a WIDE prior (cup1d's [-0.5,0.5] code default) opens the NORC failure mode on the weak legs. amp centered 0
# (0 = no distortion; additive-in-exponent no-op). Sampled iff ctx.sample_res (default False → golden).
F_RES_PIVOT_Z = 3.0
F_RES_AMP_SIGMA = 0.02         # Normal(0, .) on f_res_amp — TIGHT (physics), NOT cup1d's wide default
F_RES_SLOPE_SIGMA = 0.5        # Normal(0, .) on f_res_slope — modest z-slope (4-lens: tighter than 1.0)


def _bres_of_z(z, f_res_amp, f_res_slope, *, z_pivot=F_RES_PIVOT_Z):
    """Per-z spectral-resolution amplitude b_res(z) = f_res_amp·((1+z)/(1+z_pivot))^f_res_slope (the
    forward-only option-b nuisance; mirrors tau0_alpha_priya / the alpha_res α(z) power law). f_res_amp=0
    ⇒ b_res≡0 for ANY slope ⇒ _resolution_factor=exp(0)=1 (the golden no-op). Differentiable in both."""
    z = jnp.asarray(z)
    return f_res_amp * ((1.0 + z) / (1.0 + z_pivot)) ** f_res_slope
# FULL per-class HCD-incidence z-slope d ln w_c(z)/d ln(1+z), 60-sim-population median (measured
# 2026-06-14, scripts/diag_hcd_zslope_nsbias.py). This is the slope the mock TRUTH actually carries
# (the held-out sim's native w_c(z)) — the RIGHT center for the 2D B_HCD tilt. It is a DIFFERENT
# object from inference.HCD_LIT_OVER_SIM_SLOPE=(0.95,0.15,0.40) (the lit/sim-RATIO slope). per-sim
# LLS scatter ≈0.07; subDLA/DLA differentials δs=(0,+0.29,−0.10).
HCD_INCIDENCE_SLOPE = (2.465, 2.758, 2.366)

# Forward z-slope sanity threshold: the incidence slope is ~2.4 (LLS 2.465); the WRONG lit/sim
# RATIO slope HCD_LIT_OVER_SIM_SLOPE is ~0.95. A center below this floor means someone reverted the
# forward z-exponent to the ratio slope (the wrong-object bug). Kept comfortably below 2.465 and
# above 0.95 so it catches a 0.95 reversion but never false-trips on the legitimate incidence center.
_FWD_ZSLOPE_FLOOR = 1.5


def _assert_forward_zslope_center(center, where):
    """GUARD the FORWARD HCD z-slope PRIOR CENTER (NOT individual sampled draws): the LLS-class
    center MUST be the incidence slope HCD_INCIDENCE_SLOPE (~2.4), never the lit/sim RATIO slope
    HCD_LIT_OVER_SIM_SLOPE (~0.95). ``center`` is a CONCRETE (constant) array — HCD_INCIDENCE_SLOPE,
    ctx.zslope_mu, or hcd_btilt_mu — so float() is trace-safe (it is never a traced NUTS sample).
    Catches a 0.95 reversion of the forward exponent; see hcd-dndx-zslope-bug."""
    c0 = float(np.asarray(center).reshape(-1)[0])
    assert c0 > _FWD_ZSLOPE_FLOOR, (
        f"HCD forward z-slope center [{where}] = {c0:.4f} is below {_FWD_ZSLOPE_FLOOR} — it must be "
        f"HCD_INCIDENCE_SLOPE (~2.4, the SIM incidence-weight slope), NOT the lit/sim RATIO slope "
        f"HCD_LIT_OVER_SIM_SLOPE (~0.95). The forward exponent was reverted to the wrong object. "
        f"See hcd-dndx-zslope-bug.")


def hcd_pivot_wc_and_xbar(d, z_pivot=HCD_Z_PIVOT, z_tol=0.05):
    """The HCD-prior PIVOT structural inputs from the cache, AT the z=3 pivot (the CENTER-construction
    fix, PI 2026-06-17). Returns ``(w_c_z3 (3,), Xbar_z3 scalar)``:

      w_c_z3 = median of the cache structural w_c[LLS,subDLA,DLA] over the rows AT z≈z_pivot — the
               z=3 PIVOT structural weight, NOT the all-z median nanmedian(w_c_cache[:,1:]) (=z≈3.6,
               the BUG: w_c rises monotonically so the all-z median over-estimates the z=3 pivot ~1.45×).
      Xbar_z3 = the cache mean-absorption-path-per-sightline Xbar(z) evaluated at z_pivot (deg-2 z-fit),
               the input the lit-dN/dX-law LLS center (hcd_lls_realfit_alpha_center) needs.

    Both are built from the SAME cache the all-z median came from, only restricted to the z=3 pivot
    rows (w_c) / fit and evaluated at z=3 (Xbar). See the dN/dX low-z overshoot CENTER-construction bug.
    """
    wc = np.asarray(d["w_c_cache"])                      # (R,4) clean,LLS,subDLA,DLA
    zrow = np.asarray(d["z_grid"])
    sel = np.abs(zrow - float(z_pivot)) < float(z_tol)
    assert sel.sum() > 0, f"no cache rows within {z_tol} of z_pivot={z_pivot}"
    w_c_z3 = np.nanmedian(wc[sel, 1:], axis=0)           # (3,) z=3 structural w_c

    # Xbar(z) deg-2 fit (same construction as plot_hcd_prior_dndx_overlay.build_xbar): per-group
    # Xbar = X_tot / N_sl, N_sl from the telescoping clean-fraction, fit in z, evaluated at z=3.
    gid = np.asarray(d["snap_group_idx"]); Xtot = np.asarray(d["snap_total_path_dX"])
    dndx = np.asarray(d["snap_dNdX"]); Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(axis=1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    Xbar_z3 = float(np.polyval(cf, float(z_pivot)))
    return w_c_z3, Xbar_z3


# PER-LEG DLA-residual fraction in the closure TARGET MOCK (§0c, PI-confirmed final intent
# 2026-06-09): the 10% unmasked-DLA residual belongs in the target. The DLA finder misses ~10%
# of DLAs (completeness ~90%) → on DESI those 10% REMAIN as full systems in the target; KS fully
# masks DLAs → 0% residual. make_truth_from_sim builds the DLA-MASKED baseline + the FULL DLA
# excess add-back; make_legb_mock adds ``TRUTH_DLA_FRAC[leg]``·excess per leg. This is the truth
# side; the forward marginalizes α_DLA over it (DataLeg.dla_forward_frac: DESI 1.0 / KS 0.0).
TRUTH_DLA_FRAC = {"DESI": 0.10, "KS": 0.0}

# Default NUTS theta prior bounds (unit cube): the IGM params (herei/heref/alphaq) restricted to
# the ORIGINAL PRIYA box, n_s kept extended — see data.SAMPLING_LIMITS. Resolved at import so the
# LegBCtx default (None) maps to these in _legb_model/_legb_priors_only; override via ctx fields
# (set both to 0/1 to recover the old full-box prior for a diagnostic arm).
_THETA_UNIT_LO, _THETA_UNIT_HI = sampling_unit_bounds()

# --- Phase-4d test-3: METAL injection into the MOCK (host/numpy) ------------------------------
# Inject metal contamination the FORWARD's reduced _metal_factor cannot fully reproduce, to test
# whether the baseline cosmology stays unbiased. Two forms (metal survey 2026-06-11, agent):
#   form="desi_full" — the full DESI DR1 model (arXiv:2601.21432): Lyα–SiIII + Lyα–SiII DOUBLET
#     (1190.42/1193.28 Å, ratio r), both as sigmoid-decorrelated multiplicative cross-terms, PLUS
#     an ADDITIVE same-ion SiII–SiII term (Gaussian-damped, intra-doublet frequency) — the piece a
#     multiplicative (1+f) factor STRUCTURALLY cannot match (the bias probe).
#   form="eboss" — the McDonald/eBOSS SiIIIcorr (lya_emulator_full): 1 + aa² + 2aa·cos(Δv_SiIII·k),
#     aa = f_SiIII/(1−⟨F⟩), NO decorrelation, SiIII only.
# Amplitudes are f_X (the metal flux decrement); the effective oscillation amplitude is A_X =
# f_X/(1−⟨F⟩). Defaults: f_SiIII≈0.009 (PRIYA-eBOSS), SiII/SiII-SiII sub-dominant.
_LAMBDA_SiIIb = 1193.28      # second SiII doublet line [Å] (leading line DL.LAMBDA_SiII=1190.42)

def metal_inject(P, k, mean_flux, *, form="desi_full", f_SiIII=0.009, f_SiII=0.004,
                 f_SiII_SiII=0.002, r_doublet=0.5, k_damp=0.05, k_decorr=0.05, k_SiII=None,
                 a_SiIII_direct=None, a_SiII_direct=None, damp="sigmoid", k_cross=None, cross=False):
    """Return the metal-contaminated mock P1D (numpy). ⟨F⟩=mean_flux (=exp(−τ_eff)); a=0 ⇒ P.

    ``a_SiIII_direct`` / ``a_SiII_direct`` (default None → the f/(1−⟨F⟩) map): when given, the
    oscillation amplitude is the DIRECT value (skip the f/(1−⟨F⟩) step) — e.g. the Ma+2025 sim
    power-law amplitude, which is already an amplitude, not an f.
    ``damp`` (default ``"sigmoid"`` → BYTE-EXACT to the legacy code): ``"ma"`` REPLACES the SiIII
    cross-term decorrelation with the Ma+2025 damping ``exp(k/k_cross)`` (``k_cross<0`` ⇒ decays
    with k); ``"gauss"`` REPLACES it with the GAUSS envelope ``exp(−(k/k_cross)²)`` (Arm 4: a damping
    the sigmoid forward cannot reproduce). ``form="ma2025"`` is the SiIII-ONLY (Ma drops SiII)
    DIRECT-amplitude branch (Arm 3 = ``damp="ma"``, Arm 4 = ``damp="gauss"``).

    ``k_SiII`` (Model C+, default None → ``k_decorr``, byte-exact): a SEPARATE sigmoid decorrelation
    scale for the SiII doublet (and the cross term) — matches the forward's distinct k_SiIII/k_SiII.
    ``cross`` (Model C+, default False → byte-exact): ADD the SiIII–SiII metal-metal CROSS term
    (cup1d ``Cmm``, UNDAMPED to match cup1d), 2 a_SiIII a_SiII (cos(k Δv_b) + r cos(k Δv_a)); ∝ a_SiII
    ⇒ OFF when f_SiII/a_SiII=0 (eBOSS / back-compat). Defaults reproduce the legacy output."""
    k = np.asarray(k, float)
    one_minus_F = max(1.0 - float(mean_flux), 1e-3)
    A3 = (f_SiIII / one_minus_F) if a_SiIII_direct is None else float(a_SiIII_direct)
    dv_SiIII = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    if form == "ma2025":                                  # Ma+2025 sim: SiIII-only, DIRECT a, Ma/Gauss damp
        # Out-of-class structural stress: the damping REPLACES the forward's sigmoid D(k) (a genuine
        # shape misspecification). "ma" exp(k/k_cross) (k_cross<0 ⇒ decays); "gauss" exp(−(k/k_cross)²).
        env = np.exp(-(k / k_cross) ** 2) if damp == "gauss" else np.exp(k / k_cross)
        return P * (1.0 + A3 ** 2 + 2.0 * A3 * np.cos(dv_SiIII * k) * env)
    if form == "eboss":                                   # McDonald/eBOSS SiIIIcorr, no decorrelation
        # dv_SiIII from the code's own line constants (= 2269.96 km/s), consistent with the
        # desi_full branch below; previously a hardcoded 2271.0 (the rounded literature anchor).
        cross_t = 2.0 * A3 * np.cos(dv_SiIII * k)
        if damp == "ma":                                  # Ma damping in place of the (absent) D(k)
            cross_t = cross_t * np.exp(k / k_cross)
        elif damp == "gauss":
            cross_t = cross_t * np.exp(-(k / k_cross) ** 2)
        return P * (1.0 + A3 ** 2 + cross_t)
    # --- desi_full ---
    k2 = k_decorr if k_SiII is None else float(k_SiII)          # SiII (+cross) decorrelation scale
    dvA = dv_SiIII                                               # Lyα–SiIII
    dva = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiII)      # Lyα–SiII line a (1190.42)
    dvb = DL.C_KMS * np.log(DL.LAMBDA_LYA / _LAMBDA_SiIIb)       # Lyα–SiII line b (1193.28)
    dvd = DL.C_KMS * np.log(_LAMBDA_SiIIb / DL.LAMBDA_SiII)      # SiII intra-doublet (~719 km/s)
    if damp == "ma":                                            # Ma damping in place of the sigmoid D
        D3 = D2 = np.exp(k / k_cross)
    elif damp == "gauss":
        D3 = D2 = np.exp(-(k / k_cross) ** 2)
    else:                                                       # sigmoid decorrelation (DESI), default
        D3 = 2.0 - 2.0 / (1.0 + np.exp(-k / k_decorr))          # SiIII scale (byte-exact when k2==k_decorr)
        D2 = 2.0 - 2.0 / (1.0 + np.exp(-k / k2))                # SiII (+cross) scale
    A2 = (f_SiII / one_minus_F) if a_SiII_direct is None else float(a_SiII_direct)
    C_LyaSiIII = A3 ** 2 + 2.0 * A3 * np.cos(dvA * k) * D3
    C_LyaSiII = A2 ** 2 * (1.0 + r_doublet ** 2) \
        + 2.0 * A2 * (np.cos(dvb * k) + r_doublet * np.cos(dva * k)) * D2
    # ADDITIVE same-ion SiII–SiII (Gaussian-damped) — unfittable by the multiplicative _metal_factor:
    C_SiII_SiII = f_SiII_SiII * (1.0 + r_doublet ** 2 + 2.0 * r_doublet * np.cos(dvd * k)) \
        * np.exp(-(k / k_damp) ** 2)
    # SiIII–SiII metal-metal CROSS (Model C+, cup1d Cmm; ∝ A3·A2 ⇒ OFF when A2=0). UNDAMPED (cup1d
    # leaves Cmm undamped; the damped-vs-undamped choice is immaterial in band — see the forward).
    C_cross = 0.0
    if cross:
        dvcb = DL.C_KMS * np.log(_LAMBDA_SiIIb / DL.LAMBDA_SiIII)   # SiIII–SiII line b
        dvca = DL.C_KMS * np.log(DL.LAMBDA_SiII / DL.LAMBDA_SiIII)  # SiIII–SiII line a
        C_cross = 2.0 * A3 * A2 * (np.cos(dvcb * k) + r_doublet * np.cos(dvca * k))
    return P * (1.0 + C_LyaSiIII + C_LyaSiII + C_SiII_SiII + C_cross)


def _metal_f_of_z(z, node_z, f_nodes):
    """Per-z metal flux decrement f(z) from the 2 flat-log nodes (numpy twin of the forward's
    jnp.interp): ``log10 f(z)`` LINEAR in ``log10(1+z)`` between the nodes (a single power-law in
    (1+z) → power-law-exact), CLAMPED to the node value beyond ``[node_z[0], node_z[1]]``
    (np.interp default; eBOSS z>4.2 → f(4.2)). ``z`` scalar or array; ``node_z``/``f_nodes`` (2,)."""
    node_z = np.asarray(node_z, float)
    f_nodes = np.asarray(f_nodes, float)
    return 10.0 ** np.interp(np.log10(1.0 + np.asarray(z, float)),
                             np.log10(1.0 + node_z), np.log10(f_nodes))


# metal_zevo gate — 3 BLIND-SAFE injection arms (Model C, gate-b-data-nuisance). Inject from the
# PRIOR / an independent sim, NEVER the DESI best-fit f (that would un-blind the closure). The clean
# self-draw truth FORCES the metal f-nodes to 0 (make_leg_a_legmock does not forward them ⇒ a_SiIII=0),
# so the injected arm is the SOLE metal signal (de-double-count). node_z=(2.2,4.2). Arms 1/2 use the
# per-z f→a map a(z)=f(z)/(1−⟨F⟩(z)) (the within-z convention); arm 3 is the Ma+2025 direct power-law.
#   form="zevo"   : host loop interpolates f per z and calls metal_inject(form="desi_full", cross=True)
#                   on EVERY metals_on leg — SiIII+SiII on legs in metal_siII_legs, SiIII-only
#                   (f_SiII=0 ⇒ cross auto-OFF) elsewhere (4-lens fix (a): eBOSS is now IN-CLASS via
#                   desi_full+f_SiII=0+matched decorrelation, NOT the undamped "eboss"). f_SiII_SiII=0;
#                   k_SiIII/k_SiII = the arm's decorrelation scale (inside the forward's float bracket
#                   ⇒ the Model C+ forward can reproduce it exactly).
#   form="ma2025" : host loop computes a_SiIII(z)=0.014·((1+z)/4)^0.79 + k_cross(z)=−1.58e-2·((1+z)/4)^1.15
#                   s/km and calls metal_inject(form="ma2025", a_SiIII_direct, k_cross, damp="ma"); SiII OFF.
#   form="ma2025_gauss" : ARM 4 (4-lens fix (b)) — a DECREASING-trend SiIII metal a(z)=a0·((1+z)/4)^p
#                   (p<0) with a GAUSS damping exp(−(k/k_cross)²) the SIGMOID forward cannot reproduce
#                   (genuinely OUT-of-class for the floated-decorrelation Model C+ forward). SiII OFF.
METAL_ZEVO_ARMS = {
    "arm1_decreasing": dict(form="zevo", f_SiIII_nodes=(0.010, 0.010),
                            f_SiII_nodes=(0.006, 0.006), node_z=(2.2, 4.2),
                            k_SiIII=0.05, k_SiII=0.05),
    "arm2_increasing": dict(form="zevo", f_SiIII_nodes=(0.004, 0.020),
                            f_SiII_nodes=(0.004, 0.013), node_z=(2.2, 4.2),
                            k_SiIII=0.012, k_SiII=0.006),         # GENERIC in-prior increasing arm:
    #   NOT the DESI Table-D2 best fit (0.0074/0.0035) -- blind-safe per the 4-lens review (do not
    #   inject the published best-fit). f_SiIII_z1=0.020 keeps a(z) INCREASING (a +8%: the f-rise
    #   beats the 1/(1-<F>) fall; threshold f4>0.0186) yet sits 0.176 dex OFF the 0.03 prior ceiling
    #   (2.2x the headroom of the old railed 0.025, which could manufacture an A_p leak in the gate).
    "arm3_ma2025": dict(form="ma2025", node_z=(2.2, 4.2)),
    "arm4_decreasing_ooc": dict(form="ma2025_gauss", a0=0.012, p=-0.6, k_cross=0.02,
                                damp="gauss", node_z=(2.2, 4.2)),
}


# ============================================================================ #
#  LegBCtx — the frozen leg context the numpyro model + mock generator share.
# ============================================================================ #
class LegBCtx(NamedTuple):
    """Everything the Leg-B sampler/mock need (host + jnp mix; not jit-traced as a whole).

    The numpyro model closes over this; jit traces only over the sampled params.
      model, pf_stats, dla_core_leg : the production forward model + per-leg DLA core.
      legs                          : list[DataLeg] (the mock overwrites P_data per mock).
      cache_k                       : (Kc,) the emulator cache angular-k grid.
      z_global                      : (nZ,) ascending union of leg z (the τ₀ ladder grid).
      sigma_zb_per_leg / rho_zb_per_leg : the diagonal / cross-class C_emu error vector
                                          ALREADY sliced to each leg's z-bins.
      alpha_centres                 : (Tb,) τ₀-band centres (α units).
      tau0_mu / tau0_sigma          : (nZ,) the mean-flux prior on the GLOBAL z grid.
      alpha_hcd_mu / alpha_hcd_sigma: (3,) HCD incidence prior (LLS,subDLA,DLA).
      cemu_inflate                  : the conservative C_emu inflation scalar.
      mf                            : a ``MultiFidelity`` (eval grid == cache grid) OR None.
                                      When set, the Leg-B forward routes P_obs through the
                                      certified through-MF correction (LF P_filt × exp(g +
                                      log res_corr)); the mock TRUTH is built at the SAME MF
                                      resolution (the gate invariant — applied to BOTH forward
                                      and truth so it cancels in ΔP). None → the LF path.
      mf_floor                      : an ``MFFloor`` (the LF→HR generalization + n_s-edge
                                      C_emu floor) OR None. Added on the small-scale leg(s)
                                      (leg.mf_floor_on) when ``mf`` is set; None → no floor.
    """
    model: object
    pf_stats: dict
    dla_core_leg: dict          # {leg.name: (n_z_leg, Kc)} per-leg DLA core (cache delta[,2])
    legs: list
    cache_k: jnp.ndarray
    z_global: np.ndarray
    sigma_zb_per_leg: dict
    rho_zb_per_leg: dict
    alpha_centres: jnp.ndarray
    tau0_mu: jnp.ndarray
    tau0_sigma: jnp.ndarray
    alpha_hcd_mu: jnp.ndarray
    alpha_hcd_sigma: jnp.ndarray
    cemu_inflate: float
    mf: object = None
    mf_floor: object = None
    # SHAPE-AWARE MF floor (Phase-5a, 2026-06-12): the low-rank LOSO-eps outer-product C_emu
    # covariance (mf_shape_per_leg = {leg.name: (N,N) fractional}, mf_shape_infl the inflation).
    # Fired on a leg when an entry is present — captures the coherent k-tilt the diagonal floor
    # cannot. None → off (back-compat). See data_likelihood.MFShape / mf_shape_cov_for_leg.
    mf_shape_per_leg: object = None
    mf_shape_infl: float = 1.0
    # 60-sim LF-EMULATOR-COHERENCE C_emu term (Phase-5a, 2026-06-12): the k-coherent emulator
    # generalization gap (8-fold/60-sim LOSO), a SECOND off-diagonal C_emu term distinct from the
    # 6-HR resolution shape floor. Built/bound the same way (fixed-P_data amplitude). None → off.
    # NOT gated on `with_mf` — it is an LF-emulator residual (applies on the LF path too).
    mf_emucoh_per_leg: object = None
    mf_emucoh_infl: float = 1.0
    mf_emucoh_offdiag_only: bool = False   # per-term diagonal allocation: absorb emucoh's diagonal
                                           # into emu_var (max), add only its off-diagonal (2026-06-12)
    sample_metals: bool = False          # opt-in (2026-06-13): sample a SHARED a_SiIII metal nuisance
    a_siiii_max: float = 0.15            # and apply _metal_factor on metals_on legs (DESI/eBOSS). Off
                                         # by default → golden byte-exact (a_SiIII=0 ⇒ factor≡1).
    sample_a_siii: bool = False          # Stage C opt-in: also float the SiII DOUBLET amplitude a_SiII
                                         # (after a_SiIII). Off → a_SiII=0 ⇒ byte-exact.
    # OPT-IN FLAT-LOG metal-amplitude prior (Task A1, gate-b-data-nuisance). A STATIC python str
    # selects the prior on the SHARED a_SiIII/a_SiII oscillation-amplitude sites (resolved at TRACE
    # time → a legal python branch; NEVER trace this field):
    #   "uniform" (DEFAULT, golden-safe): dist.Uniform(0, a_siiii_max) — BYTE-EXACT to the legacy code.
    #   "flatlog": dist.LogUniform(a_lo, a_hi) on the SAME site names, with the flat-log10(f) prior on
    #     the metal flux decrement f mapped to the amplitude a=f/(1−⟨F⟩_ref):
    #       a_lo = 10**metal_logf_lo / (1−F_ref),  a_hi = 10**metal_logf_hi / (1−F_ref).
    # The flat-log f de-weights the a² P-boost (the A_p leak) while leaving the linear 2a·cos
    # oscillation (the n_s rail) likelihood-driven. Keeping the site NAMED a_SiIII/a_SiII preserves
    # every by-name downstream read (_draws_matrix/_packed_names/the re-score loglik/constrain_fn).
    metal_prior: str = "uniform"
    metal_logf_lo: float = -11.0         # flat-log10(f) lower bound (f = metal flux decrement)
    metal_logf_hi: float = -2.0          # flat-log10(f) upper bound
    metal_one_minus_F_ref: float = None  # 1−F_ref: the ONE global scalar mapping f→a (a=f/(1−F_ref)).
                                         # None → build_legb_ctx derives it from the fiducial mean flux
                                         # exp(−Kim07(z)) on the union z-grid (a CONSTANT scale on a
                                         # log-uniform → shape-neutral). Required for the flatlog branch.
    marginalize_zslope: bool = True      # DEFAULT (2026-06-10): sample the HCD per-class z-slope
                                         # s_c with the literature dN/dX slope±1σ prior — so the HCD
                                         # incidence evolves on a PHYSICAL amplitude(pivot α)+slope,
                                         # not a fixed power-law (mirrors the τ₀ amplitude+slope; the
                                         # PI: break HCD–A_p–τ₀ degeneracy via physical priors on
                                         # BOTH). Set False only for the fixed-slope ablation.
    zslope_mu: object = None             # (3,) prior center on s_c (None → HCD_INCIDENCE_SLOPE ~2.4,
                                         # the SIM incidence-weight slope; see _zslope_sites)
    zslope_sigma: object = None          # (3,) prior width on s_c (default the literature WLS σ_s)
    # PRIYA mean-flux model (replaces the 13 per-z τ₀ rungs): α(z)=τ₀·((1+z)/(1+z_p))^dτ₀, τ₀(z)=α·Kim07.
    # UNIFORM priors (Bird+2023 §2.7.1; arXiv:2509.18271). tau0_mu/tau0_sigma above are now legacy.
    tau0_amp_range: object = TAU0_AMP_RANGE   # PRIYA uniform prior on amplitude τ₀ (center 1.0=Kim)
    dtau0_range: object = DTAU0_RANGE         # PRIYA uniform prior on slope dτ₀ (center 0=Kim slope)
    tau0_pivot_z: float = TAU0_PIVOT_Z        # the (1+z)/(1+z_p) pivot (PRIYA z_p=3)
    # INFORMATIVE-τ₀ arm (2026-06-21): replace the UNIFORM prior on the two mean-flux sites with a
    # centered TruncatedNormal, truncated to the SAME physical range (tau0_amp_range/dtau0_range).
    # Each field is None (→ Uniform, the DEFAULT — byte-identical) OR a (mu, sigma) tuple (→
    # TruncatedNormal(mu, sigma, low=range[0], high=range[1])). Sampled via _sample_tau0_sites in
    # BOTH _legb_model and _legb_priors_only (identical site names/order). PRIYA/Bird+2023 note the
    # τ₀ amplitude is well measured by the data; this arm gives the closure a Kim-centered prior.
    tau0_amp_gauss: object = None             # None → Uniform; (mu, sigma) → TruncatedNormal on τ₀ amp
    dtau0_gauss: object = None                # None → Uniform; (mu, sigma) → TruncatedNormal on dτ₀
    # NUTS theta prior bounds in the UNIT cube (None → _THETA_UNIT_LO/_HI = IGM params restricted
    # to original PRIYA, n_s extended; see data.SAMPLING_LIMITS). Set both to 0/1 for the full box.
    theta_unit_lo: object = None              # (9,) lower bound on theta_unit
    theta_unit_hi: object = None              # (9,) upper bound on theta_unit
    # HIERARCHICAL HCD-incidence prior ("Option B", 2026-06-13): reparametrize the 3 independent
    # HCD α sites as ONE likelihood-constrained MULTIPLIER A_hcd (= the LLS prior verbatim) × two
    # prior-pinned RATIOS r_subdla/r_dla → α_LLS=A_hcd, α_subDLA=A_hcd·r_subdla, α_DLA=A_hcd·r_dla
    # (re-emitted as numpyro.deterministic under the EXISTING names). Collapses the flat
    # subDLA↔DLA exchange direction (the n_s-leak source) onto one fixed-shape additive amplitude
    # + prior-pinned shape ratios. The existing per-class z-slope s_c is UNCHANGED, so the ratio
    # ALREADY evolves as r_c(z)=r_c·((1+z)/(1+z_p))^(s_c−s_LLS) (the differential-CDDF-slope fix).
    hierarchical_hcd: bool = False            # OFF (default) → BYTE-IDENTICAL to the legacy 3-site code
    hcd_noncentered: bool = False             # LocScaleReparam-style non-centered fallback (same site
                                              # names → site-order test passes). Centered is correct
                                              # by the BENIGN orientation; this is the tested fallback.
    # The ratio prior centers/widths (2,) [r_subdla, r_dla]. CLOSURE (must-fix #1): centers from the
    # RAW sim w_c ratios (median(w_sub)/median(w_LLS), HCD_DLA_RESIDUAL_FRAC·median(w_DLA)/median(w_LLS))
    # so the held-out-sim closure TRUTH matches the center to within the per-sim CV; widths
    # (0.10, 0.12)·center × hcd_ratio_infl. None (default) → build_legb_ctx auto-derives.
    hcd_ratio_mu: object = None               # (2,) [r_subdla, r_dla] prior centers
    hcd_ratio_sigma: object = None            # (2,) [r_subdla, r_dla] prior widths
    hcd_ratio_infl: float = 1.0               # the MANDATORY width-scan knob {0.5,1,2,3}×
    # 2D AMPLITUDE×TILT submanifold variant of the hierarchical prior ("hcd_2d_tilt", 2026-06-14):
    # Option B (1D A_HCD × FIXED ratios) RELOCATED the n_s coupling onto the A_HCD center (the gate
    # diagnosis). The fix: give the HCD sector a GENUINE 2D submanifold = pivot AMPLITUDE A_HCD × a
    # GLOBAL z-TILT B_HCD, with the class-differential z-evolution FIXED —
    #   α_c(z) = A_HCD · r_c · ((1+z)/(1+z_p))^(B_HCD + δs_c),  δs_LLS ≡ 0.
    # A_HCD ~ TruncatedNormal(alpha_hcd_mu[0], alpha_hcd_sigma[0], low=0)  (the LLS prior verbatim);
    # B_HCD ~ Normal(hcd_btilt_mu, hcd_btilt_sigma)  (the GLOBAL HCD z-tilt — the DATA-constrained
    # 2nd submanifold dimension, orthogonal to the n_s k-tilt); r_subdla/r_dla = the SAME Option-B
    # ratios. The per-class slope s_c = B_HCD + δs_c REPLACES the marginalize_zslope sampling (so 2D
    # mode IGNORES marginalize_zslope). At B_HCD = hcd_btilt_mu the forward z-evolution MATCHES the
    # mock truth's full incidence slope HCD_INCIDENCE_SLOPE (~2.4 — the closure anchor). REQUIRES hierarchical_hcd=True
    # (it builds on the A_hcd × r reparam). OFF (default) → byte-identical (legacy OR Option B).
    hcd_2d_tilt: bool = False
    hcd_dslope: object = None                 # (3,) δs_c FIXED class-differential slopes (δs_LLS≡0)
    hcd_btilt_mu: float = None                # B_HCD prior center (= s_LLS slope center)
    hcd_btilt_sigma: float = None             # B_HCD prior width (= the LLS slope-prior width)
    fix_alpha_res: bool = False               # DIAGNOSTIC (default False → SAMPLE alpha_res, byte-
    #                                           identical to production). True → DO NOT sample the two
    #                                           res_corr-amplitude sites; pass the fixed no-op
    #                                           (alpha0=1, s=0) into the forward. Used by the decomp
    #                                           diagnostic to isolate alpha's contribution.
    # MODEL C — free 2-node flat-log f metal z-evolution (gate-b-data-nuisance). APPENDED at the END
    # (after fix_alpha_res, the last defaulted field) so positional construction does NOT shift. ALL
    # Model-C behaviour is gated behind the NEW static value metal_prior=="flatlog2node" (uniform /
    # flatlog stay BYTE-EXACT). Per metals_on leg IN ORDER, per ion, sample 2 node values of the metal
    # flux decrement f at z-nodes metal_node_z ~ dist.LogUniform(metal_fnode_lo, metal_fnode_hi); the
    # per-z amplitude is a(z)=f(z)/(1−⟨F⟩(z)) with ⟨F⟩(z)=exp(−tau0_vec[iz]) the SAMPLED mean flux.
    # SiII nodes only on legs in metal_siII_legs. metal_one_minus_F_ref (A1 scalar) is UNUSED here (the
    # per-z map uses tau0_vec, not the global scalar) — kept for flatlog back-compat.
    metal_node_z: tuple = (2.2, 4.2)          # STATIC ascending node redshifts (Chaves-Montero Sec 5.2)
    metal_fnode_lo: float = 0.003             # flat-log10 f per-node lower bracket
    metal_fnode_hi: float = 0.03              # upper bracket (spans eBOSS .006 / Ma .014 / DESI .016)
    metal_siII_legs: tuple = ("DESI",)        # legs that ALSO sample the SiII doublet f-nodes
    # MODEL C+ — FLOAT the sigmoid decorrelation SCALE k_SiIII/k_SiII (cup1d s_Lya_SiIII/s_Lya_SiII;
    # we sample the SCALE, not its inverse). Per metals_on leg, per ion, a 2-node LogUniform site (per
    # z log-interp like f) ~ dist.LogUniform(metal_knode_lo, metal_knode_hi). Bracket [1e-3, 0.1] s/km
    # spans the DESI Table-D2 best fit (k_SiIII~0.0074, k_SiII~0.0035) and the old fixed 0.05; in
    # cup1d s units that is exp([2,7]) ≈ 1/[0.135, 0.0009] (input_pipeline.set_baseline s_Lya_*).
    metal_knode_lo: float = 1e-3              # flat-log10 k (decorrelation scale, s/km) lower bracket
    metal_knode_hi: float = 0.1               # upper bracket
    sample_res: bool = False                  # option-b: sample the 2-param spectral-resolution f_res
    #                                           nuisance (b_res(z), forward-only). Default False → golden.
    f_res_amp_sigma: float = None             # option-b prior width on f_res_amp; None → tight F_RES_AMP_SIGMA
    #                                           (0.02). Set wider for arm-C (cup1d-faithful) or the eBOSS
    #                                           leg-matched prior (~0.05). Golden-safe (None → unchanged).
    res_corr_on: bool = True                  # Gate-A NORC: default True = production (res_corr applied).
    #                                           Mirrors ctx.mf.res_corr_on; False -> res_corr dropped +
    #                                           (in build_legb_ctx) KS capped at 0.045. A CONFIG field, NOT
    #                                           packed/sampled -> golden-safe (no _draws_matrix/truth_vec).


def _kim(z):
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


# ============================================================================ #
#  The certified production data-nuisance forward config, PER LEG.
# ============================================================================ #
# SINGLE SOURCE OF TRUTH consumed by BOTH the real fit (run_real_fit.build_real_ctx) AND the
# production SBC (run_prod_sbc_shard). The whole point of the pre-freeze wiring is that the real-fit
# forward == the SBC self-draw forward == the gate-certified forward; a copy-pasted per-leg config in
# two drivers is the exact drift risk this map removes. Per leg:
#   sample_res       : float the option-b spectral-resolution f_res (DESI/eBOSS; the RCINJ-certified arm).
#   f_res_amp_sigma  : that float's Normal(0, .) prior width -- DESI 0.02 / eBOSS 0.05 (leg-matched, the
#                      certified option-b widths; None where f_res is OFF).
#   metal_prior      : "flatlog2node" = the Gate-C Model C+ 2-node metals; "uniform" where metals are off.
#   metals           : whether this leg bears the SiIII/SiII metal forward + samples the metal sites.
# KS: f_res is ON via the certified KS echelle R_z instrument-f_res bracket (task #5, DONE) -- its own
# ks_kwargs (resolution_float + k_max, distinct from the DESI/eBOSS proxy) rather than the DESI/eBOSS
# f_res_amp_sigma widths; wide sigma=0.15 reflects the certified out-of-span/instrument-bracket study.
# No metals (its conservative covariance already subtracts+inflates metals/continuum/resolution).
PROD_FORWARD_BY_LEG = {
    "DESI":  dict(sample_res=True,  f_res_amp_sigma=0.02, metal_prior="flatlog2node", metals=True,  ks_kwargs=None),
    "eBOSS": dict(sample_res=True,  f_res_amp_sigma=0.05, metal_prior="flatlog2node", metals=True,  ks_kwargs=None),
    "KS":    dict(sample_res=True,  f_res_amp_sigma=0.15, metal_prior="uniform",      metals=False,
                  ks_kwargs=dict(resolution_float=True, k_max=0.065)),
}


def prod_forward_config(leg):
    """The certified production data-nuisance forward config for one leg name ('DESI'/'eBOSS'/'KS').

    Returns a fresh dict {sample_res, f_res_amp_sigma, metal_prior, metals, ks_kwargs} -- the SINGLE
    source both build_real_ctx and run_prod_sbc_shard consume so the real-fit and SBC forwards are
    identical. Deep-copied so nested mutation (e.g. ks_kwargs) cannot corrupt the module constant.
    Raises KeyError on an unknown leg (fail loud rather than silently mis-configure the forward)."""
    try:
        return copy.deepcopy(PROD_FORWARD_BY_LEG[leg])
    except KeyError:
        raise KeyError(f"no production forward config for leg {leg!r}; "
                       f"expected one of {list(PROD_FORWARD_BY_LEG)}")


# ============================================================================ #
#  The GLOBAL sim-convergence forward decision (NORC), the ONE reversal knob.
# ============================================================================ #
# res_corr is a SIM-convergence correction (small-box particle count), NOT a per-instrument nuisance
# -> it is a SINGLE global switch, not a per-leg PROD_FORWARD_BY_LEG entry (a per-leg field would make
# a one-leg flip representable but physically meaningless and would desync the fix_alpha_res invariant).
# Gate-A (2026-07-04) DEPLOYS NORC: drop res_corr (res_corr_on=False) + pin the 2 now-inert alpha_res
# sites (fix_alpha_res=True) + (in build_legb_ctx) auto-cap KS at k<=0.045. This constant is THE reversal
# knob for the 4-referee panel: flip to True to restore the res_corr/alpha_res AXIS on top of the CURRENT
# deployed forward on EVERY deployed path (real fit / SBC default / dnuis use_prod_forward) at once --
# a controlled A/B in which ONLY the res_corr treatment moves. It is NOT the literal 2026-06-16 pre-NORC
# config: KS keeps the later-certified echelle f_res float + k_max 0.065 (PROD_FORWARD_BY_LEG), DESI/eBOSS
# keep flatlog2node metals + their f_res floats; naive comparisons against the archived 2026-06-16 NUTS
# result would be confounded. Gated by the panel + freeze + PI sign-off before any unblind.
PROD_RES_CORR_ON = False


def prod_norc_forward():
    """The deployed GLOBAL sim-convergence (NORC) forward decision, as a fresh dict.

    Returns ``{"res_corr_on": PROD_RES_CORR_ON, "fix_alpha_res": not PROD_RES_CORR_ON}``. The
    ``fix_alpha_res == (not res_corr_on)`` invariant is encoded HERE, in ONE place, so no consumer can
    desync it. GLOBAL (not per-leg): every deployed driver (real fit / SBC / dnuis use_prod_forward)
    consumes this so the deployed forward's res_corr decision is defined exactly once. res_corr is a
    sim-convergence correction, not a per-instrument nuisance -> deliberately NOT keyed by leg."""
    return {"res_corr_on": bool(PROD_RES_CORR_ON), "fix_alpha_res": not bool(PROD_RES_CORR_ON)}


def forward_signature():
    """A stable sha256 hex digest over the MODULE-CONSTANT forward decision set (freeze/audit artifact).

    Canonical-JSON (sorted keys) over ``{PROD_FORWARD_BY_LEG, PROD_RES_CORR_ON}`` so the freeze task can
    record the deployed decision set in analysis.lock and audits can assert it. NOT covered (the freeze
    task must lock these separately, per the refactor handoff list): the KS k<=0.045 NORC auto-cap
    literal, mf_anchor_mult=5.0, the SIGMA_A0/SIGMA_S restore-arm alpha_res prior widths, the ensemble
    checkpoint set, and the prior constants. Consumed by NOTHING in the deployed inference path (never a
    per-mock discriminator -> avoids universal-clash). Values must stay JSON-native: a non-JSON value in
    PROD_FORWARD_BY_LEG raises TypeError (fail-loud) rather than being silently coerced."""
    payload = json.dumps({"PROD_FORWARD_BY_LEG": PROD_FORWARD_BY_LEG,
                          "PROD_RES_CORR_ON": bool(PROD_RES_CORR_ON)},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================ #
#  Build the production Leg-B context from final_fold0 + the xclass error vector.
# ============================================================================ #
def build_legb_ctx(*, ckpt=CKPT, error_vector=ERROR_VECTOR,
                   xclass_error_vector=XCLASS_ERROR_VECTOR, cemu_inflate=1.0,
                   metals_on=False, desi_kwargs=None, ks_kwargs=None,
                   with_eboss=False, eboss_kwargs=None,
                   use_xclass=True, with_mf=False, mf_fold=0, mf_with_floor=True,
                   mf_exclude_held=False, mf_target_hr_sim=None, mf_anchor_mult=5.0,
                   res_corr_on=True,
                   mf_shape=False, mf_shape_infl=1.0,
                   mf_shape_legs=("DESI", "KS"), mf_shape_npz=None,
                   mf_emucoh=False, mf_emucoh_infl=1.0,
                   mf_emucoh_legs=("DESI", "KS"), mf_emucoh_npz=None,
                   mf_emucoh_offdiag_only=False, sample_metals=False, sample_res=False,
                   coherent_res=False, coh_amp=1.0, f_res_amp_sigma=None, a_siiii_max=0.15,
                   metal_prior="uniform", metal_logf_lo=-11.0, metal_logf_hi=-2.0,
                   metal_one_minus_F_ref=None,
                   metal_node_z=(2.2, 4.2), metal_fnode_lo=0.003, metal_fnode_hi=0.03,
                   metal_siII_legs=("DESI",), metal_knode_lo=1e-3, metal_knode_hi=0.1,
                   hierarchical_hcd=False, hcd_noncentered=False, hcd_ratio_infl=1.0,
                   hcd_2d_tilt=False, ensemble_ckpts=None, survey=None):
    """Assemble the real DESI+KS legs + slice the production error vector onto each leg's
    z-bins. The cross-class ρ (``use_xclass=True``, the default; the matched
    ``error_vector_xclass.npz`` pair) is the production C_emu — the diagonal σ is carried too
    (for the diagonal/cross-class comparison figure). Returns ``(LegBCtx, dla_core_global)``.

    Metals default OFF for the closure (the SiIII model term is exercised separately; the
    sim truth carries no metals, so adding the metal MODEL term would be a misspecification
    arm, not the interior-τ₀ gate). ``desi_kwargs``/``ks_kwargs`` override the loader cuts.

    ``with_mf=True`` (default False → LF path): build + attach the production MF correction
    (``build_mf_correction(mf_fold)``) so the Leg-B forward AND the mock truth go through the
    certified through-MF path (the gate invariant). ``mf_with_floor=True`` also attaches the
    ``MFFloor`` (the LF→HR + n_s-edge C_emu floor on the small-scale leg). The MF backbone for
    ``mf_fold=0`` is byte-identical to the ``final_fold0`` ``ctx.model``, so the bare-model
    P_filt and the frozen ``mf.lf_model`` agree (the gate faithfulness invariant).
    """
    if ensemble_ckpts is not None:
        # production SBC: the N-seed ensemble forward (mean of P_filt over members). The
        # single-ckpt path (ensemble_ckpts=None) is UNCHANGED.
        from hcd_analysis.emulator.ensemble import load_ensemble
        model, meta, norm = load_ensemble(list(ensemble_ckpts))
    else:
        model, meta, norm = T.load_checkpoint(ckpt)
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}

    ev = np.load(error_vector, allow_pickle=True)
    sigma = ev["sigma"]                            # (4,K,Zb,Tb)
    C4, K, Zb, Tb = sigma.shape
    alpha_centres = jnp.asarray(ev["tau0_band_centres"])
    z_band_edges = ev["z_band_edges"]              # (Zb+1,)
    rho = None
    if use_xclass:
        evx = np.load(xclass_error_vector, allow_pickle=True)
        rho = evx["rho"]                           # (4,4,K,Zb,Tb)
        assert rho.shape[2:] == (K, Zb, Tb), \
            f"xclass rho {rho.shape} incompatible with (K,Zb,Tb)=({K},{Zb},{Tb})"
        assert np.allclose(np.asarray(evx["tau0_band_centres"]), np.asarray(alpha_centres)), \
            "xclass τ₀-band centres differ from the diagonal error vector"

    # ARM-D (coherent-cov, no forward float) is a DISTINCT covariance treatment from option-b (sample_res):
    # it does NOT set ctx.sample_res (no f_res site), only the loader's resolution_coherent cov mode.
    if coherent_res and sample_res:
        raise ValueError("build_legb_ctx: sample_res (option-b) and coherent_res (arm-D) are mutually exclusive")
    desi = DL.load_desi_leg(metals_on=metals_on, resolution_float=sample_res,
                            resolution_coherent=coherent_res, resolution_coh_amp=coh_amp, **(desi_kwargs or {}))
    # Gate-A NORC: when res_corr is dropped, cap KS at k<=0.045 (the residual high-k
    # particle-convergence uncertainty is then un-marginalized on KS, whose k_max reaches
    # furthest above the res_corr anchor 5*k_box(z) -- ~3.5x the typical anchor value, measured).
    # CORRECTED 2026-07-10 (PR#14 panel FIX 6a): DESI's k_max ALSO sits ABOVE the anchor (~2.3x
    # typical, not below it as an earlier version of this comment claimed) but was separately
    # measured safe by the RCINJ res_corr injection-recovery gate (PASS on DESI/KS/eBOSS,
    # scripts/analyze_res_corr_injection.py). eBOSS's k_max sits NEAR (barely above) the anchor,
    # so it is the one leg genuinely little-affected by dropping res_corr. Overridable via an
    # explicit ks_kwargs k_max.
    _ks_kw = dict(ks_kwargs or {})
    if not res_corr_on and "k_max" not in _ks_kw:
        _ks_kw["k_max"] = 0.045
    ks = DL.load_ks_leg(**_ks_kw)
    legs = [desi, ks]
    # eBOSS DR14 (Chabanier+2019) — opt-in third leg (the low-k production shakedown). Its own
    # flag defaults (metals_on=True SiIII / dla_forward_frac=0 / no MF floor — it's a LARGE-scale
    # leg, NOT in mf_shape_legs/mf_emucoh_legs below) apply unless overridden via eboss_kwargs.
    if with_eboss:
        # eBOSS option-b uses the "rescale" cov mode (its cov is corr⊙σσᵀ with resolution baked into σ, so
        # rebuild σ'²=σ²−res²; reference-verified 2026-07-02, SPD +0.05). sample_res wires DESI (rank1) + eBOSS.
        legs.append(DL.load_eboss_leg(resolution_float=sample_res, resolution_coherent=coherent_res,
                                      resolution_coh_amp=coh_amp, **(eboss_kwargs or {})))

    z_global = np.unique(np.round(np.concatenate([leg.z for leg in legs]), 6))

    # slice the error vector onto each leg's z-bins (digitize into the z-band edges).
    sigma_zb_per_leg, rho_zb_per_leg = {}, {}
    for leg in legs:
        zb_of_z = np.clip(np.digitize(np.asarray(leg.z), z_band_edges[1:-1]), 0, Zb - 1)
        sigma_zb_per_leg[leg.name] = jnp.asarray(
            np.stack([sigma[:, :, zb_of_z[i], :] for i in range(leg.n_z)]))   # (n_z,4,K,Tb)
        if rho is not None:
            rho_zb_per_leg[leg.name] = jnp.asarray(
                np.stack([rho[:, :, :, zb_of_z[i], :] for i in range(leg.n_z)]))  # (n_z,4,4,K,Tb)

    # per-leg DLA core (cache delta[,2]) on each leg z. The cache DLA core is the SAME for
    # all sims at a (z,τ₀) cell up to a small per-row spread; we take a representative core
    # from the held-out pool below (closure_legb assembles it per-mock at the sim's rows).
    # Here we set a fiducial (mean held-out) core per leg z; the mock overrides with the
    # actual sim's core when it builds the truth (so the FORWARD core and the TRUTH core are
    # the SAME sim's — they cancel in the emulator-error sizing).
    d = load_cache(CACHE_PATH)
    dla_core_leg = _fiducial_dla_core_per_leg(d, legs, cache_k=meta["kfkms"])

    cache_k = jnp.asarray(meta["kfkms"])

    # priors on the GLOBAL z grid: Becker+2013 τ₀ (production anchor) + HCD incidence.
    tau0_mu, tau0_sigma = meanflux_tau0_prior(jnp.asarray(z_global), center="becker13")
    # HCD incidence prior from the cache's structural w_c AT THE z=3 PIVOT (LLS,subDLA,DLA).
    # CENTER-CONSTRUCTION FIX (PI 2026-06-17): build the pivot from the z=3 STRUCTURAL w_c, NOT the
    # all-z median nanmedian(w_c_cache[:,1:]). Because w_c rises monotonically with z, the all-z median
    # (LLS 0.274) equals the z≈3.6 value → consumed as the z=3 pivot it over-stated the LLS center ~1.45×
    # (α_pivot 0.291 vs the z=3-consistent ~0.194–0.200), overshooting the lit dN/dX 2.05× @z2.4 = the
    # LLS→n_s leak. hcd_pivot_wc_and_xbar restricts the SAME cache to the z=3 pivot rows. See the dN/dX
    # low-z overshoot bug. γ_LLS (forward z-slope) and σ_LLS (width) are UNCHANGED.
    w_c_med, Xbar_z3 = hcd_pivot_wc_and_xbar(d, z_pivot=HCD_Z_PIVOT)   # (3,) z=3 structural w_c, Xbar(z=3)
    # survey=None (closure/SBC) → cosmic-average LLS pin (unchanged); survey="DESI"/"KS"/… (real fit)
    # → the per-survey LLS center+width pin (DESI 1.0×/σ0.15, KS 2.5×/σ0.40; PI re-determination
    # 2026-06-17). The PI WIDTH RULE 1× value is the lit measurement error (σ_LLS=0.15); the 2×
    # cosmic-variance hedge is HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X (toggle here if a hedge ctx is needed).
    alpha_mu, alpha_sd = hcd_incidence_prior(jnp.asarray(w_c_med), z=HCD_Z_PIVOT, survey=survey)
    # REAL-FIT LLS center: prefer the lit dN/dX law DIRECTLY (alt-(b)) — α_LLS(z=3) from
    # A=0.0201·(1+z)^2.127 through the EXACT w_c map (hcd_lls_realfit_alpha_center), round-tripping the
    # lit dN/dX to <0.34% (≈0.194×boost). On the REAL FIT PRIYA≠data, so the LLS center must track the
    # literature dN/dX, not the sim's z=3 w_c·(lit/sim). The CLOSURE/SBC (survey=None) keeps the sim z=3
    # w_c center (its held-out-sim mocks carry the sim incidence). subDLA/DLA centers are unchanged.
    if survey is not None:
        _boost = HCD_LLS_SURVEY_BOOST.get(survey, 1.0)
        alpha_mu = alpha_mu.at[0].set(hcd_lls_realfit_alpha_center(Xbar_z3, z=HCD_Z_PIVOT, boost=_boost))
        # keep the σ/μ width invariant (PI WIDTH RULE: 1× lit measurement error) at the new center.
        _fl = HCD_LLS_SURVEY_FRAC_SIGMA.get(survey, float(HCD_PRIOR_FRAC_SIGMA[0]))
        alpha_sd = alpha_sd.at[0].set(_fl * alpha_mu[0])
    # PIVOT GUARD (PI's explicit ask): the LLS α-pivot center MUST be the z=3 value, NOT the all-z
    # median (z≈3.6). A future revert to nanmedian(w_c_cache[...all z...]) trips this at build time.
    _guard_boost = HCD_LLS_SURVEY_BOOST.get(survey, 1.0) if survey is not None else 1.0
    assert_hcd_pivot_z3(float(np.asarray(alpha_mu)[0]), z=HCD_Z_PIVOT,
                        where=f"build_legb_ctx survey={survey}", boost=_guard_boost)

    # REAL-FIT LLS forward z-slope (litWLS, PI re-determination 2026-06-17): when ``survey`` is given
    # (a real-data fit), the LLS forward z-evolution must track the literature WLS slope γ_LLS=2.127
    # (the lit dN/dX_LLS(z) power-law), NOT the sim incidence slope 2.465 (which over-predicts low-z
    # LLS by +62–87% vs lit/truth → the LLS→n_s leak). subDLA/DLA keep the sim incidence slope. The
    # CLOSURE/SBC path (survey=None, sim-truth mocks) keeps zslope_mu=None → _zslope_sites centers on
    # HCD_INCIDENCE_SLOPE=(2.465,…) (the sim-truth slope the held-out-sim mock carries) — UNCHANGED.
    # γ=2.127 > the forward z-slope guard floor 1.5, so this passes _assert_forward_zslope_center.
    survey_zslope_mu = None
    if survey is not None:
        _incid = np.asarray(HCD_INCIDENCE_SLOPE, float)
        survey_zslope_mu = jnp.asarray([HCD_LLS_REALFIT_ZSLOPE, _incid[1], _incid[2]])
        _assert_forward_zslope_center(survey_zslope_mu, f"build_legb_ctx survey={survey} litWLS zslope_mu")

    # HIERARCHICAL HCD ratio-prior centers/widths (must-fix #1, the LOAD-BEARING fix). The ratio
    # centers are derived from the z=3 PIVOT sim w_c (``w_c_med`` is now the z=3 structural weight —
    # the CENTER-construction fix re-derived it at the pivot, PI 2026-06-17; they previously inherited
    # the same all-z-median bug as the LLS/subDLA pivot) —
    # r_sub = w_subDLA(z3)/w_LLS(z3), r_dla = HCD_DLA_RESIDUAL_FRAC·w_DLA(z3)/w_LLS(z3)
    # — NOT from alpha_hcd_mu (which bakes in the lit/sim 1.06/1.00/1.34 offset). This makes the
    # closure TRUTH (= the held-out sim's w_sub/w_LLS and 0.10·w_DLA/w_LLS, make_legb_mock:716-717)
    # match the prior CENTER to within the per-sim CV — a tight prior at an OFFSET center would
    # re-create the exact center-bias this redesign kills, relocated onto r (verified: alpha_hcd_mu
    # centers put the closure truth at −0.81σ subDLA / +2.64σ DLA, and fail SBC). Widths are
    # (0.10, 0.12)·center × hcd_ratio_infl (the CLOSURE defense-in-depth widths, ≥ the bare 7%/10%
    # CV; the real fit inflates to ~25–30% for CDDF-shape uncertainty via hcd_ratio_infl).
    r_sub_center = float(w_c_med[1] / w_c_med[0])
    r_dla_center = float(HCD_DLA_RESIDUAL_FRAC * w_c_med[2] / w_c_med[0])
    hcd_ratio_mu = jnp.asarray([r_sub_center, r_dla_center])
    hcd_ratio_sigma = jnp.asarray([0.10 * r_sub_center, 0.12 * r_dla_center]) * float(hcd_ratio_infl)

    # 2D AMPLITUDE×TILT submanifold (hcd_2d_tilt): the GLOBAL HCD z-tilt B_HCD + FIXED class-
    # differential slopes δs_c. CLOSURE defaults use the FULL per-class incidence z-slope the mock
    # TRUTH actually carries — the 60-sim-population median of d ln w_c(z)/d ln(1+z) (HCD_INCIDENCE_SLOPE,
    # measured 2026-06-14), NOT the lit/sim-RATIO slope HCD_LIT_OVER_SIM_SLOPE=(0.95,…) which is a
    # DIFFERENT object (the mock w_c(z) evolves at LLS slope ~2.4, not 0.95; a B-center at 0.95 would
    # put the truth at ~2.8σ and fight the data, re-leaking into n_s — the impl-review finding):
    #   δs_c       = HCD_INCIDENCE_SLOPE − HCD_INCIDENCE_SLOPE[0] = (0.0, +0.29, −0.10)
    #   btilt_mu   = HCD_INCIDENCE_SLOPE[0] = 2.465  (so at B_HCD=btilt_mu, the forward z-evolution
    #                MATCHES the truth; the B_HCD coverage-truth = btilt_mu = the population LLS slope)
    #   btilt_sigma= ZSLOPE_PRIOR_SIGMA[0] = 0.52  (kept WIDE so B floats + the data constrain it)
    # REAL-FIT (later): swap HCD_INCIDENCE_SLOPE → the literature dN/dX slopes (HR/observed). The
    # marginalize_zslope default forward (centered 0.95) is a SEPARATE, mostly-harmless mis-centering:
    # the data recover the slope ~2.5 and n_s is ⊥ slope (scripts/diag_hcd_zslope_nsbias.py).
    hcd_2d_tilt = bool(hcd_2d_tilt)
    hcd_dslope = hcd_btilt_mu = hcd_btilt_sigma = None
    if hcd_2d_tilt:
        if not hierarchical_hcd:
            raise ValueError("hcd_2d_tilt=True requires hierarchical_hcd=True "
                             "(the 2D submanifold builds on the A_hcd × r reparam)")
        _slope = np.asarray(HCD_INCIDENCE_SLOPE, float)
        hcd_dslope = jnp.asarray(_slope - _slope[0])             # (3,) δs_c, δs_LLS≡0
        hcd_btilt_mu = float(_slope[0])                          # B_HCD center = full LLS incidence slope
        hcd_btilt_sigma = float(ZSLOPE_PRIOR_SIGMA[0])           # B_HCD width (wide → B floats)
        # GUARD: the 2D B_HCD center is the incidence slope (~2.4), never the lit/sim ratio (~0.95).
        _assert_forward_zslope_center(hcd_btilt_mu, "build_legb_ctx hcd_btilt_mu (2D-tilt)")

    mf_obj = mf_floor_obj = None
    if with_mf:
        # mf_target_hr_sim (Task 1.5): with mf_exclude_held, drop EXACTLY this one HR sim from
        # the MF head fit (genuine leave-ONE-out) instead of the whole LF fold group. None →
        # whole-group held (back-compat); a NO-OP when mf_exclude_held=False.
        # mf_anchor_mult (DIAGNOSTIC, default 5.0 = production anchor / byte-identical):
        # 0.0 DISABLES the res_corr low-k anchor (raw clamped table) — used by the decomp
        # diagnostic to isolate the anchor's contribution to the coherent n_s bias.
        mf_obj, mf_floor_obj = build_mf_correction(
            fold=mf_fold, with_floor=mf_with_floor, exclude_held_hr=mf_exclude_held,
            target_hr_sim=mf_target_hr_sim, anchor_mult=mf_anchor_mult,
            res_corr_on=res_corr_on)

    # SHAPE-AWARE MF floor (Phase-5a): the per-leg fractional LOSO-eps outer-product covariance
    # (precomputed once, θ-blind). Fired on the named legs (default DESI+KS) when with_mf+mf_shape.
    mf_shape_per_leg = None
    if with_mf and mf_shape:
        shape_tab = DL.load_mf_shape(mf_shape_npz) if mf_shape_npz else DL.load_mf_shape()
        mf_shape_per_leg = {leg.name: DL.mf_shape_cov_for_leg(shape_tab, leg)
                            for leg in legs if leg.name in set(mf_shape_legs)}

    # 60-sim LF-emulator-coherence term — NOT gated on with_mf (it is an LF-emulator residual,
    # measured from the 8-fold/60-sim LOSO, valid on the LF reference path too).
    mf_emucoh_per_leg = None
    if mf_emucoh:
        ec_tab = DL.load_mf_emucoh(mf_emucoh_npz) if mf_emucoh_npz else DL.load_mf_emucoh()
        mf_emucoh_per_leg = {leg.name: DL.mf_shape_cov_for_leg(ec_tab, leg)
                             for leg in legs if leg.name in set(mf_emucoh_legs)}

    # METAL FLAT-LOG f PRIOR (Task A1): the ONE global scalar 1−F_ref mapping f→amplitude a=f/(1−F_ref)
    # for the opt-in flatlog metal prior. When None, derive it from the fiducial mean flux the forward
    # uses at the prior center (tau0_amp=1, dtau0=0 ⇒ τ₀(z)=Kim07(z)): F_ref = mean_z exp(−Kim(z)) on
    # the union z-grid. A CONSTANT scale on a log-uniform ⇒ shape-neutral (the prior stays flat-log f).
    if metal_one_minus_F_ref is None:
        _F_z = np.exp(-np.asarray(_kim(jnp.asarray(z_global))))   # exp(−Kim07(z)) at the fiducial
        metal_one_minus_F_ref = float(1.0 - np.mean(_F_z))
    else:
        metal_one_minus_F_ref = float(metal_one_minus_F_ref)

    ctx = LegBCtx(
        model=model, pf_stats=pf, dla_core_leg=dla_core_leg, legs=legs, cache_k=cache_k,
        z_global=z_global, sigma_zb_per_leg=sigma_zb_per_leg,
        rho_zb_per_leg=(rho_zb_per_leg if rho is not None else None),
        alpha_centres=alpha_centres, tau0_mu=tau0_mu, tau0_sigma=tau0_sigma,
        alpha_hcd_mu=jnp.asarray(alpha_mu), alpha_hcd_sigma=jnp.asarray(alpha_sd),
        cemu_inflate=float(cemu_inflate), mf=mf_obj, mf_floor=mf_floor_obj,
        mf_shape_per_leg=mf_shape_per_leg, mf_shape_infl=float(mf_shape_infl),
        mf_emucoh_per_leg=mf_emucoh_per_leg, mf_emucoh_infl=float(mf_emucoh_infl),
        mf_emucoh_offdiag_only=bool(mf_emucoh_offdiag_only),
        sample_metals=bool(sample_metals), sample_res=bool(sample_res),
        f_res_amp_sigma=(None if f_res_amp_sigma is None else float(f_res_amp_sigma)),
        a_siiii_max=float(a_siiii_max),
        metal_prior=str(metal_prior), metal_logf_lo=float(metal_logf_lo),
        metal_logf_hi=float(metal_logf_hi), metal_one_minus_F_ref=metal_one_minus_F_ref,
        metal_node_z=tuple(metal_node_z), metal_fnode_lo=float(metal_fnode_lo),
        metal_fnode_hi=float(metal_fnode_hi), metal_siII_legs=tuple(metal_siII_legs),
        metal_knode_lo=float(metal_knode_lo), metal_knode_hi=float(metal_knode_hi),
        hierarchical_hcd=bool(hierarchical_hcd), hcd_noncentered=bool(hcd_noncentered),
        hcd_ratio_mu=hcd_ratio_mu, hcd_ratio_sigma=hcd_ratio_sigma,
        hcd_ratio_infl=float(hcd_ratio_infl),
        hcd_2d_tilt=hcd_2d_tilt, hcd_dslope=hcd_dslope,
        hcd_btilt_mu=hcd_btilt_mu, hcd_btilt_sigma=hcd_btilt_sigma,
        # REAL-FIT (survey != None) litWLS LLS forward z-slope center (2.127, sim_sub, sim_DLA);
        # survey=None (closure/SBC) → None → _zslope_sites centers on HCD_INCIDENCE_SLOPE (sim-truth).
        zslope_mu=survey_zslope_mu,
        res_corr_on=bool(res_corr_on))
    return ctx, d


def build_mf_correction(fold=0, *, rank1=True, exclude_held_hr=False,
                        with_floor=True, floor_npz=None, target_hr_sim=None,
                        anchor_mult=5.0, res_corr_on=True):
    """Build the production MF correction (resolved separable + rank-1 FixedMeanHead +
    fixed res_corr) on the LF native cache grid for a fold, REUSING the certified gate
    construction (scripts/diag_emu_bias_allfolds_mf.build_mf_for_fold). Returns
    ``(mf, mf_floor)`` ready to thread into ``LegBCtx`` / ``predict_P_obs_on_leg``.

    ``mf.eval_logk == lf_logk`` (the cache grid) so the correction == the gate's measured
    forward. ``exclude_held_hr=True`` does HF-LOSO (the head is fit EXCLUDING the fold's
    held-out HR sims) — the honest generalization arm; the default (False) fits the head on
    ALL HR sims (the PRODUCTION correction, which is what the closure forward should use:
    the closure tests the emulator-as-likelihood with the production MF, not the LOSO floor).
    ``with_floor=True`` also loads the certified ``MFFloor`` (the LF→HR + n_s-edge C_emu
    floor) from ``mf_cemu_floor.npz``.

    ``target_hr_sim`` (Task 1.5, default None → whole-group back-compat): when set together
    with ``exclude_held_hr=True``, the head is fit EXCLUDING ONLY this one HR sim (genuine
    leave-ONE-out), independent of the LF fold group. Two HR sims can share an LF fold group
    (fold 2 → {ns0.859, ns0.885}; fold 6 → {ns0.972, ns0.979}); the whole-group held set then
    silently does leave-TWO-out → the head is fit on 4 HR sims → a worse correction → an
    inflated apparent bias. ``target_hr_sim`` pins the held set to EXACTLY {target_hr_sim} so
    the fold-mate is retained (5 HR sims, not 4). ``target_hr_sim=None`` keeps the whole-group
    held set (bit-identical back-compat); it is a NO-OP when ``exclude_held_hr=False``."""
    from hcd_analysis.emulator import multifidelity as MF
    lf_cache = MF.load_cache(MF.LF_CACHE)
    hr_cache = MF.load_cache(MF.HR_CACHE)
    pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
    fold_model, _fold_meta, fold_norm, lf_logk = MF.load_lf_backbone(fold)
    eval_logk = np.asarray(lf_logk)

    # HF-LOSO row mask (optional): exclude the held-out HR sim(s) from the head fit. The held
    # set is whole-group (target_hr_sim=None, back-compat) or EXACTLY {target_hr_sim} (true LOO).
    train_rows = None
    if exclude_held_hr:
        from .data import load_cache as _lc
        d = _lc(CACHE_PATH)
        held = held_hr_set(d, hr_cache["sim_name"], fold=fold, target_hr_sim=target_hr_sim)
        hr_row_sim = np.array([s.decode() if isinstance(s, bytes) else s
                               for s in hr_cache["sim_name"][[h for h, _ in pairs]]])
        train_rows = np.where(~np.isin(hr_row_sim, list(held)))[0] if held else None

    tg = MF.measure_delta_targets(lf_cache, hr_cache, fold_model, fold_norm, lf_logk,
                                  eval_logk, pairs)
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    comp = MF.fixed_mean_table_resolved(tg, log_rho, train_mask_rows=train_rows)
    a_k = comp["a_k"] if rank1 else np.zeros_like(comp["a_k"])
    head = MF.FixedMeanHead(
        comp["gbar_z_tab"], comp["z_tab"], resolved=True,
        gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
        a_k=a_k, u_z=comp["u_z"], u_tau=comp["u_tau"])
    mf = MF.build_multifidelity(fold_model, fold_norm, lf_logk, head,
                                eval_logk=eval_logk, log_rho=log_rho, delta_mode="none",
                                anchor_mult=anchor_mult, res_corr_on=res_corr_on)
    mf_floor = None
    if with_floor:
        mf_floor = (DL.load_mf_floor(floor_npz) if floor_npz
                    else DL.load_mf_floor())
    return mf, mf_floor


def _fiducial_dla_core_per_leg(d, legs, cache_k):
    """Mean held-out DLA core (cache ``delta[,2]``) per leg z, interpolated onto the leg-z
    cache rows. Used only as a FALLBACK; the per-mock truth supplies its own sim's core."""
    cache_k = np.asarray(cache_k)
    z_grid = d["z_grid"]
    delta = d["delta"]                              # (R,3,K)
    out = {}
    for leg in legs:
        core_z = np.zeros((leg.n_z, len(cache_k)))
        for iz, zz in enumerate(leg.z):
            sel = np.where(np.isclose(z_grid, zz, atol=0.05))[0]
            if sel.size == 0:
                sel = np.array([int(np.argmin(np.abs(z_grid - zz)))])
            core_z[iz] = np.nanmean(delta[sel, 2], axis=0)
        out[leg.name] = jnp.asarray(np.nan_to_num(core_z))
    return out


# ============================================================================ #
#  Mock-truth construction:  held-out-sim P1D → leg grids.
# ============================================================================ #
def held_out_sims(d, fold=0):
    """The fold-0 held-out validation SIMS (the production model final_fold0 never saw).
    Returns the sorted unique sim names in the val split."""
    _tr, va, _ho = make_splits(d, fold)
    return sorted(set(np.asarray(d["sim_name"])[va])), va


def held_hr_set(d, hr_sim_names, fold=0, target_hr_sim=None):
    """The set of HR sims to HOLD OUT of the MF head fit for ``fold`` (Task 1.5).

    The MF-correction HF-LOSO row mask drops the rows whose HR sim is in this set, so the
    RETAINED HR sims (the complement against the 6 HR sims) are what the head is fit on.

    * ``target_hr_sim is None`` → BACK-COMPAT (whole-group held): the LF fold-group held set
      intersected with the HR sims — byte-for-byte what ``build_mf_correction`` computed
      before, i.e. ``{s for s in held_out_sims(d, fold)[0] if s in <decoded hr_sim_names>}``.
      For a leave-TWO-out fold (two HR sims share the LF group: fold 2 / fold 6) this is a
      2-element set — the legacy (buggy) leave-two-out behavior, preserved on purpose.
    * ``target_hr_sim is not None`` → TRUE LOO: returns EXACTLY ``{target_hr_sim}`` (drop only
      the one sim under test), independent of the LF fold group, so the fold-mate is retained
      (5 HR sims fit, not 4). ``target_hr_sim`` must be one of the HR sims.

    ``hr_sim_names`` = the HR cache sim-name array (``hr_cache["sim_name"]``; bytes or str
    entries both accepted, decoded internally). PURE (no I/O): the caller supplies ``d`` and
    the HR sim-name array.
    """
    hr_sims = set(s.decode() if isinstance(s, bytes) else str(s) for s in hr_sim_names)
    if target_hr_sim is not None:
        target = target_hr_sim.decode() if isinstance(target_hr_sim, bytes) else str(target_hr_sim)
        # raise (not assert: must survive python -O) — a typo'd target would otherwise silently
        # fit the WRONG MF correction and corrupt the cert.
        if target not in hr_sims:
            raise ValueError(
                f"target_hr_sim {target!r} is not an HR sim; HR sims = {sorted(hr_sims)}")
        # defensive: the target must actually be held out of THIS fold's LF val group, else the cert
        # would fit a fold-N backbone with the wrong sim held (silent target<->fold mis-pairing).
        sims, _ = held_out_sims(d, fold=fold)
        fold_hr = set(s.decode() if isinstance(s, bytes) else str(s) for s in sims) & hr_sims
        if target not in fold_hr:
            raise ValueError(
                f"target_hr_sim {target!r} is not in fold {fold}'s held HR group {sorted(fold_hr)} "
                "— target<->fold mis-pairing")
        return {target}
    sims, _ = held_out_sims(d, fold=fold)
    decoded_sims = (s.decode() if isinstance(s, bytes) else str(s) for s in sims)
    return set(s for s in decoded_sims if s in hr_sims)


def make_truth_from_sim(d, sim_name, fold=0, tau0_anchor="priya", mf=None, lls_truth_boost=1.0):
    """Assemble a multi-z SIM-TRUTH from one held-out sim's cache rows.

    The truth P1D per z is the cache's MEASURED contaminated power at α=w_c (the sim's own
    contamination): ``P_obs_true = Σ_c coef_c·P_cls_c`` with coef=[1−Σw, w_LLS, w_sub, w_DLA],
    P_cls = cache P_filt (+ the per-row DLA core on the DLA class) — EXACTLY the held-out
    residual construction in scripts/diag_cemu_validation.py (NOT an emulator prediction).

    ``mf`` (opt-in, default None → LF-resolution truth): a ``MultiFidelity`` (eval grid ==
    cache grid). When set, the SAME fixed per-class MF correction ``exp(g + log res_corr)``
    the FORWARD applies is applied to the truth P_filt at each row's (θ, z, τ₀) BEFORE the
    clean+excess combination — the GATE INVARIANT (scripts/diag_emu_bias_allfolds_mf.py
    §emu_bias_one_sim): a θ-blind correction applied to BOTH forward and truth cancels in ΔP
    up to its (zero) θ-dependence, so the closure n_s/A_p bias is unchanged by construction.
    The mock is then an MF-resolution draw, matching "re-run the closure through the MF
    forward" — NOT a forward-vs-truth resolution mismatch.

    The LF cache stores MULTIPLE τ₀-ladder (mean-flux rescale) rows per (sim, z). For one
    mock we need ONE (P1D, τ₀) per z; we select, per z, the single ladder row whose τ₀ is
    CLOSEST to the production observational anchor (``tau0_anchor="becker13"`` ⟨τ_eff⟩(z) —
    the interior-τ₀ regime the data actually visits, plan A1 PRIMARY). This makes the mock a
    realistic interior-regime draw, not a ladder extreme. ``tau0_anchor=None`` keeps the
    central (median-τ₀) ladder row instead. ``tau0_anchor="extreme_hi"`` /``"extreme_lo"``
    select the most-/least-absorption ladder RUNG (max/min τ₀) — the STEP-A M2 probe of the
    ladder extreme where the τ₀×cosmology interaction (and the emulator residual) is hardest.

    Pools the sim's rows over z (data range z∈[2.2,4.6], finite P_filt). Returns dict:
      z          (nZs,)         the sim's available redshifts (ascending, one per z);
      P_obs_true (nZs, K)       per-z DLA-MASKED baseline P1D on the cache grid: the
                                contaminated Tier-P with the DLA class held at the CLEAN level
                                (the τ=1e6-filtered + LLS/subDLA-contaminated forest, no DLA
                                excess). The per-leg DLA residual is ADDED in make_legb_mock;
      dla_excess_true (nZs, K)  the per-z FULL DLA excess add-back w_DLA·(P_DLA^unf − P_clean),
                                P_DLA^unf = cache P_filt[DLA] + the DLA core. make_legb_mock adds
                                TRUTH_DLA_FRAC[leg]·this to the baseline (DESI 0.10 / KS 0.0) —
                                the §0c per-leg unmasked-DLA residual in the TARGET MOCK;
      params_unit (9,)          the truth θ (unit-cube; same for all the sim's rows);
      tau0       (nZs,)         the truth τ₀(z) (cache tau0 = −ln⟨F⟩, the selected ladder row);
      dla_core   (nZs, K)       the per-z DLA core (cache delta[,2]); used by the FORWARD's DLA
                                excess template (and the dla_excess_true add-back above);
      w_c        (3,)           the sim's structural α=w_c (its own contamination, z-mean).
    """
    sim_name = str(sim_name)
    _tr, va, _ho = make_splits(d, fold)
    names = np.asarray(d["sim_name"])
    rows = va[names[va] == sim_name]
    z_grid = d["z_grid"]
    P_filt = d["P_filt"]                            # (R,4,K)
    delta = d["delta"]                              # (R,3,K)
    w_c = d["w_c_cache"]                            # (R,4)
    tau0_all = d["tau0"]
    params_unit = d["params_unit"]

    # in-range, finite rows of this sim.
    cand = [int(r) for r in rows
            if (2.2 - 1e-6 <= z_grid[r] <= 4.6 + 1e-6) and np.isfinite(P_filt[r]).all()]
    if not cand:
        raise ValueError(f"sim {sim_name!r} has no in-range finite rows")

    # per z, select ONE ladder row: the τ₀ closest to the production anchor (interior regime).
    cand = np.array(cand)
    z_of = np.round(z_grid[cand], 4)
    keep_rows = []
    for zz in np.unique(z_of):
        sub = cand[z_of == zz]
        if isinstance(tau0_anchor, (tuple, list)):   # PRIYA-curve anchor (tau0_amp, dτ₀)
            amp_t, dt_t = float(tau0_anchor[0]), float(tau0_anchor[1])
            target = float(tau0_alpha_priya(jnp.asarray(float(zz)), amp_t, dt_t)
                           * _kim(jnp.asarray(float(zz))))
            pick = sub[int(np.argmin(np.abs(tau0_all[sub] - target)))]
        elif tau0_anchor == "priya":                 # PRIYA central curve τ₀=1,dτ₀=0 (=Kim) — the
            target = float(_kim(jnp.asarray(float(zz))))   # default for the PRIYA 2-param model
            pick = sub[int(np.argmin(np.abs(tau0_all[sub] - target)))]
        elif tau0_anchor == "becker13":
            target = float(becker13_tau0(jnp.asarray(float(zz))))
            pick = sub[int(np.argmin(np.abs(tau0_all[sub] - target)))]
        elif tau0_anchor == "extreme_hi":            # most-absorption ladder rung (max τ₀)
            pick = sub[int(np.argmax(tau0_all[sub]))]
        elif tau0_anchor == "extreme_lo":            # least-absorption ladder rung (min τ₀)
            pick = sub[int(np.argmin(tau0_all[sub]))]
        else:                                        # central (median-τ₀) ladder row
            pick = sub[int(np.argmin(np.abs(tau0_all[sub] - np.median(tau0_all[sub]))))]
        keep_rows.append(int(pick))
    keep_rows = np.array(sorted(keep_rows, key=lambda r: z_grid[r]))

    z = z_grid[keep_rows]
    K = P_filt.shape[-1]
    P_obs = np.zeros((len(keep_rows), K))
    dla_excess = np.zeros((len(keep_rows), K))
    dla_core = np.zeros((len(keep_rows), K))
    th_truth = jnp.asarray(params_unit[keep_rows[0]])
    z_unit_rows = (z_grid[keep_rows] - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    for i, r in enumerate(keep_rows):
        a = np.asarray(w_c[r, 1:], dtype=float).copy()  # (3,) [LLS,sub,DLA] the sim's contamination
        a[0] = a[0] * lls_truth_boost                # inject a survey-level LLS excess into the MOCK
        coef = np.concatenate([[1.0 - a.sum()], a])  # (4,) clean fraction drops as LLS rises
        core_r = delta[r, 2]                         # (K,) DLA core (forward template + excess add-back)
        if mf is None:
            corr = np.ones((4, K))                    # LF-resolution truth (no correction)
        else:
            # GATE INVARIANT: the SAME fixed per-class MF factor exp(g + log res_corr) the
            # forward applies, at this row's (θ, z, τ₀). The per-class P_filt correction
            # mirrors _excess_from_P_filt / _predict_P_obs_mf so the truth is reproduced when
            # θ→truth in the MF forward.
            corr = np.asarray(jnp.exp(DL._mf_corr_on_cache(
                mf, th_truth, jnp.asarray(float(z_unit_rows[i])),
                jnp.asarray(float(tau0_all[r])))))    # (4, K)
        # §0c DLA handling (PI-confirmed final intent 2026-06-09): the DLA-finder masking is
        # INCOMPLETE (misses ~10%). The closure TARGET MOCK carries the unmasked-DLA residual.
        # We build TWO pieces here: (1) the DLA-MASKED baseline (the DLA class held at the CLEAN
        # level — the τ=1e6-filtered + LLS/subDLA-contaminated forest, NO DLA excess), and (2)
        # the FULL DLA excess add-back w_DLA·(P_DLA^unf − P_clean), P_DLA^unf = P_filt[DLA]+core.
        # make_legb_mock then adds TRUTH_DLA_FRAC[leg]·excess to the baseline per leg (DESI 0.10,
        # KS 0.0) — i.e. DESI's target retains 10% of the full DLA systems the finder misses,
        # KS's target none. The forward marginalizes α_DLA over this (the closure asks: does
        # HCD-marginalization recover cosmology DESPITE the DLA residual?).
        P_clean_corr = P_filt[r, 0] * corr[0]
        P_cls = np.stack([P_clean_corr, P_filt[r, 1] * corr[1],
                          P_filt[r, 2] * corr[2], P_clean_corr])   # DLA class → clean (masked baseline)
        P_obs[i] = np.einsum("c,ck->k", coef, P_cls)
        # full DLA excess: w_DLA·(P_DLA^unf − P_clean), P_DLA^unf = P_filt[DLA]·corr + core.
        P_dla_unf = P_filt[r, 3] * corr[3] + core_r
        dla_excess[i] = a[2] * (P_dla_unf - P_clean_corr)
        dla_core[i] = core_r
    # the truth PRIYA (amp, dτ₀): best-fit of the selected per-z τ₀(z) in the α=τ₀/Kim coord —
    # the closure bias-z for the mean flux is computed against these (the forward samples
    # tau0_amp/dtau0; for a "priya"/tuple anchor these recover the requested curve to the
    # ladder-discretization floor, the self-consistency the 13-rung→2-param swap requires).
    _alpha_sel = tau0_all[keep_rows] / np.asarray(_kim(jnp.asarray(z)))
    tau0_amp_true, dtau0_true = fit_tau0_alpha_priya(np.asarray(z), _alpha_sel)
    # PER-Z structural w_c (nZs,3) [LLS,sub,DLA] — the Z-RESOLVED truth incidence the z-flat ``w_c``
    # (z-median, below) drops. Used to build the z-resolved truth alpha for the SBC loglik-rank
    # re-scoring (make_legb_mock → truth_pack['alpha_hcd_z']); the LLS column carries the same
    # lls_truth_boost the z-median does (matches make_truth_from_sim's per-row a[0] boost above).
    w_c_z = np.asarray(w_c[keep_rows, 1:], float).copy()
    w_c_z[:, 0] = w_c_z[:, 0] * lls_truth_boost
    return dict(
        z=z, P_obs_true=P_obs, dla_excess_true=dla_excess,
        params_unit=params_unit[keep_rows[0]],
        tau0=tau0_all[keep_rows], dla_core=dla_core,
        tau0_amp=tau0_amp_true, dtau0=dtau0_true,
        w_c=np.median(w_c[keep_rows, 1:], axis=0) * np.array([lls_truth_boost, 1.0, 1.0]),
        w_c_z=w_c_z, rows=keep_rows)


def make_hr_truth_from_cache(sim_name, target_k, *, tau0_anchor="priya"):
    """Genuine HF-LOSO truth: a held-out HR sim's REAL measured P1D (HR resolution, NO MF
    correction), built like ``make_truth_from_sim`` but read DIRECTLY from the HR cache and
    interpolated onto ``target_k`` (= the LF cache grid the forward evaluates on). The forward
    (LF emulator × the MF correction fit EXCLUDING this sim) is then compared to this real HR truth
    — the integrated (in-the-inference) version of the forward-only MF-LOSO. Same dict contract as
    ``make_truth_from_sim`` so ``make_legb_mock`` consumes it unchanged."""
    from hcd_analysis.emulator import multifidelity as MF
    d = MF.load_cache(MF.HR_CACHE)
    sim_name = str(sim_name)
    names = np.array([s.decode() if isinstance(s, bytes) else s for s in d["sim_name"]])
    z_grid = d["z_grid"]; P_filt = d["P_filt"]; delta = d["delta"]; w_c = d["w_c_cache"]
    tau0_all = d["tau0"]; params_unit = d["params_unit"]; kf = d["kfkms"]
    rows = np.where(names == sim_name)[0]
    cand = np.array([int(r) for r in rows
                     if (2.2 - 1e-6 <= z_grid[r] <= 4.6 + 1e-6) and np.isfinite(P_filt[r]).all()])
    if cand.size == 0:
        raise ValueError(f"HR sim {sim_name!r} has no in-range finite rows in the HR cache")
    z_of = np.round(z_grid[cand], 4)
    keep_rows = []
    for zz in np.unique(z_of):                       # one ladder row per z (the τ₀-anchor)
        sub = cand[z_of == zz]
        if isinstance(tau0_anchor, (tuple, list)):
            amp_t, dt_t = float(tau0_anchor[0]), float(tau0_anchor[1])
            target = float(tau0_alpha_priya(jnp.asarray(float(zz)), amp_t, dt_t) * _kim(jnp.asarray(float(zz))))
        elif tau0_anchor == "becker13":
            target = float(becker13_tau0(jnp.asarray(float(zz))))
        else:                                         # "priya" (Kim central) default
            target = float(_kim(jnp.asarray(float(zz))))
        keep_rows.append(int(sub[int(np.argmin(np.abs(tau0_all[sub] - target)))]))
    keep_rows = np.array(sorted(keep_rows, key=lambda r: z_grid[r]))
    z = z_grid[keep_rows]; tk = np.asarray(target_k, dtype=float)
    P_obs = np.zeros((len(keep_rows), tk.size)); dla_excess = np.zeros_like(P_obs); dla_core = np.zeros_like(P_obs)
    for i, r in enumerate(keep_rows):
        a = np.asarray(w_c[r, 1:], dtype=float)       # HR sim's own contamination (no boost)
        coef = np.concatenate([[1.0 - a.sum()], a])
        Pc = P_filt[r, 0]                             # REAL HR clean (corr = 1; this IS the truth)
        P_cls = np.stack([Pc, P_filt[r, 1], P_filt[r, 2], Pc])     # DLA-masked baseline
        P_obs_hr = np.einsum("c,ck->k", coef, P_cls)
        dexc_hr = a[2] * ((P_filt[r, 3] + delta[r, 2]) - Pc)       # full DLA excess
        kr = np.asarray(kf[r]); ok = np.isfinite(P_obs_hr) & (kr > 0)
        P_obs[i] = np.interp(tk, kr[ok], P_obs_hr[ok])            # HR grid (525) -> LF forward grid
        dla_excess[i] = np.interp(tk, kr[ok], dexc_hr[ok])
        dla_core[i] = np.interp(tk, kr[ok], np.asarray(delta[r, 2])[ok])
    _alpha_sel = tau0_all[keep_rows] / np.asarray(_kim(jnp.asarray(z)))
    tau0_amp_true, dtau0_true = fit_tau0_alpha_priya(np.asarray(z), _alpha_sel)
    return dict(
        z=z, P_obs_true=P_obs, dla_excess_true=dla_excess,
        params_unit=params_unit[keep_rows[0]], tau0=tau0_all[keep_rows], dla_core=dla_core,
        tau0_amp=tau0_amp_true, dtau0=dtau0_true,
        w_c=np.median(w_c[keep_rows, 1:], axis=0), rows=keep_rows)


def _chol_jitter(C, jitter=1e-10):
    """Lower Cholesky of the jittered C_data (matches gaussian_loglik's SPD floor)."""
    K = C.shape[-1]
    Cj = C + (jitter * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    return jnp.linalg.cholesky(Cj)


def _resolve_res_corr_inject(inject_res_corr, leg_name, n_rows):
    """Resolve the ``inject_res_corr`` spec into this leg's log-perturbation b-vector (on the
    leg k-grid, shape ``(n_rows,)``). Returns ``None`` (a true no-op) when no injection applies.

    TASK-1.6 res_corr injection harness (spec §4.2). The injection is a per-leg multiplicative
    LOG-res_corr perturbation applied to the leg-binned TRUTH ONLY (``P_truth_on_leg *= exp(b)``),
    analogous to the SiIII ``mfac`` multiply but in log-space. It is the OUT-OF-SPAN
    misspecification the marginalized ``alpha_res`` must absorb — it NEVER enters the forward, so
    it does NOT cancel in a closure. The strength is SEPARATE from the sampled ``alpha_res``.

    Accepted spec forms (``inject_res_corr``):
      * ``None``                → no-op (returns ``None``; the default, byte-identical).
      * ``{"path": <npz>, "member": "b1"|"b2", "strength": float}``
            → loads ``{leg_name}_{member}`` from the basis npz (the flat z-major log-perturbation
              on the leg k-grid, the SAME order/shape as ``leg.k``) and scales it by ``strength``.
              This is the form ``run_stepA``'s injection arm passes.
      * ``{leg_name: b_vector, ...}``
            → an explicit per-leg b-vector dict (already strength-scaled). Legs absent from the
              dict get no injection.

    Returns the strength-scaled b-vector ``strength * b_{member}_{leg}`` (or the explicit vector),
    or ``None`` if this leg has no injection."""
    if inject_res_corr is None:
        return None
    if not isinstance(inject_res_corr, dict):
        raise TypeError(
            f"inject_res_corr must be None or a dict, got {type(inject_res_corr).__name__}")
    # (path, member, strength) form — the run_stepA arm's spec.
    if "path" in inject_res_corr or "member" in inject_res_corr:
        path = inject_res_corr["path"]
        member = inject_res_corr.get("member", "b1")
        strength = float(inject_res_corr.get("strength", 1.0))
        basis = np.load(path, allow_pickle=True)
        key = f"{leg_name}_{member}"
        if key not in basis.files:
            raise KeyError(
                f"inject_res_corr: {key!r} not in basis {path!r} (have {sorted(basis.files)})")
        b = np.asarray(basis[key], dtype=float)
        if b.shape != (n_rows,):
            raise ValueError(
                f"inject_res_corr: basis {key} shape {b.shape} != leg-truth shape {(n_rows,)}")
        return strength * b
    # explicit {leg_name: b_vector} form.
    if leg_name not in inject_res_corr:
        return None
    b = np.asarray(inject_res_corr[leg_name], dtype=float)
    if b.shape != (n_rows,):
        raise ValueError(
            f"inject_res_corr[{leg_name!r}] shape {b.shape} != leg-truth shape {(n_rows,)}")
    return b


def _resolve_res_instr_inject(inject_resolution, leg):
    """Resolve the ``inject_resolution`` spec into (res_b_scalar, b_res_vec) for THIS leg. The
    b_res_vec is a PER-Z (shape (leg.n_z,)) instrument-resolution perturbation applied as
    P *= exp(2*b_res(z)*k^2*R_z(z)^2) per z-block; res_b_scalar is the legacy single-value form.
    Exactly one of the two is non-None when an injection applies; (None, None) is a byte-identical no-op.

    Accepted spec forms:
      * None                                   -> (None, None)          no-op (default)
      * {"b_res": s}                            -> (float(s), None)     UNCHANGED scalar path
      * {"b_res_vec": v}   (len leg.n_z)        -> (None, np.asarray(v, float))
      * {"path": npz, "member": m, "strength": x}
            -> (None, x * basis[f"{leg.name}_{m}"])   per-z (n_z,) OOS basis member
    """
    if inject_resolution is None:
        return (None, None)
    if not isinstance(inject_resolution, dict):
        raise TypeError(f"inject_resolution must be None or a dict, got {type(inject_resolution).__name__}")
    n_z = int(leg.n_z)
    if "b_res" in inject_resolution:
        return (float(inject_resolution["b_res"]), None)
    if "path" in inject_resolution or "member" in inject_resolution:
        path = inject_resolution["path"]
        member = inject_resolution.get("member", "bres1")
        strength = float(inject_resolution.get("strength", 1.0))
        basis = np.load(path, allow_pickle=True)
        key = f"{leg.name}_{member}"
        if key not in basis.files:
            raise KeyError(f"inject_resolution: {key!r} not in basis {path!r} (have {sorted(basis.files)})")
        v = np.asarray(basis[key], dtype=float)
        if v.shape != (n_z,):
            raise ValueError(f"inject_resolution: basis {key} shape {v.shape} != leg n_z {(n_z,)}")
        return (None, strength * v)
    if "b_res_vec" in inject_resolution:
        v = np.asarray(inject_resolution["b_res_vec"], dtype=float)
        if v.shape != (n_z,):
            raise ValueError(f"inject_resolution b_res_vec shape {v.shape} != leg n_z {(n_z,)}")
        return (None, v)
    raise KeyError(f"inject_resolution: unrecognized spec keys {sorted(inject_resolution)}")


def make_legb_mock(ctx: LegBCtx, truth_sim, key, *, inject_a_siiii=0.0, inject_res_corr=None):
    """Build a Leg-B mock from a sim-truth: interpolate the sim-truth P1D onto each leg's k,
    draw ε ~ N(0, C_data) (cosmic-ONLY) per leg, ``mock = truth_on_leg + ε``.

    ``inject_a_siiii`` > 0 multiplies the truth-on-leg by the SiIII metal factor (the SAME
    ``_metal_factor`` the forward uses) on metals_on legs BEFORE noise — the SiIII-injection cert
    arm (a forward with ``sample_metals`` should then absorb it into a_SiIII with no n_s/A_p leak).

    ``inject_res_corr`` (default ``None`` ⇒ byte-identical no-op) injects an OUT-OF-SPAN res_corr
    misspecification into the leg-binned TRUTH ONLY (TASK-1.6, spec §4.2): a per-leg log-res_corr
    perturbation ``b`` (on the leg k-grid) applied as ``P_truth_on_leg *= exp(b)`` at the SAME point
    ``inject_a_siiii`` multiplies (AFTER SiIII, BEFORE the cosmic-noise draw). It is the
    misspecification the marginalized ``alpha_res`` must absorb; it has a SEPARATE injection
    strength from the sampled ``alpha_res`` and NEVER touches the forward (so it does not cancel in
    a closure). See ``_resolve_res_corr_inject`` for the accepted spec forms.

    MOCK-TRUTH → LEG mapping (documented choice): the sim has one z per cache row. For each
    leg z-bin we NEAREST-Z map to the sim's available z (the cache z grid is Δz=0.2, and the
    survey z-bins fall within ~0.1 of a cache z) and interpolate that z's sim-truth P1D from
    the cache k onto the leg's k (jnp.interp, the SAME binding the model uses). If a leg z is
    farther than ``z_tol`` from any sim z, that z's rows are DROPPED for this mock (set NaN →
    excluded from the likelihood via the data being absent). The truth θ-vector is the sim's
    params_unit + the per-leg-z τ₀ (nearest-z mapped onto the GLOBAL z grid) + the sim's w_c.

    Returns ``(mock_legs, truth_pack, info)``:
      mock_legs  — copies of ctx.legs with P_data := the noisy mock (kept-z rows only);
      truth_pack — dict(theta9, tau0_global (on z_global), alpha_hcd, kept_global_z (bool));
      info       — dict(key, per-leg dropped-z, the sim z used, the noiseless truth_on_leg).

    PER-LEG DLA RESIDUAL (§0c, PI-confirmed final intent 2026-06-09): the truth-on-leg is the
    DLA-MASKED baseline ``P_obs_true`` PLUS ``TRUTH_DLA_FRAC[leg.name]``·``dla_excess_true`` (the
    full DLA excess). DESI carries 0.10·excess (the ~10% the DLA finder misses → full systems
    remain); KS carries 0% (KS fully masks DLAs). The forward marginalizes α_DLA over this
    residual (DataLeg.dla_forward_frac scales the forward DLA term per leg: DESI 1.0 / KS 0.0).
    """
    z_sim = np.asarray(truth_sim["z"])
    P_sim = np.asarray(truth_sim["P_obs_true"])         # (nZs, K) masked baseline
    dla_excess_sim = np.asarray(truth_sim["dla_excess_true"])  # (nZs, K) full DLA excess
    tau0_sim = np.asarray(truth_sim["tau0"])
    cache_k = np.asarray(ctx.cache_k)
    z_tol = 0.15                                    # nearest-z map tolerance (cache Δz=0.2)

    keys = jax.random.split(key, len(ctx.legs))
    mock_legs = []
    dropped = {}
    truth_on_leg_out = {}
    for li, leg in enumerate(ctx.legs):
        N = leg.k.shape[0]
        P_mock = np.array(leg.P_data, float).copy()
        keep_row = np.zeros(N, bool)
        drop_z = []
        # the per-leg DLA-residual fraction for the TARGET MOCK (DESI 0.10 / KS 0.0); default 0.0
        # for any unrecognised leg (no DLA residual added unless explicitly DESI).
        truth_frac = float(TRUTH_DLA_FRAC.get(leg.name, 0.0))
        # build the truth-on-leg per z = masked baseline + truth_frac·(full DLA excess), then add
        # cosmic noise over the kept rows.
        P_truth_on_leg = np.full(N, np.nan)
        for iz in range(leg.n_z):
            zz = float(leg.z[iz])
            j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) > z_tol:
                drop_z.append(zz)
                continue
            rows = np.where(np.asarray(leg.z_idx) == iz)[0]
            k_sub = np.asarray(leg.k)[rows]
            # per-leg target P1D on the cache grid: masked baseline + the leg's DLA residual.
            P_target_cache = P_sim[j] + truth_frac * dla_excess_sim[j]
            P_truth_on_leg[rows] = np.asarray(
                jnp.interp(jnp.asarray(k_sub), jnp.asarray(cache_k), jnp.asarray(P_target_cache)))
            keep_row[rows] = True
        # SiIII injection (cert arm): multiply the truth by the McDonald SiIII factor on metals_on
        # legs, with the SAME _metal_factor the forward uses (so a_SiIII can absorb it exactly).
        if inject_a_siiii > 0 and leg.metals_on:
            mfac = np.asarray(DL._metal_factor(jnp.asarray(leg.k), a_SiIII=float(inject_a_siiii)))
            P_truth_on_leg = np.where(np.isfinite(P_truth_on_leg), P_truth_on_leg * mfac, P_truth_on_leg)
        # res_corr injection (TASK-1.6 cert arm): multiply the leg-binned truth by exp(b_leg), the
        # OUT-OF-SPAN log-res_corr misspecification on the leg k-grid (same flat z-major order as
        # leg.k). TRUTH-ONLY — the forward never sees it, so it cannot cancel in a closure. Applied
        # AFTER the SiIII inject, BEFORE the noise draw, on the finite (kept) rows only.
        b_inj = _resolve_res_corr_inject(inject_res_corr, leg.name, N)
        if b_inj is not None:
            efac = np.exp(b_inj)
            P_truth_on_leg = np.where(np.isfinite(P_truth_on_leg), P_truth_on_leg * efac, P_truth_on_leg)
        dropped[leg.name] = drop_z
        truth_on_leg_out[leg.name] = P_truth_on_leg.copy()

        # ε ~ N(0, C_data) over the KEPT rows (cosmic-only; jittered Cholesky).
        if keep_row.any():
            kr = np.where(keep_row)[0]
            Cd = jnp.asarray(leg.C_data[np.ix_(kr, kr)])
            L = _chol_jitter(Cd)
            g = jax.random.normal(keys[li], (kr.size,))
            eps = np.asarray(jnp.einsum("ij,j->i", L, g))
            P_mock[kr] = P_truth_on_leg[kr] + eps
        # dropped-z rows → NaN (the likelihood NaN-sanitises; here we keep the data value but
        # the driver restricts to the kept rows when building the per-mock legs below).
        P_mock[~keep_row] = np.nan

        mock_legs.append(leg._replace(P_data=P_mock))

    # truth on the GLOBAL z grid: nearest-z map the sim τ₀ onto z_global (drop z with no sim).
    zg = np.asarray(ctx.z_global)
    tau0_global = np.zeros(len(zg))
    kept_global = np.zeros(len(zg), bool)
    for i, zz in enumerate(zg):
        j = int(np.argmin(np.abs(z_sim - zz)))
        if abs(z_sim[j] - zz) <= z_tol:
            tau0_global[i] = tau0_sim[j]
            kept_global[i] = True
        else:
            tau0_global[i] = float(becker13_tau0(jnp.asarray(zz)))  # placeholder (unused: no data there)
    # truth α: LLS/subDLA = the sim's structural w_c (the real residual the closure marginalizes).
    # DLA = TRUTH_DLA_FRAC["DESI"]·w_DLA = 0.10·w_DLA — the §0c per-leg DLA residual the forward
    # marginalizes over (PI-confirmed final intent 2026-06-09). The DESI target carries 0.10·(full
    # DLA excess) and the DESI forward DLA term is live (dla_forward_frac=1.0), so at θ→truth the
    # consistency point is α_DLA = 0.10·w_DLA. (On KS the target carries 0% AND the forward DLA
    # term is 0, so the KS leg is α_DLA-blind — consistent for any α_DLA.) The α_DLA prior is
    # centered on this 10% residual (HCD_DLA_RESIDUAL_FRAC=0.10) and is MARGINALIZED (sampled).
    alpha_truth = np.asarray(truth_sim["w_c"]).copy()             # (3,) [LLS,sub,DLA] z-MEDIAN pivot
    alpha_truth[2] = TRUTH_DLA_FRAC["DESI"] * alpha_truth[2]      # 0.10·w_DLA (the DESI residual)
    # Z-RESOLVED truth alpha (nZg,3) for the SBC loglik-rank re-scoring (ll_true): the mock
    # truth-on-leg uses the per-z sim P1D (z-resolved contamination), so re-scoring with a z-FLAT
    # alpha produces a spurious z-ramp. Map the per-z sim w_c (``truth_sim['w_c_z']``, nearest-z onto
    # z_global) and apply the SAME §0c DLA 10% residual scaling to the DLA column as the z-flat
    # pivot above. Dropped-z rows (no sim z within z_tol) fall back to the z-median pivot (they carry
    # NO data so they never affect the likelihood). Matches scripts/diag_legb_zresolved_alpha_check.
    if truth_sim.get("w_c_z") is not None:
        w_c_z = np.asarray(truth_sim["w_c_z"], float)            # (nZs,3) per-z structural w_c
        alpha_hcd_z = np.tile(alpha_truth, (len(zg), 1))        # (nZg,3) default = z-median pivot
        for i, zz in enumerate(zg):
            j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) <= z_tol:
                az = w_c_z[j].copy()
                az[2] = TRUTH_DLA_FRAC["DESI"] * az[2]           # 0.10·w_DLA(z) (the per-z residual)
                alpha_hcd_z[i] = az
    else:
        alpha_hcd_z = np.tile(alpha_truth, (len(zg), 1))
    truth_pack = dict(
        theta9=np.asarray(truth_sim["params_unit"]),
        tau0_global=tau0_global, alpha_hcd=alpha_truth, alpha_hcd_z=alpha_hcd_z,
        kept_global_z=kept_global)
    info = dict(key=key, dropped=dropped, z_sim=z_sim, truth_on_leg=truth_on_leg_out)
    return mock_legs, truth_pack, info


# ============================================================================ #
#  Leg-A (rank-uniformity SBC) on the leg grids — draw truth from the PRIOR, matched-C self-draw.
# ============================================================================ #
def draw_leg_a_leg_truth(ctx: LegBCtx, key):
    """Draw a Leg-A truth from the legb PRIORS for a rank-uniformity SBC on the leg grids.

    Traces ``_legb_priors_only`` (the EXACT prior sites of ``_legb_model``) for one sample, then
    reconstructs the deterministics (τ₀(z), z-resolved α) host-side via
    ``_legb_reconstruct_deterministics``. Returns a truth_pack matching ``make_legb_mock``'s
    contract: ``theta9`` (9,), ``tau0_global`` (nZg,), ``alpha_hcd`` (3,) PIVOT [LLS,sub,DLA] for
    the rank truth-vector, ``alpha_hcd_z`` (nZg,3) Z-RESOLVED for the forward, ``a_siiii``,
    ``kept_global_z`` (all True — Leg-A keeps every leg z), plus the raw latent sites (``raw``)."""
    from numpyro import handlers as _nph
    tr = _nph.trace(_nph.seed(_legb_priors_only, key)).get_trace(ctx)
    raw = {nm: site["value"] for nm, site in tr.items() if site.get("type") == "sample"}
    samples1 = {k: jnp.asarray(v)[None] for k, v in raw.items()}      # length-1 L axis
    rec = _legb_reconstruct_deterministics(ctx, samples1)
    a_lls = float(rec["alpha_lls"][0]) if "alpha_lls" in rec else float(raw["alpha_lls"])
    a_sub = float(rec["alpha_subdla"][0]) if "alpha_subdla" in rec else float(raw["alpha_subdla"])
    a_dla = float(rec["alpha_dla"][0])
    return dict(
        theta9=np.asarray(raw["theta_unit"]),
        tau0_global=np.asarray(rec["tau0_vec"][0]),                  # (nZg,)
        alpha_hcd=np.array([a_lls, a_sub, a_dla]),                   # (3,) pivot (rank truth-vec)
        alpha_hcd_z=np.asarray(rec["alpha_hcd_z"][0]),              # (nZg,3) z-resolved (forward)
        a_siiii=(float(raw["a_SiIII"]) if "a_SiIII" in raw else 0.0),
        # the mean-flux SITES (tau0_amp/dtau0) for the τ₀-bias report (feedback-report-tau0-dtau0-bias).
        tau0_amp=(float(raw["tau0_amp"]) if "tau0_amp" in raw else np.nan),
        dtau0=(float(raw["dtau0"]) if "dtau0" in raw else np.nan),
        kept_global_z=np.ones(len(ctx.z_global), bool), raw=raw)


# --------------------------------------------------------------------------------------------- #
#  Data-nuisance injection hooks (the BIAS gate). On the Leg-A self-draw the cosmology bias is
#  ZERO by construction; these inject a nuisance the production forward CANNOT fit (or a truth
#  offset from the per-survey LLS pin) so the recovered A_p/n_s shift isolates that nuisance.
# --------------------------------------------------------------------------------------------- #
def apply_lls_truth_boost(truth_pack, boost):
    """Return a COPY of ``truth_pack`` with the LLS incidence (pivot ``alpha_hcd[0]`` AND the
    z-resolved ``alpha_hcd_z[:,0]`` column) multiplied by ``boost`` (>1 ⇒ the mock carries a
    survey-level LLS excess relative to the prior pin center). subDLA/DLA, θ9, τ₀ and a_SiIII are
    untouched; the input is NOT mutated (deep-copies the two LLS-bearing arrays). ``boost=1`` is
    the identity. The LLS-excess arm of the data-nuisance bias gate uses this to put the truth at
    the per-survey lit/sim LLS center while the forward keeps the (cosmic-average / DESI) pin."""
    out = dict(truth_pack)                                    # shallow copy of the dict
    a = np.array(truth_pack["alpha_hcd"], float)              # fresh (3,) — input unmutated
    a[0] = a[0] * float(boost)
    out["alpha_hcd"] = a
    if truth_pack.get("alpha_hcd_z") is not None:
        az = np.array(truth_pack["alpha_hcd_z"], float)       # fresh (nZg,3)
        az[:, 0] = az[:, 0] * float(boost)
        out["alpha_hcd_z"] = az
    return out


def apply_subdla_truth_boost(truth_pack, boost):
    """Return a COPY of ``truth_pack`` with the subDLA incidence (pivot ``alpha_hcd[1]`` AND the
    z-resolved ``alpha_hcd_z[:,1]`` column) multiplied by ``boost`` — the SIBLING of
    ``apply_lls_truth_boost`` for the subDLA-displacement arm of the data-nuisance bias gate. LLS
    (index 0), DLA (index 2), θ9, τ₀ and a_SiIII are untouched; the input is NOT mutated
    (deep-copies the two subDLA-bearing arrays). ``boost=1`` is the identity."""
    out = dict(truth_pack)                                    # shallow copy of the dict
    a = np.array(truth_pack["alpha_hcd"], float)              # fresh (3,) — input unmutated
    a[1] = a[1] * float(boost)
    out["alpha_hcd"] = a
    if truth_pack.get("alpha_hcd_z") is not None:
        az = np.array(truth_pack["alpha_hcd_z"], float)       # fresh (nZg,3)
        az[:, 1] = az[:, 1] * float(boost)
        out["alpha_hcd_z"] = az
    return out


def _meanflux_on_leg(ctx, leg, truth_pack):
    """Per-leg-z mean flux ⟨F⟩(z)=exp(−τ_eff(z)) from the truth mean flux on z_global.

    NOTE (the load-bearing fix): ``truth_pack["tau0_global"]`` is ALREADY τ_eff(z)=α(z)·Kim07
    (built as ``tau0_vec = alpha_z·kim`` in _legb_model / _legb_reconstruct_deterministics, i.e.
    the cache ``tau0 = −ln⟨F⟩`` coordinate), NOT the bare α-ladder coord. So we use it DIRECTLY —
    multiplying by Kim again would double-apply it and make ⟨F⟩=exp(−α·Kim²), inflating the
    injected metal amplitude up to ~6× at low z (where Kim is smallest). Returns a (leg.n_z,)
    array aligned with ``leg.z`` (nearest-z map onto z_global, mirroring make_leg_a_legmock's
    `sel`)."""
    zg = np.asarray(ctx.z_global)
    tau_eff_global = np.asarray(truth_pack["tau0_global"])    # (nZg,) τ_eff = α·Kim = −ln⟨F⟩
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
    tau_eff = tau_eff_global[sel]                             # (n_z,)
    return np.exp(-tau_eff)                                   # ⟨F⟩(z)


def _check_resolution_injectable(legs, active):
    """RAISE (do NOT silently skip) if a resolution injection is requested (``active=True``: a scalar
    b_res OR a per-z b_res_vec / basis member) on a leg whose R_z is not trustworthy
    (``leg.resolution_ready`` False -- e.g. a proxy-R_z KS). ``active=False`` is a byte-identical no-op.
    Gated on the R_z-valid flag, NOT resolution_on."""
    if not active:
        return
    bad = [leg.name for leg in legs if not getattr(leg, "resolution_ready", True)]
    if bad:
        raise ValueError(
            f"resolution injection requested on non-resolution_ready leg(s) {bad}: their R_z proxy is "
            "untrustworthy (KS is echelle sigma~3.2 km/s; the DESI proxy R_z is ~7-15x too large) so the "
            "injected distortion is un-fittable. Wire the leg's true R_z (flip resolution_ready) first, or "
            "run the resolution arm on desi/eboss.")


def _check_single_instrument_for_res(legs, sample_res):
    """Guard for the option-b f_res site (4-referee panel #8 / Bayesian #4): ``_legb_model`` samples ONE
    global ``f_res_amp``/``f_res_slope``, but spectral resolution is a per-INSTRUMENT systematic -- DESI
    and eBOSS spectrographs are physically independent (unlike the globally-shared intergalactic metals).
    A single shared f_res across instruments would be pulled to a wrong compromise (DESI wants ~0.022,
    eBOSS ~0.044) -> a residual k^2 tilt on the under-corrected leg -> n_s bias on a JOINT fit. Until
    per-instrument sites (``f_res_amp_desi``/``f_res_amp_eboss``) are built, a multi-instrument ctx with
    ``sample_res`` must RAISE. The per-leg Gate-B cert is single-instrument, so this is a no-op there.
    ``sample_res=False`` is a no-op (golden-safe)."""
    if not sample_res:
        return
    instruments = sorted({leg.name for leg in legs})
    if len(instruments) > 1:
        raise ValueError(
            f"sample_res=True (option-b f_res) with a MULTI-instrument ctx {instruments}: f_res is a single "
            "GLOBAL site but spectral resolution is per-instrument (independent spectrographs). Run the "
            "resolution cert per-leg (single instrument), or implement per-instrument f_res sites "
            "(f_res_amp_desi/f_res_amp_eboss) before a joint fit.")


def make_leg_a_legmock(ctx: LegBCtx, dla_core_per_leg, truth_pack, key, *,
                       inject_metal_misspec=None, inject_resolution=None):
    """Leg-A self-draw on the leg grids: forward-model the prior-drawn truth on each leg with the
    SAME ``predict_P_obs_on_leg`` the likelihood uses, then add ε ~ N(0, C_total(truth)) over ALL
    rows. C_mock ≡ C_like AND the noiseless mock == P_model(truth) → the rank-uniformity null is
    EXACT (Talts+2018). Returns ``(mock_legs, info)``; ``info['chol'][leg]`` /
    ``info['truth_on_leg'][leg]`` per leg.

    DATA-NUISANCE INJECTION (the bias gate; default None ⇒ byte-identical to the clean self-draw):
    when set, the contaminant multiplies the NOISELESS ``P_model`` BEFORE the ε draw (mirroring
    make_legb_mock's SiIII inject at line 772), so the recorded ``truth_on_leg`` and the mock both
    carry it but the FORWARD likelihood (which the gate keeps clean of the unfittable mode) cannot.

      ``inject_metal_misspec``: dict, e.g. {"form":"desi_full","f_SiIII":0.009,"f_SiII":0.004,
        "f_SiII_SiII":0.002,...} forwarded as kwargs to ``metal_inject``. Applied ONLY on legs with
        ``leg.metals_on`` (DESI/eBOSS), PER z-block: ⟨F⟩(z) from the truth τ₀ (``_meanflux_on_leg``)
        feeds metal_inject on that z's rows. The desi_full form carries an ADDITIVE SiII–SiII term
        the multiplicative forward _metal_factor STRUCTURALLY cannot fit (the bias probe).
      ``inject_resolution``: dict, e.g. {"b_res":0.02} → multiply P_model by
        ``_resolution_factor(k, R_z, b_res)`` per z-block on ALL legs (the production forward has
        resolution_on=False so it cannot fit this distortion — the probe). R_z is the leg's own
        per-z resolution scale (``leg.R_z``)."""
    metal_kw = dict(inject_metal_misspec) if inject_metal_misspec else None
    _check_resolution_injectable(ctx.legs, active=(inject_resolution is not None))  # RAISE on a stray non-resolution_ready leg (e.g. KS)
    zg = np.asarray(ctx.z_global)
    theta9 = jnp.asarray(truth_pack["theta9"])
    tau0_global = jnp.asarray(truth_pack["tau0_global"])
    alpha_hcd_z = jnp.asarray(truth_pack["alpha_hcd_z"])
    a_siiii = float(truth_pack.get("a_siiii", 0.0))
    keys = jax.random.split(key, len(ctx.legs))
    mock_legs, chol_out, truth_on_leg_out = [], {}, {}
    for li, leg in enumerate(ctx.legs):
        sel = jnp.asarray([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
        msc = (ctx.mf_shape_per_leg.get(leg.name)
               if getattr(ctx, "mf_shape_per_leg", None) is not None else None)
        mec = (ctx.mf_emucoh_per_leg.get(leg.name)
               if getattr(ctx, "mf_emucoh_per_leg", None) is not None else None)
        P_model, C_total = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_global[sel], alpha_hcd_z[sel], pf_stats=ctx.pf_stats,
            dla_core=dla_core_per_leg[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, a_SiIII=a_siiii, cemu_inflate=ctx.cemu_inflate,
            rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor,
            mf_shape_cov=msc, mf_shape_infl=getattr(ctx, "mf_shape_infl", 1.0),
            mf_emucoh_cov=mec, mf_emucoh_infl=getattr(ctx, "mf_emucoh_infl", 1.0),
            mf_emucoh_offdiag_only=getattr(ctx, "mf_emucoh_offdiag_only", False))
        P_model = np.array(P_model, float)                     # host (writable): nuisance injection
        # DATA-NUISANCE INJECTION (default OFF). Multiply the noiseless P_model by the host
        # contaminant BEFORE the ε draw, per z-block (mirrors make_legb_mock's per-z loop).
        if metal_kw is not None and leg.metals_on:
            Fbar = _meanflux_on_leg(ctx, leg, truth_pack)     # (n_z,) ⟨F⟩(z) from the truth τ₀
            k_leg = np.asarray(leg.k)
            z_idx = np.asarray(leg.z_idx)
            _form = metal_kw.get("form")
            for iz in range(leg.n_z):
                rows = np.where(z_idx == iz)[0]
                if rows.size == 0:
                    continue
                if _form == "zevo":
                    # MODEL C+ metal_zevo arm (Arms 1/2): interpolate f per z and inject the IN-CLASS
                    # contaminant via desi_full+cross on EVERY metals_on leg. legs in metal_siII_legs
                    # carry the SiII doublet + the SiIII–SiII cross; other legs (eBOSS) set f_SiII=0 ⇒
                    # SiIII-only with the SAME sigmoid decorrelation (4-lens fix (a): in-class, NOT the
                    # undamped "eboss"). f_SiII_SiII=0; k_SiIII/k_SiII/r matched to the forward ⇒ the
                    # Model C+ forward reproduces it exactly. a(z)=f(z)/(1−⟨F⟩(z)) (within-z; per the
                    # f-nodes). De-double-count: the clean truth carries no metal (a_siiii=0).
                    node_z = metal_kw.get("node_z", (2.2, 4.2))
                    k3 = metal_kw.get("k_SiIII", DL.K_SiIII_DEFAULT)
                    k2 = metal_kw.get("k_SiII", DL.K_SiII_DEFAULT)
                    f3_z = float(_metal_f_of_z(float(leg.z[iz]), node_z, metal_kw["f_SiIII_nodes"]))
                    if leg.name in tuple(ctx.metal_siII_legs):
                        f2_z = float(_metal_f_of_z(float(leg.z[iz]), node_z, metal_kw["f_SiII_nodes"]))
                    else:
                        f2_z = 0.0                                 # SiIII-only legs (eBOSS): no doublet, no cross
                    P_model[rows] = metal_inject(
                        P_model[rows], k_leg[rows], float(Fbar[iz]), form="desi_full",
                        f_SiIII=f3_z, f_SiII=f2_z, f_SiII_SiII=0.0,
                        k_decorr=k3, k_SiII=k2, r_doublet=DL.R_SiII_DOUBLET, cross=True)
                elif _form == "ma2025":
                    # Arm 3 (Ma+2025 sim, OUT-of-class): SiIII-only DIRECT amplitude + Ma damping, per z.
                    z = float(leg.z[iz])
                    a_z = 0.014 * ((1.0 + z) / 4.0) ** 0.79
                    kc_z = -1.58e-2 * ((1.0 + z) / 4.0) ** 1.15
                    P_model[rows] = metal_inject(P_model[rows], k_leg[rows], float(Fbar[iz]),
                                                 form="ma2025", a_SiIII_direct=a_z, k_cross=kc_z,
                                                 damp="ma")
                elif _form == "ma2025_gauss":
                    # Arm 4 (OUT-of-class, 4-lens fix (b)): a survey-standard DECREASING-trend SiIII
                    # metal a(z)=a0·((1+z)/4)^p (p<0) with a GAUSS damping exp(−(k/k_cross)²) the
                    # sigmoid forward CANNOT reproduce. Direct amplitude, SiII OFF.
                    z = float(leg.z[iz])
                    a_z = float(metal_kw["a0"]) * ((1.0 + z) / 4.0) ** float(metal_kw["p"])
                    P_model[rows] = metal_inject(P_model[rows], k_leg[rows], float(Fbar[iz]),
                                                 form="ma2025", a_SiIII_direct=a_z,
                                                 k_cross=float(metal_kw["k_cross"]), damp="gauss")
                else:                                          # legacy forms (desi_full/eboss/…) — byte-exact
                    P_model[rows] = metal_inject(P_model[rows], k_leg[rows], float(Fbar[iz]),
                                                 **metal_kw)
        _res_b, _res_vec = _resolve_res_instr_inject(inject_resolution, leg)
        if _res_b is not None or _res_vec is not None:
            k_leg = np.asarray(leg.k)
            z_idx = np.asarray(leg.z_idx)
            R_z = np.asarray(leg.R_z)
            for iz in range(leg.n_z):
                rows = np.where(z_idx == iz)[0]
                if rows.size == 0:
                    continue
                b_res_iz = float(_res_vec[iz]) if _res_vec is not None else _res_b
                fac = np.asarray(DL._resolution_factor(
                    jnp.asarray(k_leg[rows]), float(R_z[iz]), b_res=b_res_iz))
                P_model[rows] = P_model[rows] * fac
        Lc = _chol_jitter(C_total)
        g = jax.random.normal(keys[li], (leg.k.shape[0],))
        eps = np.asarray(jnp.einsum("ij,j->i", Lc, g))
        mock_legs.append(leg._replace(P_data=P_model + eps))
        chol_out[leg.name] = np.asarray(Lc)
        truth_on_leg_out[leg.name] = P_model.copy()
    # 'dropped' (empty per leg — Leg-A keeps every z) matches make_legb_mock's info contract,
    # which run_legb reads when building the per-mock record.
    return mock_legs, dict(key=key, chol=chol_out, truth_on_leg=truth_on_leg_out,
                           dropped={leg.name: [] for leg in ctx.legs})


# ============================================================================ #
#  numpyro adapter: the same priors as sampler_numpyro, factor = a multi-leg loglik.
#
#  ``data_likelihood.data_loglik`` takes ONE (K,) ``dla_core`` shared across legs/z. The
#  Leg-B truth core is per-leg-z (the held-out sim's own DLA delta), so we use a thin
#  per-leg wrapper (``_data_loglik_legcore``) that calls ``predict_P_obs_on_leg`` per leg
#  with that leg's own core. The PRIORS are identical to ``sampler_numpyro.numpyro_model``
#  (θ~Uniform^9 + auto-bijector; τ₀ in the α-ladder coord on the GLOBAL z grid;
#  α_lls/subdla~Normal, α_dla~softplus(Normal)). The public ``data_loglik`` signature is
#  untouched (back-compat); this wrapper only differs by the per-leg core + the kept-row
#  restriction (dropped-z mock rows are NaN and carry no info).
# ============================================================================ #
def _data_loglik_legcore(ctx: LegBCtx, theta9, tau0_global, alpha_hcd, mock_legs,
                         dla_core_per_leg, *, return_parts=False, a_siiii=0.0, a_siii=0.0,
                         metal_nodes=None, alpha_res=None, b_res_global=None, require_zresolved=True):
    """``data_loglik`` but with a PER-LEG-Z dla_core (the mock's sim core). ``data_loglik``
    takes ONE (K,) core; here each leg z uses its own, so we call ``predict_P_obs_on_leg``
    per leg with that leg's core threaded through a per-z loop is overkill — instead we note
    the core is z-binned inside the binding via ``leg.z_idx`` and the SAME core value is used
    for every row of a z. ``data_loglik`` already loops z internally with a single core; to
    keep the core matched (forward core == truth core, so it cancels in the emu-error sizing
    and the truth P_obs is reproduced when θ→truth), we pass each leg its OWN (K,) core that
    is the z-MEAN of that leg's per-z cores (a documented MVP — the per-z core variation is
    tiny vs the P1D, and the DLA sector is un-certified by Leg B without arm A3 anyway).

    ``metal_nodes`` (MODEL C+, default None → the legacy scalar a_siiii/a_siii path): a dict
    ``{leg.name: (f3_nodes (2,), f2_nodes (2,)|None, k3_nodes (2,), k2_nodes (2,)|None)}`` of the
    per-leg metal f-nodes (amplitude) AND k-nodes (decorrelation scale). Per leg the matching nodes
    are forwarded into ``predict_P_obs_on_leg`` (per-z a(z)=f(z)/(1−⟨F⟩(z)) + per-z k(z) + the
    SiIII–SiII cross). None (and a leg absent from the dict) ⇒ the scalar a_siiii/a_siii path
    (byte-exact).

    ``alpha_res`` (Task 1.3): the sampled res_corr-amplitude ``(alpha0, s)`` tuple, threaded
    FORWARD-ONLY into ``predict_P_obs_on_leg`` (it scales ``log res_corr`` by α(z) in the MF
    forward). ``None`` (default) ⇒ α≡1 ⇒ byte-exact back-compat. The TRUTH path never sets it
    (α≡1 there) so the nuisance does NOT cancel in the closure.

    ``require_zresolved`` (default **True** = SAFE-BY-DEFAULT, 2026-07-06): this comparison core scores
    a loglik of the forward against the (z-RESOLVED) mock, so a (3,) z-FLAT alpha here is ALWAYS the
    recurring z-flat bug (it broadcasts to every z and fakes a spurious z-ramp vs the z-resolved truth;
    once a phantom +5.5 sigma n_s). It therefore ASSERTS ``alpha_hcd`` is z-RESOLVED ((n_zg,3)) BY
    DEFAULT and RAISES on a z-flat (3,). Build alpha z-resolved (``alpha_pivot[None,:]*shape_zg``, or
    ``closure_legb_figs._truth_alpha_zresolved_on_leg`` for a truth-pack). Pass ``require_zresolved=False``
    ONLY for a genuine self-consistent z-flat use (the deployed forward + the re-scoring paths are all
    z-resolved, so none needs it). Making this NON-OPTIONAL is the fix: the old opt-in default let the
    bug recur (a new diagnostic forgot to opt in). The raw ``predict_P_obs_on_leg`` stays permissive so
    byte-identity uniform-alpha references pass; it is only the COMPARISON core that is locked down."""
    total = 0.0
    parts = {}
    from .likelihood import gaussian_loglik
    zg = np.asarray(ctx.z_global)
    for leg in mock_legs:
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        tau0_vec = tau0_global[jnp.asarray(sel)]
        # option-b: slice the per-z b_res(z) onto this leg (like tau0_vec); None → scalar b_res=0 (golden).
        b_res_leg = None if b_res_global is None else b_res_global[jnp.asarray(sel)]
        # per-z HCD incidence: alpha_hcd may be (3,) [broadcast] or (n_z_global,3) [z-resolved]
        alpha_leg = alpha_hcd if np.ndim(alpha_hcd) == 1 else alpha_hcd[jnp.asarray(sel)]
        core = dla_core_per_leg[leg.name]           # (K,) z-mean core for this leg
        szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
        # restrict to the kept (non-NaN) rows so dropped-z rows carry no info.
        P_data = np.asarray(leg.P_data)
        keep = np.isfinite(P_data)
        if not keep.any():
            continue
        msc = (ctx.mf_shape_per_leg.get(leg.name)
               if getattr(ctx, "mf_shape_per_leg", None) is not None else None)
        mec = (ctx.mf_emucoh_per_leg.get(leg.name)
               if getattr(ctx, "mf_emucoh_per_leg", None) is not None else None)
        # MODEL C+ per-leg metal nodes (None → the scalar a_siiii/a_siii path, byte-exact).
        f3_nodes = f2_nodes = k3_nodes = k2_nodes = None
        if metal_nodes is not None:
            f3_nodes, f2_nodes, k3_nodes, k2_nodes = metal_nodes.get(leg.name, (None, None, None, None))
        P_model, C_total = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_vec, alpha_leg, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            a_SiIII=a_siiii, a_SiII=a_siii,                    # applied only on metals_on legs
            f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,     # MODEL C+ per-z amplitude (None → scalar)
            k_SiIII_nodes=k3_nodes, k_SiII_nodes=k2_nodes,     # MODEL C+ per-z decorrelation scale
            metal_node_z=getattr(ctx, "metal_node_z", (2.2, 4.2)),
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor,
            mf_shape_cov=msc, mf_shape_infl=getattr(ctx, "mf_shape_infl", 1.0),
            mf_emucoh_cov=mec, mf_emucoh_infl=getattr(ctx, "mf_emucoh_infl", 1.0),
            mf_emucoh_offdiag_only=getattr(ctx, "mf_emucoh_offdiag_only", False),
            alpha_res=alpha_res,                              # res_corr amplitude nuisance (fwd-only)
            b_res_vec=b_res_leg,                              # option-b spectral-resolution f_res (fwd-only)
            require_zresolved=require_zresolved)              # guard: assert z-resolved alpha (opt-in)
        kr = jnp.asarray(np.where(keep)[0])
        r = jnp.asarray(P_data[keep]) - P_model[kr]
        C_sub = C_total[jnp.ix_(kr, kr)]
        ll = gaussian_loglik(r, C_sub)
        total = total + ll
        if return_parts:
            Kk = int(kr.shape[0])
            Cj = C_sub + (1e-10 * jnp.mean(jnp.diag(C_sub))) * jnp.eye(Kk)
            Lc = jnp.linalg.cholesky(Cj)
            sol = jax.scipy.linalg.cho_solve((Lc, True), r)
            parts[leg.name] = (float(ll), float(r @ sol), Kk)
    if return_parts:
        return total, parts
    return total


def _sample_tau0_sites(ctx):
    """Sample (tau0_amp, dtau0): Uniform by default; TruncatedNormal centered on (mu,sigma),
    truncated to the physical range, when ctx.{tau0_amp_gauss,dtau0_gauss} is set (the
    informative-τ₀ arm). IDENTICAL site names/order to the legacy code in both model twins."""
    la, ha = ctx.tau0_amp_range; ld, hd = ctx.dtau0_range
    ga = getattr(ctx, "tau0_amp_gauss", None); gd = getattr(ctx, "dtau0_gauss", None)
    tau0_amp = (numpyro.sample("tau0_amp", dist.Uniform(la, ha)) if ga is None
                else numpyro.sample("tau0_amp", dist.TruncatedNormal(ga[0], ga[1], low=la, high=ha)))
    dtau0 = (numpyro.sample("dtau0", dist.Uniform(ld, hd)) if gd is None
             else numpyro.sample("dtau0", dist.TruncatedNormal(gd[0], gd[1], low=ld, high=hd)))
    return tau0_amp, dtau0


def _metal_amp_site(name, ctx, on):
    """Sample a SHARED metal oscillation-amplitude site (``a_SiIII`` / ``a_SiII``), branching on the
    STATIC ``ctx.metal_prior`` (resolved at trace time → a legal python branch). SHARED by
    ``_legb_model`` and ``_legb_priors_only`` so their site construction (name, distribution class,
    params) is BYTE-IDENTICAL — the constrain_fn mirror invariant.

      "uniform" (DEFAULT, golden-safe): ``numpyro.sample(name, dist.Uniform(0, a_siiii_max))`` —
        BYTE-EXACT to the legacy code (``a=0`` reachable ⇒ the golden identity).
      "flatlog": ``numpyro.sample(name, dist.LogUniform(a_lo, a_hi))`` — the flat-log10(f) prior on
        the metal flux decrement f mapped to the amplitude ``a=f/(1−F_ref)``:
          ``a_lo = 10**metal_logf_lo / (1−F_ref)``, ``a_hi = 10**metal_logf_hi / (1−F_ref)``.
        ``1−F_ref`` is the python-float scalar ``ctx.metal_one_minus_F_ref`` (build_legb_ctx derives
        it from the union-z fiducial mean flux) — a plain float ⇒ a_lo/a_hi are python floats (no
        tracing). The LIBRARY ``dist.LogUniform`` carries the tested biject_to/support; do NOT
        hand-roll the transform and do NOT wrap a ``numpyro.deterministic`` (constrain_fn would drop it).

    ``on`` (sample_metals / sample_a_siii) gates whether the site is sampled at all; ``off`` →
    ``0.0`` (a=0 ⇒ _metal_factor≡1, the uniform golden identity)."""
    if not on:
        return 0.0
    if getattr(ctx, "metal_prior", "uniform") == "flatlog":
        omf = ctx.metal_one_minus_F_ref
        if omf is None:
            raise ValueError(
                "metal_prior='flatlog' needs metal_one_minus_F_ref (1−F_ref); build_legb_ctx "
                "derives it — construct the ctx via build_legb_ctx or pass it explicitly.")
        omf = float(omf)
        a_lo = 10.0 ** float(ctx.metal_logf_lo) / omf       # python floats (static, untraced)
        a_hi = 10.0 ** float(ctx.metal_logf_hi) / omf
        return numpyro.sample(name, dist.LogUniform(a_lo, a_hi))
    # "uniform" (default) — BYTE-EXACT legacy site.
    return numpyro.sample(name, dist.Uniform(0.0, ctx.a_siiii_max))


def _metal_2node_sites(ctx):
    """MODEL C+ (``ctx.metal_prior=="flatlog2node"``): sample, PER metals_on leg IN ORDER, PER ion,
    the 2 node values of the metal flux decrement f at ``ctx.metal_node_z`` ~
    ``dist.LogUniform(metal_fnode_lo, metal_fnode_hi)`` (flat in log10 f) AND the 2 node values of
    the sigmoid decorrelation SCALE k ~ ``dist.LogUniform(metal_knode_lo, metal_knode_hi)`` (Model C+
    floats k_SiIII/k_SiII; cup1d s_Lya_*). SiII (f AND k) nodes ONLY on legs in
    ``ctx.metal_siII_legs``. Returns ``{leg.name: (f3 (2,), f2 (2,)|None, k3 (2,), k2 (2,)|None)}``.

    SHARED by ``_legb_model`` and ``_legb_priors_only`` — iterating ``ctx.legs`` in the SAME fixed
    order with the SAME site names guarantees byte-identical site construction in both twins (the
    constrain_fn mirror invariant). Per-leg site order: f_SiIII z0,z1, [f_SiII z0,z1], k_SiIII z0,z1,
    [k_SiII z0,z1]. with_eboss=True full order: f_SiIII_DESI, f_SiII_DESI, k_SiIII_DESI, k_SiII_DESI,
    f_SiIII_eBOSS, k_SiIII_eBOSS (KS metals_off → none)."""
    nodes = {}
    if not getattr(ctx, "sample_metals", False):
        return nodes
    flo, fhi = float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)   # python floats (static, untraced)
    klo, khi = float(getattr(ctx, "metal_knode_lo", 1e-3)), float(getattr(ctx, "metal_knode_hi", 0.1))
    siII_legs = tuple(ctx.metal_siII_legs)
    for leg in ctx.legs:                                            # FIXED order ⇒ identical both twins
        if not leg.metals_on:
            continue
        f3 = jnp.stack([
            numpyro.sample(f"f_SiIII_{leg.name}_z0", dist.LogUniform(flo, fhi)),
            numpyro.sample(f"f_SiIII_{leg.name}_z1", dist.LogUniform(flo, fhi))])
        f2 = None
        if leg.name in siII_legs:
            f2 = jnp.stack([
                numpyro.sample(f"f_SiII_{leg.name}_z0", dist.LogUniform(flo, fhi)),
                numpyro.sample(f"f_SiII_{leg.name}_z1", dist.LogUniform(flo, fhi))])
        # Model C+: float the per-z sigmoid decorrelation scale (k), sampled like f (after the f-nodes).
        k3 = jnp.stack([
            numpyro.sample(f"k_SiIII_{leg.name}_z0", dist.LogUniform(klo, khi)),
            numpyro.sample(f"k_SiIII_{leg.name}_z1", dist.LogUniform(klo, khi))])
        k2 = None
        if leg.name in siII_legs:
            k2 = jnp.stack([
                numpyro.sample(f"k_SiII_{leg.name}_z0", dist.LogUniform(klo, khi)),
                numpyro.sample(f"k_SiII_{leg.name}_z1", dist.LogUniform(klo, khi))])
        nodes[leg.name] = (f3, f2, k3, k2)
    return nodes


def _legb_model(ctx: LegBCtx, mock_legs, dla_core_per_leg):
    """The numpyro model: ``sampler_numpyro``'s priors (θ~Uniform^9 + auto-bijector; τ₀ in
    the α-ladder coord on the GLOBAL z grid; α_lls/subdla~Normal, α_dla~softplus(Normal)) with
    the single ``factor`` = the per-leg-core multi-leg real-cov loglik (``_data_loglik_legcore``
    over ``mock_legs``).

    HIERARCHICAL HCD prior (``ctx.hierarchical_hcd``, default OFF → byte-identical legacy code):
    the 3 independent HCD α sites become ONE likelihood-constrained MULTIPLIER ``A_hcd`` (= the LLS
    prior verbatim) × two prior-pinned RATIOS ``r_subdla``/``r_dla`` (see ``_hcd_sites``). The
    derived α are re-emitted as ``numpyro.deterministic`` under the EXISTING names; the existing
    per-class z-slope ``s_c`` is UNCHANGED, so the ratio already evolves with the DIFFERENTIAL CDDF
    slope r_c(z)=r_c·((1+z)/(1+z_p))^(s_c−s_LLS) (the must-fix #2 fix is FREE).

    FUNNEL (centered is correct — must-fix #6): the classic Neal funnel needs a WIDE multiplier ×
    a likelihood-CONSTRAINED latent. Here the orientation is BENIGN: the likelihood-constrained
    factor is the MULTIPLIER ``A_hcd`` (the dominant, cosmology-degenerate, best-resolved LLS
    amplitude), while the PINNED factor is the ratio ``r`` (σ_r ≪ the likelihood's r-resolving
    power). This is NOT "the likelihood barely sees r" — subDLA IS likelihood-informed
    (post_sd 0.021 < prior 0.036); the point is WHICH factor the likelihood pins. Centered, dense
    mass, target 0.9 → expect 0 divergences. ``ctx.hcd_noncentered`` is a tested LocScaleReparam-
    style fallback (same site names → site-order preserved); the 0-divergence gate is the empirical
    check (test_legb_hier_hcd.test_on_branch_nuts_zero_divergences_and_finite)."""
    zg = jnp.asarray(ctx.z_global)
    _lo_u = jnp.asarray(_THETA_UNIT_LO if getattr(ctx, "theta_unit_lo", None) is None else ctx.theta_unit_lo)
    _hi_u = jnp.asarray(_THETA_UNIT_HI if getattr(ctx, "theta_unit_hi", None) is None else ctx.theta_unit_hi)
    theta9 = numpyro.sample("theta_unit", dist.Uniform(_lo_u, _hi_u).to_event(1))
    kim = _kim(zg)
    # PRIYA mean-flux: α(z)=τ₀·((1+z)/(1+z_p))^dτ₀, τ₀(z)=α·Kim07 — 2 GLOBAL uniform params, NOT
    # 13 free per-z rungs, so τ₀ cannot absorb emulator residual into per-z wiggle that biases A_p.
    tau0_amp, dtau0 = _sample_tau0_sites(ctx)
    alpha_z = tau0_alpha_priya(zg, tau0_amp, dtau0, z_pivot=ctx.tau0_pivot_z)
    tau0_global = numpyro.deterministic("tau0_vec", alpha_z * kim)
    # The HCD pivot-z (z=3) amplitudes (3,) [LLS, subDLA, DLA]. The shared ``_hcd_sites`` helper
    # samples EITHER the legacy 3 independent α sites (alpha_lls/subdla TruncatedNormal(low=0),
    # alpha_dla_raw Normal→softplus; default — byte-exact) OR, when ctx.hierarchical_hcd, the
    # reparam A_hcd · r → α (re-emitted as deterministics under the SAME names alpha_lls/subdla/dla).
    alpha_pivot, s_override = _hcd_sites(ctx)  # (3,) pivot-z amplitudes (+ optional s_c override)
    # z-RESOLVED incidence α_c(z) = α_pivot · ((1+z)/(1+z_p))^s_c (the dN/dX slope) — the fix:
    # the forward must track the mock's per-z w_c(z) (rises ~3.5× over z), not a z-constant α.
    # STEP-A M3: when ctx.marginalize_zslope, s_c is SAMPLED (the real-fit config) centered on
    # HCD_INCIDENCE_SLOPE (~2.4, the SIM incidence-weight slope the mock truth carries); otherwise
    # s_c is FIXED to that same incidence slope (NOT the lit/sim ratio HCD_LIT_OVER_SIM_SLOPE).
    # 2D AMPLITUDE×TILT mode: _hcd_sites returns s_override = B_hcd + δs_c (it sets the slopes via
    # the global tilt B_hcd), which BYPASSES _zslope_sites / marginalize_zslope entirely.
    s_c = s_override if s_override is not None else _zslope_sites(ctx)
    shape_zg = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c
    alpha_hcd = numpyro.deterministic("alpha_hcd_z", alpha_pivot[None, :] * shape_zg)  # (n_zg,3)
    # SHARED SiIII metal amplitude (opt-in; eBOSS+DESI are metals_on, KS is not). OFF by default
    # (a_SiIII=0 ⇒ _metal_factor≡1 ⇒ golden byte-exact). The prior on this site is selected by the
    # STATIC ctx.metal_prior via _metal_amp_site: "uniform" (default, Uniform[0,a_siiii_max] — the
    # physical a_SiIII=f_SiIII/(1−⟨F⟩)≈0.045 sits inside) or "flatlog" (LogUniform on a=f/(1−F_ref)).
    # STATIC branch on ctx.metal_prior (a python str, resolved at trace time → a legal branch). MODEL C
    # ("flatlog2node"): sample the per-leg 2-node f sites (_metal_2node_sites) ⇒ per-z a(z)=f(z)/(1−⟨F⟩(z))
    # in the forward; the scalar a_SiIII/a_SiII are 0 (unused). "uniform"/"flatlog": the legacy scalar
    # sites via _metal_amp_site (byte-exact). metal_nodes is threaded into _data_loglik_legcore.
    if ctx.metal_prior == "flatlog2node":
        metal_nodes = _metal_2node_sites(ctx)
        a_siiii = a_siii = 0.0
    else:
        metal_nodes = None
        a_siiii = _metal_amp_site("a_SiIII", ctx, getattr(ctx, "sample_metals", False))
        # SiII DOUBLET amplitude (Stage C, opt-in ctx.sample_a_siii, default OFF → a_SiII=0 ⇒ byte-exact).
        # _metal_factor's SiII is now the true 1190.42+1193.28 doublet, so a floated a_SiII absorbs the
        # doublet the metal_misspec injection carries (the NUTS-settled −0.69 n_s driver). MUST be sampled
        # RIGHT AFTER a_SiIII and BEFORE alpha_res so _legb_priors_only's mirror order matches (constrain_fn).
        # SAME _metal_amp_site helper ⇒ SAME prior mode + byte-identical site construction as the mirror.
        a_siii = _metal_amp_site("a_SiII", ctx, getattr(ctx, "sample_a_siii", False))
    # res_corr AMPLITUDE nuisance (Task 1.3): α(z)=α₀·((1+z)/(1+Z_PIVOT))^s, marginalized
    # FORWARD-ONLY (threaded into _data_loglik_legcore → predict_P_obs_on_leg → the MF
    # chokepoint; NOT into the truth → no closure cancellation). Two sites, amplitude THEN
    # slope; _legb_priors_only MUST mirror this order (constrain_fn traces it).
    # DIAGNOSTIC (ctx.fix_alpha_res, default False → SAMPLE both sites, byte-identical): when
    # True, DO NOT sample alpha_res/alpha_res_slope — pin them to the fixed no-op (alpha0=1, s=0,
    # i.e. α(z)≡1, the production res_corr UNMODIFIED) in the forward. This isolates the
    # alpha-marginalization's contribution to the n_s bias (the decomp diagnostic).
    if getattr(ctx, "fix_alpha_res", False):
        alpha_res, alpha_res_slope = 1.0, 0.0
    else:
        alpha_res = numpyro.sample("alpha_res", dist.TruncatedNormal(1.0, SIGMA_A0, low=0.0))
        alpha_res_slope = numpyro.sample("alpha_res_slope", dist.Normal(0.0, SIGMA_S))
    # SPECTRAL-RESOLUTION nuisance f_res (option-b, Gate B; sampled iff ctx.sample_res, default False →
    # None → byte-identical golden). Forward-only b_res(z) on z_global, sliced per leg + threaded into
    # _resolution_factor; the injected truth never carries it (no closure cancellation, like alpha_res).
    # Sites appended AFTER alpha_res; _legb_priors_only mirrors this order (constrain_fn parity).
    if getattr(ctx, "sample_res", False):
        _amp_sig = getattr(ctx, "f_res_amp_sigma", None)
        _amp_sig = F_RES_AMP_SIGMA if _amp_sig is None else float(_amp_sig)   # arm-C wide / eBOSS leg-match
        f_res_amp = numpyro.sample("f_res_amp", dist.Normal(0.0, _amp_sig))
        f_res_slope = numpyro.sample("f_res_slope", dist.Normal(0.0, F_RES_SLOPE_SIGMA))
        b_res_global = _bres_of_z(zg, f_res_amp, f_res_slope)
    else:
        b_res_global = None
    numpyro.factor("loglik", _data_loglik_legcore(
        ctx, theta9, tau0_global, alpha_hcd, mock_legs, dla_core_per_leg, a_siiii=a_siiii,
        a_siii=a_siii, metal_nodes=metal_nodes, alpha_res=(alpha_res, alpha_res_slope),
        b_res_global=b_res_global, require_zresolved=True))   # alpha_hcd = z-resolved alpha_hcd_z


def _hcd_sites(ctx):
    """The HCD pivot-z (z=3) incidence amplitudes α_pivot (3,) [LLS, subDLA, DLA] AND an optional
    per-class z-slope override. Returns ``(alpha_pivot (3,), s_override)`` — ``s_override`` is
    ``None`` (legacy / Option B → ``_legb_model`` calls ``_zslope_sites`` as today) or the (3,)
    slopes s_c = B_hcd + δs_c (the 2D AMPLITUDE×TILT mode, which sets its OWN slopes and so
    BYPASSES ``_zslope_sites`` / ``marginalize_zslope``). SHARED by ``_legb_model`` (with the
    factor) and ``_legb_priors_only`` (the transform-only postprocess) so the sample sites stay
    ORDER-IDENTICAL across both — the fast-postprocess constrain_fn relies on it.

    LEGACY branch (default, ``ctx.hierarchical_hcd`` False → BYTE-EXACT): the 3 independent sites
      alpha_lls/alpha_subdla ~ TruncatedNormal(μ_c, σ_c, low=0); alpha_dla_raw ~ Normal →
      alpha_dla = deterministic(softplus(raw)). Identical to the pre-Option-B code.

    HIERARCHICAL branch (``ctx.hierarchical_hcd`` True): ONE likelihood-constrained MULTIPLIER
      A_hcd ~ TruncatedNormal(μ_LLS, σ_LLS, low=0)   (= the LLS prior verbatim)
      r_subdla ~ TruncatedNormal(r_sub_center, σ_r_sub, low=0)
      r_dla    ~ TruncatedNormal(r_dla_center, σ_r_dla, low=0)
      α = deterministic: alpha_lls=A_hcd, alpha_subdla=A_hcd·r_subdla, alpha_dla=A_hcd·r_dla
      (re-emitted under the EXISTING names → all downstream code is unchanged). When
      ``ctx.hcd_noncentered``, the SAME three sites are sampled standard-normal/uniform-base and
      transformed (a LocScaleReparam-style non-centered fallback) — site NAMES are identical so the
      site-order test passes; the 0-divergence gate decides centered vs non-centered.

    2D AMPLITUDE×TILT branch (``ctx.hcd_2d_tilt`` True, requires hierarchical_hcd): the same A_hcd
      and r_subdla/r_dla, PLUS a GLOBAL z-tilt B_hcd ~ Normal(hcd_btilt_mu, hcd_btilt_sigma)
      sampled RIGHT AFTER A_hcd (so the site order is A_hcd, B_hcd, r_subdla, r_dla). The pivot α
      are still A_hcd·r (the B_hcd z-evolution lives in alpha_hcd_z, NOT the pivot — back-compat for
      _draws_matrix/coverage). Returns s_override = B_hcd + δs_c (the per-class slope), which
      REPLACES _zslope_sites (the 2D mode sets its own slopes — marginalize_zslope is bypassed)."""
    if not getattr(ctx, "hierarchical_hcd", False):
        # ---- LEGACY (byte-exact) ----
        a_lls = numpyro.sample("alpha_lls",
                               dist.TruncatedNormal(ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0], low=0.0))
        a_sub = numpyro.sample("alpha_subdla",
                               dist.TruncatedNormal(ctx.alpha_hcd_mu[1], ctx.alpha_hcd_sigma[1], low=0.0))
        # NOTE (PR#12 e2e-review): the latent Normal SCALE is hardcoded 1.0 — NOT ctx.alpha_hcd_sigma[2].
        # So α_DLA's encoded prior is a deliberately-BROAD one-sided softplus(Normal) with effective
        # σ/μ ≈ 1.2 (right-skewed) — ~2.5× the named HCD_PRIOR_FRAC_SIGMA[2]=0.50 — and the z>3.5
        # dla_inflate widening of σ_DLA is INERT here. This is the conservative direction (the DLA sector
        # is un-certified by Leg B / needs arm A3). DO NOT rescale unilaterally: closure_mocks.py:56 and
        # sampler_numpyro.py draw the mock TRUTH from this IDENTICAL dist, so any width change must be
        # mirrored at all three sites or it breaks self-draw SBC rank-uniformity.
        a_dla_raw = numpyro.sample("alpha_dla_raw",
                                   dist.Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0))
        a_dla = numpyro.deterministic("alpha_dla", jax.nn.softplus(a_dla_raw))
        return jnp.stack([a_lls, a_sub, a_dla]), None

    # ---- HIERARCHICAL (Option B) / 2D AMPLITUDE×TILT ----
    two_d = getattr(ctx, "hcd_2d_tilt", False)
    A_mu, A_sg = ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]        # A_hcd = the LLS prior verbatim
    rmu = jnp.asarray(ctx.hcd_ratio_mu)                             # (2,) [r_sub, r_dla] centers
    rsg = jnp.asarray(ctx.hcd_ratio_sigma)                          # (2,) widths
    if getattr(ctx, "hcd_noncentered", False):
        # LocScaleReparam-style: sample a standard base, transform to the truncated-at-0 quantity.
        # We use the (rare) explicit non-centered form numpyro's TransformReparam would synthesize,
        # but keep the SAMPLE-SITE names identical (A_hcd/r_subdla/r_dla) so the site-order test is
        # unaffected. The base is a unit TruncatedNormal shifted/scaled — equivalent in distribution
        # to the centered TruncatedNormal(low=0), with the funnel-friendly geometry decoupled.
        z_A = numpyro.sample("A_hcd_base", dist.TruncatedNormal(0.0, 1.0, low=-A_mu / A_sg))
        A_hcd = numpyro.deterministic("A_hcd", A_mu + A_sg * z_A)
        s_override = _btilt_site(ctx) if two_d else None           # B_hcd sampled AFTER A_hcd
        z_rs = numpyro.sample("r_subdla_base", dist.TruncatedNormal(0.0, 1.0, low=-rmu[0] / rsg[0]))
        r_subdla = numpyro.deterministic("r_subdla", rmu[0] + rsg[0] * z_rs)
        z_rd = numpyro.sample("r_dla_base", dist.TruncatedNormal(0.0, 1.0, low=-rmu[1] / rsg[1]))
        r_dla = numpyro.deterministic("r_dla", rmu[1] + rsg[1] * z_rd)
    else:
        A_hcd = numpyro.sample("A_hcd", dist.TruncatedNormal(A_mu, A_sg, low=0.0))
        s_override = _btilt_site(ctx) if two_d else None           # B_hcd sampled AFTER A_hcd
        r_subdla = numpyro.sample("r_subdla", dist.TruncatedNormal(rmu[0], rsg[0], low=0.0))
        r_dla = numpyro.sample("r_dla", dist.TruncatedNormal(rmu[1], rsg[1], low=0.0))
    # derived α under the EXISTING names (so _draws_matrix / _loglik_of_draws / corner are unchanged).
    # In the 2D mode this is the PIVOT (z_p) α — the z-evolution lives in alpha_hcd_z via s_override.
    a_lls = numpyro.deterministic("alpha_lls", A_hcd)
    a_sub = numpyro.deterministic("alpha_subdla", A_hcd * r_subdla)
    a_dla = numpyro.deterministic("alpha_dla", A_hcd * r_dla)
    return jnp.stack([a_lls, a_sub, a_dla]), s_override


def _btilt_site(ctx):
    """The GLOBAL HCD z-tilt site B_hcd ~ Normal(hcd_btilt_mu, hcd_btilt_sigma) → the per-class
    slope vector s_c = B_hcd + δs_c (2D AMPLITUDE×TILT mode). δs_LLS≡0, so at B_hcd=hcd_btilt_mu
    the slopes equal the full incidence slope HCD_INCIDENCE_SLOPE (~2.4 — the closure anchor that
    MATCHES the mock truth's native w_c(z) evolution)."""
    # GUARD the 2D-tilt forward exponent center (Gap-1, hcd-dndx-zslope-bug): ctx.hcd_btilt_mu is
    # the CONCRETE prior center (trace-safe), NOT the sampled B_hcd — catches a 0.95 reversion that
    # a future ctx._replace(hcd_btilt_mu=...) / new builder could otherwise slip past the build guard.
    _assert_forward_zslope_center(ctx.hcd_btilt_mu, "_btilt_site")
    B_hcd = numpyro.sample("B_hcd", dist.Normal(ctx.hcd_btilt_mu, ctx.hcd_btilt_sigma))
    return B_hcd + jnp.asarray(ctx.hcd_dslope)                      # (3,) s_c = B_hcd + δs_c


def _zslope_sites(ctx):
    """The HCD per-class z-slope s_c (3,). FIXED to HCD_INCIDENCE_SLOPE unless
    ``ctx.marginalize_zslope`` (STEP-A M3) — then SAMPLE s_lls/s_subdla/s_dla ~ Normal at the
    ctx prior (default: center HCD_INCIDENCE_SLOPE, width ZSLOPE_PRIOR_SIGMA). Shared by
    ``_legb_model`` (with the factor) and ``_legb_priors_only`` (transform-only postprocess).

    CENTER = HCD_INCIDENCE_SLOPE (~2.4) — the SIM incidence-WEIGHT slope d ln w_c(z)/d ln(1+z)
    the held-out-sim mock TRUTH actually carries, so the forward dN/dX(z) RISES with z (matching
    the truth + literature). DISTINCT from inference.HCD_LIT_OVER_SIM_SLOPE (~0.95, the lit/sim
    RATIO slope — a z=3-pivot prior-center quantity): centering s_c on the ratio slope made the
    predicted dN/dX(z) FALL with z and put the mock truth 2.9–6σ off-center (the wrong-object bug).
    Matches the already-correct 2D-tilt anchor (_btilt_site / build_legb_ctx → HCD_INCIDENCE_SLOPE)."""
    if not getattr(ctx, "marginalize_zslope", False):
        _assert_forward_zslope_center(HCD_INCIDENCE_SLOPE, "_zslope_sites fixed")
        return jnp.asarray(HCD_INCIDENCE_SLOPE)
    mu_src = HCD_INCIDENCE_SLOPE if ctx.zslope_mu is None else ctx.zslope_mu
    _assert_forward_zslope_center(mu_src, "_zslope_sites marginalize_zslope center")
    mu = jnp.asarray(mu_src)
    sg = (jnp.asarray(ZSLOPE_PRIOR_SIGMA) if ctx.zslope_sigma is None
          else jnp.asarray(ctx.zslope_sigma))
    s_lls = numpyro.sample("s_lls", dist.Normal(mu[0], sg[0]))
    s_sub = numpyro.sample("s_subdla", dist.Normal(mu[1], sg[1]))
    s_dla = numpyro.sample("s_dla", dist.Normal(mu[2], sg[2]))
    return jnp.stack([s_lls, s_sub, s_dla])


def _legb_priors_only(ctx):
    """The SAME prior sample sites as ``_legb_model`` but with NO ``numpyro.factor`` loglik and
    NO deterministic sites — used only as the model passed to ``constrain_fn`` in the cheap
    transform-only postprocess (below). Tracing this is cheap (priors, no 681×681 Cholesky)."""
    _lo_u = jnp.asarray(_THETA_UNIT_LO if getattr(ctx, "theta_unit_lo", None) is None else ctx.theta_unit_lo)
    _hi_u = jnp.asarray(_THETA_UNIT_HI if getattr(ctx, "theta_unit_hi", None) is None else ctx.theta_unit_hi)
    numpyro.sample("theta_unit", dist.Uniform(_lo_u, _hi_u).to_event(1))
    _sample_tau0_sites(ctx)
    # the HCD pivot α sites — SHARED with _legb_model via _hcd_sites so the sample-site order is
    # IDENTICAL in both branches (legacy 3-site vs hierarchical A_hcd/r_subdla/r_dla, vs 2D
    # A_hcd/B_hcd/r_subdla/r_dla). The deterministics it emits are dropped by
    # constrain_fn(return_deterministic=False).
    _alpha_pivot, _s_override = _hcd_sites(ctx)
    # 2D mode sets s_c via B_hcd (sampled inside _hcd_sites) and BYPASSES _zslope_sites — so mirror
    # _legb_model: only call _zslope_sites when _hcd_sites did NOT supply the slopes (legacy/Option B).
    if _s_override is None:
        _zslope_sites(ctx)                        # mirrors _legb_model (s_c when marginalize_zslope)
    # MUST mirror _legb_model's metal sites EXACTLY (same name, order, distribution) — constrain_fn
    # traces this. SAME static branch as _legb_model: MODEL C ("flatlog2node") → the per-leg 2-node f
    # sites via the SHARED _metal_2node_sites (identical leg loop/order = constrain_fn parity); else the
    # legacy scalar a_SiIII/a_SiII via _metal_amp_site (a_SiIII before a_SiII before the res_corr block).
    if ctx.metal_prior == "flatlog2node":
        _metal_2node_sites(ctx)
    else:
        _metal_amp_site("a_SiIII", ctx, getattr(ctx, "sample_metals", False))
        _metal_amp_site("a_SiII", ctx, getattr(ctx, "sample_a_siii", False))
    # res_corr AMPLITUDE nuisance (Task 1.3) — MUST mirror _legb_model's two sites in the SAME
    # order (amplitude before slope), at the SAME relative position (last), or constrain_fn corrupts.
    # DIAGNOSTIC fix_alpha_res: when set, _legb_model does NOT sample these two sites, so the
    # priors-only mirror MUST drop them too (else constrain_fn's site set desyncs).
    if not getattr(ctx, "fix_alpha_res", False):
        numpyro.sample("alpha_res", dist.TruncatedNormal(1.0, SIGMA_A0, low=0.0))
        numpyro.sample("alpha_res_slope", dist.Normal(0.0, SIGMA_S))
    # option-b f_res mirror (MUST match _legb_model's order + WIDTH: after alpha_res). constrain_fn parity.
    if getattr(ctx, "sample_res", False):
        _amp_sig = getattr(ctx, "f_res_amp_sigma", None)
        _amp_sig = F_RES_AMP_SIGMA if _amp_sig is None else float(_amp_sig)
        numpyro.sample("f_res_amp", dist.Normal(0.0, _amp_sig))
        numpyro.sample("f_res_slope", dist.Normal(0.0, F_RES_SLOPE_SIGMA))


def _legb_reconstruct_deterministics(ctx, samples):
    """Reconstruct the ``numpyro.deterministic`` sites of ``_legb_model``
    (``tau0_vec``, ``alpha_dla``, ``alpha_hcd_z`` and — in the hierarchical branch — also
    ``alpha_lls``/``alpha_subdla``) host-side from the raw latent samples — so the cheap
    (priors-only) postprocess can SKIP the per-sample full-model deterministic replay (the
    237.6 s/mock waste flagged in MF-SMOKE-01) yet return a samples dict BYTE-IDENTICAL to
    numpyro's default ``get_samples()``. Returns a NEW dict (the input plus the deterministic keys).

    HIERARCHICAL seam (must-fix #4): in the Option-B branch, ``constrain_fn(return_deterministic=
    False)`` keeps only the SAMPLE sites (A_hcd/r_subdla/r_dla, or their *_base latents in the
    non-centered fallback) and DROPS the derived α deterministics — but ``_draws_matrix`` /
    ``_loglik_of_draws`` read alpha_lls/alpha_subdla/alpha_dla BY NAME. So we MUST rebuild AND
    RE-INSERT all three (alpha_lls=A_hcd, alpha_subdla=A_hcd·r_subdla, alpha_dla=A_hcd·r_dla)."""
    out = dict(samples)
    zg = jnp.asarray(ctx.z_global)
    kim = _kim(zg)
    tau0_amp = jnp.asarray(samples["tau0_amp"])                      # (L,)
    dtau0 = jnp.asarray(samples["dtau0"])                            # (L,)
    alpha_z = tau0_amp[:, None] * ((1.0 + zg)[None, :] / (1.0 + ctx.tau0_pivot_z)) ** dtau0[:, None]
    tau0_vec = alpha_z * kim[None, :]                                # (L, nZg)
    if getattr(ctx, "hierarchical_hcd", False):
        # rebuild A_hcd/r from the *_base latents in the non-centered fallback (constrain_fn drops
        # the A_hcd/r deterministics there); else they are the sample sites directly.
        if "A_hcd" in samples:
            A_hcd = jnp.asarray(samples["A_hcd"])
            r_subdla = jnp.asarray(samples["r_subdla"])
            r_dla = jnp.asarray(samples["r_dla"])
        else:
            A_mu, A_sg = ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]
            rmu = jnp.asarray(ctx.hcd_ratio_mu); rsg = jnp.asarray(ctx.hcd_ratio_sigma)
            A_hcd = A_mu + A_sg * jnp.asarray(samples["A_hcd_base"])
            r_subdla = rmu[0] + rsg[0] * jnp.asarray(samples["r_subdla_base"])
            r_dla = rmu[1] + rsg[1] * jnp.asarray(samples["r_dla_base"])
            out["A_hcd"] = A_hcd
            out["r_subdla"] = r_subdla
            out["r_dla"] = r_dla
        alpha_lls = A_hcd                                            # (L,)
        alpha_subdla = A_hcd * r_subdla                              # (L,)
        alpha_dla = A_hcd * r_dla                                    # (L,)
        out["alpha_lls"] = alpha_lls
        out["alpha_subdla"] = alpha_subdla
    else:
        alpha_lls = jnp.asarray(samples["alpha_lls"])
        alpha_subdla = jnp.asarray(samples["alpha_subdla"])
        alpha_dla = jax.nn.softplus(jnp.asarray(samples["alpha_dla_raw"]))  # (L,)
    alpha_pivot = jnp.stack([alpha_lls, alpha_subdla, alpha_dla], axis=-1)  # (L, 3)
    if getattr(ctx, "hcd_2d_tilt", False) and "B_hcd" in samples:
        # 2D AMPLITUDE×TILT: the per-class slope s_c = B_hcd + δs_c (NOT a sampled s_* block). The
        # pivot α (A·r) carry the amplitude; the z-evolution is the B_hcd-driven power-law.
        B = jnp.asarray(samples["B_hcd"])                            # (L,)
        s_c = B[:, None] + jnp.asarray(ctx.hcd_dslope)[None, :]      # (L, 3)
        ratio = (1.0 + zg)[None, :, None] / (1.0 + HCD_Z_PIVOT)
        shape_zg = ratio ** s_c[:, None, :]                          # (L, nZg, 3)
        alpha_hcd_z = alpha_pivot[:, None, :] * shape_zg             # (L, nZg, 3)
    elif getattr(ctx, "marginalize_zslope", False) and "s_lls" in samples:
        s_c = jnp.stack([jnp.asarray(samples["s_lls"]),
                         jnp.asarray(samples["s_subdla"]),
                         jnp.asarray(samples["s_dla"])], axis=-1)     # (L, 3)
        # shape_zg per draw: ((1+z)/(1+z_p))^{s_c} → (L, nZg, 3)
        ratio = (1.0 + zg)[None, :, None] / (1.0 + HCD_Z_PIVOT)
        shape_zg = ratio ** s_c[:, None, :]                          # (L, nZg, 3)
        alpha_hcd_z = alpha_pivot[:, None, :] * shape_zg             # (L, nZg, 3)
    else:
        # fixed-slope fallback (non-2D, non-marginalized readout) — byte-consistent with the
        # _zslope_sites FIXED branch: the SIM incidence slope HCD_INCIDENCE_SLOPE (~2.4), NOT the
        # lit/sim ratio HCD_LIT_OVER_SIM_SLOPE (the wrong-object slope).
        shape_zg = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** jnp.asarray(HCD_INCIDENCE_SLOPE)
        alpha_hcd_z = alpha_pivot[:, None, :] * shape_zg[None, :, :]      # (L, nZg, 3)
    out["tau0_vec"] = tau0_vec
    out["alpha_dla"] = alpha_dla
    out["alpha_hcd_z"] = alpha_hcd_z
    return out


def _run_nuts_legb(ctx, mock_legs, dla_core_per_leg, *, n_warmup, n_samples, seed,
                   target_accept=0.9, dense_mass=True, max_tree_depth=10,
                   fast_postprocess=True, init_strategy=None, return_extra=False):
    """NUTS on the Leg-B model. PRODUCTION uses ``dense_mass=True`` (the [θ,τ₀] correlation is
    the physics) + ``max_tree_depth=10``. The SMOKE caps ``max_tree_depth`` (and may use a
    diagonal mass) to bound the per-step leapfrog count — the dense-mass adaptation on the
    26-dim θ+τ₀+α posterior is expensive (early warmup hits the max tree depth before the
    mass matrix conditions the geometry). The cap only affects efficiency, not correctness.

    ``init_strategy`` (default ``init_to_median``): the numpyro init. The convergence path
    (``run_legb_convergence``) passes ``init_to_sample`` (a DISPERSED prior draw per chain) so
    the chains start spread across the prior — a PRE-REQUISITE for split-R-hat to be a valid
    convergence diagnostic (identical ``init_to_median`` starts defeat the between-chain
    variance R-hat relies on; CRITICAL, val-review bayesian §5/concern-1).

    ``return_extra=True`` also returns the raw extra-field dict (``diverging``, ``energy``,
    ``num_steps``) the convergence battery needs (E-BFMI from energy, max-tree-depth saturation
    from num_steps); the default keeps the legacy ``(samples, n_div)`` 2-tuple.

    EFFICIENCY (MF-SMOKE-01 fix): numpyro's DEFAULT postprocess re-runs the FULL ``_legb_model``
    (incl. the 681×681 + 132×132 Cholesky ``numpyro.factor`` loglik) once per collected sample
    just to recover the ``deterministic`` sites — 237.6 s/mock of pure waste. With
    ``fast_postprocess=True`` (default) we instead pass a TRANSFORM-ONLY postprocess that
    constrains via the cheap PRIORS-ONLY model (no loglik factor) and reconstruct ``tau0_vec`` /
    ``alpha_dla`` / ``alpha_hcd_z`` host-side. The returned samples dict is byte-identical to the
    default (verified rtol=0). Set ``fast_postprocess=False`` to restore the legacy replay."""
    strat = init_to_median if init_strategy is None else init_strategy
    kernel = NUTS(lambda: _legb_model(ctx, mock_legs, dla_core_per_leg),
                  dense_mass=bool(dense_mass), target_accept_prob=float(target_accept),
                  max_tree_depth=int(max_tree_depth), init_strategy=strat)
    postprocess_fn = None
    if fast_postprocess:
        # numpyro calls postprocess_fn(z_dict) per collected sample; constrain via the
        # priors-only trace (cheap) — no deterministic replay, no loglik factor.
        def postprocess_fn(z):
            return constrain_fn(lambda: _legb_priors_only(ctx), (), {}, z,
                                return_deterministic=False)
    mcmc = MCMC(kernel, num_warmup=int(n_warmup), num_samples=int(n_samples),
                num_chains=1, progress_bar=False, postprocess_fn=postprocess_fn)
    # accept either a PRNGKey (convergence path seeds chains on a separate fold_in axis) or an
    # int seed (the legacy base_seed+attempt retry stream).
    run_key = seed if isinstance(seed, jax.Array) else jax.random.PRNGKey(int(seed))
    mcmc.run(run_key, extra_fields=("diverging", "energy", "num_steps"))
    samples = mcmc.get_samples()
    if fast_postprocess:
        samples = _legb_reconstruct_deterministics(ctx, samples)
    ef = mcmc.get_extra_fields()
    diverging = np.asarray(ef.get("diverging", np.zeros(0, bool)))
    n_div = int(diverging.sum())
    if return_extra:
        extra = dict(diverging=diverging,
                     energy=np.asarray(ef.get("energy", np.zeros(0))),
                     num_steps=np.asarray(ef.get("num_steps", np.zeros(0, int))))
        return samples, n_div, extra
    return samples, n_div


# ============================================================================ #
#  STEP-A convergence battery (rank-R-hat / bulk+tail-ESS / E-BFMI) + multichain.
#
#  arviz is NOT installed in emu-jax; these are the standard Vehtari+2021 (2008.10250)
#  estimators implemented on numpyro's (chain,draw) primitives (split_gelman_rubin,
#  effective_sample_size) — rank-normalize/fold the pooled draws first, exactly as arviz
#  does for ``az.rhat(method="rank")`` / ``az.ess(method={"bulk","tail"})``.
# ============================================================================ #
def _rank_normalize(x):
    """Rank-normalize a (C,N) array over the POOLED CN draws → normal scores via the
    Blom (r-3/8)/(n-1/4) plotting position + Φ⁻¹ (the arviz/Vehtari rank-R-hat transform).
    Returns the same (C,N) shape."""
    x = np.asarray(x, float)
    C, N = x.shape
    flat = x.reshape(-1)
    # average ranks (1..CN), ties → mean rank (matches scipy 'average').
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    # resolve ties to the mean rank within each tie group.
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    mean_rank = (start + csum + 1) / 2.0     # mean of the integer ranks in [start+1, csum]
    ranks = mean_rank[inv]
    z = _scipy_norm.ppf((ranks - 3.0 / 8.0) / (flat.size - 0.25))
    return z.reshape(C, N)


def _ess_indicator(x, q):
    """tail-ESS building block: ESS of the indicator series 1[x <= quantile_q(x)] (Vehtari+2021
    §4.3). ``x`` is (C,N); the quantile is over the POOLED draws."""
    x = np.asarray(x, float)
    thr = np.quantile(x, q)
    ind = (x <= thr).astype(float)
    if ind.std() == 0:                        # degenerate (all on one side) → ESS undefined
        return np.nan
    return float(npd.effective_sample_size(ind))


def convergence_battery(packed, names, *, energy=None, num_steps=None,
                        max_tree_depth=10, n_div=0):
    """The STEP-A convergence battery from MULTI-CHAIN packed draws.

    ``packed`` : (C, N, P) — C chains, N draws, P params (the ``_draws_matrix`` packing).
    ``names``  : (P,) param names.
    ``energy`` : (C, N) HMC energy per chain (for E-BFMI); ``num_steps`` (C, N) tree size.

    Returns a dict with PER-PARAM rank-normalized split-R-hat, bulk-ESS, tail-ESS, plus the
    scalar E-BFMI (min over chains), divergence count, and max-tree-depth saturation fraction.
    All standard (Vehtari+2021 / Betancourt+2016 E-BFMI). Computed with numpyro's
    ``split_gelman_rubin`` / ``effective_sample_size`` on rank-normalized/folded draws (the
    arviz rank-R-hat + bulk/tail-ESS recipe), since arviz is absent in this env."""
    packed = np.asarray(packed, float)
    C, N, P = packed.shape
    rhat = np.full(P, np.nan)
    ess_bulk = np.full(P, np.nan)
    ess_tail = np.full(P, np.nan)
    can_multichain = C >= 2 and N >= 4         # split_gelman_rubin needs draws ≥4
    for p in range(P):
        col = packed[:, :, p]                  # (C, N)
        zr = _rank_normalize(col)              # rank-normalized (folded by the normal scores)
        if can_multichain:
            rhat[p] = float(npd.split_gelman_rubin(zr))
        # bulk-ESS = ESS of the rank-normalized draws.
        ess_bulk[p] = float(npd.effective_sample_size(zr))
        # tail-ESS = min ESS of the 5%/95% quantile-indicator series (on the RAW draws).
        e05 = _ess_indicator(col, 0.05); e95 = _ess_indicator(col, 0.95)
        cands = [e for e in (e05, e95) if np.isfinite(e)]
        ess_tail[p] = float(min(cands)) if cands else np.nan

    # E-BFMI per chain (Betancourt 2016, 1604.00695 Eq. 6.1): Σ(ΔE)² / Σ(E-Ē)². Healthy ≳0.3.
    ebfmi = np.array([])
    if energy is not None and np.asarray(energy).size:
        en = np.asarray(energy, float)
        if en.ndim == 1:
            en = en[None, :]
        eb = []
        for c in range(en.shape[0]):
            e = en[c]
            denom = np.sum((e - e.mean()) ** 2)
            eb.append(float(np.sum(np.diff(e) ** 2) / denom) if denom > 0 else np.nan)
        ebfmi = np.array(eb)

    # max-tree-depth saturation: fraction of draws that hit the cap (2**mtd-1 leapfrogs).
    sat_frac = np.nan
    if num_steps is not None and np.asarray(num_steps).size:
        ns = np.asarray(num_steps).reshape(-1)
        sat_frac = float(np.mean(ns >= (2 ** int(max_tree_depth) - 1)))

    def _nanmin(a):
        a = np.asarray(a, float)
        return float(np.nanmin(a)) if a.size and np.isfinite(a).any() else np.nan

    def _nanmax(a):
        a = np.asarray(a, float)
        return float(np.nanmax(a)) if a.size and np.isfinite(a).any() else np.nan

    return dict(
        names=list(names), n_chains=C, n_draws=N,
        rhat={names[p]: float(rhat[p]) for p in range(P)},
        ess_bulk={names[p]: float(ess_bulk[p]) for p in range(P)},
        ess_tail={names[p]: float(ess_tail[p]) for p in range(P)},
        rhat_max=_nanmax(rhat) if can_multichain else np.nan,
        ess_bulk_min=_nanmin(ess_bulk),
        ess_tail_min=_nanmin(ess_tail),
        ebfmi=ebfmi, ebfmi_min=_nanmin(ebfmi),
        n_divergent=int(n_div), max_tree_depth=int(max_tree_depth),
        treedepth_sat_frac=sat_frac)


def run_legb_convergence(ctx: LegBCtx, d, *, sim=None, mock_index=0, n_chains=4,
                         n_warmup=250, n_samples=400, seed=0, fold=0,
                         dense_mass=True, max_tree_depth=10, target_accept=0.9,
                         chain_ids=None, verbose=True):
    """STEP-A convergence-MODE multichain fit of ONE mock (the R-hat path).

    Differs from ``run_legb`` (the coverage path) in three review-mandated ways:
      1. DISPERSED inits — each chain uses ``init_to_sample`` (an over-dispersed prior draw),
         NOT ``init_to_median`` (identical starts make split-R-hat meaningless; CRITICAL).
      2. warmup ≥ 150 (default 250) — 40 under-conditions the 25-dim dense mass.
      3. a SEPARATE seed axis — chains seed on ``fold_in(key_nuts, chain_id)``, DISTINCT from
         the ``base_seed + attempt`` divergence-retry stream of ``run_legb`` (no collision).

    SHARDABLE 1 chain/SLURM-task: pass a single ``chain_ids=[c]`` per task (and ``n_chains``
    is then just the merge target); each task computes its own chain and the host merges them
    for the battery. We DELIBERATELY do NOT use numpyro ``num_chains>1`` (on CPU it serializes
    the chains in one process — the wrapper's (mock,chain) sharding is the embarrassing-parallel
    form, val-review CS §c/SLURM).

    Returns ``dict(packed=(C,N,P), names, truth_vec, kept_global, battery, per_chain_div, sim)``.
    The convergence battery (``convergence_battery``) carries rank-R-hat / bulk+tail-ESS /
    E-BFMI / divergences / tree-depth saturation per param."""
    sims, _va = held_out_sims(d, fold=fold)
    if sim is None:
        sim = sims[mock_index % len(sims)]
    truth_sim = make_truth_from_sim(d, sim, fold=fold, mf=ctx.mf)

    # mock-noise key on the per-mock fold_in axis (SAME mock across all chains: chains differ
    # only in their NUTS seed, sharing one mock dataset — the R-hat between-chain variance is
    # then purely the sampler's, not the data's).
    key0 = jax.random.PRNGKey(int(seed))
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, int(mock_index)), 2)
    mock_legs, truth_pack, info = make_legb_mock(ctx, truth_sim, k_mock)
    core_per_leg = _mock_core_per_leg(ctx, truth_sim)
    kept_global = truth_pack["kept_global_z"]

    truth_vec = np.concatenate([
        truth_pack["theta9"], truth_pack["tau0_global"][kept_global], truth_pack["alpha_hcd"]])
    if getattr(ctx, "hcd_2d_tilt", False):                # 2D: A_hcd, B_hcd, r_subdla, r_dla
        truth_vec = np.concatenate([truth_vec, _hcd_latent_truths_2d(truth_pack["alpha_hcd"], ctx)])
    elif getattr(ctx, "hierarchical_hcd", False):
        truth_vec = np.concatenate([truth_vec, _hcd_latent_truths(truth_pack["alpha_hcd"])])

    ids = list(range(int(n_chains))) if chain_ids is None else list(chain_ids)
    packed_chains, energies, num_steps_all, per_chain_div, init_vals = [], [], [], [], []
    for cid in ids:
        # SEPARATE seed axis: fold_in(k_nuts, chain_id) — distinct from base_seed+attempt.
        chain_key = jax.random.fold_in(k_nuts, int(cid))
        samples, n_div, extra = _run_nuts_legb(
            ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
            seed=chain_key, target_accept=target_accept, dense_mass=dense_mass,
            max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
        draws = _draws_matrix(samples, kept_global)         # (N, P)
        packed_chains.append(draws)
        energies.append(extra["energy"]); num_steps_all.append(extra["num_steps"])
        per_chain_div.append(int(n_div))
        # record the per-chain INITIAL constrained draw (1st kept sample) to verify dispersion.
        init_vals.append(draws[0])
        if verbose:
            print(f"  [conv chain {cid}] sim={sim[:20]}… draws={draws.shape[0]} div={n_div} "
                  f"E-BFMI={_ebfmi1(extra['energy']):.2f}")

    packed = np.stack(packed_chains, axis=0)                # (C, N, P)
    # the packed-draw column names mirror _draws_matrix EXACTLY (θ9, τ₀(kept), α3, [A_hcd/r…],
    # [a_SiIII]) — built from the last chain's samples dict so the hierarchical latents / metal
    # nuisance columns are named when present (must-fix #5: index by name, not position).
    packed_names = _packed_names_for(samples, kept_global)

    battery = convergence_battery(
        packed, packed_names, energy=np.stack(energies) if energies else None,
        num_steps=np.stack(num_steps_all) if num_steps_all else None,
        max_tree_depth=max_tree_depth, n_div=int(sum(per_chain_div)))
    # per-chain init spread vs the within-chain posterior sd (dispersion check, R-hat validity).
    init_arr = np.stack(init_vals)                          # (C, P)
    post_sd = packed.reshape(-1, packed.shape[-1]).std(axis=0)
    init_spread = init_arr.std(axis=0)
    battery["init_spread"] = init_spread
    battery["post_sd"] = post_sd
    battery["init_spread_over_postsd"] = np.where(post_sd > 0, init_spread / post_sd, np.nan)
    return dict(packed=packed, names=packed_names, truth_vec=truth_vec,
                kept_global=kept_global, battery=battery,
                per_chain_div=per_chain_div, sim=sim)


def _ebfmi1(energy):
    """E-BFMI of a single chain's energy series (helper for the per-chain print)."""
    e = np.asarray(energy, float)
    if e.size < 2:
        return np.nan
    denom = np.sum((e - e.mean()) ** 2)
    return float(np.sum(np.diff(e) ** 2) / denom) if denom > 0 else np.nan


def _mock_core_per_leg(ctx, truth_sim):
    """z-mean DLA core per leg from the mock's sim (the forward core matched to the truth)."""
    z_sim = np.asarray(truth_sim["z"])
    core_sim = np.asarray(truth_sim["dla_core"])     # (nZs, K)
    out = {}
    for leg in ctx.legs:
        # map each leg z to the nearest sim z, take that core; z-mean over the leg's z.
        cores = []
        for zz in leg.z:
            j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) <= 0.15:
                cores.append(core_sim[j])
        out[leg.name] = jnp.asarray(np.mean(cores, axis=0) if cores else np.zeros(core_sim.shape[1]))
    return out


# ============================================================================ #
#  Leg-B driver.
# ============================================================================ #
# the packed param order for coverage/ranking: [θ9, τ₀(kept global z), α_lls,α_sub,α_dla].
def _draws_matrix(samples, kept_global):
    """Stack NUTS samples into (L,P) in the packed order, keeping only the KEPT global τ₀ z
    (the z the mock had sim data at). theta + α come from their sites; τ₀ from tau0_vec."""
    theta = np.asarray(samples["theta_unit"])               # (L,9)
    tau0 = np.asarray(samples["tau0_vec"])[:, kept_global]  # (L, nKept)
    a_lls = np.asarray(samples["alpha_lls"])[:, None]
    a_sub = np.asarray(samples["alpha_subdla"])[:, None]
    a_dla = np.asarray(samples["alpha_dla"])[:, None]
    cols = [theta, tau0, a_lls, a_sub, a_dla]               # the derived α (back-compat: ALWAYS here)
    # HIERARCHICAL latents (Option B): append A_hcd/r_subdla/r_dla AFTER the derived α (back-compat
    # — the α block stays at its positional home; downstream indexes α by name, must-fix #5). The
    # truth_vec tail (w_LLS, w_sub/w_LLS, 0.10·w_DLA/w_LLS) is appended in the driver to match.
    # 2D AMPLITUDE×TILT: B_hcd is inserted right after A_hcd (A_hcd, B_hcd, r_subdla, r_dla); the
    # truth_vec tail then carries (w_LLS, s_LLS_center, r_sub, r_dla) via _hcd_latent_truths_2d.
    for nm in ("A_hcd", "B_hcd", "r_subdla", "r_dla"):
        if nm in samples:
            cols.append(np.asarray(samples[nm])[:, None])
    if "a_SiIII" in samples:                                 # opt-in metal nuisance (appended LAST)
        cols.append(np.asarray(samples["a_SiIII"])[:, None])
    if "a_SiII" in samples:                                  # SiII doublet (appended AFTER a_SiIII)
        cols.append(np.asarray(samples["a_SiII"])[:, None])
    return np.concatenate(cols, axis=1)


def _hcd_latent_truths(alpha_hcd_truth):
    """The (A_hcd, r_subdla, r_dla) TRUTH for an HB mock from the truth α=[w_LLS, w_sub, 0.10·w_DLA]
    (truth_pack["alpha_hcd"], make_legb_mock:716-717). A_hcd=w_LLS; r_subdla=w_sub/w_LLS;
    r_dla=(0.10·w_DLA)/w_LLS — appended to truth_vec to align with the A_hcd/r_subdla/r_dla draw
    columns _draws_matrix adds in the hierarchical branch."""
    a = np.asarray(alpha_hcd_truth, float)
    return np.array([a[0], a[1] / a[0], a[2] / a[0]])


def _hcd_latent_truths_2d(alpha_hcd_truth, ctx):
    """The (A_hcd, B_hcd, r_subdla, r_dla) TRUTH for an HT (2D AMPLITUDE×TILT) mock. A_hcd=w_LLS;
    B_hcd_truth = hcd_btilt_mu = HCD_INCIDENCE_SLOPE[0] (the population LLS incidence slope ~2.46 —
    the mock's per-z α evolves at ~2.4, within the per-sim scatter ~0.07 of this; δs_LLS≡0);
    r_subdla=w_sub/w_LLS; r_dla=0.10·w_DLA/w_LLS. Appended to truth_vec to align with the
    A_hcd/B_hcd/r_subdla/r_dla draw columns _draws_matrix adds in the 2D branch."""
    a = np.asarray(alpha_hcd_truth, float)
    return np.array([a[0], float(ctx.hcd_btilt_mu), a[1] / a[0], a[2] / a[0]])


def _packed_names_for(samples, kept_global):
    """The packed-draw column NAMES matching ``_draws_matrix(samples, kept_global)``'s columns —
    [θ9, τ₀(kept), alpha_lls, alpha_subdla, alpha_dla, (A_hcd, r_subdla, r_dla), (a_SiIII)]. Built
    from the SAME presence checks so names stay aligned with the columns (the α stay at their
    positional home; the appended latents/metal go after — must-fix #5 indexes α by name)."""
    tau0_names = [f"tau0_z{i}" for i in range(int(np.asarray(kept_global).sum()))]
    names = list(PARAM_NAMES) + tau0_names + ["alpha_lls", "alpha_subdla", "alpha_dla"]
    for nm in ("A_hcd", "B_hcd", "r_subdla", "r_dla"):
        if nm in samples:
            names.append(nm)
    if "a_SiIII" in samples:
        names.append("a_SiIII")
    if "a_SiII" in samples:
        names.append("a_SiII")
    return names


def _metal_node_truth(metal_misspec, ctx):
    """Injected-arm TRUTH for the Model C+ metal node sites (``f_SiIII``/``f_SiII``/``k_SiIII``/
    ``k_SiII``_<leg>_z0/z1), keyed by site name, for the sites_extra ceiling-check instrumentation.
    Only the IN-CLASS ``form=="zevo"`` arms (arm1/arm2) have a defined node truth: f-node truth = the
    injected f at the FIT node-z (== the injected node value when inject/fit node_z match); k-node
    truth = the injected (flat-in-z) k. Clean (None) and out-of-class (ma2025/ma2025_gauss) → {} (the
    draws are still stored, with NaN truth; the mandatory pile-up check is truth-free)."""
    d = {}
    if not metal_misspec or metal_misspec.get("form") != "zevo":
        return d
    nz_inj = metal_misspec.get("node_z", (2.2, 4.2))
    fz = getattr(ctx, "metal_node_z", (2.2, 4.2))
    siII = tuple(getattr(ctx, "metal_siII_legs", ("DESI",)))
    for leg in ctx.legs:
        if not getattr(leg, "metals_on", False):
            continue
        for i in (0, 1):
            d[f"f_SiIII_{leg.name}_z{i}"] = float(_metal_f_of_z(fz[i], nz_inj, metal_misspec["f_SiIII_nodes"]))
            d[f"k_SiIII_{leg.name}_z{i}"] = float(metal_misspec["k_SiIII"])
            if leg.name in siII:
                d[f"f_SiII_{leg.name}_z{i}"] = float(_metal_f_of_z(fz[i], nz_inj, metal_misspec["f_SiII_nodes"]))
                d[f"k_SiII_{leg.name}_z{i}"] = float(metal_misspec["k_SiII"])
    return d


def _metal_node_sites_extra(samples, step, L, inject_spec, ctx, leg_a):
    """sites_extra entries for the Model C+ metal f/k node sites — sampled by ``_metal_2node_sites``
    but NOT packed into ``_draws_matrix`` — so the mandatory "is f_SiIII_z1 railing the 0.03 ceiling"
    check is possible from the shard pkls. Stores thinned draws (SAME step/L as tau0_amp/dtau0) + the
    injected-arm truth (NaN when clean/out-of-class). EMPTY unless flatlog2node actually sampled
    ``f_``/``k_`` nodes ⇒ additive-only, byte-identical golden under uniform/flatlog/metals-off. The
    scalar ``a_SiIII``/``a_SiII`` and the ``s_*`` slopes are NOT caught (they lack the f_/k_ prefix)."""
    mspec = (inject_spec or {}).get("metal_misspec") if (leg_a and inject_spec) else None
    mz_truth = _metal_node_truth(mspec, ctx)
    out = {}
    for nm in samples:
        if nm.startswith(("f_SiIII_", "f_SiII_", "k_SiIII_", "k_SiII_")):
            dr = np.asarray(samples[nm])[::step][:L]
            out[nm] = dict(draws=dr, truth=float(mz_truth.get(nm, np.nan)))
    return out


def _resolution_sites_extra(samples, step, L, inject_spec, leg_a):
    """sites_extra entries for the option-b spectral-resolution f_res sites (``f_res_amp``/``f_res_slope``)
    — sampled by ``_legb_model`` iff ``ctx.sample_res`` but NOT packed into ``_draws_matrix`` — so the
    rail/coverage check (is ``f_res_amp`` railing the tight N(0,0.02) prior, or is the leg speaking?) is
    possible from the shard pkls (step-review #4). Stores thinned draws (SAME step/L as tau0_amp/dtau0) +
    the injected-arm TRUTH: a constant-b_res injection is reproduced EXACTLY by (amp=b*, slope=0) in the
    2-param span, so truth = (injected b_res, 0.0); a vector/basis OUT-OF-SPAN injection (Task 2C:
    --b-res-oos-member, no scalar "b_res" key) is orthogonal to the (amp,slope) span BY CONSTRUCTION, so
    its in-span truth is (0.0, 0.0) -- no (amp,slope) pair reproduces it; NaN on the clean / held-out /
    non-resolution arm (the injection is a Leg-A-only hook). EMPTY unless sample_res actually sampled
    f_res ⇒ additive-only, byte-identical golden under sample_res=False."""
    res_inj = (inject_spec or {}).get("resolution") if (leg_a and inject_spec) else None
    if res_inj is None:
        truth = {"f_res_amp": np.nan, "f_res_slope": np.nan}
    elif "b_res" in res_inj:                      # scalar in-span injection: reproduced by (amp=b_res, slope=0)
        truth = {"f_res_amp": float(res_inj["b_res"]), "f_res_slope": 0.0}
    else:                                          # vector/basis OUT-OF-SPAN injection: orthogonal to the span
        truth = {"f_res_amp": 0.0, "f_res_slope": 0.0}   # no in-span (amp,slope) reproduces it
    out = {}
    for nm in ("f_res_amp", "f_res_slope"):
        if nm in samples:
            out[nm] = dict(draws=np.asarray(samples[nm])[::step][:L], truth=truth[nm])
    return out


def run_legb(ctx: LegBCtx, d, *, n_mocks, n_warmup, n_samples, seed,
             cemu_inflate=None, fold=0, q_levels=(0.68, 0.95), verbose=True,
             dense_mass=True, max_tree_depth=10, mock_indices=None,
             return_per_mock=False, leg_a=False, inject_spec=None):
    """Leg-B coverage over ``n_mocks`` held-out-sim mocks. Per mock: make_legb_mock → NUTS
    against the real-cov multi-leg likelihood → thin → rank the truth θ per param + the
    loglik rank → per-param empirical coverage at ``q_levels`` + bias.

    SHARDING: each mock ``m`` is seeded independently via ``jax.random.fold_in(seed, m)`` so a
    subset (``mock_indices``) reproduces exactly the same mocks a single full run would — i.e.
    a SLURM array can split ``range(n_mocks)`` across tasks and the merged set is identical.
    ``return_per_mock=True`` returns the raw per-mock list (for the cross-shard merge) instead
    of the aggregate.

    Returns ``_aggregate_legb`` (coverage 68/95% + per-param bias + the DIAGNOSTIC-ONLY rank
    ECDF), or the per-mock list if ``return_per_mock``."""
    # inject_spec is a LEG-A-ONLY hook (the data-nuisance bias gate). The held-out branch below
    # (make_legb_mock) does NOT thread it, so honouring it on a held-out run would SILENTLY drop the
    # injection. Fail loud instead — PR#12 review follow-up (b); generalizes the run_prod_sbc_shard.py
    # subdla_truth_boost assert to EVERY inject key (lls/subdla_truth_boost, metal_misspec, resolution).
    if inject_spec and not leg_a:
        raise ValueError(
            "run_legb: inject_spec is honoured only on the Leg-A self-draw path (leg_a=True); the "
            "held-out branch ignores it. Refusing to silently drop the injection on a held-out run.")
    # option-b f_res is a single GLOBAL site -> forbid a multi-instrument ctx until per-instrument sites
    # exist (4-referee panel #8). No-op when sample_res is off (golden-safe).
    _check_single_instrument_for_res(ctx.legs, getattr(ctx, "sample_res", False))
    if cemu_inflate is not None:
        ctx = ctx._replace(cemu_inflate=float(cemu_inflate))
    key0 = jax.random.PRNGKey(int(seed))
    idxs = list(range(int(n_mocks))) if mock_indices is None else list(mock_indices)
    if leg_a:
        # Leg-A rank-uniformity SBC: truths drawn from the PRIOR; the fiducial DLA core is the
        # MATCHED core used by both the mock forward and the likelihood (so C_mock ≡ C_like).
        # z-mean the (n_z,K) fiducial → the (K,) per-leg core predict_P_obs_on_leg expects.
        fid_core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
                    for name, v in _fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}
    else:
        sims, _va = held_out_sims(d, fold=fold)

    # cycle through the held-out sims (n_mocks may exceed the #sims → reuse with fresh noise).
    per_mock = []
    n_div_total = 0
    n_divergent = 0
    for m in idxs:
        if leg_a:
            # Leg-A self-draw: truth ~ prior, matched-C mock; a PURE fn of (seed,m) → shardable.
            k_truth, k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, m), 3)
            truth_pack = draw_leg_a_leg_truth(ctx, k_truth)
            # DATA-NUISANCE INJECTION (the bias gate; inject_spec=None ⇒ byte-identical to the
            # clean self-draw — the no-op guarantee). "lls_truth_boost" offsets the TRUTH LLS from
            # the per-survey pin BEFORE the forward; "metal_misspec"/"resolution" inject a mode the
            # forward cannot fit into the noiseless mock.
            if inject_spec:
                if inject_spec.get("lls_truth_boost") is not None:
                    truth_pack = apply_lls_truth_boost(
                        truth_pack, float(inject_spec["lls_truth_boost"]))
                if inject_spec.get("subdla_truth_boost") is not None:
                    truth_pack = apply_subdla_truth_boost(
                        truth_pack, float(inject_spec["subdla_truth_boost"]))
                mock_legs, info = make_leg_a_legmock(
                    ctx, fid_core, truth_pack, k_mock,
                    inject_metal_misspec=inject_spec.get("metal_misspec"),
                    inject_resolution=inject_spec.get("resolution"))
            else:
                mock_legs, info = make_leg_a_legmock(ctx, fid_core, truth_pack, k_mock)
            core_per_leg = fid_core
            sim = "leg_a_prior"
        else:
            sim = sims[m % len(sims)]
            # the mock TRUTH is built at the SAME resolution as the forward: if ctx.mf is set, the
            # gate invariant applies the MF correction to BOTH (it cancels in the closure ΔP).
            truth_sim = make_truth_from_sim(d, sim, fold=fold, mf=ctx.mf)
            k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, m), 2)
            mock_legs, truth_pack, info = make_legb_mock(ctx, truth_sim, k_mock)
            core_per_leg = _mock_core_per_leg(ctx, truth_sim)

        base_seed = int(jax.random.randint(k_nuts, (), 0, 2**31 - 1))
        ta_sched = (0.9,) + tuple(DIVERGENCE_RETRY_TARGET_ACCEPT)
        samples = n_div = None
        for attempt, ta in enumerate(ta_sched):
            samples, n_div = _run_nuts_legb(
                ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
                seed=base_seed + attempt, target_accept=ta, dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
            if n_div == 0:
                break
            if verbose:
                print(f"  [mock {m}] sim={sim[:20]}… {n_div} divergence(s) at ta={ta}"
                      + (" -> retry" if attempt < len(ta_sched) - 1 else ""))
        n_div_total += n_div
        if n_div > 0:
            n_divergent += 1

        kept_global = truth_pack["kept_global_z"]
        draws = _draws_matrix(samples, kept_global)             # (Lraw, P)
        draws_t, step, ess_min = thin_to_ess(draws)
        L = draws_t.shape[0]

        # truth vector in the packed order (kept global z only).
        truth_vec = np.concatenate([
            truth_pack["theta9"],
            truth_pack["tau0_global"][kept_global],
            truth_pack["alpha_hcd"]])
        if getattr(ctx, "hcd_2d_tilt", False):               # align with A_hcd/B_hcd/r_subdla/r_dla
            truth_vec = np.concatenate([truth_vec, _hcd_latent_truths_2d(truth_pack["alpha_hcd"], ctx)])
        elif getattr(ctx, "hierarchical_hcd", False):        # align with the appended A_hcd/r columns
            truth_vec = np.concatenate([truth_vec, _hcd_latent_truths(truth_pack["alpha_hcd"])])
        # Align with _draws_matrix's LAST a_SiIII col — ONLY when the SCALAR a_SiIII site exists
        # (uniform/flatlog). Under MODEL C+ ("flatlog2node") there is NO scalar a_SiIII draw column
        # (the per-leg f-/k-nodes are not packed), so appending a_siiii here would MISALIGN truth_vec
        # vs draws. Keying off "a_SiIII" in samples mirrors _draws_matrix exactly.
        if getattr(ctx, "sample_metals", False) and "a_SiIII" in samples:
            truth_vec = np.concatenate([truth_vec, [float(truth_pack.get("a_siiii", 0.0))]])

        # loglik of the truth + draws on the SAME mock data (Modrak rank).
        # Z-RESOLVED-ALPHA FIX (2026-06-19): use the z-RESOLVED truth alpha (truth_pack
        # ['alpha_hcd_z'], (nZg,3)) — NOT the z-FLAT pivot truth_pack['alpha_hcd'] (3,). The mock
        # truth-on-leg is z-resolved (per-z sim P1D / per-z prior draw), so re-scoring the truth
        # loglik with a z-flat alpha mismatched the data and shifted ll_true → a spurious
        # loglik-rank. Both the leg-A self-draw (draw_leg_a_leg_truth) and the held-out path
        # (make_legb_mock) now carry alpha_hcd_z. require_zresolved=True fails loudly on a regression.
        _a_si_true = float(truth_pack.get("a_siiii", 0.0))
        _a_si2_true = float(truth_pack.get("a_siii", 0.0))   # SiII doublet truth (Task A1; absent ⇒ 0)
        ll_true = float(_data_loglik_legcore(
            ctx, jnp.asarray(truth_pack["theta9"]),
            jnp.asarray(truth_pack["tau0_global"]),
            jnp.asarray(truth_pack["alpha_hcd_z"]), mock_legs, core_per_leg,
            a_siiii=_a_si_true, a_siii=_a_si2_true, require_zresolved=True))
        ll_draws = _loglik_of_draws(ctx, mock_legs, core_per_leg, samples, kept_global)
        # thin ll_draws by the SAME step.
        ll_draws_t = ll_draws[::step][:L]

        # MEAN-FLUX SITES (feedback-report-tau0-dtau0-bias): the packed draws/truth carry only the
        # DETERMINISTIC tau0_z ladder, not the 2 sampled sites (tau0_amp, dtau0). Store them SEPARATELY
        # (thinned by the SAME step + the truth) so the gate can report the τ₀ amplitude/slope bias
        # JOINTLY with n_s/A_p — mean flux is the suspected n_s channel. SEPARATE field ⇒ the packed
        # draws-matrix tail layout (and the uniform/flatlog golden) is untouched.
        sites_extra = {}
        for nm in ("tau0_amp", "dtau0"):
            if nm in samples:
                dr = np.asarray(samples[nm])[::step][:L]
                sites_extra[nm] = dict(draws=dr, truth=float(truth_pack.get(nm, np.nan)))
        # MODEL C+ metal f/k node sites (ceiling-check instrumentation): sampled by _metal_2node_sites
        # but NOT packed into _draws_matrix, so store them here (SAME thinning) with the injected-arm
        # truth. Additive-only + empty under uniform/flatlog/metals-off ⇒ byte-identical golden.
        sites_extra.update(_metal_node_sites_extra(samples, step, L, inject_spec, ctx, leg_a))
        # OPTION-B f_res sites (rail/coverage instrumentation, step-review #4): same thinning; injected
        # truth (b*, 0). EMPTY + additive-only unless sample_res sampled f_res ⇒ golden-safe.
        sites_extra.update(_resolution_sites_extra(samples, step, L, inject_spec, leg_a))

        per_mock.append(dict(sim=sim, truth_vec=truth_vec, draws=draws_t, L=L,
                             ll_true=ll_true, ll_draws=ll_draws_t,
                             names=_packed_names_for(samples, kept_global),
                             kept_global=kept_global, dropped=info["dropped"],
                             sites_extra=sites_extra, n_div=n_div))
        if verbose:
            print(f"  [mock {m}] sim={sim[:24]}… L={L} (step {step}, ess {ess_min:.0f}) "
                  f"nKeptZ={int(kept_global.sum())} div={n_div}")

    if return_per_mock:
        return per_mock
    return _aggregate_legb(per_mock, q_levels=q_levels)


def _loglik_of_draws(ctx, mock_legs, core_per_leg, samples, kept_global):
    """log_lik(draw, mock) for each raw draw — the Modrak loglik rank (uses the FULL τ₀ vec,
    not just kept z; the dropped z carry no data so they don't affect the likelihood).

    Z-RESOLVED-ALPHA FIX (2026-06-19): this used to REBUILD a z-FLAT (L,3) pivot alpha from
    samples['alpha_lls'/'alpha_subdla'/'alpha_dla'] and pass it to _data_loglik_legcore, which
    silently broadcasts it to every z — but the DEPLOYED forward (_legb_model) and the mock TRUTH
    are z-RESOLVED (per-z w_c rises ~3.5× over z), so the z-flat re-scoring produced a SPURIOUS
    z-structured residual that contaminated the reported SBC loglik-rank gate. We now use the
    z-resolved ``samples['alpha_hcd_z']`` (L,nZg,3) deterministic — EXACTLY as
    scripts/run_real_fit.py::_loglik_chain — and pass require_zresolved=True so a regression to a
    z-flat alpha here fails LOUDLY (the deployed mean / param-rank SBC were always clean; only this
    loglik-rank path was contaminated)."""
    theta = jnp.asarray(np.asarray(samples["theta_unit"]))      # (L,9)
    tau0 = jnp.asarray(np.asarray(samples["tau0_vec"]))         # (L,nZg)
    a_z = jnp.asarray(np.asarray(samples["alpha_hcd_z"]))       # (L,nZg,3) Z-RESOLVED (the fix)
    a_si = (jnp.asarray(np.asarray(samples["a_SiIII"])) if "a_SiIII" in samples
            else jnp.zeros(theta.shape[0]))
    # SiII DOUBLET amplitude (Stage C / Task A1): thread a_SiII into the re-score loglik too. The
    # de-double-count cell FLOATS a_SiII, so the SBC loglik-rank (Modrak) must score it; omitting it
    # (the pre-existing gap) ignored the sampled doublet nuisance → a wrong loglik rank. Absent key ⇒
    # zeros ⇒ a_SiII=0 ⇒ byte-exact back-compat with the prior (a_SiIII-only) behaviour.
    a_si2 = (jnp.asarray(np.asarray(samples["a_SiII"])) if "a_SiII" in samples
             else jnp.zeros(theta.shape[0]))

    def one(th, t0, al, asi, asi2):
        return _data_loglik_legcore(ctx, th, t0, al, mock_legs, core_per_leg,
                                    a_siiii=asi, a_siii=asi2, require_zresolved=True)
    return np.asarray(jax.vmap(one)(theta, tau0, a_z, a_si, a_si2))


def _aggregate_legb(per_mock, *, q_levels):
    """Per-param empirical coverage at each q level + the rank ECDF (the Leg-B headline)."""
    if not per_mock:
        return dict(n_mocks=0, names=[], coverage={}, L=0, n_divergent=0)
    # the kept-z set can differ per mock (different sims) → coverage is computed per PARAM
    # NAME using the cosmo+α params (always present) + the τ₀ params for mocks that kept them.
    # For the smoke we report cosmo+α coverage (always defined) and the loglik rank ECDF.
    n_theta = 9
    n_alpha = 3
    # per mock: rank each param's truth among its thinned draws.
    cosmo_alpha_idx = list(range(n_theta))  # θ9
    names = list(PARAM_NAMES) + ["alpha_lls", "alpha_subdla", "alpha_dla"]
    L_list = [rec["L"] for rec in per_mock]
    L_eff = int(min(L_list))

    # COLUMN INDEXING (must-fix #5 — the pre-existing positional bug). The α used to be indexed by
    # NEGATIVE position (-3,-2,-1) into BOTH ``draws`` and ``truth_vec``, assuming "α are the last
    # 3". But ``_draws_matrix`` APPENDS a_SiIII (and, hierarchical, A_hcd/r_subdla/r_dla) AFTER the
    # α columns, while ``truth_vec`` does NOT → the negative index lands on [subdla, dla, a_SiIII]
    # on the metals path (mis-aligned). BOTH layouts share the SAME positive α block
    # [θ9, τ₀(nKeptZ), α_lls, α_sub, α_dla, ...] so index α by its NAME-derived POSITIVE position
    # (θ9 → j; α → 9 + nKeptZ + (j−9)); the per-mock nKeptZ comes from rec["kept_global"].
    def _col_of(rec, j):
        if j < n_theta:
            return j
        n_kept = int(np.asarray(rec["kept_global"]).sum())
        return n_theta + n_kept + (j - n_theta)     # the α block start + the class offset (0,1,2)

    # coverage at each q for the cosmo+α params (the always-present block).
    coverage = {}
    for q in q_levels:
        cov_q = {}
        for j, nm in enumerate(names):
            truths, ints = [], []
            for rec in per_mock:
                col = _col_of(rec, j)
                tv = rec["truth_vec"]
                draws_col = rec["draws"][:, col]
                lo, hi = central_interval(draws_col, q)
                truths.append(tv[col]); ints.append((lo, hi))
            cov = empirical_coverage(np.array(truths), np.array(ints))
            cov_q[nm] = cov
        coverage[q] = cov_q

    # per-param normalized bias z=(truth−mean)/std over mocks (Leg-B headline w/ coverage;
    # well-calibrated+unbiased → mean≈0). The mean over N mocks has s.e. ≈ std/√N.
    bias = {}
    for j, nm in enumerate(names):
        zs = []
        for rec in per_mock:
            col = _col_of(rec, j)
            dc = rec["draws"][:, col]; sd = float(dc.std())
            if sd > 0:
                zs.append((float(rec["truth_vec"][col]) - float(dc.mean())) / sd)
        zs = np.array(zs)
        bias[nm] = dict(mean=float(zs.mean()) if zs.size else np.nan,
                        se=float(zs.std() / np.sqrt(zs.size)) if zs.size else np.nan,
                        n=int(zs.size))

    # loglik-rank ECDF — DIAGNOSTIC ONLY for Leg-B. Leg-B's null is NOT rank-uniform (the truth
    # is a held-out SIM, not an emulator draw), so this ECDF must NEVER gate the Leg-B verdict
    # (meta-review action #3). The Leg-B headline is coverage ≥ nominal + bias≈0 ONLY. The ECDF
    # uniformity gate is reserved for Leg-A (closure_sbc), where the null IS rank-uniform.
    ll_ranks = []
    for rec in per_mock:
        sub = np.linspace(0, rec["ll_draws"].size - 1, L_eff).round().astype(int)
        ll_ranks.append(loglik_rank(rec["ll_true"], rec["ll_draws"][sub]))
    ll_ranks = np.array(ll_ranks, int)

    gate_valid = bool(L_eff >= L_FLOOR)            # L-floor for the ECDF *band sizing* only
    ll_ecdf_in_band = ecdf_ll = None
    if ll_ranks.size >= 2 and L_eff >= 2:
        lower, upper, ecdf, ok, grid = ecdf_pit_bands(ll_ranks, L_eff, prob=0.95)
        ll_ecdf_in_band = bool(ok); ecdf_ll = (lower, upper, ecdf, grid)

    n_divergent = sum(1 for rec in per_mock if rec["n_div"] > 0)
    return dict(n_mocks=len(per_mock), names=names, coverage=coverage, bias=bias, L=L_eff,
                gate_valid=gate_valid, n_divergent=n_divergent,
                ll_ranks=ll_ranks, ll_ecdf_in_band=ll_ecdf_in_band,
                ll_ecdf_diagnostic_only=True, ecdf_ll=ecdf_ll,
                per_mock=per_mock, q_levels=q_levels)


# ============================================================================ #
#  CLI / smoke.
# ============================================================================ #
def _smoke(args):
    print(f"[legb-smoke] building ctx from {CKPT} (+ xclass C_emu)")
    # SMOKE: optionally narrow the z-range (fewer τ₀ params + a cheaper per-step likelihood)
    # so the FULL real-grid path runs in minutes. The production run uses the full z-range.
    desi_kw = dict(z_lo=args.z_lo, z_hi=args.z_hi) if args.z_lo or args.z_hi < 4.2 else None
    ks_kw = None
    legs_arg = dict(desi_kwargs=desi_kw, ks_kwargs=ks_kw)
    if args.desi_only:
        # build with both, then drop KS (the loader path is the same; just slice the list).
        pass
    ctx, d = build_legb_ctx(cemu_inflate=args.cemu_inflate, use_xclass=not args.diag_cemu,
                            with_mf=args.mf, mf_with_floor=not args.no_floor, **legs_arg)
    if args.mf:
        print(f"[legb-smoke] MF forward ON (production correction, fold 0); "
              f"floor={'ON (LF→HR + ns-edge)' if ctx.mf_floor is not None else 'OFF'}")
    if args.desi_only:
        ctx = ctx._replace(legs=[leg for leg in ctx.legs if leg.name == "DESI"])
    print(f"[legb-smoke] legs: " + ", ".join(
        f"{leg.name}(n_z={leg.n_z}, N={leg.k.shape[0]})" for leg in ctx.legs))
    sims, _ = held_out_sims(d, fold=0)
    print(f"[legb-smoke] {len(sims)} held-out sims (fold 0); C_emu="
          f"{'cross-class' if ctx.rho_zb_per_leg is not None else 'diagonal'}")

    res = run_legb(ctx, d, n_mocks=args.n_mocks, n_warmup=args.n_warmup,
                   n_samples=args.n_samples, seed=args.seed,
                   dense_mass=not args.diag_mass, max_tree_depth=args.max_tree_depth)
    print("\n========== Leg-B smoke summary ==========")
    print(f"mocks={res['n_mocks']}  divergent={res['n_divergent']}  "
          f"L(thinned)={res['L']}  gate_valid(L≥{L_FLOOR})={res['gate_valid']}")
    for q in res["q_levels"]:
        print(f"\nper-param empirical coverage @ {int(q*100)}% CR (target ≥{q:.2f}):")
        for nm in res["names"]:
            cov = res["coverage"][q][nm]
            print(f"  {nm:14s}: {cov['coverage']:.2f}  "
                  f"[{cov['ci_low']:.2f},{cov['ci_high']:.2f}]  (k={cov['k']}/{cov['n']})")
    print("\nper-param normalized bias (truth−mean)/std, mean over mocks (≈0 = unbiased):")
    for nm in res["names"]:
        b = res["bias"][nm]
        print(f"  {nm:14s}: {b['mean']:+.2f} ± {b['se']:.2f}σ  (n={b['n']})")
    if res["ll_ecdf_in_band"] is not None:
        print(f"\nloglik-rank ECDF: {'in-band' if res['ll_ecdf_in_band'] else 'OUT-of-band'} "
              f"— DIAGNOSTIC ONLY (NOT a Leg-B gate; Leg-B's null is not rank-uniform).")
    print("\n[verdict basis] Leg-B = coverage ≥ nominal + bias≈0 ONLY (NOT the rank ECDF).")
    print("[caveat] smoke N is tiny → coverage is a PATH check, not a calibration verdict.")
    return ctx, d, res


def _smoke_convergence(args):
    """STEP-A convergence-MODE smoke: ONE mock, ≥2 dispersed-init chains, the full battery."""
    print(f"[legb-conv] building ctx from {CKPT} (+ xclass C_emu)")
    desi_kw = dict(z_lo=args.z_lo, z_hi=args.z_hi) if args.z_lo or args.z_hi < 4.2 else None
    ctx, d = build_legb_ctx(cemu_inflate=args.cemu_inflate, use_xclass=not args.diag_cemu,
                            with_mf=args.mf, mf_with_floor=not args.no_floor,
                            desi_kwargs=desi_kw)
    if args.desi_only:
        ctx = ctx._replace(legs=[leg for leg in ctx.legs if leg.name == "DESI"])
    print(f"[legb-conv] legs: " + ", ".join(
        f"{leg.name}(n_z={leg.n_z}, N={leg.k.shape[0]})" for leg in ctx.legs))
    print(f"[legb-conv] {args.chains} chains × {args.n_samples} samples (warmup {args.n_warmup}) "
          f"dispersed init_to_sample, dense_mass={not args.diag_mass}, mtd={args.max_tree_depth}")

    res = run_legb_convergence(
        ctx, d, mock_index=args.mock_index, n_chains=args.chains, n_warmup=args.n_warmup,
        n_samples=args.n_samples, seed=args.seed, dense_mass=not args.diag_mass,
        max_tree_depth=args.max_tree_depth)
    b = res["battery"]
    print("\n========== STEP-A convergence battery ==========")
    print(f"sim={res['sim'][:40]}  chains={b['n_chains']}  draws/chain={b['n_draws']}  "
          f"divergent={b['n_divergent']}")
    print(f"rank-split-R-hat max = {b['rhat_max']:.4f}  (gate <1.01)")
    print(f"bulk-ESS min = {b['ess_bulk_min']:.0f}   tail-ESS min = {b['ess_tail_min']:.0f}  "
          f"(gate ≥400)")
    print(f"E-BFMI min = {b['ebfmi_min']:.3f} (gate >0.3)   "
          f"tree-depth saturation = {b['treedepth_sat_frac']:.3f} (gate <~0.02)")
    print(f"\n  {'param':12s} {'R-hat':>7s} {'ESSbulk':>8s} {'ESStail':>8s} "
          f"{'init/postsd':>11s}")
    for i, nm in enumerate(res["names"]):
        print(f"  {nm:12s} {b['rhat'][nm]:7.3f} {b['ess_bulk'][nm]:8.0f} "
              f"{b['ess_tail'][nm]:8.0f} {b['init_spread_over_postsd'][i]:11.2f}")
    print(f"\nper-chain divergences: {res['per_chain_div']}")
    print("[caveat] smoke depth → R-hat is finite/computed, NOT necessarily <1.01.")
    return ctx, d, res


def main():
    ap = argparse.ArgumentParser(description="Phase-C T4b Leg-B closure on the real grids")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end Leg-B path (N≈4–8) + per-param coverage")
    ap.add_argument("--n-mocks", type=int, default=6)
    ap.add_argument("--n-warmup", type=int, default=80)
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cemu-inflate", type=float, default=1.0)
    ap.add_argument("--diag-cemu", action="store_true",
                    help="use the DIAGONAL C_emu instead of the cross-class (default)")
    ap.add_argument("--diag-mass", action="store_true",
                    help="SMOKE: use a diagonal NUTS mass (faster warmup than dense)")
    ap.add_argument("--max-tree-depth", type=int, default=7,
                    help="SMOKE: cap the NUTS tree depth (bounds leapfrogs/step; default 7)")
    ap.add_argument("--z-lo", type=float, default=0.0,
                    help="SMOKE: narrow the DESI z-range lower edge (fewer τ₀ params)")
    ap.add_argument("--z-hi", type=float, default=4.2,
                    help="SMOKE: narrow the DESI z-range upper edge")
    ap.add_argument("--desi-only", action="store_true",
                    help="SMOKE: DESI leg only (skip KS) for the fastest path proof")
    ap.add_argument("--mf", action="store_true",
                    help="route the Leg-B forward + mock truth through the MF correction "
                         "(production through-MF path; the gate invariant)")
    ap.add_argument("--no-floor", action="store_true",
                    help="with --mf, DISABLE the MF C_emu floor (default ON)")
    ap.add_argument("--figures", action="store_true", help="emit the 3 diagnostic figures")
    ap.add_argument("--convergence", action="store_true",
                    help="STEP-A convergence MODE: one mock, ≥2 dispersed-init chains, "
                         "rank-R-hat / bulk+tail-ESS / E-BFMI battery (the multichain path)")
    ap.add_argument("--chains", type=int, default=4,
                    help="convergence mode: number of dispersed-init chains (default 4)")
    ap.add_argument("--mock-index", type=int, default=0,
                    help="convergence mode: which held-out mock (cycles the fold's sims)")
    args = ap.parse_args()
    if args.convergence:
        _smoke_convergence(args)
    elif args.smoke:
        ctx, d, res = _smoke(args)
        if args.figures:
            from . import closure_legb_figs as F  # lazy (matplotlib)
            F.make_all(ctx, d, res)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
