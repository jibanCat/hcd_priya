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
import functools
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
from .data import load_cache, make_splits, KIM_AMP, KIM_SLOPE, Z_LIMITS
from .meanflux_prior import meanflux_tau0_prior, becker13_tau0
from .inference import (PARAM_NAMES, hcd_incidence_prior,
                        HCD_LIT_OVER_SIM_SLOPE, HCD_Z_PIVOT)
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
    marginalize_zslope: bool = False     # STEP-A M3: sample the HCD per-class z-slope s_c
                                         # (real-fit config) instead of the fixed power-law.
    zslope_mu: object = None             # (3,) prior center on s_c (default HCD_LIT_OVER_SIM_SLOPE)
    zslope_sigma: object = None          # (3,) prior width on s_c (default the literature WLS σ_s)


def _kim(z):
    return KIM_AMP * (1.0 + jnp.asarray(z)) ** KIM_SLOPE


# ============================================================================ #
#  Build the production Leg-B context from final_fold0 + the xclass error vector.
# ============================================================================ #
def build_legb_ctx(*, ckpt=CKPT, error_vector=ERROR_VECTOR,
                   xclass_error_vector=XCLASS_ERROR_VECTOR, cemu_inflate=1.0,
                   metals_on=False, desi_kwargs=None, ks_kwargs=None,
                   use_xclass=True, with_mf=False, mf_fold=0, mf_with_floor=True):
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

    desi = DL.load_desi_leg(metals_on=metals_on, **(desi_kwargs or {}))
    ks = DL.load_ks_leg(**(ks_kwargs or {}))
    legs = [desi, ks]

    z_global = np.unique(np.round(np.concatenate([desi.z, ks.z]), 6))

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
    # HCD incidence prior from the cache's structural w_c at z_pivot (LLS,subDLA,DLA).
    w_c_med = np.median(d["w_c_cache"][:, 1:], axis=0)     # (3,) structural weights
    alpha_mu, alpha_sd = hcd_incidence_prior(jnp.asarray(w_c_med), z=3.0)

    mf_obj = mf_floor_obj = None
    if with_mf:
        mf_obj, mf_floor_obj = build_mf_correction(
            fold=mf_fold, with_floor=mf_with_floor)

    ctx = LegBCtx(
        model=model, pf_stats=pf, dla_core_leg=dla_core_leg, legs=legs, cache_k=cache_k,
        z_global=z_global, sigma_zb_per_leg=sigma_zb_per_leg,
        rho_zb_per_leg=(rho_zb_per_leg if rho is not None else None),
        alpha_centres=alpha_centres, tau0_mu=tau0_mu, tau0_sigma=tau0_sigma,
        alpha_hcd_mu=jnp.asarray(alpha_mu), alpha_hcd_sigma=jnp.asarray(alpha_sd),
        cemu_inflate=float(cemu_inflate), mf=mf_obj, mf_floor=mf_floor_obj)
    return ctx, d


def build_mf_correction(fold=0, *, rank1=True, exclude_held_hr=False,
                        with_floor=True, floor_npz=None):
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
    floor) from ``mf_cemu_floor.npz``."""
    from hcd_analysis.emulator import multifidelity as MF
    lf_cache = MF.load_cache(MF.LF_CACHE)
    hr_cache = MF.load_cache(MF.HR_CACHE)
    pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
    fold_model, _fold_meta, fold_norm, lf_logk = MF.load_lf_backbone(fold)
    eval_logk = np.asarray(lf_logk)

    # HF-LOSO row mask (optional): exclude the fold's held-out HR sims from the head fit.
    train_rows = None
    if exclude_held_hr:
        from .data import load_cache as _lc
        d = _lc(CACHE_PATH)
        sims, _ = held_out_sims(d, fold=fold)
        hr_sims = set(s.decode() if isinstance(s, bytes) else s
                      for s in hr_cache["sim_name"])
        held = set(s for s in sims if s in hr_sims)
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
                                eval_logk=eval_logk, log_rho=log_rho, delta_mode="none")
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


def make_truth_from_sim(d, sim_name, fold=0, tau0_anchor="becker13", mf=None):
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
      P_obs_true (nZs, K)       per-z contaminated P1D on the cache grid;
      params_unit (9,)          the truth θ (unit-cube; same for all the sim's rows);
      tau0       (nZs,)         the truth τ₀(z) (cache tau0 = −ln⟨F⟩, the selected ladder row);
      dla_core   (nZs, K)       the per-z DLA core (cache delta[,2]) of the selected row;
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
        if tau0_anchor == "becker13":
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
    dla_core = np.zeros((len(keep_rows), K))
    th_truth = jnp.asarray(params_unit[keep_rows[0]])
    z_unit_rows = (z_grid[keep_rows] - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    for i, r in enumerate(keep_rows):
        a = w_c[r, 1:]                               # (3,) [LLS,sub,DLA] the sim's contamination
        coef = np.concatenate([[1.0 - a.sum()], a])  # (4,)
        core_r = delta[r, 2]                         # (K,) DLA core add-back
        if mf is None:
            corr = np.ones((4, K))                    # LF-resolution truth (no correction)
        else:
            # GATE INVARIANT: the SAME fixed per-class MF factor exp(g + log res_corr) the
            # forward applies, at this row's (θ, z, τ₀). The DLA core add-back is UNCORRECTED
            # (added AFTER the per-class P_filt correction, mirroring _excess_from_P_filt /
            # _predict_P_obs_mf), so the truth is reproduced when θ→truth in the MF forward.
            corr = np.asarray(jnp.exp(DL._mf_corr_on_cache(
                mf, th_truth, jnp.asarray(float(z_unit_rows[i])),
                jnp.asarray(float(tau0_all[r])))))    # (4, K)
        P_cls = np.stack([P_filt[r, 0] * corr[0], P_filt[r, 1] * corr[1],
                          P_filt[r, 2] * corr[2], P_filt[r, 3] * corr[3] + core_r])
        P_obs[i] = np.einsum("c,ck->k", coef, P_cls)
        dla_core[i] = core_r
    return dict(
        z=z, P_obs_true=P_obs, params_unit=params_unit[keep_rows[0]],
        tau0=tau0_all[keep_rows], dla_core=dla_core,
        w_c=np.median(w_c[keep_rows, 1:], axis=0), rows=keep_rows)


def _chol_jitter(C, jitter=1e-10):
    """Lower Cholesky of the jittered C_data (matches gaussian_loglik's SPD floor)."""
    K = C.shape[-1]
    Cj = C + (jitter * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    return jnp.linalg.cholesky(Cj)


def make_legb_mock(ctx: LegBCtx, truth_sim, key):
    """Build a Leg-B mock from a sim-truth: interpolate the sim-truth P1D onto each leg's k,
    draw ε ~ N(0, C_data) (cosmic-ONLY) per leg, ``mock = truth_on_leg + ε``.

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
      info       — dict(key, per-leg dropped-z, the sim z used).
    """
    z_sim = np.asarray(truth_sim["z"])
    P_sim = np.asarray(truth_sim["P_obs_true"])     # (nZs, K)
    tau0_sim = np.asarray(truth_sim["tau0"])
    cache_k = np.asarray(ctx.cache_k)
    z_tol = 0.15                                    # nearest-z map tolerance (cache Δz=0.2)

    keys = jax.random.split(key, len(ctx.legs))
    mock_legs = []
    dropped = {}
    for li, leg in enumerate(ctx.legs):
        N = leg.k.shape[0]
        P_mock = np.array(leg.P_data, float).copy()
        keep_row = np.zeros(N, bool)
        drop_z = []
        # build the clean truth-on-leg per z, then add cosmic noise over the kept rows.
        P_truth_on_leg = np.zeros(N)
        for iz in range(leg.n_z):
            zz = float(leg.z[iz])
            j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) > z_tol:
                drop_z.append(zz)
                continue
            rows = np.where(np.asarray(leg.z_idx) == iz)[0]
            k_sub = np.asarray(leg.k)[rows]
            P_truth_on_leg[rows] = np.asarray(
                jnp.interp(jnp.asarray(k_sub), jnp.asarray(cache_k), jnp.asarray(P_sim[j])))
            keep_row[rows] = True
        dropped[leg.name] = drop_z

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
    truth_pack = dict(
        theta9=np.asarray(truth_sim["params_unit"]),
        tau0_global=tau0_global, alpha_hcd=np.asarray(truth_sim["w_c"]),
        kept_global_z=kept_global)
    info = dict(key=key, dropped=dropped, z_sim=z_sim)
    return mock_legs, truth_pack, info


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
                         dla_core_per_leg, *, return_parts=False):
    """``data_loglik`` but with a PER-LEG-Z dla_core (the mock's sim core). ``data_loglik``
    takes ONE (K,) core; here each leg z uses its own, so we call ``predict_P_obs_on_leg``
    per leg with that leg's core threaded through a per-z loop is overkill — instead we note
    the core is z-binned inside the binding via ``leg.z_idx`` and the SAME core value is used
    for every row of a z. ``data_loglik`` already loops z internally with a single core; to
    keep the core matched (forward core == truth core, so it cancels in the emu-error sizing
    and the truth P_obs is reproduced when θ→truth), we pass each leg its OWN (K,) core that
    is the z-MEAN of that leg's per-z cores (a documented MVP — the per-z core variation is
    tiny vs the P1D, and the DLA sector is un-certified by Leg B without arm A3 anyway)."""
    total = 0.0
    parts = {}
    from .likelihood import gaussian_loglik
    zg = np.asarray(ctx.z_global)
    for leg in mock_legs:
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        tau0_vec = tau0_global[jnp.asarray(sel)]
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
        P_model, C_total = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_vec, alpha_leg, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor)
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


def _legb_model(ctx: LegBCtx, mock_legs, dla_core_per_leg):
    """The numpyro model: ``sampler_numpyro``'s priors (θ~Uniform^9 + auto-bijector; τ₀ in
    the α-ladder coord on the GLOBAL z grid; α_lls/subdla~Normal, α_dla~softplus(Normal)) with
    the single ``factor`` = the per-leg-core multi-leg real-cov loglik (``_data_loglik_legcore``
    over ``mock_legs``)."""
    zg = jnp.asarray(ctx.z_global)
    theta9 = numpyro.sample("theta_unit", dist.Uniform(jnp.zeros(9), jnp.ones(9)).to_event(1))
    kim = _kim(zg)
    alpha_ladder = numpyro.sample(
        "alpha_ladder", dist.Normal(ctx.tau0_mu / kim, ctx.tau0_sigma / kim).to_event(1))
    tau0_global = numpyro.deterministic("tau0_vec", alpha_ladder * kim)
    a_lls = numpyro.sample("alpha_lls",
                           dist.Normal(ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]))
    a_sub = numpyro.sample("alpha_subdla",
                           dist.Normal(ctx.alpha_hcd_mu[1], ctx.alpha_hcd_sigma[1]))
    a_dla_raw = numpyro.sample("alpha_dla_raw",
                               dist.Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0))
    a_dla = numpyro.deterministic("alpha_dla", jax.nn.softplus(a_dla_raw))
    alpha_pivot = jnp.stack([a_lls, a_sub, a_dla])              # (3,) pivot-z (z=3) amplitudes
    # z-RESOLVED incidence α_c(z) = α_pivot · ((1+z)/(1+z_p))^s_c (the dN/dX slope) — the fix:
    # the forward must track the mock's per-z w_c(z) (rises ~3.5× over z), not a z-constant α.
    # STEP-A M3: when ctx.marginalize_zslope, s_c is SAMPLED (the real-fit config) instead of
    # the fixed HCD_LIT_OVER_SIM_SLOPE; otherwise the fixed literature power-law slope is used.
    s_c = _zslope_sites(ctx)
    shape_zg = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c
    alpha_hcd = numpyro.deterministic("alpha_hcd_z", alpha_pivot[None, :] * shape_zg)  # (n_zg,3)
    numpyro.factor("loglik", _data_loglik_legcore(
        ctx, theta9, tau0_global, alpha_hcd, mock_legs, dla_core_per_leg))


def _zslope_sites(ctx):
    """The HCD per-class z-slope s_c (3,). FIXED to HCD_LIT_OVER_SIM_SLOPE unless
    ``ctx.marginalize_zslope`` (STEP-A M3) — then SAMPLE s_lls/s_subdla/s_dla ~ Normal at the
    ctx prior (default: center HCD_LIT_OVER_SIM_SLOPE, width ZSLOPE_PRIOR_SIGMA). Shared by
    ``_legb_model`` (with the factor) and ``_legb_priors_only`` (transform-only postprocess)."""
    if not getattr(ctx, "marginalize_zslope", False):
        return jnp.asarray(HCD_LIT_OVER_SIM_SLOPE)
    mu = (jnp.asarray(HCD_LIT_OVER_SIM_SLOPE) if ctx.zslope_mu is None
          else jnp.asarray(ctx.zslope_mu))
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
    zg = jnp.asarray(ctx.z_global)
    numpyro.sample("theta_unit", dist.Uniform(jnp.zeros(9), jnp.ones(9)).to_event(1))
    kim = _kim(zg)
    numpyro.sample("alpha_ladder",
                   dist.Normal(ctx.tau0_mu / kim, ctx.tau0_sigma / kim).to_event(1))
    numpyro.sample("alpha_lls", dist.Normal(ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]))
    numpyro.sample("alpha_subdla", dist.Normal(ctx.alpha_hcd_mu[1], ctx.alpha_hcd_sigma[1]))
    numpyro.sample("alpha_dla_raw", dist.Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0))
    if getattr(ctx, "marginalize_zslope", False):
        mu = (jnp.asarray(HCD_LIT_OVER_SIM_SLOPE) if ctx.zslope_mu is None
              else jnp.asarray(ctx.zslope_mu))
        sg = (jnp.asarray(ZSLOPE_PRIOR_SIGMA) if ctx.zslope_sigma is None
              else jnp.asarray(ctx.zslope_sigma))
        numpyro.sample("s_lls", dist.Normal(mu[0], sg[0]))
        numpyro.sample("s_subdla", dist.Normal(mu[1], sg[1]))
        numpyro.sample("s_dla", dist.Normal(mu[2], sg[2]))


def _legb_reconstruct_deterministics(ctx, samples):
    """Reconstruct the three ``numpyro.deterministic`` sites of ``_legb_model``
    (``tau0_vec``, ``alpha_dla``, ``alpha_hcd_z``) host-side from the raw latent samples —
    so the cheap (priors-only) postprocess can SKIP the per-sample full-model deterministic
    replay (the 237.6 s/mock waste flagged in MF-SMOKE-01) yet return a samples dict
    BYTE-IDENTICAL to numpyro's default ``get_samples()``. Returns a NEW dict (the input plus
    the three deterministic keys)."""
    zg = jnp.asarray(ctx.z_global)
    kim = _kim(zg)
    alpha_ladder = jnp.asarray(samples["alpha_ladder"])              # (L, nZg)
    tau0_vec = alpha_ladder * kim
    alpha_dla = jax.nn.softplus(jnp.asarray(samples["alpha_dla_raw"]))  # (L,)
    alpha_pivot = jnp.stack([jnp.asarray(samples["alpha_lls"]),
                             jnp.asarray(samples["alpha_subdla"]),
                             alpha_dla], axis=-1)                     # (L, 3)
    if getattr(ctx, "marginalize_zslope", False) and "s_lls" in samples:
        s_c = jnp.stack([jnp.asarray(samples["s_lls"]),
                         jnp.asarray(samples["s_subdla"]),
                         jnp.asarray(samples["s_dla"])], axis=-1)     # (L, 3)
        # shape_zg per draw: ((1+z)/(1+z_p))^{s_c} → (L, nZg, 3)
        ratio = (1.0 + zg)[None, :, None] / (1.0 + HCD_Z_PIVOT)
        shape_zg = ratio ** s_c[:, None, :]                          # (L, nZg, 3)
        alpha_hcd_z = alpha_pivot[:, None, :] * shape_zg             # (L, nZg, 3)
    else:
        shape_zg = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** jnp.asarray(HCD_LIT_OVER_SIM_SLOPE)
        alpha_hcd_z = alpha_pivot[:, None, :] * shape_zg[None, :, :]      # (L, nZg, 3)
    out = dict(samples)
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
    n_theta, n_alpha = 9, 3
    names = list(PARAM_NAMES) + ["alpha_lls", "alpha_subdla", "alpha_dla"]
    # the packed-draw column order is θ9, τ₀(kept), α3 — name the τ₀ block by global-z index.
    tau0_names = [f"tau0_z{i}" for i in range(int(kept_global.sum()))]
    packed_names = list(PARAM_NAMES) + tau0_names + ["alpha_lls", "alpha_subdla", "alpha_dla"]

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
    return np.concatenate([theta, tau0, a_lls, a_sub, a_dla], axis=1)


def run_legb(ctx: LegBCtx, d, *, n_mocks, n_warmup, n_samples, seed,
             cemu_inflate=None, fold=0, q_levels=(0.68, 0.95), verbose=True,
             dense_mass=True, max_tree_depth=10, mock_indices=None,
             return_per_mock=False):
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
    if cemu_inflate is not None:
        ctx = ctx._replace(cemu_inflate=float(cemu_inflate))
    sims, _va = held_out_sims(d, fold=fold)
    key0 = jax.random.PRNGKey(int(seed))
    idxs = list(range(int(n_mocks))) if mock_indices is None else list(mock_indices)

    # cycle through the held-out sims (n_mocks may exceed the #sims → reuse with fresh noise).
    per_mock = []
    n_div_total = 0
    n_divergent = 0
    for m in idxs:
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

        # loglik of the truth + draws on the SAME mock data (Modrak rank).
        ll_true = float(_data_loglik_legcore(
            ctx, jnp.asarray(truth_pack["theta9"]),
            jnp.asarray(truth_pack["tau0_global"]),
            jnp.asarray(truth_pack["alpha_hcd"]), mock_legs, core_per_leg))
        ll_draws = _loglik_of_draws(ctx, mock_legs, core_per_leg, samples, kept_global)
        # thin ll_draws by the SAME step.
        ll_draws_t = ll_draws[::step][:L]

        per_mock.append(dict(sim=sim, truth_vec=truth_vec, draws=draws_t, L=L,
                             ll_true=ll_true, ll_draws=ll_draws_t,
                             kept_global=kept_global, dropped=info["dropped"],
                             n_div=n_div))
        if verbose:
            print(f"  [mock {m}] sim={sim[:24]}… L={L} (step {step}, ess {ess_min:.0f}) "
                  f"nKeptZ={int(kept_global.sum())} div={n_div}")

    if return_per_mock:
        return per_mock
    return _aggregate_legb(per_mock, q_levels=q_levels)


def _loglik_of_draws(ctx, mock_legs, core_per_leg, samples, kept_global):
    """log_lik(draw, mock) for each raw draw — the Modrak loglik rank (uses the FULL τ₀ vec,
    not just kept z; the dropped z carry no data so they don't affect the likelihood)."""
    theta = jnp.asarray(np.asarray(samples["theta_unit"]))      # (L,9)
    tau0 = jnp.asarray(np.asarray(samples["tau0_vec"]))         # (L,nZ)
    a = jnp.asarray(np.stack([np.asarray(samples["alpha_lls"]),
                              np.asarray(samples["alpha_subdla"]),
                              np.asarray(samples["alpha_dla"])], axis=1))  # (L,3)

    def one(th, t0, al):
        return _data_loglik_legcore(ctx, th, t0, al, mock_legs, core_per_leg)
    return np.asarray(jax.vmap(one)(theta, tau0, a))


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

    # coverage at each q for the cosmo+α params (the always-present block).
    coverage = {}
    for q in q_levels:
        cov_q = {}
        for j, nm in enumerate(names):
            # column index in the packed truth_vec: θ9 are 0..8; α are the LAST 3.
            if j < n_theta:
                col = j
            else:
                col = -(n_alpha - (j - n_theta))    # -3,-2,-1
            truths, ints = [], []
            for rec in per_mock:
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
        col = j if j < n_theta else -(n_alpha - (j - n_theta))
        zs = []
        for rec in per_mock:
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
