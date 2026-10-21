"""HISTORICAL FIXTURE (test-only): the pre-2026-10 single-grid forward, moved verbatim from
hcd_analysis/emulator/data_likelihood.py at gate E (emulator-debug campaign 2026-10; spec GATE_E_SPEC.md v1 and the
incident note 2026-10-05-INCIDENT-kgrid-representation-regression in the notes repository).

These functions place every emulator mode on ONE velocity grid ``cache_k`` for every (z, theta); that is the defect
of the incident. They are kept only as (a) the equivalence reference of the gate E forward (fed each z's own grid,
the old forward IS the correct forward) and (b) the subject of the historical-behaviour tests. Production code never
imports this module (gate E criterion E1c)."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.data import Z_LIMITS
from hcd_analysis.emulator.data_likelihood import (K_SiIII_DEFAULT, K_SiII_DEFAULT, Z_PIVOT, DataLeg,  # noqa: F401
                                                   _ns_phys_from_theta9, _resolution_factor, emu_var_modes,
                                                   load_desi_leg, load_ks_leg, metal_factor_at_z)
from hcd_analysis.emulator.likelihood import gaussian_loglik, rho_at_tau0, sigma_at_tau0  # noqa: F401
from hcd_analysis.emulator.predict import _excess_from_P_filt, predict_P_filt, predict_P_obs  # noqa: F401
from hcd_analysis.paths import REPO_ROOT_STR as _REPO


# The emulator cache k-grid Nyquist (angular k, s/km == 1/Å in the velocity-equiv
# convention the data + cache share). The emulator cannot predict above this.
CACHE_KMAX = 0.069


# ============================================================================ #
#  Multi-fidelity (LF→HR) opt-in forward (T5a)
#
#  PROMOTED from scripts/diag_emu_bias_allfolds_mf.py (the certified T3 adoption gate).
#  When a ``MultiFidelity`` is supplied, the per-class LF P_filt is routed through the
#  FIXED (θ-blind) resolution correction ``g(z,τ₀,k) + log res_corr(z,k)`` BEFORE the
#  clean+excess HCD combination and BEFORE the cache-k→leg-k interp — exactly where the
#  gate applied it, so the production forward == the gate's measured forward.
#
#  CONTRACT (mirror of the gate, faithful through-MF wiring):
#    * ``mf.eval_logk`` MUST equal log10(cache_k) (the LF native cache grid), so
#      ``mf.logP_mf - mf.lf_logP == g + log res_corr`` is evaluated EXACTLY on the cache
#      k-grid and applied per class to P_filt at the cache-k level (no LF tail
#      extrapolation is hit inside the analysis band k<0.069 == the cache k_max).
#    * the LF backbone in ``mf`` is FROZEN (stop_gradient on its weights inside
#      ``MultiFidelity.lf_logP`` / ``_freeze``); grads flow to (θ9, τ₀, α) but NEVER to
#      the LF weights — so the mf= forward is NUTS-safe and differentiable.
#    * C_total is taken from the UNCHANGED LF path (the C_emu floor is sized separately
#      through the MF forward — a follow-up; this T5a wiring touches P_obs ONLY).
#
#  dN/dX/CDDF (Head-A) correction: NOT applied inside this per-leg P1D binding. The gate
#  measured the P1D path only; ``alpha_hcd`` enters the production likelihood directly
#  (the incidence is constructed upstream), so the θ-blind dN/dX resolution factor
#  (``multifidelity.apply_dndx_res_corr``) is applied where the incidence prior / w_c is
#  built — NOT here. Wiring it into this binding would diverge from the certified gate
#  path. (The MultiFidelity object also carries the dN/dX/CDDF tables for that use.)
# ============================================================================ #
def _mf_corr_on_cache(mf, theta9, z_unit, tau0, alpha_res=None):
    """Per-class MF log-correction (4, Kc) on the cache grid: ``g + α(z)·log res_corr``.

    Mirror of ``diag_emu_bias_allfolds_mf.mf_corr_on_cache``. ``mf.eval_logk`` is the
    cache log-k grid, so ``g = mf.g(x, τ₀)`` (== log_rho + resolved FixedMeanHead) is
    the exact production MF correction on the cache k-grid, and ``log res_corr(z)`` is
    the fixed particle-convergence factor (broadcast over the 4 classes). θ-blind by
    construction (g reads cond[9]=z_unit, cond[10]=τ₀ only); differentiable in (θ9, τ₀).

    ``alpha_res`` (Task 1.3, FORWARD-ONLY marginalization of the res_corr amplitude):
    a ``(alpha0, s)`` tuple ⇒ scale ``log res_corr`` by the scalar (per-z)
    ``α(z) = alpha0·((1+z)/(1+Z_PIVOT))^s`` before adding. ``None`` (the DEFAULT) ⇒
    α≡1 ⇒ this returns ``g + log res_corr`` BIT-IDENTICALLY (the multiply is skipped, so
    the Task-1.2 MF golden is byte-exact). ``(1.0, 0.0)`` is the explicit no-op.
    """
    x = jnp.concatenate([jnp.asarray(theta9), jnp.atleast_1d(z_unit)])   # (10,)
    g = mf.g(x, tau0)                                                    # (4, Kc)
    z_phys = z_unit * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
    log_rc = jnp.log(mf.res_corr(z_phys))[None, :]                       # (1, Kc) bcast
    if alpha_res is None:
        return g + log_rc                                               # (4, Kc) byte-exact no-op
    alpha0, s = alpha_res
    alpha_z = alpha0 * ((1.0 + z_phys) / (1.0 + Z_PIVOT)) ** s          # scalar (per z)
    return g + alpha_z * log_rc                                         # (4, Kc) additive


def _predict_P_obs_mf(mf, model, theta9, z_unit, tau0, alpha_hcd, pf_stats, dla_core,
                      alpha_res=None):
    """P_obs (Kc,) through the MF forward — mirror of the gate's ``predict_P_obs_mf``.

    ``alpha_res`` (Task 1.3): threaded FORWARD-ONLY into ``_mf_corr_on_cache`` to scale the
    res_corr amplitude by ``α(z)``; ``None`` (default) ⇒ α≡1 ⇒ byte-exact back-compat.

    The per-class LF P_filt is multiplied by ``exp(g + log res_corr)`` (the fixed,
    θ-blind resolution factor), then the SAME clean+excess HCD combination as
    ``predict.predict_P_obs``. The LF backbone P_filt comes from the FROZEN
    ``mf.lf_model`` inside ``predict_P_filt`` here — but to keep the production path
    BIT-IDENTICAL to the gate (which calls ``predict_P_filt(model, ...)`` on the SAME
    backbone object the MF was built from), we use the passed ``model`` as the LF P_filt
    source; the caller passes the SAME frozen backbone object as both ``model`` and
    ``mf.lf_model``. Differentiable in (θ9, τ₀, α)."""
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)       # (4,Kc) LF
    corr = jnp.exp(_mf_corr_on_cache(mf, theta9, z_unit, tau0,
                                     alpha_res=alpha_res))               # (4,Kc) MF factor
    P_filt_mf = P_filt * corr                                            # corrected per class
    P_clean = P_filt_mf[0]
    excess = _excess_from_P_filt(P_filt_mf, dla_core)                    # (3,Kc)
    return P_clean + jnp.einsum("c,ck->k", jnp.asarray(alpha_hcd), excess)


# ============================================================================ #
#  MF C_emu floor (LF→HR generalization + n_s-edge extrapolation), T4/T5b.
#
#  The spec (hcd_priya_notes/docs/superpowers/onboarding/2026-06-08-mf-cemu-floor-spec.md) sizes a
#  z-resolved, per-band ADDITIVE diagonal variance term that inflates C_emu on the
#  SMALL-SCALE leg(s) to cover the residual the FIXED θ-blind MF correction leaves
#  AFTER it is applied (the LF→HR generalization error), plus a SEPARATE n_s-edge
#  extrapolation budget for ns outside the HR cluster [0.86, 0.98].
#
#  Two terms, ADDED in quadrature on the variance, both fractional on the TOTAL P_obs:
#    §2  sigma_floor(z, band)          — in-cluster generalization, FIXED, ns-independent
#    §4  sigma_edge(z, band; ns)       — out-of-cluster ns-extrapolation, ns-DEPENDENT
#  Assembly (small-scale leg only): emu_var(k,z) += [(σ_floor + ... ⊕ σ_edge)·P_obs(k)]²
#  i.e. emu_var(k,z) += (sigma_floor·P_obs)² + (sigma_edge·P_obs)².
#
#  Both terms carry NO gradient toward the MAP (stop_gradient'd): they are a fixed
#  data-side table (σ_floor) and a fixed FUNCTION of ns (σ_edge). They only widen the
#  posterior; they cannot bias θ. The legs only reach z=4.6 (KS) / z=4.2 (DESI), so the
#  z>4.6 floor cells (incl. the flagged single-sim z=5.4 spike) are NEVER indexed — we
#  interpolate/clamp σ_floor(z,·) onto the LEG z's; out-of-range z>z_grid.max() would
#  pin to the last grid value but is never reached (asserted at build time).
# ============================================================================ #
class MFFloor(NamedTuple):
    """The z-resolved per-band MF C_emu floor table (from ``mf_cemu_floor.npz``).

    Fields (numpy on the host; jnp-cast where used in the differentiable path):
      z_grid       : (Nz_floor,)   the floor z-grid (2.0…5.4; only z ≤ 4.6 are reached).
      sigma_floor  : (Nz_floor, 2) fractional in-cluster floor σ [LFres(k<0.069), extrap(k≥0.07)].
      slope        : (Nz_floor, 2) |d coherent/d ns| per (z, band) for the edge budget.
      k_band_split : float         the LFres/extrap band edge in angular k (0.07 s/km).
      ns_box       : (2,)          the HR ns cluster box [0.86, 0.98] for d_ns(ns).
      floor_min    : float         the FLOOR_MIN lower bound on σ_floor (1.23%).
      edge_slope_mult : float      the 2× on the (noisy 6-sim) edge slope (spec §4.1).
    """
    z_grid: np.ndarray
    sigma_floor: np.ndarray
    slope: np.ndarray
    k_band_split: float
    ns_box: np.ndarray
    floor_min: float
    edge_slope_mult: float


# the LFres/extrap band edge: LF-resolvable k<0.069 | extrapolated k≥0.07. The spec
# uses 0.07 as the cut for "band(k)"; bins with k≥this are the extrapolated band.
MF_FLOOR_K_BAND_SPLIT = 0.07


def load_mf_floor(npz_path=f"{_REPO}/figures/analysis/04_emulator/mf_cemu_floor.npz",
                  *, k_band_split=MF_FLOOR_K_BAND_SPLIT, edge_slope_mult=2.0):
    """Load the certified MF C_emu floor table → an ``MFFloor`` (spec §2/§4).

    Reads the z-resolved per-band ``sigma_floor`` (in-cluster generalization) + ``slope``
    (the |d coherent/d ns| edge-budget rate) + the HR ``ns_box`` + ``FLOOR_MIN``. The
    npz stores the floor for z∈[2.0,5.4]; the leg binding only ever indexes z ≤ 4.6, so
    the z>4.6 cells (incl. the flagged z=5.4 spike) are carried but never read."""
    d = np.load(npz_path, allow_pickle=True)
    return MFFloor(
        z_grid=np.asarray(d["z_grid"], float),
        sigma_floor=np.asarray(d["sigma_floor"], float),       # (Nz,2)
        slope=np.asarray(d["slope"], float),                   # (Nz,2)
        k_band_split=float(k_band_split),
        ns_box=np.asarray(d["ns_box"], float),                 # (2,)
        floor_min=float(d["FLOOR_MIN"]),
        edge_slope_mult=float(edge_slope_mult))


def _mf_floor_sigma_at_z(floor: MFFloor, z):
    """Interpolate (and end-clamp) the per-band σ_floor + slope onto a scalar leg z.

    Returns (sigma_floor_z (2,), slope_z (2,)) at the LEG redshift ``z`` via 1-D linear
    interp on the floor z-grid, clamped to the grid ends (np.interp clamps for free).
    Host-side numpy (the floor is a FIXED θ-blind table; no gradient). The slope is
    end-clamped too (NaN slope cells — e.g. z=2.0 LFres — are nan_to_num'd to 0)."""
    zg = floor.z_grid
    sig = np.array([np.interp(float(z), zg, floor.sigma_floor[:, b]) for b in range(2)])
    slp = np.array([np.interp(float(z), zg, np.nan_to_num(floor.slope[:, b])) for b in range(2)])
    return sig, slp


def _mf_floor_var_on_k(floor: MFFloor, z, k_sub, P_obs_sub, ns):
    """The MF C_emu floor VARIANCE on a single z's leg rows (the additive diagonal term).

    Per row k: pick the band (LFres if k<k_band_split else extrap), then
        var(k) = [(σ_floor(z,band) ⊕? ) · P_obs(k)]²  +  [σ_edge(z,band;ns) · P_obs(k)]²
    with σ_edge = max( edge_slope_mult·|slope(z,band)|·d_ns , 0.5·σ_floor·(d_ns/0.03) ) and
    d_ns(ns) = max(ns − ns_hi, ns_lo − ns, 0) (zero inside the HR box; the edge term then
    vanishes and only σ_floor remains).

    The σ_floor/slope table is FIXED (θ-blind, no grad). ``ns`` enters σ_edge only through
    d_ns(ns); the caller stop_gradients ns so the edge term carries NO gradient toward the
    MAP (spec §4.1). P_obs_sub IS differentiable (the floor is fractional on the model
    P_obs), so the term grows/shrinks with the predicted power — exactly the spec's
    ``(σ·P_obs)²``. Returns a (len(k_sub),) variance, in ABSOLUTE P² units."""
    sig_z, slp_z = _mf_floor_sigma_at_z(floor, z)              # (2,), (2,) host
    # d_ns: 0 inside [ns_lo, ns_hi]; the distance outside otherwise (stop-grad'd by caller).
    ns_lo, ns_hi = float(floor.ns_box[0]), float(floor.ns_box[1])
    d_ns = jnp.maximum(jnp.maximum(ns - ns_hi, ns_lo - ns), 0.0)
    # per-band → per-row via the k mask (LFres band 0, extrap band 1).
    k_sub = jnp.asarray(k_sub)
    is_extrap = (k_sub >= floor.k_band_split)
    sig_floor_row = jnp.where(is_extrap, sig_z[1], sig_z[0])   # (Nrow,)
    slope_row = jnp.where(is_extrap, slp_z[1], slp_z[0])
    # σ_edge(z, band; ns) = max( mult·|slope|·d_ns , 0.5·σ_floor·(d_ns/0.03) )   (spec §4.1).
    sig_edge_row = jnp.maximum(floor.edge_slope_mult * jnp.abs(slope_row) * d_ns,
                               0.5 * sig_floor_row * (d_ns / 0.03))
    P = jnp.asarray(P_obs_sub)
    # ADD in quadrature on the variance: (σ_floor·P)² + (σ_edge·P)².
    return (sig_floor_row * P) ** 2 + (sig_edge_row * P) ** 2


# ============================================================================ #
#  SHAPE-AWARE MF C_emu floor (Phase-5a, 2026-06-12).
# ----------------------------------------------------------------------------#
#  The diagonal floor above (_mf_floor_var_on_k) inflates each (z,k) cell
#  INDEPENDENTLY. Phase-5a Test B (the genuine HF-LOSO closure) showed the LF→HR
#  resolution residual is a COHERENT k-tilt (78% rank-1, coherent across z) that
#  biases n_s up to +2.8σ — a CORRELATED structure a diagonal floor cannot absorb
#  without grossly over-inflating every cell. The shape floor adds the measured
#  per-held-out-sim LOSO-eps OUTER-PRODUCT as a (low-rank) covariance:
#
#    f_shape[(z,k),(z',k')] = (1/N_sim) Σ_sim eps_sim(z,k)·eps_sim(z',k')   (fractional)
#
#  whose DIAGONAL is the per-(z,k) mean-square eps (the RMS the diagonal floor
#  undersized) and whose OFF-DIAGONAL is the coherent tilt + cross-z coherence the
#  n_s direction reads. On a leg: C_shape[i,j] = infl²·f_bound[i,j]·P_obs[i]·P_obs[j]
#  (f_bound = f_shape interpolated onto the leg's flat (z,k) rows). f_shape is PSD
#  (a sum of outer products) ⇒ C_shape PSD ⇒ C_total stays PD. θ-blind & fixed (the
#  only θ-coupling is the fractional ·P_obs⊗P_obs scaling, like the diagonal floor).
#  Built by scripts/build_mf_shape_floor.py → mf_cemu_shape.npz.
# ============================================================================ #
class MFShape(NamedTuple):
    """The shape-aware MF C_emu floor table (from ``mf_cemu_shape.npz``).
      z       : (Nz_c,)            cache z grid (≤ z_max, the legs only reach z≤4.6).
      k       : (Nk_c,)            cache angular-k grid (the LF Nyquist band 0.01–0.069 s/km).
      f_shape : (Nz_c·Nk_c, ...)   fractional LOSO-eps second-moment (z-major flat, symmetric PSD).
      n_sim   : int                the number of held-out HR sims it was built from.
    """
    z: np.ndarray
    k: np.ndarray
    f_shape: np.ndarray
    n_sim: int


def load_mf_shape(npz_path=f"{_REPO}/hcd_analysis/_emulator_data/mf_cemu_shape.npz"):
    """Load the shape-aware MF floor table → an ``MFShape`` (scripts/build_mf_shape_floor.py)."""
    d = np.load(npz_path, allow_pickle=True)
    return MFShape(z=np.asarray(d["z"], float), k=np.asarray(d["k"], float),
                   f_shape=np.asarray(d["f_shape"], float), n_sim=int(d["n_sim"]))


def _logk_interp_weights(k_target, k_grid):
    """Linear-in-log10(k) interpolation weights of one target k onto ``k_grid`` (Nk,).
    Two nonzero entries (the bracketing cache-k bins); ZERO outside [k_grid.min, k_grid.max]
    (the shape floor is NOT extrapolated beyond the LF Nyquist band it was measured on)."""
    lk = np.log10(np.asarray(k_grid, float)); x = np.log10(float(k_target))
    w = np.zeros(len(lk))
    if x < lk[0] - 1e-12 or x > lk[-1] + 1e-12:
        return w
    j = int(min(max(np.searchsorted(lk, x) - 1, 0), len(lk) - 2))
    t = (x - lk[j]) / (lk[j + 1] - lk[j])
    w[j] = 1.0 - t; w[j + 1] = t
    return w


def mf_shape_cov_for_leg(shape, leg, z_tol=0.1):
    """The leg's FRACTIONAL shape covariance ``C_shape_frac`` (N,N), z-major matching the leg's
    flat rows: ``C_shape_frac = S f_shape Sᵀ`` with ``S`` (N, Nz_c·Nk_c) interpolating each flat
    row's (z,k) onto the cache grid (nearest cache-z; linear-in-log-k; zero beyond the LF band).
    Host numpy (θ-blind, fixed; precompute once per leg). PSD (f_shape PSD ⇒ S f_shape Sᵀ PSD).

    Works for any table with ``.z``/``.k``/``.f_shape`` (the resolution ``MFShape`` AND the
    LF-emulator-coherence ``MFEmuCoh``). ``z_tol`` (cache half-Δz): a leg row whose z is FARTHER
    than ``z_tol`` from every cache z gets a ZERO binder row — no silent z-extrapolation. The
    resolution table spans z≤4.6 so every leg z (DESI≤4.2, KS≤4.6) is in support (byte-identical
    to before); the emucoh table spans z≤4.4 (full leg-z), so all DESI z (≤4.2) are floored and only
    KS's z=4.6 bin (the one row beyond the table) correctly vanishes."""
    zc, kc, F = shape.z, shape.k, shape.f_shape
    Nz_c, Nk_c = len(zc), len(kc)
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]   # (N,) z of each flat row
    k_row = np.asarray(leg.k)                          # (N,)
    N = len(k_row)
    S = np.zeros((N, Nz_c * Nk_c))
    for i in range(N):
        zi = int(np.argmin(np.abs(zc - z_row[i])))     # nearest cache z
        if abs(float(zc[zi]) - float(z_row[i])) > z_tol:
            continue                                   # leg z outside the table's z support → zero row
        S[i, zi * Nk_c:(zi + 1) * Nk_c] = _logk_interp_weights(k_row[i], kc)
    return S @ F @ S.T


class MFEmuCoh(NamedTuple):
    """The 60-sim LF-EMULATOR-COHERENCE C_emu table (from ``mf_cemu_emucoh.npz``,
    scripts/build_mf_emucoh_floor.py). Same shape as ``MFShape`` (z, k, fractional second-moment
    f_shape on the (z,k) flat grid) so it binds via ``mf_shape_cov_for_leg`` verbatim — but it is
    a DIFFERENT residual: the LF emulator's held-out (8-fold, 60-sim LOSO) k-coherent generalization
    gap (clean class), NOT the LF→HR resolution residual. Built over the FULL leg-z range (z∈[2.0,4.4],
    13 bins) — the within-z coherence persists 0.59–0.79 across all z (incl. He-II reion z≈3–4), so it
    is NOT restricted to low-z; top-m truncated (top-15 ≈ 96.8% of trace).

    NOTE (diagonal allocation, DEPLOYED): its diagonal is the COHERENT part of the LF-emulator error, a
    SUBSET of the existing ``emu_var`` (the cross-class ρ diagonal = the FULL per-cell second moment).
    PRODUCTION sets ``mf_emucoh_offdiag_only=True`` (run_real_fit / run_prod_sbc): the per-term diagonal
    allocation absorbs this term's diagonal into ``emu_var`` via MAX (counted ONCE, never under-count)
    and adds ONLY its off-diagonal — so there is NO diagonal double-count with the base ``emu_var``
    (assembly at data_likelihood.py:1130-1138). The n_s-relevant value is the OFF-diagonal. Only with
    ``offdiag_only=False`` (the class DEFAULT, NOT the deployed path) is the term added in full: the
    coherent diagonal then lands ON TOP of ``emu_var`` — a small CONSERVATIVE over-count (over-widens
    ~×1.3 on the clean diagonal, never biases)."""
    z: np.ndarray
    k: np.ndarray
    f_shape: np.ndarray
    n_sim: int


def load_mf_emucoh(npz_path=f"{_REPO}/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"):
    """Load the 60-sim LF-emulator-coherence table → an ``MFEmuCoh`` (scripts/build_mf_emucoh_floor.py)."""
    d = np.load(npz_path, allow_pickle=True)
    return MFEmuCoh(z=np.asarray(d["z"], float), k=np.asarray(d["k"], float),
                    f_shape=np.asarray(d["f_shape"], float), n_sim=int(d["n_sim"]))


# ============================================================================ #
#  Model → leg binding
# ============================================================================ #
def _emu_var_on_cache(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                      pf_stats, sigma_zb, alpha_centres, dla_core, cemu_inflate,
                      rho_zb=None):
    """Per-z emulator-error variance on the CACHE k-grid (K_cache,), assembled EXACTLY as
    ``inference.predict_P_obs_and_cov_single_z`` does. ONE of two forms (C stays DIAGONAL
    IN k either way — only the per-k class structure changes):

      DIAGONAL (default, ``rho_zb=None``):
        emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c²,  coef = [1−Σα, α_LLS, α_subDLA, α_DLA].
      CROSS-CLASS (opt-in, ``rho_zb`` (4,4,Kc,Tb) given):
        emu_var = Σ_{c,c'} coef_c·coef_c'·ρ_cc'(k,z,τ₀)·P_c·P_c',
        ρ the τ₀-interp'd 4×4 cross-class block (``rho_at_tau0``); the diagonal ρ_cc=σ_c²
        recovers the diagonal form. ρ a sample covariance ⇒ SPD ⇒ emu_var ≥ 0 GUARANTEED.

    The cross-class path here is the EXACT k-grid analog of
    ``inference.predict_P_obs_and_cov_single_z``'s ``rho_zb`` branch (same einsum, same
    NaN-guard via ``rho_at_tau0``). NaN-safe (out-of-range cache cells σ/ρ→0).
    Differentiable in (θ9, τ₀, α)."""
    from hcd_analysis.emulator.predict import predict_P_filt
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)        # (4,Kc)
    return emu_var_modes(P_filt, z, tau0, alpha_hcd, dla_core=dla_core, sigma_zb=sigma_zb,
                         alpha_centres=alpha_centres, cemu_inflate=cemu_inflate, rho_zb=rho_zb)


def predict_P_obs_on_leg(model, theta9, tau0_vec, alpha_hcd, *, pf_stats, dla_core,
                         cache_k, leg, sigma_zb=None, alpha_centres=None,
                         cemu_inflate=1.0, a_SiIII=0.0, a_SiII=0.0,
                         k_SiIII=K_SiIII_DEFAULT, k_SiII=K_SiII_DEFAULT, b_res=0.0, b_res_vec=None,
                         rho_zb=None, mf=None, mf_floor=None,
                         mf_shape_cov=None, mf_shape_infl=1.0,
                         mf_emucoh_cov=None, mf_emucoh_infl=1.0,
                         mf_emucoh_offdiag_only=False, alpha_res=None,
                         f_SiIII_nodes=None, f_SiII_nodes=None, metal_node_z=(2.2, 4.2),
                         k_SiIII_nodes=None, k_SiII_nodes=None,
                         require_zresolved=False):
    """Bind the emulator forward model to ONE leg's grid → flat (P_model (N,), C_total (N,N)).

    For each z in ``leg.z``:
      1. predict P_obs on the CACHE k-grid (``predict.predict_P_obs``; z_unit=(z−2)/3.4,
         τ₀ = the per-z entry of ``tau0_vec``);
      2. apply the forward-model nuisances (metals, resolution) — per-leg-gated by
         ``leg.metals_on`` / ``leg.resolution_on`` (default OFF);
      3. ``jnp.interp`` P_obs from ``cache_k`` onto this z's leg-k subset (differentiable; the
         emulator P_obs(k) is smooth → linear interp to bin-CENTRE; see CS-REVIEW);
         PRE-2026-10 INTERFACE: a single ``cache_k`` for every (z, theta) is the defect of the incident note
         2026-10-05-INCIDENT-kgrid-representation-regression; replaced at gate E by kcoord.KGrid / EmulatorPrediction.
      4. interp the per-k emu variance (``_emu_var_on_cache``) onto the leg k → diag(C_emu).

    Returns the FLAT (z-major) (P_model, C_total) with C_total = C_data + diag(C_emu).
    ``sigma_zb``/``alpha_centres`` may be None ⇒ C_emu = 0 (a pure data-covariance fit, e.g.
    the mock-noise generator supplies its own C).  Differentiable in
    (θ9, τ₀_vec, α, a_SiIII, a_SiII, b_res).

    ``rho_zb`` (opt-in, (n_z,4,4,Kc,Tb)): the CROSS-CLASS 4×4 block per z (its leading axis
    matches ``sigma_zb``'s); when given, C_emu's per-k variance uses the cross-class form
    (``emu_var = Σ_cc' coef_c·coef_c'·ρ_cc'·P_c·P_c'``) instead of the diagonal
    Σ coef²σ²P². ``alpha_centres`` is still required (the τ₀-interp abscissa). Default None →
    the diagonal σ path (UNCHANGED). Mirrors the ``sigma_zb`` per-z plumbing.

    NOTE per-z C_emu blocks are placed on the FLAT diagonal in the SAME z-major order as the
    leg's rows (``leg.z_idx``), so C_emu's ordering matches C_data (CS-REVIEW).

    ``mf`` (opt-in, default None → the LF reference path UNCHANGED): a ``MultiFidelity``
    whose ``eval_logk == log10(cache_k)``. When given, step (1)'s per-z P_obs goes through
    the MF forward (``_predict_P_obs_mf``: LF P_filt × exp(g + log res_corr) per class, then
    the SAME clean+excess HCD combination) instead of the raw LF ``predict_P_obs`` — the
    PROMOTED, certified gate path (``scripts/diag_emu_bias_allfolds_mf.py``). The LF backbone
    in ``mf`` is FROZEN (no grad to its weights); pass the SAME frozen backbone object as
    both ``model`` and ``mf.lf_model`` so the production forward is bit-identical to the
    gate. C_total is taken from the UNCHANGED LF path EXCEPT the MF C_emu floor (below).
    Differentiable in (θ9, τ₀_vec, α).

    ``mf_floor`` (opt-in, default None): an ``MFFloor`` table (``load_mf_floor``). When
    given AND ``mf is not None`` AND ``leg.mf_floor_on`` is True (the SMALL-SCALE leg — KS,
    or a high-k DESI row above the LF Nyquist), the LF→HR generalization floor + the
    n_s-edge extrapolation budget are ADDED to the C_emu diagonal in variance units:
    ``emu_var(k,z) += (σ_floor(z,band(k))·P_obs)² + (σ_edge(z,band(k);ns)·P_obs)²``
    (spec hcd_priya_notes/docs/superpowers/onboarding/2026-06-08-mf-cemu-floor-spec.md §2.3/§4.2). The
    floor is a FIXED θ-blind data-side table; the edge term's ns-dependence is
    stop_gradient'd → it widens the posterior, it cannot bias the MAP. The floor is applied
    on the leg z's via 1-D interp/clamp of σ_floor(z,·); the legs only reach z≤4.6 so the
    z>4.6 cells (incl. the flagged z=5.4 spike) are NEVER indexed. With ``mf_floor=None``
    (or on a leg with ``mf_floor_on=False``, or ``mf=None``) C_total is the UNCHANGED LF
    path (byte-identical, back-compat).

    ``alpha_res`` (opt-in, Task 1.3, default None → byte-identical back-compat): a
    ``(alpha0, s)`` tuple marginalizing the multi-fidelity res_corr AMPLITUDE
    FORWARD-ONLY — ``log res_corr → α(z)·log res_corr`` with
    ``α(z) = alpha0·((1+z)/(1+Z_PIVOT))^s`` (Z_PIVOT=3.0), threaded into
    ``_predict_P_obs_mf → _mf_corr_on_cache``. Affects P_model ONLY (the forward), not
    C_total. ``None`` (and ``(1.0, 0.0)``) ⇒ α≡1 ⇒ the MF golden is byte-exact. Only fires
    through the MF forward (``mf is not None``).

    ``f_SiIII_nodes`` / ``f_SiII_nodes`` (opt-in, MODEL C, default None → byte-identical
    scalar path): a (2,) array of the metal flux-decrement f at the two z-nodes
    ``metal_node_z`` (ascending). When given, the per-z SiIII (and SiII) oscillation
    amplitude is ``a(z) = f(z)/(1−⟨F⟩(z))`` with ``⟨F⟩(z)=exp(−τ₀_vec[iz])`` (the SAMPLED
    mean flux, so a(z) is differentiable in τ₀) and ``log10 f(z)`` LINEAR in ``log10(1+z)``
    between the nodes (power-law-exact; ``jnp.interp`` default-CLAMPS f flat beyond the nodes,
    intended for eBOSS z>4.2). The metal FORM (``_metal_factor``) is UNCHANGED — only its
    amplitude becomes per-z. ``f_SiII_nodes=None`` ⇒ a_SiII(z)=0. The metal factor is computed
    ONCE per z and reused by BOTH P_z and the C_emu variance transform. When
    ``f_SiIII_nodes is None`` the legacy scalar ``a_SiIII``/``a_SiII`` path runs (byte-exact).

    ``k_SiIII_nodes`` / ``k_SiII_nodes`` (opt-in, MODEL C+, default None → the scalar
    ``k_SiIII``/``k_SiII``=0.05): a (2,) array of the sigmoid decorrelation SCALE at ``metal_node_z``;
    when given the per-z scale is ``10**interp(log10(1+z))`` (log10 k LINEAR in log10(1+z), like f).
    On the f-node (Model C+) path the SiIII–SiII metal-metal CROSS term is ON (``cross=True``); it is
    ∝ a_SiII so it auto-vanishes on SiIII-only legs.

    ``alpha_hcd`` is ``(n_z,3)`` z-RESOLVED per-z incidence (the DEPLOYED forward) or a ``(3,)`` z-flat
    triple broadcast to all z. A z-flat alpha on a MULTI-z leg, COMPARED to a z-resolved truth, fakes a
    spurious z-ramp -- the recurring z-flat bug that once faked a +5.5 sigma n_s. So any COMPARISON path
    (loglik / C_emu sizing / residual diagnostics) MUST build alpha via
    ``closure_legb_figs._truth_alpha_zresolved_on_leg`` and pass ``require_zresolved=True``; the comparison
    core ``_data_loglik_legcore`` DEFAULTS ``require_zresolved=True`` so the loglik / SBC re-scoring layer is
    safe-by-default. The raw forward stays permissive so byte-identity uniform-alpha references pass."""
    cache_k = jnp.asarray(cache_k)
    k_leg = jnp.asarray(leg.k)
    z_idx = np.asarray(leg.z_idx)
    R_z = jnp.asarray(leg.R_z)
    tau0_vec = jnp.asarray(tau0_vec)
    alpha_hcd = jnp.asarray(alpha_hcd)   # (n_z,3) z-RESOLVED per-z incidence (the DEPLOYED forward), OR
    #                                      (3,) z-flat broadcast to all z (byte-identity references only).
    # ===================== z-FLAT-ALPHA BUG CLASS (read before writing a diagnostic) =====================
    # A (3,) z-FLAT alpha is SILENTLY broadcast to EVERY z (ndim==1 -> the same incidence at all z). On a
    # MULTI-z leg, comparing that z-flat forward to a z-RESOLVED truth (per-z HCD incidence w_c rises ~3.5x
    # over z) manufactures a spurious z-ramp -- it once faked a PHANTOM +5.5 sigma n_s (a z-flat C_emu /
    # walkthrough DIAGNOSTIC vs the deployed z-resolved forward; the real bias was +0.6 sigma). The raw
    # forward STAYS permissive (byte-identity uniform-alpha references legitimately pass a (3,) alpha). The
    # RULE that keeps the bug dead lives one level up: EVERY code path that COMPARES this forward to a
    # z-resolved truth (loglik / C_emu sizing / residual diagnostics) MUST
    #   (a) build alpha via closure_legb_figs._truth_alpha_zresolved_on_leg  -- the ONLY sanctioned
    #       comparison-alpha builder (never hand-build alpha from a z-median w_c), AND
    #   (b) pass require_zresolved=True (the comparison core _data_loglik_legcore DEFAULTS it True, so the
    #       loglik / SBC re-scoring layer is SAFE-BY-DEFAULT: a z-flat alpha there RAISES, not broadcasts).
    # See [[feedback-diagnostic-deployment-consistency]].
    if require_zresolved and alpha_hcd.ndim != 2:
        # explicit raise (NOT assert): this guard is load-bearing now that the comparison core
        # _data_loglik_legcore defaults require_zresolved=True, so it must survive `python -O`
        # (which strips asserts and would silently reopen the z-flat broadcast on the loglik path).
        raise ValueError(
            f"predict_P_obs_on_leg(require_zresolved=True): alpha_hcd must be z-RESOLVED (n_z,3), got "
            f"ndim={alpha_hcd.ndim} shape={tuple(alpha_hcd.shape)}. A (3,) z-flat alpha would be broadcast "
            f"to every z and produce a spurious z-ramp vs the z-resolved truth (the recurring z-flat bug; "
            f"once a phantom +5.5 sigma n_s). Build alpha via _truth_alpha_zresolved_on_leg.")
    # PER-LEG DLA-forward scaling (§0c): the sampled α_DLA's DLA-excess contribution is scaled by
    # leg.dla_forward_frac (DESI 1.0 → full residual; KS 0.0 → the forward DLA term is 0, matching
    # the 0% KS closure target). We fold the per-leg fraction into the DLA component of α so BOTH
    # the forward P_obs AND the C_emu DLA-class coefficient stay consistent (KS → 0 on both).
    dff = float(getattr(leg, "dla_forward_frac", 1.0))
    if dff != 1.0:
        dla_scale = jnp.array([1.0, 1.0, dff])             # (3,) [LLS, subDLA, DLA]
        alpha_hcd = alpha_hcd * (dla_scale if alpha_hcd.ndim == 1 else dla_scale[None, :])
    N = k_leg.shape[0]

    P_model = jnp.zeros(N)
    emu_var_flat = jnp.zeros(N)
    floor_var_flat = jnp.zeros(N)
    # C_emu fires if EITHER the diagonal σ OR the cross-class ρ is supplied (with the
    # τ₀-interp abscissa). The cross-class path reads ρ only; the diagonal reads σ only.
    have_emu = ((sigma_zb is not None or rho_zb is not None)
                and alpha_centres is not None)
    # The MF C_emu floor fires only THROUGH the MF forward, on the small-scale leg(s).
    have_floor = (mf is not None) and (mf_floor is not None) and bool(leg.mf_floor_on)
    if have_floor:
        # PI invariant: the legs only reach z≤4.6 (DESI 4.2, KS 4.6); the z>4.6 floor cells
        # (incl. the flagged single-sim z=5.4 spike) are NEVER indexed. Assert it loudly so
        # a future leg-z change can't silently start reading the un-trustworthy high-z cells.
        assert float(np.max(leg.z)) <= 4.6 + 1e-6, (
            f"MF floor: leg {leg.name} z max {float(np.max(leg.z)):.3f} > 4.6 — the floor "
            f"table above z=4.6 (incl. the z=5.4 spike) is NOT trustworthy and must not be "
            f"indexed (spec PI note)")
        # the n_s-edge term is a FIXED function of ns that carries NO gradient toward the MAP
        # (it widens the posterior near the cluster edge; it must not pull θ). stop_gradient.
        ns_phys_sg = jax.lax.stop_gradient(_ns_phys_from_theta9(theta9))

    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z = float(leg.z[iz])
        z_unit = float(leg.z_unit[iz])
        tau0 = tau0_vec[iz]
        k_sub = k_leg[jnp.asarray(rows)]
        alpha_z = alpha_hcd if alpha_hcd.ndim == 1 else alpha_hcd[iz]  # per-z HCD incidence

        # (1) P_obs on the cache grid — LF reference (mf=None) OR the MF forward (opt-in).
        if mf is None:
            P_cache = predict_P_obs(model, theta9, z_unit, tau0, alpha_z, pf_stats, dla_core)
        else:
            P_cache = _predict_P_obs_mf(mf, model, theta9, z_unit, tau0, alpha_z,
                                        pf_stats, dla_core, alpha_res=alpha_res)
        # (3) interp to the leg's k (bin centres); model is smooth → linear interp.
        P_z = jnp.interp(k_sub, cache_k, P_cache)
        # (2) forward-model nuisances (gated; default OFF → factor ≡ 1). Compute the metal factor
        # ONCE per iz (MODEL C per-z amplitude OR the legacy scalar) and REUSE it for BOTH P_z and the
        # C_emu variance transform (ev_z·mfac²) so the two can never drift to a stale amplitude.
        mfac = None
        if leg.metals_on:
            mfac = metal_factor_at_z(k_sub, z, tau0, a_SiIII=a_SiIII, a_SiII=a_SiII, k_SiIII=k_SiIII,
                                     k_SiII=k_SiII, f_SiIII_nodes=f_SiIII_nodes, f_SiII_nodes=f_SiII_nodes,
                                     metal_node_z=metal_node_z, k_SiIII_nodes=k_SiIII_nodes,
                                     k_SiII_nodes=k_SiII_nodes)
            P_z = P_z * mfac
        if leg.resolution_on:
            # option-b: a per-z sampled b_res(z) (b_res_vec) overrides the scalar b_res (default None →
            # scalar, byte-identical). f_res forward threading; the injected truth never sets it.
            b_res_iz = b_res if b_res_vec is None else b_res_vec[iz]
            P_z = P_z * _resolution_factor(k_sub, R_z[iz], b_res=b_res_iz)
        P_model = P_model.at[jnp.asarray(rows)].set(P_z)

        # (4) C_emu: per-k emu variance interp'd onto the leg k
        if have_emu:
            rho_zb_z = rho_zb[iz] if rho_zb is not None else None
            sigma_zb_z = sigma_zb[iz] if sigma_zb is not None else None
            ev_cache = _emu_var_on_cache(
                model, theta9, z_unit, z, tau0, alpha_z, pf_stats=pf_stats,
                sigma_zb=sigma_zb_z, alpha_centres=alpha_centres, dla_core=dla_core,
                cemu_inflate=cemu_inflate, rho_zb=rho_zb_z)
            ev_z = jnp.interp(k_sub, cache_k, ev_cache)
            # emu var transforms by the SAME multiplicative nuisance factors² (variance units) —
            # REUSE the per-iz mfac computed above for P_z (Model C per-z OR the legacy scalar).
            if leg.metals_on:
                ev_z = ev_z * mfac ** 2
            if leg.resolution_on:
                b_res_iz = b_res if b_res_vec is None else b_res_vec[iz]
                ev_z = ev_z * _resolution_factor(k_sub, R_z[iz], b_res=b_res_iz) ** 2
            emu_var_flat = emu_var_flat.at[jnp.asarray(rows)].set(ev_z)

        # MF C_emu floor: the LF→HR generalization term + the n_s-edge extrapolation budget,
        # ADDED in variance units on the total (post-nuisance) model P_obs (spec §2.3/§4.2).
        # P_z is differentiable (fractional floor on the model power); the ns-edge term's
        # ns is stop_gradient'd so the floor never pulls the MAP — it only widens C_total.
        if have_floor:
            fv_z = _mf_floor_var_on_k(mf_floor, z, k_sub, P_z, ns_phys_sg)
            floor_var_flat = floor_var_flat.at[jnp.asarray(rows)].set(fv_z)

    # Collect the FIXED-amplitude k-coherent off-diagonal C_emu terms (each a fractional PSD
    # covariance × infl²): the LF→HR RESOLUTION shape floor (6-HR) and the LF-EMULATOR-coherence
    # term (60-sim). Both bind via mf_shape_cov_for_leg and scale by the SAME θ-INDEPENDENT
    # fiducial power (see the CRITICAL note below). The list generalizes the single-term path.
    # each term: (fractional cov, infl, offdiag_only). offdiag_only=True (the per-term diagonal
    # allocation, cosmology referee 2026-06-12) absorbs that term's diagonal into emu_var via max
    # (never under-count) and adds ONLY its off-diagonal — appropriate for emucoh, whose diagonal is
    # a SUBSET of the existing emu_var (so the default on-top add ×1.3 over-widens the clean diagonal).
    # The 6-HR resolution floor is a SEPARATE error source (not in emu_var) → always full-matrix.
    shape_terms = []
    if mf_shape_cov is not None:
        shape_terms.append((jnp.asarray(mf_shape_cov), mf_shape_infl, False))
    if mf_emucoh_cov is not None:
        shape_terms.append((jnp.asarray(mf_emucoh_cov), mf_emucoh_infl, bool(mf_emucoh_offdiag_only)))

    if not shape_terms:
        # LF / diagonal-floor path — byte-identical to before (back-compat).
        C_total = jnp.asarray(leg.C_data) + jnp.diag(emu_var_flat + floor_var_flat)
    else:
        # k-coherent off-diagonal C_emu (Phase-5a): Σ of fractional eps/coherence outer-products,
        # each scaled by infl² and a FIXED (θ-INDEPENDENT) fiducial power outer product. Conservative:
        # never reduce the existing diagonal floor — top each diagonal up to max(floor_var, Σ shape_diag)
        # and add the (PSD) coherent covariance(s). PD-safe: C_data PD + diag(≥0) + Σ PSD ⇒ C_total PD.
        #
        # CRITICAL (2026-06-12): the fiducial amplitude must be θ-INDEPENDENT (the leg's data
        # power), NOT the live P_model. A covariance that scales with the SAMPLED model power
        # (∝ P_model²) lets the fit inflate its own error in the coherent-tilt direction by
        # moving a parameter — a θ-dependent-covariance pathology that, for a rank-1 off-diagonal
        # mode, made the n_s posterior SHRINK and the MAP drift AWAY from truth (validation:
        # live-P infl1.0/1.5/2.0 → ns +3.2/+3.5/+3.7σ, worse than the +2.8σ baseline, σ shrinking
        # — impossible for a fixed PSD add, hence diagnostic of the θ-dependence). A fixed fiducial
        # makes each term a CONSTANT matrix ⇒ adding it can only WIDEN the marginal (PSD
        # monotonicity) and the MAP de-biases per the linear GLS analysis.
        P_fid = jnp.nan_to_num(jnp.asarray(leg.P_data))          # θ-independent fiducial amplitude
        PP = P_fid[:, None] * P_fid[None, :]
        # Build Σ of terms. For an offdiag_only term, absorb its diagonal into emu_var via max (so it
        # is neither double-counted nor under-counted) and add only its off-diagonal. PD-safe: with
        # emu_var_eff ≥ that term's diagonal, C_total = C_data + diag(emu_var_eff − diag + topup) +
        # Σ_full-PSD-terms, i.e. PD C_data + diag(≥0) + Σ PSD ⇒ PD (the off-diag-only term is the full
        # PSD term minus its diagonal, and the subtracted diagonal is restored inside emu_var_eff).
        C_shape = jnp.zeros_like(jnp.asarray(leg.C_data))
        emu_var_eff = emu_var_flat
        for cov, infl, offdiag_only in shape_terms:
            term = (infl ** 2) * cov * PP                        # PSD
            if offdiag_only:
                td = jnp.diagonal(term)
                emu_var_eff = jnp.maximum(emu_var_eff, td)        # absorb diagonal (never under-count)
                term = term - jnp.diag(td)                        # add only off-diagonal
            C_shape = C_shape + term
        topup = jnp.maximum(0.0, floor_var_flat - jnp.diag(C_shape))
        C_total = jnp.asarray(leg.C_data) + jnp.diag(emu_var_eff + topup) + C_shape
    return P_model, C_total


# ============================================================================ #
#  Multi-leg likelihood
# ============================================================================ #
def data_loglik(model, theta9, tau0_global, alpha_hcd, legs, *, pf_stats, dla_core,
                cache_k, z_global=None, sigma_zb_per_leg=None, alpha_centres=None,
                cemu_inflate=1.0, a_SiIII=0.0, a_SiII=0.0,
                k_SiIII=K_SiIII_DEFAULT, k_SiII=K_SiII_DEFAULT, b_res=0.0,
                jitter=1e-10, return_parts=False, rho_zb_per_leg=None, mf=None,
                mf_floor=None, mf_shape_per_leg=None, mf_shape_infl=1.0,
                mf_emucoh_per_leg=None, mf_emucoh_infl=1.0,
                mf_emucoh_offdiag_only=False):
    """PRE-2026-10 INTERFACE (single cache_k grid for every leg and z; replaced at gate E).

    Multi-leg Gaussian log-likelihood against the REAL data.

    The legs are INDEPENDENT surveys (DESI & KS share z-VALUES but are different
    instruments → no cross-covariance) ⇒ the joint covariance is BLOCK-DIAGONAL ⇒
        logL = Σ_leg gaussian_loglik(P_data_leg − P_model_leg, C_total_leg).

    τ₀ is supplied on a GLOBAL z-grid (``z_global``, ascending); each leg pulls the τ₀ for
    its own z-bins by nearest-z match (DESI z⊂global, KS z⊂global).  ``sigma_zb_per_leg`` is
    a dict {leg.name: (n_z_leg,4,Kc,Tb)} of the emulator error vector ALREADY sliced to that
    leg's z-bins (or None for a data-cov-only fit).

    ``rho_zb_per_leg`` (opt-in): a dict {leg.name: (n_z_leg,4,4,Kc,Tb)} of the CROSS-CLASS
    block sliced to each leg's z-bins → C_emu uses the cross-class form (mirrors
    ``sigma_zb_per_leg``). Default None → the diagonal σ path (UNCHANGED).

    ``mf`` (opt-in, default None → the LF reference path): a ``MultiFidelity`` (eval grid ==
    cache grid) threaded to every leg's ``predict_P_obs_on_leg`` so P_obs goes through the
    PROMOTED, certified through-MF forward (LF P_filt × exp(g + log res_corr) per class).
    The LF backbone is FROZEN. Default None keeps the LF path byte-identical (back-compat).
    The SAME frozen backbone object should be passed as both ``model`` and ``mf.lf_model``.

    ``mf_floor`` (opt-in, default None): an ``MFFloor`` (``load_mf_floor``) threaded to every
    leg's ``predict_P_obs_on_leg``. When given AND ``mf is not None``, the LF→HR generalization
    floor + the n_s-edge extrapolation budget are ADDED to C_emu on the SMALL-SCALE leg(s)
    (``leg.mf_floor_on``; KS by default, DESI off) in variance units. θ-blind (fixed table +
    stop_gradient'd ns-edge): widens the posterior, never pulls the MAP. None → no floor.

    Differentiable in (θ9, τ₀_global, α, a_SiIII, a_SiII, b_res).  ``return_parts`` →
    (logL, {name: (logL_leg, chi2_leg, dof_leg)}) for the χ²/dof diagnostic.

    LYA-CONSULT: the joint block-diagonal (no DESI–KS cross-covariance) assumes the two
    surveys' systematics are independent at the shared z — confirm.
    """
    tau0_global = jnp.asarray(tau0_global)
    z_global = np.asarray(z_global) if z_global is not None else None

    total = 0.0
    parts = {}
    for leg in legs:
        # per-leg τ₀ vector: nearest-z pull from the global ladder (each leg z ⊂ global).
        if z_global is None:
            tau0_vec = tau0_global              # already per-leg-z
        else:
            sel = np.array([int(np.argmin(np.abs(z_global - zz))) for zz in leg.z])
            tau0_vec = tau0_global[jnp.asarray(sel)]

        szb = None
        if sigma_zb_per_leg is not None:
            szb = sigma_zb_per_leg.get(leg.name)
        rzb = None
        if rho_zb_per_leg is not None:
            rzb = rho_zb_per_leg.get(leg.name)

        msc = mf_shape_per_leg.get(leg.name) if mf_shape_per_leg is not None else None
        mec = mf_emucoh_per_leg.get(leg.name) if mf_emucoh_per_leg is not None else None
        P_model, C_total = predict_P_obs_on_leg(
            model, theta9, tau0_vec, alpha_hcd, pf_stats=pf_stats, dla_core=dla_core,
            cache_k=cache_k, leg=leg, sigma_zb=szb, alpha_centres=alpha_centres,
            cemu_inflate=cemu_inflate, a_SiIII=a_SiIII, a_SiII=a_SiII,
            k_SiIII=k_SiIII, k_SiII=k_SiII, b_res=b_res, rho_zb=rzb, mf=mf,
            mf_floor=mf_floor, mf_shape_cov=msc, mf_shape_infl=mf_shape_infl,
            mf_emucoh_cov=mec, mf_emucoh_infl=mf_emucoh_infl,
            mf_emucoh_offdiag_only=mf_emucoh_offdiag_only)
        r = jnp.asarray(leg.P_data) - P_model
        ll = gaussian_loglik(r, C_total, jitter=jitter)
        total = total + ll
        if return_parts:
            # χ² = rᵀ C⁻¹ r (jittered to match the loglik factorization).
            K = C_total.shape[-1]
            Cj = C_total + (jitter * jnp.mean(jnp.diag(C_total))) * jnp.eye(K)
            L = jnp.linalg.cholesky(Cj)
            sol = jax.scipy.linalg.cho_solve((L, True), r)
            chi2 = float(r @ sol)
            parts[leg.name] = (float(ll), chi2, int(K))

    if return_parts:
        return total, parts
    return total
