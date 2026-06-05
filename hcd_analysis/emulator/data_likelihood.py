"""Phase-C data-binding layer — the REAL DESI DR1 + KODIAQ-SQUAD P1D likelihood.

This is the path the Leg-B coverage gate and the eventual production fit use; it is a
NEW, parallel module to the Leg-A cache-grid driver (``inference.py`` /
``closure_sbc``), which it leaves untouched. The Leg-A driver evaluates the likelihood
on the cache's own 172 angular-k grid (the sim/closure path); THIS module binds the
emulator forward model to the OBSERVED survey grids (DESI 85 angular-k × 12 z, KS 13
angular-k × 14 z), applying the published cuts + covariances.

Contract (per ``docs/superpowers/2026-06-05-desi-dr1-p1d-usage.md``):

  per leg ``DataLeg``:  z (Nz,), k (Nz*Nk flat or (Nz,Nk)), P_data (N,), C_data (N,N),
                        R_z (Nz,) resolution, plus systematic-model flags.
  binding:  for each z, predict P_obs on the CACHE k-grid (``predict.predict_P_obs``),
            ``jnp.interp`` onto the leg's k (differentiable; the model is smooth), then
            ×(metal) ×(resolution) nuisances (per-leg-configurable, default OFF), assemble
            C_emu on the leg grid (interp the per-k emu variance), C_total = C_data + C_emu.
  likelihood:  the two legs are INDEPENDENT surveys ⇒ block-diagonal ⇒
            logL = Σ_leg gaussian_loglik(P_data_leg − P_model_leg, C_total_leg).

Everything in the binding/likelihood path is JAX-pure and differentiable in
(θ9, τ₀_vec, α_hcd, a_SiIII, a_SiII, b_res). Loaders are numpy (host-side, eval-only).

x64 is asserted (import ``hcd_analysis.emulator`` before jax). No donate.

LYA-CONSULT items are tagged ``LYA-CONSULT:`` inline; see the module-level report.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .data import KIM_AMP, KIM_SLOPE
from .predict import predict_P_obs
from .likelihood import sigma_at_tau0, gaussian_loglik

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"

# --- physical constants -------------------------------------------------------
C_KMS = 299792.458            # speed of light [km/s]
LAMBDA_LYA = 1215.67          # Lyα rest wavelength [Å]
LAMBDA_SiIII = 1206.50        # SiIII line [Å]
LAMBDA_SiII = 1190.42         # SiII line [Å]  (1190/1193 doublet; use 1190.42 leading line)
# McDonald (2006) metal-damping smoothing scale [s/km] (companion Eq. 4.3).
METAL_KS = 0.009
# DESI spectral pixel width used for the resolution proxy R_z [Å] (usage doc Eq. 4.8).
DESI_PIXEL_ANGSTROM = 0.8

# The emulator cache k-grid Nyquist (angular k, s/km == 1/Å in the velocity-equiv
# convention the data + cache share). The emulator cannot predict above this.
CACHE_KMAX = 0.069
# KS: drop the first four k-bins (Karaçaylı 2306.06316 Fig 11 — they underestimate the
# error). The fourth bin centre is 0.0157527, so keep k > 0.0158.
KS_DROP_KMAX = 0.0158
# DESI continuum-floor low-k cut + half-Nyquist resolution high-k cut (usage doc §"cuts").
DESI_KMIN = 1e-3


# ============================================================================ #
#  DataLeg container
# ============================================================================ #
class DataLeg(NamedTuple):
    """One independent P1D measurement leg (DESI or KS), post-cut, flat z-major.

    Fields (all numpy on the host; jnp-cast at binding time):
      name        : "DESI" | "KS"
      z           : (Nz,)   unique redshifts kept (ascending)
      z_unit      : (Nz,)   (z − Z_LIMITS[0]) / (Z_LIMITS[1] − Z_LIMITS[0]) per the cache
      k           : (N,)    flat angular k per kept (z,k) row (z-major)
      z_row       : (N,)    the z value of each flat row (z-major; matches C_data order)
      z_idx       : (N,)    index into ``z`` for each flat row
      P_data      : (N,)    the data vector (DESI ``plya``; KS conservative ``power``)
      C_data      : (N,N)   full data covariance (DESI full COV + cov_diag_inflation; KS full)
      R_z         : (Nz,)   resolution scale R_z = c·0.8Å/((1+z)·1215.67Å) (DESI); KS its own
      n_z, n_per_z: counts (n_per_z is per-z if ragged; here a (Nz,) int array)
      metals_on   : bool    whether the SiIII/SiII model term is applied for this leg
      resolution_on: bool   whether the exp(2 b_res k² R_z²) template knob is applied
    """
    name: str
    z: np.ndarray
    z_unit: np.ndarray
    k: np.ndarray
    z_row: np.ndarray
    z_idx: np.ndarray
    P_data: np.ndarray
    C_data: np.ndarray
    R_z: np.ndarray
    n_z: int
    n_per_z: np.ndarray
    metals_on: bool
    resolution_on: bool


def _z_unit(z):
    """(z − 2.0)/3.4 — the cache encoder's z_unit (data.Z_LIMITS = (2.0, 5.4))."""
    return (np.asarray(z) - 2.0) / 3.4


def desi_resolution_R(z):
    """DESI resolution scale R_z = c·0.8Å / ((1+z)·1215.67Å)  [s/km] (usage doc Eq. 4.8)."""
    z = np.asarray(z)
    return C_KMS * DESI_PIXEL_ANGSTROM / ((1.0 + z) * LAMBDA_LYA)


# ============================================================================ #
#  Loaders
# ============================================================================ #
def load_desi_leg(npz_path="/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz",
                  *, z_lo=2.2, z_hi=4.2, k_min=DESI_KMIN, metals_on=True,
                  resolution_on=False, add_cov_diag_inflation=True):
    """Load DESI DR1 P1D → a post-cut ``DataLeg`` (usage doc §"Covariance + cuts").

    Cuts (z-major flat layout, ``row_is_zmajor=True``):
      * z in [z_lo, z_hi] (default 2.2–4.2 → drop z=4.4, 11 z-bins);
      * 1e-3 < k < 0.5π/R_z(z)  (z-dependent half-Nyquist high cut, continuum-floor low cut).
    Covariance: the full 1020×1020 ``cov`` (= STAT + SYST), sub-selected to the kept rows;
    ``cov_diag_inflation`` is ADDED to the diagonal (variance units) to reach χ²_ν∼1.

    metals_on=True (DESI forward-models SiIII/SiII per the usage doc); resolution_on=False
    by default (the residual resolution mode stays in C — usage doc option (a); the template
    knob is offered but OFF). The data is DECONVOLVED → compare theory directly (no window).
    """
    d = np.load(npz_path, allow_pickle=True)
    z = np.asarray(d["z"], float)              # (1020,) z-major
    k = np.asarray(d["k"], float)              # (1020,) angular k
    P = np.asarray(d["plya"], float)
    cov = np.asarray(d["cov"], float).copy()   # full STAT+SYST
    if add_cov_diag_inflation:
        cov[np.diag_indices_from(cov)] += np.asarray(d["cov_diag_inflation"], float)

    # z-dependent k cut: k < 0.5π/R_z(z) with R_z from the DESI resolution proxy.
    R_row = desi_resolution_R(z)
    k_hi_row = 0.5 * np.pi / R_row
    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k > k_min) & (k < k_hi_row)

    return _assemble_leg("DESI", z, k, P, cov, keep,
                         R_func=desi_resolution_R, metals_on=metals_on,
                         resolution_on=resolution_on)


def load_ks_leg(base="/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/",
                *, z_lo=2.0, z_hi=4.6, drop_first4=True, k_max=CACHE_KMAX,
                metals_on=False, resolution_on=False):
    """Load KODIAQ-SQUAD conservative-mode P1D → a post-cut ``DataLeg``.

    Format: pipe-separated ``final-conservative-p1d-karacayli_etal2021.txt`` (z|k|P|e) +
    the 182×182 ``final-conservative-covariance-karacayli_etal2021.txt`` (z-major,
    z∈[2.0,4.6], 13 k-bins/z). Cuts: drop the first 4 k-bins (k ≤ 0.0158; Karaçaylı
    2306.06316 Fig 11 underestimate the error there) + cut k ≤ k_max=0.069 (the emulator
    Nyquist; the analysis caps k<0.06 — note KS is HR but we cap at the cache k_max).

    metals_on=False / resolution_on=False by default: KS conservative mode already SUBTRACTS
    metals/continuum/resolution + inflates its covariance, so re-applying the SiIII/resolution
    MODEL terms would double-count.  LYA-CONSULT: confirm KS carries NO model systematics.
    """
    p1d_file = base.rstrip("/") + "/final-conservative-p1d-karacayli_etal2021.txt"
    cov_file = base.rstrip("/") + "/final-conservative-covariance-karacayli_etal2021.txt"
    z, k, P = _read_ks_p1d(p1d_file)            # z-major (182,)
    cov = np.loadtxt(cov_file)                  # (182,182) z-major

    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k <= k_max + 1e-9)
    if drop_first4:
        keep &= (k > KS_DROP_KMAX)

    # KS has no resolution proxy in this file; reuse the DESI-style proxy as a placeholder
    # for the (default-OFF) resolution knob.  LYA-CONSULT: KS resolution is OFF by default
    # (conservative mode already deconvolves + inflates), so R_z is unused unless toggled on.
    return _assemble_leg("KS", z, k, P, cov, keep,
                         R_func=desi_resolution_R, metals_on=metals_on,
                         resolution_on=resolution_on)


def _read_ks_p1d(path):
    """Parse the pipe-separated KS conservative P1D table (z|k|P|e), skipping the header.
    Returns (z, k, P) as float arrays in the file's z-major row order."""
    z, k, P = [], [], []
    with open(path) as f:
        for ln in f.readlines()[1:]:           # drop the header row
            parts = ln.split("|")
            if len(parts) < 4:
                continue
            try:
                zz, kk, pp = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            z.append(zz); k.append(kk); P.append(pp)
    return np.array(z), np.array(k), np.array(P)


def _assemble_leg(name, z_all, k_all, P_all, cov_all, keep, *, R_func,
                  metals_on, resolution_on):
    """Sub-select the kept (z,k) rows + their covariance block, build the z-major flat
    DataLeg.  The covariance is row/col-sliced by the SAME boolean mask as the data so the
    flat-row ordering matches C_data exactly (CS-REVIEW: ordering invariant)."""
    idx = np.where(keep)[0]
    z_row = z_all[idx]
    k = k_all[idx]
    P_data = P_all[idx]
    C_data = cov_all[np.ix_(idx, idx)]

    z = np.unique(np.round(z_row, 6))          # ascending unique z
    z = np.array([z_row[np.argmin(np.abs(z_row - zz))] for zz in z])  # exact stored values
    z_idx = np.searchsorted(z, z_row)
    # guard floating round-off in searchsorted by snapping to the nearest unique z
    z_idx = np.array([int(np.argmin(np.abs(z - zr))) for zr in z_row])

    n_per_z = np.array([int(np.sum(z_idx == i)) for i in range(len(z))])
    R_z = np.asarray(R_func(z))
    return DataLeg(
        name=name, z=z, z_unit=_z_unit(z), k=k, z_row=z_row, z_idx=z_idx,
        P_data=P_data, C_data=C_data, R_z=R_z, n_z=len(z), n_per_z=n_per_z,
        metals_on=metals_on, resolution_on=resolution_on)


# ============================================================================ #
#  Forward-model nuisances (differentiable; per-leg-configurable, default OFF)
# ============================================================================ #
def _metal_factor(k, *, a_SiIII=0.0, a_SiII=0.0):
    """McDonald-form metal contamination multiplier (usage doc Eq. 4.3):

        P → P · (1 + f_metals(k)·exp(−k²/2 k_s²)),
        f_metals = (a²_SiIII + 2 a_SiIII cos(k·Δv_SiIII))
                 + (a²_SiII  + 2 a_SiII  cos(k·Δv_SiII)),

    with the Gaussian damping k_s = METAL_KS, and Δv the Lyα→line velocity split
    Δv = c·ln(λ_Lyα/λ_line).  a_SiIII=a_SiII=0 ⇒ factor ≡ 1 (no metals).  Differentiable in
    (a_SiIII, a_SiII).  LYA-CONSULT: the (1+f)·exp(−k²/2k_s²) placement of the damping (whole
    f vs only the cosine) follows the usage-doc one-liner; confirm vs companion Eq. 4.3."""
    k = jnp.asarray(k)
    dv_SiIII = C_KMS * jnp.log(LAMBDA_LYA / LAMBDA_SiIII)
    dv_SiII = C_KMS * jnp.log(LAMBDA_LYA / LAMBDA_SiII)
    damp = jnp.exp(-(k ** 2) / (2.0 * METAL_KS ** 2))
    f = (a_SiIII ** 2 + 2.0 * a_SiIII * jnp.cos(k * dv_SiIII)) \
        + (a_SiII ** 2 + 2.0 * a_SiII * jnp.cos(k * dv_SiII))
    return 1.0 + f * damp


def _resolution_factor(k, R_z, *, b_res=0.0):
    """Resolution-template multiplier P → P·exp(2 b_res k² R_z²) (usage doc Eq. 4.8, option
    (b)). b_res=0 ⇒ factor ≡ 1.  Differentiable in b_res.  Default OFF (the residual
    resolution mode stays in C unless this knob is turned on per-leg)."""
    return jnp.exp(2.0 * b_res * jnp.asarray(k) ** 2 * jnp.asarray(R_z) ** 2)


# ============================================================================ #
#  Model → leg binding
# ============================================================================ #
def _emu_var_on_cache(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                      pf_stats, sigma_zb, alpha_centres, dla_core, cemu_inflate):
    """Per-z emulator-error variance on the CACHE k-grid (K_cache,), assembled EXACTLY as
    ``inference.predict_P_obs_and_cov_single_z`` does (the diagonal σ path):
      emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c²,  coef = [1−Σα, α_LLS, α_subDLA, α_DLA].
    NaN-safe (out-of-range cache cells σ→0).  Differentiable in (θ9, τ₀, α)."""
    from .predict import predict_P_filt
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)        # (4,Kc)
    P_clean = P_filt[0]
    P_dla_unf = P_filt[3] + jnp.asarray(dla_core)
    P_cls = jnp.stack([P_clean, P_filt[1], P_filt[2], P_dla_unf])         # (4,Kc)
    a = jnp.asarray(alpha_hcd)
    coef = jnp.concatenate([jnp.atleast_1d(1.0 - jnp.sum(a)), a])         # (4,)
    sigma_ck = sigma_at_tau0(sigma_zb, alpha_centres, z, tau0)            # (4,Kc) fractional
    emu_var = jnp.einsum("c,ck,ck->k", coef ** 2,
                         jnp.nan_to_num(sigma_ck) ** 2, P_cls ** 2)
    return emu_var * cemu_inflate


def predict_P_obs_on_leg(model, theta9, tau0_vec, alpha_hcd, *, pf_stats, dla_core,
                         cache_k, leg, sigma_zb=None, alpha_centres=None,
                         cemu_inflate=1.0, a_SiIII=0.0, a_SiII=0.0, b_res=0.0,
                         rho_zb=None):
    """Bind the emulator forward model to ONE leg's grid → flat (P_model (N,), C_total (N,N)).

    For each z in ``leg.z``:
      1. predict P_obs on the CACHE k-grid (``predict.predict_P_obs``; z_unit=(z−2)/3.4,
         τ₀ = the per-z entry of ``tau0_vec``);
      2. apply the forward-model nuisances (metals, resolution) — per-leg-gated by
         ``leg.metals_on`` / ``leg.resolution_on`` (default OFF);
      3. ``jnp.interp`` P_obs from ``cache_k`` onto this z's leg-k subset (differentiable; the
         emulator P_obs(k) is smooth → linear interp to bin-CENTRE; see CS-REVIEW);
      4. interp the per-k emu variance (``_emu_var_on_cache``) onto the leg k → diag(C_emu).

    Returns the FLAT (z-major) (P_model, C_total) with C_total = C_data + diag(C_emu).
    ``sigma_zb``/``alpha_centres`` may be None ⇒ C_emu = 0 (a pure data-covariance fit, e.g.
    the mock-noise generator supplies its own C).  Differentiable in
    (θ9, τ₀_vec, α, a_SiIII, a_SiII, b_res).

    NOTE per-z C_emu blocks are placed on the FLAT diagonal in the SAME z-major order as the
    leg's rows (``leg.z_idx``), so C_emu's ordering matches C_data (CS-REVIEW)."""
    cache_k = jnp.asarray(cache_k)
    k_leg = jnp.asarray(leg.k)
    z_idx = np.asarray(leg.z_idx)
    R_z = jnp.asarray(leg.R_z)
    tau0_vec = jnp.asarray(tau0_vec)
    N = k_leg.shape[0]

    P_model = jnp.zeros(N)
    emu_var_flat = jnp.zeros(N)
    have_emu = sigma_zb is not None and alpha_centres is not None

    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z = float(leg.z[iz])
        z_unit = float(leg.z_unit[iz])
        tau0 = tau0_vec[iz]
        k_sub = k_leg[jnp.asarray(rows)]

        # (1) P_obs on the cache grid
        P_cache = predict_P_obs(model, theta9, z_unit, tau0, alpha_hcd, pf_stats, dla_core)
        # (3) interp to the leg's k (bin centres); model is smooth → linear interp.
        P_z = jnp.interp(k_sub, cache_k, P_cache)
        # (2) forward-model nuisances (gated; default OFF → factor ≡ 1)
        if leg.metals_on:
            P_z = P_z * _metal_factor(k_sub, a_SiIII=a_SiIII, a_SiII=a_SiII)
        if leg.resolution_on:
            P_z = P_z * _resolution_factor(k_sub, R_z[iz], b_res=b_res)
        P_model = P_model.at[jnp.asarray(rows)].set(P_z)

        # (4) C_emu: per-k emu variance interp'd onto the leg k
        if have_emu:
            ev_cache = _emu_var_on_cache(
                model, theta9, z_unit, z, tau0, alpha_hcd, pf_stats=pf_stats,
                sigma_zb=sigma_zb[iz], alpha_centres=alpha_centres, dla_core=dla_core,
                cemu_inflate=cemu_inflate)
            ev_z = jnp.interp(k_sub, cache_k, ev_cache)
            # emu var transforms by the SAME multiplicative nuisance factors² (variance units)
            if leg.metals_on:
                ev_z = ev_z * _metal_factor(k_sub, a_SiIII=a_SiIII, a_SiII=a_SiII) ** 2
            if leg.resolution_on:
                ev_z = ev_z * _resolution_factor(k_sub, R_z[iz], b_res=b_res) ** 2
            emu_var_flat = emu_var_flat.at[jnp.asarray(rows)].set(ev_z)

    C_total = jnp.asarray(leg.C_data) + jnp.diag(emu_var_flat)
    return P_model, C_total


# ============================================================================ #
#  Multi-leg likelihood
# ============================================================================ #
def data_loglik(model, theta9, tau0_global, alpha_hcd, legs, *, pf_stats, dla_core,
                cache_k, z_global=None, sigma_zb_per_leg=None, alpha_centres=None,
                cemu_inflate=1.0, a_SiIII=0.0, a_SiII=0.0, b_res=0.0,
                jitter=1e-10, return_parts=False):
    """Multi-leg Gaussian log-likelihood against the REAL data.

    The legs are INDEPENDENT surveys (DESI & KS share z-VALUES but are different
    instruments → no cross-covariance) ⇒ the joint covariance is BLOCK-DIAGONAL ⇒
        logL = Σ_leg gaussian_loglik(P_data_leg − P_model_leg, C_total_leg).

    τ₀ is supplied on a GLOBAL z-grid (``z_global``, ascending); each leg pulls the τ₀ for
    its own z-bins by nearest-z match (DESI z⊂global, KS z⊂global).  ``sigma_zb_per_leg`` is
    a dict {leg.name: (n_z_leg,4,Kc,Tb)} of the emulator error vector ALREADY sliced to that
    leg's z-bins (or None for a data-cov-only fit).

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

        P_model, C_total = predict_P_obs_on_leg(
            model, theta9, tau0_vec, alpha_hcd, pf_stats=pf_stats, dla_core=dla_core,
            cache_k=cache_k, leg=leg, sigma_zb=szb, alpha_centres=alpha_centres,
            cemu_inflate=cemu_inflate, a_SiIII=a_SiIII, a_SiII=a_SiII, b_res=b_res)
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


# ============================================================================ #
#  Smoke driver:  python -m hcd_analysis.emulator.data_likelihood --smoke
# ============================================================================ #
def _smoke():
    """Assemble BOTH legs, evaluate data_loglik at a fiducial θ on a synthetic-shaped
    emulator, print logL + per-leg χ²/dof + the binding shapes. No checkpoint required."""
    import jax.random as jr
    from .model import Emulator

    n_k, n_basis, n_tb = 172, 12, 4
    rng = np.random.default_rng(0)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jr.PRNGKey(0))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))
    theta9 = jnp.full(9, 0.5)
    alpha_hcd = jnp.asarray([0.06, 0.02, 0.003])
    alpha_centres = jnp.asarray([0.66, 0.83, 1.15, 1.33])

    desi = load_desi_leg(metals_on=True, resolution_on=True)
    ks = load_ks_leg()
    legs = [desi, ks]
    for leg in legs:
        print(f"[{leg.name}] n_z={leg.n_z} N_rows={leg.k.shape[0]} "
              f"z=[{leg.z.min():.2f},{leg.z.max():.2f}] "
              f"k=[{leg.k.min():.4g},{leg.k.max():.4g}] "
              f"C_data {leg.C_data.shape} metals_on={leg.metals_on} "
              f"resolution_on={leg.resolution_on}")

    z_global = np.unique(np.round(np.concatenate([desi.z, ks.z]), 6))
    from .meanflux_prior import becker13_tau0
    tau0_global = jnp.asarray(becker13_tau0(jnp.asarray(z_global)))
    szb = {leg.name: jnp.asarray(rng.uniform(0.01, 0.05, (leg.n_z, 4, n_k, n_tb)))
           for leg in legs}

    total, parts = data_loglik(
        model, theta9, tau0_global, alpha_hcd, legs, pf_stats=pf, dla_core=dla_core,
        cache_k=cache_k, z_global=z_global, sigma_zb_per_leg=szb,
        alpha_centres=alpha_centres, a_SiIII=0.04, b_res=0.005, return_parts=True)
    print(f"\njoint logL = {float(total):.4g}")
    for name, (ll, chi2, dof) in parts.items():
        print(f"  [{name}] logL={ll:.4g}  chi2={chi2:.4g}  dof={dof}  "
              f"chi2/dof={chi2 / dof:.3f}")

    # gradient smoke (the inference-critical property)
    def f(th, t, a, m, b):
        return data_loglik(model, th, t, a, legs, pf_stats=pf, dla_core=dla_core,
                           cache_k=cache_k, z_global=z_global, sigma_zb_per_leg=szb,
                           alpha_centres=alpha_centres, a_SiIII=m, b_res=b)
    g = jax.grad(f, argnums=(0, 1, 2, 3, 4))(theta9, tau0_global, alpha_hcd, 0.04, 0.005)
    finite = all(np.isfinite(np.asarray(gi)).all() for gi in g)
    print(f"\ngrad(θ9,τ₀,α,a_SiIII,b_res) all finite: {finite}")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print("usage: python -m hcd_analysis.emulator.data_likelihood --smoke")
