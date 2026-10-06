"""Phase-C data-binding layer — the REAL DESI DR1 + KODIAQ-SQUAD P1D likelihood.

This is the path the Leg-B coverage gate and the eventual production fit use; it is a
NEW, parallel module to the Leg-A cache-grid driver (``inference.py`` /
``closure_sbc``), which it leaves untouched. The Leg-A driver evaluates the likelihood
on the cache's own 172 angular-k grid (the sim/closure path); THIS module binds the
emulator forward model to the OBSERVED survey grids (DESI 85 angular-k × 12 z, KS 13
angular-k × 14 z), applying the published cuts + covariances.

Contract (per ``hcd_priya_notes/docs/superpowers/2026-06-05-desi-dr1-p1d-usage.md``):

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

from .data import KIM_AMP, KIM_SLOPE, Z_LIMITS
from ..paths import REPO_ROOT_STR as _REPO
from .predict import predict_P_obs, predict_P_filt, _excess_from_P_filt
from .likelihood import sigma_at_tau0, rho_at_tau0, gaussian_loglik

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"

# --- physical constants -------------------------------------------------------
# res_corr AMPLITUDE nuisance pivot (Task 1.3): α(z) = α₀·((1+z)/(1+Z_PIVOT))^s. The
# multi-fidelity res_corr (HF→n512 particle-convergence factor) amplitude is marginalized
# FORWARD-ONLY at the chokepoint ``_mf_corr_on_cache`` via this z-slope around z=3.
Z_PIVOT = 3.0
C_KMS = 299792.458            # speed of light [km/s]
LAMBDA_LYA = 1215.67          # Lyα rest wavelength [Å]
LAMBDA_SiIII = 1206.50        # SiIII line [Å]
LAMBDA_SiII = 1190.42         # SiII line [Å]  (1190/1193 doublet; 1190.42 leading line)
LAMBDA_SiIIb = 1193.28        # SiII line [Å]  (the SECOND doublet line; r_doublet weights it)
R_SiII_DOUBLET = 0.5          # intra-doublet ratio (matches closure_legb.metal_inject r_doublet)
# SiIII/SiII–Lyα decorrelation scale k_x [s/km] (DESI DR1 companion arXiv:2601.21432 Eq. 4.3).
# NOTE: this sigmoid damping is the companion's ADDITION, NOT in McDonald 2006 (whose SiIII
# cross-term is undamped). The cosine
# cross-term is multiplied by the SIGMOID D_x(k)=2−2/(1+exp(−k/k_x)); k_x is a FREE nuisance
# the sampler fits. These are off-state defaults (irrelevant when a=0; metals default OFF).
K_SiIII_DEFAULT = 0.05         # s/km, sigmoid decorrelation scale (free nuisance)
K_SiII_DEFAULT = 0.05
# DESI spectral pixel width used for the resolution proxy R_z [Å] (usage doc Eq. 4.8).
DESI_PIXEL_ANGSTROM = 0.8

# KS keeps its FULL native k-range from klow≈0.0055 s/km. The Karaçaylı 2306.06316 Fig-11
# "first-4-bins error underestimate" caution is MISLEADING (PI/KS-author decision, reaffirmed
# repeatedly) — those low-k bins are ALWAYS kept. There is NO low-k KS drop knob.
# DESI continuum-floor low-k cut + half-Nyquist resolution high-k cut (usage doc §"cuts").
DESI_KMIN = 1e-3

# ============================================================================ #
#  COVARIANCE-CORRECTNESS FIXES (2026-06-18, low-k n_s reliability arc).
#  Both WIDEN the low-k covariance to match the published DESI DR1 cosmology
#  analysis (Chaves-Montero arXiv:2601.21432) + the eBOSS PRIYA analysis
#  (Fernandez+2024 arXiv:2309.03943). Neither de-biases the forward; they soften
#  the +5.5σ low-k n_s pull computed against the currently-deployed covariance.
#  Both are ENV-GATED and REVERSIBLE (unset → byte-identical to the deployed path).
#  See hcd_priya_notes/docs/superpowers/2026-06-18-{desi-p1d-lowk-data-reliability,
#  cosmic-variance-floor-lowk}.md (notes repo).
# ----------------------------------------------------------------------------#
#  Fix 1 — DESI SNR>3 measurement + covariance (the cosmology-paper baseline).
#  The Chaves-Montero DR1 cosmology fit uses the SNR>3 subsample (62,807 QSO) with
#  its OWN full 1020×1020 covariance + a +5% STAT-uncertainty inflation at all (k,z)
#  ("possible percent-level large-scale biases", CCD image sims). Our default loads
#  the SNR>1 baseline. Set HCD_DESI_SNR3=1 to swap to the SNR>3 npz; the +5% stat
#  inflation rides along with it (on the SEPARATE cov_stat block: cov += (1.05²−1)·cov_stat).
DESI_SNR3_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d_snr3.npz"
DESI_SNR3_STAT_INFLATE = 1.05         # +5% on the STAT uncertainty (paper baseline)


def _env_flag(name):
    """True iff the env var ``name`` is set to a truthy token (1/true/yes/on)."""
    import os
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# The env-resolved DATA-SELECTION flags (adversarial backfill F2, 2026-07-19): each silently
# changes WHICH measurement/covariance a leg loads (SNR>3 npz swap / CV floor / its rank-1 form)
# on every load. They are now (a) STAMPED on the DataLeg (use_snr3 / cv_floor_on / cv_floor_rank1)
# and threaded into meta["forward"] by closure_legb.forward_stamp, and (b) REFUSED at production
# driver entry (assert_env_data_flags_unset below) unless --allow-env-data-flags is explicit.
ENV_DATA_FLAGS = ("HCD_DESI_SNR3", "HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1")


def assert_env_data_flags_unset(where, allow=False):
    """DRIVER-ENTRY TRIPWIRE (F2): fail LOUD if any env data-selection flag is set (truthy)
    at production driver entry. A stray exported flag would otherwise silently swap the DESI
    measurement or inflate the covariance under the deployed likelihood — invisible to
    forward_signature and every pre-F2 pkl stamp. ``allow=True`` (the driver's explicit
    --allow-env-data-flags) is the only sanctioned override; falsy tokens ('0'/'false') are
    UNSET per _env_flag and do not trip."""
    if allow:
        return
    import os
    lit = {n: os.environ.get(n) for n in ENV_DATA_FLAGS if _env_flag(n)}
    assert not lit, (
        f"env data-selection flag(s) set at driver entry [{where}]: {lit}. These flags swap the "
        f"loaded measurement/covariance (SNR>3 npz / CV floor) silently under the production "
        f"likelihood. Unset them, or pass --allow-env-data-flags to run a deliberate "
        f"non-baseline arm (the resolved values are stamped on the DataLeg + meta['forward']).")


# ----------------------------------------------------------------------------#
#  Fix 2 — restore the ~2% finite-box cosmic-variance (σ_CV) floor at low k.
#  Fernandez+2024 (eBOSS PRIYA, Eq 3.1) carries K = K_BOSS + σ_GP σ_GPᵀ + σ_CV σ_CVᵀ;
#  σ_CV ≈ 2% of P, significant ONLY at k<2.5e-3 s/km (the finite 120 Mpc/h box),
#  negligible at high k. Our deployed C_total drops it. Set HCD_CV_FLOOR=1 to ADD it
#  back as an additive term scaled by the data power: var += (f_CV(k)·P_data)².
#  f_CV(k) = CV_FLOOR_FRAC for k ≤ CV_FLOOR_K_FULL, tapering linearly to 0 at
#  CV_FLOOR_K_ZERO (so it is a smooth low-k-only floor, off above ~3e-3). DIAGONAL by
#  default (HCD_CV_FLOOR_RANK1=1 makes it Fernandez's fully-correlated rank-1 σ_CV σ_CVᵀ).
CV_FLOOR_FRAC = 0.02                  # 2% of P (Fernandez/PRIYA large-scale CV level)
CV_FLOOR_K_FULL = 2.5e-3              # full amplitude at k ≤ this (s/km)
CV_FLOOR_K_ZERO = 3.0e-3              # tapered to 0 by this k (negligible above)


def _cv_floor_frac(k):
    """The fractional σ_CV(k): CV_FLOOR_FRAC at k≤K_FULL, linearly →0 at K_ZERO, 0 above."""
    k = np.asarray(k, float)
    t = (CV_FLOOR_K_ZERO - k) / (CV_FLOOR_K_ZERO - CV_FLOOR_K_FULL)   # 1 at K_FULL, 0 at K_ZERO
    return CV_FLOOR_FRAC * np.clip(t, 0.0, 1.0)


def _add_cv_floor(C_data, k, P_data, *, rank1=False):
    """Add the σ_CV low-k floor to a leg's data covariance (Fix 2). Returns a NEW array.
    DIAGONAL: C += diag((f_CV·P)²).  RANK1 (Fernandez Eq 3.1 form): C += s sᵀ, s=f_CV·P
    (fully correlated across the low-k rows). Pure numpy, host-side (C_data is fixed)."""
    s = _cv_floor_frac(k) * np.asarray(P_data, float)        # (N,) absolute σ_CV per row
    C = np.asarray(C_data, float).copy()
    if rank1:
        C += np.outer(s, s)
    else:
        C[np.diag_indices_from(C)] += s ** 2
    return C


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
      mf_floor_on : bool    whether the MF C_emu floor (LF→HR generalization, §2.3 of the
                            floor spec) is added on this leg's C_emu when ``mf`` is enabled.
                            TRUE on the SMALL-SCALE leg(s) only (KS, and high-k DESI rows
                            above the LF Nyquist); the low-k DESI leg stays FALSE (it is
                            below the resolution regime and must not be inflated). Default
                            FALSE → back-compatible (no floor on the DESI leg).
      dla_forward_frac : float  the PER-LEG scale on the sampled α_DLA's DLA-excess contribution
                            in the forward (§0c, PI-confirmed final intent 2026-06-09). The
                            DLA-finder masking is leg-specific: KS fully masks DLAs (0% residual,
                            ``KS_DLA_FORWARD_FRAC=0.0`` → the KS forward DLA term is 0, matching
                            the 0% KS closure target); DESI carries the ~10% unmasked-DLA
                            residual the forward MARGINALIZES α_DLA over (``DESI_DLA_FORWARD_FRAC
                            =1.0`` → the full sampled α_DLA). The forward DLA-excess term becomes
                            ``dla_forward_frac · α_DLA · (P_DLA_unf − P_clean)``. Default 1.0
                            (back-compat: the full DLA forward, byte-identical for the DESI leg).
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
    mf_floor_on: bool = False
    dla_forward_frac: float = 1.0
    # whether this leg's R_z is trustworthy for a spectral-resolution INJECTION (option-b / the Gate-B
    # resolution arm). DESI/eBOSS: True (R_z exact / order-correct proxy). KS: False -- load_ks_leg reuses
    # the DESI pixel proxy R_z, ~7-15x too large vs KS's echelle sigma~3.2 km/s, so a b_res injection is a
    # ~70% distortion the forward cannot fit (the -21sigma ESS collapse). Gate the injection on THIS flag,
    # NOT resolution_on (option-a injects with resolution_on=False). Default True (back-compatible).
    resolution_ready: bool = True
    # ARM-D bookkeeping: the covariance carries a COHERENT cross-z resolution mode (resolution removed the
    # deployed way, then re-added as ONE rank-1 outer(e,e)) and there is NO forward f_res float. Distinguishes
    # arm-D (resolution_on=False, resolution_coherent_on=True) from option-a (both False). Default False.
    resolution_coherent_on: bool = False
    # Reduced-covariance stamp (PI disposition 2026-07-17): True iff the DLA-completeness systematic
    # (syst_e_dla_completeness) was REMOVED from C_data because the alpha_DLA mean model floats on this
    # leg (modeled-in-mean => removed-from-covariance, the cup1d "red" convention). DESI-only; audit +
    # runner-assert hook so a driver can fail loud if it gets the un-reduced covariance.
    dla_cov_reduced: bool = False
    # ENV-FLAG DATA-SELECTION STAMPS (adversarial backfill F2, 2026-07-19; the dla_cov_reduced
    # pattern): the RESOLVED values of the env-gated loaders' data-selection knobs, so a leg loaded
    # under a stray HCD_DESI_SNR3 / HCD_CV_FLOOR (/RANK1) export is visible to audits and pkl
    # stamps (closure_legb.forward_stamp threads them into meta["forward"]). cv_floor_rank1 records
    # what was APPLIED (False whenever the floor itself is off, even if the RANK1 env var is set).
    use_snr3: bool = False
    cv_floor_on: bool = False
    cv_floor_rank1: bool = False


# PER-LEG DLA-forward fraction (§0c, PI-confirmed final intent 2026-06-09): the leg-specific
# scale on the sampled α_DLA's DLA-excess contribution. DESI carries the ~10% unmasked-DLA
# residual (the finder misses ~10% → full systems remain) so the forward keeps the FULL sampled
# α_DLA; KS fully masks DLAs so the forward DLA term is ZERO (matching the 0% KS closure target).
DESI_DLA_FORWARD_FRAC = 1.0
KS_DLA_FORWARD_FRAC = 0.0
# eBOSS DR14 (Chabanier+2019): DLAs are MASKED (the Pk1D_syst.dat carries DLAmask +
# DLAcompleteness residual in C_data), so the forward DLA-excess term is ZERO (like KS, not DESI).
EBOSS_DLA_FORWARD_FRAC = 0.0

# THE SINGLE AUTHORITY for the DESI reduced-covariance decision (PI disposition 2026-07-17):
# modeled-in-mean => removed-from-covariance, cup1d's type_analysis="red" convention (DESI DR1
# cosmology paper 2601.21432 Sec 2.1: the residual-HCD and resolution cov_syst terms are OMITTED
# "as both effects are explicitly marginalized over"). DESI floats the PRIYA alpha_DLA mean model
# (DESI_DLA_FORWARD_FRAC=1.0 above), so keeping syst_e_dla_completeness in C_data double-prices
# that named unknown on the variance side (~10% whitened overlap, conservative -- the 2026-07-17
# cup1d cross-check). True => load_desi_leg removes it by default using the SHIPPED per-z-block
# rank-1 convention (cov_syst is exactly z-block-diagonal): C -= sum_z outer(e_dla|z). Lives HERE
# next to DESI_DLA_FORWARD_FRAC (not per-driver) so every DESI leg load -- real fit, SBC, dnuis
# arms, diagnostics -- inherits the same decision with no threading (the NORC near-miss class);
# folded into closure_legb.forward_signature() for the freeze. Opt out per-call with
# dla_cov_reduce=False (archival reproduction of pre-2026-07-17 results ONLY). KS/eBOSS are
# untouched: their forward DLA term is ZERO, so the modeled-in-mean licence does not apply.
DESI_DLA_COV_REDUCE = True


def _z_unit(z):
    """(z − 2.0)/3.4 — the cache encoder's z_unit (data.Z_LIMITS = (2.0, 5.4))."""
    return (np.asarray(z) - 2.0) / 3.4


def desi_resolution_R(z):
    """DESI resolution scale R_z = c·0.8Å / ((1+z)·1215.67Å)  [km/s] (usage doc Eq. 4.8; CORRECTED
    2026-07-10, PR#14 panel FIX 6b -- an earlier version of this docstring mislabeled the units
    [s/km], the units of k not R_z; C_KMS carries [km/s] and the Å/Å ratio is dimensionless, so R_z
    is [km/s], consistent with ks_resolution_R's KS_RESOLUTION_KMS=3.2 km/s)."""
    z = np.asarray(z)
    return C_KMS * DESI_PIXEL_ANGSTROM / ((1.0 + z) * LAMBDA_LYA)


# KODIAQ+SQUAD echelle spectral-resolution scale, PINNED to the Gaussian sigma_v of the LSF:
# sigma_v = c / (R * 2.3548) (FWHM=c/R). NOTE (CORRECTED 2026-07-10, PR#14 panel FIX 6c): 3.2 km/s
# is the SQUAD (higher-R=40000) floor; the KODIAQ end (R=36000) gives sigma_v ~3.54 km/s (LOWER
# resolving power -> LARGER LSF). The deployed 3.2 km/s is therefore the OPTIMISTIC (narrowest-LSF)
# end of the KODIAQ+SQUAD range, not a KODIAQ+SQUAD-average -- documented here, value UNCHANGED.
# This REPLACES the DESI pixel proxy (~49 km/s at z=3, ~15x too large) on the KS f_res path ONLY; the
# default KS load keeps the proxy (R_z is unused when resolution_on=False). z-independent (echelle R is).
KS_RESOLUTION_KMS = 3.2


def ks_resolution_R(z):
    """KS spectral-resolution scale R_z, pinned to the KODIAQ+SQUAD echelle sigma_v = 3.2 km/s
    (z-independent). Used only on the KS f_res (option-b) path; see KS_RESOLUTION_KMS."""
    z = np.asarray(z)
    return np.full(np.shape(z), float(KS_RESOLUTION_KMS))


# ============================================================================ #
#  Loaders
# ============================================================================ #
def load_desi_leg(npz_path="/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz",
                  *, z_lo=2.2, z_hi=4.2, k_min=DESI_KMIN, metals_on=True,
                  resolution_on=False, add_cov_diag_inflation=True, mf_floor_on=False,
                  use_snr3=None, snr3_stat_inflate=None, add_cv_floor=None, resolution_float=False,
                  resolution_coherent=False, resolution_coh_amp=1.0, dla_cov_reduce=None):
    """Load DESI DR1 P1D → a post-cut ``DataLeg`` (usage doc §"Covariance + cuts").

    Cuts (z-major flat layout, ``row_is_zmajor=True``):
      * z in [z_lo, z_hi] (default 2.2–4.2 → drop z=4.4, 11 z-bins);
      * 1e-3 < k < 0.5π/R_z(z)  (z-dependent half-Nyquist high cut, continuum-floor low cut).
    Covariance: the full 1020×1020 ``cov`` (= STAT + SYST), sub-selected to the kept rows;
    ``cov_diag_inflation`` is ADDED to the diagonal (variance units) to reach χ²_ν∼1.

    metals_on=True (DESI forward-models SiIII/SiII per the usage doc); resolution_on=False
    by default (the residual resolution mode stays in C — usage doc option (a); the template
    knob is offered but OFF). The data is DECONVOLVED → compare theory directly (no window).

    COVARIANCE-CORRECTNESS FIXES (2026-06-18, REVERSIBLE, env-gated; see the module
    constants block). All three default to ``None`` → read the corresponding env flag, so
    UNSET env ⇒ byte-identical to the deployed SNR>1 path (back-compat); an explicit
    True/False overrides the env for tests.
      * ``use_snr3``         (env ``HCD_DESI_SNR3``): load the SNR>3 npz (``DESI_SNR3_NPZ``,
        the Chaves-Montero cosmology baseline, 62,807 QSO) + its OWN covariance instead of
        the SNR>1 baseline. Same z/k grid, larger low-k errors. Auto-applies the +5% stat
        inflation below unless ``snr3_stat_inflate`` is set otherwise.
      * ``snr3_stat_inflate`` (default → ``DESI_SNR3_STAT_INFLATE``=1.05 when SNR>3 is on):
        inflate the STAT uncertainty by this factor (paper's +5% for large-scale biases):
        ``cov += (f²−1)·cov_stat``. Only applied when SNR>3 is active.
      * ``add_cv_floor``     (env ``HCD_CV_FLOOR``): add the Fernandez σ_CV ~2% finite-box
        floor to the low-k rows (``_add_cv_floor``). Diagonal unless ``HCD_CV_FLOOR_RANK1``.

    ``dla_cov_reduce`` (default None → the ``DESI_DLA_COV_REDUCE`` authority constant, True):
    REMOVE the DLA-completeness systematic (``syst_e_dla_completeness``) from C_data as per-z
    rank-1 blocks — the cup1d "red" reduced covariance, licensed because DESI floats the PRIYA
    alpha_DLA mean model. Stamped on the leg as ``dla_cov_reduced``. Pass False ONLY to reproduce
    pre-2026-07-17 archival results.
    """
    use_snr3 = _env_flag("HCD_DESI_SNR3") if use_snr3 is None else bool(use_snr3)
    dla_cov_reduce = DESI_DLA_COV_REDUCE if dla_cov_reduce is None else bool(dla_cov_reduce)
    add_cv_floor = _env_flag("HCD_CV_FLOOR") if add_cv_floor is None else bool(add_cv_floor)
    if use_snr3:
        # Fix 1: the cosmology-paper baseline (SNR>3 measurement + its own covariance).
        npz_path = DESI_SNR3_NPZ
    d = np.load(npz_path, allow_pickle=True)
    z = np.asarray(d["z"], float)              # (1020,) z-major
    k = np.asarray(d["k"], float)              # (1020,) angular k
    P = np.asarray(d["plya"], float)
    cov = np.asarray(d["cov"], float).copy()   # full STAT+SYST
    if use_snr3:
        # Fix 1: +5% STAT-uncertainty inflation (paper's large-scale-bias allowance), applied
        # on the SEPARATE stat block so the syst part is untouched: cov += (f²−1)·cov_stat.
        f = DESI_SNR3_STAT_INFLATE if snr3_stat_inflate is None else float(snr3_stat_inflate)
        cov = cov + (f ** 2 - 1.0) * np.asarray(d["cov_stat"], float)
    if add_cov_diag_inflation:
        cov[np.diag_indices_from(cov)] += np.asarray(d["cov_diag_inflation"], float)
    cv_floor_rank1 = _env_flag("HCD_CV_FLOOR_RANK1") if add_cv_floor else False
    if add_cv_floor:
        # Fix 2: the σ_CV ~2% finite-box floor on the low-k rows (full-grid; the k-taper
        # zeroes it above ~3e-3, and the post-cut sub-selection keeps only the kept rows).
        cov = _add_cv_floor(cov, k, P, rank1=cv_floor_rank1)

    # z-dependent k cut: k < 0.5π/R_z(z) with R_z from the DESI resolution proxy.
    R_row = desi_resolution_R(z)
    k_hi_row = 0.5 * np.pi / R_row
    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k > k_min) & (k < k_hi_row)

    # read the per-bin resolution error ONLY on the option-b / arm-D paths (golden default must not depend
    # on a key it never uses -- code-lens #7).
    res_e = np.asarray(d["syst_e_resolution"], float) if (resolution_float or resolution_coherent) else None
    # reduced covariance (dla_cov_reduce): read the DLA-completeness column from the SAME npz that
    # supplied cov (SNR>1 baseline or the SNR>3 file -- each ships its own systematics vectors).
    dla_e = np.asarray(d["syst_e_dla_completeness"], float) if dla_cov_reduce else None
    return _assemble_leg("DESI", z, k, P, cov, keep,
                         R_func=desi_resolution_R, metals_on=metals_on,
                         resolution_on=resolution_on, mf_floor_on=mf_floor_on,
                         dla_forward_frac=DESI_DLA_FORWARD_FRAC,
                         resolution_e=res_e,
                         resolution_float=resolution_float,
                         resolution_coherent=resolution_coherent, resolution_coh_amp=resolution_coh_amp,
                         dla_e=dla_e,
                         use_snr3=use_snr3, cv_floor_on=add_cv_floor,
                         cv_floor_rank1=cv_floor_rank1)


def _read_ks_resolution_e(detail_path, z_grid, k_grid):
    """The KS spectral-resolution 1-sigma column ``esyst_res_ks`` aligned to the FULL pre-cut
    (z_grid, k_grid) that ``_read_ks_p1d`` returns (182 rows). Source = the pipe-delimited
    ``detailed-p1d-results-karacayli_etal2021.txt`` (14 cols; ``esyst_res_ks`` = index 9). The detailed
    table is a SUPERSET grid (315 rows: z=1.8 + extra high-k), so a positional read is WRONG -- we merge
    by an EXACT (z,k) lookup and RAISE on any unmatched conservative row (the alignment tripwire; never a
    silent mis-map). This is the resolution variance removed from the conservative cov diagonal (diag mode)."""
    lut = {}
    with open(detail_path) as f:
        for ln in f.readlines()[1:]:                          # drop the header row
            parts = [p.strip() for p in ln.strip().strip("|").split("|")]
            if len(parts) < 14:
                continue
            try:
                zz, kk, ee = float(parts[0]), float(parts[1]), float(parts[9])
            except ValueError:
                continue
            if zz < 1.9:                                      # drop z=1.8 (not on the conservative grid)
                continue
            lut[(round(zz, 3), round(kk, 8))] = ee
    e = np.empty(len(z_grid), float)
    for i, (zz, kk) in enumerate(zip(np.asarray(z_grid), np.asarray(k_grid))):
        key = (round(float(zz), 3), round(float(kk), 8))
        val = lut.get(key)
        if val is None or not np.isfinite(val):
            raise KeyError(f"KS esyst_res_ks: no finite detailed-table row for (z={zz}, k={kk})")
        e[i] = val
    return e


def load_ks_leg(base="/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/",
                *, z_lo=2.4, z_hi=4.6, k_max=None,
                metals_on=False, resolution_on=False, mf_floor_on=True, resolution_float=False):
    """Load KODIAQ-SQUAD conservative-mode P1D → a post-cut ``DataLeg``.

    Format: pipe-separated ``final-conservative-p1d-karacayli_etal2021.txt`` (z|k|P|e) +
    the 182×182 ``final-conservative-covariance-karacayli_etal2021.txt`` (z-major,
    z∈[2.0,4.6], 13 k-bins/z). Cuts: keep the FULL native k-range from **klow=0.0055 s/km**
    (PI/KS-author decision 2026-06-09, reaffirmed repeatedly: the Karaçaylı 2306.06316 Fig-11
    "first-4-bins error underestimate" caution is MISLEADING — those low-k bins are ALWAYS kept,
    there is NO low-k KS drop) + cut k ≤ k_max, which must be given explicitly (production 0.065; the single-grid
    default 0.069 was retired at gate E: the forward refuses any bin outside the simulated modes).
    ``z_lo`` defaults to **2.4** (drops the z=2.0+2.2 KS bins, which carried ~86% of a −0.65σ
    coherent n_s closure bias; dropping z<2.4 removes it → +0.04σ). z=2.4 is the MINIMAL
    closure-clean cut; low-z KS P1D is compromised by DLA-finder incompleteness, and the
    published KODIAQ-SQUAD analysis uses the more conservative z<2.8 — set ``z_lo=2.8`` for that
    (opt-in). PI decision 2026-06-08 = 2.4 default.
    See hcd_priya_notes/docs/superpowers/plans/2026-06-08-ns-bias-rootcause-diagnostics-plan.md.

    metals_on=False / resolution_on=False by default: KS conservative mode already SUBTRACTS
    metals/continuum/resolution + inflates its covariance, so re-applying the SiIII/resolution
    MODEL terms would double-count.  LYA-CONSULT: confirm KS carries NO model systematics.
    """
    p1d_file = base.rstrip("/") + "/final-conservative-p1d-karacayli_etal2021.txt"
    cov_file = base.rstrip("/") + "/final-conservative-covariance-karacayli_etal2021.txt"
    z, k, P = _read_ks_p1d(p1d_file)            # z-major (182,)
    cov = np.loadtxt(cov_file)                  # (182,182) z-major

    if k_max is None:
        raise ValueError("load_ks_leg: k_max must be explicit (production 0.065); the single-grid default was retired")
    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k <= k_max + 1e-9)

    # DEFAULT (resolution_float=False): KS has no echelle R_z in this file, so the DESI pixel proxy is a
    # placeholder for the (OFF) resolution knob (R_z unused when resolution_on=False), and resolution_ready
    # stays False -- a resolution INJECTION against the ~15x-too-large proxy is un-fittable (the -21sigma
    # collapse); the injection guard reads this flag. BYTE-IDENTICAL to the historical KS leg.
    #
    # resolution_float=True (task #5 V1, option-b): give KS its OWN echelle R_z (ks_resolution_R = 3.2 km/s)
    # and REMOVE its resolution systematic from the conservative cov via the "diag" mode (diag -= esyst_res_ks^2;
    # KS adds systematics to the DIAGONAL only), then FLOAT f_res in the forward. resolution_ready -> True.
    if resolution_float:
        res_e = _read_ks_resolution_e(base.rstrip("/") + "/detailed-p1d-results-karacayli_etal2021.txt", z, k)
        return _assemble_leg("KS", z, k, P, cov, keep,
                             R_func=ks_resolution_R, metals_on=metals_on,
                             resolution_on=resolution_on, mf_floor_on=mf_floor_on,
                             dla_forward_frac=KS_DLA_FORWARD_FRAC,
                             resolution_e=res_e, resolution_float=True, resolution_mode="diag",
                             resolution_ready=True)
    return _assemble_leg("KS", z, k, P, cov, keep,
                         R_func=desi_resolution_R, metals_on=metals_on,
                         resolution_on=resolution_on, mf_floor_on=mf_floor_on,
                         dla_forward_frac=KS_DLA_FORWARD_FRAC, resolution_ready=False)


def load_eboss_leg(npz_path="/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz",
                   *, z_lo=2.2, z_hi=4.6, k_min=0.0, k_max=None,
                   metals_on=True, resolution_on=False, mf_floor_on=False,
                   dla_forward_frac=EBOSS_DLA_FORWARD_FRAC, add_cv_floor=None, resolution_float=False,
                   resolution_coherent=False, resolution_coh_amp=1.0):
    """Load eBOSS DR14 P1D (Chabanier+2019, 1812.03554) → a post-cut ``DataLeg`` (block-diag cov).

    Format: the npz from ``scripts/convert_eboss_dr14_p1d.py`` (z, k, plya, sigma, cov, syst_*).
    13 z∈[2.2,4.6] × 35 k∈[0.001084,0.019512] s/km, z-MAJOR; k ANGULAR (no 2π, the cache
    convention). The covariance is BLOCK-DIAGONAL over the 13 z (per-z 35×35 corr × σσᵀ, no
    cross-z covariance); slicing by a full-z-block mask preserves the block-diagonal structure.

    Cuts: z∈[z_lo,z_hi] (default keeps all 13 z — eBOSS is LOW-k so it has NO low-z small-scale
    resolution pathology, unlike KS z<2.4); k∈(k_min,k_max] (default k_min=0 keeps all eBOSS k,
    k_max=None: no cap (eBOSS max k=0.0195 sits far inside every simulated mode range; the gate E forward refuses
    any bin outside the modes).

    PI-CONFIRMED flags (2026-06-13): metals_on=True — eBOSS data is NOT metal-subtracted (the
    reference SiIII correction is commented out, lyaemu/likelihood.py); SiIII (Δv≈2270) oscillates
    ~6.7 periods in-band at ~5–9%, so it MUST be forward-modeled (a shared ``a_SiIII`` nuisance with
    DESI) or it aliases into the n_s tilt — the single biggest eBOSS risk. (NOTE: a_SiIII SAMPLING in
    the closure is the Phase-4d test-3 follow-on; until wired, metals_on=True is a no-op with a_SiIII=0.)
    dla_forward_frac=0.0 — DLAs masked, residual in C_data (like KS). mf_floor_on=False / no emucoh —
    eBOSS is a LARGE-scale leg below the high-k floor/emucoh regime (emucoh band k≥0.01 is zero at the
    A_p pivot k≈0.009). resolution_on=False — eBOSS resolution syst (~2e-4 in-band) stays in C_data.
    """
    add_cv_floor = _env_flag("HCD_CV_FLOOR") if add_cv_floor is None else bool(add_cv_floor)
    d = np.load(npz_path, allow_pickle=True)
    z = np.asarray(d["z"], float)              # (455,) z-major
    k = np.asarray(d["k"], float)              # (455,) angular k
    P = np.asarray(d["plya"], float)
    cov = np.asarray(d["cov"], float).copy()   # (455,455) block-diag STAT+SYST
    cv_floor_rank1 = _env_flag("HCD_CV_FLOOR_RANK1") if add_cv_floor else False
    if add_cv_floor:
        # Fix 2 (2026-06-18): the Fernandez+2024 σ_CV ~2% finite-box floor at k<2.5e-3 — this
        # is the eBOSS PRIYA analysis it was MEASURED for. ENV-gated/reversible (HCD_CV_FLOOR).
        cov = _add_cv_floor(cov, k, P, rank1=cv_floor_rank1)

    keep = (z >= z_lo - 1e-6) & (z <= z_hi + 1e-6) & (k > k_min)
    if k_max is not None:
        keep &= (k <= k_max + 1e-9)

    # eBOSS has no resolution proxy in the table; reuse the DESI-style proxy as a placeholder for
    # the (default-OFF) resolution knob, exactly like load_ks_leg.
    res_e = (np.asarray(d["syst_resolution"], float)
             if (resolution_float or resolution_coherent) else None)  # option-b / arm-D only (code-lens #7)
    return _assemble_leg("eBOSS", z, k, P, cov, keep,
                         R_func=desi_resolution_R, metals_on=metals_on,
                         resolution_on=resolution_on, mf_floor_on=mf_floor_on,
                         dla_forward_frac=dla_forward_frac,
                         resolution_e=res_e,
                         resolution_float=resolution_float, resolution_mode="rescale",
                         resolution_coherent=resolution_coherent, resolution_coh_amp=resolution_coh_amp,
                         cv_floor_on=add_cv_floor, cv_floor_rank1=cv_floor_rank1)


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


def _subtract_perz_rank1(C, e, z_idx, n_zbins):
    """``C -= sum_z outer(e|z)``: remove ONE correlated systematic in the SHIPPED DESI cov_syst
    convention — each correlated term enters the covariance as a rank-1 ``outer(e|z)`` within
    every z block, exactly zero cross-z (cov_syst is z-block-diagonal; verified 6.2e-18). The
    per-z-block subtraction on the post-cut rows equals the sub-selection of the full-grid
    removal exactly (a rank-1 block outer restricted to kept rows IS the outer of the restricted
    vector). NOT one globally coherent outer product — that would assert a cross-z coherence the
    shipped covariance never carried. Mutates and returns ``C``. Shared by the resolution
    ("rank1" mode) and DLA-completeness removals so the surgery algebra exists ONCE."""
    for i in range(n_zbins):
        rows = np.where(z_idx == i)[0]
        if rows.size:
            C[np.ix_(rows, rows)] -= np.outer(e[rows], e[rows])
    return C


def _assemble_leg(name, z_all, k_all, P_all, cov_all, keep, *, R_func,
                  metals_on, resolution_on, mf_floor_on=False, dla_forward_frac=1.0,
                  resolution_e=None, resolution_float=False, resolution_mode="rank1",
                  resolution_ready=True, resolution_coherent=False, resolution_coh_amp=1.0,
                  dla_e=None, use_snr3=False, cv_floor_on=False, cv_floor_rank1=False):
    """Sub-select the kept (z,k) rows + their covariance block, build the z-major flat
    DataLeg.  The covariance is row/col-sliced by the SAME boolean mask as the data so the
    flat-row ordering matches C_data exactly (CS-REVIEW: ordering invariant).

    ``dla_e`` (DESI only; PI disposition 2026-07-17): the full-grid ``syst_e_dla_completeness``
    column. When given, its per-z rank-1 modes are REMOVED from C_data (the cup1d "red" reduced
    covariance) — licensed ONLY because the alpha_DLA mean model floats on the leg
    (``dla_forward_frac`` > 0; enforced, fail-loud). Stamped as ``dla_cov_reduced``."""
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
    resolution_coherent_on = False
    if resolution_float or resolution_coherent:
        # Rebuild the covariance WITHOUT the spectral-resolution term (shared by option-b + arm-D). The
        # resolution error is a SEPARABLE additive systematic (a diagonal error column), but each survey's
        # cov CONSTRUCTION incorporates it differently, so the removal is per-survey:
        #   "rank1"  (DESI): cov_syst is a sum of per-z-block rank-1 outer(e_i|z) modes; drop resolution's
        #            mode -> cov_b = C - sum_z outer(e_res|z). == cup1d's additive build-without to 1e-13.
        #   "rescale" (eBOSS): cov = corr(x)sigma-sigma^T with sigma^2 = sum_s e_s^2 (resolution baked into
        #            sigma); rebuild sigma'^2 = sigma^2 - e_res^2 -> cov'[i,j] = C[i,j]*(sig'_i/sig_i)(sig'_j/sig_j).
        # option-b then FLOATS f_res in the forward (models it); ARM-D instead re-adds resolution as ONE
        # coherent cross-z mode s^2*outer(e_res,e_res) (marginalizes it in the cov, no forward param -- the
        # linear-Gaussian equivalent of floating a single amplitude; 2026-07-02-coherent-cov-vs-float doc).
        # A wrong mode breaks positive-definiteness; the Cholesky assert is the tripwire.
        if resolution_float and resolution_coherent:
            raise ValueError(f"{name}: resolution_float (option-b) and resolution_coherent (arm-D) are "
                             "mutually exclusive covariance treatments")
        if resolution_e is None:
            raise ValueError(f"{name}: resolution_float/resolution_coherent needs resolution_e")
        e_res = np.asarray(resolution_e, float)[idx]
        C_data = np.array(C_data, float)
        if resolution_mode == "rescale":
            sig2 = np.diag(C_data)                                   # eBOSS: diag(cov)=sigma^2 (corr diag 1)
            ratio = np.sqrt(np.clip(1.0 - e_res ** 2 / sig2, 0.0, None))   # sigma'/sigma
            C_data = C_data * np.outer(ratio, ratio)
        elif resolution_mode == "diag":
            # "diag" (KS): the conservative cov adds each systematic in QUADRATURE to the DIAGONAL only
            # (Karacayli 2021 Sec 4.6), so REMOVE the resolution variance element-wise from the diagonal;
            # the off-diagonal (statistical) carries no resolution term. NOT "rescale" (which also scales
            # the off-diagonal): for KS diag != sum_s e_s^2 (min 0.53 / max 4.09), so a rescale is ill-posed.
            # esyst_res_ks^2 is subdominant in-band (max ~0.17 of the diag) so C_data stays SPD (Cholesky guard).
            _di = np.diag_indices_from(C_data)
            C_data[_di] = C_data[_di] - e_res ** 2
        else:                                                        # "rank1" (DESI): drop the per-z mode
            C_data = _subtract_perz_rank1(C_data, e_res, z_idx, len(z))
        if resolution_coherent:
            # ARM-D: re-add the resolution error as ONE coherent (cross-z) rank-1 mode. Restores the total
            # diagonal variance (diag == option-a) but re-correlates it coherently across z. resolution_on
            # stays False (NO forward f_res). Amplitude s: s=1 == the shipped 1-sigma resolution uncertainty.
            C_data = C_data + (float(resolution_coh_amp) ** 2) * np.outer(e_res, e_res)
            resolution_coherent_on = True
        else:
            resolution_on = True                    # option-b floats f_res in the forward
        np.linalg.cholesky(C_data)                 # SPD assert (fails loudly if the mode is mis-specified)
    dla_cov_reduced = False
    if dla_e is not None:
        # Reduced covariance (cup1d "red"): remove the DLA-completeness per-z rank-1 modes. The
        # modeled-in-mean licence is MANDATORY — removing a variance term whose effect the forward
        # does NOT model would silently under-cover, so a zero forward DLA term fails loud here.
        if not float(dla_forward_frac) > 0.0:
            raise ValueError(
                f"{name}: dla_e (reduced covariance) requires the alpha_DLA mean model to be live "
                f"on this leg (dla_forward_frac > 0, got {dla_forward_frac!r}); removing the "
                "DLA-completeness variance without modeling it in the mean would under-cover")
        C_data = _subtract_perz_rank1(np.array(C_data, float),
                                      np.asarray(dla_e, float)[idx], z_idx, len(z))
        np.linalg.cholesky(C_data)                 # SPD assert (mathematically stat + remaining syst)
        dla_cov_reduced = True
    R_z = np.asarray(R_func(z))
    return DataLeg(
        name=name, z=z, z_unit=_z_unit(z), k=k, z_row=z_row, z_idx=z_idx,
        P_data=P_data, C_data=C_data, R_z=R_z, n_z=len(z), n_per_z=n_per_z,
        metals_on=metals_on, resolution_on=resolution_on, mf_floor_on=mf_floor_on,
        dla_forward_frac=dla_forward_frac, resolution_ready=resolution_ready,
        resolution_coherent_on=resolution_coherent_on, dla_cov_reduced=dla_cov_reduced,
        use_snr3=bool(use_snr3), cv_floor_on=bool(cv_floor_on),
        cv_floor_rank1=bool(cv_floor_rank1))


# ============================================================================ #
#  Forward-model nuisances (differentiable; per-leg-configurable, default OFF)
# ============================================================================ #
def _metal_factor(k, *, a_SiIII=0.0, a_SiII=0.0, k_SiIII=K_SiIII_DEFAULT, k_SiII=K_SiII_DEFAULT,
                  cross=False):
    """Metal contamination multiplier — companion arXiv:2601.21432 Eq. 4.2–4.3:

        P → P · (1 + C_LyαSiIII + C_LyαSiII [+ C_SiIIISiII]),
        C_LyαX = a_X²  +  2 a_X · cos(k·Δv_X) · D_X(k),   D_X(k) = 2 − 2/(1 + exp(−k/k_X)).

    The CONSTANT a_X² term is UNDAMPED; the SIGMOID decorrelation D_X (k_X a FREE nuisance)
    multiplies ONLY the oscillatory cosine cross-term — NOT a Gaussian on the whole (1+f) (the
    earlier usage-doc one-liner `(1+f)·exp(−k²/2k_s²)` was wrong; Lyα-confirmed 2026-06-05).
    Δv_X = c·ln(λ_Lyα/λ_X).  a_SiIII=a_SiII=0 ⇒ factor ≡ 1.  Differentiable in
    (a_SiIII, a_SiII, k_SiIII, k_SiII).

    SiII is the true 1190.42+1193.28 DOUBLET (matches closure_legb.metal_inject, the gate injection):
        C_LyαSiII = a_SiII²·(1+r²) + 2 a_SiII·(cos(k·Δv_b) + r·cos(k·Δv_a))·D_SiII,
    r = R_SiII_DOUBLET (intra-doublet ratio), Δv_a = leading 1190.42, Δv_b = 1193.28. r=0 ⇒ the old
    single-line form; a_SiII=0 ⇒ byte-exact identity (golden-safe).

    ``cross`` (Model C+, default False → BYTE-EXACT legacy): when True ADD the SiIII–SiII metal-metal
    CROSS term (cup1d si_mult.py ``Cmm``, paper Eq. 4.5), amplitude TIED to a_SiIII·a_SiII (NO new
    DOF), UNDAMPED to match cup1d exactly:
        C_SiIIISiII = 2 a_SiIII a_SiII · (cos(k·Δv_SiIII_SiIIb) + r·cos(k·Δv_SiIII_SiIIa)),
    Δv_SiIII_SiIIX = c·ln(λ_SiIIX/λ_SiIII) (the SiIII–SiII line separations). cup1d leaves Cmm
    UNDAMPED; the damped-vs-undamped choice was checked IMMATERIAL in band (worst-case in-prior
    ΔP/P ≲ 0.41× the tightest DESI bin, a rapid oscillation not a broadband tilt — notes
    2026-06-30-metal-cross-damping.md / diag_metal_cross_damping.py), so we match cup1d.
    a_SiII=0 ⇒ cross ≡ 0 (eBOSS / back-compat byte-exact)."""
    k = jnp.asarray(k)
    dv_SiIII = C_KMS * jnp.log(LAMBDA_LYA / LAMBDA_SiIII)
    dv_SiIIa = C_KMS * jnp.log(LAMBDA_LYA / LAMBDA_SiII)        # leading doublet line 1190.42
    dv_SiIIb = C_KMS * jnp.log(LAMBDA_LYA / LAMBDA_SiIIb)       # second doublet line 1193.28
    r = R_SiII_DOUBLET
    D_SiIII = 2.0 - 2.0 / (1.0 + jnp.exp(-k / k_SiIII))      # sigmoid decorrelation, →1 low-k →0 high-k
    D_SiII = 2.0 - 2.0 / (1.0 + jnp.exp(-k / k_SiII))
    f = (a_SiIII ** 2 + 2.0 * a_SiIII * jnp.cos(k * dv_SiIII) * D_SiIII) \
        + (a_SiII ** 2 * (1.0 + r ** 2)
           + 2.0 * a_SiII * (jnp.cos(k * dv_SiIIb) + r * jnp.cos(k * dv_SiIIa)) * D_SiII)
    if cross:
        dv_cross_b = C_KMS * jnp.log(LAMBDA_SiIIb / LAMBDA_SiIII)   # SiIII–SiII line b (1193.28)
        dv_cross_a = C_KMS * jnp.log(LAMBDA_SiII / LAMBDA_SiIII)    # SiIII–SiII line a (1190.42)
        f = f + 2.0 * a_SiIII * a_SiII * (jnp.cos(k * dv_cross_b)
                                          + r * jnp.cos(k * dv_cross_a))   # UNDAMPED: cup1d Cmm
    return 1.0 + f


def metal_factor_at_z(k_sub, z, tau0, *, a_SiIII=0.0, a_SiII=0.0, k_SiIII=K_SiIII_DEFAULT, k_SiII=K_SiII_DEFAULT,
                      f_SiIII_nodes=None, f_SiII_nodes=None, metal_node_z=(2.2, 4.2), k_SiIII_nodes=None,
                      k_SiII_nodes=None):
    """The metal factor at one leg z (data k only). MODEL C+ when ``f_SiIII_nodes`` is given: a(z) = f(z)/(1 - <F>(z))
    with <F>(z) = exp(-tau0) the SAMPLED mean flux (differentiable in tau0), log10 f(z) and log10 k-scale(z) LINEAR in
    log10(1+z) between ``metal_node_z`` (jnp.interp clamps beyond the nodes, eBOSS z > 4.2), SiIII-SiII cross term on
    (it is proportional to a_SiII, so it vanishes on SiIII-only legs); otherwise the legacy scalar amplitudes."""
    if f_SiIII_nodes is None:
        return _metal_factor(k_sub, a_SiIII=a_SiIII, a_SiII=a_SiII, k_SiIII=k_SiIII, k_SiII=k_SiII)
    _logz = jnp.log10(1.0 + z)
    _xp = jnp.log10(1.0 + jnp.asarray(metal_node_z))
    _omF = 1.0 - jnp.exp(-tau0)
    a3_z = (10.0 ** jnp.interp(_logz, _xp, jnp.log10(f_SiIII_nodes))) / _omF
    a2_z = ((10.0 ** jnp.interp(_logz, _xp, jnp.log10(f_SiII_nodes))) / _omF
            if f_SiII_nodes is not None else 0.0)
    k3_z = (10.0 ** jnp.interp(_logz, _xp, jnp.log10(k_SiIII_nodes))
            if k_SiIII_nodes is not None else k_SiIII)
    k2_z = (10.0 ** jnp.interp(_logz, _xp, jnp.log10(k_SiII_nodes))
            if k_SiII_nodes is not None else k_SiII)
    return _metal_factor(k_sub, a_SiIII=a3_z, a_SiII=a2_z, k_SiIII=k3_z, k_SiII=k2_z, cross=True)


def _resolution_factor(k, R_z, *, b_res=0.0):
    """Resolution-template multiplier P → P·exp(2 b_res k² R_z²) (usage doc Eq. 4.8, option
    (b)). b_res=0 ⇒ factor ≡ 1.  Differentiable in b_res.  Default OFF (the residual
    resolution mode stays in C unless this knob is turned on per-leg)."""
    return jnp.exp(2.0 * b_res * jnp.asarray(k) ** 2 * jnp.asarray(R_z) ** 2)






























def emu_var_modes(P_filt, z, tau0, alpha_hcd, *, dla_core, alpha_centres, sigma_zb=None, rho_zb=None,
                  cemu_inflate=1.0):
    """The per-mode emulator variance at one z from the LF per-class ``P_filt`` (4, K) (the production C_emu uses the
    LF, not the MF-corrected, P_filt), the cross-class block ``rho_zb`` (4, 4, K, Tb) or the diagonal ``sigma_zb``
    (4, K, Tb), tau0-interpolated over ``alpha_centres``. See ``_emu_var_on_cache``."""
    P_clean = P_filt[0]
    P_dla_unf = P_filt[3] + jnp.asarray(dla_core)
    P_cls = jnp.stack([P_clean, P_filt[1], P_filt[2], P_dla_unf])         # (4,Kc)
    a = jnp.asarray(alpha_hcd)
    coef = jnp.concatenate([jnp.atleast_1d(1.0 - jnp.sum(a)), a])         # (4,)
    if rho_zb is not None:
        rho_ck = rho_at_tau0(rho_zb, alpha_centres, z, tau0)             # (4,4,Kc) τ₀-interp'd
        # emu_var(k) = Σ_{c,c'} coef_c·coef_c'·ρ_cc'(k)·P_c·P_c'  (≥0: ρ SPD).
        emu_var = jnp.einsum("c,d,cdk,ck,dk->k", coef, coef,
                             jnp.nan_to_num(rho_ck), P_cls, P_cls)
    else:
        sigma_ck = sigma_at_tau0(sigma_zb, alpha_centres, z, tau0)        # (4,Kc) fractional
        emu_var = jnp.einsum("c,ck,ck->k", coef ** 2,
                             jnp.nan_to_num(sigma_ck) ** 2, P_cls ** 2)
    return emu_var * cemu_inflate


def emu_var_at_data(A, core, rho_d, alpha_hcd, cemu_inflate=1.0):
    """The T1 cross-class emulator variance at data bins (gate E amendment A1 rev 1 section 1): sum_cc' coef_c coef_c'
    rho_cc' A_c A_c' with A (4, n) the LF per-class P_filt bound to the bins, the DLA core (n,) added to the DLA
    amplitude, rho_d (4, 4, n) the tau0-interpolated block bound to the bins, coef = (1 - sum alpha, alpha)."""
    A = jnp.asarray(A)
    A = A.at[3].add(jnp.asarray(core))
    a = jnp.asarray(alpha_hcd)
    coef = jnp.concatenate([jnp.atleast_1d(1.0 - jnp.sum(a)), a])
    return jnp.einsum("c,d,cdn,cn,dn->n", coef, coef, jnp.nan_to_num(jnp.asarray(rho_d)), A, A) * cemu_inflate


# ns is theta9[0] on the unit cube; PRIYA's design box maps it to physical ns.
# (data.PARAM_LIMITS[0] = [0.8, 1.05]; the floor's ns_box is in PHYSICAL ns.)
_NS_CUBE_LO, _NS_CUBE_HI = 0.8, 1.05


def _ns_phys_from_theta9(theta9):
    """Physical n_s = 0.8 + 0.25·θ9[0] (PRIYA's unit-cube → ns design box)."""
    return _NS_CUBE_LO + (jnp.asarray(theta9)[0]) * (_NS_CUBE_HI - _NS_CUBE_LO)








