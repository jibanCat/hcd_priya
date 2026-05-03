"""
b_F (Lyman-α forest bias) calibrator from the all-HCD-masked P1D.

Used by the cross-correlation pipeline (`hcd_analysis.clustering`) to
extract `b_DLA = (b_DLA · b_Lyα) / b_Lyα` from the cross-correlation
ξ_× — the cross gives `b_DLA · b_Lyα`, so we need an independent
`b_Lyα` calibrator on the same snap.

Approach
--------
We fit a linear-theory model

    P1D_F(k_par) = b_F²  ·  I(k_par; β_F)

where

    I(k_par; β_F) = (1/2π) ∫_0^∞ k_perp dk_perp  ·
                        (1 + β_F μ²)²  ·  P_lin_3D(k_3D)

is the Kaiser-RSD-modulated 1D projection of the linear matter P(k),
with `μ = k_par / k_3D` and `k_3D² = k_par² + k_perp²`.  At fixed
`β_F` (we use Slosar+2011's z ≈ 2.3 value of 1.5 by default), this is
a one-parameter fit for `b_F²`.

Observed P1D
------------
We compute the observed P1D from sightlines that have NO masked
pixels, using the existing ``P1DAccumulator``.  Restricting to
HCD-free sightlines removes the mask-window-function complication
(typically ~ 99 % of sightlines are HCD-free at z = 2-5 in PRIYA).

CAMB power-spectrum input
-------------------------
We read the per-snap linear P(k) from
``<sim>/output/powerspectrum-<a>.txt`` (k [h/Mpc], P [(Mpc/h)³]).  The
file at scale factor `a = 1/(1+z)` is the closest match.  We convert
to (km/s) units using

    k_v [s/km]   = k_r [h/Mpc]  · h · (1+z) / H(z)
    P_v [(km/s)³] = P_r [(Mpc/h)³] · ( h · (1+z) / H(z) )³

Output
------
``fit_b_F`` returns a dict with the recovered `b_F`, its 1-σ
uncertainty, the input/template P1D arrays, and the chi² of the fit.

Tests in ``tests/test_lya_bias.py`` cover:
1. Projection integral converges to a known closed form on a power-law P_lin.
2. CAMB loader returns the right units.
3. Synthetic linear forest with planted b_F → recover at 5 % level.
4. Real-data sanity: the recovered b_F at z ≈ 2.3 lies in
   [-0.25, -0.12] (Slosar+2011 / du Mas des Bourboux+2020 envelope).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid as _trapezoid  # numpy 1.x has no _trapezoid
from scipy.special import spherical_jn


# ---------------------------------------------------------------------------
# CAMB power-spectrum loader
# ---------------------------------------------------------------------------

def load_camb_pk(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a PRIYA-format CAMB power spectrum.

    Format (from MP-Gadget run): two header comment lines + a 4-column
    table with ``k [h/Mpc]   P(k) [(Mpc/h)^3]   N_modes   P(k,z=0)``.

    Returns
    -------
    k : (N,) float64, in h/Mpc
    P : (N,) float64, in (Mpc/h)^3 at the file's redshift (column 1)
    """
    data = np.loadtxt(Path(path), comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"unexpected shape for CAMB Pk file {path}: {data.shape}")
    k = data[:, 0].astype(np.float64)
    P = data[:, 1].astype(np.float64)
    if not (np.all(k > 0) and np.all(np.diff(k) > 0)):
        raise ValueError(f"k array in {path} is not strictly increasing or has nonpos values")
    return k, P


def find_camb_pk_for_z(sim_output_dir: Path, z: float) -> Path:
    """Return the powerspectrum-<a>.txt file closest to a = 1/(1+z)."""
    sim_output_dir = Path(sim_output_dir)
    target_a = 1.0 / (1.0 + z)
    candidates = sorted(sim_output_dir.glob("powerspectrum-*.txt"))
    if not candidates:
        raise FileNotFoundError(f"no powerspectrum-*.txt under {sim_output_dir}")
    def _a_of(f: Path) -> float:
        # filename is powerspectrum-<a>.txt
        try:
            return float(f.stem.split("-")[1])
        except Exception:
            return -1.0
    return min(candidates, key=lambda f: abs(_a_of(f) - target_a))


# ---------------------------------------------------------------------------
# Unit conversion: (Mpc/h, h/Mpc) ↔ (km/s, s/km) at fixed z
# ---------------------------------------------------------------------------

def hMpc_to_kms_factor(z: float, hubble: float, Hz_kms_per_Mpc: float) -> float:
    """Conversion factor F such that

        k_v  [s/km]      = k_r [h/Mpc]    * F
        L_v  [km/s]      = L_r [Mpc/h]    / F
        P_v  [(km/s)^d]  = P_r [(Mpc/h)^d] / F^d        (NOT * F^d!)

    Derivation: a comoving distance L [Mpc/h] in the LOS direction
    corresponds to a redshift-space velocity span

        Δv = a · H(z) · L_proper = (1/(1+z)) · H(z) · (L[Mpc/h] / h).

    Hence  L[km/s] = L[Mpc/h] · H(z) / (h · (1+z)) = L[Mpc/h] / F
    with   F = h · (1+z) / H(z).

    Power spectra have dimensions of length^d, so
    P_v / P_r = (L_v / L_r)^d = (1 / F)^d = 1 / F^d.
    """
    return hubble * (1.0 + z) / Hz_kms_per_Mpc


# ---------------------------------------------------------------------------
# 1-D projection of P_lin_3D with Kaiser β_F enhancement
# ---------------------------------------------------------------------------

def project_pk_3d_to_p1d(
    k_par: np.ndarray,
    k_3d: np.ndarray,
    P_3d: np.ndarray,
    beta_F: float,
    n_perp: int = 4096,
    k_perp_max_factor: float = 50.0,
) -> np.ndarray:
    """Compute  I(k_par; β_F) = (1/2π) ∫_0^∞ k_perp dk_perp · (1 + β_F μ²)² · P_lin_3D(k_3D).

    The (1 + β_F μ²)² factor is the Kaiser-RSD-monopole enhancement
    of a linearly biased tracer (Lyα).  The full Kaiser P(k, μ) =
    (b + f·μ²)² · P_lin(k) factorises out b² when expressed via
    β_F = f / b, leaving the (1 + β_F μ²)² inside the integrand.
    `b_F²` is fit OUTSIDE this function.

    Parameters
    ----------
    k_par : (n_par,) array of LOS wavenumbers; same units as k_3d, P_3d
    k_3d, P_3d : the linear 3-D matter P(k); k_3d strictly increasing
    beta_F : Kaiser parameter for the Lyα forest tracer (~ 1.5 at z = 2-3)
    n_perp : number of k_perp points in the trapezoidal integration
    k_perp_max_factor : integrate up to k_perp_max = factor · max(k_par);
        the integrand decays as k_perp · P_3d(k_3d) ~ k_perp · P_lin(k_perp)
        which falls off rapidly (P_lin ∝ k^{-3} on small scales).

    Returns
    -------
    I : (n_par,) array of values I(k_par; β_F), same units as k_par^{-2} · P_3d
        (i.e. (Mpc/h)² · (Mpc/h)³ → (Mpc/h) per unit k_par);  in (km/s)
        units throughout this becomes simply (km/s) on output.
    """
    k_par = np.asarray(k_par, dtype=np.float64)
    k_3d = np.asarray(k_3d, dtype=np.float64)
    P_3d = np.asarray(P_3d, dtype=np.float64)
    if k_3d.ndim != 1 or P_3d.ndim != 1 or k_3d.shape != P_3d.shape:
        raise ValueError("k_3d, P_3d must be matched 1-D arrays")
    if not np.all(np.diff(k_3d) > 0):
        raise ValueError("k_3d must be strictly increasing")

    # P_3d evaluator: log-log interpolation (linear matter spectrum is
    # smooth in log-log).  Outside the k_3d range, return 0.
    log_k = np.log(k_3d)
    log_P = np.log(np.maximum(P_3d, 1e-300))

    def P_3d_eval(k):
        out = np.zeros_like(k)
        # Inclusive bounds — Copilot review #2 on PR #7: strict < was
        # zeroing the integrand at the exact endpoints of k_3d, which
        # slightly suppressed the integral.  np.interp is well-defined
        # at endpoints, so include them.
        valid = (k >= k_3d.min()) & (k <= k_3d.max())
        out[valid] = np.exp(np.interp(np.log(k[valid]), log_k, log_P))
        return out

    # k_perp grid: log-spaced from k_par_min · 1e-4 to k_par_max · factor.
    k_perp_lo = max(1e-6 * k_par.min(), 1e-9)
    k_perp_hi = k_perp_max_factor * k_par.max()
    k_perp = np.logspace(np.log10(k_perp_lo), np.log10(k_perp_hi), n_perp)

    # For each k_par in the input, compute the integral
    out = np.zeros_like(k_par)
    for i, kp in enumerate(k_par):
        if kp <= 0:
            continue
        k3 = np.sqrt(kp * kp + k_perp * k_perp)
        mu = kp / k3
        kaiser = (1.0 + beta_F * mu * mu) ** 2
        integrand = k_perp * kaiser * P_3d_eval(k3)        # k_perp · ...
        # Trapezoidal integration ∫ f(k_perp) dk_perp.
        out[i] = _trapezoid(integrand, k_perp) / (2.0 * np.pi)
    return out


# ---------------------------------------------------------------------------
# Observed P1D over HCD-free sightlines (clean forest)
# ---------------------------------------------------------------------------

def compute_p1d_clean_sightlines(
    delta_F: np.ndarray,
    dv_kms: float,
    pixel_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute P1D averaged over sightlines with NO masked pixels.

    A "clean" sightline is one where ``pixel_mask[sk, :].any() == False``,
    i.e. no HCD touches it anywhere along the LOS.  Restricting to
    clean sightlines means we never have to deal with the FFT window
    function induced by zero-filling masked pixels.

    Parameters
    ----------
    delta_F : (n_skewers, n_pix) δ_F field (with masked pixels = 0)
    dv_kms  : pixel velocity width in km/s
    pixel_mask : (n_skewers, n_pix) bool, True where the pixel was masked

    Returns
    -------
    k_par : (n_pix//2 + 1,) one-sided FFT frequencies in s/km (cyclic)
    P1D : (n_pix//2 + 1,) one-sided P1D in km/s
    n_clean : number of clean sightlines used
    """
    delta_F = np.asarray(delta_F, dtype=np.float64)
    pixel_mask = np.asarray(pixel_mask, dtype=bool)
    if delta_F.shape != pixel_mask.shape:
        raise ValueError("delta_F and pixel_mask must have the same shape")
    n_skewers, n_pix = delta_F.shape

    clean_rows = ~pixel_mask.any(axis=1)
    n_clean = int(clean_rows.sum())
    if n_clean == 0:
        raise ValueError("no HCD-free sightlines available")

    # FFT only the clean rows
    df_clean = delta_F[clean_rows]                        # (n_clean, n_pix)
    ft = np.fft.rfft(df_clean * dv_kms, axis=1)            # units: km/s
    power = np.abs(ft) ** 2 / (n_pix * dv_kms)             # units: km/s
    P1D = power.mean(axis=0)
    k_par = np.fft.rfftfreq(n_pix, d=dv_kms)               # s/km
    return k_par, P1D, n_clean


# ---------------------------------------------------------------------------
# b_F fit
# ---------------------------------------------------------------------------

@dataclass
class BFFitResult:
    b_F: float
    b_F_err: float
    beta_F_assumed: float
    k_par_kms: np.ndarray
    P1D_obs_kms: np.ndarray
    P1D_template_kms: np.ndarray
    fit_mask: np.ndarray
    n_clean_sightlines: int
    chi2: float
    z: float
    note: str = ""


def fit_b_F(
    delta_F: np.ndarray,
    pixel_mask: np.ndarray,
    dv_kms: float,
    z: float,
    hubble: float,
    Hz_kms_per_Mpc: float,
    P_lin_camb_path: Path,
    beta_F_assume: float = 1.5,
    k_min_kms: float = 5.0e-4,
    k_max_kms: float = 5.0e-3,
) -> BFFitResult:
    """Fit b_F from the masked-δ_F P1D vs the linear-theory template.

    Returns a `BFFitResult` with the best-fit `b_F` (sign convention:
    negative for Lyα, since more matter → more absorption → less
    flux), its 1-σ uncertainty from the per-bin scatter, and the
    arrays needed to plot the fit.

    The fit window `[k_min_kms, k_max_kms]` defaults to the linear
    range that Slosar+2011 use (≈ 0.0005 to 0.005 s/km, cyclic).
    """
    # 1) observed P1D over clean sightlines
    k_par_kms, P1D_obs_kms, n_clean = compute_p1d_clean_sightlines(
        delta_F=delta_F, dv_kms=dv_kms, pixel_mask=pixel_mask,
    )

    # 2) load CAMB P_lin_3D, convert to (km/s) units
    k_hMpc, P_Mpch3 = load_camb_pk(P_lin_camb_path)
    F = hMpc_to_kms_factor(z, hubble, Hz_kms_per_Mpc)
    k_3d_kms = k_hMpc * F                                  # s/km
    # P has units of length^3.  L_v = L_r / F, so P_v = P_r / F^3.
    P_3d_kms = P_Mpch3 / (F ** 3)                          # (km/s)^3

    # 3) numerical projection at our k_par grid
    I_kpar = project_pk_3d_to_p1d(
        k_par=k_par_kms, k_3d=k_3d_kms, P_3d=P_3d_kms,
        beta_F=beta_F_assume,
    )                                                      # (km/s)

    # 4) fit b_F² as the geometric-mean ratio over the linear window
    fit_mask = (
        (k_par_kms >= k_min_kms) &
        (k_par_kms <= k_max_kms) &
        (I_kpar > 0) &
        (P1D_obs_kms > 0)
    )
    n_pts = int(fit_mask.sum())
    if n_pts < 5:
        raise ValueError(
            f"only {n_pts} fit points in [{k_min_kms}, {k_max_kms}] s/km; "
            f"check the k range or n_pix"
        )
    log_ratio = np.log(P1D_obs_kms[fit_mask]) - np.log(I_kpar[fit_mask])
    log_b_F_sq = float(log_ratio.mean())
    log_b_F_sq_se = float(log_ratio.std(ddof=1) / np.sqrt(n_pts))
    b_F_sq = float(np.exp(log_b_F_sq))
    # Lyα convention: b_F < 0 (more matter → more absorption → less F).
    b_F = -np.sqrt(b_F_sq)
    # Propagate ln-space SE to b_F itself: var(ln b²) ≈ (2/b · b_err)² → b_err = b/2 · SE.
    b_F_err = abs(b_F) * 0.5 * log_b_F_sq_se

    # χ² (uniform-weight at this stage; per-bin variance is hard to define
    # without bootstrap)
    P_template = b_F_sq * I_kpar
    residual = P1D_obs_kms[fit_mask] - P_template[fit_mask]
    chi2 = float((residual ** 2 / np.maximum(P_template[fit_mask], 1e-30) ** 2).sum())

    return BFFitResult(
        b_F=b_F,
        b_F_err=b_F_err,
        beta_F_assumed=beta_F_assume,
        k_par_kms=k_par_kms,
        P1D_obs_kms=P1D_obs_kms,
        P1D_template_kms=P_template,
        fit_mask=fit_mask,
        n_clean_sightlines=n_clean,
        chi2=chi2,
        z=z,
        note=f"β_F fixed at {beta_F_assume}; fit on {n_pts} bins in "
             f"[{k_min_kms:.3g}, {k_max_kms:.3g}] s/km",
    )


# ---------------------------------------------------------------------------
# Linear matter ξ_lin(r) monopole and the b_F fit from ξ_FF
# ---------------------------------------------------------------------------

def xi_lin_monopole(
    r: np.ndarray,
    k: np.ndarray,
    P_lin: np.ndarray,
) -> np.ndarray:
    """Compute the monopole linear-matter correlation function

        ξ_lin^(0)(r) = (1/(2π²)) ∫_0^∞ dk · k² · P_lin(k) · j₀(k·r)
                     = (1/(2π²)) ∫_0^∞ dk · k · sin(k·r) · P_lin(k) / r

    via direct trapezoidal quadrature on the input k-grid.  All
    arrays must be in CONSISTENT UNITS — pass k in s/km and P in
    (km/s)³ to get ξ in dimensionless natural units; pass k in
    h/Mpc and P in (Mpc/h)³ to get ξ in dimensionless units (the
    natural scaling of the correlation function).  We use the
    "k·sin(kr)/r" form because it stays well-behaved as r → 0
    (sin(k·r) ≈ k·r for small r, so the integrand becomes k²·P_lin —
    the variance of the field).

    Parameters
    ----------
    r       : (n_r,) float — separations
    k       : (n_k,) float — strictly increasing wavenumber grid
    P_lin   : (n_k,) float — P_lin at those k

    Returns
    -------
    xi : (n_r,) float
    """
    r = np.asarray(r, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    P = np.asarray(P_lin, dtype=np.float64)
    if k.shape != P.shape:
        raise ValueError("k and P_lin must match in shape")
    if not np.all(np.diff(k) > 0):
        raise ValueError("k must be strictly increasing")

    out = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= 0:
            # ξ(0) = (1/(2π²)) ∫ dk · k² · P_lin(k) — variance of the field
            integrand = k * k * P
        else:
            integrand = k * np.sin(k * ri) * P / ri
        out[i] = float(_trapezoid(integrand, k)) / (2.0 * np.pi ** 2)
    return out


def xi_lin_quadrupole(
    r: np.ndarray,
    k: np.ndarray,
    P_lin: np.ndarray,
) -> np.ndarray:
    """The j_2-transform of P_lin, used as the quadrupole-of-ξ template:

        ξ_lin^(j2)(r) = (1/2π²) ∫_0^∞ dk · k² · j_2(k·r) · P_lin(k).

    In the Kaiser cross model

        ξ_×^(2)(r) = − b_DLA · b_F · [(2/3)(β_DLA + β_F) + (4/7) β_DLA β_F]
                       · ξ_lin^(j2)(r)

    the i^ℓ = i² = −1 prefactor of the Hamilton transform is absorbed
    into the global sign in front, NOT into ``xi_lin_quadrupole`` —
    ``xi_lin_quadrupole`` is a positive-area-weighted positive
    quantity at small r (j_2 ∝ r² · P_lin · r² near the integrand
    peak), and the negative quadrupole sign of an RSD-distorted halo
    field shows up as the explicit minus in the model.

    Same units convention as ``xi_lin_monopole``: pass (k in s/km, P
    in (km/s)³) to get ξ in dimensionless natural units, or (k in
    h/Mpc, P in (Mpc/h)³) likewise.
    """
    r = np.asarray(r, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    P = np.asarray(P_lin, dtype=np.float64)
    if k.shape != P.shape:
        raise ValueError("k and P_lin must match in shape")
    if not np.all(np.diff(k) > 0):
        raise ValueError("k must be strictly increasing")

    out = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= 0:
            # j_2(0) = 0, so ξ^(2)(0) = 0.
            out[i] = 0.0
            continue
        integrand = k * k * spherical_jn(2, k * ri) * P
        out[i] = float(_trapezoid(integrand, k)) / (2.0 * np.pi ** 2)
    return out


# ---------------------------------------------------------------------------
# Hamilton multipole extraction from an (r, |μ|)-binned ξ grid
# ---------------------------------------------------------------------------

def _legendre(ell: int, mu: np.ndarray) -> np.ndarray:
    """Plain-Python Legendre polynomial for the ones we use.

    Avoids pulling scipy.special.legendre, which is slower and returns
    a callable.  Hand-coded for L_0, L_2, L_4.
    """
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu * mu - 1.0)
    if ell == 4:
        m2 = mu * mu
        return (35.0 * m2 * m2 - 30.0 * m2 + 3.0) / 8.0
    raise ValueError(f"L_{ell} not supported (only ell ∈ {{0, 2, 4}})")


def extract_multipoles_rmu(
    xi_rmu: np.ndarray,
    npairs_rmu: np.ndarray,
    mu_centres: np.ndarray,
    ells: Tuple[int, ...] = (0, 2),
) -> dict:
    """Hamilton multipole extraction from a ξ(r, |μ|) grid.

    For each r-bin, project the row ξ(r, |μ|) onto Legendre polynomials
    using the standard uniform-μ formula on the half range μ ∈ [0, 1]:

        ξ^(ℓ)(r) = (2ℓ + 1) · ∫_0^1 dμ · ξ(r, μ) · L_ℓ(μ).

    The factor is (2ℓ + 1), NOT (2ℓ + 1)/2, because ξ(r, μ) =
    ξ(r, −μ) (cosmological reflection symmetry) means the [0, 1] half
    integral × 2 equals the [−1, 1] full integral, and the standard
    [−1, 1] projection coefficient is (2ℓ + 1)/2 · ∫_{−1}^1 = (2ℓ +
    1) · ∫_0^1.  This is the picca / RascalC convention and recovers
    a Hamilton-1992 synthesis ξ_0 + ξ_2 · L_2(μ) exactly (see
    ``tests/test_lya_bias.py::TestExtractMultipolesRMu``).

    Empty bins (npairs == 0) are NaN in ``xi_rmu``.  The projection sum
    skips them and is renormalised by the surviving Σ Δμ_j so the
    estimator targets the *full* [0, 1] integral instead of the
    partial-coverage one (i.e. missing bins are filled in by the
    valid-bin average rather than contributing zero).  On a fully-
    populated row Σ(valid Δμ_j) == 1 and the renormalisation is a
    no-op.  If a r-bin has fewer than 3 valid μ-bins the recovered
    multipole is NaN for that r — the renormalisation can't be
    trusted on rows that sparse.

    Parameters
    ----------
    xi_rmu : (n_r, n_mu) float
        ξ values at each (r, |μ|) bin centre; NaN where pairs are missing.
    npairs_rmu : (n_r, n_mu) int
        Pair count per bin — used only to detect empty bins.
    mu_centres : (n_mu,) float
        μ-bin centres on [0, 1].  Must be monotonically increasing.
    ells : tuple of int
        Multipole orders to extract.  Default (0, 2).

    Returns
    -------
    multipoles : dict[int, np.ndarray]
        ``multipoles[ell]`` is a (n_r,) array of ξ^(ℓ)(r).
    """
    xi_rmu = np.asarray(xi_rmu, dtype=np.float64)
    npairs_rmu = np.asarray(npairs_rmu, dtype=np.int64)
    mu_centres = np.asarray(mu_centres, dtype=np.float64)
    if xi_rmu.shape != npairs_rmu.shape:
        raise ValueError("xi_rmu and npairs_rmu must have the same shape")
    if xi_rmu.shape[1] != mu_centres.size:
        raise ValueError(
            f"xi_rmu has {xi_rmu.shape[1]} μ-columns, mu_centres has "
            f"{mu_centres.size}"
        )
    if mu_centres.size < 2:
        raise ValueError("need at least 2 μ-bins to project multipoles")
    if not np.all(np.diff(mu_centres) > 0):
        raise ValueError("mu_centres must be strictly increasing")
    if mu_centres[0] < 0 or mu_centres[-1] > 1.0 + 1e-12:
        raise ValueError(f"mu_centres must lie in [0, 1]; got "
                         f"[{mu_centres[0]}, {mu_centres[-1]}]")

    # Approximate dμ per bin from the centres (assumes uniform μ-bins,
    # which is what production passes).  Deduce edges as midpoints; clip
    # to [0, 1].
    edges = np.empty(mu_centres.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (mu_centres[:-1] + mu_centres[1:])
    edges[0] = max(0.0, 2.0 * mu_centres[0] - edges[1])
    edges[-1] = min(1.0, 2.0 * mu_centres[-1] - edges[-2])
    dmu = np.diff(edges)                                          # (n_mu,)

    n_r, n_mu = xi_rmu.shape
    valid = (npairs_rmu > 0) & np.isfinite(xi_rmu)
    n_valid_per_r = valid.sum(axis=1)

    # Σ(valid Δμ_j) per r-row.  Equals 1 when every μ-bin is populated;
    # smaller when bins are missing.  Used to renormalise the partial
    # sum so the projection still targets ∫_0^1 instead of the
    # truncated integral.
    dmu_valid = np.where(valid, dmu[None, :], 0.0)
    norm_per_r = dmu_valid.sum(axis=1)                            # (n_r,)

    out = {}
    for ell in ells:
        L = _legendre(ell, mu_centres)                            # (n_mu,)
        weights = (L * dmu)[None, :]                              # (1, n_mu)
        contrib = np.where(valid, xi_rmu * weights, 0.0)
        proj_partial = contrib.sum(axis=1)
        # Hamilton: ξ^(ℓ)(r) = (2ℓ+1) · ∫_0^1 dμ ξ(r,μ) L_ℓ(μ).
        # On a fully-populated row, Σ_j L_j Δμ_j = ∫_0^1 L dμ and
        # Σ Δμ_j = 1, so dividing by norm_per_r is a no-op.  When some
        # bins are missing, dividing by Σ_valid Δμ_j extrapolates the
        # surviving μ-bins to the full range; this is more accurate
        # than letting missing bins contribute zero.
        proj = np.where(norm_per_r > 0, proj_partial / norm_per_r, 0.0)
        mp = (2 * ell + 1) * proj
        mp = np.where(n_valid_per_r >= 3, mp, np.nan)
        out[ell] = mp
    return out


# ---------------------------------------------------------------------------
# Monopole extraction from a 2-D (r_par, r_perp) grid
# ---------------------------------------------------------------------------
#
# A general `extract_multipoles(..., ells=(0, 2, 4))` helper used to live
# here, but it had a subtle Jacobian bug that made the recovered
# quadrupole / hexadecapole inconsistent with the standard Hamilton
# multipole formula.  See `docs/clustering_multipole_jacobian_todo.md`
# for the full diagnosis and the fix path.  Until that's done, only
# the monopole estimator below is trusted on real data — the
# quadrupole would need uniform-μ averaging at fixed r, which our
# (r_⊥, r_∥)-binned pair counter does not provide.

def extract_monopole(
    xi_2d: np.ndarray,
    npairs_2d: np.ndarray,
    r_perp_centres: np.ndarray,
    r_par_centres: np.ndarray,
    r_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a 2-D ξ(r_par, r_perp) grid over μ at fixed
    r = sqrt(r_par² + r_perp²) — the monopole.

    Each 2-D bin contributes its npairs-weighted ξ value to the 1-D
    r-bin containing its centre.

    Parameters
    ----------
    xi_2d, npairs_2d : (n_perp, n_par) — 2-D ξ values and pair counts
    r_perp_centres   : (n_perp,) centres
    r_par_centres    : (n_par,) centres (use absolute values; pass folded grid)
    r_bins           : (n_r + 1,) edges in the same units, monotonic

    Returns
    -------
    r_centres : (n_r,) float — bin centres
    xi_mono   : (n_r,) float — monopole-averaged ξ per bin
    npairs    : (n_r,) int   — pair count per 1-D bin
    """
    r_perp_centres = np.asarray(r_perp_centres, dtype=np.float64)
    r_par_centres = np.asarray(r_par_centres, dtype=np.float64)
    xi_2d = np.asarray(xi_2d, dtype=np.float64)
    npairs_2d = np.asarray(npairs_2d, dtype=np.int64)
    r_bins = np.asarray(r_bins, dtype=np.float64)

    if xi_2d.shape != npairs_2d.shape:
        raise ValueError("xi_2d and npairs_2d must have the same shape")
    if xi_2d.shape != (r_perp_centres.size, r_par_centres.size):
        raise ValueError(
            f"xi_2d shape {xi_2d.shape} doesn't match (n_perp, n_par)="
            f"({r_perp_centres.size}, {r_par_centres.size})"
        )
    if not np.all(np.diff(r_bins) > 0):
        raise ValueError("r_bins must be strictly increasing")

    # 2-D r centre per bin
    r_grid = np.sqrt(
        r_perp_centres[:, None] ** 2 + r_par_centres[None, :] ** 2
    )
    # Drop NaN ξ bins (those with 0 pairs)
    valid = npairs_2d > 0
    r_flat = r_grid[valid]
    xi_flat = xi_2d[valid]
    w_flat = npairs_2d[valid].astype(np.float64)

    n_r = r_bins.size - 1
    sum_xi_w = np.zeros(n_r, dtype=np.float64)
    sum_w = np.zeros(n_r, dtype=np.float64)
    n_p = np.zeros(n_r, dtype=np.int64)

    bin_idx = np.searchsorted(r_bins, r_flat, side="right") - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_r)

    np.add.at(sum_xi_w, bin_idx[in_range], (xi_flat * w_flat)[in_range])
    np.add.at(sum_w, bin_idx[in_range], w_flat[in_range])
    np.add.at(n_p, bin_idx[in_range], npairs_2d[valid].astype(np.int64)[in_range])

    with np.errstate(invalid="ignore", divide="ignore"):
        xi_mono = np.where(sum_w > 0, sum_xi_w / sum_w, np.nan)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    return r_centres, xi_mono, n_p


# ---------------------------------------------------------------------------
# b_F from ξ_FF monopole
# ---------------------------------------------------------------------------

@dataclass
class BFFromXiResult:
    b_F: float
    b_F_err: float
    beta_F_assumed: float
    r_centres: np.ndarray
    xi_obs: np.ndarray
    xi_template: np.ndarray
    fit_mask: np.ndarray
    n_fit_bins: int


def fit_b_F_from_xi_FF(
    xi_2d: np.ndarray,
    npairs_2d: np.ndarray,
    r_perp_centres: np.ndarray,
    r_par_centres: np.ndarray,
    k_lin: np.ndarray,
    P_lin: np.ndarray,
    beta_F: float = 1.5,
    r_min: float = 10.0,
    r_max: float = 40.0,
    n_r_bins: int = 12,
) -> BFFromXiResult:
    """Fit b_F from the ξ_FF monopole using

        ξ_FF^(0)(r) = b_F² · K(β_F) · ξ_lin^(0)(r),
        K(β_F) = 1 + (2/3)·β_F + (1/5)·β_F²

    on the linear-scale window [r_min, r_max] (Mpc/h, or matching k_lin
    units).  β_F is fixed at the input value (default 1.5, Slosar+2011
    z ≈ 2.3).

    Returns ``BFFromXiResult`` with b_F (signed, Lyα convention < 0),
    its 1-σ from per-bin scatter, and arrays for the fit panel.
    """
    r_bins = np.linspace(r_min, r_max, n_r_bins + 1)
    r_centres, xi_mono, n_p = extract_monopole(
        xi_2d=xi_2d, npairs_2d=npairs_2d,
        r_perp_centres=r_perp_centres, r_par_centres=r_par_centres,
        r_bins=r_bins,
    )
    xi_lin = xi_lin_monopole(r_centres, k_lin, P_lin)
    K = 1.0 + (2.0 / 3.0) * beta_F + (1.0 / 5.0) * beta_F ** 2
    template = K * xi_lin                                # the ξ_FF / b_F² template

    fit_mask = (
        np.isfinite(xi_mono) & (template != 0) & (n_p > 0)
    )
    if int(fit_mask.sum()) < 4:
        raise ValueError(
            f"only {int(fit_mask.sum())} bins available for the ξ_FF fit "
            f"in [{r_min}, {r_max}] Mpc/h"
        )
    # Ratio fit (geometric mean across bins)
    ratio = xi_mono[fit_mask] / template[fit_mask]
    # Some bins have negative ratio if ξ_obs has a stochastic sign flip at
    # the per-bin level — drop those for the geometric-mean fit; they
    # carry zero info about the bias squared.
    ratio = ratio[ratio > 0]
    if ratio.size < 4:
        raise ValueError(
            "after dropping non-positive ratio bins, only "
            f"{ratio.size} survive — ξ_FF has too much noise"
        )
    log_b_F_sq = float(np.log(ratio).mean())
    log_b_F_sq_se = float(np.log(ratio).std(ddof=1) / np.sqrt(ratio.size))
    b_F_sq = float(np.exp(log_b_F_sq))
    b_F = -np.sqrt(b_F_sq)                                 # Lyα convention
    b_F_err = abs(b_F) * 0.5 * log_b_F_sq_se

    return BFFromXiResult(
        b_F=b_F,
        b_F_err=b_F_err,
        beta_F_assumed=beta_F,
        r_centres=r_centres,
        xi_obs=xi_mono,
        xi_template=b_F_sq * template,
        fit_mask=fit_mask,
        # n_fit_bins reports the COUNT ACTUALLY USED by the
        # geometric-mean fit (after the wrong-sign bin drop), per
        # Copilot review #15 on PR #7.  fit_mask still records the
        # pre-drop window for plotting purposes.
        n_fit_bins=int(ratio.size),
    )


# ---------------------------------------------------------------------------
# Cross-correlation: b_DLA from ξ_× given b_F
# ---------------------------------------------------------------------------

@dataclass
class BDLAFromXiResult:
    b_DLA: float
    b_DLA_err: float
    b_F_assumed: float
    beta_DLA_assumed: float
    beta_F_assumed: float
    K_cross: float
    r_centres: np.ndarray
    xi_obs: np.ndarray
    xi_template: np.ndarray
    fit_mask: np.ndarray
    n_fit_bins: int


def fit_b_DLA_from_xi_cross(
    xi_2d: np.ndarray,
    npairs_2d: np.ndarray,
    r_perp_centres: np.ndarray,
    r_par_centres: np.ndarray,
    k_lin: np.ndarray,
    P_lin: np.ndarray,
    b_F: float,
    beta_DLA: float = 0.5,
    beta_F: float = 1.5,
    r_min: float = 10.0,
    r_max: float = 40.0,
    n_r_bins: int = 12,
) -> BDLAFromXiResult:
    """Fit b_DLA from the DLA × Lyα cross-correlation monopole.

    Cross-Kaiser monopole prefactor (FR+2012, eq. 11):

        ξ_×^(0)(r) = b_DLA · b_F · K_×(β_DLA, β_F) · ξ_lin^(0)(r)
        K_×(β_DLA, β_F) = 1 + (β_DLA + β_F)/3 + β_DLA·β_F/5

    Given the externally-fit `b_F` (from ξ_FF, the production-grade
    estimator) and assumed Kaiser parameters, the only remaining
    unknown is `b_DLA`.  We extract the monopole from the 2-D
    `ξ_×(r_par, r_perp)` grid, divide out `b_F · K_× · ξ_lin`, and
    take the geometric-mean ratio over `r ∈ [r_min, r_max]` Mpc/h.

    Parameters
    ----------
    xi_2d, npairs_2d : (n_perp, n_par) — output of xi_cross_dla_lya
        AFTER folding signed r_par to |r_par|.
    r_perp_centres, r_par_centres : 1-D bin centres for the folded grid.
    k_lin, P_lin : linear matter P at the snap's redshift, units must
        match those of r_min/r_max (i.e. Mpc/h or km/s consistently).
    b_F : externally-known Lyα forest bias (negative).
    beta_DLA : DLA Kaiser β = f/b_DLA.  Default 0.5 corresponds to
        b_DLA ≈ 2 with f ≈ 1 at z ~ 2-4.  Iterate after the first fit
        if needed.
    beta_F : Lyα forest β.  Default 1.5 (Slosar+11 z ≈ 2.3).
    r_min, r_max, n_r_bins : 1-D r-bin edges for the monopole.

    Returns
    -------
    BDLAFromXiResult
    """
    r_bins = np.linspace(r_min, r_max, n_r_bins + 1)
    r_centres, xi_mono, n_p = extract_monopole(
        xi_2d=xi_2d, npairs_2d=npairs_2d,
        r_perp_centres=r_perp_centres, r_par_centres=r_par_centres,
        r_bins=r_bins,
    )
    xi_lin = xi_lin_monopole(r_centres, k_lin, P_lin)
    K_cross = (
        1.0
        + (beta_DLA + beta_F) / 3.0
        + (beta_DLA * beta_F) / 5.0
    )
    template = b_F * K_cross * xi_lin                      # ξ_× / b_DLA

    fit_mask = np.isfinite(xi_mono) & (template != 0) & (n_p > 0)
    if int(fit_mask.sum()) < 4:
        raise ValueError(
            f"only {int(fit_mask.sum())} bins available for ξ_× fit "
            f"in [{r_min}, {r_max}] Mpc/h"
        )
    # Sign expectation: b_F < 0; for r > 0 in the linear regime,
    # ξ_lin > 0; b_DLA > 0 for halos; cross ξ_× should be NEGATIVE
    # (a DLA pulls down the local flux).  Compute ratio and take the
    # geometric-mean of the absolute value, restoring the sign at
    # the end.
    ratio = xi_mono[fit_mask] / template[fit_mask]
    # ratio = b_DLA, with sign convention preserved.  Drop bins with
    # ratio of opposite sign or clearly noisy.
    sign = np.sign(np.median(ratio))
    ratio_use = sign * ratio
    ratio_use = ratio_use[ratio_use > 0]
    if ratio_use.size < 4:
        raise ValueError(
            f"after dropping wrong-sign ratio bins, only {ratio_use.size} "
            "survive — ξ_× is too noisy or b_F sign is inconsistent"
        )
    log_b_DLA = float(np.log(ratio_use).mean())
    log_b_DLA_se = float(np.log(ratio_use).std(ddof=1) / np.sqrt(ratio_use.size))
    b_DLA = float(sign * np.exp(log_b_DLA))
    b_DLA_err = abs(b_DLA) * log_b_DLA_se

    return BDLAFromXiResult(
        b_DLA=b_DLA,
        b_DLA_err=b_DLA_err,
        b_F_assumed=b_F,
        beta_DLA_assumed=beta_DLA,
        beta_F_assumed=beta_F,
        K_cross=K_cross,
        r_centres=r_centres,
        xi_obs=xi_mono,
        xi_template=b_DLA * template,
        fit_mask=fit_mask,
        # n_fit_bins reports the COUNT ACTUALLY USED in the
        # geometric-mean fit (after the wrong-sign bin drop), per
        # Copilot review #1 on PR #7.
        n_fit_bins=int(ratio_use.size),
    )


# ---------------------------------------------------------------------------
# Joint (b_DLA, β_DLA) fit on the (r, μ)-binned ξ_× monopole + quadrupole
# ---------------------------------------------------------------------------
#
# Kaiser cross model in 3-D Fourier space:
#
#   P_×(k, μ) = b_D · b_F · (1 + β_D μ²)(1 + β_F μ²) · P_lin(k)
#             = b_D · b_F · [1 + (β_D + β_F) μ² + β_D β_F μ⁴] · P_lin(k).
#
# Decomposing 1 + (β_D + β_F)μ² + β_D β_F μ⁴ in the Legendre basis
# (using μ² = 1/3 + 2/3·L_2 and μ⁴ = 1/5 + 4/7·L_2 + 8/35·L_4) yields:
#
#   P^(0)(k) = b_D b_F · K_0 · P_lin(k),    K_0 = 1 + (β_D+β_F)/3 + β_D β_F/5
#   P^(2)(k) = b_D b_F · K_2 · P_lin(k),    K_2 = (2/3)(β_D+β_F) + (4/7) β_D β_F
#   P^(4)(k) = b_D b_F · (8/35) β_D β_F · P_lin(k)            (small, ignored)
#
# Hamilton 1992 transform:
#
#   ξ^(ℓ)(r) = i^ℓ / (2π²) · ∫ k² dk · j_ℓ(k r) · P^(ℓ)(k).
#
# i^0 = 1, i^2 = -1.  Define the j_ℓ-transform of P_lin as ξ_lin^(jℓ)(r)
# (positive-area-weighted at small r).  Then:
#
#   ξ_×^(0)(r) = + b_D · b_F · K_0 · ξ_lin^(j0)(r)
#   ξ_×^(2)(r) = − b_D · b_F · K_2 · ξ_lin^(j2)(r)            ← note minus sign
#
# At fixed (b_F, β_F), this is a 2-parameter (b_D, β_D) least-squares
# fit on the joint mono + quad data vectors.

@dataclass
class JointBDLABetaResult:
    """Output of ``fit_b_beta_from_xi_cross_multipoles``."""
    b_DLA: float
    b_DLA_err: float
    beta_DLA: float
    beta_DLA_err: float
    b_F_assumed: float
    beta_F_assumed: float
    K_0: float
    K_2: float
    r_centres: np.ndarray
    xi_mono_obs: np.ndarray
    xi_quad_obs: np.ndarray
    xi_mono_template: np.ndarray
    xi_quad_template: np.ndarray
    fit_mask: np.ndarray
    n_fit_bins: int
    chi2: float
    cov: np.ndarray         # 2x2 covariance of (b_DLA, β_DLA)


def _kaiser_K0_K2(beta_D: float, beta_F: float) -> Tuple[float, float]:
    """Kaiser monopole + quadrupole prefactors for a (b_D · b_F)-cross."""
    K0 = 1.0 + (beta_D + beta_F) / 3.0 + beta_D * beta_F / 5.0
    K2 = (2.0 / 3.0) * (beta_D + beta_F) + (4.0 / 7.0) * beta_D * beta_F
    return K0, K2


def fit_b_beta_from_xi_cross_multipoles(
    xi_rmu: np.ndarray,
    npairs_rmu: np.ndarray,
    r_centres: np.ndarray,
    mu_centres: np.ndarray,
    k_lin: np.ndarray,
    P_lin: np.ndarray,
    b_F: float,
    beta_F: float = 1.5,
    r_min: float = 10.0,
    r_max: float = 40.0,
    b_DLA_init: float = 2.0,
    beta_DLA_init: float = 0.5,
) -> JointBDLABetaResult:
    """Joint (b_DLA, β_DLA) fit using the ξ_× monopole + quadrupole.

    Consumes a (r, |μ|)-binned ξ_× grid (e.g. from
    ``hcd_analysis.clustering.xi_cross_dla_lya_rmu``), extracts the
    monopole + quadrupole via Hamilton uniform-μ averaging (the
    Jacobian-corrected estimator — see
    ``docs/clustering_multipole_jacobian_todo.md``), and fits the
    Kaiser cross template

        ξ^(0)(r) = + b_D · b_F · K_0(β_D, β_F) · ξ_lin^(j0)(r)
        ξ^(2)(r) = − b_D · b_F · K_2(β_D, β_F) · ξ_lin^(j2)(r).

    Two free parameters (b_DLA, β_DLA), fit by Levenberg-Marquardt on
    the concatenated mono/quad residuals.  Weights are derived from
    the per-bin Poisson variance propagated through the Hamilton
    projection:

        Var(ξ^(ℓ)(r)) ∝ (2ℓ+1)² · Σ_j (L_ℓ(μ_j)·Δμ_j)² / N_ij

    so the quadrupole carries the (2ℓ+1)² = 25 amplification correctly
    relative to the monopole, instead of being weighted equally with
    a single √Σ_j N_ij as the original implementation did.  The overall
    Poisson normalisation is unknown and absorbs into the χ²/dof
    rescaling on the covariance below.

    Parameters
    ----------
    xi_rmu, npairs_rmu : (n_r, n_mu) — output of xi_cross_dla_lya_rmu
    r_centres : (n_r,) bin centres in Mpc/h
    mu_centres : (n_mu,) bin centres on [0, 1]
    k_lin, P_lin : linear matter spectrum, units must match r_centres
        (Mpc/h ↔ h/Mpc, or km/s ↔ s/km)
    b_F, beta_F : externally calibrated Lyα forest bias and Kaiser β
    r_min, r_max : fit window in the same units as r_centres
    b_DLA_init, beta_DLA_init : initial guess for the optimiser

    Returns
    -------
    JointBDLABetaResult
    """
    from scipy.optimize import least_squares      # local import — heavy

    xi_rmu = np.asarray(xi_rmu, dtype=np.float64)
    npairs_rmu = np.asarray(npairs_rmu, dtype=np.int64)
    r_centres = np.asarray(r_centres, dtype=np.float64)
    mu_centres = np.asarray(mu_centres, dtype=np.float64)

    if xi_rmu.shape != (r_centres.size, mu_centres.size):
        raise ValueError(
            f"xi_rmu shape {xi_rmu.shape} doesn't match (n_r, n_mu)="
            f"({r_centres.size}, {mu_centres.size})"
        )

    multipoles = extract_multipoles_rmu(
        xi_rmu=xi_rmu, npairs_rmu=npairs_rmu,
        mu_centres=mu_centres, ells=(0, 2),
    )
    xi_mono = multipoles[0]
    xi_quad = multipoles[2]

    # Per-r-bin pair count, summed over μ.  Sets the residual weights.
    npairs_per_r = npairs_rmu.sum(axis=1)

    fit_mask = (
        np.isfinite(xi_mono) & np.isfinite(xi_quad)
        & (r_centres >= r_min) & (r_centres <= r_max)
        & (npairs_per_r > 0)
    )
    n_fit = int(fit_mask.sum())
    if n_fit < 4:
        raise ValueError(
            f"only {n_fit} valid r-bins in [{r_min}, {r_max}]; "
            f"need ≥ 4 for a 2-parameter fit"
        )

    r_fit = r_centres[fit_mask]
    xi0_fit = xi_mono[fit_mask]
    xi2_fit = xi_quad[fit_mask]

    # Per-bin Poisson variance propagated through the Hamilton projection.
    # σ²(r, ℓ) ∝ (2ℓ+1)² · Σ_j (L_ℓ(μ_j) · Δμ_j)² / N_ij; the overall
    # constant is unknown and is absorbed by the χ²/dof rescaling below.
    mu_edges_inferred = np.empty(mu_centres.size + 1, dtype=np.float64)
    mu_edges_inferred[1:-1] = 0.5 * (mu_centres[:-1] + mu_centres[1:])
    mu_edges_inferred[0] = max(0.0, 2.0 * mu_centres[0] - mu_edges_inferred[1])
    mu_edges_inferred[-1] = min(1.0, 2.0 * mu_centres[-1] - mu_edges_inferred[-2])
    dmu = np.diff(mu_edges_inferred)                                # (n_mu,)
    L0_mu = np.ones_like(mu_centres)
    L2_mu = 0.5 * (3.0 * mu_centres ** 2 - 1.0)
    inv_N = np.where(npairs_rmu > 0,
                     1.0 / np.maximum(npairs_rmu, 1).astype(np.float64),
                     0.0)
    var0_per_r = ((L0_mu * dmu) ** 2 * inv_N).sum(axis=1)           # ∝ σ²(r, 0)
    var2_per_r = (5.0 ** 2) * ((L2_mu * dmu) ** 2 * inv_N).sum(axis=1)  # ∝ σ²(r, 2)
    eps = 1e-30
    w0 = 1.0 / np.sqrt(np.maximum(var0_per_r[fit_mask], eps))
    w2 = 1.0 / np.sqrt(np.maximum(var2_per_r[fit_mask], eps))

    xi_lin_j0 = xi_lin_monopole(r_fit, k_lin, P_lin)
    xi_lin_j2 = xi_lin_quadrupole(r_fit, k_lin, P_lin)

    def residuals(params):
        b_D, beta_D = params
        K0, K2 = _kaiser_K0_K2(beta_D, beta_F)
        model_0 = +b_D * b_F * K0 * xi_lin_j0
        model_2 = -b_D * b_F * K2 * xi_lin_j2
        return np.concatenate([
            (xi0_fit - model_0) * w0,
            (xi2_fit - model_2) * w2,
        ])

    res = least_squares(
        residuals,
        x0=np.array([b_DLA_init, beta_DLA_init]),
        method="lm",
        max_nfev=200,
    )
    b_DLA, beta_DLA = float(res.x[0]), float(res.x[1])
    K0_fit, K2_fit = _kaiser_K0_K2(beta_DLA, beta_F)

    # Covariance from the LM Jacobian: cov = (J^T J)^{-1} · σ²,
    # where σ² is the residual variance per d.o.f.
    J = res.jac                                                    # (2N, 2)
    if J.size > 0 and n_fit > 1:
        # Residuals are already √npairs-weighted, so ignore the σ² term
        # if you trust the per-bin weighting; here we take the χ²-per-dof
        # rescaling so the reported errors are conservatively scaled to
        # the post-fit residuals.
        chi2 = float((res.fun ** 2).sum())
        dof = max(1, 2 * n_fit - 2)
        sigma_sq = chi2 / dof
        try:
            JtJ_inv = np.linalg.inv(J.T @ J)
            cov = JtJ_inv * sigma_sq
        except np.linalg.LinAlgError:
            cov = np.full((2, 2), np.nan)
    else:
        chi2 = float("nan")
        cov = np.full((2, 2), np.nan)

    b_DLA_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else float("nan")
    beta_DLA_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else float("nan")

    # Templates over the full r grid (for plotting).
    xi_lin_j0_full = xi_lin_monopole(r_centres, k_lin, P_lin)
    xi_lin_j2_full = xi_lin_quadrupole(r_centres, k_lin, P_lin)
    template_mono = +b_DLA * b_F * K0_fit * xi_lin_j0_full
    template_quad = -b_DLA * b_F * K2_fit * xi_lin_j2_full

    return JointBDLABetaResult(
        b_DLA=b_DLA,
        b_DLA_err=b_DLA_err,
        beta_DLA=beta_DLA,
        beta_DLA_err=beta_DLA_err,
        b_F_assumed=float(b_F),
        beta_F_assumed=float(beta_F),
        K_0=float(K0_fit),
        K_2=float(K2_fit),
        r_centres=r_centres,
        xi_mono_obs=xi_mono,
        xi_quad_obs=xi_quad,
        xi_mono_template=template_mono,
        xi_quad_template=template_quad,
        fit_mask=fit_mask,
        n_fit_bins=n_fit,
        chi2=chi2,
        cov=cov,
    )


__all__ = [
    "BDLAFromXiResult",
    "BFFitResult",
    "BFFromXiResult",
    "JointBDLABetaResult",
    "compute_p1d_clean_sightlines",
    "extract_monopole",
    "extract_multipoles_rmu",
    "find_camb_pk_for_z",
    "fit_b_DLA_from_xi_cross",
    "fit_b_F",
    "fit_b_F_from_xi_FF",
    "fit_b_beta_from_xi_cross_multipoles",
    "hMpc_to_kms_factor",
    "load_camb_pk",
    "project_pk_3d_to_p1d",
    "xi_lin_monopole",
    "xi_lin_quadrupole",
]
