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

        k_v  [s/km]    = k_r [h/Mpc]   * F
        L_v  [km/s]    = L_r [Mpc/h]   / F
        P_v  [(km/s)^d] = P_r [(Mpc/h)^d] * F^d

    Derivation: a comoving distance L [Mpc/h] in the LOS direction
    corresponds to a redshift-space velocity span

        Δv = a · H(z) · L_proper = (1/(1+z)) · H(z) · (L[Mpc/h] / h).

    Hence  L[km/s] = L[Mpc/h] · H(z) / (h · (1+z))   = L[Mpc/h] / F⁻¹
    with  F = h · (1+z) / H(z).
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
        valid = (k > k_3d.min()) & (k < k_3d.max())
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
        out[i] = np.trapezoid(integrand, k_perp) / (2.0 * np.pi)
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
    P_3d_kms = P_Mpch3 * F ** 3                            # (km/s)^3

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


__all__ = [
    "BFFitResult",
    "compute_p1d_clean_sightlines",
    "find_camb_pk_for_z",
    "fit_b_F",
    "hMpc_to_kms_factor",
    "load_camb_pk",
    "project_pk_3d_to_p1d",
]
