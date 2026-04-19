"""
Column Density Distribution Function (CDDF) and continuous perturbation model.

Definition
----------
The CDDF is defined as:

    f(N_HI, X) = d^2 n / (dN_HI dX)

where:
  N_HI : HI column density (cm^-2)
  X    : absorption path length (dimensionless)
  n    : number of absorbers per sightline

Absorption path length:
    dX/dz = (H_0 / H(z)) * (1 + z)^2

For a flat ΛCDM cosmology:
    H(z)/H_0 = sqrt(Omega_m * (1+z)^3 + Omega_Lambda)

For a simulation box at redshift z, the total absorption path per sightline is:
    Delta_X = (H_0 / H(z)) * (1+z)^2 * Delta_z_box

where Delta_z_box is the redshift interval spanned by the box:
    Delta_z_box ≈ H(z) * L_phys / c  (narrow box)
              = H(z) * (box_kpc/h / 1000 / h) / (1+z) / c_kms  km/s / (km/s)

This simplifies to:
    Delta_X = (1+z)^2 * L_phys_Mpc * H_0_kms / c_kms
            = (1+z)^2 * (box/1000/h) / (1+z) * H_0/c  (dimensionless)
            = (1+z) * (box/1000/h) * H_0/c

We measure CDDF from the simulation directly:
    f(N_HI) = n_absorbers(N_HI in bin) / (dN_HI * n_sightlines * Delta_X)

Perturbation model
------------------
We define a continuous multiplicative perturbation:

    f'(N) = A * f(N) * (N / N_pivot)^alpha

Parameters:
  A       : overall amplitude modifier (A=1, alpha=0 = no change)
  alpha   : power-law tilt
  N_pivot : pivot column density (cm^-2), default 10^20 cm^-2

To propagate this perturbation to P1D, we use importance reweighting:
  For each absorber i with column density N_i, assign weight:
    w_i = A * (N_i / N_pivot)^alpha

Then recompute masked P1D where absorbers are selected/weighted by w_i.

Two approaches are implemented:
1. Discrete resampling: draw a new absorber set from the weighted distribution.
   Used for MC error estimation.
2. Direct reweighting (deterministic): scale each absorber's contribution
   to the masked P1D by its weight. For this we compute dP1D/d_absorber
   contributions numerically.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .catalog import AbsorberCatalog, Absorber, classify_system

# Standard redshift bins for stacking across snapshots
_DEFAULT_Z_BINS = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

logger = logging.getLogger(__name__)

# H_0 = 100 km/s/Mpc (h=1 units)
_H0_KMS_MPC = 100.0
_C_KMS = 2.99792458e5


# ---------------------------------------------------------------------------
# Absorption path length
# ---------------------------------------------------------------------------

def absorption_path_per_sightline(
    box_kpc_h: float,
    hubble: float,
    omegam: float,
    omegal: float,
    z: float,
) -> float:
    """
    Compute dX per sightline for a simulation box at redshift z.

    dX = (H_0/H(z)) * (1+z)^2 * dz_box
    dz_box = H(z)/c * L_phys
           = H(z)/c * (box_kpc_h/1000/h) / (1+z)  [Mpc comoving → Mpc physical]
    → dX = (1+z) * (box_kpc_h/1000/h) * H_0/c
    """
    box_mpc = box_kpc_h / 1000.0 / hubble  # comoving Mpc
    dX = (1.0 + z) * box_mpc * (_H0_KMS_MPC / _C_KMS)
    return dX


def hz_from_cosmology(omegam: float, omegal: float, z: float) -> float:
    """H(z)/H_0 (dimensionless) for flat ΛCDM."""
    return np.sqrt(omegam * (1.0 + z)**3 + omegal)


# ---------------------------------------------------------------------------
# CDDF measurement
# ---------------------------------------------------------------------------

def measure_cddf(
    catalog: AbsorberCatalog,
    header,
    log_nhi_bins: Optional[np.ndarray] = None,
    absorber_class_filter: Optional[List[str]] = None,
) -> Dict:
    """
    Measure f(N_HI, X) from an AbsorberCatalog.

    Parameters
    ----------
    catalog      : AbsorberCatalog for one (sim, snap)
    header       : SpectraHeader (for box, hubble, omegam, omegal, n_skewers)
    log_nhi_bins : bin edges in log10(NHI). Default: 17 to 23, 30 bins.
    absorber_class_filter : if set, only count these classes

    Returns
    -------
    dict with keys:
      log_nhi_centres : bin centres (log10 NHI)
      log_nhi_edges   : bin edges
      f_nhi           : f(N_HI) values (cm^2)
      n_absorbers     : raw counts per bin
      dX              : total absorption path
    """
    if log_nhi_bins is None:
        log_nhi_bins = np.linspace(17.0, 23.0, 31)

    dX = absorption_path_per_sightline(
        header.box, header.hubble, header.omegam, header.omegal, header.redshift
    )
    total_path = header.n_skewers * dX

    absorbers = catalog.absorbers
    if absorber_class_filter:
        absorbers = [a for a in absorbers if a.absorber_class in absorber_class_filter]

    log_nhi_vals = np.array([a.log_NHI for a in absorbers])

    counts, edges = np.histogram(log_nhi_vals, bins=log_nhi_bins)
    dlogN = np.diff(edges)
    dN = 10.0**edges[1:] - 10.0**edges[:-1]  # actual dN in cm^-2

    with np.errstate(divide="ignore", invalid="ignore"):
        f_nhi = np.where(dN > 0, counts / (dN * total_path), 0.0)

    centres = 0.5 * (edges[:-1] + edges[1:])

    return {
        "log_nhi_centres": centres,
        "log_nhi_edges": edges,
        "f_nhi": f_nhi,
        "n_absorbers": counts,
        "dX_per_sightline": dX,
        "total_path": total_path,
        "n_sightlines": header.n_skewers,
        "z": header.redshift,
    }


# ---------------------------------------------------------------------------
# CDDF Perturbation model
# ---------------------------------------------------------------------------

class CDDFPerturbation:
    """
    Continuous multiplicative CDDF perturbation:
        f'(N) = A * f(N) * (N / N_pivot)^alpha

    Used to reweight absorbers when computing perturbed P1D.
    """

    def __init__(self, A: float = 1.0, alpha: float = 0.0, N_pivot: float = 1.0e20):
        self.A = A
        self.alpha = alpha
        self.N_pivot = N_pivot

    def weight(self, NHI: float) -> float:
        """Multiplicative weight for a single absorber."""
        return self.A * (NHI / self.N_pivot) ** self.alpha

    def weights_array(self, NHI_arr: np.ndarray) -> np.ndarray:
        """Multiplicative weights for an array of NHI values."""
        return self.A * (NHI_arr / self.N_pivot) ** self.alpha

    def perturbed_f_nhi(self, cddf_result: Dict) -> np.ndarray:
        """Apply the perturbation to a measured CDDF and return perturbed f(N)."""
        N_centres = 10.0 ** cddf_result["log_nhi_centres"]
        return cddf_result["f_nhi"] * self.weights_array(N_centres)

    def summary(self) -> str:
        return f"CDDFPerturbation(A={self.A}, alpha={self.alpha}, N_pivot={self.N_pivot:.2e})"


# ---------------------------------------------------------------------------
# Perturbation → P1D effect
# ---------------------------------------------------------------------------

def perturbed_mask_classes(
    catalog: AbsorberCatalog,
    perturbation: CDDFPerturbation,
    base_classes: List[str],
    n_realizations: int = 1,
    rng_seed: int = 42,
) -> List[List[Absorber]]:
    """
    Generate resampled absorber lists under the perturbed CDDF.

    For each base-class absorber with NHI = N_i, its inclusion probability is
    proportional to w_i = A * (N_i/N_pivot)^alpha relative to w=1 (unperturbed).

    Specifically, we implement this as:
      - Compute w_i for each absorber
      - Accept absorber i with probability min(1, w_i) if A*(ratio)^alpha < 1
        or include it with frequency ~ w_i if > 1.
      - For fractional additions, use Poisson sampling.

    Parameters
    ----------
    catalog         : base catalog
    perturbation    : CDDFPerturbation instance
    base_classes    : which absorber classes to perturb
    n_realizations  : number of Monte Carlo resamples to return
    rng_seed        : random seed for reproducibility

    Returns
    -------
    List of n_realizations absorber lists, each ready to be used as
    the mask absorbers in compute_p1d_single().
    """
    rng = np.random.default_rng(rng_seed)

    base_absorbers = [a for a in catalog.absorbers if a.absorber_class in base_classes]
    if not base_absorbers:
        return [[] for _ in range(n_realizations)]

    NHI_arr = np.array([a.NHI for a in base_absorbers])
    weights = perturbation.weights_array(NHI_arr)

    realizations = []
    for _ in range(n_realizations):
        # Poisson resampling: expected count for absorber i = w_i
        # For w_i < 1: include with probability w_i
        # For w_i > 1: include floor(w_i) times + Bernoulli for remainder
        n_copies = rng.poisson(weights)
        selected = []
        for abs_obj, n_cp in zip(base_absorbers, n_copies):
            selected.extend([abs_obj] * int(n_cp))
        realizations.append(selected)

    return realizations


# ---------------------------------------------------------------------------
# DataFrame-compatible CDDF (used by pipeline when catalog is a pandas DF)
# ---------------------------------------------------------------------------

def measure_cddf_from_dataframe(
    df,
    z: float,
    box_kpc_h: float,
    hubble: float,
    omegam: float,
    omegal: float,
    n_sightlines: int,
    log_nhi_bins: Optional[np.ndarray] = None,
    absorber_class_filter: Optional[List[str]] = None,
) -> Dict:
    """
    Measure CDDF from a pandas DataFrame with columns 'log_nhi' and 'absorber_class'.

    Parameters
    ----------
    df        : DataFrame with at least 'log_nhi' (and optionally 'absorber_class')
    z         : redshift of the snapshot
    ...       : box/cosmology parameters for absorption path
    n_sightlines : total number of sightlines in the file
    """
    if log_nhi_bins is None:
        log_nhi_bins = np.linspace(17.0, 23.0, 31)

    dX = absorption_path_per_sightline(box_kpc_h, hubble, omegam, omegal, z)
    total_path = n_sightlines * dX

    if absorber_class_filter is not None and "absorber_class" in df.columns:
        df = df[df["absorber_class"].isin(absorber_class_filter)]

    log_nhi_vals = np.asarray(df["log_nhi"])
    counts, edges = np.histogram(log_nhi_vals, bins=log_nhi_bins)
    dN = 10.0**edges[1:] - 10.0**edges[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_nhi = np.where(dN > 0, counts / (dN * total_path), 0.0)

    centres = 0.5 * (edges[:-1] + edges[1:])
    return {
        "log_nhi_centres": centres,
        "log_nhi_edges": edges,
        "f_nhi": f_nhi,
        "n_absorbers": counts,
        "dX_per_sightline": dX,
        "total_path": total_path,
        "n_sightlines": n_sightlines,
        "z": z,
    }


# ---------------------------------------------------------------------------
# Per-sim CDDF stacking with separate redshift bins
# ---------------------------------------------------------------------------

def stack_cddf_for_sim(
    cddf_list: List[Dict],
    z_bins: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Stack per-snapshot CDDFs into per-redshift-bin CDDFs for a single simulation.

    Parameters
    ----------
    cddf_list : list of dicts returned by measure_cddf() or measure_cddf_from_dataframe()
                Each dict must have 'z', 'f_nhi', 'n_absorbers', 'total_path',
                'log_nhi_centres', 'log_nhi_edges'.
    z_bins    : edges of redshift bins. Default: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    Returns
    -------
    dict mapping bin label (e.g. "z2.0-2.5") → stacked CDDF dict:
      {
        "z_min", "z_max", "z_snapshots",
        "f_nhi"          : counts_total / (dN * total_path_total),
        "f_nhi_per_snap" : array of per-snap f_nhi (for scatter),
        "n_absorbers"    : summed counts,
        "total_path"     : summed absorption path,
        "log_nhi_centres", "log_nhi_edges",
      }
    """
    if z_bins is None:
        z_bins = _DEFAULT_Z_BINS

    # Group snapshots into bins
    bins: Dict[str, List[Dict]] = {}
    for cddf in cddf_list:
        z_snap = cddf["z"]
        for i in range(len(z_bins) - 1):
            if z_bins[i] <= z_snap < z_bins[i + 1]:
                label = f"z{z_bins[i]:.1f}-{z_bins[i+1]:.1f}"
                bins.setdefault(label, []).append((z_bins[i], z_bins[i + 1], cddf))
                break

    result = {}
    for label, entries in bins.items():
        z_min = entries[0][0]
        z_max = entries[0][1]
        cddfs = [e[2] for e in entries]

        log_nhi_centres = cddfs[0]["log_nhi_centres"]
        log_nhi_edges = cddfs[0]["log_nhi_edges"]

        counts_total = np.zeros(len(log_nhi_centres), dtype=np.float64)
        path_total = 0.0
        z_snaps = []
        f_per_snap = []

        for c in cddfs:
            counts_total += c["n_absorbers"].astype(np.float64)
            path_total += c["total_path"]
            z_snaps.append(c["z"])
            f_per_snap.append(c["f_nhi"])

        dN = 10.0**log_nhi_edges[1:] - 10.0**log_nhi_edges[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            f_stacked = np.where(dN > 0, counts_total / (dN * path_total), 0.0)

        result[label] = {
            "z_min": z_min,
            "z_max": z_max,
            "z_snapshots": sorted(z_snaps),
            "f_nhi": f_stacked,
            "f_nhi_per_snap": np.array(f_per_snap),
            "n_absorbers": counts_total.astype(int),
            "total_path": path_total,
            "log_nhi_centres": log_nhi_centres,
            "log_nhi_edges": log_nhi_edges,
        }

    return result


def compute_perturbed_p1d(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog: AbsorberCatalog,
    perturbation: CDDFPerturbation,
    base_mask_classes: List[str],
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    fill_strategy: str = "mean_flux",
    k_bins: Optional[np.ndarray] = None,
    n_realizations: int = 1,
    rng_seed: int = 42,
) -> Dict:
    """
    Compute P1D under a perturbed CDDF.

    Creates a modified catalog (perturbed absorber population via Poisson
    resampling) and recomputes the masked P1D.

    Returns
    -------
    dict with keys:
      k           : k values (s/km)
      p1d_mean    : mean P1D over realizations (km/s)
      p1d_std     : std of P1D over realizations
      p1d_base    : unperturbed baseline P1D
      ratio_mean  : p1d_mean / p1d_base
    """
    from .p1d import compute_p1d_single
    from .catalog import AbsorberCatalog as AC

    # Compute baseline P1D (unperturbed mask)
    k, p1d_base, _ = compute_p1d_single(
        hdf5_path, nbins, dv_kms, catalog=catalog,
        mask_classes=base_mask_classes, batch_size=batch_size,
        n_skewers=n_skewers, fill_strategy=fill_strategy, k_bins=k_bins,
    )

    # Generate perturbed realizations
    realizations = perturbed_mask_classes(
        catalog, perturbation, base_mask_classes, n_realizations, rng_seed
    )

    p1d_realizations = []
    for i, abs_list in enumerate(realizations):
        # Build a temporary catalog with perturbed absorbers
        tmp_cat = AC(
            sim_name=catalog.sim_name,
            snap=catalog.snap,
            z=catalog.z,
            dv_kms=catalog.dv_kms,
            absorbers=abs_list,
        )
        _, p1d_r, _ = compute_p1d_single(
            hdf5_path, nbins, dv_kms, catalog=tmp_cat,
            mask_classes=base_mask_classes, batch_size=batch_size,
            n_skewers=n_skewers, fill_strategy=fill_strategy, k_bins=k_bins,
        )
        p1d_realizations.append(p1d_r)

    p1d_stack = np.array(p1d_realizations)
    p1d_mean = p1d_stack.mean(axis=0)
    p1d_std = p1d_stack.std(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mean = np.where(p1d_base > 0, p1d_mean / p1d_base, np.nan)

    return {
        "k": k,
        "p1d_base": p1d_base,
        "p1d_mean": p1d_mean,
        "p1d_std": p1d_std,
        "ratio_mean": ratio_mean,
        "perturbation": perturbation,
        "n_realizations": n_realizations,
    }
