"""
1D power spectrum P1D(k) computation.

Definition
----------
Let F(v) = exp(-tau(v)) be the transmitted flux along a sightline.
Let delta_F(v) = F(v) / <F> - 1 be the fractional flux fluctuation,
where <F> is the mean transmission averaged over all skewers at this redshift.

Then the 1D flux power spectrum is defined as:

    P1D(k) = (1 / L) * | integral delta_F(v) exp(-2*pi*i*k*v) dv |^2

with:
    k in units of s/km  (inverse velocity)
    v in units of km/s
    L = N_pix * dv  (box length in km/s along LOS)

In discrete Fourier convention (DFT):

    tilde_F[n] = sum_{j=0}^{N-1} delta_F[j] * exp(-2*pi*i*n*j/N) * dv

    P1D[n]  = (dv / (N * dv))^2 * |tilde_F[n]|^2 * N * dv
            = dv^2 / (N * dv) * |DFT(delta_F)[n]|^2
            = dv / N * |DFT(delta_F)[n]|^2

k modes:
    k_n = n / (N * dv)   for n = 0, ..., N/2
    k_min = 1 / (N * dv)
    k_max = 1 / (2 * dv)  (Nyquist)

This is consistent with the Lyman-alpha forest P1D convention used in the
literature (e.g. Palanque-Delabrouille et al. 2013, Chabanier et al. 2019).

Normalisation check:
    integral P1D(k) dk = <delta_F^2>  (Parseval)

Units:
    [k] = s/km
    [P1D] = (km/s)

Variants computed
-----------------
  "all"      : all absorbers included (baseline)
  "no_LLS"   : LLS pixels masked
  "no_subDLA": subDLA pixels masked
  "no_DLA"   : DLA pixels masked
  "no_HCD"   : all of LLS+subDLA+DLA masked (= forest only)
  "only_LLS" : all masked except LLS   (i.e. only LLS contribution)
  "only_subDLA", "only_DLA"  : analogous

Ratios:
  P1D_noDLA / P1D_all
  P1D_noHCD / P1D_all
  etc. — see compute_p1d_ratios()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default k bins matching the emulator's kf grid (s/km)
_DEFAULT_K_BINS = np.array([
    0.001084, 0.001626, 0.002168, 0.00271,  0.003252, 0.003794,
    0.004336, 0.004878, 0.00542,  0.005962, 0.006504, 0.007046,
    0.007588, 0.00813,  0.008672, 0.009214, 0.009756, 0.010298,
    0.01084,  0.011382, 0.011924, 0.012466, 0.013008, 0.01355,
    0.014092, 0.014634, 0.015176, 0.015718, 0.01626,  0.016802,
    0.017344, 0.017886, 0.018428, 0.01897,  0.019512,
])


# ---------------------------------------------------------------------------
# Core P1D accumulator (streaming, memory-efficient)
# ---------------------------------------------------------------------------

class P1DAccumulator:
    """
    Accumulates P1D from skewer batches without storing all tau in memory.

    Usage:
        acc = P1DAccumulator(nbins, dv_kms)
        for batch in iter_batches(...):
            acc.add_batch(batch)
        k, p1d = acc.result(k_bins)
    """

    def __init__(self, nbins: int, dv_kms: float):
        self.nbins = nbins
        self.dv_kms = dv_kms
        self._power_sum = np.zeros(nbins // 2 + 1, dtype=np.float64)
        self._n_skewers = 0
        self._flux_sum = 0.0   # sum of F values, for mean flux
        self._flux_n = 0

    def _k_native(self) -> np.ndarray:
        """Native k modes of the FFT (s/km)."""
        return np.fft.rfftfreq(self.nbins, d=self.dv_kms)

    def add_batch(self, tau_batch: np.ndarray, mean_F_global: Optional[float] = None) -> None:
        """
        Add a batch of skewers.

        If mean_F_global is provided, use it for normalisation. Otherwise,
        accumulate flux values and use the per-batch mean (two-pass mode if
        mean_F_global is None — see add_batch_two_pass for proper usage).

        Parameters
        ----------
        tau_batch    : (batch_size, nbins) float64
        mean_F_global: pre-computed global <F> to normalise delta_F
        """
        F_batch = np.exp(-tau_batch)  # shape (batch, nbins)

        if mean_F_global is None:
            mean_F = F_batch.mean()
        else:
            mean_F = mean_F_global

        if mean_F < 1e-30:
            mean_F = 1e-30

        delta_F = F_batch / mean_F - 1.0  # fractional fluctuation

        # DFT along pixel axis; rfft gives one-sided spectrum
        ft = np.fft.rfft(delta_F * self.dv_kms, axis=1)  # units: km/s

        # Power per mode
        power = np.abs(ft)**2 / (self.nbins * self.dv_kms)  # units: km/s

        # Sum over skewers in this batch
        self._power_sum += power.sum(axis=0)
        self._n_skewers += tau_batch.shape[0]
        self._flux_sum += F_batch.sum()
        self._flux_n += tau_batch.size

    @property
    def mean_flux(self) -> float:
        if self._flux_n == 0:
            return 1.0
        return self._flux_sum / self._flux_n

    @property
    def n_skewers(self) -> int:
        return self._n_skewers

    def raw_power(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (k_native, p1d_native) averaged over all accumulated skewers."""
        if self._n_skewers == 0:
            k = self._k_native()
            return k, np.zeros_like(k)
        k = self._k_native()
        p1d = self._power_sum / self._n_skewers
        return k, p1d

    def result(
        self,
        k_bins: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (k_centres, P1D) binned onto k_bins.

        k_bins: bin edges in s/km. If None, uses the default emulator kf grid
                (interpreted as bin centres → construct edges from midpoints).
        """
        k_native, p1d_native = self.raw_power()

        if k_bins is None:
            k_centres = _DEFAULT_K_BINS
            # Build edges from midpoints
            edges = _bin_centres_to_edges(k_centres)
        else:
            k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
            edges = k_bins

        p1d_binned = _bin_power(k_native, p1d_native, edges)
        return k_centres, p1d_binned


def _bin_centres_to_edges(centres: np.ndarray) -> np.ndarray:
    """Convert bin centres to edges by linear midpoint extrapolation."""
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - (edges[1] - centres[0])
    edges[-1] = centres[-1] + (centres[-1] - edges[-2])
    return edges


def _bin_power(k: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average power in each bin defined by edges."""
    n_bins = len(edges) - 1
    result = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (k >= edges[i]) & (k < edges[i + 1])
        if sel.any():
            result[i] = power[sel].mean()
    return result


# ---------------------------------------------------------------------------
# Two-pass mean-flux normalisation
# ---------------------------------------------------------------------------

def compute_mean_flux(hdf5_path, batch_size: int = 4096, n_skewers=None) -> float:
    """
    First pass: compute global mean F = <exp(-tau)> over all skewers.
    This is needed for correct delta_F normalisation.
    """
    from .io import iter_tau_batches

    flux_sum = 0.0
    n_total = 0
    for _, _, tau_batch in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers):
        flux_sum += np.exp(-tau_batch.astype(np.float64)).sum()
        n_total += tau_batch.size
    return flux_sum / n_total if n_total > 0 else 1.0


def compute_mean_flux_from_masked_iter(tau_iter) -> float:
    """Compute mean F from an iterator that yields (start, end, tau_batch)."""
    flux_sum = 0.0
    n_total = 0
    for _, _, tau_batch in tau_iter:
        flux_sum += np.exp(-tau_batch).sum()
        n_total += tau_batch.size
    return flux_sum / n_total if n_total > 0 else 1.0


# ---------------------------------------------------------------------------
# High-level: compute P1D for a single (hdf5_path, catalog, mask_classes) triplet
# ---------------------------------------------------------------------------

def compute_p1d_single(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog=None,
    mask_classes: Optional[List[str]] = None,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    fill_strategy: str = "mean_flux",
    k_bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute P1D for one variant (all, or with specified classes masked).

    Returns
    -------
    k_centres : np.ndarray (s/km)
    p1d       : np.ndarray (km/s)
    mean_F    : float (mean flux used for normalisation)
    """
    from .io import iter_tau_batches
    from .masking import iter_masked_batches

    needs_masking = bool(mask_classes and catalog is not None)

    def _make_iter():
        if needs_masking:
            return iter_masked_batches(
                hdf5_path, catalog, mask_classes,
                batch_size=batch_size, n_skewers=n_skewers, strategy=fill_strategy,
            )
        return iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers)

    # Pass 1 (streaming): compute global mean_F without storing tau in memory.
    F_sum = 0.0
    F_n = 0
    for _, _, tau_batch in _make_iter():
        F_sum += np.exp(-tau_batch.astype(np.float64)).sum()
        F_n += tau_batch.size
    mean_F_global = F_sum / F_n if F_n > 0 else 1.0

    # Pass 2 (streaming): accumulate P1D with the known global mean_F.
    acc = P1DAccumulator(nbins, dv_kms)
    for _, _, tau_batch in _make_iter():
        acc.add_batch(tau_batch.astype(np.float64), mean_F_global=mean_F_global)

    k_centres, p1d = acc.result(k_bins)
    return k_centres, p1d, mean_F_global


# ---------------------------------------------------------------------------
# Multi-variant P1D (all + masked variants) in one call
# ---------------------------------------------------------------------------

ALL_VARIANTS = [
    "all",
    "no_LLS",
    "no_subDLA",
    "no_DLA",
    "no_HCD",
]

_MASK_CLASSES: Dict[str, List[str]] = {
    "all": [],
    "no_LLS": ["LLS"],
    "no_subDLA": ["subDLA"],
    "no_DLA": ["DLA"],
    "no_HCD": ["LLS", "subDLA", "DLA"],
    "only_LLS": ["subDLA", "DLA"],      # keep only LLS: mask everything else
    "only_subDLA": ["LLS", "DLA"],
    "only_DLA": ["LLS", "subDLA"],
}


def compute_all_p1d_variants(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog=None,
    variants: Optional[List[str]] = None,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    fill_strategy: str = "mean_flux",
    k_bins: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute all requested P1D variants.

    Returns dict: variant_name → (k_centres, p1d, mean_F)
    """
    if variants is None:
        variants = ALL_VARIANTS

    results = {}
    for var in variants:
        mask_cls = _MASK_CLASSES.get(var, [])
        if mask_cls and catalog is None:
            logger.warning("No catalog provided; skipping masked variant '%s'", var)
            continue
        k, p1d, mf = compute_p1d_single(
            hdf5_path, nbins, dv_kms,
            catalog=catalog if mask_cls else None,
            mask_classes=mask_cls if mask_cls else None,
            batch_size=batch_size,
            n_skewers=n_skewers,
            fill_strategy=fill_strategy,
            k_bins=k_bins,
        )
        results[var] = (k, p1d, mf)
        logger.debug("P1D variant '%s' done, mean_F=%.4f", var, mf)

    return results


# ---------------------------------------------------------------------------
# P1D ratios
# ---------------------------------------------------------------------------

def compute_p1d_ratios(
    p1d_variants: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
) -> Dict[str, np.ndarray]:
    """
    Compute standard P1D ratios.

    Convention:
      "ratio_noDLA_all"    = P1D_no_DLA / P1D_all
      "ratio_noHCD_all"    = P1D_no_HCD / P1D_all
      "ratio_nosubDLA_all" = P1D_no_subDLA / P1D_all
      "ratio_noLLS_all"    = P1D_no_LLS / P1D_all
      "ratio_onlyDLA_all"  = 1 - ratio_noDLA_all  (DLA contribution)

    All ratios are in flux power units (dimensionless power ratio).
    """
    ratios = {}
    if "all" not in p1d_variants:
        return ratios

    k_ref, p1d_all, _ = p1d_variants["all"]

    pairs = [
        ("ratio_noDLA_all", "no_DLA"),
        ("ratio_nosubDLA_all", "no_subDLA"),
        ("ratio_noLLS_all", "no_LLS"),
        ("ratio_noHCD_all", "no_HCD"),
    ]
    for ratio_name, var in pairs:
        if var in p1d_variants:
            _, p1d_var, _ = p1d_variants[var]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(p1d_all > 0, p1d_var / p1d_all, np.nan)
            ratios[ratio_name] = r

    # DLA contribution = 1 - P1D_noDLA / P1D_all
    if "ratio_noDLA_all" in ratios:
        ratios["ratio_DLA_contribution"] = 1.0 - ratios["ratio_noDLA_all"]

    ratios["k"] = k_ref
    return ratios
