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

# Default k bins matching the emulator's kf grid (s/km), extended to k_Nyquist≈0.05
# Original emulator grid: 35 bins up to ~0.0195 s/km
# Extension: 15 additional log-spaced bins from 0.020 to 0.050 s/km
_EMULATOR_K_BINS = np.array([
    0.001084, 0.001626, 0.002168, 0.00271,  0.003252, 0.003794,
    0.004336, 0.004878, 0.00542,  0.005962, 0.006504, 0.007046,
    0.007588, 0.00813,  0.008672, 0.009214, 0.009756, 0.010298,
    0.01084,  0.011382, 0.011924, 0.012466, 0.013008, 0.01355,
    0.014092, 0.014634, 0.015176, 0.015718, 0.01626,  0.016802,
    0.017344, 0.017886, 0.018428, 0.01897,  0.019512,
])
# Extended to Nyquist (dv≈10 km/s → k_Nyq≈0.05 s/km)
_EXTENDED_K_BINS = np.concatenate([
    _EMULATOR_K_BINS,
    np.logspace(np.log10(0.020), np.log10(0.050), 15),
])
_DEFAULT_K_BINS = _EXTENDED_K_BINS


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
    mask_scheme: str = "tauspace",
    wing_threshold_by_class: Optional[dict] = None,
    tau_eff: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute P1D for one variant (all, or with specified classes masked).

    mask_scheme:
      "tauspace" (default) — mask each system by walking outward from its
          τ-peak until τ < (wing_threshold[class] + τ_eff).  Physically
          motivated damping-wing mask; generalises the PRIYA DLA recipe.
      "pixrange" — legacy pix_start..pix_end mask (τ > τ_threshold core only;
          misses damping wings; retained for regression testing only).

    Returns (k_centres, p1d, mean_F_used_for_normalisation).
    """
    from .io import iter_tau_batches
    from .masking import iter_masked_batches, iter_tauspace_masked_batches

    needs_masking = bool(mask_classes and catalog is not None)

    # For tauspace masking we need τ_eff = -ln⟨F⟩_unmasked.  If caller
    # provided it (e.g. compute_all_p1d_variants computes the "all" variant
    # first and passes its <F> down), skip the extra pass.
    if needs_masking and mask_scheme == "tauspace" and tau_eff is None:
        F_sum, F_n = 0.0, 0
        for _, _, tau_batch in iter_tau_batches(
            hdf5_path, batch_size=batch_size, n_skewers=n_skewers
        ):
            F_sum += np.exp(-tau_batch.astype(np.float64)).sum()
            F_n += tau_batch.size
        mean_F_unmasked = F_sum / F_n if F_n > 0 else 1.0
        tau_eff = -np.log(max(mean_F_unmasked, 1e-30))

    def _make_iter():
        if not needs_masking:
            return iter_tau_batches(
                hdf5_path, batch_size=batch_size, n_skewers=n_skewers,
            )
        if mask_scheme == "tauspace":
            return iter_tauspace_masked_batches(
                hdf5_path, catalog, mask_classes, tau_eff,
                batch_size=batch_size, n_skewers=n_skewers,
                wing_threshold_by_class=wing_threshold_by_class,
                fill_strategy=fill_strategy,
            )
        if mask_scheme == "pixrange":
            return iter_masked_batches(
                hdf5_path, catalog, mask_classes,
                batch_size=batch_size, n_skewers=n_skewers,
                strategy=fill_strategy,
            )
        raise ValueError(f"Unknown mask_scheme: {mask_scheme!r}")

    # Pass 1 (streaming): compute the global mean_F used for delta_F
    # normalisation.  For tauspace masking we use the unmasked <F> (so that
    # all variants are compared against the same reference and τ_eff-filled
    # pixels contribute δF = 0).  For pixrange we reproduce legacy behaviour.
    if needs_masking and mask_scheme == "tauspace":
        mean_F_global = np.exp(-tau_eff)
    else:
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

# Production P1D variants.  See docs/masking_strategy.md.
#   "all"           : no mask — the emulator baseline
#   "no_DLA_priya"  : PRIYA DLA mask (arXiv:2306.05471 §3.3).
#                     Residual LLS/subDLA contamination is handled at the
#                     emulator level via the Rogers+2018 α template, not by
#                     any spatial mask.
ALL_VARIANTS = ["all", "no_DLA_priya"]

# Catalog-based class masks.  Retained for diagnostics/regression only:
# both "pixrange" and "tauspace" schemes are known to introduce mask-edge
# artefacts above k ≈ 0.02 s/km (cyclic) and over-mask forest power at low k
# when they touch LLS/subDLA.  See docs/bugs_found.md §#6.
DIAGNOSTIC_VARIANTS = [
    "no_LLS", "no_subDLA", "no_DLA", "no_HCD",
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
    # "no_DLA_priya" is handled separately — uses tau-based detection/mask
}

# PRIYA DLA mask parameters (arXiv:2306.05471)
_PRIYA_DLA_DETECT_TAU = 1e6    # max(tau) threshold to flag a sightline as DLA
_PRIYA_DLA_MASK_SCALE = 0.25   # mask where tau > 0.25 + tau_eff


# ---------------------------------------------------------------------------
# Per-class subset-based P1D (Rogers-style templates)
# ---------------------------------------------------------------------------

def compute_p1d_per_class(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    k_bins: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Compute four subset P1Ds in a single streaming pass.

    The "highest-class" label per sightline is assigned from the catalog:
    DLA if any absorber has log N_HI ≥ 20.3, else subDLA if any ≥ 19.0,
    else LLS if any ≥ 17.2, else "clean" (no entry in the catalog).

    Each P1D uses that subset's own <F> for δF normalisation — the Rogers
    convention — so ratios like `P_DLA_only / P_clean` correspond directly
    to the HCD template `P_total / P_forest` and can be fit with
    `hcd_analysis.hcd_template.fit_alpha`.

    Returns
    -------
    dict with keys
      "k"                      : native k grid (cyclic s/km)
      "P_clean" / "P_LLS_only" / "P_subDLA_only" / "P_DLA_only"  : (n_k,)
      "mean_F_clean" / ... "mean_F_DLA"
      "n_sightlines_clean" / ... "n_sightlines_DLA"
      "n_total", "z" (if present in catalog)
    """
    from .io import iter_tau_batches
    import h5py

    # Determine total sightline count
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["tau/H/1/1215"].shape[0]
    if n_skewers is not None:
        n_total = min(n_total, n_skewers)

    # Build highest-class label per sightline (order matters: DLA wins over
    # subDLA wins over LLS).
    labels = np.full(n_total, "clean", dtype=object)
    for ab in catalog.absorbers:
        if ab.skewer_idx >= n_total:
            continue
        if ab.absorber_class == "DLA":
            labels[ab.skewer_idx] = "DLA"
        elif ab.absorber_class == "subDLA":
            if labels[ab.skewer_idx] != "DLA":
                labels[ab.skewer_idx] = "subDLA"
        elif ab.absorber_class == "LLS":
            if labels[ab.skewer_idx] not in ("DLA", "subDLA"):
                labels[ab.skewer_idx] = "LLS"

    classes = ("clean", "LLS", "subDLA", "DLA")
    counts = {c: int((labels == c).sum()) for c in classes}
    logger.debug("per-class counts: %s", counts)

    # Pass 1 — per-subset <F>
    F_sum = {c: 0.0 for c in classes}
    F_n = {c: 0 for c in classes}
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        F = np.exp(-tau.astype(np.float64))
        lab = labels[s:e]
        for c in classes:
            m = lab == c
            if m.any():
                F_sum[c] += F[m].sum()
                F_n[c] += F[m].size
    mean_F = {c: (F_sum[c] / F_n[c]) if F_n[c] > 0 else 1.0 for c in classes}

    # Pass 2 — per-subset P1D accumulator
    accs = {c: P1DAccumulator(nbins, dv_kms) for c in classes}
    for s, e, tau in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_total):
        tau = tau.astype(np.float64)
        lab = labels[s:e]
        for c in classes:
            m = lab == c
            if m.any():
                accs[c].add_batch(tau[m], mean_F_global=mean_F[c])

    k = accs["clean"]._k_native()
    if k_bins is not None:
        k_bins = np.asarray(k_bins)

    result = {"k": k, "n_total": n_total}
    for c in classes:
        if k_bins is None:
            _, p = accs[c].result(None)
            # `result()` re-bins onto default; for a clean native output use raw_power
            _, p_native = accs[c].raw_power()
            result[f"P_{c}_only" if c != "clean" else "P_clean"] = p_native
        else:
            k_c, p = accs[c].result(k_bins)
            result["k"] = k_c
            result[f"P_{c}_only" if c != "clean" else "P_clean"] = p
        result[f"mean_F_{c}"] = mean_F[c]
        result[f"n_sightlines_{c}"] = counts[c]
    return result


def save_p1d_per_class_hdf5(
    path,
    per_class: Dict[str, object],
    sim_name: str,
    snap: int,
    z: float,
    dv_kms: float,
    extra_attrs: Optional[Dict[str, object]] = None,
) -> None:
    """
    Write the per-class P1D dict as an HDF5 file with self-documenting
    file-level and dataset-level attributes.  `h5ls -v` on the output
    reveals all metadata without loading the numeric data.
    """
    import h5py
    with h5py.File(str(path), "w") as f:
        f.attrs["sim_name"] = sim_name
        f.attrs["snap"] = int(snap)
        f.attrs["z"] = float(z)
        f.attrs["dv_kms"] = float(dv_kms)
        f.attrs["k_convention"] = "cyclic (numpy rfftfreq); PRIYA_angular_k = 2*pi*this"
        f.attrs["mean_F_convention"] = "per-subset <F>; δF = F/<F>_subset - 1"
        f.attrs["description"] = (
            "Per-class subset P1D templates. Each class is the 'highest' "
            "absorber class on a sightline (DLA beats subDLA beats LLS). "
            "Rogers+2018 template P_total/P_forest = P_<class>_only / P_clean."
        )
        f.attrs["source_module"] = "hcd_analysis.p1d.compute_p1d_per_class"
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v

        # Datasets
        d = f.create_dataset("k", data=np.asarray(per_class["k"], dtype=np.float64))
        d.attrs["units"] = "s/km (cyclic)"
        d.attrs["description"] = "Frequency axis for P1D (numpy rfftfreq convention)"

        for c in ("clean", "LLS_only", "subDLA_only", "DLA_only"):
            key = f"P_{c}"
            if key in per_class:
                d = f.create_dataset(key, data=np.asarray(per_class[key], dtype=np.float64))
                d.attrs["units"] = "km/s"
                d.attrs["description"] = (
                    f"P1D(k) averaged over sightlines whose highest-class "
                    f"absorber is {c.replace('_only','')}"
                )

        # Per-class scalars (mean_F, n_sightlines)
        for key, desc, unit in [
            ("mean_F_clean",  "<F> over clean sightlines", "dimensionless"),
            ("mean_F_LLS",    "<F> over LLS sightlines",   "dimensionless"),
            ("mean_F_subDLA", "<F> over subDLA sightlines","dimensionless"),
            ("mean_F_DLA",    "<F> over DLA sightlines",   "dimensionless"),
            ("n_sightlines_clean",  "count of clean sightlines", ""),
            ("n_sightlines_LLS",    "count of LLS sightlines",   ""),
            ("n_sightlines_subDLA", "count of subDLA sightlines",""),
            ("n_sightlines_DLA",    "count of DLA sightlines",   ""),
            ("n_total",             "total sightlines in the file or subsample", ""),
        ]:
            if key in per_class:
                d = f.create_dataset(key, data=per_class[key])
                d.attrs["description"] = desc
                if unit: d.attrs["units"] = unit


# ---------------------------------------------------------------------------
# PRIYA-style tau-based DLA mask, applied in compute_p1d_priya_masked below
# ---------------------------------------------------------------------------
def compute_p1d_priya_masked(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    k_bins: Optional[np.ndarray] = None,
    tau_dla_detect: float = _PRIYA_DLA_DETECT_TAU,
    tau_mask_scale: float = _PRIYA_DLA_MASK_SCALE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute P1D with PRIYA-style DLA masking (arXiv:2306.05471).

    Two-pass algorithm:
      Pass 1: compute tau_eff = -ln(mean_F_all) from unmasked sightlines.
      Pass 2: for each sightline with max(tau) > tau_dla_detect (~10^6),
              mask pixels where tau > tau_mask_scale + tau_eff, fill with tau_eff
              (delta_F = 0 in masked region).  Other sightlines unchanged.

    Returns (k_centres, p1d, mean_F_unmasked).
    """
    from .io import iter_tau_batches
    from .masking import iter_priya_masked_batches

    # Pass 1: compute tau_eff from ALL sightlines (unmasked)
    F_sum, F_n = 0.0, 0
    for _, _, tau_batch in iter_tau_batches(
        hdf5_path, batch_size=batch_size, n_skewers=n_skewers
    ):
        F_sum += np.exp(-tau_batch.astype(np.float64)).sum()
        F_n += tau_batch.size
    mean_F_all = F_sum / F_n if F_n > 0 else 1.0
    tau_eff = -np.log(max(mean_F_all, 1e-30))

    # Pass 2: apply PRIYA mask and compute P1D
    acc = P1DAccumulator(nbins, dv_kms)
    for _, _, tau_batch in iter_priya_masked_batches(
        hdf5_path, tau_eff,
        batch_size=batch_size, n_skewers=n_skewers,
        tau_dla_detect=tau_dla_detect, tau_mask_scale=tau_mask_scale,
    ):
        acc.add_batch(tau_batch, mean_F_global=mean_F_all)

    k_centres, p1d = acc.result(k_bins)
    return k_centres, p1d, mean_F_all


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
    mask_scheme: str = "tauspace",
    wing_threshold_by_class: Optional[dict] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute all requested P1D variants.

    Returns dict: variant_name → (k_centres, p1d, mean_F)
    """
    if variants is None:
        variants = ALL_VARIANTS

    # Compute the "all" variant first so we can share its <F> as τ_eff across
    # all τ-space-masked variants (avoids recomputing it once per variant).
    results = {}
    shared_tau_eff: Optional[float] = None
    if "all" in variants:
        k0, p0, mf0 = compute_p1d_single(
            hdf5_path, nbins, dv_kms,
            catalog=None, mask_classes=None,
            batch_size=batch_size, n_skewers=n_skewers,
            fill_strategy=fill_strategy, k_bins=k_bins,
        )
        results["all"] = (k0, p0, mf0)
        shared_tau_eff = -np.log(max(mf0, 1e-30))
        logger.debug("P1D variant 'all' done, tau_eff=%.4f", shared_tau_eff)

    for var in variants:
        if var == "all":
            continue
        if var == "no_DLA_priya":
            k, p1d, mf = compute_p1d_priya_masked(
                hdf5_path, nbins, dv_kms,
                batch_size=batch_size, n_skewers=n_skewers, k_bins=k_bins,
            )
            results[var] = (k, p1d, mf)
            logger.debug("P1D variant 'no_DLA_priya' done (tau-based), mean_F=%.4f", mf)
            continue

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
            mask_scheme=mask_scheme,
            wing_threshold_by_class=wing_threshold_by_class,
            tau_eff=shared_tau_eff,
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
        ("ratio_noDLA_priya_all", "no_DLA_priya"),   # primary production ratio
        # Legacy catalog-based variants (now diagnostic-only; emit only if present)
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

    # DLA contribution = 1 - P1D_noDLA_priya / P1D_all  (primary production metric)
    if "ratio_noDLA_priya_all" in ratios:
        ratios["ratio_DLA_contribution"] = 1.0 - ratios["ratio_noDLA_priya_all"]

    ratios["k"] = k_ref
    return ratios


# ---------------------------------------------------------------------------
# Sightline exclusion P1D (continuous NHI threshold sweep)
# ---------------------------------------------------------------------------

def compute_p1d_excl_nhi(
    hdf5_path,
    nbins: int,
    dv_kms: float,
    catalog,
    log_nhi_cuts: List[float],
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    k_bins: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Compute P1D with sightline exclusion at each log10(NHI) threshold.

    For each cut in log_nhi_cuts: sightlines that contain ANY absorber with
    log10(NHI) >= cut are excluded entirely.  The remaining sightlines are
    used for a two-pass streaming P1D.

    Parameters
    ----------
    catalog       : pd.DataFrame or dict-like with columns 'skewer_idx', 'log_nhi'
    log_nhi_cuts  : list of thresholds, e.g. [17.2, 18.0, 19.0, 20.3, 21.0]

    Returns
    -------
    dict with keys:
      "k"              : np.ndarray (n_k,) bin centres
      "p1d_excl"       : np.ndarray (n_cuts, n_k)  — P1D at each cut
      "frac_remaining" : np.ndarray (n_cuts,)       — fraction of sightlines kept
      "mean_F_excl"    : np.ndarray (n_cuts,)       — mean flux at each cut
      "log_nhi_cuts"   : list passed in
    """
    from .io import iter_tau_batches

    n_cuts = len(log_nhi_cuts)

    # Extract skewer_idx and log_nhi from catalog.
    # Accepts: AbsorberCatalog (has .absorbers list), numpy-dict, or None.
    if catalog is not None:
        if hasattr(catalog, "absorbers"):
            # AbsorberCatalog object
            absorbers = catalog.absorbers
            if absorbers:
                skewer_idx = np.array([a.skewer_idx for a in absorbers], dtype=np.int64)
                log_nhi = np.array([a.log_NHI for a in absorbers], dtype=np.float64)
            else:
                skewer_idx = np.array([], dtype=np.int64)
                log_nhi = np.array([], dtype=np.float64)
        elif hasattr(catalog, "get"):
            # dict-like
            skewer_idx = np.asarray(catalog.get("skewer_idx", []), dtype=np.int64)
            log_nhi = np.asarray(catalog.get("log_nhi", []), dtype=np.float64)
        elif hasattr(catalog, "__len__") and len(catalog) > 0:
            # pandas DataFrame (optional dependency)
            skewer_idx = np.asarray(catalog["skewer_idx"], dtype=np.int64)
            log_nhi = np.asarray(catalog["log_nhi"], dtype=np.float64)
        else:
            skewer_idx = np.array([], dtype=np.int64)
            log_nhi = np.array([], dtype=np.float64)
    else:
        skewer_idx = np.array([], dtype=np.int64)
        log_nhi = np.array([], dtype=np.float64)

    # For each cut: set of skewer indices to exclude
    excluded_sets: List[set] = []
    for cut in log_nhi_cuts:
        mask = log_nhi >= cut
        excluded_sets.append(set(skewer_idx[mask].tolist()))

    # Get total skewer count
    with __import__("h5py").File(hdf5_path, "r") as f:
        total_skewers = f["tau/H/1/1215"].shape[0]
    if n_skewers is not None:
        total_skewers = min(total_skewers, n_skewers)

    # Pass 1: compute mean_F for each cut (streaming)
    F_sums = np.zeros(n_cuts)
    F_ns = np.zeros(n_cuts, dtype=np.int64)

    for start, end, tau_batch in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers):
        row_indices = np.arange(start, end)
        F_batch = np.exp(-tau_batch.astype(np.float64))
        for ci, excl in enumerate(excluded_sets):
            keep = np.array([i not in excl for i in row_indices], dtype=bool)
            if keep.any():
                F_sums[ci] += F_batch[keep].sum()
                F_ns[ci] += F_batch[keep].size

    mean_F_excl = np.where(F_ns > 0, F_sums / F_ns, 1.0)

    # Pass 2: accumulate P1D for each cut
    accumulators = [P1DAccumulator(nbins, dv_kms) for _ in range(n_cuts)]

    for start, end, tau_batch in iter_tau_batches(hdf5_path, batch_size=batch_size, n_skewers=n_skewers):
        row_indices = np.arange(start, end)
        for ci, (excl, acc) in enumerate(zip(excluded_sets, accumulators)):
            keep = np.array([i not in excl for i in row_indices], dtype=bool)
            if keep.any():
                acc.add_batch(tau_batch[keep].astype(np.float64), mean_F_global=mean_F_excl[ci])

    # Collect results
    p1d_list = []
    frac_remaining = np.zeros(n_cuts)
    for ci, acc in enumerate(accumulators):
        k, p1d = acc.result(k_bins)
        p1d_list.append(p1d)
        frac_remaining[ci] = acc.n_skewers / total_skewers if total_skewers > 0 else 0.0

    p1d_excl = np.array(p1d_list)  # shape (n_cuts, n_k)

    return {
        "k": k,
        "p1d_excl": p1d_excl,
        "frac_remaining": frac_remaining,
        "mean_F_excl": mean_F_excl,
        "log_nhi_cuts": log_nhi_cuts,
    }
