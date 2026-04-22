"""
Tau and flux masking for absorber removal.

Given a full tau array and an AbsorberCatalog, produce modified tau arrays
where the contributions of specified absorber classes are removed.

Three fill strategies are supported:

  "zero_tau"   : set tau = 0 in masked pixels → F = 1 (unphysically transparent)
  "mean_flux"  : set F = mean(F_unmasked) in masked region → conservative neutral
  "contiguous" : linearly interpolate log(tau) over the masked region from
                 boundary values (smoothest, but can underestimate if mask is long)

The choice of fill strategy affects P1D at small k (large scales). For HCD
studies, "mean_flux" is the standard choice as it preserves the mean flux
constraint without introducing large-scale power.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .catalog import AbsorberCatalog, Absorber


# Default wing-expansion threshold per absorber class, in τ units.
# The mask for one system is pixels with τ > (threshold + τ_eff), contiguous
# around the system peak. Motivated by Rogers et al. 2018 / PRIYA §3.3: a DLA
# damping wing that has τ > 0.25 above the forest baseline is still carrying
# correlated flux suppression and should be removed. Scaling up for subDLA/LLS
# where wings are weaker gives a class-appropriate mask width.
DEFAULT_WING_THRESHOLD = {
    "DLA":    0.25,   # PRIYA/Rogers default
    "subDLA": 0.50,
    "LLS":    1.00,
}


# ---------------------------------------------------------------------------
# Index-level mask builder
# ---------------------------------------------------------------------------

def build_skewer_mask(
    n_pixels: int,
    absorbers: List[Absorber],
) -> np.ndarray:
    """
    Build a boolean mask array of shape (n_pixels,).
    True = pixel belongs to a masked absorber.

    absorbers must all have the same skewer_idx (caller's responsibility).
    """
    mask = np.zeros(n_pixels, dtype=bool)
    for ab in absorbers:
        if ab.pix_end >= n_pixels:
            # Periodic boundary wrap: system spans [..., n_pixels-1] and [0, ...]
            mask[ab.pix_start:] = True
            mask[:ab.pix_end - n_pixels + 1] = True
        else:
            mask[ab.pix_start:ab.pix_end + 1] = True
    return mask


# ---------------------------------------------------------------------------
# Core fill functions
# ---------------------------------------------------------------------------

def _fill_zero_tau(tau: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = tau.copy()
    out[mask] = 0.0
    return out


def _fill_mean_flux(tau: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with F = mean(F_unmasked), then back to tau."""
    F = np.exp(-tau)
    unmasked = ~mask
    if unmasked.any():
        mean_F = F[unmasked].mean()
    else:
        mean_F = 1.0
    # Avoid log(0)
    mean_F = max(mean_F, 1e-30)
    tau_fill = -np.log(mean_F)
    out = tau.copy()
    out[mask] = tau_fill
    return out


def _fill_contiguous(tau: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Interpolate log(tau) over masked regions from boundary pixels.
    Falls back to mean_flux at edges (masked region at array boundary).
    """
    out = tau.copy()
    n = len(tau)
    # Find masked segments
    i = 0
    while i < n:
        if mask[i]:
            # Find extent of this masked segment
            start = i
            while i < n and mask[i]:
                i += 1
            end = i - 1  # inclusive

            # Boundary values
            left_val = np.log(max(tau[start - 1], 1e-30)) if start > 0 else None
            right_val = np.log(max(tau[end + 1], 1e-30)) if end < n - 1 else None

            if left_val is None and right_val is None:
                fill_vals = np.zeros(end - start + 1)
            elif left_val is None:
                fill_vals = np.full(end - start + 1, right_val)
            elif right_val is None:
                fill_vals = np.full(end - start + 1, left_val)
            else:
                fill_vals = np.linspace(left_val, right_val, end - start + 1)

            out[start:end + 1] = np.exp(fill_vals)
        else:
            i += 1
    return out


_FILL_FUNCTIONS = {
    "zero_tau": _fill_zero_tau,
    "mean_flux": _fill_mean_flux,
    "contiguous": _fill_contiguous,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_mask_to_skewer(
    tau: np.ndarray,
    absorbers: List[Absorber],
    strategy: str = "mean_flux",
) -> np.ndarray:
    """
    Apply masking to a single skewer's tau array.

    Parameters
    ----------
    tau       : 1D tau array (nbins,)
    absorbers : list of Absorber objects for this skewer
    strategy  : fill strategy ("zero_tau" | "mean_flux" | "contiguous")

    Returns
    -------
    tau_masked : tau array with absorber pixels replaced/filled
    """
    if not absorbers:
        return tau

    mask = build_skewer_mask(len(tau), absorbers)
    fill_fn = _FILL_FUNCTIONS.get(strategy)
    if fill_fn is None:
        raise ValueError(f"Unknown fill strategy: {strategy!r}. "
                         f"Choose from {list(_FILL_FUNCTIONS)}")
    return fill_fn(tau, mask)


def apply_mask_to_batch(
    tau_batch: np.ndarray,
    batch_start: int,
    catalog: AbsorberCatalog,
    mask_classes: List[str],
    strategy: str = "mean_flux",
) -> np.ndarray:
    """
    Apply class-selective masking to a batch of skewers.

    Parameters
    ----------
    tau_batch   : shape (batch_size, nbins)
    batch_start : global row index of tau_batch[0]
    catalog     : AbsorberCatalog for this (sim, snap)
    mask_classes: absorber classes to mask, e.g. ["DLA", "subDLA"]
    strategy    : fill strategy

    Returns
    -------
    tau_masked : same shape as tau_batch, with absorbers replaced
    """
    tau_out = tau_batch.copy()
    batch_size = tau_batch.shape[0]

    # Build a lookup: global_skewer_idx → list of absorbers
    absorbers_by_skewer: dict = {}
    for ab in catalog.absorbers:
        if ab.absorber_class in mask_classes:
            key = ab.skewer_idx
            absorbers_by_skewer.setdefault(key, []).append(ab)

    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx
        if global_idx not in absorbers_by_skewer:
            continue
        tau_out[local_idx] = apply_mask_to_skewer(
            tau_batch[local_idx].astype(np.float64),
            absorbers_by_skewer[global_idx],
            strategy=strategy,
        )

    return tau_out


# ---------------------------------------------------------------------------
# Convenience: build masked tau for an entire file (loads in batches)
# ---------------------------------------------------------------------------

def iter_masked_batches(
    hdf5_path,
    catalog: AbsorberCatalog,
    mask_classes: List[str],
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    strategy: str = "mean_flux",
):
    """
    Iterate over the tau file, yielding masked batches.
    Yields (row_start, row_end, tau_masked_batch).
    """
    from .io import iter_tau_batches

    for row_start, row_end, tau_batch in iter_tau_batches(
        hdf5_path, batch_size=batch_size, n_skewers=n_skewers
    ):
        tau_masked = apply_mask_to_batch(
            tau_batch.astype(np.float64),
            row_start,
            catalog,
            mask_classes,
            strategy=strategy,
        )
        yield row_start, row_end, tau_masked


# ---------------------------------------------------------------------------
# τ-space per-class mask — walk outward from each system peak until τ drops
# below (wing_threshold[class] + τ_eff).  This generalises the PRIYA DLA
# recipe to subDLA and LLS.  The mask width is driven by the actual τ profile,
# so a strong DLA gets a wide damping-wing mask and a weak LLS gets a narrow
# core mask without any hard-coded velocity limits.
# ---------------------------------------------------------------------------

def build_skewer_mask_from_tau(
    tau_row: np.ndarray,
    absorbers: List[Absorber],
    tau_eff: float,
    wing_threshold_by_class: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Build a mask for one skewer by expanding each absorber outward from its
    τ-peak until τ < (wing_threshold[class] + τ_eff).

    Vectorised implementation:
      1. For each (unique) threshold value, build `above` = tau_row > threshold.
      2. Use `np.add.reduceat` to identify contiguous "True runs" on the doubled
         array (for periodic wrap).
      3. For each absorber, locate the run containing its peak by a single
         searchsorted, and mark its span.

    absorbers must all have the same skewer_idx (caller's responsibility) and
    belong to the classes we want to mask.
    """
    if wing_threshold_by_class is None:
        wing_threshold_by_class = DEFAULT_WING_THRESHOLD

    n = len(tau_row)
    mask = np.zeros(n, dtype=bool)
    if not absorbers:
        return mask

    # Group absorbers by their threshold so we only do the run-detection work
    # once per distinct threshold value.
    groups: Dict[float, List[Absorber]] = {}
    for ab in absorbers:
        thr = wing_threshold_by_class.get(ab.absorber_class, 1.0) + tau_eff
        groups.setdefault(thr, []).append(ab)

    # Doubled array for periodic-wrap handling
    tau2 = np.concatenate([tau_row, tau_row])

    for thr, abs_list in groups.items():
        above = tau2 > thr
        if not above.any():
            continue

        # Locate run boundaries in the doubled array via diff:
        #   d[i] = +1 where a run starts, -1 where it ends (exclusive).
        diff = np.diff(np.concatenate(([False], above, [False])).astype(np.int8))
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]  # exclusive

        for ab in abs_list:
            # Find τ peak in this absorber's detected region (handle wrap)
            if ab.pix_end >= n:
                seg = tau2[ab.pix_start:ab.pix_end + 1]
                peak = ab.pix_start + int(np.argmax(seg))  # in doubled coords
            else:
                seg = tau_row[ab.pix_start:ab.pix_end + 1]
                peak = ab.pix_start + int(np.argmax(seg))  # in single coords

            # searchsorted: find the run that contains `peak`
            # run i covers [run_starts[i], run_ends[i]); peak is in run i if
            # run_starts[i] <= peak < run_ends[i].
            idx = int(np.searchsorted(run_starts, peak, side="right")) - 1
            if idx < 0:
                continue
            s, e = int(run_starts[idx]), int(run_ends[idx])
            if not (s <= peak < e):
                continue  # peak isn't inside any above-threshold run

            # Map [s, e) from doubled array back to mask indices (mod n).
            # Run length is at most 2n; clip to n to avoid masking the whole
            # skewer twice.
            length = min(e - s, n)
            s_mod = s % n
            if s_mod + length <= n:
                mask[s_mod : s_mod + length] = True
            else:
                mask[s_mod:] = True
                mask[: length - (n - s_mod)] = True

    return mask


def apply_tauspace_mask_to_batch(
    tau_batch: np.ndarray,
    batch_start: int,
    catalog: AbsorberCatalog,
    mask_classes: List[str],
    tau_eff: float,
    wing_threshold_by_class: Optional[Dict[str, float]] = None,
    fill_strategy: str = "mean_flux",
) -> np.ndarray:
    """
    Apply τ-space per-system masking to a batch of skewers.

    Unlike apply_mask_to_batch (which masks only the τ > τ_threshold core),
    this walks outward from each system's peak into the damping wings until
    τ drops below the class-specific wing threshold above the forest baseline.

    fill_strategy:
      "zero_tau"   : τ = 0 in masked pixels (F = 1; unphysical)
      "mean_flux"  : τ = τ_eff in masked pixels (δF = 0; PRIYA recipe)
      "contiguous" : log-τ interpolation across the masked segment
    """
    tau_out = tau_batch.astype(np.float64, copy=True)

    # Group absorbers (of the requested classes) by skewer index
    by_sl: Dict[int, List[Absorber]] = {}
    for ab in catalog.absorbers:
        if ab.absorber_class in mask_classes:
            by_sl.setdefault(ab.skewer_idx, []).append(ab)

    fill_fn = _FILL_FUNCTIONS.get(fill_strategy)
    if fill_fn is None:
        raise ValueError(f"Unknown fill strategy: {fill_strategy!r}")

    for local_idx in range(tau_out.shape[0]):
        global_idx = batch_start + local_idx
        absorbers = by_sl.get(global_idx)
        if not absorbers:
            continue
        mask = build_skewer_mask_from_tau(
            tau_out[local_idx], absorbers, tau_eff, wing_threshold_by_class,
        )
        # Reuse existing fill functions. _fill_mean_flux computes its own
        # per-row mean; for consistency with PRIYA we'd rather fill with the
        # global τ_eff, so handle that case specifically.
        if fill_strategy == "mean_flux":
            tau_out[local_idx, mask] = tau_eff
        else:
            tau_out[local_idx] = fill_fn(tau_out[local_idx], mask)

    return tau_out


def iter_tauspace_masked_batches(
    hdf5_path,
    catalog: AbsorberCatalog,
    mask_classes: List[str],
    tau_eff: float,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    wing_threshold_by_class: Optional[Dict[str, float]] = None,
    fill_strategy: str = "mean_flux",
):
    """
    Iterator yielding (row_start, row_end, tau_masked_batch) where the mask
    is the τ-space per-class one built by build_skewer_mask_from_tau.
    """
    from .io import iter_tau_batches

    for row_start, row_end, tau_batch in iter_tau_batches(
        hdf5_path, batch_size=batch_size, n_skewers=n_skewers
    ):
        yield row_start, row_end, apply_tauspace_mask_to_batch(
            tau_batch, row_start, catalog, mask_classes, tau_eff,
            wing_threshold_by_class=wing_threshold_by_class,
            fill_strategy=fill_strategy,
        )


# ---------------------------------------------------------------------------
# PRIYA-style DLA masking (arXiv:2306.05471)
# ---------------------------------------------------------------------------

def priya_dla_mask_row(
    tau_row: np.ndarray,
    tau_eff: float,
    tau_dla_detect: float = 1e6,
    tau_mask_scale: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Return a boolean mask for one sightline using the PRIYA DLA masking recipe.

    Algorithm (PRIYA paper, Sec. 3.3):
      1. Detect: sightline is DLA-contaminated if max(tau) > tau_dla_detect (~10^6).
      2. Locate the DLA peak pixel.
      3. Expand outward from the peak in both directions until tau drops below
         tau_mask_scale + tau_eff.  The mask is this single CONTIGUOUS region
         around the peak — scattered IGM pixels above the threshold elsewhere in
         the sightline are NOT masked (they are not part of the DLA wing).
      4. Fill: caller sets masked pixels to tau_eff so that delta_F = 0.

    Returns None if the sightline has no DLA (max tau <= tau_dla_detect).
    """
    if tau_row.max() <= tau_dla_detect:
        return None

    n = len(tau_row)
    threshold = tau_mask_scale + tau_eff
    peak = int(np.argmax(tau_row))

    # Walk left from peak until tau < threshold
    lo = peak
    while lo > 0 and tau_row[lo - 1] >= threshold:
        lo -= 1

    # Walk right from peak until tau < threshold
    hi = peak
    while hi < n - 1 and tau_row[hi + 1] >= threshold:
        hi += 1

    mask = np.zeros(n, dtype=bool)
    mask[lo:hi + 1] = True
    return mask


def iter_priya_masked_batches(
    hdf5_path,
    tau_eff: float,
    batch_size: int = 4096,
    n_skewers: Optional[int] = None,
    tau_dla_detect: float = 1e6,
    tau_mask_scale: float = 0.25,
):
    """
    Iterate over tau file applying PRIYA DLA masking.

    For each sightline with max(tau) > tau_dla_detect (contains a DLA):
      - mask pixels where tau > tau_mask_scale + tau_eff
      - fill masked pixels with tau_eff (so delta_F = 0 in masked region)

    Yields (row_start, row_end, tau_masked_batch) — same interface as
    iter_masked_batches so it can be plugged into compute_p1d_single.
    """
    from .io import iter_tau_batches

    tau_fill = tau_eff  # fill value: tau_eff → delta_F = 0

    for row_start, row_end, tau_batch in iter_tau_batches(
        hdf5_path, batch_size=batch_size, n_skewers=n_skewers
    ):
        tau_out = tau_batch.astype(np.float64).copy()
        for local_i in range(tau_out.shape[0]):
            mask = priya_dla_mask_row(
                tau_out[local_i], tau_eff,
                tau_dla_detect=tau_dla_detect,
                tau_mask_scale=tau_mask_scale,
            )
            if mask is not None:
                tau_out[local_i][mask] = tau_fill
        yield row_start, row_end, tau_out
