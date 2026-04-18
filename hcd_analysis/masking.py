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

from typing import List, Optional, Tuple

import numpy as np

from .catalog import AbsorberCatalog, Absorber


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
