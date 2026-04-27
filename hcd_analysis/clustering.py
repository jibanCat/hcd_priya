"""
HCD clustering — coordinate conversion, pair counters, bias extraction.

See ``docs/clustering_definitions.md`` for the formal spec.  This
module is **gated** on the validation tests in
``tests/test_clustering.py`` — do not call any function from here in a
production run until tests 1–9 pass.

The first slice committed here covers only the coordinate / loader
layer (tests 1–3 in the doc).  Pair counters and bias fitter come in a
follow-up commit after user approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants and conventions
# ---------------------------------------------------------------------------
#
# PRIYA stores ``spectra/axis`` 1-indexed (1 = x, 2 = y, 3 = z).  We
# convert to 0-indexed exactly once, on load, and never look back.
# ``cofm`` is in **kpc/h**; we convert to **Mpc/h** on load.  All
# downstream code assumes 0-indexed axes and Mpc/h distances.
#
# Periodic minimum image: for any Δx in [-box, box],
#     Δx_min = Δx − box · round(Δx / box)
# This sends Δx into [-box/2, box/2].
# ---------------------------------------------------------------------------


@dataclass
class SightlineGeometry:
    """Loader output: 3D-position-resolvable sightline grid for one snap.

    All distances in **comoving Mpc/h**.

    Attributes
    ----------
    box : float
        Comoving box side length in Mpc/h.
    n_pix : int
        Number of pixels per sightline.
    dx_pix : float
        Pixel pitch along the LOS in Mpc/h: ``box / n_pix``.
    n_sightlines : int
        Total number of sightlines (e.g. 691200 for the LF grid).
    axis : np.ndarray, shape (n_sightlines,), int8
        LOS axis index, **0-indexed** (0=x, 1=y, 2=z).  Caller must
        not assume the original 1-indexed convention.
    cofm_mpch : np.ndarray, shape (n_sightlines, 3), float64
        Sightline anchor (x, y, z) in Mpc/h.
    z_snap : float
        Snapshot redshift.
    hubble : float
        Dimensionless h.
    """

    box: float
    n_pix: int
    dx_pix: float
    n_sightlines: int
    axis: np.ndarray
    cofm_mpch: np.ndarray
    z_snap: float
    hubble: float

    def __post_init__(self):
        if self.axis.min() < 0 or self.axis.max() > 2:
            raise ValueError(
                f"axis must be 0-indexed in [0, 2]; got "
                f"min={self.axis.min()} max={self.axis.max()}"
            )
        if self.cofm_mpch.shape != (self.n_sightlines, 3):
            raise ValueError(
                f"cofm_mpch shape mismatch: expected "
                f"({self.n_sightlines}, 3), got {self.cofm_mpch.shape}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_sightline_geometry(spectra_h5_path: Path) -> SightlineGeometry:
    """Load the sightline grid from a fake_spectra HDF5 file.

    Side effects: subtracts 1 from ``axis`` (PRIYA → 0-indexed) and
    converts ``cofm`` from kpc/h to Mpc/h.

    Parameters
    ----------
    spectra_h5_path : Path
        ``…/SPECTRA_NNN/lya_forest_spectra_grid_480.hdf5``

    Returns
    -------
    SightlineGeometry
    """
    p = Path(spectra_h5_path)
    with h5py.File(p, "r") as f:
        box_kpch = float(f["Header"].attrs["box"])
        hubble = float(f["Header"].attrs["hubble"])
        z_snap = float(f["Header"].attrs["redshift"])
        n_pix = int(f["Header"].attrs["nbins"])
        axis = np.asarray(f["spectra/axis"][:], dtype=np.int8) - 1
        cofm_kpch = np.asarray(f["spectra/cofm"][:], dtype=np.float64)

    box = box_kpch / 1000.0
    cofm_mpch = cofm_kpch / 1000.0
    dx_pix = box / n_pix
    n_sl = axis.shape[0]
    return SightlineGeometry(
        box=box,
        n_pix=n_pix,
        dx_pix=dx_pix,
        n_sightlines=n_sl,
        axis=axis,
        cofm_mpch=cofm_mpch,
        z_snap=z_snap,
        hubble=hubble,
    )


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def pixel_to_xyz(
    geom: SightlineGeometry,
    skewer_idx: np.ndarray,
    pixel: np.ndarray,
) -> np.ndarray:
    """Map (sightline, pixel) → 3D position in Mpc/h.

    For each pair, the lateral (non-LOS) coordinates are read straight
    from ``cofm``.  The LOS coordinate is

        x[axis] = (cofm[i, axis] + (pixel + 0.5) · dx_pix)  mod  box

    Pixel 0 lands at the *centre* of the first pixel (offset = 0.5),
    so a pair with ``pixel = 0`` and ``pixel = n_pix - 1`` are
    physically box minus dx_pix apart, not exactly box.

    Parameters
    ----------
    skewer_idx : (N,) int
    pixel      : (N,) int  or float (sub-pixel position allowed)

    Returns
    -------
    xyz : (N, 3) float64, in Mpc/h
    """
    skewer_idx = np.asarray(skewer_idx, dtype=np.int64)
    pixel = np.asarray(pixel, dtype=np.float64)
    if skewer_idx.shape != pixel.shape:
        raise ValueError(
            f"skewer_idx and pixel shape mismatch: "
            f"{skewer_idx.shape} vs {pixel.shape}"
        )

    cof = geom.cofm_mpch[skewer_idx]      # (N, 3)
    ax = geom.axis[skewer_idx]            # (N,)
    los = (cof[np.arange(len(ax)), ax] + (pixel + 0.5) * geom.dx_pix) % geom.box

    xyz = cof.copy()
    xyz[np.arange(len(ax)), ax] = los
    return xyz


def xyz_to_nearest_pixel(
    geom: SightlineGeometry,
    skewer_idx: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Inverse of ``pixel_to_xyz`` along a known sightline.

    Given a sightline index and a 3D point that *should* lie on that
    sightline, return the nearest pixel index.  Used by the coord
    round-trip test (test 1).
    """
    skewer_idx = np.asarray(skewer_idx, dtype=np.int64)
    if xyz.shape[0] != skewer_idx.shape[0]:
        raise ValueError("skewer_idx and xyz length mismatch")

    cof = geom.cofm_mpch[skewer_idx]      # (N, 3)
    ax = geom.axis[skewer_idx]            # (N,)
    los_pos = xyz[np.arange(len(ax)), ax]
    cof_los = cof[np.arange(len(ax)), ax]
    pix_f = ((los_pos - cof_los) % geom.box) / geom.dx_pix - 0.5
    return np.round(pix_f).astype(np.int64) % geom.n_pix


def minimum_image(
    delta: np.ndarray,
    box: float,
) -> np.ndarray:
    """Periodic minimum-image of a 1-D or N-D Δ array.

    Returns Δ in [-box/2, box/2].
    """
    delta = np.asarray(delta, dtype=np.float64)
    return delta - box * np.round(delta / box)


def los_separation(
    delta: np.ndarray,
    los_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose a separation vector into (signed r_par, r_perp).

    The LOS direction is taken from the ``los_axis`` argument (per
    pair, 0-indexed).  Sign of ``r_par`` is the sign of ``Δ[axis]``.

    Parameters
    ----------
    delta    : (N, 3) Δr after minimum_image, Mpc/h
    los_axis : (N,)   0-indexed LOS axis per pair

    Returns
    -------
    r_par_signed : (N,) Mpc/h  (sign preserved)
    r_perp       : (N,) Mpc/h  (always ≥ 0)
    """
    delta = np.asarray(delta, dtype=np.float64)
    los_axis = np.asarray(los_axis, dtype=np.int64)
    if delta.ndim != 2 or delta.shape[1] != 3:
        raise ValueError(f"delta must be (N, 3); got {delta.shape}")
    if los_axis.shape != (delta.shape[0],):
        raise ValueError(
            f"los_axis must be (N,) matching delta; "
            f"got delta {delta.shape} vs axis {los_axis.shape}"
        )

    n = delta.shape[0]
    rows = np.arange(n)
    r_par_signed = delta[rows, los_axis]
    perp_sq = (delta * delta).sum(axis=1) - r_par_signed * r_par_signed
    # Floating-point: perp_sq can dip negative by ~1e-30 for r_par == |Δ|.
    perp_sq = np.maximum(perp_sq, 0.0)
    r_perp = np.sqrt(perp_sq)
    return r_par_signed, r_perp
