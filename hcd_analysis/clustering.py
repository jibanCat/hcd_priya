"""
HCD clustering — coordinate conversion, pair counters, ξ estimators,
δ_F field builder.

See ``docs/clustering_definitions.md`` for the formal spec and the
locked-in design decisions.  Real-PRIYA validation gates run by
``scripts/run_test10.py`` and ``scripts/run_test11.py``; results
written up in ``docs/clustering_test10_results.md`` and
``docs/clustering_test11_results.md``.

Public API:

* ``SightlineGeometry`` + ``load_sightline_geometry`` — sightline grid
  loader, axis 0-indexed, distances in Mpc/h.
* ``pixel_to_xyz`` / ``xyz_to_nearest_pixel`` — bidirectional
  (skewer, pixel) ↔ 3D position with the half-pixel-offset convention.
* ``minimum_image`` / ``los_separation`` — periodic wrap +
  decomposition into (signed r_par, r_perp).
* ``pair_count_2d`` — generic vectorised pair counter on the
  (signed r_par, r_perp) grid.
* ``xi_cross_dla_lya`` — DLA points × Lyα flux field.
* ``xi_auto_dla`` — DLA × DLA with analytic RR on a periodic box.
* ``xi_auto_lya`` — Lyα flux × Lyα flux (with optional pixel subsample).
* ``build_delta_F_field`` — all-HCD-masked δ_F field per pixel.
* ``fold_signed_to_abs`` — fold ξ(±r_par) → ξ(|r_par|), with optional
  pair-count-weighted variant.
* ``pair_count_rmu`` — pair counter binned directly in (r, |μ|) for
  unbiased Hamilton-formula multipole extraction.
* ``xi_cross_dla_lya_rmu`` / ``xi_auto_dla_rmu`` / ``xi_auto_lya_rmu``
  — wrappers that mirror the (r_⊥, r_∥) versions but return
  (r, |μ|)-binned ξ.

Multipole extraction and the joint (b_DLA, β_DLA) fit live in
``hcd_analysis.lya_bias`` (``extract_multipoles_rmu``,
``fit_b_beta_from_xi_cross_multipoles``).  They consume the (r, |μ|)
output of the wrappers above; the (r_⊥, r_∥) path is preserved for
back-compat (the monopole-only Jacobian bias is O(few %), see
``docs/clustering_multipole_jacobian_todo.md``).
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


# ---------------------------------------------------------------------------
# Generic 2D pair counter
# ---------------------------------------------------------------------------

def pair_count_2d(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_los_axis: np.ndarray,
    box: float,
    r_perp_bins: np.ndarray,
    r_par_bins_signed: np.ndarray,
    weights_left: Optional[np.ndarray] = None,
    weights_right: Optional[np.ndarray] = None,
    exclude_self: bool = False,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic 2D (signed r_par, r_perp) pair accumulator.

    For every pair (i in left, j in right) we form Δ = left[i] − right[j],
    apply periodic minimum image, decompose into (r_par_signed, r_perp)
    using ``left_los_axis[i]`` as the LOS reference, and accumulate

        weight_pair = w_L[i] * w_R[j]

    into the (r_perp, r_par) bin if the pair lands inside the binning
    grid.  When ``weights_*`` are None they default to 1.

    The LOS axis is taken from the **left** point (this is the design
    choice: for ξ_× we set ``left = pixels`` so r_par is along the
    Lyman-α pixel's sightline axis, per the doc §4).

    Parameters
    ----------
    left_xyz, right_xyz : (N_L, 3), (N_R, 3) float
        Positions in Mpc/h.  Both must lie in [0, box) modulo periodicity.
    left_los_axis : (N_L,) int
        0-indexed LOS axis per left point.
    box : float
        Comoving box side, Mpc/h.
    r_perp_bins : (n_perp + 1,) float
        Edges in Mpc/h, monotonically increasing, starting at ≥ 0.
    r_par_bins_signed : (n_par + 1,) float
        Edges in Mpc/h, monotonically increasing.  Pass a symmetric
        grid (e.g. ``np.linspace(-50, 50, 51)``) to retain the LOS
        sign; this is what production wrappers do, then call
        ``fold_signed_to_abs`` with counts and npairs to get
        ``|r_par|``.  Passing edges that start at 0 will silently
        DROP all negative-r_par pairs — this function does not take
        ``abs(r_par)`` on its own, so the previous "fold by passing
        non-negative edges" suggestion (caught by Copilot review #5
        on PR #7) was wrong.
    weights_left, weights_right : optional
        Per-point weights.  Default: 1 each.
    exclude_self : bool
        If True, drop pairs where ``i == j``.  Use for auto-correlation
        of identical catalogs.
    chunk_size : int
        Right-side chunking for memory.  Each chunk allocates an
        (N_L, chunk_size) × 3 array.

    Returns
    -------
    counts  : (n_perp, n_par) float64 — sum of weight_pair per bin
    npairs  : (n_perp, n_par) int64   — number of pairs per bin
    """
    L = np.asarray(left_xyz, dtype=np.float64)
    R = np.asarray(right_xyz, dtype=np.float64)
    ax = np.asarray(left_los_axis, dtype=np.int64)
    if L.shape[0] != ax.shape[0]:
        raise ValueError("left_xyz and left_los_axis length mismatch")
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(f"left_xyz must be (N_L, 3); got {L.shape}")
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError(f"right_xyz must be (N_R, 3); got {R.shape}")

    if weights_left is None:
        wL = np.ones(L.shape[0], dtype=np.float64)
    else:
        wL = np.asarray(weights_left, dtype=np.float64)
        if wL.shape != (L.shape[0],):
            raise ValueError(f"weights_left shape mismatch: {wL.shape} vs ({L.shape[0]},)")
    if weights_right is None:
        wR = np.ones(R.shape[0], dtype=np.float64)
    else:
        wR = np.asarray(weights_right, dtype=np.float64)
        if wR.shape != (R.shape[0],):
            raise ValueError(f"weights_right shape mismatch: {wR.shape} vs ({R.shape[0]},)")

    perp_edges = np.asarray(r_perp_bins, dtype=np.float64)
    par_edges = np.asarray(r_par_bins_signed, dtype=np.float64)
    if perp_edges.ndim != 1 or par_edges.ndim != 1:
        raise ValueError("bin edge arrays must be 1-D")
    if not np.all(np.diff(perp_edges) > 0) or not np.all(np.diff(par_edges) > 0):
        raise ValueError("bin edges must be strictly increasing")

    n_perp = perp_edges.size - 1
    n_par = par_edges.size - 1
    counts = np.zeros((n_perp, n_par), dtype=np.float64)
    npairs = np.zeros((n_perp, n_par), dtype=np.int64)

    # We chunk over the right side; each chunk is fully vectorised.
    nL = L.shape[0]
    nR = R.shape[0]
    rows = np.arange(nL)

    for r_start in range(0, nR, chunk_size):
        r_end = min(nR, r_start + chunk_size)
        Rc = R[r_start:r_end]               # (Nc, 3)
        wRc = wR[r_start:r_end]             # (Nc,)
        # Δ = L[:, None, :] − Rc[None, :, :]   shape (NL, Nc, 3)
        delta = L[:, None, :] - Rc[None, :, :]
        delta = delta - box * np.round(delta / box)   # minimum image, vectorised

        # signed r_par along left's LOS axis
        # delta[i, j, ax[i]] for all i, j → use fancy indexing
        # rows broadcast: delta[rows[:, None], np.arange(Nc)[None, :], ax[:, None]]
        cols = np.arange(r_end - r_start)
        r_par = delta[rows[:, None], cols[None, :], ax[:, None]]   # (NL, Nc)

        # |Δ|² and r_perp
        d2 = (delta * delta).sum(axis=2)                            # (NL, Nc)
        perp_sq = np.maximum(d2 - r_par * r_par, 0.0)
        r_perp = np.sqrt(perp_sq)

        # Pair weights: w_L[i] * w_R[j] → outer product
        w_pair = wL[:, None] * wRc[None, :]                         # (NL, Nc)

        if exclude_self:
            # Mark self-pairs (only possible when left and right are the same array)
            j_global = np.arange(r_start, r_end)                   # (Nc,)
            self_mask = (j_global[None, :] == rows[:, None]).ravel()  # (NL*Nc,)
        else:
            self_mask = None

        # Bin via np.digitize: indices into the bin edges
        i_perp = np.searchsorted(perp_edges, r_perp.ravel(), side="right") - 1
        i_par = np.searchsorted(par_edges, r_par.ravel(), side="right") - 1

        valid = (i_perp >= 0) & (i_perp < n_perp) & (i_par >= 0) & (i_par < n_par)
        w_flat = w_pair.ravel()
        if self_mask is not None:
            # Zero out self-pair weights so they don't contribute to counts.
            w_flat = np.where(self_mask, 0.0, w_flat)
            # Also drop self-pairs from npairs — Copilot review #8 on PR #7
            # caught the original bug where self-pairs counted toward npairs
            # despite being zero-weighted in counts, biasing xi = counts/npairs
            # at r ≈ 0 for auto-correlations.
            valid_for_npairs = valid & (~self_mask)
        else:
            valid_for_npairs = valid

        # Drop zero-weight pairs early so np.add.at doesn't waste work.
        nonzero = valid & (w_flat != 0.0)

        np.add.at(
            counts,
            (i_perp[nonzero], i_par[nonzero]),
            w_flat[nonzero],
        )
        np.add.at(
            npairs,
            (i_perp[valid_for_npairs], i_par[valid_for_npairs]),
            1,
        )

    return counts, npairs


# ---------------------------------------------------------------------------
# (r, |μ|)-binned pair counter — unbiased multipole extraction
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# `pair_count_2d` bins by (r_⊥, r_∥).  Inside an r-shell at fixed r,
# the per-μ density of npairs is *not* uniform — it is ∝ √(1−μ²)
# because the natural cell volume at fixed r is dV = 2π r² √(1−μ²) dr dμ.
# Using npairs as the weight when projecting ξ onto Legendre
# polynomials therefore evaluates ⟨L_ℓ⟩_npairs instead of the standard
# uniform-μ ⟨L_ℓ⟩ that the Hamilton 1992 multipole formula assumes.
# For ℓ = 2 this leaks ξ^(0) into the recovered ξ^(2) by ~ −1/8.
#
# Binning pairs directly in (r, |μ|) eliminates the Jacobian: every
# pair lands in its own (r_bin, μ_bin) without the volume distortion,
# so a uniform-μ Hamilton sum gives the right multipoles.
#
# See `docs/clustering_multipole_jacobian_todo.md` for the full
# diagnosis (Option A is what's implemented here).

def pair_count_rmu(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_los_axis: np.ndarray,
    box: float,
    r_bins: np.ndarray,
    mu_bins: np.ndarray,
    weights_left: Optional[np.ndarray] = None,
    weights_right: Optional[np.ndarray] = None,
    exclude_self: bool = False,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair counter binned by (r, |μ|), where r = |Δ⃗| and μ = (Δ⃗·ê_LOS)/r.

    Same chunking + minimum-image convention as ``pair_count_2d``.
    The LOS reference is the **left** point's sightline axis, matching
    the cross-correlation convention of placing pixels on the left so
    r_∥ is along the Lyα pixel's sightline.

    Pairs at exactly r = 0 (self-pairs) carry an undefined μ; they are
    dropped from both ``counts`` and ``npairs`` regardless of the
    ``exclude_self`` flag (this matches what auto-correlations want;
    cross-correlations don't have r = 0 pairs unless a DLA happens to
    coincide with the centre of a pixel).

    Parameters
    ----------
    left_xyz, right_xyz : (N_L, 3), (N_R, 3) float64
        Positions in Mpc/h.
    left_los_axis : (N_L,) int (0-indexed)
    box : float
        Periodic box side, Mpc/h.
    r_bins : (n_r + 1,) float
        Edges in Mpc/h, ≥ 0, monotonically increasing.
    mu_bins : (n_mu + 1,) float
        Edges in [0, 1], monotonically increasing.  We bin |μ| only:
        ξ(r, μ) = ξ(r, −μ) by reflection symmetry, so binning |μ| just
        doubles the per-bin pair count without biasing — this matches
        the picca / RascalC convention.
    weights_left, weights_right, exclude_self, chunk_size :
        Same semantics as ``pair_count_2d``.

    Returns
    -------
    counts  : (n_r, n_mu) float64 — Σ w_pair per bin.
    npairs  : (n_r, n_mu) int64   — pair count per bin (also drops r=0).
    """
    L = np.asarray(left_xyz, dtype=np.float64)
    R = np.asarray(right_xyz, dtype=np.float64)
    ax = np.asarray(left_los_axis, dtype=np.int64)
    if L.shape[0] != ax.shape[0]:
        raise ValueError("left_xyz and left_los_axis length mismatch")
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(f"left_xyz must be (N_L, 3); got {L.shape}")
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError(f"right_xyz must be (N_R, 3); got {R.shape}")

    if weights_left is None:
        wL = np.ones(L.shape[0], dtype=np.float64)
    else:
        wL = np.asarray(weights_left, dtype=np.float64)
        if wL.shape != (L.shape[0],):
            raise ValueError(f"weights_left shape mismatch: {wL.shape} vs ({L.shape[0]},)")
    if weights_right is None:
        wR = np.ones(R.shape[0], dtype=np.float64)
    else:
        wR = np.asarray(weights_right, dtype=np.float64)
        if wR.shape != (R.shape[0],):
            raise ValueError(f"weights_right shape mismatch: {wR.shape} vs ({R.shape[0]},)")

    r_edges = np.asarray(r_bins, dtype=np.float64)
    mu_edges = np.asarray(mu_bins, dtype=np.float64)
    if r_edges.ndim != 1 or mu_edges.ndim != 1:
        raise ValueError("bin edge arrays must be 1-D")
    if not np.all(np.diff(r_edges) > 0) or not np.all(np.diff(mu_edges) > 0):
        raise ValueError("bin edges must be strictly increasing")
    if r_edges[0] < 0:
        raise ValueError(f"r_bins must start at >= 0; got {r_edges[0]}")
    if mu_edges[0] < 0.0 or mu_edges[-1] > 1.0 + 1e-12:
        raise ValueError(
            f"mu_bins must lie in [0, 1] (we bin |μ|); got [{mu_edges[0]}, {mu_edges[-1]}]"
        )

    n_r = r_edges.size - 1
    n_mu = mu_edges.size - 1
    counts = np.zeros((n_r, n_mu), dtype=np.float64)
    npairs = np.zeros((n_r, n_mu), dtype=np.int64)

    nL = L.shape[0]
    nR = R.shape[0]
    rows = np.arange(nL)

    for r_start in range(0, nR, chunk_size):
        r_end = min(nR, r_start + chunk_size)
        Rc = R[r_start:r_end]
        wRc = wR[r_start:r_end]
        delta = L[:, None, :] - Rc[None, :, :]
        delta = delta - box * np.round(delta / box)             # minimum image

        cols = np.arange(r_end - r_start)
        r_par = delta[rows[:, None], cols[None, :], ax[:, None]]   # (NL, Nc) signed
        d2 = (delta * delta).sum(axis=2)                            # (NL, Nc)
        # r and |μ| — guard against r == 0 (self-pairs etc.).
        r_3d = np.sqrt(d2)
        finite_r = r_3d > 0
        # Compute |μ| only where r > 0; elsewhere set to 0 and let the
        # `valid` mask drop the bin.
        abs_mu = np.zeros_like(r_3d)
        np.divide(np.abs(r_par), r_3d, out=abs_mu, where=finite_r)
        # Clip just under 1.0 so |μ| == 1 (pure LOS pair) lands INSIDE
        # the rightmost μ-bin instead of being excluded by the `i_mu <
        # n_mu` boundary check (np.searchsorted with side="right"
        # places exact upper-edge matches above the last bin).  Also
        # absorbs the 1-ulp |Δ_par| > |Δ| floating-point overflow.
        abs_mu = np.minimum(abs_mu, 1.0 - 1e-12)

        w_pair = wL[:, None] * wRc[None, :]

        if exclude_self:
            j_global = np.arange(r_start, r_end)
            self_mask = (j_global[None, :] == rows[:, None]).ravel()
        else:
            self_mask = None

        i_r = np.searchsorted(r_edges, r_3d.ravel(), side="right") - 1
        i_mu = np.searchsorted(mu_edges, abs_mu.ravel(), side="right") - 1

        # Drop r==0 pairs (undefined μ) and out-of-range bins.
        valid = (
            finite_r.ravel()
            & (i_r >= 0) & (i_r < n_r)
            & (i_mu >= 0) & (i_mu < n_mu)
        )
        w_flat = w_pair.ravel()
        if self_mask is not None:
            w_flat = np.where(self_mask, 0.0, w_flat)
            valid_for_npairs = valid & (~self_mask)
        else:
            valid_for_npairs = valid

        nonzero = valid & (w_flat != 0.0)

        np.add.at(
            counts,
            (i_r[nonzero], i_mu[nonzero]),
            w_flat[nonzero],
        )
        np.add.at(
            npairs,
            (i_r[valid_for_npairs], i_mu[valid_for_npairs]),
            1,
        )

    return counts, npairs


# ---------------------------------------------------------------------------
# ξ_× : DLA point catalog × Lyα flux field
# ---------------------------------------------------------------------------

def xi_cross_dla_lya(
    pixel_xyz: np.ndarray,
    pixel_los_axis: np.ndarray,
    pixel_delta_F: np.ndarray,
    dla_xyz: np.ndarray,
    box: float,
    r_perp_bins: np.ndarray,
    r_par_bins_signed: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation ξ(r_par, r_perp) of DLAs with the Lyman-α flux field.

    Estimator (FR+2012 eq. 5):

        ξ_×(r_par, r_perp) = Σ_{(d, ℓ)} δ_F_ℓ  /  N_pairs

    The LOS reference is the **pixel's** sightline axis: pixels are
    placed on the left of the pair counter so ``r_par`` is along the
    pixel's sightline.

    Parameters
    ----------
    pixel_xyz       : (N_pix, 3) float — pixel positions in Mpc/h
    pixel_los_axis  : (N_pix,) int     — pixel sightline axis (0-indexed)
    pixel_delta_F   : (N_pix,) float   — δ_F = F/⟨F⟩ - 1 per pixel
    dla_xyz         : (N_DLA, 3) float — DLA positions in Mpc/h
    box             : float            — periodic box, Mpc/h
    r_perp_bins, r_par_bins_signed : bin edges (Mpc/h)
    chunk_size      : int              — DLA-batch for memory

    Returns
    -------
    xi_signed : (n_perp, n_par_signed) — the signed-r_par estimator
    counts    : (n_perp, n_par_signed) — Σ δ_F per bin
    npairs    : (n_perp, n_par_signed) — pair count per bin
    """
    counts, npairs = pair_count_2d(
        left_xyz=pixel_xyz,
        right_xyz=dla_xyz,
        left_los_axis=pixel_los_axis,
        box=box,
        r_perp_bins=r_perp_bins,
        r_par_bins_signed=r_par_bins_signed,
        weights_left=pixel_delta_F,
        weights_right=None,
        exclude_self=False,
        chunk_size=chunk_size,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = np.where(npairs > 0, counts / npairs, np.nan)
    return xi, counts, npairs


def xi_cross_dla_lya_rmu(
    pixel_xyz: np.ndarray,
    pixel_los_axis: np.ndarray,
    pixel_delta_F: np.ndarray,
    dla_xyz: np.ndarray,
    box: float,
    r_bins: np.ndarray,
    mu_bins: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, |μ|)-binned variant of ``xi_cross_dla_lya``.

    Same estimator (FR+2012 eq. 5: Σ δ_F over pairs / N_pairs), same
    LOS-axis convention (pixel side = left), but binned in (r, |μ|)
    so the output can be fed straight to
    ``hcd_analysis.lya_bias.extract_multipoles_rmu`` for unbiased
    Hamilton multipole extraction.
    """
    counts, npairs = pair_count_rmu(
        left_xyz=pixel_xyz,
        right_xyz=dla_xyz,
        left_los_axis=pixel_los_axis,
        box=box,
        r_bins=r_bins,
        mu_bins=mu_bins,
        weights_left=pixel_delta_F,
        weights_right=None,
        exclude_self=False,
        chunk_size=chunk_size,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = np.where(npairs > 0, counts / npairs, np.nan)
    return xi, counts, npairs


def fold_signed_to_abs(
    xi_signed: np.ndarray,
    r_par_bins_signed: np.ndarray,
    counts: Optional[np.ndarray] = None,
    npairs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold ξ(+r_par, r_perp) and ξ(-r_par, r_perp) into ξ(|r_par|, r_perp).

    For estimators where both halves of the signed grid have the same
    pair count (e.g. auto-correlations on isotropic fields), an
    unweighted nanmean is correct.  For estimators where the two
    halves can differ in pair count (the cross-correlation,
    statistically symmetric but not bit-exact, especially with small
    subsamples), pass ``counts`` and ``npairs`` so the fold is the
    proper pair-count-weighted

        xi_folded = (counts_pos + counts_neg) / (npairs_pos + npairs_neg)

    instead of ``mean(xi_pos, xi_neg)``.  This was Copilot review #13
    on PR #7.

    Parameters
    ----------
    xi_signed         : (n_perp, n_par_signed)
    r_par_bins_signed : (n_par_signed + 1,) symmetric around 0
    counts, npairs    : (n_perp, n_par_signed) raw counts/npairs from
        ``pair_count_2d``.  Both must be passed if either is — used
        only for the count-weighted fold path.

    Returns
    -------
    xi_folded     : (n_perp, n_par_abs)
    r_par_bins_abs: (n_par_abs + 1,) edges, ≥ 0
    """
    edges = np.asarray(r_par_bins_signed, dtype=np.float64)
    if not np.allclose(edges, -edges[::-1]):
        raise ValueError(
            f"r_par_bins_signed must be symmetric around 0; got {edges}"
        )
    n = edges.size - 1
    if n % 2 != 0:
        raise ValueError(f"need an even number of signed r_par bins; got {n}")
    half = n // 2

    if counts is not None or npairs is not None:
        if counts is None or npairs is None:
            raise ValueError(
                "must pass BOTH counts and npairs (or neither) to fold_signed_to_abs"
            )
        c_folded = counts[:, half:] + counts[:, :half][:, ::-1]
        n_folded = npairs[:, half:] + npairs[:, :half][:, ::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_folded = np.where(n_folded > 0, c_folded / n_folded, np.nan)
    else:
        pos = xi_signed[:, half:]                       # bins for +r_par
        neg = xi_signed[:, :half][:, ::-1]              # bins for -r_par, reversed
        # Unweighted nanmean — correct only when n_pairs is the same on
        # both halves (auto-corr on a symmetric grid).  For ξ_× pass the
        # ``counts``/``npairs`` arrays for a proper count-weighted fold.
        stacked = np.stack([pos, neg], axis=0)
        xi_folded = np.nanmean(stacked, axis=0)
    edges_abs = edges[half:]
    return xi_folded, edges_abs


# ---------------------------------------------------------------------------
# ξ_DD : DLA × DLA auto-correlation with analytic RR
# ---------------------------------------------------------------------------

def _bin_volumes_2d(
    r_perp_bins: np.ndarray,
    r_par_bins_signed: np.ndarray,
) -> np.ndarray:
    """Volume V_bin(r_perp, r_par) = π·(r_perp_hi² − r_perp_lo²)·Δr_par.

    For the **signed** r_par grid this gives the volume of the
    (r_perp, signed-r_par) cylindrical-shell bin in Mpc/h³.  The signed
    grid double-counts the symmetric volume (positive and negative
    r_par each get half).
    """
    perp = np.asarray(r_perp_bins, dtype=np.float64)
    par = np.asarray(r_par_bins_signed, dtype=np.float64)
    perp_area = np.pi * (perp[1:] ** 2 - perp[:-1] ** 2)            # (n_perp,)
    par_width = np.diff(par)                                          # (n_par,)
    return perp_area[:, None] * par_width[None, :]                    # (n_perp, n_par)


# ---------------------------------------------------------------------------
# δ_F field builder (all-HCD masked, per docs/clustering_definitions.md §2)
# ---------------------------------------------------------------------------


@dataclass
class DeltaFResult:
    """Output of build_delta_F_field — the all-HCD-masked Lyα flux field.

    Attributes
    ----------
    delta_F : (n_skewers, n_pix) float64
        F_pix / ⟨F⟩ − 1 on UNMASKED pixels; 0.0 on MASKED pixels.
        Setting masked pixels to 0 means they contribute nothing to
        ⟨δ_F · weight⟩ pair sums.
    mean_F : float
        ⟨F⟩ averaged over the UNMASKED pixels only.
    mask : (n_skewers, n_pix) bool
        True where the pixel is INSIDE any LLS / subDLA / DLA absorber's
        [pix_start, pix_end] range (i.e. the pixel was masked).
    n_masked_per_class : dict[str, int]
        Per-class pixel-mask counts: keys "LLS", "subDLA", "DLA".  Sum
        is total *newly-masked* pixels at each absorber (overlap is
        not double-counted: a pixel marked by an LLS first and re-
        covered by an overlapping subDLA increments only LLS).
    """
    delta_F: np.ndarray
    mean_F: float
    mask: np.ndarray
    n_masked_per_class: dict


def _classify_log_nhi(log_nhi: float) -> Optional[str]:
    """Map log10(N_HI) to absorber class; None for sub-LLS or NaN."""
    if not np.isfinite(log_nhi) or log_nhi < 17.2:
        return None
    if log_nhi < 19.0:
        return "LLS"
    if log_nhi < 20.3:
        return "subDLA"
    return "DLA"


def build_delta_F_field(
    tau: np.ndarray,
    skewer_idx: np.ndarray,
    pix_start: np.ndarray,
    pix_end: np.ndarray,
    NHI: np.ndarray,
) -> DeltaFResult:
    """Build the all-HCD-masked Lyman-α flux field δ_F per pixel.

    Per `docs/clustering_definitions.md` §2:

        F_ij    = exp(−τ_ij)             on unmasked pixels
                = ⟨F⟩                     on masked pixels  → δ_F = 0
        ⟨F⟩     = mean F over UNMASKED pixels in the snap
        δ_F_ij  = F_ij / ⟨F⟩ − 1

    where ``masked`` is "covered by ANY catalog absorber's
    [pix_start, pix_end] with N_HI ≥ 10^17.2".  Filling masked pixels
    with ⟨F⟩ → δ_F = 0 means they contribute nothing to ⟨δ_F · w⟩
    pair sums in ξ_× and ξ_FF — which is the right behaviour
    physically (we want the underlying *forest* bias, not the bias of
    the HCDs we already cross-correlate against in ξ_×).

    Parameters
    ----------
    tau : (n_skewers, n_pix) float
        Optical depth array, e.g. ``f["tau/H/1/1215"][:]``.
    skewer_idx, pix_start, pix_end : (n_abs,) int
        Per-absorber row in the catalog.  ``pix_start`` and
        ``pix_end`` are inclusive (matching the production
        ``find_systems_in_skewer`` convention).
    NHI : (n_abs,) float
        Per-absorber recovered N_HI in cm^-2; used only to classify
        absorbers (LLS / subDLA / DLA) for diagnostic counts.  All
        absorbers with log10(N_HI) ≥ 17.2 are masked.

    Returns
    -------
    DeltaFResult
    """
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim != 2:
        raise ValueError(f"tau must be (n_skewers, n_pix); got {tau.shape}")
    n_skewers, n_pix = tau.shape

    skewer_idx = np.asarray(skewer_idx, dtype=np.int64)
    pix_start = np.asarray(pix_start, dtype=np.int64)
    pix_end = np.asarray(pix_end, dtype=np.int64)
    NHI = np.asarray(NHI, dtype=np.float64)
    if not (pix_start.shape == pix_end.shape == NHI.shape == skewer_idx.shape):
        raise ValueError("catalog arrays must all have the same length")

    mask = np.zeros((n_skewers, n_pix), dtype=bool)
    n_masked_per_class = {"LLS": 0, "subDLA": 0, "DLA": 0}

    for i in range(skewer_idx.shape[0]):
        cls = _classify_log_nhi(np.log10(max(NHI[i], 1.0)))
        if cls is None:
            continue
        sk = int(skewer_idx[i])
        s = int(pix_start[i])
        e = int(pix_end[i])
        if not (0 <= sk < n_skewers):
            continue
        s_c = max(0, s)
        e_c = min(n_pix - 1, e)
        if e_c < s_c:
            continue
        before = int(mask[sk, s_c : e_c + 1].sum())
        mask[sk, s_c : e_c + 1] = True
        after = int(mask[sk, s_c : e_c + 1].sum())
        n_masked_per_class[cls] += after - before

    F = np.exp(-tau)
    unmasked = ~mask
    if not unmasked.any():
        raise ValueError("all pixels are masked — cannot compute ⟨F⟩")
    mean_F = float(F[unmasked].mean())
    delta_F = np.zeros_like(F)
    delta_F[unmasked] = F[unmasked] / mean_F - 1.0
    # Masked pixels stay at 0.0 (= ⟨F⟩ / ⟨F⟩ − 1 = 0).

    return DeltaFResult(
        delta_F=delta_F,
        mean_F=mean_F,
        mask=mask,
        n_masked_per_class=n_masked_per_class,
    )


def xi_auto_dla(
    dla_xyz: np.ndarray,
    dla_los_axis: np.ndarray,
    box: float,
    r_perp_bins: np.ndarray,
    r_par_bins_signed: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DLA × DLA auto-correlation on a periodic box with analytic RR.

    Estimator:
        ξ_DD(r_par, r_perp) = DD(r_par, r_perp) / RR_analytic - 1

    where
        RR_analytic = N · (N − 1) · V_bin / V_box.

    Self-pairs (i == j) are excluded.  ``dla_los_axis`` defines the
    LOS axis for each DLA — for the auto-correlation we use the DLA's
    parent-sightline axis on the **left** side of the pair.  The
    asymmetry between left and right is OK on the periodic box because
    pairs are counted twice (once with i on left, once with i on
    right) and the LOS-axis choice averages out at the bin level.

    Parameters
    ----------
    dla_xyz       : (N, 3) float
    dla_los_axis  : (N,) int (0-indexed)
    box           : float
    r_perp_bins, r_par_bins_signed : bin edges

    Returns
    -------
    xi_signed : (n_perp, n_par_signed) DD/RR_analytic - 1
    DD        : (n_perp, n_par_signed) raw pair counts
    RR        : (n_perp, n_par_signed) analytic random-pair counts
    """
    n_dla = dla_xyz.shape[0]
    if n_dla < 2:
        raise ValueError(f"need ≥ 2 DLAs for auto-corr; got {n_dla}")

    DD_counts, _ = pair_count_2d(
        left_xyz=dla_xyz,
        right_xyz=dla_xyz,
        left_los_axis=dla_los_axis,
        box=box,
        r_perp_bins=r_perp_bins,
        r_par_bins_signed=r_par_bins_signed,
        weights_left=None,
        weights_right=None,
        exclude_self=True,
        chunk_size=chunk_size,
    )
    DD = DD_counts                                                  # (n_perp, n_par)

    V_box = box ** 3
    V_bin = _bin_volumes_2d(r_perp_bins, r_par_bins_signed)          # (n_perp, n_par)
    RR = n_dla * (n_dla - 1) * V_bin / V_box

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR > 0, DD / RR - 1.0, np.nan)
    return xi, DD, RR


def _bin_volumes_rmu(
    r_bins: np.ndarray,
    mu_bins: np.ndarray,
) -> np.ndarray:
    """Volume V_bin(r, μ) = (4π/3)·(r_hi³ − r_lo³)·Δμ on the |μ| ∈ [0,1] grid.

    Each (r, |μ|) shell sweeps both hemispheres of μ (we bin |μ|, so
    each bin actually represents 2·Δμ of the full [-1, 1] range).  The
    volume of an annular shell at fixed r covering Δμ on |μ| is

        dV = ∫_shell d³x = ∫_r0^r1 4π r² dr · 2 · Δμ_abs / 2
           = (4π/3)·(r1³ − r0³) · Δμ_abs

    The factor of 2 from "both hemispheres" cancels the factor of
    1/2 you'd get if you parameterised by signed μ ∈ [−1, 1] and
    only this hemisphere — net result (4π/3)·(r1³ − r0³)·Δμ_abs.
    """
    r = np.asarray(r_bins, dtype=np.float64)
    mu = np.asarray(mu_bins, dtype=np.float64)
    r_vol = (4.0 / 3.0) * np.pi * (r[1:] ** 3 - r[:-1] ** 3)        # (n_r,)
    mu_width = np.diff(mu)                                           # (n_mu,)
    return r_vol[:, None] * mu_width[None, :]                        # (n_r, n_mu)


def xi_auto_dla_rmu(
    dla_xyz: np.ndarray,
    dla_los_axis: np.ndarray,
    box: float,
    r_bins: np.ndarray,
    mu_bins: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, |μ|)-binned variant of ``xi_auto_dla``.

    Same DD/RR_analytic − 1 estimator on the periodic box, but binned
    in (r, |μ|) for unbiased multipole extraction.  Self-pairs are
    excluded.

    Returns
    -------
    xi : (n_r, n_mu)  DD/RR_analytic - 1
    DD : (n_r, n_mu)  raw pair counts
    RR : (n_r, n_mu)  analytic random-pair counts
    """
    n_dla = dla_xyz.shape[0]
    if n_dla < 2:
        raise ValueError(f"need ≥ 2 DLAs for auto-corr; got {n_dla}")

    DD, _ = pair_count_rmu(
        left_xyz=dla_xyz,
        right_xyz=dla_xyz,
        left_los_axis=dla_los_axis,
        box=box,
        r_bins=r_bins,
        mu_bins=mu_bins,
        weights_left=None,
        weights_right=None,
        exclude_self=True,
        chunk_size=chunk_size,
    )

    V_box = box ** 3
    V_bin = _bin_volumes_rmu(r_bins, mu_bins)
    RR = n_dla * (n_dla - 1) * V_bin / V_box

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR > 0, DD / RR - 1.0, np.nan)
    return xi, DD, RR


# ---------------------------------------------------------------------------
# ξ_FF : Lyα × Lyα flux auto-correlation
# ---------------------------------------------------------------------------

def xi_auto_lya(
    pixel_xyz: np.ndarray,
    pixel_los_axis: np.ndarray,
    pixel_delta_F: np.ndarray,
    box: float,
    r_perp_bins: np.ndarray,
    r_par_bins_signed: np.ndarray,
    subsample_n: Optional[int] = None,
    rng_seed: int = 0,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lyman-α × Lyman-α flux auto-correlation ξ_FF(r_par, r_perp).

    Estimator (per docs/clustering_definitions.md §5b):

        ξ_FF(r_par, r_perp) = Σ_{(ℓ, ℓ') ∈ bin}  δ_F_ℓ · δ_F_ℓ'
                              ──────────────────────────────────
                                       N_pairs in bin

    Both legs of the pair are pixels carrying the δ_F weight.  Self-
    pairs (ℓ == ℓ') are excluded.  The LOS reference for each pair's
    r_par is the LEFT pixel's sightline axis.

    For HiRes-resolution snaps (≳ 10⁷ pixels), the full
    pixel × pixel pair count is too expensive (~ 10¹⁴ pairs) — pass
    ``subsample_n`` to randomly select that many pixels per side
    (seeded by ``rng_seed`` for reproducibility).  A 10⁵ subsample
    gives ~ 10¹⁰ pairs total which is tractable in seconds-to-minutes
    and recovers the linear-scale signal well above the noise floor.

    Parameters
    ----------
    pixel_xyz       : (N, 3) float — pixel positions in Mpc/h
    pixel_los_axis  : (N,) int     — pixel sightline axis (0-indexed)
    pixel_delta_F   : (N,) float   — δ_F per pixel
    box             : float        — periodic box, Mpc/h
    r_perp_bins, r_par_bins_signed : bin edges (Mpc/h)
    subsample_n     : optional int  — randomly down-sample to this many pixels
    rng_seed        : int          — RNG seed for reproducibility
    chunk_size      : int          — right-side chunking for memory

    Returns
    -------
    xi_signed : (n_perp, n_par) — Σ δ_F · δ_F / N_pairs per bin
    counts    : (n_perp, n_par) — Σ δ_F · δ_F per bin
    npairs    : (n_perp, n_par) — pair count per bin
    """
    pixel_xyz = np.asarray(pixel_xyz, dtype=np.float64)
    pixel_los_axis = np.asarray(pixel_los_axis, dtype=np.int64)
    pixel_delta_F = np.asarray(pixel_delta_F, dtype=np.float64)
    if pixel_xyz.ndim != 2 or pixel_xyz.shape[1] != 3:
        raise ValueError(f"pixel_xyz must be (N, 3); got {pixel_xyz.shape}")
    n_total = pixel_xyz.shape[0]
    if pixel_los_axis.shape != (n_total,) or pixel_delta_F.shape != (n_total,):
        raise ValueError("pixel arrays must all have length N")

    if subsample_n is not None and subsample_n < n_total:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n_total, size=int(subsample_n), replace=False)
        idx.sort()                                          # cache-friendly
        L_xyz = pixel_xyz[idx]
        L_axis = pixel_los_axis[idx]
        L_dF = pixel_delta_F[idx]
    else:
        L_xyz = pixel_xyz
        L_axis = pixel_los_axis
        L_dF = pixel_delta_F

    counts, npairs = pair_count_2d(
        left_xyz=L_xyz,
        right_xyz=L_xyz,
        left_los_axis=L_axis,
        box=box,
        r_perp_bins=r_perp_bins,
        r_par_bins_signed=r_par_bins_signed,
        weights_left=L_dF,
        weights_right=L_dF,
        exclude_self=True,
        chunk_size=chunk_size,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = np.where(npairs > 0, counts / npairs, np.nan)
    return xi, counts, npairs


def xi_auto_lya_rmu(
    pixel_xyz: np.ndarray,
    pixel_los_axis: np.ndarray,
    pixel_delta_F: np.ndarray,
    box: float,
    r_bins: np.ndarray,
    mu_bins: np.ndarray,
    subsample_n: Optional[int] = None,
    rng_seed: int = 0,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, |μ|)-binned variant of ``xi_auto_lya``.

    Same Σ δ_F δ_F / N_pairs estimator, same subsample mechanic, same
    self-pair exclusion, same LOS-axis convention (left = pixel) — but
    binned in (r, |μ|) so multipoles can be extracted with the
    Hamilton uniform-μ formula via ``extract_multipoles_rmu``.
    """
    pixel_xyz = np.asarray(pixel_xyz, dtype=np.float64)
    pixel_los_axis = np.asarray(pixel_los_axis, dtype=np.int64)
    pixel_delta_F = np.asarray(pixel_delta_F, dtype=np.float64)
    if pixel_xyz.ndim != 2 or pixel_xyz.shape[1] != 3:
        raise ValueError(f"pixel_xyz must be (N, 3); got {pixel_xyz.shape}")
    n_total = pixel_xyz.shape[0]
    if pixel_los_axis.shape != (n_total,) or pixel_delta_F.shape != (n_total,):
        raise ValueError("pixel arrays must all have length N")

    if subsample_n is not None and subsample_n < n_total:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n_total, size=int(subsample_n), replace=False)
        idx.sort()
        L_xyz = pixel_xyz[idx]
        L_axis = pixel_los_axis[idx]
        L_dF = pixel_delta_F[idx]
    else:
        L_xyz = pixel_xyz
        L_axis = pixel_los_axis
        L_dF = pixel_delta_F

    counts, npairs = pair_count_rmu(
        left_xyz=L_xyz,
        right_xyz=L_xyz,
        left_los_axis=L_axis,
        box=box,
        r_bins=r_bins,
        mu_bins=mu_bins,
        weights_left=L_dF,
        weights_right=L_dF,
        exclude_self=True,
        chunk_size=chunk_size,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = np.where(npairs > 0, counts / npairs, np.nan)
    return xi, counts, npairs
