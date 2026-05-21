"""Thin wrapper around fake_spectra's P1D pipeline, used by the Phase-2
emulator cache builder. The functions in this module are intentionally
shallow — every P1D-relevant choice (filter, mean-flux inversion, FFT) is
delegated to fake_spectra so the cache rows are bit-identical to PRIYA's
flux_vector training data at matching (sim, z, alpha_slope).

Requires the emu-3.9 conda env with gsl loaded (libgsl.so.25 is dlopen'd by
fake_spectra._spectra_priv).
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from fake_spectra.fluxstatistics import (
    flux_power, obs_mean_tau, _powerspectrum, _flux_power_bins,
)
from fake_spectra.spectra import Spectra
# Unbound method, reads only `self.nbins`. Calling via a SimpleNamespace
# stand-in avoids constructing a full Spectra (which needs a snapshot dir).
_filter_single_tau_complex = Spectra._filter_single_tau_complex


def _apply_priya_filter(tau: np.ndarray, tau_thresh: float = 1.0e6,
                        thresh2: float = 0.25) -> np.ndarray:
    """In-place: apply fake_spectra._filter_single_tau_complex to every
    sightline with `max(tau) > tau_thresh`. Returns the same array."""
    tau_eff = -np.log(np.mean(np.exp(-tau)))
    ii = np.where(np.max(tau, axis=1) > tau_thresh)[0]
    if len(ii) == 0:
        return tau
    self_stub = SimpleNamespace(nbins=tau.shape[1])
    for i in ii:
        tau[i], _ = _filter_single_tau_complex(self_stub, tau[i], tau_eff,
                                               tau_thresh=tau_thresh, thresh2=thresh2)
    return tau


def compute_tier_p_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       tau_thresh: float = 1.0e6,
                       ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """PRIYA-compatible total P1D over all sightlines, after filter.

    Pipeline:
      1. mean_flux_desired = exp(-alpha_slope * obs_mean_tau_Kim2013(z))
      2. Apply _filter_single_tau_complex to tau (modifies in place).
      3. fake_spectra.fluxstatistics.flux_power(tau_filtered, vmax,
         mean_flux_desired=target_F, window=False, spec_res=0.0).
      4. Drop the k=0 mode (PRIYA convention).

    Returns (kf_kms, P1D, target_F, scale).
    """
    target_F = float(np.exp(-alpha_slope * obs_mean_tau(z)))
    _apply_priya_filter(tau, tau_thresh=tau_thresh)
    kf, P = flux_power(tau, vmax, spec_res=0.0,
                       mean_flux_desired=target_F, window=False)
    # scale (the actual tau-multiplier) is solved inside flux_power; we can
    # recover it by calling mean_flux explicitly (cheap — Newton iteration on
    # the same tau).
    from fake_spectra.fluxstatistics import mean_flux
    scale = float(mean_flux(tau, target_F))
    return kf[1:], P[1:], target_F, scale


def _classify_sightlines(catalog, n_skewers: int) -> Dict[str, np.ndarray]:
    """Return a dict of boolean masks, one per class label.

    Mirrors the highest-class promotion rule already used by
    hcd_analysis.p1d.compute_p1d_per_class (DLA wins over subDLA wins over
    LLS; everything else is "clean"). Uses each Absorber's pre-computed
    `absorber_class` field, NOT a re-thresholded log_NHI — the absorber's
    class is the authoritative label (Phase-1 catalog builder applied the
    thresholds at fit time).
    """
    labels = np.full(n_skewers, "clean", dtype=object)
    for ab in catalog.absorbers:
        if ab.skewer_idx >= n_skewers:
            continue
        if ab.absorber_class == "DLA":
            labels[ab.skewer_idx] = "DLA"
        elif ab.absorber_class == "subDLA":
            if labels[ab.skewer_idx] != "DLA":
                labels[ab.skewer_idx] = "subDLA"
        elif ab.absorber_class == "LLS":
            if labels[ab.skewer_idx] not in ("DLA", "subDLA"):
                labels[ab.skewer_idx] = "LLS"
    return {c: (labels == c) for c in ("clean", "LLS", "subDLA", "DLA")}


# Fine-N_HI Tier-C bins, anchored at the physical class boundaries (user
# decision 2026-05-21): LLS 17.2, subDLA 19.0, DLA 20.3 are EXACT edges, with
# linear ~0.26-dex spacing inside the LLS and subDLA bands and a coarse DLA
# tail (most DLAs are survey-masked; only the near-20.3 edge leaks in).
# Sightlines are binned by their HIGHEST absorber's log_NHI. Class layout (15):
#   0      : clean (no absorber >= 17.2)
#   1..7   : LLS    [17.2, 19.0)  (7 uniform bins)
#   8..12  : subDLA [19.0, 20.3)  (5 uniform bins)
#   13     : DLA edge [20.3, 21.0)
#   14     : DLA tail >= 21.0
# 17.2/19.0/20.3 are exact edges -> LLS/subDLA/DLA reconstruct exactly. P1D is
# sightline-additive, so any class is a count-weighted sum (merge_fine_to_classes).
_LLS_EDGES = np.linspace(17.2, 19.0, 8)        # 7 bins (~0.257 dex)
_SUBDLA_EDGES = np.linspace(19.0, 20.3, 6)     # 5 bins (0.26 dex)
_DLA_EDGES = np.array([20.3, 21.0])            # DLA edge bin; >=21.0 overflow
FINE_NHI_EDGES = np.round(
    np.unique(np.concatenate([_LLS_EDGES, _SUBDLA_EDGES, _DLA_EDGES])), 4)  # 14 edges
N_TIER_C_BINS = len(FINE_NHI_EDGES) + 1                                     # 15


def tier_c_labels():
    """Human-readable label per Tier-C class index (len == N_TIER_C_BINS)."""
    labs = ["clean"]
    for lo, hi in zip(FINE_NHI_EDGES[:-1], FINE_NHI_EDGES[1:]):
        labs.append(f"{lo:.2f}-{hi:.2f}")
    labs.append(f">={FINE_NHI_EDGES[-1]:.2f}")
    return labs


def bin_sightlines_by_nhi(catalog, n_skewers: int) -> np.ndarray:
    """Return an int class index (0..N_TIER_C_BINS-1) per sightline, by the
    sightline's highest-log_NHI absorber. 0 = clean (no absorber)."""
    maxnhi = np.full(n_skewers, -np.inf)
    for ab in catalog.absorbers:
        if (ab.skewer_idx < n_skewers and ab.log_NHI >= FINE_NHI_EDGES[0]
                and ab.log_NHI > maxnhi[ab.skewer_idx]):
            maxnhi[ab.skewer_idx] = ab.log_NHI
    cls = np.zeros(n_skewers, dtype=np.int64)         # clean = 0
    finite = np.isfinite(maxnhi)
    d = np.digitize(maxnhi[finite], FINE_NHI_EDGES)   # digitize 0..14; clipped to 1..14 below (0 only for sub-17.2)
    cls[finite] = np.clip(d, 1, len(FINE_NHI_EDGES))  # <17.2->1; >=21.0->14
    return cls


def _per_class_p1d_at_scale(tau_class: np.ndarray, vmax: float,
                             scale: float, target_F: float):
    """Mirror fake_spectra.fluxstatistics.flux_power but with predetermined
    (scale, target_F) — used by Tier C so all four classes share Tier P's
    mean-flux normalisation. Returns (kf[1:], P[1:]) like flux_power does."""
    nspec, npix = tau_class.shape
    if nspec == 0:
        kf = _flux_power_bins(vmax, npix)
        return kf[1:], np.zeros(npix // 2 + 1)[1:]
    mfp = np.zeros(npix // 2 + 1, dtype=tau_class.dtype)
    for i in range(10):
        end = min((i + 1) * nspec // 10, nspec)
        s = i * nspec // 10
        if end == s:
            continue
        dflux = np.exp(-scale * tau_class[s:end]) / target_F - 1.0
        mfp += vmax * np.sum(_powerspectrum(dflux, axis=1), axis=0)
    mfp /= nspec
    kf = _flux_power_bins(vmax, npix)
    return kf[1:], mfp[1:]


def compute_tier_c_p1d(tau: np.ndarray, vmax: float,
                       alpha_slope: float, z: float,
                       catalog,
                       external_scale: Optional[float] = None,
                       external_target_F: Optional[float] = None,
                       ):
    """Per-class P1Ds (clean, LLS, subDLA, DLA) on the UNFILTERED tau.

    If `external_scale` / `external_target_F` are given (the production path,
    set from a prior Tier-P call), use those so the four classes share Tier
    P's mean-flux normalisation. Otherwise compute target_F from
    obs_mean_tau(z) and solve scale on the full tau (independent
    Tier-C-only path, useful for unit tests).

    Returns:
        kf      : k-grid (s/km, angular), shape (npix//2,)
        by_class: dict[label] -> P1D array, shape (npix//2,)
        n_by_class: dict[label] -> sightline count
        target_F, scale (the values used)
    """
    from fake_spectra.fluxstatistics import mean_flux

    if external_target_F is None:
        target_F = float(np.exp(-alpha_slope * obs_mean_tau(z)))
    else:
        target_F = float(external_target_F)
    if external_scale is None:
        scale = float(mean_flux(tau, target_F))
    else:
        scale = float(external_scale)

    masks = _classify_sightlines(catalog, tau.shape[0])
    out: Dict[str, np.ndarray] = {}
    n_out: Dict[str, int] = {}
    kf_ref = None
    for label, mask in masks.items():
        kf, P = _per_class_p1d_at_scale(tau[mask], vmax, scale, target_F)
        if kf_ref is None:
            kf_ref = kf
        out[label] = P
        n_out[label] = int(mask.sum())
    return kf_ref, out, n_out, target_F, scale
