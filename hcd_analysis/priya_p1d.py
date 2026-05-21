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
from typing import Tuple

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
