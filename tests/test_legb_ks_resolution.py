"""Task #5 V1 -- KS spectral-resolution f_res float (TDD, loader/cov machinery).

KS n_s FAILS the NORC res_corr-drop gate (0.385 sigma_ref) because it is R_z-blocked (reuses the
DESI proxy R_z ~15x too large) and cannot float f_res. V1 gives KS its OWN echelle resolution:
  - ks_resolution_R(z) PINNED to the KODIAQ+SQUAD echelle sigma_v ~3.2 km/s (z-independent).
  - the resolution systematic (esyst_res_ks) REMOVED from the conservative cov by a DIAGONAL subtract
    (diag -= esyst_res_ks^2) -- NOT the eBOSS "rescale": the KS paper adds systematics to the DIAGONAL
    only, so the off-diagonal (stat) has no resolution term (design-pair adjudication 2026-07-07).
  - f_res then floats in the forward.
Default load_ks_leg (resolution_float=False) stays byte-identical (proxy R_z unused, no surgery).

These are pure-numpy loader tests (no emulator / NUTS). Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_ks_resolution.py -q
"""
import os

import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

_KS_BASE = "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
_HAVE_KS = os.path.exists(_KS_BASE + "final-conservative-covariance-karacayli_etal2021.txt")
pytestmark = pytest.mark.skipif(not _HAVE_KS, reason="KS KODIAQ-SQUAD data files not present")


def _raw_grid():
    z, k, P = DL._read_ks_p1d(_KS_BASE + "final-conservative-p1d-karacayli_etal2021.txt")
    return z, k


# --------------------------------------------------------------------------------------------- #
#  1. R_z pinned to the echelle sigma (z-independent 3.2 km/s), replacing the DESI proxy.
# --------------------------------------------------------------------------------------------- #
def test_ks_resolution_R_pinned_constant():
    z = np.array([2.4, 3.0, 3.6, 4.6])
    R = np.asarray(DL.ks_resolution_R(z))
    assert R.shape == z.shape
    assert np.allclose(R, 3.2)                                  # pinned, z-independent
    # and it is genuinely different from the (too-large) DESI proxy it replaces
    assert np.asarray(DL.desi_resolution_R(3.0)) > 3.0 * R[1]   # proxy is >~15x larger


# --------------------------------------------------------------------------------------------- #
#  2. esyst_res_ks aligns 182/182 to the conservative (z,k) grid (merge, not positional).
# --------------------------------------------------------------------------------------------- #
def test_ks_esyst_aligns_182_all_finite():
    z, k = _raw_grid()
    e = DL._read_ks_resolution_e(_KS_BASE + "detailed-p1d-results-karacayli_etal2021.txt", z, k)
    assert e.shape == z.shape and np.all(np.isfinite(e))
    assert np.all(e > 0)


def test_ks_esyst_raises_on_unmatched_grid():
    # a bogus (z,k) row must fail LOUD (the alignment tripwire), never silently mis-map.
    z, k = _raw_grid()
    z_bad = z.copy(); z_bad[0] = 9.99
    with pytest.raises(KeyError):
        DL._read_ks_resolution_e(_KS_BASE + "detailed-p1d-results-karacayli_etal2021.txt", z_bad, k)


# --------------------------------------------------------------------------------------------- #
#  3/6. The DIAGONAL surgery removes EXACTLY esyst_res_ks^2 from the diagonal and leaves the
#       off-diagonal UNTOUCHED (the diag-vs-rescale distinction), and stays SPD.
# --------------------------------------------------------------------------------------------- #
def test_ks_diag_surgery_removes_exactly_esyst2_offdiag_untouched():
    off = DL.load_ks_leg(base=_KS_BASE, k_max=0.065)                       # default: no surgery
    on = DL.load_ks_leg(base=_KS_BASE, k_max=0.065, resolution_float=True)  # diag subtract
    Coff, Con = np.asarray(off.C_data), np.asarray(on.C_data)
    assert Coff.shape == Con.shape
    z, k = _raw_grid()
    e_full = DL._read_ks_resolution_e(_KS_BASE + "detailed-p1d-results-karacayli_etal2021.txt", z, k)
    keep = (z >= 2.4 - 1e-6) & (z <= 4.6 + 1e-6) & (k <= 0.065 + 1e-9)
    e_kept = e_full[keep]
    # diagonal: removed exactly e_res^2
    assert np.allclose(np.diag(Coff) - np.diag(Con), e_kept ** 2, rtol=1e-9, atol=1e-30)
    # off-diagonal: UNCHANGED (diag mode, not rescale)
    offdiag = ~np.eye(Coff.shape[0], dtype=bool)
    assert np.array_equal(Coff[offdiag], Con[offdiag])
    # SPD preserved
    np.linalg.cholesky(Con)
    assert np.all(np.diag(Con) > 0)


def test_ks_esyst_subdominant_no_clip():
    # the resolution variance is < the diagonal everywhere (so the subtract never drives diag<=0).
    off = DL.load_ks_leg(base=_KS_BASE, k_max=0.065)
    z, k = _raw_grid()
    e_full = DL._read_ks_resolution_e(_KS_BASE + "detailed-p1d-results-karacayli_etal2021.txt", z, k)
    keep = (z >= 2.4 - 1e-6) & (z <= 4.6 + 1e-6) & (k <= 0.065 + 1e-9)
    frac = e_full[keep] ** 2 / np.diag(np.asarray(off.C_data))
    assert frac.max() < 1.0                                     # never clips
    assert frac.max() < 0.5                                     # comfortably subdominant in-band


# --------------------------------------------------------------------------------------------- #
#  4. load_ks_leg default is byte-identical (proxy R_z, no surgery, resolution off/not-ready).
# --------------------------------------------------------------------------------------------- #
def test_load_ks_leg_default_byte_identical():
    leg = DL.load_ks_leg(base=_KS_BASE, k_max=0.065)
    assert leg.resolution_on is False and leg.resolution_ready is False
    assert np.allclose(np.asarray(leg.R_z), np.asarray(DL.desi_resolution_R(leg.z)))   # proxy, untouched
    # C_data == the raw conservative cov sub-block (no surgery)
    z, k, P = DL._read_ks_p1d(_KS_BASE + "final-conservative-p1d-karacayli_etal2021.txt")
    cov = np.loadtxt(_KS_BASE + "final-conservative-covariance-karacayli_etal2021.txt")
    keep = (z >= 2.4 - 1e-6) & (z <= 4.6 + 1e-6) & (k <= 0.065 + 1e-9)
    idx = np.where(keep)[0]
    assert np.array_equal(np.asarray(leg.C_data), cov[np.ix_(idx, idx)])


# --------------------------------------------------------------------------------------------- #
#  5/7. load_ks_leg(resolution_float=True): resolution on + ready, R_z=3.2, SPD (incl. high-z).
# --------------------------------------------------------------------------------------------- #
def test_load_ks_leg_fres_on_flags_and_R32():
    leg = DL.load_ks_leg(base=_KS_BASE, k_max=0.065, resolution_float=True)
    assert leg.resolution_on is True and leg.resolution_ready is True
    assert np.allclose(np.asarray(leg.R_z), 3.2)
    C = np.asarray(leg.C_data)
    np.linalg.cholesky(C)                                       # SPD over the full kept band (incl. He-II z)
    assert np.linalg.eigvalsh(C).min() > 0


# --------------------------------------------------------------------------------------------- #
#  Stage B: the RCINJ KS f_res GATE config (RCINJ_FRES=1) -- KS gets the echelle R_z + diag surgery
#  via ks_kwargs, at the env-driven width, NORC + metals-off; DESI/eBOSS arms unaffected.
# --------------------------------------------------------------------------------------------- #
def test_rcinj_ks_fres_gate_config_generates():
    import os
    from scripts.run_stepA import build_config
    _saved = {k: os.environ.get(k) for k in ("RCINJ_FRES", "RCINJ_KS_FRES_SIGMA")}
    os.environ["RCINJ_FRES"] = "1"
    os.environ["RCINJ_KS_FRES_SIGMA"] = "0.15"
    try:
        cfg = build_config()
    finally:
        for k, v in _saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    ksf = [c for c in cfg if c["id"].startswith("RCINJKF_")]
    assert ksf, "no RCINJKF (KS f_res) chains generated under RCINJ_FRES=1"
    c = ksf[0]
    assert c["survey"] == "KS" and c["sample_res"] is True
    assert c["f_res_amp_sigma"] == 0.15 and c["ks_resolution_float"] is True and c["ks_kmax"] == 0.065
    assert c["res_corr_on"] is False and c["sample_metals"] is False    # NORC + KS metals-off
    d = [c for c in cfg if c["id"].startswith("RCINJDF_")][0]           # DESI arm unaffected by the KS env
    assert d["f_res_amp_sigma"] == 0.02 and d["ks_resolution_float"] is False


def test_rcinj_ks_control_band_config():
    """RCINJ_KS_KMAX=0.065 (no RCINJ_FRES): the KS no-f_res CONTROL arm (RCINJK_) runs at k_max=0.065
    (proxy R_z, no surgery, no f_res) -> the honest 0.065 fixed sigma_ref + the 0.065 no-f_res baseline."""
    import os
    from scripts.run_stepA import build_config
    _saved = {k: os.environ.get(k) for k in ("RCINJ_FRES", "RCINJ_KS_KMAX")}
    os.environ.pop("RCINJ_FRES", None)
    os.environ["RCINJ_KS_KMAX"] = "0.065"
    try:
        cfg = build_config()
    finally:
        for k, v in _saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    ksc = [c for c in cfg if c["id"].startswith("RCINJK_")]            # the no-F KS control arm (tag "K")
    assert ksc, "no RCINJK (KS no-f_res control) chains"
    c = ksc[0]
    assert c["survey"] == "KS" and c["sample_res"] is False and c["ks_resolution_float"] is False
    assert c["ks_kmax"] == 0.065 and c["res_corr_on"] is False
