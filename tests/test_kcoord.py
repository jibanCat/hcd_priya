"""Gate A, task 2: the canonical physical k coordinate k_skm(z, theta).
Incident: hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import kcoord as KC, schema as S
from hcd_analysis.emulator.data import PARAM_LIMITS, normalize_params

LF = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def _theta_unit(hub, omegamh2):
    phys = np.array(PARAM_LIMITS, float).mean(axis=1)      # mid-box for the other 7
    phys[5] = hub
    phys[6] = omegamh2
    return np.asarray(normalize_params(phys), float)


def test_first_principles_grid_matches_box_formula():
    kcom = 2 * np.pi * np.arange(1, 6) / S.L_BOX_HMPC
    for z, hub, omh2 in [(2.2, 0.735, 0.1405), (3.0, 0.652, 0.1413), (4.6, 0.748, 0.1429)]:
        om = omh2 / hub ** 2
        E = np.sqrt(om * (1 + z) ** 3 + 1 - om)
        expect = 2 * np.pi * np.arange(1, 6) / (S.L_BOX_HMPC * 100 * E / (1 + z))
        got = np.asarray(KC.k_skm_from_theta9(kcom, z, _theta_unit(hub, omh2)))
        assert np.allclose(got, expect, rtol=1e-12)


def test_h_cancels():
    kcom = 2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC
    a = KC.k_skm_from_kcom(kcom, 3.0, 0.60, 0.30 * 0.60 ** 2)
    b = KC.k_skm_from_kcom(kcom, 3.0, 0.80, 0.30 * 0.80 ** 2)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_cosmology_response_is_exactly_minus_dlnE():
    kcom = 2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC
    z = 2.6
    for (h1, w1), (h2, w2) in [((0.70, 0.14), (0.70, 0.15)), ((0.66, 0.14), (0.74, 0.14))]:
        r = np.asarray(KC.k_skm_from_kcom(kcom, z, h2, w2)) / np.asarray(KC.k_skm_from_kcom(kcom, z, h1, w1))
        assert np.allclose(r, float(KC.E_of_z(z, h1, w1) / KC.E_of_z(z, h2, w2)), rtol=1e-12)


def test_physical_theta_is_refused():
    with pytest.raises(ValueError, match="unit cube"):
        KC.hub_omegamh2_from_theta9(np.array([0.95, 1.9e-9, 3.7, 2.9, 1.9, 0.70, 0.14, 7.0, 0.05]))


def test_differentiable_in_theta():
    kcom = jnp.asarray(2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC)
    th = jnp.asarray(_theta_unit(0.70, 0.14))
    g = jax.grad(lambda t: jnp.sum(KC.k_skm_from_theta9(kcom, 2.6, t)))(th)
    assert np.all(np.isfinite(np.asarray(g))) and float(g[5]) != 0.0 and float(g[6]) != 0.0


def test_kgrid_object_carries_the_contract():
    kcom = 2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC
    kg = KC.kgrid(kcom, 2.6, _theta_unit(0.70, 0.14))
    assert kg.schema_version == S.CHECKPOINT_SCHEMA_VERSION and kg.z == 2.6
    assert abs(kg.hub - 0.70) < 1e-12 and abs(kg.omegamh2 - 0.14) < 1e-12
    assert np.allclose(np.asarray(kg.k_skm), np.asarray(KC.k_skm_from_kcom(kcom, 2.6, 0.70, 0.14)))


@pytest.mark.skipif(not os.path.exists(LF), reason="real cache absent")
def test_real_rows_reproduced_from_theta_within_tolerance():
    import h5py
    with h5py.File(LF, "r") as h:
        kf = h["kfkms"][...]
        nb = h["nbins_native"][...].astype(float)
        dv = h["dv_kms"][...]
        params = h["params"][...]
        z = h["z_grid"][...]
    d = dict(kfkms=kf, nbins_native=nb, dv_kms=dv)
    kcom = S.k_com_hmpc_from_cache(d)
    rng = np.random.default_rng(1)
    rows = rng.choice(len(z), 60, replace=False)
    for r in rows:
        th = np.asarray(normalize_params(params[r]), float)
        got = np.asarray(KC.k_skm_from_theta9(kcom, float(z[r]), th))
        ok = np.isfinite(kf[r])
        assert np.max(np.abs(got[ok] / kf[r][ok] - 1)) < S.VBOX_RTOL
