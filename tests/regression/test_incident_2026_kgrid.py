"""Isolated reproduction of the 2026 k-grid representation regression (NOT used by production code).
History: hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md"""
import os

import numpy as np
import pytest

from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import schema as S

LF = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"


def _old_save_checkpoint_grid(kfkms):
    """The pre-2026-10 collapse, copied verbatim from train.py:679-683 at tag pre-emulator-debug-kgrid-2026-10-05."""
    kf = np.asarray(kfkms)
    kgrid = kf[0] if kf.ndim == 2 else np.atleast_1d(kf).ravel()   # the false 'shared grid'
    return kgrid


def test_old_collapse_mislabels_a_second_row_by_the_designed_factor():
    n = np.arange(1, 9)
    vbox = np.array([15565.6, 11411.0])                      # z = 5.4 and z = 2.2 of the same simulation
    kfkms = 2 * np.pi * n[None, :] / vbox[:, None]
    label = _old_save_checkpoint_grid(kfkms)
    assert np.allclose(label, kfkms[0])                      # row 0 wins ...
    assert np.allclose(kfkms[1] / label, vbox[0] / vbox[1])  # ... and row 1 is mislabelled by 1.364


def test_new_schema_refuses_the_collapsed_cache():
    nb = np.array([1556, 1141])
    vbox = np.array([15565.6, 11411.0])
    kfkms = 2 * np.pi * np.arange(1, 9)[None, :] / vbox[:, None]
    d = dict(kfkms=np.repeat(kfkms[0:1], 2, axis=0), nbins_native=nb, dv_kms=vbox / nb,
             params=np.tile([0.95, 1.9e-9, 3.7, 2.9, 1.9, 0.735, 0.1405, 7.0, 0.05], (2, 1)),
             z_grid=np.array([5.4, 2.2]))
    with pytest.raises(S.SchemaCollapseError):
        S.validate_cache_schema(d, n_rows=2)


@pytest.mark.skipif(not os.path.exists(LF), reason="real cache absent")
def test_anti_row_zero_on_the_real_cache():
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(LF)
    kcom = d["k_com_hmpc"]
    row0 = d["kfkms"][0]
    z = d["z_grid"]
    r22 = np.where(np.isclose(z, 2.2))[0]
    worst = max(float(np.nanmax(np.abs(d["kfkms"][r] / row0 - 1))) for r in r22[::50])
    assert worst > 0.20, "row 0 would be a representative grid only if this were small; it is 0.21 to 0.38 at z = 2.2"
    for r in r22[::50]:
        canon = np.asarray(KC.k_skm_from_theta9(kcom, float(z[r]), d["params_unit"][r]))
        ok = np.isfinite(d["kfkms"][r])
        assert np.max(np.abs(canon[ok] / d["kfkms"][r][ok] - 1)) < S.VBOX_RTOL
