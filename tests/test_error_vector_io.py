"""Gate A, task 6: error-vector products are labelled by the comoving modes and carry a schema version.
Incident: hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md"""
import numpy as np
import pytest

from hcd_analysis.emulator import error_vector_io as EV
from hcd_analysis.emulator import schema as S


def _arrays():
    return dict(sigma=np.ones((4, 4, 3, 4)), k_com_hmpc=2 * np.pi * np.arange(1, 5) / 120.0,
                z_band_edges=np.array([-np.inf, 3.2, 4.4, np.inf]),
                tau0_band_centres=np.array([0.66, 0.83, 1.15, 1.33]),
                class_names=np.array(["clean", "LLS", "subDLA", "DLA"]))


def test_roundtrip_carries_k_com_and_schema(tmp_path):
    p = tmp_path / "ev.npz"
    # provenance is required since gate E (QUEUES E4; refusal tested in test_products_io_v2.py)
    EV.save_error_vector(p, **_arrays(), dla_shot_flag=np.zeros(4, bool), provenance=np.array('{"seed": 0}'))
    ev = EV.load_error_vector(p)
    assert ev["schema_version"] == S.CHECKPOINT_SCHEMA_VERSION
    assert np.allclose(ev["k_com_hmpc"], _arrays()["k_com_hmpc"]) and "kfkms" not in ev
    assert ev["dla_shot_flag"].shape == (4,)


def test_old_kfkms_labelled_file_is_refused(tmp_path):
    np.savez(tmp_path / "old.npz", sigma=np.ones((4, 4, 3, 4)), kfkms=_arrays()["k_com_hmpc"])
    with pytest.raises(S.SchemaCollapseError, match="kfkms"):
        EV.load_error_vector(tmp_path / "old.npz")


def test_writing_a_velocity_label_is_refused(tmp_path):
    with pytest.raises(S.SchemaCollapseError):
        EV.save_error_vector(tmp_path / "x.npz", **_arrays(), kfkms=_arrays()["k_com_hmpc"])
    with pytest.raises(S.SchemaCollapseError):
        EV.save_error_vector(tmp_path / "x.npz", **_arrays(), k_skm=_arrays()["k_com_hmpc"])


def test_sigma_and_rho_mode_axis_must_match_k_com(tmp_path):
    a = _arrays()
    a["sigma"] = np.ones((4, 5, 3, 4))                     # 5 modes vs 4 in k_com
    with pytest.raises(S.SchemaCollapseError, match="modes"):
        EV.save_error_vector(tmp_path / "x.npz", **a)
    with pytest.raises(S.SchemaCollapseError, match="modes"):
        EV.save_error_vector(tmp_path / "y.npz", **_arrays(), rho=np.ones((4, 4, 5, 3, 4)))
