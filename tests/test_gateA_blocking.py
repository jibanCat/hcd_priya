"""Gate A blocking tests BT1-BT6 requested by the gate A blind review (notes: docs/superpowers/emulator-debug-2026-10/
reviews/gate_A/gate_A_review.md, section 11). Each failed against c99ce88."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import predict as PR
from hcd_analysis.emulator import schema as S
from hcd_analysis.emulator.data import PARAM_LIMITS, normalize_params
from hcd_analysis.emulator.model import Emulator
from tests.test_schema import _synthetic_cache


# --------------------------------------------------------------------------------------------- BT1
@pytest.mark.parametrize("key,idx", [("dv_kms", None), ("nbins_native", None), ("z_grid", None), ("params", 5), ("params", 6)])
def test_bt1_validator_refuses_non_finite_inputs_on_a_row(key, idx):
    d = _synthetic_cache()
    arr = np.array(d[key], dtype=float)
    if idx is None:
        arr[1] = np.nan
    else:
        arr[1, idx] = np.nan
    d[key] = arr
    with pytest.raises(S.SchemaCollapseError, match="non-finite"):
        S.validate_cache_schema(d)


# --------------------------------------------------------------------------------------------- BT2
def test_bt2_validator_refuses_a_row_without_finite_modes():
    d = _synthetic_cache()
    d["kfkms"][1, :] = np.nan
    with pytest.raises(S.SchemaCollapseError, match="no finite"):
        S.validate_cache_schema(d)


def test_bt2_validator_refuses_mid_band_nan():
    d = _synthetic_cache()
    d["kfkms"][1, 3] = np.nan                       # interior hole, not top-of-band padding
    with pytest.raises(S.SchemaCollapseError, match="contiguous"):
        S.validate_cache_schema(d)


def test_bt2_top_of_band_padding_still_accepted():
    d = _synthetic_cache()
    d["kfkms"][2, -2:] = np.nan
    S.validate_cache_schema(d)


# --------------------------------------------------------------------------------------------- BT3
def test_bt3_validator_refuses_a_cyclic_cache():
    d = _synthetic_cache()
    d["kfkms"] = d["kfkms"] / (2 * np.pi)           # 2 pi missing: k_com = n / L
    with pytest.raises(S.SchemaCollapseError, match="2 pi n / L"):
        S.validate_cache_schema(d)


def test_bt3_validator_refuses_an_off_by_one_cache():
    d = _synthetic_cache()
    vbox = d["nbins_native"] * d["dv_kms"]
    d["kfkms"] = 2 * np.pi * np.arange(0, d["kfkms"].shape[1])[None, :] / vbox[:, None]   # DC mode included
    with pytest.raises(S.SchemaCollapseError):
        S.validate_cache_schema(d)


# --------------------------------------------------------------------------------------------- BT4
def _theta_unit(hub=0.70, omegamh2=0.143):
    phys = np.array(PARAM_LIMITS, float).mean(axis=1)
    phys[5] = hub
    phys[6] = omegamh2
    return np.asarray(normalize_params(phys), float)


@pytest.mark.parametrize("fn", ["hub_omegamh2_from_theta9", "k_skm_from_theta9", "kgrid"])
def test_bt4_non_vector_theta_is_refused(fn):
    kcom = 2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC
    th2 = np.stack([_theta_unit(), _theta_unit(0.72, 0.141)])           # (2, 9) unit-cube, valid values
    args = (th2,) if fn == "hub_omegamh2_from_theta9" else (kcom, 2.6, th2)
    with pytest.raises(ValueError, match="shape"):
        getattr(KC, fn)(*args)


def test_bt4_physical_values_refused_for_any_concrete_shape():
    phys = np.array([0.95, 1.9e-9, 3.7, 2.9, 1.9, 0.70, 0.14, 7.0, 0.05])
    with pytest.raises(ValueError):
        KC.hub_omegamh2_from_theta9(np.stack([phys, phys]))
    with pytest.raises(ValueError, match="unit cube"):
        KC.hub_omegamh2_from_theta9(phys)


def test_bt4_traced_theta_still_differentiable():
    kcom = jnp.asarray(2 * np.pi * np.arange(1, 4) / S.L_BOX_HMPC)
    g = jax.grad(lambda t: jnp.sum(KC.k_skm_from_theta9(kcom, 2.6, t)))(jnp.asarray(_theta_unit()))
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------------------------- BT5
def test_bt5_k_com_length_must_match_the_model_output():
    K = 12
    model = Emulator(in_dim=10, n_k=K, n_basis=None, key=jax.random.PRNGKey(0))
    pf = {k: np.ones((4, K)) * v for k, v in (("mu_marg", 0.0), ("sig_marg", 1.0), ("sig_cosmo", 1.0))}
    meta = {"schema_version": "2.0", "k_com_hmpc": (2 * np.pi * np.arange(1, K - 1) / S.L_BOX_HMPC).tolist(),
            "param_limits": np.asarray(PARAM_LIMITS).tolist()}
    with pytest.raises(S.SchemaCollapseError, match="length"):
        PR.predict_on_physical_grid(model, meta, _theta_unit(), 2.6, 0.3, np.zeros(3), pf, np.zeros(K))


# --------------------------------------------------------------------------------------------- BT6
def _train_and_save(tmp_path, **kw):
    from tests.emulator._fixture import write_synthetic_cache
    from hcd_analysis.emulator.data import fit_target_norm, load_cache
    from hcd_analysis.emulator.train import save_checkpoint
    p = tmp_path / "obs.h5"
    write_synthetic_cache(p, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12)
    d = load_cache(p)
    norm = fit_target_norm(d, np.arange(d["P_filt"].shape[0]))
    m = Emulator(in_dim=10, n_k=12, n_basis=4, key=jax.random.PRNGKey(0))
    prefix = str(tmp_path / "ck")
    save_checkpoint(prefix, m, {"in_dim": 10, "n_k": 12, "n_basis": 4}, norm, seed=0, cache=d, **kw)
    return prefix, d, m, norm, p


def test_bt6_cache_path_is_required_and_hashed(tmp_path):
    with pytest.raises(S.SchemaCollapseError, match="cache_path"):
        _train_and_save(tmp_path)                                      # no cache_path
    with pytest.raises(S.SchemaCollapseError, match="cache_path"):
        _train_and_save(tmp_path, cache_path=str(tmp_path / "does_not_exist.h5"))
    prefix, *_ = _train_and_save(tmp_path, cache_path=str(tmp_path / "obs.h5"))
    meta = json.load(open(prefix + ".meta.json"))
    assert isinstance(meta["cache_sha256"], str) and len(meta["cache_sha256"]) == 64
    assert np.allclose(meta["param_limits"], np.asarray(PARAM_LIMITS))


@pytest.mark.parametrize("missing", ["k_com_hmpc", "L_box_hmpc", "k_convention", "cache_sha256", "param_limits"])
def test_bt6_reader_requires_every_schema2_field(tmp_path, missing):
    from hcd_analysis.emulator.train import load_checkpoint
    prefix, *_ = _train_and_save(tmp_path, cache_path=str(tmp_path / "obs.h5"))
    meta = json.load(open(prefix + ".meta.json"))
    del meta[missing]
    json.dump(meta, open(prefix + ".meta.json", "w"))
    with pytest.raises(S.SchemaCollapseError, match=missing):
        load_checkpoint(prefix)


def test_bt6_reader_refuses_a_velocity_grid_at_any_depth(tmp_path):
    from hcd_analysis.emulator.train import load_checkpoint
    prefix, d, *_ = _train_and_save(tmp_path, cache_path=str(tmp_path / "obs.h5"))
    meta = json.load(open(prefix + ".meta.json"))
    meta["recipe"] = {"nested": {"kfkms": [float(v) for v in d["kfkms"][0]]}}
    json.dump(meta, open(prefix + ".meta.json", "w"))
    with pytest.raises(S.SchemaCollapseError, match="kfkms"):
        load_checkpoint(prefix)


def test_bt6_writer_refuses_a_velocity_grid_in_the_recipe(tmp_path):
    with pytest.raises(S.SchemaCollapseError, match="kfkms"):
        _train_and_save(tmp_path, cache_path=str(tmp_path / "obs.h5"), recipe={"kfkms": [1.0, 2.0]})
