"""Task 2B: _resolve_res_instr_inject per-z resolution-injection resolver + the
_check_resolution_injectable guard fired on an `active` flag (blocker B1 fix).

Pure, fast unit tests (no ctx build). The end-to-end (real leg) tests live in
tests/test_dnuis_inject.py alongside the existing scalar/none golden tests.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_resinj_instr.py -q
"""
import numpy as np
import pytest
from types import SimpleNamespace

from hcd_analysis.emulator import closure_legb as C


def _leg(n_z=4, name="DESI"):
    return SimpleNamespace(name=name, n_z=n_z, resolution_ready=True)


def test_resolve_none_is_noop():
    assert C._resolve_res_instr_inject(None, _leg()) == (None, None)


def test_resolve_scalar_unchanged():
    assert C._resolve_res_instr_inject({"b_res": 0.02}, _leg()) == (0.02, None)


def test_resolve_vector_shape_ok_and_bad():
    s, v = C._resolve_res_instr_inject({"b_res_vec": [0.0, 0.1, -0.1, 0.05]}, _leg(4))
    assert s is None and np.allclose(v, [0.0, 0.1, -0.1, 0.05])
    with pytest.raises(ValueError):
        C._resolve_res_instr_inject({"b_res_vec": [0.0, 0.1]}, _leg(4))   # wrong length


def test_resolve_basis_form_loads_and_scales(tmp_path):
    npz = tmp_path / "b.npz"
    np.savez(npz, DESI_bres1=np.array([0.0, 0.1, 0.2, 0.3]))
    s, v = C._resolve_res_instr_inject({"path": str(npz), "member": "bres1", "strength": 2.0}, _leg(4))
    assert s is None and np.allclose(v, [0.0, 0.2, 0.4, 0.6])
    with pytest.raises(KeyError):
        C._resolve_res_instr_inject({"path": str(npz), "member": "bogus"}, _leg(4))


def test_guard_fires_on_active_unready():
    ready = SimpleNamespace(name="DESI", resolution_ready=True)
    unready = SimpleNamespace(name="KS", resolution_ready=False)
    C._check_resolution_injectable([ready], active=True)             # ok
    C._check_resolution_injectable([ready, unready], active=False)   # no-op
    with pytest.raises(ValueError):
        C._check_resolution_injectable([ready, unready], active=True)  # stray KS -> RAISE
