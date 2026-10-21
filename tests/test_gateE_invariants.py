"""Gate E invariants (GATE_E_SPEC v1 section 1, PU-0061/0062): the forward rewiring onto the canonical coordinate must
leave the data side byte-identical. Digests pinned at worktree 80bbc5d (before any gate E change): the three leg
loaders under their production arguments (``closure_legb.build_legb_ctx`` + ``PROD_FORWARD_BY_LEG``), and the metal
and resolution factors, which act on data k only."""
import hashlib
import os

import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

DATA = {"DESI": "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz",
        "eBOSS": "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz",
        "KS": "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"}

# Production leg arguments: build_legb_ctx(sample_res=True, coherent_res=False) with PROD_FORWARD_BY_LEG
# (DESI and eBOSS metals on, KS resolution_float=True, k_max=0.065).
PROD_LEG_ARGS = {
    "DESI": lambda: DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
    "KS": lambda: DL.load_ks_leg(resolution_float=True, k_max=0.065),
    "eBOSS": lambda: DL.load_eboss_leg(resolution_float=True, resolution_coherent=False),
}

PINNED = {   # sha256 of leg_digest() at 80bbc5d
    "DESI": "392c218e8a14ebad6c7964d753d2a51e55979d6af96224d9da2407d9542db407",
    "KS": "4fae133fb02719b881cc89e3018d7b170c6423e46401aca916d984594d20d05d",
    "eBOSS": "514415bf46162394db38a1dc3703017076186049175cb5257f8e2a5ab25ba62f",
    "factors": "1b0a1ccffc40dfdad583b9dcbcd3462444c9cb7a0d0bb6d05e0c5914b345205d",
}


def _feed(h, name, v):
    h.update(name.encode())
    if isinstance(v, (np.ndarray,)) or hasattr(v, "__array__") and not isinstance(v, (str, bytes)):
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    else:
        h.update(repr(v).encode())


def leg_digest(leg):
    h = hashlib.sha256()
    for name in leg._fields:
        _feed(h, name, getattr(leg, name))
    return h.hexdigest()


def factors_digest():
    k = np.geomspace(5e-4, 0.1, 257)
    h = hashlib.sha256()
    for kw in (dict(a_SiIII=0.01), dict(a_SiII=0.004), dict(a_SiIII=0.012, a_SiII=-0.003)):
        _feed(h, f"metal{sorted(kw.items())}", np.asarray(DL._metal_factor(k, **kw)))
    for R, b in ((10.0, 0.1), (3.2, -0.05), (25.0, 0.02)):
        _feed(h, f"res{R},{b}", np.asarray(DL._resolution_factor(k, R, b_res=b)))
    return h.hexdigest()


def _unavailable(msg):
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1)")
    pytest.skip(msg)


@pytest.mark.parametrize("leg", ["DESI", "KS", "eBOSS"])
def test_production_leg_loaders_unchanged(leg):
    if not os.path.exists(DATA[leg]):
        _unavailable(f"data product absent: {DATA[leg]}")
    assert leg_digest(PROD_LEG_ARGS[leg]()) == PINNED[leg]


def test_metal_and_resolution_factors_unchanged():
    assert factors_digest() == PINNED["factors"]
