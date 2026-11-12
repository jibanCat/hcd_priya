"""products.load_mf_product (the gate-D-style .npz MF product, historical path): every digest (LF ensemble, LF cache,
HR cache) and the composition are checked, and nothing is unpickled. Synthetic tables and made-up digests only.
"""

import hashlib
import json

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator import mf_modes as MM
from hcd_analysis.emulator import products as PR
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 8
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC
H64 = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
CK = [H64(f"member{i}") for i in range(3)]
LF, HR = H64("lf"), H64("hr")


def _tables(nz=3, nr=3):
    rng = np.random.default_rng(1)
    return dict(
        log_rho=rng.normal(size=K),
        gbar_z_tab=rng.normal(size=(nz, 4, K)),
        gtau_tab=rng.normal(size=(nr, 4, K)),
        a_k=rng.normal(size=(4, K)),
        u_z=rng.normal(size=nz),
        u_tau=rng.normal(size=nr),
        z_tab=np.linspace(2.6, 3.4, nz),
        tau_tab=np.arange(nr, dtype=float),
        tau_by_z=np.tile(np.linspace(0.2, 0.6, nr), (nz, 1)),
    )


def _product(tmp_path, name="p.npz", **prov):
    p = str(tmp_path / name)
    base = dict(lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, res_corr="off (NORC)")
    MM.save_mode_mf(p, _tables(), k_com_hmpc=KCOM, provenance=dict(base, **prov))
    return p


def test_hr_cache_digest_is_checked(tmp_path):
    p = _product(tmp_path)
    PR.load_mf_product(
        p, lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, composition="all"
    )
    with pytest.raises(PR.ProductRefused, match="HR cache"):
        PR.load_mf_product(
            p,
            lf_ckpt_sha256=CK,
            lf_cache_sha256=LF,
            hr_cache_sha256=H64("other"),
            composition="all",
        )


def test_hr_digest_and_composition_are_required():
    with pytest.raises(TypeError):
        PR.load_mf_product("x.npz", lf_ckpt_sha256=CK, lf_cache_sha256=LF)  # noqa
    with pytest.raises(TypeError):
        PR.load_mf_product(
            "x.npz", lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR
        )  # noqa


def test_composition_is_checked(tmp_path):
    implicit = _product(
        tmp_path, "implicit.npz"
    )  # a product written before compositions: all pairs
    PR.load_mf_product(
        implicit, lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, composition="all"
    )
    with pytest.raises(PR.ProductRefused, match="composition"):
        PR.load_mf_product(
            implicit,
            lf_ckpt_sha256=CK,
            lf_cache_sha256=LF,
            hr_cache_sha256=HR,
            composition={"id": "xS", "excluded_sims": ["simS"]},
        )
    comp = {"id": "xS", "excluded_sims": ["simS"], "ruling_ids": ["synthetic ruling"]}
    named = _product(tmp_path, "named.npz", composition=comp)
    PR.load_mf_product(
        named,
        lf_ckpt_sha256=CK,
        lf_cache_sha256=LF,
        hr_cache_sha256=HR,
        composition={"id": "xS", "excluded_sims": ["simS"]},
    )
    for wrong in (
        "all",
        {"id": "xS", "excluded_sims": ["simT"]},
        {"id": "xT", "excluded_sims": ["simS"]},
    ):
        with pytest.raises(PR.ProductRefused, match="composition"):
            PR.load_mf_product(
                named, lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, composition=wrong
            )


def test_an_object_array_is_never_unpickled(tmp_path):
    p = str(tmp_path / "obj.npz")
    t = _tables()
    np.savez(
        p,
        schema_version="2.0",
        k_com_hmpc=KCOM,
        provenance=json.dumps({}),
        **dict(t, log_rho=np.array([object()] * K, dtype=object)),
    )
    with pytest.raises(PR.ProductRefused):
        PR.load_mf_product(
            p, lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, composition="all"
        )


def test_a_product_that_records_no_composition_is_not_a_named_one(tmp_path):
    """R4b: a product written before compositions were recorded was fitted on every matched pair; asked for a named
    composition (an id string), it is refused, not read as "all"."""
    implicit = _product(tmp_path, "implicit.npz")
    for wrong in ("synthetic-x", "synthetic-named"):
        with pytest.raises(PR.ProductRefused, match="composition"):
            PR.load_mf_product(
                implicit,
                lf_ckpt_sha256=CK,
                lf_cache_sha256=LF,
                hr_cache_sha256=HR,
                composition=wrong,
            )
    named = _product(tmp_path, "named_str.npz", composition="synthetic-named")
    PR.load_mf_product(
        named,
        lf_ckpt_sha256=CK,
        lf_cache_sha256=LF,
        hr_cache_sha256=HR,
        composition="synthetic-named",
    )
    with pytest.raises(PR.ProductRefused, match="composition"):
        PR.load_mf_product(
            named, lf_ckpt_sha256=CK, lf_cache_sha256=LF, hr_cache_sha256=HR, composition="all"
        )
