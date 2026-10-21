"""Gate E task 3 (GATE_E_SPEC v1 section 2, criterion E3b): product identity for the forward. Every product the gate E
forward loads carries a kind, the box-mode axis and provenance with digests; the loaders refuse everything else,
including every pre-2026-10 product and the gate C error vector marked NOT FOR USE (which the old loader accepts)."""
import hashlib
import json
import os

import numpy as np
import pytest

from hcd_analysis.emulator import products as PR
from hcd_analysis.emulator import error_vector_io as EV
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC
CACHE = "d9c3783892872f8739f6c1a4039ee84cae5ab84803235a61d7d06ac119336cf2"
PROV = dict(code_commit="abc1234", cache_sha256=CACHE, inputs={"eval_a": "11" * 32}, row_rule="validation rows")

GATEC_NOT_FOR_USE = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC/error_vector.npz"
from hcd_analysis.paths import REPO_ROOT_STR  # noqa: E402
HIST = ["/home/mfho/hcd_priya/checkpoints/error_vector.npz",  # historical-artifact path
        "/home/mfho/hcd_priya/checkpoints/error_vector_xclass.npz",  # historical-artifact path
        f"{REPO_ROOT_STR}/figures/analysis/04_emulator/mf_cemu_floor.npz",
        "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"]  # historical-artifact path
MF_PRODUCT = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/mf/gateD/mf_modes_all6.npz"
GATEC = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC"


def _unavailable(msg):
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1)")
    pytest.skip(msg)


@pytest.mark.parametrize("kind", PR.PRODUCT_KINDS)
def test_round_trip(tmp_path, kind):
    p = tmp_path / f"{kind}.npz"
    PR.save_product(p, kind, k_com_hmpc=KCOM, provenance=PROV, table=np.ones((3, K)))
    arrays, k_com, prov = PR.load_product(p, kind, cache_sha256=CACHE, inputs={"eval_a": "11" * 32})
    assert np.array_equal(arrays["table"], np.ones((3, K))) and np.array_equal(k_com, KCOM) and prov == PROV


def test_refusals_on_write(tmp_path):
    p = tmp_path / "x.npz"
    with pytest.raises(PR.ProductRefused, match="kind"):
        PR.save_product(p, "error_vector", k_com_hmpc=KCOM, provenance=PROV, t=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="velocity"):
        PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM, provenance=PROV, kfkms=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="velocity"):
        PR.save_product(p, "cemu_t3", k_com_hmpc=KCOM, provenance=PROV, k=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="2 pi n / L"):
        PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM * 1.001, provenance=PROV, t=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="provenance"):
        PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM, provenance={"code_commit": "x"}, t=np.ones(K))
    PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM, provenance=PROV, t=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="exists"):
        PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM, provenance=PROV, t=np.ones(K))


def test_refusals_on_read(tmp_path):
    p = tmp_path / "x.npz"
    PR.save_product(p, "cemu_t1", k_com_hmpc=KCOM, provenance=PROV, t=np.ones(K))
    with pytest.raises(PR.ProductRefused, match="kind"):
        PR.load_product(p, "cemu_t3", cache_sha256=CACHE)
    with pytest.raises(PR.ProductRefused, match="cache"):
        PR.load_product(p, "cemu_t1", cache_sha256="00" * 32)
    with pytest.raises(PR.ProductRefused, match="inputs"):
        PR.load_product(p, "cemu_t1", cache_sha256=CACHE, inputs={"eval_a": "22" * 32})
    old = tmp_path / "old.npz"
    np.savez(old, sigma=np.ones((4, K, 3, 4)), k_com_hmpc=KCOM, schema_version="2.0")   # gate C sweep style
    with pytest.raises(PR.ProductRefused, match="unlabelled"):
        PR.load_product(old, "cemu_t1", cache_sha256=CACHE)


@pytest.mark.parametrize("path", [GATEC_NOT_FOR_USE] + HIST)
def test_every_pre_gate_e_product_is_refused(path):
    if not os.path.exists(path):
        _unavailable(f"product absent: {path}")
    for kind in PR.PRODUCT_KINDS:
        with pytest.raises(PR.ProductRefused):
            PR.load_product(path, kind, cache_sha256=CACHE)


def test_error_vector_loader_requires_provenance(tmp_path):
    a = dict(sigma=np.ones((4, K, 3, 4)), k_com_hmpc=KCOM, z_band_edges=np.array([-np.inf, 3.2, 4.4, np.inf]),
             tau0_band_centres=np.ones(4), class_names=np.array(["a", "b", "c", "d"]))
    EV.save_error_vector(tmp_path / "no_prov.npz", **a)
    with pytest.raises(EV.SchemaCollapseError, match="provenance"):
        EV.load_error_vector(tmp_path / "no_prov.npz")
    EV.save_error_vector(tmp_path / "prov.npz", **a, provenance=np.array(json.dumps({"seed": 0})))
    assert "provenance" in EV.load_error_vector(tmp_path / "prov.npz")
    if os.path.exists(GATEC_NOT_FOR_USE):
        with pytest.raises(EV.SchemaCollapseError, match="provenance"):
            EV.load_error_vector(GATEC_NOT_FOR_USE)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_mf_product_identity_is_checked_against_the_lf_ensemble():
    if not os.path.exists(MF_PRODUCT):
        _unavailable(f"gate D product absent: {MF_PRODUCT}")
    digests = [_sha(f"{GATEC}/prod_repaired_seed{i}.eqx") for i in range(5)]
    mf, k_com, prov = PR.load_mf_product(MF_PRODUCT, lf_ckpt_sha256=digests, lf_cache_sha256=CACHE)
    np.testing.assert_allclose(k_com, KCOM, rtol=1e-14, atol=0)   # the cache's validated modes
    with pytest.raises(PR.ProductRefused, match="LF ensemble"):
        PR.load_mf_product(MF_PRODUCT, lf_ckpt_sha256=digests[::-1], lf_cache_sha256=CACHE)
    with pytest.raises(PR.ProductRefused, match="cache"):
        PR.load_mf_product(MF_PRODUCT, lf_ckpt_sha256=digests, lf_cache_sha256="00" * 32)
    if os.path.exists(HIST[2]):
        with pytest.raises(PR.ProductRefused):
            PR.load_mf_product(HIST[2], lf_ckpt_sha256=digests, lf_cache_sha256=CACHE)
