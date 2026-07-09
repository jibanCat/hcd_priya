"""Task 2A -- unit + artifact tests for the z-incoherent OUT-OF-SPAN
instrument-resolution injection basis (scripts/build_res_instr_injection_basis.py).

TWO tiers:
  (1) PURE / LOAD-BEARING: the ``build_oos_bres`` helper on tiny synthetic inputs.
      Asserts the M_sub-orthogonality of the output to span{u1,u2} to 1e-8, the
      out-of-span certificate cos_M<0.8, the r_ns M-pullback identity to 1e-8, and
      the shapes -- NO heavy ctx. These pin the core math and run in milliseconds.
  (2) ARTIFACT: load the written npz (skip if absent) and assert per-leg shapes,
      the cos_span<0.8 certificate, and worst_ns_member in (1,2).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
     -m pytest tests/test_res_instr_basis.py -q
"""
from __future__ import annotations
import os
import numpy as np
import pytest

# module top-level is numpy-only, so this import is cheap (no jax/ctx pulled in)
from scripts.build_res_instr_injection_basis import (
    build_oos_bres,
    _whitened_inner,
    OUT_NPZ,
    COS_GATE,
    Z_HEII,
    F_RES_PIVOT_Z,
)


# ---------------------------------------------------------------------------- #
#  synthetic-input factory: a tiny leg with a genuine He-II sub-block
# ---------------------------------------------------------------------------- #
def _synthetic(seed=0, n_z=7, n_k=4):
    rng = np.random.default_rng(seed)
    z = np.linspace(2.2, 4.2, n_z)                    # 5 nodes are z>=2.8
    R_z = 40.0 + 5.0 * rng.random(n_z)                # order-realistic R_z (s/km)
    # z-major flat grid: n_k cells per node
    z_idx = np.repeat(np.arange(n_z), n_k)
    N = z_idx.size
    k = np.tile(np.linspace(0.01, 0.06, n_k), n_z)    # angular k per cell
    # B[i, iz] = 2 k_i^2 R_z(iz)^2 on its own z-block (0 elsewhere)
    B = np.zeros((N, n_z))
    for iz in range(n_z):
        rsel = np.where(z_idx == iz)[0]
        B[rsel, iz] = 2.0 * k[rsel] ** 2 * R_z[iz] ** 2
    # SPD C_data: A A^T + diag (well-conditioned)
    A = rng.standard_normal((N, N)) * 1e-3
    C_data = A @ A.T + np.diag(1e-4 + 1e-3 * rng.random(N))
    C_data = 0.5 * (C_data + C_data.T)
    u1 = np.ones(n_z)
    u2 = np.log((1.0 + z) / (1.0 + F_RES_PIVOT_Z))
    zrow = z[z_idx]
    hiZ_cells = zrow >= Z_HEII
    hiZ_nodes = z >= Z_HEII
    r_ns_P = rng.standard_normal(N)                   # arbitrary P-space n_s response
    return dict(B=B, C_data=C_data, u1=u1, u2=u2, hiZ_cells=hiZ_cells,
                hiZ_nodes=hiZ_nodes, r_ns_P=r_ns_P, z=z, n_z=n_z)


@pytest.fixture(scope="module")
def synth():
    return _synthetic()


@pytest.fixture(scope="module")
def res(synth):
    return build_oos_bres(synth["B"], synth["C_data"], synth["u1"], synth["u2"],
                          synth["hiZ_cells"], synth["hiZ_nodes"], synth["r_ns_P"])


# ---------------------------------------------------------------------------- #
#  (1) PURE / LOAD-BEARING tests
# ---------------------------------------------------------------------------- #
def test_shapes(res, synth):
    n_z = synth["n_z"]
    n_heII = int(synth["hiZ_nodes"].sum())
    assert res["bres1"].shape == (n_z,)
    assert res["bres2"].shape == (n_z,)
    assert res["bres1_sub"].shape == (n_heII,)
    assert res["bres2_sub"].shape == (n_heII,)
    assert res["r_ns_bres"].shape == (n_heII,)
    assert res["M_sub"].shape == (n_heII, n_heII)
    # embedded vectors are zero OUTSIDE the He-II nodes (localization)
    off = ~synth["hiZ_nodes"]
    assert np.allclose(res["bres1"][off], 0.0)
    assert np.allclose(res["bres2"][off], 0.0)


def test_M_sub_is_spd_and_matches_definition(res, synth):
    M = res["M_sub"]
    # SPD
    assert np.all(np.linalg.eigvalsh(M) > 0)
    # M_sub == B_sub^T C_sub^-1 B_sub (the pullback metric definition)
    M_def = res["B_sub"].T @ res["Cinv_sub"] @ res["B_sub"]
    assert np.allclose(M, 0.5 * (M_def + M_def.T), atol=1e-10)


def test_M_orthogonality_to_span_1e8(res):
    """THE load-bearing property: each bres_j_sub is M_sub-orthogonal to BOTH
    u1_sub and u2_sub to 1e-8 (in the M-cosine sense, which is scale-clean)."""
    M = res["M_sub"]
    u1_sub, u2_sub = res["u1_sub"], res["u2_sub"]
    _, _, cos_M = _whitened_inner(M)
    for b in (res["bres1_sub"], res["bres2_sub"]):
        assert abs(cos_M(b, u1_sub)) < 1e-8
        assert abs(cos_M(b, u2_sub)) < 1e-8


def test_cos_span_certificate(res):
    """The OUT-OF-SPAN certificate: cos_M(bres_j, span{u1,u2}) < 0.8 (in fact ~0)."""
    assert res["cos_span"][0] < COS_GATE
    assert res["cos_span"][1] < COS_GATE
    assert res["cos_span"][0] < 1e-8
    assert res["cos_span"][1] < 1e-8


def test_bres_are_M_unit(res):
    """The recovered directions are M-unit (||bres_j||_M == 1)."""
    _, norm_M, _ = _whitened_inner(res["M_sub"])
    assert abs(norm_M(res["bres1_sub"]) - 1.0) < 1e-8
    assert abs(norm_M(res["bres2_sub"]) - 1.0) < 1e-8


def test_pullback_identity_1e8(res, synth):
    """The r_ns M-pullback identity (why a naive B^T r_ns is WRONG):
        <a, r_ns_bres>_M == <B_sub a, r_ns[hiZ]>_{C_sub^-1}   for all a.
    """
    M = res["M_sub"]
    B_sub, Cinv_sub = res["B_sub"], res["Cinv_sub"]
    r_ns_bres, r_ns_sub = res["r_ns_bres"], res["r_ns_P_sub"]
    rng = np.random.default_rng(1)
    n_heII = M.shape[0]
    for _ in range(5):
        a = rng.standard_normal(n_heII)
        lhs = float(a @ M @ r_ns_bres)
        rhs = float((B_sub @ a) @ Cinv_sub @ r_ns_sub)
        assert abs(lhs - rhs) <= 1e-8 * (1.0 + abs(rhs))


def test_pullback_differs_from_naive(res):
    """Sanity that the identity is non-trivial: the correct M-pullback r_ns_bres is
    NOT proportional to the naive B_sub^T r_ns[hiZ] (would give a misleading cos)."""
    naive = res["B_sub"].T @ res["r_ns_P_sub"]
    correct = res["r_ns_bres"]
    # cosine (Euclidean) between the two must be well below 1 (they are different maps)
    c = abs(naive @ correct) / (np.linalg.norm(naive) * np.linalg.norm(correct) + 1e-30)
    assert c < 0.999


def test_worst_ns_member_in_1_2(res):
    assert res["worst_ns_member"] in (1, 2)


def test_rank_two_well_determined(res):
    """Both output directions are well-determined (2nd singular value non-negligible)."""
    S = res["S"]
    assert S.size >= 2
    assert S[1] > 1e-9 * S[0]


def test_raises_when_too_few_heII_nodes():
    s = _synthetic(n_z=7)
    # force only 2 He-II nodes -> cannot form a rank-2 basis beyond the 2-D span
    hiZ_nodes = np.zeros(7, bool); hiZ_nodes[-2:] = True
    z = s["z"]
    hiZ_cells = np.isin(np.repeat(np.arange(7), 4), np.where(hiZ_nodes)[0])
    with pytest.raises(ValueError):
        build_oos_bres(s["B"], s["C_data"], s["u1"], s["u2"], hiZ_cells, hiZ_nodes, s["r_ns_P"])


# ---------------------------------------------------------------------------- #
#  (2) ARTIFACT tests (skip if the heavy build has not run yet)
# ---------------------------------------------------------------------------- #
_HAVE_NPZ = os.path.exists(OUT_NPZ)
skip_if_no_npz = pytest.mark.skipif(not _HAVE_NPZ, reason=f"basis npz not built yet: {OUT_NPZ}")


@pytest.fixture(scope="module")
def basis():
    return np.load(OUT_NPZ, allow_pickle=True)


@skip_if_no_npz
def test_artifact_legs_present(basis):
    legs = [str(x) for x in np.atleast_1d(basis["_meta_legs"])]
    assert len(legs) >= 1
    for name in legs:
        n_z = basis[f"{name}_z"].shape[0]
        for m in ("bres1", "bres2", "bres_real"):
            assert basis[f"{name}_{m}"].shape == (n_z,), f"{name}_{m}"


@skip_if_no_npz
def test_artifact_cos_span_certificate(basis):
    legs = [str(x) for x in np.atleast_1d(basis["_meta_legs"])]
    for name in legs:
        cs = np.asarray(basis[f"{name}_cos_span"], float)
        assert cs.shape == (2,)
        assert cs[0] < COS_GATE and cs[1] < COS_GATE, f"{name} cos_span={cs}"


@skip_if_no_npz
def test_artifact_worst_ns_member(basis):
    legs = [str(x) for x in np.atleast_1d(basis["_meta_legs"])]
    for name in legs:
        assert int(basis[f"{name}_worst_ns_member"]) in (1, 2)


@skip_if_no_npz
def test_artifact_env_used_recorded(basis):
    legs = [str(x) for x in np.atleast_1d(basis["_meta_legs"])]
    for name in legs:
        assert str(basis[f"{name}_env_used"]) in ("resolution_e", "fallback_5pct")
