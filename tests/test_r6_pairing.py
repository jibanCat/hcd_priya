"""R6 pairing property (2026-07-23, HEAVY: two production KS ctx builds, no NUTS).

The load-bearing claim behind the R6 paired protocol: the self-draw mock data is a pure
function of (truth_pack, noise key) — the fit ctx's PRIOR enters the mock nowhere — so
drawing the truth from ONE source (the mapped ctx) and forwarding it through BOTH the mapped
and the legacy-override ctx yields bit-identical data vectors. If this ever breaks, R6's
"identical mocks" premise is void and analyze_r6_pairs' truth-identity asserts would pass
while the DATA silently differed (truth identity is necessary, data identity is the theorem
proved here).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_r6_pairing.py -q
"""
import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401 (x64 before jax)
import jax
import jax.numpy as jnp

import hcd_analysis.emulator.closure_legb as CL
from scripts.run_dnuis_bias_shard import build_arm_ctx


@pytest.fixture(scope="module")
def ctx_pair():
    ctx_m, d, _ = build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True)
    ctx_l, _, _ = build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True,
                                ks_legacy_alpha_param=True)
    return ctx_m, ctx_l, d


def _fid_core(ctx, d):
    return {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
            for name, v in CL._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}


def test_override_flips_only_the_parameterization(ctx_pair):
    ctx_m, ctx_l, _ = ctx_pair
    assert ctx_m.ks_dndx_mapped is True and ctx_l.ks_dndx_mapped is False
    # forward-side state identical (the override moves the PRIOR dispatch only)
    assert ctx_m.res_corr_on == ctx_l.res_corr_on
    assert ctx_m.fix_alpha_res == ctx_l.fix_alpha_res
    assert [l.name for l in ctx_m.legs] == [l.name for l in ctx_l.legs]
    np.testing.assert_array_equal(np.asarray(ctx_m.legs[0].k), np.asarray(ctx_l.legs[0].k))
    # the legacy vectors are the same objects/values on both ctxs (active vs dormant)
    np.testing.assert_allclose(np.asarray(ctx_m.alpha_hcd_mu),
                               np.asarray(ctx_l.alpha_hcd_mu), rtol=0, atol=0)


def test_mapped_truth_in_simplex_and_site_set(ctx_pair):
    ctx_m, ctx_l, _ = ctx_pair
    key = jax.random.PRNGKey(20260723)
    t = CL.draw_leg_a_leg_truth(ctx_m, key)
    a_z = np.asarray(t["alpha_hcd_z"], float)
    assert np.all(a_z >= 0.0) and np.all(a_z.sum(axis=1) < 1.0), \
        "mapped truth left the occupancy simplex"
    assert {"eps_lls", "kappa_lls", "m_sub"} <= set(t["raw"]), "mapped raw sites missing"
    # the two priors sample DIFFERENT site sets (why per-arm self-draw would break pairing)
    t_l = CL.draw_leg_a_leg_truth(ctx_l, key)
    assert "eps_lls" not in t_l["raw"] and "s_lls" in t_l["raw"]


def test_shared_truth_gives_bit_identical_mocks(ctx_pair):
    ctx_m, ctx_l, d = ctx_pair
    key = jax.random.PRNGKey(42)
    k_truth, k_mock, _ = jax.random.split(jax.random.fold_in(key, 0), 3)
    t = CL.draw_leg_a_leg_truth(ctx_m, k_truth)          # ONE truth source: the mapped prior
    core = _fid_core(ctx_m, d)
    mock_m, info_m = CL.make_leg_a_legmock(ctx_m, core, t, k_mock)
    mock_l, info_l = CL.make_leg_a_legmock(ctx_l, core, t, k_mock)
    assert [l.name for l in mock_m] == [l.name for l in mock_l]
    for lm, ll in zip(mock_m, mock_l):
        np.testing.assert_array_equal(
            np.asarray(lm.P_data), np.asarray(ll.P_data),
            err_msg=f"{lm.name}: mock data differs across the R6 ctx pair — the mock is NOT "
                    f"prior-independent; the R6 pairing premise is void")
        np.testing.assert_array_equal(np.asarray(info_m["truth_on_leg"][lm.name]),
                                      np.asarray(info_l["truth_on_leg"][ll.name]))
