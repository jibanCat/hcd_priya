"""Persistent golden guard: ``fast_postprocess`` == the legacy full-model replay.

``_run_nuts_legb(fast_postprocess=True)`` (the production default) SKIPS numpyro's default
per-sample full-model deterministic replay (237.6 s/mock waste, MF-SMOKE-01) and instead
reconstructs the three ``numpyro.deterministic`` sites — ``tau0_vec``, ``alpha_dla``,
``alpha_hcd_z`` — HOST-SIDE (``_legb_reconstruct_deterministics``). The docstring claims the
returned samples dict is byte-identical to the in-model computation. This test PINS that:
it runs ONE small mock through BOTH paths on the SAME seed and asserts the three deterministic
sites are byte-identical (``assert_array_equal``, i.e. rtol=0 atol=0).

WHY this is load-bearing: a future change to ``_legb_model`` (the τ₀ α-ladder ↔ Kim(z) map,
the softplus α_DLA link, or the HCD z-slope ``alpha_pivot · g(z)``) that is NOT mirrored in
``_legb_reconstruct_deterministics`` would silently corrupt τ₀/α for the ENTIRE STEP-A /
coverage run — surfacing as a coverage/recovery ANOMALY, never a crash. This guard turns that
into a hard test failure at the seam.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_postprocess_golden.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax

from hcd_analysis.emulator import closure_legb as C

_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"

# The 3 numpyro.deterministic sites of _legb_model that the host-side reconstruction must match.
_DET_SITES = ("tau0_vec", "alpha_dla", "alpha_hcd_z")


@pytest.mark.skipif(not os.path.exists(_DESI_NPZ), reason="DESI data not present")
def test_fast_postprocess_matches_replay_golden():
    """fast_postprocess reconstructs tau0_vec/alpha_dla/alpha_hcd_z byte-identically to the
    legacy in-model deterministic replay on a fixed seed (rtol=0, atol=0)."""
    # Small DESI-only narrow-z context so the FULL real-grid path runs in ~1 min/chain.
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])

    sims, _ = C.held_out_sims(d, fold=0)
    truth_sim = C.make_truth_from_sim(d, sims[0], fold=0)
    k_mock, _ = jax.random.split(jax.random.PRNGKey(0), 2)
    mock_legs, _truth_pack, _info = C.make_legb_mock(ctx, truth_sim, k_mock)
    core = C._mock_core_per_leg(ctx, truth_sim)

    # Tiny warmup/samples; the seed must be IDENTICAL across the two runs (same chain, same
    # latent draws) so the ONLY difference is how the deterministics are recovered.
    kw = dict(n_warmup=12, n_samples=16, seed=7, max_tree_depth=6, dense_mass=False)
    s_fast, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=True, **kw)
    s_slow, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=False, **kw)

    # The raw latent sites must already match (same kernel, same seed) — sanity.
    for site in ("theta_unit", "alpha_ladder", "alpha_lls", "alpha_subdla", "alpha_dla_raw"):
        np.testing.assert_array_equal(
            np.asarray(s_fast[site]), np.asarray(s_slow[site]),
            err_msg=f"latent site {site!r} differs across postprocess paths (seed mismatch?)")

    # The headline guard: the reconstructed deterministics are BYTE-IDENTICAL.
    for site in _DET_SITES:
        a = np.asarray(s_fast[site]); b = np.asarray(s_slow[site])
        assert a.shape == b.shape, f"{site!r} shape {a.shape} != replay {b.shape}"
        np.testing.assert_array_equal(
            a, b, err_msg=f"deterministic site {site!r}: host-side reconstruction (fast_postprocess) "
                          f"!= numpyro in-model replay — the τ₀/α reconstruction has drifted from the "
                          f"model. Re-sync _legb_reconstruct_deterministics with _legb_model.")


if __name__ == "__main__":
    test_fast_postprocess_matches_replay_golden()
    print("[golden] fast_postprocess == legacy replay on tau0_vec/alpha_dla/alpha_hcd_z (rtol=0).")
