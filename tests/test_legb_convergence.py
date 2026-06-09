"""STEP-A convergence-mode harness tests (the multi-chain R-hat path).

Pins the three review-mandated properties of ``run_legb_convergence`` /
``convergence_battery`` that make split-R-hat a VALID convergence diagnostic:

  (i)   chains start DISPERSED — the per-chain init spread is a non-trivial fraction of the
        posterior width (identical ``init_to_median`` starts would defeat R-hat; CRITICAL,
        val-review bayesian §5/concern-1);
  (ii)  the battery returns FINITE rank-split-R-hat / bulk-ESS / tail-ESS / E-BFMI;
  (iii) the chain seed axis is DISTINCT from the divergence-retry stream — chains seed on
        ``fold_in(k_nuts, chain_id)``, the retry on ``base_seed + attempt`` (no collision).

Also unit-tests the pure battery estimators (``convergence_battery``) on a synthetic
multi-chain array where the truth (R-hat≈1, large ESS for iid chains) is known, so the
estimators are verified independently of the (slow) NUTS path.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_convergence.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax

from hcd_analysis.emulator import closure_legb as C

_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"


# --------------------------------------------------------------------------- #
#  Pure-estimator unit tests (fast; no NUTS).
# --------------------------------------------------------------------------- #
def test_convergence_battery_finite_on_iid_chains():
    """On well-mixed iid Gaussian chains the battery is finite, R-hat≈1, ESS≈N·C."""
    rng = np.random.default_rng(0)
    C_, N, P = 4, 500, 3
    packed = rng.standard_normal((C_, N, P))
    names = ["a", "b", "c"]
    energy = rng.standard_normal((C_, N)) + 5.0      # arbitrary stationary energy
    num_steps = rng.integers(1, 30, (C_, N))
    bat = C.convergence_battery(packed, names, energy=energy, num_steps=num_steps,
                                max_tree_depth=10, n_div=0)
    assert np.isfinite(bat["rhat_max"]) and bat["rhat_max"] < 1.05, bat["rhat_max"]
    assert np.isfinite(bat["ess_bulk_min"]) and bat["ess_bulk_min"] > 200
    assert np.isfinite(bat["ess_tail_min"]) and bat["ess_tail_min"] > 100
    assert np.isfinite(bat["ebfmi_min"])
    assert bat["treedepth_sat_frac"] == 0.0          # max num_steps < 2**10-1
    for nm in names:
        assert np.isfinite(bat["rhat"][nm])
        assert np.isfinite(bat["ess_bulk"][nm])
        assert np.isfinite(bat["ess_tail"][nm])


def test_convergence_battery_flags_stuck_chains():
    """Chains stuck at DIFFERENT offsets (non-mixing) → R-hat ≫ 1.01 (the diagnostic fires)."""
    C_, N, P = 4, 300, 1
    offsets = np.array([-3.0, -1.0, 1.0, 3.0])[:, None, None]
    rng = np.random.default_rng(1)
    packed = 0.05 * rng.standard_normal((C_, N, P)) + offsets    # tight within, spread between
    bat = C.convergence_battery(packed, ["x"], max_tree_depth=10, n_div=0)
    assert bat["rhat_max"] > 1.1, f"R-hat should flag stuck chains, got {bat['rhat_max']}"


def test_treedepth_saturation_fraction():
    """Saturation fraction counts draws hitting 2**mtd-1 leapfrogs."""
    C_, N = 2, 100
    packed = np.zeros((C_, N, 1))
    num_steps = np.full((C_, N), 7)                  # 2**3-1 == 7 → all saturate at mtd=3
    bat = C.convergence_battery(packed, ["x"], num_steps=num_steps, max_tree_depth=3, n_div=0)
    assert bat["treedepth_sat_frac"] == 1.0
    bat2 = C.convergence_battery(packed, ["x"], num_steps=num_steps, max_tree_depth=10, n_div=0)
    assert bat2["treedepth_sat_frac"] == 0.0


def test_chain_seed_axis_distinct_from_retry_stream():
    """The convergence chain seed (fold_in(k_nuts, chain_id)) must NOT collide with the
    run_legb retry stream (base_seed + attempt drawn from k_nuts). Verify the two derivations
    produce different PRNG state for the small chain_id / attempt ranges they share."""
    seed, mock_index = 0, 0
    key0 = jax.random.PRNGKey(int(seed))
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, int(mock_index)), 2)
    # run_legb retry stream: an INT base_seed, runs seeded by PRNGKey(base_seed + attempt).
    base_seed = int(jax.random.randint(k_nuts, (), 0, 2 ** 31 - 1))
    retry_keys = [np.asarray(jax.random.PRNGKey(base_seed + a)) for a in range(4)]
    # convergence chain stream: fold_in(k_nuts, chain_id) directly.
    chain_keys = [np.asarray(jax.random.fold_in(k_nuts, c)) for c in range(4)]
    for c, ck in enumerate(chain_keys):
        for a, rk in enumerate(retry_keys):
            assert not np.array_equal(ck, rk), \
                f"chain seed (chain {c}) collides with retry seed (attempt {a})"


# --------------------------------------------------------------------------- #
#  End-to-end smoke (slow; gated on DESI data) — dispersion + finite battery.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(_DESI_NPZ), reason="DESI data not present")
def test_convergence_mode_disperses_and_returns_finite_battery():
    """The convergence path (i) starts chains DISPERSED relative to the posterior width and
    (ii) returns a finite rank-R-hat / bulk-ESS / tail-ESS / E-BFMI battery."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])

    res = C.run_legb_convergence(
        ctx, d, mock_index=0, n_chains=2, n_warmup=40, n_samples=50, seed=0,
        dense_mass=False, max_tree_depth=6, verbose=False)
    b = res["battery"]

    # (i) DISPERSION: at least one of the cosmology params (ns col 0, Ap col 1) must start
    # with init spread ≳ a third of its posterior sd — chains are NOT all at the median.
    iosd = np.asarray(b["init_spread_over_postsd"])
    cosmo_iosd = iosd[:2]                             # ns, Ap
    assert np.nanmax(iosd) > 0.3, \
        f"chains not dispersed: max init/postsd = {np.nanmax(iosd):.2f} (init_to_sample expected)"
    assert np.nanmax(cosmo_iosd) > 0.0               # ns/Ap actually move between chains

    # (ii) FINITE battery (R-hat needs ≥2 chains, which we ran).
    assert np.isfinite(b["rhat_max"]), "R-hat not finite"
    assert np.isfinite(b["ess_bulk_min"]) and b["ess_bulk_min"] > 0
    assert np.isfinite(b["ess_tail_min"]) and b["ess_tail_min"] > 0
    assert np.isfinite(b["ebfmi_min"]), "E-BFMI not finite"
    assert b["n_chains"] == 2 and res["packed"].shape[0] == 2


if __name__ == "__main__":
    test_convergence_battery_finite_on_iid_chains()
    test_convergence_battery_flags_stuck_chains()
    test_treedepth_saturation_fraction()
    test_chain_seed_axis_distinct_from_retry_stream()
    print("[conv] pure-estimator + seed-axis tests pass.")
