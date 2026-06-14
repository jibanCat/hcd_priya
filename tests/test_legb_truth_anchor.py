"""Phase-1: the closure truth τ₀ is anchored on a PRIYA curve (not Becker), so the mock's
τ₀(z) is representable by the 2-param model the forward samples — closure self-consistency."""
import numpy as np
import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp
import pytest
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.meanflux_prior import tau0_alpha_priya, kim_tau0

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"


@pytest.mark.parametrize("amp_t,dt_t", [(1.0, 0.0), (1.1, 0.15), (0.85, -0.25)])
def test_priya_anchored_truth_recovers_requested_curve(amp_t, dt_t):
    d = load_cache(CACHE)
    sims, _ = CL.held_out_sims(d, fold=0)
    t = CL.make_truth_from_sim(d, sims[0], fold=0, tau0_anchor=(amp_t, dt_t))
    assert "tau0_amp" in t and "dtau0" in t
    # the fitted truth params recover the requested PRIYA curve to the ladder-discretization floor
    assert abs(t["tau0_amp"] - amp_t) < 0.06, (t["tau0_amp"], amp_t)
    assert abs(t["dtau0"] - dt_t) < 0.12, (t["dtau0"], dt_t)
    # and the selected per-z τ₀(z) is fit by the 2-param form to the ladder-discretization floor
    # (a steep dτ₀ makes the per-z nearest-rung selection jag; worst at the lowest-z prior corner,
    # ~6-7% — inherent to the discrete cache ladder; the interior/real-fit regime is <3%).
    z = jnp.asarray(t["z"])
    rec = np.asarray(tau0_alpha_priya(z, t["tau0_amp"], t["dtau0"]) * kim_tau0(z))
    assert np.max(np.abs(rec / np.asarray(t["tau0"]) - 1)) < 0.08


def test_default_anchor_is_priya_central():
    d = load_cache(CACHE)
    sims, _ = CL.held_out_sims(d, fold=0)
    t = CL.make_truth_from_sim(d, sims[0], fold=0)   # default tau0_anchor="priya"
    # central PRIYA curve ≈ Kim (τ₀≈1, dτ₀≈0)
    assert abs(t["tau0_amp"] - 1.0) < 0.10
    assert abs(t["dtau0"] - 0.0) < 0.15
