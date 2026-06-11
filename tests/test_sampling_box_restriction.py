"""The NUTS sampling prior restricts the IGM params to the ORIGINAL PRIYA box.

Phase-4b found the catastrophic closure biases (alphaq=2.98 -> A_p +3.7sigma, herei=4.46 ->
+2.0sigma) sit in the WIDENED-box corners where the held-out emulator extrapolates. The fix
restricts the NUTS theta prior (data.SAMPLING_LIMITS / closure_legb._THETA_UNIT_*) to the
original PRIYA ranges on herei/heref/alphaq while KEEPING n_s extended. This pins:
  (1) the unit-cube bounds are right (IGM sub-intervals, n_s/Ap/etc full);
  (2) the model actually SAMPLES inside the original PRIYA IGM box (prior-predictive).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_sampling_box_restriction.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax

from hcd_analysis.emulator import data as D
from hcd_analysis.emulator import closure_legb as C

_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"

# original PRIYA (coarse_grid.py defaults) for the IGM params we restrict
_ORIG = {"herei": (3.5, 4.1), "heref": (2.6, 3.2), "alphaq": (1.3, 2.5)}
_IDX = {n: i for i, n in enumerate(("ns", "Ap", "herei", "heref", "alphaq",
                                    "hub", "omegamh2", "hireionz", "bhfeedback"))}


def test_sampling_limits_restrict_igm_keep_ns_extended():
    """SAMPLING_LIMITS == original PRIYA on herei/heref/alphaq; n_s/Ap stay at the widened box."""
    lim = D.SAMPLING_LIMITS
    for name, (lo, hi) in _ORIG.items():
        np.testing.assert_allclose(lim[_IDX[name]], [lo, hi], err_msg=f"{name} not original PRIYA")
    # n_s kept extended (matches the widened training box, NOT the original 0.995 cap)
    np.testing.assert_allclose(lim[_IDX["ns"]], D.PARAM_LIMITS[_IDX["ns"]])
    np.testing.assert_allclose(lim[_IDX["Ap"]], D.PARAM_LIMITS[_IDX["Ap"]])


def test_unit_bounds_values():
    """The unit-cube bounds: IGM params get sub-intervals, kept-extended params get [0,1]."""
    lo, hi = D.sampling_unit_bounds()
    assert lo.shape == (9,) and hi.shape == (9,)
    # n_s/Ap/hub/omegamh2/hireionz/bhfeedback span the full unit cube
    for name in ("ns", "Ap", "hub", "omegamh2", "hireionz", "bhfeedback"):
        assert lo[_IDX[name]] == pytest.approx(0.0), name
        assert hi[_IDX[name]] == pytest.approx(1.0), name
    # herei: [3.5,4.1] in [3.5,4.5] -> [0, 0.6]
    assert lo[_IDX["herei"]] == pytest.approx(0.0)
    assert hi[_IDX["herei"]] == pytest.approx(0.6)
    # heref: [2.6,3.2] in [2.2,3.2] -> [0.4, 1.0]
    assert lo[_IDX["heref"]] == pytest.approx(0.4)
    assert hi[_IDX["heref"]] == pytest.approx(1.0)
    # alphaq: [1.3,2.5] in [1.3,3.0] -> [0, 1.2/1.7]
    assert lo[_IDX["alphaq"]] == pytest.approx(0.0)
    assert hi[_IDX["alphaq"]] == pytest.approx(1.2 / 1.7)
    # closure_legb's frozen defaults match data's
    np.testing.assert_allclose(C._THETA_UNIT_LO, lo)
    np.testing.assert_allclose(C._THETA_UNIT_HI, hi)


@pytest.mark.skipif(not os.path.exists(_DESI_NPZ), reason="DESI data not present")
def test_prior_predictive_inside_original_priya_box():
    """Prior-predictive draws from the production model land inside the original PRIYA IGM box
    (and never in the widened corners), confirming the restricted prior is the one NUTS uses."""
    import numpyro
    from numpyro.infer import Predictive

    ctx, _d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    pred = Predictive(C._legb_priors_only, num_samples=2000)
    draws = pred(jax.random.PRNGKey(0), ctx)
    tu = np.asarray(draws["theta_unit"])              # (2000, 9), unit cube
    phys = D.PARAM_LIMITS[:, 0] + tu * (D.PARAM_LIMITS[:, 1] - D.PARAM_LIMITS[:, 0])  # -> physical
    for name, (lo, hi) in _ORIG.items():
        col = phys[:, _IDX[name]]
        assert col.min() >= lo - 1e-9 and col.max() <= hi + 1e-9, (
            f"{name} prior draws [{col.min():.3f},{col.max():.3f}] escape original PRIYA [{lo},{hi}]")
    # n_s still reaches the extended ridge (> original 0.995) — KEEP extended, do not clip
    ns = phys[:, _IDX["ns"]]
    assert ns.max() > 0.995, "n_s prior should still reach the extended ridge (>0.995)"


if __name__ == "__main__":
    test_sampling_limits_restrict_igm_keep_ns_extended()
    test_unit_bounds_values()
    if os.path.exists(_DESI_NPZ):
        test_prior_predictive_inside_original_priya_box()
    print("[box] IGM params restricted to original PRIYA; n_s kept extended; prior-predictive OK.")
