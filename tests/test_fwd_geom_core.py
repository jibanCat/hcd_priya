"""NO-PEEK tests for the PI #18 forward-geometry analytic core.

Every test is synthetic or a pure identity against the DEPLOYED code. None reads a
certification posterior, so this file runs safely before the scan is unblinded.

The load-bearing one is test_alpha_identity_against_deployed_code: the entire scan rests on
the jnp.interp abscissa being exactly alpha_z, which makes the surfaces straight lines.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import fwd_geom_core as G  # noqa: E402


# --------------------------------------------------------------------------- #
# the identity the whole design rests on
# --------------------------------------------------------------------------- #
def test_alpha_identity_against_deployed_code():
    """alpha (likelihood.py:115) == alpha_z (closure_legb.py:3140), exactly."""
    from hcd_analysis.emulator.likelihood import _KIM_AMP, _KIM_SLOPE
    from hcd_analysis.emulator.meanflux_prior import TAU0_PIVOT_Z
    assert (_KIM_AMP, _KIM_SLOPE) == (G.KIM_AMP, G.KIM_SLOPE)
    assert TAU0_PIVOT_Z == G.TAU0_PIVOT_Z
    rng = np.random.default_rng(0)
    for _ in range(20):
        ta = rng.uniform(*G.TAU0_AMP_RANGE)
        dt = rng.uniform(*G.DTAU0_RANGE)
        z = G.Z_GRID["DESI"]
        alpha_z = G.alpha_of(np.array([ta]), np.array([dt]), z)[0]
        tau0 = alpha_z * G.kim(z)                                  # closure_legb.py:3141
        alpha = tau0 / (_KIM_AMP * (1.0 + z) ** _KIM_SLOPE)        # likelihood.py:115
        assert np.max(np.abs(alpha - alpha_z)) < 1e-15


def test_point_on_surface_reproduces_the_knot():
    """A point placed on K(i,k) must give alpha_z == a_k exactly."""
    for leg in ("DESI", "eBOSS"):
        c, ln_a, z, k = G.surfaces(leg)
        for j in range(0, len(c), 7):
            y = -0.2
            x = ln_a[j] - c[j] * y                                 # solve x + c y = ln a
            az = G.alpha_of(np.array([np.exp(x)]), np.array([y]), np.array([z[j]]))[0, 0]
            assert az == pytest.approx(G.ALPHA_CENTRES[k[j]], abs=1e-12)


def test_surface_counts_match_the_frozen_z_grids():
    assert len(G.Z_GRID["DESI"]) == 11 and len(G.Z_GRID["eBOSS"]) == 13
    assert len(G.surfaces("DESI")[0]) == 44
    assert len(G.surfaces("eBOSS")[0]) == 52
    # eBOSS -- the arm that PASSED -- has MORE surfaces. Control fact, frozen pre-inspection.
    assert len(G.surfaces("eBOSS")[0]) > len(G.surfaces("DESI")[0])


def test_knots_are_the_frozen_artifact_values():
    import numpy as _np
    ev = _np.load("/home/mfho/hcd_priya/checkpoints/error_vector.npz", allow_pickle=True)
    assert _np.allclose(_np.asarray(ev["tau0_band_centres"]), G.ALPHA_CENTRES)
    evx = _np.load("/home/mfho/hcd_priya/checkpoints/error_vector_xclass.npz", allow_pickle=True)
    assert _np.allclose(_np.asarray(evx["tau0_band_centres"]), G.ALPHA_CENTRES)


def test_c_vanishes_at_the_pivot():
    """At z = z_pivot the surfaces are vertical lines in ln tau0_amp alone."""
    assert G.c_of_z(G.TAU0_PIVOT_Z) == pytest.approx(0.0, abs=1e-15)
    z = G.Z_GRID["DESI"]
    c = G.c_of_z(z)
    assert c[np.argmin(np.abs(z - 3.0))] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(c) > 0)                                  # monotone in z


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def test_signed_distance_zero_on_the_surface_and_sign_convention():
    c = np.array([0.1]),
    cc, aa = np.array([0.1]), np.array([-0.18])
    y = 0.05
    x_on = aa[0] - cc[0] * y
    d = G.signed_distance([[x_on, y]], cc, aa)
    assert abs(d[0, 0]) < 1e-15
    # positive side == larger alpha
    d_hi = G.signed_distance([[x_on + 0.01, y]], cc, aa)[0, 0]
    d_lo = G.signed_distance([[x_on - 0.01, y]], cc, aa)[0, 0]
    assert d_hi > 0 > d_lo


def test_raw_distance_is_the_euclidean_point_line_distance():
    cc, aa = np.array([0.25]), np.array([0.1])
    pt = np.array([[0.4, -0.3]])
    got = abs(G.signed_distance(pt, cc, aa)[0, 0])
    want = abs(pt[0, 0] + cc[0] * pt[0, 1] - aa[0]) / np.sqrt(1 + cc[0] ** 2)
    assert got == pytest.approx(want)


def test_whitened_distance_is_invariant_to_the_whitening_basis():
    """Whitened distance must equal the Euclidean distance computed in whitened space."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(2, 2))
    C = A @ A.T + 0.1 * np.eye(2)
    L = np.linalg.cholesky(C)
    cc, aa = np.array([0.3]), np.array([-0.05])
    pts = rng.normal(size=(50, 2)) * 0.1
    d_w = G.signed_distance(pts, cc, aa, L=L)[:, 0]
    # direct: map points and the line into whitened coords and measure there
    n = np.array([1.0, cc[0]])
    for i, p in enumerate(pts):
        # whitened coords w = L^{-1} p ; the plane n.p = a becomes (L.T n).w = a
        w = np.linalg.solve(L, p)
        nw = L.T @ n
        want = (nw @ w - aa[0]) / np.linalg.norm(nw)
        assert d_w[i] == pytest.approx(want, rel=1e-12, abs=1e-14)


def test_whitener_reproduces_the_sample_covariance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 2)) @ np.array([[0.3, 0.1], [0.0, 0.2]])
    mu, L = G.plane_whitener(X)
    assert np.allclose(L @ L.T, np.cov(X.T), rtol=1e-6, atol=1e-9)
    Z = np.linalg.solve(L, (X - mu).T).T
    assert np.allclose(np.cov(Z.T), np.eye(2), atol=0.15)


# --------------------------------------------------------------------------- #
# crossing
# --------------------------------------------------------------------------- #
def test_cross_prob_limits_and_monotonicity():
    assert G.cross_prob(0.0, 0.4) == pytest.approx(1.0)
    assert G.cross_prob(1e6, 0.4) == pytest.approx(0.0, abs=1e-12)
    d = np.array([0.0, 0.1, 0.5, 1.0, 3.0])
    p = G.cross_prob(d, 0.4)
    assert np.all(np.diff(p) < 0)
    # a distance of exactly one eps must give P = erfc(1/sqrt2) ~ 0.3173
    assert G.cross_prob(0.4, 0.4) == pytest.approx(0.31731, abs=1e-4)


def test_cross_prob_is_symmetric_in_sign():
    assert G.cross_prob(-0.7, 0.3) == pytest.approx(G.cross_prob(0.7, 0.3))


def test_crossings_between_detects_sign_changes():
    d = np.array([[-1.0], [-0.5], [0.5], [0.2], [-0.3]])
    got = G.crossings_between(d)[:, 0]
    assert list(got) == [False, True, False, True]


# --------------------------------------------------------------------------- #
# step-size proxy
# --------------------------------------------------------------------------- #
def test_eps_proxy_is_D_to_the_minus_quarter():
    assert G.eps_w("DESI") == pytest.approx(27 ** -0.25, rel=1e-12)
    assert G.eps_w("eBOSS") == pytest.approx(23 ** -0.25, rel=1e-12)
    # eBOSS has FEWER sampled dims so a LARGER proxy step -- recorded, since it RAISES
    # eBOSS's crossing probability and therefore cuts against the DESI hypothesis.
    assert G.eps_w("eBOSS") > G.eps_w("DESI")


def test_frozen_grids_are_what_the_prereg_declares():
    assert G.DELTA_GRID == (1e-4, 1e-3, 1e-2) and G.DELTA_PRIMARY == 1e-3
    assert G.EPS_SCALE_GRID == (0.5, 1.0, 2.0)
    assert G.TAU0_AMP_RANGE == (0.75, 1.25) and G.DTAU0_RANGE == (-0.4, 0.25)


def _reachable_counts(leg):
    """Surfaces whose line intersects the prior box, per knot."""
    lo_x, hi_x = np.log(G.TAU0_AMP_RANGE[0]), np.log(G.TAU0_AMP_RANGE[1])
    ylo, yhi = G.DTAU0_RANGE
    c, ln_a, z, k = G.surfaces(leg)
    hit = np.zeros(4, int)
    for j in range(len(c)):
        xs = (ln_a[j] - c[j] * ylo, ln_a[j] - c[j] * yhi)
        if max(min(xs), lo_x) <= min(max(xs), hi_x):
            hit[k[j]] += 1
    return hit


def test_prior_reachability_table_matches_prereg_v2():
    """Knot a0 is UNREACHABLE under the prior on both legs; a1 and a2 are fully reachable;
    a3 only partly. Pinning this caught two errors in the prereg's first table."""
    assert list(_reachable_counts("DESI")) == [0, 11, 11, 3]
    assert list(_reachable_counts("eBOSS")) == [0, 13, 13, 5]
    # eBOSS -- the arm that PASSED -- has MORE reachable surfaces, on identical priors.
    assert _reachable_counts("eBOSS").sum() == 31 > _reachable_counts("DESI").sum() == 25


def test_shift_span_sign_convention():
    """The -c_i*y span, which the prereg's first table had sign-reversed."""
    for leg, want_lo, want_hi in (("DESI", -0.0893, 0.1049), ("eBOSS", -0.0893, 0.1346)):
        c = G.c_of_z(G.Z_GRID[leg])
        prod = -np.outer(c, np.array(G.DTAU0_RANGE))
        assert prod.min() == pytest.approx(want_lo, abs=1e-3)
        assert prod.max() == pytest.approx(want_hi, abs=1e-3)
