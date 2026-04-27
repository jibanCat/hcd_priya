"""
Validation tests for ``hcd_analysis/clustering.py``.

This file currently covers tests **1–3** + **2b** + **3b** of the doc
(`docs/clustering_definitions.md` §8).  Tests 4–10 (which depend on
the pair counter) come in a follow-up commit after user approval.

Run::

    python3 tests/test_clustering.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.clustering import (
    SightlineGeometry,
    los_separation,
    minimum_image,
    pixel_to_xyz,
    xyz_to_nearest_pixel,
)


def _toy_geometry(box: float = 120.0, n_pix: int = 1250, n_per_axis: int = 8) -> SightlineGeometry:
    """Build a deterministic geometry for unit tests.

    n_per_axis × n_per_axis lateral grid replicated along each of the
    three axes, total 3 · n_per_axis² sightlines.  Uniform spacing.
    """
    rng_grid = np.linspace(0.0, box, n_per_axis, endpoint=False)
    coords = []
    axes = []
    for axis in range(3):
        for i, u in enumerate(rng_grid):
            for j, v in enumerate(rng_grid):
                pos = [0.0, 0.0, 0.0]
                # The two non-LOS axes carry the (u, v) lateral grid;
                # the LOS axis is anchored at 0.
                lateral_axes = [k for k in range(3) if k != axis]
                pos[lateral_axes[0]] = u
                pos[lateral_axes[1]] = v
                coords.append(pos)
                axes.append(axis)
    cofm = np.array(coords, dtype=np.float64)
    axis_arr = np.array(axes, dtype=np.int8)
    return SightlineGeometry(
        box=box,
        n_pix=n_pix,
        dx_pix=box / n_pix,
        n_sightlines=cofm.shape[0],
        axis=axis_arr,
        cofm_mpch=cofm,
        z_snap=3.0,
        hubble=0.7,
    )


class TestCoordinateRoundTrip(unittest.TestCase):
    """Test 1 in the doc: (skewer, pixel) → xyz → nearest pixel recovers
    the input pixel exactly on the integer grid."""

    def test_roundtrip_all_axes(self):
        geom = _toy_geometry()
        rng = np.random.default_rng(0)
        n = 5000
        sk = rng.integers(0, geom.n_sightlines, size=n)
        pix = rng.integers(0, geom.n_pix, size=n)
        xyz = pixel_to_xyz(geom, sk, pix)
        pix_back = xyz_to_nearest_pixel(geom, sk, xyz)
        np.testing.assert_array_equal(pix_back, pix)

    def test_roundtrip_pixel_zero_and_last(self):
        """Edge cases: the first and last pixel on each sightline."""
        geom = _toy_geometry()
        sk = np.array([0, 1, 2, 3, geom.n_sightlines - 1], dtype=np.int64)
        pix0 = np.zeros_like(sk)
        pix_last = np.full_like(sk, geom.n_pix - 1)
        for pix in (pix0, pix_last):
            xyz = pixel_to_xyz(geom, sk, pix)
            self.assertEqual(xyz.shape, (sk.size, 3))
            pix_back = xyz_to_nearest_pixel(geom, sk, xyz)
            np.testing.assert_array_equal(pix_back, pix)

    def test_los_lies_between_anchor_and_anchor_plus_box(self):
        """Pixel position along the LOS axis is in [cofm[axis], cofm[axis]+box)."""
        geom = _toy_geometry()
        rng = np.random.default_rng(1)
        sk = rng.integers(0, geom.n_sightlines, size=2000)
        pix = rng.integers(0, geom.n_pix, size=2000)
        xyz = pixel_to_xyz(geom, sk, pix)
        ax = geom.axis[sk]
        cof_los = geom.cofm_mpch[sk, ax]
        los_pos = xyz[np.arange(sk.size), ax]
        # los_pos modulo box equals cof_los + (pix+0.5)*dx_pix
        expected = (cof_los + (pix + 0.5) * geom.dx_pix) % geom.box
        np.testing.assert_allclose(los_pos, expected, rtol=1e-12, atol=1e-12)

    def test_lateral_axes_unchanged_by_pixel(self):
        """The two non-LOS coordinates do not depend on the pixel."""
        geom = _toy_geometry()
        rng = np.random.default_rng(2)
        sk = rng.integers(0, geom.n_sightlines, size=200)
        pix1 = rng.integers(0, geom.n_pix, size=200)
        pix2 = rng.integers(0, geom.n_pix, size=200)
        xyz1 = pixel_to_xyz(geom, sk, pix1)
        xyz2 = pixel_to_xyz(geom, sk, pix2)
        ax = geom.axis[sk]
        for k in range(3):
            mask = ax != k
            np.testing.assert_array_equal(xyz1[mask, k], xyz2[mask, k])


class TestPeriodicMinimumImage(unittest.TestCase):
    """Test 2 in the doc: random Δx in [-box, box] returns ≤ box/2 after wrap."""

    def test_minimum_image_in_half_box(self):
        rng = np.random.default_rng(3)
        box = 120.0
        delta = rng.uniform(-2 * box, 2 * box, size=20000)
        wrapped = minimum_image(delta, box)
        self.assertLessEqual(np.abs(wrapped).max(), box / 2 + 1e-12)

    def test_minimum_image_zero(self):
        wrapped = minimum_image(np.array([0.0]), box=120.0)
        np.testing.assert_array_equal(wrapped, [0.0])

    def test_minimum_image_box_boundaries(self):
        """Δx = ±box/2 should land at -box/2 (numpy banker's rounding)."""
        box = 120.0
        wrapped = minimum_image(np.array([box / 2, -box / 2, box, -box]), box)
        # ±box/2 → -box/2 (banker's rounding behaviour); ±box → 0
        np.testing.assert_allclose(np.abs(wrapped), [box / 2, box / 2, 0, 0])

    def test_idempotent(self):
        """Wrapping a wrapped Δ leaves it unchanged."""
        rng = np.random.default_rng(4)
        box = 120.0
        delta = rng.uniform(-2 * box, 2 * box, size=2000)
        once = minimum_image(delta, box)
        twice = minimum_image(once, box)
        np.testing.assert_allclose(twice, once)


class TestParPerpDecomposition(unittest.TestCase):
    """Test 3 in the doc: r_par² + r_perp² = |Δ|² for random pairs."""

    def test_pythagoras(self):
        rng = np.random.default_rng(5)
        n = 10000
        delta = rng.normal(size=(n, 3))
        axis = rng.integers(0, 3, size=n)
        r_par_signed, r_perp = los_separation(delta, axis)
        total_sq = (delta ** 2).sum(axis=1)
        np.testing.assert_allclose(
            r_par_signed ** 2 + r_perp ** 2, total_sq, rtol=1e-12, atol=1e-12
        )

    def test_r_perp_nonnegative(self):
        rng = np.random.default_rng(6)
        delta = rng.normal(size=(2000, 3))
        axis = rng.integers(0, 3, size=2000)
        _, r_perp = los_separation(delta, axis)
        self.assertGreaterEqual(r_perp.min(), 0.0)

    def test_r_par_sign_follows_delta(self):
        """Sign of signed r_par matches sign of delta along the LOS axis."""
        rng = np.random.default_rng(7)
        delta = rng.normal(size=(500, 3))
        axis = rng.integers(0, 3, size=500)
        r_par_signed, _ = los_separation(delta, axis)
        expected_sign = np.sign(delta[np.arange(500), axis])
        # zero values: both should be zero (sign convention 0 → 0).
        nz = expected_sign != 0
        np.testing.assert_array_equal(np.sign(r_par_signed[nz]), expected_sign[nz])

    def test_pure_los_pair_has_zero_perp(self):
        """A pair separated only along its LOS axis has r_perp = 0."""
        delta = np.array([[5.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.5]])
        axis = np.array([0, 1, 2], dtype=np.int64)
        r_par_signed, r_perp = los_separation(delta, axis)
        np.testing.assert_allclose(r_par_signed, [5.0, -3.0, 2.5])
        np.testing.assert_allclose(r_perp, [0.0, 0.0, 0.0])

    def test_pure_perp_pair_has_zero_par(self):
        """A pair separated only across the LOS axis has r_par = 0 and
        r_perp = |Δ|."""
        delta = np.array([
            [0.0, 4.0, 3.0],   # axis 0 → r_par=0, r_perp=5
            [4.0, 0.0, 3.0],   # axis 1
            [4.0, 3.0, 0.0],   # axis 2
        ])
        axis = np.array([0, 1, 2], dtype=np.int64)
        r_par_signed, r_perp = los_separation(delta, axis)
        np.testing.assert_allclose(r_par_signed, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(r_perp, [5.0, 5.0, 5.0])


class TestSightlineGeometryGuards(unittest.TestCase):
    """Misuse guards on the dataclass."""

    def test_axis_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            SightlineGeometry(
                box=120.0, n_pix=10, dx_pix=12.0, n_sightlines=1,
                axis=np.array([3], dtype=np.int8),
                cofm_mpch=np.zeros((1, 3)),
                z_snap=3.0, hubble=0.7,
            )

    def test_cofm_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            SightlineGeometry(
                box=120.0, n_pix=10, dx_pix=12.0, n_sightlines=2,
                axis=np.array([0, 1], dtype=np.int8),
                cofm_mpch=np.zeros((1, 3)),  # wrong: should be (2,3)
                z_snap=3.0, hubble=0.7,
            )


class TestRealSpectraLoad(unittest.TestCase):
    """Loader sanity against an actual production spectra file.

    Skipped if the path is not present (CI / non-Great-Lakes machines).
    """

    SPEC = Path(
        "/nfs/turbo/umor-yueyingn/mfho/emu_full/"
        "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735"
        "omegamh20.141hireionz7.17bhfeedback0.056/output/SPECTRA_017/"
        "lya_forest_spectra_grid_480.hdf5"
    )

    @unittest.skipUnless(SPEC.exists(), "production spectra file unavailable")
    def test_load_real(self):
        from hcd_analysis.clustering import load_sightline_geometry

        geom = load_sightline_geometry(self.SPEC)
        # Schema sanity
        self.assertEqual(geom.n_sightlines, 691200)
        self.assertEqual(geom.n_pix, 1250)
        self.assertAlmostEqual(geom.box, 120.0, places=6)        # 120000 kpc/h → 120 Mpc/h
        self.assertAlmostEqual(geom.hubble, 0.735, places=3)
        self.assertAlmostEqual(geom.z_snap, 3.0, places=2)
        # axis must be 0-indexed
        self.assertEqual(int(geom.axis.min()), 0)
        self.assertEqual(int(geom.axis.max()), 2)
        # cofm in Mpc/h, must lie in [0, box]
        self.assertGreaterEqual(geom.cofm_mpch.min(), 0.0)
        self.assertLessEqual(geom.cofm_mpch.max(), geom.box + 1e-6)
        # dx_pix sanity
        self.assertAlmostEqual(geom.dx_pix, 0.096, places=3)


if __name__ == "__main__":
    unittest.main()
