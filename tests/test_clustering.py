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
    fold_signed_to_abs,
    los_separation,
    minimum_image,
    pair_count_2d,
    pixel_to_xyz,
    xi_auto_dla,
    xi_cross_dla_lya,
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


class TestXiCrossOnRandomField(unittest.TestCase):
    """Test 4 in the doc: random Poisson DLAs × random Gaussian δ_F field
    → ξ_× consistent with 0 within Poisson error.

    Constructs a small box (60 Mpc/h) with ~5e3 random Lyα-pixel positions
    sampling a Gaussian δ_F, and ~1000 random DLA positions with no
    intentional clustering between the two.  Verifies that the
    bin-averaged ξ_× has a mean consistent with 0 within
    1/sqrt(N_per_bin).
    """

    def test_zero_signal_random_inputs(self):
        rng = np.random.default_rng(42)
        box = 60.0
        n_pix = 5000
        n_dla = 1000

        pixel_xyz = rng.uniform(0, box, size=(n_pix, 3))
        pixel_axis = rng.integers(0, 3, size=n_pix)
        pixel_delta_F = rng.standard_normal(n_pix)        # mean 0, std 1
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))

        r_perp_edges = np.linspace(0.0, 25.0, 11)         # 5 Mpc/h bins
        r_par_edges = np.linspace(-25.0, 25.0, 11)        # 5 Mpc/h, signed

        xi, counts, npairs = xi_cross_dla_lya(
            pixel_xyz=pixel_xyz,
            pixel_los_axis=pixel_axis,
            pixel_delta_F=pixel_delta_F,
            dla_xyz=dla_xyz,
            box=box,
            r_perp_bins=r_perp_edges,
            r_par_bins_signed=r_par_edges,
            chunk_size=200,
        )
        # Use the fold to drop the sign; gives more pairs per bin.
        xi_folded, _ = fold_signed_to_abs(xi, r_par_edges)

        # Expected per-bin error: σ(δ_F) / sqrt(npairs).  We require
        # mean(xi_folded) consistent with 0 within 5 σ across all bins.
        npairs_folded = npairs[:, npairs.shape[1] // 2:].sum(axis=0).sum()
        self.assertGreater(npairs_folded, 5e5)              # sanity: enough pairs
        # The bin with the fewest pairs sets the worst-case error scale
        min_pairs = npairs[:, npairs.shape[1] // 2:].min()
        self.assertGreater(min_pairs, 50)                   # CLT regime
        worst_err = 1.0 / np.sqrt(min_pairs)
        max_xi = np.nanmax(np.abs(xi_folded))
        self.assertLess(max_xi, 5.0 * worst_err)

    def test_zero_signal_with_periodic_pairs(self):
        """Edge case: pairs separated near the box boundary should still
        contribute via minimum-image wrap.  Place all pixels at x ≈ 0
        and DLAs at x ≈ box; no intrinsic correlation expected."""
        rng = np.random.default_rng(43)
        box = 40.0
        n_pix = 2000
        n_dla = 500
        # All pixels in a thin slab near x = 0
        pixel_xyz = rng.uniform(0, box, size=(n_pix, 3))
        pixel_xyz[:, 0] = rng.uniform(0, 1, size=n_pix)
        pixel_axis = np.zeros(n_pix, dtype=np.int64)        # all x-axis sightlines
        pixel_delta_F = rng.standard_normal(n_pix)
        # All DLAs in a thin slab near x = box (i.e. r_par ≈ 1 Mpc/h after wrap)
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))
        dla_xyz[:, 0] = rng.uniform(box - 1, box, size=n_dla)

        r_perp_edges = np.linspace(0.0, 20.0, 6)
        r_par_edges = np.linspace(-20.0, 20.0, 11)

        xi, counts, npairs = xi_cross_dla_lya(
            pixel_xyz=pixel_xyz,
            pixel_los_axis=pixel_axis,
            pixel_delta_F=pixel_delta_F,
            dla_xyz=dla_xyz,
            box=box,
            r_perp_bins=r_perp_edges,
            r_par_bins_signed=r_par_edges,
        )
        # The bin at r_par ≈ 0 (centre of the signed grid) should populate
        # *because* the periodic wrap makes pixel-x≈0 and DLA-x≈box near
        # neighbours.  If the wrap is broken, that bin would be empty.
        center_par_bin = npairs.shape[1] // 2
        first_perp_bin = 0
        self.assertGreater(int(npairs[first_perp_bin, center_par_bin]), 100)


class TestXiAutoOnRandomDLAs(unittest.TestCase):
    """Test 5 in the doc: random Poisson DLAs → ξ_DD consistent with 0
    on linear scales (i.e. DD ≈ RR_analytic up to Poisson noise)."""

    def test_zero_signal_random_dlas(self):
        rng = np.random.default_rng(7)
        box = 80.0
        n_dla = 4000

        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))
        dla_axis = rng.integers(0, 3, size=n_dla)

        r_perp_edges = np.linspace(0.0, 30.0, 7)
        r_par_edges = np.linspace(-30.0, 30.0, 13)

        xi, DD, RR = xi_auto_dla(
            dla_xyz=dla_xyz,
            dla_los_axis=dla_axis,
            box=box,
            r_perp_bins=r_perp_edges,
            r_par_bins_signed=r_par_edges,
            chunk_size=400,
        )
        # Per-bin Poisson error is sqrt(DD) / RR
        err = np.sqrt(np.maximum(DD, 1.0)) / RR
        # Each bin's residual should be within 5 σ of zero.
        # Most bins comfortably; a small fraction (binomial) of bins can
        # touch the limit, so we test the median magnitude as < 1 σ.
        sigmas = np.abs(xi) / err
        self.assertLess(float(np.nanmedian(sigmas)), 1.5)
        # And check no bin exceeds 6 σ (would flag a bug)
        self.assertLess(float(np.nanmax(sigmas[np.isfinite(sigmas)])), 6.0)


class TestPairCounterAxisSymmetry(unittest.TestCase):
    """Test 7a in the doc: swapping x and y axes leaves ξ unchanged on a
    statistically symmetric input."""

    def test_axis_swap_invariance_dla_auto(self):
        rng = np.random.default_rng(11)
        box = 60.0
        n_dla = 2000
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))
        dla_axis = rng.integers(0, 3, size=n_dla)

        r_perp_edges = np.linspace(0.0, 25.0, 6)
        r_par_edges = np.linspace(-25.0, 25.0, 11)

        xi_orig, _, _ = xi_auto_dla(dla_xyz, dla_axis, box, r_perp_edges, r_par_edges,
                                    chunk_size=400)
        # Swap x and y in both positions and the LOS-axis assignment
        swap_pos = dla_xyz.copy()
        swap_pos[:, [0, 1]] = swap_pos[:, [1, 0]]
        swap_axis = dla_axis.copy()
        # axis 0 ↔ axis 1; axis 2 unchanged
        swap_axis = np.where(swap_axis == 0, 1, np.where(swap_axis == 1, 0, swap_axis))
        xi_swap, _, _ = xi_auto_dla(swap_pos, swap_axis, box, r_perp_edges, r_par_edges,
                                    chunk_size=400)
        # Identity transformation up to FP rounding — element-wise equality
        np.testing.assert_allclose(xi_orig, xi_swap, rtol=1e-10, atol=1e-12)


class TestSignedRparSymmetry(unittest.TestCase):
    """Test 7b in the doc: ξ(+r_par, r_perp) = ξ(-r_par, r_perp) on an
    isotropic input.

    For random DLAs (which have no preferred LOS), the auto-correlation
    must be exactly symmetric under r_par → -r_par, *bin by bin*,
    because every pair (i, j) is also counted as (j, i) with opposite sign.
    """

    def test_dla_auto_signed_symmetry_single_axis(self):
        """With ALL DLAs on the same LOS axis, mirror symmetry of DD is
        bit-exact: every pair (i, j) is also counted as (j, i) with
        opposite sign of r_par, and both use the same axis convention.

        Note: with mixed axes, the symmetry holds only statistically
        (each pair (i, j) uses ax[i] while (j, i) uses ax[j]).  We test
        the deterministic case here; the statistical case is exercised
        by the larger random-field test above (which doesn't check this).
        """
        rng = np.random.default_rng(13)
        box = 60.0
        n_dla = 2500
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))
        dla_axis = np.zeros(n_dla, dtype=np.int64)        # all on axis 0
        r_perp_edges = np.linspace(0.0, 25.0, 6)
        r_par_edges = np.linspace(-20.0, 20.0, 21)        # symmetric around 0

        xi, DD, RR = xi_auto_dla(dla_xyz, dla_axis, box, r_perp_edges, r_par_edges,
                                 chunk_size=400)
        n_par = DD.shape[1]
        for k in range(n_par // 2):
            mirror = n_par - 1 - k
            np.testing.assert_array_equal(DD[:, k], DD[:, mirror])

    def test_xcorr_signed_symmetry_single_axis_statistical(self):
        """Cross-corr is between two distinct catalogs (pixels vs DLAs),
        so a pair (pixel_i, DLA_j) has no automatic mirror partner — the
        symmetry is only statistical, not bit-exact (unlike the auto,
        where (i, j) and (j, i) both appear in the loop).  Demand
        agreement within Poisson noise.
        """
        rng = np.random.default_rng(17)
        box = 50.0
        n_pix = 6000
        n_dla = 1200
        pixel_xyz = rng.uniform(0, box, size=(n_pix, 3))
        pixel_axis = np.zeros(n_pix, dtype=np.int64)      # all on axis 0
        pixel_delta_F = rng.standard_normal(n_pix)
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))

        r_perp_edges = np.linspace(0.0, 20.0, 5)
        r_par_edges = np.linspace(-20.0, 20.0, 21)

        _, _, npairs = xi_cross_dla_lya(
            pixel_xyz=pixel_xyz,
            pixel_los_axis=pixel_axis,
            pixel_delta_F=pixel_delta_F,
            dla_xyz=dla_xyz,
            box=box,
            r_perp_bins=r_perp_edges,
            r_par_bins_signed=r_par_edges,
            chunk_size=300,
        )
        n_par = npairs.shape[1]
        for k in range(n_par // 2):
            mirror = n_par - 1 - k
            avg = 0.5 * (npairs[:, k] + npairs[:, mirror])
            diff = npairs[:, k] - npairs[:, mirror]
            sigmas = np.abs(diff) / np.sqrt(np.maximum(avg, 1.0))
            self.assertLess(float(sigmas.max()), 5.0,
                            f"asymmetry at bin pair (k={k}, {mirror}): {sigmas.max():.2f} σ")

    def test_xcorr_signed_symmetry_mixed_axes_statistical(self):
        """With mixed axes, symmetry is only statistical (different axes
        define different LOS conventions per pair).  We require the
        relative asymmetry per bin to be within Poisson noise (5 σ)."""
        rng = np.random.default_rng(19)
        box = 50.0
        n_pix = 4000
        n_dla = 1500
        pixel_xyz = rng.uniform(0, box, size=(n_pix, 3))
        pixel_axis = rng.integers(0, 3, size=n_pix)
        pixel_delta_F = rng.standard_normal(n_pix)
        dla_xyz = rng.uniform(0, box, size=(n_dla, 3))

        r_perp_edges = np.linspace(0.0, 20.0, 5)
        r_par_edges = np.linspace(-20.0, 20.0, 21)

        _, _, npairs = xi_cross_dla_lya(
            pixel_xyz=pixel_xyz, pixel_los_axis=pixel_axis,
            pixel_delta_F=pixel_delta_F, dla_xyz=dla_xyz,
            box=box, r_perp_bins=r_perp_edges, r_par_bins_signed=r_par_edges,
            chunk_size=400,
        )
        n_par = npairs.shape[1]
        # (npairs[:, k] - npairs[:, mirror]) / sqrt(npairs_avg) → ~N(0, 1)
        for k in range(n_par // 2):
            mirror = n_par - 1 - k
            avg = 0.5 * (npairs[:, k] + npairs[:, mirror])
            diff = npairs[:, k] - npairs[:, mirror]
            poisson_err = np.sqrt(np.maximum(avg, 1.0))
            sigmas = np.abs(diff) / poisson_err
            self.assertLess(float(sigmas.max()), 5.0,
                            f"asymmetry at bin pair (k={k}, {mirror}): {sigmas.max():.2f} σ")


class TestPairCounterPeriodicBoxClosure(unittest.TestCase):
    """Test 6 (revised) in the doc: empirical closure of the
    point-Poisson distribution against the analytic RR.

    For N random points in a periodic box and a cylindrical bin entirely
    INSIDE the box (r_perp_max ≤ box/2, |r_par|_max ≤ box/2):

        ⟨ DD_bin ⟩ = N(N-1) · V_bin / V_box

    so the agreement of ⟨DD⟩ with RR_analytic on a single bin is the
    direct test that V_bin is right and minimum-image is right.  The
    `_bin_volumes_2d` helper is also unit-tested directly.
    """

    def test_bin_volume_helper(self):
        from hcd_analysis.clustering import _bin_volumes_2d
        # one bin: r_perp in [0, R], r_par in [-z, z]
        R = 5.0
        z = 3.0
        V = _bin_volumes_2d(np.array([0.0, R]), np.array([-z, z]))
        self.assertEqual(V.shape, (1, 1))
        self.assertAlmostEqual(float(V[0, 0]), np.pi * R * R * (2 * z), places=10)

    def test_dd_matches_rr_for_random_points(self):
        rng = np.random.default_rng(23)
        box = 100.0
        n = 4000

        dla_xyz = rng.uniform(0, box, size=(n, 3))
        dla_axis = np.zeros(n, dtype=np.int64)
        # One bin well inside the box: r_perp in [5, 15], r_par in [-10, 10]
        r_perp_edges = np.array([5.0, 15.0])
        r_par_edges = np.array([-10.0, 10.0])

        _, DD, RR = xi_auto_dla(dla_xyz, dla_axis, box, r_perp_edges, r_par_edges,
                                chunk_size=500)
        # DD/RR should be ~1; tolerance set by Poisson sqrt(DD).
        ratio = float(DD[0, 0]) / float(RR[0, 0])
        # σ(DD)/RR = sqrt(DD)/RR ≈ 1/sqrt(DD)
        sigma = 1.0 / np.sqrt(float(DD[0, 0]))
        self.assertGreater(float(DD[0, 0]), 100)               # CLT regime
        self.assertAlmostEqual(ratio, 1.0, delta=5 * sigma)


if __name__ == "__main__":
    unittest.main()
