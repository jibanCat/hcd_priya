"""
Unit tests for hcd_analysis.lya_bias.

Covers:
  1. CAMB loader returns the expected (k, P) format.
  2. hMpc_to_kms_factor numerical sanity at z = 3.
  3. project_pk_3d_to_p1d matches a closed-form solution on a power-law
     P_lin_3D(k) = A · k^{-3}, β_F = 0, where I(k_par) = A / (2π k_par).
  4. find_camb_pk_for_z picks the file closest to a = 1/(1+z).
  5. compute_p1d_clean_sightlines: with no mask, the FFT-based P1D
     recovers the input from a known Gaussian field.
  6. fit_b_F end-to-end on a synthetic Gaussian δ_F field with a planted
     b_F; recovery within 10 %.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.lya_bias import (
    BFFitResult,
    compute_p1d_clean_sightlines,
    find_camb_pk_for_z,
    fit_b_F,
    hMpc_to_kms_factor,
    load_camb_pk,
    project_pk_3d_to_p1d,
)


class TestCambLoader(unittest.TestCase):

    def test_load_real_camb_file(self):
        """Smoke test on a real PRIYA CAMB output."""
        p = Path(
            "/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/"
            "ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742"
            "omegamh20.141hireionz6.88bhfeedback0.04/output/"
            "powerspectrum-0.2500.txt"
        )
        if not p.exists():
            self.skipTest("real PRIYA CAMB file unavailable")
        k, P = load_camb_pk(p)
        # Range from earlier inspection: k ∈ [5e-2, 280] h/Mpc, P > 0
        self.assertGreater(k.size, 100)
        self.assertGreater(k[0], 0)
        self.assertGreater(P.min(), 0)
        # k is strictly increasing
        self.assertTrue(np.all(np.diff(k) > 0))

    def test_load_synthetic_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "powerspectrum-0.5000.txt"
            k = np.logspace(-2, 2, 100)
            P = 1e3 * (k / 0.1) ** -3
            content = "# header\n# D1 = 1.0\n# k P N P0\n"
            content += "\n".join(f"{ki:.6e} {pi:.6e} 0 0" for ki, pi in zip(k, P))
            p.write_text(content)
            k_loaded, P_loaded = load_camb_pk(p)
            np.testing.assert_allclose(k_loaded, k, rtol=1e-6)
            np.testing.assert_allclose(P_loaded, P, rtol=1e-6)


class TestHMpcToKmsFactor(unittest.TestCase):

    def test_z3_h07_unit_check(self):
        """At z=3, h=0.7, H(z)≈312 km/s/Mpc, F ≈ 0.7·4/312 = 8.97e-3."""
        F = hMpc_to_kms_factor(z=3.0, hubble=0.7, Hz_kms_per_Mpc=312.0)
        self.assertAlmostEqual(F, 0.7 * 4.0 / 312.0, places=8)
        # k_v = k_r · F: at k = 1 h/Mpc, k_v ≈ 9e-3 s/km — consistent with
        # Lyα forest k-range.
        self.assertGreater(F, 1e-3)
        self.assertLess(F, 1e-1)

    def test_box_size_consistency(self):
        """L_box [Mpc/h] / F should give L_box [km/s] = box · H(z) / (h · (1+z))."""
        h = 0.735
        z = 3.0
        Hz = 306.46
        L_mpch = 120.0
        F = hMpc_to_kms_factor(z, h, Hz)
        L_kms = L_mpch / F
        expected = L_mpch * Hz / (h * (1.0 + z))
        self.assertAlmostEqual(L_kms, expected, places=8)
        # PRIYA HiRes: 120 Mpc/h ≈ 12 510 km/s at z=3
        self.assertAlmostEqual(L_kms, 12510, delta=15)


class TestProjectionClosedForm(unittest.TestCase):
    """For P_lin_3D(k) = A · k^{-3} and β_F = 0,

        I(k_par; 0) = (1/2π) ∫_0^∞ k_perp · A · k_3d^{-3} dk_perp
                     = (A/2π) · ∫_{k_par²}^∞ u^{-3/2} / 2 du
                     = (A/4π) · [-2 · u^{-1/2}]_{k_par²}^∞
                     = (A/2π) · k_par^{-1}.
    """

    def test_kminus3_powerlaw(self):
        # Synthesise k_3d, P_3d on a power-law
        A = 1.0
        n = -3.0
        k_3d = np.logspace(-4, 4, 4096)
        P_3d = A * k_3d ** n
        k_par = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
        I = project_pk_3d_to_p1d(
            k_par=k_par, k_3d=k_3d, P_3d=P_3d, beta_F=0.0,
            n_perp=8192, k_perp_max_factor=1e3,
        )
        expected = A / (2.0 * np.pi * k_par)
        np.testing.assert_allclose(I, expected, rtol=2e-2)

    def test_beta_modulation(self):
        """With β_F > 0, I should INCREASE relative to β_F = 0 (the
        Kaiser RSD enhancement on the line of sight).  Specifically,
        averaged over μ² ∈ [0, 1], (1 + β μ²)² > 1 for any β ≠ 0."""
        A, n = 1.0, -3.0
        k_3d = np.logspace(-4, 4, 4096)
        P_3d = A * k_3d ** n
        k_par = np.array([5e-3, 5e-2])
        I0 = project_pk_3d_to_p1d(k_par, k_3d, P_3d, beta_F=0.0, n_perp=8192,
                                  k_perp_max_factor=1e3)
        I1 = project_pk_3d_to_p1d(k_par, k_3d, P_3d, beta_F=1.5, n_perp=8192,
                                  k_perp_max_factor=1e3)
        self.assertTrue(np.all(I1 > I0))


class TestFindCambPk(unittest.TestCase):

    def test_pick_closest_a(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for a in (0.20, 0.25, 0.30, 0.35):
                (d / f"powerspectrum-{a:.4f}.txt").write_text("# stub\n# D1=1\n0.1 1.0 0 0\n")
            picked = find_camb_pk_for_z(d, z=3.0)        # target a = 0.25
            self.assertEqual(picked.name, "powerspectrum-0.2500.txt")
            picked = find_camb_pk_for_z(d, z=2.333)       # target a = 0.30
            self.assertEqual(picked.name, "powerspectrum-0.3000.txt")


class TestComputeP1dClean(unittest.TestCase):

    def test_no_mask_gaussian_field(self):
        """For a δ_F drawn from a known Gaussian field, the recovered
        FFT P1D should equal the input power per mode (within Poisson)."""
        rng = np.random.default_rng(0)
        n_skewers, n_pix = 200, 256
        dv = 10.0
        # White noise: σ = 0.5 → P1D(k) ≈ σ² · dv (in km/s units)
        sigma = 0.5
        df = rng.normal(0, sigma, size=(n_skewers, n_pix))
        mask = np.zeros_like(df, dtype=bool)
        k, P, n_clean = compute_p1d_clean_sightlines(df, dv, mask)
        self.assertEqual(n_clean, n_skewers)
        # Drop the DC bin (k=0)
        P_med = float(np.median(P[1:]))
        # Expected: P_white ≈ σ² · dv
        expected = sigma * sigma * dv
        self.assertAlmostEqual(P_med, expected, delta=0.10 * expected)

    def test_skips_masked_sightlines(self):
        df = np.zeros((3, 64))
        mask = np.zeros((3, 64), dtype=bool)
        mask[1, 30] = True             # one masked pixel on row 1
        k, P, n_clean = compute_p1d_clean_sightlines(df, 1.0, mask)
        self.assertEqual(n_clean, 2)   # rows 0 and 2 only


class TestFitBFEndToEnd(unittest.TestCase):
    """Synthesise δ_F with planted b_F, verify recovery."""

    def setUp(self):
        # Synthetic CAMB-like P_lin in (Mpc/h)^3 vs h/Mpc — power law at
        # large k for analytical tractability.
        self.tmp = tempfile.TemporaryDirectory()
        td = Path(self.tmp.name)
        k_h = np.logspace(-3, 3, 2048)
        # Use a soft-broken-power-law that resembles CDM:
        # P ∝ k for k < 0.1, transitioning to k^-3 for k > 0.1
        P = 5e3 * k_h / (1.0 + (k_h / 0.1) ** 4) ** 1.0
        self.camb_path = td / "powerspectrum-0.2500.txt"
        with open(self.camb_path, "w") as f:
            f.write("# synthetic\n# D1 = 1.0\n# k P N P0\n")
            for ki, pi in zip(k_h, P):
                f.write(f"{ki:.6e} {pi:.6e} 0 0\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_recover_planted_b_F(self):
        rng = np.random.default_rng(2026)
        # Grid for the synthetic forest
        z = 3.0
        h = 0.7
        Hz = 312.0
        beta = 1.5
        b_F_in = -0.18

        # Build template I(k_par) at the FFT k bins so we can synthesise
        # the field directly with the right power.
        n_pix = 1024
        dv = 5.0
        k_par = np.fft.rfftfreq(n_pix, d=dv)             # s/km
        k_h, P_h = load_camb_pk(self.camb_path)
        F_factor = hMpc_to_kms_factor(z, h, Hz)
        k_3d_kms = k_h * F_factor
        P_3d_kms = P_h * F_factor ** 3
        I_kpar = project_pk_3d_to_p1d(k_par, k_3d_kms, P_3d_kms, beta_F=beta)

        # P1D_target = b_F² · I(k_par)
        P_target = (b_F_in ** 2) * I_kpar

        # Synthesise n_skew Gaussian skewers with P_target.
        # Round-trip identity: rfft(irfft(ft, n=N), n=N) = ft.
        # P1D in our code: P[k] = |rfft(δ_F · dv)|² / (N · dv).
        # With δ_F = irfft(ft, n=N), |rfft(δ_F · dv)|² = dv² · |ft|², so
        # P[k] = dv · |ft|² / N.  For E[P[k]] = P_target[k] we need
        # var(ft[k]) = P_target[k] · N / dv.
        n_skew = 600
        n_modes = k_par.size
        var_mode = P_target * n_pix / dv
        sigma_mode = np.sqrt(var_mode / 2.0)              # half real, half imag
        re = rng.normal(0, 1, size=(n_skew, n_modes)) * sigma_mode[None, :]
        im = rng.normal(0, 1, size=(n_skew, n_modes)) * sigma_mode[None, :]
        # k = 0 and k = Nyquist must be real-valued
        im[:, 0] = 0
        if n_pix % 2 == 0:
            im[:, -1] = 0
        ft = re + 1j * im
        delta_F_synth = np.fft.irfft(ft, n=n_pix, axis=1)

        # Now fit b_F from this synthetic field
        mask = np.zeros_like(delta_F_synth, dtype=bool)
        result = fit_b_F(
            delta_F=delta_F_synth, pixel_mask=mask, dv_kms=dv, z=z,
            hubble=h, Hz_kms_per_Mpc=Hz,
            P_lin_camb_path=self.camb_path,
            beta_F_assume=beta,
            k_min_kms=2 * k_par[1],          # a few bins past k=0
            k_max_kms=0.02,
        )
        # Recovery: |b_F_recovered| within 10% of |b_F_in|
        rel_err = abs(result.b_F - b_F_in) / abs(b_F_in)
        msg = (f"b_F recovered = {result.b_F:.4f}, planted = {b_F_in:.4f}, "
               f"rel_err = {rel_err:.3f}")
        self.assertLess(rel_err, 0.10, msg)
        # Sign convention: b_F < 0
        self.assertLess(result.b_F, 0)
        print(f"  test_recover_planted_b_F: {msg}")


if __name__ == "__main__":
    unittest.main()
